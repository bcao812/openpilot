"""Bounded, receive-only Accord radar/vision shadow fusion.

No Params writes, IPC publisher, RadarState mutation or control admission lives
here. Geometry/U11 and the engineering noise floors are NOT calibrated. A
shadow candidate is never permission to actuate, including when two sensors
agree. CAN envelope times are not physical radar measurement timestamps.
"""
from collections import deque
from dataclasses import dataclass, field
import math
import time

import numpy as np

VERSION = 'bosch-a-shadow-1'
MAX_TRACKS = 32
MAX_EVENTS = 64
MAX_AGE_NS = 200_000_000
HISTORY_NS = 500_000_000
MODEL_MAX_AGE_NS = 200_000_000
EGO_MAX_AGE_NS = 150_000_000
RADAR_TO_CAMERA = 1.52
MIN_OBSERVATIONS = 3
MIN_SPAN_NS = 120_000_000
CORRIDOR_HALF_WIDTH = 1.8
POSITION_NOISE = .5


def read(obj, name, default=None):
  return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def finite(*values):
  return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def first(obj, name):
  values = read(obj, name, [])
  return float(values[0]) if len(values) else math.nan


def fresh(timestamp, now, max_age):
  return isinstance(timestamp, int) and 0 < timestamp <= now and now - timestamp <= max_age


@dataclass(frozen=True)
class Ego:
  time_ns: int
  speed: float
  yaw: float  # left-positive; model device-frame yaw has the opposite sign


class EgoHistory:
  def __init__(self):
    self.samples = deque(maxlen=96)

  def add(self, sample):
    if not self.samples or sample.time_ns > self.samples[-1].time_ns:
      self.samples.append(sample)

  def at(self, timestamp):
    for sample in reversed(self.samples):
      if fresh(sample.time_ns, timestamp, EGO_MAX_AGE_NS):
        return sample
    return None


@dataclass
class Track:
  identifier: int
  oem_id: int
  slot: int
  frame_index: int
  lifecycle: int
  position_seq: int
  position_ns: int
  first_ns: int
  epoch_ns: int
  mean: np.ndarray  # x, left-positive y, ground velocity x/y in current body frame
  covariance: np.ndarray
  count: int = 1
  velocity_seq: int = 0
  velocity_ns: int = 0
  u10: int = 1023
  velocity_good: bool = False
  active: bool = True
  acceleration: float = 0.0
  acceleration_good: bool = False
  acceleration_ns: int = 0
  acceleration_velocity: tuple[float, float] = (0.0, 0.0)
  path: str = 'unknown'
  innovation: float = 0.0
  # Six distinct position samples; AUX updates never add a sample.
  positions: deque = field(default_factory=lambda: deque(maxlen=6))

  @property
  def mature(self):
    return self.count >= MIN_OBSERVATIONS and self.position_ns - self.first_ns >= MIN_SPAN_NS

  def project(self, target_ns, history):
    dt = (target_ns - self.epoch_ns) / 1e9
    if abs(dt) > .5:
      return None
    ego = history.at(min(target_ns, self.epoch_ns))
    if ego is None:
      return None
    angle = -ego.yaw * dt
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    transition = np.zeros((4, 4))
    transition[:2, :2] = rotation
    transition[:2, 2:] = rotation * dt
    transition[2:, 2:] = rotation
    mean = transition @ self.mean
    mean[:2] -= rotation @ np.array([ego.speed * dt, 0.0])
    # Deliberately conservative engineering process noise, not qualified sigma.
    tau = abs(dt)
    noise = np.diag([.01 * tau + tau**3, .02 * tau + tau**3, 4 * tau, 3 * tau])
    covariance = transition @ self.covariance @ transition.T + noise
    return mean, covariance

  def advance(self, target_ns, history):
    prediction = self.project(target_ns, history)
    if prediction is None:
      return False
    self.mean, self.covariance = prediction
    self.epoch_ns = target_ns
    return True

  def measure(self, index, value, variance):
    innovation = value - self.mean[index]
    denom = self.covariance[index, index] + variance
    gain = self.covariance[:, index] / denom
    self.mean += gain * innovation
    identity = np.eye(4)
    identity[:, index] -= gain
    # Joseph form keeps roundoff from producing a negative uncertainty.
    self.covariance = identity @ self.covariance @ identity.T + np.outer(gain, gain) * variance

  def measure_acceleration(self, timestamp, history):
    """Differentiate accepted velocity posteriors once per velocity sample.

    Positions and AUX arrive separately. Differentiating only the position
    correction would miss the velocity change already accepted from AUX.
    Rotate the previous velocity into the current body frame first.
    """
    dt = (timestamp - self.acceleration_ns) / 1e9
    ego = history.at(self.acceleration_ns)
    self.acceleration_good = ego is not None and 0 < dt <= .2
    if self.acceleration_good:
      angle = -ego.yaw * dt
      previous_x = math.cos(angle) * self.acceleration_velocity[0] - math.sin(angle) * self.acceleration_velocity[1]
      value = max(-10.0, min(5.0, (float(self.mean[2]) - previous_x) / dt))
      self.acceleration += dt / (.5 + dt) * (value - self.acceleration)
    else:
      self.acceleration = 0.0
    self.acceleration_ns = timestamp
    self.acceleration_velocity = float(self.mean[2]), float(self.mean[3])


def path_relation(x, y, vx, vy, path_x, path_y, previous='unknown'):
  """Finite spatial support is required for OUT; a stopped path gives UNKNOWN.

  Candidate state is in the radar/left-positive frame. Model position is in the
  device/right-positive frame. Incoming objects are retained outside the path.
  """
  if len(path_x) != len(path_y) or not 2 <= len(path_x) <= 33:
    return 'unknown', 0.0, 0.0
  if not all(finite(a, b) for a, b in zip(path_x, path_y, strict=True)):
    return 'unknown', 0.0, 0.0
  point = (x + RADAR_TO_CAMERA, y)
  # A longitudinal stop trajectory is not a geometric free-space boundary.
  if max(path_x) - min(path_x) < 1 or point[0] > max(path_x):
    return 'unknown', 0.0, 0.0
  best = None
  for a, b, u, v in zip(path_x[:-1], path_x[1:], path_y[:-1], path_y[1:], strict=True):
    dx, dy = b - a, u - v
    length_sq = dx * dx + dy * dy
    if length_sq < .01:
      continue
    along = ((point[0] - a) * dx + (point[1] + u) * dy) / length_sq
    if not 0 <= along <= 1:
      continue
    length = math.sqrt(length_sq)
    lateral = ((point[1] + u) * dx - (point[0] - a) * dy) / length
    if best is None or abs(lateral) < abs(best[0]):
      lateral_rate = (vy * dx - vx * dy) / length
      best = lateral, lateral_rate
  if best is None:
    return 'unknown', 0.0, 0.0
  lateral, lateral_rate = best
  half_width = CORRIDOR_HALF_WIDTH + (.5 if previous == 'inPath' else 0)
  if abs(lateral) <= half_width:
    return 'inPath', lateral, 0.0
  if lateral * lateral_rate < 0 and abs(lateral_rate) > .2:
    entry_time = (abs(lateral) - CORRIDOR_HALF_WIDTH) / abs(lateral_rate)
    if 0 < entry_time <= 2:
      return 'entering', lateral, entry_time
  return 'outOfPath', lateral, 0.0


class HondaRadarFusion:
  def __init__(self):
    self.history = EgoHistory()
    self.tracks: dict[int, Track] = {}
    self.next_identifier = 1
    self.session = ''
    self.sequence = 0
    self.dropped = 0
    self.last_now_ns = 0
    self.model_ns = 0
    self.bindings = {}
    self.counts = {'positions': 0, 'duplicate': 0, 'old': 0, 'resets': 0, 'rejectedPosition': 0, 'rejectedVelocity': 0,
                       'ambiguous': 0, 'expired': 0, 'capacity': 0}

  def reset(self, clear_ego=False):
    self.tracks.clear()
    self.bindings.clear()
    self.model_ns = 0
    self.counts['resets'] += 1
    if clear_ego:
      self.history.samples.clear()

  def record_inputs(self, sm, now_ns):
    """Called before publication too, retaining causal 100 Hz ego samples."""
    if sm is None:
      return
    if not all(sm.seen[k] and sm.valid[k] for k in ('carState', 'modelV2')):
      return
    cs, model = sm['carState'], sm['modelV2']
    speed, yaw = float(cs.vEgo), -first(model.orientationRate, 'z')
    timestamp = int(sm.logMonoTime['carState'])
    if (fresh(timestamp, now_ns, EGO_MAX_AGE_NS) and fresh(int(sm.logMonoTime['modelV2']), now_ns, MODEL_MAX_AGE_NS)
        and fresh(int(model.timestampEof), int(sm.logMonoTime['modelV2']), MODEL_MAX_AGE_NS) and finite(speed, yaw)
        and -.5 <= speed <= 70 and abs(yaw) <= 2):
      self.history.add(Ego(timestamp, max(0.0, speed), yaw))

  def _birth(self, observation, ego):
    if len(self.tracks) >= MAX_TRACKS:
      oldest = min(self.tracks, key=lambda key: self.tracks[key].position_ns)
      del self.tracks[oldest]
      self.counts['capacity'] += 1
    raw = read(observation, 'raw')
    timestamp = int(read(observation, 'positionMonoTime'))
    track = Track(self.next_identifier, int(read(raw, 'oemTrackId')), int(read(observation, 'slot')),
                  int(read(observation, 'frameIndex')), int(read(raw, 'lifecycleRaw')), int(read(observation, 'positionSeq')),
                  timestamp, timestamp, timestamp,
                  np.array([float(read(observation, 'dRel')), float(read(observation, 'yRel')), ego.speed, 0.0]),
                  np.diag([POSITION_NOISE**2, .8**2, 25.0, 9.0]))
    self.next_identifier += 1
    self.tracks[track.oem_id] = track
    return track

  def _velocity(self, track, observation, now_ns):
    raw = read(observation, 'raw', {})
    seq = int(read(observation, 'velocitySeq', 0))
    if not seq or seq <= track.velocity_seq:
      return
    track.velocity_seq = seq
    track.velocity_good = False
    stamp = int(read(observation, 'velocityMonoTime', 0))
    u11, u10 = int(read(raw, 'u11Raw', 2047)), int(read(raw, 'u10Raw', 1023))
    value = float(read(observation, 'vRel', math.nan))
    ego = self.history.at(stamp)
    if (not read(observation, 'velocityValid', False) or not 0 < u11 < 1728 or not 0 <= u10 < 256 or not finite(value)
        or not fresh(stamp, now_ns, MAX_AGE_NS) or abs(stamp - track.position_ns) > 50_000_000 or ego is None
        or stamp < track.epoch_ns or not track.advance(stamp, self.history)):
      self.counts['rejectedVelocity'] += 1
      return
    variance = (.35 + u10 / 100.0)**2
    residual = value + ego.speed - float(track.mean[2])
    limit = min(6.0, max(2.0, 3 * math.sqrt(float(track.covariance[2, 2]) + variance)))
    if track.count >= MIN_OBSERVATIONS and abs(residual) > limit:
      self.counts['rejectedVelocity'] += 1
      return
    track.measure(2, value + ego.speed, variance)
    track.measure_acceleration(stamp, self.history)
    track.velocity_ns, track.u10, track.velocity_good = stamp, u10, True

  def ingest(self, observations, now_ns):
    if len(observations) > MAX_EVENTS:
      self.reset()
      return
    # Only the bounded batch is sorted, never future route data.
    ordered = sorted(observations, key=lambda o: (int(read(o, 'positionMonoTime', 0)), int(read(o, 'positionSeq', 0))))
    for observation in ordered:
      slot = int(read(observation, 'slot', -1))
      if not 0 <= slot < 16:
        continue
      stamp = int(read(observation, 'positionMonoTime', 0))
      if not read(observation, 'positionValid', False):
        if read(observation, 'reason', '') in ('expired', 'invalidGeometry', 'conflictingFrame'):
          for track in self.tracks.values():
            # An old slot invalidation must not kill the identity's new slot.
            if track.slot == slot and track.position_ns <= stamp <= now_ns:
              track.active = False
        continue
      seq = int(read(observation, 'positionSeq', 0))
      raw = read(observation, 'raw', {})
      identifier = int(read(raw, 'oemTrackId', 0))
      index, life = int(read(observation, 'frameIndex', -1)), int(read(raw, 'lifecycleRaw', -1))
      x, y = float(read(observation, 'dRel', math.nan)), float(read(observation, 'yRel', math.nan))
      ego = self.history.at(stamp)
      if (not fresh(stamp, now_ns, MAX_AGE_NS) or not seq or not 1 <= identifier <= 63 or not 0 <= index < 16
          or not 0 <= life < 4095 or not finite(x, y) or not 0 < x < 250 or abs(y) > 150 or ego is None):
        self.counts['rejectedPosition'] += 1
        continue
      track = self.tracks.get(identifier)
      if track is not None and seq == track.position_seq:
        # Position/AUX are separately identified. AUX cannot refresh position age.
        self._velocity(track, observation, now_ns)
        continue
      if track is not None and (stamp <= track.position_ns or seq < track.position_seq):
        self.counts['old'] += 1
        continue
      if track is not None:
        delta = (index - track.frame_index) & 15
        life_delta = (life - track.lifecycle) & 4095
        if delta == 0 and life_delta == 0:
          self.counts['duplicate'] += 1
          if abs(x - track.mean[0]) > 2 or abs(y - track.mean[1]) > 1:
            track.active = False
            self.counts['ambiguous'] += 1
          continue
        if not (1 <= delta <= 8 and life_delta == 2 * delta and stamp - track.position_ns <= MAX_AGE_NS):
          track = None
      if track is None:
        track = self._birth(observation, ego)
      else:
        prediction = track.project(stamp, self.history)
        if prediction is None:
          self.counts['rejectedPosition'] += 1
          continue
        mean, covariance = prediction
        dx, dy = x - float(mean[0]), y - float(mean[1])
        x_gate = min(8.0, max(3.0, 3 * math.sqrt(float(covariance[0, 0]) + POSITION_NOISE**2)))
        y_gate = min(3.0, max(1.0, 3 * math.sqrt(float(covariance[1, 1]) + .8**2)))
        if abs(dx) > x_gate or abs(dy) > y_gate:
          self.counts['rejectedPosition'] += 1
          continue
        track.mean, track.covariance, track.epoch_ns = mean, covariance, stamp
        track.measure(0, x, POSITION_NOISE**2)
        track.measure(1, y, (.3 + .01 * x)**2)
        track.innovation = math.hypot(dx, dy)
        track.count += 1
        track.position_ns, track.position_seq = stamp, seq
        track.slot, track.frame_index, track.lifecycle = slot, index, life
        track.velocity_good = False
      track.active = True
      track.positions.append((stamp, x, y))
      self.counts['positions'] += 1
      self._velocity(track, observation, now_ns)
    expired = [key for key, track in self.tracks.items() if now_ns - track.position_ns > HISTORY_NS]
    for key in expired:
      del self.tracks[key]
      self.counts['expired'] += 1

  def _candidates(self, model, model_ns, ego, now_ns):
    path_x, path_y = list(model.position.x), list(model.position.y)
    if len(path_x) > 33 or len(path_y) > 33:
      path_x, path_y = [], []
    candidates, snapshots = [], []
    for track in self.tracks.values():
      if not fresh(track.position_ns, now_ns, MAX_AGE_NS):
        continue
      prediction = track.project(int(model.timestampEof), self.history)
      if prediction is None:
        continue
      mean, covariance = prediction
      if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        continue
      relation, offset, entry = path_relation(*[float(v) for v in mean], path_x, path_y, track.path)
      track.path = relation
      vx, vy = float(mean[2]), float(mean[3])
      motion = 'unknown'
      if track.mature and covariance[2, 2] < 4:
        motion = 'oncoming' if vx < -2 else 'stationary' if abs(vx) < 1 else 'moving'
      coasting = not track.active or now_ns - track.position_ns > 100_000_000
      velocity_good = track.velocity_good and fresh(track.velocity_ns, now_ns, MAX_AGE_NS)
      snapshots.append({'trackId': track.identifier, 'oemTrackId': track.oem_id, 'slot': track.slot, 'positionSeq': track.position_seq,
          'positionMonoTime': track.position_ns, 'velocityMonoTime': track.velocity_ns, 'dRel': float(mean[0]), 'yRel': float(mean[1]),
          'groundSpeed': vx, 'lateralSpeed': vy, 'positionVariance': float(covariance[0, 0]), 'velocityVariance': float(covariance[2, 2]),
          'observations': track.count, 'mature': track.mature, 'velocityGood': velocity_good, 'coasting': coasting,
          'pathState': relation, 'pathOffset': offset, 'entryTime': entry, 'motion': motion, 'u10Raw': track.u10})
      candidates.append((track, mean, covariance, relation, motion, coasting, velocity_good))
    result = []
    hypothesis_count = min(len(model.leadsV3), 3)
    new_model = model_ns != self.model_ns
    for index in range(hypothesis_count):
      vision = model.leadsV3[index]
      x, y, velocity, acceleration = (first(vision, k) for k in ('x', 'y', 'v', 'a'))
      model_speed = first(model.velocity, 'x')
      probability = float(vision.prob)
      values_ok = finite(x, y, velocity, acceleration, model_speed, probability) and 0 <= probability <= 1
      lead = {'modelIndex': index, 'present': values_ok and probability > .5, 'probability': probability if finite(probability) else 0.0,
                  'trackId': 0, 'state': 'visionOnly', 'reason': 'noVision' if not values_ok or probability <= .5 else 'noMatch',
                  'dRel': x - RADAR_TO_CAMERA if values_ok else 0.0, 'yRel': -y if values_ok else 0.0,
                  'vRel': velocity - model_speed if values_ok else 0.0, 'vLead': ego.speed + velocity - model_speed if values_ok else 0.0,
                  'aLead': acceleration if values_ok else 0.0, 'distanceWeight': 0.0, 'velocityWeight': 0.0,
                  'pathState': 'unknown', 'positionMonoTime': 0, 'velocityMonoTime': 0, 'sourceAge': 0.0}
      if not lead['present'] or probability < .8:
        self.bindings.pop(index, None)
        result.append(lead)
        continue
      if lead['dRel'] <= 0:
        # Preserve the existing visual coordinate conversion even inside the
        # radar/camera offset; a proximity check must not erase a visual lead.
        self.bindings.pop(index, None)
        lead['reason'] = 'nearVision'
        result.append(lead)
        continue
      sx, sy, sv = (first(vision, k) for k in ('xStd', 'yStd', 'vStd'))
      if not finite(sx, sy, sv) or min(sx, sy, sv) < 0:
        self.bindings.pop(index, None)
        lead['reason'] = 'visionUncertainty'
        result.append(lead)
        continue
      sx, sy, sv = max(.5, sx), max(.3, sy), max(.5, sv)
      options = []
      for candidate in candidates:
        track, mean, covariance, relation, motion, coasting, velocity_good = candidate
        if relation == 'outOfPath' or motion == 'oncoming':
          continue
        dx, dy = float(mean[0]) - lead['dRel'], float(mean[1]) - lead['yRel']
        gate_x = min(8.0, max(3.0, 3 * math.sqrt(sx * sx + float(covariance[0, 0]))))
        gate_y = min(1.5, max(.8, 3 * math.sqrt(sy * sy + float(covariance[1, 1]))))
        if abs(dx) <= gate_x and abs(dy) <= gate_y:
          score = (dx / gate_x)**2 + (dy / gate_y)**2
          options.append((score, candidate))
      options.sort(key=lambda value: value[0])
      if not options or (len(options) > 1 and options[1][0] - options[0][0] < .25):
        lead['reason'] = 'ambiguous' if options else 'noMatch'
        self.counts['ambiguous'] += bool(options)
        self.bindings.pop(index, None)
        result.append(lead)
        continue
      track, mean, covariance, relation, motion, coasting, velocity_good = options[0][1]
      previous = self.bindings.get(index)
      # A changed visual lead/cut-in starts with the original visual values.
      compatible = (previous is not None and previous[0] == track.identifier and 0 < model_ns - previous[2] <= MODEL_MAX_AGE_NS
                    and abs(lead['dRel'] - previous[3] - previous[4] * ((model_ns - previous[2]) / 1e9)) <= 3)
      if new_model:
        start = previous[1] if compatible else model_ns
        self.bindings[index] = (track.identifier, start, model_ns, lead['dRel'], lead['vRel'])
      else:
        start = previous[1] if previous and previous[0] == track.identifier else model_ns
      span = max(0.0, (model_ns - start) / 1e9)
      lead.update(trackId=track.identifier, state='associating', reason='confirmation', pathState=relation,
                  positionMonoTime=track.position_ns, velocityMonoTime=track.velocity_ns,
                  sourceAge=(now_ns - track.position_ns) / 1e9)
      if track.mature and span >= .1:
        blend = min(1.0, span / .3)
        wd = blend * min(.9, sx * sx / (sx * sx + max(.1, float(covariance[0, 0]))))
        if coasting:
          wd *= max(0.0, 1 - lead['sourceAge'] / .2)
        radar_relative_speed = float(mean[2]) - ego.speed
        velocity_difference = radar_relative_speed - lead['vRel']
        velocity_gate = min(4.0, max(2.0, 3 * sv))
        wv = 0.0
        if velocity_good and abs(velocity_difference) <= velocity_gate:
          quality = 1.0 / (1 + (track.u10 / 64.0)**2)
          wv = blend * quality * min(.85, sv * sv / (sv * sv + max(.1, float(covariance[2, 2]))))
          if coasting:
            wv *= max(0.0, 1 - lead['sourceAge'] / .2)
        lead['dRel'] += wd * (float(mean[0]) - lead['dRel'])
        lead['vRel'] += wv * velocity_difference
        lead['vLead'] = ego.speed + lead['vRel']
        # Small negative estimates of a visually confirmed stopped lead do not
        # erase the object. A confidently opposing track was excluded above.
        if -.5 <= lead['vLead'] < 0 and abs(velocity) < 1:
          lead['vLead'], lead['vRel'] = 0.0, -ego.speed
        if track.acceleration_good:
          lead['aLead'] += wv * (track.acceleration - lead['aLead'])
        lead.update(distanceWeight=wd, velocityWeight=wv, state='coasting' if coasting else 'fused' if wv > 0 else 'radarDistance',
                    reason='prediction' if coasting else 'matched' if wv > 0 else 'velocityUnqualified')
      result.append(lead)
    if new_model:
      self.model_ns = model_ns
    # Future hypotheses of the same vehicle share the track estimate. They do
    # not repeatedly update its filter or multiply independent confirmations.
    return snapshots, result

  def update(self, body, sm, now_ns):
    started = time.monotonic_ns()
    output = {'version': 1, 'algorithmVersion': VERSION, 'enabled': True, 'active': False, 'calibrated': False, 'controlAllowed': False,
                  'reason': 'scopeInactive', 'modelMonoTime': 0, 'modelTimestampEof': 0, 'egoMonoTime': 0,
                  'processingTimeNs': 0, 'tracks': [], 'leads': [], 'counters': self.counts.copy()}
    session, sequence = str(read(body, 'sessionId', '')), int(read(body, 'sequence', 0))
    dropped = int(read(read(body, 'counters', {}), 'dropped', 0))
    if (session != self.session or (self.sequence and sequence != self.sequence + 1) or dropped != self.dropped
        or now_ns <= self.last_now_ns):
      self.reset(clear_ego=now_ns <= self.last_now_ns)
    self.session, self.sequence, self.dropped, self.last_now_ns = session, sequence, dropped, now_ns
    scope = read(body, 'scope', {})
    enabled = read(body, 'schemaVersion', 0) == 1 and read(body, 'active', False) and all(read(scope, key, False) for key in
                  ('alphaLong', 'experimentalSetting', 'decSetting', 'experimentalActive', 'decEnabled', 'decActive'))
    if not enabled or sm is None:
      self.reset()
    elif not all(sm.seen[k] and sm.valid[k] for k in ('modelV2', 'carState')):
      self.reset()
      output['reason'] = 'inputsInvalid'
    else:
      model = sm['modelV2']
      model_ns, ego_ns = int(sm.logMonoTime['modelV2']), int(sm.logMonoTime['carState'])
      output.update(modelMonoTime=model_ns, modelTimestampEof=int(model.timestampEof), egoMonoTime=ego_ns)
      self.record_inputs(sm, now_ns)
      ego = self.history.at(now_ns)
      speed, yaw = float(sm['carState'].vEgo), first(model.orientationRate, 'z')
      if not finite(speed, yaw) or not -.5 <= speed <= 70 or abs(yaw) > 2:
        self.reset()
        output['reason'] = 'inputsInvalid'
      elif (not fresh(model_ns, now_ns, MODEL_MAX_AGE_NS) or not fresh(int(model.timestampEof), now_ns, MODEL_MAX_AGE_NS)
          or int(model.timestampEof) > model_ns):
        self.reset()
        output['reason'] = 'modelStale'
      elif not fresh(ego_ns, now_ns, EGO_MAX_AGE_NS) or ego is None:
        self.reset()
        output['reason'] = 'egoStale'
      elif len(model.leadsV3) > 3 or not finite(first(model.velocity, 'x')):
        self.reset()
        output['reason'] = 'modelShape'
      else:
        self.ingest(read(body, 'observations', []), now_ns)
        tracks, leads = self._candidates(model, model_ns, ego, now_ns)
        output.update(active=True, reason='shadowUncalibrated', tracks=tracks, leads=leads)
    output['processingTimeNs'] = time.monotonic_ns() - started
    output['counters'] = self.counts.copy()
    return output
