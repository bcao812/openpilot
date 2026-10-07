"""Bounded, receive-only Accord radar/vision shadow fusion.

No Params writes, IPC publisher, RadarState mutation or control admission lives
here. Geometry/U11 and the engineering noise floors are NOT calibrated. A
shadow candidate is never permission to actuate, including when two sensors
agree. CAN envelope times are not physical radar measurement timestamps.
"""
from collections import deque
from dataclasses import dataclass, field, replace
import math
import time

import numpy as np
from capnp.lib.capnp import _DynamicListBuilder, _DynamicListReader, _DynamicStructBuilder, _DynamicStructReader

from opendbc.car.honda.bosch_a_radar import radar_scan_continuous

VERSION = 'bosch-a-shadow-3'
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
# Shadow engineering hypotheses, not calibrated identity or noise limits. These
# qualify changes in an established pair's residual, not its absolute bias.
RESIDUAL_X_LIMIT = .75
RESIDUAL_Y_LIMIT = .5
RESIDUAL_X_DIFFERENT = 1.5
RESIDUAL_Y_DIFFERENT = .8
RESIDUAL_X_TRUST = .35
RESIDUAL_Y_TRUST = .25
BIAS_TIME_CONSTANT = 3.0
BIAS_X_RATE = .1
BIAS_Y_RATE = .05
_IDENTITY4 = np.eye(4)
_IDENTITY4.setflags(write=False)
# Native dynamic fields normally incur a failed ordinary attribute lookup
# before this same handler runs. Static attributes retain normal lookup order.
_CEREAL_READERS = {cls: (cls.__getattr__, frozenset(dir(cls))) for cls in (_DynamicStructBuilder, _DynamicStructReader)}


def read(obj, name, default=None):
  if isinstance(obj, dict):
    return obj.get(name, default)
  native = _CEREAL_READERS.get(type(obj))
  if native is not None and type(name) is str and name not in native[1]:
    try:
      return native[0](obj, name)
    except AttributeError:
      pass
  return getattr(obj, name, default)


def finite(*values):
  return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def _finite_median(values):
  """Median of a nonempty, at-most-16 finite float history; no optional imports."""
  ordered = sorted(values)
  middle = len(ordered) // 2
  # Match NumPy's one/two-value mean, including signed zero and underflow.
  if len(ordered) % 2:
    return 0.0 + ordered[middle]
  return (0.0 + ordered[middle - 1] + ordered[middle]) / 2.0


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


@dataclass(frozen=True)
class PairResidual:
  source_ns: int
  position_seq: int
  x: float
  y: float


@dataclass
class Binding:
  """Causal, hypothesis-specific identity evidence; never a model object ID.

  Radar and vision are already projected to the same current ego frame. Their
  residual reference is a slowly varying measurement-frame bias, not a fixed
  world vector: two reference points on a jointly turning vehicle can retain
  a nonzero bias in these coordinates. A residual-separation rejection survives
  option, gate and probability gaps; a soft suspension needs a new confirmation.
  Only new position AND camera-source samples can confirm or recover a pair.
  'different' denotes algorithmic incompatibility, not physical ground truth.
  """
  track_id: int
  start_ns: int
  last_model_ns: int = 0
  last_source_ns: int = 0
  last_pair_ns: int = 0
  last_position_seq: int = 0
  last_vision_distance: float = 0.0
  last_vision_velocity: float = 0.0
  rotation: float = 0.0  # Compatibility diagnostic; bias has no world-heading state.
  anchor_x: float = 0.0
  anchor_y: float = 0.0
  anchor_ready: bool = False
  state: str = 'unresolved'
  reason: str = 'confirmation'
  rejection_ns: int = 0
  separation_latched: bool = False
  reference_invalid_reason: str = ''
  samples: deque = field(default_factory=lambda: deque(maxlen=16))
  recovery: deque = field(default_factory=lambda: deque(maxlen=16))

  def reject(self, reason, timestamp, different=False, *, separation=False):
    self.recovery.clear()
    if separation:
      if not self.separation_latched:
        self.rejection_ns, self.reason = timestamp, reason
      self.separation_latched = True
      if different or self.state != 'different':
        self.state = 'different' if different else 'unresolved'
      if different:
        self.reason = reason
    elif not self.separation_latched:
      # A missing eligible association is not evidence of another identity.
      # Its former reference cannot qualify a new episode without confirmation.
      if not self.rejection_ns:
        self.rejection_ns, self.reason = timestamp, reason
      self.state, self.anchor_ready, self.start_ns = 'unresolved', False, 0
      self.samples.clear()

  def diagnostics(self, independent=False, dx=0.0, dy=0.0, anchor_x=0.0, anchor_y=0.0):
    reason = 'referenceInvalid:' + self.reference_invalid_reason if self.reference_invalid_reason else self.reason
    return {'identityState': self.state, 'identityReason': reason, 'bindingStartMonoTime': self.start_ns,
            'bindingSamples': len(self.samples), 'recoverySamples': len(self.recovery), 'independentPair': independent,
            'residualDeltaX': dx, 'residualDeltaY': dy, 'bindingAnchorX': anchor_x, 'bindingAnchorY': anchor_y,
            'rejectionMonoTime': self.rejection_ns}

  def observe(self, model_ns, source_ns, position_seq, distance, velocity, residual_x, residual_y, history,
              *, allow_confirmation=True):
    independent = source_ns > self.last_pair_ns and position_seq > self.last_position_seq
    dt = (source_ns - self.last_source_ns) / 1e9 if self.last_source_ns else 0.0
    vision_jump = (self.last_source_ns and dt > 0 and
                   abs(distance - self.last_vision_distance - self.last_vision_velocity * dt) > 3.0)
    timing_reason = ''
    if model_ns < self.last_model_ns or source_ns < self.last_source_ns:
      timing_reason = 'sourceOrder'
    elif self.last_source_ns and dt > 0:
      ego = history.at(self.last_source_ns)
      if dt > MODEL_MAX_AGE_NS / 1e9 or ego is None:
        timing_reason = 'sourceGap'
    if timing_reason:
      self.reject(timing_reason, model_ns)
      if self.separation_latched:
        self.reference_invalid_reason = timing_reason
    if timing_reason or self.reference_invalid_reason:
      # No aligned reference means no new identity claim and no recovery from
      # that reference. An existing separation remains blocked until a new
      # track/session; a soft suspension can build a fresh confirmed episode.
      self.recovery.clear()
      self.last_source_ns = max(self.last_source_ns, source_ns)
      self.last_model_ns = max(self.last_model_ns, model_ns)
      self.last_pair_ns = max(self.last_pair_ns, source_ns)
      self.last_position_seq = max(self.last_position_seq, position_seq)
      return self.diagnostics()
    rx, ry = residual_x, residual_y
    if not self.samples:
      self.anchor_x, self.anchor_y = rx, ry
    dx, dy = rx - self.anchor_x, ry - self.anchor_y
    diagnostic_anchor_x, diagnostic_anchor_y = self.anchor_x, self.anchor_y
    separated = abs(dx) > RESIDUAL_X_LIMIT or abs(dy) > RESIDUAL_Y_LIMIT
    trusted = abs(dx) <= RESIDUAL_X_TRUST and abs(dy) <= RESIDUAL_Y_TRUST
    if separated and self.anchor_ready:
      self.reject('visionJump' if vision_jump else 'residualSeparation', model_ns,
                  abs(dx) > RESIDUAL_X_DIFFERENT or abs(dy) > RESIDUAL_Y_DIFFERENT, separation=True)
      self.recovery.clear()
    elif allow_confirmation and self.rejection_ns and self.anchor_ready:
      # The unchanged separation bounds above qualify recovery. The narrower
      # trusted region is only for learning bias, never an extra identity gate.
      # Rejected pairs cannot teach their new residual into the frozen anchor.
      if independent:
        if self.recovery and source_ns - self.recovery[-1].source_ns > MODEL_MAX_AGE_NS:
          self.recovery.clear()
        self.recovery.append(PairResidual(source_ns, position_seq, rx, ry))
        if len(self.recovery) >= MIN_OBSERVATIONS and source_ns - self.recovery[0].source_ns >= MIN_SPAN_NS:
          self.start_ns = self.recovery[0].source_ns
          self.samples = deque(self.recovery, maxlen=16)
          self.recovery.clear()
          self.rejection_ns = 0
          self.separation_latched = False
          self.state, self.reason = 'same', 'reconfirmed'
    elif allow_confirmation and independent:
      if not self.anchor_ready and (separated or self.rejection_ns):
        # One unconfirmed pair can contain a startup transient or a changing
        # visual hypothesis. It is not an established identity whose residual
        # can permanently veto the target. Rebuild the confirmation window
        # from new independent evidence; confirmed-pair rejection stays latched.
        self.samples.clear()
        self.recovery.clear()
        self.rejection_ns = 0
        self.start_ns = source_ns
        self.anchor_x, self.anchor_y = rx, ry
        self.state, self.reason = 'unresolved', 'confirmationRestart'
      was_same = self.state == 'same'
      pair_dt = (source_ns - self.last_pair_ns) / 1e9 if self.last_pair_ns else 0.0
      self.samples.append(PairResidual(source_ns, position_seq, rx, ry))
      while len(self.samples) > 1 and source_ns - self.samples[0].source_ns > HISTORY_NS:
        self.samples.popleft()
      if len(self.samples) >= MIN_OBSERVATIONS and source_ns - self.start_ns >= MIN_SPAN_NS:
        self.state, self.reason = 'same', 'historyConsistent'
        target_x = _finite_median(sample.x for sample in self.samples)
        target_y = _finite_median(sample.y for sample in self.samples)
        if not self.anchor_ready:
          self.anchor_x, self.anchor_y = target_x, target_y
          self.anchor_ready = True
        elif was_same and trusted:
          # Only already-compatible pairs move the baseline, slowly and with
          # bounded rates. Rejected/new identities cannot teach it a new bias.
          alpha = pair_dt / (BIAS_TIME_CONSTANT + pair_dt)
          self.anchor_x += max(-BIAS_X_RATE * pair_dt, min(BIAS_X_RATE * pair_dt, alpha * (target_x - self.anchor_x)))
          self.anchor_y += max(-BIAS_Y_RATE * pair_dt, min(BIAS_Y_RATE * pair_dt, alpha * (target_y - self.anchor_y)))
    if independent:
      self.last_pair_ns, self.last_position_seq = source_ns, position_seq
    if source_ns > self.last_source_ns:
      self.last_vision_distance, self.last_vision_velocity = distance, velocity
    self.last_source_ns = max(self.last_source_ns, source_ns)
    self.last_model_ns = max(self.last_model_ns, model_ns)
    return self.diagnostics(independent and allow_confirmation, dx, dy, diagnostic_anchor_x, diagnostic_anchor_y)


@dataclass
class SaturationRecovery:
  """One witnessed 4093 -> invalid 4095 -> first 4094 transition.

  The invalid scan is an identity witness only, never a position measurement.
  A position-only return remains inactive until its own AUX qualifies motion.
  """
  sentinel_ns: int
  frame_index: int
  raw_geometry: tuple[int, int, int]
  return_position: tuple | None = None


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
  generation: int = 0
  birth_reason: str = 'newObject'
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
  last_invalid_reason: str = ''
  last_invalid_ns: int = 0
  invalid_position_ns: int = 0
  # A different OEM occupying this slot withdraws its contribution, but is not
  # by itself evidence that this OEM ended. Only a qualified new-slot position
  # can resume this posterior; all other invalidations erase that permission.
  slot_reassigned_ns: int = 0
  slot_reassigned_seq: int = 0
  previous_position_seq: int = 0
  previous_position_ns: int = 0
  previous_x: float = 0.0
  previous_y: float = 0.0
  saturation_ready: bool = False
  saturation_recovery: SaturationRecovery | None = None
  # Six distinct position samples; AUX updates never add a sample.
  positions: deque = field(default_factory=lambda: deque(maxlen=6))

  @property
  def mature(self):
    return self.count >= MIN_OBSERVATIONS and self.position_ns - self.first_ns >= MIN_SPAN_NS

  def project(self, target_ns, history, projection_cache=None):
    dt = (target_ns - self.epoch_ns) / 1e9
    if abs(dt) > .5:
      return None
    ego = history.at(min(target_ns, self.epoch_ns))
    if ego is None:
      return None
    key = self.epoch_ns, target_ns
    operators = projection_cache.get(key) if projection_cache is not None else None
    if operators is None or operators[0] is not ego:
      angle = -ego.yaw * dt
      c, s = math.cos(angle), math.sin(angle)
      rotation = np.array([[c, -s], [s, c]])
      transition = np.zeros((4, 4))
      transition[:2, :2] = rotation
      transition[:2, 2:] = rotation * dt
      transition[2:, 2:] = rotation
      displacement = rotation @ np.array([ego.speed * dt, 0.0])
      # Deliberately conservative engineering process noise, not qualified sigma.
      tau = abs(dt)
      noise = np.diag([.01 * tau + tau**3, .02 * tau + tau**3, 4 * tau, 3 * tau])
      for value in (transition, displacement, noise):
        value.setflags(write=False)
      operators = ego, transition, displacement, noise
      if projection_cache is not None:
        projection_cache[key] = operators
    _, transition, displacement, noise = operators
    # Only immutable operators are shared. Each call uses the current posterior
    # and returns new arrays, including after same-time position/AUX corrections.
    mean = transition @ self.mean
    mean[:2] -= displacement
    covariance = transition @ self.covariance @ transition.T + noise
    return mean, covariance

  def advance(self, target_ns, history, projection_cache=None):
    prediction = self.project(target_ns, history, projection_cache)
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
    identity = _IDENTITY4.copy()
    identity[:, index] -= gain
    # Joseph form keeps roundoff from producing a negative uncertainty.
    self.covariance = identity @ self.covariance @ identity.T + (gain[:, None] * gain[None, :]) * variance

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


def _prepare_path(path_x, path_y):
  """Validate one publication's path and retain its original segment order."""
  if len(path_x) != len(path_y) or not 2 <= len(path_x) <= 33:
    return None
  if not all(finite(a, b) for a, b in zip(path_x, path_y, strict=True)):
    return None
  maximum = max(path_x)
  # A longitudinal stop trajectory is not a geometric free-space boundary.
  if maximum - min(path_x) < 1:
    return None
  segments = []
  for a, b, u, v in zip(path_x[:-1], path_x[1:], path_y[:-1], path_y[1:], strict=True):
    dx, dy = b - a, u - v
    length_sq = dx * dx + dy * dy
    if length_sq < .01:
      continue
    segments.append((a, u, dx, dy, length_sq))
  return maximum, tuple(segments)


def _prepared_path_relation(x, y, vx, vy, path, previous='unknown'):
  if path is None:
    return 'unknown', 0.0, 0.0
  maximum, segments = path
  point = (x + RADAR_TO_CAMERA, y)
  if point[0] > maximum:
    return 'unknown', 0.0, 0.0
  best = None
  for a, u, dx, dy, length_sq in segments:
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


def path_relation(x, y, vx, vy, path_x, path_y, previous='unknown'):
  """Finite spatial support is required for OUT; a stopped path gives UNKNOWN.

  Candidate state is in the radar/left-positive frame. Model position is in the
  device/right-positive frame. Incoming objects are retained outside the path.
  """
  return _prepared_path_relation(x, y, vx, vy, _prepare_path(path_x, path_y), previous)


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
    self.bindings: dict[tuple[int, int], Binding] = {}
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

  def _birth(self, observation, ego, reason='newObject', previous=None):
    raw = read(observation, 'raw')
    identifier = int(read(raw, 'oemTrackId'))
    if identifier not in self.tracks and len(self.tracks) >= MAX_TRACKS:
      oldest = min(self.tracks, key=lambda key: self.tracks[key].position_ns)
      del self.tracks[oldest]
      self.counts['capacity'] += 1
    timestamp = int(read(observation, 'positionMonoTime'))
    track = Track(self.next_identifier, int(read(raw, 'oemTrackId')), int(read(observation, 'slot')),
                  int(read(observation, 'frameIndex')), int(read(raw, 'lifecycleRaw')), int(read(observation, 'positionSeq')),
                  timestamp, timestamp, timestamp,
                  np.array([float(read(observation, 'dRel')), float(read(observation, 'yRel')), ego.speed, 0.0]),
                  np.diag([POSITION_NOISE**2, .8**2, 25.0, 9.0]),
                  generation=int(read(observation, 'generation', 0)), birth_reason=reason)
    if previous is not None:
      track.last_invalid_reason, track.last_invalid_ns = previous.last_invalid_reason, previous.last_invalid_ns
    self.next_identifier += 1
    self.tracks[track.oem_id] = track
    return track

  def _invalidate(self, track, reason, position_ns, now_ns):
    track.active = track.velocity_good = track.acceleration_good = False
    track.last_invalid_reason, track.last_invalid_ns = reason, now_ns
    track.invalid_position_ns = max(track.invalid_position_ns, position_ns)
    track.slot_reassigned_ns = track.slot_reassigned_seq = 0
    track.saturation_ready, track.saturation_recovery = False, None
    for (_, identifier), binding in self.bindings.items():
      if identifier == track.identifier:
        binding.reject(reason, now_ns)

  def _suspend_slot(self, track, position_ns, position_seq, now_ns):
    self._invalidate(track, 'slotReassigned', position_ns, now_ns)
    track.slot_reassigned_ns, track.slot_reassigned_seq = position_ns, position_seq

  def _suspend_bindings(self, index, reason, timestamp, except_track=0):
    for (model_index, identifier), binding in self.bindings.items():
      if model_index == index and identifier != except_track:
        binding.reject(reason, timestamp)

  @staticmethod
  def _saturation_competition(track, observations, now_ns):
    # Examine the bounded received batch, including invalid complete claims.
    # Event order must not make a same-scan competing slot look unambiguous.
    for observation in observations:
      stamp = int(read(observation, 'positionMonoTime', 0))
      if not track.position_ns <= stamp <= now_ns or int(read(observation, 'frameMask', 0)) & 15 != 15:
        continue
      slot = int(read(observation, 'slot', -1))
      oem = int(read(read(observation, 'raw', {}), 'oemTrackId', 0))
      if 0 <= slot < 16 and ((slot == track.slot and oem != track.oem_id) or (slot != track.slot and oem == track.oem_id)):
        return True
    return False

  def _saturation_witness(self, track, observation, observations, now_ns):
    if track.saturation_recovery is None and not (track.active and track.mature and track.saturation_ready):
      return None
    raw = read(observation, 'raw', {})
    stamp, index = int(read(observation, 'positionMonoTime', 0)), int(read(observation, 'frameIndex', -1))
    status, distance, azimuth = (int(read(raw, key, -1)) for key in ('statusRaw', 'rangeRaw', 'azimuthRaw'))
    if (read(observation, 'reason', '') != 'invalidGeometry' or read(observation, 'positionValid', False)
        or int(read(observation, 'frameMask', 0)) & 15 != 15 or int(read(raw, 'lifecycleRaw', -1)) != 4095
        or int(read(raw, 'oemTrackId', 0)) != track.oem_id or int(read(observation, 'slot', -1)) != track.slot
        or not (0 <= status < 15 and 0 <= distance < 4095 and 0 <= azimuth < 2047 and distance * .05712 - 3 > 0)
        or not fresh(stamp, now_ns, MAX_AGE_NS) or self._saturation_competition(track, observations, now_ns)):
      return None
    previous = track.saturation_recovery
    geometry = status, distance, azimuth
    if previous is not None:
      # The decoder publishes F0..F3 and AUX snapshots of the same invalid
      # scan. Repetition neither creates another witness nor extends its age.
      return previous if (stamp, index, geometry) == (previous.sentinel_ns, previous.frame_index, previous.raw_geometry) else None
    if (track.active and track.mature and track.saturation_ready and track.lifecycle == 4093 and track.velocity_good
        and fresh(track.velocity_ns, stamp, MAX_AGE_NS) and fresh(track.position_ns, stamp, MAX_AGE_NS)
        and fresh(track.position_ns, now_ns, MAX_AGE_NS)
        and track.position_ns < stamp and 0 <= index < 16 and ((index - track.frame_index) & 15) == 1):
      return SaturationRecovery(stamp, index, geometry)
    return None

  def _saturation_return(self, track, observation, observations, now_ns):
    witness = track.saturation_recovery
    if witness is None:
      return ''
    raw = read(observation, 'raw', {})
    stamp, index = int(read(observation, 'positionMonoTime', 0)), int(read(observation, 'frameIndex', -1))
    status, distance, azimuth = (int(read(raw, key, -1)) for key in ('statusRaw', 'rangeRaw', 'azimuthRaw'))
    if (track.last_invalid_reason != 'invalidGeometry' or int(read(observation, 'slot', -1)) != track.slot
        or int(read(raw, 'lifecycleRaw', -1)) != 4094 or ((index - witness.frame_index) & 15) != 1
        or not track.position_ns < witness.sentinel_ns < stamp or not fresh(track.position_ns, now_ns, MAX_AGE_NS)
        or not (0 <= status < 15 and 0 <= distance < 4095 and 0 <= azimuth < 2047 and distance * .05712 - 3 > 0)
        or int(read(observation, 'frameMask', 0)) & 15 != 15 or self._saturation_competition(track, observations, now_ns)):
      return ''
    signature = (stamp, int(read(observation, 'positionSeq', 0)), int(read(observation, 'generation', 0)), index,
                 float(read(observation, 'dRel')), float(read(observation, 'yRel')),
                 status, distance, azimuth)
    if witness.return_position is not None and signature != witness.return_position:
      self._invalidate(track, 'conflictingPosition', stamp, now_ns)
      self.counts['ambiguous'] += 1
      return ''
    witness.return_position = signature
    return 'ready' if int(read(observation, 'frameMask', 0)) & 16 else 'waiting'

  def _velocity(self, track, observation, now_ns, projection_cache=None):
    if not track.active:
      return
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
        or stamp < track.epoch_ns or not track.advance(stamp, self.history, projection_cache)):
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

  def ingest(self, observations, now_ns, projection_cache=None):
    if len(observations) > MAX_EVENTS:
      self.reset()
      return
    # Only the bounded batch is sorted, never future route data. Native input
    # is unchanged during ingestion, so reuse the three scalar sorting keys.
    # Ordinary objects retain their original accessor order and exceptions.
    if type(observations) is _DynamicListBuilder or type(observations) is _DynamicListReader:
      ordered = [((int(read(o, 'positionMonoTime', 0)), not read(o, 'positionValid', False),
                   int(read(o, 'positionSeq', 0))), o) for o in observations]
      ordered.sort(key=lambda item: item[0])
    else:
      ordered = ((None, o) for o in sorted(
        observations, key=lambda o: (int(read(o, 'positionMonoTime', 0)), not read(o, 'positionValid', False), int(read(o, 'positionSeq', 0)))))
    for metadata, observation in ordered:
      slot = int(read(observation, 'slot', -1))
      if not 0 <= slot < 16:
        continue
      stamp = metadata[0] if metadata is not None else int(read(observation, 'positionMonoTime', 0))
      invalid = metadata[1] if metadata is not None else not read(observation, 'positionValid', False)
      if invalid:
        reason = read(observation, 'reason', '')
        invalid_oem = int(read(read(observation, 'raw', {}), 'oemTrackId', 0))
        if reason == 'conflictingFrame':
          for track in self.tracks.values():
            # Retract an accepted conflicting posterior, or revoke a pending
            # migration named by a newer conflict. An unfinished/older scan
            # cannot retract a different active, accepted position.
            accepted = (track.slot == slot and track.position_ns == stamp and
                        track.position_seq == (metadata[2] if metadata is not None else int(read(observation, 'positionSeq', 0))))
            pending = track.slot_reassigned_ns and invalid_oem == track.oem_id and stamp >= track.slot_reassigned_ns
            saturation = (track.saturation_recovery is not None and (slot == track.slot or invalid_oem == track.oem_id)
                          and stamp >= track.saturation_recovery.sentinel_ns)
            if 0 < stamp <= now_ns and (accepted or pending or saturation):
              self._invalidate(track, reason, stamp, now_ns)
              self.counts['rejectedPosition'] += 1
        elif reason in ('expired', 'invalidGeometry', 'withdrawn'):
          for track in self.tracks.values():
            # An old slot invalidation must not kill the identity's new slot.
            generation = int(read(observation, 'generation', 0))
            matching_slot = track.slot == slot and (not generation or generation == track.generation)
            pending = track.slot_reassigned_ns and invalid_oem == track.oem_id and stamp >= track.slot_reassigned_ns
            saturation = (track.saturation_recovery is not None and (slot == track.slot or invalid_oem == track.oem_id)
                          and stamp >= track.saturation_recovery.sentinel_ns)
            if track.position_ns <= stamp <= now_ns and (matching_slot or pending or saturation):
              witness = self._saturation_witness(track, observation, observations, now_ns)
              self._invalidate(track, reason, stamp, now_ns)
              track.saturation_recovery = witness
        continue
      seq = metadata[2] if metadata is not None else int(read(observation, 'positionSeq', 0))
      raw = read(observation, 'raw', {})
      identifier = int(read(raw, 'oemTrackId', 0))
      index, life = int(read(observation, 'frameIndex', -1)), int(read(raw, 'lifecycleRaw', -1))
      x, y = float(read(observation, 'dRel', math.nan)), float(read(observation, 'yRel', math.nan))
      ego = self.history.at(stamp)
      geometry_ok = 1 <= identifier <= 63 and 0 <= life < 4095 and finite(x, y) and 0 < x < 250 and abs(y) <= 150
      if (not fresh(stamp, now_ns, MAX_AGE_NS) or not seq or not 0 <= index < 16 or not geometry_ok or ego is None):
        if not geometry_ok:
          for previous in self.tracks.values():
            pending = previous.slot_reassigned_ns and previous.oem_id == identifier and stamp >= previous.slot_reassigned_ns
            if previous.position_ns <= stamp <= now_ns and (previous.slot == slot or pending):
              self._invalidate(previous, 'invalidGeometry', stamp, now_ns)
        self.counts['rejectedPosition'] += 1
        continue
      track = self.tracks.get(identifier)
      if track is not None and seq == track.position_seq:
        # Position/AUX are separately identified. AUX cannot refresh position age.
        last = track.positions[-1] if track.positions else (track.position_ns, math.nan, math.nan)
        if (slot != track.slot or stamp != track.position_ns or index != track.frame_index or life != track.lifecycle
            or abs(x - last[1]) > 1e-4 or abs(y - last[2]) > 1e-4):
          self._invalidate(track, 'conflictingPosition', stamp, now_ns)
          self.counts['ambiguous'] += 1
        else:
          self._velocity(track, observation, now_ns, projection_cache)
        continue
      # Distinct slots in one CAN envelope have the same source timestamp.
      # They can still make contradictory claims to the same OEM/frame. Check
      # that before rejecting equal-time samples; a genuinely older slot scan
      # must not retract an already-migrated, newer posterior.
      if (track is not None and slot != track.slot and index == track.frame_index and seq > track.position_seq
          and 0 <= stamp - track.position_ns <= MAX_AGE_NS):
        self.counts['duplicate'] += 1
        self.counts['ambiguous'] += 1
        self._invalidate(track, 'slotConflict', stamp, now_ns)
        continue
      if track is not None and (stamp <= track.position_ns or seq < track.position_seq):
        self.counts['old'] += 1
        continue
      for previous in self.tracks.values():
        if (previous is not track and previous.slot == slot and (previous.active or previous.saturation_recovery is not None)
            and previous.position_ns <= stamp):
          self._suspend_slot(previous, stamp, seq, now_ns)
      birth_reason = 'newObject'
      old_track = track
      saturation_recovery = False
      if track is not None:
        delta = (index - track.frame_index) & 15
        life_delta = (life - track.lifecycle) & 4095
        if delta == 0 and life_delta == 0:
          self.counts['duplicate'] += 1
          last = track.positions[-1]
          if slot != track.slot or abs(x - last[1]) > 1e-4 or abs(y - last[2]) > 1e-4:
            self._invalidate(track, 'slotConflict' if slot != track.slot else 'conflictingPosition', stamp, now_ns)
            self.counts['ambiguous'] += 1
          continue
        continuous = radar_scan_continuous(track.frame_index, track.lifecycle, track.position_ns, index, life, stamp)
        if not track.active:
          # The replacement and migrated slot can share a CAN envelope time,
          # or arrive in different publications. Neither ordering establishes
          # a new object if source sequence/lifecycle AND motion still agree.
          # The witnessed first saturation transition is the sole same-slot
          # invalidGeometry exception; it still needs position and native AUX.
          migration = (track.slot_reassigned_ns and slot != track.slot and stamp >= track.slot_reassigned_ns
                       and seq > track.slot_reassigned_seq and continuous)
          recovery = self._saturation_return(track, observation, observations, now_ns)
          if recovery == 'waiting':
            continue
          saturation_recovery = recovery == 'ready'
          if not migration and not saturation_recovery:
            if stamp <= track.invalid_position_ns:
              self.counts['old'] += 1
              continue
            birth_reason, track = 'invalidRecovery', None
        elif not continuous:
          birth_reason = 'positionGap' if stamp - track.position_ns > MAX_AGE_NS else 'lifecycleReset'
          track = None
      if track is None:
        track = self._birth(observation, ego, birth_reason, old_track)
      else:
        prediction = track.project(stamp, self.history, projection_cache)
        if prediction is None:
          self.counts['rejectedPosition'] += 1
          continue
        mean, covariance = prediction
        dx, dy = x - float(mean[0]), y - float(mean[1])
        x_gate = min(8.0, max(3.0, 3 * math.sqrt(float(covariance[0, 0]) + POSITION_NOISE**2)))
        y_gate = min(3.0, max(1.0, 3 * math.sqrt(float(covariance[1, 1]) + .8**2)))
        if abs(dx) > x_gate or abs(dy) > y_gate:
          self._invalidate(track, 'positionInnovation', stamp, now_ns)
          self.counts['rejectedPosition'] += 1
          continue
        saturation_ready = (track.active and track.mature and slot == track.slot and delta == 1
                            and track.lifecycle == 4091 and life == 4093)
        if saturation_recovery:
          # Qualify the ordinary position AND velocity corrections on private
          # state. A rejected AUX must never publish the old identity active.
          track = replace(track, positions=deque(track.positions, maxlen=6))
        track.mean, track.covariance, track.epoch_ns = mean, covariance, stamp
        track.measure(0, x, POSITION_NOISE**2)
        track.measure(1, y, (.3 + .01 * x)**2)
        track.innovation = math.hypot(dx, dy)
        track.count += 1
        track.previous_position_ns, track.previous_position_seq = track.position_ns, track.position_seq
        _, track.previous_x, track.previous_y = track.positions[-1]
        track.position_ns, track.position_seq = stamp, seq
        track.slot, track.frame_index, track.lifecycle = slot, index, life
        track.generation = int(read(observation, 'generation', 0))
        track.velocity_good = False
        track.saturation_ready = saturation_ready
      track.active = True
      track.slot_reassigned_ns = track.slot_reassigned_seq = 0
      track.saturation_recovery = None
      track.positions.append((stamp, x, y))
      self._velocity(track, observation, now_ns, projection_cache)
      if saturation_recovery:
        if not track.velocity_good:
          self._invalidate(old_track, 'invalidVelocity', stamp, now_ns)
          continue
        self.tracks[identifier] = track
      self.counts['positions'] += 1
    expired = [key for key, track in self.tracks.items() if now_ns - track.position_ns > HISTORY_NS]
    for key in expired:
      del self.tracks[key]
      self.counts['expired'] += 1
    live_identifiers = {track.identifier for track in self.tracks.values()}
    self.bindings = {key: binding for key, binding in self.bindings.items() if key[1] in live_identifiers}

  def _candidates(self, model, model_ns, ego, now_ns, projection_cache=None):
    path_x, path_y = list(model.position.x), list(model.position.y)
    if len(path_x) > 33 or len(path_y) > 33:
      path_x, path_y = [], []
    candidates, snapshots = [], []
    prepared_path, path_ready = None, False
    for track in self.tracks.values():
      if not fresh(track.position_ns, now_ns, MAX_AGE_NS):
        continue
      prediction = track.project(int(model.timestampEof), self.history, projection_cache)
      if prediction is None:
        continue
      mean, covariance = prediction
      if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        continue
      if not path_ready:
        prepared_path = _prepare_path(path_x, path_y)
        path_ready = True
      relation, offset, entry = _prepared_path_relation(*[float(v) for v in mean], prepared_path, track.path)
      track.path = relation
      vx, vy = float(mean[2]), float(mean[3])
      motion = 'unknown'
      if track.mature and covariance[2, 2] < 4:
        motion = 'oncoming' if vx < -2 else 'stationary' if abs(vx) < 1 else 'moving'
      coasting = not track.active or now_ns - track.position_ns > 100_000_000
      velocity_good = track.active and track.velocity_good and fresh(track.velocity_ns, now_ns, MAX_AGE_NS)
      lateral_position_variance, lateral_velocity_variance = float(covariance[1, 1]), float(covariance[3, 3])
      # Ingest appends only independent, accepted position sequences. AUX may
      # refine longitudinal velocity, but cannot supply lateral history or
      # refresh its last position time. These are engineering covariances,
      # not calibrated accuracy; the consumer owns its prediction budget.
      position_times = tuple(sample[0] for sample in track.positions)
      lateral_velocity_valid = bool(
        track.active and track.mature and len(position_times) >= MIN_OBSERVATIONS
        and 0 < position_times[0] and position_times[-1] == track.position_ns
        and position_times[-1] - position_times[0] >= MIN_SPAN_NS
        and all(a < b for a, b in zip(position_times[:-1], position_times[1:], strict=True))
        and finite(lateral_position_variance, lateral_velocity_variance)
        and min(lateral_position_variance, lateral_velocity_variance) >= 0)
      snapshots.append({'trackId': track.identifier, 'oemTrackId': track.oem_id, 'slot': track.slot, 'positionSeq': track.position_seq,
          'positionMonoTime': track.position_ns, 'velocityMonoTime': track.velocity_ns, 'dRel': float(mean[0]), 'yRel': float(mean[1]),
          'groundSpeed': vx, 'lateralSpeed': vy, 'positionVariance': float(covariance[0, 0]), 'velocityVariance': float(covariance[2, 2]),
          'observations': track.count, 'mature': track.mature, 'velocityGood': velocity_good, 'coasting': coasting,
          'pathState': relation, 'pathOffset': offset, 'entryTime': entry, 'motion': motion, 'u10Raw': track.u10,
          'active': track.active, 'generation': track.generation, 'birthReason': track.birth_reason,
          'lastInvalidReason': track.last_invalid_reason, 'lastInvalidMonoTime': track.last_invalid_ns,
          'frameIndex': track.frame_index, 'lifecycleRaw': track.lifecycle,
          'previousPositionSeq': track.previous_position_seq, 'previousPositionMonoTime': track.previous_position_ns,
          'previousDRel': track.previous_x, 'previousYRel': track.previous_y,
          'acceleration': track.acceleration,
          'accelerationGood': velocity_good and track.acceleration_good,
          'lateralPositionVariance': lateral_position_variance, 'lateralVelocityVariance': lateral_velocity_variance,
          'lateralVelocityValid': lateral_velocity_valid})
      if track.active:
        candidates.append((track, mean, covariance, relation, motion, coasting, velocity_good))
    result = []
    hypothesis_count = min(len(model.leadsV3), 3)
    source_ns = int(model.timestampEof)
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
                  'pathState': 'unknown', 'positionMonoTime': 0, 'velocityMonoTime': 0, 'sourceAge': 0.0,
                  'identityState': 'unresolved', 'identityReason': 'noCandidate', 'candidateTrackId': 0,
                  'bindingStartMonoTime': 0, 'bindingSamples': 0, 'recoverySamples': 0, 'independentPair': False,
                  'residualX': 0.0, 'residualY': 0.0, 'residualV': 0.0, 'residualDeltaX': 0.0, 'residualDeltaY': 0.0,
                  'bindingAnchorX': 0.0, 'bindingAnchorY': 0.0, 'rejectionMonoTime': 0,
                  'gateX': 0.0, 'gateY': 0.0, 'visionXStd': 0.0, 'visionYStd': 0.0, 'visionVStd': 0.0,
                  'blend': 0.0, 'coastWeight': 0.0, 'velocityGate': 0.0}
      if not lead['present'] or probability < .8:
        self._suspend_bindings(index, 'visionUnqualified', model_ns)
        lead['identityReason'] = 'visionUnqualified'
        result.append(lead)
        continue
      if lead['dRel'] <= 0:
        # Preserve the existing visual coordinate conversion even inside the
        # radar/camera offset; a proximity check must not erase a visual lead.
        self._suspend_bindings(index, 'nearVision', model_ns)
        lead['reason'] = 'nearVision'
        lead['identityReason'] = 'nearVision'
        result.append(lead)
        continue
      sx, sy, sv = (first(vision, k) for k in ('xStd', 'yStd', 'vStd'))
      if not finite(sx, sy, sv) or min(sx, sy, sv) < 0:
        self._suspend_bindings(index, 'visionUncertainty', model_ns)
        lead['reason'] = 'visionUncertainty'
        lead['identityReason'] = 'visionUncertainty'
        result.append(lead)
        continue
      sx, sy, sv = max(.5, sx), max(.3, sy), max(.5, sv)
      lead.update(visionXStd=sx, visionYStd=sy, visionVStd=sv, velocityGate=min(4.0, max(2.0, 3 * sv)))
      options = []
      considered = []
      for candidate in candidates:
        track, mean, covariance, relation, motion, coasting, velocity_good = candidate
        if relation == 'outOfPath' or motion == 'oncoming':
          continue
        dx, dy = float(mean[0]) - lead['dRel'], float(mean[1]) - lead['yRel']
        gate_x = min(8.0, max(3.0, 3 * math.sqrt(sx * sx + float(covariance[0, 0]))))
        gate_y = min(1.5, max(.8, 3 * math.sqrt(sy * sy + float(covariance[1, 1]))))
        score = (dx / gate_x)**2 + (dy / gate_y)**2
        option = (score, candidate, dx, dy, gate_x, gate_y)
        considered.append(option)
        if abs(dx) <= gate_x and abs(dy) <= gate_y:
          options.append(option)
      options.sort(key=lambda value: value[0])
      if considered:
        _, diagnostic, dx, dy, gate_x, gate_y = min(considered, key=lambda value: value[0])
        lead.update(candidateTrackId=diagnostic[0].identifier, residualX=dx, residualY=dy,
                    residualV=float(diagnostic[1][2]) - ego.speed - lead['vRel'], gateX=gate_x, gateY=gate_y)
      if not options or (len(options) > 1 and options[1][0] - options[0][0] < .25):
        lead['reason'] = 'ambiguous' if options else 'noMatch'
        lead['identityReason'] = lead['reason']
        self.counts['ambiguous'] += bool(options)
        rejected_checks = {}
        for _, candidate, dx, dy, gate_x, gate_y in considered:
          track, mean, *_ = candidate
          binding = self.bindings.get((index, track.identifier))
          if binding is not None and binding.anchor_ready:
            # A gate miss can still expose a real cut-in. Check only for
            # contradiction before suspension: no learning or confirmation
            # from a candidate that failed the association gates.
            diagnostics = binding.observe(model_ns, source_ns, track.position_seq, lead['dRel'], lead['vRel'],
                                          dx, dy, self.history, allow_confirmation=False)
            rejected_checks[track.identifier] = (diagnostics, dx, dy, float(mean[2]) - ego.speed - lead['vRel'], gate_x, gate_y)
        self._suspend_bindings(index, lead['reason'], model_ns)
        previous = [(track, self.bindings[(index, track.identifier)]) for track in self.tracks.values()
                    if (index, track.identifier) in self.bindings]
        if previous:
          old, binding = max(previous, key=lambda value: value[1].last_model_ns)
          lead.update(binding.diagnostics())
          if old.identifier in rejected_checks:
            diagnostics, dx, dy, dv, gate_x, gate_y = rejected_checks[old.identifier]
            lead.update(candidateTrackId=old.identifier, residualX=dx, residualY=dy, residualV=dv, gateX=gate_x, gateY=gate_y)
            for name in ('residualDeltaX', 'residualDeltaY', 'bindingAnchorX', 'bindingAnchorY'):
              lead[name] = diagnostics[name]
          if not old.active:
            lead.update(candidateTrackId=old.identifier, reason='radarInvalid', identityReason=old.last_invalid_reason)
        result.append(lead)
        continue
      track, mean, covariance, relation, motion, coasting, velocity_good = options[0][1]
      _, _, dx, dy, gate_x, gate_y = options[0]
      velocity_difference = float(mean[2]) - ego.speed - lead['vRel']
      lead.update(candidateTrackId=track.identifier, residualX=dx, residualY=dy, residualV=velocity_difference,
                  gateX=gate_x, gateY=gate_y)
      binding = self.bindings.get((index, track.identifier))
      self._suspend_bindings(index, 'candidateChanged', model_ns, except_track=track.identifier)
      # An extrapolated posterior can maintain a compatible existing pair, but
      # it is not new evidence for a different visual identity.
      if binding is None and coasting:
        lead.update(reason='coastingUnbound', identityReason='coastingUnbound')
        result.append(lead)
        continue
      if binding is None:
        binding = self.bindings[(index, track.identifier)] = Binding(track.identifier, source_ns)
      lead.update(binding.observe(model_ns, source_ns, track.position_seq, lead['dRel'], lead['vRel'], dx, dy, self.history))
      span = max(0.0, (source_ns - binding.start_ns) / 1e9)
      lead.update(trackId=track.identifier, state='associating', reason='confirmation', pathState=relation,
                  positionMonoTime=track.position_ns, velocityMonoTime=track.velocity_ns,
                  sourceAge=(now_ns - track.position_ns) / 1e9)
      if binding.rejection_ns:
        lead.update(state='visionOnly', reason='identityRejected')
      elif track.mature and binding.state == 'same' and span >= .1:
        blend = min(1.0, (span - .1) / .2)
        # Start decay continuously at 100 ms; do not halve the weight in one
        # step when the first coasting publication crosses that boundary.
        coast_weight = min(1.0, max(0.0, (.2 - lead['sourceAge']) / .1)) if coasting else 1.0
        wd = blend * min(.9, sx * sx / (sx * sx + max(.1, float(covariance[0, 0]))))
        wd *= coast_weight
        velocity_gate = lead['velocityGate']
        wv = 0.0
        if velocity_good and abs(velocity_difference) <= velocity_gate:
          quality = 1.0 / (1 + (track.u10 / 64.0)**2)
          wv = blend * quality * min(.85, sv * sv / (sv * sv + max(.1, float(covariance[2, 2]))))
          wv *= coast_weight
        lead['dRel'] += wd * (float(mean[0]) - lead['dRel'])
        lead['vRel'] += wv * velocity_difference
        lead['vLead'] = ego.speed + lead['vRel']
        # Small negative estimates of a visually confirmed stopped lead do not
        # erase the object. A confidently opposing track was excluded above.
        if -.5 <= lead['vLead'] < 0 and abs(velocity) < 1:
          lead['vLead'], lead['vRel'] = 0.0, -ego.speed
        if track.acceleration_good:
          lead['aLead'] += wv * (track.acceleration - lead['aLead'])
        lead.update(distanceWeight=wd, velocityWeight=wv, blend=blend, coastWeight=coast_weight,
                    state='coasting' if coasting else 'fused' if wv > 0 else 'radarDistance',
                    reason='prediction' if coasting else 'matched' if wv > 0 else 'velocityUnqualified')
      result.append(lead)
    self.model_ns = max(self.model_ns, model_ns)
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
    # Mode 4 is a perception producer, not a consumer of the plan it will help
    # construct. Keep legacy observe/shadow/assist scope exactly as recorded.
    scope_keys = ('alphaLong', 'experimentalSetting', 'decSetting') if read(body, 'mode', 0) == 4 else (
      'alphaLong', 'experimentalSetting', 'decSetting', 'experimentalActive', 'decEnabled', 'decActive')
    enabled = read(body, 'schemaVersion', 0) == 1 and read(body, 'active', False) and all(read(scope, key, False) for key in scope_keys)
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
        # These time/ego-dependent operators live only for this publication.
        projection_cache = {}
        self.ingest(read(body, 'observations', []), now_ns, projection_cache)
        tracks, leads = self._candidates(model, model_ns, ego, now_ns, projection_cache)
        output.update(active=True, reason='shadowUncalibrated', tracks=tracks, leads=leads)
    output['processingTimeNs'] = time.monotonic_ns() - started
    output['counters'] = self.counts.copy()
    return output
