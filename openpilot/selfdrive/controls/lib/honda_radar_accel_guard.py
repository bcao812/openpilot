"""Bounded positive-acceleration authority for an already confirmed target.

This adapter does not install the uncalibrated shadow d/v/a as a lead. It uses
the observation's continuity and identity evidence, independently checks the
current camera hypothesis, and can only hold a nonnegative acceleration cap
during a short visual gap when both formal leads are absent and the confirmed
radar target is not pulling away. The normal lead/model braking and stop owner
remain in charge. Modes 0/1/2 never enable this adapter.

All geometric, motion and noise thresholds here are conservative engineering
admission bounds, not calibrated sensor accuracy or a cut-in detection claim.
"""
from collections import deque
import math
from numbers import Real


MAX_AGE_NS = 200_000_000
EGO_MAX_AGE_NS = 150_000_000
SCOPE_MAX_AGE_NS = 500_000_000
MIN_PAIRS = 3
MIN_SPAN_NS = 120_000_000
MAX_VISION_GAP_NS = 500_000_000
MIN_SPEED = 0.5
MAX_SPEED = 35.0
MIN_PROBABILITY = 0.8
RADAR_TO_CAMERA = 1.52
MAX_RESIDUAL_X = 0.75
MAX_RESIDUAL_Y = 0.5
MAX_VELOCITY_DIFFERENCE = 2.0
RELEASE_JERK = 1.0  # m/s^3; a 0 -> 2 m/s^2 ceiling expires within two seconds
SUPPORTED_FUSION_VERSION = 'bosch-a-shadow-3'


def _read(obj, name, default=None):
  return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _finite(*values):
  return all(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) for value in values)


def _integer(value):
  return isinstance(value, int) and not isinstance(value, bool)


def _fresh(stamp, now, age=MAX_AGE_NS):
  return _integer(stamp) and 0 < stamp <= now and now - stamp <= age


def _first(obj, name):
  values = _read(obj, name, ())
  return values[0] if values is not None and len(values) else math.nan


def _key(track):
  values = tuple(_read(track, field) for field in ('trackId', 'generation', 'oemTrackId'))
  return values if all(_integer(value) for value in values) and values[0] > 0 and values[1] >= 0 and 1 <= values[2] <= 63 else None


def _project(track, elapsed, speed, yaw):
  # Same left-positive radar / right-positive model-yaw convention as the
  # observer. This short projection is only an admission comparison, never a
  # new obstacle state or a commanded acceleration.
  x, y, vx, vy = (_read(track, field) for field in ('dRel', 'yRel', 'groundSpeed', 'lateralSpeed'))
  c, s = math.cos(yaw * elapsed), math.sin(yaw * elapsed)
  x, y = x + (vx - speed) * elapsed, y + vy * elapsed
  return c * x - s * y, s * x + c * y, c * vx - s * vy


class HondaRadarAccelGuard:
  """One current camera hypothesis (index 0), one radar identity, one ceiling.

  ``qualified`` and ``holding`` are separate from ``accel_ceiling``. An active
  observer's sensor fault revokes qualification and unwinds an existing cap
  at RELEASE_JERK. Lifecycle closure and core-input failure clear immediately.
  Repeated publications cannot add confirmation or extend camera source time.
  """

  def __init__(self):
    self.clear()

  def clear(self, reason='disabled'):
    self.qualified = False
    self.holding = False
    self.accel_ceiling = None
    self.reason = reason
    self.track_key = None
    self.pairs = deque(maxlen=8)
    self.last_visible_ns = 0
    self._anchor = None
    self._track = None
    self._track_epoch_ns = 0
    self._position_seq = 0
    self._position_ns = 0
    self._last_now_ns = 0
    self._model_ns = 0
    self._model_eof = 0
    self._publication = None
    self.session_id = ''
    self.sequence = self.observation_ns = self.source_model_ns = self.source_eof = self.source_position_ns = 0
    self.formal_lead_present = self.projected_ground_speed = self.ego_speed = None

  def diagnostics(self):
    track_id, generation, oem_id = self.track_key or (0, 0, 0)
    return {'qualified': self.qualified, 'holding': self.holding, 'reason': self.reason,
            'accelCeiling': self.accel_ceiling, 'trackId': track_id, 'generation': generation, 'oemId': oem_id,
            'sessionId': self.session_id, 'sequence': self.sequence, 'observationMonoTime': self.observation_ns,
            'modelMonoTime': self._model_ns, 'modelTimestampEof': self._model_eof,
            'sourceModelMonoTime': self.source_model_ns, 'sourceTimestampEof': self.source_eof,
            'positionMonoTime': self.source_position_ns, 'lastConfirmedMonoTime': self.last_visible_ns,
            'independentPairs': len(self.pairs), 'formalLeadPresent': self.formal_lead_present,
            'projectedGroundSpeed': self.projected_ground_speed, 'egoSpeed': self.ego_speed}

  def _release(self, dt, ceiling_max):
    if self.accel_ceiling is not None:
      self.accel_ceiling = min(ceiling_max, self.accel_ceiling + RELEASE_JERK * dt)
      if self.accel_ceiling >= ceiling_max:
        self.accel_ceiling = None
    return self.accel_ceiling

  def _revoke(self, reason, dt, ceiling_max):
    self.qualified = self.holding = False
    self.reason = reason
    self.track_key = self._anchor = self._track = None
    self.pairs.clear()
    self.last_visible_ns = self._track_epoch_ns = self._position_seq = self._position_ns = 0
    return self._release(dt, ceiling_max)

  def update(self, *, now_ns, enabled, observation, observation_ns, observation_valid,
             model, model_ns, speed, standstill, ceiling_max, formal_lead_present):
    if (enabled is not True or standstill is not False or not _finite(speed, ceiling_max)
        or not MIN_SPEED < speed <= MAX_SPEED or not 0 < ceiling_max <= 2.0):
      self.clear()
      return None
    if type(formal_lead_present) is not bool:
      return self.clear('formalLeadInvalid')
    if (not _integer(now_ns) or now_ns <= 0
        or self._last_now_ns and not 0 < now_ns - self._last_now_ns <= MAX_AGE_NS):
      self.clear()
      self.reason = 'consumerClock'
      return None
    dt = (now_ns - self._last_now_ns) / 1e9 if self._last_now_ns else 0.0
    self._last_now_ns = now_ns
    self.holding = False
    self.formal_lead_present, self.ego_speed = formal_lead_present, speed
    self.projected_ground_speed = None
    def revoke(reason):
      return self._revoke(reason, dt, ceiling_max)

    eof = _read(model, 'timestampEof', 0)
    if (not _fresh(model_ns, now_ns) or not _fresh(eof, now_ns) or eof > model_ns
        or model_ns < self._model_ns or eof < self._model_eof):
      return self.clear('modelTime')
    self._model_ns, self._model_eof = model_ns, eof
    self.observation_ns = observation_ns if _integer(observation_ns) else 0
    if observation_valid is not True or not _fresh(observation_ns, now_ns) or observation is None:
      return revoke('observationStale')
    fusion = _read(observation, 'fusion')
    scope = _read(observation, 'scope')
    scope_flags = ('alphaLong', 'experimentalSetting', 'decSetting', 'experimentalActive', 'decEnabled', 'decActive', 'longActive')
    if (_read(observation, 'active') is False or _read(fusion, 'enabled') is False
        or any(_read(scope, flag) is False for flag in scope_flags)):
      return self.clear('scopeInactive')
    fusion_ns, fusion_eof = _read(fusion, 'modelMonoTime', 0), _read(fusion, 'modelTimestampEof', 0)
    self.source_model_ns = fusion_ns if _integer(fusion_ns) else 0
    self.source_eof = fusion_eof if _integer(fusion_eof) else 0
    if (_read(observation, 'schemaVersion') != 1 or _read(observation, 'active') is not True
        or _read(observation, 'reason') != 'observing' or _read(fusion, 'version') != 1
        or _read(fusion, 'algorithmVersion') != SUPPORTED_FUSION_VERSION
        or _read(fusion, 'enabled') is not True or _read(fusion, 'active') is not True
        or _read(fusion, 'controlAllowed') is not False
        or not all(_read(scope, flag) is True for flag in scope_flags)
        or not all(_fresh(_read(scope, field, 0), now_ns, SCOPE_MAX_AGE_NS) and _read(scope, field) <= observation_ns
                   for field in ('selfdriveMonoTime', 'planMonoTime'))
        or not _fresh(fusion_ns, now_ns) or not _fresh(fusion_eof, now_ns)
        or not fusion_eof <= fusion_ns <= observation_ns or fusion_eof > eof
        or not _fresh(_read(fusion, 'egoMonoTime', 0), now_ns, EGO_MAX_AGE_NS)
        or _read(fusion, 'egoMonoTime') > observation_ns):
      return revoke('observationScope')

    session, sequence = _read(observation, 'sessionId'), _read(observation, 'sequence')
    counters = (_read(_read(observation, 'counters'), 'dropped', 0),
                _read(_read(fusion, 'counters'), 'resets', 0), _read(observation, 'rejectedEnvelopes', 0))
    if not isinstance(session, str) or not 1 <= len(session) <= 64 or not _integer(sequence) or sequence <= 0:
      return revoke('publicationIdentity')
    self.session_id, self.sequence = session, sequence
    previous = self._publication
    self._publication = session, sequence, observation_ns, fusion_ns, fusion_eof, counters
    if previous is not None:
      old_session, old_seq, old_ns, old_model, old_eof, old_counters = previous
      if session != old_session:
        return revoke('sessionChanged')
      if counters != old_counters:
        return revoke('observerReset')
      if (sequence < old_seq or observation_ns < old_ns or fusion_ns < old_model or fusion_eof < old_eof
          or sequence == old_seq and self._publication != previous
          or sequence > old_seq and (sequence != old_seq + 1 or observation_ns <= old_ns)):
        return revoke('publicationOrder')

    tracks, leads, visions = _read(fusion, 'tracks', ()), _read(fusion, 'leads', ()), _read(model, 'leadsV3', ())
    if not 1 <= len(visions) <= 3:
      return self.clear('modelShape')
    if not 0 <= len(tracks) <= 32 or not 0 <= len(leads) <= 3:
      return revoke('inputShape')
    vision = visions[0]
    probability = _read(vision, 'prob')
    visual = tuple(_first(vision, field) for field in ('x', 'y', 'v'))
    model_speed, yaw = _first(_read(model, 'velocity'), 'x'), _first(_read(model, 'orientationRate'), 'z')
    if not _finite(probability, *visual, model_speed, yaw) or not 0 <= probability <= 1 or abs(yaw) > 2.0:
      return self.clear('visionInvalid')
    lead = next((lead for lead in leads if _read(lead, 'modelIndex') == 0), None)
    matched_id = _read(lead, 'trackId', 0)
    selected_id = self.track_key[0] if self.track_key is not None else matched_id
    selected = [track for track in tracks if _read(track, 'trackId') == selected_id]
    if len(selected) != 1:
      return revoke('trackMissing')
    track = selected[0]
    key = _key(track)
    position_ns, position_seq = _read(track, 'positionMonoTime', 0), _read(track, 'positionSeq', 0)
    velocity_ns = _read(track, 'velocityMonoTime', 0)
    self.source_position_ns = position_ns if _integer(position_ns) else 0
    values = tuple(_read(track, field) for field in
                   ('dRel', 'yRel', 'groundSpeed', 'lateralSpeed', 'positionVariance', 'velocityVariance', 'pathOffset'))
    if (key is None or not _integer(position_seq) or position_seq <= 0 or not _finite(*values)
        or not 1.0 <= values[0] <= 100.0 or abs(values[1]) > 50.0 or not -0.5 <= values[2] <= 55.0
        or abs(values[3]) > 5.0 or not 0 <= values[4] < 4.0 or not 0 <= values[5] < 4.0 or abs(values[6]) > 1.8
        or not all(_read(track, flag) is True for flag in ('active', 'mature', 'velocityGood'))
        or _read(track, 'coasting') is not False or _read(track, 'pathState') != 'inPath'
        or _read(track, 'motion') not in ('stationary', 'moving')
        or not _fresh(position_ns, now_ns) or not _fresh(velocity_ns, now_ns)
        or max(position_ns, velocity_ns) > observation_ns or abs(position_ns - velocity_ns) > 50_000_000
        or not _integer(_read(track, 'observations')) or _read(track, 'observations') < MIN_PAIRS):
      return revoke('trackUnqualified')
    if self.track_key is not None and key != self.track_key:
      return revoke('trackChanged')
    if (position_seq < self._position_seq or position_ns < self._position_ns
        or self._position_seq and position_seq == self._position_seq and position_ns != self._position_ns
        or self._position_seq and position_seq > self._position_seq and position_ns <= self._position_ns):
      return revoke('positionOrder')
    if self._track is not None:
      elapsed = (fusion_eof - self._track_epoch_ns) / 1e9
      x, y, vx = _project(self._track, elapsed, speed, yaw)
      if (abs(values[0] - x) > 1.5 or abs(values[1] - y) > 0.75
          or abs(values[2] - vx) > MAX_VELOCITY_DIFFERENCE):
        return revoke('radarMotionChanged')
    self._track = {field: _read(track, field) for field in ('dRel', 'yRel', 'groundSpeed', 'lateralSpeed')}
    self._track_epoch_ns, self._position_seq, self._position_ns = fusion_eof, position_seq, position_ns

    x, y, vx = _project(track, (eof - fusion_eof) / 1e9, speed, yaw)
    self.projected_ground_speed = vx
    residual = x - (visual[0] - RADAR_TO_CAMERA), y + visual[1]
    velocity_difference = vx - speed - (visual[2] - model_speed)
    if probability >= MIN_PROBABILITY:
      anchor = _read(lead, 'bindingAnchorX'), _read(lead, 'bindingAnchorY')
      binding_start = _read(lead, 'bindingStartMonoTime', 0)
      if (matched_id != key[0] or _read(lead, 'candidateTrackId') != key[0] or _read(lead, 'identityState') != 'same'
          or _read(lead, 'rejectionMonoTime') != 0 or _read(lead, 'pathState') != 'inPath'
          or _read(lead, 'present') is not True or not _finite(_read(lead, 'probability'), *anchor)
          or _read(lead, 'probability') < MIN_PROBABILITY or not _integer(binding_start)
          or not 0 < binding_start <= fusion_eof - MIN_SPAN_NS
          or not _integer(_read(lead, 'bindingSamples')) or _read(lead, 'bindingSamples') < MIN_PAIRS
          or abs(residual[0] - anchor[0]) > MAX_RESIDUAL_X or abs(residual[1] - anchor[1]) > MAX_RESIDUAL_Y
          or abs(velocity_difference) > MAX_VELOCITY_DIFFERENCE):
        return revoke('visionUnconfirmed')
      self.track_key, self._anchor = key, anchor
      if (_read(lead, 'independentPair') is True
          and (not self.pairs or position_seq > self.pairs[-1][0] and fusion_eof > self.pairs[-1][1])):
        if self.pairs and fusion_eof - self.pairs[-1][1] > MAX_AGE_NS:
          self.pairs.clear()
          self.qualified = False
        self.pairs.append((position_seq, fusion_eof))
        self.last_visible_ns = fusion_eof
        self.qualified = len(self.pairs) >= MIN_PAIRS and fusion_eof - self.pairs[0][1] >= MIN_SPAN_NS
      self.reason = 'confirmed' if self.qualified else 'confirming'
      return self._release(dt, ceiling_max)

    if not self.qualified or not _fresh(self.last_visible_ns, now_ns, MAX_VISION_GAP_NS):
      return revoke('visionGapExpired')
    # Weak but present new visual objects must not inherit an old identity.
    # The sensor's explicit contradiction is also a veto, even if an older
    # camera source still describes the previous track.
    if (_read(lead, 'identityState') == 'different' or _read(lead, 'reason') == 'identityRejected'
        or _read(lead, 'candidateTrackId', 0) not in (0, key[0]) or matched_id not in (0, key[0])
        or probability > 0.5 and (abs(residual[0] - self._anchor[0]) > MAX_RESIDUAL_X
                                 or abs(residual[1] - self._anchor[1]) > MAX_RESIDUAL_Y
                                 or abs(velocity_difference) > MAX_VELOCITY_DIFFERENCE)):
      return revoke('visionChanged')
    if formal_lead_present or vx > speed:
      # Existing official obstacles still own ordinary following, including
      # a 0.5--0.8 visual confidence fluctuation. A receding radar target is
      # also no reason to suppress acceleration. Neither case supplies a new
      # independent confirmation or extends the fixed camera-gap deadline.
      self.reason = 'formalLeadPresent' if formal_lead_present else 'radarPullingAway'
      return self._release(dt, ceiling_max)
    self.holding = True
    self.accel_ceiling = 0.0
    self.reason = 'visionGap'
    return self.accel_ceiling
