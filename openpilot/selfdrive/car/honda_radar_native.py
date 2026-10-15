"""Current native Honda radar distance/velocity pairs and their admission history.

Only AUX and the ego sample at its original timestamp determine longitudinal
motion. Distance can reject a pair; it never supplies or corrects its velocity.
The distance/velocity residual is diagnostic, not a calibrated rejection gate.
"""
from dataclasses import dataclass, replace
import math


VERSION = 1
MIN_PAIRS = 5
MIN_SPAN_NS = 250_000_000
MAX_PAIR_NS = 50_000_000
MAX_GAP_NS = 100_000_000
MAX_AGE_NS = 200_000_000


def _read(obj, name, default=None):
  return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _integer(value, lower, upper):
  return type(value) is int and lower <= value <= upper


def _finite(value):
  return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass
class NativeMotion:
  position_seq: int = 0
  position_ns: int = 0
  velocity_seq: int = 0
  velocity_ns: int = 0
  frame_index: int = 0
  generation: int = 0
  slot: int = -1
  sample_count: int = 0
  first_position_ns: int = 0
  pair_complete_ns: int = 0
  raw_relative_speed: float = 0.0
  ground_speed: float = 0.0
  velocity_variance: float = 0.0
  acceleration: float = 0.0
  acceleration_valid: bool = False
  u10: int = 0
  range_sigma: int = 0
  existence: int = 0
  pair_good: bool = False
  pending: bool = False
  rejected: bool = False
  reason: str = 'noMeasurement'
  distance_residual: float = 0.0
  distance_residual_valid: bool = False
  # Prior *accepted native pair*. These never come from the geometric filter.
  last_position_ns: int = 0
  last_velocity_ns: int = 0
  last_velocity_seq: int = 0
  last_distance: float = 0.0
  last_relative_speed: float = 0.0
  last_ground_speed: float = 0.0
  last_velocity_variance: float = 0.0
  last_yaw: float = 0.0
  filtered_acceleration: float = 0.0

  def copy(self):
    return replace(self)

  def _clear_support(self):
    self.sample_count = self.first_position_ns = 0
    self.last_position_ns = self.last_velocity_ns = self.last_velocity_seq = 0
    self.last_distance = self.last_relative_speed = self.last_ground_speed = self.last_yaw = 0.0
    self.last_velocity_variance = 0.0
    self.filtered_acceleration = 0.0

  def _clear_current_motion(self):
    self.velocity_seq = self.velocity_ns = self.pair_complete_ns = 0
    self.u10 = 0
    self.raw_relative_speed = self.ground_speed = self.velocity_variance = 0.0
    self.acceleration = 0.0
    self.acceleration_valid = self.pair_good = False
    self.distance_residual = 0.0
    self.distance_residual_valid = False

  def invalidate(self, reason):
    self._clear_support()
    self._clear_current_motion()
    self.pending, self.rejected, self.reason = False, True, reason

  def expire(self, now_ns):
    if not self.position_ns:
      return
    if now_ns < self.position_ns:
      self.invalidate('sourceOrder')
    elif self.pending and now_ns - self.position_ns > MAX_PAIR_NS:
      self.invalidate('pairTimeout')
    elif self.pair_good and (now_ns - self.position_ns > MAX_AGE_NS or now_ns - self.velocity_ns > MAX_AGE_NS):
      self.invalidate('expired')
    elif self.pair_good and now_ns - self.position_ns > MAX_GAP_NS:
      self.invalidate('coasting')

  def observe(self, observation, now_ns, ego_at, get=_read):
    """Return True only for a newly accepted real pair, never for a republication.

    Fusion invokes this only after its existing object identity/geometry checks.
    A slot migration accepted by that tracker may carry native history if its
    next scan is continuous. A same-slot generation change always starts over.
    """
    raw = get(observation, 'raw', {})
    pseq, stamp = get(observation, 'positionSeq', 0), get(observation, 'positionMonoTime', 0)
    index, generation, slot = (get(observation, name, -1) for name in ('frameIndex', 'generation', 'slot'))
    mask = get(observation, 'frameMask', 0)
    x, y = get(observation, 'dRel', math.nan), get(observation, 'yRel', math.nan)
    fields = {name: get(raw, name) for name in
              ('statusRaw', 'rangeRaw', 'azimuthRaw', 'lifecycleRaw', 'oemTrackId', 'rangeSigmaRaw', 'existenceRaw')}
    position_ok = (
      get(observation, 'positionValid', False) and _integer(mask, 0, 31) and mask & 15 == 15
      and _integer(pseq, 1, (1 << 64) - 1) and _integer(stamp, 1, now_ns)
      and _integer(index, 0, 15) and _integer(generation, 1, (1 << 64) - 1) and _integer(slot, 0, 15)
      and _integer(fields['statusRaw'], 0, 14) and _integer(fields['rangeRaw'], 0, 4094)
      and _integer(fields['azimuthRaw'], 0, 2046) and _integer(fields['lifecycleRaw'], 0, 4094)
      and _integer(fields['oemTrackId'], 1, 63) and _integer(fields['rangeSigmaRaw'], 0, 127)
      and _integer(fields['existenceRaw'], 0, 127) and fields['rangeRaw'] * .05712 - 3 > 0
      and _finite(x) and _finite(y) and 0 < x < 250 and abs(y) <= 150)
    if not position_ok:
      self.invalidate('invalidPosition')
      return False
    if now_ns - stamp > MAX_AGE_NS:
      self.invalidate('expired')
      return False

    new_position = pseq != self.position_seq or stamp != self.position_ns
    restart = ''
    if new_position:
      if self.position_ns and (stamp <= self.position_ns or pseq <= self.position_seq):
        self.invalidate('sourceOrder')
        return False
      if self.position_ns:
        if self.pending or stamp - self.position_ns > MAX_GAP_NS or ((index - self.frame_index) & 15) != 1:
          restart = 'sourceGap'
        elif generation != self.generation and slot == self.slot:
          restart = 'identityChanged'
      if restart:
        self._clear_support()
      self._clear_current_motion()
      self.position_seq, self.position_ns, self.frame_index = pseq, stamp, index
      self.generation, self.slot = generation, slot
      self.range_sigma, self.existence = fields['rangeSigmaRaw'], fields['existenceRaw']
      self.pending, self.rejected, self.reason = True, False, 'awaitingAux'
    elif generation != self.generation or slot != self.slot or index != self.frame_index:
      self.invalidate('identityChanged')
      return False

    if not mask & 16:
      # Repeated position-only publications cannot overwrite a complete pair.
      # A complete AUX may follow this position in the same drained batch.
      # Apply pending timeout after the whole batch (or at snapshot), never
      # between its position and AUX events. Source skew/age still bound AUX.
      return False
    u11, u10 = get(raw, 'u11Raw'), get(raw, 'u10Raw')
    vseq, vstamp = get(observation, 'velocitySeq', 0), get(observation, 'velocityMonoTime', 0)
    value = get(observation, 'vRel', math.nan)
    def reject_velocity(reason):
      self.invalidate(reason)
      # Keep the rejected measurement's provenance for diagnosis without
      # retaining a usable old speed or granting it any lifetime/authority.
      self.velocity_seq = vseq if _integer(vseq, 0, (1 << 64) - 1) else 0
      self.velocity_ns = vstamp if _integer(vstamp, 0, now_ns) else 0
      self.u10 = u10 if _integer(u10, 0, 1023) else 0
      self.raw_relative_speed = (u11 - 864) / 64.0 if _integer(u11, 0, 2047) else 0.0
      return False
    # Bad endpoints have velocitySeq=0 in the decoder. Check quality *before*
    # any duplicate-sequence early return, so these withdraw the current pair.
    velocity_ok = (get(observation, 'velocityValid', False) and _integer(u11, 1, 1727)
                   and _integer(u10, 0, 255) and _integer(vseq, 1, (1 << 64) - 1)
                   and _integer(vstamp, 1, now_ns) and _finite(value)
                   and abs(float(value) - (u11 - 864) / 64.0) <= 1e-9)
    if not velocity_ok:
      return reject_velocity('invalidVelocity')
    if abs(vstamp - stamp) > MAX_PAIR_NS:
      return reject_velocity('pairSkew')
    if now_ns - vstamp > MAX_AGE_NS or now_ns - stamp > MAX_GAP_NS:
      return reject_velocity('expired' if now_ns - vstamp > MAX_AGE_NS else 'coasting')
    if self.rejected:
      return False  # An invalidated scan cannot recover by republishing its AUX.
    if self.pair_good:
      if (vseq, vstamp, value, u10) != (self.velocity_seq, self.velocity_ns, self.raw_relative_speed, self.u10):
        return reject_velocity('conflictingVelocity')
      return False
    ego = ego_at(vstamp)
    if ego is None or not _finite(ego.speed) or not _finite(ego.yaw):
      return reject_velocity('egoUnavailable')
    if self.last_velocity_ns and (vstamp <= self.last_velocity_ns or vseq <= self.last_velocity_seq):
      return reject_velocity('sourceOrder')

    ground = float(value) + ego.speed
    variance = (.35 + u10 / 100.0)**2
    if self.last_velocity_ns:
      # Retain the existing bounded velocity-innovation check, but compare
      # only independent native measurements. A distance-induced posterior
      # must neither approve a jump nor reject a correcting AUX measurement.
      dt = (vstamp - self.last_velocity_ns) / 1e9
      expected = self.last_ground_speed + self.filtered_acceleration * dt
      limit = min(6.0, max(2.0, 3 * math.sqrt(self.last_velocity_variance + variance)))
      if abs(ground - expected) > limit:
        return reject_velocity('velocityInnovation')
    self.distance_residual_valid = False
    self.distance_residual = 0.0
    self.acceleration_valid = False
    self.acceleration = 0.0
    if self.last_velocity_ns:
      dt = (vstamp - self.last_velocity_ns) / 1e9
      self.acceleration_valid = 0 < dt <= .2
      if self.acceleration_valid:
        acceleration = max(-10.0, min(5.0, (ground - self.last_ground_speed) / dt))
        self.filtered_acceleration += dt / (.5 + dt) * (acceleration - self.filtered_acceleration)
        self.acceleration = self.filtered_acceleration
      # This diagnostic uses only a causal native-speed integral. Restrict it
      # to nearly unchanged body axes (<=0.005 rad over the interval). Neither
      # this engineering comparability flag nor the residual rejects a pair.
      position_dt = (stamp - self.last_position_ns) / 1e9
      if 0 < position_dt <= MAX_GAP_NS / 1e9 and max(abs(ego.yaw), abs(self.last_yaw)) * position_dt <= .005:
        slope = (float(value) - self.last_relative_speed) / dt
        prior_at_position = self.last_relative_speed + slope * (self.last_position_ns - self.last_velocity_ns) / 1e9
        current_at_position = float(value) + slope * (stamp - vstamp) / 1e9
        self.distance_residual = float(x) - self.last_distance - .5 * (prior_at_position + current_at_position) * position_dt
        self.distance_residual_valid = math.isfinite(self.distance_residual)

    self.velocity_seq, self.velocity_ns = vseq, vstamp
    self.raw_relative_speed, self.ground_speed = float(value), ground
    self.u10, self.velocity_variance = u10, variance
    self.sample_count = min(65535, self.sample_count + 1)
    if self.sample_count == 1:
      self.first_position_ns = stamp
    self.pair_complete_ns = now_ns
    self.last_position_ns, self.last_velocity_ns, self.last_velocity_seq = stamp, vstamp, vseq
    self.last_distance, self.last_relative_speed = float(x), float(value)
    self.last_ground_speed, self.last_yaw = ground, ego.yaw
    self.last_velocity_variance = variance
    self.pair_good, self.pending, self.rejected = True, False, False
    self.reason = 'qualified' if self.qualified else restart or 'insufficientSupport'
    return True

  @property
  def qualified(self):
    return bool(self.pair_good and self.sample_count >= MIN_PAIRS
                and self.position_ns - self.first_position_ns >= MIN_SPAN_NS)

  def snapshot(self, now_ns, *, active=True):
    self.expire(now_ns)
    valid = active and self.pair_good
    return {
      'version': VERSION, 'qualified': valid and self.qualified, 'reason': self.reason if active else 'inactive',
      'positionSeq': self.position_seq, 'positionMonoTime': self.position_ns,
      'velocitySeq': self.velocity_seq, 'velocityMonoTime': self.velocity_ns,
      'sampleCount': self.sample_count, 'firstPositionMonoTime': self.first_position_ns,
      'validUntilMonoTime': min(self.position_ns, self.velocity_ns) + MAX_AGE_NS if valid else 0,
      'rawRelativeSpeed': self.raw_relative_speed, 'groundSpeed': self.ground_speed,
      'velocityVariance': self.velocity_variance, 'acceleration': self.acceleration,
      'accelerationValid': valid and self.acceleration_valid, 'u10Raw': self.u10,
      'rangeSigmaRaw': self.range_sigma, 'existenceRaw': self.existence,
      'pairCompleteMonoTime': self.pair_complete_ns,
      'distanceResidual': self.distance_residual, 'distanceResidualValid': self.distance_residual_valid,
      'frameIndex': self.frame_index,
    }
