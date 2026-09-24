"""Conservative scalar braking envelope for an already qualified slow stop.

These fixed engineering assumptions are not measured vehicle guarantees. The
envelope grants no motion or release authority and does not modify the native
trajectory. A failed assumption or an unobservable margin ends the compact
request; an applied braking request survives observation loss until the motion
owner acknowledges stopping. Driver/off/fault authority remains outside it.
"""
from collections import OrderedDict
from dataclasses import dataclass, replace
import math
from numbers import Real
import struct

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.low_speed_stop_distance import (
  LowSpeedStopDistanceEvidence, MAX_SOURCE_GAP_NS, PARKING_START_SPEED,
)


CUT_IN_RESERVE = 3.0
RESIDUAL_DISTANCE = 2.0
DISTANCE_UNCERTAINTY = 0.5
SPEED_UNCERTAINTY = 0.1
RESPONSE_ACCEL_FLOOR = 0.2
BRAKE_BUILD_TIME = 1.2
MIN_NET_BRAKING = 1.0
FALLBACK_ACCEL = -1.5
MAX_TERMINAL_LEAD_SPEED = 0.5
# Quality domain and scale multiplier are engineering assumptions, not a
# calibrated coverage probability or an absolute vision-error bound.
MAX_POSITION_STD = 1.0
POSITION_STD_MULTIPLIER = 3.0
MODEL_CACHE_CAPACITY = 8
# Match radard.get_RadarState_from_vision; changing that producer must update
# the explicit source-mapping tests rather than tolerating a distance error.
VISION_RADAR_OFFSET = 1.52


def _finite(value):
  return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _timestamp(value):
  return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _float32(value):
  try:
    return struct.unpack('f', struct.pack('f', value))[0]
  except (OverflowError, struct.error):
    return math.nan


def _first(values):
  return values[0] if values is not None and len(values) else None


@dataclass(frozen=True)
class ParkingObservationUncertainty:
  qualified: bool = False
  reason: str = 'missing_source'
  now_ns: int | None = None
  radar_ns: int | None = None
  model_ns: int | None = None
  position_std: tuple[float | None, float | None] = (None, None)
  distance_uncertainty: float | None = None


class VisionLeadUncertainty:
  """Bounded exact-source join; latest model metadata is not a radar stamp."""

  def __init__(self):
    self._models = OrderedDict()
    self._last_model_ns = None

  def update(self, *, now_ns, model_ns, model, radar_ns, radar, valid):
    radar_model_ns = getattr(radar, 'mdMonoTime', None)
    result = ParkingObservationUncertainty(now_ns=now_ns, radar_ns=radar_ns, model_ns=radar_model_ns)
    if (not valid or not all(_timestamp(stamp) for stamp in (now_ns, model_ns, radar_ns))
        or not 0 <= now_ns - model_ns <= MAX_SOURCE_GAP_NS):
      self._models.clear()
      self._last_model_ns = None
      return replace(result, reason='invalid_model_source')
    if self._last_model_ns is not None and model_ns < self._last_model_ns:
      self._models.clear()
      self._last_model_ns = model_ns
      return replace(result, reason='model_clock_reversed')
    self._last_model_ns = model_ns
    model_speed = _first(getattr(getattr(model, 'velocity', None), 'x', ()))
    leads = getattr(model, 'leadsV3', ())
    slots = []
    for index in range(2):
      lead = leads[index] if index < len(leads) else None
      values = tuple(_first(getattr(lead, name, ())) for name in ('t', 'x', 'y', 'v', 'a', 'xStd'))
      slots.append(values + (getattr(lead, 'prob', None),))
    snapshot = (model_speed, tuple(slots))
    # An identity whose payload changed remains poisoned until expiry.
    if model_ns in self._models and self._models[model_ns] != snapshot:
      self._models[model_ns] = None
    else:
      self._models[model_ns] = snapshot
    for timestamp in list(self._models):
      if not 0 <= now_ns - timestamp <= MAX_SOURCE_GAP_NS:
        del self._models[timestamp]
    while len(self._models) > MODEL_CACHE_CAPACITY:
      self._models.popitem(last=False)
    if (not _timestamp(radar_model_ns) or not 0 < radar_model_ns <= radar_ns <= now_ns
        or now_ns - radar_model_ns > MAX_SOURCE_GAP_NS or now_ns - radar_ns > MAX_SOURCE_GAP_NS):
      return replace(result, reason='invalid_radar_model_age')
    if radar_model_ns not in self._models:
      return replace(result, reason='unmatched_model_source')
    matched = self._models[radar_model_ns]
    if matched is None:
      return replace(result, reason='mutated_model_source')
    model_speed, slots = matched
    if not _finite(model_speed):
      return replace(result, reason='invalid_model_speed')
    position_std = [None, None]
    for index, lead in enumerate((getattr(radar, 'leadOne', None), getattr(radar, 'leadTwo', None))):
      if getattr(lead, 'present', None) is False:
        continue
      if (getattr(lead, 'present', None) is not True or getattr(lead, 'radar', None) is not False
          or getattr(lead, 'radarTrackId', None) != -1):
        return replace(result, reason='unmatched_vision_slot')
      first_t, position, lateral, velocity, acceleration, std, probability = slots[index]
      if (not all(_finite(value) for value in slots[index]) or first_t != 0.0
          or not 0.98 <= probability <= 1.0):
        return replace(result, reason='invalid_model_lead')
      expected = (_float32(position - VISION_RADAR_OFFSET), _float32(-lateral),
                  _float32(velocity - model_speed), _float32(acceleration))
      consumed = tuple(getattr(lead, field, None) for field in ('dRel', 'yRel', 'vRel', 'aLeadK'))
      if not all(_finite(value) for value in expected) or consumed != expected:
        return replace(result, reason='mismatched_vision_values')
      if not 0.0 < std <= MAX_POSITION_STD:
        return replace(result, reason='position_std_outside_domain')
      position_std[index] = float(std)
    present_std = [value for value in position_std if value is not None]
    if not present_std:
      return replace(result, reason='no_present_lead')
    return replace(result, qualified=True, reason='exact_vision_source', position_std=tuple(position_std),
                   distance_uncertainty=max(DISTANCE_UNCERTAINTY, POSITION_STD_MULTIPLIER * max(present_std)))


@dataclass(frozen=True)
class ParkingApproachDecision:
  now_ns: int | None = None
  source_ns: int | None = None
  radar_ns: int | None = None
  eligible: bool = False
  limiting_lead: int | None = None
  raw_gap: float | None = None
  clearance_lower: float | None = None
  stopping_distance: float | None = None
  margin: float | None = None
  speed_cap: float | None = None
  accel_cap: float | None = None
  require_stop: bool = False
  allow_compact: bool = False
  reason: str = 'inactive'
  observation_age: float | None = None
  response_time: float | None = None
  speed_upper: float | None = None
  response_accel_upper: float | None = None
  uncertainty_qualified: bool = False
  uncertainty_reason: str = 'missing_source'
  uncertainty_model_ns: int | None = None
  position_std: tuple[float | None, float | None] = (None, None)
  distance_uncertainty: float | None = None
  cap_effective: bool = False
  accel_before: float | None = None
  accel_after: float | None = None
  stop_evidence_added: bool = False


class ParkingApproach:
  def __init__(self):
    self.clear()

  def clear(self):
    self.decision = ParkingApproachDecision()
    self.braking_since_ns = None
    self._braking_lead = None
    self._retaining_reason = None
    self._last_now_ns = None
    self._last_source_ns = None
    self._last_radar_ns = None
    self.uncertainty_cache = VisionLeadUncertainty()

  def _unavailable(self, decision, reason):
    if self.braking_since_ns is not None:
      if self._retaining_reason is None:
        self._retaining_reason = reason
      decision = replace(decision, limiting_lead=self._braking_lead, accel_cap=FALLBACK_ACCEL,
                         require_stop=True, reason='braking_retained_' + self._retaining_reason)
    else:
      decision = replace(decision, reason=reason)
    self.decision = decision
    return decision

  def update(self, *, now_ns, source_ns, radar_ns, speed, measured_accel, actuator_delay,
             radar, qualification, permitted, allow_activation, uncertainty=None, owner_stopping=False):
    decision = ParkingApproachDecision(now_ns=now_ns, source_ns=source_ns, radar_ns=radar_ns)
    if isinstance(uncertainty, ParkingObservationUncertainty):
      decision = replace(decision, uncertainty_qualified=uncertainty.qualified, uncertainty_reason=uncertainty.reason,
                         uncertainty_model_ns=uncertainty.model_ns, position_std=uncertainty.position_std,
                         distance_uncertainty=uncertainty.distance_uncertainty)
    if not permitted:
      self.clear()
      self.decision = replace(decision, reason='not_permitted')
      return self.decision
    if owner_stopping:
      # This is a fresh consumer acknowledgement supplied by the planner,
      # never the raw standstill bit. The existing owner now retains the stop.
      self.clear()
      self.decision = replace(decision, reason='owner_stopping')
      return self.decision
    if self._retaining_reason is not None:
      return self._unavailable(decision, self._retaining_reason)
    if (not all(_timestamp(value) for value in (now_ns, source_ns, radar_ns))
        or not 0 <= now_ns - source_ns <= MAX_SOURCE_GAP_NS
        or not 0 <= now_ns - radar_ns <= MAX_SOURCE_GAP_NS):
      return self._unavailable(decision, 'invalid_input_age')
    continuous = (
      (self._last_now_ns is None or 0 < now_ns - self._last_now_ns <= MAX_SOURCE_GAP_NS)
      and (self._last_source_ns is None or 0 <= source_ns - self._last_source_ns <= MAX_SOURCE_GAP_NS)
      and (self._last_radar_ns is None or 0 <= radar_ns - self._last_radar_ns <= MAX_SOURCE_GAP_NS)
    )
    self._last_now_ns, self._last_source_ns, self._last_radar_ns = now_ns, source_ns, radar_ns
    if not continuous:
      return self._unavailable(decision, 'input_discontinuity')
    if (not isinstance(uncertainty, ParkingObservationUncertainty) or not uncertainty.qualified
        or (uncertainty.now_ns, uncertainty.radar_ns) != (now_ns, radar_ns)
        or not _timestamp(uncertainty.model_ns) or not 0 <= now_ns - uncertainty.model_ns <= MAX_SOURCE_GAP_NS
        or not _finite(uncertainty.distance_uncertainty) or uncertainty.distance_uncertainty < DISTANCE_UNCERTAINTY):
      return self._unavailable(decision, 'uncertainty_unqualified')
    if (not all(_finite(value) for value in (speed, measured_accel, actuator_delay))
        or speed < 0.0 or not 0.0 <= actuator_delay <= 1.0):
      return self._unavailable(decision, 'invalid_vehicle_values')
    if not allow_activation or not 0.0 < speed <= PARKING_START_SPEED:
      return self._unavailable(decision, 'outside_rolling_approach')
    if (not isinstance(qualification, LowSpeedStopDistanceEvidence)
        or (qualification.now_ns, qualification.source_ns, qualification.radar_ns) != (now_ns, source_ns, radar_ns)
        or not qualification.low_speed_history or not any(qualification.present)):
      return self._unavailable(decision, 'unqualified_history')
    leads = (getattr(radar, 'leadOne', None), getattr(radar, 'leadTwo', None))
    if tuple(getattr(lead, 'present', None) for lead in leads) != qualification.present:
      return self._unavailable(decision, 'changed_present_slots')
    distances = []
    for index, lead in enumerate(leads):
      if not qualification.present[index]:
        continue
      values = tuple(getattr(lead, name, None) for name in ('dRel', 'vRel', 'vLead', 'aLeadK', 'modelProb', 'yRel'))
      if (not all(_finite(value) for value in values) or values != qualification.lead_observations[index]
          or not qualification.confirmed_approaches[index] or qualification.lead_fractions[index] <= 0.0
          or not 0.0 <= values[2] <= MAX_TERMINAL_LEAD_SPEED):
        return self._unavailable(decision, 'unqualified_lead')
      distances.append((float(values[0]), index))
    raw_gap, limiting_lead = min(distances)
    observation_age = max(now_ns - source_ns, now_ns - radar_ns, now_ns - uncertainty.model_ns) / 1e9
    response_time = float(actuator_delay) + BRAKE_BUILD_TIME + DT_MDL + observation_age
    speed_upper = float(speed) + SPEED_UNCERTAINTY
    response_accel_upper = max(RESPONSE_ACCEL_FLOOR, float(measured_accel))
    clearance_lower = raw_gap - CUT_IN_RESERVE - RESIDUAL_DISTANCE - uncertainty.distance_uncertainty
    response_distance = speed_upper * response_time + 0.5 * response_accel_upper * response_time ** 2
    terminal_speed = speed_upper + response_accel_upper * response_time
    stopping_distance = response_distance + terminal_speed * terminal_speed / (2.0 * MIN_NET_BRAKING)
    if not all(math.isfinite(value) for value in (response_distance, terminal_speed, stopping_distance)):
      return self._unavailable(decision, 'nonfinite_envelope')
    margin = clearance_lower - stopping_distance
    # Invert the same delay-plus-braking envelope. This is a diagnostic speed
    # budget; it is not installed as a second native cruise objective.
    radicand = (MIN_NET_BRAKING * (MIN_NET_BRAKING + response_accel_upper) * response_time ** 2
                + 2.0 * MIN_NET_BRAKING * max(0.0, clearance_lower))
    speed_cap = max(0.0, math.sqrt(radicand) - (MIN_NET_BRAKING + response_accel_upper) * response_time - SPEED_UNCERTAINTY)
    require_stop = margin < 0.0
    # While admissible, reduce speed using the distance left after the same
    # response interval. On deficient/unknown margins request the separately
    # specified fallback; never create a positive catch-up request.
    braking_space = clearance_lower - response_distance
    accel_cap = (FALLBACK_ACCEL if require_stop
                 else -min(MIN_NET_BRAKING, terminal_speed * terminal_speed / (2.0 * braking_space)))
    self.decision = replace(
      decision, eligible=True, limiting_lead=limiting_lead, raw_gap=raw_gap, clearance_lower=clearance_lower,
      stopping_distance=stopping_distance, margin=margin, speed_cap=speed_cap, accel_cap=accel_cap,
      require_stop=require_stop, allow_compact=not require_stop,
      reason='margin_deficient' if require_stop else 'margin_admissible', observation_age=observation_age,
      response_time=response_time, speed_upper=speed_upper, response_accel_upper=response_accel_upper,
    )
    return self.decision

  def record_application(self, *, accel_before, accel_after, stop_evidence_added=False):
    """Record the actual scalar arbitration; eligibility alone arms nothing."""
    decision = self.decision
    effective = (decision.accel_cap is not None and decision.accel_cap < 0.0
                 and _finite(accel_before) and _finite(accel_after)
                 and accel_after < accel_before and accel_after == min(accel_before, decision.accel_cap))
    self.decision = replace(decision, cap_effective=effective, accel_before=accel_before, accel_after=accel_after,
                            stop_evidence_added=bool(stop_evidence_added))
    if (effective or stop_evidence_added) and self.braking_since_ns is None:
      self.braking_since_ns = decision.now_ns
      self._braking_lead = decision.limiting_lead
    if decision.require_stop and self.braking_since_ns is not None and self._retaining_reason is None:
      self._retaining_reason = decision.reason
    return self.decision
