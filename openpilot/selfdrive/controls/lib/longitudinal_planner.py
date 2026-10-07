#!/usr/bin/env python3
import math
from numbers import Integral, Real
import time
import numpy as np

import openpilot.cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.longitudinal_logging import observe
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState, MPC_FAILURE_ACCEL, ROLLING_STOP_ACCEL, STOPPING_DECEL_RATE
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode, resolve_longitudinal_mode
from openpilot.selfdrive.controls.lib.lead_qualification import control_leads, qualified_lead
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_BRAKE, LEAD_DANGER_FACTOR, REFERENCE_BOUND, STOP_DISTANCE, LongitudinalMpc, LongitudinalPlanSource,
  get_T_FOLLOW, get_safe_obstacle_distance, get_stopped_equivalence_factor,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.longitudinal_observation import PlannerObservation, observe_trace
from openpilot.selfdrive.controls.lib.cruise_response import A_CRUISE_MIN, A_CRUISE_MAX_BP, J_CRUISE_VALS
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan, should_stop
from openpilot.selfdrive.controls.lib.drive_helpers import CREEP_ACCEL, CREEP_FINISH_T, CREEP_MAX_T, CREEP_SPEED, CREEP_STOP_T
from openpilot.selfdrive.controls.lib.drive_helpers import STOP_RELEASE_ACCEL, STOP_SPEED
from openpilot.selfdrive.controls.lib.drive_helpers import CLOSE_LEAD_STOP_SPEED
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5
CLOSE_LEAD_STATE_SPEED_BP = [5.0, 10.0]
CLOSE_LEAD_STATE_ACCEL_BP = [-1.0, -0.5]
CREEP_FOLLOW_MIN_GAP = 5.0
CREEP_FOLLOW_MAX_SPEED = 2.0
CREEP_FOLLOW_MIN_ACCEL = -0.3
CREEP_FOLLOW_MODEL_MARGIN = 0.1
CREEP_FOLLOW_MIN_LEAD_SPEED = 0.5
CREEP_FOLLOW_MAX_CLOSING_SPEED = 0.5
LEAD_BRAKING_DEMAND_BP = [1.5, 3.0]
LEAD_BRAKING_MIN_CHANGE_SCALE = 0.5
LEAD_STOP_MIN_CLEARANCE = 2.0

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py


def has_safe_moving_lead(radar_state, v_ego: float, actuator_delay: float, *, minimum_gap_base: float = CREEP_FOLLOW_MIN_GAP,
                         now_ns=None) -> bool:
  leads = control_leads(radar_state)
  fusion_active = getattr(radar_state, 'fusionActive', False)
  if not (math.isfinite(v_ego) and STOP_SPEED <= v_ego < CREEP_FOLLOW_MAX_SPEED
          and math.isfinite(actuator_delay) and 0.0 <= actuator_delay <= 1.0
          and not isinstance(minimum_gap_base, bool) and math.isfinite(minimum_gap_base)
          and 3.0 <= minimum_gap_base <= CREEP_FOLLOW_MIN_GAP
          and leads and (fusion_active or radar_state.leadOne.present)):
    return False
  minimum_gap = minimum_gap_base + v_ego * (actuator_delay + DT_MDL) + v_ego ** 2 / (2.0 * COMFORT_BRAKE)
  followed = leads
  if fusion_active:
    following_candidates = [lead for lead in leads if qualified_lead(lead, 'follow', 0.95, now_ns=now_ns)]
    if not following_candidates:
      return False
    # The nearest current in-path car supplies moving-follow evidence. Other
    # independent objects remain MPC risks; a distant stopped car need not
    # depart before we can continue following the nearer moving vehicle.
    followed = (min(following_candidates, key=lambda lead: lead.dRel),)
  for lead in followed:
    if not qualified_lead(lead, 'follow', 0.95, now_ns=now_ns):
      return False
    values = (lead.dRel, lead.vRel, lead.vLead, lead.aLeadK, lead.modelProb)
    if not all(math.isfinite(value) for value in values):
      return False
    if (lead.dRel < minimum_gap
        or lead.vLead < CREEP_FOLLOW_MIN_LEAD_SPEED or lead.vRel < -CREEP_FOLLOW_MAX_CLOSING_SPEED
        or lead.vLead + min(lead.aLeadK, 0.0) * CREEP_FINISH_T < STOP_SPEED
        or abs(lead.vLead - v_ego - lead.vRel) > CREEP_FOLLOW_MAX_CLOSING_SPEED):
      return False
  if fusion_active:
    for lead in leads:
      if lead is followed[0]:
        continue
      if not qualified_lead(lead, 'brake', 0.95, now_ns=now_ns):
        return False
      evidence = getattr(lead, 'evidence', None)
      horizon = CREEP_FINISH_T
      if getattr(evidence, 'version', 0) == 1:
        reference = max(evidence.estimateMonoTime, evidence.modelMonoTime, evidence.positionMonoTime,
                        evidence.velocityMonoTime) if now_ns is None else now_ns
        age = (reference - evidence.estimateMonoTime) / 1e9
        horizon = min(horizon, evidence.conflictEnd - age)
        if evidence.conflictStart - age > horizon:
          continue
      # This is only a short creep-veto geometry check. It cannot produce
      # acceleration or remove the independent obstacle from the MPC.
      projected_gap = lead.dRel + min(lead.vRel, 0.0) * horizon + 0.5 * min(lead.aLeadK, 0.0) * horizon ** 2
      if projected_gap < minimum_gap or abs(lead.vLead - v_ego - lead.vRel) > CREEP_FOLLOW_MAX_CLOSING_SPEED:
        return False
  return True


def get_lead_braking_urgency(v_ego: float, a_target: float, radar_state, actuator_delay: float, *, lead_index: int | None = None,
                            now_ns=None) -> float:
  if lead_index is not None and (type(lead_index) is not int or lead_index not in (0, 1)):
    return 0.0
  if not all(math.isfinite(value) for value in (v_ego, a_target, actuator_delay)):
    return 0.0
  if v_ego <= CLOSE_LEAD_STATE_SPEED_BP[0] or a_target >= CLOSE_LEAD_STATE_ACCEL_BP[-1] or not 0.0 <= actuator_delay <= 1.0:
    return 0.0
  braking_weight = float(np.interp(a_target, CLOSE_LEAD_STATE_ACCEL_BP, [1.0, 0.0]))
  speed_weight = float(np.interp(v_ego, CLOSE_LEAD_STATE_SPEED_BP, [0.0, 1.0]))
  urgency = 0.0
  leads = control_leads(radar_state) if lead_index is None else ((radar_state.leadOne, radar_state.leadTwo)[lead_index],)
  for lead in leads:
    values = (lead.dRel, lead.vRel, lead.vLead, lead.aLeadK, lead.modelProb)
    if not qualified_lead(lead, 'brake', 0.95, now_ns=now_ns) or not all(math.isfinite(value) for value in values):
      continue
    if (lead.dRel <= 0.0 or lead.vLead < -0.5
        or lead.vRel >= -1.0 or abs(lead.vLead - v_ego - lead.vRel) > 1.0):
      continue
    closing_speed = -lead.vRel
    available_gap = max(lead.dRel - STOP_DISTANCE - closing_speed * (actuator_delay + DT_MDL), 0.5)
    required_decel = max(-lead.aLeadK, 0.0) + closing_speed ** 2 / (2.0 * available_gap)
    demand_weight = float(np.interp(required_decel, LEAD_BRAKING_DEMAND_BP, [0.0, 1.0]))
    closing_weight = float(np.interp(closing_speed, [1.0, 3.0], [0.0, 1.0]))
    urgency = max(urgency, demand_weight * closing_weight)
  return braking_weight * speed_weight * urgency


def get_lead_approach_urgency(v_ego: float, radar_state, actuator_delay: float, *, personality, now_ns=None) -> float:
  """Reduce MPC acceleration smoothing as a confirmed closing lead enters its comfort gap.

  This weight has no target-acceleration prerequisite and grants no controller
  response permission. The existing braking-urgency contract remains separate.
  """
  if not all(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)
             for value in (v_ego, actuator_delay)):
    return 0.0
  # Use the MPC's numeric domain before squaring speed, including for a corrupt
  # but finite caller. Ordinary road speeds remain well inside this bound.
  if not CLOSE_LEAD_STATE_SPEED_BP[0] < v_ego <= REFERENCE_BOUND or not 0.0 <= actuator_delay <= 1.0:
    return 0.0
  if isinstance(personality, (bool, np.bool_)):
    return 0.0
  try:
    t_follow = get_T_FOLLOW(personality)
  except NotImplementedError:
    return 0.0
  safe_ego_distance = get_safe_obstacle_distance(v_ego, t_follow)
  speed_weight = float(np.interp(v_ego, CLOSE_LEAD_STATE_SPEED_BP, [0.0, 1.0]))
  urgency = 0.0
  for lead in control_leads(radar_state):
    values = (lead.dRel, lead.vRel, lead.vLead, lead.aLeadK, lead.modelProb)
    if (not qualified_lead(lead, 'brake', 0.95, now_ns=now_ns)
        or not all(isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value) for value in values)):
      continue
    if (lead.dRel <= 0.0 or lead.vLead < -0.5
        or lead.vRel >= -1.0 or abs(lead.vLead - v_ego - lead.vRel) > 1.0):
      continue
    closing_speed = -lead.vRel
    lead_stop_distance = get_stopped_equivalence_factor(max(lead.vLead, 0.0))
    comfort_gap = safe_ego_distance - lead_stop_distance
    danger_gap = LEAD_DANGER_FACTOR * safe_ego_distance - lead_stop_distance
    available_gap = lead.dRel - closing_speed * (actuator_delay + DT_MDL)
    gap_weight = float(np.interp(available_gap, [danger_gap, comfort_gap], [1.0, 0.0]))
    closing_weight = float(np.interp(closing_speed, [1.0, 3.0], [0.0, 1.0]))
    urgency = max(urgency, gap_weight * closing_weight)
  return speed_weight * urgency


def should_stop_for_close_lead(v_ego: float, a_target: float, radar_state, actuator_delay: float, *, a_ego: float, now_ns=None) -> bool:
  if not all(math.isfinite(value) for value in (v_ego, a_target, a_ego, actuator_delay)):
    return False
  if not (STOP_SPEED <= v_ego < CLOSE_LEAD_STOP_SPEED and ROLLING_STOP_ACCEL < a_target < CREEP_ACCEL
          and 0.0 <= actuator_delay <= 1.0):
    return False
  # A requested brake is not evidence that the vehicle has already delivered
  # it. Reserve the ramp from the weaker of requested and measured braking.
  # A non-braking measurement warrants the existing full ramp from neutral;
  # it does not establish persistent positive acceleration during that ramp.
  # This is a conservative trigger budget, not a calibrated stopping-distance
  # model or a guarantee about the remaining physical clearance.
  ramp_time = (max(a_target, min(a_ego, 0.0)) - ROLLING_STOP_ACCEL) / STOPPING_DECEL_RATE
  response_time = actuator_delay + DT_MDL + ramp_time
  stopping_budget = LEAD_STOP_MIN_CLEARANCE + v_ego * response_time + v_ego ** 2 / (-2.0 * ROLLING_STOP_ACCEL)
  for lead in control_leads(radar_state):
    values = (lead.dRel, lead.vRel, lead.vLead, lead.aLeadK, lead.yRel, lead.modelProb)
    if not qualified_lead(lead, 'stop', 0.98, now_ns=now_ns) or not all(math.isfinite(value) for value in values):
      continue
    evidence = getattr(lead, 'evidence', None)
    lateral_offset = evidence.pathOffset if getattr(evidence, 'version', 0) == 1 else lead.yRel
    if (0.0 < lead.dRel <= stopping_budget
        and abs(lead.vLead) <= STOP_SPEED and lead.vRel < -0.2 and lead.aLeadK <= 0.2
        and abs(lateral_offset) <= 1.8 and abs(lead.vLead - v_ego - lead.vRel) <= 0.5):
      return True
  return False


# Long enough that no single hill is mistaken for the mount, short enough to settle inside one
# drive. With initialized=False the filter takes its first sample whole, so this is not a warmup
# cost -- it is how hard a sustained grade has to work to drag the estimate.
PITCH_BIAS_TAU = 300.0
# The only guard against learning a hill as bias. Measured residual on the device this was written
# for is +2.2 deg, so 3 deg leaves headroom without letting a long climb be cancelled outright.
MAX_PITCH_BIAS = math.radians(3.0)
# Passive sampling is a separate contract from motion-intent readiness. The two
# inputs are produced at 100 Hz; this matches their existing ten-period alive
# window without depending on unrelated services or execution authorization.
PASSIVE_PITCH_VERSION = 1
PASSIVE_PITCH_MAX_AGE_NS = 100_000_000
PASSIVE_PITCH_INPUTS = ('carState', 'carControl')


class PitchBiasEstimator:
  """Learn the static offset in orientationNED[1] so get_coast_accel is given road grade.

  The field is a calibrated pose, but calibration does not remove a mounting/extrinsics residual:
  on the 2019 Accord the raw median runs +2.0 to +2.4 deg across four routes while
  extrinsicsCalibration reports calibrated. Fed to get_coast_accel that is a coast limit of about
  -0.50 m/s^2 on flat road where the true value is -0.30 -- a hill that is not there.

  What makes the offset separable at all is that mean road grade over a drive is ~0. Measured over
  those same four routes the GPS-altitude grade median sits within 0.001 of zero on every one, so
  the slow component of pitch IS the residual and a long low-pass recovers it.

  Replayed over the four routes this settles at +1.74 to +1.94 deg -- a spread of 0.20 deg across
  two days and four drives, which is what a mount residual should look like. The evidence that it
  recovers something real is not that the number shrank: it is that sign agreement against the
  GPS-derived grade goes from 45-60 % raw, which is a coin flip, to 71-83 % debiased.

  That premise fails on a drive which really is uphill throughout, which is what the clamp is for.
  It is deliberately not a correctness guarantee -- in hilly terrain on an unbiased device this
  subtracts that drive's mean grade. The consequence is bounded because get_coast_accel only sets
  an upper limit on cruise acceleration; it is not a feedforward on the command.

  A device with no residual learns ~0 and nothing changes, which is why this is estimated rather
  than written down as a per-platform constant.
  """

  def __init__(self, dt, min_learn_speed=MIN_ALLOW_THROTTLE_SPEED):
    # initialized=False: the first sample is taken whole rather than dragged up from zero
    self.filter = FirstOrderFilter(0.0, PITCH_BIAS_TAU, dt, initialized=False)
    self.min_learn_speed = min_learn_speed
    self.diagnostic_calls = 0
    self.diagnostic_learn_calls = 0

  @property
  def value(self):
    return float(np.clip(self.filter.x, -MAX_PITCH_BIAS, MAX_PITCH_BIAS))

  def update(self, pitch, v_ego):
    self.diagnostic_calls += 1
    # Standstill is excluded on purpose: parking on a slope is a real pitch reading and a wrong
    # bias sample, and it is exactly the reading a car dwells on.
    if v_ego > self.min_learn_speed and math.isfinite(pitch):
      self.diagnostic_learn_calls += 1
      self.filter.update(pitch)
    return self.value

def get_cruise_accel(e2e, v_cruise, v_ego, a_cruise_prev, angle_steers, CP, dt, accel_coast, allow_throttle):
  max_accel = ACCEL_MAX if e2e else get_max_accel(v_ego)

  if not e2e:
    a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
    a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
    a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))
    max_accel = min(max_accel, a_x_allowed)
    if not allow_throttle:
      clipped_accel_coast = max(accel_coast, ACCEL_MIN)
      coast_limit = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [max_accel, clipped_accel_coast])
      max_accel = min(max_accel, coast_limit)

  target_accel = np.clip(v_cruise - v_ego, A_CRUISE_MIN, max_accel)
  j_cruise = np.interp(v_ego, A_CRUISE_MAX_BP, J_CRUISE_VALS)
  target_accel = float(np.clip(target_accel, a_cruise_prev - j_cruise * dt, a_cruise_prev + j_cruise * dt))

  return target_accel


class LongitudinalPlanner(LongitudinalPlannerSP):
  def __init__(self, CP, CP_SP, init_v=0.0, init_a=0.0, dt=DT_MDL, *, trace=None, diagnostics=None):
    self.CP = CP
    self.CP_SP = CP_SP
    self.mpc = LongitudinalMpc(dt=dt)
    LongitudinalPlannerSP.__init__(self, self.CP, CP_SP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True
    self.trace = trace
    self.mpc.diagnostic_details_enabled = trace is not None
    self.observation = PlannerObservation.create(CP, CP_SP, enabled=diagnostics)
    self.observation_attempt = None
    self.observation_result = None
    self.longitudinal_mode = resolve_longitudinal_mode(CP, CP_SP)
    self.unified = self.longitudinal_mode == LongitudinalMode.UNIFIED
    self.motion_enabled = self.unified

    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.pitch_bias = PitchBiasEstimator(self.dt)
    self._passive_input_times = dict.fromkeys(PASSIVE_PITCH_INPUTS, 0)
    self._passive_now_ns = 0
    self.last_passive_reason = 'not_updated'
    self.a_cruise = init_a
    self.output_a_target = init_a
    self.output_should_stop = False
    if not self.motion_enabled:
      self.stop_latched = False
      self.creep_t = 0.0
      self.creep_v = 0.0
      self.creep_stop = False
      self.creep_total_t = 0.0

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)
    if self.unified:
      from openpilot.selfdrive.controls.lib.reference_timing import ReferenceTimingProducer
      from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import UnifiedLongitudinalPlanner
      self.reference_timing = ReferenceTimingProducer()
      self.reference_planner = UnifiedLongitudinalPlanner(self, init_a=init_a)

  def update(self, sm, *, now_ns=None):
    # Native execution samples its own monotonic clock. Replays must provide the
    # recorded consumer time explicitly; input timestamps cannot date themselves.
    if now_ns is None:
      now_ns = time.monotonic_ns()
    self.observation_result = None
    self.observation_attempt = observe(self.observation, "begin", self, sm, now_ns=now_ns, passive_version=PASSIVE_PITCH_VERSION)
    observe_trace(self.trace, "begin_attempt", self, sm)
    completed = False
    try:
      result = self._update(sm, now_ns=now_ns)
      completed = True
      return result
    finally:
      self.observation_result = observe(self.observation, "finish", self.observation_attempt, self, completed)
      observe_trace(self.trace, "finish_attempt", self)

  def _observe(self, **values):
    observe(self.observation, "record", self.observation_attempt, **values)

  @staticmethod
  def _passive_timestamp_valid(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and 0 < value < (1 << 64)

  def update_passive_state(self, sm, *, now_ns):
    """Consume one identified pitch/speed pair without granting motion authority.

    Return whether the pair reached the existing estimator. Its speed gate and
    fixed sample step remain unchanged. Recording each observed nonfuture input
    identity before field validation prevents an invalid or altered packet from
    being repaired in place and learned again. Future IDs cannot poison recovery.
    """
    self.last_passive_reason = 'legacy_reference_owner'
    if not self.motion_enabled:
      return False
    self.last_passive_reason = 'invalid_consumer_time'
    if not self._passive_timestamp_valid(now_ns):
      return False
    self.last_passive_reason = 'nonmonotonic_consumer_time'
    if now_ns <= self._passive_now_ns:
      return False
    self._passive_now_ns = int(now_ns)

    times = {service: getattr(sm, 'logMonoTime', {}).get(service) for service in PASSIVE_PITCH_INPUTS}
    timestamps_valid = all(self._passive_timestamp_valid(value) for value in times.values())
    causal = timestamps_valid and all(value <= now_ns for value in times.values())
    new_pair = causal and all(value > self._passive_input_times[service] for service, value in times.items())
    for service, value in times.items():
      if self._passive_timestamp_valid(value) and value <= now_ns:
        self._passive_input_times[service] = max(self._passive_input_times[service], int(value))
    self.last_passive_reason = 'invalid_measurement_id'
    if not timestamps_valid:
      return False
    self.last_passive_reason = 'future_measurement'
    if not causal:
      return False
    self.last_passive_reason = 'repeated_or_out_of_order_measurement'
    if not new_pair:
      return False
    self.last_passive_reason = 'expired_measurement'
    if any(now_ns - value >= PASSIVE_PITCH_MAX_AGE_NS for value in times.values()):
      return False
    self.last_passive_reason = 'invalid_measurement'
    if any(getattr(sm, 'valid', {}).get(service) is not True for service in PASSIVE_PITCH_INPUTS):
      return False
    orientation = sm['carControl'].orientationNED
    speed = sm['carState'].vEgo
    if len(orientation) != 3 or not math.isfinite(speed) or speed < 0.0 or not math.isfinite(orientation[1]):
      return False
    self.pitch_bias.update(orientation[1], speed)
    self.last_passive_reason = 'accepted'
    return True

  def _update(self, sm, *, now_ns):
    if self.motion_enabled:
      observe(self.observation, "record_analysis", self.observation_attempt, passiveAttempted=True)
      self.update_passive_state(sm, now_ns=now_ns)
      observe(self.observation, "record_analysis", self.observation_attempt, passiveCompleted=True)
    if self.unified:
      self.reference_planner.update(sm, now_ns=now_ns)
      return
    return self._update_legacy_reference(sm)

  def _update_legacy_reference(self, sm):
    # Generic reference path for platforms outside the supported UNIFIED gate.
    previous_target = self.output_a_target
    LongitudinalPlannerSP.update(self, sm)

    v_ego = sm['carState'].vEgo

    if len(sm['carControl'].orientationNED) == 3:
      # Road grade, not raw pitch: the field carries a static mount residual that would otherwise
      # read as a permanent hill. See PitchBiasEstimator.
      pitch = sm['carControl'].orientationNED[1]
      bias = self.pitch_bias.update(pitch, v_ego)
      accel_coast = get_coast_accel(pitch - bias)
    else:
      accel_coast = ACCEL_MAX
    self._observe(coastAccelKnown=True, coastAccel=float(accel_coast))

    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    if sm['controlsState'].forceDecel:
      v_cruise = 0.0

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET
    reset_state = reset_state or not v_cruise_initialized

    throttle_probs = sm['modelV2'].meta.disengagePredictions.gasPressProbs
    throttle_prob = throttle_probs[1] if len(throttle_probs) > 1 else 1.0
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['vehicleParameters'].angleOffsetDeg

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.output_a_target = np.clip(sm['carState'].aEgo, ACCEL_MIN, ACCEL_MAX)
      self.a_cruise = self.output_a_target
      self.creep_t = 0.0
      self.creep_stop = False
      self.creep_total_t = 0.0
      self.stop_latched = False

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    # Get new v_cruise and a_target from Smart Cruise Control and Speed Limit Assist
    policy = LongitudinalPlannerSP.update_targets(self, sm, self.v_desired_filter.x, self.output_a_target, v_cruise)
    v_cruise, self.output_a_target = policy.speed, policy.acceleration
    self._observe(policyTargetKnown=True, policySpeed=float(v_cruise), policyAccel=float(self.output_a_target))

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.output_a_target)
    if self.trace is not None:
      observe_trace(self.trace, "begin", self, sm, reset_state, v_cruise, accel_coast)
    observe(self.observation, "mpc_started", self.observation_attempt, self.mpc)
    self.mpc.update(sm['radarState'], personality=sm['selfdriveState'].personality)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)
    if self.mpc.solution_status != 0:
      self._observe(outcome="mpc_" + self.mpc.failure_reason)
      previous_target = previous_target if math.isfinite(previous_target) else MPC_FAILURE_ACCEL
      failure_target = min(previous_target, MPC_FAILURE_ACCEL)
      failure_e2e_allowed = self.is_e2e(sm)
      self._observe(modelAdmissionKnown=True, e2eAllowed=bool(failure_e2e_allowed))
      if failure_e2e_allowed and sm.all_checks(service_list=['modelV2']) and math.isfinite(sm['modelV2'].action.desiredAcceleration):
        failure_target = min(failure_target, sm['modelV2'].action.desiredAcceleration)
      self.output_a_target = float(np.clip(failure_target, ACCEL_MIN, ACCEL_MAX))
      self.stop_latched = self.stop_latched or should_stop(v_ego, self.output_a_target)
      self.output_should_stop = self.output_should_stop or self.stop_latched or sm['carState'].standstill
      self.fcw = False
      self.v_desired_filter.x = max(0.0, v_ego) if math.isfinite(v_ego) else 0.0
      if self.trace is not None:
        observe_trace(self.trace, "finish", self,
                      [(float('nan'), self.mpc.source, False), (float('nan'), LongitudinalPlanSource.cruise, False)], self.is_e2e(sm))
      return

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Save starting point for next iteration
    a_prev = self.output_a_target

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    self._observe(actionHorizonKnown=True, actionHorizon=float(action_t))
    output_a_target_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                              action_t=action_t)
    self._observe(sampledAccelKnown=True, sampledAccel=float(output_a_target_mpc))
    output_should_stop_mpc = should_stop(v_ego, output_a_target_mpc)
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    is_e2e = self.is_e2e(sm)
    self._observe(modelAdmissionKnown=True, e2eAllowed=bool(is_e2e))

    self.a_cruise = get_cruise_accel(is_e2e, v_cruise, v_ego,
                                     self.a_cruise, steer_angle_without_offset, self.CP, self.dt,
                                     accel_coast, self.allow_throttle)
    cruise_should_stop = should_stop(v_ego, self.a_cruise)

    candidates = [(output_a_target_mpc, self.mpc.source, output_should_stop_mpc),
                  (self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop)]
    if is_e2e:
      candidates.append((output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e))

    observe(self.observation, "reference_candidates", self.observation_attempt, candidates, self.a_cruise,
            output_a_target_e2e, output_should_stop_e2e, is_e2e)

    output_a_target, self.mpc.source, _ = min(candidates, key=lambda c: c[0])
    raw_should_stop = any(should_stop for _, _, should_stop in candidates)
    # Hysteresis on the release side. should_stop() is a bare threshold, `v_ego < STOP_SPEED and
    # a_target < STOP_ACCEL`, and at launch a_target climbs across STOP_ACCEL from underneath while
    # the car is still below STOP_SPEED -- so the state machine can flip at frame rate. Every flip
    # re-enters LongCtrlState.stopping, which re-enters the open loop ramp, which puts the brake
    # back on: a burst of pulses while the car is pulling away. Measured on the road it went from
    # 33/h to 57/h across the 08-25/08-26 pair, worst 13 flips inside 1.3 s.
    #
    # Latching is unchanged -- a stop is still asked for at exactly the same instant -- and only
    # letting go is made to require a plan that clearly wants to move. Re-derived over both road
    # pairs, releasing at 0.3 removed 21% of the flips (75 -> 59 over 83 engaged min) for a median
    # 0.48 s longer hold, and 0.5 cost far more (1.38 s).
    #
    # 0.3 was too high, and the road said so. The cost this comment predicted -- "a lead that inches
    # forward while the plan never asks for more than STOP_RELEASE_ACCEL is a stop this holds
    # through" -- was recorded on 08-28 route 21 t=2837.8: latched at v_ego 0.23 while the lead was
    # pulling away at 1.23 m/s, a_target climbed to a peak of +0.27 and fell back, the latch never
    # released, and the open-loop ramp walked the command from -0.14 to -2.00 and held the car
    # still for 1.31 s while the gap went 5.9 -> 8.5 m. It missed the threshold by 0.03.
    #
    # That is not a tail case: over the 08-27/08-28 pair a_target sits in [0.0, 0.3) on 8-10% of
    # stopped frames with p90 +0.11..+0.30, so 0.3 was sitting on the shoulder of the distribution
    # it is supposed to be clear of. 0.2 is the first rung that is not.
    #
    # Measured on the release ladder over both arms: 08-27 pays nothing at all (58 flips either
    # way, 35.9/h) and buys p90 launch delay 2.07 -> 0.71 s; 08-28 pays 7% more flips (43.9 ->
    # 46.9/h) and buys p90 1.27 -> 0.45 s.
    #
    # What this spends is hysteresis width: the band is now [STOP_ACCEL, 0.2), half of what it was.
    # The chatter it was introduced to stop is bounded by the flip numbers above and by the launch
    # burst count, which is the metric that actually names the complaint -- 0 bursts of >=4 flips
    # within 0.6 s on 08-28, and that is what must not come back.
    if self.stop_latched:
      self.stop_latched = not (output_a_target >= STOP_RELEASE_ACCEL or v_ego >= STOP_SPEED)
    else:
      self.stop_latched = raw_should_stop and not reset_state
    self.output_should_stop = raw_should_stop or self.stop_latched

    # Anti-creep fallback. should_stop() can only latch below STOP_SPEED, but a car whose actuator
    # has a dead band never gets there: Honda Bosch sends gas and clears BRAKE_REQUEST for any accel
    # above -0.2, so as the plan asymptotically converges its accel crosses -0.2 while the car is
    # still rolling and the speed floors out above STOP_SPEED. The stop can then never latch, which
    # is self-sustaining: to stop, the car must first stop.
    #
    # A stop that is already happening must not be preempted, because LongCtrlState.stopping
    # abandons the PID for an open loop ramp toward stopAccel and latching early hands that ramp a
    # higher speed to take off. But "the car is still shedding speed" is far too weak a test for
    # that: an MPC plan converges asymptotically, so it sheds speed the whole way down and simply
    # never arrives, which is the creep itself. Judge the band in windows of CREEP_STOP_T and ask
    # the question that decides it -- at the rate this window shed speed, will the stop be over
    # within CREEP_FINISH_T? If yes it is a stop, leave it alone. If not it is a creep, however
    # smooth. Once latched the test is over, because from then on the speed coming off is the latch
    # doing its work and re-testing it would let go and grab again for the rest of the stop.
    #
    # Measuring the rate over a window rather than reading carState.aEgo is deliberate: replayed
    # over six logged stops, aEgo in the creep band has a larger standard deviation than its mean
    # (0.24..0.36 against -0.27 m/s^2), and the noise breaks the dwell.
    #
    # The window test alone cannot end a creep, though. It extrapolates linearly, and a converging
    # plan settles into a_target ~ -k * v_ego, which sheds v_ego * k * CREEP_STOP_T per window and
    # so always predicts arrival in 1/k seconds -- and never arrives. For any k above
    # CREEP_STOP_T / CREEP_FINISH_T = 0.4 the test passes every window forever. A logged stop
    # measured k ~ 0.85 and spent the whole crawl re-passing it. CREEP_MAX_T is the answer to that:
    # however convincing each window looks, a stop that has been inside a one-metre-per-second band
    # for that long is a creep.
    #
    # The cap is its own test, not a clause of the window test. Hung off the window it could only
    # be read when creep_t > CREEP_STOP_T, i.e. at total dwell 0.55 / 1.10 / 1.65 s, so it could
    # never fire before 1.65 s -- and replayed over 33 logged stops the window test had already
    # failed by then on every one of them, which made the cap dead code. Read every frame it binds
    # where it says it does: 17 of those 33 stops latch earlier, and the four worst road-test gaps
    # (1.8, 1.9, 2.1, 2.6 m) are all in that set.
    #
    # The band is entered on CREEP_ACCEL, not STOP_ACCEL, because this test has no other sense of
    # direction. STOP_ACCEL is the threshold for "not asking for speed" and sits at +0.1, so the
    # window [0.0, 0.1) -- a plan asking to move off, gently -- accumulates dwell exactly like a
    # creep does. That is not hypothetical: on the 08-26 road pair the latch fired 3 times while
    # the car was ACCELERATING away from a stop, at up to 0.68 m/s, and each one dropped the
    # command from +0.07 to -1.38 and pulled the car back to a standstill. A real creep asks for
    # negative accel throughout -- aTarget below CREEP_SPEED has median -0.07 and p90 +0.12 -- so
    # requiring it to be negative costs no latch (31 -> 30 over 83 engaged min, median latch speed
    # 0.68 -> 0.72 m/s) and removes the launches.
    if v_ego < CREEP_SPEED and output_a_target < CREEP_ACCEL:
      if self.creep_t == 0.0:
        self.creep_v = v_ego
      self.creep_t += self.dt
      self.creep_total_t += self.dt
      if not self.creep_stop:
        if self.creep_total_t >= CREEP_MAX_T:
          self.creep_stop = True
        elif self.creep_t > CREEP_STOP_T:
          if (self.creep_v - v_ego) * CREEP_FINISH_T >= v_ego * CREEP_STOP_T:
            self.creep_t = 0.0
          else:
            self.creep_stop = True
    else:
      self.creep_t = 0.0
      self.creep_stop = False
      self.creep_total_t = 0.0
    self.output_should_stop = self.output_should_stop or self.creep_stop


    self.output_a_target = np.clip(output_a_target, ACCEL_MIN, ACCEL_MAX)

    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.output_a_target + a_prev) / 2.0
    if self.trace is not None:
      observe_trace(self.trace, "finish", self, candidates, is_e2e)

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan', logMonoTime=time.monotonic_ns())

    plan_send.valid = (sm.all_checks() and self.mpc.solution_status == 0
                       and (not self.motion_enabled or self.motion_intent.valid or self.reference_planner.standby))

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime - sm.logMonoTime['modelV2']) / 1e9
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time
    longitudinalPlan.solverStatus = int(self.mpc.solution_status)

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.present
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)
    if self.motion_enabled:
      self.reference_planner.publish(longitudinalPlan)
    if self.unified:
      self.reference_timing.publish(longitudinalPlan.reference)
    observe(self.observation, "publish", self.observation_result, longitudinalPlan, self)

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_sp(sm, pm, parent_plan_mono_time=plan_send.logMonoTime,
                                      legacy_acceleration=None if self.unified else self.output_a_target)
    if self.trace is not None:
      observe_trace(self.trace, "publish", plan_send.logMonoTime)
