#!/usr/bin/env python3
import math
from functools import partial
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
from openpilot.selfdrive.controls.lib.longcontrol import ROLLING_STOP_ACCEL, STOPPING_DECEL_RATE
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode, resolve_longitudinal_mode
from openpilot.selfdrive.controls.lib.motion_supervisor import StopReason
from openpilot.selfdrive.controls.lib.object_influence import mitigation_plan_object_influences, plan_object_influences
from openpilot.selfdrive.controls.lib.lead_qualification import _read, control_leads, qualified_lead
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_BRAKE, LEAD_DANGER_FACTOR, REFERENCE_BOUND, STOP_DISTANCE, FiniteRiskEvidence, LongitudinalMpc,
  LongitudinalPlanSource as LongitudinalPlanSource,
  get_T_FOLLOW, get_safe_obstacle_distance, get_stopped_equivalence_factor,
)
from openpilot.selfdrive.controls.lib.longitudinal_observation import PlannerObservation, observe_trace
from openpilot.selfdrive.controls.lib.cruise_response import A_CRUISE_MIN, A_CRUISE_MAX_BP, J_CRUISE_VALS
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.drive_helpers import CREEP_ACCEL, CREEP_FINISH_T
from openpilot.selfdrive.controls.lib.drive_helpers import STOP_SPEED
from openpilot.selfdrive.controls.lib.drive_helpers import CLOSE_LEAD_STOP_SPEED

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
      if qualified_lead(lead, 'caution', 0.0, now_ns=now_ns):
        # A qualified outer-edge return grants only the separate acceleration
        # precaution. It cannot require a different in-lane car to stop/depart.
        continue
      if not qualified_lead(lead, 'brake', 0.95, now_ns=now_ns):
        return False
      evidence = _read(lead, 'evidence')
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
    evidence = _read(lead, 'evidence')
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
    self._radar_publication = None
    self.longitudinal_mode = resolve_longitudinal_mode(CP, CP_SP)
    self.unified = self.longitudinal_mode == LongitudinalMode.UNIFIED

    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.pitch_bias = PitchBiasEstimator(self.dt)
    self._passive_input_times = dict.fromkeys(PASSIVE_PITCH_INPUTS, 0)
    self._passive_now_ns = 0
    self.last_passive_reason = 'not_updated'
    self.a_cruise = init_a
    self.output_a_target = init_a
    self.output_should_stop = False
    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)
    # Vehicle capabilities select one reference implementation for this lifetime.
    if self.unified:
      from openpilot.selfdrive.controls.lib.reference_timing import ReferenceTimingProducer
      from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import UnifiedLongitudinalPlanner
      self.reference_timing = ReferenceTimingProducer()
      self.reference_planner = UnifiedLongitudinalPlanner(self, init_a=init_a)
      self._update_reference = self._update_unified_reference
    else:
      from openpilot.selfdrive.controls.lib.legacy_longitudinal_planner import initialize_legacy_reference, update_legacy_reference
      initialize_legacy_reference(self)
      self._update_reference = partial(update_legacy_reference, self)

  def update(self, sm, *, now_ns=None):
    # Native execution samples its own monotonic clock. Replays must provide the
    # recorded consumer time explicitly; input timestamps cannot date themselves.
    if now_ns is None:
      now_ns = time.monotonic_ns()
    self.mpc.set_failure_context(sm, now_ns)
    self._radar_publication = None
    self.observation_result = None
    self.observation_attempt = observe(self.observation, "begin", self, sm, now_ns=now_ns, passive_version=PASSIVE_PITCH_VERSION)
    observe_trace(self.trace, "begin_attempt", self, sm)
    completed = False
    try:
      result = self._update_reference(sm, now_ns=now_ns)
      completed = True
      return result
    finally:
      self.mpc.finish_failure_context(completed)
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
    if not self.unified:
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

  def _update_unified_reference(self, sm, *, now_ns):
    observe(self.observation, "record_analysis", self.observation_attempt, passiveAttempted=True)
    self.update_passive_state(sm, now_ns=now_ns)
    observe(self.observation, "record_analysis", self.observation_attempt, passiveCompleted=True)
    consumption = self.reference_planner.radar_consumption
    consumption.clear_publication()
    self.reference_planner.update(sm, now_ns=now_ns)
    view = consumption.publication_view
    source = sm['radarState']
    if (self.reference_planner.fusion_admitted and view is not None and view._radar is source
        and consumption.publication_ns == sm.logMonoTime.get('radarState')):
      # Keep the complete publication for display attribution: an expired
      # coast may still identify an actually retained HOLD barrier. Readers
      # must be the exact update input; mutable builders keep the old path.
      self._radar_publication = (source, consumption.publication_ns, sm.logMonoTime.get('modelV2'), view)

  def _radar_for_publish(self, sm):
    source = sm['radarState']
    publication = getattr(self, '_radar_publication', None)
    if (publication is not None and source is publication[0]
        and sm.logMonoTime.get('radarState') == publication[1] and sm.logMonoTime.get('modelV2') == publication[2]):
      return publication[3]
    return source

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan', logMonoTime=time.monotonic_ns())

    plan_send.valid = (sm.all_checks() and self.mpc.solution_status == 0
                       and (not self.unified or self.motion_intent.valid or self.reference_planner.standby))

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
    longitudinalPlan.radarRiskUnresolved = bool(self.unified and self.reference_planner.radar_risk_unresolved)

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)
    if self.unified:
      self.reference_planner.publish(longitudinalPlan, now_ns=int(plan_send.logMonoTime))
      self.reference_timing.publish(longitudinalPlan.reference)
      # Display evidence observes the completed arbitration. It never uses a
      # nearest-node source label as proof that an object changed this plan.
      ref, intent = self.reference_planner, self.motion_intent
      evidence = ref.model_evidence
      independent_stop = (bool(intent.reasons & (StopReason.E2E | StopReason.CRUISE | StopReason.FORCE_DECEL))
                          or evidence is not None and (evidence.no_forward or evidence.has_rest_intent)
                          or ref.e2e_allowed and sm['modelV2'].action.shouldStop)
      try:
        radar = self._radar_for_publish(sm)
        longitudinalPlan.objectInfluences = plan_object_influences(
          radar, active_identities=getattr(self.mpc, 'object_influence_identities', ()),
          now_ns=int(plan_send.logMonoTime), permitted=bool(plan_send.valid and intent.valid and intent.ready and not ref.standby),
          fusion_admitted=ref.fusion_admitted,
          reference_accepted=bool(not ref.reference_overridden and abs(self.output_a_target - ref.reference_accel) < 1e-6
                                  and ref.constraint_accel > ref.reference_accel + 1e-6),
          independent_stop=independent_stop, object_stop=bool(intent.reasons & (StopReason.MPC | StopReason.CLOSE_LEAD)),
          held_identity=ref.radar_held_identity, release_identity=ref.radar_release_identity,
          caution_identities=getattr(ref, 'caution_object_identities', ()) if getattr(ref, 'caution_accel_limited', False) else (),
          standstill=bool(sm['carState'].standstill), target_accel=float(self.output_a_target), accel_min=ACCEL_MIN, accel_max=ACCEL_MAX,
        )
        mitigation = getattr(ref, 'mitigation_object_evidence', ())
        if mitigation and all(isinstance(proof, FiniteRiskEvidence) for proof in mitigation):
          longitudinalPlan.objectInfluences = mitigation_plan_object_influences(
            radar, mitigation,
            selected=bool(ref.fusion_admitted and not plan_send.valid and self.mpc.solution_status != 0 and intent.is_braking_valid()),
            now_ns=int(plan_send.logMonoTime), consumed_ns=int(longitudinalPlan.reference.producerMonoTime),
            reference_ns=int(longitudinalPlan.reference.referenceMonoTime),
            target_accel=float(self.output_a_target), braking_accel=intent.braking_accel, accel_min=ACCEL_MIN,
          )
      except (AttributeError, TypeError, ValueError, OverflowError):
        # A display-only record cannot interrupt plan publication.
        longitudinalPlan.objectInfluences = []
    observe(self.observation, "publish", self.observation_result, longitudinalPlan, self)

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_sp(sm, pm, parent_plan_mono_time=plan_send.logMonoTime,
                                      legacy_acceleration=None if self.unified else self.output_a_target)
    if self.trace is not None:
      observe_trace(self.trace, "publish", plan_send.logMonoTime)
