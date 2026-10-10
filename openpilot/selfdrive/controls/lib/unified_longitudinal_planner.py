import math

import numpy as np

from openpilot.cereal import log
from opendbc.car.honda.values import CAR, CarControllerParams
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.longitudinal_logging import observe
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.selfdrive.car.honda_radar_observer_config import ObserverSettings
from openpilot.selfdrive.controls.lib.cruise_response import (
  CRUISE_COAST_BP, CRUISE_COAST_MIN_SPEED, CRUISE_COAST_MAX_SPEED_ERROR, CRUISE_COAST_RISK_ACCEL,
  CruiseResponse,
)
from openpilot.selfdrive.controls.lib.drive_helpers import (CLOSE_LEAD_STOP_SPEED, CONTROL_N, CREEP_SPEED, STOP_RELEASE_ACCEL, STOP_SPEED,
                                                          get_departure_accel_from_plan, should_stop)
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState, MPC_FAILURE_ACCEL
from openpilot.selfdrive.controls.lib.longitudinal_continuation import CONTINUATION_VERSION
from openpilot.selfdrive.controls.lib.lead_qualification import MAX_MEASUREMENT_AGE_NS, control_leads, lead_identity, qualified_lead
from openpilot.selfdrive.controls.lib.radar_consumption import RadarConsumption
from openpilot.selfdrive.controls.lib.low_speed_stop_distance import LowSpeedStopDistance
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LEAD_DANGER_FACTOR, STOP_DISTANCE, MPC_INVALID_INPUT, MpcOutcome, MpcReference, T_IDXS,
)
from openpilot.selfdrive.controls.lib.model_reference import ModelReferenceAdapter, normalize_standstill_speed, stationary_lead_departure_compatible
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  CONTROL_N_T_IDX, ALLOW_THROTTLE_THRESHOLD, MIN_ALLOW_THROTTLE_SPEED,
  LEAD_BRAKING_MIN_CHANGE_SCALE, CREEP_FOLLOW_MIN_ACCEL, CREEP_FOLLOW_MODEL_MARGIN,
  CREEP_FOLLOW_MIN_GAP, CREEP_FOLLOW_MIN_LEAD_SPEED, CREEP_FOLLOW_MAX_CLOSING_SPEED,
  CREEP_FINISH_T, COMFORT_BRAKE, LongitudinalPlanSource,
  _A_TOTAL_MAX_V, _A_TOTAL_MAX_BP, get_max_accel, get_coast_accel,
  get_lead_approach_urgency, get_lead_braking_urgency,
  has_safe_moving_lead, should_stop_for_close_lead,
)
from openpilot.selfdrive.controls.lib.motion_supervisor import (MOTION_INTENT_VERSION, MotionIntent, MotionReferenceKind, MotionResponseKind, StopReason,
                                                               consumed_forward_reference)
from openpilot.selfdrive.controls.lib.longitudinal_observation import observe_trace
from openpilot.selfdrive.controls.lib.parking_approach import ParkingApproach
from openpilot.selfdrive.controls.lib.planner_continuation import PlannerContinuation
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_TIMING_VERSION, valid_timestamp
from openpilot.selfdrive.controls.lib.turn_lead_guard import TurnLeadGuard
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP, LongitudinalPlanSource as LongitudinalPlanSourceSP


INPUT_SERVICES = ['carState', 'radarState', 'modelV2', 'controlsState', 'carControl', 'selfdriveState', 'vehicleParameters']
LEAD_DEPARTURE_MIN_GAP = 3.0  # m, base clearance for a qualified lead departure
COMPATIBLE_MODEL_UNKNOWNS = frozenset({
  'model_proxy_after_initial_state', 'backward_then_forward_is_ambiguous', 'isolated_zero_has_no_rest_support',
})


def model_reference_from_velocity(raw_velocity):
  """Project forward speed, then derive one piecewise-linear-speed reference.

  Position is its exact interval integral and acceleration its interval slope.
  Independent model position/acceleration means are not a dynamics-consistent
  reference, and negative predicted speed is not a request to drive backwards.
  """
  raw_velocity = np.asarray(raw_velocity, dtype=float)
  if raw_velocity.shape != (len(ModelConstants.T_IDXS),) or not np.all(np.isfinite(raw_velocity)):
    raise ValueError("Invalid model velocity")
  velocity = np.maximum(np.interp(T_IDXS, ModelConstants.T_IDXS, raw_velocity), 0.0)
  intervals = np.diff(T_IDXS)
  position = np.concatenate(([0.0], np.cumsum(0.5 * (velocity[:-1] + velocity[1:]) * intervals)))
  acceleration = np.concatenate((np.diff(velocity) / intervals, [0.0]))
  return position, velocity, acceleration


def reference_requests_departure(velocity, acceleration, jerk):
  """Require forward motion, not recovery from a backward MPC excursion."""
  try:
    velocity, acceleration, jerk = (np.asarray(values, dtype=float) for values in (velocity, acceleration, jerk))
  except (TypeError, ValueError):
    return False
  if (velocity.shape != T_IDXS.shape or acceleration.shape != T_IDXS.shape or jerk.shape != (len(T_IDXS) - 1,)
      or not all(np.all(np.isfinite(values)) for values in (velocity, acceleration, jerk))):
    return False
  horizon = float(CONTROL_N_T_IDX[-1])
  progress_speeds = [float(velocity[0])]
  for index, interval in enumerate(np.diff(T_IDXS)):
    if T_IDXS[index] >= horizon:
      break
    duration = min(float(interval), horizon - float(T_IDXS[index]))
    initial_speed = float(velocity[index])
    initial_accel = float(acceleration[index])
    interval_jerk = float(jerk[index])
    samples = [0.0, duration]
    if interval_jerk != 0.0:
      extremum = -initial_accel / interval_jerk
      if 0.0 < extremum < duration:
        samples.append(extremum)
    progress_speeds.extend(initial_speed + initial_accel * sample + 0.5 * interval_jerk * sample ** 2 for sample in samples)
    if T_IDXS[index + 1] <= horizon:
      progress_speeds.append(float(velocity[index + 1]))
  return (all(math.isfinite(speed) and speed >= 0.0 for speed in progress_speeds)
          and max(progress_speeds) > max(float(velocity[0]), STOP_SPEED))


def low_speed_follow_distance(speed):
  """Comfort geometry; the MPC's independent braking boundary remains intact."""
  return float(np.interp(speed * CV.MS_TO_KPH, [15.0, 30.0], [4.0, 6.0]))


def sample_reference_acceleration(acceleration, action_t):
  """Read the accepted piecewise-jerk trajectory at the actuator horizon.

  Acceleration is linear within each native interval. A single-ramp inverse
  of the velocity increment has a different meaning when jerk varies, and
  subtracts the instantaneous measured initial acceleration from the request.
  Lead confidence and geometry must not select between those two meanings
  after the MPC has already incorporated the current obstacle observations.
  """
  return float(np.interp(action_t, T_IDXS, acceleration))


def cruise_accel_limit(params, speed, steering_angle, e2e_allowed, coast_accel, allow_throttle):
  maximum = ACCEL_MAX if e2e_allowed else float(get_max_accel(speed))
  if not e2e_allowed:
    lateral_limit = float(np.interp(speed, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V))
    lateral_accel = speed ** 2 * steering_angle * CV.DEG_TO_RAD / (params.steerRatio * params.wheelbase)
    maximum = min(maximum, math.sqrt(max(lateral_limit ** 2 - lateral_accel ** 2, 0.0)))
    if not allow_throttle:
      maximum = min(maximum, float(np.interp(speed, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED * 2.0],
                                            [maximum, max(coast_accel, ACCEL_MIN)])))
  return maximum


def departing_lead_rejection(radar_state, speed, delay, *, minimum_gap_base=LEAD_DEPARTURE_MIN_GAP, now_ns=None,
                             held_identity=None, allow_unbound_physical=False):
  """Return the first failed departure gate and its object, or None."""
  if not (math.isfinite(speed) and 0.0 <= speed < STOP_SPEED and math.isfinite(delay)
          and 0.0 <= delay <= 1.0 and radar_state.leadOne.present
          and not isinstance(minimum_gap_base, bool) and math.isfinite(minimum_gap_base)
          and LEAD_DEPARTURE_MIN_GAP <= minimum_gap_base <= CREEP_FOLLOW_MIN_GAP):
    return 'departure_inputs', None
  minimum_gap = minimum_gap_base + speed * (delay + DT_MDL) + speed ** 2 / (2.0 * COMFORT_BRAKE)
  primary_identity = lead_identity(radar_state.leadOne)
  if held_identity is not None and primary_identity != held_identity:
    return 'held_identity_mismatch', primary_identity
  unbound_physical = primary_identity is not None and held_identity is None
  if unbound_physical and allow_unbound_physical is not True:
    return 'unbound_identity', primary_identity
  for lead in control_leads(radar_state):
    if not lead.present:
      continue
    identity = lead_identity(lead)
    values = (lead.dRel, lead.vRel, lead.vLead, lead.aLeadK, lead.modelProb, lead.yRel)
    if not all(math.isfinite(value) for value in values):
      return 'nonfinite_lead', identity
    evidence = getattr(lead.as_reader() if hasattr(lead, 'as_reader') else lead, 'evidence', None)
    lateral = evidence.pathOffset if getattr(evidence, 'version', 0) == 1 else lead.yRel
    if getattr(evidence, 'version', 0) == 1 and identity != primary_identity:
      if not unbound_physical:
        # A known blocking vehicle may depart ahead of another distant car.
        # With no known barrier, each formal lead must instead pass
        # the full departure checks below; a new ID cannot stand in for an
        # independently observed stationary or occluded obstacle.
        if not qualified_lead(lead, 'brake', 0.95, now_ns=now_ns):
          return 'secondary_brake_unqualified', identity
        if lead.dRel + min(lead.vRel, 0.0) * CREEP_FINISH_T < minimum_gap:
          return 'secondary_gap', identity
        continue
    if lead.dRel < minimum_gap:
      return 'lead_gap', identity
    if not qualified_lead(lead, 'departure', 0.95, now_ns=now_ns):
      return 'lead_departure_unqualified', identity
    if abs(lateral) > 1.8:
      return 'lead_path_offset', identity
    if lead.vLead < CREEP_FOLLOW_MIN_LEAD_SPEED:
      return 'lead_speed', identity
    if lead.vRel < -CREEP_FOLLOW_MAX_CLOSING_SPEED:
      return 'lead_closing', identity
    if lead.vLead + min(lead.aLeadK, 0.0) * CREEP_FINISH_T < STOP_SPEED:
      return 'lead_stopping', identity
    if abs(lead.vLead - speed - lead.vRel) > CREEP_FOLLOW_MAX_CLOSING_SPEED:
      return 'lead_velocity_inconsistent', identity
  return None


def has_departing_lead(radar_state, speed, delay, *, minimum_gap_base=LEAD_DEPARTURE_MIN_GAP, now_ns=None, held_identity=None):
  return departing_lead_rejection(radar_state, speed, delay, minimum_gap_base=minimum_gap_base,
                                  now_ns=now_ns, held_identity=held_identity) is None


class RadarInputView:
  """A tick-local fallback; never mutate the published or logged sensor data."""
  def __init__(self, state, radar):
    self.state, self.radar = state, radar

  def __getitem__(self, service):
    return self.radar if service == 'radarState' else self.state[service]

  def __getattr__(self, name):
    return getattr(self.state, name)


class UnifiedLongitudinalPlanner:
  def __init__(self, planner, *, init_a=0.0):
    self.planner = planner
    self.reference_accel = 0.0
    self.constraint_accel = ACCEL_MAX
    self.constraint_source = LongitudinalPlanSource.cruise
    self.reference_source = LongitudinalPlanSource.cruise
    self.reference_overridden = False
    self.e2e_allowed = False
    self.reference_initialized = False
    self.reference_source_epoch_ns = None
    self._forward_execution_instance_ns = 0
    self.turn_lead_guard = TurnLeadGuard()
    self.turn_accel_limited = False
    self.radar_settings = ObserverSettings()
    self._radar_params = Params()
    self._radar_settings_next_ns = 0
    self._radar_update_ns = 0
    self.radar_held_identity = None
    self.radar_release_identity = None
    self._radar_stop_candidate = None
    self._radar_stop_pending = {}
    self.radar_consumption = RadarConsumption()
    self._radar_consumed_ns = None
    self._radar_consumed_view = None
    self.fusion_admitted = False
    self.cruise_response = CruiseResponse(dt=getattr(planner, 'dt', DT_MDL), init_a=init_a)
    self.model_reference_adapter = ModelReferenceAdapter()
    self.stop_distance_policy = LowSpeedStopDistance()
    self.parking_approach = ParkingApproach()
    self.parking_policy_distances = None
    self.parking_mpc_distances = None
    self.model_evidence = None
    self.model_reference_mode = 'not_admitted'
    self.model_fallback_reason = ''
    self._model_rest_active = False
    self.entered_rest_goal = False
    self.cruise_initialized = False
    self.cold_standby_started = False
    self.standby = False
    self.continuation = PlannerContinuation()
    self._departure_wait_started_ns = None
    self._departure_wait_last_log_ns = None
    planner.motion_intent = MotionIntent(mode=int(planner.longitudinal_mode), valid=False)

  def invalidate(self, speed, reason, *, preserve_continuation=False):
    planner = self.planner
    self.continuation.invalidate(reason, now_ns=self._radar_update_ns, keep_offer=preserve_continuation)
    if not preserve_continuation:
      self.cruise_response.clear()
    self.turn_lead_guard.clear()
    self.turn_accel_limited = False
    self.stop_distance_policy.clear()
    self._radar_stop_candidate = None
    self._radar_stop_pending.clear()
    self.parking_approach.clear()
    self.parking_policy_distances = None
    self.parking_mpc_distances = None
    self.reference_initialized = False
    self.reference_source_epoch_ns = None
    self._forward_execution_instance_ns = 0
    self._model_rest_active = False
    self.entered_rest_goal = False
    planner.mpc.invalidate_reference_history()
    self.standby = False
    planner.motion_intent = MotionIntent(mode=int(planner.longitudinal_mode), valid=False, producer_instance_ns=self.producer_instance_ns)
    self.reference_accel = 0.0
    self.constraint_accel = -0.2
    self.reference_overridden = True
    self.constraint_source = LongitudinalPlanSource.cruise
    self.reference_source = LongitudinalPlanSource.cruise
    previous = planner.output_a_target if math.isfinite(planner.output_a_target) else -0.2
    planner.output_a_target = float(np.clip(min(previous, -0.2), ACCEL_MIN, ACCEL_MAX))
    planner.output_should_stop = False
    planner.v_desired_trajectory = np.full(CONTROL_N, max(speed, 0.0) if math.isfinite(speed) else 0.0)
    planner.a_desired_trajectory = np.zeros(CONTROL_N)
    planner.j_desired_trajectory = np.zeros(CONTROL_N)
    planner._observe(outcome=reason, constraintKnown=True, constraintAccel=float(self.constraint_accel))

  def fail_mpc(self, state, speed, previous_request, *, preserve_continuation=False):
    planner = self.planner
    continuation_reason = self.continuation.reason
    self.invalidate(speed, "mpc_" + planner.mpc.failure_reason, preserve_continuation=preserve_continuation)
    self.continuation.reason = continuation_reason
    if planner.mpc.solution_status == 0 or not state.all_checks(service_list=INPUT_SERVICES):
      return
    previous = float(previous_request) if math.isfinite(previous_request) else MPC_FAILURE_ACCEL
    request = min(previous, MPC_FAILURE_ACCEL)
    independent = self.continuation.independent_braking_accel
    if independent is not None and math.isfinite(independent):
      request = min(request, independent)
    goal_braking = planner.mpc.rest_goal_braking_accel
    if goal_braking is not None and math.isfinite(goal_braking):
      request = min(request, goal_braking)
      independent = goal_braking if independent is None else min(independent, goal_braking)
    self.e2e_allowed = planner.is_e2e(state)
    model_request = state['modelV2'].action.desiredAcceleration
    if self.e2e_allowed and self.model_reference_mode != 'invalid' and math.isfinite(model_request):
      request = min(request, model_request)
      if model_request < 0.0:
        independent = model_request if independent is None else min(independent, model_request)
    self.continuation.independent_braking_accel = independent
    request = float(np.clip(request, ACCEL_MIN, ACCEL_MAX))
    raw_kind = self.describe_motion_reference(state)
    evidence_goal = self.model_evidence.rest_goal if self.model_evidence is not None else None
    goal_rejected = bool(raw_kind == MotionReferenceKind.REST and evidence_goal is not None and planner.mpc.rest_goal == evidence_goal
                         and planner.mpc.outcome is MpcOutcome.REST_REJECTED)
    planner.motion_intent = MotionIntent(mode=int(planner.longitudinal_mode), allow_creep=False,
                                         valid=False, braking_accel=request, reference_kind=int(raw_kind),
                                         rest_goal_rejected=goal_rejected, producer_instance_ns=self.producer_instance_ns)
    planner.output_a_target = request
    self.constraint_accel = request
    if self.e2e_allowed and self.model_reference_mode != 'invalid' and model_request <= request:
      self.constraint_source = LongitudinalPlanSource.e2e
    planner._observe(constraintKnown=True, constraintAccel=request, modelAdmissionKnown=True,
                     e2eAllowed=bool(self.e2e_allowed))

  def prepare(self, state):
    planner = self.planner
    car_state = state['carState']
    speed, measured_accel = float(car_state.vEgo), float(car_state.aEgo)
    model = state['modelV2']
    motion_state = state['controlsState'].motionState
    if (motion_state.version != MOTION_INTENT_VERSION or not motion_state.enabled
        or motion_state.planningMode != int(planner.longitudinal_mode)):
      self.invalidate(speed, "motion_owner_unavailable")
      planner.output_should_stop = bool(car_state.standstill or state['controlsState'].longControlState == LongCtrlState.stopping)
      return
    delay = planner.CP.longitudinalActuatorDelay
    values = (speed, measured_accel, car_state.vCruise, delay, planner.CP.steerRatio, planner.CP.wheelbase,
              car_state.steeringAngleDeg, state['vehicleParameters'].angleOffsetDeg, model.action.desiredAcceleration)
    if (not all(math.isfinite(value) for value in values) or not 0.0 <= delay <= 1.0
        or planner.CP.steerRatio <= 0.0 or planner.CP.wheelbase <= 0.0):
      self.invalidate(speed, "invalid_vehicle_values")
      return
    speed = max(0.0, speed)
    if len(model.velocity.x) != len(ModelConstants.T_IDXS) or len(model.orientationRate.z) != len(ModelConstants.T_IDXS):
      self.invalidate(speed, "invalid_model_shape")
      return
    if not np.all(np.isfinite(model.velocity.x)) or not np.all(np.isfinite(model.orientationRate.z)):
      self.invalidate(speed, "nonfinite_model_values")
      return

    self.standby = False
    healthy = state.all_checks(service_list=INPUT_SERVICES)
    if car_state.vCruise == V_CRUISE_UNSET:
      idle = (not state['selfdriveState'].enabled and not state['carControl'].enabled
              and state['controlsState'].longControlState == LongCtrlState.off)
      if healthy and idle and not self.cruise_initialized:
        self.cold_standby_started = True
      self.invalidate(speed, "cruise_unset")
      if (healthy and self.cold_standby_started and not self.cruise_initialized
          and state['controlsState'].longControlState == LongCtrlState.off):
        self.standby = True
        stopped = bool(car_state.standstill or speed < STOP_SPEED)
        planner.motion_intent = MotionIntent(mode=int(planner.longitudinal_mode), reasons=int(StopReason.CRUISE) if stopped else 0,
                                             allow_creep=False, valid=False, ready=False, producer_instance_ns=self.producer_instance_ns)
        planner.output_should_stop = stopped
        planner.v_desired_filter.x = speed
        self.e2e_allowed = False
        planner._observe(outcome="cold_standby", modelAdmissionKnown=True, e2eAllowed=False)
        if planner.trace is not None:
          observe_trace(planner.trace, "begin", planner, state, True, speed, ACCEL_MAX)
          observe_trace(planner.trace, "finish_unified", planner, False)
      return
    if healthy:
      self.cruise_initialized = True
      self.cold_standby_started = False
    return speed, measured_accel, healthy

  @property
  def producer_instance_ns(self):
    if self.planner.reference_timing.pending is not None:
      return self.planner.reference_timing.pending.instance_ns
    return 0

  def describe_motion_reference(self, state):
    """Describe this admitted raw source without creating a goal or stop latch."""
    evidence = self.model_evidence
    timing = self.planner.reference_timing.pending
    if (timing is None or evidence is None
        or evidence.unknown is not None and evidence.unknown.reason != 'model_proxy_after_initial_state'
        or evidence.raw_velocity is None or evidence.raw_motion is None or evidence.raw_motion.unknown is not None
        or evidence.source.model_mono_time_ns != getattr(state, 'logMonoTime', {}).get('modelV2')
        or evidence.source.anchor_mono_time_ns != timing.epoch_ns or evidence.source.consumed_mono_time_ns != timing.producer_ns):
      return MotionReferenceKind.UNKNOWN
    # This one existing alignment gap retains qualified raw source semantics;
    # it does not move a prediction back to the earlier carState timestamp.
    raw = evidence.raw_velocity
    if (raw.shape != (len(ModelConstants.T_IDXS),) or not np.all(np.isfinite(raw))
        or not np.array_equal(raw, np.asarray(state['modelV2'].velocity.x))):
      return MotionReferenceKind.UNKNOWN
    if evidence.has_rest_intent:
      return MotionReferenceKind.REST
    if np.all(raw == 0.0):
      return MotionReferenceKind.EXACT_ZERO
    if evidence.no_forward:
      return MotionReferenceKind.NO_FORWARD
    if evidence.allows_departure:
      return MotionReferenceKind.FORWARD
    return MotionReferenceKind.MOVING

  @property
  def _reference_consumer_clock_ns(self):
    return self.planner.reference_timing.last_consumer_ns

  def reference_source_epoch(self, state, now_ns):
    """Use the same qualified source record that the production message carries."""
    self.reference_source_epoch_ns = None
    return self.planner.reference_timing.qualify_source(
      getattr(state, 'logMonoTime', {}).get('carState'),
      float(state['carState'].vEgo), float(state['carState'].aEgo), now_ns,
    )

  def refresh_radar_settings(self, now_ns):
    # Persistent feature prerequisites follow the observer/DEC poll; actual
    # experimental state, control authority and message health are checked
    # every tick. Radar has one supported full-fusion path.
    try:
      if now_ns >= self._radar_settings_next_ns:
        self.radar_settings = ObserverSettings.read(self._radar_params)
        self._radar_settings_next_ns = now_ns + 1_000_000_000
    except Exception:
      self.radar_settings = ObserverSettings()

  def radar_input_state(self, state, now_ns):
    self._radar_consumed_ns = self._radar_consumed_view = None
    self.radar_consumption.clear()
    self.refresh_radar_settings(now_ns)
    radar = state['radarState']
    if hasattr(radar, 'as_reader'):
      radar = radar.as_reader()
      state = RadarInputView(state, radar)
    self.fusion_admitted = bool(
      (getattr(radar, 'fusionActive', False) or getattr(radar, 'fusionDegraded', False)) and self.planner.unified
      and self.radar_settings.control_requested(self.planner.CP, self.planner.CP_SP)
      and state['selfdriveState'].experimentalMode and self.planner.dec.enabled()
    )
    if not self.fusion_admitted:
      if not getattr(radar, 'fusionActive', False):
        return state
      # Scope exit is a pure sensor fallback. It does not require the vehicle
      # inputs used only by active consumption, or rewrite them.
      fallback = log.RadarState.new_message()
      fallback.mdMonoTime, fallback.radarErrors = radar.mdMonoTime, radar.radarErrors
      fallback.leadOne, fallback.leadTwo = radar.visionLeadOne, radar.visionLeadTwo
      fallback.fusionReason = 'consumerScopeExit'
      return RadarInputView(state, fallback)
    if getattr(radar, 'fusionActive', False):
      if state.valid.get('radarState', False) and state.all_checks(service_list=['radarState']):
        radar = self.radar_consumption.update(
          radar, publication_ns=state.logMonoTime.get('radarState'), now_ns=now_ns,
          speed=normalize_standstill_speed(state['carState'].vEgo, actual_standstill=state['carState'].standstill),
        )
        if (self.radar_consumption.publication_view is not None
            and self.radar_consumption.publication_ns == state.logMonoTime.get('radarState')):
          self._radar_consumed_ns, self._radar_consumed_view = now_ns, radar
      return RadarInputView(state, radar)
    return state

  def unbound_radar_departure_rejection(self, state, *, now_ns):
    """Qualify the current radar publication before using its departure evidence."""
    radar = state['radarState']
    publication_ns = state.logMonoTime.get('radarState')
    primary_identity = lead_identity(radar.leadOne)
    if (self.fusion_admitted is not True or self._radar_consumed_ns != now_ns
        or self._radar_consumed_view is not radar or self.radar_consumption.publication_view is None
        or self.radar_consumption.publication_ns != publication_ns
        or state.valid.get('radarState') is not True or not state.all_checks(service_list=['radarState'])):
      return 'radar_consumption_unconfirmed'
    if (not valid_timestamp(now_ns) or not valid_timestamp(publication_ns)
        or not 0 <= now_ns - publication_ns <= MAX_MEASUREMENT_AGE_NS
        or not valid_timestamp(getattr(radar, 'mdMonoTime', None))
        or not 0 <= now_ns - radar.mdMonoTime <= MAX_MEASUREMENT_AGE_NS
        or radar.mdMonoTime > publication_ns):
      return 'radar_publication_stale'
    if (getattr(radar, 'fusionActive', None) is not True or getattr(radar, 'fusionHealthy', None) is not True
        or getattr(radar, 'fusionDegraded', None) is not False):
      return 'radar_producer_unhealthy'
    if self.radar_consumption.degraded:
      return 'radar_measurement_expired'
    if primary_identity is None or primary_identity[0] != getattr(radar, 'fusionSessionId', None):
      return 'radar_session_mismatch'
    speed = normalize_standstill_speed(state['carState'].vEgo, actual_standstill=state['carState'].standstill)
    if state['carState'].standstill is not True or not math.isfinite(speed) or not 0.0 <= speed <= 0.05:
      return 'ego_not_stationary'
    return None

  def log_departure_blocked(self, state, *, now_ns, active, release_allowed, output, stage, rejected_identity,
                            admission_reasons, no_model_risk):
    """Sparse ordinary log evidence; neither logging nor its failures control."""
    try:
      motion = state['controlsState'].motionState
      controls_ns = state.logMonoTime.get('controlsState')
      waiting = (active and state['carState'].standstill is True and math.isfinite(output) and output > 0.0
                 and not release_allowed and motion.enabled and motion.phase in ('hold', 'approach') and motion.resolvedStop
                 and valid_timestamp(now_ns) and valid_timestamp(controls_ns) and 0 <= now_ns - controls_ns <= 250_000_000)
      if not waiting:
        self._departure_wait_started_ns = None
        return
      if self._departure_wait_started_ns is None or now_ns < self._departure_wait_started_ns:
        self._departure_wait_started_ns = now_ns
      if (now_ns - self._departure_wait_started_ns < 500_000_000
          or self._departure_wait_last_log_ns is not None and now_ns - self._departure_wait_last_log_ns < 5_000_000_000):
        return
      # Advance before constructing or sending: failed log transport must not
      # create a 20 Hz retry loop or affect any motion decision.
      self._departure_wait_last_log_ns = now_ns
      radar, evidence = state['radarState'], self.model_evidence
      timing = self.planner.reference_timing.pending
      cloudlog.event(
        'longitudinal_departure_blocked_v1', version=1, decision_mono_ns=str(now_ns),
        wait_seconds=(now_ns - self._departure_wait_started_ns) / 1e9,
        stage=stage, rejected_identity=rejected_identity,
        input_mono_ns={name: str(state.logMonoTime.get(name)) for name in ('carState', 'modelV2', 'radarState', 'controlsState')},
        plan_producer_mono_ns=str(timing.producer_ns) if timing is not None else None,
        held_identity=self.radar_held_identity, primary_identity=lead_identity(radar.leadOne),
        consumer_degraded=self.radar_consumption.degraded,
        producer_healthy=getattr(radar, 'fusionHealthy', None), producer_degraded=getattr(radar, 'fusionDegraded', None),
        consumption_mono_ns=str(self._radar_consumed_ns) if self._radar_consumed_ns is not None else None,
        model_unknown=evidence.unknown.reason if evidence is not None and evidence.unknown is not None else None,
        raw_motion=evidence.raw_motion.kind if evidence is not None and evidence.raw_motion is not None else None,
        admission_reasons=int(admission_reasons), no_model_risk=bool(no_model_risk), a_target=float(output),
        model_action=float(state['modelV2'].action.desiredAcceleration), measured_speed=float(state['carState'].vEgo),
      )
    except Exception:
      pass

  def adopt_executed_radar_stop(self, state, *, now_ns):
    """Bind an object barrier only after the owner adopted its stop request.

    Detection/qualification and display-only influence records cannot create
    a HOLD. The existing owner timing receipt identifies the actual preceding
    reference; missing feedback is not evidence that a held object cleared.
    """
    candidate = self._radar_stop_candidate
    self._radar_stop_candidate = None
    if candidate is not None:
      self._radar_stop_pending[candidate[2]] = candidate
    # A valid plan can be published just before the next planner tick, with
    # the owner's acknowledgement arriving afterwards. Retain a bounded
    # pending set for one plan lease plus one feedback lease, without treating
    # one missed acknowledgement as proof that the stop was never executed.
    self._radar_stop_pending = {
      stamp: value for stamp, value in self._radar_stop_pending.items()
      if 0 <= now_ns - stamp <= 500_000_000 and value[3] == self.producer_instance_ns
    }
    if len(self._radar_stop_pending) > 16:
      self._radar_stop_pending = dict(sorted(self._radar_stop_pending.items())[-16:])
    receipt_services = ('carControl', 'controlsState')
    if (not all(getattr(state, 'seen', {}).get(service, False) and state.valid.get(service, False)
                for service in receipt_services)
        or not state.all_checks(service_list=list(receipt_services))):
      return
    control, controls = state['carControl'], state['controlsState']
    motion = controls.motionState
    candidate = self._radar_stop_pending.get(motion.referenceProducerMonoTime)
    if candidate is None:
      return
    identity, source_ns, producer_ns, instance_ns = candidate
    control_ns = state.logMonoTime.get('carControl', 0)
    controls_ns = state.logMonoTime.get('controlsState', 0)
    if (not all(isinstance(stamp, int) and not isinstance(stamp, bool) and producer_ns < stamp <= now_ns
                and now_ns - stamp <= 250_000_000 for stamp in (control_ns, controls_ns))
        or not motion.enabled or motion.phase not in ('approach', 'hold') or not motion.resolvedStop
        or motion.degraded or motion.waitingForFreshInput or not motion.reasons & int(StopReason.MPC | StopReason.CLOSE_LEAD)
        or not math.isfinite(control.actuators.accel) or control.actuators.accel > 0.0
        or motion.referenceTimingVersion != REFERENCE_TIMING_VERSION or not motion.referenceTimingValid
        or motion.referenceProducerInstanceMonoTime != instance_ns
        or motion.referenceProducerMonoTime != producer_ns
        or motion.referenceSourceMonoTime != source_ns or motion.referenceEpochMonoTime != source_ns
        or not producer_ns < motion.referenceConsumerMonoTime <= min(control_ns, controls_ns)):
      return
    del self._radar_stop_pending[producer_ns]
    if self.radar_held_identity is None or self.radar_held_identity == identity:
      self.radar_held_identity = identity
      self.radar_release_identity = None

  def independent_constraint(self, state, reference_accel, *, speed, maximum_accel, cruise_ceiling,
                             cruise_speed, active, healthy, reset, forced_decel):
    """The ordinary current caps, without the generic failed-solver baseline."""
    planner = self.planner
    model = state['modelV2']
    coast_applicable = (
      planner.CP.carFingerprint == CAR.HONDA_ACCORD and planner.CP.openpilotLongitudinalControl
      and active and healthy and not reset and not forced_decel and not self.e2e_allowed and planner.allow_throttle
      and not planner.fcw and not should_stop(speed, reference_accel) and not model.action.shouldStop
      and math.isfinite(reference_accel) and math.isfinite(model.action.desiredAcceleration)
      and model.action.desiredAcceleration >= CRUISE_COAST_RISK_ACCEL
      and planner.source == LongitudinalPlanSourceSP.cruise and cruise_ceiling <= reference_accel
      and speed >= CRUISE_COAST_MIN_SPEED and abs(cruise_speed - speed) <= CRUISE_COAST_MAX_SPEED_ERROR
      and not (state['radarState'].leadOne.present or state['radarState'].leadTwo.present)
      and CRUISE_COAST_BP[0] < cruise_ceiling < CRUISE_COAST_BP[-1]
    )
    cruise_command = self.cruise_response.coast_command(cruise_ceiling, eligible=coast_applicable)
    ceiling = min(maximum_accel, cruise_command)
    source = LongitudinalPlanSource.cruise
    if cruise_speed <= STOP_SPEED:
      ceiling = min(ceiling, CarControllerParams.BOSCH_GAS_LOOKUP_BP[0])
    if self.e2e_allowed and self.model_reference_mode != 'invalid' and model.action.desiredAcceleration < ceiling:
      ceiling = float(model.action.desiredAcceleration)
      source = LongitudinalPlanSource.e2e
    evidence = self.model_evidence
    model_stop_evidence = evidence is not None and (evidence.no_forward or evidence.has_rest_intent)
    if model_stop_evidence and speed < STOP_SPEED and CarControllerParams.BOSCH_GAS_LOOKUP_BP[0] < ceiling:
      ceiling = CarControllerParams.BOSCH_GAS_LOOKUP_BP[0]
      source = LongitudinalPlanSource.e2e
    return float(ceiling), source, cruise_command, coast_applicable

  def update(self, state, *, now_ns):
    self.continuation.begin()
    state = self.radar_input_state(state, now_ns)
    self._radar_update_ns = now_ns
    planner = self.planner
    self.model_reference_mode = 'not_admitted'
    self.model_fallback_reason = ''
    self.model_evidence = None
    previous_source_ns = self.reference_source_epoch_ns
    previous_producer_ns = self._reference_consumer_clock_ns
    previous_instance_ns = self.producer_instance_ns
    source_epoch_ns = self.reference_source_epoch(state, now_ns)
    if source_epoch_ns is None:
      self.model_reference_adapter.clear_rest_continuity()
      self.invalidate(float(state['carState'].vEgo), "invalid_reference_source_epoch")
      return
    reference_continuous = (
      self.reference_initialized and previous_source_ns is not None
      and 0 <= source_epoch_ns - previous_source_ns <= planner.reference_timing.max_age_ns
      and 0 < now_ns - previous_producer_ns <= planner.reference_timing.max_age_ns
      and self.producer_instance_ns == previous_instance_ns != 0
    )
    previous_reference_valid = planner.mpc.prepare_reference_history(source_epoch_ns, valid=self.reference_initialized)
    prepared = self.prepare(state)
    if prepared is None:
      self.model_reference_adapter.clear_rest_continuity()
      return
    speed, measured_accel, healthy = prepared
    car_state = state['carState']
    model = state['modelV2']
    delay = planner.CP.longitudinalActuatorDelay
    LongitudinalPlannerSP.update(planner, state)
    self.e2e_allowed = planner.is_e2e(state)
    planner._observe(modelAdmissionKnown=True, e2eAllowed=bool(self.e2e_allowed))
    active = (state['selfdriveState'].enabled and state['carControl'].enabled and state['carControl'].longActive
              and not car_state.gasPressed and not car_state.brakePressed and not state['carControl'].cruiseControl.override)
    reset = state['controlsState'].longControlState == LongCtrlState.off or car_state.vCruise == V_CRUISE_UNSET or not active
    radar = state['radarState']
    motion = state['controlsState'].motionState
    motion_ns = state.logMonoTime.get('controlsState', 0)
    released = (self.radar_release_identity is not None and self.radar_release_identity == self.radar_held_identity
                and motion.enabled and motion.phase in ('release', 'track')
                and 0 < motion_ns <= now_ns and now_ns - motion_ns <= 250_000_000)
    scope_exit = (not self.radar_settings.control_requested(planner.CP, planner.CP_SP)
                  or not state['selfdriveState'].experimentalMode or not planner.dec.enabled())
    if not active or released:
      self.radar_held_identity = None
      self.radar_release_identity = None
      self._radar_stop_candidate = None
      self._radar_stop_pending.clear()
    else:
      self.adopt_executed_radar_stop(state, now_ns=now_ns)
    if scope_exit:
      # A configuration change prevents new radar stop claims. It is not
      # evidence that an already executed physical stop obstacle departed.
      self._radar_stop_candidate = None
      self._radar_stop_pending.clear()
    if self.fusion_admitted:
      clear_session = getattr(radar, 'fusionSessionId', '')
      cleared = getattr(radar, 'fusionClearedTracks', ())
      if (self.radar_held_identity is not None and self.radar_held_identity[0] == clear_session
          and self.radar_held_identity[1] in cleared and getattr(radar, 'fusionHealthy', False)):
        self.radar_held_identity = None
    retain_rest_continuity = healthy and active and not reset and self.e2e_allowed
    if not retain_rest_continuity:
      self.model_reference_adapter.clear_rest_continuity()
    # vEgo and aEgo are one carState estimator result. A second speed filter or
    # a fixed-step prediction would date the two initial-state components
    # differently, including on startup and when planner updates are skipped.
    initial_speed = speed
    # Preserve the shared trace snapshot; UNIFIED does not consume this filter
    # as either a motion estimate or a future reference state.
    planner.v_desired_filter.x = initial_speed
    initial_accel = float(np.clip(measured_accel, ACCEL_MIN, ACCEL_MAX))
    if initial_speed == 0.0:
      initial_accel = max(0.0, initial_accel)

    coast_accel = ACCEL_MAX
    if len(state['carControl'].orientationNED) == 3:
      pitch = state['carControl'].orientationNED[1]
      if not math.isfinite(pitch):
        self.model_reference_adapter.clear_rest_continuity()
        self.invalidate(speed, "nonfinite_pitch")
        return
      coast_accel = float(get_coast_accel(pitch - planner.pitch_bias.value))
    planner._observe(coastAccelKnown=True, coastAccel=float(coast_accel))
    probabilities = model.meta.disengagePredictions.gasPressProbs
    throttle_probability = probabilities[1] if len(probabilities) > 1 else 1.0
    if not math.isfinite(throttle_probability):
      self.model_reference_adapter.clear_rest_continuity()
      self.invalidate(speed, "nonfinite_throttle_probability")
      return
    planner.allow_throttle = throttle_probability > ALLOW_THROTTLE_THRESHOLD or speed <= MIN_ALLOW_THROTTLE_SPEED
    cruise_speed = min(car_state.vCruise, V_CRUISE_MAX) * CV.KPH_TO_MS
    forced_decel = state['controlsState'].forceDecel
    if forced_decel:
      cruise_speed = 0.0
    policy = LongitudinalPlannerSP.update_targets(planner, state, initial_speed, initial_accel, cruise_speed)
    cruise_speed, policy_accel = policy.speed, policy.acceleration
    planner._observe(policyTargetKnown=True, policySpeed=float(cruise_speed), policyAccel=float(policy_accel))
    if not math.isfinite(cruise_speed) or not math.isfinite(policy_accel):
      self.model_reference_adapter.clear_rest_continuity()
      self.invalidate(speed, "nonfinite_policy_target")
      return
    cruise_speed = max(0.0, cruise_speed)
    steering = car_state.steeringAngleDeg - state['vehicleParameters'].angleOffsetDeg
    maximum_accel = cruise_accel_limit(planner.CP, speed, steering, self.e2e_allowed, coast_accel, planner.allow_throttle)
    vision_curve = planner.scc.vision
    scc_braking = (healthy and active and not reset and not forced_decel
                   and vision_curve.hold_entry_speed and vision_curve.is_active
                   and vision_curve.output_v_target < speed and vision_curve.output_a_target < 0.0)
    if scc_braking:
      # The current/predicted geometry still owns necessary deceleration even
      # when another policy supplies the lower speed target. update_targets
      # has already checked the active curve outputs for finite values.
      maximum_accel = min(maximum_accel, float(np.clip(vision_curve.output_a_target, ACCEL_MIN, ACCEL_MAX)))
    turn_ceiling = self.turn_lead_guard.update(
      now_ns=now_ns, model_ns=state.logMonoTime.get('modelV2'), radar_ns=state.logMonoTime.get('radarState'),
      enabled=healthy and active and not reset and not forced_decel and planner.unified,
      speed=speed, standstill=bool(car_state.standstill), steering_angle=steering,
      steer_ratio=planner.CP.steerRatio, wheelbase=planner.CP.wheelbase,
      model_curvature=model.action.desiredCurvature,
      lead_one=state['radarState'].leadOne, lead_two=state['radarState'].leadTwo, ceiling_max=ACCEL_MAX,
      independent_braking=(self.e2e_allowed and (model.action.desiredAcceleration < 0.0 or model.action.shouldStop)
                           or scc_braking
                           or policy.source != LongitudinalPlanSourceSP.cruise and policy_accel < 0.0
                           or cruise_speed < (self.turn_lead_guard.hold_speed if self.turn_lead_guard.hold_speed is not None else speed)),
    )
    self.turn_accel_limited = turn_ceiling is not None and turn_ceiling < maximum_accel
    if turn_ceiling is not None:
      # A disappearing lead in a following turn is not evidence of clear road.
      # Keep this ceiling in both the native reference and the existing single
      # arbitration. It does not invent a lead, a stop goal or radar authority.
      maximum_accel = min(maximum_accel, turn_ceiling)
    if self.turn_lead_guard.hold_speed is not None:
      cruise_speed = min(cruise_speed, self.turn_lead_guard.hold_speed)
    if healthy:
      cruise_ceiling = self.cruise_response.update(
        cruise_speed, speed, maximum_accel, measured_accel=measured_accel, reset=reset)
    else:
      self.cruise_response.clear()
      cruise_ceiling = maximum_accel
    model_position = model_velocity = model_acceleration = None
    self.model_evidence = None
    if self.e2e_allowed:
      fields = (model.position.x, model.velocity.x, model.acceleration.x)
      if any(len(values) != len(ModelConstants.T_IDXS) or not np.all(np.isfinite(values)) for values in fields):
        self.model_reference_adapter.clear_rest_continuity()
        self.invalidate(speed, "invalid_e2e_reference")
        return
      model_position, model_velocity, model_acceleration = model_reference_from_velocity(model.velocity.x)
      self.model_evidence = self.model_reference_adapter.build(
        model.velocity.x, T_IDXS, initial_speed,
        model_mono_time_ns=getattr(state, 'logMonoTime', {}).get('modelV2'),
        anchor_mono_time_ns=source_epoch_ns, consumed_mono_time_ns=now_ns,
        retain_rest_continuity=retain_rest_continuity,
      )
      unknown = self.model_evidence.unknown
      self.model_reference_mode = 'typed'
      if unknown is not None:
        self.model_fallback_reason = unknown.reason
        self.model_reference_mode = ('compatibility_risk_only' if unknown.reason in COMPATIBLE_MODEL_UNKNOWNS else 'invalid')
        if self.model_reference_mode == 'invalid':
          # A future/expired/mutated source has no qualified new model risk
          # input. The same MPC still consumes independently valid real leads
          # and cruise; its invalid output cannot authorize forward movement.
          model_position = model_velocity = model_acceleration = None
          # Losing this source also changes the admitted objective. Do not
          # smooth independent current lead braking against a preceding
          # acceleration trajectory whose model contract is now invalid.
          planner.mpc.discard_prepared_reference_cost()
          previous_reference_valid = False
      if self.model_reference_mode != 'invalid' and model.action.shouldStop:
        cruise_speed = 0.0
    evidence = self.model_evidence
    ordinary_model_motion = (evidence is not None and not evidence.has_rest_intent and not evidence.no_forward
                             and (evidence.unknown is None
                                  or evidence.unknown.reason == 'model_proxy_after_initial_state'
                                  and evidence.raw_motion is not None and evidence.raw_motion.does_not_veto_departure))
    if ordinary_model_motion:
      # P's ordinary driving policy follows absolute lead/cruise targets and
      # the independent action cap. Preserve raw model facts for the owner,
      # without adding a second ordinary velocity target to this same MPC.
      model_position = model_velocity = model_acceleration = None
      self.model_reference_mode = 'action_capped_absolute'
    has_rest_goal = self.model_evidence is not None and self.model_evidence.rest_goal is not None
    self.entered_rest_goal = has_rest_goal and not self._model_rest_active
    self._model_rest_active = has_rest_goal
    if self.entered_rest_goal:
      # A newly admitted rest goal replaces the preceding moving semantics.
      # Keep this new epoch pending so the next same-goal frame can use its
      # accepted trajectory; do not clear history on each new model identity.
      planner.mpc.discard_prepared_reference_cost()
      previous_reference_valid = False
    # Ordinary speed response already belongs to CruiseResponse, including
    # admitted MOVING/FORWARD with an independent action cap. Keep native lead
    # risks and bounds, without a second speed target.
    # Unadmitted raw model signs do not select between two cruise policies.
    native_cruise_external = (
      healthy and active and not reset and not forced_decel
      and policy.source == LongitudinalPlanSourceSP.cruise and cruise_speed > STOP_SPEED
      and ((not self.e2e_allowed and self.model_evidence is None) or ordinary_model_motion)
      and model_position is model_velocity is model_acceleration is None
    )
    compact_permitted = (healthy and active and not reset and not forced_decel
                         and planner.CP.carFingerprint == CAR.HONDA_ACCORD
                         and planner.CP.openpilotLongitudinalControl and planner.CP.radarUnavailable and planner.unified)
    # A stopped car never acquires a closer target from distance error alone.
    # Re-entry from standstill additionally needs an independently departing
    # lead plus a positive, non-stopping model; the existing owner still decides
    # whether the resulting reference can release the brake.
    observed_departure = (bool(car_state.standstill) and not model.action.shouldStop
                          and model.action.desiredAcceleration >= STOP_RELEASE_ACCEL
                          and (self.model_evidence is None or self.model_evidence.allows_external_departure)
                          and has_departing_lead(state['radarState'], speed, delay, minimum_gap_base=LEAD_DEPARTURE_MIN_GAP,
                                                 now_ns=now_ns, held_identity=self.radar_held_identity))
    compact_activation = ((not car_state.standstill and state['controlsState'].longControlState == LongCtrlState.pid)
                          or observed_departure)
    uncertainty = self.parking_approach.uncertainty_cache.update(
      now_ns=now_ns, model_ns=state.logMonoTime.get('modelV2'), model=model, radar_ns=state.logMonoTime.get('radarState'),
      radar=state['radarState'], valid=healthy,
    )
    if not uncertainty.qualified:
      # Source/quality failures invalidate the same complete history as a
      # changed lead. A later good observation cannot spend pre-fault time.
      self.stop_distance_policy.clear(source_barrier_ns=source_epoch_ns)
    comfort_distance, lead_risk_distances = self.stop_distance_policy.update(
      now_ns=now_ns, source_ns=source_epoch_ns, radar_ns=state.logMonoTime.get('radarState'),
      speed=speed, radar=state['radarState'], permitted=compact_permitted,
      allow_activation=compact_activation,
    )
    self.parking_policy_distances = (comfort_distance, lead_risk_distances)
    parking = self.parking_approach.update(
      now_ns=now_ns, source_ns=source_epoch_ns, radar_ns=state.logMonoTime.get('radarState'),
      speed=speed, measured_accel=measured_accel, actuator_delay=delay, radar=state['radarState'],
      qualification=self.stop_distance_policy.evidence, permitted=compact_permitted,
      allow_activation=compact_activation and not car_state.standstill,
      uncertainty=uncertainty,
    )
    if not parking.allow_compact:
      # Fallback retains soft geometry, not a guaranteed clearance or an
      # executed stopping trajectory. Scalar/model/owner arbitration is unchanged.
      comfort_distance, lead_risk_distances = low_speed_follow_distance(speed), (STOP_DISTANCE, STOP_DISTANCE)
    self.parking_mpc_distances = (comfort_distance, lead_risk_distances)
    reference = MpcReference(
      cruise_speed=None if native_cruise_external else cruise_speed, accel_min=ACCEL_MIN, accel_max=maximum_accel,
      model_position=model_position, model_velocity=model_velocity, model_acceleration=model_acceleration,
      typed_model=None if ordinary_model_motion else self.model_evidence, allow_future_rest_reschedule=True,
      comfort_stop_distance=comfort_distance, lead_risk_stop_distances=lead_risk_distances,
    )
    urgency = 0.0
    if active and healthy and not reset and not forced_decel and planner.source == LongitudinalPlanSourceSP.cruise:
      urgency = max(
        get_lead_braking_urgency(speed, planner.output_a_target, state['radarState'], delay, now_ns=now_ns),
        get_lead_approach_urgency(speed, state['radarState'], delay, personality=state['selfdriveState'].personality, now_ns=now_ns),
      )
    # Only smooth against a qualified native reference aligned to this epoch.
    # Neither reset zeros nor measured acceleration are a previous request.
    reference_qualified = healthy and not (reset or car_state.standstill)
    if not reference_qualified:
      planner.mpc.invalidate_reference_history()
    planner.mpc.set_weights(previous_reference_valid and reference_qualified, personality=state['selfdriveState'].personality,
                            a_change_scale=1.0 - (1.0 - LEAD_BRAKING_MIN_CHANGE_SCALE) * urgency)
    planner.mpc.set_cur_state(initial_speed, initial_accel)
    planner.reference_timing.set_initial_state(initial_speed, initial_accel)
    if planner.trace is not None:
      observe_trace(planner.trace, "begin", planner, state, reset, cruise_speed, coast_accel)
    observe(planner.observation, "mpc_started", planner.observation_attempt, planner.mpc)
    continuation_permitted = bool(
      planner.CP.carFingerprint == CAR.HONDA_ACCORD and planner.CP.openpilotLongitudinalControl and planner.unified
      and active and healthy and not reset and not forced_decel and not car_state.standstill
      and speed >= CLOSE_LEAD_STOP_SPEED and not planner.fcw and not self.radar_consumption.degraded
      and self.radar_held_identity is None and not self.turn_lead_guard.lost
      and planner.source == LongitudinalPlanSourceSP.cruise and cruise_speed > STOP_SPEED
      and not model.action.shouldStop and self.model_reference_mode != 'invalid'
      and (not self.e2e_allowed or ordinary_model_motion)
    )
    continuation_scope = (self.e2e_allowed, self.fusion_admitted, bool(getattr(radar, 'fusionActive', False)),
                          str(state['selfdriveState'].personality))
    constraint_kwargs = dict(speed=speed, maximum_accel=maximum_accel, cruise_ceiling=cruise_ceiling,
                             cruise_speed=cruise_speed, active=active, healthy=healthy, reset=reset,
                             forced_decel=forced_decel)
    planner.mpc.update(state['radarState'], personality=state['selfdriveState'].personality, reference=reference,
                       now_ns=now_ns, reference_ns=source_epoch_ns)
    if planner.mpc.solution_status != 0:
      offer = self.continuation.attempt(
        planner.mpc, state, planner.reference_timing.pending, now_ns=now_ns,
        permitted=continuation_permitted, scope=continuation_scope, action_t=delay + DT_MDL,
        independent_ceiling=lambda value: self.independent_constraint(state, value, **constraint_kwargs)[0],
      )
      self.fail_mpc(state, speed, planner.output_a_target, preserve_continuation=offer is not None)
      return
    # Eligibility below consumes the current solve's collision warning, never
    # the preceding plan's flag.
    planner.fcw = planner.mpc.crash_cnt > 2 and not car_state.standstill
    planner.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS, planner.mpc.v_solution)
    planner.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS, planner.mpc.a_solution)
    planner.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS[:-1], planner.mpc.j_solution)
    planner._observe(actionHorizonKnown=True, actionHorizon=float(delay + DT_MDL))
    # Keep the existing actuator preview and the measured MPC initial state.
    # The accepted trajectory already carries its qualified continuity cost;
    # read it consistently instead of modulating a second inverse with noisy
    # lead estimates. Braking and motion-authority constraints remain separate.
    self.reference_accel = sample_reference_acceleration(
      planner.mpc.a_solution, delay + DT_MDL)
    planner._observe(sampledAccelKnown=True, sampledAccel=self.reference_accel)
    if not math.isfinite(self.reference_accel):
      self.invalidate(speed, "nonfinite_sampled_accel")
      return
    # The coast policy changes only this cruise candidate. Its raw response
    # and stop evidence remain separate, and stronger current references and
    # constraints still participate in the one arbitration below.
    # The selected policy owns its response history. It supplies one current
    # target bound, without rewriting the measured MPC initial state or
    # introducing another final-output decision or stop owner.
    (self.constraint_accel, self.constraint_source, cruise_command, coast_applicable) = self.independent_constraint(
      state, self.reference_accel, **constraint_kwargs)
    self.reference_source = planner.mpc.source
    evidence = self.model_evidence
    model_stop_evidence = evidence is not None and (evidence.no_forward or evidence.has_rest_intent)
    planner._observe(constraintKnown=True, constraintAccel=float(self.constraint_accel))
    output = float(np.clip(min(self.reference_accel, self.constraint_accel), ACCEL_MIN, ACCEL_MAX))
    if evidence is not None and evidence.unknown is not None and self.model_reference_mode == 'invalid':
      # The same successful MPC still consumed admitted leads. Preserve its
      # qualified braking and the preceding request while refusing movement
      # from a model source that failed its own input contract.
      previous = planner.output_a_target if math.isfinite(planner.output_a_target) else -0.2
      request = float(np.clip(min(output, previous, -0.2), ACCEL_MIN, ACCEL_MAX))
      self.invalidate(speed, 'unknown_model_' + evidence.unknown.reason)
      # This is a producer-input failure, not an acados return code. The
      # existing owner accepts a braking envelope only with a nonzero status;
      # retain diagnostic_last_solver_return as the actual native result.
      planner.mpc.solution_status = MPC_INVALID_INPUT
      planner.mpc.failure_reason = 'unqualified_model_' + evidence.unknown.reason
      planner.mpc.previous_solution_valid = False
      planner.output_a_target = request
      self.constraint_accel = request
      planner.motion_intent = MotionIntent(mode=int(planner.longitudinal_mode), allow_creep=False,
                                           valid=False, braking_accel=request, producer_instance_ns=self.producer_instance_ns)
      planner._observe(constraintKnown=True, constraintAccel=request)
      return
    self.reference_overridden = output != self.reference_accel
    if output < self.reference_accel:
      planner.mpc.source = self.constraint_source

    forward_reference = reference_requests_departure(planner.mpc.v_solution, planner.mpc.a_solution, planner.mpc.j_solution)
    low_speed_preference_stop = (planner.mpc.v_solution[-1] < STOP_SPEED
                                and (should_stop(speed, self.reference_accel) or speed < STOP_SPEED and not forward_reference))
    reasons = StopReason(0)
    if model_stop_evidence and speed < STOP_SPEED:
      reasons |= StopReason.MPC
    if self.e2e_allowed and model.action.shouldStop:
      reasons |= StopReason.E2E
    # Preserve the cruise candidate's raw stop fact as well as its numeric
    # ceiling. Only the motion owner retains the resulting stop/release state.
    if cruise_speed <= STOP_SPEED or should_stop(speed, cruise_ceiling):
      reasons |= StopReason.FORCE_DECEL if forced_decel else StopReason.CRUISE
    qualified = active and healthy and not forced_decel and not planner.fcw and planner.source == LongitudinalPlanSourceSP.cruise
    if qualified and should_stop_for_close_lead(speed, output, state['radarState'], delay, a_ego=car_state.aEgo, now_ns=now_ns):
      reasons |= StopReason.CLOSE_LEAD
    # A low-speed preference cannot prove a new release. Keep that original
    # admission mask even when it need not end already executed forward motion.
    admission_reasons = reasons | (StopReason.MPC if low_speed_preference_stop else StopReason(0))
    raw_kind = self.describe_motion_reference(state)
    control = state['carControl']
    controls = state['controlsState']
    control_ns = getattr(state, 'logMonoTime', {}).get('carControl')
    controls_ns = getattr(state, 'logMonoTime', {}).get('controlsState')
    executed_forward = (
      reference_continuous
      and active and healthy and not reset and not forced_decel and not planner.fcw and not reasons
      and not model.action.shouldStop and evidence is not None and evidence.unknown is None
      and raw_kind in (MotionReferenceKind.FORWARD, MotionReferenceKind.MOVING)
      and not state['radarState'].leadOne.present and not state['radarState'].leadTwo.present
      and math.isfinite(output) and output >= 0.0
      and consumed_forward_reference(control, controls, control_ns=control_ns, controls_ns=controls_ns,
                                      source_ns=previous_source_ns, producer_ns=previous_producer_ns,
                                      instance_ns=self.producer_instance_ns, now_ns=now_ns)
      and (raw_kind == MotionReferenceKind.FORWARD or self._forward_execution_instance_ns == self.producer_instance_ns)
    )
    # MOVING may continue this producer's confirmed FORWARD execution, but it
    # cannot establish it. Every failed guard and every invalidate clear it.
    self._forward_execution_instance_ns = self.producer_instance_ns if executed_forward else 0
    if low_speed_preference_stop and not executed_forward:
      reasons |= StopReason.MPC
    model_allows_external_departure = (evidence is None or evidence.allows_external_departure
                                      or stationary_lead_departure_compatible(
                                        evidence, actual_standstill=bool(car_state.standstill), measured_speed=float(car_state.vEgo)))
    # P's ordinary action can request departure without raw speed growth.
    # Certify that request only after every current target cap and raw veto;
    # the owner still controls HOLD, freshness and the actual brake release.
    # These arrays were projected from this solve above and are the arrays
    # publish() sends. The helper matches their wire precision and Controls'
    # CP actuator horizon; its inverse is only independent departure strength.
    departure_accel = get_departure_accel_from_plan(
      planner.v_desired_trajectory, planner.a_desired_trajectory, CONTROL_N_T_IDX, delay + DT_MDL, self.constraint_accel)
    ordinary_absolute_departure = (
      ordinary_model_motion and raw_kind == MotionReferenceKind.MOVING
      and evidence.allows_external_departure and not model.action.shouldStop
      and model.action.desiredAcceleration >= STOP_RELEASE_ACCEL and output > 0.0
      and (output >= STOP_RELEASE_ACCEL or departure_accel is not None and departure_accel >= STOP_RELEASE_ACCEL)
    )
    model_allows_trajectory_departure = evidence is None or evidence.allows_departure or ordinary_absolute_departure
    no_model_risk = (model_allows_external_departure and not model.action.shouldStop
                     and model.action.desiredAcceleration >= CREEP_FOLLOW_MIN_ACCEL
                     and model.action.desiredAcceleration >= self.reference_accel - CREEP_FOLLOW_MODEL_MARGIN)
    # This confirms continued rolling; it grants neither release nor extra
    # throttle. Keep the original reaction/stopping reserve and lead guards,
    # while aligning its base with the compact goal and the selected native
    # standstill risk boundaries. Otherwise the old 5 m base forces a stop when
    # a correctly followed compact lead crosses the old confirmation budget.
    present_risk = max((reference.lead_risk_stop_distances[i]
                        for i, lead in enumerate((state['radarState'].leadOne, state['radarState'].leadTwo))
                        if lead.present), default=STOP_DISTANCE)
    moving_gap_base = max(LEAD_DANGER_FACTOR * present_risk,
                          min(CREEP_FOLLOW_MIN_GAP, reference.comfort_stop_distance))
    moving_lead = bool(qualified and not admission_reasons and no_model_risk and output >= CREEP_FOLLOW_MIN_ACCEL
                       and cruise_speed >= CREEP_SPEED and has_safe_moving_lead(
                         state['radarState'], speed, delay, minimum_gap_base=moving_gap_base, now_ns=now_ns))
    trajectory_departure = (active and healthy and not forced_decel and not planner.fcw and not admission_reasons
                            and forward_reference and model_allows_trajectory_departure)
    # Independent lead departure may be confirmed while the MPC reference is
    # still unwinding braking history. The owner keeps the held brake until
    # this evidence is confirmed and the actual target is positive.
    unbound_physical = self.radar_held_identity is None and lead_identity(radar.leadOne) is not None
    departure_stage, rejected_identity = None, None
    if not qualified:
      departure_stage = 'planning_qualification'
    elif admission_reasons:
      departure_stage = 'raw_stop_veto'
    elif not model_allows_external_departure:
      departure_stage = 'model_compatibility'
    elif model.action.shouldStop:
      departure_stage = 'model_action_stop'
    elif not model.action.desiredAcceleration >= CREEP_FOLLOW_MIN_ACCEL:
      departure_stage = 'model_action_acceleration'
    elif not model.action.desiredAcceleration >= self.reference_accel - CREEP_FOLLOW_MODEL_MARGIN:
      departure_stage = 'model_action_margin'
    elif cruise_speed < CREEP_SPEED:
      departure_stage = 'cruise_speed'
    elif unbound_physical and output <= 0.0:
      departure_stage = 'nonpositive_target'
    else:
      if unbound_physical:
        departure_stage = self.unbound_radar_departure_rejection(state, now_ns=now_ns)
      if departure_stage is None:
        rejection = departing_lead_rejection(radar, speed, delay, now_ns=now_ns, held_identity=self.radar_held_identity,
                                             allow_unbound_physical=unbound_physical)
        if rejection is not None:
          departure_stage, rejected_identity = rejection
    lead_departure = departure_stage is None
    observe(planner.observation, "departure", planner.observation_attempt, forward_reference, planner.mpc.v_solution,
            trajectory_departure, lead_departure, qualified, no_model_risk)
    release_allowed = bool(trajectory_departure or lead_departure)
    if self.radar_held_identity is not None and not lead_departure:
      # A new track, an expired point or a disappearing packet cannot release
      # a stop held for another physical vehicle, even if the model accelerates.
      release_allowed = False
    if self.turn_lead_guard.lost or turn_ceiling is not None and turn_ceiling <= 0.0:
      # Lost-target speed support and a zero execution ceiling cannot certify
      # departure, even when another candidate wins with a negative target.
      # Preserve the raw motion description while withdrawing movement permission.
      moving_lead = release_allowed = False
      departure_stage = 'turn_lead_lost' if self.turn_lead_guard.lost else 'turn_acceleration_ceiling'
    self.log_departure_blocked(
      state, now_ns=now_ns, active=active, release_allowed=release_allowed, output=output,
      stage=departure_stage, rejected_identity=rejected_identity, admission_reasons=admission_reasons, no_model_risk=no_model_risk,
    )
    response_kind = MotionResponseKind.NONE
    # An ACC cruise cap still describes an absolute acceleration target. Track
    # that final target with nonpositive correction instead of erasing braking
    # effort whenever the native trajectory crosses its own acceleration cap.
    # Independent model constraints and policy overrides retain their guards.
    ordinary_acc_cap = (not self.e2e_allowed and evidence is None
                        and self.constraint_source == LongitudinalPlanSource.cruise
                        and self.reference_accel > maximum_accel
                        and output == self.constraint_accel)
    # Retain feedback qualification outside the separately certified request
    # domain below. A binding independent maximum still retains the
    # nonpositive-correction contract.
    cruise_response_bound = (not self.e2e_allowed and evidence is None
                             and self.constraint_source == LongitudinalPlanSource.cruise
                             and self.constraint_accel == cruise_ceiling < maximum_accel
                             and cruise_ceiling < self.reference_accel <= maximum_accel and output == cruise_ceiling)
    # Only the actual strictly winning cruise command carries a request
    # contract. This declaration does not change the numeric arbitration.
    # Equal independent constraints keep their existing response semantics;
    # raw == mapped (notably -0.19) is still coast when this candidate wins.
    cruise_selected = (not reasons and cruise_command < min(self.reference_accel, maximum_accel)
                       and self.constraint_source == LongitudinalPlanSource.cruise
                       and self.constraint_accel == cruise_command and output == cruise_command)
    # Ordinary model motion retains its raw facts while the winning cruise
    # governor owns response shaping. An equal action cap is independent
    # ownership even though the numeric arbitration labels that tie cruise.
    external_cruise = (native_cruise_external
                       and ((not self.e2e_allowed and evidence is None)
                            or ordinary_model_motion and raw_kind in (MotionReferenceKind.MOVING, MotionReferenceKind.FORWARD)
                            and cruise_command < model.action.desiredAcceleration))
    coast_selected = coast_applicable and cruise_selected
    turn_response = (self.turn_accel_limited and active and healthy and not reset and not forced_decel and not reasons
                     and output >= 0.0 and raw_kind in (MotionReferenceKind.UNKNOWN, MotionReferenceKind.MOVING,
                                                        MotionReferenceKind.FORWARD))
    if turn_response:
      # The bounded acceleration target cannot carry MODEL's positive-I
      # permission, including the zero target and gradual recovery. Stronger
      # braking and raw stop semantics keep their original contracts below.
      response_kind = MotionResponseKind.ABSOLUTE_ACC
    elif coast_selected:
      response_kind = MotionResponseKind.CRUISE_COAST
      # A negative coast request is execution shaping, never a departure
      # certificate inherited from the unconstrained native trajectory.
      moving_lead = release_allowed = False
    elif external_cruise and not planner.fcw and cruise_selected:
      # CruiseResponse already forms P's execution request from speed and
      # history. Keep independently qualified departure facts; this domain
      # grants no release and must not add a second acceleration feedback loop.
      response_kind = MotionResponseKind.CRUISE_REQUEST
    elif (ordinary_model_motion and raw_kind in (MotionReferenceKind.MOVING, MotionReferenceKind.FORWARD)
        and active and healthy and not forced_decel and not reasons):
      # This producer installed only absolute lead/cruise targets. An action
      # cap and a factual raw FORWARD label do not change that target domain.
      # Declare it even on admission/reset so cached first-frame plans cannot
      # acquire MODEL positive-I permission merely from a tiny raw slope.
      response_kind = MotionResponseKind.ABSOLUTE_ACC
    elif (active and healthy and not reset and not forced_decel and not reasons
        and (not self.reference_overridden or ordinary_acc_cap or cruise_response_bound)
        and planner.source == LongitudinalPlanSourceSP.cruise):
      lead_sources = (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1)
      lead_index = lead_sources.index(planner.mpc.source) if planner.mpc.source in lead_sources else None
      selected_lead = (state['radarState'].leadOne, state['radarState'].leadTwo)[lead_index] if lead_index is not None else None
      selected_identity = getattr(planner.mpc, 'source_object_identity', None)
      if getattr(state['radarState'], 'fusionActive', False) and selected_identity is not None:
        selected_lead = next((lead for lead in control_leads(state['radarState']) if lead_identity(lead) == selected_identity), None)
        lead_index = None
      if (not self.e2e_allowed and evidence is None
          and (planner.mpc.source == LongitudinalPlanSource.cruise or selected_lead is not None and selected_lead.present)):
        # Ordinary ACC owns an absolute speed/lead target. Missing model
        # semantics alone are not this certificate; policy, health and the
        # final target must all retain their absolute-acceleration meaning.
        # Ordinary ACC retains only braking correction, including for excess
        # response to a positive target. FCW does not revoke this negative I.
        response_kind = MotionResponseKind.ABSOLUTE_ACC
      elif (not self.reference_overridden and selected_lead is not None and output < 0.0
            and get_lead_braking_urgency(speed, output, state['radarState'], delay, lead_index=lead_index, now_ns=now_ns) > 0.0):
        # MPC source names one obstacle, not a causal braking proof. Require
        # that same fresh lead's independently measured closing/gap demand.
        # FCW blocks drive qualification, not this negative response. This
        # deliberately covers only the existing urgency subset.
        response_kind = MotionResponseKind.LEAD_BRAKING
    self.radar_release_identity = self.radar_held_identity if release_allowed and lead_departure else None
    # An object may propose an identity barrier only when it owns the MPC or
    # close-lead stop. The owner can still be braking toward that stop while
    # the delay-compensated planner target is positive; its matching receipt,
    # not this target's sign, proves that the stop was actually adopted.
    # Independent model/cruise stops and mere detections cannot claim it.
    primary = radar.leadOne
    primary_identity = lead_identity(primary)
    independent_stop = reasons & (StopReason.E2E | StopReason.CRUISE | StopReason.FORCE_DECEL)
    if (active and healthy and self.fusion_admitted and not reset and not forced_decel and not independent_stop
        and not model_stop_evidence
        and reasons & (StopReason.MPC | StopReason.CLOSE_LEAD) and not self.reference_overridden
        and primary_identity is not None and primary_identity == planner.mpc.source_object_identity
        and primary_identity in planner.mpc.object_influence_identities
        and qualified_lead(primary, 'stop', 0.98, now_ns=now_ns) and primary.vLead < STOP_SPEED):
      self._radar_stop_candidate = primary_identity, source_epoch_ns, now_ns, self.producer_instance_ns
    planner.motion_intent = MotionIntent(
      mode=int(planner.longitudinal_mode), reasons=int(reasons), moving_lead=moving_lead, release_allowed=release_allowed,
      allow_creep=bool(active and healthy), valid=bool(healthy and car_state.vCruise != V_CRUISE_UNSET),
      reference_kind=int(raw_kind), producer_instance_ns=self.producer_instance_ns, response_kind=int(response_kind),
    )
    planner.output_a_target = output
    planner.output_should_stop = bool(reasons)
    self.reference_initialized = bool(healthy and not reset)
    self.reference_source_epoch_ns = source_epoch_ns if healthy else None
    planner.mpc.mark_reference_history(valid=bool(reference_qualified and self.reference_initialized))
    self.continuation.native_success(
      planner.mpc, state, planner.reference_timing.pending, now_ns=now_ns,
      permitted=bool(continuation_permitted and planner.motion_intent.valid and planner.motion_intent.ready
                     and not reasons and not planner.fcw), scope=continuation_scope, applied_accel_ceiling=output,
    )
    planner._observe(outcome="completed" if healthy else "completed_unhealthy_inputs")
    if planner.trace is not None:
      observe_trace(planner.trace, "finish_unified", planner, self.e2e_allowed)

  def publish(self, message, *, now_ns=None):
    intent = self.planner.motion_intent
    message.motionIntent.version = intent.version
    message.motionIntent.mode = intent.mode
    message.motionIntent.reasons = int(intent.reasons)
    message.motionIntent.movingLead = intent.moving_lead
    message.motionIntent.releaseAllowed = intent.release_allowed
    message.motionIntent.allowCreep = intent.allow_creep
    message.motionIntent.valid = intent.valid
    message.motionIntent.ready = intent.ready
    message.motionIntent.brakingValid = intent.braking_accel is not None
    message.motionIntent.brakingAccel = float(intent.braking_accel) if intent.braking_accel is not None else 0.0
    message.motionIntent.referenceKind = int(intent.reference_kind)
    message.motionIntent.restGoalRejected = intent.rest_goal_rejected
    message.motionIntent.producerInstanceMonoTime = intent.producer_instance_ns
    message.motionIntent.responseKind = int(intent.response_kind)
    message.reference.accel = self.reference_accel
    message.reference.constraintAccel = self.constraint_accel
    message.reference.overridden = self.reference_overridden
    message.reference.source = self.reference_source
    message.reference.constraintSource = self.constraint_source
    continuation = self.continuation
    offer = continuation.offer
    message.continuation.version = CONTINUATION_VERSION
    if offer is not None:
      message.continuation = offer.to_dict()
      timing = self.planner.reference_timing.pending
      message.continuation.valid = bool(
        timing is not None and valid_timestamp(now_ns)
        and timing.producer_ns <= now_ns < offer.valid_until_ns)
    message.continuation.reason = (continuation.reason if offer is None or message.continuation.valid
                                   else 'expired_before_publication')
    message.continuation.independentBrakingValid = continuation.independent_braking_accel is not None
    message.continuation.independentBrakingAccel = (float(continuation.independent_braking_accel)
                                                   if continuation.independent_braking_accel is not None else 0.0)
    message.continuation.tailExtensionSeconds = float(continuation.tail_extension_s)
    message.continuation.validationTime = float(continuation.validation_time)
