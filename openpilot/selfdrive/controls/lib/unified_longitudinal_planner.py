import math

import numpy as np

from opendbc.car.honda.values import CarControllerParams
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.longitudinal_logging import observe
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, CREEP_ACCEL, CREEP_SPEED, STOP_SPEED, should_stop, get_accel_from_plan
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState, MPC_FAILURE_ACCEL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import MpcReference, T_IDXS
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  CONTROL_N_T_IDX, ALLOW_THROTTLE_THRESHOLD, MIN_ALLOW_THROTTLE_SPEED,
  LEAD_BRAKING_MIN_CHANGE_SCALE, CREEP_FOLLOW_MIN_ACCEL, CREEP_FOLLOW_MODEL_MARGIN,
  CREEP_FOLLOW_MIN_GAP, CREEP_FOLLOW_MIN_LEAD_SPEED, CREEP_FOLLOW_MAX_CLOSING_SPEED,
  CREEP_FINISH_T, COMFORT_BRAKE, LongitudinalPlanSource, LongitudinalPlanSourceSP,
  _A_TOTAL_MAX_V, _A_TOTAL_MAX_BP, get_max_accel, get_coast_accel,
  constrain_braking_speed, constrain_lead_braking_speed, get_lead_braking_urgency,
  has_safe_moving_lead, should_stop_for_close_lead,
)
from openpilot.selfdrive.controls.lib.motion_supervisor import CREEP_INPUT_SERVICES, MOTION_INTENT_VERSION, CreepSample, MotionIntent, StopReason
from openpilot.selfdrive.controls.lib.longitudinal_observation import observe_trace
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP


INPUT_SERVICES = ['carState', 'radarState', 'modelV2', 'controlsState', 'carControl', 'selfdriveState', 'vehicleParameters']


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


def has_departing_lead(radar_state, speed, delay):
  if not (math.isfinite(speed) and 0.0 <= speed < STOP_SPEED and math.isfinite(delay)
          and 0.0 <= delay <= 1.0 and radar_state.leadOne.present):
    return False
  minimum_gap = CREEP_FOLLOW_MIN_GAP + speed * (delay + DT_MDL) + speed ** 2 / (2.0 * COMFORT_BRAKE)
  for lead in (radar_state.leadOne, radar_state.leadTwo):
    if not lead.present:
      continue
    values = (lead.dRel, lead.vRel, lead.vLead, lead.aLeadK, lead.modelProb, lead.yRel)
    if not all(math.isfinite(value) for value in values):
      return False
    if (lead.dRel < minimum_gap or not 0.95 <= lead.modelProb <= 1.0
        or abs(lead.yRel) > 1.8 or lead.vLead < CREEP_FOLLOW_MIN_LEAD_SPEED
        or lead.vRel < -CREEP_FOLLOW_MAX_CLOSING_SPEED
        or lead.vLead + min(lead.aLeadK, 0.0) * CREEP_FINISH_T < STOP_SPEED
        or abs(lead.vLead - speed - lead.vRel) > CREEP_FOLLOW_MAX_CLOSING_SPEED):
      return False
  return True


class UnifiedLongitudinalPlanner:
  def __init__(self, planner):
    self.planner = planner
    self.reference_accel = 0.0
    self.constraint_accel = ACCEL_MAX
    self.constraint_source = LongitudinalPlanSource.cruise
    self.reference_source = LongitudinalPlanSource.cruise
    self.reference_overridden = False
    self.e2e_allowed = False
    self.reference_initialized = False
    self.cruise_initialized = False
    self.cold_standby_started = False
    self.standby = False
    planner.motion_intent = MotionIntent(mode=int(planner.longitudinal_mode), valid=False)

  def invalidate(self, speed, reason):
    planner = self.planner
    self.reference_initialized = False
    planner.mpc.invalidate_reference_history()
    self.standby = False
    planner.motion_intent = MotionIntent(mode=int(planner.longitudinal_mode), valid=False)
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

  def fail_mpc(self, state, speed, previous_request):
    planner = self.planner
    self.invalidate(speed, "mpc_" + planner.mpc.failure_reason)
    if planner.mpc.solution_status == 0 or not state.all_checks(service_list=INPUT_SERVICES):
      return
    previous = float(previous_request) if math.isfinite(previous_request) else MPC_FAILURE_ACCEL
    request = min(previous, MPC_FAILURE_ACCEL)
    self.e2e_allowed = planner.is_e2e(state)
    model_request = state['modelV2'].action.desiredAcceleration
    if self.e2e_allowed and math.isfinite(model_request):
      request = min(request, model_request)
    request = float(np.clip(request, ACCEL_MIN, ACCEL_MAX))
    planner.motion_intent = MotionIntent(mode=int(planner.longitudinal_mode), allow_creep=False,
                                         valid=False, braking_accel=request)
    planner.output_a_target = request
    self.constraint_accel = request
    if self.e2e_allowed and model_request <= min(previous, MPC_FAILURE_ACCEL):
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
                                             allow_creep=False, valid=False, ready=False)
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

  def update(self, state, *, now_ns):
    planner = self.planner
    previous_reference_valid = planner.mpc.prepare_reference_history(now_ns, valid=self.reference_initialized)
    prepared = self.prepare(state)
    if prepared is None:
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
    # plannerd can restart while controlsd is still active. Its constructor's
    # default zero filter is not a motion reference for an already moving car.
    if not self.reference_initialized or reset or car_state.standstill and speed == 0.0:
      planner.v_desired_filter.x = speed
    initial_speed = max(0.0, float(planner.v_desired_filter.update(speed)))
    if active and healthy and not reset:
      initial_speed = constrain_braking_speed(initial_speed, speed, planner.output_a_target)
      initial_speed = constrain_lead_braking_speed(initial_speed, speed, planner.output_a_target, state['radarState'])
    planner.v_desired_filter.x = initial_speed
    initial_accel = float(np.clip(measured_accel, ACCEL_MIN, ACCEL_MAX))
    if initial_speed == 0.0:
      initial_accel = max(0.0, initial_accel)

    coast_accel = ACCEL_MAX
    if len(state['carControl'].orientationNED) == 3:
      pitch = state['carControl'].orientationNED[1]
      if not math.isfinite(pitch):
        self.invalidate(speed, "nonfinite_pitch")
        return
      coast_accel = float(get_coast_accel(pitch - planner.pitch_bias.value))
    planner._observe(coastAccelKnown=True, coastAccel=float(coast_accel))
    probabilities = model.meta.disengagePredictions.gasPressProbs
    throttle_probability = probabilities[1] if len(probabilities) > 1 else 1.0
    if not math.isfinite(throttle_probability):
      self.invalidate(speed, "nonfinite_throttle_probability")
      return
    planner.allow_throttle = throttle_probability > ALLOW_THROTTLE_THRESHOLD or speed <= MIN_ALLOW_THROTTLE_SPEED
    cruise_speed = min(car_state.vCruise, V_CRUISE_MAX) * CV.KPH_TO_MS
    forced_decel = state['controlsState'].forceDecel
    if forced_decel:
      cruise_speed = 0.0
    cruise_speed, policy_accel = LongitudinalPlannerSP.update_targets(planner, state, initial_speed, initial_accel, cruise_speed)
    planner._observe(policyTargetKnown=True, policySpeed=float(cruise_speed), policyAccel=float(policy_accel))
    if not math.isfinite(cruise_speed) or not math.isfinite(policy_accel):
      self.invalidate(speed, "nonfinite_policy_target")
      return
    cruise_speed = max(0.0, cruise_speed)
    if self.e2e_allowed and model.action.shouldStop:
      cruise_speed = 0.0
    steering = car_state.steeringAngleDeg - state['vehicleParameters'].angleOffsetDeg
    maximum_accel = cruise_accel_limit(planner.CP, speed, steering, self.e2e_allowed, coast_accel, planner.allow_throttle)
    model_position = model_velocity = model_acceleration = None
    if self.e2e_allowed:
      fields = (model.position.x, model.velocity.x, model.acceleration.x)
      if any(len(values) != len(ModelConstants.T_IDXS) or not np.all(np.isfinite(values)) for values in fields):
        self.invalidate(speed, "invalid_e2e_reference")
        return
      model_position, model_velocity, model_acceleration = model_reference_from_velocity(model.velocity.x)
    reference = MpcReference(
      cruise_speed=cruise_speed, accel_min=ACCEL_MIN, accel_max=maximum_accel,
      model_position=model_position, model_velocity=model_velocity, model_acceleration=model_acceleration,
    )
    urgency = 0.0
    if active and healthy and not reset and not forced_decel and planner.source == LongitudinalPlanSourceSP.cruise:
      urgency = get_lead_braking_urgency(speed, planner.output_a_target, state['radarState'], delay)
    # Only smooth against a qualified native reference aligned to this epoch.
    # Neither reset zeros nor measured acceleration are a previous request.
    reference_qualified = healthy and not (reset or car_state.standstill)
    if not reference_qualified:
      planner.mpc.invalidate_reference_history()
    planner.mpc.set_weights(previous_reference_valid and reference_qualified, personality=state['selfdriveState'].personality,
                            a_change_scale=1.0 - (1.0 - LEAD_BRAKING_MIN_CHANGE_SCALE) * urgency)
    planner.mpc.set_cur_state(initial_speed, initial_accel)
    if planner.trace is not None:
      observe_trace(planner.trace, "begin", planner, state, reset, cruise_speed, coast_accel)
    observe(planner.observation, "mpc_started", planner.observation_attempt, planner.mpc)
    planner.mpc.update(state['radarState'], personality=state['selfdriveState'].personality, reference=reference)
    if planner.mpc.solution_status != 0:
      self.fail_mpc(state, speed, planner.output_a_target)
      return
    planner.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS, planner.mpc.v_solution)
    planner.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS, planner.mpc.a_solution)
    planner.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS[:-1], planner.mpc.j_solution)
    planner._observe(actionHorizonKnown=True, actionHorizon=float(delay + DT_MDL))
    self.reference_accel = float(get_accel_from_plan(
      planner.v_desired_trajectory, planner.a_desired_trajectory, CONTROL_N_T_IDX, action_t=delay + DT_MDL,
    ))
    planner._observe(sampledAccelKnown=True, sampledAccel=self.reference_accel)
    if not math.isfinite(self.reference_accel):
      self.invalidate(speed, "nonfinite_sampled_accel")
      return
    self.constraint_accel = maximum_accel
    self.constraint_source = LongitudinalPlanSource.cruise
    self.reference_source = planner.mpc.source
    if cruise_speed <= STOP_SPEED:
      self.constraint_accel = min(self.constraint_accel, CarControllerParams.BOSCH_GAS_LOOKUP_BP[0])
    if self.e2e_allowed and model.action.desiredAcceleration < self.constraint_accel:
      self.constraint_accel = float(model.action.desiredAcceleration)
      self.constraint_source = LongitudinalPlanSource.e2e
    planner._observe(constraintKnown=True, constraintAccel=float(self.constraint_accel))
    output = float(np.clip(min(self.reference_accel, self.constraint_accel), ACCEL_MIN, ACCEL_MAX))
    self.reference_overridden = output != self.reference_accel
    if output < self.reference_accel:
      planner.mpc.source = self.constraint_source
    planner.fcw = planner.mpc.crash_cnt > 2 and not car_state.standstill

    forward_reference = reference_requests_departure(planner.mpc.v_solution, planner.mpc.a_solution, planner.mpc.j_solution)
    reasons = StopReason(0)
    if (planner.mpc.v_solution[-1] < STOP_SPEED
        and (should_stop(speed, self.reference_accel) or speed < STOP_SPEED and not forward_reference)):
      reasons |= StopReason.MPC
    if self.e2e_allowed and model.action.shouldStop:
      reasons |= StopReason.E2E
    if cruise_speed <= STOP_SPEED:
      reasons |= StopReason.FORCE_DECEL if forced_decel else StopReason.CRUISE
    qualified = active and healthy and not forced_decel and not planner.fcw and planner.source == LongitudinalPlanSourceSP.cruise
    if qualified and should_stop_for_close_lead(speed, output, state['radarState'], delay):
      reasons |= StopReason.CLOSE_LEAD
    no_model_risk = (not model.action.shouldStop and model.action.desiredAcceleration >= CREEP_FOLLOW_MIN_ACCEL
                     and model.action.desiredAcceleration >= self.reference_accel - CREEP_FOLLOW_MODEL_MARGIN)
    moving_lead = bool(qualified and not reasons and no_model_risk and output >= CREEP_FOLLOW_MIN_ACCEL
                       and cruise_speed >= CREEP_SPEED and has_safe_moving_lead(state['radarState'], speed, delay))
    trajectory_departure = (active and healthy and not forced_decel and not planner.fcw and not reasons
                            and forward_reference)
    lead_departure = (qualified and not reasons and no_model_risk and output > 0.0
                      and cruise_speed >= CREEP_SPEED and has_departing_lead(state['radarState'], speed, delay))
    observe(planner.observation, "departure", planner.observation_attempt, forward_reference, planner.mpc.v_solution,
            trajectory_departure, lead_departure, qualified, no_model_risk)
    release_allowed = bool(trajectory_departure or lead_departure)
    planner.motion_intent = MotionIntent(
      mode=int(planner.longitudinal_mode), reasons=int(reasons), moving_lead=moving_lead, release_allowed=release_allowed,
      allow_creep=bool(active and healthy), valid=bool(healthy and car_state.vCruise != V_CRUISE_UNSET),
    )
    planner.output_a_target = output
    planner.output_should_stop = bool(reasons)
    planner.v_desired_filter.x = max(0.0, initial_speed + planner.dt * measured_accel)
    self.reference_initialized = bool(healthy and not reset)
    planner.mpc.mark_reference_history(valid=bool(reference_qualified and self.reference_initialized))
    planner._observe(outcome="completed" if healthy else "completed_unhealthy_inputs")
    if planner.trace is not None:
      observe_trace(planner.trace, "finish_unified", planner, self.e2e_allowed)

  def publish(self, message):
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
    message.motionIntent.creepEvidenceValid = intent.creep_sample is not None
    if intent.creep_sample is not None:
      message.motionIntent.creepSourceMonoTimes = list(intent.creep_sample.source_times_ns)
      message.motionIntent.creepMeasuredSpeed = float(intent.creep_sample.measured_speed)
      message.motionIntent.creepReset = intent.creep_sample.reset
    message.reference.accel = self.reference_accel
    message.reference.constraintAccel = self.constraint_accel
    message.reference.overridden = self.reference_overridden
    message.reference.source = self.reference_source
    message.reference.constraintSource = self.constraint_source


class SupervisedLongitudinalPlanner(UnifiedLongitudinalPlanner):
  """Publish raw motion evidence without replacing the legacy reference calculation."""

  def finish_reference(self, state, candidates, cruise_speed, reset):
    planner = self.planner
    car_state = state['carState']
    model = state['modelV2']
    speed = max(0.0, float(car_state.vEgo))
    delay = planner.CP.longitudinalActuatorDelay
    self.reference_accel = float(candidates[0][0])
    self.reference_source = candidates[0][1]
    constraint = min(candidates[1:], key=lambda candidate: candidate[0])
    self.constraint_accel, self.constraint_source = float(constraint[0]), constraint[1]
    self.reference_overridden = bool(planner.output_a_target != self.reference_accel)
    self.e2e_allowed = planner.is_e2e(state)
    healthy = state.all_checks(service_list=INPUT_SERVICES)
    active = (state['selfdriveState'].enabled and state['carControl'].enabled and state['carControl'].longActive
              and not car_state.gasPressed and not car_state.brakePressed and not state['carControl'].cruiseControl.override)
    forced_decel = state['controlsState'].forceDecel
    forward_reference = reference_requests_departure(planner.mpc.v_solution, planner.mpc.a_solution, planner.mpc.j_solution)
    reasons = StopReason.NONE
    for candidate, reason in zip(candidates, (StopReason.MPC, StopReason.CRUISE, StopReason.E2E), strict=False):
      if candidate[2]:
        reasons |= reason
    if speed < STOP_SPEED and planner.mpc.v_solution[-1] < STOP_SPEED and not forward_reference:
      reasons |= StopReason.MPC
    if cruise_speed <= STOP_SPEED:
      reasons |= StopReason.FORCE_DECEL if forced_decel else StopReason.CRUISE
    qualified = (active and healthy and not reset and not forced_decel and not planner.fcw
                 and planner.source == LongitudinalPlanSourceSP.cruise)
    if qualified and should_stop_for_close_lead(speed, planner.output_a_target, state['radarState'], delay):
      reasons |= StopReason.CLOSE_LEAD
    no_model_risk = (not model.action.shouldStop and model.action.desiredAcceleration >= CREEP_FOLLOW_MIN_ACCEL
                     and model.action.desiredAcceleration >= self.reference_accel - CREEP_FOLLOW_MODEL_MARGIN)
    moving_lead = bool(qualified and not reasons and no_model_risk and not car_state.standstill
                       and state['controlsState'].longControlState in (LongCtrlState.off, LongCtrlState.pid)
                       and planner.output_a_target >= CREEP_FOLLOW_MIN_ACCEL and self.reference_accel < CREEP_ACCEL
                       and cruise_speed >= CREEP_SPEED and has_safe_moving_lead(state['radarState'], speed, delay))
    trajectory_departure = bool(active and healthy and not forced_decel and not planner.fcw and not reasons and forward_reference)
    lead_departure = bool(qualified and not reasons and no_model_risk and planner.output_a_target > 0.0
                          and cruise_speed >= CREEP_SPEED and has_departing_lead(state['radarState'], speed, delay))
    observe(planner.observation, "departure", planner.observation_attempt, forward_reference, planner.mpc.v_solution,
            trajectory_departure, lead_departure, qualified, no_model_risk)
    # Publish this frame's measured sources only. The consumer owns elapsed
    # qualification, bounded off-sample compatibility and the stopping deadline.
    source_times = tuple(getattr(state, 'logMonoTime', {}).get(service, 0) for service in CREEP_INPUT_SERVICES)
    creep_sample = CreepSample(source_times, float(car_state.vEgo), state['controlsState'].longControlState == LongCtrlState.off)
    planner.motion_intent = MotionIntent(
      mode=int(planner.longitudinal_mode), reasons=int(reasons), moving_lead=moving_lead,
      release_allowed=trajectory_departure or lead_departure, allow_creep=bool(active and healthy),
      valid=bool(healthy and car_state.vCruise != V_CRUISE_UNSET and creep_sample.is_valid()),
      creep_sample=creep_sample if creep_sample.is_valid() else None,
    )
    planner.output_should_stop = bool(reasons)
    planner._observe(outcome="completed_supervised" if healthy else "completed_unhealthy_inputs",
                     constraintKnown=True, constraintAccel=self.constraint_accel)
