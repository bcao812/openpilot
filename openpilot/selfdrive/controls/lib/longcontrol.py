import math
import time

import numpy as np
from opendbc.car.structs import car
from opendbc.car.honda.values import CAR, CarControllerParams
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CLOSE_LEAD_STOP_SPEED, CONTROL_N, STOP_SPEED
from openpilot.common.pid import PIDController
from openpilot.selfdrive.controls.lib.launch_response import LaunchResponse, ResponseDomain
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode, resolve_longitudinal_mode
from openpilot.selfdrive.controls.lib.motion_supervisor import (DEGRADED_ACCEL, MAX_CONTROL_GAP_NS, MotionPhase, MotionReferenceKind, MotionResponseKind,
                                                               MotionSupervisor, StopReason, ROLLING_STOP_ACCEL,
                                                               STARTING_ACCEL_RATE, STOPPING_DECEL_RATE)
from openpilot.selfdrive.modeld.constants import ModelConstants

LongCtrlState = car.CarControl.Actuators.LongControlState
MPC_FAILURE_ACCEL = -0.5
# The frozen source-epoch replay consumes this public grid for plan transport.
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

def long_control_state_trans(CP_SP, active, long_control_state,
                             should_stop, brake_pressed, cruise_standstill):
  # Gas Interceptor
  cruise_standstill = cruise_standstill and not CP_SP.enableGasInterceptor

  starting_condition = (not should_stop and
                        not cruise_standstill and
                        not brake_pressed)

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if not starting_condition:
        long_control_state = LongCtrlState.stopping
      else:
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition:
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.pid:
      if should_stop:
        long_control_state = LongCtrlState.stopping

  return long_control_state

class LongControl:
  def __init__(self, CP, CP_SP):
    self.CP = CP
    self.CP_SP = CP_SP
    self.long_control_state = LongCtrlState.off
    self.pid = PIDController(0.0, (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                             rate=1 / DT_CTRL)
    self.last_output_accel = 0.0
    self.starting = False
    self.longitudinal_mode = resolve_longitudinal_mode(CP, CP_SP)
    self.unified = self.longitudinal_mode == LongitudinalMode.UNIFIED
    self.motion_supervisor = MotionSupervisor() if self.unified else None
    self.resolved_should_stop = False
    # The host Bosch CAN encoder has no acceleration-error integrator. Do not
    # duplicate a tuned PID or the separately installed response experiment.
    response_feedback = (self.longitudinal_mode == LongitudinalMode.UNIFIED and CP.carFingerprint == CAR.HONDA_ACCORD
                         and CP.openpilotLongitudinalControl and not CP_SP.enableGasInterceptor
                         and all(gain == 0.0 for gain in CP.longitudinalTuning.kiV))
    self.launch_response = LaunchResponse(self.pid, max_gap_ns=MAX_CONTROL_GAP_NS) if response_feedback else None

  @property
  def motion_phase(self):
    return self.motion_supervisor.phase if self.motion_supervisor is not None else MotionPhase.OFF

  @property
  def motion_reasons(self):
    return self.motion_supervisor.reasons if self.motion_supervisor is not None else StopReason.NONE

  @property
  def motion_degraded(self):
    return self.motion_supervisor.degraded if self.motion_supervisor is not None else False

  @property
  def motion_waiting_for_fresh_input(self):
    return self.motion_supervisor.waiting_for_fresh_input if self.motion_supervisor is not None else False

  @property
  def motion_input_constrained(self):
    return self.motion_degraded or self.motion_waiting_for_fresh_input

  @property
  def can_standby(self):
    return self.unified and self.motion_supervisor is not None and self.motion_supervisor.can_standby

  @property
  def motion_owner_established(self):
    return self.motion_supervisor is not None and self.motion_supervisor.motion_owner_established

  def reset(self):
    self.pid.reset()
    if self.launch_response is not None:
      self.launch_response.reset()

  def update(self, active, CS, a_target, should_stop, accel_limits, *,
             intent=None, plan_time_ns=None, model_time_ns=None, now_ns=None, plan_valid=True, solver_status=0, braking_plan_valid=False,
             standby_input=False):
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    if self.unified:
      return self._update_unified(active, CS, a_target, accel_limits, intent=intent, plan_time_ns=plan_time_ns,
                                  model_time_ns=model_time_ns, now_ns=now_ns, plan_valid=plan_valid and solver_status == 0,
                                  braking_plan_valid=braking_plan_valid is True, solver_status=solver_status, standby_input=standby_input)
    self.resolved_should_stop = bool(should_stop)
    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    self.long_control_state = long_control_state_trans(self.CP_SP, active, self.long_control_state,
                                                       should_stop, CS.brakePressed,
                                                       CS.cruiseState.standstill)
    if self.long_control_state == LongCtrlState.off:
      self.reset()
      output_accel = 0.
      self.starting = False

    elif self.long_control_state == LongCtrlState.stopping:
      # The PID is abandoned here: the command walks down open loop and stops reacting to the car
      # at all. Every bit of it below what the car needs is felt, and the car is still rolling --
      # arriving at stopAccel = -2.0 at 0.4 m/s is the jolt at the end of a stop. Hold the ramp at
      # ROLLING_STOP_ACCEL until the wheels stop, then let it go on to stopAccel, where it is only
      # holding pressure against a car that is already still.
      #
      # The ramp inherits last_output_accel, and an MPC plan converges asymptotically, so it starts
      # wherever the PID happened to be -- measured over 33 logged stops, a median of -0.33 m/s^2
      # at 0.7 m/s. At 1.0 m/s^2/s that is a full second before the command carries any authority,
      # and on 8 of 21 stops the car reached standstill before the ramp ever passed -1.4. The car
      # spends that second under-braked below its own creep torque, which is the crawl; when the
      # brake finally crosses that torque it does so all at once, which is the jolt. Walking down
      # twice as fast is the same ramp arriving while it can still do something.
      stop_accel = self.CP.stopAccel if CS.standstill else max(self.CP.stopAccel, ROLLING_STOP_ACCEL)
      output_accel = self.last_output_accel
      if output_accel > stop_accel:
        output_accel = min(output_accel, 0.0)
        output_accel -= STOPPING_DECEL_RATE * DT_CTRL
      self.reset()
      # Whatever the ramp reached is where the next pid frame has to climb out from.
      self.starting = True

    else:  # LongCtrlState.pid
      # Leaving stopping hands the wire a step, not a ramp: last_output_accel is at the bottom of
      # the ramp and the PID's first output is essentially a_target, because the integrator was
      # reset on every stopping frame and k_p is zero. Measured over 52 logged launches that step
      # is a median +2.10 m/s^2 in one 10 ms frame -- 210 m/s^3 -- and Honda Bosch rate-limits
      # neither accel nor gas, so BRAKE_REQUEST clears and GAS_COMMAND rises in the same CAN frame.
      # Bound the climb until the PID catches up. Only the climb: a deeper command is never held
      # back, so this cannot delay braking.
      error = a_target - CS.aEgo
      limit = self.last_output_accel + STARTING_ACCEL_RATE * DT_CTRL
      output_accel = self.pid.update(error, speed=CS.vEgo,
                                     feedforward=a_target, freeze_integrator=self.starting)
      if self.starting:
        if output_accel > limit:
          output_accel = limit
        else:
          self.starting = False

    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    return self.last_output_accel

  def _update_unified(self, active, CS, a_target, accel_limits, *, intent, plan_time_ns, model_time_ns, now_ns,
                      plan_valid, braking_plan_valid, solver_status, standby_input=False):
    active = active and not CS.gasPressed and not CS.brakePressed
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    if not active:
      # Observe identities for rearm without bypassing the immediate
      # manual-override OFF/zero-output return below.
      self.motion_supervisor.update(False, CS.vEgo, CS.aEgo, CS.standstill, a_target, intent,
                                    plan_time_ns=plan_time_ns, model_time_ns=model_time_ns, now_ns=now_ns,
                                    plan_valid=plan_valid, solver_status=solver_status, standby_input=standby_input)
      self.resolved_should_stop = False
      self.reset()
      self.long_control_state = LongCtrlState.off
      self.last_output_accel = 0.0
      self.starting = False
      return 0.0

    limits_valid = (accel_limits is not None and len(accel_limits) == 2 and all(math.isfinite(value) for value in accel_limits)
                    and accel_limits[0] <= accel_limits[1] and accel_limits[0] <= DEGRADED_ACCEL
                    and accel_limits[0] >= CarControllerParams.BOSCH_ACCEL_MIN
                    and accel_limits[1] <= CarControllerParams.BOSCH_ACCEL_MAX)
    if not limits_valid:
      accel_limits = (CarControllerParams.BOSCH_ACCEL_MIN, DEGRADED_ACCEL)
    last_output_valid = math.isfinite(self.last_output_accel)
    if not last_output_valid:
      self.last_output_accel = 0.0
    # Corrupt output history degrades normal control. A separately qualified
    # fresh braking envelope is still usable without trusting that history.
    stop_accel_valid = (math.isfinite(self.CP.stopAccel)
                        and CarControllerParams.BOSCH_ACCEL_MIN <= self.CP.stopAccel <= DEGRADED_ACCEL)
    cold_standstill = self.motion_phase == MotionPhase.OFF and CS.standstill
    self.resolved_should_stop = self.motion_supervisor.update(
      active, CS.vEgo, CS.aEgo, CS.standstill, a_target, intent, plan_time_ns=plan_time_ns,
      model_time_ns=model_time_ns, now_ns=now_ns, plan_valid=plan_valid and limits_valid and last_output_valid and stop_accel_valid,
      braking_plan_valid=braking_plan_valid and limits_valid and stop_accel_valid, solver_status=solver_status)
    if cold_standstill and self.resolved_should_stop:
      hold_accel = self.CP.stopAccel if stop_accel_valid else -2.0
      self.last_output_accel = min(self.last_output_accel, hold_accel)

    if self.motion_input_constrained:
      a_target = min(self.last_output_accel, DEGRADED_ACCEL)
      if self.motion_supervisor.degraded_braking_accel is not None:
        a_target = min(a_target, self.motion_supervisor.degraded_braking_accel)

    self.pid.neg_limit, self.pid.pos_limit = accel_limits
    protocol_hold = (CS.cruiseState.standstill
                     and (self.long_control_state in (LongCtrlState.off, LongCtrlState.stopping)
                          or self.motion_phase in (MotionPhase.HOLD, MotionPhase.RELEASE)))
    stopping = self.resolved_should_stop and (CS.standstill or CS.vEgo < CLOSE_LEAD_STOP_SPEED)
    allocation = getattr(self, '_response_candidate', None)
    response_available = self.launch_response is not None and allocation is None and not self.motion_input_constrained
    # The accepted same-frame intent declares a target domain. UNKNOWN alone
    # grants none; model launch/continuation keeps its separate qualification.
    owner_allows_response = (not self.motion_input_constrained and not self.resolved_should_stop and not protocol_hold and not intent.reasons
                             and self.motion_phase in (MotionPhase.RELEASE, MotionPhase.TRACK)
                             and self.motion_supervisor.rest_braking_accel_limit is None)
    cruise_execution = (owner_allows_response
                        and intent.response_kind in (MotionResponseKind.CRUISE_COAST, MotionResponseKind.CRUISE_REQUEST))
    response_domain = ResponseDomain.NONE
    if response_available and owner_allows_response:
      if intent.response_kind == MotionResponseKind.PURE_CRUISE and self.motion_phase == MotionPhase.TRACK:
        response_domain = ResponseDomain.PURE_CRUISE
      elif intent.response_kind == MotionResponseKind.ABSOLUTE_ACC and self.motion_phase == MotionPhase.TRACK:
        response_domain = ResponseDomain.ABSOLUTE_ACC
      elif (intent.response_kind == MotionResponseKind.LEAD_BRAKING and self.motion_phase == MotionPhase.TRACK
            and a_target < 0.0):
        response_domain = ResponseDomain.LEAD_BRAKING
      elif (intent.response_kind == MotionResponseKind.NONE
            and intent.reference_kind in (MotionReferenceKind.FORWARD, MotionReferenceKind.MOVING)):
        response_domain = ResponseDomain.MODEL
    response_feedback = response_domain != ResponseDomain.NONE
    braking_feedback = (response_available and self.motion_phase == MotionPhase.APPROACH and not stopping and not protocol_hold)
    if self.launch_response is not None and not (response_feedback or braking_feedback):
      self.launch_response.reset()
    if stopping or protocol_hold:
      self.long_control_state = LongCtrlState.stopping
      hold_accel = self.CP.stopAccel if stop_accel_valid else -2.0
      stop_accel = hold_accel if CS.standstill else max(hold_accel, ROLLING_STOP_ACCEL)
      output_accel = self.last_output_accel
      if output_accel > stop_accel:
        output_accel = min(output_accel, 0.0) - STOPPING_DECEL_RATE * DT_CTRL
      output_accel = min(output_accel, a_target, CarControllerParams.BOSCH_GAS_LOOKUP_BP[0])
      self.reset()
    else:
      self.long_control_state = LongCtrlState.pid
      if self.motion_input_constrained:
        self.reset()
        output_accel = a_target
      else:
        if cruise_execution:
          # The accepted cruise request already owns its response shaping.
          # Retire the same PID integral only after the owner accepts
          # TRACK/RELEASE; an unwithdrawn APPROACH keeps its braking history.
          # Also cover configured Ki when LaunchResponse is not installed.
          self.pid.i = 0.0
        if response_feedback and self.motion_phase == MotionPhase.RELEASE:
          # Freeze before PID evaluation, so RELEASE cannot spend an old I.
          self.launch_response.freeze_release(
            intent.producer_instance_ns, now_ns, arm=bool(intent.reference_kind == MotionReferenceKind.FORWARD and intent.release_allowed))
        output_accel = self.pid.update(a_target - CS.aEgo, speed=CS.vEgo, feedforward=a_target,
                                       freeze_integrator=cruise_execution or self.launch_response is not None or self.motion_phase == MotionPhase.RELEASE)
        # The helper owns the existing pid.i, not an extra correction added
        # after a PID which already included I. Preserve unsaturated P+D+F.
        base_output = float(self.pid.p + self.pid.d + self.pid.f)
        if response_feedback and self.motion_phase == MotionPhase.TRACK:
          output_accel = self.launch_response.propose(
            base_output, a_target, CS.aEgo, accel_limits[1], producer_instance_ns=intent.producer_instance_ns, now_ns=now_ns,
            forward=bool(intent.reference_kind == MotionReferenceKind.FORWARD), rolling=bool(not CS.standstill and CS.vEgo >= STOP_SPEED),
            positive_allowed=intent.release_allowed, response_domain=response_domain)
        elif braking_feedback:
          # Do not weaken an accepted braking response just because TRACK
          # entered APPROACH. Only its actual nonpositive I may carry over;
          # HOLD and faults already inherit last_output_accel before reset.
          output_accel = self.launch_response.freeze_braking(
            base_output, accel_limits[1], producer_instance_ns=intent.producer_instance_ns, now_ns=now_ns,
            braking_owned=self.last_output_accel < 0.0)
          if self.launch_response.braking_handoff:
            # A legitimate producer restart can leave the owner in APPROACH.
            # Preserve its actual brake without transferring I to that source.
            output_accel = min(output_accel, self.last_output_accel)
        # Isolated experiment only: one optional allocation stage after the
        # existing owner has accepted TRACK, before the existing upward limit.
        # It receives no authority to release, resume or change the intent.
        if (allocation is not None and self.longitudinal_mode == LongitudinalMode.UNIFIED
            and self.motion_phase == MotionPhase.TRACK):
          proposed = allocation(self, CS, a_target, accel_limits, now_ns=now_ns)
          if proposed is not None:
            if isinstance(proposed, bool) or not math.isfinite(proposed) or not accel_limits[0] <= proposed <= accel_limits[1]:
              raise ValueError('Invalid experimental response allocation')
            output_accel = float(proposed)
      output_accel = min(output_accel, self.last_output_accel + STARTING_ACCEL_RATE * DT_CTRL)

    if self.motion_input_constrained:
      output_accel = min(output_accel, DEGRADED_ACCEL)
    elif intent.reasons & (StopReason.CRUISE | StopReason.FORCE_DECEL):
      output_accel = min(output_accel, CarControllerParams.BOSCH_GAS_LOOKUP_BP[0])
    if self.motion_supervisor.rest_braking_accel_limit is not None:
      output_accel = min(output_accel, self.motion_supervisor.rest_braking_accel_limit)
    self.last_output_accel = float(np.clip(output_accel, accel_limits[0], accel_limits[1]))
    if self.launch_response is not None:
      self.launch_response.confirm_output(self.last_output_accel)
    self.motion_supervisor.confirm_braking_output(self.last_output_accel, accel_limits, now_ns=now_ns)
    if not protocol_hold:
      self.motion_supervisor.complete_release(self.last_output_accel, a_target)
    self.starting = self.motion_phase == MotionPhase.RELEASE
    return self.last_output_accel
