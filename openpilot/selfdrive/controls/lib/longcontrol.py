import math
import time
from types import MethodType

import numpy as np
from opendbc.car.structs import car
from opendbc.car.honda.values import CarControllerParams
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CLOSE_LEAD_STOP_SPEED, CONTROL_N, STOP_SPEED
from openpilot.common.pid import PIDController
from openpilot.selfdrive.controls.lib.launch_response import LaunchResponse, ResponseDomain
from openpilot.selfdrive.controls.lib.legacy_longcontrol import update_legacy
from openpilot.selfdrive.controls.lib.legacy_longcontrol import long_control_state_trans as long_control_state_trans
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode, resolve_longitudinal_mode
from openpilot.selfdrive.controls.lib.longitudinal_continuation import (
  CONTINUATION_STATUSES, INACTIVE_NATIVE_RECOVERY_REASON, ContinuationBudget, ContinuationDonor, ContinuationHistory, ContinuationOffer,
  execution_target, validate_continuation,
)
from openpilot.selfdrive.controls.lib.motion_supervisor import (DEGRADED_ACCEL, MAX_CONTROL_GAP_NS, MotionPhase, MotionReferenceKind, MotionResponseKind,
                                                               MotionSupervisor, StopReason, ROLLING_STOP_ACCEL,
                                                               STARTING_ACCEL_RATE, STOPPING_DECEL_RATE)
from openpilot.selfdrive.modeld.constants import ModelConstants

LongCtrlState = car.CarControl.Actuators.LongControlState
MPC_FAILURE_ACCEL = -0.5
# The frozen source-epoch replay consumes this public grid for plan transport.
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]


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
    self.continuation_budget = ContinuationBudget()
    self.continuation_history = ContinuationHistory()
    self.continuation_donor = None
    self.continuation_offer = None
    self.continuation_reason = 'none'
    # The host Bosch CAN encoder has no acceleration-error integrator. Do not
    # duplicate a tuned PID.
    response_feedback = self.unified and all(gain == 0.0 for gain in CP.longitudinalTuning.kiV)
    self.launch_response = LaunchResponse(self.pid, max_gap_ns=MAX_CONTROL_GAP_NS) if response_feedback else None
    if not self.unified:
      self.update = MethodType(update_legacy, self)

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
             standby_input=False, continuation=None, continuation_input_valid=False, reference_check=None, force_decel=False,
             car_state_ns=None, native_recovery_valid=False):
    """Run the UNIFIED owner; legacy capability binds its own update at construction."""
    plan_valid = plan_valid and solver_status == 0
    braking_plan_valid = braking_plan_valid is True
    active = active and not CS.gasPressed and not CS.brakePressed
    now_ns = time.monotonic_ns() if now_ns is None else now_ns
    self.continuation_offer = None
    self.continuation_reason = 'none'
    history_valid = self.continuation_history.observe(car_state_ns, now_ns, CS.vEgo, CS.aEgo)
    reference = reference_check.timing if reference_check is not None and reference_check.valid else None
    if not active:
      self.continuation_donor = None
      # Observe identities for rearm without bypassing the immediate
      # manual-override OFF/zero-output return below.
      self.motion_supervisor.update(False, CS.vEgo, CS.aEgo, CS.standstill, a_target, intent,
                                    plan_time_ns=plan_time_ns, model_time_ns=model_time_ns, now_ns=now_ns,
                                    plan_valid=plan_valid, solver_status=solver_status, standby_input=standby_input)
      if (self.motion_supervisor.inactive_input_valid and native_recovery_valid is True and reference is not None):
        # Observe real, advancing native successes while remaining OFF. This
        # receipt may retire a failure episode, but cannot establish a donor,
        # execute a plan, release a stop, or bypass the next-input rearm gate.
        self.continuation_budget.note_success(now_ns, producer_ns=reference.producer_ns,
                                             source_ns=reference.source_ns, model_ns=model_time_ns)
        self.continuation_reason = INACTIVE_NATIVE_RECOVERY_REASON
      elif solver_status != 0 and not standby_input:
        self.continuation_budget.note_failure(now_ns)
      else:
        self.continuation_budget.interrupt_recovery()
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
    continuation_valid = False
    continuation_target = None
    if solver_status != 0:
      episode = (continuation.episode_start_ns if isinstance(continuation, ContinuationOffer)
                 and reference is not None else None)
      clock_valid = self.continuation_budget.note_failure(now_ns, episode_start_ns=episode)
      current_state = self.continuation_history.current_state(now_ns) if history_valid else None
      if (clock_valid and current_state is not None and solver_status in CONTINUATION_STATUSES and continuation_input_valid is True
          and limits_valid and last_output_valid and stop_accel_valid and not force_decel
          and not should_stop and not CS.standstill and not CS.cruiseState.standstill and CS.vEgo >= CLOSE_LEAD_STOP_SPEED
          and self.motion_supervisor.can_continue_track and intent is not None and intent.is_braking_valid()
          and not intent.reasons and not intent.rest_goal_rejected):
        self.continuation_reason = validate_continuation(
          continuation, donor=self.continuation_donor, reference=reference, budget=self.continuation_budget,
          plan_ns=plan_time_ns, model_ns=model_time_ns, now_ns=now_ns, speed=current_state[0], accel=current_state[1], cruise_kph=CS.vCruise,
        )
        if not self.continuation_reason:
          self.continuation_reason = self.continuation_history.check(continuation.candidate, reference, now_ns)
        if not self.continuation_reason:
          continuation_target = execution_target(continuation, reference, now_ns, self.CP.longitudinalActuatorDelay)
          if continuation_target is None:
            self.continuation_reason = 'continuation_action_horizon_invalid'
        continuation_valid = not self.continuation_reason
      else:
        self.continuation_reason = 'continuation_authority_unavailable'
    if force_decel:
      self.continuation_donor = None
    cold_standstill = self.motion_phase == MotionPhase.OFF and CS.standstill
    self.resolved_should_stop = self.motion_supervisor.update(
      active, CS.vEgo, CS.aEgo, CS.standstill, a_target, intent, plan_time_ns=plan_time_ns,
      model_time_ns=model_time_ns, now_ns=now_ns, plan_valid=plan_valid and limits_valid and last_output_valid and stop_accel_valid,
      braking_plan_valid=braking_plan_valid and limits_valid and stop_accel_valid, solver_status=solver_status,
      continuation_valid=continuation_valid, hold_for_takeover=force_decel)
    if self.motion_supervisor.continuation_active:
      self.continuation_offer = continuation
      self.continuation_reason = 'accepted'
      self.long_control_state = LongCtrlState.pid
      self.starting = False
      self.pid.neg_limit, self.pid.pos_limit = accel_limits
      # This independent target was revalidated by the producer. Keep the
      # actuator limits and the already executed stronger braking. It cannot
      # raise positive output, change PID state or accumulate stop/departure
      # confirmation evidence. Later stronger braking on the curve still runs.
      self.last_output_accel = float(np.clip(min(continuation_target, self.last_output_accel),
                                             accel_limits[0], accel_limits[1]))
      return self.last_output_accel
    if (solver_status == 0 and plan_valid and continuation_input_valid is True and reference is not None
        and not force_decel and not should_stop and not CS.standstill and not CS.cruiseState.standstill
        and self.motion_supervisor.can_continue_track):
      self.continuation_donor = ContinuationDonor.from_reference(reference, model_time_ns)
    if (solver_status == 0 and plan_valid and native_recovery_valid is True and reference is not None
        and not self.motion_input_constrained and intent is not None and intent.is_valid() and intent.ready):
      # Healthy executed HOLD/APPROACH and capped native outputs can establish
      # stability without acquiring the separate moving-donor authority.
      self.continuation_budget.note_success(now_ns, producer_ns=reference.producer_ns,
                                             source_ns=reference.source_ns, model_ns=model_time_ns)
    else:
      self.continuation_budget.interrupt_recovery()
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
    response_available = self.launch_response is not None and not self.motion_input_constrained
    # The accepted same-frame intent declares a target domain. UNKNOWN alone
    # grants none; model launch/continuation keeps its separate qualification.
    owner_allows_response = (not self.motion_input_constrained and not self.resolved_should_stop and not protocol_hold and not intent.reasons
                             and self.motion_phase in (MotionPhase.RELEASE, MotionPhase.TRACK))
    cruise_execution = (owner_allows_response
                        and intent.response_kind in (MotionResponseKind.CRUISE_COAST, MotionResponseKind.CRUISE_REQUEST))
    response_domain = ResponseDomain.NONE
    if response_available and owner_allows_response:
      if intent.response_kind == MotionResponseKind.ABSOLUTE_ACC and self.motion_phase == MotionPhase.TRACK:
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
      output_accel = min(output_accel, self.last_output_accel + STARTING_ACCEL_RATE * DT_CTRL)

    if self.motion_input_constrained:
      output_accel = min(output_accel, DEGRADED_ACCEL)
    elif intent.reasons & (StopReason.CRUISE | StopReason.FORCE_DECEL):
      output_accel = min(output_accel, CarControllerParams.BOSCH_GAS_LOOKUP_BP[0])
    self.last_output_accel = float(np.clip(output_accel, accel_limits[0], accel_limits[1]))
    if self.launch_response is not None:
      self.launch_response.confirm_output(self.last_output_accel)
    if not protocol_hold:
      self.motion_supervisor.complete_release(self.last_output_accel, a_target)
    self.starting = self.motion_phase == MotionPhase.RELEASE
    return self.last_output_accel
