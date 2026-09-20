import math
from dataclasses import dataclass
from enum import IntEnum, IntFlag

from opendbc.car.honda.values import CarControllerParams
from opendbc.car.structs import car
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import (CREEP_ACCEL, CREEP_FINISH_T, CREEP_MAX_T, CREEP_SPEED,
                                                          CREEP_STOP_T, CLOSE_LEAD_STOP_SPEED, STOP_RELEASE_ACCEL, STOP_SPEED)
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_TIMING_VERSION, valid_timestamp

MOTION_INTENT_VERSION = 6
PLAN_LEASE_NS = 250_000_000
MODEL_LEASE_NS = 250_000_000
MAX_CONTROL_GAP_NS = 100_000_000
DEGRADED_ACCEL = -0.2
STOPPING_DECEL_RATE = 2.0
ROLLING_STOP_ACCEL = -1.5
STARTING_ACCEL_RATE = 8.0


class MotionReferenceKind(IntEnum):
  # REST includes an existing qualified goal continuation. EXACT_ZERO needs
  # a qualified all-zero raw array and no current RestGoal.
  UNKNOWN = 0
  EXACT_ZERO = 1
  REST = 2
  FORWARD = 3
  NO_FORWARD = 4
  MOVING = 5


class MotionResponseKind(IntEnum):
  # These describe the current target's response contract, not permission to
  # release a stop. NONE leaves only independently qualified model feedback.
  NONE = 0
  ABSOLUTE_ACC = 1
  LEAD_BRAKING = 2
  PURE_CRUISE = 3


MOTION_REFERENCE_KINDS = tuple(MotionReferenceKind)
MOTION_RESPONSE_KINDS = tuple(MotionResponseKind)


class StopReason(IntFlag):
  NONE = 0
  MPC = 1 << 0
  E2E = 1 << 1
  CRUISE = 1 << 2
  CLOSE_LEAD = 1 << 3
  FORCE_DECEL = 1 << 4
  CREEP = 1 << 5
  HOLD = 1 << 6
  INVALID = 1 << 7


RAW_STOP_REASONS = StopReason.MPC | StopReason.E2E | StopReason.CRUISE | StopReason.CLOSE_LEAD | StopReason.FORCE_DECEL
DERIVED_STOP_REASONS = StopReason.CREEP | StopReason.HOLD | StopReason.INVALID


class MotionPhase(IntEnum):
  OFF = 0
  TRACK = 1
  APPROACH = 2
  HOLD = 3
  RELEASE = 4

  @property
  def text(self) -> str:
    return self.name.lower()


def consumed_forward_reference(control, controls, *, control_ns, controls_ns, source_ns, producer_ns, instance_ns, now_ns):
  """Acknowledge the owner's execution of this instance's preceding reference.

  This is a consumption contract, not permission to depart or proof of delivered
  acceleration. The planner separately checks current intent, risk and authority.
  Fresh message headers and PID alone cannot identify which reference ran.
  """
  motion = controls.motionState
  pid = car.CarControl.Actuators.LongControlState.pid
  return (controls.longControlState == pid and control.actuators.longControlState == pid
          and motion.phase in ('release', 'track')
          and not (motion.resolvedStop or motion.degraded or motion.waitingForFreshInput or motion.reasons)
          and math.isfinite(control.actuators.accel)
          and valid_timestamp(control_ns) and producer_ns < control_ns < now_ns
          and valid_timestamp(controls_ns) and producer_ns < controls_ns <= now_ns
          and motion.referenceTimingVersion == REFERENCE_TIMING_VERSION and motion.referenceTimingValid
          and motion.referenceProducerInstanceMonoTime == instance_ns
          and motion.referenceProducerMonoTime == producer_ns
          and motion.referenceSourceMonoTime == source_ns and motion.referenceEpochMonoTime == source_ns
          and valid_timestamp(motion.referenceConsumerMonoTime)
          and producer_ns < motion.referenceConsumerMonoTime <= min(control_ns, controls_ns))


@dataclass(frozen=True)
class MotionIntent:
  """Versioned producer evidence, without a producer-owned stop latch.

  reasons accepts RAW_STOP_REASONS only; derived and unknown bits are invalid.
  moving_lead is an unconfirmed qualification, not permission to release a stop.
  release_allowed is independently qualified reference movement or departure evidence;
  a held stop still requires a positive target before releasing its brake.
  allow_creep enables anti-creep evaluation, not permission to bypass a stop.
  ready separates initialization readiness from protocol validity.
  reference_kind and rest_goal_rejected describe this source and failure;
  they grant no departure permission. producer_instance_ns binds UNIFIED
  evidence to the separately qualified measured-reference producer.
  response_kind declares a separate current-target contract; UNKNOWN model
  semantics alone grant no acceleration-response permission.
  """

  version: int = MOTION_INTENT_VERSION
  reasons: int = 0
  moving_lead: bool = False
  release_allowed: bool = False
  allow_creep: bool = True
  valid: bool = True
  ready: bool = True
  mode: int = int(LongitudinalMode.UNIFIED)
  braking_accel: float | None = None
  reference_kind: int = int(MotionReferenceKind.UNKNOWN)
  rest_goal_rejected: bool = False
  producer_instance_ns: int = 0
  response_kind: int = int(MotionResponseKind.NONE)

  @classmethod
  def from_message(cls, message):
    try:
      braking_valid = getattr(message, "brakingValid", False)
      creep_valid = getattr(message, "creepEvidenceValid", False)
      # Retired sampled-creep fields keep their schema ordinals for old logs.
      # They never grant input validity to the UNIFIED consumer.
      if type(braking_valid) is not bool or type(creep_valid) is not bool or creep_valid:
        return cls(version=0, valid=False)
      return cls(version=message.version, reasons=message.reasons, moving_lead=message.movingLead,
                 release_allowed=message.releaseAllowed, allow_creep=message.allowCreep, valid=message.valid,
                 ready=getattr(message, "ready", True), mode=getattr(message, "mode", int(LongitudinalMode.LEGACY)),
                 braking_accel=message.brakingAccel if braking_valid else None,
                 reference_kind=getattr(message, 'referenceKind', int(MotionReferenceKind.UNKNOWN)),
                 rest_goal_rejected=getattr(message, 'restGoalRejected', False),
                 producer_instance_ns=getattr(message, 'producerInstanceMonoTime', 0),
                 response_kind=getattr(message, 'responseKind', int(MotionResponseKind.NONE)))
    except (AttributeError, TypeError):
      return cls(version=0, valid=False)

  def _structure_valid(self) -> bool:
    return (type(self.version) is int and self.version == MOTION_INTENT_VERSION
            and isinstance(self.mode, int) and not isinstance(self.mode, bool)
            and self.mode == LongitudinalMode.UNIFIED
            and isinstance(self.reasons, int) and not isinstance(self.reasons, bool)
            and self.reasons >= 0 and not self.reasons & ~int(RAW_STOP_REASONS)
            and all(type(value) is bool for value in (self.moving_lead, self.release_allowed, self.allow_creep, self.valid, self.ready))
            and isinstance(self.reference_kind, int) and not isinstance(self.reference_kind, bool)
            and self.reference_kind in MOTION_REFERENCE_KINDS
            and isinstance(self.response_kind, int) and not isinstance(self.response_kind, bool)
            and self.response_kind in MOTION_RESPONSE_KINDS
            and (self.response_kind not in (MotionResponseKind.ABSOLUTE_ACC, MotionResponseKind.PURE_CRUISE)
                 or self.reference_kind == MotionReferenceKind.UNKNOWN)
            and (self.reference_kind not in (MotionReferenceKind.EXACT_ZERO, MotionReferenceKind.REST, MotionReferenceKind.NO_FORWARD)
                 or not self.release_allowed and not self.moving_lead)
            and type(self.rest_goal_rejected) is bool
            and type(self.producer_instance_ns) is int and 0 <= self.producer_instance_ns < (1 << 64)
            and (self.reference_kind == MotionReferenceKind.UNKNOWN or self.producer_instance_ns > 0)
            and (self.response_kind == MotionResponseKind.NONE
                 or self.valid and self.ready and not self.reasons and self.braking_accel is None and self.producer_instance_ns > 0)
            and (not self.rest_goal_rejected
                 or self.reference_kind == MotionReferenceKind.REST
                 and not self.valid and self.braking_accel is not None and self.producer_instance_ns > 0)
            and (self.braking_accel is None
                 or isinstance(self.braking_accel, (int, float)) and not isinstance(self.braking_accel, bool)
                 and math.isfinite(self.braking_accel) and CarControllerParams.BOSCH_ACCEL_MIN <= self.braking_accel <= DEGRADED_ACCEL))

  def is_valid(self) -> bool:
    return self._structure_valid() and self.valid and self.braking_accel is None

  def is_braking_valid(self) -> bool:
    return (self._structure_valid() and not self.valid and self.ready and not self.moving_lead
            and not self.release_allowed and not self.allow_creep and self.braking_accel is not None)

  def is_standby_valid(self) -> bool:
    return (self._structure_valid() and not self.valid and not self.ready
            and not self.moving_lead and not self.release_allowed and not self.allow_creep
            and self.reasons in (0, int(StopReason.CRUISE)) and self.braking_accel is None)


class MotionSupervisor:
  """Own normal stop phases; repeated producer samples do not add evidence.

  Plan and model timestamps share monotonic nanoseconds with now_ns. Both have
  a 250 ms lease, and producer identities never move backwards during an
  instance's lifetime. Disabled control or rejected input requires both producer
  identities to advance before rearming drive or release. Valid stronger braking
  remains available while awaiting rearm. Missing or version-zero intent is a
  degraded input on this explicitly selected path, never another controller.
  """

  def __init__(self):
    self._last_plan_time_ns = None
    self._last_model_time_ns = None
    self._last_now_ns = None
    self._last_intent = None
    self._last_target = None
    self._last_braking_plan_time_ns = None
    self._last_braking_model_time_ns = None
    self._last_braking_intent = None
    self._last_braking_target = None
    self._last_creep_observation_ns = None
    self._require_new_input = False
    self._rearm_after_fault = False
    self._rearm_plan_time_ns = 0
    self._rearm_model_time_ns = 0
    self._rest_braking_instance_high_ns = 0
    self._rest_braking_plan_high_ns = 0
    self._rest_braking_model_high_ns = 0
    self.reset()

  @property
  def can_standby(self):
    return self.phase == MotionPhase.OFF and self._last_plan_time_ns is None and self._last_braking_plan_time_ns is None

  @property
  def motion_owner_established(self):
    return self.phase != MotionPhase.OFF and (self._last_plan_time_ns is not None or self._last_braking_plan_time_ns is not None)

  def reset(self):
    self.phase = MotionPhase.OFF
    self.reasons = StopReason.NONE
    self.degraded = False
    self.waiting_for_fresh_input = False
    self.degraded_braking_accel = None
    self.resolved_should_stop = False
    self.moving_lead_t = 0.0
    self.release_lead_t = 0.0
    self._reset_creep()
    self._last_now_ns = None
    self._require_new_input = bool(self._latest_input_ids()[0])
    self._clear_rest_braking()

  def _clear_rest_braking(self):
    # Reset controller-output evidence without erasing retired producer identity.
    self._rest_braking_accel = None
    self._rest_braking_instance_ns = 0
    self._rest_braking_origin = None
    self._pending_rest_braking = None
    self.rest_braking_accel_limit = None

  def _observe_rest_braking_identity(self, intent, plan_time_ns, model_time_ns):
    instance = intent.producer_instance_ns
    if instance > self._rest_braking_instance_high_ns:
      self._clear_rest_braking()
      self._rest_braking_instance_high_ns = instance
    # A newly accepted stop plan can use the same still-fresh model.
    # Confirm its final output once; the separate drive/release rearm rule below
    # still requires both identities to advance after a fault.
    fresh = (plan_time_ns > self._rest_braking_plan_high_ns and model_time_ns >= self._rest_braking_model_high_ns)
    self._rest_braking_plan_high_ns = max(self._rest_braking_plan_high_ns, plan_time_ns)
    self._rest_braking_model_high_ns = max(self._rest_braking_model_high_ns, model_time_ns)
    return fresh

  def confirm_braking_output(self, output_accel, accel_limits, *, now_ns):
    """Consume this frame's pending cause after the final controller clip.

    A producer envelope is not the final controller output. This records a
    clipped command, not delivered deceleration. Repeated calls cannot mint
    another event, and later zero samples never update the stored magnitude.
    """
    pending, self._pending_rest_braking = self._pending_rest_braking, None
    output_valid = (accel_limits is not None and len(accel_limits) == 2 and self._finite_values(output_accel, *accel_limits)
                    and CarControllerParams.BOSCH_ACCEL_MIN <= accel_limits[0] <= output_accel <= accel_limits[1]
                    and accel_limits[1] <= CarControllerParams.BOSCH_ACCEL_MAX and output_accel <= DEGRADED_ACCEL)
    if self.phase == MotionPhase.HOLD or not output_valid:
      self._clear_rest_braking()
      return
    if pending is not None:
      instance, _, _, pending_now_ns = pending
      if now_ns != pending_now_ns or instance != self._rest_braking_instance_high_ns:
        self._clear_rest_braking()
        return
      self._rest_braking_accel = float(output_accel)
      self._rest_braking_instance_ns = instance
      self._rest_braking_origin = pending

  def _reset_creep(self):
    self.creep_t = 0.0
    self.creep_total_t = 0.0
    self.creep_v = 0.0
    self.creep_distance = 0.0
    self._creep_last_v = 0.0
    self._creep_progress_v = None

  @staticmethod
  def _timestamp_valid(timestamp) -> bool:
    return type(timestamp) is int and timestamp > 0

  @staticmethod
  def _finite_values(*values):
    try:
      return all(math.isfinite(value) for value in values)
    except (TypeError, ValueError):
      return False

  def _latest_input_ids(self):
    return (max(self._last_plan_time_ns or 0, self._last_braking_plan_time_ns or 0, self._rearm_plan_time_ns),
            max(self._last_model_time_ns or 0, self._last_braking_model_time_ns or 0, self._rearm_model_time_ns))

  def _remember_rearm_input(self, plan_time_ns, model_time_ns, now_ns, clock_valid):
    """Remember causal failed/OFF identities without accepting their payload."""
    if not clock_valid:
      return False
    if not all(self._timestamp_valid(value) for value in (plan_time_ns, model_time_ns, now_ns)):
      return False
    latest_plan, latest_model = self._latest_input_ids()
    if not (model_time_ns <= plan_time_ns <= now_ns and now_ns - plan_time_ns <= PLAN_LEASE_NS
            and now_ns - model_time_ns <= MODEL_LEASE_NS and plan_time_ns >= latest_plan and model_time_ns >= latest_model):
      return False
    self._rearm_plan_time_ns, self._rearm_model_time_ns = plan_time_ns, model_time_ns
    return True

  def _advance_observation_clock(self, now_ns):
    # The consumption clock is independent of producer validity. OFF/reset or
    # an invalid sample cannot erase it; rejected backward/repeated clocks do
    # not replace its high-water mark. The historical attribute name is kept
    # for replay snapshots; this guards all consumption, not only creep.
    if not self._timestamp_valid(now_ns):
      return False
    if self._last_creep_observation_ns is not None and now_ns <= self._last_creep_observation_ns:
      return False
    self._last_creep_observation_ns = now_ns
    return True

  def _valid_transport(self, intent, a_target, plan_time_ns, model_time_ns, now_ns):
    if intent.mode != LongitudinalMode.UNIFIED:
      return False
    if not all(self._timestamp_valid(timestamp) for timestamp in (plan_time_ns, model_time_ns, now_ns)):
      return False
    if not (model_time_ns <= plan_time_ns <= now_ns
            and now_ns - plan_time_ns <= PLAN_LEASE_NS and now_ns - model_time_ns <= MODEL_LEASE_NS):
      return False
    if intent.producer_instance_ns < self._rest_braking_instance_high_ns or intent.producer_instance_ns > plan_time_ns:
      return False
    for last_plan_time, last_model_time, last_intent, last_target in (
      (self._last_plan_time_ns, self._last_model_time_ns, self._last_intent, self._last_target),
      (self._last_braking_plan_time_ns, self._last_braking_model_time_ns, self._last_braking_intent, self._last_braking_target),
    ):
      if last_plan_time is not None:
        if plan_time_ns < last_plan_time or model_time_ns < last_model_time:
          return False
        if plan_time_ns == last_plan_time:
          same_target = a_target == last_target or math.isnan(a_target) and math.isnan(last_target)
          if model_time_ns != last_model_time or intent != last_intent or not same_target:
            return False
    return True

  def _valid_input(self, intent, a_target, plan_time_ns, model_time_ns, now_ns):
    return (isinstance(intent, MotionIntent) and intent.is_valid() and intent.ready
            and self._valid_transport(intent, a_target, plan_time_ns, model_time_ns, now_ns))

  def _valid_braking_input(self, intent, a_target, plan_time_ns, model_time_ns, now_ns):
    return (isinstance(intent, MotionIntent) and intent.is_braking_valid()
            and self._valid_transport(intent, a_target, plan_time_ns, model_time_ns, now_ns))

  def _constrain_motion(self, v_ego, standstill):
    retaining_stop = self.phase in (MotionPhase.APPROACH, MotionPhase.HOLD)
    if standstill:
      self.phase = MotionPhase.HOLD
    elif retaining_stop or self._finite_values(v_ego) and v_ego < STOP_SPEED:
      self.phase = MotionPhase.APPROACH
    else:
      self.phase = MotionPhase.TRACK
    self.reasons = self.reasons if retaining_stop else StopReason.NONE
    if self.phase == MotionPhase.HOLD:
      self.reasons |= StopReason.HOLD
    self.resolved_should_stop = self.phase in (MotionPhase.APPROACH, MotionPhase.HOLD)
    self._require_new_input = bool(self._latest_input_ids()[0])
    self.moving_lead_t = 0.0
    self.release_lead_t = 0.0
    self._reset_creep()

  def _degrade(self, v_ego, standstill):
    self._constrain_motion(v_ego, standstill)
    self.reasons |= StopReason.INVALID
    self.degraded = True
    self.waiting_for_fresh_input = False
    self._rearm_after_fault = True

  def _wait_for_fresh_input(self, v_ego, standstill):
    self._constrain_motion(v_ego, standstill)
    self.waiting_for_fresh_input = True
    self.degraded = False

  @staticmethod
  def _confirmation_time(previous, eligible, new_input, plan_step, model_step):
    if not eligible:
      return 0.0
    if new_input:
      if plan_step is None or max(plan_step, model_step) > 2.0 * DT_MDL + 1e-9:
        return DT_MDL
      # The creep window uses elapsed control time. Continuous producer pairs
      # must use that same clock: capping every 60 ms sample at 50 ms lets the
      # creep deadline win even while a healthy moving lead is confirmed.
      # Both identities must advance; the existing gap bound still restarts
      # confirmation instead of crediting an unobserved interval.
      step = min(plan_step, model_step)
      # Keep the previous first-sample bound when eligibility has just become
      # true. The preceding longer interval did not yet carry that evidence.
      if previous == 0.0:
        step = min(step, DT_MDL)
      return min(CREEP_STOP_T, previous + step)
    return previous

  def _update_creep(self, v_ego, a_target, allowed, elapsed, *, movement_evidence=False, fresh_input=False):
    if not allowed or not (0.0 <= v_ego < CREEP_SPEED and a_target < CREEP_ACCEL):
      self._reset_creep()
      return False
    if self.reasons & StopReason.CREEP and self.phase in (MotionPhase.APPROACH, MotionPhase.HOLD):
      return True
    if self.creep_t == 0.0:
      self.creep_v = v_ego
      self.creep_distance = 0.0
      self._creep_last_v = v_ego
      self._creep_progress_v = None
    previous_speed = self._creep_last_v
    previous_progress_speed = self._creep_progress_v
    if fresh_input:
      self._creep_progress_v = v_ego
    self.creep_t += elapsed
    self.creep_total_t += elapsed
    self.creep_distance += 0.5 * (previous_speed + v_ego) * elapsed
    self._creep_last_v = v_ego
    if self.creep_total_t >= CREEP_MAX_T - 1e-9:
      return True
    # The same qualified window has a motion budget: at its entry speed,
    # CREEP_STOP_T would cover this distance. Spending it while speed is still
    # rising cannot demonstrate progress toward stopping. Confirm that cause
    # only across independently advancing plan/model pairs; repeated control
    # consumption of one speed spike cannot itself add progress evidence.
    # Keep the added budget in the existing rolling-speed domain; a near-zero
    # anchor must not turn noise into a near-zero allowance. Pending movement
    # evidence keeps its original confirmation opportunity; it does not gain
    # release authority.
    distance_exhausted = (fresh_input and not movement_evidence and self.creep_v >= STOP_SPEED
                          and previous_progress_speed is not None and min(previous_progress_speed, v_ego) > self.creep_v
                          and self.creep_distance > self.creep_v * CREEP_STOP_T + 1e-9)
    if distance_exhausted:
      return True
    if self.creep_t > CREEP_STOP_T + 1e-9:
      if (self.creep_v - v_ego) * CREEP_FINISH_T < v_ego * CREEP_STOP_T:
        return True
      self.creep_t = 0.0
      self.creep_distance = 0.0
      self._creep_progress_v = None
    return False

  def _observe_inactive(self, v_ego, a_ego, a_target, intent, plan_time_ns, model_time_ns, now_ns, plan_valid, solver_status,
                        standby_input):
    """Record the rearm barrier while keeping execution reset and unarmed."""
    self.reset()
    clock_valid = self._advance_observation_clock(now_ns)
    # The caller separately confirms healthy initialization because plan_valid
    # also rejects an unset cruise speed. Standby has no execution intent and
    # may preserve, but must never erase, a previously established fault cause.
    if (standby_input is True and clock_valid and solver_status == 0 and self._finite_values(v_ego, a_ego, a_target)
        and isinstance(intent, MotionIntent) and intent.is_standby_valid()
        and self._valid_transport(intent, a_target, plan_time_ns, model_time_ns, now_ns)):
      return
    if not (clock_valid and plan_valid and solver_status == 0
            and self._finite_values(v_ego, a_ego, a_target)
            and self._valid_input(intent, a_target, plan_time_ns, model_time_ns, now_ns)):
      causal_input = self._remember_rearm_input(plan_time_ns, model_time_ns, now_ns, clock_valid)
      # Initial zero-ID placeholders are absence, not a previously seen bad
      # producer. Once input exists, OFF cannot wash a fault into healthy wait.
      initial_placeholder = (not self._latest_input_ids()[0]
                             and all(value is None or type(value) is int and value == 0 for value in (plan_time_ns, model_time_ns))
                             and (intent is None or isinstance(intent, MotionIntent) and intent.version == 0 and not intent.valid))
      if causal_input or not initial_placeholder or self._rearm_after_fault:
        self._rearm_after_fault = True
        self._require_new_input = bool(self._latest_input_ids()[0])
      return
    self._observe_rest_braking_identity(intent, plan_time_ns, model_time_ns)
    # Even an ineligible off sample establishes the rearm barrier. Repeated
    # inactive calls cannot erase it or turn this sample into a new drive plan.
    self._last_plan_time_ns, self._last_model_time_ns = plan_time_ns, model_time_ns
    self._last_intent, self._last_target = intent, a_target
    self._remember_rearm_input(plan_time_ns, model_time_ns, now_ns, clock_valid)
    self._require_new_input = True

  def update(self, active, v_ego, a_ego, standstill, a_target, intent, *,
             plan_time_ns=None, model_time_ns=None, now_ns=None, plan_valid=True, braking_plan_valid=False, solver_status=0,
             standby_input=False):
    self.degraded_braking_accel = None
    self._pending_rest_braking = None
    self.rest_braking_accel_limit = None
    if not active:
      self._observe_inactive(v_ego, a_ego, a_target, intent, plan_time_ns, model_time_ns, now_ns, plan_valid, solver_status, standby_input)
      return False

    elapsed_ns = round(DT_CTRL * 1e9)
    observation_clock_valid = self._advance_observation_clock(now_ns)
    clock_valid = self._timestamp_valid(now_ns) and observation_clock_valid
    if clock_valid and self._last_now_ns is not None:
      elapsed_ns = now_ns - self._last_now_ns
      clock_valid = 0 < elapsed_ns <= MAX_CONTROL_GAP_NS
    if self._timestamp_valid(now_ns) and observation_clock_valid and (self._last_now_ns is None or now_ns > self._last_now_ns):
      self._last_now_ns = now_ns

    vehicle_values_valid = self._finite_values(v_ego, a_ego)
    if vehicle_values_valid:
      v_ego = max(0.0, v_ego)
    if not (plan_valid and solver_status == 0 and clock_valid and vehicle_values_valid and self._finite_values(a_target)
            and self._valid_input(intent, a_target, plan_time_ns, model_time_ns, now_ns)):
      self._remember_rearm_input(plan_time_ns, model_time_ns, now_ns, clock_valid)
      self._degrade(v_ego, standstill)
      if (braking_plan_valid is True and type(solver_status) is int and solver_status != 0 and clock_valid and vehicle_values_valid
          and self._valid_braking_input(intent, a_target, plan_time_ns, model_time_ns, now_ns)):
        fresh = self._observe_rest_braking_identity(intent, plan_time_ns, model_time_ns)
        if (intent.rest_goal_rejected and solver_status in (-4, -5)
            and intent.producer_instance_ns > 0):
          if fresh:
            self._pending_rest_braking = (intent.producer_instance_ns, plan_time_ns, model_time_ns, now_ns)
        else:
          self._clear_rest_braking()
        self.degraded_braking_accel = float(intent.braking_accel)
        self._last_braking_plan_time_ns, self._last_braking_model_time_ns = plan_time_ns, model_time_ns
        self._last_braking_intent, self._last_braking_target = intent, a_target
        self._require_new_input = True
      else:
        self._clear_rest_braking()
      return self.resolved_should_stop

    plan_step = None if self._last_plan_time_ns is None else (plan_time_ns - self._last_plan_time_ns) / 1e9
    model_step = None if self._last_model_time_ns is None else (model_time_ns - self._last_model_time_ns) / 1e9
    latest_plan_time, latest_model_time = self._latest_input_ids()
    new_input = latest_plan_time == 0 or plan_time_ns > latest_plan_time and model_time_ns > latest_model_time
    self._last_plan_time_ns, self._last_model_time_ns = plan_time_ns, model_time_ns
    self._last_intent, self._last_target = intent, a_target
    fresh_rest_input = self._observe_rest_braking_identity(intent, plan_time_ns, model_time_ns)
    if (intent.reference_kind not in (MotionReferenceKind.EXACT_ZERO, MotionReferenceKind.REST)
        or intent.reference_kind == MotionReferenceKind.REST and fresh_rest_input):
      # A different normal REST ends the old zero chain. Its own executed
      # negative output may independently establish a new cause below.
      self._clear_rest_braking()
    if self._require_new_input and not new_input:
      self._clear_rest_braking()
      self.degraded_braking_accel = min(a_target, DEGRADED_ACCEL)
      if not self._rearm_after_fault:
        self._wait_for_fresh_input(v_ego, standstill)
      else:
        self._degrade(v_ego, standstill)
      return self.resolved_should_stop
    self._require_new_input = False
    self._rearm_after_fault = False
    self._rearm_plan_time_ns = self._rearm_model_time_ns = 0
    self.waiting_for_fresh_input = False
    self.degraded = False
    if (intent.reference_kind == MotionReferenceKind.REST
        and intent.producer_instance_ns > 0 and fresh_rest_input):
      # Source semantics and accepted transport are necessary but not an output
      # acknowledgement. LongControl confirms only its final clipped brake;
      # this does not add a raw stop reason, a deadline or release permission.
      self._pending_rest_braking = (intent.producer_instance_ns, plan_time_ns, model_time_ns, now_ns)
    if (intent.reference_kind == MotionReferenceKind.EXACT_ZERO and self._rest_braking_accel is not None
        and intent.producer_instance_ns == self._rest_braking_instance_ns):
      self.rest_braking_accel_limit = self._rest_braking_accel

    if self.phase == MotionPhase.OFF and standstill:
      self.phase = MotionPhase.HOLD
      self.reasons = StopReason.HOLD

    raw_reasons = StopReason(intent.reasons)
    stopped_phase = self.phase in (MotionPhase.APPROACH, MotionPhase.HOLD)
    held_stop = stopped_phase and bool(self.phase == MotionPhase.HOLD or self.reasons & StopReason.HOLD)
    moving_eligible = (intent.moving_lead and not raw_reasons and not stopped_phase and not standstill
                       and STOP_SPEED <= v_ego < CLOSE_LEAD_STOP_SPEED)
    self.moving_lead_t = self._confirmation_time(self.moving_lead_t, moving_eligible, new_input, plan_step, model_step)
    # Confirm independent departure evidence while the reference sheds its old
    # braking history. Waiting for its scalar to cross zero before starting the
    # same confirmation adds a second delay; a negative target still cannot
    # release the held stop below.
    release_eligible = intent.release_allowed and not raw_reasons and stopped_phase
    self.release_lead_t = self._confirmation_time(self.release_lead_t, release_eligible, new_input, plan_step, model_step)
    following_moving_lead = self.moving_lead_t >= CREEP_STOP_T - 1e-9
    creep_stop = self._update_creep(v_ego, a_target, intent.allow_creep and not following_moving_lead, elapsed_ns / 1e9,
                                    movement_evidence=bool(intent.moving_lead or intent.release_allowed), fresh_input=new_input)

    reasons = raw_reasons
    if creep_stop:
      reasons |= StopReason.CREEP
    if (stopped_phase and self.reasons & StopReason.CLOSE_LEAD
        and STOP_SPEED <= v_ego < CLOSE_LEAD_STOP_SPEED and ROLLING_STOP_ACCEL < a_target < STOP_RELEASE_ACCEL):
      reasons |= StopReason.CLOSE_LEAD

    near_standstill = standstill or v_ego < STOP_SPEED
    # An already rolling car may continue a newly accepted moving target
    # after the stop cause disappears; it need not request more speed first.
    # This cannot start a held car, spend a cached source, or clear the creep
    # and close-lead reasons retained above. Standstill departure keeps its
    # separate positive-target and confirmation contract.
    rolling_continuation = (
      self.phase == MotionPhase.APPROACH and not held_stop and not near_standstill and new_input and not reasons
      and (intent.reference_kind in (MotionReferenceKind.MOVING, MotionReferenceKind.FORWARD)
           or intent.response_kind in (MotionResponseKind.PURE_CRUISE, MotionResponseKind.ABSOLUTE_ACC))
    )
    release_qualified = (intent.release_allowed and a_target > CREEP_ACCEL and not raw_reasons
                         and (a_target >= STOP_RELEASE_ACCEL or self.release_lead_t >= CREEP_STOP_T - 1e-9))
    if release_qualified:
      reasons = StopReason.NONE
      self._reset_creep()
    elif stopped_phase and not reasons and (held_stop or near_standstill or not (intent.release_allowed or rolling_continuation)):
      reasons = (self.reasons & ~StopReason.INVALID) or StopReason.HOLD
      if near_standstill:
        reasons |= StopReason.HOLD

    if reasons:
      self.phase = MotionPhase.HOLD if standstill else MotionPhase.APPROACH
      # A held stop does not become a rolling-stop withdrawal merely because
      # the latest measured speed crosses STOP_SPEED on a cached plan tick.
      # Keep its ownership until the qualified departure above clears it.
      if self.phase == MotionPhase.HOLD or held_stop:
        reasons |= StopReason.HOLD
    elif stopped_phase:
      self.phase = MotionPhase.RELEASE
    elif self.phase == MotionPhase.OFF:
      self.phase = MotionPhase.TRACK

    self.reasons = reasons
    self.resolved_should_stop = self.phase in (MotionPhase.APPROACH, MotionPhase.HOLD)
    return self.resolved_should_stop

  def complete_release(self, output_accel, a_target):
    if (self.phase == MotionPhase.RELEASE and not self.degraded and not self.waiting_for_fresh_input
        and math.isfinite(output_accel) and math.isfinite(a_target) and output_accel >= a_target - 1e-9):
      self.phase = MotionPhase.TRACK
