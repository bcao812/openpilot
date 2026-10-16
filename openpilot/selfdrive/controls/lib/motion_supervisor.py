import math
from dataclasses import dataclass
from enum import IntEnum, IntFlag

from opendbc.car.honda.values import CarControllerParams
from opendbc.car.structs import car
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import (CREEP_ACCEL, CREEP_FINISH_T, CREEP_MAX_T, CREEP_SPEED,
                                                          CREEP_STOP_T, CLOSE_LEAD_STOP_SPEED, STOP_RELEASE_ACCEL, STOP_SPEED)
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.longitudinal_continuation import CONTINUATION_STATUSES
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_TIMING_VERSION, valid_timestamp

MOTION_INTENT_VERSION = 10
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
  PURE_CRUISE = 3  # Retired; reserved for decoding historical logs, never accepted.
  # Version 8: guarded cruise coasting is an execution request, without
  # acceleration-error feedback. It carries UNKNOWN model semantics and
  # neither movement nor departure permission; the owner still resolves stops.
  CRUISE_COAST = 4
  # Version 10: the strictly winning cruise governor request already owns its
  # response shaping. Keep its UNKNOWN/MOVING/FORWARD reference and independent
  # movement/departure facts; this kind adds no permission to release a stop.
  CRUISE_REQUEST = 5


MOTION_REFERENCE_KINDS = tuple(MotionReferenceKind)
MOTION_RESPONSE_KINDS = tuple(kind for kind in MotionResponseKind if kind != MotionResponseKind.PURE_CRUISE)


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
  release_accel is optional consumer-local trajectory strength, never a wire
  permission or an execution request. Only new dual-source input may spend it.
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
  release_accel: float | None = None

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
            and (self.response_kind not in (MotionResponseKind.ABSOLUTE_ACC, MotionResponseKind.CRUISE_REQUEST)
                 or self.reference_kind in (MotionReferenceKind.UNKNOWN, MotionReferenceKind.MOVING, MotionReferenceKind.FORWARD))
            and (self.response_kind != MotionResponseKind.CRUISE_COAST
                 or self.reference_kind == MotionReferenceKind.UNKNOWN)
            and (self.response_kind != MotionResponseKind.CRUISE_COAST or not self.release_allowed and not self.moving_lead)
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
            and (self.release_accel is None
                 or isinstance(self.release_accel, (int, float)) and not isinstance(self.release_accel, bool)
                 and math.isfinite(self.release_accel))
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
    self._producer_instance_high_ns = 0
    self._continuation_plan_time_ns = 0
    self._continuation_model_time_ns = 0
    self._continuation_recovery_required = False
    self.reset()

  @property
  def can_standby(self):
    # Standby describes an unarmed owner, including after a previous drive.
    # It neither erases input identity nor clears a previously observed fault.
    return self.phase == MotionPhase.OFF

  @property
  def motion_owner_established(self):
    return self.phase != MotionPhase.OFF and (self._last_plan_time_ns is not None or self._last_braking_plan_time_ns is not None)

  @property
  def can_continue_track(self):
    return (self.phase == MotionPhase.TRACK and not self.reasons and not self.resolved_should_stop
            and not self.degraded and not self.waiting_for_fresh_input
            and not self._require_new_input and not self._rearm_after_fault
            and self._last_plan_time_ns is not None)

  def reset(self):
    self.phase = MotionPhase.OFF
    self.reasons = StopReason.NONE
    self.degraded = False
    self.waiting_for_fresh_input = False
    self.inactive_input_valid = False
    self.degraded_braking_accel = None
    self.continuation_active = False
    self.resolved_should_stop = False
    self.moving_lead_t = 0.0
    self.release_lead_t = 0.0
    self._reset_creep()
    self._last_now_ns = None
    self._require_new_input = bool(self._latest_input_ids()[0])

  def _observe_producer_identity(self, intent):
    # This is a transport high-water mark, not stored target or braking
    # authority. Disabled control must not allow a retired producer to return.
    self._producer_instance_high_ns = max(self._producer_instance_high_ns, intent.producer_instance_ns)

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
    return (max(self._last_plan_time_ns or 0, self._last_braking_plan_time_ns or 0, self._rearm_plan_time_ns,
                self._continuation_plan_time_ns),
            max(self._last_model_time_ns or 0, self._last_braking_model_time_ns or 0, self._rearm_model_time_ns,
                self._continuation_model_time_ns))

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
    if intent.producer_instance_ns < self._producer_instance_high_ns or intent.producer_instance_ns > plan_time_ns:
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

  def valid_standby_input(self, intent, a_target, plan_time_ns, model_time_ns, now_ns):
    return (self.can_standby and isinstance(intent, MotionIntent) and intent.is_standby_valid()
            and self._finite_values(a_target)
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
    # Standby is healthy transport without an executable plan. Its retained
    # solver status is historical; it cannot prove native recovery or erase a
    # fault, even when returning to standby after an earlier drive.
    if (standby_input is True and clock_valid and self._finite_values(v_ego, a_ego)
        and self.valid_standby_input(intent, a_target, plan_time_ns, model_time_ns, now_ns)):
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
    latest_plan, latest_model = self._latest_input_ids()
    if plan_time_ns > latest_plan and model_time_ns > latest_model:
      # A genuinely new healthy pair can retire the old input fault while OFF.
      # Re-entry still needs another pair beyond the OFF barrier below; this
      # grants neither drive/release authority nor a reset of the fault budget.
      self._rearm_after_fault = False
    self.inactive_input_valid = not self._rearm_after_fault
    self._observe_producer_identity(intent)
    # Even an ineligible off sample establishes the rearm barrier. Repeated
    # inactive calls cannot erase it or turn this sample into a new drive plan.
    self._last_plan_time_ns, self._last_model_time_ns = plan_time_ns, model_time_ns
    self._last_intent, self._last_target = intent, a_target
    self._remember_rearm_input(plan_time_ns, model_time_ns, now_ns, clock_valid)
    self._require_new_input = True

  def update(self, active, v_ego, a_ego, standstill, a_target, intent, *,
             plan_time_ns=None, model_time_ns=None, now_ns=None, plan_valid=True, braking_plan_valid=False, solver_status=0,
             standby_input=False, continuation_valid=False, hold_for_takeover=False):
    self.continuation_active = False
    self.inactive_input_valid = False
    self.degraded_braking_accel = None
    if not active:
      self._observe_inactive(v_ego, a_ego, a_target, intent, plan_time_ns, model_time_ns, now_ns, plan_valid, solver_status, standby_input)
      return False

    # The main state machine owns the takeover deadline and existing authority.
    # Constrain only an already held stop; do not arm OFF or create a new stop.
    # A standstill observation can briefly move that same owned stop to APPROACH.
    hold_for_takeover = (hold_for_takeover is True
                         and (self.phase == MotionPhase.HOLD
                              or self.phase == MotionPhase.APPROACH and bool(self.reasons & StopReason.HOLD)))
    if hold_for_takeover:
      self.reasons |= StopReason.FORCE_DECEL | StopReason.HOLD

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
    if (continuation_valid is True and clock_valid and vehicle_values_valid and not standstill and v_ego >= CLOSE_LEAD_STOP_SPEED
        and self.can_continue_track and solver_status in CONTINUATION_STATUSES
        and self._valid_braking_input(intent, a_target, plan_time_ns, model_time_ns, now_ns)):
      # Execution is separately certified by LongControl. Do not count this
      # as native success, refresh a donor, or accumulate stop/release proof.
      self.continuation_active = True
      self._continuation_plan_time_ns = int(plan_time_ns)
      self._continuation_model_time_ns = int(model_time_ns)
      self._continuation_recovery_required = True
      return False
    if (self._continuation_recovery_required
        and (not self._timestamp_valid(plan_time_ns) or not self._timestamp_valid(model_time_ns)
             or plan_time_ns <= self._continuation_plan_time_ns or model_time_ns <= self._continuation_model_time_ns)):
      plan_valid = False
    if not (plan_valid and solver_status == 0 and clock_valid and vehicle_values_valid and self._finite_values(a_target)
            and self._valid_input(intent, a_target, plan_time_ns, model_time_ns, now_ns)):
      self._remember_rearm_input(plan_time_ns, model_time_ns, now_ns, clock_valid)
      self._degrade(v_ego, standstill)
      if (braking_plan_valid is True and type(solver_status) is int and solver_status != 0 and clock_valid and vehicle_values_valid
          and self._valid_braking_input(intent, a_target, plan_time_ns, model_time_ns, now_ns)):
        self._observe_producer_identity(intent)
        self.degraded_braking_accel = float(intent.braking_accel)
        self._last_braking_plan_time_ns, self._last_braking_model_time_ns = plan_time_ns, model_time_ns
        self._last_braking_intent, self._last_braking_target = intent, a_target
        self._require_new_input = True
      return self.resolved_should_stop

    plan_step = None if self._last_plan_time_ns is None else (plan_time_ns - self._last_plan_time_ns) / 1e9
    model_step = None if self._last_model_time_ns is None else (model_time_ns - self._last_model_time_ns) / 1e9
    latest_plan_time, latest_model_time = self._latest_input_ids()
    new_input = latest_plan_time == 0 or plan_time_ns > latest_plan_time and model_time_ns > latest_model_time
    self._last_plan_time_ns, self._last_model_time_ns = plan_time_ns, model_time_ns
    self._last_intent, self._last_target = intent, a_target
    self._observe_producer_identity(intent)
    if self._require_new_input and not new_input:
      self.degraded_braking_accel = min(a_target, DEGRADED_ACCEL)
      if not self._rearm_after_fault:
        self._wait_for_fresh_input(v_ego, standstill)
      else:
        self._degrade(v_ego, standstill)
      return self.resolved_should_stop
    self._require_new_input = False
    self._rearm_after_fault = False
    self._continuation_recovery_required = False
    self._rearm_plan_time_ns = self._rearm_model_time_ns = 0
    self.waiting_for_fresh_input = False
    self.degraded = False
    if self.phase == MotionPhase.OFF and standstill:
      self.phase = MotionPhase.HOLD
      self.reasons = StopReason.HOLD

    raw_reasons = StopReason(intent.reasons)
    if hold_for_takeover:
      # A recovered producer may arrive before its forceDecel input. Keep this
      # consumer veto separate from its identity and accept stronger braking.
      raw_reasons |= StopReason.FORCE_DECEL
    stopped_phase = self.phase in (MotionPhase.APPROACH, MotionPhase.HOLD)
    held_stop = stopped_phase and bool(self.phase == MotionPhase.HOLD or self.reasons & StopReason.HOLD)
    # CLOSE_LEAD describes a current target, not ownership of that target
    # after a healthy producer withdraws it. A rolling close-only stop may
    # follow the new target without inventing departure evidence or a HOLD.
    # Actual standstill and a previously established HOLD keep their separate
    # brake-release contract; clearing a target reason cannot start the car.
    withdrawn_close_stop = (stopped_phase and bool(self.reasons & StopReason.CLOSE_LEAD)
                            and not raw_reasons & StopReason.CLOSE_LEAD)
    close_only_withdrawal = (withdrawn_close_stop and not held_stop and not standstill
                            and not self.reasons & ~(StopReason.CLOSE_LEAD | StopReason.INVALID))
    if withdrawn_close_stop:
      self.reasons &= ~StopReason.CLOSE_LEAD
    # A creep deadline cannot outlive its current stop evidence on an unheld,
    # rolling car. A fresh healthy plan may still request ordinary braking;
    # withdrawing this added stop needs neither departure permission nor gas.
    creep_only_withdrawal = (new_input and stopped_phase and bool(self.reasons & StopReason.CREEP)
                             and not raw_reasons and not held_stop and not standstill and v_ego >= STOP_SPEED
                             and not self.reasons & ~(StopReason.CREEP | StopReason.INVALID))
    moving_eligible = (intent.moving_lead and not raw_reasons and not stopped_phase and not standstill
                       and STOP_SPEED <= v_ego < CLOSE_LEAD_STOP_SPEED)
    self.moving_lead_t = self._confirmation_time(self.moving_lead_t, moving_eligible, new_input, plan_step, model_step)
    # Confirm independent departure evidence while the reference sheds its old
    # braking history. Waiting for its scalar to cross zero before starting the
    # same confirmation adds a second delay; a negative target still cannot
    # release the held stop below.
    release_eligible = intent.release_allowed and not raw_reasons and stopped_phase
    self.release_lead_t = self._confirmation_time(self.release_lead_t, release_eligible, new_input, plan_step, model_step)
    # Creep assists a stop already requested by current evidence. Low speed,
    # weak braking or missing movement permission cannot create a stop alone.
    creep_stop = self._update_creep(v_ego, a_target, intent.allow_creep and bool(raw_reasons),
                                    elapsed_ns / 1e9,
                                    movement_evidence=bool(intent.moving_lead or intent.release_allowed), fresh_input=new_input)

    reasons = raw_reasons
    if creep_stop:
      reasons |= StopReason.CREEP

    near_standstill = standstill or v_ego < STOP_SPEED
    # An already rolling car may continue a newly accepted moving target
    # after the stop cause disappears; it need not request more speed first.
    # A fresh withdrawal of a creep-only obligation also permits continued
    # rolling, even when the current reference has no departure permission.
    # A withdrawn CLOSE_LEAD alone must not become a replacement HOLD reason.
    # This cannot start a held car or spend a cached source. Standstill
    # departure keeps its separate positive-target and confirmation contract.
    rolling_continuation = (
      self.phase == MotionPhase.APPROACH and not held_stop and not standstill and not reasons
      and (close_only_withdrawal or creep_only_withdrawal
           or not near_standstill and new_input
           and (intent.reference_kind in (MotionReferenceKind.MOVING, MotionReferenceKind.FORWARD)
                or intent.response_kind in (MotionResponseKind.ABSOLUTE_ACC, MotionResponseKind.CRUISE_COAST, MotionResponseKind.CRUISE_REQUEST)))
    )
    strong_departure = (new_input and intent.release_accel is not None and intent.release_accel >= STOP_RELEASE_ACCEL)
    release_qualified = (intent.release_allowed and a_target > CREEP_ACCEL and not raw_reasons
                         and (a_target >= STOP_RELEASE_ACCEL or strong_departure or self.release_lead_t >= CREEP_STOP_T - 1e-9))
    if release_qualified:
      reasons = StopReason.NONE
      self._reset_creep()
    elif (stopped_phase and not reasons and not rolling_continuation
          and (held_stop or near_standstill or not intent.release_allowed or self.reasons & StopReason.CREEP)):
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
