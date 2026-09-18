import math
from dataclasses import dataclass
from enum import IntEnum, IntFlag

from opendbc.car.honda.values import CarControllerParams
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import (CREEP_ACCEL, CREEP_FINISH_T, CREEP_MAX_T, CREEP_SPEED,
                                                          CREEP_STOP_T, CLOSE_LEAD_STOP_SPEED, STOP_RELEASE_ACCEL, STOP_SPEED)
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode

MOTION_INTENT_VERSION = 3
PLAN_LEASE_NS = 250_000_000
MODEL_LEASE_NS = 250_000_000
MAX_CONTROL_GAP_NS = 100_000_000
CREEP_INPUT_SERVICES = ('carState', 'radarState', 'modelV2', 'controlsState', 'carControl', 'selfdriveState', 'vehicleParameters')
CREEP_SAMPLE_MAX_GAP_NS = round(2.0 * DT_MDL * 1e9)
DEGRADED_ACCEL = -0.2
STOPPING_DECEL_RATE = 2.0
ROLLING_STOP_ACCEL = -1.5
STARTING_ACCEL_RATE = 8.0


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


@dataclass(frozen=True)
class CreepSample:
  """One producer snapshot, not a producer-owned timer or stopping decision.

  Each source retains its own monotonic identity; their maximum only names the
  snapshot endpoint and cannot make an older authority input fresh. Speed is
  the original carState measurement, never a planned/reference trajectory node.
  reset observes controlsState.off and grants no actuation authority.
  """

  source_times_ns: tuple[int, ...]
  measured_speed: float
  reset: bool

  @property
  def measurement_time_ns(self):
    return self.source_times_ns[0]

  @property
  def qualification_time_ns(self):
    return max(self.source_times_ns)

  def is_valid(self):
    return (isinstance(self.source_times_ns, tuple) and len(self.source_times_ns) == len(CREEP_INPUT_SERVICES)
            and all(type(timestamp) is int and timestamp > 0 for timestamp in self.source_times_ns)
            and isinstance(self.measured_speed, (int, float)) and not isinstance(self.measured_speed, bool)
            and math.isfinite(self.measured_speed) and type(self.reset) is bool)


@dataclass(frozen=True)
class MotionIntent:
  """Versioned producer evidence, without a producer-owned stop latch.

  reasons accepts RAW_STOP_REASONS only; derived and unknown bits are invalid.
  moving_lead is an unconfirmed qualification, not permission to release a stop.
  release_allowed is independently qualified reference movement or departure evidence;
  a held stop still requires a positive target before releasing its brake.
  allow_creep enables anti-creep evaluation, not permission to bypass a stop.
  ready separates initialization readiness from protocol validity.
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
  creep_sample: CreepSample | None = None

  @classmethod
  def from_message(cls, message):
    try:
      braking_valid = getattr(message, "brakingValid", False)
      creep_valid = getattr(message, "creepEvidenceValid", False)
      if type(braking_valid) is not bool or type(creep_valid) is not bool:
        return cls(version=0, valid=False)
      creep_sample = (CreepSample(tuple(message.creepSourceMonoTimes), message.creepMeasuredSpeed, message.creepReset)
                      if creep_valid else None)
      return cls(version=message.version, reasons=message.reasons, moving_lead=message.movingLead,
                 release_allowed=message.releaseAllowed, allow_creep=message.allowCreep, valid=message.valid,
                 ready=getattr(message, "ready", True), mode=getattr(message, "mode", int(LongitudinalMode.LEGACY)),
                 braking_accel=message.brakingAccel if braking_valid else None, creep_sample=creep_sample)
    except (AttributeError, TypeError):
      return cls(version=0, valid=False)

  def _structure_valid(self) -> bool:
    return (type(self.version) is int and self.version == MOTION_INTENT_VERSION
            and isinstance(self.mode, int) and not isinstance(self.mode, bool)
            and self.mode in (LongitudinalMode.SUPERVISED, LongitudinalMode.UNIFIED)
            and isinstance(self.reasons, int) and not isinstance(self.reasons, bool)
            and self.reasons >= 0 and not self.reasons & ~int(RAW_STOP_REASONS)
            and all(type(value) is bool for value in (self.moving_lead, self.release_allowed, self.allow_creep, self.valid, self.ready))
            and (self.creep_sample is None
                 or self.mode == LongitudinalMode.SUPERVISED and isinstance(self.creep_sample, CreepSample) and self.creep_sample.is_valid())
            and (self.braking_accel is None
                 or isinstance(self.braking_accel, (int, float)) and not isinstance(self.braking_accel, bool)
                 and math.isfinite(self.braking_accel) and CarControllerParams.BOSCH_ACCEL_MIN <= self.braking_accel <= DEGRADED_ACCEL))

  def is_valid(self) -> bool:
    return (self._structure_valid() and self.valid and self.braking_accel is None
            and (self.mode != LongitudinalMode.SUPERVISED or self.creep_sample is not None))

  def is_braking_valid(self) -> bool:
    return (self._structure_valid() and not self.valid and self.ready and not self.moving_lead
            and not self.release_allowed and not self.allow_creep and self.braking_accel is not None)

  def is_standby_valid(self) -> bool:
    return (self._structure_valid() and not self.valid and not self.ready
            and not self.moving_lead and not self.release_allowed and not self.allow_creep
            and self.reasons in (0, int(StopReason.CRUISE)) and self.braking_accel is None and self.creep_sample is None)


class MotionSupervisor:
  """Own normal stop phases; repeated producer samples do not add evidence.

  Plan and model timestamps share monotonic nanoseconds with now_ns. Both have
  a 250 ms lease, and producer identities never move backwards during an
  instance's lifetime. Disabled control or rejected input requires both producer
  identities to advance before rearming drive or release. Valid stronger braking
  remains available while awaiting rearm. Missing or version-zero intent is a
  degraded input on this explicitly selected path, never another controller.
  """

  def __init__(self, mode: LongitudinalMode = LongitudinalMode.UNIFIED):
    self.mode = mode
    self._last_plan_time_ns = None
    self._last_model_time_ns = None
    self._last_now_ns = None
    self._last_intent = None
    self._last_target = None
    self._last_braking_plan_time_ns = None
    self._last_braking_model_time_ns = None
    self._last_braking_intent = None
    self._last_braking_target = None
    self._last_creep_sample = None
    self._last_independent_creep_sample = None
    self._last_creep_observation_ns = None
    self._require_new_input = False
    self._rearm_after_fault = False
    self._rearm_plan_time_ns = 0
    self._rearm_model_time_ns = 0
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
    self._pending_off_creep = None
    self._last_now_ns = None
    self._require_new_input = bool(self._latest_input_ids()[0])

  def _reset_creep(self):
    self.creep_t = 0.0
    self.creep_total_t = 0.0
    self.creep_v = 0.0
    self._creep_window_ns = 0
    self._creep_total_ns = 0
    self._creep_off_credit_ns = 0
    self._creep_window_off_credit_ns = 0
    self._creep_auto_start_speed = 0.0

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
    # not replace its high-water mark in either motion-owner mode.
    if not self._timestamp_valid(now_ns):
      return False
    if self._last_creep_observation_ns is not None and now_ns <= self._last_creep_observation_ns:
      return False
    self._last_creep_observation_ns = now_ns
    return True

  def _valid_transport(self, intent, a_target, plan_time_ns, model_time_ns, now_ns):
    if not isinstance(self.mode, int) or isinstance(self.mode, bool) or intent.mode != self.mode:
      return False
    if not all(self._timestamp_valid(timestamp) for timestamp in (plan_time_ns, model_time_ns, now_ns)):
      return False
    if not (model_time_ns <= plan_time_ns <= now_ns
            and now_ns - plan_time_ns <= PLAN_LEASE_NS and now_ns - model_time_ns <= MODEL_LEASE_NS):
      return False
    if intent.creep_sample is not None and not self._valid_creep_transport(intent.creep_sample, plan_time_ns, model_time_ns, now_ns):
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

  def _valid_creep_transport(self, sample, plan_time_ns, model_time_ns, now_ns):
    sources = sample.source_times_ns
    if sources[2] != model_time_ns or any(timestamp > plan_time_ns or now_ns - timestamp > PLAN_LEASE_NS for timestamp in sources):
      return False
    authority_times = [sources[index] for index in (0, 3, 4, 5)]
    if max(authority_times) - min(authority_times) > CREEP_SAMPLE_MAX_GAP_NS:
      return False
    previous = self._last_creep_sample
    if previous is not None:
      if any(current < old for current, old in zip(sources, previous.source_times_ns, strict=True)):
        return False
      if sources[0] == previous.source_times_ns[0] and sample.measured_speed != previous.measured_speed:
        return False
      if sources[3] == previous.source_times_ns[3] and sample.reset != previous.reset:
        return False
    return True

  def _observe_creep_sample(self, sample, new_input):
    previous = self._last_creep_sample
    independent = (new_input and (previous is None or sample.measurement_time_ns > previous.measurement_time_ns))
    step_ns = None
    if independent:
      if self._last_independent_creep_sample is not None:
        step_ns = sample.qualification_time_ns - self._last_independent_creep_sample.qualification_time_ns
        if not 0 < step_ns <= CREEP_SAMPLE_MAX_GAP_NS:
          step_ns = None
      self._last_independent_creep_sample = sample
    self._last_creep_sample = sample
    return independent, step_ns

  def _valid_input(self, intent, a_target, plan_time_ns, model_time_ns, now_ns):
    return (isinstance(intent, MotionIntent) and intent.is_valid() and intent.ready
            and self._valid_transport(intent, a_target, plan_time_ns, model_time_ns, now_ns))

  def _valid_braking_input(self, intent, a_target, plan_time_ns, model_time_ns, now_ns):
    return (isinstance(intent, MotionIntent) and intent.is_braking_valid()
            and self._valid_transport(intent, a_target, plan_time_ns, model_time_ns, now_ns))

  def _constrain_motion(self, v_ego, standstill, *, preserve_off_sample=False):
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
    if not preserve_off_sample:
      self._pending_off_creep = None

  def _degrade(self, v_ego, standstill, *, preserve_off_sample=False):
    self._constrain_motion(v_ego, standstill, preserve_off_sample=preserve_off_sample)
    self.reasons |= StopReason.INVALID
    self.degraded = True
    self.waiting_for_fresh_input = False
    self._rearm_after_fault = True

  def _wait_for_fresh_input(self, v_ego, standstill, *, preserve_off_sample=False):
    self._constrain_motion(v_ego, standstill, preserve_off_sample=preserve_off_sample)
    self.waiting_for_fresh_input = True
    self.degraded = False

  @staticmethod
  def _confirmation_time(previous, eligible, new_input, plan_step, model_step):
    if not eligible:
      return 0.0
    if new_input:
      if plan_step is None or max(plan_step, model_step) > 2.0 * DT_MDL + 1e-9:
        return DT_MDL
      return min(CREEP_STOP_T, previous + min(DT_MDL, plan_step, model_step))
    return previous

  def _update_creep(self, v_ego, a_target, allowed, elapsed):
    if not allowed or not (0.0 <= v_ego < CREEP_SPEED and a_target < CREEP_ACCEL):
      self._reset_creep()
      return False
    if self.reasons & StopReason.CREEP and self.phase in (MotionPhase.APPROACH, MotionPhase.HOLD):
      return True
    if self.creep_t == 0.0:
      self.creep_v = v_ego
    self.creep_t += elapsed
    self.creep_total_t += elapsed
    if self.creep_total_t >= CREEP_MAX_T - 1e-9:
      return True
    if self.creep_t > CREEP_STOP_T + 1e-9:
      if (self.creep_v - v_ego) * CREEP_FINISH_T < v_ego * CREEP_STOP_T:
        return True
      self.creep_t = 0.0
    return False

  @staticmethod
  def _in_creep_band(speed, target):
    return math.isfinite(speed) and math.isfinite(target) and 0.0 <= speed < CREEP_SPEED and target < CREEP_ACCEL

  def _observe_inactive(self, v_ego, a_ego, a_target, intent, plan_time_ns, model_time_ns, now_ns, plan_valid, solver_status,
                        standby_input):
    """Retain one off observation while keeping execution reset and unarmed."""
    pending = self._pending_off_creep
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
    latest_plan_time, latest_model_time = self._latest_input_ids()
    new_input = latest_plan_time == 0 or plan_time_ns > latest_plan_time and model_time_ns > latest_model_time
    independent = False
    if self.mode == LongitudinalMode.SUPERVISED:
      independent, _ = self._observe_creep_sample(intent.creep_sample, new_input)
    # Even an ineligible off sample establishes the rearm barrier. Repeated
    # inactive calls cannot erase it or turn this sample into a new drive plan.
    self._last_plan_time_ns, self._last_model_time_ns = plan_time_ns, model_time_ns
    self._last_intent, self._last_target = intent, a_target
    self._remember_rearm_input(plan_time_ns, model_time_ns, now_ns, clock_valid)
    self._require_new_input = True
    sample = intent.creep_sample
    if (self.mode == LongitudinalMode.SUPERVISED and sample.reset and not intent.allow_creep and self._in_creep_band(v_ego, a_target)
        and self._in_creep_band(sample.measured_speed, a_target)
        and (independent or pending == (sample, plan_time_ns, model_time_ns))):
      self._pending_off_creep = (sample, plan_time_ns, model_time_ns)

  @staticmethod
  def _sample_confirmation_time(previous, eligible, independent, step_ns):
    if not eligible:
      return 0.0
    if independent:
      if previous == 0.0 or step_ns is None:
        return DT_MDL
      return min(CREEP_STOP_T, previous + step_ns / 1e9)
    return previous

  def _update_sampled_creep(self, v_ego, a_target, intent, independent, step_ns, moving_eligible, following_moving_lead,
                            plan_time_ns, model_time_ns):
    sample = intent.creep_sample
    current_band = self._in_creep_band(v_ego, a_target)
    measured_band = self._in_creep_band(sample.measured_speed, a_target)
    if not current_band or not measured_band or following_moving_lead:
      self._reset_creep()
      self._pending_off_creep = None
      return False
    if not intent.allow_creep:
      self._reset_creep()
      if independent and sample.reset:
        self._pending_off_creep = (sample, plan_time_ns, model_time_ns)
      return False
    if self.reasons & StopReason.CREEP and self.phase in (MotionPhase.APPROACH, MotionPhase.HOLD):
      return True
    if not independent:
      return False

    if moving_eligible and self._creep_off_credit_ns:
      # Any later fresh moving sample can revoke the borrowed pre-active
      # interval once, while no stop has been established. Preserve all active
      # dwell: intermittent moving evidence cannot keep extending the deadline.
      self._creep_total_ns -= self._creep_off_credit_ns
      self._creep_window_ns -= self._creep_window_off_credit_ns
      if self._creep_window_off_credit_ns:
        self.creep_v = self._creep_auto_start_speed
      self._creep_off_credit_ns = 0
      self._creep_window_off_credit_ns = 0

    # The v3 SUPERVISED contract evaluates independent measured snapshots. A
    # first sample has one explicit DT_MDL compatibility credit, not a claim
    # that its preceding 50 ms were measured. Continuations use actual source
    # snapshot intervals; control-loop repeats never mint more sample credit.
    if step_ns is None or sample.reset:
      self._reset_creep()
    if self._creep_total_ns == 0:
      elapsed_ns = round(DT_MDL * 1e9)
      self.creep_v = sample.measured_speed
      self._creep_auto_start_speed = sample.measured_speed
      if self._pending_off_creep is not None and not sample.reset and not moving_eligible:
        previous, previous_plan_ns, previous_model_ns = self._pending_off_creep
        interval_ns = sample.qualification_time_ns - previous.qualification_time_ns
        if (0 < interval_ns <= CREEP_SAMPLE_MAX_GAP_NS and sample.measurement_time_ns > previous.measurement_time_ns
            and plan_time_ns > previous_plan_ns and model_time_ns > previous_model_ns):
          elapsed_ns += interval_ns
          self.creep_v = previous.measured_speed
          self._creep_off_credit_ns = interval_ns
          self._creep_window_off_credit_ns = interval_ns
      # First fresh moving-lead evidence starts both clocks at this sample.
      # Borrowing an off credit here could stop a legitimate moving follow
      # before its movement exemption has time to confirm.
      self._pending_off_creep = None
    else:
      elapsed_ns = step_ns
      if self._creep_window_ns == 0:
        self.creep_v = sample.measured_speed
    self._creep_window_ns += elapsed_ns
    self._creep_total_ns += elapsed_ns
    self.creep_t = self._creep_window_ns / 1e9
    self.creep_total_t = self._creep_total_ns / 1e9
    if self._creep_total_ns >= round(CREEP_MAX_T * 1e9):
      return True
    if self._creep_window_ns > round(CREEP_STOP_T * 1e9):
      if (self.creep_v - sample.measured_speed) * CREEP_FINISH_T < sample.measured_speed * CREEP_STOP_T:
        return True
      self._creep_window_ns = 0
      self._creep_window_off_credit_ns = 0
      self.creep_t = 0.0
    return False

  def update(self, active, v_ego, a_ego, standstill, a_target, intent, *,
             plan_time_ns=None, model_time_ns=None, now_ns=None, plan_valid=True, braking_plan_valid=False, solver_status=0,
             standby_input=False):
    self.degraded_braking_accel = None
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

    measured_v_ego = v_ego
    vehicle_values_valid = self._finite_values(v_ego, a_ego)
    if vehicle_values_valid:
      v_ego = max(0.0, v_ego)
    if not (plan_valid and solver_status == 0 and clock_valid and vehicle_values_valid and self._finite_values(a_target)
            and self._valid_input(intent, a_target, plan_time_ns, model_time_ns, now_ns)):
      self._remember_rearm_input(plan_time_ns, model_time_ns, now_ns, clock_valid)
      self._degrade(v_ego, standstill)
      if (braking_plan_valid is True and type(solver_status) is int and solver_status != 0 and clock_valid and vehicle_values_valid
          and self._valid_braking_input(intent, a_target, plan_time_ns, model_time_ns, now_ns)):
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
    independent, sample_step_ns = False, None
    if self.mode == LongitudinalMode.SUPERVISED:
      independent, sample_step_ns = self._observe_creep_sample(intent.creep_sample, new_input)
    if self._require_new_input and not new_input:
      self.degraded_braking_accel = min(a_target, DEGRADED_ACCEL)
      retain_off = (self._pending_off_creep == (intent.creep_sample, plan_time_ns, model_time_ns)
                    and self._in_creep_band(measured_v_ego, a_target))
      if not self._rearm_after_fault:
        self._wait_for_fresh_input(v_ego, standstill, preserve_off_sample=retain_off)
      else:
        self._degrade(v_ego, standstill, preserve_off_sample=retain_off)
      return self.resolved_should_stop
    self._require_new_input = False
    self._rearm_after_fault = False
    self._rearm_plan_time_ns = self._rearm_model_time_ns = 0
    self.waiting_for_fresh_input = False
    self.degraded = False

    if self.phase == MotionPhase.OFF and standstill:
      self.phase = MotionPhase.HOLD
      self.reasons = StopReason.HOLD

    raw_reasons = StopReason(intent.reasons)
    stopped_phase = self.phase in (MotionPhase.APPROACH, MotionPhase.HOLD)
    moving_eligible = (intent.moving_lead and not raw_reasons and not stopped_phase and not standstill
                       and STOP_SPEED <= v_ego < CLOSE_LEAD_STOP_SPEED)
    if self.mode == LongitudinalMode.SUPERVISED:
      self.moving_lead_t = self._sample_confirmation_time(self.moving_lead_t, moving_eligible, independent, sample_step_ns)
    else:
      self.moving_lead_t = self._confirmation_time(self.moving_lead_t, moving_eligible, new_input, plan_step, model_step)
    # Confirm independent departure evidence while the reference sheds its old
    # braking history. Waiting for its scalar to cross zero before starting the
    # same confirmation adds a second delay; a negative target still cannot
    # release the held stop below.
    release_eligible = intent.release_allowed and not raw_reasons and stopped_phase
    self.release_lead_t = self._confirmation_time(self.release_lead_t, release_eligible, new_input, plan_step, model_step)
    following_moving_lead = self.moving_lead_t >= CREEP_STOP_T - 1e-9
    if self.mode == LongitudinalMode.SUPERVISED:
      creep_stop = self._update_sampled_creep(measured_v_ego, a_target, intent, independent, sample_step_ns,
                                              moving_eligible, following_moving_lead, plan_time_ns, model_time_ns)
    else:
      creep_stop = self._update_creep(v_ego, a_target, intent.allow_creep and not following_moving_lead, elapsed_ns / 1e9)

    reasons = raw_reasons
    if creep_stop:
      reasons |= StopReason.CREEP
    if (stopped_phase and self.reasons & StopReason.CLOSE_LEAD
        and STOP_SPEED <= v_ego < CLOSE_LEAD_STOP_SPEED and ROLLING_STOP_ACCEL < a_target < STOP_RELEASE_ACCEL):
      reasons |= StopReason.CLOSE_LEAD

    near_standstill = standstill or v_ego < STOP_SPEED
    release_qualified = (intent.release_allowed and a_target > CREEP_ACCEL and not raw_reasons
                         and (a_target >= STOP_RELEASE_ACCEL or self.release_lead_t >= CREEP_STOP_T - 1e-9))
    if release_qualified:
      reasons = StopReason.NONE
      self._reset_creep()
      self._pending_off_creep = None
    elif stopped_phase and not reasons and (near_standstill or not intent.release_allowed):
      reasons = (self.reasons & ~StopReason.INVALID) or StopReason.HOLD
      if near_standstill:
        reasons |= StopReason.HOLD

    if reasons:
      self.phase = MotionPhase.HOLD if standstill else MotionPhase.APPROACH
      if self.phase == MotionPhase.HOLD:
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
