"""Continue an already controlled axis only while its independent inputs remain healthy.

This policy never removes an event, grants initial engagement, or changes panda
authority. A stopped handoff can extend an existing soft-disable once, without
releasing the held car when the planner recovers during that extension.
"""

import math
from dataclasses import dataclass

from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode, resolve_longitudinal_mode
from openpilot.selfdrive.controls.lib.longitudinal_continuation import UNSTABLE_DISABLE_NS
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION, StopReason
from openpilot.selfdrive.selfdrived.events import ET, Alert, AlertSize, AlertStatus, AudibleAlert, Priority, VisualAlert
from openpilot.selfdrive.selfdrived.model_lag_recovery import ModelLagRecovery


EventName = log.OnroadEvent.EventName
State = log.SelfdriveState.OpenpilotState
LongState = car.CarControl.Actuators.LongControlState
DISABLE_TYPES = (ET.USER_DISABLE, ET.IMMEDIATE_DISABLE, ET.SOFT_DISABLE)
EPS_EVENTS = frozenset((EventName.steerUnavailable, EventName.steerTempUnavailable))
LONG_FAILURE = EventName.longitudinalControlUnavailable
MADS_FLAG = 1024
EXECUTION_LEASE_NS = 100_000_000
INPUT_LEASE_NS = 250_000_000
HOLD_CONFIRM_NS = 500_000_000
HOLD_EXTRA_NS = 3_000_000_000
CONTROL_STEP_NS = round(DT_CTRL * 1e9)


def fresh_time(stamp, now_ns, lease_ns):
  return type(stamp) is int and type(now_ns) is int and 0 < stamp <= now_ns and now_ns - stamp <= lease_ns


def fresh_service(sm, service, now_ns, lease_ns=EXECUTION_LEASE_NS):
  return (sm.seen.get(service) is True and sm.valid.get(service) is True
          and sm.alive.get(service) is True and sm.freq_ok.get(service) is True
          and fresh_time(sm.logMonoTime.get(service), now_ns, lease_ns))


def finite(*values):
  return all(type(value) in (int, float) and math.isfinite(value) for value in values)


def other_disable(events, allowed=()):
  mapping = events.get_events_mapping()
  return any(name not in allowed and any(kind in mapping.get(name, {}) for kind in DISABLE_TYPES) for name in events.names)


class LongitudinalEvents:
  """A read-only FSM view. Original types remain available to MADS and logging."""
  def __init__(self, events, isolated_eps=False, *, recovered_model_lag=False):
    self.events = events
    self.isolated_eps = isolated_eps is True
    self.recovered_model_lag = recovered_model_lag is True

  def contains(self, event_type):
    if ((self.isolated_eps and event_type in (ET.IMMEDIATE_DISABLE, ET.SOFT_DISABLE))
        or (self.recovered_model_lag and event_type == ET.SOFT_DISABLE)):
      mapping = self.events.get_events_mapping()
      return any(event_type in mapping.get(name, {}) for name in self.events.names
                 if not (self.isolated_eps and name in EPS_EVENTS)
                 and not (self.recovered_model_lag and event_type == ET.SOFT_DISABLE and name == EventName.modeldLagging))
    return self.events.contains(event_type)


@dataclass(frozen=True)
class EngagementDecision:
  preserve_lateral: bool = False
  preserve_longitudinal: bool = False
  hold_for_takeover: bool = False
  hold_remaining_ns: int = 0
  recovered_model_lag: bool = False


class StoppedHandoff:
  def __init__(self):
    self.reset()

  def reset(self):
    self.original_deadline_ns = 0
    self.extended_deadline_ns = 0
    self.confirmed_since_ns = 0
    self.last_car_state_ns = 0
    self.active = False
    self.spent = False

  def update(self, *, enabled, state, fault_present, only_allowed_faults, evidence_valid,
             now_ns, car_state_ns, soft_disable_timer, remaining_ns, failure_deadline_ns=None):
    if not enabled:
      self.reset()
      return False
    if state != State.softDisabling:
      self.confirmed_since_ns = 0
      self.original_deadline_ns = 0
      self.extended_deadline_ns = 0
      self.active = False
      return False
    if self.spent:
      return False

    # The upstream episode may acquire an earlier start after this policy
    # observed it. A clamped remaining=0 cannot express that change, so retain
    # and tighten against the absolute deadline even during the extension.
    if type(failure_deadline_ns) is int and failure_deadline_ns > 0:
      self.original_deadline_ns = min(self.original_deadline_ns or failure_deadline_ns, failure_deadline_ns)
      if self.extended_deadline_ns:
        self.extended_deadline_ns = min(self.extended_deadline_ns, self.original_deadline_ns + HOLD_EXTRA_NS)

    valid = bool(only_allowed_faults and evidence_valid)
    if self.active:
      if valid and now_ns < self.extended_deadline_ns:
        return True
      self.active = False
      self.spent = True
      return False

    if not fault_present:
      self.confirmed_since_ns = 0
      self.original_deadline_ns = 0
      self.extended_deadline_ns = 0
      return False
    if not valid:
      self.confirmed_since_ns = 0
      if self.extended_deadline_ns:
        self.spent = True
      return False

    # This is the timer *after* the upcoming FSM decrement. The independent
    # continuation deadline can only shorten it, never restart it.
    timer_ns = max(0, soft_disable_timer - 1) * CONTROL_STEP_NS
    if type(remaining_ns) is not int or remaining_ns < 0:
      return False
    deadline = now_ns + min(timer_ns, remaining_ns)
    self.original_deadline_ns = min(self.original_deadline_ns or deadline, deadline)
    if self.extended_deadline_ns:
      self.extended_deadline_ns = min(self.extended_deadline_ns, self.original_deadline_ns + HOLD_EXTRA_NS)

    # A replayed carState may not accumulate the stationary confirmation.
    if car_state_ns <= self.last_car_state_ns:
      self.confirmed_since_ns = 0
      if self.extended_deadline_ns:
        self.spent = True
      return False
    self.last_car_state_ns = car_state_ns
    if not self.confirmed_since_ns:
      self.confirmed_since_ns = now_ns
    if (not self.extended_deadline_ns and now_ns < self.original_deadline_ns
        and now_ns - self.confirmed_since_ns >= HOLD_CONFIRM_NS):
      self.extended_deadline_ns = self.original_deadline_ns + HOLD_EXTRA_NS
    if self.extended_deadline_ns and now_ns + CONTROL_STEP_NS > self.original_deadline_ns:
      self.active = now_ns < self.extended_deadline_ns
      self.spent = not self.active
    return self.active


class EngagementPolicy:
  def __init__(self):
    self.handoff = StoppedHandoff()
    self.model_lag_recovery = ModelLagRecovery()
    self.last_now_ns = 0
    self.speed_takeover = False
    self.steering_takeover = False
    self.lateral_available = False
    self.longitudinal_available = False
    self.hold_ended_until_ns = 0
    self.decision = EngagementDecision()

  @staticmethod
  def _panda_healthy(CP, sm, now_ns, lateral):
    if not fresh_service(sm, 'pandaStates', now_ns, INPUT_LEASE_NS):
      return False
    pandas = sm['pandaStates']
    configs = CP.safetyConfigs
    ignored = (car.CarParams.SafetyModel.silent, car.CarParams.SafetyModel.noOutput)
    if not configs or len(pandas) < len(configs):
      return False
    checked = False
    for i, ps in enumerate(pandas):
      if i >= len(configs):
        if ps.safetyModel not in ignored:
          return False
        continue
      config = configs[i]
      if config.safetyModel in ignored:
        if ps.safetyModel not in ignored:
          return False
        continue
      checked = True
      if (ps.safetyModel != config.safetyModel or ps.safetyParam != config.safetyParam
          or ps.alternativeExperience != CP.alternativeExperience or ps.safetyRxChecksInvalid or len(ps.faults)
          or not (ps.controlsAllowedLateral if lateral else ps.controlsAllowed)):
        return False
    return checked

  @staticmethod
  def _execution_healthy(daemon, CS, now_ns):
    sm = daemon.sm
    return bool(
      not daemon.CP.passive and daemon.initialized and daemon._car_state_valid
      and fresh_time(daemon._car_state_mono_ns, now_ns, EXECUTION_LEASE_NS)
      and CS.canValid and not CS.canTimeout and CS.cruiseState.available
      and CS.gearShifter == car.CarState.GearShifter.drive
      and not (CS.accFaulted or CS.espDisabled or CS.espActive or CS.parkingBrake or CS.brakeHoldActive)
      and finite(CS.vEgo, CS.vEgoRaw, CS.aEgo)
      and all(fresh_service(sm, name, now_ns) for name in ('controlsState', 'carControl', 'carOutput'))
      and fresh_service(sm, 'deviceState', now_ns, 1_500_000_000) and sm['deviceState'].started
    )

  @staticmethod
  def _lateral_inputs_healthy(sm, CS, now_ns):
    return bool(
      not (CS.steerFaultTemporary or CS.steerFaultPermanent)
      and all(fresh_service(sm, name, now_ns, INPUT_LEASE_NS) for name in ('modelV2', 'deviceMotion', 'vehicleParameters'))
      and sm['deviceMotion'].inputsOK and sm['deviceMotion'].posenetOK and sm['vehicleParameters'].valid
      and finite(CS.steeringAngleDeg, CS.steeringRateDeg, sm['modelV2'].action.desiredCurvature,
                 sm['vehicleParameters'].steerRatio, sm['vehicleParameters'].stiffnessFactor)
      and sm['vehicleParameters'].steerRatio > 0 and sm['vehicleParameters'].stiffnessFactor > 0
      and sm['carControl'].latActive
    )

  @staticmethod
  def _longitudinal_inputs_healthy(sm, now_ns):
    plan, motion, CC = sm['longitudinalPlan'], sm['controlsState'].motionState, sm['carControl']
    return bool(
      all(fresh_service(sm, name, now_ns, INPUT_LEASE_NS) for name in ('longitudinalPlan', 'modelV2'))
      and fresh_time(plan.modelMonoTime, now_ns, INPUT_LEASE_NS)
      and plan.solverStatus == 0 and plan.motionIntent.valid and plan.motionIntent.ready
      and motion.version == MOTION_INTENT_VERSION and motion.planningMode == int(LongitudinalMode.UNIFIED)
      and motion.enabled and not motion.degraded and not motion.waitingForFreshInput and motion.referenceTimingValid
      and CC.enabled and CC.longActive and not CC.cruiseControl.override
      and finite(plan.aTarget, CC.actuators.accel, sm['carOutput'].actuatorsOutput.accel)
    )

  @staticmethod
  def _holding(daemon, CS):
    sm = daemon.sm
    motion, CC, output = sm['controlsState'].motionState, sm['carControl'], sm['carOutput'].actuatorsOutput
    stop_accel = daemon.CP.stopAccel
    return bool(
      CS.standstill and abs(CS.vEgoRaw) <= 0.05 and abs(CS.vEgo) < 0.3
      and not (CS.gasPressed or CS.brakePressed or CS.regenBraking)
      and motion.version == MOTION_INTENT_VERSION and motion.planningMode == int(LongitudinalMode.UNIFIED)
      and motion.enabled and motion.phase == 'hold' and motion.resolvedStop and motion.reasons & int(StopReason.HOLD)
      and not motion.waitingForFreshInput
      and CC.enabled and CC.longActive and not CC.cruiseControl.override and not CC.cruiseControl.resume
      and sm['controlsState'].longControlState == LongState.stopping
      and CC.actuators.longControlState == output.longControlState == LongState.stopping
      and finite(stop_accel, CC.actuators.accel, output.accel, output.gas)
      and -3.5 <= stop_accel <= -0.2 and -3.5 <= CC.actuators.accel <= stop_accel
      and -3.5 <= output.accel <= stop_accel and output.gas <= 0
    )

  def update(self, daemon, CS, now_ns):
    preserve_lateral = preserve_longitudinal = hold_valid = False
    recovered_model_lag = False
    self.lateral_available = False
    self.longitudinal_available = False
    events = daemon.events
    fault_present = events.has(LONG_FAILURE)
    only_long_fault = not other_disable(events, (LONG_FAILURE,))
    clock_valid = (type(now_ns) is int and now_ns > 0
                   and (not self.last_now_ns or 0 < now_ns - self.last_now_ns <= EXECUTION_LEASE_NS))
    if type(now_ns) is int and now_ns > self.last_now_ns:
      self.last_now_ns = now_ns
    try:
      supported = resolve_longitudinal_mode(daemon.CP, daemon.CP_SP) == LongitudinalMode.UNIFIED
      model_recovered = self.model_lag_recovery.update(
        daemon.sm, now_ns, lagging=supported and events.has(EventName.modeldLagging),
      )
      execution = supported and clock_valid and self._execution_healthy(daemon, CS, now_ns)
      mads_enabled = daemon.mads.enabled_toggle is True and bool(daemon.CP.alternativeExperience & MADS_FLAG)
      if execution:
        self.lateral_available = bool(
          mads_enabled and daemon.mads.active
          and self._lateral_inputs_healthy(daemon.sm, CS, now_ns)
          and self._panda_healthy(daemon.CP, daemon.sm, now_ns, lateral=True)
        )
        preserve_lateral = self.lateral_available and fault_present and only_long_fault
        self.longitudinal_available = bool(
          daemon.enabled and not (CS.gasPressed or CS.brakePressed or CS.regenBraking)
          and self._longitudinal_inputs_healthy(daemon.sm, now_ns)
          and self._panda_healthy(daemon.CP, daemon.sm, now_ns, lateral=False)
        )
        preserve_longitudinal = bool(
          mads_enabled and daemon.enabled and daemon.state_machine.state in (State.enabled, State.overriding)
          and any(events.has(name) for name in EPS_EVENTS) and not other_disable(events, EPS_EVENTS)
          and (CS.steerFaultPermanent or CS.steerFaultTemporary)
          and self.longitudinal_available
        )
        # Historical drop percentage is diagnostic once current source and
        # consumer health are confirmed. Recover an existing soft-disable too;
        # staying in that state would continue forceDecel after the input heals.
        recovered_model_lag = bool(
          model_recovered and self.longitudinal_available
          and daemon.state_machine.state in (State.enabled, State.softDisabling, State.overriding)
        )
        hold_valid = bool(self._holding(daemon, CS) and self._panda_healthy(daemon.CP, daemon.sm, now_ns, lateral=False))
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
      # Unknown inputs may not authorize an exception to the existing exits.
      preserve_lateral = preserve_longitudinal = hold_valid = recovered_model_lag = False
      self.lateral_available = False
      self.longitudinal_available = False

    was_holding = self.handoff.active
    episode_ns = getattr(getattr(daemon, '_longitudinal_failure_budget', None), 'episode_start_ns', 0)
    hold_active = self.handoff.update(
      enabled=daemon.enabled, state=daemon.state_machine.state, fault_present=fault_present,
      only_allowed_faults=only_long_fault, evidence_valid=hold_valid,
      now_ns=now_ns, car_state_ns=getattr(daemon, '_car_state_mono_ns', 0),
      soft_disable_timer=daemon.state_machine.soft_disable_timer,
      remaining_ns=getattr(daemon, '_longitudinal_soft_disable_remaining_ns', None),
      failure_deadline_ns=episode_ns + UNSTABLE_DISABLE_NS if type(episode_ns) is int and episode_ns > 0 else None,
    )
    if was_holding and not hold_active:
      self.hold_ended_until_ns = now_ns + HOLD_EXTRA_NS
    self.decision = EngagementDecision(preserve_lateral, preserve_longitudinal, hold_active,
                                       max(0, self.handoff.extended_deadline_ns - now_ns) if hold_active else 0,
                                       recovered_model_lag)
    return self.decision

  def observe_states(self, daemon, CS):
    if self.decision.preserve_lateral and not daemon.enabled and daemon.mads.active:
      self.speed_takeover = True
    if self.decision.preserve_longitudinal:
      self.steering_takeover = True
    manual = (CS.gasPressed or CS.brakePressed or CS.regenBraking
              or any(button.type == car.CarState.ButtonEvent.Type.cancel for button in CS.buttonEvents))
    if daemon.enabled or not daemon.mads.active or manual:
      self.speed_takeover = False
    if manual:
      self.hold_ended_until_ns = 0
    if not daemon.enabled or (daemon.mads.active and daemon.sm['carControl'].latActive
                              and not CS.steerFaultPermanent and not CS.steerFaultTemporary):
      self.steering_takeover = False

  def alert(self, daemon, CS):
    """Describe the remaining axes without suppressing unrelated fault alerts."""
    title = detail = key = ''
    long_warning = (self.handoff.active or self.speed_takeover or self.decision.preserve_lateral
                    or self.last_now_ns < self.hold_ended_until_ns)
    allowed = (LONG_FAILURE,) if long_warning else EPS_EVENTS
    if other_disable(daemon.events, allowed):
      return None
    if self.handoff.active:
      seconds = max(1, math.ceil(self.decision.hold_remaining_ns / 1e9))
      title, detail, key = 'Press Brake to Take Over', f'Automatic brake hold ends in {seconds} s', 'hold'
    elif self.last_now_ns < self.hold_ended_until_ns and not daemon.enabled:
      title, detail, key = 'Take Control Immediately', 'Automatic brake hold has ended', 'holdEnded'
    elif (self.decision.preserve_lateral or self.speed_takeover) and daemon.mads.active:
      title = 'Take Over Speed Control' if self.lateral_available else 'Take Control Immediately'
      detail = ('Steering assist remains active' if not daemon.enabled else 'Longitudinal control degraded; steering active') \
        if self.lateral_available else 'Speed control off; steering not confirmed'
      key = 'longitudinal'
    elif self.steering_takeover and daemon.enabled:
      title, key = 'Steer Manually', 'lateral'
      detail = 'Steering assist unavailable; speed control active' if self.longitudinal_available else 'Speed control not confirmed; take over'
    if not key:
      return None
    # FCW/AEB and immediate hardware failures retain the highest priority.
    alert = Alert(title, detail, AlertStatus.critical, AlertSize.full, Priority.HIGH,
                  VisualAlert.steerRequired, AudibleAlert.warningImmediate, 0.1)
    alert.alert_type = f'controlTakeover/{key}'
    alert.event_type = ET.PERMANENT
    return alert
