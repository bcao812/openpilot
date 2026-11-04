"""Fault isolation must preserve the healthy axis without creating authority."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from opendbc.car.honda.values import CAR
from opendbc.car.structs import car
from openpilot.cereal import custom, log
import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION, StopReason
from openpilot.selfdrive.selfdrived.engagement_policy import (
  CONTROL_STEP_NS, EPS_EVENTS, EXECUTION_LEASE_NS, HOLD_EXTRA_NS, INPUT_LEASE_NS,
  EngagementPolicy, LongitudinalEvents, fresh_service,
)
from openpilot.selfdrive.selfdrived.events import ET, Events
from openpilot.selfdrive.selfdrived.state import StateMachine
from openpilot.sunnypilot.mads.state import StateMachine as MadsStateMachine
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP


EventName = log.OnroadEvent.EventName
State = log.SelfdriveState.OpenpilotState
MadsState = custom.ModularAssistiveDrivingSystem.ModularAssistiveDrivingSystemState
LongState = car.CarControl.Actuators.LongControlState
NOW = 10_000_000_000


@pytest.mark.parametrize('flag', ['seen', 'valid', 'alive', 'freq_ok'])
@pytest.mark.parametrize('value', [False, None, 0, 1, 'true', [], [True]])
def test_fresh_service_requires_literal_true_and_short_circuits_in_order(flag, value):
  order = ('seen', 'valid', 'alive', 'freq_ok', 'logMonoTime')
  reads = []

  class FlagMap(dict):
    def __init__(self, name, result):
      super().__init__(modelV2=result)
      self.name = name

    def get(self, service):
      reads.append((self.name, service))
      return super().get(service)

  sm = SimpleNamespace(**{name: FlagMap(name, True) for name in order[:4]})
  setattr(sm, flag, FlagMap(flag, value))
  # No timestamp map is installed: a failed health check must not read it.
  result = fresh_service(sm, 'modelV2', NOW)
  assert result is False
  assert reads == [(name, 'modelV2') for name in order[:order.index(flag) + 1]]


@pytest.mark.parametrize('lease_ns', [EXECUTION_LEASE_NS, INPUT_LEASE_NS, 1_500_000_000])
@pytest.mark.parametrize('position,expected', [
  ('now', True), ('deadline', True), ('expired', False), ('future', False),
  ('zero', False), ('negative', False), ('float', False), ('bool', False), ('missing', False),
])
def test_fresh_service_keeps_timestamp_type_and_inclusive_deadline(lease_ns, position, expected):
  stamps = {'now': NOW, 'deadline': NOW - lease_ns, 'expired': NOW - lease_ns - 1,
            'future': NOW + 1, 'zero': 0, 'negative': -1, 'float': float(NOW), 'bool': True}
  sm = SimpleNamespace(**{name: {'modelV2': True} for name in ('seen', 'valid', 'alive', 'freq_ok')},
                       logMonoTime={} if position == 'missing' else {'modelV2': stamps[position]})
  assert fresh_service(sm, 'modelV2', NOW, lease_ns) is expected


@pytest.mark.parametrize('now_ns', [None, True, float(NOW), '10000000000', 0, -1])
def test_fresh_service_does_not_coerce_the_clock(now_ns):
  sm = SimpleNamespace(**{name: {'modelV2': True} for name in ('seen', 'valid', 'alive', 'freq_ok')},
                       logMonoTime={'modelV2': NOW})
  assert fresh_service(sm, 'modelV2', now_ns) is False


@pytest.mark.parametrize('flag', ['seen', 'valid', 'alive', 'freq_ok', 'logMonoTime'])
@pytest.mark.parametrize('defect', ['missing_attribute', 'invalid_mapping', 'missing_service'])
def test_fresh_service_keeps_the_original_exception_and_missing_key_behavior(flag, defect):
  def reference(sm):
    return (all(getattr(sm, name).get('modelV2') is True for name in ('seen', 'valid', 'alive', 'freq_ok'))
            and type(sm.logMonoTime.get('modelV2')) is int
            and 0 < sm.logMonoTime.get('modelV2') <= NOW
            and NOW - sm.logMonoTime.get('modelV2') <= EXECUTION_LEASE_NS)

  def inputs():
    sm = SimpleNamespace(**{name: {'modelV2': True} for name in ('seen', 'valid', 'alive', 'freq_ok')},
                         logMonoTime={'modelV2': NOW})
    if defect == 'missing_attribute':
      delattr(sm, flag)
    else:
      setattr(sm, flag, None if defect == 'invalid_mapping' else {})
    return sm

  def outcome(call):
    try:
      return ('result', call(inputs()))
    except (AttributeError, TypeError) as error:
      return ('exception', type(error), str(error))

  assert outcome(lambda sm: fresh_service(sm, 'modelV2', NOW)) == outcome(reference)


class Streams(dict):
  def __init__(self):
    super().__init__()
    for name in ('controlsState', 'carControl', 'carOutput', 'deviceState', 'deviceMotion',
                 'vehicleParameters', 'modelV2', 'longitudinalPlan', 'pandaStates'):
      event = messaging.new_message(name, 1 if name == 'pandaStates' else None)
      self[name] = getattr(event, name)
    for flag in ('seen', 'valid', 'alive', 'freq_ok'):
      setattr(self, flag, dict.fromkeys(self, True))
    self.logMonoTime = dict.fromkeys(self, NOW)


def make_daemon(*, held=False):
  CP = car.CarParams.new_message()
  CP.carFingerprint = CAR.HONDA_ACCORD
  CP.openpilotLongitudinalControl = True
  CP.alternativeExperience = 1024
  CP.stopAccel = -2.0
  CP.safetyConfigs = [{'safetyModel': 'hondaBosch', 'safetyParam': 3}]
  CP_SP = custom.CarParamsSP.new_message()
  CS = car.CarState.new_message()
  CS.canValid = True
  CS.cruiseState.available = True
  CS.gearShifter = car.CarState.GearShifter.drive
  CS.vEgo = CS.vEgoRaw = 0.0 if held else 15.0
  CS.standstill = held
  sm = Streams()
  sm['deviceState'].started = True
  sm['deviceMotion'].inputsOK = sm['deviceMotion'].posenetOK = True
  sm['vehicleParameters'].valid = True
  sm['vehicleParameters'].steerRatio = 15.0
  sm['vehicleParameters'].stiffnessFactor = 1.0
  CC = sm['carControl']
  CC.enabled = CC.longActive = True
  CC.latActive = not held
  CC.actuators.accel = -2.0 if held else 0.1
  CC.actuators.longControlState = LongState.stopping if held else LongState.pid
  output = sm['carOutput'].actuatorsOutput
  output.accel = CC.actuators.accel
  output.gas = -30000.0 if held else 100.0
  output.longControlState = CC.actuators.longControlState
  controls = sm['controlsState']
  controls.longControlState = CC.actuators.longControlState
  motion = controls.motionState
  motion.version = MOTION_INTENT_VERSION
  motion.planningMode = int(LongitudinalMode.UNIFIED)
  motion.enabled = motion.referenceTimingValid = True
  motion.phase = 'hold' if held else 'track'
  motion.resolvedStop = held
  motion.reasons = int(StopReason.HOLD) if held else 0
  plan = sm['longitudinalPlan']
  plan.modelMonoTime = NOW
  plan.aTarget = -2.0 if held else 0.1
  plan.motionIntent.version = MOTION_INTENT_VERSION
  plan.motionIntent.mode = int(LongitudinalMode.UNIFIED)
  plan.motionIntent.valid = plan.motionIntent.ready = True
  panda = sm['pandaStates'][0]
  panda.safetyModel = CP.safetyConfigs[0].safetyModel
  panda.safetyParam = 3
  panda.alternativeExperience = 1024
  panda.controlsAllowed = panda.controlsAllowedLateral = True
  daemon = SimpleNamespace(CP=CP, CP_SP=CP_SP, sm=sm, initialized=True, enabled=True, active=True,
                           events=Events(), events_sp=EventsSP(), state_machine=StateMachine(),
                           engagement_policy=EngagementPolicy(), _car_state_valid=True, _car_state_mono_ns=NOW,
                           _longitudinal_failure_budget=SimpleNamespace(episode_start_ns=NOW),
                           _longitudinal_soft_disable_remaining_ns=3_000_000_000)
  daemon.state_machine.state = State.enabled
  daemon.mads = SimpleNamespace(selfdrive=daemon, enabled_toggle=True, enabled=True, active=True,
                               longitudinal_failure_isolated=False)
  daemon.mads.state_machine = MadsStateMachine(daemon.mads)
  daemon.mads.state_machine.state = MadsState.enabled
  return daemon, CS


def tick(daemon, CS, frame, *, update_fault_budget=True, refresh_car_state=True):
  now_ns = NOW + frame * CONTROL_STEP_NS
  daemon.sm.logMonoTime.update(dict.fromkeys(daemon.sm, now_ns))
  daemon.sm['longitudinalPlan'].modelMonoTime = now_ns
  if refresh_car_state:
    daemon._car_state_mono_ns = now_ns
  if update_fault_budget:
    daemon._longitudinal_soft_disable_remaining_ns = max(0, 3_000_000_000 - (now_ns - NOW))
  decision = daemon.engagement_policy.update(daemon, CS, now_ns)
  daemon.mads.longitudinal_failure_isolated = decision.preserve_lateral
  daemon.enabled, daemon.active = daemon.state_machine.update(
    LongitudinalEvents(daemon.events, decision.preserve_longitudinal),
    soft_disable_remaining_ns=daemon._longitudinal_soft_disable_remaining_ns,
    hold_for_takeover=decision.hold_for_takeover,
  )
  return decision


def add_long_failure(daemon):
  daemon.events.add(EventName.longitudinalControlUnavailable)
  daemon.sm['controlsState'].motionState.degraded = True
  daemon.sm['longitudinalPlan'].solverStatus = 4
  daemon.sm['longitudinalPlan'].motionIntent.valid = False
  daemon.sm.valid['longitudinalPlan'] = False


def test_longitudinal_failure_keeps_already_active_mads_after_main_expires():
  daemon, CS = make_daemon()
  add_long_failure(daemon)
  for frame in range(302):
    decision = tick(daemon, CS, frame)
    assert decision.preserve_lateral
    daemon.mads.enabled, daemon.mads.active = daemon.mads.state_machine.update()
    assert daemon.mads.active
  assert not daemon.enabled
  assert daemon.events.contains(ET.NO_ENTRY) and daemon.events.contains(ET.SOFT_DISABLE)
  assert daemon.events.to_msg()[0].softDisable
  daemon.engagement_policy.observe_states(daemon, CS)
  alert = daemon.engagement_policy.alert(daemon, CS)
  assert alert.alert_text_1 == 'Take Over Speed Control'
  assert 'Steering assist remains active' == alert.alert_text_2


@pytest.mark.parametrize('event', [EventName.commIssue, EventName.cameraMalfunction, EventName.overheat,
                                 EventName.canError, EventName.controlsMismatch, EventName.relayMalfunction,
                                 EventName.pedalPressed, EventName.buttonCancel, EventName.accFaulted])
def test_shared_or_manual_faults_never_get_axis_or_hold_exceptions(event):
  daemon, CS = make_daemon()
  add_long_failure(daemon)
  daemon.events.add(event)
  decision = tick(daemon, CS, 0)
  assert not decision.preserve_lateral and not decision.preserve_longitudinal and not decision.hold_for_takeover


@pytest.mark.parametrize('defect', ['mads_off', 'mads_inactive', 'missing_mads_safety_bit', 'stock_long', 'interceptor',
                                  'stale_model', 'invalid_pose', 'invalid_vehicle', 'panda_denied', 'wrong_safety',
                                  'panda_rx_invalid', 'nan_speed', 'can_timeout', 'car_state_invalid', 'offroad'])
def test_lateral_retention_requires_present_supported_and_independent_authority(defect):
  daemon, CS = make_daemon()
  add_long_failure(daemon)
  if defect == 'mads_off':
    daemon.mads.enabled_toggle = False
  elif defect == 'mads_inactive':
    daemon.mads.active = False
  elif defect == 'missing_mads_safety_bit':
    daemon.CP.alternativeExperience = 0
  elif defect == 'stock_long':
    daemon.CP.openpilotLongitudinalControl = False
  elif defect == 'interceptor':
    daemon.CP_SP.enableGasInterceptor = True
  elif defect == 'stale_model':
    daemon.sm.logMonoTime['modelV2'] = NOW - 250_000_001
  elif defect == 'invalid_pose':
    daemon.sm['deviceMotion'].inputsOK = False
  elif defect == 'invalid_vehicle':
    daemon.sm['vehicleParameters'].valid = False
  elif defect == 'panda_denied':
    daemon.sm['pandaStates'][0].controlsAllowedLateral = False
  elif defect == 'wrong_safety':
    daemon.sm['pandaStates'][0].safetyParam = 0
  elif defect == 'panda_rx_invalid':
    daemon.sm['pandaStates'][0].safetyRxChecksInvalid = True
  elif defect == 'nan_speed':
    CS.vEgo = float('nan')
  elif defect == 'can_timeout':
    CS.canTimeout = True
  elif defect == 'car_state_invalid':
    daemon._car_state_valid = False
  elif defect == 'offroad':
    daemon.sm['deviceState'].started = False
  decision = daemon.engagement_policy.update(daemon, CS, NOW)
  assert not decision.preserve_lateral


@pytest.mark.parametrize('event', sorted(EPS_EVENTS))
def test_eps_only_fault_keeps_existing_longitudinal_but_original_event_stops_mads(event):
  daemon, CS = make_daemon()
  CS.steerFaultPermanent = event == EventName.steerUnavailable
  CS.steerFaultTemporary = True
  daemon.events.add(event)
  decision = tick(daemon, CS, 0)
  assert decision.preserve_longitudinal
  assert daemon.state_machine.state == State.enabled and daemon.enabled
  daemon.mads.enabled, daemon.mads.active = daemon.mads.state_machine.update()
  expected = MadsState.disabled if event == EventName.steerUnavailable else MadsState.softDisabling
  assert daemon.mads.state_machine.state == expected
  assert daemon.events.contains(ET.NO_ENTRY)
  assert LongitudinalEvents(daemon.events, True).contains(ET.NO_ENTRY)
  daemon.engagement_policy.observe_states(daemon, CS)
  assert daemon.engagement_policy.alert(daemon, CS).alert_text_1 == 'Steer Manually'


@pytest.mark.parametrize('defect', ['mads_off', 'main_off', 'planner_invalid', 'degraded_owner', 'override',
                                  'gas', 'brake', 'panda_denied', 'no_eps_fault', 'shared_fault'])
def test_eps_isolation_cannot_enable_or_rescue_an_unhealthy_longitudinal_axis(defect):
  daemon, CS = make_daemon()
  CS.steerFaultPermanent = True
  daemon.events.add(EventName.steerUnavailable)
  if defect == 'mads_off':
    daemon.mads.enabled_toggle = False
  elif defect == 'main_off':
    daemon.enabled = False
    daemon.state_machine.state = State.disabled
  elif defect == 'planner_invalid':
    daemon.sm.valid['longitudinalPlan'] = False
  elif defect == 'degraded_owner':
    daemon.sm['controlsState'].motionState.degraded = True
  elif defect == 'override':
    daemon.sm['carControl'].cruiseControl.override = True
  elif defect in ('gas', 'brake'):
    setattr(CS, defect + 'Pressed', True)
  elif defect == 'panda_denied':
    daemon.sm['pandaStates'][0].controlsAllowed = False
  elif defect == 'no_eps_fault':
    CS.steerFaultPermanent = False
  elif defect == 'shared_fault':
    daemon.events.add(EventName.cameraMalfunction)
  assert not daemon.engagement_policy.update(daemon, CS, NOW).preserve_longitudinal


def start_extended_hold():
  daemon, CS = make_daemon(held=True)
  add_long_failure(daemon)
  for frame in range(301):
    decision = tick(daemon, CS, frame)
  assert decision.hold_for_takeover
  assert daemon.enabled and daemon.state_machine.holding_for_takeover
  assert daemon.engagement_policy.handoff.extended_deadline_ns == NOW + 3_000_000_000 + HOLD_EXTRA_NS
  return daemon, CS


def test_hold_extension_expires_once_even_when_fault_recovers_during_handoff():
  daemon, CS = start_extended_hold()
  daemon.events.clear()
  daemon.sm.valid['longitudinalPlan'] = True
  daemon.sm['longitudinalPlan'].solverStatus = 0
  daemon.sm['controlsState'].motionState.degraded = False
  daemon._longitudinal_soft_disable_remaining_ns = None
  daemon._longitudinal_failure_budget.episode_start_ns = 0
  for frame in range(301, 600):
    decision = tick(daemon, CS, frame, update_fault_budget=False)
    assert decision.hold_for_takeover and daemon.enabled
    assert daemon.state_machine.state == State.softDisabling
    assert daemon.state_machine.soft_disable_timer == 0
  tick(daemon, CS, 600, update_fault_budget=False)
  assert daemon.state_machine.state == State.disabled and not daemon.enabled
  tick(daemon, CS, 601, update_fault_budget=False)
  assert not daemon.enabled
  assert daemon.engagement_policy.alert(daemon, CS).alert_text_2 == 'Automatic brake hold has ended'


@pytest.mark.parametrize('defect', ['gas', 'brake', 'cancel', 'camera', 'can', 'panda', 'moving', 'raw_speed',
                                  'output_gas', 'output_released', 'owner_release', 'future_output', 'stale_panda'])
def test_handoff_loses_permission_immediately_and_cannot_rearm(defect):
  daemon, CS = start_extended_hold()
  now_ns = NOW + 301 * CONTROL_STEP_NS
  daemon.sm.logMonoTime.update(dict.fromkeys(daemon.sm, now_ns))
  daemon._car_state_mono_ns = now_ns
  if defect in ('gas', 'brake'):
    setattr(CS, defect + 'Pressed', True)
  elif defect in ('cancel', 'camera', 'can'):
    daemon.events.add({'cancel': EventName.buttonCancel, 'camera': EventName.cameraMalfunction, 'can': EventName.canError}[defect])
  elif defect == 'panda':
    daemon.sm['pandaStates'][0].controlsAllowed = False
  elif defect == 'moving':
    CS.standstill = False
  elif defect == 'raw_speed':
    CS.vEgoRaw = 0.1
  elif defect == 'output_gas':
    daemon.sm['carOutput'].actuatorsOutput.gas = 1.0
  elif defect == 'output_released':
    daemon.sm['carOutput'].actuatorsOutput.accel = 0.0
  elif defect == 'owner_release':
    daemon.sm['controlsState'].motionState.phase = 'release'
  elif defect == 'future_output':
    daemon.sm.logMonoTime['carOutput'] = now_ns + 1
  elif defect == 'stale_panda':
    daemon.sm.logMonoTime['pandaStates'] = now_ns - 250_000_001
  decision = daemon.engagement_policy.update(daemon, CS, now_ns)
  assert not decision.hold_for_takeover
  daemon.enabled, _ = daemon.state_machine.update(daemon.events, hold_for_takeover=False)
  assert not daemon.enabled and daemon.state_machine.state == State.disabled
  tick(daemon, CS, 302)
  assert not daemon.enabled


@pytest.mark.parametrize('defect', ['standby', 'no_actual_output', 'replayed_car_state', 'too_late'])
def test_a_hold_claim_cannot_create_an_extension_without_prior_stationary_execution(defect):
  daemon, CS = make_daemon(held=True)
  add_long_failure(daemon)
  if defect == 'standby':
    daemon.sm['carControl'].longActive = False
    daemon.sm['controlsState'].motionState.phase = 'off'
  elif defect == 'no_actual_output':
    daemon.sm['carOutput'].actuatorsOutput.accel = 0.0
  elif defect == 'too_late':
    daemon.state_machine.state = State.softDisabling
    daemon.state_machine.soft_disable_timer = 0
    daemon._longitudinal_soft_disable_remaining_ns = 0
    assert not tick(daemon, CS, 300).hold_for_takeover
    assert not daemon.enabled
    return
  for frame in range(302):
    decision = tick(daemon, CS, frame, refresh_car_state=defect != 'replayed_car_state')
    assert not decision.hold_for_takeover
  assert not daemon.enabled


def test_shorter_failure_deadline_is_not_restarted_by_a_new_soft_disable():
  daemon, CS = make_daemon(held=True)
  add_long_failure(daemon)
  daemon._longitudinal_failure_budget.episode_start_ns = NOW - 1_000_000_000
  for frame in range(202):
    daemon._longitudinal_soft_disable_remaining_ns = max(0, 2_000_000_000 - frame * CONTROL_STEP_NS)
    tick(daemon, CS, frame, update_fault_budget=False)
  assert daemon.engagement_policy.handoff.extended_deadline_ns == NOW + 2_000_000_000 + HOLD_EXTRA_NS


@pytest.mark.parametrize('tighten_frame', [200, 350])
def test_earlier_episode_tightens_both_reserved_and_active_handoff_deadline(tighten_frame):
  daemon, CS = make_daemon(held=True)
  add_long_failure(daemon)
  for frame in range(tighten_frame):
    tick(daemon, CS, frame)
  assert daemon.engagement_policy.handoff.extended_deadline_ns == NOW + 6_000_000_000
  daemon._longitudinal_failure_budget.episode_start_ns = NOW - 1_000_000_000
  for frame in range(tighten_frame, 501):
    daemon._longitudinal_soft_disable_remaining_ns = max(0, 2_000_000_000 - frame * CONTROL_STEP_NS)
    tick(daemon, CS, frame, update_fault_budget=False)
    assert daemon.engagement_policy.handoff.extended_deadline_ns == NOW + 5_000_000_000
  assert not daemon.enabled


def test_handoff_clock_rollback_revokes_extension():
  daemon, CS = start_extended_hold()
  decision = daemon.engagement_policy.update(daemon, CS, NOW + 2_999_999_999)
  assert not decision.hold_for_takeover
  daemon.enabled, _ = daemon.state_machine.update(daemon.events, hold_for_takeover=False)
  assert not daemon.enabled


def test_real_selfdrived_integration_keeps_original_eps_types_for_mads():
  from openpilot.selfdrive.selfdrived.selfdrived import SelfdriveD

  daemon, CS = make_daemon()
  CS.steerFaultPermanent = True
  daemon.events.add(EventName.steerUnavailable)
  with patch('openpilot.selfdrive.selfdrived.selfdrived.time.monotonic_ns', return_value=NOW):
    SelfdriveD.update_engagement_state(daemon, CS)
  assert daemon.enabled
  assert daemon.events.contains(ET.IMMEDIATE_DISABLE) and daemon.events.contains(ET.NO_ENTRY)
  daemon.mads.enabled, daemon.mads.active = daemon.mads.state_machine.update()
  assert not daemon.mads.active


def test_speed_takeover_warning_does_not_claim_unconfirmed_steering():
  daemon, CS = make_daemon()
  add_long_failure(daemon)
  daemon.enabled = False
  daemon.state_machine.state = State.disabled
  daemon.engagement_policy.speed_takeover = True
  daemon.sm['pandaStates'][0].controlsAllowedLateral = False
  daemon.engagement_policy.update(daemon, CS, NOW)
  alert = daemon.engagement_policy.alert(daemon, CS)
  assert alert.alert_text_1 == 'Take Control Immediately'
  assert 'steering not confirmed' in alert.alert_text_2


@pytest.mark.parametrize('reason', ['gas', 'brake', 'override'])
def test_steering_takeover_warning_does_not_claim_overridden_longitudinal_control(reason):
  daemon, CS = make_daemon()
  daemon.engagement_policy.steering_takeover = True
  daemon.mads.active = False
  if reason == 'override':
    daemon.sm['carControl'].longActive = False
    daemon.sm['carControl'].cruiseControl.override = True
  else:
    setattr(CS, reason + 'Pressed', True)
  daemon.engagement_policy.update(daemon, CS, NOW)
  daemon.engagement_policy.observe_states(daemon, CS)
  alert = daemon.engagement_policy.alert(daemon, CS)
  assert alert is not None and alert.alert_text_2 == 'Speed control not confirmed; take over'


@pytest.mark.parametrize('collision', [EventName.fcw, EventName.aeb, EventName.stockAeb])
def test_takeover_alert_never_masks_an_earlier_collision_warning(collision):
  from openpilot.selfdrive.selfdrived.alertmanager import AlertManager

  daemon, CS = start_extended_hold()
  collision_events = Events()
  collision_events.add(collision)
  collision_alerts = collision_events.create_alerts([ET.PERMANENT])
  assert collision_alerts
  manager = AlertManager()
  manager.add_many(1, collision_alerts)
  manager.add_many(2, [daemon.engagement_policy.alert(daemon, CS)])
  manager.process_alerts(2, set())
  assert manager.current_alert in collision_alerts


def configure_pandas(daemon, permissions, *, ignored_tail=None):
  """Independent configured boards, optionally followed by an unused board."""
  daemon.CP.safetyConfigs = [{'safetyModel': 'hondaBosch', 'safetyParam': i + 3} for i in range(len(permissions))]
  event = messaging.new_message('pandaStates', len(permissions) + (ignored_tail is not None))
  for i, (lateral, longitudinal) in enumerate(permissions):
    panda = event.pandaStates[i]
    panda.safetyModel = 'hondaBosch'
    panda.safetyParam = i + 3
    panda.alternativeExperience = daemon.CP.alternativeExperience
    panda.controlsAllowedLateral = lateral
    panda.controlsAllowed = longitudinal
  if ignored_tail is not None:
    event.pandaStates[len(permissions)].safetyModel = ignored_tail
  daemon.sm['pandaStates'] = event.pandaStates


@pytest.mark.parametrize('denied_index', [0, 1])
@pytest.mark.parametrize('permissions', [(True, True), (False, True), (True, False), (False, False)])
@pytest.mark.parametrize('ignored_tail', [None, 'silent', 'noOutput'])
def test_each_configured_panda_must_independently_permit_each_axis(denied_index, permissions, ignored_tail):
  daemon, CS = make_daemon()
  boards = [(True, True), (True, True)]
  boards[denied_index] = permissions
  configure_pandas(daemon, boards, ignored_tail=ignored_tail)
  daemon.engagement_policy.update(daemon, CS, NOW)
  assert daemon.engagement_policy.lateral_available is permissions[0]
  assert daemon.engagement_policy.longitudinal_available is permissions[1]


@pytest.mark.parametrize('defect', ['config', 'model', 'parameter', 'experience', 'rx', 'faults', 'valid', 'time'])
def test_current_panda_and_configuration_changes_are_not_cached_across_control_frames(defect):
  daemon, CS = make_daemon()
  configure_pandas(daemon, [(True, True), (True, True)])
  policy = daemon.engagement_policy
  policy.update(daemon, CS, NOW)
  assert policy.lateral_available and policy.longitudinal_available
  panda = daemon.sm['pandaStates'][1]
  if defect == 'config':
    target, field, invalid = daemon.CP.safetyConfigs[1], 'safetyParam', 8
  elif defect == 'valid':
    target, field, invalid = daemon.sm.valid, 'pandaStates', False
  elif defect == 'time':
    target, field, invalid = daemon.sm.logMonoTime, 'pandaStates', NOW - INPUT_LEASE_NS
  else:
    field, invalid = {'model': ('safetyModel', 'toyota'), 'parameter': ('safetyParam', 8),
                      'experience': ('alternativeExperience', 0), 'rx': ('safetyRxChecksInvalid', True),
                      'faults': ('faults', ['relayMalfunction'])}[defect]
    target = panda
  is_mapping = isinstance(target, dict)
  original = target[field] if is_mapping else getattr(target, field)
  if field == 'faults':
    original = list(original)
  elif field == 'safetyModel':
    original = str(original)

  def set_value(value):
    if is_mapping:
      target[field] = value
    else:
      setattr(target, field, value)

  set_value(invalid)
  # The payload may change without a new logMonoTime. These are new control
  # calls, and all common checks must be evaluated again before either axis.
  policy.update(daemon, CS, NOW + CONTROL_STEP_NS)
  assert not policy.lateral_available and not policy.longitudinal_available
  set_value(original)
  policy.update(daemon, CS, NOW + 2 * CONTROL_STEP_NS)
  assert policy.lateral_available and policy.longitudinal_available


def test_panda_permission_short_circuit_does_not_read_an_unused_later_board():
  daemon, CS = make_daemon()
  configure_pandas(daemon, [(False, False), (True, True)])

  class UnreadablePanda:
    def __getattribute__(self, name):
      raise AssertionError('The denied axis must stop before reading the next board')

  daemon.sm['pandaStates'] = [daemon.sm['pandaStates'][0], UnreadablePanda()]
  daemon.engagement_policy.update(daemon, CS, NOW)
  assert not daemon.engagement_policy.lateral_available
  assert not daemon.engagement_policy.longitudinal_available


@pytest.mark.parametrize('field,expected', [('controlsAllowedLateral', (False, True)), ('controlsAllowed', (True, False))])
def test_panda_axis_permission_changes_apply_on_the_next_frame_with_the_same_receipt(field, expected):
  daemon, CS = make_daemon()
  policy = daemon.engagement_policy
  policy.update(daemon, CS, NOW)
  setattr(daemon.sm['pandaStates'][0], field, False)
  policy.update(daemon, CS, NOW + CONTROL_STEP_NS)
  assert (policy.lateral_available, policy.longitudinal_available) == expected
  setattr(daemon.sm['pandaStates'][0], field, True)
  policy.update(daemon, CS, NOW + 2 * CONTROL_STEP_NS)
  assert policy.lateral_available and policy.longitudinal_available
