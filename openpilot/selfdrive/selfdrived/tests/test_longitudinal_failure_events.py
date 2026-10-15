"""Route planner rejection independently of transport health and owner recovery."""

from unittest.mock import patch

import pytest

from opendbc.car.honda.values import CAR
from openpilot.cereal import log
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.motion_supervisor import MODEL_LEASE_NS, MOTION_INTENT_VERSION, PLAN_LEASE_NS
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import (
  ControlsHarness, new_event, selfdrive_event_harness, serialized,
)
from openpilot.selfdrive.selfdrived.alertmanager import AlertManager
from openpilot.selfdrive.selfdrived.engagement_policy import EngagementPolicy
from openpilot.selfdrive.selfdrived.events import ET, EVENTS, EVENT_NAME
from openpilot.selfdrive.selfdrived.state import SOFT_DISABLE_TIME, StateMachine


EventName = log.OnroadEvent.EventName
State = log.SelfdriveState.OpenpilotState


def failure_harness(status=4):
  controls = ControlsHarness()
  controls.step()
  daemon, car_state = selfdrive_event_harness(controls.control.pm.messages['controlsState'])
  daemon.CP, daemon.CP_SP = controls.control.CP, controls.control.CP_SP
  daemon.big_model_ready_t = -10.0
  # selfdrived has a newer failed plan while controls still reports its previous
  # healthy one. This ordering must already produce the dedicated NO_ENTRY.
  now_ns = controls.now_ns + 20_000_000
  event = controls.events['longitudinalPlan'].as_reader().as_builder()
  event.logMonoTime = now_ns
  event.valid = False
  event.longitudinalPlan.modelMonoTime = now_ns - 10_000_000
  event.longitudinalPlan.solverStatus = status
  event.longitudinalPlan.motionIntent.valid = False
  model = new_event('modelV2', now_ns - 5_000_000)
  daemon.sm.update_msgs(now_ns / 1e9, [serialized(event), serialized(model)])
  # The harness bounds transport frequency; fault tests explicitly unset it.
  daemon.sm.freq_ok['longitudinalPlan'] = daemon.sm.freq_ok['modelV2'] = True
  return daemon, car_state, event, now_ns


def update(daemon, car_state, now_ns):
  with patch('openpilot.selfdrive.selfdrived.selfdrived.time.monotonic_ns', return_value=now_ns), \
       patch('openpilot.selfdrive.selfdrived.selfdrived.time.monotonic', return_value=now_ns / 1e9):
    daemon.update_events(car_state)


def configure_alert_output(daemon, event_type=ET.SOFT_DISABLE):
  daemon.AM = AlertManager()
  daemon.personality = log.LongitudinalPersonality.standard
  daemon.state_machine = StateMachine()
  daemon.state_machine.current_alert_types = [ET.PERMANENT, event_type]
  daemon.enabled = event_type != ET.NO_ENTRY
  daemon.state_machine.state = State.softDisabling if daemon.enabled else State.disabled


def mark_frequency_unhealthy(sm, service):
  # bounded_submaster gives services and ignore_average_freq the same list.
  # Mutating it would unsubscribe the very service this fault must exercise.
  subscriptions = tuple(sm.services)
  sm.ignore_average_freq = [name for name in sm.ignore_average_freq if name != service]
  sm.freq_ok[service] = False
  assert tuple(sm.services) == subscriptions and service in sm.services
  assert not sm.all_freq_ok([service])


@pytest.mark.parametrize('status', [4, -6, -1])
def test_fresh_semantic_failures_keep_invalidity_and_disable_with_specific_text(status):
  daemon, car_state, event, now_ns = failure_harness(status)
  ignored = list(daemon.sm.ignore_valid)
  assert not daemon.sm['controlsState'].motionState.degraded
  update(daemon, car_state, now_ns)
  assert daemon.events.names == [EventName.longitudinalControlUnavailable]
  assert daemon.events.contains(ET.NO_ENTRY) and daemon.events.contains(ET.SOFT_DISABLE)
  assert not daemon.sm.valid['longitudinalPlan'] and not daemon.sm.all_checks()
  assert daemon.sm.ignore_valid == ignored
  assert not daemon.sm['longitudinalPlan'].motionIntent.valid
  alert = EVENTS[EventName.longitudinalControlUnavailable][ET.SOFT_DISABLE](
    daemon.CP, car_state, daemon.sm, True, 300, log.LongitudinalPersonality.standard,
  )
  assert alert.alert_text_2 == 'Longitudinal Control Unavailable'
  assert 'Communication' not in alert.alert_text_2


@pytest.mark.parametrize('defect', [
  'unseen_plan', 'dead_plan', 'slow_plan', 'expired_plan', 'future_plan', 'missing_model',
  'expired_model', 'model_after_plan', 'future_model', 'unseen_model', 'dead_model', 'slow_model',
  'invalid_model', 'expired_latest_model', 'future_latest_model', 'unknown_version', 'legacy_version',
  'wrong_mode', 'local_legacy', 'stock_long', 'interceptor', 'success_status_with_invalid_packet',
])
def test_unqualified_failure_never_exempts_the_invalid_plan_from_transport_checks(defect):
  daemon, car_state, event, now_ns = failure_harness()
  plan = event.longitudinalPlan
  model_ns = daemon.sm.logMonoTime['modelV2']
  if defect == 'expired_plan':
    event.logMonoTime = now_ns - PLAN_LEASE_NS - 1
    plan.modelMonoTime = event.logMonoTime
  elif defect == 'future_plan':
    event.logMonoTime = now_ns + 2
  elif defect == 'missing_model':
    plan.modelMonoTime = 0
  elif defect == 'expired_model':
    plan.modelMonoTime = now_ns - MODEL_LEASE_NS - 1
  elif defect == 'model_after_plan':
    event.logMonoTime = now_ns - 2
    plan.modelMonoTime = now_ns - 1
  elif defect == 'future_model':
    plan.modelMonoTime = now_ns + 2
  elif defect == 'expired_latest_model':
    model_ns = now_ns - MODEL_LEASE_NS - 1
  elif defect == 'future_latest_model':
    model_ns = now_ns + 2
  elif defect == 'unknown_version':
    plan.motionIntent.version = MOTION_INTENT_VERSION + 1
  elif defect == 'legacy_version':
    plan.motionIntent.version = 0
  elif defect == 'wrong_mode':
    plan.motionIntent.mode = int(LongitudinalMode.LEGACY)
  elif defect == 'local_legacy':
    daemon.CP.carFingerprint = CAR.HONDA_CIVIC_BOSCH
  elif defect == 'stock_long':
    daemon.CP.openpilotLongitudinalControl = daemon.cp_olc = False
  elif defect == 'interceptor':
    daemon.CP_SP.enableGasInterceptor = True
  elif defect == 'success_status_with_invalid_packet':
    plan.solverStatus = 0
  daemon.sm.update_msgs((now_ns + 1) / 1e9, [serialized(event), serialized(new_event('modelV2', model_ns))])
  daemon.sm.freq_ok['longitudinalPlan'] = daemon.sm.freq_ok['modelV2'] = True
  if defect.startswith(('unseen_', 'dead_', 'slow_', 'invalid_')):
    service = 'longitudinalPlan' if defect.endswith('_plan') else 'modelV2'
    field = {'unseen': 'seen', 'dead': 'alive', 'slow': 'freq_ok', 'invalid': 'valid'}[defect.split('_')[0]]
    getattr(daemon.sm, field)[service] = False
    if field == 'freq_ok':
      mark_frequency_unhealthy(daemon.sm, service)
  update(daemon, car_state, now_ns + 1)
  expected = EventName.commIssueAvgFreq if defect.startswith('slow_') else EventName.commIssue
  assert expected in daemon.events.names
  assert EventName.longitudinalControlUnavailable not in daemon.events.names
  assert not daemon.sm.valid['longitudinalPlan']


@pytest.mark.parametrize('health', ['valid', 'alive', 'freq_ok'])
@pytest.mark.parametrize('service', ['deviceState', 'controlsState'])
def test_independent_transport_fault_coexists_with_semantic_plan_failure(health, service):
  daemon, car_state, _, now_ns = failure_harness()
  getattr(daemon.sm, health)[service] = False
  if health == 'freq_ok':
    mark_frequency_unhealthy(daemon.sm, service)
  update(daemon, car_state, now_ns)
  expected = EventName.commIssueAvgFreq if health == 'freq_ok' else EventName.commIssue
  assert sorted(daemon.events.names) == sorted([expected, EventName.longitudinalControlUnavailable])


def test_stale_failure_stays_a_transport_fault_even_while_owner_reports_degradation():
  daemon, car_state, event, now_ns = failure_harness()
  event.logMonoTime = now_ns - PLAN_LEASE_NS - 1
  event.longitudinalPlan.modelMonoTime = event.logMonoTime
  controls = new_event('controlsState', now_ns)
  controls.controlsState = daemon.sm['controlsState']
  controls.controlsState.motionState.degraded = True
  # Deliberately references a different older plan. An owner claim cannot make
  # the stale latest packet fresh, and both faults must remain observable.
  controls.controlsState.motionState.planMonoTime = event.logMonoTime - 1
  daemon.sm.update_msgs((now_ns + 1) / 1e9, [serialized(event), serialized(controls)])
  update(daemon, car_state, now_ns + 1)
  assert sorted(daemon.events.names) == sorted([EventName.commIssue, EventName.longitudinalControlUnavailable])


def test_async_recovery_waits_for_owner_and_logs_latest_and_consumed_plan_separately():
  daemon, car_state, event, now_ns = failure_harness(-6)
  consumed_before = daemon.sm['controlsState'].motionState.planMonoTime
  with patch('openpilot.selfdrive.selfdrived.selfdrived.cloudlog.event') as emit:
    update(daemon, car_state, now_ns)
    failed_ns = event.logMonoTime
    now_ns += 10_000_000
    event.logMonoTime = now_ns
    event.valid = True
    event.longitudinalPlan.solverStatus = 0
    event.longitudinalPlan.motionIntent.valid = True
    controls = new_event('controlsState', now_ns)
    controls.controlsState = daemon.sm['controlsState']
    controls.controlsState.motionState.degraded = True
    controls.controlsState.motionState.planMonoTime = failed_ns
    daemon.sm.update_msgs(now_ns / 1e9, [serialized(event), serialized(controls)])
    update(daemon, car_state, now_ns)
    assert daemon.events.names == [EventName.longitudinalControlUnavailable]
    now_ns += 10_000_000
    controls.logMonoTime = now_ns
    controls.controlsState.motionState.degraded = False
    controls.controlsState.motionState.planMonoTime = event.logMonoTime
    controls.controlsState.motionState.modelMonoTime = event.longitudinalPlan.modelMonoTime
    daemon.sm.update_msgs(now_ns / 1e9, [serialized(controls)])
    update(daemon, car_state, now_ns)
    assert not daemon.events.names
  records = [call.kwargs for call in emit.call_args_list if call.args[0] == 'longitudinalControlUnavailable']
  assert [record['transition'] for record in records] == ['entered', 'recovered']
  assert records[0]['latest_plan_solver_status'] == -6
  assert records[0]['latest_plan_mono_time'] == failed_ns
  assert records[0]['consumed_plan_mono_time'] == consumed_before != failed_ns
  assert records[1]['latest_plan_solver_status'] == 0 and records[1]['latest_plan_valid']
  assert records[1]['latest_plan_mono_time'] == records[1]['consumed_plan_mono_time'] == event.logMonoTime


def test_failure_before_consumer_feedback_blocks_entry_and_retains_three_second_disable():
  daemon, car_state, _, now_ns = failure_harness()
  update(daemon, car_state, now_ns)
  machine = StateMachine()
  daemon.events.add(EventName.buttonEnable)
  assert machine.update(daemon.events) == (False, False)
  assert machine.state == State.disabled
  machine.state = State.enabled
  assert machine.update(daemon.events) == (True, True)
  assert machine.state == State.softDisabling
  for _ in range(int(SOFT_DISABLE_TIME / DT_CTRL) - 1):
    assert machine.update(daemon.events) == (True, True)
  assert machine.update(daemon.events) == (False, False)
  assert machine.state == State.disabled


@pytest.mark.parametrize('driver_or_immediate', [EventName.pedalPressed, EventName.buttonCancel, EventName.steerUnavailable])
def test_specific_failure_does_not_override_driver_or_immediate_disable(driver_or_immediate):
  daemon, car_state, _, now_ns = failure_harness()
  update(daemon, car_state, now_ns)
  daemon.events.add(driver_or_immediate)
  machine = StateMachine()
  machine.state = State.enabled
  assert machine.update(daemon.events) == (False, False)
  assert machine.state == State.disabled and machine.soft_disable_timer == 0


@pytest.mark.parametrize('event_type', [ET.NO_ENTRY, ET.SOFT_DISABLE])
def test_driver_braking_silences_failed_plan_without_removing_diagnostics_or_entry_gate(event_type):
  daemon, car_state, _, now_ns = failure_harness(-4)
  car_state.brakePressed = True
  with patch('openpilot.selfdrive.selfdrived.selfdrived.cloudlog.event') as emit:
    update(daemon, car_state, now_ns)
  configure_alert_output(daemon, event_type)
  daemon.update_alerts(car_state)

  assert EventName.longitudinalControlUnavailable in daemon.events.names
  assert daemon.events.contains(ET.NO_ENTRY) and daemon.events.contains(ET.SOFT_DISABLE)
  assert not daemon.sm.valid['longitudinalPlan']
  assert daemon.sm['longitudinalPlan'].solverStatus == -4
  assert not daemon.AM.current_alert.alert_type.startswith('longitudinalControlUnavailable/')
  assert not any(name.startswith('longitudinalControlUnavailable/') for name in daemon.AM.alerts)
  failures = [call.kwargs for call in emit.call_args_list if call.args[0] == 'longitudinalSolverFailure']
  assert len(failures) == 1 and failures[0]['solverStatus'] == -4

  daemon.events.add(EventName.buttonEnable)
  machine = StateMachine()
  assert machine.update(daemon.events) == (False, False)
  assert machine.state == State.disabled
  machine.state = State.enabled
  assert machine.update(daemon.events) == (False, False)  # The driver's brake still disengages.


@pytest.mark.parametrize('event_type', [ET.NO_ENTRY, ET.SOFT_DISABLE])
@pytest.mark.parametrize('recovered', [False, True])
def test_brake_clears_cached_failure_and_release_uses_current_fault_state(event_type, recovered):
  daemon, car_state, event, now_ns = failure_harness(-4)
  update(daemon, car_state, now_ns)
  configure_alert_output(daemon, event_type)
  daemon.update_alerts(car_state)
  alert_type = f'longitudinalControlUnavailable/{event_type}'
  assert daemon.AM.current_alert.alert_type == alert_type
  assert daemon.AM.alerts[alert_type].end_frame > daemon.sm.frame + 1

  car_state.brakePressed = True
  daemon.sm.frame += 1
  daemon.update_alerts(car_state)
  assert daemon.AM.current_alert.alert_type == ''
  assert daemon.AM.current_alert.audible_alert == log.SelfdriveState.AudibleAlert.none
  assert alert_type not in daemon.AM.alerts

  if recovered:
    now_ns += 20_000_000
    event.logMonoTime = now_ns
    event.valid = event.longitudinalPlan.motionIntent.valid = True
    event.longitudinalPlan.solverStatus = 0
    daemon.sm.update_msgs(now_ns / 1e9, [serialized(event)])
    # Keep the driver override event out of this alert-lifecycle check.
    car_state.standstill = daemon.CS_prev.brakePressed = True
    update(daemon, car_state, now_ns)
    assert EventName.longitudinalControlUnavailable not in daemon.events.names
    daemon.update_alerts(car_state)

  car_state.brakePressed = False
  daemon.sm.frame += 1
  daemon.update_alerts(car_state)
  assert daemon.AM.current_alert.alert_type == ('' if recovered else alert_type)


@pytest.mark.parametrize('takeover', ['longitudinal', 'hold', 'holdEnded'])
def test_brake_withdraws_current_and_cached_longitudinal_takeover_prompt(takeover):
  daemon, car_state, _, now_ns = failure_harness(-4)
  update(daemon, car_state, now_ns)
  configure_alert_output(daemon)
  policy = daemon.engagement_policy = EngagementPolicy()
  policy.last_now_ns = now_ns
  policy.speed_takeover = takeover == 'longitudinal'
  policy.handoff.active = takeover == 'hold'
  policy.hold_ended_until_ns = now_ns + 1 if takeover == 'holdEnded' else 0
  daemon.enabled = takeover != 'holdEnded'
  daemon.mads.active = True
  daemon.update_alerts(car_state)
  alert_type = f'controlTakeover/{takeover}'
  assert daemon.AM.current_alert.alert_type == alert_type

  car_state.brakePressed = True
  daemon.sm.frame += 1
  daemon.update_alerts(car_state)
  assert daemon.AM.current_alert.alert_type == ''
  assert daemon.AM.current_alert.audible_alert == log.SelfdriveState.AudibleAlert.none
  assert alert_type not in daemon.AM.alerts


@pytest.mark.parametrize('other_event,event_type', [
  (EventName.commIssue, ET.SOFT_DISABLE), (EventName.steerUnavailable, ET.IMMEDIATE_DISABLE),
  (EventName.fcw, ET.PERMANENT), (EventName.aeb, ET.PERMANENT), (EventName.stockAeb, ET.PERMANENT),
])
def test_brake_suppression_keeps_independent_fault_and_collision_alerts(other_event, event_type):
  daemon, car_state, _, now_ns = failure_harness(-4)
  update(daemon, car_state, now_ns)
  configure_alert_output(daemon, event_type)
  daemon.events.add(other_event)
  car_state.brakePressed = True
  daemon.update_alerts(car_state)
  assert daemon.AM.current_alert.alert_type == f'{EVENT_NAME[other_event]}/{event_type}'


def test_driver_braking_does_not_silence_steering_takeover():
  daemon, car_state, _, _ = failure_harness(-4)
  configure_alert_output(daemon)
  daemon.events.add(EventName.steerUnavailable)
  daemon.engagement_policy = EngagementPolicy()
  daemon.engagement_policy.steering_takeover = True
  car_state.brakePressed = True
  daemon.update_alerts(car_state)
  assert daemon.AM.current_alert.alert_type == 'controlTakeover/lateral'
