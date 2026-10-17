"""Recovered perception warnings must not retain a disable or a stale banner."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import ControlsHarness
from openpilot.selfdrive.selfdrived.alertmanager import AlertManager
from openpilot.selfdrive.selfdrived.engagement_policy import LongitudinalEvents
from openpilot.selfdrive.selfdrived.events import ET, Events
from openpilot.selfdrive.selfdrived.perception_alerts import PERCEPTION_ALERT, SOURCE_ALERTS, merge_perception_alerts
from openpilot.selfdrive.selfdrived.selfdrived import SelfdriveD
from openpilot.selfdrive.selfdrived.tests.test_engagement_policy import CONTROL_STEP_NS, MadsState, NOW, make_daemon


EventName = log.OnroadEvent.EventName
State = log.SelfdriveState.OpenpilotState


def event_set(*names):
  events = Events()
  for name in names:
    events.add(name)
  return events


def raw_alerts(events, types=(ET.PERMANENT, ET.SOFT_DISABLE)):
  sm = {'modelV2': SimpleNamespace(frameDropPerc=1.47)}
  return events.create_alerts(types, [None, None, sm, True, 300, log.LongitudinalPersonality.standard])


def display(manager, events, frame, *, recovered=False, types=(ET.PERMANENT, ET.SOFT_DISABLE)):
  alerts = merge_perception_alerts(raw_alerts(events, types), events, manager, frame,
                                    recovered_model_lag=recovered)
  manager.add_many(frame, alerts)
  manager.process_alerts(frame, set())
  return manager.current_alert


def test_one_entry_replaces_both_faults_and_their_cached_banners():
  events = event_set(EventName.modeldLagging, EventName.locationdTemporaryError)
  manager = AlertManager()
  manager.add_many(0, raw_alerts(events))
  alert = display(manager, events, 1)
  assert alert.alert_type == PERCEPTION_ALERT
  assert alert.event_type == ET.SOFT_DISABLE
  assert alert.alert_text_2 == 'Model / Localization Delay'
  assert not SOURCE_ALERTS.intersection(manager.alerts)
  assert list(manager.alerts) == [PERCEPTION_ALERT]
  assert events.has(EventName.modeldLagging) and events.has(EventName.locationdTemporaryError)
  assert all(event.softDisable and event.noEntry for event in events.to_msg())


def test_recovery_replaces_two_second_takeover_cache_immediately_then_clears():
  manager = AlertManager()
  display(manager, event_set(EventName.modeldLagging, EventName.locationdTemporaryError), 0)
  assert manager.alerts[PERCEPTION_ALERT].end_frame >= 200
  recovered = display(manager, event_set(EventName.modeldLagging), 7, recovered=True,
                      types=(ET.PERMANENT, ET.WARNING))
  assert recovered.alert_type == PERCEPTION_ALERT
  assert recovered.alert_text_2 == 'Inputs recovered; monitoring frame drops'
  assert recovered.event_type == ET.PERMANENT
  assert recovered.audible_alert == log.SelfdriveState.AudibleAlert.none
  alert = display(manager, Events(), 8, recovered=True)
  assert alert.alert_type == ''
  assert PERCEPTION_ALERT not in manager.alerts


def test_recovered_model_never_downgrades_current_localization_failure():
  manager = AlertManager()
  events = event_set(EventName.modeldLagging, EventName.locationdTemporaryError)
  assert display(manager, events, 0, recovered=True).event_type == ET.SOFT_DISABLE


def test_persistent_fault_retains_takeover_minimum_after_disengagement():
  manager = AlertManager()
  events = event_set(EventName.modeldLagging)
  assert display(manager, events, 0).event_type == ET.SOFT_DISABLE
  assert display(manager, events, 1, types=(ET.PERMANENT,)).event_type == ET.SOFT_DISABLE
  assert display(manager, events, 201, types=(ET.PERMANENT,)).event_type == ET.PERMANENT


def test_no_entry_and_higher_priority_faults_remain_visible():
  events = event_set(EventName.modeldLagging)
  manager = AlertManager()
  assert display(manager, events, 0, recovered=True, types=(ET.NO_ENTRY, ET.PERMANENT)).event_type == ET.NO_ENTRY
  events.add(EventName.fcw)
  alert = display(manager, events, 1)
  assert alert.alert_type == 'fcw/permanent'
  events.remove(EventName.modeldLagging)
  alert = display(manager, events, 2)
  assert alert.alert_type == 'fcw/permanent'
  assert 'fcw/permanent' in manager.alerts
  assert PERCEPTION_ALERT not in manager.alerts


@pytest.mark.parametrize('event', [EventName.locationdTemporaryError, EventName.commIssue, EventName.commIssueAvgFreq,
                                 EventName.cameraMalfunction, EventName.sensorDataInvalid, EventName.posenetInvalid,
                                 EventName.longitudinalControlUnavailable, EventName.canError, EventName.buttonCancel,
                                 EventName.pedalPressed])
def test_recovery_view_exempts_only_historical_model_soft_disable(event):
  events = event_set(EventName.modeldLagging, event)
  view = LongitudinalEvents(events, recovered_model_lag=True)
  other = event_set(event)
  for kind in (ET.SOFT_DISABLE, ET.IMMEDIATE_DISABLE, ET.USER_DISABLE):
    assert view.contains(kind) == other.contains(kind)
  assert view.contains(ET.NO_ENTRY)
  assert events.contains(ET.SOFT_DISABLE)


def engagement_tick(daemon, CS, frame):
  now_ns = NOW + frame * CONTROL_STEP_NS
  daemon.sm.logMonoTime.update(dict.fromkeys(daemon.sm, now_ns))
  daemon.sm['longitudinalPlan'].modelMonoTime = now_ns
  daemon._car_state_mono_ns = now_ns
  daemon._longitudinal_soft_disable_remaining_ns = None
  with patch('openpilot.selfdrive.selfdrived.selfdrived.time.monotonic_ns', return_value=now_ns):
    SelfdriveD.update_engagement_state(daemon, CS)
  daemon.mads.enabled, daemon.mads.active = daemon.mads.state_machine.update()


def qualified_daemon():
  daemon, CS = make_daemon()
  daemon.events.add(EventName.modeldLagging)
  # This test isolates the source-confirmation -> FSM boundary. The observer's
  # actual frame/receipt qualification is exercised in test_model_lag_recovery.
  daemon.engagement_policy.model_lag_recovery.update = Mock(return_value=True)
  return daemon, CS


def test_confirmed_recovery_leaves_soft_disabling_and_stays_enabled_past_deadline():
  daemon, CS = qualified_daemon()
  daemon.state_machine.state = State.softDisabling
  daemon.state_machine.soft_disable_timer = 190
  daemon.mads.state_machine.state = MadsState.softDisabling
  daemon.mads.state_machine.soft_disable_timer = 190
  for frame in range(401):
    engagement_tick(daemon, CS, frame)
    assert daemon.enabled and daemon.active and daemon.mads.active
    assert daemon.state_machine.state == State.enabled
    assert daemon.mads.state_machine.state != MadsState.softDisabling
  assert daemon.events.has(EventName.modeldLagging)
  assert daemon.events.contains(ET.NO_ENTRY) and daemon.events.contains(ET.SOFT_DISABLE)
  assert not daemon.engagement_policy.steering_takeover


def test_current_localization_failure_still_expires_and_recovery_cannot_reengage():
  daemon, CS = qualified_daemon()
  daemon.events.add(EventName.locationdTemporaryError)
  for frame in range(302):
    engagement_tick(daemon, CS, frame)
  assert not daemon.enabled
  daemon.events.remove(EventName.locationdTemporaryError)
  engagement_tick(daemon, CS, 302)
  assert not daemon.enabled
  daemon.events.add(EventName.buttonEnable)
  engagement_tick(daemon, CS, 303)
  assert not daemon.enabled and ET.NO_ENTRY in daemon.state_machine.current_alert_types


@pytest.mark.parametrize('event', [EventName.buttonCancel, EventName.pedalPressed, EventName.canError,
                                 EventName.controlsMismatch, EventName.relayMalfunction, EventName.accFaulted])
def test_manual_or_immediate_fault_exits_despite_qualified_model_recovery(event):
  daemon, CS = qualified_daemon()
  daemon.events.add(event)
  engagement_tick(daemon, CS, 0)
  assert not daemon.enabled


@pytest.mark.parametrize('defect', ['inactive', 'stock_long', 'interceptor', 'wrong_platform', 'panda_denied',
                                  'can_invalid', 'unhealthy_output', 'plan_degraded'])
def test_confirmation_cannot_grant_missing_longitudinal_authority(defect):
  daemon, CS = qualified_daemon()
  if defect == 'inactive':
    daemon.enabled = False
    daemon.state_machine.state = State.disabled
  elif defect == 'stock_long':
    daemon.CP.openpilotLongitudinalControl = False
  elif defect == 'interceptor':
    daemon.CP_SP.enableGasInterceptor = True
  elif defect == 'wrong_platform':
    daemon.CP.carFingerprint = 'unsupported'
  elif defect == 'panda_denied':
    daemon.sm['pandaStates'][0].controlsAllowed = False
  elif defect == 'can_invalid':
    CS.canValid = False
  elif defect == 'unhealthy_output':
    daemon.sm.valid['carOutput'] = False
  elif defect == 'plan_degraded':
    daemon.sm['controlsState'].motionState.degraded = True
  engagement_tick(daemon, CS, 0)
  assert not daemon.engagement_policy.decision.recovered_model_lag


def test_lateral_recovery_requires_its_independent_current_inputs():
  daemon, CS = qualified_daemon()
  daemon.sm['vehicleParameters'].valid = False
  engagement_tick(daemon, CS, 0)
  assert daemon.engagement_policy.decision.recovered_model_lag
  assert not daemon.mads.model_lag_recovered
  assert daemon.mads.state_machine.state == MadsState.softDisabling


@pytest.mark.parametrize('location_fault', [False, True])
def test_actual_controlsd_uses_the_recovered_state_for_long_active_and_force_decel(location_fault):
  daemon, CS = qualified_daemon()
  daemon.state_machine.state = State.softDisabling
  daemon.state_machine.soft_disable_timer = 190
  if location_fault:
    daemon.events.add(EventName.locationdTemporaryError)
  engagement_tick(daemon, CS, 0)
  controls = ControlsHarness()
  controls.selfdrive_state.enabled = daemon.enabled
  controls.selfdrive_state.active = daemon.active
  controls.selfdrive_state.state = daemon.state_machine.state
  command, _ = controls.step(intent_fields={'ready': True})
  assert command.enabled and command.longActive
  assert controls.control.pm.messages['controlsState'].controlsState.forceDecel == location_fault
