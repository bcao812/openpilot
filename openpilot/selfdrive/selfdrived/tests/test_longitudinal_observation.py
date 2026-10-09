from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from opendbc.car.honda.values import CAR
from openpilot.cereal import custom, log
from openpilot.common.longitudinal_logging import CHECK_FLAGS, LOGGING_MARKERS
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.controls.lib.control_observation import ControlObservation
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.tests.test_motion_rearm_wait import enter_off
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import (
  CapturingPublisher, ControlsHarness, new_event, selfdrive_event_harness, serialized,
)
from openpilot.selfdrive.selfdrived.alertmanager import AlertManager
from openpilot.selfdrive.selfdrived.events import Events, ET
from openpilot.selfdrive.selfdrived.longitudinal_observation import SelfdriveObservation
from openpilot.selfdrive.selfdrived.state import SOFT_DISABLE_TIME, StateMachine
from openpilot.sunnypilot.selfdrive.selfdrived.button_state_tracker import ButtonStateTracker


State = log.SelfdriveState.OpenpilotState
EventName = log.OnroadEvent.EventName
MISSING = object()


@pytest.fixture(autouse=True)
def isolated_params():
  with OpenpilotPrefix():
    yield


def observed_selfdrive(control_state=None, *, enabled=True):
  if control_state is None:
    harness = ControlsHarness()
    harness.step()
    control_state = harness.control.pm.messages["controlsState"]
  daemon, car_state = selfdrive_event_harness(control_state)
  daemon.CP.carFingerprint = CAR.HONDA_ACCORD
  daemon.CP_SP = custom.CarParamsSP.new_message()
  daemon.longitudinal_observation = (SelfdriveObservation.create(daemon.CP, daemon.CP_SP) if enabled is None else
                                   SelfdriveObservation(daemon.CP, daemon.CP_SP, enabled=enabled))
  if daemon.longitudinal_observation is not None:
    daemon.longitudinal_observation.recorder.clock = lambda: 20_000_000_000
  daemon.active = True
  daemon.state_machine = StateMachine()
  daemon.state_machine.state = State.enabled
  daemon.prof = None
  daemon.car_state_sock = object()
  daemon.pm = CapturingPublisher()
  daemon.AM = AlertManager()
  daemon.personality = log.LongitudinalPersonality.standard
  daemon.button_state_tracker = ButtonStateTracker()
  daemon.events_prev = []
  daemon.events_sp_prev = []
  daemon.mads = SimpleNamespace(enabled=False, active=False, enabled_toggle=False,
                               state_machine=SimpleNamespace(state="disabled"), update=Mock())
  daemon.icbm.state = "inactive"
  daemon.icbm.cruise_button = "none"
  daemon.icbm.v_target = 0.0
  event = new_event("carState", 1_999_999_901)
  event.carState = car_state
  return daemon, event


def step(daemon, car_state_message=MISSING):
  if car_state_message is MISSING:
    car_state_message = new_event("carState", 2_000_000_001)
    car_state_message.carState = daemon.CS_prev
  with patch("openpilot.selfdrive.selfdrived.selfdrived.messaging.recv_one", return_value=car_state_message), \
       patch.object(daemon.sm, "update"), \
       patch("openpilot.selfdrive.selfdrived.selfdrived.time.monotonic_ns", return_value=20_000_000_000), \
       patch("openpilot.selfdrive.selfdrived.selfdrived.time.monotonic", return_value=20.0):
    daemon.step()
  return daemon.pm.messages["selfdriveState"].selfdriveState.longitudinalDiagnostics


def payload(daemon):
  result = {}
  for service, message in daemon.pm.messages.items():
    value = message.to_dict()
    if service == "selfdriveState":
      value[service].pop("longitudinalDiagnostics", None)
    result[service] = value
  return result


def test_real_motion_fault_branch_state_machine_alert_and_next_force_decel_are_observed():
  controls = ControlsHarness()
  controls.control.longitudinal_observation = ControlObservation(
    controls.control.CP, controls.control.CP_SP, enabled=True, clock=lambda: controls.now_ns,
  )
  controls.plan.motionIntent.valid = False
  controls.step()
  daemon, raw = observed_selfdrive(controls.control.pm.messages["controlsState"])
  value = step(daemon, raw)
  assert value.context.version == value.version == 1
  assert value.context.producer == "selfdrived"
  assert value.stateMachineInputKnown and value.stateMachineCalled
  assert value.stateBefore == "enabled" and value.stateAfter == value.statePublished == "softDisabling"
  assert value.enabledBefore and value.activeBefore and value.enabledAfter and value.activeAfter
  assert value.softDisableTimerBefore == 0
  assert value.softDisableTimerAfter == round(SOFT_DISABLE_TIME / 0.01)
  assert value.motionCheckKnown and value.motionDegraded and not value.motionCommIssueAdded
  assert value.longitudinalControlUnavailableAdded
  assert list(value.events) == daemon.events.names == [EventName.longitudinalControlUnavailable]
  assert list(value.currentAlertTypes) == daemon.state_machine.current_alert_types
  assert ET.SOFT_DISABLE in value.currentAlertTypes
  assert not value.controlsForceDecel and value.controlsForceDecelKnown
  assert daemon.pm.messages["selfdriveState"].selfdriveState.alertType == "longitudinalControlUnavailable/softDisable"
  controls.events["selfdriveState"] = daemon.pm.messages["selfdriveState"].as_builder()
  controls.step()
  actual = controls.control.pm.messages["controlsState"].controlsState
  assert actual.forceDecel and controls.command.driverMonitoringEscalation
  assert actual.longitudinalDiagnostics.analysis.forceDecel
  assert actual.longitudinalDiagnostics.analysis.selfdriveSoftDisabling


@pytest.mark.parametrize('mode', [LongitudinalMode.UNIFIED], ids=lambda mode: mode.name)
def test_healthy_wait_does_not_claim_a_motion_comm_issue(mode):
  controls = enter_off(mode)
  controls.step(new_plan=False)
  daemon, raw = observed_selfdrive(controls.control.pm.messages["controlsState"])
  value = step(daemon, raw)
  assert value.motionCheckKnown and value.motionWaitingForFreshInput
  assert not value.motionDegraded and not value.motionCommIssueAdded
  assert not value.longitudinalControlUnavailableAdded
  assert list(value.events) == []
  assert value.stateAfter == value.statePublished == "enabled"


def test_an_independent_communication_event_is_not_attributed_to_the_motion_branch():
  daemon, raw = observed_selfdrive()
  daemon.sm.alive["modelV2"] = False
  value = step(daemon, raw)
  assert EventName.commIssue in value.events
  assert value.motionCheckKnown and not value.motionCommIssueAdded and not value.motionDegraded
  assert not value.longitudinalControlUnavailableAdded
  assert value.stateAfter == "softDisabling"


def test_exact_ids_above_float_precision_include_direct_socket_carstate_and_all_subscriptions():
  daemon, raw = observed_selfdrive()
  timestamp = (1 << 53) + 1_000_001
  expected = {}
  incoming = []
  for index, service in enumerate(daemon.sm.logMonoTime):
    event = new_event(service, timestamp + index * 101)
    setattr(event, service, daemon.sm[service])
    expected[service] = event.logMonoTime
    incoming.append(serialized(event))
  daemon.sm.update_msgs(2.1, incoming)
  daemon.sm.updated = dict.fromkeys(daemon.sm.updated, False)
  daemon.sm.recv_frame = dict.fromkeys(daemon.sm.recv_frame, 0)
  raw.logMonoTime = timestamp - 103
  value = step(daemon, serialized(raw))
  identities = {entry.service: entry for entry in value.inputs}
  assert {name: entry.monoTime for name, entry in identities.items()} == {**expected, "carState": raw.logMonoTime}
  assert all(identities[name].idKnown for name in ("carState", "controlsState", "carControl", "pandaStates"))
  assert identities["carState"].flags & CHECK_FLAGS["updated"]
  assert not identities["carState"].knownFlags & CHECK_FLAGS["freq_ok"]
  assert not value.carStateReused


def test_socket_timeout_retains_the_exact_previously_consumed_id_without_claiming_new_data():
  daemon, raw = observed_selfdrive()
  raw.logMonoTime = (1 << 53) + 197
  value = step(daemon, serialized(raw))
  first = next(entry for entry in value.inputs if entry.service == "carState")
  value = step(daemon, None)
  reused = next(entry for entry in value.inputs if entry.service == "carState")
  assert value.carStateReused and reused.idKnown
  assert reused.monoTime == first.monoTime == raw.logMonoTime
  assert reused.knownFlags & CHECK_FLAGS["updated"]
  assert not reused.flags & CHECK_FLAGS["updated"]


def test_cached_subscriptions_refresh_health_at_unchanged_ids_and_keep_old_wire_snapshot():
  daemon, raw = observed_selfdrive()
  observer = daemon.longitudinal_observation
  first = step(daemon, serialized(raw))
  before = first.to_dict()
  assert observer.input_cache._entries["controlsState"][1] is None
  timestamps = dict(daemon.sm.logMonoTime)
  step(daemon, None)
  original = observer.input_cache._entries["controlsState"][1]
  original_values = original.to_dict()
  step(daemon, None)
  assert observer.input_cache._entries["controlsState"][1] is original

  daemon.sm.valid = {**daemon.sm.valid, "controlsState": False}
  daemon.sm.freq_ok.pop("controlsState", None)
  changed = step(daemon, None)
  assert observer.input_cache._entries["controlsState"][1] is None
  inputs = {entry.service: entry for entry in changed.inputs}
  current = inputs["controlsState"]
  assert current is not original
  assert current.monoTime == original.monoTime == timestamps["controlsState"]
  assert current.knownFlags & CHECK_FLAGS["valid"]
  assert not current.flags & CHECK_FLAGS["valid"]
  assert not current.knownFlags & CHECK_FLAGS["freq_ok"]
  step(daemon, None)
  promoted = observer.input_cache._entries["controlsState"][1]
  assert promoted is not original
  assert promoted.to_dict() == current.to_dict()
  assert original.to_dict() == original_values
  assert first.to_dict() == before
  assert daemon.sm.logMonoTime == timestamps
  assert changed.inputs[0].service == "carState"
  assert changed.carStateReused and not changed.inputs[0].flags & CHECK_FLAGS["updated"]


def test_missed_observation_of_a_receive_does_not_assign_a_stale_id_to_reused_carstate():
  daemon, first = observed_selfdrive()
  step(daemon, first)
  second = first.as_reader().as_builder()
  second.logMonoTime += 107
  second.carState.vEgo = 20.0
  with patch.object(daemon.longitudinal_observation, "begin", side_effect=RuntimeError("observer")):
    step(daemon, serialized(second))
  value = step(daemon, None)
  reused = next(entry for entry in value.inputs if entry.service == "carState")
  assert value.carStateReused and not reused.idKnown
  assert reused.monoTime == 0


@pytest.mark.parametrize("early", ["passive", "initializing"])
def test_skipped_decision_branches_are_unknown_instead_of_reported_healthy(early):
  daemon, raw = observed_selfdrive()
  if early == "passive":
    daemon.CP.passive = True
    daemon.cp_passive = True
    value = step(daemon, raw)
  else:
    daemon.initialized = False
    raw.carState.canValid = False
    with patch.object(daemon.sm, "all_checks", return_value=False):
      value = step(daemon, raw)
  assert value.context.version == value.version == 1
  assert value.stateMachineInputKnown and not value.stateMachineCalled
  assert not value.motionCheckKnown and not value.controlsForceDecelKnown
  assert not value.motionCommIssueAdded


@pytest.mark.parametrize("failure", [None, "begin", "motion_check", "longitudinal_control_unavailable_added", "state_machine_input",
                                    "state_machine_output", "publish"])
def test_observer_enable_disable_and_failure_preserve_actual_events_states_alerts_and_publications(failure, tmp_path):
  control = ControlsHarness()
  control.plan.motionIntent.valid = False
  control.step()
  marker = tmp_path / "selfdrived_logging"
  with patch.dict(LOGGING_MARKERS, {"selfdrived": marker}):
    baseline, raw = observed_selfdrive(control.control.pm.messages["controlsState"], enabled=None)
    marker.touch()
    candidate, _ = observed_selfdrive(control.control.pm.messages["controlsState"], enabled=None)
  assert baseline.longitudinal_observation is None
  assert candidate.longitudinal_observation.recorder.enabled
  with ExitStack() as stack:
    if failure is not None:
      stack.enter_context(patch.object(candidate.longitudinal_observation, failure, side_effect=RuntimeError("observer")))
    for _ in range(3):
      with patch("openpilot.selfdrive.selfdrived.longitudinal_observation.capture_inputs") as inputs:
        assert step(baseline, serialized(raw)).context.version == 0
      inputs.assert_not_called()
      diagnostics = step(candidate, serialized(raw))
      if failure is None:
        assert diagnostics.context.version == 1
      assert payload(baseline) == payload(candidate)
      assert baseline.events.names == candidate.events.names
      assert baseline.state_machine.state == candidate.state_machine.state
      assert baseline.state_machine.soft_disable_timer == candidate.state_machine.soft_disable_timer
  if failure is not None:
    assert candidate.longitudinal_observation.recorder.errors > 0


@pytest.mark.parametrize("event,state", [(EventName.gasPressedOverride, "overriding"), (EventName.pedalPressed, "disabled"),
                                         (EventName.buttonCancel, "disabled")])
def test_actual_driver_override_or_disengagement_events_are_preserved_each_cycle(event, state):
  baseline, raw = observed_selfdrive(enabled=False)
  candidate, _ = observed_selfdrive()
  for daemon in (baseline, candidate):
    events = Events()
    events.add(event)
    daemon.car_events.update = lambda *_, events=events: events
  step(baseline, serialized(raw))
  value = step(candidate, serialized(raw))
  assert payload(baseline) == payload(candidate)
  assert event in value.events
  assert value.stateAfter == state
  assert not value.motionCommIssueAdded


@pytest.mark.parametrize("method", ["update_events", "update"])
def test_original_decision_errors_still_propagate_without_publication(method):
  daemon, raw = observed_selfdrive()
  error = RuntimeError("original consumer failure")
  owner = daemon if method == "update_events" else daemon.state_machine
  with patch.object(owner, method, side_effect=error), pytest.raises(RuntimeError) as caught:
    step(daemon, raw)
  assert caught.value is error
  assert daemon.pm.history == []


def test_observer_serialization_failure_does_not_abort_real_selfdrive_publication():
  baseline, raw = observed_selfdrive(enabled=False)
  candidate, _ = observed_selfdrive()
  step(baseline, serialized(raw))
  with patch.object(log, "LongitudinalSelfdriveDiagnostics", SimpleNamespace(new_message=Mock(side_effect=RuntimeError("wire")))):
    step(candidate, serialized(raw))
  assert payload(baseline) == payload(candidate)
  assert candidate.longitudinal_observation.recorder.errors == 1


def test_input_snapshot_serialization_failure_does_not_abort_real_selfdrive_publication():
  baseline, raw = observed_selfdrive(enabled=False)
  candidate, _ = observed_selfdrive()
  observer = candidate.longitudinal_observation
  step(baseline, serialized(raw))
  step(candidate, serialized(raw))
  assert observer.recorder.errors == 0
  previous = observer.input_cache._entries
  assert previous and all(entry[1] is None for entry in previous.values())
  step(baseline, None)
  with patch.object(observer.input_cache, "_new_message", side_effect=RuntimeError("input serializer")):
    step(candidate, None)
  assert payload(baseline) == payload(candidate)
  assert candidate.pm.messages["selfdriveState"].selfdriveState.longitudinalDiagnostics.version == 0
  assert observer.recorder.errors == 1
  assert observer.input_cache._entries is previous

  step(baseline, None)
  value = step(candidate, None)
  assert payload(baseline) == payload(candidate)
  assert value.context.version == value.version == 1
  assert value.carStateReused and value.inputs[0].idKnown
  assert all(entry[1] is not None for entry in observer.input_cache._entries.values())
  assert observer.recorder.errors == 1


def test_publication_without_another_decision_does_not_reuse_a_previous_snapshot():
  daemon, raw = observed_selfdrive()
  step(daemon, raw)
  daemon.publish_selfdriveState(daemon.CS_prev)
  assert daemon.pm.messages["selfdriveState"].selfdriveState.longitudinalDiagnostics.version == 0
