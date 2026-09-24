import math
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from opendbc.car.honda.values import CAR
from openpilot.cereal import log, messaging
from openpilot.common.longitudinal_logging import CHECK_FLAGS
from openpilot.selfdrive.controls.lib.control_observation import ControlObservation
from openpilot.selfdrive.controls.lib.motion_supervisor import StopReason
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import ControlsHarness, serialized


def observed_controls(*, enabled=True, unified=True):
  harness = ControlsHarness(platform=CAR.HONDA_ACCORD if unified else CAR.HONDA_CIVIC_BOSCH)
  harness.control.longitudinal_observation = ControlObservation(
    harness.control.CP, harness.control.CP_SP, enabled=enabled, clock=lambda: harness.now_ns,
  )
  return harness


def control_payload(harness):
  events = {}
  for service in ("controlsState", "carControl"):
    payload = harness.control.pm.messages[service].to_dict()
    if service == "controlsState":
      payload[service].pop("longitudinalDiagnostics", None)
    events[service] = payload
  return events


@pytest.mark.parametrize("unified", [False, True])
def test_actual_control_inputs_and_published_command_have_distinct_exact_identities(unified):
  harness = observed_controls(unified=unified)
  timestamp = (1 << 53) + 101_000_001
  snapshots = {}
  incoming = []
  for index, (service, event) in enumerate(harness.events.items()):
    event.logMonoTime = timestamp - 100_000 + index * 1_003
    snapshots[service] = event.logMonoTime
    incoming.append(serialized(event))
  harness.plan.modelMonoTime = timestamp - 100_001
  incoming = [serialized(event) for event in harness.events.values()]
  harness.control.sm.update_msgs(timestamp / 1e9, incoming)
  observer = harness.control.longitudinal_observation
  observer.recorder.clock = Mock(side_effect=[timestamp + 101, timestamp + 509])
  update = Mock(wraps=harness.control.LoC.update)
  harness.control.LoC.update = update
  original_new_message = messaging.new_message

  def new_message(service, *arguments, **keywords):
    return original_new_message(service, *arguments, **dict(keywords, logMonoTime={
      "controlsState": timestamp + 701, "carControl": timestamp + 907,
    }[service]))

  with patch("openpilot.selfdrive.controls.controlsd.time.monotonic_ns", return_value=timestamp + 307), \
       patch("openpilot.selfdrive.controls.controlsd.messaging.new_message", side_effect=new_message):
    command, lateral_log = harness.control.state_control()
    harness.control.publish(command, lateral_log)

  state_event = harness.control.pm.messages["controlsState"]
  command_event = harness.control.pm.messages["carControl"]
  diagnostics = state_event.controlsState.longitudinalDiagnostics
  assert diagnostics.context.version == 1
  assert diagnostics.context.producer == "controlsd"
  assert diagnostics.context.startMonoTime == timestamp + 101
  assert diagnostics.context.endMonoTime == timestamp + 509
  assert diagnostics.context.frameKnown
  assert diagnostics.context.frame == harness.control.sm.frame
  assert diagnostics.carControlMonoTime == command_event.logMonoTime == timestamp + 907
  assert state_event.logMonoTime == timestamp + 701
  assert diagnostics.decisionMonoTimeKnown == unified
  assert diagnostics.decisionMonoTime == (timestamp + 307 if unified else 0)
  assert diagnostics.context.endMonoTime < state_event.logMonoTime < diagnostics.carControlMonoTime
  assert [service for service, _ in harness.control.pm.history] == ["controlsState", "carControl"]
  assert {entry.service: entry.monoTime for entry in diagnostics.inputs} == snapshots
  assert len(set(snapshots.values())) == len(snapshots)
  assert all(entry.idKnown and entry.kind == "consumerSnapshot" for entry in diagnostics.inputs)
  assert all(entry.knownFlags & CHECK_FLAGS["valid"] for entry in diagnostics.inputs)
  assert diagnostics.controllerCalled
  assert diagnostics.stateBefore == "off"
  assert diagnostics.stateAfter == command_event.carControl.actuators.longControlState
  assert diagnostics.targetAccel == update.call_args.args[2]
  assert (diagnostics.accelMin, diagnostics.accelMax) == tuple(update.call_args.args[4])
  assert diagnostics.longActiveAtController == update.call_args.args[0]
  assert diagnostics.longActivePublished == command_event.carControl.longActive
  assert diagnostics.publishedAccel == command_event.carControl.actuators.accel
  assert diagnostics.resumePublished == command_event.carControl.cruiseControl.resume
  assert not diagnostics.nonfiniteActuators
  update.assert_called_once()
  if unified:
    assert diagnostics.decisionMonoTime == update.call_args.kwargs["now_ns"]


@pytest.mark.parametrize("unified", [False, True])
def test_automatic_observation_matches_the_vehicle_scope(unified):
  harness = ControlsHarness(platform=CAR.HONDA_ACCORD if unified else CAR.HONDA_CIVIC_BOSCH)
  observer = ControlObservation.create(harness.control.CP, harness.control.CP_SP)
  assert observer.recorder.enabled == unified
  observer.recorder.clock = lambda: harness.now_ns
  harness.control.longitudinal_observation = observer
  harness.step()
  diagnostics = harness.control.pm.messages["controlsState"].controlsState.longitudinalDiagnostics
  assert diagnostics.context.version == (1 if unified else 0)
  assert diagnostics.unified == unified


@pytest.mark.parametrize("configuration", ["other_car", "stock_long", "interceptor"])
def test_automatic_observation_remains_scoped(configuration):
  harness = ControlsHarness()
  if configuration == "other_car":
    harness.control.CP.carFingerprint = CAR.HONDA_CIVIC
  elif configuration == "stock_long":
    harness.control.CP.openpilotLongitudinalControl = False
  else:
    harness.control.CP_SP.enableGasInterceptor = True
  assert not ControlObservation.create(harness.control.CP, harness.control.CP_SP).recorder.enabled


@pytest.mark.parametrize("unified", [False, True])
def test_on_off_observation_preserves_all_control_fields_across_state_changes(unified):
  disabled = observed_controls(enabled=False, unified=unified)
  enabled = observed_controls(unified=unified)
  scenarios = (
    (10.0, 0.5, False, False, False, 0),
    (0.5, -2.5, True, False, False, 0),
    (0.0, -0.2, True, False, False, 0),
    (0.0, 0.5, False, False, False, 0),
    (0.0, 0.5, False, True, False, 0),
    (0.0, -1.5, True, False, True, 0),
    (8.0, -2.0, False, False, False, 4),
    (8.0, 0.5, False, False, False, 0),
  )
  for speed, target, stopping, gas, brake, status in scenarios:
    for harness in (disabled, enabled):
      harness.car_state.vEgo = speed
      harness.car_state.standstill = speed == 0.0
      harness.car_state.cruiseState.standstill = speed == 0.0
      harness.car_state.gasPressed = gas
      harness.car_state.brakePressed = brake
      harness.plan.aTarget = target
      harness.plan.shouldStop = stopping
      harness.plan.solverStatus = status
      harness.plan.motionIntent.reasons = int(StopReason.MPC) if stopping else 0
      harness.plan.motionIntent.releaseAllowed = not stopping
      harness.events["longitudinalPlan"].valid = status == 0
      harness.step()
    assert control_payload(disabled) == control_payload(enabled)
    assert enabled.control.pm.messages["controlsState"].controlsState.longitudinalDiagnostics.context.version == 1
    assert disabled.control.pm.messages["controlsState"].controlsState.longitudinalDiagnostics.context.version == 0


def test_final_driver_monitoring_assignment_is_copied_before_command_is_sent():
  harness = observed_controls()
  harness.events["driverMonitoringState"].driverMonitoringState.noResponseForceDecel = True
  harness.step()
  controls = harness.control.pm.messages["controlsState"].controlsState
  assert controls.forceDecel
  assert harness.command.driverMonitoringEscalation
  assert controls.longitudinalDiagnostics.carControlMonoTime == harness.control.pm.messages["carControl"].logMonoTime


@pytest.mark.parametrize("field", ["accel", "torque", "steeringAngleDeg"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_flag_records_actual_sanitization_without_changing_it(field, value):
  disabled = observed_controls(enabled=False)
  enabled = observed_controls()
  for harness in (disabled, enabled):
    if field == "accel":
      harness.control.LoC.update = Mock(return_value=value)
    else:
      lateral_log = harness.control.LaC.update.return_value[2]
      harness.control.LaC.update.return_value = (value if field == "torque" else 0.0,
                                                 value if field == "steeringAngleDeg" else 0.0, lateral_log)
    with patch("openpilot.selfdrive.controls.controlsd.cloudlog.error"):
      harness.step()
  assert control_payload(disabled) == control_payload(enabled)
  diagnostics = enabled.control.pm.messages["controlsState"].controlsState.longitudinalDiagnostics
  assert diagnostics.nonfiniteActuators
  assert getattr(enabled.command.actuators, field) == 0.0
  if field == "accel":
    assert not math.isfinite(diagnostics.controllerAccel)
    assert diagnostics.publishedAccel == 0.0


@pytest.mark.parametrize("method", ["begin", "controller_input", "controller_output", "end", "publish"])
def test_observation_failures_leave_original_control_output_unchanged(method):
  baseline = observed_controls(enabled=False)
  failing = observed_controls()
  with patch.object(failing.control.longitudinal_observation, method, side_effect=RuntimeError("observation failed")):
    baseline.step()
    failing.step()
  assert control_payload(baseline) == control_payload(failing)
  assert failing.control.longitudinal_observation.recorder.errors >= 1


def test_sanitization_observation_failure_does_not_prevent_zeroing_the_original_actuator():
  baseline = observed_controls(enabled=False)
  failing = observed_controls()
  for harness in (baseline, failing):
    harness.control.LoC.update = Mock(return_value=float("nan"))
  with patch.object(failing.control.longitudinal_observation, "sanitized", side_effect=RuntimeError("observer")), \
       patch("openpilot.selfdrive.controls.controlsd.cloudlog.error"):
    baseline.step()
    failing.step()
  assert control_payload(baseline) == control_payload(failing)
  assert failing.command.actuators.accel == 0.0
  assert failing.control.longitudinal_observation.recorder.errors == 1


def test_diagnostic_serialization_error_does_not_abort_original_messages():
  baseline = observed_controls(enabled=False)
  failing = observed_controls()
  baseline.step()
  with patch.object(log, "LongitudinalControlDiagnostics", SimpleNamespace(new_message=Mock(side_effect=RuntimeError("serialize")))):
    failing.step()
  assert control_payload(baseline) == control_payload(failing)
  assert [service for service, _ in failing.control.pm.history] == ["controlsState", "carControl"]
  assert failing.control.longitudinal_observation.recorder.errors == 1


def test_input_snapshot_serialization_error_does_not_abort_original_messages():
  baseline = observed_controls(enabled=False)
  failing = observed_controls()
  observer = failing.control.longitudinal_observation
  baseline.step()
  with patch.object(observer.input_cache, "_new_message", side_effect=RuntimeError("input serializer")):
    failing.step()
  assert control_payload(baseline) == control_payload(failing)
  assert [service for service, _ in failing.control.pm.history] == ["controlsState", "carControl"]
  assert failing.control.pm.messages["controlsState"].controlsState.longitudinalDiagnostics.context.version == 0
  assert observer.recorder.errors == 1
  assert observer.input_cache._entries == {}

  baseline.step()
  failing.step()
  assert control_payload(baseline) == control_payload(failing)
  assert failing.control.pm.messages["controlsState"].controlsState.longitudinalDiagnostics.context.version == 1
  assert observer.recorder.errors == 1


def test_control_exception_is_not_swallowed_or_replaced_by_observation():
  harness = observed_controls()
  error = RuntimeError("original control error")
  harness.control.LoC.update = Mock(side_effect=error)
  with pytest.raises(RuntimeError) as caught:
    harness.step()
  assert caught.value is error
  assert harness.control.pm.history == []


def test_successive_publications_without_new_control_do_not_reuse_a_completed_sample():
  harness = observed_controls()
  harness.step()
  previous = harness.control.pm.messages["carControl"]
  harness.control.publish(previous.carControl.as_builder(), harness.control.LaC.update.return_value[2])
  assert harness.control.pm.messages["controlsState"].controlsState.longitudinalDiagnostics.context.version == 0


def test_health_flags_preserve_invalid_and_unknown_as_different_observations():
  harness = observed_controls()
  harness.step()
  observer = harness.control.longitudinal_observation
  first_inputs = {entry.service: entry for entry in observer.inputs}
  first_values = {name: entry.to_dict() for name, entry in first_inputs.items()}
  timestamps = dict(harness.control.sm.logMonoTime)
  observer.begin(harness.control.sm)
  assert all(entry is first_inputs[entry.service] for entry in observer.inputs)
  harness.control.sm.valid["longitudinalPlan"] = False
  del harness.control.sm.freq_ok["carState"]
  with patch("openpilot.selfdrive.controls.controlsd.time.monotonic_ns", return_value=harness.now_ns):
    command, lateral_log = harness.control.state_control()
    harness.control.publish(command, lateral_log)
  event = harness.control.pm.messages["controlsState"]
  inputs = {entry.service: entry for entry in event.controlsState.longitudinalDiagnostics.inputs}
  assert inputs["longitudinalPlan"].knownFlags & CHECK_FLAGS["valid"]
  assert not inputs["longitudinalPlan"].flags & CHECK_FLAGS["valid"]
  assert not inputs["carState"].knownFlags & CHECK_FLAGS["freq_ok"]
  assert event.valid
  assert harness.control.sm.logMonoTime == timestamps
  assert {name: entry.to_dict() for name, entry in first_inputs.items()} == first_values
  assert all(entry is not first_inputs[entry.service] for entry in observer.inputs
             if entry.service in ("longitudinalPlan", "carState"))
