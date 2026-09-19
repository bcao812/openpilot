from contextlib import ExitStack
from unittest.mock import Mock, patch

import pytest

from opendbc.car.honda.values import CAR
from openpilot.cereal import log
from openpilot.common.longitudinal_logging import CHECK_FLAGS
from openpilot.selfdrive.controls.lib.control_observation import ControlObservation, supervisor_snapshot
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_control_observation import control_payload
from openpilot.selfdrive.controls.tests.test_motion_fault_braking import native_command
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import CONTROL_SERVICES, ControlsHarness, bounded_submaster, new_event, serialized
from openpilot.selfdrive.locationd.helpers import PoseCalibrator


def observed(*, enabled=True, platform=CAR.HONDA_ACCORD):
  harness = ControlsHarness(platform=platform)
  harness.control.longitudinal_observation = ControlObservation(
    harness.control.CP, harness.control.CP_SP, enabled=enabled, clock=lambda: harness.now_ns,
  )
  return harness


def analysis(harness):
  return harness.control.pm.messages["controlsState"].controlsState.longitudinalDiagnostics.analysis


@pytest.mark.parametrize("platform,unified", [(CAR.HONDA_ACCORD, True), (CAR.HONDA_CIVIC_BOSCH, False)], ids=["unified", "legacy"])
def test_input_booleans_are_the_values_passed_to_the_real_controller(platform, unified):
  harness = observed(platform=platform)
  controller = harness.control.LoC
  with patch.object(controller, "update", wraps=controller.update) as update:
    harness.step()
  value = analysis(harness)
  assert value.version == 1
  assert value.motionInputsKnown == unified
  assert value.motionBefore.known == value.motionAfter.known == unified
  for name, field in (("plan_valid", "planValid"), ("braking_plan_valid", "brakingPlanValid"), ("standby_input", "standbyInput")):
    assert getattr(value, field) == update.call_args.kwargs.get(name, False)
  if unified:
    assert value.motionBefore.phase == "off"
    assert value.motionAfter.phase == controller.motion_supervisor.phase.text
  update.assert_called_once()


def test_snapshot_does_not_call_supervisor_predicates_or_mutate_qualification_state():
  harness = observed()
  harness.step()
  supervisor = harness.control.LoC.motion_supervisor
  supervisor._rearm_plan_time_ns = (1 << 53) + 101
  supervisor._rearm_model_time_ns = (1 << 53) + 97
  before = dict(vars(supervisor))
  with ExitStack() as stack:
    for name in ("_valid_input", "_valid_transport", "_valid_braking_input", "_latest_input_ids", "_remember_rearm_input"):
      stack.enter_context(patch.object(supervisor, name, side_effect=AssertionError("observer must only copy stored state")))
    value = supervisor_snapshot(harness.control.LoC)
  assert vars(supervisor) == before
  ids = {entry["service"]: entry for entry in value["ids"]}
  assert ids["rearmPlan"]["monoTime"] == (1 << 53) + 101
  assert ids["rearmModel"]["monoTime"] == (1 << 53) + 97
  assert ids["rearmPlan"]["idKnown"]
  assert ids["lastBrakingPlan"]["idKnown"] is False


def test_public_off_state_does_not_hide_sticky_fault_or_rearm_barrier():
  harness = observed()
  harness.step()
  harness.plan.motionIntent.valid = False
  harness.step()
  assert analysis(harness).motionAfter.degraded
  harness.car_state.gasPressed = True
  harness.step()
  value = analysis(harness)
  supervisor = harness.control.LoC.motion_supervisor
  assert harness.command.actuators.accel == 0.0
  assert value.motionAfter.phase == "off"
  assert not value.motionAfter.degraded
  assert value.motionAfter.rearmAfterFault
  assert value.motionBefore.degraded
  ids = {entry.service: entry for entry in value.motionAfter.ids}
  assert ids["rearmPlan"].monoTime == supervisor._rearm_plan_time_ns
  assert ids["rearmModel"].monoTime == supervisor._rearm_model_time_ns
  assert value.motionAfter.creepT == value.motionAfter.creepTotalT == 0.0


def test_rearm_barrier_and_live_creep_timing_are_copied_after_real_reentry():
  harness = observed()
  harness.car_state.vEgo = 0.8
  harness.plan.aTarget = -0.4
  harness.selfdrive_state.enabled = False
  harness.selfdrive_state.active = False
  harness.plan.motionIntent.allowCreep = False
  harness.step()
  inactive = analysis(harness).motionAfter
  assert inactive.known and inactive.phase == "off" and inactive.requireNewInput
  assert inactive.creepT == inactive.creepTotalT == 0.0
  assert not inactive.pendingOffCreep.known
  assert not inactive.independentCreepSample.known
  harness.selfdrive_state.enabled = True
  harness.selfdrive_state.active = True
  harness.step(new_plan=False)
  assert analysis(harness).motionAfter.waitingForFreshInput
  assert not analysis(harness).motionAfter.degraded
  harness.plan.motionIntent.allowCreep = True
  harness.step()
  value = analysis(harness).motionAfter
  supervisor = harness.control.LoC.motion_supervisor
  assert not value.waitingForFreshInput and not value.requireNewInput and not value.degraded
  assert value.creepT == supervisor.creep_t > 0.0
  assert value.creepTotalT == supervisor.creep_total_t
  assert value.creepV == supervisor.creep_v == pytest.approx(0.8)
  ids = {entry.service: entry.monoTime for entry in value.ids}
  assert ids["lastPlan"] == supervisor._last_plan_time_ns == harness.events["longitudinalPlan"].logMonoTime
  assert ids["lastModel"] == supervisor._last_model_time_ns == harness.plan.modelMonoTime
  previous_creep_time = value.creepT
  harness.step(new_plan=False)
  repeated = analysis(harness).motionAfter
  assert repeated.creepT == supervisor.creep_t > previous_creep_time
  assert {entry.service: entry.monoTime for entry in repeated.ids}["lastPlan"] == ids["lastPlan"]


@pytest.mark.parametrize("failure", [None, "begin", "controller_input", "controller_output", "end", "publish"])
def test_unified_on_off_and_observer_errors_preserve_controls_and_native_acc_command(failure):
  baseline, candidate = observed(enabled=False), observed()
  with ExitStack() as stack:
    if failure is not None:
      stack.enter_context(patch.object(candidate.control.longitudinal_observation, failure, side_effect=RuntimeError("observer")))
    scenarios = [
      (False, False, True, True, False),  # healthy OFF evidence
      (True, False, True, False, False),  # healthy wait on repeated plan
      (True, False, True, True, True),  # active fresh sample
      (True, False, False, True, True),  # real fault
      (True, True, False, True, True),  # manual override keeps sticky fault
      (True, False, True, True, True),  # fresh recovery
    ]
    for active, gas, valid, new_plan, creep in scenarios:
      for harness in (baseline, candidate):
        harness.car_state.vEgo = 0.8
        harness.car_state.gasPressed = gas
        harness.selfdrive_state.enabled = active
        harness.selfdrive_state.active = active
        harness.plan.aTarget = -0.4
        harness.plan.motionIntent.valid = valid
        harness.plan.motionIntent.allowCreep = creep
        harness.step(new_plan=new_plan)
      assert control_payload(baseline) == control_payload(candidate)
      baseline_output, baseline_can = native_command(baseline)
      candidate_output, candidate_can = native_command(candidate)
      assert baseline_output.to_dict() == candidate_output.to_dict()
      assert baseline_can == candidate_can
  if failure is not None:
    assert candidate.control.longitudinal_observation.recorder.errors > 0


def pose_harness(*, enabled=True):
  harness = observed(enabled=enabled)
  harness.control.sm = bounded_submaster([*CONTROL_SERVICES, "deviceMotion", "extrinsicsCalibration"])
  harness.control.pose_calibrator = PoseCalibrator()
  harness.step()
  return harness


def pose_inputs(timestamp, *, pitch=0.1):
  device = new_event("deviceMotion", timestamp)
  device.deviceMotion.timestamp = timestamp - 13
  device.deviceMotion.orientationNED.y = pitch
  device.deviceMotion.orientationNED.valid = True
  device.deviceMotion.angularVelocityDevice.z = 0.2
  device.deviceMotion.angularVelocityDevice.valid = True
  device.deviceMotion.inputsOK = True
  device.deviceMotion.posenetOK = True
  device.deviceMotion.sensorsOK = True
  calibration = new_event("extrinsicsCalibration", timestamp - 101)
  calibration.extrinsicsCalibration.rpyCalib = [0.0, 0.02, 0.0]
  calibration.extrinsicsCalibration.calStatus = log.ExtrinsicsCalibration.Status.calibrated
  return device, calibration


def update_pose_and_control(harness, incoming):
  harness.now_ns += 1_000_000
  harness.control.sm.update_msgs(harness.now_ns / 1e9, [serialized(event) for event in incoming])
  with patch.object(harness.control.sm, "update"):
    harness.control.update()
  with patch("openpilot.selfdrive.controls.controlsd.time.monotonic_ns", return_value=harness.now_ns), \
       patch("openpilot.selfdrive.controls.controlsd.time.monotonic", return_value=harness.now_ns / 1e9):
    command, lateral_log = harness.control.state_control()
    harness.control.publish(command, lateral_log)


def test_pose_retains_the_calibration_used_for_its_build_when_new_calibration_arrives_alone():
  harness = pose_harness()
  device, calibration = pose_inputs(harness.now_ns)
  update_pose_and_control(harness, [device, calibration])
  original = list(harness.command.orientationNED)
  value = analysis(harness).pose
  assert value.known and value.calibrationKnown and value.calibrationValid
  assert value.deviceHealthKnown and value.orientationValid and value.inputsOK
  assert not value.accelerationValid
  assert value.deviceTimestampKnown and value.deviceTimestamp == device.deviceMotion.timestamp
  first_ids = {entry.service: entry.monoTime for entry in value.inputs}
  assert first_ids == {"deviceMotion": device.logMonoTime, "extrinsicsCalibration": calibration.logMonoTime}
  calibration.logMonoTime += 500_000
  calibration.extrinsicsCalibration.rpyCalib = [0.0, -0.02, 0.0]
  update_pose_and_control(harness, [calibration])
  current_inputs = {entry.service: entry.monoTime for entry in harness.control.pm.messages["controlsState"].controlsState.longitudinalDiagnostics.inputs}
  assert current_inputs["extrinsicsCalibration"] == calibration.logMonoTime
  assert {entry.service: entry.monoTime for entry in analysis(harness).pose.inputs} == first_ids
  assert list(harness.command.orientationNED) == original
  device.logMonoTime += 1_500_000
  update_pose_and_control(harness, [device])
  assert {entry.service: entry.monoTime for entry in analysis(harness).pose.inputs}["extrinsicsCalibration"] == calibration.logMonoTime
  assert list(harness.command.orientationNED) != original


def test_unobserved_calibration_and_invalid_device_health_remain_explicit():
  harness = pose_harness()
  device, _ = pose_inputs(harness.now_ns)
  device.valid = False
  device.deviceMotion.inputsOK = False
  update_pose_and_control(harness, [device])
  value = analysis(harness).pose
  assert value.known and not value.calibrationKnown
  assert not value.calibrationStatusKnown and not value.calibrationValid
  assert value.deviceHealthKnown and not value.inputsOK
  identity = next(entry for entry in value.inputs if entry.service == "deviceMotion")
  assert identity.knownFlags & CHECK_FLAGS["valid"] and not identity.flags & CHECK_FLAGS["valid"]


@pytest.mark.parametrize("failed_method", ["pose_updated", "calibration_updated"])
def test_pose_observer_failure_never_relabels_a_new_pose_with_old_sources(failed_method):
  baseline, candidate = pose_harness(enabled=False), pose_harness()
  device, calibration = pose_inputs(candidate.now_ns)
  for harness in (baseline, candidate):
    update_pose_and_control(harness, [device, calibration])
  device.logMonoTime += 1_000_000
  calibration.logMonoTime += 1_000_000
  calibration.extrinsicsCalibration.rpyCalib = [0.0, 0.05, 0.0]
  with patch.object(candidate.control.longitudinal_observation, failed_method, side_effect=RuntimeError("observation")):
    for harness in (baseline, candidate):
      update_pose_and_control(harness, [device, calibration])
  assert control_payload(baseline) == control_payload(candidate)
  value = analysis(candidate).pose
  assert not value.known if failed_method == "pose_updated" else value.known and not value.calibrationKnown
  assert candidate.control.longitudinal_observation.recorder.errors > 0


def test_original_pose_build_error_is_not_swallowed_by_observation():
  harness = pose_harness()
  device, calibration = pose_inputs(harness.now_ns)
  error = RuntimeError("real pose failure")
  previous_count = len(harness.control.pm.history)
  with patch.object(harness.control.pose_calibrator, "build_calibrated_pose", Mock(side_effect=error)), \
       pytest.raises(RuntimeError) as caught:
    update_pose_and_control(harness, [device, calibration])
  assert caught.value is error
  assert len(harness.control.pm.history) == previous_count
