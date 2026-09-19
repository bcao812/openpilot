import numpy as np
import pytest

from opendbc.car import structs
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent, StopReason
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import decoded_commands
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import ControlHarness, controller, warm_tracking_harness


@pytest.mark.parametrize('speed', [0.0, 0.5, 2.0, 15.0, 30.0])
@pytest.mark.parametrize('initial,target', [(-3.0, -0.5), (-1.0, 1.5), (-0.21, 0.2), (0.0, 2.0)])
def test_accel_release_is_bounded_even_without_a_stopping_transition(speed, initial, target):
  harness = warm_tracking_harness(speed, initial)
  outputs = [initial]
  for _ in range(100):
    outputs.append(harness.step(target, MotionIntent(allow_creep=False)))
  assert max(np.diff(outputs)) <= 8.0 * DT_CTRL + 1e-9
  assert min(np.diff(outputs)) >= -1e-9
  assert outputs[-1] == target
  assert harness.control.long_control_state == LongCtrlState.pid
  assert not harness.control.starting


@pytest.mark.parametrize('initial,target', [(1.0, -3.0), (-0.21, -1.0), (-1.0, -3.5)])
def test_stronger_braking_is_applied_in_the_same_control_frame(initial, target):
  harness = warm_tracking_harness(0.0, initial)
  assert harness.step(target, MotionIntent(allow_creep=False)) == target


def test_an_emergency_interrupts_a_release_ramp_immediately():
  harness = warm_tracking_harness(0.0, -1.0)
  assert harness.step(1.0, MotionIntent(allow_creep=False)) == pytest.approx(-0.92)
  assert harness.step(-3.0, MotionIntent(allow_creep=False)) == -3.0


def test_driver_override_does_not_keep_an_automatic_brake_applied():
  harness = warm_tracking_harness(0.0, -2.0)
  assert harness.step(-2.0, active=False) == 0.0
  assert harness.control.long_control_state == LongCtrlState.off
  assert harness.step(1.0, MotionIntent(allow_creep=False)) == pytest.approx(0.08)


@pytest.mark.parametrize(
  'platform,openpilot_long', [(CAR.HONDA_CIVIC, True), (CAR.HONDA_CIVIC_BOSCH, True), (CAR.HONDA_CRV_HYBRID, True), (CAR.HONDA_ACCORD, False)]
)
def test_other_platforms_and_stock_longitudinal_are_unchanged(platform, openpilot_long):
  params = CarInterface.get_non_essential_params(platform)
  params.openpilotLongitudinalControl = openpilot_long
  extension = CarInterface.get_non_essential_params_sp(params, platform)
  control = LongControl(params, extension)
  assert not control.unified
  control.long_control_state = LongCtrlState.pid
  control.last_output_accel = -1.0
  state = structs.CarState()
  # Keep each vehicle's real integral gain: the first tick integrates the
  # unit tracking error, without the Accord owner's upward slew constraint.
  integral_gain = float(np.interp(state.vEgo, params.longitudinalTuning.kiBP, params.longitudinalTuning.kiV))
  assert control.update(True, state, 1.0, False, (-3.5, 2.0)) == pytest.approx(1.0 + DT_CTRL * integral_gain)


def test_existing_stopped_hold_is_not_relaxed():
  harness = ControlHarness(controller(initial=-2.0))
  harness.state.vEgo = 0.0
  harness.state.standstill = True
  for _ in range(200):
    assert harness.step(0.15, MotionIntent(reasons=StopReason.MPC)) == -2.0
  assert harness.control.long_control_state == LongCtrlState.stopping


def test_native_can_has_no_gas_brake_conflict_or_extra_switches_during_release():
  harness = warm_tracking_harness(0.0, -1.0)
  outputs = [harness.step(1.0, MotionIntent(allow_creep=False)) for _ in range(100)]
  commands = decoded_commands(outputs[::2])
  assert max(np.diff([row['ACCEL_COMMAND'] for row in commands])) <= 0.17
  assert all(not row['BRAKE_REQUEST'] or row['GAS_COMMAND'] <= 0 for row in commands)
  assert sum(before['BRAKE_REQUEST'] != after['BRAKE_REQUEST'] for before, after in zip(commands, commands[1:], strict=False)) == 1


def test_small_cruise_oscillations_below_the_existing_release_rate_are_unchanged():
  requests = [-0.2 + 0.04 * np.sin(index * np.pi / 25.0) for index in range(100)]
  harness = warm_tracking_harness(0.0, requests[0])
  outputs = [harness.step(request, MotionIntent(allow_creep=False)) for request in requests]
  np.testing.assert_array_equal(outputs, requests)
