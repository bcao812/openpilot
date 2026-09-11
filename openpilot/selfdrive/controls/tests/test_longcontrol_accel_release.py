import numpy as np
import pytest

from opendbc.car import structs
from opendbc.car.honda.values import CAR
from openpilot.cereal import custom
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.tests.test_longitudinal_cruise_coast import decoded_commands


def controller(platform=CAR.HONDA_ACCORD, openpilot_long=True, initial=-1.0):
  params = structs.CarParams()
  params.carFingerprint = platform
  params.openpilotLongitudinalControl = openpilot_long
  params.stopAccel = -2.0
  params.longitudinalTuning.kiBP = [0.0]
  params.longitudinalTuning.kiV = [0.0]
  control = LongControl(params, custom.CarParamsSP.new_message())
  control.long_control_state = LongCtrlState.pid
  control.last_output_accel = initial
  return control


@pytest.mark.parametrize('speed', [0.0, 0.5, 2.0, 15.0, 30.0])
@pytest.mark.parametrize('initial,target', [(-3.0, -0.5), (-1.0, 1.5), (-0.21, 0.2), (0.0, 2.0)])
def test_accel_release_is_bounded_even_without_a_stopping_transition(speed, initial, target):
  control = controller(initial=initial)
  state = structs.CarState()
  state.vEgo = speed
  outputs = [initial]
  for _ in range(100):
    outputs.append(float(control.update(True, state, target, False, (-3.5, 2.0))))
  assert max(np.diff(outputs)) <= 8.0 * DT_CTRL + 1e-9
  assert min(np.diff(outputs)) >= -1e-9
  assert outputs[-1] == target
  assert control.long_control_state == LongCtrlState.pid
  assert not control.starting


@pytest.mark.parametrize('initial,target', [(1.0, -3.0), (-0.21, -1.0), (-1.0, -3.5)])
def test_stronger_braking_is_applied_in_the_same_control_frame(initial, target):
  control = controller(initial=initial)
  assert control.update(True, structs.CarState(), target, False, (-3.5, 2.0)) == target


def test_an_emergency_interrupts_a_release_ramp_immediately():
  control, state = controller(initial=-1.0), structs.CarState()
  assert control.update(True, state, 1.0, False, (-3.5, 2.0)) == pytest.approx(-0.92)
  assert control.update(True, state, -3.0, False, (-3.5, 2.0)) == -3.0


def test_driver_override_does_not_keep_an_automatic_brake_applied():
  control = controller(initial=-2.0)
  assert control.update(False, structs.CarState(), -2.0, False, (-3.5, 2.0)) == 0.0
  assert control.long_control_state == LongCtrlState.off
  assert control.update(True, structs.CarState(), 1.0, False, (-3.5, 2.0)) == pytest.approx(0.08)


@pytest.mark.parametrize(
  'platform,openpilot_long', [(CAR.HONDA_CIVIC, True), (CAR.HONDA_CIVIC_BOSCH, True), (CAR.HONDA_CRV_HYBRID, True), (CAR.HONDA_ACCORD, False)]
)
def test_other_platforms_and_stock_longitudinal_are_unchanged(platform, openpilot_long):
  control = controller(platform, openpilot_long)
  assert control.update(True, structs.CarState(), 1.0, False, (-3.5, 2.0)) == 1.0


def test_existing_stopped_hold_is_not_relaxed():
  control, state = controller(initial=-2.0), structs.CarState()
  state.standstill = True
  for _ in range(200):
    assert control.update(True, state, 0.15, True, (-3.5, 2.0)) == -2.0
  assert control.long_control_state == LongCtrlState.stopping


def test_native_can_has_no_gas_brake_conflict_or_extra_switches_during_release():
  control, state = controller(initial=-1.0), structs.CarState()
  outputs = [float(control.update(True, state, 1.0, False, (-3.5, 2.0))) for _ in range(100)]
  commands = decoded_commands(outputs[::2])
  assert max(np.diff([row['ACCEL_COMMAND'] for row in commands])) <= 0.17
  assert all(not row['BRAKE_REQUEST'] or row['GAS_COMMAND'] <= 0 for row in commands)
  assert sum(before['BRAKE_REQUEST'] != after['BRAKE_REQUEST'] for before, after in zip(commands, commands[1:], strict=False)) == 1


def test_small_cruise_oscillations_below_the_existing_release_rate_are_unchanged():
  requests = [-0.2 + 0.04 * np.sin(index * np.pi / 25.0) for index in range(100)]
  control, state = controller(initial=requests[0]), structs.CarState()
  outputs = [control.update(True, state, request, False, (-3.5, 2.0)) for request in requests]
  np.testing.assert_array_equal(outputs, requests)
