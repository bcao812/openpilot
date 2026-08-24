from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal import custom
from opendbc.car.structs import CarParams, CarState
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import (LongControl, LongCtrlState, ROLLING_STOP_ACCEL,
                                                          STOPPING_DECEL_RATE, long_control_state_trans)

ACCEL_LIMITS = (-3.5, 2.0)


class TestLongControlStateTransition(OpenpilotTestCase):

  def test_stay_stopped(self):
    CP_SP = custom.CarParamsSP.new_message()
    active = True
    current_state = LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid
    active = False
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.off

  def test_engage(self):
    CP_SP = custom.CarParamsSP.new_message()
    active = True
    current_state = LongCtrlState.off
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=True, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=True, cruise_standstill=False)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=True)
    assert next_state == LongCtrlState.stopping
    next_state = long_control_state_trans(CP_SP, active, current_state,
                             should_stop=False, brake_pressed=False, cruise_standstill=False)
    assert next_state == LongCtrlState.pid


def _long_control(stop_accel):
  CP = CarParams()
  CP.stopAccel = stop_accel
  return LongControl(CP, custom.CarParamsSP.new_message())


def _ramp(lc, standstill, seconds=10.0):
  """Hold should_stop and run the ramp; returns the command from the last call."""
  CS = CarState()
  CS.standstill = standstill
  for _ in range(int(seconds / DT_CTRL)):
    accel = lc.update(True, CS, 0.0, True, ACCEL_LIMITS)
  return accel


def _assert_settles_at(accel, floor):
  """The ramp steps by a fixed amount and stops on the first step past the floor, so it may sit
  up to one step below it -- but never further, and never short."""
  assert floor - STOPPING_DECEL_RATE * DT_CTRL <= accel <= floor, f'{accel} is not at {floor}'


class TestStoppingRamp(OpenpilotTestCase):
  """The stopping ramp runs open loop -- nothing in it reacts to the car -- so how deep it is
  allowed to get while the car is still rolling is felt directly by the driver."""

  def test_holds_at_rolling_stop_accel_while_moving(self):
    lc = _long_control(-2.0)
    assert lc.update(True, CarState(), 0.0, True, ACCEL_LIMITS) == -STOPPING_DECEL_RATE * DT_CTRL
    _assert_settles_at(_ramp(lc, standstill=False), ROLLING_STOP_ACCEL)

  def test_goes_on_to_stop_accel_once_stopped(self):
    lc = _long_control(-2.0)
    _assert_settles_at(_ramp(lc, standstill=False), ROLLING_STOP_ACCEL)
    _assert_settles_at(_ramp(lc, standstill=True), -2.0)

  def test_does_not_raise_a_deeper_command(self):
    # Coming out of the PID already braking harder than the cap: holding is right, letting go is
    # a release the driver feels as the car failing to stop.
    lc = _long_control(-2.0)
    lc.last_output_accel = -1.9
    assert _ramp(lc, standstill=False) == -1.9

  def test_a_gentle_car_is_unchanged(self):
    # max() with the car's own stopAccel: nothing here may make any car brake harder than it does
    # today, so a car that stops at -0.5 still stops at -0.5, rolling or stopped.
    lc = _long_control(-0.5)
    _assert_settles_at(_ramp(lc, standstill=False), -0.5)
    _assert_settles_at(_ramp(lc, standstill=True), -0.5)
