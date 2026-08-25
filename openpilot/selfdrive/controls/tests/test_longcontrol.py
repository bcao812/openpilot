from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal import custom
from opendbc.car.structs import CarParams, CarState
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import (LongControl, LongCtrlState, ROLLING_STOP_ACCEL,
                                                          STARTING_ACCEL_RATE, STOPPING_DECEL_RATE,
                                                          long_control_state_trans)

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


def _stopped_long_control(stop_accel=-2.0):
  """A LongControl sitting at the bottom of the stopping ramp, with a pure feedforward PID so the
  climb out is the rate limiter and nothing else."""
  CP = CarParams()
  CP.stopAccel = stop_accel
  CP.longitudinalTuning.kiBP = [0.]
  CP.longitudinalTuning.kiV = [0.]     # k_i = 0, so the pid output is exactly a_target
  lc = LongControl(CP, custom.CarParamsSP.new_message())
  CS = CarState()
  CS.standstill = True
  for _ in range(int(10.0 / DT_CTRL)):
    lc.update(True, CS, 0.0, True, ACCEL_LIMITS)
  assert lc.long_control_state == LongCtrlState.stopping
  return lc, CS


def _release(lc, CS, a_target, frames):
  """Drop should_stop and drive the pid branch, returning every command it produced."""
  return [lc.update(True, CS, a_target, False, ACCEL_LIMITS) for _ in range(frames)]


class TestStartingRamp(OpenpilotTestCase):
  """Leaving LongCtrlState.stopping is a step, not a ramp: last_output_accel is at the bottom of
  the open loop ramp and the pid's first output is essentially a_target, because the integrator was
  reset on every stopping frame and k_p is zero. Honda Bosch rate-limits neither accel nor gas, so
  that step clears BRAKE_REQUEST and raises GAS_COMMAND inside one CAN frame."""

  def test_the_step_out_of_a_stop_is_bounded(self):
    lc, CS = _stopped_long_control()
    bottom = lc.last_output_accel
    _assert_settles_at(bottom, -2.0)

    first = _release(lc, CS, 0.5, 1)[0]
    assert first <= bottom + STARTING_ACCEL_RATE * DT_CTRL + 1e-9, \
      f'climbed {first - bottom:.3f} m/s^2 in one frame'

  def test_the_climb_is_monotonic_and_arrives(self):
    lc, CS = _stopped_long_control()
    out = _release(lc, CS, 0.5, int(1.0 / DT_CTRL))
    assert all(b >= a - 1e-9 for a, b in zip(out, out[1:])), 'the climb is not monotonic'
    assert out[-1] == 0.5, 'never reached a_target'
    # -2.0 -> +0.5 at STARTING_ACCEL_RATE, and no faster
    assert min(i for i, a in enumerate(out) if a >= 0.5) >= 2.5 / STARTING_ACCEL_RATE / DT_CTRL - 1

  def test_a_deeper_command_is_never_delayed(self):
    # The limiter is one sided. Braking harder than the ramp was must take effect immediately, or
    # this would be a change to how fast the car can brake.
    lc, CS = _stopped_long_control()
    assert _release(lc, CS, -3.0, 1)[0] == -3.0

  def test_the_limit_lets_go_once_the_pid_catches_up(self):
    # It bounds the step out of a stop, not the pid. Once the two meet, the pid is back in charge
    # and an ordinary command change is not slowed down.
    lc, CS = _stopped_long_control()
    _release(lc, CS, 0.1, int(1.0 / DT_CTRL))
    assert not lc.starting
    assert _release(lc, CS, 1.5, 1)[0] == 1.5

  def test_an_ordinary_stop_and_go_still_moves_off(self):
    # The whole climb has to fit well inside the half second a maneuver gives the planner to get
    # the car going, or this would read as a failure to start.
    lc, CS = _stopped_long_control()
    out = _release(lc, CS, 0.5, int(0.5 / DT_CTRL))
    assert out[-1] == 0.5

  def test_no_limit_without_a_stop_to_climb_out_of(self):
    # Engaging straight into pid must not inherit a limiter: there was no ramp and nothing to undo.
    CP = CarParams()
    CP.longitudinalTuning.kiBP = [0.]
    CP.longitudinalTuning.kiV = [0.]
    lc = LongControl(CP, custom.CarParamsSP.new_message())
    assert lc.update(True, CarState(), 1.5, False, ACCEL_LIMITS) == 1.5
