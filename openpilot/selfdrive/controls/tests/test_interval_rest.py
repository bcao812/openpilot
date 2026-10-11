"""The in-interval stop is an actual braking/hold curve, not zeroed samples."""

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.interval_rest import IntervalRest


CAPTURED_STATES = (
  (0.008282944560050964, -0.9812936782836914),
  (0.007247960660606623, -1.4663481712341309),
)
FIRST_INTERVAL = 10.0 / 144.0


@pytest.mark.parametrize('speed,acceleration', CAPTURED_STATES)
def test_captured_state_stops_before_first_node_without_acceleration_or_reverse(speed, acceleration):
  event = IntervalRest.from_state((7.0, speed, acceleration), FIRST_INTERVAL, 10.0 / 36.0)
  assert event is not None and 0.0 < event.stop_time < FIRST_INTERVAL
  times = np.array([0.0, event.stop_time / 4.0, event.stop_time / 2.0, event.stop_time,
                    (event.stop_time + FIRST_INTERVAL) / 2.0, FIRST_INTERVAL, 10.0])
  states, jerk = event.sample(times)
  np.testing.assert_array_equal(states[0], [7.0, speed, acceleration])
  assert np.all(states[:, 1] >= 0.0) and np.all(states[:, 2] <= 0.0)
  assert np.all(states[:, 2] >= acceleration) and np.all(np.diff(states[:, 0]) >= 0.0)
  np.testing.assert_array_equal(states[3:, 1:], 0.0)
  np.testing.assert_array_equal(jerk[3:], 0.0)
  # Independently integrate each exposed constant-jerk subinterval. This also
  # checks that the zero tail did not discard positive stopping distance.
  for left, right, initial, final, control in zip(times[:-1], times[1:], states[:-1], states[1:], jerk[:-1], strict=True):
    dt = right - left
    position, velocity, accel = initial
    expected = [position + velocity * dt + accel * dt**2 / 2.0 + control * dt**3 / 6.0,
                velocity + accel * dt + control * dt**2 / 2.0, accel + control * dt]
    np.testing.assert_allclose(final, expected, atol=2e-15, rtol=0.0)
  assert states[-1, 0] > states[0, 0]


@pytest.mark.parametrize('state,deadline', [
  ((0.0, 0.0, -1.0), 0.3), ((0.0, -0.01, -1.0), 0.3),
  ((0.0, 0.01, 0.0), 0.3), ((0.0, 0.01, 1.0), 0.3),
  ((0.0, 1.0, -1.0), 0.3), ((0.0, 0.01, -1.0), 0.01),
  ((0.0, float('nan'), -1.0), 0.3), ((0.0, 0.01, float('-inf')), 0.3),
])
def test_interval_rest_does_not_reinterpret_other_initial_states_or_deadlines(state, deadline):
  assert IntervalRest.from_state(state, FIRST_INTERVAL, deadline) is None


def test_event_has_its_own_knot_without_moving_public_observation_times():
  event = IntervalRest.from_state((0.0, 0.008, -1.0), FIRST_INTERVAL, 0.3)
  public = np.array([0.0, FIRST_INTERVAL, 10.0 / 36.0, 10.0])
  original = public.copy()
  times, states, jerks = event.expanded_trajectory(public)
  np.testing.assert_array_equal(np.delete(times, 1), original)
  np.testing.assert_array_equal(public, original)
  assert times[1] == event.stop_time
  np.testing.assert_array_equal(states[1:, 1:], 0.0)
  np.testing.assert_array_equal(jerks[1:], 0.0)
