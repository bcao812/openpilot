from unittest.mock import patch

import numpy as np
import pytest

from opendbc.car.honda.carstate import CarState as HondaCarState
from opendbc.car.interfaces import ACCEL_MAX, ACCEL_MIN
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.tests.test_longitudinal_following_scenarios import simulate_following
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_planner import advance, make_planner, make_state
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP


@pytest.fixture(autouse=True)
def isolated_params():
  with OpenpilotPrefix():
    yield


def expected_motion(state):
  speed = max(0.0, float(state['carState'].vEgo))
  acceleration = float(np.clip(state['carState'].aEgo, ACCEL_MIN, ACCEL_MAX))
  return speed, max(0.0, acceleration) if speed == 0.0 else acceleration


def assert_measured_initial_state(planner, state):
  assert planner.mpc.solution_status == 0
  assert planner.motion_intent.valid
  np.testing.assert_array_equal(planner.mpc.x0[1:], expected_motion(state))


@pytest.mark.parametrize('motion', [
  [(0, 20.0, -2.0), (100_000_000, 19.8, -2.0), (300_000_000, 19.4, -2.0)],
  [(0, 10.0, 1.0), (50_000_000, 10.05, 1.0), (150_000_000, 10.15, 1.0)],
  [(0, 20.0, -1.0), (50_000_000, 19.925, -2.0), (100_000_000, 19.8, -3.0)],
])
def test_successive_motion_estimates_use_speed_and_acceleration_from_the_same_frame(motion):
  planner, _ = make_planner(speed=motion[0][1])
  state = make_state(speed=motion[0][1], target=25.0)
  for elapsed_ns, speed, acceleration in motion:
    state['carState'].vEgo = speed
    state['carState'].vEgoRaw = speed
    state['carState'].aEgo = acceleration
    now_ns = 1_000_000_000 + elapsed_ns
    for service in state.logMonoTime:
      state.logMonoTime[service] = now_ns
    original_measurement = (state['carState'].vEgo, state['carState'].aEgo)
    planner.update(state, now_ns=now_ns)
    assert_measured_initial_state(planner, state)
    assert (state['carState'].vEgo, state['carState'].aEgo) == original_measurement


@pytest.mark.parametrize('warm', [False, True])
@pytest.mark.parametrize('old_filter_speed', [0.0, 35.0, float('nan')])
@pytest.mark.parametrize('speed,acceleration,holding', [(0.0, -0.5, True), (0.2, -0.5, True), (19.6, -2.0, False)])
def test_stale_planner_filter_and_hold_request_cannot_replace_a_current_motion_estimate(warm, old_filter_speed,
                                                                                      speed, acceleration, holding):
  planner, _ = make_planner(speed=0.0)
  if warm:
    initial_state = make_state(speed=20.0, target=25.0)
    planner.update(initial_state, now_ns=advance(initial_state, 0))
    assert planner.motion_intent.valid
  planner.v_desired_filter.x = old_filter_speed
  planner.output_a_target = -2.0
  state = make_state(speed=speed, target=20.0)
  state['carState'].aEgo = acceleration
  # A zero Kalman estimate and a standstill flag are independent evidence.
  state['carState'].standstill = False
  state['controlsState'].longControlState = LongCtrlState.stopping if holding else LongCtrlState.pid
  state['carControl'].actuators.accel = -3.0
  planner.update(state, now_ns=advance(state, 1))
  assert_measured_initial_state(planner, state)
  assert state['carControl'].actuators.accel == -3.0


@pytest.mark.parametrize('speed,acceleration,standstill,expected_accel', [
  (20.0, -9.0, False, ACCEL_MIN), (20.0, 9.0, False, ACCEL_MAX),
  (0.0, -1.5, False, 0.0), (0.0, -1.5, True, 0.0),
  (0.2, -0.5, False, -0.5), (0.2, -0.5, True, -0.5),
  (-0.02, -0.5, False, 0.0), (-0.02, 0.2, False, 0.2),
])
def test_measurement_projection_keeps_forward_motion_and_global_acceleration_bounds(speed, acceleration, standstill, expected_accel):
  planner, _ = make_planner(speed=0.0)
  state = make_state(speed=max(0.0, speed), target=20.0)
  state['carState'].vEgo = speed
  state['carState'].aEgo = acceleration
  state['carState'].standstill = standstill
  planner.update(state, now_ns=advance(state, 0))
  assert_measured_initial_state(planner, state)
  assert planner.mpc.x0[2] == pytest.approx(expected_accel)


@pytest.mark.parametrize('interruption', ['invalid_speed', 'invalid_acceleration', 'disabled', 'gas_override'])
def test_invalid_or_interrupted_motion_resumes_with_current_paired_estimates(interruption):
  planner, _ = make_planner(speed=20.0)
  state = make_state(speed=20.0, target=15.0)
  state['carState'].aEgo = -1.0
  planner.update(state, now_ns=advance(state, 0))
  if interruption == 'invalid_speed':
    state['carState'].vEgo = float('nan')
  elif interruption == 'invalid_acceleration':
    state['carState'].aEgo = float('inf')
  elif interruption == 'disabled':
    state['selfdriveState'].enabled = False
    state['carControl'].enabled = False
    state['carControl'].longActive = False
    state['controlsState'].longControlState = LongCtrlState.off
  else:
    state['carState'].gasPressed = True
  planner.update(state, now_ns=advance(state, 1))
  if interruption.startswith('invalid'):
    assert not planner.motion_intent.valid
    assert np.isfinite(planner.output_a_target)
    assert planner.output_a_target <= -0.2
  state = make_state(speed=19.85, target=15.0)
  state['carState'].aEgo = -1.5
  planner.v_desired_filter.x = 35.0
  with patch.object(planner.mpc, 'set_weights', wraps=planner.mpc.set_weights) as weights:
    planner.update(state, now_ns=advance(state, 2))
  assert_measured_initial_state(planner, state)
  assert planner.output_a_target < 0.0
  assert not weights.call_args.args[0]


def test_policy_receives_the_same_measured_pair_as_the_native_initial_state():
  planner, _ = make_planner(speed=20.0)
  state = make_state(speed=20.0, target=25.0)
  planner.update(state, now_ns=advance(state, 0))
  state['carState'].vEgo = 19.9
  state['carState'].aEgo = -1.4
  with patch.object(LongitudinalPlannerSP, 'update_targets', wraps=LongitudinalPlannerSP.update_targets) as policy:
    planner.update(state, now_ns=advance(state, 1))
  assert_measured_initial_state(planner, state)
  np.testing.assert_array_equal(policy.call_args.args[2:4], expected_motion(state))


def fixed_honda_speed_estimates(params, extension, *, profile='constant', noise=0.05, duration=5.0):
  """Fixed raw-speed inputs through the existing Honda speed Kalman filter.

  The synthetic vEgoRaw is the input after Honda's wheel/transmission blending;
  this exercises the actual estimator, not a calibrated tire or vehicle plant.
  A one-second constant-speed prefix initializes its state before the noise.
  """
  initial_speed = 1.1 if profile == 'near_stop' else 20.0
  estimator = HondaCarState(params, extension)
  for _ in range(round(1.0 / DT_CTRL)):
    estimator.update_speed_kf(initial_speed)
  samples = []
  for frame in range(round(duration / DT_CTRL)):
    timestamp = frame * DT_CTRL
    if profile == 'near_stop':
      true_speed = max(0.0, initial_speed - 0.25 * timestamp)
    elif profile == 'decelerating':
      true_speed = initial_speed - 0.8 * timestamp
    else:
      true_speed = initial_speed
    perturbation = noise * (0.7 * np.sin(2.0 * np.pi * 1.1 * timestamp) + 0.3 * np.sin(2.0 * np.pi * 3.7 * timestamp))
    raw_speed = max(0.0, true_speed + perturbation)
    speed, acceleration = estimator.update_speed_kf(raw_speed)
    if frame % round(DT_MDL / DT_CTRL) == 0:
      samples.append((1_000_000_000 + round(timestamp * 1e9), float(raw_speed), float(speed), float(acceleration)))
  return samples


@pytest.mark.parametrize('profile', ['constant', 'decelerating', 'near_stop'])
def test_fixed_noisy_honda_kalman_pairs_are_consumed_without_a_second_speed_filter(profile):
  planner, extension = make_planner(speed=0.0)
  samples = fixed_honda_speed_estimates(planner.CP, extension, profile=profile)
  assert samples == fixed_honda_speed_estimates(planner.CP, extension, profile=profile)
  assert any(abs(sample[3]) > 0.01 for sample in samples)
  for now_ns, raw_speed, speed, acceleration in samples:
    state = make_state(speed=max(0.0, speed), target=22.0)
    state['carState'].vEgoRaw = raw_speed
    state['carState'].vEgo = speed
    state['carState'].aEgo = acceleration
    state['carState'].standstill = raw_speed < 1e-5
    for service in state.logMonoTime:
      state.logMonoTime[service] = now_ns
    planner.update(state, now_ns=now_ns)
    assert_measured_initial_state(planner, state)
    assert np.isfinite(planner.output_a_target)
    assert ACCEL_MIN <= planner.output_a_target <= ACCEL_MAX


@pytest.mark.parametrize('scenario', ['moving', 'lead_stop', 'cut_in', 'model_stop'])
def test_speed_only_noise_stress_preserves_existing_following_and_stopping_gates(scenario):
  # This existing plant perturbs v only; keep that assumption separate from the
  # paired real-KF sample qualification above and from measured road experience.
  result = simulate_following(unified=True, scenario=scenario, noise=0.05)
  assert result['minimum_gap'] > (5.0 if scenario == 'moving' else 2.0)
  assert result['minimum_command'] >= ACCEL_MIN
  assert result['maximum_command'] <= ACCEL_MAX
  assert result['maximum_jerk_up'] <= 8.0 + 1e-5
  assert result['conflicts'] == 0
  if scenario == 'moving':
    assert result['stopping_seconds'] == 0.0
    assert result['minimum_speed'] > 0.3
    assert result['speed_error'] < 0.15
  else:
    assert result['final_speed'] == 0.0
    assert result['stopping_seconds'] > 0.0
