import math
from unittest.mock import patch

import numpy as np
import pytest

from opendbc.car.honda.values import CarControllerParams
from openpilot.selfdrive.controls.lib.cruise_smoothing import CruiseSmoother, CruiseSmoothingMode
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.tests.test_longitudinal_cruise_coast import cruise_planner, update_cruise
from openpilot.selfdrive.controls.tests.test_longitudinal_cruise_leads import distant_lead_state
from openpilot.selfdrive.controls.tests.replay_pedal_switches import replay_window, switch_metrics


def smoothing_planner(mode):
  planner = cruise_planner()
  planner.cruise_smoother = CruiseSmoother(mode, planner.dt)
  return planner


def confirmed_smoother(mode):
  smoother = CruiseSmoother(mode)
  for _ in range(14):
    smoother.update(-0.21, True)
  return smoother


@pytest.mark.parametrize('mode', ['brake_dwell', 'median'])
@pytest.mark.parametrize('accel', [-0.28, -0.254545, -0.25, -0.23, -0.201, -0.2, -0.19, -0.12])
def test_constant_requests_retain_their_equilibrium(mode, accel):
  smoother = CruiseSmoother(mode)
  assert all(smoother.update(accel, True) == accel for _ in range(30))


def test_brake_minimum_dwell_does_not_prevent_stronger_braking():
  smoother = confirmed_smoother('brake_dwell')
  assert smoother.update(-0.21, True) == -0.21
  assert smoother.update(-0.19, True) == -0.21
  assert smoother.update(-0.28, True) == -0.28
  assert smoother.update(-2.0, False) == -2.0
  assert not smoother.braking


def test_brake_release_requires_time_without_holding_an_accel_mode():
  smoother = confirmed_smoother('brake_dwell')
  held = [smoother.update(-0.19, True) for _ in range(3)]
  assert held == [-0.21] * 3
  assert smoother.update(-0.19, True) == -0.19
  assert smoother.update(-0.21, True) == -0.21


@pytest.mark.parametrize('pulse_samples', [1, 2, 3])
def test_minimum_dwell_does_not_extend_unconfirmed_brake_noise(pulse_samples):
  smoother = CruiseSmoother('brake_dwell')
  for _ in range(20):
    smoother.update(-0.19, True)
  requests = ([-0.21] * pulse_samples + [-0.19] * 5) * 10
  assert [smoother.update(accel, True) for accel in requests] == requests


def test_large_release_uses_original_request_not_minimum_dwell():
  smoother = confirmed_smoother('brake_dwell')
  smoother.update(-0.23, True)
  assert smoother.update(-0.15, True) == -0.15
  assert not smoother.braking


def test_median_rejects_subwindow_pulses_without_a_static_plateau():
  smoother = confirmed_smoother('median')
  for _ in range(5):
    smoother.update(-0.23, True)
  assert smoother.update(-0.19, True) == -0.23
  assert smoother.update(-0.19, True) == -0.23
  assert smoother.update(-0.19, True) == -0.19
  assert smoother.update(-0.28, False) == -0.28
  assert not smoother.samples


@pytest.mark.parametrize('mode', ['brake_dwell', 'median'])
def test_ineligible_or_nonfinite_input_clears_temporal_memory(mode):
  smoother = CruiseSmoother(mode)
  for _ in range(20):
    smoother.update(-0.23, True)
  assert math.isnan(smoother.update(float('nan'), True))
  assert not smoother.samples and not smoother.braking
  assert smoother.update(-0.19, True) == -0.19
  assert smoother.update(-0.23, False) == -0.23
  assert not smoother.samples and not smoother.braking


def test_threshold_matches_accord_can_brake_boundary():
  assert CruiseSmoother.BRAKE_THRESHOLD == CarControllerParams.BOSCH_GAS_LOOKUP_BP[0]


@pytest.mark.parametrize('mode', ['brake_dwell', 'median'])
@pytest.mark.parametrize('condition', ['fcw', 'model_nan', 'model_inf', 'mpc_inf', 'lead_brake', 'second_lead'])
def test_risk_veto_resets_preexisting_temporal_memory(mode, condition):
  planner, state = smoothing_planner(mode), distant_lead_state()
  for _ in range(20):
    update_cruise(planner, state, a_cruise=-0.21)
  if condition in ('model_nan', 'model_inf'):
    state['modelV2'] = state['modelV2'].as_builder()
    state['modelV2'].action.desiredAcceleration = float('nan' if condition == 'model_nan' else 'inf')
  elif condition == 'lead_brake':
    state['radarState'].leadOne.aLeadK = -1.0
  elif condition == 'second_lead':
    state['radarState'].leadTwo.present = True
    state['radarState'].leadTwo.dRel = 3.0
  if condition == 'fcw':
    planner.mpc.crash_cnt = 5
    with patch.object(planner.mpc, 'update', return_value=None):
      assert update_cruise(planner, state, a_cruise=-0.19) == -0.19
  else:
    assert update_cruise(planner, state, a_cruise=-0.19, a_mpc=float('inf') if condition == 'mpc_inf' else 1.0) == -0.19
  assert not planner.cruise_smoother.braking
  assert not planner.cruise_smoother.brake_confirmed
  assert not planner.cruise_smoother.samples
  assert planner.cruise_smoother.eligible_time == 0.0


@pytest.mark.parametrize('mode', ['brake_dwell', 'median'])
def test_smoothing_delta_is_bounded(mode):
  smoother = CruiseSmoother(mode)
  requests = np.random.default_rng(7).uniform(-0.3, -0.1, 1000)
  outputs = np.array([smoother.update(float(accel), True) for accel in requests])
  assert np.max(np.abs(outputs - requests)) <= (0.04 if mode == 'brake_dwell' else 0.06) + 1e-9
  if mode == 'brake_dwell':
    assert np.all(outputs <= requests)


@pytest.mark.parametrize('mode', ['brake_dwell', 'median'])
@pytest.mark.parametrize(
  'condition',
  [
    'gas',
    'brake',
    'override',
    'inactive',
    'force_decel',
    'model_stop',
    'model_brake',
    'e2e',
    'close_lead',
    'closing_lead',
    'no_lead',
    'low_speed',
    'far_speed',
    'unhealthy',
  ],
)
def test_new_temporal_memory_never_locks_risk_or_mode_transitions(mode, condition):
  planner = smoothing_planner(mode)
  state = distant_lead_state()
  for _ in range(20):
    update_cruise(planner, state, a_cruise=-0.21)
  if condition in ('gas', 'brake'):
    setattr(state['carState'], condition + 'Pressed', True)
  elif condition == 'override':
    state['carControl'].cruiseControl.override = True
  elif condition == 'inactive':
    state['carControl'].longActive = False
  elif condition == 'force_decel':
    state['controlsState'].forceDecel = True
  elif condition in ('model_stop', 'model_brake'):
    state['modelV2'] = state['modelV2'].as_builder()
    state['modelV2'].action.shouldStop = condition == 'model_stop'
    state['modelV2'].action.desiredAcceleration = -1.0 if condition == 'model_brake' else 0.1
  elif condition == 'close_lead':
    state['radarState'].leadOne.dRel = 5.0
  elif condition == 'closing_lead':
    state['radarState'].leadOne.vRel = -20.0
  elif condition == 'no_lead':
    state['radarState'].leadOne.present = False
  elif condition == 'low_speed':
    state['carState'].vEgo = 4.0
  elif condition == 'far_speed':
    state['carState'].vEgo = 21.0
  elif condition == 'unhealthy':
    state.all_checks = lambda service_list=None: False
  reference = cruise_planner()
  output = update_cruise(planner, state, a_cruise=-0.19, e2e=condition == 'e2e')
  assert output == update_cruise(reference, state, a_cruise=-0.19, e2e=condition == 'e2e')
  assert not planner.cruise_smoother.samples and not planner.cruise_smoother.braking


@pytest.mark.parametrize('mode', ['brake_dwell', 'median'])
def test_mpc_cut_in_can_brake_on_first_tick(mode):
  planner, state = smoothing_planner(mode), distant_lead_state()
  for _ in range(20):
    update_cruise(planner, state, a_cruise=-0.19)
  assert update_cruise(planner, state, a_cruise=-0.19, a_mpc=-2.5) == -2.5
  assert not planner.cruise_smoother.samples and not planner.cruise_smoother.braking


def test_new_strategies_are_default_off_and_not_stackable():
  planner = cruise_planner()
  assert planner.cruise_smoother.mode == CruiseSmoothingMode.OFF
  assert update_cruise(planner, distant_lead_state()) == -0.23
  with pytest.raises(ValueError, match='separately'):
    LongitudinalPlanner(planner.CP, None, experimental_coast_with_lead=True, experimental_cruise_smoothing='median')


@pytest.mark.parametrize('mode', ['brake_dwell', 'median'])
def test_short_noise_bursts_reduced_at_real_honda_crossover(mode):
  requests = [-0.21] * 10 + [-0.19] * 2 + [-0.21] * 3 + [-0.19] * 2 + [-0.21] * 10
  smoother = confirmed_smoother(mode)
  outputs = [smoother.update(accel, True) for accel in requests]

  def metric(values):
    rows = [{'t': index * 0.05, 'automatic': True, 'can_brake': accel < -0.2, 'can_gas': 7 if accel > -0.2 else -30000} for index, accel in enumerate(values)]
    return switch_metrics(rows)['raw']

  assert metric(requests) == 4
  assert metric(outputs) == 0


@pytest.mark.parametrize('mode', ['brake_dwell', 'median'])
def test_brief_mode_chatter_does_not_prime_smoothing(mode):
  smoother = CruiseSmoother(mode)
  for _ in range(5):
    for index in range(10):
      accel = -0.21 if index % 2 else -0.19
      assert smoother.update(accel, True) == accel
    assert smoother.update(-0.19, False) == -0.19


@pytest.mark.parametrize('timestep', [0.0, -0.05, float('nan'), float('inf')])
def test_invalid_timestep_is_rejected(timestep):
  with pytest.raises(ValueError, match='timestep'):
    CruiseSmoother('median', timestep)


def test_replay_rejects_mixed_experiments_before_reading_logs():
  with pytest.raises(ValueError, match='separately'):
    replay_window(None, {}, True, experimental_cruise_smoothing='brake_dwell')
