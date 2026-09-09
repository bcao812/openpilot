import pytest

from openpilot.selfdrive.controls.tests.test_longitudinal_cruise_scenarios import simulate_lead_cruise


@pytest.mark.parametrize('mode', ['brake_dwell', 'median'])
@pytest.mark.parametrize('offset,delay', [(0.25, 0.5), (0.254545, 0.5), (0.254545, 0.8), (0.25, 0.8)])
def test_temporal_strategies_do_not_reintroduce_static_shaping_limit_cycles(mode, offset, delay):
  result = simulate_lead_cruise(False, smoothing=mode, offset=offset, noise=0.02, delay=delay)
  assert result['switches']['raw'] == 0
  assert result['switches']['debounced'] == 0
  assert result['speed_range_kph'] < 0.02
  assert result['conflicts'] == 0


@pytest.mark.parametrize('mode', ['brake_dwell', 'median'])
def test_mode_flicker_does_not_enable_temporal_memory(mode):
  before = simulate_lead_cruise(False, scenario='flicker', noise=0.02)
  after = simulate_lead_cruise(False, smoothing=mode, scenario='flicker', noise=0.02)
  assert before == after


@pytest.mark.parametrize('mode', ['brake_dwell', 'median'])
@pytest.mark.parametrize('scenario', ['cut_in', 'lead_brake'])
def test_temporal_strategies_do_not_delay_risk_braking(mode, scenario):
  arguments = {'scenario': scenario, 'noise': 0.02, 'duration': 55.0, 'delay': 0.3, 'response_time': 0.2}
  before = simulate_lead_cruise(False, **arguments)
  after = simulate_lead_cruise(False, smoothing=mode, **arguments)
  assert after['brake_response'] <= before['brake_response'] + 1e-9
  assert after['minimum_gap'] > 2.0
  assert after['minimum_gap'] >= before['minimum_gap'] - 0.05
  assert after['conflicts'] == 0


@pytest.mark.xfail(strict=True, reason='Median filtering adds brake/gas transitions at the exact crossover with actuator delay')
def test_median_crossover_release_gate():
  before = simulate_lead_cruise(False, offset=0.2, delay=0.8)
  after = simulate_lead_cruise(False, smoothing='median', offset=0.2, delay=0.8)
  assert after['switches']['debounced'] <= before['switches']['debounced']


@pytest.mark.xfail(strict=True, reason='Median filtering adds transitions after override reentry at long actuator delay')
def test_median_reentry_release_gate():
  before = simulate_lead_cruise(False, scenario='override', noise=0.02, delay=0.8)
  after = simulate_lead_cruise(False, smoothing='median', scenario='override', noise=0.02, delay=0.8)
  assert after['switches']['debounced'] <= before['switches']['debounced']
