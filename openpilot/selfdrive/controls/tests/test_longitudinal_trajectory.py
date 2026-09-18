from dataclasses import FrozenInstanceError, replace
from math import inf, nan, nextafter
from random import Random

import pytest

from openpilot.selfdrive.controls.lib.drive_helpers import MIN_STABLE_DELAY, get_accel_from_plan
from openpilot.selfdrive.controls.lib.longitudinal_trajectory import LEGACY_MIN_STABLE_DELAY_S, PiecewiseJerkTrajectory


def make_trajectory(times=(0.0, 1.0), jerks=(0.0,), initial=(0.0, 2.0, 0.0), epoch_ns=17_000_000_000):
  positions, speeds, accelerations = [[value] for value in initial]
  for start, end, jerk in zip(times[:-1], times[1:], jerks, strict=True):
    elapsed = end - start
    positions.append(positions[-1] + elapsed * (speeds[-1] + elapsed * (accelerations[-1] / 2.0 + elapsed * jerk / 6.0)))
    speeds.append(speeds[-1] + elapsed * (accelerations[-1] + elapsed * jerk / 2.0))
    accelerations.append(accelerations[-1] + elapsed * jerk)
  return PiecewiseJerkTrajectory(epoch_ns, times, positions, speeds, accelerations, jerks, solver_status=0)


def assert_zero_residuals(trajectory):
  for residual in trajectory.consistency_residuals():
    assert (residual.position, residual.speed, residual.acceleration) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)


@pytest.mark.parametrize('jerk', [-3.0, -0.1, 0.0, 0.2, 4.0])
@pytest.mark.parametrize('horizon', [0.0, 0.01, 0.15, 0.3, 0.5, 0.9, 1.0])
def test_constant_jerk_has_cubic_position_quadratic_speed_and_linear_acceleration(jerk, horizon):
  trajectory = make_trajectory(jerks=(jerk,), initial=(5.0, 8.0, -0.6))
  sample = trajectory.sample(horizon)
  assert sample.position == pytest.approx(5.0 + 8.0 * horizon - 0.3 * horizon ** 2 + jerk * horizon ** 3 / 6.0)
  assert sample.speed == pytest.approx(8.0 - 0.6 * horizon + jerk * horizon ** 2 / 2.0)
  assert sample.acceleration == pytest.approx(-0.6 + jerk * horizon)
  assert sample.epoch_ns == trajectory.epoch_ns
  assert sample.horizon_s == horizon
  assert_zero_residuals(trajectory)
  if horizon:
    assert trajectory.audit_request(horizon).exact_ramp_request == pytest.approx(sample.acceleration)


def test_piecewise_jerk_is_right_continuous_and_terminal_has_no_outgoing_control():
  trajectory = make_trajectory((0.0, 0.2, 0.7, 1.0), (2.0, -1.0, 4.0), initial=(1.0, 3.0, -0.5))
  for node_index, horizon in enumerate(trajectory.node_times_s):
    sample = trajectory.sample(horizon)
    assert (sample.position, sample.speed, sample.acceleration) == pytest.approx(
      (trajectory.positions[node_index], trajectory.speeds[node_index], trajectory.accelerations[node_index]))
    if node_index < len(trajectory.jerks):
      assert sample.jerk == trajectory.jerks[node_index]
      assert sample.interval_index == node_index
      assert sample.side == 'right'
    else:
      assert sample.jerk is None
      assert sample.interval_index is None
      assert sample.side == 'terminal'
  for node_index, horizon in enumerate(trajectory.node_times_s[1:-1], 1):
    assert trajectory.sample(nextafter(horizon, 0.0)).jerk == trajectory.jerks[node_index - 1]
    assert trajectory.sample(nextafter(horizon, inf)).jerk == trajectory.jerks[node_index]
  assert_zero_residuals(trajectory)


def test_piecewise_samples_agree_with_independent_sequential_integration():
  random = Random(20260916)
  for _ in range(25):
    times = [0.0]
    for _ in range(5):
      times.append(times[-1] + random.uniform(0.05, 0.5))
    jerks = tuple(random.uniform(-3.0, 3.0) for _ in range(5))
    initial = (random.uniform(-1.0, 1.0), random.uniform(0.0, 8.0), random.uniform(-2.0, 2.0))
    trajectory = make_trajectory(times, jerks, initial)
    for _ in range(10):
      horizon = random.uniform(0.0, times[-1])
      position, speed, acceleration = initial
      for start, end, jerk in zip(times[:-1], times[1:], jerks, strict=True):
        if horizon <= start:
          break
        elapsed = min(horizon, end) - start
        position += elapsed * (speed + elapsed * (acceleration / 2.0 + elapsed * jerk / 6.0))
        speed += elapsed * (acceleration + elapsed * jerk / 2.0)
        acceleration += elapsed * jerk
      sample = trajectory.sample(horizon)
      assert (sample.position, sample.speed, sample.acceleration) == pytest.approx((position, speed, acceleration), abs=1e-12)
    assert_zero_residuals(trajectory)


def test_epoch_shift_changes_only_explicit_epoch_metadata():
  trajectory = make_trajectory(jerks=(1.0,))
  shifted = replace(trajectory, epoch_ns=trajectory.epoch_ns + 10**18)
  assert replace(shifted.sample(0.5), epoch_ns=trajectory.epoch_ns) == trajectory.sample(0.5)


def test_arrays_are_copied_and_the_contract_is_immutable():
  speeds = [2.0, 2.0]
  trajectory = replace(make_trajectory(), speeds=speeds)
  speeds[0] = -7.0
  assert trajectory.speeds == (2.0, 2.0)
  with pytest.raises(FrozenInstanceError):
    trajectory.epoch_ns = 0
  with pytest.raises(FrozenInstanceError):
    trajectory.sample(0.5).speed = 99.0


def test_negative_interior_speed_is_visible_even_when_every_node_is_positive():
  trajectory = make_trajectory(jerks=(2.0,), initial=(0.0, 0.1, -1.0))
  assert min(trajectory.speeds) > 0.0
  extrema = trajectory.speed_extrema()
  assert extrema.minimum.horizon_s == pytest.approx(0.5)
  assert extrema.minimum.speed == pytest.approx(-0.15)
  assert extrema.minimum.acceleration == pytest.approx(0.0)
  assert trajectory.sample(0.5).speed < 0.0
  assert trajectory.speed_extrema(0.0, 0.1).minimum.speed == pytest.approx(0.01)


def test_interior_maximum_and_clipped_window_endpoints_are_inspected():
  trajectory = make_trajectory(jerks=(-2.0,), initial=(0.0, 1.0, 1.0))
  extrema = trajectory.speed_extrema(0.2, 0.8)
  assert extrema.maximum.horizon_s == pytest.approx(0.5)
  assert extrema.maximum.speed == pytest.approx(1.25)
  assert extrema.minimum.speed == pytest.approx(1.16)
  point = trajectory.speed_extrema(0.2, 0.2)
  assert point.minimum == point.maximum == trajectory.sample(0.2)
  assert len(point.candidates) == 1


def test_zero_jerk_and_stationary_node_do_not_invent_an_interior_extremum():
  trajectory = make_trajectory((0.0, 0.5, 1.0), (-2.0, 0.0), initial=(0.0, 2.0, 1.0))
  extrema = trajectory.speed_extrema()
  assert extrema.minimum.speed == 2.0
  assert extrema.maximum.speed == 2.25
  assert all(sample.horizon_s in trajectory.node_times_s for sample in extrema.candidates)


def test_node_inconsistency_is_reported_without_overwriting_either_side():
  consistent = make_trajectory((0.0, 0.5, 1.0), (0.0, 0.0), initial=(0.0, 1.0, -1.0))
  trajectory = replace(consistent, positions=(0.0, 2.0, consistent.positions[-1]), speeds=(1.0, -1.0, consistent.speeds[-1]),
                       accelerations=(-1.0, 0.5, consistent.accelerations[-1]))
  residual = trajectory.consistency_residuals()[0]
  assert (residual.position, residual.speed, residual.acceleration) == pytest.approx((1.625, -1.5, 1.5))
  assert trajectory.sample(0.5).speed == -1.0
  left_limits = [sample for sample in trajectory.speed_extrema().candidates if sample.horizon_s == 0.5 and sample.side == 'left_limit']
  assert len(left_limits) == 1
  assert left_limits[0].speed == 0.5
  assert trajectory.speeds == (1.0, -1.0, 0.0)


@pytest.mark.parametrize('changes', [
  {'epoch_ns': -1}, {'epoch_ns': True}, {'epoch_ns': 1.0}, {'epoch_ns': None},
  {'solver_status': 1}, {'solver_status': -1}, {'solver_status': False}, {'solver_status': 0.0}, {'solver_status': '0'},
  {'node_times_s': ()}, {'node_times_s': (0.0,)}, {'node_times_s': (0.1, 1.0)},
  {'node_times_s': (0.0, 0.0)}, {'node_times_s': (0.0, -1.0)}, {'node_times_s': (0.0, 0.8, 0.5)},
  {'node_times_s': (0.0, nan)}, {'node_times_s': (0.0, inf)}, {'node_times_s': (0.0, '1.0')},
  {'positions': (0.0,)}, {'speeds': (0.0, 1.0, 2.0)}, {'accelerations': ()}, {'jerks': (0.0, 0.0)},
  {'positions': (0.0, nan)}, {'speeds': (0.0, inf)}, {'accelerations': (0.0, -inf)}, {'jerks': (nan,)},
  {'positions': ((0.0,), (0.0,))}, {'speeds': (False, 0.0)}, {'accelerations': None}, {'jerks': 0.0},
])
def test_invalid_health_shape_finiteness_and_epoch_are_rejected(changes):
  with pytest.raises(ValueError):
    replace(make_trajectory(), **changes)


@pytest.mark.parametrize('horizon', [-0.01, 1.0001, nan, inf, -inf, True, None, '0.5'])
def test_sampling_and_request_domains_reject_bad_horizons(horizon):
  trajectory = make_trajectory()
  with pytest.raises(ValueError):
    trajectory.sample(horizon)
  with pytest.raises(ValueError):
    trajectory.audit_request(horizon)
  with pytest.raises(ValueError):
    trajectory.speed_extrema(end_horizon_s=horizon if horizon is not None else nan)


def test_zero_request_and_effective_horizon_outside_trajectory_are_rejected():
  trajectory = make_trajectory((0.0, 0.2), (0.0,))
  assert trajectory.sample(0.0).horizon_s == 0.0
  assert trajectory.sample(0.2).side == 'terminal'
  for horizon in (0.0, 0.1, 0.2):
    with pytest.raises(ValueError):
      trajectory.audit_request(horizon)
  with pytest.raises(ValueError):
    trajectory.speed_extrema(0.2, 0.1)


def test_finite_inputs_cannot_produce_silent_nonfinite_samples_or_residuals():
  trajectory = make_trajectory()
  explosive = replace(trajectory, positions=(1e308, 0.0), speeds=(1e308, 0.0), accelerations=(1e308, 0.0), jerks=(1e308,))
  for operation in (lambda: explosive.sample(0.9), explosive.consistency_residuals, explosive.speed_extrema,
                    lambda: explosive.audit_request(0.9)):
    with pytest.raises(ValueError):
      operation()


@pytest.mark.parametrize('horizon', [0.01, 0.05, 0.15, 0.299, 0.3, 0.55, 1.0])
def test_legacy_audit_matches_the_unchanged_helper_and_retains_effective_horizon(horizon):
  trajectory = make_trajectory((0.0, 0.3, 0.6, 1.0), (1.0, -2.0, 0.5), initial=(0.0, 2.0, -0.7))
  audit = trajectory.audit_request(horizon)
  assert LEGACY_MIN_STABLE_DELAY_S == MIN_STABLE_DELAY
  assert audit.legacy_ramp_request == pytest.approx(get_accel_from_plan(
    trajectory.speeds, trajectory.accelerations, trajectory.node_times_s, horizon), abs=1e-12)
  assert audit.requested_sample.horizon_s == horizon
  assert audit.effective_legacy_sample.horizon_s == max(horizon, MIN_STABLE_DELAY)
  if horizon < MIN_STABLE_DELAY:
    assert audit.legacy_ramp_request == pytest.approx(trajectory.audit_request(MIN_STABLE_DELAY).legacy_ramp_request)
    assert audit.exact_ramp_request != pytest.approx(audit.effective_exact_ramp_request)


def test_resampled_legacy_arrays_are_audited_not_replaced_by_native_samples():
  trajectory = make_trajectory(jerks=(2.0,), initial=(0.0, 2.0, -1.0))
  sampled_times = (0.0, 0.25, 0.6, 1.0)
  sampled_speeds = (2.2, 2.1, 1.9, 1.7)
  sampled_accelerations = (-0.7, -0.2, 0.3, 1.0)
  audit = trajectory.audit_request(0.55, sampled_times_s=sampled_times, sampled_speeds=sampled_speeds,
                                   sampled_accelerations=sampled_accelerations)
  assert audit.legacy_ramp_request == pytest.approx(get_accel_from_plan(sampled_speeds, sampled_accelerations, sampled_times, 0.55))
  assert audit.legacy_initial_speed_residual == pytest.approx(0.2)
  assert audit.legacy_initial_acceleration_residual == pytest.approx(0.3)
  assert audit.legacy_speed_residual == pytest.approx(audit.legacy_query_speed - trajectory.sample(0.55).speed)
  assert audit.requested_sample == trajectory.sample(0.55)
  assert sampled_speeds == (2.2, 2.1, 1.9, 1.7)


@pytest.mark.parametrize('changes', [
  {'sampled_times_s': None}, {'sampled_speeds': None}, {'sampled_accelerations': None},
  {'sampled_times_s': (0.1, 1.0)}, {'sampled_times_s': (0.0, 0.0)},
  {'sampled_times_s': (0.0, 0.2)}, {'sampled_times_s': (0.0, 1.1)}, {'sampled_times_s': (0.0, nan)},
  {'sampled_speeds': (2.0,)}, {'sampled_speeds': (2.0, nan)}, {'sampled_accelerations': ()},
  {'sampled_accelerations': (nan, 0.0)},
])
def test_legacy_audit_rejects_partial_mismatched_or_out_of_domain_arrays(changes):
  arguments = {'sampled_times_s': (0.0, 1.0), 'sampled_speeds': (2.0, 2.0), 'sampled_accelerations': (0.0, 0.0)}
  arguments.update(changes)
  with pytest.raises(ValueError):
    make_trajectory().audit_request(0.1, **arguments)


def test_positive_equivalent_ramp_request_is_not_forward_acceleration_or_movement_intent():
  trajectory = make_trajectory((0.0, 0.1, 0.3, 1.0), (10.0, 0.0, 0.0), initial=(0.0, 2.0, -1.0))
  audit = trajectory.audit_request(1.0)
  assert min(trajectory.speeds) > 0.0
  assert all(sample.acceleration <= 0.0 for sample in trajectory.speed_extrema().candidates)
  assert audit.requested_sample.acceleration == 0.0
  assert audit.legacy_ramp_request == pytest.approx(0.9)
  assert audit.exact_ramp_request == pytest.approx(0.9)
  assert audit.requested_sample.speed < trajectory.speeds[0]
  assert not hasattr(audit, 'release_allowed')
  assert not hasattr(audit, 'should_stop')


@pytest.mark.parametrize('jerks', [(1.0, -3.0), (-1.0, 3.0)])
def test_point_and_ramp_requests_have_no_universal_ordering(jerks):
  trajectory = make_trajectory((0.0, 0.5, 1.0), jerks, initial=(0.0, 5.0, -1.0))
  audit = trajectory.audit_request(1.0)
  difference = audit.exact_ramp_request - audit.requested_sample.acceleration
  assert difference * jerks[0] > 0.0
  assert abs(difference) == pytest.approx(1.0)
