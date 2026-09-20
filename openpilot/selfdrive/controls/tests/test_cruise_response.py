"""P's cruise response history, with fixed goldens and the retained P helper.

These contracts concern the response ceiling; final arbitration and vehicle
effects require planner integration and closed-loop tests separately.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.selfdrive.controls.lib.cruise_response import CruiseResponse


@pytest.mark.parametrize('speed,expected', [
  (0.0, 0.08), (5.0, 0.07), (10.0, 0.06), (17.5, 0.05),
  (25.0, 0.04), (32.5, 0.035), (40.0, 0.03), (50.0, 0.03),
])
def test_first_step_keeps_p_jerk_knots_and_interpolation(speed, expected):
  response = CruiseResponse()
  assert response.update(speed + 10.0, speed, 1.6, measured_accel=0.0) == pytest.approx(expected, rel=0.0, abs=1e-12)


@pytest.mark.parametrize('target,sign', [(12.0, 1.0), (8.0, -1.0)])
def test_positive_and_negative_speed_steps_keep_the_complete_p_ramp(target, sign):
  response = CruiseResponse()
  actual = [response.update(target, 10.0, 1.2, measured_accel=0.0) for _ in range(25)]
  assert actual[:6] == pytest.approx([sign * value for value in (0.06, 0.12, 0.18, 0.24, 0.30, 0.36)], rel=0.0, abs=1e-12)
  assert actual[19:] == pytest.approx([sign * 1.2] * 6, rel=0.0, abs=1e-12)


def test_small_speed_error_settles_at_the_error_without_an_extra_gain():
  response = CruiseResponse()
  actual = [response.update(10.14, 10.0, 1.2, measured_accel=0.0) for _ in range(5)]
  assert actual == pytest.approx([0.06, 0.12, 0.14, 0.14, 0.14], rel=0.0, abs=1e-12)


@pytest.mark.parametrize('measured,target,expected', [
  (-8.0, 12.0, -3.44), (-0.8, 12.0, -0.74), (0.8, 8.0, 0.74), (8.0, 8.0, 1.94),
])
@pytest.mark.parametrize('reset', [False, True])
def test_clear_or_reset_sample_clips_measurement_then_advances(measured, target, expected, reset):
  response = CruiseResponse()
  response.update(12.0, 10.0, 1.2, measured_accel=0.0)
  if not reset:
    response.clear()
  assert response.update(target, 10.0, 1.2, measured_accel=measured, reset=reset) == pytest.approx(expected, rel=0.0, abs=1e-12)


@pytest.mark.parametrize('initial,expected', [(0.0, 0.06), (-0.6, -0.54), (0.7, 0.76)])
def test_constructor_state_matches_p_direct_active_start(initial, expected):
  response = CruiseResponse(init_a=initial)
  assert response.update(12.0, 10.0, 1.2, measured_accel=-2.0) == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_only_explicit_reset_replaces_history_with_a_new_measurement():
  response = CruiseResponse(init_a=0.5)
  actual = [response.update(12.0, 10.0, 1.2, measured_accel=value) for value in (0.5, -2.0, 2.0)]
  assert actual == pytest.approx([0.56, 0.62, 0.68], rel=0.0, abs=1e-12)
  # Off/manual control can reset every model frame, just as in P.
  for _ in range(3):
    assert response.update(12.0, 10.0, 1.2, measured_accel=-0.5, reset=True) == pytest.approx(-0.44, rel=0.0, abs=1e-12)


def test_target_source_changes_and_successive_limits_preserve_history():
  response = CruiseResponse()
  # Cruise -> SCC -> another speed limit -> cruise. Sources select speeds in
  # the policy owner; they neither belong to this helper nor reset its state.
  targets = [12.0] * 3 + [6.0] * 2 + [8.0] * 2 + [12.0] * 3
  actual = [response.update(target, 10.0, 1.2, measured_accel=0.0) for target in targets]
  assert actual == pytest.approx([0.06, 0.12, 0.18, 0.12, 0.06, 0.0, -0.06, 0.0, 0.06, 0.12], rel=0.0, abs=1e-12)


def test_maximum_changes_keep_target_clamp_before_response_slew():
  response = CruiseResponse(init_a=0.5)
  actual = [response.update(12.0, 10.0, maximum, measured_accel=0.5) for maximum in (1.2, 0.2, -0.4, -0.4, 1.2)]
  assert actual == pytest.approx([0.56, 0.50, 0.44, 0.38, 0.44], rel=0.0, abs=1e-12)
  # The separate final maximum must still constrain these first falling frames.
  assert actual[1] > 0.2 and actual[2] > -0.4


def test_cruise_floor_does_not_relax_a_stronger_existing_coast_limit():
  response = CruiseResponse()
  actual = [response.update(12.0, 10.0, -2.0, measured_accel=0.0) for _ in range(40)]
  assert actual[:3] == pytest.approx([-0.06, -0.12, -0.18], rel=0.0, abs=1e-12)
  assert actual[-1] == -2.0


def test_clear_reseeds_from_current_measurement_and_is_idempotent():
  response = CruiseResponse()
  for _ in range(5):
    response.update(12.0, 10.0, 1.2, measured_accel=0.0)
  response.clear()
  response.clear()
  assert response.update(12.0, 10.0, 1.2, measured_accel=-0.6) == pytest.approx(-0.54, rel=0.0, abs=1e-12)
  assert response.update(12.0, 10.0, 1.2, measured_accel=0.0) == pytest.approx(-0.48, rel=0.0, abs=1e-12)


@pytest.mark.parametrize('dt', [0.025, 0.05, 0.1])
@pytest.mark.parametrize('e2e', [False, True])
@pytest.mark.parametrize('initial_accel', [-4.0, -0.6, 0.7, 3.0])
def test_complete_target_history_matches_the_retained_p_helper(dt, e2e, initial_accel):
  # get_cruise_accel's numeric body is unchanged from P=e18b1de, lines 207-224.
  # Import only in this comparison: standalone golden contracts do not need MPC.
  from openpilot.selfdrive.controls.lib.longitudinal_planner import get_cruise_accel, get_max_accel

  response = CruiseResponse(dt=dt)
  params = SimpleNamespace(steerRatio=15.0, wheelbase=2.8)
  previous = float(np.clip(initial_accel, ACCEL_MIN, ACCEL_MAX))
  # Includes acceleration, SCC-like speed reductions, a second limit, recovery,
  # equilibrium and speed-dependent jerk changes. All measurements after the
  # first intentionally differ from the response; only reset may consume them.
  phases = ((10.0, 14.0), (10.0, 8.0), (10.0, 7.0), (10.0, 14.0),
            (10.0, 10.0), (17.5, 19.0), (25.0, 22.0), (40.0, 42.0), (0.0, 2.0))
  for phase, (speed, target) in enumerate(phases):
    maximum = ACCEL_MAX if e2e else get_max_accel(speed)
    for frame in range(30):
      first = phase == 0 and frame == 0
      previous = get_cruise_accel(e2e, target, speed, previous, 0.0, params, dt, ACCEL_MAX, True)
      actual = response.update(target, speed, maximum, measured_accel=initial_accel if first else -0.25, reset=first)
      assert actual == pytest.approx(previous, rel=0.0, abs=1e-12), (phase, frame)


@pytest.mark.parametrize('raw,expected', [
  (-1.2, -1.2), (-0.31, -0.31), (-0.3, -0.3), (-0.275, -0.245),
  (-0.25, -0.19), (-0.23, -0.19), (-0.19, -0.19), (-0.15, -0.19),
  (-0.125, -0.145), (-0.1, -0.1), (-0.09, -0.09), (0.0, 0.0), (0.5, 0.5),
])
def test_cruise_coast_map_keeps_p_knots_interpolation_and_open_boundaries(raw, expected):
  # P=e18b1de longitudinal_planner.py:36-40,69-73. Fixed numbers keep this
  # baseline independent of any shared production constants or current output.
  from openpilot.selfdrive.controls.lib.cruise_response import get_cruise_coast_accel

  assert get_cruise_coast_accel(raw) == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_cruise_coast_map_does_not_replace_the_raw_response_history():
  from openpilot.selfdrive.controls.lib.cruise_response import get_cruise_coast_accel
  from openpilot.selfdrive.controls.lib.longitudinal_planner import get_cruise_accel, get_max_accel

  response = CruiseResponse(init_a=-0.23)
  params = SimpleNamespace(steerRatio=15.0, wheelbase=2.8)
  previous, mapped_frames = -0.23, 0
  targets = [20.0] * 4 + [19.8] * 2 + [20.0] * 3 + [20.3] * 2 + [20.0] * 4
  for target in targets:
    previous = get_cruise_accel(False, target, 20.23, previous, 0.0, params, 0.05, ACCEL_MAX, True)
    raw = response.update(target, 20.23, get_max_accel(20.23), measured_accel=0.0)
    assert raw == pytest.approx(previous, rel=0.0, abs=1e-12)
    mapped = get_cruise_coast_accel(raw)
    mapped_frames += mapped != raw
  assert mapped_frames > 0
