"""A shorter UNIFIED standard gap with the original standard response weights."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.honda.values import CAR
from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  A_CHANGE_COST, COMFORT_STOP_DISTANCE, J_EGO_COST, LEAD_DANGER_FACTOR, MPC_INVALID_INPUT,
  N, RISK_STOP_DISTANCE, STOP_DISTANCE, LongitudinalMpc, MpcReference, get_T_FOLLOW, get_jerk_factor,
)
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import low_speed_follow_distance
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import advance, make_planner, make_state
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_mpc_problem_contract import NativeProblemRecorder
from openpilot.selfdrive.controls.tests.test_radar_fusion_mpc import (
  EPOCH_NS, action, advance as advance_lead, fused_state, object_lead,
)
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state


STANDARD = log.LongitudinalPersonality.standard


@pytest.mark.parametrize('speed_kph,expected', [
  (0.0, 1.25), (10.0, 1.25), (20.0, 1.25), (40.0, 1.25), (60.0, 1.25), (80.0, 1.25),
  (85.0, 1.28125), (90.0, 1.35), (95.0, 1.41875), (100.0, 1.45), (120.0, 1.45),
])
def test_standard_headway_has_full_shortening_through_eighty(speed_kph, expected):
  assert get_T_FOLLOW(STANDARD, v_ego=speed_kph * CV.KPH_TO_MS) == pytest.approx(expected)
  assert get_jerk_factor(STANDARD) == 1.0


def test_transition_is_monotonic_and_has_no_headway_or_slope_step():
  speeds = np.linspace(0.0, 40.0, 1001)
  times = np.array([get_T_FOLLOW(STANDARD, v_ego=speed) for speed in speeds])
  assert np.all((times >= 1.25) & (times <= 1.45))
  assert np.all(np.diff(times) >= -1e-12)
  # The equal-speed distance must also grow continuously as speed increases.
  assert np.all(np.diff(6.0 + speeds * times) > 0.0)
  step = 1e-3
  for boundary in (80.0 * CV.KPH_TO_MS, 100.0 * CV.KPH_TO_MS):
    center = get_T_FOLLOW(STANDARD, v_ego=boundary)
    slopes = [(get_T_FOLLOW(STANDARD, v_ego=boundary + sign * step) - center) / (sign * step)
              for sign in (-1, 1)]
    assert max(abs(value) for value in slopes) < 1e-4


@pytest.mark.parametrize('personality,expected', [
  (log.LongitudinalPersonality.relaxed, 1.75), (log.LongitudinalPersonality.aggressive, 1.25),
])
@pytest.mark.parametrize('speed_kph', [0.0, 60.0, 80.0, 90.0, 120.0])
def test_other_personalities_keep_their_original_time(personality, expected, speed_kph):
  assert get_T_FOLLOW(personality, v_ego=speed_kph * CV.KPH_TO_MS) == expected


@pytest.mark.parametrize('speed', [True, np.bool_(False), '20', float('nan'), float('inf'), float('-inf')])
def test_invalid_schedule_speed_is_rejected(speed):
  with pytest.raises(ValueError, match='following-policy speed'):
    get_T_FOLLOW(STANDARD, v_ego=speed)


def test_static_default_and_native_generation_keep_original_parameters():
  assert get_T_FOLLOW() == 1.45
  assert get_T_FOLLOW(log.LongitudinalPersonality.relaxed) == 1.75
  assert get_T_FOLLOW(log.LongitudinalPersonality.aggressive) == 1.25


@pytest.mark.parametrize('speed_kph,expected', [(60.0, 1.25), (80.0, 1.25), (90.0, 1.35), (100.0, 1.45)])
def test_scheduled_time_reaches_native_geometry_with_standard_weights(speed_kph, expected):
  speed = speed_kph * CV.KPH_TO_MS
  mpc = LongitudinalMpc(speed_based_following=True)
  recorder = NativeProblemRecorder(mpc.solver)
  mpc.solver = recorder
  mpc.set_cur_state(speed, 0.0)
  mpc.set_weights(True, personality=STANDARD)
  reference = MpcReference(speed + 5.0, comfort_stop_distance=low_speed_follow_distance(speed))
  radar = radar_state(distance=6.0 + 1.45 * speed + 5.0, speed=speed)
  mpc.update(radar, personality=STANDARD, reference=reference)
  assert mpc.solution_status == 0
  np.testing.assert_allclose(mpc.params[:, 4], np.full(N + 1, expected), atol=1e-12)
  np.testing.assert_array_equal(mpc.params[:, 5], LEAD_DANGER_FACTOR)
  np.testing.assert_array_equal(mpc.params[:, COMFORT_STOP_DISTANCE], STOP_DISTANCE)
  np.testing.assert_array_equal(mpc.params[:, RISK_STOP_DISTANCE], STOP_DISTANCE)
  weights = recorder.solved_costs[-1][1, 'W']
  assert weights[4, 4] == A_CHANGE_COST == 200.0
  assert weights[5, 5] == J_EGO_COST == 5.0


@pytest.mark.parametrize('platform,stock,interceptor,enabled', [
  (CAR.HONDA_ACCORD, False, False, True), (CAR.HONDA_CIVIC_BOSCH, False, False, False),
  (CAR.HONDA_ACCORD, True, False, False), (CAR.HONDA_ACCORD, False, True, False),
])
def test_vehicle_policy_gate_selects_the_schedule(platform, stock, interceptor, enabled):
  planner, _ = make_planner(platform=platform, stock=stock, interceptor=interceptor)
  assert planner.mpc.speed_based_following is enabled
  assert planner.unified is enabled


def test_same_live_planner_enters_and_leaves_the_transition_without_a_latch():
  planner, _ = make_planner(speed=80.0 * CV.KPH_TO_MS)
  for frame, speed_kph in enumerate((80.0, 85.0, 90.0, 95.0, 100.0, 95.0, 90.0, 85.0, 80.0)):
    speed = speed_kph * CV.KPH_TO_MS
    state = make_state(speed=speed, target=speed + 5.0)
    state['selfdriveState'].personality = STANDARD
    planner.update(state, now_ns=advance(state, frame))
    assert planner.mpc.solution_status == 0
    assert planner.motion_intent.valid
    np.testing.assert_allclose(planner.mpc.params[:, 4], get_T_FOLLOW(STANDARD, v_ego=state['carState'].vEgo))


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_measured_state_still_fails_before_native_solve(value):
  mpc = LongitudinalMpc(speed_based_following=True)
  mpc.set_cur_state(value, 0.0)
  native = Mock(wraps=mpc.solver)
  with patch.object(mpc, 'solver', native):
    mpc.update(radar_state(), reference=MpcReference(20.0))
  assert mpc.solution_status == MPC_INVALID_INPUT
  native.solve.assert_not_called()


@pytest.mark.parametrize('speed_kph', [20.0, 60.0, 80.0, 90.0, 100.0])
@pytest.mark.parametrize('lead_slot', [0, 1])
def test_formal_target_identity_switch_retains_immediate_braking_with_short_standard_time(speed_kph, lead_slot):
  speed = speed_kph * CV.KPH_TO_MS
  mpc = LongitudinalMpc(speed_based_following=True)
  mpc.set_cur_state(speed, 0.0)
  reference = MpcReference(speed + 5.0)
  original = object_lead(1, distance=6.0 + 1.45 * speed + 8.0, speed=speed, ego_speed=speed)
  for frame in range(6):
    stamp = EPOCH_NS + frame * 50_000_000
    advance_lead(original, stamp)
    previous = mpc.prepare_reference_history(stamp, valid=frame > 0)
    mpc.set_weights(previous, personality=STANDARD)
    mpc.update(fused_state(display=(original,)), personality=STANDARD, reference=reference,
               now_ns=stamp, reference_ns=stamp)
    assert mpc.solution_status == 0
  assert action(mpc) > 0.0
  assert mpc.source_object_identity == ('radar-test', 1)

  stamp += 50_000_000
  advance_lead(original, stamp)
  replacement = object_lead(2, distance=.75 * (6.0 + 1.25 * speed), speed=speed - 4.0,
                            ego_speed=speed, acceleration=-1.0, epoch=stamp)
  replacement.evidence.generation = 1
  radar = fused_state(display=(replacement,) if lead_slot == 0 else (original, replacement))
  previous = mpc.prepare_reference_history(stamp, valid=True)
  mpc.set_weights(previous, personality=STANDARD)
  mpc.update(radar, personality=STANDARD, reference=reference, now_ns=stamp, reference_ns=stamp)
  assert mpc.solution_status == 0
  assert mpc.source_object_identity == ('radar-test', 2)
  assert ('radar-test', 2) in mpc.object_influence_identities
  assert action(mpc) < 0.0
  np.testing.assert_allclose(mpc.params[:, 4], get_T_FOLLOW(STANDARD, v_ego=speed))
  np.testing.assert_array_equal(mpc.params[:, 5], LEAD_DANGER_FACTOR)
  assert np.min(mpc.a_solution[:-1]) >= -3.5 - 1e-3
