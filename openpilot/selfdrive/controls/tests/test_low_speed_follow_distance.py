from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.honda.values import CAR
from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_STOP_DISTANCE, LEAD_DANGER_FACTOR, MODEL_BRAKING_PRIORITY, N, PARAM_DIM, STOP_DISTANCE, T_IDXS,
  LongitudinalMpc, MpcReference, get_safe_obstacle_distance, get_stopped_equivalence_factor, get_T_FOLLOW,
)
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import low_speed_follow_distance
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import advance, make_planner, make_state
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_typed_reference_mpc import MODEL_TIMES, typed_reference
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state, solve_reference


PERSONALITIES = (
  (log.LongitudinalPersonality.relaxed, 1.75),
  (log.LongitudinalPersonality.standard, 1.45),
  (log.LongitudinalPersonality.aggressive, 1.25),
)


@pytest.mark.parametrize('personality,expected', PERSONALITIES)
@pytest.mark.parametrize('with_reference', [False, True])
def test_default_mpc_and_legacy_keep_original_gap(personality, expected, with_reference):
  speed = 10.0 * CV.KPH_TO_MS
  mpc = LongitudinalMpc()
  mpc.set_cur_state(speed, 0.0)
  reference = MpcReference(speed) if with_reference else None
  mpc.update(radar_state(), personality=personality, reference=reference)
  assert mpc.solution_status == 0
  assert get_T_FOLLOW(personality) == expected
  np.testing.assert_array_equal(mpc.params[:, 4], np.full(N+1, expected))
  np.testing.assert_array_equal(mpc.params[:, COMFORT_STOP_DISTANCE], np.full(N+1, 6.0))


@pytest.mark.parametrize('distance', [4.0, 5.0, 6.0, np.float32(4.0)])
@pytest.mark.parametrize('lead_index', [0, 1])
def test_real_lead_comfort_reaches_every_native_stage_without_moving_risk_geometry(distance, lead_index):
  speed = 10.0 * CV.KPH_TO_MS
  radar = radar_state(distance=8.0, speed=speed)
  if lead_index == 1:
    radar.leadOne, radar.leadTwo = radar.leadTwo, radar.leadOne
  mpc = LongitudinalMpc()
  mpc.set_cur_state(speed, 0.0)
  mpc.diagnostic_details_enabled = True
  native = mpc.solver
  writes = {}

  def record_set(stage, field, value):
    if field == 'p':
      writes[stage] = np.array(value, copy=True)
    return native.set(stage, field, value)

  recorded = Mock(wraps=native)
  recorded.set.side_effect = record_set
  reference = MpcReference(30.0, comfort_stop_distance=distance)
  with patch.object(mpc, 'solver', recorded):
    mpc.update(radar, reference=reference)
  assert mpc.solution_status == 0
  assert type(reference.comfort_stop_distance) is float
  assert set(writes) == set(range(N+1))
  expected_raw = 8.0 + speed * T_IDXS + get_stopped_equivalence_factor(speed)
  np.testing.assert_allclose(mpc.params[:, 2], expected_raw)
  for stage, parameters in writes.items():
    assert parameters.shape == (PARAM_DIM,)
    assert parameters[COMFORT_STOP_DISTANCE] == pytest.approx(float(distance))
    assert parameters[4] == 1.45
    assert parameters[5] == LEAD_DANGER_FACTOR
    assert parameters[2] == pytest.approx(expected_raw[stage])
  np.testing.assert_array_equal(mpc.diagnostic_last_parameters, np.array([writes[stage] for stage in range(N+1)]))


@pytest.mark.parametrize('value', [
  True, False, None, np.bool_(True), '4.0', 4.0 + 0j, [4.0], np.array(4.0), np.array([4.0]),
  float('nan'), float('inf'), float('-inf'), 3.999999, 6.000001,
])
def test_invalid_comfort_distance_is_rejected_before_solver(value):
  with pytest.raises(ValueError, match='Invalid comfort stop distance'):
    MpcReference(10.0, comfort_stop_distance=value)


@pytest.mark.parametrize('speed', [0.0, 1.0, 10.0 * CV.KPH_TO_MS, 20.0])
@pytest.mark.parametrize('model', [False, True])
def test_virtual_only_references_remain_exactly_unchanged_with_four_meter_request(speed, model):
  fields = {}
  if model:
    fields = {'model_position': speed * T_IDXS, 'model_velocity': np.full(N+1, speed), 'model_acceleration': np.zeros(N+1)}
  reference = MpcReference(speed, **fields)
  original = solve_reference(speed, reference)
  compact = solve_reference(speed, replace(reference, comfort_stop_distance=4.0))
  for name in ('params', 'yref', 'x_sol', 'u_sol'):
    np.testing.assert_array_equal(getattr(compact, name), getattr(original, name))
  np.testing.assert_array_equal(compact.params[:, COMFORT_STOP_DISTANCE], 6.0)
  # At rest the original virtual obstacle retains its 1.5 m soft-risk margin;
  # shortening that obstacle itself to 4 m would invent a 0.5 m deficit.
  if speed == 0.0:
    np.testing.assert_array_equal(compact.params[:, 2] - 0.75 * 6.0, 1.5)


@pytest.mark.parametrize('virtual_source', ['cruise', 'model'])
def test_virtual_limit_wins_comfort_crossover_without_accelerating_past_its_speed(virtual_source):
  speed = 10.0 * CV.KPH_TO_MS
  radar = radar_state(distance=9.0, speed=speed)
  fields = {}
  if virtual_source == 'model':
    fields = {'model_position': speed * T_IDXS, 'model_velocity': np.full(N+1, speed), 'model_acceleration': np.zeros(N+1)}
  reference = MpcReference(speed if virtual_source == 'cruise' else 30.0, comfort_stop_distance=4.0, **fields)
  compact = solve_reference(speed, reference, radar=radar)
  # 9 m lies between the compact 8.0278 m goal and the original 10.0278 m
  # goal. The lead still owns risk geometry, while the virtual speed cap owns
  # comfort. Treating both minima as the lead would ask for excess speed.
  np.testing.assert_allclose(compact.params[:, 2], 9.0 + speed * T_IDXS + speed**2 / 5.0)
  np.testing.assert_allclose(compact.params[:, COMFORT_STOP_DISTANCE], 9.0 - 1.45 * speed)
  np.testing.assert_allclose(compact.v_solution, speed, atol=1e-4)
  np.testing.assert_allclose(compact.a_solution, 0.0, atol=1e-4)


def test_typed_comfort_source_does_not_change_risk_braking_priority():
  speed = 10.0 * CV.KPH_TO_MS
  reference = replace(typed_reference(np.full(len(MODEL_TIMES), speed), speed, cruise=speed), comfort_stop_distance=4.0)
  compact = solve_reference(speed, reference, radar=radar_state(distance=9.0, speed=speed))
  np.testing.assert_allclose(compact.params[:, COMFORT_STOP_DISTANCE], 9.0 - 1.45 * speed)
  np.testing.assert_array_equal(compact.diagnostic_reference_weights[:, 0], 0.0)
  np.testing.assert_array_equal(compact.params[:, MODEL_BRAKING_PRIORITY], 1.0)
  np.testing.assert_allclose(compact.v_solution, speed, atol=1e-4)


def test_generated_cost_changes_comfort_without_reducing_the_original_risk_row(monkeypatch):
  from acados.acados_template import AcadosModel, AcadosOcp
  from casadi import Function

  monkeypatch.setattr(long_mpc, 'AcadosModel', AcadosModel, raising=False)
  monkeypatch.setattr(long_mpc, 'AcadosOcp', AcadosOcp, raising=False)
  model = long_mpc.gen_long_ocp().model
  evaluate = Function('comfort_and_risk', [model.x, model.u, model.p], [model.cost_y_expr, model.con_h_expr])
  parameters = np.array([-3.5, 2.0, 4.0, 0.0, 1.45, 0.75, 0.0, 0.0, 0.0, 0.0, 6.0])
  for speed in (0.0, 2.0, 10.0):
    for distance in (4.0, 5.0, 6.0):
      parameters[COMFORT_STOP_DISTANCE] = distance
      cost, constraints = (np.asarray(value).ravel() for value in evaluate([0.0, speed, 0.0], [0.0], parameters))
      assert cost[0] == pytest.approx((4.0 - speed**2 / 5.0 - 1.45 * speed - distance) / (speed + 10.0))
      assert constraints[3] == pytest.approx((4.0 - 0.75 * (speed**2 / 5.0 + 1.45 * speed + 6.0)) / (speed + 10.0))
  # A 4 m comfort target at rest still has the original risk deficit.
  parameters[COMFORT_STOP_DISTANCE] = 4.0
  cost, constraints = evaluate([0.0, 0.0, 0.0], [0.0], parameters)
  assert float(cost[0]) == 0.0
  assert float(constraints[3]) == pytest.approx(-0.05)


@pytest.mark.parametrize('speed_kph,distance', [(0.0, 4.0), (10.0, 4.0), (15.0, 4.0), (22.5, 5.0), (30.0, 6.0), (50.0, 6.0)])
@pytest.mark.parametrize('personality,follow_time', PERSONALITIES)
def test_unified_producer_selects_comfort_distance_without_changing_personality(speed_kph, distance, personality, follow_time):
  speed = speed_kph * CV.KPH_TO_MS
  assert low_speed_follow_distance(speed) == pytest.approx(distance)
  planner, _ = make_planner(speed=speed)
  state = make_state(speed=speed, target=max(speed, 10.0))
  state['selfdriveState'].personality = personality
  with patch.object(planner.mpc, 'update', wraps=planner.mpc.update) as update:
    planner.update(state, now_ns=advance(state, 0))
  assert planner.unified
  assert planner.mpc.solution_status == 0
  assert update.call_args.kwargs['reference'].comfort_stop_distance == pytest.approx(distance)
  np.testing.assert_array_equal(planner.mpc.params[:, 4], np.full(N+1, follow_time))
  # No real lead: the virtual cruise geometry keeps its original base.
  np.testing.assert_array_equal(planner.mpc.params[:, COMFORT_STOP_DISTANCE], 6.0)


def test_legacy_vehicle_does_not_consume_unified_distance_policy():
  speed = 10.0 * CV.KPH_TO_MS
  planner, _ = make_planner(speed=speed, platform=CAR.HONDA_CIVIC_BOSCH)
  state = make_state(speed=speed)
  state['selfdriveState'].personality = log.LongitudinalPersonality.standard
  with patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.low_speed_follow_distance',
             side_effect=AssertionError('UNIFIED gap policy used by LEGACY')):
    planner.update(state, now_ns=advance(state, 0))
  assert not planner.unified
  assert planner.mpc.solution_status == 0
  np.testing.assert_array_equal(planner.mpc.params[:, 4], np.full(N+1, 1.45))
  np.testing.assert_array_equal(planner.mpc.params[:, COMFORT_STOP_DISTANCE], 6.0)
  assert STOP_DISTANCE == get_safe_obstacle_distance(0.0, 1.45) == 6.0
