"""Explicit stationary-lead geometry, independently of producer qualification.

These native and mathematical contracts do not establish delivered car spacing.
"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.selfdrive.controls.lib.drive_helpers import STOP_RELEASE_ACCEL, STOP_SPEED, get_accel_from_plan
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_STOP_DISTANCE, LEAD_DANGER_FACTOR, MPC_SOURCES, N, PARAM_DIM, QP_FEASIBILITY_TOL,
  RISK_STOP_DISTANCE, STOP_DISTANCE, T_IDXS, LongitudinalMpc, LongitudinalPlanSource, MpcReference,
)
from openpilot.selfdrive.controls.tests.test_due_rest_reference import typed_due_reference
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state, solve_reference


@pytest.fixture
def native_equations(monkeypatch):
  from acados.acados_template import AcadosModel, AcadosOcp
  from casadi import Function

  monkeypatch.setattr(long_mpc, 'AcadosModel', AcadosModel, raising=False)
  monkeypatch.setattr(long_mpc, 'AcadosOcp', AcadosOcp, raising=False)
  ocp = long_mpc.gen_long_ocp()
  model = ocp.model
  evaluate = Function('stationary_lead_rows', [model.x, model.u, model.p], [model.cost_y_expr, model.con_h_expr])
  return evaluate, ocp.parameter_values.copy()


def test_three_meter_comfort_and_four_meter_risk_base_agree_at_rest(native_equations):
  evaluate, parameters = native_equations
  assert parameters.shape == (PARAM_DIM,) == (12,)
  assert parameters[COMFORT_STOP_DISTANCE] == parameters[RISK_STOP_DISTANCE] == STOP_DISTANCE
  parameters[2] = 3.0
  parameters[COMFORT_STOP_DISTANCE], parameters[RISK_STOP_DISTANCE] = 3.0, 4.0
  cost, constraints = evaluate([0.0, 0.0, 0.0], [0.0], parameters)
  assert float(cost[0]) == 0.0
  assert float(constraints[3]) == 0.0
  parameters[2] = 2.99
  _, constraints = evaluate([0.0, 0.0, 0.0], [0.0], parameters)
  assert float(constraints[3]) < 0.0


@pytest.mark.parametrize('follow_time', [1.25, 1.45, 1.75])
@pytest.mark.parametrize('speed', [0.0, 1.0, 5.0, 20.0])
def test_compact_risk_preserves_speed_squared_and_time_gap_terms(native_equations, follow_time, speed):
  evaluate, parameters = native_equations
  parameters[2], parameters[4] = 20.0, follow_time
  parameters[COMFORT_STOP_DISTANCE] = 3.0
  for risk_base in (4.0, 5.0, 6.0):
    parameters[RISK_STOP_DISTANCE] = risk_base
    cost, constraints = evaluate([0.0, speed, 0.0], [0.0], parameters)
    expected_dynamic_distance = speed**2 / 5.0 + follow_time * speed
    assert float(cost[0]) == pytest.approx((20.0 - expected_dynamic_distance - 3.0) / (speed + 10.0))
    assert float(constraints[3]) == pytest.approx(
      (20.0 - 0.75 * (expected_dynamic_distance + risk_base)) / (speed + 10.0),
    )
  assert parameters[5] == LEAD_DANGER_FACTOR == 0.75
  assert long_mpc.COMFORT_BRAKE == 2.5 and STOP_DISTANCE == 6.0


@pytest.mark.parametrize('value', [
  None, True, False, 4.0, '4,6', [4.0, 6.0], np.array([4.0, 6.0]), (), (4.0,), (4.0, 5.0, 6.0),
  (True, 6.0), (4.0, np.bool_(False)), ('4.0', 6.0), (4.0 + 0j, 6.0), (np.array(4.0), 6.0),
  (float('nan'), 6.0), (6.0, float('inf')), (float('-inf'), 6.0), (3.999999, 6.0), (4.0, 6.000001), (10**100, 6.0),
])
def test_invalid_per_lead_risk_bases_are_rejected(value):
  with pytest.raises(ValueError, match='Invalid lead risk stop distances'):
    MpcReference(10.0, lead_risk_stop_distances=value)


def test_risk_bases_are_normalized_to_an_immutable_float_pair():
  reference = MpcReference(None, comfort_stop_distance=np.float32(3.0), lead_risk_stop_distances=(np.float32(4.0), 6))
  assert type(reference.comfort_stop_distance) is float
  assert reference.lead_risk_stop_distances == (4.0, 6.0)
  assert all(type(value) is float for value in reference.lead_risk_stop_distances)
  assert MpcReference(None).lead_risk_stop_distances == (STOP_DISTANCE, STOP_DISTANCE)


def install_obstacles(mpc, obstacles, sources, risk_bases, *, factor=LEAD_DANGER_FACTOR):
  mpc.params[:, 0], mpc.params[:, 1] = ACCEL_MIN, ACCEL_MAX
  mpc.params[:, 2] = np.min(obstacles, axis=1)
  mpc.params[:, 4], mpc.params[:, 5] = 1.45, factor
  mpc.set_reference_problem(MpcReference(None, comfort_stop_distance=3.0, lead_risk_stop_distances=risk_bases),
                            obstacles, sources, ACCEL_MIN, 1.45)


@pytest.mark.parametrize('risk_bases', [(4.0, 6.0), (6.0, 4.0), (4.0, 4.0), (5.0, 4.0), (6.0, 6.0)])
def test_each_node_uses_the_tightest_real_or_virtual_risk_row(native_equations, risk_bases):
  evaluate, _ = native_equations
  sources = [*MPC_SOURCES, LongitudinalPlanSource.cruise, LongitudinalPlanSource.e2e]
  # Both lead slots and both virtual sources take turns owning the constraint.
  obstacles = np.array([
    [3.0, 9.0, 20.0, 20.0], [9.0, 3.0, 20.0, 20.0], [4.0, 4.5, 20.0, 20.0],
    [4.5, 4.0, 20.0, 20.0], [9.0, 9.0, 3.0, 20.0], [9.0, 9.0, 20.0, 3.0],
    [4.0, 9.0, 4.5, 20.0], [9.0, 4.0, 20.0, 4.5], [5.0, 5.0, 5.0, 5.0],
    [np.inf, 3.0, 20.0, 20.0], [3.0, np.inf, 20.0, 20.0],
    [np.inf, np.inf, 6.0, 20.0], [np.inf, np.inf, 20.0, 6.0],
  ])
  before = obstacles.copy()
  mpc = LongitudinalMpc()
  install_obstacles(mpc, obstacles, sources, risk_bases)
  np.testing.assert_array_equal(obstacles, before)
  np.testing.assert_array_equal(mpc.params[:, 2], np.min(obstacles, axis=1))
  bases = np.array([*risk_bases, 6.0, 6.0])
  for stage, speed in enumerate(np.linspace(0.0, 6.0, N+1)):
    _, constraints = evaluate([0.0, speed, 0.0], [0.0], mpc.params[stage])
    expected = np.min(obstacles[stage] - 0.75 * (speed**2 / 5.0 + 1.45 * speed + bases)) / (speed + 10.0)
    assert float(constraints[3]) == pytest.approx(expected)
  # Reordering columns must preserve source identity and therefore the result.
  reordered = LongitudinalMpc()
  permutation = [1, 0, 3, 2]
  install_obstacles(reordered, obstacles[:, permutation], [sources[index] for index in permutation], risk_bases)
  np.testing.assert_array_equal(reordered.params, mpc.params)
  assert np.all((4.0 <= mpc.params[:, RISK_STOP_DISTANCE]) & (mpc.params[:, RISK_STOP_DISTANCE] <= 6.0))
  np.testing.assert_array_equal(mpc.params[[4, 5, 11, 12], RISK_STOP_DISTANCE], 6.0)


@pytest.mark.parametrize('factor', [0.0, -0.75, float('nan'), float('inf')])
def test_risk_folding_rejects_an_invalid_lead_factor(factor):
  mpc = LongitudinalMpc()
  with pytest.raises(ValueError, match='Invalid lead danger factor'):
    install_obstacles(mpc, np.full((N+1, 2), 8.0), list(MPC_SOURCES), (4.0, 6.0), factor=factor)


@pytest.mark.parametrize('lead_index', [0, 1])
def test_three_meter_stationary_lead_parameters_reach_every_native_stage(lead_index):
  radar = radar_state(distance=3.0)
  if lead_index:
    radar.leadOne, radar.leadTwo = radar.leadTwo, radar.leadOne
  risk_bases = (4.0, 6.0) if lead_index == 0 else (6.0, 4.0)
  reference = MpcReference(None, comfort_stop_distance=3.0, lead_risk_stop_distances=risk_bases)
  mpc = LongitudinalMpc()
  mpc.set_cur_state(0.0, 0.0)
  mpc.diagnostic_details_enabled = True
  native, writes = mpc.solver, {}

  def record_set(stage, field, value):
    if field == 'p':
      writes[stage] = np.array(value, copy=True)
    return native.set(stage, field, value)

  recorded = Mock(wraps=native)
  recorded.set.side_effect = record_set
  with patch.object(mpc, 'solver', recorded):
    mpc.update(radar, reference=reference)
  assert mpc.solution_status == mpc.last_solver_status == 0
  assert set(writes) == set(range(N+1))
  for parameters in writes.values():
    assert parameters.shape == (PARAM_DIM,)
    assert parameters[2] == 3.0
    assert parameters[COMFORT_STOP_DISTANCE] == 3.0
    assert parameters[RISK_STOP_DISTANCE] == 4.0
    assert parameters[4] == 1.45 and parameters[5] == 0.75
  np.testing.assert_array_equal(mpc.diagnostic_last_parameters, [writes[stage] for stage in range(N+1)])
  # An untyped reference has soft speed/distance rows, with no native rest
  # constraint. A successful RTI solve need not return exact zero motion at
  # the zero-cost equilibrium; use its actual finite/dynamics contract.
  assert mpc.rest_goal is None
  assert np.all(np.isfinite(mpc.x_sol)) and np.all(np.isfinite(mpc.j_solution))
  np.testing.assert_allclose(mpc.x_sol[0], mpc.x0, atol=QP_FEASIBILITY_TOL, rtol=0.0)
  assert np.max(np.abs(mpc.trajectory_dynamics_defects(mpc.x_sol, mpc.j_solution))) <= QP_FEASIBILITY_TOL
  # These existing thresholds bound this mathematical fixture's output. They
  # do not replace the independent owner/actuator HOLD and release contracts.
  assert np.max(mpc.v_solution) < STOP_SPEED
  assert get_accel_from_plan(mpc.v_solution, mpc.a_solution, T_IDXS, action_t=0.55) < STOP_RELEASE_ACCEL


@pytest.mark.parametrize('lead_index', [0, 1])
def test_compact_lead_preserves_the_typed_due_rest_contract(lead_index):
  radar = radar_state(distance=3.0)
  if lead_index:
    radar.leadOne, radar.leadTwo = radar.leadTwo, radar.leadOne
  risk_bases = (4.0, 6.0) if lead_index == 0 else (6.0, 4.0)
  reference = replace(typed_due_reference(0.0), comfort_stop_distance=3.0, lead_risk_stop_distances=risk_bases)
  mpc = LongitudinalMpc()
  mpc.set_cur_state(0.0, 0.0)
  mpc.update(radar, reference=reference)
  assert mpc.solution_status == mpc.last_solver_status == 0
  assert mpc.rest_goal.native_node == 0 and mpc.rest_goal_state == 'satisfied'
  np.testing.assert_array_equal(mpc.params[:, COMFORT_STOP_DISTANCE], 3.0)
  np.testing.assert_array_equal(mpc.params[:, RISK_STOP_DISTANCE], 4.0)
  lower, upper = mpc.diagnostic_reference_bounds
  np.testing.assert_array_equal(lower[1:], 0.0)
  np.testing.assert_array_equal(upper[1:], 0.0)
  assert np.all(np.isfinite(mpc.x_sol)) and np.all(np.isfinite(mpc.j_solution))
  np.testing.assert_allclose(mpc.x_sol[0], mpc.x0, atol=QP_FEASIBILITY_TOL, rtol=0.0)
  assert np.all(mpc.x_sol[1:, 1:] >= lower[1:] - QP_FEASIBILITY_TOL)
  assert np.all(mpc.x_sol[1:, 1:] <= upper[1:] + QP_FEASIBILITY_TOL)
  assert np.max(np.abs(mpc.trajectory_dynamics_defects(mpc.x_sol, mpc.j_solution))) <= QP_FEASIBILITY_TOL
  assert mpc.trajectory_minimum_velocity(mpc.x_sol, mpc.j_solution) >= -QP_FEASIBILITY_TOL
  assert get_accel_from_plan(mpc.v_solution, mpc.a_solution, T_IDXS, action_t=0.55) < STOP_RELEASE_ACCEL


@pytest.mark.parametrize('cruise,model', [(None, False), (0.0, False), (10.0, False), (10.0, True)])
def test_absent_leads_and_virtual_obstacles_keep_the_original_problem(cruise, model):
  fields = {}
  if model:
    fields = {'model_position': 2.0 * T_IDXS, 'model_velocity': np.full(N+1, 2.0), 'model_acceleration': np.zeros(N+1)}
  reference = MpcReference(cruise, **fields)
  ordinary = solve_reference(2.0, reference)
  compact = solve_reference(2.0, replace(reference, comfort_stop_distance=3.0, lead_risk_stop_distances=(4.0, 4.0)))
  for name in ('params', 'yref', 'x_sol', 'u_sol'):
    np.testing.assert_array_equal(getattr(compact, name), getattr(ordinary, name))
  np.testing.assert_array_equal(compact.params[:, RISK_STOP_DISTANCE], 6.0)


@pytest.mark.parametrize('present_index', [0, 1])
def test_a_compact_request_for_an_absent_slot_does_not_relax_the_other_lead(present_index):
  radar = radar_state(distance=8.0)
  if present_index:
    radar.leadOne, radar.leadTwo = radar.leadTwo, radar.leadOne
  bases = (6.0, 4.0) if present_index == 0 else (4.0, 6.0)
  mpc = LongitudinalMpc()
  mpc.update(radar, reference=MpcReference(None, lead_risk_stop_distances=bases))
  assert mpc.solution_status == 0
  np.testing.assert_array_equal(mpc.params[:, RISK_STOP_DISTANCE], 6.0)


@pytest.mark.parametrize('with_reference', [False, True])
def test_default_none_and_reset_do_not_retain_previous_compact_risk(with_reference):
  mpc = LongitudinalMpc()
  radar = radar_state(distance=8.0)
  reference = MpcReference(None, comfort_stop_distance=3.0, lead_risk_stop_distances=(4.0, 6.0))
  mpc.update(radar, reference=reference)
  assert mpc.solution_status == 0
  np.testing.assert_array_equal(mpc.params[:, RISK_STOP_DISTANCE], 4.0)
  mpc.update(radar, reference=MpcReference(None) if with_reference else None)
  assert mpc.solution_status == 0
  np.testing.assert_array_equal(mpc.params[:, RISK_STOP_DISTANCE], 6.0)
  mpc.update(radar, reference=reference)
  np.testing.assert_array_equal(mpc.params[:, RISK_STOP_DISTANCE], 4.0)
  mpc.reset()
  np.testing.assert_array_equal(mpc.params[:, RISK_STOP_DISTANCE], 6.0)


def rest_parameters(risk_base=4.0):
  parameters = np.zeros((N+1, PARAM_DIM))
  parameters[:, 0], parameters[:, 1] = ACCEL_MIN, ACCEL_MAX
  parameters[:, 4], parameters[:, 5] = 1.45, LEAD_DANGER_FACTOR
  parameters[:, COMFORT_STOP_DISTANCE], parameters[:, RISK_STOP_DISTANCE] = 3.0, risk_base
  return parameters


@pytest.mark.parametrize('risk_base', [4.0, 5.0, 6.0])
def test_native_row_and_rest_acceptance_use_the_same_stationary_boundary(native_equations, risk_base):
  evaluate, _ = native_equations
  states, jerks = np.zeros((N+1, 3)), np.zeros(N)
  parameters = rest_parameters(risk_base)
  parameters[:, 2] = 0.75 * risk_base
  _, constraints = evaluate(states[0], [0.0], parameters[0])
  assert float(constraints[3]) == 0.0
  assert LongitudinalMpc.scheduled_rest_trajectory_error(
    states[0], states, jerks, 1, parameters, tolerance=QP_FEASIBILITY_TOL,
  ) == ''
  parameters[:, 2] -= 0.01
  _, constraints = evaluate(states[0], [0.0], parameters[0])
  assert float(constraints[3]) < 0.0
  assert LongitudinalMpc.scheduled_rest_trajectory_error(
    states[0], states, jerks, 1, parameters, tolerance=QP_FEASIBILITY_TOL,
  ) == 'scheduled_rest_lead_danger_envelope_violated'


def test_rest_recovery_and_diagnostic_use_the_same_compact_base_through_arrival():
  # No solver: construct a schedulable trajectory and a decreasing metre deficit.
  mpc = LongitudinalMpc.__new__(LongitudinalMpc)
  mpc.x0 = np.array([0.0, 0.1, 0.0])
  states, jerks = LongitudinalMpc.future_rest_primal(mpc.x0, 2)
  mpc.params = rest_parameters()
  deficits = np.zeros(N+1)
  deficits[:2] = [1.0, 0.5]
  desired = 0.75 * (states[:, 1]**2 / 5.0 + 1.45 * states[:, 1] + 4.0)
  mpc.params[:, 2] = states[:, 0] + desired - deficits
  mpc.rest_recovery_position_limit = 1.0
  mpc.rest_goal = SimpleNamespace(native_node=1, native_time_s=float(T_IDXS[1]), remaining_time_s=float(T_IDXS[1]))
  mpc.diagnostic_rest_goal = {}
  assert LongitudinalMpc.scheduled_rest_trajectory_error(
    mpc.x0, states, jerks, 2, mpc.params, tolerance=QP_FEASIBILITY_TOL, recovery_position_limit=1.0,
  ) == ''
  scheduled = mpc.schedule_future_rest()
  assert scheduled is not None and mpc.effective_rest_node == 2
  assert mpc.diagnostic_rest_goal['scheduledDangerPolicy'] == 'recover_initial_deficit'
  assert mpc.diagnostic_rest_goal['initialDangerDeficitMeters'] == pytest.approx(1.0)
  np.testing.assert_array_equal(scheduled[0], states)
  np.testing.assert_array_equal(scheduled[1], jerks)


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), 0.0, 3.999999, 6.000001, 'missing_column'])
def test_rest_acceptance_requires_the_complete_valid_risk_column(invalid):
  states, jerks = np.zeros((N+1, 3)), np.zeros(N)
  parameters = rest_parameters()
  parameters[:, 2] = 6.0
  if invalid == 'missing_column':
    parameters = parameters[:, :-1]
  else:
    parameters[-1, RISK_STOP_DISTANCE] = invalid
  assert LongitudinalMpc.scheduled_rest_trajectory_error(
    states[0], states, jerks, 1, parameters, tolerance=QP_FEASIBILITY_TOL,
  ) == 'invalid_scheduled_rest_trajectory'
