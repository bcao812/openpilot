"""Unknown occupancy tails keep braking pressure without rewarding speed."""
from copy import deepcopy

import numpy as np
import pytest
from casadi import Function, jacobian, vertcat

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.lead_qualification import qualified_lead
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_NORMALIZER, LEAD_DANGER_FACTOR, MPC_SOURCES, N, PARAM_DIM, QP_FEASIBILITY_TOL, RISK_NORMALIZER,
  T_IDXS, LongitudinalMpc, LongitudinalPlanSource, MpcReference,
)
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import sample_reference_acceleration
from openpilot.selfdrive.controls.tests.test_radar_fusion_mpc import EPOCH_NS, fused_state, object_lead
from openpilot.selfdrive.controls.tests.test_radar_occupancy_support import supported
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state


NORMALIZERS = (COMFORT_NORMALIZER, RISK_NORMALIZER)


def repeat_native(state, *, speed=10.0, acceleration=0.0, planner=None, iterations=12):
  planner = LongitudinalMpc() if planner is None else planner
  planner.set_cur_state(speed, acceleration)
  planner.set_weights(prev_accel_constraint=False)
  accepted = []
  for _ in range(iterations):
    planner.update(state, reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
    assert planner.solution_status == planner.last_solver_status == 0
    assert planner.previous_solution_valid
    assert planner.occupancy_trajectory_error(planner.x_sol, planner.j_solution,
                                              tolerance=QP_FEASIBILITY_TOL, post_native=True) == ''
    accepted.append((sample_reference_acceleration(planner.a_solution, .5 + DT_MDL),
                     planner.x_sol.copy(), planner.u_sol.copy()))
  return planner, accepted


def install_selection_problem(obstacles, unknown, *, comfort=4.0, risk=4.0, slots=(0, None, None)):
  planner = LongitudinalMpc()
  planner.set_cur_state(1.0, 0.0)
  planner.set_weights(prev_accel_constraint=False)
  matrix = np.broadcast_to(np.asarray(obstacles, dtype=float), (N+1, len(obstacles))).copy()
  masks = np.zeros_like(matrix, dtype=bool)
  for column in unknown:
    masks[1:, column] = True
  planner.params[:, 0], planner.params[:, 1] = ACCEL_MIN, ACCEL_MAX
  planner.params[:, 2] = np.min(matrix, axis=1)
  planner.params[:, 4], planner.params[:, 5] = 1.45, LEAD_DANGER_FACTOR
  reference = MpcReference(20.0, comfort_stop_distance=comfort, lead_risk_stop_distances=(risk, 6.0))
  sources = [*MPC_SOURCES, LongitudinalPlanSource.cruise]
  planner.set_reference_problem(reference, matrix, sources, ACCEL_MIN, 1.45, lead_slots=slots,
                                 unknown_tail_masks=masks)
  return planner


@pytest.mark.parametrize('unknown_distance,comfort,risk,expected', [
  (11.7, 4.0, 4.0, (11.0, 0.0)),  # Comfort picks unknown; risk still picks compact known lead.
  (11.3, 5.0, 4.0, (0.0, 11.0)),  # Risk picks unknown; comfort still picks known lead.
  (11.5, 4.0, 4.0, (11.0, 0.0)),  # Risk tie keeps the first known object.
])
def test_comfort_and_risk_choose_their_own_adjusted_minimum(unknown_distance, comfort, risk, expected):
  planner = install_selection_problem((10.0, unknown_distance, 100.0), (1,), comfort=comfort, risk=risk)
  np.testing.assert_array_equal(planner.params[0, NORMALIZERS], 0.0)
  np.testing.assert_array_equal(planner.params[1:, COMFORT_NORMALIZER], expected[0])
  np.testing.assert_array_equal(planner.params[1:, RISK_NORMALIZER], expected[1])


def test_first_minimum_tie_order_applies_to_unknown_and_virtual_objects_too():
  unknown_first = install_selection_problem((11.5, 10.0, 100.0), (0,), slots=(None, 0, None))
  np.testing.assert_array_equal(unknown_first.params[1:, NORMALIZERS], 11.0)
  virtual_nearer = install_selection_problem((10.0, 11.7, 5.0), (1,))
  no_unknown = install_selection_problem((10.0, 11.7, 5.0), ())
  np.testing.assert_array_equal(virtual_nearer.params, no_unknown.params)
  np.testing.assert_array_equal(virtual_nearer.params[:, NORMALIZERS], 0.0)
  virtual_nearer.run()
  no_unknown.run()
  assert virtual_nearer.solution_status == no_unknown.solution_status == 0
  np.testing.assert_array_equal(virtual_nearer.x_sol, no_unknown.x_sol)
  np.testing.assert_array_equal(virtual_nearer.u_sol, no_unknown.u_sol)


def test_floor_applies_before_cross_object_minimum_without_changing_raw_geometry():
  unknown = supported(object_lead(1, distance=5.0, speed=-10.0, persistent=False), 0.0)
  known = object_lead(2, distance=3.0, speed=-1.0, persistent=True)
  original = deepcopy(vars(unknown))
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  before = planner.project_fusion_lead(unknown, slot=None, reference_ns=EPOCH_NS)
  planner, _ = repeat_native(fused_state(unknown, known), planner=planner)
  first, second = planner._fusion_projections
  np.testing.assert_array_equal(first.raw_trajectory, before.raw_trajectory)
  np.testing.assert_array_equal(first.trajectory, before.trajectory)
  assert vars(unknown) == original
  obstacles = planner._influence_problem[0]
  unknown_b = first.trajectory[:, 0] + np.maximum(first.trajectory[:, 1], 0.0)**2 / 5.0
  known_b = second.trajectory[:, 0] + np.maximum(second.trajectory[:, 1], 0.0)**2 / 5.0
  np.testing.assert_array_equal(obstacles[1:, 2], np.maximum(unknown_b[1:], planner.x0[0]))
  np.testing.assert_array_equal(obstacles[:, 3], known_b)
  # A known object's own negative forecast is retained. A global floor of
  # the combined minimum would erase it and incorrectly retain both flags.
  assert known_b[-1] < 0.0 and planner.params[-1, 2] == known_b[-1]
  np.testing.assert_array_equal(planner.params[-1, NORMALIZERS], 0.0)
  assert first.lead.vLead == -10.0
  selected = np.argmin(obstacles, axis=1)
  expected = np.where((selected == 2) & (T_IDXS > 0.0), 20.0, 0.0)
  np.testing.assert_array_equal(planner.params[:, COMFORT_NORMALIZER], expected)
  np.testing.assert_array_equal(planner.params[:, RISK_NORMALIZER], expected)
  assert set(planner.object_influence_identities) == {('radar-test', 1), ('radar-test', 2)}
  assert not qualified_lead(first.lead, 'stop', 0.0, now_ns=EPOCH_NS)


def test_floored_tied_unknown_objects_retain_both_influences_without_stop_authority():
  first = supported(object_lead(1, distance=5.0, speed=-10.0, persistent=False), 0.0)
  second = supported(object_lead(2, distance=6.0, speed=-8.0, persistent=False), 0.0)
  planner, _ = repeat_native(fused_state(first, second))
  np.testing.assert_array_equal(planner._influence_problem[0][-1, 2:4], 0.0)
  assert set(planner.object_influence_identities) == {('radar-test', 1), ('radar-test', 2)}
  assert all(not qualified_lead(lead, 'stop', 0.0, now_ns=EPOCH_NS) for lead in (first, second))


@pytest.mark.parametrize('distance,speed,initial_accel,support_end', [
  (5.0, 10.0, 0.0, 0.0), (10.0, 20.0, 0.0, 0.0),
  (5.0, 10.0, 0.0, .05), (5.0, 10.0, 1.0, 0.0),
])
def test_unknown_tail_brakes_at_unified_horizon_and_does_not_reaccelerate_at_terminal(
  distance, speed, initial_accel, support_end,
):
  lead = supported(object_lead(distance=distance, speed=-10.0, ego_speed=speed, persistent=False), support_end)
  planner, accepted = repeat_native(fused_state(lead), speed=speed, acceleration=initial_accel)
  for request, states, _ in accepted:
    assert request <= -1.0  # The actual UNIFIED a(delay+DT_MDL), not the inverse velocity helper.
    assert np.min(states[1:, 2]) <= -3.0
  assert planner.v_solution[-1] <= .01
  assert np.max(planner.a_solution[T_IDXS >= 6.0]) <= .001
  assert planner.trajectory_minimum_velocity(planner.x_sol, planner.j_solution) >= -QP_FEASIBILITY_TOL
  assert np.all(planner.a_solution[1:] >= planner.params[1:, 0] - QP_FEASIBILITY_TOL)
  assert np.all(planner.a_solution[1:] <= planner.params[1:, 1] + QP_FEASIBILITY_TOL)


@pytest.mark.parametrize('speed,initial_accel', [(.1, -3.5), (.0009, -.1)])
def test_unknown_tail_keeps_the_positive_brake_release_needed_to_remain_forward(speed, initial_accel):
  lead = supported(object_lead(distance=.01, speed=-10.0, ego_speed=speed, persistent=False), 0.0)
  planner, _ = repeat_native(fused_state(lead), speed=speed, acceleration=initial_accel)
  # A hard a1<=0 would make v1<=v0+a0*h/2<0 for both inputs. The cost policy
  # must leave the physical positive-release solution available instead.
  assert speed + initial_accel*T_IDXS[1]/2.0 < 0.0
  assert planner.a_solution[1] > 0.0
  assert planner.diagnostic_reference_bounds[1][1, 1] == ACCEL_MAX
  assert planner.trajectory_minimum_velocity(planner.x_sol, planner.j_solution) >= -QP_FEASIBILITY_TOL


@pytest.mark.parametrize('kind', ['legacy', 'full_supported', 'persistent', 'empty', 'nonfusion', 'reset'])
def test_normalizers_clear_on_each_new_problem_and_reset(kind):
  unknown = supported(object_lead(distance=5.0, speed=-10.0, persistent=False), 0.0)
  planner, _ = repeat_native(fused_state(unknown))
  assert np.any(planner.params[:, NORMALIZERS] > 0.0)
  if kind == 'reset':
    planner.reset()
  else:
    next_lead = object_lead(distance=30.0, speed=5.0, persistent=kind == 'persistent')
    if kind in ('full_supported', 'persistent'):
      supported(next_lead, 10.0 if kind == 'full_supported' else 0.0)
    state = radar_state() if kind == 'nonfusion' else fused_state() if kind == 'empty' else fused_state(next_lead)
    planner, _ = repeat_native(state, planner=planner)
  np.testing.assert_array_equal(planner.params[:, NORMALIZERS], 0.0)


def test_symbolic_comfort_and_risk_residuals_use_independent_default_and_fixed_scales(monkeypatch):
  from acados.acados_template import AcadosModel, AcadosOcp
  monkeypatch.setattr(long_mpc, 'AcadosModel', AcadosModel, raising=False)
  monkeypatch.setattr(long_mpc, 'AcadosOcp', AcadosOcp, raising=False)
  ocp = long_mpc.gen_long_ocp()
  assert len(ocp.parameter_values) == PARAM_DIM == 14
  np.testing.assert_array_equal(ocp.parameter_values[list(NORMALIZERS)], 0.0)
  residuals = vertcat(ocp.model.cost_y_expr[0], ocp.model.con_h_expr[3])
  evaluate = Function('unknown_tail_residual', [ocp.model.x, ocp.model.p],
                      [residuals, jacobian(residuals, ocp.model.x)[:, 1]])
  parameters = np.asarray(ocp.parameter_values).copy()
  parameters[2] = 0.0
  initial = np.array([100.0, 10.0, 0.0])
  old_residuals, old_derivatives = map(lambda value: np.asarray(value).ravel(), evaluate(initial, parameters))
  assert np.all(old_residuals < 0.0) and np.all(old_derivatives > 0.0)
  parameters[COMFORT_NORMALIZER] = 20.0
  _, separate = evaluate(initial, parameters)
  assert float(separate[0]) < 0.0 < float(separate[1])
  parameters[RISK_NORMALIZER] = 20.0
  magnitudes = []
  for speed in (0.0, 1.0, 5.0, 10.0, 20.0, 30.0):
    residual, derivative = evaluate([100.0, speed, 0.0], parameters)
    residual, derivative = np.asarray(residual).ravel(), np.asarray(derivative).ravel()
    assert np.all(residual < 0.0) and np.all(derivative < 0.0)
    magnitudes.append(residual**2)
  assert np.all(np.diff(magnitudes, axis=0) > 0.0)


def scheduled_rest_witness():
  initial = np.array([0.0, 1.0, 0.0])
  node = 6
  states, jerks = LongitudinalMpc.future_rest_primal(initial, node)
  parameters = np.zeros((N+1, PARAM_DIM))
  parameters[:, 0], parameters[:, 1] = ACCEL_MIN, ACCEL_MAX
  parameters[:, 4], parameters[:, 5] = 1.45, LEAD_DANGER_FACTOR
  parameters[:, long_mpc.RISK_STOP_DISTANCE] = 6.0
  # An independently specified metre margin, not a normalizer-dependent one.
  safe = states[:, 1]**2/5.0 + 1.45*states[:, 1] + 6.0
  parameters[:, 2] = states[:, 0] + LEAD_DANGER_FACTOR*safe + 1e-5
  return initial, states, jerks, node, parameters


@pytest.mark.parametrize('fixed,expected', [(0.0, 'scheduled_rest_lead_danger_envelope_violated'), (11.0, '')])
def test_scheduled_rest_checks_the_same_default_or_fixed_native_risk_residual(fixed, expected):
  initial, states, jerks, node, parameters = scheduled_rest_witness()
  parameters[:, RISK_NORMALIZER] = fixed
  parameters[-1, 2] -= 1e-5 + 10.5e-9
  assert LongitudinalMpc.scheduled_rest_trajectory_error(
    initial, states, jerks, node, parameters, tolerance=1e-9,
  ) == expected


@pytest.mark.parametrize('column', NORMALIZERS)
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1.0])
def test_scheduled_rest_rejects_invalid_normalizer_parameters(column, value):
  initial, states, jerks, node, parameters = scheduled_rest_witness()
  parameters[2, column] = value
  assert LongitudinalMpc.scheduled_rest_trajectory_error(
    initial, states, jerks, node, parameters, tolerance=1e-9,
  ) == 'invalid_scheduled_rest_trajectory'
