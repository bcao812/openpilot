from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COST_E_DIM, CRASH_DISTANCE, LEAD_DANGER_FACTOR, MPC_INVALID_INPUT, MPC_INVALID_SOLUTION, MPC_SOLVER_ERROR, N,
  T_DIFFS, T_IDXS, LongitudinalMpc, LongitudinalPlanSource, MpcReference, get_safe_obstacle_distance,
  get_stopped_equivalence_factor, get_T_FOLLOW,
)


def radar_state(*, distance=None, speed=0.0, acceleration=0.0):
  def lead():
    return SimpleNamespace(present=False, dRel=50.0, vLead=0.0, aLeadK=0.0, aLeadTau=1.5, modelProb=0.99)

  state = SimpleNamespace(leadOne=lead(), leadTwo=lead())
  if distance is not None:
    state.leadOne.present = True
    state.leadOne.dRel = distance
    state.leadOne.vLead = speed
    state.leadOne.aLeadK = acceleration
  return state


def solve_reference(speed, reference, *, radar=None, acceleration=0.0, iterations=12):
  planner = LongitudinalMpc()
  planner.set_weights(prev_accel_constraint=False)
  planner.set_cur_state(speed, acceleration)
  for _ in range(iterations):
    planner.update(radar if radar is not None else radar_state(), reference=reference)
    assert planner.solution_status == 0
    assert planner.last_solver_status == 0
  return planner


def action(planner):
  return get_accel_from_plan(planner.v_solution, planner.a_solution, T_IDXS, action_t=0.55)


def raw_lead_clearance_at_nodes(planner, lead):
  lead_accel = lead.aLeadK * np.exp(-lead.aLeadTau * T_IDXS**2 / 2.0)
  lead_velocity = np.maximum(0.0, lead.vLead + np.cumsum(T_DIFFS * lead_accel))
  lead_position = lead.dRel + np.cumsum(T_DIFFS * lead_velocity)
  return lead_position - planner.x_sol[:, 0]


def lead_projection_review(planner, lead):
  """Preserve full-horizon geometry without inventing a full-horizon hard gate."""
  clearance = raw_lead_clearance_at_nodes(planner, lead)
  return {'status': 'REVIEW' if np.min(clearance) < CRASH_DISTANCE else 'NO_FLAG',
          'min_node_clearance_m': float(np.min(clearance)),
          'clearance_nodes_m': clearance.tolist(), 'time_nodes_s': T_IDXS.tolist(),
          'screening_marker_m': CRASH_DISTANCE,
          'screening_marker_source': 'Production FCW diagnostic: leadOne, modelProb > 0.9, T < 5 s; not a full-horizon safety boundary',
          'scope': 'All raw prediction nodes, including initial and terminal; no distance clipping',
          'physical_safety_gate': 'INCONCLUSIVE',
          'limitation': 'Unconstrained obstacle terminal and soft stages; no between-node, actuator, or closed-loop qualification'}


def assert_admitted_lead_plan(planner, radar, lead_index):
  """Defined admission, finite-solve, numerical-bound and response contracts.

  Full-horizon raw projection is recorded separately as REVIEW. The solver's
  soft obstacle preference and unconstrained obstacle terminal do not define
  an absolute safety boundary for every predicted node.
  """
  lead = (radar.leadOne, radar.leadTwo)[lead_index]
  assert lead.present
  assert planner.solution_status == planner.last_solver_status == 0
  assert planner.source == (LongitudinalPlanSource.lead0 if lead_index == 0 else LongitudinalPlanSource.lead1)
  for values in (planner.x0, planner.x_sol, planner.u_sol, planner.v_solution, planner.a_solution, planner.params):
    assert np.all(np.isfinite(values))
  np.testing.assert_array_equal(planner.params[:, 5], np.full(N+1, LEAD_DANGER_FACTOR))
  np.testing.assert_array_equal(planner.params[:, 0], np.full(N+1, ACCEL_MIN))
  assert np.min(planner.a_solution[:-1]) >= ACCEL_MIN - 1e-3
  assert np.max(planner.a_solution[:-1]) <= ACCEL_MAX + 1e-3
  # The older generated solver has no terminal state constraints. Apply the
  # existing stage tolerance only where that constraint is actually defined.
  assert np.min(planner.v_solution[:-1]) >= -1e-3
  assert np.min(planner.a_solution[:-1]) < 0.0
  if lead.vLead < planner.x0[1]:
    assert action(planner) < 0.0
  assert np.min(planner.v_solution[1:]) < planner.x0[1]


def simulate_reference(initial_speed, reference_at, duration=35.0):
  planner = LongitudinalMpc()
  speed, acceleration = initial_speed, 0.0
  history = []
  for frame in range(round(duration / DT_MDL)):
    reference = reference_at(frame * DT_MDL)
    planner.set_cur_state(speed, acceleration)
    planner.set_weights(prev_accel_constraint=speed > 0.001)
    planner.update(radar_state(), reference=reference)
    assert planner.solution_status == 0
    command = float(action(planner))
    assert np.isfinite(command)
    assert ACCEL_MIN - 1e-3 <= command <= ACCEL_MAX + 1e-3
    acceleration = command
    speed = max(0.0, speed + acceleration * DT_MDL)
    history.append((frame * DT_MDL, speed, command))
  return np.asarray(history)


def assert_failure(planner, status):
  assert planner.solution_status == status
  assert planner.last_solver_status == status
  assert planner.crash_cnt == 0
  for values in (planner.x0, planner.x_sol, planner.u_sol, planner.v_solution, planner.a_solution,
                 planner.j_solution, planner.a_prev, planner.params, planner.yref):
    assert np.all(np.isfinite(values))
  assert np.max(planner.a_solution) <= 0.0
  assert action(planner) <= 0.0


def test_reference_owns_an_immutable_snapshot():
  speeds = np.full(N+1, 20.0)
  reference = MpcReference(speeds)
  speeds[:] = 0.0
  np.testing.assert_array_equal(reference.cruise_speed, np.full(N+1, 20.0))
  with pytest.raises(FrozenInstanceError):
    reference.accel_max = 10.0
  with pytest.raises(ValueError):
    reference.cruise_speed[0] = 10.0


@pytest.mark.parametrize('speed', [1.0, 10.0, 20.0, 35.0])
def test_no_lead_tracks_set_speed_without_a_fake_lead(speed):
  planner = solve_reference(speed, MpcReference(speed))
  assert planner.source == LongitudinalPlanSource.cruise
  assert abs(action(planner)) <= 1e-4
  np.testing.assert_allclose(planner.v_solution, speed, atol=1e-4)
  np.testing.assert_allclose(planner.params[:,2], speed * T_IDXS + get_safe_obstacle_distance(speed, get_T_FOLLOW()))
  assert planner.crash_cnt == 0


def test_native_stationary_soft_solution_is_not_an_execution_hold_request():
  planner = solve_reference(0.0, MpcReference(0.0))
  assert planner.v_solution[0] == 0.0
  assert np.all(np.isfinite(planner.x_sol))
  assert action(planner) > 0.0
  assert np.max(planner.v_solution) > 0.0
  assert planner.source == LongitudinalPlanSource.cruise
  assert planner.crash_cnt == 0


@pytest.mark.parametrize('initial_speed,target_speed', [(0.0, 10.0), (10.0, 20.0), (25.0, 20.0), (10.0, 0.0)])
def test_ideal_closed_loop_speed_and_stop_converge(initial_speed, target_speed):
  history = simulate_reference(initial_speed, lambda timestamp: MpcReference(target_speed))
  assert abs(history[-1, 1] - target_speed) <= 0.2
  assert np.ptp(history[-100:, 1]) <= 0.2
  if target_speed == 0.0:
    assert history[-1, 1] <= 0.05
  elif initial_speed > target_speed:
    assert np.min(history[:20, 2]) < -0.05


def test_upcoming_curve_limit_is_used_before_its_horizon_point():
  cruise = solve_reference(20.0, MpcReference(20.0))
  constrained = solve_reference(20.0, MpcReference(np.where(T_IDXS < 3.0, 20.0, 10.0)))
  assert action(constrained) < action(cruise) - 0.05
  assert constrained.v_solution[-1] < cruise.v_solution[-1] - 5.0
  assert constrained.source == LongitudinalPlanSource.cruise


def test_curve_limit_entry_and_exit_share_one_continuous_trajectory():
  def reference_at(timestamp):
    return MpcReference(12.0 if 5.0 <= timestamp < 20.0 else 20.0)

  history = simulate_reference(20.0, reference_at, duration=45.0)
  assert abs(history[round(19.0 / DT_MDL), 1] - 12.0) <= 0.3
  assert abs(history[-1, 1] - 20.0) <= 0.2
  assert np.max(np.abs(np.diff(history[:, 2]))) <= 0.3


def test_cruise_comfort_reference_does_not_reduce_real_lead_braking_authority():
  reference = MpcReference(0.0)
  radar = radar_state(distance=20.0, speed=0.0)
  planner = solve_reference(10.0, reference, radar=radar, acceleration=-1.0)
  np.testing.assert_array_equal(planner.params[:,0], np.full(N+1, ACCEL_MIN))
  assert reference.cruise_min_accel == -1.2
  assert np.min(planner.a_solution[:-1]) < reference.cruise_min_accel - 0.1
  assert planner.source == LongitudinalPlanSource.lead0


def test_cruise_deceleration_only_changes_the_virtual_obstacle():
  reference = MpcReference(0.0)
  planner = solve_reference(20.0, reference)
  cruise_velocity = np.maximum(0.0, 20.0 + reference.cruise_min_accel * T_IDXS)
  expected_obstacle = np.cumsum(T_DIFFS * cruise_velocity) + get_safe_obstacle_distance(cruise_velocity, get_T_FOLLOW())
  np.testing.assert_allclose(planner.params[:,2], expected_obstacle)
  np.testing.assert_array_equal(planner.params[:,0], np.full(N+1, ACCEL_MIN))


def test_model_ego_path_uses_full_safe_distance_equivalence():
  speed = 10.0
  position = 100.0 + speed * T_IDXS
  reference = MpcReference(30.0, model_position=position,
                           model_velocity=np.full(N+1, speed), model_acceleration=np.zeros(N+1))
  planner = solve_reference(speed, reference)
  assert planner.source == LongitudinalPlanSource.e2e
  assert abs(action(planner)) <= 1e-4
  np.testing.assert_allclose(planner.params[:,2], speed * T_IDXS + get_safe_obstacle_distance(speed, get_T_FOLLOW()))
  assert planner.crash_cnt == 0


def test_admitted_stopping_model_changes_the_native_solution_without_fcw():
  model_velocity = np.maximum(10.0 - 1.25 * T_IDXS, 0.0)
  model_position = np.where(T_IDXS < 8.0, 10.0 * T_IDXS - 0.625 * T_IDXS**2, 40.0)
  model_acceleration = np.where(model_velocity > 0.0, -1.25, 0.0)
  reference = MpcReference(20.0, model_position=model_position,
                           model_velocity=model_velocity, model_acceleration=model_acceleration)
  planner = solve_reference(10.0, reference)
  unconstrained = solve_reference(10.0, MpcReference(20.0))
  assert action(planner) < -0.5
  assert action(planner) < action(unconstrained) - 0.5
  assert planner.v_solution[-1] < 1.0
  assert planner.source == LongitudinalPlanSource.e2e
  assert planner.crash_cnt == 0


@pytest.mark.parametrize('lead_index', [0, 1])
@pytest.mark.parametrize('distance,lead_speed,lead_accel,ego_speed,ego_accel',
                         [(20.0, 0.0, 0.0, 10.0, -1.0), (44.0, 10.0, -1.5, 20.0, -1.0),
                          (30.0, 20.0, -3.0, 20.0, -1.0)])
def test_faster_reference_preserves_admitted_lead_response_with_projection_review(lead_index, distance, lead_speed, lead_accel,
                                                                               ego_speed, ego_accel, record_property):
  radar = radar_state(distance=distance, speed=lead_speed, acceleration=lead_accel)
  if lead_index == 1:
    radar.leadOne, radar.leadTwo = radar.leadTwo, radar.leadOne
  unified = LongitudinalMpc()
  unified.set_cur_state(ego_speed, ego_accel)
  unified.set_weights(a_change_scale=0.5)
  failures, observations = [], []
  for iteration in range(12):
    unified.update(radar, reference=MpcReference(40.0))
    projection = lead_projection_review(unified, (radar.leadOne, radar.leadTwo)[lead_index])
    observations.append({'iteration': iteration, 'request': float(action(unified)), **projection})
    try:
      assert_admitted_lead_plan(unified, radar, lead_index)
    except AssertionError as error:
      failures.append({'iteration': iteration, 'failure': str(error)})
  record_property('fixed_motion_solver_iterations', json.dumps(observations))
  record_property('projection_status', 'REVIEW' if any(row['status'] == 'REVIEW' for row in observations) else 'NO_FLAG')
  record_property('physical_safety_gate', 'INCONCLUSIVE')
  assert not failures, failures


@pytest.mark.parametrize('accel_min,accel_max,target_speed,terminal_violation',
                         [(-3.5, 0.7, 30.0, True), (-1.2, 0.0, 0.0, True), (-2.0, 1.0, 10.0, False)])
def test_runtime_acceleration_bounds_apply_to_soft_stages_not_terminal(accel_min, accel_max, target_speed, terminal_violation):
  planner = solve_reference(20.0, MpcReference(target_speed, accel_min, accel_max))
  np.testing.assert_array_equal(planner.params[:,0], np.full(N+1, accel_min))
  np.testing.assert_array_equal(planner.params[:,1], np.full(N+1, accel_max))
  assert np.min(planner.a_solution[:-1]) >= accel_min - 1e-3
  assert np.max(planner.a_solution[:-1]) <= accel_max + 1e-3
  if terminal_violation:
    assert not accel_min <= planner.a_solution[-1] <= accel_max


def test_generated_solver_has_soft_stage_constraints_and_no_terminal_constraints():
  model_path = Path(__file__).resolve().parents[1] / 'lib/longitudinal_mpc_lib/acados_ocp_long.json'
  problem = json.loads(model_path.read_text())
  assert problem['dims']['nh'] == problem['dims']['ns'] == 4
  assert problem['dims']['nh_e'] == problem['dims']['ns_e'] == 0
  assert problem['constraints']['idxsh'] == list(range(4))


def test_equivalent_delay_sample_is_not_a_hard_actuator_limit():
  planner = LongitudinalMpc()
  reference = MpcReference(0.0, cruise_min_accel=ACCEL_MIN)
  speed, acceleration = 10.0, 0.0
  exceeded = False
  for _ in range(400):
    planner.set_cur_state(speed, acceleration)
    planner.set_weights(prev_accel_constraint=speed > 0.001)
    planner.update(radar_state(), reference=reference)
    assert planner.solution_status == 0
    command = float(action(planner))
    assert np.isfinite(command)
    if not ACCEL_MIN <= command <= ACCEL_MAX:
      assert ACCEL_MIN <= planner.x0[2] <= ACCEL_MAX
      exceeded = True
      break
    acceleration = command
    speed = max(0.0, speed + acceleration * DT_MDL)
  assert exceeded


def test_horizon_acceleration_bounds_and_authorized_global_limits():
  accel_max = np.where(T_IDXS < 3.0, 0.7, 0.3)
  planner = solve_reference(10.0, MpcReference(30.0, ACCEL_MIN - 10.0, accel_max))
  np.testing.assert_array_equal(planner.params[:,0], np.full(N+1, ACCEL_MIN))
  np.testing.assert_array_equal(planner.params[:,1], accel_max)
  assert np.all(planner.a_solution[:-1] <= accel_max[:-1] + 1e-3)


def test_reference_none_preserves_legacy_params_and_solution_each_frame():
  explicit = LongitudinalMpc()
  implicit = LongitudinalMpc()
  for frame in range(30):
    radar = radar_state(distance=50.0 - frame * 0.1, speed=15.0, acceleration=-0.5)
    for planner in (explicit, implicit):
      planner.set_cur_state(20.0 - frame * 0.05, -0.5)
      planner.set_weights(a_change_scale=0.5)
    explicit.update(radar, reference=None)
    implicit.update(radar)
    assert explicit.solution_status == implicit.solution_status == 0
    assert explicit.source == implicit.source
    for name in ('params', 'yref', 'x_sol', 'u_sol', 'a_prev'):
      np.testing.assert_array_equal(getattr(explicit, name), getattr(implicit, name))
    np.testing.assert_array_equal(explicit.yref, np.zeros((N+1, COST_E_DIM+1)))


@pytest.mark.parametrize('mode', ['legacy', 'ordinary', 'cruise'])
def test_lead_obstacle_columns_follow_current_presence_and_speed(mode):
  planner = LongitudinalMpc()
  reference = {'legacy': None, 'ordinary': MpcReference(None, comfort_stop_distance=4.0),
               'cruise': MpcReference(20.0, comfort_stop_distance=4.0)}[mode]
  radar = radar_state()
  radar.leadOne.dRel, radar.leadOne.vLead = 30.0, 9.0
  radar.leadTwo.dRel, radar.leadTwo.vLead = 25.0, 7.0
  lead_sources = [LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1]
  for speed, presence in ((12.0, (False, False)), (13.0, (True, False)), (14.0, (False, True)),
                          (15.0, (True, True)), (16.0, (False, False))):
    radar.leadOne.present, radar.leadTwo.present = presence
    planner.set_cur_state(speed, 0.0)
    planner.set_weights(prev_accel_constraint=False)
    with patch.object(planner, 'observe_obstacles', wraps=planner.observe_obstacles) as observed:
      planner.update(radar, reference=reference)
    assert planner.solution_status == 0
    obstacles, sources = observed.call_args.args
    assert obstacles.shape == (N + 1, 3 if mode == 'cruise' else 2)
    for column, lead in enumerate((radar.leadOne, radar.leadTwo)):
      if not lead.present and (mode == 'cruise' or mode == 'ordinary' and any(presence)):
        assert np.all(np.isinf(obstacles[:, column]))
      else:
        # Constant-speed lead geometry gives an independent expectation;
        # absent slots use this frame's ego+10 m/s, including after lead loss.
        distance, lead_speed = (lead.dRel, lead.vLead) if lead.present else (50.0, speed + 10.0)
        expected = distance + lead_speed * T_IDXS + get_stopped_equivalence_factor(lead_speed)
        np.testing.assert_allclose(obstacles[:, column], expected, rtol=0.0, atol=1e-12)
    expected_sources = [LongitudinalPlanSource.cruise] * 2 if mode == 'ordinary' and not any(presence) else lead_sources
    assert sources[:2] == expected_sources
    expected_source = (LongitudinalPlanSource.lead1 if presence[1] else LongitudinalPlanSource.lead0 if presence[0]
                       else LongitudinalPlanSource.lead0 if mode == 'legacy' else LongitudinalPlanSource.cruise)
    assert planner.source == expected_source
    np.testing.assert_array_equal(planner.params[:, 2], np.min(obstacles, axis=1))
    if not any(presence):
      np.testing.assert_array_equal(obstacles[:, 0], obstacles[:, 1])
      # The native matrix retains two independent columns, even when its
      # temporary input arrays share the identical no-lead continuation.
      assert not np.shares_memory(obstacles[:, 0], obstacles[:, 1])


@pytest.mark.parametrize('reference', [
  MpcReference(float('nan')),
  MpcReference(float('inf')),
  MpcReference(-1.0),
  MpcReference(np.zeros(N)),
  MpcReference(20.0, float('nan'), ACCEL_MAX),
  MpcReference(20.0, ACCEL_MIN, float('inf')),
  MpcReference(20.0, 1.0, -1.0),
  MpcReference(20.0, ACCEL_MAX + 1.0, ACCEL_MAX + 2.0),
  MpcReference(20.0, ACCEL_MIN - 2.0, ACCEL_MIN - 1.0),
  MpcReference(20.0, cruise_min_accel=float('nan')),
  MpcReference(20.0, cruise_min_accel=float('inf')),
  MpcReference(20.0, cruise_min_accel=0.1),
  MpcReference(20.0, cruise_min_accel=np.zeros(N)),
  MpcReference(20.0, model_velocity=np.zeros(N+1)),
  MpcReference(20.0, model_position=np.zeros(N), model_velocity=np.zeros(N+1), model_acceleration=np.zeros(N+1)),
  MpcReference(20.0, model_position=-T_IDXS, model_velocity=np.zeros(N+1), model_acceleration=np.zeros(N+1)),
  MpcReference(20.0, model_position=T_IDXS, model_velocity=-np.ones(N+1), model_acceleration=np.zeros(N+1)),
])
def test_invalid_references_fail_before_solving(reference):
  planner = LongitudinalMpc()
  planner.set_cur_state(20.0, 0.5)
  planner.solver = Mock(wraps=planner.solver)
  planner.update(radar_state(), reference=reference)
  assert_failure(planner, MPC_INVALID_INPUT)
  planner.solver.solve.assert_not_called()


@pytest.mark.parametrize('field', ['model_position', 'model_velocity', 'model_acceleration'])
@pytest.mark.parametrize('invalid_value', [float('nan'), float('inf')])
def test_nonfinite_model_reference_is_not_silently_replaced(field, invalid_value):
  reference = MpcReference(20.0, model_position=10.0*T_IDXS,
                           model_velocity=np.full(N+1, 10.0), model_acceleration=np.zeros(N+1))
  values = np.array(getattr(reference, field), copy=True)
  values[5] = invalid_value
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, -0.5)
  planner.update(radar_state(), reference=replace(reference, **{field: values}))
  assert_failure(planner, MPC_INVALID_INPUT)


@pytest.mark.parametrize('field', ['dRel', 'vLead', 'aLeadK', 'aLeadTau'])
@pytest.mark.parametrize('invalid_value', [float('nan'), float('inf')])
def test_nonfinite_real_lead_cannot_be_hidden_by_cruise(field, invalid_value):
  radar = radar_state(distance=30.0, speed=10.0)
  setattr(radar.leadOne, field, invalid_value)
  planner = LongitudinalMpc()
  planner.set_cur_state(20.0, 0.5)
  planner.update(radar, reference=MpcReference(30.0))
  assert_failure(planner, MPC_INVALID_INPUT)


@pytest.mark.parametrize('speed,acceleration', [(float('nan'), 0.0), (20.0, float('nan')), (float('inf'), float('inf'))])
def test_nonfinite_current_state_has_a_finite_nonpositive_fallback(speed, acceleration):
  planner = LongitudinalMpc()
  planner.set_cur_state(speed, acceleration)
  planner.update(radar_state(), reference=MpcReference(20.0))
  assert_failure(planner, MPC_INVALID_INPUT)


@pytest.mark.parametrize('status', [1, 3, 4])
def test_solver_failure_survives_reset_and_next_success_recovers(status):
  planner = LongitudinalMpc()
  planner.set_cur_state(20.0, -0.5)
  native_solver = planner.solver
  planner.solver = Mock(wraps=native_solver)
  planner.solver.solve.return_value = status
  planner.update(radar_state(), reference=MpcReference(20.0))
  assert_failure(planner, status)
  planner.solver.reset.assert_called_once()
  planner.solver = native_solver
  planner.set_cur_state(20.0, 0.0)
  planner.update(radar_state(), reference=MpcReference(20.0))
  assert planner.solution_status == planner.last_solver_status == 0


@pytest.mark.parametrize('field,invalid_value', [('x', float('nan')), ('x', float('inf')), ('u', float('nan')), ('u', float('inf'))])
def test_zero_solver_status_with_nonfinite_solution_is_a_failure(field, invalid_value):
  planner = LongitudinalMpc()
  planner.set_cur_state(20.0, 0.5)
  native_solver = planner.solver

  def invalid_get(stage, name):
    values = native_solver.get(stage, name)
    if name == field and stage == 1:
      values[0] = invalid_value
    return values

  planner.solver = Mock(wraps=native_solver)
  planner.solver.get.side_effect = invalid_get
  planner.update(radar_state(), reference=MpcReference(20.0))
  assert_failure(planner, MPC_INVALID_SOLUTION)


def test_solver_runtime_error_has_an_explicit_failure_status():
  planner = LongitudinalMpc()
  planner.set_cur_state(20.0, 0.5)
  planner.solver = Mock(wraps=planner.solver)
  planner.solver.solve.side_effect = RuntimeError("injected solver failure")
  planner.update(radar_state(), reference=MpcReference(20.0))
  assert_failure(planner, MPC_SOLVER_ERROR)


def test_solver_failure_status_is_not_overwritten_by_a_timing_error():
  planner = LongitudinalMpc()
  planner.set_cur_state(20.0, 0.5)
  planner.solver = Mock(wraps=planner.solver)
  planner.solver.solve.return_value = 4
  planner.solver.get_stats.side_effect = RuntimeError("injected timing failure")
  planner.update(radar_state(), reference=MpcReference(20.0))
  assert_failure(planner, 4)
  planner.solver.get_stats.assert_not_called()


@pytest.mark.parametrize('field', ['params', 'yref'])
def test_nonfinite_solver_parameters_are_rejected_before_solve(field):
  planner = solve_reference(20.0, MpcReference(20.0))
  getattr(planner, field)[5, 0] = np.nan
  planner.solver = Mock(wraps=planner.solver)
  planner.run()
  assert_failure(planner, MPC_INVALID_INPUT)
  planner.solver.solve.assert_not_called()
