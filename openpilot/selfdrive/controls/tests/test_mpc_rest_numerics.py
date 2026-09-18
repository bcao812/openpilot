"""Numerical rest repair must preserve dynamics, physical bounds and fault braking."""

from unittest.mock import Mock

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  MPC_GOAL_INFEASIBLE, MPC_GOAL_UNSATISFIED, QP_FEASIBILITY_TOL, REST_STATE_TOL, T_IDXS, LongitudinalMpc,
)
from openpilot.selfdrive.controls.tests.test_due_rest_reference import NEAR_ZERO_STATES, typed_due_reference
from openpilot.selfdrive.controls.tests.test_typed_model_reference import MODEL_TIMES, stopping_curve
from openpilot.selfdrive.controls.tests.test_typed_reference_mpc import StrictPlanTransportHarness, typed_reference
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state
from openpilot.selfdrive.controls.tests.test_unified_model_reference_contract import model_stop_curve
from openpilot.selfdrive.controls.tests.test_unified_startup import PLANNER_STEPS


@pytest.fixture(autouse=True)
def isolated_params():
  with OpenpilotPrefix():
    yield


def integrate(initial, jerks):
  """Independent polynomial integration for the published trajectory contract."""
  states = [np.asarray(initial).copy()]
  for jerk, duration in zip(jerks, np.diff(T_IDXS), strict=True):
    x, v, a = states[-1]
    states.append(np.array([x + v * duration + a * duration**2 / 2.0 + jerk * duration**3 / 6.0,
                            v + a * duration + jerk * duration**2 / 2.0, a + jerk * duration]))
  return np.array(states)


def test_native_corner_is_repaired_with_one_solve_and_a_dynamically_consistent_trajectory():
  mpc = LongitudinalMpc()
  mpc.set_weights(prev_accel_constraint=False)
  initial = np.array([0.0, REST_STATE_TOL, REST_STATE_TOL])
  mpc.set_cur_state(*initial[1:])
  mpc.update(radar_state(), reference=typed_due_reference(REST_STATE_TOL))
  assert mpc.diagnostic_solve_calls == 1
  assert mpc.diagnostic_last_solver_return == mpc.last_solver_status == mpc.solution_status == 0
  assert mpc.rest_goal_state == 'satisfied' and mpc.rest_goal_braking_accel is None
  np.testing.assert_array_equal(mpc.x0, initial)
  np.testing.assert_array_equal(mpc.x_sol[0], initial)
  np.testing.assert_allclose(mpc.x_sol, integrate(initial, mpc.j_solution), atol=1e-17, rtol=1e-14)
  assert np.max(np.abs(mpc.x_sol[:, 1:])) <= REST_STATE_TOL
  assert np.all(mpc.v_solution >= 0.0)
  # The only feasible first-interval endpoint is a_1=-tol, v_1=tol.
  assert mpc.a_solution[1] == -REST_STATE_TOL
  assert mpc.v_solution[1] == REST_STATE_TOL
  raw = np.array([mpc.solver.get(stage, 'x') for stage in range(len(T_IDXS))])
  projection = mpc.diagnostic_rest_goal['numericalProjection']
  assert projection['feasibilityTolerance'] == QP_FEASIBILITY_TOL < REST_STATE_TOL / 1000.0
  np.testing.assert_allclose(projection['maximumStateCorrection'], np.max(np.abs(mpc.x_sol - raw), axis=0))
  assert np.all(np.asarray(projection['maximumStateCorrection']) <= projection['maximumStateCorrectionBudget'])
  np.testing.assert_allclose(mpc.a_prev, np.interp(T_IDXS + mpc.dt, T_IDXS, mpc.a_solution), atol=0.0, rtol=0.0)


@pytest.mark.parametrize('fault', ['dynamics', 'rest_box', 'continuous_velocity'])
def test_a_native_zero_return_cannot_hide_a_defect_outside_the_numerical_budget(fault):
  mpc = LongitudinalMpc()
  mpc.set_weights(prev_accel_constraint=False)
  initial = np.array([0.0, REST_STATE_TOL, REST_STATE_TOL])
  mpc.set_cur_state(*initial[1:])
  native = mpc.solver
  solver = Mock(wraps=native)
  returned = {}

  def solve():
    status = native.solve()
    assert status == 0
    states = np.array([native.get(stage, 'x') for stage in range(len(T_IDXS))])
    jerks = np.array([native.get(stage, 'u')[0] for stage in range(len(T_IDXS) - 1)])
    if fault == 'dynamics':
      states[1, 0] += 10.0 * QP_FEASIBILITY_TOL
    else:
      if fault == 'rest_box':
        # Coherent dynamics with a real box error cannot be clipped to pass.
        jerks[0] += 100.0 * QP_FEASIBILITY_TOL / T_IDXS[1]**2
      else:
        # Preserve a finite, continuous polynomial; this is too large to be
        # explained by native numerical precision or a one-ULP boundary.
        jerks[0] -= 0.1
      states = integrate(initial, jerks)
    returned.update(states=states, jerks=jerks)
    return status

  def get(stage, field):
    if field == 'x':
      return returned['states'][stage].copy()
    if field == 'u':
      return np.array([returned['jerks'][stage]])
    return native.get(stage, field)

  solver.solve.side_effect, solver.get.side_effect = solve, get
  mpc.solver = solver
  mpc.update(radar_state(), reference=typed_due_reference(REST_STATE_TOL))
  assert mpc.diagnostic_solve_calls == 1 and mpc.diagnostic_last_solver_return == mpc.last_solver_status == 0
  assert mpc.solution_status == MPC_GOAL_UNSATISFIED and mpc.rest_goal_state == 'unsatisfied'
  assert mpc.rest_goal_braking_accel == ACCEL_MIN and not mpc.previous_solution_valid
  assert mpc.rest_goal_reason in ('native_rest_dynamics_residual_exceeded', 'native_rest_precision_budget_exceeded')
  np.testing.assert_array_equal(mpc.x_sol, returned['states'])
  np.testing.assert_array_equal(mpc.j_solution, returned['jerks'])


@pytest.mark.parametrize('speed,acceleration', [(0.005, 0.0), (0.02, -0.2), (0.001, -0.01), (0.0, -0.001)])
def test_first_node_infeasibility_is_explicit_before_the_one_real_risk_solve(speed, acceleration):
  mpc = LongitudinalMpc()
  mpc.set_weights(prev_accel_constraint=False)
  mpc.set_cur_state(speed, acceleration)
  reference = typed_reference(stopping_curve(max(speed, 0.005), 0.001), speed)
  assert reference.typed_model.rest_goal.native_node == 1
  assert speed + ACCEL_MIN * T_IDXS[1] <= REST_STATE_TOL
  mpc.update(radar_state(distance=12.0, speed=0.0), reference=reference)
  assert mpc.diagnostic_solve_calls == 1 and mpc.diagnostic_last_solver_return == mpc.last_solver_status == 0
  assert mpc.solution_status == MPC_GOAL_INFEASIBLE and mpc.rest_goal_state == 'infeasible'
  assert mpc.rest_goal_reason == 'rest_goal_time_below_discrete_lower_bound'
  assert mpc.rest_goal_braking_accel == ACCEL_MIN and not mpc.previous_solution_valid
  assert np.all(np.isfinite(mpc.x_sol))
  # The normal lead-risk problem is still solved, but is not called a stop.
  lower, upper = mpc.diagnostic_reference_bounds
  assert np.all(lower[1:, 1] < -REST_STATE_TOL) and np.all(upper[1:] > REST_STATE_TOL)


def test_a_reachable_first_node_keeps_the_original_exact_zero_native_box():
  speed = 0.005
  acceleration = -2.0 * speed / T_IDXS[1]
  mpc = LongitudinalMpc()
  mpc.set_weights(prev_accel_constraint=False)
  mpc.set_cur_state(speed, acceleration)
  reference = typed_reference(stopping_curve(speed, 0.001), speed)
  mpc.update(radar_state(), reference=reference)
  assert mpc.diagnostic_solve_calls == 1 and mpc.diagnostic_last_solver_return == mpc.solution_status == 0
  assert mpc.rest_goal_state == 'satisfied'
  lower, upper = mpc.diagnostic_reference_bounds
  np.testing.assert_array_equal(lower[1:], 0.0)
  np.testing.assert_array_equal(upper[1:], 0.0)
  np.testing.assert_allclose(mpc.x_sol[1:, 1:], 0.0, atol=REST_STATE_TOL, rtol=0.0)


def test_continuous_nonnegative_speed_is_part_of_the_first_interval_reachability_check():
  # Endpoint constraints alone admit a_next in this range, but every such
  # parabola crosses below zero before the first node.
  initial = np.array([0.0, 1e-9, -REST_STATE_TOL])
  duration = T_IDXS[1]
  endpoint_lower = max(-REST_STATE_TOL, -initial[2] - 2.0 * initial[1] / duration)
  endpoint_upper = min(REST_STATE_TOL, -initial[2] + 2.0 * (REST_STATE_TOL - initial[1]) / duration)
  assert endpoint_lower < endpoint_upper
  assert LongitudinalMpc.rest_acceleration_interval(initial, duration, np.array([0.0, -REST_STATE_TOL]),
                                                   np.full(2, REST_STATE_TOL)) is None


def test_precision_budget_rejection_keeps_braking_through_owner_and_honda_can():
  harness = StrictPlanTransportHarness(speed=10.0)
  harness.engage()
  while harness.frame % PLANNER_STEPS:
    harness.step()
  harness.model_curve = model_stop_curve(10.0, 0.001)
  for _ in range(PLANNER_STEPS):
    harness.step()
  harness.source_state.vEgo, harness.source_state.aEgo = NEAR_ZERO_STATES[0]
  native = harness.planner.mpc.solver
  solver = Mock(wraps=native)

  def get(stage, field):
    values = native.get(stage, field)
    if field == 'x' and stage == 1:
      values[0] += 10.0 * QP_FEASIBILITY_TOL
    return values

  solver.get.side_effect = get
  harness.planner.mpc.solver = solver
  zero = np.zeros(len(MODEL_TIMES))
  harness.model_curve = (zero.copy(), zero, zero.copy())
  harness.action_accel = 0.5
  for _ in range(PLANNER_STEPS):
    harness.step()
  rows = [harness.step() for _ in range(30)]
  can_rows = [row for row in harness.native_commands if row['mono'] > rows[0]['mono']]
  plan = harness.latest['longitudinalPlan'].longitudinalPlan
  assert harness.planner.mpc.diagnostic_last_solver_return == 0
  assert plan.solverStatus == MPC_GOAL_UNSATISFIED
  assert plan.motionIntent.brakingValid and not plan.motionIntent.releaseAllowed
  assert all(row['long_active'] and row['accel'] <= ACCEL_MIN + 1e-5 and not row['resume'] for row in rows)
  assert can_rows and all(row['BRAKE_REQUEST'] and row['GAS_COMMAND'] <= 0.0
                          and row['ACCEL_COMMAND'] <= ACCEL_MIN + 0.01 for row in can_rows)
