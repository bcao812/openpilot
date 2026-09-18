"""Numerical rest repair must preserve dynamics, physical bounds and fault braking."""

from unittest.mock import Mock

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  MPC_GOAL_INFEASIBLE, MPC_GOAL_UNSATISFIED, QP_FEASIBILITY_TOL, REST_STATE_TOL, T_IDXS, LongitudinalMpc, MpcReference,
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


@pytest.mark.parametrize('speed,acceleration', [
  (0.005, 0.0), (0.02, -0.2), (0.001, -0.01), (0.0, -0.001),
  # All seven native +4 captures from the four frozen report-20 risk jobs.
  # Each attempted exact v=a=0 at native node 1 from these measured states.
  (0.1264825463294983, -0.501275897026062), (0.07988708466291428, -1.2803713083267212),
  (0.15970134735107422, 0.4808322787284851), (0.10668159276247025, -0.5194023847579956),
  (0.006117075216025114, -0.6851852536201477), (0.006080630235373974, -0.008811135776340961),
  (0.020815182477235794, 0.08480892330408096),
])
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
  expected_reason = 'rest_goal_without_nonnegative_initial_segment' if speed == 0.0 and acceleration < 0.0 else \
                    'rest_goal_time_below_discrete_lower_bound'
  assert mpc.rest_goal_reason == expected_reason
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


def test_an_originally_accepted_due_rest_trajectory_is_not_forced_into_a_stricter_projection():
  """A tiny negative interval value inside the original precision is accepted.

  The first engineering rerun exposed nine false rejections when unconditional
  projection spent its numerical budget enforcing a stricter zero threshold.
  This coherent polynomial isolates that distinction from the plant model.
  """
  initial = np.array([0.0, 1e-6, -1e-4])
  jerks = np.zeros(len(T_IDXS) - 1)
  jerks[0] = 2e-4 / T_IDXS[1]
  jerks[1] = -1e-4 / (T_IDXS[2] - T_IDXS[1])
  states = integrate(initial, jerks)
  assert np.all(states[:, 1] >= 0.0) and np.max(np.abs(states[:, 1:])) < REST_STATE_TOL
  minimum = LongitudinalMpc.trajectory_minimum_velocity(states, jerks)
  assert -REST_STATE_TOL < minimum < -100.0 * QP_FEASIBILITY_TOL
  mpc = LongitudinalMpc()
  mpc.set_weights(prev_accel_constraint=False)
  mpc.set_cur_state(*initial[1:])
  native = mpc.solver
  solver = Mock(wraps=native)

  def get(stage, field):
    if field == 'x':
      return states[stage].copy()
    if field == 'u':
      return np.array([jerks[stage]])
    return native.get(stage, field)

  solver.get.side_effect = get
  mpc.solver = solver
  mpc.update(radar_state(), reference=typed_due_reference(initial[1]))
  assert mpc.diagnostic_last_solver_return == mpc.solution_status == 0
  assert mpc.rest_goal_state == 'satisfied' and mpc.rest_goal_braking_accel is None
  assert not mpc.diagnostic_rest_goal['numericalProjection']['required']
  assert not mpc.diagnostic_rest_goal['numericalProjection']['applied']
  np.testing.assert_array_equal(mpc.x_sol, states)
  np.testing.assert_array_equal(mpc.j_solution, jerks)


# First continuous-speed failures in the fixed engineering matrix at 8dd36bb8,
# model_stop_positive_action: d01_t02 t=5.90, d03_t04 t=4.80, uphill t=3.60.
FUTURE_REST_COUNTEREXAMPLES = [
  (0.020619023591279984, -0.9352537393569946, 0.020940198052347668, -0.003121203570461617, 11),
  (0.1138337254524231, -0.6702026724815369, 0.12843246493168956, -0.05869335832326215, 6),
  (1.3679653406143188, -1.8144984245300293, 1.3873872999131978, -0.36777157004207767, 8),
]


@pytest.mark.parametrize('speed,acceleration,raw_speed,raw_acceleration,node', FUTURE_REST_COUNTEREXAMPLES)
@pytest.mark.parametrize('previous_accel', [False, True])
def test_future_rest_native_problem_prevents_the_recorded_between_node_reverse_motion(
  speed, acceleration, raw_speed, raw_acceleration, node, previous_accel,
):
  raw = np.maximum(raw_speed + raw_acceleration * MODEL_TIMES, 0.0)
  mpc = LongitudinalMpc()
  mpc.set_weights(prev_accel_constraint=previous_accel)
  mpc.set_cur_state(speed, acceleration)
  if previous_accel:
    mpc.a_prev[:] = acceleration
  mpc.update(radar_state(), reference=typed_reference(raw, speed))
  assert mpc.rest_goal.native_node == node
  assert mpc.diagnostic_solve_calls == 1 and mpc.diagnostic_last_solver_return == mpc.solution_status == 0
  assert mpc.rest_goal_state == 'satisfied'
  assert mpc.diagnostic_rest_goal['minimumContinuousVelocity'] >= -REST_STATE_TOL
  assert mpc.diagnostic_rest_goal['maximumRestStateError'] <= REST_STATE_TOL
  assert 'numericalProjection' not in mpc.diagnostic_rest_goal
  assert mpc.diagnostic_rest_goal['continuousSpeedCertificate'] == 'quadratic_bernstein'
  assert len(mpc.diagnostic_rest_goal['continuousSpeedSplitTimes']) == node - 1
  np.testing.assert_array_equal(mpc.x0, [0.0, speed, acceleration])
  np.testing.assert_array_equal(mpc.x_sol, np.array([mpc.solver.get(stage, 'x') for stage in range(len(T_IDXS))]))


@pytest.mark.parametrize('replacement', ['untyped', 'moving', 'due_rest'])
def test_continuous_rest_linear_bounds_are_removed_when_the_future_goal_ends(replacement):
  mpc = LongitudinalMpc()
  mpc.set_weights(prev_accel_constraint=False)
  speed, acceleration, raw_speed, raw_acceleration, _ = FUTURE_REST_COUNTEREXAMPLES[0]
  mpc.set_cur_state(speed, acceleration)
  raw = np.maximum(raw_speed + raw_acceleration * MODEL_TIMES, 0.0)
  mpc.update(radar_state(), reference=typed_reference(raw, speed))
  assert mpc.solution_status == 0
  mpc.set_cur_state(0.0, 0.0)
  reference = {'untyped': MpcReference(20.0), 'moving': typed_reference(0.2 + 0.6 * MODEL_TIMES, 0.0),
               'due_rest': typed_due_reference(0.0)}[replacement]
  solver = Mock(wraps=mpc.solver)
  mpc.solver = solver
  mpc.update(radar_state(), reference=reference)
  assert mpc.solution_status == 0
  for call in solver.constraints_set.call_args_list:
    if call.args[1] in ('C', 'D'):
      np.testing.assert_array_equal(call.args[2], 0.0)
    if call.args[1] == 'lg':
      assert np.all(call.args[2] < 0.0)


@pytest.mark.parametrize('speed', [0.0, 1e-12])
def test_negative_acceleration_at_zero_or_too_small_speed_cannot_supply_a_continuous_future_rest_solution(speed):
  mpc = LongitudinalMpc()
  mpc.set_weights(prev_accel_constraint=False)
  mpc.set_cur_state(speed, -0.5)
  mpc.update(radar_state(), reference=typed_reference(stopping_curve(0.1, 2.5), speed))
  assert mpc.diagnostic_solve_calls == 1 and mpc.diagnostic_last_solver_return == 0
  assert mpc.solution_status == MPC_GOAL_INFEASIBLE
  assert mpc.rest_goal_reason == 'rest_goal_without_nonnegative_initial_segment'
  assert mpc.rest_goal_braking_accel == ACCEL_MIN
