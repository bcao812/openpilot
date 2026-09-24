"""Future stop scheduling preserves source meaning, feasibility and consumer authority.

These are software/ideal-trajectory contracts. A later optimizer node does not
move a road stop line or qualify a vehicle-response model.
"""

from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LEAD_DANGER_FACTOR, MPC_GOAL_INFEASIBLE, MPC_GOAL_UNSATISFIED, N, PARAM_DIM, QP_FEASIBILITY_TOL, RISK_STOP_DISTANCE,
  REST_STATE_TOL, T_IDXS, LongitudinalMpc,
)
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionPhase, MotionReferenceKind
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_due_rest_reference import typed_due_reference
from openpilot.selfdrive.controls.tests.test_motion_supervisor import MotionHarness
from openpilot.selfdrive.controls.tests.test_mpc_rest_numerics import assert_future_primal_witness, integrate
from openpilot.selfdrive.controls.tests.test_rest_braking_continuity import INSTANCE_NS, rejected_intent, tracking, zero_intent
from openpilot.selfdrive.controls.tests.test_typed_model_reference import MODEL_TIMES, SOURCE_NS, evidence, stopping_curve
from openpilot.selfdrive.controls.tests.test_typed_reference_mpc import typed_reference
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import ACCEL_LIMITS, ControlHarness
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state


def solve(reference, speed=10.0, acceleration=0.0):
  mpc = LongitudinalMpc()
  mpc.set_weights(prev_accel_constraint=False)
  mpc.set_cur_state(speed, acceleration)
  mpc.update(radar_state(), reference=reference)
  return mpc


def assert_actual_scheduled_stop(mpc, *, expected_solve_calls=1):
  node = mpc.effective_rest_node
  assert node is not None and mpc.rest_goal.native_node < node <= N
  assert mpc.diagnostic_solve_calls == expected_solve_calls
  assert mpc.diagnostic_last_solver_return == mpc.last_solver_status == mpc.solution_status == 0
  assert mpc.effective_rest_state == 'satisfied'
  np.testing.assert_allclose(mpc.x_sol, integrate(mpc.x0, mpc.j_solution), atol=QP_FEASIBILITY_TOL, rtol=0.0)
  assert np.max(np.abs(mpc.x_sol[node:, 1:])) <= REST_STATE_TOL + QP_FEASIBILITY_TOL
  assert np.max(mpc.a_solution[node:]) <= QP_FEASIBILITY_TOL
  assert np.all(mpc.a_solution >= mpc.params[:, 0] - QP_FEASIBILITY_TOL)
  assert np.all(mpc.a_solution <= mpc.params[:, 1] + QP_FEASIBILITY_TOL)
  # Independently inspect each quadratic interval, including interior extrema.
  for state, jerk, duration in zip(mpc.x_sol[:-1], mpc.j_solution, np.diff(T_IDXS), strict=True):
    velocity = np.polynomial.Polynomial([state[1], state[2], 0.5 * jerk])
    samples = [0.0, duration, *(root for root in velocity.deriv().roots() if 0.0 < root < duration)]
    assert np.min(velocity(samples)) >= -QP_FEASIBILITY_TOL
  desired_distance = mpc.v_solution**2 / 5.0 + mpc.params[:, 4] * mpc.v_solution + 6.0
  danger = (mpc.params[:, 2] - mpc.x_sol[:, 0] - mpc.params[:, 5] * desired_distance) / (mpc.v_solution + 10.0)
  assert np.min(danger) >= -QP_FEASIBILITY_TOL


def short_template_scheduler():
  """A pure scheduling object with no allocated solver or native entry point."""
  mpc = LongitudinalMpc.__new__(LongitudinalMpc)
  speed = 73.0 / 150.0
  mpc.x0 = np.array([0.0, speed, 0.0])
  mpc.params = np.zeros((N+1, PARAM_DIM))
  mpc.params[:, 0], mpc.params[:, 1], mpc.params[:, 2] = ACCEL_MIN, ACCEL_MAX, 1000.0
  mpc.params[:, 4], mpc.params[:, 5] = 1.45, LEAD_DANGER_FACTOR
  mpc.params[:, RISK_STOP_DISTANCE] = 6.0
  mpc.rest_goal = evidence(stopping_curve(speed, 0.03), speed).rest_goal
  assert mpc.rest_goal.native_node == 1
  mpc.diagnostic_rest_goal = {}
  mpc.effective_rest_node = None
  return mpc


def test_short_schedule_intersects_a1_bounds_before_choosing_the_template():
  mpc = short_template_scheduler()
  source, initial, parameters = mpc.rest_goal, mpc.x0.copy(), mpc.params.copy()
  original = asdict(source)
  unconstrained_states, _ = mpc.future_rest_primal(mpc.x0, 2)
  assert unconstrained_states[1, 2] == pytest.approx(-3.500025, abs=1e-12)
  assert unconstrained_states[1, 2] < mpc.params[1, 0]
  states, jerks = mpc.schedule_future_rest()
  assert mpc.effective_rest_node == 2
  assert ACCEL_MIN <= states[1, 2] <= -3.49605
  assert 0.0 <= states[2, 1] <= REST_STATE_TOL
  np.testing.assert_array_equal(states[2:, 2], 0.0)
  np.testing.assert_array_equal(states[2:, 1], states[2, 1])
  assert_future_primal_witness(initial, 2, states, jerks)
  np.testing.assert_array_equal(mpc.x0, initial)
  np.testing.assert_array_equal(mpc.params, parameters)
  assert mpc.rest_goal is source and asdict(source) == original
  assert not hasattr(mpc, 'solver')


def test_short_schedule_does_not_clamp_an_empty_a1_intersection_into_a_witness():
  mpc = short_template_scheduler()
  mpc.params[:, 0] = -3.4
  source, initial, parameters = mpc.rest_goal, mpc.x0.copy(), mpc.params.copy()
  original = asdict(source)
  states, jerks = mpc.schedule_future_rest()
  assert mpc.effective_rest_node == 3
  assert mpc.diagnostic_rest_goal['schedulingCandidates'][0] == {
    'node': 2, 'reason': 'scheduled_rest_acceleration_interval_empty',
  }
  assert np.all(states[:, 2] >= parameters[:, 0])
  assert_future_primal_witness(initial, 3, states, jerks)
  np.testing.assert_array_equal(mpc.x0, initial)
  np.testing.assert_array_equal(mpc.params, parameters)
  assert mpc.rest_goal is source and asdict(source) == original
  assert not hasattr(mpc, 'solver')


@pytest.mark.parametrize('endpoint', ['lower', 'upper'])
def test_short_schedule_accepts_a_closed_single_point_a1_intersection(endpoint):
  mpc = short_template_scheduler()
  minimum, maximum = mpc.future_rest_acceleration_interval(mpc.x0, 2)
  point = max(minimum, ACCEL_MIN) if endpoint == 'lower' else min(maximum, ACCEL_MAX)
  mpc.params[1, :2] = point
  states, jerks = mpc.schedule_future_rest()
  assert mpc.effective_rest_node == 2 and states[1, 2] == point
  assert_future_primal_witness(mpc.x0, 2, states, jerks)
  assert not hasattr(mpc, 'solver')


def test_short_schedule_does_not_fill_a_one_ulp_empty_a1_intersection():
  mpc = short_template_scheduler()
  _, maximum = mpc.future_rest_acceleration_interval(mpc.x0, 2)
  mpc.params[1, 0] = np.nextafter(maximum, np.inf)
  states, jerks = mpc.schedule_future_rest()
  assert mpc.effective_rest_node == 3
  assert mpc.diagnostic_rest_goal['schedulingCandidates'][0]['reason'] == 'scheduled_rest_acceleration_interval_empty'
  assert_future_primal_witness(mpc.x0, 3, states, jerks)
  assert not hasattr(mpc, 'solver')


def test_original_future_mathematical_contract_requires_explicit_scheduling_opt_in():
  reference = typed_reference(stopping_curve(10.0, 1.0), 10.0)
  assert not reference.allow_future_rest_reschedule
  mpc = solve(reference)
  assert mpc.rest_goal is reference.typed_model.rest_goal
  assert mpc.diagnostic_last_solver_return == 0 and mpc.solution_status == MPC_GOAL_INFEASIBLE
  assert mpc.rest_goal_reason == 'rest_goal_time_below_braking_lower_bound'
  assert mpc.effective_rest_node is None


def test_future_schedule_keeps_the_original_goal_and_uses_the_first_fully_witnessed_later_node():
  reference = replace(typed_reference(stopping_curve(10.0, 1.0), 10.0), allow_future_rest_reschedule=True)
  source = reference.typed_model.rest_goal
  original = asdict(source)
  mpc = solve(reference)
  assert_actual_scheduled_stop(mpc)
  assert mpc.effective_rest_node == 7  # node six needs more than the unchanged -3.5 acceleration envelope
  assert mpc.rest_goal is source and asdict(source) == original
  assert mpc.rest_goal_state == 'infeasible' and mpc.rest_goal_braking_accel is None
  diagnostic = mpc.diagnostic_rest_goal
  assert diagnostic['nativeNode'] == source.native_node
  assert diagnostic['nativeTime'] == source.native_time_s
  assert diagnostic['eventProxyEpochNs'] == source.event_proxy_epoch_ns
  assert diagnostic['nominalPosition'] == source.nominal_position and not diagnostic['physicalPositionBound']
  assert diagnostic['effectiveNativeNode'] == 7 and diagnostic['effectiveState'] == 'satisfied'
  assert diagnostic['effectiveLateness'] == T_IDXS[7] - source.remaining_time_s
  assert diagnostic['rescheduleDelay'] == T_IDXS[7] - source.native_time_s
  candidates = diagnostic['schedulingCandidates']
  assert [row['node'] for row in candidates] == list(range(source.native_node + 1, 8))
  assert all(row['reason'] for row in candidates[:-1]) and not candidates[-1]['reason']
  # Asking the original mathematical question still gets its original rejection.
  mpc.check_rest_goal()
  assert mpc.solution_status == MPC_GOAL_INFEASIBLE and mpc.rest_goal is source


def test_captured_j451_initial_pair_can_use_a_later_node_without_rewriting_the_declared_source_time():
  # Captured x0 plus a declared synthetic stop/go source; this is not the full route replay.
  speed, acceleration = 0.07988708466291428, -1.2803713083267212
  raw = np.where(MODEL_TIMES < 1.0, stopping_curve(speed, 0.03), 0.2 * (MODEL_TIMES - 1.0))
  typed = evidence(raw, speed, anchor_ns=SOURCE_NS + 860_104, now_ns=SOURCE_NS + 860_104)
  source = typed.rest_goal
  assert source.native_node == 1 and source.remaining_time_s == pytest.approx(0.038202396)
  assert source.future_forward and not typed.allows_departure
  reference = replace(typed_reference(raw, speed, typed=typed), allow_future_rest_reschedule=True)
  mpc = solve(reference, speed, acceleration)
  assert_actual_scheduled_stop(mpc)
  assert mpc.effective_rest_node == 2
  assert mpc.rest_goal is source and mpc.rest_goal_reason == 'rest_goal_time_below_discrete_lower_bound'


def test_already_reachable_future_goal_keeps_the_identical_native_problem_and_result():
  reference = typed_reference(stopping_curve(), 10.0)
  old, current = solve(reference), solve(replace(reference, allow_future_rest_reschedule=True))
  assert old.solution_status == current.solution_status == 0
  assert old.rest_goal_state == current.rest_goal_state == 'satisfied'
  assert current.effective_rest_node is None
  for name in ('x0', 'x_sol', 'u_sol', 'params', 'yref', 'diagnostic_reference_weights'):
    np.testing.assert_array_equal(getattr(current, name), getattr(old, name))
  for current_bounds, old_bounds in zip(current.diagnostic_reference_bounds, old.diagnostic_reference_bounds, strict=True):
    np.testing.assert_array_equal(current_bounds, old_bounds)


@pytest.mark.parametrize('speed,acceleration', [(0.0, 0.0), (0.2, 0.0), (0.0, 1.01 * REST_STATE_TOL)])
def test_current_due_rest_is_never_rescheduled(speed, acceleration):
  reference = typed_due_reference(speed)
  old = solve(reference, speed, acceleration)
  current = solve(replace(reference, allow_future_rest_reschedule=True), speed, acceleration)
  assert current.rest_goal.native_node == 0 and current.effective_rest_node is None
  assert current.solution_status == old.solution_status
  assert current.rest_goal_state == old.rest_goal_state
  assert current.rest_goal_reason == old.rest_goal_reason
  np.testing.assert_array_equal(current.x0, old.x0)
  np.testing.assert_array_equal(current.x_sol, old.x_sol)


def test_no_later_witness_within_the_existing_acceleration_envelope_keeps_the_original_failure():
  reference = replace(typed_reference(stopping_curve(10.0, 1.0), 10.0),
                      accel_min=-0.1, allow_future_rest_reschedule=True)
  mpc = solve(reference)
  assert mpc.diagnostic_solve_calls == 1 and mpc.diagnostic_last_solver_return == 0
  assert mpc.effective_rest_node is None and mpc.solution_status == MPC_GOAL_INFEASIBLE
  assert mpc.rest_goal_reason == 'rest_goal_time_below_braking_lower_bound'
  assert all(row['reason'] for row in mpc.diagnostic_rest_goal['schedulingCandidates'])


@pytest.mark.parametrize('constraint', ['acceleration', 'lead_danger'])
def test_a_kinematically_valid_primal_cannot_bypass_the_original_soft_constraint_rows(constraint):
  initial = np.array([0.0, 10.0, 0.0])
  states, jerks = LongitudinalMpc.future_rest_primal(initial, 7)
  assert_future_primal_witness(initial, 7, states, jerks)
  parameters = np.zeros((N+1, PARAM_DIM))
  parameters[:, 0], parameters[:, 1], parameters[:, 2] = ACCEL_MIN, ACCEL_MAX, 1000.0
  parameters[:, 4], parameters[:, 5] = 1.45, LEAD_DANGER_FACTOR
  parameters[:, RISK_STOP_DISTANCE] = 6.0
  if constraint == 'acceleration':
    parameters[:, 0] = -3.0
  else:
    parameters[:, 2] = 1.0
  failure = LongitudinalMpc.scheduled_rest_trajectory_error(initial, states, jerks, 7, parameters, tolerance=QP_FEASIBILITY_TOL)
  assert failure == 'scheduled_rest_' + constraint + '_envelope_violated'


def test_native_success_does_not_accept_a_scheduled_solution_with_broken_dynamics():
  reference = replace(typed_reference(stopping_curve(10.0, 1.0), 10.0), allow_future_rest_reschedule=True)
  mpc = LongitudinalMpc()
  mpc.set_weights(prev_accel_constraint=False)
  mpc.set_cur_state(10.0, 0.0)
  native_run = mpc.run

  def run_with_corrupted_result():
    native_run()
    assert mpc.diagnostic_last_solver_return == 0
    mpc.x_sol[1, 0] += 0.01

  with patch.object(mpc, 'run', side_effect=run_with_corrupted_result):
    mpc.update(radar_state(), reference=reference)
  assert mpc.diagnostic_solve_calls == 1 and mpc.diagnostic_last_solver_return == 0
  assert mpc.solution_status == MPC_GOAL_UNSATISFIED and mpc.effective_rest_state == 'unsatisfied'
  assert mpc.failure_reason == 'native_rest_dynamics_residual_exceeded'
  assert mpc.rest_goal_braking_accel == ACCEL_MIN and not mpc.previous_solution_valid


def normal_rest_intent():
  return zero_intent(reference_kind=int(MotionReferenceKind.REST))


@pytest.mark.parametrize('rejected', [False, True])
def test_actual_clipped_stop_braking_continues_without_requiring_a_prior_failure_code(rejected):
  harness = tracking()
  intent = rejected_intent() if rejected else normal_rest_intent()
  kwargs = {'plan_valid': False, 'braking_plan_valid': True, 'solver_status': -4} if rejected else {}
  assert harness.step(-3.5, intent, accel_limits=(-2.0, 1.0), **kwargs) == -2.0
  supervisor = harness.control.motion_supervisor
  origin = supervisor._rest_braking_origin
  assert origin == (INSTANCE_NS, harness.plan_time_ns, harness.model_time_ns, harness.now_ns)
  assert supervisor.phase == MotionPhase.TRACK and not supervisor.resolved_should_stop
  assert harness.step(-0.1, zero_intent()) == -2.0
  assert supervisor.rest_braking_accel_limit == -2.0 and supervisor._rest_braking_origin == origin
  assert not supervisor.degraded and not supervisor.reasons


def test_a_normal_rest_description_without_final_output_confirmation_cannot_arm_continuity():
  harness = MotionHarness()
  harness.step(normal_rest_intent(), a_target=-3.0, v_ego=10.0)
  assert harness.supervisor._pending_rest_braking is not None
  assert harness.supervisor._rest_braking_origin is None
  harness.step(zero_intent(), a_target=-0.1, v_ego=10.0)
  assert harness.supervisor._rest_braking_origin is None and harness.supervisor.rest_braking_accel_limit is None


def test_repeated_normal_rest_identity_cannot_reconfirm_a_stronger_output():
  harness = MotionHarness()
  intent = normal_rest_intent()
  harness.step(intent, a_target=-3.0, v_ego=10.0)
  harness.supervisor.confirm_braking_output(-1.0, ACCEL_LIMITS, now_ns=harness.now_ns)
  origin = harness.supervisor._rest_braking_origin
  harness.step(intent, a_target=-3.0, v_ego=10.0, new_plan=False, new_model=False)
  harness.supervisor.confirm_braking_output(-3.0, ACCEL_LIMITS, now_ns=harness.now_ns)
  assert harness.supervisor._rest_braking_accel == -1.0 and harness.supervisor._rest_braking_origin == origin
  harness.step(zero_intent(), a_target=-0.1, v_ego=10.0)
  assert harness.supervisor.rest_braking_accel_limit == -1.0


def test_a_normal_rest_plan_waiting_for_rearm_cannot_confirm_a_new_stop_output():
  harness = MotionHarness()
  harness.step(rejected_intent(), a_target=-3.5, v_ego=10.0, plan_valid=False, braking_plan_valid=True, solver_status=-4)
  harness.supervisor.confirm_braking_output(-3.5, ACCEL_LIMITS, now_ns=harness.now_ns)
  harness.step(normal_rest_intent(), a_target=-3.0, v_ego=10.0, new_model=False)
  assert harness.supervisor._pending_rest_braking is None and harness.supervisor._rest_braking_origin is None
  harness.supervisor.confirm_braking_output(-3.0, ACCEL_LIMITS, now_ns=harness.now_ns)
  assert harness.supervisor._rest_braking_origin is None


def test_positive_final_output_does_not_arm_normal_rest_continuity():
  harness = ControlHarness()
  harness.state.vEgo = 10.0
  assert harness.step(1.0, normal_rest_intent()) > 0.0
  assert harness.control.motion_supervisor._rest_braking_origin is None
