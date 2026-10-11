"""Small native regressions for the braking-to-rest transition inside a node."""

from dataclasses import asdict, replace
from unittest.mock import Mock

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  MPC_GOAL_INFEASIBLE, MPC_GOAL_UNSATISFIED, MPC_SOLVER_ERROR, QP_FEASIBILITY_TOL,
  T_IDXS, LongitudinalMpc, MpcReference,
)
from openpilot.selfdrive.controls.tests.test_interval_rest import CAPTURED_STATES
from openpilot.selfdrive.controls.tests.test_typed_model_reference import stopping_curve
from openpilot.selfdrive.controls.tests.test_typed_reference_mpc import typed_reference
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state


def reference(speed, *, permitted=True):
  return replace(typed_reference(stopping_curve(speed, 0.2), speed), allow_interval_rest=permitted)


@pytest.mark.parametrize('speed,acceleration', CAPTURED_STATES)
def test_recorded_node_two_goal_accepts_an_actual_native_early_stop(speed, acceleration):
  mpc = LongitudinalMpc()
  mpc.set_weights(prev_accel_constraint=False)
  mpc.set_cur_state(speed, acceleration)
  goal = reference(speed)
  original = asdict(goal.typed_model.rest_goal)
  mpc.update(radar_state(), reference=goal)
  assert mpc.diagnostic_solve_calls == 1 and mpc.diagnostic_last_solver_return == 0
  assert mpc.solution_status == 0 and mpc.rest_goal_state == 'satisfied'
  assert mpc.interval_rest is not None and mpc.interval_rest.stop_time < T_IDXS[1]
  assert mpc.effective_rest_node is None
  assert asdict(mpc.rest_goal) == original and original['native_node'] == 2
  np.testing.assert_array_equal(mpc.x0, [0.0, speed, acceleration])
  np.testing.assert_array_equal(mpc.x_sol[0], mpc.x0)
  np.testing.assert_array_equal(mpc.x_sol[1:, 1:], 0.0)
  # The old linear interpolation would retain braking beyond the event; the
  # public sampler and actuator preview must both see the stationary hold.
  event = mpc.interval_rest.stop_time
  velocity, accel, jerk = mpc.sample_solution(np.array([event, (event + T_IDXS[1]) / 2.0, 0.2]))
  np.testing.assert_array_equal(velocity, 0.0)
  np.testing.assert_array_equal(accel, 0.0)
  np.testing.assert_array_equal(jerk, 0.0)
  assert mpc.diagnostic_rest_goal['maximumNativeDynamicsDefect'] <= QP_FEASIBILITY_TOL


def test_existing_callers_keep_the_original_discrete_problem_without_opt_in():
  speed, acceleration = CAPTURED_STATES[0]
  mpc = LongitudinalMpc()
  mpc.set_cur_state(speed, acceleration)
  mpc.update(radar_state(), reference=reference(speed, permitted=False))
  assert mpc.interval_rest is None
  assert mpc.solution_status == MPC_GOAL_INFEASIBLE
  assert mpc.rest_goal_reason == 'rest_goal_time_below_discrete_lower_bound'


@pytest.mark.parametrize('corruption', ['position', 'velocity', 'jerk'])
def test_native_success_does_not_authorize_replacing_a_bad_result_with_the_seed(corruption):
  speed, acceleration = CAPTURED_STATES[0]
  mpc = LongitudinalMpc()
  mpc.set_weights(prev_accel_constraint=False)
  mpc.set_cur_state(speed, acceleration)
  native = mpc.solver
  observed = Mock(wraps=native)

  def get(stage, field):
    value = native.get(stage, field).copy()
    if corruption == 'jerk' and stage == 0 and field == 'u':
      value[0] += 0.01
    elif field == 'x' and stage == 1 and corruption != 'jerk':
      value[0 if corruption == 'position' else 1] += 1e-5
    return value

  observed.get.side_effect = get
  mpc.solver = observed
  mpc.update(radar_state(), reference=reference(speed))
  assert mpc.diagnostic_last_solver_return == 0
  assert mpc.solution_status == MPC_GOAL_UNSATISFIED
  assert mpc.rest_goal_reason == 'interval_rest_native_precision_budget_exceeded'
  assert not mpc.previous_solution_valid
  with pytest.raises(ValueError, match='unaccepted'):
    mpc.sample_solution(0.1)


def test_missing_native_support_is_an_explicit_solver_failure():
  speed, acceleration = CAPTURED_STATES[0]
  mpc = LongitudinalMpc()
  mpc.set_cur_state(speed, acceleration)
  observed = Mock(wraps=mpc.solver)
  observed.custom_update.return_value = 1
  mpc.solver = observed
  mpc.update(radar_state(), reference=reference(speed))
  assert mpc.solution_status == MPC_SOLVER_ERROR
  assert mpc.diagnostic_solve_calls == 0 and observed.solve.call_count == 0
  assert not mpc.previous_solution_valid


@pytest.mark.parametrize('velocity_residual,accepted', [(0.5e-9, True), (2.0e-9, False)])
def test_position_roundoff_may_accumulate_but_motion_state_budget_stays_fixed(velocity_residual, accepted):
  speed, acceleration = CAPTURED_STATES[0]
  mpc = LongitudinalMpc()
  mpc.set_weights(prev_accel_constraint=False)
  mpc.set_cur_state(speed, acceleration)
  native = mpc.solver
  observed = Mock(wraps=native)

  def get(stage, field):
    if field != 'x' or stage == 0:
      return native.get(stage, field)
    states, _ = mpc.interval_rest.native_trajectory(T_IDXS)
    state = states[stage].copy()
    state[0] += velocity_residual * (T_IDXS[stage] - T_IDXS[1])
    state[1] = velocity_residual
    return state

  observed.get.side_effect = get
  mpc.solver = observed
  mpc.update(radar_state(), reference=reference(speed))
  assert mpc.diagnostic_last_solver_return == 0
  assert mpc.diagnostic_rest_goal['maximumStateCorrection'] > QP_FEASIBILITY_TOL
  if accepted:
    assert mpc.solution_status == 0 and mpc.rest_goal_state == 'satisfied'
    np.testing.assert_array_equal(mpc.x_sol[1:, 1:], 0.0)
  else:
    assert mpc.solution_status == MPC_GOAL_UNSATISFIED and not mpc.previous_solution_valid


def test_ordinary_update_restores_whole_interval_integration_after_early_rest():
  speed, acceleration = CAPTURED_STATES[0]
  mpc = LongitudinalMpc()
  observed = Mock(wraps=mpc.solver)
  mpc.solver = observed
  mpc.set_cur_state(speed, acceleration)
  mpc.update(radar_state(), reference=reference(speed))
  assert mpc.solution_status == 0
  event = mpc.interval_rest.stop_time
  mpc.reset()
  mpc.set_weights(prev_accel_constraint=False)
  mpc.set_cur_state(5.0, 0.0)
  mpc.update(radar_state(), reference=MpcReference(cruise_speed=7.0))
  assert mpc.solution_status == 0 and mpc.interval_rest is None
  np.testing.assert_array_equal(observed.custom_update.call_args_list[0].args[0], [1.0, event])
  np.testing.assert_array_equal(observed.custom_update.call_args_list[-1].args[0], [1.0, T_IDXS[1]])
  assert np.max(np.abs(mpc.trajectory_dynamics_defects(mpc.x_sol, mpc.j_solution))) < 1e-8


def test_reference_history_samples_the_hold_after_the_internal_event():
  speed, acceleration = CAPTURED_STATES[0]
  mpc = LongitudinalMpc()
  epoch = 10_000_000_000
  assert not mpc.prepare_reference_history(epoch, valid=False)
  mpc.set_cur_state(speed, acceleration)
  mpc.update(radar_state(), reference=reference(speed))
  assert mpc.solution_status == 0 and mpc.mark_reference_history(valid=True)
  assert mpc.prepare_reference_history(epoch + 50_000_000, valid=True)
  np.testing.assert_array_equal(mpc.a_prev, 0.0)


def test_early_rest_does_not_hide_observed_lead_overlap_behind_the_distance_floor():
  speed, acceleration = CAPTURED_STATES[0]
  mpc = LongitudinalMpc()
  mpc.set_weights(prev_accel_constraint=False)
  mpc.set_cur_state(speed, acceleration)
  mpc.update(radar_state(distance=0.2, speed=0.0), reference=reference(speed))
  assert mpc.solution_status != 0
  assert not mpc.previous_solution_valid
  if mpc.diagnostic_last_solver_return == 0:
    assert mpc.rest_goal_reason == 'interval_rest_lead_geometry_violated'
