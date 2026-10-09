"""A failed continuous check may refine once, never change its problem."""
from unittest.mock import patch

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LongitudinalMpc, MPC_GOAL_UNSATISFIED, MPC_INVALID_SOLUTION, MPC_OCCUPANCY_UNSATISFIED, MpcReference,
)
from openpilot.selfdrive.controls.tests.test_mpc_problem_contract import NativeProblemRecorder, reference_case
from openpilot.selfdrive.controls.tests.test_radar_fusion_mpc import EPOCH_NS, fused_state, object_lead


class RefinementRecorder(NativeProblemRecorder):
  def __init__(self, native, defect='once'):
    super().__init__(native)
    self.defect = defect
    self.parameters = {}
    self.problems = []
    self.native_times = []

  def set(self, stage, field, value):
    if field == 'p':
      self.parameters[stage] = np.array(value, copy=True)
    return self.native.set(stage, field, value)

  def solve(self):
    self.problems.append({
      'parameters': {key: value.copy() for key, value in self.parameters.items()},
      'costs': {key: value.copy() for key, value in self.costs.items()},
      'constraints': {key: value.copy() for key, value in self.constraints.items()},
    })
    if self.defect == 'second_native_failure' and len(self.problems) == 2:
      return 4
    result = super().solve()
    if result == 0:
      self.native_times.append(float(self.native.get_stats('time_tot')[0]))
    return result

  def get(self, stage, field):
    value = self.native.get(stage, field)
    inject = self.defect == 'always' or self.defect in ('once', 'nonfinite', 'second_native_failure') and len(self.problems) == 1
    if inject and stage == 1 and field == 'x':
      value = value.copy()
      value[0] = np.nan if self.defect == 'nonfinite' else value[0] + .01
    return value


def run_problem(defect):
  mpc = LongitudinalMpc()
  recorder = RefinementRecorder(mpc.solver, defect)
  mpc.solver = recorder
  mpc.set_cur_state(10.0, 0.0)
  mpc.set_weights(prev_accel_constraint=False)
  lead = object_lead(17, distance=20.0, speed=2.0, ego_speed=10.0,
                     start=1.0, end=2.0, path='entering', persistent=False)
  mpc.update(fused_state(lead), reference=MpcReference(20.0),
             now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  return mpc, recorder


def assert_same_problem(recorder):
  assert len(recorder.problems) == 2
  first, second = recorder.problems
  for group in ('parameters', 'costs', 'constraints'):
    assert first[group].keys() == second[group].keys()
    for key, value in first[group].items():
      np.testing.assert_array_equal(value, second[group][key], err_msg=f'{group} {key}')


def test_single_residual_refines_same_native_problem_then_passes_all_checks():
  mpc, recorder = run_problem('once')
  assert_same_problem(recorder)
  assert mpc.solution_status == mpc.last_solver_status == 0
  assert mpc.previous_solution_valid
  assert mpc.diagnostic_occupancy_refinement['firstFailure'] == 'finite_occupancy_dynamics_residual_exceeded'
  assert mpc.diagnostic_occupancy_refinement['attempts'] == 2
  assert mpc.diagnostic_occupancy_refinement['finalStatus'] == 0
  assert mpc.diagnostic_occupancy_refinement['finalFailure'] == ''
  assert mpc.solve_time == pytest.approx(sum(recorder.native_times))
  assert mpc.occupancy_trajectory_error(mpc.x_sol, mpc.j_solution, tolerance=1e-9, post_native=True) == ''


def test_repeated_residual_is_bounded_and_cannot_publish_an_invalid_solution():
  mpc, recorder = run_problem('always')
  assert_same_problem(recorder)
  assert mpc.solution_status == MPC_OCCUPANCY_UNSATISFIED
  assert not mpc.previous_solution_valid
  assert mpc.diagnostic_occupancy_refinement['finalStatus'] == MPC_OCCUPANCY_UNSATISFIED
  np.testing.assert_array_equal(mpc.v_solution, np.zeros_like(mpc.v_solution))
  assert mpc.solve_time == pytest.approx(sum(recorder.native_times))


def test_native_failure_on_refinement_retains_explicit_failure_and_original_reset():
  mpc, recorder = run_problem('second_native_failure')
  assert_same_problem(recorder)
  assert mpc.solution_status == 4 and not mpc.previous_solution_valid
  assert mpc.diagnostic_occupancy_refinement['finalStatus'] == 4
  assert mpc.solve_time >= recorder.native_times[0]
  np.testing.assert_array_equal(mpc.v_solution, np.zeros_like(mpc.v_solution))


def test_nonfinite_native_result_is_not_retried():
  mpc, recorder = run_problem('nonfinite')
  assert len(recorder.problems) == 1
  assert mpc.solution_status == MPC_INVALID_SOLUTION
  assert mpc.diagnostic_occupancy_refinement is None


def test_ordinary_accepted_problem_keeps_single_solve():
  mpc, recorder = run_problem('none')
  assert len(recorder.problems) == 1 and mpc.solution_status == 0
  assert mpc.diagnostic_occupancy_refinement is None


def test_refinement_never_bypasses_continuous_geometry_rejection():
  with patch.object(LongitudinalMpc, 'occupancy_trajectory_error', return_value='finite_occupancy_clearance_violated'):
    mpc, recorder = run_problem('once')
  assert_same_problem(recorder)
  assert mpc.solution_status == MPC_OCCUPANCY_UNSATISFIED
  assert mpc.failure_reason == 'finite_occupancy_clearance_violated'
  assert not mpc.previous_solution_valid


@pytest.mark.parametrize('kind', ['due_rest', 'scheduled_rest'])
def test_rest_goal_is_revalidated_for_the_refined_candidate(kind):
  mpc = LongitudinalMpc()
  recorder = RefinementRecorder(mpc.solver, 'none')
  mpc.solver = recorder
  speed, reference = reference_case(kind)
  epoch = reference.typed_model.source.model_mono_time_ns
  mpc.set_cur_state(speed, 0.0)
  mpc.set_weights(prev_accel_constraint=False)
  lead = object_lead(17, distance=8.0, speed=0.0, ego_speed=speed,
                     start=.1, end=2.0, path='entering', persistent=False, epoch=epoch)
  original_check = mpc.occupancy_trajectory_error
  original_rest = mpc.project_rest_precision
  checked_solutions = []

  def reject_first_geometry(*args, **kwargs):
    # Inject the occupancy refusal only after the first real rest check has
    # accepted. A pre-rest reintegration call must retain its normal checks.
    rest_accepted = mpc.rest_goal_state == 'satisfied' if kind == 'due_rest' else mpc.effective_rest_state == 'satisfied'
    if len(recorder.problems) == 1 and rest_accepted:
      return 'finite_occupancy_clearance_violated'
    return original_check(*args, **kwargs)

  def observe_rest(*args, **kwargs):
    checked_solutions.append(len(recorder.problems))
    if kind == 'due_rest' and len(recorder.problems) == 2:
      return 'refined_candidate_rest_rejected'
    return original_rest(*args, **kwargs)

  with patch.object(mpc, 'occupancy_trajectory_error', side_effect=reject_first_geometry), \
       patch.object(mpc, 'project_rest_precision', side_effect=observe_rest):
    mpc.update(fused_state(lead), reference=reference, now_ns=epoch, reference_ns=epoch)
  assert_same_problem(recorder)
  assert 1 in checked_solutions and 2 in checked_solutions
  if kind == 'due_rest':
    assert mpc.solution_status == MPC_GOAL_UNSATISFIED and not mpc.previous_solution_valid
    assert mpc.rest_goal_reason == 'refined_candidate_rest_rejected'
  else:
    assert mpc.solution_status == 0 and mpc.effective_rest_state == 'satisfied'
