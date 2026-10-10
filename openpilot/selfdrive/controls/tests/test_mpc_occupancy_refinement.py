"""Removing extra-object solving leaves ordinary native and rest checks intact."""
from unittest.mock import patch

import pytest

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LongitudinalMpc, MPC_GOAL_UNSATISFIED, MPC_INVALID_SOLUTION, MpcReference,
)
from openpilot.selfdrive.controls.tests.test_mpc_problem_contract import NativeProblemRecorder, reference_case
from openpilot.selfdrive.controls.tests.test_radar_fusion_mpc import EPOCH_NS, fused_state, object_lead


@pytest.mark.parametrize('failure,status', [(None, 0), ('solver_status', 4), ('nonfinite', MPC_INVALID_SOLUTION)])
def test_extra_objects_do_not_add_solves_or_change_native_failure_semantics(failure, status):
  mpc = LongitudinalMpc()
  recorder = NativeProblemRecorder(mpc.solver)
  recorder.failure = failure
  mpc.solver = recorder
  mpc.set_cur_state(10.0, 0.0)
  mpc.set_weights(prev_accel_constraint=False)
  extra = object_lead(17, distance=.1, speed=-5.0, end=.2, persistent=False, physical_window=(0.0, .1))
  mpc.update(fused_state(extra), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert len(recorder.solved_costs) == mpc.diagnostic_solve_calls == 1
  assert mpc.solution_status == status
  assert mpc.previous_solution_valid == (status == 0)
  assert mpc.crash_cnt == 0 and mpc.object_influence_identities == ()


@pytest.mark.parametrize('kind', ['due_rest', 'scheduled_rest'])
def test_model_rest_keeps_its_independent_post_solve_rejection(kind):
  mpc = LongitudinalMpc()
  recorder = NativeProblemRecorder(mpc.solver)
  mpc.solver = recorder
  speed, reference = reference_case(kind)
  epoch = reference.typed_model.source.model_mono_time_ns
  mpc.set_cur_state(speed, 0.0)
  mpc.set_weights(prev_accel_constraint=False)
  extra = object_lead(17, distance=.1, speed=-5.0, end=.2, persistent=False, epoch=epoch)
  with patch.object(mpc, 'project_rest_precision', return_value='rest_fixture_rejected') as checked:
    mpc.update(fused_state(extra), reference=reference, now_ns=epoch, reference_ns=epoch)
  assert checked.called and len(recorder.solved_costs) == 1
  assert mpc.solution_status == MPC_GOAL_UNSATISFIED and not mpc.previous_solution_valid
  assert mpc.rest_goal_braking_accel is not None
  assert mpc.crash_cnt == 0 and mpc.object_influence_identities == ()
