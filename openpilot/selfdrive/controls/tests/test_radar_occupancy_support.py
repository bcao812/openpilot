"""Retired occupancy metadata cannot modify formal lead control or model rest."""
from copy import deepcopy

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc
from openpilot.selfdrive.controls.tests.test_mpc_problem_contract import NativeProblemRecorder, reference_case
from openpilot.selfdrive.controls.tests.test_radar_fusion_mpc import fused_state, object_lead, solve


@pytest.mark.parametrize('supported_end', [0.0, .848, 10.0, float('nan')])
def test_retired_support_fields_cannot_change_an_ordinary_formal_lead(supported_end):
  lead = object_lead(distance=30.0, speed=4.0)
  old_metadata = deepcopy(lead)
  old_metadata.evidence.conflictSupportVersion = 1
  old_metadata.evidence.conflictSupportedEnd = supported_end
  old_metadata.evidence.physicalConflictValid = True
  old_metadata.evidence.physicalConflictStart = 0.0
  old_metadata.evidence.physicalConflictEnd = .2
  before = solve(fused_state(display=(lead,)))
  after = solve(fused_state(display=(old_metadata,)))
  np.testing.assert_array_equal(before.params, after.params)
  np.testing.assert_array_equal(before.x_sol, after.x_sol)
  assert before.crash_cnt == after.crash_cnt
  assert before.source_object_identity == after.source_object_identity


@pytest.mark.parametrize('kind', ['due_rest', 'future_rest', 'scheduled_rest'])
def test_ignored_extra_objects_cannot_install_rows_or_change_model_rest(kind):
  speed, reference = reference_case(kind)
  epoch = reference.typed_model.source.model_mono_time_ns
  extra = object_lead(distance=.01, speed=-10.0, ego_speed=speed, end=.2,
                      persistent=False, epoch=epoch, physical_window=(0.0, .1))
  extra.evidence.conflictSupportVersion, extra.evidence.conflictSupportedEnd = 1, .1
  planners = []
  for state in (fused_state(), fused_state(extra)):
    planner = LongitudinalMpc()
    recorder = NativeProblemRecorder(planner.solver)
    planner.solver = recorder
    planner.set_cur_state(speed, 0.0)
    planner.set_weights(prev_accel_constraint=False)
    planner.update(state, reference=reference, now_ns=epoch, reference_ns=epoch)
    assert planner.solution_status == 0
    assert planner.diagnostic_solve_calls == 1
    planners.append((planner, recorder))
  (before, before_native), (after, after_native) = planners
  assert before.rest_goal_state == after.rest_goal_state
  assert before.effective_rest_state == after.effective_rest_state
  np.testing.assert_array_equal(before.params, after.params)
  np.testing.assert_array_equal(before.x_sol, after.x_sol)
  assert before_native.constraints.keys() == after_native.constraints.keys()
  for key in before_native.constraints:
    np.testing.assert_array_equal(before_native.constraints[key], after_native.constraints[key])
  assert after.crash_cnt == 0 and after.object_influence_identities == ()
