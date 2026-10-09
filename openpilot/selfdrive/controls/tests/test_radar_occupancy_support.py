"""Supported hard occupancy and conservative soft risk use separate windows."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.selfdrive.controls.lib.lead_qualification import _StructSnapshot, qualified_lead
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  CRASH_DISTANCE, MPC_INVALID_INPUT, N, QP_FEASIBILITY_TOL, REFERENCE_BOUND, T_IDXS, LongitudinalMpc, MpcReference,
)
from openpilot.selfdrive.controls.lib.radar_consumption import RadarConsumption
from openpilot.selfdrive.controls.tests.test_mpc_problem_contract import NativeProblemRecorder
from openpilot.selfdrive.controls.tests.test_radar_consumption import coast, consume_frame, radar_frame
from openpilot.selfdrive.controls.tests.test_radar_fusion_mpc import (
  EPOCH_NS, action, fused_state, late_passing_objects, object_lead, solve,
)
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state


def supported(lead, end, version=1):
  lead.evidence.conflictSupportVersion = version
  lead.evidence.conflictSupportedEnd = end
  return lead


@pytest.mark.parametrize('case', ['finite_crossing', 'stopped_front', 'near_cutin'])
def test_full_supported_window_preserves_legacy_native_problem_and_action(case):
  if case == 'finite_crossing':
    lead = object_lead(distance=8.0, start=.7, end=.85, path='entering', persistent=False)
  elif case == 'stopped_front':
    lead = object_lead(distance=30.0, speed=0.0)
  else:
    lead = object_lead(distance=20.0, speed=5.0)
  original = solve(fused_state(lead))
  current = solve(fused_state(supported(deepcopy(lead), lead.evidence.conflictEnd)))
  assert original._occupancy_bounds == current._occupancy_bounds
  np.testing.assert_array_equal(original.params, current.params)
  np.testing.assert_array_equal(original.x_sol, current.x_sol)
  np.testing.assert_array_equal(original.u_sol, current.u_sol)
  assert action(current) == action(original)


def test_supported_prefix_installs_hard_rows_without_erasing_soft_tail():
  lead = supported(object_lead(distance=5.0, speed=-10.0, ego_speed=5.0, persistent=False), .1)
  planner = solve(fused_state(lead), speed=5.0)
  projection, = planner._fusion_projections
  assert projection.conflict_end == 10.0 and projection.hard_conflict_end == .1
  assert projection.occupied.all() and planner.source_object_identity == ('radar-test', 1)
  assert any(planner._occupancy_bounds)
  for stage, bound in enumerate(planner._occupancy_bounds):
    if bound is not None:
      assert T_IDXS[stage] + bound.duration <= .1
  # The same full-horizon obstacle remains in the objective after support
  # ends; this is not the free-cruise problem with the object removed.
  obstacles, tokens, _, _ = planner._influence_problem
  column = next(index for index, token in enumerate(tokens) if token is not None)
  assert np.isfinite(obstacles[T_IDXS > .1, column]).all()
  assert action(planner) < -1.0
  assert planner.occupancy_trajectory_error(planner.x_sol, planner.j_solution, tolerance=1e-9, post_native=True) == ''
  minimum, _ = planner.occupancy_clearance_extrema(projection, planner.x_sol, planner.j_solution, supported_only=True)
  tail_minimum, _ = planner.occupancy_clearance_extrema(projection, planner.x_sol, planner.j_solution)
  assert minimum >= CRASH_DISTANCE and tail_minimum < CRASH_DISTANCE


def test_impossible_current_prefix_still_fails_and_preserves_physical_mitigation():
  lead = supported(object_lead(distance=.8, start=0.0, end=10.0, path='entering', persistent=False,
                               physical_window=(0.0, .1)), .2)
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner.set_weights(prev_accel_constraint=False)
  planner.update(fused_state(lead), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status != 0 and not planner.previous_solution_valid
  assert planner.finite_risk_braking_accel == ACCEL_MIN
  proof, = planner.finite_risk_evidence
  assert proof.identity == ('radar-test', 1) and proof.conflict_end_s == .1


def test_measured_close_opposing_object_without_lateral_support_still_brakes():
  # Current measured inPath is not erased by an unavailable lateral future.
  # It cannot prove a finite hard window, but retains its entire obstacle and
  # measured opposing speed. No new physical-emergency proof is fabricated.
  lead = supported(object_lead(distance=5.0, speed=-10.0, ego_speed=10.0, persistent=False), 0.0)
  original = deepcopy(vars(lead.evidence))
  planner = solve(fused_state(lead), speed=10.0)
  projection, = planner._fusion_projections
  assert projection.occupied.all() and projection.conflict_end == 10.0
  assert projection.hard_conflict_end == 0.0 and not any(planner._occupancy_bounds)
  assert action(planner) <= -1.0 and np.min(planner.a_solution[1:]) <= -3.0
  assert planner.finite_occupancy_physics_required and planner._native_occupancy_bounds_active
  assert planner.trajectory_minimum_velocity(planner.x_sol, planner.j_solution) >= -QP_FEASIBILITY_TOL
  assert np.all(planner.a_solution[1:] >= planner.params[1:, 0] - QP_FEASIBILITY_TOL)
  assert np.all(planner.a_solution[1:] <= planner.params[1:, 1] + QP_FEASIBILITY_TOL)
  assert planner.occupancy_trajectory_error(planner.x_sol, planner.j_solution,
                                            tolerance=QP_FEASIBILITY_TOL, post_native=True) == ''
  assert planner.finite_risk_evidence == () and vars(lead.evidence) == original


@pytest.mark.parametrize('next_kind', ['no_objects', 'persistent_follow', 'nonfusion', 'reset_direct_run'])
def test_soft_tail_physics_is_installed_and_cleared_with_its_actual_native_lifecycle(next_kind):
  planner = LongitudinalMpc()
  recorder = NativeProblemRecorder(planner.solver)
  planner.solver = recorder
  planner.set_cur_state(10.0, 0.0)
  planner.set_weights(prev_accel_constraint=False)
  lead = supported(object_lead(distance=30.0, speed=5.0, persistent=False), 0.0)
  planner.update(fused_state(lead), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status == 0
  assert planner.finite_occupancy_physics_required and planner._native_occupancy_bounds_active
  assert not any(planner._occupancy_bounds)
  for stage in range(N):
    # There is a forward-motion certificate in every interval and no position
    # row. The native state boxes, including the terminal box, are physical.
    assert recorder.constraints[stage, 'C'][0, 1] == 1.0
    assert recorder.constraints[stage, 'lg'][0] == 0.0
    np.testing.assert_array_equal(recorder.constraints[stage, 'C'][:, 0], 0.0)
  for stage in range(1, N+1):
    np.testing.assert_array_equal(recorder.constraints[stage, 'lbx'], [0.0, planner.params[stage, 0]])
    assert recorder.constraints[stage, 'ubx'][1] == planner.params[stage, 1]

  def no_finite_physics_installed():
    for stage in range(N):
      np.testing.assert_array_equal(recorder.constraints[stage, 'C'], 0.0)
      np.testing.assert_array_equal(recorder.constraints[stage, 'D'], 0.0)
      np.testing.assert_array_equal(recorder.constraints[stage, 'lg'], -1.0)
      np.testing.assert_array_equal(recorder.constraints[stage, 'ug'], 1.0)
    for stage in range(1, N+1):
      np.testing.assert_array_equal(recorder.constraints[stage, 'lbx'], -REFERENCE_BOUND)
      np.testing.assert_array_equal(recorder.constraints[stage, 'ubx'], REFERENCE_BOUND)

  recorder.before_solve = no_finite_physics_installed
  if next_kind == 'reset_direct_run':
    parameters = planner.params.copy()
    planner.reset()
    no_finite_physics_installed()  # reset itself must clear native rows first.
    planner.params[:] = parameters
    planner.set_cur_state(10.0, 0.0)
    planner.run()
  else:
    state = (fused_state(object_lead(distance=30.0, speed=5.0)) if next_kind == 'persistent_follow'
             else radar_state() if next_kind == 'nonfusion' else fused_state())
    planner.update(state, reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status == 0
  assert not planner.finite_occupancy_physics_required and not planner._native_occupancy_bounds_active


@pytest.mark.parametrize('support_end', [0.0, .1])
def test_continuous_velocity_row_after_position_support_is_still_installed_and_rechecked(support_end):
  lead = supported(object_lead(distance=30.0, speed=5.0, persistent=False), support_end)
  planner = solve(fused_state(lead))
  stage = N-1
  assert planner._occupancy_bounds[stage] is None
  C, D, lower, _ = planner._rest_constraint_buffers[stage]
  assert C[0, 1] == 1.0 and lower[0] == 0.0
  assert planner.occupancy_trajectory_error(planner.x_sol, planner.j_solution,
                                            tolerance=QP_FEASIBILITY_TOL, post_native=True) == ''
  # Reinstall a stricter row after the solve. Acceptance must use this actual
  # final problem even though no position row exists in this late interval.
  lower[0] = float(C[0] @ planner.x_sol[stage] + D[0, 0] * planner.j_solution[stage]) + .1
  planner.solver.constraints_set(stage, 'lg', lower)
  assert planner.occupancy_trajectory_error(planner.x_sol, planner.j_solution,
                                            tolerance=QP_FEASIBILITY_TOL, post_native=True) == 'finite_occupancy_reintegration_constraints_violated'


@pytest.mark.parametrize('support_end', [0.0, .005])
def test_reverse_between_positive_nodes_after_support_end_is_not_accepted(support_end):
  planner = LongitudinalMpc()
  planner.set_cur_state(.0009, -.1)
  lead = supported(object_lead(distance=2.0, speed=0.0, ego_speed=.0009, persistent=False), support_end)
  planner._fusion_projections = planner.fusion_lead_projections(fused_state(lead), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  planner._occupancy_bounds = planner.finite_occupancy_bounds()
  planner.params[:, 0], planner.params[:, 1] = ACCEL_MIN, 2.0
  jerks = np.zeros(N)
  jerks[0] = 5.0
  states = np.empty((N+1, 3))
  states[0] = planner.x0
  for stage, duration in enumerate(np.diff(T_IDXS)):
    position, velocity, acceleration = states[stage]
    jerk = jerks[stage]
    states[stage+1] = (position + velocity*duration + .5*acceleration*duration**2 + jerk*duration**3/6.0,
                       velocity + acceleration*duration + .5*jerk*duration**2, acceleration + jerk*duration)
  assert np.min(states[:, 1]) > 0.0
  assert np.min(states[:, 2]) >= ACCEL_MIN and np.max(states[:, 2]) <= 2.0
  # Its analytic minimum is at .02 s, outside either supported prefix.
  assert support_end < .02 < T_IDXS[1] and .0009 - .1**2 / (2*5.0) < 0.0
  assert planner.occupancy_trajectory_error(states, jerks, tolerance=QP_FEASIBILITY_TOL) == 'finite_occupancy_continuous_velocity_violated'


def test_expired_support_is_not_revived_by_an_older_initial_state_or_treated_as_clear():
  lead = supported(object_lead(distance=20.0, persistent=False), .02)
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  now_ns, reference_ns = EPOCH_NS + 30_000_000, EPOCH_NS - 10_000_000
  projection, = planner.fusion_lead_projections(fused_state(lead), now_ns=now_ns, reference_ns=reference_ns)
  assert projection.conflict_end == pytest.approx(10.01) and projection.occupied.all()
  assert projection.hard_conflict_end == 0.0
  assert not any(planner.finite_occupancy_bounds((projection,)))
  assert qualified_lead(lead, 'brake', 0.0, now_ns=now_ns)
  assert not qualified_lead(lead, 'departure', 0.0, now_ns=now_ns)


def test_unknown_tail_never_authorizes_pass_before():
  planner = LongitudinalMpc()
  planner.set_cur_state(11.190595626831055, .16649655997753143)
  objects = late_passing_objects()
  # The established complete-window fixture can use a joint passing witness.
  original = planner.fusion_lead_projections(fused_state(*objects), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  reference = MpcReference(20.0, allow_passing=True)
  assert any(p.pass_before for p in planner.select_occupancy_branches(original, reference))
  planner._passing_witness = planner._passing_speed_limit = None
  supported(objects[1], 4.0)
  limited = planner.fusion_lead_projections(fused_state(*objects), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  chosen = planner.select_occupancy_branches(limited, reference)
  assert not any(p.pass_before for p in chosen)
  assert planner._passing_witness is None and planner._passing_speed_limit is None
  assert planner.diagnostic_passing['reason'] == 'unknown_occupancy_tail'


def test_support_change_invalidates_the_exact_occupancy_limit_cache(monkeypatch):
  planner = LongitudinalMpc()
  planner.set_cur_state(15.0, -.2)
  lead = supported(object_lead(distance=30.0, speed=-.25, ego_speed=15.0, persistent=False), .55)
  projection = planner.project_fusion_lead(lead, slot=None, reference_ns=EPOCH_NS)
  calls = []
  original = planner.ego_progress_bound

  def counted(*args):
    calls.append(args)
    return original(*args)

  monkeypatch.setattr(planner, 'ego_progress_bound', counted)
  first = planner.occupancy_behind_position_limit(projection, 2, constraint_time=.6)
  count = len(calls)
  assert planner.occupancy_behind_position_limit(projection, 2, constraint_time=.6) == first
  assert len(calls) == count
  changed = replace(projection, conflict_supported_end=.5)
  second = planner.occupancy_behind_position_limit(changed, 2, constraint_time=.6)
  assert len(calls) > count and second > first


@pytest.mark.parametrize('version,end,physical_end', [
  (2, 1.0, None), (True, 1.0, None), (1, float('nan'), None), (1, float('inf'), None),
  (1, -.01, None), (1, 10.01, None), (1, None, None), (1, .3, .4),
])
def test_malformed_support_rejected_by_both_qualification_paths(version, end, physical_end):
  lead = supported(object_lead(persistent=False), end, version)
  if physical_end is not None:
    lead.evidence.physicalConflictValid = True
    lead.evidence.physicalConflictStart = 0.0
    lead.evidence.physicalConflictEnd = physical_end
  snapshot = _StructSnapshot({**vars(lead), 'evidence': vars(lead.evidence)})
  assert not qualified_lead(lead, 'brake', 0.0, now_ns=EPOCH_NS)
  assert not qualified_lead(snapshot, 'brake', 0.0, now_ns=EPOCH_NS)
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner.update(fused_state(lead), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status == MPC_INVALID_INPUT and planner.diagnostic_solve_calls == 0


@pytest.mark.parametrize('explicit_zero', [False, True])
def test_missing_or_legacy_support_keeps_the_complete_hard_window(explicit_zero):
  lead = object_lead(distance=20.0, persistent=False)
  if explicit_zero:
    supported(lead, 0.0, 0)
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  projection, = planner.fusion_lead_projections(fused_state(lead), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert projection.hard_conflict_end == projection.conflict_end == 10.0
  assert len([bound for bound in planner.finite_occupancy_bounds((projection,)) if bound is not None]) == N


@pytest.mark.parametrize('defect', ['version', 'support', 'physical_support', 'duplicate'])
def test_expiry_cannot_hide_support_corruption_or_duplicate_conflicts(defect):
  lead = supported(coast(), 1.0)
  frame = radar_frame(lead, display=(deepcopy(lead),))
  if defect == 'version':
    lead.evidence.conflictSupportVersion = 2
  elif defect == 'support':
    lead.evidence.conflictSupportedEnd = 20.0
  elif defect == 'physical_support':
    lead.evidence.physicalConflictValid = True
    lead.evidence.physicalConflictStart = .5
    lead.evidence.physicalConflictEnd = 1.5
  else:
    frame.leadOne.evidence.conflictSupportedEnd = 2.0
  if defect != 'duplicate':
    frame.leadOne = deepcopy(lead)
  consumer = RadarConsumption()
  result = consume_frame(consumer, frame)
  assert result is frame and result.leadOne.present and result.radarObjects == [lead]
  assert consumer.expired_identities == ()
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner.update(frame, reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status == MPC_INVALID_INPUT and planner.diagnostic_solve_calls == 0
