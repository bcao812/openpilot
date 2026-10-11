"""A carried numerical proposal never becomes a successful native solve."""
from dataclasses import replace

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.selfdrive.controls.lib.lead_qualification import MAX_MEASUREMENT_AGE_NS
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.continuation import (
  MAX_DONOR_SOURCE_AGE_NS, STATE_ENVELOPE_LEVELS, TERMINAL_ACCEL_TOL, TERMINAL_JERK_TOL,
  ContinuationDonor, ContinuationProblem, build_candidate, frozen_array,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_STOP_DISTANCE, CRASH_DISTANCE, N, PARAM_DIM, QP_FEASIBILITY_TOL, REFERENCE_BOUND,
  RISK_STOP_DISTANCE, T_IDXS, LongitudinalMpc, MpcReference, get_safe_obstacle_distance,
)
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_MAX_AGE_NS
from openpilot.selfdrive.controls.tests.test_mpc_problem_contract import NativeProblemRecorder
from openpilot.selfdrive.controls.tests.test_radar_fusion_mpc import fused_state, object_lead
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state


EPOCH = 10_000_000_000
PRODUCER = EPOCH + 10_000_000
INSTANCE = EPOCH + 1_000_000


def integrate_profile(x0, acceleration):
  jerks = np.diff(acceleration) / np.diff(T_IDXS)
  states = np.empty((N+1, 3))
  states[0] = x0
  for stage, duration in enumerate(np.diff(T_IDXS)):
    # Independent textbook form for the constructed fixtures.
    x, v, a = states[stage]
    j = jerks[stage]
    states[stage+1] = (x + v*duration + .5*a*duration**2 + j*duration**3/6,
                       v + a*duration + .5*j*duration**2, a + j*duration)
  return states, jerks


def problem_for(states, *, source=EPOCH, consumed=PRODUCER, margin=100.0, lead_trajectories=()):
  parameters = np.zeros((N+1, PARAM_DIM))
  parameters[:, 0], parameters[:, 1] = ACCEL_MIN, ACCEL_MAX
  parameters[:, 4], parameters[:, 5] = 1.45, .75
  parameters[:, COMFORT_STOP_DISTANCE] = parameters[:, RISK_STOP_DISTANCE] = 6.0
  parameters[:, 2] = states[:, 0] + .75*get_safe_obstacle_distance(states[:, 1], 1.45) + margin
  lower = np.tile([0.0, ACCEL_MIN], (N+1, 1))
  upper = np.tile([REFERENCE_BOUND, ACCEL_MAX], (N+1, 1))
  rows = tuple((frozen_array(np.zeros((2, 3))), frozen_array(np.zeros((2, 1))),
                frozen_array([-1.0, -1.0]), frozen_array([1.0, 1.0])) for _ in range(N))
  return ContinuationProblem(source, consumed, source + 250_000_000, frozen_array(states[0]), frozen_array(parameters),
                              frozen_array(lower), frozen_array(upper), rows, tuple(lead_trajectories), False)


def mathematical_donor(*, terminal_accel=0.0, terminal_jerk=0.0):
  acceleration = np.zeros(N+1)
  acceleration[:3] = [.3, .25, .1]
  acceleration[-2:] = [terminal_accel - terminal_jerk*(T_IDXS[-1]-T_IDXS[-2]), terminal_accel]
  states, jerks = integrate_profile([0.0, 8.0, .3], acceleration)
  return ContinuationDonor(EPOCH, PRODUCER, INSTANCE, frozen_array(states), frozen_array(jerks), problem_for(states))


def candidate(donor, *, delta_ns=50_000_000, current=None, now=None):
  if current is None:
    current = replace(donor.problem, reference_ns=EPOCH + delta_ns, consumed_ns=PRODUCER + delta_ns,
                      evidence_expiry_ns=EPOCH + delta_ns + 250_000_000)
  return build_candidate(donor, current, now_ns=current.consumed_ns if now is None else now, times=T_IDXS,
                          integrate=LongitudinalMpc.integrate_jerk, validate=LongitudinalMpc._continuation_trajectory_error)


def test_real_native_donor_survives_failure_only_as_an_independent_candidate():
  planner = LongitudinalMpc()
  recorder = NativeProblemRecorder(planner.solver)
  planner.solver = recorder
  planner.set_cur_state(10.0, 0.0)
  planner.set_weights(prev_accel_constraint=False)
  for _ in range(12):
    planner.update(radar_state(), reference=MpcReference(10.0), reference_ns=EPOCH, now_ns=PRODUCER)
    assert planner.solution_status == 0
  donor = planner.snapshot_continuation_donor(EPOCH, PRODUCER, INSTANCE)
  assert donor is not None, planner.continuation_donor_reason
  saved = donor.states.copy()
  recorder.failure = 'solver_status'
  now = PRODUCER + 50_000_000
  planner.update(radar_state(), reference=MpcReference(10.0), reference_ns=EPOCH + 50_000_000, now_ns=now)
  assert planner.solution_status == planner.last_solver_status == planner.diagnostic_last_solver_return == 4
  writes = len(recorder.constraint_writes), len(recorder.cost_writes), planner.diagnostic_solve_calls
  result = planner.continuation_candidate(donor, now_ns=now)
  assert result.accepted, result.reason
  assert result.speed_tolerance > 0.0 and result.accel_tolerance > 0.0
  assert planner.solution_status == planner.last_solver_status == planner.diagnostic_last_solver_return == 4
  assert writes == (len(recorder.constraint_writes), len(recorder.cost_writes), planner.diagnostic_solve_calls)
  np.testing.assert_array_equal(planner.x_sol, 0.0)
  np.testing.assert_array_equal(planner.u_sol, 0.0)
  assert not planner.previous_solution_valid
  assert planner.snapshot_continuation_donor(EPOCH + 50_000_000, now, INSTANCE) is None
  np.testing.assert_array_equal(donor.states, saved)
  assert not result.states.flags.writeable and not result.jerks.flags.writeable
  planner.reset()
  assert planner.continuation_problem is None


def test_shift_uses_source_epoch_and_reintegrates_from_the_current_state():
  donor = mathematical_donor()
  current = replace(donor.problem, reference_ns=EPOCH + 37_000_000, consumed_ns=EPOCH + 65_000_000,
                    x0=frozen_array([0.0, 8.01, .28]))
  result = candidate(donor, current=current)
  assert result.accepted, result.reason
  np.testing.assert_array_equal(result.states[0], current.x0)
  expected = np.interp(T_IDXS + .037, T_IDXS, donor.states[:, 2])
  expected[-1] = 0.0
  np.testing.assert_allclose(result.states[1:, 2], expected[1:], atol=1e-14, rtol=0.0)
  assert result.tail_extension_s == pytest.approx(.037)
  assert np.max(np.abs(LongitudinalMpc.trajectory_dynamics_defects(result.states, result.jerks))) < QP_FEASIBILITY_TOL
  assert result.donor_source_ns == EPOCH and result.donor_producer_ns == PRODUCER
  assert result.evidence_expiry_ns == EPOCH + MAX_DONOR_SOURCE_AGE_NS
  assert result.donor_max_accel_deviation > 0.0


def test_current_measurement_deadline_and_geometry_are_frozen_before_failure_reset():
  donor = mathematical_donor()
  reference_ns, now = EPOCH + 50_000_000, PRODUCER + 50_000_000
  lead = object_lead(distance=50.0, speed=10.0, ego_speed=8.0, persistent=True, epoch=reference_ns)
  lead.evidence.positionMonoTime = EPOCH - 90_000_000
  planner = LongitudinalMpc()
  recorder = NativeProblemRecorder(planner.solver)
  recorder.failure = 'solver_status'
  planner.solver = recorder
  planner.set_cur_state(8.0, .3)
  planner.set_weights(prev_accel_constraint=False)
  planner.update(fused_state(display=(lead,)), reference=MpcReference(10.0), reference_ns=reference_ns, now_ns=now)
  assert planner.solution_status == 4
  problem = planner.continuation_problem
  assert problem.evidence_expiry_ns == EPOCH - 90_000_000 + MAX_MEASUREMENT_AGE_NS
  assert not problem.lead_trajectories[0].flags.writeable
  before = problem.lead_trajectories[0].copy()
  lead.dRel = 2.0
  lead.evidence.positionMonoTime = reference_ns
  result = planner.continuation_candidate(donor, now_ns=now)
  assert result.accepted, result.reason
  assert result.evidence_expiry_ns == EPOCH - 90_000_000 + MAX_MEASUREMENT_AGE_NS
  np.testing.assert_array_equal(problem.lead_trajectories[0], before)


@pytest.mark.parametrize('source,fusion_active', [
  ('legacy', None), ('legacy', False), ('vision', False), ('radar', False), ('vision', True), ('radar', True),
])
def test_continuation_retains_the_original_fusion_scope_without_changing_rest_geometry(source, fusion_active):
  reference_ns, now_ns = EPOCH + 50_000_000, PRODUCER + 50_000_000
  if source == 'legacy':
    state = radar_state(distance=2.0, speed=0.0)
  else:
    lead = object_lead(distance=2.0, speed=0.0, ego_speed=15.0, source=source, epoch=reference_ns)
    lead.evidence.positionMonoTime = EPOCH - 90_000_000
    state = fused_state(display=(lead,))
  if fusion_active is not None:
    state.fusionActive = fusion_active
  planner = LongitudinalMpc()
  recorder = NativeProblemRecorder(planner.solver)
  recorder.failure = 'solver_status'
  planner.solver = recorder
  planner.set_cur_state(15.0, 0.0)
  recovery_limits = []
  recorder.before_solve = lambda: recovery_limits.append(planner.rest_recovery_position_limit)
  assert planner.process_lead(state.leadOne)[0, 0] > state.leadOne.dRel
  planner.update(state, reference=MpcReference(20.0), reference_ns=reference_ns, now_ns=now_ns)
  assert planner.solution_status == 4 and planner.continuation_problem is not None
  # Ordinary rest scheduling still sees the actual measured front distance.
  assert recovery_limits == [2.0 - CRASH_DISTANCE]
  problem = planner.continuation_problem
  if fusion_active:
    assert problem.evidence_expiry_ns == EPOCH - 90_000_000 + MAX_MEASUREMENT_AGE_NS
    assert len(problem.lead_trajectories) == 1
    np.testing.assert_array_equal(problem.lead_trajectories[0][:, 0], 2.0)
  else:
    assert problem.evidence_expiry_ns == reference_ns + REFERENCE_MAX_AGE_NS
    assert problem.lead_trajectories == ()


@pytest.mark.parametrize('acceleration,jerk', [
  (TERMINAL_ACCEL_TOL + 1e-6, 0.0), (0.0, TERMINAL_JERK_TOL + 1e-6),
])
def test_unsupported_terminal_motion_is_not_extrapolated(acceleration, jerk):
  result = candidate(mathematical_donor(terminal_accel=acceleration, terminal_jerk=jerk))
  assert not result.accepted and result.reason == 'donor_terminal_not_constant_speed'


def test_nonzero_but_qualified_terminal_precision_is_reintegrated_not_claimed_as_old_support():
  result = candidate(mathematical_donor(terminal_accel=TERMINAL_ACCEL_TOL*.5))
  assert result.accepted, result.reason
  assert result.tail_extension_s > 0.0 and abs(result.states[-1, 2]) <= 1e-14


@pytest.mark.parametrize('change,reason', [
  ('regressed_source', 'continuation_clock_order'), ('different_now', 'continuation_clock_order'),
  ('old_source', 'continuation_evidence_expired'), ('current_expiry', 'continuation_evidence_expired'),
  ('stop', 'continuation_stop_required'),
])
def test_age_order_and_current_stop_cannot_be_renewed_by_a_new_candidate(change, reason):
  donor = mathematical_donor()
  current = replace(donor.problem, reference_ns=EPOCH + 50_000_000, consumed_ns=PRODUCER + 50_000_000)
  now = current.consumed_ns
  if change == 'regressed_source':
    current = replace(current, reference_ns=EPOCH-1)
  elif change == 'different_now':
    now += 1
  elif change == 'old_source':
    current = replace(current, reference_ns=EPOCH + 190_000_000, consumed_ns=EPOCH + MAX_DONOR_SOURCE_AGE_NS)
    now = current.consumed_ns
  elif change == 'current_expiry':
    current = replace(current, evidence_expiry_ns=now)
  else:
    current = replace(current, stop_required=True)
  result = candidate(donor, current=current, now=now)
  assert not result.accepted and result.reason == reason


@pytest.mark.parametrize('remaining_ns', [-1, 0, 1])
def test_new_current_measurements_cannot_renew_original_donor_evidence(remaining_ns):
  donor = mathematical_donor()
  now = PRODUCER + 50_000_000
  original_deadline = now + remaining_ns
  current = replace(donor.problem, reference_ns=EPOCH + 50_000_000, consumed_ns=now,
                    evidence_expiry_ns=now + 200_000_000)
  donor = replace(donor, problem=replace(donor.problem, evidence_expiry_ns=original_deadline))
  result = candidate(donor, current=current, now=now)
  assert result.evidence_expiry_ns == original_deadline
  assert result.accepted == (remaining_ns > 0)
  if remaining_ns <= 0:
    assert result.reason == 'continuation_evidence_expired'


def test_unknown_original_evidence_expiry_is_rejected():
  donor = mathematical_donor()
  donor = replace(donor, problem=replace(donor.problem, evidence_expiry_ns=0))
  assert candidate(donor).reason == 'invalid_continuation_clock'


@pytest.mark.parametrize('kind', ['ordinary_risk', 'lead_geometry', 'accel_bound', 'linear_row'])
def test_current_tighter_problem_rejects_a_formerly_qualified_trajectory(kind):
  donor = mathematical_donor()
  current = replace(donor.problem, reference_ns=EPOCH + 50_000_000, consumed_ns=PRODUCER + 50_000_000)
  if kind == 'ordinary_risk':
    parameters = current.parameters.copy()
    parameters[:, 2] = 1.0
    current = replace(current, parameters=frozen_array(parameters))
  elif kind == 'lead_geometry':
    # A newly measured formal front vehicle invalidates the donor independently
    # of the soft risk row and process_lead's numerical close-distance floor.
    trajectory = np.column_stack((np.full(N+1, 2.0), np.zeros(N+1)))
    current = replace(current, lead_trajectories=(frozen_array(trajectory),))
  elif kind == 'accel_bound':
    parameters = current.parameters.copy()
    parameters[:, 1] = -.2
    current = replace(current, parameters=frozen_array(parameters))
  else:
    rows = list(current.linear_constraints)
    C, D, lower, upper = (value.copy() for value in rows[0])
    C[1] = [1.0, 0.0, 0.0]
    upper[1] = -.1
    rows[0] = tuple(frozen_array(value) for value in (C, D, lower, upper))
    current = replace(current, linear_constraints=tuple(rows))
  result = candidate(donor, current=current)
  assert not result.accepted and result.reason.startswith('current_')


def test_captured_geometry_preserves_observed_distance_before_native_floor():
  planner = LongitudinalMpc()
  recorder = NativeProblemRecorder(planner.solver)
  recorder.failure = 'solver_status'
  planner.solver = recorder
  planner.set_cur_state(15.0, 0.0)
  lead = object_lead(distance=2.0, speed=0.0, ego_speed=15.0)
  assert planner.process_lead(lead)[0, 0] > lead.dRel
  planner.update(fused_state(display=(lead,)), reference=MpcReference(20.0), reference_ns=EPOCH, now_ns=PRODUCER)
  problem = planner.continuation_problem
  assert problem is not None and planner.solution_status == 4
  np.testing.assert_array_equal(problem.lead_trajectories[0][:, 0], lead.dRel)
  np.testing.assert_array_equal(problem.lead_trajectories[0][:, 1], 0.0)


@pytest.mark.parametrize('slot', [0, 1])
def test_each_formal_lead_keeps_its_original_measurement_deadline(slot):
  reference = EPOCH + 50_000_000
  now = PRODUCER + 50_000_000
  lead = object_lead(distance=100.0, speed=10.0, ego_speed=8.0, epoch=reference)
  lead.evidence.velocityMonoTime = EPOCH - 90_000_000
  state = fused_state()
  setattr(state, ('leadOne', 'leadTwo')[slot], lead)
  planner = LongitudinalMpc()
  recorder = NativeProblemRecorder(planner.solver)
  recorder.failure = 'solver_status'
  planner.solver = recorder
  planner.set_cur_state(8.0, 0.0)
  planner.update(state, reference=MpcReference(10.0), reference_ns=reference, now_ns=now)
  assert planner.solution_status == 4
  assert planner.continuation_problem.evidence_expiry_ns == EPOCH + 110_000_000
  assert len(planner.continuation_problem.lead_trajectories) == 1


def test_positive_nodes_do_not_hide_between_node_reverse_motion():
  accelerations = np.full(N+1, -.1 + 5.0*T_IDXS[1])
  accelerations[0] = -.1
  states, jerks = integrate_profile([0.0, .0009, -.1], accelerations)
  assert np.min(states[:, 1]) > 0.0
  assert LongitudinalMpc._continuation_trajectory_error(problem_for(states), states, jerks) == 'continuous_negative_velocity'


@pytest.mark.parametrize('status', [-1, -2, -3, -4, -5, -6, 0, 1, 2, 3])
def test_unapproved_native_or_input_status_never_receives_a_candidate(status):
  planner = LongitudinalMpc()
  planner.solution_status = status
  result = planner.continuation_candidate(mathematical_donor(), now_ns=PRODUCER + 50_000_000)
  assert not result.accepted and result.reason == 'unsupported_failure_status'


def test_reported_envelope_is_proved_with_one_common_jerk_for_all_corners():
  donor = mathematical_donor()
  current = problem_for(donor.states, source=EPOCH, consumed=PRODUCER + 10_000_000, margin=10.0)
  result = candidate(donor, current=current)
  assert result.accepted, result.reason
  assert (result.speed_tolerance, result.accel_tolerance) in STATE_ENVELOPE_LEVELS
  assert result.speed_tolerance < STATE_ENVELOPE_LEVELS[0][0]
  assert result.corner_checks >= 4
  for speed_sign, accel_sign in ((1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)):
    corner = current.x0.copy()
    corner[1] += speed_sign*result.speed_tolerance
    corner[2] += accel_sign*result.accel_tolerance
    states = np.empty_like(result.states)
    states[0] = corner
    for stage, duration in enumerate(np.diff(T_IDXS)):
      states[stage+1] = LongitudinalMpc.integrate_jerk(states[stage], result.jerks[stage], duration)
    assert LongitudinalMpc._continuation_trajectory_error(replace(current, x0=frozen_array(corner)), states, result.jerks) == ''
    # Verify the analytic pointwise error tube, without clipping its states.
    np.testing.assert_allclose(states[:, 0] - result.states[:, 0],
                               speed_sign*result.speed_tolerance*T_IDXS + .5*accel_sign*result.accel_tolerance*T_IDXS**2,
                               atol=1e-12, rtol=0.0)


def test_a_nominally_valid_candidate_without_any_nonzero_certified_envelope_is_rejected():
  donor = mathematical_donor()
  current = problem_for(donor.states, source=EPOCH, consumed=PRODUCER + 10_000_000, margin=0.0)
  assert LongitudinalMpc._continuation_trajectory_error(current, donor.states, donor.jerks) == ''
  result = candidate(donor, current=current)
  assert not result.accepted and result.reason.startswith('no_certified_state_envelope_')
  assert result.states is None and result.jerks is None
  assert result.speed_tolerance == result.accel_tolerance == 0.0


def test_signed_unknown_constraint_rows_cannot_borrow_the_monotone_envelope_proof():
  donor = mathematical_donor()
  current = replace(donor.problem, reference_ns=EPOCH + 50_000_000, consumed_ns=PRODUCER + 50_000_000)
  rows = list(current.linear_constraints)
  C, D, lower, upper = (value.copy() for value in rows[0])
  C[0, 0] = -1.0
  rows[0] = tuple(frozen_array(value) for value in (C, D, lower, upper))
  result = candidate(donor, current=replace(current, linear_constraints=tuple(rows)))
  assert not result.accepted and result.reason == 'current_unsupported_signed_linear_row'


def ordinary_accelerating_native(initial_accel=0.0):
  """The ordinary cruise reference that a slower final response can override."""
  planner = LongitudinalMpc()
  recorder = NativeProblemRecorder(planner.solver)
  planner.solver = recorder
  planner.set_cur_state(10.0, initial_accel)
  planner.set_weights(prev_accel_constraint=False)
  planner.update(radar_state(), reference=MpcReference(20.0, accel_max=1.0), reference_ns=EPOCH, now_ns=PRODUCER)
  assert planner.solution_status == planner.diagnostic_last_solver_return == 0
  return planner, recorder


def test_actual_cruise_cap_conditions_a_real_native_reference_and_survives_short_failure():
  planner, recorder = ordinary_accelerating_native()
  cap = .05
  assert np.interp(.55, T_IDXS, planner.a_solution) > cap
  assert abs(planner.a_solution[-1]) > TERMINAL_ACCEL_TOL
  raw_states, raw_jerks = planner.x_sol.copy(), planner.j_solution.copy()
  writes = len(recorder.constraint_writes), len(recorder.cost_writes), planner.diagnostic_solve_calls
  donor = planner.snapshot_continuation_donor(EPOCH, PRODUCER, INSTANCE, accel_ceiling=cap)
  assert donor is not None, planner.continuation_donor_reason
  assert donor.applied_accel_ceiling == cap
  assert donor.conditioning_max_state_delta[2] > 0.0 and donor.conditioning_max_jerk_delta > 0.0
  assert abs(donor.states[-1, 2] - cap) < 1e-12 and abs(donor.jerks[-1]) <= TERMINAL_JERK_TOL
  assert np.max(donor.states[1:, 2]) <= cap + QP_FEASIBILITY_TOL
  assert LongitudinalMpc._continuation_trajectory_error(donor.problem, donor.states, donor.jerks) == ''
  np.testing.assert_array_equal(planner.x_sol, raw_states)
  np.testing.assert_array_equal(planner.j_solution, raw_jerks)
  assert writes == (len(recorder.constraint_writes), len(recorder.cost_writes), planner.diagnostic_solve_calls)
  recorder.failure = 'solver_status'
  now = PRODUCER + 50_000_000
  planner.set_cur_state(10.0, 0.0)
  planner.update(radar_state(), reference=MpcReference(20.0, accel_max=1.0),
                 reference_ns=EPOCH + 50_000_000, now_ns=now)
  result = planner.continuation_candidate(donor, now_ns=now)
  assert result.accepted, result.reason
  assert 0.0 < np.interp(.55, T_IDXS, result.states[:, 2]) <= cap + QP_FEASIBILITY_TOL
  assert result.states[-1, 2] == pytest.approx(donor.states[-1, 2], abs=1e-12)
  assert result.tail_extension_s == pytest.approx(.05)
  assert planner.solution_status == planner.last_solver_status == 4
  np.testing.assert_array_equal(planner.x_sol, 0.0)
  assert planner.snapshot_continuation_donor(EPOCH + 50_000_000, now, INSTANCE, accel_ceiling=cap) is None


def test_conditioned_ceiling_does_not_clip_measured_a0_or_claim_the_initial_interval_is_below_cap():
  planner, _ = ordinary_accelerating_native(initial_accel=.4)
  cap = .05
  donor = planner.snapshot_continuation_donor(EPOCH, PRODUCER, INSTANCE, accel_ceiling=cap)
  assert donor is not None, planner.continuation_donor_reason
  np.testing.assert_array_equal(donor.states[0], planner.continuation_problem.x0)
  assert donor.states[0, 2] == .4 and np.max(donor.states[1:, 2]) <= cap + QP_FEASIBILITY_TOL
  assert np.interp(.01, T_IDXS, donor.states[:, 2]) > cap  # Initial measured-to-request transition.
  assert np.interp(.55, T_IDXS, donor.states[:, 2]) <= cap + QP_FEASIBILITY_TOL


@pytest.mark.parametrize('cap', [float('nan'), float('inf'), True, ACCEL_MIN - .1])
def test_conditioning_rejects_malformed_or_empty_acceleration_intersections(cap):
  planner, _ = ordinary_accelerating_native()
  assert planner.snapshot_continuation_donor(EPOCH, PRODUCER, INSTANCE, accel_ceiling=cap) is None
  assert planner.continuation_donor_reason in ('invalid_adopted_acceleration_ceiling_or_native_shape',
                                               'adopted_acceleration_intersection_empty')


def test_conditioning_rechecks_current_risk_instead_of_accepting_native_success_as_its_proof():
  planner, _ = ordinary_accelerating_native()
  parameters = planner.continuation_problem.parameters.copy()
  parameters[:, 2] = 1.0
  planner.continuation_problem = replace(planner.continuation_problem, parameters=frozen_array(parameters))
  assert planner.solution_status == planner.diagnostic_last_solver_return == 0
  assert planner.snapshot_continuation_donor(EPOCH, PRODUCER, INSTANCE, accel_ceiling=.05) is None
  assert planner.continuation_donor_reason == 'current_risk_distance_deficit'


def constant_acceleration_donor(speed, acceleration):
  states, jerks = integrate_profile([0.0, speed, acceleration], np.full(N+1, acceleration))
  return ContinuationDonor(EPOCH, PRODUCER, INSTANCE, frozen_array(states), frozen_array(jerks),
                           problem_for(states), applied_accel_ceiling=acceleration)


@pytest.mark.parametrize('speed,acceleration', [(8.0, ACCEL_MAX), (40.0, ACCEL_MIN)])
def test_conditioned_nominal_on_a_physical_face_does_not_clip_its_violating_corners(speed, acceleration):
  donor = constant_acceleration_donor(speed, acceleration)
  assert LongitudinalMpc._continuation_trajectory_error(donor.problem, donor.states, donor.jerks) == ''
  result = candidate(donor)
  assert not result.accepted and result.reason == 'no_certified_state_envelope_physical_state_bound'
  assert result.speed_tolerance == result.accel_tolerance == 0.0


def test_negative_conditioned_terminal_acceleration_cannot_extend_through_reverse_velocity():
  donor = constant_acceleration_donor(1.01, -.1)
  assert np.min(donor.states[:, 1]) > 0.0
  current = replace(donor.problem, reference_ns=EPOCH + 150_000_000, consumed_ns=PRODUCER + 150_000_000,
                    x0=frozen_array([0.0, .995, -.1]))
  result = candidate(donor, current=current)
  assert not result.accepted and result.reason == 'current_physical_state_bound'


@pytest.mark.parametrize('ceiling,expected', [(float('nan'), 'invalid_adopted_donor_ceiling'),
                                             (-.1, 'donor_exceeds_adopted_ceiling')])
def test_conditioned_metadata_must_match_the_stored_reference(ceiling, expected):
  donor = replace(mathematical_donor(), applied_accel_ceiling=ceiling)
  result = candidate(donor)
  assert not result.accepted and result.reason == expected


def test_nonzero_terminal_acceleration_still_requires_conditioned_origin_and_current_revalidation():
  donor = constant_acceleration_donor(8.0, .05)
  assert candidate(donor).accepted
  assert candidate(replace(donor, applied_accel_ceiling=None)).reason == 'donor_terminal_not_constant_speed'
  current = replace(donor.problem, reference_ns=EPOCH + 50_000_000, consumed_ns=PRODUCER + 50_000_000)
  parameters = current.parameters.copy()
  parameters[:, 1] = 0.0
  result = candidate(donor, current=replace(current, parameters=frozen_array(parameters)))
  assert not result.accepted and result.reason == 'current_physical_state_bound'
