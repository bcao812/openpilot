"""A carried numerical proposal never becomes a successful native solve."""
from dataclasses import replace

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.continuation import (
  MAX_DONOR_SOURCE_AGE_NS, STATE_ENVELOPE_LEVELS, TERMINAL_ACCEL_TOL, TERMINAL_JERK_TOL,
  ContinuationDonor, ContinuationProblem, build_candidate, frozen_array,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_STOP_DISTANCE, MPC_OCCUPANCY_UNSATISFIED, N, PARAM_DIM, QP_FEASIBILITY_TOL, REFERENCE_BOUND,
  RISK_STOP_DISTANCE, T_IDXS, LongitudinalMpc, MpcReference, get_safe_obstacle_distance,
)
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


def problem_for(states, *, source=EPOCH, consumed=PRODUCER, margin=100.0, projections=()):
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
                              frozen_array(lower), frozen_array(upper), rows, tuple(projections), False)


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
  planner.update(fused_state(lead), reference=MpcReference(10.0), reference_ns=reference_ns, now_ns=now)
  assert planner.solution_status == 4
  problem = planner.continuation_problem
  assert problem.evidence_expiry_ns == EPOCH + 110_000_000
  assert problem.projections[0].lead is None
  before = problem.projections[0].raw_trajectory.copy()
  lead.dRel = 2.0
  lead.evidence.positionMonoTime = reference_ns
  result = planner.continuation_candidate(donor, now_ns=now)
  assert result.accepted, result.reason
  assert result.evidence_expiry_ns == EPOCH + 110_000_000
  np.testing.assert_array_equal(problem.projections[0].raw_trajectory, before)


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
  ('stop', 'continuation_stop_or_passing'),
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
    current = replace(current, stop_or_passing=True)
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


@pytest.mark.parametrize('kind', ['ordinary_risk', 'persistent_risk', 'finite_geometry', 'accel_bound', 'linear_row'])
def test_current_tighter_problem_rejects_a_formerly_qualified_trajectory(kind):
  donor = mathematical_donor()
  current = replace(donor.problem, reference_ns=EPOCH + 50_000_000, consumed_ns=PRODUCER + 50_000_000)
  if kind in ('ordinary_risk', 'persistent_risk'):
    parameters = current.parameters.copy()
    parameters[:, 2] = 1.0
    current = replace(current, parameters=frozen_array(parameters))
    if kind == 'persistent_risk':
      projector = LongitudinalMpc()
      projector.set_cur_state(*current.x0[1:])
      lead = object_lead(distance=3.0, speed=0.0, ego_speed=8.0, persistent=True, epoch=current.reference_ns)
      current = replace(current, projections=(projector.project_fusion_lead(lead, slot=None,
                                                                            reference_ns=current.reference_ns),))
  elif kind == 'finite_geometry':
    projector = LongitudinalMpc()
    projector.set_cur_state(*current.x0[1:])
    lead = object_lead(distance=2.0, speed=0.0, ego_speed=8.0, start=.5, end=.6,
                       path='entering', persistent=False, epoch=current.reference_ns)
    geometry = projector.project_fusion_lead(lead, slot=None, reference_ns=current.reference_ns)
    current = replace(current, projections=(geometry,))
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


def test_positive_nodes_do_not_hide_between_node_reverse_motion():
  accelerations = np.full(N+1, -.1 + 5.0*T_IDXS[1])
  accelerations[0] = -.1
  states, jerks = integrate_profile([0.0, .0009, -.1], accelerations)
  assert np.min(states[:, 1]) > 0.0
  assert LongitudinalMpc._continuation_trajectory_error(problem_for(states), states, jerks) == 'continuous_negative_velocity'


@pytest.mark.parametrize('status', [-1, -2, -3, -4, -5, 0, 1, 2, 3])
def test_unapproved_native_or_input_status_never_receives_a_candidate(status):
  planner = LongitudinalMpc()
  planner.solution_status = status
  result = planner.continuation_candidate(mathematical_donor(), now_ns=PRODUCER + 50_000_000)
  assert not result.accepted and result.reason == 'unsupported_failure_status'


def test_minus_six_keeps_problem_but_does_not_turn_cached_motion_into_native_success():
  donor = mathematical_donor()
  planner = LongitudinalMpc()
  current = replace(donor.problem, reference_ns=EPOCH + 50_000_000, consumed_ns=PRODUCER + 50_000_000)
  planner.continuation_problem = current
  planner.fail(MPC_OCCUPANCY_UNSATISFIED)
  result = planner.continuation_candidate(donor, now_ns=current.consumed_ns)
  assert result.accepted, result.reason
  assert planner.solution_status == planner.last_solver_status == MPC_OCCUPANCY_UNSATISFIED
  np.testing.assert_array_equal(planner.x_sol, 0.0)
  assert planner.continuation_problem is current


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
