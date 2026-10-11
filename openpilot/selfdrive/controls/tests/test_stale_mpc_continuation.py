"""Expired measurements retain a bounded hypothesis, never native success."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.lead_qualification import (
  MAX_ESTIMATE_AGE_NS, MAX_MEASUREMENT_AGE_NS, MAX_MODEL_AGE_NS, qualified_lead,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.continuation import (
  MAX_DONOR_SOURCE_AGE_NS, MAX_STALE_PREDICTION_NS, STALE_MEASUREMENT_STATUS, build_candidate, frozen_array,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  MPC_INVALID_INPUT, MPC_STALE_MEASUREMENT, N, T_IDXS, LongitudinalMpc, MpcOutcome, MpcReference,
)
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_MAX_AGE_NS
from openpilot.selfdrive.controls.tests.test_mpc_continuation import (
  EPOCH, PRODUCER, constant_acceleration_donor, mathematical_donor,
)
from openpilot.selfdrive.controls.tests.test_mpc_problem_contract import NativeProblemRecorder
from openpilot.selfdrive.controls.tests.test_radar_fusion_mpc import fused_state, object_lead, rest_reference


MEASUREMENT = EPOCH - MAX_MEASUREMENT_AGE_NS + 10_000_000
FRESH_DEADLINE = MEASUREMENT + MAX_MEASUREMENT_AGE_NS
PREDICTION_DEADLINE = FRESH_DEADLINE + MAX_STALE_PREDICTION_NS
NOW = EPOCH + 15_000_000


def old_lead(**kwargs):
  lead = object_lead(distance=120.0, speed=20.0, ego_speed=20.0, epoch=EPOCH, **kwargs)
  lead.evidence.positionMonoTime = lead.evidence.velocityMonoTime = MEASUREMENT
  return lead


def recorded():
  mpc = LongitudinalMpc()
  native = NativeProblemRecorder(mpc.solver)
  native.failure = 'solver_status'
  mpc.solver = native
  mpc.set_cur_state(20.0, 0.0)
  mpc.set_weights(prev_accel_constraint=False)
  return mpc, native


def update(mpc, lead, *, now=NOW, reference_ns=EPOCH, reference=None, second=None, allow=True, fusion=True):
  state = fused_state(display=(lead,) if second is None else (lead, second))
  state.fusionActive = fusion
  mpc.update(state, reference=MpcReference(20.0) if reference is None else reference,
             reference_ns=reference_ns, now_ns=now, allow_stale_prediction=allow)


def prediction_pair(donor=None):
  donor = mathematical_donor() if donor is None else donor
  # Fresh at original publication, expired by the current candidate. The
  # ordinary and prediction expiries remain independent, fixed timestamps.
  donor = replace(donor, problem=replace(donor.problem, evidence_expiry_ns=EPOCH + 20_000_000,
                                         prediction_expiry_ns=EPOCH + 120_000_000))
  current = replace(donor.problem, reference_ns=EPOCH + 50_000_000, consumed_ns=PRODUCER + 50_000_000)
  return donor, current


def candidate(donor, current, *, allow=True):
  return build_candidate(donor, current, now_ns=current.consumed_ns, times=T_IDXS,
                         integrate=LongitudinalMpc.integrate_jerk, validate=LongitudinalMpc._continuation_trajectory_error,
                         allow_stale_prediction=allow)


@pytest.mark.parametrize('allow,fusion,expected', [
  (False, True, MPC_INVALID_INPUT), (None, True, MPC_INVALID_INPUT), (1, True, MPC_INVALID_INPUT),
  (True, False, MPC_INVALID_INPUT), (True, 'true', MPC_INVALID_INPUT), (True, True, MPC_STALE_MEASUREMENT),
])
@pytest.mark.parametrize('source', ['radar', 'fused'])
def test_stale_prediction_needs_explicit_fusion_opt_in_and_never_runs_native(allow, fusion, expected, source, monkeypatch):
  mpc, native = recorded()
  lead = old_lead(source=source)
  before = deepcopy(vars(lead)), deepcopy(vars(lead.evidence))
  monkeypatch.setattr(mpc, 'run', lambda: pytest.fail('expired measurement entered a normal native solve'))
  update(mpc, lead, allow=allow, fusion=fusion)
  assert mpc.solution_status == mpc.last_solver_status == expected
  assert mpc.diagnostic_solve_calls == 0 and not native.solved_costs
  assert not mpc.previous_solution_valid and mpc.outcome == MpcOutcome.INPUT_REJECTED
  assert vars(lead) == before[0] and vars(lead.evidence) == before[1]
  assert not qualified_lead(lead, 'brake', 0.0, now_ns=NOW)
  assert not qualified_lead(lead, 'stop', 0.0, now_ns=NOW)
  assert not qualified_lead(lead, 'departure', 0.0, now_ns=NOW)
  assert not qualified_lead(lead, 'fcw', 0.0, now_ns=NOW)
  if expected == MPC_STALE_MEASUREMENT:
    assert MPC_STALE_MEASUREMENT == STALE_MEASUREMENT_STATUS == -7
    assert mpc.failure_reason == 'stale_measurement'
    assert mpc.continuation_problem.evidence_expiry_ns == FRESH_DEADLINE
    assert mpc.continuation_problem.prediction_expiry_ns == PREDICTION_DEADLINE
    assert not mpc.continuation_problem.parameters.flags.writeable
    np.testing.assert_array_equal(mpc.x_sol, 0.0)
  else:
    assert mpc.continuation_problem is None


@pytest.mark.parametrize('offset,status,native_calls', [(0, 4, 1), (1, MPC_STALE_MEASUREMENT, 0)])
def test_fresh_230ms_boundary_and_prediction_are_distinct(offset, status, native_calls):
  mpc, native = recorded()
  lead = old_lead()
  now = FRESH_DEADLINE + offset
  assert qualified_lead(lead, 'brake', 0.0, now_ns=now) == (offset == 0)
  update(mpc, lead, now=now)
  assert mpc.solution_status == status and mpc.diagnostic_solve_calls == native_calls
  assert len(native.solved_costs) == native_calls
  assert mpc.continuation_problem.evidence_expiry_ns == FRESH_DEADLINE
  assert mpc.continuation_problem.prediction_expiry_ns == PREDICTION_DEADLINE


@pytest.mark.parametrize('remaining_ns', [1, 0, -1])
def test_prediction_deadline_is_absolute_and_exclusive(remaining_ns):
  mpc, _ = recorded()
  update(mpc, old_lead(), now=PREDICTION_DEADLINE - remaining_ns)
  assert mpc.diagnostic_solve_calls == 0
  if remaining_ns > 0:
    assert mpc.solution_status == MPC_STALE_MEASUREMENT
    assert mpc.continuation_problem.prediction_expiry_ns == PREDICTION_DEADLINE
  else:
    assert mpc.solution_status == MPC_INVALID_INPUT and mpc.continuation_problem is None


@pytest.mark.parametrize('defect', [
  'model_age', 'estimate_age', 'future_model', 'future_measurement', 'coasting', 'path', 'future_entry',
  'vision', 'position_invalid', 'uncertainty_invalid', 'identity_missing', 'backward', 'nonfinite', 'negative_tau',
  'duplicate', 'bad_second_lead', 'reference_missing', 'consumer_missing',
])
def test_other_defects_do_not_acquire_the_stale_measurement_status(defect):
  mpc, native = recorded()
  lead, second = old_lead(), None
  now, reference_ns = NOW, EPOCH
  evidence = lead.evidence
  if defect == 'model_age':
    evidence.modelMonoTime = NOW - MAX_MODEL_AGE_NS - 1
  elif defect == 'estimate_age':
    evidence.estimateMonoTime = NOW - MAX_ESTIMATE_AGE_NS - 1
  elif defect == 'future_model':
    evidence.modelMonoTime = NOW + 1
  elif defect == 'future_measurement':
    evidence.velocityMonoTime = NOW + 1
  elif defect == 'coasting':
    evidence.coasting = True
  elif defect == 'path':
    evidence.pathState = 'entering'
  elif defect == 'future_entry':
    evidence.conflictStart = .1
  elif defect == 'vision':
    evidence.source, evidence.visionIndex, lead.modelProb = 'vision', 0, .99
  elif defect == 'position_invalid':
    evidence.positionValid = False
  elif defect == 'uncertainty_invalid':
    evidence.positionStd = 0.0
  elif defect == 'identity_missing':
    evidence.sessionId = ''
  elif defect == 'backward':
    lead.vLead = -1.0
  elif defect == 'nonfinite':
    lead.dRel = np.nan
  elif defect == 'negative_tau':
    lead.aLeadTau = -1.0
  elif defect == 'duplicate':
    second = deepcopy(lead)
    second.dRel += 1.0
  elif defect == 'bad_second_lead':
    second = object_lead(source='vision', epoch=EPOCH)
    second.evidence.positionValid = False
  elif defect == 'reference_missing':
    reference_ns = None
  else:
    now = None
  update(mpc, lead, second=second, now=now, reference_ns=reference_ns)
  assert mpc.solution_status == MPC_INVALID_INPUT and mpc.continuation_problem is None
  assert mpc.diagnostic_solve_calls == 0 and not native.solved_costs


@pytest.mark.parametrize('limiter', ['model', 'estimate', 'source', 'vision'])
def test_prediction_never_extends_independent_evidence_or_reference_leases(limiter):
  mpc, _ = recorded()
  lead, second, reference_ns = old_lead(), None, EPOCH
  if limiter == 'model':
    lead.evidence.modelMonoTime = EPOCH - 150_000_000
    deadline = lead.evidence.modelMonoTime + MAX_MODEL_AGE_NS
  elif limiter == 'estimate':
    lead.evidence.estimateMonoTime = EPOCH - 180_000_000
    deadline = lead.evidence.estimateMonoTime + MAX_ESTIMATE_AGE_NS
  elif limiter == 'source':
    reference_ns = EPOCH - 200_000_000
    deadline = reference_ns + REFERENCE_MAX_AGE_NS
  else:
    second = object_lead(source='vision', epoch=EPOCH - 180_000_000)
    deadline = second.evidence.modelMonoTime + MAX_MODEL_AGE_NS
  update(mpc, lead, second=second, reference_ns=reference_ns)
  assert mpc.solution_status == MPC_STALE_MEASUREMENT
  assert mpc.continuation_problem.evidence_expiry_ns == FRESH_DEADLINE
  assert mpc.continuation_problem.prediction_expiry_ns == deadline < PREDICTION_DEADLINE


def test_prediction_preserves_real_geometry_without_the_native_distance_floor():
  mpc, _ = recorded()
  mpc.set_cur_state(30.0, 0.0)
  lead = old_lead()
  lead.dRel, lead.vLead, lead.aLeadK = 2.0, 0.0, 0.0
  assert mpc.process_lead(lead, reference_ns=EPOCH)[0, 0] > lead.dRel
  update(mpc, lead)
  problem = mpc.continuation_problem
  assert mpc.solution_status == MPC_STALE_MEASUREMENT
  trajectory = problem.lead_trajectories[0]
  unobserved = (NOW - MEASUREMENT) / 1e9
  upper_distance = lead.dRel - 3*lead.evidence.positionStd - 3*lead.evidence.velocityStd*unobserved - 5*unobserved**2
  assert trajectory[0, 0] <= upper_distance + 1e-12 < lead.dRel
  np.testing.assert_array_equal(trajectory[:, 1], 0.0)
  assert not trajectory.flags.writeable
  saved = trajectory.copy()
  lead.dRel = 1000.0
  lead.evidence.positionMonoTime = NOW
  np.testing.assert_array_equal(problem.lead_trajectories[0], saved)


def test_prediction_uncertainty_and_positive_lead_acceleration_cannot_move_obstacle_away():
  mpc, _ = recorded()
  lead = old_lead()
  ordinary = mpc._stale_prediction_trajectory(lead, reference_ns=EPOCH, now_ns=NOW)
  uncertain = deepcopy(lead)
  uncertain.evidence.positionStd *= 2
  uncertain.evidence.velocityStd *= 2
  uncertain.aLeadK = 3.0
  conservative = mpc._stale_prediction_trajectory(uncertain, reference_ns=EPOCH, now_ns=NOW)
  assert np.all(conservative <= ordinary + 1e-12)
  assert np.all(np.diff(conservative[:, 1]) <= 1e-12)
  assert conservative[0, 1] < lead.vLead


def test_an_independent_current_rest_goal_vetoes_the_prediction():
  mpc, native = recorded()
  update(mpc, old_lead(), reference=rest_reference(20.0, 6))
  assert mpc.solution_status == MPC_STALE_MEASUREMENT and mpc.continuation_problem.stop_required
  assert not native.solved_costs
  donor, _ = prediction_pair()
  result = mpc.continuation_candidate(donor, now_ns=NOW)
  assert not result.accepted and result.reason == 'continuation_stop_required'


def test_expired_original_deadline_requires_opt_in_and_conditioning_precedes_validation():
  donor, current = prediction_pair()
  saved = donor.states.copy()
  ordinary = candidate(donor, current, allow=False)
  assert not ordinary.accepted and ordinary.reason == 'continuation_evidence_expired'
  predicted = candidate(donor, current)
  assert predicted.accepted, predicted.reason
  assert predicted.reason == 'validated_stale_measurement_prediction'
  assert np.all(predicted.states[1:, 2] <= 1e-12) and predicted.states[0, 2] == current.x0[2]
  assert predicted.corner_checks >= 4 and predicted.speed_tolerance > 0 and predicted.accel_tolerance > 0
  assert LongitudinalMpc._continuation_trajectory_error(current, predicted.states, predicted.jerks) == ''
  np.testing.assert_array_equal(donor.states, saved)


def test_stale_conditioning_keeps_an_already_adopted_negative_acceleration_ceiling():
  donor, current = prediction_pair(constant_acceleration_donor(20.0, -.2))
  result = candidate(donor, current)
  assert result.accepted, result.reason
  assert np.all(result.states[1:, 2] <= -.2 + 1e-12)
  assert result.donor_max_braking_relaxation <= 1e-12


@pytest.mark.parametrize('side', ['donor', 'current'])
@pytest.mark.parametrize('constraint', ['risk', 'clearance', 'physical', 'linear', 'stop'])
def test_both_original_and_current_constraints_remain_mandatory(side, constraint):
  donor, current = prediction_pair()
  problem = donor.problem if side == 'donor' else current
  if constraint == 'risk':
    values = problem.parameters.copy()
    values[:, 2] = 1.0
    problem = replace(problem, parameters=frozen_array(values))
  elif constraint == 'clearance':
    geometry = np.column_stack((np.full(N+1, 2.0), np.zeros(N+1)))
    problem = replace(problem, lead_trajectories=(frozen_array(geometry),))
  elif constraint == 'physical':
    values = problem.parameters.copy()
    values[:, 1] = -.5
    problem = replace(problem, parameters=frozen_array(values))
  elif constraint == 'linear':
    rows = list(problem.linear_constraints)
    C, D, lower, upper = (value.copy() for value in rows[0])
    C[1] = [1.0, 0.0, 0.0]
    upper[1] = -.1
    rows[0] = tuple(frozen_array(value) for value in (C, D, lower, upper))
    problem = replace(problem, linear_constraints=tuple(rows))
  else:
    problem = replace(problem, stop_required=True)
  if side == 'donor':
    donor = replace(donor, problem=problem)
  else:
    current = problem
  result = candidate(donor, current)
  assert not result.accepted
  if constraint == 'stop':
    assert result.reason == 'continuation_stop_required'
  else:
    assert result.reason.startswith(side + '_')


def test_new_candidates_cannot_renew_original_prediction_or_donor_source_deadlines():
  donor, current = prediction_pair()
  for consumed in (PRODUCER + 50_000_000, PRODUCER + 60_000_000, donor.problem.prediction_expiry_ns):
    current = replace(current, reference_ns=consumed - 10_000_000, consumed_ns=consumed,
                      evidence_expiry_ns=EPOCH + 80_000_000, prediction_expiry_ns=EPOCH + 180_000_000)
    result = candidate(donor, current)
    assert result.evidence_expiry_ns == donor.problem.prediction_expiry_ns
    assert result.accepted == (consumed < donor.problem.prediction_expiry_ns)
  donor = replace(donor, problem=replace(donor.problem, evidence_expiry_ns=EPOCH + 160_000_000,
                                         prediction_expiry_ns=EPOCH + 260_000_000))
  current = replace(current, reference_ns=EPOCH + 190_000_000, consumed_ns=EPOCH + MAX_DONOR_SOURCE_AGE_NS,
                    evidence_expiry_ns=EPOCH + 170_000_000, prediction_expiry_ns=EPOCH + 270_000_000)
  result = candidate(donor, current)
  assert not result.accepted and result.reason == 'continuation_evidence_expired'
  assert result.evidence_expiry_ns == donor.source_ns + MAX_DONOR_SOURCE_AGE_NS


@pytest.mark.parametrize('defect', ['no_prediction', 'renewed_prediction', 'donor_already_expired', 'donor_clock_mismatch'])
def test_prediction_cannot_retroactively_authorize_an_invalid_original_donor(defect):
  donor, current = prediction_pair()
  problem = donor.problem
  if defect == 'no_prediction':
    problem = replace(problem, prediction_expiry_ns=0)
  elif defect == 'renewed_prediction':
    problem = replace(problem, prediction_expiry_ns=problem.evidence_expiry_ns + MAX_STALE_PREDICTION_NS + 1)
  elif defect == 'donor_already_expired':
    problem = replace(problem, evidence_expiry_ns=donor.producer_ns - 1)
  else:
    problem = replace(problem, consumed_ns=donor.producer_ns - 1)
  result = candidate(replace(donor, problem=problem), current)
  assert not result.accepted
