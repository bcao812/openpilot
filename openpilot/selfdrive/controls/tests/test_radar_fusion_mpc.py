"""Only the two formal leads can affect the ordinary native MPC and FCW."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan
from openpilot.selfdrive.controls.lib.lead_qualification import qualified_lead
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_STOP_DISTANCE, CRASH_DISTANCE, FCW_IDXS, MPC_INVALID_INPUT, N,
  RISK_STOP_DISTANCE, STOP_DISTANCE, T_IDXS,
  LongitudinalMpc, LongitudinalPlanSource, MpcReference,
)
from openpilot.selfdrive.controls.lib.model_reference import ModelReferenceEvidence, ModelSource, MovingPreference, RestGoal
from openpilot.selfdrive.controls.tests.test_mpc_problem_contract import NativeProblemRecorder
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state


EPOCH_NS = 10_000_000_000


def object_lead(track=1, *, distance=30.0, speed=0.0, ego_speed=10.0, acceleration=0.0,
                source='radar', start=0.0, end=10.0, path='inPath', persistent=True, epoch=EPOCH_NS, physical_window=None):
  evidence = SimpleNamespace(
    version=1, source=source, sessionId='radar-test' if source != 'vision' else '', trackId=track,
    generation=0, positionMonoTime=epoch, velocityMonoTime=epoch, estimateMonoTime=epoch,
    modelMonoTime=epoch, positionStd=0.25, velocityStd=0.25, positionValid=True, velocityValid=True,
    accelerationValid=True, followQualified=persistent, brakeQualified=True, stopQualified=persistent,
    departureQualified=persistent, fcwQualified=True, coasting=False, pathState=path,
    pathOffset=0.0, conflictStart=start, conflictEnd=end, lateralSpeed=0.0 if persistent else 3.0,
    pathSource='model', visionIndex=0 if source == 'vision' else -1,
    physicalConflictValid=physical_window is not None,
    physicalConflictStart=0.0 if physical_window is None else physical_window[0],
    physicalConflictEnd=0.0 if physical_window is None else physical_window[1],
    followGeometryMonoTime=epoch if persistent and source in ('radar', 'fused') else 0,
  )
  return SimpleNamespace(present=True, dRel=distance, yRel=0.0, vRel=speed-ego_speed, vLead=speed,
                         vLeadK=speed, aLeadK=acceleration, aLeadTau=1.5,
                         modelProb=0.99 if source == 'vision' else 0.0, evidence=evidence)


def fused_state(*objects, display=()):
  state = radar_state()
  state.fusionActive = True
  state.fusionHealthy = True
  state.fusionDegraded = False
  state.radarObjects = list(objects)
  for slot, lead in zip(('leadOne', 'leadTwo'), display, strict=False):
    setattr(state, slot, lead)
  return state


def solve(objects, *, speed=10.0, acceleration=0.0, reference=None, iterations=12, now_ns=EPOCH_NS, reference_ns=EPOCH_NS):
  planner = LongitudinalMpc()
  planner.diagnostic_obstacles_enabled = True
  planner.set_cur_state(speed, acceleration)
  planner.set_weights(prev_accel_constraint=False)
  reference = MpcReference(20.0) if reference is None else reference
  for _ in range(iterations):
    planner.update(objects, reference=reference, now_ns=now_ns, reference_ns=reference_ns)
    assert planner.solution_status == planner.last_solver_status == 0
  assert planner.diagnostic_solve_calls == iterations
  return planner


def action(planner):
  return float(get_accel_from_plan(planner.v_solution, planner.a_solution, T_IDXS, action_t=0.55))


def advance(lead, stamp):
  evidence = lead.evidence
  evidence.positionMonoTime = evidence.velocityMonoTime = evidence.estimateMonoTime = evidence.modelMonoTime = stamp
  evidence.followGeometryMonoTime = stamp if not evidence.coasting and evidence.followQualified and evidence.source in ('radar', 'fused') else 0


def coast_from(lead, stamp):
  coast = deepcopy(lead)
  coast.dRel += coast.vRel * (stamp - coast.evidence.estimateMonoTime) / 1e9
  coast.aLeadK, coast.aLeadTau = 0.0, 0.3
  evidence = coast.evidence
  # This helper starts from the producer's latest accepted fresh object,
  # which may be newer than the last fresh publication the MPC consumed.
  evidence.followGeometryMonoTime = (lead.evidence.estimateMonoTime
                                    if lead.evidence.followQualified and not lead.evidence.coasting
                                    and lead.evidence.source in ('radar', 'fused') else 0)
  evidence.source, evidence.coasting, evidence.accelerationValid = 'predicted', True, False
  evidence.estimateMonoTime = evidence.modelMonoTime = stamp
  evidence.followQualified = evidence.stopQualified = evidence.departureQualified = evidence.fcwQualified = False
  evidence.physicalConflictValid = False
  # The actual producer retains HORIZON for an object without a known exit.
  # It does not manufacture an observed cutout solely because CAN was lost.
  return coast


def rest_reference(speed, node):
  end = float(T_IDXS[node])
  velocity = np.maximum(0.0, speed * (1.0 - T_IDXS / end))
  position = np.cumsum(np.diff(T_IDXS, prepend=0.0) * velocity)
  acceleration = np.where(velocity > 0.0, -speed / end, 0.0)
  preference = MovingPreference(position, velocity, acceleration, np.ones(N+1, dtype=bool))
  source = ModelSource(EPOCH_NS, EPOCH_NS, EPOCH_NS, proxy_age_s=0.0)
  rest = RestGoal((0, node), (0.0, end), EPOCH_NS + round(end * 1e9), end, node, end, None)
  return MpcReference(20.0, typed_model=ModelReferenceEvidence(source, preference, (rest,)))


@pytest.mark.parametrize('lead_speed', [0.0, 4.0])
def test_formal_radar_without_visual_confirmation_retains_braking(lead_speed):
  lead = object_lead(distance=30.0, speed=lead_speed, ego_speed=15.0)
  free = solve(fused_state(), speed=15.0)
  braking = solve(fused_state(display=(lead,)), speed=15.0)
  assert lead.modelProb == 0.0
  assert action(braking) < -0.05
  assert action(braking) < action(free) - 0.1
  assert braking.source_object_identity == ('radar-test', 1)
  assert ('radar-test', 1) in braking.object_influence_identities
  assert braking.diagnostic_last_obstacle_summary['objectCount'] == 1


@pytest.mark.parametrize('speed,lead_speed,distance', [(10.0, 8.0, 35.0), (5.0, 0.0, 18.0), (0.0, 0.0, 6.0)])
def test_same_measurements_retain_the_ordinary_lead_problem(speed, lead_speed, distance):
  lead = object_lead(distance=distance, speed=lead_speed, ego_speed=speed)
  fusion = solve(fused_state(display=(lead,)), speed=speed)
  legacy = solve(radar_state(distance=distance, speed=lead_speed), speed=speed)
  np.testing.assert_array_equal(fusion.params, legacy.params)
  np.testing.assert_allclose(fusion.x_sol, legacy.x_sol, atol=1e-9, rtol=0.0)


def test_second_formal_lead_still_brakes_without_its_own_fcw_path():
  distant = object_lead(1, distance=100.0, speed=20.0, ego_speed=15.0)
  near = object_lead(2, distance=25.0, speed=1.0, ego_speed=15.0)
  before = solve(fused_state(display=(distant,)), speed=15.0)
  after = solve(fused_state(display=(distant, near)), speed=15.0)
  assert action(after) < action(before) - 0.1
  assert after.source == LongitudinalPlanSource.lead1
  assert after.source_object_identity == ('radar-test', 2)
  assert after.crash_cnt == 0 and after.fcw_object_identity is None


@pytest.mark.parametrize('path,persistent', [('entering', False), ('inPath', False), ('inPath', True), ('edge', False)])
def test_extra_objects_cannot_change_the_problem_braking_or_fcw(path, persistent):
  distant = [object_lead(index, distance=distance, speed=20.0, ego_speed=15.0)
             for index, distance in ((1, 90.0), (2, 120.0))]
  near = object_lead(3, distance=.1, speed=-5.0, ego_speed=15.0,
                     path=path, start=.2 if path == 'entering' else 0.0, end=.8,
                     persistent=persistent, physical_window=(.2, .5))
  near.evidence.conflictSupportVersion, near.evidence.conflictSupportedEnd = 1, .5
  before = solve(fused_state(display=distant), speed=15.0)
  after = solve(fused_state(*distant, near, display=distant), speed=15.0)
  np.testing.assert_array_equal(after.params, before.params)
  np.testing.assert_array_equal(after.x_sol, before.x_sol)
  assert after.source_object_identity == before.source_object_identity
  assert after.object_influence_identities == before.object_influence_identities
  assert after.crash_cnt == before.crash_cnt == 0
  assert after.diagnostic_last_obstacle_summary['objectCount'] == 2


def test_invalid_ignored_object_cannot_fail_an_empty_ordinary_problem():
  extra = object_lead(distance=float('nan'), persistent=False)
  extra.evidence.positionMonoTime = 0
  extra.evidence.modelMonoTime = EPOCH_NS + 1
  baseline = solve(fused_state())
  ignored = solve(fused_state(extra))
  np.testing.assert_array_equal(ignored.params, baseline.params)
  np.testing.assert_array_equal(ignored.x_sol, baseline.x_sol)
  assert ignored.object_influence_identities == () and ignored.crash_cnt == 0


@pytest.mark.parametrize('fusion_active', [False, True])
def test_raw_objects_have_no_authority_when_neither_formal_lead_is_present(fusion_active):
  baseline = fused_state()
  state = fused_state(object_lead(distance=.01, speed=-10.0, end=.4, persistent=False))
  baseline.fusionActive = state.fusionActive = fusion_active
  before, after = solve(baseline), solve(state)
  np.testing.assert_array_equal(before.params, after.params)
  np.testing.assert_array_equal(before.x_sol, after.x_sol)
  assert before.source == after.source and before.crash_cnt == after.crash_cnt == 0


@pytest.mark.parametrize('source', ['radar', 'fused', 'vision'])
def test_fcw_uses_formal_lead_one_and_its_source_qualification(source):
  lead = object_lead(distance=2.0, speed=0.0, ego_speed=15.0, source=source)
  planner = solve(fused_state(display=(lead,)), speed=15.0)
  assert np.any(planner.process_lead(lead)[FCW_IDXS, 0] - planner.x_sol[FCW_IDXS, 0] < CRASH_DISTANCE)
  assert planner.crash_cnt > 0
  if source != 'vision':
    assert lead.modelProb == 0.0 and planner.fcw_object_identity == ('radar-test', 1)


@pytest.mark.parametrize('source', ['radar', 'vision'])
def test_fcw_cannot_borrow_visual_probability_or_a_different_purpose(source):
  lead = object_lead(distance=2.0, speed=0.0, ego_speed=15.0, source=source)
  if source == 'radar':
    lead.modelProb = 1.0
    lead.evidence.fcwQualified = False
  else:
    lead.modelProb = .8
  planner = solve(fused_state(display=(lead,)), speed=15.0)
  assert planner.crash_cnt == 0 and planner.fcw_object_identity is None
  assert action(planner) < 0.0


@pytest.mark.parametrize('source', ['radar', 'fused'])
def test_radar_fcw_counts_new_estimates_of_one_formal_front_vehicle(source):
  lead = object_lead(distance=2.0, speed=0.0, ego_speed=15.0, source=source)
  planner = solve(fused_state(display=(lead,)), speed=15.0)
  assert planner.crash_cnt == 1  # Twelve reads of one estimate are one confirmation.
  for index in (1, 2):
    stamp = EPOCH_NS + index * 50_000_000
    advance(lead, stamp)
    planner.update(fused_state(display=(lead,)), reference=MpcReference(20.0), now_ns=stamp, reference_ns=stamp)
    assert planner.solution_status == 0 and planner.crash_cnt == index + 1
  assert planner.crash_cnt == 3 and planner.fcw_object_identity == ('radar-test', 1)


@pytest.mark.parametrize('interruption', ['different_track', 'regressed_epoch', 'long_gap', 'missing', 'unqualified', 'failure'])
def test_formal_radar_fcw_cannot_inherit_unrelated_or_revoked_confirmation(interruption):
  lead = object_lead(distance=2.0, speed=0.0, ego_speed=15.0)
  planner = solve(fused_state(display=(lead,)), speed=15.0)
  for offset in (50_000_000, 100_000_000):
    stamp = EPOCH_NS + offset
    advance(lead, stamp)
    planner.update(fused_state(display=(lead,)), reference=MpcReference(20.0), now_ns=stamp, reference_ns=stamp)
  assert planner.crash_cnt == 3
  stamp = EPOCH_NS + 150_000_000
  if interruption == 'different_track':
    lead.evidence.trackId = 2
  elif interruption == 'regressed_epoch':
    stamp = EPOCH_NS + 50_000_000
  elif interruption == 'long_gap':
    stamp = EPOCH_NS + 400_000_000
  elif interruption == 'unqualified':
    lead.evidence.fcwQualified = False
  elif interruption == 'failure':
    recorder = NativeProblemRecorder(planner.solver)
    recorder.failure = 'solver_status'
    planner.solver = recorder
  advance(lead, stamp)
  state = fused_state() if interruption == 'missing' else fused_state(display=(lead,))
  planner.update(state, reference=MpcReference(20.0), now_ns=stamp, reference_ns=stamp)
  expected = 1 if interruption in ('different_track', 'regressed_epoch', 'long_gap') else 0
  assert planner.crash_cnt == expected
  if expected == 0:
    assert planner.fcw_object_identity is None and planner._radar_fcw_epoch is None
  if interruption == 'failure':
    assert planner.solution_status == 4


def test_known_formal_lead_coast_keeps_braking_but_no_fcw_or_stop_authority():
  fresh = object_lead(distance=30.0, speed=2.0, ego_speed=15.0)
  now = EPOCH_NS + 50_000_000
  coast = coast_from(fresh, now)
  free = solve(fused_state(), speed=15.0, now_ns=now, reference_ns=now)
  planner = solve(fused_state(display=(coast,)), speed=15.0, now_ns=now, reference_ns=now)
  assert qualified_lead(coast, 'brake', 0.0, now_ns=now)
  assert not any(qualified_lead(coast, purpose, 0.0, now_ns=now) for purpose in ('follow', 'stop', 'departure', 'fcw'))
  assert action(planner) < action(free) - .1
  assert planner.crash_cnt == 0 and planner.rest_recovery_position_limit is None
  assert planner.source_object_identity == ('radar-test', 1)


@pytest.mark.parametrize('defect', ['position_expired', 'future_model', 'velocity_invalid', 'nan', 'negative_tau', 'wrong_source'])
def test_invalid_formal_lead_rejects_before_native_solving(defect):
  lead = object_lead(distance=20.0, speed=3.0)
  if defect == 'position_expired':
    lead.evidence.positionMonoTime -= 400_000_000
  elif defect == 'future_model':
    lead.evidence.modelMonoTime += 1
  elif defect == 'velocity_invalid':
    lead.evidence.velocityValid = False
  elif defect == 'nan':
    lead.vLead = float('nan')
  elif defect == 'negative_tau':
    lead.aLeadTau = -1.0
  else:
    lead.evidence.source = 'unknown'
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner.update(fused_state(display=(lead,)), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status == MPC_INVALID_INPUT and planner.diagnostic_solve_calls == 0
  assert not planner.previous_solution_valid and planner.crash_cnt == 0
  assert planner.object_influence_identities == ()


@pytest.mark.parametrize('source', ['radar', 'vision'])
@pytest.mark.parametrize('field', ['distance', 'clock', 'fcw'])
def test_two_formal_views_of_one_identity_must_agree(source, field):
  lead = object_lead(distance=20.0, speed=3.0, source=source)
  other = deepcopy(lead)
  if field == 'distance':
    other.dRel += 1.0
  elif field == 'clock':
    other.evidence.positionMonoTime -= 1
  else:
    other.evidence.fcwQualified = False
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner.update(fused_state(display=(lead, other)), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status == MPC_INVALID_INPUT and planner.diagnostic_solve_calls == 0
  assert planner.crash_cnt == 0 and planner.object_influence_identities == ()


def test_qualified_formal_radar_keeps_the_compact_stop_contract():
  lead = object_lead(distance=3.0, ego_speed=0.0)
  reference = MpcReference(None, comfort_stop_distance=3.0, lead_risk_stop_distances=(4.0, 6.0))
  planner = solve(fused_state(display=(lead,)), speed=0.0, reference=reference)
  np.testing.assert_array_equal(planner.params[:, COMFORT_STOP_DISTANCE], 3.0)
  np.testing.assert_array_equal(planner.params[:, RISK_STOP_DISTANCE], 4.0)
  assert planner.rest_recovery_position_limit == pytest.approx(3.0 - CRASH_DISTANCE)


def test_compact_gap_is_not_given_to_an_unqualified_coasting_lead():
  now = EPOCH_NS + 50_000_000
  lead = coast_from(object_lead(distance=6.0, ego_speed=0.0), now)
  reference = MpcReference(None, comfort_stop_distance=3.0, lead_risk_stop_distances=(4.0, 6.0))
  planner = solve(fused_state(display=(lead,)), speed=0.0, reference=reference, now_ns=now, reference_ns=now)
  np.testing.assert_array_equal(planner.params[:, COMFORT_STOP_DISTANCE], STOP_DISTANCE)
  np.testing.assert_array_equal(planner.params[:, RISK_STOP_DISTANCE], STOP_DISTANCE)


def test_stopped_formal_lead_cannot_be_extrapolated_as_reversing():
  lead = object_lead(distance=8.0, speed=0.0, acceleration=-2.5, ego_speed=0.0)
  planner = solve(fused_state(display=(lead,)), speed=0.0)
  trajectory = planner.process_lead(lead)
  np.testing.assert_array_equal(trajectory[:, 0], lead.dRel)
  np.testing.assert_array_equal(trajectory[:, 1], 0.0)
  assert planner.crash_cnt == 0


@pytest.mark.parametrize('elapsed', [-.05, 0.0, .15])
def test_formal_measurements_align_to_ego_epoch_without_changing_source_identity(elapsed):
  lead = object_lead(distance=40.0, speed=8.0, acceleration=-1.0, ego_speed=10.0)
  original = deepcopy(vars(lead))
  reference_ns = EPOCH_NS + round(elapsed * 1e9)
  now_ns = max(EPOCH_NS, reference_ns)
  planner = solve(fused_state(display=(lead,)), speed=10.0, acceleration=.4,
                  now_ns=now_ns, reference_ns=reference_ns)
  decayed = -np.exp(-lead.aLeadTau * elapsed**2 / 2.0)
  expected_distance = (40.0 + 8.0*elapsed + (-2.0 + decayed)*elapsed**2/6.0
                       - (10.0*elapsed - .2*elapsed**2))
  expected_speed = max(0.0, 8.0 + .5*(-1.0 + decayed)*elapsed)
  trajectory, = planner.continuation_problem.lead_trajectories
  assert trajectory[0, 0] == pytest.approx(expected_distance)
  assert trajectory[0, 1] == pytest.approx(expected_speed)
  np.testing.assert_allclose(trajectory[:, 1], np.maximum(0.0, expected_speed + np.cumsum(
    np.diff(T_IDXS, prepend=0.0) * -np.exp(-lead.aLeadTau*(T_IDXS + elapsed)**2/2.0))), atol=1e-12, rtol=0.0)
  assert vars(lead) == original
  assert planner.diagnostic_last_obstacle_summary['objectCount'] == 1
  assert planner.continuation_problem.evidence_expiry_ns <= EPOCH_NS + 200_000_000


@pytest.mark.parametrize('elapsed', [-.250000001, .350000001])
def test_formal_lead_alignment_cannot_extrapolate_beyond_existing_clock_bounds(elapsed):
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  lead = object_lead(distance=30.0, speed=5.0)
  with pytest.raises(ValueError, match='cannot align'):
    planner.process_lead(lead, reference_ns=EPOCH_NS + round(elapsed * 1e9))


def test_solver_failure_with_extra_objects_never_grants_new_braking_authority():
  planner = LongitudinalMpc()
  recorder = NativeProblemRecorder(planner.solver)
  recorder.failure = 'solver_status'
  planner.solver = recorder
  planner.set_cur_state(10.0, 0.0)
  extra = object_lead(distance=.1, end=.2, persistent=False, physical_window=(0.0, .1))
  planner.update(fused_state(extra), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status == 4 and not planner.previous_solution_valid
  assert planner.crash_cnt == 0 and planner.object_influence_identities == ()
  assert planner.rest_goal_braking_accel is None
