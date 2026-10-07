"""Active fusion through the real single native MPC.

These fixtures test formal object geometry and requested acceleration, not the
road object's classification or delivered vehicle response. Independent arm
solves below are a fixed comparison experiment, never production work.
"""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan
from openpilot.selfdrive.controls.lib.lead_qualification import MAX_CONTROL_OBJECTS
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_STOP_DISTANCE, CRASH_DISTANCE, MODEL_V_REF, MPC_INVALID_INPUT, MPC_OCCUPANCY_UNSATISFIED, N,
  OCCUPANCY_POSITION_REINTEGRATION_LIMIT, QP_FEASIBILITY_TOL, REFERENCE_BOUND, REST_STATE_TOL,
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


def rest_reference(speed, node):
  end = float(T_IDXS[node])
  velocity = np.maximum(0.0, speed * (1.0 - T_IDXS / end))
  position = np.cumsum(np.diff(T_IDXS, prepend=0.0) * velocity)
  acceleration = np.where(velocity > 0.0, -speed / end, 0.0)
  preference = MovingPreference(position, velocity, acceleration, np.ones(N+1, dtype=bool))
  source = ModelSource(EPOCH_NS, EPOCH_NS, EPOCH_NS, proxy_age_s=0.0)
  rest = RestGoal((0, node), (0.0, end), EPOCH_NS + round(end * 1e9), end, node, end, None)
  return MpcReference(20.0, typed_model=ModelReferenceEvidence(source, preference, (rest,)))


def stationary_window_clearance(planner, distance, start, end):
  """Independent exact cubic-position check over a fixed occupancy interval."""
  minimum = float('inf')
  for stage, jerk in enumerate(planner.j_solution):
    lo, hi = max(start, T_IDXS[stage]) - T_IDXS[stage], min(end, T_IDXS[stage+1]) - T_IDXS[stage]
    if hi < lo:
      continue
    position, velocity, acceleration = planner.x_sol[stage]
    times = [lo, hi]
    # Zero ego velocity locates every interior position extremum. Evaluate
    # the polynomial directly instead of using the production FCW checker.
    if abs(jerk) < 1e-12:
      if abs(acceleration) > 1e-12:
        times.append(-velocity / acceleration)
    else:
      discriminant = acceleration**2 - 2.0 * jerk * velocity
      if discriminant >= 0.0:
        root = np.sqrt(discriminant)
        times.extend(((-acceleration - root) / jerk, (-acceleration + root) / jerk))
    for elapsed in times:
      if lo <= elapsed <= hi:
        ego_position = position + velocity * elapsed + acceleration * elapsed**2 / 2.0 + jerk * elapsed**3 / 6.0
        minimum = min(minimum, float(distance - ego_position))
  return minimum


@pytest.mark.parametrize('lead_speed', [0.0, 4.0])
def test_new_radar_without_any_visual_confirmation_requests_braking(lead_speed):
  lead = object_lead(distance=30.0, speed=lead_speed, ego_speed=15.0)
  free = solve(fused_state(), speed=15.0)
  braking = solve(fused_state(lead, display=(lead,)), speed=15.0)
  assert lead.modelProb == 0.0
  assert action(braking) < -0.05
  assert action(braking) < action(free) - 0.1
  assert braking.source_object_identity == ('radar-test', 1)
  assert braking.diagnostic_last_obstacle_summary['objectCount'] == 1


def test_third_object_can_brake_while_both_distant_leads_remain_present():
  distant = [object_lead(index, distance=distance, speed=20.0, ego_speed=15.0)
             for index, distance in ((1, 90.0), (2, 120.0))]
  near = object_lead(3, distance=25.0, speed=1.0, ego_speed=15.0)
  before = solve(fused_state(*distant, display=distant), speed=15.0)
  after = solve(fused_state(*distant, near, display=distant), speed=15.0)
  assert action(after) < -0.05
  assert action(after) < action(before) - 0.1
  assert after.diagnostic_last_obstacle_summary['objectCount'] == 3
  assert ('radar-test', 3) in after.diagnostic_last_obstacle_summary['objectIdentities']
  assert after.source == LongitudinalPlanSource.lead0
  assert after.source_object_identity == ('radar-test', 3)


@pytest.mark.parametrize('speed,lead_speed,distance', [(10.0, 8.0, 35.0), (5.0, 0.0, 18.0), (0.0, 0.0, 6.0)])
def test_same_aligned_kinematics_retain_the_ordinary_lead_problem(speed, lead_speed, distance):
  lead = object_lead(distance=distance, speed=lead_speed, ego_speed=speed)
  fusion = solve(fused_state(lead, display=(lead,)), speed=speed)
  legacy = solve(radar_state(distance=distance, speed=lead_speed), speed=speed)
  np.testing.assert_array_equal(fusion.params, legacy.params)
  np.testing.assert_allclose(fusion.x_sol, legacy.x_sol, atol=1e-9, rtol=0.0)


def test_fusion_is_ignored_when_its_mode_is_not_active():
  baseline = radar_state(distance=60.0, speed=15.0)
  shadow = deepcopy(baseline)
  shadow.fusionActive = False
  shadow.radarObjects = [object_lead(distance=5.0)]
  before, after = solve(baseline), solve(shadow)
  np.testing.assert_array_equal(before.params, after.params)
  np.testing.assert_array_equal(before.x_sol, after.x_sol)
  assert before.source == after.source and before.crash_cnt == after.crash_cnt


def test_short_crossing_brakes_and_releases_without_a_permanent_rest_limit():
  crossing = object_lead(distance=20.0, start=1.2, end=2.6, path='entering', persistent=False)
  planner = solve(fused_state(crossing))
  free = solve(fused_state())
  stationary = solve(fused_state(object_lead(distance=20.0)))
  assert action(planner) < action(free) - 0.05
  assert action(planner) < 0.0
  assert planner.rest_recovery_position_limit is None
  assert planner.diagnostic_last_obstacle_summary['sources'][-1] == 'cruise'
  projection = planner._fusion_projections[0]
  cover_end = T_IDXS[np.searchsorted(T_IDXS, 2.6)]
  assert projection.conflict_end == 2.6
  assert projection.occupied[T_IDXS == cover_end][0]
  assert not np.any(projection.occupied[T_IDXS > cover_end])
  assert planner.v_solution[-1] > stationary.v_solution[-1] + 1.0


@pytest.mark.parametrize('distance,start,end,speed', [(40.0, 0.1, 0.5, 15.0), (5.0, 4.0, 6.0, 20.0)])
def test_crossing_without_reachable_time_overlap_does_not_create_a_stop(distance, start, end, speed):
  crossing = object_lead(distance=distance, ego_speed=speed, start=start, end=end, path='entering', persistent=False)
  planner = solve(fused_state(crossing), speed=speed)
  free = solve(fused_state(), speed=speed)
  assert not np.any(planner._fusion_projections[0].occupied)
  np.testing.assert_array_equal(planner.params, free.params)
  np.testing.assert_allclose(planner.x_sol, free.x_sol, atol=1e-9, rtol=0.0)
  assert planner.crash_cnt == 0 and planner.rest_recovery_position_limit is None


def test_between_node_occupancy_is_kept_and_fcw_checks_its_real_time(record_property):
  crossing = object_lead(distance=8.0, start=0.7, end=0.85, path='entering', persistent=False)
  assert not np.any((T_IDXS >= 0.7) & (T_IDXS <= 0.85))
  planner = solve(fused_state(crossing))
  projection = planner._fusion_projections[0]
  interval = int(np.searchsorted(T_IDXS, 0.7) - 1)
  assert projection.occupied[interval] and projection.occupied[interval + 1]
  assert not np.any(projection.occupied[interval + 2:])
  assert action(planner) < 0.0
  clearance = stationary_window_clearance(planner, 8.0, 0.7, 0.85)
  assert clearance >= CRASH_DISTANCE
  assert np.min(planner.a_solution) >= ACCEL_MIN - QP_FEASIBILITY_TOL
  assert np.max(planner.a_solution) <= ACCEL_MAX + QP_FEASIBILITY_TOL
  record_property('minimum_occupancy_clearance', clearance)
  record_property('minimum_acceleration', float(np.min(planner.a_solution)))
  record_property('maximum_absolute_jerk', float(np.max(np.abs(planner.j_solution))))
  # The independent straight trajectory reaches x=8 inside this unsampled
  # occupancy window. Checking only the native node times would miss it.
  states = np.column_stack((10.0 * T_IDXS, np.full(N+1, 10.0), np.zeros(N+1)))
  assert planner.collision_in_occupancy(projection, states, np.zeros(N))
  clear = object_lead(distance=8.0, start=0.1, end=0.2, path='entering', persistent=False)
  clear_projection = planner.project_fusion_lead(clear, slot=None, reference_ns=EPOCH_NS)
  assert not planner.collision_in_occupancy(clear_projection, states, np.zeros(N))


def test_short_crossing_has_an_independent_bounded_jerk_feasibility_witness():
  # This is an emergency, not a comfortable-following fixture: ramp with at
  # most 12 m/s^3 to the existing -3.5 m/s^2 bound, then release after crossing.
  # It proves the fixed geometry has a feasible native-grid control sequence;
  # the preceding test still requires the actual solver to find a safe one.
  accelerations = np.zeros(N+1)
  accelerations[1:4] = np.maximum(ACCEL_MIN, -12.0 * T_IDXS[1:4])
  jerks = np.diff(accelerations) / np.diff(T_IDXS)
  states = [np.array([0.0, 10.0, 0.0])]
  for jerk, duration in zip(jerks, np.diff(T_IDXS), strict=True):
    position, velocity, acceleration = states[-1]
    states.append(np.array([position + velocity * duration + acceleration * duration**2 / 2.0 + jerk * duration**3 / 6.0,
                            velocity + acceleration * duration + jerk * duration**2 / 2.0,
                            acceleration + jerk * duration]))
  witness = SimpleNamespace(x_sol=np.array(states), j_solution=jerks)
  assert np.max(np.abs(jerks)) <= 12.0 + 1e-12
  assert np.min(accelerations) >= ACCEL_MIN and np.max(accelerations) <= ACCEL_MAX
  assert np.min(witness.x_sol[:, 1]) > 0.0
  assert stationary_window_clearance(witness, 8.0, 0.7, 0.85) > CRASH_DISTANCE


def test_crossing_and_model_rest_are_both_kept_then_all_native_rows_are_rewritten():
  crossing = object_lead(distance=6.0, ego_speed=1.0, start=0.7, end=0.85, path='entering', persistent=False)
  reference = rest_reference(1.0, 6)
  planner = LongitudinalMpc()
  recorder = NativeProblemRecorder(planner.solver)
  planner.solver = recorder
  planner.set_cur_state(1.0, 0.0)
  planner.set_weights(prev_accel_constraint=False)
  for state, candidate in ((fused_state(crossing), reference), (fused_state(), reference),
                           (radar_state(), MpcReference(20.0))):
    for _ in range(12):
      planner.update(state, reference=candidate, now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
      assert planner.solution_status == planner.last_solver_status == 0
    if candidate is reference:
      assert planner.rest_goal_state == 'satisfied'
      lower, upper = planner.diagnostic_reference_bounds
      np.testing.assert_array_equal(lower[6:, 0], 0.0)
      np.testing.assert_array_equal(lower[6:, 1], -REST_STATE_TOL)
      np.testing.assert_array_equal(upper[6:, 0], REST_STATE_TOL)
      np.testing.assert_array_equal(upper[6:, 1], 0.0)
      assert np.max(np.abs(planner.x_sol[6:, 1:])) <= REST_STATE_TOL
      if state.radarObjects:
        assert planner.diagnostic_occupancy_bounds
        assert recorder.constraints[3, 'C'][1, 0] == 1.0
        assert stationary_window_clearance(planner, 6.0, 0.7, 0.85) >= CRASH_DISTANCE
      else:
        assert not planner.diagnostic_occupancy_bounds
        for stage in range(N):
          np.testing.assert_array_equal(recorder.constraints[stage, 'C'][:, 0], 0.0)
    else:
      assert not planner.diagnostic_occupancy_bounds and not planner._native_occupancy_bounds_active
      for stage in range(N):
        np.testing.assert_array_equal(recorder.constraints[stage, 'C'], 0.0)
        np.testing.assert_array_equal(recorder.constraints[stage, 'D'], 0.0)
        np.testing.assert_array_equal(recorder.constraints[stage, 'lg'], -1.0)
        np.testing.assert_array_equal(recorder.constraints[stage, 'ug'], 1.0)
      for stage in range(1, N+1):
        np.testing.assert_array_equal(recorder.constraints[stage, 'lbx'], -REFERENCE_BOUND)
        np.testing.assert_array_equal(recorder.constraints[stage, 'ubx'], REFERENCE_BOUND)
      assert action(planner) > 0.0


def test_early_crossing_preserves_feasible_low_speed_brake_release_before_rest():
  speed, acceleration = 0.01, -0.5
  assert speed + acceleration * T_IDXS[1] / 2.0 < 0.0
  crossing = object_lead(distance=2.0, ego_speed=speed, start=0.0, end=0.03, path='entering', persistent=False)
  planner = solve(fused_state(crossing), speed=speed, acceleration=acceleration, reference=rest_reference(speed, 6))
  assert planner.rest_goal_state == 'satisfied'
  assert planner.diagnostic_occupancy_bounds[0][0] == 0
  assert planner.trajectory_minimum_velocity(planner.x_sol, planner.j_solution) >= -QP_FEASIBILITY_TOL
  assert stationary_window_clearance(planner, 2.0, 0.0, 0.03) >= CRASH_DISTANCE


def test_impossible_finite_conflict_reports_failure_and_reset_removes_its_native_bounds():
  crossing = object_lead(distance=0.8, start=0.02, end=0.10, path='entering', persistent=False, physical_window=(0.02, 0.10))
  planner = LongitudinalMpc()
  recorder = NativeProblemRecorder(planner.solver)
  planner.solver = recorder
  planner.set_cur_state(10.0, 0.0)
  planner.set_weights(prev_accel_constraint=False)
  planner.update(fused_state(crossing), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status != 0 and not planner.previous_solution_valid
  assert planner.failure_reason.startswith('finite_occupancy_')
  assert planner.diagnostic_occupancy_bounds
  assert planner.finite_risk_braking_accel == ACCEL_MIN
  assert planner.finite_risk_evidence[0].identity == ('radar-test', 1)
  assert not planner._native_occupancy_bounds_active
  # These checks precede the next update; solver.reset alone cannot be assumed
  # to erase model constraints for a direct run caller.
  for stage in range(N):
    np.testing.assert_array_equal(recorder.constraints[stage, 'C'], 0.0)
    np.testing.assert_array_equal(recorder.constraints[stage, 'D'], 0.0)
  for stage in range(1, N+1):
    np.testing.assert_array_equal(recorder.constraints[stage, 'lbx'], -REFERENCE_BOUND)
    np.testing.assert_array_equal(recorder.constraints[stage, 'ubx'], REFERENCE_BOUND)
  planner.set_cur_state(10.0, 0.0)
  for _ in range(12):
    planner.update(fused_state(), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
    assert planner.solution_status == 0
  assert action(planner) > 0.0 and not planner.diagnostic_occupancy_bounds
  assert planner.finite_risk_braking_accel is None and planner.finite_risk_evidence == ()


@pytest.mark.parametrize('source', ['radar', 'fused'])
@pytest.mark.parametrize('physical_start', [0.0, 0.02])
def test_fresh_physically_unavoidable_conflict_preserves_mitigation_evidence_through_failure(source, physical_start):
  crossing = object_lead(distance=0.8, start=0.0, end=0.35, path='entering', persistent=False, source=source,
                         physical_window=(physical_start, 0.10))
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner.set_weights(prev_accel_constraint=False)
  planner.update(fused_state(crossing), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status != 0 and not planner.previous_solution_valid
  assert planner.finite_risk_braking_accel == ACCEL_MIN
  evidence, = planner.finite_risk_evidence
  assert evidence.source == source and evidence.identity == ('radar-test', 1)
  assert evidence.consumed_ns == evidence.reference_ns == evidence.model_ns == EPOCH_NS
  assert evidence.estimate_ns == evidence.position_ns == evidence.velocity_ns == EPOCH_NS
  assert evidence.entry_ego_upper_position < evidence.entry_object_position + CRASH_DISTANCE
  assert evidence.proof_object_position - evidence.proof_ego_lower_position < CRASH_DISTANCE
  assert evidence.conflict_start_s == physical_start and evidence.conflict_end_s == 0.10
  assert evidence.conflict_start_s <= evidence.proof_time_s <= evidence.conflict_end_s
  assert planner.object_influence_identities == ()  # Failed-plan UI adoption is a separate owner receipt.
  planner.reset()
  assert planner.finite_risk_braking_accel is None and planner.finite_risk_evidence == ()


@pytest.mark.parametrize('field,value', [
  pytest.param('physicalConflictValid', ..., id='missing-validity'),
  pytest.param('physicalConflictStart', ..., id='missing-start'),
  pytest.param('physicalConflictEnd', ..., id='missing-end'),
  pytest.param('physicalConflictValid', False, id='false-validity'),
  pytest.param('physicalConflictValid', 1, id='integer-validity'),
  pytest.param('physicalConflictValid', 'true', id='string-validity'),
  pytest.param('physicalConflictValid', np.bool_(True), id='numpy-validity'),
  pytest.param('physicalConflictValid', None, id='null-validity'),
  pytest.param('physicalConflictStart', False, id='boolean-start'),
  pytest.param('physicalConflictEnd', True, id='boolean-end'),
  pytest.param('physicalConflictStart', '0.02', id='string-start'),
  pytest.param('physicalConflictEnd', '0.10', id='string-end'),
  pytest.param('physicalConflictStart', None, id='null-start'),
  pytest.param('physicalConflictEnd', None, id='null-end'),
  pytest.param('physicalConflictStart', float('nan'), id='nan-start'),
  pytest.param('physicalConflictEnd', float('nan'), id='nan-end'),
  pytest.param('physicalConflictStart', float('-inf'), id='infinite-start'),
  pytest.param('physicalConflictEnd', float('inf'), id='infinite-end'),
  pytest.param('physicalConflictStart', 10**400, id='overflowing-start'),
  pytest.param('physicalConflictStart', -0.01, id='start-outside-ordinary-window'),
  pytest.param('physicalConflictEnd', 2.01, id='end-outside-ordinary-window'),
  pytest.param('physicalConflictStart', 0.10, id='empty-physical-window'),
  pytest.param('physicalConflictEnd', 0.01, id='reversed-physical-window'),
])
def test_mitigation_requires_explicit_valid_physical_window_at_consumption(field, value):
  crossing = object_lead(distance=0.8, start=0.0, end=2.0, path='entering', persistent=False, physical_window=(0.02, 0.10))
  state = fused_state(crossing)
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner._fusion_projections = planner.fusion_lead_projections(state, now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.prove_finite_physical_risk(state, now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  if value is ...:
    delattr(crossing.evidence, field)
  else:
    setattr(crossing.evidence, field, value)
  assert planner.prove_finite_physical_risk(state, now_ns=EPOCH_NS, reference_ns=EPOCH_NS) == ()
  assert planner.diagnostic_solve_calls == 0  # Evidence validation does not depend on a numerical solve.


def test_lateral_exit_padding_cannot_prove_a_physical_emergency():
  crossing = object_lead(distance=6.0, ego_speed=15.0, start=0.0, end=0.47, path='entering', persistent=False,
                         physical_window=(0.0, 0.22))
  crossing.evidence.lateralSpeed = 5.0
  # Raw 1.10 m lateral occupancy ends at 0.22 s; padding lasts until 0.47 s.
  # The earliest ego arrival intersects only that padded tail.
  assert 15.0 * 0.22 + 0.5 * ACCEL_MAX * 0.22**2 < 6.0 - CRASH_DISTANCE
  assert 15.0 * 0.47 + 0.5 * ACCEL_MIN * 0.47**2 > 6.0 - CRASH_DISTANCE
  planner = LongitudinalMpc()
  planner.set_cur_state(15.0, 0.0)
  planner.set_weights(prev_accel_constraint=False)
  planner.update(fused_state(crossing), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status != 0 and planner.diagnostic_occupancy_bounds
  assert planner.finite_risk_braking_accel is None and planner.finite_risk_evidence == ()


@pytest.mark.parametrize('acceleration', [5.0, -5.0, 1e-12, -1e-12])
def test_nonzero_lead_acceleration_needs_its_own_physical_geometry_for_mitigation(acceleration):
  crossing = object_lead(distance=0.8, speed=5.0, start=0.0, end=0.20, path='entering', persistent=False,
                         acceleration=acceleration, physical_window=(0.0, 0.20))
  state = fused_state(crossing)
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner._fusion_projections = planner.fusion_lead_projections(state, now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert len(planner._fusion_projections) == 1 and np.any(planner._fusion_projections[0].occupied)
  assert planner._fusion_projections[0].lead.aLeadK == acceleration  # Ordinary MPC keeps the qualified measurement.
  assert planner.prove_finite_physical_risk(state, now_ns=EPOCH_NS, reference_ns=EPOCH_NS) == ()


def test_cv_only_mitigation_would_discard_a_credible_positive_acceleration_escape():
  crossing = object_lead(distance=8.0, speed=5.0, ego_speed=15.0, start=0.0, end=1.0, path='entering', persistent=False,
                         physical_window=(0.0, 1.0))
  state = fused_state(crossing)
  planner = LongitudinalMpc()
  planner.set_cur_state(15.0, 0.0)
  planner._fusion_projections = planner.fusion_lead_projections(state, now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.prove_finite_physical_risk(state, now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  crossing.aLeadK = 5.0
  planner._fusion_projections = planner.fusion_lead_projections(state, now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  projection, = planner._fusion_projections
  stage_ends = np.minimum(T_IDXS[1:], 1.0)
  lead_velocities = np.diff(projection.raw_trajectory[:, 0]) / np.diff(T_IDXS)
  # The ideal full-brake gap decreases throughout [0, 1], yet ends over 2 m:
  # substituting CV would instead yield -0.25 m and falsely prove urgency.
  assert np.all(lead_velocities[T_IDXS[:-1] < 1.0] - 15.0 - ACCEL_MIN * stage_ends[T_IDXS[:-1] < 1.0] < 0.0)
  assert np.interp(1.0, T_IDXS, projection.raw_trajectory[:, 0]) - (15.0 + 0.5 * ACCEL_MIN) > 2.0
  assert planner.prove_finite_physical_risk(state, now_ns=EPOCH_NS, reference_ns=EPOCH_NS) == ()


@pytest.mark.parametrize('reference_offset_ns', [0, 100_000_000, 150_000_000])
def test_expired_physical_window_does_not_inherit_live_padding(reference_offset_ns):
  crossing = object_lead(distance=2.0, start=0.0, end=0.35, path='entering', persistent=False, physical_window=(0.0, 0.10))
  state = fused_state(crossing)
  now_ns, reference_ns = EPOCH_NS + 150_000_000, EPOCH_NS + reference_offset_ns
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner._fusion_projections = planner.fusion_lead_projections(state, now_ns=now_ns, reference_ns=reference_ns)
  assert len(planner._fusion_projections) == 1 and np.any(planner._fusion_projections[0].occupied)
  assert planner.prove_finite_physical_risk(state, now_ns=now_ns, reference_ns=reference_ns) == ()


@pytest.mark.parametrize('distance,physical_window', [(14.0, (0.8, 1.5)), (18.0, (1.1, 1.6))])
def test_later_physical_collision_does_not_extend_the_mitigation_horizon(distance, physical_window):
  crossing = object_lead(distance=distance, ego_speed=15.0, start=0.0, end=2.0, path='entering', persistent=False,
                         physical_window=physical_window)
  state = fused_state(crossing)
  planner = LongitudinalMpc()
  planner.set_cur_state(15.0, 0.0)
  planner._fusion_projections = planner.fusion_lead_projections(state, now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert np.any(planner._fusion_projections[0].occupied)
  assert planner.prove_finite_physical_risk(state, now_ns=EPOCH_NS, reference_ns=EPOCH_NS) == ()


@pytest.mark.parametrize('reference_offset_ns,consume_offset_ns', [(-50_000_000, 10_000_000),
                                                                 (50_000_000, 70_000_000),
                                                                 (-50_000_000, 80_000_000)])
@pytest.mark.parametrize('physical_end', [0.15, 1.5])
def test_physical_proof_preserves_source_window_and_limits_the_actual_proof_time(reference_offset_ns, consume_offset_ns, physical_end):
  crossing = object_lead(distance=0.8, start=0.0, end=2.0, path='entering', persistent=False, physical_window=(0.02, physical_end))
  state = fused_state(crossing)
  now_ns, reference_ns = EPOCH_NS + consume_offset_ns, EPOCH_NS + reference_offset_ns
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner._fusion_projections = planner.fusion_lead_projections(state, now_ns=now_ns, reference_ns=reference_ns)
  evidence, = planner.prove_finite_physical_risk(state, now_ns=now_ns, reference_ns=reference_ns)
  assert evidence.conflict_start_s == pytest.approx(0.02 - reference_offset_ns / 1e9)
  assert evidence.conflict_end_s == pytest.approx(physical_end - reference_offset_ns / 1e9)
  consumed_age = (now_ns - reference_ns) / 1e9
  assert max(consumed_age, evidence.conflict_start_s) <= evidence.proof_time_s <= min(consumed_age + 1.0, evidence.conflict_end_s)
  assert evidence.entry_ego_upper_position < evidence.entry_object_position + CRASH_DISTANCE
  assert evidence.proof_object_position - evidence.proof_ego_lower_position < CRASH_DISTANCE


@pytest.mark.parametrize('field,value', [('physicalConflictValid', False), ('physicalConflictStart', 0.03), ('physicalConflictEnd', 0.09)])
def test_same_object_cannot_publish_inconsistent_physical_windows_in_its_display_view(field, value):
  crossing = object_lead(distance=0.8, start=0.0, end=0.35, path='entering', persistent=False, physical_window=(0.02, 0.10))
  display = deepcopy(crossing)
  setattr(display.evidence, field, value)
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner.update(fused_state(crossing, display=(display,)), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status == MPC_INVALID_INPUT and planner.diagnostic_solve_calls == 0
  assert planner.finite_risk_braking_accel is None and planner.finite_risk_evidence == ()


@pytest.mark.parametrize('case', ['coasting', 'stale', 'bad_input', 'caution', 'visual', 'degraded', 'late_passing', 'conservative_envelope'])
def test_solver_failure_alone_never_grants_finite_risk_emergency_braking(case):
  crossing = object_lead(distance=0.8, start=0.02, end=0.10, path='entering', persistent=False, physical_window=(0.02, 0.10))
  state = fused_state(crossing)
  now_ns = reference_ns = EPOCH_NS
  if case == 'coasting':
    crossing.evidence.source, crossing.evidence.coasting, crossing.evidence.fcwQualified = 'predicted', True, False
  elif case == 'stale':
    now_ns = reference_ns = EPOCH_NS + 300_000_000
  elif case == 'bad_input':
    state.radarObjects.append(object_lead(2, distance=float('nan')))
  elif case == 'caution':
    crossing.evidence.pathState, crossing.evidence.cautionQualified = 'edge', True
    crossing.evidence.brakeQualified = crossing.evidence.fcwQualified = False
  elif case == 'visual':
    state = fused_state(object_lead(distance=0.8, start=0.02, end=0.10, path='entering', persistent=False, source='vision',
                                   physical_window=(0.02, 0.10)))
  elif case == 'degraded':
    state.fusionDegraded = True
  elif case == 'late_passing':
    state = fused_state(object_lead(distance=5.0, start=4.0, end=6.0, path='entering', persistent=False, physical_window=(4.0, 6.0)))
  else:
    # The conservative stage-end/min-object-position row can be infeasible
    # although the real faster target maintains a positive gap throughout.
    # That software envelope is not a proof of an unavoidable physical risk.
    state = fused_state(object_lead(distance=0.8, speed=12.0, start=0.0, end=0.10, path='entering', persistent=False,
                                   physical_window=(0.0, 0.10)))
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner.set_weights(prev_accel_constraint=False)
  planner.update(state, reference=MpcReference(20.0), now_ns=now_ns, reference_ns=reference_ns)
  assert planner.finite_risk_braking_accel is None and planner.finite_risk_evidence == ()
  if case == 'conservative_envelope':
    assert planner.solution_status != 0


def test_numeric_failure_without_physical_proof_has_no_emergency_braking_evidence(monkeypatch):
  crossing = object_lead(distance=8.0, start=0.7, end=0.85, path='entering', persistent=False, physical_window=(0.7, 0.85))
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner.set_weights(prev_accel_constraint=False)
  actual_postprocessing = planner.check_effective_rest_goal

  def disconnected_postprocessing():
    actual_postprocessing()
    planner.x_sol[6, 0] += 1.0

  monkeypatch.setattr(planner, 'check_effective_rest_goal', disconnected_postprocessing)
  planner.update(fused_state(crossing), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status == MPC_OCCUPANCY_UNSATISFIED
  assert planner.finite_risk_braking_accel is None and planner.finite_risk_evidence == ()


def test_rest_postprocessing_cannot_publish_a_trajectory_that_breaks_the_crossing_row(monkeypatch):
  crossing = object_lead(distance=8.0, start=0.7, end=0.85, path='entering', persistent=False)
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner.set_weights(prev_accel_constraint=False)

  def unsafe_postprocessing():
    planner.x_sol[:, 0] = 10.0 * T_IDXS
    planner.x_sol[:, 1], planner.x_sol[:, 2] = 10.0, 0.0
    planner.j_solution[:] = 0.0

  monkeypatch.setattr(planner, 'check_effective_rest_goal', unsafe_postprocessing)
  planner.update(fused_state(crossing), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status == MPC_OCCUPANCY_UNSATISFIED
  assert planner.failure_reason == 'finite_occupancy_position_bound_violated'
  assert not planner.previous_solution_valid


def late_passing_objects():
  # Motion/occupancy values from route 8b / sample 10 of the retained
  # 1f86b1c pilot. This is a fixed geometric fixture, not a complete replay of
  # its cost history or an assertion about the physical objects' identities.
  front = object_lead(5, distance=47.113563537597656, speed=10.071088790893555, acceleration=-0.6348151564598083,
                      start=0.0, end=5.55867338180542, persistent=False)
  first = object_lead(9, distance=37.096771240234375, speed=-1.3435324430465698, acceleration=0.3417064845561981,
                      start=3.637268304824829, end=10.0, path='entering', persistent=False)
  second = object_lead(10, distance=35.18659591674805, speed=-1.8218282461166382,
                       start=5.349180698394775, end=10.0, path='entering', persistent=False)
  return front, first, second


def test_late_crossings_use_one_nonaccelerating_joint_passing_witness(record_property):
  speed, acceleration = 11.190595626831055, 0.16649655997753143
  planner = solve(fused_state(*late_passing_objects()), speed=speed, acceleration=acceleration,
                  reference=MpcReference(20.0, allow_passing=True))
  assert planner.diagnostic_passing['accepted']
  assert [p.identity[1] for p in planner._fusion_projections if p.pass_before] == [9, 10]
  witness, jerks = planner._passing_witness
  np.testing.assert_array_equal(witness[0], [0.0, speed, acceleration])
  assert witness[1, 2] == pytest.approx(0.0, abs=1e-12)
  np.testing.assert_array_equal(jerks[1:], 0.0)
  assert planner.diagnostic_passing['witnessReleaseJerk'] < 0.0
  assert planner._passing_speed_limit == pytest.approx(speed + acceleration * T_IDXS[1] / 2.0)
  assert np.max(planner.v_solution) <= planner._passing_speed_limit + QP_FEASIBILITY_TOL
  assert np.max(planner.a_solution[1:]) <= QP_FEASIBILITY_TOL
  for projection in planner._fusion_projections:
    minimum, maximum = planner.occupancy_clearance_extrema(projection, planner.x_sol, planner.j_solution)
    if projection.pass_before:
      assert maximum <= -STOP_DISTANCE
    else:
      assert minimum >= CRASH_DISTANCE
  assert planner.diagnostic_solve_calls == 12
  record_property('joint_passing_initial_request', action(planner))
  record_property('joint_passing_speed_limit', planner._passing_speed_limit)


@pytest.mark.parametrize('veto', [
  'near_persistent_car', 'model_rest', 'negative_acceleration_ceiling', 'cruise_stop', 'cruise_slower', 'not_authorized',
])
def test_passing_cannot_spend_nearer_lead_or_independent_model_braking(veto):
  speed, acceleration = 11.190595626831055, 0.16649655997753143
  objects = late_passing_objects()
  reference = MpcReference(20.0, allow_passing=True)
  if veto == 'near_persistent_car':
    objects = (*objects, object_lead(20, distance=18.0, speed=0.0, ego_speed=speed))
  elif veto == 'model_rest':
    reference = replace(rest_reference(speed, N), allow_passing=True)
  elif veto == 'negative_acceleration_ceiling':
    reference = MpcReference(20.0, accel_max=-0.3, allow_passing=True)
  elif veto == 'not_authorized':
    reference = MpcReference(20.0)
  else:
    reference = MpcReference(0.0 if veto == 'cruise_stop' else 10.0, allow_passing=True)
  planner = LongitudinalMpc()
  planner.set_cur_state(speed, acceleration)
  planner.set_weights(prev_accel_constraint=False)
  planner.update(fused_state(*objects), reference=reference, now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner._passing_speed_limit is None
  assert not planner.diagnostic_passing or not planner.diagnostic_passing['accepted']
  # This frozen initial state cannot stay behind the approaching second
  # crossing through its whole horizon. Failure remains explicit when the
  # only alternative is vetoed; it is not presented as a safe stop/pass.
  assert planner.solution_status != 0 and not planner.previous_solution_valid


@pytest.mark.parametrize('value', [None, 0, 1, 'true', np.bool_(True)])
def test_passing_requires_an_explicit_boolean_authorization(value):
  with pytest.raises(ValueError, match='passing authorization'):
    MpcReference(20.0, allow_passing=value)


def test_passing_releases_existing_braking_without_a_positive_reacceleration():
  planner = solve(fused_state(*late_passing_objects()), speed=11.190595626831055, acceleration=-0.5,
                  reference=MpcReference(20.0, allow_passing=True))
  assert planner.diagnostic_passing['accepted']
  assert planner.x_sol[0, 2] == -0.5
  assert np.max(planner.a_solution[1:]) <= QP_FEASIBILITY_TOL
  lower, upper = planner.diagnostic_reference_bounds
  assert np.max(upper[1:, 1]) == 0.0
  assert np.min(lower[1:, 1]) == ACCEL_MIN


def test_joint_passing_exit_clears_both_position_sides_and_speed_ceiling():
  speed, acceleration = 11.190595626831055, 0.16649655997753143
  planner = LongitudinalMpc()
  recorder = NativeProblemRecorder(planner.solver)
  planner.solver = recorder
  planner.set_cur_state(speed, acceleration)
  planner.set_weights(prev_accel_constraint=False)
  for _ in range(12):
    planner.update(fused_state(*late_passing_objects()), reference=MpcReference(20.0, allow_passing=True),
                   now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
    assert planner.solution_status == 0
  assert planner._passing_speed_limit is not None
  assert any(row.position_lower_limit > -REFERENCE_BOUND for row in planner._occupancy_bounds if row is not None)
  for _ in range(12):
    planner.update(radar_state(), reference=MpcReference(20.0))
    assert planner.solution_status == 0
  assert planner._passing_speed_limit is None and planner.diagnostic_passing is None
  for stage in range(N):
    np.testing.assert_array_equal(recorder.constraints[stage, 'C'], 0.0)
    np.testing.assert_array_equal(recorder.constraints[stage, 'D'], 0.0)
    np.testing.assert_array_equal(recorder.constraints[stage, 'lg'], -1.0)
    np.testing.assert_array_equal(recorder.constraints[stage, 'ug'], 1.0)
  for stage in range(1, N+1):
    np.testing.assert_array_equal(recorder.constraints[stage, 'ubx'], REFERENCE_BOUND)
  assert action(planner) > 0.0


def test_finite_occupancy_rejects_a_dynamically_disconnected_native_trajectory(monkeypatch):
  crossing = object_lead(distance=8.0, start=0.7, end=0.85, path='entering', persistent=False)
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner.set_weights(prev_accel_constraint=False)
  actual_postprocessing = planner.check_effective_rest_goal

  def disconnected_postprocessing():
    actual_postprocessing()
    planner.x_sol[6, 0] -= 1.0  # Outside the occupancy: point clearance alone cannot catch this.

  monkeypatch.setattr(planner, 'check_effective_rest_goal', disconnected_postprocessing)
  planner.update(fused_state(crossing), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status == MPC_OCCUPANCY_UNSATISFIED
  assert planner.failure_reason == 'finite_occupancy_dynamics_residual_exceeded'
  assert not planner.previous_solution_valid


@pytest.mark.parametrize('source', ['radar', 'vision', 'legacy_vision'])
def test_active_object_tokens_use_physical_or_current_visual_identity(source):
  if source == 'legacy_vision':
    state = radar_state(distance=25.0, speed=2.0)
    state.mdMonoTime = EPOCH_NS
  else:
    state = fused_state(object_lead(distance=25.0, speed=2.0, ego_speed=15.0, source=source))
  planner = solve(state, speed=15.0)
  expected = ('radar-test', 1) if source == 'radar' else ('vision', EPOCH_NS, 0)
  assert planner.object_influence_identities == (expected,)
  assert action(planner) < 0.0
  assert planner.diagnostic_solve_calls == 12


def test_legacy_capnp_builder_stays_byte_identical_through_mpc_and_ui_attribution():
  state = log.RadarState.new_message()
  state.mdMonoTime = EPOCH_NS
  state.leadOne = {'present': True, 'dRel': 25.0, 'vLead': 2.0, 'vRel': -13.0,
                   'aLeadK': 0.0, 'aLeadTau': 1.5, 'modelProb': 0.99}
  before, serialized = state.to_dict(), state.to_bytes()
  state.clear_write_flag()
  # Calling the attribution helper on a builder must not allocate evidence.
  assert LongitudinalMpc.lead_influence_token(state.leadOne, model_ns=EPOCH_NS, slot=0) == ('vision', EPOCH_NS, 0)
  planner = solve(state, speed=15.0)
  assert planner.object_influence_identities == (('vision', EPOCH_NS, 0),)
  assert state.to_dict() == before
  assert state.to_bytes() == serialized


def test_virtual_cruise_winner_does_not_attribute_an_unused_far_radar_object():
  planner = solve(fused_state(object_lead(distance=200.0, speed=25.0)))
  assert planner.object_influence_identities == ()
  assert action(planner) > 0.0


def test_objects_tight_at_different_horizon_stages_both_retain_constraint_evidence():
  first = object_lead(1, distance=15.0, speed=5.0)
  second = object_lead(2, distance=40.0, speed=0.0)
  planner = solve(fused_state(first, second))
  assert set(planner.object_influence_identities) == {('radar-test', 1), ('radar-test', 2)}


def test_new_failure_clears_previously_active_constraint_tokens():
  planner = solve(fused_state(object_lead(distance=25.0, speed=2.0)), speed=15.0)
  assert planner.object_influence_identities
  bad = object_lead(distance=float('nan'))
  planner.update(fused_state(bad), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status != 0 and planner.object_influence_identities == ()


@pytest.mark.parametrize('with_rest', [False, True])
def test_nanometer_position_reintegration_preserves_native_jerk_and_all_constraints(with_rest):
  speed = 1.0 if with_rest else 10.0
  reference = rest_reference(speed, 6) if with_rest else MpcReference(20.0)
  crossing = object_lead(distance=6.0 if with_rest else 20.0, ego_speed=speed,
                         start=0.7 if with_rest else 1.2, end=0.85 if with_rest else 2.6,
                         path='entering', persistent=False)
  state = fused_state(crossing)
  planner = solve(state, speed=speed, reference=reference)
  native = planner.solver

  class PositionNoise(NativeProblemRecorder):
    def get(self, stage, field):
      result = super().get(stage, field)
      if stage == N and field == 'x':
        result = result.copy()
        result[0] += 0.5 * OCCUPANCY_POSITION_REINTEGRATION_LIMIT
      return result

  planner.solver = PositionNoise(native)
  initial = planner.x0.copy()
  planner.update(state, reference=reference, now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status == planner.last_solver_status == 0
  diagnostic = planner.diagnostic_occupancy_projection
  assert diagnostic['applied'] and diagnostic['maximumJerkCorrection'] == 0.0
  assert diagnostic['maximumStateCorrection'][0] <= OCCUPANCY_POSITION_REINTEGRATION_LIMIT
  np.testing.assert_array_equal(planner.x_sol[0], initial)
  np.testing.assert_array_equal(planner.j_solution, [native.get(stage, 'u')[0] for stage in range(N)])
  assert np.max(np.abs(planner.trajectory_dynamics_defects(planner.x_sol, planner.j_solution))) <= QP_FEASIBILITY_TOL
  assert planner.occupancy_trajectory_error(planner.x_sol, planner.j_solution, tolerance=QP_FEASIBILITY_TOL) == ''
  if with_rest:
    assert planner.rest_goal_state == 'satisfied'
    assert np.max(np.abs(planner.x_sol[6:, 1:])) <= REST_STATE_TOL


@pytest.mark.parametrize('kind', ['micrometer_position', 'meter_position', 'velocity', 'cumulative_position'])
def test_reintegration_never_admits_larger_or_accumulated_native_errors(kind):
  crossing = object_lead(distance=20.0, start=1.2, end=2.6, path='entering', persistent=False)
  state = fused_state(crossing)
  planner = solve(state)

  class NativeNoise(NativeProblemRecorder):
    def get(self, stage, field):
      result = super().get(stage, field)
      if field == 'x':
        result = result.copy()
        if kind == 'cumulative_position':
          result[0] += stage * OCCUPANCY_POSITION_REINTEGRATION_LIMIT / 2.0
        elif stage == N:
          if kind == 'velocity':
            result[1] += 3.0 * QP_FEASIBILITY_TOL
          else:
            result[0] += 1e-6 if kind == 'micrometer_position' else 1.0
      return result

  planner.solver = NativeNoise(planner.solver)
  planner.update(state, reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status == MPC_OCCUPANCY_UNSATISFIED
  expected = ('finite_occupancy_reintegration_budget_exceeded' if kind == 'cumulative_position'
              else 'finite_occupancy_dynamics_residual_exceeded')
  assert planner.failure_reason == expected
  assert not planner.diagnostic_occupancy_projection['applied']
  assert not planner.previous_solution_valid and planner.object_influence_identities == ()


def test_failed_native_solve_cannot_enter_position_reintegration(monkeypatch):
  crossing = object_lead(distance=20.0, start=1.2, end=2.6, path='entering', persistent=False)
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  recorder = NativeProblemRecorder(planner.solver)
  recorder.failure = 'solver_status'
  planner.solver = recorder

  def forbidden_repair():
    pytest.fail('A failed native solve is not eligible for numerical repair')

  monkeypatch.setattr(planner, 'reintegrate_occupancy_precision', forbidden_repair)
  planner.update(fused_state(crossing), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status == 4 and planner.diagnostic_occupancy_projection is None


def test_valid_edge_caution_has_no_mpc_obstacle_or_fcw_authority():
  edge = object_lead(distance=2.0, speed=0.0, path='edge', persistent=False)
  edge.evidence.cautionQualified = True
  edge.evidence.brakeQualified = edge.evidence.fcwQualified = False
  planner, empty = solve(fused_state(edge)), solve(fused_state())
  np.testing.assert_array_equal(planner.params, empty.params)
  np.testing.assert_array_equal(planner.x_sol, empty.x_sol)
  assert planner.crash_cnt == 0 and planner.object_influence_identities == ()


def test_invalid_edge_caution_is_not_silently_discarded():
  edge = object_lead(distance=2.0, speed=0.0, path='edge', persistent=False)
  edge.evidence.cautionQualified = True
  edge.evidence.brakeQualified = edge.evidence.fcwQualified = False
  edge.evidence.positionValid = False
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner.update(fused_state(edge), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status == MPC_INVALID_INPUT and planner.diagnostic_solve_calls == 0


def test_other_objects_do_not_inherit_the_display_leads_compact_stop_distance():
  displayed = object_lead(1, distance=10.0, ego_speed=0.0)
  nearer = object_lead(2, distance=6.0, ego_speed=0.0)
  reference = MpcReference(20.0, comfort_stop_distance=3.0, lead_risk_stop_distances=(4.0, 6.0))
  planner = solve(fused_state(displayed, nearer, display=(displayed,)), speed=0.0, reference=reference)
  np.testing.assert_array_equal(planner.params[:, COMFORT_STOP_DISTANCE], STOP_DISTANCE)
  np.testing.assert_array_equal(planner.params[:, RISK_STOP_DISTANCE], STOP_DISTANCE)
  assert planner.rest_recovery_position_limit == pytest.approx(6.0 - 0.25)


def test_qualified_displayed_radar_retains_the_existing_compact_contract():
  lead = object_lead(distance=3.0, ego_speed=0.0)
  reference = MpcReference(None, comfort_stop_distance=3.0, lead_risk_stop_distances=(4.0, 6.0))
  planner = solve(fused_state(lead, display=(lead,)), speed=0.0, reference=reference)
  np.testing.assert_array_equal(planner.params[:, COMFORT_STOP_DISTANCE], 3.0)
  np.testing.assert_array_equal(planner.params[:, RISK_STOP_DISTANCE], 4.0)


def test_mild_negative_radar_velocity_keeps_approaching_geometry():
  lead = object_lead(distance=20.0, speed=-0.25)
  planner = solve(fused_state(lead))
  projection = planner._fusion_projections[0]
  assert np.all(projection.raw_trajectory[:, 1] < 0.0)
  assert projection.raw_trajectory[-1, 0] < lead.dRel
  assert action(planner) < 0.0


@pytest.mark.parametrize('offset_ns', [-50_000_000, 100_000_000])
def test_prediction_and_occupancy_are_aligned_to_the_initial_state_epoch(offset_ns):
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  lead = object_lead(distance=20.0, speed=4.0, start=0.4, end=1.5, path='entering', persistent=False)
  projection = planner.project_fusion_lead(lead, slot=None, reference_ns=EPOCH_NS + offset_ns)
  offset = offset_ns / 1e9
  assert projection.raw_trajectory[0, 0] == pytest.approx(20.0 - 6.0 * offset)
  assert projection.conflict_start == pytest.approx(0.4 - offset)
  assert projection.conflict_end == pytest.approx(1.5 - offset)


def test_fresh_fusion_can_align_back_to_an_earlier_car_state_without_moving_its_epoch():
  reference_ns, now_ns = EPOCH_NS - 50_000_000, EPOCH_NS + 10_000_000
  lead = object_lead(distance=20.0, speed=4.0)
  already_aligned = object_lead(distance=20.3, speed=4.0, epoch=reference_ns)
  projected = solve(fused_state(lead), now_ns=now_ns, reference_ns=reference_ns)
  baseline = solve(fused_state(already_aligned), now_ns=now_ns, reference_ns=reference_ns)
  assert projected._fusion_projections[0].raw_trajectory[0, 0] == pytest.approx(20.3)
  np.testing.assert_allclose(projected.params, baseline.params, atol=1e-12, rtol=0.0)
  np.testing.assert_allclose(projected.x_sol, baseline.x_sol, atol=1e-9, rtol=0.0)


@pytest.mark.parametrize('age_ns', [149_000_000, 150_000_000, 170_000_000])
@pytest.mark.parametrize('older_initial_state', [False, True])
def test_natural_occupancy_end_keeps_other_leads_and_the_model_running(age_ns, older_initial_state):
  crossing = object_lead(2, distance=5.0, start=0.0, end=0.15, path='entering', persistent=False)
  front = object_lead(1, distance=30.0, speed=4.0)
  velocity = np.maximum(0.0, 10.0 - T_IDXS)
  position = np.cumsum(np.diff(T_IDXS, prepend=0.0) * velocity)
  preference = MovingPreference(position, velocity, np.where(velocity > 0.0, -1.0, 0.0), np.ones(N+1, dtype=bool))
  source = ModelSource(EPOCH_NS, EPOCH_NS, EPOCH_NS, proxy_age_s=0.0)
  reference = MpcReference(20.0, typed_model=ModelReferenceEvidence(source, preference, ()))
  now_ns = EPOCH_NS + age_ns
  reference_ns = EPOCH_NS + 100_000_000 if older_initial_state else now_ns
  arguments = {'reference': reference, 'now_ns': now_ns, 'reference_ns': reference_ns}
  baseline = solve(fused_state(front, display=(front,)), **arguments)
  planner = solve(fused_state(front, crossing, display=(front,)), **arguments)
  np.testing.assert_array_equal(planner.params[:, MODEL_V_REF], velocity)
  assert action(planner) < 0.0
  assert planner.rest_recovery_position_limit == baseline.rest_recovery_position_limit
  if age_ns < 150_000_000:
    assert len(planner._fusion_projections) == 2
    assert np.any(planner._fusion_projections[1].occupied)
    assert planner.params[0, 2] < baseline.params[0, 2]
  else:
    assert len(planner._fusion_projections) == 1
    np.testing.assert_array_equal(planner.params, baseline.params)
    np.testing.assert_allclose(planner.x_sol, baseline.x_sol, atol=1e-9, rtol=0.0)


def test_expired_occupancy_does_not_hide_invalid_measurement_health():
  crossing = object_lead(2, distance=5.0, start=0.0, end=0.15, path='entering', persistent=False)
  crossing.evidence.velocityValid = False
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  now_ns = EPOCH_NS + 170_000_000
  planner.update(fused_state(object_lead(), crossing), reference=MpcReference(20.0), now_ns=now_ns, reference_ns=now_ns)
  assert planner.solution_status == MPC_INVALID_INPUT
  assert planner.diagnostic_solve_calls == 0


def test_independent_model_braking_is_not_replaced_by_a_farther_radar_car():
  velocity = np.maximum(0.0, 10.0 - 2.0 * T_IDXS)
  position = np.cumsum(np.diff(T_IDXS, prepend=0.0) * velocity)
  preference = MovingPreference(position, velocity, np.where(velocity > 0.0, -2.0, 0.0), np.ones(N+1, dtype=bool))
  source = ModelSource(EPOCH_NS, EPOCH_NS, EPOCH_NS, proxy_age_s=0.0)
  reference = MpcReference(20.0, typed_model=ModelReferenceEvidence(source, preference, ()))
  model = solve(fused_state(), reference=reference)
  joint = solve(fused_state(object_lead(distance=100.0, speed=15.0)), reference=reference)
  assert action(model) < -0.05
  assert action(joint) <= action(model) + 1e-6
  np.testing.assert_allclose(joint.x_sol, model.x_sol, atol=1e-6, rtol=0.0)


def test_radar_fcw_counts_real_identity_and_fresh_epochs_outside_display_slots():
  lead = object_lead(3, distance=0.1, ego_speed=15.0)
  far = [object_lead(index, distance=100.0, speed=20.0, ego_speed=15.0) for index in (1, 2)]
  state = fused_state(*far, lead, display=far)
  planner = LongitudinalMpc()
  planner.set_weights(prev_accel_constraint=False)
  planner.set_cur_state(15.0, 0.0)
  for frame in range(4):
    stamp = EPOCH_NS + frame * 50_000_000
    for candidate in state.radarObjects:
      advance(candidate, stamp)
    lead.evidence.generation += 1  # a slot migration is still the same physical track
    planner.update(state, reference=MpcReference(20.0), now_ns=stamp, reference_ns=stamp)
    assert planner.solution_status == 0
    assert planner.crash_cnt == frame + 1
    assert planner.fcw_object_identity == ('radar-test', 3)
  planner.update(state, reference=MpcReference(20.0), now_ns=stamp, reference_ns=stamp)
  assert planner.crash_cnt == 4  # repeated publication does not renew evidence
  lead.evidence.trackId = 4
  advance(lead, stamp + 50_000_000)
  planner.update(state, reference=MpcReference(20.0), now_ns=stamp + 50_000_000, reference_ns=stamp + 50_000_000)
  assert planner.solution_status == 0
  assert planner.crash_cnt == 1 and planner.fcw_object_identity == ('radar-test', 4)


def test_new_visual_cut_in_cannot_inherit_an_unrelated_visual_fcw_count():
  planner = LongitudinalMpc()
  planner.set_cur_state(15.0, 0.0)
  planner.x_sol = np.column_stack((15.0 * T_IDXS, np.full(N+1, 15.0), np.zeros(N+1)))
  planner.j_solution = np.zeros(N)
  lead = object_lead(source='vision', distance=2.0, ego_speed=15.0)
  for frame in range(3):
    stamp = EPOCH_NS + frame * 50_000_000
    advance(lead, stamp)
    lead.dRel = 2.0 - frame * 0.75
    projection = planner.project_fusion_lead(lead, slot=0, reference_ns=stamp)
    planner.update_fusion_fcw((projection,), now_ns=stamp)
    assert planner.crash_cnt == frame + 1
  previous_identity = planner.fcw_object_identity
  advance(lead, stamp + 50_000_000)
  lead.dRel = 8.0
  projection = planner.project_fusion_lead(lead, slot=0, reference_ns=stamp + 50_000_000)
  planner.update_fusion_fcw((projection,), now_ns=stamp + 50_000_000)
  assert planner.crash_cnt == 1
  assert planner.fcw_object_identity != previous_identity


@pytest.mark.parametrize('defect', [
  'too_many', 'nan_velocity', 'nan_window', 'backwards_window', 'stale', 'initial_state_too_old', 'inconsistent_display',
])
def test_malformed_or_stale_admitted_object_rejects_before_the_native_solve(defect):
  lead = object_lead()
  state = fused_state(lead)
  now_ns = reference_ns = EPOCH_NS
  if defect == 'too_many':
    state.radarObjects = [object_lead(index + 1) for index in range(MAX_CONTROL_OBJECTS + 1)]
  elif defect == 'nan_velocity':
    lead.vLead = float('nan')
  elif defect == 'nan_window':
    lead.evidence.conflictEnd = float('nan')
  elif defect == 'backwards_window':
    lead.evidence.conflictStart, lead.evidence.conflictEnd = 2.0, 1.0
  elif defect == 'stale':
    now_ns = reference_ns = EPOCH_NS + 300_000_000
  elif defect == 'inconsistent_display':
    state.leadOne = deepcopy(lead)
    state.leadOne.dRel += 10.0
  else:
    reference_ns = EPOCH_NS - 300_000_000
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner.update(state, reference=MpcReference(20.0), now_ns=now_ns, reference_ns=reference_ns)
  assert planner.solution_status == MPC_INVALID_INPUT
  assert planner.diagnostic_solve_calls == 0
  assert planner.crash_cnt == 0
  assert np.max(planner.a_solution) <= 0.0


def test_complete_bounded_object_capacity_reaches_the_native_problem():
  radar = [object_lead(index + 1, distance=150.0 + index, speed=20.0) for index in range(32)]
  visual = [object_lead(source='vision', distance=200.0 + index, speed=20.0) for index in range(2)]
  visual[1].evidence.visionIndex = 1
  planner = solve(fused_state(*radar, *visual, display=radar[:2]), iterations=1)
  assert planner.diagnostic_last_obstacle_summary['objectCount'] == MAX_CONTROL_OBJECTS
  assert np.all(np.isfinite(planner.params))


@pytest.mark.parametrize('radar_distance,radar_speed,vision_distance,vision_speed', [
  (25.0, 2.0, 70.0, 15.0), (60.0, 12.0, 18.0, 1.0), (28.0, 3.0, 32.0, 5.0),
])
def test_fixed_independent_acceleration_minimum_comparison(
    radar_distance, radar_speed, vision_distance, vision_speed, record_property):
  from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import sample_reference_acceleration

  radar = object_lead(distance=radar_distance, speed=radar_speed, ego_speed=15.0)
  vision = object_lead(source='vision', distance=vision_distance, speed=vision_speed, ego_speed=15.0)
  radar_arm = solve(fused_state(radar), speed=15.0)
  vision_arm = solve(fused_state(vision), speed=15.0)
  joint = solve(fused_state(radar, vision), speed=15.0)
  # Match the production UNIFIED sampling and physical clip. The older
  # velocity-inverse helper has different semantics and is diagnostic only.
  candidates = [float(np.clip(sample_reference_acceleration(planner.a_solution, 0.55), ACCEL_MIN, ACCEL_MAX))
                for planner in (radar_arm, vision_arm, joint)]
  radar_accel, vision_accel, joint_accel = candidates
  minimum = min(radar_accel, vision_accel)
  difference = joint_accel - minimum
  record_property('candidate_sampling', 'UNIFIED acceleration interpolation at 0.55 s; physical ACCEL_MIN/MAX clip')
  record_property('independent_radar_acceleration', radar_accel)
  record_property('independent_vision_acceleration', vision_accel)
  record_property('joint_acceleration', joint_accel)
  record_property('joint_minus_independent_minimum', difference)
  record_property('legacy_velocity_inverse_radar', action(radar_arm))
  record_property('legacy_velocity_inverse_vision', action(vision_arm))
  record_property('legacy_velocity_inverse_joint', action(joint))
  # This fixed tolerance checks these concrete braking cases only. It is not
  # an equivalence proof for nonlinear solves, all controls or other scenes.
  assert joint_accel < 0.0 and minimum < 0.0
  assert difference <= 0.05
