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
from openpilot.selfdrive.controls.lib.lead_qualification import MAX_CONTROL_OBJECTS, qualified_lead
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_STOP_DISTANCE, CRASH_DISTANCE, MODEL_V_REF, MPC_INVALID_INPUT, MPC_OCCUPANCY_UNSATISFIED, N,
  OCCUPANCY_ACCELERATION_REINTEGRATION_LIMIT, OCCUPANCY_POSITION_BUFFER, OCCUPANCY_POSITION_REINTEGRATION_LIMIT,
  OCCUPANCY_VELOCITY_REINTEGRATION_LIMIT, QP_FEASIBILITY_TOL, REFERENCE_BOUND, REST_STATE_TOL,
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


def coast_from(lead, stamp):
  coast = deepcopy(lead)
  coast.dRel += coast.vRel * (stamp - coast.evidence.estimateMonoTime) / 1e9
  coast.aLeadK, coast.aLeadTau = 0.0, 0.3
  evidence = coast.evidence
  evidence.source, evidence.coasting, evidence.accelerationValid = 'predicted', True, False
  evidence.estimateMonoTime = evidence.modelMonoTime = stamp
  evidence.followQualified = evidence.stopQualified = evidence.departureQualified = evidence.fcwQualified = False
  evidence.physicalConflictValid = False
  # The actual producer retains HORIZON for an object without a known exit.
  # It does not manufacture an observed cutout solely because CAN was lost.
  return coast


@pytest.mark.parametrize('distance,speed,ego_speed', [(30.0, 5.0, 15.0), (5.0, 0.0, 15.0)])
def test_known_following_coast_preserves_braking_without_new_authority(distance, speed, ego_speed):
  lead = object_lead(distance=distance, speed=speed, ego_speed=ego_speed)
  fresh_planner = solve(fused_state(lead), speed=ego_speed)
  coast_planner = solve(fused_state(lead), speed=ego_speed)
  stamp = EPOCH_NS + 50_000_000
  coast = coast_from(lead, stamp)
  retained = deepcopy(vars(coast))
  fresh = deepcopy(coast)
  fresh.evidence = deepcopy(lead.evidence)
  advance(fresh, stamp)
  fresh.aLeadTau = lead.aLeadTau
  for planner, current in ((fresh_planner, fresh), (coast_planner, coast)):
    planner.update(fused_state(current), reference=MpcReference(20.0), now_ns=stamp, reference_ns=stamp)
    assert planner.solution_status == planner.last_solver_status == 0 and planner.previous_solution_valid
  assert vars(coast) == retained
  assert qualified_lead(coast, 'brake', 0.0, now_ns=stamp)
  assert all(not qualified_lead(coast, purpose, 0.0, now_ns=stamp) for purpose in ('follow', 'stop', 'departure', 'fcw'))
  projection, = coast_planner._fusion_projections
  assert projection.persistent and projection.conflict_end == 9.95
  assert coast.evidence.conflictEnd == 10.0
  assert all(bound is None for bound in coast_planner._occupancy_bounds)
  assert coast_planner.rest_recovery_position_limit is None and coast_planner.fcw_object_identity is None
  np.testing.assert_array_equal(coast_planner.params, fresh_planner.params)
  np.testing.assert_array_equal(coast_planner.x_sol, fresh_planner.x_sol)
  assert action(coast_planner) < 0.0
  if distance == 5.0:
    # An unavoidable close target still demands substantial mitigation; the
    # cache must not replace failed hard geometry with a weak/no-brake plan.
    assert action(coast_planner) <= -1.0
    assert np.min(coast_planner.a_solution[1:]) <= -3.0


@pytest.mark.parametrize('change', ['unseen', 'generation', 'session', 'track', 'position', 'velocity', 'finite_exit', 'entering'])
def test_follow_geometry_never_softens_unknown_or_changed_coast(change):
  lead = object_lead(distance=20.0, speed=5.0)
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  if change != 'unseen':
    planner.fusion_lead_projections(fused_state(lead), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  stamp = EPOCH_NS + 50_000_000
  coast = coast_from(lead, stamp)
  if change == 'generation':
    coast.evidence.generation += 1
  elif change == 'session':
    coast.evidence.sessionId += '-new'
  elif change == 'track':
    coast.evidence.trackId += 1
  elif change == 'position':
    coast.evidence.positionMonoTime -= 1
  elif change == 'velocity':
    coast.evidence.velocityMonoTime -= 1
  elif change == 'finite_exit':
    coast.evidence.conflictEnd = 9.95
  elif change == 'entering':
    coast.evidence.pathState = 'entering'
  projection, = planner.fusion_lead_projections(fused_state(coast), now_ns=stamp, reference_ns=stamp)
  assert not projection.persistent
  assert any(bound is not None for bound in planner.finite_occupancy_bounds((projection,)))
  assert planner._fusion_follow_geometry == {}


@pytest.mark.parametrize('interruption', ['absent', 'out', 'reset', 'finite_fresh', 'clock_reversal'])
def test_follow_geometry_is_not_recovered_after_its_continuity_is_revoked(interruption):
  lead = object_lead(distance=20.0, speed=5.0)
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner.fusion_lead_projections(fused_state(lead), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  stamp = EPOCH_NS + 50_000_000
  interrupted = coast_from(lead, stamp)
  if interruption == 'reset':
    planner.reset()
    planner.set_cur_state(10.0, 0.0)
  elif interruption == 'absent':
    planner.fusion_lead_projections(fused_state(), now_ns=stamp, reference_ns=stamp)
  elif interruption == 'out':
    interrupted.evidence.pathState, interrupted.evidence.brakeQualified = 'outOfPath', False
    assert planner.fusion_lead_projections(fused_state(interrupted), now_ns=stamp, reference_ns=stamp) == ()
  elif interruption == 'finite_fresh':
    interrupted.evidence.source, interrupted.evidence.coasting = 'radar', False
    interrupted.evidence.conflictEnd = 1.0
    planner.fusion_lead_projections(fused_state(interrupted), now_ns=stamp, reference_ns=stamp)
  else:
    planner.fusion_lead_projections(fused_state(interrupted), now_ns=stamp, reference_ns=stamp)
    stamp -= 1
  coast = coast_from(lead, stamp)
  projection, = planner.fusion_lead_projections(fused_state(coast), now_ns=stamp, reference_ns=stamp)
  assert not projection.persistent and planner._fusion_follow_geometry == {}


def test_follow_geometry_keeps_all_objects_and_never_renews_its_fresh_horizon():
  lead = object_lead(distance=20.0, speed=5.0)
  crossing = object_lead(2, distance=15.0, speed=0.0, start=0.5, end=1.2, path='entering', persistent=False)
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner.fusion_lead_projections(fused_state(lead, crossing), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  for offset in (50_000_000, 100_000_000, 300_000_000):
    stamp = EPOCH_NS + offset
    next_crossing = deepcopy(crossing)
    advance(next_crossing, stamp)
    projections = planner.fusion_lead_projections(fused_state(coast_from(lead, stamp), next_crossing), now_ns=stamp, reference_ns=stamp)
    assert len(projections) == 2
    follow, finite = projections
    assert follow.identity == ('radar-test', 1) and finite.identity == ('radar-test', 2)
    assert follow.persistent and follow.conflict_end == 10.0-offset/1e9
    assert not finite.persistent and any(bound is not None for bound in planner.finite_occupancy_bounds(projections))
  stamp = EPOCH_NS + 350_000_001
  with pytest.raises(ValueError, match='Unqualified admitted'):
    planner.fusion_lead_projections(fused_state(coast_from(lead, stamp)), now_ns=stamp, reference_ns=stamp)


def test_conflicting_coast_generation_views_invalidate_follow_geometry():
  lead = object_lead(distance=20.0, speed=5.0)
  planner = solve(fused_state(lead))
  stamp = EPOCH_NS + 50_000_000
  coast = coast_from(lead, stamp)
  display = deepcopy(coast)
  display.evidence.generation += 1
  solves = planner.diagnostic_solve_calls
  planner.update(fused_state(coast, display=(display,)), reference=MpcReference(20.0), now_ns=stamp, reference_ns=stamp)
  assert planner.solution_status == MPC_INVALID_INPUT and planner._fusion_follow_geometry == {}
  assert planner.diagnostic_solve_calls == solves and planner.object_influence_identities == ()


def test_recorded_track57_fresh_geometry_survives_slot_reassignment_coast():
  # ed7d1f7 full89: fusion_degradation.samples[8], frames 3511/3512;
  # failure_diagnostics frames 3513/3514. Identity, clocks, position/speed and
  # windows are saved there. Compact fresh memory omitted aLeadK, so these
  # two setup frames use a declared zero-acceleration hypothesis; this is a
  # geometry-transition regression, not a replay of the missing fresh fields.
  planner = LongitudinalMpc()
  session = 'e48bcdb9f5e7427196849f7be202bc9f'
  for estimate, measurement, model, publication, reference, distance, speed, ego, acceleration, rate in (
    (199161575308, 199176107110, 199204260183, 199220664037, 199216790808,
     2.868521163319816, 10.042173635095105, 16.72817039489746, 0.18555878102779388, -0.05716469660830151),
    (199211575829, 199235791485, 199254899454, 199270828412, 199267106589,
     2.5880764259541107, 10.010508823782994, 16.71938133239746, 0.02808774635195732, -0.03496408699392124),
  ):
    lead = object_lead(57, distance=float(np.float32(distance)), speed=float(np.float32(speed)), ego_speed=ego, epoch=estimate)
    lead.evidence.sessionId, lead.evidence.generation = session, 128
    lead.evidence.positionMonoTime = lead.evidence.velocityMonoTime = measurement
    lead.evidence.modelMonoTime, lead.evidence.lateralSpeed = model, rate
    lead.evidence.pathSource = 'modelPath'
    planner.set_cur_state(ego, acceleration)
    planner.set_weights(prev_accel_constraint=False)
    planner.update(fused_state(lead, display=(lead,)), reference=MpcReference(20.0), now_ns=publication, reference_ns=reference)
    assert planner.solution_status == 0 and planner._fusion_projections[0].persistent
  coast = coast_from(lead, 199261579735)
  coast.dRel, coast.yRel, coast.vRel = 2.2505342960357666, -0.8906638026237488, -6.711284160614014
  coast.vLead = coast.vLeadK = 10.010696411132812
  coast.aLeadTau = 0.30000001192092896
  coast.evidence.modelMonoTime = 199304902006
  coast.evidence.pathOffset, coast.evidence.lateralSpeed = -0.9212847948074341, -0.04149603471159935
  coast.evidence.positionStd, coast.evidence.velocityStd = 0.24600131809711456, 0.8504881262779236
  before = deepcopy(vars(coast))
  for publication, reference, ego, acceleration in (
    (199313753568, 199307159714, 16.721981048583984, 0.05395398288965225),
    (199362946693, 199357071537, 16.74215316772461, 0.17761588096618652),
  ):
    planner.set_cur_state(ego, acceleration)
    planner.set_weights(prev_accel_constraint=False)
    planner.update(fused_state(coast, display=(coast,)), reference=MpcReference(20.0), now_ns=publication, reference_ns=reference)
    assert planner.solution_status == planner.last_solver_status == 0 and planner.previous_solution_valid
    projection, = planner._fusion_projections
    assert projection.persistent and not any(planner._occupancy_bounds)
    assert projection.conflict_end == 10.0 - (reference-199211575829)/1e9
    assert vars(coast) == before and coast.evidence.conflictEnd == 10.0
    assert planner.rest_recovery_position_limit is None and planner.fcw_object_identity is None
    assert action(planner) <= -1.0 and np.min(planner.a_solution[1:]) <= -3.0


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


def moving_window_clearance(planner, projection):
  """Independent polynomial-root check of each continuous relative-position segment."""
  minimum = float('inf')
  for stage, jerk in enumerate(planner.j_solution):
    left, right = T_IDXS[stage:stage+2]
    lo, hi = max(projection.conflict_start, left)-left, min(projection.conflict_end, right)-left
    if hi < lo:
      continue
    position, velocity, acceleration = planner.x_sol[stage]
    lead_position = projection.raw_trajectory[stage, 0]
    lead_velocity = (projection.raw_trajectory[stage+1, 0]-lead_position)/(right-left)
    coefficients = [lead_position-position, lead_velocity-velocity, -acceleration/2.0, -jerk/6.0]
    extrema = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(coefficients))
    times = [lo, hi, *(float(root.real) for root in extrema if abs(root.imag) < 1e-10 and lo <= root.real <= hi)]
    minimum = min(minimum, float(np.min(np.polynomial.polynomial.polyval(times, coefficients))))
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


@pytest.mark.parametrize('speed,a0,distance,lead_speed,lead_acceleration,end,alignment_ns', [
  (16.686721801757812, -0.08589210361242294, 10.332793235778809, 10.936553001403809,
   0.45718914270401, 1.340597152709961, 100621282),
  (16.70214080810547, 0.012393898330628872, 9.821605682373047, 10.865662574768066,
   0.28191542625427246, 1.2554775476455688, 100354875),
  (16.756868362426758, 0.24996110796928406, 7.364883899688721, 10.652046203613281,
   -0.10135208815336227, 0.7025191783905029, 42423000),
  (16.72172737121582, -0.11459379643201828, 7.364883899688721, 10.652046203613281,
   -0.10135208815336227, 0.7025191783905029, 91693885),
  (16.72758674621582, -0.066969133913517, 6.682292461395264, 10.601381301879883,
   -0.1549449861049652, 0.6321916580200195, 42437271),
  (16.70784568786621, -0.11034934222698212, 6.121922969818115, 10.456089973449707,
   -0.39006689190864563, 0.5813595652580261, 44257479),
  (16.70939826965332, 0.009480171836912632, 5.621070861816406, 10.334680557250977,
   -0.4930892586708069, 0.6480633616447449, 44101177),
  (16.7175350189209, -0.05677497386932373, 5.111467361450195, 10.274609565734863,
   -0.49771347641944885, 0.7227144241333008, 43521542),
])
def test_retained_moving_target_windows_admit_continuously_safe_behind_plans(
    speed, a0, distance, lead_speed, lead_acceleration, end, alignment_ns):
  # Fixed geometry from a85d0e7 / route 89 track 57, expanded samples 7..14.
  # The old earliest-lead/latest-ego boxes were physically infeasible; these
  # are cold native solves of the same geometry, not a replay of its QP history.
  crossing = object_lead(57, distance=distance, speed=lead_speed, ego_speed=speed, acceleration=lead_acceleration,
                         end=end, persistent=False)
  reference_ns = EPOCH_NS + alignment_ns
  planner = solve(fused_state(crossing), speed=speed, acceleration=a0, iterations=3,
                  now_ns=reference_ns, reference_ns=reference_ns)
  projection, = planner._fusion_projections
  assert moving_window_clearance(planner, projection) >= CRASH_DISTANCE
  assert planner.trajectory_minimum_velocity(planner.x_sol, planner.j_solution) >= -QP_FEASIBILITY_TOL
  assert np.min(planner.a_solution[1:]) >= ACCEL_MIN - QP_FEASIBILITY_TOL
  assert np.max(planner.a_solution[1:]) <= ACCEL_MAX + QP_FEASIBILITY_TOL
  h = T_IDXS[1]
  first_position = speed*h + (2.0*a0+ACCEL_MIN)*h*h/6.0
  first_velocity = speed + (a0+ACCEL_MIN)*h/2.0
  old_box_impossible = False
  for stage, bound in enumerate(planner._occupancy_bounds):
    if bound is None:
      continue
    start, finish = max(T_IDXS[stage], projection.conflict_start), min(T_IDXS[stage+1], projection.conflict_end)
    old_limit = np.min(np.interp([start, finish], T_IDXS, projection.raw_trajectory[:, 0]))
    old_limit -= CRASH_DISTANCE + OCCUPANCY_POSITION_BUFFER
    elapsed = T_IDXS[stage] + bound.duration - h
    assert elapsed >= 0.0 and first_velocity + ACCEL_MIN*elapsed > 0.0
    minimum_position = first_position + first_velocity*elapsed + ACCEL_MIN*elapsed**2/2.0
    old_box_impossible |= minimum_position > old_limit
  assert old_box_impossible


def test_transport_does_not_admit_the_next_retained_infeasible_window():
  speed, a0 = 16.70136260986328, -0.15578755736351013
  crossing = object_lead(57, distance=4.528200626373291, speed=10.232539176940918, ego_speed=speed,
                         acceleration=-0.5102481842041016, end=0.7762875556945801, persistent=False)
  reference_ns = EPOCH_NS + 44_256_125
  planner = LongitudinalMpc()
  planner.set_cur_state(speed, a0)
  planner.set_weights(prev_accel_constraint=False)
  planner.update(fused_state(crossing), reference=MpcReference(20.0), now_ns=reference_ns, reference_ns=reference_ns)
  assert planner.solution_status != 0 and not planner.previous_solution_valid
  assert planner.finite_risk_braking_accel is None


@pytest.mark.parametrize('lead_speed', [0.0, -0.25])
def test_stationary_and_approaching_targets_keep_their_original_single_object_bound(lead_speed):
  lead = object_lead(distance=30.0, speed=lead_speed, ego_speed=15.0, persistent=False)
  planner = LongitudinalMpc()
  planner.set_cur_state(15.0, 0.0)
  projection = planner.project_fusion_lead(lead, slot=None, reference_ns=EPOCH_NS)
  bounds = planner.finite_occupancy_bounds((projection,))
  for stage, bound in enumerate(bounds):
    assert bound is not None
    original = min(projection.raw_trajectory[stage:stage+2, 0]) - CRASH_DISTANCE - OCCUPANCY_POSITION_BUFFER
    assert bound.position_limit == pytest.approx(original, abs=1e-12, rel=0.0)


@pytest.mark.parametrize('a0', [-5.0, ACCEL_MIN, 1.0])
def test_transport_velocity_floor_includes_the_actual_initial_acceleration(a0):
  lead = object_lead(distance=0.8, speed=12.0, start=0.0, end=0.03, path='entering', persistent=False)
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, a0)
  projection = planner.project_fusion_lead(lead, slot=None, reference_ns=EPOCH_NS)
  limit = planner.occupancy_behind_position_limit(projection, 0, constraint_time=0.03)
  # Lead speed exceeds the velocity lower bound, so the earliest instant is
  # the analytic minimum. A fixed -3.5 assumption would overgrant when a0=-5.
  expected = 0.8 + 10.0*0.03 + min(ACCEL_MIN, a0)*0.03**2/2.0 - CRASH_DISTANCE - OCCUPANCY_POSITION_BUFFER
  assert limit == pytest.approx(expected, abs=1e-12, rel=0.0)
  with pytest.raises(ValueError, match='later in-stage row'):
    planner.occupancy_behind_position_limit(projection, 0, constraint_time=0.02)


def test_transport_bound_includes_the_interior_relative_velocity_extremum():
  lead = object_lead(distance=10.0, speed=13.0, ego_speed=15.0, start=0.3, end=0.6, path='entering', persistent=False)
  planner = LongitudinalMpc()
  planner.set_cur_state(15.0, 0.0)
  projection = planner.project_fusion_lead(lead, slot=None, reference_ns=EPOCH_NS)
  stage = int(np.searchsorted(T_IDXS, 0.3)-1)
  limit = planner.occupancy_behind_position_limit(projection, stage, constraint_time=0.6)
  extremum = (13.0-15.0)/ACCEL_MIN
  assert 0.3 < extremum < 0.6
  expected = (10.0 + 13.0*extremum + 15.0*(0.6-extremum) + ACCEL_MIN*(0.6**2-extremum**2)/2.0
              - CRASH_DISTANCE - OCCUPANCY_POSITION_BUFFER)
  assert limit == pytest.approx(expected, abs=1e-12, rel=0.0)
  endpoint_only = min(10.0 + 13.0*t + 15.0*(0.6-t) + ACCEL_MIN*(0.6**2-t*t)/2.0 for t in (0.3, 0.6))
  assert limit < endpoint_only - CRASH_DISTANCE - OCCUPANCY_POSITION_BUFFER - 1e-4


@pytest.mark.parametrize('change', [
  'projection', 'raw_array', 'raw_position', 'raw_velocity', 'x0_position', 'x0_velocity', 'x0_acceleration', 'stage', 'time',
])
def test_coast_bound_reuse_requires_the_complete_exact_input(change, monkeypatch):
  planner = LongitudinalMpc()
  planner.set_cur_state(15.0, -0.2)
  lead = object_lead(distance=30.0, speed=13.0, ego_speed=15.0, end=0.55, persistent=False)
  lead.evidence.source, lead.evidence.coasting, lead.evidence.fcwQualified = 'predicted', True, False
  projection = planner.project_fusion_lead(lead, slot=None, reference_ns=EPOCH_NS)
  stage, timestamp = 2, 0.6
  calls = []
  original = planner.ego_progress_bound

  def count_progress(*args):
    calls.append(args)
    return original(*args)

  monkeypatch.setattr(planner, 'ego_progress_bound', count_progress)
  first = planner.occupancy_behind_position_limit(projection, stage, constraint_time=timestamp)
  evaluated = len(calls)
  assert evaluated > 0
  assert planner.occupancy_behind_position_limit(projection, stage, constraint_time=timestamp) == first
  assert len(calls) == evaluated
  if change == 'projection':
    projection = replace(projection)
  elif change == 'raw_array':
    projection = replace(projection, raw_trajectory=projection.raw_trajectory.copy())
  elif change.startswith('raw_'):
    # Even an unconsumed field in the complete mutable trajectory invalidates
    # reuse; neither object identity nor a sampled row proves unchanged input.
    column = 0 if change == 'raw_position' else 1
    projection.raw_trajectory[stage, column] += 0.01
  elif change.startswith('x0_'):
    index = ('x0_position', 'x0_velocity', 'x0_acceleration').index(change)
    planner.x0[index] = np.nextafter(planner.x0[index], np.inf)
  elif change == 'stage':
    stage, timestamp = 1, T_IDXS[2]
  else:
    timestamp = np.nextafter(timestamp, np.inf)
  actual = planner.occupancy_behind_position_limit(projection, stage, constraint_time=timestamp)
  assert len(calls) > evaluated
  cold = LongitudinalMpc()
  cold.x0[:] = planner.x0
  assert actual == cold.occupancy_behind_position_limit(projection, stage, constraint_time=timestamp)


@pytest.mark.parametrize('invalidation', ['update', 'reset', 'fail', 'reference', 'state'])
def test_occupancy_bound_reuse_ends_with_the_current_problem(invalidation):
  planner = LongitudinalMpc()
  planner.set_cur_state(15.0, 0.0)
  lead = object_lead(distance=30.0, speed=13.0, ego_speed=15.0, end=0.55, persistent=False)
  projection = planner.project_fusion_lead(lead, slot=None, reference_ns=EPOCH_NS)
  planner.occupancy_behind_position_limit(projection, 2, constraint_time=0.6)
  prior = planner._occupancy_limit_cache
  assert prior
  if invalidation == 'update':
    planner.update(fused_state(), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  elif invalidation == 'reset':
    planner.reset()
  elif invalidation == 'fail':
    planner.fail(MPC_INVALID_INPUT)
  elif invalidation == 'reference':
    planner.invalidate_reference_history()
  else:
    planner.set_cur_state(15.0, 0.0)
  assert planner._occupancy_limit_cache == {}
  assert planner._occupancy_limit_cache is not prior


def test_tied_finite_objects_reuse_only_their_own_current_geometry_for_attribution(monkeypatch):
  planner = LongitudinalMpc()
  planner.set_cur_state(0.0, 0.0)
  planner.set_weights(prev_accel_constraint=False)
  # Isolate the real hard occupancy rows with a valid native jerk objective.
  # Neither soft lead cost may attribute a token after its raw row is moved.
  cost_weights, constraint_weights = planner._cost_weights.copy(), planner._constraint_cost_weights.copy()
  cost_weights[0], constraint_weights[-1] = 0.0, 0.0
  planner.set_cost_weights(cost_weights, constraint_weights)
  objects = [object_lead(track, distance=CRASH_DISTANCE+OCCUPANCY_POSITION_BUFFER,
                         speed=0.0, ego_speed=0.0, persistent=False) for track in (1, 2)]
  planner.update(fused_state(*objects), reference=MpcReference(0.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status == planner.last_solver_status == 0
  assert planner.previous_solution_valid and planner.diagnostic_solve_calls == 1
  assert all(bound is not None for bound in planner._occupancy_bounds)
  assert np.all(planner.diagnostic_reference_weights[:, 0] == 0.0)
  assert planner.occupancy_trajectory_error(planner.x_sol, planner.j_solution, tolerance=QP_FEASIBILITY_TOL) == ''
  calls = []
  original = planner.ego_progress_bound

  def count_progress(*args):
    calls.append(args)
    return original(*args)

  monkeypatch.setattr(planner, 'ego_progress_bound', count_progress)
  planner.record_object_influence()
  assert set(planner.object_influence_identities) == {('radar-test', 1), ('radar-test', 2)}
  assert not calls
  planner._fusion_projections[1].raw_trajectory[:, 0] += 1.0
  planner.record_object_influence()
  assert planner.object_influence_identities == (('radar-test', 1),)
  assert calls


@pytest.mark.parametrize('horizon', [5.0, 10.0])
@pytest.mark.parametrize('case', ['rest', 'constant_acceleration', 'two_roots', 'no_roots'])
def test_position_only_clearance_preserves_exact_dynamics_and_continuous_extrema(case, horizon):
  planner = LongitudinalMpc()
  lead = object_lead(distance=40.0, speed=2.0, persistent=False)
  projection = planner.project_fusion_lead(lead, slot=None, reference_ns=EPOCH_NS)
  states = np.zeros((N+1, 3))
  jerks = np.zeros(N)
  candidates = []
  for stage, duration in enumerate(np.diff(T_IDXS)):
    lead_velocity = (projection.raw_trajectory[stage+1, 0] - projection.raw_trajectory[stage, 0]) / duration
    states[stage] = [3.0, lead_velocity, 0.0]
    times = [0.0, min(duration, horizon-T_IDXS[stage])]
    if case == 'rest':
      states[stage, 1] = 0.0
    elif case == 'constant_acceleration':
      states[stage, 1:] = lead_velocity-duration/2.0, 1.0
      times.append((lead_velocity-states[stage, 1])/states[stage, 2])
    elif case == 'two_roots':
      # Relative velocity is -(t-duration/4)*(t-3*duration/4).
      states[stage, 1:] = lead_velocity+3.0*duration**2/16.0, -duration
      jerks[stage] = 2.0
      # Use the same actual floating inputs to the independent dynamics
      # evaluator; retain both quadratic roots even when only one is extremal.
      discriminant = states[stage, 2]**2 + 2.0*jerks[stage]*(lead_velocity-states[stage, 1])
      times.extend(((-states[stage, 2]-np.sqrt(discriminant))/jerks[stage],
                    (-states[stage, 2]+np.sqrt(discriminant))/jerks[stage]))
    else:
      states[stage, 1] = lead_velocity+1.0
      jerks[stage] = 2.0
    if T_IDXS[stage] >= horizon:
      continue
    for elapsed in times:
      if 0.0 <= elapsed <= min(duration, horizon-T_IDXS[stage]):
        position = LongitudinalMpc.integrate_jerk(states[stage], jerks[stage], elapsed)[0]
        candidates.append(float(projection.raw_trajectory[stage, 0] + lead_velocity*elapsed-position))
  actual = planner.occupancy_clearance_extrema(projection, states, jerks, horizon=horizon)
  assert actual == (min(candidates), max(candidates))


@pytest.mark.parametrize('horizon, expected', [(10.0, (-80.0, 120.0)), (float(T_IDXS[6]), (20.0, 120.0))])
def test_clearance_window_traversal_keeps_closed_node_endpoints(horizon, expected):
  planner = LongitudinalMpc()
  lead = object_lead(distance=20.0, speed=0.0, ego_speed=0.0, persistent=False)
  projection = planner.project_fusion_lead(lead, slot=None, reference_ns=EPOCH_NS)
  projection = replace(projection, conflict_start=float(T_IDXS[4]), conflict_end=float(T_IDXS[6]))
  states, jerks = np.zeros((N+1, 3)), np.zeros(N)
  # Deliberate node discontinuities make each boundary side independently
  # observable here. Dynamic continuity has its own acceptance check; this
  # geometric helper must retain the original closed-window endpoint rules.
  states[3, 0] = -100.0  # Its endpoint is exactly the occupied window's start.
  states[6, 0] = 100.0   # Its start is exactly the occupied window's end.
  assert planner.occupancy_clearance_extrema(projection, states, jerks, horizon=horizon) == expected


@pytest.mark.parametrize('with_rest', [False, True])
def test_multiple_end_times_share_one_order_independent_row_with_the_original_rest_goal(with_rest):
  early = object_lead(1, distance=2.0, speed=0.5, ego_speed=1.0, start=0.03, end=0.10, path='entering', persistent=False)
  later = object_lead(2, distance=3.0, speed=0.75, ego_speed=1.0, start=0.07, end=0.20, path='entering', persistent=False)
  reference = rest_reference(1.0, 6) if with_rest else MpcReference(20.0)
  forward = solve(fused_state(early, later), speed=1.0, reference=reference, iterations=3)
  reverse = solve(fused_state(later, early), speed=1.0, reference=reference, iterations=3)
  np.testing.assert_array_equal(forward.params, reverse.params)
  np.testing.assert_allclose(forward.x_sol, reverse.x_sol, atol=1e-9, rtol=0.0)
  for a, b in zip(forward._occupancy_bounds, reverse._occupancy_bounds, strict=True):
    assert (a is None) == (b is None)
    if a is not None:
      assert (a.duration, a.position_limit, a.position_lower_limit) == (b.duration, b.position_limit, b.position_lower_limit)
      assert set(a.identities) == set(b.identities)
  assert forward._occupancy_bounds[1].duration == pytest.approx(0.20-T_IDXS[1])
  for projection in forward._fusion_projections:
    assert moving_window_clearance(forward, projection) >= CRASH_DISTANCE
  if with_rest:
    assert forward.rest_goal_state == reverse.rest_goal_state == 'satisfied'
    assert np.max(np.abs(forward.x_sol[6:, 1:])) <= REST_STATE_TOL


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


@pytest.mark.parametrize('case', ['coasting', 'stale', 'bad_input', 'caution', 'visual', 'late_passing'])
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
  elif case == 'late_passing':
    state = fused_state(object_lead(distance=5.0, start=4.0, end=6.0, path='entering', persistent=False, physical_window=(4.0, 6.0)))
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner.set_weights(prev_accel_constraint=False)
  planner.update(state, reference=MpcReference(20.0), now_ns=now_ns, reference_ns=reference_ns)
  assert planner.finite_risk_braking_accel is None and planner.finite_risk_evidence == ()


@pytest.mark.parametrize('degraded', [False, True])
def test_unrelated_fusion_diagnostic_does_not_remove_a_fresh_physical_risk(degraded):
  crossing = object_lead(distance=0.8, start=0.02, end=0.10, path='entering', persistent=False,
                         physical_window=(0.02, 0.10))
  state = fused_state(crossing)
  state.fusionDegraded = degraded
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner.set_weights(prev_accel_constraint=False)
  planner.update(state, reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.finite_risk_braking_accel == ACCEL_MIN
  assert planner.finite_risk_evidence


def test_faster_target_no_longer_fails_the_old_box_and_has_no_emergency_proof():
  crossing = object_lead(distance=0.8, speed=12.0, start=0.0, end=0.10, path='entering', persistent=False,
                         physical_window=(0.0, 0.10))
  planner = solve(fused_state(crossing), iterations=3)
  assert moving_window_clearance(planner, planner._fusion_projections[0]) >= CRASH_DISTANCE
  assert planner.finite_risk_braking_accel is None and planner.finite_risk_evidence == ()


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
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_bounded_state_reintegration_preserves_native_jerk_and_all_constraints(with_rest, axis):
  speed = 1.0 if with_rest else 10.0
  reference = rest_reference(speed, 6) if with_rest else MpcReference(20.0)
  crossing = object_lead(distance=6.0 if with_rest else 20.0, ego_speed=speed,
                         start=0.7 if with_rest else 1.2, end=0.85 if with_rest else 2.6,
                         path='entering', persistent=False)
  state = fused_state(crossing)
  planner = solve(state, speed=speed, reference=reference)
  native = planner.solver

  budget = np.array([OCCUPANCY_POSITION_REINTEGRATION_LIMIT, OCCUPANCY_VELOCITY_REINTEGRATION_LIMIT,
                     OCCUPANCY_ACCELERATION_REINTEGRATION_LIMIT])

  class StateNoise(NativeProblemRecorder):
    def get(self, stage, field):
      result = super().get(stage, field)
      if stage == N and field == 'x':
        result = result.copy()
        result[axis] += 0.5 * budget[axis]
      return result

  planner.solver = StateNoise(native)
  initial = planner.x0.copy()
  planner.update(state, reference=reference, now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status == planner.last_solver_status == 0
  diagnostic = planner.diagnostic_occupancy_projection
  assert diagnostic['applied'] and diagnostic['maximumJerkCorrection'] == 0.0
  assert np.all(diagnostic['maximumStateCorrection'] <= budget)
  assert np.all(diagnostic['maximumContinuousStateCorrectionBound'] <= budget)
  assert diagnostic['stateCorrectionUnits'] == ['m', 'm/s', 'm/s^2']
  np.testing.assert_array_equal(planner.x_sol[0], initial)
  np.testing.assert_array_equal(planner.j_solution, [native.get(stage, 'u')[0] for stage in range(N)])
  assert np.max(np.abs(planner.trajectory_dynamics_defects(planner.x_sol, planner.j_solution))) <= QP_FEASIBILITY_TOL
  assert planner.occupancy_trajectory_error(planner.x_sol, planner.j_solution, tolerance=QP_FEASIBILITY_TOL) == ''
  if with_rest:
    assert planner.rest_goal_state == 'satisfied'
    assert np.max(np.abs(planner.x_sol[6:, 1:])) <= REST_STATE_TOL


@pytest.mark.parametrize('kind', ['position', 'meter_position', 'velocity', 'acceleration', 'cumulative_position', 'continuous_position'])
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
        elif kind == 'continuous_position' and stage > 0:
          result[0] -= 0.95 * OCCUPANCY_POSITION_REINTEGRATION_LIMIT
          result[1] -= 0.9 * OCCUPANCY_VELOCITY_REINTEGRATION_LIMIT
        elif stage == N:
          if kind == 'velocity':
            result[1] += 2.0 * OCCUPANCY_VELOCITY_REINTEGRATION_LIMIT
          elif kind == 'acceleration':
            result[2] += 2.0 * OCCUPANCY_ACCELERATION_REINTEGRATION_LIMIT
          else:
            result[0] += 2.0 * OCCUPANCY_POSITION_REINTEGRATION_LIMIT if kind == 'position' else 1.0
      return result

  planner.solver = NativeNoise(planner.solver)
  planner.update(state, reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.solution_status == MPC_OCCUPANCY_UNSATISFIED
  expected = ('finite_occupancy_reintegration_budget_exceeded' if kind in ('cumulative_position', 'continuous_position')
              else 'finite_occupancy_dynamics_residual_exceeded')
  assert planner.failure_reason == expected
  assert not planner.diagnostic_occupancy_projection['applied']
  assert not planner.previous_solution_valid and planner.object_influence_identities == ()
  if kind == 'continuous_position':
    diagnostic = planner.diagnostic_occupancy_projection
    assert diagnostic['maximumStateCorrection'][0] < OCCUPANCY_POSITION_REINTEGRATION_LIMIT
    assert diagnostic['maximumContinuousStateCorrectionBound'][0] > OCCUPANCY_POSITION_REINTEGRATION_LIMIT


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


def test_bounded_reintegration_cannot_relax_a_real_installed_position_row():
  planner = LongitudinalMpc()
  planner.set_cur_state(0.0, 0.0)
  planner.set_weights(prev_accel_constraint=False)
  cost_weights, constraint_weights = planner._cost_weights.copy(), planner._constraint_cost_weights.copy()
  cost_weights[0], constraint_weights[-1] = 0.0, 0.0
  planner.set_cost_weights(cost_weights, constraint_weights)
  lead = object_lead(distance=CRASH_DISTANCE+OCCUPANCY_POSITION_BUFFER, speed=0.0, ego_speed=0.0, persistent=False)
  native = planner.solver

  class SmallUnsafeJerk(NativeProblemRecorder):
    def get(self, stage, field):
      result = super().get(stage, field)
      if field == 'u':
        result = result.copy()
        # All state defects/corrections stay inside the numerical budgets,
        # but forward creep breaches this problem's actual zero-position row.
        result[0] += 5e-8
      return result

  planner.solver = SmallUnsafeJerk(native)
  planner.update(fused_state(lead), reference=MpcReference(0.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert planner.diagnostic_last_solver_return == 0
  assert planner.solution_status == MPC_OCCUPANCY_UNSATISFIED
  assert planner.failure_reason == 'finite_occupancy_reintegration_constraints_violated'
  diagnostic = planner.diagnostic_occupancy_projection
  assert np.all(np.array(diagnostic['maximumContinuousStateCorrectionBound']) <= diagnostic['maximumStateCorrectionBudget'])
  assert diagnostic['maximumCorrectedConstraintViolation'] > QP_FEASIBILITY_TOL
  assert not diagnostic['applied'] and not planner.previous_solution_valid
  assert planner.object_influence_identities == ()


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


@pytest.mark.parametrize('speed,a0,distance,lead_speed,lead_acceleration,start,alignment_ns,expected_x0,expected_v0,expected_x10,accepted', [
  (4.188114166259766, -0.5062192678451538, 17.479167938232422, -0.0602441169321537, -2.5708842277526855,
   0.7778034210205078, 49068416, 17.26700520236193, -0.18627953748592974, 13.676217081056151, True),
  (4.152256011962891, -0.5448019504547119, 17.479167938232422, -0.0602441169321537, -2.5708842277526855,
   0.7778034210205078, 110202583, 16.9960673302231, -0.3422777626409075, 10.61000930564217, True),
  (1.0514822006225586, -0.18802042305469513, 5.570803165435791, -0.08574940264225006, -0.7474237680435181,
   0.10351847112178802, 93279354, 5.460660318684282, -0.15524186349386138, 3.509812642972538, True),
  (0.9412540197372437, -0.5520215630531311, 5.309693336486816, -0.12712086737155914, -0.5738447904586792,
   1.4378634691238403, 52973833, 5.251518269643627, -0.15748766942879147, 3.351951817979633, True),
  (1.5158843994140625, -0.3846769332885742, 7.0452561378479, -0.21278104186058044, -1.6375515460968018,
   1.0249863862991333, 49840499, 6.9565881676161725, -0.29432147086488114, 2.2758020435090476, True),
  (1.498730182647705, -0.3723600506782532, 7.0452561378479, -0.21278104186058044, -1.6375515460968018,
   1.0249863862991333, 99785969, 6.864484965295978, -0.3755778349965633, 1.0125149541543208, True),
])
def test_recorded_small_negative_motion_preserves_inputs_and_enforces_native_acceptance(
    speed, a0, distance, lead_speed, lead_acceleration, start, alignment_ns, expected_x0, expected_v0, expected_x10, accepted):
  from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import sample_reference_acceleration
  # a85d0e7 / route 8a samples 0,1,2,3,10,11. This checks the explicit motion
  # hypothesis and requested plan, not the object's measured future behavior.
  # Sample 1 used to exceed the 10 nm / 1 nm/s repair admission despite a
  # successful native solve. Its unchanged jerk must now pass bounded state
  # reintegration AND all original final constraints and physical clearances.
  lead = object_lead(distance=distance, speed=lead_speed, ego_speed=speed, acceleration=lead_acceleration,
                     start=start, end=10.0, path='entering', persistent=False)
  recorded = deepcopy(vars(lead))
  reference_ns = EPOCH_NS + alignment_ns
  planner = LongitudinalMpc()
  for iteration in range(3):
    # fail/reset clears x0 and restores default weights. Reinstall the SAME
    # actual problem; success at a reset zero state would not be recovery.
    planner.set_cur_state(speed, a0)
    planner.set_weights(prev_accel_constraint=False)
    projection = planner.project_fusion_lead(lead, slot=None, reference_ns=reference_ns)
    np.testing.assert_allclose(projection.raw_trajectory[0], [expected_x0, expected_v0], atol=1e-12, rtol=0.0)
    assert projection.raw_trajectory[-1, 0] == pytest.approx(expected_x10, abs=1e-10, rel=0.0)
    assert np.all(projection.raw_trajectory[:, 1] < 0.0)  # No stationary substitute for measured approach.
    assert np.any(projection.occupied)
    planner.update(fused_state(lead), reference=MpcReference(20.0), now_ns=reference_ns, reference_ns=reference_ns)
    assert vars(lead) == recorded
    assert planner.diagnostic_last_x0 == (0.0, speed, a0)
    assert planner.diagnostic_solve_calls == iteration + 1
    if accepted:
      assert planner.solution_status == planner.last_solver_status == 0 and planner.previous_solution_valid
      assert moving_window_clearance(planner, planner._fusion_projections[0]) >= CRASH_DISTANCE
      request = sample_reference_acceleration(planner.a_solution, 0.55)
      assert ACCEL_MIN-QP_FEASIBILITY_TOL <= request <= ACCEL_MAX+QP_FEASIBILITY_TOL
    else:
      assert planner.solution_status == planner.last_solver_status == MPC_OCCUPANCY_UNSATISFIED
      assert planner.diagnostic_last_solver_return == 0
      assert planner.failure_reason == 'finite_occupancy_dynamics_residual_exceeded'
      diagnostics = planner.diagnostic_occupancy_projection
      assert diagnostics is not None and diagnostics['applied'] is False
      defect = diagnostics['maximumNativeDynamicsDefect']
      assert defect[0] > OCCUPANCY_POSITION_REINTEGRATION_LIMIT and defect[1] > QP_FEASIBILITY_TOL
      assert not planner.previous_solution_valid and planner.object_influence_identities == ()
      assert planner.finite_risk_braking_accel is None and planner.finite_risk_evidence == ()


@pytest.mark.parametrize('lead_speed', [-5.0, np.nextafter(-2.0, -np.inf), -2.0, 0.0, 1e-6, 5.0])
@pytest.mark.parametrize('lead_acceleration', [-2.5, 0.0, 2.5])
def test_forward_and_clearly_opposing_motion_predictions_remain_exact(lead_speed, lead_acceleration):
  lead = object_lead(distance=30.0, speed=lead_speed, acceleration=lead_acceleration)
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  projection = planner.project_fusion_lead(lead, slot=None, reference_ns=EPOCH_NS)
  dt = np.diff(T_IDXS, prepend=0.0)
  acceleration = lead_acceleration*np.exp(-lead.aLeadTau*T_IDXS**2/2.0)
  expected_velocity = lead_speed + np.cumsum(dt*acceleration)
  if lead_speed >= 0.0:
    expected_velocity = np.maximum(expected_velocity, 0.0)
  expected_position = lead.dRel + np.cumsum(dt*expected_velocity)
  np.testing.assert_array_equal(projection.raw_trajectory, np.column_stack((expected_position, expected_velocity)))


@pytest.mark.parametrize('lead_speed', [-1.9, -1.0, -1e-6])
def test_positive_acceleration_can_depart_from_small_negative_speed_without_model_attenuation(lead_speed):
  lead = object_lead(distance=30.0, speed=lead_speed, acceleration=2.5)
  recorded = deepcopy(vars(lead))
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  projection = planner.project_fusion_lead(lead, slot=None, reference_ns=EPOCH_NS)
  dt = np.diff(T_IDXS, prepend=0.0)
  acceleration = lead.aLeadK*np.exp(-lead.aLeadTau*T_IDXS**2/2.0)
  expected_velocity = lead_speed + np.cumsum(dt*acceleration)
  expected_position = lead.dRel + np.cumsum(dt*expected_velocity)
  np.testing.assert_array_equal(projection.raw_trajectory, np.column_stack((expected_position, expected_velocity)))
  assert projection.raw_trajectory[-1, 1] > 0.0 and vars(lead) == recorded


@pytest.mark.parametrize('boundary', [-2.0, 0.0])
@pytest.mark.parametrize('precision', [np.float32, np.float64])
def test_aligned_motion_model_is_continuous_at_adjacent_wire_and_double_speeds(boundary, precision):
  # Hold source and initial epochs equal. The pre-existing signed-time
  # alignment of raw lead speed is a separate contract, unchanged here.
  speeds = [float(np.nextafter(precision(boundary), precision(direction))) for direction in (-np.inf, np.inf)]
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  projections = [planner.project_fusion_lead(object_lead(distance=7.5, speed=speed, acceleration=-2.5),
                                            slot=None, reference_ns=EPOCH_NS) for speed in speeds]
  dt = np.diff(T_IDXS, prepend=0.0)
  reverse_change = np.cumsum(dt * -2.5 * np.exp(-1.5*T_IDXS**2/2.0))
  # On [-2,0], v_future=v0*(1-delta_v/2); outside it the slope is at most 1.
  # Integrating this Lipschitz bound tests continuity without a time filter.
  velocity_lipschitz = 1.0 + abs(reverse_change[-1])/2.0
  rounding = 4.0*(N+1)*np.spacing(max(np.max(np.abs(p.raw_trajectory)) for p in projections))
  assert np.max(np.abs(projections[1].raw_trajectory[:, 0]-projections[0].raw_trajectory[:, 0])) <= (
    T_IDXS[-1]*velocity_lipschitz*(speeds[1]-speeds[0]) + rounding)
  for speed, projection in zip(speeds, projections, strict=True):
    assert projection.raw_trajectory[0, 1] == speed


def test_adjacent_opposing_boundary_inputs_keep_the_same_infeasible_short_window():
  # A hard -2 switch gave one side +0.359 m behind feasibility and the other
  # -0.247 m for these nearly identical Float32 measurements. The continuous
  # model retains the same occupied-geometry failure on both sides.
  planners = []
  for speed in (-1.9999990463256836, -2.0000009536743164):
    lead = object_lead(distance=7.5, speed=speed, acceleration=-2.5, end=0.625, persistent=False)
    planner = LongitudinalMpc()
    planner.set_cur_state(10.0, 0.0)
    planner.set_weights(prev_accel_constraint=False)
    planner.update(fused_state(lead), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
    assert planner.solution_status != 0 and not planner.previous_solution_valid
    assert planner.finite_risk_braking_accel is None
    planners.append(planner)
  assert abs(planners[0].diagnostic_occupancy_bounds[-1][2]-planners[1].diagnostic_occupancy_bounds[-1][2]) < 2e-6


@pytest.mark.parametrize('boundary', [-2.0, 0.0])
def test_aligned_continuous_motion_model_has_no_matched_input_reference_acceleration_step(boundary, record_property):
  from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import sample_reference_acceleration
  requests = []
  for speed in (boundary-1e-6, boundary+1e-6):
    # Hold admission and occupancy fixed to isolate this motion-model change
    # from the producer's independent direction/authority decisions.
    lead = object_lead(distance=40.0, speed=speed, acceleration=-2.5)
    planner = solve(fused_state(lead), iterations=12)
    requests.append(float(np.clip(sample_reference_acceleration(planner.a_solution, 0.55), ACCEL_MIN, ACCEL_MAX)))
  step = abs(requests[1]-requests[0])
  record_property('boundary_reference_acceleration_pair', requests)
  record_property('boundary_reference_acceleration_step', step)
  # Numerical comparison of matched converged fixtures; no such filter or
  # acceleration-step cap is added to production.
  assert step < 1e-3


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
