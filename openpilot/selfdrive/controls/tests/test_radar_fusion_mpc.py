"""Active fusion through the real single native MPC.

These fixtures test formal object geometry and requested acceleration, not the
road object's classification or delivered vehicle response. Independent arm
solves below are a fixed comparison experiment, never production work.
"""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan
from openpilot.selfdrive.controls.lib.lead_qualification import MAX_CONTROL_OBJECTS
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_STOP_DISTANCE, MODEL_V_REF, MPC_INVALID_INPUT, N, RISK_STOP_DISTANCE, STOP_DISTANCE, T_IDXS,
  LongitudinalMpc, LongitudinalPlanSource, MpcReference,
)
from openpilot.selfdrive.controls.lib.model_reference import ModelReferenceEvidence, ModelSource, MovingPreference
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state


EPOCH_NS = 10_000_000_000


def object_lead(track=1, *, distance=30.0, speed=0.0, ego_speed=10.0, acceleration=0.0,
                source='radar', start=0.0, end=10.0, path='inPath', persistent=True, epoch=EPOCH_NS):
  evidence = SimpleNamespace(
    version=1, source=source, sessionId='radar-test' if source != 'vision' else '', trackId=track,
    generation=0, positionMonoTime=epoch, velocityMonoTime=epoch, estimateMonoTime=epoch,
    modelMonoTime=epoch, positionStd=0.25, velocityStd=0.25, positionValid=True, velocityValid=True,
    accelerationValid=True, followQualified=persistent, brakeQualified=True, stopQualified=persistent,
    departureQualified=persistent, fcwQualified=True, coasting=False, pathState=path,
    pathOffset=0.0, conflictStart=start, conflictEnd=end, lateralSpeed=0.0 if persistent else 3.0,
    pathSource='model', visionIndex=0 if source == 'vision' else -1,
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


def solve(objects, *, speed=10.0, reference=None, iterations=12, now_ns=EPOCH_NS, reference_ns=EPOCH_NS):
  planner = LongitudinalMpc()
  planner.diagnostic_obstacles_enabled = True
  planner.set_cur_state(speed, 0.0)
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
  assert not np.any(planner._fusion_projections[0].occupied[T_IDXS > 2.6])
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


def test_between_node_occupancy_is_kept_and_fcw_checks_its_real_time():
  crossing = object_lead(distance=8.0, start=0.7, end=0.85, path='entering', persistent=False)
  assert not np.any((T_IDXS >= 0.7) & (T_IDXS <= 0.85))
  planner = solve(fused_state(crossing))
  projection = planner._fusion_projections[0]
  assert np.any(projection.occupied)
  assert action(planner) < 0.0
  # The independent straight trajectory reaches x=8 inside this unsampled
  # occupancy window. Checking only the native node times would miss it.
  states = np.column_stack((10.0 * T_IDXS, np.full(N+1, 10.0), np.zeros(N+1)))
  assert planner.collision_in_occupancy(projection, states, np.zeros(N))
  clear = object_lead(distance=8.0, start=0.1, end=0.2, path='entering', persistent=False)
  clear_projection = planner.project_fusion_lead(clear, slot=None, reference_ns=EPOCH_NS)
  assert not planner.collision_in_occupancy(clear_projection, states, np.zeros(N))


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
  arguments = dict(reference=reference, now_ns=now_ns, reference_ns=reference_ns)
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
  radar = object_lead(distance=radar_distance, speed=radar_speed, ego_speed=15.0)
  vision = object_lead(source='vision', distance=vision_distance, speed=vision_speed, ego_speed=15.0)
  radar_arm = solve(fused_state(radar), speed=15.0)
  vision_arm = solve(fused_state(vision), speed=15.0)
  joint = solve(fused_state(radar, vision), speed=15.0)
  minimum = min(action(radar_arm), action(vision_arm))
  difference = action(joint) - minimum
  record_property('independent_radar_acceleration', action(radar_arm))
  record_property('independent_vision_acceleration', action(vision_arm))
  record_property('joint_acceleration', action(joint))
  record_property('joint_minus_independent_minimum', difference)
  # This fixed tolerance checks these concrete braking cases only. It is not
  # an equivalence proof for nonlinear solves, all controls or other scenes.
  assert action(joint) < 0.0 and minimum < 0.0
  assert difference <= 0.05
