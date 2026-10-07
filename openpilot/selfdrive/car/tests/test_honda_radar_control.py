"""Active fusion object contracts; execution is from a committed host snapshot."""
import copy
import math

import pytest

from openpilot.selfdrive.car.honda_radar_control import COAST_NS, MAX_OBJECTS, HondaRadarControl


START = 10_000_000_000
STEP = 66_000_000


def vision(distance=20.0, speed=10.0, ego=10.0, lateral=0.0, probability=.94, acceleration=0.0):
  return {'present': True, 'dRel': distance, 'yRel': lateral, 'vRel': speed - ego,
          'vLead': speed, 'vLeadK': speed, 'aLeadK': acceleration, 'aLeadTau': .3,
          'modelProb': probability, 'radar': False, 'radarTrackId': -1}


def radar(frame=0, *, identifier=1, oem=7, distance=20.0, speed=10.0, lateral=0.0, lateral_speed=0.0, **changes):
  stamp = START + frame * STEP
  result = {'trackId': identifier, 'oemTrackId': oem, 'generation': 1, 'slot': 0,
            'positionSeq': 10 + frame, 'positionMonoTime': stamp, 'velocityMonoTime': stamp,
            'dRel': distance, 'yRel': lateral, 'groundSpeed': speed, 'lateralSpeed': lateral_speed,
            'positionVariance': .2, 'velocityVariance': .3, 'observations': 10 + frame,
            'active': True, 'mature': True, 'velocityGood': True, 'coasting': False,
            'acceleration': -.4, 'accelerationGood': True, 'lastInvalidReason': '', 'lastInvalidMonoTime': 0,
            'previousPositionSeq': 9 + frame, 'previousPositionMonoTime': stamp - STEP}
  result.update(changes)
  return result


def binding(index=0, identifier=1, bias_x=0.0, bias_y=0.0, **changes):
  result = {'modelIndex': index, 'trackId': identifier, 'candidateTrackId': identifier,
            'present': True, 'probability': .94, 'identityState': 'same', 'identityReason': 'historyConsistent',
            'rejectionMonoTime': 0, 'bindingStartMonoTime': START - 300_000_000, 'bindingSamples': 6,
            'bindingAnchorX': bias_x, 'bindingAnchorY': bias_y, 'reason': 'matched'}
  result.update(changes)
  return result


def scene(frame=0, *, tracks=None, leads=None, bindings=None, ego=10.0, short_path=False, lanes=False):
  stamp = START + frame * STEP
  leads = leads or [{'present': False}, {'present': False}]
  hypotheses = []
  for lead in leads:
    hypotheses.append({'prob': lead.get('modelProb', 0.0), 'x': [lead.get('dRel', 0.0) + 1.52],
                       'y': [-lead.get('yRel', 0.0)], 'v': [lead.get('vLeadK', 0.0)], 'a': [lead.get('aLeadK', 0.0)],
                       'xStd': [1.0], 'yStd': [.4], 'vStd': [1.0]})
  model = {'timestampEof': stamp, 'leadsV3': hypotheses, 'velocity': {'x': [ego]}, 'orientationRate': {'z': [0.0]},
           'position': {'x': [0.0] * 33 if short_path else [5.0 * i for i in range(33)], 'y': [0.0] * 33}}
  if lanes:
    model['laneLines'] = [{'x': [6.0 * i for i in range(33)], 'y': [offset] * 33} for offset in (-5.4, -1.8, 1.8, 5.4)]
    model['laneLineProbs'] = [.1, .99, .99, .1]
    model['laneLineStds'] = [1.0, .1, .1, 1.0]
  fusion = {'version': 1, 'algorithmVersion': 'bosch-a-shadow-3', 'enabled': True, 'active': True,
            'modelMonoTime': stamp, 'modelTimestampEof': stamp, 'egoMonoTime': stamp,
            'tracks': [radar(frame)] if tracks is None else tracks, 'leads': bindings or [], 'counters': {'resets': 0}}
  observation = {'schemaVersion': 1, 'active': True, 'reason': 'observing', 'sessionId': 'control-fixture',
                 'sequence': frame + 1, 'fusion': fusion, 'counters': {'dropped': 0}}
  return {'now_ns': stamp + 2_000_000, 'enabled': True, 'observation': observation, 'observation_ns': stamp + 1_000_000,
          'observation_valid': True, 'model': model, 'model_ns': stamp, 'v_ego': ego, 'vision_leads': leads}


def evidence(result):
  return result.objects[0]['evidence']


@pytest.mark.parametrize('speed', [0.0, -.2, -.7, 4.0, 10.0])
def test_first_mature_radar_object_needs_no_visual_history_including_static(speed):
  result = HondaRadarControl().update(**scene(tracks=[radar(speed=speed)]))
  assert result.healthy and not result.degraded
  assert len(result.objects) == 1
  assert result.leads[0]['present'] and result.leads[0]['modelProb'] == 0.0
  assert result.leads[0]['vLeadK'] == speed
  assert evidence(result)['source'] == 'radar'
  assert evidence(result)['followQualified'] and evidence(result)['stopQualified']
  assert evidence(result)['brakeQualified']
  assert evidence(result)['departureQualified'] is (speed > .5)


def test_radar_only_remains_authoritative_beyond_former_visual_gap_timeout():
  controller = HondaRadarControl()
  for frame in range(25):
    result = controller.update(**scene(frame))
    assert result.healthy and not result.degraded
    assert result.leads[0]['present'] and evidence(result)['source'] == 'radar'
    assert evidence(result)['positionMonoTime'] == START + frame * STEP


def test_new_immature_radar_waits_for_observer_maturity_not_visual_confirmation():
  controller = HondaRadarControl()
  for frame in range(3):
    sample = radar(frame, mature=frame == 2, observations=frame + 1)
    result = controller.update(**scene(frame, tracks=[sample]))
    assert bool(result.objects) is (frame == 2)


def test_same_identity_replaces_dva_with_radar_without_weighting():
  visual = vision(distance=23.0, speed=12.0, acceleration=.6)
  result = HondaRadarControl().update(**scene(tracks=[radar(distance=20.0, speed=8.0, acceleration=-1.2)],
                                                   leads=[visual, {'present': False}], bindings=[binding(bias_x=-3.0)]))
  assert len(result.objects) == 1
  assert result.leads[0]['dRel'] == 20.0
  assert result.leads[0]['vLeadK'] == 8.0
  assert result.leads[0]['aLeadK'] == -1.2
  assert result.leads[0]['modelProb'] == visual['modelProb']
  assert evidence(result)['source'] == 'fused'


@pytest.mark.parametrize('state', ['different', 'unresolved'])
def test_near_visual_cut_in_is_not_blended_with_far_radar(state):
  visual = vision(distance=7.0, speed=3.0)
  result = HondaRadarControl().update(**scene(tracks=[radar(distance=50.0)], leads=[visual, {'present': False}],
                                                   bindings=[binding(identityState=state, rejectionMonoTime=START)]))
  assert len(result.objects) == 2
  assert result.leads[0]['dRel'] == 7.0 and result.leads[0]['vLeadK'] == 3.0
  assert result.leads[0]['evidence']['source'] == 'vision'
  assert next(lead for lead in result.objects if lead['radar'])['dRel'] == 50.0
  assert result.leads[0]['modelProb'] == .94


def test_current_visual_identity_jump_cannot_reuse_observers_old_same_binding():
  result = HondaRadarControl().update(**scene(tracks=[radar(distance=40.0)],
                                                   leads=[vision(distance=8.0), {'present': False}], bindings=[binding()]))
  assert len(result.objects) == 2 and result.leads[0]['dRel'] == 8.0
  assert all(lead['evidence']['source'] != 'fused' for lead in result.objects)


def test_future_hypotheses_of_same_radar_track_are_one_object():
  result = HondaRadarControl().update(**scene(leads=[vision(), vision(distance=20.2)],
                                                   bindings=[binding(0), binding(1, bias_x=-.2)]))
  assert len(result.objects) == 1
  assert evidence(result)['source'] == 'fused' and evidence(result)['trackId'] == 1


def test_third_entering_object_survives_two_existing_far_leads():
  result = HondaRadarControl().update(**scene(tracks=[radar(distance=18.0, speed=2.0, lateral=3.0, lateral_speed=-2.0)],
                                                   leads=[vision(distance=50.0), vision(distance=70.0)]))
  assert len(result.objects) == 3
  entering = next(lead for lead in result.objects if lead['radar'])
  assert entering['evidence']['pathState'] == 'entering'
  assert entering['evidence']['brakeQualified']
  assert not entering['evidence']['followQualified']
  assert not entering['evidence']['stopQualified']


def test_curved_path_uses_ground_velocity_normal_to_path():
  values = scene(tracks=[radar(lateral=2.152, lateral_speed=1.0)])
  values['model']['position']['y'] = [-.1 * x for x in values['model']['position']['x']]
  result = HondaRadarControl().update(**values)
  assert evidence(result)['pathState'] == 'inPath'
  assert abs(evidence(result)['pathOffset']) < 1e-9
  assert abs(evidence(result)['lateralSpeed']) < 1e-9
  assert evidence(result)['followQualified']


def test_short_stop_trajectory_uses_quality_lanes_only_inside_spatial_support():
  near = HondaRadarControl().update(**scene(tracks=[radar(distance=30.0, speed=0.0)], short_path=True, lanes=True))
  assert evidence(near)['pathSource'] == 'laneLines'
  assert evidence(near)['stopQualified']
  far = HondaRadarControl().update(**scene(tracks=[radar(distance=120.0, speed=0.0)], short_path=True, lanes=True))
  assert not far.objects
  bad = scene(short_path=True, lanes=True)
  bad['model']['laneLineProbs'][1] = .2
  assert not HondaRadarControl().update(**bad).objects


def test_unknown_geometry_does_not_create_new_in_path_target_or_fake_departure():
  unknown = HondaRadarControl().update(**scene(short_path=True))
  assert not unknown.objects and not unknown.cleared
  controller = HondaRadarControl()
  controller.update(**scene())
  gap = controller.update(**scene(1, short_path=True))
  assert evidence(gap)['source'] == 'predicted'
  assert evidence(gap)['pathSource'] == 'previousPath'
  assert evidence(gap)['brakeQualified'] and not evidence(gap)['departureQualified']
  assert gap.degraded and not gap.cleared
  expired = controller.update(**scene(7, short_path=True))
  assert not expired.objects and expired.degraded and not expired.cleared


def test_crossing_uses_finite_window_then_explicit_fresh_clear():
  controller = HondaRadarControl()
  start = controller.update(**scene(tracks=[radar(lateral=3.0, lateral_speed=-2.0, speed=0.0)]))
  assert evidence(start)['pathState'] == 'entering'
  assert evidence(start)['conflictStart'] == pytest.approx(.35)
  assert evidence(start)['conflictEnd'] == pytest.approx(2.75)
  assert not evidence(start)['stopQualified']
  for frame in range(1, 44):
    lateral = 3.0 - 2.0 * frame * STEP / 1e9
    # Keep longitudinal range in front while the vehicle travels slowly past
    # this finite corridor crossing; only the target lateral motion changes.
    values = scene(frame, tracks=[radar(frame, lateral=lateral, lateral_speed=-2.0, speed=0.0)], ego=0.0)
    result = controller.update(**values)
    if result.objects:
      assert evidence(result)['conflictEnd'] < 10.0
      assert not evidence(result)['stopQualified']
    if result.cleared:
      assert result.cleared == (1,) and result.session_id == 'control-fixture'
      break
  else:
    raise AssertionError('crossing target never produced observed departure')


def test_current_crossing_is_not_a_permanent_stationary_follow_lead():
  result = HondaRadarControl().update(**scene(tracks=[radar(speed=0.0, lateral_speed=-2.0)]))
  assert evidence(result)['pathState'] == 'inPath'
  assert evidence(result)['conflictEnd'] == pytest.approx(1.25)
  assert not evidence(result)['followQualified'] and not evidence(result)['stopQualified']
  assert evidence(result)['brakeQualified']


def test_alignment_rotates_velocity_without_creating_ground_motion_from_ego_turn():
  values = scene(tracks=[radar(speed=0.0)])
  dt, yaw, ego = .1, .2, 10.0
  current = START + int(dt * 1e9)
  values['now_ns'], values['model_ns'] = current + 2_000_000, current
  values['model']['timestampEof'] = current
  values['model']['orientationRate']['z'] = [yaw]
  result = HondaRadarControl().update(**values)
  angle = yaw * dt
  ego_x, ego_y = ego * math.sin(angle) / yaw, -ego * (1 - math.cos(angle)) / yaw
  expected_x = math.cos(angle) * (20.0 - ego_x) + math.sin(angle) * ego_y
  assert result.leads[0]['dRel'] == pytest.approx(expected_x)
  assert result.leads[0]['vLeadK'] == 0.0 and evidence(result)['lateralSpeed'] == 0.0
  assert evidence(result)['estimateMonoTime'] == current
  assert evidence(result)['positionMonoTime'] == START


def test_can_received_after_camera_eof_remains_valid_when_projected_back_to_eof():
  from openpilot.selfdrive.controls.lib.lead_qualification import qualified_lead
  values = scene(tracks=[radar(positionMonoTime=START + 30_000_000, velocityMonoTime=START + 30_000_000)])
  values['model_ns'] = START + 20_000_000
  values['observation']['fusion']['modelMonoTime'] = START + 20_000_000
  values['observation_ns'] = START + 40_000_000
  values['now_ns'] = START + 45_000_000
  result = HondaRadarControl().update(**values)
  assert evidence(result)['estimateMonoTime'] == START
  assert evidence(result)['positionMonoTime'] == START + 30_000_000
  assert qualified_lead(result.objects[0], 'brake', now_ns=values['now_ns'])
  assert qualified_lead(result.objects[0], 'brake')


def test_position_new_velocity_old_cannot_renew_velocity_authority():
  from openpilot.selfdrive.controls.lib.lead_qualification import qualified_lead
  controller = HondaRadarControl()
  controller.update(**scene())
  values = scene(4, tracks=[radar(4, velocityMonoTime=START, velocityGood=False, observations=3,
                                previousPositionMonoTime=0, previousPositionSeq=0)])
  result = controller.update(**values)
  assert evidence(result)['source'] == 'predicted' and evidence(result)['velocityMonoTime'] == START
  assert evidence(result)['brakeQualified'] and not evidence(result)['departureQualified']
  assert result.objects[0]['aLeadK'] == 0.0
  assert qualified_lead(result.objects[0], 'brake', now_ns=values['now_ns'])
  values = scene(6, tracks=[radar(6, velocityMonoTime=START, velocityGood=False, observations=3,
                                previousPositionMonoTime=0, previousPositionSeq=0)])
  result = controller.update(**values)
  assert not result.objects and result.degraded


def test_component_fallback_uses_only_confirmed_same_visual_velocity():
  values = scene(tracks=[radar(velocityGood=False, velocityMonoTime=0, accelerationGood=False)],
                 leads=[vision(speed=7.0)], bindings=[binding()])
  result = HondaRadarControl().update(**values)
  assert result.leads[0]['vLeadK'] == 7.0 and result.leads[0]['dRel'] == 20.0
  assert evidence(result)['source'] == 'fused' and evidence(result)['velocityMonoTime'] == START
  assert 'visionVelocity' in evidence(result)['reason']
  values['observation']['fusion']['leads'][0]['identityState'] = 'different'
  values['observation']['fusion']['tracks'][0].update(observations=3, velocityVariance=25.0)
  result = HondaRadarControl().update(**values)
  assert len(result.objects) == 1 and evidence(result)['source'] == 'vision'


def test_position_history_can_supply_supported_velocity_not_zero_substitution():
  result = HondaRadarControl().update(**scene(tracks=[radar(speed=4.0, velocityGood=False, velocityMonoTime=0)]))
  assert result.leads[0]['vLeadK'] == 4.0
  assert evidence(result)['velocityMonoTime'] == START
  assert 'positionVelocity' in evidence(result)['reason']
  assert not evidence(result)['accelerationValid']


@pytest.mark.parametrize('changes', [
  {'groundSpeed': math.nan, 'velocityGood': False, 'velocityMonoTime': 0, 'observations': 3},
  {'dRel': math.nan}, {'lateralSpeed': math.inf}, {'positionVariance': -1.0},
])
def test_invalid_components_do_not_become_zero_speed_or_fake_close_obstacles(changes):
  result = HondaRadarControl().update(**scene(tracks=[radar(**changes)], leads=[vision(distance=8.0)]))
  assert all(not lead['radar'] for lead in result.objects)
  assert result.leads[0]['dRel'] == 8.0


def test_duplicate_publication_cannot_extend_raw_measurement_age_or_coast_forever():
  controller = HondaRadarControl()
  initial = scene()
  controller.update(**initial)
  coast = copy.deepcopy(initial)
  coast['now_ns'] = START + 250_000_000
  coast['model_ns'] = coast['model']['timestampEof'] = START + 240_000_000
  result = controller.update(**coast)
  assert result.objects and evidence(result)['source'] == 'predicted'
  assert evidence(result)['positionMonoTime'] == START
  from openpilot.selfdrive.controls.lib.lead_qualification import qualified_lead
  assert qualified_lead(result.objects[0], 'brake', now_ns=coast['now_ns'])
  coast['now_ns'] = START + COAST_NS + 1
  coast['model_ns'] = coast['model']['timestampEof'] = START + COAST_NS
  result = controller.update(**coast)
  assert not result.objects and result.degraded and not result.cleared


def test_same_sequence_changed_payload_is_rejected_and_new_sequence_recovers():
  controller = HondaRadarControl()
  original = scene()
  controller.update(**original)
  changed = copy.deepcopy(original)
  changed['now_ns'] += 1
  changed['observation']['fusion']['tracks'][0]['dRel'] = 18.0
  result = controller.update(**changed)
  assert not result.objects and result.reason == 'publicationOrder' and result.degraded
  restored = controller.update(**scene(1))
  assert restored.healthy and not restored.degraded and restored.objects


def test_out_of_order_source_rejects_without_giving_old_identity_new_lifetime():
  controller = HondaRadarControl()
  controller.update(**scene(1))
  old = scene(0)
  old['now_ns'] = START + STEP + 3_000_000
  old['model_ns'] = old['model']['timestampEof'] = START + STEP
  result = controller.update(**old)
  assert result.reason == 'publicationOrder' and not result.objects
  assert not result.cleared


def test_session_and_reset_clear_identity_before_accepting_new_source():
  controller = HondaRadarControl()
  controller.update(**scene())
  restarted = scene(1)
  restarted['observation']['sessionId'] = 'new-session'
  result = controller.update(**restarted)
  assert result.reason == 'observerReset' and not result.objects and not result.cleared
  recovered = scene(2)
  recovered['observation']['sessionId'] = 'new-session'
  result = controller.update(**recovered)
  assert result.objects and evidence(result)['sessionId'] == 'new-session' and result.degraded
  # The caller owns an explicit driver disengage/re-engage reset. An observer
  # restart or waiting past TTL is not evidence that its old obstacle departed.
  controller.reset()
  reengaged = scene(3)
  reengaged['observation']['sessionId'] = 'new-session'
  result = controller.update(**reengaged)
  assert result.objects and result.healthy and not result.degraded


def test_slot_generation_change_preserves_physical_track_without_reconfirmation():
  controller = HondaRadarControl()
  controller.update(**scene())
  result = controller.update(**scene(1, tracks=[radar(1, generation=9, slot=2)]))
  assert result.objects and not result.degraded
  assert evidence(result)['trackId'] == 1 and evidence(result)['generation'] == 9
  assert evidence(result)['followQualified']


def test_explicit_invalid_position_revokes_instead_of_coasting_corrupted_state():
  controller = HondaRadarControl()
  controller.update(**scene())
  result = controller.update(**scene(1, tracks=[radar(1, active=False, lastInvalidReason='positionInnovation')]))
  assert not result.objects and result.degraded and not result.cleared


def test_near_lost_target_not_cleared_by_an_unrelated_far_vehicle():
  controller = HondaRadarControl()
  controller.update(**scene(tracks=[radar(distance=10.0)]))
  later = scene(7, tracks=[radar(7, identifier=2, oem=8, distance=60.0)])
  result = controller.update(**later)
  assert result.objects and result.degraded and not result.cleared


@pytest.mark.parametrize('speed', [0.0, 10.0])
def test_far_irrelevant_target_loss_does_not_latch_acceleration_block(speed):
  controller = HondaRadarControl()
  controller.update(**scene(tracks=[radar(distance=100.0, speed=speed)]))
  result = controller.update(**scene(7, tracks=[]))
  assert result.healthy and not result.degraded and not result.objects
  assert not result.cleared  # no invented positive claim that the road cleared


def test_lost_same_track_can_clear_with_fresh_out_of_path_even_after_memory_expiry():
  controller = HondaRadarControl()
  controller.update(**scene())
  missing = controller.update(**scene(7, tracks=[]))
  assert missing.degraded
  cleared = controller.update(**scene(8, tracks=[radar(8, lateral=4.0)]))
  assert cleared.healthy and not cleared.degraded and cleared.cleared == (1,)


def test_same_confirmed_visual_stream_recovers_from_radar_loss_without_timeout():
  from openpilot.selfdrive.controls.lib.lead_qualification import qualified_lead
  controller = HondaRadarControl()
  controller.update(**scene(leads=[vision()], bindings=[binding()]))
  for frame in range(1, 15):
    values = scene(frame, leads=[vision()])
    values['observation_valid'] = False
    result = controller.update(**values)
    assert not result.healthy and not result.degraded
    assert len(result.objects) == 1 and evidence(result)['reason'] == 'visionFallback'
    assert evidence(result)['source'] == 'fused' and evidence(result)['trackId'] == 1
    assert result.objects[0]['dRel'] == 20.0 and not result.objects[0]['radar']
    assert qualified_lead(result.objects[0], 'brake', now_ns=values['now_ns'])


def test_different_far_visual_stream_cannot_claim_recovery_of_missing_near_radar():
  controller = HondaRadarControl()
  controller.update(**scene(leads=[vision()], bindings=[binding()]))
  changed = scene(1, leads=[vision(distance=60.0)])
  changed['observation_valid'] = False
  result = controller.update(**changed)
  assert result.degraded
  assert all(lead['evidence']['reason'] != 'visionFallback' for lead in result.objects)


def test_disabled_returns_exact_original_visual_fallback_and_clears_state():
  controller = HondaRadarControl()
  controller.update(**scene())
  values = scene(1, leads=[vision(distance=8.0), {'present': False}])
  values['enabled'] = False
  result = controller.update(**values)
  assert result.leads == tuple(values['vision_leads']) and not result.objects and not result.degraded
  assert result.reason == 'disabled' and not result.cleared
  assert not controller._tracks


def test_healthy_no_target_is_distinct_from_sensor_fault():
  empty = HondaRadarControl().update(**scene(tracks=[]))
  assert empty.healthy and not empty.degraded and not empty.objects
  assert empty.reason == 'noQualifiedObjects'
  failed = scene(tracks=[])
  failed['observation_valid'] = False
  failure = HondaRadarControl().update(**failed)
  assert not failure.healthy and failure.degraded and failure.reason == 'observationStale'


def test_complete_collection_preserves_32_radar_objects_and_two_visual_candidates():
  tracks = [radar(identifier=i, oem=i, distance=40.0 + i) for i in range(1, 33)]
  result = HondaRadarControl().update(**scene(tracks=tracks, leads=[vision(distance=7.0), vision(distance=9.0)]))
  assert len(result.objects) == MAX_OBJECTS and result.healthy and not result.degraded
  assert result.leads[0]['dRel'] == 7.0
  assert {lead['evidence']['trackId'] for lead in result.objects if lead['radar']} == set(range(1, 33))


def test_capacity_overlap_is_explicit_failure_not_unbounded_or_silent_truncation():
  controller = HondaRadarControl()
  old = [radar(identifier=i, oem=i, distance=40.0 + i) for i in range(1, 33)]
  controller.update(**scene(tracks=old))
  new = [radar(1, identifier=i + 32, oem=i, distance=40.0 + i) for i in range(1, 33)]
  result = controller.update(**scene(1, tracks=new))
  assert len(result.objects) <= MAX_OBJECTS and len(controller._tracks) <= 32
  assert not result.healthy and result.degraded and result.reason == 'objectCapacity'


def test_formal_cereal_schema_accepts_full_object_contract():
  from openpilot.cereal import log
  result = HondaRadarControl().update(**scene())
  state = log.RadarState.new_message()
  state.leadOne, state.leadTwo, state.radarObjects = result.leads[0], result.leads[1], result.objects
  assert state.radarObjects[0].present
  assert str(state.radarObjects[0].evidence.source) == 'radar'
  assert state.radarObjects[0].evidence.conflictEnd == 10.0


def test_actual_shadow_producer_exports_qualified_acceleration_for_formal_adapter():
  from openpilot.selfdrive.car.honda_radar_fusion import HondaRadarFusion
  from openpilot.selfdrive.car.tests.test_honda_radar_fusion import Inputs, body, observation
  producer, controller = HondaRadarFusion(), HondaRadarControl()
  for frame in range(8):
    stamp = START + frame * STEP
    inputs = Inputs(stamp, probability=.0)
    packet = body(frame, [observation(frame)])
    fusion = producer.update(packet, inputs, stamp + 1)
    packet.update(reason='observing', fusion=fusion)
    result = controller.update(now_ns=stamp + 3_000_000, enabled=True, observation=packet,
                               observation_ns=stamp + 2_000_000, observation_valid=True, model=inputs['modelV2'],
                               model_ns=stamp, v_ego=10.0, vision_leads=[{'present': False}, {'present': False}])
  assert fusion['tracks'][0]['accelerationGood']
  assert result.objects and evidence(result)['accelerationValid']
  assert evidence(result)['source'] == 'radar' and result.leads[0]['modelProb'] == 0.0
