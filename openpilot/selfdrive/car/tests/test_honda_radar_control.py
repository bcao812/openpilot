"""Active fusion object contracts; execution is from a committed host snapshot."""
import copy
import math
from dataclasses import replace

import pytest

from openpilot.selfdrive.car.honda_radar_control import (
  FUSION_VERSION, HORIZON, MAX_TRACKS, SWEPT_HALF_WIDTH, HondaRadarControl, _Corridor, _corridors, _qualified_path,
)


START = 10_000_000_000
STEP = 66_000_000


def vision(distance=20.0, speed=10.0, ego=10.0, lateral=0.0, probability=.94, acceleration=0.0):
  return {'present': True, 'dRel': distance, 'yRel': lateral, 'vRel': speed - ego,
          'vLead': speed, 'vLeadK': speed, 'aLeadK': acceleration, 'aLeadTau': .3,
          'modelProb': probability, 'radar': False, 'radarTrackId': -1}


def position_velocity_quality(track, **changes):
  # Retained diagnostic evidence must never authorize or veto native motion.
  observations = track['observations']
  count = min(8, max(0, observations)) if type(observations) is int else 0
  qualified = count >= 6
  result = {'version': 1, 'positionSeq': track['positionSeq'], 'positionMonoTime': track['positionMonoTime'],
            'sampleCount': count, 'span': max(0, count - 1) * STEP / 1e9,
            'shortSpeed': track['groundSpeed'], 'longSpeed': track['groundSpeed'], 'speedStd': .5,
            'windowDifference': 0.0, 'windowDifferenceStd': .2, 'innovationBias': 0.0,
            'posteriorCorrection': .1, 'posteriorCorrectionRatio': .2, 'posteriorSpeedDifference': .1,
            'consistentSamples': count, 'qualified': qualified,
            'reason': 'converged' if qualified else 'insufficientSupport'}
  result.update(changes)
  return result


def native_motion_quality(track, **changes):
  """Explicit native-pair evidence for adapter tests; producer tests count CAN."""
  count = track['observations'] if type(track['observations']) is int else 0
  stamp, velocity_stamp = track['positionMonoTime'], track['velocityMonoTime']
  known_times = type(stamp) is int and type(velocity_stamp) is int
  qualified = count >= 5 and track['velocityGood'] is True and known_times and min(stamp, velocity_stamp) > 0
  result = {'version': 1, 'qualified': qualified, 'reason': 'qualified' if qualified else 'insufficientSupport',
            'positionSeq': track['positionSeq'], 'positionMonoTime': stamp, 'velocitySeq': track['positionSeq'],
            'velocityMonoTime': velocity_stamp, 'sampleCount': max(0, count),
            'firstPositionMonoTime': stamp - max(0, count - 1) * STEP if type(stamp) is int else 0,
            'validUntilMonoTime': min(stamp, velocity_stamp) + 200_000_000 if known_times else 0,
            'pairCompleteMonoTime': max(stamp, velocity_stamp) if known_times else 0,
            'rawRelativeSpeed': 0.0, 'groundSpeed': track['groundSpeed'], 'velocityVariance': track['velocityVariance'],
            'acceleration': track['acceleration'], 'accelerationValid': track['accelerationGood'],
            'u10Raw': 50, 'rangeSigmaRaw': 2, 'existenceRaw': 125,
            'distanceResidual': 0.0, 'distanceResidualValid': False, 'frameIndex': 0}
  result.update(changes)
  return result


def radar(frame=0, *, identifier=1, oem=7, distance=20.0, speed=10.0, lateral=0.0, lateral_speed=0.0, **changes):
  stamp = START + frame * STEP
  result = {'trackId': identifier, 'oemTrackId': oem, 'generation': 1, 'slot': 0,
            'positionSeq': 10 + frame, 'positionMonoTime': stamp, 'velocityMonoTime': stamp,
            'dRel': distance, 'yRel': lateral, 'groundSpeed': speed, 'lateralSpeed': lateral_speed,
            'positionVariance': .2, 'velocityVariance': .3, 'observations': 10 + frame,
            'lateralPositionVariance': .04, 'lateralVelocityVariance': .01, 'lateralVelocityValid': True,
            'active': True, 'mature': True, 'velocityGood': True, 'coasting': False,
            'acceleration': -.4, 'accelerationGood': True, 'lastInvalidReason': '', 'lastInvalidMonoTime': 0,
            'previousPositionSeq': 9 + frame, 'previousPositionMonoTime': stamp - STEP}
  result.update(changes)
  if 'positionVelocity' not in changes:
    result['positionVelocity'] = position_velocity_quality(result)
  if 'nativeMotion' not in changes:
    result['nativeMotion'] = native_motion_quality(result)
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
  fusion = {'version': 1, 'algorithmVersion': FUSION_VERSION, 'enabled': True, 'active': True,
            'modelMonoTime': stamp, 'modelTimestampEof': stamp, 'egoMonoTime': stamp,
            'tracks': [radar(frame)] if tracks is None else tracks, 'leads': bindings or [], 'counters': {'resets': 0}}
  observation = {'schemaVersion': 1, 'active': True, 'reason': 'observing', 'sessionId': 'control-fixture',
                 'sequence': frame + 1, 'fusion': fusion, 'counters': {'dropped': 0}}
  return {'now_ns': stamp + 2_000_000, 'enabled': True, 'observation': observation, 'observation_ns': stamp + 1_000_000,
          'observation_valid': True, 'model': model, 'model_ns': stamp, 'v_ego': ego, 'vision_leads': leads}


def present_leads(result):
  assert result.objects == []
  return [lead for lead in result.leads if lead.get('present') is True]


def evidence(result):
  return present_leads(result)[0]['evidence']


@pytest.mark.parametrize('points', [
  (), ((0.0, 0.0),), ((0.0, 0.0), (0.0, 0.0)), ((0.0, 0.0), (1e-6, 0.0)),
  ((0.0, 0.0), (20.0, 0.0)), ((0.0, 0.0), (20.0, 4.0)),
  ((0.0, 0.0), (10.0, 0.0), (20.0, 10.0)),
  ((0.0, 0.0), (10.0, -10.0), (20.0, -10.0)),
  tuple((2.0 * i, 4.0 * math.sin(i * .2)) for i in range(33)),
], ids=['empty', 'one-point', 'duplicate', 'degenerate', 'straight', 'sloped', 'left-bend', 'right-bend', 'full-s-bend'])
def test_static_lane_comparison_preserves_consumed_full_relation_fields(points):
  corridor = _Corridor(points, 'modelPath')
  samples = [(x, y) for x in (-1.0, 0.0, 5.0, 10.0, 20.0, 64.0, 65.0) for y in (-12.0, -2.0, 0.0, 2.0, 12.0)]
  # Include finite support endpoints/corners and both sides of the lane
  # agreement threshold, without replacing the polyline with its tangent.
  for x, y in points:
    samples.extend((x, y + offset) for offset in (-2.0, -.750000001, -.75, 0.0, .75, .750000001, 2.0))
    samples.extend((x, y + offset) for offset in (math.nextafter(.75, 0.0), math.nextafter(.75, math.inf)))
    samples.extend(((math.nextafter(x, -math.inf), y), (math.nextafter(x, math.inf), y)))
  for x, y in samples:
    full = corridor.relation(x, y, 0.0, 0.0)
    for limit in (0.0, .75, SWEPT_HALF_WIDTH, 2.0):
      expected = full.state != 'unknown' and abs(full.offset) > limit
      assert corridor.current_distance_exceeds(x, y, limit) is expected, (points, x, y, limit)


@pytest.mark.parametrize('curve', [False, True])
@pytest.mark.parametrize('offset', [0.0, -.75, .75, -.750001, .750001, 3.0])
def test_static_lane_projection_preserves_corridor_admission(monkeypatch, curve, offset):
  model = scene(lanes=True)['model']
  model['position'] = {'x': [2.0 * i for i in range(33)],
                       'y': [2.0 * math.sin(i * .2) if curve else 0.0 for i in range(33)]}
  for line, width in zip(model['laneLines'], (-5.4, -1.8, 1.8, 5.4), strict=True):
    line['x'] = list(model['position']['x'])
    line['y'] = [y + offset + width for y in model['position']['y']]
  current = _corridors(model)

  def full_relation_disagrees(self, x, y, limit):
    result = self.relation(x, y, 0.0, 0.0)
    return result.state != 'unknown' and abs(result.offset) > limit

  monkeypatch.setattr(_Corridor, 'current_distance_exceeds', full_relation_disagrees)
  assert current == _corridors(model)
  if not curve:
    assert bool(current[1].points) is (abs(offset) <= .75)


@pytest.mark.parametrize('fault', ['nonfinite', 'reversed', 'mismatched', 'too-narrow', 'too-wide', 'low-prob', 'high-std', 'nan-prob'])
def test_static_lane_projection_still_rejects_invalid_lane_evidence(fault):
  model = scene(lanes=True)['model']
  if fault == 'nonfinite':
    model['laneLines'][1]['y'][2] = math.nan
  elif fault == 'reversed':
    model['laneLines'][1]['x'][2] = -1.0
  elif fault == 'mismatched':
    model['laneLines'][1]['y'].pop()
  elif fault in ('too-narrow', 'too-wide'):
    width = 2.19 if fault == 'too-narrow' else 4.81
    model['laneLines'][1]['y'] = [-width / 2] * 33
    model['laneLines'][2]['y'] = [width / 2] * 33
  elif fault == 'low-prob':
    model['laneLineProbs'][1] = .649
  elif fault == 'high-std':
    model['laneLineStds'][2] = .501
  else:
    model['laneLineProbs'][1] = math.nan
  path, lane = _corridors(model)
  assert path.points and not lane.points


@pytest.mark.parametrize('speed', [0.0, -.2, -.7, .2999, .3, .3001, .4, .5, 4.0, 10.0])
def test_first_mature_radar_object_needs_no_visual_history_including_static(speed):
  result = HondaRadarControl().update(**scene(tracks=[radar(speed=speed)]))
  assert result.healthy and not result.degraded
  assert len(present_leads(result)) == 1
  assert result.leads[0]['present'] and result.leads[0]['modelProb'] == 0.0
  assert result.leads[0]['vLeadK'] == speed
  assert evidence(result)['source'] == 'radar'
  assert evidence(result)['followQualified'] and evidence(result)['stopQualified']
  assert evidence(result)['brakeQualified']
  assert evidence(result)['departureQualified'] is (speed > .3)


@pytest.mark.parametrize('speed', [.2999, .3, .3001, .4, .5, .5001])
def test_slow_fused_departure_keeps_the_visual_speed_threshold(speed):
  visual = vision(speed=speed, ego=0.0, probability=.99, acceleration=0.0)
  values = scene(tracks=[radar(speed=speed, acceleration=0.0)], leads=[visual], bindings=[binding()], ego=0.0)
  fused = HondaRadarControl().update(**values)
  lead = fused.leads[0]
  assert lead['radar'] and lead['evidence']['source'] == 'fused'
  assert lead['evidence']['departureQualified'] is (speed > .3)
  assert lead['evidence']['brakeQualified'] and lead['evidence']['stopQualified']

  visual_only = HondaRadarControl().update(**scene(tracks=[], leads=[visual], ego=0.0))
  lead = visual_only.leads[0]
  assert not lead['radar'] and lead['evidence']['source'] == 'vision'
  assert lead['evidence']['departureQualified'] is (speed > .5)


@pytest.mark.parametrize('defect', ['coasting', 'invalid_position', 'missing_velocity', 'out_of_path'])
def test_slow_radar_motion_does_not_replace_current_measurement_qualification(defect):
  track = radar(speed=.4, acceleration=0.0)
  if defect == 'coasting':
    track['coasting'] = True
  elif defect == 'invalid_position':
    track['positionMonoTime'] = 0
  elif defect == 'missing_velocity':
    track['velocityGood'] = False
    track['previousPositionMonoTime'] = track['previousPositionSeq'] = 0
  else:
    track['yRel'] = 4.0
  result = HondaRadarControl().update(**scene(tracks=[track], ego=0.0))
  assert not any(lead.get('present') and lead['evidence']['departureQualified'] for lead in result.leads)


def test_slow_position_derived_motion_cannot_replace_native_departure_evidence():
  track = radar(speed=.4, acceleration=0.0, velocityGood=False)
  result = HondaRadarControl().update(**scene(tracks=[track], ego=0.0))
  assert track['positionVelocity']['qualified']
  assert result.healthy and not result.degraded and not present_leads(result)


def test_radar_only_remains_authoritative_beyond_former_visual_gap_timeout():
  controller = HondaRadarControl()
  for frame in range(25):
    result = controller.update(**scene(frame))
    assert result.healthy and not result.degraded
    assert result.leads[0]['present'] and evidence(result)['source'] == 'radar'
    assert evidence(result)['positionMonoTime'] == START + frame * STEP


@pytest.mark.parametrize('visual', [False, True])
def test_missing_radar_is_withdrawn_immediately_and_current_vision_remains(visual):
  controller = HondaRadarControl()
  assert evidence(controller.update(**scene()))['followGeometryMonoTime'] == START
  assert evidence(controller.update(**scene(1)))['followGeometryMonoTime'] == START + STEP
  for frame in (2, 3):
    result = controller.update(**scene(frame, tracks=[], leads=[vision(distance=12.0)] if visual else None))
    assert result.healthy and not result.degraded and not result.cleared
    assert not controller._tracks
    assert len(present_leads(result)) == int(visual)
    if visual:
      lead = present_leads(result)[0]
      assert lead['dRel'] == 12.0 and not lead['radar']
      assert lead['evidence']['source'] == 'vision' and not lead['evidence']['coasting']
      assert lead['evidence']['positionMonoTime'] == START + frame * STEP
      assert lead['evidence']['velocityMonoTime'] == START + frame * STEP


@pytest.mark.parametrize('change', ['opposing', 'crossing', 'future_bend', 'out'])
def test_new_current_geometry_revocation_cannot_recover_an_older_follow_epoch(change):
  controller = HondaRadarControl()
  initial, revised, coast = scene(), scene(1), scene(2, tracks=[])
  if change == 'opposing':
    revised = scene(1, tracks=[radar(1, speed=-3.0)])
  elif change == 'crossing':
    revised = scene(1, tracks=[radar(1, lateral_speed=2.0)])
  elif change == 'future_bend':
    initial = scene(tracks=[radar(distance=5.0, speed=5.0, lateral=3.0)])
    initial['model']['position']['y'] = [-3.0] * 33
    revised = scene(1, tracks=[radar(1, distance=5.0, speed=5.0, lateral=3.0)])
    revised['model']['position'] = {'x': [1.52, 11.52, 21.52], 'y': [0.0, 0.0, -10.0]}
  else:
    revised['model']['position']['y'] = [-4.0] * 33
  coast['model']['position'] = copy.deepcopy(revised['model']['position'])
  original = evidence(controller.update(**initial))
  assert original['followQualified'] and original['followGeometryMonoTime'] == START
  current = controller.update(**revised)
  assert not present_leads(current)
  assert current.cleared == ((1,) if change in ('future_bend', 'out') else ())
  predicted = controller.update(**coast)
  assert not present_leads(predicted) and not controller._tracks


@pytest.mark.parametrize('changes', [
  {'followGeometryMonoTime': None}, {'followGeometryMonoTime': 0}, {'followGeometryMonoTime': True},
  {'followGeometryMonoTime': START - 1}, {'followGeometryMonoTime': START + 2 * STEP},
  {'sessionId': 'different-session'}, {'trackId': 2}, {'generation': 2},
  {'positionMonoTime': START - 1}, {'velocityMonoTime': START - 1}, {'estimateMonoTime': START - 1},
])
def test_missing_radar_cannot_republish_saved_geometry_or_components(changes):
  controller = HondaRadarControl()
  controller.update(**scene())
  accepted = controller._tracks[1].accepted['evidence']
  accepted.update(changes)
  if changes.get('followGeometryMonoTime', 0) is None:
    accepted.pop('followGeometryMonoTime')
  missing = controller.update(**scene(1, tracks=[]))
  assert not present_leads(missing) and not controller._tracks
  assert not missing.degraded and not missing.cleared


def test_new_radar_waits_for_observer_and_native_pair_maturity_without_visual_confirmation():
  controller = HondaRadarControl()
  for frame in range(6):
    sample = radar(frame, mature=frame >= 2, observations=frame + 1)
    result = controller.update(**scene(frame, tracks=[sample]))
    assert bool(present_leads(result)) is (frame >= 4)





def test_unknown_current_path_does_not_reuse_far_historical_geometry():
  controller = HondaRadarControl()
  controller.update(**scene(tracks=[radar(distance=90.0, speed=-.65)]))
  assert not controller._tracks[1].relevant
  for frame in range(1, 4):
    result = controller.update(**scene(frame, tracks=[radar(frame, distance=90.0, speed=-.65)], short_path=True))
    assert result.healthy and not result.degraded and not result.cleared
    assert not present_leads(result) and not controller._tracks
  result = controller.update(**scene(4, tracks=[radar(4, distance=90.0, active=False, lastInvalidReason='positionInnovation')]))
  assert result.healthy and not result.degraded and not present_leads(result) and not result.cleared


def test_missing_radar_does_not_project_saved_path_motion():
  controller = HondaRadarControl()
  controller.update(**scene())
  memory = controller._tracks[1]
  memory.path = replace(memory.path, rate=6.0)
  result = controller.update(**scene(1, tracks=[]))
  assert not present_leads(result) and not controller._tracks
  assert not result.cleared


def test_legacy_predicted_entry_history_cannot_become_current_occupancy():
  controller = HondaRadarControl()
  controller.update(**scene())
  memory = controller._tracks[1]
  memory.path = replace(memory.path, state='entering', offset=1.5, rate=-5.0)
  missing = controller.update(**scene(1, tracks=[]))
  assert missing.healthy and not missing.degraded and not present_leads(missing) and not missing.cleared
  assert not controller._tracks


@pytest.mark.parametrize('frame', [1, 3, 6])
def test_missing_radar_does_not_wait_for_a_prediction_deadline(frame):
  controller = HondaRadarControl()
  controller.update(**scene())
  result = controller.update(**scene(frame, tracks=[]))
  assert result.healthy and not result.degraded and not present_leads(result) and not result.cleared
  assert not controller._tracks


@pytest.mark.parametrize('previous_distance,previous_lateral,current_lateral', [
  (40.0, 1.6, 4.0), (40.0, 1.6, 0.0), (20.0, 0.0, 4.0), (80.0, 0.0, 0.0), (None, None, 0.0),
])
def test_native_quality_loss_withdraws_even_a_selected_near_lead_without_new_degraded_obligations(
    previous_distance, previous_lateral, current_lateral):
  controller = HondaRadarControl()
  if previous_distance is not None:
    first = controller.update(**scene(tracks=[radar(distance=previous_distance, lateral=previous_lateral)]))
    assert first.healthy and not first.degraded
    assert len(present_leads(first)) == int(previous_lateral == 0.0)
    if previous_lateral == 0.0:
      assert controller._tracks[1].relevant is (previous_distance == 20.0)
    else:
      assert not controller._tracks
  # Independent fresh positions cannot replace a missing velocity component.
  # This is a normal quality rejection, not corrupt values or a sensor outage.
  for frame in (7, 8):
    current = radar(frame, distance=40.0, lateral=current_lateral, velocityGood=False, velocityVariance=5.0)
    result = controller.update(**scene(frame, tracks=[current]))
    assert result.healthy and not result.degraded
    assert not present_leads(result) and not result.cleared


def test_same_identity_replaces_dva_with_radar_without_weighting():
  visual = vision(distance=23.0, speed=12.0, acceleration=.6)
  result = HondaRadarControl().update(**scene(tracks=[radar(distance=20.0, speed=8.0, acceleration=-1.2)],
                                                   leads=[visual, {'present': False}], bindings=[binding(bias_x=-3.0)]))
  assert len(present_leads(result)) == 1
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
  assert len(present_leads(result)) == 2
  assert result.leads[0]['dRel'] == 7.0 and result.leads[0]['vLeadK'] == 3.0
  assert result.leads[0]['evidence']['source'] == 'vision'
  assert next(lead for lead in present_leads(result) if lead['radar'])['dRel'] == 50.0
  assert result.leads[0]['modelProb'] == .94


def test_current_visual_identity_jump_cannot_reuse_observers_old_same_binding():
  result = HondaRadarControl().update(**scene(tracks=[radar(distance=40.0)],
                                                   leads=[vision(distance=8.0), {'present': False}], bindings=[binding()]))
  assert len(present_leads(result)) == 2 and result.leads[0]['dRel'] == 8.0
  assert all(lead['evidence']['source'] != 'fused' for lead in present_leads(result))


def test_future_hypotheses_of_same_radar_track_are_one_object():
  result = HondaRadarControl().update(**scene(leads=[vision(), vision(distance=20.2)],
                                                   bindings=[binding(0), binding(1, bias_x=-.2)]))
  assert len(present_leads(result)) == 1
  assert evidence(result)['source'] == 'fused' and evidence(result)['trackId'] == 1


def test_third_side_object_needs_current_ordinary_follow_motion_to_enter_a_lead_slot():
  controller = HondaRadarControl()
  visual = [vision(distance=50.0), vision(distance=70.0)]
  for frame, lateral in enumerate((3.0, 1.0)):
    result = controller.update(**scene(frame, tracks=[radar(frame, distance=18.0, speed=2.0,
                                                           lateral=lateral, lateral_speed=-2.0)], leads=visual))
    assert len(present_leads(result)) == 2 and all(not lead['radar'] for lead in present_leads(result))
    assert not controller._tracks
  current = controller.update(**scene(2, tracks=[radar(2, distance=18.0, speed=2.0, lateral=1.0)], leads=visual))
  assert len(present_leads(current)) == 2
  admitted = evidence(current)
  assert admitted['trackId'] == 1 and admitted['pathState'] == 'inPath'
  assert admitted['followQualified'] and admitted['brakeQualified'] and admitted['stopQualified']
  assert current.leads[1]['dRel'] == 50.0


def test_sloped_straight_path_uses_ground_velocity_normal_to_path():
  values = scene(tracks=[radar(lateral=2.152, lateral_speed=1.0)])
  values['model']['position']['y'] = [-.1 * x for x in values['model']['position']['x']]
  result = HondaRadarControl().update(**values)
  assert evidence(result)['pathState'] == 'inPath'
  assert abs(evidence(result)['pathOffset']) < 1e-9
  assert abs(evidence(result)['lateralSpeed']) < 1e-9
  assert evidence(result)['followQualified']


def test_current_outside_object_is_not_admitted_for_a_predicted_later_bend():
  values = scene(tracks=[radar(distance=5.0, lateral=3.0, speed=5.0)])
  # Radar coordinates: (0, 0) -> (10, 0) -> (20, 10). The object has
  # zero current cross-track velocity, but meets the bend at (13, 3), t=1.6.
  values['model']['position'] = {'x': [1.52, 11.52, 21.52], 'y': [0.0, 0.0, -10.0]}
  result = HondaRadarControl().update(**values)
  assert result.healthy and not result.degraded and not present_leads(result)
  assert result.cleared == (1,)
  relation = _Corridor(((0.0, 0.0), (10.0, 0.0), (20.0, 10.0)), 'modelPath').relation(5.0, 3.0, 5.0, 0.0)
  assert relation.state == 'outOfPath' and relation.offset == pytest.approx(3.0) and relation.rate == 0.0


def test_prediction_does_not_extend_the_initial_tangent_past_a_turn():
  corridor = _Corridor(((0.0, 0.0), (10.0, 0.0), (20.0, 20.0)), 'modelPath')
  # The first tangent would suggest entry after 4 s, x=25 m. No such
  # straight corridor exists; the actual finite path has already turned.
  result = corridor.relation(5.0, -6.0, 5.0, 1.0)
  assert result.state == 'outOfPath'


def test_current_bend_geometry_uses_the_nearest_supported_segment():
  corridor = _Corridor(((0.0, 0.0), (10.0, 10.0), (20.0, 10.0)), 'modelPath')
  result = corridor.relation(5.0, 3.0, 5.0, 0.0)
  assert result.state == 'inPath'
  assert result.offset == pytest.approx(-math.sqrt(2.0))
  assert result.rate == pytest.approx(-5.0 / math.sqrt(2.0))


def test_internal_corner_has_no_gap_between_the_finite_segment_strips():
  corridor = _Corridor(((0.0, 0.0), (10.0, 0.0), (20.0, 10.0)), 'modelPath')
  # Outside both finite strips' along-segment intervals, inside the corner's
  # 2 m body-overlap disk. This is a real join, not a tangent extension.
  result = corridor.relation(10.5, -1.5, 0.0, 0.0)
  assert result.state == 'inPath'
  assert result.offset == pytest.approx(-math.sqrt(2.5))


def test_later_bends_do_not_create_an_occupancy_window_for_a_current_follow_lead():
  values = scene(tracks=[radar(distance=1.0, speed=4.0)])
  values['model']['position'] = {'x': [1.52, 11.52, 21.52, 31.52, 51.52],
                               'y': [0.0, 0.0, -10.0, 0.0, 0.0]}
  result = HondaRadarControl().update(**values)
  claim = evidence(result)
  assert claim['followQualified'] and claim['stopQualified']
  assert claim['conflictStart'] == 0.0 and claim['conflictEnd'] == HORIZON
  assert claim['conflictSupportVersion'] == 0 and not claim['physicalConflictValid']


def test_fast_lateral_motion_is_only_current_geometry_without_future_intervals():
  corridor = _Corridor(((0.0, 0.0), (20.0, 0.0)), 'modelPath')
  outside = corridor.relation(10.0, -3.0, 0.0, 30.0)
  assert outside.state == 'outOfPath'
  current = corridor.relation(10.0, 0.0, 0.0, 30.0)
  assert current.state == 'inPath' and current.rate == 30.0
  assert corridor.relation(-1.0, 0.0, 10.0, 0.0).state == 'unknown'
  result = HondaRadarControl().update(**scene(tracks=[radar(lateral_speed=30.0)]))
  assert not present_leads(result)


@pytest.mark.parametrize('changes', [
  {'lateralVelocityValid': False}, {'lateralVelocityVariance': 4.01},
  {'lateralVelocityVariance': math.nan},
])
def test_lateral_prediction_limits_preserve_current_longitudinal_authority(changes):
  # Poor vy must not erase a newly observed stationary/forward-moving lead.
  for speed in (0.0, 4.0):
    result = HondaRadarControl().update(**scene(tracks=[radar(speed=speed, lateral_speed=2.0, **changes)]))
    claim = evidence(result)
    assert claim['pathState'] == 'inPath' and claim['conflictEnd'] == 10.0
    assert all(claim[purpose + 'Qualified'] for purpose in ('follow', 'brake', 'stop', 'fcw'))
    assert claim['departureQualified'] is (speed > .3)


def test_lateral_position_uncertainty_cannot_create_or_clear_an_obstacle():
  invalid = radar(lateralPositionVariance=4.01)
  result = HondaRadarControl().update(**scene(tracks=[invalid]))
  assert not present_leads(result) and not result.cleared
  controller = HondaRadarControl()
  controller.update(**scene())
  uncertain = controller.update(**scene(1, tracks=[radar(1, lateral=1.0, lateralPositionVariance=4.01)]))
  assert not present_leads(uncertain)
  assert uncertain.degraded and not uncertain.cleared


@pytest.mark.parametrize('variance', [.01, 1.0, 4.01])
def test_outside_object_stays_out_across_lateral_prediction_quality_limits(variance):
  values = scene(tracks=[radar(distance=5.0, lateral=3.0, speed=5.0, lateralVelocityVariance=variance)])
  values['model']['position'] = {'x': [1.52, 11.52, 21.52], 'y': [0.0, 0.0, -10.0]}
  result = HondaRadarControl().update(**values)
  assert result.healthy and not result.degraded and not present_leads(result)
  assert result.cleared == (1,)


def test_measured_outside_geometry_does_not_become_unknown_from_a_future_entry():
  corridor = _Corridor(((0.0, 0.0), (100.0, 0.0)), 'modelPath')
  result = _qualified_path((corridor,), 20.0, 4.0, 0.0, -.5, .04, 1.0, True)
  assert result.state == 'outOfPath'
  assert result.observed_out  # current position remains measured geometry
  objects = HondaRadarControl().update(**scene(tracks=[radar(lateral=4.0, lateral_speed=-.5,
                                                           lateralVelocityVariance=1.0)]))
  assert not present_leads(objects) and objects.cleared == (1,)


def test_fresh_measured_out_clears_old_lead_without_trusting_lateral_prediction():
  controller = HondaRadarControl()
  controller.update(**scene())
  controller.update(**scene(7, tracks=[]))
  result = controller.update(**scene(8, tracks=[radar(8, lateral=4.0, lateral_speed=-.5, lateralVelocityValid=False)]))
  assert result.healthy and not result.degraded
  assert result.cleared == (1,) and not present_leads(result)


@pytest.mark.parametrize('expired', [False, True])
def test_coasting_track_cannot_publish_clearance_from_new_model_geometry(expired):
  controller = HondaRadarControl()
  controller.update(**scene())
  if expired:
    missing = controller.update(**scene(7, tracks=[]))
    assert not present_leads(missing) and not missing.cleared and not controller._tracks
  frame = 8 if expired else 1
  values = scene(frame, tracks=[radar(frame, coasting=True)])
  values['model']['position']['y'] = [-4.0] * 33
  result = controller.update(**values)
  assert result.healthy and not present_leads(result) and not result.cleared


def test_new_model_geometry_with_reused_position_cannot_clear_tracked_obstacle():
  controller = HondaRadarControl()
  controller.update(**scene())
  values = scene(1, tracks=[radar()])
  values['model']['position']['y'] = [-4.0] * 33
  reused = controller.update(**values)
  assert reused.healthy and reused.degraded and not present_leads(reused) and not reused.cleared
  assert not controller._tracks
  fresh = scene(2)
  fresh['model']['position']['y'] = [-4.0] * 33
  measured = controller.update(**fresh)
  assert measured.healthy and not measured.degraded and not present_leads(measured) and measured.cleared == (1,)


def test_finite_path_end_is_unknown_future_space_not_proof_of_departure():
  values = scene(tracks=[radar(distance=5.0, speed=10.0)])
  values['model']['position'] = {'x': [1.52, 11.52], 'y': [0.0, 0.0]}
  result = HondaRadarControl().update(**values)
  assert evidence(result)['pathState'] == 'inPath' and evidence(result)['conflictEnd'] == 10.0
  assert evidence(result)['followQualified'] and evidence(result)['departureQualified']


@pytest.mark.parametrize('lateral', [-1.7, -1.6, -1.250001, 1.250001, 1.6, 1.7])
def test_new_outer_only_stationary_object_has_no_control_or_memory(lateral):
  controller = HondaRadarControl()
  result = controller.update(**scene(tracks=[radar(distance=3.0, speed=0.0, lateral=lateral)], ego=4.0))
  assert result.healthy and not result.degraded and not present_leads(result)
  assert not controller._tracks and not result.cleared


@pytest.mark.parametrize('lateral', [-1.25, -1.20, -1.0, 1.0, 1.20, 1.25])
def test_new_inner_stationary_radar_is_a_full_obstacle_without_vision(lateral):
  # The 2.20 m ego body plus .15 m per-side margin includes both 1.25 m
  # boundaries and the newly covered 1.20 m reflections on either side.
  result = HondaRadarControl().update(**scene(tracks=[radar(distance=3.0, speed=0.0, lateral=lateral)], ego=4.0))
  assert result.leads[0]['present'] and evidence(result)['source'] == 'radar'
  claim = evidence(result)
  assert claim['pathState'] == 'inPath' and not claim['physicalConflictValid']
  assert claim['physicalConflictStart'] == claim['physicalConflictEnd'] == 0.0
  assert all(claim[purpose + 'Qualified'] for purpose in ('brake', 'follow', 'stop', 'fcw'))
  assert not claim['cautionQualified'] and not claim['departureQualified']


def test_outer_object_waits_for_a_fresh_current_follow_lead():
  controller = HondaRadarControl()
  outer = controller.update(**scene(tracks=[radar(distance=8.0, speed=0.0, lateral=1.7, lateral_speed=-1.0)], ego=5.0))
  assert not present_leads(outer) and not controller._tracks and not outer.cleared
  inside = controller.update(**scene(1, tracks=[radar(1, distance=8.0, speed=0.0, lateral=1.1, lateral_speed=-.1)], ego=5.0))
  current = evidence(inside)
  assert current['pathState'] == 'inPath' and current['followQualified'] and current['brakeQualified']
  assert not current['cautionQualified'] and not current['coasting']
  assert current['positionMonoTime'] == START + STEP


@pytest.mark.parametrize('source', ['position_coast', 'reused_position', 'missing_velocity'])
def test_unselected_edge_cannot_supply_memory_for_first_lead_admission(source):
  controller = HondaRadarControl()
  first = controller.update(**scene(tracks=[radar(distance=8.0, speed=0.0, lateral=1.7)], ego=5.0))
  assert not present_leads(first) and not controller._tracks
  changes = {}
  frame = 1
  if source == 'position_coast':
    changes['coasting'] = True
  elif source == 'reused_position':
    frame = 0
  else:
    changes.update(velocityGood=False, velocityMonoTime=START, observations=3,
                   previousPositionMonoTime=0, previousPositionSeq=0)
  current = radar(frame, distance=8.0, speed=0.0, lateral=1.1, **changes)
  result = controller.update(**scene(1, tracks=[current], ego=5.0))
  assert result.healthy and not result.cleared and not present_leads(result)
  assert not controller._tracks


@pytest.mark.parametrize('observation_gap', [False, True])
def test_reused_outside_position_cannot_enter_only_through_camera_time_alignment(observation_gap):
  controller = HondaRadarControl()
  initial = scene(tracks=[radar(lateral=3.0, lateral_speed=-30.0)])
  outside = controller.update(**initial)
  assert not present_leads(outside) and outside.cleared == (1,)
  if observation_gap:
    missing = scene(1, tracks=[])
    missing['observation_valid'] = False
    gap = controller.update(**missing)
    assert not gap.healthy and gap.degraded and not present_leads(gap) and not gap.cleared
    assert controller._cleared == {1: START}
  frame = 2 if observation_gap else 1
  reused = scene(frame)
  reused['observation'] = initial['observation']
  reused['observation_ns'] = initial['observation_ns']
  projected = controller.update(**reused)
  assert projected.healthy and not projected.degraded and not present_leads(projected) and not projected.cleared
  assert not controller._tracks
  frame += 1
  fresh = controller.update(**scene(frame, tracks=[radar(frame, lateral=.9, lateral_speed=0.0)]))
  assert evidence(fresh)['pathState'] == 'inPath' and evidence(fresh)['brakeQualified']
  assert evidence(fresh)['positionMonoTime'] == START + frame * STEP and not evidence(fresh)['coasting']
  assert not controller._cleared


@pytest.mark.parametrize('visual', [False, True])
def test_position_coast_cannot_create_first_radar_occupancy_or_suppress_current_vision(visual):
  values = scene(tracks=[radar(coasting=True)], leads=[vision()] if visual else None,
                 bindings=[binding()] if visual else None)
  controller = HondaRadarControl()
  result = controller.update(**values)
  assert result.healthy and not result.degraded and not result.cleared and not controller._tracks
  assert len(present_leads(result)) == int(visual)
  if visual:
    assert evidence(result)['source'] == 'vision' and evidence(result)['brakeQualified']
    assert result.leads[0]['present'] and not result.leads[0]['radar']


def test_current_confirmed_visual_occupancy_does_not_require_a_new_radar_position():
  controller = HondaRadarControl()
  controller.update(**scene(tracks=[radar(lateral=1.7)]))
  result = controller.update(**scene(1, tracks=[radar(lateral=1.7)], leads=[vision()], bindings=[binding(bias_y=1.7)]))
  claim = evidence(result)
  assert result.healthy and not result.degraded and len(present_leads(result)) == 1
  assert claim['source'] == 'fused' and claim['pathState'] == 'inPath' and claim['pathSource'] == 'visionLead'
  assert claim['brakeQualified'] and claim['positionMonoTime'] == START
  assert claim['modelMonoTime'] == START + STEP and not claim['coasting']


def test_outer_object_stays_unselected_when_lateral_motion_is_uncertain():
  result = HondaRadarControl().update(**scene(tracks=[radar(lateral=1.6, lateral_speed=-.1, lateralVelocityVariance=1.0)]))
  assert not present_leads(result) and not result.cleared


def test_repeated_outer_measurements_do_not_accumulate_control_authority():
  controller = HondaRadarControl()
  for frame, lateral in enumerate((1.6, 1.5, 1.4)):
    result = controller.update(**scene(frame, tracks=[radar(frame, distance=8.0, speed=0.0,
                                                          lateral=lateral, lateral_speed=-.1)], ego=5.0))
    assert not present_leads(result) and not controller._tracks
    assert not result.degraded and not result.cleared


def test_side_reflection_loss_does_not_create_coasting_or_a_persistent_stop_latch():
  controller = HondaRadarControl()
  first = controller.update(**scene(tracks=[radar(distance=3.0, speed=0.0, lateral=1.6)], ego=4.0))
  assert not present_leads(first) and not controller._tracks
  for frame in (1, 7):
    missing = controller.update(**scene(frame, tracks=[], ego=4.0))
    assert missing.healthy and not missing.degraded and not present_leads(missing) and not missing.cleared


def test_near_independent_visual_lead_retains_authority_beside_a_side_reflection():
  visual = vision(distance=7.0, speed=3.0)
  result = HondaRadarControl().update(**scene(tracks=[radar(distance=3.0, speed=0.0, lateral=1.6)],
                                                   leads=[visual, {'present': False}]))
  assert len(present_leads(result)) == 1
  assert result.leads[0]['dRel'] == 7.0 and result.leads[0]['vLeadK'] == 3.0
  assert evidence(result)['source'] == 'vision' and evidence(result)['brakeQualified']


def test_visual_confirmation_does_not_give_later_outer_measurements_control_authority():
  controller = HondaRadarControl()
  initial = controller.update(**scene(tracks=[radar(distance=20.0, speed=8.0, lateral=1.6)],
                                     leads=[vision(distance=23.0, speed=11.0)],
                                     bindings=[binding(bias_x=-3.0, bias_y=1.6)]))
  assert evidence(initial)['source'] == 'fused' and evidence(initial)['pathSource'] == 'visionLead'
  assert initial.leads[0]['dRel'] == 20.0 and initial.leads[0]['vLeadK'] == 8.0
  for frame in range(1, 126):
    current = controller.update(**scene(frame, tracks=[radar(frame, distance=20.0, speed=8.0, lateral=1.6)]))
    assert current.healthy and not current.degraded and not current.cleared
    assert not present_leads(current) and not controller._tracks
  assert frame * STEP > 6_000_000_000


@pytest.mark.parametrize('generation', [1, 9], ids=['same-slot-generation', 'same-physical-new-generation'])
def test_inner_confirmation_cannot_extend_authority_to_the_outer_band(generation):
  controller = HondaRadarControl()
  initial = controller.update(**scene(tracks=[radar(lateral=1.0, speed=0.0)]))
  assert evidence(initial)['brakeQualified'] and evidence(initial)['stopQualified']
  edge = controller.update(**scene(1, tracks=[radar(1, lateral=1.7, speed=0.0, generation=generation)]))
  assert edge.healthy and not edge.degraded and not edge.cleared
  assert not present_leads(edge) and not controller._tracks
  outside = controller.update(**scene(2, tracks=[radar(2, lateral=2.5, speed=0.0)]))
  assert not present_leads(outside) and outside.cleared == (1,)
  returned = controller.update(**scene(3, tracks=[radar(3, lateral=1.7, speed=0.0)]))
  assert not present_leads(returned) and not controller._tracks
  direct = controller.update(**scene(4, tracks=[radar(4, lateral=1.0, speed=0.0)]))
  assert evidence(direct)['brakeQualified'] and evidence(direct)['stopQualified']
  assert direct.healthy and not direct.degraded


def test_current_same_car_vision_requalifies_offset_body_each_frame_despite_generation_changes():
  controller = HondaRadarControl()
  for frame in range(20):
    current = controller.update(**scene(frame,
      tracks=[radar(frame, distance=20.0, speed=8.0, lateral=1.6, generation=1 + frame // 3, slot=frame % 2)],
      leads=[vision(distance=23.0, speed=11.0)], bindings=[binding(bias_x=-3.0, bias_y=1.6)]))
    claim = evidence(current)
    assert current.healthy and not current.degraded and not current.cleared
    assert claim['source'] == 'fused' and claim['pathSource'] == 'visionLead'
    assert claim['trackId'] == 1 and claim['generation'] == 1 + frame // 3
    assert claim['positionMonoTime'] == claim['followGeometryMonoTime'] == START + frame * STEP
    assert all(claim[purpose + 'Qualified'] for purpose in ('follow', 'brake', 'stop', 'departure', 'fcw'))
    assert current.leads[0]['dRel'] == 20.0 and current.leads[0]['vLeadK'] == 8.0


@pytest.mark.parametrize('new_position', [False, True], ids=['intersample', 'independent-position'])
def test_current_path_change_revokes_inner_confirmation_without_declaring_body_clear(new_position):
  controller = HondaRadarControl()
  assert evidence(controller.update(**scene(tracks=[radar(lateral=1.0)])))['brakeQualified']
  changed = scene(1, tracks=[radar(int(new_position), lateral=1.0)])
  changed['model']['position']['y'] = [.6] * 33  # Current path puts the same reflection 1.6 m off center.
  withdrawn = controller.update(**changed)
  assert withdrawn.healthy and not withdrawn.degraded and not withdrawn.cleared
  assert not present_leads(withdrawn) and not controller._tracks
  current = controller.update(**scene(2, tracks=[radar(2, lateral=1.0)]))
  assert evidence(current)['followQualified'] and evidence(current)['positionMonoTime'] == START + 2 * STEP


def test_short_stop_trajectory_uses_quality_lanes_only_inside_spatial_support():
  near = HondaRadarControl().update(**scene(tracks=[radar(distance=30.0, speed=0.0)], short_path=True, lanes=True))
  assert evidence(near)['pathSource'] == 'laneLines'
  assert evidence(near)['stopQualified']
  far = HondaRadarControl().update(**scene(tracks=[radar(distance=120.0, speed=0.0)], short_path=True, lanes=True))
  assert not present_leads(far)
  bad = scene(short_path=True, lanes=True)
  bad['model']['laneLineProbs'][1] = .2
  assert not present_leads(HondaRadarControl().update(**bad))


def test_unknown_geometry_does_not_reuse_an_old_path_or_fake_departure():
  unknown = HondaRadarControl().update(**scene(short_path=True))
  assert not present_leads(unknown) and not unknown.cleared
  controller = HondaRadarControl()
  controller.update(**scene())
  gap = controller.update(**scene(1, short_path=True))
  assert not present_leads(gap) and gap.degraded and not gap.cleared
  assert not controller._tracks
  later = controller.update(**scene(7, short_path=True))
  assert not present_leads(later) and not later.degraded and not later.cleared


def test_crossing_does_not_gain_extra_braking_or_follow_memory():
  controller = HondaRadarControl()
  for frame in range(44):
    lateral = 1.0 - 2.0 * frame * STEP / 1e9
    result = controller.update(**scene(frame, tracks=[radar(frame, lateral=lateral, lateral_speed=-2.0, speed=0.0)], ego=0.0))
    assert not present_leads(result) and not controller._tracks


@pytest.mark.parametrize('speed,lateral_speed', [(0.0, -2.0), (4.0, .5), (4.0, -.5), (-2.01, 0.0)])
def test_current_crossing_or_opposing_motion_cannot_become_an_ordinary_lead(speed, lateral_speed):
  controller = HondaRadarControl()
  result = controller.update(**scene(tracks=[radar(speed=speed, lateral_speed=lateral_speed)]))
  assert not present_leads(result) and not controller._tracks


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
  assert qualified_lead(present_leads(result)[0], 'brake', now_ns=values['now_ns'])
  assert qualified_lead(present_leads(result)[0], 'brake')


def test_position_new_velocity_old_cannot_republish_previous_velocity():
  controller = HondaRadarControl()
  controller.update(**scene())
  for frame in (1, 4):
    values = scene(frame, tracks=[radar(frame, velocityMonoTime=START, velocityGood=False, observations=3,
                                      previousPositionMonoTime=0, previousPositionSeq=0)])
    result = controller.update(**values)
    assert not present_leads(result) and not controller._tracks
    # Too few independent current positions are ordinary pending evidence;
    # they do not turn a healthy sensor stream into a lead-health failure.
    assert result.healthy and not result.degraded
    assert not result.cleared
  missing = controller.update(**scene(5, tracks=[]))
  assert not present_leads(missing) and not missing.degraded and not missing.cleared


@pytest.mark.parametrize('identity', ['same', 'different'])
def test_missing_native_velocity_keeps_only_independent_current_visual_components(identity):
  track = radar(distance=20.0, velocityGood=False, velocityMonoTime=0, accelerationGood=False)
  visual = vision(distance=23.0, speed=7.0, acceleration=-.7)
  values = scene(tracks=[track], leads=[visual], bindings=[binding(identityState=identity, bias_x=-3.0)])
  result = HondaRadarControl().update(**values)
  lead, = present_leads(result)
  assert result.healthy and not result.degraded
  assert not lead['radar'] and lead['dRel'] == visual['dRel'] and lead['vLeadK'] == visual['vLeadK']
  assert lead['aLeadK'] == visual['aLeadK'] and evidence(result)['source'] == 'vision'
  assert evidence(result)['positionMonoTime'] == evidence(result)['velocityMonoTime'] == START
  assert 'visionVelocity' not in evidence(result)['reason']


def test_qualified_position_history_cannot_supply_missing_native_velocity():
  track = radar(speed=4.0, velocityGood=False, velocityMonoTime=0)
  assert track['positionVelocity']['qualified']
  result = HondaRadarControl().update(**scene(tracks=[track]))
  assert result.healthy and not result.degraded and not result.cleared and not present_leads(result)


@pytest.mark.parametrize('quality_changes', [
  None, {'version': 0}, {'qualified': False, 'reason': 'invalidVelocity'},
  {'sampleCount': 4, 'qualified': False, 'reason': 'insufficientSupport'},
  {'sampleCount': 5, 'firstPositionMonoTime': START + STEP - 240_000_000,
   'qualified': False, 'reason': 'insufficientSupport'},
  {'qualified': False, 'reason': 'awaitingAux'}, {'qualified': False, 'reason': 'pairTimeout'},
  {'qualified': False, 'reason': 'invalidPosition'}, {'qualified': False, 'reason': 'sourceGap'},
])
def test_pending_native_quality_withdraws_all_roles_and_recovers_only_from_current_qualified_pairs(quality_changes):
  controller = HondaRadarControl()
  assert evidence(controller.update(**scene()))['source'] == 'radar'
  current = radar(1)
  current['nativeMotion'] = None if quality_changes is None else native_motion_quality(current, **quality_changes)
  pending = controller.update(**scene(1, tracks=[current]))
  assert pending.healthy and not pending.degraded and not pending.cleared
  assert not present_leads(pending) and not controller._tracks
  assert pending.reason == 'noQualifiedObjects'
  # A newer publication alone is not support. The producer must independently
  # assert a newly completed, mature segment, rather than this adapter counting.
  repeated = controller.update(**scene(2, tracks=[current]))
  assert not present_leads(repeated) and not repeated.degraded
  recovered = controller.update(**scene(6, tracks=[radar(6, observations=5)]))
  assert recovered.healthy and not recovered.degraded and not recovered.cleared
  assert evidence(recovered)['positionMonoTime'] == START + 6 * STEP
  assert evidence(recovered)['velocityMonoTime'] == START + 6 * STEP


@pytest.mark.parametrize('variance', [.3, 5.0, 15.0])
def test_pending_native_quality_with_finite_uncertainty_does_not_degrade_another_lead(variance):
  controller = HondaRadarControl()
  initial = [radar(), radar(identifier=2, oem=8, distance=30.0)]
  assert len(present_leads(controller.update(**scene(tracks=initial)))) == 2
  pending = radar(1, velocityGood=False, velocityVariance=variance)
  pending['nativeMotion'] = native_motion_quality(pending, qualified=False, reason='invalidVelocity')
  other = radar(1, identifier=2, oem=8, distance=30.0)
  result = controller.update(**scene(1, tracks=[pending, other]))
  assert result.healthy and not result.degraded and not result.cleared
  assert [lead['evidence']['trackId'] for lead in present_leads(result)] == [2]
  assert set(controller._tracks) == {2}


def test_pending_native_quality_cannot_issue_measured_out_clearance():
  controller = HondaRadarControl()
  controller.update(**scene())
  current = radar(1, velocityGood=False, lateral=4.0)
  result = controller.update(**scene(1, tracks=[current]))
  assert result.healthy and not result.degraded and not result.cleared
  assert not present_leads(result) and not controller._tracks and not controller._cleared


@pytest.mark.parametrize('changes', [
  {'version': 2}, {'version': True}, {'version': []}, {'positionSeq': 99}, {'positionMonoTime': START},
  {'velocitySeq': 0}, {'velocityMonoTime': START}, {'sampleCount': 17}, {'sampleCount': 4}, {'sampleCount': True},
  {'firstPositionMonoTime': START + STEP - 249_999_999}, {'firstPositionMonoTime': 0},
  {'validUntilMonoTime': START + STEP + 200_000_001}, {'pairCompleteMonoTime': START},
  {'pairCompleteMonoTime': START + STEP + 2_000_000}, {'u10Raw': 256}, {'u10Raw': -1},
  {'frameIndex': 16}, {'existenceRaw': 128}, {'rangeSigmaRaw': None}, {'qualified': 1},
  {'rawRelativeSpeed': 13.5}, {'rawRelativeSpeed': math.nan}, {'groundSpeed': 11.0},
  {'velocityVariance': .4}, {'accelerationValid': False}, {'acceleration': -.5},
])
def test_contradictory_native_claim_preserves_existing_selected_lead_health_checks(changes):
  controller = HondaRadarControl()
  controller.update(**scene())
  current = radar(1)
  current['nativeMotion'] = native_motion_quality(current, **changes)
  result = controller.update(**scene(1, tracks=[current]))
  assert result.healthy and result.degraded and not result.cleared
  assert not present_leads(result) and not controller._tracks


@pytest.mark.parametrize('field', ['span', 'shortSpeed', 'longSpeed', 'speedStd', 'windowDifference', 'windowDifferenceStd',
                                   'innovationBias', 'posteriorCorrection', 'posteriorCorrectionRatio', 'posteriorSpeedDifference'])
def test_unused_nonfinite_position_velocity_diagnostics_cannot_veto_native_motion(field):
  controller = HondaRadarControl()
  controller.update(**scene())
  current = radar(1)
  current['positionVelocity'] = position_velocity_quality(current, qualified=False, **{field: math.nan})
  result = controller.update(**scene(1, tracks=[current]))
  assert result.healthy and not result.degraded and not result.cleared
  assert evidence(result)['source'] == 'radar' and result.leads[0]['vLeadK'] == current['groundSpeed']


@pytest.mark.parametrize('changes', [
  {'groundSpeed': math.nan}, {'lateralSpeed': math.inf}, {'velocityVariance': math.nan}, {'velocityVariance': -1.0},
])
def test_pending_quality_cannot_hide_corrupt_current_velocity_components(changes):
  controller = HondaRadarControl()
  controller.update(**scene())
  current = radar(1, velocityGood=False)
  current['nativeMotion'] = native_motion_quality(current, qualified=False)
  current.update(changes)
  result = controller.update(**scene(1, tracks=[current]))
  assert result.healthy and result.degraded and not result.cleared
  assert not present_leads(result) and not controller._tracks


def test_corrupt_native_claim_on_a_new_unselected_track_does_not_create_a_health_obligation():
  current = radar()
  current['nativeMotion'] = native_motion_quality(current, positionSeq=99)
  result = HondaRadarControl().update(**scene(tracks=[current]))
  assert result.healthy and not result.degraded and not result.cleared and not present_leads(result)


@pytest.mark.parametrize('visual', [False, True])
@pytest.mark.parametrize('quality_changes', [None, {'version': 0}, {'qualified': False}, {'positionSeq': 99}, {'speedStd': math.nan}])
def test_native_or_independent_visual_motion_does_not_depend_on_position_convergence(visual, quality_changes):
  current = radar(distance=20.0, velocityGood=not visual, velocityMonoTime=0 if visual else START)
  current['positionVelocity'] = None if quality_changes is None else position_velocity_quality(current, **quality_changes)
  inputs = scene(tracks=[current], leads=[vision(distance=23.0, speed=7.0)] if visual else None,
                 bindings=[binding(bias_x=-3.0)] if visual else None)
  result = HondaRadarControl().update(**inputs)
  assert result.healthy and not result.degraded and not result.cleared
  assert result.leads[0]['vLeadK'] == (7.0 if visual else 10.0)
  assert result.leads[0]['dRel'] == (23.0 if visual else 20.0)
  assert evidence(result)['source'] == ('vision' if visual else 'radar')


@pytest.mark.parametrize('visual', [False, True])
@pytest.mark.parametrize('nonfinite', [math.nan, math.inf, -math.inf], ids=['nan', 'positive-infinity', 'negative-infinity'])
def test_repeated_capnp_publication_with_unused_nonfinite_quality_preserves_its_identity(visual, nonfinite):
  from openpilot.cereal import custom
  controller = HondaRadarControl()
  current = radar(velocityGood=not visual, velocityMonoTime=0 if visual else START)
  current['positionVelocity']['speedStd'] = nonfinite
  current['nativeMotion']['distanceResidual'] = nonfinite
  inputs = scene(tracks=[current], leads=[vision(speed=7.0)] if visual else None,
                 bindings=[binding()] if visual else None)
  wire = custom.HondaRadarObservations.new_message(**inputs['observation']).to_bytes()
  for repeated in range(2):
    with custom.HondaRadarObservations.from_bytes(wire) as observation:
      result = controller.update(**{**inputs, 'observation': observation, 'now_ns': inputs['now_ns'] + repeated})
    assert result.healthy and not result.degraded and not result.cleared
    assert result.leads[0]['vLeadK'] == (7.0 if visual else 10.0)
    assert evidence(result)['source'] == ('vision' if visual else 'radar')

  missing_native = radar(1, velocityGood=False, velocityMonoTime=0)
  missing_native['positionVelocity']['speedStd'] = nonfinite
  inputs = scene(1, tracks=[missing_native])
  wire = custom.HondaRadarObservations.new_message(**inputs['observation']).to_bytes()
  with custom.HondaRadarObservations.from_bytes(wire) as observation:
    result = controller.update(**{**inputs, 'observation': observation})
  assert result.healthy and not result.degraded and not result.cleared
  assert not present_leads(result) and not controller._tracks


def test_same_capnp_publication_ignores_changes_to_unused_position_velocity_diagnostics():
  from openpilot.cereal import custom
  controller = HondaRadarControl()
  current = radar()
  current['positionVelocity']['speedStd'] = math.nan
  inputs = scene(tracks=[current])
  for value in (math.nan, .5):
    current['positionVelocity']['speedStd'] = value
    wire = custom.HondaRadarObservations.new_message(**inputs['observation']).to_bytes()
    with custom.HondaRadarObservations.from_bytes(wire) as observation:
      result = controller.update(**{**inputs, 'observation': observation})
    assert result.healthy and not result.degraded and present_leads(result)
    inputs['now_ns'] += 1


@pytest.mark.parametrize('field,value', [('qualified', False), ('groundSpeed', 11.0), ('sampleCount', 5)])
def test_same_publication_cannot_change_consumed_native_claim_in_place(field, value):
  controller = HondaRadarControl()
  original = scene()
  assert present_leads(controller.update(**original))
  original['now_ns'] += 1
  original['observation']['fusion']['tracks'][0]['nativeMotion'][field] = value
  result = controller.update(**original)
  assert not result.healthy and result.degraded and result.reason == 'publicationOrder'
  assert not present_leads(result) and not controller._tracks and not result.cleared


@pytest.mark.parametrize('count,span_ns', [(4, 264_000_000), (5, 249_999_999)])
def test_new_publications_of_one_native_pair_cannot_accumulate_confirmation_support(count, span_ns):
  controller = HondaRadarControl()
  current = radar()
  current['nativeMotion'] = native_motion_quality(current, sampleCount=count, firstPositionMonoTime=START - span_ns,
                                                 qualified=False, reason='insufficientSupport')
  for frame in range(3):
    result = controller.update(**scene(frame, tracks=[current]))
    assert result.healthy and not result.degraded and not result.cleared
    assert not present_leads(result) and not controller._tracks


def test_position_diagnostic_qualification_cannot_reenable_a_rejected_native_segment():
  controller = HondaRadarControl()
  current = radar(velocityGood=False)
  current['positionVelocity'] = position_velocity_quality(current, qualified=False)
  assert not present_leads(controller.update(**scene(tracks=[current])))
  current['positionVelocity'] = position_velocity_quality(current)
  repeated = controller.update(**scene(1, tracks=[current]))
  assert repeated.healthy and not repeated.degraded and not repeated.cleared
  assert not present_leads(repeated) and not controller._tracks
  fresh = controller.update(**scene(2, tracks=[radar(2)]))
  assert evidence(fresh)['positionMonoTime'] == START + 2 * STEP


@pytest.mark.parametrize('changes', [
  {'groundSpeed': math.nan, 'velocityGood': False, 'velocityMonoTime': 0, 'observations': 3},
  {'dRel': math.nan}, {'lateralSpeed': math.inf}, {'positionVariance': -1.0},
])
def test_invalid_components_do_not_become_zero_speed_or_fake_close_obstacles(changes):
  result = HondaRadarControl().update(**scene(tracks=[radar(**changes)], leads=[vision(distance=8.0)]))
  assert all(not lead['radar'] for lead in present_leads(result))
  assert result.leads[0]['dRel'] == 8.0


def test_duplicate_stale_publication_cannot_republish_a_saved_target():
  controller = HondaRadarControl()
  initial = scene()
  controller.update(**initial)
  for elapsed in (250_000_000, 350_000_001):
    old = copy.deepcopy(initial)
    old['now_ns'] = START + elapsed
    old['model_ns'] = old['model']['timestampEof'] = old['now_ns'] - 10_000_000
    result = controller.update(**old)
    assert not present_leads(result) and result.degraded and not result.cleared
    assert not controller._tracks


def test_same_sequence_changed_payload_is_rejected_and_new_sequence_recovers():
  controller = HondaRadarControl()
  original = scene()
  controller.update(**original)
  changed = copy.deepcopy(original)
  changed['now_ns'] += 1
  changed['observation']['fusion']['tracks'][0]['dRel'] = 18.0
  result = controller.update(**changed)
  assert not present_leads(result) and result.reason == 'publicationOrder' and result.degraded
  assert not controller._tracks
  restored = controller.update(**scene(1))
  assert restored.healthy and not restored.degraded and present_leads(restored)
  assert evidence(restored)['positionMonoTime'] == START + STEP


def test_out_of_order_source_rejects_without_giving_old_identity_new_lifetime():
  controller = HondaRadarControl()
  controller.update(**scene(1))
  old = scene(0)
  old['now_ns'] = START + STEP + 3_000_000
  old['model_ns'] = old['model']['timestampEof'] = START + STEP
  result = controller.update(**old)
  assert result.reason == 'publicationOrder' and not present_leads(result)
  assert not result.cleared


def test_session_and_reset_clear_identity_before_accepting_new_source():
  controller = HondaRadarControl()
  controller.update(**scene())
  restarted = scene(1)
  restarted['observation']['sessionId'] = 'new-session'
  result = controller.update(**restarted)
  assert result.reason == 'observerReset' and not present_leads(result) and not result.cleared
  assert not result.healthy and result.degraded and not controller._tracks
  recovered = scene(2)
  recovered['observation']['sessionId'] = 'new-session'
  result = controller.update(**recovered)
  assert present_leads(result) and evidence(result)['sessionId'] == 'new-session'
  assert result.healthy and not result.degraded
  assert evidence(result)['positionMonoTime'] == START + 2 * STEP
  # An explicit adapter/configuration reset also drops the active identity.
  controller.reset()
  reengaged = scene(3)
  reengaged['observation']['sessionId'] = 'new-session'
  result = controller.update(**reengaged)
  assert present_leads(result) and result.healthy and not result.degraded
  assert evidence(result)['positionMonoTime'] == START + 3 * STEP


def test_producer_qualified_slot_migration_preserves_physical_track_identity():
  controller = HondaRadarControl()
  controller.update(**scene())
  # The producer owns proof of a continuous slot migration. This fixture
  # explicitly provides qualified native pairs for that same physical track.
  # A generation rebuild with unqualified evidence is covered separately.
  result = controller.update(**scene(1, tracks=[radar(1, generation=9, slot=2)]))
  assert present_leads(result) and not result.degraded
  assert evidence(result)['trackId'] == 1 and evidence(result)['generation'] == 9
  assert evidence(result)['followQualified']


def test_explicit_invalid_position_revokes_instead_of_coasting_corrupted_state():
  controller = HondaRadarControl()
  controller.update(**scene())
  result = controller.update(**scene(1, tracks=[radar(1, active=False, lastInvalidReason='positionInnovation')]))
  assert not present_leads(result) and result.degraded and not result.cleared


@pytest.mark.parametrize('frame', [1, 7])
def test_missing_near_target_is_not_republished_beside_a_current_far_vehicle(frame):
  controller = HondaRadarControl()
  controller.update(**scene(tracks=[radar(distance=10.0)]))
  later = scene(frame, tracks=[radar(frame, identifier=2, oem=8, distance=60.0)])
  result = controller.update(**later)
  assert result.healthy and not result.degraded and not result.cleared
  assert {lead['evidence']['trackId'] for lead in present_leads(result)} == {2}
  assert set(controller._tracks) == {2}
  assert all(not lead['evidence']['coasting'] for lead in present_leads(result))


@pytest.mark.parametrize('speed', [0.0, 10.0])
def test_far_irrelevant_target_loss_does_not_latch_acceleration_block(speed):
  controller = HondaRadarControl()
  controller.update(**scene(tracks=[radar(distance=100.0, speed=speed)]))
  result = controller.update(**scene(7, tracks=[]))
  assert result.healthy and not result.degraded and not present_leads(result)
  assert not controller._tracks
  assert not result.cleared  # no invented positive claim that the road cleared


@pytest.mark.parametrize('lateral,clears', [(1.7, False), (4.0, True)])
def test_expired_track_returning_with_fresh_outside_geometry_can_clear_same_identity(lateral, clears):
  controller = HondaRadarControl()
  controller.update(**scene())
  missing = controller.update(**scene(7, tracks=[]))
  assert missing.healthy and not missing.degraded and not present_leads(missing)
  assert not controller._tracks
  outside = controller.update(**scene(8, tracks=[radar(8, lateral=lateral)]))
  assert outside.healthy and not outside.degraded and outside.session_id == 'control-fixture'
  assert outside.cleared == ((1,) if clears else ())
  assert not present_leads(outside) and not controller._tracks


def test_fresh_empty_packets_expire_missing_target_without_latching_risk():
  controller = HondaRadarControl()
  controller.update(**scene())
  missing = controller.update(**scene(1, tracks=[]))
  assert missing.healthy and not missing.degraded and not present_leads(missing)
  assert not controller._tracks
  for frame in (7, 8, 100, 500):
    result = controller.update(**scene(frame, tracks=[]))
    assert result.healthy and not result.degraded
    assert not present_leads(result) and not result.cleared
    assert not controller._tracks
  recovered = controller.update(**scene(501))
  assert present_leads(recovered) and not recovered.degraded
  assert evidence(recovered)['trackId'] == 1 and not evidence(recovered)['coasting']
  assert evidence(recovered)['positionMonoTime'] == START + 501 * STEP


def test_distinct_expired_tracks_do_not_accumulate_or_latch_capacity_fault():
  controller = HondaRadarControl()
  tracks = [radar(identifier=i, oem=i, distance=10.0) for i in range(1, MAX_TRACKS + 1)]
  controller.update(**scene(tracks=tracks))
  missing = controller.update(**scene(7, tracks=[]))
  assert missing.healthy and not missing.degraded and not present_leads(missing)
  assert not controller._tracks
  current = controller.update(**scene(8, tracks=[radar(8, identifier=MAX_TRACKS + 1, distance=10.0)]))
  assert current.healthy and not current.degraded and len(present_leads(current)) == 1
  assert set(controller._tracks) == {MAX_TRACKS + 1}
  expired = controller.update(**scene(15, tracks=[]))
  assert expired.healthy and not expired.degraded and not present_leads(expired)
  assert not controller._tracks
  recovered = [radar(16, identifier=i, oem=i, distance=10.0) for i in range(1, MAX_TRACKS + 1)]
  result = controller.update(**scene(16, tracks=recovered))
  assert result.healthy and not result.degraded and len(present_leads(result)) == 2
  assert set(controller._tracks) == {1, 2}


@pytest.mark.parametrize('boundary', ['disabled', 'core_invalid'])
def test_producer_reset_early_returns_clear_current_radar_memory(boundary):
  controller = HondaRadarControl()
  controller.update(**scene())
  current = controller.update(**scene(1))
  assert evidence(current)['source'] == 'radar' and set(controller._tracks) == {1}
  values = scene(2, tracks=[])
  if boundary == 'disabled':
    values['enabled'] = False
  else:
    values['v_ego'] = math.nan
  result = controller.update(**values)
  assert not present_leads(result) and not result.healthy
  assert result.degraded is (boundary == 'core_invalid')
  assert result.reason == ('disabled' if boundary == 'disabled' else 'coreInputInvalid')
  assert not controller._tracks and not controller._cleared and controller._session == ''


def test_same_confirmed_visual_stream_recovers_from_radar_loss_without_timeout():
  from openpilot.selfdrive.controls.lib.lead_qualification import qualified_lead
  controller = HondaRadarControl()
  controller.update(**scene(leads=[vision()], bindings=[binding()]))
  for frame in range(1, 15):
    values = scene(frame, leads=[vision()])
    values['observation_valid'] = False
    result = controller.update(**values)
    assert not result.healthy and not result.degraded
    assert len(present_leads(result)) == 1 and evidence(result)['reason'] == 'visionFallback'
    assert evidence(result)['source'] == 'fused' and evidence(result)['trackId'] == 1
    assert present_leads(result)[0]['dRel'] == 20.0 and not present_leads(result)[0]['radar']
    assert qualified_lead(present_leads(result)[0], 'brake', now_ns=values['now_ns'])


def test_different_far_visual_stream_cannot_claim_recovery_of_missing_near_radar():
  controller = HondaRadarControl()
  controller.update(**scene(leads=[vision()], bindings=[binding()]))
  changed = scene(1, leads=[vision(distance=60.0)])
  changed['observation_valid'] = False
  result = controller.update(**changed)
  assert result.degraded
  assert all(lead['evidence']['reason'] != 'visionFallback' for lead in present_leads(result))


def test_disabled_returns_exact_original_visual_fallback_and_clears_state():
  controller = HondaRadarControl()
  controller.update(**scene())
  values = scene(1, leads=[vision(distance=8.0), {'present': False}])
  values['enabled'] = False
  result = controller.update(**values)
  assert result.leads == tuple(values['vision_leads']) and result.objects == [] and not result.degraded
  assert result.reason == 'disabled' and not result.cleared
  assert not controller._tracks


def test_healthy_no_target_is_distinct_from_sensor_fault():
  empty = HondaRadarControl().update(**scene(tracks=[]))
  assert empty.healthy and not empty.degraded and not present_leads(empty)
  assert empty.reason == 'noQualifiedObjects'
  failed = scene(tracks=[])
  failed['observation_valid'] = False
  failure = HondaRadarControl().update(**failed)
  assert not failure.healthy and failure.degraded and failure.reason == 'observationStale'


def test_only_two_current_ordinary_leads_are_published():
  controller = HondaRadarControl()
  tracks = [radar(identifier=i, oem=i, distance=40.0 + i) for i in range(1, 33)]
  result = controller.update(**scene(tracks=tracks, leads=[vision(distance=7.0), vision(distance=9.0)]))
  assert result.healthy and not result.degraded
  assert [lead['dRel'] for lead in present_leads(result)] == [7.0, 9.0]
  assert not controller._tracks
  missing = controller.update(**scene(1, tracks=[]))
  assert not present_leads(missing) and not controller._tracks


def test_previously_unselected_third_lead_cannot_first_enter_via_component_coast():
  controller = HondaRadarControl()
  initial = [radar(identifier=i, oem=i, distance=10.0 * i) for i in range(1, 4)]
  result = controller.update(**scene(tracks=initial))
  assert [lead['evidence']['trackId'] for lead in present_leads(result)] == [1, 2]
  assert 3 not in controller._tracks
  # The two selected targets leave; the third has only a predicted position.
  tracks = [radar(1, identifier=i, oem=i, distance=10.0 * i, lateral=4.0) for i in (1, 2)]
  tracks.append(radar(1, identifier=3, oem=3, distance=30.0, coasting=True))
  result = controller.update(**scene(1, tracks=tracks))
  assert not present_leads(result) and 3 not in controller._tracks


@pytest.mark.parametrize('visual_fallback', [False, True])
def test_unselected_candidate_capacity_cannot_degrade_ordinary_follow_leads(visual_fallback):
  controller = HondaRadarControl()
  old = [radar(identifier=i, oem=i, distance=40.0 + i) for i in range(1, 33)]
  visual = [vision(distance=41.0)] if visual_fallback else None
  controller.update(**scene(tracks=old, leads=visual, bindings=[binding()] if visual_fallback else None))
  assert len(controller._tracks) <= 2
  new = [radar(1, identifier=i + 32, oem=i, distance=40.0 + i) for i in range(1, 33)]
  result = controller.update(**scene(1, tracks=new, leads=visual))
  assert len(present_leads(result)) == 2 and len(controller._tracks) <= 2
  assert result.healthy and not result.degraded
  assert result.leads[0]['evidence']['followQualified'] and result.leads[0]['evidence']['departureQualified']
  current = [radar(7, identifier=i + 32, oem=i, distance=40.0 + i) for i in range(1, 33)]
  recovered = controller.update(**scene(7, tracks=current))
  assert recovered.healthy and not recovered.degraded and len(present_leads(recovered)) == 2
  assert set(controller._tracks) == {33, 34}


@pytest.mark.parametrize('damage', ['lateral_nan', 'lateral_high', 'speed_nan', 'negative_projection',
                                   'position_nan', 'bad_oem', 'bad_sequence'])
def test_bad_unselected_side_track_cannot_change_ordinary_departure_or_health(damage):
  values = scene(tracks=[radar(distance=5.0, speed=2.0)], ego=0.0)
  values['observation']['fusion']['modelTimestampEof'] = START - 50_000_000
  baseline = HondaRadarControl().update(**copy.deepcopy(values))
  extra = radar(identifier=2, oem=8, distance=20.0, lateral=4.0)
  if damage == 'lateral_nan':
    extra['lateralSpeed'] = math.nan
  elif damage == 'lateral_high':
    extra['lateralSpeed'] = 31.0
  elif damage == 'speed_nan':
    extra['groundSpeed'] = math.nan
  elif damage == 'negative_projection':
    extra.update(dRel=.1, groundSpeed=-35.0)
  elif damage == 'position_nan':
    extra['dRel'] = math.nan
  elif damage == 'bad_oem':
    extra['oemTrackId'] = 0
  else:
    extra['positionSeq'] = 0
  values['observation']['fusion']['tracks'].append(extra)
  controller = HondaRadarControl()
  result = controller.update(**values)
  assert result.healthy and not result.degraded and result.leads == baseline.leads
  assert evidence(result)['departureQualified'] and evidence(result)['followQualified']
  assert set(controller._tracks) == {1}


@pytest.mark.parametrize('damage', ['bad_order', 'expired', 'invalid_velocity'])
def test_unselected_third_follow_candidate_cannot_retain_a_health_obligation(damage):
  def run(extra):
    controller = HondaRadarControl()
    for frame in (0, 1):
      tracks = [radar(frame, identifier=i, oem=i, distance=3.0 + i, speed=2.0) for i in (1, 2)]
      if extra:
        third = radar(frame, identifier=3, oem=3, distance=10.0, speed=2.0)
        if frame:
          if damage == 'bad_order':
            third.update(positionSeq=9, positionMonoTime=START - STEP)
          elif damage == 'expired':
            third.update(active=False, lastInvalidReason='expired')
          else:
            third.update(lateralSpeed=math.nan)
        tracks.append(third)
      result = controller.update(**scene(frame, tracks=tracks, ego=0.0))
      assert set(controller._tracks) == {1, 2}
    return result
  baseline, result = run(False), run(True)
  assert result.healthy and not result.degraded and result.leads == baseline.leads
  assert all(lead['evidence']['departureQualified'] for lead in present_leads(result))


@pytest.mark.parametrize('fault', ['overflow', 'duplicate_identity', 'stale_observation'])
def test_shared_observer_failures_still_degrade_the_ordinary_leads(fault):
  values = scene(tracks=[radar(distance=5.0, speed=2.0)], ego=0.0)
  if fault == 'overflow':
    values['observation']['fusion']['tracks'] = [radar(identifier=i, oem=i) for i in range(1, MAX_TRACKS + 2)]
  elif fault == 'duplicate_identity':
    values['observation']['fusion']['tracks'].append(radar(identifier=1, oem=8))
  else:
    values['observation_valid'] = False
  result = HondaRadarControl().update(**values)
  assert not result.healthy and result.degraded and not present_leads(result)


def test_formal_cereal_schema_publishes_ordinary_leads_and_no_extra_objects():
  from openpilot.cereal import log
  result = HondaRadarControl().update(**scene())
  state = log.RadarState.new_message()
  state.leadOne, state.leadTwo, state.radarObjects = result.leads[0], result.leads[1], result.objects
  assert state.leadOne.present and len(state.radarObjects) == 0
  assert str(state.leadOne.evidence.source) == 'radar'
  assert state.leadOne.evidence.conflictStart == 0.0 and state.leadOne.evidence.conflictEnd == HORIZON
  assert state.leadOne.evidence.conflictSupportVersion == 0 and not state.leadOne.evidence.physicalConflictValid


def test_legacy_loss_wire_fields_remain_readable_and_default_false():
  from openpilot.cereal import messaging
  # Old recordings may carry True; current producers leave these schema
  # fields at False instead of retaining a loss state to populate them.
  for service, field in (('radarState', 'fusionRiskUnresolved'), ('longitudinalPlan', 'radarRiskUnresolved')):
    event = messaging.new_message(service, valid=True)
    assert not getattr(getattr(event, service), field)
    wire = messaging.log_from_bytes(event.to_bytes())
    assert not getattr(getattr(wire, service), field)
    event.clear_write_flag()
    setattr(getattr(event, service), field, True)
    wire = messaging.log_from_bytes(event.to_bytes())
    assert getattr(getattr(wire, service), field)


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
  assert present_leads(result) and evidence(result)['accelerationValid']
  assert evidence(result)['source'] == 'radar' and result.leads[0]['modelProb'] == 0.0


# Frozen adapter-level geometry from road-pilot-1f86b1c/summary.json, route 8b,
# samples 0/2 (frames 2269/2274); companion images are pilot-nearby-frames-02.
# This old source did NOT record lateral covariances. The test deliberately
# supplies favorable synthetic .04/.01 covariances, rather than claiming they
# were measured, and proves that even trusted motion cannot create occupancy
# at t=0 from a currently outer-band reflection.
_PILOT_SIDE_REFLECTIONS = [{'sample': 0,
  'frame': 2269,
  'publication_ns': 136242133489,
  'model_ns': 136233587239,
  'eof_ns': 136190426270,
  'ego': 8.746807098388672,
  'x': 4.216198444366455,
  'y': -1.7217763662338257,
  'vx': -0.16044360399246216,
  'normal_rate': 0.24599501490592957,
  'position_ns': 136210734062,
  'velocity_ns': 136220964062,
  'p_var': 0.030461630975248122,
  'v_var': 0.5845580760245106,
  'path_x': [-3.9745984281580604e-14,
             0.08357738703489304,
             0.3347155749797821,
             0.7544216513633728,
             1.3450621366500854,
             2.1068882942199707,
             3.0417866706848145,
             4.145932674407959,
             5.432933807373047,
             6.902228832244873,
             8.54925537109375,
             10.397666931152344,
             12.397364616394043,
             14.597555160522461,
             16.985197067260742,
             19.547456741333008,
             22.290090560913086,
             25.194034576416016,
             28.206090927124023,
             31.420467376708984,
             34.79814910888672,
             38.31191635131836,
             41.903560638427734,
             45.70494079589844,
             49.57240295410156,
             53.660919189453125,
             57.77796936035156,
             61.96537399291992,
             66.18708038330078,
             70.63762664794922,
             75.09484100341797,
             79.53392028808594,
             84.11902618408203],
  'path_y': [2.0672685785427802e-11,
             0.0004960374790243804,
             0.001963886898010969,
             0.004851106554269791,
             0.009746572934091091,
             0.01761939935386181,
             0.02943246439099312,
             0.046008531004190445,
             0.06900731474161148,
             0.09992892295122147,
             0.1398097574710846,
             0.1879616379737854,
             0.24440155923366547,
             0.3103697597980499,
             0.38216540217399597,
             0.45976021885871887,
             0.5402615070343018,
             0.62256920337677,
             0.7048532366752625,
             0.7834905385971069,
             0.8576461672782898,
             0.9229746460914612,
             0.9842485785484314,
             1.0329585075378418,
             1.0755566358566284,
             1.1049113273620605,
             1.129662036895752,
             1.1485536098480225,
             1.154301404953003,
             1.1597819328308105,
             1.1645357608795166,
             1.1696650981903076,
             1.1650820970535278]},
 {'sample': 2,
  'frame': 2274,
  'publication_ns': 136492714010,
  'model_ns': 136483881093,
  'eof_ns': 136440433614,
  'ego': 9.178969383239746,
  'x': 2.513760805130005,
  'y': -1.5951945781707764,
  'vx': 0.35375237464904785,
  'normal_rate': 0.3788110017776489,
  'position_ns': 136413108228,
  'velocity_ns': 136413108228,
  'p_var': 0.027395029631493095,
  'v_var': 0.34474544626309367,
  'path_x': [4.084066418386101e-12,
             0.08551658689975739,
             0.3424296975135803,
             0.7709989547729492,
             1.3755521774291992,
             2.154404640197754,
             3.114359140396118,
             4.252651691436768,
             5.578297138214111,
             7.089252948760986,
             8.787909507751465,
             10.699236869812012,
             12.766985893249512,
             15.041107177734375,
             17.514619827270508,
             20.173503875732422,
             23.024845123291016,
             26.03857421875,
             29.16537094116211,
             32.523170471191406,
             36.03512191772461,
             39.702571868896484,
             43.45730209350586,
             47.420249938964844,
             51.48243713378906,
             55.75885772705078,
             60.12183380126953,
             64.5504150390625,
             69.00020599365234,
             73.71189880371094,
             78.47523498535156,
             83.23377990722656,
             88.10639953613281],
  'path_y': [1.2912060309844264e-11,
             0.0006036136182956398,
             0.002402914222329855,
             0.005783681757748127,
             0.011328121647238731,
             0.020148620009422302,
             0.033500418066978455,
             0.05253341794013977,
             0.07808314263820648,
             0.11107687652111053,
             0.15197566151618958,
             0.2005065381526947,
             0.25629085302352905,
             0.32038414478302,
             0.388311505317688,
             0.4598207473754883,
             0.5311398506164551,
             0.6023056507110596,
             0.6692330837249756,
             0.7291707992553711,
             0.7838699221611023,
             0.8272429704666138,
             0.8643614649772644,
             0.887698233127594,
             0.902121901512146,
             0.9058391451835632,
             0.9058959484100342,
             0.8944356441497803,
             0.8722158074378967,
             0.8496828079223633,
             0.8395617604255676,
             0.8172851204872131,
             0.7932747006416321]}]


@pytest.mark.parametrize('sample', _PILOT_SIDE_REFLECTIONS, ids=lambda row: str(row['frame']))
def test_recorded_side_reflection_has_no_control_even_with_favorable_motion_quality(sample):
  points = tuple((x - 1.52, -y) for x, y in zip(sample['path_x'], sample['path_y'], strict=True))
  current = _Corridor(points, 'modelPath').relation(sample['x'], sample['y'], sample['vx'], 0.0)
  assert abs(current.offset) > SWEPT_HALF_WIDTH
  # Invert the old producer's reported normal speed using its nearest segment;
  # no additional velocity samples or arbitrary zero-velocity prior are used.
  nearest = None
  for (px, py), (qx, qy) in zip(points, points[1:], strict=False):
    dx, dy = qx - px, qy - py
    length = math.hypot(dx, dy)
    along = ((sample['x'] - px) * dx + (sample['y'] - py) * dy) / length**2
    if not 0.0 <= along <= 1.0:
      continue
    offset = ((sample['y'] - py) * dx - (sample['x'] - px) * dy) / length
    if nearest is None or abs(offset) < nearest[0]:
      nearest = abs(offset), dx, dy, length
  assert nearest is not None
  _, dx, dy, length = nearest
  vy = (sample['normal_rate'] * length + sample['vx'] * dy) / dx
  track = radar(identifier=4, distance=sample['x'], lateral=sample['y'], speed=sample['vx'], lateral_speed=vy,
                positionMonoTime=sample['position_ns'], velocityMonoTime=sample['velocity_ns'],
                positionVariance=sample['p_var'], velocityVariance=sample['v_var'])
  values = scene(tracks=[track], ego=sample['ego'])
  values.update(now_ns=sample['publication_ns'], model_ns=sample['model_ns'], observation_ns=sample['publication_ns'] - 1)
  values['model'].update(timestampEof=sample['eof_ns'], position={'x': sample['path_x'], 'y': sample['path_y']})
  values['observation']['fusion'].update(modelMonoTime=sample['model_ns'], modelTimestampEof=sample['eof_ns'],
                                         egoMonoTime=sample['model_ns'])
  result = HondaRadarControl().update(**values)
  assert result.healthy and not result.degraded and not present_leads(result)
  assert not result.leads[0]['present'] and not result.leads[1]['present']
