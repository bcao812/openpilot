"""Active fusion object contracts; execution is from a committed host snapshot."""
import copy
import math
from dataclasses import replace

import pytest

from openpilot.selfdrive.car.honda_radar_control import COAST_NS, HORIZON, MAX_OBJECTS, MAX_TRACKS, SWEPT_HALF_WIDTH, HondaRadarControl, _Corridor, _corridors, _qualified_path
from openpilot.selfdrive.car.honda_radar_relevance import remaining_occupancy_window


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
            'lateralPositionVariance': .04, 'lateralVelocityVariance': .01, 'lateralVelocityValid': True,
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
    current = corridor.relation(x, y, 0.0, 0.0, horizon=0.0)
    # These are exactly the fields the lane/path disagreement gate consumes.
    assert (current.state == 'unknown', current.offset) == (full.state == 'unknown', full.offset), (points, x, y)
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


def test_latest_fresh_geometry_survives_coast_without_renewing_components_or_purposes():
  # Adapter message contract, not a vehicle/plant calibration. The planner can
  # miss the latest fresh publication while the producer has still seen it.
  controller = HondaRadarControl()
  assert evidence(controller.update(**scene()))['followGeometryMonoTime'] == START
  fresh = evidence(controller.update(**scene(1)))
  assert fresh['followGeometryMonoTime'] == START + STEP
  for frame in (2, 3):
    claim = evidence(controller.update(**scene(frame, tracks=[])))
    assert claim['followGeometryMonoTime'] == fresh['estimateMonoTime']
    assert claim['estimateMonoTime'] == START + frame * STEP
    for field in ('sessionId', 'trackId', 'generation', 'positionMonoTime', 'velocityMonoTime'):
      assert claim[field] == fresh[field]
    assert claim['coasting'] and claim['brakeQualified']
    assert not any(claim[field] for field in ('followQualified', 'stopQualified', 'departureQualified',
                                             'fcwQualified', 'accelerationValid', 'physicalConflictValid'))


@pytest.mark.parametrize('change', ['opposing', 'disjoint', 'finite', 'entering', 'out'])
def test_new_fresh_geometry_revocation_cannot_recover_an_older_follow_epoch(change):
  # Three successive producer frames exercise revocation with real path
  # calculation. They assert no frequency or likelihood of these transitions.
  controller = HondaRadarControl()
  initial, revised, coast = scene(), scene(1), scene(2, tracks=[])
  if change == 'opposing':
    initial = scene(tracks=[radar(distance=90.0)])
    revised = scene(1, tracks=[radar(1, distance=90.0, speed=-3.0)])
  elif change == 'disjoint':
    initial = scene(tracks=[radar(distance=1.0, speed=4.0)])
    revised = scene(1, tracks=[radar(1, distance=1.0, speed=4.0)])
    revised['model']['position'] = {'x': [1.52, 11.52, 21.52, 31.52, 51.52],
                                   'y': [0.0, 0.0, -10.0, 0.0, 0.0]}
  elif change == 'finite':
    revised = scene(1, tracks=[radar(1, lateral_speed=2.0)])
  elif change == 'entering':
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
  if change == 'out':
    assert not current.objects and current.cleared == (1,)
  else:
    assert current.objects
    claim = evidence(current)
    assert claim['generation'] == original['generation'] and claim['positionMonoTime'] == START + STEP
    assert not claim['coasting'] and not claim['followQualified']
    assert claim['followGeometryMonoTime'] == 0
  predicted = controller.update(**coast)
  assert bool(predicted.objects) is (change != 'out')
  assert all(lead['evidence']['followGeometryMonoTime'] == 0 for lead in predicted.objects)


@pytest.mark.parametrize('changes', [
  {'followGeometryMonoTime': None}, {'followGeometryMonoTime': 0}, {'followGeometryMonoTime': True},
  {'followGeometryMonoTime': START - 1}, {'followGeometryMonoTime': START + 2 * STEP},
  {'sessionId': 'different-session'}, {'trackId': 2}, {'generation': 2},
  {'positionMonoTime': START - 1}, {'velocityMonoTime': START - 1}, {'estimateMonoTime': START - 1},
])
def test_coast_cannot_invent_geometry_from_missing_or_mismatched_provenance(changes):
  # Corrupted internal evidence is a contract boundary, not a sensor model.
  controller = HondaRadarControl()
  controller.update(**scene())
  accepted = controller._tracks[1].accepted['evidence']
  accepted.update(changes)
  if changes.get('followGeometryMonoTime', 0) is None:
    accepted.pop('followGeometryMonoTime')
  coast = controller.update(**scene(1, tracks=[]))
  assert coast.objects and evidence(coast)['coasting']
  assert evidence(coast)['followGeometryMonoTime'] == 0


def test_new_immature_radar_waits_for_observer_maturity_not_visual_confirmation():
  controller = HondaRadarControl()
  for frame in range(3):
    sample = radar(frame, mature=frame == 2, observations=frame + 1)
    result = controller.update(**scene(frame, tracks=[sample]))
    assert bool(result.objects) is (frame == 2)


@pytest.mark.parametrize('start,end,age,expected', [
  (0.0, HORIZON, .05, (0.0, HORIZON)),
  (2.0, HORIZON, .25, (1.75, HORIZON)),
  (0.0, .3, .35, (0.0, -.05)),
  (.1, .3, .05, (.05, .25)),
  (0.0, HORIZON - 1e-6, .25, (0.0, HORIZON - 1e-6 - .25)),
])
def test_occupancy_age_preserves_only_the_exact_forecast_boundary(start, end, age, expected):
  assert remaining_occupancy_window(start, end, age) == pytest.approx(expected)


def test_far_historical_path_does_not_invent_a_crossing_or_latch_a_later_invalid_position():
  controller = HondaRadarControl()
  controller.update(**scene(tracks=[radar(distance=90.0, speed=-.65)]))
  assert not controller._tracks[1].relevant
  for frame in range(1, 4):
    result = controller.update(**scene(frame, tracks=[radar(frame, distance=90.0, speed=-.65)], short_path=True))
    claim = evidence(result)
    assert result.healthy and not result.degraded and not result.cleared
    assert claim['coasting'] and claim['conflictEnd'] == HORIZON
    assert not claim['physicalConflictValid']
    assert controller._tracks[1].path_ns == START and not controller._tracks[1].relevant
  result = controller.update(**scene(4, tracks=[radar(4, distance=90.0, active=False, lastInvalidReason='positionInnovation')]))
  assert result.healthy and not result.degraded and not result.objects and not result.cleared
  assert not controller._lost_tracks


@pytest.mark.parametrize('prediction_limited', [False, True])
def test_genuine_finite_history_still_expires_even_with_a_quality_limit(prediction_limited):
  controller = HondaRadarControl()
  controller.update(**scene())
  memory = controller._tracks[1]
  memory.path = replace(memory.path, end=.3, prediction_limited=prediction_limited)
  path = controller._historical_path(memory, START + COAST_NS)
  assert path.state == 'outOfPath' and not path.physical_valid
  assert memory.path_ns == memory.position_ns == memory.velocity_ns == START


def test_unknown_horizon_never_extends_near_measurement_or_path_authority():
  controller = HondaRadarControl()
  controller.update(**scene())
  memory = controller._tracks[1]
  assert controller._historical_path(memory, START + COAST_NS).end == HORIZON
  assert controller._historical_path(memory, START + COAST_NS + 1).state == 'unknown'
  result = controller.update(**scene(6, tracks=[]))
  assert not result.objects and result.degraded and not result.cleared
  assert controller._lost_tracks[('control-fixture', 1)].position_ns == START


@pytest.mark.parametrize('previous_distance,previous_lateral,current_lateral,expected_degraded', [
  (40.0, 1.6, 4.0, False),  # A roadside caution never had an independent braking obligation.
  (40.0, 1.6, 0.0, True),  # The same object can acquire a new current in-path obligation.
  (20.0, 0.0, 4.0, True),  # Bad velocity alone cannot clear a previously relevant obstacle.
  (80.0, 0.0, 0.0, True),  # A current in-path object still needs a usable motion component.
  (None, None, 0.0, True),  # No previous object is required for this current-position guard.
])
def test_bad_velocity_only_degrades_for_a_prior_risk_or_current_path_object(
    previous_distance, previous_lateral, current_lateral, expected_degraded):
  controller = HondaRadarControl()
  if previous_distance is not None:
    first = controller.update(**scene(tracks=[radar(distance=previous_distance, lateral=previous_lateral)]))
    assert first.healthy and not first.degraded
    assert len(first.objects) == 1
    assert controller._tracks[1].relevant is (previous_distance == 20.0)
  # Beyond the old velocity's coast deadline, with independent fresh
  # positions but no qualified raw, visual, or position-derived velocity.
  for frame in (7, 8):
    current = radar(frame, distance=40.0, lateral=current_lateral, velocityGood=False, velocityVariance=5.0)
    result = controller.update(**scene(frame, tracks=[current]))
    assert result.healthy and result.degraded is expected_degraded
    assert not result.objects and not result.cleared
  if previous_distance != 20.0:
    assert not controller._lost_tracks


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


def test_sloped_straight_path_uses_ground_velocity_normal_to_path():
  values = scene(tracks=[radar(lateral=2.152, lateral_speed=1.0)])
  values['model']['position']['y'] = [-.1 * x for x in values['model']['position']['x']]
  result = HondaRadarControl().update(**values)
  assert evidence(result)['pathState'] == 'inPath'
  assert abs(evidence(result)['pathOffset']) < 1e-9
  assert abs(evidence(result)['lateralSpeed']) < 1e-9
  assert evidence(result)['followQualified']


def test_current_outside_object_enters_a_later_bend_of_the_full_path():
  values = scene(tracks=[radar(distance=5.0, lateral=3.0, speed=5.0)])
  # Radar coordinates: (0, 0) -> (10, 0) -> (20, 10). The object has
  # zero current cross-track velocity, but meets the bend at (13, 3), t=1.6.
  values['model']['position'] = {'x': [1.52, 11.52, 21.52], 'y': [0.0, 0.0, -10.0]}
  result = HondaRadarControl().update(**values)
  assert len(result.objects) == 1
  entry = evidence(result)
  assert entry['pathState'] == 'entering' and entry['pathOffset'] == pytest.approx(3.0)
  assert entry['lateralSpeed'] == 0.0
  assert entry['conflictStart'] == pytest.approx((8.0 - SWEPT_HALF_WIDTH * math.sqrt(2.0)) / 5.0 - .15)
  assert entry['conflictEnd'] == pytest.approx((8.0 + SWEPT_HALF_WIDTH * math.sqrt(2.0)) / 5.0 + .25)
  assert entry['conflictStart'] < 1.6 < entry['conflictEnd']
  assert entry['brakeQualified'] and not entry['followQualified']


def test_prediction_does_not_extend_the_initial_tangent_past_a_turn():
  corridor = _Corridor(((0.0, 0.0), (10.0, 0.0), (20.0, 20.0)), 'modelPath')
  # The first tangent would suggest entry after 4 s, x=25 m. No such
  # straight corridor exists; the actual finite path has already turned.
  result = corridor.relation(5.0, -6.0, 5.0, 1.0)
  assert result.state == 'outOfPath'


def test_finite_bend_occupancy_does_not_fill_the_remaining_mpc_horizon():
  corridor = _Corridor(((0.0, 0.0), (10.0, 10.0), (20.0, 10.0)), 'modelPath')
  result = corridor.relation(5.0, 3.0, 5.0, 0.0)
  assert result.state == 'inPath'
  assert result.start == 0.0
  assert result.end == pytest.approx((2.0 * math.sqrt(2.0) - 2.0) / 5.0 + .25)


def test_internal_corner_has_no_gap_between_the_finite_segment_strips():
  corridor = _Corridor(((0.0, 0.0), (10.0, 0.0), (20.0, 10.0)), 'modelPath')
  # Outside both finite strips' along-segment intervals, inside the corner's
  # 2 m body-overlap disk. This is a real join, not a tangent extension.
  result = corridor.relation(10.5, -1.5, 0.0, 0.0)
  assert result.state == 'inPath'
  assert result.offset == pytest.approx(-math.sqrt(2.5))
  assert result.start == 0.0 and result.end == 10.0


def test_disjoint_bend_intervals_use_braking_envelope_without_permanent_follow():
  values = scene(tracks=[radar(distance=1.0, speed=4.0)])
  values['model']['position'] = {'x': [1.52, 11.52, 21.52, 31.52, 51.52],
                               'y': [0.0, 0.0, -10.0, 0.0, 0.0]}
  result = HondaRadarControl().update(**values)
  assert evidence(result)['conflictStart'] == 0.0 and evidence(result)['conflictEnd'] == 10.0
  assert evidence(result)['brakeQualified']
  assert not evidence(result)['followQualified'] and not evidence(result)['stopQualified']
  assert 'disjointOccupancyEnvelope' in evidence(result)['reason']


def test_short_crossing_interval_is_found_without_sampling_the_future():
  corridor = _Corridor(((0.0, 0.0), (20.0, 0.0)), 'modelPath')
  result = corridor.relation(10.0, -3.0, 0.0, 30.0)
  assert result.state == 'entering'
  assert result.start == 0.0 and result.end == pytest.approx(5.0 / 30.0 + .25)


@pytest.mark.parametrize('changes', [
  {'lateralVelocityValid': False}, {'lateralVelocityVariance': 4.01},
  {'lateralVelocityVariance': math.nan}, {'lateralPositionVariance': 4.0},
])
def test_lateral_prediction_limits_preserve_current_longitudinal_authority(changes):
  # Poor vy must not erase a newly observed stationary/forward-moving lead.
  for speed in (0.0, 4.0):
    result = HondaRadarControl().update(**scene(tracks=[radar(speed=speed, lateral_speed=2.0, **changes)]))
    claim = evidence(result)
    assert claim['pathState'] == 'inPath' and claim['conflictEnd'] == 10.0
    assert all(claim[purpose + 'Qualified'] for purpose in ('follow', 'brake', 'stop', 'fcw'))
    assert claim['departureQualified'] is (speed > .5)
    assert 'lateralPredictionLimited' in claim['reason']


def test_lateral_position_uncertainty_cannot_create_or_clear_an_obstacle():
  invalid = radar(lateralPositionVariance=4.01)
  result = HondaRadarControl().update(**scene(tracks=[invalid]))
  assert not result.objects and not result.cleared
  controller = HondaRadarControl()
  controller.update(**scene())
  uncertain = controller.update(**scene(1, tracks=[radar(1, lateral=1.0, lateralPositionVariance=4.01)]))
  assert evidence(uncertain)['source'] == 'predicted'
  assert uncertain.degraded and not uncertain.cleared


def test_qualified_entry_is_not_cleared_when_its_lateral_prediction_budget_ends():
  values = scene(tracks=[radar(distance=5.0, lateral=3.0, speed=5.0, lateralVelocityVariance=1.0)])
  values['model']['position'] = {'x': [1.52, 11.52, 21.52], 'y': [0.0, 0.0, -10.0]}
  result = HondaRadarControl().update(**values)
  # Trusted lateral horizon is (2 - sqrt(.04)) / sqrt(1) = 1.8 s;
  # entry is inside it, the clearing time is outside it.
  assert evidence(result)['pathState'] == 'entering'
  assert evidence(result)['conflictStart'] < 1.8 and evidence(result)['conflictEnd'] == 10.0
  assert evidence(result)['brakeQualified'] and not evidence(result)['departureQualified']


def test_entry_only_beyond_lateral_quality_horizon_is_unknown():
  corridor = _Corridor(((0.0, 0.0), (100.0, 0.0)), 'modelPath')
  result = _qualified_path((corridor,), 20.0, 4.0, 0.0, -.5, .04, 1.0, True)
  assert result.state == 'unknown' and result.prediction_limited
  assert result.observed_out  # current position remains measured geometry
  objects = HondaRadarControl().update(**scene(tracks=[radar(lateral=4.0, lateral_speed=-.5,
                                                           lateralVelocityVariance=1.0)]))
  assert not objects.objects and not objects.cleared


def test_fresh_measured_out_clears_old_lead_without_trusting_lateral_prediction():
  controller = HondaRadarControl()
  controller.update(**scene())
  controller.update(**scene(7, tracks=[]))
  result = controller.update(**scene(8, tracks=[radar(8, lateral=4.0, lateral_speed=-.5, lateralVelocityValid=False)]))
  assert result.healthy and not result.degraded
  assert result.cleared == (1,) and not result.objects


def test_finite_path_end_is_unknown_future_space_not_proof_of_departure():
  values = scene(tracks=[radar(distance=5.0, speed=10.0)])
  values['model']['position'] = {'x': [1.52, 11.52], 'y': [0.0, 0.0]}
  result = HondaRadarControl().update(**values)
  assert evidence(result)['pathState'] == 'inPath' and evidence(result)['conflictEnd'] == 10.0
  assert evidence(result)['followQualified'] and evidence(result)['departureQualified']
  assert 'lateralPredictionLimited' in evidence(result)['reason']


@pytest.mark.parametrize('lateral', [-1.7, -1.6, 1.6, 1.7])
def test_new_outer_only_stationary_object_is_caution_without_stop_or_fcw(lateral):
  from openpilot.selfdrive.controls.lib.lead_qualification import qualified_lead
  result = HondaRadarControl().update(**scene(tracks=[radar(distance=3.0, speed=0.0, lateral=lateral)], ego=4.0))
  assert len(result.objects) == 1 and not result.leads[0]['present'] and not result.leads[1]['present']
  target = result.objects[0]
  claim = target['evidence']
  assert target['dRel'] == 3.0 and target['vLeadK'] == 0.0 and target['modelProb'] == 0.0
  assert claim['trackId'] == 1 and claim['positionMonoTime'] == START
  assert claim['pathState'] == 'edge' and claim['cautionQualified']
  assert qualified_lead(target, 'caution', now_ns=START + 2_000_000)
  assert all(not claim[purpose + 'Qualified'] for purpose in ('brake', 'follow', 'stop', 'departure', 'fcw'))
  assert 'edgePossibleBodyOverlap' in claim['reason']


def test_new_inner_stationary_radar_is_a_full_obstacle_without_vision():
  result = HondaRadarControl().update(**scene(tracks=[radar(distance=3.0, speed=0.0, lateral=1.0)], ego=4.0))
  assert result.leads[0]['present'] and evidence(result)['source'] == 'radar'
  assert all(evidence(result)[purpose + 'Qualified'] for purpose in ('brake', 'follow', 'stop', 'fcw'))
  assert not evidence(result)['cautionQualified']


def test_outer_cut_in_is_sent_now_with_its_future_inner_occupancy():
  result = HondaRadarControl().update(**scene(tracks=[radar(distance=8.0, speed=0.0, lateral=1.7, lateral_speed=-1.0)], ego=5.0))
  claim = evidence(result)
  assert claim['pathState'] == 'entering' and claim['brakeQualified'] and claim['fcwQualified']
  assert not claim['cautionQualified'] and not claim['stopQualified']
  # Outer overlap exists now; it is not a license to invent an inner obstacle
  # at t=0. The MPC receives this interval now and can anticipate the cut-in.
  assert claim['conflictStart'] == pytest.approx((1.7 - SWEPT_HALF_WIDTH) - .15)
  assert claim['conflictEnd'] == pytest.approx((1.7 + SWEPT_HALF_WIDTH) + .25)
  assert claim['conflictStart'] > 0.0
  assert not result.leads[0]['present']


def test_outer_unknown_inner_prediction_is_recorded_without_fake_clear():
  result = HondaRadarControl().update(**scene(tracks=[radar(lateral=1.6, lateral_speed=-.1, lateralVelocityVariance=1.0)]))
  assert evidence(result)['pathState'] == 'edge' and evidence(result)['cautionQualified']
  assert 'edgeInnerUnknown' in evidence(result)['reason'] and not result.cleared


def test_unknown_inner_does_not_retire_a_previous_future_collision_risk():
  controller = HondaRadarControl()
  first = controller.update(**scene(tracks=[radar(distance=8.0, speed=0.0, lateral=1.6, lateral_speed=-2.0)], ego=5.0))
  assert evidence(first)['pathState'] == 'entering' and evidence(first)['brakeQualified']
  later = controller.update(**scene(1, tracks=[radar(1, distance=8.0, speed=0.0, lateral=1.5, lateral_speed=-.1,
                                                   lateralVelocityVariance=1.0)], ego=5.0))
  assert evidence(later)['source'] == 'predicted' and evidence(later)['brakeQualified']
  assert later.degraded and not later.cleared


def test_caution_only_loss_does_not_create_coasting_or_a_persistent_stop_latch():
  controller = HondaRadarControl()
  first = controller.update(**scene(tracks=[radar(distance=3.0, speed=0.0, lateral=1.6)], ego=4.0))
  assert evidence(first)['cautionQualified']
  missing = controller.update(**scene(1, tracks=[], ego=4.0))
  assert missing.healthy and not missing.degraded and not missing.objects and not missing.cleared
  expired = controller.update(**scene(7, tracks=[], ego=4.0))
  assert not expired.degraded and not expired.objects
  coast = HondaRadarControl().update(**scene(tracks=[radar(lateral=1.6, coasting=True)]))
  assert not coast.objects


def test_near_independent_visual_lead_retains_authority_beside_radar_caution():
  visual = vision(distance=7.0, speed=3.0)
  result = HondaRadarControl().update(**scene(tracks=[radar(distance=3.0, speed=0.0, lateral=1.6)],
                                                   leads=[visual, {'present': False}]))
  assert len(result.objects) == 2
  assert result.leads[0]['dRel'] == 7.0 and result.leads[0]['vLeadK'] == 3.0
  assert result.leads[0]['evidence']['source'] == 'vision' and result.leads[0]['evidence']['brakeQualified']
  assert next(lead for lead in result.objects if lead['radar'])['evidence']['cautionQualified']


def test_confirmed_visual_body_occupancy_keeps_radar_motion_through_long_visual_gap():
  controller = HondaRadarControl()
  initial = controller.update(**scene(tracks=[radar(distance=20.0, speed=8.0, lateral=1.6)],
                                     leads=[vision(distance=23.0, speed=11.0)],
                                     bindings=[binding(bias_x=-3.0, bias_y=1.6)]))
  assert evidence(initial)['source'] == 'fused' and evidence(initial)['pathSource'] == 'visionLead'
  assert initial.leads[0]['dRel'] == 20.0 and initial.leads[0]['vLeadK'] == 8.0
  for frame in range(1, 22):
    current = controller.update(**scene(frame, tracks=[radar(frame, distance=20.0, speed=8.0, lateral=1.6)]))
    claim = evidence(current)
    assert claim['source'] == 'radar' and not claim['coasting'] and not claim['cautionQualified']
    assert all(claim[purpose + 'Qualified'] for purpose in ('brake', 'follow', 'stop', 'departure', 'fcw'))
    assert current.leads[0]['modelProb'] == 0.0 and current.leads[0]['vLeadK'] == 8.0
    assert 'confirmedOccupancyHysteresis' in claim['reason'] and not current.degraded
  assert frame * STEP > 883_000_000


def test_inner_confirmation_uses_two_metre_exit_hysteresis_then_resets():
  controller = HondaRadarControl()
  controller.update(**scene(tracks=[radar(lateral=1.0, speed=0.0)]))
  edge = controller.update(**scene(1, tracks=[radar(1, lateral=1.7, speed=0.0)]))
  assert evidence(edge)['brakeQualified'] and evidence(edge)['stopQualified']
  assert not evidence(edge)['cautionQualified'] and not edge.cleared
  outside = controller.update(**scene(2, tracks=[radar(2, lateral=2.5, speed=0.0)]))
  assert not outside.objects and outside.cleared == (1,)
  returned = controller.update(**scene(3, tracks=[radar(3, lateral=1.7, speed=0.0)]))
  assert evidence(returned)['cautionQualified'] and not evidence(returned)['brakeQualified']


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
  assert evidence(start)['conflictStart'] == pytest.approx((3.0 - SWEPT_HALF_WIDTH) / 2.0 - .15)
  assert evidence(start)['conflictEnd'] == pytest.approx((3.0 + SWEPT_HALF_WIDTH) / 2.0 + .25)
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
  assert evidence(result)['conflictEnd'] == pytest.approx(SWEPT_HALF_WIDTH / 2.0 + .25)
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
  assert evidence(result)['followGeometryMonoTime'] == 0
  assert result.objects[0]['aLeadK'] == 0.0
  assert qualified_lead(result.objects[0], 'brake', now_ns=values['now_ns'])
  coast = controller.update(**scene(5, tracks=[]))
  assert coast.objects and evidence(coast)['coasting']
  assert evidence(coast)['followGeometryMonoTime'] == 0
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
  assert result.degraded and not result.risk_unresolved
  assert evidence(result)['positionMonoTime'] == START
  from openpilot.selfdrive.controls.lib.lead_qualification import qualified_lead
  assert qualified_lead(result.objects[0], 'brake', now_ns=coast['now_ns'])
  coast['now_ns'] = START + COAST_NS + 1
  coast['model_ns'] = coast['model']['timestampEof'] = START + COAST_NS
  result = controller.update(**coast)
  assert not result.objects and result.degraded and not result.cleared
  assert result.risk_unresolved


def test_same_sequence_changed_payload_is_rejected_and_new_sequence_recovers():
  controller = HondaRadarControl()
  original = scene()
  controller.update(**original)
  changed = copy.deepcopy(original)
  changed['now_ns'] += 1
  changed['observation']['fusion']['tracks'][0]['dRel'] = 18.0
  result = controller.update(**changed)
  assert not result.objects and result.reason == 'publicationOrder' and result.degraded
  assert result.risk_unresolved
  restored = controller.update(**scene(1))
  assert restored.healthy and not restored.degraded and restored.objects
  assert not restored.risk_unresolved


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
  assert result.risk_unresolved
  recovered = scene(2)
  recovered['observation']['sessionId'] = 'new-session'
  result = controller.update(**recovered)
  assert result.objects and evidence(result)['sessionId'] == 'new-session' and result.degraded
  assert result.risk_unresolved
  # An explicit adapter/configuration reset starts a new perception lifecycle.
  # Driver longActive edges retain this history; observer restart and TTL do
  # not prove that the old obstacle departed.
  controller.reset()
  reengaged = scene(3)
  reengaged['observation']['sessionId'] = 'new-session'
  result = controller.update(**reengaged)
  assert result.objects and result.healthy and not result.degraded
  assert not result.risk_unresolved


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
  assert not result.risk_unresolved
  assert not result.cleared  # no invented positive claim that the road cleared


def test_lost_same_track_can_clear_with_fresh_out_of_path_even_after_memory_expiry():
  controller = HondaRadarControl()
  controller.update(**scene())
  missing = controller.update(**scene(7, tracks=[]))
  assert missing.degraded and missing.risk_unresolved
  cleared = controller.update(**scene(8, tracks=[radar(8, lateral=4.0)]))
  assert cleared.healthy and not cleared.degraded and cleared.cleared == (1,)
  assert not cleared.risk_unresolved


def test_unresolved_risk_survives_fresh_empty_packets_until_same_identity_recovers():
  controller = HondaRadarControl()
  controller.update(**scene())
  for frame in (7, 8, 100, 500):
    result = controller.update(**scene(frame, tracks=[]))
    assert result.healthy and result.degraded and result.risk_unresolved
    assert not result.objects and not result.cleared
  recovered = controller.update(**scene(501))
  assert recovered.objects and not recovered.degraded and not recovered.risk_unresolved


def test_unresolved_loss_capacity_overflow_stays_explicit_until_lifecycle_reset():
  controller = HondaRadarControl()
  tracks = [radar(identifier=i, oem=i, distance=10.0) for i in range(1, MAX_TRACKS + 1)]
  controller.update(**scene(tracks=tracks))
  missing = controller.update(**scene(7, tracks=[]))
  assert missing.risk_unresolved and len(controller._lost_tracks) == MAX_TRACKS
  controller.update(**scene(8, tracks=[radar(8, identifier=MAX_TRACKS + 1, distance=10.0)]))
  overflow = controller.update(**scene(15, tracks=[]))
  assert overflow.risk_unresolved and controller._unresolved_loss
  recovered = [radar(16, identifier=i, oem=i, distance=10.0) for i in range(1, MAX_TRACKS + 1)]
  result = controller.update(**scene(16, tracks=recovered))
  assert not controller._lost_tracks and result.risk_unresolved and result.degraded
  controller.reset()
  result = controller.update(**scene(17, tracks=[]))
  assert not result.risk_unresolved


@pytest.mark.parametrize('boundary', ['disabled', 'core_invalid'])
def test_producer_reset_early_returns_cannot_republish_previous_loss(boundary):
  controller = HondaRadarControl()
  controller.update(**scene())
  assert controller.update(**scene(7, tracks=[])).risk_unresolved
  values = scene(8, tracks=[])
  if boundary == 'disabled':
    values['enabled'] = False
  else:
    values['v_ego'] = math.nan
  result = controller.update(**values)
  assert not result.risk_unresolved and not controller._lost_tracks and not controller._unresolved_loss


def test_same_confirmed_visual_stream_recovers_from_radar_loss_without_timeout():
  from openpilot.selfdrive.controls.lib.lead_qualification import qualified_lead
  controller = HondaRadarControl()
  controller.update(**scene(leads=[vision()], bindings=[binding()]))
  for frame in range(1, 15):
    values = scene(frame, leads=[vision()])
    values['observation_valid'] = False
    result = controller.update(**values)
    assert not result.healthy and not result.degraded
    assert not result.risk_unresolved
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
  assert not empty.risk_unresolved and not failure.risk_unresolved


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


def test_retained_loss_capnp_wire_and_old_payload_defaults():
  from openpilot.cereal import messaging
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
  assert result.objects and evidence(result)['accelerationValid']
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
def test_recorded_side_reflection_uses_future_inner_interval_even_with_favorable_quality(sample):
  points = tuple((x - 1.52, -y) for x, y in zip(sample['path_x'], sample['path_y'], strict=True))
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
  claim = evidence(result)
  assert result.healthy and not result.degraded and len(result.objects) == 1
  assert claim['trackId'] == 4 and claim['pathState'] == 'entering' and claim['brakeQualified']
  assert claim['conflictStart'] > .8  # current OUT is not a fabricated close in-path obstacle
  assert not claim['followQualified'] and not claim['stopQualified'] and not claim['departureQualified']
  assert not result.leads[0]['present']
