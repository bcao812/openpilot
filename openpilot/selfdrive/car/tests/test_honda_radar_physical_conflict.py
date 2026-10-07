"""Unpadded point-sweep evidence is separate from ordinary control windows.

The physical fields describe the current geometric/motion hypothesis. They
are neither scene ground truth nor a guarantee of collision avoidance.
"""
import math

import pytest

from openpilot.selfdrive.car.honda_radar_control import HondaRadarControl, SWEPT_HALF_WIDTH, _Corridor
from openpilot.selfdrive.car.tests.test_honda_radar_control import START, STEP, binding, evidence, radar, scene, vision


def assert_no_physical(claim):
  assert claim['physicalConflictValid'] is False
  assert claim['physicalConflictStart'] == claim['physicalConflictEnd'] == 0.0


@pytest.mark.parametrize('lateral_speed,end', [(5.0, .22), (11.0, .1)])
def test_current_short_sweep_publishes_original_window_not_reverse_padding(lateral_speed, end):
  result = HondaRadarControl().update(**scene(ego=15.0, tracks=[
    radar(distance=6.0, speed=0.0, lateral_speed=lateral_speed),
  ]))
  claim = evidence(result)
  assert claim['physicalConflictValid'] is True
  assert claim['physicalConflictStart'] == 0.0
  assert claim['physicalConflictEnd'] == pytest.approx(end)
  assert claim['conflictStart'] == 0.0
  assert claim['conflictEnd'] == pytest.approx(end + .25)
  assert claim['brakeQualified'] and claim['fcwQualified']
  assert not claim['followQualified'] and not claim['stopQualified']
  assert claim['estimateMonoTime'] == START
  if end == .1:
    # An attempted inverse erosion would produce [.15, .10], losing this
    # valid short interval. It must be saved before the t=0 clipping occurs.
    assert claim['conflictStart'] + .15 > claim['conflictEnd'] - .25


def test_future_single_sweep_retains_unpadded_entry_and_exit():
  result = HondaRadarControl().update(**scene(tracks=[
    radar(distance=20.0, speed=0.0, lateral=3.0, lateral_speed=-2.0),
  ]))
  claim = evidence(result)
  assert claim['physicalConflictValid']
  assert claim['physicalConflictStart'] == pytest.approx((3.0 - SWEPT_HALF_WIDTH) / 2.0)
  assert claim['physicalConflictEnd'] == pytest.approx((3.0 + SWEPT_HALF_WIDTH) / 2.0)
  assert claim['conflictStart'] == pytest.approx(claim['physicalConflictStart'] - .15)
  assert claim['conflictEnd'] == pytest.approx(claim['physicalConflictEnd'] + .25)


def test_padding_cannot_join_disjoint_raw_intervals_into_physical_evidence():
  corridor = _Corridor(((0.0, 0.0), (10.0, 0.0), (12.0, 3.0), (14.0, 0.0), (24.0, 0.0)), 'modelPath')
  path = corridor.relation(1.0, 0.0, 10.0, 0.0, horizon=2.0, half_width=SWEPT_HALF_WIDTH)
  # At x=12, y=0 the distance from either sloping segment is 6/sqrt(13),
  # outside the actual 1.10 m sweep. The .15/.25 timing margins fill that gap.
  assert 6.0 / math.sqrt(13.0) > SWEPT_HALF_WIDTH
  assert path.continuous and path.start == 0.0 and path.end == 2.0
  assert not path.physical_valid and path.physical_start == path.physical_end == 0.0


def test_disjoint_braking_envelope_keeps_ordinary_authority_without_physical_claim():
  values = scene(tracks=[radar(distance=1.0, speed=4.0)])
  values['model']['position'] = {'x': [1.52, 11.52, 21.52, 31.52, 51.52],
                               'y': [0.0, 0.0, -10.0, 0.0, 0.0]}
  claim = evidence(HondaRadarControl().update(**values))
  assert claim['conflictStart'] == 0.0 and claim['conflictEnd'] == 10.0
  assert claim['brakeQualified'] and claim['fcwQualified']
  assert not claim['followQualified'] and not claim['stopQualified']
  assert_no_physical(claim)


@pytest.mark.parametrize('limit', ['current_quality', 'future_quality', 'spatial', 'missing_lateral_velocity'])
def test_quality_or_spatial_extension_cannot_extend_the_independent_physical_window(limit):
  if limit == 'current_quality':
    # The lateral horizon is (2 - 1.7) / 1 = .3 s. Padding hits it and
    # the ordinary current-occupancy policy extends its control window to 10 s.
    values = scene(tracks=[radar(distance=6.0, speed=0.0, lateral_speed=5.0,
                                 lateralPositionVariance=1.7**2, lateralVelocityVariance=1.0)])
  elif limit == 'future_quality':
    values = scene(tracks=[radar(distance=5.0, lateral=3.0, speed=5.0, lateralVelocityVariance=1.0)])
    values['model']['position'] = {'x': [1.52, 11.52, 21.52], 'y': [0.0, 0.0, -10.0]}
  elif limit == 'spatial':
    values = scene(tracks=[radar(distance=5.0, speed=10.0)])
    values['model']['position'] = {'x': [1.52, 11.52], 'y': [0.0, 0.0]}
  else:
    values = scene(tracks=[radar(speed=0.0, lateralVelocityValid=False)])
  claim = evidence(HondaRadarControl().update(**values))
  assert claim['conflictEnd'] == 10.0 and claim['brakeQualified'] and claim['fcwQualified']
  if limit != 'future_quality':
    assert claim['followQualified'] and claim['stopQualified']
  assert_no_physical(claim)


def test_only_current_inner_sweep_can_supply_physical_evidence():
  controller = HondaRadarControl()
  first = controller.update(**scene(tracks=[radar(speed=0.0)]))
  assert evidence(first)['physicalConflictValid']
  # The same track retains ordinary authority through the existing 2 m exit
  # hysteresis, but its current reflection lies outside the 1.10 m point sweep.
  second = controller.update(**scene(1, tracks=[radar(1, speed=0.0, lateral=1.6)]))
  claim = evidence(second)
  assert claim['followQualified'] and claim['stopQualified'] and claim['fcwQualified']
  assert_no_physical(claim)
  outer = HondaRadarControl().update(**scene(tracks=[radar(speed=0.0, lateral=1.6)]))
  assert evidence(outer)['cautionQualified']
  assert_no_physical(evidence(outer))


def test_visual_path_and_visual_identity_fallback_do_not_reuse_an_old_geometric_window():
  visual = vision(distance=6.0, speed=0.0, ego=0.0)
  only_visual = HondaRadarControl().update(**scene(ego=0.0, tracks=[], leads=[visual, {'present': False}]))
  assert_no_physical(evidence(only_visual))
  synthetic_path = HondaRadarControl().update(**scene(
    ego=0.0, tracks=[radar(distance=6.0, speed=0.0, lateral=1.6)],
    leads=[vision(distance=6.0, speed=0.0, ego=0.0, lateral=1.6), {'present': False}], bindings=[binding()],
  ))
  assert evidence(synthetic_path)['pathSource'] == 'visionLead' and evidence(synthetic_path)['source'] == 'fused'
  assert evidence(synthetic_path)['followQualified'] and evidence(synthetic_path)['fcwQualified']
  assert_no_physical(evidence(synthetic_path))

  controller = HondaRadarControl()
  first = controller.update(**scene(ego=0.0, tracks=[radar(distance=6.0, speed=0.0, lateral_speed=5.0)],
                                    leads=[visual, {'present': False}], bindings=[binding()]))
  assert evidence(first)['physicalConflictValid']
  fallback = controller.update(**scene(1, ego=0.0, tracks=[],
                                       leads=[vision(distance=6.0, speed=0.0, ego=0.0, lateral=.33), {'present': False}]))
  assert evidence(fallback)['source'] == 'fused' and evidence(fallback)['pathSource'] == 'visionLead'
  assert_no_physical(evidence(fallback))


@pytest.mark.parametrize('gap', ['missing', 'migration_wait', 'position_coast', 'velocity_coast', 'unknown_path'])
def test_predicted_or_historical_path_clears_physical_evidence(gap):
  controller = HondaRadarControl()
  controller.update(**scene(ego=0.0, tracks=[radar(distance=6.0, speed=0.0, lateral_speed=5.0)]))
  historical = controller._historical_path(controller._tracks[1], START + STEP)
  assert not historical.physical_valid and historical.physical_start == historical.physical_end == 0.0
  track = radar(1, distance=6.0, speed=0.0, lateral=.33, lateral_speed=5.0)
  if gap == 'migration_wait':
    track.update(active=False, lastInvalidReason='slotReassigned')
  elif gap == 'position_coast':
    track['coasting'] = True
  elif gap == 'velocity_coast':
    track.update(velocityGood=False, velocityVariance=5.0)
  values = scene(1, ego=0.0, tracks=[] if gap == 'missing' else [track], short_path=gap == 'unknown_path')
  claim = evidence(controller.update(**values))
  assert claim['source'] == 'predicted' and claim['coasting'] and claim['brakeQualified']
  assert_no_physical(claim)


def test_fresh_migrated_track_recomputes_the_window_at_its_new_estimate_epoch():
  controller = HondaRadarControl()
  first = controller.update(**scene(ego=0.0, tracks=[radar(distance=6.0, speed=0.0, lateral_speed=5.0, generation=9)]))
  next_track = radar(1, distance=6.0, speed=0.0, lateral=.33, lateral_speed=5.0, generation=0, slot=2)
  second = controller.update(**scene(1, ego=0.0, tracks=[next_track]))
  old, new = evidence(first), evidence(second)
  assert old['trackId'] == new['trackId'] == 1 and new['generation'] == 0
  assert old['physicalConflictValid'] and new['physicalConflictValid']
  assert new['estimateMonoTime'] == START + STEP
  assert old['physicalConflictEnd'] == pytest.approx(.22)
  assert new['physicalConflictEnd'] == pytest.approx((SWEPT_HALF_WIDTH - .33) / 5.0)
  assert new['physicalConflictEnd'] < old['physicalConflictEnd']
  assert new['estimateMonoTime'] + round(new['physicalConflictEnd'] * 1e9) == START + 220_000_000


def test_physical_window_is_double_precision_and_old_schema_defaults_do_not_grant_it():
  from openpilot.cereal import log

  result = HondaRadarControl().update(**scene(tracks=[radar(distance=6.0, speed=0.0, lateral_speed=7.0)]))
  claim = evidence(result)
  assert claim['physicalConflictValid'] and claim['physicalConflictEnd'] == SWEPT_HALF_WIDTH / 7.0
  wire = log.RadarState.new_message(radarObjects=result.objects, leadOne=result.leads[0], leadTwo=result.leads[1])
  with log.RadarState.from_bytes(wire.to_bytes()) as decoded:
    assert decoded.radarObjects[0].evidence.physicalConflictValid
    assert decoded.radarObjects[0].evidence.physicalConflictStart == 0.0
    assert decoded.radarObjects[0].evidence.physicalConflictEnd == claim['physicalConflictEnd']
  legacy = log.RadarState.new_message(leadOne={'present': True, 'modelProb': .99})
  old_claim = legacy.as_reader().leadOne.evidence
  assert not old_claim.physicalConflictValid
  assert old_claim.physicalConflictStart == old_claim.physicalConflictEnd == 0.0
