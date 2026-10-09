"""Prediction support never becomes clearance or a renewed coast lifetime."""
from copy import deepcopy

import pytest

from openpilot.cereal import log
from openpilot.selfdrive.car.honda_radar_control import HORIZON, HondaRadarControl, _Corridor, _qualified_path
from openpilot.selfdrive.car.tests.test_honda_radar_control import START, STEP, evidence, radar, scene
from openpilot.selfdrive.controls.lib.lead_qualification import qualified_lead


@pytest.mark.parametrize('limitation', ['none', 'quality', 'spatial', 'no_lateral_velocity'])
def test_support_records_prediction_before_unknown_exit_extension(limitation):
  corridor = _Corridor(((0.0, 0.0), (100.0, 0.0)), 'modelPath')
  x, y, vx, vy, variance, valid = 20.0, 3.0, 0.0, -1.0, .01, True
  if limitation == 'quality':
    y, vy, variance = 1.6, -.5, 1.0
  elif limitation == 'spatial':
    y, vx, vy = 0.0, -10.0, 0.0
  elif limitation == 'no_lateral_velocity':
    y, vx, vy, valid = 0.0, -10.0, 0.0, False
  path = _qualified_path((corridor,), x, y, vx, vy, .04, variance, valid)
  assert path.state in ('inPath', 'entering')
  if limitation == 'none':
    assert path.end == path.supported_end == pytest.approx(4.5)
    assert path.physical_valid and not path.prediction_limited
  else:
    assert path.end == HORIZON and path.prediction_limited and not path.physical_valid
    assert path.supported_end == pytest.approx({'quality': 1.8, 'spatial': 2.0, 'no_lateral_velocity': 0.0}[limitation])


def test_known_finite_support_matches_float32_envelope_after_wire_round_trip():
  result = HondaRadarControl().update(**scene(tracks=[radar(speed=0.0, lateral=3.0, lateral_speed=-1.3)]))
  assert len(result.objects) == 1
  lead = result.objects[0]
  assert 0.0 < lead['evidence']['conflictSupportedEnd'] < HORIZON
  state = log.RadarState.new_message(radarObjects=[lead], leadOne=lead)
  with log.RadarState.from_bytes(state.to_bytes()) as reader:
    full, display = reader.radarObjects[0], reader.leadOne
    assert full.evidence.conflictSupportVersion == 1
    assert full.evidence.conflictSupportedEnd == full.evidence.conflictEnd
    assert full.evidence.conflictSupportedEnd == display.evidence.conflictSupportedEnd
    assert qualified_lead(full, 'brake', 0.0, now_ns=START + 2_000_000)


@pytest.mark.parametrize('gap', ['missing', 'position_coast', 'path_history'])
def test_exhausted_support_keeps_obstacle_and_risk_without_renewing_or_clearing(gap):
  controller = HondaRadarControl()
  initial = radar(distance=12.0, speed=0.0, lateralPositionVariance=3.61, lateralVelocityVariance=1.0)
  first = controller.update(**scene(tracks=[initial]))
  original = evidence(first)
  assert original['conflictSupportVersion'] == 1
  assert original['conflictSupportedEnd'] == pytest.approx(.1)
  assert controller._tracks[1].relevant
  saved = deepcopy(original)
  for frame in (1, 2, 3):
    if gap == 'missing':
      sample = scene(frame, tracks=[])
    elif gap == 'position_coast':
      sample = scene(frame, tracks=[{**initial, 'coasting': True}])
    else:
      sample = scene(frame, tracks=[radar(frame, distance=12.0, speed=0.0)], short_path=True)
    result = controller.update(**sample)
    claim = evidence(result)
    assert claim['conflictSupportedEnd'] <= max(0.0, .1 - frame * STEP / 1e9) + 1e-12
    assert claim['conflictEnd'] == HORIZON and claim['brakeQualified'] and claim['coasting']
    assert not claim['stopQualified'] and not claim['departureQualified'] and not claim['physicalConflictValid']
    assert not result.cleared
    assert controller._tracks[1].relevant
    assert qualified_lead(result.objects[0], 'brake', 0.0, now_ns=sample['now_ns'])
  assert claim['conflictSupportedEnd'] == 0.0 and original == saved
  # A support deadline is not an OUT measurement. A later loss of the actual
  # measurements still leaves the independently retained near-obstacle risk.
  expired = controller.update(**scene(10, tracks=[]))
  assert not expired.objects and not expired.cleared
  assert expired.risk_unresolved and ('control-fixture', 1) in controller._lost_tracks


def test_new_measured_occupancy_can_renew_support_after_a_coast_cannot():
  controller = HondaRadarControl()
  controller.update(**scene(tracks=[radar(distance=12.0, speed=0.0,
                                         lateralPositionVariance=3.61, lateralVelocityVariance=1.0)]))
  coast = controller.update(**scene(2, tracks=[]))
  assert evidence(coast)['conflictSupportedEnd'] == 0.0
  renewed = controller.update(**scene(3, tracks=[radar(3, distance=12.0, speed=0.0)]))
  claim = evidence(renewed)
  assert claim['positionMonoTime'] == claim['velocityMonoTime'] == START + 3 * STEP
  assert claim['conflictSupportedEnd'] == claim['conflictEnd'] == HORIZON
  assert claim['followQualified'] and claim['stopQualified'] and not claim['coasting']
  assert not claim['departureQualified'] and not renewed.cleared
