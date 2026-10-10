"""Retired occupancy support stays neutral while ordinary lead ages still apply."""
from copy import deepcopy

import pytest

from openpilot.cereal import log
from openpilot.selfdrive.car.honda_radar_control import HORIZON, HondaRadarControl
from openpilot.selfdrive.car.tests.test_honda_radar_control import START, STEP, evidence, present_leads, radar, scene
from openpilot.selfdrive.controls.lib.lead_qualification import qualified_lead


@pytest.mark.parametrize('gap', ['missing', 'position_coast', 'path_history'])
def test_ordinary_lead_coast_has_no_support_tail_or_renewed_measurement_lifetime(gap):
  controller = HondaRadarControl()
  initial = radar(distance=12.0, speed=0.0, lateralPositionVariance=3.61, lateralVelocityVariance=1.0)
  first = controller.update(**scene(tracks=[initial]))
  original = evidence(first)
  saved = deepcopy(original)
  assert original['conflictSupportVersion'] == 0 and original['conflictSupportedEnd'] == 0.0
  for frame in (1, 2, 3):
    if gap == 'missing':
      sample = scene(frame, tracks=[])
    elif gap == 'position_coast':
      sample = scene(frame, tracks=[{**initial, 'coasting': True}])
    else:
      sample = scene(frame, tracks=[radar(frame, distance=12.0, speed=0.0)], short_path=True)
    result = controller.update(**sample)
    claim = evidence(result)
    assert claim['conflictStart'] == 0.0 and claim['conflictEnd'] == HORIZON
    assert claim['conflictSupportVersion'] == 0 and claim['conflictSupportedEnd'] == 0.0
    assert claim['brakeQualified'] and claim['coasting']
    assert not any(claim[key] for key in ('followQualified', 'stopQualified', 'departureQualified', 'physicalConflictValid'))
    if gap != 'path_history':
      assert claim['positionMonoTime'] == claim['velocityMonoTime'] == START
    assert result.degraded and not result.cleared
    assert qualified_lead(present_leads(result)[0], 'brake', now_ns=sample['now_ns'])
  assert original == saved
  expired = controller.update(**scene(10, tracks=[]))
  assert expired.healthy and not expired.degraded and not present_leads(expired) and not expired.cleared
  assert not controller._tracks


def test_fresh_measurements_restore_follow_flags_without_creating_an_occupancy_window():
  controller = HondaRadarControl()
  controller.update(**scene(tracks=[radar(distance=12.0, speed=0.0)]))
  coast = controller.update(**scene(2, tracks=[]))
  assert evidence(coast)['coasting']
  renewed = controller.update(**scene(3, tracks=[radar(3, distance=12.0, speed=0.0)]))
  claim = evidence(renewed)
  assert claim['positionMonoTime'] == claim['velocityMonoTime'] == START + 3 * STEP
  assert claim['followQualified'] and claim['stopQualified'] and not claim['coasting']
  assert claim['conflictSupportVersion'] == 0 and claim['conflictSupportedEnd'] == 0.0
  assert not claim['departureQualified'] and not renewed.cleared


def test_ordinary_lead_wire_round_trip_keeps_retired_support_neutral():
  result = HondaRadarControl().update(**scene(tracks=[radar(distance=2.57, speed=.884)]))
  state = log.RadarState.new_message(radarObjects=result.objects, leadOne=result.leads[0], leadTwo=result.leads[1])
  with log.RadarState.from_bytes(state.to_bytes()) as reader:
    assert len(reader.radarObjects) == 0 and reader.leadOne.present
    claim = reader.leadOne.evidence
    assert claim.conflictStart == 0.0 and claim.conflictEnd == HORIZON
    assert claim.conflictSupportVersion == 0 and claim.conflictSupportedEnd == 0.0
    assert qualified_lead(reader.leadOne, 'brake', now_ns=START + 2_000_000)
