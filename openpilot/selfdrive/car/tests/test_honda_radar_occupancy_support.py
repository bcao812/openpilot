"""Retired occupancy support stays neutral while ordinary lead ages still apply."""
from copy import deepcopy

import pytest

from openpilot.cereal import log
from openpilot.selfdrive.car.honda_radar_control import HORIZON, HondaRadarControl
from openpilot.selfdrive.car.tests.test_honda_radar_control import START, STEP, evidence, present_leads, radar, scene
from openpilot.selfdrive.controls.lib.lead_qualification import qualified_lead


@pytest.mark.parametrize('gap', ['missing', 'position_coast', 'path_history'])
def test_missing_measurements_withdraw_the_formal_lead_without_a_support_tail(gap):
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
    # The raw fusion may diagnose coasting, but formal control never republishes
    # the missing position/path as a new braking object or as measured clearance.
    assert not present_leads(result) and not result.cleared and not controller._tracks
    assert all(not qualified_lead(lead, purpose, now_ns=sample['now_ns'])
               for lead in result.leads for purpose in ('follow', 'brake', 'stop', 'departure', 'fcw'))
    assert result.healthy
    # Losing the current path of an established relevant lead still reports
    # that current failure; an ordinary position gap does not fabricate one.
    assert result.degraded is (gap == 'path_history' and frame == 1)
  assert original == saved
  expired = controller.update(**scene(10, tracks=[]))
  assert expired.healthy and not expired.degraded and not present_leads(expired) and not expired.cleared
  assert not controller._tracks


def test_fresh_measurements_restore_follow_flags_after_explicit_withdrawal():
  controller = HondaRadarControl()
  controller.update(**scene(tracks=[radar(distance=12.0, speed=0.0)]))
  withdrawn = controller.update(**scene(2, tracks=[]))
  assert not present_leads(withdrawn) and not withdrawn.cleared and not controller._tracks
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
