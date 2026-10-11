"""Retired physical-conflict fields never acquire current control authority."""
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.car.honda_radar_control import HondaRadarControl
from openpilot.selfdrive.car.tests.test_honda_radar_control import binding, evidence, present_leads, radar, scene, vision


@pytest.mark.parametrize('kind', ['radar', 'fused', 'vision', 'radar_loss', 'visual_fallback'])
def test_ordinary_leads_never_publish_physical_conflict_predictions(kind):
  controller = HondaRadarControl()
  if kind == 'vision':
    result = controller.update(**scene(tracks=[], leads=[vision()]))
  elif kind == 'fused':
    result = controller.update(**scene(tracks=[radar(lateral=1.6)], leads=[vision()], bindings=[binding(bias_y=1.6)]))
  elif kind in ('radar_loss', 'visual_fallback'):
    controller.update(**scene(leads=[vision()], bindings=[binding()]))
    sample = scene(1, tracks=[], leads=[vision()] if kind == 'visual_fallback' else None)
    sample['observation_valid'] = False
    result = controller.update(**sample)
  else:
    result = controller.update(**scene())
  if kind == 'radar_loss':
    # Optional observer loss exposes its fault but produces no predicted lead
    # or clearance. A current same-car visual fallback is a separate case.
    assert result.degraded and not present_leads(result) and not result.cleared
    assert not controller._tracks
    return
  claim = evidence(result)
  assert claim['physicalConflictValid'] is False
  assert claim['physicalConflictStart'] == claim['physicalConflictEnd'] == 0.0
  assert not claim['cautionQualified']
  assert not claim['coasting']


@pytest.mark.parametrize('lateral,lateral_speed', [(3.0, -2.0), (0.0, 5.0), (0.0, 11.0), (1.6, -.2)])
def test_future_entry_crossing_and_outer_band_have_no_additional_object_control(lateral, lateral_speed):
  controller = HondaRadarControl()
  result = controller.update(**scene(tracks=[radar(distance=6.0, speed=0.0, lateral=lateral,
                                                 lateral_speed=lateral_speed)], ego=15.0))
  assert not present_leads(result) and not controller._tracks


def test_legacy_physical_conflict_fields_remain_readable_without_being_produced():
  state = log.RadarState.new_message(leadOne={'present': True, 'evidence': {
    'physicalConflictValid': True, 'physicalConflictStart': .1234567890123, 'physicalConflictEnd': .9876543210987,
  }})
  with log.RadarState.from_bytes(state.to_bytes()) as reader:
    assert reader.leadOne.evidence.physicalConflictValid
    assert reader.leadOne.evidence.physicalConflictStart == .1234567890123
  result = HondaRadarControl().update(**scene())
  assert not evidence(result)['physicalConflictValid']
