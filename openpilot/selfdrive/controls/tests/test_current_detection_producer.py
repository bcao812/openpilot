"""Current visual membership through RadarD and the real Honda adapter.

Sensor/model fixtures supply observations; target selection and publication
run production code. Settings permission is fixed to isolate this contract.
"""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from opendbc.car.hyundai.values import HyundaiFlags
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP
from openpilot.cereal import messaging
from openpilot.selfdrive.car.honda_radar_control import HondaRadarControl
from openpilot.selfdrive.car.honda_radar_observer_config import SERVICE
from openpilot.selfdrive.car.tests.test_honda_radar_control import radar, scene, vision
from openpilot.selfdrive.controls.radard import RadarD
from openpilot.selfdrive.controls.tests.test_radard_publication import Inputs, publish_at


def make_producer(*, fusion=False):
  producer = RadarD(SimpleNamespace(brand='honda', flags=0), SimpleNamespace(flags=0), params=SimpleNamespace(get=lambda _: None))
  producer.control_enabled = Mock(return_value=fusion)
  if fusion:
    producer.fusion_control = HondaRadarControl()
  return producer


def produce(producer, frame, probabilities, *, tracks=(), points=(), speed=10.0, model_valid=True):
  sensor = scene(frame, tracks=list(tracks), ego=speed,
                 leads=[vision(distance=20.0 + 15.0 * index, speed=speed, ego=speed, probability=probability)
                        for index, probability in enumerate(probabilities)])
  inputs = Inputs(sensor['model_ns'])
  model = messaging.new_message('modelV2')
  model.modelV2 = sensor['model']
  inputs['modelV2'] = model.modelV2
  inputs.valid['modelV2'] = model_valid
  observation = messaging.new_message(SERVICE)
  setattr(observation, SERVICE, sensor['observation'])
  inputs[SERVICE] = getattr(observation, SERVICE)
  inputs.logMonoTime[SERVICE] = sensor['observation_ns']
  inputs['carState'].vEgo = speed
  inputs['radarTracks'].points = list(points)
  producer.update(inputs, inputs['radarTracks'], now_ns=sensor['now_ns'])
  return publish_at(producer, sensor['now_ns'] + 1)


@pytest.mark.parametrize('index', [0, 1])
@pytest.mark.parametrize('probability', [0.0, .49, .5, -0.1, 1.01, float('nan'), float('inf')])
def test_visual_removal_uses_first_current_unqualified_frame_and_recovers(index, probability):
  producer = make_producer()
  initial = produce(producer, 0, (1.0, 1.0))
  assert initial.radarState.leadOne.present and initial.radarState.leadTwo.present

  current = [1.0, 1.0]
  current[index] = probability
  withdrawn = produce(producer, 1, current)
  names = ('leadOne', 'leadTwo')
  visual_names = ('visionLeadOne', 'visionLeadTwo')
  assert withdrawn.valid
  assert not getattr(withdrawn.radarState, names[index]).present
  assert not getattr(withdrawn.radarState, visual_names[index]).present
  assert getattr(withdrawn.radarState, names[1 - index]).present
  assert not withdrawn.radarState.fusionDegraded

  current[index] = .6
  recovered = produce(producer, 2, current)
  lead = getattr(recovered.radarState, names[index])
  assert recovered.valid and lead.present
  assert lead.modelProb == pytest.approx(.6)


def test_qualified_visual_probability_does_not_keep_old_stronger_authority():
  producer = make_producer(fusion=True)
  initial = produce(producer, 0, (1.0, 0.0))
  assert initial.radarState.leadOne.evidence.stopQualified
  assert initial.radarState.leadOne.evidence.fcwQualified

  current = produce(producer, 1, (.6, 0.0))
  lead = current.radarState.leadOne
  assert current.valid and current.radarState.fusionHealthy
  assert lead.present and lead.modelProb == pytest.approx(.6)
  assert lead.evidence.source == 'vision'
  assert lead.evidence.followQualified and lead.evidence.brakeQualified
  assert not lead.evidence.stopQualified and not lead.evidence.departureQualified
  assert not lead.evidence.fcwQualified


@pytest.mark.parametrize('probability', [0.0, .5, float('nan'), float('inf'), 1.01])
def test_adapter_rejects_old_visual_confidence_when_current_raw_hypothesis_is_unqualified(probability):
  controller = HondaRadarControl()
  initial = scene(tracks=[], leads=[vision(probability=.99)])
  assert controller.update(**initial).leads[0]['present']

  current = scene(1, tracks=[], leads=[vision(probability=.99)])
  current['model']['leadsV3'][0]['prob'] = probability
  result = controller.update(**current)
  assert result.healthy and not result.degraded
  assert not any(lead.get('present') for lead in result.leads)
  assert not result.cleared


def test_adapter_uses_current_probability_for_each_visual_purpose():
  current = scene(tracks=[], leads=[vision(probability=.99)])
  current['model']['leadsV3'][0]['prob'] = .6
  result = HondaRadarControl().update(**current)
  lead = result.leads[0]
  assert lead['present'] and lead['modelProb'] == .6
  assert lead['evidence']['followQualified'] and lead['evidence']['brakeQualified']
  assert not lead['evidence']['stopQualified']
  assert not lead['evidence']['departureQualified'] and not lead['evidence']['fcwQualified']


def test_radar_and_vision_remain_independent_current_sources_through_producer():
  producer = make_producer(fusion=True)
  initial = produce(producer, 0, (.99, 0.0), tracks=[radar(0)])
  assert initial.valid and initial.radarState.leadOne.present

  # No fresh raw visual detection: a current qualified radar track still works.
  # The observer's event list is empty in this fixture; it is not a complete
  # scan's negative detection and must not revoke its current fusion track.
  radar_only = produce(producer, 1, (0.0, 0.0), tracks=[radar(1)])
  lead = radar_only.radarState.leadOne
  assert radar_only.valid and radar_only.radarState.fusionHealthy
  assert not radar_only.radarState.fusionDegraded
  assert lead.present and lead.radar and lead.evidence.source == 'radar'
  assert lead.evidence.positionMonoTime == radar(1)['positionMonoTime']
  assert not radar_only.radarState.visionLeadOne.present

  absent = produce(producer, 2, (0.0, 0.0))
  assert absent.valid and absent.radarState.fusionHealthy and not absent.radarState.fusionDegraded
  assert not absent.radarState.leadOne.present and not absent.radarState.leadTwo.present

  visual_only = produce(producer, 3, (.94, 0.0))
  assert visual_only.valid and visual_only.radarState.fusionHealthy
  assert visual_only.radarState.leadOne.present and not visual_only.radarState.leadOne.radar
  assert visual_only.radarState.leadOne.evidence.source == 'vision'

  recovered = produce(producer, 4, (0.0, 0.0), tracks=[radar(4)])
  assert recovered.valid and recovered.radarState.leadOne.present
  assert recovered.radarState.leadOne.evidence.source == 'radar'
  assert recovered.radarState.leadOne.evidence.positionMonoTime == radar(4)['positionMonoTime']


def test_low_speed_current_radar_override_does_not_require_current_vision():
  producer = make_producer()
  point = {'trackId': 42, 'dRel': 12.0, 'yRel': .2, 'vRel': -3.0}
  current = produce(producer, 0, (0.0, 0.0), points=[point], speed=3.0)
  assert current.valid and current.radarState.leadOne.present
  assert current.radarState.leadOne.radar and current.radarState.leadOne.radarTrackId == 42
  assert current.radarState.leadOne.modelProb == 0.0
  absent = produce(producer, 1, (0.0, 0.0), speed=3.0)
  assert absent.valid and not absent.radarState.leadOne.present


def test_invalid_model_health_is_not_converted_to_healthy_empty_output():
  producer = make_producer()
  initial = produce(producer, 0, (.99, 0.0))
  assert initial.valid
  invalid = produce(producer, 1, (0.0, 0.0), model_valid=False)
  assert not invalid.valid


def test_generic_radar_matching_needs_current_visual_membership_above_low_speed_override():
  producer = make_producer()
  points = [{'trackId': 42, 'dRel': 20.0, 'yRel': .4, 'vRel': 0.0},
            {'trackId': 43, 'dRel': 80.0, 'yRel': 0.0, 'vRel': 0.0}]
  initial = produce(producer, 0, (.99, 0.0), points=points, speed=10.0)
  lead = initial.radarState.leadOne
  assert initial.valid and lead.present and lead.radar and lead.radarTrackId == 42
  assert lead.yRel == pytest.approx(.4)

  # Both radar points remain current. At this speed the generic vision-match
  # path requires a current hypothesis; the old high confidence cannot match.
  absent = produce(producer, 1, (0.0, 0.0), points=points, speed=10.0)
  assert absent.valid and not absent.radarState.leadOne.present
  assert not absent.radarState.leadTwo.present
  assert set(producer.tracks) == {42, 43}

  recovered = produce(producer, 2, (.6, 0.0), points=points, speed=10.0)
  lead = recovered.radarState.leadOne
  assert recovered.valid and lead.present and lead.radar and lead.radarTrackId == 42
  assert lead.modelProb == pytest.approx(.6)
  assert lead.yRel == pytest.approx(.4)


@pytest.mark.parametrize('flags,flags_sp,expected_lateral', [
  (0, 0, .4),
  (0, HyundaiFlagsSP.ENHANCED_SCC, 0.0),
  (HyundaiFlags.CANFD_CAMERA_SCC, 0, 0.0),
  (HyundaiFlags.CAMERA_SCC, 0, 0.0),
], ids=['radar-coordinates', 'enhanced-scc', 'canfd-camera-scc', 'camera-scc'])
def test_hyundai_lateral_coordinate_choice_does_not_retain_old_visual_qualification(flags, flags_sp, expected_lateral):
  producer = make_producer()
  producer.CP = SimpleNamespace(brand='hyundai', flags=flags)
  producer.CP_SP = SimpleNamespace(flags=flags_sp)
  point = {'trackId': 42, 'dRel': 20.0, 'yRel': .4, 'vRel': 0.0}
  initial = produce(producer, 0, (.99, 0.0), points=[point], speed=10.0)
  assert initial.valid and initial.radarState.leadOne.present
  assert initial.radarState.leadOne.yRel == pytest.approx(expected_lateral)

  weak = produce(producer, 1, (.6, 0.0), points=[point], speed=10.0)
  lead = weak.radarState.leadOne
  assert weak.valid and lead.present and lead.radar and lead.radarTrackId == 42
  assert lead.modelProb == pytest.approx(.6)
  assert lead.yRel == pytest.approx(expected_lateral)

  absent = produce(producer, 2, (.5, 0.0), points=[point], speed=10.0)
  assert absent.valid and not absent.radarState.leadOne.present
  assert not absent.radarState.visionLeadOne.present
  recovered = produce(producer, 3, (.99, 0.0), points=[point], speed=10.0)
  assert recovered.valid and recovered.radarState.leadOne.radarTrackId == 42
  assert recovered.radarState.leadOne.yRel == pytest.approx(expected_lateral)
