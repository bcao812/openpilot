"""Current camera qualification and revocation of a confirmed physical identity."""
import math

import pytest

from openpilot.selfdrive.car.honda_radar_control import HondaRadarControl
from openpilot.selfdrive.car.tests.test_honda_radar_control import START, STEP, binding, radar, scene, vision


def observer_gap(frame, *, leads=None):
  values = scene(frame, leads=leads or [vision()])
  values['observation_valid'] = False
  return values


@pytest.mark.parametrize('probability, fused, visual_present', [
  (.799, False, True), (.8, True, True),
  (math.nan, False, False), (math.inf, False, False), (-.1, False, False), (1.01, False, False),
], ids=['below-threshold', 'at-threshold', 'nan', 'infinity', 'negative', 'above-one'])
def test_current_compatible_visual_probability_also_qualifies_identity_binding(probability, fused, visual_present):
  controller = HondaRadarControl()
  result = controller.update(**scene(leads=[vision(probability=probability)], bindings=[binding()]))
  target = next(lead for lead in result.objects if lead['evidence']['trackId'] == 1)
  assert target['evidence']['source'] == ('fused' if fused else 'radar')
  assert target['modelProb'] == (probability if fused else 0.0)
  assert target['dRel'] == 20.0 and target['vLeadK'] == 10.0 and target['aLeadK'] == -.4
  assert target['evidence']['positionMonoTime'] == START
  assert target['evidence']['velocityMonoTime'] == START
  assert bool(controller._tracks[1].last_vision) is fused
  assert len(result.objects) == 1 + int(visual_present and not fused)


@pytest.mark.parametrize('probability, fused', [
  (.799, False), (.8, True), (math.nan, False), (math.inf, False), (-.1, False), (1.01, False), (None, False),
], ids=['below-threshold', 'at-threshold', 'nan', 'infinity', 'negative', 'above-one', 'missing'])
def test_current_raw_probability_gates_identity_separately_from_filtered_visual_lead(probability, fused):
  values = scene(leads=[vision(probability=.94)], bindings=[binding()])
  values['model']['leadsV3'][0]['prob'] = probability
  result = HondaRadarControl().update(**values)
  target = next(lead for lead in result.objects if lead['evidence']['trackId'] == 1)
  assert target['evidence']['source'] == ('fused' if fused else 'radar')
  assert target['dRel'] == 20.0 and target['vLeadK'] == 10.0 and target['aLeadK'] == -.4
  if not fused:
    # The ordinary visual object keeps its existing filtered probability and
    # purpose flags; the extra gate grants no new component/identity authority.
    visual = next(lead for lead in result.objects if lead['evidence']['source'] == 'vision')
    assert visual['modelProb'] == .94 and visual['dRel'] == 20.0
    assert visual['evidence']['brakeQualified'] and visual['evidence']['stopQualified']
    assert visual['evidence']['positionMonoTime'] == START
    assert visual['evidence']['velocityMonoTime'] == START
  assert len(result.objects) == (1 if fused else 2)


def test_older_observer_binding_cannot_qualify_new_raw_weak_visual_components():
  previous = scene(leads=[vision()], bindings=[binding()])
  values = scene(1, leads=[vision(probability=.94)])
  values['observation'] = previous['observation']
  values['observation_ns'] = previous['observation_ns']
  values['model']['leadsV3'][0]['prob'] = .2
  result = HondaRadarControl().update(**values)
  assert result.healthy and len(result.objects) == 2
  assert all(lead['evidence']['source'] != 'fused' for lead in result.objects)
  target = next(lead for lead in result.objects if lead['evidence']['trackId'] == 1)
  assert target['evidence']['positionMonoTime'] == START
  assert target['evidence']['velocityMonoTime'] == START
  assert target['evidence']['estimateMonoTime'] == START + STEP


@pytest.mark.parametrize('component', ['velocity', 'acceleration'])
def test_filtered_probability_cannot_authorize_borrowing_current_raw_weak_motion(component):
  track = radar(velocityGood=False, velocityVariance=5.0, observations=3) if component == 'velocity' else radar(accelerationGood=False)
  values = scene(tracks=[track], leads=[vision(probability=.94, speed=12.0, acceleration=1.2)], bindings=[binding()])
  values['model']['leadsV3'][0]['prob'] = .2
  result = HondaRadarControl().update(**values)
  assert all(lead['evidence']['source'] != 'fused' for lead in result.objects)
  if component == 'velocity':
    assert len(result.objects) == 1 and result.objects[0]['evidence']['source'] == 'vision'
  else:
    target = next(lead for lead in result.objects if lead['evidence']['trackId'] == 1)
    assert target['vLeadK'] == 10.0 and target['aLeadK'] == 0.0
    assert not target['evidence']['accelerationValid']


@pytest.mark.parametrize('distance, speed, continuous', [(20.0, 10.0, True), (7.0, 3.0, False)])
def test_raw_confidence_recovery_needs_current_same_slot_motion_continuity(distance, speed, continuous):
  controller = HondaRadarControl()
  controller.update(**scene(leads=[vision()], bindings=[binding()]))
  weak = observer_gap(1)
  weak['model']['leadsV3'][0]['prob'] = .2
  result = controller.update(**weak)
  assert all(lead['evidence']['reason'] != 'visionFallback' for lead in result.objects)
  assert controller._tracks[1].last_vision['evidence']['estimateMonoTime'] == START
  assert any(lead['evidence']['source'] == 'vision' for lead in result.objects)
  recovered = controller.update(**observer_gap(2, leads=[vision(distance=distance, speed=speed)]))
  fallbacks = [lead for lead in recovered.objects if lead['evidence']['reason'] == 'visionFallback']
  assert bool(fallbacks) is continuous
  if continuous:
    assert len(recovered.objects) == 1
    assert fallbacks[0]['evidence']['positionMonoTime'] == START + 2 * STEP
    assert fallbacks[0]['evidence']['velocityMonoTime'] == START + 2 * STEP
  else:
    assert any(lead['evidence']['source'] == 'vision' and lead['dRel'] == 7.0 for lead in recovered.objects)
    assert any(lead['evidence']['trackId'] == 1 and lead['evidence']['brakeQualified'] for lead in recovered.objects)


def test_raw_confidence_recovery_cannot_recreate_an_explicitly_revoked_identity():
  controller = HondaRadarControl()
  controller.update(**scene(leads=[vision()], bindings=[binding()]))
  controller.update(**scene(1, leads=[vision()], bindings=[binding(
    identityState='different', identityReason='residualSeparation', rejectionMonoTime=START + STEP)]))
  weak = scene(2, leads=[vision()], bindings=[binding(bindingStartMonoTime=START + STEP)])
  weak['model']['leadsV3'][0]['prob'] = .2
  controller.update(**weak)
  assert controller._tracks[1].last_vision is None
  recovered = controller.update(**observer_gap(3))
  assert all(lead['evidence']['reason'] != 'visionFallback' for lead in recovered.objects)
  rebound = controller.update(**scene(4, leads=[vision()], bindings=[binding(bindingStartMonoTime=START + 2 * STEP)]))
  assert len(rebound.objects) == 1 and rebound.objects[0]['evidence']['source'] == 'fused'


def test_weak_current_vision_cannot_supply_unavailable_radar_velocity():
  track = radar(velocityGood=False, velocityVariance=5.0, observations=3)
  result = HondaRadarControl().update(**scene(tracks=[track], leads=[vision(probability=.799)], bindings=[binding()]))
  assert len(result.objects) == 1
  assert result.objects[0]['evidence']['source'] == 'vision'
  assert result.objects[0]['evidence']['trackId'] == 0
  assert result.objects[0]['vLeadK'] == 10.0


@pytest.mark.parametrize('state, reason', [
  ('different', 'residualSeparation'), ('unresolved', 'residualSeparation'),
  ('unresolved', 'visionJump'), ('different', 'referenceInvalid:sourceGap'),
])
@pytest.mark.parametrize('active_radar', [True, False], ids=['active-radar', 'radar-missing'])
def test_explicit_identity_rejection_survives_later_observer_dropout(state, reason, active_radar):
  controller = HondaRadarControl()
  controller.update(**scene(leads=[vision()], bindings=[binding()]))
  rejected = binding(identityState=state, identityReason=reason, rejectionMonoTime=START + STEP)
  controller.update(**scene(1, tracks=[radar(1)] if active_radar else [], leads=[vision()], bindings=[rejected]))
  assert controller._tracks[1].last_vision is None
  result = controller.update(**observer_gap(2))
  assert all(lead['evidence']['reason'] != 'visionFallback' for lead in result.objects)
  radar_object = next(lead for lead in result.objects if lead['evidence']['trackId'] == 1)
  assert radar_object['evidence']['source'] == 'predicted'
  assert radar_object['evidence']['brakeQualified'] and radar_object['evidence']['coasting']
  assert radar_object['evidence']['positionMonoTime'] == START + (STEP if active_radar else 0)
  assert radar_object['evidence']['velocityMonoTime'] == START + (STEP if active_radar else 0)
  assert any(lead['evidence']['source'] == 'vision' for lead in result.objects)


def test_current_confirmed_rebinding_restores_visual_continuity_after_explicit_rejection():
  controller = HondaRadarControl()
  controller.update(**scene(leads=[vision()], bindings=[binding()]))
  controller.update(**scene(1, leads=[vision()], bindings=[binding(
    identityState='different', identityReason='residualSeparation', rejectionMonoTime=START + STEP)]))
  rejected_gap = controller.update(**observer_gap(2))
  assert all(lead['evidence']['reason'] != 'visionFallback' for lead in rejected_gap.objects)
  rebound = controller.update(**scene(3, leads=[vision()], bindings=[binding(bindingStartMonoTime=START + STEP)]))
  assert len(rebound.objects) == 1 and rebound.objects[0]['evidence']['source'] == 'fused'
  recovered = controller.update(**observer_gap(4))
  assert len(recovered.objects) == 1 and not recovered.degraded
  assert recovered.objects[0]['evidence']['reason'] == 'visionFallback'
  assert recovered.objects[0]['evidence']['trackId'] == 1
  assert recovered.objects[0]['evidence']['positionMonoTime'] == START + 4 * STEP
  assert recovered.objects[0]['evidence']['velocityMonoTime'] == START + 4 * STEP


def test_explicit_rejection_also_revokes_retained_loss_memory_before_radar_returns():
  controller = HondaRadarControl()
  controller.update(**scene(leads=[vision()], bindings=[binding()]))
  controller.update(**scene(1, tracks=[radar(1, positionSeq=0)]))
  assert 1 not in controller._tracks and ('control-fixture', 1) in controller._lost_tracks
  controller.update(**scene(2, leads=[vision()], bindings=[binding(
    identityState='different', identityReason='residualSeparation', rejectionMonoTime=START + 2 * STEP)]))
  assert controller._tracks[1].last_vision is None
  result = controller.update(**observer_gap(3))
  assert all(lead['evidence']['reason'] != 'visionFallback' for lead in result.objects)
  assert any(lead['evidence']['trackId'] == 1 and lead['evidence']['brakeQualified'] for lead in result.objects)


@pytest.mark.parametrize('reason', ['sourceGap', 'visionUnqualified', 'noMatch', 'ambiguous'])
@pytest.mark.parametrize('radar_distance, visual_distance', [(17.15, 22.71), (17.30, 22.45)])
def test_current_soft_rejection_blocks_farther_visual_replacement_until_reconfirmation(reason, radar_distance, visual_distance):
  controller = HondaRadarControl()
  bias = radar_distance - visual_distance
  controller.update(**scene(tracks=[radar(distance=radar_distance)], leads=[vision(distance=visual_distance)],
                            bindings=[binding(bias_x=bias)]))
  suspension = binding(bias_x=bias, identityState='unresolved', identityReason=reason, rejectionMonoTime=START + STEP)
  result = controller.update(**scene(1, tracks=[], leads=[vision(distance=visual_distance)], bindings=[suspension]))
  # These farther visual replacements were observed in the recorded window.
  # Keep the nearer radar coast until this association is currently eligible.
  assert len(result.objects) == 2
  assert all(lead['evidence']['reason'] != 'visionFallback' for lead in result.objects)
  assert result.leads[0]['dRel'] == pytest.approx(radar_distance)
  retained = next(lead for lead in result.objects if lead['evidence']['trackId'] == 1)
  assert retained['evidence']['source'] == 'predicted' and retained['evidence']['coasting']
  assert retained['evidence']['brakeQualified']
  assert not any(retained['evidence'][purpose] for purpose in ('followQualified', 'stopQualified', 'departureQualified'))
  assert retained['evidence']['positionMonoTime'] == START
  assert retained['evidence']['velocityMonoTime'] == START
  assert controller._tracks[1].last_vision['evidence']['estimateMonoTime'] == START
  assert any(lead['evidence']['source'] == 'vision' and lead['dRel'] == visual_distance for lead in result.objects)
  recovered = controller.update(**scene(4, tracks=[radar(4, distance=radar_distance)],
                                       leads=[vision(distance=visual_distance)],
                                       bindings=[binding(bias_x=bias, bindingStartMonoTime=START + 2 * STEP,
                                                         bindingSamples=3, identityReason='reconfirmed')]))
  assert len(recovered.objects) == 1
  assert recovered.objects[0]['dRel'] == pytest.approx(radar_distance)
  assert recovered.objects[0]['evidence']['source'] == 'fused'
  assert all(recovered.objects[0]['evidence'][purpose] for purpose in ('followQualified', 'stopQualified', 'departureQualified'))
  assert controller._tracks[1].last_vision['evidence']['estimateMonoTime'] == START + 4 * STEP


def test_soft_rejection_is_per_frame_when_optional_observer_goes_missing():
  controller = HondaRadarControl()
  controller.update(**scene(leads=[vision()], bindings=[binding()]))
  suspension = binding(identityState='unresolved', identityReason='sourceGap', rejectionMonoTime=START + STEP)
  rejected = controller.update(**scene(1, tracks=[], leads=[vision()], bindings=[suspension]))
  assert all(lead['evidence']['reason'] != 'visionFallback' for lead in rejected.objects)
  assert controller._tracks[1].last_vision is not None
  result = controller.update(**observer_gap(2))
  assert len(result.objects) == 1 and result.objects[0]['evidence']['reason'] == 'visionFallback'
  assert not result.degraded and not result.risk_unresolved


@pytest.mark.parametrize('changes', [{'index': 1}, {'identifier': 2}])
def test_rejection_of_another_pair_cannot_revoke_the_confirmed_visual_stream(changes):
  controller = HondaRadarControl()
  controller.update(**scene(leads=[vision()], bindings=[binding()]))
  unrelated = binding(**changes, identityState='different', identityReason='residualSeparation', rejectionMonoTime=START + STEP)
  controller.update(**scene(1, leads=[vision()], bindings=[unrelated]))
  result = controller.update(**observer_gap(2))
  assert len(result.objects) == 1 and result.objects[0]['evidence']['reason'] == 'visionFallback'
  assert not result.degraded


@pytest.mark.parametrize('changes', [
  {'probability': math.nan}, {'probability': .799}, {'probability': 1.1},
  {'rejectionMonoTime': START + 2 * STEP}, {'rejectionMonoTime': -1},
])
def test_unqualified_rejection_body_cannot_revoke_the_confirmed_visual_stream(changes):
  controller = HondaRadarControl()
  controller.update(**scene(leads=[vision()], bindings=[binding()]))
  rejected = binding(identityState='different', identityReason='residualSeparation', rejectionMonoTime=START + STEP)
  rejected.update(changes)
  controller.update(**scene(1, leads=[vision()], bindings=[rejected]))
  result = controller.update(**observer_gap(2))
  assert len(result.objects) == 1 and result.objects[0]['evidence']['reason'] == 'visionFallback'


def test_untrusted_observer_publication_cannot_revoke_the_confirmed_visual_stream():
  controller = HondaRadarControl()
  controller.update(**scene(leads=[vision()], bindings=[binding()]))
  values = scene(1, leads=[vision()], bindings=[binding(
    identityState='different', identityReason='residualSeparation', rejectionMonoTime=START + STEP)])
  values['observation_valid'] = False
  result = controller.update(**values)
  assert len(result.objects) == 1 and result.objects[0]['evidence']['reason'] == 'visionFallback'
  assert not result.degraded


def test_revoked_binding_preserves_independent_near_radar_risk():
  controller = HondaRadarControl()
  for frame in range(2):
    association = binding() if frame == 0 else binding(
      identityState='different', identityReason='residualSeparation', rejectionMonoTime=START + STEP)
    tracks = [radar(frame), radar(frame, identifier=2, oem=8, distance=7.0, speed=0.0)]
    controller.update(**scene(frame, tracks=tracks, leads=[vision()], bindings=[association]))
  result = controller.update(**observer_gap(2))
  assert all(lead['evidence']['reason'] != 'visionFallback' for lead in result.objects)
  near = next(lead for lead in result.objects if lead['evidence']['trackId'] == 2)
  assert near['dRel'] < 7.0 and near['vLeadK'] == 0.0
  assert near['evidence']['brakeQualified'] and near['evidence']['coasting']
  assert near['evidence']['positionMonoTime'] == START + STEP
  assert near['evidence']['velocityMonoTime'] == START + STEP
  assert any(lead['evidence']['source'] == 'vision' for lead in result.objects)
