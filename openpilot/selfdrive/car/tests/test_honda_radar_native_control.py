"""Consumer boundaries for native-only radar motion and independent vision."""
import copy
import math

import pytest

from openpilot.selfdrive.car.honda_radar_control import HondaRadarControl, _project
from openpilot.selfdrive.car.tests.test_honda_radar_control import (
  START, STEP, binding, evidence, native_motion_quality, present_leads, radar, scene, vision,
)
from openpilot.selfdrive.controls.lib.lead_qualification import qualified_lead


@pytest.mark.parametrize('component', ['position', 'velocity'])
@pytest.mark.parametrize('age_ns,admitted', [(199_999_000, True), (200_000_000, True), (200_001_000, False)])
@pytest.mark.parametrize('ego', [10.0, -.2])
def test_current_publication_does_not_renew_each_original_measurement_deadline(component, age_ns, admitted, ego):
  # This isolates consumer TTL; producer coasting may revoke earlier. The
  # source message/model are fresh even though one original measurement ages.
  now_ns = START + age_ns
  p_ns = START if component == 'position' else START + 1_000_000
  v_ns = START if component == 'velocity' else START + 1_000_000
  target = radar(speed=2.0, positionMonoTime=p_ns, velocityMonoTime=v_ns)
  values = scene(tracks=[target], ego=ego)
  model_ns = now_ns - 2_000_000
  values.update(now_ns=now_ns, model_ns=model_ns, observation_ns=now_ns - 1_000_000)
  values['model']['timestampEof'] = model_ns
  values['observation']['fusion'].update(modelMonoTime=model_ns, modelTimestampEof=model_ns, egoMonoTime=model_ns)
  result = HondaRadarControl().update(**values)
  assert result.healthy and not result.degraded and not result.cleared
  assert bool(present_leads(result)) is admitted
  if admitted:
    claim = evidence(result)
    assert claim['positionMonoTime'] == p_ns and claim['velocityMonoTime'] == v_ns
    assert qualified_lead(result.leads[0], 'brake', now_ns=now_ns)
    assert not qualified_lead(result.leads[0], 'brake', now_ns=START + 200_001_000)


@pytest.mark.parametrize('fault', ['coasting', 'withdrawn', 'bad_aux', 'missing_native', 'generation_restart'])
@pytest.mark.parametrize('has_visual', [False, True])
def test_current_quality_revocation_keeps_no_old_stop_or_fcw_obligation(fault, has_visual):
  controller = HondaRadarControl()
  initial = scene(tracks=[radar(distance=10.0, speed=0.0)], ego=5.0)
  first = controller.update(**initial)
  assert qualified_lead(first.leads[0], 'stop') and qualified_lead(first.leads[0], 'fcw')
  target = radar(1, distance=9.7, speed=0.0)
  if fault == 'coasting':
    target['coasting'] = True
  elif fault == 'withdrawn':
    target.update(active=False, lastInvalidReason='expired')
  elif fault == 'missing_native':
    target['nativeMotion'] = None
  else:
    if fault == 'generation_restart':
      target['generation'] += 1
    target['nativeMotion'] = native_motion_quality(target, qualified=False, sampleCount=0,
                                                   reason='identityChanged' if fault == 'generation_restart' else 'invalidVelocity')
  current = scene(1, tracks=[target], ego=5.0,
                  leads=[vision(distance=26.0, speed=7.0, ego=5.0)] if has_visual else None,
                  bindings=[binding(bias_x=-16.3)] if has_visual else None)
  result = controller.update(**current)
  assert result.healthy and not result.degraded and not result.cleared
  assert len(present_leads(result)) == int(has_visual)
  assert not controller._tracks
  if has_visual:
    lead = result.leads[0]
    assert not lead['radar'] and lead['dRel'] == 26.0 and lead['vLeadK'] == 7.0
    assert evidence(result)['source'] == 'vision' and evidence(result)['trackId'] != 1


def test_generation_restart_does_not_inherit_the_previous_qualified_segment():
  controller = HondaRadarControl()
  assert present_leads(controller.update(**scene()))
  for frame in range(1, 7):
    target = radar(frame, generation=2, observations=frame)
    result = controller.update(**scene(frame, tracks=[target]))
    assert result.healthy and not result.degraded
    assert bool(present_leads(result)) is (frame >= 5)
    if frame >= 5:
      assert evidence(result)['generation'] == 2
      assert target['nativeMotion']['firstPositionMonoTime'] == START + STEP


@pytest.mark.parametrize('lateral_speed', [-.5, 0.0, .5])
@pytest.mark.parametrize('yaw', [-.2, .2])
def test_camera_alignment_cannot_rotate_distance_derived_lateral_speed_into_native_longitudinal_motion(lateral_speed, yaw):
  target = radar(distance=20.0, speed=8.0, lateral_speed=lateral_speed, acceleration=-.7)
  values = scene(tracks=[target])
  elapsed = .1
  current = START + 100_000_000
  values.update(now_ns=current + 2_000_000, model_ns=current)
  values['model']['timestampEof'] = current
  values['model']['orientationRate']['z'] = [yaw]
  result = HondaRadarControl().update(**values)
  lead, = present_leads(result)
  expected_x, expected_y, rotated_vx, _ = _project(20.0, 0.0, 8.0, lateral_speed, elapsed, 10.0, yaw)
  assert lead['dRel'] == pytest.approx(expected_x) and lead['yRel'] == pytest.approx(expected_y)
  assert lead['vLeadK'] == 8.0 and lead['vRel'] == -2.0 and lead['aLeadK'] == -.7
  assert not math.isclose(rotated_vx, lead['vLeadK'], abs_tol=1e-8)
  assert evidence(result)['velocityMonoTime'] == START


@pytest.mark.parametrize('visual', [False, True])
def test_missing_native_acceleration_uses_existing_current_vision_or_constant_velocity_rule(visual):
  target = radar(speed=8.0, acceleration=-8.0, accelerationGood=False)
  values = scene(tracks=[target], leads=[vision(speed=7.0, acceleration=-1.25)] if visual else None,
                 bindings=[binding()] if visual else None)
  result = HondaRadarControl().update(**values)
  lead, = present_leads(result)
  assert lead['radar'] and lead['vLeadK'] == 8.0
  assert lead['aLeadK'] == (-1.25 if visual else 0.0)
  assert evidence(result)['accelerationValid'] is visual
  assert ('visionAcceleration' if visual else 'constantVelocity') in evidence(result)['reason']


def test_late_aux_may_complete_one_pair_but_cannot_add_confirmation_on_republication():
  controller = HondaRadarControl()
  target = radar(observations=4, velocityMonoTime=0, velocityGood=False)
  target['nativeMotion'] = native_motion_quality(target, sampleCount=4, reason='awaitingAux')
  values = scene(tracks=[target])
  assert not present_leads(controller.update(**values))
  complete = copy.deepcopy(target)
  complete.update(velocityGood=True, velocityMonoTime=START + 10_000_000)
  complete['nativeMotion'] = native_motion_quality(complete, sampleCount=4, reason='insufficientSupport')
  # A complete fourth pair remains four pairs, even if a later message has
  # both the source measurement and a fresh publication timestamp.
  for frame in (1, 2):
    current = scene(frame, tracks=[complete])
    result = controller.update(**current)
    assert result.healthy and not result.degraded and not present_leads(result)


def test_real_late_aux_can_complete_the_first_qualified_pair_without_another_position():
  controller = HondaRadarControl()
  pending = radar(1, velocityGood=False, velocityMonoTime=0)
  pending['nativeMotion'] = native_motion_quality(pending, qualified=False, sampleCount=4, reason='awaitingAux')
  assert not present_leads(controller.update(**scene(1, tracks=[pending])))
  completed = radar(1, velocityMonoTime=START + STEP + 10_000_000)
  completed['nativeMotion'] = native_motion_quality(completed, sampleCount=5,
                                                  firstPositionMonoTime=START + STEP - 250_000_000)
  result = controller.update(**scene(2, tracks=[completed]))
  assert result.healthy and not result.degraded
  assert present_leads(result) and evidence(result)['positionMonoTime'] == pending['positionMonoTime']
  assert evidence(result)['velocityMonoTime'] == completed['velocityMonoTime']


@pytest.mark.parametrize('kind', ['absent', 'default_wire', 'wrong_algorithm'])
def test_legacy_producer_messages_cannot_authorize_native_radar_roles(kind):
  from openpilot.cereal import custom
  target = radar()
  if kind == 'absent':
    target.pop('nativeMotion')
  elif kind == 'default_wire':
    target['nativeMotion'] = {}
  values = scene(tracks=[target])
  if kind == 'wrong_algorithm':
    values['observation']['fusion']['algorithmVersion'] = 'bosch-a-3'
  wire = custom.HondaRadarObservations.new_message(**values['observation']).to_bytes()
  with custom.HondaRadarObservations.from_bytes(wire) as observation:
    result = HondaRadarControl().update(**{**values, 'observation': observation})
  assert not present_leads(result)
  if kind != 'wrong_algorithm':
    assert result.healthy and not result.degraded
