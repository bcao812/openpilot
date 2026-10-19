"""Current narrow off-center reflections cannot supply ordinary point authority.

These are synthetic sensor contracts. They exercise admission and recovery,
not physical-object classification or a measured change in vehicle braking.
"""
import math

import pytest

from openpilot.selfdrive.car.honda_radar_control import CUT_IN_REASON, HondaRadarControl, _narrow_point_unconfirmed
from openpilot.selfdrive.car.tests.test_honda_radar_control import START, STEP, binding, present_leads, radar, scene, vision
from openpilot.selfdrive.controls.lib.lead_qualification import qualified_lead


ORDINARY_ROLES = ('follow', 'brake', 'stop', 'fcw')


def narrow_track(frame=0, *, width=.5, distance=4.0, lateral=1.2, speed=4.0, **changes):
  target = radar(frame, distance=distance, lateral=lateral, speed=speed)
  target.update(bodyValid=True, bodyMonoTime=target['positionMonoTime'],
                bodyAvailableMonoTime=target['positionMonoTime'], bodyWidth=width,
                bodyLeftX=distance, bodyRightX=distance,
                bodyLeftY=lateral + width / 2, bodyRightY=lateral - width / 2)
  target.update(changes)
  return target


def current_result(target, *, control=None, frame=0, **changes):
  return (control or HondaRadarControl()).update(**scene(frame, tracks=[target], **changes))


def assert_ordinary(lead):
  assert lead['radar'] and all(qualified_lead(lead, role) for role in ORDINARY_ROLES)


@pytest.mark.parametrize('side', [-1, 1])
@pytest.mark.parametrize('speed', [0.0, 4.0])
def test_current_narrow_near_offset_point_does_not_publish_ordinary_authority(side, speed):
  target = narrow_track(distance=2.21, lateral=side * 1.164, speed=speed)
  result = current_result(target)
  assert result.healthy and not result.degraded and not result.cleared
  assert not present_leads(result)


@pytest.mark.parametrize('side', [-1, 1])
@pytest.mark.parametrize(('width', 'distance', 'offset', 'restricted'), [
  (math.nextafter(.70, 0.0), 6.0, 1.2, True),
  (.70, 6.0, 1.2, False),
  (math.nextafter(.70, math.inf), 6.0, 1.2, False),
  (.5, math.nextafter(7.0, 0.0), 1.2, True),
  (.5, 7.0, 1.2, False),
  (.5, math.nextafter(7.0, math.inf), 1.2, False),
  (.5, 6.0, math.nextafter(1.1, 0.0), False),
  (.5, 6.0, 1.1, False),
  (.5, 6.0, math.nextafter(1.1, math.inf), True),
  (.5, 6.0, 1.3, True),
  (.5, 6.0, math.nextafter(1.3, math.inf), False),
])
def test_exact_open_and_closed_rule_boundaries(side, width, distance, offset, restricted):
  # The upper 1.3 m boundary extends beyond the existing 1.25 m ordinary
  # corridor; test the rule itself without pretending that it widens admission.
  target = narrow_track(width=width, distance=distance, lateral=side * offset)
  assert _narrow_point_unconfirmed(target, distance, side * offset, now_ns=START + 2_000_000,
                                  observation_ns=START + 1_000_000, vision_confirmed=False) is restricted


@pytest.mark.parametrize('side', [-1, 1])
@pytest.mark.parametrize(('width', 'distance', 'offset'), [(.70, 4.0, 1.2), (.5, 7.0, 1.2),
                                                           (.5, 4.0, 1.1), (.5, 4.0, .7), (.5, 4.0, 0.0)])
def test_current_targets_outside_the_restriction_keep_ordinary_roles(side, width, distance, offset):
  result = current_result(narrow_track(width=width, distance=distance, lateral=side * offset))
  lead, = present_leads(result)
  assert_ordinary(lead)


@pytest.mark.parametrize('fault', ['missing', 'invalid', 'different-scan', 'noninteger-stamp', 'unavailable',
                                  'available-before-position', 'zero-width', 'nan-width', 'nonfinite-edge',
                                  'inconsistent-span', 'reflection-outside-span', 'reflection-off-span'])
def test_missing_or_invalid_same_scan_extent_cannot_veto_a_valid_point(fault):
  target = narrow_track()
  if fault == 'missing':
    for key in tuple(target):
      if key.startswith('body'):
        del target[key]
  elif fault == 'invalid':
    target['bodyValid'] = False
  elif fault == 'different-scan':
    target['bodyMonoTime'] -= STEP
  elif fault == 'noninteger-stamp':
    target['bodyMonoTime'] = float(target['bodyMonoTime'])
  elif fault == 'unavailable':
    target['bodyAvailableMonoTime'] = START + 1_000_001
  elif fault == 'available-before-position':
    target['bodyAvailableMonoTime'] = START - 1
  elif fault == 'zero-width':
    target['bodyWidth'] = 0.0
  elif fault == 'nan-width':
    target['bodyWidth'] = math.nan
  elif fault == 'nonfinite-edge':
    target['bodyLeftY'] = math.inf
  elif fault == 'inconsistent-span':
    target['bodyLeftY'] += .1
  elif fault == 'reflection-outside-span':
    target['bodyLeftY'] += 1.0
    target['bodyRightY'] += 1.0
  else:
    target['bodyLeftX'] += .1
    target['bodyRightX'] += .1
  result = current_result(target)
  assert result.healthy and not result.degraded and not result.cleared
  lead, = present_leads(result)
  assert_ordinary(lead)


@pytest.mark.parametrize('side', [-1, 1])
@pytest.mark.parametrize(('reflection', 'left', 'right', 'restricted'), [
  (1.2, 1.2, .7, True),  # Inner edge and span center are below the offset band.
  (1.0, 1.5, 1.0, False),  # Span center is in the band, reflection is not.
])
def test_offset_is_the_reflection_not_the_near_edge_or_span_center(side, reflection, left, right, restricted):
  target = narrow_track(lateral=side * reflection)
  target['bodyLeftY'], target['bodyRightY'] = sorted((side * left, side * right), reverse=True)
  leads = present_leads(current_result(target))
  assert bool(leads) is (not restricted)
  if leads:
    assert_ordinary(leads[0])


@pytest.mark.parametrize('side', [-1, 1])
@pytest.mark.parametrize(('reflection', 'path_center', 'restricted'), [(1.2, .4, False), (.8, -.4, True)])
def test_recomputed_current_path_offset_controls_the_rule(side, reflection, path_center, restricted):
  target = narrow_track(lateral=side * reflection)
  inputs = scene(tracks=[target])
  inputs['model']['position']['y'] = [-side * path_center] * 33
  leads = present_leads(HondaRadarControl().update(**inputs))
  assert bool(leads) is (not restricted)
  if leads:
    assert leads[0]['evidence']['pathOffset'] == pytest.approx(side * .8)
    assert_ordinary(leads[0])


@pytest.mark.parametrize('side', [-1, 1])
def test_current_same_object_visual_confirmation_preserves_radar_authority(side):
  target = narrow_track(lateral=side * 1.2)
  visual = vision(distance=4.0, speed=4.0)
  result = current_result(target, leads=[visual], bindings=[binding(bias_y=side * 1.2)])
  lead, = present_leads(result)
  assert_ordinary(lead)
  assert lead['evidence']['source'] == 'fused' and lead['evidence']['visionIndex'] == 0


@pytest.mark.parametrize('case', ['unbound', 'different-object', 'current-vision-jump', 'weak-current-vision',
                                 'absent-current-vision'])
def test_visual_presence_or_old_association_alone_cannot_exempt_the_radar_point(case):
  target = narrow_track()
  visual = vision(distance=4.0, speed=4.0)
  association = binding(bias_y=1.2)
  if case == 'unbound':
    associations = []
  else:
    associations = [association]
    if case == 'different-object':
      association.update(trackId=2, candidateTrackId=2)
    elif case == 'current-vision-jump':
      visual['dRel'] = 20.0
    elif case == 'weak-current-vision':
      visual['modelProb'] = .79
    else:
      visual = {'present': False}
  result = current_result(target, leads=[visual], bindings=associations)
  assert result.healthy and not result.degraded
  leads = present_leads(result)
  assert all(not lead['radar'] for lead in leads)
  if case != 'absent-current-vision':
    assert len(leads) == 1 and qualified_lead(leads[0], 'brake')


@pytest.mark.parametrize('recovery', ['centered-point', 'wider-span', 'farther-point', 'missing-span', 'current-vision'])
def test_current_measurement_withdraws_then_recovers_without_a_blacklist(recovery):
  control = HondaRadarControl()
  initial = current_result(narrow_track(width=.8), control=control)
  assert_ordinary(present_leads(initial)[0])
  withheld = current_result(narrow_track(1), control=control, frame=1)
  assert not present_leads(withheld)
  assert withheld.healthy and not withheld.degraded and not withheld.cleared
  target, options = narrow_track(2), {}
  if recovery == 'centered-point':
    target = narrow_track(2, lateral=1.0)
  elif recovery == 'wider-span':
    target = narrow_track(2, width=.70)
  elif recovery == 'farther-point':
    target = narrow_track(2, distance=7.0)
  elif recovery == 'missing-span':
    target['bodyValid'] = False
  else:
    options = {'leads': [vision(distance=4.0, speed=4.0)], 'bindings': [binding(bias_y=1.2)]}
  recovered = current_result(target, control=control, frame=2, **options)
  assert recovered.healthy and not recovered.degraded and not recovered.cleared
  assert_ordinary(present_leads(recovered)[0])


def test_lost_current_visual_confirmation_does_not_keep_the_point_exempt():
  control = HondaRadarControl()
  initial = current_result(narrow_track(), control=control, leads=[vision(distance=4.0, speed=4.0)],
                           bindings=[binding(bias_y=1.2)])
  assert_ordinary(present_leads(initial)[0])
  # A new model frame actually lacks the object; the old same binding alone
  # cannot preserve its exemption or renew its former stopping obligation.
  current = current_result(narrow_track(1), control=control, frame=1, bindings=[binding(bias_y=1.2)])
  assert current.healthy and not current.degraded and not current.cleared
  assert not present_leads(current)


@pytest.mark.parametrize('side', [-1, 1])
def test_current_independent_measured_body_cut_in_keeps_only_its_brake_role(side):
  control = HondaRadarControl()
  for frame in range(3):
    # The .5 m measured span is asymmetric around its reflection. Its current
    # near edge penetrates the separate .85 m near cut-in boundary at frame 2.
    lateral = 1.2 + .03 * (2 - frame)
    target = narrow_track(frame, lateral=side * lateral, distance=5.5 - .1 * frame,
                          speed=4.0, lateralSpeed=-side * .45)
    target['bodyLeftY'], target['bodyRightY'] = sorted((side * (lateral + .14), side * (lateral - .36)), reverse=True)
    current = current_result(target, control=control, frame=frame)
    if frame < 2:
      assert not present_leads(current)
  lead, = present_leads(current)
  assert CUT_IN_REASON in lead['evidence']['reason']
  assert qualified_lead(lead, 'brake')
  assert not any(qualified_lead(lead, role) for role in ('follow', 'stop', 'departure', 'fcw'))


@pytest.mark.parametrize(('distance', 'speed', 'restricted'), [(7.1, 4.0, True), (6.8, 14.0, False)])
def test_distance_is_aligned_to_the_current_camera_epoch(distance, speed, restricted):
  target = narrow_track(distance=distance, speed=speed)
  inputs = scene(1, tracks=[target])
  inputs['observation_ns'] = START + 1_000_000
  inputs['observation']['fusion'].update(modelMonoTime=START, modelTimestampEof=START, egoMonoTime=START)
  result = HondaRadarControl().update(**inputs)
  leads = present_leads(result)
  assert result.healthy and not result.degraded
  assert bool(leads) is (not restricted)
  if leads:
    assert leads[0]['dRel'] == pytest.approx(distance + (speed - inputs['v_ego']) * STEP / 1e9)
    assert_ordinary(leads[0])


def test_nonzero_path_heading_uses_normal_offset_instead_of_sensor_lateral_position():
  distance, slope = 4.0, .15
  target = narrow_track(distance=distance, lateral=slope * distance + 1.2 * math.hypot(1.0, slope),
                        lateralSpeed=4.0 * slope)
  inputs = scene(tracks=[target])
  xs = [2.0 * i for i in range(33)]
  inputs['model']['position'] = {'x': [x + 1.52 for x in xs], 'y': [-slope * x for x in xs]}
  assert target['yRel'] > 1.3
  result = HondaRadarControl().update(**inputs)
  assert result.healthy and not result.degraded and not present_leads(result)


def test_restricting_one_point_preserves_the_other_current_radar_object():
  other = narrow_track(distance=6.0, lateral=0.0, trackId=2, oemTrackId=8)
  result = HondaRadarControl().update(**scene(tracks=[narrow_track(), other]))
  lead, = present_leads(result)
  assert lead['evidence']['trackId'] == 2
  assert_ordinary(lead)
  assert result.healthy and not result.degraded and not result.cleared


def test_degenerate_tiny_width_is_invalid_extent_without_a_division_error():
  target = narrow_track(width=math.nextafter(0.0, 1.0))
  assert target['bodyLeftY'] == target['bodyRightY']
  result = current_result(target)
  assert result.healthy and not result.degraded
  assert_ordinary(present_leads(result)[0])


def test_current_visual_fallback_uses_visual_components_after_point_withdrawal():
  control = HondaRadarControl()
  first = current_result(narrow_track(), control=control, leads=[vision(distance=4.0, speed=4.0)],
                         bindings=[binding(bias_y=1.2)])
  assert_ordinary(present_leads(first)[0])
  visual = vision(distance=4.0 - 6.0 * STEP / 1e9, speed=4.0)
  current = current_result(narrow_track(1, distance=3.65), control=control, frame=1, leads=[visual])
  lead, = present_leads(current)
  assert not lead['radar'] and lead['evidence']['reason'] == 'visionFallback'
  assert lead['dRel'] == visual['dRel'] and lead['yRel'] == visual['yRel']
  assert lead['evidence']['positionMonoTime'] == START + STEP
  assert current.healthy and not current.degraded and not current.cleared
