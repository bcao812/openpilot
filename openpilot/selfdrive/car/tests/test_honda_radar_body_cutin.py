"""Current measured body intrusion, with independent ordinary-follow roles.

The near/closing case resembles 95-B001's geometry; these are synthetic
observations, not a replay or a measurement of that event's response gain.
"""
import copy
import math

import pytest

from openpilot.selfdrive.car.honda_radar_control import CUT_IN_HALF_WIDTH, CUT_IN_REASON, SWEPT_HALF_WIDTH, HondaRadarControl
from openpilot.selfdrive.car.tests.test_honda_radar_control import START, STEP, native_motion_quality, present_leads, radar, scene, vision
from openpilot.selfdrive.controls.lib.lead_qualification import qualified_lead


def body_track(frame, *, side=1, lateral=None, **changes):
  lateral = side * (1.85 - .03 * frame) if lateral is None else lateral
  target = radar(frame, identifier=572, oem=46, distance=16.0 - 7.62 * frame * STEP / 1e9,
                 speed=13.3, lateral=lateral, lateral_speed=-side * .45)
  edges = sorted((lateral + side * .6, lateral - side * .8))
  target.update(bodyValid=True, bodyMonoTime=target['positionMonoTime'], bodyAvailableMonoTime=target['positionMonoTime'],
                bodyWidth=1.4, bodyLeftX=target['dRel'], bodyLeftY=edges[1],
                bodyRightX=target['dRel'], bodyRightY=edges[0])
  target.update(changes)
  if 'nativeMotion' not in changes:
    target['nativeMotion'] = native_motion_quality(target)
  return target


def update(control, frame, target, **kwargs):
  return control.update(**scene(frame, tracks=[target], ego=20.92, **kwargs))


def near_body_track(frame, *, side=1, distance=5.5, edge=.84):
  target = body_track(frame, side=side, lateral=side * (edge + .8), dRel=distance,
                      bodyLeftX=distance, bodyRightX=distance)
  # Set the measured endpoints directly so the strict .85 m boundary is not
  # shifted by adding/subtracting the reflection offset in this fixture.
  right, left = sorted((side * edge, side * (edge + 1.4)))
  target.update(bodyLeftY=left, bodyRightY=right)
  return target


@pytest.mark.parametrize('side', [-1, 1])
def test_measured_body_enters_before_the_point_and_only_gets_braking(side):
  control = HondaRadarControl()
  for frame in range(3):
    target = body_track(frame, side=side)
    result = update(control, frame, target, leads=[vision(distance=80.5, speed=20.92, ego=20.92)])
    assert abs(target['yRel']) > SWEPT_HALF_WIDTH
    if frame < 2:
      assert all(lead['evidence']['trackId'] != 572 for lead in present_leads(result))
  lead = result.leads[0]
  assert lead['evidence']['trackId'] == 572 and result.leads[1]['dRel'] == 80.5
  assert CUT_IN_REASON in lead['evidence']['reason']
  assert qualified_lead(lead, 'brake', now_ns=START + 2 * STEP + 2_000_000)
  assert all(not qualified_lead(lead, purpose) for purpose in ('follow', 'stop', 'departure', 'fcw'))
  assert lead['evidence']['followGeometryMonoTime'] == 0
  assert not lead['evidence']['accelerationValid'] and lead['aLeadK'] == 0.0
  assert not control._tracks[572].path.confirmed and not control._tracks[572].selected_follow


@pytest.mark.parametrize('side', [-1, 1])
@pytest.mark.parametrize(('edge', 'admitted'), [(.99, True), (1.0, True), (1.01, False), (1.24, False)])
def test_body_cut_in_uses_its_own_one_meter_band(side, edge, admitted):
  control = HondaRadarControl()
  for frame in range(3):
    target = body_track(frame, side=side, lateral=side * (edge + .8 + .03 * (2 - frame)))
    result = update(control, frame, target)
    assert abs(target['yRel']) > SWEPT_HALF_WIDTH
    if frame < 2:
      assert not present_leads(result)
  leads = present_leads(result)
  assert bool(leads) is admitted
  if admitted:
    assert CUT_IN_REASON in leads[0]['evidence']['reason']
    assert qualified_lead(leads[0], 'brake') and not qualified_lead(leads[0], 'follow')


def test_a_future_body_entry_is_not_current_occupancy():
  control = HondaRadarControl()
  for frame in range(4):
    target = body_track(frame, lateral=2.3 - .02 * frame)
    assert target['bodyRightY'] > CUT_IN_HALF_WIDTH
    assert not present_leads(update(control, frame, target))


@pytest.mark.parametrize('case', ['parallel', 'away', 'stationary-guardrail', 'opposing', 'unreliable-velocity',
                                 'no-lateral-evidence', 'uncertain-position', 'far-track146', 'no-closing-conflict'])
def test_width_does_not_admit_the_negative_controls(case):
  control = HondaRadarControl()
  for frame in range(4):
    changes = {}
    if case == 'parallel':
      changes = {'lateralSpeed': 0.0}
    elif case == 'away':
      changes = {'lateralSpeed': .45}
    elif case == 'stationary-guardrail':
      changes = {'groundSpeed': 0.0}
    elif case == 'opposing':
      changes = {'groundSpeed': -5.0}
    elif case == 'unreliable-velocity':
      changes = {'velocityGood': False, 'observations': 80}
    elif case == 'no-lateral-evidence':
      changes = {'lateralVelocityValid': False}
    elif case == 'uncertain-position':
      changes = {'lateralPositionVariance': 1.0}
    elif case == 'no-closing-conflict':
      changes = {'groundSpeed': 20.92}
    else:
      changes = {'trackId': 146, 'velocityGood': False, 'observations': 6 + frame,
                 'dRel': 77.0, 'bodyLeftX': 77.0, 'bodyRightX': 77.0, 'groundSpeed': -1.75}
    assert not present_leads(update(control, frame, body_track(frame, **changes)))


def test_parallel_target_on_a_curve_is_not_a_cut_in():
  control = HondaRadarControl()
  for frame in range(4):
    target = body_track(frame)
    distance = target['dRel']
    target = body_track(frame, lateral=1.75 + .003 * distance**2, lateralSpeed=13.3 * .006 * distance)
    inputs = scene(frame, tracks=[target], ego=20.92)
    xs = [2.0 * i for i in range(33)]
    inputs['model']['position'] = {'x': [x + 1.52 for x in xs], 'y': [-.003 * x**2 for x in xs]}
    assert not present_leads(control.update(**inputs))


@pytest.mark.parametrize('fault', ['missing', 'stale', 'future', 'invalid-width', 'generation-change', 'width-spike', 'native-quality'])
def test_body_evidence_loss_or_identity_change_withdraws_the_brake_role(fault):
  control = HondaRadarControl()
  for frame in range(3):
    result = update(control, frame, body_track(frame))
  assert present_leads(result)
  target = body_track(3)
  if fault == 'missing':
    target['bodyValid'] = False
  elif fault == 'stale':
    target['bodyMonoTime'] = START
  elif fault == 'future':
    target['bodyAvailableMonoTime'] = START + 4 * STEP
  elif fault == 'invalid-width':
    target['bodyWidth'] = math.nan
  elif fault == 'generation-change':
    target['generation'] += 1
  elif fault == 'native-quality':
    target['velocityGood'] = False
    target['nativeMotion'] = native_motion_quality(target, qualified=False, reason='invalidVelocity')
  else:
    target.update(bodyWidth=3.0, bodyLeftY=target['yRel'] + 1.5, bodyRightY=target['yRel'] - 1.5)
  assert not present_leads(update(control, 3, target))


def test_mature_neighbor_does_not_restart_native_confirmation_when_body_enters():
  control = HondaRadarControl()
  for frame in range(5):
    target = body_track(frame, lateral=2.0 - .06 * frame)
    assert target['nativeMotion']['qualified'] and target['nativeMotion']['sampleCount'] >= 5
    current = update(control, frame, target)
    if frame < 4:
      assert not present_leads(current)
  lead, = present_leads(current)
  assert CUT_IN_REASON in lead['evidence']['reason']
  assert qualified_lead(lead, 'brake')
  assert not any(qualified_lead(lead, purpose) for purpose in ('follow', 'stop', 'departure', 'fcw'))


def test_repeated_sample_does_not_establish_persistent_approach():
  control = HondaRadarControl()
  target = body_track(0, lateral=1.75)
  for frame in range(3):
    assert not present_leads(update(control, frame, copy.deepcopy(target)))
  assert control._body_approaches[572].samples == 1


def test_former_point_confirmation_does_not_upgrade_a_current_body_only_conflict():
  control = HondaRadarControl()
  initial = update(control, 0, body_track(0, lateral=.8))
  assert qualified_lead(initial.leads[0], 'follow') and qualified_lead(initial.leads[0], 'stop')
  for frame in range(1, 4):
    current = update(control, frame, body_track(frame))
    assert current.healthy and not current.degraded and not current.cleared
    if frame < 3:
      assert not present_leads(current)
  lead, = present_leads(current)
  assert CUT_IN_REASON in lead['evidence']['reason'] and qualified_lead(lead, 'brake')
  assert not any(qualified_lead(lead, purpose) for purpose in ('follow', 'stop', 'departure', 'fcw'))


@pytest.mark.parametrize('lateral', [-1.24, -1.1, .2, 1.1, 1.24])
def test_missing_width_preserves_existing_point_following(lateral):
  target = radar(0, distance=16.0, lateral=lateral)
  lead = update(HondaRadarControl(), 0, target).leads[0]
  assert qualified_lead(lead, 'follow') and qualified_lead(lead, 'stop')


@pytest.mark.parametrize('side', [-1, 1])
@pytest.mark.parametrize('distance', [5.999, 6.0, 6.001])
@pytest.mark.parametrize(('edge', 'near_admitted'), [(.851, False), (.850, False), (.849, True)])
def test_near_body_requires_strictly_more_than_fifteen_centimeters(side, distance, edge, near_admitted):
  control = HondaRadarControl()
  for frame in range(3):
    target = near_body_track(frame, side=side, distance=distance + (2-frame)*.1,
                             edge=edge + (2-frame)*.03)
    result = update(control, frame, target)
  leads = present_leads(result)
  assert bool(leads) is (near_admitted or distance >= 6.0)
  if leads:
    lead, = leads
    assert CUT_IN_REASON in lead['evidence']['reason']
    assert qualified_lead(lead, 'brake')
    assert not any(qualified_lead(lead, role) for role in ('follow', 'stop', 'departure', 'fcw'))


@pytest.mark.parametrize('side', [-1, 1])
def test_near_shallow_approach_history_allows_current_deep_entry_without_new_delay(side):
  control = HondaRadarControl()
  for frame, edge in enumerate([1.04, 1.01, .98, .95, .92, .89, .86, .84]):
    result = update(control, frame, near_body_track(frame, side=side, distance=5.8-.1*frame, edge=edge))
    assert bool(present_leads(result)) is (frame == 7)
  assert control._body_approaches[572].samples == 8


@pytest.mark.parametrize('side', [-1, 1])
def test_crossing_six_meters_withdraws_shallow_body_then_current_depth_restores_it(side):
  control = HondaRadarControl()
  for frame, (distance, edge, admitted) in enumerate([
      (6.3, .96, False), (6.2, .93, False), (6.1, .90, True), (5.99, .89, False), (5.9, .84, True)]):
    result = update(control, frame, near_body_track(frame, side=side, distance=distance, edge=edge))
    assert bool(present_leads(result)) is admitted
    assert result.healthy and not result.degraded and not result.cleared
  # No lock-in of the former brake role when a new observation is too shallow.
  result = update(control, 5, near_body_track(5, side=side, distance=5.8, edge=.86))
  assert not present_leads(result)


@pytest.mark.parametrize('side', [-1, 1])
@pytest.mark.parametrize(('base_distance', 'admitted'), [(6.005, False), (6.025, True)])
def test_near_body_gate_uses_the_closest_projected_endpoint(side, base_distance, admitted):
  control = HondaRadarControl()
  for frame in range(3):
    edge = .90 + (2-frame)*.03
    target = near_body_track(frame, side=side, distance=base_distance+(2-frame)*.1, edge=edge)
    target.update(bodyLeftX=target['dRel']+side*.01, bodyRightX=target['dRel']-side*.01,
                  bodyWidth=math.hypot(1.4, .02))
    # The reflection lies 0.8/1.4 of the way from the near to the far edge.
    target['dRel'] = base_distance+(2-frame)*.1 + .01*(.8/.7-1)
    result = update(control, frame, target)
  assert target['dRel'] > 6.0
  assert (min(target['bodyLeftX'], target['bodyRightX']) < 6.0) is (not admitted)
  assert control._body_approaches[572].samples == 3
  assert bool(present_leads(result)) is admitted


@pytest.mark.parametrize('side', [-1, 1])
@pytest.mark.parametrize(('edge', 'admitted'), [(.84, True), (.86, False)])
def test_near_body_depth_is_relative_to_the_curved_path(side, edge, admitted):
  control = HondaRadarControl()
  for frame in range(3):
    distance = 5.5 - .1*frame
    target = near_body_track(frame, side=side, distance=distance,
                             edge=edge+(2-frame)*.03+.003*distance**2)
    target['lateralSpeed'] = side*(target['groundSpeed']*.006*distance-.45)
    inputs = scene(frame, tracks=[target], ego=20.92)
    xs = [2.0*i for i in range(33)]
    inputs['model']['position'] = {'x': [x+1.52 for x in xs], 'y': [-side*.003*x**2 for x in xs]}
    result = control.update(**inputs)
  assert min(abs(target['bodyLeftY']), abs(target['bodyRightY'])) > .85
  assert control._body_approaches[572].samples == 3
  assert bool(present_leads(result)) is admitted


@pytest.mark.parametrize('lateral', [-1.1, 0.0, 1.1])
def test_near_body_margin_does_not_block_an_ordinary_current_point_target(lateral):
  result = update(HondaRadarControl(), 0, radar(0, distance=4.0, lateral=lateral))
  lead, = present_leads(result)
  assert CUT_IN_REASON not in lead['evidence']['reason']
  assert all(qualified_lead(lead, role) for role in ('brake', 'follow', 'stop'))
