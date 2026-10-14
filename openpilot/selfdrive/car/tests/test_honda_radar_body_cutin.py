"""Current measured body intrusion, with independent ordinary-follow roles.

The near/closing case resembles 95-B001's geometry; these are synthetic
observations, not a replay or a measurement of that event's response gain.
"""
import copy
import math

import pytest

from openpilot.selfdrive.car.honda_radar_control import CUT_IN_HALF_WIDTH, CUT_IN_REASON, SWEPT_HALF_WIDTH, HondaRadarControl
from openpilot.selfdrive.car.tests.test_honda_radar_control import START, STEP, present_leads, radar, scene, vision
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
  return target


def update(control, frame, target, **kwargs):
  return control.update(**scene(frame, tracks=[target], ego=20.92, **kwargs))


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


@pytest.mark.parametrize('fault', ['missing', 'stale', 'future', 'invalid-width', 'generation-change', 'width-spike'])
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
  else:
    target.update(bodyWidth=3.0, bodyLeftY=target['yRel'] + 1.5, bodyRightY=target['yRel'] - 1.5)
  assert not present_leads(update(control, 3, target))


def test_repeated_sample_does_not_establish_persistent_approach():
  control = HondaRadarControl()
  target = body_track(0, lateral=1.75)
  for frame in range(3):
    assert not present_leads(update(control, frame, copy.deepcopy(target)))
  assert control._body_approaches[572].samples == 1


@pytest.mark.parametrize('lateral', [-1.24, -1.1, .2, 1.1, 1.24])
def test_missing_width_preserves_existing_point_following(lateral):
  target = radar(0, distance=16.0, lateral=lateral)
  lead = update(HondaRadarControl(), 0, target).leads[0]
  assert qualified_lead(lead, 'follow') and qualified_lead(lead, 'stop')
