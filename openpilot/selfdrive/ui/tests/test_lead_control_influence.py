from copy import deepcopy
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.ui.sunnypilot.onroad.honda_radar_state import RadarLabelPoint
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_control_influence import (
  MESSAGE_MAX_AGE_NS, ObjectInfluence, accepted_influences, influence_rgb, lead_influence_strength, radar_influence_strength,
)


NOW = 1_000_000_000
MODEL = NOW - 70_000_000
PLAN = NOW - 40_000_000


class State(dict):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.valid = dict.fromkeys(self, True)
    self.logMonoTime = dict.fromkeys(self, NOW - 10_000_000)


def influence(**changes):
  return dict({'version': 1, 'sessionId': 'radar', 'trackId': 42, 'modelMonoTime': MODEL,
               'visionIndex': 0, 'kind': 'braking', 'strength': 0.6}, **changes)


def state(*items):
  return State({
    'controlsState': {'longitudinalPlanMonoTime': PLAN,
                      'motionState': {'enabled': True, 'planningMode': 2, 'phase': 'track',
                                      'degraded': False, 'waitingForFreshInput': False, 'planMonoTime': PLAN,
                                      'objectInfluences': list(items or (influence(),))}},
    'carControl': {'enabled': True, 'longActive': True},
    'selfdriveState': {'enabled': True},
    'carState': {'gasPressed': False, 'brakePressed': False},
  })


def lead(**changes):
  return {'present': True, 'dRel': 12.0, 'modelProb': 0.01,
          'evidence': dict({'version': 1, 'source': 'fused', 'sessionId': 'radar', 'trackId': 42,
                            'generation': 1, 'modelMonoTime': MODEL, 'visionIndex': 0}, **changes)}


def test_only_the_executed_receipt_colors_targets():
  sm = state()
  original = deepcopy(sm)
  # A newer planner candidate has already selected another target. Only the
  # owner-acknowledged plan may describe what the controller actually used.
  sm['longitudinalPlan'] = {'objectInfluences': [influence(trackId=63, strength=1.0)]}
  items = accepted_influences(sm, NOW)
  assert items == (ObjectInfluence('radar', 42, MODEL, 0, 'braking', 0.6),)
  assert sm['controlsState'] == original['controlsState']
  assert lead_influence_strength(lead(), items) == 0.6
  assert lead_influence_strength(lead(trackId=63), items) == 0.0
  sm['controlsState']['motionState']['objectInfluences'] = []
  assert accepted_influences(sm, NOW) == ()


def test_same_car_radar_and_formal_fused_labels_share_control_color():
  items = accepted_influences(state(), NOW)
  point = RadarLabelPoint(2, 11.5, 0.1, MODEL, 3, 2, 'radar', 42)
  assert radar_influence_strength(point, items) == lead_influence_strength(lead(generation=7), items) == 0.6
  assert lead_influence_strength(lead(source='vision', sessionId='', trackId=0), items) == 0.6
  assert lead_influence_strength(lead(source='vision', sessionId='', trackId=0, visionIndex=1), items) == 0.0
  # Neither matching distance nor a reused slot is a target identity.
  for session, identifier in (('restarted', 42), ('radar', 63), ('radar', 0), ('radar', -1)):
    other = RadarLabelPoint(2, point.distance, point.lateral, point.time_ns, 3, 2, session, identifier)
    assert radar_influence_strength(other, items) == 0.0


def test_visual_identity_is_bound_to_the_exact_model_frame_and_hypothesis():
  items = accepted_influences(state(influence(sessionId='', trackId=0, kind='follow')), NOW)
  visual = lead(source='vision', sessionId='', trackId=0)
  assert lead_influence_strength(visual, items) == 0.6
  for changes in ({'modelMonoTime': MODEL + 50_000_000}, {'visionIndex': 1}, {'version': 0}):
    other = deepcopy(visual)
    other['evidence'].update(changes)
    assert lead_influence_strength(other, items) == 0.0
  assert radar_influence_strength(RadarLabelPoint(0, 12, 0, MODEL, 3, 2, 'radar', 42), items) == 0.0


def test_current_owner_receipt_can_color_old_layout_pure_visual_lead():
  items = accepted_influences(state(influence(sessionId='', trackId=0)), NOW)
  visual = {'present': True, 'radar': False, 'dRel': 12.0, 'modelProb': 0.99}
  assert lead_influence_strength(visual, items, legacy_model_ns=MODEL, legacy_slot=0) == 0.6
  assert lead_influence_strength(visual, items) == 0.0
  assert lead_influence_strength(visual, (), legacy_model_ns=MODEL, legacy_slot=0) == 0.0
  for model_ns, slot in ((MODEL + 1, 0), (MODEL, 1), (0, 0), (MODEL, None), (MODEL, True)):
    assert lead_influence_strength(visual, items, legacy_model_ns=model_ns, legacy_slot=slot) == 0.0
  for changes in ({'radar': True}, {'evidence': {'version': 2}}, {'present': False}):
    assert lead_influence_strength(dict(visual, **changes), items, legacy_model_ns=MODEL, legacy_slot=0) == 0.0
  # An old radar lead has no proven physical identity. Even a fused record
  # with a matching visual frame cannot manufacture an old-layout binding.
  assert lead_influence_strength(visual, accepted_influences(state(), NOW), legacy_model_ns=MODEL, legacy_slot=0) == 0.0


@pytest.mark.parametrize('service', ('controlsState', 'carControl', 'carState', 'selfdriveState'))
@pytest.mark.parametrize('fault', ('invalid', 'stale', 'future'))
def test_invalid_stale_or_future_execution_messages_cannot_keep_red(service, fault):
  sm = state()
  if fault == 'invalid':
    sm.valid[service] = False
  elif fault == 'stale':
    sm.logMonoTime[service] = NOW - MESSAGE_MAX_AGE_NS - 1
  else:
    sm.logMonoTime[service] = NOW + 1
  assert accepted_influences(sm, NOW) == ()


@pytest.mark.parametrize(('service', 'field', 'value'), (
  ('carControl', 'longActive', False), ('carControl', 'enabled', False), ('selfdriveState', 'enabled', False),
  ('carState', 'gasPressed', True), ('carState', 'brakePressed', True),
))
def test_disengagement_and_driver_override_remove_control_coloring(service, field, value):
  sm = state()
  sm[service][field] = value
  assert accepted_influences(sm, NOW) == ()


@pytest.mark.parametrize('changes', (
  {'enabled': False}, {'planningMode': 0}, {'planningMode': 1}, {'phase': 'off'},
  {'degraded': True}, {'waitingForFreshInput': True}, {'planMonoTime': 0},
  {'planMonoTime': NOW + 1}, {'planMonoTime': PLAN + 1},
))
def test_old_modes_and_unaccepted_owner_state_do_not_color(changes):
  sm = state()
  sm['controlsState']['motionState'].update(changes)
  assert accepted_influences(sm, NOW) == ()


@pytest.mark.parametrize('changes', (
  {'version': 0}, {'version': 2}, {'kind': 'fcw'}, {'strength': 0.0}, {'strength': -0.1},
  {'strength': 1.01}, {'strength': float('nan')}, {'strength': float('inf')}, {'strength': True},
  {'sessionId': '', 'trackId': 42}, {'trackId': 0}, {'trackId': True},
  {'sessionId': '', 'trackId': 0, 'modelMonoTime': 0}, {'sessionId': '', 'trackId': 0, 'visionIndex': 2},
))
def test_malformed_or_noncontrol_records_cannot_paint_a_target(changes):
  assert accepted_influences(state(influence(**changes)), NOW) == ()


@pytest.mark.parametrize('kind', ('follow', 'braking', 'hold', 'release'))
def test_all_acknowledged_control_purposes_use_the_same_ui_strength_scale(kind):
  sm = state(influence(kind=kind))
  sm['controlsState']['motionState']['phase'] = {'follow': 'track', 'braking': 'approach', 'hold': 'hold', 'release': 'release'}[kind]
  items = accepted_influences(sm, NOW)
  assert items[0].kind == kind
  assert influence_rgb(lead_influence_strength(lead(), items)) == influence_rgb(0.6)


def test_adopted_caution_ceiling_has_only_the_fixed_light_red_strength():
  items = accepted_influences(state(influence(kind='caution', strength=0.15)), NOW)
  assert items[0].kind == 'caution'
  assert lead_influence_strength(lead(), items) == 0.15
  assert influence_rgb(lead_influence_strength(lead(), items)) == (246, 135, 137)
  for strength in (0.1, 0.6, 1.0):
    assert accepted_influences(state(influence(kind='caution', strength=strength)), NOW) == ()


def test_stronger_control_effort_is_darker_red_without_reusing_distance_or_confidence():
  colors = [influence_rgb(value) for value in (0.05, 0.3, 0.6, 1.0)]
  assert all(red > green and red > blue for red, green, blue in colors)
  assert all(sum(before) > sum(after) for before, after in zip(colors, colors[1:], strict=False))
  for value in (0.0, -1.0, 1.1, float('nan'), float('inf'), None, True):
    assert influence_rgb(value) is None
  items = accepted_influences(state(), NOW)
  for distance, probability in ((3.0, 0.99), (100.0, 0.0)):
    target = lead()
    target.update(dRel=distance, modelProb=probability)
    assert lead_influence_strength(target, items) == 0.6


def test_legacy_or_missing_receipt_leaves_original_display_colors():
  assert accepted_influences(SimpleNamespace(valid={}, logMonoTime={}), NOW) == ()
  sm = state()
  del sm['controlsState']['motionState']['objectInfluences']
  assert accepted_influences(sm, NOW) == ()
  assert lead_influence_strength({'present': True}, ()) == 0.0
  assert lead_influence_strength(dict(lead(), present=False), accepted_influences(state(), NOW)) == 0.0
