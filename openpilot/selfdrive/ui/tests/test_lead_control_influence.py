from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.ui.sunnypilot.onroad import lead_control_influence
from openpilot.selfdrive.ui.sunnypilot.onroad.honda_radar_state import RadarLabelPoint
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_control_influence import (
  MESSAGE_MAX_AGE_NS, ObjectInfluence, _integer, _read, accepted_influences, influence_rgb, lead_influence_strength, radar_influence_strength,
)


NOW = 1_000_000_000
MODEL = NOW - 70_000_000
PLAN = NOW - 40_000_000


class IntSubclass(int):
  pass


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


@pytest.fixture(params=('reader', 'builder'))
def capnp_message(request):
  def convert(service, fields):
    message = getattr(log.Event.new_message(**{service: fields}), service)
    return message.as_reader() if request.param == 'reader' else message
  return convert


def capnp_state(sm, capnp_message):
  result = State({service: capnp_message(service, fields) for service, fields in sm.items()})
  result.valid = sm.valid.copy()
  result.logMonoTime = sm.logMonoTime.copy()
  return result


def received_state(*items):
  sm = state(*items)
  sm.frame = 7
  sm['controlsState'] = log.Event.new_message(controlsState=sm['controlsState']).controlsState.as_reader()
  return sm


def test_receipt_decode_is_shared_only_within_the_current_reader_frame():
  sm = received_state()
  with patch.object(lead_control_influence, '_decode_influences', wraps=lead_control_influence._decode_influences) as decode:
    first = accepted_influences(sm, NOW)
    assert accepted_influences(sm, NOW + 1) is first
    assert decode.call_count == 1

    sm.frame += 1
    assert accepted_influences(sm, NOW + 2) == first
    assert decode.call_count == 2
    sm.logMonoTime['controlsState'] += 1
    assert accepted_influences(sm, NOW + 3) == first
    assert decode.call_count == 3

    # A replacement reader invalidates the cache even when the timestamp and
    # frame are unchanged. An explicit withdrawal immediately removes color.
    replacement = state(influence(trackId=63, strength=0.8))['controlsState']
    sm['controlsState'] = log.Event.new_message(controlsState=replacement).controlsState.as_reader()
    assert accepted_influences(sm, NOW + 4) == (ObjectInfluence('radar', 63, MODEL, 0, 'braking', pytest.approx(0.8)),)
    replacement['motionState']['objectInfluences'] = []
    sm['controlsState'] = log.Event.new_message(controlsState=replacement).controlsState.as_reader()
    assert accepted_influences(sm, NOW + 5) == ()
    assert decode.call_count == 5


@pytest.mark.parametrize('builder', (False, True))
def test_mutable_receipts_are_not_cached_even_when_a_frame_is_present(builder):
  sm = state()
  sm.frame = 7
  if builder:
    sm['controlsState'] = log.Event.new_message(controlsState=sm['controlsState']).controlsState
    record = sm['controlsState'].motionState.objectInfluences[0]
  else:
    record = sm['controlsState']['motionState']['objectInfluences'][0]
  assert accepted_influences(sm, NOW)[0].strength == pytest.approx(0.6)
  if builder:
    record.strength = 0.25
  else:
    record['strength'] = 0.25
  assert accepted_influences(sm, NOW + 1)[0].strength == 0.25


def test_cached_receipt_rechecks_message_health_driver_override_and_expiry():
  sm = received_state()
  with patch.object(lead_control_influence, '_decode_influences', wraps=lead_control_influence._decode_influences) as decode:
    items = accepted_influences(sm, NOW)
    for service in ('controlsState', 'carControl', 'carState', 'selfdriveState'):
      sm.valid[service] = False
      assert accepted_influences(sm, NOW) == ()
      sm.valid[service] = True
      stamp = sm.logMonoTime[service]
      for invalid_stamp in (NOW + 1, NOW - MESSAGE_MAX_AGE_NS - 1):
        sm.logMonoTime[service] = invalid_stamp
        assert accepted_influences(sm, NOW) == ()
      sm.logMonoTime[service] = stamp
    for service, field in (('carState', 'gasPressed'), ('carState', 'brakePressed'),
                           ('carControl', 'enabled'), ('carControl', 'longActive'), ('selfdriveState', 'enabled')):
      previous = sm[service][field]
      sm[service][field] = not previous
      assert accepted_influences(sm, NOW) == ()
      sm[service][field] = previous
    assert accepted_influences(sm, PLAN + MESSAGE_MAX_AGE_NS) is items
    assert accepted_influences(sm, PLAN + MESSAGE_MAX_AGE_NS + 1) == ()
    assert decode.call_count == 1


def test_shared_decode_keeps_current_motion_gates_and_physical_mitigation_rules():
  fields = state(influence(version=2, strength=1.0), influence(trackId=63))['controlsState']
  sm = received_state()

  def current_items(**changes):
    fields['motionState'].update(changes)
    sm['controlsState'] = log.Event.new_message(controlsState=fields).controlsState.as_reader()
    items = accepted_influences(sm, NOW)
    assert accepted_influences(sm, NOW + 1) == items
    return items

  assert [item.track_id for item in current_items()] == [42, 63]
  assert [item.track_id for item in current_items(degraded=True)] == [42]
  assert current_items(phase='release') == ()
  assert [item.track_id for item in current_items(degraded=False)] == [63]
  for changes in ({'enabled': False}, {'enabled': True, 'planningMode': 1},
                  {'planningMode': 2, 'planMonoTime': PLAN + 1}):
    assert current_items(**changes) == ()


def test_indexed_strengths_preserve_duplicate_and_visual_identity_matching():
  items = accepted_influences(state(
    influence(strength=0.4), influence(kind='hold', strength=0.9),
    influence(sessionId='', trackId=0, strength=0.6),
    influence(sessionId='', trackId=0, visionIndex=1, strength=0.3),
  ), NOW)
  plain_items = tuple(items)
  point = RadarLabelPoint(2, 11.5, 0.1, MODEL, 3, 2, 'radar', 42)
  assert radar_influence_strength(point, items) == radar_influence_strength(point, plain_items) == 0.9
  for changes, expected in (({}, 0.9), ({'trackId': 63}, 0.0),
                            ({'source': 'vision', 'sessionId': '', 'trackId': 0}, 0.9),
                            ({'source': 'vision', 'sessionId': '', 'trackId': 0, 'visionIndex': 1}, 0.3),
                            ({'source': 'vision', 'sessionId': '', 'trackId': 0, 'modelMonoTime': MODEL + 1}, 0.0)):
    assert lead_influence_strength(lead(**changes), items) == lead_influence_strength(lead(**changes), plain_items) == expected
  legacy = {'present': True, 'radar': False}
  for slot, expected in ((0, 0.6), (1, 0.3)):
    for lookup_items in (items, plain_items):
      assert lead_influence_strength(legacy, lookup_items, legacy_model_ns=MODEL, legacy_slot=slot) == expected


@pytest.mark.parametrize(('value', 'minimum', 'expected'), (
  (-2, -1, False), (-1, -1, True), (-1, 0, False), (0, 0, True), (0, 1, False), (1, 1, True),
  (2 ** 64 - 1, 1, True), (2 ** 64 - 1, 2 ** 64, False),
  (IntSubclass(-1), -1, True), (IntSubclass(0), 1, False), (IntSubclass(42), 1, True),
  (np.int32(-1), -1, True), (np.int64(-1), 0, False), (np.int64(42), 1, True),
  (np.uint64(0), 1, False), (np.uint64(2 ** 64 - 1), 1, True),
  (True, 0, False), (False, -1, False), (np.bool_(True), 0, False),
  (1.0, 0, False), ('1', 0, False), (None, 0, False),
))
def test_integer_types_and_minimum_boundaries(value, minimum, expected):
  assert _integer(value, minimum) == expected


@pytest.mark.parametrize('integer', (int, IntSubclass, np.int64, np.uint64))
def test_integral_message_fields_preserve_receipts_and_control_colors(integer):
  fields = {'version': integer(1), 'trackId': integer(42), 'modelMonoTime': integer(MODEL), 'visionIndex': integer(0)}
  sm = state(influence(**fields))
  sm.logMonoTime = {service: integer(stamp) for service, stamp in sm.logMonoTime.items()}
  sm['controlsState']['longitudinalPlanMonoTime'] = integer(PLAN)
  sm['controlsState']['motionState']['planMonoTime'] = integer(PLAN)
  items = accepted_influences(sm, integer(NOW))
  assert items == (ObjectInfluence('radar', 42, MODEL, 0, 'braking', 0.6),)
  assert all(type(value) is int for value in (items[0].track_id, items[0].model_ns, items[0].vision_index))
  assert influence_rgb(lead_influence_strength(lead(**fields), items)) == (219, 74, 83)


def test_field_reads_do_not_probe_dynamic_readers_for_builder_methods():
  class Reader:
    def __init__(self):
      self.lookups = []

    def __getattr__(self, field):
      self.lookups.append(field)
      if field == 'strength':
        return 0.5
      raise AttributeError(field)

  reader = Reader()
  assert _read(reader, 'strength') == 0.5
  assert _read(reader, 'missing', 'fallback') == 'fallback'
  assert reader.lookups == ['strength', 'missing']


def test_capnp_reads_preserve_field_defaults(capnp_message):
  message = capnp_message('selfdriveState', {'enabled': True})
  assert _read(message, 'enabled') is True
  assert _read(message, 'missing', 'fallback') == 'fallback'
  assert _read(message, 'missing') is None


@pytest.mark.parametrize(('version', 'kind', 'strength', 'phase'), (
  (1, 'follow', 0.5, 'track'), (1, 'braking', 0.5, 'approach'), (1, 'hold', 0.5, 'hold'),
  (1, 'release', 0.5, 'release'), (1, 'caution', 0.15, 'track'), (2, 'braking', 1.0, 'hold'),
))
def test_capnp_receipts_and_leads_preserve_control_colors(capnp_message, version, kind, strength, phase):
  sm = state(influence(version=version, kind=kind, strength=strength))
  sm['controlsState']['motionState']['phase'] = phase
  items = accepted_influences(capnp_state(sm, capnp_message), NOW)
  assert len(items) == 1
  assert items[0].kind == kind
  assert items[0].strength == pytest.approx(strength)
  target = capnp_message('radarState', {'leadOne': lead()}).leadOne
  point = RadarLabelPoint(2, 11.5, 0.1, MODEL, 3, 2, 'radar', 42)
  assert radar_influence_strength(point, items) == lead_influence_strength(target, items)
  assert influence_rgb(lead_influence_strength(target, items)) == influence_rgb(strength)
  other = capnp_message('radarState', {'leadOne': lead(trackId=63)}).leadOne
  assert lead_influence_strength(other, items) == 0.0


@pytest.mark.parametrize(('service', 'field', 'value'), (
  ('carControl', 'longActive', False), ('carControl', 'enabled', False), ('selfdriveState', 'enabled', False),
  ('carState', 'gasPressed', True), ('carState', 'brakePressed', True),
))
@pytest.mark.parametrize('version', (1, 2))
def test_capnp_disengagement_and_driver_override_remove_control_colors(capnp_message, service, field, value, version):
  sm = state(influence(version=version, strength=1.0))
  sm[service][field] = value
  assert accepted_influences(capnp_state(sm, capnp_message), NOW) == ()


def test_capnp_receipts_keep_freshness_mode_and_constrained_motion_gates(capnp_message):
  sm = state(influence(version=2, strength=1.0), influence(trackId=63, strength=0.5))
  typed = capnp_state(sm, capnp_message)
  items = accepted_influences(typed, NOW)
  assert len(items) == 2
  assert accepted_influences(typed, PLAN + MESSAGE_MAX_AGE_NS) == items
  assert accepted_influences(typed, PLAN + MESSAGE_MAX_AGE_NS + 1) == ()
  typed.valid['controlsState'] = False
  assert accepted_influences(typed, NOW) == ()
  sm['controlsState']['motionState'].update(degraded=True, waitingForFreshInput=True)
  assert accepted_influences(capnp_state(sm, capnp_message), NOW) == (ObjectInfluence('radar', 42, MODEL, 0, 'braking', 1.0),)
  sm['controlsState']['motionState']['planningMode'] = 1
  assert accepted_influences(capnp_state(sm, capnp_message), NOW) == ()


def test_capnp_visual_identity_remains_bound_to_model_frame_and_slot(capnp_message):
  sm = state(influence(sessionId='', trackId=0, strength=0.5))
  items = accepted_influences(capnp_state(sm, capnp_message), NOW)
  for changes, expected in (({}, 0.5), ({'modelMonoTime': MODEL + 1}, 0.0), ({'visionIndex': 1}, 0.0)):
    visual = lead(source='vision', sessionId='', trackId=0, **changes)
    target = capnp_message('radarState', {'leadOne': visual}).leadOne
    assert lead_influence_strength(target, items) == expected
  legacy = capnp_message('radarState', {'leadOne': {'present': True, 'radar': False}}).leadOne
  assert lead_influence_strength(legacy, items, legacy_model_ns=MODEL, legacy_slot=0) == 0.5
  assert lead_influence_strength(legacy, items, legacy_model_ns=MODEL + 1, legacy_slot=0) == 0.0


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
@pytest.mark.parametrize('version', (1, 2))
def test_invalid_stale_or_future_execution_messages_cannot_keep_red(service, fault, version):
  sm = state(influence(version=version, strength=1.0))
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
@pytest.mark.parametrize('version', (1, 2))
def test_disengagement_and_driver_override_remove_control_coloring(service, field, value, version):
  sm = state(influence(version=version, strength=1.0))
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


@pytest.mark.parametrize('changes', ({'degraded': True}, {'waitingForFreshInput': True},
                                    {'degraded': True, 'waitingForFreshInput': True}))
def test_only_explicit_owner_adopted_physical_mitigation_survives_constrained_motion(changes):
  sm = state(influence(version=2, strength=1.0), influence(trackId=63, strength=1.0))
  sm['controlsState']['motionState'].update(changes)
  items = accepted_influences(sm, NOW)
  assert [(item.track_id, item.strength) for item in items] == [(42, 1.0)]
  assert influence_rgb(lead_influence_strength(lead(), items)) == (195, 20, 35)
  assert lead_influence_strength(lead(trackId=63), items) == 0.0
  assert lead_influence_strength(lead(source='vision', sessionId='', trackId=0), items) == 1.0
  sm['longitudinalPlan'] = {'objectInfluences': [influence(version=2, strength=1.0)]}
  sm['controlsState']['motionState']['objectInfluences'] = []
  assert accepted_influences(sm, NOW) == ()


@pytest.mark.parametrize('changes', (
  {'version': 1}, {'version': 3}, {'sessionId': ''}, {'trackId': 0}, {'sessionId': '', 'trackId': 0},
  {'strength': 0.6}, {'kind': 'follow'}, {'kind': 'hold'}, {'kind': 'release'}, {'kind': 'caution'},
))
def test_constrained_motion_cannot_turn_an_ordinary_or_unidentified_record_deep_red(changes):
  record = influence(version=2, strength=1.0)
  record.update(changes)
  sm = state(record)
  sm['controlsState']['motionState'].update(degraded=True, waitingForFreshInput=True)
  assert accepted_influences(sm, NOW) == ()


@pytest.mark.parametrize('changes', (
  {'enabled': False}, {'planningMode': 0}, {'planningMode': 1}, {'phase': 'off'}, {'phase': 'release'},
  {'planMonoTime': 0}, {'planMonoTime': NOW + 1}, {'planMonoTime': PLAN + 1},
))
def test_physical_mitigation_receipt_keeps_mode_phase_and_plan_identity_gates(changes):
  sm = state(influence(version=2, strength=1.0))
  sm['controlsState']['motionState'].update(degraded=True, **changes)
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
