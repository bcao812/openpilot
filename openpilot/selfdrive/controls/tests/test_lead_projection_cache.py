"""Current radar contents and independent output ownership across cached projection."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from openpilot.cereal import custom, log
from openpilot.selfdrive.car.tests.test_cpu4_car_control_sp import frozen_values
from openpilot.selfdrive.controls.tests.test_controlsd_ext import (
  BUTTON_STATES, ICBM_STATES, MADS_STATES, control_inputs,
)
from openpilot.sunnypilot.selfdrive.controls import controlsd_ext
from openpilot.sunnypilot.selfdrive.controls.controlsd_ext import ControlsExt


class UncachedControlsExt(ControlsExt):
  @staticmethod
  def get_lead_data(target, source):
    # An overridden mapper uses the original direct mapping contract.
    ControlsExt.get_lead_data(target, source)


def controls(cls=ControlsExt):
  instance = cls.__new__(cls)
  instance.radar_signal = True
  return instance


def compare(control, radar, state):
  inputs = {'radarState': radar, 'selfdriveStateSP': state}
  reference = controls(UncachedControlsExt)
  reference.radar_signal = control.radar_signal
  expected = reference.state_control_ext(inputs)
  actual = control.state_control_ext(inputs)
  assert frozen_values(actual.to_dict()) == frozen_values(expected.to_dict())
  return actual


@pytest.mark.parametrize('lead_fields', [{}, {'leadOne': {}}, {'leadOne': {'evidence': {}}},
                                       {'leadOne': {'present': True}, 'leadTwo': {'evidence': {}}}])
def test_default_and_missing_source_pointers_remain_equivalent(lead_fields):
  control = controls()
  state = custom.SelfdriveStateSP.new_message().as_reader()
  for _ in range(3):
    radar = log.RadarState.new_message(**lead_fields)
    compare(control, radar, state)
    compare(control, radar.as_reader(), state)


def test_current_mutable_contents_replace_cached_mapping_without_timestamp_change():
  control = controls()
  radar, state, _ = control_inputs(('leadOne', 'leadTwo'))
  reader = radar.as_reader()
  initial = compare(control, reader, state.as_reader())
  before = frozen_values(initial.to_dict())
  timestamp = radar.leadOne.evidence.positionMonoTime
  for value in (1.5, -0.0, float('nan'), float('inf'), -float('inf')):
    radar.leadOne.dRel = value
    radar.leadOne.evidence.physicalConflictStart = value
    radar.leadTwo.deprecated.vLat = value
    compare(control, reader, state.as_reader())
  radar.leadOne.present = False
  radar.leadOne.evidence.brakeQualified = False
  radar.leadOne.evidence.reason = 'current observation withdrew qualification'
  radar.leadTwo.evidence.trackId = (1 << 64) - 1
  result = compare(control, reader, state.as_reader())
  assert not result.leadOne.status and not result.leadOne.evidence.brakeQualified
  assert result.leadTwo.evidence.trackId == (1 << 64) - 1
  assert radar.leadOne.evidence.positionMonoTime == timestamp
  assert frozen_values(initial.to_dict()) == before


def test_outputs_and_private_templates_are_independent():
  control = controls()
  radar, state, _ = control_inputs(('leadOne', 'leadTwo'))
  original = compare(control, radar.as_reader(), state.as_reader())
  expected = frozen_values(original.to_dict())
  repeated = compare(control, radar.as_reader(), state.as_reader())
  original.leadOne.dRel = -100.0
  original.leadOne.evidence.reason = 'output-only mutation'
  original.leadTwo.evidence.trackId = 0
  assert frozen_values(repeated.to_dict()) == expected
  current = compare(control, radar.as_reader(), state.as_reader())
  assert frozen_values(current.to_dict()) == expected
  radar.leadOne.evidence.reason = 'later source mutation'
  assert frozen_values(current.to_dict()) == expected


def test_current_buttons_and_mads_change_while_radar_contents_repeat():
  control = controls()
  radar, state, _ = control_inputs(('leadOne', 'leadTwo'))
  reader = radar.as_reader()
  for phase in range(len(ICBM_STATES) * len(BUTTON_STATES)):
    state.mads.state = MADS_STATES[phase % len(MADS_STATES)]
    state.mads.enabled = bool(phase & 1)
    state.mads.active = bool(phase & 2)
    state.mads.available = bool(phase & 4)
    state.intelligentCruiseButtonManagement.state = ICBM_STATES[phase // len(BUTTON_STATES)]
    state.intelligentCruiseButtonManagement.sendButton = BUTTON_STATES[phase % len(BUTTON_STATES)]
    state.intelligentCruiseButtonManagement.vTarget = phase / 8.0
    compare(control, reader, state.as_reader())


def test_radar_selection_and_withdrawal_apply_immediately():
  control = controls()
  radar, state, _ = control_inputs(('leadOne', 'leadTwo'))
  for enabled in (True, False, True, False):
    control.radar_signal = enabled
    actual = compare(control, radar.as_reader(), state.as_reader())
    assert actual.leadOne.status == enabled
  control.radar_signal = True
  withdrawn = compare(control, log.RadarState.new_message().as_reader(), state.as_reader())
  assert not withdrawn.leadOne.status and not withdrawn.leadTwo.status


def test_unknown_source_and_schema_keep_direct_mapping():
  control = controls()
  radar, state, _ = control_inputs(('leadOne', 'leadTwo'))
  compare(control, radar.as_reader(), state.as_reader())
  sources = {}
  for name in ('leadOne', 'leadTwo'):
    source = getattr(radar, name)
    sources[name] = SimpleNamespace(**{field: getattr(source, field) for field in source.schema.fields})
  proxy = SimpleNamespace(**sources)
  for value in (35.0, 3.0, -0.0):
    proxy.leadOne.dRel = value
    compare(control, proxy, state.as_reader())
  with patch.object(controlsd_ext, '_RADAR_LEAD_SCHEMA', object()):
    compare(control, radar.as_reader(), state.as_reader())
    radar.leadOne.dRel = 4.0
    compare(control, radar.as_reader(), state.as_reader())


def test_stateful_mapper_override_is_called_for_each_current_output():
  control = controls()
  radar, state, _ = control_inputs(('leadOne', 'leadTwo'))
  count = 0

  def mapper(target, source):
    nonlocal count
    count += 1
    ControlsExt.get_lead_data(target, source)
    target.dRel = count

  control.get_lead_data = mapper
  for _ in range(3):
    actual = control.state_control_ext({'radarState': radar.as_reader(), 'selfdriveStateSP': state.as_reader()})
    assert actual.leadOne.dRel == count - 1 and actual.leadTwo.dRel == count
