"""Exercise the complete extension assembly with real Cap'n Proto inputs."""

import math

import pytest

from openpilot.cereal import custom, log
from openpilot.selfdrive.car.tests.test_car_control_sp_conversion import lead_fixture
from openpilot.sunnypilot.selfdrive.controls.controlsd_ext import ControlsExt


MADS_STATES = tuple(custom.ModularAssistiveDrivingSystem.ModularAssistiveDrivingSystemState.schema.enumerants)
ICBM_STATES = tuple(custom.IntelligentCruiseButtonManagement.IntelligentCruiseButtonManagementState.schema.enumerants)
BUTTON_STATES = tuple(custom.IntelligentCruiseButtonManagement.SendButtonState.schema.enumerants)


def control_inputs(lead_names=(), phase=0, mads=None, icbm=None):
  radar = log.RadarState.new_message()
  expected_leads = {}
  for index, name in enumerate(('leadOne', 'leadTwo')):
    if name not in lead_names:
      # The production method explicitly copies the default evidence structure.
      expected_leads[name] = {'evidence': {}}
      continue
    expected_leads[name] = lead_fixture(index + 2 * phase)
    expected_leads[name]['status'] = True
    wire = dict(expected_leads[name])
    wire['present'] = wire.pop('status')
    wire['deprecated'] = {field: wire.pop(field) for field in ('aRel', 'dPath', 'vLat', 'fcw')}
    # This retired field deliberately has no live SP counterpart.
    wire['deprecated']['aLead'] = 123.5
    setattr(radar, name, wire)
  state = custom.SelfdriveStateSP.new_message(mads=mads or {}, intelligentCruiseButtonManagement=icbm or {})
  expected = custom.CarControlSP.new_message(**expected_leads, mads=mads or {}, intelligentCruiseButtonManagement=icbm or {})
  return radar, state, expected


def test_default_state_control_ext_copies_wire_defaults():
  control = ControlsExt.__new__(ControlsExt)  # Assembly has no Params or transport dependency.
  control.radar_signal = True
  radar, state, expected = control_inputs()
  actual = control.state_control_ext({'radarState': radar.as_reader(), 'selfdriveStateSP': state.as_reader()})
  assert actual.to_dict() == expected.to_dict()
  assert not actual.leadOne.status and not actual.leadTwo.status


@pytest.mark.parametrize('lead_names', [(), ('leadOne',), ('leadTwo',), ('leadOne', 'leadTwo')],
                         ids=['no-lead', 'first-lead', 'second-lead', 'two-leads'])
def test_state_control_ext_reads_current_leads_and_all_sp_states(lead_names):
  control = ControlsExt.__new__(ControlsExt)
  control.radar_signal = True
  previous = None
  previous_fields = None
  targets = (0.0, -0.0, -12.5, 255.75, 3.4028234663852886e38)
  for phase in range(max(len(MADS_STATES), len(ICBM_STATES)) * len(BUTTON_STATES)):
    mads = {'state': MADS_STATES[phase % len(MADS_STATES)],
            'enabled': bool(phase & 1), 'active': bool(phase & 2), 'available': bool(phase & 4)}
    icbm = {'state': ICBM_STATES[phase // len(BUTTON_STATES)], 'sendButton': BUTTON_STATES[phase % len(BUTTON_STATES)],
            'vTarget': targets[phase % len(targets)]}
    radar, state, expected = control_inputs(lead_names, phase, mads, icbm)
    before = radar.to_dict(), state.to_dict()
    actual = control.state_control_ext({'radarState': radar.as_reader(), 'selfdriveStateSP': state.as_reader()})
    assert actual.to_dict() == expected.to_dict()
    assert math.copysign(1.0, actual.intelligentCruiseButtonManagement.vTarget) == math.copysign(1.0, icbm['vTarget'])
    assert (radar.to_dict(), state.to_dict()) == before
    if previous is not None:
      assert previous.to_dict() == previous_fields
    previous, previous_fields = actual, actual.to_dict()


def test_withdrawal_and_input_mutation_leave_previous_output_independent():
  control = ControlsExt.__new__(ControlsExt)
  control.radar_signal = True
  radar, state, expected = control_inputs(('leadOne', 'leadTwo'), mads={
    'state': 'enabled', 'enabled': True, 'active': True, 'available': True,
  }, icbm={'state': 'increasing', 'sendButton': 'increase', 'vTarget': 31.25})
  actual = control.state_control_ext({'radarState': radar.as_reader(), 'selfdriveStateSP': state.as_reader()})
  saved = expected.to_dict()
  radar.leadOne.evidence.reason = 'changed-after-assembly'
  radar.leadTwo.dRel = 1.0
  state.mads.active = False
  state.intelligentCruiseButtonManagement.vTarget = 0.0
  empty_radar, empty_state, empty_expected = control_inputs()
  withdrawn = control.state_control_ext({'radarState': empty_radar.as_reader(), 'selfdriveStateSP': empty_state.as_reader()})
  assert withdrawn.to_dict() == empty_expected.to_dict()
  assert actual.to_dict() == saved
