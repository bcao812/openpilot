from unittest.mock import patch

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  CRUISE_COAST_ACCEL, CRUISE_COAST_LEAD_BUFFER, CRUISE_COAST_RISK_ACCEL,
  CRUISE_COAST_LEAD_TTC, CRUISE_COAST_MPC_MARGIN, cruise_coast_leads_clear,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  get_T_FOLLOW, get_safe_obstacle_distance, get_stopped_equivalence_factor,
)
from openpilot.selfdrive.controls.tests.test_longitudinal_cruise_coast import cruise_planner as base_cruise_planner, cruise_state, update_cruise


def cruise_planner():
  planner = base_cruise_planner()
  planner.experimental_coast_with_lead = True
  return planner


def distant_lead_state(lead_name='leadOne', distance=100.0):
  state = cruise_state()
  lead = getattr(state['radarState'], lead_name)
  lead.present = True
  lead.dRel = distance
  lead.vLead = state['carState'].vEgo
  lead.vRel = 0.0
  lead.aLeadK = 0.0
  lead.aLeadTau = 1.5
  return state


@pytest.mark.parametrize('lead_name', ['leadOne', 'leadTwo'])
def test_distant_nonbinding_lead_permits_cruise_shaping(lead_name):
  assert update_cruise(cruise_planner(), distant_lead_state(lead_name)) == CRUISE_COAST_ACCEL


@pytest.mark.parametrize('personality', list(log.LongitudinalPersonality.schema.enumerants))
@pytest.mark.parametrize('speed', [10.0, 20.0, 35.0])
def test_personality_distance_and_buffer_are_required(personality, speed):
  state = distant_lead_state()
  lead = state['radarState'].leadOne
  lead.vLead = speed
  enum_value = getattr(log.LongitudinalPersonality, personality)
  gap = get_safe_obstacle_distance(speed, get_T_FOLLOW(enum_value)) - get_stopped_equivalence_factor(speed)
  lead.dRel = gap + CRUISE_COAST_LEAD_BUFFER - 0.01
  assert not cruise_coast_leads_clear(state['radarState'], speed, enum_value)
  lead.dRel = gap + CRUISE_COAST_LEAD_BUFFER + 0.01
  assert cruise_coast_leads_clear(state['radarState'], speed, enum_value)


@pytest.mark.parametrize('lead_name', ['leadOne', 'leadTwo'])
@pytest.mark.parametrize('condition', ['close', 'closing', 'decelerating', 'model_stop', 'model_brake', 'mpc_margin', 'fcw'])
def test_risk_lead_bypasses_shaping(lead_name, condition):
  state = distant_lead_state(lead_name)
  lead = getattr(state['radarState'], lead_name)
  mpc = 1.0
  if condition == 'close':
    lead.dRel = 10.0
  elif condition == 'closing':
    lead.vRel = -lead.dRel / CRUISE_COAST_LEAD_TTC - 0.01
  elif condition == 'decelerating':
    lead.aLeadK = CRUISE_COAST_RISK_ACCEL - 0.01
  elif condition in ('model_stop', 'model_brake'):
    state['modelV2'] = state['modelV2'].as_builder()
    state['modelV2'].action.shouldStop = condition == 'model_stop'
    state['modelV2'].action.desiredAcceleration = -1.0 if condition == 'model_brake' else 0.1
  elif condition == 'mpc_margin':
    mpc = -0.23 + CRUISE_COAST_MPC_MARGIN - 0.01
  planner = cruise_planner()
  if condition == 'fcw':
    planner.mpc.crash_cnt = 5
    with patch.object(planner.mpc, 'update', return_value=None):
      assert update_cruise(planner, state, a_mpc=mpc) == -0.23
  else:
    assert update_cruise(planner, state, a_mpc=mpc) == -0.23


@pytest.mark.parametrize('field', ['dRel', 'vLead', 'vRel', 'aLeadK', 'aLeadTau'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_lead_is_not_eligible(field, value):
  state = distant_lead_state()
  setattr(state['radarState'].leadOne, field, value)
  assert not cruise_coast_leads_clear(state['radarState'], 20.2, state['selfdriveState'].personality)


def test_negative_lead_speed_is_not_eligible():
  state = distant_lead_state()
  state['radarState'].leadOne.vLead = -1
  assert not cruise_coast_leads_clear(state['radarState'], 20.2, state['selfdriveState'].personality)


@pytest.mark.parametrize('time_constant', [0.0, -1.0])
def test_nonpositive_lead_time_constant_is_not_eligible(time_constant):
  state = distant_lead_state()
  state['radarState'].leadOne.aLeadTau = time_constant
  assert not cruise_coast_leads_clear(state['radarState'], 20.2, state['selfdriveState'].personality)


def test_second_close_lead_vetoes_first_distant_lead():
  state = distant_lead_state()
  state['radarState'].leadTwo.present = True
  state['radarState'].leadTwo.dRel = 5.0
  assert update_cruise(cruise_planner(), state) == -0.23


def test_cut_in_brakes_on_the_first_frame():
  planner = cruise_planner()
  state = distant_lead_state()
  assert update_cruise(planner, state) == CRUISE_COAST_ACCEL
  state['radarState'].leadOne.dRel = 8.0
  assert update_cruise(planner, state, a_mpc=-2.5) == -2.5


def test_lead_presence_noise_does_not_toggle_qualified_cruise():
  planner = cruise_planner()
  state = distant_lead_state()
  outputs = []
  for index in range(100):
    state['radarState'].leadOne.present = bool(index % 2)
    outputs.append(update_cruise(planner, state))
  np.testing.assert_array_equal(outputs, [CRUISE_COAST_ACCEL] * 100)


def test_lead_extension_is_not_enabled_by_default():
  planner = base_cruise_planner()
  assert not planner.experimental_coast_with_lead
  assert update_cruise(planner, distant_lead_state()) == -0.23
