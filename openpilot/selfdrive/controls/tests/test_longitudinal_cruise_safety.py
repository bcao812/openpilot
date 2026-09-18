from unittest.mock import patch

import pytest

from openpilot.selfdrive.controls.tests.test_longitudinal_cruise_coast import (assert_invalid_motion_preserves_braking,
                                                                          cruise_planner, cruise_state, update_cruise)


@pytest.mark.parametrize('service', ['selfdriveState', 'carControl'])
def test_disabled_message_cannot_enable_shaping(service):
  state = cruise_state()
  if service == 'selfdriveState':
    state[service] = state[service].as_builder()
  state[service].enabled = False
  assert update_cruise(cruise_planner(), state) == -0.23


@pytest.mark.parametrize('service', ['carState', 'radarState', 'modelV2', 'controlsState', 'carControl', 'selfdriveState'])
def test_required_health_gate_includes_activation_messages(service):
  planner, state = cruise_planner(), cruise_state()
  previous_target = planner.output_a_target
  state.all_checks = lambda service_list=None: service not in (service_list or state)
  output = update_cruise(planner, state)
  if service == 'carState':
    assert_invalid_motion_preserves_braking(planner, state, previous_target)
  else:
    assert output == -0.23


@pytest.mark.parametrize('condition', ['fcw', 'model_stop', 'model_brake', 'model_nan', 'model_inf', 'mpc_inf'])
def test_risk_veto_is_not_dependent_on_lead_presence(condition):
  planner, state = cruise_planner(), cruise_state()
  if condition == 'fcw':
    planner.mpc.crash_cnt = 5
    with patch.object(planner.mpc, 'update', return_value=None):
      assert update_cruise(planner, state) == -0.23
  elif condition == 'mpc_inf':
    assert update_cruise(planner, state, a_mpc=float('inf')) == -0.23
  else:
    state['modelV2'] = state['modelV2'].as_builder()
    state['modelV2'].action.shouldStop = condition == 'model_stop'
    state['modelV2'].action.desiredAcceleration = {'model_stop': 0.1, 'model_brake': -0.6,
                                                  'model_nan': float('nan'), 'model_inf': float('inf')}[condition]
    assert update_cruise(planner, state) == -0.23


def test_manual_exit_and_reentry_have_no_coast_memory():
  planner, state = cruise_planner(), cruise_state()
  assert update_cruise(planner, state) == -0.19
  state['carState'].brakePressed = True
  assert update_cruise(planner, state, a_mpc=-2.0) == -2.0
  state['carState'].brakePressed = False
  assert update_cruise(planner, state, a_cruise=-0.4) == -0.4
