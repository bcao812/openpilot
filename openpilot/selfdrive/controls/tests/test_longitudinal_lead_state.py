"""Actual motion estimates and closing-lead risk after retiring reference-speed correction."""

from copy import deepcopy

import numpy as np
import pytest

from opendbc.car.honda.values import CAR
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import advance
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_longitudinal_planner_state import braking_planner, braking_state


def closing_state(v_ego=20.0, gap=18.0, v_rel=-3.0):
  state = braking_state(v_ego)
  state['carControl'].longActive = True
  lead = state['radarState'].leadOne
  lead.present = True
  lead.dRel = gap
  lead.vRel = v_rel
  lead.vLead = v_ego + v_rel
  lead.aLeadK = 0.0
  lead.aLeadTau = 0.3
  lead.modelProb = 0.999
  state['radarState'].leadTwo.present = False
  return state


def closing_planner(platform=CAR.HONDA_ACCORD, *, openpilot_long=True):
  planner = braking_planner(platform, openpilot_long=openpilot_long)
  planner.v_desired_filter.x = 19.0
  planner.output_a_target = -1.0
  return planner


@pytest.mark.parametrize('v_ego,gap,v_rel', [(20.0, 18.0, -3.0), (20.0, 30.0, -3.0), (20.0, 20.0, -0.5), (7.5, 7.5, -2.0)])
@pytest.mark.parametrize('lead_slot', ['leadOne', 'leadTwo'])
def test_closing_lead_geometry_cannot_replace_the_measured_speed(v_ego, gap, v_rel, lead_slot):
  planner, state = closing_planner(), closing_state(v_ego, gap, v_rel)
  if lead_slot == 'leadTwo':
    state['radarState'].leadTwo = state['radarState'].leadOne.as_reader()
    state['radarState'].leadOne.present = False
  before = deepcopy(state['radarState'].to_dict())
  planner.update(state, now_ns=advance(state, 0))
  assert planner.mpc.solution_status == 0
  assert planner.mpc.x0[1] == pytest.approx(v_ego)
  assert state['radarState'].to_dict() == before


@pytest.mark.parametrize(
  'condition',
  ['no_lead', 'distant', 'opening', 'matching_speed', 'low_probability', 'bad_probability', 'invalid_distance', 'invalid_velocity', 'nan_probability'],
)
def test_unconstrained_or_uncertain_leads_do_not_change_the_state(condition):
  state = closing_state()
  lead = state['radarState'].leadOne
  if condition == 'no_lead':
    lead.present = False
  elif condition == 'distant':
    lead.dRel = 41.0
  elif condition in ('opening', 'matching_speed', 'invalid_velocity'):
    lead.vRel = {'opening': 1.0, 'matching_speed': 0.0, 'invalid_velocity': float('nan')}[condition]
  elif condition == 'invalid_distance':
    lead.dRel = float('nan')
  else:
    lead.modelProb = {'low_probability': 0.5, 'bad_probability': 1.1, 'nan_probability': float('nan')}[condition]
  planner = closing_planner()
  planner.update(state, now_ns=advance(state, 0))
  if planner.mpc.solution_status == 0:
    assert planner.mpc.x0[1] == state['carState'].vEgo
  else:
    assert not planner.motion_intent.valid
    assert np.isfinite(planner.output_a_target) and planner.output_a_target <= -0.2


def test_real_mpc_receives_bounded_speed_without_rewriting_the_lead():
  planner, state = closing_planner(), closing_state()
  before = deepcopy(state['radarState'].to_dict())
  planner.update(state, now_ns=advance(state, 0))
  assert planner.mpc.solution_status == 0
  assert planner.mpc.x0[1] == state['carState'].vEgo
  assert state['radarState'].to_dict() == before


@pytest.mark.parametrize(
  'condition', ['gas', 'brake', 'override', 'inactive', 'disabled', 'control_disabled', 'force_decel', 'unhealthy', 'stock_long', 'other_platform']
)
def test_driver_mode_and_health_guards_preserve_the_existing_state(condition):
  platform = CAR.HONDA_CIVIC_BOSCH if condition == 'other_platform' else CAR.HONDA_ACCORD
  planner, state = closing_planner(platform, openpilot_long=condition != 'stock_long'), closing_state()
  if condition in ('gas', 'brake'):
    setattr(state['carState'], condition + 'Pressed', True)
  elif condition == 'override':
    state['carControl'].cruiseControl.override = True
  elif condition == 'inactive':
    state['carControl'].longActive = False
  elif condition == 'disabled':
    state['selfdriveState'].enabled = False
  elif condition == 'control_disabled':
    state['carControl'].enabled = False
  elif condition == 'force_decel':
    state['controlsState'].forceDecel = True
  elif condition == 'unhealthy':
    state.all_checks = lambda service_list=None: False
  planner.update(state, now_ns=advance(state, 0))
  if condition == 'unhealthy':
    assert not planner.motion_intent.valid
  elif condition in ('stock_long', 'other_platform'):
    assert planner.mpc.x0[1] == pytest.approx(19.0 + DT_MDL / (2.0 + DT_MDL))
  else:
    assert planner.mpc.x0[1] == state['carState'].vEgo


def test_critical_braking_feedback_stays_bounded_during_a_persistent_shortfall():
  planner, state = closing_planner(), closing_state()
  for frame in range(80):
    planner.output_a_target = -1.0
    planner.update(state, now_ns=advance(state, frame))
    assert planner.mpc.solution_status == 0
    assert planner.mpc.x0[1] >= 19.8 - 1e-6
    assert np.isfinite(planner.output_a_target)
    assert -3.5 <= planner.output_a_target <= 2.0


def test_reentry_uses_the_existing_measured_speed_reset():
  planner, state = closing_planner(), closing_state()
  state['controlsState'].longControlState = 'off'
  planner.update(state, now_ns=advance(state, 0))
  assert planner.mpc.x0[1] == 20.0


@pytest.mark.parametrize('speed', [15.0, 20.0, 27.8])
def test_closing_stop_keeps_clearance_at_the_selected_delay_stress_point(speed):
  from openpilot.selfdrive.controls.tests.test_longitudinal_following_scenarios import simulate_following

  result = simulate_following(scenario='lead_stop', initial_speed=speed, lead_speed=speed, initial_gap=1.5 * speed, delay=0.5, response_time=0.4)
  assert result['minimum_gap'] > 2.0
  assert result['final_speed'] == 0.0
  assert result['minimum_command'] >= -3.5
  assert result['conflicts'] == 0
