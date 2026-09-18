from copy import deepcopy

import numpy as np
import pytest

from opendbc.car.honda.values import CAR
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_planner import constrain_lead_braking_speed, has_selected_lead_braking_risk
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


def closing_planner(platform=CAR.HONDA_ACCORD):
  planner = braking_planner(platform)
  planner.v_desired_filter.x = 19.0
  planner.output_a_target = -1.0
  return planner


def test_closing_below_one_second_does_not_hide_measured_speed():
  state = closing_state()
  assert constrain_lead_braking_speed(19.0, 20.0, -1.0, state['radarState']) == pytest.approx(19.8)


@pytest.mark.parametrize('v_ego,gap,v_rel,expected', [(20.0, 30.0, -3.0, 19.4), (20.0, 20.0, -0.5, 19.4), (7.5, 7.5, -2.0, 6.9)])
def test_feedback_tapers_with_gap_closing_and_speed(v_ego, gap, v_rel, expected):
  state = closing_state(v_ego, gap, v_rel)
  assert constrain_lead_braking_speed(v_ego - 1.0, v_ego, -1.0, state['radarState']) == pytest.approx(expected)


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
  assert constrain_lead_braking_speed(19.0, 20.0, -1.0, state['radarState']) == 19.0


def test_second_lead_has_the_same_safety_priority():
  state = closing_state()
  state['radarState'].leadTwo = state['radarState'].leadOne.as_reader()
  state['radarState'].leadOne.present = False
  assert constrain_lead_braking_speed(19.0, 20.0, -1.0, state['radarState']) == pytest.approx(19.8)


@pytest.mark.parametrize('a_target', [-0.5, -0.3, -0.2, -0.1, 0.0, 0.1, 2.0, float('inf'), float('nan')])
def test_no_state_correction_while_accelerating_or_unknown(a_target):
  assert constrain_lead_braking_speed(19.0, 20.0, a_target, closing_state()['radarState']) == 19.0


@pytest.mark.parametrize('v_ego', [0.0, 0.3, 1.2, 2.0, 5.0])
def test_low_speed_braking_work_is_unchanged(v_ego):
  state = closing_state(v_ego, 2.0, -1.0)
  assert constrain_lead_braking_speed(0.0, v_ego, -1.0, state['radarState']) == 0.0


@pytest.mark.parametrize('v_desired', [19.8, 20.0, 21.0])
def test_feedback_never_pushes_an_adequate_state_higher(v_desired):
  assert constrain_lead_braking_speed(v_desired, 20.0, -1.0, closing_state()['radarState']) == v_desired


def test_guarded_feedback_is_continuous_at_all_boundaries():
  for accel in (-1.0, -0.5):
    before = constrain_lead_braking_speed(19.0, 20.0, accel - 1e-7, closing_state()['radarState'])
    after = constrain_lead_braking_speed(19.0, 20.0, accel + 1e-7, closing_state()['radarState'])
    assert abs(after - before) < 1e-5
  for speed in (5.0, 10.0):
    before = constrain_lead_braking_speed(0.0, speed - 1e-7, -1.0, closing_state(speed, speed / 2)['radarState'])
    after = constrain_lead_braking_speed(0.0, speed + 1e-7, -1.0, closing_state(speed, speed / 2)['radarState'])
    assert abs(after - before) < 1e-5
  for gap in (20.0, 40.0):
    before = constrain_lead_braking_speed(19.0, 20.0, -1.0, closing_state(gap=gap - 1e-7)['radarState'])
    after = constrain_lead_braking_speed(19.0, 20.0, -1.0, closing_state(gap=gap + 1e-7)['radarState'])
    assert abs(after - before) < 1e-5
  for closing in (-1.0, 0.0):
    before = constrain_lead_braking_speed(19.0, 20.0, -1.0, closing_state(v_rel=closing - 1e-7)['radarState'])
    after = constrain_lead_braking_speed(19.0, 20.0, -1.0, closing_state(v_rel=closing + 1e-7)['radarState'])
    assert abs(after - before) < 1e-5


def test_real_mpc_receives_bounded_speed_without_rewriting_the_lead():
  planner, state = closing_planner(), closing_state()
  before = deepcopy(state['radarState'].to_dict())
  planner.update(state)
  assert planner.mpc.solution_status == 0
  assert planner.mpc.x0[1] == pytest.approx(19.8)
  assert state['radarState'].to_dict() == before


def delayed_braking_state():
  state = closing_state(20.0, 70.0, -12.0)
  state['carState'].aEgo = -0.65
  lead = state['radarState'].leadOne
  lead.aLeadK, lead.aLeadTau, lead.modelProb = -3.0, 1.5, 0.0
  return state


def selected_risk(state, source=LongitudinalPlanSource.lead0, target=-1.6):
  return has_selected_lead_braking_risk(state['carState'].vEgo, state['carState'].aEgo, target,
                                        state['radarState'], source, 0.5)


@pytest.mark.parametrize('source', [LongitudinalPlanSource.cruise, LongitudinalPlanSource.e2e])
def test_unselected_lead_does_not_replace_the_current_state(source):
  assert not selected_risk(delayed_braking_state(), source=source)


@pytest.mark.parametrize('field,value', [
  ('present', False), ('dRel', 0.0), ('dRel', -1.0), ('dRel', float('nan')),
  ('vRel', 0.0), ('vRel', float('nan')), ('vLead', -0.6), ('vLead', 25.0),
  ('aLeadK', float('nan')), ('aLeadTau', -0.1), ('aLeadTau', float('nan')),
  ('modelProb', -0.1), ('modelProb', 1.01), ('modelProb', float('nan')),
  ('yRel', 2.0), ('yRel', float('nan')),
])
def test_invalid_inconsistent_or_adjacent_selected_lead_cannot_anchor_state(field, value):
  state = delayed_braking_state()
  setattr(state['radarState'].leadOne, field, value)
  assert not selected_risk(state)


@pytest.mark.parametrize('target', [-0.5, -0.2, 0.0, 0.5, float('nan')])
def test_a_lead_must_already_have_supplied_a_braking_request(target):
  assert not selected_risk(delayed_braking_state(), target=target)


@pytest.mark.parametrize('acceleration', [-4.0, 2.1, float('nan'), float('inf')])
def test_unknown_or_out_of_model_measured_acceleration_does_not_anchor_state(acceleration):
  state = delayed_braking_state()
  state['carState'].aEgo = acceleration
  assert not selected_risk(state)


def test_distant_mild_closure_keeps_the_existing_state():
  assert not selected_risk(closing_state(20.0, 150.0, -3.0))


@pytest.mark.parametrize('lead_slot,source', [('leadOne', LongitudinalPlanSource.lead0), ('leadTwo', LongitudinalPlanSource.lead1)])
@pytest.mark.parametrize('probability', [0.0, 0.999])
def test_selected_lead_uses_measured_initial_motion_without_erasing_native_history(lead_slot, source, probability):
  planner, state = closing_planner(), delayed_braking_state()
  state['radarState'].leadOne.modelProb = probability
  if lead_slot == 'leadTwo':
    state['radarState'].leadTwo = state['radarState'].leadOne.as_reader()
    state['radarState'].leadOne.present = False
  planner.output_a_target = -1.6
  planner.update(state)
  assert planner.mpc.solution_status == 0 and planner.mpc.source == source
  assert planner.output_a_target < -0.5
  accepted_history = planner.mpc.a_prev.copy()
  before_leads = deepcopy(state['radarState'].to_dict())
  planner.update(state)
  assert planner.mpc.solution_status == 0
  assert planner.mpc.x0[1] == state['carState'].vEgo
  assert planner.mpc.x0[2] == state['carState'].aEgo
  np.testing.assert_array_equal(planner.mpc.params[:, 3], accepted_history)
  assert state['radarState'].to_dict() == before_leads


@pytest.mark.parametrize('condition', [
  'gas', 'brake', 'override', 'inactive', 'disabled', 'control_disabled', 'force_decel', 'unhealthy', 'stock_long', 'other_platform',
  'solver_failed',
])
def test_state_anchoring_keeps_driver_mode_and_health_guards_after_a_real_lead_solution(condition):
  planner, state = closing_planner(), delayed_braking_state()
  planner.output_a_target = -1.6
  planner.update(state)
  assert planner.mpc.source == LongitudinalPlanSource.lead0
  state['carState'].aEgo = 1.5
  if condition in ('gas', 'brake'):
    setattr(state['carState'], condition + 'Pressed', True)
  elif condition == 'override':
    state['carControl'].cruiseControl.override = True
  elif condition == 'inactive':
    state['carControl'].longActive = False
  elif condition == 'disabled':
    state['selfdriveState'] = state['selfdriveState'].as_builder()
    state['selfdriveState'].enabled = False
  elif condition == 'control_disabled':
    state['carControl'].enabled = False
  elif condition == 'force_decel':
    state['controlsState'].forceDecel = True
  elif condition == 'unhealthy':
    state.all_checks = lambda service_list=None: False
  elif condition == 'stock_long':
    planner.CP.openpilotLongitudinalControl = False
  elif condition == 'other_platform':
    planner.CP.carFingerprint = CAR.HONDA_CIVIC_BOSCH
  elif condition == 'solver_failed':
    planner.mpc.solution_status = 1
  planner.update(state)
  assert planner.mpc.x0[2] != state['carState'].aEgo


@pytest.mark.parametrize(
  'condition', ['gas', 'brake', 'override', 'inactive', 'disabled', 'control_disabled', 'force_decel', 'unhealthy', 'stock_long', 'other_platform']
)
def test_driver_mode_and_health_guards_preserve_the_existing_state(condition):
  planner, state = closing_planner(), closing_state()
  if condition in ('gas', 'brake'):
    setattr(state['carState'], condition + 'Pressed', True)
  elif condition == 'override':
    state['carControl'].cruiseControl.override = True
  elif condition == 'inactive':
    state['carControl'].longActive = False
  elif condition == 'disabled':
    state['selfdriveState'] = state['selfdriveState'].as_builder()
    state['selfdriveState'].enabled = False
  elif condition == 'control_disabled':
    state['carControl'].enabled = False
  elif condition == 'force_decel':
    state['controlsState'].forceDecel = True
  elif condition == 'unhealthy':
    state.all_checks = lambda service_list=None: False
  elif condition == 'stock_long':
    planner.CP.openpilotLongitudinalControl = False
  elif condition == 'other_platform':
    planner.CP.carFingerprint = CAR.HONDA_CIVIC_BOSCH
  planner.update(state)
  assert planner.mpc.x0[1] == pytest.approx(19.0 + DT_MDL / (2.0 + DT_MDL))


def test_critical_braking_feedback_stays_bounded_during_a_persistent_shortfall():
  planner, state = closing_planner(), closing_state()
  for _ in range(80):
    planner.output_a_target = -1.0
    planner.update(state)
    assert planner.mpc.solution_status == 0
    assert planner.mpc.x0[1] >= 19.8 - 1e-6
    assert np.isfinite(planner.output_a_target)
    assert -3.5 <= planner.output_a_target <= 2.0


def test_braking_confirmation_tapers_without_perturbing_the_pedal_boundary():
  state = closing_state()
  assert constrain_lead_braking_speed(19.0, 20.0, -0.75, state['radarState']) == pytest.approx(19.4)
  for accel in np.linspace(-0.5, 0.3, 100):
    assert constrain_lead_braking_speed(19.0, 20.0, float(accel), state['radarState']) == 19.0


@pytest.mark.parametrize('service', ['carState', 'radarState', 'modelV2', 'controlsState', 'carControl', 'selfdriveState'])
def test_each_required_service_can_veto_the_correction(service):
  planner, state = closing_planner(), closing_state()
  state.all_checks = lambda service_list=None: service not in (service_list or state)
  planner.update(state)
  assert planner.mpc.x0[1] == pytest.approx(19.0 + DT_MDL / (2.0 + DT_MDL))


def test_reentry_uses_the_existing_measured_speed_reset():
  planner, state = closing_planner(), closing_state()
  state['controlsState'].longControlState = 'off'
  planner.update(state)
  assert planner.mpc.x0[1] == 20.0


@pytest.mark.parametrize('speed', [15.0, 20.0, 27.8])
def test_closing_stop_keeps_clearance_at_the_selected_delay_stress_point(speed):
  from openpilot.selfdrive.controls.tests.test_longitudinal_following_scenarios import simulate_following

  result = simulate_following(scenario='lead_stop', initial_speed=speed, lead_speed=speed, initial_gap=1.5 * speed, delay=0.5, response_time=0.4)
  assert result['minimum_gap'] > 2.0
  assert result['final_speed'] == 0.0
  assert result['minimum_command'] >= -3.5
  assert result['conflicts'] == 0
