from copy import deepcopy
from unittest.mock import patch

import pytest

from opendbc.car.honda.values import CAR
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import CREEP_STOP_T, STOP_RELEASE_ACCEL, STOP_SPEED
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_planner import has_safe_moving_lead
from openpilot.selfdrive.controls.tests.test_longitudinal_cruise_coast import update_cruise
from openpilot.selfdrive.controls.tests.test_longitudinal_planner_creep import FRAMES_CAP
from openpilot.selfdrive.controls.tests.test_longitudinal_planner_state import braking_planner, braking_state


def following_state(v_ego=1.0):
  state = braking_state(v_ego)
  state['carControl'].longActive = True
  state['modelV2'] = state['modelV2'].as_builder()
  state['modelV2'].action.desiredAcceleration = -0.2
  for lead in (state['radarState'].leadOne, state['radarState'].leadTwo):
    lead.present = True
    lead.dRel = 7.0
    lead.vLead = 0.8
    lead.vRel = 0.8 - v_ego
    lead.aLeadK = 0.0
    lead.modelProb = 0.999
  return state


def follow(planner, state, frames=2 * FRAMES_CAP, **kwargs):
  for _ in range(frames):
    update_cruise(planner, state, a_cruise=1.0, a_mpc=kwargs.get('a_mpc', -0.2), e2e=kwargs.get('e2e', False))


@pytest.mark.parametrize('e2e', [False, True])
def test_sustained_moving_lead_does_not_create_an_open_loop_stop(e2e):
  planner, state = braking_planner(), following_state()
  before = deepcopy(state['radarState'].to_dict())
  follow(planner, state, e2e=e2e)
  assert planner.moving_lead_t == CREEP_STOP_T
  assert planner.creep_total_t == 0.0
  assert planner.output_a_target == pytest.approx(-0.2)
  assert not planner.output_should_stop
  assert state['radarState'].to_dict() == before


def test_logged_moving_lead_approach_does_not_force_a_stop():
  """2026-09-10 route 5a, before the driver-assisted launch at t=2202.815."""
  planner, state = braking_planner(), following_state(1.18)
  for lead in (state['radarState'].leadOne, state['radarState'].leadTwo):
    lead.dRel = 6.1
    lead.vLead = 0.95
    lead.vRel = -0.23
    lead.aLeadK = -0.25
  state['modelV2'].action.desiredAcceleration = -0.27
  follow(planner, state, e2e=True)
  assert planner.output_a_target == pytest.approx(-0.27)
  assert not planner.creep_stop
  assert not planner.output_should_stop


def test_confirmation_is_required_before_suppressing_creep():
  planner, state = braking_planner(), following_state()
  follow(planner, state, frames=round(CREEP_STOP_T / DT_MDL) - 1)
  assert 0.0 < planner.moving_lead_t < CREEP_STOP_T
  assert planner.creep_total_t > 0.0
  follow(planner, state, frames=2)
  assert planner.moving_lead_t == CREEP_STOP_T
  assert planner.creep_total_t == 0.0


@pytest.mark.parametrize('lead_name', ['leadOne', 'leadTwo'])
@pytest.mark.parametrize(
  'field,value',
  [
    ('dRel', 5.0),
    ('vLead', 0.0),
    ('vRel', -0.6),
    ('aLeadK', -1.0),
    ('modelProb', 0.5),
    ('modelProb', 1.1),
    ('dRel', float('nan')),
    ('vRel', float('inf')),
    ('vLead', float('nan')),
    ('aLeadK', float('-inf')),
    ('modelProb', float('nan')),
  ],
)
def test_any_unsafe_or_uncertain_lead_keeps_original_creep_fallback(lead_name, field, value):
  planner, state = braking_planner(), following_state()
  setattr(getattr(state['radarState'], lead_name), field, value)
  assert not has_safe_moving_lead(state['radarState'], 1.0, 0.5)
  with patch.object(planner.mpc, 'update', return_value=None):
    follow(planner, state)
  assert planner.moving_lead_t == 0.0
  assert planner.output_should_stop


@pytest.mark.parametrize(
  'condition',
  [
    'no_lead',
    'only_second_lead',
    'gas',
    'brake',
    'override',
    'long_inactive',
    'disabled',
    'control_disabled',
    'standstill',
    'force_decel',
    'model_stop',
    'model_brake',
    'model_nan',
    'unhealthy',
    'stock_long',
  ],
)
def test_following_bypass_requires_all_activation_and_risk_guards(condition):
  planner, state = braking_planner(), following_state()
  if condition in ('no_lead', 'only_second_lead'):
    state['radarState'].leadOne.present = False
    state['radarState'].leadTwo.present = condition == 'only_second_lead'
  elif condition in ('gas', 'brake', 'standstill'):
    setattr(state['carState'], {'gas': 'gasPressed', 'brake': 'brakePressed', 'standstill': 'standstill'}[condition], True)
  elif condition == 'override':
    state['carControl'].cruiseControl.override = True
  elif condition == 'long_inactive':
    state['carControl'].longActive = False
  elif condition == 'disabled':
    state['selfdriveState'] = state['selfdriveState'].as_builder()
    state['selfdriveState'].enabled = False
  elif condition == 'control_disabled':
    state['carControl'].enabled = False
  elif condition == 'force_decel':
    state['controlsState'].forceDecel = True
  elif condition == 'model_stop':
    state['modelV2'].action.shouldStop = True
  elif condition in ('model_brake', 'model_nan'):
    state['modelV2'].action.desiredAcceleration = -0.6 if condition == 'model_brake' else float('nan')
  elif condition == 'unhealthy':
    state.all_checks = lambda service_list=None: False
  elif condition == 'stock_long':
    planner.CP.openpilotLongitudinalControl = False
  follow(planner, state)
  assert planner.moving_lead_t == 0.0
  assert planner.output_should_stop


@pytest.mark.parametrize('service', ['carState', 'radarState', 'modelV2', 'controlsState', 'carControl', 'selfdriveState'])
def test_stale_input_discards_confirmation_immediately(service):
  planner, state = braking_planner(), following_state()
  follow(planner, state)
  state.all_checks = lambda service_list=None: service not in (service_list or state)
  follow(planner, state, frames=1)
  assert planner.moving_lead_t == 0.0


def test_flickering_lead_cannot_accumulate_confirmation():
  planner, state = braking_planner(), following_state()
  for frame in range(2 * FRAMES_CAP):
    state['radarState'].leadOne.present = frame % 4 != 0
    follow(planner, state, frames=1)
    assert planner.moving_lead_t <= 3 * DT_MDL
  assert planner.output_should_stop


def test_a_new_stationary_lead_restores_the_fallback():
  planner, state = braking_planner(), following_state()
  follow(planner, state)
  state['radarState'].leadOne.vLead = 0.0
  state['radarState'].leadOne.vRel = -1.0
  follow(planner, state, frames=FRAMES_CAP, a_mpc=-0.8)
  assert planner.output_should_stop
  assert planner.output_a_target == pytest.approx(-0.8)


def test_stronger_braking_is_never_delayed_by_following():
  planner, state = braking_planner(), following_state()
  follow(planner, state)
  follow(planner, state, frames=1, a_mpc=-2.0)
  assert planner.output_a_target == -2.0
  assert planner.moving_lead_t == 0.0


@pytest.mark.parametrize('mpc_accel', [0.4, 0.0, -0.03, float('inf')])
def test_model_only_braking_intent_keeps_the_original_creep_fallback(mpc_accel):
  planner, state = braking_planner(), following_state()
  state['modelV2'].action.desiredAcceleration = -0.25
  follow(planner, state, a_mpc=mpc_accel, e2e=True)
  assert planner.output_a_target == pytest.approx(-0.25)
  assert planner.moving_lead_t == 0.0
  assert planner.output_should_stop


def test_a_model_stop_intent_is_not_hidden_by_an_unconstrained_moving_lead():
  planner, state = braking_planner(), following_state()
  follow(planner, state)
  state['modelV2'].action.desiredAcceleration = -0.25
  follow(planner, state, a_mpc=0.2, e2e=True, frames=1)
  assert planner.moving_lead_t == 0.0
  follow(planner, state, a_mpc=0.2, e2e=True)
  assert planner.output_should_stop


def test_fcw_discards_confirmation_immediately():
  planner, state = braking_planner(), following_state()
  follow(planner, state)
  planner.mpc.crash_cnt = 5
  with patch.object(planner.mpc, 'update', return_value=None):
    follow(planner, state, frames=1)
  assert planner.fcw
  assert planner.moving_lead_t == 0.0


def test_an_existing_creep_stop_cannot_be_released_by_a_moving_lead():
  planner, state = braking_planner(), following_state()
  planner.creep_stop = True
  follow(planner, state)
  assert planner.creep_stop
  assert planner.output_should_stop


def test_a_stopped_car_retains_the_original_release_threshold():
  planner, state = braking_planner(), following_state(0.0)
  planner.stop_latched = True
  state['carState'].standstill = True
  state['controlsState'].longControlState = LongCtrlState.stopping
  state['modelV2'].action.desiredAcceleration = 0.4
  follow(planner, state, a_mpc=0.15)
  assert 0.1 < planner.output_a_target < STOP_RELEASE_ACCEL
  assert planner.stop_latched
  assert planner.output_should_stop
  assert planner.moving_lead_t == 0.0


def test_first_active_frame_confirms_before_the_fallback_deadline():
  planner, state = braking_planner(), following_state()
  state['controlsState'].longControlState = LongCtrlState.off
  follow(planner, state, frames=1)
  assert planner.moving_lead_t == DT_MDL
  state['controlsState'].longControlState = LongCtrlState.pid
  for _ in range(FRAMES_CAP):
    follow(planner, state, frames=1)
    assert not planner.output_should_stop


def test_off_state_cannot_accumulate_or_retain_confirmation():
  planner, state = braking_planner(), following_state()
  follow(planner, state)
  state['controlsState'].longControlState = LongCtrlState.off
  for _ in range(FRAMES_CAP):
    follow(planner, state, frames=1)
    assert planner.moving_lead_t <= DT_MDL


@pytest.mark.parametrize('platform', [CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_HYBRID])
def test_other_platforms_retain_the_original_fallback(platform):
  planner, state = braking_planner(platform), following_state()
  follow(planner, state)
  assert planner.output_should_stop
  assert planner.moving_lead_t == 0.0


@pytest.mark.parametrize('v_ego', [float('nan'), float('inf'), -0.1, 0.0, STOP_SPEED - 0.001, 2.0, 20.0])
def test_moving_lead_guard_is_limited_to_rolling_low_speed(v_ego):
  state = following_state()
  assert not has_safe_moving_lead(state['radarState'], v_ego, 0.5)


@pytest.mark.parametrize('delay', [-0.1, 1.01, float('nan'), float('inf')])
def test_unknown_or_excessive_actuator_delay_cannot_relax_the_fallback(delay):
  state = following_state()
  assert not has_safe_moving_lead(state['radarState'], 1.0, delay)
