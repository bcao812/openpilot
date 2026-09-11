from copy import deepcopy
from unittest.mock import patch

import pytest

from opendbc.car import structs
from opendbc.car.honda.values import CAR
from openpilot.selfdrive.controls.lib.drive_helpers import STOP_RELEASE_ACCEL
from openpilot.selfdrive.controls.lib.longcontrol import ROLLING_STOP_ACCEL
from openpilot.selfdrive.controls.lib.longitudinal_planner import should_stop_for_close_lead
from openpilot.selfdrive.controls.tests.test_longcontrol_accel_release import controller
from openpilot.selfdrive.controls.tests.test_longitudinal_cruise_coast import update_cruise
from openpilot.selfdrive.controls.tests.test_longitudinal_lead_state import closing_state
from openpilot.selfdrive.controls.tests.test_longitudinal_planner_state import braking_planner


def near_stop_state():
  state = closing_state(1.3, 3.0, -1.2)
  state['radarState'].leadOne.aLeadK = -0.03
  state['radarState'].leadOne.modelProb = 0.999
  return state


def close_stop(state, speed=1.3, acceleration=-0.7, delay=0.5):
  return should_stop_for_close_lead(speed, acceleration, state['radarState'], delay)


def update(planner, state, acceleration=-0.7):
  update_cruise(planner, state, a_cruise=1.0, a_mpc=acceleration, e2e=False)


def test_nearly_stopped_lead_does_not_wait_for_the_fixed_creep_speed():
  state = near_stop_state()
  assert close_stop(state)
  planner = braking_planner()
  update(planner, state)
  assert planner.close_lead_stop
  assert planner.output_should_stop
  assert not planner.creep_stop


def test_distant_stopped_lead_keeps_the_normal_approach():
  state = near_stop_state()
  state['radarState'].leadOne.dRel = 7.0
  assert not close_stop(state)
  planner = braking_planner()
  update(planner, state)
  assert not planner.close_lead_stop
  assert not planner.output_should_stop


def test_stopping_budget_accounts_for_actuator_and_ramp_delay():
  state = near_stop_state()
  state['radarState'].leadOne.dRel = 3.3
  assert not close_stop(state, acceleration=-1.4, delay=0.0)
  assert close_stop(state, acceleration=-0.3, delay=0.5)


def test_both_lead_slots_are_supported_without_modifying_perception():
  state = near_stop_state()
  state['radarState'].leadTwo = state['radarState'].leadOne.as_reader()
  state['radarState'].leadOne.present = False
  before = deepcopy(state['radarState'].to_dict())
  assert close_stop(state)
  assert before == state['radarState'].to_dict()


@pytest.mark.parametrize('speed', [0.0, 0.29, 2.0, 10.0, float('nan'), float('inf')])
def test_speed_band_is_bounded(speed):
  assert not close_stop(near_stop_state(), speed=speed)


@pytest.mark.parametrize('acceleration', [ROLLING_STOP_ACCEL, -2.0, -3.5, 0.0, 0.1, float('nan'), float('inf')])
def test_launch_and_already_stronger_braking_do_not_enter_fallback(acceleration):
  assert not close_stop(near_stop_state(), acceleration=acceleration)


@pytest.mark.parametrize('delay', [-0.1, 1.1, float('nan'), float('inf')])
def test_invalid_delay_is_not_a_stopping_request(delay):
  assert not close_stop(near_stop_state(), delay=delay)


@pytest.mark.parametrize(
  'field,value',
  [
    ('dRel', 0.0),
    ('dRel', -0.5),
    ('dRel', float('nan')),
    ('vRel', -0.1),
    ('vRel', float('nan')),
    ('vLead', 0.4),
    ('vLead', -0.4),
    ('vLead', float('nan')),
    ('aLeadK', 0.3),
    ('aLeadK', float('nan')),
    ('yRel', 2.0),
    ('yRel', -2.0),
    ('yRel', float('nan')),
    ('modelProb', 0.97),
    ('modelProb', 1.01),
    ('modelProb', float('nan')),
  ],
)
def test_moving_adjacent_uncertain_and_invalid_leads_do_not_enter_fallback(field, value):
  state = near_stop_state()
  setattr(state['radarState'].leadOne, field, value)
  assert not close_stop(state)


def test_no_lead_does_not_enter_fallback():
  state = near_stop_state()
  state['radarState'].leadOne.present = False
  assert not close_stop(state)


def test_latched_stop_does_not_chatter_on_distance_noise():
  planner, state = braking_planner(), near_stop_state()
  update(planner, state)
  for distance in (4.0, 3.0, 4.1, 2.9):
    state['radarState'].leadOne.dRel = distance
    update(planner, state)
    assert planner.close_lead_stop
    assert planner.output_should_stop


@pytest.mark.parametrize('acceleration', [-1.5, -2.0, -3.5, STOP_RELEASE_ACCEL, 0.5])
def test_stronger_braking_or_a_clear_launch_releases_the_extra_latch(acceleration):
  planner, state = braking_planner(), near_stop_state()
  update(planner, state)
  assert planner.close_lead_stop
  update(planner, state, acceleration)
  assert not planner.close_lead_stop
  assert not planner.output_should_stop


@pytest.mark.parametrize(
  'condition',
  ['gas', 'brake', 'override', 'inactive', 'disabled', 'control_disabled', 'force_decel', 'unhealthy', 'stock_long', 'other_platform'],
)
def test_driver_mode_and_health_guards_clear_the_latch(condition):
  planner, state = braking_planner(), near_stop_state()
  update(planner, state)
  assert planner.close_lead_stop
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
    planner.CP.carFingerprint = 'OTHER'
  update(planner, state)
  assert not planner.close_lead_stop


def test_speed_above_the_low_speed_band_clears_the_extra_latch():
  planner, state = braking_planner(), near_stop_state()
  update(planner, state)
  state['carState'].vEgo = 2.0
  update(planner, state)
  assert not planner.close_lead_stop


def test_solver_failure_clears_the_extra_latch():
  planner, state = braking_planner(), near_stop_state()
  update(planner, state)
  planner.mpc.solution_status = 1
  with patch.object(planner.mpc, 'update', return_value=None):
    update(planner, state)
  assert not planner.close_lead_stop


@pytest.mark.parametrize('delay', [0.3, 0.5])
@pytest.mark.parametrize('offset', [0.0, 0.3, 0.5])
@pytest.mark.parametrize('speed', [1.0, 1.3, 1.8])
def test_low_speed_lag_plant_does_not_lose_clearance(delay, offset, speed):
  from openpilot.selfdrive.controls.tests.test_longitudinal_following_scenarios import simulate_following

  arguments = {'initial_speed': speed, 'lead_speed': 0.0, 'initial_gap': 6.0, 'delay': delay, 'response_time': 0.4, 'offset': offset, 'duration': 20.0}
  with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.should_stop_for_close_lead', return_value=False):
    before = simulate_following(**arguments)
  after = simulate_following(**arguments)
  assert after['minimum_gap'] >= before['minimum_gap'] - 1e-5
  assert after['final_speed'] == 0.0
  assert after['minimum_command'] >= -3.5
  assert after['maximum_command'] <= 2.0
  assert after['conflicts'] == 0
  if before['minimum_gap'] >= 2.0:
    assert after['minimum_gap'] >= 2.0


@pytest.mark.parametrize('speed', [0.3, 0.5, 1.3, 1.99])
@pytest.mark.parametrize('target', [-0.7, -1.3, -2.5, -3.5])
def test_stopping_transition_preserves_planner_braking(speed, target):
  control = controller(initial=-0.4)
  state = structs.CarState()
  state.vEgo = speed
  assert control.update(True, state, target, True, (-3.5, 2.0)) <= target


@pytest.mark.parametrize('speed', [0.0, 0.29, 2.0, 10.0])
def test_existing_settle_and_high_speed_ramp_are_unchanged(speed):
  control = controller(initial=-0.4)
  state = structs.CarState()
  state.vEgo = speed
  assert control.update(True, state, -1.3, True, (-3.5, 2.0)) == pytest.approx(-0.42)


def test_stopped_hold_keeps_the_existing_ramp():
  control = controller(initial=-0.4)
  state = structs.CarState()
  state.vEgo = 0.5
  state.standstill = True
  assert control.update(True, state, -1.3, True, (-3.5, 2.0)) == pytest.approx(-0.42)


@pytest.mark.parametrize('target', [0.5, float('nan'), float('inf'), float('-inf')])
def test_stopping_ramp_does_not_follow_positive_or_invalid_requests(target):
  control = controller(initial=-0.4)
  state = structs.CarState()
  state.vEgo = 1.3
  assert control.update(True, state, target, True, (-3.5, 2.0)) == pytest.approx(-0.42)


def test_stopping_brake_priority_still_respects_the_can_limit():
  control = controller(initial=-0.4)
  state = structs.CarState()
  state.vEgo = 1.3
  assert control.update(True, state, -5.0, True, (-3.5, 2.0)) == -3.5


def test_stopping_brake_priority_never_delays_manual_disable():
  control = controller(initial=-0.4)
  state = structs.CarState()
  state.vEgo = 1.3
  control.update(True, state, -1.3, True, (-3.5, 2.0))
  state.brakePressed = True
  assert control.update(False, state, -3.5, True, (-3.5, 2.0)) == 0.0


@pytest.mark.parametrize('platform,openpilot_long', [(CAR.HONDA_CIVIC_BOSCH, True), (CAR.HONDA_ACCORD, False)])
def test_other_platforms_and_stock_long_keep_the_original_stopping_ramp(platform, openpilot_long):
  control = controller(platform=platform, openpilot_long=openpilot_long, initial=-0.4)
  state = structs.CarState()
  state.vEgo = 1.3
  assert control.update(True, state, -1.3, True, (-3.5, 2.0)) == pytest.approx(-0.42)
