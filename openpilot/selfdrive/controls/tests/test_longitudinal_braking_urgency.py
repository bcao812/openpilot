from copy import deepcopy
from unittest.mock import Mock

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import A_CHANGE_COST, DANGER_ZONE_COST, J_EGO_COST, LongitudinalMpc
from openpilot.selfdrive.controls.lib.longitudinal_planner import get_lead_braking_urgency
from openpilot.selfdrive.controls.tests.test_longitudinal_cruise_coast import assert_invalid_motion_preserves_braking
from openpilot.selfdrive.controls.tests.test_longitudinal_lead_state import closing_planner, closing_state


def urgent_state():
  state = closing_state(20.0, 44.0, -10.0)
  state['radarState'].leadOne.aLeadK = -1.5
  return state


def urgency(state, speed=20.0, acceleration=-1.0, delay=0.5):
  return get_lead_braking_urgency(speed, acceleration, state['radarState'], delay)


def test_confirmed_closing_braking_uses_the_full_bounded_urgency():
  assert urgency(urgent_state()) == 1.0


def test_moving_lead_braking_is_part_of_the_relative_deceleration_budget():
  state = closing_state(20.0, 18.0, -2.0)
  assert urgency(state) == 0.0
  state['radarState'].leadOne.aLeadK = -3.0
  assert urgency(state) == pytest.approx(0.5)


def test_second_lead_can_request_urgency_without_rewriting_either_lead():
  state = urgent_state()
  state['radarState'].leadTwo = state['radarState'].leadOne.as_reader()
  state['radarState'].leadOne.present = False
  before = deepcopy(state['radarState'].to_dict())
  assert urgency(state) == 1.0
  assert before == state['radarState'].to_dict()


@pytest.mark.parametrize('acceleration', [-0.5, -0.2, 0.0, 0.5, float('nan'), float('inf')])
def test_coast_launch_and_invalid_acceleration_keep_normal_weights(acceleration):
  assert urgency(urgent_state(), acceleration=acceleration) == 0.0


@pytest.mark.parametrize('speed', [0.0, 1.2, 2.0, 5.0, float('nan'), float('inf')])
def test_low_speed_and_invalid_speed_keep_normal_weights(speed):
  assert urgency(urgent_state(), speed=speed) == 0.0


@pytest.mark.parametrize('delay', [-0.1, 1.01, float('nan'), float('inf')])
def test_invalid_actuator_delay_cannot_request_urgency(delay):
  assert urgency(urgent_state(), delay=delay) == 0.0


@pytest.mark.parametrize(
  'field,value',
  [
    ('dRel', 0.0),
    ('dRel', -1.0),
    ('dRel', float('nan')),
    ('vRel', float('nan')),
    ('vRel', 0.0),
    ('vLead', float('nan')),
    ('vLead', -0.6),
    ('vLead', 25.0),
    ('aLeadK', float('nan')),
    ('modelProb', 0.94),
    ('modelProb', 1.01),
    ('modelProb', float('nan')),
  ],
)
def test_uncertain_or_inconsistent_lead_keeps_normal_weights(field, value):
  state = urgent_state()
  setattr(state['radarState'].leadOne, field, value)
  assert urgency(state) == 0.0


def test_no_lead_and_distant_mild_closure_keep_normal_weights():
  state = urgent_state()
  state['radarState'].leadOne.present = False
  assert urgency(state) == 0.0
  state = closing_state(20.0, 150.0, -3.0)
  assert urgency(state) == 0.0


def test_delay_can_raise_urgency_but_never_reduce_it():
  state = closing_state(20.0, 35.0, -8.0)
  values = [urgency(state, delay=float(delay)) for delay in np.linspace(0.0, 1.0, 21)]
  assert all(0.0 <= value <= 1.0 for value in values)
  assert all(after >= before for before, after in zip(values, values[1:], strict=False))


def test_braking_and_speed_blends_are_continuous():
  state = urgent_state()
  assert urgency(state, acceleration=-0.75) == pytest.approx(0.5)
  for acceleration in (-1.0, -0.5):
    assert abs(urgency(state, acceleration=acceleration - 1e-7) - urgency(state, acceleration=acceleration + 1e-7)) < 1e-5
  for speed in (5.0, 10.0):
    values = []
    for varied in (speed - 1e-7, speed + 1e-7):
      current = closing_state(varied, 3.0, -3.0)
      values.append(urgency(current, speed=varied))
    assert abs(values[1] - values[0]) < 1e-5


@pytest.mark.parametrize('scale', [0.0, 0.5, 1.0])
def test_mpc_scales_only_previous_acceleration_cost(scale):
  planner = object.__new__(LongitudinalMpc)
  planner.set_cost_weights = Mock()
  planner.set_weights(a_change_scale=scale)
  weights, constraints = planner.set_cost_weights.call_args.args
  assert weights[4] == A_CHANGE_COST * scale
  assert weights[-1] == J_EGO_COST
  assert constraints[-1] == DANGER_ZONE_COST


@pytest.mark.parametrize('scale,expected', [(float('nan'), 1.0), (float('inf'), 1.0), (-1.0, 0.0), (2.0, 1.0)])
def test_mpc_scale_is_bounded_even_for_an_invalid_caller(scale, expected):
  planner = object.__new__(LongitudinalMpc)
  planner.set_cost_weights = Mock()
  planner.set_weights(a_change_scale=scale)
  assert planner.set_cost_weights.call_args.args[0][4] == A_CHANGE_COST * expected


def test_disabled_acceleration_constraint_is_still_disabled():
  planner = object.__new__(LongitudinalMpc)
  planner.set_cost_weights = Mock()
  planner.set_weights(False, a_change_scale=0.5)
  assert planner.set_cost_weights.call_args.args[0][4] == 0.0


@pytest.mark.parametrize(
  'condition',
  ['gas', 'brake', 'override', 'inactive', 'disabled', 'control_disabled', 'force_decel', 'unhealthy', 'stock_long', 'other_platform'],
)
def test_mode_and_driver_guards_preserve_normal_weights(condition):
  planner, state = closing_planner(), urgent_state()
  previous_target = planner.output_a_target
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
  original = planner.mpc.set_weights
  planner.mpc.set_weights = Mock(wraps=original)
  planner.update(state)
  if condition == 'unhealthy':
    calls = planner.mpc.set_weights.call_args_list
    assert calls[0].kwargs['a_change_scale'] == 1.0
    # Invalid-input reset calls set_weights() with its normal default scale.
    assert all(call.kwargs.get('a_change_scale', 1.0) == 1.0 for call in calls)
    assert_invalid_motion_preserves_braking(planner, state, previous_target)
  else:
    assert planner.mpc.set_weights.call_args.kwargs['a_change_scale'] == 1.0


def test_real_mpc_receives_urgency_and_keeps_its_acceleration_limits():
  planner, state = closing_planner(), urgent_state()
  original = planner.mpc.set_weights
  planner.mpc.set_weights = Mock(wraps=original)
  planner.update(state)
  assert planner.mpc.set_weights.call_args.kwargs['a_change_scale'] == 0.5
  assert planner.mpc.solution_status == 0
  assert -3.5 <= planner.output_a_target <= 2.0
