"""Current cruise/owner and native CAN contracts after retiring scalar coast remapping."""

import pytest

from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import braking_state, consume, make_handover_state, make_planner_owner
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.tools.longitudinal.replay_longitudinal_regressions import debounced_flips
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import decoded_commands


def cruise_state():
  state = braking_state(20.2)
  state['carState'].vCruise = 72.0
  state['carState'].vCruiseCluster = 72.0
  state['carControl'].longActive = True
  return state


def test_float32_and_real_can_remain_consistent():
  values = decoded_commands([-0.23, -0.19, -1.2, 0.1])
  assert values[1]['ACCEL_COMMAND'] == pytest.approx(-0.19)
  assert values[1]['BRAKE_REQUEST'] == 0
  assert values[1]['BRAKE_LIGHTS'] == 0
  assert 0 < values[1]['GAS_COMMAND'] < 10
  assert values[2]['BRAKE_REQUEST'] == 1
  assert values[2]['GAS_COMMAND'] == -30000
  assert values[2]['ACCEL_COMMAND'] == pytest.approx(-1.2)


def test_event_comparison_ignores_short_pulses():
  rows = [{'t': index * 0.05, 'can_brake': index >= 4, 'can_gas': 0 if index >= 4 else 7}
          for index in range(8)]
  assert debounced_flips(rows) == 1
  rows[1]['can_brake'] = True
  rows[1]['can_gas'] = -30000
  assert debounced_flips(rows) == 0
  assert debounced_flips([]) == 0


@pytest.mark.parametrize('a_mpc', [-2.0, -0.24, -0.22, -0.2])
def test_admitted_braking_reference_is_not_weakened_by_the_current_owner(a_mpc):
  from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import warm_tracking_harness

  harness = warm_tracking_harness(20.2, -0.2)
  output = harness.step(a_mpc, MotionIntent(allow_creep=False))
  assert output <= a_mpc
  command = decoded_commands([output])[0]
  assert not (command['BRAKE_REQUEST'] and command['GAS_COMMAND'] > 0)


def test_current_model_braking_cap_reaches_owner_and_can():
  planner, owner = make_planner_owner(speed=20.2)
  state = make_handover_state(speed=20.2, target=20.0, experimental=True)
  state['modelV2'].action.desiredAcceleration = -2.0
  plan, outputs = consume(planner, owner, state, 0)
  assert plan.motionIntent.valid
  assert plan.aTarget <= -2.0
  assert max(outputs) <= -2.0
  assert all(row['BRAKE_REQUEST'] and row['GAS_COMMAND'] <= 0 for row in decoded_commands(outputs))


@pytest.mark.parametrize('acceleration_offset', [0.0, 0.2, 0.25, 0.3])
def test_current_cruise_does_not_accumulate_speed_error(acceleration_offset):
  from openpilot.selfdrive.controls.tests.test_longitudinal_cruise_scenarios import simulate_lead_cruise

  # Preserve the old no-lead initial speed, offset, first-order response and
  # absolute limits, through the current owner and real Honda CAN path.
  result = simulate_lead_cruise(scenario='no_lead', offset=acceleration_offset, delay=0.0, response_time=0.5, duration=60.0)
  assert abs(result['speed_error_kph']) <= 1.1
  assert result['speed_range_kph'] < 0.01
  assert result['minimum_command'] >= -3.5
  assert result['maximum_command'] <= 2.0
  assert result['conflicts'] == 0
