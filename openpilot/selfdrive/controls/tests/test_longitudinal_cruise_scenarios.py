from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.hondacan import CanBus
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.tests.replay_pedal_switches import switch_metrics
from openpilot.selfdrive.controls.tests.test_longitudinal_cruise_coast import cruise_state
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP, LongitudinalPlanSource


def simulate_lead_cruise(extension, offset=0.25, noise=0.0, delay=0.5, scenario='steady', speed_target=20.0, duration=80.0,
                         response_time=0.4):
  params = CarInterface.get_non_essential_params(CAR.HONDA_ACCORD)
  params.openpilotLongitudinalControl = True
  params_sp = CarInterface.get_non_essential_params_sp(params, CAR.HONDA_ACCORD)
  speed, acceleration = speed_target + 0.2, 0.0
  planner = LongitudinalPlanner(params, params_sp, init_v=speed, experimental_coast_with_lead=extension)
  longitudinal = LongControl(params, params_sp)
  controller = CarController({Bus.pt: DBC[params.carFingerprint][Bus.pt]}, params, params_sp)
  parser = CANParser(DBC[params.carFingerprint][Bus.pt], [('ACC_CONTROL', 50)], CanBus(params).pt)
  control_sp = structs.CarControlSP()
  car_wrapper = SimpleNamespace(v_cruise_factor=1.0, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={})
  delayed_commands = deque([0.0] * max(1, round(delay / DT_CTRL)))
  state = cruise_state()
  state['carState'].vCruise = speed_target * 3.6
  state['carState'].vCruiseCluster = speed_target * 3.6
  state['carState'].cruiseState.available = True
  state['modelV2'] = state['modelV2'].as_builder()
  state['modelV2'].action.desiredAcceleration = 0.1
  gap, lead_speed = 100.0, speed_target
  command = 0.0
  history, can_rows = [], []

  def update_targets(instance, state, v_ego, a_ego, v_cruise):
    instance.source = LongitudinalPlanSource.cruise
    return v_cruise, a_ego

  with patch.object(LongitudinalPlannerSP, 'update', return_value=None), \
       patch.object(LongitudinalPlannerSP, 'update_targets', new=update_targets), \
       patch.object(LongitudinalPlanner, 'is_e2e', return_value=False):
    for frame in range(round(duration / DT_CTRL)):
      timestamp = frame * DT_CTRL
      measured_speed = max(0.0, speed + noise * np.sin(2 * np.pi * 1.1 * timestamp))
      state['carState'].vEgo = float(measured_speed)
      state['carState'].aEgo = float(acceleration)
      state['carState'].standstill = speed <= 0.001
      state['controlsState'].longControlState = longitudinal.long_control_state
      active = not (scenario == 'override' and 25 <= timestamp < 27)
      state['carControl'].longActive = active
      state['carControl'].cruiseControl.override = not active
      if scenario == 'cut_in' and frame == round(25.0 / DT_CTRL):
        gap, lead_speed = 45.0, 12.0
      lead_accel = -3.0 if scenario == 'lead_brake' and 25 <= timestamp < 25 + speed_target / 3 else 0.0
      lead_speed = max(0.0, lead_speed + lead_accel * DT_CTRL)
      lead = state['radarState'].leadOne
      lead.present = scenario != 'no_lead' and (scenario != 'flicker' or int(timestamp * 2) % 2 == 0)
      lead.dRel = float(gap)
      lead.vLead = float(lead_speed)
      lead.vRel = float(lead_speed - measured_speed)
      lead.aLeadK = lead_accel
      lead.aLeadTau = 1.5
      if frame % round(DT_MDL / DT_CTRL) == 0:
        planner.update(state)
        assert planner.mpc.solution_status == 0
        command = float(np.float32(planner.output_a_target))
      limits = CarInterface.get_pid_accel_limits(params, params_sp, measured_speed, speed_target)
      control = state['carControl'].as_reader().as_builder()
      control.actuators.accel = float(longitudinal.update(active, state['carState'], command, planner.output_should_stop, limits))
      control.actuators.longControlState = longitudinal.long_control_state
      car_wrapper.out = state['carState']
      _, messages = controller.update(control.as_reader(), control_sp, car_wrapper, round(timestamp * 1e9))
      if frame % 2 == 0:
        parser.update([(round(timestamp * 1e9), messages)])
        values = parser.vl['ACC_CONTROL']
        can_rows.append({'t': timestamp, 'automatic': active, 'can_accel': values['ACCEL_COMMAND'],
                         'can_brake': bool(values['BRAKE_REQUEST']), 'can_gas': int(values['GAS_COMMAND'])})
      applied = delayed_commands.popleft()
      delayed_commands.append(control.actuators.accel)
      acceleration += (applied + offset - acceleration) * DT_CTRL / response_time
      speed = max(0.0, speed + acceleration * DT_CTRL)
      gap += (lead_speed - speed) * DT_CTRL
      history.append((timestamp, speed, command, gap))
  values = np.array(history)
  steady = values[values[:, 0] >= duration - 20.0]
  braking = values[(values[:, 0] >= 25.0) & (values[:, 2] < -1.0), 0]
  return {'speed_error_kph': float((np.mean(steady[:, 1]) - speed_target) * 3.6),
          'speed_range_kph': float(np.ptp(steady[:, 1]) * 3.6), 'minimum_gap': float(np.min(values[:, 3])),
          'minimum_gap_time': float(values[np.argmin(values[:, 3]), 0]),
          'minimum_command': float(np.min(values[:, 2])), 'maximum_command': float(np.max(values[:, 2])),
          'brake_response': float(braking[0] - 25.0) if len(braking) else None,
          'switches': switch_metrics([row for row in can_rows if row['t'] >= 20.0]),
          'conflicts': sum(row['can_brake'] and row['can_gas'] > 0 for row in can_rows)}


@pytest.mark.parametrize('offset,noise,delay,scenario,speed_target', [
  (0.0, 0.0, 0.5, 'steady', 20.0),
  (0.2, 0.0, 0.5, 'steady', 20.0),
  (0.254545, 0.0, 0.5, 'steady', 20.0),
  (0.254545, 0.02, 0.8, 'steady', 20.0),
  (0.3, 0.02, 0.5, 'steady', 20.0),
  (0.25, 0.02, 0.5, 'flicker', 20.0),
  (0.25, 0.02, 0.5, 'override', 20.0),
  (0.25, 0.02, 0.5, 'steady', 10.0),
  (0.25, 0.02, 0.5, 'no_lead', 20.0),
])
def test_closed_loop_cruise_bounds(offset, noise, delay, scenario, speed_target):
  arguments = {'offset': offset, 'noise': noise, 'delay': delay, 'scenario': scenario, 'speed_target': speed_target}
  before = simulate_lead_cruise(False, **arguments)
  after = simulate_lead_cruise(True, **arguments)
  assert abs(after['speed_error_kph']) <= 1.1
  assert abs(after['speed_error_kph'] - before['speed_error_kph']) <= 0.25
  assert after['speed_range_kph'] <= max(0.4, before['speed_range_kph'] + 0.2)
  assert after['minimum_command'] >= -3.5
  assert after['maximum_command'] <= 2.0
  assert after['conflicts'] == 0
  if scenario == 'no_lead':
    assert before == after


@pytest.mark.parametrize('scenario', ['cut_in', 'lead_brake'])
def test_closed_loop_risk_response(scenario):
  before = simulate_lead_cruise(False, scenario=scenario, duration=55.0, delay=0.3, response_time=0.2)
  after = simulate_lead_cruise(True, scenario=scenario, duration=55.0, delay=0.3, response_time=0.2)
  assert after['minimum_gap'] > 2.0
  assert after['minimum_gap'] >= before['minimum_gap'] - 0.1
  assert after['brake_response'] <= before['brake_response'] + DT_MDL + 1e-6
  assert after['conflicts'] == 0


@pytest.mark.xfail(strict=True, reason='Experimental lead shaping introduces a limit cycle with long unmodelled actuator delay')
@pytest.mark.parametrize('offset,delay', [(0.254545, 0.8), (0.25, 0.5), (0.254545, 0.5)])
def test_experimental_lead_extension_release_gate(offset, delay):
  before = simulate_lead_cruise(False, offset=offset, noise=0.02, delay=delay)
  after = simulate_lead_cruise(True, offset=offset, noise=0.02, delay=delay)
  assert after['switches']['debounced'] <= before['switches']['debounced']


@pytest.mark.xfail(strict=True, reason='Both default and experimental control exceed the stop gap under excessive actuator lag')
def test_long_actuator_lag_collision_release_gate():
  for enabled in (False, True):
    result = simulate_lead_cruise(enabled, scenario='lead_brake', duration=55.0, delay=0.5, response_time=0.4)
    assert result['minimum_gap'] > 2.0
