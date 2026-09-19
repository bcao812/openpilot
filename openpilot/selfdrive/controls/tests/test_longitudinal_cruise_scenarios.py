"""Cruise/CAN plant regressions through the current Accord UNIFIED runtime.

The original risk inputs, lag plant and absolute thresholds remain in force.
"""

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
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import Publisher, make_handover_state
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.tools.longitudinal.replay_pedal_switches import switch_metrics
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import CruiseTarget, LongitudinalPlannerSP, LongitudinalPlanSource


def simulate_lead_cruise(*, offset=0.25, noise=0.0, delay=0.5, scenario='steady', speed_target=20.0, duration=80.0,
                         response_time=0.4, lead_braking_duration=None, lead_slot='leadOne', offset_profile=None):
  # Optional, explicitly declared extra cases. None preserves every original
  # plant input and the constant-offset arithmetic below.
  if offset_profile is not None:
    offset_profile = tuple(offset_profile)
    if (not offset_profile or offset_profile[0][0] != 0.0 or not np.all(np.isfinite(offset_profile))
        or any(current[0] >= following[0] for current, following in zip(offset_profile, offset_profile[1:], strict=False))
        or offset_profile[-1][0] >= duration):
      raise ValueError('Invalid declared disturbance profile')
  params = CarInterface.get_non_essential_params(CAR.HONDA_ACCORD)
  params.openpilotLongitudinalControl = True
  params_sp = CarInterface.get_non_essential_params_sp(params, CAR.HONDA_ACCORD)
  speed, acceleration = speed_target + 0.2, 0.0
  planner = LongitudinalPlanner(params, params_sp, init_v=speed)
  longitudinal = LongControl(params, params_sp)
  assert planner.longitudinal_mode is longitudinal.longitudinal_mode is LongitudinalMode.UNIFIED
  controller = CarController({Bus.pt: DBC[params.carFingerprint][Bus.pt]}, params, params_sp)
  parser = CANParser(DBC[params.carFingerprint][Bus.pt], [('ACC_CONTROL', 50)], CanBus(params).pt)
  control_sp = structs.CarControlSP()
  car_wrapper = SimpleNamespace(v_cruise_factor=1.0, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={})
  delayed_commands = deque([0.0] * max(1, round(delay / DT_CTRL)))
  state = make_handover_state(speed=speed, target=speed_target)
  state['carState'].vCruise = speed_target * 3.6
  state['carState'].vCruiseCluster = speed_target * 3.6
  state['carState'].cruiseState.available = True
  state['modelV2'].action.desiredAcceleration = 0.1
  gap, lead_speed = 100.0, speed_target
  command = 0.0
  publisher = Publisher()
  wire_plan = None
  history, can_rows = [], []

  def update_targets(instance, state, v_ego, a_ego, v_cruise):
    instance.source = LongitudinalPlanSource.cruise
    instance.cruise_target = CruiseTarget(instance.source, v_cruise, a_ego)
    return instance.cruise_target

  with patch.object(LongitudinalPlannerSP, 'update', return_value=None), \
       patch.object(LongitudinalPlannerSP, 'update_targets', new=update_targets), \
       patch.object(LongitudinalPlanner, 'is_e2e', return_value=False):
    for frame in range(round(duration / DT_CTRL)):
      timestamp = frame * DT_CTRL
      now_ns = 1_000_000_000 + round(timestamp * 1e9)
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
      braking_duration = speed_target / 3 if lead_braking_duration is None else lead_braking_duration
      lead_accel = -3.0 if scenario == 'lead_brake' and 25 <= timestamp < 25 + braking_duration else 0.0
      lead_speed = max(0.0, lead_speed + lead_accel * DT_CTRL)
      lead = getattr(state['radarState'], lead_slot)
      lead.present = scenario != 'no_lead' and (scenario != 'flicker' or int(timestamp * 2) % 2 == 0)
      lead.dRel = float(gap)
      lead.vLead = float(lead_speed)
      lead.vRel = float(lead_speed - measured_speed)
      lead.aLeadK = lead_accel
      lead.aLeadTau = 1.5
      if frame % round(DT_MDL / DT_CTRL) == 0:
        for service in state.logMonoTime:
          state.logMonoTime[service] = now_ns
        state['modelV2'].velocity.x = [float(measured_speed)] * len(ModelConstants.T_IDXS)
        state['modelV2'].position.x = [float(measured_speed * sample) for sample in ModelConstants.T_IDXS]
        planner.update(state, now_ns=now_ns)
        assert planner.mpc.solution_status == 0
        with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=now_ns + 500_000):
          planner.publish(state, publisher)
        wire_plan = publisher.messages['longitudinalPlan']
        command = float(wire_plan.longitudinalPlan.aTarget)
      limits = CarInterface.get_pid_accel_limits(params, params_sp, measured_speed, speed_target)
      control = state['carControl'].as_reader().as_builder()
      plan = wire_plan.longitudinalPlan
      control.actuators.accel = float(longitudinal.update(
        active, state['carState'], command, plan.shouldStop, limits,
        intent=MotionIntent.from_message(plan.motionIntent), plan_time_ns=wire_plan.logMonoTime,
        model_time_ns=plan.modelMonoTime, now_ns=now_ns + 1_000_000,
        plan_valid=wire_plan.valid, solver_status=plan.solverStatus, braking_plan_valid=state.all_checks(['carState']),
      ))
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
      disturbance = offset
      if offset_profile is not None:
        for start, value in offset_profile:
          if timestamp < start:
            break
          disturbance = value
      acceleration += (applied + disturbance - acceleration) * DT_CTRL / response_time
      speed = max(0.0, speed + acceleration * DT_CTRL)
      gap += (lead_speed - speed) * DT_CTRL
      history.append((timestamp, speed, command, gap))
  values = np.array(history)
  steady = values[values[:, 0] >= duration - 20.0]
  braking = values[(values[:, 0] >= 25.0) & (values[:, 2] < -1.0), 0]
  result = {'runtime_mode': planner.longitudinal_mode.name, 'gas_interceptor': bool(params_sp.enableGasInterceptor),
          'speed_error_kph': float((np.mean(steady[:, 1]) - speed_target) * 3.6),
          'minimum_speed': float(np.min(values[:, 1])),
          'speed_range_kph': float(np.ptp(steady[:, 1]) * 3.6), 'minimum_gap': float(np.min(values[:, 3])),
          'minimum_gap_time': float(values[np.argmin(values[:, 3]), 0]),
          'minimum_command': float(np.min(values[:, 2])), 'maximum_command': float(np.max(values[:, 2])),
          'brake_response': float(braking[0] - 25.0) if len(braking) else None,
          'switches': switch_metrics([row for row in can_rows if row['t'] >= 20.0]),
          'conflicts': sum(row['can_brake'] and row['can_gas'] > 0 for row in can_rows)}
  if offset_profile is not None:
    result['disturbance_windows'] = []
    for index, (start, value) in enumerate(offset_profile):
      end = offset_profile[index + 1][0] if index + 1 < len(offset_profile) else duration
      tail = values[(values[:, 0] >= max(start, end - 20.0)) & (values[:, 0] < end)]
      result['disturbance_windows'].append({
        'start': start, 'end': end, 'offset': value,
        'speed_error_kph': float((np.mean(tail[:, 1]) - speed_target) * 3.6),
        'speed_range_kph': float(np.ptp(tail[:, 1]) * 3.6),
      })
  return result


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
  result = simulate_lead_cruise(**arguments)
  assert abs(result['speed_error_kph']) <= 1.1
  assert result['speed_range_kph'] <= 0.4
  assert result['minimum_command'] >= -3.5
  assert result['maximum_command'] <= 2.0
  assert result['conflicts'] == 0


@pytest.mark.parametrize('offset', [-0.25, 0.25])
def test_pure_cruise_rejects_both_signed_disturbances_in_the_original_speed_tracking_plant(offset):
  result = simulate_lead_cruise(offset=offset, scenario='no_lead')
  assert abs(result['speed_error_kph']) <= 1.1
  assert result['speed_range_kph'] <= 0.4
  assert result['minimum_command'] >= -3.5
  assert result['maximum_command'] <= 2.0
  assert result['conflicts'] == 0


@pytest.mark.parametrize('first_offset', [-0.25, 0.25])
def test_pure_cruise_disturbance_reversal_recovers_in_every_declared_window(first_offset):
  profile = ((0.0, first_offset), (40.0, 0.0), (80.0, -first_offset))
  result = simulate_lead_cruise(scenario='no_lead', duration=120.0, offset_profile=profile)
  assert len(result['disturbance_windows']) == len(profile)
  for window in result['disturbance_windows']:
    assert abs(window['speed_error_kph']) <= 1.1
    assert window['speed_range_kph'] <= 0.4
  assert result['minimum_command'] >= -3.5
  assert result['maximum_command'] <= 2.0
  assert result['conflicts'] == 0


@pytest.mark.parametrize('scenario', ['cut_in', 'lead_brake'])
def test_closed_loop_risk_response(scenario):
  result = simulate_lead_cruise(scenario=scenario, duration=55.0, delay=0.3, response_time=0.2)
  assert result['minimum_gap'] > 2.0
  assert result['brake_response'] is not None
  assert result['conflicts'] == 0


def test_long_actuator_lag_collision_release_gate():
  result = simulate_lead_cruise(scenario='lead_brake', duration=55.0, delay=0.5, response_time=0.4)
  assert result['minimum_gap'] > 2.0


@pytest.mark.parametrize('lead_slot', ['leadOne', 'leadTwo'])
def test_short_lead_braking_keeps_clearance_without_an_unnecessary_stop(lead_slot):
  result = simulate_lead_cruise(scenario='lead_brake', duration=45.0, delay=0.5, response_time=0.4,
                                lead_braking_duration=0.5, lead_slot=lead_slot)
  assert result['minimum_gap'] > 2.0
  assert result['minimum_speed'] > 1.0
  assert result['minimum_command'] >= -3.5
  assert result['maximum_command'] <= 2.0
  assert result['conflicts'] == 0
