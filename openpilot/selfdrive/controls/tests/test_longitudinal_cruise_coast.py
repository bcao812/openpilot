from unittest.mock import patch

import numpy as np
import pytest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, structs
from opendbc.car.honda.hondacan import CanBus, create_acc_commands
from opendbc.car.honda.values import CAR, DBC, CarControllerParams, HondaFlags
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_planner import (CRUISE_COAST_ACCEL, LongitudinalPlanner,
                                                                 LongitudinalPlanSource, get_cruise_coast_accel)
from openpilot.selfdrive.controls.tests.test_longitudinal_planner_state import braking_planner, braking_state
from openpilot.selfdrive.controls.tests.replay_longitudinal_regressions import debounced_flips
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP, LongitudinalPlanSource as PlanSourceSP


def cruise_planner(platform=CAR.HONDA_ACCORD):
  planner = braking_planner(platform)
  planner.v_desired_filter.x = 20.2
  planner.a_cruise = -0.2
  planner.output_a_target = -0.2
  return planner


def cruise_state():
  state = braking_state(20.2)
  state['carState'].vCruise = 72.0
  state['carState'].vCruiseCluster = 72.0
  state['carControl'].longActive = True
  return state


def update_cruise(planner, state, a_cruise=-0.23, a_mpc=1.0, e2e=False, source=PlanSourceSP.cruise):
  def update_targets(instance, state, v_ego, a_ego, v_cruise):
    instance.source = source
    return v_cruise, a_ego

  with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.get_cruise_accel', return_value=a_cruise), \
       patch('openpilot.selfdrive.controls.lib.longitudinal_planner.get_accel_from_plan', return_value=a_mpc), \
       patch.object(LongitudinalPlanner, 'is_e2e', return_value=e2e), \
       patch.object(LongitudinalPlannerSP, 'update_targets', new=update_targets):
    planner.update(state)
  return planner.output_a_target


@pytest.mark.parametrize('accel', [-3.5, -1.2, -0.3, -0.1, 0.0, 0.2, 2.0])
def test_requests_outside_small_coasting_band_are_exactly_preserved(accel):
  assert get_cruise_coast_accel(accel) == accel


def test_coast_mapping_is_continuous_monotonic_and_bounded():
  requests = np.linspace(-0.4, 0.1, 2001)
  mapped = np.array([get_cruise_coast_accel(float(request)) for request in requests])
  assert np.all(np.diff(mapped) >= -1e-12)
  assert max(abs(mapped - requests)) <= 0.06 + 1e-12
  for boundary in (-0.3, -0.25, -0.15, -0.1):
    assert abs(get_cruise_coast_accel(boundary + 1e-7) - get_cruise_coast_accel(boundary - 1e-7)) < 1e-6


def test_qualified_cruise_is_mapped_but_internal_jerk_state_is_not():
  planner = cruise_planner()
  assert update_cruise(planner, cruise_state()) == CRUISE_COAST_ACCEL
  assert planner.a_cruise == -0.23
  assert planner.mpc.source == LongitudinalPlanSource.cruise


@pytest.mark.parametrize('a_mpc', [-2.0, -0.24, -0.22, -0.2])
def test_mpc_can_never_be_weakened(a_mpc):
  planner = cruise_planner()
  assert update_cruise(planner, cruise_state(), a_mpc=a_mpc) <= a_mpc


def test_e2e_braking_bypasses_coasting():
  planner = cruise_planner()
  state = cruise_state()
  state['modelV2'] = state['modelV2'].as_builder()
  state['modelV2'].action.desiredAcceleration = -2.0
  assert update_cruise(planner, state, e2e=True) == -2.0
  assert planner.mpc.source == LongitudinalPlanSource.e2e


@pytest.mark.parametrize('lead_name', ['leadOne', 'leadTwo'])
def test_any_lead_bypasses_coasting(lead_name):
  planner = cruise_planner()
  state = cruise_state()
  getattr(state['radarState'], lead_name).present = True
  assert update_cruise(planner, state) == -0.23


@pytest.mark.parametrize('platform', [CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_HYBRID])
def test_other_platforms_are_unchanged(platform):
  assert update_cruise(cruise_planner(platform), cruise_state()) == -0.23


@pytest.mark.parametrize('condition', ['gas', 'brake', 'override', 'inactive', 'force_decel', 'low_speed',
                                       'far_from_target', 'unhealthy', 'stock_long', 'throttle_veto'])
def test_coast_preconditions_are_required(condition):
  planner = cruise_planner()
  state = cruise_state()
  if condition == 'gas':
    state['carState'].gasPressed = True
  elif condition == 'brake':
    state['carState'].brakePressed = True
  elif condition == 'override':
    state['carControl'].cruiseControl.override = True
  elif condition == 'inactive':
    state['carControl'].longActive = False
  elif condition == 'force_decel':
    state['controlsState'].forceDecel = True
  elif condition == 'low_speed':
    state['carState'].vEgo = 2.0
    state['carState'].vCruise = 7.2
  elif condition == 'far_from_target':
    state['carState'].vEgo = 22.0
  elif condition == 'unhealthy':
    state.all_checks = lambda service_list=None: False
  elif condition == 'stock_long':
    planner.CP.openpilotLongitudinalControl = False
  elif condition == 'throttle_veto':
    state['modelV2'] = state['modelV2'].as_builder()
    state['modelV2'].meta.disengagePredictions.gasPressProbs = [0.0, 0.0]
  assert update_cruise(planner, state) == -0.23


@pytest.mark.parametrize('source', [PlanSourceSP.sccVision, PlanSourceSP.sccMap, PlanSourceSP.speedLimitAssist])
def test_non_cruise_speed_constraints_are_not_weakened(source):
  assert update_cruise(cruise_planner(), cruise_state(), source=source) == -0.23


def decoded_commands(accelerations):
  params = structs.CarParams()
  params.flags = int(HondaFlags.BOSCH)
  params.openpilotLongitudinalControl = True
  params.init('safetyConfigs', 1)
  bus = CanBus(params)
  dbc_name = DBC[CAR.HONDA_ACCORD][Bus.pt]
  packer = CANPacker(dbc_name)
  parser = CANParser(dbc_name, [('ACC_CONTROL', 50)], bus.pt)
  result = []
  for index, accel in enumerate(accelerations):
    control = structs.CarControl.new_message()
    control.actuators.accel = float(accel)
    serialized = control.as_reader().actuators.accel
    gas = float(np.interp(serialized, CarControllerParams.BOSCH_GAS_LOOKUP_BP, CarControllerParams.BOSCH_GAS_LOOKUP_V))
    messages = create_acc_commands(packer, bus, True, True, serialized, gas, 0, params)
    parser.update([((index + 1) * 20_000_000, messages)])
    result.append(dict(parser.vl['ACC_CONTROL']))
  return result


def test_float32_and_real_can_remain_consistent():
  values = decoded_commands([-0.23, CRUISE_COAST_ACCEL, -1.2, 0.1])
  assert values[1]['ACCEL_COMMAND'] == pytest.approx(-0.19)
  assert values[1]['BRAKE_REQUEST'] == 0
  assert values[1]['BRAKE_LIGHTS'] == 0
  assert 0 < values[1]['GAS_COMMAND'] < 10
  assert values[2]['BRAKE_REQUEST'] == 1
  assert values[2]['GAS_COMMAND'] == -30000
  assert values[2]['ACCEL_COMMAND'] == pytest.approx(-1.2)


def test_small_oscillation_does_not_switch_can_braking():
  requests = [-0.2 + 0.045 * np.sin(index * np.pi / 10) for index in range(200)]
  before = decoded_commands(requests)
  after = decoded_commands([get_cruise_coast_accel(request) for request in requests])
  before_flips = sum(previous['BRAKE_REQUEST'] != current['BRAKE_REQUEST']
                     for previous, current in zip(before, before[1:], strict=False))
  after_flips = sum(previous['BRAKE_REQUEST'] != current['BRAKE_REQUEST']
                    for previous, current in zip(after, after[1:], strict=False))
  assert before_flips >= 18
  assert after_flips == 0
  assert all(value['BRAKE_REQUEST'] == 0 for value in after)


def test_risk_request_bypasses_after_a_coasting_frame():
  planner = cruise_planner()
  state = cruise_state()
  assert update_cruise(planner, state) == CRUISE_COAST_ACCEL
  assert update_cruise(planner, state, a_cruise=-1.2) == -1.2
  assert update_cruise(planner, state, a_mpc=-2.0) == -2.0


def test_event_comparison_ignores_short_pulses():
  rows = [{'t': index * 0.05, 'can_brake': index >= 4, 'can_gas': 0 if index >= 4 else 7}
          for index in range(8)]
  assert debounced_flips(rows) == 1
  rows[1]['can_brake'] = True
  rows[1]['can_gas'] = -30000
  assert debounced_flips(rows) == 0
  assert debounced_flips([]) == 0


def simulate_cruise(platform, acceleration_offset):
  planner = cruise_planner(platform)
  speed, acceleration = 20.2, 0.0
  history = []

  def update_targets(instance, state, v_ego, a_ego, v_cruise):
    instance.source = PlanSourceSP.cruise
    return v_cruise, a_ego

  with patch.object(LongitudinalPlannerSP, 'update_targets', new=update_targets):
    for _ in range(int(60.0 / DT_MDL)):
      state = cruise_state()
      state['carState'].vEgo = float(speed)
      state['carState'].aEgo = float(acceleration)
      planner.update(state)
      assert planner.mpc.solution_status == 0
      command = float(planner.output_a_target)
      acceleration += (command + acceleration_offset - acceleration) * DT_MDL / 0.5
      speed += acceleration * DT_MDL
      history.append((speed, command))
  steady = history[-int(20.0 / DT_MDL):]
  return {'speed_error_kph': (speed - 20.0) * 3.6,
          'speed_range_kph': (max(row[0] for row in steady) - min(row[0] for row in steady)) * 3.6,
          'minimum_command': min(row[1] for row in history),
          'maximum_command': max(row[1] for row in history)}


@pytest.mark.parametrize('acceleration_offset', [0.0, 0.2, 0.25, 0.3])
def test_closed_loop_coasting_does_not_accumulate_speed_error(acceleration_offset):
  baseline = simulate_cruise(CAR.HONDA_CIVIC_BOSCH, acceleration_offset)
  candidate = simulate_cruise(CAR.HONDA_ACCORD, acceleration_offset)
  assert abs(candidate['speed_error_kph']) <= 1.1
  assert abs(candidate['speed_error_kph'] - baseline['speed_error_kph']) <= 0.22
  assert candidate['speed_range_kph'] < 0.01
  assert candidate['minimum_command'] >= -3.5
  assert candidate['maximum_command'] <= 2.0
