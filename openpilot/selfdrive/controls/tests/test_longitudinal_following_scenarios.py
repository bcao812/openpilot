from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.carstate import CarState
from opendbc.car.honda.hondacan import CanBus
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION, MotionReferenceKind
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.tests.test_longitudinal_cruise_coast import cruise_state
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import CruiseTarget, LongitudinalPlannerSP, LongitudinalPlanSource
from openpilot.tools.longitudinal.replay_pedal_switches import switch_metrics


def rate_limit_evidence(raw_commands, wire_commands, automatic_pairs, rate_limit=8.0):
  """Check raw control-request rate, then propagate Float32 serialization.

  For an increasing pair, the wire delta exceeds the raw delta by at most
  half the previous value's upward Float32 spacing plus half the current
  value's downward spacing. This per-pair bound does not change 8 m/s^3.
  Request rate here is not measured vehicle body jerk.
  """
  raw = np.asarray(raw_commands, dtype=np.float64)
  wire = np.asarray(wire_commands, dtype=np.float64)
  raw_delta, wire_delta = np.diff(raw), np.diff(wire)
  increment = rate_limit * DT_CTRL
  # Bound arithmetic rounding of the raw addition and subsequent subtraction.
  arithmetic = np.finfo(np.float64).eps * (np.abs(raw[1:]) + np.abs(raw[:-1]) + abs(increment))
  previous_up = np.nextafter(wire[:-1].astype(np.float32), np.float32(np.inf)).astype(np.float64) - wire[:-1]
  current_down = wire[1:] - np.nextafter(wire[1:].astype(np.float32), np.float32(-np.inf)).astype(np.float64)
  serialization = (previous_up + current_down) / 2.0
  return {
    'wire_is_float32_raw': bool(np.array_equal(wire, raw.astype(np.float32).astype(np.float64))),
    'maximum_raw_jerk_up': float(np.max(raw_delta[automatic_pairs]) / DT_CTRL),
    'raw_rate_violations': int(np.count_nonzero((raw_delta > increment + arithmetic)[automatic_pairs])),
    'wire_rate_violations': int(np.count_nonzero((wire_delta > increment + arithmetic + serialization)[automatic_pairs])),
    'maximum_wire_rounding_jerk_bound': float(np.max(serialization[automatic_pairs]) / DT_CTRL),
  }


def simulate_following(
  *, scenario='moving', delay=0.5, response_time=0.4, duration=25.0, initial_speed=1.1, lead_speed=0.8, initial_gap=7.0,
  offset=0.0, noise=0.0, diagnostics=None, platform=CAR.HONDA_ACCORD, braking_bias=0.0, measurement_filter=False,
  personality=None, lead_event_time=5.0, cut_in_gap=4.0, cut_in_speed=0.0,
):
  """Ideal acceleration/lag plant, not a calibrated Honda or rollback model."""
  params = CarInterface.get_non_essential_params(platform)
  params.openpilotLongitudinalControl = True
  params_sp = CarInterface.get_non_essential_params_sp(params, platform)
  speed_estimator = CarState(params, params_sp) if measurement_filter else None
  moving_preference = scenario == 'moving_preference'
  planner = LongitudinalPlanner(params, params_sp, init_v=initial_speed)
  longitudinal = LongControl(params, params_sp)
  unified = longitudinal.unified
  controller = CarController({Bus.pt: DBC[params.carFingerprint][Bus.pt]}, params, params_sp)
  parser = CANParser(DBC[params.carFingerprint][Bus.pt], [('ACC_CONTROL', 50)], CanBus(params).pt)
  wrapper = SimpleNamespace(v_cruise_factor=1.0, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={})
  state = cruise_state()
  state['modelV2'] = state['modelV2'].as_builder()
  state['selfdriveState'] = state['selfdriveState'].as_builder()
  if personality is not None:
    state['selfdriveState'].personality = personality
  state['carState'].vCruise = max(20.0, initial_speed + 5.0) * 3.6
  state['carState'].cruiseState.available = True
  delayed = deque([0.0] * max(1, round(delay / DT_CTRL)))
  speed, acceleration, gap = initial_speed, 0.0, initial_gap
  history, can_rows, raw_commands = [], [], []
  control_sp = structs.CarControlSP()
  command = 0.0

  def update_targets(instance, state, v_ego, a_ego, v_cruise):
    instance.source = LongitudinalPlanSource.cruise
    instance.cruise_target = CruiseTarget(instance.source, v_cruise, a_ego)
    return instance.cruise_target

  with (
    patch.object(LongitudinalPlannerSP, 'update', return_value=None),
    patch.object(LongitudinalPlannerSP, 'update_targets', new=update_targets),
    patch.object(LongitudinalPlanner, 'is_e2e', return_value=scenario == 'model_stop' or moving_preference),
  ):
    for frame in range(round(duration / DT_CTRL)):
      timestamp = frame * DT_CTRL
      now_ns = round((timestamp + 1.0) * 1e9)
      active = not (scenario == 'override' and 5.0 <= timestamp < 6.0)
      measured_speed = max(0.0, speed + noise * np.sin(2.0 * np.pi * 1.1 * timestamp))
      measured_acceleration = acceleration
      if speed_estimator is not None:
        measured_speed, measured_acceleration = speed_estimator.update_speed_kf(measured_speed)
      state['carState'].vEgo = float(measured_speed)
      state['carState'].aEgo = float(measured_acceleration)
      state['carState'].standstill = speed <= 0.001
      state['carControl'].longActive = active
      state['carControl'].cruiseControl.override = not active
      state['controlsState'].longControlState = longitudinal.long_control_state
      state['modelV2'].action.shouldStop = scenario == 'model_stop' and timestamp >= 5.0
      state['modelV2'].action.desiredAcceleration = -1.0 if state['modelV2'].action.shouldStop else 0.6 if moving_preference else 0.2
      if unified:
        state['controlsState'].motionState.version = MOTION_INTENT_VERSION
        state['controlsState'].motionState.planningMode = int(longitudinal.longitudinal_mode)
        state['controlsState'].motionState.enabled = True
        model_speed = speed + (0.4 if moving_preference else 0.5)
        state['modelV2'].velocity.x = [float(model_speed)] * len(ModelConstants.T_IDXS)
        state['modelV2'].position.x = [float(model_speed * sample) for sample in ModelConstants.T_IDXS]
        state['modelV2'].acceleration.x = [0.0] * len(ModelConstants.T_IDXS)
        state['modelV2'].orientationRate.z = [0.001] * len(ModelConstants.T_IDXS)
      if scenario == 'cut_in' and frame == round(lead_event_time / DT_CTRL):
        gap, lead_speed = cut_in_gap, cut_in_speed
      lead_accel = -min(3.0, lead_speed / DT_CTRL) if scenario == 'lead_stop' and timestamp >= lead_event_time else 0.0
      lead_speed = max(0.0, lead_speed + lead_accel * DT_CTRL)
      for lead in (state['radarState'].leadOne, state['radarState'].leadTwo):
        lead.present = not (scenario == 'lead_loss' and 5.0 <= timestamp < 5.5)
        lead.dRel = float(max(0.0, gap))
        lead.vLead = float(lead_speed)
        lead.vLeadK = float(lead_speed)
        lead.vRel = float(lead_speed - measured_speed)
        lead.aLeadK = float(lead_accel)
        lead.aLeadTau = 0.3
        lead.modelProb = 0.999
      if frame % round(DT_MDL / DT_CTRL) == 0:
        for service in state.logMonoTime:
          state.logMonoTime[service] = now_ns
        planner.update(state, now_ns=now_ns)
        assert planner.mpc.solution_status == 0
        command = float(np.float32(planner.output_a_target))
        if diagnostics is not None:
          diagnostics.append(
            {
              't': timestamp,
              'speed': speed,
              'acceleration': acceleration,
              'lead_speed': lead_speed,
              'gap': gap,
              'target': command,
              't_follow': float(planner.mpc.params[0, 4]),
              'stop': bool(planner.output_should_stop),
              'creep': bool(getattr(planner, 'creep_stop', False)),
              'latched': bool(getattr(planner, 'stop_latched', False)),
              'motion_phase': longitudinal.motion_phase.text,
              'resolved_stop': longitudinal.resolved_should_stop,
              'intent_version': planner.motion_intent.version if unified else 0,
              'state': str(longitudinal.long_control_state),
              'moving_lead_t': longitudinal.motion_supervisor.moving_lead_t if longitudinal.unified else 0.0,
              'mpc_v': float(planner.mpc.x0[1]),
            }
          )
      limits = CarInterface.get_pid_accel_limits(params, params_sp, measured_speed, state['carState'].vCruise / 3.6)
      control = state['carControl'].as_reader().as_builder()
      motion_input = {}
      if unified:
        motion_input = {
          'intent': planner.motion_intent,
          'plan_time_ns': int(state.logMonoTime['modelV2']),
          'model_time_ns': int(state.logMonoTime['modelV2']),
          'now_ns': now_ns,
          'plan_valid': planner.motion_intent.valid,
        }
      raw_command = float(longitudinal.update(active, state['carState'], command, planner.output_should_stop, limits, **motion_input))
      raw_commands.append(raw_command)
      control.actuators.accel = raw_command
      control.actuators.longControlState = longitudinal.long_control_state
      wrapper.out = state['carState']
      _, messages = controller.update(control.as_reader(), control_sp, wrapper, round(timestamp * 1e9))
      if frame % 2 == 0:
        parser.update([(round(timestamp * 1e9), messages)])
        values = parser.vl['ACC_CONTROL']
        can_rows.append(
          {
            't': timestamp,
            'automatic': active,
            'can_accel': values['ACCEL_COMMAND'],
            'can_brake': bool(values['BRAKE_REQUEST']),
            'can_gas': int(values['GAS_COMMAND']),
          }
        )
      applied = delayed.popleft()
      delayed.append(control.actuators.accel)
      delivered = applied + offset + (braking_bias if applied < 0.0 else 0.0)
      acceleration += (delivered - acceleration) * DT_CTRL / response_time
      speed = max(0.0, speed + acceleration * DT_CTRL)
      if speed == 0.0:
        acceleration = 0.0
      gap += (lead_speed - speed) * DT_CTRL
      history.append(
        (timestamp, speed, acceleration, command, control.actuators.accel, gap, lead_speed,
         longitudinal.long_control_state == LongCtrlState.stopping, active, longitudinal.pid.i, measured_acceleration,
         planner.motion_intent.reference_kind if unified else MotionReferenceKind.UNKNOWN)
      )
  values = np.asarray(history)
  steady = values[:, 0] >= duration - 10.0
  braking = values[(values[:, 0] >= lead_event_time) & (values[:, 4] < -1.0), 0]
  automatic_pairs = (values[1:, 8] == 1) & (values[:-1, 8] == 1)
  jerk = np.diff(values[:, 4])[automatic_pairs] / DT_CTRL
  return {
    'minimum_gap': float(np.min(values[:, 5])),
    'final_gap': float(values[-1, 5]),
    'final_speed': float(values[-1, 1]),
    'minimum_speed': float(np.min(values[:, 1])),
    'speed_error': float(np.mean(np.abs(values[steady, 1] - values[steady, 6]))),
    'stopping_seconds': float(np.sum(values[:, 7]) * DT_CTRL),
    'minimum_command': float(np.min(values[:, 4])),
    'maximum_command': float(np.max(values[:, 4])),
    'maximum_jerk_up': float(np.max(jerk)),
    'minimum_integral': float(np.min(values[:, 9])),
    'maximum_integral': float(np.max(values[:, 9])),
    'maximum_measured_acceleration': float(np.max(values[:, 10])),
    'reference_kinds': sorted(set(values[:, 11].astype(int).tolist())),
    **rate_limit_evidence(raw_commands, values[:, 4], automatic_pairs),
    'minimum_jerk_down': float(np.min(jerk)),
    'brake_response': float(braking[0] - lead_event_time) if len(braking) else None,
    'switches': switch_metrics(can_rows),
    'conflicts': sum(row['can_brake'] and row['can_gas'] > 0 for row in can_rows),
  }


@pytest.mark.parametrize('delay,response_time', [(0.1, 0.2), (0.3, 0.4), (0.5, 0.4), (0.8, 0.6)])
def test_moving_lead_is_followed_without_an_unnecessary_stop(delay, response_time):
  result = simulate_following(delay=delay, response_time=response_time)
  assert result['stopping_seconds'] == 0.0
  assert result['minimum_speed'] > 0.3
  assert result['speed_error'] < 0.15
  assert result['minimum_gap'] > 5.0
  assert result['conflicts'] == 0


@pytest.mark.parametrize('delay,response_time', [(0.1, 0.2), (0.3, 0.4), (0.5, 0.4), (0.8, 0.6)])
def test_model_moving_following_retains_progress_through_unwarmed_measurements(delay, response_time):
  """Fast ideal-plant regression; the frozen CAN-driven six-plant case remains mandatory."""
  result = simulate_following(scenario='moving_preference', initial_speed=2.0, lead_speed=2.0, initial_gap=11.0,
                              duration=20.0, delay=delay, response_time=response_time, measurement_filter=True)
  assert result['maximum_measured_acceleration'] > 10.0
  assert result['reference_kinds'] == [MotionReferenceKind.MOVING]
  # Raw MOVING facts do not change an ordinary target's ABSOLUTE_ACC domain.
  # Braking correction is permitted; positive integral remains forbidden.
  assert result['maximum_integral'] == 0.0
  assert result['stopping_seconds'] == 0.0
  assert result['minimum_speed'] > 0.3
  assert result['minimum_gap'] > 2.0
  assert result['wire_is_float32_raw']
  assert result['raw_rate_violations'] == result['wire_rate_violations'] == 0
  assert result['conflicts'] == 0


@pytest.mark.parametrize('scenario', ['lead_stop', 'cut_in', 'model_stop', 'override', 'lead_loss'])
def test_following_bypass_does_not_prevent_a_real_stop_or_override(scenario):
  result = simulate_following(scenario=scenario)
  assert result['minimum_gap'] > 2.0
  assert result['minimum_command'] >= -3.5
  assert result['maximum_command'] <= 2.0
  assert result['wire_is_float32_raw']
  assert result['raw_rate_violations'] == 0
  assert result['wire_rate_violations'] == 0
  assert result['conflicts'] == 0
  if scenario in ('lead_stop', 'cut_in', 'model_stop'):
    assert result['final_speed'] == 0.0
    assert result['stopping_seconds'] > 0.0
