from copy import deepcopy

import numpy as np
import pytest

from opendbc.car.honda.values import CAR
from openpilot.cereal import custom
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_planner import LOW_SPEED_STATE_ERROR, constrain_braking_speed
from openpilot.selfdrive.controls.tests.test_longitudinal_planner_creep import build_planner, build_sm


def braking_planner(platform=CAR.HONDA_ACCORD):
  planner = build_planner()
  planner.CP.carFingerprint = platform
  planner.CP.openpilotLongitudinalControl = True
  planner.output_a_target = -0.7
  planner.v_desired_filter.x = 0.2
  return planner


def braking_state(v_ego=1.3, active=True, gas_pressed=False, brake_pressed=False, override=False):
  state = build_sm(v_ego, False)
  for service in ('carState', 'controlsState', 'carControl'):
    state[service] = state[service].as_builder()
  state['controlsState'].longControlState = LongCtrlState.pid if active else LongCtrlState.off
  state['carState'].gasPressed = gas_pressed
  state['carState'].brakePressed = brake_pressed
  state['carControl'].cruiseControl.override = override
  return state


@pytest.mark.parametrize('v_ego', [0.0, 0.1, 0.3, 1.2, 1.3, 2.0])
def test_braking_state_cannot_hide_low_speed_motion(v_ego):
  result = constrain_braking_speed(0.0, v_ego, -0.7)
  assert result == pytest.approx(max(0.0, v_ego - LOW_SPEED_STATE_ERROR))


@pytest.mark.parametrize('a_target', [0.0, 0.07, 0.3, 2.0])
def test_launch_requests_do_not_correct_state(a_target):
  assert constrain_braking_speed(0.2, 1.3, a_target) == 0.2


@pytest.mark.parametrize('v_ego', [5.0, 10.0, 33.0])
def test_high_speed_state_is_unchanged(v_ego):
  assert constrain_braking_speed(0.2, v_ego, -0.7) == 0.2


def test_small_noise_and_faster_planned_speed_are_unchanged():
  for v_ego in np.linspace(1.1, 1.49, 50):
    assert constrain_braking_speed(1.3, v_ego, -0.7) == 1.3


def test_correction_tapers_continuously_at_speed_boundaries():
  assert constrain_braking_speed(1.0, 3.5, -0.7) == pytest.approx(2.15)
  for boundary in (2.0, 5.0):
    below = constrain_braking_speed(0.2, boundary - 1e-6, -0.7)
    above = constrain_braking_speed(0.2, boundary + 1e-6, -0.7)
    assert abs(above - below) < 1e-5


def test_real_mpc_receives_bounded_measured_speed():
  planner = braking_planner()
  planner.update(braking_state())
  assert planner.mpc.x0[1] == pytest.approx(1.3 - LOW_SPEED_STATE_ERROR)
  assert planner.v_desired_trajectory[0] == pytest.approx(planner.mpc.x0[1])
  assert planner.mpc.solution_status == 0


@pytest.mark.parametrize('platform', [CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_HYBRID])
def test_other_platforms_keep_existing_filter(platform):
  planner = braking_planner(platform)
  planner.update(braking_state())
  expected = 0.2 + (1.3 - 0.2) * DT_MDL / (2.0 + DT_MDL)
  assert planner.mpc.x0[1] == pytest.approx(expected)


@pytest.mark.parametrize('overrides', [{'gas_pressed': True}, {'brake_pressed': True}, {'override': True}])
def test_driver_override_bypasses_correction(overrides):
  planner = braking_planner()
  planner.update(braking_state(**overrides))
  expected = 0.2 + (1.3 - 0.2) * DT_MDL / (2.0 + DT_MDL)
  assert planner.mpc.x0[1] == pytest.approx(expected)


def test_disabled_planner_resets_to_actual_speed():
  planner = braking_planner()
  planner.update(braking_state(active=False))
  assert planner.mpc.x0[1] == pytest.approx(1.3)


def test_stock_longitudinal_control_is_not_changed():
  planner = braking_planner()
  planner.CP.openpilotLongitudinalControl = False
  planner.update(braking_state())
  expected = 0.2 + (1.3 - 0.2) * DT_MDL / (2.0 + DT_MDL)
  assert planner.mpc.x0[1] == pytest.approx(expected)


def test_persistent_braking_shortfall_does_not_accumulate_state_error():
  planner = braking_planner()
  for _ in range(60):
    planner.output_a_target = -0.7
    planner.update(braking_state())
    assert planner.mpc.x0[1] >= 1.3 - LOW_SPEED_STATE_ERROR - 1e-6
    assert planner.mpc.solution_status == 0


@pytest.mark.parametrize('v_ego', [0.3, 0.8, 1.3, 2.5])
def test_correction_does_not_invent_no_lead_braking(v_ego):
  outputs = []
  for platform in (CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_ACCORD):
    planner = braking_planner(platform)
    commands = []
    for _ in range(40):
      planner.update(braking_state(v_ego))
      commands.append(planner.output_a_target)
    outputs.append(commands)
  np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-8)
  assert min(outputs[1]) >= 0.0


def test_stationary_lead_gets_more_braking_without_changing_lead_input():
  outputs = []
  for platform in (CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_ACCORD):
    planner = braking_planner(platform)
    state = braking_state()
    lead = state['radarState'].leadOne
    lead.present = True
    lead.dRel = 4.6
    lead.vRel = -1.3
    lead.vLead = 0.0
    lead.vLeadK = 0.0
    lead.aLeadTau = 1.5
    lead.modelProb = 0.99
    before = deepcopy(lead.to_dict())
    for _ in range(3):
      planner.update(state)
    assert lead.to_dict() == before
    assert planner.mpc.solution_status == 0
    outputs.append(planner.output_a_target)
  assert outputs[1] < outputs[0] - 0.1


def simulate_stop(platform, braking_bias):
  planner = braking_planner(platform)
  planner.CP.stopAccel = -2.0
  planner.CP.longitudinalTuning.kiBP = [0.0]
  planner.CP.longitudinalTuning.kiV = [0.0]
  controller = LongControl(planner.CP, custom.CarParamsSP.new_message())
  speed, acceleration, gap = 1.8, 0.0, 10.0
  planner.v_desired_filter.x = speed
  planner.output_a_target = 0.0
  commands = []
  for _ in range(int(25.0 / DT_MDL)):
    state = braking_state(speed)
    state['carState'].aEgo = acceleration
    state['carState'].standstill = speed < 0.01
    state['controlsState'].longControlState = controller.long_control_state
    lead = state['radarState'].leadOne
    lead.present = True
    lead.dRel = max(gap, 0.0)
    lead.vRel = -speed
    lead.vLead = 0.0
    lead.vLeadK = 0.0
    lead.aLeadTau = 1.5
    lead.modelProb = 0.99
    with np.errstate(divide='ignore'):
      planner.update(state)
    assert planner.mpc.solution_status == 0
    for _ in range(round(DT_MDL / DT_CTRL)):
      state['carState'].vEgo = speed
      state['carState'].aEgo = acceleration
      state['carState'].standstill = speed < 0.01
      command = float(controller.update(True, state['carState'], planner.output_a_target,
                                        planner.output_should_stop, (-3.5, 2.0)))
      delivered = command + braking_bias if command < 0.0 else command
      acceleration += (delivered - acceleration) * DT_CTRL / 0.5
      speed = max(0.0, speed + acceleration * DT_CTRL)
      if speed == 0.0:
        acceleration = 0.0
      gap -= speed * DT_CTRL
      commands.append(command)
  return {'gap': gap, 'speed': speed, 'minimum_command': min(commands)}


@pytest.mark.parametrize('braking_bias', [0.0, 0.2, 0.4])
def test_closed_loop_stop_with_bounded_braking_shortfall(braking_bias):
  baseline = simulate_stop(CAR.HONDA_CIVIC_BOSCH, braking_bias)
  corrected = simulate_stop(CAR.HONDA_ACCORD, braking_bias)
  assert corrected['speed'] == 0.0
  assert corrected['gap'] > 2.5
  assert corrected['gap'] >= baseline['gap'] - 0.05
  assert corrected['minimum_command'] >= -3.5
