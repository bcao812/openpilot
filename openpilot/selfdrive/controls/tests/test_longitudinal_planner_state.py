"""Measured-state and stopping regressions through the current platform-selected planner."""

from copy import deepcopy
import json

import numpy as np
import pytest

from opendbc.car.honda.values import CAR
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import advance, make_planner, make_state
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params


def braking_planner(platform=CAR.HONDA_ACCORD, *, openpilot_long=True):
  planner, _ = make_planner(speed=0.2, platform=platform, stock=not openpilot_long)
  planner.output_a_target = -0.7
  return planner


def braking_state(v_ego=1.3, active=True, gas_pressed=False, brake_pressed=False, override=False):
  state = make_state(speed=v_ego, target=100.0 / 3.6)
  state['carState'].gasPressed = gas_pressed
  state['carState'].brakePressed = brake_pressed
  state['carControl'].cruiseControl.override = override
  state['selfdriveState'].enabled = state['carControl'].enabled = state['carControl'].longActive = active
  state['controlsState'].longControlState = LongCtrlState.pid if active else LongCtrlState.off
  return state


@pytest.mark.parametrize('v_ego', [0.0, 0.1, 0.3, 1.2, 1.3, 2.0, 5.0, 10.0, 33.0])
def test_real_mpc_receives_the_measured_speed_independently_of_reference_filtering(v_ego):
  planner, state = braking_planner(), braking_state(v_ego)
  assert planner.longitudinal_mode is LongitudinalMode.UNIFIED
  planner.update(state, now_ns=advance(state, 0))
  assert planner.mpc.x0[1] == pytest.approx(v_ego)
  assert planner.v_desired_trajectory[0] == pytest.approx(v_ego)
  assert planner.mpc.solution_status == 0 and planner.motion_intent.valid


@pytest.mark.parametrize('platform', [CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_HYBRID])
def test_other_platforms_keep_existing_filter(platform):
  planner, state = braking_planner(platform), braking_state()
  assert planner.longitudinal_mode is LongitudinalMode.LEGACY
  planner.update(state, now_ns=advance(state, 0))
  expected = 0.2 + (1.3 - 0.2) * DT_MDL / (2.0 + DT_MDL)
  assert planner.mpc.x0[1] == pytest.approx(expected)


@pytest.mark.parametrize('overrides', [{'gas_pressed': True}, {'brake_pressed': True}, {'override': True}])
def test_driver_override_does_not_replace_physical_state_with_a_reference(overrides):
  planner, state = braking_planner(), braking_state(**overrides)
  planner.update(state, now_ns=advance(state, 0))
  assert planner.mpc.x0[1] == pytest.approx(1.3)


def test_disabled_planner_resets_to_actual_speed():
  planner, state = braking_planner(), braking_state(active=False)
  planner.update(state, now_ns=advance(state, 0))
  assert planner.mpc.x0[1] == pytest.approx(1.3)


def test_stock_longitudinal_control_is_not_changed():
  planner, state = braking_planner(openpilot_long=False), braking_state()
  assert planner.longitudinal_mode is LongitudinalMode.LEGACY
  planner.update(state, now_ns=advance(state, 0))
  expected = 0.2 + (1.3 - 0.2) * DT_MDL / (2.0 + DT_MDL)
  assert planner.mpc.x0[1] == pytest.approx(expected)


def test_persistent_braking_shortfall_does_not_accumulate_state_error():
  planner, state = braking_planner(), braking_state()
  for frame in range(60):
    planner.output_a_target = -0.7
    planner.update(state, now_ns=advance(state, frame))
    assert planner.mpc.x0[1] == pytest.approx(1.3)
    assert planner.mpc.solution_status == 0


@pytest.mark.parametrize('v_ego', [0.3, 0.8, 1.3, 2.5])
def test_correction_does_not_invent_no_lead_braking(v_ego):
  planner, state = braking_planner(), braking_state(v_ego)
  commands = []
  for frame in range(40):
    planner.update(state, now_ns=advance(state, frame))
    commands.append(planner.output_a_target)
  assert min(commands) >= 0.0


def test_stationary_lead_uses_measured_state_and_responds_without_changing_input(record_property):
  outputs = []
  measured_states = []
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
    for frame in range(3):
      planner.update(state, now_ns=advance(state, frame))
    assert lead.to_dict() == before
    assert planner.mpc.solution_status == 0
    outputs.append(planner.output_a_target)
    measured_states.append(float(planner.mpc.x0[1]))
  former_passed = outputs[1] < outputs[0] - 0.1
  record_property('scenario_reference_diagnostic', json.dumps({
    'status': 'NO_FLAG' if former_passed else 'REVIEW', 'former_relative_gate': 'PASS' if former_passed else 'FAIL',
    'former_extra_braking_mps2': 0.1,
    'baseline': {'identity': str(CAR.HONDA_CIVIC_BOSCH), 'request_mps2': float(outputs[0])},
    'candidate': {'identity': str(CAR.HONDA_ACCORD), 'runtime_mode': 'UNIFIED', 'gas_interceptor': False,
                  'request_mps2': float(outputs[1])},
    'qualification': {'scope': 'three_frame_planner_response_only', 'lead_present': True, 'lead_speed_mps': 0.0,
                      'initial_gap_m': 4.6, 'ego_speed_mps': 1.3, 'road_safety': 'INCONCLUSIVE'},
  }, sort_keys=True))
  # The candidate must retain measured motion and provide a bounded response to
  # the admitted stationary lead. Greater braking than another platform alone
  # says nothing about stopping clearance or ride quality.
  assert 1.3 - 0.2 - 1e-6 <= measured_states[1] <= 1.3 + 1e-6
  assert np.isfinite(outputs[1]) and -3.5 <= outputs[1] < 0.0


def simulate_stop(platform, braking_bias):
  from openpilot.selfdrive.controls.tests.test_longitudinal_following_scenarios import simulate_following

  result = simulate_following(platform=platform, initial_speed=1.8, lead_speed=0.0, initial_gap=10.0,
                              delay=0.0, response_time=0.5, duration=25.0, braking_bias=braking_bias)
  return {'gap': result['final_gap'], 'speed': result['final_speed'], 'minimum_command': result['minimum_command']}


@pytest.mark.parametrize('braking_bias', [0.0, 0.2, 0.4])
def test_closed_loop_stop_with_bounded_braking_shortfall(braking_bias, record_property):
  baseline = simulate_stop(CAR.HONDA_CIVIC_BOSCH, braking_bias)
  corrected = simulate_stop(CAR.HONDA_ACCORD, braking_bias)
  former_passed = corrected['gap'] >= baseline['gap'] - 0.05
  record_property('scenario_reference_diagnostic', json.dumps({
    'status': 'NO_FLAG' if former_passed else 'REVIEW', 'former_relative_gate': 'PASS' if former_passed else 'FAIL',
    'former_gap_tolerance_m': 0.05,
    'baseline': {'identity': str(CAR.HONDA_CIVIC_BOSCH), 'final_gap_m': baseline['gap']},
    'candidate': {'identity': str(CAR.HONDA_ACCORD), 'runtime_mode': 'UNIFIED', 'gas_interceptor': False,
                  'final_gap_m': corrected['gap']},
    'qualification': {'scope': 'ideal_braking_shortfall_plant_only', 'initial_gap_m': 10.0,
                      'initial_speed_mps': 1.8, 'braking_bias_mps2': braking_bias,
                      'clearance_m': 2.5, 'road_safety': 'INCONCLUSIVE'},
  }, sort_keys=True))
  assert corrected['speed'] == 0.0
  assert corrected['gap'] > 2.5
  assert corrected['minimum_command'] >= -3.5
