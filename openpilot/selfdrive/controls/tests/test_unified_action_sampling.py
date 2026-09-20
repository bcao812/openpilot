"""A native motion reference must not become its opposite at the actuator."""

from unittest.mock import patch

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import following_sample_weight
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import advance, make_planner, make_state
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.modeld.constants import ModelConstants


def calm_following_state(speed, target, *, experimental=False):
  state = make_state(speed=speed, target=target, experimental=experimental)
  lead = state['radarState'].leadOne
  lead.present = True
  lead.dRel, lead.vLead, lead.vRel = 20.0, speed + 0.2, 0.2
  lead.aLeadK, lead.aLeadTau, lead.modelProb, lead.yRel = 0.0, 1.5, 0.999, 0.0
  return state


def solve_given_acceleration(planner, acceleration, speed):
  """Install a dynamically consistent native trajectory with changing jerk."""
  acceleration = np.asarray(acceleration, dtype=float)
  planner.mpc.a_solution = acceleration
  planner.mpc.v_solution = np.r_[speed, speed + np.cumsum(
    0.5 * (acceleration[:-1] + acceleration[1:]) * np.diff(T_IDXS))]
  planner.mpc.j_solution = np.diff(acceleration) / np.diff(T_IDXS)
  planner.mpc.source = log.LongitudinalPlan.LongitudinalPlanSource.cruise
  planner.mpc.solution_status = 0


@pytest.mark.parametrize('direction', [-1.0, 1.0])
def test_nonconstant_jerk_cannot_reverse_a_uniform_acceleration_request(direction):
  # Low-speed road oscillations exposed this shape: a large measured response
  # decays within the action horizon, while the requested future acceleration
  # keeps its sign. The old single-ramp inverse asks for the opposite effort.
  acceleration = direction * (0.1 + 1.2 * np.exp(-T_IDXS / 0.18))
  planner, _ = make_planner(speed=3.0, initial_accel=float(acceleration[0]))
  state = calm_following_state(speed=3.0, target=4.0)
  state['carState'].aEgo = float(acceleration[0])
  with patch.object(planner.mpc, 'update', side_effect=lambda *_args, **_kwargs:
                    solve_given_acceleration(planner, acceleration, 3.0)):
    planner.update(state, now_ns=advance(state, 0))
  assert planner.motion_intent.valid
  assert direction * planner.reference_planner.reference_accel > 0.0
  assert direction * planner.output_a_target > 0.0
  # It also remains inside the native trajectory's acceleration envelope.
  assert min(acceleration) <= planner.reference_planner.reference_accel <= max(acceleration)


def test_constant_jerk_retains_the_analytic_action_time_acceleration():
  planner, _ = make_planner(speed=3.0, initial_accel=-0.2)
  state = calm_following_state(speed=3.0, target=4.0)
  state['carState'].aEgo = -0.2
  jerk = 0.08
  acceleration = -0.2 + jerk * T_IDXS
  with patch.object(planner.mpc, 'update', side_effect=lambda *_args, **_kwargs:
                    solve_given_acceleration(planner, acceleration, 3.0)):
    planner.update(state, now_ns=advance(state, 0))
  assert planner.reference_planner.reference_accel == pytest.approx(
    -0.2 + jerk * (planner.CP.longitudinalActuatorDelay + DT_MDL))


def test_direct_action_sampling_preserves_the_independent_model_braking_envelope():
  planner, _ = make_planner(speed=5.0)
  state = make_state(speed=5.0, target=8.0, experimental=True)
  state['modelV2'].action.desiredAcceleration = -1.8
  with patch.object(planner.mpc, 'update', side_effect=lambda *_args, **_kwargs:
                    solve_given_acceleration(planner, np.full_like(T_IDXS, 0.4), 5.0)):
    planner.update(state, now_ns=advance(state, 0))
  assert planner.reference_planner.reference_accel == pytest.approx(0.4)
  assert planner.reference_planner.constraint_accel == pytest.approx(-1.8)
  assert planner.output_a_target == pytest.approx(-1.8)
  assert planner.reference_planner.reference_overridden


def test_sustained_native_braking_is_not_filtered_or_rate_limited_by_sampling():
  planner, _ = make_planner(speed=30.0, initial_accel=-2.5)
  state = make_state(speed=30.0, target=20.0)
  state['carState'].aEgo = -2.5
  with patch.object(planner.mpc, 'update', side_effect=lambda *_args, **_kwargs:
                    solve_given_acceleration(planner, np.full_like(T_IDXS, -2.5), 30.0)):
    planner.update(state, now_ns=advance(state, 0))
  assert planner.output_a_target == pytest.approx(-2.5)


@pytest.mark.parametrize('condition', ['stationary', 'closing', 'decelerating', 'uncertain_second', 'short_gap', 'high_speed', 'no_lead', 'model_stop'])
def test_braking_or_unqualified_following_retains_existing_delay_compensation(condition):
  speed = 12.0 if condition == 'high_speed' else 3.0
  planner, _ = make_planner(speed=speed)
  state = calm_following_state(speed, speed + 1.0, experimental=condition == 'model_stop')
  lead = state['radarState'].leadOne
  if condition == 'stationary':
    lead.vLead, lead.vRel = 0.0, -speed
  elif condition == 'closing':
    lead.vLead, lead.vRel = speed - 0.6, -0.6
  elif condition == 'decelerating':
    lead.aLeadK = -0.6
  elif condition == 'uncertain_second':
    state['radarState'].leadTwo = lead
    state['radarState'].leadTwo.modelProb = 0.8
  elif condition == 'short_gap':
    lead.dRel = 4.9
  elif condition == 'no_lead':
    lead.present = False
  elif condition == 'model_stop':
    state['modelV2'].action.shouldStop = True
  acceleration = 0.1 + 1.2 * np.exp(-T_IDXS / 0.18)
  state['carState'].aEgo = float(acceleration[0])
  with patch.object(planner.mpc, 'update', side_effect=lambda *_args, **_kwargs:
                    solve_given_acceleration(planner, acceleration, speed)):
    planner.update(state, now_ns=advance(state, 0))
  control_times = ModelConstants.T_IDXS[:CONTROL_N]
  legacy = get_accel_from_plan(np.interp(control_times, T_IDXS, planner.mpc.v_solution),
                               np.interp(control_times, T_IDXS, acceleration), control_times,
                               action_t=planner.CP.longitudinalActuatorDelay + DT_MDL)
  assert legacy < -0.2  # The direct-only experiment would have lost this brake.
  assert planner.reference_planner.reference_accel == legacy
  assert planner.output_a_target <= legacy


def test_following_weight_tapers_before_closing_and_speed_boundaries():
  state = calm_following_state(3.0, 4.0)
  lead = state['radarState'].leadOne
  assert following_sample_weight(state['radarState'], 3.0) == 1.0
  for relative_speed, expected in ((0.0, 1.0), (-0.25, 0.5), (-0.5, 0.0)):
    lead.vLead, lead.vRel = 3.0 + relative_speed, relative_speed
    assert following_sample_weight(state['radarState'], 3.0) == pytest.approx(expected)
  for speed_kph, expected in ((15.0, 1.0), (17.5, 0.5), (20.0, 0.0)):
    speed = speed_kph / 3.6
    lead.vLead, lead.vRel = speed + 0.2, 0.2
    assert following_sample_weight(state['radarState'], speed) == pytest.approx(expected)
