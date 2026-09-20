"""Action sampling in qualified low-speed following and at braking boundaries."""

from unittest.mock import patch

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, T_IDXS
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
  # Isolate speed/risk/acceleration gates at full confidence in these fixtures.
  # The native confidence-boundary regression below uses nonsaturated values.
  lead.aLeadK, lead.aLeadTau, lead.modelProb, lead.yRel = 0.0, 1.5, 1.0, 0.0
  return state


def solve_given_acceleration(planner, acceleration, speed, *args, **kwargs):
  """Preserve real risk geometry, then isolate the action-sampling trajectory."""
  LongitudinalMpc.update(planner.mpc, *args, **kwargs)
  assert planner.mpc.solution_status == 0
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
                    solve_given_acceleration(planner, acceleration, 3.0, *_args, **_kwargs)):
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
                    solve_given_acceleration(planner, acceleration, 3.0, *_args, **_kwargs)):
    planner.update(state, now_ns=advance(state, 0))
  assert planner.reference_planner.reference_accel == pytest.approx(
    -0.2 + jerk * (planner.CP.longitudinalActuatorDelay + DT_MDL))


def test_direct_action_sampling_preserves_the_independent_model_braking_envelope():
  planner, _ = make_planner(speed=5.0)
  state = make_state(speed=5.0, target=8.0, experimental=True)
  state['modelV2'].action.desiredAcceleration = -1.8
  with patch.object(planner.mpc, 'update', side_effect=lambda *_args, **_kwargs:
                    solve_given_acceleration(planner, np.full_like(T_IDXS, 0.4), 5.0, *_args, **_kwargs)):
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
                    solve_given_acceleration(planner, np.full_like(T_IDXS, -2.5), 30.0, *_args, **_kwargs)):
    planner.update(state, now_ns=advance(state, 0))
  assert planner.output_a_target == pytest.approx(-2.5)


@pytest.mark.parametrize('condition', [
  'stationary', 'closing', 'decelerating', 'uncertain_first', 'uncertain_second', 'invalid_probability',
  'short_gap', 'risk_deficit',
  'high_speed', 'no_lead', 'model_stop', 'model_brake', 'raw_unknown',
])
def test_braking_or_unqualified_following_retains_existing_delay_compensation(condition):
  speed = 12.0 if condition == 'high_speed' else 3.0
  planner, _ = make_planner(speed=speed)
  state = calm_following_state(speed, speed + 1.0, experimental=condition in ('model_stop', 'model_brake', 'raw_unknown'))
  lead = state['radarState'].leadOne
  if condition == 'stationary':
    lead.vLead, lead.vRel = 0.0, -speed
  elif condition == 'closing':
    lead.vLead, lead.vRel = speed - 0.6, -0.6
  elif condition == 'decelerating':
    lead.aLeadK = -0.6
  elif condition == 'uncertain_first':
    lead.modelProb = 0.949
  elif condition == 'uncertain_second':
    state['radarState'].leadTwo = lead
    state['radarState'].leadTwo.modelProb = 0.8
  elif condition == 'invalid_probability':
    lead.modelProb = 1.01
  elif condition == 'short_gap':
    lead.dRel = 4.9
  elif condition == 'risk_deficit':
    lead.dRel = 5.1  # Passes the fixed 5 m gate, but has insufficient native risk margin.
  elif condition == 'no_lead':
    lead.present = False
  elif condition == 'model_stop':
    state['modelV2'].action.shouldStop = True
  elif condition == 'model_brake':
    state['modelV2'].action.desiredAcceleration = -1.8
  elif condition == 'raw_unknown':
    state['modelV2'].velocity.x = np.linspace(-0.02, 5.0, len(ModelConstants.T_IDXS)).tolist()
  acceleration = 0.1 + 1.2 * np.exp(-T_IDXS / 0.18)
  state['carState'].aEgo = float(acceleration[0])
  with patch.object(planner.mpc, 'update', side_effect=lambda *_args, **_kwargs:
                    solve_given_acceleration(planner, acceleration, speed, *_args, **_kwargs)):
    planner.update(state, now_ns=advance(state, 0))
  control_times = ModelConstants.T_IDXS[:CONTROL_N]
  legacy = get_accel_from_plan(np.interp(control_times, T_IDXS, planner.mpc.v_solution),
                               np.interp(control_times, T_IDXS, acceleration), control_times,
                               action_t=planner.CP.longitudinalActuatorDelay + DT_MDL)
  assert legacy < -0.2  # The direct-only experiment would have lost this brake.
  assert planner.reference_planner.reference_accel == legacy
  assert planner.output_a_target <= legacy


def test_real_producer_blends_the_two_actions_at_partial_following_weight():
  speed = 3.0
  acceleration = 0.1 + 1.2 * np.exp(-T_IDXS / 0.18)
  planner, _ = make_planner(speed=speed, initial_accel=float(acceleration[0]))
  state = calm_following_state(speed, 4.0)
  state['carState'].aEgo = float(acceleration[0])
  state['radarState'].leadOne.vLead = 2.75
  state['radarState'].leadOne.vRel = -0.25
  with patch.object(planner.mpc, 'update', side_effect=lambda *_args, **_kwargs:
                    solve_given_acceleration(planner, acceleration, speed, *_args, **_kwargs)):
    planner.update(state, now_ns=advance(state, 0))
  control_times = ModelConstants.T_IDXS[:CONTROL_N]
  action_time = planner.CP.longitudinalActuatorDelay + DT_MDL
  legacy = get_accel_from_plan(np.interp(control_times, T_IDXS, planner.mpc.v_solution),
                               np.interp(control_times, T_IDXS, acceleration), control_times, action_t=action_time)
  native = np.interp(action_time, T_IDXS, acceleration)
  assert legacy < 0.0 < native
  assert planner.reference_planner.reference_accel == pytest.approx(0.5 * (legacy + native))


def test_real_producer_does_not_jump_between_actions_at_the_risk_boundary():
  speed = 3.0
  acceleration = 0.1 + 1.2 * np.exp(-T_IDXS / 0.18)
  targets = []
  # Equal 3 m/s speeds put the unchanged risk boundary at 7.3125 m.
  # Cross it by 2 mm without changing any velocity or acceleration evidence.
  for gap in (7.3115, 7.3135):
    planner, _ = make_planner(speed=speed, initial_accel=float(acceleration[0]))
    state = calm_following_state(speed, 4.0)
    state['selfdriveState'].personality = log.LongitudinalPersonality.standard
    state['carState'].aEgo = float(acceleration[0])
    lead = state['radarState'].leadOne
    lead.dRel, lead.vLead, lead.vRel = gap, speed, 0.0
    with patch.object(planner.mpc, 'update', side_effect=lambda *_args, **_kwargs:
                      solve_given_acceleration(planner, acceleration, speed, *_args, **_kwargs)):
      planner.update(state, now_ns=advance(state, 0))
    targets.append(planner.reference_planner.reference_accel)
  assert targets[0] < targets[1] < 0.0
  assert targets[1] - targets[0] < 0.01


@pytest.mark.parametrize('confidence_lead', ['leadOne', 'leadTwo'])
@pytest.mark.parametrize('initial_acceleration', [-0.68, 0.68])
def test_native_mpc_readout_is_continuous_at_either_lead_confidence_boundary(confidence_lead, initial_acceleration):
  references, targets, native_solutions = [], [], []
  # Same independent native solver histories. Only the final confidence differs;
  # the initial acceleration step is around the logged low-speed p95 magnitude.
  # The old hard gate changes the request by about 0.32 m/s² in this setting.
  for confidence in (0.94975, 0.95025):
    planner, _ = make_planner(speed=2.5)
    state = calm_following_state(2.5, 4.0)
    state['selfdriveState'].personality = log.LongitudinalPersonality.standard
    lead = state['radarState'].leadOne
    lead.dRel, lead.vLead, lead.vRel = 11.0, 2.5, 0.0
    lead.aLeadTau, lead.modelProb = 0.3, 0.999
    if confidence_lead == 'leadTwo':
      state['radarState'].leadTwo = lead
    for frame in range(40):
      planner.update(state, now_ns=advance(state, frame))
      assert planner.mpc.solution_status == 0
    state['carState'].aEgo = initial_acceleration
    getattr(state['radarState'], confidence_lead).modelProb = confidence
    # This calls the actual native MPC; no solved trajectory is substituted.
    planner.update(state, now_ns=advance(state, 40))
    assert planner.mpc.solution_status == 0
    references.append(planner.reference_planner.reference_accel)
    targets.append(planner.output_a_target)
    native_solutions.append(planner.mpc.a_solution.copy())
  np.testing.assert_array_equal(native_solutions[0], native_solutions[1])
  assert 0.0 < abs(references[1] - references[0]) < 0.01
  assert 0.0 < abs(targets[1] - targets[0]) < 0.01


def test_following_weight_tapers_before_closing_and_speed_boundaries():
  state = calm_following_state(3.0, 4.0)
  lead = state['radarState'].leadOne
  assert following_sample_weight(state['radarState'], 3.0, risk_margin=2.0) == 1.0
  for relative_speed, expected in ((0.0, 1.0), (-0.25, 0.5), (-0.5, 0.0)):
    lead.vLead, lead.vRel = 3.0 + relative_speed, relative_speed
    assert following_sample_weight(state['radarState'], 3.0, risk_margin=2.0) == pytest.approx(expected)
  for speed_kph, expected in ((15.0, 1.0), (17.5, 0.5), (20.0, 0.0)):
    speed = speed_kph / 3.6
    lead.vLead, lead.vRel = speed + 0.2, 0.2
    assert following_sample_weight(state['radarState'], speed, risk_margin=2.0) == pytest.approx(expected)


@pytest.mark.parametrize('risk_margin', [-2.3125, -0.001, float('nan'), float('inf')])
def test_close_or_unknown_risk_margin_cannot_acquire_following_sampling(risk_margin):
  state = calm_following_state(3.0, 4.0)
  state['radarState'].leadOne.dRel = 5.0
  assert following_sample_weight(state['radarState'], 3.0, risk_margin=risk_margin) == 0.0


@pytest.mark.parametrize('risk_margin,expected', [(0.0, 0.0), (0.125, 0.25), (0.25, 0.5), (0.5, 1.0)])
def test_sampling_enters_gradually_only_on_the_positive_side_of_original_risk(risk_margin, expected):
  state = calm_following_state(3.0, 4.0)
  assert following_sample_weight(state['radarState'], 3.0, risk_margin=risk_margin) == pytest.approx(expected)
