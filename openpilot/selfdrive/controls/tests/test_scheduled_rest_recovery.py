"""Bounded recovery of an existing soft distance deficit.

The native cases combine recorded J451 initial/lead observations with explicitly
synthetic stop references. They are not a replay or a vehicle-response model.
"""

from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  CRASH_DISTANCE, LEAD_DANGER_FACTOR, MPC_INVALID_INPUT, N, PARAM_DIM, QP_FEASIBILITY_TOL,
  REST_STATE_TOL, T_IDXS, LongitudinalMpc,
)
from openpilot.selfdrive.controls.tests.test_mpc_rest_numerics import integrate
from openpilot.selfdrive.controls.tests.test_typed_model_reference import MODEL_TIMES, SOURCE_NS, evidence, stopping_curve
from openpilot.selfdrive.controls.tests.test_typed_reference_mpc import typed_reference
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state


@pytest.fixture(autouse=True)
def isolated_params():
  with OpenpilotPrefix():
    yield


def two_interval_stop():
  initial = np.array([0.0, 0.1, 0.0])
  h0, h1 = np.diff(T_IDXS)[:2]
  acceleration = -2.0 * initial[1] / (h0 + h1)
  jerks = np.zeros(N)
  jerks[:2] = acceleration / h0, -acceleration / h1
  return integrate(initial, jerks), jerks


def parameters_for_deficit(states, deficit):
  """Declare the distance budget independently of the normalized MPC residual."""
  parameters = np.zeros((N+1, PARAM_DIM))
  parameters[:, 0], parameters[:, 1] = ACCEL_MIN, ACCEL_MAX
  parameters[:, 4], parameters[:, 5] = 1.45, LEAD_DANGER_FACTOR
  desired = LEAD_DANGER_FACTOR * (states[:, 1]**2 / 5.0 + 1.45 * states[:, 1] + 6.0)
  parameters[:, 2] = states[:, 0] + desired - np.asarray(deficit)
  return parameters


def check(states, jerks, parameters, *, limit=1.0, tolerance=1e-10, node=2):
  return LongitudinalMpc.scheduled_rest_trajectory_error(
    states[0], states, jerks, node, parameters, tolerance=tolerance, recovery_position_limit=limit,
  )


def independent_maximum_position(states, jerks):
  positions = list(states[:, 0])
  for state, jerk, duration in zip(states[:-1], jerks, np.diff(T_IDXS), strict=True):
    position = np.polynomial.Polynomial([state[0], state[1], state[2] / 2.0, jerk / 6.0])
    points = [0.0, duration]
    for root in position.deriv().roots():
      if abs(np.imag(root)) < 1e-12 and 0.0 < np.real(root) < duration:
        points.append(float(np.real(root)))
    positions.extend(position(points))
  return float(np.max(positions))


def test_existing_deficit_can_recover_to_rest_without_changing_the_input():
  states, jerks = two_interval_stop()
  parameters = parameters_for_deficit(states, np.linspace(1.0, 0.0, N+1))
  before = states.copy(), jerks.copy(), parameters.copy()
  assert check(states, jerks, parameters, limit=None) == 'scheduled_rest_lead_danger_envelope_violated'
  assert check(states, jerks, parameters) == ''
  assert independent_maximum_position(states, jerks) < 1.0
  assert np.max(np.abs(states[2:, 1:])) <= REST_STATE_TOL
  for actual, expected in zip((states, jerks, parameters), before, strict=True):
    np.testing.assert_array_equal(actual, expected)


def test_constant_metre_deficit_is_allowed_while_braking_changes_its_normalization():
  states, jerks = two_interval_stop()
  parameters = parameters_for_deficit(states, 1.0)
  normalized_deficit = 1.0 / (states[:, 1] + 10.0)
  assert normalized_deficit[-1] > normalized_deficit[0]
  assert check(states, jerks, parameters) == ''


def test_recovery_cannot_cross_the_declared_raw_lead_position_limit():
  states, jerks = two_interval_stop()
  parameters = parameters_for_deficit(states, np.linspace(1.0, 0.0, N+1))
  maximum = independent_maximum_position(states, jerks)
  assert 0.0 < maximum < 1.0
  assert check(states, jerks, parameters, limit=maximum + 1e-4) == ''
  assert check(states, jerks, parameters, limit=maximum / 2.0) == 'scheduled_rest_recovery_position_limit_violated'


@pytest.mark.parametrize('early_deficit', [(1.0, 0.2, 0.5), (1.0, 0.0, 0.01)], ids=['worsens_after_improving', 'restarts_after_zero'])
def test_recovered_distance_cannot_be_spent_again(early_deficit):
  states, jerks = two_interval_stop()
  deficit = np.zeros(N+1)
  deficit[:3] = early_deficit
  parameters = parameters_for_deficit(states, deficit)
  assert check(states, jerks, parameters) == 'scheduled_rest_lead_danger_recovery_worsened'


def test_a_fixed_roundoff_allowance_cannot_accumulate_at_each_node():
  states, jerks = two_interval_stop()
  tolerance = 1e-7
  distance_budget = tolerance * (states[0, 1] + 10.0)
  bounded = np.ones(N+1)
  bounded[1:4] += np.array([0.5, 0.75, 0.25]) * distance_budget
  assert check(states, jerks, parameters_for_deficit(states, bounded), tolerance=tolerance) == ''
  accumulating = 1.0 + np.arange(N+1) * (0.5 * distance_budget)
  assert np.max(np.diff(accumulating)) < distance_budget
  assert accumulating[-1] - accumulating[0] > distance_budget
  assert check(states, jerks, parameters_for_deficit(states, accumulating), tolerance=tolerance) == 'scheduled_rest_lead_danger_recovery_worsened'


def test_increasing_speed_cannot_hide_a_larger_metre_deficit():
  initial = np.array([0.0, 0.1, 0.5])
  states, jerks = LongitudinalMpc.future_rest_primal(initial, 7)
  assert states[1, 1] > states[0, 1]
  deficit = np.ones(N+1)
  deficit[1] = (states[1, 1] + 10.0) / (states[0, 1] + 10.0)
  assert deficit[1] > deficit[0]
  assert deficit[1] / (states[1, 1] + 10.0) == pytest.approx(deficit[0] / (states[0, 1] + 10.0), abs=1e-15)
  assert check(states, jerks, parameters_for_deficit(states, deficit), node=7) == 'scheduled_rest_lead_danger_recovery_worsened'


@pytest.mark.parametrize('limit', [None, 0.0, -1.0, float('nan'), float('inf'), True, '1.0', 1.0+0.0j, np.array(1.0), [1.0]])
def test_existing_deficit_needs_a_finite_forward_geometry_limit(limit):
  states, jerks = two_interval_stop()
  parameters = parameters_for_deficit(states, np.linspace(1.0, 0.0, N+1))
  assert check(states, jerks, parameters, limit=limit) == 'scheduled_rest_lead_danger_envelope_violated'


@pytest.mark.parametrize('initial_deficit', [0.0, 0.5e-10 * 10.1], ids=['zero', 'within_original_roundoff'])
def test_without_initial_deficit_the_original_zero_margin_rule_still_applies(initial_deficit):
  states, jerks = two_interval_stop()
  deficit = np.zeros(N+1)
  deficit[0] = initial_deficit
  parameters = parameters_for_deficit(states, deficit)
  assert check(states, jerks, parameters, limit=None) == ''
  # A geometry limit does not introduce an unrelated restriction in the old branch.
  assert check(states, jerks, parameters, limit=states[2, 0] / 2.0) == ''
  deficit[3] = 0.01
  assert check(states, jerks, parameters_for_deficit(states, deficit)) == 'scheduled_rest_lead_danger_envelope_violated'


@pytest.mark.parametrize('violation', ['dynamics', 'acceleration'])
def test_recovery_does_not_bypass_original_trajectory_checks(violation):
  states, jerks = two_interval_stop()
  parameters = parameters_for_deficit(states, np.linspace(1.0, 0.0, N+1))
  assert check(states, jerks, parameters) == ''
  if violation == 'dynamics':
    states[3, 0] += 0.01
    expected = 'scheduled_rest_dynamics_residual_exceeded'
  else:
    parameters[:, 0] = 0.0
    expected = 'scheduled_rest_acceleration_envelope_violated'
  assert check(states, jerks, parameters) == expected


def test_allowed_residual_rest_speed_cannot_cross_the_limit_later_in_the_tail():
  initial = np.array([0.0, 0.5 * REST_STATE_TOL, 0.0])
  jerks = np.zeros(N)
  states = integrate(initial, jerks)
  parameters = parameters_for_deficit(states, np.linspace(1.0, 0.0, N+1))
  limit = 0.5 * states[-1, 0]
  assert states[2, 0] < limit < states[-1, 0]
  assert np.max(np.abs(states[2:, 1:])) <= REST_STATE_TOL
  assert check(states, jerks, parameters, limit=states[-1, 0] + 1e-4) == ''
  assert check(states, jerks, parameters, limit=limit) == 'scheduled_rest_recovery_position_limit_violated'


@pytest.mark.parametrize('two_roots', [False, True], ids=['linear_velocity_root', 'two_quadratic_velocity_roots'])
def test_position_maximum_includes_between_node_extrema(two_roots):
  duration = T_IDXS[-1] - T_IDXS[-2]
  prefix = T_IDXS[-2]
  speed, acceleration = (0.16, -1.0 / duration) if two_roots else (1.0, -2.0 / duration)
  initial = np.array([-speed * prefix + 0.5 * acceleration * prefix**2,
                      speed - acceleration * prefix, acceleration])
  jerks = np.zeros(N)
  jerks[-1] = 2.0 / duration**2 if two_roots else 0.0
  states = integrate(initial, jerks)
  expected = duration * 11.0 / 750.0 if two_roots else duration / 4.0
  assert np.max(states[:, 0]) < 1e-12
  assert independent_maximum_position(states, jerks) == pytest.approx(expected, abs=1e-12)
  assert LongitudinalMpc.trajectory_maximum_position(states, jerks) == pytest.approx(expected, abs=1e-12)


class GeometryPrepared(Exception):
  pass


def update_until_lead_processing(mpc, radar):
  # Stop after the real update's eligibility/geometry stage, before any native
  # object, parameters, template or solver is needed by this geometry test.
  with patch.object(mpc, 'process_lead', side_effect=GeometryPrepared), pytest.raises(GeometryPrepared):
    mpc.update(radar)
  return mpc.rest_recovery_position_limit


@pytest.mark.parametrize('closer_slot', [0, 1])
def test_recovery_geometry_uses_both_slots_before_impossible_gap_clipping(closer_slot):
  mpc = LongitudinalMpc.__new__(LongitudinalMpc)
  mpc.x0 = np.array([5.0, 10.0, 0.0])
  radar = radar_state(distance=3.0)
  radar.leadTwo.present, radar.leadTwo.dRel = True, 3.0
  leads = radar.leadOne, radar.leadTwo
  leads[closer_slot].dRel = 1.0
  assert all(mpc.process_lead(lead)[0, 0] > lead.dRel for lead in leads)
  assert update_until_lead_processing(mpc, radar) == pytest.approx(5.0 + 1.0 - CRASH_DISTANCE)
  assert not hasattr(mpc, 'solver')


def test_stopping_equivalence_cannot_replace_the_closer_raw_lead_position():
  mpc = LongitudinalMpc.__new__(LongitudinalMpc)
  mpc.x0 = np.array([0.0, 0.1, 0.0])
  radar = radar_state(distance=2.0, speed=10.0)
  radar.leadTwo.present, radar.leadTwo.dRel = True, 3.0
  assert radar.leadOne.dRel + radar.leadOne.vLead**2 / 5.0 > radar.leadTwo.dRel
  assert update_until_lead_processing(mpc, radar) == pytest.approx(2.0 - CRASH_DISTANCE)


@pytest.mark.parametrize('reverse_slot', [0, 1])
def test_either_reversing_present_lead_disqualifies_frozen_position_recovery(reverse_slot):
  mpc = LongitudinalMpc.__new__(LongitudinalMpc)
  mpc.x0 = np.array([0.0, 0.1, 0.0])
  radar = radar_state(distance=2.0, speed=0.2)
  radar.leadTwo.present, radar.leadTwo.dRel, radar.leadTwo.vLead = True, 3.0, 0.2
  (radar.leadOne, radar.leadTwo)[reverse_slot].vLead = -0.1
  assert update_until_lead_processing(mpc, radar) is None


@pytest.mark.parametrize('change', ['absent', 'reverse', 'too_close', 'invalid'])
def test_previous_geometry_is_cleared_before_every_update(change):
  mpc = LongitudinalMpc.__new__(LongitudinalMpc)
  mpc.x0 = np.array([0.0, 0.1, 0.0])
  radar = radar_state(distance=2.0, speed=0.2)
  assert update_until_lead_processing(mpc, radar) == pytest.approx(2.0 - CRASH_DISTANCE)
  if change == 'absent':
    radar.leadOne.present = False
  elif change == 'reverse':
    radar.leadOne.vLead = -0.1
  elif change == 'too_close':
    radar.leadOne.dRel = CRASH_DISTANCE
  else:
    radar.leadOne.dRel = float('nan')
    with patch.object(mpc, 'fail') as fail:
      mpc.update(radar)
    fail.assert_called_once_with(MPC_INVALID_INPUT)
    assert mpc.rest_recovery_position_limit is None
    return
  assert update_until_lead_processing(mpc, radar) is None


# Exact x0 and the MPC-consumed lead fields from J451. The model curves below
# are synthetic, intentionally separate from the unavailable full native output.
J451_INPUTS = (
  pytest.param(0.1264825463294983, -0.501275897026062, 201094, 0.03, (
    (3.215118408203125, 0.30021634697914124, 0.0512176938354969, 0.30000001192092896, 0.9999983310699463),
    (3.192348003387451, 0.2875174880027771, 0.05440123751759529, 0.30000001192092896, 0.9999950528144836),
  ), id='j451-warmup-1251780939573'),
  pytest.param(0.029483791440725327, -0.9772293567657471, 676510, 0.2, (
    (2.7992348670959473, 0.3701184093952179, 0.02068858966231346, 0.30000001192092896, 0.9999845027923584),
    (2.8297300338745117, 0.36603453755378723, 0.019130725413560867, 0.30000001192092896, 0.9999573826789856),
  ), id='j451-warmup-1258579766758'),
  pytest.param(0.07988708466291428, -1.2803713083267212, 860104, 0.03, (
    (1.795154094696045, 0.12751738727092743, -0.02908012829720974, 0.30000001192092896, 0.9999964237213135),
    (1.7273471355438232, 0.11635535955429077, -0.021531345322728157, 0.30000001192092896, 0.999988317489624),
  ), id='j451-score-1275881099876'),
)


def solve_synthetic_stop(speed, acceleration, age_ns, stop_time, radar):
  raw = np.where(MODEL_TIMES < 1.0, stopping_curve(speed, stop_time), 0.2 * (MODEL_TIMES - 1.0))
  typed = evidence(raw, speed, anchor_ns=SOURCE_NS + age_ns, now_ns=SOURCE_NS + age_ns)
  reference = replace(typed_reference(raw, speed, typed=typed), allow_future_rest_reschedule=True)
  original_goal = asdict(typed.rest_goal)
  mpc = LongitudinalMpc()
  mpc.set_weights(prev_accel_constraint=False)
  mpc.set_cur_state(speed, acceleration)
  mpc.update(radar, reference=reference)
  assert mpc.rest_goal is typed.rest_goal and asdict(mpc.rest_goal) == original_goal
  return mpc


def assert_independent_native_recovery(mpc, radar):
  present = [lead for lead in (radar.leadOne, radar.leadTwo) if lead.present]
  position_limit = mpc.x0[0] + min(lead.dRel for lead in present) - CRASH_DISTANCE
  maximum_position = independent_maximum_position(mpc.x_sol, mpc.j_solution)
  desired = LEAD_DANGER_FACTOR * (mpc.x_sol[:, 1]**2 / 5.0 + mpc.params[:, 4] * mpc.x_sol[:, 1] + 6.0)
  deficit = np.maximum(0.0, desired - (mpc.params[:, 2] - mpc.x_sol[:, 0]))
  initial_deficit = max(0.0, LEAD_DANGER_FACTOR * (mpc.x0[1]**2 / 5.0 + mpc.params[0, 4] * mpc.x0[1] + 6.0)
                        - (mpc.params[0, 2] - mpc.x0[0]))
  best = initial_deficit
  largest_rise = 0.0
  for value in deficit:
    largest_rise = max(largest_rise, value - best)
    best = min(best, value)
  diagnostic = {'failure': mpc.failure_reason, 'native_return': mpc.diagnostic_last_solver_return,
                'effective_node': mpc.effective_rest_node, 'maximum_position': maximum_position,
                'position_limit': position_limit, 'largest_deficit_rise_m': largest_rise,
                'rest_diagnostic': mpc.diagnostic_rest_goal}
  assert mpc.diagnostic_solve_calls == 1 and mpc.diagnostic_last_solver_return == 0, diagnostic
  assert mpc.solution_status == 0 and mpc.effective_rest_state == 'satisfied', diagnostic
  node = mpc.effective_rest_node
  assert mpc.rest_goal.native_node < node <= N
  assert mpc.rest_goal_state == 'infeasible' and mpc.rest_goal_reason == 'rest_goal_time_below_discrete_lower_bound'
  assert mpc.diagnostic_rest_goal['scheduledDangerPolicy'] == 'recover_initial_deficit'
  assert initial_deficit > 0.0
  assert mpc.rest_recovery_position_limit == pytest.approx(position_limit, abs=1e-12)
  assert maximum_position <= position_limit + QP_FEASIBILITY_TOL
  assert largest_rise <= QP_FEASIBILITY_TOL * (mpc.x0[1] + 10.0), diagnostic
  np.testing.assert_allclose(mpc.x_sol, integrate(mpc.x0, mpc.j_solution), atol=QP_FEASIBILITY_TOL, rtol=0.0)
  assert np.max(np.abs(mpc.x_sol[node:, 1:])) <= REST_STATE_TOL + QP_FEASIBILITY_TOL
  assert np.max(mpc.a_solution[node:]) <= QP_FEASIBILITY_TOL
  assert np.all(mpc.a_solution >= mpc.params[:, 0] - QP_FEASIBILITY_TOL)
  assert np.all(mpc.a_solution <= mpc.params[:, 1] + QP_FEASIBILITY_TOL)
  for state, jerk, duration in zip(mpc.x_sol[:-1], mpc.j_solution, np.diff(T_IDXS), strict=True):
    velocity = np.polynomial.Polynomial([state[1], state[2], 0.5 * jerk])
    points = [0.0, duration, *(root for root in velocity.deriv().roots() if np.isreal(root) and 0.0 < root < duration)]
    assert np.min(velocity(points)) >= -QP_FEASIBILITY_TOL


@pytest.mark.parametrize('speed,acceleration,age_ns,stop_time,lead_values', J451_INPUTS)
def test_recorded_j451_initial_state_and_present_leads_with_a_synthetic_stop_reference(speed, acceleration, age_ns, stop_time, lead_values):
  radar = radar_state()
  for lead, values in zip((radar.leadOne, radar.leadTwo), lead_values, strict=True):
    lead.present = True
    lead.dRel, lead.vLead, lead.aLeadK, lead.aLeadTau, lead.modelProb = values
  mpc = solve_synthetic_stop(speed, acceleration, age_ns, stop_time, radar)
  assert_independent_native_recovery(mpc, radar)


def test_stationary_close_lead_requires_a_valid_native_stop_without_spending_a_creep_allowance():
  # Counterexample control: unlike the three logged observations, this lead
  # stays still. A permitted 0.001 m/s rest tail must not conceal new approach.
  radar = radar_state(distance=1.73, speed=0.0, acceleration=0.0)
  mpc = solve_synthetic_stop(0.07988708466291428, -1.2803713083267212, 860104, 0.03, radar)
  assert_independent_native_recovery(mpc, radar)
