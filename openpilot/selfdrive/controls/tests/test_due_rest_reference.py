"""Due-rest semantics, exact reachability, and the actual Honda command chain.

These fixed software inputs do not identify a physical stopping line or vehicle
response. Existing typed/lead/release counterexamples remain separate tests.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  MPC_GOAL_INFEASIBLE, N, REST_STATE_TOL, T_IDXS, LongitudinalMpc,
)
from openpilot.selfdrive.controls.lib.model_reference import ModelReferenceAdapter
from openpilot.selfdrive.controls.tests.test_typed_model_reference import MODEL_TIMES, SOURCE_NS, evidence, stopping_curve
from openpilot.selfdrive.controls.tests.test_typed_reference_mpc import StrictPlanTransportHarness, assign_curve, typed_reference
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_planner import advance, make_planner, make_state
from openpilot.selfdrive.controls.tests.test_unified_model_reference_contract import model_stop_curve
from openpilot.selfdrive.controls.tests.test_unified_startup import PLANNER_STEPS


@pytest.fixture(autouse=True)
def isolated_params():
  with OpenpilotPrefix():
    yield


def continued_evidence(raw, speed=10.0, *, adapter=None, elapsed_ns=0):
  adapter = adapter or ModelReferenceAdapter()
  prior = evidence(stopping_curve(10.0, 0.001), speed, adapter=adapter)
  source_ns = SOURCE_NS + 50_000_000
  current = evidence(raw, speed, adapter=adapter, model_ns=source_ns,
                     anchor_ns=source_ns + elapsed_ns, now_ns=source_ns + elapsed_ns)
  return current, prior


def typed_due_reference(speed):
  raw = np.zeros(len(MODEL_TIMES))
  typed, _ = continued_evidence(raw, speed)
  return typed_reference(raw, speed, typed=typed)


@pytest.mark.parametrize('speed', [0.0, 0.2, 10.0])
@pytest.mark.parametrize('elapsed_ns', [0, 50_000_000])
def test_exact_zero_continues_qualified_due_rest_with_both_source_epochs(speed, elapsed_ns):
  reference, prior = continued_evidence(np.zeros(len(MODEL_TIMES)), speed, elapsed_ns=elapsed_ns)
  assert reference.unknown is None and reference.no_forward and reference.moving is None
  assert reference.raw_motion.no_forward is not None and reference.raw_motion.rest_goal is not None
  assert reference.has_rest_intent and not reference.allows_external_departure
  goal = reference.rest_goal
  assert goal.source_node_interval == (0, 0) and goal.source_time_interval_s == (0.0, 0.0)
  assert goal.native_node == 0 and goal.native_time_s == 0.0
  assert goal.remaining_time_s == -elapsed_ns / 1e9
  assert goal.event_proxy_epoch_ns == SOURCE_NS + 50_000_000 and goal.nominal_position == 0.0
  assert goal.continued_from == (SOURCE_NS, prior.rest_goal.event_proxy_epoch_ns)
  assert goal.position_kind == 'nominal_preference' and not goal.future_forward


def test_isolated_zero_keeps_only_original_no_forward_even_with_nonrest_measured_state():
  reference = evidence(np.zeros(len(MODEL_TIMES)), 0.2)
  assert reference.no_forward and reference.moving is None
  assert not reference.has_rest_intent and reference.rest_goal is None
  assert not reference.allows_external_departure


@pytest.mark.parametrize('raw', [
  -np.ones(len(MODEL_TIMES)), stopping_curve(-0.4, 2.5),
  np.concatenate((np.zeros(2), -np.ones(len(MODEL_TIMES) - 2))),
  np.concatenate(([-1e-12], np.zeros(len(MODEL_TIMES) - 1))),
])
def test_no_positive_progress_with_negative_samples_does_not_invent_due_rest(raw):
  reference = evidence(raw, 0.0)
  assert reference.no_forward and reference.moving is None
  assert reference.rest_goal is None and not reference.has_rest_intent
  assert not reference.allows_external_departure


@pytest.mark.parametrize('speed', [1e-12, 0.2, 1.0])
def test_positive_flat_reference_keeps_independent_lead_departure_qualification(speed):
  reference = evidence(np.full(len(MODEL_TIMES), speed), 0.2)
  assert not reference.no_forward and not reference.has_rest_intent
  assert reference.raw_motion.kind == 'moving'
  assert not reference.allows_departure and reference.allows_external_departure


def test_late_zero_preserves_raw_rest_veto_without_backdating_a_native_goal():
  adapter = ModelReferenceAdapter()
  evidence(stopping_curve(10.0, 0.001), adapter=adapter)
  reference = evidence(np.zeros(len(MODEL_TIMES)), 0.2, adapter=adapter, model_ns=SOURCE_NS + 50_000_000,
                       anchor_ns=SOURCE_NS + 45_000_000, now_ns=SOURCE_NS + 55_000_000)
  assert reference.unknown.reason == 'model_proxy_after_initial_state'
  assert reference.no_forward and not reference.has_rest_intent
  assert reference.moving is None and reference.rest_goal is None
  assert not reference.allows_external_departure
  next_source = SOURCE_NS + 100_000_000
  recovered = evidence(np.zeros(len(MODEL_TIMES)), 0.2, adapter=adapter, model_ns=next_source,
                       anchor_ns=next_source, now_ns=next_source)
  assert recovered.rest_goal is None


def test_expired_zero_cannot_supply_raw_or_native_rest():
  reference = evidence(np.zeros(len(MODEL_TIMES)), 0.2, anchor_ns=SOURCE_NS + 300_000_000,
                       now_ns=SOURCE_NS + 300_000_000)
  assert reference.unknown.reason == 'expired_model_publication'
  assert reference.raw_motion is None and reference.rest_goal is None
  assert not reference.allows_external_departure


def test_new_forward_source_replaces_due_rest_without_a_persistent_stop_latch():
  adapter = ModelReferenceAdapter()
  previous, _ = continued_evidence(np.zeros(len(MODEL_TIMES)), 0.0, adapter=adapter)
  raw = 0.2 + 0.6 * MODEL_TIMES
  previous_source = SOURCE_NS + 50_000_000
  mutated = evidence(raw, 0.2, adapter=adapter, model_ns=previous_source,
                     anchor_ns=previous_source, now_ns=previous_source)
  next_source = SOURCE_NS + 100_000_000
  current = evidence(raw, 0.2, adapter=adapter, model_ns=next_source, anchor_ns=next_source, now_ns=next_source)
  assert previous.has_rest_intent
  assert mutated.unknown.reason == 'model_mutated_with_same_identity'
  assert not current.has_rest_intent and not current.no_forward and current.allows_departure


@pytest.mark.parametrize('interruption', ['negative', 'forward', 'late', 'expired', 'regressed', 'mutated', 'clear', 'not_admitted'])
def test_due_rest_continuity_cannot_survive_a_source_or_admission_interruption(interruption):
  adapter = ModelReferenceAdapter()
  zero = np.zeros(len(MODEL_TIMES))
  previous, _ = continued_evidence(zero, adapter=adapter)
  assert previous.rest_goal is not None
  next_ns = SOURCE_NS + 100_000_000
  if interruption == 'clear':
    adapter.clear_rest_continuity()
  elif interruption == 'not_admitted':
    adapter.build(stopping_curve(10.0, 0.001), T_IDXS, 10.0, model_mono_time_ns=next_ns,
                  anchor_mono_time_ns=next_ns, consumed_mono_time_ns=next_ns, retain_rest_continuity=False)
  elif interruption in ('negative', 'forward'):
    raw = stopping_curve(-0.4, 2.5) if interruption == 'negative' else 0.2 + 0.6 * MODEL_TIMES
    evidence(raw, adapter=adapter, model_ns=next_ns, anchor_ns=next_ns, now_ns=next_ns)
  elif interruption == 'late':
    evidence(zero, adapter=adapter, model_ns=next_ns, anchor_ns=next_ns - 1, now_ns=next_ns)
  elif interruption == 'expired':
    evidence(zero, adapter=adapter, model_ns=next_ns, anchor_ns=next_ns, now_ns=next_ns + 300_000_000)
  elif interruption == 'regressed':
    evidence(zero, adapter=adapter, model_ns=SOURCE_NS, anchor_ns=next_ns, now_ns=next_ns)
  elif interruption == 'mutated':
    evidence(np.ones(len(MODEL_TIMES)), adapter=adapter, model_ns=SOURCE_NS + 50_000_000,
             anchor_ns=next_ns, now_ns=next_ns)
  next_ns += 50_000_000
  current = evidence(zero, adapter=adapter, model_ns=next_ns, anchor_ns=next_ns, now_ns=next_ns)
  assert current.no_forward and current.rest_goal is None and not current.has_rest_intent


def test_repeated_zero_identity_does_not_renew_its_source_lease():
  adapter = ModelReferenceAdapter()
  raw = np.zeros(len(MODEL_TIMES))
  original, _ = continued_evidence(raw, adapter=adapter)
  source_ns = SOURCE_NS + 50_000_000
  for offset in (50_000_000, 150_000_000, 250_000_000):
    current = evidence(raw, adapter=adapter, model_ns=source_ns,
                       anchor_ns=source_ns + offset, now_ns=source_ns + offset)
    assert current.rest_goal.event_proxy_epoch_ns == original.rest_goal.event_proxy_epoch_ns
    assert current.rest_goal.remaining_time_s == -offset / 1e9
  expired = evidence(raw, adapter=adapter, model_ns=source_ns,
                     anchor_ns=source_ns + 250_000_001, now_ns=source_ns + 250_000_001)
  assert expired.rest_goal is None and expired.unknown.reason == 'expired_model_publication'


@pytest.mark.parametrize('not_due,delay_ns', [(True, 50_000_000), (False, 250_000_001)])
def test_not_due_or_old_previous_goal_does_not_turn_a_new_zero_source_into_hard_rest(not_due, delay_ns):
  adapter = ModelReferenceAdapter()
  evidence(stopping_curve(10.0, 5.0 if not_due else 0.001), adapter=adapter)
  current_ns = SOURCE_NS + delay_ns
  current = evidence(np.zeros(len(MODEL_TIMES)), adapter=adapter, model_ns=current_ns,
                     anchor_ns=current_ns, now_ns=current_ns)
  assert current.unknown is None and current.no_forward and current.rest_goal is None


@pytest.mark.parametrize('interruption', ['mode_closed', 'inactive', 'control_off', 'owner_unavailable',
                                        'unhealthy', 'invalid_cs_epoch', 'invalid_shape', 'model_restart',
                                        'nonfinite_pitch', 'nonfinite_throttle_probability'])
def test_production_admission_break_clears_rest_continuity_but_keeps_source_identity_guard(interruption):
  planner, _ = make_planner(speed=10.0)
  state = make_state(speed=10.0, target=20.0, experimental=True)
  assign_curve(state, stopping_curve(10.0, 0.001))
  planner.update(state, now_ns=advance(state, 0))
  adapter = planner.reference_planner.model_reference_adapter
  assert adapter._rest_continuity is not None
  previous_identity = adapter._last_sample
  now = advance(state, 1)
  if interruption == 'mode_closed':
    with patch.object(planner, 'is_e2e', return_value=False):
      planner.update(state, now_ns=now)
  elif interruption == 'unhealthy':
    with patch.object(state, 'all_checks', return_value=False):
      planner.update(state, now_ns=now)
  else:
    if interruption == 'inactive':
      state['carControl'].longActive = False
    elif interruption == 'control_off':
      state['controlsState'].longControlState = LongCtrlState.off
    elif interruption == 'owner_unavailable':
      state['controlsState'].motionState.enabled = False
    elif interruption == 'invalid_cs_epoch':
      state.logMonoTime['carState'] = now + 1
    elif interruption == 'invalid_shape':
      state['modelV2'].velocity.x = []
    elif interruption == 'model_restart':
      state.logMonoTime['modelV2'] = SOURCE_NS - 1
    elif interruption == 'nonfinite_pitch':
      state['carControl'].orientationNED = [0.0, float('nan'), 0.0]
    elif interruption == 'nonfinite_throttle_probability':
      state['modelV2'].meta.disengagePredictions.gasPressProbs = [1.0, float('nan')]
    planner.update(state, now_ns=now)
  assert adapter._rest_continuity is None
  assert adapter._last_sample is not None and adapter._last_sample[0] >= previous_identity[0]
  # Restore admission with a new source, not a reset of identity protection.
  state = make_state(speed=10.0, target=20.0, experimental=True)
  assign_curve(state, np.zeros(len(MODEL_TIMES)))
  planner.update(state, now_ns=advance(state, 2))
  assert planner.reference_planner.model_evidence.no_forward
  assert planner.reference_planner.model_evidence.rest_goal is None


# Exact states observed when the unchanged engineering plant was stationary.
# Their values are below the existing post-solve tolerance, but not at rest.
NEAR_ZERO_STATES = [
  (4.440742122824304e-5, 4.22437209635973e-4),
  (1.0985126209561713e-5, -5.070342740509659e-5),
  (1e-12, 0.0), (0.0, 1e-12), (0.0, -1e-12),
]


@pytest.mark.parametrize('speed,accel', NEAR_ZERO_STATES[:-1])
def test_due_rest_within_original_precision_uses_reachable_boxes_and_actual_native_acceptance(speed, accel):
  assert max(abs(speed), abs(accel)) < REST_STATE_TOL
  assert speed + 0.5 * accel * T_IDXS[1] != 0.0
  planner = LongitudinalMpc()
  planner.set_weights(prev_accel_constraint=False)
  planner.set_cur_state(speed, accel)
  planner.update(radar_state(distance=12.0, speed=0.0), reference=typed_due_reference(speed))
  assert planner.diagnostic_solve_calls == 1
  assert planner.diagnostic_last_solver_return == planner.last_solver_status == 0
  assert planner.solution_status == 0 and planner.rest_goal_state == 'satisfied'
  assert planner.rest_goal_braking_accel is None
  lower, upper = planner.diagnostic_reference_bounds
  np.testing.assert_array_equal(lower[1:, 0], 0.0)
  np.testing.assert_array_equal(lower[1:, 1], -REST_STATE_TOL)
  np.testing.assert_array_equal(upper[1:], REST_STATE_TOL)
  assert planner.diagnostic_rest_goal['maximumRestStateError'] <= REST_STATE_TOL
  assert planner.diagnostic_rest_goal['minimumContinuousVelocity'] >= -REST_STATE_TOL
  np.testing.assert_array_equal(planner.x0, [0.0, speed, accel])


@pytest.mark.parametrize('speed,accel', [
  (0.0, -1e-12), (1e-9, -REST_STATE_TOL), (-1e-12, 0.0),
  (1.001 * REST_STATE_TOL, 0.0), (0.0, 1.001 * REST_STATE_TOL),
])
def test_due_rest_cannot_relax_the_original_precision_or_invent_nonnegative_reachability(speed, accel):
  planner = LongitudinalMpc()
  planner.set_weights(prev_accel_constraint=False)
  planner.set_cur_state(speed, accel)
  planner.update(radar_state(), reference=typed_due_reference(speed))
  assert planner.solution_status == MPC_GOAL_INFEASIBLE
  assert planner.diagnostic_last_solver_return == 0 and planner.diagnostic_solve_calls == 1
  assert planner.rest_goal_state == 'infeasible' and planner.rest_goal_braking_accel == ACCEL_MIN
  lower, upper = planner.diagnostic_reference_bounds
  assert np.all(lower[1:, 1] < -REST_STATE_TOL) and np.all(upper[1:] > REST_STATE_TOL)
  assert planner.due_rest_primal(planner.x0) is None


@pytest.mark.parametrize('speed,accel', [
  (0.999 * REST_STATE_TOL, 0.999 * REST_STATE_TOL),
  (REST_STATE_TOL, REST_STATE_TOL), (REST_STATE_TOL, -REST_STATE_TOL),
  (0.0, REST_STATE_TOL), (3e-5, -REST_STATE_TOL),
])
def test_due_rest_precision_boundary_primal_obeys_dynamics_and_has_no_negative_velocity(speed, accel):
  initial = np.array([0.0, speed, accel])
  states, jerks = LongitudinalMpc.due_rest_primal(initial)
  np.testing.assert_array_equal(states[0], initial)
  assert np.all(states[:, 1] >= 0.0) and np.max(np.abs(states[:, 1:])) <= REST_STATE_TOL
  for stage, duration in enumerate(np.diff(T_IDXS)):
    x, v, a = states[stage]
    j = jerks[stage]
    np.testing.assert_allclose(states[stage+1], [x + v * duration + 0.5 * a * duration**2 + j * duration**3 / 6,
                                             v + a * duration + 0.5 * j * duration**2, a + j * duration],
                               atol=1e-17, rtol=1e-14)
    candidates = [v, states[stage+1, 1]]
    if j > 0.0 and 0.0 < -a / j < duration:
      candidates.append(v - a*a / (2*j))
    assert min(candidates) >= 0.0
  planner = LongitudinalMpc()
  planner.set_weights(prev_accel_constraint=False)
  planner.set_cur_state(speed, accel)
  planner.update(radar_state(), reference=typed_due_reference(speed))
  assert planner.diagnostic_last_solver_return == planner.solution_status == 0
  assert planner.rest_goal_state == 'satisfied'


@pytest.mark.parametrize('initial_guess', ['rest', 'moving'])
def test_exact_rest_uses_actual_native_constraints_and_passes_the_first_solve(initial_guess):
  planner = LongitudinalMpc()
  planner.set_weights(prev_accel_constraint=False)
  if initial_guess == 'moving':
    planner.set_cur_state(0.01, -3.5)
    planner.update(radar_state(), reference=typed_due_reference(0.01))
    assert planner.solution_status == MPC_GOAL_INFEASIBLE
    # A prior nonrest primal guess must not contaminate the first hard-rest
    # solve. This tests one solve, not repeated solves until the residual fades.
    for stage in range(N + 1):
      planner.solver.set(stage, 'x', np.array([0.0, 0.01, -3.5]))
      if stage < N:
        planner.solver.set(stage, 'u', np.array([1.0]))
  before = planner.diagnostic_solve_calls
  planner.set_cur_state(0.0, 0.0)
  planner.update(radar_state(), reference=typed_due_reference(0.0))
  assert planner.diagnostic_solve_calls == before + 1
  assert planner.diagnostic_last_solver_return == 0 and planner.solution_status == 0
  assert planner.rest_goal_state == 'satisfied' and planner.rest_goal_braking_accel is None
  np.testing.assert_allclose(planner.x_sol[:, 1:], 0.0, atol=REST_STATE_TOL, rtol=0.0)
  assert planner.diagnostic_rest_goal['minimumContinuousVelocity'] >= -REST_STATE_TOL


@pytest.mark.parametrize('speed,accel', NEAR_ZERO_STATES[:2])
def test_due_rest_native_can_return_to_new_forward_reference_without_reset(speed, accel):
  planner = LongitudinalMpc()
  planner.set_weights(prev_accel_constraint=False)
  planner.set_cur_state(speed, accel)
  planner.update(radar_state(), reference=typed_due_reference(speed))
  assert planner.solution_status == 0 and planner.rest_goal_state == 'satisfied'
  planner.set_cur_state(0.2, 0.0)
  planner.update(radar_state(distance=20.0, speed=1.5), reference=typed_reference(0.2 + 0.6 * MODEL_TIMES, 0.2))
  assert planner.solution_status == planner.diagnostic_last_solver_return == 0
  assert planner.rest_goal is None and planner.rest_goal_state == 'none'
  assert planner.rest_goal_braking_accel is None
  assert np.all(planner.diagnostic_reference_bounds[1][1:] > 0.0)


def test_actual_native_failure_is_not_renamed_as_due_rest_infeasibility():
  planner = LongitudinalMpc()
  planner.set_weights(prev_accel_constraint=False)
  planner.set_cur_state(*NEAR_ZERO_STATES[0])
  solver = Mock(wraps=planner.solver)
  solver.solve.return_value = 4
  with patch.object(planner, 'solver', solver):
    planner.update(radar_state(), reference=typed_due_reference(NEAR_ZERO_STATES[0][0]))
  assert planner.diagnostic_last_solver_return == planner.last_solver_status == planner.solution_status == 4
  assert planner.rest_goal_state == 'solver_failure' and planner.rest_goal_reason == 'solver_status'


def test_due_now_goal_keeps_required_braking_through_float32_owner_and_honda_can():
  harness = StrictPlanTransportHarness(speed=10.0)
  harness.engage()
  while harness.frame % PLANNER_STEPS:
    harness.step()
  harness.model_curve = model_stop_curve(10.0, 0.001)
  harness.action_accel = 0.5
  for _ in range(PLANNER_STEPS):
    harness.step()
  raw = np.zeros(len(MODEL_TIMES))
  harness.model_curve = (raw.copy(), raw, raw.copy())
  harness.action_accel = 0.5
  for _ in range(PLANNER_STEPS):
    harness.step()
  first_plan_time = int(harness.latest['longitudinalPlan'].logMonoTime)
  rows = [harness.step() for _ in range(30)]
  can_rows = [command for command in harness.native_commands if command['mono'] > rows[0]['mono']]
  plan = harness.latest['longitudinalPlan'].longitudinalPlan
  assert harness.planner.mpc.diagnostic_last_solver_return == 0
  assert plan.solverStatus == MPC_GOAL_INFEASIBLE
  assert not plan.motionIntent.valid and plan.motionIntent.brakingValid
  assert plan.motionIntent.brakingAccel == ACCEL_MIN and not plan.motionIntent.releaseAllowed
  assert harness.controls.sm.freq_ok['longitudinalPlan']
  assert harness.controls.reference_timing_check.valid
  assert all(row['consumed_plan_mono'] >= first_plan_time and row['long_active'] for row in rows)
  assert all(row['accel'] <= ACCEL_MIN + 1e-5 and not row['resume'] for row in rows)
  assert can_rows and all(command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0.0
                          and command['ACCEL_COMMAND'] <= ACCEL_MIN + 0.01 for command in can_rows)


@pytest.mark.parametrize('disturbance', ['positive_action', 'departing_lead', 'future_forward'])
def test_future_precision_goal_then_due_rest_keeps_actual_hold_despite_departure_disturbances(disturbance):
  harness = StrictPlanTransportHarness(speed=5.0)
  harness.hold()
  harness.source_state.cruiseState.standstill = False
  harness.source_state.standstill = False
  if disturbance in ('departing_lead', 'future_forward'):
    harness.departing_lead_speed = 1.5
  while harness.frame % PLANNER_STEPS:
    harness.step()
  harness.source_state.vEgo, harness.source_state.aEgo = 0.001, -0.01
  harness.model_curve = model_stop_curve(0.001, 0.001)
  harness.action_accel = 0.5
  for _ in range(PLANNER_STEPS):
    harness.step()
  assert harness.planner.mpc.rest_goal.native_node == 1
  assert harness.planner.mpc.solution_status == 0
  # The new source arrives after this qualified future event. Its current
  # rest veto must survive positive action and independent lead departure.
  harness.source_state.vEgo, harness.source_state.aEgo = NEAR_ZERO_STATES[0]
  raw = np.zeros(len(MODEL_TIMES))
  if disturbance == 'future_forward':
    raw = np.maximum(0.6 * (MODEL_TIMES - 1.0), 0.0)
  harness.model_curve = (raw * MODEL_TIMES, raw, np.gradient(raw, MODEL_TIMES))
  for _ in range(PLANNER_STEPS):
    harness.step()
  rows = [harness.step() for _ in range(30)]
  can_rows = [row for row in harness.native_commands if row['mono'] > rows[0]['mono']]
  mpc = harness.planner.mpc
  plan = harness.latest['longitudinalPlan'].longitudinalPlan
  assert mpc.rest_goal.native_node == 0 and mpc.diagnostic_last_solver_return == mpc.solution_status == 0
  assert mpc.rest_goal.future_forward == (disturbance == 'future_forward')
  assert plan.motionIntent.valid and not plan.motionIntent.releaseAllowed
  assert all(row['long_active'] and not row['resume'] and row['phase'] != 'release' for row in rows)
  assert can_rows and all(row['BRAKE_REQUEST'] and row['GAS_COMMAND'] <= 0.0 and not row['STANDSTILL_RELEASE']
                          for row in can_rows)


def test_microstate_due_rest_then_forward_restores_owner_release_in_original_point_eight_second_window():
  """A new veto: goal-failure braking must not erase the prior release window.

  Original v3 reaches owner RELEASE in this fixed transition. Keep this test
  failing if a due-rest candidate adds an earlier-layer recovery regression;
  neither extending the window nor the separate existing positive-CAN failure
  makes that regression pass.
  """
  harness = StrictPlanTransportHarness(speed=5.0)
  harness.hold()
  harness.source_state.cruiseState.standstill = False
  harness.source_state.standstill = False
  harness.departing_lead_speed = 1.5

  def phase(curve, ticks):
    while harness.frame % PLANNER_STEPS:
      harness.step()
    harness.model_curve = curve
    harness.action_accel = 0.5
    for _ in range(PLANNER_STEPS):
      harness.step()
    return [harness.step() for _ in range(ticks)]

  harness.source_state.vEgo = 10.0
  phase(model_stop_curve(10.0, 0.001), 5)
  harness.source_state.vEgo, harness.source_state.aEgo = NEAR_ZERO_STATES[0]
  zero = np.zeros(len(MODEL_TIMES))
  zero_rows = phase((zero.copy(), zero, zero.copy()), 80)
  assert all(not row['degraded'] and row['long_active'] for row in zero_rows)
  zero_can = [command for command in harness.native_commands if command['mono'] > zero_rows[0]['mono']]
  assert zero_can and all(command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0.0
                          and not command['STANDSTILL_RELEASE'] for command in zero_can)
  harness.source_state.vEgo, harness.source_state.aEgo = 0.2, 0.0
  raw = 0.2 + 0.6 * MODEL_TIMES
  rows = phase((raw * MODEL_TIMES, raw, np.gradient(raw, MODEL_TIMES)), 80)
  assert harness.planner.mpc.diagnostic_last_solver_return == harness.planner.mpc.solution_status == 0
  assert harness.planner.reference_planner.model_evidence.rest_goal is None
  assert harness.latest['longitudinalPlan'].longitudinalPlan.motionIntent.releaseAllowed
  assert any(row['phase'] == 'release' for row in rows)
