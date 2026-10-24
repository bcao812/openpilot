"""Native problem installation and domain acceptance, with independent cost/dynamics oracles."""

from dataclasses import replace
from decimal import Decimal, localcontext
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  MODEL_V_REF, MPC_GOAL_INFEASIBLE, MPC_GOAL_UNSATISFIED, N, REFERENCE_BOUND, T_IDXS, TYPED_MODEL_MODE,
  LongitudinalMpc, MpcOutcome, MpcReference,
)
from openpilot.selfdrive.controls.lib.model_reference import ModelReferenceEvidence, ModelSource, MovingPreference, NoForward, RestGoal


class NativeProblemRecorder:
  """Observe actual native calls without replacing its optimization or reset."""

  def __init__(self, native):
    self.native = native
    self.cost_writes = []
    self.constraint_writes = []
    self.costs = {}
    self.constraints = {}
    self.input_buffers = {}
    self.solved_costs = []
    self.before_solve = None
    self.failure = None

  def __getattr__(self, name):
    return getattr(self.native, name)

  def cost_set(self, stage, field, value):
    self.cost_writes.append((stage, field))
    self.costs[stage, field] = np.array(value, copy=True)
    if field == 'W':
      self.input_buffers[stage, field] = value
    return self.native.cost_set(stage, field, value)

  def constraints_set(self, stage, field, value, *args, **kwargs):
    self.constraint_writes.append((stage, field))
    self.constraints[stage, field] = np.array(value, copy=True)
    if field in ('C', 'D', 'lg', 'ug'):
      self.input_buffers[stage, field] = value
    return self.native.constraints_set(stage, field, value, *args, **kwargs)

  def solve(self):
    self.solved_costs.append({key: value.copy() for key, value in self.costs.items()})
    if self.before_solve is not None:
      self.before_solve()
    if self.failure == 'solver_status':
      return 4
    if self.failure == 'solver_exception':
      raise RuntimeError('injected native failure')
    return self.native.solve()

  def get(self, stage, field):
    value = self.native.get(stage, field)
    if stage == 1 and field == 'x' and self.failure in ('nonfinite', 'rest_defect'):
      value = value.copy()
      value[1] = np.nan if self.failure == 'nonfinite' else value[1] + 0.01
    return value


@pytest.fixture
def recorded_mpc():
  mpc = LongitudinalMpc()
  recorder = NativeProblemRecorder(mpc.solver)
  mpc.solver = recorder
  return mpc, recorder


def no_leads():
  return SimpleNamespace(leadOne=SimpleNamespace(present=False), leadTwo=SimpleNamespace(present=False))


def reference_case(kind):
  """Build MPC input directly, independently of model-adapter classification."""
  speed = 0.0 if kind in ('due_rest', 'no_forward') else 1.0 if kind in ('future_rest', 'scheduled_rest') else 10.0
  if kind == 'legacy':
    return speed, None
  if kind == 'cruise':
    return speed, MpcReference(20.0)
  source = ModelSource(1_000_000_000, 1_000_000_000, 1_000_000_000, proxy_age_s=0.0)
  moving, events = None, ()
  if kind in ('due_rest', 'no_forward'):
    events = (NoForward('no_positive_progress'),)
  else:
    supported = np.ones(N+1, dtype=bool)
    if kind == 'moving':
      supported[-2:] = False
    moving = MovingPreference(speed * T_IDXS, np.full(N+1, speed), np.zeros(N+1), supported)
  if kind in ('due_rest', 'future_rest', 'scheduled_rest'):
    node = {'due_rest': 0, 'future_rest': 6, 'scheduled_rest': 1}[kind]
    goal_time = float(T_IDXS[node])
    events += (RestGoal((0, 1), (0.0, goal_time), source.model_mono_time_ns + round(goal_time * 1e9),
                       goal_time, node, goal_time, 0.0),)
  typed = ModelReferenceEvidence(source, moving, events)
  return speed, MpcReference(20.0, typed_model=typed, allow_future_rest_reschedule=kind == 'scheduled_rest')


def assert_cost_contract(costs, kind, *, jerk_factor=1.0, history_scale=1.0, previous_accel=True):
  # This scalar objective is the pre-refactor engineering contract, not a call
  # to the builder. Distinct residuals also expose coordinate/order mistakes.
  residuals = np.vstack((np.eye(6), [1.3, -2.1, 0.7, -0.4, 0.2, -0.8]))
  typed = kind not in ('legacy', 'cruise')
  for stage, timestamp in enumerate(T_IDXS):
    supported = kind != 'moving' or stage < N-1
    matrix = costs[stage, 'W']
    np.testing.assert_array_equal(matrix - np.diag(np.diag(matrix)), 0.0)
    for residual in residuals:
      distance_cost = 0.0 if typed else 3.0 * residual[0]**2
      preference_cost = (5.0 * jerk_factor * (residual[2]**2 / 2.5**4 + residual[3]**2 / 2.5**2)
                         if typed and supported else 0.0)
      history = (200.0 * jerk_factor * history_scale * min(1.0, max(0.0, 2.0 - timestamp)) * residual[4]**2
                 if previous_accel else 0.0)
      jerk_cost = 5.0 * jerk_factor * residual[5]**2 if stage < N else 0.0
      vector = residual if stage < N else residual[:-1]
      assert matrix.shape == (len(vector), len(vector))
      assert vector @ matrix @ vector == pytest.approx(distance_cost + preference_cost + history + jerk_cost, abs=1e-13)
    if stage < N:
      np.testing.assert_array_equal(costs[stage, 'Zl'], [1e6, 1e6, 1e6, 100.0])


def assert_problem_write_order(recorder, rest_stages):
  assert recorder.cost_writes == [(stage, field) for stage in range(N) for field in ('W', 'Zl')] + [(N, 'W')]
  assert recorder.constraint_writes == (
    [(stage, field) for stage in rest_stages for field in ('C', 'D', 'lg', 'ug')]
    + [(stage, field) for stage in range(1, N+1) for field in ('lbx', 'ubx')]
    + [(0, 'lbx'), (0, 'ubx')]
  )


def assert_inactive_rest_constraints(constraints):
  for stage in range(N):
    np.testing.assert_array_equal(constraints[stage, 'C'], 0.0)
    np.testing.assert_array_equal(constraints[stage, 'D'], 0.0)
    np.testing.assert_array_equal(constraints[stage, 'lg'], -1.0)
    np.testing.assert_array_equal(constraints[stage, 'ug'], 1.0)


@pytest.mark.parametrize('kind', ['legacy', 'cruise', 'moving', 'no_forward', 'due_rest', 'future_rest', 'scheduled_rest'])
@pytest.mark.parametrize('personality,scale,previous,factor', [
  (log.LongitudinalPersonality.relaxed, 1.0, True, 1.0),
  (log.LongitudinalPersonality.standard, 0.35, True, 1.0),
  (log.LongitudinalPersonality.aggressive, 0.7, False, 0.5),
])
def test_final_costs_are_installed_once_for_each_cold_and_warm_update(recorded_mpc, kind, personality, scale, previous, factor):
  mpc, recorder = recorded_mpc
  speed, reference = reference_case(kind)
  assert mpc.outcome is MpcOutcome.NOT_SOLVED
  for warm in (False, True):
    recorder.cost_writes.clear()
    recorder.constraint_writes.clear()
    mpc.set_weights(previous, personality, a_change_scale=scale)
    assert not recorder.cost_writes
    mpc.set_cur_state(speed, 0.0)
    calls = len(recorder.solved_costs)
    mpc.update(no_leads(), personality, reference=reference)
    assert len(recorder.solved_costs) == calls + 1
    assert mpc.diagnostic_last_solver_return == 0 and mpc.outcome is MpcOutcome.ACCEPTED
    active_stages = len((mpc.diagnostic_rest_goal or {}).get('continuousSpeedSplitTimes', ()))
    assert_problem_write_order(recorder, range(active_stages) if warm else range(N))
    assert_cost_contract(recorder.solved_costs[-1], kind, jerk_factor=factor, history_scale=scale, previous_accel=previous)


def test_rest_and_no_rest_switches_clear_previous_costs_and_native_bounds(recorded_mpc):
  mpc, recorder = recorded_mpc
  # No fresh set_weights call is required for a subsequent update. In
  # particular, prior model costs and state bounds must not become base costs.
  previous_active_stages = N  # Initial native defaults have not been explicitly installed.
  for kind in ('moving', 'future_rest', 'cruise', 'future_rest', 'due_rest', 'no_forward', 'cruise', 'legacy', 'scheduled_rest', 'cruise'):
    speed, reference = reference_case(kind)
    recorder.cost_writes.clear()
    recorder.constraint_writes.clear()
    mpc.set_cur_state(speed, 0.0)
    mpc.update(no_leads(), reference=reference)
    assert mpc.outcome is MpcOutcome.ACCEPTED
    active_stages = len((mpc.diagnostic_rest_goal or {}).get('continuousSpeedSplitTimes', ()))
    assert_problem_write_order(recorder, range(max(previous_active_stages, active_stages)))
    previous_active_stages = active_stages
    assert_cost_contract(recorder.solved_costs[-1], kind)
    if kind in ('legacy', 'cruise'):
      assert mpc.rest_goal is None and mpc.effective_rest_node is None
      np.testing.assert_array_equal(mpc.params[:, MODEL_V_REF:TYPED_MODEL_MODE+1], 0.0)
      np.testing.assert_array_equal(mpc.yref, 0.0)
      for stage in range(1, N+1):
        np.testing.assert_array_equal(recorder.constraints[stage, 'lbx'], -REFERENCE_BOUND)
        np.testing.assert_array_equal(recorder.constraints[stage, 'ubx'], REFERENCE_BOUND)
      assert_inactive_rest_constraints(recorder.constraints)


@pytest.mark.parametrize('reset_between_updates', [False, True])
def test_native_problem_does_not_borrow_reused_python_buffers(recorded_mpc, reset_between_updates):
  mpc, recorder = recorded_mpc
  control = LongitudinalMpc()

  def overwrite_installed_inputs():
    # Exercise the actual native copy contract, rather than the recorder's
    # snapshots: destroy caller-owned data after every setter and before solve.
    # The following update must also clear every old matrix/constraint entry.
    assert len(recorder.input_buffers) == (N+1) + 4*N
    for value in recorder.input_buffers.values():
      assert value.dtype == np.float64 and value.flags.f_contiguous and value.flags.writeable
      value.fill(np.nan)

  recorder.before_solve = overwrite_installed_inputs
  for kind in ('future_rest', 'cruise', 'future_rest', 'due_rest', 'moving', 'cruise'):
    speed, reference = reference_case(kind)
    for candidate in (control, mpc):
      candidate.set_cur_state(speed, 0.0)
      candidate.update(no_leads(), reference=reference)
      assert candidate.outcome is MpcOutcome.ACCEPTED
    np.testing.assert_array_equal(mpc.x_sol, control.x_sol)
    np.testing.assert_array_equal(mpc.u_sol, control.u_sol)
    assert recorder.native.get_cost() == control.solver.get_cost()
    assert_cost_contract(recorder.solved_costs[-1], kind)
    if kind == 'cruise':
      assert_inactive_rest_constraints(recorder.constraints)
    if reset_between_updates:
      for candidate in (control, mpc):
        candidate.reset()
        assert candidate.outcome is MpcOutcome.NOT_SOLVED


@pytest.mark.parametrize('failure', ['invalid_input', 'solver_status', 'solver_exception', 'nonfinite'])
def test_failure_reset_restores_default_costs_without_a_new_setter(recorded_mpc, failure):
  mpc, recorder = recorded_mpc
  speed, reference = reference_case('future_rest')
  mpc.set_weights(False, log.LongitudinalPersonality.aggressive)
  mpc.set_cur_state(speed, 0.0)
  mpc.update(no_leads(), reference=reference)
  assert mpc.outcome is MpcOutcome.ACCEPTED
  recorder.failure = failure
  mpc.set_cur_state(speed, 0.0)
  calls = len(recorder.solved_costs)
  bad_reference = replace(reference, cruise_speed=np.nan) if failure == 'invalid_input' else reference
  mpc.update(no_leads(), reference=bad_reference)
  assert len(recorder.solved_costs) - calls == (0 if failure == 'invalid_input' else 1)
  assert mpc.outcome is (MpcOutcome.INPUT_REJECTED if failure == 'invalid_input' else MpcOutcome.SOLVER_FAULT)
  assert not mpc.previous_solution_valid
  recorder.failure = None
  recorder.cost_writes.clear()
  recorder.constraint_writes.clear()
  mpc.set_cur_state(10.0, 0.0)
  mpc.update(no_leads(), reference=MpcReference(20.0))
  assert mpc.outcome is MpcOutcome.ACCEPTED
  assert mpc.rest_goal is None and mpc.effective_rest_node is None
  assert_problem_write_order(recorder, ())  # reset already installed every default row.
  assert_cost_contract(recorder.solved_costs[-1], 'cruise')
  assert_inactive_rest_constraints(recorder.constraints)


def test_direct_run_applies_pending_base_costs_once(recorded_mpc):
  mpc, recorder = recorded_mpc
  mpc.set_cur_state(10.0, 0.0)
  mpc.update(no_leads())
  recorder.cost_writes.clear()
  mpc.set_weights(False, log.LongitudinalPersonality.aggressive)
  assert not recorder.cost_writes
  mpc.run()
  assert mpc.outcome is MpcOutcome.ACCEPTED
  assert_cost_contract(recorder.solved_costs[-1], 'legacy', jerk_factor=0.5, previous_accel=False)
  assert len([field for _, field in recorder.cost_writes if field == 'W']) == N+1
  recorder.cost_writes.clear()
  mpc.run()
  assert not recorder.cost_writes


@pytest.mark.parametrize('rejection', ['source_time', 'no_scheduling_witness', 'native_rest_defect'])
def test_domain_rest_rejection_keeps_native_success_and_goal_identity(recorded_mpc, rejection):
  mpc, recorder = recorded_mpc
  kind = 'scheduled_rest' if rejection == 'no_scheduling_witness' else 'due_rest'
  speed, reference = reference_case(kind)
  goal = reference.typed_model.rest_goal
  mpc.set_weights(False)
  mpc.set_cur_state(1.0 if rejection == 'source_time' else speed, 0.0)
  if rejection == 'no_scheduling_witness':
    with patch.object(mpc, 'future_rest_primal', return_value=None):
      mpc.update(no_leads(), reference=reference)
    # A missing finite-family witness rejects the requested contract; this
    # interface deliberately makes no claim of global reachability failure.
    assert mpc.effective_rest_node is None
    assert mpc.diagnostic_rest_goal['schedulingCandidates']
  else:
    recorder.failure = 'rest_defect' if rejection == 'native_rest_defect' else None
    mpc.update(no_leads(), reference=reference)
  assert mpc.rest_goal is goal
  assert mpc.outcome is MpcOutcome.REST_REJECTED
  assert mpc.diagnostic_last_solver_return == mpc.last_solver_status == 0
  assert mpc.solution_status == (MPC_GOAL_UNSATISFIED if rejection == 'native_rest_defect' else MPC_GOAL_INFEASIBLE)
  assert not mpc.previous_solution_valid
  mpc.reset()
  assert mpc.outcome is MpcOutcome.NOT_SOLVED


def test_shared_dynamics_residual_matches_high_precision_polynomial_oracle():
  def endpoint(state, jerk, duration):
    with localcontext() as context:
      context.prec = 80
      x, v, a, j, h = map(Decimal.from_float, map(float, (*state, jerk, duration)))
      return np.array([float(x + v*h + a*h*h/2 + j*h*h*h/6), float(v + a*h + j*h*h/2), float(a + j*h)])

  jerks = np.linspace(-2.0, 3.0, N)
  states = np.zeros((N+1, 3))
  states[0] = [2.0, 3.0, -0.4]
  for stage, duration in enumerate(np.diff(T_IDXS)):
    states[stage+1] = endpoint(states[stage], jerks[stage], duration)
  states[6] += [1e-5, -2e-5, 3e-5]
  before = states.copy()
  expected = np.array([states[stage+1] - endpoint(states[stage], jerks[stage], duration)
                       for stage, duration in enumerate(np.diff(T_IDXS))])
  actual = LongitudinalMpc.trajectory_dynamics_defects(states, jerks)
  np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0.0)
  np.testing.assert_array_equal(states, before)


def test_empty_object_problem_clears_previous_influence_without_array_allocation(recorded_mpc):
  mpc, _ = recorded_mpc
  mpc.set_cur_state(10.0, 0.0)
  mpc.update(no_leads(), reference=MpcReference(20.0))
  assert mpc.outcome is MpcOutcome.ACCEPTED
  assert all(token is None for token in mpc._influence_problem[1])
  mpc.object_influence_identities = (('previous-object', 1),)
  with patch('openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc.np.array',
             side_effect=AssertionError('empty object attribution allocated an array')):
    mpc.record_object_influence()
  assert mpc.object_influence_identities == ()
  assert mpc.outcome is MpcOutcome.ACCEPTED


@pytest.mark.parametrize('source', ['radar', 'vision'])
def test_physical_and_visual_tokens_still_receive_mpc_attribution(source):
  from openpilot.selfdrive.controls.tests.test_radar_fusion_mpc import EPOCH_NS, fused_state, object_lead, solve

  lead = object_lead(distance=30.0, speed=0.0, ego_speed=15.0, source=source)
  mpc = solve(fused_state(display=(lead,)), speed=15.0)
  expected = ('radar-test', 1) if source == 'radar' else ('vision', EPOCH_NS, 0)
  assert expected in mpc.object_influence_identities
  mpc.update(fused_state(), reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
  assert mpc.outcome is MpcOutcome.ACCEPTED
  assert mpc.object_influence_identities == ()


def test_current_rest_split_and_interval_mode_replace_installed_rows(recorded_mpc):
  mpc, recorder = recorded_mpc
  mpc.set_continuous_rest_bounds()
  assert_inactive_rest_constraints(recorder.constraints)
  mpc.rest_goal = SimpleNamespace(native_node=4)
  mpc.rest_goal_state = 'constrained'
  mpc.diagnostic_rest_goal = {}
  mpc.set_cur_state(1.0, 0.0)
  recorder.constraint_writes.clear()
  mpc.set_continuous_rest_bounds()
  assert recorder.constraint_writes == [(stage, field) for stage in range(3) for field in ('C', 'D', 'lg', 'ug')]
  duration = T_IDXS[1]
  np.testing.assert_array_equal(recorder.constraints[0, 'C'][0], 0.0)
  assert recorder.constraints[0, 'D'][1, 0] == duration**2 / 4.0

  # The current measured state changes the admissible first split. Active
  # rows must be submitted again even when the goal and all later rows match.
  mpc.set_cur_state(0.001, -1.0)
  recorder.constraint_writes.clear()
  mpc.set_continuous_rest_bounds()
  assert recorder.constraint_writes == [(stage, field) for stage in range(3) for field in ('C', 'D', 'lg', 'ug')]
  split = np.nextafter(0.002, 0.0)
  assert mpc.diagnostic_rest_goal['continuousSpeedSplitTimes'][0] == split
  assert recorder.constraints[0, 'C'][1, 2] == (duration + split) / 2.0
  assert recorder.constraints[0, 'D'][1, 0] == duration * split / 2.0

  # Interval-rest replaces the whole-interval certificate immediately, and
  # leaving that mode must reinstall current future-rest rows.
  mpc.interval_rest = SimpleNamespace(stop_time=0.001)
  recorder.constraint_writes.clear()
  mpc.set_continuous_rest_bounds()
  assert recorder.constraint_writes == [(stage, field) for stage in range(3) for field in ('C', 'D', 'lg', 'ug')]
  assert_inactive_rest_constraints(recorder.constraints)
  recorder.constraint_writes.clear()
  mpc.set_continuous_rest_bounds()
  assert not recorder.constraint_writes
  mpc.interval_rest = None
  mpc.set_continuous_rest_bounds()
  assert recorder.constraint_writes == [(stage, field) for stage in range(3) for field in ('C', 'D', 'lg', 'ug')]

  # A newly scheduled earlier node removes the no-longer-used stages while
  # preserving the current first-stage constraint and its diagnostics.
  mpc.effective_rest_node = 2
  mpc.effective_rest_state = 'constrained'
  recorder.constraint_writes.clear()
  mpc.set_continuous_rest_bounds()
  assert recorder.constraint_writes == [(stage, field) for stage in range(3) for field in ('C', 'D', 'lg', 'ug')]
  assert mpc.diagnostic_rest_goal['continuousSpeedSplitTimes'] == [split]
  for stage in (1, 2):
    np.testing.assert_array_equal(recorder.constraints[stage, 'C'], 0.0)
    np.testing.assert_array_equal(recorder.constraints[stage, 'D'], 0.0)
  recorder.constraint_writes.clear()
  mpc.set_continuous_rest_bounds()
  assert recorder.constraint_writes == [(0, field) for field in ('C', 'D', 'lg', 'ug')]


@pytest.mark.parametrize('field', ['C', 'D', 'lg', 'ug'])
@pytest.mark.parametrize('direction', ['enter', 'leave'])
def test_partial_constraint_write_cannot_keep_a_default_claim(recorded_mpc, field, direction):
  mpc, recorder = recorded_mpc
  mpc.set_continuous_rest_bounds()
  mpc.rest_goal = SimpleNamespace(native_node=3)
  mpc.rest_goal_state = 'constrained'
  mpc.diagnostic_rest_goal = {}
  mpc.set_cur_state(1.0, 0.0)
  if direction == 'leave':
    mpc.set_continuous_rest_bounds()
    mpc.rest_goal = None
    mpc.rest_goal_state = 'none'
  write = recorder.constraints_set

  def fail_after_write(stage, written_field, value):
    write(stage, written_field, value)
    if stage == 1 and written_field == field:
      raise RuntimeError('injected partial constraint write')

  with patch.object(recorder, 'constraints_set', side_effect=fail_after_write), pytest.raises(RuntimeError, match='partial constraint write'):
    mpc.set_continuous_rest_bounds()
  mpc.rest_goal = None
  mpc.rest_goal_state = 'none'
  recorder.constraint_writes.clear()
  mpc.set_continuous_rest_bounds()
  expected_stages = (0, 1) if direction == 'enter' else (1,)
  assert recorder.constraint_writes == [(stage, name) for stage in expected_stages for name in ('C', 'D', 'lg', 'ug')]
  assert_inactive_rest_constraints(recorder.constraints)
  recorder.constraint_writes.clear()
  mpc.set_continuous_rest_bounds()
  assert not recorder.constraint_writes


def test_reset_explicitly_clears_native_rows_before_another_update(recorded_mpc):
  mpc, recorder = recorded_mpc
  speed, reference = reference_case('future_rest')
  mpc.set_cur_state(speed, 0.0)
  mpc.update(no_leads(), reference=reference)
  assert mpc.outcome is MpcOutcome.ACCEPTED
  assert np.any(recorder.constraints[0, 'C'])
  recorder.constraint_writes.clear()
  mpc.reset()
  assert mpc.outcome is MpcOutcome.NOT_SOLVED
  assert_inactive_rest_constraints(recorder.constraints)
  assert recorder.constraint_writes == (
    [(stage, field) for stage in range(N) for field in ('C', 'D', 'lg', 'ug')]
    + [(stage, field) for stage in range(1, N+1) for field in ('lbx', 'ubx')]
  )
  recorder.constraint_writes.clear()
  mpc.set_continuous_rest_bounds()
  assert not recorder.constraint_writes
