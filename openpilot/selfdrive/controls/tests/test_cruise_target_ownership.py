"""A single ordinary cruise response with a live, bounded native continuation.

P=e18b1de long_mpc.py:288-338 consumed two lead trajectories, using its finite
50 m / ego+10 m/s continuation when neither lead existed. These tests retain
that behavior without replacing a structured reference with reference=None.
They do not claim the measured initial state or native trajectory equals P.
"""

from contextlib import ExitStack
from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.longcontrol import MPC_FAILURE_ACCEL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_STOP_DISTANCE, MPC_INVALID_INPUT, N, STOP_DISTANCE, T_IDXS, LongitudinalMpc, LongitudinalPlanSource, MpcReference,
  get_stopped_equivalence_factor, get_T_FOLLOW,
)
from openpilot.selfdrive.controls.lib.longitudinal_planner import get_cruise_accel, get_max_accel
from openpilot.selfdrive.controls.lib.longitudinal_trace import LongitudinalTrace
from openpilot.selfdrive.controls.lib.model_reference import ModelReferenceAdapter
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionResponseKind
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import sample_reference_acceleration
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import advance, consume, make_handover_state, make_planner_owner
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import CruiseTarget, LongitudinalPlannerSP, LongitudinalPlanSource as SPSource


def ordinary_case(*, speed=12.0, target=12.3, trace=None):
  params = Params()
  for name in ('SmartCruiseControlVision', 'SmartCruiseControlMap', 'DynamicExperimentalControl'):
    params.put_bool(name, False, block=True)
  planner, owner = make_planner_owner(speed=speed, trace=trace)
  return planner, owner, make_handover_state(speed=speed, target=target)


def test_absent_cruise_goal_keeps_structured_bounds_finite_continuation_and_truthful_source():
  mpc = LongitudinalMpc()
  mpc.set_weights(prev_accel_constraint=False)
  mpc.set_cur_state(12.0, -0.58)
  mpc.diagnostic_obstacles_enabled = True
  lower, upper = np.linspace(-3.4, -3.0, N + 1), np.linspace(0.8, 1.2, N + 1)
  reference = MpcReference(None, accel_min=lower, accel_max=upper, comfort_stop_distance=4.0)
  continuation = mpc.process_lead(None)
  expected_distance = continuation[:, 0] + get_stopped_equivalence_factor(continuation[:, 1])
  for _ in range(8):
    mpc.update(radar_state(), reference=reference)
    assert mpc.solution_status == 0
    assert np.all(np.isfinite(mpc.x_sol)) and np.all(np.isfinite(mpc.params))
    np.testing.assert_array_equal(mpc.params[:, 0], lower)
    np.testing.assert_array_equal(mpc.params[:, 1], upper)
    np.testing.assert_array_equal(mpc.params[:, 2], expected_distance)
    assert np.all(mpc.params[:, COMFORT_STOP_DISTANCE] == STOP_DISTANCE)
    assert np.all(mpc.diagnostic_reference_weights[:, 0] == 3.0)
    assert mpc.source == LongitudinalPlanSource.cruise
    assert mpc.diagnostic_last_obstacle_summary['known']
    assert set(mpc.diagnostic_last_obstacle_summary['sources']) == {'cruise'}
  # Removing the speed goal must not turn the live distance-cost continuation
  # into a zero-reference request or an all-infinite invalid native problem.
  assert sample_reference_acceleration(mpc.a_solution, 0.55) > 0.0


def test_absent_cruise_goal_is_distinct_from_zero_and_does_not_erase_a_compatible_model_goal():
  mpc = LongitudinalMpc()
  mpc.set_cur_state(12.0, 0.0)
  following_time = get_T_FOLLOW(log.LongitudinalPersonality.standard)
  absent, sources, _, _ = mpc.reference_obstacles(MpcReference(None), following_time)
  assert absent == sources == []
  stopped, sources, _, _ = mpc.reference_obstacles(MpcReference(0.0), following_time)
  assert len(stopped) == 1 and sources == [LongitudinalPlanSource.cruise]
  model, sources, _, _ = mpc.reference_obstacles(MpcReference(
    None, model_position=8.0 * T_IDXS, model_velocity=np.full(N + 1, 8.0), model_acceleration=np.zeros(N + 1)), following_time)
  assert len(model) == 1 and sources == [LongitudinalPlanSource.e2e]
  assert np.all(np.isfinite(model))


@pytest.mark.parametrize('invalid', [
  {'accel_min': float('nan')}, {'accel_max': float('inf')}, {'accel_min': 1.0, 'accel_max': -1.0},
  {'cruise_min_accel': 0.1}, {'cruise_min_accel': np.zeros(N)}, {'model_velocity': np.zeros(N + 1)},
])
def test_absent_cruise_goal_does_not_bypass_existing_bounds_or_model_validation(invalid):
  mpc = LongitudinalMpc()
  with pytest.raises(ValueError):
    mpc.reference_obstacles(MpcReference(None, **invalid), get_T_FOLLOW(log.LongitudinalPersonality.standard))


def test_absent_cruise_goal_rejects_unvalidated_typed_only_problem_before_native_solve():
  mpc = LongitudinalMpc()
  mpc.set_cur_state(12.0, 0.0)
  evidence = ModelReferenceAdapter().build(
    np.full(len(ModelConstants.T_IDXS), 12.0), T_IDXS, 12.0,
    model_mono_time_ns=1_000_000_000, anchor_mono_time_ns=1_000_000_000, consumed_mono_time_ns=1_000_000_000,
  )
  assert evidence.unknown is None
  reference = MpcReference(None, typed_model=evidence)
  with pytest.raises(ValueError, match='Typed model references require an explicit cruise goal'):
    mpc.reference_obstacles(reference, get_T_FOLLOW(log.LongitudinalPersonality.standard))
  solver = Mock(wraps=mpc.solver)
  with patch.object(mpc, 'solver', solver):
    mpc.update(radar_state(), reference=reference)
  assert mpc.solution_status == MPC_INVALID_INPUT
  solver.solve.assert_not_called()


@pytest.mark.parametrize('unadmitted_risk', [False, True])
def test_ordinary_acc_has_one_cruise_owner_independent_of_unadmitted_raw_model_sign(unadmitted_risk):
  planner, _, state = ordinary_case()
  if unadmitted_risk:
    state['modelV2'].velocity.x = [0.0] * len(ModelConstants.T_IDXS)
    state['modelV2'].position.x = [0.0] * len(ModelConstants.T_IDXS)
    state['modelV2'].action.desiredAcceleration = -2.0
    state['modelV2'].action.shouldStop = True
  with patch.object(planner.mpc, 'update', wraps=planner.mpc.update) as update:
    planner.update(state, now_ns=advance(state, 0))
  reference = update.call_args.kwargs['reference']
  assert isinstance(reference, MpcReference) and reference.cruise_speed is None
  assert reference.typed_model is None
  assert reference.model_position is reference.model_velocity is reference.model_acceleration is None
  assert planner.mpc.solution_status == 0 and planner.motion_intent.valid


@pytest.mark.parametrize('condition', ['e2e', 'e2e_rest', 'e2e_no_forward', 'leadOne', 'leadTwo', 'scc', 'force_decel'])
def test_risk_and_policy_domains_keep_their_existing_numeric_cruise_goal(condition):
  planner, _, state = ordinary_case()
  with ExitStack() as stack:
    if condition.startswith('e2e'):
      state['selfdriveState'].experimentalMode = True
      times = np.asarray(ModelConstants.T_IDXS)
      if condition == 'e2e_rest':
        clipped = np.minimum(times, 6.0)
        state['modelV2'].velocity.x = np.maximum(12.0 - 2.0 * times, 0.0).tolist()
        state['modelV2'].position.x = (12.0 * clipped - clipped**2).tolist()
        state['modelV2'].acceleration.x = np.where(times < 6.0, -2.0, 0.0).tolist()
      elif condition == 'e2e_no_forward':
        state['modelV2'].velocity.x = [0.0] * len(times)
        state['modelV2'].position.x = [0.0] * len(times)
    elif condition in ('leadOne', 'leadTwo'):
      lead = getattr(state['radarState'], condition)
      lead.present, lead.dRel, lead.vLead, lead.vRel = True, 25.0, 10.0, -2.0
      lead.aLeadK, lead.aLeadTau, lead.modelProb = -1.0, 1.5, 0.99
    elif condition == 'scc':
      def selected_policy(instance, _state, _speed, accel, _cruise):
        instance.source = SPSource.sccVision
        instance.cruise_target = CruiseTarget(instance.source, 11.0, accel)
        return instance.cruise_target

      stack.enter_context(patch.object(LongitudinalPlannerSP, 'update_targets', new=selected_policy))
    elif condition == 'force_decel':
      state['controlsState'].forceDecel = True
    update = stack.enter_context(patch.object(planner.mpc, 'update', wraps=planner.mpc.update))
    planner.update(state, now_ns=advance(state, 0))
  reference = update.call_args.kwargs['reference']
  assert reference.cruise_speed is not None and np.isfinite(reference.cruise_speed)
  if condition == 'force_decel':
    assert reference.cruise_speed == 0.0
  elif condition in ('e2e_rest', 'e2e_no_forward'):
    assert reference.typed_model is not None


def test_decelerating_ego_below_set_speed_recovers_the_outer_cruise_request_through_live_native_arbitration():
  trace = LongitudinalTrace()
  # F's recorded pedal-reentry point at 8.25 s supplies a counterexample input,
  # not its lost solver warm start. Repeated fixed inputs define this complete
  # local history; the common closed-loop driver must assess delivered effects.
  speed, measured = 11.983823776, -0.583296895
  planner, owner, state = ordinary_case(speed=speed, target=12.0, trace=trace)
  state['carState'].aEgo = measured
  previous = 0.0
  with patch.object(planner.mpc, 'update', wraps=planner.mpc.update) as update:
    for frame in range(12):
      if frame:
        state['controlsState'].longControlState = owner.long_control_state
      plan, outputs = consume(planner, owner, state, frame)
      previous = get_cruise_accel(False, planner.cruise_target.speed, state['carState'].vEgo, previous,
                                  0.0, planner.CP, planner.dt, ACCEL_MAX, True)
      assert update.call_args.kwargs['reference'].cruise_speed is None
      assert plan.solverStatus == 0 and plan.motionIntent.valid and not owner.motion_degraded
      assert plan.reference.initialAEgo == state['carState'].aEgo
      assert plan.reference.source == LongitudinalPlanSource.cruise
      assert plan.reference.constraintAccel == float(np.float32(previous))
      assert plan.aTarget == min(plan.reference.accel, plan.reference.constraintAccel)
  assert previous > 0.0 and plan.reference.accel >= previous
  assert plan.aTarget == float(np.float32(previous)) and outputs[-1] > 0.0
  # Preserve the existing hard-cap response contract instead of relabeling a
  # native result that exceeds its independent acceleration limit.
  maximum = get_max_accel(state['carState'].vEgo)
  expected_kind = MotionResponseKind.ABSOLUTE_ACC if planner.reference_planner.reference_accel > maximum else MotionResponseKind.PURE_CRUISE
  assert plan.motionIntent.responseKind == expected_kind
  assert trace.failures == 0 and trace.queue


def test_absent_goal_keeps_native_failure_braking_envelope():
  planner, owner, state = ordinary_case()
  solver = Mock(wraps=planner.mpc.solver)
  solver.solve.return_value = 4
  with patch.object(planner.mpc, 'solver', solver), patch.object(planner.mpc, 'update', wraps=planner.mpc.update) as update:
    plan, outputs = consume(planner, owner, state, 0)
  assert update.call_args.kwargs['reference'].cruise_speed is None
  solver.solve.assert_called_once()
  assert plan.solverStatus == 4 and not plan.motionIntent.valid
  assert plan.motionIntent.brakingValid and plan.motionIntent.brakingAccel <= MPC_FAILURE_ACCEL
  assert all(ACCEL_MIN <= output <= MPC_FAILURE_ACCEL for output in outputs)
