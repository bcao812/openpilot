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
from openpilot.selfdrive.controls.lib.cruise_response import get_cruise_coast_accel
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


# Version 9's request domain; using its wire value lets pre-fix red tests run.
CRUISE_REQUEST_KIND = 5


def ordinary_case(*, speed=12.0, target=12.3, initial_accel=0.0, trace=None):
  params = Params()
  for name in ('SmartCruiseControlVision', 'SmartCruiseControlMap', 'DynamicExperimentalControl'):
    params.put_bool(name, False, block=True)
  planner, owner = make_planner_owner(speed=speed, initial_accel=initial_accel, trace=trace)
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
    assert mpc.diagnostic_last_obstacle_summary['sourceRule'] == (
      'first_minimum_by_mpc_source_order;finite_continuation_without_cruise_speed_goal')
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
  mpc.diagnostic_obstacles_enabled = True
  for reference in (MpcReference(12.0), None):
    mpc.update(radar_state(), reference=reference)
    assert mpc.solution_status == 0
    assert mpc.diagnostic_last_obstacle_summary['sourceRule'] == 'first_minimum_by_mpc_source_order'


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


@pytest.mark.parametrize('condition', ['e2e', 'e2e_rest', 'e2e_no_forward', 'scc', 'force_decel'])
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


@pytest.mark.parametrize('slot,source,column', [
  ('leadOne', LongitudinalPlanSource.lead0, 0), ('leadTwo', LongitudinalPlanSource.lead1, 1),
])
def test_ordinary_acc_keeps_the_real_lead_goal_source_and_immediate_stronger_braking(slot, source, column):
  planner, owner, state = ordinary_case(speed=20.0, target=22.0)
  state['carState'].aEgo = -1.0
  lead = getattr(state['radarState'], slot)
  # Retain the established native-risk input from the two-slot lead regression.
  lead.present, lead.dRel, lead.vLead, lead.vRel = True, 44.0, 10.0, -10.0
  lead.aLeadK, lead.aLeadTau, lead.modelProb = -1.5, 1.5, 0.99
  with patch.object(planner.mpc, 'update', wraps=planner.mpc.update) as update, \
       patch.object(planner.mpc, 'observe_obstacles', wraps=planner.mpc.observe_obstacles) as observed:
    plan, outputs = consume(planner, owner, state, 0)
  assert update.call_args.kwargs['reference'].cruise_speed is None
  assert plan.solverStatus == 0 and plan.motionIntent.valid and not owner.motion_degraded
  obstacles, sources = observed.call_args.args
  assert obstacles.shape == (N + 1, 2) and sources[column] == source
  assert np.all(np.isfinite(obstacles[:, column])) and np.all(np.isinf(obstacles[:, 1 - column]))
  projected = planner.mpc.process_lead(lead)
  expected = projected[:, 0] + get_stopped_equivalence_factor(projected[:, 1])
  np.testing.assert_array_equal(planner.mpc.params[:, 2], expected)
  assert plan.reference.source == plan.longitudinalPlanSource == source
  assert plan.reference.accel < 0.0 < plan.reference.constraintAccel
  assert plan.aTarget == plan.reference.accel
  assert outputs[0] <= plan.aTarget and all(output <= plan.aTarget for output in outputs)


@pytest.mark.parametrize('slot,source', [('leadOne', LongitudinalPlanSource.lead0), ('leadTwo', LongitudinalPlanSource.lead1)])
def test_far_lead_appearance_loss_and_recovery_keep_one_cruise_response_and_existing_history(slot, source):
  planner, owner, state = ordinary_case(speed=20.23, target=20.0)
  lead = getattr(state['radarState'], slot)
  lead.dRel, lead.vLead, lead.vRel = 100.0, state['carState'].vEgo, 0.0
  lead.aLeadK, lead.aLeadTau, lead.modelProb = 0.0, 1.5, 0.99
  previous, instance = 0.0, None
  with patch.object(planner.mpc, 'update', wraps=planner.mpc.update) as update:
    for frame in range(20):
      lead.present = 4 <= frame < 16 and not 8 <= frame < 10
      if frame:
        state['controlsState'].longControlState = owner.long_control_state
      if frame == 12:
        state['carState'].vCruise = state['carState'].vCruiseCluster = 20.2 * 3.6
      plan, _ = consume(planner, owner, state, frame)
      assert update.call_args.kwargs['reference'].cruise_speed is None
      assert plan.solverStatus == 0 and plan.motionIntent.valid and not owner.motion_degraded
      previous = get_cruise_accel(False, planner.cruise_target.speed, state['carState'].vEgo, previous,
                                  0.0, planner.CP, planner.dt, ACCEL_MAX, True)
      # No-lead coast keeps its existing P guards; a real lead withdraws it.
      command = previous if lead.present else get_cruise_coast_accel(previous)
      assert planner.reference_planner.reference_accel > command
      assert plan.aTarget == plan.reference.constraintAccel == float(np.float32(command))
      assert plan.reference.source == (source if lead.present else LongitudinalPlanSource.cruise)
      assert plan.longitudinalPlanSource == LongitudinalPlanSource.cruise
      if instance is None:
        instance = plan.reference.producerInstanceMonoTime
      assert plan.reference.producerInstanceMonoTime == instance
      if frame:
        assert planner.mpc.diagnostic_reference_weights[0, 4] > 0.0


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
  # The request is below both independent candidates. A native reference above
  # its maximum does not make that nonbinding maximum own the final request.
  assert previous < get_max_accel(state['carState'].vEgo)
  assert plan.motionIntent.responseKind == CRUISE_REQUEST_KIND
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


def test_native_cruise_request_history_keeps_p_request_and_independent_departure_fact():
  planner, owner, state = ordinary_case(speed=3.0, target=2.6)
  previous, plans = 0.0, []
  for frame in range(20):
    if frame:
      state['controlsState'].longControlState = owner.long_control_state
    if frame == 8:
      state['carState'].vCruise = state['carState'].vCruiseCluster = 3.4 * 3.6
    plan, _ = consume(planner, owner, state, frame)
    previous = get_cruise_accel(False, planner.cruise_target.speed, state['carState'].vEgo, previous,
                                0.0, planner.CP, planner.dt, ACCEL_MAX, True)
    assert plan.solverStatus == 0 and plan.motionIntent.valid
    assert plan.aTarget == plan.reference.constraintAccel == float(np.float32(previous))
    assert plan.reference.accel > plan.aTarget
    assert previous < get_max_accel(state['carState'].vEgo)
    assert plan.reference.constraintSource == LongitudinalPlanSource.cruise
    plans.append(plan)
  assert min(plan.aTarget for plan in plans) < 0.0 < max(plan.aTarget for plan in plans)
  assert all(plan.motionIntent.responseKind == CRUISE_REQUEST_KIND for plan in plans)
  # This is an independent native trajectory fact, not new permission supplied
  # by the request domain. The owner still enforces target/freshness/HOLD gates.
  assert all(plan.motionIntent.releaseAllowed and not plan.motionIntent.movingLead for plan in plans)


def test_cruise_request_preserves_real_moving_lead_fact_without_manufacturing_it():
  planner, owner, state = ordinary_case(speed=1.0, target=2.0)
  lead = state['radarState'].leadOne
  lead.present, lead.dRel, lead.vLead, lead.vRel = True, 20.0, 1.5, 0.5
  lead.aLeadK, lead.aLeadTau, lead.modelProb = 0.0, 1.5, 0.99
  state['modelV2'].action.desiredAcceleration = 2.0
  plan, _ = consume(planner, owner, state, 0)
  assert plan.motionIntent.valid and not plan.motionIntent.reasons
  assert plan.reference.accel > plan.aTarget > 0.0
  assert plan.motionIntent.movingLead and plan.motionIntent.releaseAllowed
  assert plan.motionIntent.responseKind == CRUISE_REQUEST_KIND


@pytest.mark.parametrize('case,expected_kind', [
  ('request_wins', CRUISE_REQUEST_KIND), ('native_above_nonbinding_maximum', CRUISE_REQUEST_KIND),
  ('native_tie', MotionResponseKind.PURE_CRUISE), ('native_stronger', MotionResponseKind.PURE_CRUISE),
  ('hard_limit_tie', MotionResponseKind.ABSOLUTE_ACC), ('hard_limit_stronger', MotionResponseKind.ABSOLUTE_ACC),
])
def test_cruise_request_declaration_requires_the_strict_final_winner(case, expected_kind):
  initial = 0.4 if case == 'hard_limit_stronger' else 0.0
  planner, owner, state = ordinary_case(speed=3.0, target=4.0, initial_accel=initial)
  first_request = get_cruise_accel(False, 4.0, 3.0, 0.0, 0.0, planner.CP, planner.dt, ACCEL_MAX, True)
  maximum = None
  native = 0.5
  expected = first_request
  if case == 'native_above_nonbinding_maximum':
    native = 2.0
  elif case == 'native_tie':
    native = first_request
  elif case == 'native_stronger':
    native = expected = -0.5
  elif case == 'hard_limit_tie':
    native, maximum = 2.0, first_request
  elif case == 'hard_limit_stronger':
    native, maximum, expected = 2.0, 0.1, 0.1
  # The native solver still runs. Only its readout/current limit dependencies
  # are controlled to express exact ties, rather than relying on QP rounding.
  with ExitStack() as stack:
    stack.enter_context(patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.sample_reference_acceleration',
                              return_value=native))
    if maximum is not None:
      stack.enter_context(patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.cruise_accel_limit', return_value=maximum))
    plan, _ = consume(planner, owner, state, 0)
  assert plan.solverStatus == 0 and plan.motionIntent.valid and not plan.motionIntent.reasons
  assert plan.aTarget == float(np.float32(expected))
  assert plan.aTarget == min(plan.reference.accel, plan.reference.constraintAccel)
  assert plan.motionIntent.responseKind == expected_kind


@pytest.mark.parametrize('domain,expected_kind', [('e2e', MotionResponseKind.ABSOLUTE_ACC), ('scc', MotionResponseKind.NONE)])
def test_strict_cruise_scalar_winner_does_not_relabel_other_admitted_policy_domains(domain, expected_kind):
  planner, owner, state = ordinary_case(speed=3.0, target=4.0)
  with ExitStack() as stack:
    stack.enter_context(patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.sample_reference_acceleration', return_value=1.0))
    if domain == 'e2e':
      state['selfdriveState'].experimentalMode = True
    else:
      def selected_policy(instance, _state, _speed, accel, _cruise):
        instance.source = SPSource.sccVision
        instance.cruise_target = CruiseTarget(instance.source, 3.4, accel)
        return instance.cruise_target

      stack.enter_context(patch.object(LongitudinalPlannerSP, 'update_targets', new=selected_policy))
    plan, _ = consume(planner, owner, state, 0)
  assert plan.motionIntent.valid and plan.aTarget < plan.reference.accel
  assert plan.motionIntent.responseKind == expected_kind


def test_current_fcw_vetoes_cruise_request_domain_and_clearance_restores_it():
  planner, owner, state = ordinary_case(speed=3.0, target=4.0)
  native_update = planner.mpc.update
  crash_count = 3

  def update_current_crash_count(*args, **kwargs):
    native_update(*args, **kwargs)
    planner.mpc.crash_cnt = crash_count

  with patch.object(planner.mpc, 'update', side_effect=update_current_crash_count), \
       patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.sample_reference_acceleration', return_value=0.5):
    risk, _ = consume(planner, owner, state, 0)
    assert planner.fcw and risk.motionIntent.responseKind != CRUISE_REQUEST_KIND
    assert risk.aTarget == risk.reference.constraintAccel < risk.reference.accel
    crash_count = 0
    cleared, _ = consume(planner, owner, state, 1)
  assert not planner.fcw
  assert cleared.aTarget == cleared.reference.constraintAccel < cleared.reference.accel
  assert cleared.motionIntent.responseKind == CRUISE_REQUEST_KIND


def test_coast_execution_contract_keeps_priority_over_general_cruise_request():
  planner, owner, state = ordinary_case(speed=20.23, target=20.0, initial_accel=-0.23)
  plan, _ = consume(planner, owner, state, 0)
  assert plan.motionIntent.valid and plan.aTarget == float(np.float32(-0.19))
  assert plan.reference.accel > plan.aTarget
  assert plan.motionIntent.responseKind == MotionResponseKind.CRUISE_COAST
  assert not plan.motionIntent.movingLead and not plan.motionIntent.releaseAllowed
