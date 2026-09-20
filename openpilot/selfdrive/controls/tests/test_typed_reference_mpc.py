"""Typed evidence through the actual acados solver, planner and stop owner.

These constructed cases are software counterexamples, not road validation.
The original eight model/native/owner/Honda CAN cases remain unmodified.
"""

from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import STOP_SPEED
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState, MPC_FAILURE_ACCEL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  J_EGO_COST, MODEL_A_REF, MODEL_BRAKING_PRIORITY, MODEL_V_REF, MPC_GOAL_INFEASIBLE,
  PARAM_DIM, REST_STATE_TOL, T_IDXS, TYPED_MODEL_MODE, LongitudinalMpc, LongitudinalPlanSource, MpcReference,
)
from openpilot.selfdrive.controls.lib.model_reference import moving_preference
from openpilot.selfdrive.controls.lib.motion_supervisor import STARTING_ACCEL_RATE, StopReason
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import model_reference_from_velocity
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_typed_model_reference import MODEL_TIMES, SOURCE_NS, evidence, stopping_curve
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import action, radar_state, solve_reference
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import Publisher, advance, make_planner, make_state
from openpilot.selfdrive.controls.tests.test_unified_model_reference_contract import ModelReferenceHarness, held_harness, model_stop_curve
from openpilot.selfdrive.controls.tests.test_unified_startup import PLANNER_STEPS
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import ControlsHarness, new_event


def typed_reference(raw, speed, *, preference_time=2.5, cruise=30.0, typed=None):
  position, velocity, acceleration = model_reference_from_velocity(raw)
  return MpcReference(cruise, model_position=position, model_velocity=velocity, model_acceleration=acceleration,
                      typed_model=evidence(raw, speed) if typed is None else typed, preference_time_s=preference_time)


def assign_curve(state, raw):
  model = state['modelV2']
  model.velocity.x = list(map(float, raw))
  # These two independent model heads remain valid compatibility inputs. The
  # new preference derives its own consistent integral and slopes from raw v.
  model.position.x = (np.maximum(raw, 0.0) * MODEL_TIMES).tolist()
  model.acceleration.x = np.gradient(raw, MODEL_TIMES).tolist()
  model.action.desiredAcceleration = 0.5
  model.action.shouldStop = False


def assign_lead(state, *, distance, speed, relative_speed):
  if hasattr(state['radarState'], 'as_builder'):
    state['radarState'] = state['radarState'].as_builder()
  lead = state['radarState'].leadOne
  lead.present = True
  lead.dRel, lead.vLead, lead.vRel = distance, speed, relative_speed
  lead.aLeadK, lead.aLeadTau, lead.yRel = 0.0, 1.5, 0.0
  lead.modelProb = 0.99


class StrictPlanTransportHarness(ModelReferenceHarness):
  """Deliver a 20 Hz publication once while consuming its cached plan at 100 Hz.

  The base bounded harness excludes average frequency checks. Fault braking
  explicitly requires healthy LP frequency, so repeating a held packet into
  SubMaster would be a different, unhealthy transport fixture.
  """

  model_publication_after_state = False
  departing_lead_speed = None

  def feed(self, state, overrides):
    if state is self.planner_state and self.model_publication_after_state:
      source = overrides['carState'].as_builder()
      source.logMonoTime -= 5_000_000
      overrides = overrides | {'carState': source}
    if state is self.planner_state and self.departing_lead_speed is not None:
      radar = new_event('radarState', self.now_ns)
      lead = radar.radarState.leadOne
      lead.present = True
      lead.dRel, lead.vLead = 20.0, self.departing_lead_speed
      lead.vRel = self.departing_lead_speed - float(self.source_state.vEgo)
      lead.aLeadK, lead.aLeadTau, lead.modelProb, lead.yRel = 0.0, 1.5, 0.99, 0.0
      overrides = overrides | {'radarState': radar}
    if state is not self.controls.sm:
      return super().feed(state, overrides)
    original_update = state.update_msgs
    previous_plan_time = state.logMonoTime['longitudinalPlan']

    def ingest_new_publications(timestamp, incoming):
      messages = [message for message in incoming
                  if message.which() != 'longitudinalPlan' or message.logMonoTime != previous_plan_time]
      return original_update(timestamp, messages)

    with patch.object(state, 'update_msgs', side_effect=ingest_new_publications):
      return super().feed(state, overrides)


def test_matching_generated_solver_has_real_path_and_terminal_slots():
  model_path = Path(__file__).resolve().parents[1] / 'lib/longitudinal_mpc_lib/acados_ocp_long.json'
  generated = json.loads(model_path.read_text())
  assert generated['dims']['nx'] == 3 and generated['dims']['nu'] == 1
  assert generated['dims']['np'] == PARAM_DIM == 11
  assert generated['dims']['nbx'] == generated['dims']['nbx_e'] == 2
  assert generated['constraints']['idxbx'] == generated['constraints']['idxbx_e'] == [1, 2]


@pytest.mark.parametrize('preference_time', [1.25, 2.5, 5.0])
@pytest.mark.parametrize('after_anchor', [False, True])
def test_nonzero_native_weights_and_changed_targets_change_the_solution(preference_time, after_anchor):
  timing = {'model_ns': SOURCE_NS + 5_000_000, 'now_ns': SOURCE_NS + 10_000_000} if after_anchor else {}
  slower_raw, faster_raw = 10.0 - 0.6 * MODEL_TIMES, 10.0 + 0.6 * MODEL_TIMES
  decelerating = typed_reference(slower_raw, 10.0, preference_time=preference_time, typed=evidence(slower_raw, 10.0, **timing))
  accelerating = typed_reference(faster_raw, 10.0, preference_time=preference_time, typed=evidence(faster_raw, 10.0, **timing))
  slower = solve_reference(10.0, decelerating)
  faster = LongitudinalMpc()
  faster.solver = Mock(wraps=faster.solver)
  faster.set_cur_state(10.0, 0.0)
  faster.set_weights(prev_accel_constraint=False)
  for _ in range(12):
    faster.update(radar_state(), reference=accelerating)
    assert faster.solution_status == 0
  native_weights = [call.args[2] for call in faster.solver.cost_set.call_args_list
                    if call.args[:2] == (1, 'W')][-1]
  assert native_weights[1, 1] == 0.0
  assert native_weights[2, 2] == pytest.approx(J_EGO_COST / preference_time**4)
  assert native_weights[3, 3] == pytest.approx(J_EGO_COST / preference_time**2)
  assert native_weights[5, 5] == J_EGO_COST
  assert action(slower) < 0.0 < action(faster)
  assert action(faster) - action(slower) > 0.1
  # v/a are zero-target residuals against the separately transmitted native
  # reference parameters, so one-sided priority can retain stronger braking.
  np.testing.assert_array_equal(faster.yref[:, 2:4], 0.0)
  np.testing.assert_allclose(faster.params[:, MODEL_V_REF], accelerating.typed_model.moving.velocity)


def test_constant_model_speed_bias_cannot_create_a_model_obstacle_or_change_native_problem():
  raw = np.full(len(MODEL_TIMES), 10.0)
  original = typed_reference(raw, 10.0)
  fixed_events = replace(original.typed_model, moving=moving_preference(raw - 0.6, T_IDXS, 10.0), raw_velocity=raw - 0.6)
  shifted = typed_reference(raw - 0.6, 10.0, typed=fixed_events)
  before, after = solve_reference(10.0, original), solve_reference(10.0, shifted)
  for name in ('params', 'yref', 'x_sol', 'u_sol'):
    np.testing.assert_allclose(getattr(before, name), getattr(after, name), atol=1e-9, rtol=0.0)
  assert abs(action(after)) < 1e-4
  # Only the finite cruise envelope reaches p[2]; no model origin shift is
  # converted into a following-distance or collision-risk input.
  without_model = solve_reference(10.0, MpcReference(30.0))
  np.testing.assert_array_equal(after.params[:, 2], without_model.params[:, 2])


def test_measured_ten_raw_nine_point_four_stop_is_enforced_by_native_state_bounds():
  reference = typed_reference(stopping_curve(), 10.0)
  assert reference.typed_model.moving.velocity[-1] == pytest.approx(0.6)
  planner = solve_reference(10.0, reference)
  node = reference.typed_model.rest_goal.native_node
  assert planner.rest_goal_state == 'satisfied'
  assert action(planner) < -0.2
  np.testing.assert_allclose(planner.x_sol[node:, 1:], 0.0, atol=REST_STATE_TOL, rtol=0.0)
  np.testing.assert_array_equal(planner.yref[node:, 2:4], 0.0)
  np.testing.assert_array_equal(planner.params[node:, MODEL_V_REF:MODEL_A_REF + 1], 0.0)
  lower, upper = planner.diagnostic_reference_bounds
  # The ideal reference stays exactly zero, while the native feasible set
  # expresses the unchanged 1e-3 rest contract without positive acceleration.
  np.testing.assert_array_equal(lower[node:, 0], 0.0)
  np.testing.assert_array_equal(lower[node:, 1], -REST_STATE_TOL)
  np.testing.assert_array_equal(upper[node:, 0], REST_STATE_TOL)
  np.testing.assert_array_equal(upper[node:, 1], 0.0)
  assert planner.diagnostic_rest_goal['physicalPositionBound'] is False
  assert planner.diagnostic_rest_goal['satisfactionScope'] == 'native_prediction_only'
  assert planner.diagnostic_rest_goal['physicalStoppedKnown'] is False


def test_switching_back_to_untyped_reference_removes_every_rest_bound_and_model_weight():
  planner = solve_reference(10.0, typed_reference(stopping_curve(), 10.0))
  planner.set_cur_state(10.0, 0.0)
  planner.set_weights(prev_accel_constraint=False)
  for _ in range(12):
    planner.update(radar_state(), reference=MpcReference(20.0))
    assert planner.solution_status == 0
  assert planner.rest_goal_state == 'none' and planner.rest_goal is None
  assert planner.v_solution[-1] > 10.0
  assert np.all(planner.diagnostic_reference_bounds[0][1:] < -100.0)
  assert np.all(planner.diagnostic_reference_bounds[1][1:] > 100.0)
  np.testing.assert_array_equal(planner.diagnostic_reference_weights[:, 1:4], 0.0)


@pytest.mark.parametrize('measured_speed', [0.0, 0.2])
def test_no_forward_with_positive_measured_accel_and_action_keeps_actual_owner_and_honda_hold(measured_speed):
  harness = held_harness(measured_speed)
  harness.source_state.aEgo = 0.4
  _, plans, rows, can_rows = harness.exercise(model_stop_curve(-0.4, 2.5), action_accel=0.5)
  assert harness.planner.reference_planner.model_evidence.no_forward
  assert harness.planner.mpc.x0[2] == pytest.approx(0.4)
  assert all(not event.longitudinalPlan.motionIntent.releaseAllowed for event in plans)
  assert all(event.longitudinalPlan.motionIntent.reasons & int(StopReason.MPC) for event in plans)
  assert all(row['resolved_stop'] and not row['resume'] and row['phase'] in ('hold', 'approach') for row in rows)
  assert all(command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0 and not command['STANDSTILL_RELEASE']
             and not command['resume'] for command in can_rows)


def test_future_restart_cannot_cancel_current_rest_even_with_a_departing_real_lead():
  planner, _ = make_planner(speed=0.0)
  state = make_state(speed=0.0, target=10.0, experimental=True)
  state['controlsState'].longControlState = LongCtrlState.stopping
  raw = np.where(MODEL_TIMES < 1.0, 0.0, 0.6 * (MODEL_TIMES - 1.0))
  assign_curve(state, raw)
  assign_lead(state, distance=15.0, speed=1.0, relative_speed=1.0)
  for frame in range(12):
    planner.update(state, now_ns=advance(state, frame))
    assert planner.mpc.solution_status == 0
    assert planner.motion_intent.valid
    assert planner.motion_intent.reasons & int(StopReason.MPC)
    assert not planner.motion_intent.release_allowed
    assert planner.output_a_target <= 0.0
  assert planner.mpc.rest_goal_state == 'satisfied'
  assert planner.reference_planner.model_evidence.rest_goal.future_forward


@pytest.mark.parametrize('after_anchor', [False, True])
def test_real_far_lead_does_not_disable_model_deceleration_with_positive_action(after_anchor):
  planner, _ = make_planner(speed=20.0)
  state = make_state(speed=20.0, target=30.0, experimental=True)
  assign_curve(state, 20.0 - 0.6 * MODEL_TIMES)
  assign_lead(state, distance=200.0, speed=20.0, relative_speed=0.0)
  for frame in range(12):
    anchor = advance(state, frame)
    if after_anchor:
      state.logMonoTime['modelV2'] = anchor + 5_000_000
    planner.update(state, now_ns=anchor + 10_000_000)
    assert planner.mpc.solution_status == 0 and planner.motion_intent.valid
  assert state['modelV2'].action.desiredAcceleration > 0.0
  assert np.any(planner.mpc.diagnostic_reference_weights[:, 2:4] > 0.0)
  assert planner.output_a_target < -0.1


@pytest.mark.parametrize('after_anchor', [False, True])
def test_neutral_model_preserves_acceleration_toward_a_departing_low_speed_lead(after_anchor):
  # Rounded from L's 12 s loss-of-following frame: no stop, fault or standstill,
  # but an opening real-lead gap while a constant model preference stays neutral.
  speed, acceleration = 1.15, -0.02
  planner, _ = make_planner(speed=speed, initial_accel=acceleration)
  state = make_state(speed=speed, target=4.0, experimental=True)
  state['carState'].aEgo = acceleration
  assign_curve(state, np.full(len(MODEL_TIMES), speed + 0.4))
  state['modelV2'].action.desiredAcceleration = 0.6
  assign_lead(state, distance=10.23, speed=2.0, relative_speed=2.0 - speed)
  targets = []
  for frame in range(12):
    anchor = advance(state, frame)
    if after_anchor:
      state.logMonoTime['modelV2'] = anchor + 5_000_000
    planner.update(state, now_ns=anchor + 10_000_000)
    assert planner.mpc.solution_status == 0 and planner.motion_intent.valid
    assert not planner.output_should_stop
    targets.append(planner.output_a_target)
  # Resolve an actual positive request beyond the existing steady-state
  # numerical budget, not a solver residual on the wrong side of zero.
  assert min(targets[-4:]) > 1e-4, targets


@pytest.mark.parametrize('after_anchor', [False, True])
def test_near_zero_model_slope_round_trips_do_not_switch_control_targets(after_anchor):
  """Tiny slope changes must not turn numerical signs into a new target owner.

  Each arm uses the same real planner/Controls/CAN history. The 1e-6 m/s²
  perturbation is below the existing 1e-4 output budget; production's upward
  controller rate remains +8 m/s³. Negative steps are compared with the
  matching neutral arm, so that unbounded extra braking cannot hide here.
  """
  def run(slopes):
    harness = StrictPlanTransportHarness(speed=10.0)
    harness.model_publication_after_state = after_anchor
    harness.engage()
    raw = np.full(len(MODEL_TIMES), 10.4)
    harness.exercise((raw * MODEL_TIMES, raw, np.zeros(len(raw))), action_accel=0.5)
    history_start = len(harness.history)
    targets = []
    for slope in slopes:
      raw = 10.4 + slope * MODEL_TIMES
      _, plans, _, _ = harness.exercise((raw * MODEL_TIMES, raw, np.full(len(raw), slope)),
                                        action_accel=0.5, seconds=0.4)
      targets.extend(float(event.longitudinalPlan.aTarget) for event in plans)
    rows = harness.history[history_start:]
    assert all(row['long_active'] and not row['degraded'] and row['phase'] == 'track' for row in rows)
    output = np.array([row['accel'] for row in rows])
    assert np.max(np.diff(output)) <= STARTING_ACCEL_RATE * DT_CTRL + 1e-4
    return np.array(targets), output

  slopes = [-1e-6, 0.0, 1e-6, 0.0] * 2
  neutral, neutral_output = run([0.0] * len(slopes))
  perturbed, perturbed_output = run(slopes)
  np.testing.assert_allclose(perturbed, neutral, atol=1e-4, rtol=0.0)
  np.testing.assert_allclose(perturbed_output, neutral_output, atol=1e-4, rtol=0.0)


def test_new_rest_semantics_discard_cruise_history_once_and_reentry_discards_again():
  planner, _ = make_planner(speed=10.0)
  state = make_state(speed=10.0, target=20.0, experimental=True)
  assign_curve(state, np.full(len(MODEL_TIMES), 10.0))
  planner.update(state, now_ns=advance(state, 0))
  assert planner.motion_intent.valid and not planner.reference_planner.entered_rest_goal

  def update(raw, frame):
    assign_curve(state, raw)
    weights = Mock(wraps=planner.mpc.set_weights)
    previous = planner.mpc.set_weights
    planner.mpc.set_weights = weights
    try:
      planner.update(state, now_ns=advance(state, frame))
    finally:
      planner.mpc.set_weights = previous
    assert planner.motion_intent.valid and planner.mpc.solution_status == 0
    return weights.call_args.args[0]

  assert not update(stopping_curve(), 1)
  assert planner.reference_planner.entered_rest_goal
  assert planner.mpc._reference_epoch_ns == SOURCE_NS + 50_000_000
  np.testing.assert_array_equal(planner.mpc.params[:, 3], 0.0)
  assert update(stopping_curve(), 2)
  assert not planner.reference_planner.entered_rest_goal
  assert np.any(planner.mpc.params[:, 3] < 0.0)
  assert update(np.full(len(MODEL_TIMES), 10.0), 3)
  assert not planner.reference_planner.entered_rest_goal
  assert not update(stopping_curve(), 4)
  assert planner.reference_planner.entered_rest_goal


@pytest.mark.parametrize('after_anchor', [False, True])
def test_lead_priority_uses_tighter_geometry_and_keeps_stronger_model_braking_available(after_anchor):
  timing = {'model_ns': SOURCE_NS + 5_000_000, 'now_ns': SOURCE_NS + 10_000_000} if after_anchor else {}
  far_raw = 20.0 - 0.6 * MODEL_TIMES
  far = solve_reference(20.0, typed_reference(far_raw, 20.0, typed=evidence(far_raw, 20.0, **timing)),
                        radar=radar_state(distance=200.0, speed=20.0))
  assert np.all(far.params[:, MODEL_BRAKING_PRIORITY] == 0.0)
  assert action(far) < 0.0
  np.testing.assert_array_equal(far.params[:, TYPED_MODEL_MODE], 1.0)
  radar = radar_state(distance=30.0, speed=20.0, acceleration=-3.0)
  lead_only = solve_reference(20.0, MpcReference(40.0), radar=radar, acceleration=-1.0)
  strong_raw = np.maximum(20.0 - 3.0 * MODEL_TIMES, 0.0)
  strong_model = solve_reference(20.0, typed_reference(strong_raw, 20.0, cruise=40.0,
                                                     typed=evidence(strong_raw, 20.0, **timing)),
                                radar=radar, acceleration=-1.0)
  if not after_anchor:
    assert np.any(strong_model.diagnostic_reference_weights[:, 2:4] > 0.0)
  # A later raw rest retains its existing compatibility risk problem. The
  # stronger-braking requirement applies in both representations.
  assert action(strong_model) < action(lead_only) - 0.05


@pytest.mark.parametrize('unknown_kind', ['backward_then_forward', 'single_zero'])
@pytest.mark.parametrize('after_anchor', [False, True])
def test_unknown_retains_compatible_model_and_lead_risk_without_new_typed_problem(unknown_kind, after_anchor):
  raw = -0.4 + 0.6 * MODEL_TIMES if unknown_kind == 'backward_then_forward' else np.full(len(MODEL_TIMES), 10.0)
  if unknown_kind == 'single_zero':
    raw[16] = 0.0
  timing = {'model_ns': SOURCE_NS + 1, 'now_ns': SOURCE_NS + 2} if after_anchor else {}
  reference = typed_reference(raw, 10.0, typed=evidence(raw, 10.0, **timing))
  radar = radar_state(distance=20.0, speed=0.0)
  old_risk = solve_reference(10.0, replace(reference, typed_model=None), radar=radar)
  unknown = solve_reference(10.0, reference, radar=radar)
  assert reference.typed_model.unknown is not None
  assert unknown.rest_goal_state == 'none' and unknown.rest_goal_braking_accel is None
  for name in ('params', 'yref', 'x_sol', 'u_sol'):
    np.testing.assert_allclose(getattr(old_risk, name), getattr(unknown, name), atol=1e-9, rtol=0.0)
  assert reference.typed_model.moving is None
  assert not reference.typed_model.allows_departure
  np.testing.assert_array_equal(unknown.params[:, MODEL_V_REF:TYPED_MODEL_MODE+1], 0.0)
  np.testing.assert_array_equal(unknown.diagnostic_reference_weights[:, 1:4], 0.0)


@pytest.mark.parametrize('delay_ns', [1, 5_000_000])
def test_later_moving_preference_keeps_unknown_epoch_and_real_lead_risk(delay_ns):
  raw = 10.0 + 0.6 * MODEL_TIMES
  typed = evidence(raw, 10.0, model_ns=SOURCE_NS + delay_ns, now_ns=SOURCE_NS + delay_ns + 1)
  reference = typed_reference(raw, 10.0, typed=typed)
  radar = radar_state(distance=20.0, speed=0.0)
  lead_only = solve_reference(10.0, MpcReference(30.0), radar=radar)
  future = solve_reference(10.0, reference, radar=radar)
  assert typed.unknown.reason == 'model_proxy_after_initial_state'
  assert typed.source.proxy_age_s is None and typed.source.physical_prediction_epoch_ns is None
  assert future.rest_goal_state == 'none' and future.rest_goal_braking_accel is None
  assert action(future) <= action(lead_only) + 1e-4
  np.testing.assert_array_equal(future.params[:, TYPED_MODEL_MODE], 0.0)
  np.testing.assert_array_equal(future.diagnostic_reference_weights[~typed.moving.supported, 1:4], 0.0)


@pytest.mark.parametrize('speed', [STOP_SPEED - 0.001, STOP_SPEED + 0.001])
def test_later_nonforward_preference_retains_forward_following_on_both_sides_of_the_stop_band(speed):
  raw = np.full(len(MODEL_TIMES), 1.0)
  typed = evidence(raw, speed, model_ns=SOURCE_NS + 5_000_000, now_ns=SOURCE_NS + 10_000_000)
  reference = typed_reference(raw, speed, typed=typed)
  radar = radar_state(distance=7.0, speed=1.5)
  actual = solve_reference(speed, reference, radar=radar)
  assert typed.has_future_moving_preference and not typed.allows_departure
  assert action(actual) > 0.0 and actual.v_solution[-1] > speed
  assert actual.source in (LongitudinalPlanSource.lead0, LongitudinalPlanSource.cruise)
  np.testing.assert_array_equal(actual.params[:, TYPED_MODEL_MODE], 0.0)
  assert np.all(actual.diagnostic_reference_weights[:, 0] > 0.0)


@pytest.mark.parametrize('lead_present', [False, True])
def test_available_model_after_carstate_proxy_does_not_create_a_cruise_fault(lead_present):
  planner, _ = make_planner(speed=10.0)
  state = make_state(speed=10.0, target=10.0, experimental=True)
  assign_curve(state, np.full(len(MODEL_TIMES), 10.0))
  if lead_present:
    assign_lead(state, distance=200.0, speed=10.0, relative_speed=0.0)
  for frame in range(12):
    anchor = advance(state, frame)
    state.logMonoTime['modelV2'] = anchor + 5_000_000
    planner.update(state, now_ns=anchor + 10_000_000)
    assert planner.motion_intent.valid and planner.mpc.solution_status == 0
    assert planner.reference_planner.model_reference_mode == 'future_supported_moving'
    assert planner.reference_planner.model_fallback_reason == 'model_proxy_after_initial_state'
    assert not planner.motion_intent.release_allowed
    assert abs(planner.output_a_target) < 1e-4
    np.testing.assert_array_equal(planner.mpc.params[:, TYPED_MODEL_MODE], 0.0)
    np.testing.assert_array_equal(planner.mpc.diagnostic_reference_weights[0, 1:4], 0.0)


@pytest.mark.parametrize('publication_offset_ns', [1, 5_000_000])
@pytest.mark.parametrize('raw_speed', [9.4, 10.0, 10.6])
def test_constant_model_bias_cannot_turn_publication_order_into_braking(raw_speed, publication_offset_ns):
  """A clock-order change adds no braking evidence to this equilibrium.

  P's action input is zero; a constant velocity-origin offset is not a
  deceleration or a stop. The existing cruise-equilibrium tolerance is 1e-4.
  Exercise each order for twelve actual native updates so reference history
  and smoothing participate in the comparison.
  """
  trajectories = []
  for order in (-1, 1):
    planner, _ = make_planner(speed=10.0)
    state = make_state(speed=10.0, target=10.0, experimental=True)
    assign_curve(state, np.full(len(MODEL_TIMES), raw_speed))
    state['modelV2'].action.desiredAcceleration = 0.0
    rows = []
    for frame in range(12):
      anchor = advance(state, frame)
      state.logMonoTime['modelV2'] = anchor + order * publication_offset_ns
      planner.update(state, now_ns=anchor + 10_000_000)
      assert planner.motion_intent.valid and planner.mpc.solution_status == 0
      assert not planner.motion_intent.release_allowed
      rows.append(planner.output_a_target)
    trajectories.append(np.array(rows))
  for trajectory in trajectories:
    np.testing.assert_allclose(trajectory, 0.0, atol=1e-4, rtol=0.0)
  np.testing.assert_allclose(trajectories[1], trajectories[0], atol=1e-4, rtol=0.0)


def test_future_supported_moving_preference_does_not_weaken_model_action_braking():
  planner, _ = make_planner(speed=10.0)
  state = make_state(speed=10.0, target=20.0, experimental=True)
  assign_curve(state, 10.6 + 0.6 * MODEL_TIMES)
  state['modelV2'].action.desiredAcceleration = -1.0
  for frame in range(12):
    anchor = advance(state, frame)
    state.logMonoTime['modelV2'] = anchor + 5_000_000
    planner.update(state, now_ns=anchor + 10_000_000)
    assert planner.motion_intent.valid and planner.mpc.solution_status == 0
    assert planner.reference_planner.model_reference_mode == 'future_supported_moving'
    assert planner.output_a_target <= -1.0


def test_later_publication_raw_forward_can_release_actual_held_owner_and_honda_can():
  harness = StrictPlanTransportHarness(speed=5.0)
  harness.hold()
  harness.source_state.cruiseState.standstill = False
  harness.source_state.vEgo = 0.2
  harness.source_state.standstill = False
  harness.model_publication_after_state = True
  raw = 0.2 + 0.6 * MODEL_TIMES
  curve = (0.2 * MODEL_TIMES + 0.3 * MODEL_TIMES**2, raw, np.full(len(raw), 0.6))
  _, plans, rows, can_rows = harness.exercise(curve, action_accel=0.5)
  assert harness.planner.reference_planner.model_reference_mode == 'future_supported_moving'
  assert harness.planner.reference_planner.model_evidence.has_future_moving_preference
  assert any(event.longitudinalPlan.motionIntent.releaseAllowed for event in plans)
  assert any(row['phase'] == 'release' and not row['resolved_stop'] for row in rows)
  assert any(command['STANDSTILL_RELEASE'] and not command['BRAKE_REQUEST'] and command['GAS_COMMAND'] > 0.0
             for command in can_rows)


@pytest.mark.parametrize('kind', ['negative_prefix', 'no_forward', 'rest', 'single_zero'])
def test_later_publication_raw_veto_keeps_hold_even_with_a_qualified_departing_lead(kind):
  harness = StrictPlanTransportHarness(speed=5.0)
  harness.hold()
  harness.source_state.cruiseState.standstill = False
  harness.source_state.vEgo = 0.2
  harness.source_state.aEgo = 0.4
  harness.source_state.standstill = False
  harness.model_publication_after_state = True
  harness.departing_lead_speed = 1.5
  raw = -0.015 + 0.6 * MODEL_TIMES
  if kind == 'no_forward':
    raw = stopping_curve(-0.4, 2.5)
  elif kind == 'rest':
    raw = stopping_curve(0.2, 2.5)
  elif kind == 'single_zero':
    raw = 0.2 + 0.6 * MODEL_TIMES
    raw[16] = 0.0
  curve = (np.maximum(raw, 0.0) * MODEL_TIMES, raw, np.gradient(raw, MODEL_TIMES))
  _, plans, rows, can_rows = harness.exercise(curve, action_accel=0.5)
  assert not harness.planner.reference_planner.model_evidence.allows_external_departure
  assert all(not event.longitudinalPlan.motionIntent.releaseAllowed for event in plans)
  assert all(row['resolved_stop'] and not row['resume'] for row in rows)
  assert all(command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0.0
             and not command['STANDSTILL_RELEASE'] for command in can_rows)


def test_qualified_departing_lead_is_independent_of_missing_aligned_or_raw_growth_preference():
  harness = StrictPlanTransportHarness(speed=5.0)
  harness.hold()
  harness.source_state.cruiseState.standstill = False
  harness.source_state.vEgo = 0.2
  harness.source_state.standstill = False
  harness.model_publication_after_state = True
  harness.departing_lead_speed = 1.5
  raw = np.full(len(MODEL_TIMES), 1.0)
  curve = (raw * MODEL_TIMES, raw, np.zeros(len(raw)))
  _, plans, rows, can_rows = harness.exercise(curve, action_accel=0.5)
  evidence = harness.planner.reference_planner.model_evidence
  assert evidence.has_future_moving_preference and evidence.raw_motion.kind == 'moving'
  assert not evidence.allows_departure and evidence.allows_external_departure
  assert any(event.longitudinalPlan.motionIntent.releaseAllowed for event in plans)
  negative_plan_times = {int(event.logMonoTime) for event in plans if event.longitudinalPlan.aTarget <= 0.0}
  assert all(event.longitudinalPlan.motionIntent.releaseAllowed for event in plans if int(event.logMonoTime) in negative_plan_times)
  assert all(row['resolved_stop'] and not row['resume'] for row in rows if row['consumed_plan_mono'] in negative_plan_times)
  assert any(row['phase'] == 'release' and not row['resolved_stop'] for row in rows)
  assert any(command['STANDSTILL_RELEASE'] and not command['BRAKE_REQUEST'] and command['GAS_COMMAND'] > 0.0
             for command in can_rows)


def test_cached_departing_lead_cannot_withdraw_a_held_stop_when_speed_crosses_the_stop_band():
  harness = StrictPlanTransportHarness(speed=5.0)
  harness.hold()
  harness.source_state.cruiseState.standstill = False
  harness.source_state.vEgo = 0.2
  harness.source_state.standstill = False
  harness.model_publication_after_state = True
  harness.departing_lead_speed = 1.5
  raw = np.full(len(MODEL_TIMES), 1.0)
  curve = (raw * MODEL_TIMES, raw, np.zeros(len(raw)))
  while harness.frame % PLANNER_STEPS:
    harness.step()
  crossing_frame = harness.frame + PLANNER_STEPS + 2
  original_step = harness.step

  def crossing_step(*args, **kwargs):
    if harness.frame + 1 >= crossing_frame:
      harness.source_state.vEgo = 0.31
    row = original_step(*args, **kwargs)
    return row | {'measured_speed': float(harness.source_state.vEgo)}

  with patch.object(harness, 'step', side_effect=crossing_step):
    _, plans, rows, can_rows = harness.exercise(curve, action_accel=0.5)
  first_plan_time = int(plans[0].logMonoTime)
  assert plans[0].longitudinalPlan.aTarget < 0.0
  assert plans[0].longitudinalPlan.motionIntent.releaseAllowed
  cached_rows = [row for row in rows if row['consumed_plan_mono'] == first_plan_time]
  assert cached_rows[0]['measured_speed'] < 0.3
  assert any(row['measured_speed'] > 0.3 for row in cached_rows)
  assert all(row['resolved_stop'] and not row['resume'] for row in rows)
  assert all(command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0.0
             and not command['STANDSTILL_RELEASE'] for command in can_rows)


def mutated_model_planner(bad_raw, bad_action, *, solver_status=None):
  planner, _ = make_planner(speed=20.0)
  state = make_state(speed=20.0, target=30.0, experimental=True)
  assign_curve(state, np.full(len(MODEL_TIMES), 20.0))
  planner.update(state, now_ns=advance(state, 0))
  assert planner.motion_intent.valid
  assign_curve(state, bad_raw)
  state['modelV2'].action.desiredAcceleration = bad_action
  assign_lead(state, distance=44.0, speed=10.0, relative_speed=-10.0)
  for service in state.logMonoTime:
    if service != 'modelV2':
      state.logMonoTime[service] = SOURCE_NS + 50_000_000
  if solver_status is None:
    planner.update(state, now_ns=SOURCE_NS + 50_000_000)
  else:
    with patch.object(planner.mpc, 'solver', Mock(wraps=planner.mpc.solver)):
      planner.mpc.solver.solve.return_value = solver_status
      planner.update(state, now_ns=SOURCE_NS + 50_000_000)
  return planner, state


def test_mutated_model_source_does_not_supply_new_raw_or_action_braking_but_retains_independent_lead():
  planner, _ = mutated_model_planner(stopping_curve(20.0, 0.1), -3.5)
  opposite, _ = mutated_model_planner(np.full(len(MODEL_TIMES), 40.0), 0.5)
  assert planner.reference_planner.model_reference_mode == 'invalid'
  assert planner.reference_planner.model_fallback_reason == 'model_mutated_with_same_identity'
  assert planner.mpc.diagnostic_last_solver_return == 0 and planner.mpc.solution_status != 0
  assert planner.motion_intent.is_braking_valid() and not planner.motion_intent.release_allowed
  assert planner.mpc.rest_goal is None
  np.testing.assert_array_equal(planner.mpc.params[:, MODEL_V_REF:TYPED_MODEL_MODE+1], 0.0)
  assert planner.output_a_target < -0.5
  assert planner.output_a_target == opposite.output_a_target
  np.testing.assert_array_equal(planner.mpc.params, opposite.mpc.params)
  np.testing.assert_array_equal(planner.mpc.x_sol, opposite.mpc.x_sol)


def test_native_exception_cannot_readmit_the_invalid_model_action_head():
  planner, _ = mutated_model_planner(stopping_curve(20.0, 0.1), -3.5, solver_status=4)
  assert planner.reference_planner.model_reference_mode == 'invalid'
  assert planner.mpc.solution_status == 4
  assert planner.motion_intent.is_braking_valid()
  assert planner.output_a_target == MPC_FAILURE_ACCEL
  assert planner.motion_intent.braking_accel == MPC_FAILURE_ACCEL


def test_independent_lead_braking_from_invalid_model_reaches_real_controls_with_original_timing():
  planner, state = mutated_model_planner(stopping_curve(20.0, 0.1), -3.5)
  publisher = Publisher()
  published_ns = SOURCE_NS + 50_000_000
  with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=published_ns):
    planner.publish(state, publisher)
  event = publisher.messages['longitudinalPlan']
  harness = ControlsHarness()
  harness.car_state.vEgo = 20.0
  harness.car_state.aEgo = 0.0
  harness.car_state.vCruise = 108.0
  # Source and LP frequencies are actual measured 100/20Hz intervals here.
  for frame in range(20):
    harness.step(new_plan=frame % PLANNER_STEPS == 0,
                 now_ns=SOURCE_NS - 150_000_000 + frame * 10_000_000)
  harness.events['longitudinalPlan'] = event.as_builder()
  harness.plan = harness.events['longitudinalPlan'].longitudinalPlan
  # Suppress this synthetic harness's optional reference restamping. The
  # consumer must receive the actual planner's source/producer identities.
  harness._reference_plan_time_ns = int(event.logMonoTime)
  command, motion = harness.step(plan_time_ns=int(event.logMonoTime),
                                 model_time_ns=int(event.longitudinalPlan.modelMonoTime),
                                 now_ns=published_ns + 10_000_000)
  assert harness.control.sm.freq_ok['carState'] and harness.control.sm.freq_ok['longitudinalPlan']
  assert harness.control.reference_timing_check.valid
  assert command.longActive and motion.degraded
  assert command.actuators.accel == pytest.approx(event.longitudinalPlan.aTarget, abs=1e-6)
  assert command.actuators.accel < -0.5
  assert not command.cruiseControl.resume


def test_unreachable_one_second_five_meter_goal_is_explicit_and_keeps_real_lead_braking():
  reference = typed_reference(stopping_curve(10.0, 1.0), 10.0)
  original_goal = reference.typed_model.rest_goal
  goal = replace(original_goal, nominal_position=5.0)
  reference = replace(reference, typed_model=replace(reference.typed_model, events=(goal,)))
  planner = LongitudinalMpc()
  planner.set_weights(prev_accel_constraint=False)
  planner.set_cur_state(10.0, 0.0)
  planner.update(radar_state(distance=12.0, speed=0.0), reference=reference)
  assert planner.diagnostic_solve_calls == 1  # real lead is still consumed
  assert planner.last_solver_status == 0
  assert planner.solution_status == MPC_GOAL_INFEASIBLE
  assert planner.rest_goal_state == 'infeasible'
  assert planner.rest_goal_reason == 'rest_goal_time_below_braking_lower_bound'
  assert planner.rest_goal_braking_accel == ACCEL_MIN
  assert planner.rest_goal.nominal_position == 5.0
  assert planner.diagnostic_rest_goal['nominalPositionBelowBrakingLowerBound']
  assert not planner.diagnostic_rest_goal['physicalPositionBound']
  assert not planner.previous_solution_valid
  assert np.all(np.isfinite(planner.params)) and np.all(np.isfinite(planner.x_sol))


def test_future_nominal_miss_keeps_a_verified_stop_trajectory_through_control_and_honda_can():
  harness = StrictPlanTransportHarness(speed=10.0)
  harness.engage()
  while harness.frame % PLANNER_STEPS:
    harness.step()
  harness.model_curve = model_stop_curve(10.0, 1.0)
  harness.action_accel = 0.5
  for _ in range(PLANNER_STEPS):
    harness.step()
  first_plan_time = int(harness.latest['longitudinalPlan'].logMonoTime)
  rows = [harness.step() for _ in range(30)]
  can_rows = [command for command in harness.native_commands if command['mono'] > rows[0]['mono']]
  assert harness.planner.mpc.rest_goal_state == 'infeasible'
  assert harness.planner.mpc.effective_rest_node > harness.planner.mpc.rest_goal.native_node
  assert harness.planner.mpc.effective_rest_state == 'satisfied'
  node = harness.planner.mpc.effective_rest_node
  np.testing.assert_allclose(harness.planner.mpc.x_sol[node:, 1:], 0.0, atol=REST_STATE_TOL, rtol=0.0)
  plan = harness.latest['longitudinalPlan'].longitudinalPlan
  assert plan.solverStatus == harness.planner.mpc.diagnostic_last_solver_return == 0
  assert plan.motionIntent.valid and not plan.motionIntent.brakingValid and not plan.motionIntent.restGoalRejected
  assert not plan.motionIntent.releaseAllowed
  assert harness.controls.sm.freq_ok['longitudinalPlan']
  assert all(row['consumed_plan_mono'] >= first_plan_time and row['long_active'] for row in rows)
  assert all(row['accel'] < 0.0 and not row['resume'] and not row['degraded'] for row in rows)
  assert can_rows and all(command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0.0
                          and command['ACCEL_COMMAND'] < 0.0 for command in can_rows)


@pytest.mark.parametrize('lead_index', [0, 1])
@pytest.mark.parametrize('distance,lead_speed,lead_accel,ego_speed,ego_accel', [
  (20.0, 0.0, 0.0, 10.0, -1.0), (44.0, 10.0, -1.5, 20.0, -1.0), (30.0, 20.0, -3.0, 20.0, -1.0),
])
def test_true_lead_pointwise_comparison_remains_a_diagnostic(lead_index, distance, lead_speed, lead_accel,
                                                           ego_speed, ego_accel, record_property):
  radar = radar_state(distance=distance, speed=lead_speed, acceleration=lead_accel)
  if lead_index:
    radar.leadOne, radar.leadTwo = radar.leadTwo, radar.leadOne
  original = solve_reference(ego_speed, MpcReference(40.0), radar=radar, acceleration=ego_accel)
  candidate = solve_reference(ego_speed, typed_reference(ego_speed + 0.6 * MODEL_TIMES, ego_speed, cruise=40.0),
                              radar=radar, acceleration=ego_accel)
  # Keep all original frozen inputs and their pointwise differences. Acceptance
  # v2 uses independent complete-event clearance/margin contracts instead of
  # requiring one soft optimum to reproduce another at a single horizon point.
  # test_typed_lead_event_contract covers these same initial states and both
  # slots with separate closed-loop feedback, finite actuator limits and an
  # independently qualified maximum-brake witness; CAN/owner checks are separate.
  for planner in (original, candidate):
    assert planner.diagnostic_last_solver_return == planner.solution_status == 0
    assert np.all(np.isfinite(planner.x_sol)) and np.all(np.isfinite(planner.u_sol))
  record_property('reference_action_mps2', float(action(original)))
  record_property('candidate_action_mps2', float(action(candidate)))
  record_property('pointwise_difference_mps2', float(action(candidate) - action(original)))
