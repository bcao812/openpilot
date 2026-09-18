"""A confirmed rejected-goal controller output can constrain a qualified zero chain."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS
from openpilot.selfdrive.controls.lib.model_reference import ModelReferenceAdapter
from openpilot.selfdrive.controls.lib.motion_supervisor import (
  DEGRADED_ACCEL, MOTION_INTENT_VERSION, MotionIntent, MotionPhase, MotionReferenceKind, StopReason,
)
from openpilot.selfdrive.controls.tests.test_motion_supervisor import MotionHarness
from openpilot.selfdrive.controls.tests.test_typed_model_reference import MODEL_TIMES, stopping_curve
from openpilot.selfdrive.controls.tests.test_unified_longcontrol import ACCEL_LIMITS, ControlHarness
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_planner import make_planner, make_state


INSTANCE_NS = 1_000_000_000
SOURCE_NS = 2_000_000_000
NOW_NS = SOURCE_NS + 10_000_000


@pytest.fixture(autouse=True)
def isolated_params():
  with OpenpilotPrefix():
    yield


def zero_intent(**changes):
  return replace(MotionIntent(allow_creep=False, reference_kind=int(MotionReferenceKind.EXACT_ZERO),
                              producer_instance_ns=INSTANCE_NS), **changes)


def rejected_intent(**changes):
  return replace(MotionIntent(allow_creep=False, valid=False, braking_accel=-3.5, reference_kind=int(MotionReferenceKind.REST),
                              rest_goal_rejected=True, producer_instance_ns=INSTANCE_NS), **changes)


def tracking():
  harness = ControlHarness()
  harness.state.vEgo = 10.0
  assert harness.step(-0.5, zero_intent(reference_kind=int(MotionReferenceKind.MOVING))) == -0.5
  assert harness.control.motion_phase == MotionPhase.TRACK
  return harness


def armed(*, limits=ACCEL_LIMITS):
  harness = tracking()
  result = harness.step(-3.5, rejected_intent(), plan_valid=False, braking_plan_valid=True, solver_status=-4, accel_limits=limits)
  assert result == limits[0]
  assert harness.control.motion_supervisor._rest_braking_accel == result
  return harness


def source_description(raw, *, model_after_anchor=False):
  planner, _ = make_planner(speed=5.0)
  state = make_state(speed=5.0, experimental=True)
  for service in state.logMonoTime:
    state.logMonoTime[service] = SOURCE_NS
  model_ns = SOURCE_NS + 5_000_000 if model_after_anchor else SOURCE_NS
  state.logMonoTime['modelV2'] = model_ns
  state['modelV2'].velocity.x = list(map(float, raw))
  planner.reference_timing.qualify_source(SOURCE_NS, 5.0, 0.0, NOW_NS)
  planner.reference_timing.set_initial_state(5.0, 0.0)
  owner = planner.reference_planner
  owner.model_reference_mode = 'typed'
  owner.model_evidence = ModelReferenceAdapter().build(
    state['modelV2'].velocity.x, T_IDXS, 5.0,
    model_mono_time_ns=model_ns, anchor_mono_time_ns=SOURCE_NS, consumed_mono_time_ns=NOW_NS,
  )
  return planner, owner, state


@pytest.mark.parametrize('raw,kind', [
  (np.zeros(len(MODEL_TIMES)), MotionReferenceKind.EXACT_ZERO),
  (np.full(len(MODEL_TIMES), -1e-12), MotionReferenceKind.NO_FORWARD),
  (np.full(len(MODEL_TIMES), 1e-12), MotionReferenceKind.MOVING),
  (stopping_curve(5.0, 1.0), MotionReferenceKind.REST),
  (0.5 + 0.6 * MODEL_TIMES, MotionReferenceKind.FORWARD),
])
def test_raw_description_preserves_exact_zero_and_negative_no_forward_as_different_sources(raw, kind):
  _, owner, state = source_description(raw)
  assert owner.describe_motion_reference(state) == kind
  if kind in (MotionReferenceKind.EXACT_ZERO, MotionReferenceKind.NO_FORWARD):
    assert owner.model_evidence.no_forward and owner.model_evidence.rest_goal is None


@pytest.mark.parametrize('fault', ['model_identity', 'raw_payload', 'anchor', 'consumer', 'unknown'])
def test_exact_zero_description_requires_the_current_qualified_whole_raw_source(fault):
  _, owner, state = source_description(np.zeros(len(MODEL_TIMES)))
  if fault == 'model_identity':
    state.logMonoTime['modelV2'] += 1
  elif fault == 'raw_payload':
    state['modelV2'].velocity.x = [1e-9] + [0.0] * (len(MODEL_TIMES) - 1)
  elif fault in ('anchor', 'consumer'):
    name = 'anchor_mono_time_ns' if fault == 'anchor' else 'consumed_mono_time_ns'
    owner.model_evidence = replace(owner.model_evidence, source=replace(owner.model_evidence.source, **{name: NOW_NS + 1}))
  else:
    owner.model_evidence = ModelReferenceAdapter().build(np.zeros(len(MODEL_TIMES)), T_IDXS, 5.0,
      model_mono_time_ns=NOW_NS + 1, anchor_mono_time_ns=SOURCE_NS, consumed_mono_time_ns=NOW_NS)
  assert owner.describe_motion_reference(state) == MotionReferenceKind.UNKNOWN


def test_available_fresh_zero_after_the_carstate_anchor_keeps_its_qualified_raw_meaning():
  _, owner, state = source_description(np.zeros(len(MODEL_TIMES)), model_after_anchor=True)
  assert owner.model_evidence.unknown.reason == 'model_proxy_after_initial_state'
  assert owner.model_evidence.no_forward and owner.model_evidence.rest_goal is None
  assert owner.describe_motion_reference(state) == MotionReferenceKind.EXACT_ZERO
  harness = armed()
  assert harness.step(-0.1, zero_intent(reference_kind=int(owner.describe_motion_reference(state)))) == -3.5


def test_the_alignment_exception_does_not_accept_ambiguous_raw_motion():
  _, owner, state = source_description(-0.4 + 0.6 * MODEL_TIMES, model_after_anchor=True)
  assert owner.model_evidence.unknown.reason == 'model_proxy_after_initial_state'
  assert owner.model_evidence.raw_motion.unknown is not None
  assert owner.describe_motion_reference(state) == MotionReferenceKind.UNKNOWN


def test_a_real_continued_goal_takes_precedence_over_its_zero_raw_shape():
  planner, owner, state = source_description(np.zeros(len(MODEL_TIMES)))
  adapter = ModelReferenceAdapter()
  previous_ns = SOURCE_NS - 100_000_000
  prior = adapter.build(stopping_curve(5.0, 0.001), T_IDXS, 5.0, model_mono_time_ns=previous_ns,
                        anchor_mono_time_ns=previous_ns, consumed_mono_time_ns=previous_ns + 10_000_000,
                        retain_rest_continuity=True)
  owner.model_evidence = adapter.build(np.zeros(len(MODEL_TIMES)), T_IDXS, 5.0,
    model_mono_time_ns=SOURCE_NS, anchor_mono_time_ns=SOURCE_NS, consumed_mono_time_ns=NOW_NS, retain_rest_continuity=True)
  goal = owner.model_evidence.rest_goal
  assert goal is not None and goal.continued_from == (previous_ns, prior.rest_goal.event_proxy_epoch_ns)
  assert np.all(owner.model_evidence.raw_velocity == 0.0)
  assert owner.describe_motion_reference(state) == MotionReferenceKind.REST
  planner.mpc.rest_goal, planner.mpc.rest_goal_state, planner.mpc.solution_status = goal, 'infeasible', -4
  planner.mpc.rest_goal_braking_accel, planner.mpc.failure_reason = -3.5, 'test_continued_goal'
  owner.fail_mpc(state, 5.0, -0.5)
  assert planner.motion_intent.rest_goal_rejected
  harness = armed()
  harness.step(-0.1, zero_intent(reference_kind=int(MotionReferenceKind.REST)))
  assert harness.control.motion_supervisor._rest_braking_origin is None


@pytest.mark.parametrize('status,state_name,present,expected', [
  (-4, 'infeasible', True, True), (-4, 'unverified', True, True), (-5, 'unsatisfied', True, True),
  (0, 'satisfied', True, False), (4, 'solver_failure', True, False),
  (-4, 'satisfied', True, False), (-5, 'infeasible', True, False), (-4, 'infeasible', False, False),
])
def test_only_this_frames_rejected_rest_goal_can_describe_the_braking_cause(status, state_name, present, expected):
  planner, owner, state = source_description(stopping_curve(5.0, 1.0))
  planner.mpc.solution_status = status
  planner.mpc.rest_goal_state = state_name
  planner.mpc.rest_goal = owner.model_evidence.rest_goal if present else None
  planner.mpc.rest_goal_braking_accel = -3.5
  planner.mpc.failure_reason = 'test_goal_status'
  owner.fail_mpc(state, 5.0, -0.5)
  assert planner.motion_intent.rest_goal_rejected is expected
  assert planner.motion_intent.producer_instance_ns == planner.reference_timing.instance_ns
  if expected:
    assert planner.motion_intent.is_braking_valid()
    assert planner.motion_intent.reference_kind == MotionReferenceKind.REST


def test_goal_from_another_frame_cannot_label_the_current_brake_as_a_rest_rejection():
  planner, owner, state = source_description(stopping_curve(5.0, 1.0))
  goal = owner.model_evidence.rest_goal
  planner.mpc.rest_goal = replace(goal, event_proxy_epoch_ns=goal.event_proxy_epoch_ns - 1)
  planner.mpc.solution_status, planner.mpc.rest_goal_state = -4, 'infeasible'
  planner.mpc.rest_goal_braking_accel, planner.mpc.failure_reason = -3.5, 'test_old_goal'
  owner.fail_mpc(state, 5.0, -0.5)
  assert not planner.motion_intent.rest_goal_rejected


@pytest.mark.parametrize('intent', [zero_intent(), rejected_intent(), MotionIntent()])
def test_protocol_four_roundtrip_keeps_descriptions_and_instance_without_adding_stop_reasons(intent):
  planner, owner, _ = source_description(np.zeros(len(MODEL_TIMES)))
  planner.motion_intent = intent
  message = log.LongitudinalPlan.new_message()
  owner.publish(message)
  with log.LongitudinalPlan.from_bytes(message.to_bytes()) as decoded:
    assert MotionIntent.from_message(decoded.motionIntent) == intent
    assert decoded.motionIntent.version == MOTION_INTENT_VERSION == 4
    assert decoded.motionIntent.reasons == intent.reasons == 0


@pytest.mark.parametrize('changes', [
  {'version': 3}, {'version': 0}, {'reference_kind': 99}, {'reference_kind': True},
  {'producer_instance_ns': -1}, {'producer_instance_ns': True}, {'producer_instance_ns': 0},
  {'rest_goal_rejected': True}, {'rest_goal_rejected': 1},
  {'release_allowed': True}, {'moving_lead': True},
  {'mode': int(LongitudinalMode.SUPERVISED)},
])
def test_invalid_continuity_metadata_clears_execution_state_and_cannot_be_revived_by_zero(changes):
  harness = armed()
  harness.step(-0.1, zero_intent(**changes))
  supervisor = harness.control.motion_supervisor
  assert supervisor.degraded and supervisor._rest_braking_origin is None
  harness.step(-0.1, zero_intent())
  assert supervisor.rest_braking_accel_limit is None and supervisor._rest_braking_origin is None


@pytest.mark.parametrize('output', [0.0, 0.1, -0.1, float('nan'), -4.0])
def test_a_rejected_request_without_a_confirmed_negative_output_cannot_arm(output):
  harness = MotionHarness()
  harness.step(rejected_intent(), a_target=-3.5, v_ego=10.0, plan_valid=False, braking_plan_valid=True, solver_status=-4)
  supervisor = harness.supervisor
  assert supervisor._rest_braking_origin is None
  supervisor.confirm_braking_output(output, ACCEL_LIMITS, now_ns=harness.now_ns)
  harness.step(zero_intent(), a_target=-0.1, v_ego=10.0)
  assert supervisor.rest_braking_accel_limit is None and supervisor._rest_braking_origin is None


@pytest.mark.parametrize('failure', ['unconfirmed', 'late_confirmation', 'wrong_clock'])
def test_pending_cause_is_for_one_control_frame_and_one_confirmation(failure):
  harness = MotionHarness()
  harness.step(rejected_intent(), a_target=-3.5, v_ego=10.0, plan_valid=False, braking_plan_valid=True, solver_status=-4)
  original_now = harness.now_ns
  if failure == 'wrong_clock':
    harness.supervisor.confirm_braking_output(-3.5, ACCEL_LIMITS, now_ns=original_now - 1)
  harness.step(zero_intent(), a_target=-0.1, v_ego=10.0)
  if failure == 'late_confirmation':
    harness.supervisor.confirm_braking_output(-3.5, ACCEL_LIMITS, now_ns=original_now)
  assert harness.supervisor._rest_braking_origin is None
  assert harness.supervisor.rest_braking_accel_limit is None


@pytest.mark.parametrize('kind', [MotionReferenceKind.EXACT_ZERO, MotionReferenceKind.REST, MotionReferenceKind.NO_FORWARD])
@pytest.mark.parametrize('grant', ['release_allowed', 'moving_lead'])
def test_stopping_descriptions_cannot_contradict_their_departure_or_lead_flags(kind, grant):
  intent = zero_intent(reference_kind=int(kind), **{grant: True})
  assert not intent.is_valid()
  harness = armed()
  harness.step(-0.1, intent)
  assert harness.control.motion_supervisor.degraded
  assert harness.control.motion_supervisor._rest_braking_origin is None


def test_repeated_zero_does_not_renew_the_source_lease():
  harness = armed()
  assert harness.step(-0.1, zero_intent()) == -3.5
  supervisor = harness.control.motion_supervisor
  origin = supervisor._rest_braking_origin
  for _ in range(25):
    assert harness.step(-0.1, zero_intent(), new_input=False) == -3.5
    assert supervisor._rest_braking_origin == origin
  harness.step(-0.1, zero_intent(), new_input=False)
  assert supervisor.degraded and supervisor._rest_braking_origin is None


def test_confirmed_output_and_repeated_identifiers_do_not_refresh_the_event_or_strength():
  harness = MotionHarness()
  intent = rejected_intent()
  harness.step(intent, a_target=-3.5, v_ego=10.0, plan_valid=False, braking_plan_valid=True, solver_status=-4)
  harness.supervisor.confirm_braking_output(-1.0, ACCEL_LIMITS, now_ns=harness.now_ns)
  origin = harness.supervisor._rest_braking_origin
  harness.supervisor.confirm_braking_output(-3.5, ACCEL_LIMITS, now_ns=harness.now_ns)
  harness.step(intent, a_target=-3.5, v_ego=10.0, new_plan=False, new_model=False,
               plan_valid=False, braking_plan_valid=True, solver_status=-4)
  harness.supervisor.confirm_braking_output(-3.5, ACCEL_LIMITS, now_ns=harness.now_ns)
  assert harness.supervisor._rest_braking_origin == origin
  assert harness.supervisor._rest_braking_accel == -1.0
  harness.step(zero_intent(), a_target=-0.1, v_ego=10.0)
  assert harness.supervisor.rest_braking_accel_limit == -1.0


def test_a_new_accepted_plan_can_confirm_first_rejection_from_the_same_still_fresh_model():
  harness = tracking()
  model_ns = harness.model_time_ns
  assert harness.step(-3.5, rejected_intent(), model_time_ns=model_ns,
                      plan_valid=False, braking_plan_valid=True, solver_status=-4) == -3.5
  supervisor = harness.control.motion_supervisor
  assert supervisor._rest_braking_origin == (INSTANCE_NS, harness.plan_time_ns, model_ns, harness.now_ns)
  assert harness.step(-0.1, zero_intent()) == -3.5
  assert not supervisor.degraded


def test_off_observation_retires_an_old_instance_without_arming_new_zero():
  harness = armed()
  assert harness.step(-0.1, zero_intent(producer_instance_ns=INSTANCE_NS + 1), active=False) == 0.0
  supervisor = harness.control.motion_supervisor
  assert supervisor._rest_braking_origin is None
  assert supervisor._rest_braking_instance_high_ns == INSTANCE_NS + 1
  harness.step(-3.5, rejected_intent(), plan_valid=False, braking_plan_valid=True, solver_status=-4)
  assert supervisor._rest_braking_origin is None and supervisor.degraded_braking_accel is None


@pytest.mark.parametrize('kind', [MotionReferenceKind.FORWARD, MotionReferenceKind.REST, MotionReferenceKind.NO_FORWARD, MotionReferenceKind.UNKNOWN,
                                MotionReferenceKind.MOVING])
def test_an_accepted_nonzero_source_ends_the_chain_before_a_later_isolated_zero(kind):
  harness = armed()
  harness.step(-0.1, zero_intent(reference_kind=int(kind)))
  assert harness.control.motion_supervisor._rest_braking_origin is None
  harness.step(-0.1, zero_intent())
  assert harness.control.motion_supervisor.rest_braking_accel_limit is None


@pytest.mark.parametrize('exit_kind', ['off', 'gas', 'brake', 'invalid_plan', 'invalid_limits', 'backwards_clock', 'same_id_mutation'])
def test_manual_or_invalid_input_ends_the_chain_without_weakening_the_original_priority(exit_kind):
  harness = armed()
  options = {}
  if exit_kind == 'off':
    options['active'] = False
  elif exit_kind in ('gas', 'brake'):
    setattr(harness.state, exit_kind + 'Pressed', True)
  elif exit_kind == 'invalid_plan':
    options['plan_valid'] = False
  elif exit_kind == 'invalid_limits':
    options['accel_limits'] = (float('nan'), 2.0)
  elif exit_kind == 'backwards_clock':
    options['now_ns'] = harness.now_ns - 1
  else:
    options['new_input'] = False
  result = harness.step(-0.1, zero_intent(), **options)
  supervisor = harness.control.motion_supervisor
  assert supervisor._rest_braking_origin is None and supervisor.rest_braking_accel_limit is None
  if exit_kind in ('off', 'gas', 'brake'):
    assert result == 0.0 and harness.control.motion_phase == MotionPhase.OFF
  else:
    assert supervisor.degraded and result <= DEGRADED_ACCEL


def test_new_instance_zero_cannot_arm_and_retired_instance_cannot_return():
  harness = armed()
  newer = INSTANCE_NS + 1
  harness.step(-0.1, zero_intent(producer_instance_ns=newer))
  supervisor = harness.control.motion_supervisor
  assert supervisor._rest_braking_origin is None
  harness.step(-3.5, rejected_intent(), plan_valid=False, braking_plan_valid=True, solver_status=-4)
  assert supervisor._rest_braking_origin is None and supervisor.degraded_braking_accel is None
  harness.step(-0.1, zero_intent(producer_instance_ns=newer))
  assert supervisor.rest_braking_accel_limit is None


def test_zero_chain_keeps_the_confirmed_limit_without_ratchet_and_obeys_current_limits():
  harness = armed(limits=(-2.0, 1.0))
  supervisor = harness.control.motion_supervisor
  origin = supervisor._rest_braking_origin
  assert harness.step(-3.0, zero_intent()) == -3.0  # stronger braking is still available
  for _ in range(50):
    assert harness.step(0.5, zero_intent()) <= -2.0
    assert supervisor._rest_braking_accel == -2.0 and supervisor._rest_braking_origin == origin
  assert harness.step(0.5, zero_intent(), accel_limits=(-1.0, 1.0)) == -1.0
  assert supervisor._rest_braking_accel == -2.0
  assert harness.step(0.5, zero_intent()) == -2.0


def test_handoff_occurs_after_owner_hold_and_final_output_without_resetting_the_executed_brake():
  harness = armed()
  supervisor = harness.control.motion_supervisor
  harness.state.vEgo = 0.05
  assert harness.step(0.1, zero_intent(reasons=int(StopReason.MPC))) == -3.5
  assert harness.control.motion_phase == MotionPhase.APPROACH and supervisor._rest_braking_origin is not None
  harness.state.vEgo, harness.state.standstill = 0.0, True
  actual = supervisor.confirm_braking_output
  seen = []

  def confirm(output, limits, *, now_ns):
    seen.append((supervisor.phase, harness.control.last_output_accel, supervisor._rest_braking_origin))
    actual(output, limits, now_ns=now_ns)

  with patch.object(supervisor, 'confirm_braking_output', side_effect=confirm) as confirmation:
    assert harness.step(0.1, zero_intent(reasons=int(StopReason.MPC))) == -3.5
  assert confirmation.call_count == 1 and seen[0][0] == MotionPhase.HOLD and seen[0][1] == -3.5 and seen[0][2] is not None
  assert supervisor._rest_braking_origin is None and harness.control.last_output_accel == -3.5
  assert harness.step(0.1, zero_intent(reasons=int(StopReason.MPC))) == -3.5


@pytest.mark.parametrize('status', [0, 4, -1])
def test_other_solver_statuses_cannot_arm_a_rest_rejection(status):
  harness = tracking()
  harness.step(-3.5, rejected_intent(), plan_valid=False, braking_plan_valid=True, solver_status=status)
  assert harness.control.motion_supervisor._rest_braking_origin is None
  harness.step(-0.1, zero_intent())
  assert harness.control.motion_supervisor.rest_braking_accel_limit is None
