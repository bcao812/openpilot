"""Current REST/zero requests, healthy recovery and independent stop authority."""

from dataclasses import replace

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS
from openpilot.selfdrive.controls.lib.model_reference import ModelReferenceAdapter
from openpilot.selfdrive.controls.lib.motion_supervisor import (
  DEGRADED_ACCEL, MOTION_INTENT_VERSION, MotionIntent, MotionPhase, MotionReferenceKind, StopReason,
)
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_typed_model_reference import MODEL_TIMES, stopping_curve
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import ACCEL_LIMITS, ControlHarness
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_planner, make_state


INSTANCE_NS = 1_000_000_000
SOURCE_NS = 2_000_000_000
NOW_NS = SOURCE_NS + 10_000_000


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


def failed_rest(*, limits=ACCEL_LIMITS):
  harness = tracking()
  result = harness.step(-3.5, rejected_intent(), plan_valid=False, braking_plan_valid=True, solver_status=-4, accel_limits=limits)
  assert result == limits[0]
  assert harness.control.motion_degraded
  return harness


def follow_current_target(harness, target=-0.1, intent=None):
  """Current healthy inputs may unwind old braking through command smoothing."""
  outputs = []
  for _ in range(50):
    outputs.append(harness.step(target, zero_intent() if intent is None else intent))
    assert not harness.control.motion_degraded
    assert not harness.control.resolved_should_stop
    assert ACCEL_LIMITS[0] <= outputs[-1] <= ACCEL_LIMITS[1]
  assert outputs[-1] == pytest.approx(target)
  return outputs


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
  harness = failed_rest()
  outputs = follow_current_target(harness, intent=zero_intent(reference_kind=int(owner.describe_motion_reference(state))))
  assert outputs[0] > -3.5


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
  harness = failed_rest()
  # A present qualified REST goal keeps its source meaning, but the execution
  # request is still the current target, without a stored rejected-goal cap.
  outputs = follow_current_target(harness, intent=zero_intent(reference_kind=int(MotionReferenceKind.REST)))
  assert outputs[0] > -3.5


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
def test_current_protocol_roundtrip_keeps_descriptions_and_instance_without_adding_stop_reasons(intent):
  planner, owner, _ = source_description(np.zeros(len(MODEL_TIMES)))
  planner.motion_intent = intent
  message = log.LongitudinalPlan.new_message()
  owner.publish(message)
  with log.LongitudinalPlan.from_bytes(message.to_bytes()) as decoded:
    assert MotionIntent.from_message(decoded.motionIntent) == intent
    assert decoded.motionIntent.version == MOTION_INTENT_VERSION == 10
    assert decoded.motionIntent.reasons == intent.reasons == 0


@pytest.mark.parametrize('changes', [
  {'version': 3}, {'version': 0}, {'reference_kind': 99}, {'reference_kind': True},
  {'producer_instance_ns': -1}, {'producer_instance_ns': True}, {'producer_instance_ns': 0},
  {'rest_goal_rejected': True}, {'rest_goal_rejected': 1},
  {'release_allowed': True}, {'moving_lead': True},
  {'mode': int(LongitudinalMode.SUPERVISED)},
])
def test_invalid_metadata_degrades_but_a_new_healthy_reference_can_recover(changes):
  harness = failed_rest()
  harness.step(-0.1, zero_intent(**changes))
  assert harness.control.motion_supervisor.degraded
  outputs = follow_current_target(harness)
  assert outputs[0] > -3.5


@pytest.mark.parametrize('kind', [MotionReferenceKind.EXACT_ZERO, MotionReferenceKind.REST, MotionReferenceKind.NO_FORWARD])
@pytest.mark.parametrize('grant', ['release_allowed', 'moving_lead'])
def test_stopping_descriptions_cannot_contradict_their_departure_or_lead_flags(kind, grant):
  intent = zero_intent(reference_kind=int(kind), **{grant: True})
  assert not intent.is_valid()
  harness = failed_rest()
  assert harness.step(-0.1, intent) <= DEGRADED_ACCEL
  assert harness.control.motion_supervisor.degraded


def test_repeated_zero_does_not_renew_the_source_lease():
  harness = failed_rest()
  assert harness.step(-0.1, zero_intent()) > -3.5
  for _ in range(25):
    output = harness.step(-0.1, zero_intent(), new_input=False)
    assert -3.5 < output <= -0.1
    assert not harness.control.motion_degraded
  output = harness.step(-0.1, zero_intent(), new_input=False)
  assert harness.control.motion_degraded and output <= DEGRADED_ACCEL


def test_repeated_current_rest_request_does_not_block_the_next_healthy_zero_request():
  harness = tracking()
  current_rest = zero_intent(reference_kind=int(MotionReferenceKind.REST))
  assert harness.step(-1.0, current_rest) == -1.0
  for _ in range(20):
    assert harness.step(-1.0, current_rest, new_input=False) == -1.0
    assert not harness.control.motion_degraded
  outputs = follow_current_target(harness, target=0.0)
  assert outputs[0] > -1.0


def test_a_new_rejected_plan_from_the_same_current_model_keeps_braking_but_can_recover():
  harness = tracking()
  model_ns = harness.model_time_ns
  assert harness.step(-3.5, rejected_intent(), model_time_ns=model_ns,
                      plan_valid=False, braking_plan_valid=True, solver_status=-4) == -3.5
  assert harness.control.motion_degraded
  outputs = follow_current_target(harness)
  assert outputs[0] > -3.5


def test_off_observation_retires_an_old_producer_without_arming_a_stop():
  harness = failed_rest()
  assert harness.step(-0.1, zero_intent(producer_instance_ns=INSTANCE_NS + 1), active=False) == 0.0
  assert harness.control.motion_phase == MotionPhase.OFF
  supervisor = harness.control.motion_supervisor
  harness.step(-3.5, rejected_intent(), plan_valid=False, braking_plan_valid=True, solver_status=-4)
  assert supervisor.degraded and supervisor.degraded_braking_accel is None
  follow_current_target(harness, intent=zero_intent(producer_instance_ns=INSTANCE_NS + 1))


@pytest.mark.parametrize('kind', [MotionReferenceKind.FORWARD, MotionReferenceKind.NO_FORWARD, MotionReferenceKind.UNKNOWN,
                                MotionReferenceKind.MOVING])
def test_other_current_reference_kinds_and_later_zero_follow_their_own_target(kind):
  harness = failed_rest()
  assert harness.step(-0.1, zero_intent(reference_kind=int(kind))) > -3.5
  assert not harness.control.motion_degraded
  follow_current_target(harness)


@pytest.mark.parametrize('exit_kind', ['off', 'gas', 'brake', 'invalid_plan', 'invalid_limits', 'backwards_clock', 'same_id_mutation'])
def test_manual_and_invalid_input_keep_their_priority_over_current_zero(exit_kind):
  harness = failed_rest()
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
  if exit_kind in ('off', 'gas', 'brake'):
    assert result == 0.0 and harness.control.motion_phase == MotionPhase.OFF
  else:
    assert harness.control.motion_degraded and result <= DEGRADED_ACCEL


def test_new_instance_rejects_a_retired_producer_and_then_recovers_on_current_input():
  harness = failed_rest()
  newer = INSTANCE_NS + 1
  assert harness.step(-0.1, zero_intent(producer_instance_ns=newer)) > -3.5
  supervisor = harness.control.motion_supervisor
  harness.step(-3.5, rejected_intent(), plan_valid=False, braking_plan_valid=True, solver_status=-4)
  assert supervisor.degraded and supervisor.degraded_braking_accel is None
  follow_current_target(harness, intent=zero_intent(producer_instance_ns=newer))


def test_current_zero_request_keeps_stronger_braking_and_current_limits_without_old_cap():
  harness = failed_rest(limits=(-2.0, 1.0))
  assert harness.step(-3.0, zero_intent()) == -3.0
  outputs = follow_current_target(harness)
  assert outputs[-1] > -2.0
  assert harness.step(-3.0, zero_intent(), accel_limits=(-1.0, 1.0)) == -1.0
  outputs = follow_current_target(harness)
  assert outputs[0] > -1.0


def test_actual_hold_preserves_braking_until_current_departure_permission():
  harness = failed_rest()
  harness.state.vEgo = 0.05
  assert harness.step(0.1, zero_intent(reasons=int(StopReason.MPC))) == -3.5
  assert harness.control.motion_phase == MotionPhase.APPROACH
  harness.state.vEgo, harness.state.standstill = 0.0, True
  assert harness.step(0.1, zero_intent(reasons=int(StopReason.MPC))) == -3.5
  assert harness.control.motion_phase == MotionPhase.HOLD
  assert harness.step(0.1, zero_intent()) == -3.5
  assert harness.control.resolved_should_stop
  current_departure = zero_intent(reference_kind=int(MotionReferenceKind.FORWARD), release_allowed=True)
  follow_current_target(harness, target=0.5, intent=current_departure)
  assert harness.control.motion_phase == MotionPhase.TRACK


@pytest.mark.parametrize('status', [0, 4, -1])
def test_other_failure_codes_cannot_create_a_permanent_braking_cap(status):
  harness = tracking()
  harness.step(-3.5, rejected_intent(), plan_valid=False, braking_plan_valid=True, solver_status=status)
  assert harness.control.motion_degraded
  follow_current_target(harness)
