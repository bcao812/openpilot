"""LongControl/owner feedback contracts with declared measured inputs.

These are software command tests. The frozen CAN-driven short-response plants
must separately check the required motion under their declared assumptions.
"""

from dataclasses import replace
import math
from unittest.mock import patch

import pytest

from opendbc.car import structs
from opendbc.car.honda.carstate import CarState
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.cereal import custom
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.lib.motion_supervisor import DEGRADED_ACCEL, MotionIntent, MotionPhase, MotionReferenceKind, StopReason
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import ControlHarness, controller
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import consume, make_planner_owner, make_state
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import CruiseTarget, LongitudinalPlannerSP, LongitudinalPlanSource


INSTANCE_NS = 1_000_000_000


def forward_intent(**changes):
  return replace(MotionIntent(release_allowed=True, allow_creep=False, reference_kind=int(MotionReferenceKind.FORWARD),
                              producer_instance_ns=INSTANCE_NS), **changes)


def start_release(control=None):
  harness = ControlHarness(control)
  harness.hold()
  for _ in range(100):
    before = harness.control.last_output_accel
    output = harness.step(0.3, forward_intent())
    assert output - before <= 0.08 + 1e-12
    if harness.control.launch_response is not None:
      assert harness.control.launch_response.bias == 0.0
    if harness.control.motion_phase == MotionPhase.TRACK:
      return harness
    assert harness.control.motion_phase == MotionPhase.RELEASE
  pytest.fail('Existing owner did not complete the declared release')


def learned_feedback():
  harness = start_release()
  for _ in range(50):
    harness.step(0.3, forward_intent())
  assert harness.control.launch_response.bias > 0.1
  return harness


def test_release_is_frozen_then_existing_track_can_correct_missing_response():
  harness = start_release()
  before = harness.control.last_output_accel
  output = harness.step(0.3, forward_intent())
  assert before < output <= before + 0.08
  assert harness.control.launch_response.bias > 0.0
  assert harness.state.aEgo == 0.0  # a declared measurement, not a fabricated response


def test_track_does_not_require_a_second_release_permission_for_fresh_forward_input():
  harness = learned_feedback()
  before = harness.control.launch_response.bias
  for _ in range(20):
    harness.step(0.3, forward_intent(release_allowed=False))
    assert harness.control.motion_phase == MotionPhase.TRACK
  assert harness.control.launch_response.bias > before


def test_moving_handoff_has_no_speed_threshold_step_or_new_learning():
  harness = learned_feedback()
  learned = harness.control.launch_response.bias
  moving = forward_intent(reference_kind=int(MotionReferenceKind.MOVING), release_allowed=False)
  for speed in (0.0, 0.29, 0.31, 1.19, 1.21, 1.99, 2.01):
    harness.state.vEgo = speed
    harness.state.standstill = speed == 0.0
    output = harness.step(0.3, moving)
    assert output == pytest.approx(0.3 + learned)
    assert harness.control.launch_response.bias == pytest.approx(learned)
  harness.state.aEgo = 0.6
  harness.step(0.3, moving)
  assert 0.0 < harness.control.launch_response.bias < learned


@pytest.mark.parametrize('kind', [MotionReferenceKind.MOVING, MotionReferenceKind.EXACT_ZERO, MotionReferenceKind.NO_FORWARD])
def test_flat_or_zero_input_cannot_arm_feedback_from_hold(kind):
  harness = ControlHarness()
  harness.hold()
  intent = forward_intent(reference_kind=int(kind), release_allowed=False)
  for _ in range(100):
    assert harness.step(0.3, intent) <= -0.2
    assert harness.control.motion_phase == MotionPhase.HOLD
    assert not harness.control.launch_response.armed


@pytest.mark.parametrize('speed', [0.29, 0.31, 2.01, 20.0])
@pytest.mark.parametrize('kind', [MotionReferenceKind.FORWARD, MotionReferenceKind.MOVING, MotionReferenceKind.UNKNOWN])
def test_tracking_without_qualified_departure_cannot_learn_positive_feedback(speed, kind):
  harness = ControlHarness()
  harness.state.vEgo = speed
  for _ in range(100):
    output = harness.step(0.3, forward_intent(reference_kind=int(kind), release_allowed=False))
    assert output <= 0.3
    assert harness.control.motion_phase == MotionPhase.TRACK
    assert not harness.control.launch_response.armed
    assert harness.control.pid.i == 0.0


def rolling_feedback(*, braking=False):
  harness = ControlHarness()
  harness.state.vEgo = 20.0
  harness.state.aEgo = 0.5 if braking else 0.0
  intent = forward_intent(reference_kind=int(MotionReferenceKind.UNKNOWN), release_allowed=False) if braking else forward_intent()
  for _ in range(200):
    target = 0.0 if braking else 0.3
    output = harness.step(target, intent)
  assert abs(harness.control.pid.i) > 0.1
  assert output == pytest.approx(target + harness.control.pid.i)
  return harness


def test_cold_moving_reference_cannot_integrate_the_real_speed_filter_startup_transient():
  parameters = CarInterface.get_non_essential_params(CAR.HONDA_ACCORD)
  parameters_sp = CarInterface.get_non_essential_params_sp(parameters, CAR.HONDA_ACCORD)
  measurement = CarState(parameters, parameters_sp)
  harness = ControlHarness()
  moving = forward_intent(reference_kind=int(MotionReferenceKind.MOVING), release_allowed=False)
  measured_accelerations = []
  for _ in range(200):
    # No warm-up or estimator reset: a real constant 2 m/s input exercises the
    # startup transient seen in the frozen low-speed following reproduction.
    harness.state.vEgo, harness.state.aEgo = measurement.update_speed_kf(2.0)
    measured_accelerations.append(harness.state.aEgo)
    before = harness.control.last_output_accel
    output = harness.step(0.6, moving)
    assert output == pytest.approx(min(0.6, before + 0.08))
    assert harness.control.pid.i == 0.0
    assert harness.control.motion_phase == MotionPhase.TRACK
  assert max(measured_accelerations) > 10.0  # retain the trigger, not a warmed sensor


@pytest.mark.parametrize('kind', [MotionReferenceKind.FORWARD, MotionReferenceKind.MOVING])
@pytest.mark.parametrize('target', [-0.1, 0.3])
def test_acc_to_model_reference_retires_negative_i_but_keeps_the_upward_slew_limit(kind, target):
  harness = rolling_feedback(braking=True)
  assert harness.control.pid.i < -0.1
  model = forward_intent(reference_kind=int(kind), release_allowed=False)
  for _ in range(30):
    before = harness.control.last_output_accel
    output = harness.step(target, model)
    assert harness.control.pid.i == 0.0
    assert output == pytest.approx(min(target, before + 0.08))
    assert harness.control.motion_phase == MotionPhase.TRACK
  assert harness.control.last_output_accel == pytest.approx(target)


def test_qualified_rolling_feedback_uses_pid_i_once_and_moving_does_not_ratchet_it():
  harness = rolling_feedback()
  learned = harness.control.pid.i
  assert harness.control.launch_response.bias == learned
  moving = forward_intent(reference_kind=int(MotionReferenceKind.MOVING), release_allowed=False)
  for speed in (20.0, 2.01, 0.31, 0.29, 0.31, 20.0):
    harness.state.vEgo = speed
    assert harness.step(0.3, moving) == pytest.approx(0.3 + learned)
    assert harness.control.pid.i == pytest.approx(learned)
  harness.state.aEgo = 0.6
  harness.step(0.3, moving)
  assert 0.0 < harness.control.pid.i < learned


def test_unknown_acc_reference_revokes_old_drive_and_only_learns_nonpositive_correction():
  harness = rolling_feedback()
  unknown = forward_intent(reference_kind=int(MotionReferenceKind.UNKNOWN))
  for _ in range(100):
    assert harness.step(0.3, unknown) <= 0.3
    assert harness.control.pid.i == 0.0
  assert not harness.control.launch_response.armed
  # UNKNOWN erased the previous FORWARD qualification, even though the raw
  # ACC releaseAllowed field was true. A kind alone cannot restore it.
  for _ in range(20):
    harness.step(0.3, forward_intent(release_allowed=False))
    assert harness.control.pid.i == 0.0
  harness.state.aEgo = 0.6
  for _ in range(20):
    assert harness.step(0.3, unknown) <= 0.3
  assert harness.control.pid.i < 0.0


def test_real_acc_publication_without_model_reference_reaches_braking_feedback():
  """Normal ACC uses UNKNOWN by design; do not relabel it as model FORWARD."""
  def update_targets(instance, state, v_ego, a_ego, v_cruise):
    instance.source = LongitudinalPlanSource.cruise
    instance.cruise_target = CruiseTarget(instance.source, v_cruise, a_ego)
    return instance.cruise_target

  with OpenpilotPrefix():
    planner, control = make_planner_owner(speed=20.8)
    state = make_state(speed=20.8, target=20.0)
    with patch.object(planner, 'is_e2e', return_value=False), \
         patch.object(LongitudinalPlannerSP, 'update', return_value=None), \
         patch.object(LongitudinalPlannerSP, 'update_targets', new=update_targets), \
         patch.object(control.launch_response, 'propose', wraps=control.launch_response.propose) as feedback:
      for frame in range(20):
        state['controlsState'].longControlState = control.long_control_state
        plan, outputs = consume(planner, control, state, frame)
        assert planner.reference_planner.model_evidence is None
        assert plan.motionIntent.referenceKind == MotionReferenceKind.UNKNOWN
        assert plan.motionIntent.valid and plan.solverStatus == 0
        assert control.motion_phase == MotionPhase.TRACK
        assert not control.motion_input_constrained
        assert all(output <= plan.aTarget for output in outputs)
      assert feedback.call_count == 100
      assert control.pid.i < 0.0
      assert not control.launch_response.armed


def test_qualified_launch_to_rolling_keeps_one_integral_without_a_threshold_jump():
  harness = learned_feedback()
  learned = harness.control.pid.i
  harness.state.aEgo = 0.3
  harness.state.standstill = False
  for speed in (0.29, 0.30, 0.31, 0.29, 20.0):
    harness.state.vEgo = speed
    assert harness.step(0.3, forward_intent()) == pytest.approx(0.3 + learned)
    assert harness.control.pid.i == pytest.approx(learned)
  assert not harness.control.launch_response.armed


@pytest.mark.parametrize('kind', [MotionReferenceKind.REST, MotionReferenceKind.EXACT_ZERO, MotionReferenceKind.NO_FORWARD,
                                 MotionReferenceKind.UNKNOWN])
def test_lost_forward_intent_clears_correction_before_computing_output(kind):
  harness = learned_feedback()
  intent = forward_intent(reference_kind=int(kind), release_allowed=False)
  output = harness.step(0.3, intent)
  assert harness.control.launch_response.bias == 0.0
  assert not harness.control.launch_response.armed
  assert output <= 0.3


@pytest.mark.parametrize('reason', [StopReason.MPC, StopReason.E2E, StopReason.CRUISE, StopReason.CLOSE_LEAD, StopReason.FORCE_DECEL])
def test_stop_reason_or_braking_never_spends_a_stored_positive_correction(reason):
  harness = learned_feedback()
  assert harness.step(-3.0, forward_intent(reasons=reason, release_allowed=False)) == -3.0
  assert harness.control.launch_response.bias == 0.0
  assert not harness.control.launch_response.armed


def test_negative_target_without_stop_flag_removes_correction_in_the_same_frame():
  harness = learned_feedback()
  harness.state.vEgo = 1.0
  harness.state.standstill = False
  # Model TRACK revokes positive I without adding a new braking correction.
  assert harness.step(-0.3, forward_intent(release_allowed=False)) == -0.3
  assert harness.control.pid.i == 0.0
  assert not harness.control.launch_response.armed


@pytest.mark.parametrize('override', ['inactive', 'gas', 'brake'])
def test_manual_override_cannot_retain_feedback_for_a_later_release(override):
  harness = learned_feedback()
  harness.state.gasPressed = override == 'gas'
  harness.state.brakePressed = override == 'brake'
  assert harness.step(0.3, forward_intent(), active=override != 'inactive') == 0.0
  assert harness.control.motion_phase == MotionPhase.OFF
  assert not harness.control.launch_response.armed
  harness.state.gasPressed = harness.state.brakePressed = False
  output = harness.step(0.3, forward_intent())
  assert output <= 0.08
  assert harness.control.launch_response.bias == 0.0


@pytest.mark.parametrize('fault', ['plan_invalid', 'stale', 'clock', 'instance'])
def test_invalid_transport_or_another_instance_drops_feedback(fault):
  harness = learned_feedback()
  options = {}
  intent = forward_intent()
  if fault == 'plan_invalid':
    options['plan_valid'] = False
  elif fault == 'stale':
    options['plan_time_ns'] = harness.now_ns - 2_000_000_000
    options['model_time_ns'] = options['plan_time_ns']
  elif fault == 'clock':
    options['now_ns'] = harness.now_ns
  else:
    intent = forward_intent(producer_instance_ns=INSTANCE_NS + 1)
  output = harness.step(0.3, intent, **options)
  assert harness.control.launch_response.bias == 0.0
  assert not harness.control.launch_response.armed
  assert output <= 0.3


@pytest.mark.parametrize('status', [-4, -5])
def test_rest_rejection_and_zero_chain_take_priority_over_learned_launch_effort(status):
  harness = learned_feedback()
  harness.state.vEgo = 1.0
  harness.state.standstill = False
  rejected = forward_intent(valid=False, release_allowed=False, reference_kind=int(MotionReferenceKind.REST),
                            braking_accel=-3.5, rest_goal_rejected=True)
  assert harness.step(-3.5, rejected, plan_valid=False, braking_plan_valid=True, solver_status=status) == -3.5
  assert harness.control.launch_response.bias == 0.0
  assert not harness.control.launch_response.armed
  zero = forward_intent(reference_kind=int(MotionReferenceKind.EXACT_ZERO), release_allowed=False)
  for _ in range(20):
    assert harness.step(-0.1, zero) == -3.5
    assert harness.control.launch_response.bias == 0.0
  assert harness.step(0.3, forward_intent()) <= -3.42 + 1e-12
  assert harness.control.launch_response.bias == 0.0


def test_tightened_limit_cannot_leave_hidden_effort_for_the_next_frame():
  harness = learned_feedback()
  assert harness.step(0.3, forward_intent(), accel_limits=(-3.5, 0.31)) == 0.31
  assert harness.control.launch_response.bias <= 0.01 + 1e-12
  harness.state.aEgo = 0.3
  assert harness.step(0.3, forward_intent()) <= 0.31 + 1e-12


def test_experimental_allocator_and_configured_integrator_are_not_combined_with_feedback():
  harness = learned_feedback()
  harness.control._response_candidate = lambda *args, **kwargs: 0.3
  assert harness.step(0.3, forward_intent()) == 0.3
  assert not harness.control.launch_response.armed
  assert harness.control.pid.i == 0.0
  parameters = structs.CarParams()
  parameters.carFingerprint = CAR.HONDA_ACCORD
  parameters.openpilotLongitudinalControl = True
  parameters.stopAccel = -2.0
  parameters.longitudinalTuning.kiBP = [0.0]
  parameters.longitudinalTuning.kiV = [1.0]
  control = LongControl(parameters, custom.CarParamsSP.new_message())
  assert control.launch_response is None


@pytest.mark.parametrize('changes', [
  {'platform': CAR.HONDA_CIVIC_BOSCH}, {'platform': CAR.HONDA_CRV_HYBRID}, {'platform': CAR.HONDA_CIVIC},
  {'openpilot_long': False}, {'interceptor': True},
])
def test_mode_platform_and_authority_scope_remains_explicit(changes):
  assert controller(**changes).launch_response is None


def test_approach_freezes_actual_negative_feedback_then_hands_it_to_low_speed_stop():
  harness = rolling_feedback(braking=True)
  braking = forward_intent(reasons=StopReason.E2E, release_allowed=False)
  correction = harness.control.pid.i
  assert harness.step(-3.0, braking) == pytest.approx(-3.0 + correction)
  assert harness.control.motion_phase == MotionPhase.APPROACH
  assert harness.control.pid.i == pytest.approx(correction)
  assert harness.step(-3.0, braking, accel_limits=(-3.1, 2.0)) == -3.1
  assert harness.control.pid.i == pytest.approx(-0.1)
  for _ in range(50):
    harness.step(-0.5, braking)
    assert harness.control.pid.i == pytest.approx(-0.1)
  assert harness.control.last_output_accel == pytest.approx(-0.6)
  before = harness.control.last_output_accel
  harness.state.vEgo = 0.2
  assert harness.step(-0.5, braking) <= before
  assert harness.control.pid.i == 0.0
  harness.state.vEgo = 0.0
  harness.state.standstill = True
  assert harness.step(-0.5, braking) <= before
  assert harness.control.motion_phase == MotionPhase.HOLD
  assert harness.control.pid.i == 0.0


@pytest.mark.parametrize('fault', ['plan_invalid', 'clock', 'gap'])
def test_fault_clears_signed_feedback_after_owned_braking_takes_over(fault):
  harness = rolling_feedback(braking=True)
  braking = forward_intent(reasons=StopReason.E2E, release_allowed=False)
  before = harness.step(-0.5, braking)
  options = {'plan_invalid': {'plan_valid': False}, 'clock': {'now_ns': harness.now_ns},
             'gap': {'now_ns': harness.now_ns + 100_000_001}}[fault]
  assert harness.step(0.3, forward_intent(), **options) <= before
  assert harness.control.pid.i == 0.0
  assert harness.control.motion_input_constrained


@pytest.mark.parametrize('braking', [False, True])
def test_restarting_producer_or_manual_reentry_cannot_inherit_signed_feedback(braking):
  harness = rolling_feedback(braking=braking)
  restarted = forward_intent(producer_instance_ns=INSTANCE_NS + 1, release_allowed=not braking,
                             reference_kind=int(MotionReferenceKind.UNKNOWN if braking else MotionReferenceKind.FORWARD))
  harness.step(0.0 if braking else 0.3, restarted)
  assert harness.control.pid.i == 0.0
  assert not harness.control.launch_response.armed
  harness.step(0.3, restarted, active=False)
  assert harness.control.pid.i == 0.0
  assert harness.control.motion_phase == MotionPhase.OFF
  harness.step(0.3, restarted)
  assert harness.control.pid.i == 0.0


@pytest.mark.parametrize('fresh', [False, True])
def test_approach_same_producer_preserves_braking_for_cached_and_fresh_input(fresh):
  harness = rolling_feedback(braking=True)
  braking = forward_intent(reasons=StopReason.E2E, release_allowed=False)
  before = harness.step(-0.5, braking)
  correction = harness.control.pid.i
  for _ in range(10):
    assert harness.step(-0.5, braking, new_input=fresh) == pytest.approx(before)
    assert harness.control.pid.i == pytest.approx(correction)
    assert harness.control.motion_phase == MotionPhase.APPROACH
    assert not harness.control.motion_input_constrained
    assert not harness.control.launch_response.braking_handoff


def test_approach_new_producer_hands_actual_brake_to_owner_without_inheriting_i():
  harness = rolling_feedback(braking=True)
  braking = forward_intent(reasons=StopReason.E2E, release_allowed=False)
  before = harness.step(-0.5, braking)
  assert before < -0.5 and harness.control.pid.i < 0.0
  restarted = replace(braking, producer_instance_ns=INSTANCE_NS + 1)
  for index in range(10):
    assert harness.step(-0.5, restarted, new_input=index % 2 == 0) == pytest.approx(before)
    assert harness.control.motion_phase == MotionPhase.APPROACH
    assert not harness.control.motion_input_constrained  # restart itself is legitimate
    assert harness.control.pid.i == 0.0
    assert harness.control.launch_response.braking_handoff
  assert harness.step(-2.0, restarted) == -2.0  # stronger braking is never delayed
  assert harness.step(-0.3, restarted) == -2.0
  release = replace(restarted, reasons=0, release_allowed=True)
  assert -2.0 < harness.step(0.3, release) <= -1.92 + 1e-12
  assert harness.control.motion_phase == MotionPhase.RELEASE
  assert not harness.control.launch_response.braking_handoff
  assert harness.control.pid.i == 0.0


def test_approach_older_producer_uses_fault_envelope_after_clearing_i():
  harness = rolling_feedback(braking=True)
  braking = forward_intent(reasons=StopReason.E2E, release_allowed=False)
  before = harness.step(-0.5, braking)
  retired = replace(braking, producer_instance_ns=INSTANCE_NS - 1)
  for _ in range(3):
    assert harness.step(-0.5, retired) <= before
    assert harness.control.motion_input_constrained
    assert harness.control.pid.i == 0.0
    assert not harness.control.launch_response.braking_handoff


@pytest.mark.parametrize('clock', ['duplicate', 'backward', 'gap'])
def test_approach_invalid_control_clock_uses_owned_brake_after_clearing_i(clock):
  harness = rolling_feedback(braking=True)
  braking = forward_intent(reasons=StopReason.E2E, release_allowed=False)
  before = harness.step(-0.5, braking)
  timestamp = {'duplicate': harness.now_ns, 'backward': harness.now_ns - 1,
               'gap': harness.now_ns + 100_000_001}[clock]
  assert harness.step(-0.5, braking, now_ns=timestamp) <= before
  assert harness.control.motion_input_constrained
  assert harness.control.pid.i == 0.0
  assert not harness.control.launch_response.braking_handoff


@pytest.mark.parametrize('learned_i_with_positive_output', [False, True])
def test_cold_approach_or_prior_positive_output_does_not_gain_a_braking_handoff(learned_i_with_positive_output):
  harness = ControlHarness()
  harness.state.vEgo = 20.0
  if learned_i_with_positive_output:
    harness.state.aEgo = 0.6
    for _ in range(200):
      harness.step(0.5, forward_intent(reference_kind=int(MotionReferenceKind.UNKNOWN), release_allowed=False))
    assert harness.control.pid.i < 0.0
    assert harness.control.last_output_accel > 0.0
  braking = forward_intent(reasons=StopReason.E2E, release_allowed=False, producer_instance_ns=INSTANCE_NS + 1)
  before = harness.step(-0.5, braking)
  assert before == -0.5
  assert harness.control.pid.i == 0.0
  assert not harness.control.launch_response.braking_handoff
  assert before < harness.step(-0.3, braking) <= before + 0.08 + 1e-12
  assert not harness.control.launch_response.braking_handoff


@pytest.mark.parametrize('history', [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize('input_kind', ['fresh_braking', 'stale', 'future', 'positive_envelope', 'missing_envelope',
                                       'unqualified_transport', 'normal_positive'])
def test_corrupt_output_history_keeps_only_independently_qualified_fresh_braking(history, input_kind):
  harness = ControlHarness()
  harness.state.vEgo = 20.0
  for _ in range(3):
    harness.step(-0.4, forward_intent())
  assert harness.control.motion_phase == MotionPhase.TRACK
  assert harness.control.last_output_accel < 0.0
  harness.control.last_output_accel = history

  target = -1.5
  intent = MotionIntent(valid=False, allow_creep=False, braking_accel=target, producer_instance_ns=INSTANCE_NS)
  options = {'plan_valid': False, 'braking_plan_valid': True, 'solver_status': 4}
  if input_kind == 'stale':
    options.update(plan_time_ns=harness.now_ns - 1_000_000_000, model_time_ns=harness.now_ns - 1_000_000_000)
  elif input_kind == 'future':
    options.update(plan_time_ns=harness.now_ns + 20_000_000, model_time_ns=harness.now_ns + 20_000_000)
  elif input_kind == 'positive_envelope':
    target = 0.3
    intent = replace(intent, braking_accel=target)
  elif input_kind == 'missing_envelope':
    intent = replace(intent, braking_accel=None)
  elif input_kind == 'unqualified_transport':
    options['braking_plan_valid'] = False
  elif input_kind == 'normal_positive':
    target = 0.3
    intent = forward_intent()
    options.update(plan_valid=True, solver_status=0)

  expected = -1.5 if input_kind == 'fresh_braking' else DEGRADED_ACCEL
  assert harness.step(target, intent, **options) == expected
  assert harness.control.motion_degraded
  assert harness.control.pid.i == 0.0
