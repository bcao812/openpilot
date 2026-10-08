"""LongControl/owner feedback contracts with declared measured inputs.

These are software command tests. The frozen CAN-driven short-response plants
must separately check the required motion under their declared assumptions.
Retired PURE inputs are injected only to verify rejection of their old contract.
"""

from dataclasses import replace
import math
from unittest.mock import patch

import numpy as np
import pytest

from opendbc.car import structs
from opendbc.car.honda.carstate import CarState
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, CarControllerParams
from openpilot.cereal import custom
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.lib.motion_supervisor import (DEGRADED_ACCEL, MotionIntent, MotionPhase, MotionReferenceKind, MotionResponseKind,
                                                               StopReason)
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import ControlHarness, controller, decoded_commands
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import consume, make_planner_owner, make_state
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import CruiseTarget, LongitudinalPlannerSP, LongitudinalPlanSource


INSTANCE_NS = 1_000_000_000


def forward_intent(**changes):
  return replace(MotionIntent(release_allowed=True, allow_creep=False, reference_kind=int(MotionReferenceKind.FORWARD),
                              producer_instance_ns=INSTANCE_NS), **changes)


def absolute_intent(**changes):
  return replace(MotionIntent(allow_creep=False, response_kind=int(MotionResponseKind.ABSOLUTE_ACC),
                              producer_instance_ns=INSTANCE_NS), **changes)


def pure_intent(**changes):
  """Historical contract; the current owner must reject it."""
  return replace(absolute_intent(response_kind=int(MotionResponseKind.PURE_CRUISE)), **changes)


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
  intent = absolute_intent() if braking else forward_intent()
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


def test_unknown_reference_without_response_contract_revokes_feedback_and_cannot_learn():
  harness = rolling_feedback()
  unknown = forward_intent(reference_kind=int(MotionReferenceKind.UNKNOWN))
  for _ in range(100):
    assert harness.step(0.3, unknown) <= 0.3
    assert harness.control.pid.i == 0.0
  assert not harness.control.launch_response.armed
  # UNKNOWN erased the previous FORWARD qualification, even though the raw
  # releaseAllowed field was true. A kind alone cannot restore it.
  for _ in range(20):
    harness.step(0.3, forward_intent(release_allowed=False))
    assert harness.control.pid.i == 0.0
  harness.state.aEgo = 0.6
  for _ in range(20):
    assert harness.step(0.3, unknown) <= 0.3
  assert harness.control.pid.i == 0.0


def ordinary_cruise_target(instance, state, v_ego, a_ego, v_cruise):
  instance.source = LongitudinalPlanSource.cruise
  instance.cruise_target = CruiseTarget(instance.source, v_cruise, a_ego)
  return instance.cruise_target


def assert_honda_commands_match_requests(outputs):
  for output, command in zip(outputs, decoded_commands(outputs), strict=True):
    serialized = float(np.float32(output))
    braking = serialized < CarControllerParams.BOSCH_GAS_LOOKUP_BP[0]
    # ACC_CONTROL encodes acceleration at 0.01 m/s^2 resolution. Honda's
    # braking boundary is its gas lookup threshold, not acceleration zero.
    assert command['ACCEL_COMMAND'] == pytest.approx(serialized, rel=0.0, abs=0.005 + 1e-12)
    assert bool(command['BRAKE_REQUEST']) == bool(command['BRAKE_LIGHTS']) == braking
    assert not (command['BRAKE_REQUEST'] and command['GAS_COMMAND'] > 0.0)


@pytest.mark.parametrize('speed,request_sign', [(20.8, -1), (19.2, 1)])
def test_real_acc_publication_without_model_reference_executes_the_winning_cruise_request(speed, request_sign):
  """The real governor/native arbitration owns the request before execution."""

  with OpenpilotPrefix():
    planner, control = make_planner_owner(speed=speed)
    state = make_state(speed=speed, target=20.0)
    requests = []
    with patch.object(planner, 'is_e2e', return_value=False), \
         patch.object(LongitudinalPlannerSP, 'update', return_value=None), \
         patch.object(LongitudinalPlannerSP, 'update_targets', new=ordinary_cruise_target), \
         patch.object(control.launch_response, 'propose', wraps=control.launch_response.propose) as feedback:
      for frame in range(20):
        state['controlsState'].longControlState = control.long_control_state
        previous = control.last_output_accel
        plan, outputs = consume(planner, control, state, frame)
        assert planner.reference_planner.model_evidence is None
        assert plan.motionIntent.referenceKind == MotionReferenceKind.UNKNOWN
        assert plan.motionIntent.responseKind == (MotionResponseKind.NONE if frame == 0 else MotionResponseKind.CRUISE_REQUEST)
        assert control.pid.i == 0.0
        assert plan.motionIntent.valid and plan.solverStatus == 0
        assert control.motion_phase == MotionPhase.TRACK
        assert not control.motion_input_constrained
        assert plan.aTarget * request_sign > 0.0
        if frame > 0:
          assert plan.aTarget == float(np.float32(planner.reference_planner.constraint_accel))
        for output in outputs:
          # Startup can hand off a stronger native brake. Keep the existing
          # +8 m/s^3 owner limit while executing the dated final request.
          assert output == pytest.approx(min(plan.aTarget, previous + 0.08), rel=0.0, abs=1e-12)
          previous = output
        requests.extend(outputs)
      assert feedback.call_count == 0
      assert requests[-1] == plan.aTarget
      assert not control.launch_response.armed
    assert_honda_commands_match_requests(requests)


@pytest.mark.parametrize('speed,correction_sign', [(20.8, -1), (19.2, 1)])
def test_native_continuation_winner_publication_keeps_only_nonpositive_response(speed, correction_sign):
  """A finite risk continuation may win without acquiring a numeric cruise goal.

  Native solves still run. A declared sampled target and a neutral governor
  ceiling make that sample win actual arbitration, independently of the real
  cruise governor's history. Actual MPC inputs retain cruise_speed=None;
  this is a feedback contract, not a trajectory test or a manufactured PURE path.
  """
  target = correction_sign * 0.3
  with OpenpilotPrefix():
    planner, control = make_planner_owner(speed=speed)
    state = make_state(speed=speed, target=20.0)
    requests = []
    with patch.object(planner, 'is_e2e', return_value=False), \
         patch.object(LongitudinalPlannerSP, 'update', return_value=None), \
         patch.object(LongitudinalPlannerSP, 'update_targets', new=ordinary_cruise_target), \
         patch.object(planner.mpc, 'update', wraps=planner.mpc.update) as native_update, \
         patch.object(planner.reference_planner.cruise_response, 'update',
                      side_effect=lambda _target, _speed, maximum, **_kwargs: maximum), \
         patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.sample_reference_acceleration', return_value=target), \
         patch.object(control.launch_response, 'propose', wraps=control.launch_response.propose) as feedback:
      for frame in range(20):
        state['controlsState'].longControlState = control.long_control_state
        plan, outputs = consume(planner, control, state, frame)
        assert planner.reference_planner.model_evidence is None
        assert plan.motionIntent.referenceKind == MotionReferenceKind.UNKNOWN
        assert planner.reference_planner.reference_accel < planner.reference_planner.constraint_accel
        assert plan.aTarget == plan.reference.accel == float(np.float32(target))
        assert plan.motionIntent.responseKind == (MotionResponseKind.NONE if frame == 0 else MotionResponseKind.ABSOLUTE_ACC)
        assert plan.motionIntent.valid and plan.solverStatus == 0
        assert control.motion_phase == MotionPhase.TRACK and not control.motion_input_constrained
        assert control.pid.i <= 0.0
        if frame == 0:
          assert control.pid.i == 0.0
        else:
          assert native_update.call_args.kwargs['reference'].cruise_speed is None
        assert all(output <= plan.aTarget for output in outputs)
        requests.extend(outputs)
      assert feedback.call_count == 95
      if correction_sign < 0:
        assert control.pid.i < 0.0
      else:
        assert control.pid.i == 0.0
      assert not control.launch_response.armed
    assert_honda_commands_match_requests(requests)


@pytest.mark.parametrize('target', [-0.3, 0.3])
def test_retired_pure_contract_revokes_feedback_and_uses_fault_braking(target):
  with OpenpilotPrefix():
    harness = rolling_feedback()
    intent = pure_intent()
    assert not intent.is_valid()
    outputs = []
    for _ in range(10):
      outputs.append(harness.step(target, intent))
      assert harness.control.motion_degraded
      assert harness.control.motion_reasons & StopReason.INVALID
      assert harness.control.pid.i == 0.0
      assert outputs[-1] <= DEGRADED_ACCEL
    assert not harness.control.launch_response.armed
    assert_honda_commands_match_requests(outputs)


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


@pytest.mark.parametrize('kind', [MotionReferenceKind.FORWARD, MotionReferenceKind.MOVING])
def test_explicit_lead_braking_can_correct_a_model_target_then_retires_at_model_handoff(kind):
  harness = rolling_feedback()
  assert harness.control.pid.i > 0.0
  lead = forward_intent(reference_kind=int(kind), response_kind=int(MotionResponseKind.LEAD_BRAKING), release_allowed=False)
  for _ in range(100):
    assert harness.step(-0.5, lead) < -0.5
    assert harness.control.pid.i < 0.0
    assert harness.control.motion_phase == MotionPhase.TRACK
  model = replace(lead, response_kind=int(MotionResponseKind.NONE))
  before = harness.control.last_output_accel
  assert harness.step(0.3, model) <= before + 0.08 + 1e-12
  assert harness.control.pid.i == 0.0
  for _ in range(30):
    harness.step(0.3, model)
    assert harness.control.pid.i == 0.0


@pytest.mark.parametrize('target', [0.0, 0.3])
def test_lead_contract_cannot_continue_learning_after_negative_target_ends(target):
  harness = rolling_feedback(braking=True)
  lead = forward_intent(reference_kind=int(MotionReferenceKind.MOVING), response_kind=int(MotionResponseKind.LEAD_BRAKING),
                        release_allowed=False)
  harness.step(-0.5, lead)
  assert harness.control.pid.i < 0.0
  harness.step(target, lead)
  assert harness.control.pid.i == 0.0
  assert not harness.control.motion_input_constrained


def test_model_positive_effort_is_not_ordinary_acc_authority():
  harness = rolling_feedback()
  assert harness.control.pid.i > 0.0
  assert harness.step(-0.5, absolute_intent()) <= -0.5
  assert harness.control.pid.i <= 0.0


def test_absolute_response_does_not_release_hold_or_learn_during_release():
  harness = ControlHarness()
  harness.hold()
  for _ in range(50):
    assert harness.step(0.3, absolute_intent()) <= -0.2
    assert harness.control.motion_phase == MotionPhase.HOLD
    assert harness.control.pid.i == 0.0
  for _ in range(50):
    harness.step(0.3, absolute_intent(release_allowed=True))
    assert harness.control.pid.i == 0.0
    assert not harness.control.launch_response.armed
    if harness.control.motion_phase == MotionPhase.TRACK:
      break
  assert harness.control.motion_phase == MotionPhase.TRACK


@pytest.mark.parametrize('release_allowed', [False, True])
def test_retired_pure_contract_cannot_release_a_hold(release_allowed):
  harness = ControlHarness()
  harness.hold()
  for _ in range(10):
    assert harness.step(0.3, pure_intent(release_allowed=release_allowed)) <= -2.0
    assert harness.control.motion_phase == MotionPhase.HOLD
    assert harness.control.motion_degraded
    assert harness.control.pid.i == 0.0


@pytest.mark.parametrize('boundary', ['inactive', 'gas', 'brake', 'stale', 'mutated_contract'])
def test_rolling_feedback_obeys_manual_transport_and_same_plan_identity(boundary):
  harness = rolling_feedback()
  intent = forward_intent()
  assert harness.control.pid.i > 0.0
  options = {}
  if boundary in ('gas', 'brake'):
    setattr(harness.state, boundary + 'Pressed', True)
  elif boundary == 'inactive':
    options['active'] = False
  elif boundary == 'stale':
    options.update(plan_time_ns=harness.now_ns - 2_000_000_000, model_time_ns=harness.now_ns - 2_000_000_000)
  else:
    options['new_input'] = False
    intent = replace(intent, response_kind=int(MotionResponseKind.ABSOLUTE_ACC))
  output = harness.step(0.3, intent, **options)
  assert harness.control.pid.i == 0.0
  if boundary in ('inactive', 'gas', 'brake'):
    assert output == 0.0 and harness.control.motion_phase == MotionPhase.OFF
  else:
    assert output <= DEGRADED_ACCEL and harness.control.motion_input_constrained


@pytest.mark.parametrize('reason', [StopReason.CRUISE, StopReason.FORCE_DECEL, StopReason.E2E])
def test_stop_approach_cannot_spend_previous_model_positive_i(reason):
  harness = rolling_feedback()
  assert harness.control.pid.i > 0.0
  stop = MotionIntent(reasons=int(reason), allow_creep=False, producer_instance_ns=INSTANCE_NS)
  assert harness.step(-0.5, stop) <= -0.5
  assert harness.control.pid.i == 0.0
  assert harness.control.motion_phase == MotionPhase.APPROACH
  assert not harness.control.motion_input_constrained


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


def test_configured_integrator_is_not_combined_with_feedback():
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
                             reference_kind=int(MotionReferenceKind.UNKNOWN if braking else MotionReferenceKind.FORWARD),
                             response_kind=int(MotionResponseKind.ABSOLUTE_ACC if braking else MotionResponseKind.NONE))
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
      harness.step(0.5, absolute_intent())
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
