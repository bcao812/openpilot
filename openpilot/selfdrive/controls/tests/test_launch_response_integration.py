"""LongControl/owner feedback contracts with declared measured inputs.

These are software command tests. The frozen CAN-driven short-response plants
must separately check the required motion under their declared assumptions.
"""

from dataclasses import replace

import pytest

from opendbc.car import structs
from opendbc.car.honda.values import CAR
from openpilot.cereal import custom
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent, MotionPhase, MotionReferenceKind, StopReason
from openpilot.selfdrive.controls.tests.test_unified_longcontrol import ControlHarness, controller


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


def test_tracking_without_an_owner_release_cannot_start_feedback():
  harness = ControlHarness()
  for _ in range(100):
    output = harness.step(0.3, forward_intent())
    assert output <= 0.3
    assert harness.control.motion_phase == MotionPhase.TRACK
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
  assert harness.step(-0.3, forward_intent(release_allowed=False)) == -0.3
  assert harness.control.launch_response.bias == 0.0


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
  parameters = structs.CarParams()
  parameters.carFingerprint = CAR.HONDA_ACCORD
  parameters.openpilotLongitudinalControl = True
  parameters.stopAccel = -2.0
  parameters.longitudinalTuning.kiBP = [0.0]
  parameters.longitudinalTuning.kiV = [1.0]
  control = LongControl(parameters, custom.CarParamsSP.new_message(), unified=True)
  assert control.launch_response is None


@pytest.mark.parametrize('changes', [
  {'unified': False}, {'unified': False, 'supervised': True}, {'platform': CAR.HONDA_CIVIC},
  {'openpilot_long': False}, {'interceptor': True},
])
def test_mode_platform_and_authority_scope_remains_explicit(changes):
  assert controller(**changes).launch_response is None
