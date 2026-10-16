"""Current stop evidence bounds creep assistance, without weakening HOLD.

These are fixed-input controller checks of the B05 conflict, not claims about
delivered vehicle deceleration or closed-loop stopping distance.
"""

from dataclasses import replace

import pytest

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState, ROLLING_STOP_ACCEL, STARTING_ACCEL_RATE
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent, MotionPhase, MotionReferenceKind, MotionResponseKind, StopReason
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import ControlHarness


ROLLING_INTENT = MotionIntent(reference_kind=int(MotionReferenceKind.FORWARD), response_kind=int(MotionResponseKind.ABSOLUTE_ACC),
                              release_allowed=True, producer_instance_ns=1_000_000_000)


@pytest.mark.parametrize('release_allowed', [False, True])
@pytest.mark.parametrize('moving_lead', [False, True])
@pytest.mark.parametrize('target', [-0.215, -0.145])
def test_rolling_follow_keeps_its_negative_target_without_stop_evidence(release_allowed, moving_lead, target):
  harness = ControlHarness()
  harness.state.vEgo = 0.8
  harness.state.aEgo = target
  intent = replace(ROLLING_INTENT, release_allowed=release_allowed, moving_lead=moving_lead)
  for frame in range(250):
    output = harness.step(target, intent, new_input=frame % 5 == 0)
    assert output == pytest.approx(target)
    assert harness.control.long_control_state == LongCtrlState.pid
    assert harness.control.motion_phase == MotionPhase.TRACK
    assert not harness.control.motion_reasons
    assert not harness.control.resolved_should_stop


@pytest.mark.parametrize('release_allowed', [False, True])
def test_rolling_creep_withdrawal_preserves_the_negative_target_and_brake_release_rate(release_allowed):
  harness = ControlHarness()
  harness.state.vEgo = 0.8
  harness.state.aEgo = -0.35
  stop_intent = MotionIntent(reasons=StopReason.CLOSE_LEAD, producer_instance_ns=ROLLING_INTENT.producer_instance_ns)
  for _ in range(150):
    harness.step(-0.35, stop_intent)
  assert harness.control.motion_reasons == StopReason.CLOSE_LEAD | StopReason.CREEP
  intent = replace(ROLLING_INTENT, release_allowed=release_allowed)
  # A plan-only withdrawal removes CLOSE_LEAD but cannot spend fresh model
  # evidence to withdraw CREEP. The next complete input must clear that cause
  # without waiting for departure permission or a positive target.
  harness.step(-0.35, intent, new_input=False, plan_time_ns=harness.now_ns + round(DT_CTRL * 1e9))
  assert harness.control.motion_reasons == StopReason.CREEP
  assert harness.control.last_output_accel <= ROLLING_STOP_ACCEL
  previous = harness.control.last_output_accel
  for _ in range(30):
    output = harness.step(-0.35, intent)
    assert output <= previous + STARTING_ACCEL_RATE * DT_CTRL + 1e-12
    assert output <= -0.35
    assert not harness.control.resolved_should_stop
    previous = output
  assert previous == pytest.approx(-0.35)
  assert harness.control.motion_phase == MotionPhase.TRACK


def test_a_current_raw_stop_still_uses_the_stopping_ramp_and_stronger_braking():
  harness = ControlHarness()
  harness.state.vEgo = 0.8
  harness.state.aEgo = -0.35
  harness.step(-0.35, ROLLING_INTENT)
  stop_intent = MotionIntent(reasons=StopReason.E2E, release_allowed=True, producer_instance_ns=1_000_000_000)
  for _ in range(150):
    output = harness.step(-0.35, stop_intent)
    assert harness.control.resolved_should_stop
    assert harness.control.motion_reasons & StopReason.E2E
    assert harness.control.long_control_state == LongCtrlState.stopping
  assert output <= ROLLING_STOP_ACCEL
  assert harness.step(-3.0, stop_intent) == -3.0


def test_negative_target_with_movement_permission_cannot_release_standstill_braking():
  harness = ControlHarness()
  harness.hold()
  for _ in range(100):
    output = harness.step(-0.35, ROLLING_INTENT)
    assert output <= harness.control.CP.stopAccel
    assert harness.control.motion_phase == MotionPhase.HOLD
    assert harness.control.motion_reasons & StopReason.HOLD


@pytest.mark.parametrize('pedal', ['gasPressed', 'brakePressed'])
def test_manual_override_immediately_revokes_a_permitted_rolling_follow(pedal):
  harness = ControlHarness()
  harness.state.vEgo = 0.8
  harness.state.aEgo = -0.35
  for _ in range(60):
    harness.step(-0.35, ROLLING_INTENT)
  setattr(harness.state, pedal, True)
  assert harness.step(-0.35, ROLLING_INTENT) == 0.0
  assert harness.control.long_control_state == LongCtrlState.off
  assert harness.control.motion_phase == MotionPhase.OFF
  assert harness.control.motion_supervisor.creep_total_t == 0.0
