"""A withdrawn target cannot remain an independent stop owner.

These cases start with an executed close-lead stop and then supply a healthy
current intent without that target. Vehicle HOLD, current stop requests and
unhealthy transport are separate negative controls.
"""

import pytest

from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent, MotionPhase, MotionReferenceKind, StopReason
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import warm_tracking_harness
from openpilot.selfdrive.controls.tests.test_motion_supervisor import MotionHarness


@pytest.mark.parametrize('speed', [0.2, 0.3, 1.3])
@pytest.mark.parametrize('target', [-1.6, -0.7, 0.0, 0.15])
def test_a_current_target_replaces_a_rolling_close_only_stop(speed, target):
  harness = MotionHarness()
  assert harness.step(MotionIntent(reasons=StopReason.CLOSE_LEAD, allow_creep=False), v_ego=1.3, a_target=-0.7)
  assert harness.supervisor.phase == MotionPhase.APPROACH

  # No release permission or positive scalar is synthesized for the missing
  # object. The valid current plan's own scalar remains the requested action.
  assert not harness.step(MotionIntent(allow_creep=False), v_ego=speed, a_target=target)
  assert harness.supervisor.phase == MotionPhase.RELEASE
  assert harness.supervisor.reasons == StopReason.NONE
  assert not harness.supervisor.degraded


def test_new_plan_can_withdraw_close_target_while_latest_model_is_unchanged():
  harness = MotionHarness()
  assert harness.step(MotionIntent(reasons=StopReason.CLOSE_LEAD), v_ego=1.3, a_target=-0.7)
  # A new plan using the latest still-current model can consume changed radar
  # membership. Requiring both IDs to advance would retain the old target.
  assert not harness.step(MotionIntent(allow_creep=False), v_ego=1.3, a_target=-0.35, new_model=False)
  assert harness.supervisor.reasons == StopReason.NONE
  assert not harness.supervisor.degraded


def test_mutating_an_already_consumed_plan_cannot_fake_target_withdrawal():
  harness = MotionHarness()
  assert harness.step(MotionIntent(reasons=StopReason.CLOSE_LEAD), v_ego=1.3, a_target=-0.7)
  assert harness.step(MotionIntent(allow_creep=False), v_ego=1.3, a_target=-0.7, new_plan=False, new_model=False)
  assert harness.supervisor.degraded
  assert harness.supervisor.reasons & StopReason.INVALID


@pytest.mark.parametrize('reason', [StopReason.MPC, StopReason.E2E, StopReason.CRUISE, StopReason.CLOSE_LEAD, StopReason.FORCE_DECEL])
def test_a_current_stop_request_still_owns_stopping_after_previous_close_target(reason):
  harness = MotionHarness()
  assert harness.step(MotionIntent(reasons=StopReason.CLOSE_LEAD), v_ego=1.3, a_target=-0.7)
  assert harness.step(MotionIntent(reasons=reason, release_allowed=True, allow_creep=False), v_ego=1.3, a_target=0.5)
  assert harness.supervisor.reasons == reason
  assert not harness.supervisor.degraded


@pytest.mark.parametrize('speed,standstill', [(0.0, True), (0.2, False), (1.3, False)])
def test_actual_hold_uses_vehicle_hold_reason_after_close_target_is_withdrawn(speed, standstill):
  harness = MotionHarness()
  assert harness.step(MotionIntent(reasons=StopReason.CLOSE_LEAD), v_ego=0.0, standstill=True, a_target=-0.7)
  assert harness.supervisor.phase == MotionPhase.HOLD
  assert harness.step(MotionIntent(allow_creep=False), v_ego=speed, standstill=standstill, a_target=0.5)
  assert harness.supervisor.reasons == StopReason.HOLD
  assert not harness.supervisor.degraded

  assert not harness.step(MotionIntent(release_allowed=True, allow_creep=False),
                          v_ego=speed, standstill=standstill, a_target=0.5)
  assert harness.supervisor.phase == MotionPhase.RELEASE
  assert harness.supervisor.reasons == StopReason.NONE


@pytest.mark.parametrize('input_kwargs', [{'plan_valid': False}, {'solver_status': -1}, {'solver_status': 4}])
def test_unhealthy_input_does_not_claim_a_healthy_target_withdrawal(input_kwargs):
  harness = MotionHarness()
  assert harness.step(MotionIntent(reasons=StopReason.CLOSE_LEAD), v_ego=1.3, a_target=-0.7)
  assert harness.step(MotionIntent(allow_creep=False), v_ego=1.3, a_target=0.5, **input_kwargs)
  assert harness.supervisor.degraded
  assert harness.supervisor.reasons & StopReason.INVALID


@pytest.mark.parametrize('target', [-0.35, 0.0])
def test_withdrawal_removes_added_stop_ramp_but_follows_current_nonpositive_target(target):
  harness = warm_tracking_harness(1.3, -0.4)
  assert harness.step(-0.7, MotionIntent(reasons=StopReason.CLOSE_LEAD, allow_creep=False)) <= -0.7
  assert harness.control.resolved_should_stop
  outputs = []
  for _ in range(12):
    outputs.append(harness.step(target, MotionIntent(allow_creep=False)))
    assert not harness.control.resolved_should_stop
    assert not harness.control.motion_degraded
    assert harness.control.motion_reasons == StopReason.NONE
  # Current command smoothing can retain prior braking for this finite
  # transition; no absent-object stop ramp or extra positive action is added.
  assert all(-0.7 <= output <= target for output in outputs)
  assert outputs[-1] == pytest.approx(target)


@pytest.mark.parametrize('rejected_status', [0, -4, -5])
@pytest.mark.parametrize('current_target', [-0.3, 0.0])
def test_current_zero_does_not_inherit_previous_rest_executed_braking(rejected_status, current_target):
  harness = warm_tracking_harness(10.0, -0.5)
  rest = MotionIntent(reference_kind=MotionReferenceKind.REST, producer_instance_ns=1_000_000_000,
                      allow_creep=False, valid=rejected_status == 0,
                      rest_goal_rejected=rejected_status != 0, braking_accel=-1.0 if rejected_status else None)
  assert harness.step(-1.0, rest, plan_valid=rejected_status == 0,
                       solver_status=rejected_status, braking_plan_valid=rejected_status != 0) == -1.0
  owner = harness.control.motion_supervisor

  current_zero = MotionIntent(reference_kind=MotionReferenceKind.EXACT_ZERO,
                              producer_instance_ns=1_000_000_000, allow_creep=False)
  previous = -1.0
  for _ in range(30):
    output = harness.step(current_target, current_zero)
    assert previous <= output <= current_target
    assert output - previous <= 0.08 + 1e-9
    previous = output
    assert not owner.resolved_should_stop
    assert not owner.degraded
  assert output == pytest.approx(current_target)


@pytest.mark.parametrize('current_reason', [StopReason.MPC, StopReason.E2E, StopReason.CLOSE_LEAD])
def test_current_zero_preserves_current_stop_and_actual_hold_without_prior_rest_cap(current_reason):
  harness = MotionHarness()
  rest = MotionIntent(reference_kind=MotionReferenceKind.REST, producer_instance_ns=1_000_000_000, allow_creep=False)
  harness.step(rest, a_target=-1.0, v_ego=10.0)
  owner = harness.supervisor

  current_zero = MotionIntent(reference_kind=MotionReferenceKind.EXACT_ZERO, producer_instance_ns=1_000_000_000,
                              reasons=current_reason, allow_creep=False)
  assert harness.step(current_zero, a_target=0.0, v_ego=0.0, standstill=True)
  assert owner.phase == MotionPhase.HOLD
  assert owner.reasons & current_reason
  assert owner.reasons & StopReason.HOLD
  assert not owner.degraded
