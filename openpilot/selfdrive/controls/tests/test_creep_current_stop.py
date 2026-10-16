"""Anti-creep assists a current stop; weak rolling deceleration is insufficient.

These bounded owner checks exercise current evidence and historical CREEP-only
state. They do not claim closed-loop braking, radar fidelity, or road validation.
"""

import pytest

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CREEP_MAX_T, STOP_SPEED
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent, MotionPhase, StopReason
from openpilot.selfdrive.controls.tests.test_motion_supervisor import MotionHarness


CURRENT_STOP_REASONS = (StopReason.MPC, StopReason.E2E, StopReason.CRUISE,
                        StopReason.CLOSE_LEAD, StopReason.FORCE_DECEL)


def historical_creep_only(intent=None):
  """Seed the retired autonomous-creep state without re-enabling its trigger."""
  harness = MotionHarness()
  intent = MotionIntent() if intent is None else intent
  assert not harness.step(intent, v_ego=0.8, a_target=-0.215)
  owner = harness.supervisor
  owner.phase = MotionPhase.APPROACH
  owner.reasons = StopReason.CREEP
  owner.resolved_should_stop = True
  owner.creep_t = owner.creep_total_t = 0.6
  return harness


@pytest.mark.parametrize('a_target', [-0.215, -0.145])
@pytest.mark.parametrize('moving_lead', [False, True])
@pytest.mark.parametrize('allow_creep', [False, True])
def test_weak_rolling_deceleration_cannot_create_a_stop(a_target, moving_lead, allow_creep):
  harness = MotionHarness()
  # B148/B153 lacked departure permission. Even an older producer's permissive
  # allow_creep bit must not turn this absence into independent stop evidence.
  intent = MotionIntent(moving_lead=moving_lead, allow_creep=allow_creep)
  for frame in range(round(2.0 * CREEP_MAX_T / DT_CTRL)):
    fresh = frame % 5 == 0
    assert not harness.step(intent, v_ego=0.8, a_target=a_target, new_plan=fresh, new_model=fresh)
    assert harness.supervisor.phase == MotionPhase.TRACK
    assert harness.supervisor.reasons == StopReason.NONE
    assert not harness.supervisor.degraded
  assert harness.supervisor.creep_total_t == 0.0


@pytest.mark.parametrize('reason', CURRENT_STOP_REASONS)
@pytest.mark.parametrize('allow_creep', [False, True])
def test_current_stop_evidence_remains_authoritative_and_can_enable_creep(reason, allow_creep):
  harness = MotionHarness()
  intent = MotionIntent(reasons=reason, allow_creep=allow_creep)
  for frame in range(60):
    fresh = frame % 5 == 0
    assert harness.step(intent, v_ego=0.8, a_target=-0.215, new_plan=fresh, new_model=fresh)
    assert harness.supervisor.reasons & reason
    assert not harness.supervisor.degraded
  assert bool(harness.supervisor.reasons & StopReason.CREEP) == allow_creep


@pytest.mark.parametrize('allow_creep', [False, True])
@pytest.mark.parametrize('speed', [STOP_SPEED, 0.8])
def test_fresh_withdrawal_of_creep_only_stop_needs_no_departure_permission(allow_creep, speed):
  intent = MotionIntent(allow_creep=allow_creep)
  harness = historical_creep_only(intent)
  assert not harness.step(intent, v_ego=speed, a_target=-0.215)
  assert harness.supervisor.phase == MotionPhase.RELEASE
  assert harness.supervisor.reasons == StopReason.NONE
  assert harness.supervisor.creep_total_t == 0.0
  assert not harness.supervisor.degraded


@pytest.mark.parametrize('new_plan,new_model', [(False, False), (True, False)])
def test_cached_input_cannot_withdraw_the_historical_creep_stop(new_plan, new_model):
  intent = MotionIntent(allow_creep=False)
  harness = historical_creep_only(intent)
  assert harness.step(intent, v_ego=0.8, a_target=-0.215, new_plan=new_plan, new_model=new_model)
  assert harness.supervisor.phase == MotionPhase.APPROACH
  assert harness.supervisor.reasons == StopReason.CREEP
  assert not harness.supervisor.degraded


@pytest.mark.parametrize('speed,standstill', [(STOP_SPEED - 1e-6, False), (0.0, False), (0.8, True)])
def test_stop_speed_and_standstill_keep_the_existing_brake_release_contract(speed, standstill):
  harness = historical_creep_only()
  assert harness.step(v_ego=speed, standstill=standstill, a_target=-0.215)
  assert harness.supervisor.reasons & StopReason.CREEP
  assert harness.supervisor.reasons & StopReason.HOLD
  assert not harness.supervisor.degraded


@pytest.mark.parametrize('speed,standstill', [(0.0, True), (0.8, False)])
def test_an_existing_hold_cannot_become_rolling_creep_withdrawal(speed, standstill):
  harness = historical_creep_only()
  harness.supervisor.phase = MotionPhase.HOLD
  harness.supervisor.reasons |= StopReason.HOLD
  assert harness.step(v_ego=speed, standstill=standstill, a_target=-0.215)
  assert harness.supervisor.reasons & StopReason.HOLD
  assert harness.supervisor.resolved_should_stop


@pytest.mark.parametrize('reason', CURRENT_STOP_REASONS)
def test_current_stop_takes_precedence_over_creep_only_withdrawal(reason):
  harness = historical_creep_only()
  assert harness.step(MotionIntent(reasons=reason), v_ego=0.8, a_target=-0.215)
  assert harness.supervisor.reasons & reason
  assert harness.supervisor.phase == MotionPhase.APPROACH
  assert not harness.supervisor.degraded


def test_withdrawn_current_close_lead_does_not_leave_a_creep_obligation():
  harness = MotionHarness()
  for _ in range(60):
    assert harness.step(MotionIntent(reasons=StopReason.CLOSE_LEAD), v_ego=0.8, a_target=-0.215)
  assert harness.supervisor.reasons == StopReason.CLOSE_LEAD | StopReason.CREEP
  assert not harness.step(MotionIntent(allow_creep=False), v_ego=0.8, a_target=-0.215)
  assert harness.supervisor.reasons == StopReason.NONE
  assert harness.supervisor.creep_total_t == 0.0


@pytest.mark.parametrize('reason', [StopReason.MPC, StopReason.E2E, StopReason.CRUISE, StopReason.FORCE_DECEL])
def test_creep_withdrawal_cannot_discard_a_separate_owned_stop(reason):
  harness = historical_creep_only()
  harness.supervisor.reasons |= reason
  assert harness.step(v_ego=0.8, a_target=-0.215)
  assert harness.supervisor.reasons & reason
  assert harness.supervisor.phase == MotionPhase.APPROACH


@pytest.mark.parametrize('failure', [
  {'plan_valid': False}, {'solver_status': 4}, {'model_time_ns': 1}, {'advance_ns': 101_000_000},
])
def test_input_faults_keep_the_creep_stop_and_degraded_state(failure):
  harness = historical_creep_only()
  assert harness.step(v_ego=0.8, a_target=-0.215, **failure)
  assert harness.supervisor.reasons & StopReason.CREEP
  assert harness.supervisor.reasons & StopReason.INVALID
  assert harness.supervisor.degraded
