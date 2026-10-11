"""SET after a fault: real CarD, planner, Controls and selfdrived engagement policy.

The existing startup fixture bounds transport, sensors and lateral control.
These cases exercise re-entry, bounded stopped handoff and fault clocks, not vehicle response.
"""
from contextlib import nullcontext
from unittest.mock import patch

import pytest

from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.longitudinal_continuation import INACTIVE_NATIVE_RECOVERY_REASON, UNSTABLE_DISABLE_NS
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_unified_startup import CONTROL_STEP_NS, PLANNER_STEPS, StartupHarness, State
from openpilot.selfdrive.selfdrived.engagement_policy import HOLD_EXTRA_NS


def fail_then_disable():
  harness = StartupHarness()
  harness.engage()
  with patch.object(harness.planner.mpc, 'run', side_effect=lambda: harness.planner.mpc.fail(-1)):
    rows = [harness.step() for _ in range(PLANNER_STEPS + 2)]
  assert any(row['longitudinal_unavailable'] for row in rows)
  assert harness.selfdrive._longitudinal_failure_budget.episode_start_ns
  harness.source_state.brakePressed = True
  disabled = harness.step()
  assert not disabled['enabled'] and not disabled['long_active']
  assert disabled['accel'] == 0.0 and not disabled['resume']
  harness.source_state.brakePressed = False
  return harness


def assert_disabled(row):
  assert not row['enabled'] and not row['long_active'] and not row['resume']
  assert row['phase'] == 'off' and row['accel'] == 0.0


def assert_budgets_retired(harness):
  assert harness.controls.LoC.continuation_budget.episode_start_ns == 0
  assert harness.selfdrive._longitudinal_failure_budget.episode_start_ns == 0
  assert harness.planner.reference_planner.continuation.budget.episode_start_ns == 0


@pytest.mark.parametrize('phase', range(PLANNER_STEPS))
@pytest.mark.parametrize('main_clears_speed', [False, True], ids=['retained_set_speed', 'warm_standby'])
def test_set_recovers_after_fault_and_manual_disable_at_every_planner_phase(phase, main_clears_speed):
  harness = fail_then_disable()
  old_episode = harness.selfdrive._longitudinal_failure_budget.episode_start_ns
  # Let the original absolute deadline expire while the driver has control.
  # Subsequent health credit must come from advancing inputs, not this gap.
  harness.now_ns += 3_100_000_000
  if main_clears_speed:
    harness.source_state.cruiseState.available = False
  for _ in range(80):
    assert_disabled(harness.step())
  assert harness.now_ns > old_episode + 3_000_000_000
  assert harness.controls.LoC.continuation_donor is None
  assert harness.planner.reference_planner.continuation.donor is None

  if main_clears_speed:
    standby = harness.latest['longitudinalPlan']
    assert standby.valid
    assert not standby.longitudinalPlan.motionIntent.valid and not standby.longitudinalPlan.motionIntent.ready
    assert standby.longitudinalPlan.solverStatus == harness.planner.mpc.solution_status
    assert harness.selfdrive._longitudinal_failure_budget.episode_start_ns == old_episode
    assert harness.card.v_cruise_helper.v_cruise_kph == V_CRUISE_UNSET
    blocked = harness.first_set()
    assert blocked['no_entry']
    assert_disabled(blocked)
    harness.source_state.cruiseState.available = True
    for _ in range(2 * PLANNER_STEPS):
      assert_disabled(harness.step())
    # Standby alone may unblock initialization; it is never native recovery.
    assert harness.selfdrive._longitudinal_failure_budget.episode_start_ns == old_episode
  else:
    assert_budgets_retired(harness)
    motion = harness.latest['controlsState'].controlsState.motionState
    assert motion.continuationReason == INACTIVE_NATIVE_RECOVERY_REASON
    assert harness.selfdrive._continuation_donor is None

  while (harness.frame + 2) % PLANNER_STEPS != phase:
    assert_disabled(harness.step())
  requested = harness.first_set()
  assert requested['enabled'] and not requested['no_entry']
  assert not requested['degraded'] and not requested['longitudinal_unavailable']
  rows = [harness.step() for _ in range(80)]
  assert all(row['enabled'] and not row['degraded'] and not row['longitudinal_unavailable'] for row in rows)
  assert any(row['long_active'] for row in rows)
  assert rows[-1]['phase'] == 'track'
  assert harness.card.v_cruise_helper.v_cruise_kph != V_CRUISE_UNSET
  assert_budgets_retired(harness)


def test_repeated_successful_publication_cannot_retire_the_disabled_fault_budget():
  harness = fail_then_disable()
  episode = harness.selfdrive._longitudinal_failure_budget.episode_start_ns
  for _ in range(2 * PLANNER_STEPS):
    assert_disabled(harness.step())
  assert harness.latest['longitudinalPlan'].longitudinalPlan.solverStatus == 0
  for _ in range(60):
    assert_disabled(harness.step(run_planner=False))
  assert harness.selfdrive._longitudinal_failure_budget.episode_start_ns == episode
  assert harness.controls.LoC.continuation_budget.episode_start_ns


def test_unresolved_native_failure_still_rejects_set_after_manual_disable():
  harness = fail_then_disable()
  harness.now_ns += 3_100_000_000
  with patch.object(harness.planner.mpc, 'run', side_effect=lambda: harness.planner.mpc.fail(-1)):
    for _ in range(2 * PLANNER_STEPS):
      assert_disabled(harness.step())
    blocked = harness.first_set()
  assert blocked['no_entry'] and blocked['longitudinal_unavailable']
  assert_disabled(blocked)


def test_new_failure_after_disabled_recovery_gets_its_own_bounded_deadline():
  harness = fail_then_disable()
  old_episode = harness.selfdrive._longitudinal_failure_budget.episode_start_ns
  harness.now_ns += 3_100_000_000
  for _ in range(80):
    assert_disabled(harness.step())
  assert_budgets_retired(harness)
  assert harness.first_set()['enabled']
  for _ in range(2 * PLANNER_STEPS):
    assert harness.step()['enabled']
  with patch.object(harness.planner.mpc, 'run', side_effect=lambda: harness.planner.mpc.fail(-1)):
    rows = [harness.step() for _ in range(PLANNER_STEPS + 2)]
  assert any(row['longitudinal_unavailable'] for row in rows)
  assert all(row['enabled'] for row in rows)
  assert harness.selfdrive._longitudinal_failure_budget.episode_start_ns > old_episode
  assert 2_000_000_000 < harness.selfdrive._longitudinal_soft_disable_remaining_ns <= 3_000_000_000


def stopped_handoff():
  harness = StartupHarness()
  harness.hold()
  # Let the real owner/controller reach holding pressure before introducing
  # the native failure. No policy deadline, phase or output is assigned here.
  for _ in range(2_000_000_000 // CONTROL_STEP_NS):
    harness.step()
    if (harness.controls.LoC.last_output_accel <= harness.CP.stopAccel
        and harness.selfdrive.sm['carOutput'].actuatorsOutput.accel <= harness.CP.stopAccel):
      break
  else:
    raise AssertionError(harness.history[-5:])

  with patch.object(harness.planner.mpc, 'run', side_effect=lambda: harness.planner.mpc.fail(-1)):
    for _ in range(UNSTABLE_DISABLE_NS // CONTROL_STEP_NS + 2 * PLANNER_STEPS):
      row = harness.step()
      assert row['enabled'], harness.history[-5:]
      if row['holding_for_takeover']:
        break
    else:
      raise AssertionError(harness.history[-5:])
  handoff = harness.selfdrive.engagement_policy.handoff
  assert handoff.active
  assert handoff.extended_deadline_ns == handoff.original_deadline_ns + HOLD_EXTRA_NS
  assert handoff.original_deadline_ns <= row['mono'] < handoff.extended_deadline_ns
  return harness


def assert_handoff(row, deadline, stop_accel):
  assert row['enabled'] and row['long_active'] and row['holding_for_takeover']
  assert row['selfdrive_state'] == State.softDisabling
  assert row['force_decel'] and row['phase'] == 'hold'
  assert row['accel'] <= stop_accel and not row['resume']
  assert row['hold_deadline_ns'] == deadline
  assert row['hold_remaining_ns'] == deadline - row['mono'] > 0


@pytest.mark.parametrize('fails_again', [False, True], ids=['recovered', 'new_failure'])
def test_handoff_recovery_cannot_resume_or_extend_and_a_later_set_is_explicit(fails_again):
  harness = stopped_handoff()
  deadline = harness.selfdrive.engagement_policy.handoff.extended_deadline_ns
  original_episode = harness.selfdrive._longitudinal_failure_budget.episode_start_ns
  commands_before = len(harness.native_commands)
  harness.model_stop = False
  for _ in range(80):
    assert_handoff(harness.step(), deadline, harness.CP.stopAccel)
  # Native recovery is observed through current publications and the owner;
  # it must not end the handoff or acquire a replacement deadline.
  assert harness.latest['longitudinalPlan'].valid
  assert harness.latest['longitudinalPlan'].longitudinalPlan.solverStatus == 0
  assert not harness.history[-1]['degraded']
  assert not harness.history[-1]['longitudinal_unavailable']
  assert harness.selfdrive._longitudinal_failure_budget.episode_start_ns == 0

  fail_native = (patch.object(harness.planner.mpc, 'run', side_effect=lambda: harness.planner.mpc.fail(-1))
                 if fails_again else nullcontext())
  with fail_native:
    while harness.now_ns + CONTROL_STEP_NS < deadline:
      assert_handoff(harness.step(), deadline, harness.CP.stopAccel)
    if fails_again:
      assert harness.selfdrive._longitudinal_failure_budget.episode_start_ns > original_episode
    expired = harness.step()
    assert expired['mono'] == deadline
    assert_disabled(expired)
    assert not expired['holding_for_takeover']
    if fails_again:
      blocked = harness.first_set()
      assert blocked['no_entry']
      assert_disabled(blocked)

  held_commands = [command for command in harness.native_commands[commands_before:] if command['mono'] < deadline]
  assert held_commands
  assert all(command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0 and not command['resume'] for command in held_commands)
  for _ in range(80):
    assert_disabled(harness.step())
  assert_budgets_retired(harness)
  requested = harness.first_set()
  assert requested['enabled'] and not requested['no_entry']
  assert not requested['holding_for_takeover']
  assert harness.selfdrive.engagement_policy.handoff.extended_deadline_ns == 0


@pytest.mark.parametrize('manual', ['cancel', 'brake'])
def test_manual_handoff_takeover_clears_policy_alert_and_requires_a_new_set(manual):
  harness = stopped_handoff()
  if manual == 'brake':
    harness.source_state.brakePressed = True
  disabled = harness.step(button='cancel' if manual == 'cancel' else None)
  assert_disabled(disabled)
  assert not disabled['holding_for_takeover']
  assert not harness.selfdrive.engagement_policy.handoff.active
  # observe_states is part of the production step: manual takeover clears the
  # "hold ended" warning that would otherwise remain after revoking the hold.
  assert harness.selfdrive.engagement_policy.hold_ended_until_ns == 0
  harness.source_state.brakePressed = False
  for _ in range(80):
    assert_disabled(harness.step())
  assert_budgets_retired(harness)
  requested = harness.first_set()
  assert requested['enabled'] and not requested['no_entry']
  assert not requested['holding_for_takeover']
