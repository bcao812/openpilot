"""Consumer enforcement for a HOLD retained by the main takeover state machine."""

import pytest

from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState, STARTING_ACCEL_RATE
from openpilot.selfdrive.controls.lib.motion_supervisor import (
  PLAN_LEASE_NS, MotionIntent, MotionPhase, MotionSupervisor, StopReason,
)
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import (
  ACCEL_LIMITS, ControlHarness, decoded_commands,
)


CONTROL_INTERVAL_NS = round(DT_CTRL * 1e9)


def step(harness, target=0.5, intent=None, *, force_decel=True, active=True, new_input=True,
         accel_limits=ACCEL_LIMITS, **kwargs):
  """Use the real controller, including its forceDecel argument."""
  harness.now_ns += CONTROL_INTERVAL_NS
  if new_input:
    harness.plan_time_ns = harness.model_time_ns = harness.now_ns
  return harness.control.update(
    active, harness.state, target, False, accel_limits,
    intent=MotionIntent(release_allowed=True) if intent is None else intent,
    plan_time_ns=harness.plan_time_ns, model_time_ns=harness.model_time_ns,
    now_ns=harness.now_ns, force_decel=force_decel, **kwargs,
  )


def assert_held(harness, previous):
  control = harness.control
  assert control.motion_phase == MotionPhase.HOLD
  assert control.motion_reasons & StopReason.HOLD
  assert control.motion_reasons & StopReason.FORCE_DECEL
  assert control.resolved_should_stop
  assert control.long_control_state == LongCtrlState.stopping
  assert not control.starting
  assert control.last_output_accel <= previous


@pytest.mark.parametrize('target', [0.05, 0.5, 2.0])
@pytest.mark.parametrize('oem_standstill', [False, True])
def test_fresh_positive_recovery_cannot_release_takeover_hold(target, oem_standstill):
  harness = ControlHarness()
  harness.hold()
  harness.state.cruiseState.standstill = oem_standstill
  outputs = []
  for _ in range(70):
    previous = harness.control.last_output_accel
    outputs.append(step(harness, target))
    assert_held(harness, previous)
    assert not harness.control.motion_degraded
    assert not harness.control.motion_waiting_for_fresh_input
  commands = decoded_commands(outputs[::2])
  assert commands
  assert all(command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0 for command in commands)


def test_clearing_force_decel_keeps_the_normal_qualified_release_path():
  harness = ControlHarness()
  harness.hold()
  previous = step(harness)
  assert_held(harness, previous)
  output = step(harness, force_decel=False)
  assert previous < output <= previous + STARTING_ACCEL_RATE * DT_CTRL + 1e-9
  assert harness.control.motion_phase == MotionPhase.RELEASE
  assert not harness.control.resolved_should_stop
  assert not harness.control.motion_reasons
  assert harness.control.long_control_state == LongCtrlState.pid
  for _ in range(50):
    output = step(harness, force_decel=False)
  assert output == pytest.approx(0.5)
  assert harness.control.motion_phase == MotionPhase.TRACK


@pytest.mark.parametrize('standstill_recovers', [False, True])
def test_takeover_hold_survives_a_standstill_observation_gap(standstill_recovers):
  harness = ControlHarness()
  harness.hold()
  previous = step(harness)
  harness.state.standstill = False
  harness.state.vEgo = 0.05
  first = step(harness)
  assert first <= previous
  assert harness.control.motion_phase == MotionPhase.APPROACH
  assert harness.control.motion_reasons & StopReason.HOLD

  harness.state.standstill = standstill_recovers
  harness.state.vEgo = 0.0 if standstill_recovers else 0.05
  second = step(harness)
  assert second <= first
  assert harness.control.motion_phase == (MotionPhase.HOLD if standstill_recovers else MotionPhase.APPROACH)
  assert harness.control.motion_reasons & StopReason.HOLD
  assert harness.control.motion_reasons & StopReason.FORCE_DECEL
  assert harness.control.resolved_should_stop  # The existing resume gate remains closed.
  assert harness.control.long_control_state == LongCtrlState.stopping
  assert not harness.control.starting
  commands = decoded_commands([first, second])
  assert commands
  assert all(command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0 for command in commands)

  released = step(harness, force_decel=False)
  assert second < released <= second + STARTING_ACCEL_RATE * DT_CTRL + 1e-9
  assert harness.control.motion_phase == MotionPhase.RELEASE
  assert not harness.control.resolved_should_stop


def test_takeover_flag_does_not_capture_an_approach_that_never_held():
  reference, candidate = ControlHarness(), ControlHarness()
  for harness in (reference, candidate):
    step(harness, -0.5, MotionIntent(reasons=StopReason.MPC), force_decel=False)
    assert harness.control.motion_phase == MotionPhase.APPROACH
    assert not harness.control.motion_reasons & StopReason.HOLD
  expected = step(reference, force_decel=False)
  assert step(candidate) == pytest.approx(expected)
  assert candidate.control.motion_phase == reference.control.motion_phase == MotionPhase.RELEASE
  assert not candidate.control.motion_reasons
  assert not candidate.control.resolved_should_stop


@pytest.mark.parametrize('fault', ['intent', 'plan'])
def test_degraded_hold_and_fresh_recovery_keep_the_takeover_brake(fault):
  harness = ControlHarness()
  harness.hold()
  previous = harness.control.last_output_accel
  for _ in range(10):
    step(harness, intent=MotionIntent(valid=False) if fault == 'intent' else None,
         plan_valid=fault != 'plan')
    assert_held(harness, previous)
    assert harness.control.motion_degraded
  step(harness)
  assert_held(harness, previous)
  assert not harness.control.motion_degraded


def test_takeover_hold_does_not_extend_the_cached_plan_lease():
  harness = ControlHarness()
  harness.hold()
  previous = step(harness)
  accepted_plan_ns = harness.plan_time_ns
  for _ in range(PLAN_LEASE_NS // CONTROL_INTERVAL_NS + 1):
    step(harness, new_input=False)
    assert_held(harness, previous)
  assert harness.plan_time_ns == accepted_plan_ns
  assert harness.control.motion_degraded
  assert harness.control.motion_reasons & StopReason.INVALID
  step(harness)
  assert_held(harness, previous)
  assert not harness.control.motion_degraded


@pytest.mark.parametrize('braking_only', [False, True])
def test_takeover_hold_accepts_stronger_fresh_braking(braking_only):
  harness = ControlHarness()
  harness.hold()
  target = -3.0
  intent = (MotionIntent(valid=False, allow_creep=False, braking_accel=target)
            if braking_only else MotionIntent())
  output = step(harness, target, intent, solver_status=4 if braking_only else 0,
                plan_valid=not braking_only, braking_plan_valid=braking_only)
  assert output == pytest.approx(target)
  assert_held(harness, target)
  assert harness.control.motion_degraded is braking_only


@pytest.mark.parametrize('override', ['inactive', 'gas', 'brake'])
def test_driver_or_disabled_authority_immediately_ends_takeover_hold(override):
  harness = ControlHarness()
  harness.hold()
  step(harness)
  if override != 'inactive':
    setattr(harness.state, override + 'Pressed', True)
  # CANCEL, MAIN and global disable reach this consumer through active=False.
  output = step(harness, float('nan'), MotionIntent(version=0),
                active=override != 'inactive', accel_limits=None)
  assert output == 0.0
  assert harness.control.motion_phase == MotionPhase.OFF
  assert harness.control.long_control_state == LongCtrlState.off
  assert not harness.control.motion_reasons
  assert not harness.control.resolved_should_stop
  assert not harness.control.starting


@pytest.mark.parametrize('target', [-0.5, 0.5])
def test_force_decel_does_not_add_a_takeover_stop_to_moving_track(target):
  reference, candidate = ControlHarness(), ControlHarness()
  reference.state.vEgo = candidate.state.vEgo = 10.0
  for _ in range(30):
    expected = step(reference, target, force_decel=False)
    assert step(candidate, target) == pytest.approx(expected)
    assert candidate.control.motion_phase == MotionPhase.TRACK
    assert candidate.control.long_control_state == LongCtrlState.pid
    assert not candidate.control.motion_reasons
    assert not candidate.control.resolved_should_stop


@pytest.mark.parametrize('active', [False, True])
def test_takeover_flag_does_not_change_an_off_owners_existing_entry_rules(active):
  reference, candidate = MotionSupervisor(), MotionSupervisor()
  kwargs = {"plan_time_ns": 2_000_000_000, "model_time_ns": 2_000_000_000, "now_ns": 2_000_000_000}
  intent = MotionIntent(release_allowed=True)
  expected = reference.update(active, 0.0, 0.0, True, 0.5, intent, **kwargs)
  assert candidate.update(active, 0.0, 0.0, True, 0.5, intent, hold_for_takeover=True, **kwargs) == expected
  assert candidate.phase == reference.phase
  assert candidate.reasons == reference.reasons
  assert not candidate.reasons & StopReason.FORCE_DECEL
