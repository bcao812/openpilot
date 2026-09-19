import random
from unittest.mock import patch

import numpy as np
import pytest

from opendbc.car import structs
from opendbc.car.honda.values import CAR
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState, ROLLING_STOP_ACCEL, STOPPING_DECEL_RATE
from openpilot.selfdrive.controls.lib.motion_supervisor import (DEGRADED_ACCEL, MOTION_INTENT_VERSION, PLAN_LEASE_NS,
                                                               MotionIntent, MotionPhase, StopReason)
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import (
  ACCEL_LIMITS, controller, ControlHarness, decoded_commands, warm_tracking_harness,
)


@pytest.mark.parametrize('entry', ['inactive', 'gas_override', 'brake_override'])
@pytest.mark.parametrize('measured_accel', [-1.0, 0.0, 0.5])
def test_off_to_track_requires_actual_pedal_handover_and_observes_measured_motion(entry, measured_accel):
  """Controller authority/input contract, not a body-jerk or button-event simulation."""
  harness = ControlHarness()
  harness.state.vEgo = 20.0
  harness.state.aEgo = measured_accel
  harness.state.gasPressed = entry == 'gas_override'
  harness.state.brakePressed = entry == 'brake_override'
  assert harness.step(0.0, active=entry != 'inactive') == 0.0
  assert harness.control.motion_phase == MotionPhase.OFF
  assert harness.control.long_control_state == LongCtrlState.off
  assert harness.state.vEgo == 20.0 and harness.state.aEgo == measured_accel

  harness.state.gasPressed = harness.state.brakePressed = False
  supervisor = harness.control.motion_supervisor
  with patch.object(supervisor, 'update', wraps=supervisor.update) as observed:
    output = harness.step(0.0, MotionIntent(allow_creep=False))
  assert observed.call_args.args[:4] == (True, 20.0, measured_accel, False)
  assert harness.control.motion_phase == MotionPhase.TRACK
  assert not harness.control.motion_degraded
  assert np.isfinite(output) and ACCEL_LIMITS[0] <= output <= ACCEL_LIMITS[1]
  # OFF's zero request and aEgo are different quantities. Neither is an
  # output golden value: handover comfort requires the separate vehicle model.


def test_supported_vehicle_selects_unified_without_a_constructor_opt_in():
  reference = controller()
  implicit = LongControl(reference.CP, reference.CP_SP)
  assert implicit.unified
  assert implicit.motion_supervisor is not None
  state = structs.CarState()
  implicit.update(True, state, 1.0, False, ACCEL_LIMITS)
  assert implicit.motion_degraded
  assert implicit.resolved_should_stop


@pytest.mark.parametrize('platform,openpilot_long,interceptor', [
  (CAR.HONDA_CIVIC, True, False),
  (CAR.HONDA_CIVIC_BOSCH, True, False),
  (CAR.HONDA_CRV_HYBRID, True, False),
  (CAR.HONDA_ACCORD, False, False),
  (CAR.HONDA_ACCORD, True, True),
])
def test_constructor_cannot_expand_platform_or_actuation_authority(platform, openpilot_long, interceptor):
  candidate = controller(platform=platform, openpilot_long=openpilot_long, interceptor=interceptor)
  reference = controller(platform=platform, openpilot_long=openpilot_long, interceptor=interceptor)
  state = structs.CarState()
  assert not candidate.unified
  for target in (1.0, -2.0, -0.15, 1.5):
    assert candidate.update(True, state, target, False, ACCEL_LIMITS) == reference.update(True, state, target, False, ACCEL_LIMITS)


def test_legacy_path_ignores_new_protocol_and_remains_frame_equivalent():
  reference, candidate = controller(platform=CAR.HONDA_CIVIC_BOSCH), controller(platform=CAR.HONDA_CIVIC_BOSCH)
  source = random.Random(71029)
  state = structs.CarState()
  for _ in range(1000):
    state.vEgo = source.choice([0.0, 0.1, 0.5, 1.3, 2.0, 20.0])
    state.aEgo = source.uniform(-2.0, 1.0)
    state.standstill = source.random() < 0.15
    state.brakePressed = source.random() < 0.02
    state.cruiseState.standstill = source.random() < 0.03
    target = source.uniform(-3.5, 2.0)
    active = source.random() > 0.04
    stopping = source.random() < 0.15
    expected = reference.update(active, state, target, stopping, ACCEL_LIMITS)
    actual = candidate.update(active, state, target, stopping, ACCEL_LIMITS, intent=MotionIntent(version=7), now_ns=-1, plan_valid=False)
    assert actual == expected
    assert candidate.long_control_state == reference.long_control_state
    assert candidate.starting == reference.starting
    assert candidate.resolved_should_stop == stopping


def test_unified_path_never_silently_reverts_to_legacy_when_intent_is_missing():
  control = controller()
  state = structs.CarState()
  state.vEgo = 10.0
  output = control.update(True, state, 1.0, False, ACCEL_LIMITS)
  assert output <= DEGRADED_ACCEL
  assert control.motion_degraded
  assert control.motion_reasons & StopReason.INVALID


def test_raw_intent_is_the_only_normal_stop_owner():
  harness = ControlHarness()
  harness.state.vEgo = 0.0
  harness.state.standstill = True
  harness.step(intent=MotionIntent(reasons=StopReason.MPC), should_stop=False)
  assert harness.control.resolved_should_stop
  assert harness.control.motion_phase == MotionPhase.HOLD
  assert harness.control.long_control_state == LongCtrlState.stopping


@pytest.mark.parametrize('target', [0.0, 0.05, 0.15, 0.27, 0.448, 1.0, 2.0])
def test_a_cold_standstill_owner_cannot_emit_gas_from_a_missing_hold_history(target):
  harness = ControlHarness()
  harness.state.vEgo = 0.0
  harness.state.standstill = True
  output = harness.step(target)
  assert output == -2.0
  assert harness.control.motion_phase == MotionPhase.HOLD
  assert harness.control.long_control_state == LongCtrlState.stopping
  assert harness.control.resolved_should_stop
  assert not harness.control.motion_degraded
  commands = decoded_commands([output])
  assert commands
  assert all(command['GAS_COMMAND'] <= 0 for command in commands)


@pytest.mark.parametrize('speed,standstill', [(0.0, True), (0.025, False), (0.3, False), (1.3, False)])
@pytest.mark.parametrize('target', [0.27, 0.448, 2.0])
def test_positive_scalar_without_movement_permission_preserves_stop_and_no_gas(speed, standstill, target):
  harness = ControlHarness(controller(initial=-0.4))
  harness.state.vEgo = speed
  harness.state.aEgo = -0.4
  harness.state.standstill = standstill
  harness.step(-0.4, MotionIntent(reasons=StopReason.MPC))
  phase = MotionPhase.HOLD if standstill else MotionPhase.APPROACH
  outputs = []
  for _ in range(100):
    outputs.append(harness.step(target))
    assert harness.control.motion_phase == phase
    assert harness.control.resolved_should_stop
    assert harness.control.long_control_state == LongCtrlState.stopping
    assert harness.control.motion_reasons & StopReason.MPC
    assert not harness.control.motion_degraded
  commands = decoded_commands(outputs[::2])
  assert commands
  assert all(command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0 for command in commands)


@pytest.mark.parametrize('speed', [0.0, 0.5, 2.0, 15.0, 30.0])
@pytest.mark.parametrize('initial,target', [(-3.0, -0.5), (-1.0, 1.5), (-0.21, 0.2), (0.0, 2.0)])
def test_warm_tracking_accel_recovery_keeps_the_one_sided_eight_metre_limit(speed, initial, target):
  harness = warm_tracking_harness(speed, initial)
  outputs = [initial]
  for _ in range(100):
    outputs.append(harness.step(target, MotionIntent(allow_creep=False)))
  assert max(np.diff(outputs)) <= 8.0 * DT_CTRL + 1e-9
  assert min(np.diff(outputs)) >= -1e-9
  assert outputs[-1] == target


@pytest.mark.parametrize('speed', [0.0, 0.5, 1.3, 2.0, 15.0])
@pytest.mark.parametrize('target', [-1.0, -2.5, -3.5])
def test_warm_tracking_applies_a_qualified_deeper_request_without_an_extra_delay(speed, target):
  harness = warm_tracking_harness(speed, -0.4)
  # Fresh, valid planner input in an already executing controller. Exact
  # delivery tests this layer's authority; it does not compare with LEGACY.
  assert harness.step(target, MotionIntent(allow_creep=False)) == target


@pytest.mark.parametrize('speed', [0.3, 0.5, 1.3, 1.99])
@pytest.mark.parametrize('target', [-0.7, -1.3, -2.5, -3.5])
def test_low_speed_stopping_never_discards_the_stronger_braking_target(speed, target):
  harness = ControlHarness(controller(initial=-0.4))
  harness.state.vEgo = speed
  assert harness.step(target, MotionIntent(reasons=StopReason.CLOSE_LEAD)) <= target
  assert harness.control.long_control_state == LongCtrlState.stopping


@pytest.mark.parametrize('speed', [2.0, 10.0, 30.0])
def test_approach_at_speed_tracks_the_braking_trajectory_instead_of_the_open_loop_ramp(speed):
  harness = ControlHarness(controller(initial=-0.4))
  harness.state.vEgo = speed
  assert harness.step(-3.0, MotionIntent(reasons=StopReason.E2E)) == -3.0
  assert harness.control.motion_phase == MotionPhase.APPROACH
  assert harness.control.long_control_state == LongCtrlState.pid


def test_rolling_and_stationary_stop_pressure_bounds_are_not_relaxed():
  harness = ControlHarness()
  for _ in range(200):
    harness.step(-0.1, MotionIntent(reasons=StopReason.MPC))
  assert ROLLING_STOP_ACCEL - STOPPING_DECEL_RATE * DT_CTRL <= harness.control.last_output_accel <= ROLLING_STOP_ACCEL
  harness.hold()


@pytest.mark.parametrize('speed,standstill', [(0.0, True), (0.0, False), (0.1, False), (0.29, False), (0.5, True), (1.3, False), (2.0, False)])
def test_stopping_and_hold_do_not_delay_a_stronger_valid_braking_request(speed, standstill):
  harness = ControlHarness(controller(initial=-0.4))
  harness.state.vEgo = speed
  harness.state.standstill = standstill
  assert harness.step(-3.0, MotionIntent(reasons=StopReason.MPC)) == -3.0


@pytest.mark.parametrize('speed', [0.05, 0.2, 0.299])
def test_sub_threshold_approach_preserves_the_previously_discarded_strong_braking_request(speed):
  harness = ControlHarness(controller(initial=-0.4))
  harness.state.vEgo = speed
  harness.state.standstill = False
  assert harness.step(-3.0, MotionIntent(reasons=StopReason.MPC)) == -3.0
  assert harness.control.motion_phase == MotionPhase.APPROACH
  assert harness.control.long_control_state == LongCtrlState.stopping


def test_standstill_is_not_a_velocity_tracking_error():
  harness = ControlHarness()
  harness.hold()
  for _ in range(100):
    assert -2.0 - STOPPING_DECEL_RATE * DT_CTRL <= harness.step(-0.1, MotionIntent(reasons=StopReason.MPC)) <= -2.0
    assert harness.state.aEgo == 0.0
    assert harness.control.pid.i == 0.0


def test_release_phase_follows_the_actual_bounded_command():
  harness = ControlHarness()
  harness.hold()
  move_intent = MotionIntent(release_allowed=True)
  bottom = harness.control.last_output_accel
  outputs = [bottom]
  for _ in range(100):
    outputs.append(harness.step(0.5, move_intent))
    if outputs[-1] < 0.5:
      assert harness.control.motion_phase == MotionPhase.RELEASE
      assert harness.control.starting
    else:
      assert harness.control.motion_phase == MotionPhase.TRACK
      assert not harness.control.starting
    assert harness.control.long_control_state == LongCtrlState.pid
  assert max(np.diff(outputs)) <= 8.0 * DT_CTRL + 1e-9
  assert min(np.diff(outputs)) >= -1e-9
  assert outputs[-1] == 0.5


def test_resume_is_not_circularly_blocked_by_oem_standstill():
  harness = ControlHarness()
  harness.hold()
  move_intent = MotionIntent(release_allowed=True)
  harness.state.cruiseState.standstill = True
  for _ in range(5):
    assert harness.step(0.5, move_intent) <= -2.0
    assert not harness.control.resolved_should_stop
    assert harness.control.motion_phase == MotionPhase.RELEASE
    assert harness.control.long_control_state == LongCtrlState.stopping
  harness.state.cruiseState.standstill = False
  previous = harness.control.last_output_accel
  assert harness.step(0.5, move_intent) == pytest.approx(previous + 8.0 * DT_CTRL)
  assert harness.control.long_control_state == LongCtrlState.pid


def test_real_stop_evidence_can_interrupt_a_release():
  harness = ControlHarness()
  harness.hold()
  harness.step(0.5, MotionIntent(release_allowed=True))
  assert harness.control.motion_phase == MotionPhase.RELEASE
  harness.state.vEgo = 0.6
  harness.state.standstill = False
  assert harness.step(-3.0, MotionIntent(reasons=StopReason.E2E)) == -3.0
  assert harness.control.motion_phase == MotionPhase.APPROACH
  assert harness.control.long_control_state == LongCtrlState.stopping


def test_movement_permission_confirmation_cannot_delay_stronger_braking():
  harness = ControlHarness()
  harness.hold()
  move_intent = MotionIntent(release_allowed=True)
  for frame in range(20):
    harness.step(0.15, move_intent, new_input=frame % 5 == 0)
    assert harness.control.motion_phase == MotionPhase.HOLD
  assert harness.step(-3.0, MotionIntent(reasons=StopReason.E2E, release_allowed=True)) == -3.0
  assert harness.control.motion_phase == MotionPhase.HOLD
  assert harness.control.motion_supervisor.release_lead_t == 0.0
  command = decoded_commands([harness.control.last_output_accel])[0]
  assert command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0


def test_early_departure_confirmation_preserves_held_braking_and_the_release_rate():
  harness = ControlHarness()
  harness.state.vEgo = 0.0
  harness.state.standstill = True
  for _ in range(150):
    harness.step(-0.1, harness.normal_intent(reasons=StopReason.MPC))
  assert harness.control.motion_phase == MotionPhase.HOLD
  outputs = []
  intent = None
  for frame in range(80):
    new_input = frame % 5 == 0
    if new_input:
      intent = harness.normal_intent(release_allowed=True, allow_creep=False)
    outputs.append(harness.step(-0.1, intent, new_input=new_input))
    assert harness.control.resolved_should_stop
    assert harness.control.long_control_state == LongCtrlState.stopping
  assert all(command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0 for command in decoded_commands(outputs))

  previous = harness.control.last_output_accel
  for frame in range(40):
    new_input = frame % 5 == 0
    if new_input:
      intent = harness.normal_intent(release_allowed=True, allow_creep=False)
    output = harness.step(0.05, intent, new_input=new_input)
    assert output - previous <= 8.0 * DT_CTRL + 1e-9
    assert not harness.control.resolved_should_stop
    previous = output
  command = decoded_commands([output])[0]
  assert not command['BRAKE_REQUEST'] and command['GAS_COMMAND'] > 0


def test_stronger_braking_during_release_does_not_require_movement_permission():
  harness = ControlHarness()
  harness.hold()
  harness.step(0.5, MotionIntent(release_allowed=True))
  assert harness.control.motion_phase == MotionPhase.RELEASE
  assert harness.step(-3.0) == -3.0
  assert not harness.control.motion_degraded
  command = decoded_commands([harness.control.last_output_accel])[0]
  assert command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0


@pytest.mark.parametrize('bad_intent', [
  MotionIntent(version=0), MotionIntent(version=1), MotionIntent(version=2),
  MotionIntent(version=MOTION_INTENT_VERSION + 1), MotionIntent(reasons=1 << 20),
  MotionIntent(reasons=StopReason.HOLD), MotionIntent(valid=False),
])
def test_old_and_bad_protocol_preserves_an_existing_hold(bad_intent):
  harness = ControlHarness()
  harness.hold()
  previous = harness.control.last_output_accel
  assert harness.step(1.0, bad_intent) <= previous
  assert harness.control.motion_phase == MotionPhase.HOLD
  assert harness.control.motion_degraded
  assert harness.control.resolved_should_stop


@pytest.mark.parametrize('condition', ['disable', 'gas', 'brake'])
def test_driver_authority_wins_even_over_malformed_inputs(condition):
  harness = ControlHarness()
  harness.hold()
  if condition != 'disable':
    setattr(harness.state, condition + 'Pressed', True)
  output = harness.step(float('nan'), MotionIntent(version=8), active=condition != 'disable',
                        accel_limits=None, plan_time_ns=-1, model_time_ns=-1, now_ns=-1)
  assert output == 0.0
  assert harness.control.long_control_state == LongCtrlState.off
  assert harness.control.motion_phase == MotionPhase.OFF
  assert not harness.control.motion_reasons
  assert not harness.control.resolved_should_stop
  assert not harness.control.starting


@pytest.mark.parametrize('condition', ['disable', 'gas', 'brake'])
def test_driver_authority_immediately_revokes_an_authorized_release(condition):
  harness = ControlHarness()
  harness.hold()
  harness.step(0.5, MotionIntent(release_allowed=True))
  assert harness.control.motion_phase == MotionPhase.RELEASE
  if condition != 'disable':
    setattr(harness.state, condition + 'Pressed', True)
  assert harness.step(float('nan'), MotionIntent(version=8), active=condition != 'disable', accel_limits=None) == 0.0
  assert harness.control.motion_phase == MotionPhase.OFF
  assert harness.control.long_control_state == LongCtrlState.off
  assert not harness.control.resolved_should_stop
  assert not harness.control.starting


@pytest.mark.parametrize('field', ['vEgo', 'aEgo', 'target'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_non_finite_inputs_never_reach_the_pid_or_leave_positive_drive(field, value):
  harness = ControlHarness(controller(initial=1.0))
  target = 1.0
  harness.state.vEgo = 10.0
  if field == 'target':
    target = value
  else:
    setattr(harness.state, field, value)
  output = harness.step(target)
  assert np.isfinite(output)
  assert output <= DEGRADED_ACCEL
  assert harness.control.motion_degraded
  assert np.isfinite(harness.control.pid.i)


@pytest.mark.parametrize('accel_limits', [(float('nan'), 2.0), (-3.5, float('inf')), (2.0, -3.5), (0.0, 2.0), (-10.0, 10.0)])
def test_invalid_limits_cannot_turn_degraded_zero_into_positive_gas(accel_limits):
  harness = ControlHarness(controller(initial=0.5))
  harness.state.vEgo = 10.0
  output = harness.step(1.0, accel_limits=accel_limits)
  assert ACCEL_LIMITS[0] <= output <= DEGRADED_ACCEL
  assert harness.control.motion_degraded


def test_a_stale_positive_plan_expires_and_native_can_has_no_drive():
  harness = ControlHarness(controller(initial=1.0))
  harness.state.vEgo = 10.0
  harness.step(1.0)
  origin = harness.now_ns
  outputs = []
  for _ in range(100):
    output = harness.step(1.0, new_input=False)
    if harness.now_ns - origin > PLAN_LEASE_NS:
      outputs.append(output)
      assert output <= DEGRADED_ACCEL
      assert harness.control.motion_degraded
  commands = decoded_commands(outputs[::2])
  assert commands
  assert all(command['GAS_COMMAND'] <= 0 for command in commands)


def test_an_invalid_command_does_not_weaken_braking_already_underway():
  harness = ControlHarness(controller(initial=-3.0))
  harness.state.vEgo = 10.0
  assert harness.step(float('nan')) == -3.0
  assert harness.control.motion_degraded


@pytest.mark.parametrize('fault', ['invalid', 'old', 'nan', 'future', 'stale', 'unhealthy'])
def test_fresh_input_recovers_from_degraded_without_a_health_dependency_lock(fault):
  harness = ControlHarness()
  harness.state.vEgo = 10.0
  harness.step(1.0)
  intent = MotionIntent(valid=False) if fault == 'invalid' else MotionIntent(version=0 if fault == 'old' else MOTION_INTENT_VERSION)
  kwargs = {}
  if fault == 'future':
    kwargs['plan_time_ns'] = harness.now_ns + 100_000_000
  elif fault == 'stale':
    kwargs['model_time_ns'] = harness.now_ns - 2 * PLAN_LEASE_NS
  elif fault == 'unhealthy':
    kwargs['plan_valid'] = False
  assert harness.step(float('nan') if fault == 'nan' else 1.0, intent, **kwargs) <= DEGRADED_ACCEL
  assert harness.control.motion_degraded
  previous = harness.control.last_output_accel
  output = harness.step(1.0)
  assert not harness.control.motion_degraded
  assert harness.control.motion_phase == MotionPhase.TRACK
  assert previous <= output <= previous + 8.0 * DT_CTRL + 1e-9


def test_fresh_hold_recovery_clears_fault_but_still_requires_release_evidence():
  harness = ControlHarness()
  harness.hold()
  assert harness.step(1.0, MotionIntent(version=0)) <= -2.0
  assert harness.control.motion_degraded
  assert harness.step(0.15) <= -2.0
  assert not harness.control.motion_degraded
  assert harness.control.resolved_should_stop
  previous = harness.control.last_output_accel
  assert harness.step(0.5, MotionIntent(release_allowed=True)) == pytest.approx(previous + 8.0 * DT_CTRL)
  assert harness.control.motion_phase == MotionPhase.RELEASE


def test_native_can_release_is_mutually_exclusive_and_continuous():
  harness = ControlHarness()
  harness.hold()
  move_intent = MotionIntent(release_allowed=True)
  outputs = [harness.control.last_output_accel]
  for _ in range(100):
    outputs.append(harness.step(1.0, move_intent))
  commands = decoded_commands(outputs[::2])
  assert max(np.diff([command['ACCEL_COMMAND'] for command in commands])) <= 0.17
  assert all(not command['BRAKE_REQUEST'] or command['GAS_COMMAND'] <= 0 for command in commands)
  assert sum(before['BRAKE_REQUEST'] != after['BRAKE_REQUEST'] for before, after in zip(commands, commands[1:], strict=False)) == 1


@pytest.mark.parametrize('fault', ['unhealthy', 'nonfinite', 'rewritten', 'future'])
def test_fault_interrupted_release_cannot_reuse_old_plan_to_release_brake(fault):
  harness = ControlHarness()
  harness.hold()
  move_intent = MotionIntent(release_allowed=True)
  harness.step(0.3, move_intent)
  assert harness.control.motion_phase == MotionPhase.RELEASE
  kwargs = {'plan_valid': False} if fault == 'unhealthy' else {}
  if fault == 'nonfinite':
    harness.state.aEgo = float('nan')
  if fault == 'future':
    kwargs['plan_time_ns'] = harness.now_ns + 1_000_000_000
  harness.step(0.5 if fault == 'rewritten' else 0.3, move_intent, new_input=False, **kwargs)
  assert harness.control.motion_degraded
  harness.state.aEgo = 0.0
  outputs = [harness.step(0.3, move_intent, new_input=False) for _ in range(5)]
  assert harness.control.motion_degraded
  assert harness.control.resolved_should_stop
  assert harness.control.motion_phase == MotionPhase.HOLD
  commands = decoded_commands(outputs)
  assert commands
  assert all(command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0 for command in commands)
  previous = harness.control.last_output_accel
  assert harness.step(0.3, move_intent) <= previous + 8.0 * DT_CTRL + 1e-9
  assert not harness.control.motion_degraded
  assert harness.control.motion_phase == MotionPhase.RELEASE


def test_rearming_wait_does_not_delay_new_valid_stronger_braking():
  harness = ControlHarness()
  harness.hold()
  move_intent = MotionIntent(release_allowed=True)
  harness.step(0.3, move_intent)
  assert harness.control.motion_phase == MotionPhase.RELEASE
  harness.step(0.3, move_intent, new_input=False, plan_valid=False)
  newer_plan = harness.now_ns + round(DT_CTRL * 1e9)
  output = harness.step(-3.0, MotionIntent(reasons=StopReason.E2E), new_input=False, plan_time_ns=newer_plan)
  assert output == -3.0
  assert harness.control.motion_degraded
  assert harness.control.resolved_should_stop
  command = decoded_commands([output])[0]
  assert command['ACCEL_COMMAND'] == -3.0
  assert command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0
  previous = harness.control.last_output_accel
  harness.step(0.3, move_intent)
  assert not harness.control.motion_degraded
  assert harness.control.last_output_accel <= previous + 8.0 * DT_CTRL + 1e-9


@pytest.mark.parametrize('speed', [-0.031628, 0.2, 0.5, 1.3])
@pytest.mark.parametrize('previous', [-0.1, 0.0, 0.3])
def test_entering_a_stop_cannot_continue_positive_gas_during_the_ramp(speed, previous):
  harness = ControlHarness(controller(initial=previous))
  harness.state.vEgo = speed
  harness.state.standstill = False
  output = harness.step(0.05, MotionIntent(reasons=StopReason.MPC))
  assert harness.control.resolved_should_stop
  assert harness.control.long_control_state == LongCtrlState.stopping
  command = decoded_commands([output])[0]
  assert command['GAS_COMMAND'] <= 0
  assert output <= DEGRADED_ACCEL
