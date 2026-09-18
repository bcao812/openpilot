"""Separate a healthy producer wait from faults without weakening actuation.

The two consumer sequences come from the 2026-09-16 round3/rearm-diagnostic
consumer probe. They execute real Controls, SelfdriveD event consumption and
StateMachine with the existing bounded healthy collaborators; they are not a
claim about full daemon scheduling or delivered vehicle effects.
"""

from dataclasses import replace
from unittest.mock import patch

import pytest

from openpilot.cereal import log
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION, MotionIntent, MotionPhase, MotionSupervisor, StopReason
from openpilot.selfdrive.controls.tests.test_supervised_creep_timing import raw_intent, sample_step
from openpilot.selfdrive.controls.tests.test_unified_controlsd import ControlsHarness, selfdrive_event_harness
from openpilot.selfdrive.selfdrived.events import ET
from openpilot.selfdrive.selfdrived.state import StateMachine


State = log.SelfdriveState.OpenpilotState
EventName = log.OnroadEvent.EventName


@pytest.fixture(autouse=True)
def isolated_params():
  with OpenpilotPrefix():
    yield


@pytest.fixture(params=(LongitudinalMode.SUPERVISED, LongitudinalMode.UNIFIED), ids=lambda mode: mode.name)
def mode(request):
  return request.param


def enter_off(mode, *, trigger='enable_between_plans', speed=11.343399047851562, target=-0.590408980846405, release_allowed=False):
  harness = ControlsHarness(unified=mode == LongitudinalMode.UNIFIED, supervised=mode == LongitudinalMode.SUPERVISED)
  harness.car_state.vEgo = speed
  harness.plan.aTarget = target
  harness.plan.motionIntent.ready = True
  harness.plan.motionIntent.releaseAllowed = release_allowed
  harness.plan.motionIntent.creepReset = True
  harness.plan.motionIntent.allowCreep = False
  if trigger == 'enable_between_plans':
    harness.selfdrive_state.enabled = False
    harness.selfdrive_state.active = False
    harness.selfdrive_state.state = State.disabled
  elif trigger == 'gas_release_between_plans':
    harness.car_state.gasPressed = True
  elif trigger == 'gas_release_after_active_plan':
    harness.step()
    harness.car_state.gasPressed = True
  else:
    raise ValueError(trigger)
  command, motion = harness.step(new_plan=trigger != 'gas_release_after_active_plan')
  assert command.actuators.accel == 0.0
  assert motion.phase == 'off'
  harness.selfdrive_state.enabled = True
  harness.selfdrive_state.active = True
  harness.selfdrive_state.state = State.enabled
  harness.car_state.gasPressed = False
  return harness


def event_consumers(harness):
  daemon, car_state = selfdrive_event_harness(harness.control.pm.messages['controlsState'])
  daemon.update_events(car_state)
  machine = StateMachine()
  machine.state = State.enabled
  machine.update(daemon.events)
  return daemon, car_state, machine


@pytest.mark.parametrize('trigger,speed,target', [
  ('enable_between_plans', 11.343399047851562, -0.590408980846405),
  ('gas_release_between_plans', 0.8, -0.22158890962600708),
  ('gas_release_after_active_plan', 10.0, -0.6),
  ('gas_release_after_active_plan', 10.0, 0.5),
])
def test_healthy_rearm_does_not_create_comm_issue_or_force_decel(mode, trigger, speed, target):
  harness = enter_off(mode, trigger=trigger, speed=speed, target=target)
  old_plan = harness.control.sm.logMonoTime['longitudinalPlan']
  command, motion = harness.step(new_plan=False)
  assert harness.control.sm.all_checks(['longitudinalPlan', 'carState'])
  assert motion.planMonoTime == old_plan
  assert command.actuators.accel == pytest.approx(min(target, -0.2))
  daemon, car_state, machine = event_consumers(harness)
  # Assert the actual product outcome before inspecting the added diagnostic:
  # the old implementation fails here on commIssue, not a missing new field.
  assert EventName.commIssue not in daemon.events.names
  assert not daemon.events.contains(ET.SOFT_DISABLE)
  assert machine.state == State.enabled
  assert not motion.degraded
  assert motion.waitingForFreshInput
  assert not command.cruiseControl.resume
  assert not harness.control.pm.messages['controlsState'].controlsState.forceDecel

  harness.selfdrive_state.state = machine.state
  _, motion = harness.step(new_plan=False)
  assert motion.waitingForFreshInput and not motion.degraded
  assert not harness.control.pm.messages['controlsState'].controlsState.forceDecel
  harness.plan.motionIntent.creepReset = False
  harness.plan.motionIntent.allowCreep = True
  command, motion = harness.step()
  assert not motion.degraded and not motion.waitingForFreshInput
  assert not harness.control.pm.messages['controlsState'].controlsState.forceDecel
  assert not command.driverMonitoringEscalation
  daemon.sm.update_msgs(harness.now_ns / 1e9, [harness.control.pm.messages['controlsState']])
  daemon.update_events(car_state)
  machine.update(daemon.events)
  assert EventName.commIssue not in daemon.events.names
  assert machine.state == State.enabled


@pytest.mark.parametrize('target', [1.0, 0.0, -0.1, -0.22158890962600708, -0.590408980846405, -3.0])
@pytest.mark.parametrize('oem_standstill', [False, True])
def test_healthy_wait_preserves_r3_braking_and_never_grants_gas_or_resume(mode, target, oem_standstill):
  from openpilot.selfdrive.controls.tests.test_motion_fault_braking import native_command

  harness = enter_off(mode, speed=0.8, target=target)
  harness.car_state.cruiseState.standstill = oem_standstill
  command, motion = harness.step(new_plan=False)
  assert motion.waitingForFreshInput and not motion.degraded
  assert harness.control.LoC.motion_input_constrained
  assert command.actuators.accel == pytest.approx(min(target, -0.2))
  assert not command.cruiseControl.resume
  _, decoded = native_command(harness)
  assert decoded['GAS_COMMAND'] <= 0
  assert decoded['ACCEL_COMMAND'] <= -0.2
  assert not harness.control.pm.messages['controlsState'].controlsState.forceDecel


def test_healthy_wait_establishes_hold_without_reusing_old_release_permission(mode):
  from openpilot.selfdrive.controls.tests.test_motion_fault_braking import native_command

  harness = enter_off(mode, speed=0.0, target=1.0, release_allowed=True)
  harness.car_state.standstill = True
  harness.car_state.cruiseState.standstill = True
  command, motion = harness.step(new_plan=False)
  assert motion.waitingForFreshInput and not motion.degraded
  assert motion.phase == 'hold' and motion.resolvedStop
  assert motion.reasons & int(StopReason.HOLD)
  assert not motion.reasons & int(StopReason.INVALID)
  assert command.actuators.accel <= harness.control.CP.stopAccel
  assert not command.cruiseControl.resume
  _, decoded = native_command(harness)
  assert decoded['GAS_COMMAND'] <= 0
  assert decoded['BRAKE_REQUEST']
  daemon, _, machine = event_consumers(harness)
  assert EventName.commIssue not in daemon.events.names
  assert machine.state == State.enabled
  command, motion = harness.step()
  assert not motion.waitingForFreshInput and not motion.degraded
  assert motion.phase == 'release'
  assert command.cruiseControl.resume


def test_waiting_requires_both_producer_ids_without_regranting_old_movement(mode):
  harness = enter_off(mode, speed=0.8, target=1.0)
  _, motion = harness.step(new_plan=False)
  old_model = int(motion.modelMonoTime)
  harness.plan.motionIntent.creepReset = False
  harness.plan.motionIntent.allowCreep = True
  harness.plan.motionIntent.releaseAllowed = True
  for _ in range(3):
    command, motion = harness.step(model_time_ns=old_model)
    assert motion.waitingForFreshInput and not motion.degraded
    assert command.actuators.accel <= -0.2
    assert not command.cruiseControl.resume
    assert harness.control.LoC.motion_supervisor.release_lead_t == 0.0
  _, motion = harness.step()
  assert not motion.waitingForFreshInput and not motion.degraded
  assert not harness.control.LoC.motion_input_constrained


def test_cold_standby_then_healthy_off_does_not_create_a_fault_history(mode):
  harness = ControlsHarness(unified=mode == LongitudinalMode.UNIFIED, supervised=mode == LongitudinalMode.SUPERVISED)
  harness.selfdrive_state.enabled = False
  harness.selfdrive_state.active = False
  harness.selfdrive_state.state = State.disabled
  harness.car_state.vCruise = V_CRUISE_UNSET
  harness.plan.aTarget = -0.3
  harness.plan.motionIntent.valid = False
  harness.plan.motionIntent.ready = False
  harness.plan.motionIntent.creepEvidenceValid = False
  command, motion = harness.step()
  assert command.actuators.accel == 0.0
  assert not motion.degraded and not motion.waitingForFreshInput
  assert not harness.control.LoC.motion_supervisor._rearm_after_fault

  harness.car_state.vCruise = 80.0
  harness.plan.motionIntent.valid = True
  harness.plan.motionIntent.ready = True
  harness.plan.motionIntent.creepEvidenceValid = mode == LongitudinalMode.SUPERVISED
  harness.plan.motionIntent.creepReset = True
  harness.step()
  harness.selfdrive_state.enabled = True
  harness.selfdrive_state.active = True
  harness.selfdrive_state.state = State.enabled
  command, motion = harness.step(new_plan=False)
  assert motion.waitingForFreshInput and not motion.degraded
  assert command.actuators.accel == pytest.approx(-0.3)
  daemon, _, machine = event_consumers(harness)
  assert EventName.commIssue not in daemon.events.names
  assert machine.state == State.enabled


@pytest.mark.parametrize('fault', ['valid', 'version', 'mode', 'solver', 'clock', 'lease'])
def test_real_fault_during_waiting_still_reaches_actual_fault_consumers(mode, fault):
  harness = enter_off(mode)
  harness.step(new_plan=False)
  if fault == 'valid':
    harness.plan.motionIntent.valid = False
    command, motion = harness.step()
  elif fault == 'version':
    harness.plan.motionIntent.version = MOTION_INTENT_VERSION + 1
    command, motion = harness.step()
  elif fault == 'mode':
    harness.plan.motionIntent.mode = int(LongitudinalMode.UNIFIED if mode == LongitudinalMode.SUPERVISED else LongitudinalMode.SUPERVISED)
    command, motion = harness.step()
  elif fault == 'solver':
    harness.plan.solverStatus = 4
    harness.plan.motionIntent.valid = False
    harness.plan.motionIntent.brakingValid = True
    harness.plan.motionIntent.brakingAccel = -3.0
    command, motion = harness.step()
  elif fault == 'clock':
    previous_control_ns = harness.now_ns
    update = harness.control.LoC.update

    def repeated_control_clock(*args, **kwargs):
      kwargs['now_ns'] = previous_control_ns
      return update(*args, **kwargs)

    # Keep SubMaster reception/frequency timestamps monotonic. Inject the bad
    # consumer clock at the real LoC boundary, instead of dividing its message
    # frequency by a zero reception interval before reaching the controller.
    with patch.object(harness.control.LoC, 'update', side_effect=repeated_control_clock):
      command, motion = harness.step(new_plan=False)
  else:
    for _ in range(27):
      command, motion = harness.step(new_plan=False)
  assert motion.degraded and not motion.waitingForFreshInput
  assert harness.control.LoC.motion_input_constrained
  assert command.actuators.accel <= -0.2
  assert not command.cruiseControl.resume
  daemon, _, machine = event_consumers(harness)
  assert EventName.commIssue in daemon.events.names
  assert daemon.events.contains(ET.SOFT_DISABLE)
  assert machine.state == State.softDisabling


def test_a_causal_invalid_packet_cannot_be_repaired_with_the_same_ids_to_clear_fault(mode):
  harness = enter_off(mode)
  harness.step(new_plan=False)
  harness.plan.motionIntent.valid = False
  _, motion = harness.step()
  assert motion.degraded
  rejected_plan, rejected_model = int(motion.planMonoTime), int(motion.modelMonoTime)
  sources = tuple(harness.plan.motionIntent.creepSourceMonoTimes)
  harness.plan.motionIntent.valid = True
  command, repaired = harness.step(plan_time_ns=rejected_plan, model_time_ns=rejected_model, raw_source_times=sources)
  assert repaired.degraded and not repaired.waitingForFreshInput
  assert command.actuators.accel <= -0.2
  daemon, _, machine = event_consumers(harness)
  assert EventName.commIssue in daemon.events.names and machine.state == State.softDisabling
  _, fresh = harness.step()
  assert not fresh.degraded and not fresh.waitingForFreshInput


@pytest.mark.parametrize('cold_standby_between', [False, True])
def test_fault_cause_survives_off_and_cannot_be_relabelled_as_healthy_wait(mode, cold_standby_between):
  harness = enter_off(mode)
  harness.step(new_plan=False)
  harness.plan.motionIntent.valid = False
  _, fault = harness.step()
  assert fault.degraded
  harness.selfdrive_state.enabled = False
  harness.selfdrive_state.active = False
  harness.selfdrive_state.state = State.disabled
  if cold_standby_between:
    harness.car_state.vCruise = V_CRUISE_UNSET
    harness.plan.motionIntent.ready = False
    harness.plan.motionIntent.creepEvidenceValid = False
    harness.step()
  harness.car_state.vCruise = 80.0
  harness.plan.motionIntent.valid = True
  harness.plan.motionIntent.ready = True
  harness.plan.motionIntent.creepEvidenceValid = mode == LongitudinalMode.SUPERVISED
  harness.plan.motionIntent.creepReset = True
  harness.step()
  harness.selfdrive_state.enabled = True
  harness.selfdrive_state.active = True
  harness.selfdrive_state.state = State.enabled
  command, motion = harness.step(new_plan=False)
  assert motion.degraded and not motion.waitingForFreshInput
  assert command.actuators.accel <= -0.2
  daemon, _, machine = event_consumers(harness)
  assert EventName.commIssue in daemon.events.names and machine.state == State.softDisabling
  _, fresh = harness.step()
  assert not fresh.degraded and not fresh.waitingForFreshInput


@pytest.mark.parametrize('pedal', ['gasPressed', 'brakePressed'])
def test_manual_authority_immediately_clears_wait_and_output(mode, pedal):
  harness = enter_off(mode)
  harness.step(new_plan=False)
  setattr(harness.car_state, pedal, True)
  command, motion = harness.step(new_plan=False)
  assert command.actuators.accel == 0.0
  assert motion.phase == 'off'
  assert not motion.waitingForFreshInput and not motion.degraded
  assert not command.cruiseControl.resume


@pytest.mark.parametrize('placeholder', [None, MotionIntent(version=0, valid=False)])
def test_initial_absent_input_does_not_seed_a_fault_before_a_healthy_off_sample(mode, placeholder):
  owner = MotionSupervisor(mode)
  owner.update(False, 0.8, 0.0, False, None, placeholder, now_ns=1_900_000_000, plan_valid=False)
  assert not owner._rearm_after_fault
  off = (raw_intent(2_000_000_000, speed=0.8, reset=True, allow_creep=False) if mode == LongitudinalMode.SUPERVISED
         else MotionIntent(allow_creep=False))
  sample_step(owner, 2_000_000_000, intent=off, v_ego=0.8, active=False)
  sample_step(owner, 2_000_000_000, intent=off, v_ego=0.8, now_ns=2_012_000_000)
  assert owner.waiting_for_fresh_input and not owner.degraded


def test_noncausal_nonzero_input_does_not_masquerade_as_an_initial_placeholder(mode):
  owner = MotionSupervisor(mode)
  bad = (raw_intent(3_000_000_000, reset=True, allow_creep=False) if mode == LongitudinalMode.SUPERVISED
         else MotionIntent(allow_creep=False))
  sample_step(owner, 3_000_000_000, intent=bad, active=False, now_ns=2_000_000_000)
  assert owner._rearm_after_fault
  assert owner._latest_input_ids() == (0, 0)


def test_future_failed_header_does_not_poison_the_fresh_recovery_watermark(mode):
  harness = enter_off(mode)
  harness.step(new_plan=False)
  _, motion = harness.step(plan_time_ns=harness.now_ns + 10_000_000_000, model_time_ns=harness.now_ns + 9_000_000_000)
  assert motion.degraded
  _, fresh = harness.step()
  assert not fresh.degraded and not fresh.waitingForFreshInput


@pytest.mark.parametrize('fault', ['valid', 'version', 'mode', 'solver', 'nonfinite', 'future', 'backward', 'clock', 'expired'])
def test_real_fault_while_off_survives_a_healthy_off_packet_and_reentry(mode, fault):
  owner = MotionSupervisor(mode)

  def off_intent(timestamp):
    return (raw_intent(timestamp, reset=True, allow_creep=False) if mode == LongitudinalMode.SUPERVISED
            else MotionIntent(allow_creep=False))

  initial = off_intent(2_000_000_000)
  sample_step(owner, 2_000_000_000, intent=initial, active=False)
  candidate = off_intent(2_050_000_000)
  kwargs = {}
  if fault == 'valid':
    candidate = replace(candidate, valid=False)
  elif fault == 'version':
    candidate = replace(candidate, version=MOTION_INTENT_VERSION + 1)
  elif fault == 'mode':
    candidate = replace(candidate, mode=int(LongitudinalMode.UNIFIED if mode == LongitudinalMode.SUPERVISED else LongitudinalMode.SUPERVISED))
  elif fault == 'solver':
    kwargs['solver_status'] = 4
  elif fault == 'nonfinite':
    kwargs['a_ego'] = float('nan')
  elif fault == 'future':
    kwargs.update(plan_ns=3_000_000_000, model_ns=3_000_000_000)
  elif fault == 'backward':
    kwargs.update(plan_ns=1_900_000_000, model_ns=1_900_000_000)
  elif fault == 'clock':
    candidate = initial
    kwargs.update(plan_ns=2_001_000_000, model_ns=2_000_000_000, now_ns=2_002_000_000)
  else:
    candidate = initial
    kwargs.update(plan_ns=2_001_000_000, model_ns=2_000_000_000, now_ns=2_253_000_000)
  sample_step(owner, 2_050_000_000, intent=candidate, active=False, **kwargs)
  assert owner.phase == MotionPhase.OFF
  assert not owner.degraded and not owner.waiting_for_fresh_input
  assert owner._rearm_after_fault

  healthy = off_intent(2_600_000_000)
  sample_step(owner, 2_600_000_000, intent=healthy, active=False)
  sample_step(owner, 2_600_000_000, intent=healthy, now_ns=2_612_000_000)
  assert owner.degraded and not owner.waiting_for_fresh_input
  assert owner.reasons & StopReason.INVALID
  assert owner._require_new_input
  sample_step(owner, 2_650_000_000, intent=off_intent(2_650_000_000))
  assert not owner.degraded and not owner.waiting_for_fresh_input
  assert not owner._rearm_after_fault
