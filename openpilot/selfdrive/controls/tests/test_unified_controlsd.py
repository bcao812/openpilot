from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from opendbc.car.honda.values import CAR
from opendbc.car.structs import car
from openpilot.cereal import custom, log, messaging
from openpilot.cereal.services import SERVICE_LIST
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.car.card import Car
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.motion_supervisor import DEGRADED_ACCEL, MODEL_LEASE_NS, MOTION_INTENT_VERSION, PLAN_LEASE_NS, MotionIntent, StopReason
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import advance, make_planner, make_state
from openpilot.selfdrive.selfdrived.events import ET
from openpilot.selfdrive.selfdrived.state import SOFT_DISABLE_TIME, StateMachine
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import (
  CONTROL_STEP_NS, State, new_event, serialized, bounded_submaster, CapturingPublisher, ControlsHarness, selfdrive_event_harness,
)


EventName = log.OnroadEvent.EventName


def test_current_intent_and_stop_transition_are_consumed_and_published_in_the_same_frame():
  harness = ControlsHarness()
  command, motion = harness.step()
  assert command.longActive
  assert command.actuators.longControlState == LongCtrlState.pid
  assert motion.version == MOTION_INTENT_VERSION
  assert motion.planningMode == int(LongitudinalMode.UNIFIED)
  assert motion.enabled
  assert motion.phase == 'track'
  assert not motion.degraded
  harness.car_state.vEgo = 0.5
  harness.plan.aTarget = -2.5
  harness.plan.shouldStop = False
  harness.plan.motionIntent.reasons = int(StopReason.CLOSE_LEAD)
  command, motion = harness.step()
  controls_state = harness.control.pm.messages['controlsState'].controlsState
  assert command.longActive
  assert command.actuators.accel == -2.5
  assert command.actuators.longControlState == LongCtrlState.stopping
  assert controls_state.longControlState == command.actuators.longControlState
  assert motion.phase == 'approach'
  assert motion.reasons == int(StopReason.CLOSE_LEAD)
  assert motion.resolvedStop
  assert not motion.degraded
  assert motion.planMonoTime == controls_state.longitudinalPlanMonoTime == harness.now_ns
  assert motion.modelMonoTime == harness.plan.modelMonoTime
  assert harness.control.pm.messages['controlsState'].logMonoTime == harness.now_ns
  assert harness.control.pm.messages['carControl'].logMonoTime == harness.now_ns


@pytest.mark.parametrize('unified,version', [
  (False, MOTION_INTENT_VERSION), (True, 0), (True, 1), (True, 2),
  (True, MOTION_INTENT_VERSION + 1), (False, MOTION_INTENT_VERSION + 1),
])
def test_mode_mismatch_disables_actuation_and_resume_but_keeps_health_message_valid(unified, version):
  harness = ControlsHarness(platform=CAR.HONDA_ACCORD if unified else CAR.HONDA_CIVIC_BOSCH)
  harness.plan.motionIntent.version = version
  harness.car_state.cruiseState.standstill = True
  command, motion = harness.step()
  assert command.enabled
  assert not command.longActive
  assert command.actuators.accel == 0.0
  assert command.actuators.longControlState == LongCtrlState.off
  assert command.cruiseControl.override
  assert not command.cruiseControl.resume
  assert motion.version == MOTION_INTENT_VERSION
  assert motion.enabled == unified
  assert motion.phase == 'off'
  assert motion.degraded
  assert harness.control.pm.messages['controlsState'].valid
  assert harness.control.pm.messages['carControl'].valid


@pytest.mark.parametrize('condition', ['gas', 'brake', 'disable', 'override_event'])
def test_driver_authority_clears_motion_and_published_enable_resume_in_one_frame(condition):
  harness = ControlsHarness()
  harness.hold()
  harness.plan.aTarget = 0.5
  harness.plan.motionIntent.reasons = 0
  harness.plan.motionIntent.releaseAllowed = True
  harness.step()
  assert harness.command.cruiseControl.resume
  if condition in ('gas', 'brake'):
    setattr(harness.car_state, condition + 'Pressed', True)
  elif condition == 'disable':
    harness.selfdrive_state.enabled = False
  else:
    event = messaging.new_message('onroadEvents', 1, valid=True)
    event.onroadEvents[0].overrideLongitudinal = True
    harness.events['onroadEvents'] = event
  harness.plan.motionIntent.valid = False
  harness.plan.aTarget = float('nan')
  command, motion = harness.step(plan_time_ns=0, model_time_ns=0)
  assert not command.longActive
  assert not command.cruiseControl.resume
  assert command.actuators.accel == 0.0
  assert command.actuators.longControlState == LongCtrlState.off
  assert motion.phase == 'off'
  assert not motion.reasons
  assert not motion.resolvedStop
  assert not motion.degraded


def test_oem_standstill_does_not_circularly_block_the_real_resume_publisher():
  harness = ControlsHarness()
  harness.hold()
  harness.plan.aTarget = 0.5
  harness.plan.shouldStop = True
  harness.plan.motionIntent.reasons = 0
  harness.plan.motionIntent.releaseAllowed = True
  for _ in range(5):
    command, motion = harness.step()
    assert command.longActive
    assert command.cruiseControl.resume
    assert command.actuators.longControlState == LongCtrlState.stopping
    assert command.actuators.accel <= -2.0
    assert motion.phase == 'release'
    assert not motion.resolvedStop
    assert not motion.degraded
  previous = command.actuators.accel
  harness.car_state.cruiseState.standstill = False
  command, motion = harness.step()
  assert not command.cruiseControl.resume
  assert command.actuators.longControlState == LongCtrlState.pid
  assert command.actuators.accel == pytest.approx(previous + 8.0 * DT_CTRL)
  assert motion.phase == 'release'


@pytest.mark.parametrize('fault', [
  'intent_invalid', 'unknown_reason', 'derived_reason', 'invalid_plan', 'invalid_car_state',
  'future_plan', 'future_model', 'expired_model', 'model_after_plan', 'plan_backwards', 'model_backwards',
])
def test_real_consumer_rejects_bad_metadata_without_releasing_an_existing_hold(fault):
  harness = ControlsHarness()
  harness.hold()
  previous_time = harness.now_ns
  harness.plan.aTarget = 1.0
  harness.plan.motionIntent.reasons = 0
  kwargs = {}
  if fault == 'intent_invalid':
    harness.plan.motionIntent.valid = False
  elif fault == 'unknown_reason':
    harness.plan.motionIntent.reasons = 1 << 20
  elif fault == 'derived_reason':
    harness.plan.motionIntent.reasons = int(StopReason.HOLD)
  elif fault == 'invalid_plan':
    harness.events['longitudinalPlan'].valid = False
  elif fault == 'invalid_car_state':
    harness.events['carState'].valid = False
  elif fault == 'future_plan':
    kwargs['plan_time_ns'] = previous_time + 2 * CONTROL_STEP_NS
  elif fault == 'future_model':
    kwargs['model_time_ns'] = previous_time + 2 * CONTROL_STEP_NS
  elif fault == 'expired_model':
    kwargs['model_time_ns'] = previous_time - PLAN_LEASE_NS
  elif fault == 'model_after_plan':
    kwargs['plan_time_ns'] = previous_time
  elif fault == 'plan_backwards':
    kwargs.update(plan_time_ns=previous_time - CONTROL_STEP_NS, model_time_ns=previous_time - CONTROL_STEP_NS)
  elif fault == 'model_backwards':
    kwargs['model_time_ns'] = previous_time - CONTROL_STEP_NS
  command, motion = harness.step(**kwargs)
  assert command.longActive
  assert command.actuators.accel <= -2.0
  assert not command.cruiseControl.resume
  assert motion.phase == 'hold'
  assert motion.resolvedStop
  assert motion.reasons & int(StopReason.INVALID)
  assert motion.degraded
  assert harness.control.pm.messages['controlsState'].valid


def test_repeated_positive_plan_expires_at_the_consumer_before_submaster_alive_expires():
  harness = ControlsHarness()
  harness.step()
  origin = harness.now_ns
  for _ in range(PLAN_LEASE_NS // CONTROL_STEP_NS):
    command, motion = harness.step(new_plan=False)
    assert harness.control.sm.all_checks(['longitudinalPlan', 'carState'])
    assert not motion.degraded
    assert command.actuators.accel > 0.0
  assert harness.now_ns - origin == PLAN_LEASE_NS
  command, motion = harness.step(new_plan=False)
  assert harness.control.sm.alive['longitudinalPlan']
  assert not harness.control.sm.updated['longitudinalPlan']
  assert command.actuators.accel <= DEGRADED_ACCEL + 1e-6
  assert motion.degraded
  assert motion.reasons & int(StopReason.INVALID)
  assert motion.planMonoTime == origin


@pytest.mark.parametrize('fault', ['invalid', 'unknown_reason', 'expired'])
def test_degraded_tracking_cannot_issue_resume_when_oem_reports_standstill(fault):
  harness = ControlsHarness()
  harness.car_state.vEgo = 0.5
  harness.step()
  assert harness.motion.phase == 'track'
  assert not harness.car_state.standstill
  harness.car_state.cruiseState.standstill = True
  if fault == 'invalid':
    harness.plan.motionIntent.valid = False
    command, motion = harness.step()
  elif fault == 'unknown_reason':
    harness.plan.motionIntent.reasons = 1 << 20
    command, motion = harness.step()
  else:
    for _ in range(PLAN_LEASE_NS // CONTROL_STEP_NS + 1):
      command, motion = harness.step(new_plan=False)
  assert motion.degraded
  assert command.actuators.accel <= DEGRADED_ACCEL + 1e-6
  assert not command.cruiseControl.resume


def test_republished_same_identity_does_not_fabricate_departure_confirmation():
  harness = ControlsHarness()
  harness.hold()
  harness.plan.aTarget = 0.15
  harness.plan.motionIntent.reasons = 0
  harness.plan.motionIntent.releaseAllowed = True
  harness.step(now_ns=harness.now_ns + round(DT_MDL * 1e9))
  origin = harness.now_ns
  for _ in range(20):
    command, motion = harness.step(plan_time_ns=origin, model_time_ns=origin)
    assert harness.control.sm.updated['longitudinalPlan']
    assert not motion.degraded
    assert motion.resolvedStop
    assert not command.cruiseControl.resume
    assert harness.control.LoC.motion_supervisor.release_lead_t == pytest.approx(DT_MDL)


@pytest.mark.parametrize('mutation', ['target', 'reasons', 'model_time'])
def test_changed_payload_with_the_same_plan_identity_is_rejected(mutation):
  harness = ControlsHarness()
  harness.hold()
  origin = harness.now_ns
  if mutation == 'target':
    harness.plan.aTarget = 1.0
  elif mutation == 'reasons':
    harness.plan.motionIntent.reasons = 0
  command, motion = harness.step(plan_time_ns=origin, model_time_ns=origin - 1 if mutation == 'model_time' else origin)
  assert command.actuators.accel <= -2.0
  assert not command.cruiseControl.resume
  assert motion.resolvedStop
  assert motion.degraded


def test_disable_clears_a_fault_immediately_and_reenable_requires_new_evidence():
  harness = ControlsHarness()
  harness.step()
  harness.events['longitudinalPlan'].valid = False
  command, motion = harness.step()
  assert motion.degraded
  harness.selfdrive_state.enabled = False
  command, motion = harness.step(new_plan=False)
  assert not command.longActive
  assert command.actuators.accel == 0.0
  assert motion.phase == 'off'
  assert not motion.degraded
  assert not motion.reasons
  harness.events['longitudinalPlan'].valid = True
  harness.selfdrive_state.enabled = True
  original_identity = harness.control.LoC.motion_supervisor._last_plan_time_ns
  harness.plan.aTarget = 0.5
  command, motion = harness.step(plan_time_ns=original_identity, model_time_ns=original_identity)
  assert motion.degraded
  assert command.actuators.accel <= DEGRADED_ACCEL + 1e-6
  command, motion = harness.step()
  assert command.longActive
  assert not motion.degraded
  assert motion.phase == 'track'


@pytest.mark.parametrize('local_unified', [False, True])
def test_published_mode_acknowledgement_is_consumed_by_the_real_planner(local_unified):
  with OpenpilotPrefix():
    harness = ControlsHarness(platform=CAR.HONDA_ACCORD if local_unified else CAR.HONDA_CIVIC_BOSCH)
    harness.plan.motionIntent.version = 0 if local_unified else MOTION_INTENT_VERSION
    harness.plan.motionIntent.mode = int(LongitudinalMode.LEGACY if local_unified else LongitudinalMode.UNIFIED)
    command, motion = harness.step()
    assert motion.version == MOTION_INTENT_VERSION
    assert motion.planningMode == int(LongitudinalMode.UNIFIED if local_unified else LongitudinalMode.LEGACY)
    assert motion.enabled == local_unified
    assert motion.degraded
    assert not command.longActive
    state = make_state(speed=10.0)
    state['controlsState'] = harness.control.pm.messages['controlsState'].controlsState.as_builder()
    state['carControl'] = harness.control.pm.messages['carControl'].carControl.as_builder()
    for service in state.logMonoTime:
      state.logMonoTime[service] = harness.now_ns
    planner, extension = make_planner(speed=10.0)
    planner.update(state, now_ns=harness.now_ns)
    assert planner.motion_intent.valid == local_unified
    publisher = CapturingPublisher()
    with patch('openpilot.cereal.messaging.time.monotonic_ns', return_value=harness.now_ns):
      planner.publish(state, publisher)
    plan_event = publisher.messages['longitudinalPlan']
    assert plan_event.longitudinalPlan.motionIntent.version == MOTION_INTENT_VERSION
    assert plan_event.valid == local_unified
    harness.events['longitudinalPlan'] = plan_event.as_builder()
    harness.plan = harness.events['longitudinalPlan'].longitudinalPlan
    command, motion = harness.step()
    if local_unified:
      assert command.longActive
      assert not motion.degraded
    else:
      assert not command.longActive
      assert motion.degraded


@pytest.mark.parametrize('fault', ['missing_ack', 'nonfinite_speed', 'invalid_model_shape'])
def test_cold_planner_rejection_can_still_publish_both_plan_channels(fault):
  with OpenpilotPrefix():
    planner, extension = make_planner(speed=10.0)
    state = make_state(speed=10.0)
    if fault == 'missing_ack':
      state['controlsState'].motionState.version = 0
    elif fault == 'nonfinite_speed':
      state['carState'].vEgo = float('nan')
    else:
      state['modelV2'].velocity.x = []
    planner.update(state, now_ns=advance(state, 0))
    assert not planner.motion_intent.valid
    publisher = CapturingPublisher()
    planner.publish(state, publisher)
    assert set(publisher.messages) == {'longitudinalPlan', 'longitudinalPlanSP'}
    assert not publisher.messages['longitudinalPlan'].valid
    assert not publisher.messages['longitudinalPlanSP'].valid
    assert not publisher.messages['longitudinalPlan'].longitudinalPlan.motionIntent.valid


@pytest.mark.parametrize('local_unified', [False, True])
def test_real_selfdrived_events_convert_degradation_to_three_second_soft_disable(local_unified):
  harness = ControlsHarness(platform=CAR.HONDA_ACCORD if local_unified else CAR.HONDA_CIVIC_BOSCH)
  if local_unified:
    harness.plan.motionIntent.valid = False
  else:
    harness.plan.motionIntent.version = MOTION_INTENT_VERSION
    harness.plan.motionIntent.mode = int(LongitudinalMode.UNIFIED)
  harness.step()
  state_event = harness.control.pm.messages['controlsState']
  assert state_event.valid
  assert state_event.controlsState.motionState.degraded
  daemon, car_state = selfdrive_event_harness(state_event)
  assert daemon.sm.all_checks()
  daemon.update_events(car_state)
  assert daemon.events.names == [EventName.longitudinalControlUnavailable]
  assert daemon.events.contains(ET.NO_ENTRY)
  assert daemon.events.contains(ET.SOFT_DISABLE)
  machine = StateMachine()
  machine.state = State.enabled
  assert machine.update(daemon.events) == (True, True)
  assert machine.state == State.softDisabling
  assert SOFT_DISABLE_TIME == 3
  assert machine.soft_disable_timer == 300
  harness.selfdrive_state.state = machine.state
  command, motion = harness.step()
  assert harness.control.pm.messages['controlsState'].controlsState.forceDecel
  assert command.driverMonitoringEscalation
  for _ in range(int(SOFT_DISABLE_TIME / DT_CTRL) - 1):
    assert machine.update(daemon.events) == (True, True)
    assert machine.state == State.softDisabling
  assert machine.update(daemon.events) == (False, False)
  assert machine.state == State.disabled
  harness.selfdrive_state.state = machine.state
  harness.selfdrive_state.enabled = False
  command, motion = harness.step()
  assert not command.longActive
  assert not command.cruiseControl.resume
  assert command.actuators.accel == 0.0
  assert motion.phase == 'off'


@pytest.mark.parametrize('version,degraded,openpilot_long', [
  (0, True, True), (MOTION_INTENT_VERSION, False, True), (MOTION_INTENT_VERSION, True, False),
  (MOTION_INTENT_VERSION + 1, True, True),
])
def test_motion_event_gate_does_not_reinterpret_legacy_or_stock_longitudinal(version, degraded, openpilot_long):
  state_event = new_event('controlsState')
  state_event.controlsState.lateralControlState.init('pidState')
  state_event.controlsState.motionState.version = version
  state_event.controlsState.motionState.degraded = degraded
  daemon, car_state = selfdrive_event_harness(serialized(state_event))
  daemon.CP.openpilotLongitudinalControl = openpilot_long
  daemon.update_events(car_state)
  assert EventName.commIssue not in daemon.events.names
  assert EventName.longitudinalControlUnavailable not in daemon.events.names


def test_fresh_recovery_clears_the_longitudinal_fault_without_a_validity_feedback_deadlock():
  harness = ControlsHarness()
  harness.plan.motionIntent.valid = False
  harness.step()
  daemon, car_state = selfdrive_event_harness(harness.control.pm.messages['controlsState'])
  daemon.update_events(car_state)
  machine = StateMachine()
  machine.state = State.enabled
  machine.update(daemon.events)
  assert machine.state == State.softDisabling
  harness.plan.motionIntent.valid = True
  harness.step()
  recovered = harness.control.pm.messages['controlsState']
  assert recovered.valid
  assert not recovered.controlsState.motionState.degraded
  daemon.sm.update_msgs(2.01, [recovered])
  daemon.update_events(car_state)
  assert not daemon.events.names
  assert machine.update(daemon.events) == (True, True)
  assert machine.state == State.enabled


def radar_loss_event_harness():
  harness = ControlsHarness()
  harness.plan.radarRiskUnresolved = True
  harness.step()
  assert not harness.motion.degraded
  daemon, car_state = selfdrive_event_harness(harness.control.pm.messages['controlsState'])
  daemon.CP, daemon.CP_SP = harness.control.CP, harness.control.CP_SP
  daemon.sm.update_msgs(harness.now_ns / 1e9, [serialized(harness.events['longitudinalPlan'])])
  return harness, daemon, car_state


@pytest.mark.parametrize('defect', [
  'none', 'no_claim', 'unseen', 'invalid', 'invalid_intent', 'dead', 'bad_frequency', 'old_plan', 'old_model',
  'future_plan', 'future_model', 'model_after_plan', 'missing_model', 'wrong_version', 'wrong_mode',
  'local_legacy', 'stock_long', 'interceptor',
])
def test_historical_radar_loss_claim_never_adds_a_dedicated_sensor_failure_event(defect):
  harness, daemon, car_state = radar_loss_event_harness()
  event = harness.events['longitudinalPlan']
  plan = event.longitudinalPlan
  now = harness.now_ns
  if defect == 'no_claim':
    plan.radarRiskUnresolved = False
  elif defect == 'invalid':
    event.valid = False
  elif defect == 'invalid_intent':
    assert event.valid
    plan.motionIntent.valid = False
  elif defect == 'old_plan':
    event.logMonoTime = now - PLAN_LEASE_NS - 1
    plan.modelMonoTime = event.logMonoTime
  elif defect == 'old_model':
    plan.modelMonoTime = now - MODEL_LEASE_NS - 1
  elif defect == 'future_plan':
    event.logMonoTime = now + 1
  elif defect == 'future_model':
    plan.modelMonoTime = now + 1
  elif defect == 'model_after_plan':
    event.logMonoTime = now - 2
    plan.modelMonoTime = now - 1
  elif defect == 'missing_model':
    plan.modelMonoTime = 0
  elif defect == 'wrong_version':
    plan.motionIntent.version = MOTION_INTENT_VERSION + 1
  elif defect == 'wrong_mode':
    plan.motionIntent.mode = int(LongitudinalMode.LEGACY)
  elif defect == 'local_legacy':
    daemon.CP.carFingerprint = CAR.HONDA_CIVIC_BOSCH
  elif defect == 'stock_long':
    daemon.CP.openpilotLongitudinalControl = daemon.cp_olc = False
  elif defect == 'interceptor':
    daemon.CP_SP.enableGasInterceptor = True
  daemon.sm.update_msgs(now / 1e9, [serialized(event)])
  if defect == 'unseen':
    daemon.sm.seen['longitudinalPlan'] = False
  elif defect == 'dead':
    daemon.sm.alive['longitudinalPlan'] = False
  elif defect == 'bad_frequency':
    daemon.sm.ignore_average_freq.remove('longitudinalPlan')
    daemon.sm.freq_ok['longitudinalPlan'] = False
  with patch('openpilot.selfdrive.selfdrived.selfdrived.time.monotonic_ns', return_value=now):
    daemon.update_events(car_state)
  assert EventName.radarTempUnavailable not in daemon.events.names
  if defect == 'none':
    assert EventName.commIssue not in daemon.events.names


def test_healthy_control_remains_enabled_after_more_than_three_seconds_of_radar_loss_diagnostics():
  harness, daemon, car_state = radar_loss_event_harness()
  background = []
  for service in daemon.sm.data:
    if service not in ('controlsState', 'longitudinalPlan'):
      event = new_event(service)
      setattr(event, service, daemon.sm[service])
      background.append(event)
  machine = StateMachine()
  machine.state = State.enabled
  origin = harness.now_ns
  for _ in range(int(SOFT_DISABLE_TIME / DT_CTRL) + 20):
    harness.selfdrive_state.state = machine.state
    command, motion = harness.step()
    incoming = [harness.control.pm.messages['controlsState'], serialized(harness.events['longitudinalPlan'])]
    for event in background:
      event.logMonoTime = harness.now_ns
      incoming.append(serialized(event))
    daemon.sm.update_msgs(harness.now_ns / 1e9, incoming)
    # Preserve the harness's bounded collaborators while keeping every
    # transport fresh; this test changes only the historical radar claim.
    daemon.sm.updated = dict.fromkeys(daemon.sm.updated, False)
    with patch('openpilot.selfdrive.selfdrived.selfdrived.time.monotonic_ns', return_value=harness.now_ns):
      daemon.update_events(car_state)
    assert EventName.radarTempUnavailable not in daemon.events.names
    assert EventName.commIssue not in daemon.events.names
    assert machine.update(daemon.events) == (True, True)
    assert machine.state == State.enabled and command.longActive and not motion.degraded
    assert not harness.control.pm.messages['controlsState'].controlsState.forceDecel
    assert not command.driverMonitoringEscalation
  assert harness.now_ns - origin > int(SOFT_DISABLE_TIME * 1e9)
  assert harness.plan.radarRiskUnresolved


def test_clearing_a_diagnostic_claim_does_not_change_engagement_state():
  harness, daemon, car_state = radar_loss_event_harness()
  with patch('openpilot.selfdrive.selfdrived.selfdrived.time.monotonic_ns', return_value=harness.now_ns):
    daemon.update_events(car_state)
  machine = StateMachine()
  machine.state = State.enabled
  machine.update(daemon.events)
  assert machine.state == State.enabled
  harness.plan.radarRiskUnresolved = False
  harness.step()
  daemon.sm.update_msgs(harness.now_ns / 1e9, [serialized(harness.events['longitudinalPlan'])])
  with patch('openpilot.selfdrive.selfdrived.selfdrived.time.monotonic_ns', return_value=harness.now_ns):
    daemon.update_events(car_state)
  assert EventName.radarTempUnavailable not in daemon.events.names
  assert machine.update(daemon.events) == (True, True) and machine.state == State.enabled


@pytest.mark.parametrize('situation', ['adopted_hold', 'model_braking'])
def test_ignoring_a_history_flag_does_not_release_a_real_hold_or_weaken_braking(situation):
  outputs = []
  for risk in (False, True):
    harness = ControlsHarness()
    if situation == 'adopted_hold':
      harness.hold()
      harness.plan.motionIntent.reasons = int(StopReason.E2E)
      harness.plan.motionIntent.releaseAllowed = False
      harness.plan.aTarget = .5
    else:
      harness.plan.aTarget = -.8
    harness.plan.radarRiskUnresolved = risk
    command, motion = harness.step()
    assert command.actuators.accel < 0.0
    if situation == 'adopted_hold':
      assert motion.phase == 'hold' and motion.resolvedStop
      assert not command.cruiseControl.resume
    outputs.append((command.actuators.accel, motion.phase, motion.resolvedStop, motion.degraded))
  assert outputs[0] == outputs[1]


def test_card_can_send_the_last_positive_command_for_its_alive_tail_after_controlsd_stops():
  """This is nominal send eligibility, not a 250 ms hard-real-time deadline or proof of physical actuation."""
  harness = ControlsHarness()
  harness.step()
  plan_origin = harness.now_ns
  for _ in range(PLAN_LEASE_NS // CONTROL_STEP_NS):
    harness.step(new_plan=False)
  assert harness.now_ns == plan_origin + PLAN_LEASE_NS
  last_command = harness.control.pm.messages['carControl']
  assert not harness.motion.degraded
  assert last_command.carControl.actuators.accel > 0.0
  card = Car.__new__(Car)
  card.CP = harness.control.CP
  card.CP_SP = harness.control.CP_SP
  card.sm = bounded_submaster(['carControl'])
  card.sm.update_msgs(harness.now_ns / 1e9, [last_command])
  card.initialized_prev = True
  card.CI = SimpleNamespace(apply=Mock(return_value=(car.CarControl.Actuators.new_message(), [])))
  card.pm = CapturingPublisher()
  command_sp = custom.CarControlSP.new_message()
  assert 10.0 / SERVICE_LIST['carControl'].frequency == pytest.approx(0.1)
  with patch('openpilot.selfdrive.car.card.REPLAY', False):
    for index, elapsed_ns in enumerate((0, 50_000_000, 99_000_000), start=1):
      card.sm.update_msgs((harness.now_ns + elapsed_ns) / 1e9, [])
      assert card.sm.all_alive(['carControl'])
      card.controls_update(harness.car_state, card.sm['carControl'], command_sp)
      assert card.CI.apply.call_count == index
      assert card.CI.apply.call_args.args[0].longActive
      assert card.CI.apply.call_args.args[0].actuators.accel > 0.0
      assert card.pm.messages['sendcan'].valid
    card.sm.update_msgs((harness.now_ns + 101_000_000) / 1e9, [])
    assert not card.sm.all_alive(['carControl'])
    card.controls_update(harness.car_state, card.sm['carControl'], command_sp)
    assert card.CI.apply.call_count == 3
    assert len(card.pm.history) == 3


@pytest.mark.parametrize('fault', ['plan_invalid', 'car_invalid', 'nonfinite', 'future', 'control_gap'])
def test_rejected_release_needs_new_producer_ids_before_actual_resume(fault):
  harness = ControlsHarness()
  harness.hold()
  harness.plan.aTarget = 0.3
  harness.plan.motionIntent.reasons = 0
  harness.plan.motionIntent.releaseAllowed = True
  command, motion = harness.step()
  assert command.cruiseControl.resume
  assert motion.phase == 'release'
  accepted_time = harness.now_ns
  kwargs = {'plan_time_ns': accepted_time, 'model_time_ns': accepted_time}
  if fault == 'plan_invalid':
    harness.events['longitudinalPlan'].valid = False
  elif fault == 'car_invalid':
    harness.events['carState'].valid = False
  elif fault == 'nonfinite':
    harness.car_state.aEgo = float('nan')
  elif fault == 'future':
    kwargs['plan_time_ns'] = harness.now_ns + 1_000_000_000
  else:
    kwargs['now_ns'] = harness.now_ns + 110_000_000
  command, motion = harness.step(**kwargs)
  assert motion.degraded
  assert not command.cruiseControl.resume
  harness.events['longitudinalPlan'].valid = True
  harness.events['carState'].valid = True
  harness.car_state.aEgo = 0.0
  command, motion = harness.step(plan_time_ns=accepted_time, model_time_ns=accepted_time)
  assert motion.degraded
  assert motion.resolvedStop
  assert motion.phase == 'hold'
  assert command.actuators.accel <= -2.0
  assert not command.cruiseControl.resume
  command, motion = harness.step()
  assert not motion.degraded
  assert motion.phase == 'release'
  assert command.cruiseControl.resume


@pytest.mark.parametrize('speed', [0.0, 0.2, 1.0, 20.0])
def test_unified_consumer_uses_qualified_solver_failure_braking_without_a_positive_ramp_or_resume(speed):
  harness = ControlsHarness()
  harness.car_state.vEgo = speed
  harness.car_state.standstill = speed == 0.0
  harness.car_state.cruiseState.standstill = True
  # Establish actual 100 Hz carState / 20 Hz plan reception while keeping the
  # owner OFF. A first message alone cannot qualify independent fault braking.
  harness.selfdrive_state.enabled = harness.selfdrive_state.active = False
  harness.selfdrive_state.state = State.disabled
  for frame in range(10):
    harness.step(new_plan=frame % 5 == 0)
  assert harness.motion.phase == 'off'
  for service in ('carState', 'longitudinalPlan'):
    assert all(getattr(harness.control.sm, health)[service] is True for health in ('seen', 'alive', 'freq_ok'))
  harness.selfdrive_state.enabled = harness.selfdrive_state.active = True
  harness.selfdrive_state.state = State.enabled
  harness.control.LoC.last_output_accel = 0.3
  harness.plan.aTarget = -0.5
  harness.plan.shouldStop = speed < 0.3
  harness.plan.solverStatus = 4
  harness.plan.motionIntent.valid = False
  harness.plan.motionIntent.brakingValid = True
  harness.plan.motionIntent.brakingAccel = -0.5
  harness.events['longitudinalPlan'].valid = False
  command, motion = harness.step()
  assert command.longActive
  assert motion.degraded
  assert harness.control.reference_timing_check.valid
  assert harness.plan.motionIntent.producerInstanceMonoTime == harness.plan.reference.producerInstanceMonoTime > 0
  assert MotionIntent.from_message(harness.plan.motionIntent).is_braking_valid()
  assert harness.control.LoC.motion_supervisor.degraded_braking_accel == -0.5
  assert command.actuators.accel <= (-2.0 if speed == 0.0 else -0.5)
  assert not command.cruiseControl.resume
  harness.car_state.brakePressed = True
  command, _ = harness.step()
  assert not command.longActive
  assert command.actuators.accel == 0.0
  assert command.actuators.longControlState == LongCtrlState.off


def test_solver_failure_cannot_be_overridden_by_inconsistent_valid_flags():
  harness = ControlsHarness()
  harness.hold()
  harness.plan.aTarget = 1.0
  harness.plan.motionIntent.reasons = 0
  harness.plan.solverStatus = 4
  assert harness.events['longitudinalPlan'].valid
  assert harness.plan.motionIntent.valid
  command, motion = harness.step()
  assert motion.degraded
  assert motion.resolvedStop
  assert command.actuators.accel <= -2.0
  assert not command.cruiseControl.resume


@pytest.mark.parametrize('pedal', ['gasPressed', 'brakePressed'])
@pytest.mark.parametrize('unified', [False, True])
def test_card_pedal_edge_restriction_is_scoped_and_keeps_cached_control_immutable(pedal, unified):
  harness = ControlsHarness(platform=CAR.HONDA_ACCORD if unified else CAR.HONDA_CIVIC_BOSCH)
  harness.step()
  last_command = harness.control.pm.messages['carControl']
  assert last_command.carControl.longActive
  assert last_command.carControl.actuators.accel > 0.0
  card = Car.__new__(Car)
  card.CP = harness.control.CP
  card.CP_SP = harness.control.CP_SP
  card.sm = bounded_submaster(['carControl'])
  card.sm.update_msgs(harness.now_ns / 1e9, [last_command])
  card.initialized_prev = True
  card.CI = SimpleNamespace(apply=Mock(return_value=(car.CarControl.Actuators.new_message(), [])))
  card.pm = CapturingPublisher()
  setattr(harness.car_state, pedal, True)
  with patch('openpilot.selfdrive.car.card.REPLAY', False):
    card.controls_update(harness.car_state, card.sm['carControl'], custom.CarControlSP.new_message())
  applied = card.CI.apply.call_args.args[0]
  if unified:
    assert not applied.longActive
    assert applied.actuators.accel == 0.0
    assert applied.actuators.longControlState == LongCtrlState.off
    assert not applied.cruiseControl.resume
  else:
    assert applied.to_dict() == last_command.carControl.to_dict()
  assert applied.enabled == last_command.carControl.enabled
  assert applied.latActive == last_command.carControl.latActive
  assert card.sm['carControl'].longActive
  assert card.sm['carControl'].actuators.accel > 0.0
  assert card.pm.messages['sendcan'].valid
