from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.hondacan import CanBus
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC, CarControllerParams
from opendbc.car.structs import car
from openpilot.cereal import log, messaging
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.motion_supervisor import DEGRADED_ACCEL, MOTION_INTENT_VERSION, PLAN_LEASE_NS, MotionIntent, MotionPhase, StopReason
from openpilot.selfdrive.controls.tests.test_longitudinal_modes import MOTION_MODES, mode_controls
from openpilot.selfdrive.controls.tests.test_supervised_longitudinal_planner import make_supervised, make_supervised_state
from openpilot.selfdrive.controls.tests.test_unified_controlsd import selfdrive_event_harness, serialized
from openpilot.selfdrive.controls.tests.test_unified_longcontrol import ControlHarness, controller
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_planner import Publisher, advance


MISSING_HEALTH = object()


@pytest.fixture(autouse=True)
def isolated_params():
  with OpenpilotPrefix():
    yield


def brake_intent(expected_mode, **changes):
  return replace(MotionIntent(mode=int(expected_mode), valid=False, allow_creep=False, braking_accel=-3.0), **changes)


def warm_controller(mode):
  harness = ControlHarness(controller(unified=mode == LongitudinalMode.UNIFIED, supervised=mode == LongitudinalMode.SUPERVISED))
  harness.state.vEgo = 10.0
  assert harness.step(-0.4, harness.normal_intent()) == -0.4
  assert harness.control.motion_owner_established
  return harness


def native_command(harness):
  parameters = CarInterface.get_non_essential_params(CAR.HONDA_ACCORD)
  parameters.openpilotLongitudinalControl = True
  extension = CarInterface.get_non_essential_params_sp(parameters, CAR.HONDA_ACCORD)
  dbc_name = DBC[parameters.carFingerprint][Bus.pt]
  native = CarController({Bus.pt: dbc_name}, parameters, extension)
  parser = CANParser(dbc_name, [('ACC_CONTROL', 50)], CanBus(parameters).pt)
  vehicle = SimpleNamespace(out=harness.car_state, v_cruise_factor=1.0, is_metric=True,
                            acc_hud={}, lkas_hud={}, stock_brake={})
  output, packets = native.update(harness.command, structs.CarControlSP(), vehicle, harness.now_ns)
  parser.update([(harness.now_ns, packets)])
  return output, dict(parser.vl['ACC_CONTROL'])


def warm_controls(mode, phase):
  harness = mode_controls(mode)
  if phase == 'approach':
    harness.car_state.vEgo = 0.5
    harness.plan.aTarget = -1.2
    harness.plan.motionIntent.reasons = int(StopReason.MPC)
    harness.step()
  elif phase == 'track':
    harness.plan.aTarget = -0.4
    for frame in range(10):
      harness.step(new_plan=frame % 5 == 0)
    assert harness.control.sm.freq_ok['longitudinalPlan']
  else:
    harness.hold()
    if phase == 'release':
      harness.plan.aTarget = 0.5
      harness.plan.motionIntent.reasons = 0
      harness.plan.motionIntent.releaseAllowed = True
      harness.step()
  assert harness.motion.phase == phase
  assert harness.control.LoC.motion_owner_established
  return harness


def wire_brake(harness, acceleration=-3.0):
  harness.plan.aTarget = acceleration
  harness.plan.motionIntent.valid = False
  harness.plan.motionIntent.ready = True
  harness.plan.motionIntent.movingLead = False
  harness.plan.motionIntent.releaseAllowed = False
  harness.plan.motionIntent.allowCreep = False
  harness.plan.motionIntent.brakingValid = True
  harness.plan.motionIntent.brakingAccel = acceleration
  harness.plan.solverStatus = 4
  harness.events['longitudinalPlan'].valid = False


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('phase', ['hold', 'approach', 'release', 'track'])
@pytest.mark.parametrize('fault', ['v1', 'mode'])
def test_warm_protocol_fault_keeps_braking_and_disables_resume_through_real_honda_can(mode, phase, fault):
  harness = warm_controls(mode, phase)
  previous = harness.command.actuators.accel
  if fault == 'v1':
    harness.plan.motionIntent.version = 1
  else:
    harness.plan.motionIntent.mode = int(LongitudinalMode.UNIFIED if mode == LongitudinalMode.SUPERVISED else LongitudinalMode.SUPERVISED)
  command, motion = harness.step()
  output, decoded = native_command(harness)
  assert command.longActive
  assert command.actuators.accel <= min(previous, DEGRADED_ACCEL) + 1e-6
  assert output.accel <= min(previous, DEGRADED_ACCEL) + 1e-6
  assert decoded['BRAKE_REQUEST']
  assert decoded['GAS_COMMAND'] <= 0
  assert not command.cruiseControl.resume
  assert motion.degraded
  assert motion.phase != 'off'
  assert harness.control.LoC.motion_owner_established
  daemon, car_state = selfdrive_event_harness(harness.control.pm.messages['controlsState'])
  daemon.update_events(car_state)
  assert log.OnroadEvent.EventName.commIssue in daemon.events.names


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('phase', ['hold', 'approach', 'release'])
@pytest.mark.parametrize('authority', ['gas', 'brake', 'disabled', 'override'])
def test_driver_cancellation_wins_over_warm_protocol_fault_and_native_braking(mode, phase, authority):
  harness = warm_controls(mode, phase)
  harness.plan.motionIntent.version = 1
  if authority in ('gas', 'brake'):
    setattr(harness.car_state, authority + 'Pressed', True)
  elif authority == 'disabled':
    harness.selfdrive_state.enabled = False
  else:
    event = messaging.new_message('onroadEvents', 1, valid=True)
    event.onroadEvents[0].overrideLongitudinal = True
    harness.events['onroadEvents'] = event
  command, motion = harness.step()
  output, decoded = native_command(harness)
  assert not command.longActive
  assert command.actuators.accel == output.accel == decoded['ACCEL_COMMAND'] == 0.0
  assert not decoded['BRAKE_REQUEST']
  assert decoded['GAS_COMMAND'] <= 0
  assert not command.cruiseControl.resume
  assert motion.phase == 'off'
  assert not harness.control.LoC.motion_owner_established
  if authority == 'disabled':
    assert not decoded['CONTROL_ON']


@pytest.mark.parametrize('mode', MOTION_MODES)
def test_previous_motion_history_does_not_reactivate_an_off_owner_on_mismatch(mode):
  harness = warm_controls(mode, 'hold')
  harness.selfdrive_state.enabled = False
  harness.step()
  assert harness.motion.phase == 'off'
  assert not harness.control.LoC.motion_owner_established
  harness.selfdrive_state.enabled = True
  harness.plan.motionIntent.version = 1
  command, motion = harness.step()
  assert not command.longActive
  assert command.actuators.accel == 0.0
  assert motion.phase == 'off'
  assert motion.degraded


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('reason', [StopReason.NONE, StopReason.MPC, StopReason.E2E, StopReason.CRUISE, StopReason.FORCE_DECEL])
def test_braking_contract_is_separate_from_normal_movement_permission(mode, reason):
  intent = brake_intent(mode, reasons=reason)
  assert intent.is_braking_valid()
  assert not intent.is_valid()
  event = messaging.new_message('longitudinalPlan')
  wire = event.longitudinalPlan.motionIntent
  wire.version = intent.version
  wire.mode = intent.mode
  wire.reasons = int(intent.reasons)
  wire.valid = False
  wire.ready = True
  wire.allowCreep = False
  wire.brakingValid = True
  wire.brakingAccel = intent.braking_accel
  assert MotionIntent.from_message(wire) == intent


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('changes', [
  {'version': 0}, {'version': 1}, {'version': 2}, {'version': MOTION_INTENT_VERSION + 1}, {'mode': 0}, {'mode': 3},
  {'reasons': 1 << 20}, {'reasons': StopReason.HOLD},
  {'ready': False}, {'ready': 1}, {'valid': True}, {'valid': 'false'},
  {'moving_lead': True}, {'release_allowed': True}, {'allow_creep': True},
  {'braking_accel': None}, {'braking_accel': True}, {'braking_accel': 'negative'},
  {'braking_accel': float('nan')}, {'braking_accel': float('inf')},
  {'braking_accel': -3.51}, {'braking_accel': 0.0}, {'braking_accel': 1.0},
])
def test_bad_or_mixed_envelope_cannot_promote_an_invalid_target_to_stronger_braking(mode, changes):
  harness = warm_controller(mode)
  intent = brake_intent(mode, **changes)
  assert not intent.is_braking_valid()
  assert harness.step(-3.5, intent, plan_valid=False, solver_status=4, braking_plan_valid=True) == -0.4
  assert harness.control.motion_degraded
  assert harness.control.motion_supervisor.degraded_braking_accel is None


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('target', [2.0, -3.5, float('nan'), float('inf')])
def test_solver_failure_uses_only_the_envelope_and_does_not_trust_the_ordinary_scalar(mode, target):
  harness = warm_controller(mode)
  supervisor = harness.control.motion_supervisor
  trusted = (supervisor._last_plan_time_ns, supervisor._last_model_time_ns, supervisor._last_intent, supervisor._last_target)
  assert harness.step(target, brake_intent(mode, braking_accel=-2.5), plan_valid=False, solver_status=4, braking_plan_valid=True) == -2.5
  assert harness.control.motion_degraded
  assert supervisor.degraded_braking_accel == -2.5
  assert trusted == (supervisor._last_plan_time_ns, supervisor._last_model_time_ns, supervisor._last_intent, supervisor._last_target)
  assert supervisor._require_new_input
  assert supervisor.release_lead_t == 0.0
  assert supervisor._last_braking_plan_time_ns == harness.plan_time_ns


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('solver_status,braking_valid', [
  (0, True), (4, False), (4, None), (4, 1), (4, 'true'), (True, True), ('4', True), (None, True),
])
def test_envelope_requires_explicit_health_gate_and_real_nonzero_solver_status(mode, solver_status, braking_valid):
  harness = warm_controller(mode)
  assert harness.step(-3.0, brake_intent(mode), plan_valid=False, solver_status=solver_status, braking_plan_valid=braking_valid) == -0.4
  assert harness.control.motion_degraded
  assert harness.control.motion_supervisor.degraded_braking_accel is None


@pytest.mark.parametrize('mode', MOTION_MODES)
def test_a_structurally_valid_envelope_from_the_other_stage_cannot_deepen_braking(mode):
  harness = warm_controller(mode)
  other_mode = LongitudinalMode.UNIFIED if mode == LongitudinalMode.SUPERVISED else LongitudinalMode.SUPERVISED
  intent = brake_intent(other_mode)
  assert intent.is_braking_valid()
  assert harness.step(-3.5, intent, plan_valid=False, solver_status=4, braking_plan_valid=True) == -0.4
  assert harness.control.motion_degraded
  assert harness.control.motion_supervisor.degraded_braking_accel is None


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('fault', [
  'old_normal', 'old_braking', 'same_braking', 'same_target_changed', 'same_model_changed',
  'new_plan_old_model', 'future', 'stale',
])
def test_normal_and_braking_identity_histories_jointly_reject_replay_or_rewritten_packets(mode, fault):
  harness = warm_controller(mode)
  trusted_time = harness.plan_time_ns
  accepted = brake_intent(mode, braking_accel=-2.0)
  assert harness.step(-0.5, accepted, plan_valid=False, solver_status=4, braking_plan_valid=True) == -2.0
  braking_time = harness.plan_time_ns
  intent, target = brake_intent(mode), -0.5
  timestamp_args = {}
  if fault == 'old_normal':
    timestamp_args = {'plan_time_ns': trusted_time - 1, 'model_time_ns': trusted_time - 1}
  elif fault == 'old_braking':
    timestamp_args = {'plan_time_ns': braking_time - 1, 'model_time_ns': braking_time - 1}
  elif fault in ('same_braking', 'same_target_changed', 'same_model_changed'):
    timestamp_args = {'plan_time_ns': braking_time, 'model_time_ns': braking_time}
    if fault == 'same_target_changed':
      intent, target = accepted, 2.0
    elif fault == 'same_model_changed':
      timestamp_args['model_time_ns'] -= 1
  elif fault == 'new_plan_old_model':
    timestamp_args = {'model_time_ns': trusted_time}
  elif fault == 'future':
    timestamp_args = {'plan_time_ns': harness.now_ns + PLAN_LEASE_NS, 'model_time_ns': braking_time}
  elif fault == 'stale':
    timestamp_args = {'model_time_ns': harness.now_ns - 2 * PLAN_LEASE_NS}
  assert harness.step(target, intent, plan_valid=False, solver_status=4, braking_plan_valid=True, **timestamp_args) == -2.0
  assert harness.control.motion_supervisor.degraded_braking_accel is None
  assert harness.control.motion_supervisor._last_braking_plan_time_ns == braking_time
  assert harness.control.motion_supervisor._last_plan_time_ns == trusted_time


@pytest.mark.parametrize('mode', MOTION_MODES)
def test_a_new_braking_payload_cannot_rewrite_an_already_accepted_normal_identity(mode):
  harness = warm_controller(mode)
  assert harness.step(-3.5, brake_intent(mode), new_input=False, plan_valid=False, solver_status=4, braking_plan_valid=True) == -0.4
  assert harness.control.motion_supervisor.degraded_braking_accel is None
  assert harness.control.motion_supervisor._last_braking_plan_time_ns is None


@pytest.mark.parametrize('mode', MOTION_MODES)
def test_identical_braking_samples_are_repeatable_but_cannot_rearm_normal_motion(mode):
  harness = warm_controller(mode)
  envelope = brake_intent(mode)
  assert harness.step(float('nan'), envelope, plan_valid=False, solver_status=4, braking_plan_valid=True) == -3.0
  supervisor = harness.control.motion_supervisor
  trusted_time = supervisor._last_plan_time_ns
  for _ in range(5):
    assert harness.step(float('nan'), envelope, new_input=False, plan_valid=False, solver_status=4, braking_plan_valid=True) == -3.0
    assert supervisor.degraded_braking_accel == -3.0
    assert supervisor._last_plan_time_ns == trusted_time
    assert supervisor._require_new_input
  assert harness.step(1.0, harness.normal_intent(new_input=False, release_allowed=True), new_input=False) == -3.0
  assert supervisor.degraded
  assert supervisor._last_plan_time_ns == trusted_time
  assert not supervisor.release_lead_t
  fresh = harness.step(1.0, harness.normal_intent())
  assert not supervisor.degraded
  assert -3.0 <= fresh <= -2.92 + 1e-6


@pytest.mark.parametrize('mode', MOTION_MODES)
def test_braking_watermark_survives_disable_and_rejects_an_older_normal_plan(mode):
  harness = warm_controller(mode)
  trusted_time = harness.plan_time_ns
  harness.step(-0.5, brake_intent(mode), plan_valid=False, solver_status=4, braking_plan_valid=True)
  brake_time = harness.plan_time_ns
  assert harness.step(active=False) == 0.0
  assert not harness.control.motion_owner_established
  assert harness.control.motion_supervisor._last_braking_plan_time_ns == brake_time
  assert harness.step(1.0, harness.normal_intent(timestamp=trusted_time + 1), plan_time_ns=trusted_time + 1,
                      model_time_ns=trusted_time + 1) <= DEGRADED_ACCEL
  assert harness.control.motion_degraded


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('field', ['vEgo', 'aEgo'])
def test_invalid_vehicle_values_cannot_enable_the_independent_envelope(mode, field):
  harness = warm_controller(mode)
  setattr(harness.state, field, float('nan'))
  assert harness.step(-3.0, brake_intent(mode), plan_valid=False, solver_status=4, braking_plan_valid=True) == -0.4
  assert harness.control.motion_supervisor.degraded_braking_accel is None


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('authority', ['gas', 'brake', 'disabled'])
def test_driver_authority_beats_an_otherwise_qualified_envelope(mode, authority):
  harness = warm_controller(mode)
  if authority != 'disabled':
    setattr(harness.state, authority + 'Pressed', True)
  assert harness.step(-3.0, brake_intent(mode), active=authority != 'disabled', plan_valid=False,
                      solver_status=4, braking_plan_valid=True) == 0.0
  assert harness.control.motion_phase == MotionPhase.OFF
  assert harness.control.long_control_state == LongCtrlState.off
  assert harness.control.motion_supervisor.degraded_braking_accel is None


@pytest.mark.parametrize('mode', MOTION_MODES)
def test_a_healthy_invalid_solver_plan_can_preserve_its_independent_braking_on_native_can(mode):
  harness = warm_controls(mode, 'track')
  wire_brake(harness)
  command, motion = harness.step()
  output, decoded = native_command(harness)
  assert not harness.control.sm.valid['longitudinalPlan']
  assert command.longActive
  assert motion.degraded
  assert command.actuators.accel == output.accel == pytest.approx(-3.0)
  assert decoded['ACCEL_COMMAND'] == pytest.approx(-3.0)
  assert decoded['BRAKE_REQUEST']
  assert decoded['GAS_COMMAND'] <= 0
  assert not command.cruiseControl.resume


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('health,service', [
  ('valid', 'carState'), ('seen', 'carState'), ('alive', 'carState'), ('freq_ok', 'carState'),
  ('seen', 'longitudinalPlan'), ('alive', 'longitudinalPlan'), ('freq_ok', 'longitudinalPlan'),
])
def test_controlsd_uses_actual_input_health_not_only_the_producer_braking_flag(mode, health, service):
  harness = warm_controls(mode, 'track')
  wire_brake(harness)
  original_update = harness.control.sm.update_msgs

  def unhealthy_update(*arguments, **keywords):
    original_update(*arguments, **keywords)
    getattr(harness.control.sm, health)[service] = False

  with patch.object(harness.control.sm, 'update_msgs', side_effect=unhealthy_update):
    command, motion = harness.step()
  assert motion.degraded
  assert command.actuators.accel == pytest.approx(-0.4)
  assert harness.control.LoC.motion_supervisor.degraded_braking_accel is None


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('health,service', [
  ('valid', 'carState'), ('seen', 'carState'), ('alive', 'carState'), ('freq_ok', 'carState'),
  ('seen', 'longitudinalPlan'), ('alive', 'longitudinalPlan'), ('freq_ok', 'longitudinalPlan'),
])
@pytest.mark.parametrize('value', [
  pytest.param(MISSING_HEALTH, id='missing'), None, 1, 1.0, 'true', True,
])
def test_independent_braking_requires_known_boolean_health_when_the_normal_plan_is_unavailable(mode, health, service, value):
  harness = warm_controls(mode, 'track')
  wire_brake(harness)
  original_update = harness.control.sm.update_msgs

  def update_health(*arguments, **keywords):
    original_update(*arguments, **keywords)
    values = getattr(harness.control.sm, health)
    if value is MISSING_HEALTH:
      values.pop(service)
    else:
      values[service] = value

  with patch.object(harness.control.sm, 'update_msgs', side_effect=update_health), \
       patch.object(harness.control.sm, 'all_checks', return_value=False):
    command, motion = harness.step()
  assert command.longActive
  assert motion.degraded
  if value is True:
    assert command.actuators.accel == pytest.approx(-3.0)
    assert harness.control.LoC.motion_supervisor.degraded_braking_accel == -3.0
  else:
    assert command.actuators.accel == pytest.approx(-0.4)
    assert harness.control.LoC.motion_supervisor.degraded_braking_accel is None
  assert not command.cruiseControl.resume


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('field,value', [('canValid', False), ('vCruise', V_CRUISE_UNSET), ('vCruise', float('nan'))])
def test_controlsd_checks_payload_vehicle_health_before_consuming_an_envelope(mode, field, value):
  harness = warm_controls(mode, 'track')
  wire_brake(harness)
  setattr(harness.car_state, field, value)
  command, motion = harness.step()
  assert command.actuators.accel == pytest.approx(-0.4)
  assert motion.degraded
  assert harness.control.LoC.motion_supervisor.degraded_braking_accel is None


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('reason', [StopReason.CRUISE, StopReason.FORCE_DECEL])
@pytest.mark.parametrize('speed', [2.0, 10.0, 20.0])
@pytest.mark.parametrize('target', [1.0, -3.0])
def test_accepted_raw_stop_request_cannot_leave_positive_gas_at_tracking_speed(mode, reason, speed, target):
  harness = warm_controls(mode, 'track')
  harness.car_state.vEgo = speed
  harness.plan.motionIntent.reasons = int(reason)
  harness.plan.aTarget = target
  command, motion = harness.step()
  output, decoded = native_command(harness)
  assert command.longActive
  assert not motion.degraded
  assert motion.reasons & int(reason)
  assert command.actuators.accel <= min(target, CarControllerParams.BOSCH_GAS_LOOKUP_BP[0])
  assert output.accel <= min(target, CarControllerParams.BOSCH_GAS_LOOKUP_BP[0])
  assert decoded['GAS_COMMAND'] <= 0
  assert not command.cruiseControl.resume


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('origin', ['warm_hold', 'fault_only_hold'])
@pytest.mark.parametrize('recovery', ['both_advance', 'same_braking_model', 'same_identity', 'mode_mismatch'])
def test_qualified_braking_watermarks_require_two_fresh_identities_and_keep_an_owned_hold(mode, origin, recovery):
  harness = mode_controls(mode)
  if origin == 'warm_hold':
    harness.hold()
  else:
    harness.car_state.vEgo = 0.0
    harness.car_state.standstill = True
    harness.car_state.cruiseState.standstill = True
    harness.plan.motionIntent.ready = False
    harness.plan.motionIntent.valid = False
    harness.step()
  for tick in range(1, 11):
    harness.step(new_plan=tick % 5 == 0)
  assert harness.control.sm.freq_ok['longitudinalPlan']
  for _ in range(4):
    harness.step(new_plan=False)
  wire_brake(harness, -2.5)
  harness.step()
  supervisor = harness.control.LoC.motion_supervisor
  braking_plan_time = int(harness.control.sm.logMonoTime['longitudinalPlan'])
  braking_model_time = int(harness.plan.modelMonoTime)
  assert supervisor._last_braking_plan_time_ns == braking_plan_time
  assert supervisor._last_braking_model_time_ns == braking_model_time
  assert harness.command.actuators.accel <= -2.5
  if origin == 'fault_only_hold':
    assert supervisor._last_plan_time_ns is None
  for _ in range(4):
    harness.step(new_plan=False)
  if recovery == 'mode_mismatch':
    harness.plan.motionIntent.version = 1
    command, motion = harness.step()
  else:
    harness.plan.motionIntent.valid = True
    harness.plan.motionIntent.brakingValid = False
    harness.plan.motionIntent.releaseAllowed = True
    harness.plan.motionIntent.reasons = 0
    harness.plan.aTarget = 1.0
    harness.plan.solverStatus = 0
    harness.events['longitudinalPlan'].valid = True
    timestamps = {}
    if recovery == 'same_braking_model':
      timestamps['model_time_ns'] = braking_model_time
    elif recovery == 'same_identity':
      timestamps = {'plan_time_ns': braking_plan_time, 'model_time_ns': braking_model_time}
    command, motion = harness.step(**timestamps)
  assert harness.control.sm.freq_ok['longitudinalPlan']
  assert command.longActive
  output, decoded = native_command(harness)
  if recovery == 'both_advance':
    assert not motion.degraded
    assert command.cruiseControl.resume
    assert motion.phase == 'release'
  else:
    assert motion.degraded
    assert not command.cruiseControl.resume
    assert command.actuators.accel <= -2.5
    assert output.accel <= -2.5
    assert decoded['BRAKE_REQUEST']
    assert decoded['GAS_COMMAND'] <= 0
    assert motion.phase == 'hold'
    assert harness.control.LoC.motion_owner_established
  harness.selfdrive_state.enabled = False
  command, motion = harness.step()
  output, decoded = native_command(harness)
  assert not command.longActive
  assert command.actuators.accel == output.accel == decoded['ACCEL_COMMAND'] == 0.0
  assert not decoded['BRAKE_REQUEST']
  assert decoded['GAS_COMMAND'] <= 0
  assert not command.cruiseControl.resume
  assert motion.phase == 'off'
  assert not harness.control.LoC.motion_owner_established


def e2e_actuation_arm(mode, speed, model_request):
  supervised, supervised_owner = make_supervised(speed=speed, initial_accel=model_request)
  if mode == LongitudinalMode.SUPERVISED:
    planner, owner = supervised, supervised_owner
  else:
    planner = LongitudinalPlanner(supervised.CP, supervised_owner.CP_SP, init_v=speed, init_a=model_request,
                                  unified=False, supervised=False)
    owner = LongControl(supervised.CP, supervised_owner.CP_SP)
  state = make_supervised_state(speed=speed, target=speed + 10.0, experimental=True)
  dbc_name = DBC[planner.CP.carFingerprint][Bus.pt]
  native = CarController({Bus.pt: dbc_name}, planner.CP, owner.CP_SP)
  parser = CANParser(dbc_name, [('ACC_CONTROL', 50)], CanBus(planner.CP).pt)
  vehicle = SimpleNamespace(out=state['carState'], v_cruise_factor=1.0, is_metric=True,
                            acc_hud={}, lkas_hud={}, stock_brake={})
  rows = []
  for frame in range(50):
    state['modelV2'].action.shouldStop = 10 <= frame < 30
    state['modelV2'].action.desiredAcceleration = model_request if frame < 30 else 1.0
    state['controlsState'].longControlState = owner.long_control_state
    timestamp = advance(state, frame)
    planner.update(state)
    publisher = Publisher()
    planner.publish(state, publisher)
    published = publisher.messages['longitudinalPlan'].as_builder()
    published.logMonoTime = timestamp + 1_000_000
    event = serialized(published)
    plan = event.longitudinalPlan
    for tick in range(5):
      now_ns = timestamp + 2_000_000 + tick * 10_000_000
      output = owner.update(
        True, state['carState'], plan.aTarget, plan.shouldStop, (-3.5, 2.0),
        intent=MotionIntent.from_message(plan.motionIntent), plan_time_ns=event.logMonoTime,
        model_time_ns=plan.modelMonoTime, now_ns=now_ns, plan_valid=event.valid, solver_status=int(plan.solverStatus),
      )
      command = car.CarControl.new_message()
      command.enabled = True
      command.longActive = True
      command.actuators.accel = float(output)
      command.actuators.longControlState = owner.long_control_state
      native_output, packets = native.update(command.as_reader(), structs.CarControlSP(), vehicle, now_ns)
      decoded = None
      if native.frame % 2 == 1:
        parser.update([(now_ns, packets)])
        decoded = dict(parser.vl['ACC_CONTROL'])
      rows.append({
        'frame': frame, 'tick': tick, 'target': float(plan.aTarget), 'plan_valid': bool(event.valid),
        'solver_status': int(plan.solverStatus), 'raw_reasons': int(plan.motionIntent.reasons),
        'owned_reasons': int(owner.motion_reasons), 'request': float(output), 'applied': float(native_output.accel),
        'state': owner.long_control_state, 'phase': owner.motion_phase, 'degraded': owner.motion_degraded,
        'resolved_stop': owner.resolved_should_stop, 'can': decoded,
      })
  return rows


@pytest.mark.parametrize('speed', [1.99, 2.0, 2.01, 5.0, 20.0])
@pytest.mark.parametrize('model_request', [-3.0, -0.5, -0.1, 0.3])
def test_supervised_e2e_stop_preserves_legacy_braking_with_20_100_50_hz_execution_and_releases_on_withdrawal(speed, model_request):
  legacy = e2e_actuation_arm(LongitudinalMode.LEGACY, speed, model_request)
  supervised = e2e_actuation_arm(LongitudinalMode.SUPERVISED, speed, model_request)
  assert len(legacy) == len(supervised) == 250
  stopping_can_frames = 0
  for reference, candidate in zip(legacy, supervised, strict=True):
    assert candidate['plan_valid'] and reference['plan_valid']
    assert candidate['solver_status'] == reference['solver_status'] == 0
    assert not candidate['degraded']
    if 10 <= candidate['frame'] < 30:
      assert candidate['target'] == reference['target']
      assert candidate['raw_reasons'] & int(StopReason.E2E)
      assert candidate['state'] == LongCtrlState.stopping
      assert candidate['request'] <= reference['request'] + 1e-6
      assert candidate['request'] <= candidate['target'] + 1e-6
      assert candidate['applied'] <= reference['applied'] + 1e-6
      if candidate['can'] is not None:
        stopping_can_frames += 1
        assert candidate['can']['GAS_COMMAND'] <= 0
        assert candidate['can']['ACCEL_COMMAND'] <= reference['can']['ACCEL_COMMAND'] + 1e-6
    elif candidate['frame'] >= 30:
      assert not candidate['raw_reasons'] & int(StopReason.E2E)
      assert not candidate['owned_reasons'] & int(StopReason.E2E)
      assert not candidate['resolved_stop']
  assert stopping_can_frames == 50
  assert supervised[-1]['phase'] == MotionPhase.TRACK
  assert supervised[-1]['state'] == LongCtrlState.pid


def test_unified_high_speed_e2e_execution_is_not_changed_by_the_supervised_compatibility_rule():
  harness = warm_controller(LongitudinalMode.UNIFIED)
  assert harness.step(-0.5, MotionIntent(reasons=StopReason.E2E)) == -0.5
  assert harness.control.long_control_state == LongCtrlState.pid
  assert harness.control.motion_phase == MotionPhase.APPROACH


@pytest.mark.parametrize('authority', ['gas', 'brake', 'disabled'])
def test_driver_cancellation_wins_over_the_supervised_high_speed_e2e_stop_ramp(authority):
  harness = warm_controller(LongitudinalMode.SUPERVISED)
  intent = harness.normal_intent(reasons=StopReason.E2E)
  harness.step(-0.5, intent)
  assert harness.control.long_control_state == LongCtrlState.stopping
  if authority != 'disabled':
    setattr(harness.state, authority + 'Pressed', True)
  assert harness.step(-3.0, intent, active=authority != 'disabled') == 0.0
  assert harness.control.long_control_state == LongCtrlState.off
  assert harness.control.motion_phase == MotionPhase.OFF
  assert not harness.control.motion_reasons
