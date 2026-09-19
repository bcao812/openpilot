from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.hondacan import CanBus
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC
from opendbc.car.structs import car
from openpilot.cereal import custom, log
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.car.card import Car
from openpilot.selfdrive.car.car_events import CarEvents
from openpilot.selfdrive.car.cruise import VCruiseHelper, V_CRUISE_INITIAL, V_CRUISE_UNSET
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import (
  CapturingPublisher, ControlsHarness, bounded_submaster, new_event, selfdrive_event_harness, serialized,
)
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_state
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.selfdrived.events import ET
from openpilot.selfdrive.selfdrived.state import StateMachine
from openpilot.sunnypilot.selfdrive.car.cruise_helpers import CruiseHelper


EventName = log.OnroadEvent.EventName
ButtonType = car.CarState.ButtonEvent.Type
State = log.SelfdriveState.OpenpilotState
CONTROL_STEP_NS = round(DT_CTRL * 1e9)
PLANNER_STEPS = round(DT_MDL / DT_CTRL)


class StartupHarness:
  """Execute the longitudinal startup chain; transport, sensors and lateral control remain bounded fixtures."""

  def __init__(self, *, platform=CAR.HONDA_ACCORD, openpilot_long=True, interceptor=False, speed=5.0):
    self.now_ns = 10_000_000_000
    self.frame = 0
    self.history = []
    self.native_commands = []
    self.model_stop = False
    self.source_state = car.CarState.new_message()
    self.source_state.canValid = True
    self.source_state.vEgo = speed
    self.source_state.aEgo = 0.0
    self.source_state.standstill = speed == 0.0
    self.source_state.gearShifter = car.CarState.GearShifter.drive
    self.source_state.cruiseState.available = True
    self.source_state.cruiseState.enabled = False
    self.source_state.cruiseState.standstill = speed == 0.0

    self.CP = CarInterface.get_non_essential_params(platform)
    self.CP.openpilotLongitudinalControl = openpilot_long
    self.CP.pcmCruise = False
    self.CP.passive = False
    self.CP.lateralTuning.init('pid')
    self.CP_SP = CarInterface.get_non_essential_params_sp(self.CP, platform)
    self.CP_SP.enableGasInterceptor = interceptor
    base = ControlsHarness(platform=platform, openpilot_long=openpilot_long, interceptor=interceptor)
    self.controls = base.control
    self.controls.CP = self.CP
    self.controls.CP_SP = self.CP_SP
    self.controls.LoC = LongControl(self.CP, self.CP_SP)

    self.latest = {
      service: new_event(service, self.now_ns)
      for service in ('carControl', 'carControlSP', 'controlsState', 'selfdriveState', 'selfdriveStateSP',
                      'longitudinalPlan', 'longitudinalPlanSP', 'onroadEvents')
    }
    self.latest['controlsState'].controlsState.lateralControlState.init('pidState')
    self.latest['longitudinalPlan'].valid = False
    self.latest['selfdriveState'].selfdriveState.personality = log.LongitudinalPersonality.standard

    self.selfdrive, _ = selfdrive_event_harness(serialized(self.latest['controlsState']))
    self.selfdrive.CP = self.CP
    self.selfdrive.cp_olc = True
    self.selfdrive.cp_pcm_cruise = False
    self.selfdrive.enabled = False
    self.selfdrive.active = False
    self.selfdrive.state_machine = StateMachine()
    self.selfdrive.car_events = CarEvents(self.CP)
    self.selfdrive.CS_prev = self.source_state.as_reader().as_builder()
    self.selfdrive.sm.ignore_valid.extend(['longitudinalPlanSP', 'modelDataV2SP', 'lateralManeuverPlan', 'alertDebug'])
    self.selfdrive.sm.ignore_alive.extend(['longitudinalPlanSP', 'modelDataV2SP', 'lateralManeuverPlan', 'alertDebug'])
    CruiseHelper.__init__(self.selfdrive, self.CP)

    templates = make_state(speed=speed)
    templates['vehicleParameters'].valid = True
    self.planner_state = bounded_submaster(list(templates))
    for service, payload in templates.items():
      self.planner_state.data[service] = payload.as_reader() if hasattr(payload, 'as_reader') else payload
    self.restart_planner()

    self.card = Car.__new__(Car)
    self.card.CP = self.CP
    self.card.CP_SP = self.CP_SP
    self.card.CP_SP_capnp = convert_to_capnp(self.CP_SP)
    self.card.CC_prev = car.CarControl.new_message()
    self.card.CS_prev = self.source_state.as_reader().as_builder()
    self.card.CS_SP_prev = custom.CarStateSP.new_message()
    self.card.prof = None
    self.card.can_sock = object()
    self.card.can_rcv_cum_timeout_counter = 0
    self.card.initialized_prev = True
    self.card.is_metric = True
    self.card.experimental_mode = False
    self.card.dynamic_experimental_control = False
    self.card.sm = bounded_submaster(['carControl', 'carControlSP', 'onroadEvents', 'longitudinalPlanSP'])
    self.card.pm = CapturingPublisher()
    self.card.rk = SimpleNamespace(remaining=0.0)
    self.card.last_actuators_output = car.CarControl.Actuators.new_message()
    self.card.v_cruise_helper = VCruiseHelper(self.CP, self.CP_SP)
    self.card.v_cruise_helper.enable_button_timers = dict.fromkeys(self.card.v_cruise_helper.enable_button_timers, 0)
    self.vehicle = SimpleNamespace(v_cruise_factor=1.0, is_metric=True, acc_hud={}, lkas_hud={}, stock_brake={})
    dbc_name = DBC[self.CP.carFingerprint][Bus.pt]
    self.native_controller = CarController({Bus.pt: dbc_name}, self.CP, self.CP_SP)
    self.can_parser = CANParser(dbc_name, [('ACC_CONTROL', 50)], CanBus(self.CP).pt)
    self.card.CI = SimpleNamespace(update=self.receive_state, apply=self.apply_native)
    self.card.RI = SimpleNamespace(update=lambda _: None)

  def restart_planner(self):
    # Match plannerd's production construction, including its default init_v.
    self.planner = LongitudinalPlanner(self.CP, self.CP_SP)
    self.mpc_update = Mock(wraps=self.planner.mpc.update)
    self.planner.mpc.update = self.mpc_update
    self.planner_publisher = CapturingPublisher()

  def receive_state(self, packets):
    state = self.source_state.as_reader().as_builder()
    self.vehicle.out = state
    return state, structs.CarStateSP()

  def apply_native(self, command, extension, timestamp):
    output, packets = self.native_controller.update(command, extension, self.vehicle, timestamp)
    self.can_parser.update([(timestamp, packets)])
    if any(address == 0x1df for address, _, _ in packets):
      self.native_commands.append({
        'mono': timestamp, 'v_cruise': self.vehicle.out.vCruise,
        'long_active': bool(command.longActive), 'resume': bool(command.cruiseControl.resume),
        **dict(self.can_parser.vl['ACC_CONTROL']),
      })
    return output, packets

  def feed(self, state, overrides):
    incoming = []
    for service in state.services:
      event = overrides.get(service)
      if event is None:
        if service in ('lateralManeuverPlan', 'alertDebug', 'userBookmark'):
          continue
        event = new_event(service, self.now_ns)
        if service != 'pandaStates':
          setattr(event, service, state[service])
      incoming.append(serialized(event.as_builder() if hasattr(event, 'as_builder') else event))
    state.update_msgs(self.now_ns / 1e9, incoming)
    state.updated = dict.fromkeys(state.services, False)

  def model_event(self):
    event = new_event('modelV2', self.now_ns)
    event.modelV2 = self.planner_state['modelV2']
    model = event.modelV2
    speed = self.source_state.vEgo
    model.position.x = [float(speed * timestamp) for timestamp in ModelConstants.T_IDXS]
    model.velocity.x = [float(speed)] * len(ModelConstants.T_IDXS)
    model.acceleration.x = [0.0] * len(ModelConstants.T_IDXS)
    model.action.shouldStop = self.model_stop
    model.action.desiredAcceleration = -2.0 if self.model_stop else 0.5
    return event

  def step(self, *, button=None, run_planner=True, override_plan=None, planner_overrides=None):
    self.now_ns += CONTROL_STEP_NS
    self.frame += 1
    self.source_state.buttonEnable = button == 'set_release'
    self.source_state.buttonEvents = []
    if button is not None:
      events = self.source_state.init('buttonEvents', 1)
      events[0].type = ButtonType.cancel if button == 'cancel' else ButtonType.decelCruise
      events[0].pressed = button in ('set_press', 'cancel')
    with patch('time.monotonic_ns', return_value=self.now_ns), patch('time.monotonic', return_value=self.now_ns / 1e9), \
         patch('openpilot.selfdrive.car.card.messaging.drain_sock_raw', return_value=[]):
      self.feed(self.card.sm, self.latest)
      state, extension, radar = self.card.state_update()
      self.card.state_publish(state, extension, radar)
      self.card.controls_update(state, self.card.sm['carControl'], self.card.sm['carControlSP'])
      self.card.CS_prev = state.as_reader()
      self.latest.update(self.card.pm.messages)
      self.latest['modelV2'] = self.model_event()

      if override_plan is not None:
        override_plan.logMonoTime = self.now_ns
        override_plan.longitudinalPlan.modelMonoTime = self.now_ns
        self.latest['longitudinalPlan'] = serialized(override_plan)
      self.feed(self.selfdrive.sm, self.latest)
      self.selfdrive.update_events(state)
      self.selfdrive.enabled, self.selfdrive.active = self.selfdrive.state_machine.update(self.selfdrive.events)
      self.selfdrive.CS_prev = state.as_reader()
      selfdrive_event = new_event('selfdriveState', self.now_ns)
      selfdrive_event.selfdriveState.enabled = self.selfdrive.enabled
      selfdrive_event.selfdriveState.active = self.selfdrive.active
      selfdrive_event.selfdriveState.state = self.selfdrive.state_machine.state
      selfdrive_event.selfdriveState.experimentalMode = self.selfdrive.experimental_mode
      selfdrive_event.selfdriveState.engageable = not self.selfdrive.events.contains(ET.NO_ENTRY)
      selfdrive_event.selfdriveState.personality = log.LongitudinalPersonality.standard
      onroad_events = new_event('onroadEvents', self.now_ns)
      onroad_events.onroadEvents = self.selfdrive.events.to_msg()
      self.latest.update(selfdriveState=serialized(selfdrive_event), onroadEvents=serialized(onroad_events))

      consumed_plan = self.latest['longitudinalPlan']
      self.feed(self.controls.sm, self.latest)
      command, lateral_log = self.controls.state_control()
      self.controls.publish(command, lateral_log)
      self.latest.update(self.controls.pm.messages)
      if run_planner and self.frame % PLANNER_STEPS == 0:
        self.feed(self.planner_state, self.latest | (planner_overrides or {}))
        self.planner.update(self.planner_state)
        self.planner.publish(self.planner_state, self.planner_publisher)
        self.latest.update(self.planner_publisher.messages)

    row = {
      'mono': self.now_ns, 'v_cruise': state.vCruise,
      'enabled': bool(command.enabled), 'long_active': bool(command.longActive),
      'resume': bool(command.cruiseControl.resume), 'accel': float(command.actuators.accel),
      'phase': str(self.latest['controlsState'].controlsState.motionState.phase),
      'degraded': bool(self.latest['controlsState'].controlsState.motionState.degraded),
      'no_entry': self.selfdrive.events.contains(ET.NO_ENTRY),
      'comm_issue': EventName.commIssue in self.selfdrive.events.names,
      'consumed_plan_mono': int(consumed_plan.logMonoTime),
      'consumed_ready': consumed_plan.longitudinalPlan.motionIntent.to_dict().get('ready', True),
      'plan_valid': bool(self.latest['longitudinalPlan'].valid),
    }
    self.history.append(row)
    return row

  def boot(self):
    for _ in range(12):
      self.step()
    assert self.card.v_cruise_helper.v_cruise_kph == V_CRUISE_UNSET

  def first_set(self):
    self.step(button='set_press')
    return self.step(button='set_release')

  def engage(self):
    self.boot()
    requested = self.first_set()
    assert requested['enabled'], self.history[-3:]
    initialized = self.step()
    assert initialized['v_cruise'] == V_CRUISE_INITIAL
    for _ in range(15):
      row = self.step()
      if row['long_active']:
        return row
    raise AssertionError(self.history[-5:])

  def hold(self):
    self.engage()
    self.source_state.vEgo = 0.0
    self.source_state.standstill = True
    self.source_state.cruiseState.standstill = True
    self.model_stop = True
    self.selfdrive.experimental_mode = True
    for _ in range(12):
      self.step()
    assert self.controls.LoC.resolved_should_stop
    assert self.history[-1]['phase'] == 'hold'
    assert not self.controls.LoC.can_standby


@pytest.mark.parametrize('unified', [False, True])
def test_first_set_reaches_actual_card_initialization_without_an_entry_deadlock(unified):
  harness = StartupHarness(platform=CAR.HONDA_ACCORD if unified else CAR.HONDA_CIVIC_BOSCH)
  harness.boot()
  requested = harness.first_set()
  assert requested['enabled'], harness.history[-3:]
  assert not requested['no_entry']
  assert requested['v_cruise'] == V_CRUISE_UNSET
  initialized = harness.step()
  assert initialized['v_cruise'] == V_CRUISE_INITIAL
  assert harness.card.v_cruise_helper.v_cruise_initialized


@pytest.mark.parametrize('speed', [0.0, 5.0])
def test_healthy_cold_unset_speed_is_standby_not_a_placeholder_mpc_trajectory(speed):
  harness = StartupHarness(speed=speed)
  harness.boot()
  event = harness.latest['longitudinalPlan']
  assert event.valid
  assert not event.longitudinalPlan.motionIntent.valid
  assert not event.longitudinalPlan.motionIntent.ready
  assert harness.mpc_update.call_count == 0
  assert not harness.history[-1]['comm_issue']
  assert not harness.history[-1]['no_entry']
  assert harness.controls.LoC.long_control_state == LongCtrlState.off
  assert harness.native_commands
  assert all(command['GAS_COMMAND'] <= 0 and not command['resume'] and not command['long_active']
             for command in harness.native_commands)


def test_set_and_card_initialization_do_not_authorize_motion_before_a_new_ready_plan():
  harness = StartupHarness()
  harness.boot()
  standby_id = harness.latest['longitudinalPlan'].logMonoTime
  requested = harness.first_set()
  assert requested['enabled']
  assert not requested['long_active']
  assert not requested['resume']
  assert harness.controls.LoC.long_control_state == LongCtrlState.off
  initialized = harness.step(run_planner=False)
  assert initialized['v_cruise'] == V_CRUISE_INITIAL
  assert initialized['consumed_plan_mono'] == standby_id
  for _ in range(3):
    waiting = harness.step(run_planner=False)
    assert waiting['enabled']
    assert not waiting['long_active']
    assert not waiting['resume']
    assert waiting['accel'] == 0.0
    assert harness.controls.LoC.long_control_state == LongCtrlState.off
  assert all(command['GAS_COMMAND'] <= 0 and not command['resume'] and not command['long_active']
             for command in harness.native_commands)
  while harness.frame % PLANNER_STEPS != 0:
    harness.step()
  fresh_id = harness.latest['longitudinalPlan'].logMonoTime
  assert fresh_id > initialized['mono']
  assert harness.latest['longitudinalPlan'].longitudinalPlan.motionIntent.ready
  assert harness.latest['longitudinalPlan'].longitudinalPlan.motionIntent.valid
  running = harness.step()
  assert running['consumed_plan_mono'] == fresh_id
  assert running['long_active']
  assert not running['degraded']
  assert harness.mpc_update.call_count > 0


def test_losing_initialized_set_speed_is_a_fault_not_a_return_to_cold_standby():
  harness = StartupHarness()
  harness.engage()
  assert not harness.controls.LoC.can_standby
  harness.card.v_cruise_helper.v_cruise_kph = V_CRUISE_UNSET
  harness.card.v_cruise_helper.v_cruise_cluster_kph = V_CRUISE_UNSET
  previous_calls = harness.mpc_update.call_count
  for _ in range(8):
    row = harness.step()
  event = harness.latest['longitudinalPlan']
  assert not event.valid
  assert event.longitudinalPlan.motionIntent.ready
  assert not event.longitudinalPlan.motionIntent.valid
  assert row['degraded']
  assert row['comm_issue']
  assert not row['resume']
  assert harness.mpc_update.call_count == previous_calls


def test_ready_false_cannot_release_an_existing_hold():
  harness = StartupHarness()
  harness.hold()
  standby = harness.latest['longitudinalPlan'].as_builder()
  standby.valid = True
  standby.longitudinalPlan.motionIntent.valid = False
  standby.longitudinalPlan.motionIntent.ready = False
  standby.longitudinalPlan.motionIntent.reasons = 0
  standby.longitudinalPlan.aTarget = 1.0
  row = harness.step(run_planner=False, override_plan=standby)
  assert row['long_active']
  assert row['degraded']
  assert row['phase'] == 'hold'
  assert row['accel'] <= -2.0
  assert not row['resume']
  assert harness.controls.LoC.resolved_should_stop


def test_real_warm_hold_setpoint_loss_keeps_pressure_and_publishes_a_fault():
  harness = StartupHarness()
  harness.hold()
  harness.card.v_cruise_helper.v_cruise_kph = V_CRUISE_UNSET
  harness.card.v_cruise_helper.v_cruise_cluster_kph = V_CRUISE_UNSET
  commands_before = len(harness.native_commands)
  for _ in range(8):
    row = harness.step()
    assert row['phase'] == 'hold'
    assert row['degraded']
    assert row['accel'] <= -2.0
    assert not row['resume']
  event = harness.latest['longitudinalPlan']
  assert not event.valid
  assert event.longitudinalPlan.motionIntent.ready
  assert not event.longitudinalPlan.motionIntent.valid
  assert row['comm_issue']
  emitted = harness.native_commands[commands_before:]
  assert emitted
  assert all(command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0 and not command['resume'] for command in emitted)


def test_disabling_after_motion_does_not_erase_the_initialized_speed_failure():
  harness = StartupHarness()
  harness.engage()
  harness.step(button='cancel')
  assert not harness.history[-1]['enabled']
  harness.card.v_cruise_helper.v_cruise_kph = V_CRUISE_UNSET
  harness.card.v_cruise_helper.v_cruise_cluster_kph = V_CRUISE_UNSET
  for _ in range(8):
    row = harness.step()
  assert not harness.controls.LoC.can_standby
  assert not harness.latest['longitudinalPlan'].valid
  assert harness.latest['longitudinalPlan'].longitudinalPlan.motionIntent.ready
  assert row['no_entry']
  assert row['comm_issue']
  assert not row['long_active']
  assert not row['resume']


def test_invalid_default_set_speed_cannot_poison_first_set_initialization():
  harness = StartupHarness()
  invalid_state = new_event('carState', harness.now_ns + PLANNER_STEPS * CONTROL_STEP_NS)
  invalid_state.valid = False
  assert invalid_state.carState.vCruise == 0.0
  for _ in range(PLANNER_STEPS):
    harness.step(planner_overrides={'carState': invalid_state})
  assert not harness.planner_state.valid['carState']
  assert harness.planner_state['carState'].vCruise == 0.0
  assert not harness.latest['longitudinalPlan'].valid
  initialized_from_invalid = harness.planner.reference_planner.cruise_initialized
  harness.boot()
  assert harness.planner_state.all_checks()
  assert harness.planner_state['carState'].vCruise == V_CRUISE_UNSET
  recovered_plan = harness.latest['longitudinalPlan']
  requested = harness.first_set()
  assert requested['enabled'], {
    'initialized_from_invalid': initialized_from_invalid,
    'recovered_standby': harness.planner.reference_planner.standby,
    'recent_controls': harness.history[-3:],
  }
  assert not initialized_from_invalid
  assert recovered_plan.valid
  assert not recovered_plan.longitudinalPlan.motionIntent.valid
  assert not recovered_plan.longitudinalPlan.motionIntent.ready
  assert not requested['no_entry']
  assert not requested['comm_issue']
  assert not requested['long_active']
  assert not requested['resume']
  assert harness.controls.LoC.long_control_state == LongCtrlState.off
  assert harness.native_commands
  assert all(command['GAS_COMMAND'] <= 0 and not command['resume'] and not command['long_active']
             for command in harness.native_commands)
  initialized = harness.step()
  assert initialized['v_cruise'] == V_CRUISE_INITIAL
  assert harness.card.v_cruise_helper.v_cruise_initialized


@pytest.mark.parametrize('fault', ['unset', 'not_ready'])
def test_restarting_only_the_planner_cannot_turn_an_existing_hold_into_cold_standby(fault):
  harness = StartupHarness()
  harness.hold()
  previous_planner = harness.planner
  previous_supervisor = harness.controls.LoC.motion_supervisor
  harness.restart_planner()
  assert harness.planner is not previous_planner
  assert harness.controls.LoC.motion_supervisor is previous_supervisor
  assert not harness.controls.LoC.can_standby
  harness.card.v_cruise_helper.v_cruise_kph = V_CRUISE_UNSET
  harness.card.v_cruise_helper.v_cruise_cluster_kph = V_CRUISE_UNSET
  commands_before = len(harness.native_commands)
  for _ in range(PLANNER_STEPS):
    row = harness.step()
    assert row['long_active']
    assert row['phase'] == 'hold'
    assert row['degraded']
    assert row['accel'] <= -2.0
    assert not row['resume']
  produced = harness.latest['longitudinalPlan']
  assert not produced.valid
  assert not produced.longitudinalPlan.motionIntent.valid
  assert produced.longitudinalPlan.motionIntent.ready
  assert not harness.planner.reference_planner.standby
  assert not harness.planner.reference_planner.cold_standby_started
  assert harness.mpc_update.call_count == 0
  if fault == 'not_ready':
    standby = produced.as_builder()
    standby.valid = True
    standby.longitudinalPlan.motionIntent.ready = False
    standby.longitudinalPlan.motionIntent.reasons = 0
    standby.longitudinalPlan.aTarget = 1.0
    for _ in range(PLANNER_STEPS):
      row = harness.step(run_planner=False, override_plan=standby)
      assert not row['consumed_ready']
      assert row['long_active']
      assert row['phase'] == 'hold'
      assert row['degraded']
      assert row['accel'] <= -2.0
      assert not row['resume']
  assert row['comm_issue']
  assert harness.controls.LoC.motion_supervisor is previous_supervisor
  assert harness.controls.LoC.resolved_should_stop
  assert not harness.controls.LoC.can_standby
  emitted = harness.native_commands[commands_before:]
  assert emitted
  assert all(command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0 and not command['resume'] for command in emitted)


def test_restarting_only_the_planner_while_tracking_starts_from_actual_motion():
  harness = StartupHarness(speed=20.0)
  harness.boot()
  assert harness.first_set()['enabled']
  for _ in range(15):
    row = harness.step()
  assert row['long_active']
  harness.card.v_cruise_helper.v_cruise_kph = 54.0
  harness.card.v_cruise_helper.v_cruise_cluster_kph = 54.0
  for _ in range(PLANNER_STEPS):
    row = harness.step()
  assert row['long_active']
  assert row['phase'] == 'track'
  previous_supervisor = harness.controls.LoC.motion_supervisor
  harness.restart_planner()
  assert harness.planner.v_desired_filter.x == 0.0
  for _ in range(PLANNER_STEPS):
    row = harness.step()
    if harness.mpc_update.call_count:
      break
  assert harness.mpc_update.call_count == 1
  assert harness.controls.LoC.motion_supervisor is previous_supervisor
  assert harness.planner_state['controlsState'].longControlState == LongCtrlState.pid
  assert harness.planner.mpc.solution_status == 0
  assert harness.planner.mpc.x0[1] == pytest.approx(harness.source_state.vEgo)
  assert harness.planner.output_a_target < 0.0
  assert harness.latest['longitudinalPlan'].valid
  assert not row['comm_issue']
