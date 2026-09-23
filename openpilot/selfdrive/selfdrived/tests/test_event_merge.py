from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from opendbc.car.honda.values import CAR
from opendbc.car.structs import car
from openpilot.cereal import custom, log
from openpilot.selfdrive.car.car_events import CarEvents
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import (
  CapturingPublisher, new_event, selfdrive_event_harness, serialized,
)
from openpilot.selfdrive.selfdrived.alertmanager import AlertManager
from openpilot.selfdrive.selfdrived.events import Events, ET
from openpilot.selfdrive.selfdrived.state import StateMachine
from openpilot.sunnypilot.selfdrive.car.car_specific import CarSpecificEventsSP
from openpilot.sunnypilot.selfdrive.selfdrived.button_state_tracker import ButtonStateTracker
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP
from openpilot.sunnypilot.selfdrive.selfdrived.events_base import EventsBase


EventName = log.OnroadEvent.EventName
EventNameSP = custom.OnroadEventSP.EventName
State = log.SelfdriveState.OpenpilotState


def legacy_add_from(destination, source):
  destination.add_from_msg(source.to_msg())


def event_snapshot(events):
  return (events.names.copy(), events.static_events.copy(), events.event_counters.copy(),
          [event.to_dict() for event in events.to_msg()])


@pytest.mark.parametrize('event_class', [Events, EventsSP])
@pytest.mark.parametrize('empty_source,empty_destination', [(False, False), (True, False), (False, True), (True, True)])
def test_merge_preserves_all_mapped_flags_order_duplicates_and_ownership(event_class, empty_source, empty_destination):
  names = list(event_class().get_events_mapping())
  source = event_class()
  destination = event_class()
  if not empty_source:
    for name in reversed(names + names[:3]):
      source.add(name, static=True)
    source.event_counters = dict.fromkeys(names, 17)
  if not empty_destination:
    for name in names[::3]:
      destination.add(name, static=True)
    destination.event_counters = dict.fromkeys(names, 5)

  original_source = event_snapshot(source)
  previous_names = destination.names
  expected_names = sorted(destination.names + source.names)
  reference = deepcopy(destination)
  legacy_add_from(reference, source)
  destination.add_from(source)

  assert event_snapshot(destination) == event_snapshot(reference)
  assert destination.names == expected_names
  assert destination.names is previous_names
  assert event_snapshot(source) == original_source
  source.clear()
  source.add(names[-1])
  assert destination.names == expected_names


@pytest.mark.parametrize('event_class', [Events, EventsSP])
def test_merging_a_collection_into_itself_duplicates_once(event_class):
  events = event_class()
  for name in list(events.get_events_mapping())[:3]:
    events.add(name)
  reference = deepcopy(events)
  legacy_add_from(reference, reference)
  events.add_from(events)
  assert event_snapshot(events) == event_snapshot(reference)


@pytest.mark.parametrize('event_class,event_name,static_name', [
  (Events, EventName.reverseGear, EventName.dashcamMode),
  (EventsSP, EventNameSP.silentReverseGear, EventNameSP.manualLongitudinalRequired),
])
def test_merge_retains_receiver_lifetime_and_permanent_alert_delay(event_class, event_name, static_name):
  source = event_class()
  source.add(event_name, static=True)
  source.event_counters[event_name] = 200
  actual = event_class()
  actual.add(static_name, static=True)
  reference = deepcopy(actual)
  alert_type = f'{actual.get_event_name(event_name)}/{ET.PERMANENT}'

  for frame in range(53):
    actual.clear()
    reference.clear()
    if frame < 51:
      actual.add_from(source)
      legacy_add_from(reference, source)
    assert event_snapshot(actual) == event_snapshot(reference)
    alerts = [vars(alert).copy() for alert in actual.create_alerts([ET.PERMANENT])]
    assert alerts == [vars(alert).copy() for alert in reference.create_alerts([ET.PERMANENT])]
    assert any(alert['alert_type'] == alert_type for alert in alerts) == (49 <= frame < 51)
    assert actual.static_events == [static_name]

  assert actual.names == [static_name]
  assert actual.event_counters[event_name] == 0
  assert source.event_counters[event_name] == 200


def event_daemon(initial_state=State.enabled):
  control_state = new_event('controlsState')
  control_state.controlsState.lateralControlState.init('pidState')
  daemon, car_state = selfdrive_event_harness(control_state)
  daemon.CP.brand = 'honda'
  daemon.CP.carFingerprint = CAR.HONDA_ACCORD
  daemon.CP.pcmCruise = daemon.cp_pcm_cruise = True
  daemon.CP.minEnableSpeed = -1.0
  daemon.CP_SP = custom.CarParamsSP.new_message()
  daemon.car_events = SimpleNamespace(update=Mock(wraps=CarEvents(daemon.CP).update))
  daemon.car_events_sp = SimpleNamespace(update=Mock(wraps=CarSpecificEventsSP(daemon.CP, daemon.CP_SP).update))
  daemon.disengage_on_accelerator = True
  daemon.state_machine = StateMachine()
  daemon.state_machine.state = initial_state
  daemon.active = daemon.enabled = initial_state == State.enabled
  daemon.AM = AlertManager()
  daemon.pm = CapturingPublisher()
  daemon.personality = log.LongitudinalPersonality.standard
  daemon.events_prev = []
  daemon.events_sp_prev = []
  daemon.mads = SimpleNamespace(enabled=False, active=False, enabled_toggle=False,
                               state_machine=SimpleNamespace(state='disabled'))
  daemon.icbm.state = 'inactive'
  daemon.icbm.cruise_button = 'none'
  daemon.icbm.v_target = 0.0
  daemon.button_state_tracker = ButtonStateTracker()
  daemon.button_frame_counts = {car.CarState.ButtonEvent.Type.gapAdjustCruise: 0}
  daemon.experimental_mode_switched = False
  car_state.cruiseState.available = True
  car_state.cruiseState.enabled = True
  daemon.CS_prev = car_state.as_reader().as_builder()
  return daemon, car_state


def apply_fields(message, fields):
  for name, value in fields.items():
    if isinstance(value, dict):
      apply_fields(getattr(message, name), value)
    else:
      setattr(message, name, value)


def event_step(daemon, car_state, *, legacy=False):
  with (patch.object(EventsBase, 'add_from', legacy_add_from) if legacy else nullcontext()):
    daemon.update_events(car_state)
  daemon.enabled, daemon.active = daemon.state_machine.update(daemon.events)
  daemon.update_alerts(car_state)
  daemon.button_state_tracker.update(car_state)
  daemon.publish_selfdriveState(car_state)
  messages = []
  for service, message in daemon.pm.history:
    payload = message.to_dict()
    payload.pop('logMonoTime')
    messages.append((service, payload))
  return (event_snapshot(daemon.events), event_snapshot(daemon.events_sp),
          vars(daemon.state_machine).copy(), messages)


SCENARIOS = [
  ('steady', {}, {}, State.enabled, [], State.enabled),
  ('pcm_enable', {}, {'cruiseState': {'enabled': False}}, State.disabled, [EventName.pcmEnable], State.enabled),
  ('pcm_disable', {'cruiseState': {'enabled': False}}, {}, State.enabled, [EventName.cruiseDisabled], State.disabled),
  ('button_enable', {'buttonEnable': True}, {}, State.disabled, [EventName.buttonEnable], State.enabled),
  ('button_cancel', {'buttonEvents': [{'type': 'cancel', 'pressed': True}]}, {}, State.enabled,
   [EventName.buttonCancel], State.disabled),
  ('brake_rise', {'brakePressed': True}, {}, State.enabled, [EventName.pedalPressed], State.disabled),
  ('brake_held', {'brakePressed': True}, {'brakePressed': True}, State.enabled, [EventName.pedalPressed], State.disabled),
  ('brake_still', {'brakePressed': True, 'standstill': True, 'vEgo': 0.0}, {'brakePressed': True}, State.enabled,
   [EventName.preEnableStandstill], State.enabled),
  ('pre_enable', {'brakePressed': True, 'standstill': True, 'vEgo': 0.0},
   {'brakePressed': True, 'cruiseState': {'enabled': False}}, State.disabled,
   [EventName.preEnableStandstill, EventName.pcmEnable], State.preEnabled),
  ('gas_rise', {'gasPressed': True}, {}, State.enabled, [EventName.gasPressedOverride, EventName.pedalPressed], State.disabled),
  ('gas_held', {'gasPressed': True}, {'gasPressed': True}, State.enabled, [EventName.gasPressedOverride], State.overriding),
  ('regen_rise', {'regenBraking': True}, {}, State.enabled, [EventName.pedalPressed], State.disabled),
  ('regen_held', {'regenBraking': True}, {'regenBraking': True}, State.enabled, [EventName.pedalPressed], State.disabled),
  ('regen_still', {'regenBraking': True, 'standstill': True, 'vEgo': 0.0}, {'regenBraking': True}, State.enabled, [], State.enabled),
  ('reverse', {'gearShifter': 'reverse'}, {}, State.enabled, [EventName.wrongGear, EventName.reverseGear], State.disabled),
  ('steer_fault', {'steerFaultPermanent': True}, {}, State.enabled, [EventName.steerUnavailable], State.disabled),
]


@pytest.mark.parametrize('can_valid', [True, False])
@pytest.mark.parametrize('name,current,previous,initial_state,expected_events,expected_state', SCENARIOS,
                         ids=[scenario[0] for scenario in SCENARIOS])
def test_real_car_events_keep_complete_messages_alerts_and_state(
    name, current, previous, initial_state, expected_events, expected_state, can_valid):
  results = []
  for legacy in (True, False):
    daemon, car_state = event_daemon(initial_state)
    apply_fields(car_state, current)
    apply_fields(daemon.CS_prev, previous)
    car_state.canValid = can_valid
    results.append(event_step(daemon, car_state, legacy=legacy))
    assert daemon.car_events.update.call_count == int(can_valid)
    assert daemon.car_events_sp.update.call_count == int(can_valid)
    assert daemon.events.names == sorted(expected_events if can_valid else [EventName.canError])
    if can_valid:
      assert daemon.state_machine.state == expected_state
  assert results[0] == results[1]


def test_local_car_events_are_recomputed_on_every_valid_frame():
  inputs = [{}, {'brakePressed': True}, {'canValid': False}, {'brakePressed': True}, {}]
  expected = [[], [EventName.pedalPressed], [EventName.canError], [EventName.pedalPressed], []]
  results = []
  for legacy in (True, False):
    daemon, initial_car_state = event_daemon()
    frames = []
    valid_frames = 0
    for fields, event_names in zip(inputs, expected, strict=True):
      car_state = initial_car_state.as_reader().as_builder()
      apply_fields(car_state, fields)
      frames.append(event_step(daemon, car_state, legacy=legacy))
      valid_frames += int(car_state.canValid)
      assert daemon.car_events.update.call_count == valid_frames
      assert daemon.car_events_sp.update.call_count == valid_frames
      assert daemon.events.names == event_names
      daemon.CS_prev = car_state
      daemon.sm.frame += 1
    results.append(frames)
  assert results[0] == results[1]


@pytest.mark.parametrize('can_valid', [True, False])
def test_sp_update_sees_merged_events_and_preserves_external_message_decode(can_valid):
  results = []
  for legacy in (True, False):
    daemon, car_state = event_daemon(State.disabled)
    car_state.canValid = can_valid
    source = Events()
    for name in (EventName.belowSteerSpeed, EventName.buttonEnable, EventName.belowSteerSpeed):
      source.add(name)
    source_sp = EventsSP()
    source_sp.add(EventNameSP.lkasEnable)
    source_sp.add(EventNameSP.lkasEnable)

    def update_sp(_car_state, events, *, source=source, source_sp=source_sp):
      assert events.names == source.names
      events.remove(EventName.belowSteerSpeed)
      return source_sp

    daemon.car_events.update = Mock(return_value=source)
    daemon.car_events_sp.update = Mock(side_effect=update_sp)
    incoming = new_event('longitudinalPlanSP')
    external = EventsSP()
    external.add(EventNameSP.manualLongitudinalRequired)
    incoming.longitudinalPlanSP.events = external.to_msg()
    daemon.sm.update_msgs(2.0, [serialized(incoming)])
    results.append(event_step(daemon, car_state, legacy=legacy))
    assert daemon.car_events.update.call_count == int(can_valid)
    assert daemon.car_events_sp.update.call_count == int(can_valid)
    assert daemon.events.names == sorted([EventName.buttonEnable, EventName.belowSteerSpeed] if can_valid else [EventName.canError])
    assert daemon.events_sp.names == sorted(external.names + (source_sp.names if can_valid else []))
  assert results[0] == results[1]
