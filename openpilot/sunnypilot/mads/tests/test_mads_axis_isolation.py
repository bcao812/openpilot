"""Preserve established lateral authority without hiding faults or entry gates."""

from types import SimpleNamespace

import pytest

from openpilot.cereal import custom, log
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.selfdrived.events import ET, Events
from openpilot.selfdrive.selfdrived.state import StateMachine as MainStateMachine
from openpilot.sunnypilot.mads.state import SOFT_DISABLE_TIME, StateMachine
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP


State = custom.ModularAssistiveDrivingSystem.ModularAssistiveDrivingSystemState
MainState = log.SelfdriveState.OpenpilotState
EventName = log.OnroadEvent.EventName
EventNameSP = custom.OnroadEventSP.EventName
SOFT_DISABLE_FRAMES = int(SOFT_DISABLE_TIME / DT_CTRL)


def make_mads(initial=State.enabled, *, isolated=True, main_enabled=False, main_timer=0):
  main = MainStateMachine()
  main.state = MainState.enabled if main_enabled else MainState.disabled
  main.soft_disable_timer = main_timer
  selfdrive = SimpleNamespace(enabled=main_enabled, state_machine=main, events=Events(), events_sp=EventsSP())
  mads = SimpleNamespace(selfdrive=selfdrive, longitudinal_failure_isolated=isolated)
  mads.state_machine = StateMachine(mads)
  mads.state_machine.state = initial
  selfdrive.events.add(EventName.longitudinalControlUnavailable)
  return mads


@pytest.mark.parametrize('initial', [State.enabled, State.overriding, State.softDisabling])
def test_isolated_longitudinal_failure_preserves_only_established_active_states(initial):
  mads = make_mads(initial)
  machine = mads.state_machine
  if initial == State.overriding:
    mads.selfdrive.events.add(EventName.steerOverride)

  assert machine.update() == (True, True)
  assert machine.state == (State.overriding if initial == State.overriding else State.enabled)
  assert not machine.check_contains(ET.SOFT_DISABLE)
  assert machine.check_contains(ET.NO_ENTRY)


@pytest.mark.parametrize('initial', [State.disabled, State.paused])
def test_isolation_cannot_enable_or_resume_lateral_while_no_entry_remains(initial):
  mads = make_mads(initial)
  mads.selfdrive.events_sp.add(EventNameSP.lkasEnable)

  assert mads.state_machine.check_contains(ET.SOFT_DISABLE)
  enabled, active = mads.state_machine.update()
  assert mads.state_machine.state == initial
  assert enabled == (initial == State.paused)
  assert not active
  assert ET.NO_ENTRY in mads.selfdrive.state_machine.current_alert_types


@pytest.mark.parametrize('isolated', [False, None, 0, 1, 'true'])
def test_only_explicit_true_qualifies_for_isolation(isolated):
  mads = make_mads(isolated=isolated)

  assert mads.state_machine.update() == (True, True)
  assert mads.state_machine.state == State.softDisabling
  assert mads.state_machine.soft_disable_timer == SOFT_DISABLE_FRAMES


def test_isolation_does_not_remove_event_flags_or_alerts():
  mads = make_mads()
  events = mads.selfdrive.events
  original_names = tuple(events.names)

  mads.state_machine.update()

  assert tuple(events.names) == original_names
  assert events.contains(ET.SOFT_DISABLE) and events.contains(ET.NO_ENTRY)
  published = next(event for event in events.to_msg() if event.name == EventName.longitudinalControlUnavailable)
  assert published.softDisable and published.noEntry
  alerts = events.create_alerts([ET.SOFT_DISABLE], [None, None, None, True, SOFT_DISABLE_FRAMES, None])
  assert len(alerts) == 1
  assert alerts[0].event_type == ET.SOFT_DISABLE
  assert alerts[0].alert_text_2 == 'Longitudinal Control Unavailable'


@pytest.mark.parametrize('namespace,event,expected', [
  ('base', EventName.cameraMalfunction, State.softDisabling),
  ('base', EventName.canError, State.disabled),
  ('base', EventName.controlsMismatch, State.disabled),
  ('base', EventName.steerUnavailable, State.disabled),
  ('sp', EventNameSP.controlsMismatchLateral, State.disabled),
  ('sp', EventNameSP.lkasDisable, State.disabled),
])
def test_common_faults_and_lateral_user_disable_are_not_masked(namespace, event, expected):
  mads = make_mads()
  events = mads.selfdrive.events if namespace == 'base' else mads.selfdrive.events_sp
  events.add(event)

  enabled, active = mads.state_machine.update()

  assert mads.state_machine.state == expected
  assert enabled == active == (expected == State.softDisabling)
  assert events.has(event)
  assert mads.selfdrive.events.has(EventName.longitudinalControlUnavailable)


def test_main_longitudinal_deadline_does_not_end_isolated_lateral_control():
  mads = make_mads(main_enabled=True)
  selfdrive = mads.selfdrive
  for _ in range(SOFT_DISABLE_FRAMES + 2):
    selfdrive.enabled, _ = selfdrive.state_machine.update(selfdrive.events)
    assert mads.state_machine.update() == (True, True)

  assert not selfdrive.enabled
  assert selfdrive.state_machine.state == MainState.disabled
  assert selfdrive.state_machine.soft_disable_timer == 0
  assert mads.state_machine.state == State.enabled


@pytest.mark.parametrize('main_enabled', [False, True])
@pytest.mark.parametrize('main_timer', [0, 2 * SOFT_DISABLE_FRAMES])
def test_lateral_fault_owns_its_timeout_without_modifying_main_timer(main_enabled, main_timer):
  mads = make_mads(main_enabled=main_enabled, main_timer=main_timer)
  mads.selfdrive.events.add(EventName.cameraMalfunction)
  machine = mads.state_machine

  assert machine.update() == (True, True)
  assert machine.state == State.softDisabling
  assert machine.soft_disable_timer == SOFT_DISABLE_FRAMES
  for remaining in range(SOFT_DISABLE_FRAMES - 1, 0, -1):
    assert machine.update() == (True, True)
    assert machine.soft_disable_timer == remaining
    assert mads.selfdrive.state_machine.soft_disable_timer == main_timer
  assert machine.update() == (False, False)
  assert machine.state == State.disabled
  assert machine.soft_disable_timer == 0
  assert mads.selfdrive.state_machine.soft_disable_timer == main_timer


def test_lateral_recovers_when_common_fault_clears_but_cannot_reenable_after_timeout():
  mads = make_mads()
  events = mads.selfdrive.events
  machine = mads.state_machine
  events.add(EventName.cameraMalfunction)
  machine.update()
  for _ in range(10):
    machine.update()
  events.remove(EventName.cameraMalfunction)

  assert machine.update() == (True, True)
  assert machine.state == State.enabled

  events.add(EventName.cameraMalfunction)
  machine.update()
  for _ in range(SOFT_DISABLE_FRAMES):
    machine.update()
  assert machine.state == State.disabled
  events.remove(EventName.cameraMalfunction)
  mads.selfdrive.events_sp.add(EventNameSP.lkasEnable)
  assert machine.update() == (False, False)
  assert machine.state == State.disabled
  assert machine.check_contains(ET.NO_ENTRY)


def test_revoking_isolation_restores_lateral_soft_disable():
  mads = make_mads()
  assert mads.state_machine.update() == (True, True)
  assert mads.state_machine.state == State.enabled

  mads.longitudinal_failure_isolated = False
  assert mads.state_machine.update() == (True, True)
  assert mads.state_machine.state == State.softDisabling
  assert mads.state_machine.soft_disable_timer == SOFT_DISABLE_FRAMES
