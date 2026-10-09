"""Offer/owner ordering and the original absolute three-second fault deadline."""
from unittest.mock import patch

import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.tests.test_continuation_control_chain import ContinuationChain, START
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import new_event, serialized, selfdrive_event_harness
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.selfdrived.events import ET, Events
from openpilot.selfdrive.selfdrived.state import StateMachine


State = log.SelfdriveState.OpenpilotState
EventName = log.OnroadEvent.EventName


def observe_at(daemon, chain, sample_ns, *, offset_ns=4_000_000):
  now = sample_ns + offset_ns
  model = new_event('modelV2', chain.model_ns)
  model.modelV2 = chain.state['modelV2']
  messages = [chain.plan_event, serialized(model),
              chain.controls.control.pm.messages['controlsState'], chain.controls.control.pm.messages['carControl']]
  daemon.sm.update_msgs(now / 1e9, messages)
  daemon.sm.freq_ok = dict.fromkeys(daemon.sm.services, True)
  daemon._car_state_mono_ns, daemon._car_state_valid = sample_ns, True
  with patch('openpilot.selfdrive.selfdrived.selfdrived.time.monotonic_ns', return_value=now), \
       patch('openpilot.selfdrive.selfdrived.selfdrived.time.monotonic', return_value=now / 1e9):
    daemon.update_events(chain.state['carState'])
  return now


def established_events():
  chain = ContinuationChain()
  daemon = None
  for offset in (0, 50_000_000):
    chain.produce(START + offset)
    for tick in range(5):
      sample = START + offset + tick * 10_000_000
      chain.consume(sample)
      if daemon is None:
        daemon, _ = selfdrive_event_harness(chain.controls.control.pm.messages['controlsState'])
        daemon.CP, daemon.CP_SP = chain.controls.control.CP, chain.controls.control.CP_SP
        daemon.big_model_ready_t = -10.0
        daemon.state_machine = StateMachine()
        daemon.state_machine.state = State.enabled
      observe_at(daemon, chain, sample)
      assert not daemon.events.names
  assert chain.planner.reference_planner.continuation.donor is not None
  return daemon, chain


def test_qualified_offer_before_owner_ack_and_current_receipt_both_keep_existing_track_running():
  daemon, chain = established_events()
  event = chain.produce(START + 100_000_000, 4)
  assert event.longitudinalPlan.continuation.valid
  consumed_before = daemon.sm['controlsState'].motionState.planMonoTime
  observe_at(daemon, chain, START + 100_000_000, offset_ns=2_000_000)
  assert consumed_before != event.logMonoTime and not daemon.sm['controlsState'].motionState.continuationActive
  assert not daemon.events.names
  assert daemon._longitudinal_failure_budget.episode_start_ns == event.longitudinalPlan.continuation.episodeStartMonoTime
  assert not daemon.sm.valid['longitudinalPlan'] and not daemon.sm['longitudinalPlan'].motionIntent.valid
  chain.consume(START + 100_000_000)
  observe_at(daemon, chain, START + 100_000_000, offset_ns=6_000_000)
  assert daemon.sm['controlsState'].motionState.continuationActive
  assert not daemon.events.names
  # A cached offer is not entry permission after a driver/configuration exit.
  daemon.enabled = False
  daemon.state_machine.state = State.disabled
  observe_at(daemon, chain, START + 110_000_000)
  assert daemon.events.contains(ET.NO_ENTRY)
  daemon.events.add(EventName.buttonEnable)
  assert daemon.state_machine.update(daemon.events) == (False, False)


def test_first_unavailable_event_spends_remaining_time_from_the_first_failed_solve():
  daemon, chain = established_events()
  event = chain.produce(START + 100_000_000, 4)
  chain.consume(START + 100_000_000)
  observe_at(daemon, chain, START + 100_000_000)
  assert not daemon.events.names
  epoch = daemon._longitudinal_failure_budget.episode_start_ns
  chain.consume(START + 200_000_000)
  now = observe_at(daemon, chain, START + 200_000_000)
  assert EventName.longitudinalControlUnavailable in daemon.events.names
  assert daemon._longitudinal_soft_disable_remaining_ns == epoch + 3_000_000_000 - now
  enabled, active = daemon.state_machine.update(
    daemon.events, soft_disable_remaining_ns=daemon._longitudinal_soft_disable_remaining_ns)
  assert enabled and active and daemon.state_machine.state == State.softDisabling
  assert 0 < daemon.state_machine.soft_disable_timer < 300
  # New LP publications and a fresh donor cannot restart this deadline.
  assert not event.valid and event.longitudinalPlan.solverStatus == 4
  assert daemon.state_machine.update(daemon.events, soft_disable_remaining_ns=0) == (False, False)


@pytest.mark.parametrize('initial', [State.enabled, State.overriding, State.preEnabled, State.softDisabling])
def test_expired_absolute_deadline_has_no_extra_active_tick_from_any_engaged_state(initial):
  machine = StateMachine()
  machine.state, machine.soft_disable_timer = initial, 240
  events = Events()
  events.add(EventName.longitudinalControlUnavailable)
  assert machine.update(events, soft_disable_remaining_ns=0) == (False, False)
  assert machine.state == State.disabled and machine.soft_disable_timer == 0


def test_episode_bound_never_extends_an_existing_shorter_disable_or_overrides_driver_priority():
  machine = StateMachine()
  machine.state, machine.soft_disable_timer = State.softDisabling, 10
  events = Events()
  events.add(EventName.longitudinalControlUnavailable)
  machine.update(events, soft_disable_remaining_ns=2_000_000_000)
  assert machine.soft_disable_timer == 9
  events.add(EventName.buttonCancel)
  assert machine.update(events, soft_disable_remaining_ns=2_000_000_000) == (False, False)
