"""Offer/owner ordering and the original absolute three-second fault deadline."""
from unittest.mock import patch

import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_continuation import (
  NATIVE_RECOVERY_NS, UNSTABLE_DISABLE_NS, ContinuationOffer,
)
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
  with patch.object(ContinuationOffer, 'from_message', wraps=ContinuationOffer.from_message) as decode:
    observe_at(daemon, chain, START + 100_000_000, offset_ns=2_000_000)
    decode.assert_called_once()
  assert consumed_before != event.logMonoTime and not daemon.sm['controlsState'].motionState.continuationActive
  assert not daemon.events.names
  assert daemon._longitudinal_failure_budget.episode_start_ns == event.longitudinalPlan.continuation.episodeStartMonoTime
  assert not daemon.sm.valid['longitudinalPlan'] and not daemon.sm['longitudinalPlan'].motionIntent.valid
  chain.consume(START + 100_000_000)
  with patch.object(ContinuationOffer, 'from_message', wraps=ContinuationOffer.from_message) as decode:
    observe_at(daemon, chain, START + 100_000_000, offset_ns=6_000_000)
    decode.assert_called_once()
  assert daemon.sm['controlsState'].motionState.continuationActive
  assert not daemon.events.names
  # A cached offer is not entry permission after a driver/configuration exit.
  daemon.enabled = False
  daemon.state_machine.state = State.disabled
  observe_at(daemon, chain, START + 110_000_000)
  assert daemon.events.contains(ET.NO_ENTRY)
  daemon.events.add(EventName.buttonEnable)
  assert daemon.state_machine.update(daemon.events) == (False, False)


def test_healthy_frames_skip_offer_decode_while_preserving_history_donor_and_budget_recovery():
  daemon, chain = established_events()
  failed = chain.produce(START + 100_000_000, 4)
  chain.consume(START + 100_000_000)
  observe_at(daemon, chain, START + 100_000_000)
  episode = failed.longitudinalPlan.continuation.episodeStartMonoTime
  assert episode > 0 and daemon._longitudinal_failure_budget.episode_start_ns == episode

  recovery_start = START + 150_000_000
  for elapsed in range(0, NATIVE_RECOVERY_NS + 1, 50_000_000):
    event = chain.produce(recovery_start + elapsed)
    assert event.valid and event.longitudinalPlan.solverStatus == 0
    # This harness only publishes plan/model/owner messages. Keep its fixed
    # ancillary inputs alive while those real clocks span the recovery window.
    for service in daemon.sm.services:
      if service not in ('longitudinalPlan', 'modelV2', 'controlsState', 'carControl'):
        daemon.sm.recv_time[service] = (recovery_start + elapsed) / 1e9
    for tick in range(5):
      sample = recovery_start + elapsed + tick * 10_000_000
      _, motion, _ = chain.consume(sample)
      assert not motion.degraded and not motion.continuationActive
      # Limit the spy to selfdrived's observation, excluding producer/owner decoding.
      with patch.object(ContinuationOffer, 'from_message', wraps=ContinuationOffer.from_message) as decode:
        now = observe_at(daemon, chain, sample)
        decode.assert_not_called()
      assert not daemon.events.names
      assert daemon._continuation_history.samples[-1][0] == sample
      assert daemon._continuation_history.current_state(now) is not None
      assert daemon._continuation_donor.model_ns == chain.model_ns
      assert daemon._continuation_donor.producer_ns == motion.referenceProducerMonoTime
      if elapsed < NATIVE_RECOVERY_NS:
        assert daemon._longitudinal_failure_budget.episode_start_ns == episode
        assert daemon._longitudinal_soft_disable_remaining_ns == episode + UNSTABLE_DISABLE_NS - now
      else:
        assert daemon._longitudinal_failure_budget.episode_start_ns == 0
        assert daemon._longitudinal_soft_disable_remaining_ns is None


def test_healthy_plan_with_degraded_owner_still_decodes_the_original_failure_episode():
  daemon, chain = established_events()
  failed = chain.produce(START + 100_000_000, 4)
  assert failed.longitudinalPlan.continuation.valid
  chain.consume(START + 100_000_000)
  healthy = chain.produce(START + 150_000_000).as_builder()
  # Preserve the old envelope to exercise its deadline even with native success.
  healthy.longitudinalPlan.continuation = failed.longitudinalPlan.continuation
  chain.plan_event = serialized(healthy)
  chain.consume(START + 150_000_000)
  owner = chain.controls.control.pm.messages['controlsState'].as_builder()
  owner.controlsState.motionState.degraded = True
  chain.controls.control.pm.messages['controlsState'] = serialized(owner)
  assert chain.plan_event.valid and chain.plan_event.longitudinalPlan.solverStatus == 0
  with patch.object(ContinuationOffer, 'from_message', wraps=ContinuationOffer.from_message) as decode:
    now = observe_at(daemon, chain, START + 150_000_000)
    decode.assert_called_once()
  episode = failed.longitudinalPlan.continuation.episodeStartMonoTime
  assert daemon._longitudinal_failure_budget.episode_start_ns == episode
  assert daemon._longitudinal_soft_disable_remaining_ns == episode + UNSTABLE_DISABLE_NS - now
  assert EventName.longitudinalControlUnavailable in daemon.events.names


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
