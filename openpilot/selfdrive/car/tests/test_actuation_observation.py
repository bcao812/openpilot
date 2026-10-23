from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from opendbc.car import structs
from opendbc.car.honda.values import CAR
from opendbc.car.structs import car
from openpilot.cereal import custom, log, messaging
from openpilot.common.longitudinal_logging import CHECK_FLAGS, LOGGING_MARKERS, can_digest
from openpilot.selfdrive.car.actuation_observation import ActuationObservation
from openpilot.selfdrive.car.card import Car
from openpilot.selfdrive.car.helpers import convert_to_capnp
from openpilot.selfdrive.car.tests.test_longitudinal_authority import honda_authority_chain
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import CapturingPublisher, bounded_submaster, new_event, serialized
from openpilot.selfdrive.pandad import can_capnp_to_list, can_list_to_can_capnp


class CardHarness:
  def __init__(self, *, enabled=True, native=False):
    self.now_ns = 10_000_000_000
    self.requests = []
    self.sent_frames = []
    self.chain = honda_authority_chain() if native else None
    self.events = {service: new_event(service, self.now_ns - (index + 1) * 1_000_003)
                   for index, service in enumerate(("carControl", "carControlSP", "onroadEvents", "longitudinalPlanSP"))}
    self.command = self.events["carControl"].carControl
    self.command.enabled = True
    self.command.latActive = True
    self.command.longActive = True
    self.command.actuators.accel = 0.5
    self.command.actuators.torque = 0.2
    self.command.actuators.longControlState = "pid"
    self.command.cruiseControl.resume = True
    self.state = car.CarState.new_message()
    self.state.canValid = True
    self.state.standstill = True
    self.state.cruiseState.available = True
    self.state.vCruise = 80.0
    self.state.vCruiseCluster = 80.0
    card = Car.__new__(Car)
    card.CP = self.chain.params if native else car.CarParams.new_message(carFingerprint=CAR.HONDA_ACCORD,
                                                                        openpilotLongitudinalControl=True)
    card.CP_SP = self.chain.params_sp if native else structs.CarParamsSP()
    card.CP_SP_capnp = convert_to_capnp(card.CP_SP)
    card.CC_prev = car.CarControl.new_message()
    card.CS_prev = car.CarState.new_message()
    card.CS_SP_prev = custom.CarStateSP.new_message()
    card.initialized_prev = True
    card.prof = None
    card.can_sock = object()
    card.can_callbacks = ()
    card.can_rcv_cum_timeout_counter = 0
    card.is_metric = True
    card.experimental_mode = False
    card.radar_signal = True
    card.dynamic_experimental_control = False
    card.rk = SimpleNamespace(remaining=0.0)
    card.last_actuators_output = car.CarControl.Actuators.new_message()
    card.params = SimpleNamespace(put_bool=Mock())
    card.sm = bounded_submaster(list(self.events))
    card.sm.update = Mock()
    card.pm = CapturingPublisher()
    card.RI = SimpleNamespace(update=Mock(return_value=None))
    card.v_cruise_helper = SimpleNamespace(update_speed_limit_assist=Mock(), update_v_cruise=Mock(),
                                           initialize_v_cruise=Mock(), v_cruise_kph=80.0, v_cruise_cluster_kph=80.0)
    card.CI = SimpleNamespace(
      CC=self.chain.controller if native else SimpleNamespace(frame=0),
      update=lambda _: (self.state, structs.CarStateSP()), apply=self.apply, init=Mock(),
    )
    card.longitudinal_observation = (ActuationObservation.create(card.CP, card.CP_SP_capnp) if enabled is None else
                                     ActuationObservation(card.CP, card.CP_SP_capnp, enabled=enabled, clock=lambda: self.now_ns))
    self.card = card

  def apply(self, command, extension, timestamp):
    self.requests.append((command.to_dict(), asdict(extension), timestamp))
    if self.chain is not None:
      self.chain.wrapper.out = self.state.as_reader()
      output, frames = self.chain.controller.update(command, extension, self.chain.wrapper, timestamp)
    else:
      output = command.actuators.as_builder()
      frames = [(0x1DF, bytes([self.card.CI.CC.frame % 256, int(command.longActive)]), 0),
                (0xE4, b"\x12\x34", 1)]
      self.card.CI.CC.frame += 1
    self.sent_frames.append(tuple(frames))
    return output, frames

  def feed(self, *, fresh_control=True, timestamp=None):
    self.now_ns += 10_000_000
    incoming = []
    for index, (service, event) in enumerate(self.events.items()):
      if service == "carControl" and not fresh_control:
        continue
      event.logMonoTime = self.now_ns - (index + 1) * 1_000_003 if timestamp is None else timestamp + index
      incoming.append(serialized(event))
    self.card.sm.update_msgs(self.now_ns / 1e9, incoming)

  def step(self, *, fresh_control=True, stale=False, distinct_clock=False, observation_clock=None):
    self.feed(fresh_control=fresh_control)
    if stale:
      self.card.sm.alive["carControl"] = False
    offset = 0

    def clock():
      nonlocal offset
      offset += 1_003
      return self.now_ns + offset

    if self.card.longitudinal_observation is not None:
      self.card.longitudinal_observation.recorder.clock = (observation_clock if observation_clock is not None else
                                                         clock if distinct_clock else lambda: int((self.now_ns / 1e9) * 1e9))
    with patch("openpilot.selfdrive.car.card.REPLAY", False), \
         patch("time.monotonic", side_effect=(lambda: clock() / 1e9) if distinct_clock else lambda: self.now_ns / 1e9), \
         patch("time.monotonic_ns", side_effect=clock if distinct_clock else lambda: self.now_ns), \
         patch("openpilot.selfdrive.car.card.messaging.drain_sock_raw", return_value=[]):
      self.card.step()
    return self.card.pm.messages["carOutput"].carOutput.longitudinalDiagnostics

  @property
  def outputs(self):
    return [event for service, event in self.card.pm.history if service == "carOutput"]


def message_payloads(harness):
  payloads = []
  for service, event in harness.card.pm.history:
    payload = event.to_dict()
    if service == "carOutput":
      payload[service].pop("longitudinalDiagnostics", None)
    payloads.append((service, payload))
  return payloads


def test_missing_marker_skips_receipts_snapshots_and_can_digest(tmp_path):
  with patch.dict(LOGGING_MARKERS, {"card": tmp_path / "missing"}), \
       patch("openpilot.common.longitudinal_logging.source_manifest") as manifest, \
       patch("openpilot.selfdrive.car.actuation_observation.capture_inputs") as inputs, \
       patch("openpilot.selfdrive.car.actuation_observation.can_digest") as digest:
    harness = CardHarness(enabled=None)
    assert harness.card.longitudinal_observation is None
    for _ in range(2):
      assert harness.step().context.version == 0
    assert len(harness.requests) == len(harness.sent_frames) == 2
  manifest.assert_not_called()
  inputs.assert_not_called()
  digest.assert_not_called()


def test_card_publishes_previous_immutable_apply_receipt_with_exact_consumed_identities():
  harness = CardHarness()
  first = harness.step(distinct_clock=True)
  assert first.context.version == 1
  assert first.context.sequence == 1
  assert not first.hasApply
  assert first.lastDisposition == "unobserved"
  receipt = harness.card.longitudinal_observation.receipt
  state_id = harness.card.pm.messages["carState"].logMonoTime
  command_id = harness.card.sm.logMonoTime["carControl"]
  extension_id = harness.card.sm.logMonoTime["carControlSP"]
  sendcan = harness.card.pm.messages["sendcan"]
  assert len({state_id, command_id, extension_id, receipt["applyStartMonoTime"], receipt["applyEndMonoTime"], sendcan.logMonoTime}) == 6
  assert command_id < state_id < receipt["applyStartMonoTime"] < receipt["applyEndMonoTime"] < sendcan.logMonoTime
  assert receipt["carStateMonoTime"] == state_id
  assert receipt["carControlMonoTime"] == command_id
  assert receipt["carControlSPMonoTime"] == extension_id
  assert receipt["applyFrame"] == harness.card.sm.frame
  assert receipt["controllerFrameKnown"]
  assert receipt["controllerFrame"] == 0
  assert receipt["sendcanMonoTime"] == sendcan.logMonoTime
  assert receipt["canCount"] == len(sendcan.sendcan)
  assert receipt["canDigest"] == can_digest(harness.sent_frames[0])
  with pytest.raises(TypeError):
    receipt["applySequence"] = 99
  with pytest.raises(TypeError):
    receipt["inputs"][0]["monoTime"] = 0

  previous_inputs = tuple(dict(entry) for entry in receipt["inputs"])
  second = harness.step(distinct_clock=True)
  assert second.hasApply
  assert second.applySequence == 1
  assert second.carStateMonoTime == state_id != harness.card.pm.messages["carState"].logMonoTime
  assert second.carControlMonoTime == command_id != harness.card.sm.logMonoTime["carControl"]
  assert second.carControlSPMonoTime == extension_id != harness.card.sm.logMonoTime["carControlSP"]
  assert tuple(entry.to_dict() for entry in second.inputs) == previous_inputs
  assert second.applyStartMonoTime == receipt["applyStartMonoTime"]
  assert second.applyEndMonoTime == receipt["applyEndMonoTime"]
  assert second.sendcanMonoTime == receipt["sendcanMonoTime"]
  assert second.lastDisposition == "sendcanPublished"
  assert second.context.sequence == 2
  assert second.context.session == first.context.session
  assert [service for service, _ in harness.card.pm.history] == [
    "carParams", "carOutput", "carState", "carParamsSP", "carStateSP", "sendcan",
    "carOutput", "carState", "carStateSP", "sendcan",
  ]


def test_one_car_control_can_have_many_apply_receipts_and_tail_is_not_invented():
  harness = CardHarness()
  harness.step()
  command_id = harness.card.sm.logMonoTime["carControl"]
  for _frame in range(4):
    harness.step(fresh_control=False)
  diagnostics = [event.carOutput.longitudinalDiagnostics for event in harness.outputs]
  assert [entry.applySequence for entry in diagnostics if entry.hasApply] == [1, 2, 3, 4]
  assert all(entry.carControlMonoTime == command_id for entry in diagnostics if entry.hasApply)
  assert len({entry.carStateMonoTime for entry in diagnostics if entry.hasApply}) == 4
  assert harness.card.longitudinal_observation.receipt["applySequence"] == 5
  assert all(entry.applySequence != 5 for entry in diagnostics)
  assert len(harness.requests) == 5


@pytest.mark.parametrize("failure", ["digest", "capture_inputs", "begin_apply", "end_apply", "sent"])
def test_new_apply_without_a_complete_receipt_cannot_reuse_the_previous_output_identity(failure):
  harness = CardHarness()
  harness.step()
  previous = harness.card.longitudinal_observation.receipt
  assert previous["applySequence"] == 1
  harness.command.actuators.accel = -0.5
  if failure in ("digest", "capture_inputs"):
    symbol = "can_digest" if failure == "digest" else "capture_inputs"
    injection = patch(f"openpilot.selfdrive.car.actuation_observation.{symbol}", side_effect=RuntimeError(failure))
  else:
    injection = patch.object(harness.card.longitudinal_observation, failure, side_effect=RuntimeError(failure))
  with injection:
    harness.step()
  assert len(harness.requests) == 2
  assert len(harness.sent_frames) == 2
  assert harness.card.longitudinal_observation.receipt is None
  observed = harness.step(stale=True)
  assert not observed.hasApply
  assert observed.context.errors > 0
  assert observed.lastDisposition == "sendcanPublished"
  assert harness.outputs[-1].carOutput.actuatorsOutput.accel == -0.5
  assert observed.applySequence == 0
  assert observed.sendcanMonoTime == 0
  assert len(harness.requests) == 2


@pytest.mark.parametrize("pedal", ["gasPressed", "brakePressed", "both"])
def test_latest_pedal_seen_only_by_card_is_recorded_without_mutating_cached_control(pedal):
  harness = CardHarness()
  harness.step()
  previous_control = harness.card.sm["carControl"]
  original = previous_control.to_dict()
  if pedal == "both":
    harness.state.gasPressed = True
    harness.state.brakePressed = True
  else:
    setattr(harness.state, pedal, True)
  harness.step(fresh_control=False)
  receipt = harness.card.longitudinal_observation.receipt
  assert previous_control.to_dict() == original
  assert harness.card.sm["carControl"].to_dict() == original
  assert receipt["requestedAccel"] == 0.5
  assert receipt["appliedAccel"] == 0.0
  assert receipt["requestedLongActive"] and not receipt["appliedLongActive"]
  assert receipt["requestedResume"] and not receipt["appliedResume"]
  assert receipt["appliedLongControlState"] == "off"
  assert receipt["authorityRestricted"]
  assert receipt["authorityReason"] == ("gasPressed+brakePressed" if pedal == "both" else pedal)
  applied = harness.requests[-1][0]
  assert not applied["longActive"]
  assert applied["latActive"] and applied["enabled"]
  assert applied["actuators"]["torque"] == original["actuators"]["torque"]
  assert receipt["carControlMonoTime"] == harness.card.sm.logMonoTime["carControl"]
  harness.step(fresh_control=False)
  published = harness.outputs[-1].carOutput.longitudinalDiagnostics
  assert published.applySequence == receipt["applySequence"]
  assert published.authorityReason == receipt["authorityReason"]


@pytest.mark.parametrize("condition", ["staleCarControl", "passive", "uninitialized"])
def test_skip_disposition_does_not_fabricate_new_apply_or_relabel_old_receipt(condition):
  harness = CardHarness()
  harness.step()
  receipt = harness.card.longitudinal_observation.receipt
  if condition == "passive":
    harness.card.CP.passive = True
  elif condition == "uninitialized":
    initializing = messaging.new_message("onroadEvents", 1, valid=True)
    initializing.onroadEvents[0].name = log.OnroadEvent.EventName.selfdriveInitializing
    harness.events["onroadEvents"] = initializing
  harness.step(stale=condition == "staleCarControl")
  observed = harness.step(stale=condition == "staleCarControl")
  assert len(harness.requests) == 1
  assert observed.lastDisposition == condition
  assert observed.hasApply
  assert observed.applySequence == 1
  assert observed.carStateMonoTime == receipt["carStateMonoTime"]
  assert observed.carControlMonoTime == receipt["carControlMonoTime"]
  assert observed.sendcanMonoTime == receipt["sendcanMonoTime"]
  assert harness.card.longitudinal_observation.apply_sequence == 1
  assert harness.outputs[-1].valid == (condition != "staleCarControl")


@pytest.mark.parametrize("condition", ["staleCarControl", "passive", "uninitialized"])
def test_initial_skip_has_no_apply_identity_and_reports_only_the_observed_branch(condition):
  harness = CardHarness()
  if condition == "passive":
    harness.card.CP.passive = True
  elif condition == "uninitialized":
    initializing = messaging.new_message("onroadEvents", 1, valid=True)
    initializing.onroadEvents[0].name = log.OnroadEvent.EventName.selfdriveInitializing
    harness.events["onroadEvents"] = initializing
  first = harness.step(stale=condition == "staleCarControl")
  second = harness.step(stale=condition == "staleCarControl")
  assert not first.hasApply and not second.hasApply
  assert first.lastDisposition == "unobserved"
  assert second.lastDisposition == condition
  assert first.applySequence == second.applySequence == 0
  assert first.carStateMonoTime == second.carStateMonoTime == 0
  assert first.carControlMonoTime == second.carControlMonoTime == 0
  assert first.sendcanMonoTime == second.sendcanMonoTime == 0
  assert harness.requests == []
  assert harness.card.longitudinal_observation.receipt is None


def test_unavailable_controller_frame_is_unknown_not_a_copy_of_card_frame():
  harness = CardHarness()
  harness.feed()
  harness.card.CI.CC = None
  attempt = harness.card.longitudinal_observation.begin_apply(
    harness.card.sm, harness.card.CI, harness.state, harness.command, harness.command,
  )
  assert not attempt["controllerFrameKnown"]
  assert attempt["controllerFrame"] == 0


def test_fresh_but_invalid_control_preserves_existing_apply_and_message_validity_semantics():
  harness = CardHarness()
  harness.events["carControl"].valid = False
  harness.step()
  assert len(harness.requests) == 1
  assert not harness.outputs[-1].valid
  inputs = {entry["service"]: entry for entry in harness.card.longitudinal_observation.receipt["inputs"]}
  assert inputs["carControl"]["knownFlags"] & CHECK_FLAGS["valid"]
  assert not inputs["carControl"]["flags"] & CHECK_FLAGS["valid"]
  assert inputs["carControl"]["flags"] & CHECK_FLAGS["alive"]
  assert harness.card.pm.messages["sendcan"].valid


@pytest.mark.parametrize("known_previous_state", [False, True])
def test_missing_current_published_state_is_unknown_not_now_or_previous_frame(known_previous_state):
  harness = CardHarness()
  harness.feed()
  observer = harness.card.longitudinal_observation
  if known_previous_state:
    observer.state_published(123_456_789, True, harness.card.sm.frame - 1)
  attempt = observer.begin_apply(harness.card.sm, harness.card.CI, harness.state, harness.command, harness.command)
  state = next(entry for entry in attempt["inputs"] if entry["service"] == "carState")
  assert attempt["carStateMonoTime"] == 0
  assert not state["idKnown"]
  assert state["kind"] == "unknown"


@pytest.mark.parametrize("method", ["publish", "state_published", "disposition", "begin_apply", "end_apply", "sent"])
def test_observation_failures_preserve_requests_and_every_existing_message_field(method):
  baseline = CardHarness(enabled=False)
  failing = CardHarness()
  with patch.object(failing.card.longitudinal_observation, method, side_effect=RuntimeError("observation")):
    baseline.step()
    failing.step()
  assert baseline.requests == failing.requests
  assert baseline.sent_frames == failing.sent_frames
  assert message_payloads(baseline) == message_payloads(failing)
  assert failing.card.longitudinal_observation.recorder.errors > 0


@pytest.mark.parametrize("failure", ["digest", "serialize"])
def test_digest_and_serialization_failure_are_observation_only(failure):
  baseline = CardHarness(enabled=False)
  failing = CardHarness()
  if failure == "digest":
    injection = patch("openpilot.selfdrive.car.actuation_observation.can_digest", side_effect=RuntimeError("digest"))
  else:
    injection = patch.object(car, "LongitudinalApplyDiagnostics", SimpleNamespace(new_message=Mock(side_effect=RuntimeError("serialize"))))
  baseline.step()
  with injection:
    failing.step()
  assert message_payloads(baseline) == message_payloads(failing)
  assert baseline.requests == failing.requests
  assert failing.card.longitudinal_observation.recorder.errors > 0


@pytest.mark.parametrize("failure", ["apply", "sendcan"])
def test_original_actuation_or_send_error_is_not_swallowed_and_has_no_success_receipt(failure):
  harness = CardHarness()
  error = RuntimeError(f"original {failure} error")
  original_send = harness.card.pm.send

  def send(service, message):
    if service == "sendcan":
      raise error
    original_send(service, message)

  if failure == "apply":
    harness.card.CI.apply = Mock(side_effect=error)
  else:
    harness.card.pm.send = send
  with pytest.raises(RuntimeError) as caught:
    harness.step()
  assert caught.value is error
  assert harness.card.longitudinal_observation.receipt is None
  assert not harness.card.pm.messages["carOutput"].carOutput.longitudinalDiagnostics.hasApply
  assert "sendcan" not in harness.card.pm.messages
  assert harness.card.longitudinal_observation.disposition_values["lastDisposition"] == (
    "applying" if failure == "apply" else "sendcanPending")


@pytest.mark.parametrize("failure", ["apply", "sendcan"])
def test_original_failure_after_success_invalidates_the_receipt_until_a_new_send_completes(failure):
  harness = CardHarness()
  harness.step(distinct_clock=True)
  observer = harness.card.longitudinal_observation
  previous = observer.receipt
  previous_send = harness.card.pm.messages["sendcan"]
  assert previous["applySequence"] == 1
  assert previous["sendcanMonoTime"] == previous_send.logMonoTime
  harness.command.actuators.accel = -0.5
  error = RuntimeError(f"original {failure} error after success")
  original_send = harness.card.pm.send

  def send(service, message):
    if service == "sendcan":
      raise error
    original_send(service, message)

  injection = (patch.object(harness.card.CI, "apply", side_effect=error) if failure == "apply" else
               patch.object(harness.card.pm, "send", side_effect=send))
  with injection, pytest.raises(RuntimeError) as caught:
    harness.step(distinct_clock=True)
  assert caught.value is error
  assert observer.receipt is None
  assert observer.apply_sequence == 2
  assert observer.disposition_values["lastDisposition"] == ("applying" if failure == "apply" else "sendcanPending")
  assert harness.card.pm.messages["sendcan"] is previous_send
  assert sum(service == "sendcan" for service, _ in harness.card.pm.history) == 1
  assert len(harness.requests) == len(harness.sent_frames) == (1 if failure == "apply" else 2)

  # card publishes the previous success before attempting the new apply/send.
  before_failure = harness.outputs[-1].carOutput.longitudinalDiagnostics
  assert before_failure.hasApply
  assert before_failure.applySequence == previous["applySequence"]
  assert before_failure.sendcanMonoTime == previous["sendcanMonoTime"]
  after_failure = harness.step(stale=True)
  assert not after_failure.hasApply
  assert after_failure.applySequence == after_failure.sendcanMonoTime == 0
  assert after_failure.lastDisposition == ("applying" if failure == "apply" else "sendcanPending")
  assert harness.outputs[-1].carOutput.actuatorsOutput.accel == (0.5 if failure == "apply" else -0.5)
  assert observer.receipt is None

  assert not harness.step().hasApply
  recovered = observer.receipt
  assert recovered["applySequence"] == 3
  assert recovered["sendcanMonoTime"] > previous["sendcanMonoTime"]
  after_recovery = harness.step(stale=True)
  assert after_recovery.hasApply
  assert after_recovery.applySequence == recovered["applySequence"]
  assert after_recovery.sendcanMonoTime == recovered["sendcanMonoTime"]
  assert after_recovery.context.session == before_failure.context.session
  assert sum(service == "sendcan" for service, _ in harness.card.pm.history) == 2


@pytest.mark.parametrize("failure", ["invalidStart", "endBeforeStart", "sendcanBeforeEnd"])
def test_real_observer_rejects_invalid_apply_timing_and_recovers_without_changing_actuation(failure):
  baseline = CardHarness(enabled=False)
  harness = CardHarness()
  for current in (baseline, harness):
    current.step()
    current.command.actuators.accel = -0.5
  observer = harness.card.longitudinal_observation
  previous = observer.receipt
  errors = observer.recorder.errors
  now_ns = int(((harness.now_ns + 10_000_000) / 1e9) * 1e9)
  # Real clock reads: publish start/end, applying, apply start/end,
  # sendcanPending and sendcanPublished. The sendcan clock stays unmodified.
  ticks = [now_ns] * 7
  if failure == "invalidStart":
    ticks[3] = -1
    del ticks[4]  # An absent begin result has no completion clock read.
  elif failure == "endBeforeStart":
    ticks[4] = now_ns - 1
  else:
    ticks[4] = now_ns + 1
  clock = Mock(side_effect=ticks)
  baseline.step()
  before_failure = harness.step(observation_clock=clock)
  assert clock.call_count == len(ticks)
  assert before_failure.hasApply
  assert before_failure.applySequence == previous["applySequence"]
  assert observer.receipt is None
  assert observer.apply_sequence == 2
  assert observer.recorder.errors == errors + 1
  assert len(harness.requests) == len(harness.sent_frames) == 2
  assert sum(service == "sendcan" for service, _ in harness.card.pm.history) == 2

  baseline.step(stale=True)
  after_failure = harness.step(stale=True)
  assert not after_failure.hasApply
  assert after_failure.applySequence == after_failure.sendcanMonoTime == 0
  assert after_failure.lastDisposition == "sendcanPublished"
  assert after_failure.context.errors == errors + 1
  assert observer.receipt is None

  baseline.step()
  assert not harness.step().hasApply
  recovered = observer.receipt
  assert recovered["applySequence"] == 3
  assert recovered["applyStartMonoTime"] <= recovered["applyEndMonoTime"] <= recovered["sendcanMonoTime"]
  assert recovered["sendcanMonoTime"] > previous["sendcanMonoTime"]
  baseline.step(stale=True)
  after_recovery = harness.step(stale=True)
  assert after_recovery.hasApply
  assert after_recovery.applySequence == recovered["applySequence"]
  assert after_recovery.sendcanMonoTime == recovered["sendcanMonoTime"]
  assert after_recovery.context.session == before_failure.context.session
  assert after_recovery.context.errors == errors + 1
  assert baseline.requests == harness.requests
  assert baseline.sent_frames == harness.sent_frames
  assert message_payloads(baseline) == message_payloads(harness)


def test_reconstructed_observer_restarts_sequences_in_a_distinct_session_without_reusing_a_receipt():
  with patch("openpilot.common.longitudinal_logging.secrets.randbits", side_effect=(101, 202)) as sessions:
    harness = CardHarness()
    harness.step()
    before_restart = harness.step()
    previous = harness.card.longitudinal_observation
    previous_receipt = previous.receipt
    assert previous.apply_sequence == previous.recorder.sequence == 2
    assert before_restart.hasApply and before_restart.applySequence == 1
    command_id = harness.card.sm.logMonoTime["carControl"]

    replacement = ActuationObservation(harness.card.CP, harness.card.CP_SP_capnp, enabled=True, clock=lambda: harness.now_ns)
    harness.card.longitudinal_observation = replacement
  assert sessions.call_count == 2
  assert replacement is not previous
  assert replacement.recorder.session == 202 != previous.recorder.session == 101
  assert replacement.apply_sequence == replacement.recorder.sequence == 0
  assert replacement.receipt is replacement.local_state is replacement.local_state_frame is None

  first = harness.step(fresh_control=False)
  assert not first.hasApply
  assert first.applySequence == first.sendcanMonoTime == 0
  assert first.lastDisposition == "unobserved"
  assert first.context.sequence == 1
  assert first.context.session == replacement.recorder.session
  assert first.context.manifest
  assert first.context.identity == before_restart.context.identity
  second = harness.step(fresh_control=False)
  assert second.hasApply
  assert second.context.sequence == before_restart.context.sequence == 2
  assert second.applySequence == before_restart.applySequence == 1
  assert second.context.session != before_restart.context.session
  assert second.carControlMonoTime == command_id
  assert second.sendcanMonoTime > previous_receipt["sendcanMonoTime"]
  assert second.controllerFrame == 2
  assert previous.receipt is previous_receipt
  assert previous.apply_sequence == previous.recorder.sequence == 2


@pytest.mark.parametrize("pedal", ["gasPressed", "brakePressed"])
@pytest.mark.parametrize("edge_parity", [0, 1])
def test_logging_on_off_preserves_actual_honda_100hz_apply_and_50hz_acc_can(pedal, edge_parity, tmp_path):
  marker = tmp_path / "card_logging"
  with patch.dict(LOGGING_MARKERS, {"card": marker}):
    baseline = CardHarness(enabled=None, native=True)
    marker.touch()
    enabled = CardHarness(enabled=None, native=True)
  assert baseline.card.longitudinal_observation is None
  assert enabled.card.longitudinal_observation.recorder.enabled
  for frame in range(100):
    for harness in (baseline, enabled):
      if frame < 20 + edge_parity:
        harness.command.actuators.accel = -2.0
        harness.command.actuators.longControlState = "stopping"
        harness.command.cruiseControl.resume = False
      elif frame < 60:
        setattr(harness.state, pedal, True)
      else:
        setattr(harness.state, pedal, False)
        harness.command.actuators.accel = 0.5
        harness.command.actuators.longControlState = "pid"
        harness.command.cruiseControl.resume = True
      harness.step()
  assert len(baseline.requests) == len(enabled.requests) == 100
  assert baseline.requests == enabled.requests
  assert baseline.sent_frames == enabled.sent_frames
  assert message_payloads(baseline) == message_payloads(enabled)
  assert all(event.carOutput.longitudinalDiagnostics.context.version == 0 for event in baseline.outputs)
  acc_address = enabled.chain.acc_parser.dbc.name_to_msg["ACC_CONTROL"].address
  assert sum(any(address == acc_address for address, _, _ in frames) for frames in enabled.sent_frames) == 50
  diagnostics = [event.carOutput.longitudinalDiagnostics for event in enabled.outputs]
  assert [entry.applySequence for entry in diagnostics if entry.hasApply] == list(range(1, 100))
  assert [entry.controllerFrame for entry in diagnostics if entry.hasApply] == list(range(99))
  assert all(entry.canDigest == can_digest(enabled.sent_frames[index])
             for index, entry in enumerate(diagnostics[1:]))
  assert all(entry.controllerFrameKnown for entry in diagnostics[1:])


def test_can_encoder_explicit_creation_timestamp_is_exact_and_default_bytes_are_unchanged():
  frames = [(0x1DF, b"\x00\x12\xFE", 0), (0xE4, b"\x0F\xFF", 1)]
  timestamp = (1 << 53) + 37
  with patch("openpilot.selfdrive.pandad.pandad_api_impl.time.monotonic", return_value=12.5):
    default = can_list_to_can_capnp(frames, msgtype="sendcan", valid=False)
    supplied = can_list_to_can_capnp(frames, msgtype="sendcan", valid=False, log_mono_time=12_500_000_000)
    exact = can_list_to_can_capnp(frames, msgtype="sendcan", log_mono_time=timestamp)
  assert default == supplied
  assert can_capnp_to_list([exact], msgtype="sendcan") == [(timestamp, frames)]
  assert messaging.log_from_bytes(default).valid is False
