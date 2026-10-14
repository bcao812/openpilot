from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from opendbc.car.honda.values import CAR, HondaFlags
from opendbc.car.structs import car
from openpilot.cereal import log, messaging
from openpilot.selfdrive.car.honda_startup import HondaStartupGate, inactive_startup_control
from openpilot.selfdrive.car.tests.test_actuation_observation import CardHarness
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import bounded_submaster, serialized


def params():
  cp = car.CarParams.new_message(carFingerprint=CAR.HONDA_ACCORD, openpilotLongitudinalControl=True,
                                flags=int(HondaFlags.BOSCH | HondaFlags.HYBRID), alternativeExperience=1024)
  cp.safetyConfigs = [{'safetyModel': 'hondaBosch', 'safetyParam': 2}]
  return cp


def health(*, mode='elm327', harness='normal', rx=100, errors=0):
  bus = lambda: SimpleNamespace(busOff=False, errorPassive=False, errorWarning=False, totalErrorCnt=errors, totalRxCnt=rx)
  return SimpleNamespace(harnessStatus=harness, safetyModel=mode, safetyParam=2 if mode == 'hondaBosch' else 1,
                         alternativeExperience=1024 if mode == 'hondaBosch' else 0, uptime=100,
                         faults=[], heartbeatLost=False, canState0=bus(), canState1=bus(), canState2=bus())


def update(gate, index, panda=None, **kwargs):
  now = 10_000_000_000 + index * 100_000_000
  panda = health(rx=100+index) if panda is None else panda
  options = dict(valid=True, alive=True, control_ns=now-1_000_000, control_alive=True, control_enabled=False)
  options.update(kwargs)
  gate.update([panda], now, now, **options)


def qualify(gate, **kwargs):
  for index in range(4):
    update(gate, index, **kwargs)


def test_platform_scope():
  cp = params()
  assert HondaStartupGate.applies(cp)
  for field, value in [('openpilotLongitudinalControl', False), ('passive', True),
                       ('flags', int(HondaFlags.BOSCH)), ('carFingerprint', CAR.HONDA_CIVIC)]:
    other = cp.as_reader().as_builder()
    setattr(other, field, value)
    assert not HondaStartupGate.applies(other)


def test_fresh_health_then_confirmed_safety_and_disabled_host():
  gate = HondaStartupGate(params())
  qualify(gate)
  assert gate.can_initialize and not gate.ready
  gate.begin_handoff(10_310_000_000)
  update(gate, 4)  # Panda has not switched out of diagnostic mode yet.
  assert not gate.ready
  update(gate, 5, health(mode='hondaBosch', rx=105), control_enabled=True)
  assert not gate.ready  # An old enable request cannot become active after the wait.
  update(gate, 6, health(mode='hondaBosch', rx=106))
  assert gate.ready


@pytest.mark.parametrize('field', ['busOff', 'errorPassive', 'errorWarning'])
@pytest.mark.parametrize('bus', [1, 2])
def test_current_pt_or_camera_errors_never_time_out_into_handoff(field, bus):
  gate = HondaStartupGate(params())
  for index in range(120):
    panda = health(rx=100+index)
    setattr(getattr(panda, f'canState{bus}'), field, True)
    update(gate, index, panda)
  assert not gate.can_initialize and not gate.ready


def test_error_counter_changes_reset_window_but_history_does_not_lock_out():
  gate = HondaStartupGate(params())
  for index in range(5):
    update(gate, index, health(rx=100+index, errors=28_000+index))
    assert not gate.can_initialize
  for index in range(5, 9):
    update(gate, index, health(rx=100+index, errors=28_004))
  assert gate.can_initialize


def test_duplicate_stale_missing_and_nonreceiving_health():
  gate = HondaStartupGate(params())
  panda = health()
  for index in range(30):
    now = 10_000_000_000 + index * 10_000_000
    gate.update([panda], 10_000_000_000, now, valid=True, alive=True,
                control_ns=now, control_alive=True, control_enabled=False)
  assert not gate.can_initialize
  for index in range(4, 9):
    update(gate, index, panda)  # Health publishes, but no CAN frames arrive.
  assert not gate.can_initialize
  update(gate, 9, valid=False)
  assert not gate.can_initialize
  gate.update([], 11_000_000_000, 11_000_000_000, valid=True, alive=True,
              control_ns=11_000_000_000, control_alive=True, control_enabled=False)
  assert not gate.can_initialize


def test_normal_fw_query_radar_tx_latch_does_not_block_camera_readiness():
  gate = HondaStartupGate(params())
  for index in range(4):
    panda = health(rx=100+index)
    panda.canState0.errorPassive = True
    panda.canState0.errorWarning = True
    update(gate, index, panda)
  assert gate.can_initialize


@pytest.mark.parametrize('bus', [1, 2])
@pytest.mark.parametrize('stalled_rx', [101, 99])
def test_one_rx_increment_then_stall_or_rollback_cannot_qualify_when_control_disables(bus, stalled_rx):
  gate = HondaStartupGate(params())
  update(gate, 0, control_enabled=True)
  update(gate, 1, control_enabled=True)
  for index in range(2, 31):
    panda = health(rx=100+index)
    getattr(panda, f'canState{bus}').totalRxCnt = stalled_rx
    update(gate, index, panda, control_enabled=index < 30)
    assert not gate.can_initialize
  for index in range(31, 35):
    update(gate, index)
  assert gate.can_initialize  # Recovery needs a full window with fresh reception.


def test_flipped_harness_uses_physical_can_zero_for_camera():
  gate = HondaStartupGate(params())
  for index in range(4):
    panda = health(harness='flipped', rx=100+index)
    panda.canState0.errorPassive = True
    update(gate, index, panda)
  assert not gate.can_initialize
  for index in range(4, 8):
    panda = health(harness='flipped', rx=100+index)
    panda.canState2.errorPassive = True  # Radar side in this orientation.
    update(gate, index, panda)
  assert gate.can_initialize


def test_started_handoff_requires_matching_safety_and_fresh_disabled_control():
  gate = HondaStartupGate(params())
  qualify(gate)
  gate.begin_handoff(10_310_000_000)
  panda = health(mode='hondaBosch', rx=105)
  panda.safetyParam = 0
  update(gate, 5, panda)
  assert not gate.ready
  panda.safetyParam = 2
  panda.canState1.totalRxCnt = panda.canState2.totalRxCnt = 106
  update(gate, 6, panda, control_ns=9_000_000_000)
  assert not gate.ready
  update(gate, 7, health(mode='hondaBosch', rx=107))
  assert gate.ready
  gate.update([], 0, 20_000_000_000, valid=False, alive=False,
              control_ns=20_000_000_000, control_alive=True, control_enabled=True)
  assert gate.ready  # Runtime faults retain their existing handling.


def test_card_restart_can_continue_inactive_can_without_disabling_ecu_again():
  gate = HondaStartupGate(params())
  panda = health(mode='hondaBosch')
  panda.canState2.errorPassive = True
  update(gate, 0, panda, control_enabled=True)
  assert gate.can_initialize and gate.resuming and not gate.ready


def test_inactive_startup_control_clears_both_axes_and_keeps_input_immutable():
  command = car.CarControl.new_message(enabled=True, latActive=True, longActive=True)
  command.actuators.accel = 1.0
  command.actuators.torque = 0.5
  command.actuators.longControlState = 'stopping'
  command.cruiseControl.resume = True
  command.cruiseControl.override = True
  before = command.to_dict()
  safe = inactive_startup_control(command.as_reader())
  assert not safe.enabled and not safe.latActive and not safe.longActive
  assert safe.actuators.accel == safe.actuators.torque == 0.0
  assert str(safe.actuators.longControlState) == 'off'
  assert not safe.cruiseControl.resume and not safe.cruiseControl.override
  assert command.to_dict() == before


class StartupHarness(CardHarness):
  def __init__(self):
    super().__init__(enabled=False, native=True)
    self.card.CP.flags |= int(HondaFlags.HYBRID)
    self.card.CP.safetyConfigs = [{'safetyModel': 'hondaBosch', 'safetyParam': 2}]
    self.card.CP.alternativeExperience = 1024
    self.card.honda_startup_gate = HondaStartupGate(self.card.CP)
    self.card.initialized_prev = False
    self.events['pandaStates'] = messaging.new_message('pandaStates', 1, valid=True)
    self.panda = self.events['pandaStates'].pandaStates[0]
    self.panda.harnessStatus = 'normal'
    self.panda.safetyModel = 'elm327'
    self.panda.safetyParam = 1
    self.card.sm = bounded_submaster(list(self.events))
    self.card.sm.update = Mock()
    self.frames = 0
    self.command.enabled = self.command.latActive = self.command.longActive = False

  def configure_panda(self):
    self.panda.safetyModel = self.card.CP.safetyConfigs[0].safetyModel
    self.panda.safetyParam = self.card.CP.safetyConfigs[0].safetyParam
    self.panda.alternativeExperience = self.card.CP.alternativeExperience

  def initializing(self, enabled):
    self.events['onroadEvents'] = messaging.new_message('onroadEvents', int(enabled), valid=True)
    if enabled:
      self.events['onroadEvents'].onroadEvents[0].name = log.OnroadEvent.EventName.selfdriveInitializing

  def feed(self, *, fresh_control=True, timestamp=None):
    self.now_ns += 10_000_000
    self.state.cruiseState.available = True  # A newly decoded CAN state on each step.
    incoming = []
    for index, (service, event) in enumerate(self.events.items()):
      if service == 'pandaStates':
        if self.frames % 10:
          continue
        for bus in (self.panda.canState1, self.panda.canState2):
          bus.totalRxCnt += 100
      if service == 'carControl' and not fresh_control:
        continue
      event.logMonoTime = self.now_ns - (index+1)*1_000_003
      incoming.append(serialized(event))
    self.card.sm.update_msgs(self.now_ns/1e9, incoming)
    self.frames += 1


def test_real_card_waits_then_hands_off_once_and_keeps_inactive_can_flowing():
  h = StartupHarness()
  h.panda.canState2.errorPassive = True
  h.command.enabled = h.command.latActive = h.command.longActive = True
  h.events['carControlSP'].carControlSP.intelligentCruiseButtonManagement.sendButton = 'increase'
  for _ in range(1100):
    h.step()
  assert not h.card.initialized_prev
  h.card.CI.init.assert_not_called()
  h.card.params.put_bool.assert_not_called()
  assert not h.requests and not h.sent_frames
  assert not h.card.pm.messages['carState'].carState.cruiseState.available

  h.panda.canState2.errorPassive = False
  for _ in range(50):
    h.step()
  h.card.CI.init.assert_not_called()  # Cached active request must first be withdrawn.
  h.command.enabled = h.command.latActive = h.command.longActive = False
  for _ in range(20):
    h.step()
  h.card.CI.init.assert_called_once()
  h.card.params.put_bool.assert_called_once_with('ControlsReady', True)
  assert h.requests and all(not r[0]['enabled'] and not r[0]['latActive'] and not r[0]['longActive'] for r in h.requests)
  assert any(frame[0] == 0x18DAB0F1 for batch in h.sent_frames for frame in batch)
  assert not h.card.honda_startup_gate.ready

  h.panda.safetyModel = h.card.CP.safetyConfigs[0].safetyModel
  h.panda.safetyParam = h.card.CP.safetyConfigs[0].safetyParam
  h.panda.alternativeExperience = h.card.CP.alternativeExperience
  for _ in range(20):
    h.step()
  assert h.card.honda_startup_gate.ready
  assert h.card.pm.messages['carState'].carState.cruiseState.available
  h.command.enabled = h.command.latActive = h.command.longActive = True
  h.step()
  assert h.requests[-1][0]['longActive']
  h.card.CI.init.assert_called_once()


def test_real_card_restart_preserves_keepalive_and_skips_ecu_knockout():
  h = StartupHarness()
  h.configure_panda()
  h.panda.canState2.errorPassive = True
  for _ in range(21):
    h.step()
  h.card.CI.init.assert_not_called()
  assert len(h.sent_frames) == 21
  assert sum(frame[0] == 0x18DAB0F1 for batch in h.sent_frames for frame in batch) == 3
  assert all(not r[0]['enabled'] and not r[0]['longActive'] for r in h.requests)


@pytest.mark.parametrize('phase', ['handoff', 'ready', 'card_restart'])
@pytest.mark.parametrize('interruption', ['initializing', 'stale_control', 'invalid_control'])
def test_real_card_control_restart_keeps_inactive_can_and_requires_fresh_disabled_request(phase, interruption):
  h = StartupHarness()
  if phase == 'card_restart':
    h.configure_panda()
  else:
    for _ in range(41):
      h.step()
    assert h.card.honda_startup_gate.handoff_ns is not None
    h.card.CI.init.assert_called_once()
    if phase == 'ready':
      h.configure_panda()
      for _ in range(20):
        h.step()
      assert h.card.honda_startup_gate.ready

  h.command.enabled = h.command.latActive = h.command.longActive = True
  if interruption == 'initializing':
    h.initializing(True)
  elif interruption == 'invalid_control':
    h.events['carControl'].valid = False
  first_request = len(h.requests)
  for _ in range(120):
    h.step(stale=interruption == 'stale_control')
  assert len(h.requests) == first_request+120
  for command, _, _ in h.requests[first_request:]:
    assert not command['enabled'] and not command['latActive'] and not command['longActive']
    assert command['actuators']['accel'] == command['actuators']['torque'] == 0.0
  assert all(extension['intelligentCruiseButtonManagement']['sendButton'] == 'none'
             for _, extension, _ in h.requests[first_request:])
  batches = h.sent_frames[first_request:]
  assert sum(frame[0] == 0x18DAB0F1 for batch in batches for frame in batch) == 12
  assert sum(frame[0] == 0x1DF for batch in batches for frame in batch) == 60
  assert sum(frame[0] == 0x1EF for batch in batches for frame in batch) == 60
  assert all(any(frame[0] == 0xE4 for frame in batch) for batch in batches)
  assert not h.card.honda_startup_gate.ready

  h.initializing(False)
  h.events['carControl'].valid = True
  h.configure_panda()
  for _ in range(50):
    h.step()
  assert not h.card.honda_startup_gate.ready
  assert all(not command[0]['enabled'] for command in h.requests[first_request:])
  h.command.enabled = h.command.latActive = h.command.longActive = False
  for _ in range(50):
    h.step()
  assert h.card.honda_startup_gate.ready
  h.command.enabled = h.command.latActive = h.command.longActive = True
  h.step()
  assert h.requests[-1][0]['longActive']
  assert h.card.CI.init.call_count == int(phase != 'card_restart')
  h.card.params.put_bool.assert_called_once_with('ControlsReady', True)


@pytest.mark.parametrize('axis', ['latActive', 'longActive'])
def test_real_card_requires_both_axes_disabled_before_ecu_handoff(axis):
  h = StartupHarness()
  setattr(h.command, axis, True)
  for _ in range(50):
    h.step()
  h.card.CI.init.assert_not_called()
  assert not h.requests
  setattr(h.command, axis, False)
  for _ in range(20):
    h.step()
  h.card.CI.init.assert_called_once()
