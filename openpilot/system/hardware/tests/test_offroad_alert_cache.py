import queue
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest

from openpilot.cereal import log
from openpilot.system.hardware import hardwared


TEMPERATURE_ALERT = 'Offroad_TemperatureTooHigh'
SUPPORT_ALERT = 'Offroad_TiciSupport'


@pytest.fixture
def alert_writer(monkeypatch):
  monkeypatch.setattr(hardwared, 'prev_offroad_states', {})
  writer = Mock()
  monkeypatch.setattr(hardwared, 'set_offroad_alert', writer)
  return writer


def test_first_hidden_update_clears_once_despite_changing_text(alert_writer):
  hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, False, '40.0C')
  for extra_text in ('41.0C', '42.0C', None, ''):
    hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, False, extra_text)
  alert_writer.assert_called_once_with(TEMPERATURE_ALERT, False, '40.0C')


def test_visibility_transitions_and_visible_text_changes_are_written(alert_writer):
  for show_alert, extra_text in (
    (False, '40.0C'), (True, '100.0C'), (True, '100.0C'), (True, '101.0C'),
    (False, '90.0C'), (False, '89.0C'), (True, '102.0C'),
  ):
    hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, show_alert, extra_text)
  assert alert_writer.call_args_list == [
    call(TEMPERATURE_ALERT, False, '40.0C'),
    call(TEMPERATURE_ALERT, True, '100.0C'),
    call(TEMPERATURE_ALERT, True, '101.0C'),
    call(TEMPERATURE_ALERT, False, '90.0C'),
    call(TEMPERATURE_ALERT, True, '102.0C'),
  ]


def test_empty_cache_after_restart_clears_hidden_alert_again(alert_writer, monkeypatch):
  hardwared.set_offroad_alert_if_changed(SUPPORT_ALERT, False, 'c3xl-dev')
  hardwared.set_offroad_alert_if_changed(SUPPORT_ALERT, False, 'c3xl-dev')
  monkeypatch.setattr(hardwared, 'prev_offroad_states', {})
  hardwared.set_offroad_alert_if_changed(SUPPORT_ALERT, False, 'c3xl-dev')
  assert alert_writer.call_args_list == [call(SUPPORT_ALERT, False, 'c3xl-dev')] * 2


def test_alert_keys_have_independent_cache_entries(alert_writer):
  for alert in (TEMPERATURE_ALERT, SUPPORT_ALERT, TEMPERATURE_ALERT, SUPPORT_ALERT):
    hardwared.set_offroad_alert_if_changed(alert, False, 'same text')
  assert alert_writer.call_args_list == [
    call(TEMPERATURE_ALERT, False, 'same text'),
    call(SUPPORT_ALERT, False, 'same text'),
  ]


@pytest.mark.parametrize('previous,requested', [
  (None, (False, '40.0C')),
  ((False, '40.0C'), (True, '100.0C')),
  ((True, '100.0C'), (False, '90.0C')),
  ((True, '100.0C'), (True, '101.0C')),
])
def test_failed_write_does_not_cache_success_or_suppress_retry(alert_writer, previous, requested):
  if previous is not None:
    hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, *previous)
  previous_cache = hardwared.prev_offroad_states.copy()
  alert_writer.reset_mock()
  alert_writer.side_effect = [OSError('write failed'), None]

  with pytest.raises(OSError, match='write failed'):
    hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, *requested)
  assert hardwared.prev_offroad_states == previous_cache

  hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, *requested)
  hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, *requested)
  assert alert_writer.call_args_list == [call(TEMPERATURE_ALERT, *requested)] * 2


def test_hardware_thread_routes_support_and_temperature_alerts_through_cache(alert_writer, monkeypatch):
  # Stop immediately after the real alert call sites; power/onroad actions are outside this fixture.
  class AlertsReached(Exception):
    pass

  helper = Mock(wraps=hardwared.set_offroad_alert_if_changed)

  def stop_after_alerts(alert, show_alert, extra_text=None):
    helper(alert, show_alert, extra_text=extra_text)
    if alert == TEMPERATURE_ALERT:
      raise AlertsReached

  monkeypatch.setattr(hardwared, 'set_offroad_alert_if_changed', stop_after_alerts)

  state = MagicMock()
  state.frame = 0
  state.updated = {'pandaStates': False, 'selfdriveState': False}
  state.recv_time = {'pandaStates': 1000.0}
  state.alive = {'chestnutState': False}
  state.__getitem__.side_effect = {'pandaStates': [], 'peripheralState': None, 'chestnutState': None}.__getitem__
  monkeypatch.setattr(hardwared.messaging, 'SubMaster', Mock(return_value=state))
  monkeypatch.setattr(hardwared.messaging, 'PubMaster', Mock())

  params = Mock()
  params.get.return_value = None
  params.get_bool.return_value = False
  monkeypatch.setattr(hardwared, 'Params', Mock(return_value=params))
  system_stats = Mock()
  system_stats.memory_usage_percent.return_value = 10.0
  system_stats.cpu_usage_percent.return_value = [0.0]
  monkeypatch.setattr(hardwared, 'LinuxSystemStats', Mock(return_value=system_stats))

  hardware = Mock()
  hardware.get_device_type.return_value = 'tici'
  hardware.get_gpu_usage_percent.return_value = 0.0
  hardware.get_screen_brightness.return_value = 50.0
  hardware.booted.return_value = True
  monkeypatch.setattr(hardwared, 'HARDWARE', hardware)
  monkeypatch.setattr(hardwared, 'PowerMonitoring', Mock())
  monkeypatch.setattr(hardwared, 'FanController', Mock(return_value=SimpleNamespace(update=lambda *_: 0)))
  monkeypatch.setattr(hardwared, 'Chestnut', Mock())
  monkeypatch.setattr(hardwared, 'ChestnutStatus', Mock())
  monkeypatch.setattr(hardwared, 'get_short_branch', lambda: 'c3xl-dev')
  monkeypatch.setattr(hardwared, 'get_build_metadata', lambda: SimpleNamespace(channel='c3xl-dev'))
  monkeypatch.setattr(hardwared, 'get_available_percent', lambda **_: 90.0)
  monkeypatch.setattr(hardwared, 'set_usb_state', Mock())
  monkeypatch.setattr(hardwared.time, 'monotonic', lambda: 1000.0)

  for temperature in (40.0, 50.0):
    thermal_message = log.DeviceState.new_message(cpuTempC=[temperature], memoryTempC=temperature)
    hardware.get_thermal_config.return_value.get_msg.return_value = thermal_message
    end_event = Mock()
    end_event.is_set.side_effect = [False, True]
    with pytest.raises(AlertsReached):
      hardwared.hardware_thread(end_event, queue.Queue())

  assert helper.call_args_list == [
    call(SUPPORT_ALERT, False, extra_text='c3xl-dev'), call(TEMPERATURE_ALERT, False, extra_text='40.0C'),
    call(SUPPORT_ALERT, False, extra_text='c3xl-dev'), call(TEMPERATURE_ALERT, False, extra_text='50.0C'),
  ]
  assert alert_writer.call_args_list == [
    call(SUPPORT_ALERT, False, 'c3xl-dev'), call(TEMPERATURE_ALERT, False, '40.0C'),
  ]
