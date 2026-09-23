import queue
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, call

import pytest

from openpilot.cereal import log
from openpilot.common import params as params_module
from openpilot.common.params import Params
from openpilot.selfdrive.selfdrived import alertmanager
from openpilot.system.hardware import hardwared


TEMPERATURE_ALERT = 'Offroad_TemperatureTooHigh'
SUPPORT_ALERT = 'Offroad_TiciSupport'


@pytest.fixture
def alert_params(monkeypatch, tmp_path):
  params = Params(str(tmp_path))
  monkeypatch.setattr(hardwared, 'prev_offroad_states', {})
  monkeypatch.setattr(hardwared, 'Params', lambda: params)
  monkeypatch.setattr(alertmanager, 'Params', lambda: params)
  return params


@pytest.fixture
def alert_writer(monkeypatch, alert_params):
  writer = Mock(wraps=alertmanager.set_offroad_alert)
  monkeypatch.setattr(hardwared, 'set_offroad_alert', writer)
  return writer


def test_first_hidden_update_clears_once_despite_changing_text(alert_writer, alert_params):
  alertmanager.set_offroad_alert(TEMPERATURE_ALERT, True, 'stale')
  hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, False, '40.0C')
  for extra_text in ('41.0C', '42.0C', None, ''):
    hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, False, extra_text)
  alert_writer.assert_called_once_with(TEMPERATURE_ALERT, False, '40.0C')
  assert alert_params.get(TEMPERATURE_ALERT) is None


def test_absent_hidden_alert_does_not_need_a_remove(alert_writer, alert_params):
  for extra_text in ('40.0C', '41.0C', None, ''):
    hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, False, extra_text)
  alert_writer.assert_not_called()
  assert alert_params.get(TEMPERATURE_ALERT) is None


def test_visibility_transitions_and_visible_text_changes_are_written(alert_writer):
  alertmanager.set_offroad_alert(TEMPERATURE_ALERT, True, 'stale')
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


def test_empty_cache_after_restart_checks_storage_before_clearing(alert_writer, monkeypatch):
  hardwared.set_offroad_alert_if_changed(SUPPORT_ALERT, False, 'c3xl-dev')
  monkeypatch.setattr(hardwared, 'prev_offroad_states', {})
  hardwared.set_offroad_alert_if_changed(SUPPORT_ALERT, False, 'c3xl-dev')
  alert_writer.assert_not_called()

  alertmanager.set_offroad_alert(SUPPORT_ALERT, True, 'stale')
  monkeypatch.setattr(hardwared, 'prev_offroad_states', {})
  hardwared.set_offroad_alert_if_changed(SUPPORT_ALERT, False, 'c3xl-dev')
  alert_writer.assert_called_once_with(SUPPORT_ALERT, False, 'c3xl-dev')


def test_hidden_alert_recreated_externally_is_cleared_again(alert_writer, alert_params):
  for _ in range(2):
    alertmanager.set_offroad_alert(SUPPORT_ALERT, True, 'external writer')
    hardwared.set_offroad_alert_if_changed(SUPPORT_ALERT, False, 'c3xl-dev')
    assert alert_params.get(SUPPORT_ALERT) is None
    hardwared.set_offroad_alert_if_changed(SUPPORT_ALERT, False, 'c3xl-dev')
  assert alert_writer.call_args_list == [call(SUPPORT_ALERT, False, 'c3xl-dev')] * 2


def test_alert_keys_have_independent_cache_entries(alert_writer):
  for alert in (TEMPERATURE_ALERT, SUPPORT_ALERT, TEMPERATURE_ALERT, SUPPORT_ALERT):
    hardwared.set_offroad_alert_if_changed(alert, True, 'same text')
  assert alert_writer.call_args_list == [
    call(TEMPERATURE_ALERT, True, 'same text'),
    call(SUPPORT_ALERT, True, 'same text'),
  ]


@pytest.mark.parametrize('previous,requested', [
  (None, (False, '40.0C')),
  (None, (True, '100.0C')),
  ((False, '40.0C'), (True, '100.0C')),
  ((True, '100.0C'), (False, '90.0C')),
  ((True, '100.0C'), (True, '101.0C')),
])
def test_write_exception_propagates_and_manual_retry_succeeds(alert_writer, alert_params, previous, requested):
  if previous is not None:
    hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, *previous)
  elif not requested[0]:
    alertmanager.set_offroad_alert(TEMPERATURE_ALERT, True, 'stale')
  alert_writer.reset_mock()
  alert_writer.side_effect = OSError('write failed')

  with pytest.raises(OSError, match='write failed'):
    hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, *requested)
  assert TEMPERATURE_ALERT not in hardwared.prev_offroad_states

  alert_writer.side_effect = None
  hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, *requested)
  hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, *requested)
  assert alert_writer.call_args_list == [call(TEMPERATURE_ALERT, *requested)] * 2
  expected = dict(alertmanager.OFFROAD_ALERTS[TEMPERATURE_ALERT], extra=requested[1]) if requested[0] else None
  assert alert_params.get(TEMPERATURE_ALERT) == expected


@pytest.mark.parametrize('alert', [TEMPERATURE_ALERT, SUPPORT_ALERT])
def test_silent_remove_failure_is_retried_until_the_real_key_is_gone(alert_writer, alert_params, monkeypatch, alert):
  alertmanager.set_offroad_alert(alert, True, 'stale')
  previous = alert_params.get(alert)
  real_remove = params_module.params_remove
  remove = Mock(return_value=-1)
  monkeypatch.setattr(params_module, 'params_remove', remove)

  # The real Params.remove() swallows this negative C binding result, without raising.
  for _ in range(2):
    assert hardwared.set_offroad_alert_if_changed(alert, False, 'hidden') is None
    assert alert_params.get(alert) == previous
    assert alert not in hardwared.prev_offroad_states
  assert remove.call_count == 2

  remove.side_effect = real_remove
  hardwared.set_offroad_alert_if_changed(alert, False, 'hidden')
  assert alert_params.get(alert) is None
  hardwared.set_offroad_alert_if_changed(alert, False, 'changed hidden text')
  assert remove.call_count == 3
  assert alert_writer.call_count == 3


@pytest.mark.parametrize('previous', [None, '100.0C', 'wrong template'])
def test_silent_put_failure_requires_full_readback_before_caching(alert_writer, alert_params, monkeypatch, previous):
  if previous == 'wrong template':
    # Matching extra text alone is insufficient to confirm the alert payload.
    alert_params.put(TEMPERATURE_ALERT, {'text': 'stale template', 'extra': '101.0C'}, block=True)
  elif previous is not None:
    hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, True, previous)
  previous_value = alert_params.get(TEMPERATURE_ALERT)
  alert_writer.reset_mock()
  real_put = params_module.params_put
  put = Mock(return_value=-1)
  monkeypatch.setattr(params_module, 'params_put', put)

  # This exercises Params.put() and the real store, with failure injected below the Python API.
  for _ in range(2):
    assert hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, True, '101.0C') is None
    assert alert_params.get(TEMPERATURE_ALERT) == previous_value
    assert TEMPERATURE_ALERT not in hardwared.prev_offroad_states
  assert put.call_count == 2

  put.side_effect = real_put
  hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, True, '101.0C')
  expected = dict(alertmanager.OFFROAD_ALERTS[TEMPERATURE_ALERT], extra='101.0C')
  assert alert_params.get(TEMPERATURE_ALERT) == expected
  assert hardwared.prev_offroad_states[TEMPERATURE_ALERT] == (True, '101.0C')
  hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, True, '101.0C')
  assert put.call_count == 3
  assert alert_writer.call_count == 3


@pytest.mark.parametrize('contents', [b'', b'not json'])
def test_hidden_alert_clears_existing_key_without_decoding_json(alert_writer, alert_params, contents):
  path = Path(alert_params.get_param_path(TEMPERATURE_ALERT))
  path.write_bytes(contents)
  hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, False)
  assert not path.exists()
  hardwared.set_offroad_alert_if_changed(TEMPERATURE_ALERT, False)
  alert_writer.assert_called_once_with(TEMPERATURE_ALERT, False, None)


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
  for alert in (SUPPORT_ALERT, TEMPERATURE_ALERT):
    alertmanager.set_offroad_alert(alert, True, 'stale')

  state = MagicMock()
  state.frame = 0
  state.updated = {'pandaStates': False, 'selfdriveState': False}
  state.recv_time = {'pandaStates': 1000.0}
  state.alive = {'chestnutState': False}
  state.__getitem__.side_effect = {'pandaStates': [], 'peripheralState': None, 'chestnutState': None}.__getitem__
  monkeypatch.setattr(hardwared.messaging, 'SubMaster', Mock(return_value=state))
  monkeypatch.setattr(hardwared.messaging, 'PubMaster', Mock())

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
