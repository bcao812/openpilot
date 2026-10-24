import importlib.util
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, mock_open, patch

from opendbc.car.structs import car
from openpilot.common.sound_mode import SoundMode, get_sound_mode


class MemoryParams:
  def __init__(self, value=SoundMode.ALL):
    self.value = value

  def get(self, key, return_default=False):
    return self.value


def load_beep():
  module_path = Path(__file__).resolve().parents[1] / 'beep.py'
  spec = importlib.util.spec_from_file_location('_beep_test', module_path)
  module = importlib.util.module_from_spec(spec)
  imports = {
    'openpilot.cereal': SimpleNamespace(messaging=SimpleNamespace()),
    'openpilot.cereal.messaging': SimpleNamespace(),
    'openpilot.common.params': SimpleNamespace(Params=MemoryParams),
    'openpilot.common.realtime': SimpleNamespace(Ratekeeper=object),
  }
  with patch.dict('sys.modules', imports):
    spec.loader.exec_module(module)
  return module


beep = load_beep()
AudibleAlert = car.CarControl.HUDControl.AudibleAlert


class RecordingBeepd(beep.Beepd):
  def __init__(self, params=None):
    self.transitions = []
    self.transition_condition = threading.Condition()
    super().__init__(params)
    self.startup_transitions = list(self.transitions)
    self.transitions.clear()

  def enable_gpio(self):
    pass

  def _write_gpio(self, on):
    with self.transition_condition:
      self.transitions.append(bool(on))
      self.transition_condition.notify_all()

  def _legacy_beep(self, on):
    self._write_gpio(on)


class TestBeepGPIO(unittest.TestCase):
  def setUp(self):
    self.driver = RecordingBeepd()
    self.addCleanup(self.driver.close)

  def test_on_holds_high_until_off(self):
    self.driver._beep(True)
    self.assertEqual(self.driver.transitions, [True])
    with self.driver.transition_condition:
      self.assertFalse(self.driver.transition_condition.wait_for(lambda: len(self.driver.transitions) > 1, timeout=0.02))
    self.driver._beep(False)
    self.assertEqual(self.driver.transitions, [True, False])

  def test_direct_gpio_writes(self):
    with patch('builtins.open', mock_open()) as gpio:
      beep.Beepd._write_gpio(self.driver, True)
      beep.Beepd._write_gpio(self.driver, False)
      self.assertEqual(gpio.call_args_list, [call('/sys/class/gpio/gpio42/value', 'w')] * 2)
      self.assertEqual(gpio().write.call_args_list, [call('1'), call('0')])

  def test_idle_does_not_write(self):
    time.sleep(0.02)
    self.assertEqual(self.driver.transitions, [])

  def test_rapid_transitions_keep_every_edge(self):
    for _ in range(100):
      self.driver._beep(True)
      self.driver._beep(False)
    self.assertEqual(self.driver.transitions, [True, False] * 100)

  def test_close_waits_for_inflight_output_and_ends_low(self):
    high_started = threading.Event()
    release_high = threading.Event()
    close_started = threading.Event()
    close_finished = threading.Event()
    transitions = []

    def write(on):
      if on:
        high_started.set()
        release_high.wait()
      transitions.append(on)

    def close():
      close_started.set()
      self.driver.close()
      close_finished.set()

    writer = threading.Thread(target=self.driver._beep, args=(True,), daemon=True)
    closer = threading.Thread(target=close, daemon=True)
    with patch.object(self.driver, '_write_gpio', side_effect=write):
      writer.start()
      try:
        self.assertTrue(high_started.wait(1))
        closer.start()
        self.assertTrue(close_started.wait(1))
        self.assertFalse(close_finished.wait(0.02))
      finally:
        release_high.set()
        writer.join(1)
        if closer.ident is not None:
          closer.join(1)
      self.assertFalse(writer.is_alive())
      self.assertFalse(closer.is_alive())
      self.assertTrue(close_finished.is_set())
      self.driver._beep(True)
    self.assertEqual(transitions, [True, False])

  def test_close_releases_buzzer_and_prevents_reactivation(self):
    self.driver._beep(True)
    self.driver.close()
    self.assertTrue(self.driver._repeat_stop.is_set())
    self.driver._beep(True)
    self.driver._beep(False)
    self.driver.close()
    self.assertEqual(self.driver.transitions, [True, False])

  def test_direct_gpio_failure_falls_back_to_original_driver(self):
    with patch.object(self.driver, '_write_gpio', side_effect=PermissionError) as direct, \
         patch.object(self.driver, '_legacy_beep') as fallback:
      self.driver._beep(True)
      self.driver._beep(False)
      self.driver._beep(True)
      self.driver.close()
      direct.assert_called_once_with(True)
      self.assertEqual(fallback.call_args_list, [call(True), call(False), call(True), call(False)])

  def test_gpio_failure_during_close_still_releases_buzzer(self):
    self.driver._beep(True)
    with patch.object(self.driver, '_write_gpio', side_effect=OSError), patch.object(self.driver, '_legacy_beep') as fallback:
      self.driver.close()
      self.driver._beep(True)
      fallback.assert_called_once_with(False)

  def test_gpio_failure_turning_off_still_releases_buzzer(self):
    self.driver._beep(True)
    with patch.object(self.driver, '_write_gpio', side_effect=OSError) as direct, \
         patch.object(self.driver, '_legacy_beep') as fallback:
      self.driver._beep(False)
      self.driver.close()
      direct.assert_called_once_with(False)
      self.assertEqual(fallback.call_args_list, [call(False), call(False)])


class TestBeepAlerts(unittest.TestCase):
  def setUp(self):
    self.driver = RecordingBeepd()
    self.addCleanup(self.driver.close)

  def test_short_patterns_keep_their_original_timing(self):
    patterns = (
      (self.driver.engage, [True, False], [0.05]),
      (self.driver.disengage, [True, False] * 2, [0.01] * 4),
      (self.driver.warning, [True, False] * 3, [0.01] * 6),
    )
    for pattern, transitions, delays in patterns:
      with self.subTest(pattern=pattern.__name__), patch.object(self.driver, '_beep') as output, patch.object(beep.time, 'sleep') as sleep:
        pattern()
        self.assertEqual(output.call_args_list, [call(on, None) for on in transitions])
        self.assertEqual(sleep.call_args_list, [call(delay) for delay in delays])

  def test_all_audible_alerts_remain_mapped(self):
    patterns = {
      AudibleAlert.engage: 'engage',
      AudibleAlert.disengage: 'disengage',
      AudibleAlert.refuse: 'warning',
      AudibleAlert.warningSoft: 'warning',
      AudibleAlert.prompt: 'warning',
      AudibleAlert.promptRepeat: 'warning',
      AudibleAlert.promptDistracted: 'warning',
    }
    for alert, pattern in patterns.items():
      with self.subTest(alert=alert), patch.object(self.driver, 'dispatch_beep') as dispatch, patch.object(self.driver, pattern) as play:
        self.driver.update_alert(alert)
        dispatch.assert_called_once()
        dispatch.call_args.args[0]()
        play.assert_called_once_with(self.driver._repeat_stop)
    with patch.object(self.driver, 'dispatch_beep') as dispatch, patch.object(self.driver, 'warning_immediate') as warning:
      self.driver.update_alert(AudibleAlert.warningImmediate)
      dispatch.call_args.args[0]()
      warning.assert_called_once_with(self.driver._repeat_stop)
    with patch.object(self.driver, 'dispatch_beep') as dispatch:
      self.driver.update_alert(AudibleAlert.none)
      dispatch.assert_not_called()

  def test_immediate_warning_repeats_and_cancels_low(self):
    bursts = []
    with patch.object(self.driver, '_beep', side_effect=lambda on, stop=None: bursts.append(on)):
      self.driver.update_alert(AudibleAlert.warningImmediate)
      stop = self.driver._repeat_stop
      time.sleep(1.35)
      self.assertGreaterEqual(bursts.count(True), 5)
      self.driver.update_alert(AudibleAlert.none)
      self.assertTrue(stop.is_set())
      time.sleep(0.05)
      self.assertFalse(bursts[-1])
      burst_count = len(bursts)
      time.sleep(0.05)
      self.assertEqual(len(bursts), burst_count)


class TestSoundModes(unittest.TestCase):
  def setUp(self):
    self.params = MemoryParams()
    self.driver = RecordingBeepd(self.params)
    self.addCleanup(self.driver.close)

  def test_missing_and_invalid_values_preserve_all_sounds(self):
    for value in (None, -1, 3, 'invalid'):
      with self.subTest(value=value):
        self.assertEqual(get_sound_mode(MemoryParams(value)), SoundMode.ALL)

  def test_all_modes_start_with_gpio_low(self):
    for mode in SoundMode:
      with self.subTest(mode=mode):
        driver = RecordingBeepd(MemoryParams(mode))
        try:
          self.assertEqual(driver.startup_transitions, [False])
          self.assertEqual(driver.sound_mode, mode)
        finally:
          driver.close()

  def test_mode_matrix_filters_dispatch_and_gpio(self):
    alerts = (AudibleAlert.engage, AudibleAlert.disengage, AudibleAlert.refuse, AudibleAlert.warningSoft,
              AudibleAlert.warningImmediate, AudibleAlert.prompt, AudibleAlert.promptRepeat, AudibleAlert.promptDistracted)
    warnings = set(alerts) - {AudibleAlert.engage, AudibleAlert.disengage}
    for mode in SoundMode:
      self.params.value = mode
      with patch.object(self.driver, 'dispatch_beep') as dispatch:
        self.driver.update_alert(AudibleAlert.none)
        for alert in alerts:
          with self.subTest(mode=mode, alert=alert):
            dispatch.reset_mock()
            allowed = mode == SoundMode.ALL or (mode == SoundMode.WARNING and alert in warnings)
            self.driver.update_alert(alert)
            self.assertEqual(dispatch.call_count, int(allowed))
            self.driver._beep(True, self.driver._repeat_stop)
            self.assertEqual(self.driver.transitions[-1], allowed)
            self.driver._beep(False, self.driver._repeat_stop)

  def test_none_stops_active_warning_without_a_new_message(self):
    workers = []

    def dispatch(pattern):
      worker = threading.Thread(target=pattern, daemon=True)
      workers.append(worker)
      worker.start()

    with patch.object(self.driver, 'dispatch_beep', side_effect=dispatch):
      try:
        self.driver.update_alert(AudibleAlert.warningImmediate)
        stop = self.driver._repeat_stop
        with self.driver.transition_condition:
          self.assertTrue(self.driver.transition_condition.wait_for(lambda: True in self.driver.transitions, timeout=1))
        self.params.value = SoundMode.NONE
        self.driver.get_audible_alert(SimpleNamespace(updated={'selfdriveState': False}))
        self.assertTrue(stop.is_set())
        for worker in workers:
          worker.join(1)
        self.assertTrue(all(not worker.is_alive() for worker in workers))
        self.assertFalse(self.driver.transitions[-1])
      finally:
        self.driver.close()
        for worker in workers:
          worker.join(1)
      self.assertTrue(all(not worker.is_alive() for worker in workers))
      self.assertFalse(self.driver.transitions[-1])

  def test_mode_changes_retire_old_threads_and_resume_current_warning(self):
    sm = SimpleNamespace(updated={'selfdriveState': False})
    with patch.object(self.driver, 'dispatch_beep') as dispatch:
      self.driver.update_alert(AudibleAlert.warningImmediate)
      retired = self.driver._repeat_stop
      self.driver._beep(True, retired)
      self.params.value = SoundMode.NONE
      self.driver.get_audible_alert(sm)
      self.assertFalse(self.driver.transitions[-1])
      self.driver._beep(True, retired)
      self.driver._beep(True, self.driver._repeat_stop)
      self.assertFalse(self.driver.transitions[-1])
      self.assertEqual(dispatch.call_count, 1)

      self.params.value = SoundMode.WARNING
      self.driver.get_audible_alert(sm)
      self.assertEqual(dispatch.call_count, 2)
      self.driver._beep(True, self.driver._repeat_stop)
      self.assertTrue(self.driver.transitions[-1])
      transitions = list(self.driver.transitions)
      self.driver._beep(False, retired)
      self.driver._beep(True, retired)
      self.driver.get_audible_alert(sm)
      self.assertEqual(self.driver.transitions, transitions)
      self.assertEqual(dispatch.call_count, 2)

  def test_warning_mode_cancels_engagement_and_allows_new_warning(self):
    with patch.object(self.driver, 'dispatch_beep') as dispatch:
      self.driver.update_alert(AudibleAlert.engage)
      retired = self.driver._repeat_stop
      self.driver._beep(True, retired)
      self.params.value = SoundMode.WARNING
      self.driver.get_audible_alert(SimpleNamespace(updated={'selfdriveState': False}))
      self.driver._beep(True, retired)
      self.assertFalse(self.driver.transitions[-1])
      self.assertEqual(dispatch.call_count, 1)
      self.driver.update_alert(AudibleAlert.refuse)
      self.driver._beep(True, self.driver._repeat_stop)
      self.assertTrue(self.driver.transitions[-1])
      self.assertEqual(dispatch.call_count, 2)

  def test_unmute_uses_new_message_without_replaying_previous_alert(self):
    self.params.value = SoundMode.NONE
    with patch.object(self.driver, 'dispatch_beep') as dispatch:
      self.driver.update_alert(AudibleAlert.warningImmediate)
      self.params.value = SoundMode.ALL
      sm = MagicMock()
      sm.updated = {'selfdriveState': True}
      sm.__getitem__.return_value = SimpleNamespace(alertSound=SimpleNamespace(raw=AudibleAlert.none))
      self.driver.get_audible_alert(sm)
      dispatch.assert_not_called()
      self.assertFalse(self.driver.transitions[-1])


if __name__ == '__main__':
  unittest.main()
