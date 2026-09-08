import importlib.util
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, mock_open, patch

from opendbc.car.structs import car


def load_beep():
  module_path = Path(__file__).resolve().parents[1] / 'beep.py'
  spec = importlib.util.spec_from_file_location('_beep_test', module_path)
  module = importlib.util.module_from_spec(spec)
  imports = {
    'openpilot.cereal': SimpleNamespace(messaging=SimpleNamespace()),
    'openpilot.cereal.messaging': SimpleNamespace(),
    'openpilot.common.realtime': SimpleNamespace(Ratekeeper=object),
  }
  with patch.dict('sys.modules', imports):
    spec.loader.exec_module(module)
  return module


beep = load_beep()
AudibleAlert = car.CarControl.HUDControl.AudibleAlert


class RecordingBeepd(beep.Beepd):
  def __init__(self):
    self.transitions = []
    self.transition_condition = threading.Condition()
    super().__init__()

  def enable_gpio(self):
    pass

  def _write_gpio(self, on):
    with self.transition_condition:
      self.transitions.append(bool(on))
      self.transition_condition.notify_all()

  def _legacy_beep(self, on):
    self._write_gpio(on)

  def wait_for_transitions(self, count):
    with self.transition_condition:
      return self.transition_condition.wait_for(lambda: len(self.transitions) >= count, timeout=1)


class TestBeepPWM(unittest.TestCase):
  def setUp(self):
    self.driver = RecordingBeepd()
    self.addCleanup(self.driver.close)

  def test_pwm_duty_cycle(self):
    self.assertEqual(beep.BEEP_DUTY_CYCLE, 0.6)
    self.assertEqual(beep.BEEP_PWM_PERIOD, 0.002)
    with patch.object(self.driver._pwm_condition, 'wait_for', wraps=self.driver._pwm_condition.wait_for) as wait:
      self.driver._beep(True)
      self.assertTrue(self.driver.wait_for_transitions(6))
      self.driver.close()
      timeouts = [arguments.kwargs['timeout'] for arguments in wait.call_args_list if 'timeout' in arguments.kwargs]
    self.assertEqual(timeouts[:4], [0.0012, 0.0008] * 2)

  def test_direct_gpio_writes(self):
    with patch('builtins.open', mock_open()) as gpio:
      beep.Beepd._write_gpio(self.driver, True)
      beep.Beepd._write_gpio(self.driver, False)
      self.assertEqual(gpio.call_args_list, [call('/sys/class/gpio/gpio42/value', 'w')] * 2)
      self.assertEqual(gpio().write.call_args_list, [call('1'), call('0')])

  def test_idle_worker_does_not_write(self):
    time.sleep(0.02)
    self.assertEqual(self.driver.transitions, [])

  def test_pwm_repeats_and_stops_low(self):
    self.driver._beep(True)
    self.assertTrue(self.driver.wait_for_transitions(6))
    self.driver._beep(False)
    with self.driver.transition_condition:
      self.assertTrue(self.driver.transition_condition.wait_for(lambda: not self.driver.transitions[-1], timeout=1))
    time.sleep(0.02)
    transitions = list(self.driver.transitions)
    time.sleep(0.02)
    self.assertEqual(self.driver.transitions, transitions)
    self.assertEqual(transitions[:6], [True, False] * 3)
    self.assertFalse(transitions[-1])

  def test_stop_interrupts_high_phase(self):
    with patch.object(beep, 'BEEP_PWM_PERIOD', 1):
      self.driver._beep(True)
      self.assertTrue(self.driver.wait_for_transitions(1))
      started = time.monotonic()
      self.driver._beep(False)
      self.assertTrue(self.driver.wait_for_transitions(2))
      self.assertLess(time.monotonic() - started, 0.2)
      self.assertFalse(self.driver.transitions[-1])

  def test_close_stops_worker_and_prevents_reactivation(self):
    self.driver._beep(True)
    self.assertTrue(self.driver.wait_for_transitions(1))
    self.driver.close()
    self.assertFalse(self.driver._pwm_thread.is_alive())
    self.assertFalse(self.driver.transitions[-1])
    transitions = list(self.driver.transitions)
    self.driver._beep(True)
    self.assertEqual(self.driver.transitions, transitions)

  def test_pwm_failure_falls_back_to_original_driver(self):
    fallback_started = threading.Event()
    with patch.object(self.driver, '_write_gpio', side_effect=PermissionError), \
         patch.object(self.driver, '_legacy_beep', side_effect=lambda on: fallback_started.set()) as fallback:
      self.driver._beep(True)
      self.assertTrue(fallback_started.wait(1))
      self.driver._beep(False)
      self.driver._beep(True)
      self.driver.close()
      self.assertEqual(fallback.call_args_list, [call(True), call(False), call(True), call(False)])
      self.assertFalse(self.driver._pwm_thread.is_alive())

  def test_gpio_failure_during_close_still_releases_buzzer(self):
    with patch.object(self.driver, '_write_gpio', side_effect=OSError), patch.object(self.driver, '_legacy_beep') as fallback:
      self.driver.close()
      fallback.assert_called_once_with(False)
      self.assertFalse(self.driver._pwm_thread.is_alive())

  def test_rapid_transitions_leave_one_worker_and_end_low(self):
    worker = self.driver._pwm_thread
    for _ in range(100):
      self.driver._beep(True)
      self.driver._beep(False)
    self.driver._beep(True)
    self.assertTrue(self.driver.wait_for_transitions(2))
    self.driver.close()
    self.assertIs(self.driver._pwm_thread, worker)
    self.assertFalse(worker.is_alive())
    self.assertFalse(self.driver.transitions[-1])


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
        self.assertEqual(output.call_args_list, [call(on) for on in transitions])
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
      with self.subTest(alert=alert), patch.object(self.driver, 'dispatch_beep') as dispatch:
        self.driver.update_alert(alert)
        dispatch.assert_called_once_with(getattr(self.driver, pattern))
    with patch.object(self.driver, 'dispatch_beep') as dispatch, patch.object(self.driver, 'warning_immediate') as warning:
      self.driver.update_alert(AudibleAlert.warningImmediate)
      dispatch.call_args.args[0]()
      warning.assert_called_once_with(self.driver._repeat_stop)
    with patch.object(self.driver, 'dispatch_beep') as dispatch:
      self.driver.update_alert(AudibleAlert.none)
      dispatch.assert_not_called()

  def test_immediate_warning_repeats_and_cancels_low(self):
    bursts = []
    with patch.object(self.driver, '_beep', side_effect=lambda on: bursts.append(on)):
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


if __name__ == '__main__':
  unittest.main()
