#!/usr/bin/env python3
import subprocess
import time
import threading

import openpilot.cereal.messaging as messaging
from opendbc.car.structs import car
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.sound_mode import SoundMode, get_sound_mode

AudibleAlert = car.CarControl.HUDControl.AudibleAlert
WARNING_ALERTS = frozenset((AudibleAlert.refuse, AudibleAlert.warningSoft, AudibleAlert.prompt,
                          AudibleAlert.promptRepeat, AudibleAlert.promptDistracted, AudibleAlert.warningImmediate))


class Beepd:
  def __init__(self, params=None):
    self.params = Params() if params is None else params
    self.sound_mode = get_sound_mode(self.params)
    self.current_alert = AudibleAlert.none
    # Every pattern owns a token, so a retired thread cannot affect a new alert.
    self._repeat_stop = threading.Event()
    self.enable_gpio()
    self._gpio_lock = threading.Lock()
    self._closed = False
    self._gpio_failed = False
    with self._gpio_lock:
      self._set_gpio(False)
    #self.startup_beep()

  def enable_gpio(self):
    # 尝试 export，忽略已 export 的错误
    try:
      subprocess.run("echo 42 | sudo tee /sys/class/gpio/export",
                     shell=True,
                     stderr=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL,
                     encoding='utf8')
    except Exception:
      pass
    subprocess.run("echo \"out\" | sudo tee /sys/class/gpio/gpio42/direction",
                   shell=True,
                   stderr=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL,
                   encoding='utf8')

  def _write_gpio(self, on):
    with open('/sys/class/gpio/gpio42/value', 'w') as gpio:
      gpio.write('1' if on else '0')

  def _legacy_beep(self, on):
    val = "1" if on else "0"
    subprocess.run(f"echo \"{val}\" | sudo tee /sys/class/gpio/gpio42/value",
                   shell=True,
                   stderr=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL,
                   encoding='utf8')

  def _set_gpio(self, on):
    # Call with _gpio_lock held so shutdown cannot race a pulse turning on.
    if not self._gpio_failed:
      try:
        self._write_gpio(on)
        return
      except OSError as error:
        self._gpio_failed = True
        print(f"[BEEP] Direct GPIO unavailable, using legacy GPIO: {error}")
    self._legacy_beep(on)

  def _should_play_alert(self, alert):
    return self.sound_mode == SoundMode.ALL or (self.sound_mode == SoundMode.WARNING and alert in WARNING_ALERTS)

  def _beep(self, on, stop=None):
    with self._gpio_lock:
      if self._closed or (stop is not None and stop is not self._repeat_stop):
        return
      if on and (not self._should_play_alert(self.current_alert) or (stop is not None and stop.is_set())):
        return
      # Keep the GPIO steady for the whole pulse; timing belongs to the patterns.
      self._set_gpio(bool(on))

  def close(self):
    self._repeat_stop.set()
    with self._gpio_lock:
      if self._closed:
        return
      self._closed = True
      self._set_gpio(False)

  def engage(self, stop=None):
    self._beep(True, stop)
    time.sleep(0.05)
    self._beep(False, stop)

  def disengage(self, stop=None):
    for _ in range(2):
      self._beep(True, stop)
      time.sleep(0.01)
      self._beep(False, stop)
      time.sleep(0.01)

  def warning(self, stop=None):
    for _ in range(3):
      self._beep(True, stop)
      time.sleep(0.01)
      self._beep(False, stop)
      time.sleep(0.01)

  def warning_immediate(self, stop):
    # warningImmediate is a takeover request: it persists for as long as its condition does, and
    # soundd -- which this replaces -- plays it on a loop. Sounding once on the transition, the
    # way every other pattern here does, would answer the highest-severity alert in the set with
    # one burst and then silence. So this repeats until the alert changes.
    #
    # Longer pulses than warning() so the two are told apart by ear rather than by counting.
    # stop.wait() rather than time.sleep() throughout: the pre-emption has to be felt inside a
    # pulse, not only between bursts.
    try:
      while not stop.is_set():
        for _ in range(4):
          if stop.is_set():
            return
          self._beep(True, stop)
          if stop.wait(0.12):
            return
          self._beep(False, stop)
          if stop.wait(0.08):
            return
        stop.wait(0.4)
    finally:
      # A current pattern releases its output; update_alert releases retired ones
      # before publishing the replacement token, so they cannot clip a new pulse.
      self._beep(False, stop)

  #def startup_beep(self):
    #self._beep(True)
    #time.sleep(0.1)
    #self._beep(False)

  def dispatch_beep(self, func):
    threading.Thread(target=func, daemon=True).start()

  def update_alert(self, new_alert):
    sound_mode = get_sound_mode(self.params)
    with self._gpio_lock:
      if self._closed or (new_alert == self.current_alert and sound_mode == self.sound_mode):
        return
      self.current_alert = new_alert
      self.sound_mode = sound_mode
      print(f"[BEEP] Alert: {new_alert}, sound: {sound_mode.name}")

      # Stop output immediately on a mode/alert change. Retired threads cannot
      # restart a muted pulse or release a newer alert's GPIO output.
      self._repeat_stop.set()
      self._set_gpio(False)
      self._repeat_stop = threading.Event()
      stop = self._repeat_stop
      if not self._should_play_alert(new_alert):
        return

      # Exhaustive over car.capnp's AudibleAlert. It was not: warningImmediate and promptRepeat
      # fell through to nothing, and with soundd disabled (process_config.py) this is the only
      # audible channel on the device, so they were silent rather than merely different.
      # promptDistracted cannot currently fire -- DISABLE_DRIVER=1 stops dmonitoringd -- but it
      # is wired anyway so repairing the cabin camera does not quietly re-open the same hole.
      if new_alert == AudibleAlert.engage:
        pattern = self.engage
      elif new_alert == AudibleAlert.disengage:
        pattern = self.disengage
      elif new_alert == AudibleAlert.warningImmediate:
        pattern = self.warning_immediate
      elif new_alert in (AudibleAlert.refuse, AudibleAlert.warningSoft, AudibleAlert.prompt,
                         AudibleAlert.promptRepeat, AudibleAlert.promptDistracted):
        pattern = self.warning
      else:
        return
    self.dispatch_beep(lambda: pattern(stop))

  def get_audible_alert(self, sm):
    new_alert = sm['selfdriveState'].alertSound.raw if sm.updated['selfdriveState'] else self.current_alert
    # Poll the preference even without a new message, so None stops active audio.
    self.update_alert(new_alert)

  def test_beepd_thread(self):
    frame = 0
    rk = Ratekeeper(20)
    pm = messaging.PubMaster(['selfdriveState'])
    while True:
      cs = messaging.new_message('selfdriveState')
      if frame == 20:
        cs.selfdriveState.alertSound = AudibleAlert.engage
      if frame == 40:
        cs.selfdriveState.alertSound = AudibleAlert.disengage
      if frame == 60:
        cs.selfdriveState.alertSound = AudibleAlert.prompt
      if frame == 80:
        cs.selfdriveState.alertSound = AudibleAlert.disengage
      if frame == 85:
        cs.selfdriveState.alertSound = AudibleAlert.prompt

      pm.send("selfdriveState", cs)
      frame += 1
      rk.keep_time()

  def beepd_thread(self, test=False):
    if test:
      threading.Thread(target=self.test_beepd_thread, daemon=True).start()

    sm = messaging.SubMaster(['selfdriveState'])
    rk = Ratekeeper(20)

    while True:
      sm.update(0)
      self.get_audible_alert(sm)
      rk.keep_time()

def main():
  beepd = Beepd()
  try:
    beepd.beepd_thread(test=False)
  finally:
    beepd.close()

if __name__ == "__main__":
  main()
