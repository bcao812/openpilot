"""Keep the stock ECU in service until the Accord Hybrid CAN handoff is ready."""

from opendbc.car.honda.values import CAR, HondaFlags
from opendbc.car.structs import car


class HondaStartupGate:
  MAX_AGE_NS = 250_000_000
  STABLE_NS = 300_000_000

  @staticmethod
  def applies(params):
    return (params.carFingerprint == CAR.HONDA_ACCORD and params.openpilotLongitudinalControl
            and bool(params.flags & HondaFlags.HYBRID) and not params.passive)

  def __init__(self, params):
    self.safety_model = str(params.safetyConfigs[0].safetyModel)
    self.safety_param = int(params.safetyConfigs[0].safetyParam)
    self.alternative_experience = int(params.alternativeExperience)
    self.ready = False
    self.can_initialize = False
    self.resuming = False
    self.handoff_ns = None
    self.last_ns = None
    self.identity = None
    self.last_errors = None
    self.stable_since = None
    self.start_rx = None
    self.reason = 'waitingForPanda'

  def _reset_health(self, reason):
    self.can_initialize = False
    self.stable_since = None
    self.start_rx = None
    self.reason = reason

  def update(self, pandas, panda_ns, now_ns, *, valid, alive, control_ns, control_alive, control_enabled):
    if self.ready:
      return  # This is a startup handoff, not a new runtime fault policy.

    if not (valid and alive and len(pandas) == 1 and 0 < panda_ns <= now_ns
            and now_ns - panda_ns <= self.MAX_AGE_NS):
      self._reset_health('waitingForFreshPanda')
      return
    panda = pandas[0]
    harness = str(panda.harnessStatus)
    if harness not in ('normal', 'flipped'):
      self._reset_health('waitingForHarness')
      return
    configured = (str(panda.safetyModel) == self.safety_model and panda.safetyParam == self.safety_param
                  and panda.alternativeExperience == self.alternative_experience)
    if self.handoff_ns is None:
      # A restarted card must not wait with an already disabled radar or send
      # diagnostic session commands in the running Honda safety mode.
      self.resuming = configured
      self.can_initialize = configured

    disabled = (control_alive and not control_enabled and 0 < control_ns <= now_ns
                and now_ns - control_ns <= self.MAX_AGE_NS)
    if not disabled and not self.resuming:
      self.can_initialize = False

    if self.last_ns is not None and panda_ns <= self.last_ns:
      if panda_ns < self.last_ns:
        self._reset_health('pandaClockChanged')
        self.last_ns = None
      return  # Re-reading one 10 Hz health sample cannot qualify a handoff.

    identity = (harness, int(panda.uptime))
    camera = panda.canState0 if harness == 'flipped' else panda.canState2
    buses = (panda.canState1, camera)
    errors = tuple(int(bus.totalErrorCnt) for bus in buses)
    rx = tuple(int(bus.totalRxCnt) for bus in buses)
    reset = (self.last_ns is not None and panda_ns - self.last_ns > self.MAX_AGE_NS)
    reset |= (self.identity is not None and (identity[0] != self.identity[0] or identity[1] < self.identity[1]))
    reset |= self.last_errors is not None and errors != self.last_errors
    self.last_ns, self.identity, self.last_errors = panda_ns, identity, errors

    # PT carries actuation; camera is the forwarding destination implicated in
    # the startup stall. The radar-side TX error can be left by normal FW queries
    # until the existing Honda safety switch resets it: do not wait on that latch.
    healthy = (not panda.faults and not panda.heartbeatLost
               and all(not (bus.busOff or bus.errorPassive or bus.errorWarning) for bus in buses))
    if not healthy or reset:
      self._reset_health('waitingForStableCan')
      self.can_initialize = self.resuming and self.handoff_ns is None
      return
    if self.stable_since is None:
      self.stable_since, self.start_rx = panda_ns, rx
    stable = (panda_ns - self.stable_since >= self.STABLE_NS
              and all(current > first for current, first in zip(rx, self.start_rx, strict=True)))
    if self.handoff_ns is None:
      self.can_initialize = self.resuming or (stable and disabled and str(panda.safetyModel) == 'elm327')
      self.reason = 'readyToInitialize' if self.can_initialize else 'waitingForStableCan'
    elif stable and configured and panda_ns > self.handoff_ns and disabled and control_ns > self.handoff_ns:
      self.ready = True
      self.reason = 'ready'
    else:
      self.reason = 'waitingForSafetyHandoff'

  def begin_handoff(self, now_ns):
    self.handoff_ns = now_ns
    self.can_initialize = False
    self.reason = 'waitingForSafetyHandoff'


def inactive_startup_control(control):
  restricted = control.as_builder()
  restricted.enabled = False
  restricted.latActive = False
  restricted.longActive = False
  restricted.actuators = car.CarControl.Actuators.new_message()
  restricted.cruiseControl.resume = False
  restricted.cruiseControl.override = False
  restricted.cruiseControl.cancel = False
  return restricted.as_reader()
