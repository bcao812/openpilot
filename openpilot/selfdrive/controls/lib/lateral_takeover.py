"""Keep a permanent EPS takeover until MADS has disabled and explicitly rearmed."""

MADS_STATE_MAX_AGE_NS = 100_000_000
ACTIVE_MADS_STATES = ("enabled", "softDisabling", "overriding")


def _fresh(timestamp_ns: int, now_ns: int) -> bool:
  return type(timestamp_ns) is int and 0 < timestamp_ns <= now_ns and now_ns - timestamp_ns <= MADS_STATE_MAX_AGE_NS


class LateralTakeoverLatch:
  def __init__(self):
    self._authorized = False
    self._fault_ns = 0
    self._disabled_ns = 0
    self._last_mads = None
    self._last_car_state_ns = 0
    self._last_now_ns = 0

  @property
  def latched(self) -> bool:
    return self._fault_ns != 0

  def update(self, requested_active: bool, *, enabled: bool, mads_available: bool,
             mads_state: str, mads_active: bool, permanent_fault: bool, temporary_fault: bool,
             mads_mono_ns: int, car_state_mono_ns: int, now_ns: int, inputs_valid: bool) -> bool:
    if not enabled:
      return requested_active

    clock_valid = type(now_ns) is int and now_ns > 0 and now_ns >= self._last_now_ns
    if permanent_fault and self._authorized and not self.latched:
      # The caller has already applied its EPS mask to requested_active. Use
      # the earlier actual passage, and never renew this boundary each frame.
      fresh_fault = clock_valid and _fresh(car_state_mono_ns, now_ns) and car_state_mono_ns > self._last_car_state_ns
      self._fault_ns = car_state_mono_ns if fresh_fault else now_ns if clock_valid else self._last_now_ns
      self._disabled_ns = 0
      self._authorized = False
    if not clock_valid:
      return False
    self._last_now_ns = now_ns

    if not (inputs_valid and _fresh(car_state_mono_ns, now_ns) and _fresh(mads_mono_ns, now_ns)):
      return False
    if car_state_mono_ns < self._last_car_state_ns:
      return False

    mads = (mads_mono_ns, mads_available, mads_state, mads_active)
    if self._last_mads is not None:
      if mads_mono_ns < self._last_mads[0] or (mads_mono_ns == self._last_mads[0] and mads != self._last_mads):
        return False
    new_mads = self._last_mads is None or mads_mono_ns > self._last_mads[0]
    self._last_mads = mads
    self._last_car_state_ns = car_state_mono_ns

    # Availability disappearing is not a disabled acknowledgement or rearm.
    if not mads_available:
      return False

    if new_mads and mads_state == "disabled" and not mads_active:
      self._authorized = False
      if self.latched and mads_mono_ns > self._fault_ns:
        self._disabled_ns = mads_mono_ns

    if (self.latched and self._disabled_ns and new_mads and mads_mono_ns > self._disabled_ns
        and mads_state == "enabled" and mads_active and not permanent_fault and not temporary_fault
        and car_state_mono_ns > self._fault_ns):
      self._fault_ns = 0
      self._disabled_ns = 0

    active = bool(requested_active and not self.latched and not permanent_fault and not temporary_fault
                  and mads_active and mads_state in ACTIVE_MADS_STATES)
    if active:
      self._authorized = True
    return active
