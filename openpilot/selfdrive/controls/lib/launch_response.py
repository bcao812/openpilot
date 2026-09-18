"""Bounded acceleration-error feedback inside an already admitted launch.

The owner supplies release/tracking eligibility; this helper cannot grant it.
KI=1 is a declared engineering-plant candidate, not a Honda road calibration.
No grade, minimum GAS value or long-horizon vehicle prediction is assumed.
"""

import math


LAUNCH_RESPONSE_KI = 1.0  # 1/s; candidate for the frozen short-response engineering tests


class LaunchResponse:
  def __init__(self, *, max_gap_ns):
    if type(max_gap_ns) is not int or max_gap_ns <= 0:
      raise ValueError('Invalid launch feedback clock bound')
    self.max_gap_ns = max_gap_ns
    self.reset()

  def reset(self):
    self.bias = 0.0
    self.producer_instance_ns = 0
    self._last_now_ns = None
    self._pending = None

  @property
  def armed(self):
    return self.producer_instance_ns > 0

  def _advance(self, producer_instance_ns, now_ns, *, arm=False):
    if (type(producer_instance_ns) is not int or type(now_ns) is not int
        or not 0 < producer_instance_ns <= now_ns):
      self.reset()
      return None
    if producer_instance_ns != self.producer_instance_ns:
      self.reset()
      if not arm:
        return None
      self.producer_instance_ns = producer_instance_ns
      self._last_now_ns = now_ns
      return 0.0
    elapsed_ns = now_ns - self._last_now_ns
    if not 0 < elapsed_ns <= self.max_gap_ns:
      self.reset()
      return None
    self._last_now_ns = now_ns
    return elapsed_ns / 1e9

  def freeze_release(self, producer_instance_ns, now_ns, *, arm):
    """A qualified owner RELEASE may arm; its brake-removal ramp never learns."""
    self._pending = None
    if self._advance(producer_instance_ns, now_ns, arm=arm) is not None:
      # RELEASE belongs to a new launch, not to a previous tracking correction.
      self.bias = 0.0

  def propose(self, base_output, a_target, a_ego, upper_limit, *, producer_instance_ns, now_ns, forward):
    """Propose TRACK effort; commit only its part in the final limited request.

    FORWARD can learn in either direction. A subsequent MOVING reference only
    retains or unwinds the already learned correction. It cannot create more
    drive effort from a flat reference, and crossing a speed threshold does not
    discard the correction. STOP/OFF/invalid input is revoked by the caller.
    """
    self._pending = None
    values = (base_output, a_target, a_ego, upper_limit, self.bias)
    if (not all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in values)
        or a_target < 0.0 or upper_limit < 0.0 or type(forward) is not bool):
      self.reset()
      return base_output
    elapsed = self._advance(producer_instance_ns, now_ns)
    if elapsed is None:
      return base_output
    error = a_target - a_ego
    change = LAUNCH_RESPONSE_KI * elapsed * error
    if not math.isfinite(change):
      self.reset()
      return base_output
    if not forward:
      change = min(change, 0.0)
    headroom = max(0.0, upper_limit - base_output)
    # A reduced limit also removes old stored bias before any new error update.
    candidate = min(headroom, max(0.0, min(self.bias, headroom) + change))
    self._pending = (base_output, candidate)
    return base_output + candidate

  def confirm_output(self, output_accel):
    """Back-calculate after all LongControl limits, including the upward ramp.

    This confirms a requested command, not measured force or a CAN delivery.
    A blocked positive increment is never saved for a later unsaturated frame.
    """
    pending, self._pending = self._pending, None
    if pending is None:
      return
    if not isinstance(output_accel, (int, float)) or isinstance(output_accel, bool) or not math.isfinite(output_accel):
      self.reset()
      return
    base_output, candidate = pending
    self.bias = min(candidate, max(0.0, output_accel - base_output))
