"""One timed acceleration correction owned by LongControl's existing PID state.

The owner supplies TRACK/RELEASE eligibility; this helper cannot grant it.
Only pid.i accumulates. Launch and rolling tracking choose mutually exclusive
gains and permissions, then commit only correction present in the final output.
"""

import math


LAUNCH_RESPONSE_KI = 1.0  # 1/s; candidate for the frozen short-response engineering tests
# At 100 Hz the frozen-reference delayed first-order plant has characteristic
# z^(d-1)*(z-1)*(z-(1-r)) + r*Ki*dt. Ki=.4 gives >=49.4 deg phase margin for
# delay<=1 s, response<=.8 s; coupled MPC/mode changes require separate tests.
# This is an engineering envelope, not identified Honda response or calibration.
TRACKING_RESPONSE_KI = 0.4


class LaunchResponse:
  def __init__(self, pid, *, max_gap_ns):
    if type(max_gap_ns) is not int or max_gap_ns <= 0:
      raise ValueError('Invalid launch feedback clock bound')
    self.max_gap_ns = max_gap_ns
    self._pid = pid
    self.reset()

  def reset(self):
    self._pid.i = 0.0
    self.producer_instance_ns = 0
    self._domain = None
    self._positive_authorized = False
    self._last_now_ns = None
    self._pending = None

  @property
  def armed(self):
    return self._domain == 'launch'

  @property
  def braking_handoff(self):
    """Expired negative I was handed to the owner's current braking output."""
    return self._domain == 'approach_handoff'

  @property
  def bias(self):
    """Diagnostic alias; no second stored integral or launch correction."""
    return float(self._pid.i)

  def _advance(self, producer_instance_ns, now_ns, *, start=False):
    if (type(producer_instance_ns) is not int or type(now_ns) is not int
        or not 0 < producer_instance_ns <= now_ns):
      self.reset()
      return None
    if producer_instance_ns != self.producer_instance_ns:
      self.reset()
      if not start:
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
    if self._advance(producer_instance_ns, now_ns, start=arm) is not None:
      # RELEASE belongs to a new launch, not to a previous tracking correction.
      self._pid.i = 0.0
      if arm:
        self._domain = 'launch'
        self._positive_authorized = True
      elif not self.armed:
        self._domain = None
        self._positive_authorized = False

  def _valid_values(self, *values):
    return all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
               for value in (*values, self._pid.i, self._pid.neg_limit))

  def _propose_output(self, base_output, candidate, upper_limit, *, qualify=False):
    # Keep the raw P+D+F through addition. Subtracting I from a clipped PID
    # result, or clipping the base before adding I, changes saturation behavior.
    minimum = min(0.0, self._pid.neg_limit - base_output)
    maximum = max(0.0, upper_limit - base_output)
    candidate = min(maximum, max(minimum, candidate))
    baseline = min(upper_limit, max(self._pid.neg_limit, base_output))
    self._pending = (baseline, candidate, qualify)
    return min(upper_limit, max(self._pid.neg_limit, base_output + candidate))

  def freeze_braking(self, base_output, upper_limit, *, producer_instance_ns, now_ns, braking_owned=False):
    """Carry only confirmed braking into APPROACH, without learning or drive.

    Timing and producer identity still apply. Actual HOLD, faults and manual
    revocation reset the state after the owner's stopping envelope takes over.
    """
    self._pending = None
    had_braking = braking_owned is True and (self.braking_handoff or self._valid_values() and self.bias < 0.0)
    if (not self._valid_values(base_output, upper_limit) or self._pid.neg_limit > upper_limit or type(braking_owned) is not bool
        or self._advance(producer_instance_ns, now_ns) is None):
      self.reset()
      if had_braking:
        # The new producer cannot inherit I. Its loss also cannot release
        # the owner's existing stop. Keep only handoff provenance here;
        # LongControl owns the actual last request until APPROACH ends.
        self._domain = 'approach_handoff'
      return base_output
    self._domain = None
    self._positive_authorized = False
    return self._propose_output(base_output, min(self.bias, 0.0), upper_limit)

  def propose(self, base_output, a_target, a_ego, upper_limit, *, producer_instance_ns, now_ns,
              forward, rolling=False, positive_allowed=False, continuation_allowed=True):
    """Propose correction from a base EXCLUDING pid.i; do not accumulate yet.

    A qualified launch uses its original gain until first rolling motion,
    then changes gain without jumping I. FORWARD plus independent permission
    can qualify positive tracking; a confirmed qualification can continue in
    that producer instance. MOVING only retains/unwinds existing positive I.
    Rolling alone grants no drive permission. Near rest, ordinary tracking
    retains/unwinds its existing correction without learning another one.
    Normal ACC has no admitted model motion reference (UNKNOWN); its measured
    response can justify braking correction, but cannot inherit drive effort.
    """
    self._pending = None
    lower_limit = self._pid.neg_limit
    if (not self._valid_values(base_output, a_target, a_ego, upper_limit)
        or lower_limit > upper_limit
        or any(type(value) is not bool for value in (forward, rolling, positive_allowed, continuation_allowed))):
      self.reset()
      return base_output

    if a_target < 0.0 or not continuation_allowed:
      # A new braking target/lost launch context cannot spend positive effort.
      self._pid.i = min(self.bias, 0.0)
      self._positive_authorized = False
      if self.armed:
        self._domain = None
    if self.armed and rolling:
      # One-way gain handoff. A later speed crossing cannot re-enable Ki=1.
      self._domain = 'tracking'
    if self._domain is None and not rolling:
      self.reset()
      return base_output
    elapsed = self._advance(producer_instance_ns, now_ns, start=rolling)
    if elapsed is None:
      return base_output
    if self._domain is None:
      self._domain = 'tracking'

    error = a_target - a_ego
    gain = LAUNCH_RESPONSE_KI if self.armed else TRACKING_RESPONSE_KI
    change = gain * elapsed * error
    if not math.isfinite(change):
      self.reset()
      return base_output
    minimum = min(0.0, lower_limit - base_output)
    maximum = max(0.0, upper_limit - base_output)
    can_learn_positive = continuation_allowed and forward and (positive_allowed or self._positive_authorized) and rolling
    if self.armed:
      minimum = 0.0
      if not forward:
        change = min(change, 0.0)
    else:
      if a_target < 0.0:
        maximum = min(maximum, 0.0)
      elif not can_learn_positive:
        # Permission is independent of speed and reference kind. Continuing
        # MOVING may keep actual prior effort, never ratchet it upward.
        maximum = min(maximum, max(self.bias, 0.0))
      if not rolling:
        # Do not turn a noisy speed threshold into either new learning or a
        # discontinuous withdrawal of previously confirmed launch effort.
        minimum = max(minimum, min(self.bias, 0.0))
    stored = min(maximum, max(minimum, self.bias))
    candidate = min(maximum, max(minimum, stored + change))
    return self._propose_output(base_output, candidate, upper_limit,
                                qualify=bool(continuation_allowed and forward and positive_allowed and a_target >= 0.0))

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
    base_output, candidate, qualify = pending
    delivered = output_accel - base_output
    self._pid.i = float(min(candidate, max(0.0, delivered)) if candidate >= 0.0 else max(candidate, min(0.0, delivered)))
    if qualify and output_accel >= 0.0:
      self._positive_authorized = True
