"""Qualified slow-lead stand-off, separate from dynamic headway and HOLD.

The complete recent-speed window prevents a high-speed approach from acquiring
the compact target merely by passing through its low-speed tail. Slot history
checks continuity of observations, not an asserted physical tracking identity.
"""
from collections import deque
import math
from numbers import Real


DEFAULT_RISK_DISTANCE = 6.0
COMPACT_COMFORT_DISTANCE = 3.0
COMPACT_RISK_DISTANCE = 4.0  # Existing 0.75 risk factor -> nominal 3 m at zero speed.
LOW_SPEED_WINDOW_NS = 5_000_000_000
MAX_SOURCE_GAP_NS = 250_000_000
LEAD_STABLE_NS = 500_000_000
LEAD_RAMP_NS = 500_000_000
APPROACH_GRACE_NS = 500_000_000
NEAR_STOP_LEAD_SPEED = 0.3
APPROACH_LEAD_DECEL = -0.05
PARKING_START_SPEED = 1.2
PARKING_FULL_SPEED = 0.3
MAX_COMPACT_SPEED = 30.0 / 3.6


def _distance_profile(speed, low):
  fraction = min(1.0, max(0.0, (speed * 3.6 - 15.0) / 15.0))
  return low + (DEFAULT_RISK_DISTANCE - low) * fraction


def _tracked_observation(lead, speed):
  if not getattr(lead, 'present', False):
    return None
  values = tuple(getattr(lead, name, None) for name in ('dRel', 'vRel', 'vLead', 'aLeadK', 'modelProb', 'yRel'))
  if not all(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) for value in values):
    return None
  values = tuple(float(value) for value in values)
  distance, relative_speed, lead_speed, lead_accel, probability, lateral = values
  if (distance <= 0.0 or not 0.98 <= probability <= 1.0 or abs(lateral) > 1.2
      or abs(lead_speed - speed - relative_speed) > 0.5):
    return None
  return values


def _observation(lead, speed):
  values = _tracked_observation(lead, speed)
  if values is None or not 0.0 <= values[2] <= 2.0 or not -0.5 <= values[3] <= 0.2:
    return None
  return values


class LowSpeedStopDistance:
  def __init__(self):
    self.clear()

  def clear(self, *, source_barrier_ns=None):
    self._speeds = deque()
    self._last_now_ns = None
    self._last_source_ns = None
    self._last_radar_ns = None
    self._lead_since = [None, None]
    self._lead_previous = [None, None]
    self._lead_was_present = [False, False]
    self._approach_since = [None, None]
    self._last_approach_ns = [None, None]
    self._fractions = [0.0, 0.0]
    self._applied_fraction = 0.0
    self._has_compact_history = False
    self._source_barrier_ns = source_barrier_ns

  def update(self, *, now_ns, source_ns, radar_ns, speed, radar, permitted, allow_activation=True):
    speed_valid = isinstance(speed, Real) and not isinstance(speed, bool) and math.isfinite(speed) and speed >= 0.0
    fallback = (_distance_profile(speed, 4.0) if speed_valid else DEFAULT_RISK_DISTANCE,
                (DEFAULT_RISK_DISTANCE, DEFAULT_RISK_DISTANCE))
    leads = (getattr(radar, 'leadOne', None), getattr(radar, 'leadTwo', None))
    stamps_valid = all(isinstance(value, int) and not isinstance(value, bool) and value > 0
                       for value in (now_ns, source_ns, radar_ns))
    if (not permitted or not speed_valid or not stamps_valid
        or any(lead is None or getattr(lead, 'present', None) is None for lead in leads)
        or not 0 <= now_ns - source_ns <= MAX_SOURCE_GAP_NS
        or not 0 <= now_ns - radar_ns <= MAX_SOURCE_GAP_NS):
      self.clear(source_barrier_ns=self._source_barrier_ns)
      return fallback
    if self._source_barrier_ns is not None:
      if source_ns <= self._source_barrier_ns:
        return fallback
      self._source_barrier_ns = None
    if self._last_now_ns is not None and not 0 < now_ns - self._last_now_ns <= MAX_SOURCE_GAP_NS:
      self.clear()
    if self._last_source_ns is not None:
      source_delta = source_ns - self._last_source_ns
      if (not 0 <= source_delta <= MAX_SOURCE_GAP_NS
          or (source_delta == 0 and self._speeds and self._speeds[-1][1] != speed)):
        self.clear()
    self._last_now_ns = now_ns
    if self._last_source_ns != source_ns:
      self._speeds.append((source_ns, speed))
    self._last_source_ns = source_ns
    cutoff = source_ns - LOW_SPEED_WINDOW_NS
    # Retain the sample at/before the left boundary as well as every update.
    while len(self._speeds) > 1 and self._speeds[1][0] <= cutoff:
      self._speeds.popleft()
    low_speed_history = (bool(self._speeds) and self._speeds[0][0] <= cutoff
                         and all(value < MAX_COMPACT_SPEED for _, value in self._speeds))

    radar_delta = radar_ns - self._last_radar_ns if self._last_radar_ns is not None else None
    armed = self._has_compact_history or any(fraction > 0.0 for fraction in self._fractions)
    if radar_delta is not None and not 0 <= radar_delta <= MAX_SOURCE_GAP_NS:
      if armed:
        self.clear(source_barrier_ns=source_ns)
        return fallback
      self._lead_since = [None, None]
      self._lead_previous = [None, None]
      self._lead_was_present = [False, False]
      self._approach_since = [None, None]
      self._last_approach_ns = [None, None]
      self._fractions = [0.0, 0.0]
      self._applied_fraction = 0.0
      radar_delta = None
    fractions = []
    confirmed_approaches = []
    for i, lead in enumerate(leads):
      tracked = _tracked_observation(lead, speed)
      observation = _observation(lead, speed)
      previous = self._lead_previous[i]
      # Normal lead acceleration or speed outside the parking domain revokes
      # this request, but does not turn a continuously observed car into a lost
      # target. Keep the ego history through ordinary stop-and-go.
      continuous = tracked is not None and previous is not None and radar_delta is not None
      if continuous:
        if radar_delta == 0:
          # vRel is an observation from this radar publication, not resampled
          # against a new ego-speed estimate. Other fields must be unchanged.
          continuous = tracked == previous
        else:
          predicted = previous[0] + 0.5 * (previous[1] + tracked[1]) * radar_delta / 1e9
          continuous = abs(tracked[0] - predicted) <= 0.5 and abs(tracked[5] - previous[5]) <= 0.3
      if armed and ((previous is not None and not continuous)
                    or (lead.present and not self._lead_was_present[i])):
        # A changed/lost target must not regain a smaller risk envelope one
        # second later using speed history collected before that uncertainty.
        # The next fresh carState starts a new complete confirmation window.
        # An absent second slot that has stayed absent is not a target loss.
        self.clear(source_barrier_ns=source_ns)
        return fallback
      if observation is None:
        self._lead_since[i] = None
        self._fractions[i] = 0.0
      elif not continuous or self._lead_since[i] is None:
        self._lead_since[i] = radar_ns
        self._fractions[i] = 0.0
      if not continuous:
        self._approach_since[i] = None
        self._last_approach_ns[i] = None
      self._lead_previous[i] = tracked
      # Start reducing the parking target only during an observed stopping
      # approach. A constant slow lead does not ask us to catch up merely
      # because the speed-history timer has become complete.
      approaching = (observation is not None
                     and (observation[2] <= NEAR_STOP_LEAD_SPEED or observation[3] <= APPROACH_LEAD_DECEL))
      if not approaching:
        self._approach_since[i] = None
      elif self._approach_since[i] is None:
        self._approach_since[i] = radar_ns
      approach_since = self._approach_since[i]
      approach_confirmed = (continuous and approach_since is not None
                            and radar_ns - approach_since >= LEAD_STABLE_NS)
      if approach_confirmed:
        self._last_approach_ns[i] = radar_ns
      confirmed_approaches.append(approach_confirmed)
      since = self._lead_since[i]
      if not low_speed_history or not allow_activation or since is None:
        self._fractions[i] = 0.0
      elif approach_confirmed and radar_delta and radar_ns - since >= LEAD_STABLE_NS:
        stable_increment = min(radar_delta, radar_ns - since - LEAD_STABLE_NS, radar_ns - approach_since - LEAD_STABLE_NS)
        self._fractions[i] = min(1.0, self._fractions[i] + stable_increment / LEAD_RAMP_NS)
      elif self._last_approach_ns[i] is None or radar_ns - self._last_approach_ns[i] > APPROACH_GRACE_NS:
        self._fractions[i] = 0.0
      # Briefly weakened approach evidence can only freeze an existing
      # fraction. The deadline comes from the last continuously confirmed
      # approach, so isolated noisy samples cannot keep renewing it.
      fraction = self._fractions[i]
      fractions.append(fraction)
    self._last_radar_ns = radar_ns
    # The current native comfort parameter is shared by real leads. Require
    # every present lead to qualify before reducing it, so a good far target
    # cannot shorten the comfort target for a bad/close second observation.
    present = [bool(getattr(lead, 'present', False)) for lead in leads]
    common_fraction = min((fraction for fraction, exists in zip(fractions, present, strict=True) if exists), default=0.0)
    # Retain ordinary following geometry until the final rolling-stop range.
    # Reducing the stand-off earlier can move the car closer before a cut-in
    # or a lead dropout, even though the later fault restores the old target.
    parking_fraction = min(1.0, max(0.0, (PARKING_START_SPEED - speed) / (PARKING_START_SPEED - PARKING_FULL_SPEED)))
    common_fraction *= parking_fraction
    if not all(confirmed for confirmed, exists in zip(confirmed_approaches, present, strict=True) if exists):
      # Freeze actual relaxation during grace, including any change of the
      # ego-speed factor. Slowing down is not new lead-approach evidence.
      common_fraction = min(common_fraction, self._applied_fraction)
    self._applied_fraction = common_fraction
    # Revoking today's request does not undo forward travel that a previous
    # compact target may have induced. Preserve the uncertainty barrier until
    # the complete policy history is cleared, including through HOLD/grace.
    self._has_compact_history |= common_fraction > 0.0
    self._lead_was_present = present
    comfort = fallback[0] + common_fraction * (_distance_profile(speed, COMPACT_COMFORT_DISTANCE) - fallback[0])
    risk = tuple(DEFAULT_RISK_DISTANCE + (common_fraction if exists else 0.0)
                 * (_distance_profile(speed, COMPACT_RISK_DISTANCE) - DEFAULT_RISK_DISTANCE) for exists in present)
    return float(comfort), risk
