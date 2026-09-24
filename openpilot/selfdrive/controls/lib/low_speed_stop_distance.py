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
LOW_SPEED_WINDOW_NS = 10_000_000_000
MAX_SOURCE_GAP_NS = 250_000_000
LEAD_STABLE_NS = 500_000_000
LEAD_RAMP_NS = 500_000_000
APPROACH_GRACE_NS = 500_000_000
NEAR_STOP_LEAD_SPEED = 0.3
APPROACH_LEAD_DECEL = -0.05
MAX_COMPACT_SPEED = 30.0 / 3.6


def _distance_profile(speed, low):
  fraction = min(1.0, max(0.0, (speed * 3.6 - 15.0) / 15.0))
  return low + (DEFAULT_RISK_DISTANCE - low) * fraction


def _observation(lead, speed):
  if not getattr(lead, 'present', False):
    return None
  values = tuple(getattr(lead, name, None) for name in ('dRel', 'vRel', 'vLead', 'aLeadK', 'modelProb', 'yRel'))
  if not all(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) for value in values):
    return None
  values = tuple(float(value) for value in values)
  distance, relative_speed, lead_speed, lead_accel, probability, lateral = values
  if (distance <= 0.0 or not 0.0 <= lead_speed <= 2.0 or not -0.5 <= lead_accel <= 0.2
      or not 0.98 <= probability <= 1.0 or abs(lateral) > 1.2
      or abs(lead_speed - speed - relative_speed) > 0.5):
    return None
  return values


class LowSpeedStopDistance:
  def __init__(self):
    self.clear()

  def clear(self):
    self._speeds = deque()
    self._last_now_ns = None
    self._last_source_ns = None
    self._last_radar_ns = None
    self._lead_since = [None, None]
    self._lead_previous = [None, None]
    self._approach_since = [None, None]
    self._last_approach_ns = [None, None]
    self._fractions = [0.0, 0.0]

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
      self.clear()
      return fallback
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
    if radar_delta is not None and not 0 <= radar_delta <= MAX_SOURCE_GAP_NS:
      self._lead_since = [None, None]
      self._lead_previous = [None, None]
      self._approach_since = [None, None]
      self._last_approach_ns = [None, None]
      self._fractions = [0.0, 0.0]
      radar_delta = None
    fractions = []
    for i, lead in enumerate(leads):
      observation = _observation(lead, speed)
      previous = self._lead_previous[i]
      continuous = observation is not None and previous is not None and radar_delta is not None
      if continuous:
        if radar_delta == 0:
          # vRel is an observation from this radar publication, not resampled
          # against a new ego-speed estimate. Other fields must be unchanged.
          continuous = observation == previous
        else:
          predicted = previous[0] + 0.5 * (previous[1] + observation[1]) * radar_delta / 1e9
          continuous = abs(observation[0] - predicted) <= 0.5 and abs(observation[5] - previous[5]) <= 0.3
      if observation is None:
        self._lead_since[i] = None
        self._fractions[i] = 0.0
      elif not continuous or self._lead_since[i] is None:
        self._lead_since[i] = radar_ns
        self._fractions[i] = 0.0
      if not continuous:
        self._approach_since[i] = None
        self._last_approach_ns[i] = None
      self._lead_previous[i] = observation
      # Start reducing the parking target only during an observed stopping
      # approach. A constant slow lead does not ask us to catch up merely
      # because the ten-second speed-history timer has become complete.
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
    comfort = fallback[0] + common_fraction * (_distance_profile(speed, COMPACT_COMFORT_DISTANCE) - fallback[0])
    risk = tuple(DEFAULT_RISK_DISTANCE + (common_fraction if exists else 0.0)
                 * (_distance_profile(speed, COMPACT_RISK_DISTANCE) - DEFAULT_RISK_DISTANCE) for exists in present)
    return float(comfort), risk
