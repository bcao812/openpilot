"""An early comfort-braking constraint for a continuously observed lead.

This is constant-lead-speed geometry, not proof of a safe following distance.
Continuity checks are a proxy for track consistency, not object identity. The
caller owns platform, driver, straight-road and longitudinal-mode eligibility.
The additional request is limited to comfort braking; the caller must retain
stronger existing MPC requests when applying this acceleration ceiling.
"""

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
import math
from numbers import Integral, Real


QUALIFICATION_NS = 300_000_000
MAX_SOURCE_AGE_NS = 250_000_000
MAX_SAMPLE_GAP_NS = 200_000_000
MIN_DISTINCT_SAMPLES = 5
ENTRY_SPEED = 50.0 / 3.6
EXIT_SPEED = 8.0
MAX_DECEL = 3.5
COMFORT_DECEL = 2.5
# An engineering allowance for response build-up, not a measured actuator model.
RESPONSE_BUILD_TIME = 0.4
PLANNER_INTERVAL = 0.05


@dataclass(frozen=True)
class ClosingConstraint:
  accel: float
  lead_index: int
  required_decel: float
  clearance: float
  remaining_gap: float
  qualified_duration: float
  # Geometry exceeds the total 3.5 m/s^2 budget, not merely this comfort cap.
  # False does not imply that this additional constraint alone preserves 15 m.
  infeasible: bool


def _bounded(value, minimum, maximum):
  return (isinstance(value, Real) and not isinstance(value, bool)
          and minimum <= value <= maximum and math.isfinite(value))


def _timestamp(value):
  return isinstance(value, Integral) and not isinstance(value, bool) and 0 < value <= 2 ** 63 - 1


@dataclass
class _LeadHistory:
  first_ns: int = 0
  source_ns: int = 0
  count: int = 0
  distance: float = 0.0
  relative_speed: float = 0.0
  lead_speed: float = 0.0
  active: bool = False
  residuals: deque = field(default_factory=deque)

  def clear(self):
    self.first_ns = self.source_ns = self.count = 0
    self.distance = self.relative_speed = self.lead_speed = 0.0
    self.active = False
    self.residuals.clear()

  def seed(self, source_ns, distance, relative_speed, lead_speed):
    self.clear()
    self.first_ns = self.source_ns = source_ns
    self.count = 1
    self.distance, self.relative_speed, self.lead_speed = distance, relative_speed, lead_speed

  def observe(self, source_ns, distance, relative_speed, lead_speed):
    if self.count == 0:
      self.seed(source_ns, distance, relative_speed, lead_speed)
      return False
    if source_ns == self.source_ns:
      # One model epoch cannot supply new evidence or new geometry. A changed
      # observation with a reused epoch must acquire its qualification again.
      if (distance, relative_speed, lead_speed) != (self.distance, self.relative_speed, self.lead_speed):
        self.seed(source_ns, distance, relative_speed, lead_speed)
        return False
      return self.qualified

    gap_ns = source_ns - self.source_ns
    if gap_ns <= 0 or gap_ns > MAX_SAMPLE_GAP_NS:
      self.seed(source_ns, distance, relative_speed, lead_speed)
      return False
    predicted_change = 0.5 * (relative_speed + self.relative_speed) * (gap_ns / 1e9)
    residual = abs(distance - self.distance - predicted_change)
    if residual > max(1.5, 0.03 * min(distance, self.distance)):
      self.seed(source_ns, distance, relative_speed, lead_speed)
      return False

    self.residuals.append((self.source_ns, source_ns, residual))
    cutoff_ns = source_ns - QUALIFICATION_NS
    while self.residuals and self.residuals[0][1] <= cutoff_ns:
      self.residuals.popleft()
    # Sum absolute residuals so opposite errors cannot cancel. Prorating the
    # oldest interval retains an actual 0.30 s window at variable cadence.
    accumulated = sum(error * (end - max(start, cutoff_ns)) / (end - start)
                      for start, end, error in self.residuals)
    if accumulated > max(3.0, 0.03 * distance):
      self.seed(source_ns, distance, relative_speed, lead_speed)
      return False

    self.source_ns = source_ns
    self.distance, self.relative_speed, self.lead_speed = distance, relative_speed, lead_speed
    self.count = min(self.count + 1, MIN_DISTINCT_SAMPLES)
    return self.qualified

  @property
  def qualified(self):
    return self.count >= MIN_DISTINCT_SAMPLES and self.source_ns - self.first_ns >= QUALIFICATION_NS


class LeadClosingConstraint:
  def __init__(self):
    self._leads = (_LeadHistory(), _LeadHistory())
    self._last_now_ns = 0
    self._last_source_ns = 0

  def reset(self):
    for history in self._leads:
      history.clear()
    self._last_now_ns = self._last_source_ns = 0

  def update(self, *, now_ns: int, source_ns: int, eligible: bool, v_ego: float, a_ego: float,
             leads: Sequence, actuator_delay: float, t_follow: float) -> ClosingConstraint | None:
    """Return an additional acceleration ceiling, or no participating constraint.

    ``source_ns`` is radarState.mdMonoTime: the real model epoch, not receipt
    time. Valid repeated epochs may retain a fresh constraint but never qualify
    it. Lead slots qualify independently; their inputs are never modified.
    """
    if (eligible is not True or not _timestamp(now_ns) or not _timestamp(source_ns)
        or not 0 <= now_ns - source_ns <= MAX_SOURCE_AGE_NS
        or not _bounded(v_ego, 0.0, 100.0) or not _bounded(a_ego, -20.0, 20.0)
        or not _bounded(actuator_delay, 0.0, 1.0) or not _bounded(t_follow, 0.0, 3.0) or t_follow <= 0.0
        or not isinstance(leads, Sequence)):
      self.reset()
      return None
    if now_ns < self._last_now_ns or source_ns < self._last_source_ns:
      self.reset()
      return None
    self._last_now_ns, self._last_source_ns = now_ns, source_ns
    if v_ego < EXIT_SPEED:
      self.reset()
      return None

    selected = None
    for index, history in enumerate(self._leads):
      lead = leads[index] if index < len(leads) else None
      distance = getattr(lead, 'dRel', None)
      relative_speed = getattr(lead, 'vRel', None)
      lead_speed = getattr(lead, 'vLead', None)
      probability = getattr(lead, 'modelProb', None)
      if (getattr(lead, 'present', False) is not True or not _bounded(distance, 0.0, 10000.0) or distance <= 0.0
          or not _bounded(relative_speed, -100.0, -1.0) or relative_speed >= -1.0
          or not _bounded(lead_speed, 0.0, 100.0) or not _bounded(probability, 0.95, 1.0)
          or abs(lead_speed - v_ego - relative_speed) > 1.0):
        history.clear()
        continue
      if not history.observe(source_ns, distance, relative_speed, lead_speed):
        continue
      if not history.active and v_ego < ENTRY_SPEED:
        continue
      comfort = (v_ego ** 2 - lead_speed ** 2) / 5.0 + t_follow * v_ego + 6.0
      if comfort <= 0.0 or distance > comfort:
        continue

      history.active = True
      # A desired time gap is an admission policy, not a hard clearance. Using
      # it here would turn an ordinary small gap error into emergency braking.
      clearance = 15.0
      response_time = actuator_delay + PLANNER_INTERVAL + RESPONSE_BUILD_TIME
      positive_accel = max(a_ego, 0.0)
      closing_speed = -relative_speed
      remaining = distance - clearance - closing_speed * response_time - 0.5 * positive_accel * response_time ** 2
      response_closing_speed = closing_speed + positive_accel * response_time
      required = response_closing_speed ** 2 / (2.0 * remaining) if remaining > 0.0 else math.inf
      comfort_cap = COMFORT_DECEL * min(max((closing_speed - 1.0) / 2.0, 0.0), 1.0)
      constraint = ClosingConstraint(
        accel=-min(required, comfort_cap, MAX_DECEL), lead_index=index, required_decel=required, clearance=clearance,
        remaining_gap=remaining, qualified_duration=(source_ns - history.first_ns) / 1e9,
        infeasible=remaining <= 0.0 or required > MAX_DECEL,
      )
      if selected is None or constraint.accel < selected.accel:
        selected = constraint
    return selected
