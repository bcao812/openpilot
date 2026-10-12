"""Bounded continuity for an independently qualified, low-speed lead launch.

The caller owns perception/risk admission and the motion owner owns brake
release. Vision-slot continuity below is a conservative consistency check,
not proof of physical object identity or a replacement for cut-in perception.
"""
import math
from dataclasses import dataclass
from numbers import Integral


LEAD_LAUNCH_CONFIRM_NS = 250_000_000
LEAD_LAUNCH_MAX_TIME_NS = 5_000_000_000
LEAD_LAUNCH_MAX_SPEED = 2.0
LEAD_LAUNCH_MAX_DISTANCE = 5.0
SOURCE_MAX_AGE_NS = 200_000_000
SOURCE_MAX_STEP_NS = 150_000_000
STANDSTILL_RESIDUE = 1e-6  # Only with standstill and independently zero vEgoRaw.


@dataclass(frozen=True)
class LeadLaunchSample:
  identity: tuple
  distance: float
  speed: float
  relative_speed: float
  lateral: float


class LeadLaunch:
  def __init__(self):
    self.clear()

  def clear(self, reason='inactive'):
    self.active = False
    self.reason = reason
    self.exhausted = False
    self.confirmed_ns = 0
    self.started_ns = None
    self.last_now_ns = None
    self.last_model_ns = None
    self.last_radar_ns = None
    self.last_speed = 0.0
    self.distance = 0.0
    self.leads = ()

  @staticmethod
  def _timestamp(value):
    return isinstance(value, Integral) and not isinstance(value, bool) and 0 < value < (1 << 64)

  def _remember(self, now_ns, model_ns, radar_ns, speed, leads):
    self.last_now_ns, self.last_model_ns, self.last_radar_ns = now_ns, model_ns, radar_ns
    self.last_speed, self.leads = speed, leads

  def _exhaust(self, reason):
    self.active, self.exhausted, self.reason = False, True, reason
    return False

  def update(self, *, now_ns, model_ns, radar_ns, enabled, can_start, speed, leads):
    if not enabled:
      self.clear('risk_or_scope')
      return False
    if not math.isfinite(speed) or not 0.0 <= speed < LEAD_LAUNCH_MAX_SPEED:
      self.clear('speed_scope')
      return False
    if (not all(self._timestamp(value) for value in (now_ns, model_ns, radar_ns))
        or not 0 <= now_ns - model_ns <= SOURCE_MAX_AGE_NS
        or not 0 <= now_ns - radar_ns <= SOURCE_MAX_AGE_NS):
      self.clear('source_age')
      return False
    if (not leads or any(not isinstance(lead.identity, tuple) or not lead.identity
                         or not all(math.isfinite(value) for value in
                                    (lead.distance, lead.speed, lead.relative_speed, lead.lateral))
                         or lead.distance <= 0.0 for lead in leads)):
      self.clear('lead_input')
      return False
    if self.exhausted:
      return False
    if self.last_now_ns is None:
      if not can_start:
        self.reason = 'not_held'
        return False
      self.reason = 'confirming'
      self._remember(now_ns, model_ns, radar_ns, speed, leads)
      return False
    if (now_ns <= self.last_now_ns or model_ns < self.last_model_ns or radar_ns < self.last_radar_ns
        or now_ns - self.last_now_ns > SOURCE_MAX_STEP_NS):
      self.clear('source_discontinuity')
      return False
    if not self.active and not can_start:
      self.clear('not_held')
      return False
    # A changed/new target cannot spend the old target's confirmation. Bounds
    # are candidate continuity limits, not calibrated perception uncertainties.
    same_ids = tuple(lead.identity for lead in leads) == tuple(lead.identity for lead in self.leads)
    dt = (radar_ns - self.last_radar_ns) / 1e9
    continuous = same_ids and all(
      abs(new.distance - old.distance - 0.5 * (new.relative_speed + old.relative_speed) * dt) <= 0.75
      and abs(new.lateral - old.lateral) <= 0.5 and abs(new.speed - old.speed) <= 1.0
      for old, new in zip(self.leads, leads, strict=True)
    )
    if not continuous:
      self.clear('lead_changed')
      return False
    elapsed = (now_ns - self.last_now_ns) / 1e9
    if self.active:
      self.distance += 0.5 * (speed + self.last_speed) * elapsed
      if now_ns - self.started_ns >= LEAD_LAUNCH_MAX_TIME_NS:
        return self._exhaust('time_budget')
      if self.distance >= LEAD_LAUNCH_MAX_DISTANCE:
        return self._exhaust('distance_budget')
    fresh = model_ns > self.last_model_ns and radar_ns > self.last_radar_ns
    if fresh:
      step = min(model_ns - self.last_model_ns, radar_ns - self.last_radar_ns)
      if max(model_ns - self.last_model_ns, radar_ns - self.last_radar_ns) > SOURCE_MAX_STEP_NS:
        self.clear('source_discontinuity')
        return False
      if not self.active:
        self.confirmed_ns += step
        if self.confirmed_ns >= LEAD_LAUNCH_CONFIRM_NS:
          self.active = True
          self.started_ns = now_ns
    # Count confirmation only when both sources advance. Retain their paired
    # identities across a one-sided update, while dating ego travel every tick.
    self.last_now_ns, self.last_speed = now_ns, speed
    if fresh:
      self.last_model_ns, self.last_radar_ns, self.leads = model_ns, radar_ns, leads
    self.reason = 'active' if self.active else 'confirming'
    return self.active
