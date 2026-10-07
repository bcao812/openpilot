"""Limit speed recovery after losing a recently followed car in a turn.

This is an upper bound for the planner's existing scalar arbitration. It does
not create a lead, request braking or provide stop/departure authority. An
empty-road turn cannot establish the following context on its own.
"""
import math
from numbers import Real

from openpilot.selfdrive.controls.lib.lead_qualification import qualified_lead


MAX_INPUT_AGE_NS = 250_000_000
LEAD_HISTORY_NS = 3_000_000_000
LEAD_RECOVERY_NS = 200_000_000
STRAIGHT_CONFIRM_NS = 1_000_000_000
TURN_ENTER_CURVATURE = 1.0 / 30.0
TURN_EXIT_CURVATURE = 1.0 / 50.0
MIN_ENTRY_SPEED = 0.5
MAX_ENTRY_SPEED = 10.0
MIN_LEAD_PROBABILITY = 0.8
MAX_LEAD_DISTANCE = 50.0
RECOVERY_JERK = 1.0
HOLD_SPEED_GAIN = 0.5
HOLD_ACCEL_MAX = 0.3


def _finite(value):
  if not isinstance(value, Real) or isinstance(value, bool):
    return False
  try:
    return math.isfinite(value)
  except (OverflowError, ValueError):
    return False


def _timestamp(value):
  return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _lead_state(lead, now_ns=None):
  present = getattr(lead, 'present', None)
  if not isinstance(present, bool):
    return None
  if not present:
    return False, False
  values = tuple(getattr(lead, field, None) for field in ('dRel', 'vRel', 'vLead', 'aLeadK', 'modelProb', 'yRel'))
  if not all(_finite(value) for value in values) or not 0.0 <= values[4] <= 1.0:
    return None
  reliable = (0.0 < values[0] <= MAX_LEAD_DISTANCE
              and qualified_lead(lead, 'follow', MIN_LEAD_PROBABILITY, now_ns=now_ns))
  return True, reliable


class TurnLeadGuard:
  """One following-turn context with a captured speed and a gradual release.

  ``binding`` means that this helper supplies an acceleration ceiling, including
  its release ramp. Whether it lowers the final request is decided by the
  caller's single minimum arbitration. ``lost`` remains true through weak lead
  flicker until fresh reliable radarState publications confirm recovery.
  """

  def __init__(self):
    self.clear()

  def clear(self):
    self.armed = False
    self.lost = False
    self.accel_ceiling = None
    self.hold_speed = None
    self.speed_recovery_blocked = False
    self._last_now_ns = None
    self._last_model_ns = None
    self._last_radar_ns = None
    self._last_reliable_radar_ns = None
    self._reliable_since_ns = None
    self._straight_since_ns = None

  @property
  def binding(self):
    return self.accel_ceiling is not None

  def update(self, *, now_ns, model_ns, radar_ns, enabled, speed, standstill,
             steering_angle, steer_ratio, wheelbase, model_curvature, lead_one, lead_two, ceiling_max,
             independent_braking=False):
    """Return an optional acceleration ceiling in m/s².

    ``steering_angle`` is the measured steering wheel angle in degrees with the learned
    angle offset already removed. Both steering geometry and the model's
    desired curvature must agree that the turn has ended. The caller supplies
    its existing input-health and driver-override checks through ``enabled``.
    """
    if enabled is not True or standstill is not False or not isinstance(independent_braking, bool):
      self.clear()
      return None
    if (not all(_timestamp(stamp) for stamp in (now_ns, model_ns, radar_ns))
        or not 0 <= now_ns - model_ns <= MAX_INPUT_AGE_NS
        or not 0 <= now_ns - radar_ns <= MAX_INPUT_AGE_NS
        or not all(_finite(value) for value in (speed, steering_angle, steer_ratio, wheelbase, model_curvature, ceiling_max))
        or speed < 0.0 or steer_ratio <= 0.0 or wheelbase <= 0.0 or ceiling_max < 0.0):
      self.clear()
      return None

    leads = (lead_one, lead_two)
    lead_states = tuple(_lead_state(lead, now_ns) for lead in leads)
    if any(state is None for state in lead_states):
      self.clear()
      return None
    wheel_angle = math.radians(steering_angle) / steer_ratio
    if not math.isfinite(wheel_angle) or abs(wheel_angle) >= math.pi / 2.0:
      self.clear()
      return None
    curvature = max(abs(math.tan(wheel_angle) / wheelbase), abs(model_curvature))
    if not math.isfinite(curvature):
      self.clear()
      return None

    if ((self._last_now_ns is not None and not 0 < now_ns - self._last_now_ns <= MAX_INPUT_AGE_NS)
        or (self._last_model_ns is not None and model_ns < self._last_model_ns)
        or (self._last_radar_ns is not None and radar_ns < self._last_radar_ns)):
      self.clear()
      return None
    dt = (now_ns - self._last_now_ns) / 1e9 if self._last_now_ns is not None else 0.0
    new_radar = radar_ns != self._last_radar_ns
    self._last_now_ns, self._last_model_ns, self._last_radar_ns = now_ns, model_ns, radar_ns
    present = any(state[0] for state in lead_states)
    reliable = any(state[1] for state in lead_states)
    if new_radar and reliable:
      # Source time, not repeated consumer ticks, dates the last observation.
      self._last_reliable_radar_ns = max(
        getattr(lead.evidence, 'positionMonoTime', radar_ns) if getattr(getattr(lead, 'evidence', None), 'version', 0) == 1 else radar_ns
        for lead, state in zip(leads, lead_states, strict=True) if state[1])

    released = False
    if self.armed:
      if curvature <= TURN_EXIT_CURVATURE:
        if self._straight_since_ns is None:
          self._straight_since_ns = now_ns
        if now_ns - self._straight_since_ns >= STRAIGHT_CONFIRM_NS:
          released = self.lost
          self.armed = False
          self.lost = False
          self._last_reliable_radar_ns = None
          self._reliable_since_ns = None
          self._straight_since_ns = None
      else:
        self._straight_since_ns = None
    elif (MIN_ENTRY_SPEED < speed < MAX_ENTRY_SPEED and curvature >= TURN_ENTER_CURVATURE
          and self._last_reliable_radar_ns is not None
          and 0 <= now_ns - self._last_reliable_radar_ns <= LEAD_HISTORY_NS):
      self.armed = True

    if self.armed:
      if not present:
        if not self.lost:
          # Capture once. Recapturing every frame would ratchet the target
          # down when drag or a grade lowers speed while the lead is hidden.
          self.hold_speed = float(speed)
          self.speed_recovery_blocked = False
        self.lost = True
        self._reliable_since_ns = None
      elif self.lost:
        if not reliable:
          self._reliable_since_ns = None
        elif new_radar:
          recovery_ns = self._last_reliable_radar_ns
          if self._reliable_since_ns is None:
            self._reliable_since_ns = recovery_ns
          elif recovery_ns - self._reliable_since_ns >= LEAD_RECOVERY_NS:
            self.lost = False
            self._reliable_since_ns = None
            released = True

    if self.lost:
      # A separate admitted braking/stop/speed-limit request may deliberately
      # lower speed after loss. Do not chase the former speed once it ends.
      # A returning weak lead likewise has priority over drag compensation.
      self.speed_recovery_blocked |= independent_braking or present
      # Allow a small, bounded correction for speed lost to resistance. This
      # remains an upper bound: model/lead/curve braking still wins the
      # planner's minimum, and no correction is allowed above the held speed.
      target = (0.0 if self.speed_recovery_blocked else
                min(float(ceiling_max), HOLD_ACCEL_MAX, max(0.0, HOLD_SPEED_GAIN * (self.hold_speed - speed))))
      previous = self.accel_ceiling if self.accel_ceiling is not None else 0.0
      self.accel_ceiling = min(target, previous + RECOVERY_JERK * dt)
    elif self.accel_ceiling is not None and not released:
      self.accel_ceiling = min(float(ceiling_max), self.accel_ceiling + RECOVERY_JERK * dt)
      if self.accel_ceiling >= ceiling_max:
        self.accel_ceiling = None
    if not self.lost:
      self.hold_speed = None
      self.speed_recovery_blocked = False
    # The confirmation interval still belonged to the zero-ceiling state.
    # Start ramping on the next update, using only actual time since release.
    return self.accel_ceiling
