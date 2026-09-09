from collections import deque
from enum import StrEnum
import math
from statistics import median


class CruiseSmoothingMode(StrEnum):
  OFF = 'off'
  BRAKE_DWELL = 'brake_dwell'
  MEDIAN = 'median'


class CruiseSmoother:
  BRAKE_THRESHOLD = -0.2
  MINIMUM_BRAKE_TIME = 0.3
  BRAKE_CONFIRMATION_TIME = 0.2
  RELEASE_CONFIRMATION_TIME = 0.15
  MAXIMUM_HOLD_DELTA = 0.04
  MEDIAN_WINDOW = 0.2
  MAXIMUM_FILTER_DELTA = 0.06
  ELIGIBILITY_CONFIRMATION_TIME = 0.5

  def __init__(self, mode=CruiseSmoothingMode.OFF, dt=0.05):
    self.mode = CruiseSmoothingMode(mode)
    if not math.isfinite(dt) or dt <= 0:
      raise ValueError('Cruise smoothing requires a positive finite timestep')
    self.dt = dt
    samples = max(3, round(self.MEDIAN_WINDOW / dt) + 1)
    self.samples = deque(maxlen=samples if samples % 2 else samples + 1)
    self.reset()

  def reset(self):
    self.samples.clear()
    self.braking = False
    self.brake_confirmed = False
    self.input_brake_time = 0.0
    self.brake_time = 0.0
    self.release_time = 0.0
    self.brake_accel = self.BRAKE_THRESHOLD
    self.eligible_time = 0.0

  def update(self, accel, eligible):
    if self.mode == CruiseSmoothingMode.OFF or not eligible or not math.isfinite(accel):
      self.reset()
      return accel
    self.eligible_time = min(self.eligible_time + self.dt, self.ELIGIBILITY_CONFIRMATION_TIME + self.dt)
    if self.eligible_time <= self.ELIGIBILITY_CONFIRMATION_TIME + 1e-9:
      return accel
    if self.mode == CruiseSmoothingMode.MEDIAN:
      self.samples.append(accel)
      if len(self.samples) < self.samples.maxlen:
        return accel
      return max(accel - self.MAXIMUM_FILTER_DELTA, min(median(self.samples), accel + self.MAXIMUM_FILTER_DELTA))
    if accel < self.BRAKE_THRESHOLD:
      self.input_brake_time += self.dt
      self.brake_confirmed |= self.input_brake_time + 1e-9 >= self.BRAKE_CONFIRMATION_TIME
      self.brake_time = self.brake_time + self.dt if self.braking else 0.0
      self.braking = True
      self.brake_accel = accel
      self.release_time = 0.0
      return accel
    self.input_brake_time = 0.0
    if self.braking and self.brake_confirmed:
      self.brake_time += self.dt
      self.release_time += self.dt
      hold = max(self.brake_accel, accel - self.MAXIMUM_HOLD_DELTA)
      pending = self.brake_time + 1e-9 < self.MINIMUM_BRAKE_TIME or self.release_time <= self.RELEASE_CONFIRMATION_TIME + 1e-9
      if pending and hold < self.BRAKE_THRESHOLD:
        return hold
    self.braking = False
    self.brake_confirmed = False
    self.brake_time = self.release_time = 0.0
    return accel
