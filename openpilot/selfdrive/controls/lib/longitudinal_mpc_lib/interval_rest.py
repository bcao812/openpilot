"""A braking ramp followed by rest inside the first public MPC interval.

The measured initial state is retained. A constant positive jerk removes the
negative acceleration exactly when forward speed reaches zero; the rest of
the interval holds position. This is an explicit motion-mode transition, not
clipping a negative velocity after integrating the original whole interval.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class IntervalRest:
  position: float
  velocity: float
  acceleration: float
  stop_time: float

  @classmethod
  def from_state(cls, state, first_interval, deadline):
    position, velocity, acceleration = map(float, state)
    if (not all(math.isfinite(value) for value in (position, velocity, acceleration, first_interval, deadline))
        or velocity <= 0.0 or acceleration >= 0.0 or first_interval <= 0.0 or deadline <= 0.0):
      return None
    duration = -2.0 * velocity / acceleration
    if not 0.0 < duration < first_interval or duration > deadline:
      return None
    if not math.isfinite(-acceleration / duration) or not math.isfinite(position + velocity * duration / 3.0):
      return None
    return cls(position, velocity, acceleration, duration)

  def sample(self, times):
    times = np.asarray(times, dtype=float)
    if not np.all(np.isfinite(times)) or np.any(times < 0.0):
      raise ValueError('Invalid interval-rest sampling times')
    progress = np.minimum(times, self.stop_time) / self.stop_time
    remaining = 1.0 - progress
    position = self.position + self.velocity * self.stop_time * (progress - progress**2 + progress**3 / 3.0)
    velocity = self.velocity * remaining**2
    acceleration = self.acceleration * remaining
    jerk = np.where(times < self.stop_time, -self.acceleration / self.stop_time, 0.0)
    return np.stack((position, velocity, acceleration), axis=-1), jerk

  def native_trajectory(self, public_times):
    """Endpoint states, with u[0] applying only until the internal rest event."""
    states, _ = self.sample(public_times)
    jerks = np.zeros(len(public_times) - 1)
    jerks[0] = -self.acceleration / self.stop_time
    return states, jerks

  def expanded_trajectory(self, public_times):
    """Expose the event as a knot for independent ordinary-segment checks."""
    times = np.insert(np.asarray(public_times, dtype=float), 1, self.stop_time)
    states, _ = self.sample(times)
    _, jerks = self.sample(times[:-1])
    return times, states, jerks
