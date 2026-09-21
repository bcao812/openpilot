"""The existing speed-error/jerk cruise response, independent of final arbitration."""

import numpy as np

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL


A_CRUISE_MIN = -1.2
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
J_CRUISE_VALS = [1.6, 1.2, 0.8, 0.6]
CRUISE_COAST_BP = [-0.3, -0.25, -0.15, -0.1]
CRUISE_COAST_ACCEL = -0.19
CRUISE_COAST_MIN_SPEED = 10.0
CRUISE_COAST_MAX_SPEED_ERROR = 1.0 * CV.KPH_TO_MS
CRUISE_COAST_RISK_ACCEL = -0.5


def get_cruise_coast_accel(a_cruise: float) -> float:
  """P's coast command mapping; the caller owns eligibility and arbitration.

  This stateless mapping must not replace CruiseResponse's raw jerk history.
  """
  if not CRUISE_COAST_BP[0] < a_cruise < CRUISE_COAST_BP[-1]:
    return a_cruise
  return float(np.interp(a_cruise, CRUISE_COAST_BP,
                         [CRUISE_COAST_BP[0], CRUISE_COAST_ACCEL, CRUISE_COAST_ACCEL, CRUISE_COAST_BP[-1]]))


class CruiseResponse:
  """Own only the previous cruise response, advanced once per model update.

  The caller selects the target speed, computes the existing acceleration
  limit, and validates inputs. Target-source changes preserve this history.
  This helper neither owns a motion phase nor consumes the final actuator
  request. Its result is one acceleration ceiling for the caller's arbitration.
  """

  def __init__(self, dt: float = DT_MDL, *, init_a: float = 0.0):
    self.dt = dt
    self._previous_accel: float | None = float(init_a)

  def clear(self) -> None:
    """Discard history; the next update seeds from measured acceleration."""
    self._previous_accel = None

  def update(self, target_speed: float, speed: float, maximum_accel: float, *, measured_accel: float, reset: bool = False) -> float:
    """Apply the pre-refactor reset, target clamp, then jerk clamp in that order.

    Clear/reset seeds from the globally clipped measurement and still advances
    this model step. Otherwise the constructor/prior response state is retained.
    A falling maximum may be below the jerk-limited result; the caller must
    independently enforce that maximum in final arbitration.
    """
    previous = self._previous_accel
    if reset or previous is None:
      previous = float(np.clip(measured_accel, ACCEL_MIN, ACCEL_MAX))

    target = np.clip(target_speed - speed, A_CRUISE_MIN, maximum_accel)
    jerk = np.interp(speed, A_CRUISE_MAX_BP, J_CRUISE_VALS)
    self._previous_accel = float(np.clip(target, previous - jerk * self.dt, previous + jerk * self.dt))
    return self._previous_accel
