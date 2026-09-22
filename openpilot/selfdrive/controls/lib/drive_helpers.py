import math
import numpy as np
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.common.realtime import DT_CTRL, DT_MDL

MIN_SPEED = 1.0
CONTROL_N = 17
CAR_ROTATION_RADIUS = 0.0
# This is a turn radius smaller than most cars can achieve
MAX_CURVATURE = 0.2
MIN_STABLE_DELAY = 0.3

# EU guidelines
MAX_LATERAL_JERK = 5.0  # m/s^3
MAX_LATERAL_ACCEL_NO_ROLL = 3.0  # m/s^2

STOP_SPEED = 0.3        # m/s, speed below which a stop can be latched
STOP_ACCEL = 0.1        # m/s^2, plan is considered to not be asking for speed below this
STOP_RELEASE_ACCEL = 0.2  # m/s^2, plan must ask for at least this before a latched stop lets go
CREEP_SPEED = 1.2       # m/s, upper bound of the creep band
CREEP_ACCEL = 0.0       # m/s^2, plan must be asking for LESS than this for a creep to accumulate
CREEP_STOP_T = 0.5      # s, window the creep band is judged over
CREEP_FINISH_T = 2.0    # s, how much longer a stop in the creep band may still need
CREEP_MAX_T = 1.0       # s, total time in the band before it is a creep whatever the window says
CLOSE_LEAD_STOP_SPEED = 2.0


def should_stop(v_ego: float, a_target: float) -> bool:
  return bool(v_ego < STOP_SPEED and a_target < STOP_ACCEL)

def clamp(val, min_val, max_val):
  clamped_val = float(np.clip(val, min_val, max_val))
  return clamped_val, clamped_val != val

def smooth_value(val, prev_val, tau, dt=DT_MDL):
  alpha = 1 - np.exp(-dt/tau) if tau > 0 else 1
  return alpha * val + (1 - alpha) * prev_val

def clip_curvature(v_ego, prev_curvature, new_curvature, roll) -> tuple[float, bool]:
  # This function respects ISO lateral jerk and acceleration limits + a max curvature
  v_ego = max(v_ego, MIN_SPEED)
  max_curvature_rate = MAX_LATERAL_JERK / (v_ego ** 2)  # inexact calculation, check https://github.com/commaai/openpilot/pull/24755
  new_curvature = np.clip(new_curvature,
                          prev_curvature - max_curvature_rate * DT_CTRL,
                          prev_curvature + max_curvature_rate * DT_CTRL)

  roll_compensation = roll * ACCELERATION_DUE_TO_GRAVITY
  max_lat_accel = MAX_LATERAL_ACCEL_NO_ROLL + roll_compensation
  min_lat_accel = -MAX_LATERAL_ACCEL_NO_ROLL + roll_compensation
  new_curvature, limited_accel = clamp(new_curvature, min_lat_accel / v_ego ** 2, max_lat_accel / v_ego ** 2)

  new_curvature, limited_max_curv = clamp(new_curvature, -MAX_CURVATURE, MAX_CURVATURE)
  return float(new_curvature), limited_accel or limited_max_curv


def get_accel_from_plan(speeds, accels, t_idxs, action_t=DT_MDL):
  if len(speeds) == len(t_idxs):
    v_now = speeds[0]
    a_now = accels[0]
    if action_t < MIN_STABLE_DELAY:
      v_target = v_now + (action_t / MIN_STABLE_DELAY) * (np.interp(MIN_STABLE_DELAY, t_idxs, speeds) - v_now)
    else:
      v_target = np.interp(action_t, t_idxs, speeds)
    a_target = 2 * (v_target - v_now) / (action_t) - a_now
  else:
    a_target = 0.0
  return a_target


def get_departure_accel_from_plan(speeds, accels, t_idxs, action_t, constraint_accel):
  """Recover the old departure-strength fact, never an execution target.

  Use the published Float32 precision at both ends of the plan transport.
  This inverse can be positive for a stopping trajectory; independent motion
  permission and a positive execution request must still qualify any release.
  Missing or malformed trajectories supply no additional strength evidence.
  """
  scalar_types = (int, float, np.integer, np.floating)
  if any(not isinstance(value, scalar_types) or isinstance(value, (bool, np.bool_))
         for value in (action_t, constraint_accel)):
    return None
  try:
    action_t, constraint_accel = float(action_t), float(constraint_accel)
    if not math.isfinite(action_t) or not math.isfinite(constraint_accel):
      return None
    with np.errstate(over='ignore', invalid='ignore'):
      speeds = np.asarray(speeds, dtype=np.float32).astype(float)
      accels = np.asarray(accels, dtype=np.float32).astype(float)
      constraint_accel = float(np.float32(constraint_accel))
    t_idxs = np.asarray(t_idxs, dtype=float)
  except (TypeError, ValueError, OverflowError):
    return None
  if (t_idxs.ndim != 1 or len(t_idxs) < 2 or speeds.shape != t_idxs.shape or accels.shape != t_idxs.shape
      or not all(np.isfinite(values).all() for values in (speeds, accels, t_idxs))
      or not math.isfinite(constraint_accel) or t_idxs[0] != 0.0 or (t_idxs[1:] <= t_idxs[:-1]).any()
      or action_t <= 0.0 or max(action_t, MIN_STABLE_DELAY) > t_idxs[-1]):
    return None
  strength = get_accel_from_plan(speeds, accels, t_idxs, action_t)
  return float(min(strength, constraint_accel)) if math.isfinite(strength) else None


def curv_from_psis(psi_target, psi_rate, vego, action_t):
  vego = np.clip(vego, MIN_SPEED, np.inf)
  curv_from_psi = psi_target / (vego * action_t)
  return 2*curv_from_psi - psi_rate / vego

def get_curvature_from_plan(yaws, yaw_rates, t_idxs, vego, action_t):
  if action_t < MIN_STABLE_DELAY:
    psi_target = (action_t / MIN_STABLE_DELAY) * np.interp(MIN_STABLE_DELAY, t_idxs, yaws)
  else:
    psi_target = np.interp(action_t, t_idxs, yaws)
  psi_rate = yaw_rates[0]
  return curv_from_psis(psi_target, psi_rate, vego, action_t)
