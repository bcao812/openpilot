"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants

LANE_WIDTH_DEFAULT = 3.6  # m; fixed prior when only one current boundary is usable
LANE_WIDTH_CLIP = (2.8, 4.5)  # m
LOOKAHEAD_TIME = 1.5  # s
MIN_LOOKAHEAD = 10.0  # m
MIN_SPEED = 5.0  # m/s
MIN_LINE_PROB = 0.3
LINE_STD = (0.2, 0.6)
POINT_STD = 0.1
YIELD = (0.2, 0.45)
X_IDXS = np.array(ModelConstants.X_IDXS)


class Blendful:
  """Lane-center correction with filtering only while its contributing lines remain current."""

  def __init__(self):
    self.reset()

  def reset(self, model_curvature: float = 0.0) -> None:
    self.lane_width = FirstOrderFilter(LANE_WIDTH_DEFAULT, 10.0, DT_MDL, initialized=False)
    self.line_probs_slow = FirstOrderFilter(np.zeros(2), 3.0, DT_MDL)
    self.line_probs = FirstOrderFilter(np.zeros(2), 0.5, DT_MDL)
    self.usual_offset = FirstOrderFilter(0.0, 10.0, DT_MDL, initialized=False)
    self.current_sides = (False, False)
    self.going_around = False
    self.since_turning = 0.0
    self.trust = self.line_trust = self.dk = 0.0
    self.l_prob = self.r_prob = 0.0
    self.model_curvature = model_curvature
    self.smooth = self.curvature = 0.0

  def update(self, model_output: dict[str, np.ndarray], v_ego: float, active: bool, model_curvature: float, enabled: bool = True) -> float:
    valid_model_curvature = 0.0
    try:
      v_ego, model_curvature = float(v_ego), float(model_curvature)
      finite_curvature = np.isfinite(model_curvature)
      valid_model_curvature = model_curvature if finite_curvature else 0.0
      if not enabled or not active or not np.isfinite(v_ego) or v_ego < MIN_SPEED or not finite_curvature:
        self.reset(model_curvature if finite_curvature else 0.0)
        return 0.0

      # Reject numerical failures locally: this optional feature must not turn a
      # malformed lane/fit into a modeld exception or a nonfinite correction.
      with np.errstate(over='raise', invalid='raise', divide='raise'):
        target = self._target(model_output, v_ego, model_curvature)
        if target is None:
          self.reset(model_curvature)
          return 0.0
        self.model_curvature = model_curvature
        self.smooth += (target - self.smooth) * DT_MDL / (DT_MDL + np.interp(abs(model_curvature) * v_ego ** 2,
                                                                                          [0.18, 0.53], [0.5, 0.0]))
        build = 0.5 / max(v_ego, 1.0) ** 2 * DT_MDL
        release = 2.0 / max(v_ego, 1.0) ** 2 * DT_MDL
        k = self.curvature
        self.curvature = float(np.clip(self.smooth, max(k - release, -build) if k > 0 else k - build,
                                                  min(k + release, build) if k < 0 else k + build))
        if not np.isfinite([self.curvature, self.smooth, self.trust, self.line_trust, self.dk, self.lane_width.x,
                            self.l_prob, self.r_prob, self.usual_offset.x]).all():
          self.reset(model_curvature)
      return self.curvature
    except (KeyError, IndexError, TypeError, ValueError, OverflowError, FloatingPointError, np.linalg.LinAlgError):
      self.reset(valid_model_curvature)
      return 0.0

  def _target(self, model_output: dict[str, np.ndarray], v_ego: float, model_curvature: float) -> float | None:
    lookahead = max(v_ego * LOOKAHEAD_TIME, MIN_LOOKAHEAD)
    lines_all = np.asarray(model_output['lane_lines'], dtype=float)
    stds_all = np.asarray(model_output['lane_lines_stds'], dtype=float)
    probs_all = np.asarray(model_output['lane_lines_prob'], dtype=float)
    plan_all = np.asarray(model_output['plan'], dtype=float)
    line_shape = (1, 4, len(X_IDXS), 2)
    if (lines_all.shape != line_shape or stds_all.shape != line_shape or probs_all.shape != (1, 8) or
        plan_all.ndim != 3 or plan_all.shape[:2] != (1, len(X_IDXS)) or plan_all.shape[2] not in (3, 15)):
      return None
    plan = plan_all[0]
    # A valid stopping plan can have a flat tail. Keep the source's endpoint
    # interpolation when the planned travel is shorter than the lane-fit window.
    if not np.isfinite(plan).all() or not np.all(np.diff(plan[:, 0]) >= 0) or plan[-1, 0] <= plan[0, 0]:
      return None

    win = (X_IDXS >= 0.5 * lookahead) & (X_IDXS <= 1.5 * lookahead)
    x = X_IDXS[win]
    if len(x) < 3 or plan[0, 0] > x[0]:
      return None

    raw_probs = probs_all[0, (3, 5)]
    reliability = np.zeros(2)
    # Qualify raw, current measurements BEFORE looking at filtered confidence.
    # An excluded side's coordinates/stds never enter center or width arithmetic.
    for side, index in enumerate((1, 2)):
      if not np.isfinite(raw_probs[side]) or not MIN_LINE_PROB <= raw_probs[side] <= 1.0:
        continue
      if (not np.isfinite(lines_all[0, index]).all() or not np.isfinite(stds_all[0, index]).all() or
          not np.all(stds_all[0, index] > 0)):
        continue
      reliability[side] = np.interp(np.interp(0.5 * lookahead, X_IDXS, stds_all[0, index, :, 0]), LINE_STD, [1.0, 0.0])
    sides = tuple(bool(value > 0) for value in reliability)
    if not any(sides):
      return None
    if sides != self.current_sides:
      # A former boundary must not survive indirectly through width, confidence,
      # the fitted offset, trust, or either correction smoother.
      self.reset(model_curvature)
      self.current_sides = sides

    current_probs = np.array([raw_probs[i] if present else 0.0 for i, present in enumerate(sides)])
    probs = self.line_probs.update(current_probs)
    slow_probs = self.line_probs_slow.update(current_probs)
    dropping = bool(np.any((probs < 0.6 * slow_probs) & np.array(sides)))
    self.l_prob, self.r_prob = (float(value) for value in probs * reliability)
    l_prob, r_prob = self.l_prob, self.r_prob
    lone_weight = np.interp(abs(model_curvature) * v_ego ** 2, [0.18, 0.53], [0.5, 1.0])
    if not all(sides):
      lane_trust = (l_prob + r_prob) * lone_weight
    elif dropping:
      lane_trust = min(l_prob, r_prob)
    else:
      lane_trust = l_prob + r_prob - l_prob * r_prob
    if lone_weight > 0.5 and lane_trust < self.line_trust:
      lane_trust = self.line_trust + (lane_trust - self.line_trust) * DT_MDL / (lone_weight - 0.5 + DT_MDL)
    self.line_trust = lane_trust

    if all(sides):
      left, right = lines_all[0, 1, :, 0], lines_all[0, 2, :, 0]
      half_width = self.lane_width.x / 2
      half_width_now = half_width
      if l_prob * r_prob > 0.5:
        width = np.interp(lookahead, X_IDXS, right) - np.interp(lookahead, X_IDXS, left)
        self.lane_width.update(float(np.clip(width, *LANE_WIDTH_CLIP)))
        half_width = np.clip(right[win] - left[win], *LANE_WIDTH_CLIP) / 2
        half_width_now = np.clip(right[0] - left[0], *LANE_WIDTH_CLIP) / 2
      l_var = stds_all[0, 1, win, 0] ** 2 / max(l_prob, 1e-4)
      r_var = stds_all[0, 2, win, 0] ** 2 / max(r_prob, 1e-4)
      center = (r_var * (left[win] + half_width) + l_var * (right[win] - half_width)) / (l_var + r_var)
      weight = 1.0 / (l_var * r_var / (l_var + r_var) + POINT_STD ** 2)
      center_now = (l_prob * (left[0] + half_width_now) + r_prob * (right[0] - half_width_now)) / (l_prob + r_prob)
    else:
      # A lone current line uses the disclosed fixed width, never a width
      # learned from a boundary missing in this observation.
      side = 0 if sides[0] else 1
      index = side + 1
      line = lines_all[0, index, :, 0]
      self.lane_width.x = LANE_WIDTH_DEFAULT
      half_width = LANE_WIDTH_DEFAULT / 2 if side == 0 else -LANE_WIDTH_DEFAULT / 2
      center = line[win] + half_width
      variance = stds_all[0, index, win, 0] ** 2 / max(l_prob + r_prob, 1e-4)
      weight = 1.0 / (variance + POINT_STD ** 2)
      center_now = line[0] + half_width

    # Fit curvature to place the plant in center: e ~ dk x^2 / 2.
    plan_x = np.interp(x, plan[:, 0], plan[:, 1])
    plan_share = np.interp(abs(model_curvature) * lookahead ** 2 / 2, [0.5, 2.0], [0.3, 0.5])
    error = np.clip(center - ((1 - plan_share) * model_curvature * x ** 2 / 2 + plan_share * plan_x), -1.0, 1.0)
    denominator = np.sum(weight * x ** 4)
    if not np.isfinite(denominator) or denominator <= 0:
      return None
    dk = 2.0 * np.sum(weight * error * x ** 2) / denominator

    # Preserve source yielding when the current model plan goes around an
    # obstacle rather than pulling it back toward its usual lane position.
    plan_y = np.interp(lookahead, plan[:, 0], plan[:, 1])
    offset = plan_y - np.interp(lookahead, x, center)
    usual = self.usual_offset.x if self.usual_offset.initialized else offset
    swerve = abs(offset - usual) - 0.25 * abs(model_curvature) * lookahead ** 2 / 2
    away = swerve - abs(-center_now - usual)
    lane_curvature = 2.0 * np.polyfit(x, center, 2)[0]
    inside_curve = abs(lane_curvature) * lookahead ** 2 / 2 > 0.1 and (offset - usual) * lane_curvature > 0
    self.since_turning = 0.0 if abs(model_curvature) * lookahead ** 2 / 2 > 0.5 else self.since_turning + DT_MDL
    if inside_curve:
      self.going_around = False
    elif away > YIELD[0] and self.since_turning > 2.0:
      self.going_around = True
    elif swerve < YIELD[0] / 2:
      self.going_around = False
    self.usual_offset.update_alpha(20.0 if self.going_around else 10.0)
    self.usual_offset.update(offset)

    if inside_curve:
      lane_trust *= np.interp(abs(offset) - abs(center_now) - 0.25 * abs(model_curvature) * lookahead ** 2 / 2, (0.5, 1.0), [1.0, 0.0])
    else:
      lane_trust *= (np.interp(swerve, (YIELD[0] / 2, YIELD[0]), [1.0, 0.0]) if self.going_around else
                     np.interp(away, YIELD, [1.0, 0.0]))

    if lane_trust < self.trust:
      dk = float(np.clip(dk, min(self.dk, 0.0), max(self.dk, 0.0)))
    self.trust = lane_trust if lane_trust < self.trust else self.trust + (lane_trust - self.trust) * DT_MDL / (1.5 + DT_MDL)
    self.dk = dk
    return float(self.trust * dk)
