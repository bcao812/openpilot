"""
Copyright (c) 2021-, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import time
import numpy as np
import pyray as rl

from openpilot.cereal import custom
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.sunnypilot.ui_state import UIStateSP
from openpilot.sunnypilot.modeld_v2.blendful import LANE_WIDTH_CLIP, LOOKAHEAD_TIME, MIN_LOOKAHEAD
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.shader_polygon import draw_polygon, PreparedPolygon
from openpilot.system.ui.lib.text_measure import measure_text_cached

BLENDFUL_COLOR = (0, 200, 255)
FULL_BAR = 1.0
# Both messages normally arrive at 20 Hz. Allow five periods of transport delay,
# but never freeze the cyan line when either producer stops. Adjacent model
# frames must not borrow one another's Blendful confidence/geometry.
MESSAGE_MAX_AGE_NS = int(5 * DT_MDL * 1e9)
MESSAGE_PAIR_MAX_SKEW_NS = int(DT_MDL / 2 * 1e9)
# carControl publishes at 100 Hz; ten periods match its normal alive window.
CONTROL_MAX_AGE_NS = 100_000_000


class BlendfulRenderer:
  def __init__(self, readout: bool = True, rect_relative: bool = False):
    self._readout = readout
    self._rect_relative = rect_relative
    self._empty_points = np.empty((0, 2), dtype=np.float32)
    self._center_points = self._empty_points
    self._prepared_polygon: PreparedPolygon | None = None
    self._prepared_source: np.ndarray | None = None
    self._prepared_origin: tuple[float, float] | None = None
    self._trust = 0.0
    self._model_ns = 0
    self._output_ns = 0
    self._font = gui_app.font(FontWeight.BOLD) if readout else None
    self._pull_display = FirstOrderFilter(0.0, 0.5, DT_MDL)
    self._trust_display = FirstOrderFilter(0.0, 0.5, DT_MDL)
    self._alpha = FirstOrderFilter(0.0, 0.3, DT_MDL)

  def clear(self) -> None:
    self._clear_geometry()
    self._trust = 0.0
    self._model_ns = self._output_ns = 0
    self._alpha.x = self._pull_display.x = self._trust_display.x = 0.0

  def _invalidate_polygon(self) -> None:
    self._prepared_polygon = self._prepared_source = self._prepared_origin = None

  def _clear_geometry(self) -> None:
    self._center_points = self._empty_points
    self._invalidate_polygon()

  @staticmethod
  def _current_messages() -> bool:
    sm = ui_state.sm
    if (not ui_state.active_runner_tinygrad or not ui_state.params.get_bool("Blendful") or
        ui_state.params.get("ModelRunnerTypeCache") != custom.ModelManagerSP.Runner.tinygrad or
        not UIStateSP.lateral_active(ui_state)):
      return False
    now_ns = time.monotonic_ns()
    for service, max_age in (("modelDataV2SP", MESSAGE_MAX_AGE_NS), ("modelV2", MESSAGE_MAX_AGE_NS),
                             ("carControl", CONTROL_MAX_AGE_NS)):
      stamp = sm.logMonoTime.get(service, 0)
      if (not sm.valid.get(service, False) or sm.recv_frame.get(service, -1) < ui_state.started_frame or
          not 0 < stamp <= now_ns or now_ns - stamp > max_age):
        return False
    return (sm["carControl"].latActive and
            abs(sm.logMonoTime["modelV2"] - sm.logMonoTime["modelDataV2SP"]) <= MESSAGE_PAIR_MAX_SKEW_NS)

  def update(self, renderer, max_idx: int, *map_args) -> None:
    sm = ui_state.sm
    if not self._current_messages() or renderer._model_geometry_ns != sm.logMonoTime["modelV2"]:
      self.clear()
      return

    blendful = sm["modelDataV2SP"]
    trust = blendful.blendfulTrust
    l_prob, r_prob = blendful.blendfulLeftProb, blendful.blendfulRightProb
    stds = list(sm["modelV2"].laneLineStds)
    if (not np.isfinite(trust) or not 0 <= trust <= 1 or not np.isfinite(blendful.blendfulCurvature) or
        len(stds) < 3 or len(sm["modelV2"].laneLines) < 3 or len(renderer._lane_lines) < 3 or
        not all(np.isfinite(p) and 0 <= p <= 1 for p in (l_prob, r_prob)) or
        (self._readout and not np.isfinite(sm["carState"].vEgo))):
      self.clear()
      return

    left, right = renderer._lane_lines[1].raw_points, renderer._lane_lines[2].raw_points
    if (any(line.ndim != 2 or line.shape[1] != 3 or len(line) < 2 or not np.isfinite(line).all() for line in (left, right)) or
        len(left) != len(right) or any(not np.isfinite(std) or std < 0 for std in stds[1:3])):
      self.clear()
      return

    output_ns = sm.logMonoTime["modelDataV2SP"]
    if self._output_ns != output_ns:
      self._alpha.update(float(trust > 0.0))
      if trust > 0.0:
        self._trust = trust
      if self._readout:
        lookahead = max(sm["carState"].vEgo * LOOKAHEAD_TIME, MIN_LOOKAHEAD)
        self._pull_display.update(blendful.blendfulCurvature * lookahead ** 2 / 2)
        self._trust_display.update(self._trust)
    # These identify the healthy inputs processed, not the age of the last fit.
    # Source behavior retains that fit while fresh confidence/alpha lets go.
    self._model_ns, self._output_ns = sm.logMonoTime["modelV2"], output_ns
    if self._alpha.x < 0.01:
      self._clear_geometry()
      return
    if trust <= 0.0 or l_prob + r_prob <= 0.0:
      return

    if l_prob * r_prob > 0.5:
      half_width = np.clip(right[:, 1] - left[:, 1], *LANE_WIDTH_CLIP) / 2
    else:
      lane_width = blendful.blendfulLaneWidth
      if not np.isfinite(lane_width) or not LANE_WIDTH_CLIP[0] <= lane_width <= LANE_WIDTH_CLIP[1]:
        self.clear()
        return
      half_width = lane_width / 2

    # Match the source's variance weighting, including its probability floor and
    # learned width for weak/single-sided confidence. Raw probabilities do not
    # override modeld's filtered confidence here.
    l_var = stds[1] ** 2 / max(l_prob, 1e-4)
    r_var = stds[2] ** 2 / max(r_prob, 1e-4)
    variance = l_var + r_var
    if not np.isfinite(variance) or variance <= 0:
      self.clear()
      return
    center = left.copy()
    center[:, 1] = (r_var * (left[:, 1] + half_width) + l_var * (right[:, 1] - half_width)) / variance
    if not np.isfinite(center).all():
      self.clear()
      return

    points = renderer._map_line_to_polygon(center, 0.15, 0.0, max_idx, *map_args)
    if points.size == 0 or not np.isfinite(points).all():
      self.clear()
      return

    if self._center_points is not points:
      self._invalidate_polygon()
      self._center_points = points

  def _color(self, rgb, alpha: float) -> rl.Color:
    return rl.Color(*rgb, int(alpha * self._alpha.x))

  def draw(self, rect: rl.Rectangle) -> None:
    # Draw also runs on frames with no new messages. Health and parameter checks
    # therefore cannot live only in the 20 Hz update branch.
    if (not self._current_messages() or self._model_ns != ui_state.sm.logMonoTime["modelV2"] or
        self._output_ns != ui_state.sm.logMonoTime["modelDataV2SP"]):
      self.clear()
      return
    if self._center_points.size and self._trust > 0.05:
      origin = (rect.x, rect.y) if self._rect_relative else None
      if self._prepared_polygon is None or self._prepared_source is not self._center_points or self._prepared_origin != origin:
        # Keep the original MICI addition before float32 conversion. One current
        # snapshot also covers source-history holds; color/health stay per draw.
        points = self._center_points + (rect.x, rect.y) if self._rect_relative else self._center_points
        self._prepared_polygon = PreparedPolygon(points)
        self._prepared_source, self._prepared_origin = self._center_points, origin
      draw_polygon(rect, self._prepared_polygon, self._color(BLENDFUL_COLOR, np.clip(self._trust, 0.2, 1.0) * 255))
    if self._center_points.size and self._readout:
      self._draw_readout(rect)

  def _draw_readout(self, rect: rl.Rectangle) -> None:
    pull, trust = self._pull_display.x, self._trust_display.x
    w, h = 190, 76
    x, y = rect.x + (rect.width - w) / 2, rect.y + 10
    cx, cy = x + w / 2, y + 30
    rl.draw_rectangle_rounded(rl.Rectangle(x, y, w, h), 0.5, 12, self._color((0, 0, 0), 170))

    text = f"{abs(pull) * 100:.0f}"
    text_w = measure_text_cached(self._font, text, 44, 0).x
    rl.draw_text_ex(self._font, text, rl.Vector2(cx - text_w / 2, cy - 24), 44, 0, self._color((255, 255, 255), 255))

    active = self._color(BLENDFUL_COLOR, np.clip(abs(pull) / 0.1, 0.0, 1.0) * 255)
    idle = self._color((255, 255, 255), 40)
    lx, rx = x + 18, x + w - 18
    rl.draw_triangle(rl.Vector2(lx, cy), rl.Vector2(lx + 16, cy + 12), rl.Vector2(lx + 16, cy - 12), active if pull < 0 else idle)
    rl.draw_triangle(rl.Vector2(rx - 16, cy - 12), rl.Vector2(rx - 16, cy + 12), rl.Vector2(rx, cy), active if pull > 0 else idle)

    bar = rl.Rectangle(x + 24, y + h - 16, w - 48, 6)
    rl.draw_rectangle_rounded(bar, 1.0, 6, self._color((255, 255, 255), 40))
    rl.draw_rectangle(int(cx - 1), int(bar.y - 2), 2, int(bar.height + 4), self._color((255, 255, 255), 120))
    fill = bar.width / 2 * float(np.sqrt(np.clip(abs(pull) / FULL_BAR, 0.0, 1.0)))
    if fill > 1:
      fill_x = cx if pull > 0 else cx - fill
      rl.draw_rectangle_rounded(rl.Rectangle(fill_x, bar.y, fill, bar.height), 1.0, 6,
                                self._color(BLENDFUL_COLOR, np.clip(0.3 + 0.7 * trust, 0.0, 1.0) * 255))
