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
from openpilot.sunnypilot.modeld_v2.blendful import LANE_WIDTH_CLIP, LANE_WIDTH_DEFAULT, LOOKAHEAD_TIME, MIN_LOOKAHEAD
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.shader_polygon import draw_polygon
from openpilot.system.ui.lib.text_measure import measure_text_cached

BLENDFUL_COLOR = (0, 200, 255)
FULL_BAR = 1.0
# Both messages normally arrive at 20 Hz. Allow five periods of transport delay,
# but never freeze the cyan line when either producer stops. Model/SP must
# share the exact publication time accepted by the lateral consumer.
MESSAGE_MAX_AGE_NS = int(5 * DT_MDL * 1e9)
# Only opacity can bridge a healthy in-flight consumer receipt, for at most one
# model period. Geometry/readout stay hidden; the first wait deadline never renews.
RECEIPT_WAIT_MAX_AGE_NS = int(DT_MDL * 1e9)
# carControl publishes at 100 Hz; ten periods match its normal alive window.
CONTROL_MAX_AGE_NS = 100_000_000
CURRENT_LINE_PROB_MIN = 0.3


class BlendfulRenderer:
  def __init__(self, readout: bool = True, rect_relative: bool = False):
    self._readout = readout
    self._rect_relative = rect_relative
    self._center_points = np.empty((0, 2), dtype=np.float32)
    self._trust = 0.0
    self._model_ns = 0
    self._output_ns = 0
    self._control_epoch = 0
    self._receipt_ns = 0
    self._side_mask = (False, False)
    self._font = gui_app.font(FontWeight.BOLD) if readout else None
    self._pull_display = FirstOrderFilter(0.0, 0.5, DT_MDL)
    self._trust_display = FirstOrderFilter(0.0, 0.5, DT_MDL)
    self._alpha = FirstOrderFilter(0.0, 0.3, DT_MDL)
    self._clear_opacity_wait()

  def clear(self) -> None:
    self._hide_geometry()
    self._alpha.x = 0.0
    self._clear_opacity_wait()

  def _hide_geometry(self) -> None:
    self._center_points = np.empty((0, 2), dtype=np.float32)
    self._trust = 0.0
    self._model_ns = self._output_ns = 0
    self._control_epoch = self._receipt_ns = 0
    self._side_mask = (False, False)
    self._pull_display.x = self._trust_display.x = 0.0

  def _clear_opacity_wait(self) -> None:
    self._opacity_epoch = 0
    self._opacity_sides = (False, False)
    self._opacity_deadline_ns = 0
    self._opacity_model_ns = 0

  def _wait_for_receipt(self, epoch: int, side_mask: tuple[bool, bool]) -> None:
    sm = ui_state.sm
    if self._opacity_deadline_ns:
      if (epoch != self._opacity_epoch or side_mask != self._opacity_sides or
          time.monotonic_ns() > self._opacity_deadline_ns):
        self.clear()
        return
    else:
      if epoch != self._control_epoch or side_mask != self._side_mask:
        self.clear()
        return
      self._opacity_epoch, self._opacity_sides = epoch, side_mask
      self._opacity_deadline_ns = min(sm.logMonoTime["modelV2"] + RECEIPT_WAIT_MAX_AGE_NS,
                                     sm["modelV2"].timestampEof + MESSAGE_MAX_AGE_NS)
    self._opacity_model_ns = sm.logMonoTime["modelV2"]
    self._hide_geometry()

  @staticmethod
  def _current_messages(allow_pending_receipt: bool = False) -> bool:
    sm = ui_state.sm
    if (not ui_state.active_runner_tinygrad or not ui_state.params.get_bool("Blendful") or
        ui_state.params.get("ModelRunnerTypeCache") != custom.ModelManagerSP.Runner.tinygrad or
        not UIStateSP.lateral_active(ui_state)):
      return False
    now_ns = time.monotonic_ns()
    for service, max_age in (("modelDataV2SP", MESSAGE_MAX_AGE_NS), ("modelV2", MESSAGE_MAX_AGE_NS),
                             ("carControl", CONTROL_MAX_AGE_NS), ("carControlSP", CONTROL_MAX_AGE_NS)):
      stamp = sm.logMonoTime.get(service, 0)
      if (not sm.valid.get(service, False) or not sm.seen.get(service, False) or
          sm.recv_frame.get(service, -1) < ui_state.started_frame or
          not 0 < stamp <= now_ns or now_ns - stamp > max_age):
        return False
    receipt = sm["carControlSP"]
    epoch = receipt.blendfulControlEpoch
    capture_ns = sm["modelV2"].timestampEof
    model_ns = sm.logMonoTime["modelV2"]
    if not (sm["carControl"].latActive and 0 < epoch == sm["modelDataV2SP"].blendfulControlEpoch and
            model_ns == sm.logMonoTime["modelDataV2SP"] and epoch <= capture_ns <= now_ns and
            now_ns - capture_ns <= MESSAGE_MAX_AGE_NS and np.isfinite(receipt.blendfulAppliedCurvature)):
      return False
    if receipt.blendfulModelMonoTime == model_ns:
      return True
    # An explicit zero receipt is withdrawal. Only an accepted older frame whose
    # receipt predates this publication can be a normal producer/consumer phase gap.
    return (allow_pending_receipt and 0 < receipt.blendfulModelMonoTime < model_ns and
            sm.logMonoTime["carControlSP"] < model_ns and now_ns - model_ns <= RECEIPT_WAIT_MAX_AGE_NS)

  def needs_update(self) -> bool:
    """A late consumer receipt can accept geometry that arrived a frame earlier.

    Routine receipt updates do not require another lane projection. Rebuild only
    after withdrawal or when the accepted model/session has changed.
    """
    sm = ui_state.sm
    return (self._model_ns != sm.logMonoTime.get("modelV2", 0) or
            self._output_ns != sm.logMonoTime.get("modelDataV2SP", 0) or
            self._control_epoch != sm["carControlSP"].blendfulControlEpoch)

  def update(self, renderer, max_idx: int, *map_args) -> None:
    sm = ui_state.sm
    if not self._current_messages(allow_pending_receipt=True) or renderer._model_geometry_ns != sm.logMonoTime["modelV2"]:
      self.clear()
      return

    blendful = sm["modelDataV2SP"]
    trust = blendful.blendfulTrust
    probs = [blendful.blendfulLeftProb, blendful.blendfulRightProb]
    current_probs = list(sm["modelV2"].laneLineProbs)
    stds = list(sm["modelV2"].laneLineStds)
    if (not np.isfinite(trust) or not 0 < trust <= 1 or not np.isfinite(blendful.blendfulCurvature) or
        len(stds) < 3 or len(current_probs) < 3 or len(sm["modelV2"].laneLines) < 3 or
        not all(np.isfinite(p) and 0 <= p <= 1 for p in probs)):
      self.clear()
      return

    # Diagnostic confidence cannot qualify a boundary absent from the current
    # model observation. Do this before touching that side's geometry or std.
    for side, current in enumerate(current_probs[1:3]):
      if not np.isfinite(current) or not CURRENT_LINE_PROB_MIN <= current <= 1:
        probs[side] = 0.0

    # Ignore zero-weight sides before inspecting or multiplying their geometry:
    # 0 * NaN is still NaN. A currently weighted invalid line withdraws the display.
    lines, weights = [], []
    for index, prob in zip((1, 2), probs, strict=True):
      line = renderer._lane_lines[index].raw_points
      if prob > 0:
        if (line.ndim != 2 or line.shape[1] != 3 or len(line) < 2 or not np.isfinite(line).all() or
            not np.isfinite(stds[index]) or stds[index] < 0):
          self.clear()
          return
        lines.append(line)
        weights.append(prob / max(stds[index] ** 2, 1e-4))
      else:
        lines.append(None)
        weights.append(0.0)

    total_weight = sum(weights)
    if total_weight <= 0 or not np.isfinite(total_weight):
      self.clear()
      return
    left, right = lines
    if left is not None and right is not None and len(left) != len(right):
      self.clear()
      return
    side_mask = tuple(weight > 0 for weight in weights)
    if not all(side_mask):
      # A lone current boundary must not inherit width learned from a lost one.
      half_width = LANE_WIDTH_DEFAULT / 2
    elif probs[0] * probs[1] > 0.5:
      half_width = np.clip(right[:, 1] - left[:, 1], *LANE_WIDTH_CLIP) / 2
    else:
      lane_width = blendful.blendfulLaneWidth
      if not np.isfinite(lane_width) or not LANE_WIDTH_CLIP[0] <= lane_width <= LANE_WIDTH_CLIP[1]:
        self.clear()
        return
      half_width = lane_width / 2

    center = (left if left is not None else right).copy()
    center[:, 1] = 0.0
    if left is not None:
      center[:, 1] += (weights[0] / total_weight) * (left[:, 1] + half_width)
    if right is not None:
      center[:, 1] += (weights[1] / total_weight) * (right[:, 1] - half_width)

    if not np.isfinite(center).all() or (self._readout and not np.isfinite(sm["carState"].vEgo)):
      self.clear()
      return

    epoch = sm["carControlSP"].blendfulControlEpoch
    if sm["carControlSP"].blendfulModelMonoTime != sm.logMonoTime["modelV2"]:
      # Current lane geometry has been validated, but no old geometry or number
      # is shown while waiting for the consumer to accept this observation.
      self._wait_for_receipt(epoch, side_mask)
      return

    points = renderer._map_line_to_polygon(center, 0.15, 0.0, max_idx, *map_args)
    if points.size == 0 or not np.isfinite(points).all():
      self.clear()
      return
    output_ns = sm.logMonoTime["modelDataV2SP"]
    if side_mask != self._side_mask or epoch != self._control_epoch:
      if not (self._opacity_epoch == epoch and self._opacity_sides == side_mask and
              time.monotonic_ns() <= self._opacity_deadline_ns):
        self.clear()
    self._clear_opacity_wait()
    if self._output_ns != output_ns:
      self._alpha.update(1.0)
    self._center_points = points
    self._trust = trust
    self._side_mask = side_mask
    self._model_ns, self._output_ns = sm.logMonoTime["modelV2"], output_ns
    self._control_epoch = epoch
    self._update_readout()

  def _update_readout(self) -> None:
    if not self._readout:
      return
    sm = ui_state.sm
    receipt_ns = sm.logMonoTime["carControlSP"]
    if receipt_ns == self._receipt_ns:
      return
    # This is the accepted request before curvature clipping, not a measurement
    # of the wheel's response. Receipt cadence can be faster than model cadence.
    dt = DT_MDL if not self._receipt_ns else max((receipt_ns - self._receipt_ns) * 1e-9, 0.0)
    for display_filter in (self._pull_display, self._trust_display):
      display_filter.dt = dt
      display_filter.update_alpha(0.5)
    lookahead = max(sm["carState"].vEgo * LOOKAHEAD_TIME, MIN_LOOKAHEAD)
    self._pull_display.update(sm["carControlSP"].blendfulAppliedCurvature * lookahead ** 2 / 2)
    self._trust_display.update(self._trust)
    self._receipt_ns = receipt_ns

  def _color(self, rgb, alpha: float) -> rl.Color:
    return rl.Color(*rgb, int(alpha * self._alpha.x))

  def draw(self, rect: rl.Rectangle) -> None:
    # Draw also runs on frames with no new messages. Health and parameter checks
    # therefore cannot live only in the 20 Hz update branch.
    if not self._current_messages():
      if (self._opacity_deadline_ns and time.monotonic_ns() <= self._opacity_deadline_ns and
          self._opacity_model_ns == ui_state.sm.logMonoTime.get("modelV2", 0) and
          self._opacity_epoch == ui_state.sm["carControlSP"].blendfulControlEpoch and
          self._current_messages(allow_pending_receipt=True)):
        self._hide_geometry()
      else:
        self.clear()
      return
    if (self._model_ns != ui_state.sm.logMonoTime["modelV2"] or
        self._output_ns != ui_state.sm.logMonoTime["modelDataV2SP"] or
        self._control_epoch != ui_state.sm["carControlSP"].blendfulControlEpoch or
        (self._readout and not np.isfinite(ui_state.sm["carState"].vEgo))):
      self.clear()
      return
    self._update_readout()
    if self._center_points.size and self._trust > 0.05:
      points = self._center_points + (rect.x, rect.y) if self._rect_relative else self._center_points
      draw_polygon(rect, points, self._color(BLENDFUL_COLOR, np.clip(self._trust, 0.2, 1.0) * 255))
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
