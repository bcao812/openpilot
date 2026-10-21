import time
from dataclasses import dataclass

import pyray as rl

from openpilot.cereal import log, messaging
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.sunnypilot.onroad.honda_radar_state import (
  RadarLabelPoint, RadarOverlayState, project_radar_coordinates, project_radar_point,
)
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_control_influence import accepted_influences, influence_rgb, radar_influence_strength
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, font_fallback, gui_app
from openpilot.system.ui.lib.multilang import multilang
from openpilot.system.ui.lib.text_measure import measure_text_cached

SERVICE = 'hondaRadarObservations'
MARKER_RADIUS = 6
LABEL_FONT_SIZE = 32
BOUNDARY_LINE_WIDTH = 2
BOUNDARY_TICK_HEIGHT = 16
LABEL_GAP = 6


@dataclass(frozen=True)
class RadarLabelLayout:
  point: RadarLabelPoint
  marker: rl.Vector2
  text_position: rl.Vector2
  lines: tuple[tuple[rl.Vector2, rl.Vector2], ...]
  text: str


class HondaRadarOverlay:
  def __init__(self):
    self.state = RadarOverlayState()
    self.sm = None  # no extra subscription while the display is off
    self.enabled = False
    self.next_params_ns = 0
    self.camera_offset = 0.0
    self.last_error_ns = 0
    self.font = None
    self.layout_key = None
    self.layout_context = None
    self.layout_font = None
    self.point_layouts = {}
    self.labels = []

  def _clear_layout(self):
    self.layout_key = None
    self.layout_context = None
    self.layout_font = None
    self.point_layouts = {}
    self.labels = []

  def render(self, view, rect):
    try:
      self._render(view, rect)
    except Exception:
      # An optional diagnostic overlay cannot take down the main driving UI.
      self.state.clear()
      self._clear_layout()
      now = time.monotonic_ns()
      if now - self.last_error_ns > 5_000_000_000:
        self.last_error_ns = now
        cloudlog.exception('Honda radar overlay unavailable')

  def _render(self, view, rect):
    now = time.monotonic_ns()
    if now >= self.next_params_ns:
      self.enabled = ui_state.params.get_bool('HondaBoschARadarOverlay')
      self.camera_offset = ui_state.params.get('CameraOffset', return_default=True) if ui_state.active_bundle else 0.0
      self.next_params_ns = now + 1_000_000_000
      # The layout key below already includes the offset. Polling unchanged
      # settings needs no new projection; scope and freshness remain per-frame.
    sm = ui_state.sm
    cp = ui_state.CP
    if (not self.enabled or cp is None or cp.carFingerprint != 'HONDA_ACCORD' or not cp.openpilotLongitudinalControl
        or not ui_state.started or not sm['selfdriveState'].experimentalMode
        or not sm['longitudinalPlanSP'].dec.enabled or not sm['longitudinalPlanSP'].dec.active):
      if self.state.active:
        self.state.clear()
      self._clear_layout()
      return
    if any(not sm.valid[name] or not 0 <= now - sm.logMonoTime[name] <= 500_000_000
           for name in ('selfdriveState', 'longitudinalPlanSP')):
      self.state.clear()
      self._clear_layout()
      return
    if self.sm is None:
      self.sm = messaging.SubMaster([SERVICE], frequency=20)
    self.sm.update(0)
    if self.sm.updated[SERVICE]:
      if self.sm.valid[SERVICE]:
        self.state.ingest(self.sm[SERVICE], self.sm.logMonoTime[SERVICE], now)
      else:
        self.state.clear()
    if self.font is None:
      self.font = gui_app.font(FontWeight.SEMI_BOLD)
    calib = sm['extrinsicsCalibration']
    if (not sm.valid['extrinsicsCalibration'] or calib.calStatus != log.ExtrinsicsCalibration.Status.calibrated
        or not calib.height or view.frame is None or view.device_camera is None):
      self._clear_layout()
      return
    frame_ns = int(view.client.timestamp_eof)
    points = self.state.points(frame_ns, now)
    if not self.state.active:
      self._clear_layout()
      return
    transform = view.model_renderer._car_space_transform
    height = float(calib.height[0])
    # Text measurement/drawing may use a language fallback even though the
    # overlay's nominal font object has not changed.
    layout_font = font_fallback(self.font)
    font_key = (id(self.font), id(layout_font), getattr(getattr(layout_font, 'texture', None), 'id', None), multilang.language)
    context = (transform.tobytes(), height, self.camera_offset, rect.x, rect.y, rect.width, rect.height, font_key)
    key = (tuple(points), context)
    if key != self.layout_key:
      # A staggered scan often changes just one slot. Rebuild those points in
      # one batch, retaining only the current selection (at most 16 slots).
      # Full point identity includes source/session, sample time and boundaries;
      # a cached offscreen result is still tied to that exact current point.
      previous = self.point_layouts if context == self.layout_context else {}
      current, changed = {}, []
      for point in points:
        cached = previous.get(point.slot)
        if cached is not None and cached[0] == point:
          current[point.slot] = cached
        else:
          current[point.slot] = (point, None)
          changed.append(point)
      if changed:
        for label in self._layout_labels(changed, transform.tolist(), height, rect):
          current[label.point.slot] = (label.point, label)
      self.labels = [cached[1] for point in points if (cached := current[point.slot])[1] is not None]
      self.point_layouts = current
      self.layout_context = context
      self.layout_font = layout_font
      self.layout_key = key
    # Scope, source freshness and camera time are checked on every frame above.
    # Cached layout never caches validity or opacity.
    influences = accepted_influences(sm, now)
    for label in self.labels:
      rgb = influence_rgb(radar_influence_strength(label.point, influences)) or (0, 255, 0)
      color = rl.Color(*rgb, 255 if frame_ns - label.point.time_ns <= 80_000_000 else 140)
      for start, end in label.lines:
        rl.draw_line_ex(start, end, BOUNDARY_LINE_WIDTH, color)
      rl.draw_circle_v(label.marker, MARKER_RADIUS, color)
      rl.draw_text_ex(self.font, label.text, label.text_position, LABEL_FONT_SIZE, 0, color)

  def _layout_labels(self, points, transform, camera_height, rect):
    labels = []

    def in_view(x, y):
      return rect.x <= x <= rect.x + rect.width and rect.y <= y <= rect.y + rect.height

    for point in points:
      position = project_radar_point(point, transform, camera_height, self.camera_offset)
      if position is None:
        continue
      x, y = position
      if not in_view(x, y):
        continue
      marker = rl.Vector2(x, y)
      lines = []
      label_x, label_top = x, y - MARKER_RADIUS
      if point.boundary_laterals is not None:
        ends = [project_radar_coordinates(point.distance, lateral, transform, camera_height, self.camera_offset)
                for lateral in point.boundary_laterals]
        if len(ends) == 2 and all(end is not None for end in ends):
          left, right = (rl.Vector2(*end) for end in ends)
          # The same-distance boundary and reflection point are collinear.
          # AugmentedRoadView's existing content scissor clips lines and text;
          # clipping a line never manufactures an endpoint marker at the border.
          lines.append((left, right))
          for end in (left, right):
            if in_view(end.x, end.y):
              lines.append((rl.Vector2(end.x, end.y - BOUNDARY_TICK_HEIGHT / 2),
                            rl.Vector2(end.x, end.y + BOUNDARY_TICK_HEIGHT / 2)))
          label_x = (left.x + right.x) / 2
          label_top = min(left.y, right.y, y) - BOUNDARY_TICK_HEIGHT / 2
      # Never shift detections to make room for text. Only the transparent
      # distance label is centered above the projected span (or lone dot).
      text = point.text
      size = measure_text_cached(self.font, text, LABEL_FONT_SIZE)
      text_position = rl.Vector2(label_x - size.x / 2, label_top - LABEL_GAP - size.y)
      labels.append(RadarLabelLayout(point, marker, text_position, tuple(lines), text))
    return labels
