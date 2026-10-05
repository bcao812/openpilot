import time

import pyray as rl

from openpilot.cereal import log, messaging
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.sunnypilot.onroad.honda_radar_state import RadarOverlayState, project_radar_point
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached

SERVICE = 'hondaRadarObservations'


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
    self.labels = []

  def render(self, view, rect):
    try:
      self._render(view, rect)
    except Exception:
      # An optional diagnostic overlay cannot take down the main driving UI.
      self.state.clear()
      now = time.monotonic_ns()
      if now - self.last_error_ns > 5_000_000_000:
        self.last_error_ns = now
        cloudlog.exception('Honda radar overlay unavailable')

  def _render(self, view, rect):
    now = time.monotonic_ns()
    if now >= self.next_params_ns:
      self.enabled = (ui_state.params.get('HondaBoschARadarMode', return_default=True) == 1
                      and ui_state.params.get_bool('HondaBoschARadarOverlay'))
      self.camera_offset = ui_state.params.get('CameraOffset', return_default=True) if ui_state.active_bundle else 0.0
      self.next_params_ns = now + 1_000_000_000
      self.layout_key = None  # also refresh font/language measurements
    sm = ui_state.sm
    cp = ui_state.CP
    if (not self.enabled or cp is None or cp.carFingerprint != 'HONDA_ACCORD' or not cp.openpilotLongitudinalControl
        or not ui_state.started or not sm['selfdriveState'].experimentalMode
        or not sm['longitudinalPlanSP'].dec.enabled or not sm['longitudinalPlanSP'].dec.active):
      if self.state.active:
        self.state.clear()
      return
    if any(not sm.valid[name] or not 0 <= now - sm.logMonoTime[name] <= 500_000_000
           for name in ('selfdriveState', 'longitudinalPlanSP')):
      self.state.clear()
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
      return
    frame_ns = int(view.client.timestamp_eof)
    points = self.state.points(frame_ns, now)
    if not self.state.active:
      return
    transform = view.model_renderer._car_space_transform
    height = float(calib.height[0])
    key = (tuple(points), transform.tobytes(), height, self.camera_offset, rect.x, rect.y, rect.width, rect.height, id(self.font))
    if key != self.layout_key:
      self.labels = self._layout_labels(points, transform.tolist(), height, rect)
      self.layout_key = key
    # Scope, source freshness and camera time are checked on every frame above.
    # Cached layout never caches validity or opacity.
    for point, text, box, line in self.labels:
      color = rl.Color(255, 215, 90, 255 if frame_ns - point.time_ns <= 80_000_000 else 140)
      if line is not None:
        rl.draw_line_v(*line, color)
      rl.draw_rectangle_rec(box, rl.Color(0, 0, 0, 130))
      rl.draw_text_ex(self.font, text, rl.Vector2(box.x + 6, box.y), 32, 0, color)
    rl.draw_text_ex(self.font, tr('Radar: slot distance (m), uncalibrated'),
                    rl.Vector2(rect.x + 16, rect.y + 100), 24, 0, rl.Color(255, 215, 90, 230))

  def _layout_labels(self, points, transform, camera_height, rect):
    boxes, labels = [], []
    for point in points:
      position = project_radar_point(point, transform, camera_height, self.camera_offset)
      if position is None:
        continue
      x, y = position
      if not (rect.x <= x <= rect.x + rect.width and rect.y <= y <= rect.y + rect.height):
        continue
      text = point.text
      size = measure_text_cached(self.font, text, 32)
      width, height = size.x + 12, 38
      x = max(rect.x, min(x - width / 2, rect.x + rect.width - width))
      base_y = max(rect.y, min(y - height, rect.y + rect.height - height))
      box_y = base_y
      # Bounded, deterministic label separation. Slot coordinates stay unchanged.
      # At an edge, clamping collapses half of the alternating candidates. Search
      # far enough in either direction to place all 16 labels on a C3 viewport.
      for row in range(32):
        shift = ((row + 1) // 2) * (height + 2) * (-1 if row % 2 else 1)
        candidate_y = max(rect.y, min(base_y + shift, rect.y + rect.height - height))
        # Plain bounds avoid allocating CFFI rectangles and crossing into raylib
        # for every candidate/pair in the dense 16-label case.
        if not any(x < right and x + width > left and candidate_y < bottom and candidate_y + height > top
                   for left, top, right, bottom in boxes):
          box_y = candidate_y
          break
      boxes.append((x, box_y, x + width, box_y + height))
      box = rl.Rectangle(x, box_y, width, height)
      line = None
      if abs(box_y - base_y) > 1:
        # Preserve the actual projected location when text is moved for spacing.
        endpoint = rl.Vector2(max(box.x, min(position[0], box.x + box.width)),
                              max(box.y, min(position[1], box.y + box.height)))
        line = (rl.Vector2(*position), endpoint)
      labels.append((point, text, box, line))
    return labels
