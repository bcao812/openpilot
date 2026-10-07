import time

import pyray as rl

from openpilot.cereal import log, messaging
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.car.honda_radar_observer_config import FUSION_MODES, OBSERVE
from openpilot.selfdrive.ui.sunnypilot.onroad.honda_radar_state import RadarOverlayState, project_radar_point
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_control_influence import accepted_influences, influence_rgb, radar_influence_strength
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app

SERVICE = 'hondaRadarObservations'
MARKER_RADIUS = 6
LABEL_FONT_SIZE = 32


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
      self.enabled = (ui_state.params.get('HondaBoschARadarMode', return_default=True) in (OBSERVE, *FUSION_MODES)
                      and ui_state.params.get_bool('HondaBoschARadarOverlay'))
      self.camera_offset = ui_state.params.get('CameraOffset', return_default=True) if ui_state.active_bundle else 0.0
      self.next_params_ns = now + 1_000_000_000
      self.layout_key = None
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
    influences = accepted_influences(sm, now)
    for point, marker, text_position in self.labels:
      rgb = influence_rgb(radar_influence_strength(point, influences)) or (255, 215, 0)
      color = rl.Color(*rgb, 255 if frame_ns - point.time_ns <= 80_000_000 else 140)
      rl.draw_circle_v(marker, MARKER_RADIUS, color)
      rl.draw_text_ex(self.font, point.text, text_position, LABEL_FONT_SIZE, 0, color)

  def _layout_labels(self, points, transform, camera_height, rect):
    labels = []
    for point in points:
      position = project_radar_point(point, transform, camera_height, self.camera_offset)
      if position is None:
        continue
      x, y = position
      if not (rect.x <= x <= rect.x + rect.width and rect.y <= y <= rect.y + rect.height):
        continue
      # Never shift detections to make room for text. The dot marks the exact
      # ground-plane projection; distance stays immediately to its right.
      labels.append((point, rl.Vector2(x, y), rl.Vector2(x + MARKER_RADIUS + 6, y - LABEL_FONT_SIZE / 2)))
    return labels
