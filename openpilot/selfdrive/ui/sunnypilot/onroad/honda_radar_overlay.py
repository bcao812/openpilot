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
    sm = ui_state.sm
    cp = ui_state.CP
    if (not self.enabled or cp is None or cp.carFingerprint != 'HONDA_ACCORD' or not cp.openpilotLongitudinalControl
        or not ui_state.started or not sm['selfdriveState'].experimentalMode
        or not sm['longitudinalPlanSP'].dec.enabled or not sm['longitudinalPlanSP'].dec.active):
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
        self.state.ingest(self.sm[SERVICE].to_dict(), self.sm.logMonoTime[SERVICE], now)
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
    transform = view.model_renderer._car_space_transform.tolist()
    boxes = []
    for point in points:
      position = project_radar_point(point, transform, float(calib.height[0]), self.camera_offset)
      if position is None:
        continue
      x, y = position
      if not (rect.x <= x <= rect.x + rect.width and rect.y <= y <= rect.y + rect.height):
        continue
      size = measure_text_cached(self.font, point.text, 32)
      width, height = size.x + 12, 38
      x = max(rect.x, min(x - width / 2, rect.x + rect.width - width))
      base_y = max(rect.y, min(y - height, rect.y + rect.height - height))
      box = rl.Rectangle(x, base_y, width, height)
      # Bounded, deterministic label separation. Slot coordinates stay unchanged.
      for row in range(16):
        shift = ((row + 1) // 2) * (height + 2) * (-1 if row % 2 else 1)
        candidate = rl.Rectangle(x, max(rect.y, min(base_y + shift, rect.y + rect.height - height)), width, height)
        if not any(rl.check_collision_recs(candidate, old) for old in boxes):
          box = candidate
          break
      boxes.append(box)
      color = rl.Color(255, 215, 90, 255 if frame_ns - point.time_ns <= 80_000_000 else 140)
      rl.draw_rectangle_rec(box, rl.Color(0, 0, 0, 130))
      rl.draw_text_ex(self.font, point.text, rl.Vector2(box.x + 6, box.y), 32, 0, color)
    rl.draw_text_ex(self.font, tr('Radar: slot distance (m), uncalibrated'),
                    rl.Vector2(rect.x + 16, rect.y + 100), 24, 0, rl.Color(255, 215, 90, 230))
