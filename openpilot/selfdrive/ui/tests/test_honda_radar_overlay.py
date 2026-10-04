from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from openpilot.selfdrive.ui.tests.test_sunnylink_removed import HeadlessUITestCase
from openpilot.selfdrive.car.tests.test_honda_radar_observer import SETTINGS, START, can_message, configuration, scan_frames, state
from openpilot.selfdrive.car.honda_radar_observer import HondaRadarObserver
from openpilot.selfdrive.ui.sunnypilot.onroad.honda_radar_state import RadarLabelPoint, RadarOverlayState, project_radar_point


def sample():
  observer = HondaRadarObserver(*configuration(), session_id='fixture')
  msg = observer.update([can_message(START, scan_frames())], state(START), SETTINGS, START + 1)
  return msg.hondaRadarObservations.to_dict()


class TestHondaRadarOverlay(HeadlessUITestCase):
  def test_format_exactly_slot_space_one_decimal(self):
    for slot, distance, expected in ((0, 8.0, '0 8.0'), (3, 24.74, '3 24.7'), (3, 24.76, '3 24.8'), (15, 120, '15 120.0')):
      self.assertEqual(RadarLabelPoint(slot, distance, 0, START, 1, 1).text, expected)

  def test_camera_time_selection_no_future_or_immortal_labels(self):
    data = sample()
    overlay = RadarOverlayState()
    overlay.ingest(data, START + 1, START + 1)
    self.assertEqual(len(overlay.history[0]), 1)  # position and AUX are one sample
    self.assertFalse(overlay.points(START - 1, START + 1))
    self.assertEqual(overlay.points(START, START + 1)[0].text, '0 54.1')
    self.assertFalse(overlay.points(START + 300_000_000, START + 300_000_000))

  def test_scope_exit_session_change_and_missing_batches_clear_history(self):
    data = sample()
    for update in ({'active': False}, {'sessionId': 'new', 'observations': []}, {'sequence': 3, 'observations': []}):
      overlay = RadarOverlayState()
      overlay.ingest(data, START + 1, START + 1)
      overlay.ingest(dict(data, sequence=2, **update) if 'sequence' not in update else dict(data, **update), START + 2, START + 2)
      self.assertFalse(overlay.points(START, START + 2))

  def test_slot_reuse_and_invalid_geometry(self):
    data = sample()
    overlay = RadarOverlayState()
    overlay.ingest(data, START + 1, START + 1)
    event = dict(data['observations'][0], generation=2, positionSeq=9, positionMonoTime=START + 2, dRel=25.0)
    overlay.ingest(dict(data, sequence=2, observations=[event]), START + 3, START + 3)
    self.assertEqual(len(overlay.history[0]), 1)
    self.assertEqual(overlay.points(START + 2, START + 3)[0].text, '0 25.0')
    for distance in (float('nan'), float('inf'), -1.0):
      bad = dict(event, dRel=distance, positionSeq=10)
      overlay.ingest(dict(data, sequence=overlay.sequence + 1, observations=[bad]), START + 4, START + 4)
      self.assertEqual(len(overlay.history[0]), 1)

  def test_ground_projection_signs_and_depth_for_both_camera_transforms(self):
    for focal in (500.0, 1000.0):
      matrix = ((640, focal, 0), (360, 0, focal), (1, 0, 0))
      left = project_radar_point(RadarLabelPoint(0, 20, 2, START, 1, 1), matrix, 1.2)
      right = project_radar_point(RadarLabelPoint(0, 20, -2, START, 1, 1), matrix, 1.2)
      self.assertLess(left[0], 640)
      self.assertGreater(right[0], 640)
      self.assertAlmostEqual(left[1], 360 + focal * 1.2 / 20)
    bad_depth = ((1, 0, 0), (0, 1, 0), (-1, 0, 0))
    self.assertIsNone(project_radar_point(RadarLabelPoint(0, 20, 0, START, 1, 1), bad_depth, 1.2))

  def test_optional_render_exception_does_not_escape(self):
    from openpilot.selfdrive.ui.sunnypilot.onroad.honda_radar_overlay import HondaRadarOverlay
    overlay = HondaRadarOverlay()
    with patch.object(overlay, '_render', side_effect=ValueError('malformed diagnostic')), \
         patch('openpilot.selfdrive.ui.sunnypilot.onroad.honda_radar_overlay.cloudlog.exception'):
      overlay.render(None, None)
    self.assertFalse(overlay.state.active)

  def test_real_render_path_uses_text_and_camera_frame_time(self):
    import numpy as np
    import pyray as rl
    from openpilot.cereal import log, messaging
    from openpilot.selfdrive.ui.sunnypilot.onroad import honda_radar_overlay as renderer
    data = sample()
    sm = state(START)
    wire = messaging.new_message(renderer.SERVICE)
    setattr(wire, renderer.SERVICE, data)
    calib = messaging.new_message('extrinsicsCalibration').extrinsicsCalibration
    calib.calStatus = log.ExtrinsicsCalibration.Status.calibrated
    calib.height = [1.2]
    sm['extrinsicsCalibration'] = calib
    sm.valid['extrinsicsCalibration'] = True
    observer_sm = Mock(updated={renderer.SERVICE: True}, valid={renderer.SERVICE: True}, logMonoTime={renderer.SERVICE: START + 1})
    observer_sm.__getitem__ = Mock(return_value=getattr(wire, renderer.SERVICE).as_reader())
    overlay = renderer.HondaRadarOverlay()
    overlay.sm = observer_sm
    overlay.enabled = True
    overlay.next_params_ns = START + 1_000_000_000
    view = SimpleNamespace(frame=object(), client=SimpleNamespace(timestamp_eof=START), device_camera=object(),
                           model_renderer=SimpleNamespace(_car_space_transform=np.array(((640., 1000., 0), (360., 0, 1000.), (1., 0, 0)))))
    ui = SimpleNamespace(CP=configuration()[0], started=True, sm=sm)
    with patch.object(renderer, 'ui_state', ui), patch.object(renderer.time, 'monotonic_ns', return_value=START + 2), \
         patch.object(renderer.gui_app, 'font', return_value=object()), \
         patch.object(renderer, 'measure_text_cached', return_value=SimpleNamespace(x=120, y=32)), \
         patch.object(rl, 'draw_rectangle_rec'), patch.object(rl, 'draw_text_ex') as draw:
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      self.assertEqual(draw.call_args_list[0].args[1], '0 54.1')
      self.assertEqual(draw.call_count, 2)
      draw.reset_mock()
      view.client.timestamp_eof = START - 1
      observer_sm.updated[renderer.SERVICE] = False
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      self.assertEqual(draw.call_count, 1)  # legend only; future position not drawn

  def test_settings_callbacks_keep_other_switches_and_are_offroad_only(self):
    from openpilot.selfdrive.ui.sunnypilot.layouts.settings.vehicle.brands import honda
    params = Mock()
    params.get.return_value = 0
    params.get_bool.return_value = False
    fake_ui = SimpleNamespace(params=params, started=False)
    with patch.object(honda, 'ui_state', fake_ui), patch.object(honda, 'toggle_item_sp', side_effect=lambda *a, **kw: Mock()):
      settings = honda.HondaSettings()
      settings._set_observer(True)
      params.put.assert_called_once_with('HondaBoschARadarMode', 1, block=True)
      params.put_bool.assert_not_called()
      fake_ui.started = True
      settings._set_observer(False)
      params.put.assert_called_once()
      settings._set_overlay(True)
      params.put_bool.assert_not_called()


if __name__ == '__main__':
  unittest.main()
