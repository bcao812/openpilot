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
  def test_late_aux_cannot_restore_a_slot_after_invalid_geometry(self):
    observer = HondaRadarObserver(*configuration(), session_id='fixture')
    overlay = RadarOverlayState()
    frames = scan_frames()
    for t, incoming in ((START, frames[:4]),
                        (START + 60_000_000, scan_frames(index=1, life=3, distance=4095)),
                        (START + 120_000_000, frames[4:])):
      msg = observer.update([can_message(t, incoming)], state(t), SETTINGS, t + 1)
      overlay.ingest(msg.hondaRadarObservations.as_reader(), msg.logMonoTime, t + 1)
    self.assertFalse(overlay.points(START + 120_000_000, START + 120_000_001))

  def test_conflicting_old_scan_does_not_hide_a_newer_position(self):
    for part in (0, 4):
      with self.subTest(part=part):
        observer = HondaRadarObserver(*configuration(), session_id='fixture')
        overlay = RadarOverlayState()
        for t, incoming in ((START, scan_frames()),
                            (START + 60_000_000, scan_frames(index=1, life=3, distance=500)),
                            (START + 120_000_000, [scan_frames(distance=1001, velocity=737)[part]])):
          msg = observer.update([can_message(t, incoming)], state(t), SETTINGS, t + 1)
          overlay.ingest(msg.hondaRadarObservations.as_reader(), msg.logMonoTime, t + 1)
        points = overlay.points(START + 120_000_000, START + 120_000_001)
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].text, '25.6')

  def test_uncompleted_scan_conflict_cannot_retract_another_scan(self):
    observer = HondaRadarObserver(*configuration(), session_id='fixture')
    overlay = RadarOverlayState()
    for t, incoming in ((START, [scan_frames()[1]]),
                        (START + 60_000_000, scan_frames(index=1, life=3, distance=500)),
                        (START + 120_000_000, [scan_frames()[0], scan_frames(distance=1001)[0]])):
      msg = observer.update([can_message(t, incoming)], state(t), SETTINGS, t + 1)
      overlay.ingest(msg.hondaRadarObservations.as_reader(), msg.logMonoTime, t + 1)
    self.assertEqual(overlay.points(START + 120_000_000, START + 120_000_001)[0].text, '25.6')

  def test_current_scan_conflict_retracts_only_until_a_new_observation(self):
    observer = HondaRadarObserver(*configuration(), session_id='fixture')
    overlay = RadarOverlayState()
    for t, incoming, expected in ((START, scan_frames(), True),
                                  (START + 60_000_000, [scan_frames(velocity=737)[4]], False),
                                  (START + 120_000_000, scan_frames(index=1, life=3, distance=500), True)):
      msg = observer.update([can_message(t, incoming)], state(t), SETTINGS, t + 1)
      overlay.ingest(msg.hondaRadarObservations.as_reader(), msg.logMonoTime, t + 1)
      self.assertEqual(bool(overlay.points(t, t + 1)), expected)
    self.assertEqual(overlay.points(START + 120_000_000, START + 120_000_001)[0].text, '25.6')

  def test_format_distance_only_one_decimal(self):
    for slot, distance, expected in ((0, 8.0, '8.0'), (3, 24.74, '24.7'), (3, 24.76, '24.8'), (15, 120, '120.0')):
      self.assertEqual(RadarLabelPoint(slot, distance, 0, START, 1, 1).text, expected)

  def test_camera_time_selection_no_future_or_immortal_labels(self):
    data = sample()
    overlay = RadarOverlayState()
    overlay.ingest(data, START + 1, START + 1)
    self.assertEqual(len(overlay.history[0]), 1)  # position and AUX are one sample
    self.assertFalse(overlay.points(START - 1, START + 1))
    self.assertEqual(overlay.points(START, START + 1)[0].text, '54.1')
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
    self.assertEqual(overlay.points(START + 2, START + 3)[0].text, '25.0')
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
    view = SimpleNamespace(frame=object(), client=SimpleNamespace(timestamp_eof=START), device_camera=object(),
                           model_renderer=SimpleNamespace(_car_space_transform=np.array(((640., 1000., 0), (360., 0, 1000.), (1., 0, 0)))))
    params = Mock()
    params.get.return_value = renderer.ASSIST
    params.get_bool.return_value = True
    ui = SimpleNamespace(CP=configuration()[0], started=True, sm=sm, params=params, active_bundle=False)
    with patch.object(renderer, 'ui_state', ui), patch.object(renderer.time, 'monotonic_ns', return_value=START + 2) as clock, \
         patch.object(renderer.gui_app, 'font', return_value=object()), \
         patch.object(renderer, 'project_radar_point', wraps=project_radar_point) as project, \
         patch.object(rl, 'draw_circle_v') as circles, patch.object(rl, 'draw_line_v') as lines, \
         patch.object(rl, 'draw_rectangle_rec') as backgrounds, patch.object(rl, 'draw_text_ex') as draw:
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      self.assertTrue(overlay.enabled)  # the assist mode keeps the requested radar overlay
      self.assertEqual(draw.call_args_list[0].args[1], '54.1')
      self.assertEqual(draw.call_count, 1)
      self.assertEqual(circles.call_count, 1)
      marker, radius, color = circles.call_args.args
      expected = project_radar_point(overlay.state.points(START, START + 2)[0], view.model_renderer._car_space_transform, 1.2)
      self.assertAlmostEqual(marker.x, expected[0], places=3)
      self.assertAlmostEqual(marker.y, expected[1], places=3)
      self.assertGreater(draw.call_args.args[2].x, marker.x + radius)
      self.assertEqual((color.r, color.g, color.b), (255, 215, 0))
      backgrounds.assert_not_called()
      lines.assert_not_called()
      original_x = draw.call_args_list[0].args[2].x
      observer_sm.updated[renderer.SERVICE] = False
      draw.reset_mock()
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      self.assertEqual(project.call_count, 1)  # same observation and camera transform
      self.assertEqual(draw.call_args_list[0].args[2].x, original_x)
      view.model_renderer._car_space_transform[0, 0] += 25
      draw.reset_mock()
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      self.assertEqual(project.call_count, 2)
      self.assertAlmostEqual(draw.call_args_list[0].args[2].x, original_x + 25, places=3)
      # Fading and expiry must remain live even when geometry is cached.
      clock.return_value = START + 120_000_000
      view.client.timestamp_eof = START + 100_000_000
      draw.reset_mock()
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      self.assertEqual(project.call_count, 2)
      self.assertEqual(draw.call_args_list[0].args[-1].a, 140)
      clock.return_value = START + 210_000_000
      view.client.timestamp_eof = START + 200_000_000
      draw.reset_mock()
      circles.reset_mock()
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      draw.assert_not_called()
      circles.assert_not_called()
      draw.reset_mock()
      clock.return_value = START + 2
      view.client.timestamp_eof = START - 1
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      draw.assert_not_called()  # future position not drawn
      circles.assert_not_called()

  def test_sixteen_markers_near_screen_edge_keep_projected_positions(self):
    import numpy as np
    import pyray as rl
    from openpilot.cereal import log, messaging
    from openpilot.selfdrive.ui.sunnypilot.onroad import honda_radar_overlay as renderer
    observer = HondaRadarObserver(*configuration(), session_id='fixture')
    frames = [f for slot in range(16) for f in scan_frames(slot=slot)]
    wire = observer.update([can_message(START, frames)], state(START), SETTINGS, START + 1)
    sm = state(START)
    calib = messaging.new_message('extrinsicsCalibration').extrinsicsCalibration
    calib.calStatus = log.ExtrinsicsCalibration.Status.calibrated
    calib.height = [1.2]
    sm['extrinsicsCalibration'] = calib
    sm.valid['extrinsicsCalibration'] = True
    observer_sm = Mock(updated={renderer.SERVICE: True}, valid={renderer.SERVICE: True}, logMonoTime={renderer.SERVICE: START + 1})
    observer_sm.__getitem__ = Mock(return_value=wire.hondaRadarObservations.as_reader())
    overlay = renderer.HondaRadarOverlay()
    overlay.sm, overlay.enabled, overlay.next_params_ns = observer_sm, True, START + 1_000_000_000
    view = SimpleNamespace(frame=object(), client=SimpleNamespace(timestamp_eof=START), device_camera=object(),
                           model_renderer=SimpleNamespace(_car_space_transform=np.array(((640., 1000., 0), (660., 0, 1000.), (1., 0, 0)))))
    ui = SimpleNamespace(CP=configuration()[0], started=True, sm=sm)
    with patch.object(renderer, 'ui_state', ui), patch.object(renderer.time, 'monotonic_ns', return_value=START + 2), \
         patch.object(renderer.gui_app, 'font', return_value=object()), \
         patch.object(rl, 'draw_circle_v') as circles, \
         patch.object(rl, 'draw_line_v') as lines, patch.object(rl, 'draw_rectangle_rec') as rectangles, \
         patch.object(rl, 'draw_text_ex') as text:
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
    markers = [call.args[0] for call in circles.call_args_list]
    self.assertEqual(len(markers), 16)
    rectangles.assert_not_called()
    lines.assert_not_called()
    self.assertEqual([call.args[1] for call in text.call_args_list], ['54.1'] * 16)
    # Identical radar positions must remain identical on screen, even when
    # crowded. Label avoidance must never fabricate separated detections.
    self.assertEqual(len({(marker.x, marker.y) for marker in markers}), 1)
    for marker, call in zip(markers, text.call_args_list, strict=True):
      self.assertTrue(0 <= marker.x <= 1280 and 0 <= marker.y <= 720)
      self.assertGreater(call.args[2].x, marker.x)

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
      settings._set_assist(True)
      params.put.assert_called_once()
      settings._set_overlay(True)
      params.put_bool.assert_not_called()


  def test_shadow_and_assist_switches_preserve_modes_and_are_offroad_only(self):
    from openpilot.selfdrive.ui.sunnypilot.layouts.settings.vehicle.brands import honda
    from openpilot.common.params import Params
    params = Params()
    fake_ui = SimpleNamespace(started=False, params=params)
    with patch.object(honda, 'ui_state', fake_ui):
      # Exercise the real settings callbacks/state refresh without allocating
      # GPU fonts in this headless fixture.
      settings = honda.HondaSettings.__new__(honda.HondaSettings)
      settings.observer, settings.shadow, settings.assist, settings.overlay = Mock(), Mock(), Mock(), Mock()
      settings.control = Mock()
      settings._set_assist(True)
      self.assertEqual(params.get('HondaBoschARadarMode', return_default=True), 0)
      settings._set_shadow(True)
      self.assertNotEqual(params.get('HondaBoschARadarMode', return_default=True), 2)
      settings._set_observer(True)
      settings._set_shadow(True)
      self.assertEqual(params.get('HondaBoschARadarMode'), 2)
      settings._set_observer(True)
      self.assertEqual(params.get('HondaBoschARadarMode'), 2)
      fake_ui.started = True
      settings._set_shadow(False)
      settings._set_assist(True)
      self.assertEqual(params.get('HondaBoschARadarMode'), 2)
      fake_ui.started = False
      settings._set_assist(True)
      self.assertEqual(params.get('HondaBoschARadarMode'), 3)
      settings.observer.action_item.set_state.assert_called_with(True)
      settings.shadow.action_item.set_state.assert_called_with(True)
      settings.assist.action_item.set_state.assert_called_with(True)
      settings.update_settings()
      settings._set_observer(True)
      settings._set_shadow(True)
      settings._set_shadow(False)
      self.assertEqual(params.get('HondaBoschARadarMode'), 3)
      fake_ui.started = True
      settings._set_assist(False)
      settings._set_observer(False)
      self.assertEqual(params.get('HondaBoschARadarMode'), 3)
      fake_ui.started = False
      settings._set_assist(False)
      self.assertEqual(params.get('HondaBoschARadarMode'), 2)
      settings.assist.action_item.set_state.assert_called_with(False)
      settings._set_shadow(False)
      self.assertEqual(params.get('HondaBoschARadarMode'), 1)
      settings._set_assist(False)
      self.assertEqual(params.get('HondaBoschARadarMode'), 1)
      settings._set_assist(True)
      self.assertEqual(params.get('HondaBoschARadarMode'), 3)
      settings._set_control(True)
      self.assertEqual(params.get('HondaBoschARadarMode'), 4)
      settings.control.action_item.set_state.assert_called_with(True)
      settings._set_observer(True)
      settings._set_assist(False)
      settings._set_shadow(False)
      self.assertEqual(params.get('HondaBoschARadarMode'), 4)
      fake_ui.started = True
      settings._set_control(False)
      self.assertEqual(params.get('HondaBoschARadarMode'), 4)
      fake_ui.started = False
      settings._set_control(False)
      self.assertEqual(params.get('HondaBoschARadarMode'), 2)
      settings._set_assist(True)
      settings._set_observer(False)
      self.assertEqual(params.get('HondaBoschARadarMode'), 0)
      settings.observer.action_item.set_state.assert_called_with(False)
      settings.shadow.action_item.set_state.assert_called_with(False)
      settings.assist.action_item.set_state.assert_called_with(False)


if __name__ == '__main__':
  unittest.main()
