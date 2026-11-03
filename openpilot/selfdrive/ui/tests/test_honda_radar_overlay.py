from dataclasses import replace
import math
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from openpilot.selfdrive.ui.tests.test_sunnylink_removed import HeadlessUITestCase
from openpilot.selfdrive.car.tests.test_honda_radar_observer import SETTINGS, START, can_message, configuration, scan_frames, state
from openpilot.selfdrive.car.honda_radar_observer import HondaRadarObserver
from openpilot.selfdrive.ui.sunnypilot.onroad.honda_radar_state import (
  RadarLabelPoint, RadarOverlayState, project_radar_coordinates, project_radar_point,
)


def sample():
  observer = HondaRadarObserver(*configuration(), session_id='fixture')
  msg = observer.update([can_message(START, scan_frames())], state(START), SETTINGS, START + 1)
  return msg.hondaRadarObservations.to_dict()


class TestHondaRadarOverlay(HeadlessUITestCase):
  def test_wire_reader_and_builder_preserve_exact_track_identity(self):
    from openpilot.cereal import log

    data = sample()
    event = next(event for event in data['observations'] if event['positionValid'])
    track = {field: event[field] for field in ('slot', 'generation', 'positionSeq', 'positionMonoTime')}
    track['trackId'] = 2 ** 64 - 1
    data['fusion'] = {'version': 1, 'active': True, 'tracks': [track]}
    expected = RadarOverlayState()
    expected.ingest(data, START + 1, START + 1)
    self.assertEqual(expected.points(START, START + 1)[0].track_id, track['trackId'])
    wire = log.Event.new_message(hondaRadarObservations=data)
    for kind, body in (('builder', wire.hondaRadarObservations), ('reader', wire.hondaRadarObservations.as_reader())):
      with self.subTest(kind=kind):
        overlay = RadarOverlayState()
        overlay.ingest(body, START + 1, START + 1)
        self.assertEqual(overlay.points(START, START + 1), expected.points(START, START + 1))
        self.assertFalse(overlay.points(START, START + 200_000_001))

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

  def test_asymmetric_boundaries_use_the_same_projection_and_camera_offset_as_the_dot(self):
    import pyray as rl
    from openpilot.selfdrive.ui.sunnypilot.onroad import honda_radar_overlay as renderer
    point = RadarLabelPoint(0, 20, 0.2, START, 1, 1, boundary_laterals=(2.0, -0.5), boundary_time_ns=START)
    overlay = renderer.HondaRadarOverlay()
    overlay.font, overlay.camera_offset = object(), 0.35
    for focal in (500.0, 1000.0):
      with self.subTest(focal=focal), patch.object(renderer, 'measure_text_cached', return_value=rl.Vector2(64, 40)):
        # The lateral term in row 2 also exercises a nonhorizontal image span.
        matrix = ((640, focal, 0), (360, 100, focal), (1, 0, 0))
        label, = overlay._layout_labels([point], matrix, 1.2, rl.Rectangle(0, 0, 1280, 720))
      self.assertAlmostEqual(label.marker.x, 640 + focal * (0.35 - 0.2) / 20, places=3)
      self.assertAlmostEqual(label.marker.y, 360 + (100 * (0.35 - 0.2) + focal * 1.2) / 20, places=3)
      span, left_tick, right_tick = label.lines
      for endpoint, lateral, tick in zip(span, point.boundary_laterals, (left_tick, right_tick), strict=True):
        self.assertAlmostEqual(endpoint.x, 640 + focal * (0.35 - lateral) / 20, places=3)
        self.assertAlmostEqual(endpoint.y, 360 + (100 * (0.35 - lateral) + focal * 1.2) / 20, places=3)
        self.assertAlmostEqual(tick[0].x, endpoint.x)
        self.assertAlmostEqual(tick[1].x, endpoint.x)
        self.assertAlmostEqual((tick[0].y + tick[1].y) / 2, endpoint.y)
        self.assertEqual(tick[1].y - tick[0].y, renderer.BOUNDARY_TICK_HEIGHT)
      self.assertLess(span[0].x, label.marker.x)
      self.assertLess(label.marker.x, span[1].x)
      self.assertAlmostEqual(label.text_position.x + 32, (span[0].x + span[1].x) / 2, places=3)
      self.assertLess(label.text_position.y + 40, min(left_tick[0].y, right_tick[0].y))

  def test_missing_incomplete_or_unprojectable_boundary_keeps_the_dot_and_upper_label(self):
    import pyray as rl
    from openpilot.selfdrive.ui.sunnypilot.onroad import honda_radar_overlay as renderer
    overlay = renderer.HondaRadarOverlay()
    overlay.font = object()
    matrix = ((640, 1000, 0), (360, 0, 1000), (1, 0, 0))
    behind_camera = ((640, 1000, 0), (360, 0, 1000), (1, 4, 0))
    for boundaries, transform in ((None, matrix), ((2.0,), matrix), ((float('nan'), -0.5), matrix),
                                  ((6.0, -0.5), behind_camera)):
      with self.subTest(boundaries=boundaries), patch.object(renderer, 'measure_text_cached', return_value=rl.Vector2(64, 40)):
        point = RadarLabelPoint(0, 20, 0, START, 1, 1, boundary_laterals=boundaries)
        label, = overlay._layout_labels([point], transform, 1.2, rl.Rectangle(0, 0, 1280, 720))
      self.assertFalse(label.lines)
      self.assertEqual((label.marker.x, label.marker.y), (640, 420))
      self.assertEqual(label.text_position.x + 32, label.marker.x)
      self.assertLess(label.text_position.y + 40, label.marker.y - renderer.MARKER_RADIUS)
      self.assertEqual(label.point.text, '20.0')

  def test_offscreen_boundary_does_not_create_a_false_endpoint_at_the_view_border(self):
    import pyray as rl
    from openpilot.selfdrive.ui.sunnypilot.onroad import honda_radar_overlay as renderer
    overlay = renderer.HondaRadarOverlay()
    overlay.font = object()
    point = RadarLabelPoint(0, 20, 0, START, 1, 1, boundary_laterals=(20.0, -1.0), boundary_time_ns=START)
    matrix = ((640, 1000, 0), (360, 0, 1000), (1, 0, 0))
    with patch.object(renderer, 'measure_text_cached', return_value=rl.Vector2(64, 40)):
      label, = overlay._layout_labels([point], matrix, 1.2, rl.Rectangle(0, 0, 1280, 720))
    span, right_tick = label.lines
    self.assertEqual((label.marker.x, label.marker.y), (640, 420))
    self.assertEqual((span[0].x, span[1].x), (-360, 690))
    # The enclosing camera scissor clips the span; only its real visible end
    # gets a short vertical marker, never x=0 where the line crosses the border.
    self.assertEqual((right_tick[0].x, right_tick[1].x), (690, 690))

  def test_raw_boundary_wire_roundtrip_reaches_the_renderer(self):
    import numpy as np
    import pyray as rl
    from openpilot.cereal import log, messaging
    from openpilot.selfdrive.ui.sunnypilot.onroad import honda_radar_overlay as renderer
    data = sample()
    for event in data['observations']:
      if event['positionValid']:
        azimuth = event['raw']['azimuthRaw']
        event['raw'].update(edgeARaw=azimuth + 60, edgeBRaw=azimuth - 20)
    wire = messaging.new_message(renderer.SERVICE)
    wire.valid, wire.logMonoTime = True, START + 1
    setattr(wire, renderer.SERVICE, data)
    decoded = messaging.log_from_bytes(wire.to_bytes())
    sm = state(START)
    calib = messaging.new_message('extrinsicsCalibration').extrinsicsCalibration
    calib.calStatus, calib.height = log.ExtrinsicsCalibration.Status.calibrated, [1.2]
    sm['extrinsicsCalibration'], sm.valid['extrinsicsCalibration'] = calib, True
    observer = Mock(updated={renderer.SERVICE: True}, valid={renderer.SERVICE: decoded.valid},
                    logMonoTime={renderer.SERVICE: decoded.logMonoTime})
    observer.__getitem__ = Mock(return_value=decoded.hondaRadarObservations)
    overlay = renderer.HondaRadarOverlay()
    overlay.sm, overlay.enabled, overlay.next_params_ns = observer, True, START + 1_000_000_000
    view = SimpleNamespace(frame=object(), client=SimpleNamespace(timestamp_eof=START), device_camera=object(),
                           model_renderer=SimpleNamespace(_car_space_transform=np.array(((640., 1000., 0), (360., 0, 1000.), (1., 0, 0)))))
    ui = SimpleNamespace(CP=configuration()[0], started=True, sm=sm, radar_signal=True)
    with patch.object(renderer, 'ui_state', ui), patch.object(renderer.time, 'monotonic_ns', return_value=START + 2), \
         patch.object(renderer.gui_app, 'font', return_value=object()), \
         patch.object(renderer, 'measure_text_cached', return_value=rl.Vector2(64, 40)), \
         patch.object(rl, 'draw_circle_v') as circles, patch.object(rl, 'draw_line_ex') as lines, \
         patch.object(rl, 'draw_rectangle_rec') as backgrounds, patch.object(rl, 'draw_text_ex') as text:
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
    point, = overlay.state.points(START, START + 2)
    raw = decoded.hondaRadarObservations.observations[0].raw
    self.assertEqual((raw.edgeARaw, raw.edgeBRaw), (raw.azimuthRaw + 60, raw.azimuthRaw - 20))
    self.assertIsNotNone(point.boundary_laterals)
    self.assertEqual(lines.call_count, 3)
    self.assertEqual(circles.call_count, 1)
    span = lines.call_args_list[0].args
    marker = circles.call_args.args[0]
    for projected, angle_raw in ((span[0], raw.edgeARaw), (marker, raw.azimuthRaw), (span[1], raw.edgeBRaw)):
      self.assertAlmostEqual(projected.x, 640 - 1000 * math.tan((angle_raw - 1024) / 2048.0), places=3)
      self.assertAlmostEqual(projected.y, 360 + 1000 * 1.2 / point.distance, places=3)
    self.assertLess(span[0].x, marker.x)
    self.assertLess(marker.x, span[1].x)
    self.assertEqual(text.call_count, 1)
    self.assertEqual(text.call_args.args[1], '54.1')
    label_position = text.call_args.args[2]
    self.assertAlmostEqual(label_position.x + 32, (span[0].x + span[1].x) / 2, places=3)
    self.assertLess(label_position.y + 40, marker.y - renderer.BOUNDARY_TICK_HEIGHT / 2)
    for call in (*lines.call_args_list, circles.call_args, text.call_args):
      self.assertEqual((call.args[-1].r, call.args[-1].g, call.args[-1].b, call.args[-1].a), (255, 165, 0, 255))
    backgrounds.assert_not_called()

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
    params.get.return_value = 3
    params.get_bool.return_value = True
    ui = SimpleNamespace(CP=configuration()[0], started=True, sm=sm, params=params, active_bundle=False, radar_signal=True)
    with patch.object(renderer, 'ui_state', ui), patch.object(renderer.time, 'monotonic_ns', return_value=START + 2) as clock, \
         patch.object(renderer.gui_app, 'font', return_value=object()), \
         patch.object(renderer, 'measure_text_cached', return_value=rl.Vector2(64, 40)) as measure, \
         patch.object(renderer, 'radar_influence_strength', return_value=0.0) as strength, \
         patch.object(renderer, 'project_radar_point', wraps=project_radar_point) as project, \
         patch.object(renderer, 'project_radar_coordinates', wraps=project_radar_coordinates) as project_boundary, \
         patch.object(rl, 'draw_circle_v') as circles, patch.object(rl, 'draw_line_ex') as lines, \
         patch.object(rl, 'draw_rectangle_rec') as backgrounds, patch.object(rl, 'draw_text_ex') as draw:
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      self.assertTrue(overlay.enabled)
      params.get.assert_not_called()
      self.assertEqual(draw.call_args_list[0].args[1], '54.1')
      self.assertEqual(draw.call_count, 1)
      self.assertEqual(circles.call_count, 1)
      marker, radius, color = circles.call_args.args
      expected = project_radar_point(overlay.state.points(START, START + 2)[0], view.model_renderer._car_space_transform, 1.2)
      self.assertAlmostEqual(marker.x, expected[0], places=3)
      self.assertAlmostEqual(marker.y, expected[1], places=3)
      self.assertAlmostEqual(draw.call_args.args[2].x + 32, marker.x, places=3)
      self.assertLess(draw.call_args.args[2].y + 40, marker.y - radius)
      self.assertEqual((color.r, color.g, color.b), (255, 165, 0))
      backgrounds.assert_not_called()
      lines.assert_not_called()
      original_x = draw.call_args_list[0].args[2].x
      observer_sm.updated[renderer.SERVICE] = False
      draw.reset_mock()
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      self.assertEqual(project.call_count, 1)  # same observation and camera transform
      self.assertEqual(measure.call_count, 1)
      self.assertEqual(draw.call_args_list[0].args[2].x, original_x)
      self.assertEqual(overlay.labels[0].text, '54.1')
      # Polling unchanged settings preserves both geometry and displayed text.
      previous_labels = overlay.labels
      overlay.next_params_ns = 0
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      self.assertIs(overlay.labels, previous_labels)
      self.assertEqual(project.call_count, 1)
      self.assertEqual(measure.call_count, 1)
      self.assertEqual(draw.call_args.args[1], '54.1')
      view.model_renderer._car_space_transform[0, 0] += 25
      draw.reset_mock()
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      self.assertEqual(project.call_count, 2)
      self.assertAlmostEqual(draw.call_args_list[0].args[2].x, original_x + 25, places=3)
      # Adding boundaries to the same accepted position changes cached geometry
      # without moving its reflection point or changing its physical identity.
      point = overlay.state.history[0][-1]
      overlay.state.history[0][-1] = replace(point, boundary_laterals=(point.lateral + 1.0, point.lateral - 0.5),
                                          boundary_time_ns=START)
      original_marker = circles.call_args.args[0]
      draw.reset_mock()
      lines.reset_mock()
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      self.assertEqual(project.call_count, 3)
      self.assertEqual(project_boundary.call_count, 2)
      self.assertEqual(measure.call_count, 3)
      self.assertEqual(lines.call_count, 3)  # span plus the two real end ticks
      self.assertEqual((circles.call_args.args[0].x, circles.call_args.args[0].y), (original_marker.x, original_marker.y))
      span = lines.call_args_list[0].args
      self.assertLess(span[0].x, original_marker.x)
      self.assertLess(original_marker.x, span[1].x)
      self.assertAlmostEqual(draw.call_args.args[2].x + 32, (span[0].x + span[1].x) / 2, places=3)
      for call in (*lines.call_args_list, circles.call_args, draw.call_args):
        self.assertEqual((call.args[-1].r, call.args[-1].g, call.args[-1].b, call.args[-1].a), (255, 165, 0, 255))
      backgrounds.assert_not_called()
      # Fading and expiry must remain live even when geometry is cached.
      clock.return_value = START + 120_000_000
      view.client.timestamp_eof = START + 100_000_000
      strength.return_value = 0.6
      draw.reset_mock()
      lines.reset_mock()
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      self.assertEqual(project.call_count, 3)
      self.assertEqual(project_boundary.call_count, 2)
      self.assertEqual(measure.call_count, 3)
      for call in (*lines.call_args_list, circles.call_args, draw.call_args):
        self.assertEqual((call.args[-1].r, call.args[-1].g, call.args[-1].b), renderer.influence_rgb(0.6))
        self.assertEqual(call.args[-1].a, 140)
      # A real offset change takes effect in this frame, including the bounds.
      previous_labels = overlay.labels
      previous_x = circles.call_args.args[0].x
      ui.active_bundle = True
      params.get.return_value = 0.35
      overlay.next_params_ns = 0
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      self.assertIsNot(overlay.labels, previous_labels)
      self.assertEqual(project.call_count, 4)
      self.assertEqual(project_boundary.call_count, 4)
      self.assertAlmostEqual(circles.call_args.args[0].x, previous_x + 1000 * 0.35 / point.distance, places=3)
      self.assertEqual(draw.call_args.args[1], '54.1')
      self.assertEqual(draw.call_args.args[-1].a, 140)
      clock.return_value = START + 210_000_000
      view.client.timestamp_eof = START + 200_000_000
      draw.reset_mock()
      circles.reset_mock()
      lines.reset_mock()
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      draw.assert_not_called()
      circles.assert_not_called()
      lines.assert_not_called()
      draw.reset_mock()
      clock.return_value = START + 2
      view.client.timestamp_eof = START - 1
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      draw.assert_not_called()  # future position not drawn
      circles.assert_not_called()
      lines.assert_not_called()

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
    ui = SimpleNamespace(CP=configuration()[0], started=True, sm=sm, radar_signal=True)
    with patch.object(renderer, 'ui_state', ui), patch.object(renderer.time, 'monotonic_ns', return_value=START + 2), \
         patch.object(renderer.gui_app, 'font', return_value=object()), \
         patch.object(renderer, 'measure_text_cached', return_value=rl.Vector2(64, 40)), \
         patch.object(rl, 'draw_circle_v') as circles, \
         patch.object(rl, 'draw_line_ex') as lines, patch.object(rl, 'draw_rectangle_rec') as rectangles, \
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
      self.assertAlmostEqual(call.args[2].x + 32, marker.x, places=3)
      self.assertLess(call.args[2].y + 40, marker.y - renderer.MARKER_RADIUS)

  def test_visuals_overlay_follows_steering_arc_and_changes_are_offroad_only(self):
    from openpilot.selfdrive.ui.sunnypilot.layouts.settings import visuals
    params = Mock()
    params.get_bool.return_value = False
    fake_ui = SimpleNamespace(params=params, started=False)
    with patch.object(visuals, 'ui_state', fake_ui), patch.object(visuals, 'toggle_item_sp', side_effect=lambda *a, **kw: Mock()), \
         patch.object(visuals, 'multiple_button_item_sp'):
      settings = visuals.VisualsLayout.__new__(visuals.VisualsLayout)
      settings._params = params
      items = settings._initialize_items()
      self.assertIs(items[items.index(settings._toggles['TorqueBar']) + 1], settings._toggles['HondaBoschARadarOverlay'])
      settings._set_radar_overlay(True)
      params.put_bool.assert_called_once_with('HondaBoschARadarOverlay', True, block=True)
      fake_ui.started = True
      settings._set_radar_overlay(False)
      params.put_bool.assert_called_once()
      params.get.assert_not_called()
      params.put.assert_not_called()


if __name__ == '__main__':
  unittest.main()
