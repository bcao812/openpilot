from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pyray as rl

from openpilot.cereal import log
from openpilot.selfdrive.car.tests.test_honda_radar_observer import START, configuration
from openpilot.selfdrive.ui.sunnypilot.onroad import honda_radar_overlay as renderer
from openpilot.selfdrive.ui.sunnypilot.onroad.honda_radar_state import RadarLabelPoint, project_radar_point
from openpilot.selfdrive.ui.tests.test_control_influence_render import executing_state
from openpilot.selfdrive.ui.tests.test_sunnylink_removed import HeadlessUITestCase
from openpilot.system.ui.lib import text_measure


class TestRadarLabelLayoutCache(HeadlessUITestCase):
  def setUp(self):
    super().setUp()
    self.sm = executing_state()
    self.sm['extrinsicsCalibration'] = SimpleNamespace(calStatus=log.ExtrinsicsCalibration.Status.calibrated, height=[1.2])
    self.sm.valid['extrinsicsCalibration'] = True
    self.ui = SimpleNamespace(CP=configuration()[0], started=True, sm=self.sm)
    self.view = SimpleNamespace(frame=object(), client=SimpleNamespace(timestamp_eof=START + 2), device_camera=object(),
                                model_renderer=SimpleNamespace(_car_space_transform=np.array(
                                  ((640., 1000., 0.), (360., 0., 1000.), (1., 0., 0.)))))
    self.rect = rl.Rectangle(0, 0, 1280, 720)
    self.font = object()
    self.overlay, self.uncached = renderer.HondaRadarOverlay(), renderer.HondaRadarOverlay()
    for overlay in (self.overlay, self.uncached):
      overlay.enabled, overlay.next_params_ns, overlay.font = True, START + 1_000_000_000, self.font
      overlay.sm = Mock(updated={renderer.SERVICE: False}, valid={renderer.SERVICE: True},
                        logMonoTime={renderer.SERVICE: START})
      overlay.sm.__getitem__ = Mock()
    self.enterContext(patch.object(renderer, 'ui_state', self.ui))
    self.enterContext(patch.object(renderer.multilang, '_language', 'en'))
    self.clock = self.enterContext(patch.object(renderer.time, 'monotonic_ns', return_value=START + 3))
    self.project = self.enterContext(patch.object(renderer, 'project_radar_point', wraps=project_radar_point))
    self.measure = self.enterContext(patch.object(renderer, 'measure_text_cached',
                                                 side_effect=lambda font, text, size: rl.Vector2(len(text) * 16, size)))
    self.error = self.enterContext(patch.object(renderer.cloudlog, 'exception'))
    self.draws = []
    self.enterContext(patch.object(rl, 'draw_line_ex', side_effect=lambda a, b, width, color:
                                  self.draws.append(('line', (a.x, a.y), (b.x, b.y), width, self.rgba(color)))))
    self.enterContext(patch.object(rl, 'draw_circle_v', side_effect=lambda pos, radius, color:
                                  self.draws.append(('circle', (pos.x, pos.y), radius, self.rgba(color)))))
    self.enterContext(patch.object(rl, 'draw_text_ex', side_effect=lambda font, text, pos, size, spacing, color:
                                  self.draws.append(('text', id(font), text, (pos.x, pos.y), size, spacing, self.rgba(color)))))

  @staticmethod
  def rgba(color):
    return color.r, color.g, color.b, color.a

  @staticmethod
  def point(slot=0, **changes):
    return replace(RadarLabelPoint(slot, 20.0 + slot, slot / 10, START, 1, 1, 'fixture', 42 + slot,
                                   (slot / 10 + 1, slot / 10 - 1), START), **changes)

  def select(self, points):
    for overlay in (self.overlay, self.uncached):
      overlay.state.clear()
      overlay.state.active = True
      overlay.state.message_ns = self.clock.return_value
      for point in points:
        overlay.state.history[point.slot].append(point)

  def render_equal(self, projections=None):
    self.draws.clear()
    before = self.project.call_count
    self.overlay.render(self.view, self.rect)
    actual = self.draws[:]
    if projections is not None:
      self.assertEqual(self.project.call_count - before, projections)
    # _layout_labels is the unchanged geometry implementation. Running the
    # complete renderer with empty caches each frame provides its uncached
    # draw order, float coordinates, text and current colors as the oracle.
    self.uncached._clear_layout()
    self.draws.clear()
    self.uncached.render(self.view, self.rect)
    self.assertEqual(actual, self.draws)
    self.assertLessEqual(len(self.overlay.point_layouts), 16)
    return actual

  def test_staggered_full_updates_and_deletion_preserve_all_draw_arguments(self):
    points = [self.point(slot) for slot in range(16)]
    self.select(points)
    self.render_equal(16)
    self.render_equal(0)
    points[7] = replace(points[7], distance=27.37, sequence=2, time_ns=START + 1)
    self.select(points)
    self.render_equal(1)
    points = [replace(point, distance=point.distance + 0.2, sequence=point.sequence + 1, time_ns=START + 2) for point in points]
    self.select(points)
    self.render_equal(16)
    removed = points.pop(7)
    self.select(points)
    self.render_equal(0)
    self.assertEqual(set(self.overlay.point_layouts), {point.slot for point in points})
    points.insert(7, removed)
    self.select(points)
    self.render_equal(1)  # deleted entries are not retained as history
    self.select([])
    self.assertFalse(self.render_equal(0))
    self.assertFalse(self.overlay.point_layouts)
    self.select(points)
    self.render_equal(16)

  def test_projection_context_changes_invalidate_every_point(self):
    self.select([self.point(0), self.point(1)])
    self.render_equal(2)
    for field in ('transform', 'height', 'offset', 'x', 'y', 'width', 'height_rect', 'font'):
      with self.subTest(field=field):
        if field == 'transform':
          self.view.model_renderer._car_space_transform[0, 0] += 5
        elif field == 'height':
          self.sm['extrinsicsCalibration'].height = [1.3]
        elif field == 'offset':
          self.overlay.camera_offset = self.uncached.camera_offset = 0.15
        elif field == 'font':
          self.overlay.font = self.uncached.font = object()
        else:
          attr = 'height' if field == 'height_rect' else field
          setattr(self.rect, attr, getattr(self.rect, attr) + 1)
        self.render_equal(2)
        self.render_equal(0)

  def test_exact_point_identity_and_boundary_content_invalidate_one_slot(self):
    point = self.point()
    self.select([point, self.point(1)])
    self.render_equal(2)
    changes = (
      {'distance': 20.2}, {'lateral': 0.25}, {'time_ns': START + 1}, {'sequence': 2}, {'generation': 2},
      {'session_id': 'other-session'}, {'track_id': 43}, {'boundary_laterals': (1.5, -0.5)}, {'boundary_time_ns': START + 1},
    )
    for change in changes:
      with self.subTest(change=change):
        point = replace(point, **change)
        self.select([point, self.point(1)])
        self.render_equal(1)
        self.assertEqual(self.overlay.labels[0].point, point)
    point = replace(point, slot=2)
    self.select([point, self.point(1)])
    self.render_equal(1)
    self.assertEqual(set(self.overlay.point_layouts), {1, 2})

  def test_fallback_font_and_language_refresh_the_unchanged_slots(self):
    normal = SimpleNamespace(texture=SimpleNamespace(id=1))
    fallback = SimpleNamespace(texture=SimpleNamespace(id=2))
    self.overlay.font = self.uncached.font = normal
    font_lookup = self.enterContext(patch.object(renderer.gui_app, 'fallback_font', return_value=fallback))
    self.enterContext(patch.object(text_measure, '_cache', {}))
    self.enterContext(patch.object(rl, 'measure_text_ex', side_effect=lambda font, text, size, spacing:
                                  rl.Vector2(len(text) * font.texture.id * size / 4, size)))
    self.measure.side_effect = text_measure.measure_text_cached
    points = [self.point(0), self.point(1)]
    self.select(points)
    self.render_equal(2)
    previous_position = self.overlay.labels[1].text_position.x
    for change in ('language', 'fallback_font', 'texture', 'second_language'):
      with self.subTest(change=change):
        if change == 'language':
          renderer.multilang._language = 'zh-CHS'
        elif change == 'fallback_font':
          fallback = SimpleNamespace(texture=SimpleNamespace(id=3))
          font_lookup.return_value = fallback
        elif change == 'texture':
          fallback.texture.id = 4
        else:
          renderer.multilang._language = 'ja'
        points[0] = replace(points[0], distance=points[0].distance + 0.1)
        self.select(points)
        self.render_equal(2)
        current_position = self.overlay.labels[1].text_position.x
        if change != 'second_language':
          self.assertNotEqual(current_position, previous_position)
        previous_position = current_position
        self.render_equal(0)

  def test_offscreen_results_are_cached_only_for_the_current_point(self):
    visible, offscreen = self.point(), self.point(1, lateral=100)
    self.select([visible, offscreen])
    self.render_equal(2)
    self.assertEqual(len(self.overlay.labels), 1)
    visible = replace(visible, distance=20.5)
    self.select([visible, offscreen])
    self.render_equal(1)
    visible_again = replace(offscreen, lateral=0.1)
    self.select([visible, visible_again])
    self.render_equal(1)
    self.assertEqual(len(self.overlay.labels), 2)
    self.select([visible])
    self.render_equal(0)
    self.select([visible, offscreen])
    self.render_equal(1)
    self.assertEqual(len(self.overlay.labels), 1)

  def test_camera_selection_boundary_time_fading_and_expiry_remain_live(self):
    self.clock.return_value = START + 50_000_000
    self.view.client.timestamp_eof = START + 20_000_000
    self.select([self.point(boundary_time_ns=START + 40_000_000)])
    calls = self.render_equal(1)
    self.assertEqual([call[0] for call in calls], ['circle', 'text'])
    self.view.client.timestamp_eof = START + 45_000_000
    calls = self.render_equal(1)
    self.assertEqual([call[0] for call in calls], ['line', 'line', 'line', 'circle', 'text'])
    self.clock.return_value = START + 120_000_000
    self.view.client.timestamp_eof = START + 100_000_000
    self.assertTrue(all(call[-1][-1] == 140 for call in self.render_equal(0)))
    self.clock.return_value = START + 210_000_000
    self.view.client.timestamp_eof = START + 200_000_000
    self.assertFalse(self.render_equal(0))
    self.assertFalse(self.overlay.point_layouts)
    self.select([self.point(time_ns=START + 205_000_000)])
    self.assertFalse(self.render_equal(0))  # not in the displayed camera image
    self.view.client.timestamp_eof = START + 206_000_000
    self.render_equal(1)
    self.view.client.timestamp_eof = START + 220_000_000
    self.assertFalse(self.render_equal(0))  # future camera frame
    self.assertFalse(self.overlay.point_layouts)

  def test_current_identity_override_and_receipt_expiry_do_not_cache_red(self):
    self.clock.return_value = START + 150_000_000
    self.view.client.timestamp_eof = START + 120_000_000
    point = self.point(time_ns=START + 100_000_000)
    self.select([point])
    self.assertEqual(self.render_equal(1)[-1][-1][:3], renderer.influence_rgb(0.25))
    self.sm['carState'].gasPressed = True
    self.assertEqual(self.render_equal(0)[-1][-1][:3], (0, 255, 0))
    self.sm['carState'].gasPressed = False
    self.assertEqual(self.render_equal(0)[-1][-1][:3], renderer.influence_rgb(0.25))
    self.select([replace(point, track_id=99)])
    self.assertEqual(self.render_equal(1)[-1][-1][:3], (0, 255, 0))
    self.select([point])
    self.assertEqual(self.render_equal(1)[-1][-1][:3], renderer.influence_rgb(0.25))
    self.clock.return_value = START + 200_000_001
    self.view.client.timestamp_eof = START + 200_000_000
    self.assertEqual(self.render_equal(0)[-1][-1], (0, 255, 0, 140))

  def test_unavailable_contexts_and_errors_clear_derived_layout(self):
    for case in ('scope', 'health', 'calibration', 'height', 'frame', 'camera', 'transform', 'inactive'):
      with self.subTest(case=case):
        self.select([self.point()])
        self.render_equal(1)
        if case == 'scope':
          self.ui.started = False
        elif case == 'health':
          self.sm.valid['selfdriveState'] = False
        elif case == 'calibration':
          self.sm.valid['extrinsicsCalibration'] = False
        elif case == 'height':
          self.sm['extrinsicsCalibration'].height = []
        elif case == 'frame':
          self.view.frame = None
        elif case == 'camera':
          self.view.device_camera = None
        elif case == 'transform':
          original = self.view.model_renderer._car_space_transform
          self.view.model_renderer._car_space_transform = None
        else:
          self.overlay.state.clear()
          self.uncached.state.clear()
        self.assertFalse(self.render_equal(0))
        self.assertFalse(self.overlay.labels)
        self.assertFalse(self.overlay.point_layouts)
        self.assertIsNone(self.overlay.layout_key)
        self.ui.started = self.sm.valid['selfdriveState'] = self.sm.valid['extrinsicsCalibration'] = True
        self.sm['extrinsicsCalibration'].height = [1.2]
        self.view.frame, self.view.device_camera = object(), object()
        if case == 'transform':
          self.view.model_renderer._car_space_transform = original
        self.select([self.point()])
        self.render_equal(1)
        self.select([])
        self.render_equal(0)

  def test_source_reset_invalid_message_and_explicit_withdrawal_do_not_restore_layout(self):
    scope = dict.fromkeys(('alphaLong', 'experimentalSetting', 'decSetting', 'experimentalActive', 'decEnabled', 'decActive'), True)
    bodies = (
      {'schemaVersion': 1, 'sessionId': 'fixture', 'sequence': 2, 'active': True, 'scope': scope,
       'observations': [{'slot': 0, 'positionValid': False, 'reason': 'invalidGeometry', 'positionMonoTime': START}]},
      {'schemaVersion': 1, 'sessionId': 'new-source', 'sequence': 1, 'active': True, 'scope': scope, 'observations': []},
      {'schemaVersion': 99},
      None,
    )
    for body in bodies:
      with self.subTest(body=body):
        self.select([self.point()])
        self.render_equal(1)
        for overlay in (self.overlay, self.uncached):
          overlay.state.session, overlay.state.sequence = 'fixture', 1
          overlay.sm.updated[renderer.SERVICE] = True
          overlay.sm.valid[renderer.SERVICE] = body is not None
          overlay.sm.__getitem__.return_value = body
        self.assertFalse(self.render_equal(0))
        self.assertFalse(self.overlay.point_layouts)
        for overlay in (self.overlay, self.uncached):
          overlay.sm.updated[renderer.SERVICE] = False
        self.assertFalse(self.render_equal(0))
