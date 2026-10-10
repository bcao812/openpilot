import gc
import weakref
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pyray as rl

from openpilot.selfdrive.ui.tests.test_sunnylink_removed import HeadlessUITestCase
from openpilot.system.ui.lib import shader_polygon as polygon


def reference_strip(points):
  points = np.ascontiguousarray(points, dtype=np.float32)
  if len(points) < 3:
    return np.empty((0, 2), dtype=np.float32)
  points = points[:len(points) - len(points) % 2]
  return np.array([point for i in range(len(points) // 2) for point in (points[i], points[-i - 1])], dtype=np.float32)


def prepared_points(prepared):
  return np.array([(prepared.vertices[i].x, prepared.vertices[i].y) for i in range(prepared.vertex_count)], dtype=np.float32).reshape(-1, 2)


@contextmanager
def capture_polygons():
  """Capture exact draw inputs while running the real per-frame shader configuration."""
  frames, uniforms = [], {}

  def set_uniform(shader, location, value, kind):
    if kind == polygon.UNIFORM_VEC2:
      uniforms[location] = (value.x, value.y)
    else:
      count = 4 if kind == polygon.UNIFORM_VEC4 else 1
      uniforms[location] = tuple(value[i] for i in range(count))

  def set_uniform_array(shader, location, value, kind, count):
    size = count * (4 if kind == polygon.UNIFORM_VEC4 else 1)
    uniforms[location] = tuple(value[i] for i in range(size))

  def draw(vertices, count, color):
    if isinstance(vertices, list):
      points = np.array(vertices, dtype=np.float32)
    else:
      points = np.array([(vertices[i].x, vertices[i].y) for i in range(count)], dtype=np.float32)
    rgba = color if isinstance(color, tuple) else (color.r, color.g, color.b, color.a)
    active = ({name: value for name, value in uniforms.items() if name.startswith('gradient')}
              if uniforms['useGradient'][0] else {'fillColor': uniforms['fillColor']})
    frames.append((points.tobytes(), count, {'useGradient': uniforms['useGradient'], **active}, rgba))

  with ExitStack() as stack:
    stack.enter_context(patch.object(polygon.ShaderState, "_instance", None))
    state = polygon.ShaderState.get_instance()
    state.locations = {name: name for name in state.locations}
    stack.enter_context(patch.object(state, "initialize"))
    for name, side_effect in (("set_shader_value", set_uniform), ("set_shader_value_v", set_uniform_array),
                              ("draw_triangle_strip", draw)):
      stack.enter_context(patch.object(rl, name, side_effect=side_effect))
    for name in ("begin_shader_mode", "end_shader_mode", "draw_circle"):
      stack.enter_context(patch.object(rl, name))
    yield frames


class TestPreparedPolygon(HeadlessUITestCase):
  def test_exact_vertices_for_float_conversion_strides_and_odd_counts(self):
    for count in (0, 1, 2, 3, 4, 5, 33, 66, 67, 99, 226):
      for dtype in (np.float32, np.float64, np.int32):
        source = (np.arange(count * 4).reshape(count, 4) / 7 - 90).astype(dtype)
        for points in (source[:, :2], source[:, ::2], source[:, 1::-1]):
          with self.subTest(count=count, dtype=dtype, strides=points.strides):
            prepared = polygon.PreparedPolygon(points)
            self.assertEqual(len(prepared), count)
            np.testing.assert_array_equal(prepared_points(prepared), reference_strip(points))

  def test_snapshot_owns_vertices_and_uncached_draw_observes_inplace_changes(self):
    points = np.arange(12, dtype=np.float32).reshape(6, 2).copy()
    original = points.copy()
    prepared = polygon.PreparedPolygon(points)
    points += 100
    rect = rl.Rectangle(0, 0, 100, 100)
    with capture_polygons() as frames:
      polygon.draw_polygon(rect, prepared, rl.Color(*rl.WHITE))
      polygon.draw_polygon(rect, points, rl.Color(*rl.WHITE))
    self.assertEqual(frames[0][0], reference_strip(original).tobytes())
    self.assertEqual(frames[1][0], reference_strip(points).tobytes())
    self.assertNotEqual(frames[0][0], frames[1][0])
    source_ref = weakref.ref(points)
    del points
    gc.collect()
    self.assertIsNone(source_ref())
    np.testing.assert_array_equal(prepared_points(prepared), reference_strip(original))

  def test_cffi_pointer_passes_through_the_installed_binding_without_copy(self):
    prepared = polygon.PreparedPolygon(np.arange(12, dtype=np.float32).reshape(6, 2))
    captured = []

    @rl.ffi.callback("void(*)(const struct Vector2 *, int)")
    def receive(points, count):
      captured.append((int(rl.ffi.cast("uintptr_t", points)), count))

    rl._wrap_function(receive)(prepared.vertices, prepared.vertex_count)
    self.assertEqual(captured, [(int(rl.ffi.cast("uintptr_t", prepared.vertices)), prepared.vertex_count)])

  def test_buffer_layout_and_snapshot_owner_lifetime(self):
    self.assertEqual(rl.ffi.sizeof("Vector2"), 2 * np.dtype(np.float32).itemsize)
    self.assertEqual(rl.ffi.offsetof("Vector2", "x"), 0)
    self.assertEqual(rl.ffi.offsetof("Vector2", "y"), np.dtype(np.float32).itemsize)
    self.assertTrue(all(field.type.cname == 'float' for _, field in rl.ffi.typeof("Vector2").fields))
    arrays = []
    build_strip = polygon._triangle_strip_array

    def capture_array(points):
      strip = build_strip(points)
      self.assertTrue(strip.flags.c_contiguous)
      arrays.append(weakref.ref(strip))
      return strip

    points = np.arange(12, dtype=np.float32).reshape(6, 2).copy()
    source_ref = weakref.ref(points)
    expected = reference_strip(points)
    with patch.object(polygon, "_triangle_strip_array", side_effect=capture_array):
      prepared = polygon.PreparedPolygon(points)
    del points
    gc.collect()
    self.assertIsNone(source_ref())
    self.assertIsNotNone(arrays[0]())
    self.assertEqual(int(rl.ffi.cast("uintptr_t", prepared.vertices)), arrays[0]().ctypes.data)
    np.testing.assert_array_equal(prepared_points(prepared), expected)
    del prepared
    gc.collect()
    self.assertIsNone(arrays[0]())

  def test_degenerate_polygons_keep_the_uncached_draw_behavior(self):
    rect = rl.Rectangle(0, 0, 100, 100)
    for count in range(7):
      points = np.full((count, 2), 5, dtype=np.float32)
      with self.subTest(count=count), capture_polygons() as frames:
        polygon.draw_polygon(rect, points, rl.Color(*rl.WHITE))
        polygon.draw_polygon(rect, polygon.PreparedPolygon(points), rl.Color(*rl.WHITE))
        if count < 3:
          self.assertEqual(frames, [])
        else:
          self.assertEqual(frames[0], frames[1])

  def test_colors_gradient_and_origin_are_current_on_every_cached_draw(self):
    points = np.arange(12, dtype=np.float32).reshape(6, 2)
    prepared = polygon.PreparedPolygon(points)
    gradient = polygon.Gradient((0.2, 0.1), (0.7, 0.9), [rl.Color(*rl.RED), rl.Color(*rl.BLUE)], [0, 1])
    with capture_polygons() as frames:
      for rect, color in ((rl.Rectangle(0, 0, 100, 200), rl.Color(20, 30, 40, 10)),
                          (rl.Rectangle(5, 10, 70, 150), rl.Color(50, 60, 70, 200))):
        for current_gradient in (None, gradient):
          kwargs = {"gradient": current_gradient} if current_gradient else {"color": color}
          polygon.draw_polygon(rect, points, **kwargs)
          with patch.object(polygon, "_triangle_strip_array", side_effect=AssertionError("cached draw rebuilt vertices")):
            polygon.draw_polygon(rect, prepared, **kwargs)
          self.assertEqual(frames[-2], frames[-1])
        gradient.colors[0] = rl.Color(80, 90, 100, 120)
        gradient.stops[0] = 0.25
    self.assertNotEqual(frames[0][2]['fillColor'], frames[4][2]['fillColor'])
    self.assertNotEqual(frames[2][2]['gradientColors'], frames[6][2]['gradientColors'])
    self.assertNotEqual(frames[2][2]['gradientStart'], frames[6][2]['gradientStart'])


class TestModelPolygonCache(HeadlessUITestCase):
  def make_renderer(self):
    from openpilot.selfdrive.ui.onroad import model_renderer as model

    x = np.linspace(1, 80, 33, dtype=np.float32)
    lines = [SimpleNamespace(x=x, y=np.full(33, y, dtype=np.float32), z=np.ones(33)) for y in (0, -4, -2, 2, 4, -6, 6)]
    model_message = SimpleNamespace(position=lines[0], laneLines=lines[1:5], roadEdges=lines[5:],
                                    laneLineProbs=[0.8] * 4, roadEdgeStds=[0.1, 0.2], acceleration=SimpleNamespace(x=np.zeros(33)))

    class Messages(dict):
      recv_frame = {'extrinsicsCalibration': 1, 'modelV2': 1}
      updated = {'carParams': False, 'modelV2': True, 'radarState': False}
      valid = {'radarState': False}

    sm = Messages(modelV2=model_message, selfdriveState=SimpleNamespace(experimentalMode=False),
                  extrinsicsCalibration=SimpleNamespace(height=[1.22]), longitudinalPlan=SimpleNamespace(allowThrottle=True))
    state = SimpleNamespace(sm=sm, started_frame=0, active_bundle=False, rainbow_path=False)
    self.enterContext(patch.object(model, "ui_state", state))
    self.enterContext(patch.object(model, "Params", return_value=Mock(get=Mock(return_value=None))))
    self.enterContext(patch.object(model.gui_app, "font", return_value=rl.Font()))
    self.enterContext(patch.object(model.gui_app, "mark_phase"))
    renderer = model.ModelRenderer()
    self.enterContext(patch.object(renderer, "_get_path_half_width", return_value=0.9))
    renderer.set_rect(rl.Rectangle(0, 0, 2160, 1080))
    renderer.set_transform(np.array([[0, 20, 1080], [-2, 0, 850], [0, 0, 1]], dtype=np.float32))
    return renderer, sm

  @staticmethod
  def shapes(renderer):
    return [*renderer._lane_lines, *renderer._road_edges, renderer._path]

  def test_repeated_frames_reuse_prepared_geometry_and_keep_dynamic_color(self):
    renderer, sm = self.make_renderer()
    with capture_polygons() as frames:
      renderer._render(renderer.rect)
      self.assertEqual(len(frames), 7)
      prepared = [shape.polygon for shape in self.shapes(renderer)]
      sm.updated['modelV2'] = False
      with patch.object(polygon, "_triangle_strip_array", side_effect=AssertionError("unchanged model rebuilt vertices")):
        renderer._render(renderer.rect)
        self.assertEqual(frames[:7], frames[7:])
        renderer._lane_line_probs[0] = 0.25
        renderer._draw_lane_lines()
      self.assertEqual([shape.polygon for shape in self.shapes(renderer)], prepared)
      self.assertNotEqual(frames[0][2]['fillColor'], frames[14][2]['fillColor'])

  def test_radar_updates_reuse_lanes_and_keep_path_lead_width_and_gradient_updates(self):
    renderer, sm = self.make_renderer()
    sm.valid['radarState'] = True
    sm['radarState'] = SimpleNamespace(leadOne=SimpleNamespace(present=True, dRel=30.0))
    sm['selfdriveState'].experimentalMode = True

    with capture_polygons() as frames:
      renderer._render(renderer.rect)
      for distance, width in ((10.0, 0.8), (10.0, 0.7), (40.0, 0.7)):
        old = [shape.polygon for shape in self.shapes(renderer)]
        old_gradient = renderer._exp_gradient
        width_calls = renderer._get_path_half_width.call_count
        renderer._get_path_half_width.return_value = width
        sm.updated['modelV2'] = False
        sm.updated['radarState'] = True
        sm['radarState'].leadOne.dRel = distance
        with patch.object(renderer, "_map_line_to_polygon", wraps=renderer._map_line_to_polygon) as project:
          renderer._render(renderer.rect)

        self.assertEqual(project.call_count, 1)
        self.assertEqual(renderer._get_path_half_width.call_count, width_calls + 1)
        self.assertEqual([shape.polygon for shape in self.shapes(renderer)[:-1]], old[:-1])
        self.assertIsNot(old[-1], renderer._path.polygon)
        self.assertFalse(np.array_equal(prepared_points(old[-1]), prepared_points(renderer._path.polygon)))
        self.assertIsNot(old_gradient, renderer._exp_gradient)

        # Reusing lanes must draw exactly the same frame as a full projection refresh.
        radar_frame = frames[-7:]
        sm.updated['modelV2'] = True
        renderer._render(renderer.rect)
        self.assertEqual(radar_frame, frames[-7:])

  def test_model_transform_and_layout_changes_replace_geometry(self):
    renderer, sm = self.make_renderer()
    with capture_polygons():
      renderer._render(renderer.rect)
      old = [shape.polygon for shape in self.shapes(renderer)]
      for line in (sm['modelV2'].position, *sm['modelV2'].laneLines, *sm['modelV2'].roadEdges):
        line.y += 0.5
      renderer._render(renderer.rect)
      for before, shape in zip(old, self.shapes(renderer), strict=True):
        self.assertIsNot(before, shape.polygon)
        self.assertFalse(np.array_equal(prepared_points(before), prepared_points(shape.polygon)))

      sm.updated['modelV2'] = False
      old = [shape.polygon for shape in self.shapes(renderer)]
      transform = renderer._car_space_transform.copy()
      transform[0, 2] += 25
      renderer.set_transform(transform)
      renderer._render(renderer.rect)
      for before, shape in zip(old, self.shapes(renderer), strict=True):
        self.assertIsNot(before, shape.polygon)
        self.assertFalse(np.array_equal(prepared_points(before), prepared_points(shape.polygon)))

      renderer.set_rect(rl.Rectangle(4000, 0, 2160, 1080))
      renderer._render(renderer.rect)
      self.assertTrue(all(shape.polygon.vertex_count == 0 for shape in self.shapes(renderer)))

  def test_model_points_release_replaced_snapshots(self):
    from openpilot.selfdrive.ui.onroad.model_renderer import ModelPoints

    shape = ModelPoints()
    refs = []
    for i in range(100):
      shape.set_projected_points(np.full((66, 2), i, dtype=np.float32))
      refs.append(weakref.ref(shape.polygon))
    gc.collect()
    self.assertEqual(sum(ref() is not None for ref in refs), 1)
    self.assertIs(refs[-1](), shape.polygon)


class TestRainbowPolygonCache(HeadlessUITestCase):
  def test_cached_path_reuses_vertices_and_keeps_animated_gradient(self):
    from openpilot.selfdrive.ui.onroad.model_renderer import ModelPoints
    from openpilot.selfdrive.ui.sunnypilot.onroad import rainbow_path

    points = np.arange(12, dtype=np.float32).reshape(6, 2)
    path = ModelPoints(projected_points=points)
    prepared = path.polygon
    rainbow = rainbow_path.RainbowPath()
    rect = rl.Rectangle(0, 0, 100, 200)
    with capture_polygons() as frames, patch.object(rainbow_path.time, "monotonic") as monotonic:
      for timestamp in (1.0, 2.0):
        monotonic.return_value = timestamp
        polygon.draw_polygon(rect, points, gradient=rainbow.get_gradient())
        with patch.object(polygon, "_triangle_strip_array", side_effect=AssertionError("rainbow draw rebuilt cached vertices")):
          rainbow.draw_rainbow_path(rect, path)
        self.assertEqual(frames[-2], frames[-1])
    self.assertIs(path.polygon, prepared)
    self.assertEqual(frames[0][0], frames[2][0])
    self.assertNotEqual(frames[0][2]['gradientColors'], frames[2][2]['gradientColors'])

  def test_mici_path_without_prepared_polygon_keeps_current_points_and_colors(self):
    from openpilot.selfdrive.ui.mici.onroad.model_renderer import ModelPoints
    from openpilot.selfdrive.ui.sunnypilot.onroad import rainbow_path

    path = ModelPoints(projected_points=np.arange(12, dtype=np.float32).reshape(6, 2))
    rainbow = rainbow_path.RainbowPath()
    rect = rl.Rectangle(0, 0, 100, 200)
    with capture_polygons() as frames, patch.object(rainbow_path.time, "monotonic") as monotonic:
      for timestamp, offset in ((1.0, 0), (2.0, 50)):
        monotonic.return_value = timestamp
        path.projected_points += offset
        polygon.draw_polygon(rect, path.projected_points, gradient=rainbow.get_gradient())
        rainbow.draw_rainbow_path(rect, path)
        self.assertEqual(frames[-2], frames[-1])
    self.assertNotEqual(frames[0][0], frames[2][0])
    self.assertNotEqual(frames[0][2]['gradientColors'], frames[2][2]['gradientColors'])


class TestTorquePolygonCache(HeadlessUITestCase):
  def test_translation_is_exact_and_changes_invalidate_one_entry(self):
    from openpilot.selfdrive.ui.mici.onroad.torque_bar import _OffsetPolygon

    source = np.array([[0.1, -1200], [0.3, -1250], [10.9, -1250], [9.7, -1200]], dtype=np.float32)
    cache = _OffsetPolygon()
    for offset in ((1088.3, 2222.6), (1088.3, 2222.60001), (1200, 2300)):
      previous = cache.polygon
      cache.update(source, offset)
      self.assertIsNot(previous, cache.polygon)
      expected = source + np.array(offset, dtype=np.float32)
      np.testing.assert_array_equal(cache.points, expected)
      np.testing.assert_array_equal(prepared_points(cache.polygon), reference_strip(expected))
      previous = cache.polygon
      cache.update(source, offset)
      self.assertIs(previous, cache.polygon)
    cache.update(source.copy(), offset)
    self.assertIsNot(previous, cache.polygon)

  def test_arc_lru_and_translated_snapshots_have_fixed_bounds(self):
    from openpilot.selfdrive.ui.mici.onroad.torque_bar import _OffsetPolygon, quantized_lru_cache

    @quantized_lru_cache(maxsize=256)
    def arc(radius, thickness, start, end):
      return np.full((66, 2), radius, dtype=np.float32)

    caches = [_OffsetPolygon(), _OffsetPolygon()]
    refs, snapshot_refs = [], []
    for i in range(302):
      source = arc(i, 10, 0, 1)
      refs.append(weakref.ref(source))
      if i < 2:
        caches[i].update(source, (0, 0))
    del source
    gc.collect()
    self.assertEqual(sum(ref() is not None for ref in refs), 258)  # 256 LRU + two widget-held origins
    for i in range(100):
      cache = caches[i % 2]
      cache.update(arc(i, 10, 0, 1), (i, i))
      snapshot_refs.append(weakref.ref(cache.polygon))
    gc.collect()
    self.assertEqual(sum(ref() is not None for ref in snapshot_refs), 2)

  def test_widget_reuses_quantized_hits_but_respects_exact_offset_scale_and_layout(self):
    from openpilot.selfdrive.ui.mici.onroad.torque_bar import TorqueBar

    bar = TorqueBar(demo=True)
    bar.set_rect(rl.Rectangle(0, 0, 2160, 1080))
    bar._torque_filter.x = 0.3
    bar._torque_line_alpha_filter.x = 1.0
    with capture_polygons() as frames:
      bar._render(bar.rect)
      before = (bar._bg_polygon.polygon, bar._sl_polygon.polygon)
      bar._torque_filter.x += 0.00001
      bar._render(bar.rect)
      self.assertEqual(before, (bar._bg_polygon.polygon, bar._sl_polygon.polygon))
      self.assertEqual(frames[:2], frames[2:])

      bar._torque_filter.x = 0.31
      bar._render(bar.rect)
      self.assertIs(before[0], bar._bg_polygon.polygon)
      self.assertIsNot(before[1], bar._sl_polygon.polygon)

      bar._torque_filter.x = 0.75001
      bar._render(bar.rect)
      before = (bar._bg_polygon.polygon, bar._sl_polygon.polygon)
      source = bar._bg_polygon._source
      bar._torque_filter.x = 0.75002
      bar._render(bar.rect)
      self.assertIs(source, bar._bg_polygon._source)
      self.assertTrue(all(a is not b for a, b in zip(before, (bar._bg_polygon.polygon, bar._sl_polygon.polygon), strict=True)))

      for rect, scale in ((rl.Rectangle(5, 10, 1600, 900), 1.0), (rl.Rectangle(5, 10, 1600, 900), 0.7)):
        before = (bar._bg_polygon.polygon, bar._sl_polygon.polygon)
        bar.set_rect(rect)
        bar._scale = scale
        bar._render(bar.rect)
        self.assertTrue(all(a is not b for a, b in zip(before, (bar._bg_polygon.polygon, bar._sl_polygon.polygon), strict=True)))

  def test_cached_torque_geometry_still_updates_color_and_alpha(self):
    from openpilot.selfdrive.ui.mici.onroad import torque_bar as torque
    from openpilot.selfdrive.ui.ui_state import UIStatus

    bar = torque.TorqueBar(demo=True)
    bar.set_rect(rl.Rectangle(0, 0, 2160, 1080))
    bar._torque_filter.x = 0.3
    bar._torque_line_alpha_filter.x = 1.0
    self.enterContext(patch.object(torque, "ui_state", SimpleNamespace(status=UIStatus.DISENGAGED)))
    with capture_polygons() as frames:
      bar._render(bar.rect)
      before = (bar._bg_polygon.polygon, bar._sl_polygon.polygon)
      bar._demo = False
      # Preserve geometry while the status alone changes the fill/gradient.
      with patch.object(bar._torque_line_alpha_filter, "update"):
        bar._render(bar.rect)
      self.assertEqual(before, (bar._bg_polygon.polygon, bar._sl_polygon.polygon))
      self.assertNotEqual(frames[0][2]['fillColor'], frames[2][2]['fillColor'])
      self.assertNotEqual(frames[1][2]['gradientColors'], frames[3][2]['gradientColors'])
