from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pyray as rl

from openpilot.cereal import custom
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params, ParamKeyFlag, ParamKeyType
from openpilot.selfdrive.ui.tests.test_polygon_cache import capture_polygons, reference_strip
from openpilot.selfdrive.ui.tests.test_sunnylink_removed import HeadlessUITestCase
from openpilot.selfdrive.ui.sunnypilot.ui_state import UIStateSP, MADSState
from openpilot.selfdrive.ui.translations.potools import parse_po
from openpilot.selfdrive.ui.ui_state import UIStatus
from openpilot.system.ui.lib import shader_polygon as polygon

START = 1_000_000_000


@contextmanager
def capture_native_polygons():
  """Run real shader setup and pyray vertex conversion into a no-op native draw."""
  with capture_polygons() as frames:
    record_draw = rl.draw_triangle_strip

    @rl.ffi.callback("void(*)(const struct Vector2 *, int)")
    def receive(vertices, count):
      record_draw(vertices, count, rl.WHITE)

    native_draw = rl._wrap_function(receive)
    with patch.object(rl, 'draw_triangle_strip', side_effect=lambda vertices, count, _color: native_draw(vertices, count)):
      yield frames


class Messages(dict):
  def __init__(self, **messages):
    super().__init__(messages)
    self.logMonoTime = dict.fromkeys(messages, START)
    self.valid = dict.fromkeys(messages, True)
    self.recv_frame = dict.fromkeys(messages, 10)
    self.updated = dict.fromkeys(messages, False)


def state_with_runner(runner=custom.ModelManagerSP.Runner.tinygrad):
  params = Params()
  params.put("ModelRunnerTypeCache", runner, block=True)
  state = SimpleNamespace(params=params, chestnut_present=False, chestnut_active=False, chestnut_loading=False,
                          chestnut_compiled=False, _enforce_constraints=Mock(), _sp_initialized=True,
                          is_offroad=lambda: True, is_metric=True, CP=None, status=UIStatus.ENGAGED,
                          started_frame=0, radar_signal=True)
  UIStateSP.update_params(state)
  return state


def model_messages():
  x = np.linspace(1, 80, 33, dtype=np.float32)
  lines = [SimpleNamespace(x=x, y=np.full(33, y, dtype=np.float32), z=np.zeros(33, dtype=np.float32))
           for y in (-5.25, -1.75, 1.75, 5.25)]
  model = SimpleNamespace(frameId=7, position=SimpleNamespace(x=x, y=np.zeros(33), z=np.zeros(33)),
                          laneLines=lines, laneLineStds=[0.1] * 4, laneLineProbs=[1.0] * 4,
                          roadEdges=[lines[0], lines[3]], roadEdgeStds=[0.1, 0.1],
                          acceleration=SimpleNamespace(x=np.zeros(33)))
  sm = Messages(modelV2=model,
                modelDataV2SP=SimpleNamespace(blendfulTrust=0.8, blendfulCurvature=0.001,
                                            blendfulLeftProb=1.0, blendfulRightProb=1.0, blendfulLaneWidth=3.5),
                carState=SimpleNamespace(vEgo=15.0), carParams=SimpleNamespace(),
                carControl=SimpleNamespace(latActive=True),
                carOutput=SimpleNamespace(actuatorsOutput=SimpleNamespace(torque=0.0)),
                selfdriveState=SimpleNamespace(experimentalMode=False),
                selfdriveStateSP=SimpleNamespace(mads=SimpleNamespace(available=False)),
                radarState=SimpleNamespace(), extrinsicsCalibration=SimpleNamespace(height=[1.2]))
  sm.valid['radarState'] = False
  return sm


class TestBlendfulSettings(HeadlessUITestCase):
  @contextmanager
  def settings(self, variant):
    from openpilot.selfdrive.ui.sunnypilot.layouts.settings import steering
    from openpilot.selfdrive.ui.sunnypilot.mici.layouts import models as mici
    from openpilot.system.ui.lib.application import gui_app
    from openpilot.system.ui.sunnypilot.widgets.list_view import ToggleActionSP

    module = steering if variant == 'big' else mici
    state = state_with_runner()
    with ExitStack() as stack:
      stack.enter_context(patch.object(module, 'ui_state', state))
      stack.enter_context(patch.object(module, 'tr', side_effect=lambda value: value))
      stack.enter_context(patch.object(gui_app, 'font', return_value=rl.Font()))
      if variant == 'big':
        def toggle_item(title, description=None, **kwargs):
          item = Mock()
          item.title = title() if callable(title) else title
          item.description, item.description_visible = description, False
          item.action_item = ToggleActionSP(param=kwargs['param'])
          item.set_description.side_effect = lambda value: setattr(item, 'description', value)
          item.show_description.side_effect = lambda value: setattr(item, 'description_visible', value)
          return item

        stack.enter_context(patch.object(module, 'toggle_item_sp', side_effect=toggle_item))
        for name in ('simple_button_item_sp', 'option_item_sp', 'LineSeparatorSP', 'Scroller',
                     'LaneChangeSettingsLayout', 'MadsSettingsLayout', 'TorqueSettingsLayout'):
          stack.enter_context(patch.object(module, name))
        layout = module.SteeringLayout()
        control = layout.blendful_toggle.action_item

        def selected():
          return control.get_state()

        def select():
          # Match the render-time propagation from action to its real ToggleSP.
          control.toggle.set_enabled(control.enabled)
          params = Params()
          with patch.object(control.toggle.params, 'put_bool', side_effect=lambda key, value: params.put_bool(key, value, block=True)):
            control.toggle._handle_mouse_release(None)
      else:
        def initialize(layout):
          layout._scroller = Mock()

        stack.enter_context(patch.object(module.NavScroller, '__init__', initialize))
        stack.enter_context(patch.object(module.NavScroller, 'show_event'))
        stack.enter_context(patch.object(module, 'CurrentModelInfo'))
        stack.enter_context(patch.object(module.BigButton, '_load_images'))
        stack.enter_context(patch.object(module.BigParamControl, '_load_images'))
        layout = module.ModelsLayoutMici()
        control = layout.blendful_toggle

        def selected():
          return control._checked

        def select():
          if control.enabled:
            control._handle_mouse_release(None)

      yield module, state, layout, control, selected, select

  def test_default_v2_exposes_switch_and_changes_persist_without_restart(self):
    for variant in ('big', 'mici'):
      with self.subTest(variant=variant), self.settings(variant) as (module, state, layout, control, selected, select):
        self.assertIsNone(state.active_bundle)
        self.assertTrue(control.enabled)
        items = module.Scroller.call_args.args[0] if variant == 'big' else layout.main_items
        self.assertIn(layout.blendful_toggle, items)
        if variant == 'big':
          self.assertEqual(items.index(layout._blinker_control_toggle), items.index(layout.blendful_toggle) + 1)
        state.params.put_bool('Blendful', False, block=True)
        layout.show_event()
        self.assertFalse(selected())
        for expected in (True, False):
          select()
          self.assertEqual(state.params.get_bool('Blendful'), expected)
          layout.show_event()
          self.assertEqual(selected(), expected)
        state.params.put_bool('Blendful', True, block=True)
        layout.show_event()
        self.assertTrue(selected())
        self.assertFalse(state.params.get_bool('OnroadCycleRequested'))

  def test_stock_runner_disables_entry_without_erasing_preference(self):
    for variant in ('big', 'mici'):
      with self.subTest(variant=variant), self.settings(variant) as (_, state, layout, control, selected, _):
        state.params.put_bool('Blendful', True, block=True)
        state.params.put('ModelRunnerTypeCache', custom.ModelManagerSP.Runner.stock, block=True)
        UIStateSP.update_params(state)
        layout.show_event()
        self.assertFalse(control.enabled)
        self.assertFalse(selected())
        self.assertTrue(state.params.get_bool('Blendful'))
        if variant == 'big':
          self.assertIn('active model does not support', layout.blendful_toggle.description)
          self.assertTrue(layout.blendful_toggle.description_visible)
        else:
          self.assertEqual(control.value, 'v2 model required')

        state.params.put('ModelRunnerTypeCache', custom.ModelManagerSP.Runner.tinygrad, block=True)
        UIStateSP.update_params(state)
        layout.show_event()
        self.assertTrue(control.enabled)
        self.assertTrue(selected())

  def test_both_settings_explain_speed_default_v2_and_cyan_overlay(self):
    with self.settings('big') as (_, _, layout, _, _, _):
      text = layout.blendful_toggle.description
      self.assertIn('18 km/h (5 m/s)', text)
      self.assertIn('cyan line', text)
      self.assertIn('Default v2', text)
      self.assertIn('filtered lane confidence', text)
      self.assertIn('learned lane width', text)
      self.assertIn('no fixed hard timeout', text)
      self.assertIn('Model state can build while lateral control is inactive', text)
      self.assertIn('display clears if model data stops', text)
      self.assertNotIn('only currently valid', text)
      self.assertNotIn('fixed nominal lane width', text)
    with self.settings('mici') as (module, _, layout, _, _, _):
      page = Mock()
      with patch.object(module, 'NavScroller', return_value=page), \
           patch.object(module, 'GreyBigButton', side_effect=lambda title, text='': (title, text)), \
           patch.object(module.gui_app, 'push_widget') as push:
        layout.blendful_info_btn._click_callback()
        push.assert_called_once_with(page)
        text = ' '.join(value for card in page._scroller.add_widgets.call_args.args[0] for value in card)
        for expected in ('18 km/h (5 m/s)', 'cyan line', 'top readout', 'Default v2', 'filtered lane confidence',
                         'learned lane width', 'no fixed hard timeout', 'Correction fades at low speed',
                         'Model state can build while lateral control is inactive', 'display clears if model data stops'):
          self.assertIn(expected, text)
        self.assertNotIn('only currently valid', text)
        self.assertNotIn('fixed nominal lane width', text)

  def test_native_key_is_default_off_persistent_and_backed_up(self):
    params = Params()
    params.remove('Blendful')
    self.assertEqual(params.get_type('Blendful'), ParamKeyType.BOOL)
    self.assertFalse(params.get('Blendful', return_default=True))
    self.assertIn(b'Blendful', params.all_keys(ParamKeyFlag.PERSISTENT))
    self.assertIn(b'Blendful', params.all_keys(ParamKeyFlag.BACKUP))
    for value in (True, False):
      params.put_bool('Blendful', value, block=True)
      params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
      params.clear_all(ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION)
      self.assertEqual(Params().get_bool('Blendful'), value)

  def test_new_strings_have_both_chinese_translations(self):
    for language in ('zh-CHS', 'zh-CHT'):
      _, entries = parse_po(Path(BASEDIR) / f'openpilot/selfdrive/ui/translations/app_{language}.po')
      translations = {entry.msgid: entry.msgstr for entry in entries}
      for text in ('Blendful (Alpha)', 'blendful (alpha)', 'about blendful', 'v2 model required',
                   'Starts at 18 km/h (5 m/s), outside turns and lane changes.',
                   'Uses filtered lane confidence and a learned lane width.',
                   'On lane-line loss, the previous fit and confidence can persist.',
                   'This history has no fixed hard timeout.',
                   'Correction fades at low speed, during turns or lane changes.',
                   'Model state can build while lateral control is inactive.',
                   'The display clears if model data stops or becomes invalid.',
                   'A cyan line shows the lane center while active.',
                   'Supports v2 models, including the bundled Default v2 model.'):
        with self.subTest(language=language, text=text):
          self.assertTrue(translations[text])
          self.assertNotEqual(translations[text], text)


class TestBlendfulRenderer(HeadlessUITestCase):
  def setUp(self):
    super().setUp()
    from openpilot.selfdrive.ui.sunnypilot.onroad import blendful
    self.module = blendful
    self.state = state_with_runner()
    self.state.params.put_bool('Blendful', True, block=True)
    self.state.sm = model_messages()
    self.enterContext(patch.object(blendful, 'ui_state', self.state))
    self.enterContext(patch.object(blendful.gui_app, 'font', return_value=rl.Font()))
    self.clock = self.enterContext(patch.object(blendful.time, 'monotonic_ns', return_value=START + 1))
    self.draw_polygon = self.enterContext(patch.object(blendful, 'draw_polygon'))
    lines = [SimpleNamespace(raw_points=np.array([line.x, line.y, line.z], dtype=np.float32).T)
             for line in self.state.sm['modelV2'].laneLines]
    self.renderer = SimpleNamespace(_model_geometry_ns=START, _lane_lines=lines,
                                    _map_line_to_polygon=Mock(side_effect=lambda center, *_: center[:, :2].copy()))
    self.rect = rl.Rectangle(10, 20, 1280, 720)
    self.overlay = blendful.BlendfulRenderer(readout=False)

  def update(self):
    self.overlay.update(self.renderer, 32, 80.0)

  def advance_frame(self):
    stamp = self.state.sm.logMonoTime['modelV2'] + 50_000_000
    for service in ('modelV2', 'modelDataV2SP', 'carControl'):
      self.state.sm.logMonoTime[service] = stamp
    self.renderer._model_geometry_ns = stamp
    self.clock.return_value = stamp + 1

  def assert_cleared(self):
    self.assertEqual(self.overlay._center_points.size, 0)
    self.assertEqual(self.overlay._trust, 0)
    self.assertEqual(self.overlay._alpha.x, 0)
    self.assertIs(self.overlay._center_points, self.overlay._empty_points)
    self.assertIsNone(self.overlay._prepared_polygon)
    self.assertIsNone(self.overlay._prepared_source)
    self.assertIsNone(self.overlay._prepared_origin)

  def assert_same_native_draw_as_raw(self, frames, rect):
    count = len(frames)
    self.overlay.draw(rect)
    self.assertEqual(len(frames), count + 1)
    points = self.overlay._center_points + (rect.x, rect.y) if self.overlay._rect_relative else self.overlay._center_points
    color = rl.Color(0, 200, 255, int(np.clip(self.overlay._trust, 0.2, 1.0) * 255 * self.overlay._alpha.x))
    polygon.draw_polygon(rect, points, color)
    self.assertEqual(len(frames), count + 2)
    self.assertEqual(frames[-2], frames[-1])
    self.assertEqual(frames[-2][0], reference_strip(points).tobytes())

  def test_cyan_center_uses_current_lines_without_changing_model_geometry(self):
    before = [line.raw_points.copy() for line in self.renderer._lane_lines]
    self.update()
    self.overlay.draw(self.rect)
    center = self.renderer._map_line_to_polygon.call_args.args[0]
    np.testing.assert_allclose(center[:, 1], 0)
    self.assertEqual(self.renderer._map_line_to_polygon.call_args.args[1:], (0.15, 0.0, 32, 80.0))
    color = self.draw_polygon.call_args.args[-1]
    self.assertEqual((color.r, color.g, color.b), (0, 200, 255))
    self.assertGreater(color.a, 0)
    for original, current in zip(before, self.renderer._lane_lines, strict=True):
      np.testing.assert_array_equal(current.raw_points, original)

  def test_mici_adds_relative_origin_and_numeric_direction_readout(self):
    self.overlay = self.module.BlendfulRenderer(rect_relative=True)
    self.update()
    with patch.object(self.module, 'measure_text_cached', return_value=rl.Vector2(40, 44)), \
         patch.object(rl, 'draw_text_ex') as text, patch.object(rl, 'draw_triangle') as arrows, \
         patch.object(rl, 'draw_rectangle_rounded') as bars, patch.object(rl, 'draw_rectangle'), \
         patch.object(self.module, 'draw_polygon', polygon.draw_polygon), capture_native_polygons() as frames:
      self.overlay.draw(self.rect)
      text.assert_called_once()
      self.assertEqual(arrows.call_count, 2)
      self.assertGreaterEqual(bars.call_count, 2)
    self.assertEqual(frames[0][0], reference_strip(self.overlay._center_points + (10, 20)).tobytes())

  def test_prepared_center_matches_native_raw_vertices_and_colors_in_both_layouts(self):
    for relative in (False, True):
      with self.subTest(relative=relative):
        self.overlay = self.module.BlendfulRenderer(readout=False, rect_relative=relative)
        self.update()
        # Include noncontiguous, odd-count vertices and fractional origins to
        # exercise the original addition, float32 conversion and strip ordering.
        self.overlay._center_points = (np.arange(268, dtype=np.float64).reshape(67, 4) / 7 - 23)[:, ::2]
        with patch.object(self.module, 'draw_polygon', polygon.draw_polygon), capture_native_polygons() as frames, \
             patch.object(self.module, 'PreparedPolygon', wraps=polygon.PreparedPolygon) as prepare:
          for alpha, trust in ((0.2, 0.1), (0.6, 0.8), (0.95, 1.0)):
            self.overlay._alpha.x, self.overlay._trust = alpha, trust
            self.assert_same_native_draw_as_raw(frames, rl.Rectangle(10.125, -20.375, 1280, 720))
          self.assertEqual(prepare.call_count, 1)
          self.assertNotEqual(frames[0][2]['fillColor'], frames[-2][2]['fillColor'])

  def test_prepares_lazily_and_rebuilds_only_for_new_geometry_or_exact_mici_origin(self):
    for relative in (False, True):
      with self.subTest(relative=relative):
        self.overlay = self.module.BlendfulRenderer(readout=False, rect_relative=relative)
        self.update()
        with patch.object(self.module, 'draw_polygon', polygon.draw_polygon), capture_native_polygons() as frames, \
             patch.object(self.module, 'PreparedPolygon', wraps=polygon.PreparedPolygon) as prepare:
          self.assertIsNone(self.overlay._prepared_polygon)
          self.overlay._trust = 0.04
          self.overlay.draw(self.rect)
          prepare.assert_not_called()
          self.assertEqual(frames, [])
          self.overlay._trust = 0.8
          self.assert_same_native_draw_as_raw(frames, self.rect)
          first = self.overlay._prepared_polygon
          self.assert_same_native_draw_as_raw(frames, self.rect)
          self.assertIs(self.overlay._prepared_polygon, first)
          self.assertEqual(prepare.call_count, 1)

          changed = rl.Rectangle(self.rect.x + 0.125, self.rect.y - 0.25, 640, 480)
          self.assert_same_native_draw_as_raw(frames, changed)
          self.assertEqual(prepare.call_count, 2 if relative else 1)
          if relative:
            self.assertEqual(self.overlay._prepared_origin, (changed.x, changed.y))
          else:
            self.assertIs(self.overlay._prepared_polygon, first)
          current = self.overlay._prepared_polygon
          resized = rl.Rectangle(changed.x, changed.y, 1920, 1080)
          self.assert_same_native_draw_as_raw(frames, resized)
          self.assertIs(self.overlay._prepared_polygon, current)

          # A real model update replaces the array even if the coordinates match.
          self.advance_frame()
          self.update()
          self.assertIsNone(self.overlay._prepared_polygon)
          self.assertIsNone(self.overlay._prepared_source)
          self.assertIsNone(self.overlay._prepared_origin)
          self.assert_same_native_draw_as_raw(frames, resized)
          self.assertEqual(prepare.call_count, 3 if relative else 2)
          self.assertIs(self.overlay._prepared_source, self.overlay._center_points)

  def test_source_held_fit_reuses_one_prepared_snapshot_with_live_confidence_and_fade(self):
    self.overlay = self.module.BlendfulRenderer(readout=False, rect_relative=True)
    self.update()
    source = self.overlay._center_points
    with patch.object(self.module, 'draw_polygon', polygon.draw_polygon), capture_native_polygons() as frames, \
         patch.object(self.module, 'PreparedPolygon', wraps=polygon.PreparedPolygon) as prepare:
      self.assert_same_native_draw_as_raw(frames, self.rect)
      prepared = self.overlay._prepared_polygon
      output = self.state.sm['modelDataV2SP']
      output.blendfulLeftProb = output.blendfulRightProb = 0.0
      self.renderer._lane_lines[1].raw_points[:, 1] += 1.0
      self.renderer._lane_lines[2].raw_points[:, 1] += 1.0
      for trust in (0.4, 0.1, 0.0, 0.0):
        output.blendfulTrust = trust
        self.advance_frame()
        self.update()
        self.assertIs(self.overlay._center_points, source)
        self.assert_same_native_draw_as_raw(frames, self.rect)
        self.assertIs(self.overlay._prepared_polygon, prepared)
      self.assertEqual(prepare.call_count, 1)
      self.assertEqual(len({frame[2]['fillColor'] for frame in frames[::2]}), 5)

  def test_20hz_geometry_60hz_draw_uses_one_snapshot_per_current_model_array(self):
    for relative in (False, True):
      with self.subTest(relative=relative):
        self.state.sm = model_messages()
        self.clock.return_value = START + 1
        self.renderer._model_geometry_ns = START
        self.overlay = self.module.BlendfulRenderer(readout=False, rect_relative=relative)
        with patch.object(self.module, 'draw_polygon', polygon.draw_polygon), capture_native_polygons() as frames, \
             patch.object(self.module, 'PreparedPolygon', wraps=polygon.PreparedPolygon) as prepare:
          for frame in range(4):
            if frame:
              self.advance_frame()
            self.update()
            stamp = self.state.sm.logMonoTime['modelV2']
            for tick in range(3):
              self.clock.return_value = stamp + tick * 16_666_667 + 1
              self.assert_same_native_draw_as_raw(frames, self.rect)
            self.assertEqual(prepare.call_count, frame + 1)
          self.assertEqual(len(frames), 24)

  def test_hard_clear_reuses_empty_array_and_releases_prepared_references(self):
    self.overlay = self.module.BlendfulRenderer(readout=False, rect_relative=True)
    self.update()
    self.overlay.draw(self.rect)
    self.assertIsNotNone(self.overlay._prepared_polygon)
    self.state.params.put_bool('Blendful', False, block=True)
    with patch.object(self.module.np, 'empty', side_effect=AssertionError('clear allocated another empty array')):
      for _ in range(10):
        self.overlay.draw(self.rect)
        self.overlay.clear()
        self.assert_cleared()

  def test_alpha_empty_path_releases_cache_but_preserves_source_filter_history(self):
    self.overlay = self.module.BlendfulRenderer(rect_relative=True)
    self.update()
    with patch.object(self.overlay, '_draw_readout'):
      self.overlay.draw(self.rect)
    self.assertIsNotNone(self.overlay._prepared_polygon)
    trust = self.overlay._trust
    self.overlay._alpha.x = 0.009
    output = self.state.sm['modelDataV2SP']
    output.blendfulTrust = output.blendfulCurvature = 0.0
    with patch.object(self.module.np, 'empty', side_effect=AssertionError('fade allocated another empty array')):
      for _ in range(3):
        self.advance_frame()
        self.update()
        self.assertIs(self.overlay._center_points, self.overlay._empty_points)
        self.assertIsNone(self.overlay._prepared_polygon)
        self.assertIsNone(self.overlay._prepared_source)
        self.assertIsNone(self.overlay._prepared_origin)
        self.assertEqual(self.overlay._trust, trust)
        self.assertGreater(self.overlay._alpha.x, 0)
        self.assertLess(self.overlay._alpha.x, 0.01)
        self.assertGreater(self.overlay._pull_display.x, 0)
        self.assertGreater(self.overlay._trust_display.x, 0)

  def test_disabled_unsupported_inactive_and_paused_clear_on_draw_without_updates(self):
    for case in ('disabled', 'stock', 'runner_cache_changed', 'inactive', 'paused'):
      with self.subTest(case=case):
        self.state.params.put_bool('Blendful', True, block=True)
        self.state.params.put('ModelRunnerTypeCache', custom.ModelManagerSP.Runner.tinygrad, block=True)
        self.state.active_runner_tinygrad = True
        self.state.status = UIStatus.ENGAGED
        self.state.sm['selfdriveStateSP'].mads = SimpleNamespace(available=False)
        self.update()
        self.assertGreater(self.overlay._center_points.size, 0)
        self.overlay.draw(self.rect)
        self.assertIsNotNone(self.overlay._prepared_polygon)
        if case == 'disabled':
          self.state.params.put_bool('Blendful', False, block=True)
        elif case == 'stock':
          self.state.active_runner_tinygrad = False
        elif case == 'runner_cache_changed':
          self.state.params.put('ModelRunnerTypeCache', custom.ModelManagerSP.Runner.stock, block=True)
        elif case == 'inactive':
          self.state.status = UIStatus.DISENGAGED
        else:
          self.state.sm['selfdriveStateSP'].mads = SimpleNamespace(available=True, enabled=True, state=MADSState.paused)
        self.draw_polygon.reset_mock()
        self.overlay.draw(self.rect)
        self.assert_cleared()
        self.draw_polygon.assert_not_called()

  def test_invalid_missing_old_future_or_unpaired_messages_clear_on_every_draw(self):
    for service in ('modelDataV2SP', 'modelV2'):
      for case in ('invalid', 'missing', 'old', 'future', 'other_frame', 'previous_drive'):
        with self.subTest(service=service, case=case):
          self.state.sm = model_messages()
          self.update()
          self.overlay.draw(self.rect)
          self.assertIsNotNone(self.overlay._prepared_polygon)
          if case == 'invalid':
            self.state.sm.valid[service] = False
          elif case == 'missing':
            del self.state.sm.logMonoTime[service]
          elif case == 'old':
            self.clock.return_value = START + self.module.MESSAGE_MAX_AGE_NS + 1
            self.state.sm.logMonoTime['carControl'] = self.clock.return_value
          elif case == 'future':
            self.state.sm.logMonoTime[service] = START + 2
          elif case == 'other_frame':
            self.state.sm.logMonoTime[service] = START - 50_000_000
          else:
            self.state.sm.recv_frame[service] = -1
          self.draw_polygon.reset_mock()
          self.overlay.draw(self.rect)
          self.assert_cleared()
          self.draw_polygon.assert_not_called()
          self.clock.return_value = START + 1

  def test_projected_geometry_must_match_current_model_and_output(self):
    self.update()
    self.overlay.draw(self.rect)
    self.assertIsNotNone(self.overlay._prepared_polygon)
    self.renderer._model_geometry_ns = START - 1
    self.update()
    self.assert_cleared()
    self.renderer._model_geometry_ns = START
    self.update()
    self.overlay.draw(self.rect)
    self.assertIsNotNone(self.overlay._prepared_polygon)
    self.state.sm.logMonoTime['modelV2'] = START + 1
    self.overlay.draw(self.rect)
    self.assert_cleared()

  def test_current_control_exit_invalidity_or_loss_clears_even_if_status_and_mads_lag(self):
    for case in ('inactive', 'invalid', 'missing', 'stale'):
      with self.subTest(case=case):
        self.state.sm = model_messages()
        self.clock.return_value = START + 1
        self.state.sm['selfdriveStateSP'].mads = SimpleNamespace(available=True, enabled=True, state=MADSState.enabled)
        self.update()
        self.assertGreater(self.overlay._center_points.size, 0)
        self.overlay.draw(self.rect)
        self.assertIsNotNone(self.overlay._prepared_polygon)
        if case == 'inactive':
          self.state.sm['carControl'].latActive = False
        elif case == 'invalid':
          self.state.sm.valid['carControl'] = False
        elif case == 'missing':
          del self.state.sm.logMonoTime['carControl']
        else:
          self.clock.return_value = START + self.module.CONTROL_MAX_AGE_NS + 1
        self.overlay.draw(self.rect)
        self.assert_cleared()

  def test_source_variance_weighting_uses_filtered_probabilities_and_learned_width(self):
    output, model = self.state.sm['modelDataV2SP'], self.state.sm['modelV2']
    output.blendfulLaneWidth = 4.2
    model.laneLineStds[1:3] = [0.12, 0.24]
    for left_prob, right_prob in ((0.8, 0.1), (1.0, 0.0)):
      with self.subTest(left_prob=left_prob, right_prob=right_prob), np.errstate(all='raise'):
        output.blendfulLeftProb, output.blendfulRightProb = left_prob, right_prob
        self.update()
        left_variance = 0.12 ** 2 / max(left_prob, 1e-4)
        right_variance = 0.24 ** 2 / max(right_prob, 1e-4)
        expected = (right_variance * (-1.75 + 2.1) + left_variance * (1.75 - 2.1)) / (left_variance + right_variance)
        center = self.renderer._map_line_to_polygon.call_args.args[0]
        np.testing.assert_allclose(center[:, 1], expected, atol=1e-6)
        self.assertGreater(float(center[0, 1]), 0.3)  # A fixed 3.6 m width would give a different center.

  def test_source_filtered_probabilities_are_not_remasked_by_current_raw_probabilities(self):
    self.state.sm['modelV2'].laneLineProbs[1:3] = [0.0, 0.1]
    self.update()
    self.assertGreater(self.overlay._center_points.size, 0)
    self.assertGreater(self.overlay._trust, 0)

  def test_source_readout_and_opacity_continue_across_side_probability_changes(self):
    self.overlay = self.module.BlendfulRenderer()
    self.update()
    pull, alpha = self.overlay._pull_display.x, self.overlay._alpha.x
    self.state.sm['modelV2'].laneLineProbs[2] = 0.0
    self.state.sm['modelDataV2SP'].blendfulRightProb = 0.0
    self.state.sm['modelDataV2SP'].blendfulLaneWidth = 3.9
    self.state.sm['modelDataV2SP'].blendfulCurvature = 0.0
    self.advance_frame()
    self.update()
    self.assertGreater(self.overlay._alpha.x, alpha)
    self.assertGreater(self.overlay._pull_display.x, 0)
    self.assertLess(self.overlay._pull_display.x, pull)

  def test_source_last_fit_can_remain_while_fresh_diagnostics_have_no_effective_new_fit(self):
    self.update()
    previous = self.overlay._center_points.copy()
    projected = self.renderer._map_line_to_polygon.call_count
    output = self.state.sm['modelDataV2SP']
    output.blendfulLeftProb = output.blendfulRightProb = 0.0
    self.renderer._lane_lines[1].raw_points[:, 1] += 1.0
    self.renderer._lane_lines[2].raw_points[:, 1] += 1.0
    for _ in range(100):
      self.advance_frame()
      self.update()
    np.testing.assert_array_equal(self.overlay._center_points, previous)
    self.assertEqual(self.renderer._map_line_to_polygon.call_count, projected)
    self.assertGreater(self.overlay._alpha.x, 0.99)
    self.overlay.draw(self.rect)
    self.draw_polygon.assert_called_once()

  def test_source_zero_trust_fades_previous_center_and_remembers_confidence_until_hidden(self):
    for _ in range(20):
      self.advance_frame()
      self.update()
    previous, confidence = self.overlay._center_points.copy(), self.overlay._trust
    alpha = self.overlay._alpha.x
    output = self.state.sm['modelDataV2SP']
    output.blendfulTrust = output.blendfulCurvature = 0.0
    output.blendfulLeftProb = output.blendfulRightProb = 0.0
    self.advance_frame()
    self.update()
    np.testing.assert_array_equal(self.overlay._center_points, previous)
    self.assertEqual(self.overlay._trust, confidence)
    self.assertAlmostEqual(self.overlay._alpha.x, alpha * (1 - self.overlay._alpha.alpha))
    self.overlay.draw(self.rect)
    self.draw_polygon.assert_called_once()
    for _ in range(40):
      self.advance_frame()
      self.update()
    self.assertEqual(self.overlay._center_points.size, 0)
    self.assertGreater(self.overlay._alpha.x, 0)
    self.assertLess(self.overlay._alpha.x, 0.01)
    self.assertEqual(self.overlay._trust, confidence)

  def test_stream_loss_clears_retained_source_fit_without_another_sp_update(self):
    self.update()
    self.overlay.draw(self.rect)
    prepared = self.overlay._prepared_polygon
    self.assertIsNotNone(prepared)
    self.state.sm['modelDataV2SP'].blendfulTrust = 0.0
    self.advance_frame()
    self.update()
    self.assertGreater(self.overlay._center_points.size, 0)
    self.assertIs(self.overlay._prepared_polygon, prepared)
    self.clock.return_value = self.state.sm.logMonoTime['modelV2'] + self.module.MESSAGE_MAX_AGE_NS + 1
    self.state.sm.logMonoTime['carControl'] = self.clock.return_value
    self.overlay.draw(self.rect)
    self.assert_cleared()

  def test_malformed_geometry_or_diagnostics_clear_instead_of_using_invalid_values(self):
    for case in ('nan_trust', 'nan_curvature', 'nan_line', 'nan_zero_prob_line', 'empty_line', 'missing_line',
                 'missing_std', 'zero_variance', 'nan_width'):
      with self.subTest(case=case):
        self.state.sm = model_messages()
        self.renderer._lane_lines = [SimpleNamespace(raw_points=np.array([line.x, line.y, line.z], dtype=np.float32).T)
                                     for line in self.state.sm['modelV2'].laneLines]
        self.update()
        self.assertGreater(self.overlay._center_points.size, 0)
        self.overlay.draw(self.rect)
        self.assertIsNotNone(self.overlay._prepared_polygon)
        output, model = self.state.sm['modelDataV2SP'], self.state.sm['modelV2']
        if case == 'nan_trust':
          output.blendfulTrust = np.nan
        elif case == 'nan_curvature':
          output.blendfulCurvature = np.nan
        elif case in ('nan_line', 'nan_zero_prob_line'):
          self.renderer._lane_lines[1].raw_points[0, 1] = np.nan
          if case == 'nan_zero_prob_line':
            output.blendfulLeftProb = 0.0
        elif case == 'empty_line':
          self.renderer._lane_lines[1].raw_points = np.empty((0, 3))
        elif case == 'missing_line':
          model.laneLines = []
        elif case == 'missing_std':
          model.laneLineStds = []
        elif case == 'zero_variance':
          model.laneLineStds[1:3] = [0.0, 0.0]
        else:
          output.blendfulLeftProb = output.blendfulRightProb = 0.5
          output.blendfulLaneWidth = np.nan
        with np.errstate(all='raise'):
          self.update()
        self.assert_cleared()

  def test_effective_fit_refreshes_geometry_without_advancing_alpha_twice(self):
    self.update()
    alpha = self.overlay._alpha.x
    self.renderer._lane_lines[1].raw_points[:, 1] += 0.2
    self.renderer._lane_lines[2].raw_points[:, 1] += 0.2
    self.update()
    self.assertEqual(self.overlay._alpha.x, alpha)
    np.testing.assert_allclose(self.overlay._center_points[:, 1], 0.2, atol=1e-6)

  def test_empty_or_nonfinite_projection_clears_a_prepared_snapshot(self):
    for points in (np.empty((0, 2), dtype=np.float32), np.full((4, 2), np.nan, dtype=np.float32)):
      with self.subTest(point_count=len(points)):
        self.update()
        self.overlay.draw(self.rect)
        self.assertIsNotNone(self.overlay._prepared_polygon)
        with patch.object(self.renderer, '_map_line_to_polygon', return_value=points):
          self.update()
        self.assert_cleared()

  def test_both_layouts_keep_main_path_and_projection_caches_while_overlay_expires(self):
    from openpilot.selfdrive.ui.onroad import model_renderer as normal
    from openpilot.selfdrive.ui.mici.onroad import model_renderer as mici
    from openpilot.selfdrive.ui.sunnypilot.onroad import model_renderer as normal_sp

    for module in (normal, mici):
      with self.subTest(module=module.__name__), ExitStack() as stack:
        self.state.sm = model_messages()
        self.state.sm.updated['modelV2'] = self.state.sm.updated['modelDataV2SP'] = True
        self.clock.return_value = START + 1
        stack.enter_context(patch.object(module, 'ui_state', self.state))
        stack.enter_context(patch.object(normal_sp, 'ui_state', self.state))
        stack.enter_context(patch.object(module.gui_app, 'mark_phase'))
        renderer = module.ModelRenderer()
        renderer.blendful._draw_readout = Mock()
        renderer.set_rect(self.rect)
        renderer._map_line_to_polygon = Mock(side_effect=lambda points, *args, **kwargs: points[:, :2].copy())
        renderer._update_experimental_gradient = Mock()
        renderer._draw_lane_lines, renderer._draw_path = Mock(), Mock()
        renderer._render(self.rect)
        self.assertGreater(renderer.blendful._center_points.size, 0)
        self.assertEqual(renderer.blendful._readout, module is mici)
        projected = renderer._map_line_to_polygon.call_count
        self.state.sm.updated = dict.fromkeys(self.state.sm, False)
        self.clock.return_value = START + self.module.MESSAGE_MAX_AGE_NS + 1
        renderer._render(self.rect)
        self.assertEqual(renderer._map_line_to_polygon.call_count, projected)
        self.assertEqual(renderer.blendful._center_points.size, 0)
        self.assertEqual(renderer._draw_lane_lines.call_count, 2)
        self.assertEqual(renderer._draw_path.call_count, 2)

  def test_sp_only_update_after_empty_model_path_or_lanes_does_not_crash_either_layout(self):
    from openpilot.selfdrive.ui.onroad import model_renderer as normal
    from openpilot.selfdrive.ui.mici.onroad import model_renderer as mici
    from openpilot.selfdrive.ui.sunnypilot.onroad import model_renderer as normal_sp

    for module in (normal, mici):
      for missing in ('path', 'lane'):
        with self.subTest(module=module.__name__, missing=missing), ExitStack() as stack:
          self.state.sm = model_messages()
          stack.enter_context(patch.object(module, 'ui_state', self.state))
          stack.enter_context(patch.object(normal_sp, 'ui_state', self.state))
          stack.enter_context(patch.object(module.gui_app, 'mark_phase'))
          renderer = module.ModelRenderer()
          renderer.set_rect(self.rect)
          renderer.blendful._draw_readout = Mock()
          renderer._map_line_to_polygon = Mock(side_effect=lambda points, *args, **kwargs: points[:, :2].copy())
          renderer._update_experimental_gradient = Mock()
          renderer._draw_lane_lines, renderer._draw_path = Mock(), Mock()
          self.state.sm.updated['modelV2'] = self.state.sm.updated['modelDataV2SP'] = True
          renderer._render(self.rect)
          self.assertGreater(renderer.blendful._center_points.size, 0)

          model = self.state.sm['modelV2']
          if missing == 'path':
            model.position = SimpleNamespace(x=[], y=[], z=[])
          else:
            model.laneLines = [SimpleNamespace(x=[], y=[], z=[])] * 4
          renderer._render(self.rect)
          self.state.sm.updated = dict.fromkeys(self.state.sm, False)
          self.state.sm.updated['modelDataV2SP'] = True
          renderer._render(self.rect)
          self.assertEqual(renderer.blendful._center_points.size, 0)
