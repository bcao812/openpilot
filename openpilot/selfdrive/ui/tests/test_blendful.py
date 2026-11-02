from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pyray as rl

from openpilot.cereal import custom
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params, ParamKeyFlag, ParamKeyType
from openpilot.selfdrive.ui.tests.test_sunnylink_removed import HeadlessUITestCase
from openpilot.selfdrive.ui.sunnypilot.ui_state import UIStateSP, MADSState
from openpilot.selfdrive.ui.translations.potools import parse_po
from openpilot.selfdrive.ui.ui_state import UIStatus

START = 1_000_000_000
EPOCH = START - 100_000_000


class Messages(dict):
  def __init__(self, **messages):
    super().__init__(messages)
    self.logMonoTime = dict.fromkeys(messages, START)
    self.valid = dict.fromkeys(messages, True)
    self.seen = dict.fromkeys(messages, True)
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
  model = SimpleNamespace(frameId=7, timestampEof=START - 10_000_000,
                          position=SimpleNamespace(x=x, y=np.zeros(33), z=np.zeros(33)),
                          laneLines=lines, laneLineStds=[0.1] * 4, laneLineProbs=[1.0] * 4,
                          roadEdges=[lines[0], lines[3]], roadEdgeStds=[0.1, 0.1],
                          acceleration=SimpleNamespace(x=np.zeros(33)))
  sm = Messages(modelV2=model,
                modelDataV2SP=SimpleNamespace(blendfulTrust=0.8, blendfulCurvature=0.001,
                                            blendfulLeftProb=1.0, blendfulRightProb=1.0, blendfulLaneWidth=3.5,
                                            blendfulControlEpoch=EPOCH),
                carState=SimpleNamespace(vEgo=15.0), carParams=SimpleNamespace(),
                carControl=SimpleNamespace(latActive=True),
                carControlSP=SimpleNamespace(blendfulControlEpoch=EPOCH, blendfulModelMonoTime=START,
                                             blendfulAppliedCurvature=0.001),
                carOutput=SimpleNamespace(actuatorsOutput=SimpleNamespace(torque=0.0)),
                selfdriveState=SimpleNamespace(experimentalMode=False),
                selfdriveStateSP=SimpleNamespace(mads=SimpleNamespace(available=False)),
                radarState=SimpleNamespace(), extrinsicsCalibration=SimpleNamespace(height=[1.2]))
  sm.valid['radarState'] = False
  return sm


class TestBlendfulSettings(HeadlessUITestCase):
  def test_ui_subscribes_to_the_optional_consumer_receipt(self):
    with patch('openpilot.selfdrive.ui.sunnypilot.ui_state.ScreenSaverSP'):
      state = UIStateSP()
    self.assertIn('carControlSP', state.sm_services_ext)
    self.assertIn('modelDataV2SP', state.sm_services_ext)

  @contextmanager
  def settings(self, variant):
    from openpilot.selfdrive.ui.sunnypilot.layouts.settings import models
    from openpilot.selfdrive.ui.sunnypilot.mici.layouts import models as mici
    from openpilot.system.ui.sunnypilot.widgets.list_view import ToggleActionSP

    module = models if variant == 'big' else mici
    state = state_with_runner()
    with ExitStack() as stack:
      stack.enter_context(patch.object(module, 'ui_state', state))
      stack.enter_context(patch.object(module, 'tr', side_effect=lambda value: value))
      stack.enter_context(patch.object(module.gui_app, 'font', return_value=rl.Font()))
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
        for name in ('ListItemSP', 'button_item', 'download_status_item', 'option_item_sp', 'ScrollingButtonAction', 'NoElideButtonAction', 'Scroller'):
          stack.enter_context(patch.object(module, name, create=True))
        stack.enter_context(patch.object(module.ModelsLayout, 'calculate_cache_size', return_value=0))
        layout = module.ModelsLayout()
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
      with self.subTest(variant=variant), self.settings(variant) as (_, state, layout, control, selected, select):
        self.assertIsNone(state.active_bundle)
        self.assertTrue(control.enabled)
        self.assertIn(layout.blendful_toggle, layout.items if variant == 'big' else layout.main_items)
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
      self.assertIn('currently valid lane lines', text)
      self.assertIn('lateral control is inactive', text)
      self.assertIn('3.6 m', text)
    with self.settings('mici') as (module, _, layout, _, _, _):
      page = Mock()
      with patch.object(module, 'NavScroller', return_value=page), \
           patch.object(module, 'GreyBigButton', side_effect=lambda title, text='': (title, text)), \
           patch.object(module.gui_app, 'push_widget') as push:
        layout.blendful_info_btn._click_callback()
        push.assert_called_once_with(page)
        text = ' '.join(value for card in page._scroller.add_widgets.call_args.args[0] for value in card)
        for expected in ('18 km/h (5 m/s)', 'cyan line', 'top readout', 'Default v2', 'currently valid lane lines',
                         'lateral control is inactive', '3.6 m'):
          self.assertIn(expected, text)

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
                   'Uses only currently valid lane lines.',
                   'Withdraws its correction when the lines are lost or lateral control is inactive.',
                   'With one valid lane line, uses a fixed nominal lane width of 3.6 m.',
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

  def assert_cleared(self):
    self.assertEqual(self.overlay._center_points.size, 0)
    self.assertEqual(self.overlay._trust, 0)
    self.assertEqual(self.overlay._alpha.x, 0)
    self.assertEqual(self.overlay._control_epoch, 0)
    self.assertEqual(self.overlay._pull_display.x, 0)
    self.assertEqual(self.overlay._opacity_epoch, 0)
    self.assertEqual(self.overlay._opacity_deadline_ns, 0)

  def publish_model(self, stamp, capture_ns=None):
    sm = self.state.sm
    for service in ('modelV2', 'modelDataV2SP', 'carControl', 'carControlSP'):
      sm.logMonoTime[service] = stamp
    sm['modelV2'].timestampEof = stamp - 10_000_000 if capture_ns is None else capture_ns
    sm['carControlSP'].blendfulModelMonoTime = stamp
    self.renderer._model_geometry_ns = stamp
    self.clock.return_value = stamp + 1

  def publish_pending_model(self, stamp, capture_ns=None):
    receipt = self.state.sm['carControlSP']
    accepted_id = receipt.blendfulModelMonoTime
    receipt_stamp = max(self.state.sm.logMonoTime['carControlSP'], stamp - 10_000_000)
    self.publish_model(stamp, capture_ns)
    receipt.blendfulModelMonoTime = accepted_id
    self.state.sm.logMonoTime['carControlSP'] = receipt_stamp

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
         patch.object(rl, 'draw_rectangle_rounded') as bars, patch.object(rl, 'draw_rectangle'):
      self.overlay.draw(self.rect)
      text.assert_called_once()
      self.assertEqual(arrows.call_count, 2)
      self.assertGreaterEqual(bars.call_count, 2)
    np.testing.assert_array_equal(self.draw_polygon.call_args.args[1], self.overlay._center_points + (10, 20))

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
          if case == 'invalid':
            self.state.sm.valid[service] = False
          elif case == 'missing':
            del self.state.sm.logMonoTime[service]
          elif case == 'old':
            self.clock.return_value = START + self.module.MESSAGE_MAX_AGE_NS + 1
            self.state.sm.logMonoTime['carControl'] = self.clock.return_value
            self.state.sm.logMonoTime['carControlSP'] = self.clock.return_value
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
    self.renderer._model_geometry_ns = START - 1
    self.update()
    self.assert_cleared()
    self.renderer._model_geometry_ns = START
    self.update()
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
        if case == 'inactive':
          self.state.sm['carControl'].latActive = False
        elif case == 'invalid':
          self.state.sm.valid['carControl'] = False
        elif case == 'missing':
          del self.state.sm.logMonoTime['carControl']
        else:
          self.clock.return_value = START + self.module.CONTROL_MAX_AGE_NS + 1
          self.state.sm.logMonoTime['carControlSP'] = self.clock.return_value
        self.overlay.draw(self.rect)
        self.assert_cleared()

  def test_consumer_receipt_is_required_current_and_finite_on_every_draw(self):
    cases = ('invalid', 'unseen', 'missing', 'stale', 'future', 'previous_drive', 'zero_epoch',
             'other_epoch', 'rejected', 'other_model', 'nan_term', 'infinite_term')
    for case in cases:
      with self.subTest(case=case):
        self.state.sm = model_messages()
        self.update()
        self.assertGreater(self.overlay._center_points.size, 0)
        sm, receipt = self.state.sm, self.state.sm['carControlSP']
        if case == 'invalid':
          sm.valid['carControlSP'] = False
        elif case == 'unseen':
          sm.seen['carControlSP'] = False
        elif case == 'missing':
          del sm.logMonoTime['carControlSP']
        elif case == 'stale':
          sm.logMonoTime['carControlSP'] = START - self.module.CONTROL_MAX_AGE_NS
        elif case == 'future':
          sm.logMonoTime['carControlSP'] = START + 2
        elif case == 'previous_drive':
          sm.recv_frame['carControlSP'] = -1
        elif case == 'zero_epoch':
          receipt.blendfulControlEpoch = 0
        elif case == 'other_epoch':
          receipt.blendfulControlEpoch = EPOCH + 1
        elif case == 'rejected':
          receipt.blendfulModelMonoTime = 0
        elif case == 'other_model':
          receipt.blendfulModelMonoTime = START - 1
        else:
          receipt.blendfulAppliedCurvature = np.nan if case == 'nan_term' else np.inf
        self.draw_polygon.reset_mock()
        self.overlay.draw(self.rect)
        self.assert_cleared()
        self.draw_polygon.assert_not_called()

  def test_pairing_is_exact_and_fresh_publication_cannot_renew_old_capture(self):
    self.update()
    self.state.sm.logMonoTime['modelDataV2SP'] = START - 1
    self.overlay.draw(self.rect)
    self.assert_cleared()
    self.state.sm = model_messages()
    self.update()
    self.publish_model(START + 300_000_000, capture_ns=START - 10_000_000)
    self.update()
    self.assert_cleared()

  def test_capture_deadline_and_epoch_are_checked_independently_of_publication(self):
    for case in ('deadline', 'pre_epoch', 'future'):
      with self.subTest(case=case):
        self.state.sm = model_messages()
        self.clock.return_value = START
        self.update()
        sm = self.state.sm
        if case == 'deadline':
          sm['carControlSP'].blendfulControlEpoch = sm['modelDataV2SP'].blendfulControlEpoch = START - 300_000_000
          sm['modelV2'].timestampEof = START - self.module.MESSAGE_MAX_AGE_NS
          self.update()
          self.assertGreater(self.overlay._center_points.size, 0)
          self.clock.return_value = START + 1
        elif case == 'pre_epoch':
          sm['modelV2'].timestampEof = EPOCH - 1
        else:
          sm['modelV2'].timestampEof = START + 1
        self.overlay.draw(self.rect)
        self.assert_cleared()

  def test_brief_exit_reentry_requires_new_epoch_capture_and_receipt(self):
    self.update()
    receipt = self.state.sm['carControlSP']
    epoch = START + 20_000_000
    self.clock.return_value = epoch + 1
    self.state.sm.logMonoTime['carControlSP'] = epoch
    receipt.blendfulControlEpoch, receipt.blendfulModelMonoTime = epoch, 0
    # UI never observed carControl.latActive=False between these two receipts.
    self.overlay.draw(self.rect)
    self.assert_cleared()
    receipt.blendfulModelMonoTime = START
    self.update()
    self.assert_cleared()
    self.state.sm['modelDataV2SP'].blendfulControlEpoch = epoch
    self.update()
    self.assert_cleared()  # the old capture still predates this epoch
    self.publish_model(START + 50_000_000)
    self.update()
    self.assertEqual(self.overlay._control_epoch, epoch)
    self.assertGreater(self.overlay._center_points.size, 0)

  def test_mici_readout_uses_accepted_preclip_term_and_allows_zero_accepted_term(self):
    self.overlay = self.module.BlendfulRenderer()
    self.state.sm['modelDataV2SP'].blendfulCurvature = 0.1
    receipt = self.state.sm['carControlSP']
    receipt.blendfulAppliedCurvature = -0.001
    self.update()
    self.assertLess(self.overlay._pull_display.x, 0)
    self.overlay.clear()
    receipt.blendfulAppliedCurvature = 0.0
    self.update()
    self.assertEqual(self.overlay._pull_display.x, 0)
    self.assertGreater(self.overlay._center_points.size, 0)
    self.draw_polygon.reset_mock()
    with patch.object(self.overlay, '_draw_readout'):
      self.overlay.draw(self.rect)
    self.draw_polygon.assert_called_once()

  def test_receipt_only_updates_refresh_mici_readout_without_new_projection(self):
    self.overlay = self.module.BlendfulRenderer()
    self.update()
    before = self.overlay._pull_display.x
    projected = self.renderer._map_line_to_polygon.call_count
    self.state.sm.logMonoTime['carControlSP'] = START + 10_000_000
    self.state.sm['carControlSP'].blendfulAppliedCurvature = 0.002
    self.clock.return_value = START + 10_000_001
    self.assertFalse(self.overlay.needs_update())
    with patch.object(self.overlay, '_draw_readout'):
      self.overlay.draw(self.rect)
    self.assertGreater(self.overlay._pull_display.x, before)
    self.assertEqual(self.renderer._map_line_to_polygon.call_count, projected)

  def test_repeating_model_control_ui_phase_keeps_only_opacity_while_receipt_is_in_flight(self):
    self.overlay = self.module.BlendfulRenderer()
    self.update()
    expected_alpha = self.overlay._alpha.x
    with patch.object(self.overlay, '_draw_readout') as readout:
      for frame in range(1, 13):
        # 20 Hz model at t=0, latest 100 Hz receipt at t=-10 ms, UI at t=0.
        stamp = START + frame * 50_000_000
        self.publish_pending_model(stamp)
        self.update()
        self.draw_polygon.reset_mock()
        readout.reset_mock()
        self.overlay.draw(self.rect)
        self.assertAlmostEqual(self.overlay._alpha.x, expected_alpha)
        self.assertEqual(self.overlay._center_points.size, 0)
        self.assertEqual(self.overlay._pull_display.x, 0)
        self.assertEqual(self.overlay._trust_display.x, 0)
        self.draw_polygon.assert_not_called()
        readout.assert_not_called()

        # The next control tick accepts it; the next 60 Hz UI tick can draw it.
        sm = self.state.sm
        sm['carControlSP'].blendfulModelMonoTime = stamp
        sm['carControlSP'].blendfulAppliedCurvature = 0.001 if frame % 2 else -0.001
        sm.logMonoTime['carControlSP'] = sm.logMonoTime['carControl'] = stamp + 10_000_000
        self.clock.return_value = stamp + 16_666_667
        self.assertTrue(self.overlay.needs_update())
        self.update()
        self.overlay.draw(self.rect)
        expected_alpha += (1 - expected_alpha) * self.overlay._alpha.alpha
        self.assertAlmostEqual(self.overlay._alpha.x, expected_alpha)
        self.assertEqual(self.draw_polygon.call_count, 1)
        readout.assert_called_once()
        self.assertEqual(np.sign(self.overlay._pull_display.x), 1 if frame % 2 else -1)

        # t=33 ms has another current receipt, but no new model/projection.
        projected = self.renderer._map_line_to_polygon.call_count
        sm.logMonoTime['carControlSP'] = sm.logMonoTime['carControl'] = stamp + 30_000_000
        self.clock.return_value = stamp + 33_333_333
        self.assertFalse(self.overlay.needs_update())
        self.overlay.draw(self.rect)
        self.assertEqual(self.renderer._map_line_to_polygon.call_count, projected)
    self.assertGreater(self.overlay._alpha.x, 0.85)
    self.assertEqual(self.renderer._map_line_to_polygon.call_count, 13)

  def test_in_flight_opacity_is_discarded_for_invalid_lines_withdrawal_or_new_session(self):
    for case in ('no_lines', 'nonfinite_line', 'side_change', 'withdrawn', 'new_epoch', 'stale_receipt', 'inactive'):
      with self.subTest(case=case):
        self.overlay = self.module.BlendfulRenderer()
        self.state.sm = model_messages()
        self.renderer._model_geometry_ns = START
        self.renderer._lane_lines = [SimpleNamespace(raw_points=np.array([line.x, line.y, line.z], dtype=np.float32).T)
                                     for line in self.state.sm['modelV2'].laneLines]
        self.clock.return_value = START + 1
        self.update()
        self.assertGreater(self.overlay._alpha.x, 0)
        self.publish_pending_model(START + 50_000_000)
        sm = self.state.sm
        if case == 'no_lines':
          sm['modelV2'].laneLineProbs[1:3] = [0.0, 0.0]
        elif case == 'nonfinite_line':
          self.renderer._lane_lines[1].raw_points[0, 1] = np.nan
        elif case == 'side_change':
          sm['modelV2'].laneLineProbs[2] = 0.0
        elif case == 'withdrawn':
          sm['carControlSP'].blendfulModelMonoTime = 0
        elif case == 'new_epoch':
          sm['carControlSP'].blendfulControlEpoch = sm['modelDataV2SP'].blendfulControlEpoch = START + 20_000_000
        elif case == 'stale_receipt':
          sm.logMonoTime['carControlSP'] = START - self.module.CONTROL_MAX_AGE_NS
        else:
          sm['carControl'].latActive = False
        self.update()
        self.draw_polygon.reset_mock()
        with patch.object(self.overlay, '_draw_readout') as readout:
          self.overlay.draw(self.rect)
          readout.assert_not_called()
        self.draw_polygon.assert_not_called()
        self.assert_cleared()
        self.assertEqual(self.overlay._opacity_deadline_ns, 0)

  def test_pending_publications_do_not_renew_the_opacity_wait_deadline(self):
    self.update()
    self.publish_pending_model(START + 50_000_000)
    self.update()
    deadline = self.overlay._opacity_deadline_ns
    self.assertEqual(deadline, START + 50_000_000 + self.module.RECEIPT_WAIT_MAX_AGE_NS)
    self.publish_pending_model(START + 90_000_000)
    self.update()
    self.assertEqual(self.overlay._opacity_deadline_ns, deadline)
    self.clock.return_value = deadline
    self.overlay.draw(self.rect)
    self.assertGreater(self.overlay._alpha.x, 0)
    self.clock.return_value = deadline + 1
    self.overlay.draw(self.rect)
    self.assert_cleared()

  def test_opacity_wait_does_not_extend_the_current_capture_deadline(self):
    sm = self.state.sm
    sm['carControlSP'].blendfulControlEpoch = sm['modelDataV2SP'].blendfulControlEpoch = START - 500_000_000
    self.publish_model(START, capture_ns=START - 245_000_000)
    self.update()
    self.publish_pending_model(START + 1_000_000, capture_ns=START - 244_000_000)
    self.update()
    deadline = sm['modelV2'].timestampEof + self.module.MESSAGE_MAX_AGE_NS
    self.assertEqual(self.overlay._opacity_deadline_ns, deadline)
    self.clock.return_value = deadline
    self.overlay.draw(self.rect)
    self.assertGreater(self.overlay._alpha.x, 0)
    self.assertEqual(self.overlay._center_points.size, 0)
    self.clock.return_value = deadline + 1
    self.overlay.draw(self.rect)
    self.assert_cleared()

  def test_zero_weight_nonfinite_side_is_masked_and_zero_variance_is_bounded(self):
    output = self.state.sm['modelDataV2SP']
    output.blendfulRightProb = 0.0
    self.renderer._lane_lines[2].raw_points[:] = np.nan
    self.state.sm['modelV2'].laneLineStds[2] = np.nan
    self.state.sm['modelV2'].laneLineStds[1] = 0.0
    with np.errstate(all='raise'):
      self.update()
    self.assertTrue(np.isfinite(self.overlay._center_points).all())
    self.assertGreater(self.overlay._center_points.size, 0)
    np.testing.assert_allclose(self.renderer._map_line_to_polygon.call_args.args[0][:, 1], 0.05, atol=1e-6)

  def test_current_probability_masks_old_confidence_and_uses_fixed_single_line_width(self):
    output, model = self.state.sm['modelDataV2SP'], self.state.sm['modelV2']
    model.laneLineProbs[1] = model.laneLineProbs[2] = 0.3
    self.update()
    self.assertEqual(self.overlay._side_mask, (True, True))
    np.testing.assert_allclose(self.renderer._map_line_to_polygon.call_args.args[0][:, 1], 0)
    # Both diagnostic probabilities remain high, but only the current left side qualifies.
    output.blendfulLaneWidth = 4.5
    self.renderer._lane_lines[2].raw_points[:] = np.nan
    for probability in (0.0, 0.299, np.nan):
      with self.subTest(probability=probability), np.errstate(all='raise'):
        model.laneLineProbs[2] = probability
        self.update()
        self.assertGreater(self.overlay._center_points.size, 0)
        np.testing.assert_allclose(self.renderer._map_line_to_polygon.call_args.args[0][:, 1], 0.05, atol=1e-6)
    model.laneLineProbs[1] = 0.299
    self.update()
    self.assert_cleared()

  def test_readout_filters_do_not_inherit_correction_from_a_lost_side(self):
    self.overlay = self.module.BlendfulRenderer()
    self.update()
    self.assertGreater(self.overlay._pull_display.x, 0)
    self.state.sm['modelV2'].laneLineProbs[2] = 0.0
    self.state.sm['modelDataV2SP'].blendfulCurvature = 0.0
    self.state.sm['carControlSP'].blendfulAppliedCurvature = 0.0
    self.publish_model(START + 50_000_000)
    self.update()
    self.assertEqual(self.overlay._side_mask, (True, False))
    self.assertEqual(self.overlay._pull_display.x, 0)

  def test_current_geometry_loss_or_zero_trust_never_reuses_old_cyan_line(self):
    for case in ('zero_prob', 'zero_trust', 'nan_trust', 'nan_weighted_line', 'empty_line', 'missing_line',
                 'missing_std', 'missing_probs', 'nan_width'):
      with self.subTest(case=case):
        self.state.sm = model_messages()
        self.renderer._lane_lines = [SimpleNamespace(raw_points=np.array([line.x, line.y, line.z], dtype=np.float32).T)
                                     for line in self.state.sm['modelV2'].laneLines]
        self.update()
        self.assertGreater(self.overlay._center_points.size, 0)
        output, model = self.state.sm['modelDataV2SP'], self.state.sm['modelV2']
        if case == 'zero_prob':
          output.blendfulLeftProb = output.blendfulRightProb = 0.0
        elif case in ('zero_trust', 'nan_trust'):
          output.blendfulTrust = 0.0 if case == 'zero_trust' else np.nan
        elif case == 'nan_weighted_line':
          self.renderer._lane_lines[1].raw_points[0, 1] = np.nan
        elif case == 'empty_line':
          self.renderer._lane_lines[1].raw_points = np.empty((0, 3))
        elif case == 'missing_line':
          model.laneLines = []
        elif case == 'missing_std':
          model.laneLineStds = []
        elif case == 'missing_probs':
          model.laneLineProbs = []
        else:
          output.blendfulLeftProb = output.blendfulRightProb = 0.5
          output.blendfulLaneWidth = np.nan
        with np.errstate(all='raise'):
          self.update()
        self.assert_cleared()

  def test_alpha_advances_once_per_sp_output_and_retains_no_old_geometry(self):
    self.update()
    alpha = self.overlay._alpha.x
    self.renderer._lane_lines[1].raw_points[:, 1] += 0.2
    self.renderer._lane_lines[2].raw_points[:, 1] += 0.2
    self.update()
    self.assertEqual(self.overlay._alpha.x, alpha)
    np.testing.assert_allclose(self.overlay._center_points[:, 1], 0.2, atol=1e-6)

  def test_both_layouts_accept_late_receipts_and_keep_projection_caches_while_overlay_expires(self):
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
        self.state.sm['carControlSP'].blendfulModelMonoTime = 0
        renderer._render(self.rect)
        self.assertEqual(renderer.blendful._center_points.size, 0)

        # controlsd's receipt arrives after the model/SP update, with no new geometry.
        self.state.sm.updated = dict.fromkeys(self.state.sm, False)
        self.state.sm.updated['carControlSP'] = True
        self.state.sm['carControlSP'].blendfulModelMonoTime = START
        self.state.sm.logMonoTime['carControlSP'] = START + 10_000_000
        self.clock.return_value = START + 10_000_001
        renderer._render(self.rect)
        self.assertGreater(renderer.blendful._center_points.size, 0)
        self.assertEqual(renderer.blendful._readout, module is mici)
        projected = renderer._map_line_to_polygon.call_count

        # The following ordinary 100 Hz receipt reuses that projection.
        self.state.sm.logMonoTime['carControlSP'] = START + 20_000_000
        self.clock.return_value = START + 20_000_001
        renderer._render(self.rect)
        self.assertEqual(renderer._map_line_to_polygon.call_count, projected)
        self.state.sm.updated = dict.fromkeys(self.state.sm, False)
        self.clock.return_value = START + self.module.MESSAGE_MAX_AGE_NS + 1
        renderer._render(self.rect)
        self.assertEqual(renderer._map_line_to_polygon.call_count, projected)
        self.assertEqual(renderer.blendful._center_points.size, 0)
        self.assertEqual(renderer._draw_lane_lines.call_count, 4)
        self.assertEqual(renderer._draw_path.call_count, 4)

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
