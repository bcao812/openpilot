from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pyray as rl

from openpilot.cereal import messaging
from openpilot.selfdrive.car.tests.test_honda_radar_observer import START
from openpilot.selfdrive.ui.tests.test_control_influence_render import executing_state
from openpilot.selfdrive.ui.tests.test_sunnylink_removed import HeadlessUITestCase


def receipt(**changes):
  return {'version': 1, 'sessionId': 'fixture', 'trackId': 75, 'modelMonoTime': START,
          'visionIndex': 0, 'kind': 'braking', 'strength': 0.25, **changes}


class TestBrakingTargetRendering(HeadlessUITestCase):
  def setUp(self):
    super().setUp()
    from openpilot.selfdrive.ui.sunnypilot.onroad import braking_target

    self.enterContext(patch.object(braking_target.gui_app, 'font', return_value=object()))
    self.enterContext(patch.object(braking_target, 'tr', side_effect=lambda text: text))
    self.clock = self.enterContext(patch.object(braking_target.time, 'monotonic_ns', return_value=START + 1))
    self.enterContext(patch.object(braking_target, 'measure_text_cached',
                                  side_effect=lambda font, text, size: rl.Vector2(len(text) * size * 0.55, size)))
    self.text = self.enterContext(patch.object(rl, 'draw_text_ex'))
    self.panel = self.enterContext(patch.object(rl, 'draw_rectangle_rounded'))
    self.enterContext(patch.object(rl, 'draw_rectangle_rounded_lines_ex'))
    self.bars = self.enterContext(patch.object(rl, 'draw_rectangle_rec'))
    self.renderer = braking_target.BrakingTargetRenderer()
    self.rect = rl.Rectangle(40, 30, 1280, 720)

  def drawn_text(self):
    return [item.args[1] for item in self.text.call_args_list]

  def reset_draws(self):
    self.text.reset_mock()
    self.panel.reset_mock()
    self.bars.reset_mock()

  def test_visual_identity_follows_receipt_even_when_in_second_display_slot(self):
    sm = executing_state()
    sm['radarState'].leadOne = {'present': True, 'dRel': 18.92, 'vRel': -1.07,
                                'evidence': {'version': 1, 'source': 'radar', 'sessionId': 'fixture', 'trackId': 75}}
    sm['radarState'].leadTwo = {'present': True, 'dRel': 20.31, 'vRel': -2.25,
                                'evidence': {'version': 1, 'source': 'vision', 'modelMonoTime': START, 'visionIndex': 0}}
    sm['controlsState'].motionState.objectInfluences = [receipt(sessionId='', trackId=0)]
    event = messaging.new_message('controlsState')
    event.valid, event.logMonoTime, event.controlsState = True, START, sm['controlsState']
    sm['controlsState'] = messaging.log_from_bytes(event.to_bytes()).controlsState

    self.renderer.render(self.rect, sm)

    self.assertEqual(self.drawn_text(), ['Target constraint', 'VISION 1', 'Braking', '25%'])

  def test_two_close_targets_keep_both_adopted_constraints(self):
    sm = executing_state()
    sm['radarState'].leadOne = {'present': True, 'dRel': 20.0}
    sm['radarState'].leadTwo = {'present': True, 'dRel': 20.0}
    sm['controlsState'].motionState.objectInfluences = [receipt(), receipt(sessionId='', trackId=0, strength=0.75)]

    self.renderer.render(self.rect, sm)

    self.assertEqual(self.drawn_text(), ['Target constraint', 'RADAR #75 + V1', 'Braking', '25%', 'VISION 1', 'Braking', '75%'])

  def test_far_target_and_disabled_lead_labels_do_not_hide_panel(self):
    from openpilot.selfdrive.ui.ui_state import ui_state

    sm = executing_state()
    sm['radarState'].leadOne = {'present': True, 'dRel': 65.0}
    sm['controlsState'].motionState.objectInfluences = [receipt(modelMonoTime=0, visionIndex=-1)]
    with patch.object(ui_state, 'chevron_metrics', 0):
      self.renderer.render(self.rect, sm)
      self.assertIn('RADAR #75', self.drawn_text())
      self.reset_draws()
      # Camera projection and radarState are not needed to display an owner receipt.
      del sm['radarState']
      self.renderer.render(self.rect, sm)
      self.assertIn('RADAR #75', self.drawn_text())

  def test_withdrawal_and_override_clear_on_next_draw_without_geometry_update(self):
    for case in ('expired', 'gas', 'brake', 'cancelled', 'long_inactive', 'withdrawn', 'wrong_plan', 'other_mode'):
      with self.subTest(case=case):
        sm = executing_state()
        self.clock.return_value = START + 1
        self.reset_draws()
        self.renderer.render(self.rect, sm)
        self.assertTrue(self.drawn_text())
        if case == 'expired':
          self.clock.return_value = START + 200_000_001
        elif case == 'gas':
          sm['carState'].gasPressed = True
        elif case == 'brake':
          sm['carState'].brakePressed = True
        elif case == 'cancelled':
          sm['selfdriveState'].enabled = False
        elif case == 'long_inactive':
          sm['carControl'].longActive = False
        elif case == 'withdrawn':
          sm['controlsState'].motionState.objectInfluences = []
        elif case == 'wrong_plan':
          sm['controlsState'].longitudinalPlanMonoTime = START - 1
        else:
          sm['controlsState'].motionState.planningMode = 1
        self.reset_draws()

        self.renderer.render(self.rect, sm)

        self.text.assert_not_called()
        self.panel.assert_not_called()
        self.bars.assert_not_called()

  def test_only_object_braking_and_hold_receipts_are_shown(self):
    for kind in ('follow', 'release', 'caution'):
      with self.subTest(kind=kind):
        sm = executing_state()
        sm['controlsState'].motionState.objectInfluences = [receipt(kind=kind, strength=0.15)]
        self.reset_draws()
        self.renderer.render(self.rect, sm)
        self.text.assert_not_called()
        self.panel.assert_not_called()

    sm = executing_state()
    sm['controlsState'].motionState.phase = 'hold'
    sm['controlsState'].motionState.objectInfluences = [receipt(kind='hold', strength=1.0)]
    self.renderer.render(self.rect, sm)
    self.assertEqual(self.drawn_text(), ['Target constraint', 'RADAR #75 + V1', 'Hold', '100%'])
    color = self.bars.call_args.args[-1]
    self.assertEqual((color.r, color.g, color.b), (195, 20, 35))
    self.assertTrue(all((item.args[-1].r, item.args[-1].g, item.args[-1].b) == (255, 255, 255)
                        for item in self.text.call_args_list))

  def test_full_identity_deduplication_and_overflow_are_explicit(self):
    sm = executing_state()
    sm['controlsState'].motionState.objectInfluences = [
      receipt(), receipt(strength=0.75), receipt(sessionId='new-session'), receipt(sessionId='', trackId=0),
    ]
    self.renderer.render(self.rect, sm)
    self.assertEqual(self.drawn_text(), ['Target constraint', '+1 more',
                                       'RADAR #75 + V1', 'Braking', '75%', 'RADAR #75 + V1', 'Braking', '25%'])

    self.reset_draws()
    sm['controlsState'].motionState.objectInfluences = [
      receipt(sessionId='', trackId=0), receipt(sessionId='', trackId=0, modelMonoTime=START - 1),
    ]
    self.renderer.render(self.rect, sm)
    self.assertEqual(self.drawn_text().count('VISION 1'), 2)

  def test_two_rows_fit_offset_view_and_use_qualitative_color_scale(self):
    sm = executing_state()
    sm['controlsState'].motionState.objectInfluences = [receipt(), receipt(sessionId='', trackId=0, visionIndex=1, strength=1.0)]
    self.renderer.render(self.rect, sm)

    panel = self.panel.call_args.args[0]
    self.assertGreaterEqual(panel.x, self.rect.x)
    self.assertGreaterEqual(panel.y, self.rect.y + 300)
    self.assertLessEqual(panel.x + panel.width, self.rect.x + self.rect.width / 2)
    self.assertLessEqual(panel.y + panel.height, self.rect.y + self.rect.height)
    for item in self.text.call_args_list:
      _, text, pos, size, _, _ = item.args
      self.assertGreaterEqual(pos.x, panel.x)
      self.assertGreaterEqual(pos.y, panel.y)
      self.assertLessEqual(pos.x + len(text) * size * 0.55, panel.x + panel.width)
      self.assertLessEqual(pos.y + size, panel.y + panel.height)
    fill_colors = [(item.args[-1].r, item.args[-1].g, item.args[-1].b) for item in self.bars.call_args_list[1::2]]
    self.assertEqual(fill_colors, [(240, 121, 125), (195, 20, 35)])
    self.assertIn('VISION 2', self.drawn_text())

  def test_fused_identity_and_chinese_labels_fit_beside_action(self):
    from openpilot.selfdrive.ui.sunnypilot.onroad import braking_target
    from openpilot.system.ui.lib.multilang import TRANSLATIONS_DIR, load_translations

    sm = executing_state()
    sm['controlsState'].motionState.objectInfluences = [receipt(trackId=4294967295)]
    for language in ('en', 'zh-CHS', 'zh-CHT'):
      with self.subTest(language=language):
        translations, _ = load_translations(TRANSLATIONS_DIR / f'app_{language}.po')
        self.reset_draws()
        with patch.object(braking_target, 'tr', side_effect=lambda text, translations=translations: translations.get(text, text) or text):
          self.renderer.render(self.rect, sm)
        source, action = (item.args for item in self.text.call_args_list[1:3])
        self.assertIn('4294967295', source[1])
        self.assertLess(source[2].x + len(source[1]) * source[3] * 0.55, action[2].x)
        self.assertEqual(self.drawn_text()[-1], '25%')
        if language == 'zh-CHS':
          self.assertEqual(self.drawn_text(), ['目标约束强度', '雷达 #4294967295 + 视觉 1', '制动', '25%'])
        elif language == 'zh-CHT':
          self.assertEqual(self.drawn_text(), ['目標約束強度', '雷達 #4294967295 + 視覺 1', '制動', '25%'])

  def test_hud_calls_panel_on_every_draw(self):
    from openpilot.selfdrive.ui.sunnypilot.onroad import hud_renderer

    hud = hud_renderer.HudRendererSP.__new__(hud_renderer.HudRendererSP)
    for name in ('developer_ui', 'speed_limit_renderer', 'smart_cruise_control_renderer', 'braking_target_renderer',
                 'turn_signal_controller', 'circular_alerts_renderer', 'rocket_fuel'):
      setattr(hud, name, Mock())
    sm = executing_state()
    with patch.object(hud_renderer.HudRenderer, '_render'), patch.object(hud_renderer.gui_app, 'mark_phase'), \
         patch.object(hud_renderer, 'ui_state', SimpleNamespace(torque_bar=False, sm=sm)):
      hud._render(self.rect)
      hud._render(self.rect)
    self.assertEqual(hud.braking_target_renderer.render.call_args_list, [call(self.rect, sm), call(self.rect, sm)])
