from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openpilot.common.params import Params, ParamKeyFlag, ParamKeyType
from openpilot.common.sound_mode import SoundMode, get_sound_mode
from openpilot.selfdrive.ui.tests.test_sunnylink_removed import HeadlessUITestCase


class TestSoundSettings(HeadlessUITestCase):
  @contextmanager
  def settings(self, variant):
    from openpilot.selfdrive.ui.layouts.settings import toggles
    from openpilot.selfdrive.ui.mici.layouts.settings import toggles as mici

    module = toggles if variant == 'big' else mici
    state = SimpleNamespace(params=Params(), CP=None, engaged=False, started=False,
                            update_params=Mock(), add_engaged_transition_callback=Mock(), is_offroad=lambda: True)
    with ExitStack() as stack:
      stack.enter_context(patch.object(module, 'ui_state', state))
      if variant == 'big':
        def multiple_buttons(*args, **kwargs):
          item = Mock()
          item.title = args[0]
          item.action_item.buttons = kwargs['buttons']
          item.action_item.callback = kwargs['callback']
          item.action_item.selected_button = kwargs['selected_index']
          item.action_item.set_selected_button.side_effect = lambda index: setattr(item.action_item, 'selected_button', index)
          return item

        stack.enter_context(patch.object(module, 'tr', side_effect=lambda text: text))
        stack.enter_context(patch.object(module, 'multiple_button_item', side_effect=multiple_buttons))
        stack.enter_context(patch.object(module, 'toggle_item', side_effect=lambda *args, **kwargs: Mock()))
        stack.enter_context(patch.object(module, 'Scroller'))
        stack.enter_context(patch.object(module.Widget, 'show_event'))
        layout = module.TogglesLayout()
        def selected():
          return layout._sound_setting.action_item.selected_button

        select = layout._sound_setting.action_item.callback
      else:
        def initialize(layout):
          layout._scroller = Mock()

        def multi_toggle(title, options, select_callback):
          item = Mock()
          item.title, item._options, item._select_callback = title, options, select_callback
          item.set_value.side_effect = lambda value: setattr(item, 'value', value)
          return item

        stack.enter_context(patch.object(module.NavScroller, '__init__', initialize))
        stack.enter_context(patch.object(module.NavScroller, 'show_event'))
        for name in ('BigMultiParamToggle', 'BigToggle', 'BigParamControl'):
          stack.enter_context(patch.object(module, name, side_effect=lambda *args, **kwargs: Mock()))
        stack.enter_context(patch.object(module, 'BigMultiToggle', side_effect=multi_toggle))
        layout = module.TogglesLayoutMici()
        def selected():
          return layout._sound_options.index(layout._sound_toggle.value)

        def select(index):
          layout._sound_toggle._select_callback(layout._sound_options[index])
      yield layout, selected, select

  def test_sound_follows_radar_with_three_choices_and_default_all(self):
    Params().remove('SoundMode')
    for variant in ('big', 'mici'):
      with self.subTest(variant=variant), self.settings(variant) as (layout, selected, _):
        if variant == 'big':
          keys = list(layout._toggles)
          self.assertEqual(keys[keys.index('RadarSignal') + 1], 'SoundMode')
          self.assertEqual(layout._sound_setting.title(), 'Sound')
          self.assertEqual([label() for label in layout._sound_setting.action_item.buttons], ['None', 'Warning', 'All'])
        else:
          items = layout._scroller.add_widgets.call_args.args[0]
          self.assertIs(items[items.index(layout._radar_signal_btn) + 1], layout._sound_toggle)
          self.assertEqual(layout._sound_toggle.title, 'Sound')
          self.assertEqual(layout._sound_toggle._options, ['None', 'Warning', 'All'])
        self.assertEqual(selected(), SoundMode.ALL)

  def test_selection_persists_and_page_refreshes_without_restart(self):
    params = Params()
    params.remove('OnroadCycleRequested')
    for variant in ('big', 'mici'):
      with self.subTest(variant=variant), self.settings(variant) as (layout, selected, select):
        for mode in SoundMode:
          select(int(mode))
          self.assertEqual(Params().get('SoundMode'), mode)
          layout.show_event()
          self.assertEqual(selected(), mode)

          external = (int(mode) + 1) % len(SoundMode)
          params.put('SoundMode', external, block=True)
          layout.show_event()
          self.assertEqual(selected(), external)
        self.assertFalse(params.get_bool('OnroadCycleRequested'))

  def test_native_key_default_retention_and_invalid_value(self):
    params = Params()
    params.remove('SoundMode')
    self.assertEqual(params.get_type('SoundMode'), ParamKeyType.INT)
    self.assertEqual(params.get('SoundMode', return_default=True), SoundMode.ALL)
    self.assertIn(b'SoundMode', params.all_keys(ParamKeyFlag.BACKUP))
    for mode in SoundMode:
      params.put('SoundMode', int(mode), block=True)
      params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
      params.clear_all(ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION)
      self.assertEqual(get_sound_mode(Params()), mode)
    params.put('SoundMode', 99, block=True)
    self.assertEqual(get_sound_mode(params), SoundMode.ALL)
    for variant in ('big', 'mici'):
      with self.subTest(variant=variant), self.settings(variant) as (_, selected, _):
        self.assertEqual(selected(), SoundMode.ALL)
