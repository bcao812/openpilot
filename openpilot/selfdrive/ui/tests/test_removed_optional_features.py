from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pyray as rl

from openpilot.selfdrive.ui.tests.test_sunnylink_removed import HeadlessUITestCase


class TestSettingsWithoutOptionalFeatures(HeadlessUITestCase):
  @contextmanager
  def big_settings(self, extended=False):
    from openpilot.selfdrive.ui.layouts.settings import settings as core

    modules = [(core, ("DeviceLayout", "NetworkUI", "TogglesLayout", "SoftwareLayout", "DeveloperLayout"))]
    if extended:
      from openpilot.selfdrive.ui.sunnypilot.layouts.settings import settings as extension
      modules.append((extension, (
        "DeviceLayoutSP", "NetworkUISP", "TogglesLayout", "SoftwareLayoutSP", "ModelsLayout",
        "SteeringLayout", "CruiseLayout", "VisualsLayout", "DisplayLayout", "VehicleLayout", "DeveloperLayoutSP",
      )))

    with ExitStack() as stack:
      for module, constructors in modules:
        stack.enter_context(patch.object(module, "gui_app"))
        stack.enter_context(patch.object(module, "WifiManager"))
        for constructor in constructors:
          stack.enter_context(patch.object(module, constructor))
      if extended:
        stack.enter_context(patch.object(extension, "Scroller"))

      # Retained panels are isolated above. A forgotten Trips/Firehose constructor
      # must not create a session or start a polling worker behind the menu.
      session = stack.enter_context(patch("requests.Session"))
      start_thread = stack.enter_context(patch("threading.Thread.start"))
      layout = extension.SettingsLayoutSP() if extended else core.SettingsLayout()
      yield core, layout
      session.assert_not_called()
      start_thread.assert_not_called()

  def assert_panels_switch(self, layout):
    panel_types = list(layout._panels)
    for panel_type in panel_types[1:] + panel_types[:1]:
      previous = layout._panels[layout._current_panel].instance
      layout.set_current_panel(panel_type)
      self.assertEqual(layout._current_panel, panel_type)
      previous.hide_event.assert_called_once()
      layout._panels[panel_type].instance.show_event.assert_called_once()

  def test_core_settings_keep_all_remaining_panels_usable(self):
    with self.big_settings() as (_, layout):
      self.assertEqual([panel.name for panel in layout._panels], ["DEVICE", "NETWORK", "TOGGLES", "SOFTWARE", "DEVELOPER"])
      self.assert_panels_switch(layout)

  def test_extended_settings_preserve_panel_ids_used_by_main_callbacks(self):
    # MainLayout captures PanelType before the sunnypilot extension replaces it.
    from openpilot.selfdrive.ui.layouts import main

    with self.big_settings(extended=True) as (core, layout):
      for name, value in {"DEVICE": 0, "NETWORK": 1, "TOGGLES": 2, "SOFTWARE": 3, "DEVELOPER": 5}.items():
        with self.subTest(panel=name):
          captured_panel = getattr(main.PanelType, name)
          self.assertEqual(int(captured_panel), value)
          self.assertEqual(captured_panel, getattr(core.PanelType, name))
          self.assertEqual(next(panel.name for panel in layout._panels if panel == captured_panel), name)
      for name, value in {"MODELS": 6, "STEERING": 7, "CRUISE": 8, "VISUALS": 9, "DISPLAY": 10, "NAVIGATION": 12, "VEHICLE": 14}.items():
        with self.subTest(panel=name):
          self.assertEqual(int(getattr(core.PanelType, name)), value)
      self.assert_panels_switch(layout)

  def test_main_callbacks_keep_settings_and_bookmarks_connected(self):
    from openpilot.selfdrive.ui.layouts import home, main
    from openpilot.selfdrive.ui.widgets.setup import SetupWidget

    with self.big_settings(extended=True) as (core, settings), ExitStack() as stack:
      home_layout = Mock(spec=home.HomeLayout)
      home_layout._setup_widget = Mock(spec=SetupWidget)
      sidebar = Mock()
      for constructor, instance in (
        ("Sidebar", sidebar), ("HomeLayout", home_layout), ("SettingsLayout", settings),
        ("BodyLayout", Mock()), ("AugmentedRoadView", Mock()), ("OnboardingWindow", Mock(completed=True)),
      ):
        stack.enter_context(patch.object(main, constructor, return_value=instance))
      stack.enter_context(patch.object(main, "gui_app"))
      stack.enter_context(patch.object(main, "device"))
      stack.enter_context(patch.object(main, "ui_state", Mock(started=False, is_body=False)))
      publisher = stack.enter_context(patch.object(main.messaging, "PubMaster")).return_value
      messages = {service: object() for service in ("bookmarkButton", "userBookmark")}
      new_message = stack.enter_context(patch.object(main.messaging, "new_message", side_effect=lambda service, **kwargs: messages[service]))

      layout = main.MainLayout()
      home_layout.set_settings_callback.call_args.args[0]()
      self.assertEqual(settings._current_panel, core.PanelType.TOGGLES)
      self.assertEqual(layout._current_mode, main.MainState.SETTINGS)

      callbacks = sidebar.set_callbacks.call_args.kwargs
      callbacks["on_settings"]()
      self.assertEqual(settings._current_panel, core.PanelType.DEVICE)
      callbacks["open_settings"]()
      self.assertEqual(settings._current_panel, core.PanelType.TOGGLES)
      settings._close_callback()
      self.assertEqual(layout._current_mode, main.MainState.HOME)

      callbacks["on_flag"]()
      self.assertEqual(new_message.call_args_list, [call(service, valid=True) for service in messages])
      self.assertEqual(publisher.send.call_args_list, [call(service, message) for service, message in messages.items()])

  def test_mici_settings_keep_pairing_and_remaining_panel_callbacks(self):
    from openpilot.selfdrive.ui.mici.layouts.settings import device, settings
    from openpilot.selfdrive.ui.mici.widgets import button

    items = []
    scroller = SimpleNamespace(add_widgets=items.extend)
    buttons = {name: Mock() for name in ("toggles", "network", "device", "software", "developer")}
    app = Mock(target_fps=20)
    state = Mock()
    state.prime_state.is_paired.return_value = False
    state.params.get.return_value = "registered-device"

    def initialize(layout):
      layout._scroller = scroller

    with ExitStack() as stack:
      stack.enter_context(patch.object(settings.NavScroller, "__init__", initialize))
      stack.enter_context(patch.object(settings, "SettingsBigButton", side_effect=lambda name, *args: buttons[name]))
      panels = {}
      for name, constructor in (
        ("toggles", "TogglesLayoutMici"), ("network", "NetworkLayoutMici"), ("device", "DeviceLayoutMici"),
        ("software", "SoftwareLayoutMici"), ("developer", "DeveloperLayoutMici"),
      ):
        panels[name] = stack.enter_context(patch.object(settings, constructor)).return_value
      for module in (settings, device, button):
        stack.enter_context(patch.object(module, "gui_app", app))
      stack.enter_context(patch.object(button, "UnifiedLabel"))
      stack.enter_context(patch.object(device, "ui_state", state))
      stack.enter_context(patch.object(device, "system_time_valid", return_value=True))
      pairing = stack.enter_context(patch.object(device, "PairingDialog"))
      stack.enter_context(patch.object(rl, "get_time", return_value=0))
      start_thread = stack.enter_context(patch("threading.Thread.start"))

      settings.SettingsLayout()
      pair_button = items[4]
      self.assertIsInstance(pair_button, device.PairBigButton)
      self.assertEqual(items, [buttons["toggles"], buttons["network"], buttons["device"], buttons["software"], pair_button, buttons["developer"]])
      for name, panel in panels.items():
        with self.subTest(panel=name):
          app.push_widget.reset_mock()
          buttons[name].set_click_callback.call_args.args[0]()
          app.push_widget.assert_called_once_with(panel)

      app.push_widget.reset_mock()
      pair_button._handle_mouse_release(rl.Vector2(0, 0))
      pairing.assert_called_once_with()
      app.push_widget.assert_called_once_with(pairing.return_value)

      app.push_widget.reset_mock()
      state.prime_state.is_paired.return_value = True
      pair_button._handle_mouse_release(rl.Vector2(0, 0))
      app.push_widget.assert_not_called()
      start_thread.assert_not_called()


class TestSetupWithoutFirehose(HeadlessUITestCase):
  def test_registration_is_shown_only_until_pairing(self):
    from openpilot.selfdrive.ui.widgets import setup

    state = Mock()
    state.prime_state.is_paired.return_value = False
    with ExitStack() as stack:
      stack.enter_context(patch.object(setup, "ui_state", state))
      stack.enter_context(patch.object(setup, "gui_app"))
      stack.enter_context(patch.object(setup, "wrap_text", return_value=["Pair with comma connect"]))
      button = stack.enter_context(patch.object(setup, "Button"))
      draw_background = stack.enter_context(patch.object(rl, "draw_rectangle_rounded"))
      draw_text = stack.enter_context(patch.object(rl, "draw_text_ex"))
      draw_texture = stack.enter_context(patch.object(rl, "draw_texture_ex"))

      widget = setup.SetupWidget()
      widget._render(rl.Rectangle(0, 0, 750, 800))
      button.assert_called_once()
      button.return_value.render.assert_called_once()
      self.assertTrue(any(args.args[1] == setup.tr("Finish Setup") for args in draw_text.call_args_list))

      state.prime_state.is_paired.return_value = True
      for draw in (draw_background, draw_text, draw_texture, button.return_value.render):
        draw.reset_mock()
      widget._render(rl.Rectangle(0, 0, 750, 800))
      for draw in (draw_background, draw_text, draw_texture, button.return_value.render):
        draw.assert_not_called()

  def test_pairing_button_keeps_valid_time_guard_and_dialog(self):
    from openpilot.selfdrive.ui.widgets import setup

    for valid_time in (False, True):
      with self.subTest(valid_time=valid_time), ExitStack() as stack:
        stack.enter_context(patch.object(setup, "system_time_valid", return_value=valid_time))
        app = stack.enter_context(patch.object(setup, "gui_app"))
        button = stack.enter_context(patch.object(setup, "Button"))
        pairing = stack.enter_context(patch.object(setup, "PairingDialog"))
        alert = stack.enter_context(patch.object(setup, "alert_dialog"))

        setup.SetupWidget()
        button.call_args.args[1]()
        if valid_time:
          pairing.assert_called_once_with()
          alert.assert_not_called()
          app.push_widget.assert_called_once_with(pairing.return_value)
        else:
          pairing.assert_not_called()
          alert.assert_called_once()
          app.push_widget.assert_called_once_with(alert.return_value)
