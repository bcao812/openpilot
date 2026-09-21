import ast
import importlib
import itertools
import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pyray as rl

from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.version import terms_version, terms_version_sp, training_version
from openpilot.selfdrive.ui.translations.potools import parse_po


class HeadlessUITestCase(OpenpilotTestCase):
  def setUp(self):
    super().setUp()
    self.enterContext(patch.dict(os.environ, {"SCALE": "1"}))
    self.enterContext(patch.object(rl, "init_window", side_effect=AssertionError("Headless UI tests must not create a window")))


class TestOnboardingWithoutSunnylink(HeadlessUITestCase):
  @contextmanager
  def onboarding(self, variant: str, accepted_terms: bool, accepted_sp_terms: bool, trained: bool):
    params = Params()
    for key, value, enabled in (
      ("HasAcceptedTerms", terms_version, accepted_terms),
      ("HasAcceptedTermsSP", terms_version_sp, accepted_sp_terms),
      ("CompletedTrainingVersion", training_version, trained),
    ):
      if enabled:
        params.put(key, value, block=True)
      else:
        params.remove(key)

    module_path = "openpilot.selfdrive.ui.layouts.onboarding" if variant == "big" else "openpilot.selfdrive.ui.mici.layouts.onboarding"
    module = importlib.import_module(module_path)
    app = Mock(width=2160, height=1080)
    completed_callback = Mock()
    with ExitStack() as stack:
      stack.enter_context(patch.object(module, "ui_state", SimpleNamespace(params=params)))
      stack.enter_context(patch.object(module, "gui_app", app))
      stack.enter_context(patch.object(module, "TermsPage"))
      stack.enter_context(patch.object(module, "TrainingGuide"))
      stack.enter_context(patch.object(module.rl, "draw_rectangle_rec"))
      if variant == "big":
        stack.enter_context(patch.object(module, "DeclinePage"))
        window = module.OnboardingWindow()
      else:
        stack.enter_context(patch.object(module, "device"))
        window = module.OnboardingWindow(completed_callback)
      yield module, window, app, completed_callback

  def test_completion_requires_both_terms_and_training(self):
    for variant, accepted_terms, accepted_sp_terms, trained in itertools.product(("big", "mici"), (False, True), (False, True), (False, True)):
      with self.subTest(variant=variant, terms=accepted_terms, sp_terms=accepted_sp_terms, trained=trained):
        with self.onboarding(variant, accepted_terms, accepted_sp_terms, trained) as (_, window, _, _):
          self.assertEqual(window.completed, accepted_terms and accepted_sp_terms and trained)

  def test_current_terms_skip_directly_to_training(self):
    for variant in ("big", "mici"):
      with self.subTest(variant=variant), self.onboarding(variant, True, True, False) as (module, window, app, _):
        window.show_event()
        window._render(None)
        if variant == "big":
          self.assertEqual(window._state, module.OnboardingState.ONBOARDING)
          window._training_guide.render.assert_called_once_with(window._rect)
          app.pop_widget.assert_not_called()
        else:
          app.push_widget.assert_called_once_with(window._training_guide)
          window._render(None)
          app.push_widget.assert_called_once()
        self.assertFalse(window.completed)

  def test_missing_terms_do_not_start_training(self):
    for variant, accepted, trained in itertools.product(("big", "mici"), ((False, False), (True, False), (False, True)), (False, True)):
      with self.subTest(variant=variant, accepted=accepted, trained=trained):
        with self.onboarding(variant, *accepted, trained) as (module, window, app, _):
          window.show_event()
          window._render(None)
          window._terms.render.assert_called_once_with(window._rect)
          if variant == "big":
            self.assertEqual(window._state, module.OnboardingState.TERMS)
          window._training_guide.render.assert_not_called()
          app.push_widget.assert_not_called()
          app.pop_widget.assert_not_called()
          self.assertFalse(window.completed)

  def test_acceptance_preserves_terms_and_training(self):
    for variant, trained in itertools.product(("big", "mici"), (False, True)):
      with self.subTest(variant=variant, trained=trained), self.onboarding(variant, False, False, trained) as (module, window, app, finished):
        window._on_terms_accepted()
        self.assertEqual(Params().get("HasAcceptedTerms"), terms_version)
        self.assertEqual(Params().get("HasAcceptedTermsSP"), terms_version_sp)
        if trained:
          if variant == "big":
            app.pop_widget.assert_called_once()
          else:
            finished.assert_called_once()
          app.push_widget.assert_not_called()
        else:
          self.assertFalse(window.completed)
          if variant == "big":
            self.assertEqual(window._state, module.OnboardingState.ONBOARDING)
            app.pop_widget.assert_not_called()
          else:
            app.push_widget.assert_called_once_with(window._training_guide)
            finished.assert_not_called()
          window._on_completed_training()
          self.assertEqual(Params().get("CompletedTrainingVersion"), training_version)
          if variant == "mici":
            finished.assert_called_once()
            self.assertFalse(Params().get_bool("IsDriverViewEnabled"))
        self.assertTrue(window.completed)

  def test_declining_terms_keeps_onboarding_incomplete(self):
    for variant in ("big", "mici"):
      with self.subTest(variant=variant), self.onboarding(variant, False, False, False) as (module, window, app, finished):
        if variant == "big":
          window._on_terms_declined()
          self.assertEqual(window._state, module.OnboardingState.DECLINE)
          window._on_decline_back()
          self.assertEqual(window._state, module.OnboardingState.TERMS)
        else:
          window._on_uninstall()
          self.assertTrue(Params().get_bool("DoUninstall"))
        self.assertFalse(window.completed)
        self.assertIsNone(Params().get("HasAcceptedTerms"))
        self.assertIsNone(Params().get("HasAcceptedTermsSP"))
        app.push_widget.assert_not_called()
        finished.assert_not_called()


class TestUIWithoutSunnylink(HeadlessUITestCase):
  def test_big_settings_keep_other_panels(self):
    from openpilot.selfdrive.ui.sunnypilot.layouts.settings import settings

    constructors = (
      "DeviceLayoutSP", "NetworkUISP", "TogglesLayout", "SoftwareLayoutSP", "ModelsLayout",
      "SteeringLayout", "CruiseLayout", "VisualsLayout", "DisplayLayout", "VehicleLayout", "DeveloperLayoutSP",
    )
    with ExitStack() as stack:
      stack.enter_context(patch.object(settings.OP.SettingsLayout, "__init__", return_value=None))
      stack.enter_context(patch.object(settings, "Scroller"))
      stack.enter_context(patch.object(settings, "WifiManager"))
      for constructor in constructors:
        stack.enter_context(patch.object(settings, constructor))
      layout = settings.SettingsLayoutSP()
      self.assertEqual(
        [panel.name for panel in layout._panels],
        ["DEVICE", "NETWORK", "TOGGLES", "SOFTWARE", "MODELS", "STEERING", "CRUISE", "VISUALS",
         "DISPLAY", "VEHICLE", "DEVELOPER"],
      )
      self.assertNotIn("SUNNYLINK", settings.OP.PanelType.__members__)
      self.assertIs(layout._panels[settings.OP.PanelType.MODELS].instance, settings.ModelsLayout.return_value)

  def test_mici_settings_keep_models_and_existing_items(self):
    from openpilot.selfdrive.ui.sunnypilot.mici.layouts import settings

    base_items = [Mock(name=name) for name in ("toggles", "network", "device", "software", "pairing", "developer")]
    scroller = SimpleNamespace(_items=base_items.copy())
    scroller.add_widget = scroller._items.append

    def initialize(layout):
      layout._scroller = scroller

    with ExitStack() as stack:
      stack.enter_context(patch.object(settings.OP.SettingsLayout, "__init__", initialize))
      stack.enter_context(patch.object(settings, "DeviceLayoutMici"))
      stack.enter_context(patch.object(settings, "ModelsLayoutMici"))
      stack.enter_context(patch.object(settings, "SettingsBigButton"))
      stack.enter_context(patch.object(settings, "BigCircleButton", side_effect=lambda *args, **kwargs: Mock()))
      app = stack.enter_context(patch.object(settings, "gui_app"))
      layout = settings.SettingsLayoutSP()
      model_button = settings.SettingsBigButton.return_value
      self.assertEqual(
        layout._scroller._items,
        [layout._disable_offroad_btn, layout._enable_offroad_btn_onroad, base_items[0], model_button,
         *base_items[1:], layout._enable_offroad_btn_offroad],
      )
      settings.SettingsBigButton.assert_called_once()
      model_button.set_click_callback.call_args.args[0]()
      app.push_widget.assert_called_once_with(settings.ModelsLayoutMici.return_value)
      app.push_widget.reset_mock()
      base_items[2].set_click_callback.call_args.args[0]()
      app.push_widget.assert_called_once_with(settings.DeviceLayoutMici.return_value)

  def test_ui_state_has_no_sunnylink_polling_or_backup_subscription(self):
    from openpilot.selfdrive.ui.sunnypilot import ui_state as extension
    from openpilot.selfdrive.ui import ui_state as core

    with patch.object(extension, "ScreenSaverSP") as screensaver:
      state = extension.UIStateSP()
      self.assertNotIn("backupManagerSP", state.sm_services_ext)
      self.assertIn("modelManagerSP", state.sm_services_ext)
      self.assertIn("selfdriveStateSP", state.sm_services_ext)
      self.assertFalse(hasattr(state, "sunnylink_state"))
      self.assertFalse(hasattr(state, "sunnylink_enabled"))
      screensaver.assert_called_once_with(params=state.params)

    state = SimpleNamespace(
      prime_state=Mock(), _params_thread=object(), sm=Mock(),
      _update_state=Mock(), _update_status=Mock(), _update_chestnut_state=Mock(),
    )
    with patch.object(core, "device") as device:
      core.UIState.update(state)
      state.prime_state.start.assert_called_once()
      state.sm.update.assert_called_once_with(0)
      state._update_state.assert_called_once()
      state._update_status.assert_called_once()
      state._update_chestnut_state.assert_called_once()
      device.update.assert_called_once()

  def test_sidebar_keeps_chestnut_icons_and_three_metrics(self):
    from openpilot.selfdrive.ui.layouts import sidebar
    from openpilot.selfdrive.ui.sunnypilot.layouts import sidebar as extension
    from openpilot.selfdrive.ui.ui_state import ChestnutState

    icons = [SimpleNamespace(width=180, height=133) for _ in range(3)]
    with patch.object(extension.gui_app, "texture", side_effect=icons):
      state = extension.SidebarSP()
    default_icon = SimpleNamespace(width=180, height=180)
    for chestnut_state, expected_icon in (
      (ChestnutState.DISCONNECTED, default_icon),
      (ChestnutState.UNCOMPILED, icons[2]),
      (ChestnutState.FAILED, icons[2]),
      (ChestnutState.READY, icons[0]),
      (ChestnutState.ACTIVE, icons[0]),
      (ChestnutState.LOADING, icons[1]),
    ):
      with self.subTest(chestnut_state=chestnut_state):
        with patch.object(extension, "ui_state", SimpleNamespace(chestnut_state=chestnut_state)), patch.object(rl, "get_time", return_value=0):
          icon, _, opacity = state._get_home_icon(default_icon)
          self.assertIs(icon, expected_icon)
          self.assertAlmostEqual(opacity, 0.35 if chestnut_state == ChestnutState.LOADING else 1.0)

    state = SimpleNamespace(_temp_status=object(), _panda_status=object(), _connect_status=object(), _draw_metric=Mock())
    rect = rl.Rectangle(0, 10, 300, 1080)
    sidebar.Sidebar._draw_metrics(state, rect)
    self.assertEqual(
      state._draw_metric.call_args_list,
      [call(rect, state._temp_status, 348), call(rect, state._panda_status, 506), call(rect, state._connect_status, 664)],
    )

  def test_both_ui_entrypoints_import(self):
    from openpilot.selfdrive.ui import ui

    self.assertTrue(callable(ui.MainLayout))
    self.assertTrue(callable(ui.MiciMainLayout))

  def test_ui_sources_do_not_reference_removed_services(self):
    for relative_root in ("openpilot/selfdrive/ui", "openpilot/system/ui"):
      for source_path in (Path(BASEDIR) / relative_root).rglob("*.py"):
        if "tests" in source_path.parts:
          continue
        with self.subTest(path=source_path.relative_to(BASEDIR)):
          tree = ast.parse(source_path.read_text())
          for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
              values = [node.module or ""]
            elif isinstance(node, ast.Import):
              values = [alias.name for alias in node.names]
            elif isinstance(node, ast.Name):
              values = [node.id]
            elif isinstance(node, ast.Attribute):
              values = [node.attr]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
              values = [node.value]
            else:
              continue
            for value in values:
              self.assertNotIn("sunnylink", value.lower())
              self.assertNotEqual(value, "backupManagerSP")

  def test_translations_do_not_advertise_sunnylink(self):
    for path in (Path(BASEDIR) / "openpilot/selfdrive/ui/translations").glob("*.po*"):
      with self.subTest(path=path.name):
        _, entries = parse_po(path)
        for entry in entries:
          self.assertNotIn("sunnylink", entry.msgid.lower())
          self.assertFalse(any("sunnylink" in reference for reference in entry.source_refs))
