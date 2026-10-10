"""Run the actual render generator with fake graphics, without initializing a GUI."""
import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call


def load_render(globals_dict):
  # Importing application creates gui_app, fonts and graphics/hardware dependencies.
  # Compile only its unmodified method so this test still executes the production
  # screen-off branch, yield/resume sequence and frame completion.
  path = Path(__file__).resolve().parents[1] / "application.py"
  module = ast.parse(path.read_text())
  app_class = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "GuiApplication")
  method = next(node for node in app_class.body if isinstance(node, ast.FunctionDef) and node.name == "render")
  namespace = dict(globals_dict)
  exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
  return namespace["render"]


class TestApplicationFrameTimer(unittest.TestCase):
  def test_onroad_phase_marks_keep_the_timer_running(self):
    from openpilot.sunnypilot.common.phase_timer import PhaseTimer

    application = Path(__file__).resolve().parents[1] / "application.py"
    source = ast.parse(application.read_text())
    phases = next(ast.literal_eval(node.value) for node in source.body
                  if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "FRAME_PHASES"
                                                        for target in node.targets))
    ui = Path(__file__).resolve().parents[4] / "selfdrive/ui"
    sources = (application, ui / "onroad/augmented_road_view.py", ui / "onroad/model_renderer.py",
               ui / "sunnypilot/onroad/hud_renderer.py")
    marks = [node.args[0].value for path in sources for node in ast.walk(ast.parse(path.read_text()))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr in ("mark", "mark_phase") and node.args
             and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)]
    timer = PhaseTimer("unused-test-marker", "test.ui.frame_timing", phases, 1800)
    try:
      timer.begin()
      for phase in marks:
        timer.mark(phase)
      timer.end_frame()
      self.assertTrue(timer._enabled, timer._failure)
      self.assertEqual(timer.i, 1)
      self.assertEqual(sum(map(sum, timer.wall.values())), len(marks))
      self.assertEqual(sum(timer.wall["honda_radar"]), 1)
    finally:
      timer.stop()

  def test_only_rendered_frames_contribute_phase_samples(self):
    for pc in (False, True):
      with self.subTest(pc=pc):
        timer = Mock(PERIOD_FRAMES=1800)
        mouse = Mock()
        event = object()
        mouse.get_events.return_value = [event]
        clock = Mock()
        clock.monotonic.return_value = 1.0
        graphics = Mock()
        graphics.window_should_close.return_value = False
        graphics.get_frame_time.return_value = 1 / 60
        render = load_render({"rl": graphics, "time": clock, "PC": pc, "RECORD": False,
                              "cloudlog": Mock(), "FRAME_TIMER_MARKER": "test"})
        app = SimpleNamespace(
          _profile_render_frames=0, _frame_timer=timer, _window_close_requested=False,
          _should_render=False, _mouse=mouse, _target_fps=60, _render_texture=None,
          _scale=1.0, _nav_stack_ticks=[], _nav_stack=[], _nav_stack_widgets_to_render=1,
          _show_fps=False, _show_touches=False, _show_mouse_coords=False, _grid_size=0,
          _monitor_fps=Mock(), _frame=0,
        )
        frames = render(app)
        try:
          self.assertEqual(next(frames), (False, 0.0, 0.0))
          self.assertEqual(next(frames), (False, 0.0, 0.0))
          timer.begin.assert_not_called()
          timer.mark.assert_not_called()
          timer.end_frame.assert_not_called()

          app._should_render = True
          self.assertEqual(next(frames), (True, 1 / 60, 0.0))
          timer.begin.assert_called_once()
          timer.end_frame.assert_not_called()  # frame finishes after caller's yielded work

          app._should_render = False
          self.assertEqual(next(frames), (False, 0.0, 0.0))
          timer.begin.assert_called_once()
          timer.end_frame.assert_called_once()
          self.assertEqual(timer.mark.call_args_list.count(call("mouse_drain")), 1)
          self.assertEqual(timer.mark.call_args_list.count(call("monitor_fps")), 1)
          self.assertEqual(app._frame, 1)
          self.assertIs(app._last_mouse_event, event)
          self.assertEqual(mouse.get_events.call_count, 4)
          self.assertEqual(mouse._handle_mouse_event.call_count, 4 if pc else 0)
          self.assertEqual(graphics.poll_input_events.call_count, 3 if pc else 0)
          self.assertEqual(clock.sleep.call_args_list, [call(1 / 60)] * 3)
          graphics.begin_drawing.assert_called_once()
          graphics.end_drawing.assert_called_once()
        finally:
          frames.close()


if __name__ == "__main__":
  unittest.main()
