import subprocess
import threading
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import pyray as rl

from openpilot.selfdrive.ui.tests.test_sunnylink_removed import HeadlessUITestCase


class DeferredThread:
  """An explicit completion boundary; any production wait fails the test."""
  def __init__(self, *, target, name, daemon):
    self.target, self.name, self.daemon = target, name, daemon
    self.running = False

  def start(self):
    self.running = True

  def is_alive(self):
    return self.running

  def complete(self):
    self.target()
    self.running = False

  def join(self, *args, **kwargs):
    raise AssertionError("the UI must not wait for an IP lookup")


def reference_metric(module, sidebar, rect, metric, y):
  """Uncached drawing contract from 07274acc, independent of candidate layout."""
  metric_rect = rl.Rectangle(rect.x + module.METRIC_MARGIN, y, module.METRIC_WIDTH, module.METRIC_HEIGHT)
  edge_rect = rl.Rectangle(metric_rect.x + 4, metric_rect.y + 4, 100, 118)
  rl.begin_scissor_mode(int(metric_rect.x + 4), int(metric_rect.y), 18, int(metric_rect.height))
  rl.draw_rectangle_rounded(edge_rect, 0.3, 10, metric.color)
  rl.end_scissor_mode()
  rl.draw_rectangle_rounded_lines_ex(metric_rect, 0.3, 10, 2, module.Colors.METRIC_BORDER)
  labels = [module.tr(metric.label), module.tr(metric.value)]
  text_sizes = [module.measure_text_cached(sidebar._font_bold, text, module.FONT_SIZE) for text in labels]
  total_h = sum(size.y for size in text_sizes)
  text_y = metric_rect.y + (metric_rect.height / 2 - total_h / 2)
  for text, size in zip(labels, text_sizes):
    position = rl.Vector2(metric_rect.x + 22 + (metric_rect.width - 22 - size.x) / 2, text_y)
    rl.draw_text_ex(sidebar._font_bold, text, position, module.FONT_SIZE, 0, module.Colors.WHITE)
    text_y += size.y


@contextmanager
def capture_metric_draws():
  draws = []

  def rectangle(rect):
    return rect.x, rect.y, rect.width, rect.height

  def color(rgba):
    return rgba.r, rgba.g, rgba.b, rgba.a

  def rounded(rect, roundness, segments, rgba):
    draws.append(("edge", rectangle(rect), roundness, segments, color(rgba)))

  def border(rect, roundness, segments, thickness, rgba):
    draws.append(("border", rectangle(rect), roundness, segments, thickness, color(rgba)))

  def text(font, value, position, size, spacing, rgba):
    draws.append(("text", id(font), value, (position.x, position.y), size, spacing, color(rgba)))

  with ExitStack() as stack:
    stack.enter_context(patch.object(rl, "begin_scissor_mode", side_effect=lambda *args: draws.append(("scissor", args))))
    stack.enter_context(patch.object(rl, "end_scissor_mode", side_effect=lambda: draws.append(("end_scissor",))))
    stack.enter_context(patch.object(rl, "draw_rectangle_rounded", side_effect=rounded))
    stack.enter_context(patch.object(rl, "draw_rectangle_rounded_lines_ex", side_effect=border))
    stack.enter_context(patch.object(rl, "draw_text_ex", side_effect=text))
    yield draws


class SidebarTestCase(HeadlessUITestCase):
  def setUp(self):
    super().setUp()
    from openpilot.selfdrive.ui.layouts import sidebar
    self.module = sidebar
    font = rl.Font()
    font.texture.id = 10
    with patch.object(sidebar.gui_app, "texture", return_value=SimpleNamespace(width=180, height=180)), \
         patch.object(sidebar.gui_app, "font", return_value=font):
      self.sidebar = sidebar.Sidebar()


class TestSidebarIP(SidebarTestCase):
  def setUp(self):
    super().setUp()
    self.clock = self.enterContext(patch.object(self.module.time, "monotonic", return_value=100.0))
    self.demote = self.enterContext(patch.object(self.module, "drop_realtime"))
    self.affinity = self.enterContext(patch.object(self.module, "set_core_affinity"))
    self.query = self.enterContext(patch.object(self.module.subprocess, "check_output", return_value="192.168.0.54 100.64.0.1\n"))

  def test_single_pending_request_and_five_second_request_cadence(self):
    with patch.object(self.module, "Thread", side_effect=DeferredThread) as thread_factory:
      self.assertEqual(self.sidebar._get_ip(), "")
      first = self.sidebar._ip_thread
      self.assertTrue(first.daemon)
      for now in (100.1, 101.0, 104.999):
        self.clock.return_value = now
        self.assertEqual(self.sidebar._get_ip(), "")
      thread_factory.assert_called_once()
      self.query.assert_not_called()
      first.complete()
      self.assertEqual(self.sidebar._get_ip(), "192.168.0.54")
      thread_factory.assert_called_once()

      self.clock.return_value = 105.0
      self.assertEqual(self.sidebar._get_ip(), "192.168.0.54")
      self.assertEqual(thread_factory.call_count, 2)
      pending = self.sidebar._ip_thread
      for now in (106.0, 120.0, 140.0):
        self.clock.return_value = now
        self.assertEqual(self.sidebar._get_ip(), "192.168.0.54")
        self.assertIs(self.sidebar._ip_thread, pending)
      self.assertEqual(thread_factory.call_count, 2)
      self.query.assert_called_once_with(["hostname", "-I"], text=True, timeout=2)

  def test_real_slow_lookup_does_not_block_the_caller(self):
    entered, release = threading.Event(), threading.Event()
    query_threads = []

    def delayed_query(*args, **kwargs):
      query_threads.append(threading.get_ident())
      entered.set()
      if not release.wait(5):
        raise TimeoutError("test cleanup did not release lookup")
      return "192.168.0.55\n"

    self.query.side_effect = delayed_query
    worker = None
    try:
      self.assertEqual(self.sidebar._get_ip(), "")
      worker = self.sidebar._ip_thread
      self.assertTrue(entered.wait(1))
      self.assertFalse(release.is_set())
      self.clock.return_value = 110.0
      for _ in range(30):
        self.assertEqual(self.sidebar._get_ip(), "")
        self.assertIs(self.sidebar._ip_thread, worker)
      self.query.assert_called_once()
      self.assertNotEqual(query_threads, [threading.get_ident()])
    finally:
      release.set()
      if worker is not None:
        worker.join(2)  # Tests own cleanup; production never joins.
    self.assertFalse(worker.is_alive())
    self.assertEqual(self.sidebar._ip, "192.168.0.55")

  def test_demote_and_best_effort_affinity_precede_subprocess(self):
    order = Mock()
    for name, method in (("demote", self.demote), ("affinity", self.affinity), ("query", self.query)):
      order.attach_mock(method, name)
    self.affinity.side_effect = OSError("affinity unavailable")
    self.sidebar._refresh_ip()
    self.assertEqual(order.mock_calls, [call.demote(), call.affinity([0, 1, 2]), call.query(["hostname", "-I"], text=True, timeout=2)])
    self.assertEqual(self.sidebar._ip, "192.168.0.54")

  def test_demotion_failure_does_not_launch_optional_subprocess(self):
    self.sidebar._ip = "192.168.0.54"
    self.demote.side_effect = PermissionError("cannot drop scheduler policy")
    self.sidebar._refresh_ip()
    self.query.assert_not_called()
    self.affinity.assert_not_called()
    self.assertEqual(self.sidebar._ip, "")

  def test_empty_output_errors_timeout_and_recovery_replace_old_result(self):
    for result, expected in ((" \n\t", ""), (subprocess.TimeoutExpired(["hostname", "-I"], 2), ""),
                             (subprocess.CalledProcessError(1, ["hostname", "-I"]), ""),
                             (OSError("hostname unavailable"), ""), ("fe80::1 192.168.0.8\n", "fe80::1")):
      with self.subTest(result=result):
        self.sidebar._ip = "192.168.0.54"
        self.query.side_effect = result if isinstance(result, Exception) else None
        self.query.return_value = result
        self.sidebar._refresh_ip()
        self.assertEqual(self.sidebar._ip, expected)

  def test_thread_start_failure_is_throttled_and_can_recover(self):
    with patch.object(self.module, "Thread") as thread_factory:
      thread_factory.return_value.start.side_effect = RuntimeError("cannot start new thread")
      self.sidebar._ip = "192.168.0.54"
      self.assertEqual(self.sidebar._get_ip(), "")
      self.assertIsNone(self.sidebar._ip_thread)
      self.clock.return_value = 104.999
      self.assertEqual(self.sidebar._get_ip(), "")
      thread_factory.assert_called_once()
    self.clock.return_value = 105.0
    with patch.object(self.module, "Thread", side_effect=DeferredThread):
      self.assertEqual(self.sidebar._get_ip(), "")
      self.sidebar._ip_thread.complete()
      self.assertEqual(self.sidebar._get_ip(), "192.168.0.54")

  def test_network_text_uses_completed_ip_or_translated_fallback(self):
    self.sidebar._net_type = "Wi-Fi"
    rect = rl.Rectangle(10, 20, 300, 1080)
    with patch.object(self.module, "tr", return_value="无线网络"), patch.object(rl, "draw_circle"), \
         patch.object(rl, "draw_text_ex") as draw:
      for ip, expected in (("", "无线网络"), ("192.168.0.54", "192.168.0.54"), ("", "无线网络")):
        with patch.object(self.sidebar, "_get_ip", return_value=ip):
          self.sidebar._draw_network_indicator(rect)
        self.assertEqual(draw.call_args.args[1], expected)
        self.assertEqual((draw.call_args.args[2].x, draw.call_args.args[2].y), (68, 267))


class TestSidebarMetricLayout(SidebarTestCase):
  def setUp(self):
    super().setUp()
    from openpilot.system.ui.lib import text_measure
    self.text_measure = text_measure
    self.enterContext(patch.object(text_measure, "_cache", {}))
    self.enterContext(patch.object(self.module.multilang, "_language", "en"))
    self.translations = {}
    self.enterContext(patch.object(self.module.multilang, "_translations", self.translations))
    self.fallback = rl.Font()
    self.fallback.texture.id = 20
    self.fallback_font = self.enterContext(patch.object(self.module.gui_app, "fallback_font", return_value=self.fallback))
    self.enterContext(patch.object(rl, "measure_text_ex", side_effect=lambda font, text, size, spacing:
                                 rl.Vector2(len(text) * size * 0.41 + font.texture.id / 10, size * 0.73)))

  def assert_baseline_frame(self, rect):
    with capture_metric_draws() as expected:
      for metric, offset in ((self.sidebar._temp_status, 338), (self.sidebar._panda_status, 496), (self.sidebar._connect_status, 654)):
        reference_metric(self.module, self.sidebar, rect, metric, rect.y + offset)
    with capture_metric_draws() as actual:
      self.sidebar._draw_metrics(rect)
    self.assertEqual(actual, expected)
    self.assertEqual(len(actual), 18)
    return actual

  def test_exact_draws_for_values_colors_translations_rect_and_font_changes(self):
    rect = rl.Rectangle(0, 0, 300, 1080)
    self.assert_baseline_frame(rect)
    self.assert_baseline_frame(rect)
    self.sidebar._temp_status.update("TEMP", "58°C", self.module.Colors.DANGER)
    self.assert_baseline_frame(rect)
    layout = self.sidebar._temp_status._layout
    self.sidebar._temp_status.color = self.module.Colors.GOOD
    self.assert_baseline_frame(rect)
    self.assertIs(self.sidebar._temp_status._layout, layout)

    self.module.multilang._language = "zh-CHS"
    self.translations.update({"TEMP": "温度", "VI": "车辆接口", "ONLINE": "在线", "OFFLINE": "离线"})
    self.assert_baseline_frame(rect)
    self.translations["TEMP"] = "CPU 温度"
    self.assert_baseline_frame(rect)
    for rect in (rl.Rectangle(7.25, 11.75, 280, 960), rl.Rectangle(0, 0, 300, 1080)):
      self.assert_baseline_frame(rect)

    new_font = rl.Font()
    new_font.texture.id = 21
    self.fallback_font.return_value = new_font
    self.assert_baseline_frame(rect)
    new_font.texture.id = 22
    self.assert_baseline_frame(rect)
    self.module.multilang._language = "en"
    self.translations.clear()
    self.sidebar._font_bold = new_font
    self.assert_baseline_frame(rect)

    with patch.object(self.module, "FONT_SIZE", 41):
      self.assert_baseline_frame(rect)
    self.assert_baseline_frame(rect)
    with patch.object(self.module, "FONT_SCALE", 1.1), patch.object(self.text_measure, "FONT_SCALE", 1.1), \
         patch.object(self.text_measure, "_cache", {}):
      self.assert_baseline_frame(rect)
    self.assert_baseline_frame(rect)

  def test_current_three_layouts_avoid_repeated_measurement_and_keep_live_color(self):
    rect = rl.Rectangle(0, 0, 300, 1080)
    metrics = (self.sidebar._temp_status, self.sidebar._panda_status, self.sidebar._connect_status)
    with capture_metric_draws(), patch.object(self.module, "measure_text_cached", wraps=self.module.measure_text_cached) as measure:
      self.sidebar._draw_metrics(rect)
      self.assertEqual(measure.call_count, 6)
      layouts = tuple(metric._layout for metric in metrics)
      for _ in range(30):
        self.sidebar._draw_metrics(rect)
      self.assertEqual(measure.call_count, 6)
      self.assertTrue(all(metric._layout is layout for metric, layout in zip(metrics, layouts)))
      self.sidebar._temp_status.color = self.module.Colors.DANGER
      self.sidebar._draw_metrics(rect)
      self.assertEqual(measure.call_count, 6)
      for temp in range(30, 80):
        self.sidebar._temp_status.value = f"{temp}°C"
        self.sidebar._draw_metrics(rect)
        self.assertIs(metrics[1]._layout, layouts[1])
        self.assertIs(metrics[2]._layout, layouts[2])
      self.assertEqual(measure.call_count, 106)
      self.assertEqual(len({id(metric._layout) for metric in metrics}), 3)

  def test_hidden_sidebar_draws_nothing_and_resumes_with_current_layout(self):
    rect = rl.Rectangle(0, 0, 300, 1080)
    self.sidebar.set_visible(False)
    with patch.object(self.sidebar, "_update_state"), patch.object(self.sidebar, "_render") as render:
      self.sidebar.render(rect)
      render.assert_not_called()
    self.sidebar._temp_status.value = "75°C"
    self.sidebar.set_visible(True)
    self.assert_baseline_frame(rl.Rectangle(8, 12, 300, 1080))
