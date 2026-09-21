import queue
import unittest
from unittest.mock import Mock, patch

from openpilot.sunnypilot.common import phase_timer
from openpilot.sunnypilot.common.phase_timer import PhaseTimer


class TestPhaseTimer(unittest.TestCase):
  def setUp(self):
    self.wall = 100_000
    self.cpu = 100_000
    self.mono = (1 << 63) + 123
    self.clock_patches = [
      patch.object(phase_timer.time, "perf_counter_ns", side_effect=lambda: self.wall),
      patch.object(phase_timer.time, "thread_time_ns", side_effect=lambda: self.cpu),
      patch.object(phase_timer.time, "monotonic_ns", side_effect=lambda: self.mono),
      patch.object(phase_timer.time, "sleep"),
      patch.object(phase_timer, "drop_realtime"),
      patch.object(phase_timer, "set_core_affinity"),
      patch.object(phase_timer.cloudlog, "event"),
    ]
    self.wall_clock, self.cpu_clock, self.mono_clock, self.sleep, self.demote, self.affinity, self.event = [
      self.enterContext(patcher) for patcher in self.clock_patches
    ]

  @staticmethod
  def timer(phases=("work",), period=2):
    return PhaseTimer("/unused/marker", "test.phase_timing", phases, period)

  def frame(self, timer, marks=(("work", 1000, 1000),)):
    timer.begin()
    for phase, wall, cpu in marks:
      self.wall += wall
      self.cpu += cpu
      self.mono += wall
      timer.mark(phase)
    timer.end_frame()

  def emit(self, timer, item):
    timer._emit(*item)
    self.assertEqual(self.event.call_args.args, ("test.phase_timing",))
    return self.event.call_args.kwargs

  def test_conditional_and_repeated_phases_have_counts_and_per_frame_totals(self):
    timer = self.timer(("base", "optional", "repeated", "unused"), period=4)
    for i in range(4):
      marks = [("base", 2000, 1000)]
      if i == 0:
        marks.append(("optional", 8000, 4000))
      marks.extend([("repeated", 4000, 2000)] * (2 if i % 2 == 0 else 1))
      self.frame(timer, marks)
    record = self.emit(timer, timer._full.get_nowait())
    self.assertEqual(record["frames"], 4)
    self.assertEqual(record["stats_version"], 2)
    self.assertEqual(record["nominal_frames"], 4)
    self.assertEqual(record["cpu_us"]["base"]["count"], 4)
    self.assertEqual(record["cpu_us"]["optional"]["count"], 1)
    self.assertEqual(record["cpu_us"]["repeated"]["count"], 6)
    self.assertEqual(record["cpu_us"]["repeated"]["sum_us"], 12.0)
    self.assertIsNone(record["cpu_us"]["unused"])
    self.assertIsNone(record["wall_us"]["unused"])
    self.assertEqual(record["cpu_total_us"], 5.0)
    self.assertEqual(record["wall_total_us"], 10.0)
    self.assertEqual(record["cpu_us"]["optional"]["mean"], 4.0)

  def test_backpressure_preserves_frames_samples_bounds_and_bounded_retries(self):
    timer = self.timer()
    first_start = self.mono
    for _ in range(2):
      self.frame(timer)
    first_end = self.mono
    first, first_n = timer._full.get_nowait()  # The writer owns it but has not cleared it.
    second_start = self.mono
    free = timer._free
    timer._free = Mock(wraps=free)
    for _ in range(4):
      self.frame(timer)
    self.assertEqual(timer.i, 4)
    self.assertEqual(timer.dropped, 2)
    self.assertEqual(timer._free.get_nowait.call_count, 2)
    self.assertEqual(sum(first["wall"]["work"]), 2)
    first_record = self.emit(timer, (first, first_n))
    self.assertEqual(first_record["window_start_mono_ns"], str(first_start))
    self.assertEqual(first_record["window_end_mono_ns"], str(first_end))
    self.assertEqual(first_record["deferred_windows"], 0)
    timer._clear(first)
    timer._free.put_nowait(first)
    for _ in range(2):
      self.frame(timer)
    second = timer._full.get_nowait()
    self.assertEqual(second[1], 6)
    record = self.emit(timer, second)
    self.assertEqual(record["frames"], 6)
    self.assertEqual(record["cpu_us"]["work"]["count"], 6)
    self.assertEqual(record["cpu_us"]["work"]["sum_us"], 6.0)
    self.assertEqual(record["cpu_total_us"], 1.0)
    self.assertEqual(record["window_start_mono_ns"], str(second_start))
    self.assertEqual(record["window_end_mono_ns"], str(self.mono))
    self.assertEqual(record["deferred_windows"], 2)
    self.assertEqual(record["dropped"], 2)
    self.assertEqual(timer._free.get_nowait.call_count, 3)
    self.assertIs(timer._active, first)
    self.assertEqual(sum(timer.wall["work"]), 0)
    # Exactly one monotonic read per window start and per nominal rotation attempt.
    self.assertEqual(self.mono_clock.call_count, 6)

  def test_queued_buffer_is_never_reused_until_writer_returns_it(self):
    timer = self.timer()
    first = timer._active
    for _ in range(6):
      self.frame(timer)
    self.assertIsNot(timer._active, first)
    queued, frames = timer._full.get_nowait()
    self.assertIs(queued, first)
    self.assertEqual(frames, 2)
    self.assertEqual(sum(queued["cpu"]["work"]), 2)
    self.assertEqual(sum(timer.cpu["work"]), 4)
    with self.assertRaises(queue.Empty):
      timer._full.get_nowait()

  def test_linear_clock_buckets_and_long_stall_saturation(self):
    for duration, bucket in ((0, 0), (1, 1), (31, 31), (32, 32), (63, 63), (64, 64), (65, 64),
                             (127, 95), (128, 96), (129, 96), (10**12, PhaseTimer.MAX_BUCKET)):
      with self.subTest(duration=duration):
        timer = self.timer(period=1)
        self.frame(timer, [("work", duration, duration)])
        buf, frames = timer._full.get_nowait()
        self.assertEqual(frames, 1)
        self.assertEqual(buf["wall"]["work"][bucket], 1)
        self.assertEqual(buf["cpu"]["work"][bucket], 1)
        self.assertEqual(sum(buf["wall"]["work"]), 1)
        if duration == 0:
          self.assertEqual(self.emit(timer, (buf, frames))["cpu_total_us"], 0.0)

  def test_create_is_optional_and_contains_marker_or_construction_failures(self):
    with patch.object(phase_timer.os.path, "exists", return_value=False):
      self.assertIsNone(PhaseTimer.create("marker", "test.phase_timing", ("work",), 2))
    self.event.assert_not_called()
    with patch.object(phase_timer.os.path, "exists", side_effect=OSError("unavailable")):
      self.assertIsNone(PhaseTimer.create("marker", "test.phase_timing", ("work",), 2))
    self.assertEqual(self.event.call_args.kwargs["operation"], "create")
    with patch.object(phase_timer.os.path, "exists", return_value=True):
      for phases, period in ((("work",), 0), ((), 2), (("work", "work"), 2)):
        with self.subTest(phases=phases, period=period):
          self.assertIsNone(PhaseTimer.create("marker", "test.phase_timing", phases, period))
      with patch.object(PhaseTimer, "_new_set", side_effect=MemoryError):
        self.assertIsNone(PhaseTimer.create("marker", "test.phase_timing", ("work",), 2))

  def test_start_failure_disables_once_even_when_logger_fails(self):
    timer = self.timer()
    self.event.side_effect = OSError("logger unavailable")
    with patch.object(phase_timer.threading, "Thread") as thread:
      thread.return_value.start.side_effect = RuntimeError("no threads")
      timer.start()
      timer.start()
    self.assertFalse(timer._enabled)
    self.assertEqual(timer._failure, ("start", "RuntimeError"))
    self.assertEqual(thread.return_value.start.call_count, 1)
    self.assertEqual(self.event.call_count, 1)
    self.frame(timer)
    self.wall_clock.assert_not_called()

  def test_hot_path_failure_defers_diagnostic_and_subsequent_calls_are_noops(self):
    timer = self.timer()
    self.wall_clock.side_effect = OSError("clock unavailable")
    timer.begin()
    self.assertEqual(timer._failure, ("begin", "OSError"))
    self.event.assert_not_called()
    self.frame(timer)
    self.assertEqual(self.wall_clock.call_count, 1)
    self.assertEqual(timer.i, 0)
    timer._writer()
    self.assertEqual(self.event.call_args.args, ("test.phase_timing.disabled",))
    self.assertEqual(self.event.call_count, 1)
    timer._report_failure()
    self.assertEqual(self.event.call_count, 1)

  def test_mark_faults_disable_without_logging_on_calling_thread(self):
    for fault in ("missing_begin", "unknown_phase", "wall_regression", "cpu_regression"):
      with self.subTest(fault=fault):
        timer = self.timer()
        if fault != "missing_begin":
          timer.begin()
        if fault == "wall_regression":
          self.wall -= 1
        if fault == "cpu_regression":
          self.cpu -= 1
        timer.mark("unknown" if fault == "unknown_phase" else "work")
        self.assertFalse(timer._enabled)
        self.assertEqual(timer._failure[0], "mark")
    self.event.assert_not_called()

  def test_rotation_fault_disables_without_logging_or_reusing_buffers(self):
    timer = self.timer(period=1)
    timer._full = Mock()
    timer._full.put_nowait.side_effect = RuntimeError("handoff failed")
    self.frame(timer)
    self.assertFalse(timer._enabled)
    self.assertEqual(timer._failure, ("end_frame", "RuntimeError"))
    self.assertTrue(timer._free.empty())
    self.event.assert_not_called()

  def test_window_clock_regression_is_contained(self):
    timer = self.timer(period=1)
    timer.begin()
    timer.mark("work")
    self.mono -= 1
    timer.end_frame()
    self.assertEqual(timer._failure, ("end_frame", "ValueError"))
    self.event.assert_not_called()

  def test_stop_drains_completed_windows_without_flushing_or_clearing_active_frame(self):
    timer = self.timer()
    for _ in range(3):
      self.frame(timer)
    active = timer._active
    timer.stop()
    timer._writer()
    self.assertEqual(self.event.call_count, 1)
    self.assertEqual(self.event.call_args.kwargs["frames"], 2)
    self.assertIs(timer._active, active)
    self.assertEqual(sum(active["cpu"]["work"]), 1)
    self.assertEqual(timer.i, 1)
    self.frame(timer)
    self.assertEqual(timer.i, 1)

  def test_failed_demotion_disables_before_affinity_or_histogram_work(self):
    timer = self.timer()
    for _ in range(2):
      self.frame(timer)
    self.demote.side_effect = PermissionError("cannot demote")
    with patch.object(timer, "_emit") as emit, patch.object(timer, "_clear") as clear:
      timer._writer()
    emit.assert_not_called()
    clear.assert_not_called()
    self.affinity.assert_not_called()
    self.assertFalse(timer._enabled)
    self.assertEqual(timer._failure, ("drop_realtime", "PermissionError"))
    self.assertEqual(timer._failure_count, 1)
    self.event.assert_not_called()

  def test_affinity_failure_is_best_effort_after_demotion(self):
    timer = self.timer(period=1)
    self.frame(timer)
    timer.stop()
    self.affinity.side_effect = OSError("cannot pin")
    timer._writer()
    self.assertIsNone(timer._failure)
    self.assertEqual([call.args[0] for call in self.event.call_args_list],
                     ["test.phase_timing.warning", "test.phase_timing"])
    self.demote.assert_called_once()

  def test_emit_and_diagnostic_failures_do_not_escape_writer(self):
    timer = self.timer(period=1)
    self.frame(timer)
    self.event.side_effect = OSError("logger unavailable")
    timer._writer()
    self.assertEqual(timer._failure, ("emit", "OSError"))
    self.assertFalse(timer._enabled)
    returned = timer._free.get_nowait()
    self.assertIsNot(returned, timer._active)
    self.assertEqual(sum(returned["cpu"]["work"]), 0)
    self.assertEqual(self.event.call_count, 2)

  def test_failed_clear_never_returns_dirty_buffer_to_hot_thread(self):
    timer = self.timer(period=1)
    self.frame(timer)
    with patch.object(timer, "_clear", side_effect=MemoryError("cannot clear")):
      timer._writer()
    self.assertEqual(timer._failure, ("clear", "MemoryError"))
    self.assertFalse(timer._enabled)
    self.assertTrue(timer._free.empty())
    self.assertEqual(self.event.call_args.args, ("test.phase_timing.disabled",))

  def test_unexpected_queue_failure_disables_writer(self):
    timer = self.timer()
    timer._full = Mock()
    timer._full.get.side_effect = RuntimeError("queue failed")
    timer._writer()
    self.assertEqual(timer._failure, ("writer", "RuntimeError"))
    self.assertFalse(timer._enabled)


if __name__ == "__main__":
  unittest.main()
