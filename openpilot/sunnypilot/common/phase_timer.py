"""Per-phase wall and CPU timing for a hot loop, written to the log every N frames. Off by
default, armed by a marker file.

  timer = PhaseTimer.create("/data/foo_profile", "foo.phase_timing", PHASES, 3000)
  if timer:
    timer.start()
  ...
  while True:
    if timer:
      timer.begin()
    do_a()
    if timer:
      timer.mark('a')
    do_b()
    if timer:
      timer.mark('b')
      timer.end_frame()

WHY THIS IS ONE MODULE AND NOT A CLASS PER PROCESS

It was two, for a day, and that day is the argument. selfdrived got a PhaseTimer (2dd99563d) and
the UI got a FrameTimer copied from it (acbe47496); they came to 152 identical lines. The next
fix -- bucket saturation, below -- went into the UI's copy and NOT into selfdrived's, which was
running on the car at the time with an unbounded index on the 100 Hz actuation chain. One
defect, two copies, fixed in one, one day later.

This class has now needed five revisions, every one of them found on the car:

  1. In-line flush. Sorted 3000 samples inside the hot loop. ~30 ms stall, once per window.
  2. Writer thread with 1 ms sleeps. Better, and still 14 ms: a single sorted() of 3000 PyLongs
     holds the GIL ~5.3 ms and that was the bound, not the thread.
  3. Log histogram, sleeps removed on the reasoning that no long GIL hold was left. The car
     disagreed -- MORE stalls, smaller ones. The bound is sys.getswitchinterval(), 5 ms, not the
     duration of any single call.
  4. Sleeps restored.
  5. Bucket saturation, after mark() raised IndexError in the process that draws the screen.

Four copies would mean applying the sixth four times.

WHAT IS PARAMETERISED AND WHAT IS NOT

Marker, event name, phase names, window length, the blocking phase and the writer's affinity are
arguments -- they genuinely differ per process. Everything else is the accumulated result of
those five revisions and is deliberately not adjustable.

WHY A MARKER FILE AND NOT A PARAM OR AN ENV VAR

A Param key lives in common/params_keys.h, which is compiled into libparams_c.so -- so adding
one turns a pure-Python change into a host build plus a committed aarch64 .so plus a sync. An
env var in launch_env.sh needs either a sync to flip or an in-place ssh edit, and the latter
leaves the device tree dirty, which breaks the only verification c3-sync has. /data is not in
git.

  on:   ssh comma@<device> touch <marker>     (takes effect at next process start)
  off:  ssh comma@<device> rm <marker>

WHY TWO CLOCKS

perf_counter_ns is wall, thread_time_ns is this thread's CPU. A loop paced by a blocking call --
recv_one(carState), drain_sock_raw, set_target_fps parking on vblank -- shows wall large and CPU
near zero there, and that gap is the point: it separates pacing from work. A single clock cannot
make that call, and CPU alone hides the pacing entirely.
"""
import os
import queue
import threading
import time

from openpilot.common.swaglog import cloudlog
from openpilot.common.realtime import drop_realtime, set_core_affinity


class PhaseTimer:
  # HdrHistogram-style: SUB_BITS sub-buckets per octave, so relative resolution is constant at
  # ~1/32 = 3% across the range instead of being fixed in absolute terms. These phases span
  # three orders of magnitude in one frame -- tens of microseconds to the ~10 ms pacing wait.
  #
  # This replaced per-frame sample buffers and the reason is measured, not aesthetic: the writer
  # had to sort 3000 samples per phase per clock, and sorted() is one C call that holds the GIL
  # until it returns -- ~5.3 ms on the car, 18 calls per window. Reading buckets is an
  # interpreter-bound loop instead, ~26 us on a host.
  #
  # NB_BUCKETS covers 1 ns to well past 1 s: the largest index for 1e9 ns is 831.
  SUB_BITS = 5
  SUB_MASK = (1 << SUB_BITS) - 1
  NB_BUCKETS = 1024

  # Largest index NB_BUCKETS allows. Anything past it saturates into the top bucket, which is
  # what a histogram is supposed to do -- and is NOT what this did before.
  #
  # It raised IndexError. mark() computes a bucket from the delta since the previous mark, and
  # 1024 buckets run out at about 137 s; a mark reached with a stale cursor produces a delta of
  # "nanoseconds since boot", indexes past the end, and takes the process down. A host wiring
  # check hit exactly that. It is reachable on a real car too: selfdrived's recv_one calls
  # sock.receive() with NO timeout, so a long CAN stall lands there. On a 100 Hz control loop an
  # instrument that can raise is bad; in the process that draws the screen it is worse. One
  # comparison per clock read -- pay it.
  MAX_BUCKET = NB_BUCKETS - 1

  @classmethod
  def create(cls, marker, event, phases, period_frames, affinity=(0, 1, 2)):
    """A PhaseTimer if the marker is present, else None.

    Checked once, here, and never on the hot path -- so flipping the marker takes effect at the
    next process start, which on this device means the next ignition cycle.
    """
    try:
      if not os.path.exists(marker):
        return None
      return cls(marker, event, phases, period_frames, affinity)
    except Exception as error:
      cls._diagnostic(event, "create", type(error).__name__)
      return None

  def __init__(self, marker, event, phases, period_frames, affinity=(0, 1, 2)):
    self.MARKER = marker
    self.event = event
    self.PHASES = tuple(phases)
    self.PERIOD_FRAMES = period_frames
    self.affinity = list(affinity)
    if not isinstance(period_frames, int) or period_frames < 1:
      raise ValueError("period_frames must be a positive integer")
    if not self.PHASES or len(self.PHASES) != len(set(self.PHASES)):
      raise ValueError("phases must be nonempty and unique")

    # A set is always in exactly one of four places: written by the hot thread, sitting in
    # _full, held by the writer inside _emit, or sitting in _free. Two queues rather than one
    # because a set that has been get() off a single queue is in NEITHER the queue nor the hot
    # thread's hands, and an earlier revision lost track of exactly that: with one queue and two
    # sets, the second rotate found the queue empty (the writer had already taken the item),
    # succeeded, and handed the hot thread back the set the writer was still reading.
    self._sets = [self._new_set() for _ in range(2)]
    self._active = self._sets[0]
    self.wall = self._active['wall']
    self.cpu = self._active['cpu']
    self.i = 0
    self._rotate_at = self.PERIOD_FRAMES
    self._window_start = None
    self.periods = 0
    # Kept as a legacy alias for deferred_windows in version 2 events. A busy writer merges
    # windows; it does not discard their frames or samples.
    self.dropped = 0
    # Ownership of the two preallocated sets bounds these queues, without Queue's hot-path
    # mutex acquisition. SimpleQueue.put_nowait never waits for a reader or free queue space.
    self._full = queue.SimpleQueue()
    self._free = queue.SimpleQueue()
    self._free.put_nowait(self._sets[1])
    self._w = None
    self._c = None
    self._stop = threading.Event()
    self._thread = None
    self._enabled = True
    self._failure = None
    self._failure_count = 0
    self._failure_reported = False

  @staticmethod
  def _diagnostic(event, operation, exception, *, disabled=True):
    """Best effort, called only during setup or on the writer, never by a hot-path failure."""
    try:
      cloudlog.event(f"{event}.{'disabled' if disabled else 'warning'}", operation=operation, exception=exception, error=True)
    except Exception:
      # A broken logger must not turn an optional instrument into a process failure.
      pass

  def _disable(self, operation, error):
    # No logging, locks, traceback formatting or retained exception/stack on the hot thread.
    # The writer observes this once its bounded queue wait finishes and reports it once.
    if self._failure is None:
      self._failure = operation, type(error).__name__
    self._failure_count += 1
    self._enabled = False

  def _report_failure(self):
    if self._failure is not None and not self._failure_reported:
      self._failure_reported = True
      self._diagnostic(self.event, *self._failure)

  @classmethod
  def bucket_mid(cls, idx):
    """Midpoint of a bucket, in ns. The inverse of the index arithmetic in mark()."""
    sh = (idx >> cls.SUB_BITS) - 1
    if sh < 0:
      return idx
    lo = ((1 << cls.SUB_BITS) | (idx & cls.SUB_MASK)) << sh
    return lo + (1 << sh) // 2

  def _new_set(self):
    return {'wall': {p: [0] * self.NB_BUCKETS for p in self.PHASES},
            'cpu': {p: [0] * self.NB_BUCKETS for p in self.PHASES}}

  def start(self):
    if self._enabled and self._thread is None:
      try:
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()
      except Exception as error:
        self._disable("start", error)
        self._report_failure()

  def stop(self):
    # Do not flush/clear the active (possibly incomplete) frame or join a logging thread here.
    # The writer drains already completed windows, then exits. Further hot calls are no-ops.
    self._enabled = False
    self._stop.set()

  def begin(self):
    if not self._enabled:
      return
    try:
      if self._window_start is None:
        self._window_start = time.monotonic_ns()
      self._w = time.perf_counter_ns()
      self._c = time.thread_time_ns()
    except Exception as error:
      self._disable("begin", error)

  def mark(self, phase):
    if not self._enabled:
      return
    try:
      w = time.perf_counter_ns()
      c = time.thread_time_ns()
      dw = w - self._w
      dc = c - self._c
      if dw < 0 or dc < 0:
        raise ValueError("phase clock moved backwards")
      # Below 64 ns the buckets are linear, including a real zero for a coarse CPU clock.
      # Larger deltas retain constant relative resolution and saturate safely after long stalls.
      sh = dw.bit_length() - 6
      i = dw if sh <= 0 else ((sh + 1) << 5) | ((dw >> sh) & 31)
      self.wall[phase][i if i <= self.MAX_BUCKET else self.MAX_BUCKET] += 1
      sh = dc.bit_length() - 6
      i = dc if sh <= 0 else ((sh + 1) << 5) | ((dc >> sh) & 31)
      self.cpu[phase][i if i <= self.MAX_BUCKET else self.MAX_BUCKET] += 1
      self._w = w
      self._c = c
    except Exception as error:
      self._disable("mark", error)

  def end_frame(self):
    if not self._enabled:
      return
    try:
      self.i += 1
      if self.i >= self._rotate_at:
        self._rotate()
    except Exception as error:
      self._disable("end_frame", error)

  def _rotate(self):
    """Hand off a complete window using bounded queue work, with no clearing or logging."""
    n = self.i
    if not n:
      return
    window_end = time.monotonic_ns()
    if self._window_start is None or window_end < self._window_start:
      raise ValueError("invalid phase timing window")
    try:
      spare = self._free.get_nowait()
    except queue.Empty:
      # The writer still owns the other set. Keep BOTH the histograms and their frame count;
      # resetting only the count makes conditional phases and loop averages lie. Retry only at
      # the next nominal boundary, not on every following frame while the writer is busy.
      self.dropped += 1
      self._rotate_at = n + self.PERIOD_FRAMES
      return
    filled = self._active
    # Strings preserve integer nanoseconds through JSON readers backed by IEEE doubles.
    filled['window_start_mono_ns'] = str(self._window_start)
    filled['window_end_mono_ns'] = str(window_end)
    filled['deferred_windows'] = self.dropped
    self._active = spare
    self.wall = spare['wall']
    self.cpu = spare['cpu']
    self.i = 0
    self._rotate_at = self.PERIOD_FRAMES
    self._window_start = None
    self._full.put_nowait((filled, n))

  def _writer(self):
    # Demote FIRST, and in its own try. Every caller of this class has already put its process
    # at SCHED_FIFO on a pinned core, and a thread inherits its creator's policy -- so without
    # this the emit runs at real-time priority on the control or display core, which is the
    # whole problem the thread exists to remove. If demotion fails, do no histogram or logging
    # work at inherited FIFO priority: stop this optional instrument before processing a window.
    demoted = False
    try:
      try:
        drop_realtime()
        demoted = True
      except Exception as error:
        self._disable("drop_realtime", error)
        return
      try:
        # Little cluster by default, and NOT core 3: that is pandad at FIFO 54 plus the spi_geni
        # IRQ, and a thread preempted by pandad while holding the GIL would block the very loop
        # this thread exists to keep clear. After demotion, affinity is best effort; the GIL
        # yields inside _emit remain necessary even when the worker has its own core.
        set_core_affinity(self.affinity)
      except Exception as error:
        self._diagnostic(self.event, "set_core_affinity", type(error).__name__, disabled=False)
      while self._failure is None:
        try:
          buf, n = self._full.get_nowait() if self._stop.is_set() else self._full.get(timeout=0.5)
        except queue.Empty:
          if self._stop.is_set():
            break
          continue
        try:
          self._emit(buf, n)
        except Exception as error:
          self._disable("emit", error)
        # Clear only on the writer. A failed clear must NEVER return a partially cleared set:
        # that would silently mix old samples into a later window. Disable instead of retrying
        # allocations or running a persistent failing logger every period.
        try:
          self._clear(buf)
        except Exception as error:
          self._disable("clear", error)
          break
        self._free.put_nowait(buf)
    except Exception as error:
      self._disable("writer", error)
    finally:
      # If even demotion failed, retain the reason/count in memory and exit silently. A
      # diagnostic logger can also serialize or block; do not run it at inherited FIFO.
      if demoted:
        self._report_failure()

  def _emit(self, buf, n):
    self.periods += 1

    def stats(h):
      """Percentiles straight out of the buckets. No sort, no slice, no sum over samples -- the
      whole point of the histogram. Values are bucket midpoints, so everything here carries ~3%
      relative error; that is stated in the event rather than hidden behind a decimal place."""
      total = 0
      for c in h:
        total += c
      if total == 0:
        return None
      want = (int(total * 0.5), int(total * 0.95), int(total * 0.99))
      out = [0.0, 0.0, 0.0]
      seen = 0
      acc = 0
      last = 0
      k = 0
      for idx, c in enumerate(h):
        if not c:
          continue
        last = idx
        mid = PhaseTimer.bucket_mid(idx)
        acc += mid * c
        seen += c
        while k < 3 and seen > want[k]:
          out[k] = mid
          k += 1
      while k < 3:
        out[k] = PhaseTimer.bucket_mid(last)
        k += 1
      return {
        "count": total,
        "sum_us": round(acc / 1000.0, 3),
        "p50": round(out[0] / 1000.0, 1),
        "p95": round(out[1] / 1000.0, 1),
        "p99": round(out[2] / 1000.0, 1),
        "max": round(PhaseTimer.bucket_mid(last) / 1000.0, 1),
        "mean": round(acc / total / 1000.0, 1),
      }

    # Yield between phases. These were introduced to break up the sorts, removed when the sorts
    # went, and the car said the stalls got MORE frequent: 7 of 20 windows, worst 8.89 ms.
    #
    # The bound is not how long one call runs, it is sys.getswitchinterval() -- 5 ms by default.
    # Pure Python does release the GIL, but only once a waiting thread has asked and that
    # interval has elapsed, so an emit of uninterrupted bytecode makes the hot thread wait most
    # of an interval no matter how short each operation is. An explicit sleep offers the GIL up
    # immediately. 200 us per phase against a writer that has the whole window is the cheapest
    # thing in this file.
    cpu = {}
    wall = {}
    for p in self.PHASES:
      cpu[p] = stats(buf['cpu'][p])
      time.sleep(0.0002)
      wall[p] = stats(buf['wall'][p])
      time.sleep(0.0002)

    # Drop the window only when NOTHING ran. Dropping if ANY phase was empty was right while
    # every phase was unconditional, and became a silent hole as soon as a caller had a
    # conditional path -- the UI's onroad sub-phases do not run offroad, so every offroad window
    # would have been thrown away whole. Empty phases emit as null; the reader skips them and
    # says how many windows each phase covers. A phase that never ran is data, not a fault.
    if all(v is None for v in cpu.values()):
      return

    cloudlog.event(
      self.event,
      stats_version=2,
      frames=n,
      nominal_frames=self.PERIOD_FRAMES,
      window_start_mono_ns=buf['window_start_mono_ns'],
      window_end_mono_ns=buf['window_end_mono_ns'],
      period=self.periods,
      cpu_us=cpu,
      wall_us=wall,
      # Conditional phases may occur less than once per frame, and phases inside a drained
      # message loop may occur several times. Their conditional means cannot just be added.
      cpu_total_us=round(sum(c["sum_us"] for c in cpu.values() if c) / n, 1),
      wall_total_us=round(sum(w["sum_us"] for w in wall.values() if w) / n, 1),
      dropped=buf['deferred_windows'],
      deferred_windows=buf['deferred_windows'],
      bucket_pct=round(100.0 / (1 << self.SUB_BITS), 1),
    )

  def _clear(self, buf):
    """Fresh lists rather than an in-place loop: a handful of C-level allocations beat tens of
    thousands of Python stores, and refcounting reclaims the old ones immediately -- gc.disable()
    only stops cycle collection, and there are no cycles here."""
    for clk in ('wall', 'cpu'):
      d = buf[clk]
      for p in self.PHASES:
        d[p] = [0] * self.NB_BUCKETS
