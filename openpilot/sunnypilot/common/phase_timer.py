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
    except OSError:
      return None
    return cls(marker, event, phases, period_frames, affinity)

  def __init__(self, marker, event, phases, period_frames, affinity=(0, 1, 2)):
    self.MARKER = marker
    self.event = event
    self.PHASES = tuple(phases)
    self.PERIOD_FRAMES = period_frames
    self.affinity = list(affinity)

    # A set is always in exactly one of four places: written by the hot thread, sitting in
    # _full, held by the writer inside _emit, or sitting in _free. Two queues rather than one
    # because a set that has been get() off a single queue is in NEITHER the queue nor the hot
    # thread's hands, and an earlier revision lost track of exactly that: with one queue and two
    # sets, the second rotate found the queue empty (the writer had already taken the item),
    # succeeded, and handed the hot thread back the set the writer was still reading.
    self._sets = [self._new_set() for _ in range(2)]
    self.wall = self._sets[0]['wall']
    self.cpu = self._sets[0]['cpu']
    self.i = 0
    self.periods = 0
    self.dropped = 0
    self._full = queue.Queue(maxsize=1)
    self._free = queue.Queue(maxsize=2)
    self._free.put_nowait(self._sets[1])
    self._w = 0
    self._c = 0
    self._stop = threading.Event()
    self._thread = None

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
    if self._thread is None:
      self._thread = threading.Thread(target=self._writer, daemon=True)
      self._thread.start()

  def stop(self):
    self._stop.set()

  def begin(self):
    self._w = time.perf_counter_ns()
    self._c = time.thread_time_ns()

  def mark(self, phase):
    w = time.perf_counter_ns()
    c = time.thread_time_ns()
    dw = w - self._w
    dc = c - self._c
    # The bit_length guard is not branchless for style: the equivalent branch on the bucket
    # index cost 22 ns per call and every real delta is far above the linear region it
    # protected. A delta under 64 ns mis-files by one bucket; the smallest real phase measured
    # on this device is ~10 us.
    sh = dw.bit_length() - 6
    if sh < 0:
      sh = 0
    i = ((sh + 1) << 5) | ((dw >> sh) & 31)
    self.wall[phase][i if i <= self.MAX_BUCKET else self.MAX_BUCKET] += 1
    sh = dc.bit_length() - 6
    if sh < 0:
      sh = 0
    i = ((sh + 1) << 5) | ((dc >> sh) & 31)
    self.cpu[phase][i if i <= self.MAX_BUCKET else self.MAX_BUCKET] += 1
    self._w = w
    self._c = c

  def end_frame(self):
    self.i += 1
    if self.i >= self.PERIOD_FRAMES:
      self._rotate()

  def _rotate(self):
    """Hand the filled buffers off and take a free set. Runs on the HOT thread, so it does no
    work beyond two queue operations and three stores, and never blocks."""
    n = self.i
    self.i = 0
    if n < 2:
      return
    try:
      spare = self._free.get_nowait()
    except queue.Empty:
      # The writer still owns the other set. Overwrite the active one rather than wait: losing a
      # window of diagnostics beats stalling the hot loop, and the count rides along in the next
      # event so the loss is visible rather than silent.
      self.dropped += 1
      return
    filled = self.wall, self.cpu
    self.wall = spare['wall']
    self.cpu = spare['cpu']
    self._full.put_nowait(({'wall': filled[0], 'cpu': filled[1]}, n))

  def _writer(self):
    # Demote FIRST, and in its own try. Every caller of this class has already put its process
    # at SCHED_FIFO on a pinned core, and a thread inherits its creator's policy -- so without
    # this the emit runs at real-time priority on the control or display core, which is the
    # whole problem the thread exists to remove. Sharing one try block with the affinity call
    # would let a raise from either skip the other; selfdrived hit exactly that and left the
    # writer at FIFO 53 on core 4.
    try:
      drop_realtime()
    except OSError:
      cloudlog.exception(f"{self.event} writer: could not drop to SCHED_OTHER")
    try:
      # Little cluster by default, and NOT core 3: that is pandad at FIFO 54 plus the spi_geni
      # IRQ, and a thread preempted by pandad while holding the GIL would block the very loop
      # this thread exists to keep clear. Callers pinned to the little cluster themselves should
      # pass something that does not include their own core.
      set_core_affinity(self.affinity)
    except OSError:
      cloudlog.exception(f"{self.event} writer: could not set affinity")
    while not self._stop.is_set():
      try:
        buf, n = self._full.get(timeout=0.5)
      except queue.Empty:
        continue
      try:
        self._emit(buf, n)
      except Exception:
        cloudlog.exception(f"{self.event} writer failed")
      finally:
        # ALWAYS hand the set back, including after a failed emit. Skipping this on the error
        # path strands the set, leaves _free empty, and turns every later window into a silent
        # drop. Cleared HERE, on the writer, because zeroing NB_BUCKETS x 2 x len(PHASES)
        # counters is far too much work for the hot thread to do at a window boundary.
        try:
          self._clear(buf)
        except Exception:
          cloudlog.exception(f"{self.event} clear failed")
        try:
          self._free.put_nowait(buf)
        except queue.Full:
          pass

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
      acc = 0.0
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
      frames=n,
      period=self.periods,
      cpu_us=cpu,
      wall_us=wall,
      cpu_total_us=round(sum(c["mean"] for c in cpu.values() if c), 1),
      wall_total_us=round(sum(w["mean"] for w in wall.values() if w), 1),
      dropped=self.dropped,
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
