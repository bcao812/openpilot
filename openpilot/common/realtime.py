"""Utilities for reading real time clocks and keeping soft real time constraints."""
import gc
import os
import sys
import time

from setproctitle import getproctitle

from openpilot.common.utils import MovingAverage
from openpilot.common.hardware import PC


# time step for each process
DT_CTRL = 0.01  # controlsd
DT_MDL = 0.05  # model
DT_HW = 0.5  # hardwared and manager
DT_DMON = 0.05  # driver monitoring


class Priority:
  # The layout these numbers exist to express, as it actually is on this device. Cores 0-3 are
  # Kryo 385 Silver and 4-7 are Gold, so "which core" is not interchangeable with "how busy".
  # Derive the live picture with _logs/procs.py --affinity rather than trusting this comment.
  #
  #   core 3  pandad 54            + spi_geni IRQ
  #   core 4  card / controlsd / selfdrived, all CTRL_HIGH -- one serial 100 Hz chain,
  #           equal priority on one core so the stages run to completion in order
  #   core 5  ui CTRL_HIGH         + fts_ts (touch) and msm_drm (display) IRQs
  #   core 6  camerad, no RT       + the camera IRQ group
  #   core 7  modeld 54, plannerd and radard CTRL_LOW   + kgsl-3d0 (GPU) IRQ
  #
  # plannerd and radard sit with modeld because both poll modelV2, which modeld publishes:
  # same core means the wake is cache-local. They were on core 5 until 2026-08-28, where the
  # UI at CTRL_HIGH preempted them every frame.
  CTRL_LOW = 51 # plannerd & radard

  CTRL_HIGH = 53


def drop_realtime() -> None:
  if sys.platform == 'linux' and not PC:
    os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))


def set_core_affinity(cores: list[int]) -> None:
  if sys.platform == 'linux' and not PC:
    os.sched_setaffinity(0, cores)


def config_realtime_process(cores: int | list[int], priority: int) -> None:
  gc.disable()
  if sys.platform == 'linux' and not PC:
    os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
  c = cores if isinstance(cores, list) else [cores, ]
  set_core_affinity(c)


class Ratekeeper:
  def __init__(self, rate: float, print_delay_threshold: float | None = 0.0) -> None:
    """Rate in Hz for ratekeeping. print_delay_threshold must be nonnegative."""
    self._interval = 1. / rate
    self._print_delay_threshold = print_delay_threshold
    self._frame = 0
    self._remaining = 0.0
    self._process_name = getproctitle()
    self._last_monitor_time = -1.
    self._next_frame_time = -1.

    self.avg_dt = MovingAverage(100)
    self.avg_dt.add_value(self._interval)

  @property
  def frame(self) -> int:
    return self._frame

  @property
  def remaining(self) -> float:
    return self._remaining

  @property
  def lagging(self) -> bool:
    expected_dt = self._interval * (1 / 0.9)
    return self.avg_dt.get_average() > expected_dt

  # Maintain loop rate by calling this at the end of each loop
  def keep_time(self) -> bool:
    lagged = self.monitor_time()
    if self._remaining > 0:
      time.sleep(self._remaining)
    return lagged

  # Monitors the cumulative lag, but does not enforce a rate
  def monitor_time(self) -> bool:
    if self._last_monitor_time < 0:
      self._next_frame_time = time.monotonic() + self._interval
      self._last_monitor_time = time.monotonic()

    prev = self._last_monitor_time
    self._last_monitor_time = time.monotonic()
    self.avg_dt.add_value(self._last_monitor_time - prev)

    lagged = False
    remaining = self._next_frame_time - time.monotonic()
    self._next_frame_time += self._interval
    if self._print_delay_threshold is not None and remaining < -self._print_delay_threshold:
      print(f"{self._process_name} lagging by {-remaining * 1000:.2f} ms")
      lagged = True
    self._frame += 1
    self._remaining = remaining
    return lagged
