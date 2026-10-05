"""Synthetic receive/decode/serialize/zstd cost, not device CPU or road evidence.

Run under the workspace resource runner. Traffic is advanced at 15 Hz simulated
sensor time; measured CPU cost is normalized per simulated second, not sleep time.
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import time
from types import SimpleNamespace
from unittest.mock import patch

import zstandard

from openpilot.selfdrive.car.honda_radar_observer import HondaRadarObserver
from openpilot.common.basedir import BASEDIR
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.car.tests.test_honda_radar_observer import SETTINGS, START, can_message, configuration, scan_frames, state
from openpilot.selfdrive.ui.sunnypilot.onroad.honda_radar_state import RadarOverlayState, project_radar_point


def percentiles(values):
  ordered = sorted(values)
  return {k: ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p))] / 1000
          for k, p in (('p50_us', .5), ('p95_us', .95), ('p99_us', .99))}


def measure(slots, seconds, level, valid_slots=None):
  valid_slots = slots if valid_slots is None else valid_slots
  observer = HondaRadarObserver(*configuration(), session_id='benchmark')
  overlay = RadarOverlayState()
  compressor = zstandard.ZstdCompressor(level=level).compressobj()
  totals = {'raw_bytes': 0, 'compressed_bytes': 0, 'messages': 0, 'observations': 0}
  costs = {name: [] for name in ('update', 'serialize', 'compress', 'overlay_select_project')}
  started_cpu = time.process_time_ns()
  measured_cpu_ns = 0
  for i in range(seconds * 15):
    t = START + i * 66_666_667
    frames = [f for s in range(slots) for f in scan_frames(s, index=i % 16, life=1 + 2 * i,
                                                        distance=400 + s * 170 + ((i * 7 + s * 13) % 200) if s < valid_slots else 4095,
                                                        oem_id=s + 1 if s < valid_slots else 0,
                                                        velocity=700 + i % 150 if s < valid_slots else 2046)]
    envelopes = [can_message(t, frames)] if frames else []
    sm = state(t, 'acc' if i % 60 < 30 else 'blended')
    cpu_begin = time.thread_time_ns()
    begin = time.perf_counter_ns()
    msg = observer.update(envelopes, sm, SETTINGS, t + 1)
    costs['update'].append(time.perf_counter_ns() - begin)
    if msg is None:
      measured_cpu_ns += time.thread_time_ns() - cpu_begin
      continue
    begin = time.perf_counter_ns()
    raw = msg.to_bytes()
    costs['serialize'].append(time.perf_counter_ns() - begin)
    begin = time.perf_counter_ns()
    totals['compressed_bytes'] += len(compressor.compress(raw))
    costs['compress'].append(time.perf_counter_ns() - begin)
    measured_cpu_ns += time.thread_time_ns() - cpu_begin
    totals['raw_bytes'] += len(raw)
    totals['messages'] += 1
    totals['observations'] += len(msg.hondaRadarObservations.observations)
    begin = time.perf_counter_ns()
    overlay.ingest(msg.hondaRadarObservations, t + 1, t + 1)
    for point in overlay.points(t, t + 1):
      project_radar_point(point, ((640., 1000., 0), (360., 0, 1000.), (1., 0, 0)), 1.2)
      _ = point.text
    costs['overlay_select_project'].append(time.perf_counter_ns() - begin)
  cpu_begin = time.thread_time_ns()
  totals['compressed_bytes'] += len(compressor.flush())
  measured_cpu_ns += time.thread_time_ns() - cpu_begin
  return dict(slots=slots, valid_slots=valid_slots, scenario=f'frames_{slots}_valid_{valid_slots}', simulated_seconds=seconds, **totals,
              raw_mib_per_hour=totals['raw_bytes'] / seconds * 3600 / 2**20,
              compressed_mib_per_hour=totals['compressed_bytes'] / seconds * 3600 / 2**20,
              host_observer_and_zstd_one_core_percent=measured_cpu_ns / (seconds * 1e9) * 100,
              total_benchmark_cpu_seconds=(time.process_time_ns() - started_cpu) / 1e9,
              costs={name: dict(**percentiles(values), mean_us=statistics.mean(values) / 1000) for name, values in costs.items()},
              dropped_events=observer.decoder.counts['dropped'])


def measure_render_logic(slots, seconds):
  """Measure the actual label layout loop at 60 FPS, without a display/GPU.

  Fixed text dimensions stand in for warmed font measurements; all rectangles,
  collision searches, source-time selection and projection use production code.
  The 16-slot case deliberately crowds the bottom edge to exercise separation.
  """
  os.environ.setdefault('SCALE', '1')
  import numpy as np
  import pyray as rl
  from openpilot.cereal import log, messaging
  from openpilot.selfdrive.ui.sunnypilot.onroad import honda_radar_overlay as renderer

  class ObservationSubMaster:
    def __init__(self):
      self.updated = {renderer.SERVICE: True}
      self.valid = {renderer.SERVICE: True}
      self.logMonoTime = {renderer.SERVICE: START}
      self.body = None

    def update(self, _):
      pass

    def __getitem__(self, _):
      return self.body

  sm = state(START)
  calib = messaging.new_message('extrinsicsCalibration').extrinsicsCalibration
  calib.calStatus, calib.height = log.ExtrinsicsCalibration.Status.calibrated, [1.2]
  sm['extrinsicsCalibration'] = calib
  sm.valid['extrinsicsCalibration'] = True
  observer_sm = ObservationSubMaster()
  overlay = renderer.HondaRadarOverlay()
  overlay.sm, overlay.enabled, overlay.font = observer_sm, True, object()
  overlay.next_params_ns = START + seconds * 1_000_000_000 + 1
  view = SimpleNamespace(frame=object(), client=SimpleNamespace(timestamp_eof=START), device_camera=object(),
                         model_renderer=SimpleNamespace(_car_space_transform=np.array(((1080., 1000., 0), (1000., 0, 1000.), (1., 0, 0)))))
  ui = SimpleNamespace(CP=configuration()[0], started=True, sm=sm)
  rect, size = rl.Rectangle(0, 0, 2160, 1080), rl.Vector2(120, 32)
  costs, cpu_ns = [], 0
  now = START
  with patch.object(renderer, 'ui_state', ui), patch.object(renderer.time, 'monotonic_ns', lambda: now), \
       patch.object(renderer, 'measure_text_cached', lambda *_: size), \
       patch.object(rl, 'draw_rectangle_rec', lambda *_: None), patch.object(rl, 'draw_text_ex', lambda *_: None), \
       patch.object(rl, 'draw_line_v', lambda *_: None):
    for i in range(seconds * 60):
      now = START + i * 1_000_000_000 // 60
      sm.logMonoTime = dict.fromkeys(sm.logMonoTime, now)
      view.client.timestamp_eof = now
      observer_sm.updated[renderer.SERVICE] = i % 4 == 0
      if i % 4 == 0:
        message = messaging.new_message(renderer.SERVICE)
        body = getattr(message, renderer.SERVICE)
        body.schemaVersion, body.sessionId, body.sequence, body.active = 1, 'render-benchmark', i // 4 + 1, True
        body.scope = dict.fromkeys(('alphaLong', 'experimentalSetting', 'decSetting', 'experimentalActive', 'decEnabled', 'decActive'), True)
        body.observations = [{'slot': s, 'positionSeq': i * 32 + 2 * s + 1, 'generation': s + 1,
                              'positionMonoTime': now, 'dRel': 54.12, 'yRel': 0.0, 'positionValid': True}
                             for s in range(slots) for _ in range(2)]
        observer_sm.body, observer_sm.logMonoTime[renderer.SERVICE] = body.as_reader(), now
      start_cpu, begin = time.thread_time_ns(), time.perf_counter_ns()
      overlay._render(view, rect)  # exceptions must fail this benchmark
      costs.append(time.perf_counter_ns() - begin)
      cpu_ns += time.thread_time_ns() - start_cpu
  assert len(overlay.state.points(now, now)) == slots
  return dict(slots=slots, simulated_seconds=seconds, frames=len(costs), viewport=[2160, 1080],
              host_logic_one_core_percent_at_60fps=cpu_ns / (seconds * 1e9) * 100,
              **percentiles(costs), mean_us=statistics.mean(costs) / 1000)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seconds', type=int, default=30)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  if not 1 <= args.seconds <= 60:
    parser.error('Use 1..60 simulated seconds for this bounded fixture.')
  header = (Path(BASEDIR) / 'openpilot/system/loggerd/logger.h').read_text()
  level = int(re.search(r'LOG_COMPRESSION_LEVEL\s*=\s*(\d+)', header).group(1))
  cases = [(0, 0), (1, 1), (16, 0), (16, 1), (16, 16)]
  measurements = [measure(total, args.seconds, level, valid) for total, valid in cases]
  with OpenpilotPrefix():  # UI module initialization must use a private host namespace
    render_costs = [measure_render_logic(slots, args.seconds) for slots in (1, 16)]
  results = {'method': 'synthetic-15Hz-host-v2', 'zstd_level': level,
             'cases': measurements,
             'render_logic_60fps': render_costs,
             'limitations': ['Synthetic fields; real compression depends on traffic.',
                             'Excludes CAN socket receive, loggerd queueing/fsync and actual GPU text drawing.',
                             'Render loop uses fixed warmed text dimensions and no-op draw calls; not full UI frame cost.',
                             'Host measurements are not C3 CPU/load measurements.']}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(results, indent=2) + '\n')
  print(json.dumps(results, indent=2))


if __name__ == '__main__':
  main()
