"""Synthetic receive/decode/serialize/zstd cost, not device CPU or road evidence.

Run under the workspace resource runner. Traffic is advanced at 15 Hz simulated
sensor time; measured CPU cost is normalized per simulated second, not sleep time.
"""
import argparse
import json
from pathlib import Path
import re
import statistics
import time

import zstandard

from openpilot.selfdrive.car.honda_radar_observer import HondaRadarObserver
from openpilot.common.basedir import BASEDIR
from openpilot.selfdrive.car.tests.test_honda_radar_observer import SETTINGS, START, can_message, configuration, scan_frames, state
from openpilot.selfdrive.ui.sunnypilot.onroad.honda_radar_state import RadarOverlayState, project_radar_point


def percentiles(values):
  ordered = sorted(values)
  return {k: ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p))] / 1000
          for k, p in (('p50_us', .5), ('p95_us', .95), ('p99_us', .99))}


def measure(slots, seconds, level):
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
                                                        distance=400 + s * 170 + ((i * 7 + s * 13) % 200), velocity=700 + i % 150)]
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
    overlay.ingest(msg.hondaRadarObservations.to_dict(), t + 1, t + 1)
    for point in overlay.points(t, t + 1):
      project_radar_point(point, ((640., 1000., 0), (360., 0, 1000.), (1., 0, 0)), 1.2)
      _ = point.text
    costs['overlay_select_project'].append(time.perf_counter_ns() - begin)
  cpu_begin = time.thread_time_ns()
  totals['compressed_bytes'] += len(compressor.flush())
  measured_cpu_ns += time.thread_time_ns() - cpu_begin
  return dict(slots=slots, simulated_seconds=seconds, **totals,
              raw_mib_per_hour=totals['raw_bytes'] / seconds * 3600 / 2**20,
              compressed_mib_per_hour=totals['compressed_bytes'] / seconds * 3600 / 2**20,
              host_observer_and_zstd_one_core_percent=measured_cpu_ns / (seconds * 1e9) * 100,
              total_benchmark_cpu_seconds=(time.process_time_ns() - started_cpu) / 1e9,
              costs={name: dict(**percentiles(values), mean_us=statistics.mean(values) / 1000) for name, values in costs.items()},
              dropped_events=observer.decoder.counts['dropped'])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--seconds', type=int, default=30)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  if not 1 <= args.seconds <= 60:
    parser.error('Use 1..60 simulated seconds for this bounded fixture.')
  header = (Path(BASEDIR) / 'openpilot/system/loggerd/logger.h').read_text()
  level = int(re.search(r'LOG_COMPRESSION_LEVEL\s*=\s*(\d+)', header).group(1))
  results = {'method': 'synthetic-15Hz-host-v1', 'zstd_level': level, 'cases': [measure(n, args.seconds, level) for n in (0, 1, 16)],
             'limitations': ['Synthetic fields; real compression depends on traffic.',
                             'Excludes CAN socket receive, loggerd queueing/fsync and actual GPU text drawing.',
                             'Host measurements are not C3 CPU/load measurements.']}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(results, indent=2) + '\n')
  print(json.dumps(results, indent=2))


if __name__ == '__main__':
  main()
