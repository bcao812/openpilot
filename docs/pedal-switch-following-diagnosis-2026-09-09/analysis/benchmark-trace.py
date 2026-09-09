"""Historical offline analysis; hard-coded local run/corpus paths, no device or network access."""

import json
from pathlib import Path
import time

import numpy as np

from openpilot.selfdrive.controls.lib.longitudinal_trace import TRACE_FIELDS, LongitudinalTrace
from openpilot.selfdrive.controls.tests.test_longitudinal_trace import traced_update
from openpilot.selfdrive.controls.tests.replay_pedal_switches import save_json


run = Path('/tmp/c3-follow-chain-20260909-BtK1Bx')
planner, state = traced_update()
samples = []
lengths = []


def serialize(event, **payload):
  started = time.perf_counter_ns()
  encoded = json.dumps({'msg': dict(payload, event=event)}, allow_nan=False)
  samples.append((time.perf_counter_ns() - started) / 1000)
  lengths.append(len(encoded))


candidates = [(1.0, 1, False), (-0.19, 0, False)]
results = {}
for mode in ('disabled', 'full', 'writer'):
  recorder = None if mode == 'disabled' else LongitudinalTrace(emit=serialize)
  if mode == 'writer':
    assert recorder.start()
  durations = []
  for frame in range(4000):
    started = time.perf_counter_ns()
    if recorder is not None:
      recorder.begin(planner, state, False, 20.0, 2.0)
    if recorder is not None:
      recorder.finish(planner, candidates, False)
    if recorder is not None:
      recorder.publish(20_000 + frame)
    durations.append((time.perf_counter_ns() - started) / 1000)
    if mode == 'writer' and frame % 10 == 0:
      time.sleep(0.02)
  if recorder is not None:
    recorder.close()
  measured = durations[100:]
  results[mode] = {
    'calls': len(measured),
    'us': dict(zip(('p50', 'p95', 'p99', 'max'), np.percentile(measured, [50, 95, 99, 100]).tolist(), strict=True)),
    'dropped': recorder.dropped if recorder is not None else 0,
    'failures': recorder.failures if recorder is not None else 0,
  }
results['serialization'] = {
  'batches': len(samples),
  'us': dict(zip(('p50', 'p95', 'p99', 'max'), np.percentile(samples, [50, 95, 99, 100]).tolist(), strict=True)),
  'max_bytes': max(lengths),
  'fields': len(TRACE_FIELDS),
}
results['scope'] = 'Host-only microbenchmark; excludes actual C3 scheduling, ZMQ, logmessaged, compression and storage'
save_json(run / 'artifacts/trace-benchmark.json', results)
print(json.dumps(results, indent=2))
