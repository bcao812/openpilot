"""Verify trace off/on against identical fixed short log windows in a host-built copy."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import inspect
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.longitudinal_trace import LongitudinalTrace, TRACE_EVENT
from openpilot.tools.longitudinal.evaluate_pedal_full_routes import native_clean, exact_alignment, columns, compare_can
from openpilot.tools.longitudinal.window_replay import load_window, window_messages
from openpilot.tools.longitudinal.replay_pedal_full_routes import NumericSink, ROW_FIELDS, NATIVE_FIELDS
from openpilot.tools.longitudinal.replay_pedal_switches import replay_window, save_json, sha256


def run_window(arguments):
  corpus, destination, window, enabled = arguments
  mode = 'on' if enabled else 'off'
  output = destination / mode
  output.mkdir(parents=True, exist_ok=True)
  original_init = LongitudinalPlanner.__init__
  records = []
  emitted = {'batches': 0, 'samples': 0, 'bytes': 0}
  source_files = [
    Path(__file__),
    Path(__file__).with_name('replay_pedal_switches.py'),
    Path(__file__).with_name('window_replay.py'),
    Path(inspect.getfile(LongitudinalTrace)),
    Path(inspect.getfile(LongitudinalPlanner)),
  ]
  source_hashes = {path.name: sha256(path) for path in source_files}

  def emit(event, **payload):
    encoded = json.dumps({'msg': dict(payload, event=event)}, allow_nan=False)
    emitted['batches'] += 1
    emitted['samples'] += len(payload['samples'])
    emitted['bytes'] += len(encoded)

  def initialize(planner, *args, **kwargs):
    recorder = LongitudinalTrace(emit=emit) if enabled else None
    if recorder is not None:
      if not recorder.start():
        raise RuntimeError('Diagnostic writer did not start')
      records.append(recorder)
    original_init(planner, *args, trace=recorder, **kwargs)

  class TraceSink(NumericSink):
    def __call__(self, kind, row):
      if kind == 'plan' and records:
        records[-1].publish(row['mono'])
      return super().__call__(kind, row)

  sink = TraceSink(output, window['id'])
  started = time.monotonic()
  try:
    with patch.object(LongitudinalPlanner, '__init__', new=initialize):
      result = replay_window(None, dict(window, cache=window['id']), messages=window_messages(corpus, window, sink), row_sink=sink, strict=True)
  finally:
    for recorder in records:
      recorder.close()
  sink.flush()
  for name in ('plan_rows', 'control_rows', 'can_rows', 'planner_error', 'switches'):
    result.pop(name)
  trace = dict(
    emitted,
    attempted=sum(recorder.sequence for recorder in records),
    dropped=sum(recorder.dropped for recorder in records),
    failures=sum(recorder.failures for recorder in records),
    disabled=any(recorder.disabled for recorder in records),
  )
  if trace['disabled'] or trace['failures'] or trace['samples'] + trace['dropped'] != trace['attempted']:
    raise ValueError(f'Diagnostic capture failed: {trace}')
  result.update(mode=mode, chunks=sink.chunks, fields=ROW_FIELDS, trace=trace, elapsed_s=time.monotonic() - started, source_hashes=source_hashes)
  save_json(output / f'{window["id"]}.json', result)
  print(window['id'], mode, result['coverage'], trace, flush=True)


def evaluate(selection, output):
  corpus = Path(selection['corpus'])
  if sha256(corpus / 'manifest.json') != selection['manifest_sha256']:
    raise ValueError('Corpus manifest changed')
  manifest = json.loads((corpus / 'manifest.json').read_text())
  results = []
  for window in selection['windows']:
    before_meta, before = load_window(output, 'off', window['id'])
    after_meta, after = load_window(output, 'on', window['id'])
    expected_window = dict(window, cache=window['id'])
    if before_meta['window'] != expected_window or after_meta['window'] != expected_window:
      raise ValueError('Sampled replay window changed')
    for kind in ROW_FIELDS:
      for field in ROW_FIELDS[kind]:
        if not np.array_equal(before[kind][field], after[kind][field], equal_nan=True):
          raise ValueError(f'Trace changed {window["id"]}/{kind}/{field}')
    audit = Path(manifest['audit'])
    for relative, expected in window['audit_hashes'].items():
      if sha256(audit / relative) != expected:
        raise ValueError('Native negative-control input changed')
    with np.load(audit / f'{window["route"]}.native.npz', allow_pickle=False) as saved:
      native = columns(saved['frames'], NATIVE_FIELDS)
    positions = exact_alignment(native['t'], before['can']['t'])
    clean = native_clean(native)[positions]
    result = {
      'id': window['id'],
      'group': window['group'],
      'reason': window['reason'],
      'exact_equal': True,
      'coverage': before_meta['coverage'],
      'trace': after_meta['trace'],
      'solver_failures': after_meta['solver_failures'],
      'can': compare_can(before['can'], after['can'], before['can']['t'], clean),
    }
    if after_meta['trace']['attempted'] != after_meta['coverage']['plans']:
      raise ValueError('Trace sample attempts did not cover all measured plans')
    trace = after_meta['trace']
    if trace['disabled'] or trace['failures'] or trace['samples'] + trace['dropped'] != trace['attempted']:
      raise ValueError('Incomplete diagnostic capture')
    results.append(result)
  save_json(output / 'evaluation.json', results)
  print(json.dumps({'windows': len(results), 'exact_equal': all(result['exact_equal'] for result in results), 'trace_event': TRACE_EVENT}))
  return results


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('command', choices=('run', 'evaluate'))
  parser.add_argument('--selection', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--workers', type=int, default=1)
  parser.add_argument('--mode', choices=('both', 'off', 'on'), default='both')
  arguments = parser.parse_args()
  selection = json.loads(arguments.selection.read_text())
  if arguments.command == 'evaluate':
    evaluate(selection, arguments.output)
    return
  corpus = Path(selection['corpus'])
  if sha256(corpus / 'manifest.json') != selection['manifest_sha256']:
    raise ValueError('Corpus manifest changed')
  if arguments.workers < 1:
    parser.error('--workers must be positive')
  arguments.output.mkdir(parents=True, exist_ok=True)
  modes = (False, True) if arguments.mode == 'both' else (arguments.mode == 'on',)
  jobs = [(corpus, arguments.output, window, enabled) for window in selection['windows'] for enabled in modes]
  if arguments.workers == 1:
    for job in jobs:
      run_window(job)
  else:
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
      list(pool.map(run_window, jobs))


if __name__ == '__main__':
  main()
