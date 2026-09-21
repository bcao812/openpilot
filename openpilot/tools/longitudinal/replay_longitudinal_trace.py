"""Verify trace off/on against identical fixed short log windows in a host-built copy."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import wraps
import json
from pathlib import Path
import re
import time
from unittest.mock import patch

import numpy as np

from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.longitudinal_trace import LongitudinalTrace, TRACE_EVENT
from openpilot.tools.longitudinal.evaluate_pedal_full_routes import native_clean, exact_alignment, columns, compare_can
from openpilot.tools.longitudinal.window_replay import load_window, window_messages
from openpilot.tools.longitudinal.replay_pedal_full_routes import NumericSink, ROW_FIELDS, NATIVE_FIELDS
from openpilot.tools.longitudinal.replay_pedal_switches import (
  DIAGNOSTIC_SCHEMA_VERSION, ReplayDiagnostics, replay_window, save_json, selected_mode, sha256,
)


# Selected disk-source identities for this comparison, not native-library or
# loaded-code attestation. Keep path names: the two planner modules share a basename.
SOURCE_PATHS = (
  'openpilot/tools/longitudinal/replay_longitudinal_trace.py',
  'openpilot/tools/longitudinal/replay_pedal_switches.py',
  'openpilot/tools/longitudinal/replay_pedal_full_routes.py',
  'openpilot/tools/longitudinal/evaluate_pedal_full_routes.py',
  'openpilot/tools/longitudinal/window_replay.py',
  'openpilot/tools/longitudinal/replay_longitudinal_regressions.py',
  'openpilot/tools/lib/logreader.py',
  'openpilot/common/longitudinal_logging.py',
  'openpilot/common/logging_extra.py',
  'openpilot/common/swaglog.py',
  'openpilot/common/constants.py',
  'openpilot/common/pid.py',
  'openpilot/common/filter_simple.py',
  'openpilot/selfdrive/controls/lib/longitudinal_trace.py',
  'openpilot/selfdrive/controls/lib/longitudinal_observation.py',
  'openpilot/selfdrive/controls/lib/longitudinal_planner.py',
  'openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py',
  'openpilot/selfdrive/controls/lib/longcontrol.py',
  'openpilot/selfdrive/controls/lib/motion_supervisor.py',
  'openpilot/selfdrive/controls/lib/unified_longitudinal_planner.py',
  'openpilot/selfdrive/controls/lib/longitudinal_mode.py',
  'openpilot/selfdrive/controls/lib/longitudinal_trajectory.py',
  'openpilot/selfdrive/controls/lib/reference_timing.py',
  'openpilot/selfdrive/controls/lib/launch_response.py',
  'openpilot/selfdrive/controls/lib/cruise_response.py',
  'openpilot/selfdrive/controls/lib/drive_helpers.py',
  'openpilot/selfdrive/car/longitudinal_authority.py',
  'openpilot/selfdrive/modeld/constants.py',
  'openpilot/sunnypilot/selfdrive/controls/lib/longitudinal_planner.py',
  'opendbc_repo/opendbc/car/honda/carcontroller.py',
  'opendbc_repo/opendbc/car/honda/hondacan.py',
  'opendbc_repo/opendbc/car/honda/interface.py',
  'opendbc_repo/opendbc/car/honda/values.py',
)


def source_fingerprints():
  root = Path(__file__).resolve().parents[3]
  return {relative: sha256(root / relative) for relative in SOURCE_PATHS}


def validate_sources(metadata):
  hashes = metadata.get('source_hashes')
  if (not isinstance(hashes, dict) or set(hashes) != set(SOURCE_PATHS)
      or any(not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None for value in hashes.values())):
    raise ValueError('Missing or invalid replay source identities')
  return hashes


def validate_execution(metadata):
  diagnostics = metadata.get('diagnostics')
  if (not isinstance(diagnostics, dict) or type(diagnostics.get('diagnostic_schema')) is not int
      or diagnostics['diagnostic_schema'] != DIAGNOSTIC_SCHEMA_VERSION
      or diagnostics.get('status') != 'COMPLETED' or diagnostics.get('initialized') is not True
      or diagnostics.get('native_supported') is not True):
    raise ValueError('Incomplete replay execution diagnostics')
  coverage = metadata.get('coverage')
  fields = ('plans', 'matched_plans', 'controls', 'matched_controls', 'sendcan', 'matched_sendcan')
  if (not isinstance(coverage, dict) or any(type(coverage.get(name)) is not int or coverage[name] <= 0 for name in fields)
      or diagnostics.get('coverage') != coverage):
    raise ValueError('Missing or invalid replay score coverage')
  if any(coverage[expected] != coverage[matched] for expected, matched in zip(fields[::2], fields[1::2], strict=True)):
    raise ValueError('Incomplete replay score coverage')
  counts = diagnostics.get('counts')
  if not isinstance(counts, dict) or set(counts) != set(ReplayDiagnostics.LAYERS):
    raise ValueError('Missing replay runtime counts')
  for layer, calls in counts.items():
    if not isinstance(calls, dict) or set(calls) != {'attempted', 'completed'}:
      raise ValueError(f'Missing replay {layer} execution counts')
    for values in calls.values():
      if (not isinstance(values, dict) or set(values) != {'total', *ReplayDiagnostics.REGIONS}
          or any(type(value) is not int or value < 0 for value in values.values())
          or values['total'] != sum(values[region] for region in ReplayDiagnostics.REGIONS)):
        raise ValueError(f'Invalid replay {layer} execution counts')
    if calls['attempted'] != calls['completed']:
      raise ValueError(f'Incomplete replay {layer} execution')
  for layer, measured in (('planner', 'plans'), ('longcontrol', 'controls')):
    if counts[layer]['completed']['score'] != coverage[measured]:
      raise ValueError(f'Replay {layer} score count does not match coverage')
  # CarController executes at every sendcan cycle; coverage selects ACC frames.
  if counts['carcontroller']['completed']['score'] < coverage['sendcan']:
    raise ValueError('Replay carcontroller score count does not cover CAN output')
  return counts


def validate_trace(trace, attempted):
  if (not isinstance(trace, dict)
      or any(type(trace.get(name)) is not int or trace[name] < 0 for name in ('attempted', 'samples', 'dropped', 'failures'))
      or trace.get('disabled') is not False or trace['failures'] or trace['attempted'] != attempted
      or trace['samples'] + trace['dropped'] != attempted):
    raise ValueError('Incomplete diagnostic capture')


def run_window(arguments):
  corpus, destination, window, enabled, *options = arguments
  unified = bool(options and options[0])
  supervised = bool(len(options) > 1 and options[1])
  selected_mode(unified, supervised)
  mode = 'on' if enabled else 'off'
  output = destination / mode
  output.mkdir(parents=True, exist_ok=True)
  original_init = LongitudinalPlanner.__init__
  records = []
  emitted = {'batches': 0, 'samples': 0, 'bytes': 0}
  source_hashes = source_fingerprints()

  def emit(event, **payload):
    encoded = json.dumps({'msg': dict(payload, event=event)}, allow_nan=False)
    emitted['batches'] += 1
    emitted['samples'] += len(payload['samples'])
    emitted['bytes'] += len(encoded)

  @wraps(original_init)
  def initialize(planner, *args, **kwargs):
    recorder = LongitudinalTrace(emit=emit) if enabled else None
    if recorder is not None:
      if not recorder.start():
        raise RuntimeError('Diagnostic writer did not start')
      records.append(recorder)
    original_init(planner, *args, trace=recorder, **kwargs)

  sink = NumericSink(output, window['id'])
  started = time.monotonic()
  try:
    with patch.object(LongitudinalPlanner, '__init__', new=initialize):
      result = replay_window(None, dict(window, cache=window['id']), messages=window_messages(corpus, window, sink), row_sink=sink, strict=True,
                             unified=unified, supervised=supervised)
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
  counts = validate_execution(result)
  validate_trace(trace, counts['planner']['attempted']['total'] if enabled else 0)
  if source_fingerprints() != source_hashes:
    raise ValueError('Replay source changed during execution')
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
    if validate_sources(before_meta) != validate_sources(after_meta):
      raise ValueError('Trace comparison changed replay source identities')
    before_counts, after_counts = validate_execution(before_meta), validate_execution(after_meta)
    if before_counts != after_counts or before_meta['coverage'] != after_meta['coverage']:
      raise ValueError('Trace comparison changed replay execution coverage')
    validate_trace(before_meta.get('trace'), 0)
    validate_trace(after_meta.get('trace'), after_counts['planner']['attempted']['total'])
    for name in ('requested_mode', 'planner_mode', 'controller_mode'):
      modes = []
      for metadata in (before_meta, after_meta):
        capabilities = metadata.get('capabilities')
        if not isinstance(capabilities, dict) or type(capabilities.get(name)) is not int or capabilities[name] not in (0, 1, 2):
          raise ValueError(f'Missing or invalid longitudinal {name}')
        modes.append(capabilities[name])
      if modes[0] != modes[1]:
        raise ValueError(f'Trace comparison changed longitudinal {name}')
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
      'execution_counts': after_counts,
      'source_hashes': after_meta['source_hashes'],
      'trace': after_meta['trace'],
      'solver_failures': after_meta['solver_failures'],
      'can': compare_can(before['can'], after['can'], before['can']['t'], clean),
    }
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
  motion_modes = parser.add_mutually_exclusive_group()
  motion_modes.add_argument('--unified', action='store_true')
  motion_modes.add_argument('--supervised', action='store_true', help='Replay a historical source tree implementing SUPERVISED')
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
  jobs = [(corpus, arguments.output, window, enabled, arguments.unified, arguments.supervised)
          for window in selection['windows'] for enabled in modes]
  if arguments.workers == 1:
    for job in jobs:
      run_window(job)
  else:
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
      list(pool.map(run_window, jobs))


if __name__ == '__main__':
  main()
