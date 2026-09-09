"""Fixed native-log sampling for short, paired cruise-smoothing experiments."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import time

import numpy as np

from openpilot.selfdrive.controls.tests.evaluate_pedal_full_routes import (
  columns,
  compare_can,
  delta_stats,
  exact_alignment,
  intervals,
  native_clean,
  normal_mask,
  switch_analysis,
  time_runs,
  unmatched_events,
)
from openpilot.selfdrive.controls.tests.replay_pedal_full_routes import NATIVE_FIELDS, ROW_FIELDS, NumericSink, block_messages
from openpilot.selfdrive.controls.tests.replay_pedal_switches import replay_window, save_json, sha256
from openpilot.tools.lib.logreader import LogReader


MODES = ('off', 'brake_dwell', 'median')


def select_windows(corpus, output):
  manifest = json.loads((corpus / 'manifest.json').read_text())
  audit = Path(manifest['audit'])
  native, masks, selected = {}, {}, []
  for record in manifest['routes']:
    route = record['route']
    with np.load(audit / f'{route}.native.npz', allow_pickle=False) as saved:
      native[route] = columns(saved['frames'], NATIVE_FIELDS)
    events = json.loads((audit / f'{route}.events.json').read_text())
    masks[route] = normal_mask(native[route]['t'], native_clean(native[route]), events['events'] + events['raw_events'])
    masks[route] &= native[route]['t'] >= 65.0

  def add(route, start, end, group, reason):
    record = next(item for item in manifest['routes'] if item['route'] == route)
    warmup = max(0.0, start - 60.0)
    candidates = [
      block
      for block in record['blocks']
      if block[0]['first_mono'] / 1e9 - record['origin'] <= warmup and block[-1]['last_mono'] / 1e9 - record['origin'] >= end
    ]
    if not candidates:
      return False
    block = candidates[0]
    segments = [
      segment
      for segment in block
      if segment['last_mono'] / 1e9 - record['origin'] >= warmup - 0.1 and segment['first_mono'] / 1e9 - record['origin'] <= end + 0.3
    ]
    window = {
      'id': f'{len(selected):03d}',
      'route': route,
      'start': float(start),
      'end': float(end),
      'warmup': warmup,
      'group': group,
      'reason': reason,
      'origin': record['origin'],
      'commit': record['commit'],
      'parameter_segment': block[0],
      'segments': segments,
      'audit_hashes': record['audit_hashes'],
    }
    selected.append(window)
    masks[route] &= (native[route]['t'] < start - 1.0) | (native[route]['t'] > end + 1.0)
    return True

  for route, timestamp in [('0000002c--0545fb49cb', 349.94), ('00000050--076fbccae6', 3937.55), ('00000051--4c602a02a6', 616.5)]:
    add(route, timestamp - 10.0, timestamp + 10.0, 'normal', 'previous_disabled_experiment_raw_pulse')
  for record in manifest['routes']:
    route = record['route']
    if record['original_events'] or not record['native_enabled_s'].get('selfdriveState', 0):
      continue
    spans = time_runs(native[route]['t'], masks[route])
    if spans:
      span = max(spans, key=lambda value: value['seconds'])
      duration = min(75.0, span['seconds'])
      start = span['start'] + max(0.0, (span['seconds'] - duration) / 2)
      add(route, start, start + duration, 'normal', 'originally_nonoscillating_engaged_route')
  conditions = {
    'low_speed': lambda data: data['v_ego'] < 10 / 3.6,
    'urban': lambda data: (data['v_ego'] >= 10 / 3.6) & (data['v_ego'] < 10),
    'mid_speed': lambda data: (data['v_ego'] >= 10) & (data['v_ego'] < 20),
    'high_speed': lambda data: data['v_ego'] >= 20,
    'no_lead': lambda data: (data['lead1'] == 0) & (data['lead2'] == 0),
    'e2e': lambda data: (data['experimental'] == 1) & (data['dec'] == 1),
    'model_braking': lambda data: (data['model_accel'] < -0.5) | (data['model_stop'] == 1),
    'cruise_lead': lambda data: (data['sp_cruise'] == 1) & ((data['lead1'] == 1) | (data['lead2'] == 1)),
  }
  for reason, condition in conditions.items():
    choices = [(route, span) for route, data in native.items() for span in time_runs(data['t'], masks[route] & condition(data))]
    choices.sort(key=lambda value: value[1]['seconds'], reverse=True)
    for route, span in choices:
      duration = min(90.0, span['seconds'])
      start = span['start'] + max(0.0, (span['seconds'] - duration) / 2)
      if add(route, start, start + duration, 'normal', reason):
        break
  original = json.loads((audit / 'events-enriched.json').read_text())
  by_group = {}
  for event in original:
    if event['category'] == 'cruise_with_lead':
      by_group.setdefault((event['frequency_group'], event['duration_group']), []).append(event)
  chosen = []
  for values in by_group.values():
    chosen.extend(sorted(values, key=lambda event: (-event['switches'], event['route']))[:2])
  for event in sorted(chosen, key=lambda event: (event['route'], event['start'])):
    add(event['route'], event['start'] - 5.0, event['end'] + 5.0, 'oscillation', f'{event["frequency_group"]}/{event["duration_group"]}')
  for route, start, end, reason in [
    ('00000027--b204e98ece', 1165.0, 1185.0, 'model_gate_brake'),
    ('0000004e--b6f1c9bc4e', 635.0, 650.0, 'manual_brake_edge'),
    ('00000050--076fbccae6', 1669.0, 1678.06, 'strong_brake'),
    ('0000004e--b6f1c9bc4e', 2280.0, 2305.0, 'cut_in'),
    ('00000047--f3ea9093b8', 3117.0, 3140.0, 'low_speed_stop'),
    ('00000050--076fbccae6', 3080.0, 3110.0, 'lead_brake'),
  ]:
    add(route, start, end, 'risk', reason)
  output.mkdir(parents=True, exist_ok=True)
  save_json(
    output / 'selection.json',
    {
      'corpus': str(corpus),
      'manifest_sha256': sha256(corpus / 'manifest.json'),
      'method': 'Original native masks and fixed historical sentinels; no candidate outcome selection',
      'windows': selected,
    },
  )
  print(
    'selected',
    len(selected),
    'windows',
    Counter(window['group'] for window in selected),
    'measured minutes',
    sum(window['end'] - window['start'] for window in selected) / 60,
  )


def window_messages(corpus, window, sink):
  seen = set()
  segment = window['parameter_segment']
  cache = corpus / segment['cache']
  if sha256(cache) != segment['sha256']:
    raise ValueError('Changed parameter segment')
  for message in LogReader([str(cache)], sort_by_time=True):
    if message.which() in ('carParams', 'carParamsSP') and message.which() not in seen:
      if message.logMonoTime / 1e9 - window['origin'] > window['warmup']:
        raise ValueError('Parameter arrival is later than sampled warmup')
      seen.add(message.which())
      yield message.as_builder().as_reader()
    if len(seen) == 2:
      break
  if len(seen) != 2:
    raise ValueError('Missing sample parameters')
  for message in block_messages(corpus, window['segments'], sink):
    if message.which() in ('carParams', 'carParamsSP', 'initData'):
      continue
    timestamp = message.logMonoTime / 1e9 - window['origin']
    if timestamp < window['warmup']:
      continue
    if timestamp > window['end'] + 0.25:
      break
    yield message


def run_window(job):
  corpus, output, window, mode, warmup = job
  if mode not in MODES:
    raise ValueError('Unknown sampled cruise smoothing mode')
  if warmup is not None and (not math.isfinite(warmup) or warmup < 0):
    raise ValueError('Sampled warmup must be finite and nonnegative')
  destination = output / mode
  destination.mkdir(parents=True, exist_ok=True)
  sink = NumericSink(destination, window['id'])
  local = dict(window, cache=window['id'])
  if warmup is not None:
    local['warmup'] = max(0.0, local['start'] - warmup)
    manifest = json.loads((corpus / 'manifest.json').read_text())
    record = next(item for item in manifest['routes'] if item['route'] == window['route'])
    block = next(block for block in record['blocks'] if block[0]['segment'] == window['parameter_segment']['segment'])
    if block[0]['first_mono'] / 1e9 - record['origin'] > local['warmup']:
      raise ValueError('Requested warmup crosses a missing log interval')
    local['segments'] = [
      segment
      for segment in block
      if segment['last_mono'] / 1e9 - record['origin'] >= local['warmup'] - 0.1 and segment['first_mono'] / 1e9 - record['origin'] <= local['end'] + 0.3
    ]
  start = time.monotonic()
  result = replay_window(None, local, messages=window_messages(corpus, local, sink), row_sink=sink, strict=True, experimental_cruise_smoothing=mode)
  sink.flush()
  for field in ('plan_rows', 'control_rows', 'can_rows', 'planner_error', 'switches'):
    result.pop(field)
  result.update(mode=mode, chunks=sink.chunks, fields=ROW_FIELDS, elapsed_s=time.monotonic() - start)
  save_json(destination / f'{window["id"]}.json', result)
  print(mode, window['id'], window['group'], result['coverage'], flush=True)


def load_window(output, mode, identifier):
  root = output / mode
  metadata = json.loads((root / f'{identifier}.json').read_text())
  if metadata['mode'] != mode or metadata['window']['id'] != identifier or metadata['fields'] != {kind: list(fields) for kind, fields in ROW_FIELDS.items()}:
    raise ValueError('Mismatched sampled replay identity or schema')
  for expected, matched in (('plans', 'matched_plans'), ('controls', 'matched_controls'), ('sendcan', 'matched_sendcan')):
    if not metadata['coverage'][expected] or metadata['coverage'][expected] != metadata['coverage'][matched]:
      raise ValueError('Incomplete sampled replay coverage')
  pieces = {kind: [] for kind in ROW_FIELDS}
  for chunk in metadata['chunks']:
    if sha256(root / chunk['file']) != chunk['sha256']:
      raise ValueError('Changed sampled replay results')
    with np.load(root / chunk['file'], allow_pickle=False) as saved:
      for kind in pieces:
        if saved[kind].shape != (chunk['rows'][kind], len(ROW_FIELDS[kind])):
          raise ValueError('Mismatched sampled replay array shape')
        pieces[kind].append(saved[kind])
  for kind, expected in (('plan', 'plans'), ('control', 'controls'), ('can', 'sendcan')):
    if sum(len(piece) for piece in pieces[kind]) != metadata['coverage'][expected]:
      raise ValueError('Missing sampled replay output rows')
  return metadata, {kind: columns(np.concatenate(values), ROW_FIELDS[kind]) for kind, values in pieces.items()}


def evaluate(selection, output):
  corpus = Path(selection['corpus'])
  manifest = json.loads((corpus / 'manifest.json').read_text())
  if sha256(corpus / 'manifest.json') != selection['manifest_sha256']:
    raise ValueError('Changed source corpus')
  results = []
  for window in selection['windows']:
    route = window['route']
    for relative, expected in window['audit_hashes'].items():
      if sha256(Path(manifest['audit']) / relative) != expected:
        raise ValueError('Changed native cohort')
    with np.load(Path(manifest['audit']) / f'{route}.native.npz', allow_pickle=False) as saved:
      native = columns(saved['frames'], NATIVE_FIELDS)
    native_mask = native_clean(native)
    if window['group'] == 'normal':
      events = json.loads((Path(manifest['audit']) / f'{route}.events.json').read_text())
      native_mask = normal_mask(native['t'], native_mask, events['events'] + events['raw_events'])
    measured = (native['t'] >= window['start']) & (native['t'] <= window['end']) & native_mask
    arms = {mode: load_window(output, mode, window['id']) for mode in MODES}
    times = arms['off'][1]['can']['t']
    indices = exact_alignment(native['t'], times)
    available = np.zeros(len(native['t']), dtype=bool)
    available[indices] = True
    if np.any(measured & ~available):
      raise ValueError(f'Missing clean sampled CAN frames: {window["id"]}')
    selected = measured[indices]
    result = {
      'id': window['id'],
      'route': route,
      'group': window['group'],
      'reason': window['reason'],
      'start': window['start'],
      'end': window['end'],
      'seconds': float(np.sum(intervals(times)[selected])),
      'modes': {},
      'comparisons': {},
    }
    for mode, (metadata, values) in arms.items():
      for kind in ROW_FIELDS:
        if not np.array_equal(values[kind]['mono'], arms['off'][1][kind]['mono']):
          raise ValueError('Sampled arms have unequal frame alignment')
      analyses = {
        label: switch_analysis(times, values['can']['can_brake'], values['can']['can_gas'], selected, dwell)
        for label, dwell in [('raw', 0.0), ('debounced', 0.15)]
      }
      result['modes'][mode] = {
        'raw': len(analyses['raw']['changes']),
        'debounced': len(analyses['debounced']['changes']),
        'raw_events': analyses['raw']['events'],
        'events': analyses['debounced']['events'],
        'solver_failures': metadata['solver_failures'],
        'conflicts': int(np.sum(selected & (values['can']['can_brake'] == 1) & (values['can']['can_gas'] > 0))),
      }
      if mode != 'off':
        comparison = compare_can(arms['off'][1]['can'], values['can'], times, selected)
        for label in ('events', 'raw_events'):
          comparison['new_' + label] = unmatched_events(result['modes'][mode][label], result['modes']['off'][label])
        original, plan = arms['off'][1]['plan'], values['plan']
        valid = (original['automatic'] == 1) & (original['valid'] == 1)
        comparison['plan'] = delta_stats(original['accel'], plan['accel'], valid)
        comparison['stop_changes'] = int(np.sum(valid & (original['stop'] != plan['stop'])))
        comparison['fcw_changes'] = int(np.sum(valid & (original['fcw'] != plan['fcw'])))
        result['comparisons'][mode] = comparison
    results.append(result)
  summary = {}
  for group in ('normal', 'oscillation', 'risk'):
    windows = [result for result in results if result['group'] == group]
    summary[group] = {'windows': len(windows), 'seconds': sum(window['seconds'] for window in windows), 'modes': {}, 'comparisons': {}}
    for mode in MODES:
      summary[group]['modes'][mode] = {
        field: sum(window['modes'][mode][field] for window in windows) for field in ('raw', 'debounced', 'solver_failures', 'conflicts')
      }
      for field in ('events', 'raw_events'):
        summary[group]['modes'][mode][field] = sum(len(window['modes'][mode][field]) for window in windows)
      if mode != 'off':
        metrics = [window['comparisons'][mode] for window in windows]
        comparison = {
          field: sum(metric[field] for metric in metrics)
          for field in (
            'extra_brake_frames',
            'extra_brake_seconds',
            'removed_brake_frames',
            'removed_brake_seconds',
            'weaker_strong_brake_frames',
            'stop_changes',
            'fcw_changes',
          )
        }
        comparison['maximum_strong_brake_reduction'] = max((metric['maximum_strong_brake_reduction'] for metric in metrics), default=0.0)
        for field in ('new_events', 'new_raw_events'):
          comparison[field] = [dict(event, id=window['id'], route=window['route']) for window in windows for event in window['comparisons'][mode][field]]
        comparison['worse_debounced_windows'] = [window['id'] for window in windows if window['modes'][mode]['debounced'] > window['modes']['off']['debounced']]
        comparison['improved_debounced_windows'] = [
          window['id'] for window in windows if window['modes'][mode]['debounced'] < window['modes']['off']['debounced']
        ]
        summary[group]['comparisons'][mode] = comparison
  save_json(output / 'evaluation.json', {'summary': summary, 'windows': results, 'selection_sha256': sha256(output / 'selection.json')})
  print(json.dumps(summary, indent=2))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('action', choices=('select', 'run', 'evaluate'))
  parser.add_argument('--corpus', type=Path)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--workers', type=int, default=3)
  arguments = parser.parse_args()
  if arguments.action == 'select':
    select_windows(arguments.corpus, arguments.output)
    return
  selection = json.loads((arguments.output / 'selection.json').read_text())
  if arguments.action == 'evaluate':
    evaluate(selection, arguments.output)
    return
  jobs = [(Path(selection['corpus']), arguments.output, window, mode, None) for window in selection['windows'] for mode in MODES]
  with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
    list(pool.map(run_window, jobs))


if __name__ == '__main__':
  main()
