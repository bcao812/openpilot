#!/usr/bin/env python3
"""Evaluate full-route arms on fixed native negative controls, never on candidate-selected exposure."""

import argparse
from collections import Counter
import hashlib
from itertools import pairwise
import json
from pathlib import Path

import numpy as np

from openpilot.selfdrive.controls.tests.replay_pedal_full_routes import NATIVE_FIELDS, ROW_FIELDS
from openpilot.selfdrive.controls.tests.replay_pedal_switches import save_json, sha256


def columns(values, fields):
  return dict(zip(fields, values.T, strict=True))


def intervals(times):
  differences = np.diff(times, append=times[-1] if len(times) else 0)
  return np.where((differences > 0) & (differences <= 0.15), differences, 0.0)


def native_clean(native):
  clean = np.ones(len(native['t']), dtype=bool)
  for service in ('cs', 'cc', 'sd'):
    clean &= (native[service + '_valid'] == 1) & (native[service + '_age'] >= 0) & (native[service + '_age'] <= 0.15)
  return (clean & (native['enabled'] == 1) & (native['cc_enabled'] == 1) & (native['long_active'] == 1) & (native['long_state'] > 0)
          & (native['gas_pressed'] == 0) & (native['brake_pressed'] == 0) & (native['override'] == 0)
          & (native['can_valid'] == 1) & (native['can_on'] == 5))


def normal_mask(times, clean, events, margin=10.0):
  normal = clean.copy()
  for event in events:
    normal &= (times < event['start'] - margin) | (times > event['end'] + margin)
  return normal


def switch_analysis(times, brake, gas, clean, dwell=0.15):
  elapsed = intervals(times)
  labels = np.zeros(len(times), dtype=int)
  labels[(brake == 1) & (gas <= 0)] = -1
  labels[(brake == 0) & (gas > 0)] = 1
  labels[~clean | ((brake == 1) & (gas > 0))] = 9
  if not len(labels):
    return {'changes': [], 'events': []}
  boundaries = np.r_[0, np.flatnonzero((labels[1:] != labels[:-1]) | (elapsed[:-1] == 0)) + 1, len(labels)]
  phases, block = [], 0
  for start, stop in pairwise(boundaries):
    side = int(labels[start])
    duration = times[stop - 1] + elapsed[stop - 1] - times[start]
    if start and elapsed[start - 1] == 0:
      block += 1
    if side == 9 or (side == 0 and duration > 0.1):
      block += 1
    elif side != 0 and duration + 1e-8 >= dwell:
      phases.append({'index': int(start), 't': float(times[start]), 'duration': float(duration), 'block': block, 'side': side})
  changes = [after for before, after in pairwise(phases) if before['block'] == after['block'] and before['side'] != after['side']]
  bursts, current = [], []
  for change in changes:
    if current and (change['block'] != current[-1]['block'] or change['t'] - current[-1]['t'] > 5.0):
      if len(current) >= 4:
        bursts.append(current)
      current = []
    current.append(change)
  if len(current) >= 4:
    bursts.append(current)
  events = []
  for burst in bursts:
    duration = burst[-1]['t'] - burst[0]['t']
    events.append({'start': burst[0]['t'], 'end': burst[-1]['t'], 'duration': duration, 'switches': len(burst),
                   'switches_per_s': (len(burst) - 1) / duration, 'block': burst[0]['block']})
  return {'changes': changes, 'events': events}


def unmatched_events(candidate, baseline, margin=0.15):
  return [event for event in candidate if not any(event['start'] <= previous['end'] + margin
                                                 and event['end'] >= previous['start'] - margin for previous in baseline)]


def load_arm(root, route, manifest_hash):
  metadata = json.loads((root / f'{route}.json').read_text())
  if metadata['manifest_sha256'] != manifest_hash or metadata['fields'] != {key: list(value) for key, value in ROW_FIELDS.items()}:
    raise ValueError(f'Mismatched replay corpus or fields: {root}/{route}')
  pieces = {kind: [] for kind in ROW_FIELDS}
  for chunk in metadata['chunks']:
    path = root / chunk['file']
    if sha256(path) != chunk['sha256']:
      raise ValueError(f'Changed replay results: {path}')
    with np.load(path, allow_pickle=False) as saved:
      for kind in pieces:
        pieces[kind].append(saved[kind])
  values = {kind: columns(np.concatenate(rows) if rows else np.empty((0, len(ROW_FIELDS[kind]))), ROW_FIELDS[kind])
            for kind, rows in pieces.items()}
  return metadata, values


def exact_alignment(reference, actual, tolerance=1e-6):
  locations = np.searchsorted(reference, actual)
  before = np.clip(locations - 1, 0, max(0, len(reference) - 1))
  after = np.clip(locations, 0, max(0, len(reference) - 1))
  if not len(reference):
    if len(actual):
      raise ValueError('Cannot align nonempty output to empty reference')
    return locations
  locations = np.where(np.abs(reference[before] - actual) < np.abs(reference[after] - actual), before, after)
  if np.any(np.abs(reference[locations] - actual) > tolerance) or len(np.unique(locations)) != len(locations):
    raise ValueError('Nonunique or missing native timestamp alignment')
  return locations


def delta_stats(before, after, selected):
  differences = after[selected] - before[selected]
  if not len(differences):
    return {'frames': 0, 'changed': 0, 'minimum': None, 'maximum': None, 'p95_abs': None}
  return {'frames': len(differences), 'changed': int(np.count_nonzero(np.abs(differences) > 1e-6)),
          'minimum': float(np.min(differences)), 'maximum': float(np.max(differences)),
          'p95_abs': float(np.percentile(np.abs(differences), 95))}


def time_runs(times, selected):
  elapsed = intervals(times)
  boundaries = np.r_[0, np.flatnonzero((selected[1:] != selected[:-1]) | (elapsed[:-1] == 0)) + 1, len(times)]
  return [{'start': float(times[start]), 'end': float(times[stop - 1] + elapsed[stop - 1]),
           'seconds': float(np.sum(elapsed[start:stop])), 'frames': int(stop - start)}
          for start, stop in pairwise(boundaries) if stop > start and selected[start]]


def compare_can(before, after, times, selected):
  elapsed = intervals(times)
  added = selected & (before['can_brake'] == 0) & (after['can_brake'] == 1)
  removed = selected & (before['can_brake'] == 1) & (after['can_brake'] == 0)
  weaker_strong = selected & (before['can_accel'] <= -1.0) & (after['can_accel'] > before['can_accel'] + 1e-6)
  added_values = after['can_accel'][added]
  return {'accel': delta_stats(before['can_accel'], after['can_accel'], selected),
          'extra_brake_frames': int(np.sum(added)), 'extra_brake_seconds': float(np.sum(elapsed[added])),
          'removed_brake_frames': int(np.sum(removed)), 'removed_brake_seconds': float(np.sum(elapsed[removed])),
          'weaker_strong_brake_frames': int(np.sum(weaker_strong)),
          'maximum_strong_brake_reduction': float(np.max(after['can_accel'][weaker_strong] - before['can_accel'][weaker_strong]))
                                           if np.any(weaker_strong) else 0.0,
          'extra_brake_minimum_accel': float(np.min(added_values)) if len(added_values) else None,
          'extra_brake_maximum_accel': float(np.max(added_values)) if len(added_values) else None,
          'control_on_changes': int(np.sum(selected & (before['can_on'] != after['can_on']))),
          'standstill_changes': int(np.sum(selected & (before['can_stop'] != after['can_stop']))),
          'extra_brake_runs': time_runs(times, added)}


def describe_events(events, native):
  described = []
  for event in events:
    selected = (native['t'] >= event['start']) & (native['t'] <= event['end'])
    speed = float(np.median(native['v_ego'][selected]) * 3.6)
    leads = float(np.mean((native['lead1'][selected] == 1) | (native['lead2'][selected] == 1)))
    described.append(dict(event, speed_kph=speed, lead_fraction=leads,
                          frequency_group=('0.2-0.5', '0.5-1', '1-2', '>=2')[int(np.searchsorted([0.5, 1, 2], event['switches_per_s'], side='right'))],
                          duration_group=('<5', '5-15', '15-30', '>=30')[int(np.searchsorted([5, 15, 30], event['duration'], side='right'))]))
  return described


def reentry_mask(times, enabled, duration=2.0):
  output = np.zeros(len(times), dtype=bool)
  entries = np.flatnonzero((enabled == 1) & np.r_[True, enabled[:-1] != 1])
  for index in entries:
    output |= (times >= times[index]) & (times <= times[index] + duration)
  return output


def evaluate_route(record, audit, arm_roots, manifest_hash):
  route = record['route']
  for name, expected in record['audit_hashes'].items():
    if sha256(audit / name) != expected:
      raise ValueError(f'Changed native audit: {name}')
  with np.load(audit / f'{route}.native.npz', allow_pickle=False) as saved:
    native = columns(saved['frames'], NATIVE_FIELDS)
  original = json.loads((audit / f'{route}.events.json').read_text())
  times, elapsed = native['t'], intervals(native['t'])
  clean = native_clean(native)
  normal = normal_mask(times, clean, original['events'] + original['raw_events'])
  baseline_native = switch_analysis(times, native['can_brake'], native['can_gas'], clean)
  native_raw = switch_analysis(times, native['can_brake'], native['can_gas'], clean, dwell=0.0)
  if len(baseline_native['events']) != len(original['events']) or len(baseline_native['changes']) != original['debounced_switches']:
    raise ValueError(f'Native event definition drift: {route}')
  if len(native_raw['changes']) != original['raw_switches']:
    raise ValueError(f'Native raw definition drift: {route}')
  metadata, arms, can = {}, {}, {'native': native}
  common = np.ones(len(times), dtype=bool)
  for name, root in arm_roots.items():
    metadata[name], arms[name] = load_arm(root, route, manifest_hash)
    matched = exact_alignment(times, arms[name]['can']['t'])
    present = np.zeros(len(times), dtype=bool)
    present[matched] = True
    common &= present
    for recorded_field, field in [('recorded_accel', 'can_accel'), ('recorded_brake', 'can_brake'), ('recorded_gas', 'can_gas')]:
      if not np.allclose(native[field][matched], arms[name]['can'][recorded_field], rtol=0, atol=1e-10):
        raise ValueError(f'CAN decode alignment mismatch: {route}/{name}/{field}')
    can[name] = {}
    for field in ('can_brake', 'can_gas', 'can_accel', 'can_on', 'can_stop'):
      can[name][field] = np.full(len(times), np.nan)
      can[name][field][matched] = arms[name]['can'][field]
  for name, values in arms.items():
    if name not in ('baseline', 'default', 'experimental'):
      continue
    for kind in ROW_FIELDS:
      if not np.array_equal(values[kind]['mono'], arms['baseline'][kind]['mono']):
        raise ValueError(f'Three-arm alignment mismatch: {route}/{name}/{kind}')
  exposure = {'clean': clean & common, 'normal': normal & common}
  result = {'route': route, 'commit': record['commit'], 'original_events': len(original['events']),
            'native_enabled_s': record['native_enabled_s'], 'clean_s': float(np.sum(elapsed[clean])),
            'normal_s': float(np.sum(elapsed[normal])), 'paired_clean_s': float(np.sum(elapsed[exposure['clean']])),
            'paired_normal_s': float(np.sum(elapsed[exposure['normal']])),
            'missing_clean_s': float(np.sum(elapsed[clean & ~common])), 'missing_clean_frames': int(np.sum(clean & ~common)),
            'normal_mask_sha256': hashlib.sha256(normal.tobytes()).hexdigest(),
            'coverage': {name: value['blocks'] for name, value in metadata.items()}, 'metrics': {}, 'comparisons': {}, 'strata': {}}
  for scope, selected in exposure.items():
    result['metrics'][scope] = {}
    for name, values in can.items():
      analyses = {key: switch_analysis(times, values['can_brake'], values['can_gas'], selected, dwell)
                  for key, dwell in [('raw', 0.0), ('debounced', 0.15)]}
      result['metrics'][scope][name] = {'raw': len(analyses['raw']['changes']), 'debounced': len(analyses['debounced']['changes']),
                                       'events': describe_events(analyses['debounced']['events'], native),
                                       'conflicts': int(np.sum(selected & (values['can_brake'] == 1) & (values['can_gas'] > 0))),
                                       'outside_accel_limits': int(np.sum(selected & ((values['can_accel'] < -3.5) | (values['can_accel'] > 2.0))))}
    for title, before, after in [('default_vs_baseline', 'baseline', 'default'), ('experimental_vs_default', 'default', 'experimental')]:
      comparison = compare_can(can[before], can[after], times, selected)
      comparison['new_events'] = unmatched_events(result['metrics'][scope][after]['events'], result['metrics'][scope][before]['events'])
      comparison['resolved_events'] = unmatched_events(result['metrics'][scope][before]['events'], result['metrics'][scope][after]['events'])
      result['comparisons'][f'{scope}/{title}'] = comparison
  strata = {
    'speed_lt10kph': native['v_ego'] * 3.6 < 10,
    'speed_10to36kph': (native['v_ego'] * 3.6 >= 10) & (native['v_ego'] < 10),
    'speed_36to72kph': (native['v_ego'] >= 10) & (native['v_ego'] < 20),
    'speed_ge72kph': native['v_ego'] >= 20,
    'lead_present': (native['lead1'] == 1) | (native['lead2'] == 1),
    'no_lead': (native['lead1'] == 0) & (native['lead2'] == 0),
    'acc': (native['experimental'] == 0) | (native['dec'] == 0),
    'e2e': (native['experimental'] == 1) & (native['dec'] == 1),
    'model_braking_or_stop': (native['model_accel'] < -0.5) | (native['model_stop'] == 1),
    'force_decel': native['force_decel'] == 1,
    'following_source': (native['source'] == 1) | (native['source'] == 2),
    'cruise_source': (native['source'] == 0) & (native['sp_cruise'] == 1),
    'first_2s_after_engagement': reentry_mask(times, native['long_active']),
  }
  normal_changes = {name: {key: switch_analysis(times, values['can_brake'], values['can_gas'], exposure['normal'], dwell)['changes']
                           for key, dwell in [('raw', 0.0), ('debounced', 0.15)]} for name, values in can.items()}
  for group, condition in strata.items():
    selected = exposure['normal'] & condition
    result['strata'][group] = {'seconds': float(np.sum(elapsed[selected])),
                               'switches': {name: {key: sum(bool(condition[change['index']]) for change in changes)
                                                   for key, changes in metrics.items()} for name, metrics in normal_changes.items()},
                               'default': compare_can(can['baseline'], can['default'], times, selected),
                               'experimental': compare_can(can['default'], can['experimental'], times, selected)}
  result['plans'] = {}
  for title, before, after in [('default', 'baseline', 'default'), ('experimental', 'default', 'experimental')]:
    original_plan, candidate = arms[before]['plan'], arms[after]['plan']
    selected = (original_plan['automatic'] == 1) & (original_plan['valid'] == 1)
    result['plans'][title] = {
      'engaged': delta_stats(original_plan['accel'], candidate['accel'], selected),
      'low_speed': delta_stats(original_plan['accel'], candidate['accel'], selected & (original_plan['v_ego'] < 10)),
      'model_risk': delta_stats(original_plan['accel'], candidate['accel'], selected & ((original_plan['model_accel'] < -0.5)
                                                                                     | (original_plan['model_stop'] == 1))),
      'strong_brake': delta_stats(original_plan['accel'], candidate['accel'], selected & (original_plan['accel'] <= -1.0)),
      'stop_changes': int(np.sum(selected & (original_plan['stop'] != candidate['stop']))),
      'fcw_changes': int(np.sum(selected & (original_plan['fcw'] != candidate['fcw']))),
      'nonfinite': int(np.sum(~np.isfinite(candidate['accel']))),
      'solver_failures': int(np.sum(candidate['solver_status'] != 0)),
    }
  result['event_windows'] = []
  all_changes = {name: switch_analysis(times, values['can_brake'], values['can_gas'], exposure['clean'])['changes']
                 for name, values in can.items()}
  for event in original['events']:
    counts = {name: sum(event['start'] - 1e-7 <= change['t'] <= event['end'] + 1e-7 for change in changes)
              for name, changes in all_changes.items()}
    result['event_windows'].append({'start': event['start'], 'end': event['end'], 'category': event['category'],
                                    'frequency_group': event['frequency_group'], 'duration_group': event['duration_group'], 'switches': counts})
  unpaired = clean & ~common
  result['missing_clean_runs'] = time_runs(times, unpaired)
  result['manual'] = {}
  for title, before, after in [('default', 'baseline', 'default'), ('experimental', 'default', 'experimental')]:
    selected = common & ((native['gas_pressed'] == 1) | (native['brake_pressed'] == 1) | (native['override'] == 1))
    result['manual'][title] = compare_can(can[before], can[after], times, selected)
  cold = np.zeros(len(times), dtype=bool)
  for segments in record['blocks']:
    if segments[0]['segment'] != 0:
      first = segments[0]['first_mono'] / 1e9 - record['origin']
      cold |= (times >= first) & (times < first + 60.0)
  for block in metadata['default']['blocks']:
    for reset in block['resets']:
      if 'after' in reset:
        cold |= (times >= reset['after']) & (times < reset['after'] + 60.0)
  result['cold_normal_s'] = float(np.sum(elapsed[exposure['normal'] & cold]))
  result['warm_normal_metrics'] = {}
  for name, values in can.items():
    metrics = switch_analysis(times, values['can_brake'], values['can_gas'], exposure['normal'] & ~cold)
    result['warm_normal_metrics'][name] = {'debounced': len(metrics['changes']), 'events': len(metrics['events'])}
  result['artifacts'] = {name: sha256(root / f'{route}.json') for name, root in arm_roots.items()}
  return result


def aggregate(results):
  totals = {field: sum(result[field] for result in results) for field in
            ('clean_s', 'normal_s', 'paired_clean_s', 'paired_normal_s', 'missing_clean_s', 'missing_clean_frames')}
  totals.update(routes=len(results), routes_with_engagement=sum(result['native_enabled_s'].get('selfdriveState', 0) > 0 for result in results),
                native_enabled_s=sum(result['native_enabled_s'].get('selfdriveState', 0) for result in results))
  totals['metrics'] = {}
  for scope in ('clean', 'normal'):
    totals['metrics'][scope] = {}
    for name in results[0]['metrics'][scope]:
      metrics = [result['metrics'][scope][name] for result in results]
      totals['metrics'][scope][name] = {key: sum(metric[key] for metric in metrics) for key in ('raw', 'debounced', 'conflicts', 'outside_accel_limits')}
      totals['metrics'][scope][name]['events'] = sum(len(metric['events']) for metric in metrics)
  totals['negative_routes'] = [result['route'] for result in results if result['original_events'] == 0
                              and result['native_enabled_s'].get('selfdriveState', 0) > 0]
  totals['negative_route_metrics'] = {}
  negative = [result for result in results if result['route'] in totals['negative_routes']]
  for name in results[0]['metrics']['clean']:
    metrics = [result['metrics']['clean'][name] for result in negative]
    totals['negative_route_metrics'][name] = {key: sum(metric[key] for metric in metrics) for key in ('raw', 'debounced', 'conflicts')}
    totals['negative_route_metrics'][name]['events'] = sum(len(metric['events']) for metric in metrics)
  totals['negative_route_clean_s'] = sum(result['paired_clean_s'] for result in negative)
  totals['negative_routes_with_output_changes'] = {arm: [result['route'] for result in negative if result['plans'][arm]['engaged']['changed']]
                                                   for arm in ('default', 'experimental')}
  totals['comparisons'] = {}
  for title in results[0]['comparisons']:
    metrics = [result['comparisons'][title] for result in results]
    totals['comparisons'][title] = {field: sum(metric[field] for metric in metrics) for field in
                                   ('extra_brake_frames', 'extra_brake_seconds', 'removed_brake_frames', 'removed_brake_seconds',
                                    'weaker_strong_brake_frames', 'control_on_changes', 'standstill_changes')}
    for field in ('new_events', 'resolved_events'):
      totals['comparisons'][title][field] = [dict(event, route=result['route']) for result in results for event in result['comparisons'][title][field]]
    totals['comparisons'][title]['maximum_strong_brake_reduction'] = max(metric['maximum_strong_brake_reduction'] for metric in metrics)
  totals['strata'] = {}
  for group in results[0]['strata']:
    metrics = [result['strata'][group] for result in results]
    totals['strata'][group] = {'seconds': sum(metric['seconds'] for metric in metrics),
                              'switches': {arm: {kind: sum(metric['switches'][arm][kind] for metric in metrics) for kind in ('raw', 'debounced')}
                                           for arm in metrics[0]['switches']}}
  coverage = {}
  for name in results[0]['coverage']:
    counts, missing = Counter(), Counter()
    resets = []
    for result in results:
      for block in result['coverage'][name]:
        counts.update(block['coverage'])
        missing.update(block['missing'])
        resets.extend(dict(reset, route=result['route']) for reset in block['resets'])
    coverage[name] = {'counts': counts, 'missing': missing, 'resets': resets}
  totals['coverage'] = coverage
  totals['event_groups'] = {}
  for result in results:
    for event in result['event_windows']:
      group = totals['event_groups'].setdefault(event['category'], {'events': 0, 'switches': Counter(), 'experimental_improved': 0,
                                                                   'experimental_worse': 0, 'default_worse': 0})
      group['events'] += 1
      group['switches'].update(event['switches'])
      group['experimental_improved'] += event['switches']['experimental'] < event['switches']['default']
      group['experimental_worse'] += event['switches']['experimental'] > event['switches']['default']
      group['default_worse'] += event['switches']['default'] > event['switches']['baseline']
  totals['cold_normal_s'] = sum(result['cold_normal_s'] for result in results)
  totals['warm_normal_metrics'] = {name: {key: sum(result['warm_normal_metrics'][name][key] for result in results)
                                         for key in ('debounced', 'events')} for name in results[0]['warm_normal_metrics']}
  totals['plans'] = {}
  for arm in ('default', 'experimental'):
    metrics = [result['plans'][arm] for result in results]
    totals['plans'][arm] = {key: sum(metric[key] for metric in metrics) for key in ('stop_changes', 'fcw_changes', 'nonfinite', 'solver_failures')}
    for scope in ('engaged', 'low_speed', 'model_risk', 'strong_brake'):
      selected = [metric[scope] for metric in metrics if metric[scope]['frames']]
      totals['plans'][arm][scope] = {'frames': sum(metric['frames'] for metric in selected), 'changed': sum(metric['changed'] for metric in selected),
                                    'minimum': min((metric['minimum'] for metric in selected), default=None),
                                    'maximum': max((metric['maximum'] for metric in selected), default=None)}
  return totals


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('corpus', 'baseline', 'default', 'experimental', 'output'):
    parser.add_argument('--' + name, type=Path, required=True)
  parser.add_argument('--route')
  arguments = parser.parse_args()
  manifest = json.loads((arguments.corpus / 'manifest.json').read_text())
  manifest_hash = sha256(arguments.corpus / 'manifest.json')
  roots = {name: getattr(arguments, name) for name in ('baseline', 'default', 'experimental')}
  arguments.output.mkdir(parents=True, exist_ok=True)
  results = []
  for record in manifest['routes']:
    if arguments.route and not record['route'].startswith(arguments.route):
      continue
    result = evaluate_route(record, Path(manifest['audit']), roots, manifest_hash)
    save_json(arguments.output / f'{record["route"]}.json', result)
    results.append(result)
    print(f'evaluated {record["route"]}: normal={result["paired_normal_s"]:.2f}s', flush=True)
  summary = aggregate(results)
  summary.update(manifest_sha256=manifest_hash, evaluator_sha256=sha256(Path(__file__)), normal_margin_s=10.0)
  save_json(arguments.output / 'summary.json', summary)


if __name__ == '__main__':
  main()
