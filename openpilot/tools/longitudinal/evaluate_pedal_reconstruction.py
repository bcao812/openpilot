#!/usr/bin/env python3
"""Check recording-version reconstruction separately from candidate-vs-baseline effects."""

import argparse
import json
from pathlib import Path

import numpy as np

from openpilot.tools.longitudinal.evaluate_pedal_full_routes import (
  columns, delta_stats, exact_alignment, intervals, load_arm, native_clean, normal_mask, switch_analysis,
)
from openpilot.tools.longitudinal.replay_pedal_full_routes import NATIVE_FIELDS
from openpilot.tools.longitudinal.replay_pedal_switches import save_json, sha256


def evaluate_recorded(record, audit, recorded, manifest_hash):
  route = record['route']
  metadata, values = load_arm(recorded / route, route, manifest_hash)
  with np.load(audit / f'{route}.native.npz', allow_pickle=False) as saved:
    native = columns(saved['frames'], NATIVE_FIELDS)
  original = json.loads((audit / f'{route}.events.json').read_text())
  times, elapsed = native['t'], intervals(native['t'])
  clean = native_clean(native)
  normal = normal_mask(times, clean, original['events'] + original['raw_events'])
  indices = exact_alignment(times, values['can']['t'])
  available = np.zeros(len(times), dtype=bool)
  available[indices] = True
  can = {field: np.full(len(times), np.nan) for field in ('can_accel', 'can_gas', 'can_brake')}
  for field in can:
    can[field][indices] = values['can'][field]
  plan = values['plan']
  valid_plans = (plan['automatic'] == 1) & (plan['valid'] == 1)
  errors = np.abs(plan['accel'][valid_plans] - plan['recorded_accel'][valid_plans])
  metrics = {}
  for scope, mask in [('clean', clean), ('normal', normal)]:
    selected = mask & available
    for name, commands in [('native', native), ('recorded_source', can)]:
      raw = switch_analysis(times, commands['can_brake'], commands['can_gas'], selected, 0.0)
      debounced = switch_analysis(times, commands['can_brake'], commands['can_gas'], selected)
      metrics[f'{scope}/{name}'] = {'raw': len(raw['changes']), 'debounced': len(debounced['changes']), 'events': debounced['events']}
  selected_can = clean & available
  result = {'route': route, 'commit': record['commit'], 'recorded_plan': metadata['recorded_plan'],
            'planner_sha256': metadata['planner_sha256'], 'result_sha256': sha256(recorded / route / f'{route}.json'),
            'paired_clean_s': float(np.sum(elapsed[selected_can])), 'missing_clean_s': float(np.sum(elapsed[clean & ~available])),
            'plan_error': delta_stats(plan['recorded_accel'], plan['accel'], valid_plans),
            'can_accel_error': delta_stats(native['can_accel'], can['can_accel'], selected_can),
            'can_brake_mismatches': int(np.sum(selected_can & (native['can_brake'] != can['can_brake']))),
            'can_gas_mismatches': int(np.sum(selected_can & (native['can_gas'] != can['can_gas']))),
            'stop_mismatches': int(np.sum(valid_plans & (plan['stop'] != plan['recorded_stop']))),
            'metrics': metrics, 'coverage': metadata['blocks']}
  return result, errors


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('corpus', 'recorded', 'output'):
    parser.add_argument('--' + name, type=Path, required=True)
  arguments = parser.parse_args()
  manifest = json.loads((arguments.corpus / 'manifest.json').read_text())
  manifest_hash = sha256(arguments.corpus / 'manifest.json')
  arguments.output.mkdir(parents=True, exist_ok=True)
  results, errors = [], []
  for record in manifest['routes']:
    result, route_errors = evaluate_recorded(record, Path(manifest['audit']), arguments.recorded, manifest_hash)
    save_json(arguments.output / f'{record["route"]}.json', result)
    results.append(result)
    errors.append(route_errors)
    print(record['route'], result['plan_error'], flush=True)
  all_errors = np.concatenate(errors)
  summary = {'routes': len(results), 'recording_commits': len({result['commit'] for result in results}),
             'manifest_sha256': manifest_hash, 'evaluator_sha256': sha256(Path(__file__)),
             'plan_error': {'frames': len(all_errors), 'p95_abs': float(np.percentile(all_errors, 95)), 'maximum_abs': float(np.max(all_errors))}}
  for key in ('paired_clean_s', 'missing_clean_s', 'can_brake_mismatches', 'can_gas_mismatches', 'stop_mismatches'):
    summary[key] = sum(result[key] for result in results)
  summary['metrics'] = {scope: {'events': sum(len(result['metrics'][scope]['events']) for result in results),
                                **{key: sum(result['metrics'][scope][key] for result in results) for key in ('raw', 'debounced')}}
                        for scope in results[0]['metrics']}
  save_json(arguments.output / 'summary.json', summary)


if __name__ == '__main__':
  main()
