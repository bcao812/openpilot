#!/usr/bin/env python3
"""Compare aligned offline command arms, retaining recording reconstruction errors."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from openpilot.selfdrive.controls.tests.replay_pedal_switches import save_json, sha256, switch_metrics


def quantiles(values):
  if not len(values):
    return None
  return {name: float(value) for name, value in zip(('minimum', 'median', 'p95', 'maximum'),
                                                  np.percentile(values, [0, 50, 95, 100]), strict=True)}


def recorded_rows(rows):
  return [dict(row, can_accel=row['recorded_accel'], can_brake=row['recorded_brake'], can_gas=row['recorded_gas']) for row in rows]


def evaluate(corpus, recorded, baseline, candidate, output, allow_partial=False):
  manifest = json.loads((corpus / 'manifest.json').read_text())
  windows, events = [], []
  reconstruction_errors, command_deltas = [], []
  missing = []
  for index, window in enumerate(manifest['windows']):
    paths = {name: root / f'{index:03d}.json' for name, root in [('recorded', recorded), ('baseline', baseline), ('candidate', candidate)]}
    if not all(path.is_file() for path in paths.values()):
      missing.append(index)
      continue
    arms = {name: json.loads(path.read_text()) for name, path in paths.items()}
    for name, arm in arms.items():
      if arm.get('recorded_plan', False):
        raise ValueError('Recorded-plan control validation is not a full planner replay arm')
      if arm['window']['sha256'] != window['sha256']:
        raise ValueError(f'{name} input hash mismatch: {index}')
      for field in ('plan_rows', 'control_rows', 'can_rows'):
        if [row['t'] for row in arm[field]] != [row['t'] for row in arms['baseline'][field]]:
          raise ValueError(f'{name} time alignment mismatch: {index}, {field}')
    selected = [row for row in arms['recorded']['plan_rows'] if row['automatic'] and row['valid']]
    errors = [abs(row['accel'] - row['recorded_accel']) for row in selected]
    reconstruction_errors.extend(errors)
    before, after = arms['baseline']['plan_rows'], arms['candidate']['plan_rows']
    delta = np.array([second['accel'] - first['accel'] for first, second in zip(before, after, strict=True)])
    eligible = np.array([row['automatic'] and row['valid'] for row in before], dtype=bool)
    command_deltas.extend(delta[eligible].tolist())
    changes = np.abs(delta) > 1e-6
    low_speed = np.array([row['v_ego'] < 10.0 for row in before], dtype=bool)
    stop_changes = sum(first['stop'] != second['stop'] for first, second in zip(before, after, strict=True))
    raw = recorded_rows(arms['recorded']['can_rows'])
    reconstruction_can = arms['recorded']['can_rows']
    can_valid = [row for row in reconstruction_can if row['automatic']]
    windows.append({'index': index, 'route': window['route'], 'start': window['start'], 'end': window['end'],
                    'risks': window['risks'], 'frames': len(before), 'automatic_frames': int(np.count_nonzero(eligible)),
                    'coverage': {name: arm['coverage'] for name, arm in arms.items()},
                    'solver_failures': {name: arm['solver_failures'] for name, arm in arms.items()},
                    'reconstruction_error': quantiles(errors), 'candidate_delta': quantiles(delta[eligible]),
                    'changed_frames': int(np.count_nonzero(changes)), 'low_speed_changed_frames': int(np.count_nonzero(changes & low_speed)),
                    'low_speed_delta': quantiles(delta[low_speed]),
                    'stop_changed_frames': stop_changes,
                    'can_brake_disagreement_fraction': sum(row['can_brake'] != row['recorded_brake'] for row in can_valid) / max(1, len(can_valid)),
                    'switches': {name: switch_metrics(arm['can_rows']) for name, arm in arms.items()},
                    'raw_switches': switch_metrics(raw),
                    'conflicts': {name: sum(row['can_brake'] and row['can_gas'] > 0 for row in arm['can_rows']) for name, arm in arms.items()}})
    for event_id in window['events']:
      event = next(event for event in manifest['events'] if event['event_id'] == event_id)
      event_errors = [abs(row['accel'] - row['recorded_accel']) for row in selected if event['start'] <= row['t'] <= event['end']]
      switches = {}
      dwell = 0.0 if manifest.get('events_file') == 'raw-only-events.json' else 0.15
      for name, rows in [('native', raw), *((name, arm['can_rows']) for name, arm in arms.items())]:
        metrics = switch_metrics(rows, dwell=dwell)
        switches[name] = sum(event['start'] - 1e-7 <= timestamp <= event['end'] + 1e-7 for timestamp in metrics['switches'])
      event_delta = [value for row, value in zip(before, delta, strict=True) if event['start'] <= row['t'] <= event['end']]
      events.append({'event_id': event_id, 'route': event['route'], 'category': event['category'], 'start': event['start'], 'end': event['end'],
                     'duration': event['duration'], 'frequency_group': event['frequency_group'], 'duration_group': event['duration_group'],
                     'audit_switches': event['switches'], 'switches': switches, 'reconstruction_error': quantiles(event_errors),
                     'candidate_delta': quantiles(event_delta),
                     'baseline_frequency': max(0, switches['baseline'] - 1) / event['duration'],
                     'candidate_frequency': max(0, switches['candidate'] - 1) / event['duration']})
  if missing and not allow_partial:
    raise ValueError(f'Missing windows: {missing}')
  groups = defaultdict(list)
  for event in events:
    groups[event['category']].append(event)
  grouped = {}
  for name, group in groups.items():
    grouped[name] = {'events': len(group), 'duration': sum(event['duration'] for event in group),
                     'switches': {arm: sum(event['switches'][arm] for event in group) for arm in ('native', 'recorded', 'baseline', 'candidate')},
                     'improved': sum(event['switches']['candidate'] < event['switches']['baseline'] for event in group),
                     'unchanged': sum(event['switches']['candidate'] == event['switches']['baseline'] for event in group),
                     'worse': sum(event['switches']['candidate'] > event['switches']['baseline'] for event in group)}
  summary = {'windows': len(windows), 'missing_windows': missing, 'events': len(events), 'groups': grouped,
             'frames': sum(window['frames'] for window in windows), 'automatic_frames': sum(window['automatic_frames'] for window in windows),
             'candidate_delta': quantiles(command_deltas), 'reconstruction_error': quantiles(reconstruction_errors),
             'solver_failures': {name: sum(window['solver_failures'][name] for window in windows) for name in ('recorded', 'baseline', 'candidate')},
             'changed_frames': sum(window['changed_frames'] for window in windows),
             'low_speed_changed_frames': sum(window['low_speed_changed_frames'] for window in windows),
             'stop_changed_frames': sum(window['stop_changed_frames'] for window in windows),
             'conflicts': {name: sum(window['conflicts'][name] for window in windows) for name in ('recorded', 'baseline', 'candidate')}}
  result = {'method': 'open loop; same recorded motion/perception/SP decisions; native 20/100/50Hz alignment; not vehicle outcomes',
            'manifest_sha256': sha256(corpus / 'manifest.json'), 'evaluator_sha256': sha256(Path(__file__)),
            'summary': summary, 'windows': windows, 'events': sorted(events, key=lambda event: event['event_id'])}
  save_json(output, result)
  print(json.dumps(summary, indent=2))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for name in ('corpus', 'recorded', 'baseline', 'candidate', 'output'):
    parser.add_argument('--' + name, type=Path, required=True)
  parser.add_argument('--allow-partial', action='store_true')
  arguments = parser.parse_args()
  evaluate(arguments.corpus, arguments.recorded, arguments.baseline, arguments.candidate, arguments.output, arguments.allow_partial)


if __name__ == '__main__':
  main()
