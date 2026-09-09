"""Historical offline analysis; hard-coded local run/corpus paths, no device or network access."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from openpilot.selfdrive.controls.tests.evaluate_pedal_full_routes import switch_analysis
from openpilot.selfdrive.controls.tests.replay_pedal_switches import save_json


run = Path('/tmp/c3-follow-chain-20260909-BtK1Bx')
selection = json.loads((run / 'selection.json').read_text())
outputs = []
for window in selection['windows']:
  path = run / 'trace' / f'{window["id"]}.trace.json'
  if not path.exists():
    continue
  rows = json.loads(path.read_text())
  times = np.array([row['t'] for row in rows])
  selected = np.array([row['automatic'] and row['valid'] for row in rows])
  event = window.get('original_event')
  if event:
    selected &= (times >= event['start'] - 0.5) & (times <= event['end'] + 0.5)
  target = np.array([row['a_target'] for row in rows])
  current = {'id': window['id'], 'reason': window['reason'], 'plan_frames': len(rows), 'stages': {}, 'variation': {}, 'correlations': {}}
  for name in ('a_target', 'mpc_accel', 'cruise_accel', 'model_accel'):
    data = np.array([row[name] for row in rows])
    current['stages'][name] = {
      label: len(switch_analysis(times, data < -0.2, (data > -0.2).astype(int), selected, dwell)['changes'])
      for label, dwell in [('raw', 0.0), ('debounced', 0.15)]
    }
  for name, data in {
    'lead_distance': [row['leadOne']['dRel'] for row in rows],
    'lead_relative_speed': [row['leadOne']['vRel'] for row in rows],
    'lead_acceleration': [row['leadOne']['aLeadK'] for row in rows],
    'ego_speed': [row['ego_speed'] for row in rows],
    'ego_acceleration': [row['ego_accel'] for row in rows],
    'ego_state_mismatch': [row['ego_speed'] - row['mpc_input_speed'] for row in rows],
    'mpc_relative_speed': [row['leadOne']['vLead'] - row['mpc_input_speed'] for row in rows],
    'obstacle_t1': [row['obstacle_t1'] for row in rows],
    'obstacle_t2': [row['obstacle_t2'] for row in rows],
  }.items():
    data = np.array(data)
    sliced = data[selected]
    if not len(sliced):
      continue
    changes = np.diff(data)
    same = selected[:-1] & selected[1:]
    current['variation'][name] = {
      'p05': float(np.percentile(sliced, 5)),
      'p95': float(np.percentile(sliced, 95)),
      'step_p95': float(np.percentile(np.abs(changes[same]), 95)) if any(same) else 0.0,
    }
    if np.std(changes[same]) > 1e-9 and np.std(np.diff(target)[same]) > 1e-9:
      current['correlations'][name] = float(np.corrcoef(changes[same], np.diff(target)[same])[0, 1])
  current['source_counts'] = dict(Counter(row['source'] for row, keep in zip(rows, selected, strict=True) if keep))
  current['e2e_frames'] = sum(row['e2e'] for row, keep in zip(rows, selected, strict=True) if keep)
  current['model_delay_s'] = {
    'minimum': min((row['mono'] - row['radar_model_mono']) / 1e9 for row in rows),
    'maximum': max((row['mono'] - row['radar_model_mono']) / 1e9 for row in rows),
  }
  outputs.append(current)
save_json(run / 'artifacts/trace-summary.json', outputs)
for result in outputs[:16]:
  print(result['id'], result['reason'], result['stages'], 'src', result['source_counts'])
  print('variation', result['variation'])
  print('correlation', result['correlations'])
