"""Historical offline analysis; hard-coded local run/corpus paths, no device or network access."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from openpilot.selfdrive.controls.tests.evaluate_pedal_full_routes import columns, delta_stats, native_clean, switch_analysis
from openpilot.selfdrive.controls.tests.replay_pedal_full_routes import NATIVE_FIELDS
from openpilot.selfdrive.controls.tests.replay_pedal_switches import save_json
from importlib.util import module_from_spec, spec_from_file_location


run = Path('/tmp/c3-follow-chain-20260909-BtK1Bx')
spec = spec_from_file_location('probe_evaluation', run / 'evaluate-probes.py')
loader = module_from_spec(spec)
spec.loader.exec_module(loader)
selection = json.loads((run / 'selection.json').read_text())
audit = Path('/home/bcao/SourceCode/_logs/pedal-switch-2026-09-09')
results = []


def matches(before, after, maximum_delay):
  matched, used = [], set()
  for current in after:
    candidates = [
      (index, previous)
      for index, previous in enumerate(before)
      if index not in used and current['side'] == previous['side'] and 0 <= current['t'] - previous['t'] <= maximum_delay
    ]
    if candidates:
      index, previous = candidates[-1]
      used.add(index)
      matched.append({'before_t': previous['t'], 'after_t': current['t'], 'delay_s': current['t'] - previous['t'], 'side': current['side']})
  return {'before': len(before), 'after': len(after), 'matched': matched}


for window in selection['windows'][:16]:
  data = loader.load(run / 'trace', window['id'])
  with np.load(audit / f'{window["route"]}.native.npz', allow_pickle=False) as saved:
    native = columns(saved['frames'], NATIVE_FIELDS)
  native_ok = native_clean(native)
  original = window['original_event']
  selected = {}
  for kind, rows in data.items():
    positions = np.clip(np.searchsorted(native['t'], rows['t'], side='right') - 1, 0, len(native['t']) - 1)
    mask = native_ok[positions] & (rows['t'] - native['t'][positions] >= 0) & (rows['t'] - native['t'][positions] <= 0.15)
    mask &= (rows['t'] >= original['start'] - 0.5) & (rows['t'] <= original['end'] + 0.5)
    if kind == 'plan':
      mask &= (rows['automatic'] == 1) & (rows['valid'] == 1)
    selected[kind] = mask
  result = {'id': window['id'], 'reason': window['reason'], 'stages': {}, 'links': {}, 'replay_error': {}}
  for kind, rows in data.items():
    acceleration = rows['recorded_accel']
    brake = rows['recorded_brake'] if kind == 'can' else acceleration < -0.2
    gas = rows['recorded_gas'] if kind == 'can' else (acceleration > -0.2).astype(int)
    result['stages'][kind] = {
      label: switch_analysis(rows['t'], brake, gas, selected[kind], dwell)['changes'] for label, dwell in (('raw', 0), ('debounced', 0.15))
    }
    actual = rows['can_accel'] if kind == 'can' else rows['accel']
    result['replay_error'][kind] = delta_stats(acceleration, actual, selected[kind])
  for label in ('raw', 'debounced'):
    stages = result['stages']
    result['links'][label] = {
      'plan_to_control': matches(stages['plan'][label], stages['control'][label], 0.15),
      'control_to_can': matches(stages['control'][label], stages['can'][label], 0.15),
    }
    linked = []
    for change in stages['control'][label]:
      plan_mono = data['control']['plan_mono'][change['index']]
      candidates = np.flatnonzero(data['plan']['mono'] == plan_mono)
      if len(candidates) != 1:
        continue
      position = candidates[0]
      plan_accel = data['plan']['recorded_accel'][position]
      plan_side = -1 if plan_accel < -0.2 else 1 if plan_accel > -0.2 else 0
      linked.append(
        {
          'control_t': change['t'],
          'plan_mono': int(plan_mono),
          'side_matches': plan_side == change['side'],
          'delay_s': change['t'] - data['plan']['t'][position],
        }
      )
    result['links'][label]['exact_control_plan_reference'] = linked
  results.append(result)
save_json(run / 'artifacts/transition-alignment.json', results)
for label in ('raw', 'debounced'):
  print(label)
  for link in ('plan_to_control', 'control_to_can'):
    links = [record['links'][label][link] for record in results]
    delays = [pair['delay_s'] for record in links for pair in record['matched']]
    print(
      link,
      'before',
      sum(record['before'] for record in links),
      'after',
      sum(record['after'] for record in links),
      'matched',
      len(delays),
      'delays',
      np.percentile(delays, [0, 50, 95, 100]).tolist(),
    )
  exact = [record for result in results for record in result['links'][label]['exact_control_plan_reference']]
  print('exact_control_plan_reference', len(exact), Counter(record['side_matches'] for record in exact))
