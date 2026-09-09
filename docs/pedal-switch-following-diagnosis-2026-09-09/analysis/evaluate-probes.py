"""Historical offline analysis; hard-coded local run/corpus paths, no device or network access."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from openpilot.selfdrive.controls.tests.evaluate_pedal_full_routes import columns, delta_stats, switch_analysis
from openpilot.selfdrive.controls.tests.replay_pedal_full_routes import ROW_FIELDS
from openpilot.selfdrive.controls.tests.replay_pedal_switches import save_json, sha256


run = Path('/tmp/c3-follow-chain-20260909-BtK1Bx')


def load(folder, identifier):
  metadata = json.loads((folder / f'{identifier}.json').read_text())
  pieces = defaultdict(list)
  for chunk in metadata['chunks']:
    path = folder / chunk['file']
    assert sha256(path) == chunk['sha256']
    with np.load(path, allow_pickle=False) as saved:
      for kind in ROW_FIELDS:
        pieces[kind].append(saved[kind])
  return {kind: columns(np.concatenate(pieces[kind]), fields) for kind, fields in ROW_FIELDS.items()}


def analyze(data, window):
  result = {}
  for kind in ('plan', 'can'):
    rows = data[kind]
    event = window['original_event']
    selected = (rows['automatic'] == 1) & (rows['t'] >= event['start'] - 0.5) & (rows['t'] <= event['end'] + 0.5)
    brake = rows['can_brake'] if kind == 'can' else rows['accel'] < -0.2
    gas = rows['can_gas'] if kind == 'can' else (rows['accel'] > -0.2).astype(int)
    result[kind] = {label: len(switch_analysis(rows['t'], brake, gas, selected, dwell)['changes']) for label, dwell in (('raw', 0), ('debounced', 0.15))}
  return result


def main():
  selection = json.loads((run / 'selection.json').read_text())
  results = []
  for window in selection['windows']:
    if window['reason'] not in ('fast_pure_lead', 'slow_pure_lead'):
      continue
    baseline = load(run / 'trace', window['id'])
    result = {'id': window['id'], 'reason': window['reason'], 'off': analyze(baseline, window), 'probes': {}}
    for field in ('dRel', 'vRel', 'aLeadK'):
      arm = load(run / ('probe-' + field), window['id'])
      assert np.array_equal(baseline['plan']['mono'], arm['plan']['mono'])
      selected = baseline['plan']['automatic'] == 1
      result['probes'][field] = dict(analyze(arm, window), delta=delta_stats(baseline['plan']['accel'], arm['plan']['accel'], selected))
    results.append(result)
  save_json(run / 'artifacts/probe-evaluation.json', results)
  for result in results:
    print(result['id'], result['reason'], 'off', result['off'], 'probes', result['probes'])


if __name__ == '__main__':
  main()
