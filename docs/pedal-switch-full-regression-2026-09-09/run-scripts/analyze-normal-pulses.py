import json
from pathlib import Path

import numpy as np

from openpilot.selfdrive.controls.tests.evaluate_pedal_full_routes import (
  columns, exact_alignment, native_clean, normal_mask, switch_analysis, unmatched_events,
)
from openpilot.selfdrive.controls.tests.replay_pedal_full_routes import NATIVE_FIELDS, ROW_FIELDS
from openpilot.selfdrive.controls.tests.replay_pedal_switches import save_json


run = Path('/tmp/c3-pedal-full-20260909-WgogBo')
manifest = json.loads((run / 'corpus/manifest.json').read_text())
audit = Path(manifest['audit'])
braking = json.loads((run / 'artifacts/braking-contexts.json').read_text())
metrics, raw_events = [], []
for record in manifest['routes']:
  route = record['route']
  route_braking = next(item for item in braking['routes'] if item['route'] == route)
  removed = route_braking['comparisons']['normal/experimental']['removed_brake']
  with np.load(audit / f'{route}.native.npz', allow_pickle=False) as saved:
    native = columns(saved['frames'], NATIVE_FIELDS)
  events = json.loads((audit / f'{route}.events.json').read_text())
  normal = normal_mask(native['t'], native_clean(native), events['events'] + events['raw_events'])
  values, changes, analyses = {}, {}, {}
  for arm in ('baseline', 'default', 'experimental'):
    root = run / f'{arm}-results'
    metadata = json.loads((root / f'{route}.json').read_text())
    pieces = []
    for chunk in metadata['chunks']:
      with np.load(root / chunk['file'], allow_pickle=False) as saved:
        pieces.append(saved['can'])
    values[arm] = columns(np.concatenate(pieces), ROW_FIELDS['can'])
    indices = exact_alignment(native['t'], values[arm]['t'])
    selected = normal[indices]
    analyses[arm] = switch_analysis(values[arm]['t'], values[arm]['can_brake'], values[arm]['can_gas'], selected, dwell=0.0)
    changes[arm] = analyses[arm]['changes']
  raw_events.append({'route': route, 'baseline': analyses['baseline']['events'], 'default': analyses['default']['events'],
                     'experimental': analyses['experimental']['events'],
                     'default_new': unmatched_events(analyses['default']['events'], analyses['baseline']['events']),
                     'default_resolved': unmatched_events(analyses['baseline']['events'], analyses['default']['events']),
                     'new': unmatched_events(analyses['experimental']['events'], analyses['default']['events']),
                     'resolved': unmatched_events(analyses['default']['events'], analyses['experimental']['events'])})
  default, experimental = values['default'], values['experimental']
  for span in removed:
    first = int(np.searchsorted(default['t'], span['start']))
    last = first + span['frames']
    before = max(first - 1, 0)
    after = min(last, len(default['t']) - 1)
    neighbor_gap = default['t'][first] - default['t'][before] <= 0.15 and default['t'][after] - default['t'][last - 1] <= 0.15
    all_normal = bool(np.all(selected[before:after + 1]))
    baseline_brakes = bool(np.all(default['can_brake'][before:after + 1] == 1))
    experiment_pulse = (experimental['can_brake'][before] == experimental['can_brake'][after] == 1
                        and np.all(experimental['can_brake'][first:last] == 0) and np.all(experimental['can_gas'][first:last] > 0))
    metrics.append(dict(span, route=route, new_gas_pulse_between_unchanged_brakes=bool(
      neighbor_gap and all_normal and baseline_brakes and experiment_pulse),
      raw_changes_nearby={arm: [change['t'] for change in arm_changes if span['start'] - 0.15 <= change['t'] <= span['end'] + 0.15]
                         for arm, arm_changes in changes.items()}))
result = {'normal_removed_brake_runs': len(metrics), 'isolated_new_gas_pulses': [span for span in metrics if span['new_gas_pulse_between_unchanged_brakes']],
          'all_runs': metrics, 'raw_events': raw_events,
          'default_new_raw_events': [dict(event, route=item['route']) for item in raw_events for event in item['default_new']],
          'new_raw_events': [dict(event, route=item['route']) for item in raw_events for event in item['new']],
          'definition': 'Normal fixed mask, baseline braking across interval and both neighbors; experimental gas strictly inside.'}
save_json(run / 'artifacts/normal-pulse-classification.json', result)
print('normal removed brake runs', len(metrics), 'new gas pulses', len(result['isolated_new_gas_pulses']))
for span in result['isolated_new_gas_pulses']:
  print(span['route'], span['start'], span['seconds'], span['frames'])
print('new raw recurring events', result['new_raw_events'])
print('default new raw recurring events', result['default_new_raw_events'])
