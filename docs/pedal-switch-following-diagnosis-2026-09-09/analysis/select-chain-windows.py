"""Historical offline analysis; hard-coded local run/corpus paths, no device or network access."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from openpilot.selfdrive.controls.tests.evaluate_pedal_full_routes import columns, native_clean, normal_mask, time_runs
from openpilot.selfdrive.controls.tests.replay_pedal_full_routes import NATIVE_FIELDS
from openpilot.selfdrive.controls.tests.replay_pedal_switches import save_json, sha256


run = Path('/tmp/c3-follow-chain-20260909-BtK1Bx')
corpus = Path('/tmp/c3-pedal-full-20260909-WgogBo/corpus')
manifest = json.loads((corpus / 'manifest.json').read_text())
screen = json.loads((run / 'artifacts/following-screen.json').read_text())['events']
records = {record['route']: record for record in manifest['routes']}
chosen = []
groups = {
  'fast_pure_lead': [
    event for event in screen if event['switches_per_s'] >= 1.0 and event['source_fraction']['lead0'] + event['source_fraction']['lead1'] >= 0.95
  ],
  'fast_mixed_candidates': [
    event for event in screen if event['switches_per_s'] >= 1.0 and event['source_fraction']['lead0'] + event['source_fraction']['lead1'] < 0.95
  ],
  'slow_pure_lead': [event for event in screen if event['switches_per_s'] < 0.5 and event['source_flips'] == 0 and event['source_fraction']['lead0'] > 0.95],
  'slow_mixed_candidates': [event for event in screen if event['switches_per_s'] < 0.5 and event['source_flips'] > 0],
}


def add(route, start, end, group, reason, event=None):
  record = records[route]
  warmup = max(0.0, start - (120.0 if start >= 130.0 else 60.0))
  blocks = [
    block for block in record['blocks'] if block[0]['first_mono'] / 1e9 - record['origin'] <= warmup and block[-1]['last_mono'] / 1e9 - record['origin'] >= end
  ]
  if not blocks:
    return False
  block = blocks[0]
  window = {
    'id': f'{len(chosen):03d}',
    'route': route,
    'start': start,
    'end': end,
    'warmup': warmup,
    'group': group,
    'reason': reason,
    'origin': record['origin'],
    'commit': record['commit'],
    'parameter_segment': block[0],
    'audit_hashes': record['audit_hashes'],
    'segments': [
      segment
      for segment in block
      if segment['last_mono'] / 1e9 - record['origin'] >= warmup - 0.1 and segment['first_mono'] / 1e9 - record['origin'] <= end + 0.3
    ],
  }
  if event is not None:
    window['original_event'] = event
  chosen.append(window)
  return True


for reason, pool in groups.items():
  used = set()
  count = 6 if reason == 'fast_pure_lead' else 4 if reason == 'slow_pure_lead' else 3
  for event in sorted(pool, key=lambda value: (-value['switches'] if reason.startswith('slow') else -value['switches_per_s'], value['route'])):
    if event['route'] in used:
      continue
    if add(event['route'], event['start'] - 5.0, event['end'] + 5.0, 'oscillation', reason, event):
      used.add(event['route'])
    if len(used) == count:
      break
for route in sorted({window['route'] for window in chosen}):
  with np.load(Path(manifest['audit']) / f'{route}.native.npz', allow_pickle=False) as saved:
    native = columns(saved['frames'], NATIVE_FIELDS)
  audit = json.loads((Path(manifest['audit']) / f'{route}.events.json').read_text())
  clean = normal_mask(native['t'], native_clean(native), audit['events'] + audit['raw_events'])
  clean &= (native['lead1'] == 1) & (native['v_ego'] >= 5.0) & (native['t'] >= 130.0) & (native['source'] == 1)
  for span in sorted(time_runs(native['t'], clean), key=lambda value: -value['seconds']):
    if span['seconds'] < 8:
      break
    duration = min(40.0, span['seconds'])
    start = span['start'] + (span['seconds'] - duration) / 2
    if add(route, start, start + duration, 'normal', 'same_route_continuous_lead_control'):
      break
previous = json.loads(Path('/tmp/c3-pedal-modes-20260909-uCaYf5/final/selection.json').read_text())
for window in previous['windows']:
  if window['reason'] == 'originally_nonoscillating_engaged_route' or window['group'] == 'risk':
    add(window['route'], window['start'], window['end'], window['group'], window['reason'])
save_json(
  run / 'selection.json',
  {
    'corpus': str(corpus),
    'manifest_sha256': sha256(corpus / 'manifest.json'),
    'method': ('Fixed original following categories before candidate changes; distinct-route frequency/source strata and same-route negative controls'),
    'windows': chosen,
  },
)
print(Counter(window['group'] for window in chosen), 'minutes', sum(window['end'] - window['start'] for window in chosen) / 60)
for window in chosen:
  print(window['id'], window['route'], round(window['start'], 2), round(window['end'], 2), window['reason'])
