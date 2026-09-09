"""Historical offline analysis; hard-coded local run/corpus paths, no device or network access."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from openpilot.selfdrive.controls.tests.evaluate_pedal_full_routes import columns, native_clean, switch_analysis
from openpilot.selfdrive.controls.tests.replay_pedal_full_routes import NATIVE_FIELDS
from openpilot.selfdrive.controls.tests.replay_pedal_switches import save_json, sha256


run = Path('/tmp/c3-follow-chain-20260909-BtK1Bx')
corpus = Path('/tmp/c3-pedal-full-20260909-WgogBo/corpus')
manifest = json.loads((corpus / 'manifest.json').read_text())
audit = Path(manifest['audit'])
events = [
  dict(event, event_id=index) for index, event in enumerate(json.loads((audit / 'events-enriched.json').read_text())) if event['category'] == 'lead_following'
]
results = []
for record in manifest['routes']:
  local = [event for event in events if event['route'] == record['route']]
  if not local:
    continue
  for name, expected in record['audit_hashes'].items():
    assert sha256(audit / name) == expected
  with np.load(audit / f'{record["route"]}.native.npz', allow_pickle=False) as saved:
    data = columns(saved['frames'], NATIVE_FIELDS)
  clean = native_clean(data)
  for event in local:
    mask = clean & (data['t'] >= event['start'] - 1.0) & (data['t'] <= event['end'] + 1.0)
    counts = {}
    for label, value in [
      ('can', None),
      ('command', data['accel_cmd']),
      ('target', data['a_target']),
      ('mpc', data['mpc_accel']),
      ('model', data['model_accel']),
    ]:
      braking = data['can_brake'] if value is None else value < -0.2
      gas = data['can_gas'] if value is None else (value > -0.2).astype(float)
      counts[label] = {kind: len(switch_analysis(data['t'], braking, gas, mask, dwell)['changes']) for kind, dwell in [('raw', 0.0), ('debounced', 0.15)]}
    lead_changes = data['lead_vrel'][mask] + data['v_ego'][mask]
    difference = np.diff(lead_changes)
    selected = data['a_target'][mask]
    event['screen'] = {
      'counts': counts,
      'lead_speed_span': float(np.ptp(lead_changes)),
      'lead_speed_step_p95': float(np.percentile(np.abs(difference), 95)) if len(difference) else 0.0,
      'command_target_error_p95': float(np.percentile(np.abs(data['accel_cmd'][mask] - selected), 95)),
      'target_mpc_error_p95': float(np.percentile(np.abs(selected - data['mpc_accel'][mask]), 95)),
      'target_p05': float(np.percentile(selected, 5)),
      'target_p95': float(np.percentile(selected, 95)),
    }
    results.append(event)
save_json(run / 'artifacts/following-screen.json', {'manifest_sha256': sha256(corpus / 'manifest.json'), 'events': results})
print('events', len(results))
print(
  'stage counts',
  {
    kind: {stage: sum(event['screen']['counts'][stage][kind] for event in results) for stage in ('can', 'command', 'target', 'mpc', 'model')}
    for kind in ('raw', 'debounced')
  },
)
print(
  'stage equality',
  Counter(
    tuple(event['screen']['counts'][stage]['debounced'] == event['screen']['counts']['can']['debounced'] for stage in ('command', 'target', 'mpc'))
    for event in results
  ),
)
for event in sorted(results, key=lambda entry: (-entry['switches_per_s'], -entry['switches']))[:25]:
  print(
    event['event_id'],
    event['route'],
    round(event['start'], 3),
    round(event['duration'], 2),
    round(event['switches_per_s'], 2),
    event['screen']['counts'],
    'sources',
    event['source_fraction'],
    'a',
    event['screen']['target_p05'],
    event['screen']['target_p95'],
  )
