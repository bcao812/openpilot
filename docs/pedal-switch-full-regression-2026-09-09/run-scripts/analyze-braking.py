import json
from pathlib import Path

import numpy as np

from openpilot.selfdrive.controls.tests.evaluate_pedal_full_routes import columns, exact_alignment, native_clean, time_runs
from openpilot.selfdrive.controls.tests.replay_pedal_full_routes import NATIVE_FIELDS, ROW_FIELDS
from openpilot.selfdrive.controls.tests.replay_pedal_switches import save_json

run = Path('/tmp/c3-pedal-full-20260909-WgogBo')
manifest = json.loads((run / 'corpus/manifest.json').read_text())
output = run / 'braking-details'
output.mkdir(exist_ok=True)


def load(root, route, kind):
  metadata = json.loads((root / f'{route}.json').read_text())
  pieces = []
  for chunk in metadata['chunks']:
    with np.load(root / chunk['file'], allow_pickle=False) as saved:
      pieces.append(saved[kind])
  return columns(np.concatenate(pieces), ROW_FIELDS[kind])


for route in manifest['routes']:
  name = route['route']
  if not all((run / f'{arm}-results' / f'{name}.json').exists() for arm in ('baseline', 'default', 'experimental')):
    continue
  with np.load(Path(manifest['audit']) / f'{name}.native.npz', allow_pickle=False) as saved:
    native = columns(saved['frames'], NATIVE_FIELDS)
  arms = {arm: load(run / f'{arm}-results', name, 'can') for arm in ('baseline', 'default', 'experimental')}
  native_indices = exact_alignment(native['t'], arms['default']['t'])
  clean = native_clean(native)[native_indices]
  result = {'route': name, 'comparisons': {}}
  for title, before_name, after_name in [('default', 'baseline', 'default'), ('experimental', 'default', 'experimental')]:
    before, after = arms[before_name], arms[after_name]
    delta = after['can_accel'] - before['can_accel']
    risk = clean & (before['can_accel'] <= -1.0) & (delta > 1e-6)
    extra = clean & (before['can_brake'] == 0) & (after['can_brake'] == 1)
    maximum_index = int(np.argmax(np.where(risk, delta, -1))) if np.any(risk) else None
    metric = {'weaker_strong_frames': int(np.sum(risk)), 'weaker_strong_seconds': sum(item['seconds'] for item in time_runs(before['t'], risk)),
              'maximum_reduction': float(delta[maximum_index]) if maximum_index is not None else 0.0,
              'runs': time_runs(before['t'], risk), 'extra_brake_runs': time_runs(before['t'], extra)}
    if maximum_index is not None:
      index = native_indices[maximum_index]
      metric['worst_context'] = {field: float(native[field][index]) for field in
                                 ('t', 'v_ego', 'a_ego', 'lead_drel', 'lead_vrel', 'lead_alead', 'model_accel', 'a_target', 'accel_cmd')}
      metric['worst_context'].update(before=float(before['can_accel'][maximum_index]), after=float(after['can_accel'][maximum_index]))
    result['comparisons'][title] = metric
  save_json(output / f'{name}.json', result)
  print(name, {arm: value['maximum_reduction'] for arm, value in result['comparisons'].items()}, flush=True)
