import json
from pathlib import Path

import numpy as np

from openpilot.selfdrive.controls.tests.evaluate_pedal_full_routes import columns, exact_alignment, native_clean, normal_mask, time_runs
from openpilot.selfdrive.controls.tests.replay_pedal_full_routes import NATIVE_FIELDS, ROW_FIELDS
from openpilot.selfdrive.controls.tests.replay_pedal_switches import save_json


run = Path('/tmp/c3-pedal-full-20260909-WgogBo')
manifest = json.loads((run / 'corpus/manifest.json').read_text())
audit = Path(manifest['audit'])


def load_values(root, route, kind):
  metadata = json.loads((root / f'{route}.json').read_text())
  pieces = []
  for chunk in metadata['chunks']:
    with np.load(root / chunk['file'], allow_pickle=False) as saved:
      pieces.append(saved[kind])
  return columns(np.concatenate(pieces), ROW_FIELDS[kind])


def context(native, index):
  return {field: float(native[field][index]) for field in (
    't', 'v_ego', 'a_ego', 'lead1', 'lead2', 'lead_drel', 'lead_vrel', 'model_accel', 'model_stop',
    'a_target', 'accel_cmd', 'long_state', 'gas_pressed', 'brake_pressed', 'override')}


braking = {'routes': [], 'comparisons': {}}
reconstruction = {'routes': [], 'maximum_can_abs': 0.0}
all_errors = []
for record in manifest['routes']:
  route = record['route']
  with np.load(audit / f'{route}.native.npz', allow_pickle=False) as saved:
    native = columns(saved['frames'], NATIVE_FIELDS)
  original = json.loads((audit / f'{route}.events.json').read_text())
  original_normal = normal_mask(native['t'], native_clean(native), original['events'] + original['raw_events'])
  arms = {arm: load_values(run / f'{arm}-results', route, 'can') for arm in ('baseline', 'default', 'experimental')}
  indices = exact_alignment(native['t'], arms['default']['t'])
  selected = {'clean': native_clean(native)[indices], 'normal': original_normal[indices]}
  route_result = {'route': route, 'comparisons': {}}
  for title, before_name, after_name in [('default', 'baseline', 'default'), ('experimental', 'default', 'experimental')]:
    before, after = arms[before_name], arms[after_name]
    assert np.array_equal(before['mono'], after['mono'])
    delta = after['can_accel'] - before['can_accel']
    for scope, mask in selected.items():
      key = f'{scope}/{title}'
      metric = {'strong_brake_reduction': [], 'added_brake': [], 'removed_brake': []}
      for label, condition in (
        ('strong_brake_reduction', (before['can_accel'] <= -1.0) & (delta > 1e-6)),
        ('added_brake', (before['can_brake'] == 0) & (after['can_brake'] == 1)),
        ('removed_brake', (before['can_brake'] == 1) & (after['can_brake'] == 0)),
      ):
        chosen = mask & condition
        for span in time_runs(before['t'], chosen):
          within = chosen & (before['t'] >= span['start']) & (before['t'] <= span['end'])
          if not np.any(within):
            continue
          first = int(np.flatnonzero(within)[0])
          metric[label].append(dict(
            span, context=context(native, indices[first]),
            before_accel_min=float(np.min(before['can_accel'][within])),
            before_accel_max=float(np.max(before['can_accel'][within])),
            after_accel_min=float(np.min(after['can_accel'][within])),
            after_accel_max=float(np.max(after['can_accel'][within])),
            maximum_accel_delta=float(np.max(delta[within])),
            minimum_accel_delta=float(np.min(delta[within]))))
      route_result['comparisons'][key] = metric
  braking['routes'].append(route_result)

  root = run / 'recorded-results' / route
  recorded_can = load_values(root, route, 'can')
  native_indices = exact_alignment(native['t'], recorded_can['t'])
  clean = native_clean(native)[native_indices]
  delta = recorded_can['can_accel'] - native['can_accel'][native_indices]
  errors = np.abs(delta[clean])
  all_errors.append(errors)
  maximum = float(np.max(errors)) if len(errors) else 0.0
  item = {'route': route, 'recording_commit': record['commit'], 'clean_frames': len(errors),
          'maximum_can_abs': maximum, 'maximum_can_context': None, 'maximum_plan_context': None}
  if np.any(clean):
    worst = int(np.argmax(np.where(clean, np.abs(delta), -1)))
    item['maximum_can_context'] = dict(
      context(native, native_indices[worst]), reconstructed_accel=float(recorded_can['can_accel'][worst]),
      native_accel=float(native['can_accel'][native_indices[worst]]),
      reconstructed_brake=float(recorded_can['can_brake'][worst]),
      native_brake=float(native['can_brake'][native_indices[worst]]),
      normal=bool(original_normal[native_indices[worst]]))
  plan = load_values(root, route, 'plan')
  valid = (plan['automatic'] == 1) & (plan['valid'] == 1)
  if np.any(valid):
    worst = int(np.argmax(np.where(valid, np.abs(plan['accel'] - plan['recorded_accel']), -1)))
    item['maximum_plan_context'] = {field: float(plan[field][worst]) for field in (
      't', 'accel', 'recorded_accel', 'v_ego', 'lead', 'model_accel', 'model_stop', 'stop', 'recorded_stop',
      'force_decel', 'allow_throttle', 'sp_cruise', 'solver_status')}
  reconstruction['routes'].append(item)
  reconstruction['maximum_can_abs'] = max(reconstruction['maximum_can_abs'], maximum)
  print(route, 'CAN reconstruction maximum', maximum, flush=True)

for key in braking['routes'][0]['comparisons']:
  groups = [item['comparisons'][key] for item in braking['routes']]
  braking['comparisons'][key] = {}
  for label in groups[0]:
    spans = [span for group in groups for span in group[label]]
    braking['comparisons'][key][label] = {
      'runs': len(spans), 'frames': sum(span['frames'] for span in spans),
      'seconds': sum(span['seconds'] for span in spans),
      'maximum_run_s': max((span['seconds'] for span in spans), default=0.0),
      'maximum_abs_accel_delta': max((max(abs(span['maximum_accel_delta']), abs(span['minimum_accel_delta'])) for span in spans), default=0.0),
      'runs_lt_150ms': sum(span['seconds'] < 0.15 for span in spans),
    }
summary = json.loads((run / 'evaluation-final/summary.json').read_text())
for key, metrics in braking['comparisons'].items():
  scope, arm = key.split('/')
  comparison = summary['comparisons'][f'{scope}/{arm}_vs_{"baseline" if arm == "default" else "default"}']
  for label, field in [('strong_brake_reduction', 'weaker_strong_brake_frames'), ('added_brake', 'extra_brake_frames'),
                       ('removed_brake', 'removed_brake_frames')]:
    assert metrics[label]['frames'] == comparison[field], (key, label, metrics[label]['frames'], comparison[field])
errors = np.concatenate(all_errors)
reconstruction['can_frames'] = len(errors)
reconstruction['can_p95_abs'] = float(np.percentile(errors, 95))
save_json(run / 'artifacts/braking-contexts.json', braking)
save_json(run / 'artifacts/reconstruction-extremes.json', reconstruction)
