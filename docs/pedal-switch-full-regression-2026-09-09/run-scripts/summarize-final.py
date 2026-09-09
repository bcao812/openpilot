import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from openpilot.selfdrive.controls.tests.evaluate_pedal_full_routes import unmatched_events
from openpilot.selfdrive.controls.tests.replay_pedal_switches import save_json, sha256


run = Path('/tmp/c3-pedal-full-20260909-WgogBo')
manifest = json.loads((run / 'corpus/manifest.json').read_text())
routes = [record['route'] for record in manifest['routes']]
evaluations = [json.loads((run / 'evaluation-final' / f'{route}.json').read_text()) for route in routes]
reconstructions = {route: json.loads((run / 'reconstruction' / f'{route}.json').read_text()) for route in routes}
summary = json.loads((run / 'evaluation-final/summary.json').read_text())
reconstruction_summary = json.loads((run / 'reconstruction/summary.json').read_text())
assert summary['routes'] == reconstruction_summary['routes'] == len(routes) == 55
assert summary['manifest_sha256'] == reconstruction_summary['manifest_sha256'] == sha256(run / 'corpus/manifest.json')
jobs = json.loads((run / 'artifacts/all-job-status.json').read_text())
assert all(result['exit_code'] == 0 for result in jobs)

segments = [segment for record in manifest['routes'] for block in record['blocks'] for segment in block]
result = {
  'evaluated_at': datetime.now(UTC).isoformat(),
  'manifest_sha256': sha256(run / 'corpus/manifest.json'),
  'evaluation_summary_sha256': sha256(run / 'evaluation-final/summary.json'),
  'reconstruction_summary_sha256': sha256(run / 'reconstruction/summary.json'),
  'routes': len(routes),
  'blocks': sum(len(record['blocks']) for record in manifest['routes']),
  'segments': len(segments),
  'input_bytes': sum(segment['input']['size'] for segment in segments),
  'warning_routes': sorted({segment['route'] for segment in segments if segment['warnings']}),
  'resumed_jobs': len(jobs),
  'arms': {},
}
for arm in ('baseline', 'default', 'experimental', 'recorded'):
  metadata = []
  for route in routes:
    root = run / f'{arm}-results'
    if arm == 'recorded':
      root /= route
    item = json.loads((root / f'{route}.json').read_text())
    assert item['manifest_sha256'] == result['manifest_sha256']
    assert item['experimental'] == (arm == 'experimental')
    assert not item['recorded_plan'] and not item.get('preloaded_parameters', False)
    metadata.append(item)
  counts, missing, rows = Counter(), Counter(), Counter()
  resets, solver_failures = [], 0
  for item in metadata:
    for block in item['blocks']:
      counts.update(block['coverage'])
      missing.update(block['missing'])
      resets.extend(dict(reset, route=item['route']) for reset in block['resets'])
      solver_failures += block['solver_failures']
    for chunk in item['chunks']:
      rows.update(chunk['rows'])
  result['arms'][arm] = {
    'routes': len(metadata), 'chunks': sum(len(item['chunks']) for item in metadata),
    'rows': rows, 'coverage': counts, 'missing': missing, 'solver_failures': solver_failures,
    'resets': resets, 'planner_hashes': sorted({item['planner_sha256'] for item in metadata}),
    'replayer_hashes': sorted({item['replayer_sha256'] for item in metadata}),
    'helper_hashes': sorted({item['helper_sha256'] for item in metadata}),
  }
result['completed_route_arms'] = sum(value['routes'] for value in result['arms'].values())
result['exposure'] = {field: summary[field] for field in (
  'native_enabled_s', 'clean_s', 'normal_s', 'paired_clean_s', 'paired_normal_s', 'missing_clean_s', 'missing_clean_frames', 'cold_normal_s')}
result['exposure']['paired_clean_fraction'] = summary['paired_clean_s'] / summary['clean_s']
result['exposure']['paired_normal_fraction'] = summary['paired_normal_s'] / summary['normal_s']
result['missing_clean_routes'] = [
  {key: item[key] for key in ('route', 'missing_clean_s', 'missing_clean_frames', 'missing_clean_runs')}
  for item in evaluations if item['missing_clean_frames']]

result['normal_events'] = {}
for arm in ('native', 'baseline', 'default', 'experimental'):
  events = [dict(event, route=item['route']) for item in evaluations for event in item['metrics']['normal'][arm]['events']]
  result['normal_events'][arm] = {'events': events, 'count': len(events), 'per_hour': len(events) * 3600 / summary['paired_normal_s']}
result['normal_events']['recorded_source'] = {
  'events': [dict(event, route=route) for route, item in reconstructions.items() for event in item['metrics']['normal/recorded_source']['events']]}
result['baseline_normal_event_provenance'] = []
for item in evaluations:
  recorded_events = reconstructions[item['route']]['metrics']['normal/recorded_source']['events']
  for event in item['metrics']['normal']['baseline']['events']:
    result['baseline_normal_event_provenance'].append(dict(
      event, route=item['route'], recording_commit=item['commit'],
      overlaps_recording_source_reconstruction=not unmatched_events([event], recorded_events),
      patch_introduced=False))

result['event_window_groups'] = {field: {} for field in ('category', 'frequency_group', 'duration_group', 'frequency_by_duration')}
for item in evaluations:
  for event in item['event_windows']:
    for field, groups in result['event_window_groups'].items():
      label = event[field] if field != 'frequency_by_duration' else f'{event["frequency_group"]}/{event["duration_group"]}'
      group = groups.setdefault(label, {'events': 0, 'switches': Counter(), 'default': Counter(), 'experimental': Counter()})
      group['events'] += 1
      group['switches'].update(event['switches'])
      for arm, before in [('default', 'baseline'), ('experimental', 'default')]:
        difference = event['switches'][arm] - event['switches'][before]
        group[arm]['worse' if difference > 0 else 'improved' if difference < 0 else 'unchanged'] += 1
assert sum(group['events'] for group in result['event_window_groups']['category'].values()) == 378

result['strata'] = {}
for group in summary['strata']:
  metrics = [item['strata'][group] for item in evaluations]
  result['strata'][group] = {'seconds': summary['strata'][group]['seconds'], 'switches': summary['strata'][group]['switches']}
  for arm in ('default', 'experimental'):
    values = [metric[arm] for metric in metrics]
    result['strata'][group][arm] = {
      field: sum(value[field] for value in values) for field in (
        'extra_brake_frames', 'extra_brake_seconds', 'removed_brake_frames', 'removed_brake_seconds',
        'weaker_strong_brake_frames', 'control_on_changes', 'standstill_changes')}
    result['strata'][group][arm]['maximum_strong_brake_reduction'] = max(value['maximum_strong_brake_reduction'] for value in values)

result['manual'] = {}
for arm in ('default', 'experimental'):
  values = [item['manual'][arm] for item in evaluations]
  result['manual'][arm] = {
    'frames': sum(value['accel']['frames'] for value in values),
    'accel_changed': sum(value['accel']['changed'] for value in values),
    'maximum_accel_delta': max(value['accel']['maximum'] for value in values if value['accel']['frames']),
    'minimum_accel_delta': min(value['accel']['minimum'] for value in values if value['accel']['frames']),
    **{field: sum(value[field] for value in values) for field in (
      'extra_brake_frames', 'removed_brake_frames', 'control_on_changes', 'standstill_changes')},
  }
result['worst_plan_contexts'] = {}
for arm in ('default', 'experimental'):
  result['worst_plan_contexts'][arm] = {}
  for scope in ('engaged', 'low_speed', 'model_risk', 'strong_brake'):
    relevant = [item for item in evaluations if item['plans'][arm][scope]['frames']]
    highest = max(relevant, key=lambda item: item['plans'][arm][scope]['maximum'])
    lowest = min(relevant, key=lambda item: item['plans'][arm][scope]['minimum'])
    result['worst_plan_contexts'][arm][scope] = {
      'maximum': {'route': highest['route'], **highest['plans'][arm][scope]},
      'minimum': {'route': lowest['route'], **lowest['plans'][arm][scope]},
    }
save_json(run / 'artifacts/final-accounting.json', result)
print(json.dumps({key: result[key] for key in ('routes', 'blocks', 'segments', 'completed_route_arms', 'exposure')}, indent=2))
