"""Historical offline analysis; hard-coded local run/corpus paths, no device or network access."""

from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as etree

import numpy as np

from openpilot.selfdrive.controls.tests.replay_pedal_full_routes import ROW_FIELDS
from openpilot.selfdrive.controls.tests.replay_pedal_switches import save_json, sha256


run = Path('/tmp/c3-follow-chain-20260909-BtK1Bx')
screen = json.loads((run / 'artifacts/following-screen.json').read_text())
selection = json.loads((run / 'selection.json').read_text())
alignment = json.loads((run / 'artifacts/transition-alignment.json').read_text())
probes = json.loads((run / 'artifacts/probe-evaluation.json').read_text())
evaluation = json.loads((run / 'trace-regression/evaluation.json').read_text())
summary = {
  'scope': {
    'base': '9d95ca380705d3537ba625d2a578aa10e2944379',
    'branch': 'c3xl-dev',
    'date': '2026-09-09',
    'vehicle': 'Honda Accord 2019 Hybrid, openpilot longitudinal, radarUnavailable=True; not BOSCH_RADARLESS',
  }
}
groups = Counter(
  (
    ('0.2-0.5', '0.5-1', '1-2', '>=2')[int(np.searchsorted([0.5, 1, 2], event['switches_per_s'], side='right'))],
    ('<5', '5-15', '15-30', '>=30')[int(np.searchsorted([5, 15, 30], event['duration'], side='right'))],
  )
  for event in screen['events']
)
summary['screen'] = {
  'events': len(screen['events']),
  'groups': [{'frequency': key[0], 'duration': key[1], 'count': value} for key, value in sorted(groups.items())],
  'padded_stage_counts': {
    kind: {stage: sum(event['screen']['counts'][stage][kind] for event in screen['events']) for stage in ('can', 'command', 'target', 'mpc', 'model')}
    for kind in ('raw', 'debounced')
  },
  'all_four_counts_equal': sum(
    all(event['screen']['counts'][stage]['debounced'] == event['screen']['counts']['can']['debounced'] for stage in ('command', 'target', 'mpc'))
    for event in screen['events']
  ),
  'count_window_padding_s': 1.0,
  'counts_do_not_match_transitions': True,
}
summary['alignment'] = {}
for label in ('raw', 'debounced'):
  output = {}
  for link in ('plan_to_control', 'control_to_can'):
    links = [record['links'][label][link] for record in alignment]
    delays = [pair['delay_s'] for record in links for pair in record['matched']]
    output[link] = {
      'before': sum(record['before'] for record in links),
      'after': sum(record['after'] for record in links),
      'matched': len(delays),
      'delay_s': dict(zip(('min', 'p50', 'p95', 'max'), np.percentile(delays, [0, 50, 95, 100]).tolist(), strict=True)),
    }
  exact = [record for result in alignment for record in result['links'][label]['exact_control_plan_reference']]
  output['exact_control_plan_reference'] = {'checked': len(exact), 'side_matches': sum(record['side_matches'] for record in exact)}
  summary['alignment'][label] = output
summary['probes'] = {}
for reason in ('fast_pure_lead', 'slow_pure_lead'):
  selected = [result for result in probes if result['reason'] == reason]
  summary['probes'][reason] = {}
  for mode in ('off', 'dRel', 'vRel', 'aLeadK'):
    values = [result['off'] if mode == 'off' else result['probes'][mode] for result in selected]
    summary['probes'][reason][mode] = {
      kind: {label: sum(result[kind][label] for result in values) for label in ('raw', 'debounced')} for kind in ('plan', 'can')
    }
summary['regression'] = {
  'groups': dict(Counter(window['group'] for window in selection['windows'])),
  'routes': len({window['route'] for window in selection['windows']}),
  'gross_minutes': sum(window['end'] - window['start'] for window in selection['windows']) / 60,
  'coverage': {kind: sum(result['coverage'][kind] for result in evaluation) for kind in evaluation[0]['coverage']},
  'exact_equal_windows': sum(result['exact_equal'] for result in evaluation),
  'trace': {field: sum(result['trace'][field] for result in evaluation) for field in ('attempted', 'samples', 'dropped', 'failures', 'bytes')},
  'solver_status_nonzero_frames': sum(result['solver_failures'] for result in evaluation),
}
baseline_equal = []
for window in selection['windows']:
  arms = []
  for folder in (run / 'trace', run / 'trace-regression/off'):
    metadata = json.loads((folder / f'{window["id"]}.json').read_text())
    arrays = {kind: [] for kind in ROW_FIELDS}
    for chunk in metadata['chunks']:
      path = folder / chunk['file']
      assert sha256(path) == chunk['sha256']
      with np.load(path, allow_pickle=False) as data:
        for kind in ROW_FIELDS:
          arrays[kind].append(data[kind])
    arms.append({kind: np.concatenate(pieces) for kind, pieces in arrays.items()})
  baseline_equal.append({'id': window['id'], 'exact_equal': all(np.array_equal(arms[0][kind], arms[1][kind], equal_nan=True) for kind in ROW_FIELDS)})
assert all(result['exact_equal'] for result in baseline_equal)
summary['regression']['unchanged_default_vs_start'] = baseline_equal
accounting = {}
latest = {}
subtests = Counter()
for name in ('final-regression', 'honda-safety', 'trace-final-unit', 'trace-tool-final', 'release-gates'):
  path = run / 'artifacts' / (name + '.xml')
  if not path.exists():
    continue
  cases = []
  counts = Counter()
  skipped_reasons = Counter()
  for case in etree.parse(path).iter('testcase'):
    status = 'failed' if case.find('failure') is not None else 'error' if case.find('error') is not None else 'passed'
    skipped = case.find('skipped')
    if skipped is not None:
      status = 'xfailed' if skipped.attrib.get('type') == 'pytest.xfail' else 'skipped'
      skipped_reasons[skipped.attrib.get('message', '')] += 1
    identity = (case.attrib['classname'], case.attrib['name'])
    if '[' in identity[1] and ']' in identity[1] and case.find('properties') is not None:
      pass
    cases.append({'class': identity[0], 'name': identity[1], 'status': status})
    counts[status] += 1
    if name != 'release-gates':
      latest[identity] = status
  accounting[name] = {
    'counts_with_junit_subtest_duplicates': dict(counts),
    'skip_reasons': dict(skipped_reasons),
    'xml_sha256': sha256(path),
    'log_sha256': sha256(path.with_suffix('.log')),
    'cases': cases,
  }
summary['test_unique_junit_names'] = dict(Counter(latest.values()))
save_json(run / 'artifacts/run-summary.json', summary)
save_json(run / 'artifacts/test-accounting.json', accounting)
print(json.dumps(summary, indent=2))
