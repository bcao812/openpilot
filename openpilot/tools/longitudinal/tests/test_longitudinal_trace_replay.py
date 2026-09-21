from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from openpilot.tools.longitudinal import replay_longitudinal_trace as replay
from openpilot.tools.longitudinal.replay_pedal_full_routes import ROW_FIELDS, NATIVE_FIELDS
from openpilot.tools.longitudinal.replay_pedal_switches import save_json, sha256


def prepare_results(root, *, warmup=0, after_score=0):
  corpus = root / 'corpus'
  corpus.mkdir()
  audit = root / 'audit'
  audit.mkdir()
  save_json(corpus / 'manifest.json', {'audit': str(audit)})
  native = dict.fromkeys(NATIVE_FIELDS, 0.0)
  for field in ('enabled', 'cc_enabled', 'long_active', 'long_state', 'cs_valid', 'cc_valid', 'sd_valid', 'can_valid'):
    native[field] = 1.0
  native['can_on'] = 5.0
  native_path = audit / 'route.native.npz'
  np.savez_compressed(native_path, frames=np.array([[native[field] for field in NATIVE_FIELDS]]))
  window = {'id': '000', 'route': 'route', 'group': 'normal', 'reason': 'fixed', 'audit_hashes': {native_path.name: sha256(native_path)}}
  selection = {'corpus': str(corpus), 'manifest_sha256': sha256(corpus / 'manifest.json'), 'windows': [window]}
  calls = dict.fromkeys(('total', *replay.ReplayDiagnostics.REGIONS), 0)
  calls.update(total=warmup + 1 + after_score, warmup=warmup, score=1, after_score=after_score)
  metadata = {
    'window': dict(window, cache=window['id']),
    'coverage': dict.fromkeys(('plans', 'matched_plans', 'controls', 'matched_controls', 'sendcan', 'matched_sendcan'), 1),
    'trace': {'attempted': calls['total'], 'samples': calls['total'], 'dropped': 0, 'failures': 0, 'disabled': False},
    'solver_failures': 0,
    'capabilities': {'requested_mode': 2, 'planner_mode': 2, 'controller_mode': 2},
    'source_hashes': dict.fromkeys(replay.SOURCE_PATHS, 'a' * 64),
    'diagnostics': {
      'diagnostic_schema': replay.DIAGNOSTIC_SCHEMA_VERSION, 'status': 'COMPLETED', 'initialized': True, 'native_supported': True,
      'counts': {layer: {kind: deepcopy(calls) for kind in ('attempted', 'completed')} for layer in replay.ReplayDiagnostics.LAYERS},
    },
  }
  metadata['diagnostics']['coverage'] = deepcopy(metadata['coverage'])
  before_metadata = deepcopy(metadata)
  before_metadata['trace'].update(attempted=0, samples=0)
  before = {kind: {field: np.zeros(1) for field in fields} for kind, fields in ROW_FIELDS.items()}
  after = {kind: {field: values.copy() for field, values in rows.items()} for kind, rows in before.items()}
  return selection, before_metadata, metadata, before, after


@pytest.mark.parametrize(
  'condition',
  ['equal', 'plan_accel', 'control_accel', 'can_gas', 'time', 'stop', 'schema', 'native', 'missing_trace', 'window', 'lost_sample', 'failed_writer',
   'requested_mode', 'planner_mode', 'controller_mode'],
)
def test_sampled_trace_evaluation_rejects_any_control_or_input_drift(tmp_path, condition):
  selection, metadata, after_metadata, before, after = prepare_results(tmp_path)
  if condition in ('plan_accel', 'control_accel'):
    after[condition.split('_')[0]]['accel'][0] = 0.01
  elif condition == 'can_gas':
    after['can']['can_gas'][0] = 1
  elif condition == 'time':
    after['plan']['mono'][0] = 1
  elif condition == 'stop':
    after['plan']['stop'][0] = 1
  elif condition == 'schema':
    selection['manifest_sha256'] = 'changed'
  elif condition == 'native':
    selection['windows'][0]['audit_hashes']['route.native.npz'] = 'changed'
  elif condition == 'missing_trace':
    after_metadata['trace']['attempted'] = 0
  elif condition == 'window':
    metadata['window']['start'] = 99.0
  elif condition == 'lost_sample':
    after_metadata['trace']['samples'] = 0
  elif condition == 'failed_writer':
    after_metadata['trace']['failures'] = 1
  if condition in ('requested_mode', 'planner_mode', 'controller_mode'):
    after_metadata['capabilities'][condition] = 0
  with patch.object(replay, 'load_window', side_effect=[(metadata, before), (after_metadata, after)]):
    if condition == 'equal':
      result = replay.evaluate(selection, tmp_path)
      assert result[0]['exact_equal']
      assert result[0]['can']['extra_brake_frames'] == 0
    else:
      with pytest.raises(ValueError):
        replay.evaluate(selection, tmp_path)


def test_trace_covers_warmup_score_and_after_score_calls(tmp_path):
  selection, before_metadata, after_metadata, before, after = prepare_results(tmp_path, warmup=1, after_score=1)
  # Controller cycles without an ACC frame still count as executed calls.
  for metadata in (before_metadata, after_metadata):
    for counts in metadata['diagnostics']['counts']['carcontroller'].values():
      counts['total'] += 1
      counts['score'] += 1
  with patch.object(replay, 'load_window', side_effect=[(before_metadata, before), (after_metadata, after)]):
    result = replay.evaluate(selection, tmp_path)[0]
  assert result['trace']['attempted'] == 3
  assert result['coverage']['plans'] == 1
  assert result['execution_counts']['planner']['completed'] == {
    'total': 3, 'read_only_prefix': 0, 'dependency_prefix': 0, 'warmup': 1, 'score': 1, 'after_score': 1,
  }
  assert result['execution_counts']['carcontroller']['completed']['score'] == 2


def test_run_window_leaves_all_trace_publications_to_the_planner(tmp_path, monkeypatch):
  selection, _, metadata, _, _ = prepare_results(tmp_path, warmup=1, after_score=1)
  published = []

  class Recorder:
    def __init__(self, *, emit):
      self.emit = emit
      self.sequence = self.dropped = self.failures = 0
      self.disabled = False

    def start(self):
      return True

    def publish(self, mono):
      published.append(mono)
      self.emit(replay.TRACE_EVENT, samples=[mono])

    def close(self):
      pass

  class Planner:
    def __init__(self, *, trace=None):
      self.trace = trace

    def update(self):
      self.trace.sequence += 1

    def publish(self, mono):
      self.trace.publish(mono)

  def replay_window(_cache, window, *, row_sink, **_options):
    planner = Planner()
    for index, mono in enumerate((1_000_000_000, 1_100_000_000, 1_200_000_000)):
      planner.update()
      planner.publish(mono)
      if index == 1:
        for kind, fields in ROW_FIELDS.items():
          row_sink(kind, dict.fromkeys(fields, 0) | {'mono': mono, 't': mono / 1e9})
    result = deepcopy(metadata)
    result.update(window=window, plan_rows=[], control_rows=[], can_rows=[], planner_error=None, switches={})
    return result

  monkeypatch.setattr(replay, 'LongitudinalPlanner', Planner)
  monkeypatch.setattr(replay, 'LongitudinalTrace', Recorder)
  monkeypatch.setattr(replay, 'replay_window', replay_window)
  monkeypatch.setattr(replay, 'window_messages', lambda *_: iter(()))
  monkeypatch.setattr(replay, 'source_fingerprints', lambda: metadata['source_hashes'].copy())
  destination = tmp_path / 'replay'
  replay.run_window((Path(selection['corpus']), destination, selection['windows'][0], True))
  result = json.loads((destination / 'on' / '000.json').read_text())
  assert published == [1_000_000_000, 1_100_000_000, 1_200_000_000]
  assert result['trace']['attempted'] == result['trace']['samples'] == 3
  assert result['coverage']['plans'] == 1


@pytest.mark.parametrize('condition', [
  'missing', 'empty', 'missing_dependency', 'unexpected_dependency', 'unknown_digest', 'invalid_digest', 'changed_source',
])
def test_trace_evaluation_requires_bound_matching_source_identities(tmp_path, condition):
  selection, before_metadata, after_metadata, before, after = prepare_results(tmp_path)
  dependency = 'openpilot/common/longitudinal_logging.py'
  for metadata in (before_metadata, after_metadata):
    if condition == 'missing':
      metadata.pop('source_hashes')
    elif condition == 'empty':
      metadata['source_hashes'] = {}
    elif condition == 'missing_dependency':
      metadata['source_hashes'].pop(dependency)
    elif condition == 'unexpected_dependency':
      metadata['source_hashes']['unbound.py'] = 'a' * 64
    elif condition in ('unknown_digest', 'invalid_digest'):
      metadata['source_hashes'][dependency] = None if condition == 'unknown_digest' else 'changed'
  if condition == 'changed_source':
    after_metadata['source_hashes'][dependency] = 'b' * 64
  with patch.object(replay, 'load_window', side_effect=[(before_metadata, before), (after_metadata, after)]):
    with pytest.raises(ValueError, match='source identities'):
      replay.evaluate(selection, tmp_path)


@pytest.mark.parametrize('value', [None, True, -1, 2.0, '2'])
def test_equal_but_unknown_or_invalid_modes_cannot_pass_trace_evaluation(tmp_path, value):
  selection, before_metadata, after_metadata, before, after = prepare_results(tmp_path)
  for metadata in (before_metadata, after_metadata):
    metadata['capabilities']['requested_mode'] = value
  with patch.object(replay, 'load_window', side_effect=[(before_metadata, before), (after_metadata, after)]):
    with pytest.raises(ValueError, match='Missing or invalid longitudinal'):
      replay.evaluate(selection, tmp_path)


@pytest.mark.parametrize('condition', [
  'missing', 'schema', 'unfinished', 'uninitialized', 'unsupported', 'missing_counts', 'missing_region',
  'unknown_count', 'negative_count', 'boolean_count', 'fractional_count', 'wrong_total', 'unfinished_call',
  'wrong_score', 'missing_coverage', 'invalid_coverage', 'incomplete_coverage', 'changed_execution',
])
def test_trace_evaluation_rejects_missing_invalid_or_different_execution(tmp_path, condition):
  selection, before_metadata, after_metadata, before, after = prepare_results(tmp_path, warmup=1, after_score=1)
  diagnostics = after_metadata['diagnostics']
  calls = diagnostics['counts']['planner']
  if condition == 'missing':
    after_metadata.pop('diagnostics')
  elif condition == 'schema':
    diagnostics['diagnostic_schema'] = 0
  elif condition == 'unfinished':
    diagnostics['status'] = 'RUNNING'
  elif condition == 'uninitialized':
    diagnostics['initialized'] = False
  elif condition == 'unsupported':
    diagnostics['native_supported'] = None
  elif condition == 'missing_counts':
    diagnostics['counts'] = None
  elif condition == 'missing_region':
    calls['attempted'].pop('warmup')
  elif condition in ('unknown_count', 'negative_count', 'boolean_count', 'fractional_count'):
    calls['attempted']['warmup'] = {'unknown_count': None, 'negative_count': -1, 'boolean_count': True, 'fractional_count': 1.0}[condition]
  elif condition == 'wrong_total':
    calls['attempted']['total'] += 1
  elif condition == 'unfinished_call':
    calls['attempted']['total'] += 1
    calls['attempted']['after_score'] += 1
  elif condition == 'wrong_score':
    for counts in calls.values():
      counts['total'] -= 1
      counts['score'] -= 1
  elif condition == 'missing_coverage':
    diagnostics['coverage'] = None
  elif condition == 'invalid_coverage':
    after_metadata['coverage']['plans'] = diagnostics['coverage']['plans'] = True
  elif condition == 'incomplete_coverage':
    after_metadata['coverage']['matched_plans'] = diagnostics['coverage']['matched_plans'] = 2
  elif condition == 'changed_execution':
    for counts in calls.values():
      counts['warmup'] += 1
      counts['after_score'] -= 1
  with patch.object(replay, 'load_window', side_effect=[(before_metadata, before), (after_metadata, after)]):
    with pytest.raises(ValueError, match='replay|Replay'):
      replay.evaluate(selection, tmp_path)


@pytest.mark.parametrize('condition', ['missing', 'unknown_count', 'negative_count', 'boolean_count', 'unknown_disabled', 'active_off_arm'])
def test_trace_evaluation_rejects_unknown_or_invalid_capture(tmp_path, condition):
  selection, before_metadata, after_metadata, before, after = prepare_results(tmp_path)
  if condition == 'missing':
    after_metadata.pop('trace')
  elif condition == 'unknown_count':
    after_metadata['trace']['samples'] = None
  elif condition == 'negative_count':
    after_metadata['trace'].update(samples=2, dropped=-1)
  elif condition == 'boolean_count':
    after_metadata['trace']['samples'] = True
  elif condition == 'unknown_disabled':
    after_metadata['trace']['disabled'] = None
  elif condition == 'active_off_arm':
    before_metadata['trace'].update(samples=1, attempted=1)
  with patch.object(replay, 'load_window', side_effect=[(before_metadata, before), (after_metadata, after)]):
    with pytest.raises(ValueError, match='Incomplete diagnostic capture'):
      replay.evaluate(selection, tmp_path)


def test_local_corpus_required_before_any_replay(tmp_path):
  with patch('sys.argv', ['replay_longitudinal_trace', 'run', '--selection', str(tmp_path / 'missing'), '--output', str(tmp_path)]):
    with pytest.raises(FileNotFoundError):
      replay.main()
  assert not list(Path(tmp_path).iterdir())
