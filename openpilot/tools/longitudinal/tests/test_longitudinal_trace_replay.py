from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from openpilot.tools.longitudinal import replay_longitudinal_trace as replay
from openpilot.tools.longitudinal.replay_pedal_full_routes import ROW_FIELDS, NATIVE_FIELDS
from openpilot.tools.longitudinal.replay_pedal_switches import save_json, sha256


def prepare_results(root):
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
  metadata = {
    'window': dict(window, cache=window['id']),
    'coverage': dict.fromkeys(('plans', 'matched_plans', 'controls', 'matched_controls', 'sendcan', 'matched_sendcan'), 1),
    'trace': {'attempted': 1, 'samples': 1, 'dropped': 0, 'failures': 0, 'disabled': False},
    'solver_failures': 0,
  }
  before = {kind: {field: np.zeros(1) for field in fields} for kind, fields in ROW_FIELDS.items()}
  after = {kind: {field: values.copy() for field, values in rows.items()} for kind, rows in before.items()}
  return selection, metadata, before, after


@pytest.mark.parametrize(
  'condition',
  ['equal', 'plan_accel', 'control_accel', 'can_gas', 'time', 'stop', 'schema', 'native', 'missing_trace', 'window', 'lost_sample', 'failed_writer'],
)
def test_sampled_trace_evaluation_rejects_any_control_or_input_drift(tmp_path, condition):
  selection, metadata, before, after = prepare_results(tmp_path)
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
    metadata['trace']['attempted'] = 0
  elif condition == 'window':
    metadata['window']['start'] = 99.0
  elif condition == 'lost_sample':
    metadata['trace']['samples'] = 0
  elif condition == 'failed_writer':
    metadata['trace']['failures'] = 1
  with patch.object(replay, 'load_window', side_effect=[(metadata, before), (metadata, after)]):
    if condition == 'equal':
      result = replay.evaluate(selection, tmp_path)
      assert result[0]['exact_equal']
      assert result[0]['can']['extra_brake_frames'] == 0
    else:
      with pytest.raises(ValueError):
        replay.evaluate(selection, tmp_path)


def test_local_corpus_required_before_any_replay(tmp_path):
  with patch('sys.argv', ['replay_longitudinal_trace', 'run', '--selection', str(tmp_path / 'missing'), '--output', str(tmp_path)]):
    with pytest.raises(FileNotFoundError):
      replay.main()
  assert not list(Path(tmp_path).iterdir())
