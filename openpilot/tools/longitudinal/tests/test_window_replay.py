from unittest.mock import patch

import numpy as np
import pytest

from openpilot.cereal import messaging
from openpilot.tools.longitudinal import window_replay as replay
from openpilot.tools.longitudinal.replay_pedal_full_routes import ROW_FIELDS
from openpilot.tools.longitudinal.replay_pedal_switches import save_json, sha256


def message(service, timestamp):
  result = messaging.new_message(service)
  result.logMonoTime = round(timestamp * 1e9)
  return result.as_reader()


@pytest.mark.parametrize('condition', ['valid', 'future', 'missing', 'changed'])
def test_parameter_preload_respects_arrival_and_warmup(tmp_path, condition):
  window = {'parameter_segment': {'cache': 'params.rlog.zst', 'sha256': 'expected'}, 'warmup': 10.0, 'end': 11.0, 'origin': 0.0, 'segments': []}
  parameters = [message('carParams', 10.01 if condition == 'future' else 1.0)]
  if condition != 'missing':
    parameters.append(message('carParamsSP', 1.1))
  messages = [message('carParams', 1.0)] + [message('modelV2', timestamp) for timestamp in (9.99, 10.0, 11.0, 11.25, 11.26)]
  with (
    patch.object(replay, 'sha256', return_value='unexpected' if condition == 'changed' else 'expected'),
    patch.object(replay, 'LogReader', return_value=parameters),
    patch.object(replay, 'block_messages', return_value=iter(messages)),
  ):
    if condition == 'valid':
      recovered = list(replay.window_messages(tmp_path, window, None))
      assert [entry.which() for entry in recovered] == ['carParams', 'carParamsSP', 'modelV2', 'modelV2', 'modelV2']
      assert [entry.logMonoTime / 1e9 for entry in recovered[2:]] == [10.0, 11.0, 11.25]
    else:
      with pytest.raises(ValueError):
        list(replay.window_messages(tmp_path, window, None))


@pytest.mark.parametrize('condition', ['valid', 'schema', 'identity', 'coverage', 'hash', 'shape', 'rows'])
def test_sampled_results_validate_identity_and_complete_coverage(tmp_path, condition):
  destination = tmp_path / 'off'
  destination.mkdir()
  values = {kind: np.zeros((1, len(fields))) for kind, fields in ROW_FIELDS.items()}
  if condition == 'shape':
    values['plan'] = values['plan'][:, :-1]
  path = destination / '000-0000.npz'
  np.savez_compressed(path, **values)
  metadata = {
    'mode': 'off',
    'window': {'id': '000'},
    'fields': {kind: list(fields) for kind, fields in ROW_FIELDS.items()},
    'coverage': dict.fromkeys(('plans', 'matched_plans', 'controls', 'matched_controls', 'sendcan', 'matched_sendcan'), 1),
    'chunks': [{'file': path.name, 'sha256': sha256(path), 'rows': dict.fromkeys(ROW_FIELDS, 1)}],
  }
  if condition == 'schema':
    metadata['fields']['plan'].reverse()
  elif condition == 'identity':
    metadata['mode'] = 'median'
  elif condition == 'coverage':
    metadata['coverage']['matched_sendcan'] = 0
  elif condition == 'hash':
    metadata['chunks'][0]['sha256'] = 'changed'
  elif condition == 'rows':
    metadata['coverage']['plans'] = metadata['coverage']['matched_plans'] = 2
  save_json(destination / '000.json', metadata)
  if condition == 'valid':
    recovered, arrays = replay.load_window(tmp_path, 'off', '000')
    assert recovered == metadata
    assert all(len(arrays[kind]['mono']) == 1 for kind in ROW_FIELDS)
  else:
    with pytest.raises(ValueError):
      replay.load_window(tmp_path, 'off', '000')
