import json

import pytest

from openpilot.selfdrive.controls.lib.longitudinal_trace import TRACE_EVENT, TRACE_FIELDS, TRACE_SCHEMA
from openpilot.selfdrive.controls.tests.extract_longitudinal_trace import decode_trace


def record():
  sample = dict.fromkeys(TRACE_FIELDS, 0)
  sample.update(sequence=1, plan_mono=20_000, modelV2_mono=19_000)
  return {'msg': {'event': TRACE_EVENT, 'schema': TRACE_SCHEMA, 'fields': list(TRACE_FIELDS), 'samples': [list(sample.values())], 'dropped': 2, 'failures': 0}}


def test_decode_carries_exact_plan_and_input_timestamps():
  rows = decode_trace(json.dumps(record()))
  assert len(rows) == 1
  assert rows[0]['plan_mono'] == 20_000
  assert rows[0]['modelV2_mono'] == 19_000
  assert rows[0]['sequence'] == 1
  assert rows[0]['dropped'] == 2
  assert rows[0]['failures'] == 0


@pytest.mark.parametrize('unrelated', [None, [], 'unrelated', {}, {'msg': 'status'}, {'msg': {'event': 'plannerd.phase_timing'}}])
def test_other_log_events_are_ignored(unrelated):
  assert decode_trace(json.dumps(unrelated)) == []


@pytest.mark.parametrize('invalid', ['schema', 'fields', 'truncated', 'nested', 'sequence', 'plan_mono', 'oversized'])
def test_bad_trace_fails_loudly(invalid):
  payload = record()
  message = payload['msg']
  if invalid == 'schema':
    message['schema'] += 1
  elif invalid == 'fields':
    message['fields'] = message['fields'][::-1]
  elif invalid == 'truncated':
    message['samples'][0].pop()
  elif invalid == 'nested':
    message['samples'][0][-1] = {}
  elif invalid in ('sequence', 'plan_mono'):
    message['samples'][0][TRACE_FIELDS.index(invalid)] = 0
  elif invalid == 'oversized':
    message['samples'] *= 41
  with pytest.raises(ValueError):
    decode_trace(json.dumps(payload))
