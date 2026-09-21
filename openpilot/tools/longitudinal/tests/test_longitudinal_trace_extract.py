import json

import pytest

from openpilot.selfdrive.controls.lib.longitudinal_trace import (
  MOTION_TRACE_SCHEMA, TRACE_EVENT, TRACE_FIELDS, TRACE_SCHEMA, UNIFIED_TRACE_FIELDS, LongitudinalTrace,
)
from openpilot.selfdrive.controls.tests.test_longitudinal_trace_observation import InputState, enqueue_attempt, observation_result, planner_state
from openpilot.tools.longitudinal.extract_longitudinal_trace import decode_trace, trace_json_row


HISTORICAL_MOTION_FIELDS = TRACE_FIELDS + (
  'intent_version', 'intent_reasons', 'intent_valid', 'moving_lead_evidence', 'release_evidence',
  'allow_creep', 'reference_accel', 'constraint_accel', 'reference_overridden', 'constraint_source', 'intent_ready',
)


def record(schema=TRACE_SCHEMA, *, with_detail=False):
  fields = {1: TRACE_FIELDS, 2: HISTORICAL_MOTION_FIELDS, 3: UNIFIED_TRACE_FIELDS}[schema]
  sample = dict.fromkeys(fields, 0)
  sample.update(sequence=1, plan_mono=20_000, modelV2_mono=19_000)
  message = {'event': TRACE_EVENT, 'schema': schema, 'fields': list(fields), 'samples': [list(sample.values())], 'dropped': 2, 'failures': 0}
  if with_detail:
    message.update(
      detail_schema=1,
      details=[{
        'session': '7', 'sequence': '1', 'plan_mono': '20000', 'outcome': 'completed',
        'inputs': [{'service': 'modelV2', 'monoTime': '19000', 'idKnown': True, 'knownFlags': 31, 'flags': 31, 'kind': 'consumerSnapshot'}],
        'native': {'solve_calls': '2', 'reset_calls': '1', 'params': [[0.1, 0.2]]},
      }],
      counters={'session': '7', 'sequence': '1', 'disabled': False, 'closed': False, 'queue_dropped': 2},
      counter_scope='cumulative before this batch send attempt',
      delivery_scope='local queue and transport send acceptance only; no logger or disk acknowledgement',
    )
  return {'msg': message}


def emit_record(recorder):
  events = []
  recorder.emit = lambda event, **payload: events.append({'msg': {'event': event, **payload}})
  recorder._emit(list(recorder.queue))
  assert len(events) == 1
  return json.loads(json.dumps(events[0], allow_nan=False))


def test_decode_carries_exact_plan_and_input_timestamps():
  rows = decode_trace(json.dumps(record()))
  assert len(rows) == 1
  assert rows[0]['plan_mono'] == 20_000
  assert rows[0]['modelV2_mono'] == 19_000
  assert rows[0]['sequence'] == 1
  assert rows[0]['dropped'] == 2
  assert rows[0]['failures'] == 0
  assert rows[0]['detail'] is None
  assert rows[0]['session'] is None


@pytest.mark.parametrize('schema', [1, 2, 3])
@pytest.mark.parametrize('with_detail', [False, True])
def test_historical_and_current_field_contracts(schema, with_detail):
  payload = record(schema, with_detail=with_detail)
  row, = decode_trace(json.dumps(payload))
  assert row['schema'] == schema
  assert row['field_contract'] == payload['msg']['fields']
  assert ('intent_ready' in row) == (schema >= 2)
  assert ('intent_mode' in row) == (schema == 3)
  assert ('braking_accel' in row) == (schema == 3)
  if with_detail:
    assert row['detail'] == payload['msg']['details'][0]
    assert row['session'] == '7'
  else:
    assert row['detail_schema'] is None
    assert row['counters'] is None


@pytest.mark.parametrize('unified', [False, True])
def test_producer_to_decoder_preserves_attempt_and_delivery_evidence(unified):
  recorder = LongitudinalTrace()
  recorder.session = (1 << 64) - 3
  recorder.sequence = (1 << 53) + 7
  _, state = enqueue_attempt(recorder, unified=unified, mono=(1 << 64) - 2)
  payload = emit_record(recorder)['msg']
  row, = decode_trace(json.dumps({'msg': payload}))
  assert row['schema'] == (MOTION_TRACE_SCHEMA if unified else TRACE_SCHEMA)
  assert row['plan_mono'] == (1 << 64) - 2
  assert row['sequence'] == (1 << 53) + 8
  assert row['modelV2_mono'] == state.logMonoTime['modelV2']
  assert row['session'] == str(recorder.session)
  assert row['detail'] == payload['details'][0]
  assert row['detail']['native']['trajectory_known'] is True
  assert row['detail']['native']['x'] == [0.0, 5.0, 9.0]
  assert row['detail']['planner_session'] == str((1 << 64) - 1)
  assert row['counters'] == payload['counters']
  assert row['counter_scope'] == payload['counter_scope']
  assert row['delivery_scope'] == payload['delivery_scope']
  assert row['counters']['transport_accepted_batches'] == 0


@pytest.mark.parametrize('unified', [False, True])
def test_unpublished_producer_attempt_does_not_inherit_next_publication(unified):
  recorder = LongitudinalTrace()
  enqueue_attempt(recorder, unified=unified, mono=None)
  enqueue_attempt(recorder, unified=unified, mono=47)
  first, second = decode_trace(json.dumps(emit_record(recorder)))
  assert first['sequence'] == 1
  assert first['plan_mono'] is None
  assert first['detail']['plan_mono'] is None
  assert first['detail']['outcome'] == 'completed'
  assert second['sequence'] == 2
  assert second['plan_mono'] == 47
  assert first['counters']['unpublished'] == 1


@pytest.mark.parametrize('with_detail', [False, True])
def test_historical_unpublished_attempts_stay_unpublished(with_detail):
  payload = record(2, with_detail=with_detail)
  payload['msg']['samples'][0][TRACE_FIELDS.index('plan_mono')] = None
  if with_detail:
    payload['msg']['details'][0]['plan_mono'] = None
  row, = decode_trace(json.dumps(payload))
  assert row['plan_mono'] is None
  assert trace_json_row(row)['plan_mono'] is None


def test_producer_early_return_keeps_unknown_inputs_and_solver_evidence():
  recorder, planner, state = LongitudinalTrace(), planner_state(True), InputState()
  state.logMonoTime['modelV2'] = float((1 << 61) + 1)
  del state.logMonoTime['radarState']
  del state.alive['carState']
  recorder.begin_attempt(planner, state)
  observation_result(planner, outcome='invalid_model', updated=False)
  recorder.finish_attempt(planner)
  recorder.publish(51)
  row, = decode_trace(json.dumps(emit_record(recorder)))
  assert row['detail']['outcome'] == 'invalid_model'
  assert row['mpc_accel'] is None
  assert row['solver_status'] is None
  assert row['detail']['native']['solve_attempted'] is False
  assert row['detail']['native']['trajectory_known'] is False
  inputs = {item['service']: item for item in row['detail']['inputs']}
  assert inputs['modelV2']['monoTime'] is None
  assert inputs['modelV2']['idKnown'] is False
  assert 'radarState' not in inputs
  assert not inputs['carState']['knownFlags'] & 2


def test_producer_solver_failure_preserves_inputs_without_a_successful_trajectory():
  recorder = LongitudinalTrace()
  enqueue_attempt(recorder, unified=True, status=2)
  row, = decode_trace(json.dumps(emit_record(recorder)))
  assert row['detail']['outcome'] == 'mpc_solver_status'
  native = row['detail']['native']
  assert native['solver_return'] == 2
  assert native['params'][0][2] == 2.0
  assert native['trajectory_known'] is False
  assert native['x'] is None


def test_json_export_is_lossless_for_float_based_readers_without_changing_python_api():
  recorder = LongitudinalTrace()
  recorder.sequence = (1 << 53) + 7
  enqueue_attempt(recorder, unified=True, mono=(1 << 64) - 1)
  row, = decode_trace(json.dumps(emit_record(recorder)))
  row['log_mono'] = (1 << 64) - 2
  exported = json.loads(json.dumps(trace_json_row(row), allow_nan=False), parse_int=float)
  for name in ('sequence', 'plan_mono', 'modelV2_mono', 'radar_model_mono', 'log_mono'):
    assert type(row[name]) is int
    assert exported[name] == str(row[name])
  assert exported['detail']['plan_mono'] == exported['plan_mono']
  assert exported['detail']['sequence'] == exported['sequence']
  assert exported['a_target'] == row['a_target']
  assert isinstance(exported['solver_status'], float)


@pytest.mark.parametrize('unrelated', [None, [], 'unrelated', {}, {'msg': 'status'}, {'msg': {'event': 'plannerd.phase_timing'}}])
def test_other_log_events_are_ignored(unrelated):
  assert decode_trace(json.dumps(unrelated)) == []


@pytest.mark.parametrize('invalid', ['schema', 'fields', 'truncated', 'nested', 'sequence', 'plan_mono', 'oversized'])
def test_bad_trace_fails_loudly(invalid):
  payload = record()
  message = payload['msg']
  if invalid == 'schema':
    message['schema'] = 99
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


@pytest.mark.parametrize('name', ['sequence', 'plan_mono', 'modelV2_mono', 'radar_model_mono'])
@pytest.mark.parametrize('value', [True, 1.0, -1, 1 << 64, '1.5', 'unknown'])
def test_inexact_or_out_of_range_sample_identities_are_rejected(name, value):
  payload = record()
  payload['msg']['samples'][0][TRACE_FIELDS.index(name)] = value
  with pytest.raises(ValueError, match='identity'):
    decode_trace(json.dumps(payload))


@pytest.mark.parametrize('mutate', [
  pytest.param(lambda p: p.update(schema=True), id='boolean-schema'),
  pytest.param(lambda p: p.update(schema=1.0), id='float-schema'),
  pytest.param(lambda p: p.update(details=[]), id='misaligned-details'),
  pytest.param(lambda p: p.update(details=None), id='null-detail-batch'),
  pytest.param(lambda p: p.pop('details'), id='missing-detail-batch'),
  pytest.param(lambda p: p.pop('detail_schema'), id='missing-detail-schema'),
  pytest.param(lambda p: p.update(detail_schema=2), id='unknown-detail-schema'),
  pytest.param(lambda p: p.update(detail_schema=True), id='boolean-detail-schema'),
  pytest.param(lambda p: p['details'].__setitem__(0, []), id='invalid-detail'),
  pytest.param(lambda p: p['details'][0].update(sequence='2'), id='conflicting-sequence'),
  pytest.param(lambda p: p['details'][0].update(sequence=None), id='unknown-sequence'),
  pytest.param(lambda p: p['details'][0].update(plan_mono='20001'), id='conflicting-plan'),
  pytest.param(lambda p: p['details'][0].update(plan_mono=None), id='conflicting-unpublished'),
  pytest.param(lambda p: p['details'][0].update(session='8'), id='conflicting-detail-session'),
  pytest.param(lambda p: p.update(session='8'), id='conflicting-batch-session'),
  pytest.param(lambda p: p['details'][0].update(planner_session=1.0), id='inexact-planner-session'),
  pytest.param(lambda p: p['details'][0].update(start_mono='200', end_mono='100'), id='reversed-attempt-interval'),
  pytest.param(lambda p: p['details'][0].update(inputs={}), id='invalid-input-list'),
  pytest.param(lambda p: p['details'][0]['inputs'].append(p['details'][0]['inputs'][0]), id='duplicate-input-service'),
  pytest.param(lambda p: p['details'][0]['inputs'][0].update(monoTime=19_000.0), id='inexact-input-id'),
  pytest.param(lambda p: p['details'][0]['inputs'][0].update(monoTime='19001'), id='conflicting-input-id'),
  pytest.param(lambda p: p['details'][0]['inputs'][0].update(monoTime=None), id='missing-known-input-id'),
  pytest.param(lambda p: p['details'][0]['inputs'][0].update(idKnown=False), id='identity-marked-unknown'),
  pytest.param(lambda p: p['details'][0]['inputs'][0].update(idKnown=1), id='invalid-input-status'),
  pytest.param(lambda p: p['details'][0]['inputs'][0].update(knownFlags=0), id='contradictory-input-health'),
  pytest.param(lambda p: p['details'][0]['inputs'][0].update(flags=True), id='invalid-input-health'),
  pytest.param(lambda p: p['details'][0].update(native=[]), id='invalid-native-detail'),
  pytest.param(lambda p: p['details'][0]['native'].update(solve_calls=1.0), id='inexact-solve-counter'),
  pytest.param(lambda p: p['details'][0]['native'].update(params=[[float('inf')]]), id='nonfinite-native-value'),
  pytest.param(lambda p: p['samples'][0].__setitem__(TRACE_FIELDS.index('a_target'), float('nan')), id='nonfinite-sample'),
  pytest.param(lambda p: p.update(counters=[]), id='invalid-counters'),
  pytest.param(lambda p: p['counters'].update(sequence=1.0), id='inexact-counter-sequence'),
  pytest.param(lambda p: p['counters'].update(session='0'), id='zero-counter-session'),
  pytest.param(lambda p: p['counters'].update(queue_dropped=-1), id='negative-counter'),
  pytest.param(lambda p: p['counters'].update(closed=0), id='invalid-counter-status'),
  pytest.param(lambda p: p.update(dropped=-1), id='negative-dropped'),
  pytest.param(lambda p: p.update(failures=True), id='invalid-failures'),
  pytest.param(lambda p: p.update(counter_scope=[]), id='invalid-counter-scope'),
  pytest.param(lambda p: p.update(delivery_scope=[]), id='invalid-delivery-scope'),
])
def test_malformed_metadata_fails_loudly(mutate):
  payload = record(with_detail=True)
  mutate(payload['msg'])
  with pytest.raises(ValueError):
    decode_trace(json.dumps(payload))
