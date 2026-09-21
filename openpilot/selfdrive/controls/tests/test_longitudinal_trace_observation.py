import ast
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import zmq

from opendbc.car.honda.values import CAR
from openpilot.common.logging_extra import SwagFormatter, SwagLogger
from openpilot.common.longitudinal_logging import PLANNER_INPUTS
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.longitudinal_trace import (
  DETAIL_SCHEMA, INPUT_SERVICES, LEAD_FIELDS, MOTION_TRACE_SCHEMA, TRACE_FIELDS, UNIFIED_TRACE_FIELDS,
  LongitudinalTrace, TraceSample, TraceSerializationError, UnixDomainSocketHandler, immutable_value, json_value,
)
from openpilot.selfdrive.controls.lib.longitudinal_observation import observe_trace
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION


class InputState(dict):
  def __init__(self):
    lead = SimpleNamespace(**dict.fromkeys(LEAD_FIELDS, 0))
    super().__init__({
      'carState': SimpleNamespace(vEgo=10.0, aEgo=-0.3, gasPressed=False, brakePressed=False),
      'carControl': SimpleNamespace(longActive=True, cruiseControl=SimpleNamespace(override=False)),
      'controlsState': SimpleNamespace(
        forceDecel=False, longControlState='pid', motionState=SimpleNamespace(
          version=MOTION_INTENT_VERSION, enabled=True, planningMode=int(LongitudinalMode.UNIFIED)),
      ),
      'selfdriveState': SimpleNamespace(enabled=True, personality='standard'),
      'radarState': SimpleNamespace(mdMonoTime=(1 << 61) + 19, leadOne=lead, leadTwo=lead),
      'modelV2': SimpleNamespace(action=SimpleNamespace(desiredAcceleration=-0.2, shouldStop=False)),
    })
    self.logMonoTime = {service: (1 << 61) + index + 1 for index, service in enumerate(PLANNER_INPUTS)}
    for field in ('valid', 'alive', 'freq_ok', 'seen', 'updated'):
      setattr(self, field, dict.fromkeys(PLANNER_INPUTS, True))

  def all_checks(self):
    return all(self.valid.values())


def planner_state(unified=False):
  mpc = SimpleNamespace(
    x0=np.array([0.0, 10.0, -0.3]),
    x_sol=np.array([[0.0, 10.0, -0.3], [5.0, 9.5, -0.2], [9.0, 9.0, -0.1]]),
    params=np.arange(18.0).reshape(3, 6),
    v_solution=np.array([10.0, 9.5, 9.0]),
    a_solution=np.array([-0.3, -0.2, -0.1]),
    j_solution=np.array([0.2, 0.1]),
    yref=np.arange(18.0).reshape(3, 6) / 100,
    a_prev=np.array([-0.2, -0.1, 0.0]),
    solution_status=0, source='lead0',
    diagnostic_solve_calls=1, diagnostic_reset_calls=1,
    diagnostic_last_solver_return=0, diagnostic_last_x0=None,
    diagnostic_last_parameters=None, diagnostic_last_yref=None, diagnostic_last_a_prev=None,
    diagnostic_times=np.array([0.0, 0.5, 1.0]),
  )
  return SimpleNamespace(
    mpc=mpc, unified=unified, motion_enabled=unified,
    longitudinal_mode=LongitudinalMode.UNIFIED if unified else LongitudinalMode.LEGACY,
    a_cruise=-0.2, output_a_target=-0.4, output_should_stop=False,
    allow_throttle=True, fcw=False, v_desired_filter=SimpleNamespace(x=9.9),
    reference_planner=SimpleNamespace(
      reference_accel=-0.45, reference_source='lead0', constraint_accel=1.0,
      reference_overridden=False, constraint_source='cruise',
    ),
    motion_intent=SimpleNamespace(
      version=MOTION_INTENT_VERSION, mode=int(LongitudinalMode.UNIFIED),
      reasons=0, valid=True, moving_lead=False, release_allowed=False, allow_creep=True, ready=True, braking_accel=None,
    ),
    observation=SimpleNamespace(recorder=SimpleNamespace(session=(1 << 64) - 1, identity='scoped-source-identity')),
    observation_result=None,
  )


def solver_attempt(planner, status=0):
  mpc = planner.mpc
  mpc.diagnostic_solve_calls += 1
  mpc.diagnostic_last_solver_return = status
  mpc.diagnostic_last_x0 = tuple(mpc.x0)
  mpc.diagnostic_last_parameters = mpc.params.tolist()
  mpc.diagnostic_last_yref = mpc.yref.tolist()
  mpc.diagnostic_last_a_prev = mpc.a_prev.tolist()
  mpc.solution_status = status
  if status:
    mpc.diagnostic_reset_calls += 1
    for values in (mpc.x_sol, mpc.params, mpc.v_solution, mpc.a_solution, mpc.j_solution):
      values.fill(0.0)


def observation_result(planner, *, outcome='completed', updated=True, completed=True, sequence=1):
  planner.observation_result = {
    'sample': {'sequence': sequence},
    'values': {'outcome': outcome, 'mpcUpdated': updated, 'updateCompleted': completed},
  }


def enqueue_attempt(recorder, *, unified=False, mono=(1 << 61) + 31, status=0):
  planner, state = planner_state(unified), InputState()
  recorder.begin_attempt(planner, state)
  recorder.begin(planner, state, False, 12.0, 2.0)
  solver_attempt(planner, status)
  if not status:
    if unified:
      recorder.finish_unified(planner, False)
    else:
      recorder.finish(planner, [(-0.45, 'lead0', False), (-0.2, 'cruise', False)], False)
  observation_result(planner, outcome='mpc_solver_status' if status else 'completed', sequence=recorder.sequence)
  recorder.finish_attempt(planner)
  recorder.publish(mono)
  return planner, state


def row_dict(sample):
  return dict(zip(sample.fields, sample, strict=True))


@pytest.mark.parametrize('unified', [False, True])
def test_legacy_and_motion_sample_schemas_have_aligned_details(unified):
  events = []
  recorder = LongitudinalTrace(emit=lambda event, **payload: events.append(payload))
  enqueue_attempt(recorder, unified=unified)
  queued = recorder.queue[0]
  expected_fields = UNIFIED_TRACE_FIELDS if unified else TRACE_FIELDS
  assert queued.fields == expected_fields
  assert queued.schema == MOTION_TRACE_SCHEMA if unified else queued.schema == 1
  row = row_dict(queued)
  assert row['plan_mono'] == (1 << 61) + 31
  assert row['a_target'] == -0.4
  assert row['previous_cruise'] == -0.2
  assert row['modelV2_mono'] == InputState().logMonoTime['modelV2']
  if unified:
    assert row['mpc_accel'] == -0.45
    assert row['cruise_raw'] is None
    assert row['mpc_stop'] is None
    assert row['intent_ready'] is True
  else:
    assert row['mpc_accel'] == -0.45
    assert row['cruise_raw'] == -0.2
  recorder._emit(list(recorder.queue))
  assert events[0]['fields'] == expected_fields
  assert events[0]['detail_schema'] == DETAIL_SCHEMA == 1
  assert len(events[0]['details']) == len(events[0]['samples']) == 1
  assert events[0]['details'][0]['outcome'] == 'completed'
  assert events[0]['details'][0]['native']['trajectory_known']
  assert events[0]['details'][0]['native']['x'] == (0.0, 5.0, 9.0)
  json.dumps(events, allow_nan=False)


@pytest.mark.parametrize('outcome', ['motion_protocol', 'invalid_vehicle_state', 'uninitialized_cruise', 'invalid_model', 'cold_standby'])
def test_early_returns_are_observable_without_stale_solver_values(outcome):
  recorder, planner, state = LongitudinalTrace(), planner_state(True), InputState()
  planner.mpc.diagnostic_last_parameters = [[99.0]]
  planner.mpc.diagnostic_last_x0 = (0.0, 100.0, 2.0)
  recorder.begin_attempt(planner, state)
  observation_result(planner, outcome=outcome, updated=False)
  planner.motion_intent.valid = False
  recorder.finish_attempt(planner)
  recorder.publish(37)
  sample = recorder.queue[0]
  row = row_dict(sample)
  assert row['a_target'] == -0.4
  assert row['mpc_accel'] is None
  assert row['mpc_initial_v'] is None
  assert row['v_cruise'] is None
  assert row['solver_status'] is None
  assert row['intent_valid'] is False
  native = sample.detail['native']
  assert sample.detail['outcome'] == outcome
  assert native['solve_known'] and not native['solve_attempted']
  assert native['solver_return'] is None
  assert native['solution_status'] is None
  assert not native['trajectory_known']
  for field in ('initial_state', 'params', 'yref', 'a_prev', 'x', 'v', 'a', 'j'):
    assert native[field] is None
    assert not native['known'][field]


@pytest.mark.parametrize('status', [1, 2, 4, 1001])
def test_solver_failures_keep_input_snapshot_not_reset_sentinels(status):
  recorder = LongitudinalTrace()
  enqueue_attempt(recorder, unified=True, status=status)
  sample = recorder.queue[0]
  assert sample.detail['outcome'] == 'mpc_solver_status'
  native = sample.detail['native']
  assert native['solve_attempted'] is True
  assert native['solver_return'] == status
  assert native['solution_status'] == status
  assert native['resets_this_attempt'] == 1
  assert native['params'][0][2] == 2.0
  assert native['initial_state'] == (0.0, 10.0, -0.3)
  assert native['a_prev'] == (-0.2, -0.1, 0.0)
  assert not native['trajectory_known']
  assert all(native[field] is None for field in ('x', 'v', 'a', 'j'))


def test_input_rejection_before_solve_does_not_reuse_previous_solver_return():
  recorder, planner, state = LongitudinalTrace(), planner_state(True), InputState()
  recorder.begin_attempt(planner, state)
  planner.mpc.diagnostic_last_x0 = (0.0, float('nan'), -0.1)
  planner.mpc.diagnostic_last_parameters = [[float('inf')]]
  planner.mpc.solution_status = 1001
  observation_result(planner, outcome='mpc_invalid_input')
  recorder.finish_attempt(planner)
  recorder.publish(41)
  native = recorder.queue[0].detail['native']
  assert native['solve_known'] and not native['solve_attempted']
  assert native['solver_return'] is None
  assert native['solution_status'] == 1001
  assert native['known']['initial_state']
  assert native['known']['params']
  assert not native['known']['yref']


def test_unknown_solver_counters_are_not_inferred_from_zero_status():
  recorder, planner, state = LongitudinalTrace(), planner_state(), InputState()
  del planner.mpc.diagnostic_solve_calls
  del planner.mpc.diagnostic_reset_calls
  recorder.begin_attempt(planner, state)
  observation_result(planner, updated=False)
  recorder.finish_attempt(planner)
  recorder.publish(43)
  native = recorder.queue[0].detail['native']
  assert not native['solve_known']
  assert native['solve_attempted'] is None
  assert native['solve_calls'] is None
  assert native['reset_calls'] is None
  assert not native['trajectory_known']


def test_current_attempt_clears_unpublished_previous_data():
  recorder = LongitudinalTrace()
  planner, state = planner_state(), InputState()
  recorder.begin_attempt(planner, state)
  recorder.begin(planner, state, False, 15.0, 2.0)
  recorder.finish(planner, [(-1.0, 'lead0', False), (-0.2, 'cruise', False)], False)
  observation_result(planner)
  recorder.finish_attempt(planner)
  recorder.begin_attempt(planner, state)
  observation_result(planner, outcome='invalid_model', updated=False, sequence=2)
  recorder.finish_attempt(planner)
  recorder.publish(47)
  assert recorder.unpublished == 1
  previous, current = recorder.queue
  assert row_dict(previous)['plan_mono'] is None
  assert previous.detail['plan_mono'] is None
  assert row_dict(current)['mpc_accel'] is None
  assert current.detail['sequence'] == '2'
  assert current.detail['outcome'] == 'invalid_model'


def test_all_ids_roundtrip_above_float64_precision():
  events = []
  with patch('openpilot.selfdrive.controls.lib.longitudinal_trace.secrets.randbits', return_value=(1 << 64) - 3):
    recorder = LongitudinalTrace(emit=lambda event, **payload: events.append(payload))
  recorder.sequence = (1 << 53) + 7
  enqueue_attempt(recorder, mono=(1 << 64) - 2)
  recorder._emit(list(recorder.queue))
  detail = json.loads(json.dumps(events))[0]['details'][0]
  assert detail['session'] == str((1 << 64) - 3)
  assert detail['sequence'] == str((1 << 53) + 8)
  assert detail['plan_mono'] == str((1 << 64) - 2)
  assert detail['planner_session'] == str((1 << 64) - 1)
  assert detail['planner_sequence'] == str((1 << 53) + 8)
  assert all(isinstance(item['monoTime'], str) for item in detail['inputs'])
  assert {item['service'] for item in detail['inputs']} == set(PLANNER_INPUTS)


def test_sessions_are_independent_and_zero_random_value_is_not_an_identity():
  first, second = LongitudinalTrace(), LongitudinalTrace()
  assert first.session != second.session
  with patch('openpilot.selfdrive.controls.lib.longitudinal_trace.secrets.randbits', return_value=0):
    assert LongitudinalTrace().session == 1


def test_missing_or_invalid_input_ids_stay_unknown():
  recorder, planner, state = LongitudinalTrace(), planner_state(), InputState()
  state.logMonoTime['modelV2'] = float((1 << 61) + 1)
  del state.logMonoTime['radarState']
  del state.alive['carState']
  recorder.begin_attempt(planner, state)
  observation_result(planner, outcome='invalid_inputs', updated=False)
  recorder.finish_attempt(planner)
  recorder.publish(51)
  inputs = {item['service']: item for item in recorder.queue[0].detail['inputs']}
  assert inputs['modelV2']['monoTime'] is None
  assert not inputs['modelV2']['idKnown']
  assert 'radarState' not in inputs
  assert not inputs['carState']['knownFlags'] & 2


def test_snapshot_is_immutable_before_writer_and_after_next_planner_update():
  recorder, planner, state = LongitudinalTrace(), planner_state(True), InputState()
  recorder.begin_attempt(planner, state)
  recorder.begin(planner, state, False, 12.0, 2.0)
  solver_attempt(planner)
  recorder.finish_unified(planner, False)
  observation_result(planner)
  recorder.finish_attempt(planner)
  for values in (planner.mpc.x_sol, planner.mpc.v_solution, planner.mpc.params, planner.mpc.diagnostic_times):
    values.fill(99.0)
  planner.mpc.diagnostic_last_parameters[0][2] = 99.0
  planner.mpc.diagnostic_last_a_prev[0] = 99.0
  state.logMonoTime['modelV2'] = 99
  state['carState'].vEgo = 99.0
  recorder.publish(53)
  sample = recorder.queue[0]
  assert row_dict(sample)['v_ego'] == 10.0
  assert sample.detail['native']['v'] == (10.0, 9.5, 9.0)
  assert sample.detail['native']['params'][0][2] == 2.0
  assert sample.detail['native']['a_prev'][0] == -0.2
  assert sample.detail['native']['times'] == (0.0, 0.5, 1.0)
  with pytest.raises(TypeError):
    sample.detail['native']['params'][0][2] = 100.0
  with pytest.raises(TypeError):
    sample.detail['native']['known']['x'] = False

  def assert_copied(value):
    assert not isinstance(value, (np.ndarray, list, dict))
    if isinstance(value, Mapping):
      for item in value.values():
        assert_copied(item)
    elif isinstance(value, tuple):
      for item in value:
        assert_copied(item)

  assert_copied(sample.detail)


def test_external_readonly_view_is_copied_before_reusing_trace_snapshot():
  backing = {'native': {'x': [1.0, 2.0]}, 'sequence': (1 << 64) - 1}
  external = MappingProxyType(backing)
  frozen = immutable_value(external)
  pending = immutable_value({'native': frozen, 'plan_mono': None})
  backing['native']['x'][0] = 99.0
  backing['sequence'] = 0
  sample = TraceSample((1,), dict(pending, plan_mono=str((1 << 63) + 1)), 3, ('sequence',))
  assert sample.detail['native']['native']['x'] == (1.0, 2.0)
  assert sample.detail['native']['sequence'] == (1 << 64) - 1
  assert immutable_value(external)['native']['x'] == (99.0, 2.0)
  with pytest.raises(TypeError):
    frozen._data = MappingProxyType({})
  with pytest.raises(TypeError):
    del frozen._data


def test_trace_scalar_fast_path_preserves_numpy_subclass_and_nonfinite_values():
  class IntSubclass(int):
    pass

  class FloatSubclass(float):
    pass

  values = [np.uint64((1 << 64) - 1), np.int64(7), np.float64(1.5), IntSubclass(4), FloatSubclass(2.5),
            True, False, None, float('nan'), float('inf'), -float('inf')]
  frozen = immutable_value(values)
  assert all(type(value) is int for value in (frozen[0], frozen[1], frozen[3]))
  assert all(type(value) is float for value in (frozen[2], frozen[4]))
  assert json_value(frozen) == ((1 << 64) - 1, 7, 1.5, 4, 2.5, True, False, None, None, None, None)


@pytest.mark.parametrize('scalar_type', [int, float, str])
def test_json_scalar_mapping_subclasses_keep_mapping_semantics(scalar_type):
  class ScalarMapping(scalar_type, Mapping):
    def __getitem__(self, key):
      if key != 'native':
        raise KeyError(key)
      return ((1 << 64) - 1, float('inf'))

    def __iter__(self):
      return iter(('native',))

    def __len__(self):
      return 1

  assert json_value(ScalarMapping()) == {'native': ((1 << 64) - 1, None)}


def test_json_scalar_subclasses_preserve_values_without_snapshot_normalization():
  class IntSubclass(int):
    pass

  class FloatSubclass(float):
    pass

  values = (IntSubclass((1 << 64) - 1), FloatSubclass(-0.0), FloatSubclass('nan'),
            FloatSubclass('inf'), FloatSubclass('-inf'), True, False, None)
  converted = json_value(values)
  assert converted[0] is values[0]
  assert converted[1] is values[1]
  assert json.dumps(converted, allow_nan=False) == '[18446744073709551615, -0.0, null, null, null, true, false, null]'


@pytest.mark.parametrize('value', [object(), np.int64(7), np.uint64((1 << 64) - 1), np.float32(1.5)])
def test_json_conversion_requires_nonbuiltin_scalars_to_be_frozen(value):
  with pytest.raises(TypeError, match='immutable JSON snapshot'):
    json_value(value)


@pytest.mark.parametrize('invalid', [{1: 'non-text key'}, {'native': [object()]}])
def test_trace_snapshot_still_rejects_invalid_nested_data(invalid):
  with pytest.raises(TypeError):
    immutable_value(invalid)


def test_nested_nonfinite_values_become_json_null_without_modifying_state():
  events = []
  recorder, planner, state = LongitudinalTrace(emit=lambda event, **payload: events.append(payload)), planner_state(), InputState()
  state['carState'].aEgo = float('nan')
  recorder.begin_attempt(planner, state)
  solver_attempt(planner)
  planner.mpc.diagnostic_last_parameters[0][0] = float('inf')
  planner.mpc.diagnostic_last_yref[0][0] = float('-inf')
  observation_result(planner)
  recorder.finish_attempt(planner)
  recorder.publish(59)
  recorder._emit(list(recorder.queue))
  encoded = json.dumps(events, allow_nan=False)
  payload = json.loads(encoded)[0]
  row = dict(zip(payload['fields'], payload['samples'][0], strict=True))
  assert row['a_ego'] is None
  assert payload['details'][0]['native']['params'][0][0] is None
  assert payload['details'][0]['native']['yref'][0][0] is None
  assert np.isnan(state['carState'].aEgo)
  assert np.isposinf(planner.mpc.diagnostic_last_parameters[0][0])
  assert np.isneginf(planner.mpc.diagnostic_last_yref[0][0])


def test_full_queue_exposes_drops_and_attempt_sequence_gap():
  recorder = LongitudinalTrace(capacity=2, batch_size=1)
  for index in range(5):
    enqueue_attempt(recorder, mono=index + 1)
  assert recorder.dropped == recorder.status()['queue_dropped'] == 3
  assert [sample.detail['sequence'] for sample in recorder.queue] == ['1', '2']
  recorder.queue.popleft()
  enqueue_attempt(recorder, mono=6)
  assert [sample.detail['sequence'] for sample in recorder.queue] == ['2', '6']
  recorder.publish(7)
  assert recorder.sequence == 6


def test_queue_failure_is_counted_and_disabled_attempts_do_not_read_planner():
  class FailingQueue(deque):
    def append(self, value):
      raise RuntimeError('queue unavailable')

  recorder = LongitudinalTrace()
  recorder.queue = FailingQueue()
  enqueue_attempt(recorder)
  assert recorder.disabled
  assert recorder.capture_failures == recorder.failures == 1
  recorder.begin_attempt(object(), object())
  assert recorder.sequence == 2
  assert recorder.disabled_attempts == 1


def test_serialization_failure_is_not_reported_as_transport_loss():
  recorder = LongitudinalTrace(emit=Mock())
  recorder._emit([(object(),)])
  assert recorder.disabled
  assert recorder.serialization_failures == recorder.failures == 1
  assert recorder.serialization_dropped == 1
  assert recorder.transport_failures == 0
  recorder.emit.assert_not_called()


@pytest.mark.parametrize('failure', [RuntimeError('transport error'), TraceSerializationError('bad encoding')])
def test_emitter_exceptions_are_separated_and_fail_open(failure):
  recorder = LongitudinalTrace(emit=Mock(side_effect=failure))
  enqueue_attempt(recorder)
  recorder._emit(list(recorder.queue))
  assert recorder.disabled and recorder.failures == 1
  serialization = isinstance(failure, TraceSerializationError)
  assert recorder.serialization_failures == int(serialization)
  assert recorder.transport_failures == int(not serialization)
  assert recorder.serialization_dropped == int(serialization)
  assert recorder.transport_dropped == int(not serialization)


def test_nonblocking_transport_loss_does_not_silently_pass_or_disable_trace():
  events = []
  recorder = LongitudinalTrace(emit=Mock(side_effect=zmq.Again()))
  enqueue_attempt(recorder)
  recorder._emit(list(recorder.queue))
  assert recorder.transport_failures == recorder.transport_dropped == 1
  assert recorder.transport_accepted_samples == 0
  assert not recorder.disabled
  recorder.emit = lambda event, **payload: events.append(payload)
  recorder._emit(list(recorder.queue))
  assert recorder.transport_accepted_samples == 1
  assert events[0]['counters']['transport_failures'] == 1
  assert events[0]['counters']['transport_accepted_samples'] == 0
  assert 'no logger or disk acknowledgement' in events[0]['delivery_scope']


def test_private_socket_has_bounded_buffers_and_propagates_noblock_failure():
  socket, context = Mock(), Mock()
  context.socket.return_value = socket
  socket.send.side_effect = zmq.Again()
  logger = SwagLogger()
  with patch('openpilot.selfdrive.controls.lib.longitudinal_trace.zmq.Context', return_value=context):
    handler = UnixDomainSocketHandler(SwagFormatter(logger))
    socket.setsockopt.assert_any_call(zmq.SNDHWM, 8)
    socket.setsockopt.assert_any_call(zmq.LINGER, 0)
    socket.setsockopt.assert_any_call(zmq.SNDTIMEO, 0)
    socket.setsockopt.assert_any_call(zmq.IMMEDIATE, 1)
    record = logging.LogRecord('trace', logging.DEBUG, __file__, 1, {'event': 'longitudinal.trace'}, (), None)
    with pytest.raises(zmq.Again):
      handler.emit(record)
    assert socket.send.call_args.args[1] == zmq.NOBLOCK
    encoded = socket.send.call_args.args[0]
    assert encoded[0] == logging.DEBUG
    assert json.loads(encoded[1:])['msg']['event'] == 'longitudinal.trace'
    handler.close()
  socket.close.assert_called_once_with(linger=0)
  context.term.assert_called_once()


def test_private_socket_classifies_its_own_encoding_failure():
  context = Mock()
  context.socket.return_value = Mock()
  formatter = Mock()
  formatter.format_dict.side_effect = TypeError('cannot encode')
  with patch('openpilot.selfdrive.controls.lib.longitudinal_trace.zmq.Context', return_value=context):
    handler = UnixDomainSocketHandler(formatter)
    with pytest.raises(TraceSerializationError):
      handler.emit(logging.LogRecord('trace', logging.DEBUG, __file__, 1, '', (), None))
    context.socket.return_value.send.assert_not_called()
    handler.close()


def test_partial_batch_flushes_without_next_frame_or_shutdown():
  events, emitted = [], threading.Event()

  def emit(event, **payload):
    events.append(payload)
    emitted.set()

  recorder = LongitudinalTrace(emit=emit, flush_interval=0.01)
  assert recorder.start()
  try:
    enqueue_attempt(recorder)
    assert emitted.wait(timeout=1.0)
    assert len(events) == 1
    assert len(events[0]['samples']) == len(events[0]['details']) == 1
    assert not recorder.stop.is_set()
  finally:
    assert recorder.close()


def test_shutdown_flushes_partial_batch_with_aligned_details():
  events = []
  recorder = LongitudinalTrace(emit=lambda event, **payload: events.append(payload), flush_interval=60.0)
  for index in range(3):
    enqueue_attempt(recorder, mono=index + 1)
  assert recorder.start()
  assert recorder.close()
  assert [len(event['samples']) for event in events] == [3]
  assert [len(event['details']) for event in events] == [3]
  assert not recorder.thread.is_alive()


def test_close_is_bounded_when_injected_writer_is_blocked():
  entered, release = threading.Event(), threading.Event()

  def emit(event, **payload):
    entered.set()
    release.wait(timeout=2.0)

  recorder = LongitudinalTrace(emit=emit, capacity=2, batch_size=1)
  assert recorder.start()
  enqueue_attempt(recorder)
  assert entered.wait(timeout=1.0)
  try:
    for index in range(5):
      enqueue_attempt(recorder, mono=index + 1)
    assert len(recorder.queue) == 2
    assert recorder.dropped == 3
    started = time.monotonic()
    assert not recorder.close(timeout=0.01)
    assert time.monotonic() - started < 0.5
    assert recorder.close_timeouts == 1
  finally:
    release.set()
    assert recorder.close()


def test_failed_transport_counts_already_queued_rows_as_discarded():
  recorder = LongitudinalTrace(emit=Mock(side_effect=RuntimeError('failed')), batch_size=1)
  for index in range(4):
    enqueue_attempt(recorder, mono=index + 1)
  recorder.start()
  assert recorder.thread is not None
  recorder.close()
  assert recorder.transport_dropped == 1
  assert recorder.disabled_dropped == 3


def test_failed_writer_initialization_remains_observable_when_marker_exists(tmp_path):
  marker = tmp_path / 'trace'
  marker.touch()
  params = SimpleNamespace(carFingerprint=CAR.HONDA_ACCORD, openpilotLongitudinalControl=True)
  with patch('openpilot.selfdrive.controls.lib.longitudinal_trace.drop_realtime', side_effect=OSError('no priority permission')):
    recorder = LongitudinalTrace.create(params, marker)
  assert recorder is not None
  assert recorder.disabled
  assert not recorder.enabled
  assert recorder.startup_failures == recorder.failures == 1
  assert recorder.close()


def test_original_control_exception_is_not_suppressed_and_shutdown_records_attempt():
  events = []
  recorder = LongitudinalTrace(emit=lambda event, **payload: events.append(payload))
  planner, state = planner_state(), InputState()
  assert recorder.start()
  with pytest.raises(ValueError, match='original control error'):
    try:
      recorder.begin_attempt(planner, state)
      try:
        raise ValueError('original control error')
      finally:
        recorder.finish_attempt(planner)
    finally:
      recorder.close()
  assert recorder.unpublished == 1
  assert events[0]['details'][0]['outcome'] == 'control_exception'
  assert events[0]['details'][0]['plan_mono'] is None
  assert events[0]['samples'][0][TRACE_FIELDS.index('plan_mono')] is None


def test_schema_changes_flush_separate_batches():
  events = []
  recorder = LongitudinalTrace(emit=lambda event, **payload: events.append(payload))
  enqueue_attempt(recorder, unified=False)
  enqueue_attempt(recorder, unified=True)
  assert recorder.start()
  assert recorder.close()
  assert [event['schema'] for event in events] == [1, MOTION_TRACE_SCHEMA]
  for event in events:
    assert len(event['samples']) == len(event['details']) == 1
    assert len(event['samples'][0]) == len(event['fields'])


@pytest.mark.parametrize('interval', [0, -1.0, float('nan'), float('inf')])
def test_flush_interval_is_finite_and_bounded(interval):
  with pytest.raises(ValueError):
    LongitudinalTrace(flush_interval=interval)


def test_old_direct_hooks_remain_compatible_without_claiming_native_evidence():
  recorder, planner, state = LongitudinalTrace(), planner_state(), InputState()
  recorder.begin(planner, state, False, 12.0, 2.0)
  recorder.finish(planner, [(-0.45, 'lead0', False), (-0.2, 'cruise', False)], False)
  recorder.publish(61)
  sample = recorder.queue[0]
  assert isinstance(sample, tuple)
  assert tuple(sample[:2]) == (1, 61)
  assert len(sample) == len(TRACE_FIELDS)
  assert sample.detail['native'] is None
  assert sample.detail['outcome'] == 'legacy_hooks_only'
  assert tuple(INPUT_SERVICES) == tuple(field[:-5] for field in TRACE_FIELDS[2:2 + len(INPUT_SERVICES)])


def test_plain_rows_can_still_be_serialized_by_existing_test_and_reader_contract():
  events = []
  recorder = LongitudinalTrace(emit=lambda event, **payload: events.append(payload))
  recorder._emit([(float('nan'), float('inf'), float('-inf'), 2.0)])
  assert events[0]['samples'] == [(None, None, None, 2.0)]
  assert events[0]['details'] == [None]
  assert events[0]['detail_schema'] == 1


def test_sample_copies_dictionary_payload_instead_of_retaining_live_objects():
  details = {'plan_mono': '71', 'native': {'params': [[1.0, 2.0]]}}
  sample = TraceSample((1, 71), details, 1, ('sequence', 'plan_mono'))
  details['native']['params'][0][0] = 5.0
  assert sample.detail['native']['params'] == ((1.0, 2.0),)


@pytest.mark.parametrize('failure_stage', ['shutdown', 'initialization', 'update', 'publish'])
def test_plannerd_always_closes_trace_without_changing_loop_order(failure_stage):
  calls = []
  trace = SimpleNamespace(disabled=False, close=lambda: calls.append('close'))

  def record(stage):
    calls.append(stage)
    if stage == failure_stage:
      raise ValueError('original ' + stage)

  state = InputState()
  state['selfdriveStateSP'] = SimpleNamespace(buttonsReleaseToggle=False)
  state.updated = {'modelV2': True}
  state.frame = 0

  def update_state():
    if state.frame:
      record('shutdown')
    calls.append('sm_update')
    state.frame += 1

  state.update = update_state
  planner = SimpleNamespace(
    sla=SimpleNamespace(update_buttons=lambda buttons: calls.append('sla_buttons')),
    update=lambda state: record('update'),
    publish=lambda state, publisher: record('publish'),
  )

  def initialize(*arguments, **keywords):
    record('initialization')
    return planner

  def publish_message(service, message):
    calls.append(service)

  namespace = {
    'config_realtime_process': lambda core, priority: calls.append(('realtime', core, priority)),
    'Priority': SimpleNamespace(CTRL_LOW=51),
    'cloudlog': Mock(),
    'Params': lambda: SimpleNamespace(get=lambda *arguments, **keywords: b''),
    'messaging': SimpleNamespace(
      log_from_bytes=lambda *arguments: SimpleNamespace(brand='honda'),
      PubMaster=lambda *arguments: SimpleNamespace(send=publish_message),
      SubMaster=lambda *arguments, **keywords: state,
      new_message=lambda service: SimpleNamespace(driverAssistance=SimpleNamespace()),
    ),
    'car': SimpleNamespace(CarParams=None),
    'custom': SimpleNamespace(CarParamsSP=None),
    'get_gps_location_service': lambda params: 'gpsLocation',
    'LaneDepartureWarning': lambda: SimpleNamespace(
      update=lambda *arguments: calls.append('ldw'), left=False, right=False,
    ),
    'LongitudinalTrace': SimpleNamespace(create=lambda params: trace),
    'LongitudinalPlanner': initialize,
    'PhaseTimer': SimpleNamespace(create=lambda *arguments: None),
    'TRACE_MARKER': '/data/longitudinal_trace',
    'PHASE_TIMER_MARKER': '/data/plannerd_profile',
    'PHASE_TIMER_EVENT': 'plannerd.phase_timing',
    'PHASES': (),
    'PHASE_TIMER_PERIOD': 600,
    'observe_trace': observe_trace,
  }
  path = Path(__file__).resolve().parents[1] / 'plannerd.py'
  source = ast.parse(path.read_text())
  main = next(item for item in source.body if isinstance(item, ast.FunctionDef) and item.name == 'main')
  exec(compile(ast.Module(body=[main], type_ignores=[]), str(path), 'exec'), namespace)
  with pytest.raises(ValueError, match='original ' + failure_stage):
    namespace['main']()
  assert calls[0] == ('realtime', 7, 51)
  assert calls[-1] == 'close'
  if failure_stage == 'shutdown':
    assert calls[1:-1] == ['initialization', 'sm_update', 'sla_buttons', 'update', 'publish', 'ldw', 'driverAssistance', 'shutdown']
