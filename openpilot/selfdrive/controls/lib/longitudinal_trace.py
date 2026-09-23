"""Opt-in, output-neutral planner evidence with bounded asynchronous logging."""

import math
import os
import threading
import logging
import json
import secrets
import sys
import time
from collections import deque
from collections.abc import Mapping
from numbers import Integral, Real
from types import MappingProxyType

import zmq

from opendbc.car.honda.values import CAR
from openpilot.common.hardware.hw import Paths
from openpilot.common.longitudinal_logging import PLANNER_INPUTS, capture_inputs, is_uint64
from openpilot.common.realtime import drop_realtime, set_core_affinity
from openpilot.common.logging_extra import SwagFormatter, SwagLogger
from openpilot.common.swaglog import cloudlog


TRACE_MARKER = '/data/longitudinal_trace'
TRACE_EVENT = 'longitudinal.trace'
TRACE_SCHEMA = 1
MOTION_TRACE_SCHEMA = 3
DETAIL_SCHEMA = 1
INPUT_SERVICES = ('modelV2', 'radarState', 'carState', 'carControl', 'controlsState', 'selfdriveState', 'vehicleParameters')
LEAD_FIELDS = ('present', 'radar', 'modelProb', 'dRel', 'vRel', 'vLead', 'aLeadK', 'aLeadTau', 'radarTrackId')
TRACE_FIELDS = (
  'sequence',
  'plan_mono',
  *(service + '_mono' for service in INPUT_SERVICES),
  'radar_model_mono',
  'valid',
  'enabled',
  'long_active',
  'gas_pressed',
  'brake_pressed',
  'override',
  'force_decel',
  'reset_state',
  'long_control_state',
  'personality',
  'v_ego',
  'a_ego',
  'v_cruise',
  'mpc_initial_v',
  'mpc_initial_a',
  'previous_cruise',
  'coast_limit',
  *(name + '_' + field for name in ('leadOne', 'leadTwo') for field in LEAD_FIELDS),
  'model_accel',
  'model_stop',
  'mpc_accel',
  'mpc_source',
  'mpc_stop',
  'cruise_raw',
  'cruise_accel',
  'cruise_stop',
  'e2e_allowed',
  'source',
  'a_target',
  'should_stop',
  'allow_throttle',
  'fcw',
  'solver_status',
  'desired_v_next',
  'obstacle_initial',
)
UNIFIED_TRACE_FIELDS = TRACE_FIELDS + (
  'intent_version', 'intent_reasons', 'intent_valid', 'moving_lead_evidence', 'release_evidence',
  'allow_creep', 'reference_accel', 'constraint_accel', 'reference_overridden', 'constraint_source',
  'intent_ready', 'intent_mode', 'braking_valid', 'braking_accel',
)


def optional_value(callback):
  try:
    return callback()
  except Exception:
    return None


class _TraceMapping(Mapping):
  """Own a deep-frozen copy so later trace stages can reuse the snapshot."""

  __slots__ = ('_data',)

  def __init__(self, value):
    if any(not isinstance(key, str) for key in value):
      raise TypeError('Trace mapping keys must be strings')
    object.__setattr__(self, '_data', MappingProxyType({key: immutable_value(item) for key, item in value.items()}))

  def __getitem__(self, key):
    return self._data[key]

  def __iter__(self):
    return iter(self._data)

  def __len__(self):
    return len(self._data)

  def __setattr__(self, name, value):
    raise TypeError('Trace snapshot cannot be reassigned')

  def __delattr__(self, name):
    raise TypeError('Trace snapshot cannot be changed')


class _TraceTuple(tuple):
  __slots__ = ()

  def __new__(cls, values):
    return tuple.__new__(cls, (immutable_value(item) for item in values))


def immutable_value(value):
  value_type = type(value)
  if value_type is _TraceMapping or value_type is _TraceTuple:
    return value
  if value_type is int or value_type is float:
    return value
  if value is None or isinstance(value, (str, bool)):
    return value
  if isinstance(value, Integral):
    return int(value)
  if isinstance(value, Real):
    return float(value)
  if isinstance(value, Mapping):
    return _TraceMapping(value)
  if isinstance(value, (tuple, list)):
    return _TraceTuple(value)
  if hasattr(value, 'tolist'):
    return immutable_value(value.tolist())
  raise TypeError('Unsupported trace value')


def json_value(value):
  # Snapshot leaves are usually builtins; subclasses retain the Mapping precedence below.
  value_type = type(value)
  if value_type is float:
    return value if math.isfinite(value) else None
  if value is None or value_type is str or value_type is bool or value_type is int:
    return value
  # Only our exact frozen containers can bypass the generic Mapping protocol;
  # subclasses may override items/iteration and keep the existing precedence.
  if value_type is _TraceMapping:
    return {key: json_value(item) for key, item in value._data.items()}
  if value_type is _TraceTuple:
    return tuple(json_value(item) for item in value)
  if isinstance(value, Mapping):
    return {key: json_value(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return tuple(json_value(item) for item in value)
  if isinstance(value, float) and not math.isfinite(value):
    return None
  if value is None or isinstance(value, (str, bool, int, float)):
    return value
  raise TypeError('Trace payload is not an immutable JSON snapshot')


def decimal_id(value):
  return str(int(value)) if is_uint64(value) else None


class TraceSample(tuple):
  def __new__(cls, values, detail, schema, fields):
    sample = super().__new__(cls, values)
    sample.detail = immutable_value(detail)
    sample.schema = schema
    sample.fields = fields
    return sample


class TraceSerializationError(Exception):
  pass


class UnixDomainSocketHandler(logging.Handler):
  """Private bounded sender; unlike the shared handler, it exposes send failures."""

  def __init__(self, formatter):
    super().__init__()
    self.setFormatter(formatter)
    self.zctx = None
    self.sock = None
    try:
      self.zctx = zmq.Context()
      self.sock = self.zctx.socket(zmq.PUSH)
      self.sock.setsockopt(zmq.LINGER, 0)
      self.sock.setsockopt(zmq.SNDHWM, 8)
      self.sock.setsockopt(zmq.SNDTIMEO, 0)
      self.sock.setsockopt(zmq.IMMEDIATE, 1)
      self.sock.connect(Paths.swaglog_ipc())
    except Exception:
      self.close()
      raise

  def emit(self, record):
    try:
      payload = self.formatter.format_dict(record)
      encoded = (chr(record.levelno) + json.dumps(payload, separators=(',', ':'), allow_nan=False)).encode('utf8')
    except Exception as error:
      raise TraceSerializationError('Could not serialize trace log record') from error
    self.sock.send(encoded, zmq.NOBLOCK)

  def close(self):
    if self.sock is not None:
      self.sock.close(linger=0)
      self.sock = None
    if self.zctx is not None:
      self.zctx.term()
      self.zctx = None
    super().close()


class LongitudinalTrace:
  """Single-producer/single-consumer CPython deque; logging threads never hold a producer mutex."""

  @classmethod
  def create(cls, params, marker=TRACE_MARKER):
    if params.carFingerprint != CAR.HONDA_ACCORD or not params.openpilotLongitudinalControl:
      return None
    try:
      if not os.path.isfile(marker):
        return None
      recorder = cls()
      recorder.start()
      return recorder
    except OSError:
      return None

  def __init__(self, *, emit=None, capacity=40, batch_size=10, flush_interval=0.5, clock=None):
    if capacity <= 0 or not 0 < batch_size <= capacity:
      raise ValueError('Trace queue and batch sizes must be positive and bounded')
    if not math.isfinite(flush_interval) or flush_interval <= 0:
      raise ValueError('Trace flush interval must be positive and finite')
    self.emit = emit
    self.queue = deque()
    self.capacity = capacity
    self.batch_size = batch_size
    self.flush_interval = float(flush_interval)
    self.clock = time.monotonic if clock is None else clock
    self.session = secrets.randbits(64) or 1
    self.sequence = 0
    self.dropped = 0
    self.capture_failures = 0
    self.serialization_failures = 0
    self.transport_failures = 0
    self.startup_failures = 0
    self.writer_failures = 0
    self.serialization_dropped = 0
    self.transport_dropped = 0
    self.disabled_dropped = 0
    self.disabled_attempts = 0
    self.unpublished = 0
    self.close_timeouts = 0
    self.transport_accepted_samples = 0
    self.transport_accepted_batches = 0
    self.disabled = False
    self.closed = False
    self.pending = None
    self.complete = False
    self.attempt = None
    self.pending_detail = None
    self.stop = threading.Event()
    self.ready = threading.Event()
    self.thread = None
    self.schema = TRACE_SCHEMA
    self.fields = TRACE_FIELDS

  @property
  def failures(self):
    return self.capture_failures + self.serialization_failures + self.transport_failures + self.startup_failures + self.writer_failures

  @property
  def enabled(self):
    return not self.disabled and not self.closed

  def status(self):
    return {
      'session': decimal_id(self.session), 'sequence': decimal_id(self.sequence),
      'disabled': self.disabled, 'closed': self.closed, 'queue_depth': len(self.queue),
      'queue_dropped': self.dropped, 'capture_failures': self.capture_failures,
      'serialization_failures': self.serialization_failures, 'transport_failures': self.transport_failures,
      'startup_failures': self.startup_failures, 'writer_failures': self.writer_failures,
      'serialization_dropped': self.serialization_dropped, 'transport_dropped': self.transport_dropped,
      'disabled_dropped': self.disabled_dropped, 'disabled_attempts': self.disabled_attempts,
      'unpublished': self.unpublished, 'close_timeouts': self.close_timeouts,
      'transport_accepted_samples': self.transport_accepted_samples, 'transport_accepted_batches': self.transport_accepted_batches,
    }

  def fail(self):
    self.capture_failures += 1
    self.disabled = True
    self.pending = None
    self.complete = False

  def begin_attempt(self, planner, state):
    if self.attempt is not None and self.complete:
      self.publish(None)
    self.pending = None
    self.pending_detail = None
    self.complete = False
    self.attempt = None
    self.sequence += 1
    if not self.enabled:
      self.disabled_attempts += 1
      return
    try:
      motion_enabled = getattr(planner, 'motion_enabled', planner.unified)
      self.schema = MOTION_TRACE_SCHEMA if motion_enabled else TRACE_SCHEMA
      self.fields = UNIFIED_TRACE_FIELDS if motion_enabled else TRACE_FIELDS
      inputs = optional_value(lambda: capture_inputs(state, PLANNER_INPUTS))
      input_rows = None if inputs is None else tuple({
        **item, 'monoTime': decimal_id(item['monoTime']) if item['idKnown'] else None,
      } for item in inputs)
      self.attempt = {
        'sequence': self.sequence,
        'inputs': immutable_value(input_rows),
        'base': self._input_values(state),
        'solve_calls_before': getattr(planner.mpc, 'diagnostic_solve_calls', None),
        'reset_calls_before': getattr(planner.mpc, 'diagnostic_reset_calls', None),
        'native_before': {name: getattr(planner.mpc, 'diagnostic_last_' + name, None)
                          for name in ('x0', 'parameters', 'yref', 'a_prev')},
        'start_mono': time.monotonic_ns(),
        'motion_version': optional_value(lambda: int(state['controlsState'].motionState.version)),
        'motion_enabled': optional_value(lambda: bool(state['controlsState'].motionState.enabled)),
      }
    except Exception:
      self.fail()

  def _input_values(self, state):
    values = {service + '_mono': optional_value(lambda service=service: int(state.logMonoTime[service])) for service in INPUT_SERVICES}
    values.update({
      'radar_model_mono': optional_value(lambda: int(state['radarState'].mdMonoTime)),
      'valid': optional_value(lambda: bool(state.all_checks())),
      'enabled': optional_value(lambda: bool(state['selfdriveState'].enabled)),
      'long_active': optional_value(lambda: bool(state['carControl'].longActive)),
      'gas_pressed': optional_value(lambda: bool(state['carState'].gasPressed)),
      'brake_pressed': optional_value(lambda: bool(state['carState'].brakePressed)),
      'override': optional_value(lambda: bool(state['carControl'].cruiseControl.override)),
      'force_decel': optional_value(lambda: bool(state['controlsState'].forceDecel)),
      'long_control_state': optional_value(lambda: str(state['controlsState'].longControlState)),
      'personality': optional_value(lambda: str(state['selfdriveState'].personality)),
      'v_ego': optional_value(lambda: float(state['carState'].vEgo)),
      'a_ego': optional_value(lambda: float(state['carState'].aEgo)),
      'model_accel': optional_value(lambda: float(state['modelV2'].action.desiredAcceleration)),
      'model_stop': optional_value(lambda: bool(state['modelV2'].action.shouldStop)),
    })
    for name in ('leadOne', 'leadTwo'):
      for field in LEAD_FIELDS:
        values[name + '_' + field] = optional_value(lambda name=name, field=field: getattr(getattr(state['radarState'], name), field))
    return immutable_value(values)

  def _native_detail(self, planner, observation):
    mpc = planner.mpc
    solve_calls = getattr(mpc, 'diagnostic_solve_calls', None)
    solve_before = self.attempt['solve_calls_before']
    solve_known = isinstance(solve_calls, Integral) and isinstance(solve_before, Integral) and solve_calls >= solve_before
    attempted = bool(solve_calls > solve_before) if solve_known else None
    resets = getattr(mpc, 'diagnostic_reset_calls', None)
    reset_before = self.attempt['reset_calls_before']
    reset_known = isinstance(resets, Integral) and isinstance(reset_before, Integral) and resets >= reset_before
    solver_return = getattr(mpc, 'diagnostic_last_solver_return', None) if attempted else None
    before = self.attempt['native_before']
    inputs = {}
    for field, source in (('initial_state', 'x0'), ('params', 'parameters'), ('yref', 'yref'), ('a_prev', 'a_prev')):
      value = getattr(mpc, 'diagnostic_last_' + source, None)
      inputs[field] = value if value is not None and value is not before[source] else None
    updated = bool(observation.get('mpcUpdated', False) or attempted or inputs['initial_state'] is not None)
    solution_status = int(mpc.solution_status) if updated else None
    solved = bool(attempted and solver_return == 0 and solution_status == 0 and reset_known and resets == reset_before)
    module = sys.modules.get(type(mpc).__module__)
    times = getattr(module, 'T_IDXS', None)
    if times is None:
      times = getattr(mpc, 'diagnostic_times', None)
    arrays = {
      'times': times,
      'x': optional_value(lambda: mpc.x_sol[:, 0]) if solved else None,
      'v': getattr(mpc, 'v_solution', None) if solved else None,
      'a': getattr(mpc, 'a_solution', None) if solved else None,
      'j': getattr(mpc, 'j_solution', None) if solved else None,
      **inputs,
    }
    return immutable_value({
      **arrays, 'known': {name: value is not None for name, value in arrays.items()},
      'solve_known': solve_known, 'solve_attempted': attempted, 'trajectory_known': solved,
      'solver_return': int(solver_return) if solver_return is not None else None,
      'solver_return_known': solver_return is not None, 'solution_status': solution_status,
      'solution_status_known': updated,
      'solve_calls': decimal_id(solve_calls), 'reset_calls': decimal_id(resets),
      'reset_known': reset_known, 'resets_this_attempt': int(resets - reset_before) if reset_known else None,
    })

  def finish_attempt(self, planner):
    if not self.enabled or self.attempt is None:
      return
    try:
      result = getattr(planner, 'observation_result', None)
      values = result['values'] if result is not None else {}
      sample = result['sample'] if result is not None else {}
      observer = getattr(planner, 'observation', None)
      observer_recorder = getattr(observer, 'recorder', None)
      outcome = values.get('outcome')
      if outcome is None:
        outcome = 'control_exception' if sys.exc_info()[0] is not None else 'completed_without_observer' if self.complete else 'early_return_unknown'
      self.pending_detail = immutable_value({
        'session': decimal_id(self.session), 'sequence': decimal_id(self.attempt['sequence']),
        'plan_mono': None, 'outcome': outcome, 'update_completed': values.get('updateCompleted'),
        'mode': 'unified' if planner.unified else 'legacy',
        'start_mono': decimal_id(self.attempt['start_mono']), 'end_mono': decimal_id(time.monotonic_ns()),
        'planner_session': decimal_id(getattr(observer_recorder, 'session', None)),
        'planner_sequence': decimal_id(sample.get('sequence')),
        'planner_identity': getattr(observer_recorder, 'identity', None),
        'inputs': self.attempt['inputs'],
        'motion_version': self.attempt['motion_version'], 'motion_enabled': self.attempt['motion_enabled'],
        'native': self._native_detail(planner, values),
      })
      if not self.complete:
        row = dict(self.attempt['base'])
        if self.pending is not None:
          row.update(zip(TRACE_FIELDS[2:], self.pending, strict=False))
        row.update({
          'a_target': optional_value(lambda: float(planner.output_a_target)),
          'should_stop': optional_value(lambda: bool(planner.output_should_stop)),
          'solver_status': self.pending_detail['native']['solution_status'],
        })
        if self.schema == MOTION_TRACE_SCHEMA:
          intent = getattr(planner, 'motion_intent', None)
          for field, attribute in (
            ('intent_version', 'version'), ('intent_reasons', 'reasons'), ('intent_valid', 'valid'),
            ('moving_lead_evidence', 'moving_lead'), ('release_evidence', 'release_allowed'),
            ('allow_creep', 'allow_creep'), ('intent_ready', 'ready'), ('intent_mode', 'mode'), ('braking_accel', 'braking_accel'),
          ):
            row[field] = optional_value(lambda attribute=attribute: getattr(intent, attribute))
          row['braking_valid'] = optional_value(lambda: intent.braking_accel is not None)
        self.pending = tuple(immutable_value(row.get(field)) for field in self.fields[2:])
        self.complete = True
    except Exception:
      self.fail()

  def begin(self, planner, state, reset_state, cruise_target, coast_limit):
    self.pending = None
    self.complete = False
    if not self.enabled:
      return
    try:
      self.pending = (
        *(int(state.logMonoTime[service]) for service in INPUT_SERVICES),
        int(state['radarState'].mdMonoTime),
        bool(state.all_checks()),
        bool(state['selfdriveState'].enabled),
        bool(state['carControl'].longActive),
        bool(state['carState'].gasPressed),
        bool(state['carState'].brakePressed),
        bool(state['carControl'].cruiseControl.override),
        bool(state['controlsState'].forceDecel),
        bool(reset_state),
        str(state['controlsState'].longControlState),
        str(state['selfdriveState'].personality),
        float(state['carState'].vEgo),
        float(state['carState'].aEgo),
        float(cruise_target),
        float(planner.mpc.x0[1]),
        float(planner.mpc.x0[2]),
        float(planner.a_cruise),
        float(coast_limit),
        *(getattr(getattr(state['radarState'], name), field) for name in ('leadOne', 'leadTwo') for field in LEAD_FIELDS),
        float(state['modelV2'].action.desiredAcceleration),
        bool(state['modelV2'].action.shouldStop),
      )
    except Exception:
      self.fail()

  def finish(self, planner, candidates, is_e2e):
    if not self.enabled or self.pending is None:
      return
    try:
      mpc_accel, mpc_source, mpc_stop = candidates[0]
      cruise_accel, _, cruise_stop = candidates[1]
      self.pending += (
        float(mpc_accel),
        str(mpc_source),
        bool(mpc_stop),
        float(planner.a_cruise),
        float(cruise_accel),
        bool(cruise_stop),
        bool(is_e2e),
        str(planner.mpc.source),
        float(planner.output_a_target),
        bool(planner.output_should_stop),
        bool(planner.allow_throttle),
        bool(planner.fcw),
        int(planner.mpc.solution_status),
        float(planner.v_desired_filter.x),
        float(planner.mpc.params[0, 2]),
      )
      self.complete = True
    except Exception:
      self.fail()

  def publish(self, plan_mono):
    pending, complete = self.pending, self.complete
    attempt, detail = self.attempt, self.pending_detail
    self.pending = None
    self.complete = False
    self.attempt = None
    self.pending_detail = None
    if not self.enabled or pending is None or not complete:
      return
    try:
      if attempt is None:
        self.sequence += 1
      sequence = self.sequence if attempt is None else attempt['sequence']
      mono = int(plan_mono) if is_uint64(plan_mono) else None
      if mono is None:
        self.unpublished += 1
      if len(self.queue) >= self.capacity:
        self.dropped += 1
      else:
        if detail is None:
          detail = {
            'session': decimal_id(self.session), 'sequence': decimal_id(sequence),
            'outcome': 'legacy_hooks_only', 'update_completed': None, 'native': None,
          }
        detail = dict(detail, plan_mono=decimal_id(mono))
        self.queue.append(TraceSample(immutable_value((sequence, mono, *pending)), detail, self.schema, self.fields))
    except Exception:
      self.fail()

  def finish_unified(self, planner, is_e2e):
    if not self.enabled or self.pending is None:
      return
    try:
      self.schema = MOTION_TRACE_SCHEMA
      self.fields = UNIFIED_TRACE_FIELDS
      reference = planner.reference_planner
      intent = planner.motion_intent
      candidate_values = (float(reference.reference_accel), str(reference.reference_source), None, None, None, None)
      self.pending += (
        *candidate_values, bool(is_e2e), str(planner.mpc.source),
        float(planner.output_a_target), bool(planner.output_should_stop),
        bool(planner.allow_throttle), bool(planner.fcw), int(planner.mpc.solution_status),
        float(planner.v_desired_filter.x), float(planner.mpc.params[0, 2]),
        int(intent.version), int(intent.reasons), bool(intent.valid), bool(intent.moving_lead),
        bool(intent.release_allowed), bool(intent.allow_creep), float(reference.reference_accel),
        float(reference.constraint_accel), bool(reference.reference_overridden), str(reference.constraint_source),
        bool(intent.ready), int(intent.mode), intent.braking_accel is not None, intent.braking_accel,
      )
      self.complete = True
    except Exception:
      self.fail()

  def start(self):
    if self.thread is not None or not self.enabled:
      return False
    try:
      self.thread = threading.Thread(target=self._writer, name='longitudinal_trace', daemon=True)
      self.thread.start()
      if not self.ready.wait(timeout=1.0) or self.disabled:
        if not self.disabled:
          self.startup_failures += 1
        self.disabled = True
        self.stop.set()
        return False
      return True
    except RuntimeError:
      self.startup_failures += 1
      self.disabled = True
      return False

  def close(self, timeout=1.0):
    if self.attempt is not None and self.complete:
      self.publish(None)
    self.closed = True
    self.stop.set()
    if self.thread is not None:
      self.thread.join(timeout=max(0.0, min(float(timeout), 1.0)))
      if self.thread.is_alive():
        self.close_timeouts += 1
        return False
    else:
      self._discard_queue()
    return True

  def _discard_queue(self):
    while True:
      try:
        self.queue.popleft()
        self.disabled_dropped += 1
      except IndexError:
        return

  def _writer(self):
    handler = None
    try:
      drop_realtime()
      set_core_affinity([0, 1, 2])
      if self.emit is None:
        logger = SwagLogger()
        logger.setLevel(logging.DEBUG)
        logger.bind_global(**cloudlog.get_ctx())
        handler = UnixDomainSocketHandler(SwagFormatter(logger))
        logger.addHandler(handler)
        self.emit = logger.event
    except Exception:
      self.startup_failures += 1
      self.disabled = True
      self.ready.set()
      return
    self.ready.set()
    try:
      self._drain()
    except Exception:
      self.writer_failures += 1
      self.disabled = True
      self._discard_queue()
    finally:
      if handler is not None:
        try:
          handler.close()
        except Exception:
          self.writer_failures += 1
          self.disabled = True

  def _drain(self):
    batch = []
    deadline = None
    while not self.stop.is_set() or self.queue:
      try:
        sample = self.queue.popleft()
        if batch and getattr(batch[0], 'schema', self.schema) != getattr(sample, 'schema', self.schema):
          self._emit(batch)
          batch = []
          deadline = None
        if not batch:
          deadline = self.clock() + self.flush_interval
        batch.append(sample)
      except IndexError:
        delay = 0.05 if deadline is None else min(0.05, max(0.0, deadline - self.clock()))
        self.stop.wait(timeout=delay)
      if batch and (len(batch) >= self.batch_size or self.clock() >= deadline or not self.queue and self.stop.is_set()):
        self._emit(batch)
        batch = []
        deadline = None
      if self.disabled:
        self.disabled_dropped += len(batch)
        self._discard_queue()
        return
    if batch:
      self._emit(batch)

  def _emit(self, batch):
    if not batch:
      return
    try:
      samples = [json_value(sample) for sample in batch]
      details = [json_value(getattr(sample, 'detail', None)) for sample in batch]
      payload = {
        'schema': getattr(batch[0], 'schema', self.schema), 'fields': getattr(batch[0], 'fields', self.fields),
        'samples': samples, 'detail_schema': DETAIL_SCHEMA, 'details': details,
        'dropped': self.dropped, 'failures': self.failures, 'counters': self.status(),
        'counter_scope': 'cumulative before this batch send attempt',
        'delivery_scope': 'local queue and transport send acceptance only; no logger or disk acknowledgement',
        'debug': True,
      }
    except Exception:
      self.serialization_failures += 1
      self.serialization_dropped += len(batch)
      self.disabled = True
      return
    try:
      self.emit(TRACE_EVENT, **payload)
      self.transport_accepted_samples += len(batch)
      self.transport_accepted_batches += 1
    except TraceSerializationError:
      self.serialization_failures += 1
      self.serialization_dropped += len(batch)
      self.disabled = True
    except zmq.Again:
      self.transport_failures += 1
      self.transport_dropped += len(batch)
    except Exception:
      self.transport_failures += 1
      self.transport_dropped += len(batch)
      self.disabled = True
