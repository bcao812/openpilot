import json
import threading
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from opendbc.car.honda.values import CAR
from openpilot.cereal import messaging
from openpilot.common.logging_extra import SwagFormatter, SwagLogger
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.longitudinal_trace import INPUT_SERVICES, TRACE_FIELDS, TRACE_SCHEMA, LongitudinalTrace
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_planner, make_state
from openpilot.selfdrive.controls.tests.test_longitudinal_cruise_scenarios import simulate_lead_cruise
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import CruiseTarget, LongitudinalPlannerSP, LongitudinalPlanSource


def trace_planner():
  """Observe the retained generic scalar planner on a real Civic Bosch configuration."""
  planner, _ = make_planner(speed=20.2, platform=CAR.HONDA_CIVIC_BOSCH)
  planner.a_cruise = planner.output_a_target = -0.2
  assert not planner.unified
  return planner


def trace_state():
  state = make_state(speed=20.2, target=20.0)
  state.all_checks = lambda service_list=None: all(state.valid[service] for service in (state.valid if service_list is None else service_list))
  return state


def update_scalar_planner(planner, state, a_cruise=-0.23, a_mpc=1.0, e2e=False):
  def update_targets(instance, current_state, v_ego, a_ego, v_cruise):
    instance.source = LongitudinalPlanSource.cruise
    instance.cruise_target = CruiseTarget(instance.source, v_cruise, a_ego)
    return instance.cruise_target

  with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.get_cruise_accel', return_value=a_cruise), \
       patch('openpilot.selfdrive.controls.lib.longitudinal_planner.get_accel_from_plan', return_value=a_mpc), \
       patch.object(LongitudinalPlanner, 'is_e2e', return_value=e2e), \
       patch.object(LongitudinalPlannerSP, 'update_targets', new=update_targets):
    planner.update(state)
  return planner.output_a_target


def traced_update(recorder=None, **kwargs):
  planner = trace_planner()
  planner.trace = LongitudinalTrace() if recorder is None else recorder
  state = trace_state()
  state.logMonoTime = {service: 10_000 + index for index, service in enumerate(INPUT_SERVICES)}
  state['radarState'].mdMonoTime = 9_000
  update_scalar_planner(planner, state, **kwargs)
  return planner, state


def unpack(recorder):
  return dict(zip(TRACE_FIELDS, recorder.queue.popleft(), strict=True))


@pytest.mark.parametrize(
  'platform,longitudinal,marker',
  [
    (CAR.HONDA_ACCORD, True, False),
    (CAR.HONDA_ACCORD, False, True),
    (CAR.HONDA_CIVIC, True, True),
  ],
)
def test_startup_gate(platform, longitudinal, marker, tmp_path):
  path = tmp_path / 'trace'
  if marker:
    path.touch()
  params = SimpleNamespace(carFingerprint=platform, openpilotLongitudinalControl=longitudinal)
  with patch.object(LongitudinalTrace, 'start') as start:
    assert LongitudinalTrace.create(params, path) is None
    start.assert_not_called()


def test_marker_is_only_read_at_startup(tmp_path):
  path = tmp_path / 'trace'
  path.touch()
  params = SimpleNamespace(carFingerprint=CAR.HONDA_ACCORD, openpilotLongitudinalControl=True)
  with patch.object(LongitudinalTrace, 'start', return_value=True) as start:
    recorder = LongitudinalTrace.create(params, path)
    assert recorder is not None
    start.assert_called_once()
  path.unlink()
  with patch('os.path.isfile', side_effect=AssertionError('Filesystem access on control loop')):
    planner, _ = traced_update(recorder)
    recorder.publish(20_000)
  assert unpack(recorder)['a_target'] == planner.output_a_target


def test_marker_read_failure_is_safe():
  params = SimpleNamespace(carFingerprint=CAR.HONDA_ACCORD, openpilotLongitudinalControl=True)
  with patch('os.path.isfile', side_effect=OSError):
    assert LongitudinalTrace.create(params) is None


def test_exact_timestamps_candidates_and_copied_input():
  planner, state = traced_update(a_cruise=-0.23, a_mpc=1.0)
  state['carState'].vEgo = 99.0
  state.logMonoTime['modelV2'] = 77
  planner.trace.publish(20_000)
  sample = unpack(planner.trace)
  assert sample['modelV2_mono'] == 10_000
  assert sample['radar_model_mono'] == 9_000
  assert sample['plan_mono'] == 20_000
  assert sample['v_ego'] == pytest.approx(20.2)
  assert sample['previous_cruise'] == -0.2
  assert sample['cruise_raw'] == -0.23
  assert sample['cruise_accel'] == -0.23
  assert sample['mpc_accel'] == 1.0
  assert sample['a_target'] == planner.output_a_target
  assert sample['desired_v_next'] == planner.v_desired_filter.x
  assert sample['source'] == str(planner.mpc.source)
  assert not sample['e2e_allowed']
  assert len(TRACE_FIELDS) == len(set(TRACE_FIELDS))
  planner.trace.publish(30_000)
  assert not planner.trace.queue


def test_publish_happens_after_control_messages():
  planner, state = traced_update()
  observed = []

  def send(service, message):
    assert not planner.trace.queue
    observed.append((service, message.logMonoTime))

  def send_sp(state, publisher, parent_plan_mono_time=0, *, legacy_acceleration=None):
    assert parent_plan_mono_time == observed[0][1]
    assert legacy_acceleration == planner.output_a_target
    publisher.send('longitudinalPlanSP', messaging.new_message('longitudinalPlanSP'))

  with patch.object(planner, 'publish_longitudinal_plan_sp', side_effect=send_sp):
    planner.publish(state, SimpleNamespace(send=send))
  assert [service for service, _ in observed] == ['longitudinalPlan', 'longitudinalPlanSP']
  assert unpack(planner.trace)['plan_mono'] == observed[0][1]


def test_incomplete_sample_is_never_published():
  planner, state = traced_update()
  planner.trace.begin(planner, state, False, 20.0, 2.0)
  planner.trace.publish(20_000)
  assert not planner.trace.queue


def test_full_queue_drops_newest_and_exposes_sequence_gap():
  recorder = LongitudinalTrace(capacity=2, batch_size=2)
  for sequence in range(5):
    traced_update(recorder)
    recorder.publish(20_000 + sequence)
  assert recorder.dropped == 3
  assert [unpack(recorder)['sequence'] for _ in range(2)] == [1, 2]
  traced_update(recorder)
  recorder.publish(30_000)
  assert unpack(recorder)['sequence'] == 6
  assert not recorder.queue


def test_snapshot_failure_disables_trace_not_controller():
  planner, state = traced_update()
  del state.logMonoTime
  assert update_scalar_planner(planner, state) == -0.23
  assert planner.trace.disabled
  assert planner.trace.failures == 1
  planner.trace.publish(20_000)
  assert planner.trace.unpublished == 1
  previous = planner.trace.queue.popleft()
  previous_values = dict(zip(TRACE_FIELDS, previous, strict=True))
  assert previous_values['sequence'] == 1
  assert previous_values['plan_mono'] is None
  assert previous.detail['plan_mono'] is None
  assert not planner.trace.queue
  assert update_scalar_planner(planner, state, a_mpc=-2.0) == -2.0


def test_finish_and_queue_failures_are_contained():
  class FailingBuffer(deque):
    def append(self, sample):
      raise RuntimeError('Buffer unavailable')

  planner, _ = traced_update()
  planner.trace.finish(planner, [], False)
  assert planner.trace.disabled
  planner, _ = traced_update()
  planner.trace.queue = FailingBuffer()
  planner.trace.publish(20_000)
  assert planner.trace.disabled


def test_writer_emits_bounded_batches_and_flushes_on_close():
  events = []
  recorder = LongitudinalTrace(emit=lambda event, **payload: events.append((event, payload)), capacity=20)
  with (
    patch('openpilot.selfdrive.controls.lib.longitudinal_trace.drop_realtime') as drop,
    patch('openpilot.selfdrive.controls.lib.longitudinal_trace.set_core_affinity') as affinity,
  ):
    assert recorder.start()
    for sequence in range(12):
      traced_update(recorder)
      recorder.publish(20_000 + sequence)
    recorder.close()
  drop.assert_called_once()
  affinity.assert_called_once_with([0, 1, 2])
  assert not recorder.thread.is_alive()
  assert not recorder.disabled
  assert [len(payload['samples']) for _, payload in events] == [10, 2]
  assert all(payload['schema'] == TRACE_SCHEMA for _, payload in events)
  assert sum(len(payload['samples']) for _, payload in events) == 12
  json.dumps(events, allow_nan=False)


def test_slow_writer_cannot_grow_queue_or_block_publish():
  entered, release = threading.Event(), threading.Event()

  def slow_emit(*args, **kwargs):
    entered.set()
    release.wait(timeout=3.0)

  recorder = LongitudinalTrace(emit=slow_emit, capacity=2, batch_size=1)
  assert recorder.start()
  try:
    traced_update(recorder)
    recorder.publish(20_000)
    assert entered.wait(timeout=1.0)
    for sequence in range(5):
      traced_update(recorder)
      recorder.publish(30_000 + sequence)
    assert len(recorder.queue) == 2
    assert recorder.dropped == 3
  finally:
    release.set()
    recorder.close()
  assert not recorder.thread.is_alive()


@pytest.mark.parametrize('failure', ['drop_realtime', 'set_core_affinity'])
def test_failed_writer_demote_never_logs(failure):
  with (
    patch('openpilot.selfdrive.controls.lib.longitudinal_trace.' + failure, side_effect=OSError),
    patch('openpilot.selfdrive.controls.lib.longitudinal_trace.cloudlog.event') as emit,
  ):
    recorder = LongitudinalTrace()
    assert not recorder.start()
    recorder.close()
    assert recorder.disabled
    emit.assert_not_called()


def test_writer_error_does_not_reach_control_loop():
  def failing_emit(*args, **kwargs):
    raise RuntimeError('Logging unavailable')

  recorder = LongitudinalTrace(emit=failing_emit)
  assert recorder.start()
  planner, state = traced_update(recorder)
  recorder.publish(20_000)
  recorder.close()
  assert recorder.disabled
  assert recorder.failures == 1
  assert update_scalar_planner(planner, state, a_mpc=-2.0) == -2.0


def test_nonfinite_diagnostics_are_json_null():
  events = []
  recorder = LongitudinalTrace(emit=lambda event, **payload: events.append(payload))
  recorder._emit([(float('nan'), float('inf'), float('-inf'), 2.0)])
  assert events[0]['samples'] == [(None, None, None, 2.0)]
  json.dumps(events, allow_nan=False)


def test_cloudlog_debug_record_roundtrips_through_logmessage():
  import logging

  logger = SwagLogger()
  logger.setLevel(logging.DEBUG)
  encoded = []
  formatter = SwagFormatter(logger)
  handler = logging.Handler()
  handler.emit = lambda record: encoded.append(formatter.format(record))
  logger.addHandler(handler)
  recorder = LongitudinalTrace(emit=logger.event)
  assert recorder.start()
  traced_update(recorder)
  recorder.publish(20_000)
  recorder.close()
  assert len(encoded) == 1
  assert json.loads(encoded[0])['level'] == 'DEBUG'
  message = messaging.new_message(None, valid=True, logMessage=encoded[0])
  payload = json.loads(message.logMessage)['msg']
  assert dict(zip(payload['fields'], payload['samples'][0], strict=True))['plan_mono'] == 20_000


def test_default_writer_uses_private_handler_not_realtime_logging_locks():
  import logging

  events = []

  def private_handler(formatter):
    handler = logging.Handler()
    handler.emit = lambda record: events.append(formatter.format(record))
    return handler

  with (
    patch('openpilot.selfdrive.controls.lib.longitudinal_trace.UnixDomainSocketHandler', side_effect=private_handler) as handler,
    patch('openpilot.selfdrive.controls.lib.longitudinal_trace.cloudlog.event', side_effect=AssertionError('Shared logging handler')),
  ):
    recorder = LongitudinalTrace()
    assert recorder.start()
    traced_update(recorder)
    recorder.publish(20_000)
    recorder.close()
  handler.assert_called_once()
  assert not recorder.disabled
  payload = json.loads(events[0])['msg']
  assert dict(zip(payload['fields'], payload['samples'][0], strict=True))['plan_mono'] == 20_000


def test_private_socket_without_receiver_does_not_block(tmp_path):
  endpoint = 'ipc://' + str(tmp_path / 'unavailable.ipc')
  with patch('openpilot.common.swaglog.Paths.swaglog_ipc', return_value=endpoint):
    recorder = LongitudinalTrace()
    assert recorder.start()
    traced_update(recorder)
    recorder.publish(20_000)
    recorder.close()
  assert not recorder.disabled
  assert not recorder.thread.is_alive()


@pytest.mark.parametrize('condition', ['normal', 'lead_brake', 'model_brake', 'stop', 'override', 'invalid'])
def test_instrumentation_has_identical_planner_state(condition):
  baseline, instrumented = trace_planner(), trace_planner()
  recorder = LongitudinalTrace(capacity=2, batch_size=1)
  instrumented.trace = recorder
  state = trace_state()
  state.logMonoTime = dict.fromkeys(INPUT_SERVICES, 1)
  if condition == 'override':
    state['carControl'].cruiseControl.override = True
  elif condition == 'invalid':
    state.valid['radarState'] = False
  elif condition == 'stop':
    state['carState'].vEgo = 0.1
  elif condition == 'model_brake':
    state['modelV2'].action.desiredAcceleration = -2.0
  for frame in range(12):
    acceleration = -2.0 if condition == 'lead_brake' else 1.0
    for planner in (baseline, instrumented):
      update_scalar_planner(planner, state, a_mpc=acceleration, e2e=condition == 'model_brake')
    recorder.publish(frame + 1)
    assert baseline.output_a_target == instrumented.output_a_target
    assert baseline.output_should_stop == instrumented.output_should_stop
    assert baseline.v_desired_filter.x == instrumented.v_desired_filter.x
    assert baseline.mpc.source == instrumented.mpc.source
    assert baseline.stop_latched == instrumented.stop_latched
    assert baseline.creep_t == instrumented.creep_t
    assert np.array_equal(baseline.mpc.x0, instrumented.mpc.x0)
    assert np.array_equal(baseline.mpc.a_prev, instrumented.mpc.a_prev)
    assert np.array_equal(baseline.a_desired_trajectory, instrumented.a_desired_trajectory)
  assert recorder.dropped == 10


@pytest.mark.parametrize('scenario', ['steady', 'lead_brake', 'cut_in', 'override', 'flicker', 'no_lead'])
def test_closed_loop_with_real_mpc_control_and_honda_can_is_identical(scenario):
  options = {'scenario': scenario, 'duration': 42.0, 'delay': 0.5, 'noise': 0.02}
  baseline = simulate_lead_cruise(**options)
  recorder = LongitudinalTrace(capacity=2, batch_size=1)
  original = LongitudinalPlanner.update

  def observed_update(planner, state, *args, **kwargs):
    planner.trace = recorder
    original(planner, state, *args, **kwargs)
    recorder.publish(state.logMonoTime['modelV2'])

  with patch.object(LongitudinalPlanner, 'update', new=observed_update):
    actual = simulate_lead_cruise(**options)
  assert actual == baseline
  assert not recorder.disabled
  assert recorder.dropped > 0
