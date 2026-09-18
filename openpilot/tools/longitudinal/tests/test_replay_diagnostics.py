from copy import deepcopy
import json

import numpy as np
import pytest

from openpilot.tools.longitudinal import replay_pedal_switches as replay
from openpilot.tools.longitudinal.tests.test_unified_replay import cadence_fixture, replay_fixture


def shifted_messages(messages, offset):
  shifted = deepcopy(messages)
  for message in shifted:
    message.logMonoTime += offset
    if message.which() == 'longitudinalPlan':
      message.longitudinalPlan.modelMonoTime += offset
    elif message.which() == 'controlsState' and message.controlsState.longitudinalPlanMonoTime:
      message.controlsState.longitudinalPlanMonoTime += offset
  return shifted


def counter_fixture(monkeypatch, *, solve_delta=1, reset_delta=0, native_return=0, mutate=None, update_error=None,
                    return_marker=True):
  fixture = replay_fixture(monkeypatch, solver_status=9)
  previous = replay.planner_module.LongitudinalPlanner

  class ObservedPlanner(previous):
    def __init__(self, params, params_sp, *, unified=False, supervised=False):
      super().__init__(params, params_sp, unified=unified, supervised=supervised)
      self.mpc.diagnostic_solve_calls = 2 ** 53 + 17
      self.mpc.diagnostic_reset_calls = 4
      if return_marker:
        self.mpc.diagnostic_last_solver_return = 123

    def update(self, state):
      super().update(state)
      self.mpc.diagnostic_solve_calls += solve_delta
      self.mpc.diagnostic_reset_calls += reset_delta
      if solve_delta and return_marker:
        self.mpc.diagnostic_last_solver_return = native_return
      if mutate is not None:
        mutate(self)
      if update_error is not None:
        raise update_error

  monkeypatch.setattr(replay.planner_module, 'LongitudinalPlanner', ObservedPlanner)
  return fixture


@pytest.mark.parametrize('parent_metadata', [False, True], ids=['legacy_bound_call', 'new_keyword_bound_call'])
def test_actual_planner_publish_to_sp_noop_binding(monkeypatch, parent_metadata):
  actual_planner = replay.planner_module.LongitudinalPlanner
  messages, window, _, encoded, _ = replay_fixture(monkeypatch)
  prepared_planner = replay.planner_module.LongitudinalPlanner
  received_parent_ids = []

  class ActualPublishingPlanner(actual_planner):
    def __init__(self, params, params_sp, *, unified=False, supervised=False):
      prepared_planner.__init__(self, params, params_sp, unified=unified, supervised=supervised)
      self.mpc.solve_time = 0.0
      self.v_desired_trajectory = np.asarray(self.v_desired_trajectory)
      self.a_desired_trajectory = np.zeros(1)
      self.j_desired_trajectory = np.zeros(1)
      self.observation = self.observation_result = self.trace = None

    def update(self, state):
      return prepared_planner.update(self, state)

    def publish_longitudinal_plan_sp(self, state, publisher, parent_plan_mono_time=0):
      received_parent_ids.append(parent_plan_mono_time)
      if parent_metadata:
        return super().publish_longitudinal_plan_sp(state, publisher, parent_plan_mono_time=parent_plan_mono_time)
      return super().publish_longitudinal_plan_sp(state, publisher)

  assert ActualPublishingPlanner.publish is actual_planner.publish
  monkeypatch.setattr(replay.planner_module, 'LongitudinalPlanner', ActualPublishingPlanner)
  result = replay.replay_window(None, window, messages=messages, unified=False)
  assert received_parent_ids == [1_010_000_000]
  assert result['coverage'] == {'plans': 1, 'matched_plans': 1, 'controls': 1, 'matched_controls': 1,
                                'sendcan': 1, 'matched_sendcan': 1}
  assert result['plan_rows'][0]['accel'] == float(np.float32(0.15))
  assert len(encoded) == 1
  assert result['diagnostics']['status'] == 'COMPLETED'


@pytest.mark.parametrize('calling_style', ['legacy', 'positional_metadata', 'keyword_metadata'])
def test_frozen_sp_publisher_does_not_read_or_publish_recorded_decisions(calling_style):
  class Unreadable:
    def __getattr__(self, name):
      raise AssertionError('Frozen SP publisher must not inspect inputs or publish output')

  instance, state, publisher = Unreadable(), Unreadable(), Unreadable()
  if calling_style == 'legacy':
    result = replay.frozen_sp_publish(instance, state, publisher)
  elif calling_style == 'positional_metadata':
    result = replay.frozen_sp_publish(instance, state, publisher, 2 ** 53 + 17)
  else:
    result = replay.frozen_sp_publish(instance, state, publisher, parent_plan_mono_time=2 ** 53 + 17)
  assert result is None


def test_frozen_sp_publisher_does_not_silently_accept_unreviewed_arguments():
  with pytest.raises(TypeError, match='unexpected keyword'):
    replay.frozen_sp_publish(None, None, None, another_future_protocol=1)


@pytest.mark.parametrize('unified', [False, True])
def test_successful_progress_is_separate_from_score_coverage(monkeypatch, unified):
  messages, window, _, _, _ = replay_fixture(monkeypatch)
  cycles = [message for message in messages if message.logMonoTime >= 1_000_000_000]
  messages += shifted_messages(cycles, 100_000_000) + shifted_messages(cycles, 200_000_000)
  window.update(start=1.1, end=1.15)
  diagnostics = {}
  result = replay.replay_window(None, window, messages=messages, unified=unified, diagnostics=diagnostics)
  assert diagnostics['diagnostic_schema'] == replay.DIAGNOSTIC_SCHEMA_VERSION
  assert diagnostics['status'] == 'COMPLETED'
  assert result['diagnostics'] is diagnostics
  assert result['capabilities']['replay_protocol_version'] == replay.REPLAY_PROTOCOL_VERSION
  assert diagnostics['coverage'] == {
    'plans': 1, 'matched_plans': 1, 'controls': 1, 'matched_controls': 1, 'sendcan': 1, 'matched_sendcan': 1}
  for layer in ('planner', 'longcontrol', 'carcontroller'):
    for kind in ('attempted', 'completed'):
      assert diagnostics['counts'][layer][kind] == {'total': 3, 'read_only_prefix': 0, 'dependency_prefix': 0,
                                                   'warmup': 1, 'score': 1, 'after_score': 1}
  assert diagnostics['first_attempted'] == diagnostics['first_completed'] == result['first_executed']
  assert 'completed_update_return' in diagnostics['first_executed_semantics']
  assert diagnostics['coverage_scope'] == 'score_only_not_runtime_call_counts'


def test_before_initialization_error_keeps_unknowns_and_original_rejection():
  diagnostics = {}
  with pytest.raises(ValueError, match='own planner intent'):
    replay.replay_window(None, {}, recorded_plan=True, unified=True, diagnostics=diagnostics)
  assert diagnostics['status'] == 'ERROR'
  assert not diagnostics['initialized']
  assert diagnostics['counts'] is None
  assert diagnostics['first_attempted'] is None
  assert diagnostics['first_completed'] is None
  assert diagnostics['coverage'] is None
  assert diagnostics['current'] is None


def test_log_reader_initialization_failure_does_not_claim_zero_runtime_calls(monkeypatch):
  def missing_reader(*arguments, **options):
    raise OSError('synthetic unavailable recording')

  monkeypatch.setattr(replay, 'LogReader', missing_reader)
  diagnostics = {}
  with pytest.raises(OSError, match='unavailable recording'):
    replay.replay_window('absent-cache', {}, diagnostics=diagnostics)
  assert diagnostics['status'] == 'ERROR'
  assert diagnostics['counts'] is None
  assert diagnostics['first_completed'] is None
  assert diagnostics['coverage'] is None


@pytest.mark.parametrize('layer', ['planner', 'longcontrol', 'carcontroller'])
def test_exception_during_update_records_attempt_but_not_completion(monkeypatch, layer):
  messages, window, _, _, _ = replay_fixture(monkeypatch)
  component = {'planner': replay.planner_module.LongitudinalPlanner,
               'longcontrol': replay.LongControl, 'carcontroller': replay.CarController}[layer]
  original_update = component.update

  def failing_update(*arguments, **options):
    raise RuntimeError('synthetic update interruption')

  failing_update.__signature__ = replay.inspect.signature(original_update)
  monkeypatch.setattr(component, 'update', failing_update)
  diagnostics = {}
  rows = []
  with pytest.raises(RuntimeError, match='synthetic update interruption'):
    replay.replay_window(None, window, messages=messages, row_sink=lambda kind, row: rows.append((kind, row)),
                         unified=True, diagnostics=diagnostics)
  assert diagnostics['status'] == 'ERROR'
  assert diagnostics['counts'][layer]['attempted']['total'] == 1
  assert diagnostics['counts'][layer]['completed']['total'] == 0
  key = replay.ReplayDiagnostics.LAYERS[layer]
  assert diagnostics['first_attempted'][key] is not None
  assert diagnostics['first_completed'][key] is None
  assert diagnostics['first_executed'][key] is None
  assert len(rows) == {'planner': 0, 'longcontrol': 1, 'carcontroller': 2}[layer]


def test_constructor_error_is_not_an_update_attempt(monkeypatch):
  messages, window, _, _, _ = replay_fixture(monkeypatch)

  def failing_constructor(*arguments, **options):
    raise RuntimeError('synthetic planner construction failure')

  monkeypatch.setattr(replay.planner_module.LongitudinalPlanner, '__init__', failing_constructor)
  diagnostics = {}
  with pytest.raises(RuntimeError, match='construction failure'):
    replay.replay_window(None, window, messages=messages, diagnostics=diagnostics)
  assert diagnostics['initialized']
  assert not diagnostics['components']['planner']
  assert diagnostics['counts']['planner']['attempted']['total'] == 0
  assert diagnostics['first_attempted']['plan_mono'] is None
  assert diagnostics['current']['service'] == 'carParamsSP'


def test_publish_failure_keeps_completed_planner_call_without_matched_row(monkeypatch):
  messages, window, _, _, _ = replay_fixture(monkeypatch)

  def failing_publish(*arguments, **options):
    raise RuntimeError('synthetic publication failure')

  monkeypatch.setattr(replay.planner_module.LongitudinalPlanner, 'publish', failing_publish)
  diagnostics = {}
  with pytest.raises(RuntimeError, match='publication failure'):
    replay.replay_window(None, window, messages=messages, unified=True, diagnostics=diagnostics)
  assert diagnostics['counts']['planner']['completed']['total'] == 1
  assert diagnostics['first_completed']['plan_mono'] == 1_010_000_000
  assert diagnostics['first_executed']['plan_mono'] == 1_010_000_000
  assert diagnostics['coverage']['matched_plans'] == 0


def test_output_sink_error_keeps_completed_plan_call_without_attempting_control(monkeypatch):
  messages, window, _, _, _ = replay_fixture(monkeypatch)

  def failing_sink(kind, row):
    raise OSError('synthetic numeric row failure')

  diagnostics = {}
  with pytest.raises(OSError, match='numeric row failure'):
    replay.replay_window(None, window, messages=messages, unified=True, row_sink=failing_sink, diagnostics=diagnostics)
  assert diagnostics['counts']['planner']['completed']['total'] == 1
  assert diagnostics['counts']['longcontrol']['attempted']['total'] == 0
  assert diagnostics['current']['mono'] == 1_010_000_000
  assert diagnostics['status'] == 'ERROR'


def test_end_coverage_failure_does_not_drop_completed_prefix(monkeypatch):
  messages, window, _, _, _ = replay_fixture(monkeypatch)
  messages = [message for message in messages if message.which() != 'sendcan']
  diagnostics = {}
  with pytest.raises(ValueError, match='Incomplete replay'):
    replay.replay_window(None, window, messages=messages, unified=True, diagnostics=diagnostics)
  assert diagnostics['status'] == 'ERROR'
  assert diagnostics['coverage']['matched_plans'] == 1
  assert diagnostics['coverage']['matched_controls'] == 1
  assert diagnostics['coverage']['sendcan'] == 0
  assert diagnostics['counts']['carcontroller']['attempted']['total'] == 0
  assert diagnostics['first_completed']['controller_mono'] is None
  assert diagnostics['current'] == {'service': 'carState', 'mono': 1_025_000_000, 'region': 'score', 'in_score': True}


@pytest.mark.parametrize('unified', [False, True])
def test_cadence_mismatch_still_rejects_and_keeps_actual_completed_can_calls(monkeypatch, unified):
  messages, window, _, dispatched = cadence_fixture(monkeypatch, warmup=0.94)
  diagnostics = {}
  with pytest.raises(ValueError, match='Controller cadence mismatch'):
    replay.replay_window(None, window, messages=messages, unified=unified, diagnostics=diagnostics)
  assert len(dispatched) == 2
  assert diagnostics['counts']['carcontroller']['attempted']['total'] == 2
  assert diagnostics['counts']['carcontroller']['completed']['total'] == 2
  assert diagnostics['counts']['carcontroller']['completed']['warmup'] == 1
  assert diagnostics['counts']['carcontroller']['completed']['score'] == 1
  assert diagnostics['coverage']['matched_sendcan'] == 0
  assert diagnostics['current']['mono'] == 1_040_000_000
  assert diagnostics['resets'] == []


@pytest.mark.parametrize('unified', [False, True])
def test_before_boundary_records_input_only_without_new_execution(monkeypatch, unified):
  messages, window, _, dispatched = cadence_fixture(monkeypatch)
  diagnostics = {}
  replay.replay_window(None, window, messages=messages, unified=unified, diagnostics=diagnostics)
  assert len(dispatched) == 1
  for layer in ('planner', 'longcontrol', 'carcontroller'):
    assert diagnostics['counts'][layer]['attempted']['read_only_prefix'] == 0
    assert diagnostics['counts'][layer]['completed']['read_only_prefix'] == 0
  assert diagnostics['read_only_prefix']['controls'] == 3
  assert diagnostics['read_only_prefix']['sendcan'] == 2
  assert diagnostics['first_attempted']['controller_mono'] == 1_040_000_000


def test_exact_provider_failure_keeps_requested_dependency_and_current_plan(monkeypatch):
  messages, window, _, _, _ = replay_fixture(monkeypatch)
  required_mono = 2 ** 53 + 17

  def missing_provider(plan_mono, state):
    error = ValueError(f'Missing consumed carState input {required_mono}')
    error.replay_dependency = {'reason': 'missing_consumed_carState', 'required': [{'service': 'carState', 'mono': required_mono}]}
    raise error

  diagnostics = {}
  with pytest.raises(ValueError, match='Missing consumed carState'):
    replay.replay_window(None, window, messages=messages, unified=True, planner_inputs=missing_provider, diagnostics=diagnostics)
  sample = diagnostics['missing_dependencies']['latest']
  assert sample['mono'] == 1_010_000_000
  assert sample['service'] == 'longitudinalPlan'
  assert sample['required'] == [{'service': 'carState', 'mono': required_mono}]
  assert sample['execution_boundary_mono'] == 0
  assert sample['score_interval_mono'] == [1_000_000_000, 1_050_000_000]
  assert sample['in_score']
  assert diagnostics['counts']['planner']['attempted']['total'] == 0
  assert json.loads(json.dumps(diagnostics))['missing_dependencies']['latest']['required'][0]['mono'] == required_mono


def test_missing_generated_plan_keeps_uint64_request_without_float_conversion(monkeypatch):
  messages, window, _, _, _ = replay_fixture(monkeypatch)
  required_mono = 2 ** 53 + 17
  for message in messages:
    if message.which() == 'controlsState' and message.logMonoTime == 1_015_000_000:
      message.controlsState.longitudinalPlanMonoTime = required_mono
  diagnostics = {}
  with pytest.raises(ValueError, match='Incomplete replay'):
    replay.replay_window(None, window, messages=messages, unified=True, diagnostics=diagnostics)
  sample = next(sample for sample in diagnostics['missing_dependencies']['samples']
                if sample['reason'] == 'control_plan' and sample['in_score'])
  assert sample['required'] == [{'service': 'longitudinalPlan', 'mono': required_mono}]
  assert type(sample['required'][0]['mono']) is int
  assert sample['mono'] == 1_020_000_000
  assert diagnostics['coverage']['controls'] == 1
  assert diagnostics['coverage']['matched_controls'] == 0


def test_high_input_ids_are_exact_in_all_new_json_receipts(monkeypatch):
  messages, window, _, _, _ = replay_fixture(monkeypatch)
  offset = 2 ** 53 + 17
  messages = shifted_messages(messages, offset)
  window['origin'] = offset / 1e9
  diagnostics = {}
  replay.replay_window(None, window, messages=messages, unified=True, diagnostics=diagnostics)
  restored = json.loads(json.dumps(diagnostics))
  expected = {'plan_mono': offset + 1_010_000_000, 'control_mono': offset + 1_020_000_000,
              'controller_mono': offset + 1_040_000_000}
  assert restored['first_attempted'] == expected
  assert restored['first_completed'] == expected
  assert restored['first_executed'] == expected
  assert restored['current']['mono'] == offset + 1_040_000_000
  assert all(type(value) is int for value in restored['first_completed'].values())


def test_missing_samples_are_bounded_but_counts_and_latest_survive(monkeypatch):
  messages, window, _, _, _ = replay_fixture(monkeypatch)
  messages = [message for message in messages if message.which() not in ('longitudinalPlan', 'longitudinalPlanSP')]
  controls = [message for message in messages if message.which() in ('controlsState', 'carControl') and message.logMonoTime >= 1_000_000_000]
  for cycle in range(1, 51):
    messages += shifted_messages(controls, cycle * 10_000_000)
  messages.sort(key=lambda message: message.logMonoTime)
  window['end'] = 2.0
  diagnostics = {}
  with pytest.raises(ValueError, match='Incomplete replay'):
    replay.replay_window(None, window, messages=messages, unified=True, diagnostics=diagnostics)
  missing = diagnostics['missing_dependencies']
  assert len(missing['samples']) <= replay.ReplayDiagnostics.MAX_MISSING_SAMPLES
  assert missing['samples_omitted'] > 0
  assert missing['counts']['control_plan']['score'] == 51
  assert missing['latest']['mono'] == 1_520_000_000
  assert missing['samples_omitted'] + len(missing['samples']) == sum(counts['total'] for counts in missing['counts'].values())


def test_unscored_prefix_samples_cannot_displace_first_score_failure(monkeypatch):
  messages, window, _, _, _ = replay_fixture(monkeypatch)
  window['warmup'] = 1.0
  original_plan = next(message for message in messages if message.which() == 'longitudinalPlan')
  original_sp = next(message for message in messages if message.which() == 'longitudinalPlanSP')
  for index in range(50):
    plan, decision = deepcopy(original_plan), deepcopy(original_sp)
    plan.logMonoTime = 920_000_000 + index * 1_000_000
    plan.longitudinalPlan.modelMonoTime = 0
    decision.logMonoTime = plan.logMonoTime + 500_000
    messages += [plan, decision]
  required_mono = 2 ** 53 + 17
  for message in messages:
    if message.which() == 'controlsState' and message.logMonoTime == 1_015_000_000:
      message.controlsState.longitudinalPlanMonoTime = required_mono
  messages.sort(key=lambda message: message.logMonoTime)
  diagnostics = {}
  with pytest.raises(ValueError, match='Incomplete replay'):
    replay.replay_window(None, window, messages=messages, unified=True, diagnostics=diagnostics)
  missing = diagnostics['missing_dependencies']
  assert missing['counts']['plan_model']['read_only_prefix'] == 50
  assert missing['samples_omitted'] >= 48
  score_failure = next(sample for sample in missing['samples'] if sample['reason'] == 'control_plan' and sample['in_score'])
  assert score_failure['required'] == [{'service': 'longitudinalPlan', 'mono': required_mono}]


@pytest.mark.parametrize('solver_status', [0, 9])
def test_solution_status_is_not_used_as_a_solver_invocation_count(monkeypatch, solver_status):
  messages, window, _, _, _ = replay_fixture(monkeypatch, solver_status=solver_status)
  diagnostics = {}
  result = replay.replay_window(None, window, messages=messages, unified=True, diagnostics=diagnostics)
  assert diagnostics['solver_failures'] == result['solver_failures'] == int(solver_status != 0)
  assert diagnostics['solver_invocations']['known'] is False
  assert diagnostics['solver_invocations']['attempted'] is None
  assert diagnostics['solver_invocations']['completed'] is None
  assert diagnostics['counts']['planner']['completed']['score'] == 1


@pytest.mark.parametrize('solve_delta,reset_delta,native_return,completed', [(0, 0, 123, 0), (1, 0, 0, 1), (1, 1, 9, 1), (1, 1, None, 0)])
def test_actual_solver_counter_deltas_do_not_use_cached_status(monkeypatch, solve_delta, reset_delta, native_return, completed):
  messages, window, _, _, _ = counter_fixture(
    monkeypatch, solve_delta=solve_delta, reset_delta=reset_delta, native_return=native_return)
  diagnostics = {}
  replay.replay_window(None, window, messages=messages, unified=True, diagnostics=diagnostics)
  solver = diagnostics['solver_invocations']
  assert solver['known'] and solver['attempted_known'] and solver['completed_known'] and solver['reset_known']
  assert solver['attempted']['total'] == solver['attempted']['score'] == solve_delta
  assert solver['completed']['total'] == solver['completed']['score'] == completed
  assert solver['reset_calls']['total'] == solver['reset_calls']['score'] == reset_delta
  assert solver['initial_snapshot']['solve_calls'] == 2 ** 53 + 17
  assert solver['initial_snapshot']['reset_calls'] == 4
  assert solver['returned_statuses'] == ({str(native_return): 1} if completed else {})
  assert diagnostics['solver_failures'] == 1
  assert solver['first_attempted_mono'] == (1_010_000_000 if solve_delta else None)
  assert solver['first_completed_mono'] == (1_010_000_000 if completed else None)
  restored = json.loads(json.dumps(diagnostics))
  assert restored['solver_invocations']['initial_snapshot']['solve_calls'] == 2 ** 53 + 17


def test_actual_solver_counter_totals_are_partitioned_by_execution_region(monkeypatch):
  messages, window, _, _, _ = counter_fixture(monkeypatch, reset_delta=1)
  cycles = [message for message in messages if message.logMonoTime >= 1_000_000_000]
  messages += shifted_messages(cycles, 100_000_000) + shifted_messages(cycles, 200_000_000)
  window.update(start=1.1, end=1.15)
  diagnostics = {}
  replay.replay_window(None, window, messages=messages, unified=True, diagnostics=diagnostics)
  solver = diagnostics['solver_invocations']
  for name in ('attempted', 'completed', 'reset_calls'):
    assert solver[name] == {'total': 3, 'read_only_prefix': 0, 'dependency_prefix': 0, 'warmup': 1, 'score': 1, 'after_score': 1}
  assert solver['returned_statuses'] == {'0': 3}
  assert diagnostics['solver_failures'] == 1


def test_solver_attempt_is_retained_if_planner_update_raises(monkeypatch):
  messages, window, _, _, _ = counter_fixture(
    monkeypatch, native_return=None, update_error=RuntimeError('native solver interrupted'))
  diagnostics = {}
  with pytest.raises(RuntimeError, match='native solver interrupted'):
    replay.replay_window(None, window, messages=messages, unified=True, diagnostics=diagnostics)
  solver = diagnostics['solver_invocations']
  assert solver['known']
  assert solver['attempted']['total'] == 1
  assert solver['completed']['total'] == 0
  assert diagnostics['counts']['planner']['attempted']['total'] == 1
  assert diagnostics['counts']['planner']['completed']['total'] == 0
  assert diagnostics['status'] == 'ERROR'


@pytest.mark.parametrize('native_return', [0, None])
def test_multiple_solver_calls_with_one_return_marker_do_not_invent_completion_counts(monkeypatch, native_return):
  messages, window, _, _, _ = counter_fixture(monkeypatch, solve_delta=2, native_return=native_return)
  diagnostics = {}
  replay.replay_window(None, window, messages=messages, unified=True, diagnostics=diagnostics)
  solver = diagnostics['solver_invocations']
  assert solver['attempted_known']
  assert solver['attempted']['total'] == 2
  assert not solver['known'] and not solver['completed_known']
  assert solver['completed'] is None
  assert solver['returned_statuses'] is None


def test_missing_return_marker_leaves_return_count_unknown_even_with_real_attempt_counter(monkeypatch):
  messages, window, _, _, _ = counter_fixture(monkeypatch, return_marker=False)
  diagnostics = {}
  replay.replay_window(None, window, messages=messages, unified=True, diagnostics=diagnostics)
  solver = diagnostics['solver_invocations']
  assert solver['attempted_known'] and solver['reset_known']
  assert solver['attempted']['total'] == 1
  assert not solver['known'] and not solver['completed_known']
  assert solver['completed'] is None


@pytest.mark.parametrize('mutation', ['missing', 'backwards', 'replacement'])
def test_changed_counter_source_is_unknown_not_an_artificial_zero(monkeypatch, mutation):
  def mutate(planner):
    if mutation == 'missing':
      del planner.mpc.diagnostic_reset_calls
    elif mutation == 'backwards':
      planner.mpc.diagnostic_solve_calls = 0
    else:
      planner.mpc = deepcopy(planner.mpc)

  messages, window, _, _, _ = counter_fixture(monkeypatch, mutate=mutate)
  diagnostics = {}
  replay.replay_window(None, window, messages=messages, unified=True, diagnostics=diagnostics)
  solver = diagnostics['solver_invocations']
  assert not solver['known'] and not solver['attempted_known']
  assert solver['attempted'] is None and solver['completed'] is None and solver['reset_calls'] is None
  assert solver['observation_gap']['mono'] == 1_010_000_000
  assert solver['last_observation']['before']['solve_calls'] == 2 ** 53 + 17
  assert diagnostics['status'] == 'COMPLETED'


def test_counter_observation_error_does_not_replace_original_update_error(monkeypatch):
  class BrokenCounter:
    @property
    def diagnostic_solve_calls(self):
      raise OSError('diagnostic property unreadable')

  def mutate(planner):
    planner.mpc = BrokenCounter()

  messages, window, _, _, _ = counter_fixture(
    monkeypatch, mutate=mutate, update_error=RuntimeError('original planner interruption'))
  diagnostics = {}
  with pytest.raises(RuntimeError, match='original planner interruption'):
    replay.replay_window(None, window, messages=messages, unified=True, diagnostics=diagnostics)
  assert not diagnostics['solver_invocations']['known']
  assert diagnostics['solver_invocations']['attempted'] is None
  assert diagnostics['error'] == 'RuntimeError: original planner interruption'


@pytest.mark.parametrize('unified', [False, True])
def test_diagnostic_destination_does_not_change_control_order_or_outputs(monkeypatch, unified):
  records = []
  for with_destination in (False, True):
    with monkeypatch.context() as context:
      messages, window, controls, encoded, _ = replay_fixture(context)
      options = {'diagnostics': {}} if with_destination else {}
      result = replay.replay_window(None, window, messages=messages, unified=unified, **options)
      records.append((result, deepcopy(controls), deepcopy(encoded)))
  assert records[0] == records[1]
