"""Default failure evidence cannot change solve, reset, or acceptance behavior."""
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LongitudinalMpc, MPC_INVALID_INPUT, MPC_INVALID_SOLUTION, MPC_SOLVER_ERROR, MpcReference,
)
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.tests.test_radar_fusion_mpc import EPOCH_NS, fused_state, object_lead


def no_leads():
  absent = SimpleNamespace(present=False, modelProb=0.0)
  return SimpleNamespace(leadOne=absent, leadTwo=absent)


@pytest.fixture
def diagnosed_mpc():
  # Native numerics have their own tests. This boundary double independently
  # supplies distinct outer/native-QP statuses and erases statistics on reset.
  native = Mock(spec=['reset', 'set', 'cost_set', 'constraints_set', 'solve', 'get', 'get_stats', 'get_residuals'])
  native.status = 4

  def solve():
    native.statistics = np.array([[0., 1.], [0., 2.], [0., 24.]])
    return native.status

  def get_stats(field):
    return native.statistics.copy() if field == 'statistics' else np.array([0.001])

  native.solve.side_effect = solve
  native.reset.side_effect = lambda: setattr(native, 'statistics', np.zeros((3, 2)))
  native.get_stats.side_effect = get_stats
  native.get.side_effect = lambda stage, field: np.zeros(3 if field == 'x' else 1)
  with patch.object(long_mpc, 'AcadosOcpSolverCython', return_value=native), \
       patch.object(long_mpc.cloudlog, 'event') as event, patch.object(long_mpc.cloudlog, 'warning'):
    mpc = LongitudinalMpc()
    yield mpc, native, event


def attempt(mpc, radar=None, *, now_ns=EPOCH_NS):
  mpc.set_cur_state(10.0, -0.25)
  mpc.update(no_leads() if radar is None else radar, reference=MpcReference(20.0), now_ns=now_ns, reference_ns=now_ns)


def failure_record(event):
  name, = event.call_args.args
  assert name == 'long_mpc_failure_v1'
  record = event.call_args.kwargs
  # No capnp readers, exceptions, arrays, NaN, or unbounded input serialization.
  assert len(json.dumps(record, allow_nan=False)) < 12_000
  return record


@pytest.mark.parametrize('kind,reason', [
  ('clock', 'Invalid fusion clock'),
  ('expired', 'Unqualified admitted fusion object'),
  ('edge', 'Invalid edge caution object'),
])
def test_pre_solve_rejection_records_exact_branch_and_offending_object(diagnosed_mpc, kind, reason):
  mpc, native, event = diagnosed_mpc
  lead = object_lead(track=751, path='edge' if kind == 'edge' else 'inPath')
  now_ns = 0 if kind == 'clock' else EPOCH_NS + 400_000_000
  attempt(mpc, fused_state(lead), now_ns=now_ns)
  assert mpc.solution_status == MPC_INVALID_INPUT
  assert mpc.failure_reason == 'invalid_input'
  record = failure_record(event)
  assert record['stage'] == 'fusion_projection'
  assert record['reason'] == reason and record['exception'] == 'ValueError'
  assert record['nativeReturn'] is None and not record['native']['known']
  assert record['x0'] == [0.0, 10.0, -0.25]
  assert not record['constraints']['problemInstalled']
  if kind != 'clock':
    rejected = [row for row in record['radar']['objects'] if row['rejected']]
    assert len(rejected) == 1 and rejected[0]['trackId'] == 751
    assert rejected[0]['positionMonoTime'] == EPOCH_NS
  native.solve.assert_not_called()
  native.get_stats.assert_not_called()
  np.testing.assert_array_equal(mpc.x0, [0.0, 0.0, 0.0])


def test_native_statistics_and_problem_survive_reset_as_original_evidence(diagnosed_mpc):
  mpc, native, event = diagnosed_mpc
  attempt(mpc)
  record = failure_record(event)
  assert record['status'] == record['nativeReturn'] == 4
  assert record['stage'] == 'native_solve'
  assert record['native'] == {'known': True, 'qpResidualsKnown': False, 'reason': '',
                              'iteration': 1, 'qpStatus': 2, 'qpIterations': 24}
  assert record['constraints']['problemInstalled']
  assert record['x0'] == [0.0, 10.0, -0.25]
  assert record['constraints']['parameterRanges'][2][0] > 0.0
  assert mpc.solution_status == mpc.last_solver_status == 4
  assert mpc.failure_reason == 'solver_status' and not mpc.previous_solution_valid
  np.testing.assert_array_equal(mpc.x0, np.zeros(3))
  np.testing.assert_array_equal(native.statistics, np.zeros((3, 2)))
  native.get_residuals.assert_not_called()


@pytest.mark.parametrize('layout', ['residuals', 'unknown'])
def test_optional_native_statistics_never_require_recomputation(diagnosed_mpc, layout):
  mpc, native, event = diagnosed_mpc
  stats = np.array([[1.], [2.], [24.], [1.e-5], [2.e-5], [3.e-5], [4.e-5]]) if layout == 'residuals' else np.zeros((4, 2))
  native.get_stats.side_effect = lambda field: stats
  attempt(mpc)
  record = failure_record(event)['native']
  if layout == 'residuals':
    assert record['known'] and record['qpResidualsKnown']
    assert record['qpResiduals'] == [1.e-5, 2.e-5, 3.e-5, 4.e-5]
  else:
    assert not record['known'] and record['reason'] == 'unknown_statistics_layout'
  assert mpc.solution_status == 4
  native.get_residuals.assert_not_called()


def test_pre_solve_failure_never_attributes_the_previous_native_success(diagnosed_mpc):
  mpc, native, event = diagnosed_mpc
  native.status = 0
  attempt(mpc)
  assert mpc.diagnostic_last_solver_return == 0
  native.get_stats.reset_mock()
  attempt(mpc, fused_state(object_lead()), now_ns=0)
  record = failure_record(event)
  assert record['status'] == MPC_INVALID_INPUT and record['nativeReturn'] is None
  assert not record['native']['known']
  native.get_stats.assert_not_called()


@pytest.mark.parametrize('mode,status,stage,native_return', [
  ('solve', MPC_SOLVER_ERROR, 'native_solve', None),
  ('readback', MPC_SOLVER_ERROR, 'native_readback', 0),
  ('nonfinite', MPC_INVALID_SOLUTION, 'post_solve', 0),
])
def test_exception_stage_is_separate_from_native_return(diagnosed_mpc, mode, status, stage, native_return):
  mpc, native, event = diagnosed_mpc
  native.status = 0
  if mode == 'solve':
    native.solve.side_effect = RuntimeError('native solve exception')
  elif mode == 'readback':
    native.get.side_effect = RuntimeError('native readback exception')
  else:
    native.get.side_effect = lambda stage, field: np.full(3 if field == 'x' else 1, np.nan)
  attempt(mpc)
  record = failure_record(event)
  assert mpc.solution_status == record['status'] == status
  assert record['stage'] == stage and record['nativeReturn'] == native_return
  assert record['exception'] == ('' if mode == 'nonfinite' else 'RuntimeError')
  for name in ('x0', 'x_sol', 'u_sol', 'a_prev'):
    assert np.all(getattr(mpc, name) == 0.0)


def test_first_failure_category_change_and_success_reset_can_use_the_burst_budget(diagnosed_mpc):
  mpc, native, event = diagnosed_mpc
  with patch.object(long_mpc.time, 'monotonic', return_value=1.0) as clock:
    attempt(mpc)
    assert event.call_count == 1  # First fault must log even during early boot.
    clock.return_value = 1.05
    attempt(mpc)
    assert event.call_count == 1
    clock.return_value = 6.0
    attempt(mpc)
    assert event.call_count == 2 and failure_record(event)['suppressedCount'] == 1
    clock.return_value = 6.05
    attempt(mpc, fused_state(object_lead()), now_ns=0)
    assert event.call_count == 3
    native.status = 0
    attempt(mpc)
    assert mpc.solution_status == 0
    native.status = 4
    attempt(mpc)
    assert event.call_count == 4


@pytest.mark.parametrize('alternate_success', [False, True])
def test_global_budget_bounds_one_hundred_alternating_ticks_before_payload_work(diagnosed_mpc, alternate_success):
  mpc, native, event = diagnosed_mpc
  with patch.object(long_mpc.time, 'monotonic', return_value=0.0) as clock, \
       patch.object(mpc, '_failure_radar_summary', wraps=mpc._failure_radar_summary) as summarize:
    failures = 0
    for tick in range(100):
      clock.return_value = tick / 20.0
      native.status = 4 if tick % 2 == 0 else (0 if alternate_success else 2)
      attempt(mpc)
      failures += native.status != 0
      assert mpc.solution_status == native.status
      if native.status:
        assert not mpc.previous_solution_valid and np.all(mpc.x_sol == 0.0)
    assert event.call_count == summarize.call_count == 3
    assert event.call_args_list[0].kwargs['status'] == 4
    assert event.call_args_list[0].kwargs['suppressedCount'] == 0
    assert sum(call.args == ('statistics',) for call in native.get_stats.call_args_list) == 3
    clock.return_value = 5.0
    native.status = 2  # A changed category is eligible exactly at the refill boundary.
    attempt(mpc)
    assert event.call_count == summarize.call_count == 4
    assert failure_record(event)['suppressedCount'] == failures - 3
    # A following success/reset cannot replenish the exhausted global budget.
    native.status = 0
    attempt(mpc)
    native.status = 4
    attempt(mpc)
    assert event.call_count == 4


def test_global_budget_clock_boundaries_preserve_the_high_water_mark(diagnosed_mpc):
  mpc, _, event = diagnosed_mpc
  with patch.object(long_mpc.time, 'monotonic', return_value=0.0) as clock:
    for status in (4, MPC_INVALID_INPUT, MPC_SOLVER_ERROR):
      mpc._log_failure(status, 'clock_fixture')
    assert event.call_count == 3
    rejected_clocks = (float('nan'), float('inf'), -1.0, 4.999999999, 2.0, 0.0, 4.999999999)
    for timestamp in rejected_clocks:
      clock.return_value = timestamp
      mpc._log_failure(4, 'clock_fixture')
    assert event.call_count == 3
    clock.return_value = 5.0
    mpc._log_failure(4, 'clock_fixture')
    assert event.call_count == 4
    assert failure_record(event)['suppressedCount'] == len(rejected_clocks)
    # Large forward jumps cap the available credit at the same three-record
    # burst; neither clock regression nor idle time creates an unbounded dump.
    clock.return_value = 1000.0
    for status in (MPC_INVALID_INPUT, MPC_SOLVER_ERROR, 4, MPC_INVALID_SOLUTION):
      mpc._log_failure(status, 'clock_fixture')
    assert event.call_count == 7


@pytest.mark.parametrize('failure', ['statistics', 'payload', 'event', 'warning'])
def test_diagnostic_failures_cannot_change_control_failure_or_retry_without_limit(diagnosed_mpc, failure):
  mpc, native, event = diagnosed_mpc
  if failure == 'statistics':
    native.get_stats.side_effect = RuntimeError('statistics unavailable')
  elif failure == 'payload':
    mpc._failure_radar_summary = Mock(side_effect=RuntimeError('malformed radar diagnostic'))
  elif failure == 'event':
    event.side_effect = RuntimeError('logger unavailable')
  else:
    long_mpc.cloudlog.warning.side_effect = RuntimeError('warning logger unavailable')
  with patch.object(long_mpc.time, 'monotonic', return_value=10.0):
    attempt(mpc)
    attempt(mpc)
  assert native.solve.call_count == 2 and native.reset.call_count == 3
  assert mpc.solution_status == mpc.last_solver_status == 4
  assert mpc.failure_reason == 'solver_status' and not mpc.previous_solution_valid
  for name in ('x0', 'x_sol', 'u_sol', 'v_solution', 'a_solution', 'j_solution', 'a_prev'):
    assert np.all(getattr(mpc, name) == 0.0)
  assert event.call_count == 1
  if failure == 'statistics':
    assert not failure_record(event)['native']['known']
    assert native.get_stats.call_count == 1
  elif failure == 'payload':
    assert not failure_record(event)['radar']['known']
  native.get_residuals.assert_not_called()


def test_success_does_not_collect_native_diagnostic_statistics(diagnosed_mpc):
  mpc, native, event = diagnosed_mpc
  native.status = 0
  attempt(mpc)
  assert mpc.solution_status == 0 and mpc.previous_solution_valid
  assert [call.args for call in native.get_stats.call_args_list] == [('time_tot',)]
  native.get_residuals.assert_not_called()
  event.assert_not_called()


@pytest.mark.parametrize('post_solve_rejection', [False, True])
def test_planner_records_actual_consumed_publications_without_opt_in(diagnosed_mpc, post_solve_rejection):
  mpc, native, event = diagnosed_mpc

  class State(dict):
    logMonoTime = {'carState': EPOCH_NS - 1_000_000, 'modelV2': EPOCH_NS - 2_000_000, 'radarState': EPOCH_NS - 3_000_000}

  state = State(radarState=no_leads())
  planner = LongitudinalPlanner.__new__(LongitudinalPlanner)
  planner.mpc, planner.trace, planner.observation = mpc, None, None

  def update(current, *, now_ns):
    attempt(mpc, current['radarState'], now_ns=now_ns)
    if post_solve_rejection:
      mpc.solution_status = MPC_INVALID_INPUT
      mpc.failure_reason = 'unqualified_model_unknown_fixture'
      mpc.previous_solution_valid = False

  native.status = 0 if post_solve_rejection else 4
  planner._update_reference = update
  planner.update(state, now_ns=EPOCH_NS)
  assert event.call_count == 1
  record = failure_record(event)
  assert record['consumerMonoTime'] == record['mpcConsumerMonoTime'] == EPOCH_NS
  assert record['carStateMonoTime'] == state.logMonoTime['carState']
  assert record['modelMonoTime'] == state.logMonoTime['modelV2']
  assert record['radarStateMonoTime'] == state.logMonoTime['radarState']
  assert record['stage'] == ('planner_result' if post_solve_rejection else 'native_solve')
  assert record['nativeReturn'] == (0 if post_solve_rejection else 4)


def test_context_read_failure_cannot_replace_the_underlying_planner_result(diagnosed_mpc):
  mpc, _, event = diagnosed_mpc

  class BrokenTimes(dict):
    @property
    def logMonoTime(self):
      raise RuntimeError('diagnostic timestamp access failed')

  planner = LongitudinalPlanner.__new__(LongitudinalPlanner)
  planner.mpc, planner.trace, planner.observation = mpc, None, None
  planner._update_reference = lambda sm, now_ns: 'underlying result'
  assert planner.update(BrokenTimes(), now_ns=EPOCH_NS) == 'underlying result'
  assert mpc.diagnostic_failure_log_errors == 1
  event.assert_not_called()
