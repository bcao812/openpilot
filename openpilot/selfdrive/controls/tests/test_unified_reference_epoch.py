from numbers import Integral
from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.honda.values import CAR
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_planner, make_state


BASE_NS = 1_000_000_000
STEP_NS = round(DT_MDL * 1e9)


def native_reference(speed, initial_accel=0.0):
  """A finite forward trajectory with jerk -2 m/s^3 over the first 625 ms.

  Acceleration is held at -2 after a transition interval. Integrate each native
  interval exactly, so this fixture does not confuse a reference acceleration
  with an actuator command or an independently sampled model velocity.
  """
  acceleration = np.maximum(initial_accel - 2.0 * T_IDXS, -2.0)
  jerk = np.diff(acceleration) / np.diff(T_IDXS)
  states = np.zeros((len(T_IDXS), 3))
  states[0] = (0.0, speed, initial_accel)
  for index, duration in enumerate(np.diff(T_IDXS)):
    position, velocity, accel = states[index]
    states[index + 1] = (
      position + velocity * duration + 0.5 * accel * duration ** 2 + jerk[index] * duration ** 3 / 6.0,
      velocity + accel * duration + 0.5 * jerk[index] * duration ** 2,
      acceleration[index + 1],
    )
  return states, jerk


def update_with_native_reference(planner, state, now_ns, *, initial_accel=0.0, solver_status=0):
  """Keep the production update/run plumbing and control only solver output.

  These tests also execute against the old implementation: no fallback calls
  emulate the new epoch API, and the failure is observable in consumed a_prev
  and its cost, rather than an AttributeError for a missing helper.
  """
  if isinstance(now_ns, Integral) and not isinstance(now_ns, bool) and 0 < now_ns < (1 << 64):
    for service in state.logMonoTime:
      state.logMonoTime[service] = int(now_ns)
  state['carState'].aEgo = initial_accel
  native_solver = planner.mpc.solver
  solver = Mock(wraps=native_solver)
  solver.solve.return_value = solver_status
  solver.get_stats.return_value = [0.0]

  def solution(index, field):
    states, jerk = native_reference(float(planner.mpc.x0[1]), initial_accel)
    return states[index] if field == 'x' else np.array([jerk[index]])

  solver.get.side_effect = solution
  original_run = planner.mpc.run
  consumed_history = []

  def run():
    consumed_history.append(planner.mpc.params[:, 3].copy())
    return original_run()

  with patch.object(planner.mpc, 'solver', solver), \
       patch.object(planner.mpc, 'run', side_effect=run), \
       patch.object(planner.mpc, 'set_weights', wraps=planner.mpc.set_weights) as weights:
    planner.update(state, now_ns=now_ns)
  used_history = bool(weights.call_args_list[0].args[0]) if weights.call_args_list else None
  return used_history, consumed_history[-1] if consumed_history else None


@pytest.mark.parametrize('elapsed_ns', [STEP_NS, 100_000_000, 150_000_000, 225_000_000])
def test_native_reference_history_uses_elapsed_consumer_time(elapsed_ns):
  planner, _ = make_planner()
  state = make_state()
  first_used, _ = update_with_native_reference(planner, state, BASE_NS)
  previous_native_acceleration = planner.mpc.a_solution.copy()
  used_history, consumed = update_with_native_reference(planner, state, BASE_NS + elapsed_ns)
  assert not first_used
  assert used_history
  assert consumed[0] == pytest.approx(-2.0 * elapsed_ns / 1e9)
  np.testing.assert_allclose(consumed, np.interp(T_IDXS + elapsed_ns / 1e9, T_IDXS, previous_native_acceleration))


def test_exact_50_ms_alignment_preserves_the_previous_interpolation():
  planner, _ = make_planner()
  state = make_state()
  update_with_native_reference(planner, state, BASE_NS)
  previous_fixed_step = planner.mpc.a_prev.copy()
  used_history, consumed = update_with_native_reference(planner, state, BASE_NS + STEP_NS)
  assert used_history
  np.testing.assert_array_equal(consumed, previous_fixed_step)


@pytest.mark.parametrize('bad_time', [
  BASE_NS + STEP_NS, BASE_NS, BASE_NS - STEP_NS,
  0, -1, True, False, 1.5, float('nan'), float('inf'), 1 << 64, '1100000000',
])
def test_bad_consumer_clock_drops_history_and_does_not_date_a_new_reference(bad_time):
  planner, _ = make_planner()
  state = make_state()
  update_with_native_reference(planner, state, BASE_NS)
  update_with_native_reference(planner, state, BASE_NS + STEP_NS)
  used_bad, _ = update_with_native_reference(planner, state, bad_time, initial_accel=-0.4)
  used_recovery, _ = update_with_native_reference(planner, state, BASE_NS + 2 * STEP_NS, initial_accel=-0.6)
  used_next, consumed = update_with_native_reference(planner, state, BASE_NS + 3 * STEP_NS)
  assert not used_bad
  assert not used_recovery
  assert used_next
  assert consumed[0] == pytest.approx(-0.7)


def test_repeated_and_backwards_updates_do_not_accumulate_reference_age():
  planner, _ = make_planner()
  state = make_state()
  update_with_native_reference(planner, state, BASE_NS)
  update_with_native_reference(planner, state, BASE_NS + 2 * STEP_NS)
  for timestamp in (BASE_NS + 2 * STEP_NS, BASE_NS + STEP_NS, BASE_NS + 2 * STEP_NS):
    used_history, _ = update_with_native_reference(planner, state, timestamp, initial_accel=-0.4)
    assert not used_history
  used_recovery, _ = update_with_native_reference(planner, state, BASE_NS + 3 * STEP_NS, initial_accel=-0.6)
  used_next, consumed = update_with_native_reference(planner, state, BASE_NS + 4 * STEP_NS)
  assert not used_recovery
  assert used_next
  assert consumed[0] == pytest.approx(-0.7)


@pytest.mark.parametrize('elapsed_ns', [10_000_000_000, 10_000_000_001, 30_000_000_000])
def test_expired_native_reference_is_not_extrapolated_into_a_new_history(elapsed_ns):
  planner, _ = make_planner()
  state = make_state()
  update_with_native_reference(planner, state, BASE_NS)
  used_history, _ = update_with_native_reference(planner, state, BASE_NS + elapsed_ns, initial_accel=-0.4)
  used_next, consumed = update_with_native_reference(planner, state, BASE_NS + elapsed_ns + STEP_NS)
  assert not used_history
  assert used_next
  assert consumed[0] == pytest.approx(-0.5)


def test_reset_on_failed_solve_preserves_consumer_clock_high_water():
  planner, _ = make_planner()
  state = make_state()
  update_with_native_reference(planner, state, BASE_NS)
  update_with_native_reference(planner, state, BASE_NS + 2 * STEP_NS)
  update_with_native_reference(planner, state, BASE_NS + 4 * STEP_NS, solver_status=4)
  assert planner.mpc.solution_status == 4
  assert not planner.mpc.previous_solution_valid
  for timestamp in (BASE_NS + STEP_NS, BASE_NS + 3 * STEP_NS, BASE_NS + 4 * STEP_NS):
    used_history, _ = update_with_native_reference(planner, state, timestamp, initial_accel=-0.4)
    assert not used_history
  used_recovery, _ = update_with_native_reference(planner, state, BASE_NS + 5 * STEP_NS, initial_accel=-0.6)
  used_next, consumed = update_with_native_reference(planner, state, BASE_NS + 6 * STEP_NS)
  assert not used_recovery
  assert used_next
  assert consumed[0] == pytest.approx(-0.7)


@pytest.mark.parametrize('interruption', ['disabled', 'gas_override', 'standstill', 'unhealthy', 'invalid_output', 'uninitialized'])
def test_interrupted_motion_cannot_supply_history_to_the_first_recovered_reference(interruption):
  planner, _ = make_planner()
  state = make_state()
  update_with_native_reference(planner, state, BASE_NS)
  update_with_native_reference(planner, state, BASE_NS + STEP_NS)
  if interruption == 'disabled':
    state['selfdriveState'].enabled = False
    state['carControl'].enabled = False
    state['carControl'].longActive = False
    state['controlsState'].longControlState = LongCtrlState.off
  elif interruption == 'gas_override':
    state['carState'].gasPressed = True
  elif interruption == 'standstill':
    state['carState'].standstill = True
  elif interruption == 'unhealthy':
    state.all_checks = lambda **_: False
  elif interruption == 'uninitialized':
    planner.reference_planner.reference_initialized = False
  if interruption == 'invalid_output':
    with patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.get_accel_from_plan', return_value=float('nan')):
      update_with_native_reference(planner, state, BASE_NS + 2 * STEP_NS)
    assert not planner.motion_intent.valid
  else:
    used_interrupted, _ = update_with_native_reference(planner, state, BASE_NS + 2 * STEP_NS, initial_accel=-0.4)
    assert not used_interrupted
  state = make_state()
  used_recovery, _ = update_with_native_reference(planner, state, BASE_NS + 3 * STEP_NS, initial_accel=-0.6)
  used_next, consumed = update_with_native_reference(planner, state, BASE_NS + 4 * STEP_NS)
  # A reset reference can start a new epoch after completing that same update;
  # a disabled, stopped or invalid result cannot seed the next active reference.
  assert used_recovery == (interruption == 'uninitialized')
  assert used_next
  assert consumed[0] == pytest.approx(-0.7)


def test_legacy_native_history_still_uses_the_fixed_model_step():
  results = []
  for offsets in ((0, STEP_NS, 2 * STEP_NS), (0, 3 * STEP_NS, 7 * STEP_NS)):
    template, extension = make_planner(platform=CAR.HONDA_CIVIC_BOSCH)
    planner = LongitudinalPlanner(template.CP, extension, init_v=20.0)
    state = make_state()
    state['controlsState'].motionState.planningMode = int(planner.longitudinal_mode)
    consumed_history = []
    requests = []
    for offset in offsets:
      _, consumed = update_with_native_reference(planner, state, BASE_NS + offset)
      consumed_history.append(consumed)
      requests.append(planner.output_a_target)
    results.append((np.array(consumed_history), np.array(requests)))
  np.testing.assert_array_equal(results[0][0], results[1][0])
  np.testing.assert_array_equal(results[0][1], results[1][1])
