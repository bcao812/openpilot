"""Rest admission must not claim a stop obtained with undeliverable acceleration.

The three inputs below are captured from the second approach in the frozen
complete_stop_hold_go_stop/d03_t04 engineering plant, candidate d15cddc3.
Only measured MPC state, wire-quantized raw velocity and source times are
reproduced. Native warm-start state before this window was not recorded, so
these are cold/short-sequence counterexamples, not a full route replay or a
claim about delivered braking on the car.
"""

from dataclasses import asdict, replace

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  MPC_GOAL_INFEASIBLE, QP_FEASIBILITY_TOL, REST_STATE_TOL, T_IDXS, LongitudinalMpc, MpcOutcome,
)
from openpilot.selfdrive.controls.lib.model_reference import ModelReferenceAdapter
from openpilot.selfdrive.controls.tests.test_future_rest_scheduling import assert_actual_scheduled_stop
from openpilot.selfdrive.controls.tests.test_typed_model_reference import MODEL_TIMES, evidence, stopping_curve
from openpilot.selfdrive.controls.tests.test_typed_reference_mpc import typed_reference
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state


# trace.jsonl lines 2711, 2716 and 2721 in
# _logs/longitudinal-fidelity-2026-09-20/f-closed-v4/complete_stop_hold_go_stop--d03_t04/.
# All omitted raw source-grid velocities are exactly zero. Their timestamps
# are fresh and the producer consumes each model 1000 ns after its state anchor.
APPROACH = (
  (28_100_000_000, 1.1646994352340698, -2.313807249069214,
   (1.186125649864291, 1.14163855489456, 1.008177269985367, 0.7857417951367124,
    0.47433213034859567, 0.07394827562101725)),
  (28_150_000_000, 1.0482574701309204, -2.3191792964935303,
   (1.0717342666829761, 1.0144929552145592, 0.8427690208093079, 0.5565624634672227,
    0.1558732831883034)),
  (28_200_000_000, 0.9341553449630737, -2.3036110401153564,
   (0.9596800556317722, 0.8644183844373359, 0.5786333708540271, 0.10232501488184575)),
)


def captured_reference(sample, *, reschedule, adapter=None):
  source_ns, speed, _, prefix = sample
  raw = np.zeros(len(MODEL_TIMES), dtype=np.float32)
  raw[:len(prefix)] = prefix
  raw = raw.astype(float)  # The real model publication carries Float32 values.
  typed = evidence(raw, speed, adapter=adapter, model_ns=source_ns, anchor_ns=source_ns,
                   now_ns=source_ns + 1000)
  return replace(typed_reference(raw, speed, cruise=5.0, typed=typed),
                 allow_future_rest_reschedule=reschedule, comfort_stop_distance=4.0)


def assert_acceleration_deliverable(mpc):
  # Existing physical envelope and numerical residual allowance; no new tuning
  # threshold. Constant jerk makes endpoint acceleration bounds sufficient on
  # each interval, while existing scheduled-rest checks cover velocity extrema.
  assert np.all(mpc.a_solution >= mpc.params[:, 0] - QP_FEASIBILITY_TOL)
  assert np.all(mpc.a_solution <= mpc.params[:, 1] + QP_FEASIBILITY_TOL)


@pytest.mark.parametrize('reschedule', [False, True])
def test_captured_node_two_rest_cannot_use_acceleration_beyond_the_actuator_envelope(reschedule):
  sample = APPROACH[-1]
  _, speed, acceleration, _ = sample
  reference = captured_reference(sample, reschedule=reschedule)
  goal = reference.typed_model.rest_goal
  source_goal = asdict(goal)
  assert goal.native_node == 2

  # The previous continuous-time lower bound passes. Independently integrate
  # the two native constant-jerk intervals with the strongest allowed a1 and
  # the lowest allowed rest a2: even that leaves positive v2 outside the box.
  h0, h1 = np.diff(T_IDXS[:3])
  assert speed + ACCEL_MIN * goal.native_time_s <= REST_STATE_TOL
  minimum_terminal_speed = speed + 0.5 * (acceleration + ACCEL_MIN) * h0
  minimum_terminal_speed += 0.5 * (ACCEL_MIN - REST_STATE_TOL) * h1
  assert minimum_terminal_speed > REST_STATE_TOL

  mpc = LongitudinalMpc()
  mpc.set_weights(prev_accel_constraint=False)
  mpc.set_cur_state(speed, acceleration)
  mpc.update(radar_state(), reference=reference)

  assert mpc.diagnostic_solve_calls == 1 and mpc.diagnostic_last_solver_return == 0
  assert mpc.rest_goal is goal and asdict(goal) == source_goal
  np.testing.assert_array_equal(mpc.x0, [0.0, speed, acceleration])
  np.testing.assert_array_equal(mpc.params[:, 0], ACCEL_MIN)
  np.testing.assert_array_equal(mpc.params[:, 1], ACCEL_MAX)
  assert mpc.rest_goal_state == 'infeasible'
  if reschedule:
    # Reuse the already-supported first later physically witnessed node. The
    # source event and its nominal position must not be rewritten as success.
    assert_actual_scheduled_stop(mpc)
    assert_acceleration_deliverable(mpc)
  else:
    assert mpc.effective_rest_node is None
    assert mpc.solution_status == MPC_GOAL_INFEASIBLE
    assert mpc.outcome is MpcOutcome.REST_REJECTED
    assert mpc.rest_goal_braking_accel == ACCEL_MIN


def test_short_captured_history_never_accepts_rest_via_undeliverable_braking():
  mpc = LongitudinalMpc()
  adapter = ModelReferenceAdapter()
  accepted_envelopes = []
  for index, sample in enumerate(APPROACH):
    _, speed, acceleration, _ = sample
    reference = captured_reference(sample, reschedule=True, adapter=adapter)
    # Construct an explicit cold initial state, then retain the actual previous
    # solve's history for each subsequent 50 ms update. No private history is
    # copied from the candidate's currently observed output.
    mpc.set_weights(prev_accel_constraint=index > 0)
    mpc.set_cur_state(speed, acceleration)
    calls = mpc.diagnostic_solve_calls
    mpc.update(radar_state(), reference=reference)
    assert mpc.diagnostic_solve_calls == calls + 1
    assert mpc.diagnostic_last_solver_return == 0
    assert mpc.rest_goal is reference.typed_model.rest_goal
    if mpc.outcome is MpcOutcome.ACCEPTED:
      accepted_envelopes.append((mpc.a_solution.copy(), mpc.params[:, :2].copy()))
    else:
      assert mpc.outcome is MpcOutcome.REST_REJECTED
      assert mpc.rest_goal_braking_accel == ACCEL_MIN

  assert accepted_envelopes  # This is not a blanket rejection of the approach.
  for acceleration, bounds in accepted_envelopes:
    assert np.all(acceleration >= bounds[:, 0] - QP_FEASIBILITY_TOL)
    assert np.all(acceleration <= bounds[:, 1] + QP_FEASIBILITY_TOL)
  assert mpc.rest_goal_state == 'infeasible'
  assert_actual_scheduled_stop(mpc)


def test_reachable_short_rest_keeps_its_original_goal_and_native_result():
  speed, acceleration = 0.2, -0.5
  original = typed_reference(stopping_curve(speed, 0.15), speed, cruise=5.0)
  assert original.typed_model.rest_goal.native_node == 2
  results = []
  for reschedule in (False, True):
    reference = replace(original, allow_future_rest_reschedule=reschedule)
    mpc = LongitudinalMpc()
    mpc.set_weights(prev_accel_constraint=False)
    mpc.set_cur_state(speed, acceleration)
    mpc.update(radar_state(), reference=reference)
    assert mpc.diagnostic_solve_calls == 1 and mpc.diagnostic_last_solver_return == 0
    assert mpc.outcome is MpcOutcome.ACCEPTED and mpc.rest_goal_state == 'satisfied'
    assert mpc.effective_rest_node is None
    assert mpc.rest_goal is original.typed_model.rest_goal
    assert_acceleration_deliverable(mpc)
    results.append(mpc)
  for name in ('x0', 'x_sol', 'u_sol', 'params', 'yref', 'diagnostic_reference_weights'):
    np.testing.assert_array_equal(getattr(results[0], name), getattr(results[1], name))
  for before, after in zip(results[0].diagnostic_reference_bounds, results[1].diagnostic_reference_bounds, strict=True):
    np.testing.assert_array_equal(before, after)
