"""A native motion reference must not become its opposite at the actuator."""

from unittest.mock import patch

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import advance, make_planner, make_state
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params


def solve_given_acceleration(planner, acceleration, speed):
  """Install a dynamically consistent native trajectory with changing jerk."""
  acceleration = np.asarray(acceleration, dtype=float)
  planner.mpc.a_solution = acceleration
  planner.mpc.v_solution = np.r_[speed, speed + np.cumsum(
    0.5 * (acceleration[:-1] + acceleration[1:]) * np.diff(T_IDXS))]
  planner.mpc.j_solution = np.diff(acceleration) / np.diff(T_IDXS)
  planner.mpc.source = log.LongitudinalPlan.LongitudinalPlanSource.cruise
  planner.mpc.solution_status = 0


@pytest.mark.parametrize('direction', [-1.0, 1.0])
def test_nonconstant_jerk_cannot_reverse_a_uniform_acceleration_request(direction):
  # Low-speed road oscillations exposed this shape: a large measured response
  # decays within the action horizon, while the requested future acceleration
  # keeps its sign. The old single-ramp inverse asks for the opposite effort.
  acceleration = direction * (0.1 + 1.2 * np.exp(-T_IDXS / 0.18))
  planner, _ = make_planner(speed=5.0, initial_accel=float(acceleration[0]))
  state = make_state(speed=5.0, target=6.0)
  state['carState'].aEgo = float(acceleration[0])
  with patch.object(planner.mpc, 'update', side_effect=lambda *_args, **_kwargs:
                    solve_given_acceleration(planner, acceleration, 5.0)):
    planner.update(state, now_ns=advance(state, 0))
  assert planner.motion_intent.valid
  assert direction * planner.reference_planner.reference_accel > 0.0
  assert direction * planner.output_a_target > 0.0
  # It also remains inside the native trajectory's acceleration envelope.
  assert min(acceleration) <= planner.reference_planner.reference_accel <= max(acceleration)


def test_constant_jerk_retains_the_analytic_action_time_acceleration():
  planner, _ = make_planner(speed=8.0, initial_accel=-0.2)
  state = make_state(speed=8.0, target=9.0)
  state['carState'].aEgo = -0.2
  jerk = 0.08
  acceleration = -0.2 + jerk * T_IDXS
  with patch.object(planner.mpc, 'update', side_effect=lambda *_args, **_kwargs:
                    solve_given_acceleration(planner, acceleration, 8.0)):
    planner.update(state, now_ns=advance(state, 0))
  assert planner.reference_planner.reference_accel == pytest.approx(
    -0.2 + jerk * (planner.CP.longitudinalActuatorDelay + DT_MDL))


def test_direct_action_sampling_preserves_the_independent_model_braking_envelope():
  planner, _ = make_planner(speed=5.0)
  state = make_state(speed=5.0, target=8.0, experimental=True)
  state['modelV2'].action.desiredAcceleration = -1.8
  with patch.object(planner.mpc, 'update', side_effect=lambda *_args, **_kwargs:
                    solve_given_acceleration(planner, np.full_like(T_IDXS, 0.4), 5.0)):
    planner.update(state, now_ns=advance(state, 0))
  assert planner.reference_planner.reference_accel == pytest.approx(0.4)
  assert planner.reference_planner.constraint_accel == pytest.approx(-1.8)
  assert planner.output_a_target == pytest.approx(-1.8)
  assert planner.reference_planner.reference_overridden


def test_sustained_native_braking_is_not_filtered_or_rate_limited_by_sampling():
  planner, _ = make_planner(speed=30.0, initial_accel=-2.5)
  state = make_state(speed=30.0, target=20.0)
  state['carState'].aEgo = -2.5
  with patch.object(planner.mpc, 'update', side_effect=lambda *_args, **_kwargs:
                    solve_given_acceleration(planner, np.full_like(T_IDXS, -2.5), 30.0)):
    planner.update(state, now_ns=advance(state, 0))
  assert planner.output_a_target == pytest.approx(-2.5)
