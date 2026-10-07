"""SCC curve policy through actual DEC, native MPC, wire intent and owner.

The prescribed model and vehicle signals test command contracts. They are not
a closed-loop prediction of the vehicle's speed when it reaches a real curve.
"""
from contextlib import contextmanager

import numpy as np
import pytest

from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import consume, make_planner_owner, make_state
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanSource


@contextmanager
def curve_pipeline(*, speed=10., predicted_lateral=1.5, current_lateral=.5, experimental=False):
  with OpenpilotPrefix():
    params = Params()
    params.put_bool('SmartCruiseControlVision', True, block=True)
    # Exercise real DEC's disabled path, and actual is_e2e's mode decision.
    params.put_bool('DynamicExperimentalControl', False, block=True)
    planner, control = make_planner_owner(speed=speed)
    state = make_state(speed=speed, target=20., experimental=experimental)
    state['controlsState'].curvature = current_lateral / speed ** 2
    state['modelV2'].orientationRate.z = [predicted_lateral / speed] * len(ModelConstants.T_IDXS)
    yield planner, control, state


def step(planner, control, state, frame):
  state['controlsState'].longControlState = control.long_control_state
  return consume(planner, control, state, frame)


def test_comfortable_curve_holds_entry_without_an_added_deceleration_request():
  with curve_pipeline() as (planner, control, state):
    for frame in range(12):
      plan, outputs = step(planner, control, state, frame)
    assert planner.source == LongitudinalPlanSource.sccVision
    assert planner.scc.vision.entry_speed == 10.
    assert planner.cruise_target.speed == pytest.approx(10.)
    assert plan.motionIntent.valid and plan.solverStatus == 0
    assert plan.aTarget > -.05
    assert min(outputs) > -.05
    assert not plan.shouldStop


def test_m8_predicted_curve_still_sends_required_deceleration_to_owner():
  with curve_pipeline(speed=25.1 / 3.6, predicted_lateral=2.25, current_lateral=1.17) as (planner, control, state):
    for frame in range(12):
      plan, outputs = step(planner, control, state, frame)
      if planner.scc.vision.is_active:
        assert plan.aTarget <= planner.scc.vision.output_a_target + 1e-6
    assert planner.cruise_target.speed * 3.6 > 23.
    assert planner.cruise_target.speed < state['carState'].vEgo
    assert plan.solverStatus == 0 and plan.motionIntent.valid
    assert plan.aTarget < -.5
    assert max(outputs) < 0.
    assert not plan.shouldStop


def test_stronger_model_braking_wins_over_comfortable_curve_hold():
  with curve_pipeline(experimental=True) as (planner, control, state):
    for frame in range(8):
      step(planner, control, state, frame)
    state['modelV2'].action.desiredAcceleration = -1.5
    for frame in range(8, 15):
      plan, outputs = step(planner, control, state, frame)
      assert planner.reference_planner.e2e_allowed
      assert plan.aTarget <= -1.5
      assert max(outputs) <= 0.


def test_sharp_predicted_curve_keeps_physical_mpc_acceleration_bounds_ordered():
  with curve_pipeline(speed=30., predicted_lateral=8., current_lateral=1.) as (planner, control, state):
    for frame in range(12):
      plan, outputs = step(planner, control, state, frame)
    assert planner.scc.vision.output_a_target < -3.5
    assert plan.solverStatus == 0 and plan.motionIntent.valid
    assert -3.5 <= plan.aTarget < 0.
    assert all(-3.5 <= output <= 0. for output in outputs)


@pytest.mark.parametrize('field', ['prediction', 'current'])
@pytest.mark.parametrize('invalid', [float('nan'), float('inf')])
def test_active_invalid_curve_reaches_planner_rejection_instead_of_cruise(field, invalid):
  with curve_pipeline() as (planner, control, state):
    for frame in range(8):
      step(planner, control, state, frame)
    assert planner.scc.vision.is_active
    if field == 'prediction':
      state['modelV2'].orientationRate.z = [invalid] * len(ModelConstants.T_IDXS)
    else:
      state['controlsState'].curvature = invalid
    with np.errstate(invalid='ignore'):
      plan, outputs = step(planner, control, state, 8)
    if field == 'current':
      assert planner.source == LongitudinalPlanSource.sccVision
      assert not np.isfinite(planner.cruise_target.speed)
    # Invalid model trajectories are rejected earlier, before policy update;
    # their previous finite curve target is not a new actuation permission.
    assert not plan.motionIntent.valid
    assert max(outputs) <= 0.
