"""ACC cap admission through the real planner wire and LongControl owner.

The native solve still runs, then its acceleration trajectory is substituted
to isolate cap crossings. These are response-contract counterexamples, not a
claim that the substituted trajectory is safe for an obstacle or a road test.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import STARTING_ACCEL_RATE
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionPhase, MotionResponseKind
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import consume, make_planner_owner, make_state
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_unified_action_sampling import solve_given_acceleration
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import CruiseTarget, LongitudinalPlannerSP, LongitudinalPlanSource as PolicySource


@contextmanager
def controlled_acc(*, speed=3.0, initial_accel=0.0):
  planner, control = make_planner_owner(speed=speed, initial_accel=initial_accel)
  state = make_state(speed=speed, target=speed + 2.0)
  run = SimpleNamespace(planner=planner, control=control, state=state, frame=0,
                        acceleration=0.3, policy_source=PolicySource.cruise)

  def targets(instance, state, speed, accel, cruise):
    instance.source = run.policy_source
    if run.policy_source != PolicySource.cruise:
      cruise, accel = min(cruise, speed - 0.5), min(accel, -0.2)
    instance.cruise_target = CruiseTarget(instance.source, cruise, accel)
    return instance.cruise_target

  def update(*args, **kwargs):
    solve_given_acceleration(planner, np.full_like(T_IDXS, run.acceleration), speed, *args, **kwargs)

  def step(reference=None, measured=None):
    if reference is not None:
      run.acceleration = reference
    if measured is not None:
      state['carState'].aEgo = measured
    state['controlsState'].longControlState = control.long_control_state
    result = consume(planner, control, state, run.frame)
    run.frame += 1
    return result

  run.step = step
  with patch.object(LongitudinalPlannerSP, 'update', return_value=None), \
       patch.object(LongitudinalPlannerSP, 'update_targets', new=targets), \
       patch.object(planner, 'is_e2e', side_effect=lambda _: state['selfdriveState'].experimentalMode), \
       patch.object(planner.mpc, 'update', side_effect=update):
    yield run


def learn_negative_response(run):
  for _ in range(6):
    plan, _ = run.step(reference=0.3, measured=0.8)
  assert plan.motionIntent.responseKind == MotionResponseKind.PURE_CRUISE
  assert run.control.pid.i < -0.03
  return run.planner.reference_planner.constraint_accel


@pytest.mark.parametrize('crossing', [5e-5, 0.3])
def test_cruise_cap_preserves_negative_response_for_small_and_large_crossings(crossing):
  with controlled_acc() as run:
    cap = learn_negative_response(run)
    for _ in range(4):
      before, before_outputs = run.step(reference=cap - crossing, measured=cap - crossing)
    assert not before.reference.overridden
    assert before.motionIntent.responseKind == MotionResponseKind.PURE_CRUISE
    negative_i = run.control.pid.i
    assert negative_i < -0.03

    plan, outputs = run.step(reference=cap + crossing, measured=cap)
    assert plan.reference.overridden
    assert run.planner.reference_planner.reference_accel == pytest.approx(cap + crossing)
    assert plan.reference.constraintAccel == pytest.approx(cap)
    assert plan.aTarget == pytest.approx(cap)
    assert plan.motionIntent.responseKind == MotionResponseKind.ABSOLUTE_ACC
    assert run.control.motion_phase == MotionPhase.TRACK
    assert run.control.pid.i == pytest.approx(negative_i, abs=1e-7)
    assert outputs[-1] == pytest.approx(plan.aTarget + negative_i, abs=1e-7)
    assert outputs[-1] - before_outputs[-1] == pytest.approx(crossing, abs=1e-7)
    assert all(ACCEL_MIN <= output <= min(plan.aTarget, ACCEL_MAX) for output in outputs)
    assert np.max(np.diff([before_outputs[-1], *outputs])) <= STARTING_ACCEL_RATE * DT_CTRL + 1e-7


def test_cap_projection_withdraws_positive_i_on_the_first_control_tick():
  with controlled_acc() as run:
    run.step()
    cap = run.planner.reference_planner.constraint_accel
    for _ in range(8):
      before, before_outputs = run.step(reference=cap - 5e-5, measured=cap - 0.5)
    assert before.motionIntent.responseKind == MotionResponseKind.PURE_CRUISE
    assert not before.reference.overridden
    assert run.control.pid.i > 0.03
    assert before_outputs[-1] > cap

    plan, outputs = run.step(reference=cap + 5e-5, measured=cap)
    assert plan.reference.overridden
    assert plan.motionIntent.responseKind == MotionResponseKind.ABSOLUTE_ACC
    assert plan.aTarget == pytest.approx(cap)
    assert run.control.pid.i == 0.0
    assert outputs == pytest.approx([plan.aTarget] * 5, abs=1e-7)


@pytest.mark.parametrize('constraint', ['policy', 'model'])
def test_independent_constraint_cannot_inherit_capped_acc_response(constraint):
  with controlled_acc() as run:
    cap = learn_negative_response(run)
    for _ in range(4):
      before, _ = run.step(reference=cap + 5e-5, measured=cap)
    assert before.motionIntent.responseKind == MotionResponseKind.ABSOLUTE_ACC
    assert run.control.pid.i < -0.03

    if constraint == 'policy':
      run.policy_source = PolicySource.sccVision
    else:
      run.state['selfdriveState'].experimentalMode = True
      run.state['modelV2'].action.desiredAcceleration = -0.3
    plan, outputs = run.step()
    assert plan.reference.overridden
    assert plan.motionIntent.responseKind == MotionResponseKind.NONE
    assert plan.motionIntent.valid and not plan.motionIntent.reasons
    assert run.control.pid.i == 0.0
    if constraint == 'policy':
      assert run.planner.cruise_target.speed < run.state['carState'].vEgo
      assert plan.aTarget == pytest.approx(cap)
    else:
      assert run.planner.reference_planner.constraint_source == LongitudinalPlanSource.e2e
      assert plan.aTarget == pytest.approx(-0.3)
    assert outputs == pytest.approx([plan.aTarget] * 5, abs=1e-7)


def test_global_lower_clip_cannot_masquerade_as_an_acc_upper_cap():
  # At 40 m/s both the native -3.5 and diagnostic -3.75 trajectories remain
  # moving over the complete horizon, avoiding an unrelated stop boundary.
  with controlled_acc(speed=40.0, initial_accel=ACCEL_MIN) as run, \
       patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.cruise_accel_limit', return_value=ACCEL_MIN):
    for _ in range(2):
      plan, outputs = run.step(reference=ACCEL_MIN - 0.25, measured=ACCEL_MIN)
    reference = run.planner.reference_planner
    assert plan.solverStatus == 0
    assert plan.motionIntent.valid and not plan.motionIntent.reasons
    assert run.control.motion_phase == MotionPhase.TRACK
    assert reference.constraint_source == LongitudinalPlanSource.cruise
    assert reference.reference_accel < reference.constraint_accel == ACCEL_MIN
    assert plan.reference.overridden
    assert plan.aTarget == ACCEL_MIN
    assert plan.motionIntent.responseKind == MotionResponseKind.NONE
    assert run.control.pid.i == 0.0
    assert outputs == [ACCEL_MIN] * 5
