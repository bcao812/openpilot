"""ACC cap admission through the real planner wire and LongControl owner.

The native solve still runs, then its acceleration trajectory is substituted
to isolate cap crossings. These are response-contract counterexamples, not a
claim that the substituted trajectory is safe for an obstacle or a road test.
Ordinary cruise has an external goal; its finite native continuation cannot
qualify signed cruise feedback merely by carrying the cruise source label.
"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.cruise_response import A_CRUISE_MAX_BP, J_CRUISE_VALS
from openpilot.selfdrive.controls.lib.longcontrol import STARTING_ACCEL_RATE
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanSource, get_max_accel
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
                        acceleration=0.3, policy_source=PolicySource.cruise, policy_speed=None, native_reference=None)

  def targets(instance, state, speed, accel, cruise):
    instance.source = run.policy_source
    if run.policy_source != PolicySource.cruise:
      policy_speed = speed - 0.5 if run.policy_speed is None else run.policy_speed
      cruise, accel = min(cruise, policy_speed), min(accel, -0.2)
    instance.cruise_target = CruiseTarget(instance.source, cruise, accel)
    return instance.cruise_target

  def update(*args, **kwargs):
    run.native_reference = kwargs['reference']
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


def settle_independent_hard_cap(run):
  """Reach the hard cap through real plans before testing a cap crossing.

  The fixture is straight, allows throttle and requests more acceleration
  than the hard maximum. Starting with measured a=0, the existing cruise jerk
  bound gives a finite settling budget; it is not a trial-and-error sleep or
  a write to CruiseResponse history. Zero tracking error keeps I unseeded.
  """
  assert run.frame == 0
  speed = float(run.state['carState'].vEgo)
  hard_cap = float(get_max_accel(speed))
  assert run.state['carState'].vCruise / 3.6 - speed > hard_cap
  jerk = float(np.interp(speed, A_CRUISE_MAX_BP, J_CRUISE_VALS))
  ramp_frames = int(np.ceil(hard_cap / (jerk * run.planner.dt)))
  limits = []
  for _ in range(ramp_frames + 1):
    plan, _ = run.step(reference=0.0, measured=0.0)
    limits.append(float(plan.reference.constraintAccel))
  np.testing.assert_allclose(limits[-2:], hard_cap, atol=1e-7, rtol=0.0)

  # Restore the original 0.3 baseline using the public control path, so the
  # response-history tests start from this known command rather than zero.
  plan, outputs = run.step(reference=0.3, measured=0.3)
  assert plan.reference.constraintAccel == pytest.approx(hard_cap, abs=1e-7)
  assert outputs[-1] == pytest.approx(0.3, abs=1e-7)
  assert run.control.pid.i == 0.0
  return hard_cap


def learn_negative_response(run):
  cap = settle_independent_hard_cap(run)
  for _ in range(6):
    plan, _ = run.step(reference=0.3, measured=0.8)
    assert run.native_reference.cruise_speed is None
  assert plan.reference.constraintAccel == pytest.approx(cap, abs=1e-7)
  assert plan.motionIntent.responseKind == MotionResponseKind.ABSOLUTE_ACC
  assert run.control.pid.i < -0.03
  return cap


@pytest.mark.parametrize('crossing', [5e-5, 0.3])
def test_cruise_cap_preserves_negative_response_for_small_and_large_crossings(crossing):
  with controlled_acc() as run:
    cap = learn_negative_response(run)
    for _ in range(4):
      before, before_outputs = run.step(reference=cap - crossing, measured=cap - crossing)
    assert not before.reference.overridden
    assert before.motionIntent.responseKind == MotionResponseKind.ABSOLUTE_ACC
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


@pytest.mark.parametrize('crossing', [1.6e-5, 5e-5])
def test_finite_continuation_cannot_learn_positive_i_or_drop_it_at_a_hard_cap_crossing(crossing):
  with controlled_acc() as run:
    cap = settle_independent_hard_cap(run)
    previous = run.control.last_output_accel
    rows = []
    # The first delta comes from the observed reentry counterexample; the
    # second retains the existing small-crossing input. Neither changes the
    # sampler or injects a numeric goal that production would not install.
    for _ in range(2):
      for sample, measured in [(cap - crossing, cap - 0.5)] * 8 + [(cap + crossing, cap)]:
        plan, outputs = run.step(reference=sample, measured=measured)
        assert run.native_reference.cruise_speed is None
        assert plan.solverStatus == 0 and plan.motionIntent.valid and not plan.motionIntent.reasons
        assert run.control.motion_phase == MotionPhase.TRACK
        assert plan.reference.accel == float(np.float32(sample))
        assert plan.reference.constraintAccel == pytest.approx(cap, abs=1e-7)
        assert plan.aTarget == min(plan.reference.accel, plan.reference.constraintAccel)
        assert plan.reference.overridden == (sample > cap)
        rows.append((plan, outputs, run.control.pid.i))
    # Capture the complete boundary history before asserting permission. In
    # the regression, the cruise label learns positive I below the cap and
    # ABS discards it on the very first tick above the cap.
    assert [integral for _, _, integral in rows] == [0.0] * len(rows)
    assert all(plan.motionIntent.responseKind == MotionResponseKind.ABSOLUTE_ACC for plan, _, _ in rows)
    for plan, outputs, _ in rows:
      assert all(ACCEL_MIN <= output <= min(plan.aTarget, ACCEL_MAX) for output in outputs)
      assert np.max(np.diff([previous, *outputs])) <= STARTING_ACCEL_RATE * DT_CTRL + 1e-7
      previous = outputs[-1]
    for index in (8, 17):
      before, before_outputs, _ = rows[index - 1]
      plan, outputs, _ = rows[index]
      assert not before.reference.overridden and plan.reference.overridden
      assert plan.aTarget == pytest.approx(cap)
      assert outputs == pytest.approx([plan.aTarget] * 5, abs=1e-7)
      assert outputs[0] - before_outputs[-1] == pytest.approx(crossing, abs=1e-7)


@pytest.mark.parametrize('constraint', ['policy', 'model'])
def test_policy_override_and_ordinary_model_cap_keep_their_declared_response_domains(constraint):
  with controlled_acc() as run:
    cap = learn_negative_response(run)
    for _ in range(4):
      before, _ = run.step(reference=cap + 5e-5, measured=cap)
    assert before.motionIntent.responseKind == MotionResponseKind.ABSOLUTE_ACC
    assert run.control.pid.i < -0.03

    if constraint == 'policy':
      # Isolate withdrawal of response permission when SCC becomes the source:
      # lower cruise while its speed error still exceeds the independent hard
      # max. Negative speed error and real ENTERING/TURNING/LEAVING cycles are
      # covered by the SCC integration tests and frozen common scenarios.
      previous_cruise = float(run.planner.cruise_target.speed)
      run.policy_speed = min(previous_cruise, float(run.state['carState'].vEgo) + cap + 0.5)
      run.policy_source = PolicySource.sccVision
    else:
      run.state['selfdriveState'].experimentalMode = True
      run.state['modelV2'].action.desiredAcceleration = -0.3
    plan, outputs = run.step()
    assert plan.reference.overridden
    assert plan.motionIntent.valid and not plan.motionIntent.reasons
    if constraint == 'policy':
      assert plan.motionIntent.responseKind == MotionResponseKind.NONE
      assert run.control.pid.i == 0.0
      assert run.state['carState'].vEgo + cap < run.planner.cruise_target.speed < previous_cruise
      assert plan.aTarget == pytest.approx(cap)
      assert outputs == pytest.approx([plan.aTarget] * 5, abs=1e-7)
    else:
      # An ordinary model action caps the same absolute target; its raw
      # moving description must not reset braking I into a MODEL domain.
      assert plan.motionIntent.responseKind == MotionResponseKind.ABSOLUTE_ACC
      assert run.control.pid.i < 0.0
      assert run.planner.reference_planner.constraint_source == LongitudinalPlanSource.e2e
      assert plan.aTarget == pytest.approx(-0.3)
      assert all(output <= plan.aTarget for output in outputs)


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
