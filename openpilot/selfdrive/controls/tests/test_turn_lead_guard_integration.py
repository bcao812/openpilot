"""Native planner -> wire intent -> owner checks for lost leads in a turn.

These are request-level contracts with prescribed vehicle measurements, not
closed-loop stopping-distance or road-performance claims. DEC is unchanged;
the recorded blended/ACC admission is supplied explicitly to this layer.
"""
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionPhase, MotionResponseKind
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import consume, make_planner_owner, make_state
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import CruiseTarget, LongitudinalPlannerSP, LongitudinalPlanSource


def cruise_target(_planner, _state, speed, accel, target):
  return CruiseTarget(LongitudinalPlanSource.cruise, target, accel)


def set_lead(state, present, *, probability=0.99):
  lead = state['radarState'].leadOne
  lead.present = present
  lead.dRel = 20.0
  lead.yRel = 0.0
  lead.vRel = 0.0
  lead.vLead = state['carState'].vEgo
  lead.aLeadK = 0.0
  lead.modelProb = probability


@contextmanager
def pipeline(*, experimental=True, measured_accel=0.4):
  with OpenpilotPrefix():
    planner, control = make_planner_owner(speed=3.0, initial_accel=measured_accel)
    state = make_state(speed=3.0, target=12.0, experimental=experimental)
    state['carState'].aEgo = measured_accel
    state['carState'].steeringAngleDeg = 250.0
    state['modelV2'].action.desiredAcceleration = 1.1
    with patch.object(planner, 'is_e2e', return_value=experimental), \
         patch.object(LongitudinalPlannerSP, 'update', return_value=None), \
         patch.object(LongitudinalPlannerSP, 'update_targets', new=cruise_target):
      yield planner, control, state


def step(planner, control, state, frame):
  state['controlsState'].longControlState = control.long_control_state
  plan, outputs = consume(planner, control, state, frame)
  assert plan.solverStatus == 0
  assert plan.motionIntent.valid
  assert not plan.shouldStop
  assert control.motion_phase == MotionPhase.TRACK
  return plan, outputs


@pytest.mark.parametrize('experimental', [False, True])
@pytest.mark.parametrize('initial_i', [-0.3, 0.3])
def test_missing_lead_ceiling_reaches_controller_without_positive_integrator(experimental, initial_i):
  with pipeline(experimental=experimental) as (planner, control, state):
    set_lead(state, True)
    for frame in range(8):
      step(planner, control, state, frame)
    set_lead(state, False)
    control.pid.i = initial_i
    for frame in range(8, 15):
      plan, outputs = step(planner, control, state, frame)
      assert planner.reference_planner.turn_lead_guard.lost
      assert planner.mpc.x0[2] == pytest.approx(0.4)
      assert plan.aTarget <= 0.0
      assert plan.reference.constraintAccel <= 0.0
      if plan.aTarget == 0.0:
        assert plan.motionIntent.responseKind == MotionResponseKind.ABSOLUTE_ACC
        assert not plan.motionIntent.releaseAllowed
      assert max(outputs) <= 1e-7
      assert control.pid.i <= 0.0


def test_empty_turn_without_previous_lead_retains_normal_acceleration():
  with pipeline() as (planner, control, state):
    for frame in range(12):
      plan, outputs = step(planner, control, state, frame)
      assert not planner.reference_planner.turn_lead_guard.armed
    assert plan.aTarget > 0.2
    assert max(outputs) > 0.1


def test_stronger_model_braking_is_not_replaced_by_zero_ceiling():
  with pipeline() as (planner, control, state):
    set_lead(state, True)
    for frame in range(8):
      step(planner, control, state, frame)
    set_lead(state, False)
    state['modelV2'].action.desiredAcceleration = -1.5
    for frame in range(8, 14):
      plan, outputs = step(planner, control, state, frame)
      assert planner.reference_planner.turn_lead_guard.lost
      assert plan.aTarget <= -1.5
      assert max(outputs) <= 0.0


def test_short_loss_and_lead_flicker_cannot_release_full_acceleration():
  with pipeline(measured_accel=0.0) as (planner, control, state):
    set_lead(state, True)
    for frame in range(10):
      step(planner, control, state, frame)
    set_lead(state, False)
    plan, _ = step(planner, control, state, 10)
    assert plan.aTarget <= 0.0
    set_lead(state, True)
    for frame in range(11, 20):
      plan, outputs = step(planner, control, state, frame)
      ceiling = planner.reference_planner.turn_lead_guard.accel_ceiling
      assert ceiling is not None
      assert ceiling <= max(0.0, (frame - 15) * 0.05) + 1e-9
      assert plan.aTarget <= ceiling + 1e-7
      assert max(outputs) <= ceiling + 1e-7
    set_lead(state, False)
    plan, outputs = step(planner, control, state, 20)
    assert plan.aTarget <= 0.0
    assert max(outputs) <= 1e-7


def test_manual_takeover_revokes_guard_history():
  with pipeline() as (planner, control, state):
    set_lead(state, True)
    for frame in range(8):
      step(planner, control, state, frame)
    set_lead(state, False)
    step(planner, control, state, 8)
    assert planner.reference_planner.turn_lead_guard.lost
    state['carState'].gasPressed = True
    consume(planner, control, state, 9)
    assert not planner.reference_planner.turn_lead_guard.armed
    assert planner.reference_planner.turn_lead_guard.accel_ceiling is None
