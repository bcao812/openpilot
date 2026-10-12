"""Current-input planner -> wire intent -> owner checks after turn lead loss.

These are request-level contracts with prescribed vehicle measurements, not
closed-loop stopping-distance or road-performance claims. DEC is unchanged;
the recorded blended/ACC admission is supplied explicitly to this layer.
"""
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import CRASH_DISTANCE
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionPhase
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
def pipeline(*, experimental=True, measured_accel=0.4, speed=3.0):
  with OpenpilotPrefix():
    planner, control = make_planner_owner(speed=speed, initial_accel=measured_accel)
    state = make_state(speed=speed, target=12.0, experimental=experimental)
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
@pytest.mark.parametrize('speed', [0.8, 3.0])
def test_current_empty_turn_has_no_old_lead_speed_or_acceleration_ceiling(experimental, speed):
  with pipeline(experimental=experimental, speed=speed, measured_accel=0.0) as (planner, control, state):
    set_lead(state, True)
    for frame in range(8):
      step(planner, control, state, frame)
    set_lead(state, False)
    for frame in range(8, 32):
      plan, outputs = step(planner, control, state, frame)
      assert plan.reference.constraintAccel > 0.0
      assert not planner.mpc._source_leads
      assert not plan.motionIntent.reasons
    # Normal current cruise/model inputs can recover speed; no old target
    # must return and no straight-road confirmation is needed.
    assert plan.aTarget > 0.0 and max(outputs) > 0.0


def test_empty_turn_without_previous_lead_retains_normal_acceleration():
  with pipeline() as (planner, control, state):
    for frame in range(12):
      plan, outputs = step(planner, control, state, frame)
    assert plan.aTarget > 0.2
    assert max(outputs) > 0.1


def test_current_model_braking_still_wins_after_old_lead_disappears():
  with pipeline() as (planner, control, state):
    set_lead(state, True)
    for frame in range(8):
      step(planner, control, state, frame)
    set_lead(state, False)
    state['modelV2'].action.desiredAcceleration = -1.5
    for frame in range(8, 14):
      plan, outputs = step(planner, control, state, frame)
      assert plan.aTarget <= -1.5
      assert max(outputs) <= 0.0
      assert not planner.mpc._source_leads


def test_a_new_current_lead_brakes_after_an_empty_turn():
  with pipeline(measured_accel=0.0) as (planner, control, state):
    set_lead(state, True)
    for frame in range(8):
      step(planner, control, state, frame)
    set_lead(state, False)
    for frame in range(8, 18):
      step(planner, control, state, frame)
    set_lead(state, True)
    lead = state['radarState'].leadOne
    lead.dRel, lead.vRel, lead.vLead = 5.0, -3.0, 0.0
    for frame in range(18, 23):
      plan, outputs = consume(planner, control, state, frame)
      assert plan.solverStatus == 0 and plan.motionIntent.valid
      assert plan.aTarget < 0.0 and max(outputs) <= 0.0
      # Check the newly solved request trajectory against current geometry;
      # a zero native status alone does not exclude a predicted collision.
      mpc = planner.mpc
      assert mpc.lead_clearance_minimum(mpc._lead_trajectories[0], mpc.x_sol, mpc.j_solution) > CRASH_DISTANCE


@pytest.mark.parametrize('speed,distance,lead_speed', [
  (3.0, 40.0, 0.0),  # Distant stationary target outside the history-cost support.
  (3.0, 5.0, 6.0),   # Nearby current target is moving away.
  (0.8, 20.0, 0.8),  # Normal low-speed following remains available.
])
def test_nonconflicting_current_leads_keep_reference_smoothing_and_normal_motion(speed, distance, lead_speed):
  with pipeline(speed=speed, measured_accel=0.0) as (planner, control, state):
    set_lead(state, True)
    for frame in range(8):
      step(planner, control, state, frame)
    set_lead(state, False)
    for frame in range(8, 18):
      step(planner, control, state, frame)
    set_lead(state, True)
    lead = state['radarState'].leadOne
    lead.dRel, lead.vRel, lead.vLead = distance, lead_speed - speed, lead_speed
    for frame in range(18, 23):
      plan, outputs = step(planner, control, state, frame)
      assert planner.mpc._cost_weights[4] > 0.0
      assert plan.aTarget > 0.0 and max(outputs) > 0.0


@pytest.mark.parametrize('speed', [0.0, 0.2, 0.8])
def test_current_low_speed_stop_still_brakes_and_holds(speed):
  with pipeline(speed=speed, measured_accel=0.0) as (planner, control, state):
    set_lead(state, True)
    lead = state['radarState'].leadOne
    lead.dRel, lead.vRel, lead.vLead = 5.0, -speed, 0.0
    state['modelV2'].action.shouldStop = True
    state['modelV2'].action.desiredAcceleration = -0.5
    for frame in range(6):
      plan, outputs = consume(planner, control, state, frame)
      assert plan.solverStatus == 0 and plan.motionIntent.valid
      assert plan.aTarget <= -0.5 and max(outputs) <= 0.0
      assert plan.shouldStop and not plan.motionIntent.releaseAllowed
      assert control.resolved_should_stop
