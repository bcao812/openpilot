"""Real MPC/owner coverage for weak-model lead launch and subsequent risks."""
from unittest.mock import patch

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.motion_supervisor import MotionPhase, MotionReferenceKind, StopReason
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import consume
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_lead_launch import LOGGED_WEAK_VELOCITY
from openpilot.selfdrive.controls.tests.test_stationary_lead_departure import held_recorded_departure_case
from openpilot.selfdrive.controls.tests.test_typed_model_reference import MODEL_TIMES


def held_weak_model_case():
  planner, owner, state = held_recorded_departure_case()
  state['carState'].vEgo = -7.101848276125367e-11
  state['carState'].vEgoRaw = 0.0
  raw = LOGGED_WEAK_VELOCITY
  state['modelV2'].velocity.x = raw.tolist()
  state['modelV2'].position.x = (np.maximum(raw, 0.0) * MODEL_TIMES).tolist()
  state['modelV2'].acceleration.x = np.gradient(raw, MODEL_TIMES).tolist()
  state['modelV2'].action.desiredAcceleration = 0.11845896393060684
  for lead, gap, speed in ((state['radarState'].leadOne, 7.6987557, 0.7767373),
                           (state['radarState'].leadTwo, 7.8280778, 0.7798193)):
    lead.dRel, lead.vLead, lead.vRel, lead.aLeadK, lead.modelProb = gap, speed, speed, 0.02, 0.997
  return planner, owner, state


def step(planner, owner, state, frame):
  state['controlsState'].longControlState = owner.long_control_state
  motion = state['controlsState'].motionState
  motion.phase, motion.resolvedStop = owner.motion_phase.text, owner.resolved_should_stop
  motion.degraded, motion.waitingForFreshInput = owner.motion_degraded, False
  for lead in (state['radarState'].leadOne, state['radarState'].leadTwo):
    if lead.present:
      lead.dRel += lead.vRel * 0.05
  return consume(planner, owner, state, frame)


@pytest.mark.parametrize('model_accel', [0.0, 0.11845896393060684])
def test_mpc_supplies_launch_acceleration_without_the_weak_model_obstacle_or_cap(model_accel):
  planner, owner, state = held_weak_model_case()
  state['modelV2'].action.desiredAcceleration = model_accel
  accelerated = released = False
  for frame in range(20, 51):
    plan, outputs = step(planner, owner, state, frame)
    assert plan.solverStatus == 0 and plan.motionIntent.valid
    if planner.reference_planner.lead_launch_active:
      reference = planner.mpc._influence_problem[3]
      assert reference.model_position is reference.model_velocity is reference.model_acceleration is reference.typed_model is None
      assert reference.cruise_speed is not None
      assert plan.motionIntent.referenceKind == MotionReferenceKind.UNKNOWN
      assert not planner.reference_planner.model_evidence.allows_external_departure
      accelerated |= plan.aTarget > model_accel + 0.001
      released |= plan.motionIntent.releaseAllowed and not owner.resolved_should_stop and max(outputs) > 0.0
  assert accelerated and released
  assert owner.motion_phase in (MotionPhase.RELEASE, MotionPhase.TRACK)


def test_new_cut_in_with_stop_request_cancels_launch_in_the_same_continuous_history():
  planner, owner, state = held_weak_model_case()
  for frame in range(20, 39):
    step(planner, owner, state, frame)
  assert planner.reference_planner.lead_launch_active
  # The vision slot has no physical ID. A new closer geometry must withdraw
  # its old continuity even before the new model stop publication arrives.
  state['radarState'].leadOne.dRel -= 2.0
  plan, _ = step(planner, owner, state, 39)
  assert not planner.reference_planner.lead_launch_active and not plan.motionIntent.releaseAllowed
  state['modelV2'].action.shouldStop = True
  state['modelV2'].action.desiredAcceleration = -0.2
  state['modelV2'].velocity.x = [0.0] * len(MODEL_TIMES)
  state['modelV2'].position.x = [0.0] * len(MODEL_TIMES)
  state['modelV2'].acceleration.x = [0.0] * len(MODEL_TIMES)
  plan, outputs = step(planner, owner, state, 40)
  assert not planner.reference_planner.lead_launch_active and not plan.motionIntent.releaseAllowed
  assert plan.shouldStop and plan.motionIntent.reasons & int(StopReason.E2E)
  assert owner.motion_phase == MotionPhase.HOLD and owner.resolved_should_stop
  assert max(outputs) <= 0.0


def test_current_solve_stop_veto_also_clears_the_action_cap_exception():
  planner, owner, state = held_weak_model_case()
  for frame in range(20, 39):
    step(planner, owner, state, frame)
  assert planner.reference_planner.lead_launch_active
  with patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.should_stop_for_close_lead', return_value=True):
    plan, outputs = step(planner, owner, state, 39)
  assert not planner.reference_planner.lead_launch_active and not planner.reference_planner.lead_launch.active
  assert not plan.motionIntent.releaseAllowed and plan.motionIntent.reasons & int(StopReason.CLOSE_LEAD)
  assert plan.aTarget <= state['modelV2'].action.desiredAcceleration + 1e-6
  assert owner.resolved_should_stop and max(outputs) <= 0.0


@pytest.mark.parametrize('risk', ['negative_action', 'model_stop', 'hard_brake', 'driver_gas', 'rollback', 'no_lead'])
def test_current_risk_cannot_enter_the_mpc_launch_exception(risk):
  planner, owner, state = held_weak_model_case()
  if risk == 'negative_action':
    state['modelV2'].action.desiredAcceleration = -0.02
  elif risk == 'model_stop':
    state['modelV2'].action.shouldStop = True
  elif risk == 'hard_brake':
    state['modelV2'].meta.hardBrakePredicted = True
  elif risk == 'driver_gas':
    state['carState'].gasPressed = True
  elif risk == 'rollback':
    state['carState'].vEgo = state['carState'].vEgoRaw = -0.001
  elif risk == 'no_lead':
    state['radarState'].leadOne.present = state['radarState'].leadTwo.present = False
  for frame in range(20, 36):
    plan, _ = step(planner, owner, state, frame)
    assert not planner.reference_planner.lead_launch_active
    assert not plan.motionIntent.releaseAllowed
    if risk == 'negative_action':
      assert plan.aTarget <= -0.02 + 1e-6
