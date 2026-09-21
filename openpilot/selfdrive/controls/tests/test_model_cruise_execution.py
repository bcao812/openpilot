"""A strictly winning cruise request keeps its execution meaning with a model.

The B=e18b1de mode-switch counterexample has the same cruise target before
R=1a7aefc3 adds braking integral. These tests preserve the factual MOVING or
FORWARD reference and exercise the real producer/owner boundary. Exact ties
control upstream candidate readouts; they do not replace final arbitration.
Fixed-input tests are not a closed-loop or vehicle acceptance claim.
"""

from contextlib import ExitStack
from dataclasses import replace
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import CREEP_STOP_T, STOP_RELEASE_ACCEL
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.motion_supervisor import (
  STARTING_ACCEL_RATE, MotionIntent, MotionPhase, MotionReferenceKind, MotionResponseKind, StopReason,
)
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import UnifiedLongitudinalPlanner
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import decoded_commands
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import consume, make_handover_state, make_planner_owner
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_cruise_coast_response import learned_harness
from openpilot.selfdrive.controls.tests.test_motion_supervisor import MotionHarness
from openpilot.selfdrive.modeld.constants import ModelConstants


INSTANCE_NS = 1_000_000_000
UPWARD_STEP = STARTING_ACCEL_RATE * DT_CTRL
MODEL_KINDS = (MotionReferenceKind.MOVING, MotionReferenceKind.FORWARD)
PRODUCER_MODULE = 'openpilot.selfdrive.controls.lib.unified_longitudinal_planner'


def model_request(kind, **changes):
  return replace(MotionIntent(reference_kind=int(kind), response_kind=int(MotionResponseKind.CRUISE_REQUEST),
                              producer_instance_ns=INSTANCE_NS, allow_creep=False), **changes)


def model_case(kind, *, target=4.0):
  params = Params()
  for name in ('SmartCruiseControlVision', 'SmartCruiseControlMap', 'DynamicExperimentalControl'):
    params.put_bool(name, False, block=True)
  planner, control = make_planner_owner(speed=3.0)
  state = make_handover_state(speed=3.0, target=target, experimental=True)
  times = np.asarray(ModelConstants.T_IDXS)
  slope = 0.6 if kind == MotionReferenceKind.FORWARD else 0.0
  model = state['modelV2']
  model.position.x = (3.0 * times + 0.5 * slope * times**2).tolist()
  model.velocity.x = (3.0 + slope * times).tolist()
  model.acceleration.x = np.full(len(times), slope).tolist()
  model.action.desiredAcceleration = 2.0
  return planner, control, state


@pytest.mark.parametrize('kind', MODEL_KINDS)
@pytest.mark.parametrize('release_allowed', [False, True])
def test_request_wire_preserves_raw_model_facts_without_inventing_release(kind, release_allowed):
  intent = model_request(kind, release_allowed=release_allowed)
  assert intent.is_valid()
  planner = SimpleNamespace(longitudinal_mode=LongitudinalMode.UNIFIED, motion_intent=intent)
  message = log.LongitudinalPlan.new_message()
  UnifiedLongitudinalPlanner(planner).publish(message)
  with log.LongitudinalPlan.from_bytes(message.to_bytes()) as decoded:
    actual = MotionIntent.from_message(decoded.motionIntent)
    assert actual == intent and actual.is_valid()
    assert actual.reference_kind == kind
    assert actual.release_allowed == release_allowed
  # Other execution kinds still cannot relabel an admitted model as UNKNOWN.
  assert not replace(intent, response_kind=int(MotionResponseKind.CRUISE_COAST)).is_valid()
  assert not replace(intent, response_kind=int(MotionResponseKind.PURE_CRUISE)).is_valid()


@pytest.mark.parametrize('kind', [MotionReferenceKind.EXACT_ZERO, MotionReferenceKind.REST, MotionReferenceKind.NO_FORWARD])
def test_stop_or_no_forward_model_cannot_carry_a_cruise_execution_contract(kind):
  assert not model_request(kind).is_valid()


@pytest.mark.parametrize('kind', MODEL_KINDS)
def test_real_acc_model_acc_switch_keeps_the_same_cruise_request_without_braking_integral(kind):
  planner, control, state = model_case(kind, target=2.6)
  plans = []
  for frame in range(24):
    experimental = 8 <= frame < 16
    state['selfdriveState'].experimentalMode = experimental
    if frame:
      state['controlsState'].longControlState = control.long_control_state
    plan, outputs = consume(planner, control, state, frame)
    assert plan.solverStatus == 0 and plan.motionIntent.valid and not plan.motionIntent.reasons
    assert plan.motionIntent.referenceKind == (kind if experimental else MotionReferenceKind.UNKNOWN)
    assert plan.reference.accel > plan.aTarget
    assert plan.reference.constraintSource == log.LongitudinalPlan.LongitudinalPlanSource.cruise
    assert plan.aTarget == plan.reference.constraintAccel
    assert plan.motionIntent.responseKind == MotionResponseKind.CRUISE_REQUEST
    assert control.motion_phase == MotionPhase.TRACK and not control.motion_input_constrained
    assert control.pid.i == 0.0
    assert outputs[-1] == plan.aTarget
    plans.append(plan)
  # Set speed, ego feedback and model action stay fixed across both switches.
  assert plans[7].aTarget == plans[8].aTarget == plans[15].aTarget == plans[16].aTarget


@pytest.mark.parametrize('kind', MODEL_KINDS)
@pytest.mark.parametrize('case', [
  'cruise_wins', 'native_tie', 'native_stronger', 'action_tie', 'action_stronger',
  'maximum_tie', 'maximum_stronger', 'lead_tie', 'lead_stronger',
])
def test_model_cruise_execution_requires_strict_independent_winner(kind, case):
  planner, control, state = model_case(kind)
  # Binary-exact values also survive Float32 action transport, so a tie here
  # is a real tie, not an accidental consequence of different wire rounding.
  request, native, action, maximum = 0.125, 0.5, 2.0, 2.0
  if case in ('native_tie', 'lead_tie'):
    native = request
  elif case in ('native_stronger', 'lead_stronger'):
    native = -0.5
  elif case == 'action_tie':
    action = request
  elif case == 'action_stronger':
    action = -0.5
  elif case == 'maximum_tie':
    maximum = request
  elif case == 'maximum_stronger':
    maximum = 0.0625
  if case.startswith('lead_'):
    lead = state['radarState'].leadOne
    lead.present, lead.dRel, lead.vLead, lead.vRel = True, 7.0, 3.0, 0.0
    lead.aLeadK, lead.aLeadTau, lead.modelProb = 0.0, 1.5, 0.99
  state['modelV2'].action.desiredAcceleration = action
  with ExitStack() as stack:
    update = stack.enter_context(patch.object(planner.mpc, 'update', wraps=planner.mpc.update))
    stack.enter_context(patch.object(planner.reference_planner.cruise_response, 'update', return_value=request))
    stack.enter_context(patch(f'{PRODUCER_MODULE}.sample_reference_acceleration', return_value=native))
    stack.enter_context(patch(f'{PRODUCER_MODULE}.cruise_accel_limit', return_value=maximum))
    plan, _ = consume(planner, control, state, 0)
  reference = update.call_args.kwargs['reference']
  assert reference.cruise_speed is None
  assert reference.typed_model is None
  assert plan.solverStatus == 0 and plan.motionIntent.valid and not plan.motionIntent.reasons
  assert plan.motionIntent.referenceKind == kind
  assert plan.aTarget == min(request, native, action, maximum)
  if case.startswith('lead_'):
    assert plan.reference.source == log.LongitudinalPlan.LongitudinalPlanSource.lead0
  expected = MotionResponseKind.CRUISE_REQUEST if case == 'cruise_wins' else MotionResponseKind.ABSOLUTE_ACC
  assert plan.motionIntent.responseKind == expected


@pytest.mark.parametrize('kind', MODEL_KINDS)
def test_current_fcw_vetoes_model_cruise_execution_even_when_its_scalar_wins(kind):
  planner, control, state = model_case(kind)
  native_update = planner.mpc.update

  def update_with_fcw(*args, **kwargs):
    native_update(*args, **kwargs)
    planner.mpc.crash_cnt = 3

  with patch.object(planner.mpc, 'update', side_effect=update_with_fcw), \
       patch(f'{PRODUCER_MODULE}.sample_reference_acceleration', return_value=1.0):
    plan, _ = consume(planner, control, state, 0)
  assert planner.fcw and plan.motionIntent.valid
  assert plan.aTarget == plan.reference.constraintAccel < plan.reference.accel
  assert plan.motionIntent.referenceKind == kind
  assert plan.motionIntent.responseKind == MotionResponseKind.ABSOLUTE_ACC
  assert not plan.motionIntent.releaseAllowed


@pytest.mark.parametrize('kind', MODEL_KINDS)
@pytest.mark.parametrize('tuned_ki', [False, True], ids=['launch-response', 'configured-pid'])
def test_model_cruise_execution_clears_learned_integral_and_does_not_create_a_brake_request(kind, tuned_ki):
  harness = learned_harness(tuned_ki=tuned_ki)
  command = float(np.float32(-0.19))
  outputs = [harness.control.last_output_accel]
  settle_ticks = math.ceil(max(0.0, command - outputs[-1]) / UPWARD_STEP) + 2
  for _ in range(settle_ticks + 10):
    outputs.append(harness.step(command, model_request(kind)))
    assert not harness.control.motion_input_constrained and not harness.control.resolved_should_stop
    assert harness.control.pid.i == 0.0
  assert np.max(np.diff(outputs)) <= UPWARD_STEP + 1e-12
  assert outputs[-10:] == [command] * 10
  assert harness.control.motion_phase == MotionPhase.TRACK
  assert all(not encoded['BRAKE_REQUEST'] for encoded in decoded_commands(outputs[-10:]))
  # A fresh independently binding action/lead target still learns braking I.
  absolute = replace(model_request(kind), response_kind=int(MotionResponseKind.ABSOLUTE_ACC))
  for _ in range(10):
    assert harness.step(-0.8, absolute) <= -0.8
  assert harness.control.pid.i < 0.0


@pytest.mark.parametrize('kind', MODEL_KINDS)
def test_model_cruise_kind_cannot_release_hold_without_independent_permission(kind):
  harness = MotionHarness()
  harness.stopped()
  for _ in range(round(2.0 * CREEP_STOP_T / DT_MDL)):
    assert harness.step(model_request(kind), a_target=STOP_RELEASE_ACCEL * 2.0, v_ego=0.0, standstill=True,
                        advance_ns=round(DT_MDL * 1e9))
    assert not harness.supervisor.degraded and not harness.supervisor.waiting_for_fresh_input
    assert harness.supervisor.phase == MotionPhase.HOLD


@pytest.mark.parametrize('kind', MODEL_KINDS)
def test_plan_only_model_cruise_cannot_withdraw_approach_or_retire_its_braking_history(kind):
  harness = learned_harness()
  stop = MotionIntent(reasons=int(StopReason.E2E), producer_instance_ns=INSTANCE_NS, allow_creep=False)
  for _ in range(3):
    harness.step(-0.35, stop)
  assert harness.control.motion_phase == MotionPhase.APPROACH
  previous_output, previous_integral = harness.control.last_output_accel, harness.control.pid.i
  plan_time_ns = harness.now_ns + round(DT_CTRL * 1e9)
  for _ in range(5):
    output = harness.step(-0.35, model_request(kind), new_input=False,
                          plan_time_ns=plan_time_ns, model_time_ns=harness.model_time_ns)
    assert not harness.control.motion_input_constrained
    assert harness.control.resolved_should_stop and harness.control.motion_phase == MotionPhase.APPROACH
    assert harness.control.motion_reasons & StopReason.E2E
    assert harness.control.pid.i <= previous_integral + 1e-12
    assert output <= previous_output + 1e-12
    previous_output, previous_integral = output, harness.control.pid.i
