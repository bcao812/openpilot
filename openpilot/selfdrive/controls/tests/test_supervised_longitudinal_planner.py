from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, CarControllerParams
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.longitudinal_trace import LongitudinalTrace
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION, MotionIntent, MotionPhase, StopReason
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import UnifiedLongitudinalPlanner
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_planner import Publisher, advance, make_state
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanSource as SPSource


LEGACY_OWNER_FIELDS = ('stop_latched', 'creep_t', 'creep_v', 'creep_stop', 'creep_total_t', 'moving_lead_t', 'close_lead_stop')
ACCEL_LIMITS = (CarControllerParams.BOSCH_ACCEL_MIN, CarControllerParams.BOSCH_ACCEL_MAX)


@pytest.fixture(autouse=True)
def isolated_params():
  with OpenpilotPrefix():
    yield


def make_supervised(speed=20.0, initial_accel=0.0, *, trace=None):
  params = CarInterface.get_non_essential_params(CAR.HONDA_ACCORD)
  params.openpilotLongitudinalControl = True
  extension = CarInterface.get_non_essential_params_sp(params, CAR.HONDA_ACCORD)
  planner = LongitudinalPlanner(params, extension, init_v=speed, init_a=initial_accel, unified=False, supervised=True, trace=trace)
  return planner, LongControl(params, extension, supervised=True)


def make_supervised_state(speed=20.0, target=22.0, experimental=False):
  state = make_state(speed=speed, target=target, experimental=experimental)
  state['carState'].canValid = True
  state['controlsState'].motionState.version = MOTION_INTENT_VERSION
  state['controlsState'].motionState.planningMode = int(LongitudinalMode.SUPERVISED)
  return state


def consume(planner, controller, state, frame):
  model_time = advance(state, frame)
  planner.update(state)
  publisher = Publisher()
  planner.publish(state, publisher)
  message = publisher.messages['longitudinalPlan']
  plan = message.longitudinalPlan
  outputs = []
  for tick in range(5):
    outputs.append(float(controller.update(
      True, state['carState'], plan.aTarget, plan.shouldStop, ACCEL_LIMITS,
      intent=MotionIntent.from_message(plan.motionIntent),
      plan_time_ns=model_time + 1_000_000, model_time_ns=model_time,
      now_ns=model_time + 2_000_000 + tick * 10_000_000,
      plan_valid=message.valid, solver_status=plan.solverStatus, braking_plan_valid=state.all_checks(['carState']),
    )))
  return plan, outputs


def test_supervised_selects_one_owner_and_does_not_run_unified_reference():
  planner, controller = make_supervised()
  assert planner.longitudinal_mode == LongitudinalMode.SUPERVISED
  assert planner.motion_enabled and not planner.unified
  assert controller.longitudinal_mode == LongitudinalMode.SUPERVISED
  state = make_supervised_state()
  with patch.object(UnifiedLongitudinalPlanner, 'update', side_effect=AssertionError('Unified reference executed')), \
       patch.object(planner.mpc, 'update', wraps=planner.mpc.update) as update:
    for frame in range(5):
      consume(planner, controller, state, frame)
      assert all(not hasattr(planner, attribute) for attribute in LEGACY_OWNER_FIELDS)
  assert update.call_count == 5
  assert all(call.kwargs.get('reference') is None for call in update.call_args_list)
  assert planner.motion_intent.valid
  assert not controller.motion_degraded


@pytest.mark.parametrize('speed,target,model_accel', [
  (0.6, 2.0, 0.5), (2.5, 3.0, 0.5), (10.0, 10.2, 0.5),
  (20.0, 22.0, 0.5), (20.0, 17.0, -1.2), (10.0, 14.0, -2.0),
])
@pytest.mark.parametrize('experimental', [False, True])
def test_legacy_normal_reference_and_request_are_reused_exactly(speed, target, model_accel, experimental):
  supervised, controller = make_supervised(speed)
  legacy = LongitudinalPlanner(supervised.CP, controller.CP_SP, init_v=speed, unified=False, supervised=False)
  state = make_supervised_state(speed, target, experimental)
  state['carState'].aEgo = -0.7
  state['modelV2'].action.desiredAcceleration = model_accel
  for frame in range(30):
    advance(state, frame)
    legacy.update(state)
    supervised.update(state)
    assert supervised.motion_intent.valid
    assert supervised.output_a_target == legacy.output_a_target
    assert supervised.a_cruise == legacy.a_cruise
    assert supervised.v_desired_filter.x == legacy.v_desired_filter.x
    assert supervised.mpc.source == legacy.mpc.source
    np.testing.assert_array_equal(supervised.mpc.x0, legacy.mpc.x0)
    np.testing.assert_array_equal(supervised.mpc.a_prev, legacy.mpc.a_prev)
    np.testing.assert_array_equal(supervised.v_desired_trajectory, legacy.v_desired_trajectory)
    np.testing.assert_array_equal(supervised.a_desired_trajectory, legacy.a_desired_trajectory)
    np.testing.assert_array_equal(supervised.j_desired_trajectory, legacy.j_desired_trajectory)
  assert all(not hasattr(supervised, attribute) for attribute in LEGACY_OWNER_FIELDS)


@pytest.mark.parametrize('lead_slot', ['leadOne', 'leadTwo'])
@pytest.mark.parametrize('measured_accel', [-0.2, -1.367, -2.0])
def test_real_sp_producer_keeps_legacy_critical_request_in_supervised_stage(lead_slot, measured_accel):
  Params().put_bool('SmartCruiseControlVision', True)
  supervised, owner = make_supervised()
  legacy = LongitudinalPlanner(supervised.CP, owner.CP_SP, init_v=20.0, unified=False, supervised=False)
  legacy_controller = LongControl(supervised.CP, owner.CP_SP)
  state = make_supervised_state(target=30.0)
  state['carState'].aEgo = measured_accel
  state['modelV2'].orientationRate.z = [0.15] * len(ModelConstants.T_IDXS)
  lead = getattr(state['radarState'], lead_slot)
  lead.present = True
  lead.dRel = 40.0
  lead.vRel = -5.0
  lead.vLead = lead.vLeadK = 15.0
  lead.aLeadK = -0.5
  lead.aLeadTau = 1.5
  lead.modelProb = 1.0
  braking_rows = 0
  for frame in range(40):
    advance(state, frame)
    legacy.update(state)
    plan, actual = consume(supervised, owner, state, frame)
    assert supervised.source == legacy.source
    assert supervised.mpc.x0[2] == legacy.mpc.x0[2]
    assert supervised.output_a_target == legacy.output_a_target
    for output in actual:
      expected = float(legacy_controller.update(
        True, state['carState'], plan.aTarget, legacy.output_should_stop, ACCEL_LIMITS,
      ))
      assert output == expected
      braking_rows += expected < CarControllerParams.BOSCH_GAS_LOOKUP_BP[0]
  assert supervised.source == SPSource.sccVision
  assert supervised.mpc.x0[2] != pytest.approx(measured_accel)
  assert braking_rows > 0
  assert not owner.motion_degraded


def test_supervised_publishes_raw_stop_not_a_second_latched_stop():
  planner, controller = make_supervised(speed=0.0)
  state = make_supervised_state(speed=0.0, experimental=True)
  for frame in range(8):
    advance(state, frame)
    planner.update(state)
  state['modelV2'].action.shouldStop = True
  advance(state, 8)
  planner.update(state)
  assert planner.motion_intent.reasons & int(StopReason.E2E)
  state['modelV2'].action.shouldStop = False
  advance(state, 9)
  planner.update(state)
  assert not planner.motion_intent.reasons
  assert not planner.output_should_stop
  assert all(not hasattr(planner, attribute) for attribute in LEGACY_OWNER_FIELDS)


@pytest.mark.parametrize('target', [0.0, 20.0])
@pytest.mark.parametrize('lead_slot', ['leadOne', 'leadTwo'])
@pytest.mark.parametrize('measured_accel', [-0.4, -1.0])
def test_warm_stop_entry_and_hold_do_not_depend_on_positive_scalar(target, lead_slot, measured_accel):
  planner, owner = make_supervised(speed=0.1, initial_accel=measured_accel)
  state = make_supervised_state(speed=0.1, target=target)
  lead = getattr(state['radarState'], lead_slot)
  lead.present = True
  lead.dRel = 6.0
  lead.vLead = lead.vLeadK = 0.0
  lead.aLeadK = 0.0
  lead.aLeadTau = 1.5
  lead.modelProb = 1.0
  for frame, speed in enumerate([0.1, 0.075, 0.05, 0.025, 0.0, 0.0]):
    state['carState'].vEgo = speed
    state['carState'].aEgo = measured_accel
    state['carState'].standstill = speed == 0.0
    lead.vRel = -speed
    plan, outputs = consume(planner, owner, state, frame)
    assert plan.motionIntent.valid
    assert plan.motionIntent.reasons
    assert not owner.motion_degraded
    assert owner.resolved_should_stop
    assert all(output <= CarControllerParams.BOSCH_GAS_LOOKUP_BP[0] for output in outputs)
  assert owner.motion_phase == MotionPhase.HOLD


def test_absent_release_permission_is_not_a_general_stop_command():
  planner, owner = make_supervised(speed=12.0)
  state = make_supervised_state(speed=12.0, target=10.0)
  with patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.reference_requests_departure', return_value=False):
    for frame in range(10):
      plan, outputs = consume(planner, owner, state, frame)
      assert not plan.motionIntent.releaseAllowed
      assert not plan.motionIntent.reasons
      assert not owner.resolved_should_stop
      assert owner.motion_phase == MotionPhase.TRACK


def test_supervised_mode_and_raw_evidence_survive_wire_and_trace():
  trace = LongitudinalTrace()
  planner, owner = make_supervised(trace=trace)
  state = make_supervised_state()
  plan, outputs = consume(planner, owner, state, 0)
  assert plan.motionIntent.version == MOTION_INTENT_VERSION
  assert plan.motionIntent.mode == int(LongitudinalMode.SUPERVISED)
  assert MotionIntent.from_message(plan.motionIntent) == planner.motion_intent
  assert trace.queue and trace.failures == 0
  sample = trace.queue[-1]
  values = dict(zip(sample.fields, sample, strict=True))
  assert values['intent_mode'] == int(LongitudinalMode.SUPERVISED)
  assert sample.detail['mode'] == 'supervised'
  assert values['reference_accel'] == planner.reference_planner.reference_accel


@pytest.mark.parametrize('overrides', ['gasPressed', 'brakePressed'])
def test_supervised_immediate_override_does_not_preserve_hold_authority(overrides):
  planner, owner = make_supervised(speed=0.0)
  state = make_supervised_state(speed=0.0, target=0.0)
  consume(planner, owner, state, 0)
  assert owner.resolved_should_stop
  setattr(state['carState'], overrides, True)
  plan, outputs = consume(planner, owner, state, 1)
  assert outputs == [0.0] * 5
  assert owner.motion_phase == MotionPhase.OFF
  assert owner.long_control_state == LongCtrlState.off


@pytest.mark.parametrize('experimental', [False, True])
@pytest.mark.parametrize('previous_request', [-0.1, -1.5, -3.0])
def test_failed_reference_publishes_separately_qualified_braking_only(experimental, previous_request):
  planner, owner = make_supervised(speed=5.0, initial_accel=previous_request)
  owner.last_output_accel = previous_request
  state = make_supervised_state(speed=5.0, experimental=experimental)
  state['modelV2'].action.desiredAcceleration = -2.5
  solver = Mock(wraps=planner.mpc.solver)
  solver.solve.return_value = 4
  with patch.object(planner.mpc, 'solver', solver):
    plan, outputs = consume(planner, owner, state, 0)
  expected = min(previous_request, -0.5, -2.5 if experimental else -0.5)
  assert plan.solverStatus == 4
  assert not plan.motionIntent.valid
  assert plan.motionIntent.brakingValid
  assert plan.motionIntent.brakingAccel == expected
  assert not plan.motionIntent.releaseAllowed
  assert owner.motion_degraded
  assert all(output <= expected for output in outputs)


@pytest.mark.parametrize('fault', ['model_shape', 'owner_mode', 'unhealthy'])
def test_non_solver_fault_cannot_invent_independent_braking_evidence(fault):
  planner, owner = make_supervised()
  state = make_supervised_state()
  if fault == 'model_shape':
    state['modelV2'].velocity.x = []
  elif fault == 'owner_mode':
    state['controlsState'].motionState.planningMode = int(LongitudinalMode.UNIFIED)
  else:
    state.all_checks = lambda service_list=None: False
  planner.update(state)
  assert planner.motion_intent.braking_accel is None
