"""Stop, policy and authority cases retained after the transitional owner retired."""
from unittest.mock import Mock, patch

import pytest

from opendbc.car.honda.values import CarControllerParams
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.longitudinal_trace import LongitudinalTrace
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION, MotionIntent, MotionPhase, StopReason
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import advance, consume, make_handover_state, make_planner_owner
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params  # noqa: F401
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanSource as SPSource


LEGACY_OWNER_FIELDS = ('stop_latched', 'creep_t', 'creep_v', 'creep_stop', 'creep_total_t', 'moving_lead_t', 'close_lead_stop')
ACCEL_LIMITS = (CarControllerParams.BOSCH_ACCEL_MIN, CarControllerParams.BOSCH_ACCEL_MAX)


def test_unified_selects_one_owner_and_never_runs_legacy_reference():
  with patch('openpilot.selfdrive.controls.lib.legacy_longitudinal_planner.initialize_legacy_reference',
             side_effect=AssertionError('Legacy reference initialized')), \
       patch('openpilot.selfdrive.controls.lib.legacy_longitudinal_planner.update_legacy_reference',
             side_effect=AssertionError('Legacy reference executed')):
    planner, controller = make_planner_owner()
    assert planner.longitudinal_mode == LongitudinalMode.UNIFIED
    assert planner.unified
    assert controller.longitudinal_mode == LongitudinalMode.UNIFIED
    state = make_handover_state()
    with patch.object(planner.mpc, 'update', wraps=planner.mpc.update) as update:
      for frame in range(5):
        consume(planner, controller, state, frame)
        assert all(not hasattr(planner, attribute) for attribute in LEGACY_OWNER_FIELDS)
  assert update.call_count == 5
  assert all(call.kwargs.get('reference') is not None for call in update.call_args_list)
  assert planner.motion_intent.valid
  assert not controller.motion_degraded


@pytest.mark.parametrize('lead_slot', ['leadOne', 'leadTwo'])
@pytest.mark.parametrize('measured_accel', [-0.2, -1.367, -2.0])
def test_real_sp_policy_preserves_measured_state_and_critical_lead_braking(lead_slot, measured_accel):
  Params().put_bool('SmartCruiseControlVision', True)
  planner, owner = make_planner_owner()
  state = make_handover_state(target=30.0)
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
    plan, outputs = consume(planner, owner, state, frame)
    assert plan.motionIntent.valid
    assert planner.mpc.x0[2] == pytest.approx(measured_accel)
    braking_rows += sum(output < CarControllerParams.BOSCH_GAS_LOOKUP_BP[0] for output in outputs)
  assert planner.source == SPSource.sccVision
  assert planner.cruise_target.speed < state['carState'].vCruise / 3.6
  assert braking_rows > 0
  assert not owner.motion_degraded


def test_unified_publishes_current_stop_evidence_without_a_second_owner():
  planner, controller = make_planner_owner(speed=0.0)
  state = make_handover_state(speed=0.0, experimental=True)
  for frame in range(8):
    advance(state, frame)
    planner.update(state, now_ns=state.logMonoTime['carState'] + 500_000)
  state['modelV2'].action.shouldStop = True
  advance(state, 8)
  planner.update(state, now_ns=state.logMonoTime['carState'] + 500_000)
  assert planner.motion_intent.reasons & int(StopReason.E2E)
  state['modelV2'].action.shouldStop = False
  advance(state, 9)
  planner.update(state, now_ns=state.logMonoTime['carState'] + 500_000)
  assert not planner.motion_intent.reasons & int(StopReason.E2E)
  assert all(not hasattr(planner, attribute) for attribute in LEGACY_OWNER_FIELDS)


@pytest.mark.parametrize('target', [0.0, 20.0])
@pytest.mark.parametrize('lead_slot', ['leadOne', 'leadTwo'])
@pytest.mark.parametrize('measured_accel', [-0.4, -1.0])
def test_warm_stop_entry_and_hold_do_not_depend_on_positive_scalar(target, lead_slot, measured_accel):
  planner, owner = make_planner_owner(speed=0.1, initial_accel=measured_accel)
  state = make_handover_state(speed=0.1, target=target)
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
  planner, owner = make_planner_owner(speed=12.0)
  state = make_handover_state(speed=12.0, target=10.0)
  with patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.reference_requests_departure', return_value=False):
    for frame in range(10):
      plan, outputs = consume(planner, owner, state, frame)
      assert not plan.motionIntent.releaseAllowed
      assert not plan.motionIntent.reasons
      assert not owner.resolved_should_stop
      assert owner.motion_phase == MotionPhase.TRACK


def test_unified_mode_and_raw_evidence_survive_wire_and_trace():
  trace = LongitudinalTrace()
  planner, owner = make_planner_owner(trace=trace)
  state = make_handover_state()
  plan, outputs = consume(planner, owner, state, 0)
  assert plan.motionIntent.version == MOTION_INTENT_VERSION
  assert plan.motionIntent.mode == int(LongitudinalMode.UNIFIED)
  assert MotionIntent.from_message(plan.motionIntent) == planner.motion_intent
  assert trace.queue and trace.failures == 0
  sample = trace.queue[-1]
  values = dict(zip(sample.fields, sample, strict=True))
  assert values['intent_mode'] == int(LongitudinalMode.UNIFIED)
  assert sample.detail['mode'] == 'unified'
  assert values['reference_accel'] == planner.reference_planner.reference_accel


@pytest.mark.parametrize('overrides', ['gasPressed', 'brakePressed'])
def test_unified_immediate_override_does_not_preserve_hold_authority(overrides):
  planner, owner = make_planner_owner(speed=0.0)
  state = make_handover_state(speed=0.0, target=0.0)
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
  planner, owner = make_planner_owner(speed=5.0, initial_accel=previous_request)
  owner.last_output_accel = previous_request
  state = make_handover_state(speed=5.0, experimental=experimental)
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
  planner, owner = make_planner_owner()
  state = make_handover_state()
  if fault == 'model_shape':
    state['modelV2'].velocity.x = []
  elif fault == 'owner_mode':
    state['controlsState'].motionState.planningMode = int(LongitudinalMode.SUPERVISED)
  else:
    state.all_checks = lambda service_list=None: False
  planner.update(state, now_ns=state.logMonoTime['carState'] + 500_000)
  assert planner.motion_intent.braking_accel is None
