"""Only formal-lead or solver failure retains the ordinary braking envelope."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, MPC_FAILURE_ACCEL
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent, StopReason
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import consume, make_state
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_lead_qualification import caution_lead, formal_lead
from openpilot.selfdrive.controls.tests.test_mpc_failure_chain import can_from_car_control
from openpilot.selfdrive.controls.tests.test_radar_consumption import native_frame, radar_frame
from openpilot.selfdrive.controls.tests.test_unified_radar_consumption import enabled_planner
from openpilot.selfdrive.controls.tests.test_unified_radar_passing import controlled_mode


def failure_request(kind):
  planner = enabled_planner(10.0)
  controller = LongControl(planner.CP, planner.CP_SP)
  state = make_state(speed=10.0, target=20.0, experimental=True)
  state['carState'].canValid = True
  state['carState'].cruiseState.available = True
  state['carState'].cruiseState.enabled = True
  state['controlsState'].motionState.phase = 'track'
  state['modelV2'].action.desiredAcceleration = .2
  with controlled_mode(planner):
    for frame in range(8):
      stamp = 1_000_000_000 + frame * 50_000_000
      state['radarState'] = native_frame(radar_frame(publication=stamp))
      state['modelV2'].timestampEof = stamp
      plan, commands = consume(planner, controller, state, frame)
    assert plan.solverStatus == 0 and max(commands) > 0.0
    planner.output_a_target = .2
    frame, stamp = 8, 1_400_000_000
    state['modelV2'].timestampEof = stamp
    objects = []
    if kind not in ('clear', 'solver'):
      lead = formal_lead(now_ns=stamp, dRel=.8, vLead=0.0, vLeadK=0.0, vRel=-10.0,
                         evidence={'conflictEnd': .10, 'followQualified': False})
      if kind in ('extra_stale', 'formal_stale'):
        lead.evidence.positionMonoTime -= 500_000_000
      elif kind == 'extra_future':
        lead.evidence.pathState, lead.evidence.conflictStart = 'entering', .02
      elif kind == 'extra_caution':
        lead = caution_lead(now_ns=stamp, dRel=.8, vLead=0.0, vLeadK=0.0, vRel=-10.0)
      elif kind == 'extra_malformed':
        lead.dRel = float('nan')
      objects.append(lead)
    slots = tuple(objects) if kind == 'formal_stale' else ()
    state['radarState'] = native_frame(radar_frame(*objects, publication=stamp, display=slots))
    if kind == 'solver':
      planner.mpc.solver = Mock(wraps=planner.mpc.solver)
      planner.mpc.solver.solve.return_value = 4
    plan, commands = consume(planner, controller, state, frame)
  return planner, controller, state, plan, commands


@pytest.mark.parametrize('kind', ['extra_imminent', 'extra_stale', 'extra_future', 'extra_caution', 'extra_malformed'])
def test_extra_objects_never_create_failure_braking_fcw_or_control_influence(kind):
  _, _, _, clear_plan, clear_commands = failure_request('clear')
  planner, _, _, plan, commands = failure_request(kind)
  assert plan.solverStatus == clear_plan.solverStatus == 0
  assert plan.motionIntent.valid and clear_plan.motionIntent.valid
  assert planner.mpc.source_object_identity is None and not planner.fcw
  assert not planner.reference_planner.radar_consumption.degraded
  assert plan.aTarget == pytest.approx(clear_plan.aTarget)
  assert commands == pytest.approx(clear_commands)
  assert not plan.objectInfluences


@pytest.mark.parametrize('kind', ['solver', 'formal_stale'])
def test_ordinary_failure_reaches_owner_and_can_without_extra_urgent_authority(kind):
  planner, controller, state, plan, commands = failure_request(kind)
  assert planner.mpc.solution_status != 0 and not plan.motionIntent.valid
  assert plan.aTarget == MPC_FAILURE_ACCEL and plan.motionIntent.brakingAccel == MPC_FAILURE_ACCEL
  assert MotionIntent.from_message(plan.motionIntent).is_braking_valid()
  assert controller.motion_supervisor.degraded and controller.motion_supervisor.reasons & StopReason.INVALID
  assert min(commands) >= MPC_FAILURE_ACCEL and max(commands) <= 0.0
  assert not plan.objectInfluences
  control = state['carControl'].as_reader().as_builder()
  control.actuators.accel = commands[-1]
  control.actuators.longControlState = controller.long_control_state
  control.cruiseControl.resume = False
  values = can_from_car_control(control.as_reader(), state['carState'])
  assert values['ACCEL_COMMAND'] == pytest.approx(commands[-1], abs=.01)
  assert values['GAS_COMMAND'] <= 0


def test_obsolete_finite_risk_attributes_cannot_strengthen_an_ordinary_failure():
  planner, _, state, _, _ = failure_request('solver')
  planner.mpc.finite_risk_braking_accel = ACCEL_MIN
  planner.mpc.finite_risk_evidence = (SimpleNamespace(consumed_ns=planner.reference_planner._radar_update_ns),)
  planner.reference_planner.continuation.begin()
  planner.reference_planner.fail_mpc(state, 10.0, .2)
  assert planner.output_a_target == MPC_FAILURE_ACCEL


@pytest.mark.parametrize('independent', ['previous', 'model_tied', 'model_stronger'])
def test_ordinary_failure_retains_a_stronger_previous_or_model_braking_request(independent):
  planner, _, state, _, _ = failure_request('solver')
  previous = ACCEL_MIN if independent == 'previous' else .2
  state['modelV2'].action.desiredAcceleration = (ACCEL_MIN - .2 if independent == 'model_stronger'
                                                else ACCEL_MIN if independent == 'model_tied' else .2)
  with controlled_mode(planner):
    planner.reference_planner.fail_mpc(state, 10.0, previous)
  assert planner.output_a_target == ACCEL_MIN
