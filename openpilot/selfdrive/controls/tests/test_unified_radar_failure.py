"""Fresh imminent infeasibility preserves a proved braking-only CAN request."""
from unittest.mock import Mock

import pytest

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, MPC_FAILURE_ACCEL
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent, StopReason
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import consume, make_state
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_lead_qualification import caution_lead
from openpilot.selfdrive.controls.tests.test_mpc_failure_chain import can_from_car_control
from openpilot.selfdrive.controls.tests.test_radar_consumption import native_frame, radar_frame
from openpilot.selfdrive.controls.tests.test_radar_fusion_mpc import object_lead
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
    if kind in ('imminent', 'stale'):
      lead = object_lead(7, distance=.8, ego_speed=10.0, start=.02, end=.10,
                         path='entering', persistent=False, epoch=stamp, physical_window=(.02, .10))
      lead.radar, lead.radarTrackId = True, 7
      if kind == 'stale':
        lead.evidence.positionMonoTime -= 500_000_000
      objects.append(lead)
    elif kind == 'caution':
      objects.append(caution_lead(now_ns=stamp, dRel=.8, vLead=0.0, vLeadK=0.0, vRel=-10.0))
    session = objects[0].evidence.sessionId if objects else 'radar-session-a'
    state['radarState'] = native_frame(radar_frame(*objects, publication=stamp, session=session))
    if kind in ('solver', 'caution'):
      planner.mpc.solver = Mock(wraps=planner.mpc.solver)
      planner.mpc.solver.solve.return_value = 4
    plan, commands = consume(planner, controller, state, frame)
  return planner, controller, state, plan, commands


def test_fresh_imminent_unavoidable_geometry_reaches_owner_and_honda_can(record_property):
  planner, controller, state, plan, commands = failure_request('imminent')
  assert planner.mpc.solution_status != 0 and not plan.motionIntent.valid
  assert planner.mpc.finite_risk_braking_accel == ACCEL_MIN
  assert planner.mpc.finite_risk_evidence
  assert planner.reference_planner.mitigation_object_evidence == planner.mpc.finite_risk_evidence
  assert all(proof.consumed_ns == planner.reference_planner._radar_update_ns for proof in planner.mpc.finite_risk_evidence)
  assert plan.aTarget == ACCEL_MIN and plan.motionIntent.brakingAccel == ACCEL_MIN
  assert len(plan.objectInfluences) == 1
  influence = plan.objectInfluences[0]
  assert influence.version == 2 and influence.kind == 'braking' and influence.strength == 1.0
  assert influence.sessionId == 'radar-test' and influence.trackId == 7
  assert MotionIntent.from_message(plan.motionIntent).is_braking_valid()
  assert controller.motion_supervisor.degraded and controller.motion_supervisor.reasons & StopReason.INVALID
  assert controller.motion_supervisor.degraded_braking_accel == ACCEL_MIN
  assert max(commands) <= ACCEL_MIN
  control = state['carControl'].as_reader().as_builder()
  control.actuators.accel = commands[-1]
  control.actuators.longControlState = controller.long_control_state
  control.cruiseControl.resume = False
  values = can_from_car_control(control.as_reader(), state['carState'])
  assert values['ACCEL_COMMAND'] == pytest.approx(ACCEL_MIN, abs=.01)
  assert values['BRAKE_REQUEST'] == 1 and values['GAS_COMMAND'] <= 0
  record_property('finite_risk_braking_request', float(plan.aTarget))
  record_property('honda_acceleration_command', float(values['ACCEL_COMMAND']))


@pytest.mark.parametrize('kind', ['solver', 'stale', 'caution'])
def test_ordinary_failure_or_unqualified_objects_cannot_inherit_urgent_braking(kind):
  planner, controller, state, plan, commands = failure_request(kind)
  assert plan.solverStatus != 0 and not plan.motionIntent.valid
  assert planner.mpc.finite_risk_braking_accel is None and not planner.mpc.finite_risk_evidence
  assert planner.reference_planner.mitigation_object_evidence == ()
  assert plan.aTarget == MPC_FAILURE_ACCEL and plan.motionIntent.brakingAccel == MPC_FAILURE_ACCEL
  assert min(commands) >= MPC_FAILURE_ACCEL


def test_old_finite_proof_cannot_strengthen_a_new_ordinary_failure():
  planner, _, state, _, _ = failure_request('imminent')
  old_evidence = planner.mpc.finite_risk_evidence
  assert old_evidence
  planner.reference_planner._radar_update_ns += 50_000_000
  # A new planner tick clears its previous independent request before handling
  # the new native outcome. Keep only the deliberately stale finite proof.
  planner.reference_planner.continuation.begin()
  planner.reference_planner.fail_mpc(state, 10.0, .2)
  assert planner.mpc.finite_risk_evidence == old_evidence
  assert planner.output_a_target == MPC_FAILURE_ACCEL
  assert planner.reference_planner.mitigation_object_evidence == ()


@pytest.mark.parametrize('independent', ['previous', 'model_tied', 'model_stronger'])
def test_finite_risk_receipt_does_not_claim_an_independently_owned_stronger_request(independent):
  planner, _, state, _, _ = failure_request('imminent')
  previous = ACCEL_MIN if independent == 'previous' else .2
  state['modelV2'].action.desiredAcceleration = (ACCEL_MIN - .2 if independent == 'model_stronger'
                                                else ACCEL_MIN if independent == 'model_tied' else .2)
  with controlled_mode(planner):
    planner.reference_planner.fail_mpc(state, 10.0, previous)
  assert planner.output_a_target == ACCEL_MIN
  assert planner.reference_planner.mitigation_object_evidence == ()
