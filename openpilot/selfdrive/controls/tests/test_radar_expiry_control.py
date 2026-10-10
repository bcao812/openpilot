"""New expiry handling keeps a moving car's existing failure braking path."""
from unittest.mock import patch

import pytest

from opendbc.car.honda.values import CarControllerParams
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_planner_owner
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import CapturingPublisher
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_radar_consumption import PUBLICATION, coast, measured, native_frame, radar_frame
from openpilot.selfdrive.controls.tests.test_radard_publication import make_radard, publish_at, update_with_result
from openpilot.selfdrive.controls.tests.test_unified_radar_consumption import state_at


@pytest.mark.parametrize('boundary,source', [
  ('consumer', 'radar'), ('consumer', 'fused'), ('consumer', 'predicted'),
  ('late_publication', 'radar'), ('late_publication', 'fused'), ('late_publication', 'predicted'),
  ('publisher', 'radar'), ('publisher', 'fused'), ('publisher', 'predicted'),
])
def test_expired_near_obstacle_does_not_turn_ongoing_braking_into_cruise(boundary, source):
  for name in ('AlphaLongitudinalEnabled', 'ExperimentalMode', 'DynamicExperimentalControl'):
    Params().put_bool(name, True, block=True)
  planner, owner = make_planner_owner(speed=10.0, initial_accel=-1.0)
  lead = coast(relevant=True) if source == 'predicted' else measured(source=source, relevant=True)
  lead.dRel = 25.0
  frame = radar_frame(lead, display=(lead,))
  rd = make_radard()
  update_with_result(rd, frame)

  def step(now):
    state = state_at(frame, speed=10.0, target=22.0, now=now)
    state['carState'].aEgo = -1.0
    if boundary == 'publisher':
      event = publish_at(rd, now - 1_000_000)
      state['radarState'], state.logMonoTime['radarState'] = event.radarState, event.logMonoTime
    else:
      state['radarState'] = native_frame(frame)
      if boundary == 'late_publication':
        state.logMonoTime['radarState'] = now - 1_000_000
    # The visual model asks for forward motion and supplies no visual lead.
    assert state['modelV2'].action.desiredAcceleration > 0.0
    assert not state['radarState'].visionLeadOne.present
    with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
      planner.update(state, now_ns=now)
    publisher = CapturingPublisher()
    with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=now + 1_000_000):
      planner.publish(state, publisher)
    event = publisher.messages['longitudinalPlan']
    plan = event.longitudinalPlan
    commands = [owner.update(
      True, state['carState'], plan.aTarget, plan.shouldStop,
      (CarControllerParams.BOSCH_ACCEL_MIN, CarControllerParams.BOSCH_ACCEL_MAX),
      intent=MotionIntent.from_message(plan.motionIntent), plan_time_ns=event.logMonoTime,
      model_time_ns=plan.modelMonoTime, now_ns=now + 2_000_000 + tick * 10_000_000,
      plan_valid=event.valid, solver_status=plan.solverStatus, braking_plan_valid=True,
    ) for tick in range(3)]
    return plan, commands

  before, first_commands = step(PUBLICATION + 20_000_000)
  assert before.solverStatus == 0 and before.aTarget < 0.0
  assert max(first_commands) < 0.0
  after, commands = step(PUBLICATION + 60_000_000)
  assert after.solverStatus != 0 and not after.motionIntent.valid
  assert after.aTarget <= before.aTarget and after.motionIntent.brakingAccel <= before.aTarget
  assert max(commands) <= min(0.0, first_commands[-1])
  assert not after.motionIntent.releaseAllowed
  assert planner.reference_planner.radar_held_identity is None
  expiry = rd.publication_expiry if boundary == 'publisher' else planner.reference_planner.radar_consumption
  assert expiry.degraded
