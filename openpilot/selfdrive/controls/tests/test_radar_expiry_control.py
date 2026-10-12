"""Age-only expiry replans from current inputs through the real owner."""
from unittest.mock import patch

import numpy as np
import pytest

from opendbc.car.honda.values import CarControllerParams
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_planner_owner
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import CapturingPublisher
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_radar_consumption import PUBLICATION, measured, native_frame, radar_frame
from openpilot.selfdrive.controls.tests.test_radard_publication import make_radard, publish_at, update_with_result
from openpilot.selfdrive.controls.tests.test_unified_radar_consumption import state_at


@pytest.mark.parametrize('boundary,source', [
  ('consumer', 'radar'), ('consumer', 'fused'),
  ('late_publication', 'radar'), ('late_publication', 'fused'),
  ('publisher', 'radar'), ('publisher', 'fused'),
])
@pytest.mark.parametrize('current_action', [0.5, -1.5])
def test_expired_near_obstacle_replans_and_preserves_current_model_braking(boundary, source, current_action):
  for name in ('AlphaLongitudinalEnabled', 'ExperimentalMode', 'DynamicExperimentalControl'):
    Params().put_bool(name, True, block=True)
  results = []
  for explicit_absence in (False, True):
    planner, owner = make_planner_owner(speed=10.0, initial_accel=-1.0)
    lead = measured(source=source, relevant=True)
    lead.dRel = 25.0
    frame = radar_frame(lead, display=(lead,))
    rd = make_radard()
    update_with_result(rd, frame)

    def step(now, action=0.5, *, absent=False):
      current = radar_frame() if absent else frame
      state = state_at(current, speed=10.0, target=22.0, now=now)
      state['carState'].aEgo = -1.0
      if boundary == 'publisher':
        if absent:
          update_with_result(rd, current)
        event = publish_at(rd, now - 1_000_000)
        state['radarState'], state.logMonoTime['radarState'] = event.radarState, event.logMonoTime
      else:
        state['radarState'] = native_frame(current)
        if boundary == 'late_publication':
          state.logMonoTime['radarState'] = now - 1_000_000
      # No current visual object is invented; independent current action is kept.
      state['modelV2'].action.desiredAcceleration = action
      assert not state['radarState'].visionLeadOne.present
      with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='blended'), \
           patch.object(planner, 'is_e2e', return_value=True):
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
    native_calls = planner.mpc.diagnostic_solve_calls
    after, commands = step(PUBLICATION + 60_000_000, current_action, absent=explicit_absence)
    assert planner.mpc.diagnostic_solve_calls == native_calls + 1
    assert after.solverStatus == 0 and after.motionIntent.valid
    assert not owner.motion_degraded and not after.continuation.valid
    assert not planner.mpc._source_leads
    expiry = rd.publication_expiry if boundary == 'publisher' else planner.reference_planner.radar_consumption
    assert bool(expiry.expired_identities) is (not explicit_absence)
    assert not expiry.degraded
    if current_action < 0.0:
      assert after.aTarget <= current_action and max(commands) <= 0.0
    else:
      assert not after.shouldStop
    results.append((after, commands, planner.mpc.params.copy(), planner.mpc.x_sol.copy()))

  retired, absent = results
  # Both arms have the same actual -1 m/s² state and previous braking plan.
  # Ordinary cruise/trajectory/PID smoothing may still brake; it must match
  # an explicitly empty current observation, not retain an old-object ceiling.
  assert retired[0].aTarget == pytest.approx(absent[0].aTarget, abs=1e-8)
  assert retired[0].reference.constraintAccel == pytest.approx(absent[0].reference.constraintAccel, abs=1e-8)
  assert retired[0].motionIntent.to_dict() == absent[0].motionIntent.to_dict()
  assert retired[1] == pytest.approx(absent[1], abs=1e-8)
  np.testing.assert_allclose(retired[2], absent[2], rtol=0.0, atol=1e-9)
  np.testing.assert_allclose(retired[3], absent[3], rtol=0.0, atol=1e-9)
