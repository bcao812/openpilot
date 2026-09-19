"""Physical-state identity is independent of reference and perception thresholds."""

import numpy as np
import pytest

from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import decoded_commands
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import advance, consume, make_handover_state, make_planner_owner


def measured_case(speed=20.0):
  planner, owner = make_planner_owner(speed=speed, initial_accel=-1.5)
  planner.v_desired_filter.x = speed - 1.0
  state = make_handover_state(speed=speed)
  state['carState'].aEgo = -0.65
  return planner, owner, state


def test_perception_boundary_jitter_does_not_switch_physical_initial_state():
  results = []
  for jitter in (False, True):
    planner, _, state = measured_case()
    lead = state['radarState'].leadOne
    lead.present, lead.modelProb, lead.aLeadTau = True, 0.0, 1.5
    targets, gap = [], 70.0
    for frame in range(40):
      elapsed = frame * 0.05
      speed = 20.0 - 0.65 * elapsed
      lead_speed = 8.0 - 3.0 * elapsed
      state['carState'].vEgo, state['carState'].aEgo = speed, -0.65
      lead.dRel, lead.vLead, lead.vRel, lead.aLeadK = gap, lead_speed, lead_speed - speed, -3.0
      lead.yRel = 1.81 if jitter and frame % 2 else 1.79
      planner.update(state, now_ns=advance(state, frame))
      assert planner.mpc.solution_status == 0
      np.testing.assert_array_equal(planner.mpc.x0[1:], [state['carState'].vEgo, state['carState'].aEgo])
      targets.append(planner.output_a_target)
      gap += (lead_speed - speed) * 0.05
    results.append(targets)
  np.testing.assert_allclose(results[0], results[1], atol=1e-6, rtol=0.0)


def test_measured_initial_state_preserves_independent_reference_history():
  planner, _, state = measured_case()
  planner.update(state, now_ns=advance(state, 0))
  assert planner.mpc.solution_status == 0
  previous_reference = planner.mpc.a_prev.copy()
  assert np.any(previous_reference != 0.0)
  state['carState'].aEgo = -0.4
  planner.v_desired_filter.x = 0.0
  planner.update(state, now_ns=advance(state, 1))
  assert planner.mpc.solution_status == 0
  np.testing.assert_array_equal(planner.mpc.x0[1:], [state['carState'].vEgo, state['carState'].aEgo])
  np.testing.assert_array_equal(planner.mpc.params[:, 3], previous_reference)
  assert np.any(previous_reference != state['carState'].aEgo)
  assert planner.v_desired_filter.x == planner.mpc.x0[1]


@pytest.mark.parametrize('fault', ['speed_nan', 'acceleration_nan', 'acceleration_inf', 'unhealthy'])
def test_unavailable_motion_uses_failure_and_retains_braking(fault):
  planner, owner, state = measured_case()
  owner.last_output_accel = -1.5
  if fault == 'unhealthy':
    state.all_checks = lambda service_list=None: 'carState' not in (service_list or state)
  elif fault == 'speed_nan':
    state['carState'].vEgo = float('nan')
  else:
    state['carState'].aEgo = float('nan') if fault.endswith('nan') else float('inf')
  plan, outputs = consume(planner, owner, state, 0)
  assert not plan.motionIntent.valid and owner.motion_degraded
  assert all(np.isfinite(output) and output <= -1.5 for output in outputs)
  assert all(command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0.0 for command in decoded_commands(outputs))
