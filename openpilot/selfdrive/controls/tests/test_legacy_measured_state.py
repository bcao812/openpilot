"""Physical-state identity is independent of reference and perception thresholds."""

import numpy as np
import pytest

from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import MPC_INVALID_INPUT
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.tests.test_longitudinal_cruise_coast import decoded_commands
from openpilot.selfdrive.controls.tests.test_supervised_longitudinal_planner import make_supervised, make_supervised_state, consume
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_planner import advance


@pytest.fixture(autouse=True)
def isolated_params():
  with OpenpilotPrefix():
    yield


def measured_case(supervised, speed=20.0):
  planner, owner = make_supervised(speed=speed, initial_accel=-1.5)
  if not supervised:
    planner = LongitudinalPlanner(planner.CP, owner.CP_SP, init_v=speed, init_a=-1.5)
    owner = LongControl(planner.CP, owner.CP_SP)
  planner.v_desired_filter.x = speed - 1.0
  state = make_supervised_state(speed=speed)
  state['carState'].aEgo = -0.65
  return planner, owner, state


@pytest.mark.parametrize('supervised', [False, True])
def test_perception_boundary_jitter_does_not_switch_physical_initial_state(supervised):
  results = []
  for jitter in (False, True):
    planner, _, state = measured_case(supervised)
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


@pytest.mark.parametrize('supervised', [False, True])
def test_measured_initial_state_preserves_independent_reference_history(supervised):
  planner, _, state = measured_case(supervised)
  planner.update(state, now_ns=advance(state, 0))
  assert planner.mpc.solution_status == 0
  previous_reference = planner.mpc.a_prev.copy()
  assert np.any(previous_reference != 0.0)
  state['carState'].aEgo = -0.4
  planner.update(state, now_ns=advance(state, 1))
  assert planner.mpc.solution_status == 0
  np.testing.assert_array_equal(planner.mpc.x0[1:], [state['carState'].vEgo, state['carState'].aEgo])
  np.testing.assert_array_equal(planner.mpc.params[:, 3], previous_reference)
  assert planner.v_desired_filter.x != planner.mpc.x0[1]


@pytest.mark.parametrize('supervised', [False, True])
@pytest.mark.parametrize('fault', ['speed_nan', 'acceleration_nan', 'acceleration_inf', 'unhealthy'])
def test_unavailable_motion_uses_failure_and_retains_braking(supervised, fault):
  planner, owner, state = measured_case(supervised)
  owner.last_output_accel = -1.5
  if fault == 'unhealthy':
    state.all_checks = lambda service_list=None: 'carState' not in (service_list or state)
  elif fault == 'speed_nan':
    state['carState'].vEgo = float('nan')
  else:
    state['carState'].aEgo = float('nan') if fault.endswith('nan') else float('inf')
  if supervised:
    plan, outputs = consume(planner, owner, state, 0)
    assert not plan.motionIntent.valid and owner.motion_degraded
  else:
    planner.update(state, now_ns=advance(state, 0))
    assert planner.mpc.solution_status == MPC_INVALID_INPUT
    outputs = [owner.update(True, state['carState'], planner.output_a_target, planner.output_should_stop,
                            (-3.5, 2.0), solver_status=planner.mpc.solution_status)]
  assert all(np.isfinite(output) and output <= -1.5 for output in outputs)
  assert all(command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0.0 for command in decoded_commands(outputs))
