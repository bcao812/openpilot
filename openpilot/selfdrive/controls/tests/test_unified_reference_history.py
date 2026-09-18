from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, MpcReference
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_planner import advance, make_planner, make_state


@pytest.fixture(autouse=True)
def isolated_params():
  with OpenpilotPrefix():
    yield


def production_planner():
  params = CarInterface.get_non_essential_params(CAR.HONDA_ACCORD)
  params.openpilotLongitudinalControl = True
  extension = CarInterface.get_non_essential_params_sp(params, CAR.HONDA_ACCORD)
  extension.enableGasInterceptor = False
  return LongitudinalPlanner(params, extension, unified=True)


def first_active_reference(seed_speed, measured_accel, lead_distance=None):
  planner = make_planner(speed=20.0)[0] if seed_speed else production_planner()
  state = make_state(speed=20.0, target=15.0)
  state['carState'].aEgo = measured_accel
  if lead_distance is not None:
    lead = state['radarState'].leadOne
    lead.present = True
    lead.dRel = lead_distance
    lead.vRel = -20.0
    lead.vLead = 0.0
    lead.vLeadK = 0.0
    lead.aLeadK = 0.0
    lead.aLeadTau = 1.5
    lead.modelProb = 0.99
  with patch.object(planner.mpc, 'set_weights', wraps=planner.mpc.set_weights) as weights:
    planner.update(state, now_ns=advance(state, 0))
  assert planner.motion_intent.valid
  assert planner.mpc.solution_status == 0
  assert planner.mpc.x0[1] == pytest.approx(20.0)
  assert planner.mpc.x0[2] == pytest.approx(measured_accel)
  assert np.isfinite(planner.output_a_target)
  assert np.all(np.isfinite(planner.mpc.x_sol))
  assert not weights.call_args.args[0]
  return planner


@pytest.mark.parametrize('seed_speed', [False, True])
@pytest.mark.parametrize('measured_accel', [0.0, -1.0])
def test_first_overspeed_reference_uses_measured_state_and_plans_speed_recovery(seed_speed, measured_accel):
  planner = first_active_reference(seed_speed, measured_accel)
  # A speed error without a lead is a convergence/comfort scenario. The first
  # scalar request need not reproduce aEgo or a particular old braking frame.
  assert planner.mpc.v_solution[-1] < planner.mpc.x0[1]


@pytest.mark.parametrize('seed_speed', [False, True])
@pytest.mark.parametrize('measured_accel', [0.0, -1.0])
@pytest.mark.parametrize('lead_distance', [40.0, 20.0])
def test_first_stationary_lead_reference_retains_immediate_braking_response(seed_speed, measured_accel, lead_distance):
  planner = first_active_reference(seed_speed, measured_accel, lead_distance)
  # These 20 m/s / 20-40 m fixtures demand braking, and some cannot stop within
  # the measured gap. This is response/initialization evidence, not avoidance.
  assert planner.output_a_target < 0.0
  assert np.min(planner.mpc.v_solution[1:]) < planner.mpc.x0[1]


def test_native_failure_recovery_does_not_smooth_against_a_fabricated_zero_request():
  planner, _ = make_planner(speed=20.0)
  state = make_state(speed=20.0, target=15.0)
  state['carState'].aEgo = -1.0
  used_history = []
  requests = []
  for frame in range(25):
    state['carState'].vEgo = 20.0 - frame * DT_MDL
    state['carState'].vEgoRaw = state['carState'].vEgo
    with patch.object(planner.mpc, 'set_weights', wraps=planner.mpc.set_weights) as weights:
      if frame == 20:
        failing_solver = Mock(wraps=planner.mpc.solver)
        failing_solver.solve.return_value = 4
        with patch.object(planner.mpc, 'solver', failing_solver):
          planner.update(state, now_ns=advance(state, frame))
        assert planner.mpc.solution_status == 4
        assert not planner.motion_intent.valid
        assert not planner.mpc.previous_solution_valid
      else:
        planner.update(state, now_ns=advance(state, frame))
        assert planner.mpc.solution_status == 0
        assert planner.motion_intent.valid
        assert planner.output_a_target < 0.0
      used_history.append(weights.call_args_list[0].args[0])
    requests.append(planner.output_a_target)
    if frame == 21:
      assert planner.mpc.x0[1] == pytest.approx(state['carState'].vEgo)
  assert requests[20] <= requests[19]
  assert not used_history[0]
  assert all(used_history[1:21])
  assert not used_history[21]
  assert all(used_history[22:])


@pytest.mark.parametrize('interruption', ['invalid_motion', 'unhealthy_inputs', 'disabled', 'gas_override'])
def test_interrupted_reference_is_reinitialized_on_the_first_qualified_motion(interruption):
  planner = production_planner()
  state = make_state(speed=20.0, target=15.0)
  state['carState'].aEgo = -1.0
  planner.update(state, now_ns=advance(state, 0))
  if interruption == 'invalid_motion':
    state['carState'].aEgo = float('nan')
  elif interruption == 'unhealthy_inputs':
    state.all_checks = lambda **_: False
  elif interruption == 'disabled':
    state['selfdriveState'].enabled = False
    state['carControl'].enabled = False
    state['carControl'].longActive = False
    state['controlsState'].longControlState = LongCtrlState.off
  else:
    state['carState'].gasPressed = True
  planner.update(state, now_ns=advance(state, 1))
  state = make_state(speed=19.9, target=15.0)
  state['carState'].aEgo = -1.0
  with patch.object(planner.mpc, 'set_weights', wraps=planner.mpc.set_weights) as weights:
    planner.update(state, now_ns=advance(state, 2))
  assert planner.motion_intent.valid
  assert planner.mpc.solution_status == 0
  assert planner.mpc.x0[1] == pytest.approx(19.9)
  assert planner.output_a_target < 0.0
  assert not weights.call_args.args[0]


@pytest.mark.parametrize('failure', ['reset', 'solver_status', 'nonfinite_solution', 'invalid_input'])
def test_native_history_becomes_valid_only_after_a_finite_successful_solve(failure):
  mpc = LongitudinalMpc()
  reference = MpcReference(15.0)
  assert not mpc.previous_solution_valid
  mpc.set_cur_state(20.0, -1.0)
  mpc.set_weights(prev_accel_constraint=False)
  mpc.update(radar_state(), reference=reference)
  assert mpc.solution_status == 0
  assert mpc.previous_solution_valid
  if failure == 'reset':
    mpc.reset()
  elif failure == 'invalid_input':
    mpc.set_cur_state(float('nan'), -1.0)
    mpc.update(radar_state(), reference=reference)
  else:
    native_solver = mpc.solver
    failing_solver = Mock(wraps=native_solver)
    if failure == 'solver_status':
      failing_solver.solve.return_value = 4
    else:
      failing_solver.get.side_effect = lambda index, field: (np.full(3, np.nan) if field == 'x' and index == 1
                                                           else native_solver.get(index, field))
    with patch.object(mpc, 'solver', failing_solver):
      mpc.update(radar_state(), reference=reference)
  assert not mpc.previous_solution_valid
  np.testing.assert_array_equal(mpc.a_prev, 0.0)
  mpc.set_cur_state(19.9, -1.0)
  mpc.set_weights(prev_accel_constraint=False)
  mpc.update(radar_state(), reference=reference)
  assert mpc.solution_status == 0
  assert mpc.previous_solution_valid


def test_continuous_qualified_reference_keeps_smoothing_with_same_CS_initial_state():
  planner = production_planner()
  state = make_state(speed=20.0, target=25.0)
  planner.update(state, now_ns=advance(state, 0))
  state['carState'].vEgo = 20.2
  with patch.object(planner.mpc, 'set_weights', wraps=planner.mpc.set_weights) as weights:
    planner.update(state, now_ns=advance(state, 1))
  assert planner.motion_intent.valid
  # The same-CS measurement and qualified native history are distinct; the
  # unified planner uses the measured state without a second speed filter.
  assert planner.mpc.x0[1] == float(state['carState'].vEgo)
  assert planner.mpc.x0[2] == float(state['carState'].aEgo)
  assert weights.call_args.args[0]
