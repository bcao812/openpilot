from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.cereal import log
from openpilot.common.hardware.hw import Paths
from openpilot.common.longitudinal_logging import observe
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import N, T_IDXS
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner, PASSIVE_PITCH_VERSION
from openpilot.selfdrive.controls.lib.longitudinal_trace import LongitudinalTrace
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import reference_requests_departure
from openpilot.selfdrive.controls.tests.test_longitudinal_observation_planner import diagnostics, reference_snapshot
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_state


MODES = (LongitudinalMode.UNIFIED,)


@pytest.fixture(autouse=True)
def isolated_params(tmp_path, monkeypatch):
  monkeypatch.setenv('PARAMS_ROOT', str(tmp_path / 'params'))
  with patch.object(Paths, 'comma_home', return_value=str(tmp_path / 'comma')):
    with OpenpilotPrefix():
      yield


def planner_for(mode, *, enabled=True, speed=20.0, trace=None):
  params = CarInterface.get_non_essential_params(CAR.HONDA_ACCORD)
  params.openpilotLongitudinalControl = True
  extension = CarInterface.get_non_essential_params_sp(params, CAR.HONDA_ACCORD)
  return LongitudinalPlanner(params, extension, init_v=speed, diagnostics=enabled, trace=trace)


def state_for(mode, *, speed=20.0, target=22.0, mono=1_000_000_000):
  state = make_state(speed=speed, target=target)
  state['controlsState'].motionState.planningMode = int(mode)
  state['carControl'].orientationNED = [0.0, 0.02, 0.0]
  for service in state.logMonoTime:
    state.logMonoTime[service] = mono
  return state


def snapshot(planner):
  result = {name: getattr(planner, name) for name in ('output_a_target', 'output_should_stop', 'a_cruise', 'allow_throttle', 'fcw')}
  result.update(
    velocity=planner.v_desired_filter.x, pitch=planner.pitch_bias.filter.x, initialized=planner.pitch_bias.filter.initialized,
    pitch_calls=planner.pitch_bias.diagnostic_calls, learn_calls=planner.pitch_bias.diagnostic_learn_calls,
    intent=planner.motion_intent, reference=reference_snapshot(planner.reference_planner),
    status=planner.mpc.solution_status, source=planner.mpc.source,
  )
  for name in ('x0', 'params', 'x_sol', 'u_sol', 'a_prev', 'yref', 'v_solution', 'a_solution', 'j_solution'):
    result[name] = getattr(planner.mpc, name).tolist()
  return result


@pytest.mark.parametrize('mode', MODES)
def test_passive_analysis_records_exact_consumer_clock_and_owned_watermarks(mode):
  planner = planner_for(mode)
  mono = (1 << 60) + 27
  now = mono + 5_000_003
  state = state_for(mode, mono=mono)
  state.logMonoTime['carControl'] -= 1
  with patch.object(planner.pitch_bias.filter, 'update', wraps=planner.pitch_bias.filter.update) as learn:
    planner.update(state, now_ns=now)
  analysis = diagnostics(planner, state).analysis
  assert analysis.version == 1 and analysis.planningMode == int(mode)
  assert analysis.decisionMonoTimeKnown and analysis.decisionMonoTime == now
  assert analysis.passiveApiVersion == PASSIVE_PITCH_VERSION
  assert analysis.passiveAttempted and analysis.passiveCompleted
  assert analysis.passiveReasonKnown and analysis.passiveReason == 'accepted'
  assert analysis.passiveBefore.known and not analysis.passiveBefore.pitchInitialized
  assert analysis.passiveBefore.consumerMonoTime == 0
  assert analysis.passiveAfter.known and analysis.passiveAfter.pitchInitialized
  assert analysis.passiveAfter.consumerMonoTime == now
  assert analysis.passiveAfter.carStateMonoTime == mono
  assert analysis.passiveAfter.carControlMonoTime == mono - 1
  assert analysis.passiveAfter.pitchFilterValue == state['carControl'].orientationNED[1]
  learn.assert_called_once()
  recorded = analysis.to_dict()
  planner._passive_now_ns += 123
  planner._passive_input_times['carControl'] += 234
  planner.pitch_bias.filter.x = 0.5
  assert diagnostics(planner, state).analysis.to_dict() == recorded


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('failure', ['duplicate', 'future', 'invalid'])
def test_rejected_passive_attempt_keeps_actual_reason_and_separate_id_state(mode, failure):
  planner = planner_for(mode)
  state = state_for(mode)
  planner.update(state, now_ns=1_005_000_000)
  old_value = planner.pitch_bias.filter.x
  if failure != 'duplicate':
    for service in state.logMonoTime:
      state.logMonoTime[service] += 50_000_000
  now = 1_055_000_000
  if failure == 'future':
    state.logMonoTime['carControl'] = now + 1
  elif failure == 'invalid':
    state.valid['carControl'] = False
  planner.update(state, now_ns=now)
  observed = diagnostics(planner, state)
  expected = {'duplicate': 'repeated_or_out_of_order_measurement', 'future': 'future_measurement', 'invalid': 'invalid_measurement'}
  assert observed.analysis.passiveAttempted and observed.analysis.passiveCompleted
  assert observed.analysis.passiveReasonKnown and observed.analysis.passiveReason == expected[failure]
  assert observed.analysis.passiveAfter.consumerMonoTime == now
  assert observed.analysis.passiveAfter.carControlMonoTime == (1_050_000_000 if failure == 'invalid' else 1_000_000_000)
  assert observed.analysis.passiveAfter.pitchFilterValue == old_value
  assert not observed.pitchCalled and not observed.pitchLearned


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('now,known', [(True, False), (1.5, False), (-1, False), (0, True)])
def test_invalid_consumer_clock_is_not_rounded_or_replaced_by_observer_time(mode, now, known):
  planner = planner_for(mode)
  state = state_for(mode)
  planner.update(state, now_ns=now)
  analysis = diagnostics(planner, state).analysis
  assert analysis.version == 1
  assert analysis.decisionMonoTimeKnown == known
  assert analysis.decisionMonoTime == 0
  assert analysis.passiveReason == 'invalid_consumer_time'
  assert analysis.passiveAfter.consumerMonoTime == 0


@pytest.mark.parametrize('mode', MODES)
def test_early_return_cannot_publish_last_frames_candidate_departure_or_obstacles(mode):
  planner = planner_for(mode)
  state = state_for(mode)
  planner.update(state, now_ns=1_005_000_000)
  first = diagnostics(planner, state)
  assert first.analysis.departure.known and first.analysis.obstacleSummary.known
  state['controlsState'].motionState.enabled = False
  for service in state.logMonoTime:
    state.logMonoTime[service] += 50_000_000
  planner.update(state, now_ns=1_055_000_000)
  after = diagnostics(planner, state)
  assert after.outcome == 'motion_owner_unavailable'
  assert after.analysis.passiveAttempted and after.analysis.passiveReason == 'accepted'
  assert not after.analysis.referenceCandidates.known
  assert not after.analysis.departure.known and not after.analysis.obstacleSummary.known
  assert not after.mpcUpdated and not after.solverAttempted


@pytest.mark.parametrize('mode', MODES)
def test_cold_standby_passive_learning_remains_visible_without_a_reference(mode):
  planner = planner_for(mode)
  state = state_for(mode)
  state['carState'].vCruise = V_CRUISE_UNSET
  state['carControl'].enabled = state['carControl'].longActive = state['selfdriveState'].enabled = False
  state['controlsState'].longControlState = LongCtrlState.off
  planner.update(state, now_ns=1_005_000_000)
  observed = diagnostics(planner, state)
  assert observed.outcome == 'cold_standby'
  assert observed.analysis.passiveReason == 'accepted' and observed.pitchLearned
  assert not observed.analysis.referenceCandidates.known and not observed.analysis.departure.known
  assert not observed.analysis.obstacleSummary.known and not observed.solverAttempted


@pytest.mark.parametrize('admitted,model_accel', [(False, 0.5), (False, -2.0), (True, -2.0)])
def test_unified_observation_keeps_reference_and_admitted_model_constraint_separate(admitted, model_accel):
  trace = LongitudinalTrace()
  planner = planner_for(LongitudinalMode.UNIFIED, speed=20.2, trace=trace)
  state = state_for(LongitudinalMode.UNIFIED, speed=20.2, target=20.2)
  state['modelV2'].action.desiredAcceleration = model_accel
  # Isolate the whole stateful cruise strategy at the physical upper ceiling.
  # This observer test checks the independent acceleration limit and admitted
  # model action; P's raw cruise history has separate differential coverage.
  with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.get_cruise_accel', side_effect=AssertionError('retired cruise executed')), \
       patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.sample_reference_acceleration', return_value=1.0) as sample, \
       patch.object(planner.reference_planner.cruise_response, 'update', return_value=2.0) as cruise, \
       patch.object(planner, 'is_e2e', return_value=admitted):
    planner.update(state, now_ns=1_005_000_000)
  sample.assert_called_once()
  cruise.assert_called_once()
  observed = diagnostics(planner, state)
  assert not observed.analysis.referenceCandidates.known
  assert observed.modelAdmissionKnown and observed.e2eAllowed == admitted
  assert observed.sampledAccelKnown and observed.sampledAccel == 1.0
  assert planner.reference_planner.reference_accel == 1.0
  # Straight-line cruise at 20.2 m/s has a 0.928 m/s^2 upper bound. An
  # admitted -2.0 model constraint is separate from the sampled MPC request.
  expected_ceiling = model_accel if admitted else 0.928
  assert observed.constraintKnown
  assert observed.constraintAccel == pytest.approx(expected_ceiling)
  assert planner.reference_planner.constraint_accel == pytest.approx(expected_ceiling)
  assert planner.output_a_target == pytest.approx(min(1.0, expected_ceiling))
  assert -3.5 <= planner.output_a_target <= 2.0
  assert trace.failures == 0 and trace.queue


@pytest.mark.parametrize('mode', MODES)
def test_departure_observation_reads_actual_predicate_once(mode):
  planner = planner_for(mode)
  state = state_for(mode)
  with patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.reference_requests_departure',
             wraps=reference_requests_departure) as departure:
    planner.update(state, now_ns=1_005_000_000)
  departure.assert_called_once()
  observed = diagnostics(planner, state).analysis.departure
  assert observed.known
  assert observed.terminalMpcSpeed == planner.mpc.v_solution[-1]
  assert planner.motion_intent.release_allowed == (observed.trajectoryDeparture or observed.leadDeparture)
  assert not observed.trajectoryDeparture or observed.forwardReference


def test_unified_obstacle_summary_uses_existing_bounded_solver_inputs_without_another_reference_or_solve():
  planner = planner_for(LongitudinalMode.UNIFIED)
  state = state_for(LongitudinalMode.UNIFIED)
  state['selfdriveState'].experimentalMode = True
  with patch.object(planner.mpc, 'observe_obstacles', wraps=planner.mpc.observe_obstacles) as capture, \
       patch.object(planner.mpc, 'reference_obstacles', wraps=planner.mpc.reference_obstacles) as reference, \
       patch.object(planner.mpc, 'solver', Mock(wraps=planner.mpc.solver)) as solver:
    planner.update(state, now_ns=1_005_000_000)
  capture.assert_called_once()
  reference.assert_called_once()
  solver.solve.assert_called_once()
  observed = diagnostics(planner, state).analysis
  assert not observed.referenceCandidates.known
  summary = observed.obstacleSummary
  matrix, sources = capture.call_args.args
  assert summary.known and len(summary.times) == len(summary.sources) == len(summary.distances) == N + 1 == 13
  assert list(summary.times) == list(T_IDXS)
  assert list(summary.distances) == list(planner.mpc.params[:, 2])
  assert [source.raw for source in summary.sources] == [sources[index] for index in np.argmin(matrix, axis=1)]
  assert reference.call_args.args[0].cruise_speed is None
  assert summary.sourceRule == 'first_minimum_by_mpc_source_order;finite_continuation_without_cruise_speed_goal'
  assert str(summary.sources[0]) == 'cruise'  # equal initial cruise/e2e obstacles use the existing column order


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('method', ['record_analysis', 'departure', 'passive_snapshot'])
def test_new_observer_failure_does_not_change_control_solver_or_learning(mode, method):
  enabled = planner_for(mode)
  disabled = planner_for(mode, enabled=False)
  state = state_for(mode)
  with patch.object(enabled.observation, method, side_effect=RuntimeError('observation failed')):
    enabled.update(state, now_ns=1_005_000_000)
  disabled.update(state, now_ns=1_005_000_000)
  assert snapshot(enabled) == snapshot(disabled)
  assert enabled.mpc.diagnostic_solve_calls == disabled.mpc.diagnostic_solve_calls == 1
  assert enabled.observation.recorder.errors > 0


@pytest.mark.parametrize('mode', MODES)
def test_native_summary_failure_is_unknown_but_does_not_cancel_solver(mode):
  enabled = planner_for(mode)
  disabled = planner_for(mode, enabled=False)
  state = state_for(mode)
  with patch.object(enabled.mpc, 'observe_obstacles', side_effect=RuntimeError('summary failed')):
    enabled.update(state, now_ns=1_005_000_000)
  disabled.update(state, now_ns=1_005_000_000)
  assert snapshot(enabled) == snapshot(disabled)
  observed = diagnostics(enabled, state)
  assert observed.solverAttempted and observed.solverReturnKnown and observed.solverReturnStatus == 0
  assert not observed.analysis.obstacleSummary.known and observed.context.errors > 0


@pytest.mark.parametrize('mode', MODES)
def test_original_passive_exception_still_propagates_with_incomplete_attempt(mode):
  planner = planner_for(mode)
  state = state_for(mode)
  with patch.object(planner.pitch_bias.filter, 'update', side_effect=RuntimeError('original learning failure')):
    with pytest.raises(RuntimeError, match='original learning failure'):
      planner.update(state, now_ns=1_005_000_000)
  observed = diagnostics(planner, state)
  assert not observed.updateCompleted and observed.outcome == 'control_exception'
  assert observed.analysis.passiveAttempted and not observed.analysis.passiveCompleted
  assert observed.analysis.passiveReasonKnown
  assert not observed.solverAttempted


@pytest.mark.parametrize('velocity', [[], None, 0.0])
def test_malformed_terminal_speed_cannot_grant_departure_or_escape_observation_isolation(velocity):
  enabled = planner_for(LongitudinalMode.UNIFIED)
  disabled = planner_for(LongitudinalMode.UNIFIED, enabled=False)
  state = state_for(LongitudinalMode.UNIFIED, speed=20.0)
  for planner in (enabled, disabled):
    planner.mpc.v_solution = velocity
    planner.output_a_target = 0.1
    planner.observation_attempt = observe(planner.observation, 'begin', planner, state, now_ns=1_005_000_000, passive_version=1)
    forward = reference_requests_departure(velocity, planner.mpc.a_solution, planner.mpc.j_solution)
    assert not forward
    observe(planner.observation, 'departure', planner.observation_attempt, forward, velocity, False, False, False, False)
    planner.observation_result = observe(planner.observation, 'finish', planner.observation_attempt, planner, True)
  assert enabled.motion_intent == disabled.motion_intent
  assert enabled.output_a_target == disabled.output_a_target == 0.1
  observed = diagnostics(enabled, state)
  assert observed.context.version == 1 and observed.context.errors > 0
  assert not observed.analysis.departure.known
  assert not enabled.motion_intent.release_allowed


def test_missing_analysis_is_unknown_and_does_not_change_motion_protocol_version():
  message = log.LongitudinalPlan.new_message()
  message.motionIntent.version = MOTION_INTENT_VERSION
  encoded = message.to_bytes()
  with log.LongitudinalPlan.from_bytes(encoded) as old:
    assert old.motionIntent.version == MOTION_INTENT_VERSION
    assert old.diagnostics.analysis.version == 0
    assert not old.diagnostics.analysis.passiveBefore.known
    assert not old.diagnostics.analysis.referenceCandidates.known
    assert not old.diagnostics.analysis.departure.known
    assert not old.diagnostics.analysis.obstacleSummary.known
