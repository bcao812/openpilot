from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.cereal import log, messaging
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.longitudinal_trace import LongitudinalTrace, MOTION_TRACE_SCHEMA, UNIFIED_TRACE_FIELDS
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION, MotionIntent, StopReason
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import model_reference_from_velocity
from openpilot.selfdrive.controls.tests.test_longitudinal_following_scenarios import simulate_following
from openpilot.selfdrive.controls.tests.test_longitudinal_planner_state import braking_state
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanSource as SPSource


@pytest.fixture(autouse=True)
def isolated_params():
  with OpenpilotPrefix():
    yield


def make_planner(speed=20.0, *, enabled=True, platform=CAR.HONDA_ACCORD, stock=False, interceptor=False, trace=None):
  params = CarInterface.get_non_essential_params(platform)
  params.openpilotLongitudinalControl = not stock
  extension = CarInterface.get_non_essential_params_sp(params, platform)
  extension.enableGasInterceptor = interceptor
  return LongitudinalPlanner(params, extension, init_v=speed, unified=enabled, trace=trace), extension


def make_state(speed=20.0, target=22.0, experimental=False):
  state = braking_state(speed)
  for service in ('modelV2', 'selfdriveState'):
    state[service] = state[service].as_builder()
  state['carState'].vCruise = target * 3.6
  state['carState'].vCruiseCluster = target * 3.6
  state['carState'].standstill = speed == 0.0
  state['carState'].aEgo = 0.0
  state['carControl'].enabled = True
  state['carControl'].longActive = True
  state['controlsState'].motionState.version = MOTION_INTENT_VERSION
  state['controlsState'].motionState.planningMode = int(LongitudinalMode.UNIFIED)
  state['controlsState'].motionState.enabled = True
  state['selfdriveState'].enabled = True
  state['selfdriveState'].experimentalMode = experimental
  model = state['modelV2']
  model.position.x = [float(speed * sample) for sample in ModelConstants.T_IDXS]
  model.velocity.x = [float(speed)] * len(ModelConstants.T_IDXS)
  model.acceleration.x = [0.0] * len(ModelConstants.T_IDXS)
  model.orientation.x = [0.0] * len(ModelConstants.T_IDXS)
  model.orientationRate.z = [0.001] * len(ModelConstants.T_IDXS)
  model.action.desiredAcceleration = 0.5
  model.meta.disengagePredictions.gasPressProbs = [1.0] * 6
  for service in state.logMonoTime:
    state.logMonoTime[service] = 1_000_000_000
  return state


def advance(state, frame):
  timestamp = 1_000_000_000 + frame * 50_000_000
  for service in state.logMonoTime:
    state.logMonoTime[service] = timestamp
  return timestamp


class Publisher:
  def __init__(self):
    self.messages = {}

  def send(self, service, message):
    self.messages[service] = message.as_reader()


def test_candidate_is_default_off_and_platform_gated():
  assert not Params().get_bool('LongitudinalUnifiedEnabled')
  for platform, stock, interceptor in (
    (CAR.HONDA_CIVIC, False, False),
    (CAR.HONDA_CIVIC_BOSCH, False, False),
    (CAR.HONDA_ACCORD, True, False),
    (CAR.HONDA_ACCORD, False, True),
  ):
    planner, extension = make_planner(platform=platform, stock=stock, interceptor=interceptor)
    assert not planner.unified
    assert not LongControl(planner.CP, extension, unified=True).unified


def test_candidate_does_not_allocate_legacy_stop_owners():
  planner, extension = make_planner()
  assert planner.unified
  assert LongControl(planner.CP, extension, unified=True).unified
  for name in ('stop_latched', 'creep_t', 'creep_stop', 'close_lead_stop', 'moving_lead_t'):
    assert not hasattr(planner, name)


def test_new_pipeline_does_not_execute_independent_cruise_controller():
  planner, extension = make_planner()
  with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.get_cruise_accel', side_effect=AssertionError('legacy cruise executed')):
    planner.update(make_state())
  assert planner.motion_intent.valid
  assert planner.motion_intent.version == MOTION_INTENT_VERSION
  assert planner.mpc.source == log.LongitudinalPlan.LongitudinalPlanSource.cruise


def test_real_scc_producer_does_not_overwrite_estimated_initial_acceleration():
  Params().put_bool('SmartCruiseControlVision', True)
  planner, extension = make_planner()
  state = make_state(target=30.0)
  state['carState'].aEgo = 0.35
  state['modelV2'].orientationRate.z = [0.15] * len(ModelConstants.T_IDXS)
  for frame in range(4):
    planner.update(state, now_ns=advance(state, frame))
  assert planner.source == SPSource.sccVision
  assert planner.cruise_target.acceleration < 0.0
  assert planner.mpc.x0[2] == pytest.approx(0.35)
  assert planner.motion_intent.valid


@pytest.mark.parametrize('experimental,dec_enabled,mode,allowed', [
  (False, False, 'acc', False), (False, True, 'blended', False),
  (True, False, 'acc', True), (True, True, 'acc', False), (True, True, 'blended', True),
])
def test_model_action_admission_is_not_silently_widened(experimental, dec_enabled, mode, allowed):
  planner, extension = make_planner()
  state = make_state(experimental=experimental)
  state['modelV2'].action.desiredAcceleration = -3.0
  with patch.object(planner.dec, 'active', return_value=dec_enabled), \
       patch.object(planner.dec, 'mode', return_value=mode), \
       patch.object(planner.dec, 'update', return_value=None):
    planner.update(state)
  assert planner.reference_planner.e2e_allowed == allowed
  assert planner.motion_intent.valid
  if allowed:
    assert planner.output_a_target <= -3.0
    assert planner.reference_planner.reference_overridden
  else:
    assert planner.output_a_target > -1.0


def test_stopped_hold_request_is_not_used_as_motion_initial_acceleration():
  planner, extension = make_planner(speed=0.0)
  state = make_state(speed=0.0, target=0.0)
  state['controlsState'].longControlState = LongCtrlState.stopping
  state['carControl'].actuators.accel = -2.0
  planner.output_a_target = -2.0
  planner.update(state)
  assert planner.mpc.x0[2] == 0.0
  assert planner.motion_intent.reasons & int(StopReason.CRUISE)


def test_admitted_model_stop_is_a_planning_goal_not_only_a_boolean():
  planner, extension = make_planner()
  state = make_state(experimental=True)
  state['modelV2'].action.shouldStop = True
  state['modelV2'].action.desiredAcceleration = 0.5
  for frame in range(10):
    planner.update(state, now_ns=advance(state, frame))
  assert planner.motion_intent.valid
  assert planner.motion_intent.reasons & int(StopReason.E2E)
  assert planner.output_a_target < 0.0


@pytest.mark.parametrize('version,consumer_enabled', [
  (0, False), (0, True), (MOTION_INTENT_VERSION, False), (1, True), (2, True), (MOTION_INTENT_VERSION + 1, True),
])
def test_missing_or_mismatched_consumer_ack_does_not_release_legacy_hold(version, consumer_enabled):
  planner, extension = make_planner(speed=0.0)
  state = make_state(speed=0.0, target=20.0)
  state['controlsState'].motionState.version = version
  state['controlsState'].motionState.enabled = consumer_enabled
  state['controlsState'].longControlState = LongCtrlState.stopping
  planner.update(state)
  assert not planner.motion_intent.valid
  assert planner.output_a_target <= -0.2
  assert planner.output_should_stop


@pytest.mark.parametrize('attribute', ['vEgo', 'aEgo', 'vCruise'])
@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_motion_input_invalidates_intent_without_positive_request(attribute, value):
  planner, extension = make_planner()
  state = make_state()
  setattr(state['carState'], attribute, value)
  planner.update(state)
  assert not planner.motion_intent.valid
  assert np.isfinite(planner.output_a_target)
  assert planner.output_a_target <= -0.2


def test_solver_failure_is_invalid_not_a_successful_zero_trajectory():
  planner, extension = make_planner()
  state = make_state()
  failing_solver = Mock(wraps=planner.mpc.solver)
  failing_solver.solve.return_value = 4
  with patch.object(planner.mpc, 'solver', failing_solver):
    planner.update(state)
  assert planner.mpc.solution_status == 4
  assert not planner.motion_intent.valid
  publisher = Publisher()
  planner.publish(state, publisher)
  assert not publisher.messages['longitudinalPlan'].valid


def test_reference_and_intent_round_trip_preserve_distinct_semantics():
  planner, extension = make_planner()
  state = make_state(experimental=True)
  state['modelV2'].action.desiredAcceleration = -3.0
  planner.update(state)
  publisher = Publisher()
  planner.publish(state, publisher)
  message = publisher.messages['longitudinalPlan']
  with log.Event.from_bytes(message.as_builder().to_bytes()) as decoded:
    intent = MotionIntent.from_message(decoded.longitudinalPlan.motionIntent)
    assert intent == planner.motion_intent
    assert decoded.longitudinalPlan.reference.accel == pytest.approx(planner.reference_planner.reference_accel)
    assert decoded.longitudinalPlan.reference.overridden
    assert decoded.longitudinalPlan.aTarget <= -3.0
    assert list(decoded.longitudinalPlan.accels) == pytest.approx(planner.a_desired_trajectory)


def test_old_plan_has_zero_intent_version():
  message = messaging.new_message('longitudinalPlan').longitudinalPlan
  assert message.motionIntent.version == 0
  assert not MotionIntent.from_message(message.motionIntent).is_valid()


def test_trace_is_output_neutral_and_marks_unified_schema():
  events = []
  recorder = LongitudinalTrace(emit=lambda *args, **kwargs: events.append(kwargs))
  baseline, extension = make_planner()
  instrumented, extension = make_planner(trace=recorder)
  for frame in range(20):
    state = make_state()
    now_ns = advance(state, frame)
    baseline.update(state, now_ns=now_ns)
    instrumented.update(state, now_ns=now_ns)
    assert baseline.output_a_target == instrumented.output_a_target
    assert baseline.motion_intent == instrumented.motion_intent
    recorder.publish(state.logMonoTime['modelV2'] + 1)
  recorder._emit(list(recorder.queue))
  assert events[0]['schema'] == MOTION_TRACE_SCHEMA == 3
  assert events[0]['fields'] == UNIFIED_TRACE_FIELDS
  assert all(len(sample) == len(UNIFIED_TRACE_FIELDS) for sample in events[0]['samples'])
  assert events[0]['failures'] == 0


@pytest.mark.parametrize('target', [0.0, 15.0, 25.0])
def test_real_cruise_targets_reach_the_optimizer(target):
  planner, extension = make_planner()
  state = make_state(target=target)
  for frame in range(20):
    planner.update(state, now_ns=advance(state, frame))
  assert planner.motion_intent.valid
  assert -3.5 <= planner.output_a_target <= 2.0
  if target < 20.0:
    assert planner.output_a_target < 0.0
  else:
    assert planner.output_a_target > 0.0


@pytest.mark.parametrize('delay,response_time', [(0.1, 0.2), (0.3, 0.4), (0.5, 0.4), (0.8, 0.6)])
def test_new_following_chain_keeps_existing_behavior_gates(delay, response_time):
  diagnostics = []
  result = simulate_following(unified=True, delay=delay, response_time=response_time, diagnostics=diagnostics)
  assert result['stopping_seconds'] == 0.0
  assert result['minimum_speed'] > 0.3
  assert result['speed_error'] < 0.15
  assert result['minimum_gap'] > 5.0
  assert result['conflicts'] == 0
  assert all(frame['intent_version'] == MOTION_INTENT_VERSION for frame in diagnostics)
  assert any(frame['motion_phase'] == 'track' for frame in diagnostics)


@pytest.mark.parametrize('scenario', ['lead_stop', 'cut_in', 'model_stop', 'override', 'lead_loss'])
def test_new_stop_chain_keeps_existing_risk_gates(scenario):
  result = simulate_following(unified=True, scenario=scenario)
  assert result['minimum_gap'] > 2.0
  assert result['minimum_command'] >= -3.5
  assert result['maximum_command'] <= 2.0
  # The 8 m/s^3 limit applies to request rate, not vehicle body jerk. Verify
  # raw requests first, then the exact per-pair Float32 rounding allowance.
  assert result['wire_is_float32_raw']
  assert result['raw_rate_violations'] == 0
  assert result['wire_rate_violations'] == 0
  assert result['conflicts'] == 0
  if scenario in ('lead_stop', 'cut_in', 'model_stop'):
    assert result['final_speed'] == 0.0
    assert result['stopping_seconds'] > 0.0


@pytest.mark.parametrize('speed', [0.0, 0.16, 1.1, 20.0])
def test_model_reference_has_one_consistent_forward_progression(speed):
  raw_velocity = speed - 1.1 * np.asarray(ModelConstants.T_IDXS)
  original = raw_velocity.copy()
  position, velocity, acceleration = model_reference_from_velocity(raw_velocity)
  intervals = np.diff(T_IDXS)
  np.testing.assert_array_equal(raw_velocity, original)
  np.testing.assert_array_equal(velocity, np.maximum(np.interp(T_IDXS, ModelConstants.T_IDXS, raw_velocity), 0.0))
  assert position[0] == 0.0
  assert np.all(np.diff(position) >= 0.0)
  np.testing.assert_allclose(np.diff(velocity), acceleration[:-1] * intervals)
  np.testing.assert_allclose(np.diff(position), velocity[:-1] * intervals + 0.5 * acceleration[:-1] * intervals ** 2)
  assert np.all(acceleration[:-1][(velocity[:-1] == 0.0) & (velocity[1:] == 0.0)] == 0.0)


@pytest.mark.parametrize('raw_velocity', [[], [0.0] * 32, [float('nan')] * 33, [float('inf')] * 33])
def test_forward_reference_still_rejects_missing_and_nonfinite_prediction(raw_velocity):
  with pytest.raises(ValueError, match='Invalid model velocity'):
    model_reference_from_velocity(raw_velocity)


@pytest.mark.parametrize('speed,negative_tail', [(0.0, -0.013411), (1.1, -0.125525)])
def test_near_stop_model_rebound_is_not_misclassified_as_a_solver_failure(speed, negative_tail):
  planner, extension = make_planner(speed=speed)
  state = make_state(speed=speed, experimental=True)
  model = state['modelV2']
  model_times = np.asarray(ModelConstants.T_IDXS)
  model.velocity.x = np.maximum(speed - 1.1 * model_times, negative_tail).tolist()
  model.position.x = (speed * model_times - 0.5 * model_times ** 2).tolist()
  model.action.shouldStop = True
  model.action.desiredAcceleration = 0.3
  for frame in range(12):
    planner.update(state, now_ns=advance(state, frame))
    assert planner.mpc.solution_status == 0
    assert planner.motion_intent.valid
    assert planner.motion_intent.reasons & int(StopReason.E2E)
  control = LongControl(planner.CP, extension, unified=True)
  output = control.update(
    True, state['carState'], planner.output_a_target, planner.output_should_stop, (-3.5, 2.0),
    intent=planner.motion_intent, plan_time_ns=state.logMonoTime['modelV2'],
    model_time_ns=state.logMonoTime['modelV2'], now_ns=state.logMonoTime['modelV2'],
  )
  assert control.resolved_should_stop
  assert output <= 0.0


@pytest.mark.parametrize('estimated_speed', [-0.001184, -0.031628, -0.039136])
@pytest.mark.parametrize('standstill', [False, True])
def test_finite_kalman_stop_undershoot_is_projected_without_mutating_car_state(estimated_speed, standstill):
  planner, extension = make_planner(speed=0.0)
  state = make_state(speed=0.0, target=0.0, experimental=True)
  state['carState'].vEgo = estimated_speed
  state['carState'].vEgoRaw = 0.0
  state['carState'].standstill = standstill
  state['modelV2'].velocity.x = [0.1] * len(ModelConstants.T_IDXS)
  state['modelV2'].action.shouldStop = True
  planner.update(state)
  assert state['carState'].vEgo == pytest.approx(estimated_speed)
  assert planner.mpc.x0[1] == 0.0
  assert planner.mpc.solution_status == 0
  assert planner.motion_intent.valid
  control = LongControl(planner.CP, extension, unified=True)
  output = control.update(
    True, state['carState'], planner.output_a_target, planner.output_should_stop, (-3.5, 2.0),
    intent=planner.motion_intent, plan_time_ns=state.logMonoTime['modelV2'],
    model_time_ns=state.logMonoTime['modelV2'], now_ns=state.logMonoTime['modelV2'],
  )
  assert not control.motion_degraded
  assert control.resolved_should_stop
  assert output <= 0.0


@pytest.mark.parametrize('estimated_speed', [-0.031628, 0.0])
@pytest.mark.parametrize('measured_accel', [-0.1, -0.5, -1.5])
def test_zero_forward_speed_and_negative_acceleration_cannot_invent_a_release(estimated_speed, measured_accel):
  planner, extension = make_planner(speed=0.0)
  state = make_state(speed=0.0, target=20.0)
  state['carState'].vEgo = estimated_speed
  state['carState'].aEgo = measured_accel
  state['controlsState'].longControlState = LongCtrlState.stopping
  planner.output_a_target = -2.0
  lead = state['radarState'].leadOne
  lead.present = True
  lead.dRel = 6.0
  lead.vLead = 0.0
  lead.vRel = 0.0
  lead.aLeadK = 0.0
  lead.aLeadTau = 1.5
  lead.modelProb = 0.99
  planner.update(state)
  assert planner.mpc.solution_status == 0
  assert planner.mpc.x0[1] == planner.mpc.x0[2] == 0.0
  assert state['carState'].aEgo == pytest.approx(measured_accel)
  assert planner.motion_intent.valid
  assert planner.motion_intent.reasons & int(StopReason.MPC)
  assert planner.output_a_target < 0.2
  controller = LongControl(planner.CP, extension, unified=True)
  output = controller.update(
    True, state['carState'], planner.output_a_target, planner.output_should_stop, (-3.5, 2.0),
    intent=planner.motion_intent, plan_time_ns=state.logMonoTime['modelV2'],
    model_time_ns=state.logMonoTime['modelV2'], now_ns=state.logMonoTime['modelV2'],
  )
  assert controller.resolved_should_stop
  assert output <= -2.0


def test_positive_forward_speed_does_not_discard_measured_braking():
  planner, extension = make_planner(speed=0.2)
  state = make_state(speed=0.2)
  state['carState'].aEgo = -0.5
  planner.update(state)
  assert planner.mpc.x0[1] > 0.0
  assert planner.mpc.x0[2] == -0.5
