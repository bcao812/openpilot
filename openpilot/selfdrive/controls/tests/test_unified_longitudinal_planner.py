from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.honda.values import CAR
from openpilot.cereal import log, messaging
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_trace import LongitudinalTrace, MOTION_TRACE_SCHEMA, UNIFIED_TRACE_FIELDS
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION, MotionIntent, StopReason
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import departing_lead_rejection, has_departing_lead, model_reference_from_velocity
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_planner, make_state, advance, Publisher
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_longitudinal_following_scenarios import simulate_following
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanSource as SPSource


def test_unified_is_selected_by_vehicle_capabilities():
  planner, extension = make_planner()
  assert planner.unified and LongControl(planner.CP, extension).unified
  for platform, stock, interceptor in (
    (CAR.HONDA_CIVIC, False, False),
    (CAR.HONDA_CIVIC_BOSCH, False, False),
    (CAR.HONDA_ACCORD, True, False),
    (CAR.HONDA_ACCORD, False, True),
  ):
    planner, extension = make_planner(platform=platform, stock=stock, interceptor=interceptor)
    assert not planner.unified
    assert not LongControl(planner.CP, extension).unified


def test_candidate_does_not_allocate_legacy_stop_owners():
  planner, extension = make_planner()
  assert planner.unified
  assert LongControl(planner.CP, extension).unified
  for name in ('stop_latched', 'creep_t', 'creep_stop', 'close_lead_stop', 'moving_lead_t'):
    assert not hasattr(planner, name)


def test_new_pipeline_does_not_execute_independent_cruise_controller():
  planner, extension = make_planner()
  state = make_state()
  with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.get_cruise_accel', side_effect=AssertionError('legacy cruise executed')):
    planner.update(state, now_ns=advance(state, 0))
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
    planner.update(state, now_ns=advance(state, 0))
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
  planner.update(state, now_ns=advance(state, 0))
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
  planner.update(state, now_ns=advance(state, 0))
  assert not planner.motion_intent.valid
  assert planner.output_a_target <= -0.2
  assert planner.output_should_stop


@pytest.mark.parametrize('attribute', ['vEgo', 'aEgo', 'vCruise'])
@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_motion_input_invalidates_intent_without_positive_request(attribute, value):
  planner, extension = make_planner()
  state = make_state()
  setattr(state['carState'], attribute, value)
  planner.update(state, now_ns=advance(state, 0))
  assert not planner.motion_intent.valid
  assert np.isfinite(planner.output_a_target)
  assert planner.output_a_target <= -0.2


def test_solver_failure_is_invalid_not_a_successful_zero_trajectory():
  planner, extension = make_planner()
  state = make_state()
  failing_solver = Mock(wraps=planner.mpc.solver)
  failing_solver.solve.return_value = 4
  with patch.object(planner.mpc, 'solver', failing_solver):
    planner.update(state, now_ns=advance(state, 0))
  assert planner.mpc.solution_status == 4
  assert not planner.motion_intent.valid
  publisher = Publisher()
  planner.publish(state, publisher)
  assert not publisher.messages['longitudinalPlan'].valid


def test_reference_and_intent_round_trip_preserve_distinct_semantics():
  planner, extension = make_planner()
  state = make_state(experimental=True)
  state['modelV2'].action.desiredAcceleration = -3.0
  planner.update(state, now_ns=advance(state, 0))
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
  result = simulate_following(delay=delay, response_time=response_time, diagnostics=diagnostics)
  assert result['stopping_seconds'] == 0.0
  assert result['minimum_speed'] > 0.3
  assert result['speed_error'] < 0.15
  assert result['minimum_gap'] > 5.0
  assert result['conflicts'] == 0
  assert all(frame['intent_version'] == MOTION_INTENT_VERSION for frame in diagnostics)
  assert any(frame['motion_phase'] == 'track' for frame in diagnostics)


@pytest.mark.parametrize('scenario', ['lead_stop', 'cut_in', 'model_stop', 'override', 'lead_loss'])
def test_new_stop_chain_keeps_existing_risk_gates(scenario):
  result = simulate_following(scenario=scenario)
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
  control = LongControl(planner.CP, extension)
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
  planner.update(state, now_ns=advance(state, 0))
  assert state['carState'].vEgo == pytest.approx(estimated_speed)
  assert planner.mpc.x0[1] == 0.0
  assert planner.mpc.solution_status == 0
  assert planner.motion_intent.valid
  control = LongControl(planner.CP, extension)
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
  planner.update(state, now_ns=advance(state, 0))
  assert planner.mpc.solution_status == 0
  assert planner.mpc.x0[1] == planner.mpc.x0[2] == 0.0
  assert state['carState'].aEgo == pytest.approx(measured_accel)
  assert planner.motion_intent.valid
  assert planner.motion_intent.reasons & int(StopReason.MPC)
  assert planner.output_a_target < 0.2
  controller = LongControl(planner.CP, extension)
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
  planner.update(state, now_ns=advance(state, 0))
  assert planner.mpc.x0[1] > 0.0
  assert planner.mpc.x0[2] == -0.5


@pytest.mark.parametrize('held', [False, True])
@pytest.mark.parametrize('speed,delay,gap,allowed', [
  (0.0, .5, 2.999, False), (0.0, .5, 3.0, True), (0.0, .5, 4.0, True),
  (.2, .5, 3.0, False), (.2, .5, 3.117, False), (.2, .5, 3.119, True),
  (.2, 1.0, 3.217, False), (.2, 1.0, 3.219, True),
])
def test_departure_default_gap_is_three_metres_with_reaction_and_braking_reserve(held, speed, delay, gap, allowed):
  from openpilot.selfdrive.controls.tests.test_lead_qualification import formal_lead
  from openpilot.selfdrive.controls.tests.test_radar_consumption import CONSUMPTION, PUBLICATION, SESSION, radar_frame

  primary = formal_lead(now_ns=PUBLICATION, dRel=gap, vLead=1.0, vLeadK=1.0, vRel=1.0 - speed,
                        evidence={'trackId': 5})
  radar = radar_frame(primary, display=(primary,))
  identity = (SESSION, 5) if held else None
  rejection = departing_lead_rejection(radar, speed, delay, now_ns=CONSUMPTION, held_identity=identity,
                                       allow_unbound_physical=True)
  assert rejection == (None if allowed else ('lead_gap', (SESSION, 5)))
  if held:
    assert has_departing_lead(radar, speed, delay, now_ns=CONSUMPTION, held_identity=identity) is allowed
    assert not has_departing_lead(radar, speed, delay, minimum_gap_base=5.0, now_ns=CONSUMPTION, held_identity=identity)


@pytest.mark.parametrize('defect,reason', [
  ('qualification', 'lead_departure_unqualified'), ('stale', 'lead_departure_unqualified'),
  ('lateral', 'lead_path_offset'), ('relative_speed', 'lead_departure_unqualified'),
  ('lead_speed', 'lead_speed'), ('stopping', 'lead_stopping'), ('identity', 'held_identity_mismatch'),
])
def test_three_metre_departure_keeps_the_other_lead_guards(defect, reason):
  from openpilot.selfdrive.controls.tests.test_lead_qualification import formal_lead
  from openpilot.selfdrive.controls.tests.test_radar_consumption import CONSUMPTION, PUBLICATION, SESSION, radar_frame

  primary = formal_lead(now_ns=PUBLICATION, dRel=3.2, vLead=1.0, vLeadK=1.0, vRel=1.0, evidence={'trackId': 5})
  identity = (SESSION, 5)
  if defect == 'qualification':
    primary.evidence.departureQualified = False
  elif defect == 'stale':
    primary.evidence.positionMonoTime = PUBLICATION - 200_000_000
  elif defect == 'lateral':
    primary.evidence.pathOffset = 1.81
  elif defect == 'relative_speed':
    primary.vRel = 0.0
  elif defect == 'lead_speed':
    primary.vLead = primary.vLeadK = primary.vRel = .49
  elif defect == 'stopping':
    primary.aLeadK = -.4
  elif defect == 'identity':
    identity = (SESSION, 7)
  radar = radar_frame(primary, display=(primary,))
  assert departing_lead_rejection(radar, 0.0, .5, now_ns=CONSUMPTION, held_identity=identity) == (reason, (SESSION, 5))
  assert not has_departing_lead(radar, 0.0, .5, now_ns=CONSUMPTION, held_identity=identity)


@pytest.mark.parametrize('held', ['none', 'matching', 'different'])
@pytest.mark.parametrize('other', ['none', 'departing', 'stationary', 'occluded', 'caution'])
def test_unbound_physical_departure_requires_formal_leads_without_replacing_a_held_identity(held, other):
  from openpilot.selfdrive.controls.tests.test_lead_qualification import caution_lead, formal_lead
  from openpilot.selfdrive.controls.tests.test_radar_consumption import CONSUMPTION, PUBLICATION, SESSION, radar_frame

  primary = formal_lead(now_ns=PUBLICATION, dRel=15.0, vLead=1.0, vLeadK=1.0, vRel=1.0, evidence={'trackId': 5})
  objects = [primary]
  if other != 'none':
    factory = caution_lead if other == 'caution' else formal_lead
    speed = 1.0 if other == 'departing' else 0.0
    objects.append(factory(now_ns=PUBLICATION, dRel=8.0, vLead=speed, vLeadK=speed, vRel=speed, evidence={'trackId': 7}))
    if other == 'occluded':
      objects[-1].evidence.source = 'predicted'
      objects[-1].evidence.coasting = True
  radar = radar_frame(*objects, display=(primary,) if other == 'caution' else tuple(objects))
  identity = None if held == 'none' else (SESSION, 5 if held == 'matching' else 7)
  rejection = departing_lead_rejection(radar, 0.0, .5, now_ns=CONSUMPTION, held_identity=identity, allow_unbound_physical=True)
  assert (rejection is None) is (held != 'different' and (held == 'matching' or other not in ('stationary', 'occluded')))
  if held == 'none':
    assert not has_departing_lead(radar, 0.0, .5, now_ns=CONSUMPTION)
    if other in ('stationary', 'occluded'):
      assert rejection == ('lead_departure_unqualified', (SESSION, 7))
  elif held == 'different':
    assert rejection == ('held_identity_mismatch', (SESSION, 5))


@pytest.mark.parametrize('speed,standstill,consumed', [
  (0.0, True, True), (-1.9180846832417676e-23, True, True),
  (-np.finfo(float).eps, True, True), (-2.0 * np.finfo(float).eps, True, False),
  (-0.001, True, False), (-1.9180846832417676e-23, False, False),
])
def test_radar_consumption_uses_only_stopped_arithmetic_roundoff_without_mutating_car_state(speed, standstill, consumed):
  from openpilot.selfdrive.controls.tests.test_lead_qualification import formal_lead
  from openpilot.selfdrive.controls.tests.test_radar_consumption import CONSUMPTION, PUBLICATION, radar_frame
  from openpilot.selfdrive.controls.tests.test_unified_radar_consumption import enabled_planner, state_at

  primary = formal_lead(now_ns=PUBLICATION, dRel=15.0, vLead=1.0, vLeadK=1.0, vRel=1.0)
  state = state_at(radar_frame(primary, display=(primary,)), speed=0.0)
  state['carState'].vEgo, state['carState'].standstill = float(speed), standstill
  wire_speed = state['carState'].as_reader().vEgo
  planner = enabled_planner(0.0)
  with patch.object(planner.dec, 'enabled', return_value=True):
    view = planner.reference_planner.radar_input_state(state, CONSUMPTION)
  reference = planner.reference_planner
  assert (reference._radar_consumed_view is view['radarState']) is consumed
  assert (reference._radar_consumed_ns == CONSUMPTION) is consumed
  assert state['carState'].vEgo == wire_speed


@pytest.mark.parametrize('defect', [
  'none', 'previous_expiry', 'legacy_producer_risk', 'unconsumed', 'wrong_tick', 'wrong_publication',
  'unhealthy', 'degraded', 'current_expiry', 'wrong_session', 'not_standstill', 'invalid_message',
])
def test_unbound_radar_departure_uses_current_health_without_retaining_previous_expiry(defect):
  from openpilot.selfdrive.controls.tests.test_lead_qualification import formal_lead
  from openpilot.selfdrive.controls.tests.test_radar_consumption import CONSUMPTION, PUBLICATION, coast, native_frame, radar_frame
  from openpilot.selfdrive.controls.tests.test_unified_radar_consumption import enabled_planner, state_at

  primary = formal_lead(now_ns=PUBLICATION, dRel=15.0, vLead=1.0, vLeadK=1.0, vRel=1.0)
  objects = [primary]
  if defect == 'current_expiry':
    objects.append(coast(publication=PUBLICATION, relevant=True, track=99))
  radar = radar_frame(*objects, display=tuple(objects), degraded=defect == 'degraded',
                      risk_unresolved=defect == 'legacy_producer_risk')
  if defect == 'unhealthy':
    radar.fusionHealthy = False
  if defect == 'wrong_session':
    radar.fusionSessionId = 'different-session'
  state = state_at(radar, speed=0.0)
  if defect == 'not_standstill':
    state['carState'].standstill = False
  planner = enabled_planner(0.0)
  reference = planner.reference_planner
  if defect == 'previous_expiry':
    old = PUBLICATION - 100_000_000
    lost = coast(publication=old, relevant=True, track=99)
    reference.radar_consumption.update(native_frame(radar_frame(lost, publication=old, display=(lost,))),
                                       publication_ns=old, now_ns=old + 60_000_000, speed=10.0)
    assert reference.radar_consumption.degraded
  with patch.object(planner.dec, 'enabled', return_value=True):
    view = state if defect == 'unconsumed' else reference.radar_input_state(state, CONSUMPTION)
  if defect == 'wrong_publication':
    state.logMonoTime['radarState'] += 1
  elif defect == 'invalid_message':
    state.valid['radarState'] = False
  rejection = reference.unbound_radar_departure_rejection(view, now_ns=CONSUMPTION + (defect == 'wrong_tick'))
  assert (rejection is None) is (defect in ('none', 'previous_expiry', 'legacy_producer_risk'))
  assert reference.radar_consumption.degraded is (defect == 'current_expiry')
  if defect == 'current_expiry':
    assert rejection == 'radar_measurement_expired'


@pytest.mark.parametrize('field', ['fusionHealthy', 'fusionDegraded'])
def test_unknown_radar_health_cannot_be_interpreted_as_clear(field):
  from types import SimpleNamespace
  from openpilot.selfdrive.controls.tests.test_lead_qualification import formal_lead
  from openpilot.selfdrive.controls.tests.test_radar_consumption import CONSUMPTION, PUBLICATION, radar_frame
  from openpilot.selfdrive.controls.tests.test_unified_radar_consumption import enabled_planner, state_at

  primary = formal_lead(now_ns=PUBLICATION, dRel=15.0, vLead=1.0, vLeadK=1.0, vRel=1.0)
  planner = enabled_planner(0.0)
  state = state_at(radar_frame(primary, display=(primary,)), speed=0.0)
  reference = planner.reference_planner
  with patch.object(planner.dec, 'enabled', return_value=True):
    view = reference.radar_input_state(state, CONSUMPTION)
  assert reference.unbound_radar_departure_rejection(view, now_ns=CONSUMPTION) is None
  fields = ('leadOne', 'mdMonoTime', 'fusionSessionId', 'fusionActive', 'fusionHealthy', 'fusionDegraded')
  # A consumer adapter with unknown metadata must fail even after a prior
  # healthy consumption. Preserve the receipt to isolate that metadata gate.
  unknown = SimpleNamespace(**{name: getattr(view['radarState'], name) for name in fields if name != field})
  view.radar = reference._radar_consumed_view = unknown
  assert reference.unbound_radar_departure_rejection(view, now_ns=CONSUMPTION) is not None


def run_unbound_physical_departure(condition='none', *, frames=24):
  from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import consume
  from openpilot.selfdrive.controls.tests.test_lead_qualification import caution_lead, formal_lead
  from openpilot.selfdrive.controls.tests.test_radar_consumption import SESSION, coast, native_frame, radar_frame
  from openpilot.selfdrive.controls.tests.test_stationary_lead_departure import held_recorded_departure_case

  Params().put_bool('DynamicExperimentalControl', False, block=True)
  planner, owner, state = held_recorded_departure_case()
  for name in ('AlphaLongitudinalEnabled', 'ExperimentalMode', 'DynamicExperimentalControl'):
    Params().put_bool(name, True, block=True)
  reference = planner.reference_planner
  assert reference.radar_held_identity is None
  state['carState'].vEgo = -1.9180846832417676e-23
  if condition == 'held_mismatch':
    reference.radar_held_identity = (SESSION, 7)
  elif condition == 'previous_expiry':
    old = 1_800_000_000
    lost = coast(publication=old, relevant=True, track=99)
    reference.radar_consumption.update(native_frame(radar_frame(lost, publication=old, display=(lost,))),
                                       publication_ns=old, now_ns=old + 60_000_000, speed=10.0)
    assert reference.radar_consumption.degraded
  elif condition == 'model_stop':
    state['modelV2'].action.shouldStop = True
  records = []
  with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'active', return_value=True), \
       patch.object(planner.dec, 'mode', return_value='blended'):
    for frame in range(20, 20 + frames):
      now = 1_000_000_000 + frame * 50_000_000
      factory = caution_lead if condition == 'side_primary' else formal_lead
      primary = factory(now_ns=now, dRel=15.0, vLead=1.6, vLeadK=1.6, vRel=1.6, evidence={'trackId': 5})
      objects = [primary]
      if condition in ('stationary_other', 'occluded_other'):
        other = formal_lead(now_ns=now, dRel=8.0, vLead=0.0, vLeadK=0.0, vRel=0.0, evidence={'trackId': 7})
        if condition == 'occluded_other':
          other.evidence.source, other.evidence.coasting = 'predicted', True
        objects.append(other)
      radar = radar_frame(*objects, publication=now, display=tuple(objects), risk_unresolved=condition == 'legacy_producer_risk')
      if condition == 'unhealthy':
        radar.fusionHealthy = False
      state['radarState'] = native_frame(radar)
      state['controlsState'].longControlState = owner.long_control_state
      state['controlsState'].motionState.phase = owner.motion_phase.text
      state['controlsState'].motionState.resolvedStop = owner.resolved_should_stop
      plan, outputs = consume(planner, owner, state, frame)
      if condition == 'side_primary':
        assert plan.solverStatus != 0 and not plan.motionIntent.valid
        assert owner.motion_degraded
      else:
        assert plan.solverStatus == 0 and plan.motionIntent.valid
        assert not owner.motion_degraded
      records.append((plan.aTarget, plan.motionIntent.releaseAllowed, owner.resolved_should_stop, tuple(outputs)))
  return reference, records


@pytest.mark.parametrize('condition', [
  'none', 'previous_expiry', 'legacy_producer_risk', 'held_mismatch', 'stationary_other', 'occluded_other',
  'side_primary', 'model_stop', 'unhealthy',
])
def test_actual_hold_uses_unbound_departure_only_when_all_independent_guards_hold(condition):
  reference, records = run_unbound_physical_departure(condition)
  if condition in ('none', 'previous_expiry', 'legacy_producer_risk'):
    assert any(released and not held and max(outputs) > 0.0 for _, released, held, outputs in records)
    assert reference.radar_held_identity is None  # current evidence never invents an adopted physical barrier
    assert not reference.radar_consumption.degraded
  else:
    assert all(not released and held and max(outputs) <= 0.0 for _, released, held, outputs in records)


def test_departure_diagnostics_are_rate_limited_and_logging_failure_does_not_change_control():
  outcomes, logs = [], []
  for failure in (None, RuntimeError('transport unavailable')):
    with patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.cloudlog.event', side_effect=failure) as event:
      _, records = run_unbound_physical_departure('held_mismatch', frames=120)
    outcomes.append(records)
    logs.append([call for call in event.call_args_list if call.args[0] == 'longitudinal_departure_blocked_v1'])
  assert outcomes[0] == outcomes[1]
  assert len(logs[0]) == len(logs[1]) == 2
  for recorded in logs:
    first, last = (call.kwargs for call in recorded)
    assert first['stage'] == last['stage'] == 'held_identity_mismatch'
    assert first['wait_seconds'] >= .5
    assert int(last['decision_mono_ns']) - int(first['decision_mono_ns']) >= 5_000_000_000
    assert first['held_identity'][1] == 7 and first['primary_identity'][1] == 5
    assert set(first['input_mono_ns']) == {'carState', 'modelV2', 'radarState', 'controlsState'}
