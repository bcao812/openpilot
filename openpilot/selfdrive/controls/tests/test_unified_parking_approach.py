"""Producer arbitration and the unchanged owner contract for parking caps."""
from unittest.mock import patch

import numpy as np
import pytest

from opendbc.car.honda.values import CAR
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CREEP_MAX_T, CREEP_STOP_T
from openpilot.selfdrive.controls.lib.launch_response import ResponseDomain
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import COMFORT_STOP_DISTANCE, RISK_STOP_DISTANCE
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent, MotionPhase, MotionResponseKind, StopReason
from openpilot.selfdrive.controls.lib.parking_approach import FALLBACK_ACCEL
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_TIMING_VERSION
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import ControlHarness
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import advance, make_planner, make_state
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_motion_supervisor import MotionHarness


SAMPLE_ACCEL = 'openpilot.selfdrive.controls.lib.unified_longitudinal_planner.sample_reference_acceleration'


def set_vision_lead(state, timestamp, gap, *, lead_index=0, std=.1):
  """Explicit model/radard publication identity, with actual Float32 fields."""
  model, radar = state['modelV2'], state['radarState']
  model.init('leadsV3', 2)
  radar.mdMonoTime = timestamp
  for index in range(2):
    source = model.leadsV3[index]
    source.t = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    source.x = [gap + 1.52 + index] * 6
    source.y = [0.0] * 6
    source.v = [0.0] * 6
    source.a = [0.0] * 6
    source.xStd = [std] * 6
    source.prob = .995
    lead = (radar.leadOne, radar.leadTwo)[index]
    lead.present = index == lead_index
    lead.radar = False
    lead.radarTrackId = -1
    lead.dRel = float(np.float32(source.x[0] - 1.52))
    lead.yRel = float(np.float32(-source.y[0]))
    lead.vRel = float(np.float32(source.v[0] - model.velocity.x[0]))
    lead.vLead = float(state['carState'].vEgo + lead.vRel)
    lead.aLeadK = float(np.float32(source.a[0]))
    lead.modelProb = .995


def warm_parking(*, lead_index=0, experimental=False, model_accel=.5, radar_available=False):
  planner, extension = make_planner(speed=.6)
  planner.CP.radarUnavailable = not radar_available
  state = make_state(speed=.6, target=10.0, experimental=experimental)
  state['modelV2'].action.desiredAcceleration = model_accel
  # Isolate the scalar arbitration from native preference differences. The
  # native solver still runs, retaining its original bounds and geometry.
  with patch(SAMPLE_ACCEL, return_value=.4):
    for frame in range(121):
      now = advance(state, frame)
      set_vision_lead(state, now, 15.0 + .6 * (120 - frame) * .05, lead_index=lead_index)
      planner.update(state, now_ns=now)
      assert planner.mpc.solution_status == 0
  return planner, extension, state


@pytest.mark.parametrize('lead_index', [0, 1])
def test_negative_parking_cap_has_exact_source_and_no_departure_permission(lead_index):
  planner, _, _ = warm_parking(lead_index=lead_index)
  decision = planner.reference_planner.parking_approach.decision
  assert decision.eligible and decision.cap_effective and decision.allow_compact
  assert decision.limiting_lead == lead_index
  assert decision.accel_after == planner.output_a_target == decision.accel_cap
  assert decision.accel_after < decision.accel_before
  assert planner.reference_planner.constraint_source == (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1)[lead_index]
  assert planner.motion_intent.response_kind == MotionResponseKind.ABSOLUTE_ACC
  assert not planner.motion_intent.release_allowed


def test_stronger_model_cap_remains_stronger_and_is_not_counted_as_parking_use():
  planner, _, _ = warm_parking(experimental=True, model_accel=-3.0)
  decision = planner.reference_planner.parking_approach.decision
  assert decision.eligible and decision.accel_cap > -3.0
  assert not decision.cap_effective
  assert planner.output_a_target <= -3.0
  assert planner.reference_planner.constraint_source == LongitudinalPlanSource.e2e


@pytest.mark.parametrize('fault', ['std', 'source'])
def test_uncertainty_failure_before_cap_application_rebuilds_full_history_in_producer(fault):
  planner, _, state = warm_parking(experimental=True, model_accel=-3.0)
  producer = planner.reference_planner
  assert producer.stop_distance_policy.evidence.low_speed_history
  assert producer.parking_approach.decision.allow_compact
  assert producer.parking_mpc_distances[0] < 4.0
  assert producer.parking_mpc_distances[1][0] < 6.0
  assert not producer.parking_approach.decision.cap_effective
  assert producer.parking_approach.braking_since_ns is None

  with patch(SAMPLE_ACCEL, return_value=.4):
    for frame in range(121, 235):
      now = advance(state, frame)
      set_vision_lead(state, now, 15.0 - .03 * (frame - 120),
                      std=1000.0 if frame == 121 and fault == 'std' else .1)
      if frame == 121 and fault == 'source':
        # Fresh, nearby time is still not an exact observed model identity.
        state['radarState'].mdMonoTime = now - 1
      planner.update(state, now_ns=now)
      assert planner.mpc.solution_status == 0
      assert planner.output_a_target <= -3.0
      assert not producer.parking_approach.decision.cap_effective
      assert producer.parking_approach.braking_since_ns is None
      if frame < 222:
        # The first valid carState after the fault is frame 122. Frame 221
        # contains only 4.95 seconds of new history. Exercise ULP's own reset,
        # its public returned distances and the values actually sent native.
        assert not producer.stop_distance_policy.evidence.low_speed_history
        assert not producer.parking_approach.decision.allow_compact
        assert producer.parking_policy_distances == (4.0, (6.0, 6.0))
        assert producer.parking_mpc_distances == (4.0, (6.0, 6.0))
        assert np.min(planner.mpc.params[:, COMFORT_STOP_DISTANCE]) >= 4.0
        np.testing.assert_array_equal(planner.mpc.params[:, RISK_STOP_DISTANCE],
                                      np.full_like(planner.mpc.params[:, RISK_STOP_DISTANCE], 6.0))

  assert producer.stop_distance_policy.evidence.low_speed_history
  assert producer.parking_approach.decision.allow_compact
  assert producer.parking_policy_distances[0] < 4.0
  assert producer.parking_mpc_distances == producer.parking_policy_distances
  assert np.min(planner.mpc.params[:, RISK_STOP_DISTANCE]) < 6.0


def test_radar_available_platform_cannot_acquire_vision_parking_policy():
  planner, _, _ = warm_parking(radar_available=True)
  decision = planner.reference_planner.parking_approach.decision
  assert decision.reason == 'not_permitted'
  assert not decision.eligible and decision.accel_cap is None
  assert np.min(planner.mpc.params[:, COMFORT_STOP_DISTANCE]) >= 4.0


@pytest.mark.parametrize('platform,stock', [(CAR.HONDA_CIVIC, False), (CAR.HONDA_ACCORD, True)])
def test_other_platform_and_stock_modes_have_no_unified_parking_producer(platform, stock):
  planner, _ = make_planner(speed=.6, platform=platform, stock=stock)
  assert not planner.unified
  assert not hasattr(planner, 'reference_planner')


def test_loss_after_applied_cap_retains_stop_and_restores_fallback_geometry():
  planner, _, state = warm_parking()
  assert planner.reference_planner.parking_approach.decision.cap_effective
  now = advance(state, 121)
  state['radarState'].leadOne.present = False
  with patch(SAMPLE_ACCEL, return_value=.4):
    planner.update(state, now_ns=now)
  decision = planner.reference_planner.parking_approach.decision
  assert decision.require_stop and not decision.allow_compact
  assert planner.output_a_target <= FALLBACK_ACCEL
  assert planner.motion_intent.reasons & StopReason.CLOSE_LEAD
  assert not planner.motion_intent.release_allowed
  assert planner.reference_planner.parking_approach.braking_since_ns is not None


def test_single_raw_standstill_bit_does_not_erase_applied_braking():
  planner, _, state = warm_parking()
  since = planner.reference_planner.parking_approach.braking_since_ns
  now = advance(state, 121)
  set_vision_lead(state, now, 14.97)
  state['carState'].standstill = True
  with patch(SAMPLE_ACCEL, return_value=.4):
    planner.update(state, now_ns=now)
  decision = planner.reference_planner.parking_approach.decision
  assert decision.require_stop and planner.output_a_target <= FALLBACK_ACCEL
  assert planner.reference_planner.parking_approach.braking_since_ns == since
  assert not planner.motion_intent.release_allowed


def test_acc_parking_cap_preserves_existing_nonpositive_response_correction():
  planner, extension, _ = warm_parking()
  assert planner.reference_planner.parking_approach.decision.cap_effective
  assert planner.motion_intent.response_kind == MotionResponseKind.ABSOLUTE_ACC
  controller = LongControl(planner.CP, extension)
  harness = ControlHarness(controller)
  harness.state.vEgo = .6
  harness.state.aEgo = .3
  original = MotionIntent(producer_instance_ns=planner.motion_intent.producer_instance_ns,
                          response_kind=MotionResponseKind.ABSOLUTE_ACC)
  for _ in range(40):
    harness.step(0.0, original)
  assert controller.motion_phase == MotionPhase.TRACK
  prior_bias, prior_output = controller.launch_response.bias, controller.last_output_accel
  assert prior_bias < 0.0
  with patch.object(controller.launch_response, 'propose', wraps=controller.launch_response.propose) as propose:
    output = harness.step(planner.output_a_target, planner.motion_intent)
  assert propose.call_args.kwargs['response_domain'] == ResponseDomain.ABSOLUTE_ACC
  assert controller.launch_response.bias <= prior_bias < 0.0
  assert output < prior_output and output <= planner.output_a_target


def owner_feedback(state, planner, controller, producer_ns, consumer_ns):
  """Copy genuine owner outputs into the same fields controlsd publishes."""
  controls, control = state['controlsState'], state['carControl']
  controls.longControlState = controller.long_control_state
  control.actuators.longControlState = controller.long_control_state
  control.actuators.accel = controller.last_output_accel
  motion = controls.motionState
  motion.phase = controller.motion_phase.text
  motion.resolvedStop = controller.resolved_should_stop
  motion.reasons = int(controller.motion_reasons)
  motion.degraded = controller.motion_degraded
  motion.waitingForFreshInput = controller.motion_waiting_for_fresh_input
  motion.referenceTimingVersion = REFERENCE_TIMING_VERSION
  motion.referenceTimingValid = True
  motion.referenceProducerInstanceMonoTime = planner.motion_intent.producer_instance_ns
  motion.referenceProducerMonoTime = producer_ns
  motion.referenceConsumerMonoTime = consumer_ns
  motion.referenceSourceMonoTime = planner.reference_timing.pending.source_ns
  motion.referenceEpochMonoTime = planner.reference_timing.pending.epoch_ns
  state.logMonoTime['controlsState'] = state.logMonoTime['carControl'] = consumer_ns


def parking_owner_step(planner, owner, state, frame, *, gap=6.4, lead_speed=0.0, speed=0.0,
                       model_accel=.5, model_stop=False):
  previous_feedback = {name: state.logMonoTime[name] for name in ('controlsState', 'carControl')}
  now = advance(state, frame)
  state.logMonoTime.update(previous_feedback)
  state['carState'].vEgo = speed
  state['carState'].standstill = speed == 0.0
  state['modelV2'].velocity.x = [speed] * len(state['modelV2'].velocity.x)
  state['modelV2'].action.desiredAcceleration = model_accel
  state['modelV2'].action.shouldStop = model_stop
  set_vision_lead(state, now, gap)
  source, lead = state['modelV2'].leadsV3[0], state['radarState'].leadOne
  source.v = [lead_speed] * 6
  lead.vRel = float(np.float32(source.v[0] - state['modelV2'].velocity.x[0]))
  lead.vLead = float(state['carState'].vEgo + lead.vRel)
  planner.update(state, now_ns=now)
  assert planner.mpc.solution_status == 0
  owner.now_ns = now
  owner.state.vEgo = speed
  owner.state.standstill = speed == 0.0
  for _ in range(4):
    owner.step(planner.output_a_target, planner.motion_intent, new_input=False,
                plan_time_ns=now + 1_000_000, model_time_ns=now)
  owner_feedback(state, planner, owner.control, now, owner.now_ns)
  return planner.reference_planner.parking_approach.decision


def parking_episode_in_hold():
  planner, extension, state = warm_parking()
  assert planner.reference_planner.parking_approach.decision.cap_effective
  owner = ControlHarness(LongControl(planner.CP, extension))
  # The existing owner genuinely consumes the stop and reaches HOLD. Keep
  # fresh source history while stopped, so departure need not invent a new
  # five-second delay. This is a contract fixture, not a vehicle plant model.
  for frame in range(121, 235):
    parking_owner_step(planner, owner, state, frame)
    assert owner.control.motion_phase == MotionPhase.HOLD
  assert planner.reference_planner.stop_distance_policy.evidence.low_speed_history
  assert planner.reference_planner.parking_approach.decision.held_episode
  return planner, owner, state


def test_stationary_lead_and_positive_gap_request_cannot_release_parking_hold():
  with patch(SAMPLE_ACCEL, return_value=.4):
    planner, owner, state = parking_episode_in_hold()
    for frame in range(235, 255):
      decision = parking_owner_step(planner, owner, state, frame)
      assert planner.output_a_target > 0.0
      assert decision.release_blocked and decision.held_episode
      assert decision.accel_cap is None and not decision.require_stop
      assert not planner.motion_intent.release_allowed
      assert owner.control.motion_phase == MotionPhase.HOLD and owner.control.last_output_accel < 0.0


def test_sustained_independent_departure_retires_episode_only_after_moving_track():
  with patch(SAMPLE_ACCEL, return_value=.4):
    planner, owner, state = parking_episode_in_hold()
    saw_pending_release = False
    saw_consumed_departure = False
    for frame in range(235, 255):
      moving = owner.control.motion_phase == MotionPhase.TRACK and owner.control.last_output_accel > 0.0
      decision = parking_owner_step(planner, owner, state, frame, gap=6.4 + .03 * (frame - 234),
                                     lead_speed=.6, speed=.35 if moving else 0.0)
      if decision.reason == 'independent_departure_consumed':
        saw_consumed_departure = True
        assert moving and not decision.held_episode
      elif not saw_consumed_departure:
        assert decision.held_episode and not decision.release_blocked
        assert planner.reference_planner.parking_approach.braking_since_ns is not None
        saw_pending_release |= owner.control.motion_phase == MotionPhase.RELEASE
    assert saw_pending_release and saw_consumed_departure
    assert owner.control.motion_phase == MotionPhase.TRACK and owner.control.last_output_accel > 0.0


def test_transient_departure_withdrawal_restores_hold_before_episode_can_retire():
  with patch(SAMPLE_ACCEL, return_value=.4):
    planner, owner, state = parking_episode_in_hold()
    pending = parking_owner_step(planner, owner, state, 235, gap=6.43, lead_speed=.6)
    assert pending.held_episode and not pending.release_blocked
    assert owner.control.motion_phase == MotionPhase.RELEASE
    withdrawn = parking_owner_step(planner, owner, state, 236, gap=6.43)
    assert withdrawn.held_episode and withdrawn.release_blocked and withdrawn.require_stop
    assert planner.output_a_target <= FALLBACK_ACCEL
    assert owner.control.motion_phase == MotionPhase.HOLD
    for frame in range(237, 255):
      held = parking_owner_step(planner, owner, state, frame, gap=6.43)
      assert held.held_episode and held.release_blocked and not held.require_stop
      assert not planner.motion_intent.release_allowed
      assert owner.control.motion_phase == MotionPhase.HOLD


def test_positive_lead_cannot_override_model_braking_or_stop_during_parking_hold():
  with patch(SAMPLE_ACCEL, return_value=.4):
    planner, owner, state = parking_episode_in_hold()
    for frame, model_accel, model_stop in ((235, -.5, False), (236, .5, True)):
      decision = parking_owner_step(planner, owner, state, frame, gap=6.4 + .03 * (frame - 234),
                                     lead_speed=.6, model_accel=model_accel, model_stop=model_stop)
      assert decision.held_episode and decision.release_blocked
      assert not planner.motion_intent.release_allowed
      assert owner.control.motion_phase == MotionPhase.HOLD


def test_only_actual_fresh_hold_acknowledges_braking_not_approach_or_stale_feedback():
  planner, extension, state = warm_parking()
  now = advance(state, 121)
  state['radarState'].leadOne.present = False
  with patch(SAMPLE_ACCEL, return_value=.4):
    planner.update(state, now_ns=now)
  assert planner.motion_intent.reasons & StopReason.CLOSE_LEAD
  controller = LongControl(planner.CP, extension)
  harness = ControlHarness(controller)
  harness.now_ns = now
  harness.state.vEgo = .6
  harness.step(planner.output_a_target, planner.motion_intent)
  assert controller.motion_phase == MotionPhase.APPROACH
  owner_feedback(state, planner, controller, now, harness.now_ns)
  assert not planner.reference_planner.parking_stop_acknowledged(state, harness.now_ns + 1)
  harness.state.vEgo = 0.0
  harness.state.standstill = True
  harness.step(planner.output_a_target, planner.motion_intent)
  assert controller.motion_phase == MotionPhase.HOLD
  state['carState'].vEgo = 0.0
  state['carState'].standstill = True
  owner_feedback(state, planner, controller, now, harness.now_ns)
  assert planner.reference_planner.parking_stop_acknowledged(state, harness.now_ns + 1)
  assert not planner.reference_planner.parking_stop_acknowledged(state, harness.now_ns + 250_000_001)
  # Existing HOLD retains braking when no independent departure is supplied.
  for _ in range(60):
    harness.step(.5, MotionIntent(producer_instance_ns=planner.motion_intent.producer_instance_ns))
    assert controller.motion_phase == MotionPhase.HOLD and controller.last_output_accel < 0.0


def test_parking_candidate_does_not_change_either_existing_creep_deadline():
  assert CREEP_STOP_T == .5 and CREEP_MAX_T == 1.0
  harness = MotionHarness()
  for _ in range(round(CREEP_STOP_T / DT_CTRL) + 2):
    harness.step(v_ego=.6, a_target=-.1)
  assert harness.supervisor.reasons & StopReason.CREEP
  progressing = MotionHarness()
  speed = 1.15
  for frame in range(round(CREEP_MAX_T / DT_CTRL)):
    progressing.step(v_ego=speed, a_target=-.1)
    speed *= 1.0 - .6 * DT_CTRL
    if frame < 99:
      assert not progressing.supervisor.resolved_should_stop
  assert progressing.supervisor.reasons & StopReason.CREEP
