"""Geometry-only admission, with unchanged scalar and motion-owner contracts."""
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from opendbc.car.honda.values import CAR
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CREEP_MAX_T, CREEP_STOP_T
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import COMFORT_STOP_DISTANCE, RISK_STOP_DISTANCE
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionPhase, StopReason
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


def warm_parking(*, lead_index=0, experimental=False, model_accel=.5, radar_available=False, final_gap=15.0):
  planner, extension = make_planner(speed=.6)
  planner.CP.radarUnavailable = not radar_available
  state = make_state(speed=.6, target=10.0, experimental=experimental)
  state['modelV2'].action.desiredAcceleration = model_accel
  # Isolate the scalar arbitration from native preference differences. The
  # native solver still runs, retaining its original bounds and geometry.
  with patch(SAMPLE_ACCEL, return_value=.4):
    for frame in range(121):
      now = advance(state, frame)
      set_vision_lead(state, now, final_gap + .6 * (120 - frame) * .05, lead_index=lead_index)
      planner.update(state, now_ns=now)
      assert planner.mpc.solution_status == 0
  return planner, extension, state


@pytest.mark.parametrize('lead_index', [0, 1])
def test_admitted_compact_geometry_reaches_native_parameters_without_scalar_cap(lead_index):
  planner, _, _ = warm_parking(lead_index=lead_index)
  producer = planner.reference_planner
  decision = producer.parking_approach.decision
  assert decision.policy_kind == 'geometry_margin_only'
  assert decision.eligible and decision.allow_compact and decision.limiting_lead == lead_index
  assert decision.accel_cap is None and not decision.require_stop and not decision.release_blocked
  assert producer.parking_mpc_distances == producer.parking_policy_distances
  assert producer.parking_mpc_distances[0] < 4.0
  assert producer.parking_mpc_distances[1][lead_index] < 6.0
  assert np.min(planner.mpc.params[:, COMFORT_STOP_DISTANCE]) < 4.0
  assert np.min(planner.mpc.params[:, RISK_STOP_DISTANCE]) < 6.0
  # A fixed accepted reference isolates the scalar chain. Geometry may change
  # real native requests, but there is no additional parking acceleration cap.
  assert planner.output_a_target == pytest.approx(.4)
  assert producer.constraint_source == LongitudinalPlanSource.cruise


def test_deficient_margin_tightens_native_geometry_without_inventing_stop_evidence():
  planner, _, _ = warm_parking(final_gap=5.0)
  producer = planner.reference_planner
  decision = producer.parking_approach.decision
  assert decision.eligible and decision.margin < 0.0 and not decision.allow_compact
  assert producer.parking_policy_distances[0] < 4.0
  assert producer.parking_mpc_distances == (4.0, (6.0, 6.0))
  assert np.min(planner.mpc.params[:, COMFORT_STOP_DISTANCE]) >= 4.0
  np.testing.assert_array_equal(planner.mpc.params[:, RISK_STOP_DISTANCE],
                                np.full_like(planner.mpc.params[:, RISK_STOP_DISTANCE], 6.0))
  assert decision.accel_cap is None and not decision.require_stop
  assert not decision.stop_evidence_added and not decision.release_blocked
  assert planner.output_a_target == pytest.approx(.4)
  assert not planner.motion_intent.reasons & StopReason.CLOSE_LEAD


def test_stronger_model_braking_is_preserved_with_admitted_compact_geometry():
  planner, _, _ = warm_parking(experimental=True, model_accel=-3.0)
  decision = planner.reference_planner.parking_approach.decision
  assert decision.eligible and decision.allow_compact and decision.accel_cap is None
  assert not decision.cap_effective
  assert planner.output_a_target <= -3.0
  assert planner.reference_planner.constraint_source == LongitudinalPlanSource.e2e


@pytest.mark.parametrize('fault', ['std', 'source'])
def test_uncertainty_failure_after_native_compact_use_rebuilds_full_history_in_producer(fault):
  """Native compact geometry is used before the fault; C has no scalar policy."""
  planner, _, state = warm_parking(experimental=True, model_accel=-3.0)
  producer = planner.reference_planner
  assert producer.stop_distance_policy.evidence.low_speed_history
  assert producer.parking_approach.decision.allow_compact
  assert producer.parking_mpc_distances[0] < 4.0
  assert producer.parking_mpc_distances[1][0] < 6.0
  assert np.min(planner.mpc.params[:, RISK_STOP_DISTANCE]) < 6.0

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
      assert producer.parking_approach.decision.accel_cap is None
      assert not producer.parking_approach.decision.require_stop
      if frame < 222:
        # Frame 122 starts the new history; frame 221 contains only 4.95 s.
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


def test_exact_older_model_metadata_remains_usable_for_an_async_radar_publication():
  planner, _, state = warm_parking()
  producer = planner.reference_planner
  original_model_ns = state['radarState'].mdMonoTime
  older_radar = state['radarState'].as_reader().as_builder()
  now = advance(state, 121)
  set_vision_lead(state, now, 14.97)
  state['radarState'] = older_radar
  with patch(SAMPLE_ACCEL, return_value=.4):
    planner.update(state, now_ns=now)
  decision = producer.parking_approach.decision
  assert planner.mpc.solution_status == 0
  assert decision.uncertainty_qualified and decision.uncertainty_model_ns == original_model_ns
  assert decision.observation_age == pytest.approx(.05)
  assert producer.stop_distance_policy.evidence.low_speed_history
  assert decision.allow_compact and producer.parking_mpc_distances[0] < 4.0


def test_legacy_diagnostic_fields_cannot_change_scalar_or_motion_authority():
  baseline, _ = make_planner(speed=.6)
  observed, _ = make_planner(speed=.6)
  states = [make_state(speed=.6, target=10.0) for _ in range(2)]
  update_evidence = observed.reference_planner.parking_approach.update

  def inject_obsolete_authority_fields(**kwargs):
    decision = update_evidence(**kwargs)
    # Historical actuator-candidate fields remain decodable, but are no
    # longer a production input to scalar arbitration or owner permissions.
    return replace(decision, accel_cap=-3.0, require_stop=True, cap_effective=True,
                   stop_evidence_added=True, held_episode=True, release_blocked=True)

  with patch.object(observed.reference_planner.parking_approach, 'update', side_effect=inject_obsolete_authority_fields):
    for frame in range(121):
      for planner, state in zip((baseline, observed), states, strict=True):
        now = advance(state, frame)
        set_vision_lead(state, now, 15.0 + .6 * (120 - frame) * .05)
        planner.update(state, now_ns=now)
        assert planner.mpc.solution_status == 0
      assert baseline.output_a_target == observed.output_a_target
      assert baseline.motion_intent == observed.motion_intent
      assert baseline.reference_planner.constraint_accel == observed.reference_planner.constraint_accel
      np.testing.assert_array_equal(baseline.mpc.params, observed.mpc.params)


def test_real_owner_model_hold_remains_without_a_parking_hold_state():
  planner, extension, state = warm_parking(experimental=True, model_accel=-3.0)
  state['carState'].vEgo = 0.0
  state['carState'].standstill = True
  state['modelV2'].velocity.x = [0.0] * len(state['modelV2'].velocity.x)
  state['modelV2'].action.shouldStop = True
  now = advance(state, 121)
  set_vision_lead(state, now, 6.4)
  with patch(SAMPLE_ACCEL, return_value=.4):
    planner.update(state, now_ns=now)
  decision = planner.reference_planner.parking_approach.decision
  assert decision.accel_cap is None and not decision.require_stop
  assert not decision.held_episode and not decision.release_blocked
  assert planner.output_a_target <= -3.0
  assert planner.motion_intent.reasons & StopReason.E2E
  owner = ControlHarness(LongControl(planner.CP, extension))
  owner.state.vEgo = 0.0
  owner.state.standstill = True
  for _ in range(60):
    owner.step(planner.output_a_target, planner.motion_intent)
    assert owner.control.motion_phase == MotionPhase.HOLD
    assert owner.control.last_output_accel <= planner.CP.stopAccel


def test_lead_loss_after_native_compact_use_withdraws_only_geometry():
  planner, _, state = warm_parking()
  producer = planner.reference_planner
  assert producer.parking_approach.decision.allow_compact
  assert np.min(planner.mpc.params[:, RISK_STOP_DISTANCE]) < 6.0
  now = advance(state, 121)
  state['radarState'].leadOne.present = False
  with patch(SAMPLE_ACCEL, return_value=.4):
    planner.update(state, now_ns=now)
  decision = producer.parking_approach.decision
  assert not decision.allow_compact
  assert producer.parking_mpc_distances == (4.0, (6.0, 6.0))
  assert decision.accel_cap is None and not decision.require_stop and not decision.release_blocked
  assert planner.output_a_target == pytest.approx(.4)
  assert not planner.motion_intent.reasons & StopReason.CLOSE_LEAD


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
