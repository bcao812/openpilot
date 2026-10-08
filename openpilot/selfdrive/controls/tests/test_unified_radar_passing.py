"""Passing admission through the real UNIFIED/native/final-min combination.

The scalar policy and DEC mode are prescribed inputs; the native solve,
reference sampling, independent model cap and final arbitration are real.
These are command-contract checks, not a claim of on-road collision avoidance.
"""
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.drive_helpers import CLOSE_LEAD_STOP_SPEED
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import QP_FEASIBILITY_TOL, T_IDXS
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_lead_qualification import caution_lead
from openpilot.selfdrive.controls.tests.test_radar_consumption import coast, native_frame, radar_frame
from openpilot.selfdrive.controls.tests.test_radar_fusion_mpc import EPOCH_NS, late_passing_objects
from openpilot.selfdrive.controls.tests.test_unified_radar_consumption import enabled_planner, state_at
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import (
  CruiseTarget, LongitudinalPlannerSP, LongitudinalPlanSource,
)


SPEED = 11.190595626831055


@contextmanager
def controlled_mode(planner, *, curve_policy=False, blended=True):
  def targets(instance, state, speed, accel, cruise):
    instance.source = LongitudinalPlanSource.sccVision if curve_policy else LongitudinalPlanSource.cruise
    target = CruiseTarget(instance.source, max(0.0, speed - 1.0) if curve_policy else cruise,
                          -.4 if curve_policy else accel)
    instance.cruise_target = target
    return target

  with patch.object(planner, 'is_e2e', return_value=blended), \
       patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'active', return_value=True), \
       patch.object(LongitudinalPlannerSP, 'update', return_value=None), \
       patch.object(LongitudinalPlannerSP, 'update_targets', new=targets):
    yield


def passing_state(*, initial_accel=.16649655997753143, blocker=None, action_cap=.3):
  planner = enabled_planner(SPEED)
  objects = list(late_passing_objects())
  for lead in objects:
    lead.vRel = lead.vLead - SPEED
    lead.radar, lead.radarTrackId = True, lead.evidence.trackId
  if blocker == 'caution':
    objects.append(caution_lead(now_ns=EPOCH_NS, dRel=15.0, vRel=-SPEED, vLead=0.0, vLeadK=0.0,
                                yRel=1.6, evidence={'sessionId': 'radar-test', 'trackId': 12, 'pathOffset': 1.6}))
  state = state_at(radar_frame(*objects, publication=EPOCH_NS, session='radar-test'),
                   speed=SPEED, now=EPOCH_NS, publication=EPOCH_NS, target=20.0)
  state['carState'].aEgo = initial_accel
  state['controlsState'].motionState.phase = 'track'
  state['modelV2'].action.desiredAcceleration = action_cap
  if blocker == 'negative_model':
    state['modelV2'].action.desiredAcceleration = -.7
  elif blocker == 'model_stop':
    state['modelV2'].action.shouldStop = True
  elif blocker == 'cruise_history_brake':
    planner.reference_planner.cruise_response.update(0.0, SPEED, 2.0, measured_accel=-.4, reset=True)
  elif blocker in ('cruise_stop', 'cruise_below_witness'):
    state['carState'].vCruise = (0.0 if blocker == 'cruise_stop' else SPEED - .2) * 3.6
  elif blocker == 'force_decel':
    state['controlsState'].forceDecel = True
  elif blocker == 'fusion_degraded':
    state['radarState'].fusionDegraded = True
  elif blocker == 'producer_loss':
    state['radarState'].fusionRiskUnresolved = True
  elif blocker == 'consumer_loss':
    publication = EPOCH_NS - 100_000_000
    lost = coast(publication=publication, relevant=True, track=99)
    planner.reference_planner.radar_consumption.update(
      native_frame(radar_frame(lost, publication=publication)), publication_ns=publication,
      now_ns=EPOCH_NS - 40_000_000, speed=SPEED,
    )
    assert planner.reference_planner.radar_consumption.degraded
  elif blocker in ('gas', 'brake'):
    setattr(state['carState'], blocker + 'Pressed', True)
  elif blocker == 'override':
    state['carControl'].cruiseControl.override = True
  elif blocker == 'short_preview':
    planner.CP.longitudinalActuatorDelay = 0.0
  elif blocker == 'held_stop':
    planner.reference_planner.radar_held_identity = ('radar-test', 5)
  elif blocker == 'model_rest':
    state['modelV2'].velocity.x = [0.0] * 33
    state['modelV2'].position.x = [0.0] * 33
  elif blocker == 'model_unknown':
    state.logMonoTime['modelV2'] = EPOCH_NS + 1
  return planner, state


def run_case(*, blocker=None, initial_accel=.16649655997753143, action_cap=.3):
  planner, state = passing_state(initial_accel=initial_accel, blocker=blocker, action_cap=action_cap)
  with controlled_mode(planner, curve_policy=blocker == 'curve_policy'), \
       patch.object(planner.mpc, 'update', wraps=planner.mpc.update) as native_update:
    if blocker == 'negative_maximum':
      with patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.cruise_accel_limit', return_value=-.3):
        planner.update(state, now_ns=EPOCH_NS)
    else:
      planner.update(state, now_ns=EPOCH_NS)
  assert native_update.call_count == 1
  return planner, state, native_update.call_args.kwargs['reference']


@pytest.mark.parametrize('initial_accel', [-.5, 0.0, .16649655997753143])
@pytest.mark.parametrize('action_cap', [0.0, .3])
def test_passing_final_request_matches_actual_native_a_interpolation(initial_accel, action_cap):
  planner, state, reference = run_case(initial_accel=initial_accel, action_cap=action_cap)
  assert reference.allow_passing and reference.typed_model is None and reference.model_velocity is None
  assert planner.reference_planner.model_reference_mode == 'action_capped_absolute'
  assert planner.mpc.solution_status == 0 and planner.mpc.diagnostic_passing['accepted']
  preview = planner.CP.longitudinalActuatorDelay + .05
  assert preview >= T_IDXS[1]
  actual_native_request = float(np.interp(preview, T_IDXS, planner.mpc.a_solution))
  assert planner.reference_planner.reference_accel == actual_native_request
  assert actual_native_request <= QP_FEASIBILITY_TOL
  assert planner.output_a_target == pytest.approx(actual_native_request, abs=QP_FEASIBILITY_TOL, rel=0.0)
  assert not planner.output_should_stop and not planner.reference_planner.caution_accel_limited
  assert planner.mpc.finite_risk_braking_accel is None


@pytest.mark.parametrize('blocker', [
  'negative_model', 'model_stop', 'cruise_history_brake', 'cruise_stop', 'cruise_below_witness', 'force_decel',
  'negative_maximum', 'curve_policy', 'caution', 'gas', 'brake', 'override', 'short_preview',
  'held_stop', 'model_rest', 'model_unknown',
])
def test_independent_final_braking_or_uncertainty_vetoes_passing_before_native_solve(blocker):
  planner, state, reference = run_case(blocker=blocker)
  assert not reference.allow_passing
  assert not (planner.mpc.diagnostic_passing or {}).get('accepted', False)
  assert planner.mpc._passing_witness is None
  if blocker == 'negative_model':
    assert planner.reference_planner.model_reference_mode == 'action_capped_absolute'
    assert reference.typed_model is None and reference.model_velocity is None
    assert planner.output_a_target <= state['modelV2'].action.desiredAcceleration
  if blocker == 'cruise_history_brake':
    assert planner.reference_planner.cruise_response._previous_accel is None or planner.reference_planner.cruise_response._previous_accel < 0.0
  if blocker == 'short_preview':
    assert planner.CP.longitudinalActuatorDelay + .05 < T_IDXS[1]
  # These late windows cannot borrow urgent finite-risk authority merely
  # because a conservative behind branch or an ordinary solve then fails.
  assert planner.mpc.finite_risk_braking_accel is None


@pytest.mark.parametrize('diagnostic', ['fusion_degraded', 'producer_loss', 'consumer_loss'])
def test_historical_radar_diagnostics_do_not_veto_a_fresh_passing_solution(diagnostic):
  baseline, _, expected = run_case()
  planner, _, reference = run_case(blocker=diagnostic)
  assert expected.allow_passing and reference.allow_passing
  assert planner.mpc.solution_status == baseline.mpc.solution_status == 0
  assert planner.mpc.diagnostic_passing['accepted'] and baseline.mpc.diagnostic_passing['accepted']
  np.testing.assert_array_equal(planner.mpc.params, baseline.mpc.params)
  assert planner.output_a_target == pytest.approx(baseline.output_a_target, abs=QP_FEASIBILITY_TOL, rel=0.0)
  assert planner.motion_intent.release_allowed == baseline.motion_intent.release_allowed
  assert not planner.output_should_stop
  if diagnostic == 'consumer_loss':
    assert planner.reference_planner.radar_consumption.degraded


def test_passing_authorization_does_not_persist_into_a_later_braking_frame():
  planner, state, reference = run_case()
  assert reference.allow_passing and planner.mpc.diagnostic_passing['accepted']
  now = EPOCH_NS + 50_000_000
  for service in state.logMonoTime:
    state.logMonoTime[service] = now
  objects = late_passing_objects()
  for lead in objects:
    for field in ('positionMonoTime', 'velocityMonoTime', 'estimateMonoTime', 'modelMonoTime'):
      setattr(lead.evidence, field, now)
  state['radarState'] = native_frame(radar_frame(*objects, publication=now, session='radar-test'))
  state['modelV2'].timestampEof = now
  state['modelV2'].action.desiredAcceleration = -.8
  with controlled_mode(planner), patch.object(planner.mpc, 'update', wraps=planner.mpc.update) as native_update:
    planner.update(state, now_ns=now)
  assert not native_update.call_args.kwargs['reference'].allow_passing
  assert planner.mpc._passing_witness is None
  assert planner.output_a_target <= -.8


@pytest.mark.parametrize('speed,allowed', [
  (float(np.nextafter(np.float32(CLOSE_LEAD_STOP_SPEED), np.float32(0.0))), False),
  (CLOSE_LEAD_STOP_SPEED, True),
])
def test_passing_respects_the_existing_low_speed_owner_boundary(speed, allowed):
  planner, state = passing_state(initial_accel=0.0)
  state['carState'].vEgo = speed
  state['modelV2'].velocity.x = [speed] * 33
  state['modelV2'].position.x = [value * speed / SPEED for value in state['modelV2'].position.x]
  for lead in state['radarState'].radarObjects:
    lead.vRel = lead.vLead - speed
  with controlled_mode(planner), patch.object(planner.mpc, 'update', wraps=planner.mpc.update) as native_update:
    planner.update(state, now_ns=EPOCH_NS)
  reference = native_update.call_args.kwargs['reference']
  assert reference.allow_passing is allowed
  if not allowed:
    assert planner.mpc._passing_witness is None
