"""Retired occupancy objects cannot change ordinary UNIFIED lead planning.

The native solve, reference sampling and final arbitration remain real; only
the scalar policy and DEC mode are prescribed inputs.
"""
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import CapturingPublisher
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_lead_qualification import caution_lead, formal_lead
from openpilot.selfdrive.controls.tests.test_radar_consumption import expiring_lead, native_frame, radar_frame
from openpilot.selfdrive.controls.tests.test_radar_fusion_mpc import EPOCH_NS
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


def run_case(*, extra_kind=None, initial_accel=.16649655997753143, action_cap=.3, diagnostic=None):
  planner = enabled_planner(SPEED)
  front = formal_lead(now_ns=EPOCH_NS, dRel=40.0, vLead=SPEED + .5, vLeadK=SPEED + .5, vRel=.5,
                      evidence={'sessionId': 'radar-test', 'trackId': 5})
  extras = []
  if extra_kind is not None:
    extra = formal_lead(now_ns=EPOCH_NS, dRel=.8, vLead=0.0, vLeadK=0.0, vRel=-SPEED,
                        evidence={'sessionId': 'radar-test', 'trackId': 7, 'followQualified': False,
                                  'conflictStart': 0.0, 'conflictEnd': .15})
    if extra_kind == 'future':
      extra.evidence.pathState, extra.evidence.conflictStart = 'entering', .05
    elif extra_kind == 'caution':
      extra = caution_lead(now_ns=EPOCH_NS, dRel=.8, vLead=0.0, vLeadK=0.0, vRel=-SPEED)
    elif extra_kind == 'stale':
      extra.evidence.positionMonoTime -= 500_000_000
    elif extra_kind == 'malformed':
      extra.dRel = float('nan')
    extras = [extra] * (40 if extra_kind == 'oversize' else 1)
  state = state_at(radar_frame(*extras, publication=EPOCH_NS, display=(front,), session='radar-test',
                               degraded=diagnostic == 'fusion_degraded', risk_unresolved=diagnostic == 'legacy_loss_flag'),
                   speed=SPEED, now=EPOCH_NS, publication=EPOCH_NS, target=20.0)
  state['carState'].aEgo = initial_accel
  state['controlsState'].motionState.phase = 'track'
  state['modelV2'].action.desiredAcceleration = action_cap
  if diagnostic == 'prior_expiry':
    publication = EPOCH_NS - 100_000_000
    expired = expiring_lead(publication=publication, relevant=True, track=99)
    planner.reference_planner.radar_consumption.update(
      native_frame(radar_frame(expired, publication=publication, display=(expired,))), publication_ns=publication,
      now_ns=EPOCH_NS - 40_000_000, speed=SPEED,
    )
    assert planner.reference_planner.radar_consumption.expired_identities
    assert not planner.reference_planner.radar_consumption.degraded
  with controlled_mode(planner):
    planner.update(state, now_ns=EPOCH_NS)
  publisher = CapturingPublisher()
  with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=EPOCH_NS):
    planner.publish(state, publisher)
  return planner, publisher.messages['longitudinalPlan'].longitudinalPlan


@pytest.mark.parametrize('extra_kind', ['finite', 'future', 'caution', 'stale', 'malformed', 'oversize'])
@pytest.mark.parametrize('initial_accel', [-.5, .16649655997753143])
def test_extra_objects_preserve_the_formal_front_native_problem_and_final_request(extra_kind, initial_accel):
  baseline, clear_plan = run_case(initial_accel=initial_accel)
  planner, plan = run_case(extra_kind=extra_kind, initial_accel=initial_accel)
  assert planner.mpc.solution_status == baseline.mpc.solution_status == 0
  assert planner.motion_intent.valid and not planner.fcw
  assert not planner.reference_planner.radar_consumption.degraded
  np.testing.assert_array_equal(planner.mpc.params, baseline.mpc.params)
  np.testing.assert_array_equal(planner.mpc.x_sol, baseline.mpc.x_sol)
  np.testing.assert_array_equal(planner.mpc.u_sol, baseline.mpc.u_sol)
  assert plan.aTarget == clear_plan.aTarget and plan.shouldStop == clear_plan.shouldStop
  assert plan.motionIntent.to_dict() == clear_plan.motionIntent.to_dict()
  assert [row.to_dict() for row in plan.objectInfluences] == [row.to_dict() for row in clear_plan.objectInfluences]
  assert all(row.trackId == 5 for row in plan.objectInfluences)


@pytest.mark.parametrize('action_cap', [-.8, 0.0, .3])
def test_ordinary_formal_lead_keeps_native_sampling_and_independent_model_cap(action_cap):
  planner, plan = run_case(extra_kind='finite', action_cap=action_cap)
  assert planner.mpc.solution_status == 0
  preview = planner.CP.longitudinalActuatorDelay + .05
  native_request = float(np.interp(preview, T_IDXS, planner.mpc.a_solution))
  assert planner.reference_planner.reference_accel == native_request
  assert plan.aTarget <= min(native_request, action_cap) + 1e-6


@pytest.mark.parametrize('diagnostic', ['fusion_degraded', 'legacy_loss_flag', 'prior_expiry'])
def test_historical_radar_diagnostics_do_not_change_a_fresh_current_front_solution(diagnostic):
  baseline, clear_plan = run_case()
  planner, plan = run_case(diagnostic=diagnostic)
  assert planner.mpc.solution_status == baseline.mpc.solution_status == 0
  np.testing.assert_array_equal(planner.mpc.params, baseline.mpc.params)
  assert plan.aTarget == clear_plan.aTarget
  assert plan.motionIntent.releaseAllowed == clear_plan.motionIntent.releaseAllowed
  assert not planner.reference_planner.radar_consumption.degraded
