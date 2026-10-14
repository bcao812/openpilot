"""Accepted acceleration history needs current formal risk support.

These tests inspect the installed native problem as well as actual solutions.
They do not treat a successful soft-constrained solve as collision avoidance.
"""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LongitudinalMpc, MpcReference, N, T_IDXS,
)
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_mpc_problem_contract import NativeProblemRecorder
from openpilot.selfdrive.controls.tests.test_radar_fusion_mpc import EPOCH_NS, action, advance, fused_state, object_lead
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state


STEP_NS = 50_000_000


def history_mpc():
  mpc = LongitudinalMpc()
  mpc.solver = NativeProblemRecorder(mpc.solver)
  mpc.diagnostic_obstacles_enabled = True
  return mpc


def history_tick(mpc, radar, timestamp, *, healthy=True, accept=True, speed=20.0, acceleration=0.398):
  history = mpc.prepare_reference_history(timestamp, valid=True)
  mpc.set_weights(history)
  mpc.set_cur_state(speed, acceleration)
  mpc.update(radar, reference=MpcReference(25.0), now_ns=timestamp, reference_ns=timestamp,
             reference_inputs_valid=healthy)
  accepted = mpc.mark_reference_history(valid=healthy and accept)
  return history, accepted


def installed_history_weights(mpc):
  return np.array([mpc.solver.solved_costs[-1][stage, 'W'][4, 4] for stage in range(N + 1)])


def seed_reference(mpc, *, source='radar', track=1883):
  lead = object_lead(track, distance=45.0, speed=8.0, ego_speed=20.0, source=source)
  history, accepted = history_tick(mpc, fused_state(display=(lead,)), EPOCH_NS)
  assert not history and accepted and mpc.solution_status == 0
  return lead


@pytest.mark.parametrize('remaining', ['none', 'replacement', 'anonymous'])
def test_same_current_inputs_ignore_different_withdrawn_acceleration_histories(remaining):
  solutions, parameters = [], []
  for old_acceleration in (-0.25, -3.0):
    mpc = history_mpc()
    seed_reference(mpc)
    # Hold the accepted object's support, native warm start and current input
    # fixed; vary only the old smoothing reference at its public boundary.
    mpc._reference_acceleration[:] = old_acceleration
    stamp = EPOCH_NS + STEP_NS
    current = (() if remaining == 'none' else
               (object_lead(1872, distance=60.0, speed=20.0, ego_speed=20.0,
                            source='vision' if remaining == 'anonymous' else 'radar', epoch=stamp),))
    history, accepted = history_tick(mpc, fused_state(display=current), stamp)
    assert history and accepted and mpc.solution_status == 0
    diagnostic = mpc.diagnostic_reference_history
    assert diagnostic['reason'] == 'formal_support_removed'
    assert diagnostic['supportChanged'] and mpc.reference_support_changed
    assert diagnostic['previousSupport'][0]['identity'] == ('radar-test', 1883)
    assert not any(row['identity'] == ('radar-test', 1883) for row in diagnostic['currentSupport'])
    np.testing.assert_array_equal(installed_history_weights(mpc), 0.0)
    np.testing.assert_array_equal(mpc.params[:, 3], 0.0)
    assert mpc.solver.solved_costs[-1][0, 'W'][5, 5] > 0.0
    assert mpc.x0[2] == pytest.approx(0.398)
    assert mpc.x_sol[0, 2] == pytest.approx(0.398)
    solutions.append(mpc.x_sol.copy())
    parameters.append(mpc.params.copy())
  np.testing.assert_array_equal(parameters[0], parameters[1])
  np.testing.assert_allclose(solutions[0], solutions[1], atol=1e-10, rtol=0.0)


def test_rebased_reference_keeps_new_emergency_lead_and_its_braking():
  mpc = history_mpc()
  seed_reference(mpc)
  stamp = EPOCH_NS + STEP_NS
  current = object_lead(1872, distance=35.0, speed=0.0, ego_speed=20.0, epoch=stamp)
  _, accepted = history_tick(mpc, fused_state(display=(current,)), stamp)
  assert accepted and mpc.solution_status == 0
  assert mpc.reference_support_changed
  np.testing.assert_array_equal(installed_history_weights(mpc), 0.0)
  assert mpc.source_object_identity == ('radar-test', 1872)
  assert ('radar-test', 1872) in mpc.object_influence_identities
  assert len(mpc._lead_trajectories) == 1
  assert action(mpc) < 0.0
  assert np.min(mpc.v_solution[1:]) < mpc.x0[1]
  # This is current-input braking evidence, not a guarantee that these soft
  # bounds avoid every collision in this close stationary-lead fixture.


@pytest.mark.parametrize('change', ['swap_slots', 'generation', 'radar_to_fused'])
def test_current_physical_support_keeps_history_through_nonphysical_changes(change):
  mpc = history_mpc()
  first = object_lead(1, distance=45.0, speed=10.0, ego_speed=20.0)
  second = object_lead(2, distance=65.0, speed=15.0, ego_speed=20.0)
  _, accepted = history_tick(mpc, fused_state(display=(first, second)), EPOCH_NS)
  assert accepted
  previous = mpc._reference_acceleration.copy()
  stamp = EPOCH_NS + STEP_NS
  for lead in (first, second):
    advance(lead, stamp)
  if change == 'swap_slots':
    first, second = second, first
  elif change == 'generation':
    first.evidence.generation += 1
    second.evidence.generation += 7
  else:
    first.evidence.source = 'fused'
  history, accepted = history_tick(mpc, fused_state(display=(first, second)), stamp)
  assert history and accepted
  assert not mpc.reference_support_changed
  assert mpc.diagnostic_reference_history['reason'] == 'inherited'
  assert np.any(installed_history_weights(mpc) > 0.0)
  np.testing.assert_allclose(mpc.params[:, 3], np.interp(T_IDXS + 0.05, T_IDXS, previous))


@pytest.mark.parametrize('source', ['radar', 'vision'])
def test_same_formal_lead_losing_follow_and_stop_rebuilds_history(source):
  mpc = history_mpc()
  lead = seed_reference(mpc, source=source)
  stamp = EPOCH_NS + STEP_NS
  advance(lead, stamp)
  lead.evidence.followQualified = lead.evidence.stopQualified = False
  lead.evidence.departureQualified = lead.evidence.fcwQualified = False
  _, accepted = history_tick(mpc, fused_state(display=(lead,)), stamp)
  assert accepted and mpc.reference_support_changed
  expected = 'formal_qualification_reduced' if source == 'radar' else 'anonymous_support_removed'
  assert mpc.diagnostic_reference_history['reason'] == expected
  assert mpc.diagnostic_reference_history['currentSupport'][0]['purposes'] == ('brake',)
  np.testing.assert_array_equal(installed_history_weights(mpc), 0.0)
  assert action(mpc) < 0.0  # Current brake-qualified geometry remains active.


@pytest.mark.parametrize('source', ['vision', 'legacy'])
def test_observable_anonymous_support_withdrawal_rebuilds_history(source):
  mpc = history_mpc()
  radar = (fused_state(display=(object_lead(source='vision', distance=45.0, speed=8.0, ego_speed=20.0),))
           if source == 'vision' else radar_state(distance=45.0, speed=8.0))
  assert history_tick(mpc, radar, EPOCH_NS)[1]
  _, accepted = history_tick(mpc, radar_state(), EPOCH_NS + STEP_NS)
  assert accepted and mpc.reference_support_changed
  assert mpc.diagnostic_reference_history['reason'] == 'anonymous_support_removed'
  assert mpc.diagnostic_reference_history['previousSupport'][0]['identity'] is None
  np.testing.assert_array_equal(installed_history_weights(mpc), 0.0)


def test_anonymous_same_purposes_do_not_claim_a_physical_identity_or_reset_every_frame():
  mpc = history_mpc()
  lead = seed_reference(mpc, source='vision')
  stamp = EPOCH_NS + STEP_NS
  advance(lead, stamp)
  history, accepted = history_tick(mpc, fused_state(display=(lead,)), stamp)
  assert history and accepted and not mpc.reference_support_changed
  assert np.any(installed_history_weights(mpc) > 0.0)
  diagnostic = mpc.diagnostic_reference_history
  assert not diagnostic['anonymousIdentityKnown']
  assert diagnostic['currentSupport'][0]['identity'] is None
  # A 1 -> 1 anonymous replacement is unobservable to this contract. It is
  # explicitly not covered by the physical radar identity withdrawal fix.


@pytest.mark.parametrize('fault', ['services', 'fusion_degraded', 'fusion_unhealthy'])
def test_bad_input_is_not_a_healthy_empty_support_observation(fault):
  mpc = history_mpc()
  seed_reference(mpc)
  radar = fused_state()
  radar.fusionDegraded = fault == 'fusion_degraded'
  radar.fusionHealthy = fault != 'fusion_unhealthy'
  _, accepted = history_tick(mpc, radar, EPOCH_NS + STEP_NS, healthy=fault != 'services')
  assert not accepted and not mpc.reference_support_changed
  assert mpc.diagnostic_reference_history['reason'] == 'input_health_unqualified'
  assert mpc._reference_epoch_ns is None and mpc._reference_support is None
  np.testing.assert_array_equal(installed_history_weights(mpc), 0.0)
  # This history check grants neither valid motion intent nor a healthy
  # sensor status; the caller retains its existing input-failure handling.


@pytest.mark.parametrize('failure', ['solver_status', 'nonfinite', 'producer_rejection'])
def test_failed_rebuild_cannot_commit_or_redate_the_withdrawn_reference(failure):
  mpc = history_mpc()
  seed_reference(mpc)
  mpc.solver.failure = failure if failure != 'producer_rejection' else None
  stamp = EPOCH_NS + STEP_NS
  _, accepted = history_tick(mpc, fused_state(), stamp, accept=failure != 'producer_rejection')
  assert not accepted
  assert mpc.reference_support_changed
  assert mpc._reference_epoch_ns is None and mpc._reference_support is None
  assert mpc._reference_pending_ns is None
  assert mpc.diagnostic_reference_history['reason'] == 'formal_support_removed'
  mpc.solver.failure = None
  history, accepted = history_tick(mpc, fused_state(), stamp + STEP_NS)
  assert not history and accepted
  np.testing.assert_array_equal(installed_history_weights(mpc), 0.0)
  assert mpc._reference_support == ()
  history, accepted = history_tick(mpc, fused_state(), stamp + 2 * STEP_NS)
  assert history and accepted
  assert np.any(installed_history_weights(mpc) > 0.0)


def test_unselected_object_cannot_reauthorize_the_old_reference():
  mpc = history_mpc()
  old = seed_reference(mpc)
  stamp = EPOCH_NS + STEP_NS
  advance(old, stamp)
  current = object_lead(1872, distance=70.0, speed=20.0, ego_speed=20.0, epoch=stamp)
  _, accepted = history_tick(mpc, fused_state(old, display=(current,)), stamp)
  assert accepted and mpc.reference_support_changed
  assert mpc.diagnostic_reference_history['reason'] == 'formal_support_removed'
  assert tuple(item.identity for item in mpc._reference_support) == (('radar-test', 1872),)


def test_unprepared_legacy_problem_keeps_its_existing_history_contract():
  mpc = history_mpc()
  mpc.set_cur_state(20.0, 0.398)
  mpc.set_weights(True)
  mpc.a_prev[:] = -1.0
  radar = fused_state()
  radar.fusionHealthy = False
  mpc.update(radar, reference=MpcReference(25.0))
  assert mpc.solution_status == 0
  np.testing.assert_array_equal(mpc.params[:, 3], -1.0)
  assert np.any(installed_history_weights(mpc) > 0.0)
  assert not mpc.reference_support_changed


def test_successful_rebuild_records_current_support_before_later_withdrawal():
  mpc = history_mpc()
  first = seed_reference(mpc)
  stamp = EPOCH_NS + STEP_NS
  second = object_lead(1872, distance=55.0, speed=12.0, ego_speed=20.0, epoch=stamp)
  assert history_tick(mpc, fused_state(display=(second,)), stamp)[1]
  assert mpc.reference_support_changed
  assert tuple(item.identity for item in mpc._reference_support) == (('radar-test', 1872),)
  advance(second, stamp + STEP_NS)
  assert history_tick(mpc, fused_state(display=(second,)), stamp + STEP_NS)[1]
  assert not mpc.reference_support_changed
  advance(first, stamp + 2 * STEP_NS)
  assert history_tick(mpc, fused_state(display=(first,)), stamp + 2 * STEP_NS)[1]
  assert mpc.reference_support_changed
  assert mpc.diagnostic_reference_history['previousSupport'][0]['identity'] == ('radar-test', 1872)


def test_qualification_rebuild_failure_withdraws_same_identity_continuation_donor():
  from openpilot.selfdrive.controls.tests.test_lead_qualification import formal_lead
  from openpilot.selfdrive.controls.tests.test_radar_consumption import radar_frame
  from openpilot.selfdrive.controls.tests.test_unified_radar_consumption import enabled_planner, state_at

  planner = enabled_planner(20.0)
  primary = formal_lead(now_ns=EPOCH_NS, dRel=45.0, vLead=8.0, vLeadK=8.0, vRel=-12.0)
  state = state_at(radar_frame(primary, display=(primary,)), speed=20.0, target=25.0,
                   now=EPOCH_NS, publication=EPOCH_NS)
  with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
    planner.update(state, now_ns=EPOCH_NS)
  assert planner.motion_intent.valid
  recorder = NativeProblemRecorder(planner.mpc.solver)
  recorder.failure = 'solver_status'
  planner.mpc.solver = recorder
  continuation = planner.reference_planner.continuation
  continuation.donor = object()  # A revoked donor must be discarded before its contents are inspected.
  stamp = EPOCH_NS + STEP_NS
  reduced = deepcopy(primary)
  for field in ('positionMonoTime', 'velocityMonoTime', 'estimateMonoTime', 'modelMonoTime'):
    setattr(reduced.evidence, field, stamp)
  reduced.evidence.followQualified = reduced.evidence.stopQualified = False
  reduced.evidence.departureQualified = reduced.evidence.fcwQualified = False
  reduced.evidence.followGeometryMonoTime = 0
  state = state_at(radar_frame(reduced, display=(reduced,)), speed=20.0, target=25.0,
                   now=stamp, publication=stamp)
  with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
    planner.update(state, now_ns=stamp)
  assert planner.mpc.solution_status == 4
  assert planner.mpc.reference_support_changed
  assert not planner.motion_intent.valid
  assert continuation.donor is None and continuation.offer is None
  assert planner.mpc._reference_epoch_ns is None and planner.mpc._reference_support is None
