from unittest.mock import Mock, patch

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_continuation import CONTINUATION_VERSION
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  get_T_FOLLOW, get_safe_obstacle_distance, get_stopped_equivalence_factor,
)
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_TIMING_VERSION
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_planner, make_state, advance, Publisher
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_lead_qualification import formal_lead
from openpilot.selfdrive.controls.tests.test_radar_consumption import native_frame, radar_frame
from openpilot.selfdrive.controls.tests.test_unified_radar_consumption import enabled_planner


def acknowledge_native(state, planner, now_ns):
  """A recorded owner acknowledgement, separate from successful production."""
  timing = planner.reference_timing.pending
  motion = state['controlsState'].motionState
  motion.version = MOTION_INTENT_VERSION
  motion.continuationVersion = CONTINUATION_VERSION
  motion.continuationActive = False
  motion.enabled = True
  motion.phase = 'track'
  motion.reasons = 0
  motion.resolvedStop = motion.degraded = motion.waitingForFreshInput = False
  motion.referenceTimingVersion = REFERENCE_TIMING_VERSION
  motion.referenceTimingValid = True
  motion.referenceSourceMonoTime = motion.referenceEpochMonoTime = timing.source_ns
  motion.referenceProducerMonoTime = timing.producer_ns
  motion.referenceProducerInstanceMonoTime = timing.instance_ns
  motion.referenceConsumerMonoTime = now_ns + 1_000_000
  motion.modelMonoTime = state.logMonoTime['modelV2']
  state.logMonoTime['controlsState'] = state.logMonoTime['carControl'] = now_ns + 2_000_000


def ready_planner():
  planner, _ = make_planner(speed=20.0)
  state = make_state(speed=20.0, target=25.0)
  # The first healthy publication bootstraps an owner receipt; the following
  # publication is the actual native donor, then acknowledged independently.
  now = advance(state, 0)
  planner.update(state, now_ns=now)
  assert planner.mpc.solution_status == 0 and planner.motion_intent.valid
  acknowledge_native(state, planner, now)
  now = advance(state, 1)
  planner.update(state, now_ns=now)
  assert planner.mpc.solution_status == 0 and planner.motion_intent.valid
  acknowledge_native(state, planner, now)
  assert planner.reference_planner.continuation.donor is not None
  return planner, state


def fail_next(planner, state, frame=2, status=4):
  now = advance(state, frame)
  if status == 4:
    solver = planner.mpc.solver
    planner.mpc.solver = Mock(wraps=solver)
    planner.mpc.solver.solve.return_value = 4
    try:
      planner.update(state, now_ns=now)
    finally:
      planner.mpc.solver = solver
  else:
    with patch.object(planner.mpc, 'run', side_effect=lambda: planner.mpc.fail(status)):
      planner.update(state, now_ns=now)
  return now


@pytest.mark.parametrize('status', [4])
def test_qualified_native_donor_has_separate_candidate_without_rewriting_failure(status):
  planner, state = ready_planner()
  now = fail_next(planner, state, status=status)
  carry = planner.reference_planner.continuation
  assert carry.offer is not None, carry.reason
  assert planner.mpc.solution_status == status
  assert not planner.motion_intent.valid
  assert planner.output_a_target <= -0.5
  assert planner.motion_intent.braking_accel == planner.output_a_target
  np.testing.assert_array_equal(planner.a_desired_trajectory, 0.0)
  assert carry.offer.candidate.target_accel > -0.1
  assert carry.offer.candidate.speed_tolerance > 0.0
  assert carry.offer.candidate.accel_tolerance > 0.0
  assert carry.offer.donor_source_ns == 1_050_000_000
  publisher = Publisher()
  with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=now + 1_000_000):
    planner.publish(state, publisher)
  message = publisher.messages['longitudinalPlan']
  assert not message.valid and message.longitudinalPlan.solverStatus == status
  assert message.longitudinalPlan.continuation.valid
  assert len(message.longitudinalPlan.continuation.candidate.speeds) == 13
  assert len(message.longitudinalPlan.continuation.candidate.jerks) == 12
  assert message.longitudinalPlan.aTarget <= -0.5


@pytest.mark.parametrize('status', [-1, -6, -7])
def test_input_or_retired_failure_keeps_original_protection_and_never_offers_continuation(status):
  planner, state = ready_planner()
  fail_next(planner, state, status=status)
  carry = planner.reference_planner.continuation
  assert carry.offer is None and carry.donor is None
  assert carry.reason == 'unclassified_or_input_failure'
  assert planner.output_a_target <= -0.5


@pytest.mark.parametrize('case', ['no_ack', 'old_consumer', 'hold', 'degraded', 'waiting', 'gas', 'brake',
                                  'override', 'force_decel', 'standstill', 'model_stop', 'model_brake'])
def test_current_authority_and_risk_veto_the_donor(case):
  planner, state = ready_planner()
  motion = state['controlsState'].motionState
  if case == 'no_ack':
    motion.referenceProducerMonoTime -= 1
  elif case == 'old_consumer':
    motion.continuationVersion = 0
  elif case == 'hold':
    motion.phase, motion.resolvedStop = 'hold', True
  elif case == 'degraded':
    motion.degraded = True
  elif case == 'waiting':
    motion.waitingForFreshInput = True
  elif case in ('gas', 'brake'):
    setattr(state['carState'], case + 'Pressed', True)
  elif case == 'override':
    state['carControl'].cruiseControl.override = True
  elif case == 'force_decel':
    state['controlsState'].forceDecel = True
  elif case == 'standstill':
    state['carState'].standstill = True
  elif case == 'model_stop':
    state['modelV2'].action.shouldStop = True
  elif case == 'model_brake':
    state['selfdriveState'].experimentalMode = True
    state['modelV2'].action.desiredAcceleration = -2.5
  fail_next(planner, state)
  assert planner.reference_planner.continuation.offer is None
  assert not planner.motion_intent.valid
  if case == 'model_brake':
    assert planner.output_a_target <= -2.5


def test_unknown_or_new_lead_identity_cannot_spend_an_unobstructed_donor():
  planner, state = ready_planner()
  lead = state['radarState'].leadOne
  lead.present = True
  lead.dRel, lead.vLead, lead.vLeadK = 45.0, 20.0, 20.0
  lead.vRel, lead.aLeadK, lead.aLeadTau = 0.0, 0.0, 1.5
  lead.modelProb = 0.99
  fail_next(planner, state)
  assert planner.reference_planner.continuation.offer is None


def test_tighter_current_cap_rejects_continuation_and_strengthens_fallback():
  planner, state = ready_planner()
  reference = planner.reference_planner
  with patch.object(reference, 'independent_constraint', return_value=(-1.2, planner.mpc.source, False, False)):
    fail_next(planner, state)
  assert reference.continuation.offer is None
  assert reference.continuation.reason == 'current_independent_constraint_tighter'
  assert reference.continuation.independent_braking_accel == pytest.approx(-1.2)
  assert planner.output_a_target <= -1.2
  assert planner.motion_intent.braking_accel <= -1.2


def test_final_cap_is_part_of_the_dynamically_consistent_donor_reference():
  planner, state = ready_planner()
  reference = planner.reference_planner
  with patch.object(reference, 'independent_constraint', return_value=(-0.4, planner.mpc.source, False, False)):
    planner.update(state, now_ns=advance(state, 2))
  assert planner.mpc.solution_status == 0
  assert reference.reference_overridden
  donor = reference.continuation.donor
  assert donor is not None
  assert donor.trajectory.applied_accel_ceiling == pytest.approx(-0.4)
  assert np.all(donor.trajectory.states[1:, 2] <= -0.4 + 1e-12)
  assert np.max(np.abs(planner.mpc.trajectory_dynamics_defects(donor.trajectory.states, donor.trajectory.jerks))) < 1e-9


def test_publication_cannot_extend_an_accepted_candidate_deadline():
  planner, state = ready_planner()
  fail_next(planner, state)
  carry = planner.reference_planner.continuation
  assert carry.offer is not None, carry.reason
  until = carry.offer.valid_until_ns
  publisher = Publisher()
  with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=until + 1):
    planner.publish(state, publisher)
  plan = publisher.messages['longitudinalPlan'].longitudinalPlan
  assert not plan.continuation.valid
  assert plan.continuation.validUntilMonoTime == until
  assert plan.continuation.reason == 'expired_before_publication'
  assert plan.aTarget <= -0.5


def test_a_continuation_cannot_become_the_next_native_donor():
  planner, state = ready_planner()
  original = planner.reference_planner.continuation.donor
  fail_next(planner, state)
  carry = planner.reference_planner.continuation
  assert carry.offer is not None, carry.reason
  assert carry.donor is original
  motion = state['controlsState'].motionState
  motion.continuationActive = True
  motion.continuationDonorProducerMonoTime = original.producer_ns
  motion.continuationEpisodeMonoTime = carry.offer.episode_start_ns
  motion.continuationValidUntilMonoTime = carry.offer.valid_until_ns
  motion.referenceProducerMonoTime = 1_100_000_000
  motion.referenceConsumerMonoTime = 1_101_000_000
  fail_next(planner, state, frame=3)
  assert carry.offer is not None, carry.reason
  assert carry.donor is original
  assert carry.offer.episode_start_ns == 1_100_000_000
  assert carry.offer.valid_until_ns <= 1_200_000_000


@pytest.mark.parametrize('change', ['unchanged', 'slot_generation_source', 'follow', 'stop', 'body_only'])
def test_consecutive_failures_recheck_native_donor_purposes_after_mpc_history_is_cleared(change):
  planner = enabled_planner(20.0)
  state = make_state(speed=20.0, target=20.0, experimental=True)
  following_time = get_T_FOLLOW(state['selfdriveState'].personality,
                                v_ego=20.0 if planner.mpc.speed_based_following else None)
  # A balanced following state admits a numerical continuation witness. A
  # different distance can fail that certificate before this guard is tested.
  distance = float(get_safe_obstacle_distance(20.0, following_time) - get_stopped_equivalence_factor(20.0))

  def install(frame, update='unchanged'):
    now = advance(state, frame)
    state['modelV2'].timestampEof = now
    lead = formal_lead(now_ns=now, dRel=distance, vLead=20.0, vLeadK=20.0, vRel=0.0)
    if update in ('follow', 'stop'):
      setattr(lead.evidence, update + 'Qualified', False)
    elif update == 'body_only':
      for purpose in ('follow', 'stop', 'departure', 'fcw'):
        setattr(lead.evidence, purpose + 'Qualified', False)
      lead.evidence.followGeometryMonoTime = 0
    if update == 'slot_generation_source':
      lead.evidence.generation += 1
      lead.evidence.source = 'fused'
      lead.evidence.visionIndex = 0
      display = (formal_lead(present=False), lead)
    else:
      display = (lead,)
    state['radarState'] = native_frame(radar_frame(lead, publication=now, display=display))
    return now

  with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
    for frame in (0, 1):
      now = install(frame)
      planner.update(state, now_ns=now)
      assert planner.mpc.solution_status == 0 and planner.motion_intent.valid
      acknowledge_native(state, planner, now)
    carry = planner.reference_planner.continuation
    donor = carry.donor
    assert donor is not None
    assert donor.object_purposes == frozenset((('radar-session-a', 7), purpose) for purpose in ('brake', 'follow', 'stop'))
    install(2)
    now = fail_next(planner, state, frame=2)
    first = carry.offer
    assert first is not None, carry.reason
    assert carry.donor is donor and planner.mpc._reference_support is None
    assert not planner.reference_planner.reference_initialized

    motion = state['controlsState'].motionState
    motion.continuationActive = True
    motion.continuationDonorProducerMonoTime = donor.producer_ns
    motion.continuationEpisodeMonoTime = first.episode_start_ns
    motion.continuationValidUntilMonoTime = first.valid_until_ns
    motion.referenceProducerMonoTime = now
    motion.referenceConsumerMonoTime = now + 1_000_000
    install(3, change)
    with patch.object(planner.mpc, 'continuation_candidate', wraps=planner.mpc.continuation_candidate) as certificate:
      fail_next(planner, state, frame=3)
    assert planner.mpc.solution_status == 4 and not planner.motion_intent.valid
    assert not planner.mpc.reference_support_changed  # The prior failure removed this independent MPC guard.
    assert carry.budget.episode_start_ns == first.episode_start_ns
    assert carry.budget.deadline(donor_producer_ns=donor.producer_ns, donor_source_ns=donor.source_ns,
                                donor_model_ns=donor.model_ns) == first.valid_until_ns
    if change in ('follow', 'stop', 'body_only'):
      assert carry.offer is None and carry.donor is None
      assert carry.reason == 'object_qualification_reduced'
      certificate.assert_not_called()
    else:
      assert carry.offer is not None, carry.reason
      assert carry.donor is donor and carry.offer.donor_source_ns == donor.source_ns
      assert carry.offer.valid_until_ns == first.valid_until_ns
      certificate.assert_called_once()


def test_healthy_native_stop_clears_episode_without_becoming_a_moving_donor():
  planner, state = ready_planner()
  fail_next(planner, state)
  carry = planner.reference_planner.continuation
  assert carry.budget.episode_start_ns != 0
  state = make_state(speed=0.0, target=0.0)
  for frame in range(3, 17):
    now = advance(state, frame)
    planner.update(state, now_ns=now)
    assert planner.mpc.solution_status == 0 and planner.motion_intent.valid
    acknowledge_native(state, planner, now)
    state['controlsState'].motionState.phase = 'hold'
    state['controlsState'].motionState.resolvedStop = True
  assert carry.budget.episode_start_ns == 0
  assert carry.donor is None
