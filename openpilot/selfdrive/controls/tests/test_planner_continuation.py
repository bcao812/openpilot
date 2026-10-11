from unittest.mock import Mock, patch

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.longitudinal_continuation import CONTINUATION_VERSION
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_TIMING_VERSION
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_planner, make_state, advance, Publisher
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params


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


@pytest.mark.parametrize('status', [-1, -6])
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
