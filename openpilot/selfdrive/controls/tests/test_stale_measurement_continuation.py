"""Expired physical observations retain identity without becoming fresh input."""
from contextlib import contextmanager
from copy import deepcopy
from unittest.mock import Mock, patch

import pytest

from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.lead_qualification import MAX_MEASUREMENT_AGE_NS, lead_identity, qualified_lead
from openpilot.selfdrive.controls.lib.longitudinal_continuation import CONTINUATION_EPISODE_NS, candidate_state
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.continuation import STALE_MEASUREMENT_STATUS
from openpilot.selfdrive.controls.lib.planner_continuation import PlannerContinuation
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import selfdrive_event_harness
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_continuation_control_chain import ContinuationChain, START
from openpilot.selfdrive.controls.tests.test_lead_qualification import formal_lead
from openpilot.selfdrive.controls.tests.test_radar_consumption import native_frame, radar_frame
from openpilot.selfdrive.selfdrived.events import ET
from openpilot.selfdrive.selfdrived.state import StateMachine
from openpilot.selfdrive.selfdrived.tests.test_continuation_events import observe_at


MEASURED = START + 100_000_000 - MAX_MEASUREMENT_AGE_NS
FIRST_STALE = START + 100_000_000


class StaleContinuationChain(ContinuationChain):
  """Supply a physically consistent measured history during the short offer.

  This is an ideal tracking fixture, not a vehicle plant or delivered-effect
  claim. The separate control-chain tests cover rejecting mismatched history.
  """
  def _advance_measured_state(self, timestamp):
    carry = self.planner.reference_planner.continuation
    reference = self.planner.reference_timing.pending
    if carry.offer is not None and reference is not None:
      speed, accel = candidate_state(carry.offer.candidate, (timestamp - reference.epoch_ns) / 1e9)
      self.state['carState'].vEgo, self.state['carState'].aEgo = speed, accel

  def produce(self, source_ns, status=0):
    self._advance_measured_state(source_ns)
    return super().produce(source_ns, status)

  def consume(self, sample_ns, **kwargs):
    self._advance_measured_state(sample_ns)
    return super().consume(sample_ns, **kwargs)


def physical_lead(stamp=START + 50_000_000, *, measured_ns=MEASURED, source='radar', **changes):
  return formal_lead(now_ns=stamp, source=source, dRel=100.0, vLead=20.0, vLeadK=20.0, vRel=0.0,
                     evidence={'positionMonoTime': measured_ns, 'velocityMonoTime': measured_ns, **changes})


@pytest.mark.parametrize('replacement', ['missing', 'other_track', 'vision', 'generation', 'regressed_position', 'invalid'])
def test_an_expired_physical_target_cannot_be_cleared_by_absence_or_unrelated_evidence(replacement):
  carry = PlannerContinuation()
  old = physical_lead()
  frame = radar_frame(old, publication=START + 50_000_000, display=(old,))
  now = FIRST_STALE + 1
  original = deepcopy(vars(old.evidence))
  assert not qualified_lead(old, 'brake', 0.0, now_ns=now)
  assert carry.retain_measurement_loss(frame, now, active=True) == ''
  assert lead_identity(old) in carry.retained_measurements
  assert vars(old.evidence) == original

  fresh = physical_lead(stamp=now + 50_000_000, measured_ns=now + 50_000_000)
  if replacement == 'missing':
    slots = ()
  else:
    if replacement == 'other_track':
      fresh.evidence.trackId += 1
    elif replacement == 'vision':
      fresh = formal_lead(now_ns=now + 50_000_000, source='vision', dRel=110.0, vLead=20.0, vLeadK=20.0)
    elif replacement == 'generation':
      fresh.evidence.generation += 1
    elif replacement == 'regressed_position':
      fresh.evidence.positionMonoTime = MEASURED - 1
    else:
      fresh.evidence.positionValid = False
    slots = (fresh,)
  replacement_frame = radar_frame(*slots, publication=now + 50_000_000, display=slots)
  assert carry.retain_measurement_loss(replacement_frame, now + 50_000_000, active=True)
  assert lead_identity(old) in carry.retained_measurements


@pytest.mark.parametrize('recovery', ['fresh_same_object', 'inactive'])
def test_new_same_object_measurements_or_control_exit_release_the_retained_loss(recovery):
  carry = PlannerContinuation()
  old = physical_lead()
  now = FIRST_STALE + 1
  frame = radar_frame(old, publication=START + 50_000_000, display=(old,))
  assert carry.retain_measurement_loss(frame, now, active=True) == ''
  assert carry.retained_measurements
  if recovery == 'fresh_same_object':
    fresh = physical_lead(stamp=now + 50_000_000, measured_ns=now + 50_000_000)
    frame = radar_frame(fresh, publication=now + 50_000_000, display=(fresh,))
  assert carry.retain_measurement_loss(frame, now + 50_000_000, active=recovery != 'inactive') == ''
  assert not carry.retained_measurements


def test_unhealthy_fresh_publication_cannot_clear_a_retained_target():
  carry = PlannerContinuation()
  old = physical_lead()
  frame = radar_frame(old, display=(old,))
  assert carry.retain_measurement_loss(frame, FIRST_STALE + 1, active=True) == ''
  fresh = physical_lead(FIRST_STALE + 50_000_000, measured_ns=FIRST_STALE + 50_000_000)
  frame = radar_frame(fresh, display=(fresh,))
  assert carry.retain_measurement_loss(frame, FIRST_STALE + 50_000_000, active=True, healthy=False) == 'retained_stale_radar_unhealthy'
  assert carry.retained_measurements


@pytest.mark.parametrize('clearance', ['measured_out', 'other_session', 'stale_publication', 'older_than_occupancy', 'unhealthy'])
def test_only_a_new_healthy_same_session_measured_out_receipt_clears_known_loss(clearance):
  carry = PlannerContinuation()
  old = physical_lead()
  frame = radar_frame(old, display=(old,))
  publication = START + 50_000_000
  assert carry.retain_measurement_loss(frame, publication, active=True, speed=20.0, publication_ns=publication) == ''
  assert carry.known_measurements
  now = FIRST_STALE + 50_000_000
  cleared_ns = now
  if clearance == 'stale_publication':
    cleared_ns = now - 250_000_000
  elif clearance == 'older_than_occupancy':
    cleared_ns = publication - 1
  session = 'another-session' if clearance == 'other_session' else old.evidence.sessionId
  frame = radar_frame(publication=cleared_ns, cleared=(old.evidence.trackId,), session=session)
  reason = carry.retain_measurement_loss(frame, now, active=True, healthy=clearance != 'unhealthy', speed=20.0,
                                         publication_ns=cleared_ns)
  if clearance == 'measured_out':
    assert reason == '' and not carry.known_measurements and not carry.retained_measurements
  else:
    assert carry.known_measurements


@contextmanager
def established_stale_chain(*, source='radar', braking_limit=-.4):
  """A real native donor and owner under a fixed, modest braking ceiling.

  Only DEC's selected mode and the independent ceiling are fixture inputs.
  Prediction, MPC qualification, transport, owner admission, CAN generation
  and the event observer execute their production code.
  """
  for name in ('AlphaLongitudinalEnabled', 'ExperimentalMode', 'DynamicExperimentalControl'):
    Params().put_bool(name, True, block=True)
  chain = StaleContinuationChain()
  chain.state['selfdriveState'].experimentalMode = True
  chain.state['carState'].vCruise = chain.state['carState'].vCruiseCluster = 72.0
  chain.state['modelV2'].action.desiredAcceleration = 0.0
  reference = chain.planner.reference_planner
  independent = reference.independent_constraint

  def braking_ceiling(*args, **kwargs):
    accel, selected_source, override, warning = independent(*args, **kwargs)
    return min(accel, braking_limit), selected_source, override, warning

  daemon = None
  with patch.object(chain.planner.dec, 'enabled', return_value=True), \
       patch.object(chain.planner.dec, 'mode', return_value='acc'), \
       patch.object(reference, 'independent_constraint', side_effect=braking_ceiling):
    for offset in (0, 50_000_000):
      stamp = START + offset
      lead = physical_lead(stamp, source=source)
      chain.state['radarState'] = native_frame(radar_frame(lead, publication=stamp, display=(lead,)))
      event = chain.produce(stamp)
      assert event.valid and event.longitudinalPlan.solverStatus == 0
      for tick in range(5):
        sample = stamp + tick * 10_000_000
        _, motion, _ = chain.consume(sample)
        assert motion.phase == 'track' and not motion.degraded and not motion.continuationActive
        if daemon is None:
          daemon, _ = selfdrive_event_harness(chain.controls.control.pm.messages['controlsState'])
          daemon.CP, daemon.CP_SP = chain.controls.control.CP, chain.controls.control.CP_SP
          daemon.big_model_ready_t = -10.0
          daemon.state_machine = StateMachine()
          daemon.state_machine.state = log.SelfdriveState.OpenpilotState.enabled
        observe_at(daemon, chain, sample)
        assert not daemon.events.names
    assert reference.continuation.donor is not None, chain.planner.mpc.continuation_donor_reason
    assert chain.controls.control.LoC.continuation_donor is not None
    yield chain, daemon


def test_nonsteady_donor_tail_keeps_the_original_failure_protection():
  with established_stale_chain(braking_limit=-.05) as (chain, daemon):
    failed = chain.produce(FIRST_STALE)
    carry = chain.planner.reference_planner.continuation
    assert not failed.longitudinalPlan.continuation.valid
    assert carry.reason == 'conditioned_donor_terminal_jerk'
    command, motion, _ = chain.consume(FIRST_STALE)
    observe_at(daemon, chain, FIRST_STALE)
    assert motion.degraded and command.actuators.accel <= -.5
    assert log.OnroadEvent.EventName.longitudinalControlUnavailable in daemon.events.names


@pytest.mark.parametrize('source', ['radar', 'fused'])
def test_stale_measurement_keeps_original_status_and_braking_without_a_takeover_alert(source):
  with established_stale_chain(source=source) as (chain, daemon):
    raw = chain.state['radarState']
    original = raw.to_dict()
    previous_accel = chain.controls.command.actuators.accel
    solver = Mock(wraps=chain.planner.mpc.solver)
    with patch.object(chain.planner.mpc, 'solver', solver):
      event = chain.produce(FIRST_STALE)
      solver.solve.assert_not_called()
    plan = event.longitudinalPlan
    carry = chain.planner.reference_planner.continuation
    assert not event.valid and plan.solverStatus == STALE_MEASUREMENT_STATUS
    assert plan.continuation.valid, carry.reason
    assert not plan.motionIntent.valid and not plan.motionIntent.releaseAllowed
    assert all(accel <= 0.0 for accel in plan.continuation.candidate.accels[1:])
    assert not qualified_lead(raw.leadOne, 'brake', 0.0, now_ns=FIRST_STALE + 500_000)
    assert raw.to_dict() == original
    observe_at(daemon, chain, FIRST_STALE, offset_ns=2_000_000)
    assert not daemon.events.names
    for tick in range(5):
      sample = FIRST_STALE + tick * 10_000_000
      command, motion, can = chain.consume(sample)
      assert motion.continuationActive, chain.controls.control.LoC.continuation_reason
      assert not motion.degraded and command.longActive and not command.cruiseControl.resume
      assert command.actuators.accel <= previous_accel + 1e-7
      assert can['ACCEL_COMMAND'] <= 0.0
      observe_at(daemon, chain, sample, offset_ns=6_000_000)
      assert not daemon.events.names
      previous_accel = command.actuators.accel
    assert carry.retained_measurements


def test_repeated_stale_publications_cannot_renew_the_deadline_or_hide_expiry():
  with established_stale_chain() as (chain, daemon):
    first = chain.produce(FIRST_STALE)
    assert first.longitudinalPlan.continuation.valid, chain.planner.reference_planner.continuation.reason
    deadline = first.longitudinalPlan.continuation.validUntilMonoTime
    assert deadline <= FIRST_STALE + 500_000 + CONTINUATION_EPISODE_NS
    for tick in range(5):
      sample = FIRST_STALE + tick * 10_000_000
      chain.consume(sample)
      observe_at(daemon, chain, sample)
    second = chain.produce(FIRST_STALE + 50_000_000)
    assert second.longitudinalPlan.continuation.valid, chain.planner.reference_planner.continuation.reason
    assert second.longitudinalPlan.continuation.validUntilMonoTime == deadline
    chain.consume(FIRST_STALE + 50_000_000)
    observe_at(daemon, chain, FIRST_STALE + 50_000_000)
    assert not daemon.events.names
    for offset in (60, 70, 80, 90):
      sample = FIRST_STALE + offset * 1_000_000
      _, motion, _ = chain.consume(sample)
      observe_at(daemon, chain, sample)
      assert motion.continuationActive and not daemon.events.names
    command, motion, can = chain.consume(deadline)
    observe_at(daemon, chain, deadline)
    assert not motion.continuationActive and motion.degraded
    assert command.actuators.accel <= -.5 and can['BRAKE_REQUEST']
    assert log.OnroadEvent.EventName.longitudinalControlUnavailable in daemon.events.names
    assert daemon.events.contains(ET.SOFT_DISABLE)


def test_removing_a_retained_target_does_not_produce_an_unobstructed_success():
  with established_stale_chain() as (chain, daemon):
    first = chain.produce(FIRST_STALE)
    assert first.longitudinalPlan.continuation.valid, chain.planner.reference_planner.continuation.reason
    for tick in range(5):
      chain.consume(FIRST_STALE + tick * 10_000_000)
    missing_ns = FIRST_STALE + 50_000_000
    chain.state['radarState'] = native_frame(radar_frame(publication=missing_ns))
    failed = chain.produce(missing_ns)
    assert not failed.valid and failed.longitudinalPlan.solverStatus != 0
    assert not failed.longitudinalPlan.continuation.valid
    command, motion, can = chain.consume(missing_ns)
    observe_at(daemon, chain, missing_ns)
    assert motion.degraded and not motion.continuationActive
    assert command.actuators.accel <= -.5 and can['BRAKE_REQUEST']
    assert log.OnroadEvent.EventName.longitudinalControlUnavailable in daemon.events.names


def test_first_expired_publication_may_withdraw_the_target_without_erasing_its_loss():
  with established_stale_chain() as (chain, daemon):
    carry = chain.planner.reference_planner.continuation
    assert carry.known_measurements and not carry.retained_measurements
    chain.state['radarState'] = native_frame(radar_frame(publication=FIRST_STALE))
    solver = Mock(wraps=chain.planner.mpc.solver)
    with patch.object(chain.planner.mpc, 'solver', solver):
      failed = chain.produce(FIRST_STALE)
      solver.solve.assert_not_called()
    assert carry.reason == 'retained_stale_target_missing'
    assert carry.retained_measurements
    assert not failed.valid and not failed.longitudinalPlan.continuation.valid
    assert chain.planner.mpc._diagnostic_now_ns == FIRST_STALE + 500_000
    command, motion, can = chain.consume(FIRST_STALE)
    observe_at(daemon, chain, FIRST_STALE)
    assert motion.degraded and not motion.continuationActive
    assert command.actuators.accel <= -.5 and can['BRAKE_REQUEST']
    assert log.OnroadEvent.EventName.longitudinalControlUnavailable in daemon.events.names


def test_fresh_measured_out_keeps_normal_planning_after_the_old_measurement_deadline():
  with established_stale_chain() as (chain, daemon):
    previous = chain.state['radarState'].leadOne.evidence
    for offset in (0, 50_000_000):
      stamp = FIRST_STALE + offset
      cleared = (previous.trackId,) if offset == 0 else ()
      chain.state['radarState'] = native_frame(radar_frame(publication=stamp, cleared=cleared, session=previous.sessionId))
      event = chain.produce(stamp)
      assert event.valid and event.longitudinalPlan.solverStatus == 0
      assert not chain.planner.reference_planner.continuation.retained_measurements
      assert not chain.planner.reference_planner.continuation.known_measurements
      for tick in range(5):
        sample = stamp + tick * 10_000_000
        _, motion, _ = chain.consume(sample)
        observe_at(daemon, chain, sample)
        assert not motion.degraded and not daemon.events.names


def test_a_new_measurement_of_the_retained_object_restores_normal_planning():
  with established_stale_chain() as (chain, daemon):
    first = chain.produce(FIRST_STALE)
    assert first.longitudinalPlan.continuation.valid, chain.planner.reference_planner.continuation.reason
    for tick in range(5):
      chain.consume(FIRST_STALE + tick * 10_000_000)
    fresh_ns = FIRST_STALE + 50_000_000
    fresh = physical_lead(fresh_ns, measured_ns=fresh_ns)
    chain.state['radarState'] = native_frame(radar_frame(fresh, publication=fresh_ns, display=(fresh,)))
    recovered = chain.produce(fresh_ns)
    assert recovered.valid and recovered.longitudinalPlan.solverStatus == 0
    assert not recovered.longitudinalPlan.continuation.valid
    assert not chain.planner.reference_planner.continuation.retained_measurements
    _, motion, _ = chain.consume(fresh_ns)
    observe_at(daemon, chain, fresh_ns)
    assert not motion.degraded and not motion.continuationActive
    assert not daemon.events.names
