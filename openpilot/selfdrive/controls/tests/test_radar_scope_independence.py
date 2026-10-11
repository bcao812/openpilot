"""DEC scope is a fresh input receipt, independent of a successful MPC solve."""
from dataclasses import replace
from unittest.mock import Mock

import pytest

from openpilot.cereal import messaging
from openpilot.selfdrive.car.honda_radar_observer_config import SCOPE_MAX_AGE_NS
from openpilot.selfdrive.car.tests.honda_radar_control_test_support import START_NS, Target, _publish_at, control_pipeline
from openpilot.selfdrive.controls.lib.motion_supervisor import StopReason
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import bounded_submaster, serialized


SCOPE_INPUTS = ('carState', 'controlsState', 'selfdriveState')


def publish_scope(planner, state, now):
  return _publish_at(lambda pm: planner.publish_longitudinal_plan_sp(state, pm), now)['longitudinalPlanSP']


def accept_scope(state, event, now=None):
  state.update_msgs((event.logMonoTime if now is None else now) / 1e9, [event])


@pytest.fixture
def scope_case():
  with control_pipeline(speed=8.0) as pipeline:
    pipeline.step(speed=8.0)
    now = START_NS + 100_000_000
    state = bounded_submaster([*SCOPE_INPUTS, 'longitudinalPlanSP'])
    events = []
    for service in SCOPE_INPUTS:
      event = messaging.new_message(service, logMonoTime=now - 1_000_000, valid=True)
      setattr(event, service, pipeline.state[service])
      events.append(serialized(event))
    state.update_msgs(now / 1e9, events)
    yield pipeline, state, now


def test_success_failure_success_keeps_scope_while_failed_plan_stays_invalid(scope_case):
  pipeline, state, now = scope_case
  planner, radar = pipeline.planner, pipeline.radard
  original = planner.motion_intent
  assert original.valid and not planner.reference_planner.standby
  for offset, valid in enumerate((True, False, True)):
    planner.motion_intent = replace(original, valid=valid)
    event = publish_scope(planner, state, now + offset)
    assert event.valid is valid
    assert event.longitudinalPlanSP.dec.scopeVersion == 1
    assert event.longitudinalPlanSP.dec.scopeValid
    assert event.longitudinalPlanSP.dec.scopeSourceMonoTime == now - 1_000_000
    accept_scope(state, event)
    assert radar.control_enabled(state, event.logMonoTime)
    assert not radar.control_scope_unavailable


@pytest.mark.parametrize('service', SCOPE_INPUTS)
@pytest.mark.parametrize('fault', ['invalid', 'dead', 'bad_frequency', 'unseen', 'zero', 'future', 'stale'])
def test_publisher_rejects_unhealthy_or_unidentified_scope_inputs(scope_case, service, fault):
  pipeline, state, now = scope_case
  if fault == 'invalid':
    state.valid[service] = False
  elif fault == 'dead':
    state.alive[service] = False
  elif fault == 'bad_frequency':
    state.ignore_average_freq = []
    state.freq_ok = dict.fromkeys(state.services, True)
    state.freq_ok[service] = False
  elif fault == 'unseen':
    state.seen[service] = False
  else:
    state.logMonoTime[service] = {'zero': 0, 'future': now + 1, 'stale': now - SCOPE_MAX_AGE_NS - 1}[fault]
  event = publish_scope(pipeline.planner, state, now)
  assert not event.longitudinalPlanSP.dec.scopeValid
  if service == 'selfdriveState':
    # This proves the independent receipt closes a gap that Event.valid's
    # original carState/controlsState-only transport check does not cover.
    assert event.valid
  accept_scope(state, event)
  assert not pipeline.radard.control_enabled(state, now)
  assert pipeline.radard.control_scope_unavailable


@pytest.mark.parametrize('service', SCOPE_INPUTS)
def test_scope_source_expires_at_its_own_boundary_even_with_a_new_plan(scope_case, service):
  pipeline, state, now = scope_case
  state.logMonoTime[service] = now - SCOPE_MAX_AGE_NS
  event = publish_scope(pipeline.planner, state, now)
  assert event.longitudinalPlanSP.dec.scopeValid
  assert event.longitudinalPlanSP.dec.scopeSourceMonoTime == now - SCOPE_MAX_AGE_NS
  accept_scope(state, event)
  assert pipeline.radard.control_enabled(state, now)
  assert not pipeline.radard.control_enabled(state, now + 1)
  assert pipeline.radard.control_scope_unavailable
  expired = publish_scope(pipeline.planner, state, now + 1)
  assert not expired.longitudinalPlanSP.dec.scopeValid
  assert expired.longitudinalPlanSP.dec.scopeSourceMonoTime == event.longitudinalPlanSP.dec.scopeSourceMonoTime


@pytest.mark.parametrize('fault', [
  'sp_unseen', 'sp_zero', 'sp_stale', 'sp_future', 'scope_invalid', 'scope_source_zero',
  'scope_source_stale', 'scope_source_after_publication', 'unknown_version',
  'selfdrive_invalid', 'selfdrive_unseen', 'selfdrive_zero', 'selfdrive_stale', 'selfdrive_future',
])
def test_consumer_rejects_missing_or_malformed_scope_without_clearing_history(scope_case, fault):
  pipeline, state, now = scope_case
  event = publish_scope(pipeline.planner, state, now).as_builder()
  if fault == 'scope_invalid':
    event.longitudinalPlanSP.dec.scopeValid = False
  elif fault == 'scope_source_zero':
    event.longitudinalPlanSP.dec.scopeSourceMonoTime = 0
  elif fault == 'scope_source_stale':
    event.longitudinalPlanSP.dec.scopeSourceMonoTime = now - SCOPE_MAX_AGE_NS - 1
  elif fault == 'scope_source_after_publication':
    event.longitudinalPlanSP.dec.scopeSourceMonoTime = now + 1
  elif fault == 'unknown_version':
    event.longitudinalPlanSP.dec.scopeVersion = 2
  accept_scope(state, serialized(event))
  if fault.startswith(('sp_', 'selfdrive_')):
    prefix, defect = fault.split('_', 1)
    service = 'longitudinalPlanSP' if prefix == 'sp' else 'selfdriveState'
    if defect == 'unseen':
      state.seen[service] = False
    elif defect == 'invalid':
      state.valid[service] = False
    else:
      state.logMonoTime[service] = {'zero': 0, 'stale': now - SCOPE_MAX_AGE_NS - 1, 'future': now + 1}[defect]
  # The scope method classifies unavailable feedback; it must not turn the
  # failure into an explicit configuration exit that clears adapter history.
  assert not pipeline.radard.control_enabled(state, now)
  assert pipeline.radard.control_scope_unavailable


def test_future_source_cannot_become_valid_just_because_consumer_time_caught_up(scope_case):
  pipeline, state, now = scope_case
  event = publish_scope(pipeline.planner, state, now).as_builder()
  event.longitudinalPlanSP.dec.scopeSourceMonoTime = now + 1
  accept_scope(state, serialized(event), now + 2)
  assert not pipeline.radard.control_enabled(state, now + 2)
  assert pipeline.radard.control_scope_unavailable


@pytest.mark.parametrize('event_valid', [False, True])
def test_legacy_scope_keeps_its_original_event_valid_contract(scope_case, event_valid):
  pipeline, state, now = scope_case
  # A wire message written by an older producer has the new fields' defaults.
  event = messaging.new_message('longitudinalPlanSP', valid=event_valid, logMonoTime=now)
  event.longitudinalPlanSP.dec.enabled = event.longitudinalPlanSP.dec.active = True
  accept_scope(state, serialized(event))
  assert pipeline.radard.control_enabled(state, now) is event_valid
  assert pipeline.radard.control_scope_unavailable is not event_valid


@pytest.mark.parametrize('exit_kind', ['dec_disabled', 'dec_inactive', 'experimental_off', 'configuration_exit'])
@pytest.mark.parametrize('plan_valid', [False, True])
def test_healthy_explicit_scope_exit_keeps_its_lifecycle_meaning(scope_case, exit_kind, plan_valid):
  pipeline, state, now = scope_case
  pipeline.planner.motion_intent = replace(pipeline.planner.motion_intent, valid=plan_valid)
  if exit_kind == 'dec_disabled':
    pipeline.planner.dec._enabled = False
  elif exit_kind == 'dec_inactive':
    pipeline.planner.dec._active = False
  elif exit_kind == 'experimental_off':
    now += 1_000_000  # A new receive tick; the other scope inputs remain fresh.
    event = messaging.new_message('selfdriveState', logMonoTime=now, valid=True)
    event.selfdriveState = state['selfdriveState']
    event.selfdriveState.experimentalMode = False
    state.update_msgs(now / 1e9, [serialized(event)])
  else:
    pipeline.params.put_bool('ExperimentalMode', False, block=True)
    pipeline.radard.settings_next_ns = 0
  event = publish_scope(pipeline.planner, state, now)
  assert event.valid is plan_valid and event.longitudinalPlanSP.dec.scopeValid
  accept_scope(state, event)
  assert not pipeline.radard.control_enabled(state, now)
  assert not pipeline.radard.control_scope_unavailable


@pytest.mark.parametrize('rejection', ['owner_contract', 'model_shape', 'future_epoch'])
def test_scope_health_does_not_claim_dec_or_mpc_recomputation_after_early_return(rejection):
  with control_pipeline(speed=8.0) as pipeline:
    pipeline.step(speed=8.0)
    state, planner = pipeline.state, pipeline.planner
    now = START_NS + 50_000_000
    for service in SCOPE_INPUTS:
      state.logMonoTime[service] = now
    state.logMonoTime['modelV2'] = now
    dec_frame = planner.dec._frame
    if rejection == 'owner_contract':
      state['controlsState'].motionState.version = 0
    elif rejection == 'model_shape':
      model = state['modelV2'].as_reader().as_builder()
      model.velocity.x = []
      state['modelV2'] = model.as_reader()
    else:
      state.logMonoTime['carState'] = now + 10_000_000
    planner.update(state, now_ns=now + 500_000)
    assert planner.dec._frame == dec_frame and not planner.motion_intent.valid
    event = publish_scope(planner, state, now + 1_000_000)
    assert not event.valid
    assert event.longitudinalPlanSP.dec.scopeValid is (rejection != 'future_epoch')
    pipeline._accept('longitudinalPlanSP', event)
    assert pipeline.radard.control_enabled(state, event.logMonoTime) is (rejection != 'future_epoch')
    # Even while the planner returns before DEC.update, radard's independent
    # configuration read still revokes scope, without making the plan valid.
    pipeline.params.put_bool('DynamicExperimentalControl', False, block=True)
    pipeline.radard.settings_next_ns = 0
    assert not pipeline.radard.control_enabled(state, event.logMonoTime)
    assert not pipeline.radard.control_scope_unavailable


def test_mpc_failure_does_not_withdraw_formal_radar_leads_from_the_next_tick():
  def formal_evidence(row):
    # Only the two published leads carry current control authority. The old
    # radarObjects list is deliberately empty, including during recovery.
    assert row['radar'].get('radarObjects', []) == []
    leads = [row['radar'][name] for name in ('leadOne', 'leadTwo') if row['radar'][name]['present']]
    assert leads
    evidence_by_identity = {}
    for lead in leads:
      evidence = lead['evidence']
      assert evidence['version'] == 1 and evidence['source'] == 'radar'
      assert evidence['brakeQualified'] and not evidence['coasting']
      assert evidence['sessionId'] == row['radar']['fusionSessionId'] and evidence['sessionId']
      assert evidence['trackId'] > 0
      identity = evidence['sessionId'], evidence['trackId']
      assert identity not in evidence_by_identity
      evidence_by_identity[identity] = evidence
    return evidence_by_identity

  with control_pipeline(speed=8.0) as pipeline:
    target = Target(30.0, 8.0)
    for _ in range(24):
      before = pipeline.step(speed=8.0, radar=[target])
    identities = set(formal_evidence(before))
    assert identities and before['solver'] == 0
    adapter = pipeline.radard.fusion_control
    solver = pipeline.planner.mpc.solver
    pipeline.planner.mpc.solver = Mock(wraps=solver)
    pipeline.planner.mpc.solver.solve.return_value = 4
    failed_rows = []
    try:
      # Inject the native return code, not a synthetic SP.valid mutation.
      # The real planner, both publishers, owner and Honda CAN run normally.
      for _ in range(3):
        failed_rows.append(pipeline.step(speed=8.0, radar=[target], expected_solver_status=4))
        assert not pipeline.state.valid['longitudinalPlanSP']
        assert pipeline.state['longitudinalPlanSP'].dec.scopeValid
        assert pipeline.state['longitudinalPlanSP'].dec.scopeVersion == 1
        assert pipeline.owner.motion_degraded and pipeline.owner.motion_reasons & StopReason.INVALID
    finally:
      pipeline.planner.mpc.solver = solver
    recovered = pipeline.step(speed=8.0, radar=[target])
    after = pipeline.step(speed=8.0, radar=[target])
    previous_position = before['observer_position_ns']
    for row in (*failed_rows, recovered, after):
      assert row['radar']['fusionActive'] and row['radar']['fusionReason'] != 'controlScopeUnavailable'
      evidence_by_identity = formal_evidence(row)
      assert set(evidence_by_identity) == identities
      for evidence in evidence_by_identity.values():
        assert evidence['positionMonoTime'] == row['observer_position_ns'] > previous_position
        assert evidence['velocityMonoTime'] == row['source_ns']
        assert evidence['modelMonoTime'] == row['source_ns']
      previous_position = row['observer_position_ns']
    assert pipeline.radard.fusion_control is adapter
    assert pipeline.state.valid['longitudinalPlanSP'] and not pipeline.radard.control_scope_unavailable
