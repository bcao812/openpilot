"""Measurement expiry has one risk policy on either side of publication."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.lead_qualification import (
  MAX_MEASUREMENT_AGE_NS, MAX_MODEL_AGE_NS, _qualified_lead_evidence, control_leads,
  measurement_expired_since, qualified_lead,
)
from openpilot.selfdrive.controls.lib.radar_consumption import RadarConsumption, RadarConsumptionView
from openpilot.selfdrive.controls.tests.test_lead_qualification import caution_lead, formal_lead


PUBLICATION = 10_000_000_000
CONSUMPTION = PUBLICATION + 60_000_000
SESSION = 'radar-session-a'
# The shared fixture expires 50 ms after publication, before default consumption.
MEASUREMENT_AGE_AT_PUBLICATION_NS = MAX_MEASUREMENT_AGE_NS - 50_000_000


def expiring_lead(*, publication=PUBLICATION, relevant=False, track=7):
  return formal_lead(now_ns=publication, source='radar', dRel=6.6 if relevant else 100.0, vLead=0.0, vLeadK=0.0, vRel=-10.0,
                     evidence={'trackId': track, 'positionMonoTime': publication - MEASUREMENT_AGE_AT_PUBLICATION_NS,
                               'velocityMonoTime': publication - MEASUREMENT_AGE_AT_PUBLICATION_NS, 'generation': 7,
                               'followQualified': False, 'stopQualified': False, 'departureQualified': False,
                               'fcwQualified': False, 'accelerationValid': False,
                               'pathState': 'inPath', 'conflictStart': 0.0, 'conflictEnd': 10.0, 'lateralSpeed': 0.0})


def future_entry(*, publication=PUBLICATION, source='radar', track=7, path_state='entering', start=.5):
  return formal_lead(now_ns=publication, source=source,
                     evidence={'trackId': track, 'pathState': path_state, 'conflictStart': start, 'conflictEnd': 2.0})


def radar_frame(*objects, publication=PUBLICATION, display=(), degraded=False, risk_unresolved=False, cleared=(), session=SESSION):
  empty = SimpleNamespace(present=False)
  slots = (*display, empty, empty)
  return SimpleNamespace(fusionActive=True, fusionHealthy=True, fusionDegraded=degraded, fusionReason='active',
                         fusionRiskUnresolved=risk_unresolved,
                         fusionSessionId=session, fusionClearedTracks=cleared, mdMonoTime=publication,
                         radarObjects=list(objects), leadOne=slots[0], leadTwo=slots[1])


def consume_frame(consumer, frame, *, publication=PUBLICATION, now=CONSUMPTION, speed=10.0):
  return consumer.update(frame, publication_ns=publication, now_ns=now, speed=speed)


def test_measurement_deadline_is_inclusive_and_removal_does_not_refresh_a_stamp():
  expiring = expiring_lead()
  raw = radar_frame(expiring, display=(expiring,))
  consumer = RadarConsumption()
  boundary = expiring.evidence.positionMonoTime + MAX_MEASUREMENT_AGE_NS
  before = deepcopy(vars(expiring.evidence))
  assert qualified_lead(expiring, 'brake', now_ns=boundary)
  assert consume_frame(consumer, raw, now=boundary) is raw
  view = consume_frame(consumer, raw, now=boundary + 1)
  assert not qualified_lead(expiring, 'brake', now_ns=boundary + 1)
  assert view.radarObjects == () and not view.leadOne.present
  assert not view.leadOne.as_reader().present
  assert control_leads(view) == ()
  assert raw.radarObjects == [expiring] and raw.leadOne.present
  assert vars(expiring.evidence) == before
  assert not consumer.degraded
  assert raw.fusionClearedTracks == view.fusionClearedTracks == ()


def measured(*, source='radar', caution=False, publication=PUBLICATION, relevant=False, track=7):
  factory = caution_lead if caution else formal_lead
  changes = {'source': source} if not caution else {}
  result = factory(now_ns=publication, dRel=6.6 if relevant else 100.0, vLead=0.0, vLeadK=0.0, vRel=-10.0,
                   evidence={'trackId': track, 'positionMonoTime': publication - MEASUREMENT_AGE_AT_PUBLICATION_NS,
                             'velocityMonoTime': publication - MEASUREMENT_AGE_AT_PUBLICATION_NS}, **changes)
  result.evidence.source = source
  return result


@pytest.mark.parametrize('source,caution', [('radar', False), ('fused', False)])
@pytest.mark.parametrize('oldest', ['positionMonoTime', 'velocityMonoTime'])
def test_measured_deadline_retires_every_view_and_preserves_other_objects(source, caution, oldest):
  old = measured(source=source, caution=caution)
  other_stamp = 'velocityMonoTime' if oldest == 'positionMonoTime' else 'positionMonoTime'
  setattr(old.evidence, other_stamp, PUBLICATION)
  fresh = formal_lead(now_ns=PUBLICATION, evidence={'trackId': 9})
  native = native_frame(radar_frame(old, fresh, display=(old, fresh)))
  before, serialized = native.to_dict(), native.to_bytes()
  native.clear_write_flag()
  consumer = RadarConsumption()
  deadline = getattr(old.evidence, oldest) + MAX_MEASUREMENT_AGE_NS
  purpose = 'caution' if caution else 'brake'
  at = consume_frame(consumer, native, now=deadline)
  assert len(control_leads(at)) == 2 and qualified_lead(at.leadOne, purpose, 0.0, now_ns=deadline)
  after = consume_frame(consumer, native, now=deadline + 1)
  assert [lead.evidence.trackId for lead in control_leads(after)] == [9]
  assert not after.leadOne.present and after.leadTwo.present
  assert consumer.expired_identities == ((SESSION, 7),) and not consumer.degraded
  assert native.to_dict() == before and native.to_bytes() == serialized
  assert tuple(after.fusionClearedTracks) == tuple(native.fusionClearedTracks)


@pytest.mark.parametrize('source,caution', [('radar', False), ('fused', False)])
@pytest.mark.parametrize('future_entry_input', [False, True])
@pytest.mark.parametrize('defect', [
  'already_expired', 'future_measurement', 'future_model', 'expired_model', 'expired_estimate',
  'invalid_position', 'identity_missing', 'duplicate_geometry', 'duplicate_identity',
])
def test_measured_expiry_does_not_hide_other_invalid_evidence(source, caution, defect, future_entry_input):
  old = measured(source=source, caution=caution)
  if future_entry_input:
    old.evidence.conflictStart = .05
    if not caution:
      old.evidence.pathState = 'entering'
  frame = radar_frame(old, display=(old,))
  if defect == 'already_expired':
    old.evidence.positionMonoTime = PUBLICATION - MAX_MEASUREMENT_AGE_NS - 1
  elif defect == 'future_measurement':
    old.evidence.velocityMonoTime = PUBLICATION + 1
  elif defect == 'future_model':
    old.evidence.modelMonoTime = PUBLICATION + 1
  elif defect == 'expired_model':
    old.evidence.modelMonoTime = PUBLICATION - MEASUREMENT_AGE_AT_PUBLICATION_NS
  elif defect == 'expired_estimate':
    old.evidence.estimateMonoTime = PUBLICATION - 200_000_000
  elif defect == 'invalid_position':
    old.evidence.positionValid = False
  elif defect == 'identity_missing':
    old.evidence.sessionId = ''
  else:
    frame.leadTwo = deepcopy(old)
    if defect == 'duplicate_geometry':
      frame.leadTwo.dRel += 1.0
    else:
      frame.leadTwo.evidence.generation += 1
  consumer = RadarConsumption()
  view = consume_frame(consumer, frame)
  assert view is frame and view.leadOne.present and view.radarObjects == [old]
  assert not consumer.expired_identities and not consumer.retired_identities


@pytest.mark.parametrize('qualified,now', [
  (None, CONSUMPTION), (0, CONSUMPTION), (True, CONSUMPTION), (CONSUMPTION + 1, CONSUMPTION),
  (PUBLICATION, None), (PUBLICATION, True), (PUBLICATION, 0),
])
def test_natural_expiry_requires_actual_ordered_processing_clocks(qualified, now):
  assert not measurement_expired_since(measured(), qualified_ns=qualified, now_ns=now)


@pytest.mark.parametrize('source', ['radar', 'fused'])
def test_relevant_measured_expiry_keeps_current_failure_path_without_persistent_risk(source):
  old = measured(source=source, relevant=True)
  frame = radar_frame(old, display=(old,))
  consumer = RadarConsumption()
  view = consume_frame(consumer, frame)
  # A diagnostic degraded flag cannot preserve a moving car's braking envelope.
  # Keep relevant measured expiry on the existing input-failure path.
  assert view is frame and view.leadOne.present and not control_leads(view, 'departure', now_ns=CONSUMPTION)
  assert not consumer.expired_identities
  assert consumer.degraded
  now = CONSUMPTION + 5_000_000_000
  view = consume_frame(consumer, radar_frame(publication=now), publication=now, now=now)
  assert not consumer.degraded and not view.fusionClearedTracks
  recovered = formal_lead(now_ns=now + 1, source=source, evidence={'trackId': 9})
  consume_frame(consumer, radar_frame(recovered, publication=now + 1), publication=now + 1, now=now + 1)
  assert not consumer.degraded


@pytest.mark.parametrize('source', ['radar', 'fused'])
def test_publication_finalization_and_consumer_remove_the_same_irrelevant_expiry(source):
  old = measured(source=source)
  frame = radar_frame(old, display=(old,))
  publisher = RadarConsumption()
  view = publisher.finalize_publication(frame, qualified_ns=PUBLICATION, publication_ns=CONSUMPTION, speed=10.0)
  assert not view.radarObjects and not view.leadOne.present
  assert publisher.expired_identities == ((SESSION, 7),)
  assert frame.leadOne.present
  # A late publication does not turn the same irrelevant age-only loss into
  # a whole-frame failure, or make it qualified at the later clock.
  consumer = RadarConsumption()
  consumed = consume_frame(consumer, frame, publication=CONSUMPTION, now=CONSUMPTION)
  assert not consumed.radarObjects and not consumed.leadOne.present
  assert consumer.expired_identities == publisher.expired_identities
  assert not qualified_lead(old, 'brake', 0.0, now_ns=CONSUMPTION)
  assert not publisher.degraded and not consumer.degraded


@pytest.mark.parametrize('source', ['radar', 'fused'])
@pytest.mark.parametrize('relevant', [False, True])
@pytest.mark.parametrize('publication_offset', [-1, 0, 1])
def test_expiry_before_or_after_publication_preserves_the_same_objects_and_risk(source, relevant, publication_offset):
  old = measured(source=source, relevant=relevant)
  threshold = MAX_MEASUREMENT_AGE_NS
  deadline = old.evidence.positionMonoTime + threshold
  publication, now = deadline + publication_offset, deadline + 10_000_000
  fresh = formal_lead(now_ns=PUBLICATION, evidence={'trackId': 9})
  frame = native_frame(radar_frame(old, fresh, publication=publication, display=(old, fresh)))
  before, serialized = frame.to_dict(), frame.to_bytes()
  frame.clear_write_flag()
  consumer = RadarConsumption()
  consumed = consume_frame(consumer, frame, publication=publication, now=now)
  assert [lead.evidence.trackId for lead in control_leads(consumed)] == ([7, 9] if relevant else [9])
  assert consumed.leadOne.present is relevant and consumed.leadTwo.present
  assert [lead.evidence.trackId for lead in control_leads(consumed, 'brake', 0.0, now_ns=now)] == [9]
  assert not control_leads(consumed, 'departure', now_ns=now)
  assert consumer.expired_identities == (() if relevant else ((SESSION, 7),))
  assert consumer.degraded is relevant
  assert frame.to_dict() == before and frame.to_bytes() == serialized
  assert not consumed.fusionClearedTracks


@pytest.mark.parametrize('source', ['radar', 'fused'])
@pytest.mark.parametrize('relevant', [False, True])
def test_expiry_boundary_is_inclusive_when_it_is_also_the_publication_clock(source, relevant):
  old = measured(source=source, relevant=relevant)
  threshold = MAX_MEASUREMENT_AGE_NS
  deadline = old.evidence.positionMonoTime + threshold
  frame = radar_frame(old, publication=deadline, display=(old,))
  consumer = RadarConsumption()
  view = consume_frame(consumer, frame, publication=deadline, now=deadline)
  assert view.leadOne.present and control_leads(view) == (old,)
  assert qualified_lead(old, 'brake', 0.0, now_ns=deadline)
  assert not consumer.degraded and not consumer.expired_identities


@pytest.mark.parametrize('source', ['radar', 'fused'])
@pytest.mark.parametrize('defect', ['model_too_old', 'estimate_too_old', 'future_measurement', 'duplicate_geometry', 'invalid_position'])
def test_expired_at_publication_does_not_hide_an_independent_defect(source, defect):
  old = measured(source=source)
  frame = radar_frame(old, publication=CONSUMPTION, display=(old,))
  if defect == 'model_too_old':
    threshold = MAX_MEASUREMENT_AGE_NS
    old.evidence.modelMonoTime = CONSUMPTION - threshold - 1
  elif defect == 'estimate_too_old':
    old.evidence.estimateMonoTime = CONSUMPTION - 250_000_001
  elif defect == 'future_measurement':
    old.evidence.velocityMonoTime = CONSUMPTION + 1
  elif defect == 'invalid_position':
    old.evidence.positionValid = False
  else:
    frame.leadTwo = deepcopy(old)
    frame.leadTwo.dRel += 1.0
  consumer = RadarConsumption()
  view = consume_frame(consumer, frame, publication=CONSUMPTION, now=CONSUMPTION)
  assert view is frame and view.radarObjects == [old] and view.leadOne.present
  assert not consumer.expired_identities and not consumer.degraded


def test_pilot_sample_9_extra_future_entry_never_enters_control_or_expiry_risk():
  publication, consumption = 138_184_600_103, 138_250_174_634
  measurement, estimate, model = 137_876_723_332, 138_140_460_228, 138_181_986_405
  expiring = expiring_lead(publication=publication)
  expiring.dRel, expiring.vLead, expiring.vRel = 6.634197235107422, -.4386683702468872, -11.579492568969727
  expiring.evidence.positionMonoTime = expiring.evidence.velocityMonoTime = measurement
  expiring.evidence.estimateMonoTime, expiring.evidence.modelMonoTime = estimate, model
  expiring.evidence.pathState = 'entering'
  expiring.evidence.conflictStart, expiring.evidence.conflictEnd = 7.688506126403809, 9.749994277954102
  expiring.evidence.lateralSpeed = -.24781207740306854
  near = formal_lead(now_ns=estimate, dRel=46.99228286743164, vLead=10.578011512756348,
                     vLeadK=10.578011512756348, vRel=-.5628129243850708,
                     evidence={'trackId': 5, 'modelMonoTime': model})
  visual = formal_lead(now_ns=estimate, source='vision', dRel=58.28355026245117,
                       evidence={'modelMonoTime': model})
  other_visual = formal_lead(now_ns=estimate, source='vision', dRel=58.32783508300781,
                             evidence={'modelMonoTime': model, 'visionIndex': 1})
  raw = radar_frame(near, visual, other_visual, expiring, publication=publication, display=(near, visual))
  raw.mdMonoTime = model
  assert publication - measurement == 307_876_771
  assert consumption - publication == 65_574_531
  assert consumption - measurement == 373_451_302
  # The recorded measurement already exceeded the measurement lifetime at publication.
  assert not _qualified_lead_evidence(expiring, 'brake', now_ns=publication)
  assert not qualified_lead(expiring, 'brake', now_ns=publication)
  assert not qualified_lead(expiring, 'brake', now_ns=consumption)
  consumer = RadarConsumption()
  view = consume_frame(consumer, raw, publication=publication, now=consumption, speed=11.128089904785156)
  assert control_leads(view) == (near, visual)
  assert view.leadOne is near and view.leadTwo is visual
  assert not consumer.expired_identities and not consumer.retired_identities
  assert not consumer.degraded and not view.fusionDegraded
  assert all(qualified_lead(lead, 'brake', 0.0, now_ns=consumption) for lead in control_leads(view))


@pytest.mark.parametrize('source', ['radar', 'fused'])
@pytest.mark.parametrize('path_state,start', [('entering', 0.0), ('entering', .5), ('inPath', .5)])
def test_future_entry_retirement_removes_all_views_and_keeps_current_objects(source, path_state, start):
  future = future_entry(source=source, path_state=path_state, start=start)
  current = formal_lead(now_ns=PUBLICATION, evidence={'trackId': 9})
  edge = caution_lead(now_ns=PUBLICATION, evidence={'trackId': 12})
  native = native_frame(radar_frame(future, current, edge, display=(future, current)))
  before, wire = native.to_dict(), native.to_bytes()
  native.clear_write_flag()
  consumer = RadarConsumption()
  view = consume_frame(consumer, native)
  assert [lead.evidence.trackId for lead in control_leads(view)] == [9]
  assert not view.leadOne.present and view.leadTwo.evidence.trackId == 9
  assert [lead.evidence.trackId for lead in control_leads(view, 'brake', 0.0, now_ns=CONSUMPTION)] == [9]
  assert not control_leads(view, 'caution', 0.0, now_ns=CONSUMPTION)
  assert consumer.retired_identities == ((SESSION, 7),)
  assert not consumer.expired_identities and not consumer.degraded
  assert not view.fusionDegraded and not view.fusionClearedTracks
  assert native.to_dict() == before and native.to_bytes() == wire

  # A new current-path observation of this identity is admitted normally.
  now = CONSUMPTION + 50_000_000
  observed = formal_lead(now_ns=now)
  current_view = consume_frame(consumer, radar_frame(observed, publication=now, display=(observed,)), publication=now, now=now)
  assert control_leads(current_view, 'brake', 0.0, now_ns=now) == (observed,)
  assert not consumer.retired_identities and not consumer.expired_identities and not consumer.degraded


@pytest.mark.parametrize('field,value', [
  ('aLeadTau', -.1), ('aLeadTau', float('nan')), ('conflictEnd', 11.0),
  ('physicalConflictValid', 1), ('physicalConflictStart', float('nan')), ('physicalConflictEnd', -.1),
])
def test_future_entry_retirement_preserves_invalid_projection_fields_for_failure_handling(field, value):
  lead = future_entry()
  lead.evidence.physicalConflictValid = True
  lead.evidence.physicalConflictStart, lead.evidence.physicalConflictEnd = .5, 1.0
  setattr(lead if field == 'aLeadTau' else lead.evidence, field, value)
  frame = radar_frame(lead, display=(lead,))
  consumer = RadarConsumption()
  view = consume_frame(consumer, frame)
  assert view is frame and view.leadOne.present and control_leads(view) == (lead,)
  assert not consumer.retired_identities and not consumer.expired_identities and not consumer.degraded


@pytest.mark.parametrize('path_state,brake', [('inPath', True), ('entering', True), ('entering', False)])
def test_future_entry_mixed_roles_and_nonedge_caution_stay_on_failure_path(path_state, brake):
  lead = caution_lead(now_ns=PUBLICATION, evidence={'pathState': path_state, 'conflictStart': .5, 'brakeQualified': brake})
  frame = radar_frame(lead, display=(lead,))
  consumer = RadarConsumption()
  view = consume_frame(consumer, frame)
  assert view is frame and view.leadOne.present and control_leads(view, now_ns=CONSUMPTION) == (lead,)
  assert not consumer.retired_identities and not consumer.expired_identities


@pytest.mark.parametrize('defect', [
  'already_expired', 'future_model', 'future_measurement', 'old_model', 'old_estimate', 'invalid_position',
  'nonfinite_velocity', 'invalid_decay', 'unqualified', 'forged_coast', 'different_display', 'different_generation',
])
@pytest.mark.parametrize('future_entry_input', [False, True])
def test_bad_or_already_stale_publication_is_not_hidden_by_expiry(defect, future_entry_input):
  expiring = expiring_lead()
  if future_entry_input:
    expiring.evidence.pathState = 'entering'
    expiring.evidence.conflictStart = .5
  raw = radar_frame(expiring, display=(expiring,))
  if defect == 'already_expired':
    expiring.evidence.positionMonoTime = PUBLICATION - MAX_MEASUREMENT_AGE_NS - 1
  elif defect == 'future_model':
    expiring.evidence.modelMonoTime = PUBLICATION + 1
  elif defect == 'future_measurement':
    expiring.evidence.velocityMonoTime = PUBLICATION + 1
  elif defect == 'old_model':
    expiring.evidence.modelMonoTime = PUBLICATION - MEASUREMENT_AGE_AT_PUBLICATION_NS
  elif defect == 'old_estimate':
    expiring.evidence.estimateMonoTime = PUBLICATION - 220_000_000
  elif defect == 'invalid_position':
    expiring.evidence.positionValid = False
  elif defect == 'nonfinite_velocity':
    expiring.vLead = float('nan')
  elif defect == 'invalid_decay':
    expiring.aLeadTau = -1.0
  elif defect == 'unqualified':
    expiring.evidence.brakeQualified = False
  elif defect == 'forged_coast':
    expiring.evidence.coasting = True
  else:
    raw.leadTwo = deepcopy(expiring)
    if defect == 'different_display':
      raw.leadTwo.dRel += 1.0
    else:
      raw.leadTwo.evidence.generation += 1
  consumer = RadarConsumption()
  view = consume_frame(consumer, raw)
  assert view is raw and raw.radarObjects == [expiring] and raw.leadOne.present
  assert consumer.expired_identities == ()
  assert consumer.retired_identities == ()


@pytest.mark.parametrize('publication', [0, None, True, CONSUMPTION + 1])
def test_missing_or_future_publication_cannot_prove_natural_age(publication):
  expiring = expiring_lead()
  assert not measurement_expired_since(expiring, qualified_ns=publication, now_ns=CONSUMPTION)
  raw = radar_frame(expiring)
  assert consume_frame(RadarConsumption(), raw, publication=publication) is raw


@pytest.mark.parametrize('source', ['radar', 'fused'])
@pytest.mark.parametrize('publication', [PUBLICATION, CONSUMPTION])
@pytest.mark.parametrize('next_frame', ['empty', 'unrelated'])
def test_relevant_expiry_does_not_survive_empty_or_unrelated_fresh_messages(source, publication, next_frame):
  consumer = RadarConsumption()
  lost = measured(source=source, relevant=True)
  raw = radar_frame(lost, publication=publication, display=(lost,))
  view = consume_frame(consumer, raw, publication=publication)
  assert view.radarObjects == [lost] and consumer.degraded
  assert not consumer.expired_identities
  assert not raw.fusionRiskUnresolved and not view.fusionRiskUnresolved
  for elapsed in (100_000_000, 2_000_000_000, 30_000_000_000):
    now = PUBLICATION + elapsed
    other = formal_lead(now_ns=now, evidence={'trackId': 9})
    frame = radar_frame(publication=now) if next_frame == 'empty' else radar_frame(other, publication=now, display=(other,))
    consume_frame(consumer, frame, publication=now, now=now)
    assert not consumer.degraded


@pytest.mark.parametrize('distance,expected_loss', [(80.0, False), (6.6, True)])
def test_consumption_age_does_not_turn_a_forecast_boundary_into_a_far_crossing(distance, expected_loss):
  expiring = expiring_lead(relevant=True)
  expiring.dRel = distance
  expiring.evidence.conflictEnd = 10.0
  consumer = RadarConsumption()
  frame = radar_frame(expiring, display=(expiring,))
  view = consume_frame(consumer, frame)
  assert bool(view.radarObjects) is expected_loss and view.leadOne.present is expected_loss
  assert consumer.expired_identities == (() if expected_loss else ((SESSION, 7),))
  assert consumer.degraded is expected_loss
  assert frame.fusionClearedTracks == view.fusionClearedTracks == ()
  assert expiring.evidence.positionMonoTime == PUBLICATION - MEASUREMENT_AGE_AT_PUBLICATION_NS
  assert expiring.evidence.estimateMonoTime == PUBLICATION


@pytest.mark.parametrize('source', ['radar', 'fused'])
@pytest.mark.parametrize('publication', [PUBLICATION, CONSUMPTION])
@pytest.mark.parametrize('recovery', ['same', 'different_session', 'different_track', 'migrated_slot', 'coast', 'caution',
                                      'old_position', 'old_velocity'])
def test_current_frame_does_not_require_recovery_of_previous_identity(recovery, source, publication):
  consumer = RadarConsumption()
  lost = measured(source=source, relevant=True)
  consume_frame(consumer, radar_frame(lost, publication=publication, display=(lost,)), publication=publication)
  now = CONSUMPTION + 100_000_000
  lead = formal_lead(now_ns=now, evidence={'generation': 8})
  if recovery == 'different_session':
    lead.evidence.sessionId = 'restarted'
  elif recovery == 'different_track':
    lead.evidence.trackId = 8
  elif recovery == 'migrated_slot':
    # Generation belongs to the slot, not the physical object's lifetime.
    lead.evidence.generation = 0
  elif recovery == 'coast':
    lead.evidence.source, lead.evidence.coasting = 'predicted', True
  elif recovery == 'caution':
    lead = caution_lead(now_ns=now)
  elif recovery in ('old_position', 'old_velocity'):
    field = 'positionMonoTime' if recovery == 'old_position' else 'velocityMonoTime'
    setattr(lead.evidence, field, getattr(lost.evidence, field))
  view = consume_frame(consumer, radar_frame(lead, publication=now, display=(lead,)), publication=now, now=now)
  assert not consumer.degraded
  if recovery in ('old_position', 'old_velocity'):
    # Invalid source-time measurements remain available to failure handling;
    # their timestamps cannot renew this object's control authority.
    assert view.leadOne.present and not qualified_lead(view.leadOne, 'brake', 0.0, now_ns=now)


def test_clear_metadata_is_preserved_without_remembering_previous_expiry():
  consumer = RadarConsumption()
  lost = expiring_lead(relevant=True)
  consume_frame(consumer, radar_frame(lost, display=(lost,)))
  for session in ('other-session', SESSION):
    now = CONSUMPTION + 100_000_000
    frame = radar_frame(publication=now, session=session, cleared=(7,))
    view = consume_frame(consumer, frame, publication=now, now=now)
    assert not consumer.degraded
    assert view.fusionSessionId == session and view.fusionClearedTracks == (7,)
  consume_frame(consumer, radar_frame(lost, display=(lost,)))
  assert consumer.degraded
  consumer.clear()
  assert not consumer.degraded


def test_consumer_does_not_clear_original_degraded_status_or_publish_a_clear():
  expiring = expiring_lead()
  raw = radar_frame(expiring, display=(expiring,), degraded=True, cleared=(12,))
  consumer = RadarConsumption()
  view = consume_frame(consumer, raw)
  assert not consumer.degraded and view.fusionDegraded and raw.fusionDegraded
  assert view.fusionClearedTracks == raw.fusionClearedTracks == (12,)


def test_expiry_view_ignores_legacy_persistent_risk_metadata():
  old = expiring_lead()
  raw = radar_frame(old, display=(old,), risk_unresolved=True)
  consumer = RadarConsumption()
  view = consume_frame(consumer, raw)
  assert not view.fusionRiskUnresolved and not consumer.degraded
  assert raw.fusionRiskUnresolved
  raw.fusionRiskUnresolved = False
  assert not view.fusionRiskUnresolved
  del raw.fusionRiskUnresolved
  assert not consume_frame(consumer, raw).fusionRiskUnresolved


def test_extra_caution_does_not_prevent_a_measured_lead_expiring_or_create_a_loss_latch():
  edge = caution_lead(now_ns=PUBLICATION, evidence={'trackId': 12})
  old = expiring_lead()
  raw = radar_frame(edge, old, display=(old,))
  consumer = RadarConsumption()
  view = consume_frame(consumer, raw)
  assert not control_leads(view)
  assert not control_leads(view, 'caution', now_ns=CONSUMPTION)
  assert not consumer.degraded
  now = CONSUMPTION + 5_000_000_000
  consume_frame(consumer, radar_frame(publication=now), publication=now, now=now)
  assert not consumer.degraded


def native_frame(frame):
  from openpilot.cereal import log

  def lead_dict(lead):
    data = vars(lead).copy()
    if 'evidence' in data:
      data['evidence'] = vars(data['evidence']).copy()
    return data

  data = vars(frame).copy()
  data['radarObjects'] = [lead_dict(lead) for lead in frame.radarObjects]
  data['leadOne'], data['leadTwo'] = (lead_dict(lead) for lead in (frame.leadOne, frame.leadTwo))
  return log.RadarState.new_message(**data)


@pytest.mark.parametrize('extra_kind', ['expired', 'malformed', 'same_identity', 'oversize', 'caution', 'future'])
def test_extra_objects_cannot_invalidate_or_replace_a_current_formal_lead(extra_kind):
  current = formal_lead(now_ns=PUBLICATION, evidence={'trackId': 9})
  extra = expiring_lead(relevant=True)
  if extra_kind == 'malformed':
    extra.dRel = float('nan')
  elif extra_kind == 'same_identity':
    extra = deepcopy(current)
    extra.evidence.positionValid = False
  elif extra_kind == 'caution':
    extra = caution_lead(now_ns=PUBLICATION)
  elif extra_kind == 'future':
    extra = future_entry()
  objects = [extra] * (40 if extra_kind == 'oversize' else 1)
  native = native_frame(radar_frame(*objects, display=(current,)))
  wire = native.to_bytes()
  native.clear_write_flag()
  consumer = RadarConsumption()
  view = consume_frame(consumer, native)
  assert [lead.evidence.trackId for lead in control_leads(view)] == [9]
  assert qualified_lead(view.leadOne, 'brake', 0.0, now_ns=CONSUMPTION)
  assert not view.radarObjects and not consumer.degraded
  assert not consumer.expired_identities and not consumer.retired_identities
  # Byte equality also checks the unchanged NaN payload, without NaN != NaN.
  assert native.to_bytes() == wire


def test_extra_wire_decode_errors_are_not_control_input_errors():
  current = formal_lead(now_ns=PUBLICATION, evidence={'trackId': 9})
  extra = formal_lead(now_ns=PUBLICATION)
  native = native_frame(radar_frame(extra, display=(current,)))
  native.radarObjects[0].evidence.sessionId = b'\xff'
  native.radarObjects[0].evidence.source = 65535
  consumer = RadarConsumption()
  view = consume_frame(consumer, native)
  assert [lead.evidence.trackId for lead in control_leads(view)] == [9]
  assert not view.radarObjects and not consumer.degraded


def test_raw_collection_and_old_cached_collection_are_never_read_as_control_authority():
  class UnreadObjects(SimpleNamespace):
    @property
    def radarObjects(self):
      raise AssertionError('extra control collection must not be read')

  current = formal_lead(now_ns=PUBLICATION, evidence={'trackId': 9})
  frame = radar_frame(display=(current,))
  raw = UnreadObjects(**{key: value for key, value in vars(frame).items() if key != 'radarObjects'})
  consumer = RadarConsumption()
  view = consume_frame(consumer, raw)
  assert control_leads(view, 'brake', 0.0, now_ns=CONSUMPTION) == (current,)
  snapshot = consume_frame(consumer, native_frame(frame))
  object.__setattr__(snapshot, '_control_objects', (expiring_lead(relevant=True),))
  assert [lead.evidence.trackId for lead in control_leads(snapshot)] == [9]
  assert not consumer.degraded


def test_native_message_bytes_and_all_source_timestamps_remain_unchanged():
  expiring = expiring_lead()
  fresh = formal_lead(now_ns=PUBLICATION, evidence={'trackId': 5})
  native = native_frame(radar_frame(fresh, expiring, display=(expiring, fresh), risk_unresolved=True))
  before, serialized = native.to_dict(), native.to_bytes()
  native.clear_write_flag()
  consumer = RadarConsumption()
  view = consume_frame(consumer, native)
  assert not view.leadOne.present and view.leadTwo.present
  assert [lead.evidence.trackId for lead in control_leads(view)] == [5]
  assert native.to_dict() == before and native.to_bytes() == serialized
  assert len(native.radarObjects) == 2 and native.leadOne.present
  assert not view.fusionRiskUnresolved and not view.to_dict()['fusionRiskUnresolved']


def test_native_tick_snapshot_freezes_values_without_caching_qualification():
  lead = formal_lead(now_ns=PUBLICATION, source='vision', vRel=1.0,
                     evidence={'conflictEnd': .1})
  native = native_frame(radar_frame(lead, display=(lead,)))
  consumer = RadarConsumption()
  view = consume_frame(consumer, native, now=PUBLICATION)
  frozen = view.leadOne
  assert qualified_lead(frozen, 'brake', .95, now_ns=PUBLICATION)
  assert not qualified_lead(frozen, 'brake', 1.0, now_ns=PUBLICATION)
  assert not qualified_lead(frozen, 'caution', 0.0, now_ns=PUBLICATION)
  raw = native.as_reader().leadOne
  assert frozen.evidence.conflictEnd == raw.evidence.conflictEnd
  # The earliest source lease expires the ordinary lead. The retired finite
  # occupancy endpoint cannot shorten or extend any of those leases.
  boundary = PUBLICATION + min(MAX_MEASUREMENT_AGE_NS, MAX_MODEL_AGE_NS)
  for candidate in (raw, frozen):
    assert qualified_lead(candidate, 'brake', .95, now_ns=boundary - 1)
    assert qualified_lead(candidate, 'brake', .95, now_ns=boundary)
    assert not qualified_lead(candidate, 'brake', .95, now_ns=boundary + 1)
    assert not qualified_lead(candidate, 'brake', .95, now_ns=boundary + 1, require_active_window=False)
  with pytest.raises(AttributeError):
    frozen.dRel = 0.0
  with pytest.raises(AttributeError):
    frozen.evidence.modelMonoTime = PUBLICATION + 1

  native.radarObjects[0].dRel = native.leadOne.dRel = 40.0
  later = consume_frame(consumer, native, now=PUBLICATION)
  assert later.leadOne.dRel == 40.0 and not later.radarObjects
  assert frozen.dRel == 20.0 and view.leadOne.dRel == 20.0


@pytest.mark.parametrize('source', ['radar', 'fused', 'vision', 'predicted'])
def test_native_field_snapshot_preserves_each_source_and_purpose(source):
  lead = formal_lead(now_ns=PUBLICATION, source=source, vRel=1.0)
  native = native_frame(radar_frame(lead, display=(lead,)))
  view = consume_frame(RadarConsumption(), native, now=PUBLICATION)
  for purpose in ('follow', 'brake', 'stop', 'departure', 'fcw', 'caution'):
    for probability in (0.0, .95, 1.0):
      assert qualified_lead(view.leadOne, purpose, probability, now_ns=PUBLICATION) == qualified_lead(
        native.leadOne, purpose, probability, now_ns=PUBLICATION)


@pytest.mark.parametrize('field', ['dRel', 'generation', 'positionMonoTime', 'followGeometryMonoTime'])
def test_native_duplicate_identity_conflicts_survive_snapshotting(field):
  from openpilot.cereal import log

  expiring = expiring_lead()
  native = native_frame(radar_frame(expiring, display=(expiring, deepcopy(expiring))))
  if field == 'dRel':
    native.leadTwo.dRel += 1.0
  else:
    evidence = native.leadTwo.evidence
    setattr(evidence, field, getattr(evidence, field) + 1)
  before, wire = native.to_dict(), native.to_bytes()
  native.clear_write_flag()
  consumer = RadarConsumption()
  with log.RadarState.from_bytes(wire) as reader:
    view = consume_frame(consumer, reader)
    assert view is reader and consumer.publication_view is None
    assert consumer.expired_identities == () and not consumer.degraded
    assert len(view.radarObjects) == 1 and view.leadOne.present
    assert view.to_dict() == before
  assert native.to_dict() == before and native.to_bytes() == wire


def test_same_native_publication_rechecked_at_a_later_consumption_time():
  expiring = expiring_lead()
  native = native_frame(radar_frame(expiring, display=(expiring,)))
  consumer = RadarConsumption()
  boundary = expiring.evidence.positionMonoTime + MAX_MEASUREMENT_AGE_NS
  first = consume_frame(consumer, native, now=boundary)
  assert first.leadOne.present and qualified_lead(first.leadOne, 'brake', now_ns=boundary)
  second = consume_frame(consumer, native, now=boundary + 1)
  assert not second.leadOne.present and second.radarObjects == ()
  assert first.leadOne.present and not qualified_lead(first.leadOne, 'brake', now_ns=boundary + 1)
  assert native.leadOne.present and len(native.radarObjects) == 1


def test_snapshotting_does_not_allocate_absent_builder_evidence():
  native = native_frame(radar_frame())
  assert not native.leadOne._has('evidence') and not native.leadTwo._has('evidence')
  before, serialized = native.to_dict(), native.to_bytes()
  native.clear_write_flag()
  consume_frame(RadarConsumption(), native, now=PUBLICATION)
  assert not native.leadOne._has('evidence') and not native.leadTwo._has('evidence')
  assert native.to_dict() == before and native.to_bytes() == serialized


@pytest.mark.parametrize('defect', ['absent_unknown_enum', 'absent_bad_utf8', 'nan_probability_bad_utf8'])
def test_unread_malformed_wire_fields_do_not_add_a_snapshot_exception(defect):
  from openpilot.cereal import log

  native = native_frame(radar_frame())
  native.leadOne.evidence.version = 0
  if defect == 'absent_unknown_enum':
    native.leadOne.evidence.source = 65535
  else:
    native.leadOne.evidence.sessionId = b'\xff'
  if defect == 'nan_probability_bad_utf8':
    native.leadOne.present, native.leadOne.modelProb = True, float('nan')
  serialized = native.to_bytes()
  native.clear_write_flag()
  consumer = RadarConsumption()
  with log.RadarState.from_bytes(serialized) as reader:
    view = consume_frame(consumer, reader, now=PUBLICATION)
    assert view is reader
    assert view.leadOne.present is (defect == 'nan_probability_bad_utf8')
    assert consumer.expired_identities == () and not consumer.degraded
  # The fallback must also remain observational when given a builder.
  builder_view = consume_frame(consumer, native, now=PUBLICATION)
  assert builder_view.leadOne.present is (defect == 'nan_probability_bad_utf8')
  assert not native.leadTwo._has('evidence')
  assert native.to_bytes() == serialized


def test_snapshot_decode_fallback_retains_relevant_expiry():
  from openpilot.cereal import log

  expiring = expiring_lead(relevant=True)
  native = native_frame(radar_frame(expiring, display=(SimpleNamespace(present=False), expiring)))
  native.leadOne.evidence.version = 0
  native.leadOne.evidence.sessionId = b'\xff'
  serialized = native.to_bytes()
  native.clear_write_flag()
  consumer = RadarConsumption()
  with log.RadarState.from_bytes(serialized) as reader:
    view = consume_frame(consumer, reader)
    assert len(view.radarObjects) == 1 and not view.leadOne.present and view.leadTwo.present
    assert consumer.expired_identities == ()
    assert consumer.degraded
    assert len(reader.radarObjects) == 1 and reader.leadTwo.present
  builder_consumer = RadarConsumption()
  view = consume_frame(builder_consumer, native)
  assert len(view.radarObjects) == 1 and builder_consumer.degraded
  assert native.to_bytes() == serialized


@pytest.mark.parametrize('relevant', [False, True])
def test_validated_collection_is_immutable_and_expiry_preserves_relevant_objects(relevant):
  expiring, fresh = expiring_lead(relevant=relevant), formal_lead(now_ns=PUBLICATION, evidence={'trackId': 8})
  native = native_frame(radar_frame(expiring, fresh, display=(expiring, fresh)))
  consumer = RadarConsumption()
  boundary = expiring.evidence.positionMonoTime + MAX_MEASUREMENT_AGE_NS
  first = consume_frame(consumer, native, now=boundary)
  objects = control_leads(first)
  assert len(objects) == 2 and control_leads(first) == objects
  assert not first.radarObjects
  for field in ('radarObjects', 'leadOne', 'fusionActive', '_control_objects'):
    with pytest.raises(AttributeError):
      setattr(first, field, None)
  second = consume_frame(consumer, native, now=boundary + 1)
  assert [lead.evidence.trackId for lead in control_leads(second)] == ([7, 8] if relevant else [8])
  assert control_leads(second) == control_leads(second)
  assert consumer.degraded is relevant
  assert [lead.evidence.trackId for lead in control_leads(first)] == [7, 8]
  assert not control_leads(first, 'brake', now_ns=boundary + 250_000_000)
  native.fusionActive = False
  assert first.fusionActive and first.to_dict()['fusionActive']
  assert control_leads(first) == objects


@pytest.mark.parametrize('field', ['followQualified', 'stopQualified', 'departureQualified', 'fcwQualified'])
def test_duplicate_role_conflicts_prevent_installing_a_deduplicated_collection(field):
  lead = formal_lead(now_ns=PUBLICATION)
  native = native_frame(radar_frame(lead, display=(lead, deepcopy(lead))))
  setattr(native.leadTwo.evidence, field, False)
  consumer = RadarConsumption()
  view = consume_frame(consumer, native, now=PUBLICATION)
  assert not isinstance(view, RadarConsumptionView)
  assert len(view.radarObjects) == 1 and view.leadOne.present
  assert getattr(view.leadOne.evidence, field) is True
  assert getattr(view.leadTwo.evidence, field) is False


def test_complete_publication_reuse_keeps_expired_hold_attribution():
  from openpilot.selfdrive.controls.tests.test_object_influence import plan_records

  old = expiring_lead(relevant=True)
  native = native_frame(radar_frame(old, display=(old,)))
  raw = native.as_reader()
  consumer = RadarConsumption()
  consumed = consume_frame(consumer, raw)
  publication = consumer.publication_view
  assert publication is not None and publication._radar is raw and consumer.publication_ns == PUBLICATION
  assert not consumed.radarObjects and not publication.radarObjects
  assert consumed.leadOne.present and publication.leadOne.present
  assert consumer.degraded
  kwargs = {'now_ns': CONSUMPTION, 'reference_accepted': False, 'held_identity': (SESSION, 7), 'target_accel': 0.0}
  raw_records = plan_records(raw, **kwargs)
  assert raw_records and raw_records[0]['kind'] == 'hold'
  assert plan_records(publication, **kwargs) == raw_records
  assert plan_records(consumed, **kwargs) == raw_records
  consumer.clear_publication()
  assert consumer.publication_view is None and consumer.publication_ns is None
  assert consumer.degraded


@pytest.mark.parametrize('defect', ['role_conflict', 'inactive', 'invalid_speed', 'undecodable_absent'])
def test_publication_reuse_is_cleared_before_each_invalid_or_unoptimized_input(defect):
  lead = formal_lead(now_ns=PUBLICATION)
  native = native_frame(radar_frame(lead, display=(lead, deepcopy(lead))))
  consumer = RadarConsumption()
  consume_frame(consumer, native.as_reader(), now=PUBLICATION)
  assert consumer.publication_view is not None
  speed = 10.0
  if defect == 'role_conflict':
    native.leadOne.evidence.followQualified = False
  elif defect == 'inactive':
    native.fusionActive = False
  elif defect == 'invalid_speed':
    speed = float('nan')
  else:
    native.leadTwo.evidence.version = 0
    native.leadTwo.evidence.sessionId = b'\xff'
  consume_frame(consumer, native.as_reader(), now=PUBLICATION, speed=speed)
  assert consumer.publication_view is None and consumer.publication_ns is None


@pytest.mark.parametrize('age_ns', [0, 150_000_000, 350_000_000])
def test_legacy_prediction_is_not_renewed_or_admitted_by_consumption(age_ns):
  old = formal_lead(now_ns=PUBLICATION, source='predicted', evidence={
    'positionMonoTime': PUBLICATION - age_ns, 'velocityMonoTime': PUBLICATION - age_ns,
  })
  frame = native_frame(radar_frame(old, display=(old,)))
  before = frame.to_dict()
  consumer = RadarConsumption()
  view = consume_frame(consumer, frame, now=PUBLICATION)
  assert view.to_dict() == frame.to_dict() == before
  for purpose in ('follow', 'brake', 'stop', 'departure', 'fcw'):
    assert not qualified_lead(view.leadOne, purpose, 0.0, now_ns=PUBLICATION)
    assert not control_leads(view, purpose, 0.0, now_ns=PUBLICATION)
  assert not consumer.expired_identities and not consumer.degraded
