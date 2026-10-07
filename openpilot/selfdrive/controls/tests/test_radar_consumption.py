"""Message age must retire one coast without laundering invalid evidence."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.lead_qualification import (
  MAX_PREDICTION_AGE_NS, coast_measurement_expired, control_leads, qualified_lead,
)
from openpilot.selfdrive.controls.lib.radar_consumption import RadarConsumption
from openpilot.selfdrive.controls.tests.test_lead_qualification import caution_lead, formal_lead


PUBLICATION = 10_000_000_000
CONSUMPTION = PUBLICATION + 60_000_000
SESSION = 'radar-session-a'


def coast(*, publication=PUBLICATION, relevant=False, track=7):
  return formal_lead(now_ns=publication, source='predicted', dRel=6.6, vLead=0.0, vLeadK=0.0, vRel=-10.0,
                     evidence={'trackId': track, 'positionMonoTime': publication - 300_000_000,
                               'velocityMonoTime': publication - 300_000_000, 'generation': 7,
                               'followQualified': False, 'stopQualified': False, 'departureQualified': False,
                               'fcwQualified': False, 'accelerationValid': False,
                               'pathState': 'inPath' if relevant else 'entering',
                               'conflictStart': 0.0 if relevant else 7.7, 'conflictEnd': 9.8,
                               'lateralSpeed': 0.0 if relevant else -.25})


def radar_frame(*objects, publication=PUBLICATION, display=(), degraded=False, risk_unresolved=False, cleared=(), session=SESSION):
  empty = SimpleNamespace(present=False)
  slots = (*display, empty, empty)
  return SimpleNamespace(fusionActive=True, fusionHealthy=True, fusionDegraded=degraded, fusionReason='active',
                         fusionRiskUnresolved=risk_unresolved,
                         fusionSessionId=session, fusionClearedTracks=cleared, mdMonoTime=publication,
                         radarObjects=list(objects), leadOne=slots[0], leadTwo=slots[1])


def consume_frame(consumer, frame, *, publication=PUBLICATION, now=CONSUMPTION, speed=10.0):
  return consumer.update(frame, publication_ns=publication, now_ns=now, speed=speed)


def test_350ms_boundary_is_inclusive_and_removal_does_not_refresh_a_stamp():
  predicted = coast()
  raw = radar_frame(predicted, display=(predicted,))
  consumer = RadarConsumption()
  boundary = predicted.evidence.positionMonoTime + MAX_PREDICTION_AGE_NS
  before = deepcopy(vars(predicted.evidence))
  assert qualified_lead(predicted, 'brake', now_ns=boundary)
  assert consume_frame(consumer, raw, now=boundary) is raw
  view = consume_frame(consumer, raw, now=boundary + 1)
  assert not qualified_lead(predicted, 'brake', now_ns=boundary + 1)
  assert view.radarObjects == () and not view.leadOne.present
  assert not view.leadOne.as_reader().present
  assert control_leads(view) == ()
  assert raw.radarObjects == [predicted] and raw.leadOne.present
  assert vars(predicted.evidence) == before
  assert not consumer.degraded
  assert raw.fusionClearedTracks == view.fusionClearedTracks == ()


def test_pilot_sample_9_uses_actual_publication_and_consumer_clocks():
  publication, consumption = 138_184_600_103, 138_250_174_634
  measurement, estimate, model = 137_876_723_332, 138_140_460_228, 138_181_986_405
  predicted = coast(publication=publication)
  predicted.dRel, predicted.vLead, predicted.vRel = 6.634197235107422, -.4386683702468872, -11.579492568969727
  predicted.evidence.positionMonoTime = predicted.evidence.velocityMonoTime = measurement
  predicted.evidence.estimateMonoTime, predicted.evidence.modelMonoTime = estimate, model
  predicted.evidence.conflictStart, predicted.evidence.conflictEnd = 7.688506126403809, 9.749994277954102
  predicted.evidence.lateralSpeed = -.24781207740306854
  near = formal_lead(now_ns=estimate, dRel=46.99228286743164, vLead=10.578011512756348,
                     vLeadK=10.578011512756348, vRel=-.5628129243850708,
                     evidence={'trackId': 5, 'modelMonoTime': model})
  visual = formal_lead(now_ns=estimate, source='vision', dRel=58.28355026245117,
                       evidence={'modelMonoTime': model})
  other_visual = formal_lead(now_ns=estimate, source='vision', dRel=58.32783508300781,
                             evidence={'modelMonoTime': model, 'visionIndex': 1})
  raw = radar_frame(near, visual, other_visual, predicted, publication=publication, display=(near, visual))
  raw.mdMonoTime = model
  assert publication - measurement == 307_876_771
  assert consumption - publication == 65_574_531
  assert consumption - measurement == 373_451_302
  assert qualified_lead(predicted, 'brake', now_ns=publication)
  assert not qualified_lead(predicted, 'brake', now_ns=consumption)
  consumer = RadarConsumption()
  view = consume_frame(consumer, raw, publication=publication, now=consumption, speed=11.128089904785156)
  assert control_leads(view) == (near, visual, other_visual)
  assert view.leadOne is near and view.leadTwo is visual
  assert consumer.expired_identities == ((SESSION, 7),)
  assert not consumer.degraded and not view.fusionDegraded
  assert all(qualified_lead(lead, 'brake', 0.0, now_ns=consumption) for lead in control_leads(view))


@pytest.mark.parametrize('defect', [
  'already_expired', 'future_model', 'future_measurement', 'old_model', 'old_estimate', 'invalid_position',
  'nonfinite_velocity', 'invalid_decay', 'unqualified', 'bad_window', 'not_coast', 'different_display', 'different_generation',
  'different_physical_valid', 'different_physical_start', 'different_physical_end',
])
def test_bad_or_already_stale_publication_is_not_hidden_by_expiry(defect):
  predicted = coast()
  raw = radar_frame(predicted, display=(predicted,))
  if defect == 'already_expired':
    predicted.evidence.positionMonoTime = PUBLICATION - MAX_PREDICTION_AGE_NS - 1
  elif defect == 'future_model':
    predicted.evidence.modelMonoTime = PUBLICATION + 1
  elif defect == 'future_measurement':
    predicted.evidence.velocityMonoTime = PUBLICATION + 1
  elif defect == 'old_model':
    predicted.evidence.modelMonoTime = PUBLICATION - 300_000_000
  elif defect == 'old_estimate':
    predicted.evidence.estimateMonoTime = PUBLICATION - 220_000_000
  elif defect == 'invalid_position':
    predicted.evidence.positionValid = False
  elif defect == 'nonfinite_velocity':
    predicted.vLead = float('nan')
  elif defect == 'invalid_decay':
    predicted.aLeadTau = -1.0
  elif defect == 'unqualified':
    predicted.evidence.brakeQualified = False
  elif defect == 'bad_window':
    predicted.evidence.conflictEnd = 11.0
  elif defect == 'not_coast':
    predicted.evidence.coasting = False
  else:
    raw.leadOne = deepcopy(predicted)
    if defect == 'different_display':
      raw.leadOne.dRel += 1.0
    elif defect == 'different_physical_valid':
      raw.leadOne.evidence.physicalConflictValid = True
    elif defect == 'different_physical_start':
      raw.leadOne.evidence.physicalConflictStart = 0.1
    elif defect == 'different_physical_end':
      raw.leadOne.evidence.physicalConflictEnd = 0.2
    else:
      raw.leadOne.evidence.generation += 1
  consumer = RadarConsumption()
  view = consume_frame(consumer, raw)
  assert view is raw and raw.radarObjects == [predicted] and raw.leadOne.present
  assert consumer.expired_identities == ()


@pytest.mark.parametrize('publication', [0, None, True, CONSUMPTION + 1])
def test_missing_or_future_publication_cannot_prove_natural_age(publication):
  predicted = coast()
  assert not coast_measurement_expired(predicted, publication_ns=publication, now_ns=CONSUMPTION)
  raw = radar_frame(predicted)
  assert consume_frame(RadarConsumption(), raw, publication=publication) is raw


def test_relevant_loss_persists_through_empty_fresh_messages_and_unrelated_recovery():
  consumer = RadarConsumption()
  predicted = coast(relevant=True)
  raw = radar_frame(predicted, display=(predicted,))
  view = consume_frame(consumer, raw)
  assert view.radarObjects == () and consumer.degraded
  assert not raw.fusionRiskUnresolved and not view.fusionRiskUnresolved
  for elapsed in (100_000_000, 2_000_000_000, 30_000_000_000):
    now = PUBLICATION + elapsed
    other = formal_lead(now_ns=now, evidence={'trackId': 9})
    consume_frame(consumer, radar_frame(other, publication=now, display=(other,)), publication=now, now=now)
    assert consumer.degraded and consumer.lost_identities == ((SESSION, 7),)


@pytest.mark.parametrize('distance,expected_loss', [(80.0, False), (6.6, True)])
def test_consumption_age_does_not_turn_a_forecast_boundary_into_a_far_crossing(distance, expected_loss):
  predicted = coast(relevant=True)
  predicted.dRel = distance
  predicted.evidence.conflictEnd = 10.0
  consumer = RadarConsumption()
  frame = radar_frame(predicted, display=(predicted,))
  view = consume_frame(consumer, frame)
  assert not view.radarObjects and not view.leadOne.present
  assert consumer.expired_identities == ((SESSION, 7),)
  assert consumer.degraded is expected_loss
  assert frame.fusionClearedTracks == view.fusionClearedTracks == ()
  assert predicted.evidence.positionMonoTime == PUBLICATION - 300_000_000
  assert predicted.evidence.estimateMonoTime == PUBLICATION


@pytest.mark.parametrize('recovery', ['same', 'different_session', 'different_track', 'migrated_slot', 'coast', 'caution'])
def test_only_fresh_same_physical_identity_recovers_the_loss(recovery):
  consumer = RadarConsumption()
  consume_frame(consumer, radar_frame(coast(relevant=True)))
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
  consume_frame(consumer, radar_frame(lead, publication=now, display=(lead,)), publication=now, now=now)
  assert consumer.degraded is (recovery not in ('same', 'migrated_slot'))


def test_explicit_fresh_clear_is_scoped_to_session_and_not_invented_from_ttl():
  consumer = RadarConsumption()
  consume_frame(consumer, radar_frame(coast(relevant=True)))
  for session, expected in [('other-session', True), (SESSION, False)]:
    now = CONSUMPTION + 100_000_000
    frame = radar_frame(publication=now, session=session, cleared=(7,))
    consume_frame(consumer, frame, publication=now, now=now)
    assert consumer.degraded is expected
  consume_frame(consumer, radar_frame(coast(relevant=True)))
  assert consumer.degraded
  consumer.clear()
  assert not consumer.degraded


def test_consumer_does_not_clear_original_degraded_status_or_publish_a_clear():
  predicted = coast()
  raw = radar_frame(predicted, display=(predicted,), degraded=True, cleared=(12,))
  consumer = RadarConsumption()
  view = consume_frame(consumer, raw)
  assert not consumer.degraded and view.fusionDegraded and raw.fusionDegraded
  assert view.fusionClearedTracks == raw.fusionClearedTracks == (12,)


def test_expiry_view_freezes_producer_risk_metadata_and_old_schema_defaults_false():
  raw = radar_frame(coast(), risk_unresolved=True)
  consumer = RadarConsumption()
  view = consume_frame(consumer, raw)
  assert view.fusionRiskUnresolved and not consumer.degraded
  raw.fusionRiskUnresolved = False
  assert view.fusionRiskUnresolved
  del raw.fusionRiskUnresolved
  assert not consume_frame(consumer, raw).fusionRiskUnresolved


def test_valid_caution_does_not_prevent_another_coast_expiring_or_create_a_loss_latch():
  edge = caution_lead(now_ns=PUBLICATION, evidence={'trackId': 12})
  raw = radar_frame(edge, coast())
  consumer = RadarConsumption()
  view = consume_frame(consumer, raw)
  assert control_leads(view) == (edge,)
  assert control_leads(view, 'caution', now_ns=CONSUMPTION) == (edge,)
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


def test_native_message_bytes_and_all_source_timestamps_remain_unchanged():
  predicted = coast()
  fresh = formal_lead(now_ns=PUBLICATION, evidence={'trackId': 5})
  native = native_frame(radar_frame(fresh, predicted, display=(predicted, fresh), risk_unresolved=True))
  before, serialized = native.to_dict(), native.to_bytes()
  native.clear_write_flag()
  consumer = RadarConsumption()
  view = consume_frame(consumer, native)
  assert not view.leadOne.present and view.leadTwo.present
  assert [lead.evidence.trackId for lead in control_leads(view)] == [5]
  assert native.to_dict() == before and native.to_bytes() == serialized
  assert len(native.radarObjects) == 2 and native.leadOne.present
  assert view.fusionRiskUnresolved and view.to_dict()['fusionRiskUnresolved']


def test_native_tick_snapshot_freezes_values_without_caching_qualification():
  lead = formal_lead(now_ns=PUBLICATION, source='vision', vRel=1.0,
                     evidence={'conflictEnd': .1})
  native = native_frame(radar_frame(lead, display=(lead,)))
  consumer = RadarConsumption()
  view = consume_frame(consumer, native, now=PUBLICATION)
  frozen = view.radarObjects[0]
  assert qualified_lead(frozen, 'brake', .95, now_ns=PUBLICATION)
  assert not qualified_lead(frozen, 'brake', 1.0, now_ns=PUBLICATION)
  assert not qualified_lead(frozen, 'caution', 0.0, now_ns=PUBLICATION)
  assert not qualified_lead(frozen, 'brake', .95, now_ns=PUBLICATION + 100_000_000)
  assert qualified_lead(frozen, 'brake', .95, now_ns=PUBLICATION + 100_000_000, require_active_window=False)
  assert not qualified_lead(frozen, 'brake', .95, now_ns=PUBLICATION + 200_000_001, require_active_window=False)
  with pytest.raises(AttributeError):
    frozen.dRel = 0.0
  with pytest.raises(AttributeError):
    frozen.evidence.modelMonoTime = PUBLICATION + 1

  native.radarObjects[0].dRel = native.leadOne.dRel = 40.0
  later = consume_frame(consumer, native, now=PUBLICATION)
  assert later.radarObjects[0].dRel == 40.0
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


@pytest.mark.parametrize('field', ['dRel', 'generation', 'positionMonoTime', 'physicalConflictValid'])
def test_native_duplicate_identity_conflicts_survive_snapshotting(field):
  predicted = coast()
  native = native_frame(radar_frame(predicted, display=(predicted,)))
  if field == 'dRel':
    native.leadOne.dRel += 1.0
  elif field == 'physicalConflictValid':
    native.leadOne.evidence.physicalConflictValid = True
  else:
    evidence = native.leadOne.evidence
    setattr(evidence, field, getattr(evidence, field) + 1)
  before = native.to_dict()
  consumer = RadarConsumption()
  view = consume_frame(consumer, native)
  assert consumer.expired_identities == () and not consumer.degraded
  assert len(view.radarObjects) == 1 and view.leadOne.present
  assert view.to_dict() == before and native.to_dict() == before


def test_same_native_publication_rechecked_at_a_later_consumption_time():
  predicted = coast()
  native = native_frame(radar_frame(predicted, display=(predicted,)))
  consumer = RadarConsumption()
  boundary = predicted.evidence.positionMonoTime + MAX_PREDICTION_AGE_NS
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
