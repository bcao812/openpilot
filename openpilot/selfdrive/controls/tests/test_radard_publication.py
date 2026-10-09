"""RadarD finalizes measurement age at the timestamp of its actual Event."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from opendbc.car.structs import car
from openpilot.cereal import messaging
from openpilot.selfdrive.car.honda_radar_observer_config import SERVICE
from openpilot.selfdrive.controls.radard import RadarD
from openpilot.selfdrive.controls.tests.test_lead_qualification import formal_lead
from openpilot.selfdrive.controls.tests.test_radar_consumption import (
  CONSUMPTION, PUBLICATION, SESSION, coast, measured, native_frame, radar_frame,
)


class Inputs(dict):
  def __init__(self, now):
    super().__init__((service, getattr(messaging.new_message(service), service))
                     for service in ('modelV2', 'carState', SERVICE))
    self['modelV2'].timestampEof = now
    self['carState'].vEgo = 10.0
    self['radarTracks'] = car.RadarData.new_message()
    self.seen = dict.fromkeys(self, True)
    self.valid = dict.fromkeys(self, True)
    self.recv_frame = dict.fromkeys(self, now)
    self.logMonoTime = dict.fromkeys(self, now)

  def all_checks(self, service_list=None):
    return all(self.seen[service] and self.valid[service] for service in (service_list or self))


def update_with_result(rd, frame, now=PUBLICATION, *, enabled=True, scope_missing=False):
  payload = native_frame(frame).to_dict()
  rd.fusion_control = SimpleNamespace(update=Mock(return_value=SimpleNamespace(
    leads=(payload['leadOne'], payload['leadTwo']), objects=payload['radarObjects'], healthy=True,
    degraded=False, risk_unresolved=False, reason='active', cleared=frame.fusionClearedTracks, session_id=SESSION,
  )), _unresolved_loss=False, _lost_tracks={}, _session=SESSION)
  rd.control_enabled = Mock(return_value=enabled)
  rd.control_scope_unavailable = scope_missing
  inputs = Inputs(now)
  rd.update(inputs, inputs['radarTracks'], now_ns=now)


def publish_at(rd, now):
  original = messaging.new_message
  captured = []

  def dated(service, *args, **kwargs):
    return original(service, *args, **kwargs, logMonoTime=now)

  publisher = SimpleNamespace(send=lambda service, event: captured.append(messaging.log_from_bytes(event.to_bytes())))
  with patch.object(messaging, 'new_message', new=dated):
    rd.publish(publisher)
  assert len(captured) == 1
  return captured[0]


def make_radard():
  return RadarD(SimpleNamespace(), SimpleNamespace(), params=SimpleNamespace())


@pytest.mark.parametrize('source,caution', [
  ('radar', False), ('fused', False), ('predicted', False), ('radar', True), ('fused', True),
])
@pytest.mark.parametrize('after_deadline', [0, 1])
def test_publisher_removes_only_naturally_expired_measurement_at_actual_event_time(source, caution, after_deadline):
  old = coast() if source == 'predicted' else measured(source=source, caution=caution)
  fresh = formal_lead(now_ns=PUBLICATION, evidence={'trackId': 9})
  frame = radar_frame(old, fresh, display=(old, fresh))
  rd = make_radard()
  update_with_result(rd, frame)
  before = rd.radar_state.to_dict()
  actual_publication = PUBLICATION + 50_000_000 + after_deadline
  event = publish_at(rd, actual_publication)
  assert rd.radar_qualified_ns == PUBLICATION and event.logMonoTime == actual_publication
  assert event.valid and event.radarState.leadTwo.evidence.trackId == 9
  assert event.radarState.leadOne.present is (after_deadline == 0)
  expected_tracks = [9] if after_deadline else [7, 9]
  assert [lead.evidence.trackId for lead in event.radarState.radarObjects] == expected_tracks
  assert event.radarState.leadOne.evidence.to_dict() == before['leadOne']['evidence']
  assert not event.radarState.fusionClearedTracks and not event.radarState.fusionRiskUnresolved
  assert rd.radar_state.to_dict() == before


@pytest.mark.parametrize('source', ['radar', 'fused', 'predicted'])
@pytest.mark.parametrize('defect', [
  'old_at_qualification', 'old_model_at_publication', 'different_display', 'invalid_position', 'future_measurement',
  'support_version', 'support_nonfinite', 'support_outside_window',
])
def test_publication_expiry_cannot_erase_malformed_or_previously_stale_objects(source, defect):
  old = coast() if source == 'predicted' else measured(source=source)
  frame = radar_frame(old, display=(old,))
  maximum_age = 350_000_000 if source == 'predicted' else 200_000_000
  if defect == 'old_at_qualification':
    old.evidence.positionMonoTime = PUBLICATION - maximum_age - 1
  elif defect == 'old_model_at_publication':
    old.evidence.modelMonoTime = PUBLICATION - maximum_age + 50_000_000
  elif defect == 'different_display':
    from copy import deepcopy
    frame.leadOne = deepcopy(old)
    frame.leadOne.dRel += 1.0
  elif defect == 'invalid_position':
    old.evidence.positionValid = False
  elif defect.startswith('support_'):
    old.evidence.conflictSupportVersion = 2 if defect == 'support_version' else 1
    old.evidence.conflictSupportedEnd = float('nan') if defect == 'support_nonfinite' else 11.0
  else:
    old.evidence.velocityMonoTime = PUBLICATION + 1
  rd = make_radard()
  update_with_result(rd, frame)
  event = publish_at(rd, CONSUMPTION)
  assert event.logMonoTime == CONSUMPTION
  assert event.radarState.leadOne.present and len(event.radarState.radarObjects) == 1
  assert not rd.publication_expiry.expired_identities


@pytest.mark.parametrize('source', ['radar', 'predicted'])
def test_publication_loss_survives_scope_gap_and_empty_messages_until_fresh_same_identity(source):
  rd = make_radard()
  old = coast(relevant=True) if source == 'predicted' else measured(relevant=True)
  update_with_result(rd, radar_frame(old, display=(old,)))
  event = publish_at(rd, CONSUMPTION)
  assert event.radarState.leadOne.present and event.radarState.fusionRiskUnresolved
  assert not rd.publication_expiry.expired_identities
  assert event.radarState.fusionDegraded and not event.radarState.fusionClearedTracks
  assert rd.publication_expiry.lost_identities == ((SESSION, 7),)
  now = CONSUMPTION + 50_000_000
  update_with_result(rd, radar_frame(publication=now), now, enabled=False, scope_missing=True)
  gap = publish_at(rd, now + 1)
  assert not gap.radarState.fusionActive and gap.radarState.fusionRiskUnresolved
  now += 50_000_000
  update_with_result(rd, radar_frame(publication=now), now)
  empty = publish_at(rd, now + 1)
  assert empty.radarState.fusionRiskUnresolved and not empty.radarState.fusionClearedTracks
  now += 50_000_000
  fresh = formal_lead(now_ns=now)
  update_with_result(rd, radar_frame(fresh, publication=now, display=(fresh,)), now)
  recovered = publish_at(rd, now + 1)
  assert not recovered.radarState.fusionRiskUnresolved and not rd.publication_expiry.degraded


def test_explicit_configuration_exit_clears_only_the_publication_ledger_lifecycle():
  rd = make_radard()
  old = coast(relevant=True)
  update_with_result(rd, radar_frame(old, display=(old,)))
  publish_at(rd, CONSUMPTION)
  assert rd.publication_expiry.degraded
  now = CONSUMPTION + 50_000_000
  update_with_result(rd, radar_frame(publication=now), now, enabled=False)
  event = publish_at(rd, now + 1)
  assert not rd.publication_expiry.degraded and not event.radarState.fusionRiskUnresolved
