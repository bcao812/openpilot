from copy import deepcopy

from openpilot.selfdrive.ui.sunnypilot.onroad.honda_radar_state import RadarOverlayState


NOW = 1_000_000_000


def observations():
  sample = {'slot': 2, 'generation': 7, 'positionSeq': 18, 'positionMonoTime': NOW,
            'positionValid': True, 'dRel': 20.0, 'yRel': 0.0}
  return {
    'schemaVersion': 1, 'active': True, 'sessionId': 'radar-instance', 'sequence': 1,
    'scope': dict.fromkeys(('alphaLong', 'experimentalSetting', 'decSetting',
                           'experimentalActive', 'decEnabled', 'decActive'), True),
    'observations': [sample],
    'fusion': {'version': 1, 'active': True, 'tracks': [dict(sample, trackId=42)]},
  }


def test_radar_labels_follow_exact_consumed_sample_identity():
  body = observations()
  # A second car at the same projected point must keep its own identity.
  other = dict(body['observations'][0], slot=3)
  body['observations'].append(other)
  body['fusion']['tracks'].append(dict(other, trackId=63))
  overlay = RadarOverlayState()
  overlay.ingest(body, NOW + 1, NOW + 1)
  points = overlay.points(NOW, NOW + 2)
  assert [(p.session_id, p.track_id) for p in points] == [('radar-instance', 42), ('radar-instance', 63)]


def test_raw_slot_reuse_cannot_inherit_previous_control_target():
  body = observations()
  overlay = RadarOverlayState()
  overlay.ingest(body, NOW + 1, NOW + 1)
  replacement = dict(body['observations'][0], generation=8, positionSeq=19, positionMonoTime=NOW + 10)
  body.update(sequence=2, observations=[replacement])
  # The fusion publication still describes the old slot occupant.
  overlay.ingest(body, NOW + 11, NOW + 11)
  point = overlay.points(NOW + 10, NOW + 12)[0]
  assert point.track_id == 0
  assert point.generation == 8


def test_late_fusion_identity_can_bind_the_same_cached_raw_sample():
  body = observations()
  unbound = deepcopy(body)
  unbound['fusion']['tracks'] = []
  overlay = RadarOverlayState()
  overlay.ingest(unbound, NOW + 1, NOW + 1)
  assert overlay.points(NOW, NOW + 1)[0].track_id == 0
  body.update(sequence=2, observations=[])
  overlay.ingest(body, NOW + 2, NOW + 2)
  assert overlay.points(NOW, NOW + 2)[0].track_id == 42


def test_conflicting_track_mapping_cannot_restore_color_until_new_measurement():
  body = observations()
  overlay = RadarOverlayState()
  overlay.ingest(body, NOW + 1, NOW + 1)
  body.update(sequence=2, observations=[])
  body['fusion']['tracks'][0]['trackId'] = 43
  overlay.ingest(body, NOW + 2, NOW + 2)
  assert overlay.points(NOW, NOW + 2)[0].track_id == -1
  body['sequence'] = 3
  overlay.ingest(body, NOW + 3, NOW + 3)
  assert overlay.points(NOW, NOW + 3)[0].track_id == -1


def test_restarted_observer_cannot_borrow_previous_session_identity():
  body = observations()
  overlay = RadarOverlayState()
  overlay.ingest(body, NOW + 1, NOW + 1)
  body.update(sessionId='restarted', sequence=1)
  overlay.ingest(body, NOW + 2, NOW + 2)
  point = overlay.points(NOW, NOW + 2)[0]
  assert (point.session_id, point.track_id) == ('restarted', 42)


def test_duplicate_conflicting_tracks_and_wrong_measurement_epochs_are_unbound():
  for changes in ({'positionSeq': 17}, {'positionMonoTime': NOW - 1}, {'generation': 6}, {'slot': 1}):
    body = observations()
    body['fusion']['tracks'][0].update(changes)
    overlay = RadarOverlayState()
    overlay.ingest(body, NOW + 1, NOW + 1)
    assert overlay.points(NOW, NOW + 1)[0].track_id == 0
  body = observations()
  body['fusion']['tracks'].append(dict(body['fusion']['tracks'][0], trackId=43))
  overlay = RadarOverlayState()
  overlay.ingest(body, NOW + 1, NOW + 1)
  assert overlay.points(NOW, NOW + 1)[0].track_id == -1
