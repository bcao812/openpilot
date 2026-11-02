from copy import deepcopy

import pytest

from openpilot.selfdrive.car.honda_radar_native import NativeMotion
from openpilot.selfdrive.car.tests.test_honda_radar_native import ego_at, sample as native_sample
from openpilot.selfdrive.ui.sunnypilot.onroad.honda_radar_state import RadarOverlayState
from openpilot.selfdrive.ui.tests.test_radar_label_identity import NOW, observations, qualified_track


@pytest.mark.parametrize('step,expected', (
  (50_000_000, (False, False, False, False, False, True)),
  (62_500_000, (False, False, False, False, True)),
  (100_000_000, (False, False, False, False, True)),
))
def test_color_qualification_follows_real_native_pair_count_and_time_gate(step, expected):
  native, overlay = NativeMotion(), RadarOverlayState()
  for index, qualified in enumerate(expected):
    event = native_sample(index, step=step, relative=0.0)
    now = event['positionMonoTime'] + 1
    native.observe(event, now, ego_at)
    track = qualified_track(event)
    track['nativeMotion'] = native.snapshot(now)
    body = observations()
    body.update(sequence=index + 1, observations=[event])
    body['fusion']['tracks'] = [track]
    overlay.ingest(body, now, now)
    point, = overlay.points(event['positionMonoTime'], now)
    assert overlay.is_qualified(point, now) is qualified


def test_late_aux_and_withdrawal_update_the_same_cached_measurement():
  body = observations(qualified=True)
  track = body['fusion']['tracks'][0]
  track['nativeMotion'].update(qualified=False, reason='awaitingAux')
  overlay = RadarOverlayState()
  overlay.ingest(body, NOW + 1, NOW + 1)
  point, = overlay.points(NOW, NOW + 1)
  assert not overlay.is_qualified(point, NOW + 1)

  body.update(sequence=2, observations=[])
  track['nativeMotion'].update(qualified=True, reason='qualified')
  overlay.ingest(body, NOW + 2, NOW + 2)
  assert overlay.points(NOW, NOW + 2)[0] is point
  assert overlay.is_qualified(point, NOW + 2)

  body['sequence'] = 3
  track['nativeMotion'].update(qualified=False, reason='motionDisagreement')
  overlay.ingest(body, NOW + 3, NOW + 3)
  assert overlay.points(NOW, NOW + 3)[0] is point
  assert not overlay.is_qualified(point, NOW + 3)


def test_qualification_expires_at_original_coasting_deadline_without_a_new_publication():
  body, overlay = observations(qualified=True), RadarOverlayState()
  overlay.ingest(body, NOW + 1, NOW + 1)
  point, = overlay.points(NOW, NOW + 1)
  body.update(sequence=2, observations=[])
  overlay.ingest(body, NOW + 90_000_000, NOW + 90_000_000)
  assert overlay.is_qualified(point, NOW + 100_000_000)
  assert not overlay.is_qualified(point, NOW + 100_000_001)
  assert overlay.points(NOW, NOW + 100_000_001) == [point]


def test_camera_lag_preserves_qualified_samples_without_renewing_their_deadlines():
  body, overlay = observations(qualified=True), RadarOverlayState()
  overlay.ingest(body, NOW + 1, NOW + 1)
  first, = overlay.points(NOW, NOW + 1)
  event = dict(body['observations'][0], positionSeq=19, positionMonoTime=NOW + 50_000_000)
  body.update(sequence=2, observations=[event])
  body['fusion']['tracks'] = [qualified_track(event)]
  overlay.ingest(body, NOW + 50_000_001, NOW + 50_000_001)
  displayed, = overlay.points(NOW + 20_000_000, NOW + 50_000_001)
  second, = overlay.points(NOW + 50_000_000, NOW + 50_000_001)
  assert displayed is first
  assert overlay.is_qualified(displayed, NOW + 50_000_001)
  assert overlay.is_qualified(second, NOW + 50_000_001)
  assert not overlay.is_qualified(first, NOW + 100_000_001)
  assert overlay.is_qualified(second, NOW + 100_000_001)
  body.update(sequence=3, observations=[])
  body['fusion']['tracks'][0]['nativeMotion']['qualified'] = False
  overlay.ingest(body, NOW + 50_000_002, NOW + 50_000_002)
  assert not overlay.is_qualified(first, NOW + 50_000_002)
  assert not overlay.is_qualified(second, NOW + 50_000_002)


def test_new_qualified_scan_cannot_retroactively_qualify_an_earlier_scan():
  body, overlay = observations(), RadarOverlayState()
  overlay.ingest(body, NOW + 1, NOW + 1)
  first, = overlay.points(NOW, NOW + 1)
  event = dict(body['observations'][0], positionSeq=19, positionMonoTime=NOW + 50_000_000)
  body.update(sequence=2, observations=[event])
  body['fusion']['tracks'] = [qualified_track(event)]
  overlay.ingest(body, NOW + 50_000_001, NOW + 50_000_001)
  assert not overlay.is_qualified(first, NOW + 50_000_001)
  second, = overlay.points(NOW + 50_000_000, NOW + 50_000_001)
  assert overlay.is_qualified(second, NOW + 50_000_001)


@pytest.mark.parametrize('change', ('missing', 'inactive', 'coasting', 'velocity_pending', 'new_sample', 'invalid_fusion'))
def test_latest_snapshot_immediately_withdraws_previous_qualification(change):
  body, overlay = observations(qualified=True), RadarOverlayState()
  overlay.ingest(body, NOW + 1, NOW + 1)
  point, = overlay.points(NOW, NOW + 1)
  assert overlay.is_qualified(point, NOW + 1)
  body.update(sequence=2, observations=[])
  track = body['fusion']['tracks'][0]
  if change == 'missing':
    body['fusion']['tracks'] = []
  elif change == 'inactive':
    track['active'] = False
  elif change == 'coasting':
    track['coasting'] = True
  elif change == 'velocity_pending':
    track['velocityGood'] = False
  elif change == 'new_sample':
    track['positionSeq'] += 1
    track['positionMonoTime'] += 1
  else:
    body['fusion']['active'] = False
  overlay.ingest(body, NOW + 2, NOW + 2)
  assert not overlay.is_qualified(point, NOW + 2)


@pytest.mark.parametrize('changes', (
  {'version': 1}, {'qualified': False}, {'reason': 'awaitingAux'},
  {'positionSeq': 17}, {'positionMonoTime': NOW - 1},
  {'velocityMonoTime': NOW - 1}, {'pairCompleteMonoTime': NOW + 2},
  {'validUntilMonoTime': NOW + 200_000_001},
))
def test_native_qualification_must_belong_to_this_sample_and_keep_its_source_deadline(changes):
  body, overlay = observations(qualified=True), RadarOverlayState()
  body['fusion']['tracks'][0]['nativeMotion'].update(changes)
  overlay.ingest(body, NOW + 1, NOW + 1)
  point, = overlay.points(NOW, NOW + 1)
  assert not overlay.is_qualified(point, NOW + 1)


@pytest.mark.parametrize('reverse', (False, True))
def test_duplicate_qualified_claim_cannot_override_a_withdrawal(reverse):
  body, overlay = observations(qualified=True), RadarOverlayState()
  withdrawn = deepcopy(body['fusion']['tracks'][0])
  withdrawn['nativeMotion']['qualified'] = False
  body['fusion']['tracks'].append(withdrawn)
  if reverse:
    body['fusion']['tracks'].reverse()
  overlay.ingest(body, NOW + 1, NOW + 1)
  point, = overlay.points(NOW, NOW + 1)
  assert point.track_id == 42
  assert not overlay.is_qualified(point, NOW + 1)
