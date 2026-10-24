"""Native measurement admission, independent motion, and source-time contracts."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.car.honda_radar_native import NativeMotion


START = 10_000_000_000
STEP = 70_000_000


def sample(index, *, step=STEP, relative=-1.0, distance=30.0, lateral=0.0, generation=1, slot=0):
  stamp = START + index * step
  u11 = round(relative * 64 + 864)
  return {
    'positionSeq': 100 * index + 1, 'positionMonoTime': stamp,
    'velocitySeq': 100 * index + 2, 'velocityMonoTime': stamp,
    'frameIndex': index % 16, 'frameMask': 31, 'generation': generation, 'slot': slot,
    'positionValid': True, 'velocityValid': True, 'dRel': distance, 'yRel': lateral,
    'vRel': (u11 - 864) / 64.0,
    'raw': {'statusRaw': 7, 'rangeRaw': round((distance + 3) / .05712),
            'azimuthRaw': round(math.atan2(lateral, distance) * 2048 + 1024),
            'rangeSigmaRaw': 20, 'existenceRaw': 100, 'lifecycleRaw': 1 + 2 * index,
            'oemTrackId': 7, 'u11Raw': u11, 'u10Raw': 20},
  }


def ego_at(_stamp):
  return SimpleNamespace(speed=20.0, yaw=0.0)


def observe(native, value, delay=1, ego=ego_at):
  return native.observe(value, max(value['positionMonoTime'], value['velocityMonoTime']) + delay, ego)


def mature():
  native = NativeMotion()
  for index in range(5):
    assert observe(native, sample(index))
  assert native.qualified
  return native


def test_native_velocity_jump_withdraws_instead_of_following_the_position_posterior():
  native = mature()
  assert not observe(native, sample(5, relative=-7.0, distance=29.0))
  assert native.reason == 'velocityInnovation'
  assert not native.qualified and native.sample_count == 0
  assert native.ground_speed == 0.0
  for index in range(6, 11):
    assert observe(native, sample(index, relative=-7.0, distance=29.0 - 7.0 * (index - 5) * STEP / 1e9))
  assert native.qualified and native.ground_speed == 13.0


def test_coherent_native_hard_braking_can_remain_qualified():
  native = NativeMotion()
  for index in range(12):
    elapsed = index * STEP / 1e9
    assert observe(native, sample(index, relative=-1.0 - 8.0 * elapsed, distance=30.0 - elapsed - 4.0 * elapsed**2))
    if index >= 4:
      assert native.qualified
  assert native.acceleration_valid and native.acceleration < -3.0


@pytest.mark.parametrize('step,count,qualified', [
  (50_000_000, 5, False), (50_000_000, 6, True),
  (62_499_999, 5, False), (62_500_000, 5, True),
  (100_000_000, 4, False), (100_000_000, 5, True),
])
def test_count_and_elapsed_source_span_are_both_required(step, count, qualified):
  native = NativeMotion()
  for index in range(count):
    assert observe(native, sample(index, step=step))
  assert native.sample_count == count
  assert native.qualified is qualified


def test_duplicate_publications_never_add_support_or_renew_source_expiry():
  native = mature()
  current = sample(4)
  before = native.snapshot(current['positionMonoTime'] + 1)
  for delay in (2, 10_000_000, 30_000_000, 90_000_000):
    assert not observe(native, current, delay)
    after = native.snapshot(current['positionMonoTime'] + delay)
    for key in ('sampleCount', 'firstPositionMonoTime', 'validUntilMonoTime', 'pairCompleteMonoTime',
                'positionMonoTime', 'velocityMonoTime', 'groundSpeed', 'acceleration'):
      assert after[key] == before[key]


def test_pending_aux_suspends_authority_but_completing_same_scan_counts_once():
  native = mature()
  complete = sample(5)
  position = copy.deepcopy(complete)
  position.update(frameMask=15, velocitySeq=0, velocityMonoTime=0, velocityValid=False)
  assert not observe(native, position)
  waiting = native.snapshot(position['positionMonoTime'] + 1)
  assert waiting['reason'] == 'awaitingAux'
  assert not waiting['qualified'] and waiting['sampleCount'] == 5
  assert waiting['velocityMonoTime'] == 0 and waiting['groundSpeed'] == 0
  complete['velocityMonoTime'] += 10_000_000
  assert observe(native, complete)
  assert native.qualified and native.sample_count == 6
  assert native.velocity_ns == complete['velocityMonoTime']
  assert not observe(native, complete, 2)
  assert native.sample_count == 6


def test_pending_timeout_clears_motion_and_cannot_be_undone_by_late_same_scan():
  native = mature()
  complete = sample(5)
  position = {**complete, 'frameMask': 15, 'velocitySeq': 0, 'velocityMonoTime': 0, 'velocityValid': False}
  observe(native, position)
  at_limit = native.snapshot(position['positionMonoTime'] + 50_000_000)
  assert at_limit['sampleCount'] == 5 and at_limit['reason'] == 'awaitingAux'
  expired = native.snapshot(position['positionMonoTime'] + 50_000_001)
  assert expired['sampleCount'] == 0 and expired['reason'] == 'pairTimeout'
  assert not observe(native, complete, 60_000_000)
  assert not native.qualified and native.sample_count == 0
  assert observe(native, sample(6))
  assert native.sample_count == 1 and not native.acceleration_valid


@pytest.mark.parametrize('u11', [0, 1728, 2047])
def test_invalid_velocity_with_zero_sequence_revokes_the_entire_current_pair(u11):
  native = mature()
  bad = sample(5)
  bad.update(velocitySeq=0, velocityValid=False, vRel=0.0)
  bad['raw']['u11Raw'] = u11
  assert not observe(native, bad)
  assert native.reason == 'invalidVelocity'
  assert not native.pair_good and not native.qualified and native.sample_count == 0
  assert native.ground_speed == 0 and not native.acceleration_valid
  assert not observe(native, sample(5), 2)
  assert observe(native, sample(6, relative=2.0))
  assert native.ground_speed == 22.0 and native.sample_count == 1
  assert native.acceleration == 0 and not native.acceleration_valid


@pytest.mark.parametrize('u10,valid', [(0, True), (255, True), (256, False), (511, False)])
def test_native_uncertainty_gate_is_strictly_below_256(u10, valid):
  native = NativeMotion()
  value = sample(0)
  value['raw']['u10Raw'] = u10
  assert observe(native, value) is valid


@pytest.mark.parametrize('field', ['statusRaw', 'rangeRaw', 'azimuthRaw', 'lifecycleRaw', 'oemTrackId',
                                  'rangeSigmaRaw', 'existenceRaw'])
def test_missing_native_distance_quality_never_defaults_to_good(field):
  native = mature()
  value = sample(5)
  del value['raw'][field]
  assert not observe(native, value)
  assert native.reason == 'invalidPosition' and native.sample_count == 0


@pytest.mark.parametrize('field,value', [('statusRaw', 15), ('rangeRaw', 4095), ('azimuthRaw', 2047),
                                       ('lifecycleRaw', 4095), ('rangeSigmaRaw', -1),
                                       ('rangeSigmaRaw', 128), ('existenceRaw', None), ('existenceRaw', 128)])
def test_raw_geometry_sentinels_and_invalid_quality_types_reject_the_pair(field, value):
  native = mature()
  current = sample(5)
  current['raw'][field] = value
  assert not observe(native, current)
  assert not native.qualified


@pytest.mark.parametrize('change,reason', [('timeGap', 'sourceGap'), ('scanGap', 'sourceGap'),
                                        ('generation', 'identityChanged'), ('missingPair', 'sourceGap')])
def test_missing_scans_identity_rebuilds_and_incomplete_scans_restart_support(change, reason):
  native = mature()
  current = sample(5)
  if change == 'timeGap':
    current['positionMonoTime'] += 31_000_000
    current['velocityMonoTime'] += 31_000_000
  elif change == 'scanGap':
    current['frameIndex'] = 6
  elif change == 'generation':
    current['generation'] += 1
  else:
    pending = {**current, 'frameMask': 15, 'velocitySeq': 0, 'velocityMonoTime': 0, 'velocityValid': False}
    observe(native, pending)
    current = sample(6)
  assert observe(native, current)
  assert native.reason == reason and native.sample_count == 1
  assert not native.qualified and not native.acceleration_valid


@pytest.mark.parametrize('skew,accepted', [(-50_000_001, False), (-50_000_000, True),
                                        (50_000_000, True), (50_000_001, False)])
def test_native_pairing_uses_absolute_original_source_time_difference(skew, accepted):
  native = NativeMotion()
  value = sample(0)
  value['velocityMonoTime'] += skew
  assert observe(native, value) is accepted


def test_aux_cannot_change_or_double_count_an_already_complete_scan():
  native = mature()
  value = sample(4)
  value['velocitySeq'] += 10
  value['velocityMonoTime'] += 1_000_000
  assert not observe(native, value)
  assert native.reason == 'conflictingVelocity'
  assert not native.qualified and native.sample_count == 0


def test_native_motion_depends_only_on_native_velocity_and_aligned_ego():
  reference, perturbed = NativeMotion(), NativeMotion()
  for index in range(12):
    # A real decrease in native speed remains eligible; there is no
    # nearly-constant-velocity qualification gate.
    original = sample(index, relative=2.0 - index * .25)
    altered = sample(index, relative=2.0 - index * .25, distance=30 + 3 * math.sin(index), lateral=.2 * index)
    observe(reference, original)
    observe(perturbed, altered)
    assert (reference.ground_speed, reference.acceleration, reference.acceleration_valid) == (
      perturbed.ground_speed, perturbed.acceleration, perturbed.acceleration_valid)
  # Native numerical motion remains independent. Deliberately contradictory
  # positions can separately revoke the current consistency qualification.
  assert reference.qualified
  assert reference.acceleration_valid and reference.acceleration < -2
  assert reference.distance_residual != perturbed.distance_residual


def test_measurement_and_ego_timestamps_align_without_position_or_lateral_rotation():
  native = NativeMotion()
  def ego(stamp):
    return SimpleNamespace(speed=10.0 + (stamp - START) / 1e9, yaw=.5)
  for index in range(6):
    current = sample(index, relative=-1.0)
    current['velocityMonoTime'] += 20_000_000
    assert observe(native, current, ego=ego)
    assert native.ground_speed == current['vRel'] + ego(current['velocityMonoTime']).speed
  assert native.qualified and native.acceleration_valid
  assert not native.distance_residual_valid  # Coordinates rotated beyond diagnostic scope.


def test_coasting_and_withdrawal_do_not_keep_prior_speed_for_requalification():
  native = mature()
  stamp = sample(4)['positionMonoTime']
  assert native.snapshot(stamp + 100_000_000)['qualified']
  assert not native.snapshot(stamp + 100_000_001)['qualified']
  assert native.reason == 'coasting' and native.sample_count == 0 and native.ground_speed == 0
  assert observe(native, sample(6))
  assert native.sample_count == 1 and not native.acceleration_valid
  native.invalidate('withdrawn')
  assert native.sample_count == 0 and not native.pair_good
  assert not observe(native, sample(6), 2)


def test_gross_current_position_conflict_cannot_be_hidden_as_one_trimmed_outlier():
  native = mature()
  current = sample(5, distance=24.0)
  assert observe(native, current)
  assert not native.qualified and native.distance_residual_valid
  assert native.reason == 'motionDisagreement'
  assert native.raw_relative_speed == -1.0 and native.ground_speed == 19.0
  assert native.distance_residual < -5


def test_formal_track_speed_and_acceleration_ignore_joint_distance_and_lateral_state():
  from openpilot.selfdrive.car.honda_radar_fusion import HondaRadarFusion
  from openpilot.selfdrive.car.tests.test_honda_radar_fusion import Inputs, START as FUSION_START, STEP as FUSION_STEP
  from openpilot.selfdrive.car.tests.test_honda_radar_fusion import advance, body, observation
  reference, perturbed = HondaRadarFusion(), HondaRadarFusion()
  for index in range(10):
    expected = advance(reference, index, [observation(index, relative=-1.0)], probability=0)['tracks'][0]
    # Quantization-scale geometry perturbations retain a qualified current
    # pair. Sustained contradictory position motion is covered separately.
    actual = advance(perturbed, index, [observation(index, relative=-1.0, distance=20 + .03 * math.sin(index),
                                                   lateral=.1 * math.sin(index))], probability=0)['tracks'][0]
    for field in ('groundSpeed', 'acceleration', 'accelerationGood'):
      assert expected[field] == actual[field]
  track = perturbed.tracks[7]
  track.mean[2], track.mean[3] = -30.0, 20.0
  stamp = FUSION_START + 9 * FUSION_STEP + 20_000_000
  current = perturbed.update(body(10, []), Inputs(stamp, yaw=.5, probability=0), stamp + 1)['tracks'][0]
  assert current['nativeMotion']['qualified']
  assert current['groundSpeed'] == current['nativeMotion']['groundSpeed'] == 9.0
  assert current['acceleration'] == 0


def test_confirmed_off_path_object_does_not_restart_native_confirmation_when_geometry_changes():
  from openpilot.selfdrive.car.honda_radar_fusion import HondaRadarFusion
  from openpilot.selfdrive.car.tests.test_honda_radar_fusion import advance, observation
  fusion = HondaRadarFusion()
  for index in range(6):
    out = advance(fusion, index, [observation(index, lateral=2.4)], probability=0)
  before = out['tracks'][0]['nativeMotion']
  assert before['qualified'] and out['tracks'][0]['pathState'] == 'outOfPath'
  for index, lateral in enumerate((2.1, 1.8, 1.5), 6):
    out = advance(fusion, index, [observation(index, lateral=lateral)], probability=0)
  after = out['tracks'][0]['nativeMotion']
  assert after['qualified'] and after['firstPositionMonoTime'] == before['firstPositionMonoTime']
  assert after['sampleCount'] == 9


def test_fusion_qualified_migration_retains_only_unbroken_native_pairs():
  from openpilot.selfdrive.car.honda_radar_fusion import HondaRadarFusion
  from openpilot.selfdrive.car.tests.test_honda_radar_fusion import advance, observation
  fusion = HondaRadarFusion()
  for index in range(5):
    out = advance(fusion, index)
  identifier = out['tracks'][0]['trackId']
  current = observation(5, slot=1)
  current['generation'] = 2  # A different decoder slot, same proven OEM identity.
  out = advance(fusion, 5, [current])
  assert out['tracks'][0]['trackId'] == identifier
  assert out['tracks'][0]['nativeMotion']['qualified']
  assert out['tracks'][0]['nativeMotion']['sampleCount'] == 6
  current = observation(6, slot=1)
  current['generation'] = 3  # Same-slot generation rebuild cannot borrow that history.
  out = advance(fusion, 6, [current])
  assert not out['tracks'][0]['nativeMotion']['qualified']
  assert out['tracks'][0]['nativeMotion']['sampleCount'] == 1


def _decoded_migration_fixture():
  from opendbc.car.honda.bosch_a_radar import BoschARadarObserver
  from opendbc.car.honda.tests.test_bosch_a_observer import scan_frames
  from openpilot.selfdrive.car.honda_radar_fusion import HondaRadarFusion
  from openpilot.selfdrive.car.tests.test_honda_radar_fusion import STEP as FUSION_STEP, advance
  decoder, fusion = BoschARadarObserver(2), HondaRadarFusion()
  for index in range(8):
    decoder.feed(START + index * FUSION_STEP,
                 scan_frames(slot=0, index=index, life=1 + 2 * index, distance=403,
                             azimuth=1024, velocity=885, oem_id=7))
    advance(fusion, index, decoder.drain(), probability=0)
  track = fusion.tracks[7]
  assert track.native_motion.qualified and track.native_motion.sample_count == 8
  return decoder, fusion, track


@pytest.mark.parametrize('takeover_first', [True, False])
def test_same_batch_decoded_old_slot_takeover_preserves_proved_current_migration(takeover_first):
  from opendbc.car.honda.tests.test_bosch_a_observer import scan_frames
  from openpilot.selfdrive.car.tests.test_honda_radar_fusion import STEP as FUSION_STEP, advance
  decoder, fusion, old = _decoded_migration_fixture()
  first = old.native_motion.first_position_ns
  takeover = scan_frames(slot=0, index=8, life=1, distance=1000, azimuth=1024, velocity=864, oem_id=40)
  migrated = scan_frames(slot=1, index=8, life=17, distance=403, azimuth=1024, velocity=885, oem_id=7)
  decoder.feed(START + 8 * FUSION_STEP, takeover + migrated if takeover_first else migrated + takeover)
  out = advance(fusion, 8, decoder.drain(), probability=0)
  target = next(track for track in out['tracks'] if track['oemTrackId'] == 7)
  assert fusion.tracks[7] is old and old.slot == 1
  assert target['nativeMotion']['qualified'] and target['nativeMotion']['sampleCount'] == 9
  assert target['nativeMotion']['firstPositionMonoTime'] == first
  assert target['nativeMotion']['rawRelativeSpeed'] == .328125
  assert target['groundSpeed'] == target['nativeMotion']['groundSpeed'] == 10.328125
  assert target['accelerationGood'] and target['acceleration'] == 0


@pytest.mark.parametrize('takeover_first', [True, False])
@pytest.mark.parametrize('fault', ['missingAux', 'invalidAux', 'uncertainAux', 'velocityInnovation',
                                 'conflictingFrame', 'invalidPosition', 'missedScan', 'sourceGap', 'newIdentity'])
def test_same_batch_migration_rejects_bad_pairs_and_bounds_pending_assembly(fault, takeover_first):
  from opendbc.can import CANPacker
  from opendbc.car.honda.bosch_a_radar import AUX_IDS
  from opendbc.car.honda.tests.test_bosch_a_observer import scan_frames
  from openpilot.selfdrive.car.tests.test_honda_radar_fusion import Inputs, STEP as FUSION_STEP, body
  decoder, fusion, old = _decoded_migration_fixture()
  stamp = START + 8 * FUSION_STEP
  index = 9 if fault == 'missedScan' else 8
  identity = 9 if fault == 'newIdentity' else 7
  if fault == 'sourceGap':
    stamp = old.position_ns + 100_000_001
  velocity = 0 if fault == 'invalidAux' else 1504 if fault == 'velocityInnovation' else 885
  takeover = scan_frames(slot=0, index=index, life=1, distance=1000, azimuth=1024, velocity=864, oem_id=40)
  migrated = scan_frames(slot=1, index=index, life=1 + 2 * index, distance=4095 if fault == 'invalidPosition' else 403,
                         azimuth=1024, velocity=velocity, oem_id=identity)
  auxiliary = migrated[-1]
  if fault == 'missingAux':
    migrated = migrated[:-1]
  elif fault == 'uncertainAux':
    migrated[-1] = CANPacker('honda_bosch_a_radar').make_can_msg(AUX_IDS[1], 2, {
      'FRAME_IDX': index, 'REL_VELOCITY_RAW': 885, 'REL_VELOCITY_UNCERTAINTY_RAW': 256,
    })
  elif fault == 'conflictingFrame':
    migrated += scan_frames(slot=1, index=index, life=1 + 2 * index, distance=404,
                             azimuth=1024, velocity=885, oem_id=7)[:1]
  decoder.feed(stamp, takeover + migrated if takeover_first else migrated + takeover)
  out = fusion.update(body(8, decoder.drain()), Inputs(stamp, probability=0), stamp + 1)
  target = fusion.tracks.get(identity)
  direct_pending = fault == 'missingAux' and not takeover_first
  if target is not None:
    assert not target.native_motion.qualified
    if direct_pending:
      assert target.native_motion.sample_count == 8
    else:
      assert target.native_motion.sample_count <= 1
  assert not any(track['nativeMotion']['qualified'] for track in out['tracks'])
  if fault == 'missingAux':
    # An explicitly suspended target cannot recover discarded history in a
    # later batch. A direct position-first migration had no withdrawal and
    # retains the existing <=50 ms ordinary pending-assembly contract.
    decoder.feed(stamp + 10_000_000, [auxiliary])
    out = fusion.update(body(9, decoder.drain()), Inputs(stamp + 10_000_000, probability=0), stamp + 10_000_001)
    assert fusion.tracks[7].native_motion.sample_count == (9 if direct_pending else 1)
    assert fusion.tracks[7].native_motion.qualified is direct_pending


def test_direct_migrated_position_can_wait_for_timely_aux_without_an_explicit_withdrawal():
  from opendbc.car.honda.tests.test_bosch_a_observer import scan_frames
  from openpilot.selfdrive.car.tests.test_honda_radar_fusion import Inputs, STEP as FUSION_STEP, body
  decoder, fusion, old = _decoded_migration_fixture()
  first = old.native_motion.first_position_ns
  stamp = START + 8 * FUSION_STEP
  migrated = scan_frames(slot=1, index=8, life=17, distance=403, azimuth=1024, velocity=885, oem_id=7)
  decoder.feed(stamp, migrated[:-1])
  fusion.update(body(8, decoder.drain()), Inputs(stamp, probability=0), stamp + 1)
  assert old.active and not old.velocity_good and not old.native_motion.qualified
  assert old.native_motion.pending and old.native_motion.sample_count == 8
  assert old.native_motion.first_position_ns == first
  decoder.feed(stamp + 10_000_000, migrated[-1:])
  fusion.update(body(9, decoder.drain()), Inputs(stamp + 10_000_000, probability=0), stamp + 10_000_001)
  assert old.native_motion.qualified and old.native_motion.sample_count == 9
  assert old.native_motion.first_position_ns == first and old.native_motion.ground_speed == 10.328125


def test_cross_publication_decoded_slot_withdrawal_never_carries_native_confirmation():
  from opendbc.car.honda.tests.test_bosch_a_observer import scan_frames
  from openpilot.selfdrive.car.tests.test_honda_radar_fusion import Inputs, STEP as FUSION_STEP, body
  decoder, fusion, old = _decoded_migration_fixture()
  stamp = START + 8 * FUSION_STEP
  decoder.feed(stamp, scan_frames(slot=0, index=8, life=1, distance=1000, azimuth=1024, velocity=864, oem_id=40))
  fusion.update(body(8, decoder.drain()), Inputs(stamp, probability=0), stamp + 1)
  assert not old.active and old.native_motion.sample_count == 0
  decoder.feed(stamp + 10_000_000,
               scan_frames(slot=1, index=8, life=17, distance=403, azimuth=1024, velocity=885, oem_id=7))
  fusion.update(body(9, decoder.drain()), Inputs(stamp + 10_000_000, probability=0), stamp + 10_000_001)
  assert fusion.tracks[7] is old and old.active and old.velocity_good
  assert old.native_motion.sample_count == 1 and not old.native_motion.qualified
  assert not old.native_motion.acceleration_valid


def test_same_batch_migration_cannot_undo_an_earlier_published_coasting_timeout():
  from opendbc.car.honda.tests.test_bosch_a_observer import scan_frames
  from openpilot.selfdrive.car.tests.test_honda_radar_fusion import Inputs, STEP as FUSION_STEP, body
  decoder, fusion, old = _decoded_migration_fixture()
  expired_at = old.position_ns + 100_000_001
  fusion.update(body(8, []), Inputs(expired_at, probability=0), expired_at)
  assert old.native_motion.reason == 'coasting' and old.native_motion.sample_count == 0
  # A later received batch can contain a still-fresh source scan. Its arrival
  # cannot undo the already published timeout or recover the discarded history.
  stamp = START + 8 * FUSION_STEP
  takeover = scan_frames(slot=0, index=8, life=1, distance=1000, azimuth=1024, velocity=864, oem_id=40)
  migrated = scan_frames(slot=1, index=8, life=17, distance=403, azimuth=1024, velocity=885, oem_id=7)
  decoder.feed(stamp, takeover + migrated)
  fusion.update(body(9, decoder.drain()), Inputs(expired_at + 1_000_000, probability=0), expired_at + 1_000_000)
  assert old.active and old.velocity_good
  assert old.native_motion.sample_count == 1 and not old.native_motion.qualified


@pytest.mark.parametrize('reason', ['withdrawn', 'invalidGeometry', 'conflictingFrame'])
def test_same_batch_real_invalidation_discards_saved_migration_support(reason):
  from openpilot.selfdrive.car.honda_radar_fusion import HondaRadarFusion
  from openpilot.selfdrive.car.tests.test_honda_radar_fusion import Inputs, STEP as FUSION_STEP, advance, body, observation
  fusion = HondaRadarFusion()
  for frame in range(8):
    advance(fusion, frame, probability=0)
  old = fusion.tracks[7]
  takeover = observation(8, oem=40, distance=60.0)
  invalid = {'slot': 0, 'positionValid': False, 'positionMonoTime': takeover['positionMonoTime'] + 1,
             'reason': reason, 'raw': {'oemTrackId': 7}}
  migrated = observation(8, slot=1)
  migrated['positionMonoTime'] += 2
  migrated['velocityMonoTime'] += 2
  stamp = START + 8 * FUSION_STEP + 3
  out = fusion.update(body(8, [takeover, invalid, migrated]), Inputs(stamp, probability=0), stamp)
  assert old.native_motion.sample_count == 0
  assert fusion.tracks[7].native_motion.sample_count <= 1
  assert not any(track['nativeMotion']['qualified'] for track in out['tracks'])


def test_saturation_identity_recovery_cannot_inherit_old_motion_or_pair_confirmation():
  from openpilot.selfdrive.car.tests.test_honda_radar_fusion import advance, saturation_fixture
  fusion, sentinel, returned = saturation_fixture()
  identifier = fusion.tracks[7].identifier
  assert fusion.tracks[7].native_motion.qualified
  advance(fusion, 8, [sentinel])
  returned['vRel'], returned['raw']['u11Raw'] = 12.0, 1632
  out = advance(fusion, 9, [returned])
  track = out['tracks'][0]
  assert track['trackId'] == identifier  # Existing geometric identity exception.
  assert track['velocityGood']
  assert track['groundSpeed'] == 22.0  # No distance-posterior residual/velocity repair.
  assert track['nativeMotion']['sampleCount'] == 1
  assert not track['nativeMotion']['qualified'] and not track['accelerationGood']


def test_append_only_native_schema_roundtrip_and_old_wire_default_is_unqualified():
  import capnp
  from openpilot.cereal import custom
  native = mature()
  state = native.snapshot(sample(4)['positionMonoTime'] + 1)
  message = custom.HondaRadarObservations.new_message(fusion={'tracks': [{'nativeMotion': state}]})
  with custom.HondaRadarObservations.from_bytes(message.to_bytes()) as reader:
    assert reader.fusion.tracks[0].nativeMotion.to_dict() == state
  parser = capnp.SchemaParser()
  old = parser.load(str(Path(__file__).with_name('fixtures') / 'honda_radar_observations_86f3dc03.capnp'))
  legacy = old.HondaRadarObservations.new_message(fusion={'tracks': [{'velocityGood': True, 'mature': True}]})
  with custom.HondaRadarObservations.from_bytes(legacy.to_bytes()) as reader:
    assert reader.fusion.tracks[0].velocityGood
    assert reader.fusion.tracks[0].nativeMotion.version == 0
    assert not reader.fusion.tracks[0].nativeMotion.qualified
