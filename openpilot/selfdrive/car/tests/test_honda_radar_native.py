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
    assert observe(native, sample(index, relative=-7.0))
  assert native.qualified and native.ground_speed == 13.0


def test_coherent_native_hard_braking_can_remain_qualified():
  native = NativeMotion()
  for index in range(12):
    assert observe(native, sample(index, relative=-1.0 - 8.0 * index * STEP / 1e9))
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
  assert reference.qualified and perturbed.qualified
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


def test_uncalibrated_distance_residual_remains_diagnostic_only():
  native = mature()
  current = sample(5, distance=24.0)
  assert observe(native, current)
  assert native.qualified and native.distance_residual_valid
  assert native.distance_residual < -5


def test_formal_track_speed_and_acceleration_ignore_joint_distance_and_lateral_state():
  from openpilot.selfdrive.car.honda_radar_fusion import HondaRadarFusion
  from openpilot.selfdrive.car.tests.test_honda_radar_fusion import Inputs, START as FUSION_START, STEP as FUSION_STEP
  from openpilot.selfdrive.car.tests.test_honda_radar_fusion import advance, body, observation
  reference, perturbed = HondaRadarFusion(), HondaRadarFusion()
  for index in range(10):
    expected = advance(reference, index, [observation(index, relative=-1.0)], probability=0)['tracks'][0]
    actual = advance(perturbed, index, [observation(index, relative=-1.0, distance=20 + .3 * math.sin(index),
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
