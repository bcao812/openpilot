"""Shadow fusion contracts, including the 2026-10-05 road-log counterexamples."""
import copy
from collections import Counter
import math
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from opendbc.car.honda.bosch_a_radar import BoschARadarObserver
from opendbc.car.honda.tests.test_bosch_a_observer import scan_frames
from openpilot.selfdrive.car.honda_radar_fusion import Binding, Ego, EgoHistory, HondaRadarFusion, MAX_TRACKS, Track, _finite_median, finite, path_relation, read

START = 10_000_000_000
STEP = 66_000_000


class Inputs(dict):
  def __init__(self, t, distance=20.0, lateral=0.0, relative=0.0, speed=10.0, probability=.99, stopped_path=False, yaw=0.0,
               x_std=1.0, y_std=.4, v_std=1.0):
    lead = {'prob': probability, 'x': [distance + 1.52], 'y': [-lateral], 'v': [speed + relative], 'a': [0.0],
                'xStd': [x_std], 'yStd': [y_std], 'vStd': [v_std]}
    model = NS(timestampEof=t, leadsV3=[NS(**copy.deepcopy(lead)) for _ in range(3)],
               velocity=NS(x=[speed]), orientationRate=NS(z=[-yaw]),
               position=NS(x=[0.0] * 33 if stopped_path else list(np.linspace(0, 160, 33)), y=[0.0] * 33))
    super().__init__(carState=NS(vEgo=speed), modelV2=model)
    self.seen = dict.fromkeys(self, True)
    self.valid = dict.fromkeys(self, True)
    self.logMonoTime = dict.fromkeys(self, t)


def observation(frame, distance=20.0, lateral=0.0, relative=0.0, u10=20, slot=0, oem=7, velocity=True):
  t = START + frame * STEP
  seq = frame * 100 + 2 * slot + 1
  return {'slot': slot, 'frameIndex': frame % 16, 'frameMask': 31 if velocity else 15, 'generation': 1,
              'positionSeq': seq, 'velocitySeq': seq + 1 if velocity else 0, 'positionMonoTime': t, 'velocityMonoTime': t if velocity else 0,
              'dRel': distance, 'yRel': lateral, 'vRel': relative, 'positionValid': True, 'velocityValid': velocity,
              'reason': 'uncalibrated' if velocity else 'positionOnly',
              'raw': {'oemTrackId': oem, 'lifecycleRaw': 1 + 2 * frame, 'u10Raw': u10, 'u11Raw': round(relative * 64 + 864)}}


def source_observation(timestamp, sequence, index, lifecycle, generation=1, **kwargs):
  sample = observation(0, **kwargs)
  sample.update(positionMonoTime=timestamp, positionSeq=sequence, frameIndex=index, generation=generation)
  if sample['velocityValid']:
    sample.update(velocityMonoTime=timestamp, velocitySeq=sequence + 1)
  sample['raw']['lifecycleRaw'] = lifecycle
  return sample


def body(frame, observations, **overrides):
  value = {'schemaVersion': 1, 'sessionId': 'fusion-test', 'sequence': frame + 1, 'active': True, 'observations': observations,
               'scope': dict.fromkeys(('alphaLong', 'experimentalSetting', 'decSetting', 'experimentalActive', 'decEnabled', 'decActive'), True),
               'counters': {'dropped': 0}}
  value.update(overrides)
  return value


def advance(fusion, frame, observations=None, **kwargs):
  t = START + frame * STEP
  sm = Inputs(t, **kwargs)
  packet = body(frame, observations if observations is not None else [observation(frame)])
  return fusion.update(packet, sm, t + 1)


def projection_fixture():
  track = Track(identifier=1, oem_id=7, slot=0, frame_index=0, lifecycle=1, position_seq=1,
                position_ns=START, first_ns=START, epoch_ns=START,
                mean=np.array([20.0, 1.0, 12.0, -.3]),
                covariance=np.array([[2.0, .1, .3, 0.0], [.1, 1.0, 0.0, .2],
                                     [.3, 0.0, 4.0, .1], [0.0, .2, .1, 2.0]]))
  history = EgoHistory()
  history.add(Ego(START - 100_000_000, 8.0, -.1))
  history.add(Ego(START, 10.0, .2))
  return track, history


def saturation_fixture():
  fusion = HondaRadarFusion()
  for frame in range(8):
    sample = observation(frame)
    sample['raw'].update(lifecycleRaw=4079 + 2 * frame, statusRaw=7, rangeRaw=403, azimuthRaw=1024)
    advance(fusion, frame, [sample])
  sentinel = observation(8)
  sentinel.update(positionValid=False, velocityValid=False, positionSeq=0, velocitySeq=0, generation=0,
                  dRel=0.0, yRel=0.0, vRel=0.0, reason='invalidGeometry')
  sentinel['raw'].update(lifecycleRaw=4095, statusRaw=7, rangeRaw=403, azimuthRaw=1024)
  returned = observation(9)
  returned['generation'] = 2
  returned['raw'].update(lifecycleRaw=4094, statusRaw=7, rangeRaw=403, azimuthRaw=1024)
  return fusion, sentinel, returned


def position_only(sample):
  result = copy.deepcopy(sample)
  result.update(frameMask=15, velocitySeq=0, velocityMonoTime=0, velocityValid=False, vRel=0.0)
  result['raw'].update(u10Raw=0, u11Raw=0)
  if result['positionValid']:
    result['reason'] = 'positionOnly'
  return result


class TestHondaRadarFusion(unittest.TestCase):
  def test_native_reads_preserve_scalars_defaults_properties_and_errors(self):
    from openpilot.cereal import custom
    native = custom.HondaRadarObservations.new_message()
    native.observations = [observation(0)]
    native.observations[0].yRel = -0.0
    marker = object()
    for root in (native, native.as_reader()):
      target = root.observations[0]
      groups = ((root, ('schemaVersion', 'sessionId', 'sequence', 'active', 'calibrated')),
                (target, ('slot', 'positionMonoTime', 'velocitySeq', 'dRel', 'yRel', 'positionValid', 'reason')),
                (target.raw, ('oemTrackId', 'lifecycleRaw', 'u10Raw', 'u11Raw')))
      for value, fields in groups:
        for name in fields:
          with self.subTest(reader=type(value).__name__, field=name):
            actual, expected = read(value, name, marker), getattr(value, name, marker)
            self.assertIs(type(actual), type(expected))
            self.assertEqual(actual.hex() if isinstance(actual, float) else str(actual),
                             expected.hex() if isinstance(expected, float) else str(expected))
        self.assertIs(read(value, 'notAnObservationField', marker), marker)
        self.assertEqual(read(value, 'schema'), value.schema)
        self.assertTrue(callable(read(value, 'to_dict')))
        for name in (None, 1, []):
          with self.assertRaises(TypeError) as actual_error:
            read(value, name, marker)
          with self.assertRaises(TypeError) as expected_error:
            getattr(value, name, marker)
          self.assertEqual(str(actual_error.exception), str(expected_error.exception))
    class Ordinary:
      value = 4
      def __getattr__(self, name):
        if name == 'dynamic':
          return 7
        if name == 'broken':
          raise RuntimeError('ordinary accessor failed')
        raise AttributeError(name)
    self.assertEqual(read({'value': 3}, 'value'), 3)
    self.assertIs(read({}, 'value', marker), marker)
    self.assertEqual(read(Ordinary(), 'value'), 4)
    self.assertEqual(read(Ordinary(), 'dynamic'), 7)
    self.assertIs(read(Ordinary(), 'missing', marker), marker)
    with self.assertRaisesRegex(RuntimeError, 'ordinary accessor failed'):
      read(Ordinary(), 'broken', marker)

  def test_native_fields_bypass_generic_lookup_without_caching_values(self):
    from openpilot.cereal import custom
    native = custom.HondaRadarObservations.new_message()
    native.observations = [observation(0)]
    builder, reader = native.observations[0], native.as_reader().observations[0]
    with patch('openpilot.selfdrive.car.honda_radar_fusion.getattr', create=True,
               side_effect=AssertionError('generic dynamic attribute lookup')):
      for target in (builder, reader):
        self.assertEqual(read(target, 'positionMonoTime'), START)
        self.assertEqual(read(read(target, 'raw'), 'oemTrackId'), 7)
        self.assertEqual(read(read(target, 'raw'), 'u11Raw'), 864)
        self.assertEqual(str(read(target, 'reason')), 'uncalibrated')
      builder.dRel = 23.0
      self.assertEqual(read(builder, 'dRel'), 23.0)
      self.assertEqual(read(reader, 'dRel'), 23.0)

  def test_native_ingest_reuses_sort_keys_without_reordering_events(self):
    from openpilot.cereal import custom
    t = START + STEP
    keys = ('positionMonoTime', 'positionValid', 'positionSeq')
    events = [observation(1, slot=1, oem=8, velocity=False), observation(1, velocity=False), observation(1),
              observation(1, distance=21.0, velocity=False),
              {**observation(1, slot=1, oem=8, velocity=False), 'positionValid': False, 'reason': 'conflictingFrame'}]
    accesses = Counter()
    def counted_read(obj, name, default=None):
      if name in keys:
        accesses[name] += 1
      return read(obj, name, default)
    for reader in (False, True):
      with self.subTest(reader=reader):
        packet = custom.HondaRadarObservations.new_message()
        packet.observations = events
        before = packet.to_dict()
        native = packet.as_reader() if reader else packet
        fusion, reference = HondaRadarFusion(), HondaRadarFusion()
        for engine in (fusion, reference):
          advance(engine, 0, [observation(0), observation(0, slot=1, oem=8)])
          engine.record_inputs(Inputs(t), t + 1)
        accesses.clear()
        with patch('openpilot.selfdrive.car.honda_radar_fusion.read', side_effect=counted_read):
          fusion.ingest(native.observations, t + 1)
        self.assertEqual(accesses, dict.fromkeys(keys, len(events)))
        reference.ingest(before['observations'], t + 1)
        self.assertEqual(packet.to_dict(), before)
        self.assertEqual(fusion.counts, reference.counts)
        self.assertEqual(fusion.bindings, reference.bindings)
        self.assertEqual(fusion.tracks.keys(), reference.tracks.keys())
        for identifier, track in fusion.tracks.items():
          actual, expected = vars(track).copy(), vars(reference.tracks[identifier]).copy()
          for name in ('mean', 'covariance'):
            left, right = actual.pop(name), expected.pop(name)
            self.assertEqual((left.dtype, left.shape, left.tobytes()), (right.dtype, right.shape, right.tobytes()))
          self.assertEqual(actual, expected)
        # Three equal sorting keys must retain position, AUX, then conflict.
        self.assertEqual(fusion.tracks[7].positions[-1][1], 20.0)
        self.assertEqual(fusion.tracks[7].last_invalid_reason, 'conflictingPosition')
        self.assertEqual(fusion.tracks[8].last_invalid_reason, 'conflictingFrame')

  def test_ordinary_ingest_rechecks_dynamic_sort_keys(self):
    values = observation(1, velocity=False)
    accesses, counts = [], Counter()
    first_values = {'positionMonoTime': START, 'positionValid': False, 'positionSeq': 1}
    class Changing:
      def __getattr__(self, name):
        accesses.append(name)
        counts[name] += 1
        if counts[name] == 1 and name in first_values:
          return first_values[name]
        if name in values:
          return values[name]
        raise AttributeError(name)
    fusion = HondaRadarFusion()
    advance(fusion, 0)
    fusion.record_inputs(Inputs(START + STEP), START + STEP + 1)
    fusion.ingest([Changing()], START + STEP + 1)
    self.assertEqual(accesses[:7], ['positionMonoTime', 'positionValid', 'positionSeq', 'slot',
                                   'positionMonoTime', 'positionValid', 'positionSeq'])
    self.assertEqual(fusion.tracks[7].position_ns, values['positionMonoTime'])
    self.assertEqual(fusion.tracks[7].position_seq, values['positionSeq'])
    self.assertEqual(fusion.tracks[7].count, 2)

  def test_ordinary_ingest_preserves_sort_and_processing_accessor_errors(self):
    values = observation(1, velocity=False)
    class Failing:
      def __init__(self, name, occurrence):
        self.name, self.occurrence, self.counts = name, occurrence, Counter()

      def __getattr__(self, field):
        self.counts[field] += 1
        if field == self.name and self.counts[field] == self.occurrence:
          raise RuntimeError(f'failed {field}')
        if field in values:
          return values[field]
        raise AttributeError(field)
    for name in ('positionMonoTime', 'positionValid', 'positionSeq'):
      for occurrence in (1, 2):
        with self.subTest(field=name, occurrence=occurrence):
          fusion = HondaRadarFusion()
          advance(fusion, 0)
          fusion.record_inputs(Inputs(START + STEP), START + STEP + 1)
          with self.assertRaisesRegex(RuntimeError, f'failed {name}'):
            fusion.ingest([Failing(name, occurrence)], START + STEP + 1)
          self.assertEqual(fusion.tracks[7].count, 1)
          self.assertEqual(fusion.counts['positions'], 1)

  def test_measure_preserves_joseph_updates_without_generic_array_constructors(self):
    track, _ = projection_fixture()
    track.mean[:] = 0.0
    track.covariance = np.diag([1.0] * 4)
    expected, mean, variances = [], np.zeros(4), np.ones(4)
    values = (4.0, -2.0, 6.0, 8.0)
    for index, value in enumerate(values):
      mean[index], variances[index] = value / 2, .5
      expected.append((mean.copy(), np.diag(variances)))
    with patch.object(np, 'eye', side_effect=AssertionError('generic identity construction')), \
         patch.object(np, 'outer', side_effect=AssertionError('generic outer construction')):
      for index, value in enumerate(values):
        track.measure(index, value, 1.0)
        self.assertEqual(track.mean.tobytes(), expected[index][0].tobytes())
        self.assertEqual(track.covariance.tobytes(), expected[index][1].tobytes())
    # The shared identity template must survive all four independent corrections.
    other, _ = projection_fixture()
    other.mean[:] = 0.0
    other.covariance = np.diag([1.0] * 4)
    other.measure(0, 4.0, 1.0)
    self.assertEqual(other.mean.tobytes(), expected[0][0].tobytes())
    self.assertEqual(other.covariance.tobytes(), expected[0][1].tobytes())

  def test_finite_history_median_matches_numpy(self):
    smallest = float.fromhex('0x1p-1074')
    largest = sys.float_info.max
    cases = [[-0.0], [-0.0, -0.0], [0.0, -0.0], [-smallest, -0.0], [0.0, smallest],
             [-smallest, smallest], [largest], [-largest], [-largest, largest], [largest, largest], [-largest, -largest],
             [largest / 2.0, largest / 2.0], [-sys.float_info.min, -smallest, 0.0]]
    for size in range(1, 17):
      cases.extend(([-0.0] * size, [0.125] * size,
                    [(-1.0)**index * .01 * (index % 5) for index in range(size)],
                    [(-8.0, -.3, -.3, .1, 0.0, .4, 1.5, 8.0)[index % 8] for index in range(size)]))
    for values in cases:
      with self.subTest(values=values):
        self.assertTrue(all(math.isfinite(value) for value in values))
        with np.errstate(over='ignore', under='ignore'):
          expected = float(np.median(values))
        # Hex equality checks signed zero as well as all finite mantissa bits.
        self.assertEqual(_finite_median(iter(values)).hex(), expected.hex())
        self.assertEqual(_finite_median(reversed(values)).hex(), expected.hex())

  def test_first_confirmation_and_even_history_do_not_call_numpy_median(self):
    fusion = HondaRadarFusion()
    # The former first-confirmation path calls this six times and can lazily
    # import numpy.ma. Fail that dependency, even if another test warmed it.
    with patch.object(np, 'median', side_effect=AssertionError('optional median path called')) as median:
      for frame in range(4):
        out = advance(fusion, frame, distance=20.2, lateral=.1)
        self.assertEqual(len(out['leads']), 3)
        for lead in out['leads']:
          self.assertEqual(lead['bindingSamples'], frame + 1)
          self.assertEqual(lead['identityState'], 'unresolved' if frame < 2 else 'same')
          self.assertEqual(lead['state'], 'associating' if frame < 2 else 'fused')
        self.assertFalse(out['controlAllowed'])
        self.assertFalse(out['calibrated'])
      median.assert_not_called()
    for binding in fusion.bindings.values():
      self.assertTrue(binding.anchor_ready)
      self.assertAlmostEqual(binding.anchor_x, -.2)
      self.assertAlmostEqual(binding.anchor_y, -.1)

  def test_slot_migration_keeps_identity_and_does_not_consume_aux_twice(self):
    fusion = HondaRadarFusion()
    identifiers = []
    for frame in range(12):
      full = observation(frame, slot=frame % 3)
      main = {**full, 'frameMask': 15, 'velocitySeq': 0, 'velocityValid': False}
      out = advance(fusion, frame, [main, full, full])
      identifiers.append(out['tracks'][0]['trackId'])
    self.assertEqual(len(set(identifiers)), 1)
    self.assertEqual(out['tracks'][0]['observations'], 12)
    self.assertEqual(out['counters']['positions'], 12)
    self.assertEqual(len({l['trackId'] for l in out['leads']}), 1)
    self.assertTrue(all(l['state'] == 'fused' for l in out['leads']))
    self.assertFalse(out['controlAllowed'])
    self.assertFalse(out['calibrated'])

  def test_recorded_m4_reassignment_across_publications_preserves_filter_but_reconfirms_pair(self):
    # 2026-10-07 route 8b, M4: OEM 11 occupied OEM 16's slot 10.294531 ms
    # before OEM 16's new-slot F0, across two publications. Only the preceding
    # warm-up is synthetic; these boundary source times/IDs/geometry are logged.
    mark = 431_461_214_261
    old_ns = mark - 19_933_329_003
    fusion = HondaRadarFusion()
    for frame in range(12):
      back = 11 - frame
      sample = source_observation(old_ns - back * STEP, 17751 - 2 * back, (14 - back) & 15, 27 - 2 * back,
                                  generation=868, oem=16, distance=6.31056, lateral=-.805476)
      stamp = sample['positionMonoTime']
      out = fusion.update(body(frame, [sample]), Inputs(stamp, distance=6.31056, lateral=-.805476), stamp + 1)
    track = fusion.tracks[16]
    identifier, first_ns = track.identifier, track.first_ns
    mean, covariance = track.mean.copy(), track.covariance.copy()
    self.assertGreater(out['leads'][0]['distanceWeight'], 0)

    replacement = source_observation(mark - 19_850_653_273, 17763, 15, 17, generation=874,
                                     oem=11, distance=29.844, lateral=9.6422)
    publication = mark - 19_847_315_669
    out = fusion.update(body(12, [replacement]), Inputs(publication, distance=6.31056, lateral=-.805476), publication)
    self.assertFalse(track.active)
    self.assertFalse(track.velocity_good)
    self.assertFalse(track.acceleration_good)
    self.assertEqual(track.last_invalid_reason, 'slotReassigned')
    self.assertEqual((track.identifier, track.count), (identifier, 12))
    np.testing.assert_array_equal(track.mean, mean)
    np.testing.assert_array_equal(track.covariance, covariance)
    self.assertEqual((out['leads'][0]['distanceWeight'], out['leads'][0]['velocityWeight']), (0, 0))

    migrated = source_observation(mark - 19_840_358_742, 17771, 15, 29, generation=878,
                                  slot=4, oem=16, distance=6.31056, lateral=-.7929542787)
    main = {**migrated, 'frameMask': 15, 'velocitySeq': 0, 'velocityValid': False}
    publication = mark - 19_789_831_294
    out = fusion.update(body(13, [main, migrated, migrated]),
                        Inputs(publication, distance=6.31056, lateral=-.7929542787), publication)
    self.assertIs(fusion.tracks[16], track)
    self.assertEqual((track.identifier, track.first_ns, track.count, track.slot), (identifier, first_ns, 13, 4))
    self.assertEqual(track.position_ns, migrated['positionMonoTime'])
    self.assertEqual(track.previous_position_ns, old_ns)
    self.assertTrue(track.active)
    self.assertTrue(track.velocity_good)
    self.assertTrue(track.mature)
    self.assertEqual((track.slot_reassigned_ns, track.slot_reassigned_seq), (0, 0))
    self.assertEqual(out['counters']['positions'], 14)  # Two OEMs; AUX/duplicates add none.
    self.assertEqual(out['leads'][0]['bindingSamples'], 1)
    self.assertEqual(out['leads'][0]['identityState'], 'unresolved')
    self.assertEqual((out['leads'][0]['distanceWeight'], out['leads'][0]['velocityWeight']), (0, 0))

  def test_recorded_m9_same_envelope_reassignment_accepts_later_sequence(self):
    # M9: equal F0 time and distinct global sequence numbers formerly caused
    # OEM 20's migrated position to fail stamp <= invalid_position_ns.
    mark = 2_420_533_395_534
    old_ns, next_ns = mark - 9_969_549_007, mark - 9_900_247_132
    for reverse in (False, True):
      with self.subTest(reverse=reverse):
        fusion = HondaRadarFusion()
        for frame in range(3):
          back = 2 - frame
          sample = source_observation(old_ns - back * STEP, 193655 - 2 * back, (6 - back) & 15, 449 - 2 * back,
                                      generation=5220, oem=20, distance=36.6984, lateral=2.3326)
          stamp = sample['positionMonoTime']
          fusion.update(body(frame, [sample]), Inputs(stamp), stamp + 1)
        track = fusion.tracks[20]
        replacement = source_observation(next_ns, 193657, 7, 1, generation=5252,
                                         oem=36, distance=18.42, lateral=6.2707)
        migrated = source_observation(next_ns, 193659, 7, 451, generation=5253,
                                      slot=1, oem=20, distance=36.64128, lateral=2.4009)
        samples = [replacement, migrated]
        out = fusion.update(body(3, list(reversed(samples)) if reverse else samples), Inputs(next_ns), next_ns + 1)
        self.assertIs(fusion.tracks[20], track)
        self.assertEqual((track.slot, track.count, track.position_seq), (1, 4, 193659))
        self.assertTrue(track.active)
        self.assertEqual(out['counters']['positions'], 5)
        self.assertEqual(out['counters']['old'], 0)

  def test_decoder_slot_swap_preserves_both_identities_across_frame_wrap(self):
    for plateau in (False, True):
      for order in ((0, 1), (1, 0)):
        with self.subTest(plateau=plateau, order=order):
          decoder, fusion = BoschARadarObserver(2), HondaRadarFusion()
          for frame in range(16):
            frames = [can for slot, oem, distance in ((0, 7, 403), (1, 8, 753))
                      for can in scan_frames(slot=slot, index=frame, life=4094 if plateau else 1 + 2 * frame,
                                             distance=distance, azimuth=1024, velocity=864, oem_id=oem)]
            decoder.feed(START + frame * STEP, frames)
            advance(fusion, frame, decoder.drain())
          tracks = dict(fusion.tracks)
          frames = [can for slot in order for can in scan_frames(slot=slot, index=0, life=4094 if plateau else 33,
                    distance=753 if slot == 0 else 403, azimuth=1024, velocity=864, oem_id=8 if slot == 0 else 7)]
          decoder.feed(START + 16 * STEP, frames)
          out = advance(fusion, 16, decoder.drain())
          for oem, slot in ((7, 1), (8, 0)):
            self.assertIs(fusion.tracks[oem], tracks[oem])
            self.assertEqual((tracks[oem].slot, tracks[oem].count, tracks[oem].frame_index), (slot, 17, 0))
            self.assertTrue(tracks[oem].active)
          self.assertEqual(out['counters']['positions'], 34)

  def test_reassignment_cannot_restore_a_really_invalidated_posterior(self):
    failures = ('invalidGeometry', 'expired', 'withdrawn', 'conflictingFrame', 'newSlotInvalid', 'newSlotConflict',
                'invalidLife', 'slotConflict', 'conflictingPosition', 'positionInnovation')
    for failure in failures:
      with self.subTest(failure=failure):
        fusion = HondaRadarFusion()
        for frame in range(8):
          advance(fusion, frame)
        track = fusion.tracks[7]
        advance(fusion, 8, [observation(8, oem=8, distance=60, lateral=6)])
        self.assertFalse(track.active)
        self.assertNotEqual(track.slot_reassigned_ns, 0)
        mean, covariance = track.mean.copy(), track.covariance.copy()
        sample = observation(9, slot=1)
        if failure in ('invalidGeometry', 'expired', 'withdrawn'):
          sample.update(slot=0, positionValid=False, reason=failure)
        elif failure == 'conflictingFrame':
          sample = {**observation(7), 'positionValid': False, 'reason': failure}
        elif failure in ('newSlotInvalid', 'newSlotConflict'):
          sample.update(positionValid=False, generation=27,
                        reason='invalidGeometry' if failure == 'newSlotInvalid' else 'conflictingFrame')
        elif failure == 'invalidLife':
          sample['raw']['lifecycleRaw'] = 4095
        elif failure == 'slotConflict':
          sample['frameIndex'] = 7
          sample['raw']['lifecycleRaw'] = 15
        elif failure == 'conflictingPosition':
          sample = observation(7, distance=21)
        else:
          sample['dRel'] = 90
        out = advance(fusion, 9, [sample])
        self.assertIs(fusion.tracks[7], track)
        self.assertFalse(track.active)
        self.assertFalse(track.velocity_good)
        self.assertEqual(track.count, 8)
        np.testing.assert_array_equal(track.mean, mean)
        np.testing.assert_array_equal(track.covariance, covariance)
        self.assertEqual((track.slot_reassigned_ns, track.slot_reassigned_seq), (0, 0))
        self.assertEqual((out['leads'][0]['distanceWeight'], out['leads'][0]['velocityWeight']), (0, 0))
        # Still within 200 ms of the original sample: a cleared pending flag,
        # not expiry, must prevent this otherwise continuous migration.
        out = advance(fusion, 10, [observation(10, slot=1)])
        self.assertNotEqual(fusion.tracks[7].identifier, track.identifier)
        self.assertEqual(fusion.tracks[7].count, 1)
        self.assertEqual(out['leads'][0]['distanceWeight'], 0)

  def test_reassignment_requires_new_slot_continuity_and_fresh_source(self):
    for failure in ('sameSlot', 'lifecycleRestart', 'sourceGap', 'olderSequence', 'olderSource', 'futureSource'):
      with self.subTest(failure=failure):
        fusion = HondaRadarFusion()
        for frame in range(8):
          advance(fusion, frame)
        track = fusion.tracks[7]
        replacement = observation(8, oem=8, distance=60, lateral=6)
        advance(fusion, 8, [replacement])
        sample = observation(9, slot=1)
        now = START + 9 * STEP
        if failure == 'sameSlot':
          sample['slot'] = 0
        elif failure == 'lifecycleRestart':
          sample['raw']['lifecycleRaw'] = 1
        elif failure == 'sourceGap':
          sample = observation(11, slot=1)
          now = START + 11 * STEP
        elif failure == 'olderSequence':
          sample.update(positionSeq=replacement['positionSeq'] - 1, positionMonoTime=replacement['positionMonoTime'])
        elif failure == 'olderSource':
          sample['positionMonoTime'] = replacement['positionMonoTime'] - 1
        else:
          sample['positionMonoTime'] = now + 1
        out = fusion.update(body(9, [sample]), Inputs(now), now)
        current = fusion.tracks[7]
        if failure in ('sameSlot', 'lifecycleRestart', 'sourceGap'):
          self.assertNotEqual(current.identifier, track.identifier)
          self.assertEqual(current.count, 1)
        else:
          self.assertIs(current, track)
          self.assertFalse(current.active)
          self.assertEqual(current.count, 8)
        self.assertEqual((out['leads'][0]['distanceWeight'], out['leads'][0]['velocityWeight']), (0, 0))

  def test_reassignment_waiting_aux_and_old_slot_events_do_not_extend_or_revoke_migration(self):
    fusion = HondaRadarFusion()
    for frame in range(8):
      advance(fusion, frame)
    track = fusion.tracks[7]
    old = observation(7)
    advance(fusion, 8, [observation(8, oem=8, distance=60, lateral=6)])
    late_aux = {**old, 'velocitySeq': 1800, 'velocityMonoTime': START + 9 * STEP}
    fusion.record_inputs(Inputs(START + 9 * STEP), START + 9 * STEP)
    fusion.ingest([late_aux], START + 9 * STEP)
    self.assertFalse(track.active)
    self.assertEqual((track.position_ns, track.count), (old['positionMonoTime'], 8))
    self.assertFalse(track.velocity_good)
    advance(fusion, 9, [observation(9, slot=1)])
    self.assertIs(fusion.tracks[7], track)
    self.assertTrue(track.active)
    self.assertEqual(track.count, 9)
    invalid = {'slot': 0, 'positionValid': False, 'generation': old['generation'],
               'positionMonoTime': old['positionMonoTime'], 'reason': 'invalidGeometry'}
    advance(fusion, 10, [invalid, late_aux])
    self.assertIs(fusion.tracks[7], track)
    self.assertTrue(track.active)
    self.assertEqual((track.slot, track.count, track.position_ns), (1, 9, START + 9 * STEP))

  def test_old_slot_invalidation_cannot_remove_migrated_target(self):
    fusion = HondaRadarFusion()
    for frame in range(6):
      out = advance(fusion, frame)
    identifier = out['tracks'][0]['trackId']
    newer = observation(6, slot=1)
    invalid_old_slot = {'slot': 0, 'positionValid': False, 'positionMonoTime': newer['positionMonoTime'], 'reason': 'invalidGeometry'}
    out = advance(fusion, 6, [newer, invalid_old_slot])
    self.assertEqual(out['tracks'][0]['trackId'], identifier)
    self.assertFalse(out['tracks'][0]['coasting'])
    self.assertEqual(out['tracks'][0]['slot'], 1)

  def test_same_scan_two_slots_is_not_two_measurements(self):
    fusion = HondaRadarFusion()
    a, b = observation(0), observation(0, slot=1)
    b['positionMonoTime'] += 1_000_000
    sm = Inputs(START)
    out = fusion.update(body(0, [a, b]), sm, START + 2_000_000)
    self.assertEqual(out['counters']['positions'], 1)
    self.assertEqual(len(out['tracks']), 1)
    self.assertEqual(out['counters']['duplicate'], 1)

  def test_conflict_in_accepted_measurement_retracts_only_that_posterior(self):
    for accepted_conflict in (True, False):
      fusion = HondaRadarFusion()
      for frame in range(6):
        advance(fusion, frame)
      raw = observation(5 if accepted_conflict else 6)
      conflict = {'slot': 0, 'positionValid': False, 'positionMonoTime': raw['positionMonoTime'],
                  'positionSeq': raw['positionSeq'] if accepted_conflict else 0, 'reason': 'conflictingFrame'}
      out = advance(fusion, 6, [conflict])
      self.assertTrue(out['leads'][0]['present'])
      if accepted_conflict:
        # The invalid posterior remains diagnostic-only until its normal TTL.
        self.assertFalse(out['tracks'][0]['active'])
        self.assertEqual(out['tracks'][0]['lastInvalidReason'], 'conflictingFrame')
        self.assertEqual(out['leads'][0]['state'], 'visionOnly')
        self.assertEqual(out['leads'][0]['distanceWeight'], 0)
        self.assertEqual(out['leads'][0]['velocityWeight'], 0)
      else:
        self.assertEqual(len(out['tracks']), 1)
        self.assertFalse(out['tracks'][0]['coasting'])

  def test_admission_and_coasting_weights_have_no_boundary_step(self):
    fusion = HondaRadarFusion()
    for frame in range(12):
      out = advance(fusion, frame, distance=22)
      if frame == 2:
        self.assertLess(out['leads'][0]['distanceWeight'], .2)
    last = START + 11 * STEP
    weights = []
    for seq, age in enumerate((99_000_000, 101_000_000), 12):
      t = last + age
      out = fusion.update(body(seq, []), Inputs(t, distance=22), t + 1)
      weights.append(out['leads'][0]['distanceWeight'])
    self.assertGreater(weights[0], .5)
    self.assertGreater(weights[1], .9 * weights[0])

  def test_dropped_observations_clear_previous_confirmation(self):
    fusion = HondaRadarFusion()
    for frame in range(8):
      out = advance(fusion, frame)
    self.assertGreater(out['leads'][0]['distanceWeight'], 0)
    t = START + 8 * STEP
    out = fusion.update(body(8, [], counters={'dropped': 32}), Inputs(t), t + 1)
    self.assertEqual(out['tracks'], [])
    self.assertTrue(out['leads'][0]['present'])
    self.assertEqual(out['leads'][0]['distanceWeight'], 0)

  def test_lifecycle_restart_changes_internal_identity(self):
    fusion = HondaRadarFusion()
    for frame in range(4):
      out = advance(fusion, frame)
    old = out['tracks'][0]['trackId']
    new = observation(4)
    new['raw']['lifecycleRaw'] = 1
    out = advance(fusion, 4, [new])
    self.assertNotEqual(out['tracks'][0]['trackId'], old)
    self.assertEqual(out['tracks'][0]['observations'], 1)
    self.assertEqual(out['leads'][0]['state'], 'associating')

  def test_4094_plateau_preserves_maturity_across_wrap_and_slot_migration(self):
    fusion = HondaRadarFusion()
    identifiers = set()
    for frame in range(36):
      sample = observation(frame, slot=frame % 3)
      sample['raw']['lifecycleRaw'] = 4094
      sample['generation'] = frame + 1  # Decoder slot generations are not object IDs.
      out = advance(fusion, frame, [sample, sample])
      track = out['tracks'][0]
      identifiers.add(track['trackId'])
      self.assertEqual(track['observations'], frame + 1)
      self.assertEqual(track['mature'], frame >= 2)
    self.assertEqual(len(identifiers), 1)
    self.assertEqual(out['counters']['positions'], 36)
    self.assertGreater(out['leads'][0]['distanceWeight'], .5)

  def test_recorded_342_first_saturation_return_keeps_physical_identity(self):
    # Route 89, segment 15: source/ego/model clocks and raw measurements are
    # copied from track342-reacquisition.json. Recorded decoder generations
    # change at the sentinel; they are deliberately not physical object IDs.
    rows = (
      (924450550062, 50318, 5, 4087, 1195, 690, 1047, 885, 65, 924442196312, 18.6826114654541,
       924467680739, 924423661646, -.002433937741443515, 924481354646),
      (924511323344, 50326, 6, 4089, 1195, 690, 1047, 886, 64, 924502036260, 18.677932739257812,
       924517201573, 924473663416, -.00264874123968184, 924537474073),
      (924571265635, 50334, 7, 4091, 1195, 690, 1047, 885, 64, 924563254906, 18.688756942749023,
       924567966312, 924523665344, -.0032938183285295963, 924592373239),
      (924642906364, 50342, 8, 4093, 1195, 689, 1047, 881, 64, 924635500583, 18.700733184814453,
       924618017146, 924573665552, -.003514645854011178, 924645072927),
      (924713209333, 0, 9, 4095, 0, 688, 1047, 879, 64, 924705037823, 18.702844619750977,
       924717383604, 924673665916, -.0036914919037371874, 924753765583),
      (924783916364, 50356, 10, 4094, 1206, 687, 1046, 875, 64, 924774263968, 18.714942932128906,
       924767525531, 924723668312, -.0038886340335011482, 924804038812),
    )
    fusion = HondaRadarFusion()
    identifier = None
    for packet, (stamp, seq, index, life, generation, distance, azimuth, u11, u10, ego_ns, speed,
                 model_ns, eof_ns, yaw, now) in enumerate(rows):
      x = distance * .05712 - 3
      sample = source_observation(stamp, seq, index, life, generation, oem=60, slot=1, distance=x,
                                  lateral=x * math.tan((azimuth - 1024) / 2048.0), relative=(u11 - 864) / 64, u10=u10)
      sample['raw'].update(statusRaw=7, rangeRaw=distance, azimuthRaw=azimuth)
      if life == 4095:
        sample.update(positionValid=False, velocityValid=False, velocitySeq=0, dRel=0.0, yRel=0.0, vRel=0.0,
                      reason='invalidGeometry')
      sm = Inputs(ego_ns, speed=speed, yaw=-yaw, probability=0)
      sm.logMonoTime['modelV2'], sm['modelV2'].timestampEof = model_ns, eof_ns
      out = fusion.update(body(packet, [position_only(sample), sample]), sm, now)
      track = fusion.tracks[60]
      if identifier is None:
        identifier = track.identifier
      self.assertEqual(track.identifier, identifier)
      self.assertEqual(track.count, packet + 1 if packet < 4 else 4 if packet == 4 else 5)
      self.assertEqual(track.active, life != 4095)
      if life == 4095:
        self.assertEqual(track.position_ns, rows[3][0])
        self.assertFalse(out['tracks'][0]['velocityGood'])
      if packet == 5:
        self.assertTrue(track.velocity_good)
        self.assertTrue(track.mature)
        self.assertEqual((track.position_ns, track.position_seq, track.generation), (stamp, seq, generation))
        self.assertEqual(track.previous_position_ns, rows[3][0])
        self.assertLess(track.innovation, .3)
    self.assertEqual(out['counters']['rejectedPosition'], 0)
    self.assertEqual(out['counters']['rejectedVelocity'], 0)

  def test_saturation_return_waits_for_its_own_aux_without_refreshing_old_position(self):
    fusion, sentinel, returned = saturation_fixture()
    old = fusion.tracks[7]
    identity, position = old.identifier, old.position_ns
    out = advance(fusion, 8, [position_only(sentinel), sentinel, sentinel])
    self.assertFalse(old.active)
    self.assertEqual(old.count, 8)
    self.assertEqual(out['counters']['positions'], 8)
    main = position_only(returned)
    for delay in (1, 10_000_000):
      now = START + 9 * STEP + delay
      out = fusion.update(body(fusion.sequence, [main, main]), Inputs(now), now)
      self.assertFalse(fusion.tracks[7].active)
      self.assertEqual((fusion.tracks[7].position_ns, fusion.tracks[7].count), (position, 8))
      self.assertEqual(out['leads'][0]['distanceWeight'], 0)
    now = START + 9 * STEP + 20_000_000
    out = fusion.update(body(fusion.sequence, [returned]), Inputs(now), now)
    track = fusion.tracks[7]
    self.assertEqual(track.identifier, identity)
    self.assertTrue(track.active and track.velocity_good and track.mature)
    self.assertEqual((track.position_ns, track.count), (returned['positionMonoTime'], 9))
    self.assertEqual(out['counters']['positions'], 9)
    # Repeating this AUX is not a second position or recovery observation.
    fusion.update(body(fusion.sequence, [returned]), Inputs(now + 1), now + 1)
    self.assertEqual(fusion.tracks[7].count, 9)

  def test_saturation_return_requires_witnessed_growth_and_lifecycle_only_sentinel(self):
    failures = ('noSentinel', 'noGrowth', 'status', 'range', 'azimuth', 'oem', 'incomplete', 'scanSkip',
                'expired', 'withdrawn', 'secondInvalidScan', 'competingInvalidSlot')
    for failure in failures:
      with self.subTest(failure=failure):
        fusion, sentinel, returned = saturation_fixture()
        if failure == 'noGrowth':
          fusion = HondaRadarFusion()
          sample = observation(7)
          sample['raw'].update(lifecycleRaw=4093, statusRaw=7, rangeRaw=403, azimuthRaw=1024)
          advance(fusion, 7, [sample])
        identity = fusion.tracks[7].identifier
        if failure in ('status', 'range', 'azimuth', 'oem'):
          key, value = {'status': ('statusRaw', 15), 'range': ('rangeRaw', 4095),
                        'azimuth': ('azimuthRaw', 2047), 'oem': ('oemTrackId', 255)}[failure]
          sentinel['raw'][key] = value
        elif failure == 'incomplete':
          sentinel['frameMask'] = 7
        elif failure == 'scanSkip':
          sentinel['frameIndex'] += 1
        elif failure in ('expired', 'withdrawn'):
          sentinel['reason'] = failure
        claims = [sentinel]
        if failure == 'competingInvalidSlot':
          other = copy.deepcopy(sentinel)
          other['slot'] = 1
          claims = [other, sentinel]
        if failure != 'noSentinel':
          advance(fusion, 8, claims)
        frame = 9
        if failure == 'secondInvalidScan':
          extra = copy.deepcopy(sentinel)
          extra.update(positionMonoTime=START + 9 * STEP, frameIndex=9)
          advance(fusion, 9, [extra])
          returned.update(positionMonoTime=START + 10 * STEP, velocityMonoTime=START + 10 * STEP,
                          frameIndex=10, positionSeq=1001, velocitySeq=1002)
          frame = 10
        now = START + frame * STEP + 1
        fusion.update(body(fusion.sequence, [returned]), Inputs(now), now)
        self.assertNotEqual(fusion.tracks[7].identifier, identity)
        self.assertEqual(fusion.tracks[7].count, 1)

  def test_saturation_return_rejects_velocity_identity_and_position_failures(self):
    failures = ('positionOnly', 'badQuality', 'badNative', 'invalidVelocity', 'velocityInnovation', 'velocityAge',
                'velocityFuture', 'velocityDuplicate', 'sourceExpired', 'positionDuplicate', 'sourceOld',
                'positionJump', 'lateralJump', 'changedAuxPosition', 'replacement', 'slotConflict', 'frameConflict')
    for failure in failures:
      with self.subTest(failure=failure):
        fusion, sentinel, returned = saturation_fixture()
        old = fusion.tracks[7]
        identity, position, count = old.identifier, old.position_ns, old.count
        advance(fusion, 8, [sentinel])
        now = START + 9 * STEP + 1
        claims = [returned]
        if failure == 'positionOnly':
          claims = [position_only(returned)]
        elif failure == 'badQuality':
          returned['raw']['u10Raw'] = 256
        elif failure == 'badNative':
          returned['raw']['u11Raw'] = 1728
        elif failure == 'invalidVelocity':
          returned['velocityValid'] = False
        elif failure == 'velocityInnovation':
          returned['vRel'], returned['raw']['u11Raw'] = 12.0, 1632
        elif failure == 'velocityAge':
          returned['velocityMonoTime'] -= 50_000_001
        elif failure == 'velocityFuture':
          returned['velocityMonoTime'] = now + 1
        elif failure == 'velocityDuplicate':
          returned['velocitySeq'] = old.velocity_seq
        elif failure == 'sourceExpired':
          now += 201_000_000
        elif failure == 'positionDuplicate':
          returned['positionSeq'] = old.position_seq
        elif failure == 'sourceOld':
          returned['positionMonoTime'] = old.position_ns - 1
        elif failure == 'positionJump':
          returned['dRel'] += 20
        elif failure == 'lateralJump':
          returned['yRel'] += 8
        elif failure == 'changedAuxPosition':
          advance(fusion, 9, [position_only(returned)])
          returned['dRel'] += .1
          now += 10_000_000
        elif failure == 'replacement':
          stamp = START + 8 * STEP + 2
          fusion.update(body(fusion.sequence, [observation(8, oem=8, distance=60)]), Inputs(stamp), stamp)
        elif failure == 'slotConflict':
          other = copy.deepcopy(returned)
          other.update(slot=1, positionSeq=returned['positionSeq'] + 1)
          claims = [other, returned]
        else:
          conflict = copy.deepcopy(sentinel)
          conflict['reason'] = 'conflictingFrame'
          stamp = START + 8 * STEP + 2
          fusion.update(body(fusion.sequence, [conflict]), Inputs(stamp), stamp)
        out = fusion.update(body(fusion.sequence, claims), Inputs(now), now)
        track = fusion.tracks[7]
        self.assertFalse(track.identifier == identity and track.active)
        self.assertEqual((old.position_ns, old.count), (position, count))
        self.assertEqual(out['leads'][0]['distanceWeight'], 0)

  def test_plateau_does_not_inherit_invalid_geometry_or_a_real_lifecycle_restart(self):
    for failure in ('invalidLife', 'positionJump', 'lifecycleRestart'):
      with self.subTest(failure=failure):
        fusion = HondaRadarFusion()
        for frame in range(8):
          sample = observation(frame)
          sample['raw']['lifecycleRaw'] = 4094
          out = advance(fusion, frame, [sample])
        old = out['tracks'][0]['trackId']
        sample = observation(8)
        sample['raw']['lifecycleRaw'] = 4095 if failure == 'invalidLife' else 1 if failure == 'lifecycleRestart' else 4094
        if failure == 'positionJump':
          sample['dRel'] = 90.0
        out = advance(fusion, 8, [sample])
        self.assertEqual(out['leads'][0]['distanceWeight'], 0)
        self.assertEqual(out['leads'][0]['velocityWeight'], 0)
        if failure == 'lifecycleRestart':
          self.assertNotEqual(out['tracks'][0]['trackId'], old)
          self.assertEqual(out['tracks'][0]['observations'], 1)
        else:
          self.assertFalse(out['tracks'][0]['active'])
          recovery = observation(9)
          recovery['raw']['lifecycleRaw'] = 4094
          out = advance(fusion, 9, [recovery])
          self.assertNotEqual(out['tracks'][0]['trackId'], old)
          self.assertEqual(out['tracks'][0]['observations'], 1)
          self.assertEqual(out['tracks'][0]['birthReason'], 'invalidRecovery')
          self.assertEqual(out['leads'][0]['distanceWeight'], 0)

  def test_plateau_same_scan_slot_conflict_revokes_existing_pair(self):
    fusion = HondaRadarFusion()
    for frame in range(8):
      sample = observation(frame)
      sample['raw']['lifecycleRaw'] = 4094
      out = advance(fusion, frame, [sample])
    original = observation(8)
    conflicting = observation(8, slot=1)
    original['raw']['lifecycleRaw'] = conflicting['raw']['lifecycleRaw'] = 4094
    out = advance(fusion, 8, [original, conflicting])
    self.assertEqual(out['counters']['positions'], 9)
    self.assertFalse(out['tracks'][0]['active'])
    self.assertEqual(out['tracks'][0]['lastInvalidReason'], 'slotConflict')
    self.assertEqual(out['leads'][0]['distanceWeight'], 0)
    self.assertEqual(out['leads'][0]['velocityWeight'], 0)

  def test_decoder_same_envelope_two_slot_claims_revoke_the_accepted_posterior(self):
    for plateau in (False, True):
      for slots in ((0, 1), (1, 0)):
        with self.subTest(plateau=plateau, slots=slots):
          decoder, fusion = BoschARadarObserver(2), HondaRadarFusion()
          for frame in range(12):
            decoder.feed(START + frame * STEP, scan_frames(slot=0, index=frame % 16,
                         life=4094 if plateau else 1 + 2 * frame, distance=403, azimuth=1024, velocity=864, oem_id=7))
            out = advance(fusion, frame, decoder.drain())
          identifier = out['tracks'][0]['trackId']
          self.assertGreater(out['leads'][0]['distanceWeight'], .5)
          # Real decoder events: one envelope, equal F0 times, distinct global
          # positionSeq values, and separate position/AUX publications per slot.
          frames = [can for slot in slots for can in scan_frames(slot=slot, index=12,
                    life=4094 if plateau else 25, distance=403, azimuth=1024, velocity=864, oem_id=7)]
          decoder.feed(START + 12 * STEP, frames)
          events = decoder.drain()
          valid = [event for event in events if event['positionValid']]
          self.assertEqual({event['slot'] for event in valid}, {0, 1})
          self.assertEqual(len({event['positionMonoTime'] for event in valid}), 1)
          self.assertEqual(len({event['positionSeq'] for event in valid}), 2)
          out = advance(fusion, 12, events)
          self.assertEqual(len(out['tracks']), 1)
          self.assertEqual(out['tracks'][0]['trackId'], identifier)
          self.assertEqual(out['counters']['positions'], 13)
          self.assertFalse(out['tracks'][0]['active'])
          self.assertEqual(out['tracks'][0]['lastInvalidReason'], 'slotConflict')
          self.assertTrue(out['leads'][0]['present'])
          self.assertEqual(out['leads'][0]['distanceWeight'], 0)
          self.assertEqual(out['leads'][0]['velocityWeight'], 0)

  def test_decoder_late_old_slot_completion_does_not_retract_newer_migration(self):
    decoder, fusion = BoschARadarObserver(2), HondaRadarFusion()
    for frame in range(8):
      decoder.feed(START + frame * STEP, scan_frames(slot=0, index=frame, life=4094,
                   distance=403, azimuth=1024, velocity=864, oem_id=7))
      out = advance(fusion, frame, decoder.drain())
    identifier = out['tracks'][0]['trackId']
    late_scan = scan_frames(slot=0, index=8, life=4094, distance=403, azimuth=1024, velocity=864, oem_id=7)
    decoder.feed(START + 7 * STEP + STEP // 2, late_scan[:1])
    decoder.feed(START + 8 * STEP, scan_frames(slot=1, index=8, life=4094,
                 distance=403, azimuth=1024, velocity=864, oem_id=7))
    out = advance(fusion, 8, decoder.drain())
    self.assertEqual(out['tracks'][0]['trackId'], identifier)
    self.assertTrue(out['tracks'][0]['active'])
    self.assertEqual(out['tracks'][0]['slot'], 1)
    accepted_time = out['tracks'][0]['positionMonoTime']
    decoder.feed(START + 9 * STEP, late_scan[1:])
    events = decoder.drain()
    self.assertTrue(any(event['positionValid'] and event['positionMonoTime'] < accepted_time for event in events))
    out = advance(fusion, 9, events)
    self.assertEqual(out['tracks'][0]['trackId'], identifier)
    self.assertEqual(out['tracks'][0]['positionMonoTime'], accepted_time)
    self.assertTrue(out['tracks'][0]['active'])
    self.assertEqual(out['tracks'][0]['lastInvalidReason'], '')
    self.assertGreater(out['leads'][0]['distanceWeight'], .5)

  def test_stopped_path_is_unknown_and_does_not_delete_confirmed_lead(self):
    fusion = HondaRadarFusion()
    for frame in range(12):
      out = advance(fusion, frame, distance=20, speed=0, stopped_path=True)
    self.assertEqual(out['tracks'][0]['pathState'], 'unknown')
    self.assertTrue(out['leads'][0]['present'])
    self.assertGreater(out['leads'][0]['distanceWeight'], 0)

  def test_curved_corridor_and_approaching_side_target(self):
    self.assertEqual(path_relation(18.48, 4, 10, 0, [0, 10, 20, 30], [0, -1, -4, -9])[0], 'inPath')
    relation, _, entry = path_relation(20, 4, 10, -2, [0, 10, 30, 60], [0, 0, 0, 0])
    self.assertEqual(relation, 'entering')
    self.assertAlmostEqual(entry, 1.1)
    self.assertEqual(path_relation(80, 0, 10, 0, [0, 5, 10], [0, 0, 0])[0], 'unknown')
    self.assertEqual(path_relation(10, 0, 10, 0, [0, float('nan')], [0, 0])[0], 'unknown')

  def test_path_geometry_preserves_boundaries_order_and_unknowns(self):
    straight = ([0.0, 10.0, 20.0, 30.0], [0.0] * 4)
    cases = ((1.8, 0.0, 'unknown', 'inPath'), (math.nextafter(1.8, math.inf), 0.0, 'unknown', 'outOfPath'),
             (2.3, 0.0, 'inPath', 'inPath'), (math.nextafter(2.3, math.inf), 0.0, 'inPath', 'outOfPath'),
             (2.0, -.5, 'unknown', 'entering'), (2.0, -.5, 'inPath', 'inPath'),
             (1.9, -.2, 'unknown', 'outOfPath'), (1.9, math.nextafter(-.2, -math.inf), 'unknown', 'entering'),
             (2.8, -.5, 'unknown', 'entering'), (math.nextafter(2.8, math.inf), -.5, 'unknown', 'outOfPath'),
             (-4.0, 2.0, 'unknown', 'entering'))
    for lateral, lateral_speed, previous, expected in cases:
      with self.subTest(lateral=lateral, lateral_speed=lateral_speed, previous=previous):
        self.assertEqual(path_relation(18.48, lateral, 10.0, lateral_speed, *straight, previous)[0], expected)
    for path_x, expected_offset in (([0, 20, 10, 30], 2.0), ([30, 10, 20, 0], -2.0)):
      with self.subTest(path_x=path_x):
        # Equal-distance overlapping segments keep the first segment's sign.
        self.assertEqual(path_relation(18.48, 2.0, 10.0, 0.0, path_x, [0.0] * 4), ('outOfPath', expected_offset, 0.0))
    self.assertEqual(path_relation(18.48, 0.0, 10.0, 0.0, [0, 10, 10, 20], [0.0] * 4), ('inPath', 0.0, 0.0))
    for first_step, expected in ((.099, 'unknown'), (.1, 'inPath')):
      self.assertEqual(path_relation(.05 - 1.52, 0.0, 10.0, 0.0, [0.0, first_step, 2.0], [0.0] * 3)[0], expected)
    unknown_paths = (([], []), ([0.0], [0.0]), ([0.0, 10.0], [0.0]), (list(range(34)), [0.0] * 34),
                     ([0.0, math.nan], [0.0] * 2), ([0.0, 10.0], [0.0, math.inf]),
                     ([0.0, 10.0], [0.0, 'bad']), ([0.0] * 33, [0.0] * 33),
                     ([0.0, math.nextafter(1.0, 0.0)], [0.0] * 2),
                     ([index / 32 for index in range(33)], [0.0] * 33))
    for path_x, path_y in unknown_paths:
      for previous in ('unknown', 'inPath'):
        with self.subTest(path_x=path_x, path_y=path_y, previous=previous):
          self.assertEqual(path_relation(.5 - 1.52, 0.0, 10.0, 0.0, path_x, path_y, previous), ('unknown', 0.0, 0.0))
    self.assertEqual(path_relation(.5 - 1.52, 0.0, 10.0, 0.0, [0.0, 1.0], [0.0] * 2), ('inPath', 0.0, 0.0))
    self.assertEqual(path_relation(28.48, 0.0, 10.0, 0.0, *straight), ('inPath', 0.0, 0.0))
    self.assertEqual(path_relation(30.0, 0.0, 10.0, 0.0, *straight), ('unknown', 0.0, 0.0))

  def test_path_is_prepared_once_per_publication_and_replaced_next_time(self):
    class PathCoordinate(float):
      pass
    checked = []
    def checked_finite(*values):
      if len(values) == 2 and all(isinstance(value, PathCoordinate) for value in values):
        checked.append(values)
      return finite(*values)
    fusion, sm = HondaRadarFusion(), Inputs(START, probability=0)
    straight = list(np.linspace(0.0, 160.0, 33))
    changes = ((straight, [0.0] * 33, 'inPath'), (straight, [5.0] * 33, 'outOfPath'),
               ([0.0] * 33, [0.0] * 33, 'unknown'), (straight, [0.0] * 33, 'inPath'))
    with patch('openpilot.selfdrive.car.honda_radar_fusion.finite', side_effect=checked_finite):
      sm['modelV2'].position.x[:] = [PathCoordinate(value) for value in straight]
      sm['modelV2'].position.y[:] = [PathCoordinate(0.0)] * 33
      fusion.update(body(0, []), sm, START + 1)
      self.assertEqual(checked, [])  # No eligible track needs no preparation.
      for frame, (path_x, path_y, expected) in enumerate(changes, 1):
        stamp = START + frame * STEP
        sm.logMonoTime.update(dict.fromkeys(sm, stamp))
        sm['modelV2'].timestampEof = stamp
        # Mutate the same model/path objects; object identity cannot justify reuse.
        sm['modelV2'].position.x[:] = [PathCoordinate(value) for value in path_x]
        sm['modelV2'].position.y[:] = [PathCoordinate(value) for value in path_y]
        observations = [observation(frame, slot=slot, oem=slot + 1, distance=20.0 + slot * .2) for slot in range(16)]
        before = len(checked)
        out = fusion.update(body(frame, observations), sm, stamp + 1)
        self.assertEqual(len(out['tracks']), 16)
        self.assertEqual(len(checked) - before, 33)
        self.assertTrue(all(track['pathState'] == expected for track in out['tracks']))

  def test_prepared_path_keeps_each_tracks_hysteresis(self):
    fusion = HondaRadarFusion()
    def targets(frame):
      return [observation(frame, slot=slot, oem=slot + 1, lateral=2.0, distance=20.0 + slot) for slot in range(2)]
    advance(fusion, 0, targets(0), probability=0)
    for track in fusion.tracks.values():
      track.path = 'inPath' if track.oem_id == 1 else 'unknown'
    out = advance(fusion, 1, targets(1), probability=0)
    states = {track['oemTrackId']: track['pathState'] for track in out['tracks']}
    self.assertEqual(states, {1: 'inPath', 2: 'outOfPath'})
    self.assertTrue(all(track['pathOffset'] == 2.0 for track in out['tracks']))
    self.assertFalse(out['controlAllowed'])
    self.assertFalse(out['calibrated'])

  def test_side_targets_are_tracked_without_creating_new_leads(self):
    fusion = HondaRadarFusion()
    for frame in range(8):
      out = advance(fusion, frame, [observation(frame, lateral=5)], probability=0)
    self.assertEqual(len(out['tracks']), 1)
    self.assertEqual(out['tracks'][0]['pathState'], 'outOfPath')
    self.assertTrue(all(not l['present'] for l in out['leads']))

  def test_visual_cut_in_is_not_delayed_for_radar_confirmation(self):
    fusion = HondaRadarFusion()
    for frame in range(10):
      out = advance(fusion, frame)
    self.assertGreater(out['leads'][0]['distanceWeight'], 0)
    out = advance(fusion, 10, distance=6.0, relative=-3.0)
    lead = out['leads'][0]
    self.assertTrue(lead['present'])
    self.assertEqual(lead['state'], 'visionOnly')
    self.assertEqual(lead['dRel'], 6.0)
    self.assertEqual(lead['vRel'], -3.0)

  def test_gradual_separation_is_rejected_before_blending_and_stays_latched(self):
    fusion = HondaRadarFusion()
    for frame in range(12):
      out = advance(fusion, frame, distance=20 - .076, x_std=2.5)
    self.assertGreater(out['leads'][0]['distanceWeight'], .5)
    # Every visual step is below the old 3 m continuity gate. The absolute
    # position gate still admits even the final, approximately 5.5 m bias.
    residuals = (.25, .5, .921, 1.544, 2.317, 3.1, 3.9, 4.739, 5.539)
    for frame, residual in enumerate(residuals, 12):
      out = advance(fusion, frame, distance=20 - residual, x_std=2.5)
      if residual >= .921:
        lead = out['leads'][0]
        self.assertIn(lead['identityState'], ('unresolved', 'different'))
        self.assertEqual(lead['reason'], 'identityRejected')
        self.assertGreater(lead['rejectionMonoTime'], 0)
        self.assertEqual(lead['distanceWeight'], 0)
        self.assertEqual(lead['velocityWeight'], 0)
        self.assertAlmostEqual(lead['dRel'], 20 - residual)
        self.assertEqual(lead['aLead'], 0)
    # Leaving the geometric gate must not pop the rejection and let the same
    # old, farther object earn a new confirmation timer on re-entry.
    advance(fusion, 21, distance=2.0, x_std=2.5)
    for frame in range(22, 42):
      out = advance(fusion, frame, distance=20 - residuals[-1], x_std=2.5)
      self.assertEqual(out['leads'][0]['distanceWeight'], 0)
      self.assertEqual(out['leads'][0]['velocityWeight'], 0)
      self.assertTrue(out['leads'][0]['present'])

  def test_unconfirmed_anchor_restarts_from_recorded_normal_control_residuals(self):
    # Extracted from a317f4c6's case-rows.jsonl.gz, modelIndex 0, 2026-10-06.
    # All three rejected controls retained bindingSamples=1. The initial pair
    # here represents the frozen bad anchor seen in each scoring window, not
    # the original birth coordinates (R261's anchor had already been rotated).
    # Subsequent residuals and model-message intervals are recorded values;
    # using those intervals as camera-source times isolates this state-machine
    # regression. This fixture is not raw-CAN replay or measurement truth.
    controls = {
      'R593': ((-5.7920886676197005, .02005371777141185), (
        (1493728633907, -3.924990357109465, .08904118491824085),
        (1493778743855, -4.5826750714371585, .0944010486147473),
        (1493879832657, -4.634486883059154, .060832681239691905))),
      'R261_V116': ((-4.729434305607998, -4.005135438324269), (
        (1328323633085, -6.071545733728303, .9131668062119549),
        (1328422421210, -6.064989164941629, .8581950641518272),
        (1328472049908, -5.974625120937443, .9042223332804173))),
      'normal_support_8a': ((2.9234684482028266, .9629951636801168), (
        (216954748978, -1.9295942370739212, .04203073164606558),
        (217005046687, -1.7848578559885961, .03191549312926778),
        (217105023249, -1.4408101636266153, .00021697739462706034))),
    }
    for name, (anchor, rows) in controls.items():
      with self.subTest(control=name):
        seed_ns = rows[0][0] - STEP
        binding, history = Binding(1, seed_ns), EgoHistory()
        history.add(Ego(seed_ns, 10.0, 0.0))
        binding.observe(seed_ns, seed_ns, 1, 20.0 - anchor[0], 0.0, *anchor, history)
        self.assertFalse(binding.anchor_ready)
        for sequence, (stamp, rx, ry) in enumerate(rows, 2):
          history.add(Ego(stamp, 10.0, 0.0))
          result = binding.observe(stamp, stamp, sequence, 20.0 - rx, 0.0, rx, ry, history)
          self.assertEqual(result['rejectionMonoTime'], 0)
          if sequence < 4:
            self.assertEqual(result['identityState'], 'unresolved')
            self.assertFalse(binding.anchor_ready)
        self.assertEqual(result['bindingSamples'], 3)
        self.assertEqual(result['identityState'], 'same')
        self.assertTrue(binding.anchor_ready)
        self.assertEqual(binding.start_ns, rows[0][0])

  def test_unconfirmed_instability_does_not_earn_or_permanently_veto_identity(self):
    binding, history = Binding(1, START), EgoHistory()
    for frame in range(12):
      stamp, residual = START + frame * STEP, 2.0 * (frame % 2)
      history.add(Ego(stamp, 10.0, 0.0))
      result = binding.observe(stamp, stamp, frame + 1, 20.0 - residual, 0.0, residual, 0.0, history)
      self.assertEqual(result['identityState'], 'unresolved')
      self.assertEqual(result['bindingSamples'], 1)
      self.assertFalse(binding.anchor_ready)
      self.assertEqual(result['rejectionMonoTime'], 0)

  def test_unconfirmed_pair_gap_restarts_three_sample_confirmation(self):
    binding, history = Binding(1, START), EgoHistory()
    for frame in range(2):
      stamp = START + frame * STEP
      history.add(Ego(stamp, 10.0, 0.0))
      binding.observe(stamp, stamp, frame + 1, 20.0, 0.0, 0.0, 0.0, history)
    binding.reject('noMatch', START + 2 * STEP)
    for frame in range(3, 6):
      stamp = START + frame * STEP
      history.add(Ego(stamp, 10.0, 0.0))
      result = binding.observe(stamp, stamp, frame + 1, 18.0, 0.0, 2.0, 0.0, history)
      self.assertEqual(result['rejectionMonoTime'], 0)
      self.assertEqual(result['identityState'], 'same' if frame == 5 else 'unresolved')
      self.assertEqual(result['bindingSamples'], frame - 2)
    self.assertEqual(binding.start_ns, START + 3 * STEP)

  def test_confirmed_gate_boundary_gap_reconfirms_recorded_controls(self):
    # First qualified losses in trace-7be99836: R593 |rx|=5.17799
    # exceeds gx=5.10759; R261 |ry|=1.59242 exceeds gy=1.5. Neither
    # departure contradicted its prior residual reference by .75/.5.
    # Fixed radar positions and zero yaw isolate that gate/state transition;
    # the residual numbers come from the trace, not measurement ground truth.
    controls = (
      ('R593', 42.5, (-4.931, -.097), (-5.177987768681298, -.11142307043087668),
       ((-4.158813367868902, -.13718354342055566), (-3.917114585820144, -.14760380910066095),
        (-3.762161462945656, -.13876966110687344)), 1.69, .6),
      ('R261', 35.0, (-5.045, -1.149), (-5.6776976448845105, -1.5924203804285284),
       ((-6.437005571497117, -1.4863352667095562), (-6.37742125499679, -1.4921873133359584),
        (-6.304464963696802, -1.42373376084993)), 2.6, .6),
    )
    for name, radar_x, initial, missed, resumed, sx, sy in controls:
      with self.subTest(control=name):
        fusion = HondaRadarFusion()
        for frame in range(12):
          out = advance(fusion, frame, [observation(frame, distance=radar_x)],
                        distance=radar_x - initial[0], lateral=-initial[1], x_std=sx, y_std=sy)
        self.assertGreater(out['leads'][0]['distanceWeight'], .5)
        out = advance(fusion, 12, [observation(12, distance=radar_x)],
                      distance=radar_x - missed[0], lateral=-missed[1], x_std=sx, y_std=sy)
        lead = out['leads'][0]
        self.assertEqual(lead['reason'], 'noMatch')
        self.assertLess(abs(lead['residualDeltaX']), .75)
        self.assertLess(abs(lead['residualDeltaY']), .5)
        self.assertEqual(lead['distanceWeight'], 0)
        for frame, (rx, ry) in enumerate(resumed, 13):
          out = advance(fusion, frame, [observation(frame, distance=radar_x)],
                        distance=radar_x - rx, lateral=-ry, x_std=sx, y_std=sy)
          lead = out['leads'][0]
          if frame < 15:
            self.assertEqual(lead['state'], 'associating')
            self.assertEqual(lead['distanceWeight'], 0)
            self.assertEqual(lead['velocityWeight'], 0)
          else:
            self.assertEqual(lead['identityState'], 'same')
            self.assertGreater(lead['distanceWeight'], 0)
          binding = fusion.bindings[(0, out['tracks'][0]['trackId'])]
          self.assertFalse(binding.separation_latched)
          self.assertEqual(lead['bindingSamples'], frame - 12)

  def test_out_of_gate_cut_in_keeps_old_pair_rejected(self):
    fusion = HondaRadarFusion()
    for frame in range(12):
      out = advance(fusion, frame, x_std=2.5)
    identifier = out['tracks'][0]['trackId']
    sample_count = out['leads'][0]['bindingSamples']
    # This jump skips the entire absolute association gate. Missing that gate
    # must not hide its explicit contradiction of the confirmed old pair.
    out = advance(fusion, 12, distance=6.0, x_std=2.5)
    self.assertEqual(out['leads'][0]['reason'], 'noMatch')
    self.assertEqual(out['leads'][0]['identityState'], 'different')
    self.assertEqual(out['leads'][0]['bindingSamples'], sample_count)
    self.assertTrue(fusion.bindings[(0, identifier)].separation_latched)
    for frame in range(13, 25):
      out = advance(fusion, frame, distance=16.0, x_std=2.5)
      self.assertEqual(out['leads'][0]['distanceWeight'], 0)
      self.assertEqual(out['leads'][0]['velocityWeight'], 0)
      self.assertEqual(out['leads'][0]['dRel'], 16.0)

  def test_ambiguous_match_checks_rejection_without_confirming_old_pair(self):
    fusion = HondaRadarFusion()
    for frame in range(12):
      advance(fusion, frame)
    old_binding = next(binding for (index, _), binding in fusion.bindings.items() if index == 0)
    count = len(old_binding.samples)
    samples = [observation(12), observation(12, distance=24.0, slot=1, oem=8),
               observation(12, distance=26.0, slot=2, oem=9)]
    out = advance(fusion, 12, samples, distance=25.0, x_std=2.5)
    self.assertEqual(out['leads'][0]['reason'], 'ambiguous')
    self.assertTrue(old_binding.separation_latched)
    self.assertEqual(len(old_binding.samples), count)
    self.assertEqual(out['leads'][0]['distanceWeight'], 0)

  def test_hard_rejection_survives_every_soft_gap(self):
    for reason in ('visionUnqualified', 'visionUncertainty', 'nearVision', 'noMatch', 'ambiguous',
                   'candidateChanged', 'sourceGap', 'sourceOrder'):
      with self.subTest(reason=reason):
        binding, history = Binding(1, START), EgoHistory()
        for frame in range(4):
          stamp = START + frame * STEP
          history.add(Ego(stamp, 10.0, 0.0))
          binding.observe(stamp, stamp, frame + 1, 20.0, 0.0, 0.0, 0.0, history)
        stamp = START + 4 * STEP
        history.add(Ego(stamp, 10.0, 0.0))
        binding.observe(stamp, stamp, 5, 19.0, 0.0, 1.0, 0.0, history)
        rejection = binding.rejection_ns
        self.assertEqual(binding.state, 'unresolved')
        self.assertTrue(binding.separation_latched)
        binding.reject(reason, stamp + 1)
        self.assertEqual(binding.rejection_ns, rejection)
        self.assertTrue(binding.anchor_ready)
        self.assertTrue(binding.separation_latched)
        for frame in range(5, 10):
          stamp = START + frame * STEP
          history.add(Ego(stamp, 10.0, 0.0))
          result = binding.observe(stamp, stamp, frame + 1, 18.0, 0.0, 2.0, 0.0, history)
          self.assertEqual(result['identityState'], 'different')
          self.assertEqual(result['rejectionMonoTime'], rejection)

  def test_source_time_discontinuity_never_creates_or_escalates_different(self):
    for reason in ('sourceGap', 'sourceOrder'):
      for already_rejected in (False, True):
        with self.subTest(reason=reason, already_rejected=already_rejected):
          binding, history = Binding(1, START), EgoHistory()
          for frame in range(5):
            stamp = START + frame * STEP
            residual = 1.0 if already_rejected and frame == 4 else 0.0
            history.add(Ego(stamp, 10.0, 0.0))
            binding.observe(stamp, stamp, frame + 1, 20.0 - residual, 0.0, residual, 0.0, history)
          last = stamp
          # R261's real gap was 200,006,979 ns: just beyond valid reference
          # continuity, yet the previous implementation created 'different'.
          source = last + 200_006_979 if reason == 'sourceGap' else last - 1
          now = max(source, last + 1)
          history.add(Ego(now, 10.0, 0.0))
          result = binding.observe(now, source, 6, 16.0, 0.0, 4.0, 0.0, history)
          self.assertEqual(result['identityState'], 'unresolved')
          self.assertFalse(result['independentPair'])
          self.assertEqual(result['recoverySamples'], 0)
          if already_rejected:
            self.assertEqual(result['identityReason'], 'referenceInvalid:' + reason)
          # Re-publication of that same invalid source cannot become fresh
          # confirmation merely because the validation clock moved forward.
          repeated = binding.observe(now + 1, source, 6, 16.0, 0.0, 4.0, 0.0, history)
          self.assertFalse(repeated['independentPair'])
          self.assertEqual(repeated['bindingSamples'], result['bindingSamples'])
          for step in range(1, 5):
            stamp = now + step * STEP
            history.add(Ego(stamp, 10.0, 0.0))
            result = binding.observe(stamp, stamp, 6 + step, 20.0, 0.0, 0.0, 0.0, history)
            if already_rejected:
              self.assertEqual(result['identityState'], 'unresolved')
              self.assertGreater(result['rejectionMonoTime'], 0)
              self.assertEqual(result['recoverySamples'], 0)
          if not already_rejected:
            self.assertEqual(result['identityState'], 'same')
            self.assertEqual(result['rejectionMonoTime'], 0)

  def test_rejection_only_checks_do_not_learn_confirm_or_recover(self):
    binding, history = Binding(1, START), EgoHistory()
    for frame in range(4):
      stamp = START + frame * STEP
      history.add(Ego(stamp, 10.0, 0.0))
      binding.observe(stamp, stamp, frame + 1, 20.0, 0.0, 0.0, 0.0, history)
    samples, anchor = tuple(binding.samples), (binding.anchor_x, binding.anchor_y)
    stamp = START + 4 * STEP
    history.add(Ego(stamp, 10.0, 0.0))
    result = binding.observe(stamp, stamp, 5, 19.8, 0.0, .2, 0.0, history, allow_confirmation=False)
    self.assertEqual(tuple(binding.samples), samples)
    self.assertEqual((binding.anchor_x, binding.anchor_y), anchor)
    self.assertFalse(result['independentPair'])
    # Refused inputs have still been seen: another publication cannot turn
    # them into a new confirmation or an extra recovery vote.
    repeated = binding.observe(stamp + 1, stamp, 5, 19.8, 0.0, .2, 0.0, history)
    self.assertFalse(repeated['independentPair'])
    self.assertEqual(tuple(binding.samples), samples)
    stamp += STEP
    history.add(Ego(stamp, 10.0, 0.0))
    binding.observe(stamp, stamp, 6, 18.0, 0.0, 2.0, 0.0, history)
    self.assertTrue(binding.separation_latched)
    for sequence in range(7, 11):
      stamp += STEP
      history.add(Ego(stamp, 10.0, 0.0))
      result = binding.observe(stamp, stamp, sequence, 20.0, 0.0, 0.0, 0.0, history, allow_confirmation=False)
      self.assertEqual(result['recoverySamples'], 0)
      self.assertGreater(result['rejectionMonoTime'], 0)

  def test_hard_rejection_survives_probability_and_candidate_changes(self):
    fusion = HondaRadarFusion()
    for frame in range(12):
      out = advance(fusion, frame)
    identifier = out['tracks'][0]['trackId']
    advance(fusion, 12, distance=18.0)
    advance(fusion, 13, [observation(13), observation(13, distance=30.0, slot=1, oem=8)], probability=0)
    out = advance(fusion, 14, distance=18.0)
    self.assertEqual(out['leads'][0]['distanceWeight'], 0)
    advance(fusion, 15, [observation(15), observation(15, distance=30.0, slot=1, oem=8)], distance=30.0)
    out = advance(fusion, 16, distance=18.0)
    self.assertEqual(out['leads'][0]['distanceWeight'], 0)
    self.assertEqual(out['leads'][0]['velocityWeight'], 0)
    self.assertTrue(fusion.bindings[(0, identifier)].separation_latched)

  def test_large_constant_same_vehicle_bias_and_slow_drift_remain_eligible(self):
    fusion = HondaRadarFusion()
    for frame in range(300):
      # A large but constant initial bias is allowed by the unchanged position
      # gate. Slow drift can teach only a compatible, already-qualified pair.
      bias = 4.0 + .025 * frame * STEP / 1e9
      out = advance(fusion, frame, distance=20 - bias, x_std=2.5)
      if frame > 8:
        self.assertEqual(out['leads'][0]['identityState'], 'same')
        self.assertGreater(out['leads'][0]['distanceWeight'], .5)
    self.assertGreater(out['leads'][0]['bindingAnchorX'], 4.2)
    self.assertLess(len(fusion.bindings), 4)
    self.assertLessEqual(out['leads'][0]['bindingSamples'], 16)

  def test_hypotheses_do_not_share_identity_confirmation_or_rejection(self):
    fusion = HondaRadarFusion()
    for frame in range(12):
      advance(fusion, frame)
    t = START + 12 * STEP
    sm = Inputs(t)
    sm['modelV2'].leadsV3[0].x = [18.0 + 1.52]
    out = fusion.update(body(12, [observation(12)]), sm, t + 1)
    self.assertEqual(out['leads'][0]['distanceWeight'], 0)
    self.assertEqual(out['leads'][0]['identityState'], 'different')
    self.assertGreater(out['leads'][1]['distanceWeight'], .5)
    self.assertEqual(out['leads'][1]['identityState'], 'same')

  def test_recovery_needs_new_position_and_camera_samples_not_publications(self):
    binding = Binding(1, START)
    history = EgoHistory()
    def pair(frame, seq, residual, source=None):
      timestamp = START + frame * STEP
      history.add(Ego(timestamp, 10.0, 0.0))
      return binding.observe(timestamp, timestamp if source is None else source, seq,
                             20 - residual, 0.0, residual, 0.0, history)
    for frame in range(4):
      pair(frame, frame + 1, 0.0)
    self.assertEqual(binding.state, 'same')
    pair(4, 5, 2.0)
    self.assertEqual(binding.state, 'different')
    first = pair(5, 6, 0.0)
    self.assertEqual(first['recoverySamples'], 1)
    # New publication/position IDs with the same camera source cannot confirm.
    duplicate = pair(6, 7, 0.0, source=START + 5 * STEP)
    self.assertFalse(duplicate['independentPair'])
    self.assertEqual(duplicate['recoverySamples'], 1)
    # A new camera frame reusing the same last paired radar sample also cannot.
    duplicate = pair(7, 6, 0.0)
    self.assertFalse(duplicate['independentPair'])
    self.assertEqual(duplicate['recoverySamples'], 1)
    pair(8, 8, 0.0)
    self.assertGreater(binding.rejection_ns, 0)
    recovered = pair(9, 9, 0.0)
    self.assertEqual(recovered['identityState'], 'same')
    self.assertEqual(binding.rejection_ns, 0)

  def test_recorded_recovery_uses_identity_bounds_without_learning_the_bias(self):
    # R593/model0, trace-d5ab6837: the first three independent pairs cover
    # only 100 ms. The fourth is inside the original .75/.5 identity bounds
    # and reaches 150 ms, but D cleared the three votes because it was outside
    # the narrower .35/.25 bias-learning region. These are recorded residual
    # deltas/source times/position IDs, not a claim of sensor accuracy.
    samples = (
      (1492833066637, 86644, .08631451803365138, .00657912068678997),
      (1492883066897, 86646, .10508108847867013, -.0030497208225145566),
      (1492933067574, 86648, .3465165134483009, -.01505616096365104),
      (1492983068303, 86650, -.727664360335393, .00597069772411854),
    )
    start = samples[0][0] - 5 * STEP
    binding, history = Binding(1, start), EgoHistory()
    for frame in range(4):
      stamp = start + frame * STEP
      history.add(Ego(stamp, 10.0, 0.0))
      binding.observe(stamp, stamp, frame + 1, 20.0, 0.0, 0.0, 0.0, history)
    stamp = start + 4 * STEP
    history.add(Ego(stamp, 10.0, 0.0))
    binding.observe(stamp, stamp, 5, 18.0, 0.0, 2.0, 0.0, history)
    self.assertTrue(binding.separation_latched)
    original_anchor = binding.anchor_x, binding.anchor_y
    for index, (stamp, seq, rx, ry) in enumerate(samples):
      history.add(Ego(stamp, 10.0, 0.0))
      result = binding.observe(stamp, stamp, seq, 20.0 - rx, 0.0, rx, ry, history)
      self.assertEqual((binding.anchor_x, binding.anchor_y), original_anchor)
      if index < 3:
        self.assertTrue(binding.separation_latched)
        self.assertEqual(result['identityState'], 'different')
        # Re-publication cannot provide the missing elapsed time or position.
        repeated = binding.observe(stamp + 1, stamp, seq, 20.0 - rx, 0.0, rx, ry, history)
        self.assertEqual(repeated['recoverySamples'], index + 1)
        self.assertFalse(repeated['independentPair'])
    self.assertEqual(result['identityState'], 'same')
    self.assertFalse(binding.separation_latched)
    self.assertEqual(result['rejectionMonoTime'], 0)
    self.assertGreaterEqual(samples[-1][0] - binding.start_ns, 120_000_000)
    # Continued eligible residuals outside the learning region cannot walk
    # the reference toward a previously rejected, farther target.
    for step in range(1, 5):
      stamp = samples[-1][0] + step * STEP
      history.add(Ego(stamp, 10.0, 0.0))
      binding.observe(stamp, stamp, 86650 + step, 19.4, 0.0, .6, 0.0, history)
      self.assertEqual((binding.anchor_x, binding.anchor_y), original_anchor)

  def test_short_missing_measurements_preserve_only_existing_compatible_pair(self):
    fusion = HondaRadarFusion()
    for frame in range(12):
      out = advance(fusion, frame)
    identifier = out['tracks'][0]['trackId']
    for frame in (12, 13):
      out = advance(fusion, frame, [])
      self.assertEqual(out['tracks'][0]['trackId'], identifier)
      self.assertGreater(out['leads'][0]['distanceWeight'], 0)
    self.assertEqual(out['leads'][0]['state'], 'coasting')
    out = advance(fusion, 14)
    self.assertEqual(out['tracks'][0]['trackId'], identifier)
    self.assertEqual(out['leads'][0]['identityState'], 'same')
    self.assertGreater(out['leads'][0]['distanceWeight'], .5)

    unbound = HondaRadarFusion()
    for frame in range(12):
      advance(unbound, frame, probability=0)
    advance(unbound, 12, [], probability=0)
    out = advance(unbound, 13, [])
    self.assertEqual(out['leads'][0]['reason'], 'coastingUnbound')
    self.assertEqual(out['leads'][0]['distanceWeight'], 0)
    self.assertEqual(unbound.bindings, {})

  def test_switch_to_unbound_coasting_candidate_suspends_the_old_pair(self):
    fusion = HondaRadarFusion()
    for frame in range(12):
      samples = [observation(frame)]
      if frame < 11:
        samples.append(observation(frame, distance=35.0, slot=1, oem=8))
      out = advance(fusion, frame, samples)
    self.assertGreater(out['leads'][0]['distanceWeight'], .5)
    # A remains freshly measured, while B's last position is 132 ms old.
    # The visual hypothesis briefly moves to B, then back to A one tick later.
    out = advance(fusion, 12, distance=35.0)
    self.assertEqual(out['leads'][0]['reason'], 'coastingUnbound')
    self.assertEqual(out['leads'][0]['distanceWeight'], 0)
    for frame in (13, 14):
      out = advance(fusion, frame)
      self.assertEqual(out['leads'][0]['state'], 'associating')
      self.assertEqual(out['leads'][0]['identityState'], 'unresolved')
      self.assertEqual(out['leads'][0]['bindingSamples'], frame - 12)
      self.assertEqual(out['leads'][0]['distanceWeight'], 0)
      self.assertEqual(out['leads'][0]['velocityWeight'], 0)
    out = advance(fusion, 15)
    self.assertEqual(out['leads'][0]['identityState'], 'same')
    self.assertGreater(out['leads'][0]['distanceWeight'], 0)

  def test_common_turn_and_deceleration_do_not_look_like_a_new_identity(self):
    for turning in (False, True):
      with self.subTest(turning=turning):
        fusion = HondaRadarFusion()
        position = np.array([20.0, 0.0])
        target_velocity = np.array([10.0, 0.0])
        dt, yaw = STEP / 1e9, .15 if turning else 0.0
        angle = -yaw * dt
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for frame in range(28):
          if frame:
            position = rotation @ (position + (target_velocity - np.array([10.0, 0.0])) * dt)
            target_velocity = rotation @ target_velocity
            if not turning:
              target_velocity[0] -= .5 * dt
          x, y = map(float, position)
          relative = float(target_velocity[0]) - 10.0
          t = START + frame * STEP
          sm = Inputs(t, distance=x, lateral=y, relative=relative, yaw=yaw)
          sm['modelV2'].position.y = [-y / x * value for value in sm['modelV2'].position.x]
          out = fusion.update(body(frame, [observation(frame, distance=x, lateral=y, relative=relative)]), sm, t + 1)
          if frame > 8:
            self.assertEqual(out['leads'][0]['identityState'], 'same')
            self.assertGreater(out['leads'][0]['distanceWeight'], .5)

  def test_nonzero_measurement_bias_with_shared_turn_preserves_only_compatible_identity(self):
    # Both vehicles turn at the same angular rate with a fixed separation
    # angle on a 50 m radius. Radar and vision refer to points 4 m apart along
    # the lead's heading, giving a constant nonzero ego-frame measurement bias.
    # Treating that bias as a world-fixed vector would manufacture separation.
    speed, yaw, radius, separation_angle = 5.0, .1, 50.0, .25
    radar_x = radius * float(np.sin(separation_angle))
    radar_y = radius * (1.0 - float(np.cos(separation_angle)))
    bias_x, bias_y = 4.0 * float(np.cos(separation_angle)), 4.0 * float(np.sin(separation_angle))
    radar_relative = -yaw * radar_y
    vision_relative = -yaw * (radar_y + bias_y)
    for change in ('gradual', 'singleStep'):
      with self.subTest(change=change):
        fusion = HondaRadarFusion()
        def sample(frame, closer=0.0):
          stamp = START + frame * STEP
          sm = Inputs(stamp, distance=radar_x + bias_x - closer, lateral=radar_y + bias_y,
                      relative=vision_relative, speed=speed, yaw=yaw, x_std=2.0, y_std=.6)
          sm['modelV2'].position.x = list(np.linspace(0.0, 40.0, 33))
          sm['modelV2'].position.y = [-(radius - float(np.sqrt(radius * radius - x * x)))
                                    for x in sm['modelV2'].position.x]
          raw = observation(frame, distance=radar_x, lateral=radar_y, relative=radar_relative)
          return fusion.update(body(frame, [raw]), sm, stamp + 1)
        for frame in range(72):
          out = sample(frame)
          if frame > 8:
            self.assertEqual(out['leads'][0]['identityState'], 'same')
            self.assertGreater(out['leads'][0]['distanceWeight'], .5)
        self.assertLess(out['leads'][0]['bindingAnchorX'], -3.5)
        self.assertLess(out['leads'][0]['bindingAnchorY'], -.7)
        # The coordinate choice cannot pardon a real change in the residual,
        # either as a series of small steps or as one abrupt visual switch.
        shifts = (.2, .4, .6, .8, 1.0, 1.2) if change == 'gradual' else (3.0,)
        for frame, shift in enumerate(shifts, 72):
          out = sample(frame, shift)
          if shift >= 1.0:
            self.assertIn(out['leads'][0]['identityState'], ('unresolved', 'different'))
            self.assertEqual(out['leads'][0]['distanceWeight'], 0)
            self.assertEqual(out['leads'][0]['velocityWeight'], 0)
            self.assertTrue(out['leads'][0]['present'])

  def test_zero_near_negative_ground_speed_preserves_static_lead(self):
    fusion = HondaRadarFusion()
    for frame in range(12):
      distance = 12 - .5 * frame * STEP / 1e9
      out = advance(fusion, frame, [observation(frame, distance=distance, relative=-.5)],
                    distance=distance, speed=.3, relative=-.3, stopped_path=True)
    self.assertTrue(out['leads'][0]['present'])
    self.assertEqual(out['leads'][0]['vLead'], 0)
    self.assertGreater(out['leads'][0]['distanceWeight'], 0)

  def test_opposing_motion_does_not_replace_or_delete_visual_lead(self):
    fusion = HondaRadarFusion()
    for frame in range(14):
      distance = 35 - 12 * frame * STEP / 1e9
      out = advance(fusion, frame, [observation(frame, distance=distance, relative=-12)],
                    distance=distance, speed=5, relative=0)
    self.assertEqual(out['tracks'][0]['motion'], 'oncoming')
    self.assertEqual(out['leads'][0]['state'], 'visionOnly')
    self.assertTrue(out['leads'][0]['present'])
    self.assertEqual(out['leads'][0]['vLead'], 5)

  def test_bad_u10_and_absent_velocity_keep_radar_distance(self):
    for kwargs in ({'u10': 800}, {'velocity': False}, {'relative': -13.5}):
      with self.subTest(kwargs=kwargs):
        fusion = HondaRadarFusion()
        for frame in range(12):
          out = advance(fusion, frame, [observation(frame, **kwargs)])
        self.assertTrue(out['leads'][0]['present'])
        self.assertEqual(out['leads'][0]['state'], 'radarDistance')
        self.assertGreater(out['leads'][0]['distanceWeight'], 0)
        self.assertEqual(out['leads'][0]['velocityWeight'], 0)
        self.assertEqual(out['leads'][0]['vRel'], 0)

  def test_lateral_quality_exports_projected_covariance_without_borrowing_aux_quality(self):
    for lateral_position, lateral_velocity, offset in ((.75, 2.5, 10_000_000), (0.0, 0.0, 0), (8.0, 9.0, 0)):
      with self.subTest(position_variance=lateral_position, velocity_variance=lateral_velocity, offset=offset):
        fusion = HondaRadarFusion()
        for frame in range(4):
          advance(fusion, frame, [observation(frame, velocity=False)], probability=0)
        track = fusion.tracks[7]
        # Prepare the controlled covariance at a new model epoch. Reusing the
        # last publication's now_ns would correctly reset the entire producer.
        epoch = START + 4 * STEP
        self.assertTrue(track.advance(epoch, fusion.history))
        track.covariance = np.diag([.125, lateral_position, .0625, lateral_velocity])
        stamp = epoch + offset
        _, expected = track.project(stamp, fusion.history)
        resets = fusion.counts['resets']
        out = fusion.update(body(4, []), Inputs(stamp, probability=0), stamp + 1)
        self.assertEqual(out['counters']['resets'], resets)
        self.assertIs(fusion.tracks[7], track)
        snapshot = out['tracks'][0]
        self.assertEqual(snapshot['lateralPositionVariance'], expected[1, 1])
        self.assertEqual(snapshot['lateralVelocityVariance'], expected[3, 3])
        self.assertNotEqual(snapshot['lateralPositionVariance'], snapshot['positionVariance'])
        self.assertNotEqual(snapshot['lateralVelocityVariance'], snapshot['velocityVariance'])
        self.assertFalse(snapshot['velocityGood'])
        self.assertTrue(snapshot['lateralVelocityValid'])
        self.assertFalse(out['calibrated'])

  def test_lateral_quality_requires_three_independent_positions_and_120_ms(self):
    for span in (119_999_999, 120_000_000):
      with self.subTest(span=span):
        fusion = HondaRadarFusion()
        for frame, offset in enumerate((0, 60_000_000, span)):
          stamp = START + offset
          sample = source_observation(stamp, 100 * frame + 1, frame, 1 + 2 * frame, velocity=False)
          out = fusion.update(body(frame, [sample]), Inputs(stamp, probability=0), stamp + 1)
          snapshot = out['tracks'][0]
          self.assertEqual(snapshot['observations'], frame + 1)
          self.assertEqual(snapshot['lateralVelocityValid'], frame == 2 and span >= 120_000_000)

  def test_aux_updates_cannot_create_or_refresh_lateral_position_history(self):
    fusion = HondaRadarFusion()
    advance(fusion, 0)
    original = observation(0)
    for frame in (1, 2):
      auxiliary = {**original, 'velocitySeq': 1000 + frame, 'velocityMonoTime': START + frame * 20_000_000}
      out = advance(fusion, frame, [auxiliary], probability=0)
      snapshot = out['tracks'][0]
      self.assertEqual(snapshot['positionMonoTime'], START)
      self.assertEqual(snapshot['observations'], 1)
      self.assertEqual(len(fusion.tracks[7].positions), 1)
      self.assertFalse(snapshot['lateralVelocityValid'])
    for frame in (3, 4):
      out = advance(fusion, frame, probability=0)
      self.assertEqual(out['tracks'][0]['lateralVelocityValid'], frame == 4)
    latest = observation(4)
    auxiliary = {**latest, 'velocitySeq': 2000, 'velocityMonoTime': START + 4 * STEP + 33_000_000}
    out = advance(fusion, 5, [auxiliary], probability=0)
    self.assertTrue(out['tracks'][0]['lateralVelocityValid'])
    self.assertEqual(out['tracks'][0]['positionMonoTime'], latest['positionMonoTime'])
    self.assertEqual(len(fusion.tracks[7].positions), 3)
    self.assertEqual(advance(fusion, 8, [auxiliary], probability=0)['tracks'], [])

  def test_invalid_lateral_covariance_never_grants_prediction_quality(self):
    for axis in (1, 3):
      for value in (-.01, math.nan, math.inf, -math.inf):
        with self.subTest(axis=axis, value=value):
          fusion = HondaRadarFusion()
          for frame in range(3):
            advance(fusion, frame, probability=0)
          track = fusion.tracks[7]
          stamp = START + 3 * STEP
          self.assertTrue(track.advance(stamp, fusion.history))
          # Inject at the new epoch so process noise cannot replace the
          # requested invalid covariance, and chronology cannot cause a reset.
          track.covariance = np.diag([.25, .5, 1.0, 2.0])
          track.covariance[axis, axis] = value
          resets = fusion.counts['resets']
          with np.errstate(invalid='ignore'):
            out = fusion.update(body(3, []), Inputs(stamp, probability=0), stamp + 1)
          self.assertEqual(out['counters']['resets'], resets)
          self.assertIs(fusion.tracks[7], track)
          if math.isfinite(value):
            self.assertTrue(out['tracks'][0]['active'])
            self.assertFalse(out['tracks'][0]['lateralVelocityValid'])
          else:
            self.assertEqual(out['tracks'], [])

  def test_distinct_ambiguous_ids_are_not_averaged(self):
    fusion = HondaRadarFusion()
    for frame in range(8):
      out = advance(fusion, frame, [observation(frame, oem=7, lateral=.2), observation(frame, oem=8, slot=1, lateral=-.2)])
    self.assertEqual(len(out['tracks']), 2)
    self.assertEqual(out['leads'][0]['reason'], 'ambiguous')
    self.assertEqual(out['leads'][0]['state'], 'visionOnly')
    self.assertEqual(out['leads'][0]['distanceWeight'], 0)

  def test_position_jump_rejected_and_no_extrapolation_past_ttl(self):
    fusion = HondaRadarFusion()
    for frame in range(8):
      out = advance(fusion, frame)
    accepted = out['counters']['positions']
    out = advance(fusion, 8, [observation(8, distance=90)])
    self.assertEqual(out['counters']['positions'], accepted)
    self.assertGreater(out['counters']['rejectedPosition'], 0)
    for frame in range(9, 15):
      out = advance(fusion, frame, [])
    self.assertEqual(out['tracks'], [])
    self.assertTrue(out['leads'][0]['present'])
    self.assertEqual(out['leads'][0]['state'], 'visionOnly')

  def test_aux_never_extends_lifetime_or_restores_invalidated_position(self):
    fusion = HondaRadarFusion()
    for frame in range(6):
      out = advance(fusion, frame)
    old = observation(5)
    invalid = {'slot': 0, 'positionValid': False, 'positionMonoTime': old['positionMonoTime'], 'reason': 'invalidGeometry'}
    for frame in range(6, 11):
      out = advance(fusion, frame, [invalid, old])
    self.assertEqual(out['counters']['positions'], 6)
    self.assertEqual(out['tracks'], [])

  def test_explicit_withdrawal_revokes_all_components_before_age_decay_and_requires_new_confirmation(self):
    for reason in ('invalidGeometry', 'conflictingFrame', 'expired', 'withdrawn'):
      with self.subTest(reason=reason):
        fusion = HondaRadarFusion()
        for frame in range(12):
          out = advance(fusion, frame)
        old = observation(11)
        self.assertTrue(out['tracks'][0]['lateralVelocityValid'])
        old_identifier = out['tracks'][0]['trackId']
        invalid = {'slot': 0, 'positionValid': False, 'positionMonoTime': old['positionMonoTime'],
                   'positionSeq': old['positionSeq'], 'reason': reason}
        out = advance(fusion, 12, [invalid])
        track, lead = out['tracks'][0], out['leads'][0]
        self.assertFalse(track['active'])
        self.assertFalse(track['velocityGood'])
        self.assertFalse(track['lateralVelocityValid'])
        self.assertEqual(track['lastInvalidReason'], reason)
        self.assertEqual(track['lastInvalidMonoTime'], START + 12 * STEP + 1)
        self.assertEqual(track['positionMonoTime'], old['positionMonoTime'])
        self.assertTrue(lead['present'])
        self.assertEqual((lead['distanceWeight'], lead['velocityWeight'], lead['aLead']), (0, 0, 0))
        # A later AUX is still tied to the explicitly invalid old position.
        auxiliary = {**old, 'velocitySeq': old['velocitySeq'] + 1000,
                     'velocityMonoTime': START + 12 * STEP}
        out = advance(fusion, 13, [auxiliary])
        self.assertFalse(out['tracks'][0]['active'])
        self.assertFalse(out['tracks'][0]['lateralVelocityValid'])
        self.assertEqual(out['leads'][0]['distanceWeight'], 0)
        out = advance(fusion, 14)
        self.assertNotEqual(out['tracks'][0]['trackId'], old_identifier)
        self.assertEqual(out['tracks'][0]['observations'], 1)
        self.assertFalse(out['tracks'][0]['lateralVelocityValid'])
        self.assertEqual(out['tracks'][0]['lastInvalidReason'], reason)
        self.assertEqual(out['leads'][0]['distanceWeight'], 0)
        out = advance(fusion, 15)
        self.assertEqual(out['leads'][0]['distanceWeight'], 0)
        self.assertFalse(out['tracks'][0]['lateralVelocityValid'])
        out = advance(fusion, 16)
        self.assertTrue(out['tracks'][0]['mature'])
        self.assertTrue(out['tracks'][0]['lateralVelocityValid'])
        self.assertGreater(out['leads'][0]['distanceWeight'], 0)

  def test_conflicting_duplicate_position_immediately_revokes_the_posterior(self):
    fusion = HondaRadarFusion()
    for frame in range(12):
      advance(fusion, frame)
    duplicate = observation(11)
    duplicate['dRel'] += .05712
    out = advance(fusion, 12, [duplicate])
    self.assertEqual(out['counters']['positions'], 12)
    self.assertFalse(out['tracks'][0]['active'])
    self.assertEqual(out['tracks'][0]['lastInvalidReason'], 'conflictingPosition')
    self.assertEqual(out['leads'][0]['distanceWeight'], 0)
    self.assertEqual(out['leads'][0]['velocityWeight'], 0)

  def test_source_diagnostics_describe_the_actual_identity_and_weight_operands(self):
    fusion = HondaRadarFusion()
    for frame in range(12):
      out = advance(fusion, frame, distance=19.8)
    lead = out['leads'][0]
    self.assertEqual(lead['candidateTrackId'], lead['trackId'])
    self.assertAlmostEqual(lead['residualDeltaX'], lead['residualX'] - lead['bindingAnchorX'])
    self.assertAlmostEqual(lead['residualDeltaY'], lead['residualY'] - lead['bindingAnchorY'])
    self.assertLess(abs(lead['residualX']), lead['gateX'])
    self.assertLess(abs(lead['residualY']), lead['gateY'])
    self.assertLess(abs(lead['residualV']), lead['velocityGate'])
    self.assertEqual(lead['blend'], 1)
    self.assertEqual(lead['coastWeight'], 1)
    self.assertEqual(out['algorithmVersion'], 'bosch-a-shadow-3')
    self.assertFalse(out['controlAllowed'])
    self.assertFalse(out['calibrated'])

  def test_missing_future_and_stale_dependencies_clear_candidates(self):
    for kind in ('modelStale', 'egoStale', 'futureCamera', 'inputsInvalid', 'scopeInactive', 'schemaUnknown', 'badEgo', 'badYaw'):
      with self.subTest(kind=kind):
        fusion = HondaRadarFusion()
        for frame in range(5):
          advance(fusion, frame)
        t = START + 5 * STEP
        sm = Inputs(t)
        packet = body(5, [])
        if kind == 'modelStale':
          sm.logMonoTime['modelV2'] -= 1_000_000_000
        elif kind == 'egoStale':
          sm.logMonoTime['carState'] -= 1_000_000_000
        elif kind == 'futureCamera':
          sm['modelV2'].timestampEof += 1
        elif kind == 'inputsInvalid':
          sm.valid['modelV2'] = False
        elif kind == 'scopeInactive':
          packet['scope']['decActive'] = False
        elif kind == 'badEgo':
          sm['carState'].vEgo = float('nan')
        elif kind == 'badYaw':
          sm['modelV2'].orientationRate.z = [float('nan')]
        else:
          packet['schemaVersion'] = 2
        out = fusion.update(packet, sm, t + 2)
        self.assertFalse(out['active'])
        self.assertFalse(out['controlAllowed'])
        self.assertEqual(out['tracks'], [])
        self.assertEqual(out['leads'], [])

  def test_scope_gap_and_session_change_do_not_inherit_track_confirmation(self):
    fusion = HondaRadarFusion()
    for frame in range(6):
      out = advance(fusion, frame)
    old_id = out['tracks'][0]['trackId']
    t = START + 6 * STEP
    out = fusion.update(body(6, [observation(6)], sessionId='new-session'), Inputs(t), t + 1)
    self.assertNotEqual(out['tracks'][0]['trackId'], old_id)
    self.assertEqual(out['tracks'][0]['observations'], 1)
    self.assertEqual(out['leads'][0]['distanceWeight'], 0)

  def test_ego_rotation_projection_and_covariance_remain_finite(self):
    fusion = HondaRadarFusion()
    advance(fusion, 0, speed=0, yaw=1)
    track = next(iter(fusion.tracks.values()))
    track.mean[:] = [20, 0, 0, 0]
    fusion.history.samples.clear()
    fusion.history.add(Ego(START, 0, 1))
    mean, covariance = track.project(START + 100_000_000, fusion.history)
    self.assertAlmostEqual(mean[0], 20 * np.cos(.1), places=7)
    self.assertAlmostEqual(mean[1], -20 * np.sin(.1), places=7)
    self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), 0)

  def test_projection_operators_keep_current_state_and_prediction_time(self):
    track, history = projection_fixture()
    cache = {}
    for offset in (0, -50_000_000, 40_000_000, 80_000_000, 40_000_000):
      stamp = START + offset
      cached, fresh = track.project(stamp, history, cache), track.project(stamp, history)
      self.assertEqual(tuple(value.tobytes() for value in cached), tuple(value.tobytes() for value in fresh))
    stamp = START + 40_000_000
    self.assertTrue(track.advance(stamp, history, cache))
    previous = track.project(stamp, history, cache)
    for index, value, variance in ((0, 23.0, .25), (2, 9.0, .1)):
      # Same target/epoch and operators, but position and AUX change the posterior.
      track.measure(index, value, variance)
      cached, fresh = track.project(stamp, history, cache), track.project(stamp, history)
      self.assertEqual(tuple(value.tobytes() for value in cached), tuple(value.tobytes() for value in fresh))
      self.assertNotEqual(cached[0].tobytes(), previous[0].tobytes())
      self.assertNotEqual(cached[1].tobytes(), previous[1].tobytes())
      previous = cached
    other = copy.deepcopy(track)
    other.identifier = 2
    other.mean += np.array([3.0, 2.0, -1.0, .5])
    other.covariance *= .8
    cached, fresh = other.project(stamp, history, cache), other.project(stamp, history)
    self.assertEqual(tuple(value.tobytes() for value in cached), tuple(value.tobytes() for value in fresh))
    self.assertNotEqual(cached[0].tobytes(), previous[0].tobytes())
    self.assertEqual(track.epoch_ns, stamp)
    self.assertEqual(track.position_ns, START)

  def test_projection_cache_has_no_posterior_alias_and_checks_ego_again(self):
    track, history = projection_fixture()
    cache, stamp = {}, START + 80_000_000
    state = track.mean.tobytes(), track.covariance.tobytes()
    first_prediction = track.project(stamp, history, cache)
    first_prediction[0][:] = 999.0
    first_prediction[1][:] = -999.0
    second, fresh = track.project(stamp, history, cache), track.project(stamp, history)
    self.assertEqual(tuple(value.tobytes() for value in second), tuple(value.tobytes() for value in fresh))
    self.assertEqual((track.mean.tobytes(), track.covariance.tobytes()), state)
    for predicted in second:
      self.assertFalse(np.shares_memory(predicted, track.mean))
      self.assertFalse(np.shares_memory(predicted, track.covariance))
      self.assertTrue(all(not np.shares_memory(predicted, operand) for operators in cache.values() for operand in operators[1:]))
    for speed, yaw in ((10.0, 0.0), (10.0, -0.0), (7.0, -.25)):
      replacement = EgoHistory()
      replacement.add(Ego(START, speed, yaw))
      cached, fresh = track.project(stamp, replacement, cache), track.project(stamp, replacement)
      self.assertEqual(tuple(value.tobytes() for value in cached), tuple(value.tobytes() for value in fresh))
    self.assertNotEqual(cached[0].tobytes(), second[0].tobytes())
    history.samples.clear()
    self.assertIsNone(track.project(stamp, history, cache))
    self.assertIsNone(track.project(START + 500_000_001, replacement, cache))

  def test_projection_operators_are_shared_only_within_one_publication(self):
    fusion = HondaRadarFusion()
    def publish(frame, model_offset, can_offset, speed, yaw):
      stamp = START + model_offset
      observations = []
      if can_offset is not None:
        observations = [observation(frame, slot=slot, oem=slot + 1, distance=20.0 + slot * .2) for slot in range(16)]
        for value in observations:
          value['positionMonoTime'] = value['velocityMonoTime'] = START + can_offset
      return fusion.update(body(frame, observations), Inputs(stamp, probability=0, speed=speed, yaw=yaw), stamp + 1)
    publish(0, 0, 0, 10.0, 0.0)
    calls, publication_caches = [], []
    original = Track.project
    def project(track, target_ns, history, projection_cache=None):
      calls.append(projection_cache)
      return original(track, target_ns, history, projection_cache)
    cases = ((100_000_000, 66_666_666, 10.0, 0.0, 3, 48),
             (150_000_000, 133_333_333, 10.2, .02, 3, 48),
             (200_000_000, 200_000_000, 10.4, -.01, 2, 48),
             (250_000_000, None, 10.6, .03, 1, 16))
    last_position = START
    with patch.object(Track, 'project', project):
      for frame, (model_offset, can_offset, speed, yaw, constructions, predictions) in enumerate(cases, 1):
        calls.clear()
        with patch.object(np, 'diag', wraps=np.diag) as diagonal:
          out = publish(frame, model_offset, can_offset, speed, yaw)
        self.assertEqual(diagonal.call_count, constructions)
        self.assertEqual(len(calls), predictions)
        cache = calls[0]
        self.assertIsNotNone(cache)
        self.assertTrue(all(value is cache for value in calls))
        self.assertTrue(all(cache is not prior for prior in publication_caches))
        publication_caches.append(cache)
        last_position = START + can_offset if can_offset is not None else last_position
        self.assertTrue(all(track.epoch_ns == last_position and track.position_ns == last_position for track in fusion.tracks.values()))
        self.assertEqual(len(out['tracks']), 16)
        self.assertFalse(out['controlAllowed'])
        self.assertFalse(out['calibrated'])

  def test_acceleration_uses_velocity_updates_and_survives_ego_frame_rotation(self):
    fusion = HondaRadarFusion()
    for frame in range(40):
      t = frame * STEP / 1e9
      distance, relative = 20 + t * t, 2 * t
      out = advance(fusion, frame, [observation(frame, distance=distance, relative=relative)],
                    distance=distance, relative=relative)
    track = next(iter(fusion.tracks.values()))
    self.assertGreater(track.acceleration, 1.4)
    self.assertLess(track.acceleration, 2.4)
    self.assertGreater(out['leads'][0]['aLead'], .8)
    # No actual world acceleration: rotating coordinates must not manufacture it.
    track.acceleration = 0
    track.acceleration_ns = START
    track.acceleration_velocity = (10, 5)
    fusion.history.samples.clear()
    fusion.history.add(Ego(START, 10, .5))
    angle = -.05
    track.mean[2:] = [np.cos(angle) * 10 - np.sin(angle) * 5, np.sin(angle) * 10 + np.cos(angle) * 5]
    track.measure_acceleration(START + 100_000_000, fusion.history)
    self.assertAlmostEqual(track.acceleration, 0)

  def test_capacity_bounds_and_future_position_are_fail_closed(self):
    fusion = HondaRadarFusion()
    observations = [observation(0, oem=i, slot=i % 16) for i in range(1, 64)]
    out = advance(fusion, 0, observations)
    self.assertLessEqual(len(out['tracks']), MAX_TRACKS)
    self.assertLessEqual(len(fusion.tracks), MAX_TRACKS)
    self.assertFalse(out['controlAllowed'])
    future = observation(1)
    future['positionMonoTime'] += 1_000_000_000
    accepted = out['counters']['positions']
    out = advance(fusion, 1, [future])
    self.assertEqual(out['counters']['positions'], accepted)


if __name__ == '__main__':
  unittest.main()
