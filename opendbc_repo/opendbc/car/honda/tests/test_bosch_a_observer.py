import unittest
from dataclasses import replace
from itertools import permutations
from types import SimpleNamespace
from unittest.mock import patch

from opendbc.can.dbc import DBC
from opendbc.can.packer import CANPacker
from opendbc.can.parser import get_raw_value
from opendbc.car.honda.bosch_a_radar import (ADDRESS_MAP, AUX_IDS, DBC_NAME, MAIN_IDS, MAX_AGE_NS, MAX_EVENTS, RAW_SIGNALS,
                                          BoschARadarObserver, radar_scan_continuous)

START = 10_000_000_000


def scan_frames(slot=0, index=0, life=1, distance=1000, oem_id=None, velocity=736, ratio=500, bus=2, azimuth=1124):
  packer = CANPacker('honda_bosch_a_radar')
  values = [dict(STATUS=7, RANGE_RAW=distance, AZIMUTH_RAW=azimuth, RANGE_SIGMA_RAW=1),
            dict(OBJECT_EXISTENCE_PROBABILITY_RAW=126), dict(LIFECYCLE_RAW=life),
            dict(TRACK_ID=slot + 1 if oem_id is None else oem_id),
            dict(REL_VELOCITY_RAW=velocity, REL_VELOCITY_UNCERTAINTY_RAW=0, RANGE_RATIO_RAW=ratio)]
  return [packer.make_can_msg(addr, bus, dict(v, FRAME_IDX=index)) for addr, v in zip((*MAIN_IDS[slot], AUX_IDS[slot]), values, strict=True)]


class TestBoschADecodeFields(unittest.TestCase):
  def test_all_dbc_fields_match_raw_getter_on_bit_basis(self):
    # All 352 used fields (including FRAME_IDX): zero, all ones and every
    # individual payload bit. The reference getter remains independent of the
    # compiled shifts, and both accepted byte container types exercise feed.
    dbc = DBC(DBC_NAME)
    decoder = BoschARadarObserver(2)
    payloads = (bytes(8), b'\xff' * 8, *((1 << bit).to_bytes(8, 'big') for bit in range(64)))
    for address, (slot, part) in ADDRESS_MAP.items():
      signals = dbc.msgs[address].sigs
      for payload in payloads:
        index = get_raw_value(payload, signals['FRAME_IDX'])
        expected = {key: get_raw_value(payload, signals[name]) for key, name in RAW_SIGNALS[part].items()}
        for data_type in (bytes, bytearray):
          with self.subTest(address=hex(address), payload=payload.hex(), data_type=data_type.__name__):
            decoder.reset()
            decoder.feed(START, [(address, data_type(payload), 2)])
            self.assertEqual(list(decoder.slots[slot].scans), [index])
            frame = decoder.slots[slot].scans[index].frames[part]
            self.assertEqual(frame.values, expected)
            self.assertEqual(frame.data, payload)
            self.assertIsInstance(frame.data, bytes)

  def test_every_frame_index_in_every_slot(self):
    decoder = BoschARadarObserver(2)
    for slot in range(16):
      for index in range(16):
        with self.subTest(slot=slot, index=index):
          decoder.reset()
          frames = scan_frames(slot=slot, index=index)
          decoder.feed(START, [(a, bytearray(d) if index & 1 else d, b) for a, d, b in frames])
          events = decoder.drain()
          self.assertEqual([e['frameIndex'] for e in events], [index, index])
          self.assertEqual(set(decoder.slots[slot].scans[index].frames), set(range(5)))
          self.assertTrue(events[-1]['positionValid'] and events[-1]['velocityValid'])

  def test_unexpected_dbc_layouts_preserve_original_raw_semantics(self):
    dbc = DBC(DBC_NAME)
    address = MAIN_IDS[0][0]
    original = dbc.msgs[address]
    cases = {'message_size': replace(original, size=7)}
    for name in ('FRAME_IDX', 'RANGE_RAW'):
      signal = original.sigs[name]
      signed = replace(signal, is_signed=True)
      little_endian = replace(signal, start_bit=0, msb=signal.size - 1, lsb=0, is_little_endian=True)
      for label, changed in (('signed', signed), ('little_endian', little_endian)):
        cases[f'{label}_{name}'] = replace(original, sigs=original.sigs | {name: changed})
    for label, message in cases.items():
      with self.subTest(layout=label):
        # Do not mutate the cached DBC shared by other tests/observers.
        with patch('opendbc.car.honda.bosch_a_radar.DBC', return_value=SimpleNamespace(msgs=dbc.msgs | {address: message})):
          decoder = BoschARadarObserver(2)
        for payload in (b'\xff' * 8, bytes.fromhex('a7c13d59f06b824e')):
          decoder.reset()
          with patch('opendbc.car.honda.bosch_a_radar.get_raw_value', wraps=get_raw_value) as getter:
            decoder.feed(START, [(address, bytearray(payload), 2)])
          self.assertEqual(getter.call_count, 1 + len(RAW_SIGNALS[0]))
          index = get_raw_value(payload, message.sigs['FRAME_IDX'])
          frame = decoder.slots[0].scans[index].frames[0]
          self.assertEqual(frame.values, {key: get_raw_value(payload, message.sigs[name]) for key, name in RAW_SIGNALS[0].items()})

  def test_wrong_lengths_are_filtered_before_decoding(self):
    decoder = BoschARadarObserver(2)
    lengths = (*range(8), 9, 64)
    frames = [(MAIN_IDS[0][0], bytearray(size), 2) for size in lengths]
    decoder.feed(START, [(MAIN_IDS[0][0], bytes(7), 0), (0x100, bytes(7), 2), *frames])
    self.assertEqual(decoder.counts['frames'], len(lengths))
    self.assertEqual(decoder.counts['badLength'], len(lengths))
    self.assertEqual(decoder.last_frame_ns, 0)
    self.assertFalse(any(slot.scans for slot in decoder.slots))
    self.assertFalse(decoder.drain())

  def test_event_preserves_part_order_alias_precedence_and_snapshot(self):
    decoder = BoschARadarObserver(2)
    frames = scan_frames()
    decoder.feed(START, list(reversed(frames)))
    decoder.drain()
    scan = decoder.slots[0].scans[0]
    for part, frame in scan.frames.items():
      frame.values['sharedRaw'] = part
    event = decoder._event(0, scan, 'fixture')
    self.assertEqual(event['raw']['sharedRaw'], 4)
    self.assertEqual(event['frames'], [dict(part=part, address=frames[part][0], monoTime=START, frameInEnvelope=4 - part)
                                       for part in range(5)])
    self.assertEqual(event['frameMask'], 31)
    scan.frames[4].values['u11Raw'] = 0
    self.assertEqual(event['raw']['u11Raw'], 736)


class TestBoschAObserver(unittest.TestCase):
  def test_independent_golden_bytes_and_position_velocity(self):
    decoder = BoschARadarObserver(2)
    data = ['02703e838c800000', '00000006007e0000', '0073000000000000', '8006800000000500', '5c06000000007d00']
    decoder.feed(START, [(a, bytes.fromhex(d), 2) for a, d in zip((*MAIN_IDS[0], AUX_IDS[0]), data, strict=True)])
    event = decoder.drain()[-1]
    self.assertEqual((event['raw']['rangeRaw'], event['raw']['oemTrackId']), (1000, 5))
    self.assertAlmostEqual(event['dRel'], 54.12)
    self.assertAlmostEqual(event['yRel'], 2.644682, places=4)
    self.assertEqual(event['vRel'], -2.0)
    self.assertTrue(event['positionValid'] and event['velocityValid'])

  def test_sparse_all_slots_and_wrong_bus(self):
    for slot in range(16):
      decoder = BoschARadarObserver(2)
      decoder.feed(START, scan_frames(slot, bus=0))
      self.assertFalse(decoder.drain())
      decoder.feed(START, scan_frames(slot))
      self.assertEqual(decoder.drain()[-1]['slot'], slot)
      self.assertEqual(decoder.counts['badLength'], 0)
    decoder = BoschARadarObserver(2)
    decoder.feed(START, [f for s in range(16) for f in scan_frames(s)])
    self.assertEqual({e['slot'] for e in decoder.drain() if e['velocityValid']}, set(range(16)))

  def test_partitions_preserve_physical_envelopes_and_late_aux(self):
    expected = None
    for mask in range(16):
      decoder = BoschARadarObserver(2)
      frames = scan_frames()
      outputs = []
      for i, frame in enumerate(frames):
        decoder.feed(START + i * 1_000_000, [frame])
        if i == 4 or mask & (1 << i):
          outputs.extend(decoder.drain())
      if expected is None:
        expected = outputs
      self.assertEqual(outputs, expected)
    self.assertEqual(len(expected), 2)
    self.assertEqual(expected[0]['positionSeq'], expected[1]['positionSeq'])
    self.assertEqual(expected[0]['velocitySeq'], 0)
    self.assertGreater(expected[1]['velocitySeq'], 0)
    self.assertEqual(expected[1]['positionMonoTime'], START)
    self.assertEqual(expected[1]['velocityMonoTime'], START + 4_000_000)

  def test_multiple_scans_per_batch_do_not_collapse(self):
    decoder = BoschARadarObserver(2)
    for i in range(3):
      decoder.feed(START + i * 60_000_000, scan_frames(index=i, life=1 + 2 * i))
    events = decoder.drain()
    self.assertEqual(len({e['positionSeq'] for e in events}), 3)

  def test_missing_aux_still_has_position_and_no_ratio_velocity(self):
    decoder = BoschARadarObserver(2)
    decoder.feed(START, scan_frames()[:4])
    event = decoder.drain()[-1]
    self.assertTrue(event['positionValid'])
    self.assertFalse(event['velocityValid'])
    decoder.feed(START + 60_000_000, scan_frames(index=1, life=3, distance=1750, velocity=2046, ratio=0))
    event = decoder.drain()[-1]
    self.assertFalse(event['velocityValid'])
    self.assertEqual(event['vRel'], 0)
    self.assertEqual(event['raw']['ratioRaw'], 0)

  def test_duplicates_never_refresh_measurement_even_after_expiry(self):
    decoder = BoschARadarObserver(2)
    frames = scan_frames()
    decoder.feed(START, frames)
    decoder.drain()
    for i in range(1, 20):
      decoder.feed(START + i * 60_000_000, frames)
      self.assertFalse(any(e.get('positionValid') for e in decoder.drain()))
    self.assertEqual(decoder.last_position_ns, START)
    self.assertTrue(decoder.slots[0].expired)

  def test_idle_expiry_and_rebirth_not_other_bus_freshness(self):
    decoder = BoschARadarObserver(2)
    decoder.feed(START, scan_frames())
    old_generation = decoder.drain()[-1]['generation']
    decoder.feed(START + MAX_AGE_NS + 1, [(10, bytes(8), 0)])
    self.assertEqual(decoder.drain()[-1]['reason'], 'expired')
    self.assertEqual(decoder.last_frame_ns, START)
    decoder.feed(START + MAX_AGE_NS + 2, scan_frames(index=1, life=3))
    self.assertGreater(decoder.drain()[-1]['generation'], old_generation)

  def test_frame_wrap_and_inconsistent_lifecycle_generation(self):
    decoder = BoschARadarObserver(2)
    decoder.feed(START, scan_frames(index=15, life=4094))
    generation = decoder.drain()[-1]['generation']
    decoder.feed(START + 60_000_000, scan_frames(index=0, life=0))
    self.assertEqual(decoder.drain()[-1]['generation'], generation)
    decoder.feed(START + 120_000_000, scan_frames(index=1, life=50))
    self.assertGreater(decoder.drain()[-1]['generation'], generation)

  def test_recorded_plateau_windows_preserve_generation_across_wrap(self):
    # Three positions around the first 15->0 wrap in each frozen V092/V431
    # tracking-probe.json window (2026-10-06, recording source 9983a048).
    # Time/index are recorded; range/azimuth are reconstructed from decoded
    # geometry on its original quantization grid. These are not raw CAN bytes.
    windows = {
      60: ((997102187326, 15, 644, 1047), (997172030399, 0, 643, 1047), (997231839201, 1, 643, 1046)),
      38: ((1629995300105, 15, 843, 1033), (1630055255209, 0, 842, 1033), (1630135305053, 1, 841, 1033)),
    }
    for identity, rows in windows.items():
      with self.subTest(identity=identity):
        decoder = BoschARadarObserver(2)
        events = []
        for stamp, index, distance, azimuth in rows:
          decoder.feed(stamp, scan_frames(slot=1, index=index, life=4094, distance=distance,
                                          oem_id=identity, azimuth=azimuth))
          events.extend(decoder.drain())
        self.assertEqual(len({e['generation'] for e in events}), 1)
        self.assertEqual(len({e['positionSeq'] for e in events}), 3)
        self.assertTrue(all(e['positionValid'] for e in events))
        self.assertGreater(rows[-1][0] - rows[0][0], 120_000_000)

  def test_plateau_aux_and_repeats_do_not_create_position_samples(self):
    decoder = BoschARadarObserver(2)
    frames = scan_frames(index=15, life=4094)
    decoder.feed(START, frames[:4])
    position = decoder.drain()[-1]
    decoder.feed(START + 1_000_000, frames[4:])
    aux = decoder.drain()[-1]
    self.assertEqual(aux['positionSeq'], position['positionSeq'])
    self.assertEqual(aux['generation'], position['generation'])
    self.assertEqual(aux['positionMonoTime'], START)
    decoder.feed(START + 2_000_000, frames)
    self.assertFalse(any(e.get('positionValid') for e in decoder.drain()))
    decoder.feed(START + 70_000_000, scan_frames(index=0, life=4094))
    next_position = decoder.drain()[-1]
    self.assertEqual(next_position['generation'], position['generation'])
    self.assertGreater(next_position['positionSeq'], position['positionSeq'])

  def test_plateau_geometry_or_identity_discontinuity_starts_new_generation(self):
    # Large raw-position changes are identity breaks even with the same OEM ID
    # and plateau. Velocity agreement is not used to choose a presumed truth.
    changes = ({'distance': 1300}, {'azimuth': 1400}, {'oem_id': 6}, {'life': 2})
    for changed in changes:
      with self.subTest(changed=changed):
        decoder = BoschARadarObserver(2)
        decoder.feed(START, scan_frames(index=0, life=4094, oem_id=5))
        generation = decoder.drain()[-1]['generation']
        values = dict(index=1, life=4094, oem_id=5)
        values.update(changed)
        decoder.feed(START + 70_000_000, scan_frames(**values))
        event = decoder.drain()[-1]
        self.assertTrue(event['positionValid'])
        self.assertGreater(event['generation'], generation)

  def test_invalid_plateau_position_and_late_aux_cannot_restore_generation(self):
    decoder = BoschARadarObserver(2)
    decoder.feed(START, scan_frames(index=0, life=4094))
    first = decoder.drain()[-1]
    invalid = scan_frames(index=1, life=4095)
    decoder.feed(START + 70_000_000, invalid[:4])
    self.assertFalse(decoder.drain()[-1]['positionValid'])
    decoder.feed(START + 71_000_000, invalid[4:])
    event = decoder.drain()[-1]
    self.assertFalse(event['positionValid'] or event['velocityValid'])
    self.assertEqual(decoder.last_position_ns, START)
    decoder.feed(START + 140_000_000, scan_frames(index=2, life=4094))
    self.assertGreater(decoder.drain()[-1]['generation'], first['generation'])

  def test_conflicting_plateau_position_requires_new_generation(self):
    decoder = BoschARadarObserver(2)
    decoder.feed(START, scan_frames(index=0, life=4094))
    generation = decoder.drain()[-1]['generation']
    decoder.feed(START + 1_000_000, scan_frames(index=0, life=4094, distance=1100)[:1])
    self.assertEqual(decoder.drain()[-1]['reason'], 'conflictingFrame')
    decoder.feed(START + 70_000_000, scan_frames(index=1, life=4094))
    self.assertGreater(decoder.drain()[-1]['generation'], generation)

  def test_simultaneous_slot_claims_break_plateau_continuity(self):
    decoder = BoschARadarObserver(2)
    decoder.feed(START, scan_frames(slot=0, index=15, life=4094, oem_id=5))
    first_generation = decoder.drain()[-1]['generation']
    decoder.feed(START + 70_000_000, scan_frames(slot=0, index=0, life=4094, oem_id=5)
                 + scan_frames(slot=1, index=0, life=4094, oem_id=5))
    events = [e for e in decoder.drain() if e['velocityValid']]
    self.assertEqual({e['slot'] for e in events}, {0, 1})
    self.assertNotEqual(events[0]['generation'], events[1]['generation'])
    self.assertTrue(decoder.slots[0].identity_conflict and decoder.slots[1].identity_conflict)
    second_generation = events[1]['generation']
    decoder.feed(START + 140_000_000, scan_frames(slot=0, index=1, life=4094, oem_id=5)
                 + scan_frames(slot=1, index=1, life=4094, oem_id=5))
    events = [e for e in decoder.drain() if e['velocityValid']]
    self.assertGreater(events[0]['generation'], first_generation)
    self.assertGreater(events[1]['generation'], second_generation)

  def test_next_frame_slot_migration_is_not_a_simultaneous_claim(self):
    decoder = BoschARadarObserver(2)
    decoder.feed(START, scan_frames(slot=0, index=15, life=4094, oem_id=5))
    decoder.drain()
    decoder.feed(START + 70_000_000, scan_frames(slot=1, index=0, life=4094, oem_id=5))
    generation = decoder.drain()[-1]['generation']
    self.assertFalse(decoder.slots[0].identity_conflict or decoder.slots[1].identity_conflict)
    decoder.feed(START + 140_000_000, scan_frames(slot=1, index=1, life=4094, oem_id=5))
    self.assertEqual(decoder.drain()[-1]['generation'], generation)

  def test_slot_conflict_uses_position_time_with_interleaved_main_frames(self):
    decoder = BoschARadarObserver(2)
    delayed = scan_frames(slot=0, life=4094, oem_id=5)
    decoder.feed(START, delayed[:1])
    decoder.feed(START + 1_000_000, scan_frames(slot=1, life=4094, oem_id=5))
    decoder.feed(START + 2_000_000, delayed[1:])
    observations = [e for e in decoder.drain() if e['velocityValid']]
    self.assertEqual({e['slot'] for e in observations}, {0, 1})
    self.assertTrue(decoder.slots[0].identity_conflict and decoder.slots[1].identity_conflict)

  def test_plateau_expiry_does_not_inherit_generation(self):
    decoder = BoschARadarObserver(2)
    decoder.feed(START, scan_frames(index=0, life=4094))
    generation = decoder.drain()[-1]['generation']
    decoder.feed(START + MAX_AGE_NS + 1, scan_frames(index=1, life=4094))
    self.assertGreater(decoder.drain()[-1]['generation'], generation)

  def test_invalid_geometry_endpoints_dlc_and_incomplete(self):
    for distance, identity in ((0, 1), (4095, 1), (1000, 0), (1000, 64)):
      decoder = BoschARadarObserver(2)
      decoder.feed(START, scan_frames(distance=distance, oem_id=identity))
      self.assertFalse(any(e['positionValid'] for e in decoder.drain()))
    for raw in (0, 1728, 2046, 2047):
      decoder = BoschARadarObserver(2)
      decoder.feed(START, scan_frames(velocity=raw))
      self.assertFalse(decoder.drain()[-1]['velocityValid'])
    decoder = BoschARadarObserver(2)
    decoder.feed(START, [(MAIN_IDS[0][0], bytes(7), 2)])
    self.assertEqual(decoder.counts['badLength'], 1)
    decoder.feed(START, scan_frames()[:3])
    decoder.advance(START + MAX_AGE_NS + 1)
    self.assertEqual(decoder.drain()[-1]['reason'], 'incomplete')

  def test_mixed_indices_conflict_and_bounded_backlog(self):
    decoder = BoschARadarObserver(2)
    first, second = scan_frames(), scan_frames(index=1, life=3)
    decoder.feed(START, first[:2] + second[2:])
    self.assertFalse(decoder.drain())
    changed = bytearray(first[0][1])
    changed[2] ^= 1
    decoder.feed(START + 1, [(first[0][0], bytes(changed), 2)])
    self.assertEqual(decoder.drain()[-1]['reason'], 'conflictingFrame')
    decoder = BoschARadarObserver(2)
    for i in range(40):
      decoder.feed(START + i * 60_000_000, scan_frames(index=i % 16, life=1 + 2 * i))
    self.assertLessEqual(len(decoder.events), MAX_EVENTS)
    self.assertLessEqual(len(decoder.slots[0].scans), 3)
    self.assertGreater(decoder.counts['dropped'], 0)

  def test_all_main_aux_orders_preserve_one_position_sample(self):
    frames = scan_frames()
    for order in permutations(range(5)):
      with self.subTest(order=order):
        decoder = BoschARadarObserver(2)
        decoder.feed(START, [frames[i] for i in order])
        events = decoder.drain()
        self.assertEqual(len({e['positionSeq'] for e in events}), 1)
        self.assertTrue(events[-1]['positionValid'] and events[-1]['velocityValid'])
        self.assertAlmostEqual(events[-1]['dRel'], 54.12)
        self.assertEqual(events[-1]['vRel'], -2.0)

  def test_same_oem_id_in_two_slots_remains_two_observations(self):
    decoder = BoschARadarObserver(2)
    decoder.feed(START, scan_frames(slot=0, oem_id=5) + scan_frames(slot=1, oem_id=5))
    observations = [e for e in decoder.drain() if e['velocityValid']]
    self.assertEqual([e['slot'] for e in observations], [0, 1])
    self.assertEqual([e['raw']['oemTrackId'] for e in observations], [5, 5])
    self.assertNotEqual(observations[0]['generation'], observations[1]['generation'])

  def test_rejected_geometry_retains_late_aux_raw_fields(self):
    for slot in range(16):
      with self.subTest(slot=slot):
        decoder = BoschARadarObserver(2)
        frames = scan_frames(slot=slot, distance=4095, velocity=1728, ratio=1023)
        decoder.feed(START, frames[:4])
        self.assertEqual(decoder.drain()[-1]['reason'], 'invalidGeometry')
        decoder.feed(START + 1_000_000, frames[4:])
        events = decoder.drain()
        self.assertTrue(events, 'AUX of an empty/rejected slot must remain available for analysis')
        self.assertEqual(events[-1]['frameMask'], 31)
        self.assertEqual(events[-1]['raw']['u11Raw'], 1728)
        self.assertEqual(events[-1]['raw']['ratioRaw'], 1023)
        self.assertFalse(events[-1]['positionValid'] or events[-1]['velocityValid'])


class TestRadarScanContinuity(unittest.TestCase):
  def test_sequence_rule_keeps_normal_progression_and_qualified_plateau(self):
    for previous_index, previous_life, index, life in ((0, 10, 1, 12), (15, 4094, 0, 0),
                                                      (15, 4094, 0, 4094), (3, 4094, 5, 4094)):
      with self.subTest(values=(previous_index, previous_life, index, life)):
        self.assertTrue(radar_scan_continuous(previous_index, previous_life, START, index, life, START + 70_000_000))

  def test_sequence_rule_rejects_reuse_duplicates_reverse_and_invalid_fields(self):
    for previous_index, previous_life, index, life in ((0, 4094, 0, 4094), (0, 4094, 15, 4094),
                                                      (0, 4094, 1, 2), (0, 4095, 1, 4094),
                                                      (0, 4094, 1, 4095), (0, -1, 1, 1),
                                                      (-1, 4094, 0, 4094), (15, 4094, 16, 4094)):
      with self.subTest(values=(previous_index, previous_life, index, life)):
        self.assertFalse(radar_scan_continuous(previous_index, previous_life, START, index, life, START + 70_000_000))

  def test_sequence_rule_rejects_missing_reversed_or_expired_time(self):
    for before, after in ((0, 70_000_000), (START, START), (START, START - 1), (START, START + MAX_AGE_NS + 1)):
      with self.subTest(before=before, after=after):
        self.assertFalse(radar_scan_continuous(15, 4094, before, 0, 4094, after))


if __name__ == '__main__':
  unittest.main()
