import unittest
from itertools import permutations

from opendbc.can.packer import CANPacker
from opendbc.car.honda.bosch_a_radar import AUX_IDS, MAIN_IDS, MAX_AGE_NS, MAX_EVENTS, BoschARadarObserver

START = 10_000_000_000


def scan_frames(slot=0, index=0, life=1, distance=1000, oem_id=None, velocity=736, ratio=500, bus=2):
  packer = CANPacker('honda_bosch_a_radar')
  values = [dict(STATUS=7, RANGE_RAW=distance, AZIMUTH_RAW=1124, RANGE_SIGMA_RAW=1),
            dict(OBJECT_EXISTENCE_PROBABILITY_RAW=126), dict(LIFECYCLE_RAW=life),
            dict(TRACK_ID=slot + 1 if oem_id is None else oem_id),
            dict(REL_VELOCITY_RAW=velocity, REL_VELOCITY_UNCERTAINTY_RAW=0, RANGE_RATIO_RAW=ratio)]
  return [packer.make_can_msg(addr, bus, dict(v, FRAME_IDX=index)) for addr, v in zip((*MAIN_IDS[slot], AUX_IDS[slot]), values, strict=True)]


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


if __name__ == '__main__':
  unittest.main()
