"""Display-only angular-bound tests without UI, CAN or vehicle native libraries."""
from copy import deepcopy
import math
from types import SimpleNamespace
import unittest

import numpy as np

from openpilot.selfdrive.ui.sunnypilot.onroad.honda_radar_state import POSITION_MAX_AGE_NS, RadarLabelPoint, RadarOverlayState

START = 10_000_000_000
SCOPE = dict.fromkeys(('alphaLong', 'experimentalSetting', 'decSetting', 'experimentalActive', 'decEnabled', 'decActive'), True)


def observation(slot=0, sequence=1, generation=1, stamp=START, azimuth=1054, edges=(1020, 1100), range_raw=1000, main_times=None):
  distance = range_raw * .05712 - 3.0
  base = 0x280 + 4 * slot if slot < 4 else 0x2D0 + 4 * (slot - 4)
  stamps = main_times if main_times is not None else (stamp,) * 4
  return dict(slot=slot, frameIndex=0, frameMask=15, generation=generation, positionSeq=sequence, positionMonoTime=stamp,
              positionValid=True, reason='positionOnly', dRel=distance, yRel=distance * math.tan((azimuth - 1024) / 2048.0),
              raw=dict(rangeRaw=range_raw, azimuthRaw=azimuth, edgeARaw=edges[0], edgeBRaw=edges[1]),
              frames=[dict(part=part, address=base + part, monoTime=t, frameInEnvelope=part) for part, t in enumerate(stamps)])


def publication(events, sequence=1, session='fixture', **fields):
  return dict(schemaVersion=1, sessionId=session, sequence=sequence, active=True, scope=SCOPE,
              observations=events, **fields)


def with_aux(event, stamp):
  result = deepcopy(event)
  slot = result['slot']
  result['frameMask'] |= 16
  result['frames'].append(dict(part=4, address=0x2C8 + slot if slot < 8 else 0x290 + slot - 8, monoTime=stamp, frameInEnvelope=0))
  return result


class TestHondaRadarBoundaries(unittest.TestCase):
  def ingest(self, events, message_ns=START + 1):
    overlay = RadarOverlayState()
    overlay.ingest(publication(events), message_ns, message_ns)
    return overlay

  def test_absolute_asymmetric_bounds_keep_reflection_and_sample_identity(self):
    event = observation()
    point = self.ingest([event]).points(START, START + 1)[0]
    left, right = point.boundary_laterals
    self.assertAlmostEqual(left, point.distance * math.tan(76 / 2048.0))
    self.assertAlmostEqual(right, point.distance * math.tan(-4 / 2048.0))
    self.assertGreater(left, point.lateral)
    self.assertLess(right, point.lateral)
    self.assertNotAlmostEqual((left + right) / 2, point.lateral)
    self.assertEqual(point.lateral, event['yRel'])
    self.assertEqual(point.distance, event['dRel'])
    self.assertEqual(point.text, '54.1')
    self.assertEqual(point.sample_key, (0, 1, 1, START))
    self.assertEqual(point.boundary_time_ns, START)
    self.assertEqual(point.time_ns, START)
    self.assertIsNone(RadarLabelPoint(0, 10, 0, START, 1, 1).boundary_laterals)

  def test_edge_order_signs_and_all_slot_addresses(self):
    for slot in range(16):
      for azimuth, edges in ((1054, (1020, 1100)), (1054, (1100, 1020)), (974, (930, 1024))):
        with self.subTest(slot=slot, azimuth=azimuth, edges=edges):
          point = self.ingest([observation(slot=slot, azimuth=azimuth, edges=edges)]).points(START, START + 1)[0]
          left, right = point.boundary_laterals
          self.assertAlmostEqual(left, point.distance * math.tan((max(edges) - 1024) / 2048.0))
          self.assertAlmostEqual(right, point.distance * math.tan((min(edges) - 1024) / 2048.0))
          self.assertGreaterEqual(left, point.lateral)
          self.assertLessEqual(right, point.lateral)

  def test_zero_and_maximum_edge_encodings_are_not_invented_sentinels(self):
    for azimuth, edges in ((32, (0, 64)), (2020, (1980, 2047)), (1024, (1024, 1100))):
      with self.subTest(azimuth=azimuth, edges=edges):
        point = self.ingest([observation(azimuth=azimuth, edges=edges)]).points(START, START + 1)[0]
        self.assertIsNotNone(point.boundary_laterals)

  def test_missing_raw_fields_preserve_only_the_reflection(self):
    for field in ('rangeRaw', 'azimuthRaw', 'edgeARaw', 'edgeBRaw'):
      with self.subTest(field=field):
        event = observation()
        del event['raw'][field]
        point = self.ingest([event]).points(START, START + 1)[0]
        self.assertIsNone(point.boundary_laterals)
        self.assertEqual(point.boundary_time_ns, 0)
        self.assertEqual(point.text, '54.1')

  def test_frame_presence_is_required_even_when_zero_is_a_valid_angle(self):
    for mask in (None, 0, 7, 11, 13, 14, 31, 47, True, 15.0, '15'):
      with self.subTest(mask=mask):
        event = observation(azimuth=32, edges=(0, 64))
        event['frameMask'] = mask
        # 31 claims AUX but supplies no corresponding frame reference.
        point = self.ingest([event]).points(START, START + 1)[0]
        self.assertIsNone(point.boundary_laterals)

  def test_invalid_angles_or_degenerate_bounds_never_use_angular_width_fallback(self):
    for edges in ((-1, 1100), (1020, 2048), (1020, 1020), (0, 0), (2047, 2047), (1100, 1200), (900, 1000)):
      with self.subTest(edges=edges):
        event = observation(edges=edges)
        event['raw']['angularWidthRaw'] = 512
        point = self.ingest([event]).points(START, START + 1)[0]
        self.assertIsNone(point.boundary_laterals)

  def test_malformed_boundary_fields_do_not_hide_other_targets(self):
    for field in ('rangeRaw', 'azimuthRaw', 'edgeARaw', 'edgeBRaw'):
      for value in (None, True, '1024', 1024.0, math.inf, math.nan):
        with self.subTest(field=field, value=value):
          bad = observation()
          bad['raw'][field] = value
          points = self.ingest([bad, observation(slot=1, sequence=2)]).points(START, START + 1)
          self.assertEqual(len(points), 2)
          self.assertIsNone(points[0].boundary_laterals)
          self.assertIsNotNone(points[1].boundary_laterals)

  def test_malformed_position_target_does_not_hide_a_valid_target(self):
    for field, value in (('slot', None), ('slot', True), ('slot', '0'), ('slot', -1), ('slot', 16), ('slot', 0.0),
                         ('generation', None), ('generation', -1), ('generation', True), ('generation', 1.0),
                         ('positionSeq', None), ('positionSeq', True), ('positionSeq', 0), ('positionSeq', -1),
                         ('positionMonoTime', None), ('positionMonoTime', True), ('positionMonoTime', 0),
                         ('positionMonoTime', START + 2),
                         ('dRel', None), ('dRel', math.nan), ('yRel', '0'), ('yRel', math.inf)):
      with self.subTest(field=field, value=value):
        bad = observation()
        bad[field] = value
        points = self.ingest([bad, observation(slot=1, sequence=2)]).points(START, START + 1)
        self.assertEqual([point.slot for point in points], [1])
        self.assertIsNotNone(points[0].boundary_laterals)

  def test_frame_references_must_identify_this_scan(self):
    mutations = (
      lambda e: e.pop('frames'),
      lambda e: e.update(frames=None),
      lambda e: e.update(frames=e['frames'][:3]),
      lambda e: e.update(frames=e['frames'] * 16),
      lambda e: e['frames'][3].update(part=2),
      lambda e: e['frames'][3].update(part=3.0),
      lambda e: e['frames'][3].update(part=True),
      lambda e: e['frames'][3].update(address=0x287),
      lambda e: e['frames'][3].update(address='643'),
      lambda e: e['frames'][3].update(monoTime=0),
      lambda e: e['frames'][3].update(monoTime=None),
      lambda e: e['frames'][3].update(monoTime=START + 2),
      lambda e: e['frames'][0].update(monoTime=START - 1),
      lambda e: e['frames'][1].update(monoTime=START - POSITION_MAX_AGE_NS - 1),
      lambda e: e.update(frameIndex=16),
      lambda e: e.update(frameIndex=True),
    )
    for index, mutate in enumerate(mutations):
      with self.subTest(index=index):
        event = observation()
        mutate(event)
        points = self.ingest([event, observation(slot=1, sequence=2)]).points(START, START + 1)
        self.assertEqual(len(points), 2)
        self.assertIsNone(points[0].boundary_laterals)
        self.assertIsNotNone(points[1].boundary_laterals)

  def test_raw_geometry_must_match_the_existing_reflection(self):
    for field, delta in (('dRel', 3.0), ('yRel', .1)):
      with self.subTest(field=field):
        event = observation()
        event[field] += delta
        point = self.ingest([event]).points(START, START + 1)[0]
        self.assertIsNone(point.boundary_laterals)
        self.assertEqual(point.distance, event['dRel'])
        self.assertEqual(point.lateral, event['yRel'])

  def test_late_main_frames_do_not_appear_in_earlier_camera_images(self):
    late = START + 60_000_000
    for part in (1, 2, 3):
      with self.subTest(part=part):
        stamps = [START] * 4
        stamps[part] = late
        overlay = self.ingest([observation(main_times=stamps)], late + 1)
        for camera_ns in (START, late - 1):
          point = overlay.points(camera_ns, late + 1)[0]
          self.assertIsNone(point.boundary_laterals)
          self.assertEqual(point.time_ns, START)
          self.assertEqual(point.sample_key, (0, 1, 1, START))
        point = overlay.points(late, late + 1)[0]
        self.assertIsNotNone(point.boundary_laterals)
        self.assertEqual(point.boundary_time_ns, late)
        self.assertEqual(overlay.history[0][0].boundary_laterals, point.boundary_laterals)

  def test_boundary_freshness_is_limited_by_the_original_position(self):
    late = START + 180_000_000
    overlay = self.ingest([observation(main_times=(START, START, START, late))], late + 1)
    self.assertIsNotNone(overlay.points(late, START + POSITION_MAX_AGE_NS)[0].boundary_laterals)
    self.assertFalse(overlay.points(late, START + POSITION_MAX_AGE_NS + 1))

  def test_late_aux_never_refreshes_position_or_boundary_timestamps(self):
    event = observation()
    overlay = self.ingest([event])
    original = overlay.history[0][0]
    late = START + 180_000_000
    overlay.ingest(publication([with_aux(event, late)], sequence=2), late + 1, late + 1)
    self.assertEqual(list(overlay.history[0]), [original])
    self.assertEqual(overlay.points(late, late + 1)[0].boundary_time_ns, START)
    self.assertFalse(overlay.points(late, START + POSITION_MAX_AGE_NS + 1))

  def test_aux_does_not_set_boundary_availability_even_on_first_received_snapshot(self):
    late = START + 80_000_000
    for aux_ns in (START - 20_000_000, late):
      with self.subTest(aux_ns=aux_ns):
        overlay = self.ingest([with_aux(observation(), aux_ns)], late + 1)
        point = overlay.points(START, late + 1)[0]
        self.assertIsNotNone(point.boundary_laterals)
        self.assertEqual(point.boundary_time_ns, START)

  def test_each_position_needs_its_own_complete_bounds(self):
    overlay = self.ingest([observation()])
    later = START + 50_000_000
    event = observation(sequence=2, stamp=later, edges=(0, 0))
    overlay.ingest(publication([event], sequence=2), later + 1, later + 1)
    self.assertIsNone(overlay.points(later, later + 1)[0].boundary_laterals)
    self.assertIsNotNone(overlay.points(START, later + 1)[0].boundary_laterals)
    self.assertEqual(len(overlay.history[0]), 2)

  def test_slot_generation_reuse_discards_old_boundaries(self):
    overlay = self.ingest([observation()])
    later = START + 50_000_000
    event = observation(sequence=2, generation=2, stamp=later, edges=(0, 0))
    overlay.ingest(publication([event], sequence=2), later + 1, later + 1)
    self.assertEqual(len(overlay.history[0]), 1)
    self.assertFalse(overlay.points(START, later + 1))
    self.assertIsNone(overlay.points(later, later + 1)[0].boundary_laterals)

  def test_partial_edges_from_different_slots_cannot_be_combined(self):
    first, second = observation(), observation(slot=1, sequence=2)
    del first['raw']['edgeBRaw']
    del second['raw']['edgeARaw']
    points = self.ingest([first, second]).points(START, START + 1)
    self.assertEqual(len(points), 2)
    self.assertTrue(all(point.boundary_laterals is None for point in points))

  def test_late_aux_cannot_restore_invalidated_or_cleared_boundaries(self):
    for discontinuity in (publication([dict(slot=0, positionValid=False, positionMonoTime=START, reason='invalidGeometry')], sequence=2),
                          publication([], sequence=3)):
      with self.subTest(sequence=discontinuity['sequence']):
        event = observation()
        overlay = self.ingest([event])
        overlay.ingest(discontinuity, START + 50_000_000, START + 50_000_000)
        overlay.ingest(publication([with_aux(event, START + 80_000_000)], sequence=discontinuity['sequence'] + 1),
                       START + 80_000_001, START + 80_000_001)
        self.assertFalse(overlay.points(START + 80_000_000, START + 80_000_001))

  def test_track_identity_refresh_keeps_this_samples_boundaries(self):
    overlay = self.ingest([observation()])
    original = overlay.history[0][0]
    fusion = dict(version=1, active=True,
                  tracks=[dict(slot=0, generation=1, positionSeq=1, positionMonoTime=START, trackId=7)])
    overlay.ingest(publication([], sequence=2, fusion=fusion), START + 2, START + 2)
    point = overlay.points(START, START + 2)[0]
    self.assertEqual(point.track_id, 7)
    self.assertEqual(point.sample_key, original.sample_key)
    self.assertEqual(point.boundary_laterals, original.boundary_laterals)
    self.assertEqual(point.boundary_time_ns, original.boundary_time_ns)

  def test_integral_track_fields_preserve_exact_sample_identity(self):
    class IntSubclass(int):
      pass

    event = observation(slot=15, generation=0, sequence=2 ** 32)
    identifier = 2 ** 63 - 1
    for integer in (int, IntSubclass, np.int64, np.uint64):
      with self.subTest(integer=integer.__name__):
        track = {field: integer(event[field]) for field in ('slot', 'generation', 'positionSeq', 'positionMonoTime')}
        track['trackId'] = integer(identifier)
        overlay = RadarOverlayState()
        overlay.ingest(publication([event], fusion=dict(version=1, active=True, tracks=[track])), START + 1, START + 1)
        point, = overlay.points(START, START + 1)
        self.assertEqual(point.track_id, identifier)
        self.assertEqual(point.sample_key, (15, 0, 2 ** 32, START))
        self.assertIsNotNone(point.boundary_laterals)

  def test_invalid_track_fields_leave_the_current_reflection_unassociated(self):
    event = observation()
    track = {field: event[field] for field in ('slot', 'generation', 'positionSeq', 'positionMonoTime')}
    track['trackId'] = 7
    bounds = {'slot': (-1, 16), 'generation': (-1,), 'positionSeq': (-1, 0),
              'positionMonoTime': (-1, 0), 'trackId': (-1, 0)}
    for field in track:
      for value in (*bounds[field], True, np.bool_(True), float(track[field]), str(track[field]), None):
        with self.subTest(field=field, value=value):
          overlay = RadarOverlayState()
          fusion = dict(version=1, active=True, tracks=[dict(track, **{field: value})])
          overlay.ingest(publication([event], fusion=fusion), START + 1, START + 1)
          point, = overlay.points(START, START + 1)
          self.assertEqual(point.track_id, 0)
          self.assertEqual(point.sample_key, (0, 1, 1, START))
          self.assertIsNotNone(point.boundary_laterals)

  def test_dictionary_and_attribute_readers_use_the_same_boundary_contract(self):
    def reader(value):
      if isinstance(value, dict):
        return SimpleNamespace(**{key: reader(item) for key, item in value.items()})
      return [reader(item) for item in value] if isinstance(value, list) else value

    data = publication([observation()])
    expected = self.ingest(data['observations']).points(START, START + 1)
    overlay = RadarOverlayState()
    overlay.ingest(reader(data), START + 1, START + 1)
    self.assertEqual(overlay.points(START, START + 1), expected)


if __name__ == '__main__':
  unittest.main()
