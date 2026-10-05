"""Shadow fusion contracts, including the 2026-10-05 road-log counterexamples."""
import copy
from types import SimpleNamespace as NS
import unittest

import numpy as np

from openpilot.selfdrive.car.honda_radar_fusion import Ego, HondaRadarFusion, MAX_TRACKS, path_relation

START = 10_000_000_000
STEP = 66_000_000


class Inputs(dict):
  def __init__(self, t, distance=20.0, lateral=0.0, relative=0.0, speed=10.0, probability=.99, stopped_path=False, yaw=0.0):
    lead = {'prob': probability, 'x': [distance + 1.52], 'y': [-lateral], 'v': [speed + relative], 'a': [0.0],
                'xStd': [1.0], 'yStd': [.4], 'vStd': [1.0]}
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


class TestHondaRadarFusion(unittest.TestCase):
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

  def test_missing_future_and_stale_dependencies_clear_candidates(self):
    for kind in ('modelStale', 'egoStale', 'futureCamera', 'inputsInvalid', 'scopeInactive', 'schemaUnknown'):
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
