"""Shadow fusion contracts, including the 2026-10-05 road-log counterexamples."""
import copy
from types import SimpleNamespace as NS
import unittest

import numpy as np

from opendbc.car.honda.bosch_a_radar import BoschARadarObserver
from opendbc.car.honda.tests.test_bosch_a_observer import scan_frames
from openpilot.selfdrive.car.honda_radar_fusion import Binding, Ego, EgoHistory, HondaRadarFusion, MAX_TRACKS, path_relation

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

  def test_explicit_withdrawal_revokes_all_components_before_age_decay_and_requires_new_confirmation(self):
    for reason in ('invalidGeometry', 'conflictingFrame', 'expired', 'withdrawn'):
      with self.subTest(reason=reason):
        fusion = HondaRadarFusion()
        for frame in range(12):
          out = advance(fusion, frame)
        old = observation(11)
        old_identifier = out['tracks'][0]['trackId']
        invalid = {'slot': 0, 'positionValid': False, 'positionMonoTime': old['positionMonoTime'],
                   'positionSeq': old['positionSeq'], 'reason': reason}
        out = advance(fusion, 12, [invalid])
        track, lead = out['tracks'][0], out['leads'][0]
        self.assertFalse(track['active'])
        self.assertFalse(track['velocityGood'])
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
        self.assertEqual(out['leads'][0]['distanceWeight'], 0)
        out = advance(fusion, 14)
        self.assertNotEqual(out['tracks'][0]['trackId'], old_identifier)
        self.assertEqual(out['tracks'][0]['observations'], 1)
        self.assertEqual(out['tracks'][0]['lastInvalidReason'], reason)
        self.assertEqual(out['leads'][0]['distanceWeight'], 0)
        out = advance(fusion, 15)
        self.assertEqual(out['leads'][0]['distanceWeight'], 0)
        out = advance(fusion, 16)
        self.assertTrue(out['tracks'][0]['mature'])
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
    self.assertEqual(out['algorithmVersion'], 'bosch-a-shadow-2')
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
