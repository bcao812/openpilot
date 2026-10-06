"""Shadow fusion contracts, including the 2026-10-05 road-log counterexamples."""
import copy
from types import SimpleNamespace as NS
import unittest

import numpy as np

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
    conflicting['positionMonoTime'] += 1
    out = advance(fusion, 8, [original, conflicting])
    self.assertEqual(out['counters']['positions'], 9)
    self.assertFalse(out['tracks'][0]['active'])
    self.assertEqual(out['tracks'][0]['lastInvalidReason'], 'slotConflict')
    self.assertEqual(out['leads'][0]['distanceWeight'], 0)
    self.assertEqual(out['leads'][0]['velocityWeight'], 0)

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
