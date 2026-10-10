"""CAN-origin regressions for active radar fusion and its execution boundary."""
import json
from unittest.mock import Mock

import pytest

from openpilot.cereal import messaging
from openpilot.selfdrive.car.honda_radar_control import SWEPT_HALF_WIDTH
from openpilot.selfdrive.car.honda_radar_observer_config import SERVICE, SCOPE_MAX_AGE_NS, ObserverSettings
from openpilot.selfdrive.car.tests import honda_radar_control_test_support as pipeline_support
from openpilot.selfdrive.car.tests.honda_radar_control_test_support import Target, control_pipeline
from openpilot.selfdrive.controls.lib.drive_helpers import STOP_SPEED
from openpilot.selfdrive.controls.lib.motion_supervisor import StopReason


def objects(row):
  assert row['radar'].get('radarObjects', []) == []
  return [row['radar'][name] for name in ('leadOne', 'leadTwo') if row['radar'][name].get('present')]


def record(record_property, name, rows):
  record_property(name, json.dumps(rows, separators=(',', ':'), allow_nan=False))


def confirmed_offset_target(pipeline):
  """M4-inspired body offset with matched synthetic visual input."""
  target = Target(20.0, 8.0, lateral=1.3, slot=2, oem_id=16)
  rows = [pipeline.step(speed=8.0, radar=[target], visual=[target]) for _ in range(24)]
  matched = next(lead for lead in objects(rows[-1]) if lead['evidence']['source'] == 'fused')
  identity = matched['evidence']['sessionId'], matched['evidence']['trackId']
  # Let the fixture's visual probability/association disappear while fresh CAN
  # keeps this physical identity and its independently confirmed body history.
  rows.extend(pipeline.step(speed=8.0, radar=[target]) for _ in range(24))
  memory = pipeline.radard.fusion_control._tracks[identity[1]]
  assert memory.path.confirmed and SWEPT_HALF_WIDTH < abs(memory.path.offset) < 2.0
  current = next(lead for lead in objects(rows[-1]) if lead['evidence']['trackId'] == identity[1])
  assert current['evidence']['source'] == 'radar' and current['modelProb'] == 0.0
  assert current['evidence']['brakeQualified'] and current['evidence']['pathState'] == 'inPath'
  return target, identity, memory.generation, rows


@pytest.mark.parametrize('intervention', ['gas', 'brake', 'disabled'])
def test_driver_edges_preserve_fresh_confirmed_body_during_long_visual_gap(intervention, record_property):
  with control_pipeline(speed=8.0) as pipeline:
    target, identity, generation, rows = confirmed_offset_target(pipeline)
    adapter = pipeline.radard.fusion_control
    last_position_ns = adapter._tracks[identity[1]].position_ns
    for _ in range(4):
      row = pipeline.step(speed=8.0, radar=[target], engaged=intervention != 'disabled',
                          pedal=intervention if intervention != 'disabled' else None)
      rows.append(row)
      memory = adapter._tracks[identity[1]]
      assert pipeline.radard.fusion_control is adapter
      assert memory.generation == generation and memory.path.confirmed
      assert memory.position_ns > last_position_ns
      last_position_ns = memory.position_ns
      lead = next(lead for lead in objects(row) if lead['evidence']['trackId'] == identity[1])
      assert lead['evidence']['sessionId'] == identity[0] and lead['evidence']['brakeQualified']
      assert not row['long_active'] and not pipeline.state['carControl'].longActive
      assert all(not owner['active'] and owner['phase'] == 'off' and owner['accel'] == 0.0 for owner in row['owner'])
      assert all(not command['active'] and not command['brake'] and command['accel'] == 0.0
                 and not command['release'] and not command['resume'] for command in row['can'])
      assert pipeline.planner.reference_planner.radar_held_identity is None
      assert not pipeline.planner.reference_planner._radar_stop_pending
    last_manual_ns = rows[-1]['source_ns']
    for _ in range(12):
      row = pipeline.step(speed=8.0, radar=[target])
      rows.append(row)
      memory = adapter._tracks[identity[1]]
      assert memory.generation == generation and memory.path.confirmed
      assert memory.position_ns > last_position_ns
      last_position_ns = memory.position_ns
      assert row['long_active'] and pipeline.state['carControl'].longActive
      motion = pipeline.state['controlsState'].motionState
      assert motion.referenceTimingValid and motion.referenceSourceMonoTime == row['source_ns'] > last_manual_ns
      # This moving fixture never manufactures a stop ACK. Recovery consumes
      # its new owner publication, with no remembered object HOLD to borrow.
      assert pipeline.planner.reference_planner.radar_held_identity is None
      assert not pipeline.planner.reference_planner._radar_stop_pending
      assert all(owner['active'] and owner['phase'] != 'hold' for owner in row['owner'])
      assert all(command['active'] for command in row['can'])
    record(record_property, f'confirmed_body_driver_edge_{intervention}', rows)


@pytest.mark.parametrize('change', ['observed_out', 'new_identity'])
def test_confirmed_body_history_still_yields_to_current_out_or_identity_change(change, record_property):
  with control_pipeline(speed=8.0) as pipeline:
    target, identity, _, rows = confirmed_offset_target(pipeline)
    confirmed_position_ns = pipeline.radard.fusion_control._tracks[identity[1]].position_ns
    if change == 'observed_out':
      # A bounded lateral transition keeps the same raw identity while valid
      # current model/lane geometry observes it leaving the existing 2 m band.
      for index in range(1, 23):
        rows.append(pipeline.step(speed=8.0, radar=[Target(20.0, 8.0, lateral=1.3 + .1*index, slot=2, oem_id=16)]))
      outside = Target(20.0, 8.0, lateral=3.5, slot=2, oem_id=16)
      rows.extend(pipeline.step(speed=8.0, radar=[outside]) for _ in range(12))
      adapter = pipeline.radard.fusion_control
      same_track, = [track for track in pipeline.state[SERVICE].fusion.tracks if track.trackId == identity[1]]
      assert same_track.active and same_track.oemTrackId == target.oem_id
      assert same_track.positionMonoTime > confirmed_position_ns
      assert rows[-1]['radar']['fusionSessionId'] == identity[0]
      assert identity[1] in rows[-1]['radar']['fusionClearedTracks']
      assert identity[1] in adapter._cleared and identity[1] not in adapter._tracks
      assert not rows[-1]['radar']['fusionRiskUnresolved']
      assert not any(lead['evidence']['trackId'] == identity[1] and lead['evidence']['brakeQualified'] for lead in objects(rows[-1]))
    else:
      replacement = Target(target.distance, target.speed, lateral=target.lateral, slot=target.slot, oem_id=17)
      rows.extend(pipeline.step(speed=8.0, radar=[replacement]) for _ in range(20))
      tracks = [track for track in pipeline.state[SERVICE].fusion.tracks if track.active and track.oemTrackId == 17]
      track, = tracks
      assert track.trackId != identity[1]
      assert int(track.trackId) not in pipeline.radard.fusion_control._tracks
      assert not any(lead['evidence']['trackId'] == track.trackId for lead in objects(rows[-1]))
    record(record_property, f'confirmed_body_revocation_{change}', rows)


def establish_prior_object_hold(pipeline):
  """Explicit lifecycle-test premise, paired with the actual owner receipt.

  These stationary fixtures test an already established barrier, not how the
  car attributed its original stop. Rolling acquisition is verified separately
  by the unchanged ideal-plant trajectory and its real MPC/owner/CAN chain.
  """
  state, ref = pipeline.state, pipeline.planner.reference_planner
  evidence = state['radarState'].leadOne.evidence
  identity = (evidence.sessionId, int(evidence.trackId))
  assert state['radarState'].leadOne.present and evidence.stopQualified
  motion = state['controlsState'].motionState
  assert motion.phase == 'hold' and motion.resolvedStop
  assert motion.reasons & int(StopReason.MPC | StopReason.CLOSE_LEAD)
  assert motion.referenceTimingValid and not motion.degraded and not motion.waitingForFreshInput
  assert state['carControl'].actuators.accel <= 0.0
  ref._radar_stop_candidate = (identity, int(motion.referenceSourceMonoTime), int(motion.referenceProducerMonoTime),
                              int(motion.referenceProducerInstanceMonoTime))
  ref.adopt_executed_radar_stop(state, now_ns=max(state.logMonoTime['carControl'], state.logMonoTime['controlsState']))
  assert ref.radar_held_identity == identity
  return identity


@pytest.mark.parametrize('lead_speed', [0.0, 4.0], ids=['new_stationary', 'new_moving'])
def test_never_seen_by_vision_radar_target_changes_real_honda_braking(lead_speed, record_property):
  arms = {}
  for in_path in (False, True):
    with control_pipeline() as pipeline:
      rows = []
      for frame in range(36):
        distance = 36.0 + (lead_speed - 10.0) * frame * .05
        rows.append(pipeline.step(speed=10.0, radar=[Target(distance, lead_speed, lateral=0.0 if in_path else 7.0)]))
      arms[in_path] = rows
      record(record_property, f'radar_only_in_path_{in_path}', rows)
  baseline, candidate = arms[False], arms[True]
  assert not any(row['radar']['leadOne']['present'] for row in baseline)
  admitted = [row for row in candidate if any(obj['evidence']['source'] == 'radar' for obj in objects(row))]
  assert admitted and admitted[0]['time'] <= .35
  for row in admitted:
    lead = row['radar']['leadOne']
    assert lead['present'] and lead['radar'] and lead['modelProb'] == 0.0
    assert lead['evidence']['brakeQualified'] and lead['evidence']['stopQualified']
    assert row['solver'] == 0
  assert candidate[-1]['target'] < -.1
  assert candidate[-1]['target'] < baseline[-1]['target'] - .2
  assert any(sample['brake'] and sample['accel'] < -.1 and sample['gas'] == -30000
             for row in candidate[-6:] for sample in row['can'])
  assert all(sample['active'] for row in candidate[-6:] for sample in row['can'])


def test_same_car_uses_radar_motion_and_brakes_through_long_visual_gap(record_property):
  rows = []
  with control_pipeline() as pipeline:
    for frame in range(46):
      distance = 32.0 - 6.0 * frame * .05
      radar = Target(distance, 4.0)
      # A fixed visual range bias must not average away the radar replacement.
      visual = [Target(distance + 2.0, 4.0)] if frame < 24 else []
      rows.append(pipeline.step(speed=10.0, radar=[radar], visual=visual))
  record(record_property, 'same_car_then_visual_gap', rows)
  bound = rows[23]['radar']['leadOne']
  assert bound['evidence']['source'] == 'fused'
  assert abs(bound['dRel'] - rows[23]['sensor_targets'][0]['distance']) < .35
  assert abs(bound['vLeadK'] - 4.0) < .35
  identity = bound['evidence']['trackId']
  # The final one second is beyond the obsolete 500 ms visual-loss cap and
  # beyond the formal vision probability filter's disappearance interval.
  for row in rows[34:]:
    lead = row['radar']['leadOne']
    assert lead['present'] and lead['evidence']['source'] == 'radar'
    assert lead['evidence']['trackId'] == identity and lead['modelProb'] == 0.0
    assert lead['evidence']['brakeQualified'] and row['target'] < 0.0
    assert row['solver'] == 0
    assert all(sample['accel'] <= 0.0 for sample in row['can'])


@pytest.mark.parametrize('recovery', ['same_identity', 'missing_aux', 'different_oem'])
def test_saturation_reacquisition_requires_fresh_components_without_latched_loss(recovery, monkeypatch, record_property):
  """Missing components revoke authority without making an old ID a later veto."""
  rows = []
  with control_pipeline() as pipeline:
    original_scan = pipeline_support.scan_frames

    def recovery_scan(**fields):
      # Only the sensor input changes. Decoder, fusion, RadarD, planner and
      # Honda control all run normally, including their serialized messages.
      frame = pipeline.frame
      fields['life'] = 4047 + 2 * frame if frame <= 24 else 4094
      fields['index'] = (frame - int(frame >= 26)) % 16
      frames = original_scan(**fields)
      if frame == 25:
        return frames[:4]  # Position alone cannot prove the return.
      if frame >= 26 and recovery == 'missing_aux':
        return []
      if frame == 26:
        return frames[4:]  # Same scan's native velocity arrives 50 ms later.
      return frames

    monkeypatch.setattr(pipeline_support, 'scan_frames', recovery_scan)
    for frame in range(43):
      oem_id = 61 if recovery == 'different_oem' and frame >= 25 else 60
      row = pipeline.step(speed=10.0, radar=[Target(36.0 - 6.0 * frame * .05, 4.0, slot=1, oem_id=oem_id)])
      observation = pipeline.state[SERVICE]
      row['fusion_tracks'] = [track.to_dict() for track in observation.fusion.tracks]
      row['raw_observations'] = [sample.to_dict() for sample in observation.observations]
      row['consumer_degraded'] = pipeline.planner.reference_planner.radar_consumption.degraded
      rows.append(row)
  record(record_property, f'saturation_reacquisition_{recovery}', rows)

  initial = rows[23]['radar']['leadOne']
  identity = initial['evidence']['sessionId'], initial['evidence']['trackId']
  assert initial['present'] and initial['evidence']['source'] == 'radar'
  assert initial['evidence']['followQualified'] and initial['evidence']['brakeQualified']
  assert not any(row['radar']['fusionRiskUnresolved'] for row in rows)
  assert rows[23]['fusion_tracks'][0]['lifecycleRaw'] == 4093
  invalid = rows[24]
  assert invalid['fusion_tracks'][0]['trackId'] == identity[1]
  assert not invalid['fusion_tracks'][0]['active']
  assert invalid['fusion_tracks'][0]['lastInvalidReason'] == 'invalidGeometry'
  assert invalid['radar']['fusionDegraded']
  position = next(sample for sample in rows[25]['raw_observations'] if sample['positionValid'])
  assert position['raw']['lifecycleRaw'] == 4094 and not position['velocityValid']
  unrefreshed = rows[25:26] if recovery == 'same_identity' else rows[25:]
  for row in unrefreshed:
    # An incomplete scan or another OEM cannot refresh the old object's
    # measurements or grant it following/departure authority.
    for track in row['fusion_tracks']:
      if track['trackId'] == identity[1]:
        assert not track['active'] and not track['velocityGood']
    for obj in objects(row):
      evidence = obj['evidence']
      if (evidence['sessionId'], evidence['trackId']) == identity:
        # A bounded prediction can still brake using the original stamps.
        assert evidence['source'] == 'predicted' and evidence['coasting']
        assert evidence['positionMonoTime'] == initial['evidence']['positionMonoTime']
        assert evidence['velocityMonoTime'] == initial['evidence']['velocityMonoTime']
        assert not evidence['followQualified'] and not evidence['departureQualified']

  if recovery == 'same_identity':
    restored = rows[26]['fusion_tracks'][0]
    assert restored['active'] and restored['velocityGood'] and restored['trackId'] == identity[1]
    assert restored['positionSeq'] == position['positionSeq']
    assert restored['positionMonoTime'] == position['positionMonoTime']
    assert restored['velocityMonoTime'] == rows[26]['source_ns']
    for row in rows[26:]:
      assert row['radar']['fusionHealthy'] and not row['radar']['fusionDegraded']
      assert not row['consumer_degraded']
      lead = row['radar']['leadOne']
      assert lead['present'] and lead['evidence']['source'] == 'radar'
      assert (lead['evidence']['sessionId'], lead['evidence']['trackId']) == identity
      assert lead['evidence']['followQualified'] and lead['evidence']['brakeQualified']
      assert not lead['evidence']['coasting'] and row['solver'] == 0
    assert rows[-1]['target'] < -.1
    assert any(sample['active'] and sample['brake'] and sample['accel'] < -.1
               for row in rows[-6:] for sample in row['can'])
  elif recovery == 'different_oem':
    # Once the replacement has its own fresh qualified measurements, the
    # missing old identity does not keep either consumer degraded.
    for row in rows[-6:]:
      assert row['radar']['fusionHealthy'] and not row['radar']['fusionDegraded']
      assert not row['consumer_degraded']
      replacement = row['radar']['leadOne']
      assert replacement['present'] and replacement['evidence']['brakeQualified']
      assert replacement['evidence']['followQualified'] and not replacement['evidence']['coasting']
      assert (replacement['evidence']['sessionId'], replacement['evidence']['trackId']) != identity
    assert rows[-1]['target'] < -.1
    assert any(sample['active'] and sample['brake'] and sample['accel'] < -.1
               for row in rows[-6:] for sample in row['can'])
  else:
    # Missing velocity never grants authority; expired predictions disappear
    # without leaving an unresolved-ID risk on later publications.
    assert objects(rows[-1]) == []


def test_near_visual_cut_in_is_not_diluted_by_a_different_far_radar_car(record_property):
  arms = {}
  for cut_in in (False, True):
    with control_pipeline() as pipeline:
      rows = []
      for frame in range(36):
        visual = [Target(50.0, 10.0)]
        if cut_in and frame >= 16:
          visual = [Target(16.0 - 8.0 * (frame - 16) * .05, 2.0), *visual]
        row = pipeline.step(speed=10.0, radar=[Target(50.0, 10.0)], visual=visual)
        row['far_binding'] = next(({
          'track_id': int(lead.trackId), 'candidate_track_id': int(lead.candidateTrackId),
          'state': str(lead.identityState), 'rejection_ns': int(lead.rejectionMonoTime),
        } for lead in pipeline.state[SERVICE].fusion.leads if lead.modelIndex == 1), {})
        rows.append(row)
      arms[cut_in] = rows
      record(record_property, f'visual_cut_in_{cut_in}', rows)
  baseline, candidate = arms[False], arms[True]
  original = candidate[15]['radar']['leadOne']
  assert original['evidence']['source'] == 'fused' and original['radar']
  far_identity = original['evidence']['trackId']
  waiting, rebound = [], []
  for row in candidate[16:]:
    near = row['radar']['leadOne']
    expected = 16.0 - 8.0 * (row['frame'] - 16) * .05
    assert near['evidence']['source'] == 'vision' and not near['radar']
    assert abs(near['dRel'] - expected) < .001
    assert len(objects(row)) == 2 and row['radar']['fusionHealthy'] and not row['radar']['fusionDegraded']
    far = row['radar']['leadTwo']
    assert far['dRel'] == pytest.approx(50.0, abs=.02) and far['vLeadK'] == pytest.approx(10.0)
    assert far['evidence']['followQualified'] and far['evidence']['brakeQualified']
    binding = row['far_binding']
    assert binding['track_id'] == binding['candidate_track_id'] == far_identity
    assert binding['rejection_ns'] == 0
    # Moving the far car from visual slot 0 to slot 1 needs a new same-car
    # confirmation. Until then its 50 m visual lead remains ahead of the
    # 50.007 m unbound radar candidate in the two ordinary lead slots.
    if binding['state'] == 'same':
      rebound.append(row['frame'])
      assert far['evidence']['source'] == 'fused' and far['radar']
      assert far['evidence']['trackId'] == far_identity and far['evidence']['visionIndex'] == 1
    else:
      waiting.append(row['frame'])
      assert binding['state'] == 'unresolved'
      assert far['evidence']['source'] == 'vision' and not far['radar']
  assert waiting and rebound and max(waiting) < min(rebound)
  assert candidate[-1]['target'] < baseline[-1]['target'] - .5
  assert any(sample['accel'] < -.5 and sample['brake'] for row in candidate[-6:] for sample in row['can'])


def test_third_lateral_object_cannot_brake_via_future_entry_or_extra_object(record_property):
  arms = {}
  for crossing in (False, True):
    with control_pipeline(speed=8.0) as pipeline:
      rows = []
      for frame in range(72):
        timestamp = frame * .05
        # The same fresh track spends many frames outside both body bands,
        # then overlaps the outer body band and finally the inner sweep.
        radar = [Target(45.0 - 8.0 * timestamp, 0.0, 3.0 - .6 * timestamp)] if crossing else []
        rows.append(pipeline.step(speed=8.0, radar=radar, visual=[Target(35.0, 8.0), Target(50.0, 8.0)]))
      arms[crossing] = rows
      record(record_property, f'third_crossing_{crossing}', rows)
  baseline, candidate = arms[False], arms[True]
  outside = [row for row in candidate[8:] if row['sensor_targets'][0]['lateral'] > 2.1]
  assert len(outside) >= 12
  for row in outside:
    assert row['radar']['leadOne']['present'] and row['radar']['leadTwo']['present']
    assert row['radar']['leadOne']['dRel'] >= 34.9 and row['radar']['leadTwo']['dRel'] >= 49.9
    assert len(objects(row)) == 2 and all(obj['evidence']['source'] == 'vision' for obj in objects(row))
    assert row['target'] == pytest.approx(baseline[row['frame']]['target'], abs=1e-6)
    assert [sample['accel'] for sample in row['can']] == pytest.approx(
      [sample['accel'] for sample in baseline[row['frame']]['can']], abs=1e-6)
  # Even after current lateral overlap, its measured crossing motion is not
  # an ordinary forward follow lead. No third-object braking path remains.
  for row in candidate[8:]:
    assert len(objects(row)) == 2
    assert all(obj['evidence']['source'] == 'vision' for obj in objects(row))
    assert row['target'] == pytest.approx(baseline[row['frame']]['target'], abs=1e-6)
    assert [sample['accel'] for sample in row['can']] == pytest.approx(
      [sample['accel'] for sample in baseline[row['frame']]['can']], abs=1e-6)


@pytest.mark.parametrize('outcome', ['dropout', 'departure'])
def test_established_radar_hold_survives_dropout_and_releases_for_same_car_departure(outcome, record_property):
  """Dropout/departure lifecycle after an explicitly established object hold."""
  rows = []
  with control_pipeline(speed=0.0) as pipeline:
    for frame in range(100):
      elapsed = max(0.0, (frame - 28) * .05)
      departing = outcome == 'departure' and frame >= 28
      target = Target(3.0 + .4 * elapsed**2 if departing else 3.0, .8 * elapsed if departing else 0.0)
      rows.append(pipeline.step(speed=0.0, radar=[target], observer_online=outcome != 'dropout' or frame < 28))
      if frame == 27:
        establish_prior_object_hold(pipeline)
  record(record_property, f'hold_{outcome}', rows)
  held = rows[27]['radar']['leadOne']
  assert rows[27]['radar']['fusionHealthy'] and held['present']
  assert held['evidence']['source'] == 'radar' and held['evidence']['stopQualified']
  assert held['evidence']['trackId'] > 0
  assert all(owner['phase'] == 'hold' and owner['stop'] for row in rows[20:28] for owner in row['owner'])
  assert any(sample['stop'] and sample['brake'] for row in rows[20:28] for sample in row['can'])
  if outcome == 'dropout':
    frozen = rows[27]['observer_ns'], rows[27]['observer_position_ns']
    assert all((row['observer_ns'], row['observer_position_ns']) == frozen for row in rows[28:])
    coasts = [obj for row in rows[28:36] for obj in objects(row) if obj['evidence']['coasting']]
    assert coasts
    for obj in coasts:
      evidence = obj['evidence']
      assert evidence['source'] == 'predicted' and evidence['brakeQualified']
      assert not evidence['followQualified'] and not evidence['departureQualified']
      assert evidence['positionMonoTime'] == held['evidence']['positionMonoTime']
      assert evidence['velocityMonoTime'] == held['evidence']['velocityMonoTime']
    assert any(row['radar'].get('fusionDegraded', False) for row in rows[36:])
    assert not any(row['radar']['fusionRiskUnresolved'] for row in rows)
    # The executed object-specific HOLD remains its own state. Sensor loss
    # cannot release it even though no persistent missing-ID risk is kept.
    for row in rows[36:]:
      assert not row['release']
      assert all(owner['phase'] == 'hold' and owner['stop'] for owner in row['owner'])
      assert all(sample['brake'] and not sample['release'] and sample['accel'] <= 0.0 for sample in row['can'])
  else:
    assert not any(row['radar']['fusionRiskUnresolved'] for row in rows)
    identity = rows[27]['radar']['leadOne']['evidence']['trackId']
    assert all(row['radar']['leadOne']['evidence']['trackId'] == identity for row in rows[28:])
    assert any(row['release'] and not row['stop'] for row in rows[28:])
    assert any(sample['release'] and not sample['stop'] and sample['accel'] > 0.0
               for row in rows[28:] for sample in row['can'])


@pytest.mark.parametrize('scope_exit', ['experimental_disabled', 'dec_disabled', 'scope_stale'])
def test_scope_exit_cannot_release_radar_hold_before_driver_starts_a_new_engagement(scope_exit, record_property):
  """Configuration/transport loss cannot release an explicitly established hold."""
  rows = []
  with control_pipeline(speed=0.0) as pipeline:
    for _ in range(28):
      rows.append(pipeline.step(speed=0.0, radar=[Target(3.0, 0.0)]))
    assert all(owner['phase'] == 'hold' and owner['stop'] for row in rows[20:] for owner in row['owner'])
    held_identity = establish_prior_object_hold(pipeline)
    if scope_exit == 'experimental_disabled':
      pipeline.params.put_bool('ExperimentalMode', False, block=True)
    elif scope_exit == 'dec_disabled':
      pipeline.params.put_bool('DynamicExperimentalControl', False, block=True)
    pipeline.settings = ObserverSettings.read(pipeline.params)
    revoked = []
    for frame in range(48):
      # Stop selfdriveState updates for 1.2 seconds, then restore them. The
      # source timestamp remains old; the fusion gate itself is not mocked.
      online = scope_exit != 'scope_stale' or frame >= 24
      # One speed-estimator excursion is not a departure receipt. It occurs
      # after each scope gate has actually revoked the radar publication.
      speed = STOP_SPEED + .05 if frame == 18 else 0.0
      row = pipeline.step(speed=speed, radar=[Target(3.0, 0.0)], scope_online=online)
      rows.append(row)
      revoked.append(row)
      assert pipeline.planner.reference_planner.radar_held_identity == held_identity
      assert not row['release']
      phases = ('hold', 'approach') if frame == 18 else ('hold',)
      assert all(owner['phase'] in phases and owner['stop'] for owner in row['owner'])
      assert all(sample['brake'] and sample['stop'] and not sample['release'] and sample['accel'] <= 0.0
                 for sample in row['can'])
    assert any(not row['radar'].get('fusionActive', False) for row in revoked)
    if scope_exit == 'scope_stale':
      assert any(row['radar'].get('fusionActive', False) for row in revoked[30:])

    # A real driver disengagement is a separate lifecycle, unlike a missing
    # sensor or DEC publication. Give the next engagement fresh healthy scans
    # of an independent out-of-path target, with no old blocking vehicle.
    pipeline.params.put_bool('ExperimentalMode', True, block=True)
    pipeline.params.put_bool('DynamicExperimentalControl', True, block=True)
    pipeline.settings = ObserverSettings.read(pipeline.params)
    side_target = Target(20.0, 0.0, lateral=7.0, slot=1, oem_id=2)
    for _ in range(24):
      row = pipeline.step(speed=0.0, radar=[side_target], engaged=False)
      rows.append(row)
      assert pipeline.planner.reference_planner.radar_held_identity is None
      assert all(not sample['active'] and not sample['release'] for sample in row['can'])
    reengaged = [pipeline.step(speed=0.0, radar=[side_target]) for _ in range(48)]
    rows.extend(reengaged)
    assert pipeline.planner.reference_planner.radar_held_identity is None
    assert any(row['release'] and not row['stop'] for row in reengaged)
    assert any(sample['active'] and sample['release'] and sample['accel'] > 0.0
               for row in reengaged for sample in row['can'])
  record(record_property, f'hold_scope_exit_{scope_exit}', rows)


@pytest.mark.parametrize('intervention', ['model_stop', 'gas', 'brake', 'cancel'])
def test_radar_cannot_cancel_model_stop_or_driver_authority(intervention, record_property):
  rows = []
  with control_pipeline(speed=8.0) as pipeline:
    for frame in range(44):
      intervened = frame >= 24
      model_stop = intervened and intervention == 'model_stop'
      rows.append(pipeline.step(speed=8.0, radar=[Target(32.0, 8.0)], model_stop=model_stop,
                                model_accel=-2.0 if model_stop else .6,
                                pedal=intervention if intervened else None,
                                engaged=not (intervened and intervention == 'cancel')))
  record(record_property, f'authority_{intervention}', rows)
  assert any(objects(row) for row in rows[12:24])
  if intervention == 'model_stop':
    assert any(row['dec_mode'] == 'blended' for row in rows[24:])
    assert all(row['stop'] and row['reasons'] & StopReason.E2E and not row['release'] for row in rows[-5:])
    assert all(sample['brake'] and sample['accel'] <= -1.5 for row in rows[-5:] for sample in row['can'])
  else:
    assert all(not sample['active'] and not sample['resume'] and not sample['release']
               and not sample['brake'] and sample['accel'] == 0.0
               for row in rows[24:] for sample in row['can'])


@pytest.mark.parametrize('scope,enabled,unavailable', [
  ('enabled', True, False), ('sp_invalid', False, True), ('sp_invalid_dec_off', False, True),
  ('sp_stale', False, True), ('sp_future', False, True), ('sp_unseen', False, True),
  ('selfdrive_stale', False, True), ('selfdrive_invalid', False, True),
  ('selfdrive_invalid_experimental_off', False, True), ('experimental_off', False, False),
  ('experimental_off_bad_sp', False, False), ('dec_off', False, False), ('configuration_exit', False, False),
  ('configuration_unavailable', False, True),
])
def test_scope_feedback_failure_is_distinct_from_an_explicit_exit(scope, enabled, unavailable):
  with control_pipeline() as pipeline:
    state = pipeline.state
    now = pipeline_support.START_NS
    # Absent scopeVersion explicitly exercises the legacy Event.valid gate.
    feedback = messaging.new_message('longitudinalPlanSP').longitudinalPlanSP
    feedback.dec.enabled = feedback.dec.active = True
    state['longitudinalPlanSP'] = feedback
    for service in ('selfdriveState', 'longitudinalPlanSP'):
      state.valid[service] = state.seen[service] = True
      state.logMonoTime[service] = now
    if scope in ('sp_invalid', 'sp_invalid_dec_off', 'experimental_off_bad_sp'):
      state.valid['longitudinalPlanSP'] = False
    if scope in ('sp_invalid_dec_off', 'dec_off'):
      feedback.dec.enabled = feedback.dec.active = False
    if scope == 'sp_stale':
      state.logMonoTime['longitudinalPlanSP'] -= SCOPE_MAX_AGE_NS + 1
    if scope == 'sp_future':
      state.logMonoTime['longitudinalPlanSP'] += 1
    if scope == 'sp_unseen':
      state.seen['longitudinalPlanSP'] = False
    if scope == 'selfdrive_stale':
      state.logMonoTime['selfdriveState'] -= SCOPE_MAX_AGE_NS + 1
    if scope in ('selfdrive_invalid', 'selfdrive_invalid_experimental_off'):
      state.valid['selfdriveState'] = False
    if scope in ('experimental_off', 'experimental_off_bad_sp', 'selfdrive_invalid_experimental_off'):
      state['selfdriveState'].experimentalMode = False
    if scope == 'configuration_exit':
      pipeline.params.put_bool('ExperimentalMode', False, block=True)
    if scope == 'configuration_unavailable':
      pipeline.radard.params = Mock()
      pipeline.radard.params.get_bool.side_effect = OSError('unavailable settings fixture')
    assert pipeline.radard.control_enabled(state, now) is enabled
    assert pipeline.radard.control_scope_unavailable is unavailable


@pytest.mark.parametrize('feedback_fault', ['invalid', 'invalid_dec_off', 'stale', 'future', 'unseen'])
def test_feedback_gap_preserves_real_radar_hold_without_latched_loss(feedback_fault, record_property):
  """Invalid scope cannot remove the explicitly established object-hold premise."""
  rows = []
  with control_pipeline(speed=0.0) as pipeline:
    for _ in range(28):
      rows.append(pipeline.step(speed=0.0, radar=[Target(3.0, 0.0)]))
    held = establish_prior_object_hold(pipeline)
    assert all(owner['phase'] == 'hold' and owner['stop'] for owner in rows[-1]['owner'])
    for _ in range(10):
      rows.append(pipeline.step(speed=0.0, observer_online=False))
    adapter = pipeline.radard.fusion_control
    assert rows[-1]['radar']['fusionDegraded'] and not rows[-1]['radar']['fusionRiskUnresolved']
    for _ in range(12):
      state = pipeline.state
      if feedback_fault in ('invalid', 'invalid_dec_off'):
        state.valid['longitudinalPlanSP'] = False
        feedback = state['longitudinalPlanSP'].as_builder()
        feedback.dec.scopeValid = False
        if feedback_fault == 'invalid_dec_off':
          feedback.dec.enabled = feedback.dec.active = False
        state['longitudinalPlanSP'] = feedback.as_reader()
      if feedback_fault == 'stale':
        state.logMonoTime['longitudinalPlanSP'] -= SCOPE_MAX_AGE_NS + 1
      if feedback_fault == 'future':
        state.logMonoTime['longitudinalPlanSP'] += 1_000_000_000
      if feedback_fault == 'unseen':
        state.seen['longitudinalPlanSP'] = False
      # A different, healthy visual lead remains available, but it cannot
      # prove departure of the previously adopted stopped radar vehicle.
      row = pipeline.step(speed=0.0, visual=[Target(40.0, 6.0)], observer_online=False)
      rows.append(row)
      assert pipeline.radard.fusion_control is adapter
      assert not row['radar']['fusionActive'] and row['radar']['fusionReason'] == 'controlScopeUnavailable'
      assert row['radar']['fusionDegraded'] and not row['radar']['fusionRiskUnresolved']
      assert row['radar'].get('radarObjects', []) == []
      assert row['radar']['leadOne'] == row['radar']['visionLeadOne'] and row['radar']['leadOne']['present']
      assert pipeline.planner.reference_planner.radar_held_identity == held
      assert not row['release']
      assert all(owner['phase'] == 'hold' and owner['stop'] for owner in row['owner'])
      assert all(sample['brake'] and not sample['release'] and sample['accel'] <= 0.0 for sample in row['can'])
    # Scope recovery alone leaves the current observer stale and does not
    # release the explicitly established object HOLD.
    row = pipeline.step(speed=0.0, visual=[Target(40.0, 6.0)], observer_online=False)
    rows.append(row)
    assert row['radar']['fusionActive'] and row['radar']['fusionDegraded']
    assert not row['radar']['fusionRiskUnresolved'] and not row['release']
    # Fresh sensing restores current object authority independently of the
    # old HOLD. A stationary close lead still supplies no departure proof.
    for _ in range(12):
      row = pipeline.step(speed=0.0, radar=[Target(3.0, 0.0)])
      rows.append(row)
    assert row['radar']['fusionHealthy'] and not row['radar']['fusionDegraded']
    assert not row['radar']['fusionRiskUnresolved']
    assert not pipeline.planner.reference_planner.radar_consumption.degraded
    assert pipeline.planner.reference_planner.radar_held_identity == held
    lead = row['radar']['leadOne']
    assert lead['present'] and lead['evidence']['brakeQualified'] and lead['evidence']['stopQualified']
    assert not lead['evidence']['coasting'] and not row['release']
    assert all(owner['phase'] == 'hold' and owner['stop'] for owner in row['owner'])
    assert all(sample['brake'] and not sample['release'] and sample['accel'] <= 0.0 for sample in row['can'])
    # A fresh explicit configuration exit is a different lifecycle boundary.
    pipeline.params.put_bool('ExperimentalMode', False, block=True)
    for _ in range(21):  # allow the ordinary one-second settings refresh
      row = pipeline.step(speed=0.0, visual=[Target(40.0, 6.0)], observer_online=False)
      rows.append(row)
    assert not row['radar']['fusionActive'] and not row['radar']['fusionRiskUnresolved']
  record(record_property, f'scope_feedback_{feedback_fault}', rows)
