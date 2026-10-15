"""CAN-origin regressions for active radar fusion and its execution boundary."""
import json
from unittest.mock import Mock

import pytest

from openpilot.cereal import messaging
from openpilot.selfdrive.car.honda_radar_control import CUT_IN_REASON, SWEPT_HALF_WIDTH
from openpilot.selfdrive.car.honda_radar_observer_config import SERVICE, SCOPE_MAX_AGE_NS, ObserverSettings
from openpilot.selfdrive.car.tests import honda_radar_control_test_support as pipeline_support
from openpilot.selfdrive.car.tests.honda_radar_control_test_support import Target, control_pipeline
from openpilot.selfdrive.controls.lib.motion_supervisor import StopReason


def objects(row):
  assert row['radar'].get('radarObjects', []) == []
  return [row['radar'][name] for name in ('leadOne', 'leadTwo') if row['radar'][name].get('present')]


def record(record_property, name, rows):
  record_property(name, json.dumps(rows, separators=(',', ':'), allow_nan=False))


def body_target(frame, kind, side):
  elapsed = frame * .05
  lateral = 1.8 if kind == 'parallel' else (4.0 if kind == 'side' else 2.2) - .4 * elapsed
  extent = None if kind == 'no_edges' else ((.6, -.9) if side > 0 else (.9, -.6))
  return Target(28.0 - 7.62 * elapsed, 13.3, lateral=side * lateral, oem_id=46, body_offsets=extent)


def body_row(pipeline, kind, side):
  """Capture unmodified producer and MPC receipts for one raw sensor scan."""
  row = pipeline.step(speed=20.92, radar=[body_target(pipeline.frame, kind, side)])
  observation = pipeline.state[SERVICE]
  row['body_tracks'] = [track.to_dict() for track in observation.fusion.tracks]
  row['body_scans'] = [sample.to_dict() for sample in observation.observations if sample.positionValid]
  mpc, planner = pipeline.planner.mpc, pipeline.planner.reference_planner
  body_leads = [lead for lead in mpc._source_leads if CUT_IN_REASON in str(lead.evidence.reason)]
  row['mpc_body_tracks'] = [int(lead.evidence.trackId) for lead in body_leads]
  row['mpc_body_follow_geometry'] = [int(lead.evidence.followGeometryMonoTime) for lead in body_leads]
  row['fcw_count'], row['fcw'] = int(mpc.crash_cnt), bool(pipeline.planner.fcw)
  row['consumer_degraded'] = planner.radar_consumption.degraded
  return row


@pytest.mark.parametrize('side', [-1, 1], ids=['right', 'left'])
def test_raw_measured_body_cut_in_brakes_through_fusion_mpc_and_honda_can(side, record_property):
  """Only sensor edges/motion differ; no formal lead or controller is patched."""
  arms = {}
  for kind in ('no_edges', 'side', 'parallel', 'intruding'):
    with control_pipeline(speed=20.92, cruise=24.0) as pipeline:
      rows = [body_row(pipeline, kind, side) for _ in range(34)]
      arms[kind] = rows
      record(record_property, f'raw_body_{side}_{kind}', rows)
      if kind == 'intruding':
        withdrawn = [body_row(pipeline, 'no_edges', side) for _ in range(3)]
        record(record_property, f'raw_body_{side}_withdrawn', withdrawn)
        for row in withdrawn:
          assert row['body_tracks'] and all(not track['bodyValid'] for track in row['body_tracks'])
          assert not objects(row) and not row['mpc_body_tracks']
          assert not row['radar']['fusionClearedTracks']

  baseline = arms['no_edges']
  for kind, rows in arms.items():
    mature = [track for row in rows for track in row['body_tracks']
              if track['active'] and track['mature'] and track['velocityGood'] and track['lateralVelocityValid']]
    assert mature, f'{kind}: the real producer never supplied mature motion evidence'
    assert any(row['body_scans'] for row in rows)
    if kind == 'no_edges':
      assert all(not track['bodyValid'] for row in rows for track in row['body_tracks'])
    else:
      assert all(track['bodyValid'] for track in mature)
      for row in rows:
        for track in row['body_tracks']:
          if not track['bodyValid']:
            continue
          scan = next(sample for sample in row['body_scans']
                      if sample['positionSeq'] == track['positionSeq'] and sample['raw']['oemTrackId'] == track['oemTrackId'])
          assert track['bodyMonoTime'] == track['positionMonoTime'] == scan['positionMonoTime'] == row['source_ns']
          assert track['bodyAvailableMonoTime'] == row['source_ns']
          assert scan['raw']['edgeARaw'] != scan['raw']['edgeBRaw']
          assert track['bodyWidth'] == pytest.approx(1.5, abs=.06)  # Quantized angular CAN edges.
    if kind != 'intruding':
      assert all(not objects(row) and not row['mpc_body_tracks'] for row in rows)
      # Side/parallel motion and a missing width must not create braking merely
      # because radar reports a closing object outside the ordinary point path.
      for row, expected in zip(rows, baseline, strict=True):
        assert row['target'] == pytest.approx(expected['target'], abs=1e-6)
        assert [sample['accel'] for sample in row['can']] == pytest.approx(
          [sample['accel'] for sample in expected['can']], abs=1e-6)

  admitted = [row for row in arms['intruding'] if objects(row)]
  assert admitted, 'Fresh raw body intrusion never reached a formal brake-only lead'
  for row in admitted:
    lead, = objects(row)
    claim = lead['evidence']
    assert CUT_IN_REASON in claim['reason'] and claim['brakeQualified']
    assert not any(claim[purpose + 'Qualified'] for purpose in ('follow', 'stop', 'departure', 'fcw'))
    assert not claim['coasting'] and not claim['accelerationValid'] and lead['aLeadK'] == 0.0
    assert claim['followGeometryMonoTime'] == 0 and row['mpc_body_follow_geometry'] == [0]
    assert abs(lead['yRel']) > SWEPT_HALF_WIDTH  # The reflection has not entered the point path.
    assert row['mpc_body_tracks'] == [claim['trackId']]
    assert row['solver'] == 0 and row['radar']['fusionHealthy'] and not row['consumer_degraded']
    assert row['fcw_count'] == 0 and not row['fcw'] and not row['moving']
    assert all(owner['phase'] != 'hold' for owner in row['owner'])
    assert all(not command['resume'] for command in row['can'])
    if row['target'] < 0.0:
      assert not row['release']
    assert not row['reasons'] & int(StopReason.CLOSE_LEAD)
  assert any(row['target'] < baseline[row['frame']]['target'] - .2 for row in admitted)
  assert any(sample['active'] and sample['brake'] and sample['accel'] < -.2 and sample['gas'] == -30000
             for row in admitted for sample in row['can'])


def confirmed_offset_target(pipeline, *, visual_gap=True):
  """Establish current visual body support independently of its radar point."""
  target = Target(20.0, 8.0, lateral=1.3, slot=2, oem_id=16)
  rows = [pipeline.step(speed=8.0, radar=[target], visual=[target]) for _ in range(24)]
  matched = next(lead for lead in objects(rows[-1]) if lead['evidence']['source'] == 'fused')
  identity = matched['evidence']['sessionId'], matched['evidence']['trackId']
  memory = pipeline.radard.fusion_control._tracks[identity[1]]
  assert memory.path.confirmed and SWEPT_HALF_WIDTH < abs(memory.path.offset) < 2.0
  if visual_gap:
    # Fresh CAN preserves the physical track, while absence of current vision
    # withdraws this outer point's ordinary authority on the first new frame.
    gap = [pipeline.step(speed=8.0, radar=[target]) for _ in range(24)]
    rows.extend(gap)
    for row in gap:
      assert not objects(row)
      assert row['radar']['fusionHealthy'] and not row['radar']['fusionDegraded']
      assert not row['radar']['fusionClearedTracks'] and not row['radar']['fusionRiskUnresolved']
    assert identity[1] not in pipeline.radard.fusion_control._tracks
  return target, identity, memory.generation, rows


@pytest.mark.parametrize('intervention', ['gas', 'brake', 'disabled'])
def test_driver_edges_do_not_revive_past_outer_band_qualification(intervention, record_property):
  with control_pipeline(speed=8.0) as pipeline:
    target, identity, generation, rows = confirmed_offset_target(pipeline)
    adapter = pipeline.radard.fusion_control
    last_position_ns = rows[-1]['source_ns']
    for _ in range(4):
      row = pipeline.step(speed=8.0, radar=[target], engaged=intervention != 'disabled',
                          pedal=intervention if intervention != 'disabled' else None)
      rows.append(row)
      assert pipeline.radard.fusion_control is adapter
      current, = [track for track in pipeline.state[SERVICE].fusion.tracks if track.trackId == identity[1]]
      assert current.active and current.generation == generation and current.positionMonoTime > last_position_ns
      last_position_ns = current.positionMonoTime
      assert not objects(row) and identity[1] not in adapter._tracks
      assert row['radar']['fusionHealthy'] and not row['radar']['fusionDegraded']
      assert not row['radar']['fusionClearedTracks']
      assert not row['long_active'] and not pipeline.state['carControl'].longActive
      assert all(not owner['active'] and owner['phase'] == 'off' and owner['accel'] == 0.0 for owner in row['owner'])
      assert all(not command['active'] and not command['brake'] and command['accel'] == 0.0
                 and not command['release'] and not command['resume'] for command in row['can'])
    last_manual_ns = rows[-1]['source_ns']
    for _ in range(12):
      row = pipeline.step(speed=8.0, radar=[target])
      rows.append(row)
      current, = [track for track in pipeline.state[SERVICE].fusion.tracks if track.trackId == identity[1]]
      assert current.active and current.generation == generation and current.positionMonoTime > last_position_ns
      last_position_ns = current.positionMonoTime
      assert not objects(row) and identity[1] not in adapter._tracks
      assert row['radar']['fusionHealthy'] and not row['radar']['fusionDegraded']
      assert not row['radar']['fusionClearedTracks']
      assert row['long_active'] and pipeline.state['carControl'].longActive
      motion = pipeline.state['controlsState'].motionState
      assert motion.referenceTimingValid and motion.referenceSourceMonoTime == row['source_ns'] > last_manual_ns
      assert not motion.waitingForFreshInput and not motion.degraded
      # Recovery consumes current plans without reviving the former target.
      assert all(owner['active'] and owner['phase'] != 'hold' for owner in row['owner'])
      assert all(command['active'] for command in row['can'])
    record(record_property, f'outer_qualification_driver_edge_{intervention}', rows)


def test_scope_recovery_cannot_reuse_outer_confirmation_as_current_occupancy(record_property):
  with control_pipeline(speed=8.0) as pipeline:
    target, identity, _, rows = confirmed_offset_target(pipeline, visual_gap=False)
    adapter = pipeline.radard.fusion_control
    for _ in range(6):
      # The observer continues current sensing while RadarD cannot establish
      # control scope. This gap must not renew the former path qualification.
      state = pipeline.state
      state.valid['longitudinalPlanSP'] = False
      feedback = state['longitudinalPlanSP'].as_builder()
      feedback.dec.scopeValid = False
      state['longitudinalPlanSP'] = feedback.as_reader()
      row = pipeline.step(speed=8.0, radar=[target])
      rows.append(row)
      assert pipeline.radard.fusion_control is adapter
      assert not row['radar']['fusionActive'] and row['radar']['fusionReason'] == 'controlScopeUnavailable'
      assert row['radar']['fusionDegraded'] and not objects(row)
    for _ in range(12):
      row = pipeline.step(speed=8.0, radar=[target])
      rows.append(row)
      assert row['radar']['fusionActive'] and row['radar']['fusionHealthy'] and not row['radar']['fusionDegraded']
      assert not objects(row) and identity[1] not in adapter._tracks
      assert not row['radar']['fusionClearedTracks'] and not row['radar']['fusionRiskUnresolved']
      motion = pipeline.state['controlsState'].motionState
      assert not motion.degraded and not motion.waitingForFreshInput
    # Current same-car visual support can requalify the measured target; a
    # scope gap must not turn withdrawal into a persistent denial of sensing.
    rows.extend(pipeline.step(speed=8.0, radar=[target], visual=[target]) for _ in range(24))
    lead = next(lead for lead in objects(rows[-1]) if lead['evidence']['source'] == 'fused')
    assert (lead['evidence']['sessionId'], lead['evidence']['trackId']) == identity
    assert lead['evidence']['followQualified'] and lead['evidence']['stopQualified']
    record(record_property, 'outer_qualification_scope_recovery', rows)


@pytest.mark.parametrize('change', ['observed_out', 'new_identity'])
def test_withdrawn_outer_target_keeps_current_out_and_identity_change_distinct(change, record_property):
  with control_pipeline(speed=8.0) as pipeline:
    target, identity, _, rows = confirmed_offset_target(pipeline)
    last_outer_position_ns = rows[-1]['source_ns']
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
      assert same_track.positionMonoTime > last_outer_position_ns
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


def assert_current_radar_hold(pipeline):
  """Check the real sensed target and executed hold without injecting a latch."""
  state = pipeline.state
  evidence = state['radarState'].leadOne.evidence
  identity = (evidence.sessionId, int(evidence.trackId))
  assert state['radarState'].leadOne.present and evidence.stopQualified
  motion = state['controlsState'].motionState
  assert motion.phase == 'hold' and motion.resolvedStop
  assert motion.reasons & int(StopReason.MPC | StopReason.CLOSE_LEAD)
  assert motion.referenceTimingValid and not motion.degraded and not motion.waitingForFreshInput
  assert state['carControl'].actuators.accel <= 0.0
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
  # Fresh radar remains independently usable throughout a long visual gap.
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
    assert not any((obj['evidence']['sessionId'], obj['evidence']['trackId']) == identity for obj in objects(row))
    assert all(not obj['evidence']['coasting'] and obj['evidence']['source'] != 'predicted' for obj in objects(row))

  if recovery == 'same_identity':
    restored = rows[26]['fusion_tracks'][0]
    assert restored['active'] and restored['velocityGood'] and restored['trackId'] == identity[1]
    assert restored['positionSeq'] == position['positionSeq']
    assert restored['positionMonoTime'] == position['positionMonoTime']
    assert restored['velocityMonoTime'] == rows[26]['source_ns']
    assert restored['nativeMotion']['sampleCount'] == 1 and not restored['nativeMotion']['qualified']
    for row in rows[26:]:
      assert row['radar']['fusionHealthy'] and not row['radar']['fusionDegraded']
      assert not row['consumer_degraded']
      quality = row['fusion_tracks'][0]['nativeMotion']
      assert quality['sampleCount'] == row['frame'] - 25
      assert quality['firstPositionMonoTime'] == position['positionMonoTime']
      lead = row['radar']['leadOne']
      if row['frame'] < 30:
        # The late AUX completes one real pair. It cannot reuse the support
        # from before the invalid scan or bypass native reconfirmation.
        assert not quality['qualified'] and not lead['present'] and not objects(row)
        assert row['solver'] == 0
        continue
      assert quality['qualified'] and quality['sampleCount'] >= 5
      assert quality['positionMonoTime'] - quality['firstPositionMonoTime'] >= 250_000_000
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


@pytest.mark.parametrize('outcome', ['still_present', 'absent', 'departure', 'new_identity'])
def test_current_radar_and_model_determine_hold_or_release_without_old_identity(outcome, record_property):
  """A current obstacle stops; its old identity does not veto a new clear scene."""
  rows = []
  with control_pipeline(speed=0.0) as pipeline:
    for frame in range(100):
      elapsed = max(0.0, (frame - 28) * .05)
      departing = outcome == 'departure' and frame >= 28
      target = Target(3.0 + .4 * elapsed**2 if departing else 3.0, .8 * elapsed if departing else 0.0)
      targets = [target]
      if frame >= 28 and outcome == 'absent':
        # Keep real radar traffic alive with an out-of-path observation. An
        # empty raw CAN stream tests radarSilence, not healthy target absence.
        targets = [Target(20.0, 0.0, lateral=7.0, slot=1, oem_id=2)]
      elif frame >= 28 and outcome == 'new_identity':
        targets = [Target(12.0 + 3.0 * elapsed, 3.0, oem_id=2)]
      rows.append(pipeline.step(speed=0.0, radar=targets))
      if frame == 27:
        assert_current_radar_hold(pipeline)
  record(record_property, f'hold_{outcome}', rows)
  held = rows[27]['radar']['leadOne']
  assert rows[27]['radar']['fusionHealthy'] and held['present']
  assert held['evidence']['source'] == 'radar' and held['evidence']['stopQualified']
  assert held['evidence']['trackId'] > 0
  assert all(owner['phase'] == 'hold' and owner['stop'] for row in rows[20:28] for owner in row['owner'])
  assert any(sample['stop'] and sample['brake'] for row in rows[20:28] for sample in row['can'])
  if outcome == 'absent':
    # Healthy observer publications continue. Its raw asynchronous target
    # expires before this tail; an empty event list alone is not a new scan.
    assert all(row['observer_ns'] > rows[27]['observer_ns'] for row in rows[28:])
    assert all(not obj['evidence']['coasting'] and obj['evidence']['source'] != 'predicted'
               for row in rows[28:] for obj in objects(row))
    assert all(not objects(row) for row in rows[36:])
    assert all(row['observer']['reason'] == 'observing' for row in rows[36:])
    assert all(row['radar']['fusionHealthy'] and not row['radar']['fusionDegraded'] for row in rows[36:])
    assert all(not row['reasons'] & int(StopReason.CLOSE_LEAD) for row in rows[36:])
  elif outcome == 'still_present':
    for row in rows[36:]:
      lead = row['radar']['leadOne']
      assert lead['present'] and lead['evidence']['stopQualified']
      assert lead['evidence']['positionMonoTime'] == row['source_ns']
      assert not row['release']
      assert all(owner['phase'] == 'hold' and owner['stop'] for owner in row['owner'])
      assert all(sample['brake'] and not sample['release'] and sample['accel'] <= 0.0 for sample in row['can'])
  elif outcome == 'departure':
    identity = rows[27]['radar']['leadOne']['evidence']['trackId']
    assert all(row['radar']['leadOne']['evidence']['trackId'] == identity for row in rows[28:])
  else:
    assert all(row['radar']['leadOne']['present'] for row in rows[36:])
    assert all(row['radar']['leadOne']['evidence']['trackId'] != held['evidence']['trackId'] for row in rows[36:])
  assert not any(row['radar']['fusionRiskUnresolved'] for row in rows)
  if outcome != 'still_present':
    assert any(row['release'] and not row['stop'] for row in rows[28:])
    assert any(sample['release'] and not sample['stop'] and sample['accel'] > 0.0
               for row in rows[28:] for sample in row['can'])


@pytest.mark.parametrize('current_risk', [False, True], ids=['current-forward-model', 'new-current-stationary-target'])
def test_hold_outer_qualification_withdrawal_uses_only_current_stop_and_departure_inputs(current_risk, record_property):
  """Real CAN qualifications enter HOLD; later raw sensing decides its exit."""
  with control_pipeline(speed=0.0) as pipeline:
    rows = [pipeline.step(speed=0.0, radar=[Target(3.0, 0.0, lateral=1.0)]) for _ in range(28)]
    former_identity = assert_current_radar_hold(pipeline)
    held_position_ns = rows[-1]['radar']['leadOne']['evidence']['positionMonoTime']
    later = []
    for frame in range(1, 61):
      # The same measured object moves from the inner band to a stable outer
      # point. No width/vision observation supplies current body occupancy.
      targets = [Target(3.0, 0.0, lateral=min(1.6, 1.0 + .02 * frame))]
      if current_risk and frame >= 5:
        # Give this independent current obstacle time to qualify before the
        # first target leaves the inner band; no historical stop is needed.
        targets.append(Target(2.0, 0.0, slot=1, oem_id=2))
      row = pipeline.step(speed=0.0, radar=targets, model_stop=False, model_accel=.6)
      track, = [track for track in pipeline.state[SERVICE].fusion.tracks if track.trackId == former_identity[1]]
      row['former_track'] = {'track_id': int(track.trackId), 'position_ns': int(track.positionMonoTime),
                             'lateral': float(track.yRel), 'active': bool(track.active),
                             'coasting': bool(track.coasting), 'body_valid': bool(track.bodyValid)}
      later.append(row)
    rows.extend(later)
    tail = later[-12:]
    for row in tail:
      track = row['former_track']
      assert track['active'] and not track['coasting'] and not track['body_valid']
      assert track['position_ns'] == row['source_ns'] > held_position_ns
      assert SWEPT_HALF_WIDTH < abs(track['lateral']) < 2.0
      assert row['radar']['fusionSessionId'] == former_identity[0]
      assert row['radar']['fusionHealthy'] and not row['radar']['fusionDegraded']
      assert not row['radar']['fusionClearedTracks'] and not row['radar']['fusionRiskUnresolved']
      assert all((lead['evidence']['sessionId'], lead['evidence']['trackId']) != former_identity for lead in objects(row))
      assert not row['reasons'] & int(StopReason.E2E)
    motion = pipeline.state['controlsState'].motionState
    assert not motion.degraded and not motion.waitingForFreshInput
    assert former_identity[1] not in pipeline.radard.fusion_control._tracks
    if current_risk:
      for row in later:
        # Check the complete handoff, not only its final HOLD: the old inner
        # target supplies current stop evidence until the successor qualifies.
        assert any(lead['evidence']['stopQualified'] and lead['evidence']['brakeQualified']
                   and lead['evidence']['positionMonoTime'] == row['source_ns'] and lead['dRel'] < 3.1
                   for lead in objects(row))
        assert not row['release']
        assert all(sample['brake'] and not sample['release'] and sample['accel'] <= 0.0 for sample in row['can'])
      for row in tail:
        current, = objects(row)
        claim = current['evidence']
        assert claim['brakeQualified'] and claim['stopQualified'] and not claim['departureQualified']
        assert claim['positionMonoTime'] == row['source_ns'] and current['dRel'] < 2.1
        assert not row['release']
        assert all(owner['phase'] == 'hold' and owner['stop'] for owner in row['owner'])
        assert all(sample['brake'] and not sample['release'] and sample['accel'] <= 0.0 for sample in row['can'])
    else:
      assert all(not objects(row) and not row['reasons'] & int(StopReason.CLOSE_LEAD) for row in tail)
      assert any(row['release'] and not row['stop'] for row in later)
      assert any(sample['active'] and sample['release'] and sample['accel'] > 0.0
                 for row in later for sample in row['can'])
      assert all(owner['phase'] != 'hold' and not owner['stop'] for owner in tail[-1]['owner'])
    record(record_property, f'hold_outer_qualification_current_risk_{current_risk}', rows)


@pytest.mark.parametrize('scope_exit', ['experimental_disabled', 'dec_disabled', 'scope_stale'])
def test_scope_exit_keeps_current_model_stop_and_allows_fresh_release_without_reengagement(scope_exit, record_property):
  """Scope health is explicit; a past target creates no permanent stop duty."""
  rows = []
  with control_pipeline(speed=0.0) as pipeline:
    for _ in range(28):
      rows.append(pipeline.step(speed=0.0, radar=[Target(3.0, 0.0)]))
    assert all(owner['phase'] == 'hold' and owner['stop'] for row in rows[20:] for owner in row['owner'])
    assert_current_radar_hold(pipeline)
    if scope_exit == 'experimental_disabled':
      pipeline.params.put_bool('ExperimentalMode', False, block=True)
    elif scope_exit == 'dec_disabled':
      pipeline.params.put_bool('DynamicExperimentalControl', False, block=True)
    pipeline.settings = ObserverSettings.read(pipeline.params)
    revoked = []
    side_target = Target(20.0, 0.0, lateral=7.0, slot=1, oem_id=2)
    for frame in range(48):
      # Stop selfdriveState updates for 1.2 seconds, then restore them. The
      # source timestamp remains old; the fusion gate itself is not mocked.
      online = scope_exit != 'scope_stale' or frame >= 24
      # A new model stop is a current independent reason to keep braking.
      # No old radar ID is injected or needed for this negative control.
      row = pipeline.step(speed=0.0, radar=[side_target], scope_online=online, model_stop=True, model_accel=-1.0)
      rows.append(row)
      revoked.append(row)
      assert not row['release']
      assert all(owner['phase'] == 'hold' and owner['stop'] for owner in row['owner'])
      assert all(sample['brake'] and sample['stop'] and not sample['release'] and sample['accel'] <= 0.0
                 for sample in row['can'])
    assert any(not row['radar'].get('fusionActive', False) for row in revoked)
    if scope_exit == 'scope_stale':
      assert any(row['radar'].get('fusionActive', False) for row in revoked[30:])

    # Fresh current forward output replaces the stop. No driver cancellation
    # or old target departure/OUT receipt is required to clear a former ID.
    pipeline.params.put_bool('ExperimentalMode', True, block=True)
    pipeline.params.put_bool('DynamicExperimentalControl', True, block=True)
    pipeline.settings = ObserverSettings.read(pipeline.params)
    resumed = [pipeline.step(speed=0.0, radar=[side_target], visual=[Target(40.0 + .3 * frame, 6.0)])
               for frame in range(48)]
    rows.extend(resumed)
    assert all(row['long_active'] for row in resumed)
    assert any(row['release'] and not row['stop'] for row in resumed)
    assert any(sample['active'] and sample['release'] and sample['accel'] > 0.0
               for row in resumed for sample in row['can'])
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
def test_feedback_gap_preserves_current_stop_and_recovery_uses_current_targets(feedback_fault, record_property):
  """Input health stays observable while new model/target evidence owns motion."""
  rows = []
  with control_pipeline(speed=0.0) as pipeline:
    for _ in range(28):
      rows.append(pipeline.step(speed=0.0, radar=[Target(3.0, 0.0)]))
    former_identity = assert_current_radar_hold(pipeline)
    assert all(owner['phase'] == 'hold' and owner['stop'] for owner in rows[-1]['owner'])
    for _ in range(10):
      rows.append(pipeline.step(speed=0.0, observer_online=False, model_stop=True, model_accel=-1.0))
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
      # A current independent model stop still vetoes the fresh farther visual
      # target. This stop does not depend on the missing prior radar identity.
      row = pipeline.step(speed=0.0, visual=[Target(40.0, 6.0)], observer_online=False, model_stop=True, model_accel=-1.0)
      rows.append(row)
      assert pipeline.radard.fusion_control is adapter
      assert not row['radar']['fusionActive'] and row['radar']['fusionReason'] == 'controlScopeUnavailable'
      assert row['radar']['fusionDegraded'] and not row['radar']['fusionRiskUnresolved']
      assert row['radar'].get('radarObjects', []) == []
      assert row['radar']['leadOne'] == row['radar']['visionLeadOne'] and row['radar']['leadOne']['present']
      assert not row['release']
      assert row['reasons'] & int(StopReason.E2E)
      assert not row['reasons'] & int(StopReason.CLOSE_LEAD)
      assert all(owner['phase'] == 'hold' and owner['stop'] for owner in row['owner'])
      assert all(sample['brake'] and not sample['release'] and sample['accel'] <= 0.0 for sample in row['can'])
    # Scope recovery leaves the observer stale. The separate current model
    # stop remains; recovering a scope receipt alone is not a release input.
    row = pipeline.step(speed=0.0, visual=[Target(40.0, 6.0)], observer_online=False, model_stop=True, model_accel=-1.0)
    rows.append(row)
    assert row['radar']['fusionActive'] and row['radar']['fusionDegraded']
    assert not row['radar']['fusionRiskUnresolved'] and not row['release']
    # Fresh sensing supplies a different current close stationary target.
    # It must still prevent release after its own qualification completes.
    for _ in range(12):
      row = pipeline.step(speed=0.0, radar=[Target(3.0, 0.0, oem_id=2)])
      rows.append(row)
    assert row['radar']['fusionHealthy'] and not row['radar']['fusionDegraded']
    assert not row['radar']['fusionRiskUnresolved']
    assert not pipeline.planner.reference_planner.radar_consumption.degraded
    lead = row['radar']['leadOne']
    assert lead['present'] and lead['evidence']['brakeQualified'] and lead['evidence']['stopQualified']
    assert (lead['evidence']['sessionId'], lead['evidence']['trackId']) != former_identity
    assert not lead['evidence']['coasting'] and not row['release']
    assert all(owner['phase'] == 'hold' and owner['stop'] for owner in row['owner'])
    assert all(sample['brake'] and not sample['release'] and sample['accel'] <= 0.0 for sample in row['can'])
    # A newly qualified moving target may release with current forward model
    # input, without requiring a receipt from either historical stopped ID.
    moving = [pipeline.step(speed=0.0, radar=[Target(12.0 + .15 * frame, 3.0, oem_id=3)])
              for frame in range(48)]
    rows.extend(moving)
    assert all(row['radar']['fusionHealthy'] and not row['radar']['fusionDegraded'] for row in moving[8:])
    assert all(row['long_active'] for row in moving)
    assert any(row['release'] and not row['stop'] for row in moving)
    assert any(sample['active'] and sample['release'] and sample['accel'] > 0.0
               for row in moving for sample in row['can'])
  record(record_property, f'scope_feedback_{feedback_fault}', rows)
