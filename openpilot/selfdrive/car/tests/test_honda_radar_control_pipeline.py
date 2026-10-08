"""CAN-origin regressions for active radar fusion and its execution boundary."""
import json

import pytest

from openpilot.cereal import messaging
from openpilot.selfdrive.car.honda_radar_observer_config import CONTROL, SERVICE, SHADOW, SCOPE_MAX_AGE_NS, ObserverSettings
from openpilot.selfdrive.car.tests import honda_radar_control_test_support as pipeline_support
from openpilot.selfdrive.car.tests.honda_radar_control_test_support import Target, control_pipeline
from openpilot.selfdrive.controls.lib.drive_helpers import STOP_SPEED
from openpilot.selfdrive.controls.lib.motion_supervisor import StopReason


def objects(row):
  return row['radar'].get('radarObjects', [])


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
  assert memory.path.confirmed and 1.1 < abs(memory.path.offset) < 2.0
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
      assert identity not in adapter._lost_tracks
      assert not any(lead['evidence']['trackId'] == identity[1] and lead['evidence']['brakeQualified'] for lead in objects(rows[-1]))
    else:
      replacement = Target(target.distance, target.speed, lateral=target.lateral, slot=target.slot, oem_id=17)
      rows.extend(pipeline.step(speed=8.0, radar=[replacement]) for _ in range(20))
      tracks = [track for track in pipeline.state[SERVICE].fusion.tracks if track.active and track.oemTrackId == 17]
      track, = tracks
      assert track.trackId != identity[1]
      memory = pipeline.radard.fusion_control._tracks[int(track.trackId)]
      assert not memory.path.confirmed
      for lead in objects(rows[-1]):
        if lead['evidence']['trackId'] == track.trackId:
          assert not lead['evidence']['followQualified'] and not lead['evidence']['stopQualified']
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
  for mode in (SHADOW, CONTROL):
    with control_pipeline(mode=mode) as pipeline:
      rows = []
      for frame in range(36):
        distance = 36.0 + (lead_speed - 10.0) * frame * .05
        rows.append(pipeline.step(speed=10.0, radar=[Target(distance, lead_speed)]))
      arms[mode] = rows
      record(record_property, f'radar_only_mode_{mode}', rows)
  baseline, candidate = arms[SHADOW], arms[CONTROL]
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
def test_saturation_reacquisition_clears_only_the_proven_identity_loss(recovery, monkeypatch, record_property):
  """Fresh CAN recovery clears only its identity's diagnostic loss history."""
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
      rows.append(row)
  record(record_property, f'saturation_reacquisition_{recovery}', rows)

  initial = rows[23]['radar']['leadOne']
  identity = initial['evidence']['sessionId'], initial['evidence']['trackId']
  assert initial['present'] and initial['evidence']['source'] == 'radar'
  assert initial['evidence']['followQualified'] and initial['evidence']['brakeQualified']
  assert not rows[23]['radar']['fusionRiskUnresolved']
  assert rows[23]['fusion_tracks'][0]['lifecycleRaw'] == 4093
  invalid = rows[24]
  assert invalid['fusion_tracks'][0]['trackId'] == identity[1]
  assert not invalid['fusion_tracks'][0]['active']
  assert invalid['fusion_tracks'][0]['lastInvalidReason'] == 'invalidGeometry'
  assert all(row['radar']['fusionRiskUnresolved'] for row in rows[24:26])
  position = next(sample for sample in rows[25]['raw_observations'] if sample['positionValid'])
  assert position['raw']['lifecycleRaw'] == 4094 and not position['velocityValid']
  unresolved = rows[25:26] if recovery == 'same_identity' else rows[25:]
  for row in unresolved:
    # The old identity has no fresh following/departure authority. Its loss
    # history does not revoke the independent healthy model's permissions.
    assert row['radar']['fusionRiskUnresolved']
    for track in row['fusion_tracks']:
      if track['trackId'] == identity[1]:
        assert not track['active'] and not track['velocityGood']
    for obj in objects(row):
      evidence = obj['evidence']
      if (evidence['sessionId'], evidence['trackId']) == identity:
        # A retained risk may still brake by prediction. Another OEM must
        # neither refresh its measurements nor grant it follow/departure.
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
      assert not row['radar']['fusionRiskUnresolved']
      lead = row['radar']['leadOne']
      assert lead['present'] and lead['evidence']['source'] == 'radar'
      assert (lead['evidence']['sessionId'], lead['evidence']['trackId']) == identity
      assert lead['evidence']['followQualified'] and lead['evidence']['brakeQualified']
      assert not lead['evidence']['coasting'] and row['solver'] == 0
    assert rows[-1]['target'] < -.1
    assert any(sample['active'] and sample['brake'] and sample['accel'] < -.1
               for row in rows[-6:] for sample in row['can'])
  else:
    # Neither expiry nor a fully qualified replacement can erase the old
    # physical identity's risk. This extends beyond fusion's 500 ms history.
    assert all(row['radar']['fusionRiskUnresolved'] for row in rows[26:])
    if recovery == 'different_oem':
      replacement = rows[-1]['radar']['leadOne']
      assert replacement['present'] and replacement['evidence']['brakeQualified']
      assert (replacement['evidence']['sessionId'], replacement['evidence']['trackId']) != identity


def test_near_visual_cut_in_is_not_diluted_by_a_different_far_radar_car(record_property):
  arms = {}
  for cut_in in (False, True):
    with control_pipeline() as pipeline:
      rows = []
      for frame in range(36):
        visual = [Target(50.0, 10.0)]
        if cut_in and frame >= 16:
          visual = [Target(16.0 - 8.0 * (frame - 16) * .05, 2.0), *visual]
        rows.append(pipeline.step(speed=10.0, radar=[Target(50.0, 10.0)], visual=visual))
      arms[cut_in] = rows
      record(record_property, f'visual_cut_in_{cut_in}', rows)
  baseline, candidate = arms[False], arms[True]
  for row in candidate[16:]:
    near = row['radar']['leadOne']
    expected = 16.0 - 8.0 * (row['frame'] - 16) * .05
    assert near['evidence']['source'] == 'vision' and not near['radar']
    assert abs(near['dRel'] - expected) < .001
    assert any(obj['evidence']['source'] in ('radar', 'fused') and obj['dRel'] > 45.0 for obj in objects(row))
  assert candidate[-1]['target'] < baseline[-1]['target'] - .5
  assert any(sample['accel'] < -.5 and sample['brake'] for row in candidate[-6:] for sample in row['can'])


def test_third_entering_object_brakes_while_two_distant_visual_leads_remain(record_property):
  arms = {}
  for crossing in (False, True):
    with control_pipeline(speed=8.0) as pipeline:
      rows = []
      for frame in range(24):
        timestamp = frame * .05
        radar = [Target(27.0 - 8.0 * timestamp, 0.0, 4.5 - 1.2 * timestamp)] if crossing else []
        rows.append(pipeline.step(speed=8.0, radar=radar, visual=[Target(35.0, 8.0), Target(50.0, 8.0)]))
      arms[crossing] = rows
      record(record_property, f'third_crossing_{crossing}', rows)
  baseline, candidate = arms[False], arms[True]
  entering = [row for row in candidate if any(obj['evidence']['pathState'] == 'entering' for obj in objects(row))]
  assert entering
  for row in entering:
    assert row['radar']['leadOne']['present'] and row['radar']['leadTwo']['present']
    assert row['radar']['leadOne']['dRel'] >= 34.9 and row['radar']['leadTwo']['dRel'] >= 49.9
    assert len(objects(row)) == 3
    risk = next(obj for obj in objects(row) if obj['evidence']['pathState'] == 'entering')
    assert risk['evidence']['brakeQualified'] and not risk['evidence']['departureQualified']
    assert 0.0 < risk['evidence']['conflictStart'] < risk['evidence']['conflictEnd'] <= 10.0
    assert abs(risk['yRel']) > 2.0  # The response precedes actual lane occupancy.
  assert candidate[-1]['target'] < baseline[-1]['target'] - .2
  assert any(sample['brake'] and sample['accel'] < -.1 for row in entering for sample in row['can'])


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
    assert any(row['radar'].get('fusionDegraded', False) for row in rows[36:])
    assert all(row['radar']['fusionRiskUnresolved'] for row in rows[36:])
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


@pytest.mark.parametrize('scope_exit', ['mode_to_shadow', 'dec_disabled', 'scope_stale'])
def test_scope_exit_cannot_release_radar_hold_before_driver_starts_a_new_engagement(scope_exit, record_property):
  """Configuration/transport loss cannot release an explicitly established hold."""
  rows = []
  with control_pipeline(speed=0.0) as pipeline:
    for _ in range(28):
      rows.append(pipeline.step(speed=0.0, radar=[Target(3.0, 0.0)]))
    assert all(owner['phase'] == 'hold' and owner['stop'] for row in rows[20:] for owner in row['owner'])
    held_identity = establish_prior_object_hold(pipeline)
    if scope_exit == 'mode_to_shadow':
      pipeline.params.put('HondaBoschARadarMode', SHADOW, block=True)
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
    pipeline.params.put('HondaBoschARadarMode', CONTROL, block=True)
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
  ('experimental_off_bad_sp', False, False), ('dec_off', False, False), ('mode_shadow', False, False),
])
def test_scope_feedback_failure_is_distinct_from_an_explicit_exit(scope, enabled, unavailable):
  with control_pipeline() as pipeline:
    state = pipeline.state
    now = pipeline_support.START_NS
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
    if scope == 'mode_shadow':
      pipeline.params.put('HondaBoschARadarMode', SHADOW, block=True)
    assert pipeline.radard.control_enabled(state, now) is enabled
    assert pipeline.radard.control_scope_unavailable is unavailable


@pytest.mark.parametrize('feedback_fault', ['invalid', 'invalid_dec_off', 'stale', 'future', 'unseen'])
def test_feedback_gap_preserves_real_radar_hold_and_perception_loss_history(feedback_fault, record_property):
  """Invalid SP cannot remove the explicitly established object-hold premise."""
  rows = []
  with control_pipeline(speed=0.0) as pipeline:
    for _ in range(28):
      rows.append(pipeline.step(speed=0.0, radar=[Target(3.0, 0.0)]))
    held = establish_prior_object_hold(pipeline)
    assert all(owner['phase'] == 'hold' and owner['stop'] for owner in rows[-1]['owner'])
    for _ in range(10):
      rows.append(pipeline.step(speed=0.0, observer_online=False))
    adapter = pipeline.radard.fusion_control
    retained = set(adapter._lost_tracks)
    assert retained and held in retained and rows[-1]['radar']['fusionRiskUnresolved']
    for _ in range(12):
      state = pipeline.state
      if feedback_fault in ('invalid', 'invalid_dec_off'):
        state.valid['longitudinalPlanSP'] = False
      if feedback_fault == 'invalid_dec_off':
        feedback = state['longitudinalPlanSP'].as_builder()
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
      assert pipeline.radard.fusion_control is adapter and set(adapter._lost_tracks) == retained
      assert not row['radar']['fusionActive'] and row['radar']['fusionReason'] == 'controlScopeUnavailable'
      assert row['radar']['fusionRiskUnresolved'] and row['radar'].get('radarObjects', []) == []
      assert row['radar']['leadOne'] == row['radar']['visionLeadOne'] and row['radar']['leadOne']['present']
      assert pipeline.planner.reference_planner.radar_held_identity == held
      assert not row['release']
      assert all(owner['phase'] == 'hold' and owner['stop'] for owner in row['owner'])
      assert all(sample['brake'] and not sample['release'] and sample['accel'] <= 0.0 for sample in row['can'])
    # A successful new plan alone is not a fresh measurement of the lost
    # obstacle. Normal sensing recovery must still resolve its identity.
    row = pipeline.step(speed=0.0, visual=[Target(40.0, 6.0)], observer_online=False)
    rows.append(row)
    assert row['radar']['fusionActive'] and row['radar']['fusionRiskUnresolved']
    assert set(adapter._lost_tracks) == retained and not row['release']
    # A fresh explicit mode change is a different lifecycle boundary.
    pipeline.params.put('HondaBoschARadarMode', SHADOW, block=True)
    row = pipeline.step(speed=0.0, visual=[Target(40.0, 6.0)], observer_online=False)
    rows.append(row)
    assert not row['radar']['fusionActive'] and not row['radar']['fusionRiskUnresolved']
    assert not adapter._lost_tracks
  record(record_property, f'scope_feedback_{feedback_fault}', rows)
