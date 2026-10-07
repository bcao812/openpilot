"""CAN-origin regressions for active radar fusion and its execution boundary."""
import json

import pytest

from openpilot.selfdrive.car.honda_radar_observer_config import CONTROL, SHADOW, ObserverSettings
from openpilot.selfdrive.car.tests.honda_radar_control_test_support import Target, control_pipeline
from openpilot.selfdrive.controls.lib.drive_helpers import STOP_SPEED
from openpilot.selfdrive.controls.lib.motion_supervisor import StopReason


def objects(row):
  return row['radar'].get('radarObjects', [])


def record(record_property, name, rows):
  record_property(name, json.dumps(rows, separators=(',', ':'), allow_nan=False))


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
def test_radar_stop_holds_on_dropout_and_releases_for_same_car_departure(outcome, record_property):
  rows = []
  with control_pipeline(speed=0.0) as pipeline:
    for frame in range(100):
      elapsed = max(0.0, (frame - 28) * .05)
      departing = outcome == 'departure' and frame >= 28
      target = Target(3.0 + .4 * elapsed**2 if departing else 3.0, .8 * elapsed if departing else 0.0)
      rows.append(pipeline.step(speed=0.0, radar=[target], observer_online=outcome != 'dropout' or frame < 28))
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
  """Configuration/transport loss cannot turn the stopped obstacle into clear road."""
  rows = []
  with control_pipeline(speed=0.0) as pipeline:
    for _ in range(28):
      rows.append(pipeline.step(speed=0.0, radar=[Target(3.0, 0.0)]))
    assert all(owner['phase'] == 'hold' and owner['stop'] for row in rows[20:] for owner in row['owner'])
    held_identity = pipeline.planner.reference_planner.radar_held_identity
    assert held_identity is not None
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
