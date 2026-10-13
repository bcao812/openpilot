"""Known-motion CAN inputs through the real Accord sensing and control chain.

These are synthetic truth checks, not evidence that an unlabelled route object
was false or that position-only braking has been validated on the vehicle.
Only the CAN observations and model geometry are supplied by the fixture; no
producer quality, formal lead, MPC decision or Honda command is patched.
"""
from contextlib import contextmanager
from dataclasses import replace
import json
import math
from unittest.mock import patch

import numpy as np
import pytest

from openpilot.selfdrive.car.honda_radar_observer_config import SERVICE
from openpilot.selfdrive.car.tests import honda_radar_control_test_support as pipeline_support
from openpilot.selfdrive.car.tests.honda_radar_control_test_support import Target, control_pipeline
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS
from openpilot.selfdrive.modeld.constants import ModelConstants


STEP = .05
CLOSING_CASES = {
  'moving_slow_car': (14.0, 4.0, 30.0),
  'near_stationary_object': (10.0, 0.0, 22.0),
  'fast_closing_car': (25.0, 5.0, 80.0),
}


def formal_objects(row):
  assert row['radar'].get('radarObjects', []) == []
  return [row['radar'][name] for name in ('leadOne', 'leadTwo') if row['radar'][name].get('present')]


def record(record_property, name, rows):
  record_property(name, json.dumps(rows, separators=(',', ':'), allow_nan=False))


def capture(pipeline, row, *, truth_speed, truth_distance, predict_gap=False):
  observation = pipeline.state[SERVICE]
  row['truth_speed'], row['truth_distance'] = truth_speed, truth_distance
  row['fusion_tracks'] = [track.to_dict() for track in observation.fusion.tracks]
  row['raw_observations'] = [sample.to_dict() for sample in observation.observations]
  row['consumer_degraded'] = pipeline.planner.reference_planner.radar_consumption.degraded
  if predict_gap:
    # Analyse the already-solved ego plan against the known straight, constant-
    # speed physical target, including frames where sensing did not admit it.
    # This never enters the producer and is not an executed/closed-loop gap.
    mpc = pipeline.planner.mpc
    assert mpc._diagnostic_reference_ns == row['source_ns']
    target_trajectory = np.column_stack((truth_distance + truth_speed * T_IDXS, np.full(len(T_IDXS), truth_speed)))
    row['planned_truth_minimum_gap'] = mpc.lead_clearance_minimum(target_trajectory, mpc.x_sol, mpc.u_sol[:, 0])
    row['truth_prediction'] = {'source_ns': row['source_ns'], 'times': T_IDXS.tolist(),
                               'ego_states': mpc.x_sol.tolist(), 'ego_jerks': mpc.u_sol[:, 0].tolist(),
                               'lead_positions': target_trajectory[:, 0].tolist(),
                               'meaning': 'Predicted clearance of this MPC plan; no executed-motion feedback'}
  return row


@contextmanager
def sensor_input(pipeline, *, native_velocity=False, freeze_until=0, range_error=None):
  """Change only the actual sensor packets, before decoder and producer run."""
  original = pipeline_support.scan_frames
  original_raw = pipeline_support.raw_scan
  frozen = None

  def scan(**fields):
    nonlocal frozen
    if not native_velocity:
      fields['velocity'] = 0  # Explicit invalid native AUX, not a zero relative velocity.
    if range_error is not None:
      fields['distance'] += round(range_error(pipeline.frame) / .05712)
    frames = original(**fields)
    if pipeline.frame < freeze_until:
      if frozen is None:
        frozen = frames
      return frozen
    return frames

  def raw_scan(target, ego_speed, frame, bus):
    # The original helper validates native-velocity encoding even if its AUX
    # will be invalid. An invalid field carries no velocity measurement, so
    # encode a neutral placeholder before scan() marks it invalid. Positions,
    # scan identity and all externally recorded motion truth are unchanged.
    encoded = target if native_velocity else replace(target, speed=ego_speed)
    return original_raw(encoded, ego_speed, frame, bus)

  with patch.object(pipeline_support, 'scan_frames', new=scan), patch.object(pipeline_support, 'raw_scan', new=raw_scan):
    yield


def run_closing_truth(case, *, native_velocity=False, visual=False, frames=28):
  """Also callable against the previous commit for an identical-input baseline.

  Receipt capture deliberately does not require the added schema fields; the
  tests below, rather than the input generator, check the new quality contract.
  """
  ego_speed, lead_speed, initial_distance = CLOSING_CASES[case]
  rows = []
  with control_pipeline(speed=ego_speed, cruise=ego_speed + 4.0) as pipeline:
    with sensor_input(pipeline, native_velocity=native_velocity):
      for frame in range(frames):
        distance = initial_distance + (lead_speed - ego_speed) * frame * STEP
        target = Target(distance, lead_speed, oem_id=25)
        row = pipeline.step(speed=ego_speed, radar=[target], visual=[target] if visual else [])
        rows.append(capture(pipeline, row, truth_speed=lead_speed, truth_distance=distance, predict_gap=True))
  return rows


def timing_and_peak(rows):
  first_lead = next((row['time'] for row in rows if formal_objects(row)), None)
  # The support fixture records all 100 Hz owner ticks and 50 Hz CAN commands
  # in each 50 ms row. Its row end is a conservative first-command time bound.
  first_brake = next((row['time'] + STEP for row in rows
                      if any(command['active'] and command['brake'] and command['accel'] < -.1
                             for command in row['can'])), None)

  def minimum_predicted_gap(start):
    return min((row['planned_truth_minimum_gap'] for row in rows
                if start is not None and row['time'] >= start and 'planned_truth_minimum_gap' in row), default=None)

  return {'first_lead': first_lead, 'first_brake_upper_bound': first_brake,
          'minimum_plan_accel': min(row['target'] for row in rows),
          'minimum_can_accel': min(command['accel'] for row in rows for command in row['can']),
          'planned_truth_minimum_gap': minimum_predicted_gap(0.0),
          'planned_truth_minimum_gap_after_first_lead': minimum_predicted_gap(first_lead),
          'planned_truth_minimum_gap_from_half_second': minimum_predicted_gap(.5)}


def current_quality(track):
  quality = track['positionVelocity']
  assert quality['version'] == 1
  assert quality['positionSeq'] == track['positionSeq']
  assert quality['positionMonoTime'] == track['positionMonoTime']
  assert 0 <= quality['sampleCount'] <= track['observations']
  if quality['qualified']:
    assert track['active'] and not track['coasting']
    assert quality['sampleCount'] >= 6 and quality['span'] > 0.0
    assert all(math.isfinite(quality[name]) for name in
               ('shortSpeed', 'longSpeed', 'speedStd', 'windowDifference', 'innovationBias', 'posteriorCorrection'))
  return quality


def assert_formal_position_quality(row):
  """No unqualified position-derived component may become a present lead."""
  if not row['radar']['fusionActive']:
    return  # The first DEC bootstrap frame still publishes the legacy vision path.
  tracks = {track['trackId']: track for track in row['fusion_tracks']}
  for lead in formal_objects(row):
    claim = lead['evidence']
    assert claim['positionValid'] and claim['velocityValid'] and not claim['coasting']
    assert claim['brakeQualified']
    if 'positionVelocity' in claim['reason']:
      track = tracks[claim['trackId']]
      assert current_quality(track)['qualified']
      assert claim['positionMonoTime'] == claim['velocityMonoTime'] == track['positionMonoTime']


@pytest.mark.parametrize('case', CLOSING_CASES)
def test_real_closing_object_brakes_from_positions_within_half_a_second(case, record_property):
  ego_speed, lead_speed, _ = CLOSING_CASES[case]
  native_encodable = abs(lead_speed - ego_speed) < 13.5
  reference_name = 'native' if native_encodable else 'current_vision'
  reference = run_closing_truth(case, native_velocity=native_encodable, visual=not native_encodable)
  rows = run_closing_truth(case)
  record(record_property, f'{case}_{reference_name}', reference)
  record(record_property, f'{case}_position_only', rows)
  record_property(f'{case}_timing_and_peak', json.dumps({
    reference_name: timing_and_peak(reference), 'position_only': timing_and_peak(rows),
  }, allow_nan=False))

  admitted = [row for row in rows if formal_objects(row)]
  summary = timing_and_peak(rows)
  assert admitted and summary['first_lead'] <= .5
  assert summary['first_brake_upper_bound'] is not None and summary['first_brake_upper_bound'] <= .5 + 1e-9
  assert timing_and_peak(reference)['first_brake_upper_bound'] is not None
  for row in rows:
    assert row['solver'] == 0 and not row['radar']['fusionRiskUnresolved']
    assert_formal_position_quality(row)
    for track in row['fusion_tracks']:
      assert not track['velocityGood'] and track['velocityMonoTime'] == 0
      current_quality(track)
    assert all(not sample['velocityValid'] for sample in row['raw_observations'])
    for lead in formal_objects(row):
      claim = lead['evidence']
      assert lead['radar'] and lead['modelProb'] == 0.0 and claim['source'] == 'radar'
      assert 'positionVelocity' in claim['reason'] and claim['followQualified'] and claim['stopQualified']
      assert row['radar']['fusionHealthy'] and not row['consumer_degraded']
  for row in rows[-5:]:
    lead, = formal_objects(row)
    assert abs(lead['vLeadK'] - row['truth_speed']) < .6
    track, = row['fusion_tracks']
    quality = current_quality(track)
    assert quality['qualified']
    assert abs(quality['longSpeed'] - row['truth_speed']) < .6
  assert any(command['active'] and command['brake'] and command['accel'] < -.1 and command['gas'] == -30000
             for row in admitted for command in row['can'])


def world_motion(frame, *, acceleration, yaw):
  """World-fixed straight target and independently integrated ego motion."""
  elapsed, initial_speed, lead_speed = frame * STEP, 10.0, 8.0
  ego_speed, heading = initial_speed + acceleration * elapsed, yaw * elapsed
  if yaw:
    assert acceleration == 0.0
    ego_x = initial_speed * math.sin(heading) / yaw
    ego_y = initial_speed * (1.0 - math.cos(heading)) / yaw
  else:
    ego_x, ego_y = initial_speed * elapsed + .5 * acceleration * elapsed**2, 0.0
  relative_x, relative_y = 40.0 + lead_speed * elapsed - ego_x, -ego_y
  c, s = math.cos(heading), math.sin(heading)
  distance, lateral = c * relative_x + s * relative_y, -s * relative_x + c * relative_y
  return ego_speed, Target(distance, lead_speed * c, lateral=lateral, oem_id=25)


@contextmanager
def model_turn(pipeline, yaw):
  original = pipeline._model

  def model(timestamp, speed, visual, acceleration, should_stop):
    event = original(timestamp, speed, visual, acceleration, should_stop)
    if yaw:
      assert acceleration == 0.0
      model = event.modelV2
      model.orientationRate.z = [-yaw] * len(ModelConstants.T_IDXS)
      model.position.x = [float(speed * math.sin(yaw * t) / yaw) for t in ModelConstants.T_IDXS]
      model.position.y = [float(-speed * (1.0 - math.cos(yaw * t)) / yaw) for t in ModelConstants.T_IDXS]
      radius = abs(speed / yaw)
      for lane in model.laneLines:
        offset = float(lane.y[0])
        lane.y = [offset - math.copysign(radius - math.sqrt(radius**2 - x**2), yaw) for x in lane.x]
    return event

  with patch.object(pipeline, '_model', new=model):
    yield


@pytest.mark.parametrize('acceleration,yaw', [(0.0, 0.0), (1.5, 0.0), (-1.5, 0.0), (0.0, .06), (0.0, -.06)],
                         ids=['constant_ego', 'accelerating_ego', 'braking_ego', 'left_turn', 'right_turn'])
def test_current_position_velocity_matches_truth_during_normal_ego_motion(acceleration, yaw, record_property):
  rows = []
  with control_pipeline(speed=10.0, cruise=14.0) as pipeline:
    with sensor_input(pipeline), model_turn(pipeline, yaw):
      for frame in range(28):
        ego_speed, target = world_motion(frame, acceleration=acceleration, yaw=yaw)
        row = pipeline.step(speed=ego_speed, accel=acceleration, radar=[target], model_accel=0.0)
        rows.append(capture(pipeline, row, truth_speed=target.speed, truth_distance=target.distance))
  record(record_property, f'normal_motion_{acceleration}_{yaw}', rows)
  assert any(current_quality(track)['qualified'] for row in rows for track in row['fusion_tracks'])
  for row in rows[-8:]:
    track, = row['fusion_tracks']
    quality = current_quality(track)
    assert quality['qualified'] and not track['velocityGood']
    assert abs(quality['longSpeed'] - row['truth_speed']) < .6
    assert abs(track['groundSpeed'] - row['truth_speed']) < .6
    assert row['solver'] == 0 and not row['radar']['fusionRiskUnresolved']
    assert_formal_position_quality(row)


def test_repeated_can_publications_do_not_count_as_independent_position_scans(record_property):
  frozen_frames, rows = 8, []
  with control_pipeline(speed=14.0, cruise=18.0) as pipeline:
    with sensor_input(pipeline, freeze_until=frozen_frames):
      for frame in range(30):
        distance = 30.0 - 10.0 * frame * STEP
        row = pipeline.step(speed=14.0, radar=[Target(distance, 4.0, oem_id=25)])
        rows.append(capture(pipeline, row, truth_speed=4.0, truth_distance=distance))
  record(record_property, 'duplicate_can_then_independent_scans', rows)
  first_track, = rows[0]['fusion_tracks']
  for row in rows[:frozen_frames]:
    assert not formal_objects(row)
    for track in row['fusion_tracks']:
      quality = current_quality(track)
      assert track['positionSeq'] == first_track['positionSeq']
      assert track['positionMonoTime'] == first_track['positionMonoTime']
      assert track['observations'] == 1 and quality['sampleCount'] <= 1 and not quality['qualified']
  returned = rows[frozen_frames:]
  admitted = next(row for row in returned if formal_objects(row))
  assert admitted['time'] - returned[0]['time'] <= .5 + 1e-9
  for row in returned:
    assert_formal_position_quality(row)
    assert not row['radar']['fusionRiskUnresolved']
  assert rows[-1]['radar']['fusionHealthy'] and not rows[-1]['consumer_degraded']


def test_new_range_errors_withdraw_quality_and_fresh_scans_restore_without_latched_target(record_property):
  error_frame, rows = 20, []
  errors = {error_frame: 2.8}
  with control_pipeline(speed=14.0, cruise=18.0) as pipeline:
    # One physical target continues the same motion. The range sensor has a
    # single artificial +2.8 m error below the old 3 m position innovation floor.
    # The 500/300 ms fits have different sensitivity to its newest sample;
    # smaller alternating errors need not fail their statistical comparison.
    # This is a quality withdrawal check, not a false-object/braking verdict.
    with sensor_input(pipeline, range_error=lambda frame: errors.get(frame, 0.0)):
      for frame in range(38):
        distance = 40.0 - 10.0 * frame * STEP
        row = pipeline.step(speed=14.0, radar=[Target(distance, 4.0, oem_id=25)])
        rows.append(capture(pipeline, row, truth_speed=4.0, truth_distance=distance))
  record(record_property, 'range_error_quality_withdrawal_and_recovery', rows)
  prior, = formal_objects(rows[error_frame - 1])
  prior_track, = rows[error_frame - 1]['fusion_tracks']
  assert current_quality(prior_track)['qualified']
  withdrawn = []
  for fault in rows[error_frame:error_frame + len(errors)]:
    fault_track, = fault['fusion_tracks']
    assert fault_track['active'] and fault_track['trackId'] == prior['evidence']['trackId']
    assert fault_track['lastInvalidReason'] == prior_track['lastInvalidReason']
    assert fault_track['positionSeq'] > prior_track['positionSeq']
    if not current_quality(fault_track)['qualified']:
      withdrawn.append(fault)
      assert not formal_objects(fault)
  assert withdrawn, 'Inconsistent current range samples reused the old qualified position velocity'
  for row in rows:
    assert_formal_position_quality(row)
    assert not row['radar']['fusionRiskUnresolved']
  for row in rows[-5:]:
    lead, = formal_objects(row)
    assert lead['evidence']['trackId'] == prior['evidence']['trackId']
    assert 'positionVelocity' in lead['evidence']['reason'] and lead['evidence']['brakeQualified']
    assert row['radar']['fusionHealthy'] and not row['radar']['fusionDegraded'] and not row['consumer_degraded']
    assert row['solver'] == 0


@pytest.mark.parametrize('component', ['native', 'current_same_vision'])
def test_independent_current_velocity_does_not_wait_for_position_convergence(component, record_property):
  rows = run_closing_truth('moving_slow_car', native_velocity=component == 'native',
                           visual=component == 'current_same_vision')
  record(record_property, f'position_quality_bypass_{component}', rows)
  independently_qualified = []
  for row in rows:
    if not row['radar']['fusionActive']:
      assert row['frame'] == 0  # Only DEC's first legacy-vision bootstrap lacks fusion evidence.
      continue
    tracks = {track['trackId']: track for track in row['fusion_tracks']}
    assert_formal_position_quality(row)
    for lead in formal_objects(row):
      claim = lead['evidence']
      track = tracks.get(claim['trackId'])
      if track is None or current_quality(track)['qualified']:
        continue
      if component == 'native':
        if track['velocityGood'] and claim['source'] == 'radar' and 'positionVelocity' not in claim['reason']:
          independently_qualified.append(row)
      elif claim['source'] == 'fused' and 'visionVelocity' in claim['reason']:
        assert not track['velocityGood']
        independently_qualified.append(row)
  assert independently_qualified, 'Current native/same-vision velocity was blocked by position-only convergence'
  assert all(row['radar']['fusionHealthy'] and not row['consumer_degraded'] for row in independently_qualified)
