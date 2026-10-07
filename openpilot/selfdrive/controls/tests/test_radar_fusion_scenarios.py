"""Consistent-motion radar-only stop/departure through decoded Honda CAN.

Both native radar range and relative speed are regenerated from the moving
plant and the same analytic target. This is an ideal delay/lag regression,
not a replay of fixed sensor observations or a road-performance estimate.
"""
import json

import numpy as np

from openpilot.selfdrive.car.tests.honda_radar_control_test_support import IdealPlant, Target, control_pipeline


def target_motion(timestamp):
  """3 m/s, brake at 2 s, stop at 4 s, leave at 9 s with .8 m/s²."""
  if timestamp < 2.0:
    return 22.0 + 3.0 * timestamp, 3.0
  if timestamp < 4.0:
    elapsed = timestamp - 2.0
    return 28.0 + 3.0 * elapsed - .75 * elapsed**2, 3.0 - 1.5 * elapsed
  if timestamp < 9.0:
    return 31.0, 0.0
  elapsed = min(timestamp - 9.0, 5.0)
  return 31.0 + .4 * elapsed**2 + 4.0 * max(0.0, timestamp - 14.0), .8 * elapsed


def simulate_radar_following(*, dropout=False):
  plant = IdealPlant(6.0, delay=.2, response_time=.25)
  rows = []
  duration = 12.0 if dropout else 17.0
  with control_pipeline(speed=plant.speed, cruise=10.0) as pipeline:
    for frame in range(round(duration / .05)):
      timestamp = frame * .05
      position, speed = target_motion(timestamp)
      ego_position = plant.position
      gap = position - ego_position
      assert gap > 0.0, f'Collision at {timestamp:.2f}s: gap={gap:.3f}, ego={plant.speed:.3f}, target={speed:.3f}'
      row = pipeline.step(speed=plant.speed, accel=plant.accel, radar=[Target(gap, speed)],
                          observer_online=not dropout or timestamp < 8.0, plant=plant)
      row.update(gap=gap, ego_position=ego_position, target_position=position, target_speed=speed,
                 end_speed=plant.speed, end_position=plant.position)
      rows.append(row)
  commands = [sample['accel'] for row in rows for sample in row['can']]
  request_jerk = np.diff(commands) / .02  # Real CAN cadence, not model cadence.
  hold = [row for row in rows if 6.0 <= row['time'] < 8.0 and all(owner['phase'] == 'hold' for owner in row['owner'])]
  departures = [row for row in rows if row['time'] >= 9.0 and row['speed'] > .2]
  return rows, dict(
    plant='ideal decoded-ACCEL_COMMAND delay .2 s / response .25 s',
    minimum_gap=min(row['gap'] for row in rows),
    final_speed=plant.speed, hold_seconds=len(hold) * .05,
    departure_delay=departures[0]['time'] - 9.0 if departures else None,
    maximum_request_jerk=float(np.max(np.abs(request_jerk))),
    p95_request_jerk=float(np.quantile(np.abs(request_jerk), .95)),
    solver_failures=sum(row['solver'] != 0 for row in rows),
    pedal_conflicts=sum(sample['brake'] and sample['gas'] > 0 for row in rows for sample in row['can']),
  )


def test_radar_only_closed_loop_stops_and_follows_same_target_departure(record_property):
  rows, metrics = simulate_radar_following()
  record_property('radar_stop_departure_metrics', json.dumps(metrics, allow_nan=False))
  record_property('radar_stop_departure_trace', json.dumps(rows, allow_nan=False, separators=(',', ':')))
  # These are bounded regression goals for this ideal plant, not a claimed
  # safe road envelope. Request jerk remains separately reported, not averaged
  # against a collision or a failure to hold/release.
  assert metrics['minimum_gap'] > 1.5, metrics
  assert metrics['hold_seconds'] >= .5, metrics
  assert metrics['departure_delay'] is not None and metrics['departure_delay'] <= 3.0, metrics
  assert metrics['final_speed'] > 1.0, metrics
  assert metrics['solver_failures'] == metrics['pedal_conflicts'] == 0, metrics
  assert all(row['radar']['leadOne']['modelProb'] == 0.0 for row in rows if row['radar']['leadOne']['present'])
  identities = {row['radar']['leadOne']['evidence']['trackId'] for row in rows[20:]}
  assert len(identities) == 1
  assert any(sample['stop'] and sample['brake'] for row in rows for sample in row['can'])
  assert any(sample['release'] and sample['accel'] > 0.0 for row in rows if row['time'] >= 9.0 for sample in row['can'])
  # Verify the measurements came from the current plant rather than an old
  # trajectory or an independently prescribed closing velocity.
  assert all(abs(row['sensor_targets'][0]['distance'] + row['ego_position'] - row['target_position']) < 1e-9
             for row in rows)


def test_radar_dropout_during_closed_loop_hold_cannot_be_cleared_by_ttl(record_property):
  rows, metrics = simulate_radar_following(dropout=True)
  record_property('radar_hold_dropout_metrics', json.dumps(metrics, allow_nan=False))
  record_property('radar_hold_dropout_trace', json.dumps(rows, allow_nan=False, separators=(',', ':')))
  assert metrics['minimum_gap'] > 1.5 and metrics['hold_seconds'] >= .5, metrics
  assert metrics['final_speed'] <= .01 and metrics['departure_delay'] is None, metrics
  dropped = [row for row in rows if row['time'] >= 8.5]
  assert dropped and all(row['radar'].get('fusionDegraded', False) for row in dropped)
  assert all(not row['release'] and all(owner['phase'] == 'hold' and owner['stop'] for owner in row['owner'])
             for row in dropped)
  assert all(sample['brake'] and not sample['release'] and sample['accel'] <= 0.0
             for row in dropped for sample in row['can'])
