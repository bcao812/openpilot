"""Geometry and independent-sample contracts for position-speed evidence."""
import math

import pytest

from openpilot.selfdrive.car.honda_radar_velocity import PositionVelocityQuality


START = 10_000_000_000
STEP = 66_000_000


def observe(quality, frame, *, distance=None, speed=5.0, ego=25.0, **changes):
  args = dict(stamp=START + frame * STEP, sequence=frame + 1,
              x=80.0 + (speed - ego) * frame * STEP / 1e9 if distance is None else distance,
              y=0.0, ego_speed=ego, ego_yaw=0.0, speed_x=speed, velocity_variance=.5,
              innovation=(0.0, 0.0), innovation_covariance=(.5, 0.0, 1.0), correction=(0.0, 0.0))
  args.update(changes)
  quality.observe(**args)
  return quality.snapshot(active=True)


@pytest.mark.parametrize('speed,ego', [(0.0, 25.0), (5.0, 25.0), (12.0, 20.0)])
def test_constant_ground_motion_converges_without_native_or_visual_speed(speed, ego):
  quality = PositionVelocityQuality()
  results = [observe(quality, frame, speed=speed, ego=ego) for frame in range(12)]
  assert results[5]['qualified']
  for result in results[5:]:
    assert result['shortSpeed'] == pytest.approx(speed, abs=1e-9)
    assert result['longSpeed'] == pytest.approx(speed, abs=1e-9)
    assert result['windowDifference'] < 1e-9
    assert result['span'] <= .5


def test_ego_acceleration_does_not_become_target_acceleration():
  quality = PositionVelocityQuality()
  distance, ego = 60.0, 15.0
  for frame in range(12):
    result = observe(quality, frame, distance=distance, speed=8.0, ego=ego)
    next_ego = ego + .2
    distance += (8.0 - next_ego) * STEP / 1e9
    # The following sample uses this interval's exact ego speed.
    ego = next_ego
  assert result['qualified']
  assert result['shortSpeed'] == pytest.approx(8.0, abs=1e-9)


@pytest.mark.parametrize('yaw', [-.2, .2])
def test_turning_frame_preserves_a_stationary_world_target(yaw):
  quality = PositionVelocityQuality()
  x, y, ego = 60.0, .4, 12.0
  dt = STEP / 1e9
  c, s = math.cos(-yaw * dt), math.sin(-yaw * dt)
  for frame in range(12):
    result = observe(quality, frame, distance=x, speed=0.0, ego=ego, y=y, ego_yaw=yaw)
    x, y = c * (x - ego * dt) - s * y, s * (x - ego * dt) + c * y
  assert result['qualified']
  assert result['shortSpeed'] == pytest.approx(0.0, abs=1e-9)
  assert result['longSpeed'] == pytest.approx(0.0, abs=1e-9)


def test_repeated_publication_cannot_mature_a_position():
  quality = PositionVelocityQuality()
  first = observe(quality, 0)
  for _ in range(25):
    assert observe(quality, 0) == first
  assert first['sampleCount'] == 1 and not first['qualified']


def test_new_inconsistent_measurement_revokes_quality_and_requires_new_support():
  quality = PositionVelocityQuality()
  for frame in range(9):
    assert observe(quality, frame)['positionSeq'] == frame + 1
  assert quality.evidence['qualified']
  rejected = observe(quality, 9, correction=(3.0, 0.0))
  assert not rejected['qualified'] and rejected['reason'] == 'posteriorCorrection'
  assert rejected['consistentSamples'] == 0
  for _ in range(20):
    assert observe(quality, 9) == rejected
  for frame in range(10, 17):
    result = observe(quality, frame)
  assert result['qualified']
  assert not quality.snapshot(active=False)['qualified']


def test_gap_and_bad_order_do_not_inherit_old_consistency():
  quality = PositionVelocityQuality()
  for frame in range(9):
    observe(quality, frame)
  assert quality.evidence['qualified']
  after_gap = observe(quality, 15)
  assert after_gap['sampleCount'] == 1 and not after_gap['qualified']
  backwards = observe(quality, 14)
  assert backwards['reason'] == 'positionOrder' and not backwards['qualified']


def test_tentative_recovery_copy_has_independent_quality_state():
  quality = PositionVelocityQuality()
  for frame in range(9):
    observe(quality, frame)
  original = quality.snapshot(active=True)
  tentative = quality.copy()
  observe(tentative, 9, correction=(3.0, 0.0))
  assert quality.snapshot(active=True) == original
  assert not tentative.evidence['qualified']
