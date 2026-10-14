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


@pytest.mark.parametrize('direction', [-1, 1])
def test_quarter_turn_rotates_position_vectors_and_covariances(direction):
  quality = PositionVelocityQuality()
  observe(quality, 0, distance=37.0, y=2.3, position_variance=2.25, lateral_position_variance=.81,
          innovation=(.3, -.8), innovation_covariance=(.9, .17, 1.7), correction=(.7, -.4),
          velocity_covariance=(.6, -.13, 1.4))
  previous = quality.samples[0]
  dt, ego = STEP / 1e9, 7.0
  observe(quality, 1, ego=ego, ego_yaw=direction * math.pi / (2 * dt))
  transformed = quality.samples[0]

  # A quarter turn swaps axes, negates cross covariance and rotates both vectors.
  assert (transformed.x, transformed.y) == pytest.approx(
    (direction * previous.y, -direction * (previous.x - ego * dt)), rel=1e-12, abs=1e-12)
  assert (transformed.innovation_x, transformed.innovation_y) == pytest.approx(
    (direction * previous.innovation_y, -direction * previous.innovation_x), rel=1e-12, abs=1e-12)
  assert (transformed.correction_x, transformed.correction_y) == pytest.approx(
    (direction * previous.correction_y, -direction * previous.correction_x), rel=1e-12, abs=1e-12)
  for rotated, expected in (
    ((transformed.rxx, transformed.rxy, transformed.ryy), (previous.ryy, -previous.rxy, previous.rxx)),
    ((transformed.sxx, transformed.sxy, transformed.syy), (previous.syy, -previous.sxy, previous.sxx)),
    ((transformed.vxx, transformed.vxy, transformed.vyy), (previous.vyy, -previous.vxy, previous.vxx)),
  ):
    assert rotated == pytest.approx(expected, rel=1e-12, abs=1e-12)
  assert (transformed.stamp, transformed.sequence, transformed.innovation_known) == (
    previous.stamp, previous.sequence, previous.innovation_known)


def test_history_sample_stays_immutable_across_observations():
  quality = PositionVelocityQuality()
  observe(quality, 0, distance=37.0, y=2.3)
  historical = quality.samples[0]
  with pytest.raises(AttributeError):
    historical.x = 0.0
  observe(quality, 1, ego_yaw=.2)
  assert (historical.x, historical.y) == (37.0, 2.3)
  assert quality.samples[0] is not historical


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
  original_samples = tuple(quality.samples)
  last_position = quality.samples[-1].x, quality.samples[-1].y
  tentative = quality.copy()
  observe(tentative, 9, ego_yaw=.2, correction=(3.0, 0.0))
  assert quality.snapshot(active=True) == original
  assert tuple(quality.samples) == original_samples
  assert (quality.samples[-1].x, quality.samples[-1].y) == last_position
  assert tentative.samples[-2].sequence == original_samples[-1].sequence
  assert tentative.samples[-2] is not original_samples[-1]
  assert not tentative.evidence['qualified']
  assert observe(tentative, 10, distance=math.nan)['reason'] == 'invalidComponents'
  assert not tentative.samples
  assert tuple(quality.samples) == original_samples
  assert quality.snapshot(active=True) == original
