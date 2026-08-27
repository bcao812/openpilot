import math

import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_planner import (MAX_PITCH_BIAS, MIN_ALLOW_THROTTLE_SPEED,
                                                                   PITCH_BIAS_TAU, PitchBiasEstimator,
                                                                   get_coast_accel)

# A residual in the range the 2019 Accord actually shows: raw pitch median runs +2.0 to +2.4 deg
# across four routes, and the offline replay of this estimator settles at +1.74 to +1.94.
MEASURED_BIAS = math.radians(2.17)
MOVING = MIN_ALLOW_THROTTLE_SPEED + 1.0


def _run(est, pitch_seq, v_ego=MOVING):
  out = 0.0
  for p in pitch_seq:
    out = est.update(p, v_ego)
  return out


def _seconds(t):
  return int(t / DT_MDL)


class TestPitchBiasEstimator:
  def test_converges_to_a_constant_offset(self):
    """Flat road plus a fixed mount residual: the estimate is the residual."""
    est = PitchBiasEstimator(DT_MDL)
    bias = _run(est, [MEASURED_BIAS] * _seconds(5 * PITCH_BIAS_TAU))
    assert math.isclose(bias, MEASURED_BIAS, abs_tol=math.radians(0.01))

  def test_first_sample_is_taken_whole(self):
    """initialized=False, so there is no warmup from zero -- one frame is already close.

    This is what keeps a long tau affordable: the filter starts at the road rather than at 0.
    """
    est = PitchBiasEstimator(DT_MDL)
    assert math.isclose(est.update(MEASURED_BIAS, MOVING), MEASURED_BIAS, abs_tol=1e-9)

  def test_unbiased_device_learns_nothing(self):
    """A device with no residual must be left alone -- this is why the bias is estimated rather
    than written down as a per-platform constant."""
    est = PitchBiasEstimator(DT_MDL)
    rng = np.random.default_rng(0)
    grade = rng.normal(0.0, math.radians(1.5), _seconds(10 * PITCH_BIAS_TAU))
    assert abs(_run(est, grade)) < math.radians(0.2)

  def test_zero_mean_grade_does_not_move_the_estimate(self):
    """The premise the whole thing rests on: mean road grade over a drive is ~0, so an alternating
    hill leaves the bias where it was."""
    est = PitchBiasEstimator(DT_MDL)
    n = _seconds(PITCH_BIAS_TAU)
    seq = []
    for _ in range(10):
      seq += [MEASURED_BIAS + math.radians(4.0)] * (n // 10)
      seq += [MEASURED_BIAS - math.radians(4.0)] * (n // 10)
    assert math.isclose(_run(est, seq), MEASURED_BIAS, abs_tol=math.radians(0.5))

  def test_sustained_climb_is_clamped(self):
    """The failure mode the clamp exists for. A drive that is uphill throughout would otherwise
    have its grade learned as bias and cancelled."""
    est = PitchBiasEstimator(DT_MDL)
    bias = _run(est, [math.radians(10.0)] * _seconds(20 * PITCH_BIAS_TAU))
    assert math.isclose(bias, MAX_PITCH_BIAS, abs_tol=1e-9)

    est = PitchBiasEstimator(DT_MDL)
    bias = _run(est, [math.radians(-10.0)] * _seconds(20 * PITCH_BIAS_TAU))
    assert math.isclose(bias, -MAX_PITCH_BIAS, abs_tol=1e-9)

  def test_standstill_is_not_learned(self):
    """Parking on a slope is a real pitch reading and a wrong bias sample, and it is the reading a
    car dwells on longest."""
    est = PitchBiasEstimator(DT_MDL)
    parked = _run(est, [math.radians(5.0)] * _seconds(10 * PITCH_BIAS_TAU), v_ego=0.0)
    assert parked == 0.0
    assert not est.filter.initialized

  def test_learning_starts_only_above_the_speed_gate(self):
    est = PitchBiasEstimator(DT_MDL)
    assert est.update(MEASURED_BIAS, MIN_ALLOW_THROTTLE_SPEED) == 0.0
    assert est.update(MEASURED_BIAS, MIN_ALLOW_THROTTLE_SPEED + 0.1) != 0.0

  def test_non_finite_pitch_is_ignored(self):
    est = PitchBiasEstimator(DT_MDL)
    est.update(MEASURED_BIAS, MOVING)
    before = est.filter.x
    for bad in (float('nan'), float('inf'), float('-inf')):
      est.update(bad, MOVING)
    assert est.filter.x == before

  def test_estimate_is_never_outside_the_clamp(self):
    est = PitchBiasEstimator(DT_MDL)
    rng = np.random.default_rng(1)
    for p in rng.normal(math.radians(8.0), math.radians(5.0), _seconds(3 * PITCH_BIAS_TAU)):
      assert abs(est.update(p, MOVING)) <= MAX_PITCH_BIAS


class TestCoastAccelDebias:
  def test_flat_road_no_longer_reads_as_a_hill(self):
    """The defect this exists to fix: on flat road with the measured residual, the coast limit was
    -0.51 m/s^2 where the true value is -0.30."""
    biased = get_coast_accel(MEASURED_BIAS)
    assert biased < -0.5

    est = PitchBiasEstimator(DT_MDL)
    bias = _run(est, [MEASURED_BIAS] * _seconds(5 * PITCH_BIAS_TAU))
    corrected = get_coast_accel(MEASURED_BIAS - bias)
    assert math.isclose(corrected, get_coast_accel(0.0), abs_tol=0.01)

  def test_a_real_hill_still_reaches_the_limit(self):
    """Debiasing must not flatten the signal -- a genuine grade on top of the residual has to
    survive, or the term it feeds is dead."""
    est = PitchBiasEstimator(DT_MDL)
    bias = _run(est, [MEASURED_BIAS] * _seconds(5 * PITCH_BIAS_TAU))

    hill = MEASURED_BIAS + math.radians(4.0)
    assert math.isclose(get_coast_accel(hill - bias), get_coast_accel(math.radians(4.0)), abs_tol=0.01)

    dip = MEASURED_BIAS - math.radians(4.0)
    assert math.isclose(get_coast_accel(dip - bias), get_coast_accel(math.radians(-4.0)), abs_tol=0.01)

  def test_correction_is_monotonic_in_pitch(self):
    est = PitchBiasEstimator(DT_MDL)
    bias = _run(est, [MEASURED_BIAS] * _seconds(5 * PITCH_BIAS_TAU))
    pitches = [MEASURED_BIAS + math.radians(d) for d in (-6, -3, 0, 3, 6)]
    accels = [get_coast_accel(p - bias) for p in pitches]
    assert all(a > b for a, b in zip(accels[:-1], accels[1:], strict=True))
