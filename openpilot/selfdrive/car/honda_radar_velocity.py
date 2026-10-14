"""Current position-history consistency for the Accord velocity fallback.

This is model-consistency evidence, not calibrated radar accuracy. Samples are
independent accepted position scans. Their coordinates are brought into the
current ego frame before fitting; no missing target is predicted or published.
The native velocity estimator and its measurement-noise model are unchanged.
"""
from collections import deque
import math
from typing import NamedTuple


VERSION = 1
MAX_SAMPLES = 16
LONG_WINDOW_NS = 500_000_000
SHORT_WINDOW_NS = 300_000_000
MAX_GAP_NS = 200_000_000
MIN_SAMPLES = 6
MIN_SPAN = .25
FIT_MIN_SPAN = .12
MIN_CONSISTENT_SAMPLES = 3
# Initial engineering consistency bounds. Physical positive cases and native
# road replay must qualify changes; these values are not OEM sensor limits.
MIN_SPEED_DIFFERENCE = 2.5
SIGMA_MULTIPLIER = 2.0
MAX_INNOVATION_BIAS = 1.75
MAX_CORRECTION_RATIO = 1.5


class _Sample(NamedTuple):
  stamp: int
  sequence: int
  x: float
  y: float
  rxx: float
  rxy: float
  ryy: float
  innovation_x: float
  innovation_y: float
  sxx: float
  sxy: float
  syy: float
  correction_x: float
  correction_y: float
  vxx: float
  vxy: float
  vyy: float
  innovation_known: bool


def _rotate_covariance(xx, xy, yy, cc, ss, cs, twice_cs, cc_minus_ss):
  return (cc * xx - twice_cs * xy + ss * yy,
          cs * (xx - yy) + cc_minus_ss * xy,
          ss * xx + twice_cs * xy + cc * yy)


def _fit(samples, epoch):
  """Weighted slope and coefficients, retaining overlapping-window covariance."""
  if len(samples) < 3:
    return None
  times = [(sample.stamp - epoch) / 1e9 for sample in samples]
  weights = [1.0 / max(sample.rxx, .01) for sample in samples]
  total = sum(weights)
  center = sum(t * w for t, w in zip(times, weights, strict=True)) / total
  denominator = sum(w * (t - center) ** 2 for t, w in zip(times, weights, strict=True))
  if denominator <= 1e-9:
    return None
  coefficients = {sample.sequence: w * (t - center) / denominator
                  for sample, t, w in zip(samples, times, weights, strict=True)}
  speed = sum(coefficients[sample.sequence] * sample.x for sample in samples)
  variance = sum(coefficients[sample.sequence] ** 2 * sample.rxx for sample in samples)
  return speed, variance, coefficients


class PositionVelocityQuality:
  def __init__(self):
    self.samples = deque(maxlen=MAX_SAMPLES)
    self.epoch_ns = 0
    self.consistent_samples = 0
    self.evidence = self._empty(0, 0, 'noPosition')

  @staticmethod
  def _empty(stamp, sequence, reason):
    return {'version': VERSION, 'positionSeq': sequence, 'positionMonoTime': stamp,
            'sampleCount': 0, 'span': 0.0, 'shortSpeed': 0.0, 'longSpeed': 0.0,
            'speedStd': 0.0, 'windowDifference': 0.0, 'windowDifferenceStd': 0.0,
            'innovationBias': 0.0, 'posteriorCorrection': 0.0, 'posteriorCorrectionRatio': 0.0,
            'posteriorSpeedDifference': 0.0, 'consistentSamples': 0, 'qualified': False, 'reason': reason}

  def copy(self):
    other = PositionVelocityQuality()
    other.samples = deque(self.samples, maxlen=MAX_SAMPLES)
    other.epoch_ns = self.epoch_ns
    other.consistent_samples = self.consistent_samples
    other.evidence = self.evidence.copy()
    return other

  def _reset(self):
    self.samples.clear()
    self.consistent_samples = 0
    self.epoch_ns = 0

  def observe(self, *, stamp, sequence, x, y, ego_speed, ego_yaw,
              speed_x, velocity_variance, position_variance=.25, lateral_position_variance=.64,
              innovation=None, innovation_covariance=None, correction=(0.0, 0.0), velocity_covariance=None):
    """Append once per current position; AUX/publication cannot add support.

    ego_speed/yaw describe the interval starting at the preceding position,
    using the same causal ego convention as Track.project. innovation and
    correction are captured before/after that position update, before its AUX.
    """
    velocity_covariance = velocity_covariance if velocity_covariance is not None else (velocity_variance, 0.0, velocity_variance)
    values = (x, y, ego_speed, ego_yaw, speed_x, velocity_variance,
              position_variance, lateral_position_variance, *correction, *velocity_covariance)
    known = innovation is not None and innovation_covariance is not None
    if known:
      values += (*innovation, *innovation_covariance)
    if (type(stamp) is not int or stamp <= 0 or type(sequence) is not int or sequence <= 0
        or not all(math.isfinite(value) for value in values)
        or min(position_variance, lateral_position_variance) <= 0 or min(velocity_variance, velocity_covariance[0], velocity_covariance[2]) < 0
        or known and (innovation_covariance[0] <= 0 or innovation_covariance[2] <= 0)):
      self._reset()
      self.evidence = self._empty(0, 0, 'invalidComponents')
      return
    if self.samples and (stamp <= self.epoch_ns or sequence <= self.samples[-1].sequence):
      if stamp == self.epoch_ns and sequence == self.samples[-1].sequence:
        # The caller already distinguishes conflicting same-sequence scans.
        return
      self._reset()
      self.evidence = self._empty(stamp, sequence, 'positionOrder')
      return
    if self.epoch_ns and stamp - self.epoch_ns > MAX_GAP_NS:
      self._reset()
    if self.epoch_ns:
      dt = (stamp - self.epoch_ns) / 1e9
      c, s = math.cos(-ego_yaw * dt), math.sin(-ego_yaw * dt)
      cc, ss, cs = c * c, s * s, c * s
      # Preserve the original multiplication order: (2 * c) * s.
      twice_cs, cc_minus_ss = 2 * c * s, cc - ss
      ego_distance = ego_speed * dt
      transformed = deque(maxlen=MAX_SAMPLES)
      for sample in self.samples:
        if stamp - sample.stamp > LONG_WINDOW_NS:
          continue
        rxx, rxy, ryy = _rotate_covariance(sample.rxx, sample.rxy, sample.ryy, cc, ss, cs, twice_cs, cc_minus_ss)
        sxx, sxy, syy = _rotate_covariance(sample.sxx, sample.sxy, sample.syy, cc, ss, cs, twice_cs, cc_minus_ss)
        vxx, vxy, vyy = _rotate_covariance(sample.vxx, sample.vxy, sample.vyy, cc, ss, cs, twice_cs, cc_minus_ss)
        # Same short-step ego translation/rotation as the existing producer.
        px = sample.x - ego_distance
        # Fresh immutable samples keep tentative recovery copies independent.
        transformed.append(_Sample(sample.stamp, sample.sequence, c * px - s * sample.y, s * px + c * sample.y,
          rxx, rxy, ryy,
          c * sample.innovation_x - s * sample.innovation_y, s * sample.innovation_x + c * sample.innovation_y,
          sxx, sxy, syy,
          c * sample.correction_x - s * sample.correction_y, s * sample.correction_x + c * sample.correction_y,
          vxx, vxy, vyy, sample.innovation_known))
      self.samples = transformed
    self.epoch_ns = stamp
    ix, iy = innovation if known else (0.0, 0.0)
    sxx, sxy, syy = innovation_covariance if known else (1.0, 0.0, 1.0)
    self.samples.append(_Sample(stamp, sequence, float(x), float(y), float(position_variance), 0.0,
      float(lateral_position_variance), float(ix), float(iy), float(sxx), float(sxy), float(syy),
      float(correction[0]), float(correction[1]), *(float(value) for value in velocity_covariance), known))
    evidence = self._empty(stamp, sequence, 'insufficientPositions')
    samples = tuple(self.samples)
    span = (stamp - samples[0].stamp) / 1e9
    evidence.update(sampleCount=len(samples), span=span)
    long_fit = _fit(samples, stamp)
    short_samples = tuple(sample for sample in samples if stamp - sample.stamp <= SHORT_WINDOW_NS)
    short_fit = _fit(short_samples, stamp)
    if long_fit is None or short_fit is None or span < FIT_MIN_SPAN:
      self.consistent_samples = 0
      self.evidence = evidence
      return
    long_speed, long_variance, long_coefficients = long_fit
    short_speed, short_variance, short_coefficients = short_fit
    difference_variance = sum((short_coefficients.get(sample.sequence, 0.0) - long_coefficients[sample.sequence]) ** 2
                              * sample.rxx for sample in samples)
    measured = [sample for sample in samples if sample.innovation_known]
    bias = sum(sample.innovation_x / math.sqrt(max(.01, sample.sxx)) for sample in measured) / max(1, len(measured))
    recent = measured[-3:]
    max_correction = max((abs(sample.correction_x) for sample in recent), default=0.0)
    correction_ratio = max((abs(sample.correction_x) / math.sqrt(max(.25, sample.vxx))
                            for sample in recent), default=0.0)
    difference = abs(short_speed - long_speed)
    posterior_difference = abs(speed_x - short_speed)
    window_bound = max(MIN_SPEED_DIFFERENCE, SIGMA_MULTIPLIER * math.sqrt(max(0.0, difference_variance)))
    posterior_bound = max(MIN_SPEED_DIFFERENCE, SIGMA_MULTIPLIER * (math.sqrt(max(0.0, short_variance))
                                                                 + math.sqrt(velocity_variance)))
    reasons = []
    if difference > window_bound:
      reasons.append('windowDisagreement')
    if abs(bias) > MAX_INNOVATION_BIAS:
      reasons.append('innovationBias')
    if correction_ratio > MAX_CORRECTION_RATIO:
      reasons.append('posteriorCorrection')
    if posterior_difference > posterior_bound:
      reasons.append('posteriorDisagreement')
    self.consistent_samples = 0 if reasons else min(MAX_SAMPLES, self.consistent_samples + 1)
    qualified = not reasons and len(samples) >= MIN_SAMPLES and span >= MIN_SPAN and self.consistent_samples >= MIN_CONSISTENT_SAMPLES
    reason = '+'.join(reasons) if reasons else 'converged' if qualified else 'insufficientSupport'
    evidence.update(shortSpeed=short_speed, longSpeed=long_speed, speedStd=math.sqrt(max(0.0, long_variance)),
      windowDifference=difference, windowDifferenceStd=math.sqrt(max(0.0, difference_variance)),
      innovationBias=bias, posteriorCorrection=max_correction, posteriorCorrectionRatio=correction_ratio,
      posteriorSpeedDifference=posterior_difference, consistentSamples=self.consistent_samples,
      qualified=qualified, reason=reason)
    self.evidence = evidence

  def snapshot(self, *, active):
    evidence = self.evidence.copy()
    if not active:
      evidence.update(qualified=False, reason='inactive')
    return evidence
