"""Offline trajectory diagnostics, not a control request or movement-authority policy."""

from bisect import bisect_right
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real


LEGACY_MIN_STABLE_DELAY_S = 0.3


def _finite_scalar(value, name):
  if isinstance(value, bool) or not isinstance(value, Real):
    raise ValueError(f"{name} must be a finite real number")
  try:
    result = float(value)
  except OverflowError as error:
    raise ValueError(f"{name} must be a finite real number") from error
  if not isfinite(result):
    raise ValueError(f"{name} must be a finite real number")
  return result


def _finite_series(values, name):
  try:
    return tuple(_finite_scalar(value, name) for value in values)
  except TypeError as error:
    raise ValueError(f"{name} must be a one-dimensional sequence") from error


def _validate_grid(times_s):
  if len(times_s) < 2 or times_s[0] != 0.0:
    raise ValueError("time grid must contain at least two nodes and start at the declared epoch")
  if any(end <= start for start, end in zip(times_s[:-1], times_s[1:], strict=True)):
    raise ValueError("time grid must be strictly increasing")


def _linear_sample(times_s, values, horizon_s):
  if not 0.0 <= horizon_s <= times_s[-1]:
    raise ValueError("legacy sampling horizon is outside the supplied time grid")
  if horizon_s == times_s[-1]:
    return values[-1]
  index = bisect_right(times_s, horizon_s) - 1
  slope = (values[index + 1] - values[index]) / (times_s[index + 1] - times_s[index])
  return _finite_scalar(values[index] + slope * (horizon_s - times_s[index]), "linear sample")


@dataclass(frozen=True)
class TrajectorySample:
  epoch_ns: int
  horizon_s: float
  position: float
  speed: float
  acceleration: float
  jerk: float | None
  interval_index: int | None
  side: str


@dataclass(frozen=True)
class NodeConsistencyResidual:
  interval_index: int
  horizon_s: float
  position: float
  speed: float
  acceleration: float


@dataclass(frozen=True)
class SpeedExtrema:
  start_horizon_s: float
  end_horizon_s: float
  minimum: TrajectorySample
  maximum: TrajectorySample
  candidates: tuple[TrajectorySample, ...]


@dataclass(frozen=True)
class RequestAudit:
  requested_sample: TrajectorySample
  effective_legacy_sample: TrajectorySample
  exact_ramp_request: float
  effective_exact_ramp_request: float
  legacy_ramp_request: float
  legacy_query_speed: float
  legacy_target_speed: float
  legacy_speed_residual: float
  legacy_initial_speed_residual: float
  legacy_initial_acceleration_residual: float


@dataclass(frozen=True)
class PiecewiseJerkTrajectory:
  """Sample stored x=(position, speed, acceleration) nodes under constant interval jerk.

  Node times are seconds relative to the caller-declared epoch_ns, not implicitly relative
  to publication or reception. The physical correspondence of that epoch requires separate
  evidence. Samples are right-continuous at internal nodes; the terminal node uses its stored
  state with jerk=None because no outgoing control interval was supplied. No extrapolation
  or speed clipping is performed. Solver success does not imply physical feasibility.

  Each polynomial is anchored at its own stored left node. Dynamics residuals are reported
  separately, never repaired by overwriting nodes or silently integrating across a mismatch.
  """

  epoch_ns: int
  node_times_s: tuple[float, ...]
  positions: tuple[float, ...]
  speeds: tuple[float, ...]
  accelerations: tuple[float, ...]
  jerks: tuple[float, ...]
  solver_status: int

  def __post_init__(self):
    if isinstance(self.epoch_ns, bool) or not isinstance(self.epoch_ns, Integral) or self.epoch_ns < 0:
      raise ValueError("epoch_ns must be a nonnegative integer")
    if isinstance(self.solver_status, bool) or not isinstance(self.solver_status, Integral) or self.solver_status != 0:
      raise ValueError("sampling requires an explicitly successful solver status")
    object.__setattr__(self, "epoch_ns", int(self.epoch_ns))
    object.__setattr__(self, "solver_status", int(self.solver_status))
    for name in ("node_times_s", "positions", "speeds", "accelerations", "jerks"):
      object.__setattr__(self, name, _finite_series(getattr(self, name), name))
    _validate_grid(self.node_times_s)
    node_count = len(self.node_times_s)
    if any(len(values) != node_count for values in (self.positions, self.speeds, self.accelerations)):
      raise ValueError("position, speed and acceleration must each have one value per node")
    if len(self.jerks) != node_count - 1:
      raise ValueError("jerk must have one value per interval, with no terminal control")

  def _horizon(self, horizon_s):
    horizon_s = _finite_scalar(horizon_s, "horizon_s")
    if not 0.0 <= horizon_s <= self.node_times_s[-1]:
      raise ValueError("sampling horizon is outside the trajectory")
    return horizon_s

  def _interval_sample(self, interval_index, horizon_s, side="right"):
    elapsed = horizon_s - self.node_times_s[interval_index]
    speed = self.speeds[interval_index]
    acceleration = self.accelerations[interval_index]
    jerk = self.jerks[interval_index]
    position_at_horizon = self.positions[interval_index] + speed * elapsed + acceleration * elapsed * elapsed / 2.0
    position_at_horizon += jerk * elapsed * elapsed * elapsed / 6.0
    speed_at_horizon = speed + acceleration * elapsed + jerk * elapsed * elapsed / 2.0
    acceleration_at_horizon = acceleration + jerk * elapsed
    return TrajectorySample(
      self.epoch_ns, horizon_s,
      _finite_scalar(position_at_horizon, "sample position"),
      _finite_scalar(speed_at_horizon, "sample speed"),
      _finite_scalar(acceleration_at_horizon, "sample acceleration"),
      jerk, interval_index, side,
    )

  def sample(self, horizon_s):
    """Return a state and its right-hand jerk, or the terminal state with no jerk."""
    horizon_s = self._horizon(horizon_s)
    if horizon_s == self.node_times_s[-1]:
      return TrajectorySample(
        self.epoch_ns, horizon_s, self.positions[-1], self.speeds[-1], self.accelerations[-1], None, None, "terminal",
      )
    return self._interval_sample(bisect_right(self.node_times_s, horizon_s) - 1, horizon_s)

  def consistency_residuals(self):
    """Return stored next node minus the preceding interval's polynomial endpoint."""
    residuals = []
    for interval_index, horizon_s in enumerate(self.node_times_s[1:]):
      predicted = self._interval_sample(interval_index, horizon_s, "left_limit")
      node_index = interval_index + 1
      residuals.append(NodeConsistencyResidual(
        interval_index, horizon_s,
        _finite_scalar(self.positions[node_index] - predicted.position, "position residual"),
        _finite_scalar(self.speeds[node_index] - predicted.speed, "speed residual"),
        _finite_scalar(self.accelerations[node_index] - predicted.acceleration, "acceleration residual"),
      ))
    return tuple(residuals)

  def speed_extrema(self, start_horizon_s=0.0, end_horizon_s=None):
    """Inspect nodes, interval endpoints and every interior root of acceleration.

    Both sides of inconsistent nodes are retained. Thus minimum/maximum may be an interval
    limit rather than an attained node value; the sample's side makes that distinction explicit.
    Negative speeds remain visible and are not converted into movement intent.
    """
    start_horizon_s = self._horizon(start_horizon_s)
    end_horizon_s = self._horizon(self.node_times_s[-1] if end_horizon_s is None else end_horizon_s)
    if end_horizon_s < start_horizon_s:
      raise ValueError("speed-extrema window must not run backwards")
    candidates = [self.sample(start_horizon_s)]
    if end_horizon_s != start_horizon_s:
      candidates.append(self.sample(end_horizon_s))
    candidates.extend(self.sample(horizon_s) for horizon_s in self.node_times_s if start_horizon_s < horizon_s < end_horizon_s)
    for interval_index, (start, end) in enumerate(zip(self.node_times_s[:-1], self.node_times_s[1:], strict=True)):
      if end <= start_horizon_s or start >= end_horizon_s:
        continue
      window_start, window_end = max(start, start_horizon_s), min(end, end_horizon_s)
      if window_end == window_start:
        continue
      candidates.append(self._interval_sample(interval_index, window_start))
      candidates.append(self._interval_sample(interval_index, window_end, "left_limit" if window_end == end else "right"))
      if self.jerks[interval_index] != 0.0:
        stationary_elapsed = -self.accelerations[interval_index] / self.jerks[interval_index]
        if window_start - start < stationary_elapsed < window_end - start:
          candidates.append(self._interval_sample(interval_index, start + stationary_elapsed))
    return SpeedExtrema(
      start_horizon_s, end_horizon_s,
      min(candidates, key=lambda sample: sample.speed), max(candidates, key=lambda sample: sample.speed), tuple(candidates),
    )

  def audit_request(self, horizon_s, *, sampled_times_s=None, sampled_speeds=None, sampled_accelerations=None):
    """Compare exact a(H), exact equivalent-ramp requests and the legacy sampled request.

    Optional legacy samples must describe the same declared epoch. Omitting all three uses
    linear interpolation of native speed nodes; supplying all three reproduces a published
    grid without regenerating or replacing its values. Initial-state and speed residuals
    expose disagreements with the native trajectory.

    The exact ramp request is 2*[v(H)-v(0)]/H-a(0), not generally a(H). For H<0.3 the old
    helper scales the speed change at 0.3 seconds before applying its formula, so its
    effective horizon is 0.3. The requested and effective samples are both retained.
    Out-of-domain queries are rejected, not silently endpoint-clamped as numpy.interp can do.
    This comparison does not identify an actuator inverse or authorize either request.
    """
    horizon_s = self._horizon(horizon_s)
    if horizon_s == 0.0:
      raise ValueError("request horizon must be positive")
    effective_horizon_s = self._horizon(max(horizon_s, LEGACY_MIN_STABLE_DELAY_S))
    supplied = (sampled_times_s is not None, sampled_speeds is not None, sampled_accelerations is not None)
    if any(supplied) and not all(supplied):
      raise ValueError("legacy time, speed and acceleration arrays must be supplied together")
    if not any(supplied):
      sampled_times_s, sampled_speeds, sampled_accelerations = self.node_times_s, self.speeds, self.accelerations
    else:
      sampled_times_s = _finite_series(sampled_times_s, "sampled_times_s")
      sampled_speeds = _finite_series(sampled_speeds, "sampled_speeds")
      sampled_accelerations = _finite_series(sampled_accelerations, "sampled_accelerations")
      _validate_grid(sampled_times_s)
      if len(sampled_speeds) != len(sampled_times_s) or len(sampled_accelerations) != len(sampled_times_s):
        raise ValueError("legacy arrays must have matching lengths")
      if sampled_times_s[-1] > self.node_times_s[-1]:
        raise ValueError("legacy grid extends beyond the native trajectory")
    requested_sample = self.sample(horizon_s)
    effective_sample = self.sample(effective_horizon_s)
    legacy_query_speed = _linear_sample(sampled_times_s, sampled_speeds, effective_horizon_s)
    legacy_target_speed = legacy_query_speed
    if horizon_s < LEGACY_MIN_STABLE_DELAY_S:
      legacy_target_speed = sampled_speeds[0] + horizon_s / LEGACY_MIN_STABLE_DELAY_S * (legacy_query_speed - sampled_speeds[0])
    return RequestAudit(
      requested_sample, effective_sample,
      _finite_scalar(2.0 * (requested_sample.speed - self.speeds[0]) / horizon_s - self.accelerations[0], "exact ramp request"),
      _finite_scalar(2.0 * (effective_sample.speed - self.speeds[0]) / effective_horizon_s - self.accelerations[0], "effective ramp request"),
      _finite_scalar(2.0 * (legacy_target_speed - sampled_speeds[0]) / horizon_s - sampled_accelerations[0], "legacy ramp request"),
      legacy_query_speed, legacy_target_speed,
      _finite_scalar(legacy_query_speed - effective_sample.speed, "legacy speed residual"),
      _finite_scalar(sampled_speeds[0] - self.speeds[0], "legacy initial speed residual"),
      _finite_scalar(sampled_accelerations[0] - self.accelerations[0], "legacy initial acceleration residual"),
    )
