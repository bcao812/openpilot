"""Bounded numerical candidates from a previous accepted native trajectory.

These objects do not grant control authority, renew source leases, or change
native status. The producer/consumer additionally gate input health, current
braking policy, operating state and the independent failure-episode budget.
"""
from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np


MAX_DONOR_SOURCE_AGE_NS = 200_000_000
TERMINAL_ACCEL_TOL = 1e-3  # m/s^2; eligibility, not a relaxed acceptance bound
TERMINAL_JERK_TOL = 1e-3  # m/s^3
STATE_ENVELOPE_LEVELS = tuple((.2 / 2**level, .25 / 2**level) for level in range(8)) + ((1e-3, 1e-3),)


def frozen_array(value, *, dtype=float):
  result = np.array(value, dtype=dtype, copy=True)
  result.setflags(write=False)
  return result


def valid_timestamp(value):
  return isinstance(value, Integral) and not isinstance(value, bool) and 0 < value < 1 << 64


@dataclass(frozen=True)
class ContinuationProblem:
  reference_ns: int
  consumed_ns: int
  evidence_expiry_ns: int
  x0: np.ndarray
  parameters: np.ndarray
  state_lower: np.ndarray
  state_upper: np.ndarray
  linear_constraints: tuple
  projections: tuple
  stop_or_passing: bool


@dataclass(frozen=True)
class ContinuationDonor:
  source_ns: int
  producer_ns: int
  instance_ns: int
  states: np.ndarray
  jerks: np.ndarray
  problem: ContinuationProblem


@dataclass(frozen=True)
class ContinuationResult:
  accepted: bool
  reason: str
  states: np.ndarray | None = None
  jerks: np.ndarray | None = None
  tail_extension_s: float = 0.0
  evidence_expiry_ns: int = 0
  donor_source_ns: int = 0
  donor_producer_ns: int = 0
  donor_instance_ns: int = 0
  reference_ns: int = 0
  speed_tolerance: float = 0.0
  accel_tolerance: float = 0.0
  corner_checks: int = 0
  donor_max_accel_deviation: float = 0.0
  donor_max_braking_relaxation: float = 0.0


def build_candidate(donor, problem, *, now_ns, times, integrate, validate):
  """Shift acceleration, then integrate once from the current measured x0.

  The native grids before/after elapsed time have different jerk boundaries.
  Resampling acceleration and independently integrating the resulting constant
  jerk intervals defines a new candidate; interpolating old x/v independently
  would not. Beyond the old horizon, only an approximately constant-speed old
  terminal state may supply a zero-acceleration endpoint. The final fixed-grid
  interval is then reintegrated and checked too, never claimed as old support.
  """
  if (not isinstance(donor, ContinuationDonor) or not isinstance(problem, ContinuationProblem)
      or not isinstance(donor.problem, ContinuationProblem)):
    return ContinuationResult(False, 'missing_donor_or_current_problem')
  clocks = (now_ns, donor.source_ns, donor.producer_ns, donor.instance_ns,
            problem.reference_ns, problem.consumed_ns, problem.evidence_expiry_ns, donor.problem.evidence_expiry_ns)
  if not all(valid_timestamp(value) for value in clocks):
    return ContinuationResult(False, 'invalid_continuation_clock')
  deadline = min(int(problem.evidence_expiry_ns), int(donor.problem.evidence_expiry_ns),
                 int(donor.source_ns) + MAX_DONOR_SOURCE_AGE_NS)
  common = dict(evidence_expiry_ns=deadline, donor_source_ns=int(donor.source_ns),
                donor_producer_ns=int(donor.producer_ns), donor_instance_ns=int(donor.instance_ns),
                reference_ns=int(problem.reference_ns))

  def reject(reason, tail=0.0, **details):
    return ContinuationResult(False, reason, tail_extension_s=tail, **common, **details)

  if (donor.source_ns > donor.producer_ns or donor.instance_ns > donor.producer_ns
      or donor.producer_ns >= now_ns or now_ns != problem.consumed_ns
      or problem.reference_ns > now_ns or problem.reference_ns < donor.source_ns):
    return reject('continuation_clock_order')
  if now_ns >= deadline:
    return reject('continuation_evidence_expired')
  if donor.problem.reference_ns != donor.source_ns:
    return reject('donor_source_identity_mismatch')
  if donor.problem.stop_or_passing or problem.stop_or_passing:
    return reject('continuation_stop_or_passing')
  if problem.x0.shape != (3,) or not np.all(np.isfinite(problem.x0)):
    return reject('invalid_current_initial_state')
  times = np.asarray(times)
  if (times.ndim != 1 or len(times) < 2 or times[0] != 0.0 or not np.all(np.isfinite(times))
      or np.any(np.diff(times) <= 0.0) or donor.states.shape != (len(times), 3)
      or donor.jerks.shape != (len(times)-1,) or not np.all(np.isfinite(donor.states))
      or not np.all(np.isfinite(donor.jerks))):
    return reject('invalid_donor_trajectory')
  if (reason := validate(donor.problem, donor.states, donor.jerks)):
    return reject('donor_' + reason)
  elapsed = (int(problem.reference_ns) - int(donor.source_ns)) / 1e9
  if not 0.0 <= elapsed <= MAX_DONOR_SOURCE_AGE_NS / 1e9:
    return reject('continuation_source_shift_too_large')
  shifted_times = times + elapsed
  tail = max(0.0, float(shifted_times[-1] - times[-1]))
  if tail > 0.0 and (abs(donor.states[-1, 2]) > TERMINAL_ACCEL_TOL or abs(donor.jerks[-1]) > TERMINAL_JERK_TOL):
    return reject('donor_terminal_not_constant_speed', tail)
  accelerations = np.interp(np.minimum(shifted_times, times[-1]), times, donor.states[:, 2])
  accelerations[shifted_times > times[-1]] = 0.0
  accelerations[0] = problem.x0[2]
  jerks = np.diff(accelerations) / np.diff(times)
  states = np.empty_like(donor.states)
  states[0] = problem.x0
  for stage, duration in enumerate(np.diff(times)):
    states[stage+1] = integrate(states[stage], jerks[stage], float(duration))
  if (reason := validate(problem, states, jerks)):
    return reject('current_' + reason, tail)
  # Both acceleration profiles are piecewise linear. Their difference has
  # no interior extremum outside the union of the old and new knot sets.
  overlap_end = float(times[-1] - elapsed)
  knots = np.unique(np.concatenate(([0.0, overlap_end], times[times <= overlap_end],
                                    (times - elapsed)[(times >= elapsed) & (times - elapsed <= overlap_end)])))
  old_acceleration = np.interp(knots + elapsed, times, donor.states[:, 2])
  new_acceleration = np.interp(knots, times, states[:, 2])
  differences = new_acceleration - old_acceleration
  details = dict(donor_max_accel_deviation=float(np.max(np.abs(differences))),
                  donor_max_braking_relaxation=float(np.max(np.where(old_acceleration < 0.0, differences, 0.0), initial=0.0)))
  # Final wire state/jerk fields are Float64. Prove the same common jerk for
  # all four initial v/a corners; do not clip a corner or re-solve it. The
  # consumer additionally needs a continuous history/position-error bound:
  # |e_a(t)|<=A, |e_v(0)|<=V imply |e_x(t)|<=V*t+A*t^2/2. A last-sample
  # check alone does not establish this history premise.
  checks, last_reason = 0, ''
  for speed_tolerance, accel_tolerance in STATE_ENVELOPE_LEVELS:
    for speed_sign, accel_sign in ((1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)):
      corner = problem.x0.copy()
      corner[1] += speed_sign * speed_tolerance
      corner[2] += accel_sign * accel_tolerance
      corner_problem = replace(problem, x0=frozen_array(corner))
      corner_states = np.empty_like(states)
      corner_states[0] = corner
      for stage, duration in enumerate(np.diff(times)):
        corner_states[stage+1] = integrate(corner_states[stage], jerks[stage], float(duration))
      checks += 1
      if (last_reason := validate(corner_problem, corner_states, jerks)):
        break
    else:
      return ContinuationResult(True, 'validated_previous_trajectory', frozen_array(states), frozen_array(jerks), tail,
                                 **common, speed_tolerance=speed_tolerance, accel_tolerance=accel_tolerance,
                                 corner_checks=checks, **details)
  return reject('no_certified_state_envelope_' + last_reason, tail, corner_checks=checks, **details)
