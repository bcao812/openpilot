"""A bounded, separately identified continuation of an accepted moving plan.

This protocol never changes the failed native plan's validity or solver status.
The producer rechecks current risks; the consumer independently checks identity,
time, current state and its existing actuation authority on every control tick.
"""
import math
import struct
from bisect import bisect_right
from collections import deque
from dataclasses import dataclass

from opendbc.car.honda.values import CarControllerParams
from opendbc.car.structs import car
from openpilot.selfdrive.controls.lib.drive_helpers import CLOSE_LEAD_STOP_SPEED
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.continuation import STALE_MEASUREMENT_STATUS
from openpilot.selfdrive.controls.lib.reference_timing import (
  REFERENCE_MAX_AGE_NS, REFERENCE_TIMING_VERSION, ReferenceTiming, finite_state, valid_timestamp,
)
from openpilot.selfdrive.modeld.constants import index_function


CONTINUATION_VERSION = 1
# Only a native numerical failure may use a revalidated successful trajectory.
# Retired occupancy (-6) and stale-measurement (-7) offers must not regain
# execution authority after their old-object producer paths have been removed.
CONTINUATION_STATUSES = (4,)
CONTINUATION_EPISODE_NS = 100_000_000
CONTINUATION_DONOR_SOURCE_NS = 200_000_000
NATIVE_RECOVERY_NS = 500_000_000
NATIVE_SUCCESS_GAP_NS = 100_000_000
UNSTABLE_DISABLE_NS = 3_000_000_000
INACTIVE_NATIVE_RECOVERY_REASON = 'inactive_native_recovery'
SPEED_TOLERANCE_MAX = 0.2
ACCEL_TOLERANCE_MAX = 0.25
CONTINUATION_N = 12
CONTINUATION_TIMES = tuple(index_function(index, max_val=10.0, max_idx=CONTINUATION_N) for index in range(CONTINUATION_N + 1))
HISTORY_GAP_NS = 20_000_000
HISTORY_TAIL_NS = 10_000_000


def inactive_native_recovery(controls, control):
  """An OFF consumer's native-input receipt, with no execution authority.

  Callers must additionally check transport, time and the exact plan/reference
  identity. Standby and an accepted continuation never publish this receipt.
  """
  motion = controls.motionState
  off = car.CarControl.Actuators.LongControlState.off
  return (motion.continuationVersion == CONTINUATION_VERSION
          and motion.continuationReason == INACTIVE_NATIVE_RECOVERY_REASON
          and motion.phase == 'off' and controls.longControlState == off
          and not (motion.degraded or motion.waitingForFreshInput or motion.resolvedStop or motion.reasons or motion.continuationActive)
          and not control.longActive and control.actuators.longControlState == off
          and control.actuators.accel == 0.0 and not control.cruiseControl.resume)


@dataclass(frozen=True)
class ContinuationCandidate:
  target_accel: float
  speeds: tuple[float, ...]
  accels: tuple[float, ...]
  jerks: tuple[float, ...]
  speed_tolerance: float = 0.0
  accel_tolerance: float = 0.0


def braking_only_continuation(offer):
  """The stale-measurement path cannot offer positive future acceleration.

  The measured initial acceleration remains untouched. All later knots and
  the sampled action must be nonpositive; the ordinary candidate validation
  independently checks shape, dynamics, finite values and state tolerances.
  """
  return (isinstance(offer, ContinuationOffer) and valid_candidate(offer.candidate)
          and offer.candidate.target_accel <= 0.0
          and all(value <= 0.0 for value in offer.candidate.accels[1:]))


@dataclass(frozen=True)
class ContinuationOffer:
  episode_start_ns: int
  donor_producer_ns: int
  donor_source_ns: int
  donor_model_ns: int
  valid_until_ns: int
  candidate: ContinuationCandidate
  version: int = CONTINUATION_VERSION
  valid: bool = True
  source_cruise_kph: float = 0.0

  @classmethod
  def from_message(cls, message):
    try:
      candidate = message.candidate
      return cls(message.episodeStartMonoTime, message.donorProducerMonoTime, message.donorSourceMonoTime,
                 message.donorModelMonoTime, message.validUntilMonoTime,
                 ContinuationCandidate(candidate.targetAccel, tuple(candidate.speeds), tuple(candidate.accels), tuple(candidate.jerks),
                                       candidate.speedTolerance, candidate.accelTolerance),
                 message.version, message.valid, message.sourceCruiseKph)
    except (AttributeError, TypeError, ValueError, RuntimeError):
      return None

  def to_dict(self):
    return {
      'version': self.version, 'valid': self.valid, 'episodeStartMonoTime': self.episode_start_ns,
      'donorProducerMonoTime': self.donor_producer_ns, 'donorSourceMonoTime': self.donor_source_ns,
      'donorModelMonoTime': self.donor_model_ns, 'validUntilMonoTime': self.valid_until_ns,
      'sourceCruiseKph': self.source_cruise_kph,
      'candidate': {'targetAccel': self.candidate.target_accel, 'speeds': self.candidate.speeds,
                    'accels': self.candidate.accels, 'jerks': self.candidate.jerks,
                    'speedTolerance': self.candidate.speed_tolerance, 'accelTolerance': self.candidate.accel_tolerance},
    }


@dataclass(frozen=True)
class ContinuationDonor:
  producer_ns: int
  source_ns: int
  model_ns: int
  instance_ns: int

  @classmethod
  def from_reference(cls, reference, model_ns):
    return cls(reference.producer_ns, reference.source_ns, model_ns, reference.instance_ns)

  def matches(self, offer, reference):
    return (self.producer_ns == offer.donor_producer_ns and self.source_ns == offer.donor_source_ns
            and self.model_ns == offer.donor_model_ns and self.instance_ns == reference.instance_ns)


class ContinuationBudget:
  """One failure episode survives isolated successes and repeated publications.

  Only 500 ms of continuously advancing native successes retire an episode.
  Calling note_success repeatedly for one publication cannot accumulate credit.
  Return values from note_success indicate that an episode was cleared, not
  that a payload acquired execution authority.
  """
  def __init__(self):
    self.episode_start_ns = 0
    self._last_now_ns = 0
    self._recovery_start_ns = 0
    self._last_success_ns = 0
    self._last_success_ids = None

  def _clock(self, now_ns):
    if not valid_timestamp(now_ns) or now_ns < self._last_now_ns:
      self._recovery_start_ns = self._last_success_ns = 0
      self._last_success_ids = None
      return False
    self._last_now_ns = int(now_ns)
    return True

  def interrupt_recovery(self):
    """Break consecutive health evidence without renewing the failure deadline."""
    self._recovery_start_ns = self._last_success_ns = 0
    self._last_success_ids = None

  def note_failure(self, now_ns, *, episode_start_ns=None):
    if not self._clock(now_ns):
      return False
    self._recovery_start_ns = self._last_success_ns = 0
    self._last_success_ids = None
    if not self.episode_start_ns:
      self.episode_start_ns = int(now_ns)
    if episode_start_ns is not None:
      if not valid_timestamp(episode_start_ns) or episode_start_ns > now_ns:
        return False
      self.episode_start_ns = min(self.episode_start_ns, int(episode_start_ns))
    return True

  def note_success(self, now_ns, *, producer_ns, source_ns, model_ns):
    if not self._clock(now_ns):
      return False
    values = producer_ns, source_ns, model_ns
    if (not all(valid_timestamp(value) for value in values)
        or not max(source_ns, model_ns) <= producer_ns <= now_ns
        or now_ns - min(source_ns, model_ns) > REFERENCE_MAX_AGE_NS):
      self._recovery_start_ns = self._last_success_ns = 0
      self._last_success_ids = None
      return False
    old = self._last_success_ids
    if old is not None and not all(value > previous for value, previous in zip(values, old, strict=True)):
      if values != old or now_ns - self._last_success_ns > NATIVE_SUCCESS_GAP_NS:
        self._recovery_start_ns = self._last_success_ns = 0
        self._last_success_ids = None
      return False
    if not self._recovery_start_ns or now_ns - self._last_success_ns > NATIVE_SUCCESS_GAP_NS:
      self._recovery_start_ns = int(now_ns)
    self._last_success_ns, self._last_success_ids = int(now_ns), values
    if self.episode_start_ns and now_ns - self._recovery_start_ns >= NATIVE_RECOVERY_NS:
      self.episode_start_ns = 0
      return True
    return False

  def deadline(self, *, donor_producer_ns, donor_source_ns, donor_model_ns):
    if (not self.episode_start_ns
        or not all(valid_timestamp(value) for value in (donor_producer_ns, donor_source_ns, donor_model_ns))
        or max(donor_source_ns, donor_model_ns) > donor_producer_ns):
      return 0
    return min(self.episode_start_ns + CONTINUATION_EPISODE_NS,
               donor_source_ns + CONTINUATION_DONOR_SOURCE_NS,
               donor_producer_ns + REFERENCE_MAX_AGE_NS, donor_model_ns + REFERENCE_MAX_AGE_NS)

  def allows(self, now_ns, **donor):
    return bool(valid_timestamp(now_ns) and self.episode_start_ns <= now_ns < self.deadline(**donor)
                and now_ns >= self._last_now_ns)


def continuation_reference(message, *, plan_ns, now_ns):
  """Stateless reference checks shared by the offer observer and real owner.

  The owner additionally applies its existing stateful ReferenceTimingConsumer.
  """
  try:
    if message.timingVersion != REFERENCE_TIMING_VERSION or message.timingKnown is not True:
      return None
    reference = ReferenceTiming.from_message(message)
  except (AttributeError, TypeError, ValueError, RuntimeError):
    return None
  if (not all(valid_timestamp(value) for value in (plan_ns, now_ns, reference.source_ns, reference.epoch_ns,
                                                  reference.producer_ns, reference.instance_ns))
      or not reference.source_ns == reference.epoch_ns <= reference.producer_ns <= plan_ns <= now_ns
      or reference.instance_ns > reference.producer_ns or now_ns - reference.source_ns > REFERENCE_MAX_AGE_NS
      or reference.initial_known is not True or not finite_state(reference.state_identity) or reference.initial_speed < 0.0):
    return None
  return reference


def candidate_state(candidate, elapsed):
  """Propagate on the native piecewise-constant-jerk grid, never a UI grid."""
  index = min(max(bisect_right(CONTINUATION_TIMES, elapsed) - 1, 0), CONTINUATION_N - 1)
  offset = elapsed - CONTINUATION_TIMES[index]
  speed = candidate.speeds[index] + candidate.accels[index] * offset + 0.5 * candidate.jerks[index] * offset * offset
  accel = candidate.accels[index] + candidate.jerks[index] * offset
  return speed, accel


def execution_target(offer, reference, now_ns, actuator_delay):
  if (not isinstance(offer, ContinuationOffer) or offer.valid is not True or not isinstance(reference, ReferenceTiming)
      or not valid_candidate(offer.candidate) or not valid_timestamp(now_ns)
      or not finite_state((actuator_delay,)) or not 0.0 <= actuator_delay <= 1.0):
    return None
  # Match the producer's actuator preview, but advance along the immutable
  # native curve with the actual consumption clock on every tick.
  action_time = (now_ns - reference.epoch_ns) / 1e9 + actuator_delay + 0.05
  if not 0.0 <= action_time <= CONTINUATION_TIMES[-1]:
    return None
  return min(offer.candidate.target_accel, candidate_state(offer.candidate, action_time)[1])


def _float32(value):
  try:
    return struct.unpack('f', struct.pack('f', value))[0]
  except (OverflowError, TypeError, ValueError, struct.error):
    return math.nan


def valid_candidate(candidate):
  try:
    if (not isinstance(candidate, ContinuationCandidate)
        or len(candidate.speeds) != CONTINUATION_N + 1 or len(candidate.accels) != CONTINUATION_N + 1
        or len(candidate.jerks) != CONTINUATION_N
        or not finite_state((candidate.target_accel, candidate.speed_tolerance, candidate.accel_tolerance,
                             *candidate.speeds, *candidate.accels, *candidate.jerks))
        or not 0.0 < candidate.speed_tolerance <= SPEED_TOLERANCE_MAX
        or not 0.0 < candidate.accel_tolerance <= ACCEL_TOLERANCE_MAX
        or any(value < 0.0 for value in candidate.speeds)
        or not CarControllerParams.BOSCH_ACCEL_MIN <= candidate.target_accel <= CarControllerParams.BOSCH_ACCEL_MAX
        or any(not CarControllerParams.BOSCH_ACCEL_MIN <= value <= CarControllerParams.BOSCH_ACCEL_MAX for value in candidate.accels)):
      return False
    for index, (left, right) in enumerate(zip(CONTINUATION_TIMES, CONTINUATION_TIMES[1:])):
      duration = right - left
      speed = candidate.speeds[index] + candidate.accels[index] * duration + 0.5 * candidate.jerks[index] * duration * duration
      accel = candidate.accels[index] + candidate.jerks[index] * duration
      if abs(speed - candidate.speeds[index + 1]) > 1e-8 or abs(accel - candidate.accels[index + 1]) > 1e-8:
        return False
    return True
  except (TypeError, ValueError, OverflowError):
    return False


class ContinuationHistory:
  """Bounded observed-state coverage, with real carState publication clocks.

  The timestamps remain observation proxies. A missing history interval cannot
  be certified by a matching current endpoint. The final at-most-10-ms gap uses
  explicit constant-acceleration propagation and the same certified error tube.
  Between observed points, v/a use a piecewise linear observation model; each
  velocity segment's slope is separately checked against the native acceleration.
  This bounds that model, not unobserved physical motion between vehicle samples.
  """
  def __init__(self):
    self.samples = deque(maxlen=64)

  def observe(self, sample_ns, now_ns, speed, accel):
    if (not valid_timestamp(sample_ns) or not valid_timestamp(now_ns) or sample_ns > now_ns
        or now_ns - sample_ns > HISTORY_GAP_NS or not finite_state((speed, accel)) or speed < 0.0):
      return False
    sample = int(sample_ns), float(speed), float(accel)
    if self.samples and sample_ns <= self.samples[-1][0]:
      return sample == self.samples[-1]
    self.samples.append(sample)
    return True

  def current_state(self, now_ns):
    if not self.samples or not valid_timestamp(now_ns) or not 0 <= now_ns - self.samples[-1][0] <= HISTORY_TAIL_NS:
      return None
    stamp, speed, accel = self.samples[-1]
    return speed + accel * ((now_ns - stamp) / 1e9), accel

  def check(self, candidate, reference, now_ns):
    current = self.current_state(now_ns)
    if current is None or not self.samples or self.samples[0][0] > reference.epoch_ns or self.samples[-1][0] < reference.epoch_ns:
      return 'continuation_history_missing'
    samples = list(self.samples)
    if samples[-1][0] < now_ns:
      samples.append((now_ns, *current))
    epoch = reference.epoch_ns
    v_bound, a_bound = candidate.speed_tolerance, candidate.accel_tolerance
    lower_progress = upper_progress = 0.0
    last_endpoint = epoch
    for first, last in zip(samples, samples[1:]):
      first_ns, first_v, first_a = first
      last_ns, last_v, last_a = last
      if last_ns <= epoch or first_ns >= now_ns:
        continue
      if last_ns - first_ns > HISTORY_GAP_NS:
        return 'continuation_history_gap'
      start_ns, end_ns = max(first_ns, epoch), min(last_ns, now_ns)
      if start_ns != last_endpoint:
        return 'continuation_history_gap'
      start, end = (start_ns - epoch) / 1e9, (end_ns - epoch) / 1e9
      knots = [start, *(value for value in CONTINUATION_TIMES if start < value < end), end]
      span = (last_ns - first_ns) / 1e9

      def error_at(elapsed):
        fraction = ((epoch - first_ns) / 1e9 + elapsed) / span
        observed_v = first_v + fraction * (last_v - first_v)
        observed_a = first_a + fraction * (last_a - first_a)
        nominal_v, nominal_a = candidate_state(candidate, elapsed)
        return observed_v - nominal_v, observed_a - nominal_a, nominal_a

      for left, right in zip(knots, knots[1:]):
        e0, a0, nominal_a0 = error_at(left)
        e1, a1, nominal_a1 = error_at(right)
        duration = right - left
        observed_slope = (last_v - first_v) / span
        if (max(abs(a0), abs(a1), abs(observed_slope - nominal_a0), abs(observed_slope - nominal_a1)) > a_bound
            or abs(e0) > v_bound + a_bound * left or abs(e1) > v_bound + a_bound * right
            or abs(e1 - e0) > a_bound * duration):
          return 'continuation_history_state_mismatch'
        # Tight integral enclosure for endpoint velocity errors and a
        # Lipschitz acceleration-error bound; not a bare trapezoidal estimate.
        extra = max(0.0, a_bound * duration * duration / 4.0 - (e1 - e0) ** 2 / (4.0 * a_bound))
        middle = duration * (e0 + e1) / 2.0
        lower_progress += middle - extra
        upper_progress += middle + extra
        bound = v_bound * right + 0.5 * a_bound * right * right
        if lower_progress < -bound or upper_progress > bound:
          return 'continuation_progress_mismatch'
      last_endpoint = end_ns
    return '' if last_endpoint == now_ns else 'continuation_history_missing'


def validate_continuation(offer, *, donor, reference, budget, plan_ns, model_ns, now_ns, speed, accel, cruise_kph):
  """Return a reason; only the empty string means this certificate is valid."""
  if (not isinstance(offer, ContinuationOffer) or type(offer.version) is not int or offer.version != CONTINUATION_VERSION
      or offer.valid is not True):
    return 'unsupported_continuation'
  if not finite_state((offer.source_cruise_kph, cruise_kph)) or not 0.0 < offer.source_cruise_kph <= cruise_kph:
    return 'continuation_cruise_reduced'
  if not isinstance(reference, ReferenceTiming) or not isinstance(donor, ContinuationDonor) or not donor.matches(offer, reference):
    return 'continuation_donor_mismatch'
  if reference.initial_speed < CLOSE_LEAD_STOP_SPEED:
    return 'continuation_initial_speed_too_low'
  if (not all(valid_timestamp(value) for value in (offer.episode_start_ns, offer.valid_until_ns, plan_ns, model_ns, now_ns))
      or not offer.episode_start_ns <= reference.producer_ns <= plan_ns <= now_ns
      or not offer.donor_producer_ns < reference.producer_ns or not offer.donor_source_ns < reference.source_ns
      or not offer.donor_model_ns <= model_ns <= reference.producer_ns
      or now_ns - model_ns > REFERENCE_MAX_AGE_NS):
    return 'continuation_time_invalid'
  maximum_deadline = budget.deadline(donor_producer_ns=offer.donor_producer_ns, donor_source_ns=offer.donor_source_ns,
                                     donor_model_ns=offer.donor_model_ns)
  if (offer.episode_start_ns != budget.episode_start_ns or not now_ns < offer.valid_until_ns <= maximum_deadline
      or now_ns < budget._last_now_ns):
    return 'continuation_expired'
  candidate = offer.candidate
  if not valid_candidate(candidate) or not finite_state((speed, accel)) or speed < CLOSE_LEAD_STOP_SPEED:
    return 'continuation_candidate_invalid'
  # The reference fields use Float32; this checks the same published initial
  # identity. It grants no extra physical state-error allowance.
  if _float32(candidate.speeds[0]) != reference.initial_speed or _float32(candidate.accels[0]) != reference.initial_accel:
    return 'continuation_initial_state_mismatch'
  elapsed = (now_ns - reference.epoch_ns) / 1e9
  if not 0.0 <= elapsed <= CONTINUATION_TIMES[-1]:
    return 'continuation_reference_expired'
  nominal_speed, nominal_accel = candidate_state(candidate, elapsed)
  accel_error = accel - nominal_accel
  initial_speed_error = speed - nominal_speed - accel_error * elapsed
  if abs(initial_speed_error) > candidate.speed_tolerance or abs(accel_error) > candidate.accel_tolerance:
    return 'continuation_vehicle_state_mismatch'
  return ''
