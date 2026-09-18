"""Identity and age of the measured initial state carried by a UNIFIED plan.

carState Event time is an observation-time proxy, not an ECU sample time. A
plan's Event time is its availability time; publishing again cannot renew the
source lease. All timestamps remain integer monotonic nanoseconds.
"""

import math
from dataclasses import dataclass, replace
from numbers import Integral, Real


REFERENCE_TIMING_VERSION = 1
REFERENCE_MAX_AGE_NS = 250_000_000


def valid_timestamp(value):
  return isinstance(value, Integral) and not isinstance(value, bool) and 0 < value < (1 << 64)


def finite_state(values):
  return all(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) for value in values)


@dataclass(frozen=True)
class ReferenceTiming:
  source_ns: int
  epoch_ns: int
  producer_ns: int
  instance_ns: int
  source_speed: float
  source_accel: float
  initial_speed: float = 0.0
  initial_accel: float = 0.0
  initial_known: bool = False

  @property
  def state_identity(self):
    return self.source_speed, self.source_accel, self.initial_speed, self.initial_accel

  @classmethod
  def from_message(cls, message):
    return cls(message.sourceCarStateMonoTime, message.referenceMonoTime, message.producerMonoTime,
               message.producerInstanceMonoTime, message.sourceVEgo, message.sourceAEgo,
               message.initialVEgo, message.initialAEgo, message.initialStateKnown)


class ReferenceTimingProducer:
  """One planner instance; failed input never advances accepted source identity.

The first qualified consumer timestamp identifies this instance without random
state or persistent Params. A restarted producer receives a later anchor. The
consumer retains its accepted source/instance high water across that restart.
"""

  def __init__(self, max_age_ns=REFERENCE_MAX_AGE_NS):
    self.max_age_ns = max_age_ns
    self.instance_ns = 0
    self.last_consumer_ns = 0
    self.last_source = None
    self.pending = None

  def qualify_source(self, source_ns, speed, accel, now_ns):
    self.pending = None
    if not valid_timestamp(now_ns) or now_ns <= self.last_consumer_ns:
      return None
    source_time_valid = valid_timestamp(source_ns)
    clock_near_source = source_time_valid and 0 <= now_ns - source_ns <= self.max_age_ns
    clock_near_previous = self.last_consumer_ns > 0 and now_ns - self.last_consumer_ns <= self.max_age_ns
    first_unavailable_source = self.last_consumer_ns == 0 and (not source_time_valid or source_ns > now_ns)
    if not (clock_near_source or clock_near_previous or first_unavailable_source):
      return None
    # Qualify the local clock independently of source identity. A plausible
    # consumer followed by malformed/future/mutated source still happened;
    # rejecting that source must not hide a later backwards consumer time.
    self.last_consumer_ns = int(now_ns)
    if not source_time_valid:
      return None
    if not source_ns <= now_ns or now_ns - source_ns > self.max_age_ns or not finite_state((speed, accel)):
      return None
    source_ns, now_ns = int(source_ns), int(now_ns)
    values = float(speed), float(accel)
    if self.last_source is not None:
      previous_ns, previous_values = self.last_source
      if source_ns < previous_ns or source_ns == previous_ns and values != previous_values:
        return None
    self.last_source = source_ns, values
    if self.instance_ns == 0:
      self.instance_ns = now_ns
    self.pending = ReferenceTiming(source_ns, source_ns, now_ns, self.instance_ns, *values)
    return source_ns

  def set_initial_state(self, speed, accel):
    if self.pending is not None and finite_state((speed, accel)) and speed >= 0.0:
      self.pending = replace(self.pending, initial_speed=float(speed), initial_accel=float(accel), initial_known=True)

  def publish(self, message):
    message.timingVersion = REFERENCE_TIMING_VERSION
    message.timingKnown = self.pending is not None
    if self.pending is None:
      return
    record = self.pending
    message.sourceCarStateMonoTime = record.source_ns
    message.referenceMonoTime = record.epoch_ns
    message.producerMonoTime = record.producer_ns
    message.producerInstanceMonoTime = record.instance_ns
    message.sourceVEgo, message.sourceAEgo = record.source_speed, record.source_accel
    message.initialVEgo, message.initialAEgo = record.initial_speed, record.initial_accel
    message.initialStateKnown = record.initial_known


@dataclass(frozen=True)
class ReferenceTimingCheck:
  valid: bool
  reason: str
  now_ns: int
  clock_valid: bool
  timing: ReferenceTiming | None = None
  source_advanced: bool = False


class ReferenceTimingConsumer:
  """Validate serialized provenance before the existing motion owner uses it.

Only accepted records advance source and producer high water. A clock sample
must also be causal and within the existing availability lease, so a future
clock paired with an old message cannot poison the next healthy consumption.
No state is shared with another Controls instance or restored across reboot.
"""

  def __init__(self, max_age_ns=REFERENCE_MAX_AGE_NS):
    self.max_age_ns = max_age_ns
    self.last_consumer_ns = 0
    self.last_timing = None
    self.last_available_ns = 0
    self.last_command_identity = None

  def consume(self, message, available_ns, now_ns, *, command_identity=()):
    now_valid = valid_timestamp(now_ns)
    safe_now = int(now_ns) if now_valid else 0
    clock_valid = bool(now_valid and now_ns > self.last_consumer_ns and valid_timestamp(available_ns)
                       and available_ns <= now_ns and now_ns - available_ns <= self.max_age_ns)

    def rejected(reason, timing=None):
      return ReferenceTimingCheck(False, reason, safe_now, clock_valid, timing)

    if not clock_valid:
      return rejected('invalid_consumer_or_availability_time')
    # This is a plausible local clock sample, independent of source metadata.
    self.last_consumer_ns = int(now_ns)
    if message.timingVersion != REFERENCE_TIMING_VERSION or message.timingKnown is not True:
      return rejected('unknown_reference_timing')
    record = ReferenceTiming.from_message(message)
    if not all(valid_timestamp(value) for value in (record.source_ns, record.epoch_ns, record.producer_ns, record.instance_ns)):
      return rejected('invalid_reference_time', record)
    if not (record.source_ns == record.epoch_ns and record.source_ns <= record.producer_ns <= available_ns
            and record.instance_ns <= record.producer_ns):
      return rejected('noncausal_reference_time', record)
    if now_ns - record.source_ns > self.max_age_ns:
      return rejected('expired_reference_source', record)
    if record.initial_known is not True or not finite_state(record.state_identity) or record.initial_speed < 0.0:
      return rejected('unknown_reference_initial_state', record)

    previous = self.last_timing
    if previous is not None:
      if record.source_ns < previous.source_ns or record.producer_ns < previous.producer_ns or available_ns < self.last_available_ns:
        return rejected('reordered_reference', record)
      if record.source_ns == previous.source_ns and record.state_identity != previous.state_identity:
        return rejected('changed_source_identity', record)
      if record.instance_ns < previous.instance_ns:
        return rejected('retired_producer_instance', record)
      if record.instance_ns != previous.instance_ns:
        if not (record.source_ns > previous.source_ns and record.producer_ns > previous.producer_ns
                and available_ns > self.last_available_ns):
          return rejected('restart_requires_new_source', record)
      if record.producer_ns == previous.producer_ns or available_ns == self.last_available_ns:
        if record != previous or command_identity != self.last_command_identity:
          return rejected('changed_producer_identity', record)

    source_advanced = previous is None or record.source_ns > previous.source_ns
    self.last_timing, self.last_available_ns = record, int(available_ns)
    self.last_command_identity = command_identity
    return ReferenceTimingCheck(True, 'accepted' if source_advanced else 'reused_source', safe_now, True, record, source_advanced)
