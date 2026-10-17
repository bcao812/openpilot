"""Prove recovery from a burst without granting old model or plan authority.

The caller decides whether an already engaged longitudinal axis may use this
evidence. This observer only stores identities, never targets or commands. A
new fault immediately withdraws recovery; the original control-state fault
timer remains the caller's responsibility.
"""

import math
from collections import deque
from dataclasses import dataclass, replace

from openpilot.selfdrive.controls.lib.longitudinal_continuation import continuation_reference
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_MAX_AGE_NS, REFERENCE_TIMING_VERSION, valid_timestamp


CLEAN_WINDOW_NS = 1_000_000_000
MAX_MODEL_GAP_NS = 100_000_000
MIN_RECOVERY_FRAMES = 3
IDENTITY_HISTORY_SIZE = 8
SERVICE_LEASES = {'modelV2': MAX_MODEL_GAP_NS, 'deviceMotion': REFERENCE_MAX_AGE_NS,
                  'longitudinalPlan': REFERENCE_MAX_AGE_NS, 'controlsState': MAX_MODEL_GAP_NS}


@dataclass(frozen=True)
class ModelIdentity:
  published_ns: int
  capture_ns: int
  frame_id: int


@dataclass(frozen=True)
class PlanIdentity:
  published_ns: int
  model_ns: int
  source_ns: int
  producer_ns: int
  instance_ns: int
  capture_ns: int = 0

  @property
  def oldest_ns(self):
    return min(self.model_ns, self.source_ns, self.capture_ns)


def fresh(sm, service, now_ns, lease_ns):
  stamp = sm.logMonoTime.get(service)
  return (all(getattr(sm, flag).get(service) is True for flag in ('seen', 'valid', 'alive', 'freq_ok'))
          and valid_timestamp(stamp) and 0 <= now_ns - stamp <= lease_ns)


class ModelLagRecovery:
  def __init__(self):
    self.reset()

  def reset(self):
    self.recovered = False
    self.clean_since_ns = 0
    self.good_frames = 0
    self._last_now_ns = 0
    self._fault_ns = 0
    self._service_times = {}
    self._last_model = None
    self._last_drop_perc = 0.0
    self._last_plan = None
    self._last_consumed = None
    self._models = deque(maxlen=IDENTITY_HISTORY_SIZE)
    self._plans = deque(maxlen=IDENTITY_HISTORY_SIZE)

  def _fault(self, now_ns):
    self.recovered = False
    self.clean_since_ns = 0
    self.good_frames = 0
    self._fault_ns = max(self._fault_ns, now_ns)
    self._models.clear()
    self._plans.clear()
    self._last_consumed = None
    return False

  def update(self, sm, now_ns, *, lagging: bool) -> bool:
    if lagging is not True:
      self.reset()
      return False
    if not valid_timestamp(now_ns):
      return self._fault(self._last_now_ns)
    previous_ns = self._last_now_ns
    self._last_now_ns = max(previous_ns, now_ns)
    if previous_ns and not 0 < now_ns - previous_ns <= MAX_MODEL_GAP_NS:
      return self._fault(self._last_now_ns)
    if not previous_ns:
      self._fault(now_ns)
    try:
      self.recovered = self._update(sm, now_ns)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError, RuntimeError):
      # Missing or malformed evidence cannot create an exception to a fault.
      return self._fault(now_ns)
    return self.recovered

  def _update(self, sm, now_ns):
    if not all(fresh(sm, name, now_ns, lease) for name, lease in SERVICE_LEASES.items()):
      return self._fault(now_ns)
    stamps = {name: sm.logMonoTime[name] for name in SERVICE_LEASES}
    if any(stamp < self._service_times.get(name, 0) for name, stamp in stamps.items()):
      return self._fault(now_ns)
    self._service_times = stamps
    device = sm['deviceMotion']
    motion = sm['controlsState'].motionState
    if (not all(value is True for value in (device.inputsOK, device.sensorsOK, device.posenetOK,
                                           motion.enabled, motion.referenceTimingValid))
        or motion.version != MOTION_INTENT_VERSION or motion.planningMode != int(LongitudinalMode.UNIFIED)
        or motion.referenceTimingVersion != REFERENCE_TIMING_VERSION
        or not all(value is False for value in (motion.degraded, motion.waitingForFreshInput, motion.continuationActive))):
      return self._fault(now_ns)

    model = sm['modelV2']
    current = ModelIdentity(stamps['modelV2'], model.timestampEof, model.frameId)
    percent = model.frameDropPerc
    if (not valid_timestamp(current.capture_ns) or not 0 <= now_ns - current.capture_ns <= REFERENCE_MAX_AGE_NS
        or current.capture_ns > current.published_ns or type(current.frame_id) is not int or not 0 <= current.frame_id < 2**32
        or type(percent) not in (float, int) or not math.isfinite(percent) or not 0.0 <= percent <= 100.0):
      return self._fault(now_ns)
    previous, old_percent = self._last_model, self._last_drop_perc
    self._last_model, self._last_drop_perc = current, percent
    new_frame = previous is None or current.published_ns != previous.published_ns
    if previous is not None:
      if not new_frame:
        if current != previous or percent != old_percent:
          return self._fault(now_ns)
      elif (current.frame_id != previous.frame_id + 1 or not 0 < current.capture_ns - previous.capture_ns <= MAX_MODEL_GAP_NS
            or not 0 < current.published_ns - previous.published_ns <= MAX_MODEL_GAP_NS or percent > old_percent):
        return self._fault(now_ns)

    plan = sm['longitudinalPlan']
    reference = continuation_reference(plan.reference, plan_ns=stamps['longitudinalPlan'], now_ns=now_ns)
    intent = plan.motionIntent
    if (reference is None or type(plan.solverStatus) is not int or plan.solverStatus != 0
        or intent.valid is not True or intent.ready is not True
        or intent.version != MOTION_INTENT_VERSION or intent.mode != int(LongitudinalMode.UNIFIED)
        or intent.producerInstanceMonoTime != reference.instance_ns
        or not valid_timestamp(plan.modelMonoTime) or not 0 <= now_ns - plan.modelMonoTime <= REFERENCE_MAX_AGE_NS
        or plan.modelMonoTime > stamps['longitudinalPlan']):
      return self._fault(now_ns)
    identity = PlanIdentity(stamps['longitudinalPlan'], plan.modelMonoTime, reference.source_ns,
                            reference.producer_ns, reference.instance_ns)
    previous_plan, self._last_plan = self._last_plan, identity
    if previous_plan is not None:
      if (identity.published_ns < previous_plan.published_ns or identity.model_ns < previous_plan.model_ns
          or identity.source_ns < previous_plan.source_ns or identity.producer_ns < previous_plan.producer_ns
          or identity.instance_ns != previous_plan.instance_ns
          or (identity.published_ns == previous_plan.published_ns and identity != previous_plan)
          or (identity.producer_ns == previous_plan.producer_ns
              and (identity.model_ns, identity.source_ns) != (previous_plan.model_ns, previous_plan.source_ns))):
        return self._fault(now_ns)

    # Each original source expires on its own clock, even if a publisher sends
    # another message with a newer envelope or the observer runs at 100 Hz.
    while self._models and now_ns - self._models[0].capture_ns > REFERENCE_MAX_AGE_NS:
      self._models.popleft()
    while self._plans and now_ns - self._plans[0].oldest_ns > REFERENCE_MAX_AGE_NS:
      self._plans.popleft()
    if new_frame and current.capture_ns > self._fault_ns:
      self._models.append(current)
      self.good_frames += 1
      if not self.clean_since_ns:
        self.clean_since_ns = now_ns
    observed_model = next((item for item in self._models if item.published_ns == identity.model_ns), None)
    if observed_model is not None and identity.source_ns > self._fault_ns:
      cached = replace(identity, capture_ns=observed_model.capture_ns)
      if not any(item.published_ns == cached.published_ns for item in self._plans):
        self._plans.append(cached)

    confirmed = self._consumer_confirmed(motion, stamps['controlsState'], now_ns)
    return bool(confirmed and self.clean_since_ns and self.good_frames >= MIN_RECOVERY_FRAMES
                and now_ns - self.clean_since_ns >= CLEAN_WINDOW_NS)

  def _consumer_confirmed(self, motion, controls_ns, now_ns):
    times = (motion.planMonoTime, motion.modelMonoTime, motion.referenceSourceMonoTime, motion.referenceEpochMonoTime,
             motion.referenceProducerMonoTime, motion.referenceProducerInstanceMonoTime, motion.referenceConsumerMonoTime)
    if (not all(valid_timestamp(stamp) for stamp in times)
        or not motion.referenceSourceMonoTime == motion.referenceEpochMonoTime <= motion.referenceProducerMonoTime
        or not motion.referenceProducerInstanceMonoTime <= motion.referenceProducerMonoTime <= motion.planMonoTime
        or not motion.modelMonoTime <= motion.planMonoTime <= motion.referenceConsumerMonoTime <= controls_ns
        or now_ns - min(motion.modelMonoTime, motion.referenceSourceMonoTime) > REFERENCE_MAX_AGE_NS):
      return self._fault(now_ns)
    record = next((item for item in self._plans if (
      item.published_ns, item.model_ns, item.source_ns, item.producer_ns, item.instance_ns
    ) == (motion.planMonoTime, motion.modelMonoTime, motion.referenceSourceMonoTime,
          motion.referenceProducerMonoTime, motion.referenceProducerInstanceMonoTime)), None)
    if record is None:
      # New messages and their consumer receipt may arrive on different ticks.
      # An unknown/pre-fault receipt grants nothing, but does not restart an
      # otherwise clean source window while the matching identity is in flight.
      return False
    if now_ns - record.oldest_ns > REFERENCE_MAX_AGE_NS:
      return self._fault(now_ns)
    previous = self._last_consumed
    consumer_ns = motion.referenceConsumerMonoTime
    self._last_consumed = (record, controls_ns, consumer_ns)
    if previous is not None:
      prior, prior_controls_ns, prior_consumer_ns = previous
      if (consumer_ns < prior_consumer_ns or record.published_ns < prior.published_ns or record.model_ns < prior.model_ns
          or record.source_ns < prior.source_ns or record.producer_ns < prior.producer_ns or record.instance_ns != prior.instance_ns
          or controls_ns == prior_controls_ns and (record != prior or consumer_ns != prior_consumer_ns)):
        return self._fault(now_ns)
    return True
