"""Producer side of a short, independently checked trajectory continuation.

The failed native plan is never rewritten.  A donor is only an offer until the
motion owner acknowledges executing that exact successful reference.  Current
objects and constraints must still admit a newly integrated numerical witness.
"""
from dataclasses import dataclass
import math
import time

import numpy as np

from openpilot.selfdrive.controls.lib.drive_helpers import CLOSE_LEAD_STOP_SPEED
from openpilot.selfdrive.controls.lib.lead_qualification import control_leads, lead_identity, qualified_lead
from openpilot.selfdrive.controls.lib.longitudinal_continuation import (
  CONTINUATION_STATUSES, CONTINUATION_VERSION,
  ContinuationBudget, ContinuationCandidate, ContinuationDonor as NativeIdentity, ContinuationOffer, inactive_native_recovery,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_TIMING_VERSION, valid_timestamp


@dataclass(frozen=True)
class _Donor:
  trajectory: object
  producer_ns: int
  source_ns: int
  model_ns: int
  instance_ns: int
  scope: tuple
  objects: frozenset | None
  object_purposes: frozenset


class PlannerContinuation:
  def __init__(self):
    self.budget = ContinuationBudget()
    self.donor = None
    self.native_receipt = None
    self.offer = None
    self.reason = 'no_native_donor'
    self.independent_braking_accel = None
    self.tail_extension_s = 0.0
    self.validation_time = 0.0

  def begin(self):
    self.offer = None
    self.reason = 'not_requested'
    self.independent_braking_accel = None
    self.tail_extension_s = 0.0
    self.validation_time = 0.0

  def invalidate(self, reason, *, now_ns=0, keep_offer=False, failure=True):
    self.native_receipt = None
    if failure and valid_timestamp(now_ns):
      self.budget.note_failure(now_ns)
    else:
      self.budget.interrupt_recovery()
    if not keep_offer:
      self.donor = None
      self.offer = None
      self.reason = reason

  @staticmethod
  def object_support(radar, now_ns):
    """Snapshot the physical objects and purposes supporting a native donor.

    Current numerical validation still includes every admitted object.  This
    deliberately restrictive first version also requires stable identities;
    a visual slot number alone cannot establish physical continuity. The
    donor retains its own immutable follow/brake/stop support even after a
    failed solve has cleared the MPC's reference-history bookkeeping.
    """
    identities, purposes = set(), set()
    for lead in control_leads(radar, 'brake', 0.0, now_ns=now_ns):
      identity = lead_identity(lead)
      if identity is None:
        return None
      identities.add(identity)
      purposes.update((identity, purpose) for purpose in ('brake', 'follow', 'stop')
                      if qualified_lead(lead, purpose, 0.0, now_ns=now_ns))
    return frozenset(identities), frozenset(purposes)

  @staticmethod
  def receipt_available(state, now_ns, *, allow_inactive=False):
    motion = state['controlsState'].motionState
    car = state['carState']
    control = state['carControl']
    times = getattr(state, 'logMonoTime', {})
    source_times = times.get('controlsState', 0), times.get('carControl', 0)
    executing = (state['selfdriveState'].enabled and control.enabled and control.longActive
                 and not control.cruiseControl.override and not car.gasPressed and not car.brakePressed
                 and not state['controlsState'].forceDecel)
    recovering = (allow_inactive and inactive_native_recovery(state['controlsState'], control)
                  and state.all_checks(service_list=['controlsState', 'carControl', 'carState', 'modelV2']))
    return bool(
      getattr(motion, 'continuationVersion', 0) == CONTINUATION_VERSION
      and motion.enabled and not (motion.degraded or motion.waitingForFreshInput)
      and motion.referenceTimingVersion == REFERENCE_TIMING_VERSION and motion.referenceTimingValid
      and all(valid_timestamp(value) and 0 <= now_ns - value <= 100_000_000 for value in source_times)
      and valid_timestamp(motion.referenceConsumerMonoTime)
      and motion.referenceProducerMonoTime <= motion.referenceConsumerMonoTime <= min(source_times)
      and (executing or recovering)
    )

  @classmethod
  def owner_available(cls, state, now_ns):
    motion, car = state['controlsState'].motionState, state['carState']
    return bool(cls.receipt_available(state, now_ns)
                and motion.phase == 'track' and not motion.reasons and not motion.resolvedStop
                and not car.standstill and not car.cruiseState.standstill and car.vEgo >= CLOSE_LEAD_STOP_SPEED)

  def _owner_matches(self, state, now_ns):
    donor = self.donor
    if donor is None or not self.owner_available(state, now_ns):
      return False
    motion = state['controlsState'].motionState
    if motion.referenceProducerInstanceMonoTime != donor.instance_ns:
      return False
    if getattr(motion, 'continuationActive', False):
      return (motion.continuationDonorProducerMonoTime == donor.producer_ns
              and motion.continuationEpisodeMonoTime == self.budget.episode_start_ns
              and now_ns < motion.continuationValidUntilMonoTime)
    return (motion.referenceProducerMonoTime == donor.producer_ns
            and motion.referenceSourceMonoTime == donor.source_ns
            and motion.referenceEpochMonoTime == donor.source_ns
            and motion.modelMonoTime == donor.model_ns)

  def native_success(self, mpc, state, timing, *, now_ns, permitted, scope, applied_accel_ceiling):
    """Seal only after native, all postchecks and final arbitration succeeded."""
    motion, receipt = state['controlsState'].motionState, self.native_receipt
    receipt_healthy = self.receipt_available(state, now_ns, allow_inactive=True)
    if (receipt is not None and receipt_healthy
        and not getattr(motion, 'continuationActive', False)
        and motion.referenceProducerInstanceMonoTime == receipt.instance_ns
        and motion.referenceProducerMonoTime == receipt.producer_ns
        and motion.referenceSourceMonoTime == motion.referenceEpochMonoTime == receipt.source_ns
        and motion.modelMonoTime == receipt.model_ns):
      # An ordinary accepted stop or a selected cruise cap also proves a
      # healthy native cycle; donor eligibility is a separate, narrower test.
      self.budget.note_success(now_ns, producer_ns=receipt.producer_ns,
                               source_ns=receipt.source_ns, model_ns=receipt.model_ns)
    elif not receipt_healthy:
      self.budget.interrupt_recovery()
    model_ns = state.logMonoTime.get('modelV2', 0)
    self.native_receipt = (NativeIdentity.from_reference(timing, model_ns)
                           if timing is not None and valid_timestamp(model_ns) else None)
    if not permitted or timing is None or not self.owner_available(state, now_ns):
      self.donor = None
      return
    support = self.object_support(state['radarState'], now_ns)
    if (support is None or not valid_timestamp(model_ns) or model_ns > now_ns
        or timing.producer_ns != now_ns or timing.instance_ns <= 0):
      self.donor = None
      return
    trajectory = mpc.snapshot_continuation_donor(timing.epoch_ns, timing.producer_ns, timing.instance_ns,
                                                accel_ceiling=applied_accel_ceiling)
    if trajectory is None:
      self.donor = None
      return
    objects, purposes = support
    self.donor = _Donor(trajectory, timing.producer_ns, timing.epoch_ns, int(model_ns), timing.instance_ns,
                        scope, objects, purposes)

  def attempt(self, mpc, state, timing, *, now_ns, permitted, scope, action_t, independent_ceiling):
    self.budget.note_failure(now_ns)

    def reject(reason):
      self.reason = reason
      self.offer = None
      return None

    if mpc.solution_status not in CONTINUATION_STATUSES:
      return reject('unclassified_or_input_failure')
    if not permitted or timing is None:
      return reject('current_authority_or_risk')
    donor = self.donor
    if donor is None:
      return reject('no_native_donor')
    if (donor.scope != scope or donor.instance_ns != timing.instance_ns
        or not self._owner_matches(state, now_ns)):
      return reject('donor_not_currently_owned')
    support = self.object_support(state['radarState'], now_ns)
    if support is None or support[0] != donor.objects:
      return reject('object_identity_changed')
    if not donor.object_purposes <= support[1]:
      # Recheck on every attempt: an earlier accepted continuation can outlive
      # the MPC history that detected qualification changes on native ticks.
      return reject('object_qualification_reduced')
    donor_times = dict(donor_producer_ns=donor.producer_ns, donor_source_ns=donor.source_ns, donor_model_ns=donor.model_ns)
    if not self.budget.allows(now_ns, **donor_times):
      return reject('continuation_budget_exhausted')
    if mpc.rest_goal is not None or mpc.rest_goal_braking_accel is not None:
      return reject('current_stop_or_imminent_risk')
    started = time.perf_counter()
    result = mpc.continuation_candidate(donor.trajectory, now_ns=now_ns)
    self.validation_time = time.perf_counter() - started
    if not result.accepted:
      return reject(result.reason)
    target = float(np.interp(action_t, T_IDXS, result.states[:, 2]))
    prior_cap = donor.trajectory.applied_accel_ceiling
    if prior_cap is not None and target > prior_cap + 1e-9:
      return reject('initial_state_above_donor_execution_cap')
    ceiling = float(independent_ceiling(target))
    self.independent_braking_accel = ceiling if math.isfinite(ceiling) and ceiling < 0.0 else None
    # A tighter current cap is a new control requirement. Do not clip a
    # witness after its physical validation and pretend that it was rechecked.
    if not math.isfinite(ceiling) or target > ceiling + 1e-6:
      return reject('current_independent_constraint_tighter')
    deadline = self.budget.deadline(**donor_times)
    if result.evidence_expiry_ns:
      deadline = min(deadline, result.evidence_expiry_ns)
    if now_ns >= deadline:
      return reject('current_evidence_expired')
    candidate = ContinuationCandidate(
      target,
      tuple(float(value) for value in result.states[:, 1]),
      tuple(float(value) for value in result.states[:, 2]),
      tuple(float(value) for value in result.jerks),
      speed_tolerance=result.speed_tolerance, accel_tolerance=result.accel_tolerance,
    )
    self.offer = ContinuationOffer(self.budget.episode_start_ns, donor.producer_ns, donor.source_ns,
                                   donor.model_ns, int(deadline), candidate,
                                   source_cruise_kph=float(state['carState'].vCruise))
    self.tail_extension_s = result.tail_extension_s
    self.reason = 'accepted'
    return self.offer
