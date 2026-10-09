"""Episode clocks and independently versioned continuation wire evidence."""
from dataclasses import replace

import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_continuation import (
  CONTINUATION_N, CONTINUATION_TIMES, ContinuationBudget, ContinuationCandidate, ContinuationDonor, ContinuationHistory,
  ContinuationOffer, execution_target, validate_continuation,
)
from openpilot.selfdrive.controls.lib.reference_timing import ReferenceTiming


START = 10_000_000_000


def protocol_fixture():
  reference = ReferenceTiming(START, START, START + 1_000_000, START - 1_000_000_000, 10.0, 0.0, 10.0, 0.0, True)
  donor = ContinuationDonor(START - 40_000_000, START - 50_000_000, START - 50_000_000, reference.instance_ns)
  budget = ContinuationBudget()
  assert budget.note_failure(reference.producer_ns)
  candidate = ContinuationCandidate(0.0, (10.0,) * 13, (0.0,) * 13, (0.0,) * 12, .2, .25)
  until = budget.deadline(donor_producer_ns=donor.producer_ns, donor_source_ns=donor.source_ns, donor_model_ns=donor.model_ns)
  offer = ContinuationOffer(budget.episode_start_ns, donor.producer_ns, donor.source_ns, donor.model_ns, until,
                            candidate, source_cruise_kph=90.0)
  return reference, donor, budget, offer


def test_episode_cannot_be_renewed_by_republication_or_a_different_donor():
  reference, donor, budget, offer = protocol_fixture()
  assert budget.note_failure(START + 50_000_000, episode_start_ns=START + 50_000_000)
  assert budget.episode_start_ns == reference.producer_ns
  newer = {'donor_producer_ns': START + 40_000_000, 'donor_source_ns': START + 30_000_000, 'donor_model_ns': START + 30_000_000}
  assert budget.deadline(**newer) == offer.valid_until_ns
  assert budget.allows(offer.valid_until_ns - 1, **newer)
  assert not budget.allows(offer.valid_until_ns, **newer)
  assert not budget.note_failure(START - 1)
  assert not budget.allows(START - 1, **newer)


def test_single_success_and_repeated_samples_cannot_clear_an_episode():
  _, _, budget, _ = protocol_fixture()
  ids = {'producer_ns': START + 20_000_000, 'source_ns': START + 19_000_000, 'model_ns': START + 18_000_000}
  for offset in range(20_000_000, 700_000_001, 10_000_000):
    assert not budget.note_success(START + offset, **ids)
  assert budget.episode_start_ns


def test_only_half_a_second_of_fresh_advancing_native_successes_clears_budget():
  _, _, budget, _ = protocol_fixture()
  for index in range(11):
    now = START + 20_000_000 + index * 50_000_000
    cleared = budget.note_success(now, producer_ns=now - 1_000_000, source_ns=now - 2_000_000, model_ns=now - 2_000_000)
    assert cleared is (index == 10)
  assert budget.episode_start_ns == 0


@pytest.mark.parametrize('defect', ['none', 'revoked', 'version', 'donor', 'expired', 'no_tolerance', 'low_speed', 'cruise_reduced', 'bad_curve'])
def test_offer_validation_keeps_clock_identity_state_and_curve_independent(defect):
  reference, donor, budget, offer = protocol_fixture()
  now, speed, cruise = START + 5_000_000, 10.0, 90.0
  if defect == 'revoked':
    offer = replace(offer, valid=False)
  elif defect == 'version':
    offer = replace(offer, version=2)
  elif defect == 'donor':
    offer = replace(offer, donor_source_ns=donor.source_ns - 1)
  elif defect == 'expired':
    now = offer.valid_until_ns
  elif defect == 'no_tolerance':
    offer = replace(offer, candidate=replace(offer.candidate, accel_tolerance=0.0))
  elif defect == 'low_speed':
    speed = 1.99
  elif defect == 'cruise_reduced':
    cruise = 89.0
  elif defect == 'bad_curve':
    offer = replace(offer, candidate=replace(offer.candidate, jerks=(1.0,) * 12))
  reason = validate_continuation(offer, donor=donor, reference=reference, budget=budget,
                                 plan_ns=START + 2_000_000, model_ns=START, now_ns=now, speed=speed, accel=0.0, cruise_kph=cruise)
  assert bool(reason) is (defect != 'none')


def test_wire_preserves_final_float64_proof_and_header():
  _, _, _, offer = protocol_fixture()
  message = log.LongitudinalContinuation.new_message(**offer.to_dict())
  with log.LongitudinalContinuation.from_bytes(message.to_bytes()) as wire:
    assert ContinuationOffer.from_message(wire) == offer


@pytest.mark.parametrize('defect', ['none', 'missing_source', 'gap', 'uncovered_tail', 'earlier_velocity_error'])
def test_history_requires_the_entire_source_to_consumer_interval(defect):
  reference, _, _, offer = protocol_fixture()
  history = ContinuationHistory()
  offsets = [0, 10, 20, 30, 40]
  if defect == 'missing_source':
    offsets = offsets[1:]
  elif defect == 'gap':
    offsets = [0, 30, 40]
  for offset in offsets:
    speed = 10.3 if defect == 'earlier_velocity_error' and offset == 10 else 10.0
    assert history.observe(START + offset * 1_000_000, START + offset * 1_000_000, speed, 0.0)
  now = START + (51 if defect == 'uncovered_tail' else 42) * 1_000_000
  reason = history.check(offer.candidate, reference, now)
  assert bool(reason) is (defect != 'none')


def test_consumer_uses_native_knots_and_advances_the_actual_action_horizon():
  reference, _, _, offer = protocol_fixture()
  speeds, accels, jerks = [10.0], [0.0], []
  for left, right in zip(CONTINUATION_TIMES, CONTINUATION_TIMES[1:]):
    jerk = -1.0 if left < .1 else 0.0
    duration = right - left
    speeds.append(speeds[-1] + accels[-1] * duration + .5 * jerk * duration * duration)
    accels.append(accels[-1] + jerk * duration)
    jerks.append(jerk)
  candidate = ContinuationCandidate(0.0, tuple(speeds), tuple(accels), tuple(jerks), .2, .25)
  offer = replace(offer, candidate=candidate)
  assert len(candidate.speeds) == CONTINUATION_N + 1 and len(candidate.jerks) == CONTINUATION_N
  assert execution_target(offer, reference, START + 20_000_000, .05) < execution_target(offer, reference, START, .05)
