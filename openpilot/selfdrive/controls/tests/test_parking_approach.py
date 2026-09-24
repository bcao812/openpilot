"""Engineering-envelope properties and exact-source observation countercases."""
from types import SimpleNamespace
import struct

import pytest

from openpilot.selfdrive.controls.lib.low_speed_stop_distance import LowSpeedStopDistance
from openpilot.selfdrive.controls.lib.parking_approach import (
  FALLBACK_ACCEL, MAX_POSITION_STD, ParkingApproach, VisionLeadUncertainty,
)


BASE_NS = 10_000_000_000
STEP_NS = 50_000_000


def float32(value):
  return struct.unpack('f', struct.pack('f', value))[0]


def vision_observations(timestamp, *, speed=.6, gaps=(15.0, 20.0), present=(True, False),
                        lead_speeds=(0.0, 0.0), stds=(.1, .1), probabilities=(.995, .995)):
  models, radar_leads = [], []
  for index in range(2):
    acceleration = -.1 if lead_speeds[index] > .3 else 0.0
    model = SimpleNamespace(t=[0.0], x=[gaps[index] + 1.52], y=[0.0], v=[lead_speeds[index]],
                            a=[acceleration], xStd=[stds[index]], prob=probabilities[index])
    models.append(model)
    radar_leads.append(SimpleNamespace(
      present=present[index], radar=False, radarTrackId=-1, dRel=float32(model.x[0] - 1.52), yRel=0.0,
      vRel=float32(model.v[0] - speed), vLead=model.v[0], aLeadK=float32(acceleration), modelProb=.995,
    ))
  return (SimpleNamespace(velocity=SimpleNamespace(x=[speed]), leadsV3=models),
          SimpleNamespace(mdMonoTime=timestamp, leadOne=radar_leads[0], leadTwo=radar_leads[1]))


def frame_update(guard, policy, frame, *, speed=.6, gaps=(15.0, 20.0), present=(True, False), delay=.3,
                 measured_accel=0.0, age_ns=0, lead_speeds=(0.0, 0.0), stds=(.1, .1), allow_activation=True):
  now = BASE_NS + frame * STEP_NS
  timestamp = now - age_ns
  model, radar = vision_observations(timestamp, speed=speed, gaps=gaps, present=present,
                                     lead_speeds=lead_speeds, stds=stds)
  uncertainty = guard.uncertainty_cache.update(now_ns=now, model_ns=timestamp, model=model,
                                              radar_ns=timestamp, radar=radar, valid=True)
  if not uncertainty.qualified:
    policy.clear(source_barrier_ns=timestamp)
  policy.update(now_ns=now, source_ns=timestamp, radar_ns=timestamp, speed=speed, radar=radar,
                permitted=True, allow_activation=allow_activation)
  decision = guard.update(now_ns=now, source_ns=timestamp, radar_ns=timestamp, speed=speed,
                          measured_accel=measured_accel, actuator_delay=delay, radar=radar,
                          qualification=policy.evidence, uncertainty=uncertainty, permitted=True,
                          allow_activation=allow_activation)
  return decision, model, radar


def qualified_approach(*, speed=.6, gaps=(15.0, 20.0), lead_speeds=(0.0, 0.0), **kwargs):
  guard, policy = ParkingApproach(), LowSpeedStopDistance()
  for frame in range(121):
    current_gaps = tuple(gap + (speed - lead_speed) * (120 - frame) * .05
                         for gap, lead_speed in zip(gaps, lead_speeds, strict=True))
    decision, model, radar = frame_update(guard, policy, frame, speed=speed, gaps=current_gaps,
                                         lead_speeds=lead_speeds, **kwargs)
    if frame < 100:
      assert not decision.eligible
  return guard, policy, decision, model, radar


def test_complete_history_has_distinct_eligibility_and_actual_application():
  guard, _, decision, _, _ = qualified_approach()
  assert decision.eligible and decision.allow_compact
  assert decision.reason == 'margin_admissible'
  assert decision.accel_cap < 0.0
  assert not decision.cap_effective
  assert guard.braking_since_ns is None
  applied = guard.record_application(accel_before=.4, accel_after=decision.accel_cap)
  assert applied.cap_effective
  assert guard.braking_since_ns == decision.now_ns


@pytest.mark.parametrize('variable,values,increasing', [
  ('gap', (7.0, 10.0, 15.0), True),
  ('speed', (.2, .6, 1.0), False),
  ('delay', (.1, .3, .8), False),
  ('std', (.1, .5, 1.0), False),
  ('age_ns', (0, 100_000_000, 250_000_000), False),
  ('measured_accel', (0.0, .3, .6), False),
])
def test_margin_monotonicity(variable, values, increasing):
  margins = []
  for value in values:
    kwargs = {'gaps': (value, 20.0)} if variable == 'gap' else {'stds': (value, value)} if variable == 'std' else {variable: value}
    decision = qualified_approach(**kwargs)[2]
    assert decision.eligible
    margins.append(decision.margin)
  differences = [right - left for left, right in zip(margins, margins[1:])]
  assert all(delta > 0.0 if increasing else delta < 0.0 for delta in differences)


def test_budget_deficit_requests_a_farther_stop_without_a_three_meter_goal():
  _, _, decision, _, _ = qualified_approach(gaps=(5.0, 20.0))
  assert decision.eligible and decision.margin < 0.0
  assert not decision.allow_compact and decision.require_stop
  assert decision.accel_cap == FALLBACK_ACCEL


def test_every_present_slot_limits_admission_and_raw_nearest_slot_owns_cap():
  _, _, decision, _, _ = qualified_approach(gaps=(15.0, 8.0), present=(True, True), stds=(.1, .5))
  assert decision.eligible and decision.limiting_lead == 1
  assert decision.raw_gap == pytest.approx(8.0)
  assert decision.distance_uncertainty == pytest.approx(1.5)
  assert decision.clearance_lower == pytest.approx(1.5)
  invalid = qualified_approach(present=(True, True), stds=(.1, 1000.0))[2]
  assert not invalid.eligible and not invalid.allow_compact and invalid.accel_cap is None


def test_ordinary_moving_lead_cannot_acquire_a_new_stop_cap():
  _, _, decision, _, _ = qualified_approach(lead_speeds=(.8, .8))
  assert not decision.eligible and decision.accel_cap is None and not decision.require_stop


@pytest.mark.parametrize('fault', ['lost_lead', 'raw_standstill', 'unknown_std'])
def test_fault_after_real_application_keeps_braking_until_owner_ack(fault):
  guard, policy, decision, _, _ = qualified_approach()
  assert guard.record_application(accel_before=.4, accel_after=decision.accel_cap).cap_effective
  kwargs = {'present': (False, False)} if fault == 'lost_lead' else {'allow_activation': False} if fault == 'raw_standstill' else {'stds': (float('nan'), .1)}
  retained, _, radar = frame_update(guard, policy, 121, gaps=(14.97, 19.97), **kwargs)
  assert retained.require_stop and retained.accel_cap == FALLBACK_ACCEL
  assert not retained.allow_compact
  # Fresh/recovered data and elapsed time cannot silently release the pending
  # stop. Only the separately checked owner acknowledgement transfers it.
  for frame in range(122, 325):
    retained, _, radar = frame_update(guard, policy, frame, gaps=(14.97, 19.97))
    assert retained.require_stop and retained.accel_cap == FALLBACK_ACCEL
  now = BASE_NS + 325 * STEP_NS
  handed_over = guard.update(now_ns=now, source_ns=now, radar_ns=now, speed=0.0, measured_accel=0.0,
                             actuator_delay=.3, radar=radar, qualification=None, permitted=True,
                             allow_activation=False, owner_stopping=True)
  assert handed_over.reason == 'owner_stopping' and handed_over.accel_cap is None
  assert guard.braking_since_ns is None
  assert not handed_over.allow_compact and not handed_over.require_stop


def test_stronger_independent_braking_is_not_reported_as_candidate_application():
  guard, policy, decision, _, _ = qualified_approach()
  assert not guard.record_application(accel_before=-3.0, accel_after=-3.0).cap_effective
  assert guard.braking_since_ns is None
  lost, _, _ = frame_update(guard, policy, 121, gaps=(14.97, 19.97), present=(False, False))
  assert lost.accel_cap is None and not lost.require_stop


def test_inactive_authority_clears_pending_braking_without_new_motion_permission():
  guard, _, decision, _, radar = qualified_approach()
  guard.record_application(accel_before=.4, accel_after=decision.accel_cap)
  cleared = guard.update(now_ns=decision.now_ns + STEP_NS, source_ns=decision.source_ns + STEP_NS,
                         radar_ns=decision.radar_ns + STEP_NS, speed=.6, measured_accel=0.0,
                         actuator_delay=.3, radar=radar, qualification=None, permitted=False, allow_activation=True)
  assert cleared.reason == 'not_permitted' and cleared.accel_cap is None
  assert not cleared.allow_compact and guard.braking_since_ns is None


@pytest.mark.parametrize('std', [0.0, -.1, MAX_POSITION_STD + .001, 34469.535, float('nan'), float('inf')])
def test_bad_uncertainty_does_not_create_an_initial_phantom_stop(std):
  decision = qualified_approach(stds=(std, .1))[2]
  assert not decision.uncertainty_qualified and not decision.allow_compact
  assert decision.accel_cap is None and not decision.require_stop


def test_uncertainty_cache_joins_the_exact_older_radar_model_source():
  cache = VisionLeadUncertainty()
  first_model, first_radar = vision_observations(BASE_NS, stds=(.5, .1))
  first = cache.update(now_ns=BASE_NS, model_ns=BASE_NS, model=first_model,
                       radar_ns=BASE_NS, radar=first_radar, valid=True)
  assert first.qualified and first.distance_uncertainty == pytest.approx(1.5)
  next_ns = BASE_NS + STEP_NS
  newer_model, _ = vision_observations(next_ns, stds=(1000.0, .1))
  joined = cache.update(now_ns=next_ns, model_ns=next_ns, model=newer_model,
                        radar_ns=BASE_NS, radar=first_radar, valid=True)
  assert joined.qualified and joined.model_ns == BASE_NS
  assert joined.distance_uncertainty == pytest.approx(1.5)
  first_radar.mdMonoTime = BASE_NS + 1
  missing = cache.update(now_ns=next_ns + STEP_NS, model_ns=next_ns, model=newer_model,
                         radar_ns=next_ns, radar=first_radar, valid=True)
  assert not missing.qualified and missing.reason == 'unmatched_model_source'


def test_envelope_counts_underlying_model_age_even_with_new_radar_publication():
  guard, policy, decision, _, old_radar = qualified_approach()
  now = decision.now_ns + 200_000_000
  new_model, _ = vision_observations(now, gaps=(14.88, 19.88))
  uncertainty = guard.uncertainty_cache.update(now_ns=now, model_ns=now, model=new_model,
                                              radar_ns=now, radar=old_radar, valid=True)
  assert uncertainty.qualified and uncertainty.model_ns == decision.now_ns
  policy.update(now_ns=now, source_ns=now, radar_ns=now, speed=.6, radar=old_radar,
                permitted=True, allow_activation=True)
  aged = guard.update(now_ns=now, source_ns=now, radar_ns=now, speed=.6, measured_accel=0.0,
                       actuator_delay=.3, radar=old_radar, qualification=policy.evidence,
                       uncertainty=uncertainty, permitted=True, allow_activation=True)
  assert aged.eligible and aged.observation_age == pytest.approx(.2)
  assert aged.response_time == pytest.approx(decision.response_time + .2)
  assert aged.margin < decision.margin


def test_uncertainty_fault_before_application_requires_new_complete_history():
  guard, policy, decision, _, _ = qualified_approach()
  assert decision.eligible and guard.braking_since_ns is None
  invalid, _, _ = frame_update(guard, policy, 121, gaps=(14.97, 19.97), stds=(100.0, .1))
  assert not invalid.eligible and not invalid.require_stop
  for frame in range(122, 235):
    recovered, _, _ = frame_update(guard, policy, frame, gaps=(15.0 - (frame - 120) * .03, 20.0))
    if frame < 222:
      assert not recovered.eligible and not recovered.allow_compact
  assert recovered.eligible


@pytest.mark.parametrize('fault', ['distance', 'slot', 'raw_probability', 'stale', 'future', 'zero_stamp', 'std_missing'])
def test_source_mapping_requires_observed_values_and_fresh_identity(fault):
  cache = VisionLeadUncertainty()
  model, radar = vision_observations(BASE_NS)
  now, radar_ns = BASE_NS, BASE_NS
  if fault == 'distance':
    radar.leadOne.dRel += .0001
  elif fault == 'slot':
    radar.leadOne, radar.leadTwo = radar.leadTwo, radar.leadOne
  elif fault == 'raw_probability':
    model.leadsV3[0].prob = .5
  elif fault == 'stale':
    now += 250_000_001
  elif fault == 'future':
    radar.mdMonoTime += 1
  elif fault == 'zero_stamp':
    radar.mdMonoTime = 0
  else:
    model.leadsV3[0].xStd = []
  result = cache.update(now_ns=now, model_ns=BASE_NS, model=model, radar_ns=radar_ns, radar=radar, valid=True)
  assert not result.qualified


def test_repeated_identity_mutation_stays_unknown_and_clock_reversal_clears_cache():
  cache = VisionLeadUncertainty()
  model, radar = vision_observations(BASE_NS)
  assert cache.update(now_ns=BASE_NS, model_ns=BASE_NS, model=model, radar_ns=BASE_NS, radar=radar, valid=True).qualified
  model.leadsV3[0].xStd = [.5]
  mutated = cache.update(now_ns=BASE_NS + STEP_NS, model_ns=BASE_NS, model=model,
                         radar_ns=BASE_NS, radar=radar, valid=True)
  assert mutated.reason == 'mutated_model_source'
  model.leadsV3[0].xStd = [.1]
  assert not cache.update(now_ns=BASE_NS + 2 * STEP_NS, model_ns=BASE_NS, model=model,
                           radar_ns=BASE_NS, radar=radar, valid=True).qualified
  reversed_source = cache.update(now_ns=BASE_NS + 3 * STEP_NS, model_ns=BASE_NS - 1, model=model,
                                 radar_ns=BASE_NS, radar=radar, valid=True)
  assert reversed_source.reason == 'model_clock_reversed'
