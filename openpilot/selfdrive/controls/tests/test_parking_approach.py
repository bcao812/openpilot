"""Engineering-envelope properties and exact-source observation countercases."""
from types import SimpleNamespace
import struct

import pytest

from openpilot.selfdrive.controls.lib.low_speed_stop_distance import LowSpeedStopDistance
from openpilot.selfdrive.controls.lib.parking_approach import MAX_POSITION_STD, ParkingApproach, VisionLeadUncertainty
from openpilot.selfdrive.controls.tests.test_lead_qualification import formal_lead


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


def radar_observations(timestamp, *, gap=15.0, radar_std=.2, vision_std=1000.0):
  model, radar = vision_observations(timestamp, stds=(vision_std, vision_std))
  model.timestampEof = timestamp
  radar.fusionActive = True
  radar.leadOne = formal_lead(now_ns=timestamp, dRel=gap, vLead=0.0, vRel=-.6,
                              evidence={'positionStd': radar_std})
  return model, radar


def test_radar_parking_uncertainty_uses_radar_measurements_even_when_visual_std_is_unusable():
  cache = VisionLeadUncertainty()
  model, radar = radar_observations(BASE_NS, radar_std=.5)
  result = cache.update(now_ns=BASE_NS, model_ns=BASE_NS, model=model, radar_ns=BASE_NS, radar=radar, valid=True)
  assert result.qualified and result.reason == 'formal_radar_source'
  assert result.distance_uncertainty == pytest.approx(1.5)
  assert result.measurement_ns == BASE_NS
  assert result.lead_identities == (('radar-session-a', 7), None)


@pytest.mark.parametrize('field,value', [('positionStd', 0.0), ('positionStd', float('nan')),
                                        ('positionStd', 1.01), ('velocityStd', .51), ('positionValid', False)])
def test_good_visual_precision_cannot_endorse_bad_radar_uncertainty(field, value):
  cache = VisionLeadUncertainty()
  model, radar = radar_observations(BASE_NS, vision_std=.1)
  setattr(radar.leadOne.evidence, field, value)
  result = cache.update(now_ns=BASE_NS, model_ns=BASE_NS, model=model, radar_ns=BASE_NS, radar=radar, valid=True)
  assert not result.qualified


def test_mixed_radar_and_vision_source_joins_original_hypothesis_after_display_reordering():
  cache = VisionLeadUncertainty()
  model, radar = radar_observations(BASE_NS, vision_std=.1)
  radar.leadTwo = formal_lead(source='vision', now_ns=BASE_NS, dRel=15.0, vLead=0.0, vRel=float32(-.6),
                              modelProb=.995, evidence={'visionIndex': 0, 'positionStd': .1})
  result = cache.update(now_ns=BASE_NS, model_ns=BASE_NS, model=model, radar_ns=BASE_NS, radar=radar, valid=True)
  assert result.qualified
  assert result.position_std == pytest.approx((.2, .1))


def test_continuous_radar_only_approach_can_acquire_compact_geometry():
  guard, policy = ParkingApproach(), LowSpeedStopDistance()
  for frame in range(121):
    now = BASE_NS + frame * STEP_NS
    model, radar = radar_observations(now, gap=15.0 + .6 * (120 - frame) * .05)
    uncertainty = guard.uncertainty_cache.update(now_ns=now, model_ns=now, model=model, radar_ns=now, radar=radar, valid=True)
    policy.update(now_ns=now, source_ns=now, radar_ns=now, speed=.6, radar=radar, permitted=True)
    decision = guard.update(now_ns=now, source_ns=now, radar_ns=now, speed=.6, measured_accel=0.0,
                            actuator_delay=.3, radar=radar, qualification=policy.evidence,
                            uncertainty=uncertainty, permitted=True, allow_activation=True)
  assert decision.eligible and decision.allow_compact
  assert decision.uncertainty_reason == 'formal_radar_source'
  now += STEP_NS
  model, radar = radar_observations(now, gap=14.97)
  uncertainty = guard.uncertainty_cache.update(now_ns=now, model_ns=now, model=model, radar_ns=now, radar=radar, valid=True)
  policy.update(now_ns=now, source_ns=now, radar_ns=now, speed=.6, radar=radar, permitted=True)
  radar.leadOne.evidence.trackId = 8
  changed = guard.update(now_ns=now, source_ns=now, radar_ns=now, speed=.6, measured_accel=0.0,
                         actuator_delay=.3, radar=radar, qualification=policy.evidence,
                         uncertainty=uncertainty, permitted=True, allow_activation=True)
  assert not changed.allow_compact and changed.reason == 'changed_lead_identity'


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


def test_complete_history_admits_geometry_without_actuator_or_owner_authority():
  _, _, decision, _, _ = qualified_approach()
  assert decision.eligible and decision.allow_compact
  assert decision.reason == 'margin_admissible'
  assert decision.policy_kind == 'geometry_margin_only'
  assert decision.accel_cap is None and not decision.cap_effective
  assert not decision.require_stop and not decision.stop_evidence_added
  assert not decision.held_episode and not decision.release_blocked


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


def test_budget_deficit_rejects_compact_geometry_without_adding_stop_evidence():
  _, _, decision, _, _ = qualified_approach(gaps=(5.0, 20.0))
  assert decision.eligible and decision.margin < 0.0
  assert not decision.allow_compact and not decision.require_stop
  assert decision.accel_cap is None


def test_every_present_slot_limits_admission_and_raw_nearest_slot_defines_margin():
  _, _, decision, _, _ = qualified_approach(gaps=(15.0, 8.0), present=(True, True), stds=(.1, .5))
  assert decision.eligible and decision.limiting_lead == 1
  assert decision.raw_gap == pytest.approx(8.0)
  assert decision.distance_uncertainty == pytest.approx(1.5)
  assert decision.clearance_lower == pytest.approx(1.5)
  invalid = qualified_approach(present=(True, True), stds=(.1, 1000.0))[2]
  assert not invalid.eligible and not invalid.allow_compact and invalid.accel_cap is None


def test_ordinary_moving_lead_cannot_acquire_compact_geometry():
  _, _, decision, _, _ = qualified_approach(lead_speeds=(.8, .8))
  assert not decision.eligible and decision.accel_cap is None and not decision.require_stop


@pytest.mark.parametrize('fault', ['lost_lead', 'raw_standstill', 'unknown_std'])
def test_fault_after_admission_withdraws_only_compact_geometry(fault):
  guard, policy, decision, _, _ = qualified_approach()
  assert decision.allow_compact
  kwargs = {'present': (False, False)} if fault == 'lost_lead' else {'allow_activation': False} if fault == 'raw_standstill' else {'stds': (float('nan'), .1)}
  withdrawn, _, _ = frame_update(guard, policy, 121, gaps=(14.97, 19.97), **kwargs)
  assert not withdrawn.allow_compact
  assert withdrawn.accel_cap is None and not withdrawn.require_stop
  assert not withdrawn.held_episode and not withdrawn.release_blocked


def test_inactive_inputs_withdraw_geometry_admission():
  guard, _, decision, _, radar = qualified_approach()
  cleared = guard.update(now_ns=decision.now_ns + STEP_NS, source_ns=decision.source_ns + STEP_NS,
                         radar_ns=decision.radar_ns + STEP_NS, speed=.6, measured_accel=0.0,
                         actuator_delay=.3, radar=radar, qualification=None, permitted=False, allow_activation=True)
  assert cleared.reason == 'not_permitted' and cleared.accel_cap is None
  assert not cleared.allow_compact and not cleared.require_stop and not cleared.release_blocked


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
  assert decision.eligible and decision.allow_compact and decision.accel_cap is None
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
