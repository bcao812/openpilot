"""Lead launch continuity and model-risk checks without a native MPC import."""
from dataclasses import replace

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.lead_launch import LeadLaunch, LeadLaunchSample
from openpilot.selfdrive.controls.lib.model_reference import ModelReferenceAdapter, lead_launch_model_compatible, stationary_lead_departure_compatible
from openpilot.selfdrive.controls.tests.test_typed_model_reference import MODEL_TIMES, SOURCE_NS, evidence


# 00000097--ec04c76355, modelV2=721395803838 ns, recorded 476a8e30.
# This snapshot supplies model shape, not counterfactual vehicle motion.
LOGGED_WEAK_VELOCITY = np.array([
  -0.015455774031579494, -0.014618771150708199, -0.014657740481197834, -0.01227776613086462,
  -0.016952596604824066, -0.008557301945984364, -0.013123937882483006, -0.007659140974283218,
  -0.006712107453495264, -0.002959253964945674, -0.004078581929206848, 0.0021161003969609737,
  0.010988502763211727, 0.009156341664493084, 0.017624957486987114, 0.03194738179445267,
  0.04682901129126549, 0.060226526111364365, 0.07597185671329498, 0.09375514090061188,
  0.10178571939468384, 0.11723214387893677, 0.13301463425159454, 0.15322226285934448,
  0.16596026718616486, 0.18117552995681763, 0.213136687874794, 0.24087536334991455,
  0.2686641812324524, 0.3093359172344208, 0.35051408410072327, 0.4149084985256195,
  0.4988994598388672,
])


def tick(policy, time_s, **changes):
  now = SOURCE_NS + round(time_s * 1e9)
  args = dict(now_ns=now, model_ns=now - 2_000_000, radar_ns=now - 3_000_000,
              enabled=True, can_start=True, speed=0.0,
              leads=(LeadLaunchSample(('vision', 0), 7.7 + 0.8 * time_s, 0.8, 0.8, 0.0),))
  args.update(changes)
  return policy.update(**args)


def confirmed_policy():
  policy = LeadLaunch()
  for frame in range(6):
    active = tick(policy, frame * 0.05)
    assert active == (frame == 5)
  return policy


def test_only_continuous_fresh_departure_can_start_a_held_vehicle():
  policy = confirmed_policy()
  assert policy.active and policy.reason == 'active'
  assert tick(policy, 0.3, can_start=False, speed=0.2)
  moving = LeadLaunch()
  assert not tick(moving, 0.0, can_start=False, speed=0.2)
  assert not tick(moving, 0.3, can_start=False, speed=0.2)


def test_reusing_either_source_does_not_build_confirmation():
  for frozen in ('model_ns', 'radar_ns'):
    policy = LeadLaunch()
    tick(policy, 0.0)
    timestamp = getattr(policy, 'last_' + frozen)
    for frame in range(1, 4):
      assert not tick(policy, frame * 0.05, **{frozen: timestamp})
      assert policy.confirmed_ns == 0


@pytest.mark.parametrize('changes', [
  {'enabled': False}, {'speed': -0.001}, {'speed': 2.0}, {'speed': np.nan},
  {'model_ns': SOURCE_NS}, {'radar_ns': SOURCE_NS}, {'model_ns': True},
  {'model_ns': SOURCE_NS + 400_000_000}, {'leads': ()},
])
def test_current_veto_or_invalid_source_revokes_existing_authority(changes):
  policy = confirmed_policy()
  assert not tick(policy, 0.3, **changes)
  assert not policy.active


@pytest.mark.parametrize('sample', [
  LeadLaunchSample(('vision', 1), 7.94, 0.8, 0.8, 0.0),
  LeadLaunchSample(('radar', 'session', 7), 7.94, 0.8, 0.8, 0.0),
  LeadLaunchSample(('vision', 0), 5.5, 0.8, 0.8, 0.0),
  LeadLaunchSample(('vision', 0), 7.94, 0.8, 0.8, 0.6),
  LeadLaunchSample(('vision', 0), 7.94, 2.0, 2.0, 0.0),
])
def test_changed_target_cannot_spend_old_confirmation(sample):
  policy = confirmed_policy()
  assert not tick(policy, 0.3, leads=(sample,))
  assert policy.reason == 'lead_changed'
  assert not tick(policy, 0.35, can_start=False, leads=(sample,))


def test_source_gap_requires_new_confirmation():
  policy = confirmed_policy()
  assert not tick(policy, 0.5)
  assert policy.reason == 'source_discontinuity'
  assert not tick(policy, 0.55)


def test_time_budget_cannot_rearm_in_the_same_stopped_window():
  policy = confirmed_policy()
  for frame in range(6, 106):
    active = tick(policy, frame * 0.05)
  assert not active and policy.exhausted and policy.reason == 'time_budget'
  assert not tick(policy, 5.3)
  assert not tick(policy, 5.35, enabled=False)
  assert not policy.exhausted
  assert not tick(policy, 5.4)


def test_travel_budget_ends_the_launch_before_the_time_budget():
  policy = confirmed_policy()
  # Keep the observed lead consistent while the ego travels at 1.5 m/s.
  travel, previous_speed = 0.0, 0.0
  for frame in range(6, 90):
    time_s, speed = frame * 0.05, 1.5
    travel += 0.5 * (previous_speed + speed) * 0.05
    lead = LeadLaunchSample(('vision', 0), 7.7 + 0.8 * time_s - travel, 0.8, 0.8 - speed, 0.0)
    active = tick(policy, time_s, can_start=False, speed=speed, leads=(lead,))
    previous_speed = speed
    if not active:
      break
  assert policy.exhausted and policy.reason == 'distance_budget'
  assert travel >= 5.0 and time_s < 5.0


@pytest.mark.parametrize('later_publication', [False, True])
def test_weak_recorded_model_is_compatible_without_reclassifying_its_raw_intent(later_publication):
  ref = evidence(LOGGED_WEAK_VELOCITY, 0.0, model_ns=SOURCE_NS + int(later_publication) * 5_000_000,
                 now_ns=SOURCE_NS + 10_000_000)
  before = ref.events
  assert not stationary_lead_departure_compatible(ref, actual_standstill=True, measured_speed=0.0, raw_speed=0.0)
  assert lead_launch_model_compatible(ref)
  assert ref.events == before and not ref.allows_departure and not ref.allows_external_departure


@pytest.mark.parametrize('raw', [
  np.zeros_like(MODEL_TIMES),
  np.where(MODEL_TIMES < 1.0, 0.0, 0.1 * MODEL_TIMES),
  np.where(MODEL_TIMES >= 3.0, 0.0, LOGGED_WEAK_VELOCITY),
  np.where(MODEL_TIMES < 2.0, -0.04, 0.1 * MODEL_TIMES),
  np.where(MODEL_TIMES == MODEL_TIMES[4], -0.06, LOGGED_WEAK_VELOCITY),
  np.where(MODEL_TIMES == MODEL_TIMES[22], -0.001, LOGGED_WEAK_VELOCITY),
  np.where(MODEL_TIMES == MODEL_TIMES[16], np.nan, LOGGED_WEAK_VELOCITY),
])
def test_launch_does_not_erase_real_rest_reverse_or_invalid_model_evidence(raw):
  assert not lead_launch_model_compatible(evidence(raw, 0.0))


def test_launch_does_not_admit_a_stale_or_mutated_model_identity():
  ref = evidence(LOGGED_WEAK_VELOCITY, 0.0)
  assert not lead_launch_model_compatible(replace(ref, source=replace(ref.source, consumed_mono_time_ns=SOURCE_NS + 300_000_000)))
  assert not lead_launch_model_compatible(None)
  adapter = ModelReferenceAdapter()
  assert lead_launch_model_compatible(evidence(LOGGED_WEAK_VELOCITY, 0.0, adapter=adapter))
  changed = LOGGED_WEAK_VELOCITY.copy()
  changed[-1] += 0.1
  mutated = evidence(changed, 0.0, adapter=adapter)
  assert mutated.unknown.reason == 'model_mutated_with_same_identity'
  assert not lead_launch_model_compatible(mutated)
  regressed = evidence(LOGGED_WEAK_VELOCITY, 0.0, adapter=adapter, model_ns=SOURCE_NS - 1)
  assert regressed.unknown.reason == 'model_publication_regressed'
  assert not lead_launch_model_compatible(regressed)
