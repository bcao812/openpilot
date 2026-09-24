"""Behavioral examples for the qualified low-speed stand-off policy.

Synthetic publications have explicit times and coherent lead-slot observations.
The examples assert public targets, never private queues or implementation state.
Execute in the isolated host snapshot after committing the complete change.
"""
import math
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.low_speed_stop_distance import LowSpeedStopDistance


BASE_NS = 10_000_000_000
FRAME_NS = 50_000_000
AUTO = object()


def stamp(frame):
  return BASE_NS + frame * FRAME_NS


def lead(frame, ego_speed=.3, lead_speed=0.0, initial_distance=150.0, **changes):
  relative_speed = lead_speed - ego_speed
  values = {'present': True, 'dRel': initial_distance + relative_speed * frame * .05,
            'vRel': relative_speed, 'vLead': lead_speed, 'aLeadK': 0.0, 'modelProb': .995, 'yRel': 0.0}
  values.update(changes)
  return SimpleNamespace(**values)


def radar(frame, speed=.3, lead_speed=0.0):
  return SimpleNamespace(leadOne=lead(frame, speed, lead_speed),
                         leadTwo=lead(frame, speed, lead_speed, initial_distance=180.0))


def update(policy, frame, *, speed=.3, observations=AUTO, now_ns=AUTO,
           source_ns=AUTO, radar_ns=AUTO, permitted=True, allow_activation=True):
  return policy.update(now_ns=stamp(frame) if now_ns is AUTO else now_ns,
                       source_ns=stamp(frame) if source_ns is AUTO else source_ns,
                       radar_ns=stamp(frame) if radar_ns is AUTO else radar_ns,
                       speed=speed, radar=radar(frame, speed) if observations is AUTO else observations,
                       permitted=permitted, allow_activation=allow_activation)


def run(policy, first=0, last=220, **kwargs):
  result = None
  for frame in range(first, last + 1):
    result = update(policy, frame, **kwargs)
  return result


def qualified():
  policy = LowSpeedStopDistance()
  comfort, risk = run(policy)
  assert comfort == pytest.approx(3.0)
  assert risk == pytest.approx((4.0, 4.0))
  return policy


def assert_default_low_speed(result):
  comfort, risk = result
  assert comfort == pytest.approx(4.0)
  assert risk == pytest.approx((6.0, 6.0))


def estimated_approach_radar(frame, acceleration=-.1):
  """Continuous .8 m/s slot observations with a chosen acceleration estimate."""
  observations = radar(frame, lead_speed=.8)
  observations.leadOne.aLeadK = observations.leadTwo.aLeadK = acceleration
  return observations


def cruising_history():
  policy = LowSpeedStopDistance()
  for frame in range(221):
    result = update(policy, frame, observations=radar(frame, lead_speed=.8))
  assert_default_low_speed(result)
  return policy


def halfway_through_approach_ramp():
  policy = cruising_history()
  # 500 ms continuously negative acceleration establishes an approach; the
  # next 250 ms reaches the middle of the separate 500 ms target ramp.
  for frame in range(221, 237):
    result = update(policy, frame, observations=estimated_approach_radar(frame))
  assert result[0] == pytest.approx(3.5)
  assert result[1] == pytest.approx((5.0, 5.0))
  return policy, result


def assert_fresh_ten_second_requalification(policy, first_frame, distances, *, previous_speed=.3,
                                          present=(True, True)):
  """Follow the same stationary slots through recovery; no private state access."""
  distances = list(distances)
  for frame in range(first_frame, first_frame + 213):
    speed = .3
    travelled = .5 * (previous_speed + speed) * .05
    distances = [distance - travelled for distance in distances]
    observations = radar(frame, speed)
    for index, item in enumerate((observations.leadOne, observations.leadTwo)):
      item.present = present[index]
      item.dRel = distances[index]
    result = update(policy, frame, speed=speed, observations=observations)
    if frame < first_frame + 200:
      # The first genuinely new source sample is time zero. In particular,
      # first_frame+199 is only 9.95 s, even if the fault sample was valid CS.
      assert_default_low_speed(result)
    previous_speed = speed
  # Recovery must remain possible after the new complete window and ramp.
  assert result[0] == pytest.approx(3.0)
  assert result[1] == pytest.approx(tuple(4.0 if exists else 6.0 for exists in present))


def test_short_low_speed_recording_does_not_shorten_the_ten_second_requirement():
  policy = LowSpeedStopDistance()
  # A fully observed lead does not substitute for the missing 50 ms of ego history.
  assert_default_low_speed(run(policy, last=199))
  # Allow both a complete history and the documented settling/ramp duration.
  assert run(policy, first=200, last=240)[1] == pytest.approx((4.0, 4.0))


def test_held_car_state_cannot_complete_history_using_processing_time():
  policy = LowSpeedStopDistance()
  assert_default_low_speed(run(policy, last=196))  # Only 9.8 s of source history.
  for frame in range(197, 202):
    # Processing time crosses 10 s, but this is the same last carState publication.
    assert_default_low_speed(update(policy, frame, source_ns=stamp(196)))


def test_recent_high_speed_tail_cannot_acquire_compact_distance():
  policy = qualified()
  distance, previous_speed = lead(220).dRel, .3
  for frame in range(221, 462):
    speed = 40.0 / 3.6 if frame == 221 else .3
    # Keep dRel consistent across the speed changes, so the test cannot pass
    # merely because an artificial target jump erased the earlier history.
    distance -= .5 * (previous_speed + speed) * .05
    observations = radar(frame, speed)
    observations.leadOne.dRel = distance
    observations.leadTwo.dRel = distance + 30.0
    result = update(policy, frame, speed=speed, observations=observations)
    if frame == 221:
      assert result[1] == pytest.approx((6.0, 6.0))
    elif frame <= 421:
      # Include the old high-speed sample at the exact ten-second boundary.
      assert_default_low_speed(result)
    previous_speed = speed
  assert result[1] == pytest.approx((4.0, 4.0))


def test_exactly_thirty_kmh_in_recent_history_is_not_below_thirty():
  policy = LowSpeedStopDistance()
  update(policy, 0, speed=30.0 / 3.6)
  assert_default_low_speed(run(policy, first=1, last=200))
  assert run(policy, first=201, last=241)[1] == pytest.approx((4.0, 4.0))


@pytest.mark.parametrize('speed_kmh, comfort, risk', [
  (0.0, 3.0, 4.0),
  (1.08, 3.0, 4.0),   # .3 m/s: full terminal reduction.
  (2.7, 3.5, 5.0),    # .75 m/s: halfway between .3 and 1.2.
  (4.32, 4.0, 6.0),   # 1.2 m/s: no geometric reduction.
  (15.0, 4.0, 6.0),
  (22.5, 5.0, 6.0),
  (30.0, 6.0, 6.0),
])
def test_speed_profile_boundaries_and_midpoint(speed_kmh, comfort, risk):
  result = run(LowSpeedStopDistance(), last=240, speed=speed_kmh / 3.6)
  assert result[0] == pytest.approx(comfort)
  assert result[1] == pytest.approx((risk, risk))


def test_zero_speed_with_closed_activation_gate_returns_default_targets():
  policy = qualified()
  # Ego slows toward a stationary lead, integrating the distance explicitly
  # so changing ego speed does not fabricate a lead-slot position jump.
  distance, previous_speed = lead(220).dRel, .3
  for frame in range(221, 282):
    speed = max(0.0, .3 - .3 * (frame - 220) / 60.0)
    distance -= .5 * (previous_speed + speed) * .05
    observations = radar(frame, speed)
    observations.leadOne.dRel = distance
    observations.leadTwo.dRel = distance + 30.0
    result = update(policy, frame, speed=speed, observations=observations, allow_activation=speed > 0.0)
    if speed > 0.0:
      assert result[0] == pytest.approx(3.0)
      assert result[1] == pytest.approx((4.0, 4.0))
    else:
      assert_default_low_speed(result)
    previous_speed = speed


def test_new_lead_needs_half_second_stability_then_half_second_ramp():
  policy = LowSpeedStopDistance()
  absent = radar(0)
  absent.leadOne.present = absent.leadTwo.present = False
  assert_default_low_speed(run(policy, observations=absent))
  assert_default_low_speed(run(policy, first=221, last=230))  # 450 ms of lead observations.
  assert_default_low_speed(update(policy, 231))  # Stability just reaches 500 ms.
  halfway = run(policy, first=232, last=236)  # Another 250 ms, halfway through the ramp.
  assert halfway[0] == pytest.approx(3.5)
  assert halfway[1] == pytest.approx((5.0, 5.0))
  complete = run(policy, first=237, last=241)
  assert complete[0] == pytest.approx(3.0)
  assert complete[1] == pytest.approx((4.0, 4.0))


def test_repeated_radar_publication_does_not_advance_an_in_progress_ramp():
  policy = LowSpeedStopDistance()
  absent = radar(0)
  absent.leadOne.present = absent.leadTwo.present = False
  run(policy, observations=absent)
  before = run(policy, first=221, last=233)
  assert 4.0 < before[1][0] < 6.0
  held = radar(233)
  for frame in range(234, 238):
    after = update(policy, frame, observations=held, radar_ns=stamp(233))
    assert after[0] == pytest.approx(before[0])
    assert after[1] == pytest.approx(before[1])


def test_same_timestamp_replays_never_create_more_compact_targets():
  policy = LowSpeedStopDistance()
  absent = radar(0)
  absent.leadOne.present = absent.leadTwo.present = False
  run(policy, observations=absent)
  before = run(policy, first=221, last=233)
  for _ in range(30):
    after = update(policy, 233)
    assert after[0] >= before[0]
    assert all(actual >= earlier for actual, earlier in zip(after[1], before[1], strict=True))


@pytest.mark.parametrize('bad_slot', [0, 1])
def test_unqualified_present_slot_blocks_shared_compact_targets(bad_slot):
  policy = qualified()
  observations = radar(221)
  (observations.leadOne, observations.leadTwo)[bad_slot].modelProb = .97
  # Shared comfort cannot favor a good far lead at the expense of a bad close
  # present slot. Both current risk rows also return to the common default.
  assert_default_low_speed(update(policy, 221, observations=observations))


def test_participating_slot_loss_cannot_keep_compact_target_from_surviving_slot():
  policy = qualified()
  observations = radar(221)
  observations.leadOne.present = False
  assert_default_low_speed(update(policy, 221, observations=observations))


def test_swapped_distinct_slots_restart_observation_qualification():
  policy = qualified()
  observations = radar(221)
  observations.leadOne, observations.leadTwo = observations.leadTwo, observations.leadOne
  # Both distance slots jump by 30 m, inconsistent with the observed vRel.
  assert_default_low_speed(update(policy, 221, observations=observations))


@pytest.mark.parametrize('field, value', [('dRel', 155.0), ('yRel', .31)])
def test_observable_distance_or_lateral_jump_blocks_shared_qualification(field, value):
  policy = qualified()
  observations = radar(221)
  setattr(observations.leadOne, field, value)
  assert_default_low_speed(update(policy, 221, observations=observations))


@pytest.mark.parametrize('lead_speed', [0.0, 2.0])
def test_stationary_and_two_mps_leads_are_both_supported(lead_speed):
  policy = LowSpeedStopDistance()
  for frame in range(241):
    observations = radar(frame, lead_speed=lead_speed)
    if lead_speed > .3:
      # Isolate the original 2 m/s quality boundary with explicit stopping
      # approach evidence; a constant 2 m/s lead is no longer an entry case.
      observations.leadOne.aLeadK = observations.leadTwo.aLeadK = -.1
    result = update(policy, frame, observations=observations)
  assert result[0] == pytest.approx(3.0)
  assert result[1] == pytest.approx((4.0, 4.0))


@pytest.mark.parametrize('field, value', [
  ('dRel', 0.0), ('dRel', -1.0),
  ('vLead', -0.01), ('vLead', 2.01), ('aLeadK', -.51), ('aLeadK', .21),
  ('modelProb', .979), ('modelProb', 1.01), ('yRel', -1.21), ('yRel', 1.21),
  ('vRel', .51),
])
def test_reverse_fast_braking_uncertain_or_inconsistent_leads_fall_back(field, value):
  policy = qualified()
  observations = radar(221)
  setattr(observations.leadOne, field, value)
  assert_default_low_speed(update(policy, 221, observations=observations))


@pytest.mark.parametrize('field, value', [('aLeadK', -.5), ('aLeadK', .2), ('modelProb', .98),
                                         ('yRel', -1.2), ('yRel', 1.2)])
def test_lead_qualification_thresholds_are_inclusive(field, value):
  policy = LowSpeedStopDistance()
  for frame in range(241):
    observations = radar(frame)
    setattr(observations.leadOne, field, value)
    result = update(policy, frame, observations=observations)
  assert result[1] == pytest.approx((4.0, 4.0))


@pytest.mark.parametrize('field', ['dRel', 'vRel', 'vLead', 'aLeadK', 'modelProb', 'yRel'])
@pytest.mark.parametrize('value', [math.nan, math.inf, -math.inf])
def test_nonfinite_present_lead_measurements_immediately_remove_compact_targets(field, value):
  policy = qualified()
  observations = radar(221)
  setattr(observations.leadOne, field, value)
  assert_default_low_speed(update(policy, 221, observations=observations))


@pytest.mark.parametrize('field', ['dRel', 'vRel', 'vLead', 'aLeadK', 'modelProb', 'yRel'])
def test_missing_lead_field_is_unqualified_not_an_exception_or_default_zero(field):
  policy = qualified()
  observations = radar(221)
  delattr(observations.leadOne, field)
  assert_default_low_speed(update(policy, 221, observations=observations))


def test_missing_presence_field_clears_all_qualification():
  policy = qualified()
  observations = radar(221)
  del observations.leadOne.present
  assert_default_low_speed(update(policy, 221, observations=observations))


def test_missing_slot_structure_clears_all_qualification():
  policy = qualified()
  observations = SimpleNamespace(leadTwo=lead(221, initial_distance=180.0))
  assert_default_low_speed(update(policy, 221, observations=observations))


def test_missing_radar_object_safely_returns_default_distances():
  assert_default_low_speed(update(qualified(), 221, observations=None))


@pytest.mark.parametrize('field', ['source_ns', 'radar_ns'])
def test_freshness_boundary_is_250_ms_and_expiry_falls_back(field):
  policy = qualified()
  for frame in range(221, 226):
    observations = radar(220 if field == 'radar_ns' else frame)
    result = update(policy, frame, observations=observations, **{field: stamp(220)})
    assert result[1] == pytest.approx((4.0, 4.0))
  expiry = stamp(220) + 251_000_000
  other = 'radar_ns' if field == 'source_ns' else 'source_ns'
  result = update(policy, 226, now_ns=expiry, **{field: stamp(220), other: expiry})
  assert_default_low_speed(result)


@pytest.mark.parametrize('field', ['source_ns', 'radar_ns'])
def test_future_publication_cannot_keep_compact_target(field):
  policy = qualified()
  assert_default_low_speed(update(policy, 221, **{field: stamp(221) + 1}))


@pytest.mark.parametrize('field', ['now_ns', 'source_ns', 'radar_ns'])
def test_clock_rollback_invalidates_existing_target(field):
  policy = qualified()
  assert_default_low_speed(update(policy, 221, **{field: stamp(219)}))


@pytest.mark.parametrize('field', ['now_ns', 'source_ns', 'radar_ns'])
@pytest.mark.parametrize('bad_stamp', [0, -1, True, 1.5, None])
def test_invalid_timestamp_types_and_values_do_not_grant_compact_distance(field, bad_stamp):
  policy = qualified()
  assert_default_low_speed(update(policy, 221, **{field: bad_stamp}))


def test_observation_gap_requires_a_new_complete_speed_history():
  policy = qualified()
  assert_default_low_speed(update(policy, 227))  # 350 ms after the last update.
  assert_default_low_speed(run(policy, first=228, last=426))  # Only 9.95 s since recovery.
  assert run(policy, first=427, last=467)[1] == pytest.approx((4.0, 4.0))


@pytest.mark.parametrize('speed', [math.nan, math.inf, -math.inf, -.01])
def test_invalid_ego_speed_returns_conservative_six_metre_targets(speed):
  policy = qualified()
  result = update(policy, 221, speed=speed, observations=radar(221))
  assert result[0] == pytest.approx(6.0)
  assert result[1] == pytest.approx((6.0, 6.0))


def test_rewritten_held_car_state_speed_invalidates_the_old_history():
  policy = qualified()
  result = update(policy, 221, speed=.31, source_ns=stamp(220))
  assert_default_low_speed(result)


def test_rewritten_held_radar_is_not_a_new_stable_observation():
  policy = qualified()
  observations = radar(221)
  observations.leadOne.modelProb = .99  # Same timestamp, different actual payload.
  result = update(policy, 221, radar_ns=stamp(220), observations=observations)
  assert result[1][0] == pytest.approx(6.0)


def test_no_new_activation_when_standstill_or_non_pid_gate_is_closed():
  policy = LowSpeedStopDistance()
  # The caller supplies this false gate for standstill or non-pid longState.
  assert_default_low_speed(run(policy, last=260, allow_activation=False))
  assert run(policy, first=261, last=301, allow_activation=True)[1] == pytest.approx((4.0, 4.0))


def test_closing_activation_gate_immediately_revokes_previously_active_targets():
  policy = qualified()
  assert_default_low_speed(update(policy, 221, allow_activation=False))
  assert_default_low_speed(run(policy, first=222, last=260, allow_activation=False))


def test_reopening_activation_gate_reramps_without_requiring_another_ten_seconds():
  policy = qualified()
  result = run(policy, first=221, last=260, allow_activation=False)
  assert_default_low_speed(result)
  first = update(policy, 261, allow_activation=True)
  assert 3.0 < first[0] <= 4.0
  assert all(4.0 < value <= 6.0 for value in first[1])
  halfway = run(policy, first=262, last=265, allow_activation=True)
  assert 3.0 < halfway[0] < 4.0
  assert all(4.0 < value < 6.0 for value in halfway[1])
  complete = run(policy, first=266, last=272, allow_activation=True)
  assert complete[0] == pytest.approx(3.0)
  assert complete[1] == pytest.approx((4.0, 4.0))


def test_regained_lead_cannot_reactivate_while_activation_gate_remains_closed():
  policy = qualified()
  observations = radar(221)
  observations.leadOne.modelProb = .5
  assert_default_low_speed(update(policy, 221, observations=observations, allow_activation=False))
  assert_default_low_speed(run(policy, first=222, last=260, allow_activation=False))


def test_permitted_false_clears_active_targets_and_does_not_latch_permission():
  policy = qualified()
  assert_default_low_speed(update(policy, 221, permitted=False))
  assert_default_low_speed(run(policy, first=222, last=462, allow_activation=False))


def test_explicit_clear_discards_prior_qualification():
  policy = qualified()
  policy.clear()
  assert_default_low_speed(update(policy, 221))


def test_constant_point_eight_mps_lead_never_creates_a_catchup_target():
  policy = LowSpeedStopDistance()
  for frame in range(601):
    # Thirty seconds spans both history completion and many stability/ramp
    # durations. Time alone must not change the requested stopping distance.
    assert_default_low_speed(update(policy, frame, observations=radar(frame, lead_speed=.8)))


@pytest.mark.parametrize('lead_speed,acceleration,eligible', [
  (.3, 0.0, True), (.300001, 0.0, False),
  (.8, -.05, True), (.8, -.049999, False),
])
def test_stopping_approach_evidence_thresholds_are_inclusive(lead_speed, acceleration, eligible):
  policy = LowSpeedStopDistance()
  for frame in range(241):
    observations = radar(frame, lead_speed=lead_speed)
    observations.leadOne.aLeadK = observations.leadTwo.aLeadK = acceleration
    result = update(policy, frame, observations=observations)
  if eligible:
    assert result[0] == pytest.approx(3.0)
    assert result[1] == pytest.approx((4.0, 4.0))
  else:
    assert_default_low_speed(result)


def test_gentle_lead_deceleration_requires_confirmation_before_gradual_entry():
  policy = cruising_history()
  distance, previous_lead_speed = lead(220, lead_speed=.8).dRel, .8
  requests = []
  for frame in range(221, 242):
    lead_speed = .8 - .1 * (frame - 220) * .05
    distance += .5 * ((previous_lead_speed - .3) + (lead_speed - .3)) * .05
    observations = radar(frame, lead_speed=lead_speed)
    observations.leadOne.dRel = distance
    observations.leadTwo.dRel = distance + 30.0
    observations.leadOne.aLeadK = observations.leadTwo.aLeadK = -.1
    result = update(policy, frame, observations=observations)
    if frame <= 231:  # At most 500 ms of newly continuous approach evidence.
      assert_default_low_speed(result)
    elif frame < 241:
      assert 3.0 < result[0] < 4.0
      assert all(4.0 < value < 6.0 for value in result[1])
    requests.append(result[0])
    previous_lead_speed = lead_speed
  assert result[0] == pytest.approx(3.0)
  assert result[1] == pytest.approx((4.0, 4.0))
  assert all(after <= before for before, after in zip(requests, requests[1:], strict=False))


def test_one_frame_deceleration_noise_cannot_activate_compact_target():
  policy = cruising_history()
  assert_default_low_speed(update(policy, 221, observations=estimated_approach_radar(221, -.05)))
  for frame in range(222, 262):
    assert_default_low_speed(update(policy, frame, observations=estimated_approach_radar(frame, 0.0)))


def test_briefly_weakened_evidence_freezes_but_cannot_increase_fraction():
  policy, before = halfway_through_approach_ramp()
  for frame in range(237, 242):
    result = update(policy, frame, observations=estimated_approach_radar(frame, -.049))
    assert result[0] == pytest.approx(before[0])
    assert result[1] == pytest.approx(before[1])


def test_grace_expires_after_500_ms_since_last_confirmed_approach():
  policy, before = halfway_through_approach_ramp()
  for frame in range(237, 247):
    result = update(policy, frame, observations=estimated_approach_radar(frame, 0.0))
    assert result[0] == pytest.approx(before[0])
    assert result[1] == pytest.approx(before[1])
  # All streams remain fresh. Only the approach evidence is 550 ms old.
  assert_default_low_speed(update(policy, 247, observations=estimated_approach_radar(247, 0.0)))


def test_alternating_single_frame_noise_does_not_renew_the_grace_deadline():
  policy, before = halfway_through_approach_ramp()
  for frame in range(237, 278):
    acceleration = -.05 if frame % 2 == 0 else 0.0
    result = update(policy, frame, observations=estimated_approach_radar(frame, acceleration))
    if frame <= 246:
      assert result[0] == pytest.approx(before[0])
      assert result[1] == pytest.approx(before[1])
    else:
      assert_default_low_speed(result)


def test_returning_approach_needs_new_continuous_confirmation_then_reramp():
  policy, before = halfway_through_approach_ramp()
  update(policy, 237, observations=estimated_approach_radar(237, 0.0))
  for frame in range(238, 247):
    result = update(policy, frame, observations=estimated_approach_radar(frame))
    assert result[0] == pytest.approx(before[0])
    assert result[1] == pytest.approx(before[1])
  # The old grace expires before the new 500 ms run is complete. Completing
  # confirmation must not restore the old fraction as a one-frame jump.
  for frame in (247, 248):
    assert_default_low_speed(update(policy, frame, observations=estimated_approach_radar(frame)))
  first_new = update(policy, 249, observations=estimated_approach_radar(249))
  assert 3.0 < first_new[0] < 4.0
  assert all(4.0 < value < 6.0 for value in first_new[1])
  for frame in range(250, 259):
    result = update(policy, frame, observations=estimated_approach_radar(frame))
  assert result[0] == pytest.approx(3.0)
  assert result[1] == pytest.approx((4.0, 4.0))


@pytest.mark.parametrize('fault', ['distance_jump', 'target_loss', 'confidence', 'nonfinite', 'strong_acceleration',
                                   'radar_rollback', 'source_rollback', 'message_gap'])
def test_grace_does_not_mask_target_quality_or_time_failures(fault):
  policy, _ = halfway_through_approach_ramp()
  update(policy, 237, observations=estimated_approach_radar(237, 0.0))
  frame = 244 if fault == 'message_gap' else 238
  observations = estimated_approach_radar(frame, 0.0)
  changes = {}
  if fault == 'distance_jump':
    observations.leadOne.dRel += 2.0
  elif fault == 'target_loss':
    observations.leadOne.present = observations.leadTwo.present = False
  elif fault == 'confidence':
    observations.leadOne.modelProb = .97
  elif fault == 'nonfinite':
    observations.leadOne.vLead = math.nan
  elif fault == 'strong_acceleration':
    observations.leadOne.aLeadK = .201
  elif fault == 'radar_rollback':
    changes['radar_ns'] = stamp(236)
  elif fault == 'source_rollback':
    changes['source_ns'] = stamp(236)
  assert_default_low_speed(update(policy, frame, observations=observations, **changes))


@pytest.mark.parametrize('boundary,expected_comfort,expected_risk', [(.3, 3.0, 4.0), (1.2, 4.0, 6.0)])
def test_terminal_speed_blend_is_continuous_at_both_boundaries(boundary, expected_comfort, expected_risk):
  results = [run(LowSpeedStopDistance(), last=240, speed=speed)
             for speed in (boundary - 1e-7, boundary, boundary + 1e-7)]
  assert results[1][0] == pytest.approx(expected_comfort)
  assert results[1][1] == pytest.approx((expected_risk, expected_risk))
  for result in results:
    assert abs(result[0] - expected_comfort) < 1e-6
    assert all(abs(value - expected_risk) < 1e-6 for value in result[1])
  assert results[0][0] <= results[1][0] <= results[2][0]


@pytest.mark.parametrize('initial_speed,expected_comfort,expected_risk', [(.75, 3.5, 5.0), (1.2, 4.0, 6.0)])
def test_weakened_approach_cannot_shorten_target_merely_because_ego_slows(initial_speed, expected_comfort, expected_risk):
  policy = LowSpeedStopDistance()
  for frame in range(221):
    observations = radar(frame, initial_speed, lead_speed=.8)
    observations.leadOne.aLeadK = observations.leadTwo.aLeadK = -.1
    result = update(policy, frame, speed=initial_speed, observations=observations)
  assert result[0] == pytest.approx(expected_comfort)
  assert result[1] == pytest.approx((expected_risk, expected_risk))
  distance = lead(220, initial_speed, lead_speed=.8).dRel
  previous_speed = initial_speed
  for frame in range(221, 231):
    speed = max(.3, initial_speed - (initial_speed - .3) * (frame - 220) / 9.0)
    distance += .5 * ((.8 - previous_speed) + (.8 - speed)) * .05
    observations = radar(frame, speed, lead_speed=.8)
    observations.leadOne.dRel = distance
    observations.leadTwo.dRel = distance + 30.0
    # vLead remains above .3 and the acceleration estimate no longer says
    # braking. The observations are still fresh, valid and continuous.
    result = update(policy, frame, speed=speed, observations=observations)
    assert result[0] == pytest.approx(expected_comfort)
    assert result[1] == pytest.approx((expected_risk, expected_risk))
    previous_speed = speed


def test_second_slot_absent_from_start_does_not_repeatedly_reset_single_lead():
  policy = LowSpeedStopDistance()
  for frame in range(321):
    observations = radar(frame)
    observations.leadTwo.present = False
    result = update(policy, frame, observations=observations)
    if frame >= 220:
      assert result[0] == pytest.approx(3.0)
      assert result[1] == pytest.approx((4.0, 6.0))


@pytest.mark.parametrize('warm_speed', [.3, 1.5])
@pytest.mark.parametrize('fault', ['participating_loss', 'quality_failure', 'slot_swap', 'new_near_slot', 'radar_rollback'])
def test_armed_target_failure_requires_new_history_even_before_geometry_reduction(warm_speed, fault):
  policy = LowSpeedStopDistance()
  initial_second_present = fault != 'new_near_slot'
  for frame in range(221):
    observations = radar(frame, warm_speed)
    observations.leadTwo.present = initial_second_present
    result = update(policy, frame, speed=warm_speed, observations=observations)
  if warm_speed == .3:
    assert result[0] == pytest.approx(3.0)
    assert result[1] == pytest.approx((4.0, 4.0 if initial_second_present else 6.0))
  else:
    # The same complete speed/approach history exists at 1.5 m/s, although
    # the terminal-speed blend has not yet relaxed geometric distances.
    assert_default_low_speed(result)
  observations = radar(221, warm_speed)
  observations.leadTwo.present = initial_second_present
  changes = {}
  if fault == 'participating_loss':
    observations.leadOne.present = False  # The second qualified lead survives.
  elif fault == 'quality_failure':
    observations.leadOne.modelProb = .97
  elif fault == 'slot_swap':
    observations.leadOne, observations.leadTwo = observations.leadTwo, observations.leadOne
  elif fault == 'new_near_slot':
    observations.leadTwo.present = True
    observations.leadTwo.dRel = 5.0
  elif fault == 'radar_rollback':
    changes['radar_ns'] = stamp(219)
  assert_default_low_speed(update(policy, 221, speed=warm_speed, observations=observations, **changes))
  # Recover every target's healthy fields, but retain the actual slot gaps.
  assert_fresh_ten_second_requalification(policy, 222,
      (observations.leadOne.dRel, observations.leadTwo.dRel), previous_speed=warm_speed)


def test_armed_radar_stream_gap_discards_all_previous_speed_history():
  policy = qualified()
  for frame in range(221, 226):
    result = update(policy, frame, radar_ns=stamp(220), observations=radar(220))
    assert result[1] == pytest.approx((4.0, 4.0))
  # Source/consumer updates stayed at 50 ms; only the next distinct radar
  # publication is 300 ms after the previous one.
  observations = radar(226)
  assert_default_low_speed(update(policy, 226, observations=observations))
  assert_fresh_ten_second_requalification(policy, 227,
      (observations.leadOne.dRel, observations.leadTwo.dRel))


def test_held_fault_time_car_state_cannot_seed_the_new_ten_second_window():
  policy = qualified()
  observations = radar(221)
  observations.leadOne.modelProb = .97
  assert_default_low_speed(update(policy, 221, observations=observations))
  # The next consumer/radar messages are fresh, but CS is still the failed
  # transaction's publication. It must not become recovery's time zero.
  held_source_observations = radar(222)
  assert_default_low_speed(update(policy, 222, source_ns=stamp(221), observations=held_source_observations))
  assert_fresh_ten_second_requalification(policy, 223,
      (held_source_observations.leadOne.dRel, held_source_observations.leadTwo.dRel))


@pytest.mark.parametrize('withdrawal', ['activation_closed', 'approach_grace_expired'])
def test_prior_actual_compaction_still_requires_new_history_if_target_is_lost_after_withdrawal(withdrawal):
  if withdrawal == 'activation_closed':
    policy = qualified()  # Actual 3/4 m targets were applied before HOLD.
    assert_default_low_speed(run(policy, first=221, last=240, allow_activation=False))
    fault_frame = 241
    observations = radar(fault_frame)
  else:
    policy, applied = halfway_through_approach_ramp()
    assert applied[0] < 4.0 and applied[1][0] < 6.0
    for frame in range(237, 248):
      withdrawn = update(policy, frame, observations=estimated_approach_radar(frame, 0.0))
    assert_default_low_speed(withdrawn)  # No current fraction remains after grace expires.
    fault_frame = 248
    observations = estimated_approach_radar(fault_frame, 0.0)
  # A surviving second lead cannot preserve history through loss of a target
  # that participated in earlier actual compaction, even with fraction now 0.
  observations.leadOne.present = False
  assert_default_low_speed(update(policy, fault_frame, observations=observations,
                                  allow_activation=withdrawal != 'activation_closed'))
  assert_fresh_ten_second_requalification(policy, fault_frame + 1,
      (observations.leadOne.dRel, observations.leadTwo.dRel))
