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


def lead(frame, ego_speed=1.0, lead_speed=1.0, initial_distance=150.0, **changes):
  relative_speed = lead_speed - ego_speed
  values = dict(present=True, dRel=initial_distance + relative_speed * frame * .05,
                vRel=relative_speed, vLead=lead_speed, aLeadK=0.0, modelProb=.995, yRel=0.0)
  values.update(changes)
  return SimpleNamespace(**values)


def radar(frame, speed=1.0, lead_speed=1.0):
  return SimpleNamespace(leadOne=lead(frame, speed, lead_speed),
                         leadTwo=lead(frame, speed, lead_speed, initial_distance=180.0))


def update(policy, frame, *, speed=1.0, observations=AUTO, now_ns=AUTO,
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
  assert update(policy, 221, speed=40.0 / 3.6)[1] == pytest.approx((6.0, 6.0))
  # Nine seconds of slow driving after that high-speed sample are insufficient.
  assert_default_low_speed(run(policy, first=222, last=401))
  assert_default_low_speed(run(policy, first=402, last=420))
  # The high-speed observation has now left a complete recent 10 s window.
  assert run(policy, first=421, last=461)[1] == pytest.approx((4.0, 4.0))


def test_exactly_thirty_kmh_in_recent_history_is_not_below_thirty():
  policy = LowSpeedStopDistance()
  update(policy, 0, speed=30.0 / 3.6)
  assert_default_low_speed(run(policy, first=1, last=200))
  assert run(policy, first=201, last=241)[1] == pytest.approx((4.0, 4.0))


@pytest.mark.parametrize('speed_kmh, comfort, risk', [
  (0.0, 3.0, 4.0),
  (15.0, 3.0, 4.0),
  (22.5, 4.5, 5.0),
  (30.0, 6.0, 6.0),
])
def test_speed_profile_boundaries_and_midpoint(speed_kmh, comfort, risk):
  result = run(LowSpeedStopDistance(), last=240, speed=speed_kmh / 3.6)
  assert result[0] == pytest.approx(comfort)
  assert result[1] == pytest.approx((risk, risk))


def test_zero_speed_with_closed_activation_gate_returns_default_targets():
  policy = qualified()
  # Ego and slow lead decelerate together: constant gap, matching v/vRel and
  # a bounded lead deceleration. No discontinuous target jump is fabricated.
  for frame in range(221, 282):
    speed = max(0.0, 1.0 - (frame - 220) / 60.0)
    observations = radar(frame, speed, lead_speed=speed)
    observations.leadOne.aLeadK = observations.leadTwo.aLeadK = -1.0 / 3.0
    result = update(policy, frame, speed=speed, observations=observations, allow_activation=speed > 0.0)
    if speed > 0.0:
      assert result[0] == pytest.approx(3.0)
      assert result[1] == pytest.approx((4.0, 4.0))
    else:
      assert_default_low_speed(result)


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


def test_absent_slot_keeps_its_default_without_blocking_qualified_present_lead():
  policy = qualified()
  observations = radar(221)
  observations.leadOne.present = False
  comfort, distances = update(policy, 221, observations=observations)
  assert comfort == pytest.approx(3.0)
  assert distances == pytest.approx((6.0, 4.0))


def test_swapped_distinct_slots_restart_observation_qualification():
  policy = qualified()
  observations = radar(221)
  observations.leadOne, observations.leadTwo = observations.leadTwo, observations.leadOne
  # Both distance slots jump by 30 m, inconsistent with zero relative speed.
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
    result = update(policy, frame, observations=radar(frame, lead_speed=lead_speed))
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
    result = update(policy, frame, **{field: stamp(220)})
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
  result = update(policy, 221, speed=1.01, source_ns=stamp(220))
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
