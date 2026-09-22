"""Geometry and evidence gates only; no claim of delivered vehicle braking."""

from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.lead_closing import (
  ClosingRequestStep, LeadClosingConstraint, LeadClosingRequest, REQUEST_DECEL_RATE,
)


BASE_NS = 1_000_000_000
STEP_NS = 50_000_000


def lead(speed=25.0, closing=10.0, distance=70.0, **overrides):
  values = dict(present=True, dRel=distance, vRel=-closing, vLead=speed - closing, modelProb=0.99, aLeadK=0.0)
  values.update(overrides)
  return SimpleNamespace(**values)


def update(guard, sample, frame=0, **overrides):
  # Geometry-only fixtures start from an already published -2.5 request.
  # Request-establishment cases below supply their actual publication history.
  values = dict(now_ns=BASE_NS + frame * STEP_NS, source_ns=BASE_NS + frame * STEP_NS, eligible=True,
                v_ego=25.0, a_ego=0.0, leads=[sample], actuator_delay=0.5, t_follow=1.2,
                request=ClosingRequestStep(-2.5, 0.05))
  values.update(overrides)
  return guard.update(**values)


def qualify(guard=None, *, speed=25.0, closing=10.0, distance=70.0, **overrides):
  guard = guard if guard is not None else LeadClosingConstraint()
  result = None
  for frame in range(7):
    sample = lead(speed, closing, distance + closing * (6 - frame) * 0.05)
    result = update(guard, sample, frame, v_ego=speed, **overrides)
  return guard, result


def test_constant_speed_geometry_has_a_hand_calculated_answer():
  _, result = qualify()
  assert result is not None
  assert result.clearance == 15.0
  assert result.remaining_gap == 45.5
  assert result.required_decel == pytest.approx(100.0 / 91.0)
  assert result.accel == -result.required_decel
  assert result.qualified_duration == 0.3
  assert not result.infeasible
  # Relative distance used after the budgeted delay is u^2/(2*b).
  assert result.remaining_gap - 10.0 ** 2 / (2.0 * -result.accel) == pytest.approx(0.0)


@pytest.mark.parametrize('speed_kph', [50.0, 60.0, 80.0, 100.0, 120.0, 140.0])
def test_all_requested_speed_bands_can_act_without_an_existing_brake_request(speed_kph):
  _, result = qualify(speed=speed_kph / 3.6, closing=5.0, distance=35.0 + (speed_kph - 50.0) / 3.6, a_ego=0.0,
                      request=ClosingRequestStep(0.0, 0.05))
  assert result is not None and -2.5 <= result.accel < 0.0
  assert result.accel >= -0.125


def test_smaller_gap_and_larger_closing_speed_never_relax_the_bound():
  distance_results = [qualify(distance=distance)[1].accel for distance in (80.0, 70.0, 60.0, 50.0)]
  closing_results = [qualify(closing=closing)[1].accel for closing in (4.0, 6.0, 8.0, 10.0)]
  assert all(after <= before for before, after in zip(distance_results, distance_results[1:]))
  assert all(after <= before for before, after in zip(closing_results, closing_results[1:]))


def test_more_delay_or_positive_ego_acceleration_never_relaxes_the_bound():
  delays = [qualify(actuator_delay=delay)[1].accel for delay in (0.0, 0.25, 0.5, 1.0)]
  acceleration = [qualify(a_ego=value)[1].accel for value in (0.0, 0.5, 1.0, 2.0)]
  assert all(after <= before for before, after in zip(delays, delays[1:]))
  assert all(after <= before for before, after in zip(acceleration, acceleration[1:]))


def test_positive_acceleration_consumes_gap_and_increases_response_closing_speed():
  _, result = qualify(a_ego=1.0)
  assert result.remaining_gap == pytest.approx(45.04875)
  assert result.required_decel == pytest.approx(119.9025 / 90.0975)


def test_negative_measured_acceleration_cannot_turn_the_constraint_into_acceleration():
  results = [qualify(a_ego=value)[1] for value in (0.0, -1.0, -3.5, -10.0)]
  assert all(result == results[0] for result in results)
  assert results[0].accel < 0.0


@pytest.mark.parametrize('distance,remaining_positive', [(35.0, True), (20.0, False)])
def test_saturation_and_exhausted_gap_are_explicitly_infeasible(distance, remaining_positive):
  _, result = qualify(distance=distance)
  assert result.accel == -2.5 and result.infeasible
  assert (result.remaining_gap > 0.0) == remaining_positive
  assert result.required_decel > 3.5
  assert math.isfinite(result.required_decel) == remaining_positive


def test_hard_clearance_is_fifteen_meters_even_for_a_nearly_stopped_lead():
  _, result = qualify(speed=20.0, closing=18.0, distance=70.0)
  assert result.clearance == 15.0


def test_desired_time_gap_does_not_inflate_the_hard_clearance():
  results = [qualify(closing=5.0, distance=40.0, t_follow=time_gap)[1] for time_gap in (0.5, 1.2, 3.0)]
  assert all(result == results[0] for result in results)
  assert results[0].clearance == 15.0
  # Personality can still admit the same geometry at a different distance.
  assert qualify(closing=1.2, distance=40.0, t_follow=0.5)[1] is None
  assert qualify(closing=1.2, distance=40.0, t_follow=1.2)[1] is not None


def test_small_closing_speed_at_thirty_meters_does_not_request_hard_braking():
  _, result = qualify(speed=25.0, closing=1.2, distance=30.0)
  assert result.clearance == 15.0
  assert result.remaining_gap == pytest.approx(13.86)
  assert result.accel == pytest.approx(-1.44 / 27.72)
  assert not result.infeasible


def test_comfort_braking_enters_gradually_even_when_the_gap_is_already_below_fifteen():
  for closing, expected in ((1.01, -0.0125), (1.2, -0.25), (2.0, -1.25), (3.0, -2.5), (10.0, -2.5)):
    _, result = qualify(closing=closing, distance=14.0)
    assert result.accel == pytest.approx(expected)
    assert result.infeasible and math.isinf(result.required_decel)


def test_comfort_saturation_is_distinct_from_exceeding_the_total_braking_budget():
  _, result = qualify(distance=40.0)
  assert 2.5 < result.required_decel < 3.5
  assert result.accel == -2.5 and not result.infeasible
  # The diagnostic refers to the total budget. This comfort request alone is
  # deliberately weaker than the geometric demand and cannot certify 15 m.
  assert result.remaining_gap - 10.0 ** 2 / (2.0 * -result.accel) < 0.0


def test_qualification_requires_elapsed_source_time_and_distinct_samples():
  guard = LeadClosingConstraint()
  for frame in range(6):
    assert update(guard, lead(distance=73.0 - frame * 0.5), frame) is None
  assert update(guard, lead(), 6) is not None
  guard.reset()
  for elapsed in (0, 100_000_000, 200_000_000, 300_000_000):
    assert update(guard, lead(distance=73.0 - 10.0 * elapsed / 1e9),
                  now_ns=BASE_NS + elapsed, source_ns=BASE_NS + elapsed) is None
  assert update(guard, lead(distance=69.0), now_ns=BASE_NS + 400_000_000, source_ns=BASE_NS + 400_000_000) is not None


def test_repeated_source_does_not_accumulate_qualification_and_then_expires():
  guard = LeadClosingConstraint()
  for frame in range(10):
    assert update(guard, lead(), frame, source_ns=BASE_NS) is None


def test_qualified_repeated_source_can_only_reuse_unchanged_fresh_geometry():
  guard, result = qualify()
  assert update(guard, lead(), 7, source_ns=BASE_NS + 6 * STEP_NS) == result
  assert update(guard, lead(distance=60.0), 8, source_ns=BASE_NS + 6 * STEP_NS) is None


def test_source_age_boundary_is_inclusive_but_never_extends_qualification():
  guard, result = qualify()
  source = BASE_NS + 6 * STEP_NS
  assert update(guard, lead(), now_ns=source + 250_000_000, source_ns=source) == result
  assert update(guard, lead(), now_ns=source + 250_000_001, source_ns=source) is None


def test_variable_sample_cadence_can_qualify_with_five_distinct_epochs():
  guard = LeadClosingConstraint()
  for elapsed in (0, 70_000_000, 130_000_000, 210_000_000, 300_000_000):
    result = update(guard, lead(distance=73.0 - 10.0 * elapsed / 1e9),
                    now_ns=BASE_NS + elapsed, source_ns=BASE_NS + elapsed)
    assert (result is not None) == (elapsed == 300_000_000)


@pytest.mark.parametrize('clock', ['source_ns', 'now_ns'])
def test_clock_rollback_clears_history(clock):
  guard, _ = qualify()
  overrides = {clock: BASE_NS + 5 * STEP_NS}
  if clock == 'now_ns':
    overrides['source_ns'] = BASE_NS + 5 * STEP_NS
  assert update(guard, lead(), 7, **overrides) is None
  assert update(guard, lead(distance=69.0), 8) is None


def test_source_gap_and_source_age_clear_history():
  guard, _ = qualify()
  assert update(guard, lead(distance=67.5), 11) is None
  guard, _ = qualify()
  assert update(guard, lead(), 12, source_ns=BASE_NS + 6 * STEP_NS) is None
  assert update(guard, lead(distance=66.5), 13) is None


@pytest.mark.parametrize('timestamps', [dict(source_ns=0), dict(source_ns=True), dict(source_ns=1.0),
                                       dict(source_ns=BASE_NS + 1), dict(now_ns=0), dict(now_ns=float('inf'))])
def test_invalid_timestamps_are_not_evidence(timestamps):
  assert update(LeadClosingConstraint(), lead(), **timestamps) is None


def test_single_range_jump_restarts_the_full_qualification():
  guard, _ = qualify()
  assert update(guard, lead(distance=50.0), 7) is None
  for frame in range(8, 13):
    assert update(guard, lead(distance=50.0 - 0.5 * (frame - 7)), frame) is None
  assert update(guard, lead(distance=47.0), 13) is not None


def test_single_speed_jump_and_rebound_restart_the_full_qualification():
  # Both speeds jump; then relative speed alone; then absolute lead speed alone.
  # Each case passes the instantaneous vLead - vEgo == vRel consistency check.
  for speed, closing in ((25.0, 20.0), (35.0, 11.2), (35.0, 1.2)):
    guard, before = qualify(closing=1.2, distance=30.0)
    assert before.accel == pytest.approx(-1.44 / 27.72)
    # The first case's trapezoidal range residual is only 0.47 m, despite an
    # 18.8 m/s velocity jump. Range continuity alone would still admit it.
    jumped = lead(speed=speed, closing=closing, distance=29.94)
    assert update(guard, jumped, 7, v_ego=speed) is None, (speed, closing)
    assert update(guard, lead(closing=1.2, distance=29.88), 8) is None
    for frame in range(9, 14):
      assert update(guard, lead(closing=1.2, distance=30.0 - 0.06 * (frame - 6)), frame) is None
    restored = update(guard, lead(closing=1.2, distance=29.52), 14)
    assert restored is not None and restored.qualified_duration == 0.3


def test_sustained_eight_meter_per_second_squared_lead_braking_can_qualify_and_continue():
  guard = LeadClosingConstraint()
  for frame in range(21):
    elapsed = frame * 0.05
    # vLead = 23.8 - 8*t, and dRel is the integral of -1.2 - 8*t.
    sample = lead(closing=1.2 + 8.0 * elapsed, distance=30.0 - 1.2 * elapsed - 4.0 * elapsed ** 2, aLeadK=-8.0)
    result = update(guard, sample, frame)
    if frame < 6:
      assert result is None
    else:
      assert result is not None
      assert result.qualified_duration == pytest.approx(elapsed)
      assert -2.5 <= result.accel < 0.0


def test_small_but_persistent_range_velocity_mismatch_cannot_qualify():
  guard = LeadClosingConstraint()
  # Each residual is 0.8 m (< the per-step limit), but six sum to 4.8 m.
  for frame in range(20):
    assert update(guard, lead(distance=70.0 - 1.3 * frame), frame) is None


@pytest.mark.parametrize('bad', [dict(present=False), dict(modelProb=0.94), dict(vLead=25.0),
                                dict(vRel=-1.0), dict(vLead=-1.0), dict(dRel=float('nan'))])
def test_untrusted_lead_removes_previous_qualification_and_active_state(bad):
  guard, _ = qualify()
  assert update(guard, lead(distance=69.5, **bad), 7) is None
  assert update(guard, lead(distance=69.0), 8) is None


def test_missing_lead_and_empty_lead_list_clear_the_affected_slot():
  for leads in ([], [None]):
    guard, _ = qualify()
    assert update(guard, None, 7, leads=leads) is None
    assert update(guard, lead(distance=69.0), 8) is None


def test_driver_ineligibility_and_explicit_reset_require_fresh_qualification():
  guard, _ = qualify()
  assert update(guard, lead(distance=69.5), 7, eligible=False) is None
  assert update(guard, lead(distance=69.0), 8) is None
  guard, _ = qualify()
  guard.reset()
  guard.reset()
  assert update(guard, lead(distance=69.5), 7) is None


def test_two_slots_are_independent_choose_the_stronger_and_keep_first_on_tie():
  guard = LeadClosingConstraint()
  for frame in range(7):
    samples = [lead(distance=73.0 - frame * 0.5), lead(distance=63.0 - frame * 0.5)]
    result = update(guard, None, frame, leads=samples)
  assert result.lead_index == 1
  # Losing one slot does not revoke the other qualified observation.
  result = update(guard, None, 7, leads=[lead(distance=69.5), None])
  assert result.lead_index == 0
  guard.reset()
  for frame in range(7):
    sample = lead(distance=73.0 - frame * 0.5)
    result = update(guard, None, frame, leads=[sample, deepcopy(sample)])
  assert result.lead_index == 0


def test_far_lead_can_qualify_before_it_enters_the_comfort_gap():
  guard, result = qualify(speed=20.0, closing=5.0, distance=66.0)
  # Comfort is 65 m; entering it should not restart evidence collection.
  assert result is None
  for frame in range(7, 11):
    result = update(guard, lead(speed=20.0, closing=5.0, distance=66.0 - (frame - 6) * 0.25), frame, v_ego=20.0)
  assert result is not None and result.qualified_duration == 0.5


def test_active_constraint_persists_below_entry_speed_but_releases_below_eight():
  guard, result = qualify(speed=14.0, closing=5.0, distance=26.0)
  assert result is not None
  # Decelerate through the entry boundary with continuous lead velocities.
  for step in range(1, 13):
    speed = 14.0 - 0.5 * step
    sample = lead(speed=speed, closing=5.0, distance=26.0 - 0.25 * step)
    assert update(guard, sample, 6 + step, v_ego=speed) is not None
  assert update(guard, lead(speed=7.99, closing=5.0, distance=22.75), 19, v_ego=7.99) is None
  assert update(guard, lead(speed=13.5, closing=5.0, distance=22.5), 20, v_ego=13.5) is None
  # A never-active target cannot start braking below 50 km/h.
  _, result = qualify(speed=13.5, closing=5.0, distance=30.0)
  assert result is None


def test_matching_lead_speed_releases_the_active_constraint():
  guard, _ = qualify()
  assert update(guard, lead(closing=1.0, distance=69.75), 7) is None
  assert update(guard, lead(distance=69.5), 8) is None


def test_leaving_the_comfort_domain_revokes_below_entry_hysteresis():
  guard, result = qualify(speed=14.0, closing=1.2, distance=33.0, t_follow=1.75)
  assert result is not None
  assert update(guard, lead(speed=13.5, closing=1.2, distance=32.94), 7, v_ego=13.5, t_follow=1.25) is None
  # The still-observed lead re-enters the comfort domain, but this is a new
  # activation below the entry threshold, not continuation of an active cap.
  assert update(guard, lead(speed=13.5, closing=1.2, distance=32.88), 8, v_ego=13.5, t_follow=1.75) is None


def test_lead_acceleration_jumps_have_no_effect_and_inputs_are_not_mutated():
  normal, altered = LeadClosingConstraint(), LeadClosingConstraint()
  for frame in range(7):
    sample = lead(distance=73.0 - frame * 0.5)
    noisy = deepcopy(sample)
    noisy.aLeadK = [float('nan'), float('inf'), -100.0, 100.0][frame % 4]
    before = dict(noisy.__dict__)
    assert update(normal, sample, frame) == update(altered, noisy, frame)
    assert noisy.__dict__ == before


def test_invalid_numeric_inputs_fail_closed_and_erase_previous_history():
  cases = [dict(v_ego=float('nan')), dict(v_ego=101.0), dict(v_ego=True), dict(v_ego=10 ** 1000),
           dict(a_ego=21.0), dict(a_ego=float('inf')), dict(actuator_delay=-0.1), dict(actuator_delay=1.01),
           dict(t_follow=0.0), dict(t_follow=3.01), dict(t_follow=None)]
  for bad in cases:
    guard, _ = qualify()
    assert update(guard, lead(distance=69.5), 7, **bad) is None
    assert update(guard, lead(distance=69.0), 8) is None
  for field, value in [('dRel', 10001.0), ('dRel', 0.0), ('vLead', float('inf')), ('vRel', -101.0),
                       ('modelProb', 1.01), ('modelProb', float('nan'))]:
    guard, _ = qualify()
    bad = lead(distance=69.5)
    setattr(bad, field, value)
    assert update(guard, bad, 7) is None
    assert update(guard, lead(distance=69.0), 8) is None


def test_request_history_requires_a_real_fresh_publication():
  request = LeadClosingRequest()
  assert request.step(BASE_NS) is None
  request.publish(BASE_NS, 0.4)
  step = request.step(BASE_NS + STEP_NS)
  assert step == ClosingRequestStep(0.4, 0.05)
  assert step.minimum_accel == pytest.approx(0.275)
  # Neither a second read nor an unissued computed target advances history.
  assert request.step(BASE_NS + STEP_NS) == step
  assert request.step(BASE_NS + 2 * STEP_NS).minimum_accel == pytest.approx(0.15)
  request.publish(BASE_NS + 2 * STEP_NS, -0.3)
  assert request.step(BASE_NS + 3 * STEP_NS).minimum_accel == pytest.approx(-0.425)


def test_republishing_one_unchanged_request_does_not_refresh_its_epoch():
  request = LeadClosingRequest()
  request.publish(BASE_NS, 0.0)
  request.publish(BASE_NS, 0.0)
  assert request.step(BASE_NS + 200_000_000).elapsed == 0.2
  assert request.step(BASE_NS + 200_000_001) is None


@pytest.mark.parametrize('bad', [None, float('nan'), float('inf'), True, -3.5001, 2.0001])
def test_invalid_published_request_erases_request_history(bad):
  request = LeadClosingRequest()
  request.publish(BASE_NS, -0.5)
  request.publish(BASE_NS + STEP_NS, bad)
  assert request.step(BASE_NS + 2 * STEP_NS) is None


@pytest.mark.parametrize('kind', ['same_epoch_changed', 'publication_rollback', 'computation_rollback', 'future', 'expired'])
def test_invalid_request_clocks_cannot_keep_a_building_constraint(kind):
  request = LeadClosingRequest()
  request.publish(BASE_NS, -0.5)
  if kind == 'same_epoch_changed':
    request.publish(BASE_NS, -1.0)
    now = BASE_NS + STEP_NS
  elif kind == 'publication_rollback':
    request.publish(BASE_NS - 1, -0.5)
    now = BASE_NS + STEP_NS
  elif kind == 'computation_rollback':
    assert request.step(BASE_NS + 2 * STEP_NS) is not None
    now = BASE_NS + STEP_NS
  elif kind == 'future':
    now = BASE_NS - 1
  else:
    now = BASE_NS + 200_000_001
  assert request.step(now) is None
  guard, _ = qualify()
  assert update(guard, lead(distance=69.5), 7, request=None) is None
  assert update(guard, lead(distance=69.0), 8) is None


@pytest.mark.parametrize('bad', [None, ClosingRequestStep(float('nan'), 0.05), ClosingRequestStep(True, 0.05),
                                ClosingRequestStep(2.01, 0.05), ClosingRequestStep(-3.51, 0.05),
                                ClosingRequestStep(0.0, 0.0), ClosingRequestStep(0.0, 0.200001)])
def test_invalid_request_step_revokes_the_lead_constraint(bad):
  guard, _ = qualify()
  assert update(guard, lead(distance=69.5), 7, request=bad) is None
  assert update(guard, lead(distance=69.0), 8) is None


def test_added_request_build_time_is_separate_from_actuator_allowance():
  _, result = qualify(request=ClosingRequestStep(2.0, 0.05), a_ego=-1.0)
  assert result.request_build_time == 1.8
  assert result.actuator_build_time == 0.4
  assert result.response_time == 2.75  # 0.5 delay + 0.05 planner + 0.4 actuator + 1.8 request
  # Reserve positive published effort even when the measured car is braking.
  assert result.remaining_gap == pytest.approx(70.0 - 15.0 - 10.0 * 2.75 - 2.75 ** 2)
  assert result.accel == 1.875  # A min-only ceiling; never a new acceleration request.
  assert result.geometric_accel == -2.5


def test_establishing_a_constraint_is_bounded_from_each_published_request():
  guard, request = LeadClosingConstraint(), LeadClosingRequest()
  previous = 0.0
  request.publish(BASE_NS - STEP_NS, previous)
  first_frame = None
  for frame in range(26):
    now = BASE_NS + frame * STEP_NS
    sample = lead(closing=5.0, distance=15.5 - 0.25 * frame)
    result = update(guard, sample, frame, request=request.step(now))
    if result is None:
      assert frame < 6
      output = 0.0
    else:
      first_frame = frame if first_frame is None else first_frame
      assert result.geometric_accel == -2.5
      assert result.accel == pytest.approx(max(-2.5, previous - REQUEST_DECEL_RATE * 0.05))
      assert result.request_build_time == pytest.approx(max(previous + 2.5, 0.0) / REQUEST_DECEL_RATE)
      output = result.accel
    request.publish(now, output)
    previous = output
  assert first_frame == 6
  assert previous == -2.5


def test_slot_switch_uses_the_same_latest_publication_and_loss_is_immediate():
  guard = LeadClosingConstraint()
  request = ClosingRequestStep(0.2, 0.05)
  for frame in range(7):
    result = update(guard, None, frame, leads=[lead(distance=73.0 - 0.5 * frame), lead(distance=63.0 - 0.5 * frame)],
                    request=request)
  assert result.lead_index == 1 and result.accel == pytest.approx(0.075)
  # Slot 0 has its own qualified history; it cannot inherit a stronger ramp
  # simply because the previously selected slot disappeared.
  request = ClosingRequestStep(result.accel, 0.05)
  result = update(guard, None, 7, leads=[lead(distance=69.5), None], request=request)
  assert result.lead_index == 0 and result.accel == pytest.approx(-0.05)
  assert update(guard, None, 8, leads=[None, None], request=ClosingRequestStep(result.accel, 0.05)) is None
  # An actual new lead observation needs a complete new qualification.
  assert update(guard, lead(distance=68.5), 9, request=ClosingRequestStep(-0.05, 0.05)) is None
