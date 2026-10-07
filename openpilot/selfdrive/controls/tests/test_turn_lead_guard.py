"""State and source-time boundaries for the turn lead-loss acceleration cap."""
import math
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.turn_lead_guard import HOLD_ACCEL_MAX, TurnLeadGuard


NS = 1_000_000_000
STEP_NS = 50_000_000


def lead(*, present=True, **values):
  fields = dict(present=present, dRel=20.0, vRel=0.0, vLead=5.0, aLeadK=0.0, modelProb=.9, yRel=0.0)
  fields.update(values)
  return SimpleNamespace(**fields)


class Harness:
  def __init__(self):
    self.guard = TurnLeadGuard()
    self.now = 10 * NS

  def step(self, *, dt=STEP_NS, **values):
    self.now += dt
    inputs = dict(now_ns=self.now, model_ns=self.now, radar_ns=self.now, enabled=True,
                  speed=5.0, standstill=False, steering_angle=0.0, steer_ratio=16.33, wheelbase=2.83,
                  model_curvature=.04, lead_one=lead(present=False), lead_two=lead(present=False), ceiling_max=2.0)
    inputs.update(values)
    return self.guard.update(**inputs)

  def lose_lead(self, *, lead_index=0):
    assert self.step(**{('lead_one', 'lead_two')[lead_index]: lead()}) is None
    assert self.guard.armed and not self.guard.lost
    assert self.step() == 0.0
    assert self.guard.binding and self.guard.lost

  def recover_lead(self, *, lead_index=0):
    slot = ('lead_one', 'lead_two')[lead_index]
    for frame in range(5):
      assert self.step(**{slot: lead()}) == 0.0
      assert self.guard.lost == (frame < 4)
    assert self.guard.armed and self.guard.binding


def test_empty_road_turn_never_arms_or_limits_acceleration():
  h = Harness()
  for _ in range(240):
    assert h.step() is None
    assert not h.guard.armed and not h.guard.lost and not h.guard.binding


@pytest.mark.parametrize('lead_index', [0, 1])
def test_either_slot_establishes_and_recovers_following_context(lead_index):
  h = Harness()
  h.lose_lead(lead_index=lead_index)
  h.recover_lead(lead_index=lead_index)
  assert h.step(**{('lead_one', 'lead_two')[lead_index]: lead()}) == pytest.approx(.05)


@pytest.mark.parametrize('lead_index', [0, 1])
def test_one_present_weak_slot_prevents_first_loss_but_cannot_release_existing_loss(lead_index):
  h = Harness()
  slot = ('lead_one', 'lead_two')[lead_index]
  assert h.step(**{slot: lead()}) is None
  for _ in range(8):
    assert h.step(**{slot: lead(modelProb=.6)}) is None
    assert h.guard.armed and not h.guard.lost
  assert h.step() == 0.0
  for _ in range(8):
    assert h.step(**{slot: lead(modelProb=.6)}) == 0.0
    assert h.guard.lost


@pytest.mark.parametrize('values', [dict(modelProb=.799), dict(dRel=50.01), dict(dRel=0.0), dict(dRel=-1.0)])
def test_unqualified_lead_does_not_create_following_context(values):
  h = Harness()
  for _ in range(5):
    assert h.step(lead_one=lead(**values)) is None
  assert h.step() is None
  assert not h.guard.armed


def test_probability_and_distance_qualification_boundaries():
  h = Harness()
  assert h.step(lead_one=lead(modelProb=.8, dRel=50.0)) is None
  assert h.guard.armed
  assert h.step() == 0.0


@pytest.mark.parametrize(('curvature', 'armed'), [(0.0, False), (.025, False), (1.0 / 30.0 - 1e-6, False), (1.0 / 30.0, True)])
def test_only_sharp_turns_establish_following_context(curvature, armed):
  h = Harness()
  assert h.step(model_curvature=curvature, lead_one=lead()) is None
  assert h.guard.armed is armed
  assert h.step(model_curvature=curvature) == (0.0 if armed else None)


@pytest.mark.parametrize('expired', [False, True])
def test_recent_lead_uses_three_second_entry_window(expired):
  h = Harness()
  assert h.step(model_curvature=0.0, lead_one=lead()) is None
  for _ in range(59 + int(expired)):
    assert h.step(model_curvature=0.0) is None
  result = h.step()
  assert result == (None if expired else 0.0)
  assert h.guard.armed is not expired


def test_last_reliable_observation_is_dated_by_radar_source_not_consumer():
  h = Harness()
  source = h.now
  assert h.step(dt=200_000_000, model_curvature=0.0, radar_ns=source, lead_one=lead()) is None
  # The radar observation was already 200 ms old when first consumed.
  for _ in range(56):
    assert h.step(model_curvature=0.0) is None
  assert h.now - source == 3 * NS
  assert h.step() is None
  assert not h.guard.armed


def test_duplicate_radar_does_not_refresh_entry_history():
  h = Harness()
  assert h.step(model_curvature=0.0, lead_one=lead()) is None
  source = h.now
  for _ in range(4):
    assert h.step(model_curvature=0.0, radar_ns=source, lead_one=lead()) is None
  for _ in range(57):
    assert h.step(model_curvature=0.0) is None
  assert h.now - source == 3_050_000_000
  assert h.step() is None
  assert not h.guard.armed


def test_continuous_lost_turn_outlasts_history_and_nine_second_gap():
  h = Harness()
  h.lose_lead()
  for _ in range(240):
    assert h.step() == 0.0
    assert h.guard.armed and h.guard.lost


@pytest.mark.parametrize('speed', [0.0, .49, .5, 10.0, 10.01, 15.0])
def test_entry_speed_boundary_does_not_drop_an_armed_ceiling(speed):
  h = Harness()
  h.lose_lead()
  ceiling = h.step(speed=speed)
  assert 0.0 <= ceiling <= HOLD_ACCEL_MAX
  if speed >= 5.0:
    assert ceiling == 0.0
  assert h.guard.armed and h.guard.lost


def test_speed_at_first_loss_is_held_without_ratchet_or_acceleration_above_it():
  h = Harness()
  h.lose_lead()
  assert h.guard.hold_speed == 5.0
  previous = 0.0
  for speed in (4.95, 4.9, 4.8, 4.6, 4.3, 4.0, 3.5):
    ceiling = h.step(speed=speed)
    assert h.guard.hold_speed == 5.0
    assert 0.0 < ceiling <= HOLD_ACCEL_MAX
    assert ceiling - previous <= .05 + 1e-9
    previous = ceiling
  assert h.step(speed=5.0) == 0.0
  assert h.step(speed=5.1) == 0.0


def test_weak_lead_flicker_does_not_replace_captured_speed():
  h = Harness()
  h.lose_lead()
  for _ in range(5):
    assert h.step(speed=4.0, lead_one=lead(modelProb=.6)) == 0.0
    assert h.guard.hold_speed == 5.0
  assert h.step(speed=4.0) == 0.0
  assert h.guard.hold_speed == 5.0
  h.step(enabled=False)
  assert h.guard.hold_speed is None


def test_independent_braking_latches_out_speed_recovery_until_lead_reconfirmation():
  h = Harness()
  h.lose_lead()
  assert h.step(speed=4.8) > 0.0
  assert h.step(speed=4.0, independent_braking=True) == 0.0
  for _ in range(40):
    assert h.step(speed=3.0, independent_braking=False) == 0.0
  assert h.guard.speed_recovery_blocked and h.guard.lost
  h.recover_lead()
  assert not h.guard.speed_recovery_blocked
  assert h.step(speed=3.0) == 0.0
  assert h.guard.hold_speed == 3.0


@pytest.mark.parametrize('speed', [0.0, .5, 10.0, 20.0])
def test_speed_outside_entry_domain_cannot_arm(speed):
  h = Harness()
  assert h.step(speed=speed, lead_one=lead()) is None
  assert h.step(speed=speed) is None
  assert not h.guard.armed


@pytest.mark.parametrize('curvature_source', ['steering', 'model'])
@pytest.mark.parametrize('direction', [-1.0, 1.0])
def test_either_curvature_can_arm_and_both_must_clear(curvature_source, direction):
  h = Harness()
  curve = dict(model_curvature=0.0, steering_angle=0.0)
  if curvature_source == 'model':
    curve['model_curvature'] = direction * .04
  else:
    curve['steering_angle'] = direction * math.degrees(math.atan(2.83 * .04)) * 16.33
  assert h.step(lead_one=lead(), **curve) is None
  assert h.guard.armed
  for _ in range(25):
    assert h.step(**curve) == 0.0
    assert h.guard.armed


def test_intermediate_curvature_interrupts_exit_confirmation():
  h = Harness()
  h.lose_lead()
  for _ in range(20):
    assert h.step(model_curvature=.02) == 0.0
  assert h.guard.armed
  assert h.step(model_curvature=.025) == 0.0
  for _ in range(20):
    assert h.step(model_curvature=.02) == 0.0
    assert h.guard.armed
  assert h.step(model_curvature=.02) == 0.0
  assert not h.guard.armed and not h.guard.lost
  assert h.step(model_curvature=0.0) == pytest.approx(.05)


def test_confirmed_exit_clears_history_without_losing_gradual_release():
  h = Harness()
  h.lose_lead()
  for _ in range(21):
    assert h.step(model_curvature=0.0) == 0.0
  assert not h.guard.armed and not h.guard.lost and h.guard.binding
  assert h.step() == pytest.approx(.05)
  assert not h.guard.armed
  for _ in range(40):
    h.step()
  assert h.guard.accel_ceiling is None and not h.guard.binding


def test_straight_driving_still_records_lead_before_a_new_turn():
  h = Harness()
  h.lose_lead()
  for _ in range(21):
    h.step(model_curvature=0.0)
  for _ in range(40):
    h.step(model_curvature=0.0, lead_one=lead())
  assert not h.guard.armed
  assert h.step() == 0.0
  assert h.guard.armed and h.guard.lost


@pytest.mark.parametrize('present', [False, True])
def test_weak_or_absent_flicker_restarts_recovery_confirmation(present):
  h = Harness()
  h.lose_lead()
  for _ in range(3):
    assert h.step(lead_one=lead()) == 0.0
  assert h.step(lead_one=lead(present=present, modelProb=.6)) == 0.0
  assert h.guard.lost
  h.recover_lead()


def test_duplicate_radar_cannot_confirm_recovery_using_consumer_elapsed_time():
  h = Harness()
  h.lose_lead()
  assert h.step(lead_one=lead()) == 0.0
  source = h.now
  for _ in range(4):
    assert h.step(radar_ns=source, lead_one=lead()) == 0.0
    assert h.guard.lost
  assert h.step(lead_one=lead()) == 0.0
  assert not h.guard.lost
  assert h.step(lead_one=lead()) == pytest.approx(.05)


def test_recovery_ramp_uses_actual_elapsed_time_and_cannot_overshoot_global_limit():
  h = Harness()
  h.lose_lead()
  h.recover_lead()
  assert h.step(dt=20_000_000, lead_one=lead()) == pytest.approx(.02)
  assert h.step(dt=100_000_000, lead_one=lead()) == pytest.approx(.12)
  assert h.step(dt=130_000_000, lead_one=lead(), ceiling_max=.3) == pytest.approx(.25)
  assert h.step(dt=80_000_000, lead_one=lead(), ceiling_max=.3) is None
  assert not h.guard.binding and h.guard.armed


def test_recovered_turn_remains_armed_for_another_loss():
  h = Harness()
  h.lose_lead()
  h.recover_lead()
  for _ in range(45):
    h.step(lead_one=lead())
  assert not h.guard.binding and h.guard.armed
  assert h.step() == 0.0
  assert h.guard.lost


def test_loss_during_release_immediately_restores_zero_ceiling():
  h = Harness()
  h.lose_lead()
  h.recover_lead()
  assert h.step(lead_one=lead()) == pytest.approx(.05)
  assert h.step() == 0.0
  assert h.guard.lost


@pytest.mark.parametrize('values', [dict(enabled=False), dict(standstill=True), dict(enabled=1), dict(standstill=0)])
def test_inactive_standstill_and_non_boolean_flags_clear_all_context(values):
  h = Harness()
  h.lose_lead()
  assert h.step(**values) is None
  assert not h.guard.armed and not h.guard.lost and not h.guard.binding
  assert h.step() is None


@pytest.mark.parametrize('clock', ['now_ns', 'model_ns', 'radar_ns'])
@pytest.mark.parametrize('value', [True, False, 0, -1, 1.5, math.nan, math.inf, None])
def test_invalid_timestamps_clear_context(clock, value):
  h = Harness()
  h.lose_lead()
  assert h.step(**{clock: value}) is None
  assert not h.guard.armed and not h.guard.binding
  assert h.step() is None


@pytest.mark.parametrize('clock', ['model_ns', 'radar_ns'])
@pytest.mark.parametrize('offset', [-250_000_001, 1])
def test_stale_and_future_sources_clear_context(clock, offset):
  h = Harness()
  h.lose_lead()
  assert h.step(**{clock: h.now + STEP_NS + offset}) is None
  assert not h.guard.armed and not h.guard.binding
  assert h.step() is None


@pytest.mark.parametrize('clock', ['model_ns', 'radar_ns'])
def test_source_rollback_inside_allowed_age_still_clears_context(clock):
  h = Harness()
  h.lose_lead()
  assert h.step(**{clock: h.now - 1}) is None
  assert not h.guard.armed and not h.guard.binding


@pytest.mark.parametrize('dt', [0, -1, 250_000_001])
def test_consumer_clock_repeat_rollback_and_gap_clear_context(dt):
  h = Harness()
  h.lose_lead()
  assert h.step(dt=dt) is None
  assert not h.guard.armed and not h.guard.binding
  assert h.step() is None


def test_maximum_valid_consumer_gap_and_source_age_keep_context():
  h = Harness()
  h.lose_lead()
  source = h.now
  assert h.step(dt=250_000_000, model_ns=source, radar_ns=source) == 0.0
  assert h.guard.armed and h.guard.lost


@pytest.mark.parametrize('field', ['speed', 'steering_angle', 'steer_ratio', 'wheelbase', 'model_curvature', 'ceiling_max'])
@pytest.mark.parametrize('value', [True, math.nan, math.inf, -math.inf, None, '1.0'])
def test_nonfinite_or_non_numeric_vehicle_parameters_clear_context(field, value):
  h = Harness()
  h.lose_lead()
  assert h.step(**{field: value}) is None
  assert not h.guard.armed and not h.guard.binding


@pytest.mark.parametrize('values', [dict(speed=-.1), dict(steer_ratio=0.0), dict(wheelbase=0.0),
                                  dict(ceiling_max=-.1), dict(steering_angle=16.33 * 91.0),
                                  dict(steering_angle=1e308, steer_ratio=1e-308), dict(steering_angle=100.0, wheelbase=1e-310)])
def test_invalid_geometry_and_parameter_domains_clear_context(values):
  h = Harness()
  h.lose_lead()
  assert h.step(**values) is None
  assert not h.guard.armed and not h.guard.binding


@pytest.mark.parametrize('field', ['dRel', 'vRel', 'vLead', 'aLeadK', 'modelProb', 'yRel'])
@pytest.mark.parametrize('value', [True, math.nan, math.inf, None])
def test_invalid_present_lead_fields_cannot_supply_evidence(field, value):
  h = Harness()
  h.lose_lead()
  assert h.step(lead_one=lead(**{field: value})) is None
  assert not h.guard.armed and not h.guard.binding
  assert h.step() is None


@pytest.mark.parametrize('values', [dict(lead_one=None), dict(lead_one=lead(present=1)),
                                  dict(lead_one=lead(modelProb=1.01)), dict(lead_two=lead(modelProb=-.1))])
def test_invalid_lead_slot_and_probability_domain_clear_context(values):
  h = Harness()
  h.lose_lead()
  assert h.step(**values) is None
  assert not h.guard.armed and not h.guard.binding


def test_clear_during_release_does_not_retain_history_or_ceiling():
  h = Harness()
  h.lose_lead()
  h.recover_lead()
  assert h.step(lead_one=lead()) == pytest.approx(.05)
  h.guard.clear()
  assert not h.guard.armed and not h.guard.lost and not h.guard.binding
  assert h.step() is None
