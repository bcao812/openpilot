"""Current-pair consistency at acquisition, after admission, and on recovery."""
import math
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.car.honda_radar_native import NativeMotion
from openpilot.selfdrive.car.tests.test_honda_radar_native import STEP, observe, sample


# M06 / OEM 44 / track 2318, recording abc94211bb03b6443f5148ea42aa241c216f9c8f.
# Original position/AUX time, range, relative speed, ego speed, and U10. These
# are the five real pairs preceding the first recorded qualified publication;
# the test does not identify the photographed vehicle or invent speed truth.
M06_PAIRS = (
  (3153944757129, 34.87056, -12.28125, 11.140432357788086, 106),
  (3154005996295, 34.64208, -12.15625, 11.230971336364746, 85),
  (3154085076660, 34.29936, -12.140625, 11.357285499572754, 95),
  (3154144962753, 34.18512, -11.40625, 11.435977935791016, 176),
  (3154214316868, 33.89952, -10.765625, 11.524638175964355, 200),
)


def m06_startup():
  native = NativeMotion()
  for index, (stamp, distance, relative, ego_speed, uncertainty) in enumerate(M06_PAIRS):
    current = sample(index, distance=distance, relative=relative, slot=2, generation=8157)
    current.update(positionMonoTime=stamp, velocityMonoTime=stamp, frameIndex=index + 5)
    current['raw'].update(oemTrackId=44, lifecycleRaw=29 + 2 * index,
                          u10Raw=uncertainty, rangeSigmaRaw=5 if index == 0 else 4)
    assert observe(native, current, ego=lambda _: SimpleNamespace(speed=ego_speed, yaw=0.0))
    assert native.sample_count == index + 1
    assert native.ground_speed == relative + ego_speed
  return native


def corrected_m06(index):
  # New current observations correct AUX to -6 m/s; no old or visual speed is
  # substituted. Position changes agree with the correcting current stream.
  stamp, distance, _, _, _ = M06_PAIRS[-1]
  current = sample(index + 4, distance=distance - 6.0 * index * STEP / 1e9,
                   relative=-6.0, slot=2, generation=8157)
  current.update(positionMonoTime=stamp + index * STEP, velocityMonoTime=stamp + index * STEP,
                 frameIndex=(9 + index) % 16)
  current['raw'].update(oemTrackId=44, lifecycleRaw=37 + 2 * index)
  return current


def test_m06_correcting_aux_is_accepted_without_premature_control_qualification():
  native = m06_startup()
  assert native.sample_count == 5 and native.pair_good
  assert native.admission_span_ns == 269_559_739
  assert native.admission_residual == pytest.approx(2.211849488554685)
  assert native.admission_residual > native.admission_limit
  assert not native.qualified and native.reason == 'motionDisagreement'
  assert native.ground_speed == pytest.approx(.7590131759643555)

  for index in range(1, 6):
    assert observe(native, corrected_m06(index),
                   ego=lambda _: SimpleNamespace(speed=M06_PAIRS[-1][3], yaw=0.0))
    assert native.ground_speed == -6.0 + M06_PAIRS[-1][3]
  assert native.qualified and native.sample_count == 10


@pytest.mark.parametrize('lead_speed,acceleration', [(0.0, 0.0), (4.0, 0.0), (9.0, -8.0)])
def test_real_stationary_slow_and_braking_leads_keep_original_admission_time(lead_speed, acceleration):
  native = NativeMotion()
  ego_speed = 10.0
  for index in range(8):
    elapsed = index * STEP / 1e9
    relative = lead_speed - ego_speed + acceleration * elapsed
    distance = 35.0 + (lead_speed - ego_speed) * elapsed + .5 * acceleration * elapsed**2
    current = sample(index, relative=relative, distance=distance)
    assert observe(native, current, ego=lambda _: SimpleNamespace(speed=ego_speed, yaw=0.0))
    assert native.qualified is (index >= 4)
    assert native.ground_speed == current['vRel'] + ego_speed


@pytest.mark.parametrize('lead_speed', [0.0, 8.0])
def test_turning_lead_uses_current_body_geometry_without_waiting_for_a_straight_road(lead_speed):
  native = NativeMotion()
  # Exercise mature updates while staying inside the raw azimuth field's
  # +/-0.5 radian range for the entire longer sequence.
  ego_speed, yaw = 10.0, .3
  for index in range(12):
    elapsed = index * STEP / 1e9
    heading = yaw * elapsed
    # World-fixed target trajectory and independently integrated circular ego
    # motion generate the body-frame observations, including rotational x-dot.
    ego_x = ego_speed * math.sin(heading) / yaw
    ego_y = ego_speed * (1.0 - math.cos(heading)) / yaw
    world_x, world_y = 40.0 + lead_speed * elapsed - ego_x, -ego_y
    c, s = math.cos(heading), math.sin(heading)
    current = sample(index, distance=c * world_x + s * world_y,
                     lateral=-s * world_x + c * world_y, relative=lead_speed * c - ego_speed)
    assert observe(native, current, ego=lambda _: SimpleNamespace(speed=ego_speed, yaw=yaw))
    assert native.ground_speed == current['vRel'] + ego_speed
    assert native.qualified is (index >= 4)
  assert native.qualified and native.sample_count == 12
  assert not native.distance_residual_valid  # The old straight-axis diagnostic is still separate.


def test_sustained_disagreement_never_qualifies_from_elapsed_time_alone():
  native = NativeMotion()
  for index in range(40):
    assert observe(native, sample(index, relative=-10.0, distance=30.0))
    assert not native.qualified
  assert native.sample_count == 40 and native.reason == 'motionDisagreement'
  assert len(native.admission_samples) < native.sample_count
  assert native.admission_span_ns <= 500_000_000


def test_split_aux_does_not_erase_or_bypass_pending_admission():
  native = m06_startup()
  complete = corrected_m06(1)
  position = {**complete, 'frameMask': 15, 'velocitySeq': 0, 'velocityMonoTime': 0, 'velocityValid': False}
  before = native.admission_samples
  assert not observe(native, position)
  assert native.pending and not native.qualified
  assert native.sample_count == 5 and native.admission_samples == before
  assert observe(native, complete, ego=lambda _: SimpleNamespace(speed=M06_PAIRS[-1][3], yaw=0.0))
  assert native.sample_count == 6 and not native.qualified
  assert not observe(native, complete, delay=2)
  assert native.sample_count == 6


def test_tentative_migration_copy_cannot_grant_the_original_object_admission():
  native = m06_startup()
  copied = native.copy()
  original_samples = native.admission_samples
  for index in range(1, 6):
    current = corrected_m06(index)
    current.update(slot=1, generation=8158)
    assert observe(copied, current, ego=lambda _: SimpleNamespace(speed=M06_PAIRS[-1][3], yaw=0.0))
  assert copied.qualified and copied.sample_count == 10
  assert native.sample_count == 5 and not native.qualified
  assert native.admission_samples == original_samples


@pytest.mark.parametrize('reset', ['invalidVelocity', 'coasting', 'sourceGap', 'identityChanged'])
def test_support_reset_cannot_inherit_prior_consistency_admission(reset):
  native = NativeMotion()
  for index in range(5):
    assert observe(native, sample(index, distance=30.0 - index * STEP / 1e9))
  assert native.qualified
  start, generation = 5, 1
  if reset == 'invalidVelocity':
    invalid = sample(5)
    invalid.update(velocityValid=False, velocitySeq=0)
    invalid['raw']['u11Raw'] = 0
    assert not observe(native, invalid)
    start = 6
  elif reset == 'coasting':
    native.snapshot(native.position_ns + 100_000_001)
    start = 6
  elif reset == 'sourceGap':
    start = 7
  else:
    generation = 2

  for index in range(start, start + 5):
    assert observe(native, sample(index, relative=-10.0, distance=30.0, generation=generation))
    assert not native.admission_ready and not native.qualified
  assert native.sample_count == 5 and native.reason == 'motionDisagreement'


@pytest.mark.parametrize('bias', [2.3, 2.7, 6.3, -2.3, -2.7, -6.3])
def test_mature_target_loses_authority_on_sustained_native_position_conflict(bias):
  native = NativeMotion()
  distance = 35.0
  for index in range(16):
    if index:
      distance += (-1.0 + (bias if index >= 8 else 0.0)) * STEP / 1e9
    current = sample(index, relative=-1.0, distance=distance)
    assert observe(native, current)
    # Contradictory geometry cannot correct or replace the current AUX speed.
    assert native.ground_speed == 19.0 and native.raw_relative_speed == -1.0
    if 4 <= index < 8:
      assert native.qualified
    if index >= 12:
      assert not native.qualified and native.reason == 'motionDisagreement'
      assert not native.snapshot(current['positionMonoTime'] + 1)['qualified']

  count = native.sample_count
  # Correct NEW current pairs recover normally, without a timer or old speed.
  for index in range(16, 21):
    distance -= STEP / 1e9
    assert observe(native, sample(index, relative=-1.0, distance=distance))
  assert native.qualified and native.sample_count == count + 5


def test_opposite_sustained_interval_errors_cannot_cancel_into_admission():
  native = NativeMotion()
  distance = 35.0
  for index in range(16):
    if index:
      distance += (-1.0 + (8.0 * (-1 if index % 2 else 1) if index >= 8 else 0.0)) * STEP / 1e9
    assert observe(native, sample(index, relative=-1.0, distance=distance))
    if 4 <= index < 8:
      assert native.qualified
    if index >= 12:
      assert abs(native.admission_residual) < 1e-10
      assert not native.qualified and native.reason == 'motionDisagreement'


@pytest.mark.parametrize('reflection_step', [-.5, .5])
def test_single_reflection_step_and_range_quantization_do_not_remove_real_target(reflection_step):
  native = NativeMotion()
  for index in range(28):
    distance = 35.0 - index * STEP / 1e9 + (reflection_step if index >= 12 else 0.0)
    distance = round(distance / .05712) * .05712
    assert observe(native, sample(index, relative=-1.0, distance=distance))
    if index >= 4:
      assert native.qualified
  assert native.admission_span_ns <= 500_000_000
  assert native.sample_count > len(native.admission_samples)


def test_republication_cannot_recover_a_current_motion_conflict():
  native = NativeMotion()
  for index in range(12):
    current = sample(index, relative=-10.0, distance=35.0 - 7.3 * index * STEP / 1e9)
    assert observe(native, current)
  assert not native.qualified
  count, samples = native.sample_count, native.admission_samples
  for delay in (2, 10_000_000, 30_000_000):
    assert not observe(native, current, delay=delay)
    assert not native.qualified and native.sample_count == count
    assert native.admission_samples == samples


def test_disconnected_quantization_residuals_keep_their_independent_endpoint_budget():
  native = NativeMotion()
  for index in range(40):
    # A stationary return on a quantization boundary alternates adjacent raw
    # range bins. Faster valid source pairs produce several disconnected
    # positive/negative residual groups in the same 250 ms support window.
    distance = (650 + index % 2) * .05712 - 3.0
    current = sample(index, step=17_000_000, relative=0.0, distance=distance)
    assert observe(native, current, ego=lambda _: SimpleNamespace(speed=0.0, yaw=0.0))
    if index >= 15:
      assert native.qualified
