from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import CREEP_MAX_T, CREEP_STOP_T, STOP_RELEASE_ACCEL
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.motion_supervisor import (MODEL_LEASE_NS, MOTION_INTENT_VERSION, PLAN_LEASE_NS, MotionIntent, MotionPhase,
                                                               MotionSupervisor, RAW_STOP_REASONS, StopReason)

CONTROL_INTERVAL_NS = round(DT_CTRL * 1e9)


class MotionHarness:
  def __init__(self, mode=LongitudinalMode.UNIFIED):
    self.supervisor = MotionSupervisor(mode=mode)
    self.now_ns = 2_000_000_000
    self.plan_time_ns = self.now_ns
    self.model_time_ns = self.now_ns

  def step(self, intent=None, a_target=-0.1, v_ego=0.5, standstill=False, active=True, a_ego=0.0,
           new_plan=True, new_model=True, plan_valid=True, braking_plan_valid=False, solver_status=0,
           advance_ns=CONTROL_INTERVAL_NS, **timestamps):
    self.now_ns += advance_ns
    if new_plan:
      self.plan_time_ns = self.now_ns
    if new_model:
      self.model_time_ns = self.now_ns
    return self.supervisor.update(
      active, v_ego, a_ego, standstill, a_target, MotionIntent() if intent is None else intent,
      plan_time_ns=timestamps.get('plan_time_ns', self.plan_time_ns), model_time_ns=timestamps.get('model_time_ns', self.model_time_ns),
      now_ns=timestamps.get('now_ns', self.now_ns), plan_valid=plan_valid, braking_plan_valid=braking_plan_valid, solver_status=solver_status)

  def stopped(self):
    self.step(MotionIntent(reasons=StopReason.MPC), v_ego=0.0, standstill=True)
    assert self.supervisor.phase == MotionPhase.HOLD


def test_contract_is_immutable_and_wire_reasons_are_separate():
  intent = MotionIntent()
  with pytest.raises(FrozenInstanceError):
    intent.reasons = StopReason.HOLD
  assert int(RAW_STOP_REASONS) == 31
  assert [int(reason) for reason in StopReason] == [1, 2, 4, 8, 16, 32, 64, 128]
  assert MotionPhase.HOLD.text == 'hold'


def test_decode_preserves_types_and_absent_protocol_is_invalid():
  message = SimpleNamespace(version=MOTION_INTENT_VERSION, mode=int(LongitudinalMode.UNIFIED),
                            reasons=8, movingLead=False, releaseAllowed=True, allowCreep=False, valid=True)
  intent = MotionIntent.from_message(message)
  assert intent == MotionIntent(reasons=StopReason.CLOSE_LEAD, release_allowed=True, allow_creep=False)
  assert intent.is_valid()
  assert not MotionIntent.from_message(None).is_valid()
  message.valid = 'true'
  assert not MotionIntent.from_message(message).is_valid()


@pytest.mark.parametrize('changes', [
  {'version': 0}, {'version': 1}, {'version': 2}, {'version': MOTION_INTENT_VERSION + 1},
  {'version': True}, {'version': float(MOTION_INTENT_VERSION)},
  {'reasons': -1}, {'reasons': 1 << 25}, {'reasons': StopReason.CREEP},
  {'reasons': StopReason.HOLD}, {'reasons': StopReason.INVALID}, {'reasons': True},
  {'moving_lead': 1}, {'release_allowed': 'yes'}, {'allow_creep': None}, {'valid': False},
])
def test_bad_contract_cannot_release_a_stopped_car(changes):
  harness = MotionHarness()
  harness.stopped()
  assert harness.step(replace(MotionIntent(), **changes), a_target=1.0, v_ego=0.0, standstill=True)
  assert harness.supervisor.phase == MotionPhase.HOLD
  assert harness.supervisor.reasons & StopReason.INVALID
  assert harness.supervisor.degraded


def test_raw_evidence_takes_priority_over_release_permission():
  harness = MotionHarness()
  harness.stopped()
  for reason in (StopReason.MPC, StopReason.E2E, StopReason.CRUISE, StopReason.CLOSE_LEAD, StopReason.FORCE_DECEL):
    assert harness.step(MotionIntent(reasons=reason, release_allowed=True), a_target=1.0, v_ego=0.0, standstill=True)
    assert harness.supervisor.reasons & reason


@pytest.mark.parametrize('a_target', [0.0, 0.05, 0.15, 0.27, 0.448, 1.0, 2.0])
def test_a_restarted_owner_establishes_hold_instead_of_treating_standstill_as_tracking(a_target):
  harness = MotionHarness()
  assert harness.step(a_target=a_target, v_ego=0.0, standstill=True)
  assert harness.supervisor.phase == MotionPhase.HOLD
  assert harness.supervisor.reasons == StopReason.HOLD
  assert not harness.supervisor.degraded


@pytest.mark.parametrize('speed,standstill', [(0.0, True), (0.025, False), (0.3, False), (1.3, False), (10.0, False)])
@pytest.mark.parametrize('a_target', [0.27, 0.448, 1.0, 2.0])
def test_positive_scalar_cannot_release_an_owned_stop_without_movement_permission(speed, standstill, a_target):
  harness = MotionHarness()
  harness.step(MotionIntent(reasons=StopReason.MPC), a_target=-0.4, v_ego=speed, standstill=standstill, a_ego=-0.4)
  phase = MotionPhase.HOLD if standstill else MotionPhase.APPROACH
  assert harness.supervisor.phase == phase
  for _ in range(60):
    assert harness.step(a_target=a_target, v_ego=speed, standstill=standstill, a_ego=-0.4)
    assert harness.supervisor.phase == phase
    assert harness.supervisor.reasons & StopReason.MPC
    assert not harness.supervisor.degraded
    assert harness.supervisor.release_lead_t == 0.0


@pytest.mark.parametrize('a_target', [-2.5, -0.7, 0.05, 0.15])
@pytest.mark.parametrize('speed', [0.3, 0.5, 1.3, 10.0])
def test_a_rolling_stop_cannot_use_reason_clearance_as_implicit_movement_permission(speed, a_target):
  harness = MotionHarness()
  assert harness.step(MotionIntent(reasons=StopReason.E2E), v_ego=speed, a_target=-2.0)
  assert harness.step(v_ego=speed, a_target=a_target)
  assert harness.supervisor.phase == MotionPhase.APPROACH
  assert harness.supervisor.reasons & StopReason.E2E
  assert not harness.supervisor.degraded


def test_stop_hysteresis_and_regular_cruise_release_are_preserved():
  harness = MotionHarness()
  harness.stopped()
  move_intent = MotionIntent(release_allowed=True)
  for acceleration in (0.11, 0.15, STOP_RELEASE_ACCEL - 0.001):
    assert harness.step(move_intent, a_target=acceleration, v_ego=0.0, standstill=True)
  assert not harness.step(move_intent, a_target=STOP_RELEASE_ACCEL, v_ego=0.0, standstill=True)
  assert harness.supervisor.phase == MotionPhase.RELEASE
  harness.supervisor.complete_release(-1.0, STOP_RELEASE_ACCEL)
  assert harness.supervisor.phase == MotionPhase.RELEASE
  harness.supervisor.complete_release(STOP_RELEASE_ACCEL, STOP_RELEASE_ACCEL)
  assert harness.supervisor.phase == MotionPhase.TRACK


def test_safe_departure_can_release_below_the_old_acceleration_threshold():
  harness = MotionHarness()
  harness.stopped()
  intent = MotionIntent(release_allowed=True)
  model_period = round(DT_MDL / DT_CTRL)
  confirmation_frames = round(CREEP_STOP_T / DT_CTRL)
  for frame in range(confirmation_frames + model_period):
    new_input = frame % model_period == 0
    harness.step(intent, a_target=0.15, v_ego=0.0, standstill=True, new_plan=new_input, new_model=new_input)
    if frame < confirmation_frames:
      assert harness.supervisor.phase == MotionPhase.HOLD
  assert harness.supervisor.phase == MotionPhase.RELEASE
  assert not harness.supervisor.resolved_should_stop


@pytest.mark.parametrize('a_target', [-3.0, -0.1, 0.0])
@pytest.mark.parametrize('speed,standstill', [(0.0, True), (0.2, False)])
def test_departure_evidence_can_be_confirmed_before_a_positive_target_without_releasing(a_target, speed, standstill):
  harness = MotionHarness()
  harness.stopped()
  intent = MotionIntent(release_allowed=True, allow_creep=False)
  for frame in range(120):
    new_input = frame % 5 == 0
    assert harness.step(intent, a_target=a_target, v_ego=speed, standstill=standstill,
                        new_plan=new_input, new_model=new_input)
    assert harness.supervisor.phase in (MotionPhase.APPROACH, MotionPhase.HOLD)
  assert harness.supervisor.release_lead_t == pytest.approx(CREEP_STOP_T)
  assert not harness.step(intent, a_target=0.05, v_ego=speed, standstill=standstill)
  assert harness.supervisor.phase == MotionPhase.RELEASE


def test_withdrawal_discards_departure_evidence_confirmed_during_negative_targets():
  harness = MotionHarness()
  harness.stopped()
  intent = MotionIntent(release_allowed=True, allow_creep=False)
  for frame in range(60):
    new_input = frame % 5 == 0
    assert harness.step(intent, a_target=-0.1, v_ego=0.0, standstill=True, new_plan=new_input, new_model=new_input)
  assert harness.supervisor.release_lead_t == pytest.approx(CREEP_STOP_T)
  assert harness.step(MotionIntent(allow_creep=False), a_target=-0.1, v_ego=0.0, standstill=True)
  assert harness.supervisor.release_lead_t == 0.0
  for frame in range(40):
    new_input = frame % 5 == 0
    assert harness.step(intent, a_target=0.05, v_ego=0.0, standstill=True, new_plan=new_input, new_model=new_input)


def test_cached_negative_target_publications_do_not_prepay_departure_confirmation():
  harness = MotionHarness()
  harness.stopped()
  intent = MotionIntent(release_allowed=True, allow_creep=False)
  harness.step(intent, a_target=-0.1, v_ego=0.0, standstill=True)
  for _ in range(24):
    assert harness.step(intent, a_target=-0.1, v_ego=0.0, standstill=True, new_plan=False, new_model=False)
  assert harness.supervisor.release_lead_t <= DT_CTRL
  assert not harness.supervisor.degraded
  for frame in range(40):
    new_input = frame % 5 == 0
    assert harness.step(intent, a_target=0.05, v_ego=0.0, standstill=True, new_plan=new_input, new_model=new_input)


@pytest.mark.parametrize('speed', [0.31, 1.3])
def test_a_held_stop_keeps_its_departure_contract_when_measurements_cross_the_stop_band(speed):
  harness = MotionHarness()
  harness.stopped()
  # A still-published raw stop can be consumed after measured speed leaves
  # zero. It must not erase the fact that this stop already reached HOLD.
  assert harness.step(MotionIntent(reasons=StopReason.E2E), a_target=-0.1, v_ego=0.2)
  intent = MotionIntent(release_allowed=True, allow_creep=False)
  assert harness.step(intent, a_target=-0.1, v_ego=0.2)
  assert harness.step(intent, a_target=-0.1, v_ego=speed, new_plan=False, new_model=False)
  assert harness.supervisor.reasons & StopReason.HOLD
  assert harness.supervisor.release_lead_t < CREEP_STOP_T
  for frame in range(40):
    new_input = frame % 5 == 0
    assert harness.step(intent, a_target=0.05, v_ego=speed, new_plan=new_input, new_model=new_input)
  for frame in range(20):
    new_input = frame % 5 == 0
    harness.step(intent, a_target=0.05, v_ego=speed, new_plan=new_input, new_model=new_input)
  assert harness.supervisor.phase == MotionPhase.RELEASE
  assert not harness.supervisor.reasons


def test_withdrawing_movement_permission_restarts_low_target_confirmation():
  harness = MotionHarness()
  harness.stopped()
  move_intent = MotionIntent(release_allowed=True)
  model_period = round(DT_MDL / DT_CTRL)
  confirmation_frames = round(CREEP_STOP_T / DT_CTRL)
  for frame in range(confirmation_frames // 2):
    new_input = frame % model_period == 0
    assert harness.step(move_intent, a_target=0.15, v_ego=0.0, standstill=True, new_plan=new_input, new_model=new_input)
  assert harness.supervisor.release_lead_t > 0.0
  assert harness.step(a_target=0.15, v_ego=0.0, standstill=True)
  assert harness.supervisor.release_lead_t == 0.0
  for frame in range(confirmation_frames + model_period):
    new_input = frame % model_period == 0
    harness.step(move_intent, a_target=0.15, v_ego=0.0, standstill=True, new_plan=new_input, new_model=new_input)
    if frame < confirmation_frames:
      assert harness.supervisor.phase == MotionPhase.HOLD
  assert harness.supervisor.phase == MotionPhase.RELEASE


def test_moving_follow_qualification_alone_cannot_release_hold():
  harness = MotionHarness()
  harness.stopped()
  for _ in range(100):
    assert harness.step(MotionIntent(moving_lead=True), a_target=0.15, v_ego=0.0, standstill=True)
  assert harness.supervisor.moving_lead_t == 0.0


@pytest.mark.parametrize('new_plan,new_model', [(False, False), (True, False)])
def test_duplicate_producer_evidence_cannot_confirm_a_departure(new_plan, new_model):
  harness = MotionHarness()
  harness.stopped()
  intent = MotionIntent(release_allowed=True)
  harness.step(intent, a_target=0.15, v_ego=0.0, standstill=True)
  for _ in range(24):
    assert harness.step(intent, a_target=0.15, v_ego=0.0, standstill=True, new_plan=new_plan, new_model=new_model)
    assert harness.supervisor.release_lead_t <= DT_CTRL
  assert harness.supervisor.phase == MotionPhase.HOLD


def test_fresh_moving_lead_confirms_before_anti_creep_can_stop_a_follow():
  harness = MotionHarness()
  intent = MotionIntent(moving_lead=True)
  for frame in range(250):
    new_input = frame % 5 == 0
    assert not harness.step(intent, v_ego=1.0, a_target=-0.2, new_plan=new_input, new_model=new_input)
  assert harness.supervisor.moving_lead_t == pytest.approx(CREEP_STOP_T)
  assert harness.supervisor.creep_total_t == 0.0


def test_a_real_creep_uses_control_time_and_latches_once():
  harness = MotionHarness()
  for frame in range(60):
    new_input = frame % 5 == 0
    harness.step(v_ego=0.5, new_plan=new_input, new_model=new_input)
    if frame < 50:
      assert not harness.supervisor.resolved_should_stop
  assert harness.supervisor.reasons & StopReason.CREEP
  for frame in range(20):
    assert harness.step(v_ego=max(0.31, 0.5 - frame * 0.01))
  assert not harness.step(MotionIntent(release_allowed=True), v_ego=0.5, a_target=0.05)
  assert harness.supervisor.phase == MotionPhase.RELEASE


def test_total_creep_deadline_and_prompt_stop_are_distinct():
  harness = MotionHarness()
  speed = 1.15
  for _ in range(60):
    harness.step(v_ego=speed)
    speed -= 0.65 * DT_CTRL
  assert not harness.supervisor.resolved_should_stop

  harness = MotionHarness()
  speed = 1.15
  for frame in range(round(CREEP_MAX_T / DT_CTRL)):
    harness.step(v_ego=speed)
    speed *= 1.0 - 0.6 * DT_CTRL
    if frame < 99:
      assert not harness.supervisor.resolved_should_stop
  assert harness.supervisor.reasons & StopReason.CREEP


@pytest.mark.parametrize('a_target', [0.0, 0.05, 0.15, 0.5])
def test_positive_launch_does_not_accumulate_anti_creep(a_target):
  harness = MotionHarness()
  for _ in range(150):
    assert not harness.step(a_target=a_target)
  assert harness.supervisor.creep_total_t == 0.0


def test_anti_creep_requires_explicit_operating_qualification():
  harness = MotionHarness()
  for _ in range(150):
    assert not harness.step(MotionIntent(allow_creep=False))
  assert harness.supervisor.creep_total_t == 0.0


def test_late_moving_evidence_cannot_undo_an_existing_creep_stop():
  harness = MotionHarness()
  for _ in range(60):
    harness.step()
  for _ in range(100):
    assert harness.step(MotionIntent(moving_lead=True))
  assert harness.supervisor.reasons & StopReason.CREEP
  assert harness.supervisor.moving_lead_t == 0.0


def test_close_lead_noise_does_not_own_a_second_latch():
  harness = MotionHarness()
  harness.step(MotionIntent(reasons=StopReason.CLOSE_LEAD), v_ego=1.3, a_target=-0.7)
  for _ in range(25):
    assert harness.step(v_ego=1.3, a_target=-0.7)
    assert harness.supervisor.reasons == StopReason.CLOSE_LEAD
  assert not harness.step(MotionIntent(release_allowed=True), v_ego=1.3, a_target=-2.5)
  assert not harness.supervisor.reasons


def test_a_model_stop_can_be_withdrawn_while_the_car_is_moving():
  harness = MotionHarness()
  assert harness.step(MotionIntent(reasons=StopReason.E2E), v_ego=10.0, a_target=-2.0)
  assert not harness.step(MotionIntent(release_allowed=True), v_ego=10.0, a_target=-0.7)
  assert harness.supervisor.phase == MotionPhase.RELEASE


@pytest.mark.parametrize('a_target', [-2.5, -0.7, 0.05, 0.15])
@pytest.mark.parametrize('speed', [0.3, 0.5, 1.3, 10.0])
def test_explicit_movement_permission_preserves_existing_rolling_stop_withdrawal(speed, a_target):
  harness = MotionHarness()
  assert harness.step(MotionIntent(reasons=StopReason.E2E), v_ego=speed, a_target=-2.0)
  assert not harness.step(MotionIntent(release_allowed=True), v_ego=speed, a_target=a_target)
  assert harness.supervisor.phase == MotionPhase.RELEASE
  assert not harness.supervisor.reasons
  assert not harness.supervisor.degraded


@pytest.mark.parametrize('field', ['plan_time_ns', 'model_time_ns', 'now_ns'])
@pytest.mark.parametrize('timestamp', [0, -1, True, 2.0, float('nan'), float('inf')])
def test_invalid_timestamps_fail_closed(field, timestamp):
  harness = MotionHarness()
  harness.stopped()
  assert harness.step(a_target=1.0, v_ego=0.0, standstill=True, **{field: timestamp})
  assert harness.supervisor.degraded


@pytest.mark.parametrize('field', ['plan_time_ns', 'model_time_ns'])
def test_future_or_reordered_producer_input_cannot_release_hold(field):
  harness = MotionHarness()
  harness.stopped()
  assert harness.step(a_target=1.0, v_ego=0.0, standstill=True, **{field: harness.now_ns + 1_000_000_000})
  assert harness.supervisor.degraded
  assert harness.step(a_target=1.0, v_ego=0.0, standstill=True, **{field: harness.now_ns - 50_000_000})
  assert harness.supervisor.degraded


def test_clock_rollback_and_large_unobserved_gap_do_not_add_dwell():
  harness = MotionHarness()
  harness.step(MotionIntent(moving_lead=True), v_ego=1.0)
  harness.step(MotionIntent(moving_lead=True), v_ego=1.0, advance_ns=-1)
  assert harness.supervisor.degraded
  assert harness.supervisor.moving_lead_t == 0.0
  harness.step(MotionIntent(moving_lead=True), v_ego=1.0, advance_ns=500_000_000)
  assert harness.supervisor.degraded
  assert harness.supervisor.creep_total_t == 0.0


@pytest.mark.parametrize('field,lease', [('plan_time_ns', PLAN_LEASE_NS), ('model_time_ns', MODEL_LEASE_NS)])
def test_each_input_has_an_independent_finite_lease(field, lease):
  harness = MotionHarness()
  harness.stopped()
  assert harness.step(a_target=1.0, v_ego=0.0, standstill=True, **{field: harness.now_ns - lease})
  assert harness.supervisor.degraded


def test_same_plan_identity_cannot_be_rewritten_to_drop_stop_evidence():
  harness = MotionHarness()
  harness.stopped()
  assert harness.step(a_target=1.0, v_ego=0.0, standstill=True, new_plan=False, new_model=False)
  assert harness.supervisor.degraded


@pytest.mark.parametrize('field', ['a_target', 'v_ego', 'a_ego'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_non_finite_vehicle_or_plan_values_are_degraded(field, value):
  harness = MotionHarness()
  harness.step(**{field: value})
  assert harness.supervisor.degraded
  assert harness.supervisor.reasons & StopReason.INVALID


def test_disable_wins_and_reengagement_requires_new_producer_identity():
  harness = MotionHarness()
  harness.stopped()
  assert not harness.step(active=False, new_plan=False, new_model=False, a_target=float('nan'), now_ns=-1)
  assert harness.supervisor.phase == MotionPhase.OFF
  assert not harness.supervisor.reasons
  harness.step(a_target=1.0, new_plan=False, new_model=False)
  assert harness.supervisor.degraded
  harness.step(a_target=1.0)
  assert not harness.supervisor.degraded


@pytest.mark.parametrize('fault', ['unhealthy', 'nonfinite', 'rewritten', 'future', 'control_gap'])
def test_fault_revokes_a_previously_accepted_release_until_both_inputs_advance(fault):
  harness = MotionHarness()
  harness.stopped()
  move_intent = MotionIntent(release_allowed=True)
  harness.step(move_intent, a_target=0.3, v_ego=0.0, standstill=True)
  assert harness.supervisor.phase == MotionPhase.RELEASE
  fault_args = {
    'unhealthy': {'plan_valid': False},
    'nonfinite': {'a_ego': float('nan')},
    'rewritten': {'a_target': 0.5},
    'future': {'plan_time_ns': harness.now_ns + 1_000_000_000},
    'control_gap': {'advance_ns': 110_000_000},
  }[fault]
  fault_args.setdefault('a_target', 0.3)
  harness.step(move_intent, v_ego=0.0, standstill=True, new_plan=False, new_model=False, **fault_args)
  assert harness.supervisor.degraded
  assert harness.supervisor.phase == MotionPhase.HOLD
  for new_plan, new_model in ((False, False), (True, False)):
    assert harness.step(move_intent, a_target=0.3, v_ego=0.0, standstill=True, new_plan=new_plan, new_model=new_model)
    assert harness.supervisor.degraded
    assert harness.supervisor.phase == MotionPhase.HOLD
  assert not harness.step(move_intent, a_target=0.3, v_ego=0.0, standstill=True)
  assert not harness.supervisor.degraded
  assert harness.supervisor.phase == MotionPhase.RELEASE


@pytest.mark.parametrize('estimated_speed', [-0.001184, -0.031628, -0.039136])
def test_nonnegative_wheel_speed_kalman_undershoot_does_not_create_a_protocol_fault(estimated_speed):
  harness = MotionHarness()
  harness.step(MotionIntent(reasons=StopReason.MPC), v_ego=estimated_speed, a_target=-0.5)
  assert not harness.supervisor.degraded
  assert harness.supervisor.phase == MotionPhase.APPROACH
  assert harness.supervisor.resolved_should_stop


def test_readiness_default_preserves_the_existing_positional_contract():
  intent = MotionIntent(MOTION_INTENT_VERSION, 0, False, False, True, True)
  assert intent.ready
  assert intent.is_valid()
  with pytest.raises(FrozenInstanceError):
    intent.ready = False


def test_current_messages_without_readiness_default_to_ready():
  message = SimpleNamespace(version=MOTION_INTENT_VERSION, mode=int(LongitudinalMode.UNIFIED),
                            reasons=0, movingLead=False, releaseAllowed=False, allowCreep=True, valid=True)
  intent = MotionIntent.from_message(message)
  assert intent.ready
  assert intent.is_valid()


@pytest.mark.parametrize('ready', [False, True])
def test_readiness_is_decoded_without_becoming_a_protocol_fault(ready):
  message = SimpleNamespace(version=MOTION_INTENT_VERSION, mode=int(LongitudinalMode.UNIFIED),
                            reasons=0, movingLead=False, releaseAllowed=False, allowCreep=True, valid=True, ready=ready)
  intent = MotionIntent.from_message(message)
  assert intent.ready is ready
  assert intent.is_valid()


@pytest.mark.parametrize('ready', [None, 0, 1, 0.0, 'false', [], {}])
def test_readiness_requires_an_actual_boolean(ready):
  intent = MotionIntent(ready=ready)
  assert not intent.is_valid()
  harness = MotionHarness()
  harness.stopped()
  assert harness.step(intent, a_target=1.0, v_ego=0.0, standstill=True)
  assert harness.supervisor.degraded
  assert harness.supervisor.phase == MotionPhase.HOLD


@pytest.mark.parametrize('valid', [False, True])
def test_cold_off_consumer_can_wait_for_first_set_without_accepting_motion_input(valid):
  harness = MotionHarness()
  assert harness.supervisor.can_standby
  for _ in range(20):
    assert not harness.step(MotionIntent(ready=False, valid=valid), a_target=0.0, active=False)
    assert harness.supervisor.can_standby
    assert harness.supervisor.phase == MotionPhase.OFF
    assert not harness.supervisor.degraded
    assert harness.supervisor._last_plan_time_ns is None


def test_unready_producer_without_an_executable_trajectory_is_not_valid_for_motion():
  intent = MotionIntent(ready=False, valid=False)
  assert not intent.is_valid()
  harness = MotionHarness()
  harness.step(intent, a_target=1.0, v_ego=10.0)
  assert harness.supervisor.degraded
  assert harness.supervisor.degraded_braking_accel is None
  assert not harness.supervisor.can_standby


def test_first_ready_input_permanently_consumes_cold_standby_eligibility():
  harness = MotionHarness()
  assert harness.supervisor.can_standby
  assert not harness.step(a_target=1.0, v_ego=10.0)
  assert harness.supervisor.phase == MotionPhase.TRACK
  assert not harness.supervisor.can_standby
  plan_time = harness.supervisor._last_plan_time_ns
  model_time = harness.supervisor._last_model_time_ns
  harness.step(active=False)
  assert harness.supervisor.phase == MotionPhase.OFF
  assert not harness.supervisor.can_standby
  # Healthy OFF evidence establishes the next rearm barrier without granting
  # drive authority or restoring the one-time cold standby exemption.
  assert harness.supervisor._last_plan_time_ns == harness.plan_time_ns > plan_time
  assert harness.supervisor._last_model_time_ns == harness.model_time_ns > model_time
  assert harness.supervisor._require_new_input
  harness.step(MotionIntent(ready=False), active=False)
  assert not harness.supervisor.can_standby


def test_accepted_hold_also_prevents_return_to_cold_standby():
  harness = MotionHarness()
  harness.stopped()
  assert not harness.supervisor.can_standby
  harness.supervisor.reset()
  assert harness.supervisor.phase == MotionPhase.OFF
  assert not harness.supervisor.can_standby


@pytest.mark.parametrize('a_target', [0.0, 0.05, 0.15, 1.0])
def test_unready_input_cannot_authorize_an_active_consumer_to_drive(a_target):
  harness = MotionHarness()
  intent = MotionIntent(ready=False)
  assert intent.is_valid()
  harness.step(intent, a_target=a_target, v_ego=10.0)
  assert harness.supervisor.degraded
  assert harness.supervisor.reasons & StopReason.INVALID
  assert harness.supervisor.degraded_braking_accel is None
  assert harness.supervisor._last_plan_time_ns is None
  assert not harness.supervisor.can_standby


def test_unready_input_cannot_release_a_previously_accepted_hold():
  harness = MotionHarness()
  harness.stopped()
  plan_time = harness.supervisor._last_plan_time_ns
  model_time = harness.supervisor._last_model_time_ns
  for _ in range(60):
    assert harness.step(MotionIntent(ready=False, release_allowed=True), a_target=1.0, v_ego=0.0, standstill=True)
    assert harness.supervisor.degraded
    assert harness.supervisor.phase == MotionPhase.HOLD
    assert harness.supervisor.release_lead_t == 0.0
    assert harness.supervisor._last_plan_time_ns == plan_time
    assert harness.supervisor._last_model_time_ns == model_time
    assert not harness.supervisor.can_standby


def test_readiness_loss_after_motion_is_not_a_new_cold_start():
  harness = MotionHarness()
  harness.step(a_target=1.0, v_ego=10.0)
  harness.step(active=False)
  assert not harness.supervisor.can_standby
  harness.step(MotionIntent(ready=False), a_target=1.0, v_ego=10.0)
  assert harness.supervisor.degraded
  assert not harness.supervisor.can_standby
  harness.step(a_target=1.0, v_ego=10.0)
  assert not harness.supervisor.degraded
  assert harness.supervisor.phase == MotionPhase.TRACK
  assert not harness.supervisor.can_standby
