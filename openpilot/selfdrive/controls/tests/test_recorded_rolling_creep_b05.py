"""B05 owner-only recorded-input regression, including flickering permission.

The fixture freezes successful LPs and every owner consumption clock/identity,
including repeated samples. Vehicle values are a bounded backward-asof join,
not an exposed consumed carState identity. The recorded trajectory includes
baseline braking: these checks do not predict vehicle motion after the edit.
No controller output, MPC solve, or closed-loop plant is replayed here.
"""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, CREEP_MAX_T, STOP_SPEED, get_departure_accel_from_plan
from openpilot.selfdrive.controls.lib.motion_supervisor import MAX_CONTROL_GAP_NS, MotionIntent, MotionPhase, MotionSupervisor, StopReason
from openpilot.selfdrive.modeld.constants import ModelConstants


FIXTURE = json.loads((Path(__file__).parent / 'fixtures/rolling_creep_b05_9e3a811d.json').read_text())
PLANS = {plan['plan_ns']: plan for plan in FIXTURE['plans']}
FIRST_PERMISSION_NS = 1_648_572_113_327
FIRST_PERMISSION_WITHDRAWAL_NS = 1_648_620_945_410
FIRST_WITHDRAWN_PLAN_NS = 1_648_614_206_816
RETURNED_PERMISSION_NS = 1_649_223_142_076
RECORDED_FIRST_CREEP_NS = 1_648_651_218_951


def replay_b05(*, seed_hold=False, withdraw_permission=False):
  """Exercise only owner inputs; never import the baseline owner's phases."""
  owner = MotionSupervisor()
  intents = {}
  for plan_ns, plan in PLANS.items():
    intent = MotionIntent.from_message(SimpleNamespace(**plan['intent']))
    # Recreate the consumer-local fact from the saved wire precision, as
    # controlsd does. It remains independent of the actual target request.
    intent = replace(intent, release_accel=get_departure_accel_from_plan(
      tuple(plan['speeds']), tuple(plan['accels']), ModelConstants.T_IDXS[:CONTROL_N],
      FIXTURE['actuator_delay_s'] + DT_MDL, plan['constraint_accel']))
    if withdraw_permission and plan_ns >= FIRST_WITHDRAWN_PLAN_NS:
      # Controlled negative case: after the actual first withdrawal, prevent
      # later plans from restoring permission. All clocks, targets and vehicle
      # values are unchanged. Each LP identity retains one immutable payload.
      intent = replace(intent, release_allowed=False)
    assert intent.is_valid()
    intents[plan_ns] = intent

  if seed_hold:
    first = FIXTURE['consumptions'][0]
    assert owner.update(
      True, 0.0, 0.0, True, -0.35,
      MotionIntent(reasons=StopReason.MPC, producer_instance_ns=intents[first['plan_ns']].producer_instance_ns),
      plan_time_ns=first['plan_ns'] - round(DT_MDL * 1e9),
      model_time_ns=first['model_ns'] - round(DT_MDL * 1e9), now_ns=first['now_ns'] - round(DT_CTRL * 1e9))
    assert owner.phase == MotionPhase.HOLD

  for row in FIXTURE['consumptions']:
    plan = PLANS[row['plan_ns']]
    # The seeded negative test stops before positive targets could legitimately
    # exercise the separate standstill departure contract.
    if seed_hold and plan['target'] > 0.0:
      break
    assert row['model_ns'] == plan['model_ns']
    assert plan['valid'] and plan['solver_status'] == 0
    assert not plan['should_stop'] and plan['intent']['reasons'] == 0
    assert 0 <= row['now_ns'] - row['car_state_ns'] <= FIXTURE['max_vehicle_age_ns']
    stopped = owner.update(
      row['active'], row['v_ego'], row['a_ego'], row['standstill'], plan['target'], intents[row['plan_ns']],
      plan_time_ns=row['plan_ns'], model_time_ns=row['model_ns'], now_ns=row['now_ns'],
      plan_valid=plan['valid'], solver_status=plan['solver_status'])
    assert not owner.degraded and not owner.waiting_for_fresh_input
    yield row, plan, stopped, owner.phase, owner.reasons


def test_recorded_b05_flickering_permission_removes_creep_on_the_fixed_vehicle_trace():
  rows = FIXTURE['consumptions']
  assert len(rows) == 399 and len(PLANS) == 81
  assert FIXTURE['max_owner_gap_ns'] <= MAX_CONTROL_GAP_NS
  assert all(a['now_ns'] < b['now_ns'] for a, b in zip(rows, rows[1:]))
  assert any(a['plan_ns'] == b['plan_ns'] for a, b in zip(rows, rows[1:]))
  transitions, previous = [], False
  for row in rows:
    permission = PLANS[row['plan_ns']]['intent']['releaseAllowed']
    if permission != previous:
      transitions.append((row['now_ns'], permission))
      previous = permission
  assert transitions[:3] == [(FIRST_PERMISSION_NS, True), (FIRST_PERMISSION_WITHDRAWAL_NS, False),
                             (RETURNED_PERMISSION_NS, True)]
  assert next(row['now_ns'] for row in rows if row['recorded_reasons'] & StopReason.CREEP) == RECORDED_FIRST_CREEP_NS

  results = list(replay_b05())
  assert not any(reasons & StopReason.CREEP for _, _, _, _, reasons in results)
  # There is still a real false-permission interval across the recorded stop
  # onset. Passing this test cannot be explained by continuous permission or
  # by silently dropping repeated control consumption between LP messages.
  assert any(FIRST_PERMISSION_WITHDRAWAL_NS <= row['now_ns'] < RETURNED_PERMISSION_NS
             and row['recorded_reasons'] & StopReason.CREEP and not stopped
             for row, _, stopped, _, _ in results)


def test_recorded_b05_negative_targets_cannot_unlock_a_preexisting_hold():
  results = list(replay_b05(seed_hold=True))
  assert any(plan['intent']['releaseAllowed'] and row['v_ego'] >= STOP_SPEED for row, plan, _, _, _ in results)
  assert all(stopped and reasons & StopReason.HOLD for _, _, stopped, _, reasons in results)
  assert all(phase in (MotionPhase.APPROACH, MotionPhase.HOLD) for _, _, _, phase, _ in results)


def test_b05_permission_not_returning_still_reaches_the_bounded_creep_stop():
  results = list(replay_b05(withdraw_permission=True))
  first_stop_ns = next(row['now_ns'] for row, _, stopped, _, reasons in results if stopped and reasons & StopReason.CREEP)
  # The first fresh permission discards its preceding budget. Permanent loss
  # then remains bounded by the original one-second guard and one control gap.
  assert FIRST_PERMISSION_WITHDRAWAL_NS <= first_stop_ns <= FIRST_PERMISSION_NS + round(CREEP_MAX_T * 1e9) + MAX_CONTROL_GAP_NS
  held = [result for result in results if result[0]['standstill']]
  assert held and all(stopped and reasons & StopReason.HOLD for _, _, stopped, _, reasons in held)
