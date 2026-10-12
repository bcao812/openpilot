"""Current rest/zero requests across the real planner, consumer and Honda CAN path.

These fixed software inputs test command continuity, not vehicle response or
fulfilment of an unreachable model deadline. The original deadline tests stay
independent: a fresh zero source must not revive a not-yet-due RestGoal.
"""

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import MPC_GOAL_INFEASIBLE
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION, MotionIntent, MotionReferenceKind
from openpilot.selfdrive.controls.lib.reference_timing import ReferenceTimingConsumer
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_typed_model_reference import MODEL_TIMES
from openpilot.selfdrive.controls.tests.test_typed_reference_mpc import StrictPlanTransportHarness
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import CONTROL_STEP_NS, serialized
from openpilot.selfdrive.controls.tests.test_unified_model_reference_contract import model_stop_curve
from openpilot.selfdrive.controls.tests.test_unified_startup import PLANNER_STEPS


def zero_curve():
  raw = np.zeros(len(MODEL_TIMES))
  return raw.copy(), raw, raw.copy()


def publish_curve(harness, curve, *, model_publication_after_state=None):
  """Return a real fresh publication before its first 100 Hz consumer tick."""
  while harness.frame % PLANNER_STEPS:
    harness.step()
  if model_publication_after_state is not None:
    harness.model_publication_after_state = model_publication_after_state
  harness.model_curve = curve
  harness.action_accel = 0.5
  for _ in range(PLANNER_STEPS):
    harness.step()
  event = harness.latest['longitudinalPlan']
  assert event.logMonoTime > harness.controls.sm.logMonoTime['longitudinalPlan']
  return event


def accepted_stop_then_zero(*, late_zero=False):
  harness = StrictPlanTransportHarness(speed=10.0)
  harness.engage()
  stopped = publish_curve(harness, model_stop_curve(10.0, 1.0))
  plan = stopped.longitudinalPlan
  assert plan.solverStatus == 0
  assert not plan.motionIntent.restGoalRejected and not plan.motionIntent.brakingValid
  assert plan.motionIntent.valid and not plan.motionIntent.releaseAllowed
  assert plan.motionIntent.referenceKind == MotionReferenceKind.REST
  assert harness.planner.mpc.rest_goal.remaining_time_s > 0.05
  assert harness.planner.mpc.diagnostic_last_solver_return == 0
  assert harness.planner.mpc.rest_goal_state == 'infeasible'
  assert harness.planner.mpc.effective_rest_state == 'satisfied'
  # Verify the actual consumer request; publication alone is not execution.
  first = harness.step()
  assert first['long_active'] and not first['degraded'] and first['phase'] == 'track'
  confirmed = first['accel']
  assert confirmed < -0.2
  current = publish_curve(harness, zero_curve(), model_publication_after_state=late_zero)
  assert any(row['BRAKE_REQUEST'] and row['ACCEL_COMMAND'] <= confirmed + 0.01
             for row in harness.native_commands if first['mono'] < row['mono'] <= current.logMonoTime)
  return harness, stopped, current, confirmed


@pytest.mark.parametrize('late_zero', [False, True])
def test_current_zero_executes_without_a_previous_rest_cap_or_revived_deadline(late_zero):
  harness, stopped, current, _ = accepted_stop_then_zero(late_zero=late_zero)
  # The comparison owner has the same last command for permitted output slew,
  # but never consumed the old REST. Current zero input must execute equally.
  fresh_owner = LongControl(harness.CP, harness.CP_SP)
  fresh_owner.last_output_accel = harness.controls.LoC.last_output_accel
  rows = []
  for _ in range(30):
    row = harness.step()
    consumed = harness.controls.sm['longitudinalPlan']
    state = harness.controls.sm['carState']
    limits = (harness.controls.LoC.pid.neg_limit, harness.controls.LoC.pid.pos_limit)
    expected = fresh_owner.update(
      row['long_active'], state, consumed.aTarget, consumed.shouldStop, limits,
      intent=MotionIntent.from_message(consumed.motionIntent), plan_time_ns=row['consumed_plan_mono'],
      model_time_ns=int(consumed.modelMonoTime), now_ns=row['mono'], solver_status=consumed.solverStatus)
    assert row['accel'] == pytest.approx(expected, abs=1e-6)
    rows.append(row)
  plan = harness.latest['longitudinalPlan'].longitudinalPlan
  evidence = harness.planner.reference_planner.model_evidence
  assert stopped.longitudinalPlan.solverStatus == 0
  assert evidence.no_forward and evidence.rest_goal is None
  if late_zero:
    assert evidence.unknown.reason == 'model_proxy_after_initial_state'
  else:
    assert evidence.unknown is None
  assert harness.planner.mpc.rest_goal is None
  assert harness.planner.mpc.solution_status == harness.planner.mpc.diagnostic_last_solver_return == 0
  assert plan.solverStatus == 0 and plan.motionIntent.valid
  assert plan.motionIntent.version == MOTION_INTENT_VERSION
  assert plan.motionIntent.referenceKind == MotionReferenceKind.EXACT_ZERO
  assert not plan.motionIntent.restGoalRejected and not plan.motionIntent.releaseAllowed
  assert plan.motionIntent.producerInstanceMonoTime == plan.reference.producerInstanceMonoTime > 0
  assert all(row['consumed_plan_mono'] >= current.logMonoTime for row in rows)
  assert all(row['long_active'] and not row['degraded'] and row['phase'] == 'track' for row in rows)
  assert all(not row['resume'] for row in rows)
  can_rows = [row for row in harness.native_commands if row['mono'] > rows[0]['mono']]
  # Bosch's moving PID encoding keeps STANDSTILL_RELEASE=1 even when requesting
  # negative acceleration. It is not cruiseControl.resume.
  assert can_rows and all(-3.5 <= row['ACCEL_COMMAND'] <= 2.0
                          and row['long_active'] and row['STANDSTILL_RELEASE'] == 1 for row in can_rows)
  commands_by_time = {row['mono']: row['accel'] for row in rows}
  matched = 0
  for packet in can_rows:
    consumed_ns = packet['mono'] - CONTROL_STEP_NS
    if consumed_ns in commands_by_time:
      assert packet['ACCEL_COMMAND'] == pytest.approx(commands_by_time[consumed_ns], abs=0.011)
      matched += 1
  assert matched > 0


def test_current_standstill_and_hold_keep_real_stop_can():
  harness, _, _, _ = accepted_stop_then_zero()
  for _ in range(10):
    harness.step()
  assert not harness.controls.LoC.motion_degraded
  harness.source_state.vEgo, harness.source_state.aEgo = 0.0, 0.0
  harness.source_state.standstill = True
  harness.source_state.cruiseState.standstill = True
  publish_curve(harness, zero_curve())
  rows = [harness.step() for _ in range(30)]
  assert all(row['phase'] == 'hold' and row['long_active'] and not row['resume'] for row in rows)
  assert all(row['accel'] < 0.0 for row in rows)
  can_rows = [row for row in harness.native_commands if row['mono'] > rows[2]['mono']]
  assert can_rows and all(row['BRAKE_REQUEST'] and row['GAS_COMMAND'] <= 0.0
                          and row['STANDSTILL'] and not row['STANDSTILL_RELEASE'] for row in can_rows)


def test_isolated_fresh_zero_is_a_healthy_current_reference():
  harness = StrictPlanTransportHarness(speed=10.0)
  harness.engage()
  publish_curve(harness, zero_curve())
  rows = [harness.step() for _ in range(20)]
  assert harness.planner.reference_planner.model_evidence.rest_goal is None
  assert harness.planner.mpc.solution_status == 0
  assert all(row['long_active'] and not row['degraded'] for row in rows)
  assert any(row['accel'] > ACCEL_MIN + 0.1 for row in rows)


@pytest.mark.parametrize('interruption', ['negative_no_forward', 'forward'])
def test_new_nonzero_and_later_zero_references_keep_current_source_meaning(interruption):
  harness, _, _, _ = accepted_stop_then_zero()
  rows = [harness.step() for _ in range(10)]
  assert all(row['long_active'] and not row['degraded'] for row in rows)
  raw = -np.ones(len(MODEL_TIMES)) if interruption == 'negative_no_forward' else 10.0 + 0.6 * MODEL_TIMES
  event = publish_curve(harness, (np.maximum(raw, 0.0) * MODEL_TIMES, raw, np.gradient(raw, MODEL_TIMES)))
  expected = MotionReferenceKind.NO_FORWARD if interruption == 'negative_no_forward' else MotionReferenceKind.FORWARD
  assert event.longitudinalPlan.motionIntent.referenceKind == expected
  rows = [harness.step() for _ in range(20)]
  assert all(row['long_active'] and not row['degraded'] for row in rows)
  assert any(row['accel'] > ACCEL_MIN + 0.01 for row in rows)
  publish_curve(harness, zero_curve())
  rows = [harness.step() for _ in range(20)]
  assert all(row['long_active'] and not row['degraded'] for row in rows)
  assert any(row['accel'] > ACCEL_MIN + 0.01 for row in rows)


@pytest.mark.parametrize('interruption', ['gas', 'brake', 'cancel'])
def test_driver_authority_clears_current_control_in_the_same_frame(interruption):
  harness, _, _, _ = accepted_stop_then_zero()
  for _ in range(10):
    harness.step()
  assert not harness.controls.LoC.motion_degraded
  if interruption == 'cancel':
    row = harness.step(button='cancel')
  else:
    setattr(harness.source_state, interruption + 'Pressed', True)
    row = harness.step()
  assert not row['long_active'] and not row['resume']
  assert row['phase'] == 'off' and row['accel'] == 0.0


@pytest.mark.parametrize('channel', ['normal', 'braking_only'])
@pytest.mark.parametrize('bad_instance', ['missing', 'different'])
def test_reference_instance_disagreement_cannot_use_either_consumer_entry(channel, bad_instance):
  harness, _, _, _ = accepted_stop_then_zero()
  for _ in range(10):
    harness.step()
  if channel == 'braking_only':
    # A fresh source after this short event now states a current due-rest goal.
    # Unlike future scheduling, that unchanged mathematical failure still takes
    # the independently health-qualified braking-only consumer entry.
    publish_curve(harness, model_stop_curve(10.0, 0.001))
  published = publish_curve(harness, zero_curve())
  candidate = published.as_builder()
  plan = candidate.longitudinalPlan
  assert plan.motionIntent.valid == (channel == 'normal')
  assert plan.motionIntent.brakingValid == (channel == 'braking_only')
  assert plan.solverStatus == (MPC_GOAL_INFEASIBLE if channel == 'braking_only' else 0)
  original_instance = int(plan.reference.producerInstanceMonoTime)
  assert original_instance == plan.motionIntent.producerInstanceMonoTime > 0
  plan.motionIntent.producerInstanceMonoTime = 0 if bad_instance == 'missing' else original_instance + 1
  # All source/availability/consumption times are still qualified. The failure
  # under test is the new inner/outer instance disagreement, not a bad clock.
  timing = ReferenceTimingConsumer().consume(plan.reference, int(candidate.logMonoTime),
                                             harness.now_ns + CONTROL_STEP_NS)
  assert timing.valid
  harness.latest['longitudinalPlan'] = serialized(candidate)
  row = harness.step(run_planner=False)
  check = harness.controls.reference_timing_check
  assert not check.valid and check.clock_valid
  assert check.reason == 'intent_reference_instance_mismatch'
  assert row['degraded'] and not row['resume']
  assert row['accel'] <= 0.0
