"""Rest-rejection braking across the real planner, consumer and Honda CAN path.

These fixed software inputs test command continuity, not vehicle response or
fulfilment of an unreachable model deadline. The original deadline tests stay
independent: a fresh zero source must not revive a not-yet-due RestGoal.
"""

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import MPC_GOAL_INFEASIBLE
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION, MotionReferenceKind
from openpilot.selfdrive.controls.lib.reference_timing import ReferenceTimingConsumer
from openpilot.selfdrive.controls.tests.test_typed_model_reference import MODEL_TIMES
from openpilot.selfdrive.controls.tests.test_typed_reference_mpc import StrictPlanTransportHarness
from openpilot.selfdrive.controls.tests.test_unified_controlsd import CONTROL_STEP_NS, serialized
from openpilot.selfdrive.controls.tests.test_unified_model_reference_contract import model_stop_curve
from openpilot.selfdrive.controls.tests.test_unified_startup import PLANNER_STEPS


@pytest.fixture(autouse=True)
def isolated_params():
  with OpenpilotPrefix():
    yield


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


def failed_stop_then_zero(*, late_zero=False):
  harness = StrictPlanTransportHarness(speed=10.0)
  harness.engage()
  failed = publish_curve(harness, model_stop_curve(10.0, 1.0))
  plan = failed.longitudinalPlan
  assert plan.solverStatus == MPC_GOAL_INFEASIBLE
  assert plan.motionIntent.restGoalRejected and plan.motionIntent.brakingValid
  assert not plan.motionIntent.valid and not plan.motionIntent.releaseAllowed
  assert harness.planner.mpc.rest_goal.remaining_time_s > 0.05
  assert harness.planner.mpc.diagnostic_last_solver_return == 0
  # Arming requires the actual consumer output; publication alone is not an ACK.
  first = harness.step()
  assert first['long_active'] and first['degraded'] and first['phase'] == 'track'
  assert first['accel'] == pytest.approx(ACCEL_MIN, abs=1e-5)
  current = publish_curve(harness, zero_curve(), model_publication_after_state=late_zero)
  assert any(row['BRAKE_REQUEST'] and row['ACCEL_COMMAND'] <= ACCEL_MIN + 0.01
             for row in harness.native_commands if first['mono'] < row['mono'] <= current.logMonoTime)
  return harness, failed, current


@pytest.mark.parametrize('late_zero', [False, True])
def test_accepted_failed_rest_braking_survives_fresh_zero_without_reviving_deadline(late_zero):
  harness, failed, current = failed_stop_then_zero(late_zero=late_zero)
  rows = []
  for _ in range(30):
    row = harness.step()
    rows.append(row | {'continuity_limit': harness.controls.LoC.motion_supervisor.rest_braking_accel_limit})
  plan = harness.latest['longitudinalPlan'].longitudinalPlan
  evidence = harness.planner.reference_planner.model_evidence
  assert failed.longitudinalPlan.solverStatus == MPC_GOAL_INFEASIBLE
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
  # The aligned first zero solve actually supplies the weaker request. The
  # asynchronous solve may itself request full braking; its qualified raw
  # evidence must still preserve, rather than silently clear, the same cap.
  if not late_zero:
    assert current.longitudinalPlan.aTarget > ACCEL_MIN + 0.1
  assert all(row['continuity_limit'] == pytest.approx(ACCEL_MIN) for row in rows)
  assert all(row['consumed_plan_mono'] >= current.logMonoTime for row in rows)
  # The existing selfdrived fault response can legitimately publish FORCE_DECEL
  # and establish APPROACH; the continuity constraint must not bypass that owner.
  assert all(row['long_active'] and not row['degraded'] and row['phase'] in ('track', 'approach') for row in rows)
  assert all(row['accel'] <= ACCEL_MIN + 1e-5 and not row['resume'] for row in rows)
  can_rows = [row for row in harness.native_commands if row['mono'] > rows[0]['mono']]
  # Bosch's moving PID encoding keeps STANDSTILL_RELEASE=1 even when requesting
  # negative acceleration. It is not cruiseControl.resume.
  assert can_rows and all(row['BRAKE_REQUEST'] and row['GAS_COMMAND'] <= 0.0
                          and row['ACCEL_COMMAND'] <= ACCEL_MIN + 0.01
                          and row['STANDSTILL_RELEASE'] == 1 for row in can_rows)


def test_established_hold_takes_over_the_continuation_with_real_stop_can():
  harness, _, _ = failed_stop_then_zero()
  for _ in range(10):
    harness.step()
  assert harness.controls.LoC.motion_supervisor.rest_braking_accel_limit is not None
  harness.source_state.vEgo, harness.source_state.aEgo = 0.0, 0.0
  harness.source_state.standstill = True
  harness.source_state.cruiseState.standstill = True
  publish_curve(harness, zero_curve())
  rows = [harness.step() for _ in range(30)]
  assert all(row['phase'] == 'hold' and row['long_active'] and not row['resume'] for row in rows)
  assert all(row['accel'] < 0.0 for row in rows)
  assert harness.controls.LoC.motion_supervisor.rest_braking_accel_limit is None
  can_rows = [row for row in harness.native_commands if row['mono'] > rows[2]['mono']]
  assert can_rows and all(row['BRAKE_REQUEST'] and row['GAS_COMMAND'] <= 0.0
                          and row['STANDSTILL'] and not row['STANDSTILL_RELEASE'] for row in can_rows)


def test_isolated_fresh_zero_does_not_create_failed_rest_braking_history():
  harness = StrictPlanTransportHarness(speed=10.0)
  harness.engage()
  publish_curve(harness, zero_curve())
  rows = [harness.step() for _ in range(20)]
  assert harness.planner.reference_planner.model_evidence.rest_goal is None
  assert harness.planner.mpc.solution_status == 0
  assert harness.controls.LoC.motion_supervisor.rest_braking_accel_limit is None
  assert all(row['long_active'] and not row['degraded'] for row in rows)
  assert any(row['accel'] > ACCEL_MIN + 0.1 for row in rows)


@pytest.mark.parametrize('interruption', ['negative_no_forward', 'forward'])
def test_new_nonzero_reference_ends_the_chain_and_later_zero_cannot_revive_it(interruption):
  harness, _, _ = failed_stop_then_zero()
  rows = [harness.step() for _ in range(10)]
  assert all(row['accel'] <= ACCEL_MIN + 1e-5 for row in rows)
  raw = -np.ones(len(MODEL_TIMES)) if interruption == 'negative_no_forward' else 10.0 + 0.6 * MODEL_TIMES
  event = publish_curve(harness, (np.maximum(raw, 0.0) * MODEL_TIMES, raw, np.gradient(raw, MODEL_TIMES)))
  expected = MotionReferenceKind.NO_FORWARD if interruption == 'negative_no_forward' else MotionReferenceKind.FORWARD
  assert event.longitudinalPlan.motionIntent.referenceKind == expected
  rows = [harness.step() for _ in range(20)]
  assert harness.controls.LoC.motion_supervisor.rest_braking_accel_limit is None
  assert all(row['long_active'] and not row['degraded'] for row in rows)
  assert any(row['accel'] > ACCEL_MIN + 0.01 for row in rows)
  publish_curve(harness, zero_curve())
  rows = [harness.step() for _ in range(20)]
  assert harness.controls.LoC.motion_supervisor.rest_braking_accel_limit is None
  assert all(row['long_active'] and not row['degraded'] for row in rows)
  assert any(row['accel'] > ACCEL_MIN + 0.01 for row in rows)


@pytest.mark.parametrize('interruption', ['gas', 'brake', 'cancel'])
def test_driver_authority_clears_a_real_braking_continuation_in_the_same_frame(interruption):
  harness, _, _ = failed_stop_then_zero()
  for _ in range(10):
    harness.step()
  assert harness.controls.LoC.motion_supervisor.rest_braking_accel_limit is not None
  if interruption == 'cancel':
    row = harness.step(button='cancel')
  else:
    setattr(harness.source_state, interruption + 'Pressed', True)
    row = harness.step()
  assert not row['long_active'] and not row['resume']
  assert row['phase'] == 'off' and row['accel'] == 0.0
  assert harness.controls.LoC.motion_supervisor.rest_braking_accel_limit is None


@pytest.mark.parametrize('channel', ['normal', 'braking_only'])
@pytest.mark.parametrize('bad_instance', ['missing', 'different'])
def test_reference_instance_disagreement_cannot_use_either_consumer_entry(channel, bad_instance):
  harness, _, _ = failed_stop_then_zero()
  for _ in range(10):
    harness.step()
  curve = zero_curve() if channel == 'normal' else model_stop_curve(10.0, 1.0)
  published = publish_curve(harness, curve)
  candidate = published.as_builder()
  plan = candidate.longitudinalPlan
  assert plan.motionIntent.valid == (channel == 'normal')
  assert plan.motionIntent.brakingValid == (channel == 'braking_only')
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
  assert harness.controls.LoC.motion_supervisor.rest_braking_accel_limit is None
