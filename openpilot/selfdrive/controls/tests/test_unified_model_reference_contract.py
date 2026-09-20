"""Model-speed boundaries through native planning and the actual stop owner.

These are constructed, open-loop inputs. They protect stop/departure semantics
against an unconditional model-speed translation; they do not qualify a model
speed-bias correction, physical stopping distance or vehicle brake release.
"""

import numpy as np
import pytest

from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import should_stop
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionReferenceKind, MotionResponseKind, StopReason
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import serialized
from openpilot.selfdrive.controls.tests.test_unified_startup import CONTROL_STEP_NS, PLANNER_STEPS, StartupHarness
from openpilot.selfdrive.modeld.constants import ModelConstants


PLAN_STEP_NS = round(DT_MDL * 1e9)


def model_stop_curve(initial_speed, stop_time):
  """Analytic velocity, position and acceleration of a quadratic approach to rest.

  Negative initial speed describes backward motion returning to rest. Its
  positive acceleration alone is not a prediction of forward departure.
  """
  times = np.asarray(ModelConstants.T_IDXS)
  elapsed = np.minimum(times, stop_time)
  remaining = np.maximum(1.0 - times / stop_time, 0.0)
  velocity = initial_speed * remaining ** 2
  position = initial_speed * (elapsed - elapsed ** 2 / stop_time + elapsed ** 3 / (3.0 * stop_time ** 2))
  acceleration = -2.0 * initial_speed / stop_time * remaining
  return position, velocity, acceleration


def model_departure_curve(initial_speed):
  times = np.asarray(ModelConstants.T_IDXS)
  # An independently specified forward control with constant 0.6 m/s^2.
  # The action head remains a separate input below.
  return initial_speed * times + 0.3 * times ** 2, initial_speed + 0.6 * times, np.full(len(times), 0.6)


class ModelReferenceHarness(StartupHarness):
  """Reuse one production planner/LongControl/MotionSupervisor/card chain."""

  def __init__(self, *, speed):
    self.model_curve = None
    self.action_accel = 0.5
    self.cached_model = None
    super().__init__(speed=speed)
    self.selfdrive.experimental_mode = True

  def model_event(self):
    # Keep model identity and data held between 20 Hz publications while the
    # inherited harness executes controlsd and card at 100 Hz.
    if self.cached_model is None or self.frame % PLANNER_STEPS == 0:
      event = super().model_event()
      if self.model_curve is not None:
        position, velocity, acceleration = self.model_curve
        model = event.modelV2
        model.position.x = position.tolist()
        model.velocity.x = velocity.tolist()
        model.acceleration.x = acceleration.tolist()
        model.action.desiredAcceleration = self.action_accel
        # A steady, independent action head: the production modeld predicate
        # is evaluated at measured speed before smoothing. In particular,
        # negative action at 10 m/s does not manufacture a high-speed stop bit.
        model.action.shouldStop = should_stop(self.source_state.vEgo, self.action_accel)
      self.cached_model = serialized(event)
    return self.cached_model

  def exercise(self, curve, *, action_accel, seconds=0.8):
    while self.frame % PLANNER_STEPS:
      self.step()
    self.model_curve = curve
    self.action_accel = action_accel
    owner = self.controls.LoC.motion_supervisor
    native_calls_before = self.mpc_update.call_count
    solves_before = self.planner.mpc.diagnostic_solve_calls
    published_before = len(self.planner_publisher.history)

    # Publish the first new model/plan at the next 20 Hz boundary. Its first
    # control consumer is the following 100 Hz tick, and card applies that
    # command one tick later. Do not judge the preceding command as its output.
    for _ in range(PLANNER_STEPS):
      self.step()
    first_plan = self.latest['longitudinalPlan']
    first_plan_time = int(first_plan.logMonoTime)
    rows = []
    for _ in range(round(seconds / DT_CTRL)):
      row = self.step()
      motion = self.latest['controlsState'].controlsState.motionState
      row = row | {'resolved_stop': bool(motion.resolvedStop), 'model_mono': int(motion.modelMonoTime)}
      rows.append(row)
      assert self.controls.LoC.motion_supervisor is owner
      assert row['long_active'], row
      assert not row['degraded'] and not row['comm_issue'], row
      assert row['consumed_plan_mono'] >= first_plan_time, row

    plans = [event for service, event in self.planner_publisher.history[published_before:] if service == 'longitudinalPlan']
    references = [call.kwargs['reference'] for call in self.mpc_update.call_args_list[native_calls_before:]]
    can_rows = [command for command in self.native_commands if command['mono'] > rows[0]['mono']]
    assert references and can_rows
    assert len(references) == len(plans) == self.planner.mpc.diagnostic_solve_calls - solves_before
    for event in plans:
      assert event.valid and event.longitudinalPlan.solverStatus == 0
      assert event.longitudinalPlan.motionIntent.valid and event.longitudinalPlan.motionIntent.ready
      assert event.longitudinalPlan.modelMonoTime == event.logMonoTime
    np.testing.assert_array_equal(np.diff([event.logMonoTime for event in plans]), PLAN_STEP_NS)
    np.testing.assert_array_equal(np.diff([row['mono'] for row in rows]), CONTROL_STEP_NS)
    assert all(row['model_mono'] == row['consumed_plan_mono'] for row in rows)
    assert all(command['long_active'] for command in can_rows)
    return references, plans, rows, can_rows


def held_harness(measured_speed):
  harness = ModelReferenceHarness(speed=5.0)
  harness.hold()
  assert harness.history[-1]['phase'] == 'hold'
  # Clear the separate OEM standstill protocol in both the negative and
  # positive controls. Otherwise it could mask a false owner release on CAN.
  # This is supplied sensor status, not a claimed physical response to CAN.
  harness.source_state.cruiseState.standstill = False
  harness.source_state.vEgo = measured_speed
  harness.source_state.standstill = measured_speed == 0.0
  return harness


@pytest.mark.parametrize('action_accel', [-1.0, 0.5], ids=['negative_action', 'positive_action'])
def test_measured_speed_bias_does_not_lift_the_models_zero_speed_destination(action_accel):
  harness = ModelReferenceHarness(speed=10.0)
  harness.engage()
  curve = model_stop_curve(9.4, 5.0)
  references, plans, rows, can_rows = harness.exercise(curve, action_accel=action_accel)

  assert harness.source_state.vEgo == 10.0
  assert curve[1][0] == 9.4 and curve[1][-1] == 0.0
  assert not harness.latest['modelV2'].modelV2.action.shouldStop
  for reference in references:
    # Assert the contract actually delivered to native MPC, not another call
    # to the helper under test. A +0.6 m/s translation would fail this check.
    assert reference.model_velocity[-1] == 0.0
    assert reference.model_acceleration[-1] == 0.0
  assert all(not event.longitudinalPlan.motionIntent.reasons & int(StopReason.E2E) for event in plans)
  assert all(row['accel'] < 0.0 for row in rows)
  assert all(command['GAS_COMMAND'] <= 0 for command in can_rows)
  assert any(command['ACCEL_COMMAND'] < -0.2 and command['BRAKE_REQUEST'] for command in can_rows)


@pytest.mark.parametrize('measured_speed', [0.0, 0.2])
@pytest.mark.parametrize('action_accel', [-0.5, 0.5], ids=['negative_action', 'positive_action'])
def test_backward_prediction_returning_to_rest_cannot_release_the_actual_hold(measured_speed, action_accel):
  harness = held_harness(measured_speed)
  # Compare against the actual Float32 field, preserving an exact no-mutation check.
  measured_input = harness.source_state.as_reader().vEgo
  curve = model_stop_curve(-0.4, 2.5)
  references, plans, rows, can_rows = harness.exercise(curve, action_accel=action_accel)

  assert harness.source_state.vEgo == measured_input
  assert curve[1][0] == -0.4 and np.max(curve[1]) == 0.0
  for reference in references:
    assert np.max(reference.model_velocity) == 0.0
    assert reference.model_position[-1] == 0.0
  for event in plans:
    intent = event.longitudinalPlan.motionIntent
    assert not intent.releaseAllowed
    if action_accel > 0.0:
      # This arm must retain the stop without the independent E2E stop bit.
      assert not harness.latest['modelV2'].modelV2.action.shouldStop
      assert not intent.reasons & int(StopReason.E2E)
      assert intent.reasons & int(StopReason.MPC)
    else:
      assert intent.reasons & int(StopReason.E2E)
  assert all(row['resolved_stop'] and row['phase'] in ('hold', 'approach') and not row['resume'] for row in rows)
  assert all(command['BRAKE_REQUEST'] and command['GAS_COMMAND'] <= 0 and not command['STANDSTILL_RELEASE']
             and not command['resume'] for command in can_rows)


@pytest.mark.parametrize('measured_speed', [0.0, 0.2])
def test_actual_forward_model_departure_can_release_the_same_hold_and_reach_honda_can(measured_speed):
  harness = held_harness(measured_speed)
  references, plans, rows, can_rows = harness.exercise(model_departure_curve(measured_speed), action_accel=0.5)

  assert not harness.latest['modelV2'].modelV2.action.shouldStop
  evidence = harness.planner.reference_planner.model_evidence
  assert evidence.raw_motion.kind == 'forward' and evidence.allows_departure
  assert evidence.raw_velocity[-1] > measured_speed
  np.testing.assert_array_equal(evidence.raw_velocity, harness.latest['modelV2'].modelV2.velocity.x)
  assert all(reference.model_position is None and reference.model_velocity is None
             and reference.model_acceleration is None and reference.typed_model is None
             and reference.cruise_speed > 0.0 for reference in references)
  assert all(event.longitudinalPlan.motionIntent.referenceKind == MotionReferenceKind.FORWARD
             and event.longitudinalPlan.motionIntent.responseKind == MotionResponseKind.ABSOLUTE_ACC for event in plans)
  assert any(event.longitudinalPlan.motionIntent.releaseAllowed for event in plans)
  assert any(row['phase'] == 'release' and not row['resolved_stop'] for row in rows)
  assert any(row['accel'] > 0.0 for row in rows)
  assert any(command['STANDSTILL_RELEASE'] and not command['BRAKE_REQUEST'] and command['GAS_COMMAND'] > 0
             for command in can_rows)
  assert all(not command['BRAKE_REQUEST'] or command['GAS_COMMAND'] <= 0 for command in can_rows)
