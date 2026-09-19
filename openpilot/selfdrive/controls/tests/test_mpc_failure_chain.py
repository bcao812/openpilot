from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.hondacan import CanBus
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.cereal import messaging
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState, STARTING_ACCEL_RATE
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  MPC_INVALID_INPUT, MPC_INVALID_SOLUTION, MPC_SOLVER_ERROR, T_IDXS, LongitudinalMpc, MpcReference,
)
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent, StopReason
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_handover_state
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import (
  CruiseTarget, LongitudinalPlannerSP, LongitudinalPlanSource as PlanSourceSP,
)


class PlanPublisher:
  def __init__(self, target_override=None):
    self.target_override = target_override
    self.published = None
    self.received = None

  def send(self, service, message):
    if service != 'longitudinalPlan':
      return
    self.published = messaging.log_from_bytes(message.to_bytes())
    self.received = self.published
    if self.target_override is not None:
      corrupted = self.published.as_builder()
      corrupted.longitudinalPlan.aTarget = self.target_override
      self.received = messaging.log_from_bytes(corrupted.to_bytes())


def inject_failure(planner, failure):
  native_solver = planner.solver
  planner.solver = Mock(wraps=native_solver)
  if failure == 'solver_status':
    planner.solver.solve.return_value = 4
  else:
    planner.solver.solve.return_value = 0

    def nonfinite_solution(stage, field):
      values = native_solver.get(stage, field)
      if stage == 1 and field == 'x':
        values[0] = np.nan
      return values

    planner.solver.get.side_effect = nonfinite_solution
  return native_solver


@pytest.mark.parametrize('speed', [0.0, 0.001, 0.05, 0.2, 0.29, 1.0, 10.0])
@pytest.mark.parametrize('acceleration', [-3.5, -1.5, -0.1, 0.0, 0.5])
@pytest.mark.parametrize('reference', [None, MpcReference(20.0)])
def test_failure_is_a_zero_sentinel_not_a_synthesized_deceleration(speed, acceleration, reference):
  planner = LongitudinalMpc()
  planner.set_cur_state(speed, acceleration)
  inject_failure(planner, 'solver_status')
  absent = SimpleNamespace(present=False, modelProb=0.0)
  planner.update(SimpleNamespace(leadOne=absent, leadTwo=absent), reference=reference)
  assert planner.solution_status == planner.last_solver_status == 4
  assert planner.failure_reason == 'solver_status'
  assert np.isfinite(planner.solve_time) and planner.solve_time > 0.0
  for field in ('x0', 'x_sol', 'u_sol', 'v_solution', 'a_solution', 'j_solution', 'a_prev', 'params', 'yref'):
    values = getattr(planner, field)
    np.testing.assert_array_equal(values, np.zeros_like(values))
  for delay in (0.05, 0.1, 0.55, 1.0):
    assert get_accel_from_plan(planner.v_solution, planner.a_solution, T_IDXS, action_t=delay) == 0.0


@pytest.mark.parametrize('status,reason', [
  (4, 'solver_status'), (MPC_INVALID_INPUT, 'invalid_input'),
  (MPC_INVALID_SOLUTION, 'nonfinite_solution'), (MPC_SOLVER_ERROR, 'solver_exception'),
])
def test_zero_sentinel_preserves_diagnostics_and_timing(status, reason):
  planner = LongitudinalMpc()
  planner.set_cur_state(0.2, -1.5)
  planner.solve_time = 0.0042
  planner.fail(status)
  assert planner.solution_status == planner.last_solver_status == status
  assert planner.failure_reason == reason
  assert planner.solve_time == 0.0042
  assert np.all(planner.x0 == 0.0)
  absent = SimpleNamespace(present=False, modelProb=0.0)
  planner.set_cur_state(10.0, 0.0)
  planner.update(SimpleNamespace(leadOne=absent, leadTwo=absent))
  assert planner.solution_status == 0
  assert planner.failure_reason == ''


def failure_chain(speed, e2e, failure, *, latched=True, previous_target=-1.5, previous_output=None,
                  sp_target=None, model_target=0.3, model_healthy=True, wire_target=None, observations=None,
                  manual=None, recover_after=None, transmit_solver_status=True):
  parameters = CarInterface.get_non_essential_params(CAR.HONDA_ACCORD)
  parameters.openpilotLongitudinalControl = True
  parameters.longitudinalActuatorDelay = 0.5
  parameters_sp = CarInterface.get_non_essential_params_sp(parameters, CAR.HONDA_ACCORD)
  initial = previous_target if np.isfinite(previous_target) else 0.0
  planner = LongitudinalPlanner(parameters, parameters_sp, init_v=speed, init_a=initial)
  longitudinal = LongControl(parameters, parameters_sp)
  assert planner.unified and longitudinal.unified
  controller = CarController({Bus.pt: DBC[parameters.carFingerprint][Bus.pt]}, parameters, parameters_sp)
  parser = CANParser(DBC[parameters.carFingerprint][Bus.pt], [('ACC_CONTROL', 50)], CanBus(parameters).pt)
  control_sp = structs.CarControlSP()
  state = make_handover_state(speed=speed, target=100.0 / 3.6, experimental=e2e)
  state['carState'].aEgo = initial
  state['carState'].cruiseState.available = True
  state['modelV2'].action.desiredAcceleration = model_target
  state['modelV2'].action.shouldStop = False

  # An owned request must have actually passed through the current controller.
  # Merely assigning legacy state/last_output fields cannot establish authority.
  prior_output = previous_target if previous_output is None else previous_output
  warm_request = float(prior_output) if np.isfinite(prior_output) else -0.5
  state['carState'].standstill = latched and speed == 0.0
  warm_intent = MotionIntent(reasons=int(StopReason.MPC) if latched else 0, allow_creep=False)
  for tick in range(20):
    timestamp = 2_000_000_000 - (20 - tick) * round(DT_CTRL * 1e9)
    longitudinal.update(True, state['carState'], warm_request, latched, (ACCEL_MIN, ACCEL_MAX),
                        intent=warm_intent, plan_time_ns=timestamp, model_time_ns=timestamp, now_ns=timestamp)
  if not latched:
    assert longitudinal.last_output_accel == pytest.approx(warm_request)
  state['carState'].standstill = speed == 0.0
  planner.output_a_target = previous_target
  if not np.isfinite(prior_output):
    # Deliberate stored-command corruption, after establishing valid ownership.
    longitudinal.last_output_accel = prior_output

  wrapper = SimpleNamespace(out=state['carState'], v_cruise_factor=1.0, is_metric=False,
                            acc_hud={}, lkas_hud={}, stock_brake={})
  native_solver = inject_failure(planner.mpc, failure)
  publisher = PlanPublisher(wire_target)
  observations = [] if observations is None else observations
  state.valid['modelV2'] = model_healthy
  state.all_checks = lambda service_list=None: all(state.valid[service] for service in (state.valid if service_list is None else service_list))
  commands, plan_rows, control_rows, can_rows = [], [], [], []

  def update_targets(instance, current_state, v_ego, a_ego, v_cruise):
    instance.source = PlanSourceSP.cruise
    instance.cruise_target = CruiseTarget(instance.source, v_cruise, a_ego if sp_target is None else sp_target)
    return instance.cruise_target

  with patch.object(LongitudinalPlannerSP, 'update', return_value=None), \
       patch.object(LongitudinalPlannerSP, 'update_targets', new=update_targets), \
       patch.object(LongitudinalPlanner, 'is_e2e', return_value=e2e):
    for frame in range(round(1.0 / DT_CTRL)):
      timestamp = 2_000_000_000 + frame * round(DT_CTRL * 1e9)
      active = not (manual == 'disable' and frame >= 25)
      state['carControl'].enabled = state['carControl'].longActive = active
      state['selfdriveState'].enabled = active
      if frame >= 25 and manual in ('gas', 'brake'):
        setattr(state['carState'], manual + 'Pressed', True)
      if recover_after is not None and frame >= round(recover_after / DT_CTRL):
        planner.mpc.solver = native_solver
      state['controlsState'].longControlState = longitudinal.long_control_state
      if frame % round(DT_MDL / DT_CTRL) == 0:
        for service in state.logMonoTime:
          state.logMonoTime[service] = timestamp
        planner.update(state, now_ns=timestamp + 500_000)
        with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=timestamp + 1_000_000):
          planner.publish(state, publisher)
        event = publisher.received
        plan = event.longitudinalPlan
        assert plan.solverStatus == planner.mpc.solution_status
        if plan.solverStatus != 0:
          assert not publisher.published.valid
        plan_rows.append((plan.aTarget, plan.shouldStop, longitudinal.resolved_should_stop))
      options = {
        'intent': MotionIntent.from_message(plan.motionIntent), 'plan_time_ns': event.logMonoTime,
        'model_time_ns': plan.modelMonoTime, 'now_ns': timestamp + 2_000_000,
        'plan_valid': event.valid, 'braking_plan_valid': state.all_checks(['carState']),
      }
      if transmit_solver_status:
        options['solver_status'] = int(plan.solverStatus)
      command = float(longitudinal.update(active, state['carState'], plan.aTarget, plan.shouldStop,
                                           (ACCEL_MIN, ACCEL_MAX), **options))
      commands.append(command)
      control_rows.append(longitudinal.long_control_state)
      control = state['carControl'].as_reader().as_builder()
      control.longActive = active and not (state['carState'].gasPressed or state['carState'].brakePressed)
      control.actuators.accel = command
      control.actuators.longControlState = longitudinal.long_control_state
      _, messages = controller.update(control.as_reader(), control_sp, wrapper, timestamp)
      can_values = None
      if frame % 2 == 0:
        parser.update([(timestamp, messages)])
        can_values = dict(parser.vl['ACC_CONTROL'])
        can_rows.append(can_values)
      observations.append({'frame': frame, 'event': event, 'published': publisher.published, 'command': command,
                           'control_state': longitudinal.long_control_state, 'can': can_values, 'long_active': control.longActive})

  return planner, longitudinal, plan_rows, control_rows, commands, can_rows


@pytest.mark.parametrize('speed', [0.0, 0.05, 0.2, 0.29])
@pytest.mark.parametrize('e2e', [False, True])
@pytest.mark.parametrize('failure', ['solver_status', 'nonfinite_solution'])
def test_native_failure_cannot_release_the_owned_stop_or_emit_honda_gas(speed, e2e, failure):
  planner, longitudinal, plan_rows, control_rows, commands, can_rows = failure_chain(speed, e2e, failure)
  assert planner.unified and longitudinal.unified
  assert all(target <= 0.0 and owned_stop for target, _, owned_stop in plan_rows), plan_rows
  assert all(value == LongCtrlState.stopping for value in control_rows), control_rows
  assert all(ACCEL_MIN <= command <= 0.0 for command in commands), commands
  assert can_rows
  assert all(ACCEL_MIN <= row['ACCEL_COMMAND'] <= 0.0 for row in can_rows), can_rows
  assert all(row['BRAKE_REQUEST'] == 1 and row['GAS_COMMAND'] <= 0 for row in can_rows), can_rows


@pytest.mark.parametrize('speed,previous_target', [(0.2, -0.5), (1.0, -0.5), (20.0, -3.0),
                                                 (0.0, 0.3), (0.2, 0.3), (1.0, 0.3)])
def test_unlatched_failure_never_weakens_braking_or_requests_gas(speed, previous_target):
  planner, longitudinal, plan_rows, _, commands, can_rows = failure_chain(
    speed, False, 'solver_status', latched=False, previous_target=previous_target)
  assert planner.unified and longitudinal.unified
  bound = min(previous_target, -0.5)
  assert all(target <= bound for target, _, _ in plan_rows), plan_rows
  assert all(ACCEL_MIN <= command <= bound for command in commands), commands
  assert can_rows
  assert all(ACCEL_MIN <= row['ACCEL_COMMAND'] <= 0.0 for row in can_rows), can_rows
  assert all(row['BRAKE_REQUEST'] == 1 and row['GAS_COMMAND'] <= 0 for row in can_rows), can_rows


@pytest.mark.parametrize('failure,status', [('solver_status', 4), ('nonfinite_solution', MPC_INVALID_SOLUTION)])
def test_published_fault_status_and_invalid_sentinel_survive_wire_round_trip(failure, status):
  observations = []
  failure_chain(0.2, False, failure, observations=observations)
  assert observations
  for row in observations:
    event = row['event']
    plan = event.longitudinalPlan
    assert not event.valid
    assert plan.solverStatus == status
    assert np.all(np.asarray(plan.speeds) == 0.0)
    assert np.all(np.asarray(plan.accels) == 0.0)
    assert np.isfinite(plan.solverExecutionTime)
    assert plan.aTarget <= -1.5
    assert not plan.motionIntent.valid
    assert row['control_state'] == LongCtrlState.stopping
    assert row['command'] <= -1.5


@pytest.mark.parametrize('speed', [0.0, 0.2, 20.0])
def test_wire_fault_preserves_stronger_loc_request_than_the_planner_request(speed):
  observations = []
  _, _, plan_rows, _, commands, can_rows = failure_chain(
    speed, False, 'solver_status', latched=False, previous_target=-0.5, previous_output=-3.0,
    observations=observations)
  assert all(target == -0.5 for target, _, _ in plan_rows)
  assert all(row['event'].longitudinalPlan.solverStatus == 4 for row in observations)
  assert all(command == -3.0 for command in commands)
  assert all(row['ACCEL_COMMAND'] == -3.0 and row['BRAKE_REQUEST'] == 1 and row['GAS_COMMAND'] <= 0 for row in can_rows)


def test_other_invalid_wire_evidence_preserves_braking_when_fault_status_is_omitted():
  _, _, _, _, commands, _ = failure_chain(
    20.0, False, 'solver_status', latched=False, previous_target=-0.5, previous_output=-3.0,
    transmit_solver_status=False)
  # Current protocol has independent publication and intent validity. Omitting
  # the numeric diagnostic alone must not turn their invalid packet healthy.
  assert all(command == -3.0 for command in commands)


@pytest.mark.parametrize('wire_target', [float('nan'), float('inf'), -float('inf'), 2.0])
@pytest.mark.parametrize('speed', [0.0, 0.2, 20.0])
def test_corrupted_wire_target_cannot_override_the_published_failure_status(wire_target, speed):
  observations = []
  _, _, _, _, commands, can_rows = failure_chain(
    speed, False, 'solver_status', latched=False, previous_target=-0.5, previous_output=-1.5,
    wire_target=wire_target, observations=observations)
  assert all(row['published'].longitudinalPlan.aTarget == -0.5 for row in observations)
  assert all(row['event'].longitudinalPlan.solverStatus == 4 for row in observations)
  assert all(np.isfinite(command) and ACCEL_MIN <= command <= -1.5 for command in commands)
  assert all(row['BRAKE_REQUEST'] == 1 and row['GAS_COMMAND'] <= 0 for row in can_rows)


@pytest.mark.parametrize('previous_target', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('speed', [0.0, 5.0])
def test_nonfinite_previous_planner_and_loc_targets_have_a_finite_wire_failure(previous_target, speed):
  observations = []
  _, _, plan_rows, _, commands, can_rows = failure_chain(
    speed, False, 'solver_status', latched=False, previous_target=previous_target, observations=observations)
  assert all(row['event'].longitudinalPlan.solverStatus != 0 and not row['event'].valid for row in observations)
  assert all(np.isfinite(target) and ACCEL_MIN <= target <= -0.5 for target, _, _ in plan_rows)
  assert all(np.isfinite(command) and ACCEL_MIN <= command <= -0.5 for command in commands)
  assert all(row['BRAKE_REQUEST'] == 1 and row['GAS_COMMAND'] <= 0 for row in can_rows)


@pytest.mark.parametrize('sp_target', [1.0, -3.0, float('nan')])
def test_sp_hint_cannot_replace_the_previous_executed_planner_target_during_failure(sp_target):
  _, _, plan_rows, _, commands, can_rows = failure_chain(
    5.0, False, 'solver_status', latched=False, previous_target=-1.5, sp_target=sp_target)
  assert all(target == -1.5 for target, _, _ in plan_rows)
  assert all(command <= -1.5 for command in commands)
  assert all(row['ACCEL_COMMAND'] <= -1.5 and row['GAS_COMMAND'] <= 0 for row in can_rows)


@pytest.mark.parametrize('e2e,healthy,model_target,expected', [
  (True, True, -2.5, -2.5),
  (True, True, -5.0, ACCEL_MIN),
  (False, True, -3.0, -1.5),
  (True, False, -3.0, -1.5),
  (True, True, float('nan'), -1.5),
  (True, True, -float('inf'), -1.5),
])
def test_only_healthy_admitted_model_braking_can_strengthen_the_failure_target(e2e, healthy, model_target, expected):
  _, _, plan_rows, _, commands, can_rows = failure_chain(
    5.0, e2e, 'solver_status', latched=False, previous_target=-1.5,
    model_target=model_target, model_healthy=healthy)
  assert all(target == expected for target, _, _ in plan_rows)
  assert all(command == expected for command in commands)
  assert all(row['ACCEL_COMMAND'] == expected and row['BRAKE_REQUEST'] == 1 and row['GAS_COMMAND'] <= 0 for row in can_rows)


@pytest.mark.parametrize('manual', ['gas', 'brake', 'disable'])
@pytest.mark.parametrize('wire_target', [None, float('nan'), -float('inf')])
def test_manual_exit_has_priority_over_fault_hold_and_malformed_wire_targets(manual, wire_target):
  observations = []
  failure_chain(0.0, False, 'solver_status', wire_target=wire_target, manual=manual, observations=observations)
  before = [row for row in observations if row['frame'] < 25]
  after = [row for row in observations if row['frame'] >= 25]
  assert all(row['command'] <= -1.5 and row['long_active'] for row in before)
  assert all(row['event'].longitudinalPlan.solverStatus != 0 for row in after)
  assert all(row['command'] == 0.0 and row['control_state'] == LongCtrlState.off and not row['long_active'] for row in after)
  assert all(row['can']['BRAKE_REQUEST'] == 0 and row['can']['GAS_COMMAND'] <= 0 for row in after if row['can'] is not None)


@pytest.mark.parametrize('speed', [0.0, 0.2, 5.0])
def test_recovered_solver_restores_valid_protocol_without_an_acceleration_release_step(speed):
  observations = []
  failure_chain(speed, False, 'solver_status', latched=False, previous_target=-1.5,
                recover_after=0.25, observations=observations)
  before = [row for row in observations if row['frame'] < 25]
  after = [row for row in observations if row['frame'] >= 25]
  assert all(row['event'].longitudinalPlan.solverStatus == 4 and not row['event'].valid for row in before)
  assert any(row['event'].longitudinalPlan.solverStatus == 0 and row['event'].valid for row in after)
  assert all(row['event'].valid == (row['event'].longitudinalPlan.solverStatus == 0) for row in after)
  commands = [row['command'] for row in observations]
  assert np.max(np.diff(commands)) <= STARTING_ACCEL_RATE * DT_CTRL + 1e-9
  assert all(np.isfinite(command) and ACCEL_MIN <= command <= ACCEL_MAX for command in commands)
  assert all(not (row['can']['BRAKE_REQUEST'] and row['can']['GAS_COMMAND'] > 0)
             for row in observations if row['can'] is not None)


def can_from_car_control(control, car_state):
  parameters = CarInterface.get_non_essential_params(CAR.HONDA_ACCORD)
  parameters.openpilotLongitudinalControl = True
  parameters_sp = CarInterface.get_non_essential_params_sp(parameters, CAR.HONDA_ACCORD)
  dbc_name = DBC[parameters.carFingerprint][Bus.pt]
  controller = CarController({Bus.pt: dbc_name}, parameters, parameters_sp)
  parser = CANParser(dbc_name, [('ACC_CONTROL', 50)], CanBus(parameters).pt)
  wrapper = SimpleNamespace(out=car_state, v_cruise_factor=1.0, is_metric=False,
                            acc_hud={}, lkas_hud={}, stock_brake={})
  _, messages = controller.update(control, structs.CarControlSP(), wrapper, 1_000_000_000)
  parser.update([(1_000_000_000, messages)])
  return dict(parser.vl['ACC_CONTROL'])


@pytest.mark.parametrize('manual', ['gas', 'brake', 'disable', 'override_event'])
def test_real_controlsd_consumes_published_fault_status_and_manual_exit_before_can(manual):
  from openpilot.selfdrive.controls.tests.test_unified_controlsd import ControlsHarness

  observations = []
  failure_chain(20.0, False, 'solver_status', latched=False, previous_target=-0.5, observations=observations)
  harness = ControlsHarness()
  harness.car_state.vEgo = 20.0
  harness.plan.aTarget = -3.0
  for _ in range(20):
    harness.step()
  assert harness.control.LoC.last_output_accel == -3.0
  harness.events['longitudinalPlan'] = observations[0]['published'].as_builder()
  harness.plan = harness.events['longitudinalPlan'].longitudinalPlan
  harness.car_state.vEgo = 0.0
  harness.car_state.standstill = True
  harness.car_state.cruiseState.standstill = True
  assert harness.plan.solverStatus == 4
  assert not harness.plan.shouldStop
  command, _ = harness.step()
  assert command.longActive
  assert command.actuators.accel == -3.0
  assert not command.cruiseControl.resume
  can_values = can_from_car_control(command, harness.car_state)
  assert can_values['ACCEL_COMMAND'] == -3.0
  assert can_values['BRAKE_REQUEST'] == 1 and can_values['GAS_COMMAND'] <= 0

  if manual in ('gas', 'brake'):
    setattr(harness.car_state, manual + 'Pressed', True)
  elif manual == 'disable':
    harness.selfdrive_state.enabled = False
  else:
    override = messaging.new_message('onroadEvents', 1, valid=True)
    override.onroadEvents[0].overrideLongitudinal = True
    harness.events['onroadEvents'] = override
  command, _ = harness.step()
  assert harness.control.sm['longitudinalPlan'].solverStatus == 4
  assert not command.longActive
  assert not command.cruiseControl.resume
  assert command.actuators.accel == 0.0
  assert command.actuators.longControlState == LongCtrlState.off
  can_values = can_from_car_control(command, harness.car_state)
  assert can_values['BRAKE_REQUEST'] == 0 and can_values['GAS_COMMAND'] <= 0
