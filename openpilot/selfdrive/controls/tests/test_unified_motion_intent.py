import numpy as np
import pytest

from opendbc.car.honda.values import CarControllerParams
from opendbc.car.interfaces import ACCEL_MAX
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import CREEP_STOP_T, STOP_RELEASE_ACCEL, STOP_SPEED, get_accel_from_plan
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_planner import CONTROL_N_T_IDX
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS
from openpilot.selfdrive.controls.lib.motion_supervisor import StopReason
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import reference_requests_departure
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import decoded_commands
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import ControlsHarness
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_planner, make_state
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.tools.longitudinal.replay_pedal_switches import publish_plan


def stopped_lead_state(speed, gap=4.0, measured_accel=0.0):
  # Already at the requested low-speed comfort goal; the independent native
  # braking boundary may ask for still more room, never forward creep here.
  state = make_state(speed=speed, target=40.0 / 3.6)
  state['controlsState'].longControlState = LongCtrlState.stopping
  state['carState'].aEgo = measured_accel
  state['carControl'].actuators.accel = -2.0
  lead = state['radarState'].leadOne
  lead.present = True
  lead.dRel = gap
  lead.vLead = 0.0
  lead.vRel = -speed
  lead.aLeadK = 0.0
  lead.aLeadTau = 1.5
  lead.modelProb = 0.99
  lead.yRel = 0.0
  return state


def held_consumer(planner, extension):
  consumer = ControlsHarness()
  consumer.control.CP = planner.CP
  consumer.control.CP_SP = extension
  consumer.control.LoC = LongControl(planner.CP, extension)
  consumer.hold()
  return consumer


def tracking_consumer(planner, extension, previous_request=0.0):
  consumer = ControlsHarness()
  consumer.control.CP = planner.CP
  consumer.control.CP_SP = extension
  consumer.control.LoC = LongControl(planner.CP, extension)
  consumer.plan.aTarget = previous_request
  consumer.step()
  assert consumer.motion.phase == 'track'
  assert consumer.command.actuators.accel == previous_request
  return consumer


def publish_and_consume(planner, state, consumer, plan_step_ns=50_000_000):
  timestamp = consumer.now_ns + plan_step_ns
  for service in state.logMonoTime:
    state.logMonoTime[service] = timestamp
  planner.update(state, now_ns=timestamp)
  event = publish_plan(planner, state, timestamp)
  consumer.events['longitudinalPlan'] = event.as_builder()
  consumer.plan = consumer.events['longitudinalPlan'].longitudinalPlan
  # Consume the real producer metadata; ControlsHarness must not synthesize a
  # replacement source lease for an already published plan.
  published_reference = event.longitudinalPlan.reference.to_dict()
  consumer._reference_plan_time_ns = timestamp
  for name in ('vEgo', 'aEgo', 'standstill', 'vCruise', 'vCruiseCluster'):
    setattr(consumer.car_state, name, getattr(state['carState'], name))
  consumer.car_state.cruiseState.standstill = state['carState'].cruiseState.standstill
  command, motion = consumer.step(now_ns=timestamp, plan_time_ns=timestamp, model_time_ns=timestamp)
  assert consumer.plan.reference.to_dict() == published_reference
  assert event.valid
  assert event.longitudinalPlan.solverStatus == 0
  assert consumer.plan.motionIntent.valid
  assert consumer.plan.motionIntent.ready
  assert not motion.degraded
  return event.longitudinalPlan, command, motion


def assert_kept_stop(command, motion):
  assert command.longActive
  assert motion.phase in ('approach', 'hold')
  assert motion.resolvedStop
  assert not command.cruiseControl.resume
  assert command.actuators.accel <= -2.0
  encoded = decoded_commands([command.actuators.accel])
  assert encoded
  assert all(item['BRAKE_REQUEST'] and item['GAS_COMMAND'] <= 0 for item in encoded)


@pytest.mark.parametrize('residual_speed', [0.01, 0.03, 0.05, 0.1])
@pytest.mark.parametrize('measured_accel', [-0.3, -0.4, -0.5, -1.0])
def test_residual_motion_state_cannot_turn_a_stationary_lead_into_departure(residual_speed, measured_accel):
  planner, extension = make_planner(speed=0.0)
  planner.CP.pcmCruise = False
  planner.v_desired_filter.x = residual_speed
  planner.output_a_target = -2.0
  state = stopped_lead_state(0.0, measured_accel=measured_accel)
  state['carState'].cruiseState.standstill = True
  consumer = held_consumer(planner, extension)
  plan, command, motion = publish_and_consume(planner, state, consumer)
  assert planner.mpc.x0[1] == 0.0
  assert planner.mpc.x0[2] == 0.0
  assert not plan.motionIntent.releaseAllowed
  assert_kept_stop(command, motion)


@pytest.mark.parametrize('protocol_hold', [False, True])
@pytest.mark.parametrize('lead_slot', ['leadOne', 'leadTwo'])
@pytest.mark.parametrize('speed_history,measured_accel', [
  ((0.1, 0.075, 0.05, 0.025, 0.0, 0.0, 0.0, 0.0), -0.4),
  ((0.201, 0.15, 0.1, 0.05, 0.0, 0.0, 0.0, 0.0), -1.0),
])
def test_warm_braking_history_cannot_release_during_approach_or_hold(protocol_hold, lead_slot, speed_history, measured_accel):
  planner, extension = make_planner(speed=speed_history[0])
  planner.CP.pcmCruise = False
  planner.output_a_target = -2.0
  state = stopped_lead_state(speed_history[0], measured_accel=measured_accel)
  if lead_slot == 'leadTwo':
    state['radarState'].leadTwo = state['radarState'].leadOne
    state['radarState'].leadOne.present = False
  state['modelV2'].action.desiredAcceleration = 2.0
  consumer = held_consumer(planner, extension)
  positive_compensation = []
  for speed in speed_history:
    state['carState'].vEgo = speed
    state['carState'].standstill = speed == 0.0
    state['carState'].cruiseState.standstill = protocol_hold and speed == 0.0
    getattr(state['radarState'], lead_slot).vRel = -speed
    state['modelV2'].position.x = [float(speed * sample) for sample in ModelConstants.T_IDXS]
    state['modelV2'].velocity.x = [float(speed)] * len(ModelConstants.T_IDXS)
    plan, command, motion = publish_and_consume(planner, state, consumer)
    # Keep this counterexample to the old inverse independently of the new
    # production sampler. The owner must retain the stop in either case.
    positive_compensation.append(get_accel_from_plan(
      planner.v_desired_trajectory, planner.a_desired_trajectory, CONTROL_N_T_IDX,
      action_t=planner.CP.longitudinalActuatorDelay + DT_MDL))
    if speed > 0.0:
      assert planner.mpc.x0[2] == pytest.approx(measured_accel)
    assert not plan.motionIntent.releaseAllowed
    assert_kept_stop(command, motion)
  assert max(positive_compensation) > 0.2


@pytest.mark.parametrize('scenario', ['clear_cruise', 'departing_lead'])
def test_actual_forward_reference_supplies_explicit_permission_to_leave_hold(scenario, monkeypatch):
  planner, extension = make_planner(speed=0.0)
  planner.CP.pcmCruise = False
  # Independently positive cruise request: isolate its entire response policy
  # while retaining the real MPC, producer permission, owner and CAN output.
  monkeypatch.setattr(planner.reference_planner.cruise_response, 'update', lambda *args, **kwargs: ACCEL_MAX)
  state = stopped_lead_state(0.0, gap=10.0)
  if scenario == 'clear_cruise':
    state['radarState'].leadOne.present = False
  elif scenario == 'departing_lead':
    state['radarState'].leadOne.vLead = 0.8
    state['radarState'].leadOne.vRel = 0.8
  consumer = held_consumer(planner, extension)
  plan, command, motion = publish_and_consume(planner, state, consumer)
  assert plan.motionIntent.releaseAllowed
  assert motion.phase == 'release'
  assert not motion.resolvedStop
  assert command.actuators.accel > -2.0
  assert command.actuators.accel <= -2.0 + 8.0 * 0.01 + 1e-6


def test_distant_stationary_lead_uses_fresh_confirmation_before_a_small_positive_release(monkeypatch):
  planner, extension = make_planner(speed=0.0)
  planner.CP.pcmCruise = False
  # The independent cruise response is positive before confirmation starts;
  # the real lead trajectory still owns the small final request being timed.
  monkeypatch.setattr(planner.reference_planner.cruise_response, 'update', lambda *args, **kwargs: ACCEL_MAX)
  state = stopped_lead_state(0.0, gap=7.0)
  consumer = held_consumer(planner, extension)
  previous_request = -2.0
  released = False
  for frame in range(24):
    state['controlsState'].longControlState = consumer.control.LoC.long_control_state
    # Four cached 10 ms ticks follow each plan below; the next fresh tick is
    # another 10 ms, giving the actual 20 Hz producer / 100 Hz consumer cadence.
    plan, command, motion = publish_and_consume(
      planner, state, consumer, plan_step_ns=50_000_000 if frame == 0 else 10_000_000)
    assert plan.motionIntent.releaseAllowed
    assert 0.0 < plan.aTarget < STOP_RELEASE_ACCEL
    confirmation = consumer.control.LoC.motion_supervisor.release_lead_t
    if frame < 10:
      assert confirmation == pytest.approx((frame + 1) * DT_MDL)
    if frame < 9:
      assert_kept_stop(command, motion)
    elif frame == 9:
      assert confirmation == pytest.approx(CREEP_STOP_T)
      assert motion.phase == 'release' and not motion.resolvedStop
      assert -2.0 < command.actuators.accel <= -2.0 + 0.08 + 1e-6
      released = True
    for tick in range(5):
      if tick:
        command, motion = consumer.step(new_plan=False)
        if not released:
          assert consumer.control.LoC.motion_supervisor.release_lead_t == confirmation
          assert_kept_stop(command, motion)
      assert command.actuators.accel - previous_request <= 0.08 + 1e-6
      assert command.longActive and not motion.degraded
      if released:
        assert not motion.resolvedStop
      previous_request = command.actuators.accel
  assert released and not motion.resolvedStop
  assert motion.phase == 'track' and command.actuators.accel > 0.0


def test_nonnegative_stopping_trajectory_is_not_a_move_intent_despite_positive_scalar():
  stop_time = T_IDXS[2]
  remaining = np.maximum(stop_time - T_IDXS, 0.0)
  speeds = 1.5 / (2.0 * stop_time) * remaining ** 2
  accels = -1.5 * remaining / stop_time
  jerks = np.diff(accels) / np.diff(T_IDXS)
  assert np.all(speeds >= 0.0)
  assert np.all(accels <= 0.0)
  assert get_accel_from_plan(speeds, accels, T_IDXS, action_t=0.55) > 0.7
  assert not reference_requests_departure(speeds, accels, jerks)


@pytest.mark.parametrize('initial_speed,acceleration', [
  (0.0, 0.1),
  (1.0, -0.1),
  (0.02, 0.0),
])
def test_stopping_or_sub_stop_speed_reference_is_not_departure_evidence(initial_speed, acceleration):
  speeds = initial_speed + acceleration * T_IDXS
  accels = np.full(len(T_IDXS), acceleration)
  jerks = np.zeros(len(T_IDXS) - 1)
  assert not reference_requests_departure(speeds, accels, jerks)


@pytest.mark.parametrize('initial_speed,expected', [(0.001, False), (0.01, True)])
def test_forward_feasibility_includes_between_node_velocity_extrema(initial_speed, expected):
  speeds, accels = np.zeros(len(T_IDXS)), np.zeros(len(T_IDXS))
  jerks = np.zeros(len(T_IDXS) - 1)
  speeds[0], accels[0], jerks[0] = initial_speed, -0.1, 4.0
  for index, interval in enumerate(np.diff(T_IDXS)):
    speeds[index + 1] = speeds[index] + accels[index] * interval + 0.5 * jerks[index] * interval ** 2
    accels[index + 1] = accels[index] + jerks[index] * interval
  assert np.all(speeds >= 0.0)
  assert max(speeds[T_IDXS <= CONTROL_N_T_IDX[-1]]) > STOP_SPEED
  assert reference_requests_departure(speeds, accels, jerks) is expected


@pytest.mark.parametrize('force_decel', [False, True])
@pytest.mark.parametrize('lead_slot', ['leadOne', 'leadTwo'])
@pytest.mark.parametrize('previous_request', [0.0, -1.0, -3.0])
def test_track_enters_a_real_stop_even_when_the_delay_compensation_is_positive(force_decel, lead_slot, previous_request):
  planner, extension = make_planner(speed=0.201)
  planner.CP.pcmCruise = False
  planner.output_a_target = previous_request
  state = stopped_lead_state(0.201, measured_accel=-1.0)
  state['controlsState'].forceDecel = force_decel
  if lead_slot == 'leadTwo':
    state['radarState'].leadTwo = state['radarState'].leadOne
    state['radarState'].leadOne.present = False
  consumer = tracking_consumer(planner, extension, previous_request)
  reference_accels = []
  for speed in (0.201, 0.15, 0.1, 0.05, 0.0):
    state['carState'].vEgo = speed
    state['carState'].standstill = speed == 0.0
    state['controlsState'].longControlState = consumer.control.LoC.long_control_state
    getattr(state['radarState'], lead_slot).vRel = -speed
    state['modelV2'].position.x = [float(speed * sample) for sample in ModelConstants.T_IDXS]
    state['modelV2'].velocity.x = [float(speed)] * len(ModelConstants.T_IDXS)
    plan, command, motion = publish_and_consume(planner, state, consumer, plan_step_ns=10_000_000)
    reference_accels.append(get_accel_from_plan(
      planner.v_desired_trajectory, planner.a_desired_trajectory, CONTROL_N_T_IDX,
      action_t=planner.CP.longitudinalActuatorDelay + DT_MDL))
    assert plan.shouldStop
    assert plan.motionIntent.reasons & (int(StopReason.FORCE_DECEL) if force_decel else int(StopReason.MPC))
    for tick in range(5):
      if tick:
        command, motion = consumer.step(new_plan=False)
      assert motion.phase in ('approach', 'hold')
      assert motion.resolvedStop
      assert not motion.degraded
      assert not command.cruiseControl.resume
      assert command.actuators.accel <= min(previous_request, CarControllerParams.BOSCH_GAS_LOOKUP_BP[0])
      assert all(item['GAS_COMMAND'] <= 0 for item in decoded_commands([command.actuators.accel]))
  assert max(reference_accels) > 0.2


@pytest.mark.parametrize('force_decel', [False, True])
@pytest.mark.parametrize('speed', [0.1, 0.5, 1.0, 3.0, 20.0])
@pytest.mark.parametrize('measured_accel', [-3.0, -1.0, 0.0, 1.0])
def test_explicit_zero_speed_goal_never_requests_positive_drive(force_decel, speed, measured_accel):
  planner, extension = make_planner(speed=speed)
  planner.CP.pcmCruise = False
  state = make_state(speed=speed, target=40.0 / 3.6 if force_decel else 0.0)
  state['controlsState'].forceDecel = force_decel
  state['carState'].aEgo = measured_accel
  state['radarState'].leadOne.present = False
  state['radarState'].leadTwo.present = False
  consumer = tracking_consumer(planner, extension)
  plan, command, motion = publish_and_consume(planner, state, consumer)
  expected_reason = StopReason.FORCE_DECEL if force_decel else StopReason.CRUISE
  assert plan.motionIntent.reasons & int(expected_reason)
  assert plan.shouldStop
  assert plan.motionIntent.allowCreep
  assert not plan.motionIntent.releaseAllowed
  assert plan.aTarget <= CarControllerParams.BOSCH_GAS_LOOKUP_BP[0]
  assert motion.resolvedStop
  assert command.actuators.accel <= CarControllerParams.BOSCH_GAS_LOOKUP_BP[0]
  assert not command.cruiseControl.resume
  assert all(item['GAS_COMMAND'] <= 0 for item in decoded_commands([command.actuators.accel]))


@pytest.mark.parametrize('speed,target', [(20.0, 10.0), (1.0, 0.5), (0.8, 0.5)])
def test_no_departure_permission_does_not_turn_nonzero_cruise_into_a_stop(monkeypatch, speed, target):
  planner, extension = make_planner(speed=speed)
  state = make_state(speed=speed, target=target)
  state['radarState'].leadOne.present = False
  state['radarState'].leadTwo.present = False
  # A lower cruise request no longer implies a particular internal MPC motion
  # shape. Make this permission-negative fixture explicit, retaining the real
  # negative cruise request and all stop/owner/output decisions below.
  monkeypatch.setattr('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.reference_requests_departure',
                      lambda *args: False)
  consumer = tracking_consumer(planner, extension)
  # Stay beyond both old creep windows using the actual producer and consumer.
  for _ in range(30):
    plan, command, motion = publish_and_consume(planner, state, consumer)
    assert plan.aTarget < 0.0
    assert not plan.motionIntent.releaseAllowed
    assert not plan.motionIntent.reasons
    assert not plan.motionIntent.allowCreep
    assert not plan.shouldStop
    assert motion.phase == 'track'
    assert not motion.resolvedStop
    assert command.longActive
    assert command.actuators.accel < 0.0
