"""Message and planner factories shared by longitudinal component tests."""

from contextlib import contextmanager
from unittest.mock import DEFAULT, Mock, patch

import numpy as np

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, CarControllerParams
from openpilot.cereal import messaging
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION, MotionIntent
from openpilot.selfdrive.modeld.constants import ModelConstants


class MockSubMaster(dict):
  def __init__(self, services):
    super().__init__(services)
    self.valid = dict.fromkeys(services, True)
    self.logMonoTime = dict.fromkeys(services, 0)
    self.updated = dict.fromkeys(services, True)
    self.recv_frame = dict.fromkeys(services, 1)

  def all_checks(self, service_list=None) -> bool:
    return True


def build_sm(v_ego: float, force_decel: bool, enabled: bool = True, experimental_mode: bool = False,
             v_cruise_kph: float = 100.0):
  services = {}
  for service in ("radarState", "vehicleParameters", "carStateSP", "gpsLocationExternal", "gpsLocation"):
    services[service] = getattr(messaging.new_message(service), service)

  controls_state = messaging.new_message('controlsState')
  controls_state.controlsState.forceDecel = force_decel
  services['controlsState'] = controls_state.controlsState.as_reader()

  car_state = messaging.new_message('carState')
  car_state.carState.vEgo = v_ego
  car_state.carState.vCruise = v_cruise_kph
  car_state.carState.vCruiseCluster = v_cruise_kph
  services['carState'] = car_state.carState.as_reader()

  selfdrive_state = messaging.new_message('selfdriveState')
  selfdrive_state.selfdriveState.enabled = enabled
  selfdrive_state.selfdriveState.experimentalMode = experimental_mode
  services['selfdriveState'] = selfdrive_state.selfdriveState.as_reader()

  car_control = messaging.new_message('carControl')
  car_control.carControl.enabled = enabled
  services['carControl'] = car_control.carControl.as_reader()

  model = messaging.new_message('modelV2')
  model.modelV2.orientationRate.z = [1e-3] * 33  # straight ahead, but non-zero to keep SCC out of the way
  model.modelV2.velocity.x = [v_ego] * 33
  model.modelV2.position.x = [float(i) for i in range(33)]
  # the model is asymptotically converging: it asks for no speed, but never dips below the
  # actuator dead band, which is exactly the case should_stop() cannot resolve on its own
  model.modelV2.action.desiredAcceleration = -0.1
  services['modelV2'] = model.modelV2.as_reader()

  return MockSubMaster(services)


def braking_state(v_ego=1.3, active=True, gas_pressed=False, brake_pressed=False, override=False):
  state = build_sm(v_ego, False)
  for service in ('carState', 'controlsState', 'carControl'):
    state[service] = state[service].as_builder()
  state['controlsState'].longControlState = LongCtrlState.pid if active else LongCtrlState.off
  state['carState'].gasPressed = gas_pressed
  state['carState'].brakePressed = brake_pressed
  state['carControl'].cruiseControl.override = override
  return state


def make_planner(speed=20.0, *, platform=CAR.HONDA_ACCORD, stock=False, interceptor=False, trace=None, initial_accel=0.0,
                 diagnostics=None):
  params = CarInterface.get_non_essential_params(platform)
  params.openpilotLongitudinalControl = not stock
  extension = CarInterface.get_non_essential_params_sp(params, platform)
  extension.enableGasInterceptor = interceptor
  return LongitudinalPlanner(params, extension, init_v=speed, init_a=initial_accel, trace=trace, diagnostics=diagnostics), extension


def make_state(speed=20.0, target=22.0, experimental=False):
  state = braking_state(speed)
  for service in ('modelV2', 'selfdriveState'):
    state[service] = state[service].as_builder()
  state['carState'].vCruise = target * 3.6
  state['carState'].vCruiseCluster = target * 3.6
  state['carState'].standstill = speed == 0.0
  state['carState'].aEgo = 0.0
  state['carControl'].enabled = True
  state['carControl'].longActive = True
  state['controlsState'].motionState.version = MOTION_INTENT_VERSION
  state['controlsState'].motionState.planningMode = int(LongitudinalMode.UNIFIED)
  state['controlsState'].motionState.enabled = True
  state['selfdriveState'].enabled = True
  state['selfdriveState'].experimentalMode = experimental
  model = state['modelV2']
  model.position.x = [float(speed * sample) for sample in ModelConstants.T_IDXS]
  model.velocity.x = [float(speed)] * len(ModelConstants.T_IDXS)
  model.acceleration.x = [0.0] * len(ModelConstants.T_IDXS)
  model.orientation.x = [0.0] * len(ModelConstants.T_IDXS)
  model.orientationRate.z = [0.001] * len(ModelConstants.T_IDXS)
  model.action.desiredAcceleration = 0.5
  model.meta.disengagePredictions.gasPressProbs = [1.0] * 6
  for service in state.logMonoTime:
    state.logMonoTime[service] = 1_000_000_000
  return state


def advance(state, frame):
  timestamp = 1_000_000_000 + frame * 50_000_000
  for service in state.logMonoTime:
    state.logMonoTime[service] = timestamp
  return timestamp


@contextmanager
def patch_sampled_accel(planner, *, return_value=DEFAULT, side_effect=None):
  """Inject only the scalar action acceleration after real MPC sampling.

  Native solves and trajectory-array reads remain intact. The yielded mock
  records only scalar reads, so tests can change their injected value or
  verify that a fault reached the actual action-sampling boundary.
  """
  native_sample = planner.mpc.sample_solution
  action_accel = Mock(return_value=return_value, side_effect=side_effect)

  def sample_solution(times):
    sampled = native_sample(times)
    if np.ndim(times) == 0:
      return sampled[0], action_accel(times), sampled[2]
    return sampled

  with patch.object(planner.mpc, 'sample_solution', side_effect=sample_solution):
    yield action_accel


class Publisher:
  def __init__(self):
    self.messages = {}

  def send(self, service, message):
    self.messages[service] = message.as_reader()


def make_planner_owner(speed=20.0, initial_accel=0.0, *, trace=None):
  planner, extension = make_planner(speed, initial_accel=initial_accel, trace=trace)
  return planner, LongControl(planner.CP, extension)


def make_handover_state(speed=20.0, target=22.0, experimental=False):
  state = make_state(speed, target, experimental)
  state['carState'].canValid = True
  return state


def consume(planner, controller, state, frame, *, active=None):
  """Produce one dated plan and consume its real wire identity at five ticks."""
  model_time = advance(state, frame)
  planner.update(state, now_ns=model_time + 500_000)
  publisher = Publisher()
  with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=model_time + 1_000_000):
    planner.publish(state, publisher)
  message = publisher.messages['longitudinalPlan']
  plan = message.longitudinalPlan
  if active is None:
    active = (state['selfdriveState'].enabled and state['carControl'].enabled and state['carControl'].longActive
              and not state['carControl'].cruiseControl.override)
  outputs = [float(controller.update(
    active, state['carState'], plan.aTarget, plan.shouldStop,
    (CarControllerParams.BOSCH_ACCEL_MIN, CarControllerParams.BOSCH_ACCEL_MAX),
    intent=MotionIntent.from_message(plan.motionIntent), plan_time_ns=message.logMonoTime, model_time_ns=plan.modelMonoTime,
    now_ns=model_time + 2_000_000 + tick * 10_000_000, plan_valid=message.valid,
    solver_status=plan.solverStatus, braking_plan_valid=state.all_checks(['carState']),
  )) for tick in range(5)]
  return plan, outputs
