"""Controller and Honda CAN factories shared by longitudinal tests."""

import numpy as np
import pytest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, structs
from opendbc.car.honda.hondacan import CanBus, create_acc_commands
from opendbc.car.honda.values import CAR, DBC, CarControllerParams, HondaFlags
from openpilot.cereal import custom
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState, STOPPING_DECEL_RATE
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent, MotionPhase, StopReason


ACCEL_LIMITS = (-3.5, 2.0)


def normal_intent(mode, timestamp, speed, **changes):
  """Explicit healthy synthetic producer input; absent evidence is not repaired."""
  return MotionIntent(mode=int(mode), **changes)


def controller(*, platform=CAR.HONDA_ACCORD, openpilot_long=True, interceptor=False, initial=0.0):
  parameters = structs.CarParams()
  parameters.carFingerprint = platform
  parameters.openpilotLongitudinalControl = openpilot_long
  parameters.stopAccel = -2.0
  parameters.longitudinalTuning.kiBP = [0.0]
  parameters.longitudinalTuning.kiV = [0.0]
  parameters_sp = custom.CarParamsSP.new_message()
  parameters_sp.enableGasInterceptor = interceptor
  control = LongControl(parameters, parameters_sp)
  control.last_output_accel = initial
  return control


class ControlHarness:
  def __init__(self, control=None):
    self.control = controller() if control is None else control
    self.state = structs.CarState()
    self.state.vEgo = 1.0
    self.now_ns = 2_000_000_000
    self.plan_time_ns = self.now_ns
    self.model_time_ns = self.now_ns

  def normal_intent(self, *, new_input=True, timestamp=None, **changes):
    if timestamp is None:
      timestamp = self.now_ns + round(DT_CTRL * 1e9) if new_input else self.model_time_ns
    return normal_intent(self.control.longitudinal_mode, timestamp, self.state.vEgo, **changes)

  def step(self, a_target=-0.1, intent=None, active=True, should_stop=False, new_input=True, accel_limits=ACCEL_LIMITS, **kwargs):
    self.now_ns += round(DT_CTRL * 1e9)
    if new_input:
      self.plan_time_ns = self.now_ns
      self.model_time_ns = self.now_ns
    return self.control.update(
      active, self.state, a_target, should_stop, accel_limits, intent=MotionIntent() if intent is None else intent,
      plan_time_ns=kwargs.get('plan_time_ns', self.plan_time_ns), model_time_ns=kwargs.get('model_time_ns', self.model_time_ns),
      now_ns=kwargs.get('now_ns', self.now_ns), plan_valid=kwargs.get('plan_valid', True),
      braking_plan_valid=kwargs.get('braking_plan_valid', False), solver_status=kwargs.get('solver_status', 0))

  def hold(self):
    self.state.vEgo = 0.0
    self.state.standstill = True
    for _ in range(150):
      self.step(intent=MotionIntent(reasons=StopReason.MPC))
    assert self.control.motion_phase == MotionPhase.HOLD
    assert -2.0 - STOPPING_DECEL_RATE * DT_CTRL <= self.control.last_output_accel <= -2.0


def warm_tracking_harness(speed, acceleration):
  """Establish executed TRACK history; a manually seeded OFF request is not history."""
  harness = ControlHarness()
  harness.state.vEgo = float(speed)
  harness.state.aEgo = float(acceleration)
  for _ in range(20):
    harness.step(acceleration, MotionIntent(allow_creep=False))
  assert harness.control.motion_phase == MotionPhase.TRACK
  assert harness.control.long_control_state == LongCtrlState.pid
  assert not harness.control.motion_degraded
  assert harness.control.last_output_accel == pytest.approx(acceleration)
  return harness


def decoded_commands(accelerations):
  params = structs.CarParams()
  params.flags = int(HondaFlags.BOSCH)
  params.openpilotLongitudinalControl = True
  params.init('safetyConfigs', 1)
  bus = CanBus(params)
  dbc_name = DBC[CAR.HONDA_ACCORD][Bus.pt]
  packer = CANPacker(dbc_name)
  parser = CANParser(dbc_name, [('ACC_CONTROL', 50)], bus.pt)
  result = []
  for index, accel in enumerate(accelerations):
    control = structs.CarControl.new_message()
    control.actuators.accel = float(accel)
    serialized = control.as_reader().actuators.accel
    gas = float(np.interp(serialized, CarControllerParams.BOSCH_GAS_LOOKUP_BP, CarControllerParams.BOSCH_GAS_LOOKUP_V))
    messages = create_acc_commands(packer, bus, True, True, serialized, gas, 0, params)
    parser.update([((index + 1) * 20_000_000, messages)])
    result.append(dict(parser.vl['ACC_CONTROL']))
  return result
