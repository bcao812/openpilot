"""Legacy longitudinal state and output logic for other platform capabilities."""

import numpy as np

from opendbc.car.structs import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.motion_supervisor import ROLLING_STOP_ACCEL, STARTING_ACCEL_RATE, STOPPING_DECEL_RATE


LongCtrlState = car.CarControl.Actuators.LongControlState


def long_control_state_trans(CP_SP, active, long_control_state,
                             should_stop, brake_pressed, cruise_standstill):
  # Gas Interceptor
  cruise_standstill = cruise_standstill and not CP_SP.enableGasInterceptor

  starting_condition = (not should_stop and
                        not cruise_standstill and
                        not brake_pressed)

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if not starting_condition:
        long_control_state = LongCtrlState.stopping
      else:
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition:
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.pid:
      if should_stop:
        long_control_state = LongCtrlState.stopping

  return long_control_state


def update_legacy(self, active, CS, a_target, should_stop, accel_limits, *,
                  intent=None, plan_time_ns=None, model_time_ns=None, now_ns=None, plan_valid=True, solver_status=0, braking_plan_valid=False,
                  standby_input=False):
  """Keep legacy stop/ramp behavior; motion-protocol inputs are unused."""
  self.resolved_should_stop = bool(should_stop)
  self.pid.neg_limit = accel_limits[0]
  self.pid.pos_limit = accel_limits[1]

  self.long_control_state = long_control_state_trans(self.CP_SP, active, self.long_control_state,
                                                     should_stop, CS.brakePressed,
                                                     CS.cruiseState.standstill)
  if self.long_control_state == LongCtrlState.off:
    self.reset()
    output_accel = 0.
    self.starting = False

  elif self.long_control_state == LongCtrlState.stopping:
    # The PID is abandoned here: the command walks down open loop and stops reacting to the car
    # at all. Every bit of it below what the car needs is felt, and the car is still rolling --
    # arriving at stopAccel = -2.0 at 0.4 m/s is the jolt at the end of a stop. Hold the ramp at
    # ROLLING_STOP_ACCEL until the wheels stop, then let it go on to stopAccel, where it is only
    # holding pressure against a car that is already still.
    #
    # The ramp inherits last_output_accel, and an MPC plan converges asymptotically, so it starts
    # wherever the PID happened to be -- measured over 33 logged stops, a median of -0.33 m/s^2
    # at 0.7 m/s. At 1.0 m/s^2/s that is a full second before the command carries any authority,
    # and on 8 of 21 stops the car reached standstill before the ramp ever passed -1.4. The car
    # spends that second under-braked below its own creep torque, which is the crawl; when the
    # brake finally crosses that torque it does so all at once, which is the jolt. Walking down
    # twice as fast is the same ramp arriving while it can still do something.
    stop_accel = self.CP.stopAccel if CS.standstill else max(self.CP.stopAccel, ROLLING_STOP_ACCEL)
    output_accel = self.last_output_accel
    if output_accel > stop_accel:
      output_accel = min(output_accel, 0.0)
      output_accel -= STOPPING_DECEL_RATE * DT_CTRL
    self.reset()
    # Whatever the ramp reached is where the next pid frame has to climb out from.
    self.starting = True

  else:  # LongCtrlState.pid
    # Leaving stopping hands the wire a step, not a ramp: last_output_accel is at the bottom of
    # the ramp and the PID's first output is essentially a_target, because the integrator was
    # reset on every stopping frame and k_p is zero. Measured over 52 logged launches that step
    # is a median +2.10 m/s^2 in one 10 ms frame -- 210 m/s^3 -- and Honda Bosch rate-limits
    # neither accel nor gas, so BRAKE_REQUEST clears and GAS_COMMAND rises in the same CAN frame.
    # Bound the climb until the PID catches up. Only the climb: a deeper command is never held
    # back, so this cannot delay braking.
    error = a_target - CS.aEgo
    limit = self.last_output_accel + STARTING_ACCEL_RATE * DT_CTRL
    output_accel = self.pid.update(error, speed=CS.vEgo,
                                   feedforward=a_target, freeze_integrator=self.starting)
    if self.starting:
      if output_accel > limit:
        output_accel = limit
      else:
        self.starting = False

  self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
  return self.last_output_accel
