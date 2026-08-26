#!/usr/bin/env python3
import math
import numpy as np

import openpilot.cereal.messaging as messaging
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan, should_stop
from openpilot.selfdrive.controls.lib.drive_helpers import CREEP_ACCEL, CREEP_FINISH_T, CREEP_MAX_T, CREEP_SPEED, CREEP_STOP_T
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
J_CRUISE_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MIN = -1.2
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

def get_max_accel(v_ego):
  return np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py

def get_cruise_accel(e2e, v_cruise, v_ego, a_cruise_prev, angle_steers, CP, dt, accel_coast, allow_throttle):
  max_accel = ACCEL_MAX if e2e else get_max_accel(v_ego)

  if not e2e:
    a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
    a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
    a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))
    max_accel = min(max_accel, a_x_allowed)
    if not allow_throttle:
      clipped_accel_coast = max(accel_coast, ACCEL_MIN)
      coast_limit = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [max_accel, clipped_accel_coast])
      max_accel = min(max_accel, coast_limit)

  target_accel = np.clip(v_cruise - v_ego, A_CRUISE_MIN, max_accel)
  j_cruise = np.interp(v_ego, A_CRUISE_MAX_BP, J_CRUISE_VALS)
  target_accel = float(np.clip(target_accel, a_cruise_prev - j_cruise * dt, a_cruise_prev + j_cruise * dt))

  return target_accel


class LongitudinalPlanner(LongitudinalPlannerSP):
  def __init__(self, CP, CP_SP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    LongitudinalPlannerSP.__init__(self, self.CP, CP_SP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True

    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.a_cruise = init_a
    self.output_a_target = init_a
    self.output_should_stop = False
    self.creep_t = 0.0
    self.creep_v = 0.0
    self.creep_stop = False
    self.creep_total_t = 0.0

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  def update(self, sm):
    LongitudinalPlannerSP.update(self, sm)

    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    if sm['controlsState'].forceDecel:
      v_cruise = 0.0

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET
    reset_state = reset_state or not v_cruise_initialized

    throttle_probs = sm['modelV2'].meta.disengagePredictions.gasPressProbs
    throttle_prob = throttle_probs[1] if len(throttle_probs) > 1 else 1.0
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['vehicleParameters'].angleOffsetDeg

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.output_a_target = np.clip(sm['carState'].aEgo, ACCEL_MIN, ACCEL_MAX)
      self.a_cruise = self.output_a_target
      self.creep_t = 0.0
      self.creep_stop = False
      self.creep_total_t = 0.0

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    # Get new v_cruise and a_target from Smart Cruise Control and Speed Limit Assist
    v_cruise, self.output_a_target = LongitudinalPlannerSP.update_targets(self, sm, self.v_desired_filter.x, self.output_a_target, v_cruise)

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.output_a_target)
    self.mpc.update(sm['radarState'], personality=sm['selfdriveState'].personality)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Save starting point for next iteration
    a_prev = self.output_a_target

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                              action_t=action_t)
    output_should_stop_mpc = should_stop(v_ego, output_a_target_mpc)
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    is_e2e = self.is_e2e(sm)

    self.a_cruise = get_cruise_accel(is_e2e, v_cruise, v_ego,
                                     self.a_cruise, steer_angle_without_offset, self.CP, self.dt,
                                     accel_coast, self.allow_throttle)
    cruise_should_stop = should_stop(v_ego, self.a_cruise)

    candidates = [(output_a_target_mpc, self.mpc.source, output_should_stop_mpc),
                  (self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop)]
    if is_e2e:
      candidates.append((output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e))

    output_a_target, self.mpc.source, _ = min(candidates, key=lambda c: c[0])
    self.output_should_stop = any(should_stop for _, _, should_stop in candidates)

    # Anti-creep fallback. should_stop() can only latch below STOP_SPEED, but a car whose actuator
    # has a dead band never gets there: Honda Bosch sends gas and clears BRAKE_REQUEST for any accel
    # above -0.2, so as the plan asymptotically converges its accel crosses -0.2 while the car is
    # still rolling and the speed floors out above STOP_SPEED. The stop can then never latch, which
    # is self-sustaining: to stop, the car must first stop.
    #
    # A stop that is already happening must not be preempted, because LongCtrlState.stopping
    # abandons the PID for an open loop ramp toward stopAccel and latching early hands that ramp a
    # higher speed to take off. But "the car is still shedding speed" is far too weak a test for
    # that: an MPC plan converges asymptotically, so it sheds speed the whole way down and simply
    # never arrives, which is the creep itself. Judge the band in windows of CREEP_STOP_T and ask
    # the question that decides it -- at the rate this window shed speed, will the stop be over
    # within CREEP_FINISH_T? If yes it is a stop, leave it alone. If not it is a creep, however
    # smooth. Once latched the test is over, because from then on the speed coming off is the latch
    # doing its work and re-testing it would let go and grab again for the rest of the stop.
    #
    # Measuring the rate over a window rather than reading carState.aEgo is deliberate: replayed
    # over six logged stops, aEgo in the creep band has a larger standard deviation than its mean
    # (0.24..0.36 against -0.27 m/s^2), and the noise breaks the dwell.
    #
    # The window test alone cannot end a creep, though. It extrapolates linearly, and a converging
    # plan settles into a_target ~ -k * v_ego, which sheds v_ego * k * CREEP_STOP_T per window and
    # so always predicts arrival in 1/k seconds -- and never arrives. For any k above
    # CREEP_STOP_T / CREEP_FINISH_T = 0.4 the test passes every window forever. A logged stop
    # measured k ~ 0.85 and spent the whole crawl re-passing it. CREEP_MAX_T is the answer to that:
    # however convincing each window looks, a stop that has been inside a one-metre-per-second band
    # for that long is a creep.
    #
    # The cap is its own test, not a clause of the window test. Hung off the window it could only
    # be read when creep_t > CREEP_STOP_T, i.e. at total dwell 0.55 / 1.10 / 1.65 s, so it could
    # never fire before 1.65 s -- and replayed over 33 logged stops the window test had already
    # failed by then on every one of them, which made the cap dead code. Read every frame it binds
    # where it says it does: 17 of those 33 stops latch earlier, and the four worst road-test gaps
    # (1.8, 1.9, 2.1, 2.6 m) are all in that set.
    #
    # The band is entered on CREEP_ACCEL, not STOP_ACCEL, because this test has no other sense of
    # direction. STOP_ACCEL is the threshold for "not asking for speed" and sits at +0.1, so the
    # window [0.0, 0.1) -- a plan asking to move off, gently -- accumulates dwell exactly like a
    # creep does. That is not hypothetical: on the 08-26 road pair the latch fired 3 times while
    # the car was ACCELERATING away from a stop, at up to 0.68 m/s, and each one dropped the
    # command from +0.07 to -1.38 and pulled the car back to a standstill. A real creep asks for
    # negative accel throughout -- aTarget below CREEP_SPEED has median -0.07 and p90 +0.12 -- so
    # requiring it to be negative costs no latch (31 -> 30 over 83 engaged min, median latch speed
    # 0.68 -> 0.72 m/s) and removes the launches.
    if v_ego < CREEP_SPEED and output_a_target < CREEP_ACCEL:
      if self.creep_t == 0.0:
        self.creep_v = v_ego
      self.creep_t += self.dt
      self.creep_total_t += self.dt
      if not self.creep_stop:
        if self.creep_total_t >= CREEP_MAX_T:
          self.creep_stop = True
        elif self.creep_t > CREEP_STOP_T:
          if (self.creep_v - v_ego) * CREEP_FINISH_T >= v_ego * CREEP_STOP_T:
            self.creep_t = 0.0
          else:
            self.creep_stop = True
    else:
      self.creep_t = 0.0
      self.creep_stop = False
      self.creep_total_t = 0.0
    self.output_should_stop = self.output_should_stop or self.creep_stop

    self.output_a_target = np.clip(output_a_target, ACCEL_MIN, ACCEL_MAX)

    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.output_a_target + a_prev) / 2.0

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks()

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.present
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_sp(sm, pm)
