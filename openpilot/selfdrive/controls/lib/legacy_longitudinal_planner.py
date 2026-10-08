"""Scalar longitudinal planning retained for platforms outside the UNIFIED gate."""
import math

import numpy as np

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.longitudinal_logging import observe
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.drive_helpers import (
  CREEP_ACCEL, CREEP_FINISH_T, CREEP_MAX_T, CREEP_SPEED, CREEP_STOP_T, STOP_RELEASE_ACCEL, STOP_SPEED,
  get_accel_from_plan, should_stop,
)
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState, MPC_FAILURE_ACCEL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalPlanSource, T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.longitudinal_observation import observe_trace
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  ALLOW_THROTTLE_THRESHOLD, CONTROL_N_T_IDX, MIN_ALLOW_THROTTLE_SPEED, get_coast_accel, get_cruise_accel,
)
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP


def initialize_legacy_reference(planner):
  # These fields remain on the owner for scalar trace consumers.
  planner.stop_latched = False
  planner.creep_t = 0.0
  planner.creep_v = 0.0
  planner.creep_stop = False
  planner.creep_total_t = 0.0


def update_legacy_reference(planner, sm, *, now_ns):
  # Generic reference path for platforms outside the supported UNIFIED gate.
  previous_target = planner.output_a_target
  LongitudinalPlannerSP.update(planner, sm)

  v_ego = sm['carState'].vEgo

  if len(sm['carControl'].orientationNED) == 3:
    # Road grade, not raw pitch: the field carries a static mount residual that would otherwise
    # read as a permanent hill. See PitchBiasEstimator.
    pitch = sm['carControl'].orientationNED[1]
    bias = planner.pitch_bias.update(pitch, v_ego)
    accel_coast = get_coast_accel(pitch - bias)
  else:
    accel_coast = ACCEL_MAX
  planner._observe(coastAccelKnown=True, coastAccel=float(accel_coast))

  v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
  v_cruise = v_cruise_kph * CV.KPH_TO_MS
  if sm['controlsState'].forceDecel:
    v_cruise = 0.0

  long_control_off = sm['controlsState'].longControlState == LongCtrlState.off

  # Reset current state when not engaged, or user is controlling the speed
  reset_state = long_control_off if planner.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
  # PCM cruise speed may be updated a few cycles later, check if initialized
  v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET
  reset_state = reset_state or not v_cruise_initialized

  throttle_probs = sm['modelV2'].meta.disengagePredictions.gasPressProbs
  throttle_prob = throttle_probs[1] if len(throttle_probs) > 1 else 1.0
  planner.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

  steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['vehicleParameters'].angleOffsetDeg

  if reset_state:
    planner.v_desired_filter.x = v_ego
    planner.output_a_target = np.clip(sm['carState'].aEgo, ACCEL_MIN, ACCEL_MAX)
    planner.a_cruise = planner.output_a_target
    planner.creep_t = 0.0
    planner.creep_stop = False
    planner.creep_total_t = 0.0
    planner.stop_latched = False

  # Prevent divergence, smooth in current v_ego
  planner.v_desired_filter.x = max(0.0, planner.v_desired_filter.update(v_ego))

  # No change cost when user is controlling the speed, or when standstill
  prev_accel_constraint = not (reset_state or sm['carState'].standstill)

  # Get new v_cruise and a_target from Smart Cruise Control and Speed Limit Assist
  policy = LongitudinalPlannerSP.update_targets(planner, sm, planner.v_desired_filter.x, planner.output_a_target, v_cruise)
  v_cruise, planner.output_a_target = policy.speed, policy.acceleration
  planner._observe(policyTargetKnown=True, policySpeed=float(v_cruise), policyAccel=float(planner.output_a_target))

  planner.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
  planner.mpc.set_cur_state(planner.v_desired_filter.x, planner.output_a_target)
  if planner.trace is not None:
    observe_trace(planner.trace, "begin", planner, sm, reset_state, v_cruise, accel_coast)
  observe(planner.observation, "mpc_started", planner.observation_attempt, planner.mpc)
  planner.mpc.update(sm['radarState'], personality=sm['selfdriveState'].personality)

  planner.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, planner.mpc.v_solution)
  planner.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, planner.mpc.a_solution)
  planner.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], planner.mpc.j_solution)
  if planner.mpc.solution_status != 0:
    planner._observe(outcome="mpc_" + planner.mpc.failure_reason)
    previous_target = previous_target if math.isfinite(previous_target) else MPC_FAILURE_ACCEL
    failure_target = min(previous_target, MPC_FAILURE_ACCEL)
    failure_e2e_allowed = planner.is_e2e(sm)
    planner._observe(modelAdmissionKnown=True, e2eAllowed=bool(failure_e2e_allowed))
    if failure_e2e_allowed and sm.all_checks(service_list=['modelV2']) and math.isfinite(sm['modelV2'].action.desiredAcceleration):
      failure_target = min(failure_target, sm['modelV2'].action.desiredAcceleration)
    planner.output_a_target = float(np.clip(failure_target, ACCEL_MIN, ACCEL_MAX))
    planner.stop_latched = planner.stop_latched or should_stop(v_ego, planner.output_a_target)
    planner.output_should_stop = planner.output_should_stop or planner.stop_latched or sm['carState'].standstill
    planner.fcw = False
    planner.v_desired_filter.x = max(0.0, v_ego) if math.isfinite(v_ego) else 0.0
    if planner.trace is not None:
      observe_trace(planner.trace, "finish", planner,
                    [(float('nan'), planner.mpc.source, False), (float('nan'), LongitudinalPlanSource.cruise, False)], planner.is_e2e(sm))
    return

  # TODO counter is only needed because radar is glitchy, remove once radar is gone
  planner.fcw = planner.mpc.crash_cnt > 2 and not sm['carState'].standstill
  if planner.fcw:
    cloudlog.info("FCW triggered")

  # Save starting point for next iteration
  a_prev = planner.output_a_target

  action_t =  planner.CP.longitudinalActuatorDelay + DT_MDL
  planner._observe(actionHorizonKnown=True, actionHorizon=float(action_t))
  output_a_target_mpc = get_accel_from_plan(planner.v_desired_trajectory, planner.a_desired_trajectory, CONTROL_N_T_IDX,
                                            action_t=action_t)
  planner._observe(sampledAccelKnown=True, sampledAccel=float(output_a_target_mpc))
  output_should_stop_mpc = should_stop(v_ego, output_a_target_mpc)
  output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
  output_should_stop_e2e = sm['modelV2'].action.shouldStop

  is_e2e = planner.is_e2e(sm)
  planner._observe(modelAdmissionKnown=True, e2eAllowed=bool(is_e2e))

  planner.a_cruise = get_cruise_accel(is_e2e, v_cruise, v_ego,
                                   planner.a_cruise, steer_angle_without_offset, planner.CP, planner.dt,
                                   accel_coast, planner.allow_throttle)
  cruise_should_stop = should_stop(v_ego, planner.a_cruise)

  candidates = [(output_a_target_mpc, planner.mpc.source, output_should_stop_mpc),
                (planner.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop)]
  if is_e2e:
    candidates.append((output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e))

  observe(planner.observation, "reference_candidates", planner.observation_attempt, candidates, planner.a_cruise,
          output_a_target_e2e, output_should_stop_e2e, is_e2e)

  output_a_target, planner.mpc.source, _ = min(candidates, key=lambda c: c[0])
  raw_should_stop = any(should_stop for _, _, should_stop in candidates)
  # Hysteresis on the release side. should_stop() is a bare threshold, `v_ego < STOP_SPEED and
  # a_target < STOP_ACCEL`, and at launch a_target climbs across STOP_ACCEL from underneath while
  # the car is still below STOP_SPEED -- so the state machine can flip at frame rate. Every flip
  # re-enters LongCtrlState.stopping, which re-enters the open loop ramp, which puts the brake
  # back on: a burst of pulses while the car is pulling away. Measured on the road it went from
  # 33/h to 57/h across the 08-25/08-26 pair, worst 13 flips inside 1.3 s.
  #
  # Latching is unchanged -- a stop is still asked for at exactly the same instant -- and only
  # letting go is made to require a plan that clearly wants to move. Re-derived over both road
  # pairs, releasing at 0.3 removed 21% of the flips (75 -> 59 over 83 engaged min) for a median
  # 0.48 s longer hold, and 0.5 cost far more (1.38 s).
  #
  # 0.3 was too high, and the road said so. The cost this comment predicted -- "a lead that inches
  # forward while the plan never asks for more than STOP_RELEASE_ACCEL is a stop this holds
  # through" -- was recorded on 08-28 route 21 t=2837.8: latched at v_ego 0.23 while the lead was
  # pulling away at 1.23 m/s, a_target climbed to a peak of +0.27 and fell back, the latch never
  # released, and the open-loop ramp walked the command from -0.14 to -2.00 and held the car
  # still for 1.31 s while the gap went 5.9 -> 8.5 m. It missed the threshold by 0.03.
  #
  # That is not a tail case: over the 08-27/08-28 pair a_target sits in [0.0, 0.3) on 8-10% of
  # stopped frames with p90 +0.11..+0.30, so 0.3 was sitting on the shoulder of the distribution
  # it is supposed to be clear of. 0.2 is the first rung that is not.
  #
  # Measured on the release ladder over both arms: 08-27 pays nothing at all (58 flips either
  # way, 35.9/h) and buys p90 launch delay 2.07 -> 0.71 s; 08-28 pays 7% more flips (43.9 ->
  # 46.9/h) and buys p90 1.27 -> 0.45 s.
  #
  # What this spends is hysteresis width: the band is now [STOP_ACCEL, 0.2), half of what it was.
  # The chatter it was introduced to stop is bounded by the flip numbers above and by the launch
  # burst count, which is the metric that actually names the complaint -- 0 bursts of >=4 flips
  # within 0.6 s on 08-28, and that is what must not come back.
  if planner.stop_latched:
    planner.stop_latched = not (output_a_target >= STOP_RELEASE_ACCEL or v_ego >= STOP_SPEED)
  else:
    planner.stop_latched = raw_should_stop and not reset_state
  planner.output_should_stop = raw_should_stop or planner.stop_latched

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
  # never arrives, which is the creep itplanner. Judge the band in windows of CREEP_STOP_T and ask
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
    if planner.creep_t == 0.0:
      planner.creep_v = v_ego
    planner.creep_t += planner.dt
    planner.creep_total_t += planner.dt
    if not planner.creep_stop:
      if planner.creep_total_t >= CREEP_MAX_T:
        planner.creep_stop = True
      elif planner.creep_t > CREEP_STOP_T:
        if (planner.creep_v - v_ego) * CREEP_FINISH_T >= v_ego * CREEP_STOP_T:
          planner.creep_t = 0.0
        else:
          planner.creep_stop = True
  else:
    planner.creep_t = 0.0
    planner.creep_stop = False
    planner.creep_total_t = 0.0
  planner.output_should_stop = planner.output_should_stop or planner.creep_stop


  planner.output_a_target = np.clip(output_a_target, ACCEL_MIN, ACCEL_MAX)

  planner.v_desired_filter.x = planner.v_desired_filter.x + planner.dt * (planner.output_a_target + a_prev) / 2.0
  if planner.trace is not None:
    observe_trace(planner.trace, "finish", planner, candidates, is_e2e)
