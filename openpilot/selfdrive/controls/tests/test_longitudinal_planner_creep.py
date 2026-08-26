from typing import cast

from opendbc.car import structs
from openpilot.cereal import custom, messaging
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import (CREEP_ACCEL, CREEP_FINISH_T, CREEP_MAX_T, CREEP_SPEED,
                                                            CREEP_STOP_T, STOP_ACCEL, STOP_SPEED, should_stop)
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController

# above STOP_SPEED, so should_stop() alone can never latch here, and inside the creep band
V_CREEP = 0.5
FRAMES_DWELL = int(CREEP_STOP_T / DT_MDL)
# the window test is only read once creep_t has passed CREEP_STOP_T, so a window is one frame
# longer than that
WINDOW_T = CREEP_STOP_T + DT_MDL


def _frames_to_cap() -> int:
  """The cap is read every frame, so it binds on the first frame whose accumulated dwell reaches
  CREEP_MAX_T. Accumulated the way the planner accumulates it -- DT_MDL does not divide CREEP_MAX_T
  exactly in binary, and a closed form would be off by one."""
  t, n = 0.0, 0
  while t < CREEP_MAX_T:
    t += DT_MDL
    n += 1
  return n


FRAMES_CAP = _frames_to_cap()
# the cap has to be able to fire before the window test would have failed anyway, or it is dead
# code: hung off the window it could only ever be read at 0.55 / 1.10 / 1.65 s of dwell
assert FRAMES_CAP * DT_MDL < 2 * WINDOW_T, "CREEP_MAX_T is not reachable before the second window"


class MockDec:
  """Stand-in for DynamicExperimentalController, so the tests pin the mode instead of the model."""

  def __init__(self, active: bool, mode: str):
    self._active = active
    self._mode = mode

  def update(self, sm):
    pass

  def active(self) -> bool:
    return self._active

  def mode(self) -> str:
    return self._mode

  def enabled(self) -> bool:
    return True


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
  for service in ("radarState", "vehicleParameters", "carStateSP", "liveMapDataSP", "gpsLocationExternal", "gpsLocation"):
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


def build_planner(dec_active: bool = True, dec_mode: str = "acc") -> LongitudinalPlanner:
  CP = structs.CarParams()
  CP.steerRatio = 15.0
  CP.wheelbase = 2.7
  CP.longitudinalActuatorDelay = 0.5  # Honda Bosch
  CP_SP = custom.CarParamsSP.new_message().as_reader()

  planner = LongitudinalPlanner(CP, CP_SP, init_v=V_CREEP)
  planner.dec = cast(DynamicExperimentalController, MockDec(dec_active, dec_mode))
  return planner


def run(planner: LongitudinalPlanner, frames: int, **kwargs) -> None:
  for _ in range(frames):
    planner.update(build_sm(**kwargs))


def run_decel(planner: LongitudinalPlanner, frames: int, v0: float, decel: float, **kwargs) -> float:
  """Feed a speed that is actually coming down, which the constant speed above can never be."""
  v_ego = v0
  for _ in range(frames):
    planner.update(build_sm(v_ego=v_ego, **kwargs))
    v_ego = max(0.0, v_ego - decel * DT_MDL)
  return v_ego


def run_converging(planner: LongitudinalPlanner, frames: int, v0: float, k: float, **kwargs) -> float:
  """A converging plan: the speed comes off in proportion to itself, so it never arrives."""
  v_ego = v0
  for _ in range(frames):
    planner.update(build_sm(v_ego=v_ego, **kwargs))
    v_ego = max(0.0, v_ego * (1.0 - k * DT_MDL))
  return v_ego


def run_pulling_away(planner: LongitudinalPlanner, frames: int, v0: float, accel: float, **kwargs) -> float:
  """A launch: the speed is going UP through the creep band."""
  v_ego = v0
  for _ in range(frames):
    planner.update(build_sm(v_ego=v_ego, **kwargs))
    v_ego += accel * DT_MDL
  return v_ego


class TestAntiCreepNeedsADirection:
  """The band test has no sense of direction of its own.

  STOP_ACCEL is the threshold for "the plan is not asking for speed" and sits at +0.1, so a plan
  asking to move off *gently* -- anywhere in [0.0, 0.1) -- accumulates creep dwell exactly like a
  car that is failing to stop. On the 08-26 road pair that fired 3 times while the car was
  accelerating away from a stop, at up to 0.68 m/s, and each one dropped the command from +0.07 to
  -1.38 and pulled the car back to a standstill. CREEP_ACCEL is what separates the two.
  """

  # a cruise target 0.55 m/s above V_CREEP settles the plan at +0.05 m/s^2: inside the window the
  # old gate admitted, outside the one it should have been using
  V_CRUISE_GENTLE = 1.98

  def test_the_gate_is_below_the_no_speed_threshold(self):
    assert CREEP_ACCEL < STOP_ACCEL, "the gate is not narrower than should_stop()'s"

  def test_no_latch_while_the_plan_asks_to_move_off_gently(self):
    planner = build_planner()
    run(planner, 4 * FRAMES_CAP, v_ego=V_CREEP, force_decel=False, v_cruise_kph=self.V_CRUISE_GENTLE)

    # the premise: this plan is exactly what the old gate could not tell from a creep
    assert CREEP_ACCEL <= planner.output_a_target < STOP_ACCEL, \
        f"the run is not in the gap the fix closes (a_target {planner.output_a_target})"
    assert planner.creep_total_t == 0.0, "accumulated dwell against a plan that wants to move"
    assert not planner.output_should_stop, "latched a launch"

  def test_no_latch_while_the_car_is_pulling_away(self):
    # the road case: already rolling, speed rising, plan asking for a little accel
    planner = build_planner()
    v_end = run_pulling_away(planner, 4 * FRAMES_CAP, v0=0.35, accel=0.05,
                             force_decel=False, v_cruise_kph=self.V_CRUISE_GENTLE)
    assert STOP_SPEED < v_end < CREEP_SPEED, "the run left the band, it is not testing the gate"
    assert not planner.output_should_stop, "pulled a launch back to a standstill"

  def test_a_real_creep_still_latches(self):
    # the gate must not cost the latch its job: a creep asks for negative accel throughout
    planner = build_planner()
    run_converging(planner, FRAMES_CAP, v0=CREEP_SPEED - 0.05, k=0.6, force_decel=True)
    assert planner.output_a_target < CREEP_ACCEL
    assert planner.output_should_stop, "the gate broke the latch it was meant to narrow"


class TestAntiCreepStop:
  def test_should_stop_cannot_latch_in_the_creep_band(self):
    # the premise: above STOP_SPEED no candidate can ever request a stop, however hard it brakes
    assert not should_stop(V_CREEP, -3.0)
    assert STOP_SPEED < V_CREEP < CREEP_SPEED

  def test_latches_after_dwelling_in_the_creep_band(self):
    planner = build_planner()

    run(planner, FRAMES_DWELL // 2, v_ego=V_CREEP, force_decel=True)
    assert planner.output_a_target < 0.0
    assert not planner.output_should_stop, "latched before the dwell elapsed"

    run(planner, FRAMES_DWELL, v_ego=V_CREEP, force_decel=True)
    assert planner.output_should_stop, "never latched, the car would creep forever"

  def test_latches_in_blended_mode_too(self):
    # all six logged stops were in blended mode, so the e2e candidate is in the running as well
    planner = build_planner(dec_active=True, dec_mode="blended")
    run(planner, 2 * FRAMES_DWELL, v_ego=V_CREEP, force_decel=True, experimental_mode=True)
    assert planner.output_should_stop

  def test_no_latch_while_the_plan_asks_for_speed(self):
    # stop and go: the plan wants to move off, so the dwell must never accumulate
    planner = build_planner()
    run(planner, 4 * FRAMES_DWELL, v_ego=V_CREEP, force_decel=False)
    assert planner.output_a_target > 0.0
    assert planner.creep_t == 0.0
    assert not planner.output_should_stop

  def test_no_latch_above_the_creep_band(self):
    # braking hard from speed is not creeping, the ordinary stop logic still owns that
    planner = build_planner()
    run(planner, 4 * FRAMES_DWELL, v_ego=CREEP_SPEED + 1.0, force_decel=True)
    assert planner.output_a_target < 0.0
    assert planner.creep_t == 0.0
    assert not planner.output_should_stop

  def test_releases_when_the_plan_asks_for_speed_again(self):
    # the latch must not outlive its cause, or the car would be stuck stopped
    planner = build_planner()
    run(planner, 2 * FRAMES_DWELL, v_ego=V_CREEP, force_decel=True)
    assert planner.output_should_stop

    # the cruise accel is jerk limited, so give it a moment to climb back out of the band. That
    # climb is a property of J_CRUISE_VALS, not of the window, so it is counted in seconds here
    run(planner, int(1.0 / DT_MDL), v_ego=V_CREEP, force_decel=False)
    assert planner.creep_t == 0.0
    assert not planner.output_should_stop

  def test_disengaging_restarts_the_dwell(self):
    planner = build_planner()
    run(planner, FRAMES_DWELL, v_ego=V_CREEP, force_decel=True)
    assert planner.creep_t > 0.0

    # a stale dwell must not carry across an engagement, or the next one could latch instantly
    run(planner, 1, v_ego=V_CREEP, force_decel=True, enabled=False)
    assert planner.creep_t <= DT_MDL
    assert not planner.output_should_stop


class TestAntiCreepDoesNotPreemptAStop:
  """LongCtrlState.stopping abandons the PID for an open loop ramp toward stopAccel, so latching a
  stop the car is already carrying out only hands that ramp a higher speed to take off. The test
  for "already carrying out" has to be whether the stop will finish, not whether the speed is
  moving: an MPC plan converges asymptotically, so it sheds speed the whole way down and never
  arrives, which is the creep itself."""

  # the window resets only when shed * CREEP_FINISH_T >= v_ego * CREEP_STOP_T, and one window of
  # constant decel sheds decel * CREEP_STOP_T, so the cutoff is decel >= v0 / (CREEP_FINISH_T + 1)
  V0 = CREEP_SPEED - 0.05
  # 1.4x clears the cutoff with margin and still leaves the car above STOP_SPEED at the window
  # boundary, so the ordinary should_stop() has not taken over and the latch is what is on trial
  DECEL_PROMPT = 1.4 * V0 / (CREEP_FINISH_T + CREEP_STOP_T)
  DECEL_CREEP = 0.5 * V0 / (CREEP_FINISH_T + CREEP_STOP_T)

  def test_no_latch_while_the_stop_will_finish_in_time(self):
    planner = build_planner()
    v_end = run_decel(planner, FRAMES_DWELL + 2, v0=self.V0, decel=self.DECEL_PROMPT, force_decel=True)
    assert v_end > STOP_SPEED, "the run left the band, it is not testing the latch"
    assert planner.output_a_target < 0.0
    assert not planner.output_should_stop, "preempted a stop that was going to finish on its own"

  def test_latches_when_the_stop_will_not_finish_in_time(self):
    # the road test case: smooth, unhurried, and never actually arriving
    planner = build_planner()
    run_decel(planner, FRAMES_DWELL + 2, v0=self.V0, decel=self.DECEL_CREEP, force_decel=True)
    assert planner.output_should_stop, "let an asymptotic approach creep on"

  def test_latches_once_a_prompt_stop_floors_out(self):
    # shedding fine, then the dead band bites and it simply stops arriving
    planner = build_planner()
    v_floor = run_decel(planner, FRAMES_DWELL, v0=self.V0, decel=self.DECEL_PROMPT, force_decel=True)
    assert not planner.output_should_stop
    assert v_floor > STOP_SPEED

    run(planner, 3 * FRAMES_DWELL, v_ego=v_floor, force_decel=True)
    assert planner.output_should_stop, "never latched, the car would creep forever"

  def test_the_latch_does_not_let_go_of_the_stop_it_caused(self):
    # Once latched, longcontrol brakes open loop and the speed does start coming off. Re-testing
    # progress against that would release the latch, put the PID back in charge, let the car floor
    # out again and re-latch: the car would chatter its way to a standstill.
    planner = build_planner()
    run(planner, 2 * FRAMES_DWELL, v_ego=V_CREEP, force_decel=True)
    assert planner.output_should_stop

    v = V_CREEP
    for _ in range(2 * FRAMES_DWELL):
      v = max(0.31, v - 0.5 * DT_MDL)  # the stop the latch is carrying out, held above STOP_SPEED
      planner.update(build_sm(v_ego=v, force_decel=True))
      assert planner.output_should_stop, "released the stop it had latched"


class TestAntiCreepTotalTimeCap:
  """The window test extrapolates linearly, and a converging plan settles into a_target ~ -k * v_ego,
  which sheds the same fraction of the speed every window. So it always predicts arrival one over k
  seconds hence, and never arrives -- the window test alone cannot end a creep, only recognise a
  stop. CREEP_MAX_T is what ends it."""

  V0 = CREEP_SPEED - 0.05
  # A window resets when it leaves a fraction CREEP_FINISH_T / (CREEP_FINISH_T + CREEP_STOP_T) = 0.8
  # of the speed, so any rate above -log(0.8) / WINDOW_T ~ 0.41 passes every window forever. 0.6
  # clears that, and is still gentle enough that the speed is above STOP_SPEED at the cap, so the
  # ordinary should_stop() cannot be what latches and the cap is what is on trial.
  K = 0.6

  def test_every_window_on_its_own_says_the_stop_is_arriving(self):
    planner = build_planner()
    v_end = run_converging(planner, FRAMES_CAP - 1, v0=self.V0, k=self.K, force_decel=True)
    assert v_end > STOP_SPEED
    assert not planner.output_should_stop, "latched early, this is not testing the cap"
    assert planner.creep_total_t > CREEP_STOP_T, "never accumulated, the band was left"

  def test_the_cap_latches_a_converging_plan(self):
    planner = build_planner()
    v_end = run_converging(planner, FRAMES_CAP, v0=self.V0, k=self.K, force_decel=True)
    assert v_end > STOP_SPEED, "the run left the band, it is not testing the cap"
    assert planner.output_should_stop, "a converging plan crept on past CREEP_MAX_T"

  def test_the_cap_is_cleared_by_leaving_the_band(self):
    # the cap counts one approach, not the car's whole life: a stop that gives up and moves off
    # must not carry a spent budget into the next one
    planner = build_planner()
    run_converging(planner, FRAMES_CAP, v0=self.V0, k=self.K, force_decel=True)
    assert planner.creep_total_t > 0.0

    run(planner, int(1.0 / DT_MDL), v_ego=V_CREEP, force_decel=False)
    assert planner.creep_total_t == 0.0
    assert not planner.output_should_stop

  def test_the_cap_binds_before_the_window_test_would_have(self):
    # The point of reading it every frame. Hung off the window test the cap could only be acted on
    # at a window boundary, and over 33 logged stops the window test had already failed by the
    # first boundary past 1.5 s on every one of them -- so the cap never decided anything.
    planner = build_planner()
    v_end = run_converging(planner, FRAMES_CAP, v0=self.V0, k=self.K, force_decel=True)
    assert planner.output_should_stop
    assert v_end > STOP_SPEED
    assert planner.creep_total_t < 2 * WINDOW_T, "the cap did not bind until a window boundary"
    # and it is the cap, not the window: every window this run completed reset the dwell
    assert planner.creep_t < WINDOW_T
