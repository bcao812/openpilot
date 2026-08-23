from typing import cast

from opendbc.car import structs
from openpilot.cereal import custom, messaging
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import CREEP_FINISH_T, CREEP_SPEED, CREEP_STOP_T, STOP_SPEED, should_stop
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController

# above STOP_SPEED, so should_stop() alone can never latch here, and inside the creep band
V_CREEP = 0.5
FRAMES_DWELL = int(CREEP_STOP_T / DT_MDL)


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


def build_sm(v_ego: float, force_decel: bool, enabled: bool = True, experimental_mode: bool = False):
  services = {}
  for service in ("radarState", "vehicleParameters", "carStateSP", "liveMapDataSP", "gpsLocationExternal", "gpsLocation"):
    services[service] = getattr(messaging.new_message(service), service)

  controls_state = messaging.new_message('controlsState')
  controls_state.controlsState.forceDecel = force_decel
  services['controlsState'] = controls_state.controlsState.as_reader()

  car_state = messaging.new_message('carState')
  car_state.carState.vEgo = v_ego
  car_state.carState.vCruise = 100.0
  car_state.carState.vCruiseCluster = 100.0
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
