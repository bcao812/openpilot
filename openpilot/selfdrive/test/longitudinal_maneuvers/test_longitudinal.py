import itertools
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.parameterized import parameterized_class

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE
from openpilot.selfdrive.test.longitudinal_maneuvers.maneuver import Maneuver


# TODO: make new FCW tests
def create_maneuvers(kwargs):
  maneuvers = [
    Maneuver(
      'approach stopped car at 25m/s, initial distance: 120m',
      duration=20.,
      initial_speed=25.,
      lead_relevancy=True,
      initial_distance_lead=120.,
      speed_lead_values=[30., 0.],
      breakpoints=[0., 1.],
      **kwargs,
    ),
    Maneuver(
      'approach stopped car at 20m/s, initial distance 90m',
      duration=20.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=90.,
      speed_lead_values=[20., 0.],
      breakpoints=[0., 1.],
      **kwargs,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 1m/s^2',
      duration=50.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=35.,
      speed_lead_values=[20., 20., 0.],
      breakpoints=[0., 15., 35.0],
      **kwargs,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 2m/s^2',
      duration=50.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=35.,
      speed_lead_values=[20., 20., 0.],
      breakpoints=[0., 15., 25.0],
      **kwargs,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 3m/s^2',
      duration=50.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=35.,
      speed_lead_values=[20., 20., 0.],
      breakpoints=[0., 15., 21.66],
      **kwargs,
    ),
    Maneuver(
      'steady state following a car at 20m/s, then lead decel to 0mph at 3+m/s^2',
      duration=40.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=35.,
      speed_lead_values=[20., 20., 0.],
      prob_lead_values=[0., 1., 1.],
      cruise_values=[20., 20., 20.],
      breakpoints=[2., 2.01, 8.8],
      **kwargs,
    ),
    Maneuver(
      "approach stopped car at 20m/s, with prob_lead_values",
      duration=30.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=120.,
      speed_lead_values=[0.0, 0., 0.],
      prob_lead_values=[0.0, 0., 1.],
      cruise_values=[20., 20., 20.],
      breakpoints=[0.0, 2., 2.01],
      **kwargs,
    ),
    Maneuver(
      "approach stopped car at 20m/s, with prob_throttle_values and pitch = -0.1",
      duration=30.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=120.,
      speed_lead_values=[0.0, 0., 0.],
      prob_throttle_values=[1., 0., 0.],
      cruise_values=[20., 20., 20.],
      pitch_values=[0., -0.1, -0.1],
      breakpoints=[0.0, 2., 2.01],
      **kwargs,
    ),
    Maneuver(
      "approach stopped car at 20m/s, with prob_throttle_values and pitch = +0.1",
      duration=30.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=120.,
      speed_lead_values=[0.0, 0., 0.],
      prob_throttle_values=[1., 0., 0.],
      cruise_values=[20., 20., 20.],
      pitch_values=[0., 0.1, 0.1],
      breakpoints=[0.0, 2., 2.01],
      **kwargs,
    ),
    Maneuver(
      "approach slower cut-in car at 20m/s",
      duration=20.,
      initial_speed=20.,
      lead_relevancy=True,
      initial_distance_lead=50.,
      speed_lead_values=[15., 15.],
      breakpoints=[1., 11.],
      only_lead2=True,
      **kwargs,
    ),
    Maneuver(
      "stay stopped behind radar override lead",
      duration=20.,
      initial_speed=0.,
      lead_relevancy=True,
      initial_distance_lead=10.,
      speed_lead_values=[0., 0.],
      prob_lead_values=[0., 0.],
      breakpoints=[1., 11.],
      only_radar=True,
      **kwargs,
    ),
    Maneuver(
      "NaN recovery",
      duration=30.,
      initial_speed=15.,
      lead_relevancy=True,
      initial_distance_lead=60.,
      speed_lead_values=[0., 0., 0.0],
      breakpoints=[1., 1.01, 11.],
      cruise_values=[float("nan"), 15., 15.],
      **kwargs,
    ),
    Maneuver(
      'cruising at 25 m/s while disabled',
      duration=20.,
      initial_speed=25.,
      lead_relevancy=False,
      enabled=False,
      **kwargs,
    ),
  ]
  if not kwargs['e2e']:
    maneuvers.append(Maneuver(
      "slow to 5m/s with allow_throttle = False and pitch = +0.1",
      duration=30.,
      initial_speed=20.,
      lead_relevancy=False,
      prob_throttle_values=[1., 0., 0.],
      cruise_values=[20., 20., 20.],
      pitch_values=[0., 0.1, 0.1],
      breakpoints=[0.0, 2., 2.01],
      ensure_slowdown=True,
      **kwargs,
    ))
  if not kwargs['force_decel']:
    # controls relies on planner commanding to move for stock-ACC resume spamming
    maneuvers.append(Maneuver(
      "resume from a stop",
      duration=20.,
      initial_speed=0.,
      lead_relevancy=True,
      initial_distance_lead=STOP_DISTANCE,
      speed_lead_values=[0., 0., 7.],
      breakpoints=[1., 10., 15.],
      ensure_start=True,
      **kwargs,
    ))
    if kwargs['e2e']:
      # An actuator that cannot brake gently, against a plan that converges asymptotically. Honda
      # Bosch turns any accel above BOSCH_GAS_LOOKUP_BP[0] into gas with BRAKE_REQUEST cleared, so
      # the plan's accel crosses that band while the car is still rolling, the speed floors out
      # above STOP_SPEED and should_stop() can no longer latch. Six logged stops behind a stopped
      # car on a 2019 Accord all ended this way, crawling 3.1 to 4.1 s to a standstill.
      #
      # A stopped lead does not reproduce it in this plant -- the MPC brakes hard enough to reach
      # STOP_SPEED on its own -- so the converging plan is what pins the anti-creep latch here.
      maneuvers.append(Maneuver(
        "stop with a converging plan and a -0.2 m/s^2 actuator dead band",
        duration=40.,
        initial_speed=20.,
        lead_relevancy=False,
        actuator_dead_band=-0.2,
        actuator_delay=0.5,
        model_decel_gain=0.3,
        ensure_stop=True,
        # No lead, so no gap to judge. ROLLING_STOP_ACCEL caps the ramp, and the ramp is the only
        # thing driving the command down here, so -1.55 holds that cap to account -- one ramp step
        # of slack below -1.5. The window test already latches this one inside CREEP_MAX_T, so the
        # crawl is the ramp's own length, and 1.8 is what that costs once the ramp descends at
        # STOPPING_DECEL_RATE (measured 1.90 -> 1.65 s).
        max_stopping_accel=-1.55,
        max_stopping_jerk=-1.5,
        max_crawl_t=1.8,
        **kwargs,
      ))
    # The same stop, but with a lead to stop behind, so the gap at standstill is a real number
    # rather than 200. The MPC's plan is steep enough here to reach STOP_SPEED on its own, so this
    # one is about where the car ends up, not about how long the creep lasts -- and it holds in acc
    # mode too, because the lead policy is not gated on experimental mode.
    maneuvers.append(Maneuver(
      "stop behind a stationary lead from 12 m/s",
      duration=30.,
      initial_speed=12.,
      lead_relevancy=True,
      initial_distance_lead=60.,
      speed_lead_values=[0., 0.],
      actuator_dead_band=-0.2,
      actuator_delay=0.5,
      ensure_stop=True,
      # History on this maneuver: accel -1.16, jerk -0.85, gap 5.36 m, crawl 3.10 s before the
      # anti-creep work; -1.19 / -0.83 / 5.53 / 2.60 with it; -1.52 / -1.29 / 5.81 / 1.85 once the
      # ramp walks down at STOPPING_DECEL_RATE and the cap is read every frame.
      #
      # The gap floor is the one number taken from the road rather than from a baseline -- 2.0 m is
      # where a stop stops feeling like a stop, and two logged stops were already under it, so it
      # stays where it is however much headroom the sim reports.
      #
      # accel is held to ROLLING_STOP_ACCEL plus one ramp step: the ramp now has time to reach that
      # cap, and reaching it is the point, but going past it while the wheels still turn is exactly
      # what ROLLING_STOP_ACCEL exists to prevent.
      #
      # jerk is the one that got worse (-0.83 -> -1.29) and it is left at -1.5. That is deliberate:
      # this plant's actuator is a clean first order lag, so its jerk column tracks the commanded
      # ramp almost directly, while on the road the commanded peak is 2 m/s^3 against 18 delivered.
      # The sim can say the command did not get more abrupt than the gate; it cannot say the car
      # did not. Only the road test can, and until it runs this number is a ceiling, not a result.
      max_stopping_accel=-1.55,
      max_stopping_jerk=-1.5,
      min_final_gap=2.0,
      max_crawl_t=2.0,
      **kwargs,
    ))
  return maneuvers


@parameterized_class(("e2e", "force_decel"), itertools.product([True, False], repeat=2))
class TestLongitudinalControl(OpenpilotTestCase):
  e2e: bool
  force_decel: bool

  def test_maneuver(self, subtests):
    for maneuver in create_maneuvers({"e2e": self.e2e, "force_decel": self.force_decel}):
      with subtests.test(title=maneuver.title, e2e=maneuver.e2e, force_decel=maneuver.force_decel):
        print(maneuver.title, f'in {"e2e" if maneuver.e2e else "acc"} mode')
        valid, _ = maneuver.evaluate()
        assert valid
