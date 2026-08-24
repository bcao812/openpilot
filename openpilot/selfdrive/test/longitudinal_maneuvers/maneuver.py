import numpy as np
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import CREEP_SPEED
from openpilot.selfdrive.test.longitudinal_maneuvers.plant import Plant

# Same recipe the log analysis uses, so a number here is comparable to a number from a drive.
STILL = 0.05      # m/s, wheels stopped
SMOOTH_N = 5      # frames, 0.25 s centred mean before differentiating -- raw acceleration is far
                  # too noisy at 20 Hz for its derivative to mean anything


def _smooth_jerk(accel):
  """Jerk of the 0.25 s centred mean of acceleration, in m/s^3."""
  h = SMOOTH_N // 2
  sm = np.array([accel[max(0, k - h):k + h + 1].mean() for k in range(len(accel))])
  return np.diff(sm) / DT_MDL


class Maneuver:
  def __init__(self, title, duration, **kwargs):
    # Was tempted to make a builder class
    self.distance_lead = kwargs.get("initial_distance_lead", 200.0)
    self.speed = kwargs.get("initial_speed", 0.0)
    self.lead_relevancy = kwargs.get("lead_relevancy", 0)

    self.breakpoints = kwargs.get("breakpoints", [0.0, duration])
    self.speed_lead_values = kwargs.get("speed_lead_values", [0.0 for i in range(len(self.breakpoints))])
    self.prob_lead_values = kwargs.get("prob_lead_values", [1.0 for i in range(len(self.breakpoints))])
    self.prob_throttle_values = kwargs.get("prob_throttle_values", [1.0 for i in range(len(self.breakpoints))])
    self.cruise_values = kwargs.get("cruise_values", [50.0 for i in range(len(self.breakpoints))])
    self.pitch_values = kwargs.get("pitch_values", [0.0 for i in range(len(self.breakpoints))])

    self.only_lead2 = kwargs.get("only_lead2", False)
    self.only_radar = kwargs.get("only_radar", False)
    self.ensure_start = kwargs.get("ensure_start", False)
    self.ensure_slowdown = kwargs.get("ensure_slowdown", False)
    self.ensure_stop = kwargs.get("ensure_stop", False)
    self.enabled = kwargs.get("enabled", True)
    self.e2e = kwargs.get("e2e", False)
    self.personality = kwargs.get("personality", 0)
    self.force_decel = kwargs.get("force_decel", False)
    self.actuator_dead_band = kwargs.get("actuator_dead_band", 0.0)
    self.model_decel_gain = kwargs.get("model_decel_gain", 0.0)
    self.actuator_delay = kwargs.get("actuator_delay", 0.0)

    # How the last second of a stop is allowed to feel. All four are measured only once the car has
    # actually stopped, so a maneuver that never stops is judged by ensure_stop and nothing else.
    self.max_stopping_accel = kwargs.get("max_stopping_accel", None)  # deepest command while rolling
    self.max_stopping_jerk = kwargs.get("max_stopping_jerk", None)    # m/s^3, most abrupt change
    self.min_final_gap = kwargs.get("min_final_gap", None)            # m, gap at standstill
    self.max_crawl_t = kwargs.get("max_crawl_t", None)                # s, CREEP_SPEED to standstill

    self.duration = duration
    self.title = title

  def _check_stop_quality(self, logs):
    """Judge the last second of the stop. Returns (valid, measured) -- measured is printed either
    way, because these numbers are the point of the maneuver even when they pass."""
    gates = (self.max_stopping_accel, self.max_stopping_jerk, self.min_final_gap, self.max_crawl_t)
    if all(g is None for g in gates):
      return True, {}

    speed = np.array([lg['speed'] for lg in logs])
    still = np.flatnonzero(speed <= STILL)
    if not len(still):
      print('never stopped, cannot judge the stop')
      return False, {}
    i_stop = int(still[0])

    # The stop begins where the car drops under walking pace. Everything before that is the
    # approach, which these gates deliberately say nothing about.
    moving = np.flatnonzero(speed[:i_stop] >= CREEP_SPEED)
    i_crawl = int(moving[-1]) + 1 if len(moving) else 0

    accel = np.array([lg['acceleration'] for lg in logs])
    m = {
      'stopping_accel': min(lg['accel_cmd'] for lg in logs[i_crawl:i_stop + 1]),
      'stopping_jerk': _smooth_jerk(accel[max(0, i_crawl - SMOOTH_N):i_stop + 1]).min(),
      'final_gap': logs[i_stop]['d_rel'],
      'crawl_t': (i_stop - i_crawl) * DT_MDL,
    }
    print('stop quality: ' + '  '.join(f'{k} {v:.2f}' for k, v in m.items()))

    # The first two are floors on a quantity that is negative and gets worse downward; the last two
    # are ceilings on a quantity that gets worse upward.
    valid = True
    for name, limit, floor in (('stopping_accel', self.max_stopping_accel, True),
                               ('stopping_jerk', self.max_stopping_jerk, True),
                               ('final_gap', self.min_final_gap, True),
                               ('crawl_t', self.max_crawl_t, False)):
      if limit is None:
        continue
      if m[name] < limit if floor else m[name] > limit:
        print(f'{name} {m[name]:.2f} is worse than {limit:.2f}')
        valid = False
    return valid, m

  def evaluate(self):
    plant = Plant(
      lead_relevancy=self.lead_relevancy,
      speed=self.speed,
      distance_lead=self.distance_lead,
      enabled=self.enabled,
      only_lead2=self.only_lead2,
      only_radar=self.only_radar,
      e2e=self.e2e,
      personality=self.personality,
      force_decel=self.force_decel,
      actuator_dead_band=self.actuator_dead_band,
      model_decel_gain=self.model_decel_gain,
      actuator_delay=self.actuator_delay,
    )

    valid = True
    logs = []
    frames = []
    not_starting_t = 0.0
    while plant.current_time < self.duration:
      speed_lead = np.interp(plant.current_time, self.breakpoints, self.speed_lead_values)
      prob_lead = np.interp(plant.current_time, self.breakpoints, self.prob_lead_values)
      cruise = np.interp(plant.current_time, self.breakpoints, self.cruise_values)
      pitch = np.interp(plant.current_time, self.breakpoints, self.pitch_values)
      prob_throttle = np.interp(plant.current_time, self.breakpoints, self.prob_throttle_values)
      log = plant.step(speed_lead, prob_lead, cruise, pitch, prob_throttle)

      d_rel = log['distance_lead'] - log['distance'] if self.lead_relevancy else 200.
      v_rel = speed_lead - log['speed'] if self.lead_relevancy else 0.
      log['d_rel'] = d_rel
      log['v_rel'] = v_rel
      frames.append(log)
      logs.append(np.array([plant.current_time,
                            log['distance'],
                            log['distance_lead'],
                            log['speed'],
                            speed_lead,
                            log['acceleration'],
                            log['d_rel']]))

      if d_rel < .4 and (self.only_radar or prob_lead > 0.5):
        print("Crashed!!!!")
        valid = False

      if self.ensure_start and log['v_rel'] > 0 and log['acceleration'] < 1e-3:
        if not_starting_t == 0.0:
          not_starting_t = plant.current_time
        elif plant.current_time - not_starting_t > 0.5:
          print('LongitudinalPlanner not starting!')
          valid = False
      else:
        not_starting_t = 0.0

    if self.ensure_slowdown and log['speed'] > 5.5:
      print('LongitudinalPlanner not slowing down!')
      valid = False

    if self.ensure_stop and (log['speed'] > 1e-2 or not log['should_stop']):
      print('LongitudinalPlanner not stopping!')
      valid = False

    # force_decel is a planner input, so judge the planner's output. The actuator is a separate
    # question: LongControl commands nothing at all while openpilot is not longitudinally active.
    if self.force_decel and log['speed'] > 1e-1 and log['a_target'] > -0.04:
      print('Not stopping with force decel')
      valid = False

    valid &= self._check_stop_quality(frames)[0]

    print("maneuver end", valid)
    return valid, np.array(logs)
