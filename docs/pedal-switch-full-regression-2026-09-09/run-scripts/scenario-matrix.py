from concurrent.futures import ProcessPoolExecutor
import itertools
import json
from pathlib import Path
import time

from openpilot.selfdrive.controls.tests.test_longitudinal_cruise_scenarios import simulate_lead_cruise

run = Path('/tmp/c3-pedal-full-20260909-WgogBo')
cases = [{'offset': offset, 'noise': noise, 'delay': delay, 'response_time': response, 'scenario': 'steady', 'speed_target': 20.0}
         for offset, noise, delay, response in itertools.product([0.0, 0.2, 0.25, 0.254545, 0.3], [0.0, 0.02],
                                                                 [0.3, 0.5, 0.8], [0.2, 0.4])]
cases += [{'offset': 0.25, 'noise': 0.02, 'delay': delay, 'response_time': response, 'scenario': scenario, 'speed_target': 20.0,
           'duration': 55.0 if scenario in ('lead_brake', 'cut_in') else 80.0}
          for scenario, delay, response in itertools.product(['flicker', 'override', 'no_lead', 'cut_in', 'lead_brake'],
                                                               [0.3, 0.5, 0.8], [0.2, 0.4])]
cases += [{'offset': 0.25, 'noise': 0.02, 'delay': 0.5, 'response_time': 0.4, 'scenario': 'steady', 'speed_target': speed}
          for speed in (9.9, 10.0, 10.1, 30.0)]


def execute(case):
  start = time.monotonic()
  before = simulate_lead_cruise(False, **case)
  after = simulate_lead_cruise(True, **case)
  result = {'case': case, 'default': before, 'experimental': after, 'elapsed_s': time.monotonic() - start}
  print(json.dumps({'case': case, 'switches': [before['switches']['debounced'], after['switches']['debounced']],
                    'gaps': [before['minimum_gap'], after['minimum_gap']]}), flush=True)
  return result


if __name__ == '__main__':
  with ProcessPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(execute, cases))
  (run / 'artifacts/scenario-matrix.json').write_text(json.dumps(results, indent=2) + '\n')
