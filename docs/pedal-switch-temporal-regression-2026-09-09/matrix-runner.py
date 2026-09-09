import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

from openpilot.selfdrive.controls.tests.test_longitudinal_cruise_scenarios import simulate_lead_cruise
from openpilot.selfdrive.controls.tests.replay_pedal_switches import save_json, sha256


def execute(case):
  start = time.monotonic()
  results = {mode: simulate_lead_cruise(False, smoothing=mode, **case) for mode in ('off', 'brake_dwell', 'median')}
  print(json.dumps({'case': case, 'switches': {mode: {key: value['switches'][key] for key in ('raw', 'debounced')}
                                            for mode, value in results.items()}}), flush=True)
  return {'case': case, 'results': results, 'elapsed_s': time.monotonic() - start}


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Replay the fixed temporal-smoothing closed-loop scenario matrix in an isolated host copy')
  parser.add_argument('--cases', type=Path, default=Path(__file__).with_name('scenario-matrix.json'))
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--workers', type=int, default=3)
  arguments = parser.parse_args()
  if arguments.output.exists():
    parser.error('Output already exists; use a new evidence file')
  source = json.loads(arguments.cases.read_text())
  cases = [item['case'] for item in source['results']]
  start = time.monotonic()
  with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
    results = list(pool.map(execute, cases))
  save_json(arguments.output, {'results': results, 'source_cases_sha256': sha256(arguments.cases), 'elapsed_s': time.monotonic() - start})
