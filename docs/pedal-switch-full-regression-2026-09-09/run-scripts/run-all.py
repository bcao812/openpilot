from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time

run = Path('/tmp/c3-pedal-full-20260909-WgogBo')
manifest = json.loads((run / 'corpus/manifest.json').read_text())
routes = sorted(manifest['routes'], key=lambda route: (route['original_events'] != 0,
                                                       -sum(len(block) for block in route['blocks'])))
jobs = []
for route in routes:
  for arm in ('baseline', 'default', 'experimental', 'recorded'):
    output = run / ('recorded-results' if arm == 'recorded' else f'{arm}-results')
    if arm == 'recorded':
      output /= route['route']
    if (output / f'{route["route"]}.json').exists():
      continue
    source = run / ('baseline' if arm == 'baseline' else 'current')
    if arm == 'recorded':
      source = run / 'recorded' / route['commit'][:9]
    jobs.append((arm, route, output, source))


def execute(job):
  arm, route, output, source = job
  command = ['bash', str(run / 'run-python.sh'), str(source), '-m', 'openpilot.selfdrive.controls.tests.replay_pedal_full_routes',
             'run', '--corpus', str(run / 'corpus'), '--output', str(output), '--route', route['route']]
  if arm == 'experimental':
    command.append('--experimental-coast-with-lead')
  start = time.monotonic()
  with (run / 'artifacts' / f'all-{arm}-{route["route"]}.log').open('w') as stream:
    completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
  result = {'arm': arm, 'route': route['route'], 'exit_code': completed.returncode, 'elapsed_s': time.monotonic() - start}
  print(json.dumps(result), flush=True)
  return result


print('remaining jobs', len(jobs), flush=True)
with ThreadPoolExecutor(max_workers=10) as pool:
  results = [future.result() for future in as_completed([pool.submit(execute, job) for job in jobs])]
(run / 'artifacts/all-job-status.json').write_text(json.dumps(results, indent=2) + '\n')
if any(result['exit_code'] for result in results):
  raise SystemExit('At least one replay failed; inspect per-route logs')
