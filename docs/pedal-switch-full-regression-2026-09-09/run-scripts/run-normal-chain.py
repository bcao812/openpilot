from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

run = Path('/tmp/c3-pedal-full-20260909-WgogBo')
manifest = json.loads((run / 'corpus/manifest.json').read_text())
routes = [route for route in manifest['routes'] if route['original_events'] == 0 and route['native_enabled_s'].get('selfdriveState', 0) > 0]


def execute(route):
  name = route['route']
  command = ['bash', str(run / 'run-python.sh'), str(run / 'recorded' / route['commit'][:9]),
             '-m', 'openpilot.selfdrive.controls.tests.replay_pedal_full_routes', 'run',
             '--corpus', str(run / 'corpus'), '--output', str(run / 'normal-chain' / name), '--route', name, '--recorded-plan']
  with (run / 'artifacts' / f'normal-chain-{name}.log').open('w') as stream:
    completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
  print(name, completed.returncode, flush=True)
  if completed.returncode:
    raise ValueError(name)


with ThreadPoolExecutor(max_workers=1) as pool:
  list(pool.map(execute, routes))
