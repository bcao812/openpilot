from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

run = Path('/tmp/c3-pedal-full-20260909-WgogBo')


def execute(arm):
  source = run / ('baseline' if arm == 'baseline' else 'recorded/c657fa948' if arm == 'recorded' else 'current')
  command = ['bash', str(run / 'run-python.sh'), str(source), '-m', 'openpilot.selfdrive.controls.tests.replay_pedal_full_routes',
             'run', '--corpus', str(run / 'corpus'), '--output', str(run / 'preloaded' / arm), '--route', '00000013', '--preload-parameters']
  if arm == 'experimental':
    command.append('--experimental-coast-with-lead')
  with (run / 'artifacts' / f'preloaded-{arm}.log').open('w') as stream:
    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
  print(arm, result.returncode, flush=True)
  if result.returncode:
    raise ValueError(arm)


with ThreadPoolExecutor(max_workers=2) as pool:
  list(pool.map(execute, ['baseline', 'default', 'experimental', 'recorded']))
