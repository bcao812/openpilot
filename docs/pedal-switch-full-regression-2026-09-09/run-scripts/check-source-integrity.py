import hashlib
import json
from pathlib import Path
import subprocess

run = Path('/tmp/c3-pedal-full-20260909-WgogBo')
repo = Path('/home/bcao/SourceCode/openpilot-3xl-0817')
identities = json.loads((run / 'artifacts/source-identities.json').read_text())
for arm, files in identities.items():
  for relative, expected in files.items():
    actual = hashlib.sha256((run / arm / relative).read_bytes()).hexdigest()
    if actual != expected:
      raise ValueError((arm, relative))
for relative, expected in identities['current'].items():
  if hashlib.sha256((repo / relative).read_bytes()).hexdigest() != expected:
    raise ValueError(('vehicle checkout', relative))
recorded = json.loads((run / 'artifacts/recorded-source-identities.json').read_text())
for commit, files in recorded.items():
  for relative, expected in files.items():
    if hashlib.sha256((run / 'recorded' / commit[:9] / relative).read_bytes()).hexdigest() != expected:
      raise ValueError((commit, relative))
print('Current, baseline, 18 recording versions: all production source hashes unchanged')
checked = subprocess.run(['sha256sum', '-c', str(run / 'artifacts/device-binaries-before.txt')], cwd=repo, capture_output=True, text=True)
print(checked.stdout, end='')
checked.check_returncode()
(run / 'artifacts/source-integrity.txt').write_text('All pinned production source hashes unchanged\n' + checked.stdout)
