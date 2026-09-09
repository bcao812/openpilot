import hashlib
import json
from pathlib import Path
import shutil

run = Path('/tmp/c3-pedal-full-20260909-WgogBo')
repo = Path('/home/bcao/SourceCode/openpilot-3xl-0817')
output = repo / 'docs/pedal-switch-full-regression-2026-09-09'
accounting = json.loads((run / 'artifacts/final-accounting.json').read_text())
if accounting['routes'] != 55 or accounting['completed_route_arms'] != 220:
  raise ValueError('Refusing to persist incomplete all-route evidence')
output.mkdir(exist_ok=True)


def digest(path):
  with path.open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


for name in ['source-identities.json', 'recorded-source-identities.json', 'corpus-integrity.json', 'negative-cohort-check.json',
             'scenario-matrix.json', 'scenario-matrix-summary.json', 'streaming-equivalence.json', 'final-tool-equivalence.json',
             'preparer-equivalence.json', 'final-tool-source-hashes.json', 'preloaded-comparison.json', 'preloaded-sensitivity.json',
             'negative-route-added-raw.json', 'normal-pulse-classification.json', 'manual-transition-contexts.json',
             'raw-burst-context.json', 'default-gate-brake-contexts.json', 'source-integrity.txt', 'source-integrity-final.log',
             'device-binaries-before.txt',
             'device-binary-sweep.txt', 'host-environment.json', 'release-decision.json', 'final-accounting.json',
             'all-job-status.json', 'braking-contexts.json',
             'reconstruction-extremes.json', 'final-output-integrity.json', 'test-accounting.json',
             'full-tool-tests.log', 'full-tool-tests.xml', 'release-final.log', 'release-final.xml',
             'expanded-release-gates.log', 'expanded-release-gates.xml', 'honda-safety.log', 'honda-safety.xml', 'card-replay.log',
             'adjacent-longitudinal.log', 'adjacent-longitudinal.xml', 'adjacent-baseline.log', 'adjacent-baseline.xml']:
  source = run / 'artifacts' / name
  shutil.copy2(source, output / name)
shutil.copy2(run / 'corpus/manifest.json', output / 'corpus-manifest.json')
for source_name, target_name in [('evaluation-final', 'evaluation'), ('reconstruction', 'reconstruction'),
                                  ('normal-chain-evaluation', 'normal-chain'), ('braking-details', 'braking-details')]:
  source = run / source_name
  if source.exists():
    destination = output / target_name
    destination.mkdir(exist_ok=True)
    for path in source.glob('*.json'):
      shutil.copy2(path, destination / path.name)
routes = json.loads((run / 'corpus/manifest.json').read_text())['routes']
identities = {}
for route in routes:
  name = route['route']
  identities[name] = {}
  for arm in ('baseline', 'default', 'experimental', 'recorded'):
    source = run / f'{arm}-results'
    if arm == 'recorded':
      source /= name
    path = source / f'{name}.json'
    identities[name][arm] = json.loads(path.read_text())
(output / 'result-manifests.json').write_text(json.dumps(identities, indent=2) + '\n')
supplementary = {}
route = '00000013--f1538e1a32'
for arm in ('baseline', 'default', 'experimental', 'recorded'):
  supplementary[arm] = json.loads((run / 'preloaded' / arm / f'{route}.json').read_text())
(output / 'preloaded-result-manifests.json').write_text(json.dumps(supplementary, indent=2) + '\n')
scripts = output / 'run-scripts'
scripts.mkdir(exist_ok=True)
for name in ('run-python.sh', 'run-all.py', 'run-normal-chain.py', 'run-preloaded.py', 'scenario-matrix.py', 'summarize-final.py',
             'analyze-braking.py', 'analyze-normal-pulses.py', 'inspect-extremes.py', 'check-source-integrity.py',
             'verify-corpus.py', 'verify-output-integrity.py',
             'release-decision.py', 'persist-evidence.py'):
  shutil.copy2(run / name, scripts / name)
hashes = {str(path.relative_to(output)): digest(path) for path in output.rglob('*') if path.is_file() and path.name != 'artifact-hashes.json'}
(output / 'artifact-hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
print(f'Persisted {len(hashes)} evidence files to {output}')
