import hashlib
import json
from pathlib import Path

import numpy as np


run = Path('/tmp/c3-pedal-full-20260909-WgogBo')
manifest_path = run / 'corpus/manifest.json'
manifest = json.loads(manifest_path.read_text())
manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
artifacts = []
for record in manifest['routes']:
  route = record['route']
  for arm in ('baseline', 'default', 'experimental', 'recorded'):
    root = run / f'{arm}-results'
    if arm == 'recorded':
      root /= route
    artifacts.append((root, route, arm))
for arm in ('baseline', 'default', 'experimental', 'recorded'):
  artifacts.append((run / 'preloaded' / arm, '00000013--f1538e1a32', f'preloaded/{arm}'))
for path in (run / 'normal-chain').glob('*/000*.json'):
  artifacts.append((path.parent, path.stem, 'normal-chain'))
checked, total_bytes = 0, 0
manifests = {}
for root, route, arm in artifacts:
  path = root / f'{route}.json'
  metadata = json.loads(path.read_text())
  assert metadata['manifest_sha256'] == manifest_hash
  for chunk in metadata['chunks']:
    source = root / chunk['file']
    with source.open('rb') as stream:
      actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert actual == chunk['sha256'], source
    with np.load(source, allow_pickle=False) as saved:
      for kind, rows in chunk['rows'].items():
        assert saved[kind].shape == (rows, len(metadata['fields'][kind])), (source, kind)
    checked += 1
    total_bytes += source.stat().st_size
  manifests[f'{arm}/{route}'] = hashlib.sha256(path.read_bytes()).hexdigest()
assert len(artifacts) == 231, len(artifacts)
result = {'corpus_manifest_sha256': manifest_hash, 'route_arm_outputs': len(artifacts), 'verified_chunks': checked,
          'verified_bytes': total_bytes, 'manifests': manifests, 'all_sha256_and_shapes_match': True}
(run / 'artifacts/final-output-integrity.json').write_text(json.dumps(result, indent=2) + '\n')
print(f'Verified {len(artifacts)} route-arm results, {checked} chunks, {total_bytes} bytes')
