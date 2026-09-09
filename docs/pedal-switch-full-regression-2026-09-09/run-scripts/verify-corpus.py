from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

run = Path('/tmp/c3-pedal-full-20260909-WgogBo')
manifest = json.loads((run / 'corpus/manifest.json').read_text())
segments = [segment for route in manifest['routes'] for block in route['blocks'] for segment in block]


def verify(segment):
  for root, relative, expected in [(Path(manifest['logs']), segment['input']['path'], segment['input']['sha256']),
                                    (run / 'corpus', segment['cache'], segment['sha256'])]:
    with (root / relative).open('rb') as stream:
      actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != expected:
      raise ValueError(relative)
  return segment['input']['size']


with ThreadPoolExecutor(max_workers=2) as pool:
  total = sum(pool.map(verify, segments))
result = {'routes': len(manifest['routes']), 'segments': len(segments), 'original_bytes': total,
          'original_and_filtered_sha256_verified': True}
(run / 'artifacts/corpus-integrity.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
