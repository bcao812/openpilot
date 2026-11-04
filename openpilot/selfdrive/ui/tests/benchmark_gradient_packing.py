"""Compare production gradient setup in two explicit, isolated host snapshots.

GPU uploads are no-ops during timing. This measures Python/CFFI preparation,
including gradient replacement at the selected cadence; it is not device FPS.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import pyray as rl

MODULE = 'openpilot/system/ui/lib/shader_polygon.py'


def load(root, name):
  path = root / MODULE
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  marker = root / '.host-build-source.json'
  source = json.loads(marker.read_text())
  state = module.ShaderState.get_instance()
  state.locations = {key: key for key in state.locations}
  return module, state, {'source_commit': source['source_commit'], 'module_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'source_manifest_sha256': hashlib.sha256(marker.read_bytes()).hexdigest()}


def source_gradient(module, count, variant=0):
  colors = [rl.Color((i * 37 + variant) % 256, (i * 19 + variant) % 256,
                     255 - i * 11 % 256, (i * 47 + variant) % 256) for i in range(count)]
  stops = [-0.1 + 1.2 * i / max(1, count - 1) for i in range(count)]
  return module.Gradient((0.2, 0.1), (0.7, 0.9), colors, stops)


@contextmanager
def capture_uploads(module):
  uploads = []

  def scalar(shader, location, value, kind):
    count = 2 if kind == module.UNIFORM_VEC2 else 4 if kind == module.UNIFORM_VEC4 else 1
    pointer = rl.ffi.addressof(value) if kind == module.UNIFORM_VEC2 else value
    uploads.append((location, kind, count, bytes(rl.ffi.buffer(pointer, count * 4))))

  def array(shader, location, value, kind, count):
    size = count * (4 if kind == module.UNIFORM_VEC4 else 1)
    uploads.append((location, kind, count, bytes(rl.ffi.buffer(value, size * 4))))

  with patch.object(rl, 'set_shader_value', scalar), patch.object(rl, 'set_shader_value_v', array):
    yield uploads


def equivalent(modules, states, count):
  packets = {}
  for arm, module in modules.items():
    gradient = source_gradient(module, count)
    if arm == 'candidate':
      gradient = module.prepare_gradient(gradient)
    with capture_uploads(module) as uploads:
      for rect in (rl.Rectangle(0, 0, 2160, 1080), rl.Rectangle(-11, 23, 400, 600)):
        module._configure_shader_color(states[arm], None, gradient, rect)
        module._configure_shader_color(states[arm], rl.Color(255, 0, 0, 81), None, rect)
        module._configure_shader_color(states[arm], None, source_gradient(module, 2, 13), rect)
        module._configure_shader_color(states[arm], None, gradient, rect)
      packets[arm] = uploads
  assert packets['baseline'] == packets['candidate'], f'Uniform bytes differ for {count} colors'


def measure(module, state, prepared, count, update_every, frames):
  # Geometry-dependent colors are already computed by ModelRenderer. Preserve the
  # same construction cadence while measuring the added packing work separately.
  sources = [source_gradient(module, count, variant) for variant in (0, 1, 5, 17)]

  def build(index):
    source = sources[index % len(sources)]
    gradient = module.Gradient(source.start, source.end, source.colors, source.stops)
    return module.prepare_gradient(gradient) if prepared else gradient

  gradient = build(0)
  rects = (rl.Rectangle(0, 0, 2160, 1080), rl.Rectangle(7, 15, 2153, 1065))
  configure = module._configure_shader_color
  for _ in range(50):
    configure(state, None, gradient, rects[0])
  cpu_start, wall_start = time.thread_time_ns(), time.perf_counter_ns()
  for frame in range(frames):
    if update_every and frame % update_every == 0:
      gradient = build(frame // update_every)
    configure(state, None, gradient, rects[frame % 2])
  return {'cpu_us_per_draw': (time.thread_time_ns() - cpu_start) / frames / 1000,
          'wall_us_per_draw': (time.perf_counter_ns() - wall_start) / frames / 1000}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--baseline', type=Path, required=True)
  parser.add_argument('--candidate', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--frames', type=int, default=2000)
  parser.add_argument('--repeats', type=int, default=8)
  args = parser.parse_args()
  if args.frames < 1 or args.repeats < 1:
    parser.error('frames and repeats must be positive')
  modules, states, identities = {}, {}, {}
  for arm in ('baseline', 'candidate'):
    modules[arm], states[arm], identities[arm] = load(getattr(args, arm), 'gradient_benchmark_' + arm)
  results = []
  for count in (2, 8, 16):
    equivalent(modules, states, count)
    for update_every in (0, 3, 2, 1):
      costs = {arm: [] for arm in modules}
      with patch.object(rl, 'set_shader_value', lambda *args: None), patch.object(rl, 'set_shader_value_v', lambda *args: None):
        for repeat in range(args.repeats):
          order = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
          for arm in order:
            costs[arm].append(measure(modules[arm], states[arm], arm == 'candidate', count, update_every, args.frames))
      medians = {arm: {key: statistics.median(row[key] for row in rows) for key in rows[0]} for arm, rows in costs.items()}
      results.append({'colors': count, 'replace_every_draws': update_every or None, 'equivalence': 'PASS',
                      'medians': medians, 'blocks': costs,
                      'cpu_reduction_percent': 100 * (1 - medians['candidate']['cpu_us_per_draw'] / medians['baseline']['cpu_us_per_draw'])})
  output = {'method': 'interleaved-production-gradient-uniforms-v1', 'identities': identities,
            'python': sys.version, 'frames_per_block': args.frames, 'repeats': args.repeats, 'results': results,
            'limitations': ['GPU calls are no-ops; no rendering, camera, display pacing or device FPS measurement.',
                           'Measures gradient setup only. Model color derivation and polygon geometry are unchanged and excluded.',
                           'Replacement cases include construction and packing; null replacement cadence measures an already-held snapshot.']}
  args.output.write_text(json.dumps(output, indent=2) + '\n')
  print(json.dumps({'identities': identities, 'results': [{key: value for key, value in row.items() if key != 'blocks'}
                                                        for row in results]}, indent=2))


if __name__ == '__main__':
  main()
