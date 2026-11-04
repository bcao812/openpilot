"""Measure complete CarControlSP conversion from two fixed host snapshots.

Includes the candidate's serialization key, cache misses and independent output
copies. This is host conversion CPU, not complete card or on-device CPU4 load.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

from openpilot.cereal import custom
from openpilot.selfdrive.car.tests.test_cpu4_car_control_sp import frozen_values, populated_message

MODULE = 'openpilot/selfdrive/car/helpers.py'
BOUND_INPUTS = ('openpilot/cereal/custom.capnp', 'opendbc_repo/opendbc/car/structs.py')


def digest(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root, name):
  path = root / MODULE
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  manifest = root / '.host-build-source.json'
  source = json.loads(manifest.read_text())
  return module, {
    'source_commit': source['source_commit'], 'module_sha256': digest(path),
    'source_manifest_sha256': digest(manifest),
    'dependencies': {relative: digest(root / relative) for relative in BOUND_INPUTS},
  }


def converter(module, cached):
  return module.CarControlSPConverter() if cached else module.convert_carControlSP


def messages(kind):
  result = []
  for index in range(32):
    if kind == 'absent':
      message = custom.CarControlSP.new_message()
    elif kind == 'explicit-defaults':
      message = custom.CarControlSP.new_message(leadOne={}, leadTwo={}, mads={}, intelligentCruiseButtonManagement={})
    else:
      message = populated_message(index)
    result.append(message.as_reader())
  return result


def measure(module, cached, inputs, repeat_lead, frames):
  convert = converter(module, cached)
  for frame in range(100):
    convert(inputs[(frame // repeat_lead) % len(inputs)])
  cpu_start, wall_start = time.thread_time_ns(), time.perf_counter_ns()
  for frame in range(frames):
    convert(inputs[(frame // repeat_lead) % len(inputs)])
  return {'cpu_us_per_conversion': (time.thread_time_ns() - cpu_start) / frames / 1000,
          'wall_us_per_conversion': (time.perf_counter_ns() - wall_start) / frames / 1000}


def equivalent(modules, inputs, repeat_lead):
  convert = {arm: converter(module, arm == 'candidate') for arm, module in modules.items()}
  for frame in range(2 * len(inputs) * repeat_lead):
    message = inputs[(frame // repeat_lead) % len(inputs)]
    values = {arm: frozen_values(module.asdictref(convert[arm](message))) for arm, module in modules.items()}
    assert values['baseline'] == values['candidate'], frame


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--baseline', type=Path, required=True)
  parser.add_argument('--candidate', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--frames', type=int, default=5000)
  parser.add_argument('--repeats', type=int, default=8)
  args = parser.parse_args()
  if args.frames < 1 or args.repeats < 1:
    parser.error('frames and repeats must be positive')
  modules, identities = {}, {}
  for arm in ('baseline', 'candidate'):
    modules[arm], identities[arm] = load(getattr(args, arm), 'control_sp_benchmark_' + arm)
  assert identities['baseline']['dependencies'] == identities['candidate']['dependencies']
  results = []
  for kind, repeat_lead in (('absent', 1), ('explicit-defaults', 1), ('populated', 5), ('populated', 2), ('populated', 1)):
    inputs = messages(kind)
    equivalent(modules, inputs, repeat_lead)
    blocks = {arm: [] for arm in modules}
    for repeat in range(args.repeats):
      order = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
      for arm in order:
        blocks[arm].append(measure(modules[arm], arm == 'candidate', inputs, repeat_lead, args.frames))
    medians = {arm: {key: statistics.median(row[key] for row in samples) for key in samples[0]}
               for arm, samples in blocks.items()}
    results.append({
      'case': kind, 'conversions_per_lead_content': repeat_lead, 'equivalence': 'PASS',
      'medians': medians, 'blocks': blocks,
      'cpu_reduction_percent': 100 * (1 - medians['candidate']['cpu_us_per_conversion'] /
                                    medians['baseline']['cpu_us_per_conversion']),
    })
  output = {
    'method': 'interleaved-complete-car-control-sp-conversion-v1', 'identities': identities,
    'python': sys.version, 'frames_per_block': args.frames, 'repeats': args.repeats, 'results': results,
    'limitations': ['Synthetic complete wire fixtures; no route, C3 ARM, scheduling or complete card measurement.',
                    'Includes lead serialization keys and miss work; every-frame-changing input is a worst-case control.',
                    'No values, authority, timestamps or qualification flags are discarded.'],
  }
  args.output.write_text(json.dumps(output, indent=2) + '\n')
  print(json.dumps({**output, 'results': [{key: value for key, value in row.items() if key != 'blocks'}
                                        for row in results]}, indent=2))


if __name__ == '__main__':
  main()
