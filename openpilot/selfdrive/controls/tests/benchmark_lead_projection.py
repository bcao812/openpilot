"""Compare complete ControlsExt.run_ext with a bounded serializing publisher.

Both arms map current radar leads, read current MADS/ICBM, build the published
event and serialize it. Includes cache keys, misses and independent output copies;
excludes sockets, scheduling, other controlsd phases and C3/ARM execution.
"""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

from opendbc.car.structs import car
from openpilot.cereal import custom, messaging
from openpilot.selfdrive.car.tests.test_cpu4_car_control_sp import frozen_values
from openpilot.selfdrive.controls.tests.test_controlsd_ext import (
  BUTTON_STATES, ICBM_STATES, MADS_STATES, control_inputs,
)

MODULE = 'openpilot/sunnypilot/selfdrive/controls/controlsd_ext.py'
BOUND_INPUTS = ('openpilot/cereal/custom.capnp', 'openpilot/cereal/log.capnp', 'opendbc_repo/opendbc/car/car.capnp',
                'openpilot/common/radar_signal.py')


def digest(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def function_source(root, relative, name):
  source = (root / relative).read_text()
  node = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name)
  return ast.get_source_segment(source, node)


def load(root, name):
  path = root / MODULE
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  manifest = root / '.host-build-source.json'
  source = json.loads(manifest.read_text())
  return module, {'source_commit': source['source_commit'], 'module_sha256': digest(path),
                  'source_manifest_sha256': digest(manifest),
                  'dependencies': {relative: digest(root / relative) for relative in BOUND_INPUTS}}


class Publisher:
  def __init__(self):
    self.payload = None

  def send(self, service, message):
    assert service == 'carControlSP'
    self.payload = message.to_bytes()


def inputs(kind):
  radars, states = [], []
  for index in range(32):
    radar, _, _ = control_inputs(('leadOne', 'leadTwo') if kind == 'populated' else (), phase=index)
    if kind == 'explicit-defaults':
      radar.leadOne = {'evidence': {}}
      radar.leadTwo = {'evidence': {}}
    radars.append(radar.as_reader())
    state = custom.SelfdriveStateSP.new_message(
      mads={'state': MADS_STATES[index % len(MADS_STATES)],
            'enabled': bool(index & 1), 'active': bool(index & 2), 'available': bool(index & 4)},
      intelligentCruiseButtonManagement={'state': ICBM_STATES[index % len(ICBM_STATES)],
                                        'sendButton': BUTTON_STATES[index % len(BUTTON_STATES)], 'vTarget': index / 8.0},
    )
    states.append(state.as_reader())
  return radars, states


def setup(module):
  control = module.ControlsExt.__new__(module.ControlsExt)
  control.radar_signal = True
  state = {'carState': car.CarState.new_message(canValid=True).as_reader()}
  return control, state, Publisher()


def run_frame(control, state, publisher, radars, states, cadence, frame):
  state['radarState'] = radars[(frame // cadence) % len(radars)]
  state['selfdriveStateSP'] = states[frame % len(states)]
  control.run_ext(state, publisher)


def equivalent(modules, radars, states, cadence):
  arms = {name: setup(module) for name, module in modules.items()}
  trace = hashlib.sha256()
  frames = 2 * len(radars) * cadence
  for frame in range(frames):
    values = {}
    for arm, (control, state, publisher) in arms.items():
      run_frame(control, state, publisher, radars, states, cadence, frame)
      event = messaging.log_from_bytes(publisher.payload)
      value = event.to_dict()
      value.pop('logMonoTime')  # Both arms use the real, different publication clock.
      values[arm] = frozen_values(value)
    assert values['baseline'] == values['candidate'], frame
    trace.update(repr(values['baseline']).encode())
  return {'status': 'PASS', 'frames': frames, 'trace_sha256': trace.hexdigest(),
          'comparison': 'all published fields and float bits except the real publication clock'}


def measure(module, radars, states, cadence, frames):
  control, state, publisher = setup(module)
  for frame in range(100):
    run_frame(control, state, publisher, radars, states, cadence, frame)
  cpu_start, wall_start = time.thread_time_ns(), time.perf_counter_ns()
  for frame in range(frames):
    run_frame(control, state, publisher, radars, states, cadence, frame)
  return {'cpu_us_per_frame': (time.thread_time_ns() - cpu_start) / frames / 1000,
          'wall_us_per_frame': (time.perf_counter_ns() - wall_start) / frames / 1000}


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
    modules[arm], identities[arm] = load(getattr(args, arm), 'lead_projection_' + arm)
  assert identities['baseline']['dependencies'] == identities['candidate']['dependencies']
  # Other concurrent optimizations may alter FrequencyTracker in this module;
  # this benchmark executes only new_message/log_from_bytes from messaging.
  messaging_path = 'openpilot/cereal/messaging/__init__.py'
  shared_messaging = {}
  for name in ('new_message', 'log_from_bytes'):
    baseline = function_source(args.baseline, messaging_path, name)
    candidate = function_source(args.candidate, messaging_path, name)
    assert baseline == candidate, f'Changed shared benchmark dependency: {name}'
    shared_messaging[name] = hashlib.sha256(candidate.encode()).hexdigest()
  gc.disable()
  results = []
  for kind, cadence in (('absent', 1), ('explicit-defaults', 1), ('populated', 5), ('populated', 2), ('populated', 1)):
    radars, states = inputs(kind)
    comparison = equivalent(modules, radars, states, cadence)
    blocks = []
    for repeat in range(args.repeats):
      order = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
      blocks.append({arm: measure(modules[arm], radars, states, cadence, args.frames) for arm in order})
    medians = {arm: statistics.median(block[arm]['cpu_us_per_frame'] for block in blocks) for arm in modules}
    results.append({'case': kind, 'frames_per_lead_content': cadence, 'equivalence': comparison,
                    'cpu_us_per_frame_median': medians, 'blocks': blocks,
                    'cpu_reduction_percent': 100 * (1 - medians['candidate'] / medians['baseline'])})
  output = {'method': 'interleaved-complete-controls-extension-v1', 'identities': identities,
            'shared_messaging_function_sha256': shared_messaging, 'gc_enabled': gc.isenabled(),
            'python': sys.version, 'frames_per_block': args.frames, 'repeats': args.repeats,
            'results': results, 'limitations': __doc__}
  args.output.write_text(json.dumps(output, indent=2) + '\n')
  print(json.dumps({**output, 'results': [{key: value for key, value in row.items() if key != 'blocks'}
                                       for row in results]}, indent=2))


if __name__ == '__main__':
  main()
