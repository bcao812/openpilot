"""Compare complete Honda controller calls from two committed host snapshots.

Both arms use identical, source-bound dependencies and prebuilt wire inputs.
The differential includes ordered CAN bytes, actuator fields, controller state,
and non-finite-input exceptions. Timing excludes fixture construction and the
state comparison. It is host controller CPU, not card/CPU4 or C3 timing.
"""
import argparse
from dataclasses import fields, is_dataclass
from enum import Enum
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import struct
import sys
import time
from types import SimpleNamespace

import numpy as np

from opendbc.car import Bus, structs
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC
from opendbc.car.structs import car


MODULE = 'opendbc_repo/opendbc/car/honda/carcontroller.py'
CASES = (
  ('accord_active', CAR.HONDA_ACCORD, True, False, 'active'),
  ('accord_inactive', CAR.HONDA_ACCORD, True, False, 'inactive'),
  ('accord_stock', CAR.HONDA_ACCORD, False, False, 'mixed'),
  ('civic_bosch', CAR.HONDA_CIVIC_BOSCH, True, False, 'mixed'),
  ('civic_radarless', CAR.HONDA_CIVIC_2022, True, False, 'mixed'),
  ('nidec_civic', CAR.HONDA_CIVIC, True, False, 'mixed'),
  ('nidec_alt_pcm', CAR.HONDA_ODYSSEY, True, False, 'mixed'),
  ('nidec_interceptor', CAR.HONDA_CIVIC, True, True, 'mixed'),
  ('nidec_stock', CAR.HONDA_CIVIC, False, False, 'mixed'),
)
TIMED_CASES = {'accord_active', 'accord_inactive', 'accord_stock', 'nidec_civic'}


def digest(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root, name):
  path = root / MODULE
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  marker = root / '.host-build-source.json'
  manifest = json.loads(marker.read_text())
  return module, {'source_commit': manifest['source_commit'], 'module_sha256': digest(path),
                  'source_manifest_sha256': digest(marker)}


def freeze(value):
  """Bit-preserving observations, including signed zero and NaN payloads."""
  if isinstance(value, np.generic):
    value = value.item()
  if isinstance(value, Enum):
    return freeze(value.value)
  if isinstance(value, float):
    return ('float64', struct.pack('!d', value).hex())
  if isinstance(value, (bytes, bytearray)):
    return ('bytes', bytes(value).hex())
  if value is None or isinstance(value, (str, bool, int)):
    return value
  if isinstance(value, dict):
    return tuple((freeze(key), freeze(item)) for key, item in sorted(value.items(), key=lambda row: repr(row[0])))
  if isinstance(value, (list, tuple)):
    return tuple(freeze(item) for item in value)
  if is_dataclass(value):
    return tuple((field.name, freeze(getattr(value, field.name))) for field in fields(value))
  if hasattr(value, 'to_dict'):
    return freeze(value.to_dict())
  if hasattr(value, '__dict__'):
    return freeze(vars(value))
  raise TypeError(f'Unobserved state type: {type(value)}')


def controller_state(controller):
  # DBC metadata is bound as a dependency; counters are its only mutable
  # per-controller packing state. Every other instance attribute is compared.
  state = {name: value for name, value in vars(controller).items() if name != 'packer'}
  state['packer_counters'] = controller.packer.counters
  assert set(vars(controller.packer)) == {'dbc', 'counters'}
  return freeze(state)


def make_controller(module, case):
  _, platform, openpilot_long, interceptor, _ = case
  params = CarInterface.get_non_essential_params(platform)
  params.openpilotLongitudinalControl = openpilot_long
  params_sp = CarInterface.get_non_essential_params_sp(params, platform)
  params_sp.enableGasInterceptor = interceptor
  return module.CarController({Bus.pt: DBC[platform][Bus.pt]}, params, params_sp)


def make_input(index, mode):
  command = car.CarControl.new_message()
  command.enabled = mode != 'inactive' and (mode == 'active' or index % 17 != 0)
  command.longActive = command.enabled and (mode == 'active' or index % 11 != 0)
  command.latActive = command.enabled and index % 7 != 0
  command.actuators.accel = (-4.0, -2.5, -0.5, -0.2, -0.0, 0.0, 0.4, 1.5, 3.5)[index % 9]
  command.actuators.torque = (-0.9, -0.2, -0.0, 0.0, 0.25, 0.75)[index % 6]
  command.actuators.longControlState = ('off', 'pid', 'stopping', 'starting')[(index // 9) % 4]
  command.cruiseControl.cancel = index % 29 == 0
  command.cruiseControl.resume = index % 31 == 0
  command.hudControl.setSpeed = 25.0 + index % 10
  command.hudControl.speedVisible = index % 3 != 0
  command.hudControl.leadVisible = index % 2 == 0
  command.hudControl.lanesVisible = index % 3 == 0
  command.hudControl.leadDistanceBars = index % 5
  command.hudControl.visualAlert = ('none', 'fcw', 'steerRequired', 'ldw')[index % 4]
  state = car.CarState.new_message()
  state.vEgo = (-0.0, 0.0, 0.01, 2.299, 2.3, 10.0, 34.99, 35.0, 45.0)[(index // 9) % 9]
  state.standstill = state.vEgo < 0.01
  state.cruiseState.available = index % 7 != 0
  wrapper = SimpleNamespace(out=state.as_reader(), v_cruise_factor=1.0, is_metric=index % 2 == 0,
                            acc_hud={'FCM_OFF': index % 2, 'FCM_OFF_2': index % 2,
                                     'FCM_PROBLEM': 0, 'ICONS': index % 4},
                            lkas_hud={'LKAS_PROBLEM': 0}, stock_brake={'CHIME': index % 4})
  extension = structs.CarControlSP()
  extension.mads.available = index % 2 == 0
  extension.mads.enabled = index % 3 != 0
  extension.mads.active = command.latActive
  extension.intelligentCruiseButtonManagement.sendButton = (
    structs.IntelligentCruiseButtonManagement.SendButtonState.none,
    structs.IntelligentCruiseButtonManagement.SendButtonState.increase,
    structs.IntelligentCruiseButtonManagement.SendButtonState.decrease,
  )[(index // 13) % 3]
  return command.as_reader(), extension, wrapper


def call(controller, inputs, frame):
  try:
    actuators, messages = controller.update(*inputs, (frame + 1) * 10_000_000)
    result = ('output', freeze(actuators), freeze(messages))
  except Exception as error:
    result = ('exception', type(error).__name__, str(error))
  return result, controller_state(controller)


def equivalent(modules, case, frames):
  controllers = {arm: make_controller(module, case) for arm, module in modules.items()}
  pools = {arm: [make_input(index, case[-1]) for index in range(162)] for arm in modules}
  trace = hashlib.sha256()
  message_count = 0
  for frame in range(frames):
    results = {arm: call(controller, pools[arm][frame % 162], frame) for arm, controller in controllers.items()}
    assert results['baseline'] == results['candidate'], (case[0], frame, results)
    assert results['baseline'][0][0] == 'output', (case[0], frame, results)
    message_count += len(results['baseline'][0][2])
    trace.update(repr(results['baseline']).encode())
  return {'status': 'PASS', 'frames': frames, 'can_messages': message_count, 'trace_sha256': trace.hexdigest()}


def exceptional_inputs(modules):
  checks = 0
  trace = hashlib.sha256()
  for field in ('speed', 'accel', 'torque'):
    for value in (float('nan'), float('inf'), -float('inf'), -0.0):
      for active in (False, True):
        for parity in (0, 1):
          case = CASES[0] if active else CASES[1]
          controllers = {arm: make_controller(module, case) for arm, module in modules.items()}
          for frame in range(20 + parity):
            inputs = make_input(frame, case[-1])
            for controller in controllers.values():
              controller.update(*inputs, (frame + 1) * 10_000_000)
          command, extension, wrapper = make_input(0, case[-1])
          command = command.as_builder()
          if field == 'speed':
            state = wrapper.out.as_builder()
            state.vEgo = value
            wrapper.out = state.as_reader()
          else:
            setattr(command.actuators, field, value)
          inputs = command.as_reader(), extension, wrapper
          results = {arm: call(controller, inputs, 20 + parity) for arm, controller in controllers.items()}
          assert results['baseline'] == results['candidate'], (field, value, active, parity, results)
          trace.update(repr((field, value, active, parity, results['baseline'])).encode())
          checks += 1
  return {'status': 'PASS', 'cases': checks, 'trace_sha256': trace.hexdigest()}


def measure(module, case, frames):
  controller = make_controller(module, case)
  pool = [make_input(index, case[-1]) for index in range(162)]
  for frame in range(200):
    controller.update(*pool[frame % len(pool)], (frame + 1) * 10_000_000)
  started_cpu, started_wall = time.thread_time_ns(), time.perf_counter_ns()
  for frame in range(frames):
    controller.update(*pool[frame % len(pool)], (frame + 201) * 10_000_000)
  return {'thread_cpu_us_per_update': (time.thread_time_ns() - started_cpu) / frames / 1000,
          'wall_us_per_update': (time.perf_counter_ns() - started_wall) / frames / 1000}


def bound_dependencies(baseline, candidate):
  paths = {'opendbc_repo/opendbc/car/car.capnp', 'openpilot/cereal/custom.capnp'}
  for directory, suffixes in (('opendbc_repo/opendbc/dbc/generator/honda', {'.dbc', '.py'}),
                              ('opendbc_repo/opendbc/car', {'.capnp'}),
                              ('openpilot/cereal', {'.capnp'})):
    inputs = {arm: {str(path.relative_to(root)) for path in (root / directory).rglob('*')
                    if path.is_file() and path.suffix in suffixes}
              for arm, root in (('baseline', baseline), ('candidate', candidate))}
    assert inputs['baseline'] == inputs['candidate'], directory
    paths.update(inputs['candidate'])
  paths.add('opendbc_repo/opendbc/dbc/generator/generator.py')
  for name, module in tuple(sys.modules.items()):
    if name != 'opendbc' and not name.startswith(('opendbc.', 'openpilot.cereal')):
      continue
    filename = getattr(module, '__file__', None)
    if filename is None or not filename.endswith('.py'):
      continue
    path = Path(filename).resolve()
    if not path.is_relative_to(candidate):
      raise ValueError(f'Shared dependency is outside the candidate snapshot: {path}')
    relative = str(path.relative_to(candidate))
    if relative != MODULE:
      paths.add(relative)
  result = {}
  for relative in sorted(paths):
    assert digest(baseline / relative) == digest(candidate / relative), relative
    result[relative] = digest(candidate / relative)
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--baseline', type=Path, required=True)
  parser.add_argument('--candidate', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--frames', type=int, default=3000)
  parser.add_argument('--repeats', type=int, default=8)
  parser.add_argument('--equivalence-frames', type=int, default=1200)
  parser.add_argument('--differential-only', action='store_true')
  args = parser.parse_args()
  if min(args.frames, args.repeats, args.equivalence_frames) < 1:
    parser.error('frame counts and repeats must be positive')
  roots = {arm: getattr(args, arm).resolve() for arm in ('baseline', 'candidate')}
  modules, identities = {}, {}
  for arm, root in roots.items():
    modules[arm], identities[arm] = load(root, 'honda_cpu4_' + arm)
  dependencies = bound_dependencies(roots['baseline'], roots['candidate'])
  gc.disable()
  results = []
  for case in CASES:
    item = {'case': case[0], 'platform': str(case[1]), 'openpilot_longitudinal_control': case[2],
            'gas_interceptor': case[3], 'equivalence': equivalent(modules, case, args.equivalence_frames)}
    if not args.differential_only and case[0] in TIMED_CASES:
      rounds = []
      for repeat in range(args.repeats):
        order = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
        rounds.append({arm: measure(modules[arm], case, args.frames) for arm in order})
      medians = {arm: statistics.median(row[arm]['thread_cpu_us_per_update'] for row in rounds) for arm in modules}
      item.update(rounds=rounds, median_thread_cpu_us_per_update=medians,
                  cpu_reduction_percent=100 * (1 - medians['candidate'] / medians['baseline']))
    results.append(item)
    print(json.dumps({key: value for key, value in item.items() if key != 'rounds'}), flush=True)
  output = {'method': 'interleaved-complete-honda-controller-v1', 'identities': identities,
            'shared_dependencies': {**dependencies, **bound_dependencies(roots['baseline'], roots['candidate'])},
            'python': sys.version, 'numpy_version': np.__version__,
            'script_sha256': digest(Path(__file__)), 'gc_enabled': gc.isenabled(),
            'frames_per_timing_block': args.frames, 'repeats': args.repeats,
            'results': results, 'non_finite_and_signed_zero': exceptional_inputs(modules),
            'limitations': ['Synthetic complete controller inputs; no CAN input decoder or controlsd scheduling.',
                            'No C3 ARM, complete card/CPU4, on-device timing or driving validation.']}
  args.output.write_text(json.dumps(output, indent=2) + '\n')
  print(json.dumps({'non_finite_and_signed_zero': output['non_finite_and_signed_zero']}), flush=True)


if __name__ == '__main__':
  main()
