"""Compare committed CAN parsers on the same bounded Accord CAN workload.

Subscriptions are obtained from the real Honda CarState startup path. The CAN
payloads and cadences are synthetic, so the results measure parser CPU rather
than a recorded route, complete card, or ARM CPU4 load. No CAN is transmitted.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

from opendbc import get_generated_dbcs
from opendbc.can import CANPacker
from opendbc.car import structs
from opendbc.car.honda.carstate import CarState
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, HondaFlags
from opendbc.can.tests.test_raw_value_fast import signal_at

MODULE = 'opendbc_repo/opendbc/can/parser.py'
BOUND_INPUTS = (
  'opendbc_repo/opendbc/can/dbc.py', 'opendbc_repo/opendbc/can/packer.py',
  'opendbc_repo/opendbc/car/honda/carstate.py', 'opendbc_repo/opendbc/car/honda/interface.py',
  'opendbc_repo/opendbc/car/honda/values.py', 'opendbc_repo/opendbc/car/honda/hondacan.py',
  'opendbc_repo/opendbc/car/interfaces.py', 'opendbc_repo/opendbc/car/structs.py',
  'opendbc_repo/opendbc/sunnypilot/car/honda/carstate_ext.py',
)


def digest(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root, name):
  path = root / MODULE
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  manifest = root / '.host-build-source.json'
  source = json.loads(manifest.read_text())
  return module, {
    'source_commit': source['source_commit'], 'module_sha256': digest(path),
    'source_manifest_sha256': digest(manifest),
    'dependencies': {relative: digest(root / relative) for relative in BOUND_INPUTS},
  }


def subscriptions():
  params = CarInterface.get_non_essential_params(CAR.HONDA_ACCORD)
  params.openpilotLongitudinalControl = True
  params.pcmCruise = False
  params.flags |= HondaFlags.HYBRID
  params.transmissionType = structs.CarParams.TransmissionType.automatic
  extension = CarInterface.get_non_essential_params_sp(params, CAR.HONDA_ACCORD)
  state = CarState(params, extension)
  parsers = state.get_can_parsers(params, extension)
  state.update(parsers)  # real VLDict reads discover the runtime subscriptions
  result = []
  for bus_name, parser in parsers.items():
    result.append({'bus_name': str(bus_name), 'bus': parser.bus, 'dbc': parser.dbc_name,
                   'messages': [st.name for st in parser.message_states.values()]})
  return result


def inputs(layout, count, *, mixed_cadence, faults=False):
  packers = [CANPacker(item['dbc']) for item in layout]
  packets = []
  for frame in range(count):
    messages = []
    for parser_index, item in enumerate(layout):
      packer = packers[parser_index]
      for message_index, name in enumerate(item['messages']):
        divisor = (1, 2, 5, 10)[message_index % 4] if mixed_cadence else 1
        if frame % divisor:
          continue
        msg = packer.dbc.name_to_msg[name]
        values = {}
        for index, signal in enumerate(msg.sigs.values()):
          if signal.type != 0 or signal.name == 'COUNTER':
            continue
          raw = (17 * frame + 3 * index) & ((1 << min(signal.size, 12)) - 1)
          if signal.is_signed and raw & (1 << (signal.size - 1)):
            raw -= 1 << signal.size
          values[signal.name] = raw * signal.factor + signal.offset
        address, data, bus = packer.make_can_msg(name, item['bus'], values)
        if faults and frame % 41 == 17 and message_index == 0:
          data = data[:-1]  # short frame, original parser contract is preserved
        elif faults and frame % 37 == 23 and message_index == 1:
          data = bytes([data[0] ^ 0x80]) + data[1:]
        messages.append((address, data, bus))
    if faults and frame % 43 == 5:
      messages.extend(messages[:1])  # same tick duplicate/counter failure
    if faults and 150 <= frame < 210:
      messages = []  # alive/bus timeout and recovery use the original clocks
    messages.append((0x7FFFFFF, b'\x11', 9))  # ignored bus/unknown address
    packets.append([(frame + 1) * 10_000_000, messages])
  return packets


def make_parsers(module, layout):
  return [module.CANParser(item['dbc'], [(name, 0) for name in item['messages']], item['bus'])
          for item in layout]


def signature(parser, updated):
  # can_valid changes its own debounce count. Call once at the same point in
  # both arms, just as the timed receive/health boundary does.
  valid, timeout = parser.can_valid, parser.bus_timeout
  state_names = ('counter', 'counter_fail', 'first_seen_nanos', 'last_warning_log_nanos',
                 'frequency', 'timeout_threshold', 'vals')
  return {
    'updated': sorted(updated), 'valid': valid, 'timeout': timeout,
    'can_invalid_cnt': parser.can_invalid_cnt,
    'last_nonempty_nanos': parser.last_nonempty_nanos, 'last_update_nanos': parser._last_update_nanos,
    'messages': {address: {
      'vl': parser.vl[address].copy(),
      'vl_all': {name: values.copy() for name, values in parser.vl_all[address].items()},
      'ts_nanos': parser.ts_nanos[address].copy(),
      'state': {name: getattr(state, name).copy() if name == 'vals' else getattr(state, name) for name in state_names},
      'all_vals': [values.copy() for values in state.all_vals], 'timestamps': list(state.timestamps),
      'aliases': (parser.vl[address] is parser.vl[state.name],
                  parser.vl_all[address] is parser.vl_all[state.name],
                  parser.ts_nanos[address] is parser.ts_nanos[state.name]),
    } for address, state in parser.message_states.items()},
  }


def equivalent(modules, layout, packets):
  arms = {arm: make_parsers(module, layout) for arm, module in modules.items()}
  trace = hashlib.sha256()
  for frame, packet in enumerate(packets):
    states = {arm: [signature(parser, parser.update(packet)) for parser in parsers]
              for arm, parsers in arms.items()}
    assert states['baseline'] == states['candidate'], frame
    trace.update(json.dumps(states['baseline'], sort_keys=True).encode())
  return {'status': 'PASS', 'frames': len(packets), 'trace_sha256': trace.hexdigest()}


def raw_equivalent(modules):
  payload = bytes((index * 173 + 61) % 256 for index in range(64))
  count = 0
  for little_endian in (False, True):
    for size in range(1, 65):
      for start in (*range(16), 31, 63):
        signal = signal_at(start, size, little_endian)
        for length in (0, 1, 2, 4, 8, 16, 64):
          for buffer in (payload[:length], bytearray(payload[:length])):
            values = {arm: module.get_raw_value(buffer, signal) for arm, module in modules.items()}
            assert values['baseline'] == values['candidate'], (start, size, little_endian, length)
            count += 1
  return {'status': 'PASS', 'comparisons': count}


def measure(module, layout, packets, warmup):
  parsers = make_parsers(module, layout)
  for packet in packets[:warmup]:
    for parser in parsers:
      parser.update(packet)
      _ = parser.can_valid, parser.bus_timeout
  cpu_start, wall_start = time.thread_time_ns(), time.perf_counter_ns()
  for packet in packets[warmup:]:
    for parser in parsers:
      parser.update(packet)
      _ = parser.can_valid, parser.bus_timeout
  frames = len(packets) - warmup
  return {'cpu_us_per_tick': (time.thread_time_ns() - cpu_start) / frames / 1000,
          'wall_us_per_tick': (time.perf_counter_ns() - wall_start) / frames / 1000}


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
  gc.disable()
  modules, identities = {}, {}
  for arm in ('baseline', 'candidate'):
    modules[arm], identities[arm] = load(getattr(args, arm), 'cpu4_parser_' + arm)
  assert identities['baseline']['dependencies'] == identities['candidate']['dependencies']
  layout = subscriptions()
  dbcs = get_generated_dbcs()
  dbc_identity = {item['dbc']: hashlib.sha256(dbcs[item['dbc']].encode()).hexdigest() for item in layout}
  raw_result = raw_equivalent(modules)
  faults_result = equivalent(modules, layout, inputs(layout, 500, mixed_cadence=True, faults=True))
  results = []
  warmup = 300
  for name, mixed_cadence in (('accord-mixed-100-50-20-10hz', True), ('accord-all-messages-100hz', False)):
    packets = inputs(layout, args.frames + warmup, mixed_cadence=mixed_cadence)
    equivalence = equivalent(modules, layout, packets)
    rounds = []
    for repeat in range(args.repeats):
      order = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
      rounds.append({arm: measure(modules[arm], layout, packets, warmup) for arm in order})
    medians = {arm: {key: statistics.median(row[arm][key] for row in rounds) for key in rounds[0][arm]}
               for arm in modules}
    results.append({'case': name, 'equivalence': equivalence, 'medians': medians, 'rounds': rounds,
                    'cpu_reduction_percent': 100 * (1 - medians['candidate']['cpu_us_per_tick'] /
                                                   medians['baseline']['cpu_us_per_tick'])})
  output = {'method': 'interleaved-accord-canparser-update-and-health-v1', 'identities': identities,
            'layout': layout, 'generated_dbc_sha256': dbc_identity, 'raw_equivalence': raw_result,
            'fault_recovery_equivalence': faults_result, 'frames_per_round': args.frames,
            'repeats': args.repeats, 'results': results, 'script_sha256': digest(Path(__file__)),
            'scope': __doc__, 'python': sys.version, 'gc_enabled': gc.isenabled()}
  args.output.write_text(json.dumps(output, indent=2) + '\n')
  print(json.dumps({**output, 'results': [{key: value for key, value in row.items() if key != 'rounds'}
                                        for row in results]}, indent=2))


if __name__ == '__main__':
  main()
