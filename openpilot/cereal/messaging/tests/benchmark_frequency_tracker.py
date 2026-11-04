"""Evaluate frequency-window work without changing production messaging behavior.

Both experimental variants are confined to this benchmark. The real SubMaster
update_msgs implementation executes with serialized synthetic messages, unchanged
clock/health checks and bounded windows. Socket polling and device timing are not
measured. Use committed isolated snapshots and the workspace resource runner.
"""
import argparse
from contextlib import ExitStack
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import struct
import sys
import time
from unittest.mock import patch

MODULE = 'openpilot/cereal/messaging/__init__.py'
DEPENDENCIES = ('openpilot/common/utils.py', 'openpilot/cereal/services.py')
SERVICES = ('carState', 'controlsState', 'selfdriveState', 'modelV2', 'radarState',
            'longitudinalPlan', 'deviceState', 'pandaStates', 'managerState', 'carParams')
VARIANTS = ('baseline', 'strict_cache', 'inline_average')


def sha(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root, name):
  spec = importlib.util.spec_from_file_location(name, root / MODULE)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  marker = root / '.host-build-source.json'
  return module, {'source_commit': json.loads(marker.read_text())['source_commit'],
                  'module_sha256': sha(root / MODULE), 'source_manifest_sha256': sha(marker)}


def experiment_class(module, variant):
  original = module.FrequencyTracker
  original_valid = original.valid.fget
  window_type = module.MovingAverage
  if variant == 'baseline':
    return original

  if variant == 'strict_cache':
    class StrictCache(original):
      def __init__(self, *args):
        super().__init__(*args)
        self._valid_key = None
        self._valid_result = False

      @property
      def valid(self):
        long, recent = self.avg_dt, self.recent_avg_dt
        if type(long) is not window_type or type(recent) is not window_type:
          return original_valid(self)
        key = (long.sum, long.count, recent.sum, recent.count, self.min_freq, self.max_freq)
        # Public mutable fields and nonstandard numeric values keep their original
        # behavior. No exception is cached or moved into record_recv_time().
        if (type(key[0]) is not float or type(key[1]) is not int or
            type(key[2]) is not float or type(key[3]) is not int or
            type(key[4]) is not float or type(key[5]) is not float):
          return original_valid(self)
        if key == self._valid_key:
          return self._valid_result
        result = original_valid(self)
        self._valid_key, self._valid_result = key, result
        return result
    return StrictCache

  class InlineAverage(original):
    @property
    def valid(self):
      long, recent = self.avg_dt, self.recent_avg_dt
      if type(long) is not window_type or type(recent) is not window_type:
        return original_valid(self)
      if long.count == 0:
        return False
      # Keep the original division order; count / sum has different rounding.
      avg_freq = 1.0 / (long.sum / long.count)
      if self.min_freq <= avg_freq <= self.max_freq:
        return True
      recent_average = recent.sum / recent.count if recent.count else float('nan')
      avg_freq_recent = 1.0 / recent_average
      return self.min_freq <= avg_freq_recent <= self.max_freq
  return InlineAverage


def frozen(value):
  if isinstance(value, float):
    return ('float', struct.pack('>d', value).hex())
  if isinstance(value, dict):
    return {key: frozen(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return [frozen(item) for item in value]
  return value


def outcome(tracker):
  try:
    return ('value', bool(tracker.valid))
  except Exception as error:
    return ('exception', type(error).__name__, str(error))


def tracker_state(tracker):
  return frozen({'prev_time': tracker.prev_time, 'min_freq': tracker.min_freq, 'max_freq': tracker.max_freq,
                 'avg_dt': {key: value for key, value in vars(tracker.avg_dt).items() if key != 'buffer'},
                 'recent_avg_dt': {key: value for key, value in vars(tracker.recent_avg_dt).items() if key != 'buffer'}})


def tracker_buffers(tracker):
  return frozen((tracker.avg_dt.buffer, tracker.recent_avg_dt.buffer))


def tracker_equivalence(modules):
  count = 0
  configs = [(freq, 100., poll) for freq in (0., 2., 10., 20., 100.) for poll in (False, True)]
  for config in configs:
    trackers = {name: module.FrequencyTracker(*config) for name, module in modules.items()}
    for frame in range(1600):
      now = 2. + frame / 100.
      period = max(1, round(100. / max(config[0], 1.)))
      for tracker in trackers.values():
        if frame % period == 0 and not 300 <= frame < 450:
          tracker.record_recv_time(now)
      values = {name: (outcome(tracker), tracker_state(tracker)) for name, tracker in trackers.items()}
      assert all(value == values['baseline'] for value in values.values()), (config, frame, values)
      count += 1
    buffers = {name: tracker_buffers(tracker) for name, tracker in trackers.items()}
    assert all(value == buffers['baseline'] for value in buffers.values()), config

  mutations = (
    lambda t: setattr(t, 'min_freq', 1000.),
    lambda t: setattr(t, 'max_freq', 2000.),
    lambda t: setattr(t, 'min_freq', -1000.),
    lambda t: t.avg_dt.add_value(.5),
    lambda t: t.recent_avg_dt.add_value(.01),
    lambda t: setattr(t.avg_dt, 'sum', -0.0),
    lambda t: setattr(t.avg_dt, 'sum', 0.0),
    lambda t: setattr(t.avg_dt, 'sum', float('nan')),
    lambda t: setattr(t.avg_dt, 'sum', float('inf')),
    lambda t: setattr(t.avg_dt, 'sum', 1),  # non-float fallback
    lambda t: setattr(t.recent_avg_dt, 'count', 0),
    lambda t: setattr(t.avg_dt, 'count', 0),
  )
  for mutate in mutations:
    trackers = {name: module.FrequencyTracker(20., 100., False) for name, module in modules.items()}
    for tracker in trackers.values():
      tracker.record_recv_time(2.)
      tracker.record_recv_time(2.05)
      outcome(tracker)  # populate a possible cache before direct mutation
      mutate(tracker)
    values = {name: (outcome(tracker), tracker_state(tracker)) for name, tracker in trackers.items()}
    assert all(value == values['baseline'] for value in values.values()), values
    count += 1

  for sequence in ((2., 2.), (2., 1.), (2., float('nan')), (2., float('inf'))):
    trackers = {name: module.FrequencyTracker(20., 100., True) for name, module in modules.items()}
    for now in sequence:
      for tracker in trackers.values():
        tracker.record_recv_time(now)
      values = {name: (outcome(tracker), tracker_state(tracker)) for name, tracker in trackers.items()}
      assert all(value == values['baseline'] for value in values.values()), values
      count += 1

  class CustomWindow:
    def __init__(self):
      self.count, self.calls = 1, 0

    def get_average(self):
      self.calls += 1
      return .05 if self.calls % 2 else 1.

  trackers = {name: module.FrequencyTracker(20., 100., False) for name, module in modules.items()}
  for tracker in trackers.values():
    tracker.avg_dt, tracker.recent_avg_dt = CustomWindow(), CustomWindow()
  for _ in range(8):
    values = {name: (outcome(tracker), tracker_state(tracker)) for name, tracker in trackers.items()}
    assert all(value == values['baseline'] for value in values.values()), values
    count += 1
  return {'status': 'PASS', 'compared_frames_and_mutations': count,
          'comparison': 'every validity/exception and scalar window state; full ring buffers at each sequence end'}


def make_submaster(module):
  with ExitStack() as stack:
    stack.enter_context(patch.object(module, 'Poller'))
    stack.enter_context(patch.object(module, 'sub_sock'))
    master = module.SubMaster(list(SERVICES), frequency=100.)
  master.simulation = False
  return master


def schedule(module, case, frames):
  messages = []
  for frame in range(frames):
    now = 2. + frame / 100.
    incoming = []
    for service in SERVICES:
      frequency = module.SERVICE_LIST[service].frequency
      period = max(1, round(100. / max(frequency, 1.)))
      if case == 'every_frame':
        period = 1
      elif case == 'slow_and_recovery' and 300 <= frame < 800:
        period *= 4
      due = frame % period == 0
      if case == 'outage_and_recovery' and 300 <= frame < 550 and service != 'carState':
        due = False
      if not due:
        continue
      event = module.new_message(service, 0 if service == 'pandaStates' else None,
                                 valid=not (frame % 181 == 0), logMonoTime=round(now * 1e9))
      incoming.append(module.log_from_bytes(event.to_bytes()))
    messages.append((now, incoming))
  return messages


def master_state(master):
  fields = ('frame', 'seen', 'updated', 'recv_time', 'recv_frame', 'logMonoTime', 'alive', 'freq_ok', 'valid')
  result = {field: getattr(master, field) for field in fields}
  result['checks'] = (master.all_alive(), master.all_freq_ok(), master.all_valid(), master.all_checks())
  result['windows'] = {name: tracker_state(tracker) for name, tracker in master.freq_tracker.items()}
  return frozen(result)


def measure(module, events):
  master = make_submaster(module)
  for now, incoming in events[:100]:
    master.update_msgs(now, incoming)
  start = time.thread_time_ns()
  for now, incoming in events[100:]:
    master.update_msgs(now, incoming)
  return (time.thread_time_ns() - start) / (len(events) - 100) / 1000.


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--baseline', type=Path, required=True)
  parser.add_argument('--candidate', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--frames', type=int, default=1600)
  parser.add_argument('--repeats', type=int, default=8)
  args = parser.parse_args()
  if args.frames < 1000 or args.repeats < 1:
    parser.error('frames must be >= 1000 and repeats must be positive')
  if (args.baseline / MODULE).read_bytes() != (args.candidate / MODULE).read_bytes():
    raise ValueError('This prototype requires unchanged production messaging in both snapshots')
  for dependency in DEPENDENCIES:
    assert (args.baseline / dependency).read_bytes() == (args.candidate / dependency).read_bytes(), dependency
  modules, identities = {}, {}
  for name in VARIANTS:
    root = args.baseline if name == 'baseline' else args.candidate
    module, identities[name] = load(root, 'frequency_experiment_' + name)
    module.FrequencyTracker = experiment_class(module, name)
    modules[name] = module
  for module_name in ('openpilot.common.utils', 'openpilot.cereal.services'):
    if not Path(sys.modules[module_name].__file__).resolve().is_relative_to(args.candidate.resolve()):
      raise ValueError(f'Shared dependency is outside the candidate snapshot: {module_name}')
  gc.disable()
  verification = tracker_equivalence(modules)
  results = []
  for case in ('mixed_rates', 'every_frame', 'outage_and_recovery', 'slow_and_recovery'):
    events = schedule(modules['baseline'], case, args.frames)
    masters = {name: make_submaster(module) for name, module in modules.items()}
    outage_seen = False
    for frame, (now, incoming) in enumerate(events):
      states = {}
      for name, master in masters.items():
        if case == 'outage_and_recovery':
          master.simulation = 400 <= frame < 425
        master.update_msgs(now, incoming)
        states[name] = master_state(master)
      assert all(state == states['baseline'] for state in states.values()), (case, frame)
      if case == 'outage_and_recovery' and 350 < frame < 550:
        outage_seen |= not masters['baseline'].alive['modelV2']
    if case == 'outage_and_recovery':
      assert outage_seen, 'outage fixture must exercise alive expiration'
    buffers = {name: {service: tracker_buffers(tracker) for service, tracker in master.freq_tracker.items()}
               for name, master in masters.items()}
    assert all(value == buffers['baseline'] for value in buffers.values()), case
    rounds = []
    for repeat in range(args.repeats):
      order = VARIANTS if repeat % 2 == 0 else tuple(reversed(VARIANTS))
      rounds.append({name: measure(modules[name], events) for name in order})
    medians = {name: statistics.median(row[name] for row in rounds) for name in VARIANTS}
    item = {'case': case, 'equivalent_frames': len(events), 'cpu_us_per_update_median': medians,
            'cpu_reduction_pct': {name: 100. * (1. - medians[name] / medians['baseline']) for name in VARIANTS[1:]},
            'rounds': rounds}
    results.append(item)
    print(json.dumps({key: value for key, value in item.items() if key != 'rounds'}), flush=True)
  result = {'source': identities, 'prototype_sha256': sha(Path(__file__)),
            'shared_dependencies': {name: sha(args.candidate / name) for name in DEPENDENCIES},
            'tracker_equivalence': verification, 'results': results,
            'services_hz': {name: modules['baseline'].SERVICE_LIST[name].frequency for name in SERVICES},
            'scope': __doc__, 'production_code_changed': False, 'device_access': False,
            'timing': 'thread CPU, unprofiled, GC disabled, alternating variant order'}
  args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
  main()
