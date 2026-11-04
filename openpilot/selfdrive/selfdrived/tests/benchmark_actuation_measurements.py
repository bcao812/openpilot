"""Compare committed selfdrived update_events with the two pose-input paths.

Each arm loads its own selfdrived, pose-helper and actuation-check source. The
other imports and bounded event fixture are shared from the invoking candidate.
Timers include the real update_events, current measurement projection and the
100 Hz check; socket IO, vehicle event producers and ICBM remain fixture bounds.
This is a host software benchmark, not device CPU4 or complete-process timing.
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
from types import SimpleNamespace

import numpy as np

from openpilot.cereal import log
from openpilot.selfdrive.selfdrived.tests.test_actuation_measurements import (
  CALIBRATIONS, device_motion, event_harness, event_state, scalar_bits,
)


MODULES = {'pose': 'openpilot/selfdrive/locationd/helpers.py',
           'check': 'openpilot/selfdrive/selfdrived/helpers.py',
           'daemon': 'openpilot/selfdrive/selfdrived/selfdrived.py'}


def sha(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root, arm):
  modules, files = {}, {}
  for kind, relative in MODULES.items():
    path = root / relative
    name = f'actuation_bench_{arm}_{kind}'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    modules[kind] = module
    files[relative] = sha(path)
  marker = root / '.host-build-source.json'
  return modules, {'source_commit': json.loads(marker.read_text())['source_commit'],
                   'source_manifest_sha256': sha(marker), 'module_sha256': files}


class EventArm:
  def __init__(self, modules):
    self.modules = modules
    self.daemon, self.CS = event_harness()
    self.daemon.__class__ = modules['daemon'].SelfdriveD
    self.daemon.pose_calibrator = modules['pose'].PoseCalibrator()
    self.daemon.excessive_actuation_check = modules['check'].ExcessiveActuationCheck()
    # Each source arm only uses its own private field.
    self.daemon.calibrated_pose = self.daemon.calibrated_actuation = None
    self.alerts = []
    self.now_ns = 2_000_000_000
    modules['daemon'].set_offroad_alert = lambda *args, **kwargs: self.alerts.append((args, kwargs))
    modules['daemon'].cloudlog = SimpleNamespace(event=lambda *args, **kwargs: None)
    modules['daemon'].time = SimpleNamespace(monotonic=lambda: self.now_ns / 1e9,
                                            monotonic_ns=lambda: self.now_ns)
    self.calibrations = [log.ExtrinsicsCalibration(rpyCalib=list(rpy), calStatus='calibrated') for rpy in CALIBRATIONS]
    self.normal_motion = device_motion()
    self.scenario_motion = [device_motion((accel, -0.1, 0.2), (0.004, -0.002, yaw))
                            for accel, yaw in ((0.2, 0.01), (8.0, 0.01), (-9.0, 0.01),
                                               (0.2, 0.7), (0.2, 0.7), (np.nan, 0.01), (0.2, 0.01))]

  def step(self, frame, period, *, scenario=False):
    daemon, CS = self.daemon, self.CS
    self.now_ns = 2_000_000_000 + frame * 10_000_000
    daemon.sm.frame = frame
    daemon.sm.updated['extrinsicsCalibration'] = frame % 73 == 0
    daemon.sm.updated['deviceMotion'] = frame % period == 0
    daemon.sm.data['extrinsicsCalibration'] = self.calibrations[(frame // 73) % len(self.calibrations)]
    stage = frame // 90 % len(self.scenario_motion) if scenario else 0
    motion = self.scenario_motion[stage] if scenario else self.normal_motion
    daemon.sm.data['deviceMotion'] = motion
    CS.aEgo = motion.accelerationDevice.x
    CS.vEgo = 20.0
    CS.steeringPressed = scenario and stage == 4
    daemon.sm.data['carControl'].latActive = not (scenario and frame % 160 >= 145)
    daemon.sm.data['carControl'].longActive = not (scenario and frame % 110 >= 100)
    daemon.update_events(CS)

  def state(self):
    if self.daemon.calibrated_pose is not None:
      pose = self.daemon.calibrated_pose
      values = pose.acceleration.x, pose.angular_velocity.yaw
    else:
      values = self.daemon.calibrated_actuation
    return {**event_state(self.daemon), 'values_bits': scalar_bits(values).hex(), 'alerts': self.alerts}


def equivalent(modules):
  results = []
  for period in (5, 1):
    arms = {name: EventArm(module) for name, module in modules.items()}
    trace = hashlib.sha256()
    with np.errstate(all='ignore'):
      for frame in range(720):
        for arm in arms.values():
          arm.step(frame, period, scenario=True)
        baseline, candidate = arms['baseline'].state(), arms['candidate'].state()
        assert baseline == candidate, (period, frame, baseline, candidate)
        trace.update(json.dumps(baseline, sort_keys=True).encode())
    results.append({'motion_period_frames': period, 'equivalent_frames': 720,
                    'observable_state_sha256': trace.hexdigest()})
  return results


def measure(modules, frames, period):
  arm = EventArm(modules)
  for frame in range(100):
    arm.step(frame, period)
  started_cpu, started_wall = time.thread_time_ns(), time.perf_counter_ns()
  for frame in range(100, 100 + frames):
    arm.step(frame, period)
  return {'cpu_us_per_event_update': (time.thread_time_ns() - started_cpu) / frames / 1000,
          'wall_us_per_event_update': (time.perf_counter_ns() - started_wall) / frames / 1000}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--baseline', type=Path, required=True)
  parser.add_argument('--candidate', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--frames', type=int, default=3000)
  parser.add_argument('--repeats', type=int, default=8)
  args = parser.parse_args()
  if args.frames < 1 or args.repeats < 1:
    parser.error('frames and repeats must be positive')
  modules, identities = {}, {}
  for name in ('baseline', 'candidate'):
    modules[name], identities[name] = load(getattr(args, name), name)
  dependencies = {}
  for name in ('openpilot.cereal.messaging',
               'openpilot.selfdrive.controls.tests.longitudinal_process_test_support',
               'openpilot.selfdrive.selfdrived.tests.test_actuation_measurements'):
    path = Path(sys.modules[name].__file__).resolve()
    if not path.is_relative_to(args.candidate.resolve()):
      raise ValueError(f'Shared dependency outside candidate: {path}')
    dependencies[name] = {'path': str(path), 'sha256': sha(path)}
  gc.disable()
  checks = equivalent(modules)
  results = []
  for period in (5, 1):
    rounds = []
    for repeat in range(args.repeats):
      order = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
      rounds.append({name: measure(modules[name], args.frames, period) for name in order})
    medians = {name: statistics.median(row[name]['cpu_us_per_event_update'] for row in rounds) for name in modules}
    results.append({'motion_period_frames': period, 'cpu_us_per_event_update_median': medians,
                    'reduction_pct': 100 * (1 - medians['candidate'] / medians['baseline']), 'rounds': rounds})
  output = {'identities': identities, 'shared_candidate_dependencies': dependencies,
            'scope': __doc__, 'equivalence': checks, 'results': results, 'gc_enabled': gc.isenabled()}
  args.output.write_text(json.dumps(output, indent=2) + '\n')
  print(json.dumps({'equivalence': checks, 'results': [{key: value for key, value in item.items() if key != 'rounds'}
                                                    for item in results]}))


if __name__ == '__main__':
  main()
