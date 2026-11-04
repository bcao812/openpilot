"""Paired CPU benchmark of plan decoding plus unchanged reference consumption.

Run inside the committed candidate's isolated host snapshot. The baseline uses
its actual controlsd module; shared timing-validation code must be identical.
This does not measure a full control cycle, ARM execution or scheduling.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

from openpilot.cereal import log, messaging
from openpilot.selfdrive.controls import controlsd
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_TIMING_VERSION, ReferenceTimingConsumer
from openpilot.selfdrive.modeld.constants import ModelConstants


BASE_NS = 10_000_000_000
STEP_NS = 10_000_000
CONTROL_PATH = Path('openpilot/selfdrive/controls/controlsd.py')
TIMING_PATH = Path('openpilot/selfdrive/controls/lib/reference_timing.py')


def sha(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def load_control(tree):
  name = '_cpu4_baseline_controlsd'
  spec = importlib.util.spec_from_file_location(name, tree / CONTROL_PATH)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


def plans(count, reuse):
  result = []
  for index in range(count):
    stamp = BASE_NS + index * reuse * STEP_NS
    event = log.Event.new_message()
    event.logMonoTime = stamp
    event.valid = True
    plan = event.init('longitudinalPlan')
    plan.modelMonoTime = stamp
    plan.aTarget = 0.5
    plan.solverStatus = 0
    plan.speeds = [10.0 + 0.5 * float(t) for t in ModelConstants.T_IDXS[:controlsd.CONTROL_N]]
    plan.accels = [0.5] * controlsd.CONTROL_N
    intent = plan.motionIntent
    intent.version = MOTION_INTENT_VERSION
    intent.mode = int(LongitudinalMode.UNIFIED)
    intent.valid = intent.ready = True
    intent.producerInstanceMonoTime = BASE_NS
    reference = plan.reference
    reference.accel = 0.5
    reference.constraintAccel = 2.0
    reference.timingVersion = REFERENCE_TIMING_VERSION
    reference.timingKnown = reference.initialStateKnown = True
    reference.sourceCarStateMonoTime = reference.referenceMonoTime = reference.producerMonoTime = stamp
    reference.producerInstanceMonoTime = BASE_NS
    reference.sourceVEgo = reference.initialVEgo = 10.0
    reference.sourceAEgo = reference.initialAEgo = 0.0
    received = messaging.log_from_bytes(event.to_bytes())
    result.append(received.longitudinalPlan)
  return result


class Runner:
  def __init__(self, module, use_projection):
    self.control = module.Controls.__new__(module.Controls)
    self.control.CP = SimpleNamespace(longitudinalActuatorDelay=0.5)
    self.control.LoC = SimpleNamespace(longitudinal_mode=LongitudinalMode.UNIFIED)
    self.control.sm = SimpleNamespace(recv_frame={}, logMonoTime={})
    self.control._longitudinal_plan_cache = None
    self.consumer = ReferenceTimingConsumer()
    self.use_projection = use_projection

  def step(self, plan, index, reuse):
    received = index // reuse
    available = BASE_NS + received * reuse * STEP_NS
    self.control.sm.recv_frame['longitudinalPlan'] = received + 1
    self.control.sm.logMonoTime['longitudinalPlan'] = available
    _, identity = self.control._decode_longitudinal_plan(plan)
    if self.use_projection:
      cached = self.control._longitudinal_plan_cache
      message = cached.reference_timing if cached is not None else plan.reference
    else:
      message = plan.reference
    return self.consumer.consume(message, available, BASE_NS + (index + 1) * STEP_NS, command_identity=identity)


def check_equivalence(baseline, messages, reuse):
  old, new = Runner(baseline, False), Runner(controlsd, True)
  frames = min(len(messages) * reuse, 300)
  for index in range(frames):
    plan = messages[index // reuse]
    expected, actual = old.step(plan, index, reuse), new.step(plan, index, reuse)
    assert expected.valid and actual == expected, (index, expected, actual)
    assert vars(old.consumer) == vars(new.consumer), index
  return frames


def measure(module, use_projection, messages, reuse, frames):
  runner = Runner(module, use_projection)
  started = time.thread_time_ns()
  for index in range(frames):
    runner.step(messages[index // reuse], index, reuse)
  return (time.thread_time_ns() - started) / frames / 1000


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--baseline', type=Path, required=True)
  parser.add_argument('--candidate', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--frames', type=int, default=20000)
  parser.add_argument('--rounds', type=int, default=8)
  args = parser.parse_args()
  assert args.frames > 0 and args.frames % 5 == 0 and args.rounds >= 2
  assert Path(controlsd.__file__).resolve() == (args.candidate / CONTROL_PATH).resolve()
  assert sha(args.baseline / TIMING_PATH) == sha(args.candidate / TIMING_PATH)
  baseline = load_control(args.baseline)
  rows = []
  for reuse in (5, 1):
    messages = plans(args.frames // reuse, reuse)
    equivalent_frames = check_equivalence(baseline, messages, reuse)
    timings = {'baseline': [], 'candidate': []}
    for arm, module, projected in (('baseline', baseline, False), ('candidate', controlsd, True)):
      measure(module, projected, messages, reuse, min(args.frames, 1000))
    for iteration in range(args.rounds):
      order = ('baseline', 'candidate') if iteration % 2 == 0 else ('candidate', 'baseline')
      for arm in order:
        value = measure(baseline if arm == 'baseline' else controlsd, arm == 'candidate', messages, reuse, args.frames)
        timings[arm].append(value)
    before, after = (statistics.median(timings[arm]) for arm in ('baseline', 'candidate'))
    row = {'new_plan_every_frames': reuse, 'equivalent_frames': equivalent_frames,
           'frames_per_round': args.frames, 'rounds': args.rounds, 'cpu_us_per_frame': timings,
           'baseline_median_us': before, 'candidate_median_us': after,
           'saved_us': before - after, 'change_pct': 100 * (after / before - 1)}
    rows.append(row)
    print(json.dumps(row), flush=True)
  output = {'scope': __doc__, 'baseline_commit': json.loads((args.baseline / '.host-build-source.json').read_text())['source_commit'],
            'candidate_commit': json.loads((args.candidate / '.host-build-source.json').read_text())['source_commit'],
            'controlsd_sha256': {arm: sha(tree / CONTROL_PATH) for arm, tree in
                                (('baseline', args.baseline), ('candidate', args.candidate))},
            'unchanged_reference_timing_sha256': sha(args.candidate / TIMING_PATH),
            'benchmark_sha256': sha(Path(__file__)), 'rows': rows}
  args.output.write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
  main()
