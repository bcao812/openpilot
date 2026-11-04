"""Compare policy updates from two committed host snapshots on identical inputs.

Timing includes identical input timestamp advancement. It excludes fixture
construction and tests this Python policy, not device CPU4 or control latency.
Dependencies and schema are shared from the invoking candidate environment;
only engagement_policy.py is selected separately for each arm.
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

from opendbc.car.structs import car
from openpilot.cereal import log
import openpilot.cereal.messaging as messaging
from openpilot.selfdrive.selfdrived.tests.test_engagement_policy import (
  CONTROL_STEP_NS, NOW, EventName, add_long_failure, configure_pandas, make_daemon,
)

MODULE = 'openpilot/selfdrive/selfdrived/engagement_policy.py'
CASES = ('moving', 'multiple_pandas', 'longitudinal_fault', 'eps_fault', 'held', 'invalid_panda', 'invalid_model')


def digest(path):
  return hashlib.sha256(path.read_bytes()).hexdigest()


def load(root, name):
  path = root / MODULE
  spec = importlib.util.spec_from_file_location(name, path)
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  marker = root / '.host-build-source.json'
  source = json.loads(marker.read_text())
  return module, {'source_commit': source['source_commit'], 'module_sha256': digest(path),
                  'source_manifest_sha256': digest(marker)}


def inputs(module, case):
  daemon, CS = make_daemon(held=case in ('held', 'held_fault'))
  daemon.engagement_policy = module.EngagementPolicy()
  if case == 'multiple_pandas':
    configure_pandas(daemon, [(True, True), (True, True)], ignored_tail='noOutput')
  elif case in ('longitudinal_fault', 'held_fault'):
    add_long_failure(daemon)
  elif case == 'eps_fault':
    CS.steerFaultTemporary = True
    daemon.events.add(EventName.steerTempUnavailable)
  elif case == 'invalid_panda':
    daemon.sm['pandaStates'][0].safetyRxChecksInvalid = True
  elif case == 'invalid_model':
    daemon.sm.valid['modelV2'] = False
  return daemon, CS


def freeze_inputs(daemon, CS, stack):
  """Use received schema readers for static bodies, keeping LP's clock writable."""
  daemon.CP = stack.enter_context(car.CarParams.from_bytes(daemon.CP.to_bytes()))
  CS = stack.enter_context(car.CarState.from_bytes(CS.to_bytes()))
  for name, body in daemon.sm.items():
    if name == 'longitudinalPlan':
      continue
    event = messaging.new_message(name, len(body) if name == 'pandaStates' else None)
    value = [item.to_dict() for item in body] if name == 'pandaStates' else body.to_dict()
    setattr(event, name, value)
    reader = stack.enter_context(log.Event.from_bytes(event.to_bytes()))
    daemon.sm[name] = getattr(reader, name)
  return CS


def advance(daemon, frame):
  now_ns = NOW + frame * CONTROL_STEP_NS
  for name in daemon.sm.logMonoTime:
    daemon.sm.logMonoTime[name] = now_ns
  daemon._car_state_mono_ns = now_ns
  daemon.sm['longitudinalPlan'].modelMonoTime = now_ns
  daemon._longitudinal_soft_disable_remaining_ns = max(0, 3_000_000_000 - (now_ns - NOW))
  return now_ns


def observable_state(module, daemon, CS, frame):
  policy = daemon.engagement_policy
  decision = policy.update(daemon, CS, advance(daemon, frame))
  daemon.mads.longitudinal_failure_isolated = decision.preserve_lateral
  daemon.enabled, daemon.active = daemon.state_machine.update(
    module.LongitudinalEvents(daemon.events, decision.preserve_longitudinal,
                              recovered_model_lag=decision.recovered_model_lag),
    soft_disable_remaining_ns=daemon._longitudinal_soft_disable_remaining_ns,
    hold_for_takeover=decision.hold_for_takeover,
  )
  policy.observe_states(daemon, CS)
  alert = policy.alert(daemon, CS)
  return {'decision': asdict(decision), 'handoff': vars(policy.handoff),
          'available': (policy.lateral_available, policy.longitudinal_available),
          'takeover': (policy.speed_takeover, policy.steering_takeover, policy.hold_ended_until_ns),
          'state': str(daemon.state_machine.state), 'enabled': daemon.enabled, 'active': daemon.active,
          'timer': daemon.state_machine.soft_disable_timer,
          'alert': None if alert is None else (alert.alert_type, alert.alert_text_1, alert.alert_text_2)}


def equivalent(modules):
  evidence = {}
  for case in (*CASES, 'held_fault'):
    arms = {arm: inputs(module, case) for arm, module in modules.items()}
    trace = hashlib.sha256()
    for frame in range(650):
      if case == 'held_fault' and frame == 310:
        for daemon, _ in arms.values():
          daemon.events.clear()
          daemon.sm['controlsState'].motionState.degraded = False
          daemon.sm['longitudinalPlan'].solverStatus = 0
          daemon.sm['longitudinalPlan'].motionIntent.valid = True
          daemon.sm.valid['longitudinalPlan'] = True
      states = {arm: observable_state(modules[arm], *value, frame) for arm, value in arms.items()}
      assert states['baseline'] == states['candidate'], (case, frame, states)
      trace.update(json.dumps(states['baseline'], sort_keys=True).encode())
    evidence[case] = {'equivalent_frames': 650, 'observable_trace_sha256': trace.hexdigest()}
  return evidence


def measure(module, case, frames):
  daemon, CS = inputs(module, case)
  with ExitStack() as stack:
    CS = freeze_inputs(daemon, CS, stack)
    policy = daemon.engagement_policy
    for frame in range(50):
      policy.update(daemon, CS, advance(daemon, frame))
    started_cpu, started_wall = time.thread_time_ns(), time.perf_counter_ns()
    for frame in range(50, 50 + frames):
      policy.update(daemon, CS, advance(daemon, frame))
    return {'cpu_us_per_update': (time.thread_time_ns() - started_cpu) / frames / 1000,
            'wall_us_per_update': (time.perf_counter_ns() - started_wall) / frames / 1000}


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
    modules[arm], identities[arm] = load(getattr(args, arm), 'engagement_benchmark_' + arm)
  dependency_names = (
    'openpilot.selfdrive.controls.lib.longitudinal_mode',
    'openpilot.selfdrive.controls.lib.longitudinal_continuation',
    'openpilot.selfdrive.controls.lib.motion_supervisor',
    'openpilot.selfdrive.selfdrived.model_lag_recovery',
    'openpilot.selfdrive.selfdrived.events',
    'openpilot.selfdrive.selfdrived.state',
    'openpilot.selfdrive.selfdrived.tests.test_engagement_policy',
  )
  dependencies = {}
  for name in dependency_names:
    path = Path(sys.modules[name].__file__).resolve()
    if not path.is_relative_to(args.candidate.resolve()):
      raise ValueError(f'Shared dependency is outside the candidate snapshot: {path}')
    dependencies[name] = {'path': str(path), 'sha256': digest(path)}
  equivalence = equivalent(modules)
  results = []
  for case in CASES:
    rounds = []
    for repeat in range(args.repeats):
      order = ('baseline', 'candidate') if repeat % 2 == 0 else ('candidate', 'baseline')
      rounds.append({arm: measure(modules[arm], case, args.frames) for arm in order})
    medians = {arm: statistics.median(row[arm]['cpu_us_per_update'] for row in rounds) for arm in modules}
    ratios = [row['candidate']['cpu_us_per_update'] / row['baseline']['cpu_us_per_update'] for row in rounds]
    results.append({'case': case, 'cpu_us_per_update_median': medians,
                    'paired_reduction_pct_median': 100 * (1 - statistics.median(ratios)), 'rounds': rounds})
  output = {'identities': identities, 'equivalence': equivalence, 'results': results,
            'frames_per_round': args.frames, 'repeats': args.repeats, 'script_sha256': digest(Path(__file__)),
            'shared_candidate_dependencies': dependencies, 'python': sys.version,
            'scope': __doc__, 'device_validation': 'NOT RUN'}
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(output, indent=2) + '\n')
  print(json.dumps({row['case']: row['paired_reduction_pct_median'] for row in results}, indent=2))


if __name__ == '__main__':
  main()
