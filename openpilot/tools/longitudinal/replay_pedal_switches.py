#!/usr/bin/env python3
"""Local, timestamp-aligned open-loop planner/LongControl/Honda CAN regression.

Writes only below --output. Recorded motion, perception, SP decisions and controller
scheduling stay fixed; command differences are not changes in vehicle motion.
"""

import argparse
from collections import Counter
import hashlib
import inspect
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import zstandard

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.hondacan import CanBus
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC, CarControllerParams
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib import longitudinal_planner as planner_module
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.tools.longitudinal.replay_longitudinal_regressions import (
  ReplayState, SERVICES, frozen_sp_target, sp_target_return_type,
)
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP
from openpilot.tools.lib.logreader import LogReader

try:
  from openpilot.selfdrive.controls.lib import motion_supervisor as motion_runtime
  from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION as MOTION_PROTOCOL_VERSION
except ModuleNotFoundError as error:
  if error.name != 'openpilot.selfdrive.controls.lib.motion_supervisor':
    raise
  MOTION_PROTOCOL_VERSION = None
  motion_runtime = None

try:
  from openpilot.selfdrive.car.longitudinal_authority import restrict_longitudinal_authority
except ModuleNotFoundError as error:
  if error.name != 'openpilot.selfdrive.car.longitudinal_authority':
    raise
  restrict_longitudinal_authority = None


SCHEMA = 1
REPLAY_PROTOCOL_VERSION = 6
DIAGNOSTIC_SCHEMA_VERSION = 2
EXECUTION_POLICY = 'recorded_id_layerwise_continuous_dependency_closure_v1'
RETAINED = set(SERVICES) | {'longitudinalPlan', 'longitudinalPlanSP', 'sendcan'}
PARAMETERS = {'carParams', 'carParamsSP', 'initData'}
INITIALIZATION_UDS = {bytes.fromhex('0210030000000000'), bytes.fromhex('0328830300000000')}
RISK_WINDOWS = [
  ('00000050--076fbccae6', 1669.0, 1678.06),
  ('0000004e--b6f1c9bc4e', 2280.0, 2305.0),
  ('00000047--f3ea9093b8', 3117.0, 3140.0),
  ('00000050--076fbccae6', 3080.0, 3110.0),
  ('00000053--82133ac34f', 3785.0, 3801.2),
  ('00000048--af2db898b0', 6222.0, 6235.1),
]


def save_json(path, value):
  temporary = path.with_suffix(path.suffix + '.tmp')
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
  temporary.replace(path)


def sha256(path):
  with path.open('rb') as stream:
    return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(logs, audit, output, all_events=False, workers=1, events_file='events-enriched.json'):
  events = json.loads((audit / events_file).read_text())
  selected = [dict(event, event_id=index) for index, event in enumerate(events)
              if all_events or event['category'] in ('cruise_with_lead', 'steady_no_lead_cruise') or event['switches_per_s'] >= 1]
  intervals = [{'route': event['route'], 'start': max(0.0, event['start'] - 2.0), 'end': event['end'] + 2.0,
                'events': [event['event_id']], 'risks': []} for event in selected]
  intervals += [{'route': route, 'start': start, 'end': end, 'events': [], 'risks': [index]}
                for index, (route, start, end) in enumerate(RISK_WINDOWS)]
  merged = []
  for interval in sorted(intervals, key=lambda value: (value['route'], value['start'])):
    if merged and merged[-1]['route'] == interval['route'] and interval['start'] - merged[-1]['end'] <= 60.0:
      merged[-1]['end'] = max(merged[-1]['end'], interval['end'])
      merged[-1]['events'] += interval['events']
      merged[-1]['risks'] += interval['risks']
    else:
      merged.append(interval)
  output.mkdir(parents=True, exist_ok=True)
  jobs = []
  for index, window in enumerate(merged):
    route = window['route']
    metadata = json.loads((audit / f'{route}.native.json').read_text())
    if metadata['schema'] != 1:
      raise ValueError('Unsupported native audit schema')
    with np.load(audit / f'{route}.native.npz', allow_pickle=False) as native:
      times, segments = native['frames'][:, :2].T
      wanted = (times >= window['start'] - 62.0) & (times <= window['end'] + 1.0)
      segment_ids = sorted({int(segment) for segment in segments[wanted]})
    if not segment_ids or any(after - before != 1 for before, after in zip(segment_ids, segment_ids[1:], strict=False)):
      raise ValueError(f'Missing context segments: {window}')
    window.update(origin=metadata['t0_mono'], commit=metadata['commit'], segments=segment_ids,
                  cache=f'{index:03d}-{route}.rlog.zst')
    jobs.append((logs, output, window, metadata))
  if workers == 1:
    results = [prepare_one(job) for job in jobs]
  else:
    with ProcessPoolExecutor(max_workers=workers) as pool:
      results = list(pool.map(prepare_one, jobs))
  inputs = {name: identity for _, sources in results for name, identity in sources.items()}
  merged = [window for window, _ in results]
  save_json(output / 'manifest.json', {'schema': SCHEMA, 'method': 'native audit time and segment IDs, 60s warmup',
                                     'preparer_sha256': sha256(Path(__file__)), 'audit_sha256': sha256(audit / events_file),
                                     'events_file': events_file,
                                     'inputs': inputs, 'windows': merged, 'events': selected})


def prepare_one(arguments):
  logs, output, window, metadata = arguments
  compressed = zstandard.ZstdCompressor(level=3)
  parameter_seen, inputs = set(), {}
  messages_written = 0
  parse_warnings = []
  with (output / window['cache']).open('wb') as target, compressed.stream_writer(target) as writer:
    for segment in window['segments']:
      relative = f'{window["route"]}--{segment}/rlog.zst'
      path = logs / relative
      fingerprint = next(record for record in metadata['fingerprint'] if record[0] == relative)
      if [path.stat().st_size, path.stat().st_mtime_ns] != fingerprint[1:]:
        raise ValueError(f'Changed audit input: {path}')
      inputs[relative] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
      with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always', RuntimeWarning)
        for message in LogReader([str(path)], sort_by_time=True):
          service = message.which()
          timestamp = message.logMonoTime / 1e9 - window['origin']
          if service == 'initData' and (message.initData.gitCommit != window['commit'] or message.initData.dirty):
            raise ValueError(f'Recording version mismatch: {path}')
          if service in PARAMETERS and service not in parameter_seen:
            parameter_seen.add(service)
          elif service not in RETAINED or not window['start'] - 61.0 <= timestamp <= window['end'] + 0.1:
            continue
          writer.write(message.as_builder().to_bytes())
          messages_written += 1
      parse_warnings += [str(item.message) for item in captured]
  if parameter_seen != PARAMETERS:
    raise ValueError(f'Missing recording metadata: {window}')
  window.update(messages=messages_written, sha256=sha256(output / window['cache']), warnings=parse_warnings)
  save_json((output / window['cache']).with_suffix('.json'), {'window': window, 'inputs': inputs})
  print(f'prepared {window["cache"]}', flush=True)
  return window, inputs


def pedal_side(row):
  if row['can_brake'] and row['can_gas'] <= 0:
    return -1
  if not row['can_brake'] and row['can_gas'] > 0:
    return 1
  return 0


def switch_metrics(rows, dwell=0.15):
  if not rows:
    return {'raw': 0, 'debounced': 0, 'switches': []}
  times = np.array([row['t'] for row in rows])
  elapsed = np.diff(times, append=times[-1])
  elapsed = np.where((elapsed > 0) & (elapsed <= 0.15), elapsed, 0.0)
  labels = np.array([pedal_side(row) if row['automatic'] and not (row['can_brake'] and row['can_gas'] > 0) else 9 for row in rows])
  boundaries = np.r_[0, np.flatnonzero((labels[1:] != labels[:-1]) | (elapsed[:-1] == 0)) + 1, len(labels)]
  phases, block = [], 0
  for start, stop in pairwise(boundaries):
    side = int(labels[start])
    duration = times[stop - 1] + elapsed[stop - 1] - times[start]
    if start and elapsed[start - 1] == 0:
      block += 1
    if side == 9 or (side == 0 and duration > 0.1 + 1e-8):
      block += 1
    elif side != 0:
      phases.append({'t': float(times[start]), 'duration': float(duration), 'side': side, 'block': block})
  def changes(minimum):
    accepted = [phase for phase in phases if phase['duration'] + 1e-8 >= minimum]
    return [after['t'] for before, after in pairwise(accepted) if before['block'] == after['block'] and before['side'] != after['side']]
  switches = changes(dwell)
  return {'raw': len(changes(0.0)), 'debounced': len(switches), 'switches': switches}


def paired_messages(messages, maximum_delay=0.25, maximum_messages=512, *, with_identity=False):
  pending = []
  for message in messages:
    service = message.which()
    if pending and (service == 'longitudinalPlan' or len(pending) >= maximum_messages
                    or message.logMonoTime - pending[0].logMonoTime > maximum_delay * 1e9):
      yield from ((buffered, None) for buffered in pending)
      pending = []
    if service == 'longitudinalPlan':
      pending = [message]
    elif pending:
      pending.append(message)
      if service == 'longitudinalPlanSP':
        decision = (SimpleNamespace(value=message.longitudinalPlanSP, mono=int(message.logMonoTime))
                    if with_identity else message.longitudinalPlanSP)
        yield pending[0], decision
        yield from ((buffered, None) for buffered in pending[1:])
        pending = []
    else:
      yield message, None
  yield from ((buffered, None) for buffered in pending)


def recorded_sp_targets(plan_sp, previous_accel, cruise_speed):
  source = str(plan_sp.longitudinalPlanSource)
  if source == 'cruise':
    return cruise_speed, previous_accel
  elif source == 'sccVision':
    target_accel = plan_sp.smartCruiseControl.vision.aTarget
  elif source == 'sccMap':
    target_accel = plan_sp.smartCruiseControl.map.aTarget
  elif source == 'speedLimitAssist':
    target_accel = plan_sp.speedLimit.assist.aTarget
  else:
    raise ValueError(f'Unsupported recorded SP source: {source}')
  return plan_sp.vTarget, target_accel


def selected_mode(unified=False, supervised=False):
  if unified and supervised:
    raise ValueError('Unified and supervised replay modes are mutually exclusive')
  return 2 if unified else 1 if supervised else 0


def component_mode(component):
  actual = getattr(component, 'longitudinal_mode', None)
  if actual is not None:
    if not isinstance(actual, int) or isinstance(actual, bool) or actual not in (0, 1, 2):
      raise ValueError('Runtime component returned an invalid longitudinal mode')
    return int(actual)
  return selected_mode(bool(getattr(component, 'unified', False)), bool(getattr(component, 'supervised', False)))


def component_options(component, unified, supervised=False):
  """Pass historical mode switches; automatic platforms are checked after construction.

  The absence of a constructor flag says nothing about the resulting runtime
  mode. In particular, it must neither reject automatic UNIFIED nor relabel a
  requested LEGACY/SUPERVISED replay as UNIFIED.
  """
  selected_mode(unified, supervised)
  parameters = inspect.signature(component).parameters
  options = {}
  for name, enabled in (('unified', unified), ('supervised', supervised)):
    if name in parameters:
      options[name] = enabled
    elif enabled and name == 'supervised':
      raise ValueError(f'{component.__name__} does not support retired supervised replay; select its historical source tree')
  return options


def require_component_mode(component, mode, name):
  actual_mode = component_mode(component)
  if actual_mode != mode or int(getattr(component, 'mode', actual_mode)) != mode:
    raise ValueError(f'Requested replay mode {mode} is not enabled on the actual {name} (runtime mode {actual_mode}); '
                     'select a source tree that implements the requested mode')
  return actual_mode


def controller_mode_protocols(controller):
  """Inspect consumer inputs and runtime identity, independent of rollout flags.

  Early historical consumers exposed only a unified constructor switch and
  version-1 intent. The explicit longitudinal_mode identity distinguishes the
  mode-bearing protocol; a current automatic constructor needs no switch.
  """
  parameters = inspect.signature(controller.update).parameters
  mode_protocol = ('intent' in parameters or 'unified' in inspect.signature(type(controller)).parameters)
  planning_mode_protocol = mode_protocol and hasattr(controller, 'longitudinal_mode')
  return mode_protocol, planning_mode_protocol


def solver_status_supported(component, unified):
  supported = 'solver_status' in inspect.signature(component.update).parameters
  if unified and not supported:
    raise ValueError('Unified LongControl must explicitly accept solver_status')
  return supported


def frozen_sp_publish(instance, state, publisher, parent_plan_mono_time=0, *, legacy_acceleration=None):
  return None


def publish_plan(planner, state, plan_mono):
  published = []

  def send(service, message):
    if service == 'longitudinalPlan':
      published.append(message.as_reader())

  create_message = planner_module.messaging.new_message

  def scheduled_message(service, *arguments, **options):
    options['logMonoTime'] = int(plan_mono)
    return create_message(service, *arguments, **options)

  with patch.object(planner_module.messaging, 'new_message', new=scheduled_message):
    planner.publish(state, SimpleNamespace(send=send))
  if len(published) != 1 or published[0].longitudinalPlan.modelMonoTime != state.logMonoTime['modelV2']:
    raise ValueError('Planner did not publish exactly one matching model identity')
  return published[0]


def decode_motion_intent(plan):
  from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent

  return MotionIntent.from_message(plan.motionIntent)


def initialization_only(raw_can):
  return bool(raw_can) and all(address == 0x18dab0f1 and bus == 1 and data in INITIALIZATION_UDS
                              for address, data, bus in raw_can)


def motion_diagnostics(longitudinal, intent, target_accel, plan_mono, model_mono, active, mode):
  supervisor = longitudinal.motion_supervisor
  if component_mode(longitudinal) == 0 or supervisor is None:
    raise ValueError('Motion replay fell back to legacy LongControl')
  waiting, constrained = motion_constraints(longitudinal)
  consumed = active and not constrained
  if consumed and (supervisor._last_plan_time_ns != plan_mono or supervisor._last_model_time_ns != model_mono
                   or supervisor._last_intent != intent or supervisor._last_target != target_accel):
    raise ValueError('Unified controller did not consume the supplied intent identity')
  return {
    'resolved_stop': bool(longitudinal.resolved_should_stop), 'intent_version': intent.version if intent is not None else 0,
    'motion_phase': int(longitudinal.motion_phase), 'motion_degraded': bool(longitudinal.motion_degraded),
    'motion_waiting_for_fresh_input': waiting, 'motion_input_constrained': constrained,
    'motion_invalid_reason': bool(longitudinal.motion_reasons & motion_runtime.StopReason.INVALID)
    if hasattr(longitudinal, 'motion_reasons') else -1,
    'motion_active': bool(active), 'motion_consumed': bool(consumed), 'model_mono': model_mono,
    'intent_ready': getattr(intent, 'ready', -1), 'intent_valid': bool(intent.valid) if intent is not None else -1,
    'intent_mode': int(getattr(intent, 'mode', 0)), 'controller_mode': mode,
  }


def motion_constraints(longitudinal):
  degraded = longitudinal.motion_degraded
  waiting = getattr(longitudinal, 'motion_waiting_for_fresh_input', False)
  constrained = getattr(longitudinal, 'motion_input_constrained', degraded)
  if (any(type(value) is not bool for value in (degraded, waiting, constrained))
      or constrained != (degraded or waiting) or degraded and waiting):
    raise ValueError('Motion wait/fault/constraint properties contradict their declared contract')
  return waiting, constrained


def waiting_input_evidence(longitudinal, active, state, target, limits, options):
  """Read eligibility before update; never advance/reset the consumer clock.

  This records the bounded rearm predicate for both motion modes separately from the
  returned waiting flag. Actual fault outputs remain faults in the comparator.
  """
  absent = {'motion_waiting_eligible': False, 'motion_waiting_brake_ceiling': 0.0}
  if (not active or component_mode(longitudinal) not in (1, 2)
      or not hasattr(longitudinal, 'motion_waiting_for_fresh_input')):
    return absent
  supervisor = longitudinal.motion_supervisor
  if (supervisor._require_new_input is not True or supervisor._rearm_after_fault is not False
      or options.get('plan_valid') is not True or options.get('solver_status') != 0
      or state.gasPressed or state.brakePressed):
    return absent
  values = (state.vEgo, state.aEgo, target, longitudinal.last_output_accel, longitudinal.CP.stopAccel)
  if any(value is None or not np.isfinite(value) for value in values):
    return absent
  if (limits is None or len(limits) != 2 or not all(np.isfinite(value) for value in limits)
      or not CarControllerParams.BOSCH_ACCEL_MIN <= limits[0] <= limits[1] <= CarControllerParams.BOSCH_ACCEL_MAX
      or limits[0] > motion_runtime.DEGRADED_ACCEL
      or not CarControllerParams.BOSCH_ACCEL_MIN <= longitudinal.CP.stopAccel <= motion_runtime.DEGRADED_ACCEL):
    return absent
  now, plan, model = (options.get(name) for name in ('now_ns', 'plan_time_ns', 'model_time_ns'))
  if not all(supervisor._timestamp_valid(value) for value in (now, plan, model)):
    return absent
  observed = supervisor._last_creep_observation_ns
  previous = supervisor._last_now_ns
  if (observed is not None and now <= observed
      or previous is not None and not 0 < now - previous <= motion_runtime.MAX_CONTROL_GAP_NS):
    return absent
  latest_plan, latest_model = supervisor._latest_input_ids()
  if latest_plan == 0 or plan > latest_plan and model > latest_model:
    return absent
  if not supervisor._valid_input(options['intent'], target, plan, model, now):
    return absent
  ceiling = np.clip(min(longitudinal.last_output_accel, target, motion_runtime.DEGRADED_ACCEL), *limits)
  return {'motion_waiting_eligible': True, 'motion_waiting_brake_ceiling': float(np.float32(ceiling))}


def apply_motion_feedback(state, longitudinal):
  if type(MOTION_PROTOCOL_VERSION) is not int or MOTION_PROTOCOL_VERSION <= 0:
    raise ValueError('Motion feedback requires the tested tree\'s semantic intent version')
  controls = state['controlsState'].as_builder()
  mode = component_mode(longitudinal)
  controls.motionState.version = MOTION_PROTOCOL_VERSION
  controls.motionState.enabled = mode != 0
  controls.motionState.planningMode = mode
  state['controlsState'] = controls.as_reader()


def planner_update_options(planner, plan_mono):
  """Use the recording's clock only where the tested public API declares it."""
  parameters = inspect.signature(planner.update).parameters
  if 'now_ns' not in parameters:
    return {}
  if parameters['now_ns'].kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
    raise ValueError('Planner now_ns must accept an explicit recorded clock keyword')
  if type(plan_mono) is not int or not 0 < plan_mono < 2 ** 64:
    raise ValueError('Recorded planner clock must be an exact positive uint64 ID')
  return {'now_ns': plan_mono}


class ExecutionContract:
  SERVICES = {'planner': 'longitudinalPlan', 'longcontrol': 'carControl', 'carcontroller': 'sendcan'}

  def __init__(self, contract, window):
    self.contract = contract
    requested = round((window['origin'] + window.get('execution_start', window.get('warmup', window['start'] - 60.0))) * 1e9)
    self.boundaries = dict.fromkeys(self.SERVICES, requested)
    self.ends = dict.fromkeys(self.SERVICES, 2 ** 64 - 1)
    self.completed = dict.fromkeys(self.SERVICES, 0)
    self.dependency_controls = set()
    self.edges = {}
    self.initialization_wait = {}
    self.required_until = {'modelV2': {}, 'longitudinalPlan': {}, 'carControl': {}}
    if contract is None:
      return
    identity = hashlib.sha256(json.dumps({key: value for key, value in contract.items() if key != 'identity_sha256'},
                                         sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    if (contract.get('schema') != 1 or contract.get('policy') != EXECUTION_POLICY or contract.get('status') != 'CLOSED'
        or contract.get('identity_sha256') != identity or contract.get('requested_warmup_mono') != requested):
      raise ValueError('Invalid runtime execution dependency contract')
    self.boundaries = contract['runtime_boundaries_mono']
    self.ends = contract['runtime_ends_mono']
    if set(self.boundaries) != set(self.SERVICES) or set(self.ends) != set(self.SERVICES):
      raise ValueError('Execution contract omits a runtime layer')
    for layer, boundary in self.boundaries.items():
      identifiers = contract['runtime_ids'][layer]
      if (type(boundary) is not int or boundary < 0 or boundary > requested
          or type(self.ends[layer]) is not int or self.ends[layer] < boundary
          or any(type(mono) is not int or not boundary <= mono <= self.ends[layer] for mono in identifiers)
          or identifiers != sorted(set(identifiers))):
        raise ValueError('Execution contract contains invalid runtime IDs or boundaries')
    if self.boundaries['carcontroller'] != requested:
      raise ValueError('Dependency closure must not add unrelated Honda controller prefix cycles')
    self.dependency_controls = set(contract['dependency_control_ids'])
    self.initialization_wait = {item['mono']: item for item in contract.get('initialization_wait', [])}
    self.edges = {
      'planner': {row[0]: row for row in contract['edges']['plan_inputs']},
      'longcontrol': {row[0]: row for row in contract['edges']['control_plan_state']},
      'carcontroller': {row[0]: row for row in contract['edges']['can_control_state']},
    }
    model_index = contract['edges']['plan_inputs_fields'].index('modelV2')
    for service, rows, input_index in (
        ('modelV2', contract['edges']['plan_inputs'], model_index),
        ('longitudinalPlan', contract['edges']['control_plan_state'], 1),
        ('carControl', contract['edges']['can_control_state'], 1)):
      for row in rows:
        identity = row[input_index]
        self.required_until[service][identity] = max(row[0], self.required_until[service].get(identity, 0))

  def executing(self, layer, mono):
    return self.boundaries[layer] <= mono <= self.ends[layer]

  def attempt(self, layer, mono, dependencies=None):
    if self.contract is None:
      return
    position = self.completed[layer]
    expected = self.contract['runtime_ids'][layer]
    if position >= len(expected) or expected[position] != mono:
      raise ValueError(f'Runtime execution ID mismatch or skipped {layer} cycle: {mono}')
    if dependencies is not None and self.edges[layer][mono][1:] != list(dependencies):
      raise ValueError(f'Runtime {layer} consumed a different dependency ID: {mono}')

  def returned(self, layer):
    self.completed[layer] += 1

  def planner_inputs(self, mono, state, decision_mono):
    if self.contract is None:
      return
    expected = self.edges['planner'][mono]
    if expected[1] != decision_mono:
      raise ValueError(f'Planner paired SP dependency identity changed: {mono}')
    for name, identity in zip(self.contract['edges']['plan_inputs_fields'][2:], expected[2:], strict=True):
      if state.logMonoTime.get(name) != identity:
        raise ValueError(f'Planner input dependency identity changed for {name}: {mono}')

  def waiting_for_initial_inputs(self, mono, model_mono, decision_mono, state, measured):
    waiting = self.initialization_wait.get(mono)
    if waiting is None:
      return False
    if (measured or state is not None or model_mono != waiting['model_mono'] or decision_mono != waiting['sp_mono']):
      raise ValueError('Initialization waiting conditions differ from the exact dependency contract')
    return True

  def evict(self, history, maximum, service, now_mono):
    while len(history) > maximum:
      removable = next((mono for mono in history if self.required_until[service].get(mono, 0) < now_mono), None)
      if removable is None:
        break
      del history[removable]

  def finish(self):
    if self.contract is not None and any(self.completed[layer] != len(values)
                                         for layer, values in self.contract['runtime_ids'].items()):
      raise ValueError('Incomplete runtime execution dependency closure')

  def receipt(self):
    return {'identity_sha256': self.contract['identity_sha256'] if self.contract is not None else None,
            'runtime_boundaries_mono': self.boundaries, 'runtime_ends_mono': self.ends,
            'completed_runtime_counts': dict(self.completed)}


class ReplayConsumerHealth:
  def __init__(self):
    from openpilot.cereal.messaging import FrequencyTracker
    from openpilot.cereal.services import SERVICE_LIST

    update_frequency = SERVICE_LIST['selfdriveState'].frequency
    self.frequencies = {name: SERVICE_LIST[name].frequency for name in ('carState', 'longitudinalPlan')}
    self.trackers = {name: FrequencyTracker(frequency, update_frequency, False) for name, frequency in self.frequencies.items()}
    self.received = {}
    self.identities = {}

  def observe(self, cycle_mono, car_state_mono, generated_plan_mono):
    for service, mono in (('carState', car_state_mono), ('longitudinalPlan', generated_plan_mono)):
      if mono is not None and mono > 0 and self.identities.get(service) != mono:
        self.trackers[service].record_recv_time(cycle_mono / 1e9)
        self.received[service] = cycle_mono
        self.identities[service] = mono

  def at(self, now_mono):
    return {service: {
      'seen': service in self.received,
      'alive': service in self.received and 0 <= now_mono - self.received[service] < 10.0 / frequency * 1e9,
      'frequency_ok': self.trackers[service].valid,
    } for service, frequency in self.frequencies.items()}


class ReplayDiagnostics:
  LAYERS = {'planner': 'plan_mono', 'longcontrol': 'control_mono', 'carcontroller': 'controller_mono'}
  REGIONS = ('read_only_prefix', 'dependency_prefix', 'warmup', 'score', 'after_score')
  MAX_MISSING_SAMPLES = 32
  MAX_SAMPLES_PER_REGION = MAX_MISSING_SAMPLES // len(REGIONS)
  MAX_SAMPLES_PER_REASON_REGION = 2

  def __init__(self, destination):
    self.data = destination
    self.intent_input_hash = None
    self.sample_region_counts = dict.fromkeys(self.REGIONS, 0)
    self.data.clear()
    self.data.update(
      diagnostic_schema=DIAGNOSTIC_SCHEMA_VERSION, native_supported=True, status='STARTING', initialized=False,
      current=None, execution_boundary_mono=None, score_interval_mono=None,
      counts=None, first_attempted=None, first_completed=None, first_executed=None,
      first_executed_semantics='first_completed_update_return_not_first_attempt',
      coverage=None, coverage_scope='score_only_not_runtime_call_counts', read_only_prefix=None,
      missing=None, missing_dependencies=None, resets=None, solver_failures=None,
      solver_failures_scope='score_plan_solution_status_not_solver_invocations',
      solver_invocations={'known': False, 'attempted': None, 'completed': None,
                          'attempted_known': False, 'completed_known': False, 'reset_known': False, 'reset_calls': None,
                          'scope': 'counter_deltas_within_planner_update_initial_snapshot_separate',
                          'identity_semantics': 'enclosing_plan_publication_id_not_solver_start_timestamp',
                          'first_attempted_mono': None, 'first_completed_mono': None,
                          'initial_snapshot': None, 'last_observation': None, 'returned_statuses': None,
                          'reason': 'solver_call_not_instrumented_solution_status_is_not_invocation_evidence'},
      capabilities=None, motion_replay=None, initialization_sends=None,
      components=None, error=None)

  @classmethod
  def counts(cls):
    return dict.fromkeys(('total', *cls.REGIONS), 0)

  def initialize(self, window, warmup_start, *, coverage, missing, resets, capabilities, motion, first_executed,
                 read_only_prefix, initialization_sends):
    self.data.update(
      status='RUNNING', initialized=True, execution_boundary_mono=round((window['origin'] + warmup_start) * 1e9),
      score_interval_mono=[round((window['origin'] + window[name]) * 1e9) for name in ('start', 'end')],
      counts={layer: {kind: self.counts() for kind in ('attempted', 'completed')} for layer in self.LAYERS},
      first_attempted=dict.fromkeys(self.LAYERS.values()), first_completed=dict.fromkeys(self.LAYERS.values()),
      first_executed=first_executed, coverage=coverage, missing=missing, resets=resets, capabilities=capabilities,
      motion_replay=motion, read_only_prefix=read_only_prefix, initialization_sends=initialization_sends,
      missing_dependencies={'counts': {}, 'samples': [], 'sample_limit': self.MAX_MISSING_SAMPLES,
                            'samples_per_region': self.MAX_SAMPLES_PER_REGION,
                            'samples_per_reason_region': self.MAX_SAMPLES_PER_REASON_REGION,
                            'samples_omitted': 0, 'latest': None},
      components=dict.fromkeys(self.LAYERS, False), solver_failures=0)

  def observe(self, service, mono, timestamp, window, warmup_start):
    boundaries = self.data.get('runtime_execution_boundaries_mono')
    layer = next((name for name, runtime_service in ExecutionContract.SERVICES.items() if runtime_service == service), None)
    dependency = (timestamp < warmup_start and layer is not None and boundaries is not None and mono >= boundaries[layer])
    region = ('dependency_prefix' if dependency else 'read_only_prefix' if timestamp < warmup_start else 'warmup' if timestamp < window['start']
              else 'score' if timestamp <= window['end'] else 'after_score')
    self.data['current'] = {'service': service, 'mono': int(mono), 'region': region,
                            'in_score': window['start'] <= timestamp <= window['end']}

  def call(self, layer, kind):
    current = self.data['current']
    counts = self.data['counts'][layer][kind]
    counts['total'] += 1
    counts[current['region']] += 1
    first = self.data['first_' + kind]
    if first[self.LAYERS[layer]] is None:
      first[self.LAYERS[layer]] = current['mono']

  @staticmethod
  def solver_snapshot(planner):
    try:
      component = getattr(planner, 'mpc', None)
      solve_calls = getattr(component, 'diagnostic_solve_calls', None)
      reset_calls = getattr(component, 'diagnostic_reset_calls', None)
      if type(solve_calls) is not int or solve_calls < 0 or type(reset_calls) is not int or reset_calls < 0:
        return None
      absent = object()
      last_return = getattr(component, 'diagnostic_last_solver_return', absent)
      return_known = last_return is None or type(last_return) is int
      return component, {'solve_calls': solve_calls, 'reset_calls': reset_calls, 'last_return_known': return_known,
                          'last_return': last_return if return_known else None}
    except Exception:
      return None

  def initialize_solver(self, planner):
    snapshot = self.solver_snapshot(planner)
    solver = self.data['solver_invocations']
    if snapshot is None:
      self.unknown_solver('missing_or_invalid_actual_solver_counters')
    elif solver['initial_snapshot'] is None and solver.get('observation_gap') is None:
      solver.update(
        known=snapshot[1]['last_return_known'], attempted_known=True, completed_known=snapshot[1]['last_return_known'],
        reset_known=True, attempted=self.counts(), completed=self.counts() if snapshot[1]['last_return_known'] else None,
        reset_calls=self.counts(), initial_snapshot=dict(self.data['current'], **snapshot[1]),
        returned_statuses={} if snapshot[1]['last_return_known'] else None,
        reason=None if snapshot[1]['last_return_known'] else 'actual_call_counters_present_but_return_marker_unknown')

  def unknown_solver(self, reason):
    solver = self.data['solver_invocations']
    if solver['attempted_known'] and solver.get('observed_before_gap') is None:
      solver['observed_before_gap'] = {name: solver[name] for name in ('attempted', 'completed', 'reset_calls', 'returned_statuses')}
    solver.update(known=False, attempted_known=False, completed_known=False, reset_known=False,
                  attempted=None, completed=None, reset_calls=None, returned_statuses=None,
                  reason=reason, observation_gap=self.data['current'])

  def observe_solver(self, planner, before):
    solver = self.data['solver_invocations']
    if not solver['attempted_known']:
      return
    after = self.solver_snapshot(planner)
    current = self.data['current']
    solver['last_observation'] = dict(current, before=before[1] if before is not None else None,
                                      after=after[1] if after is not None else None,
                                      same_component=before[0] is after[0] if before is not None and after is not None else None)
    if before is None or after is None or before[0] is not after[0]:
      self.unknown_solver('solver_counter_source_missing_or_replaced_during_planner_update')
      return
    solve_delta = after[1]['solve_calls'] - before[1]['solve_calls']
    reset_delta = after[1]['reset_calls'] - before[1]['reset_calls']
    solver['last_observation'].update(solve_delta=solve_delta, reset_delta=reset_delta)
    if solve_delta < 0 or reset_delta < 0:
      self.unknown_solver('actual_solver_counters_moved_backwards')
      return
    for name, delta in (('attempted', solve_delta), ('reset_calls', reset_delta)):
      solver[name]['total'] += delta
      solver[name][current['region']] += delta
    if solve_delta and solver['first_attempted_mono'] is None:
      solver['first_attempted_mono'] = current['mono']
    if solve_delta > 1 or not after[1]['last_return_known']:
      solver.update(known=False, completed_known=False, completed=None, returned_statuses=None,
                    reason='multiple_calls_or_missing_return_marker_cannot_prove_every_solver_return')
    if solve_delta == 1 and after[1]['last_return_known'] and after[1]['last_return'] is not None:
      if solver['first_completed_mono'] is None:
        solver['first_completed_mono'] = current['mono']
      if solver['completed_known']:
        solver['completed']['total'] += 1
        solver['completed'][current['region']] += 1
        status = str(after[1]['last_return'])
        solver['returned_statuses'][status] = solver['returned_statuses'].get(status, 0) + 1

  def unavailable(self, reason, required):
    if reason == 'plan_warmup':
      return
    current = self.data['current']
    dependencies = self.data['missing_dependencies']
    counts = dependencies['counts'].setdefault(reason, self.counts())
    counts['total'] += 1
    counts[current['region']] += 1
    sample = dict(current, reason=reason, required=required,
                  execution_boundary_mono=self.data['execution_boundary_mono'],
                  score_interval_mono=self.data['score_interval_mono'])
    dependencies['latest'] = sample
    if (counts[current['region']] <= self.MAX_SAMPLES_PER_REASON_REGION
        and self.sample_region_counts[current['region']] < self.MAX_SAMPLES_PER_REGION):
      dependencies['samples'].append(sample)
      self.sample_region_counts[current['region']] += 1
    else:
      dependencies['samples_omitted'] += 1

  def failed(self, error):
    dependency = getattr(error, 'replay_dependency', None)
    if dependency is not None and self.data['current'] is not None:
      self.unavailable(dependency['reason'], dependency['required'])
    self.data['status'] = 'ERROR'
    self.data['error'] = f'{type(error).__name__}: {error}'


def replay_window(cache, window, *, recorded_plan=False, messages=None, row_sink=None, strict=True, unified=False,
                  supervised=False, planner_inputs=None, diagnostics=None, execution_contract=None, dependency_sink=None,
                  execution_observer=None):
  progress = ReplayDiagnostics({} if diagnostics is None else diagnostics)
  try:
    result = _replay_window(cache, window, recorded_plan=recorded_plan, messages=messages, row_sink=row_sink,
                            strict=strict, unified=unified, supervised=supervised, planner_inputs=planner_inputs, progress=progress,
                            execution_contract=execution_contract, dependency_sink=dependency_sink, execution_observer=execution_observer)
    progress.data['status'] = 'COMPLETED'
    result['diagnostics'] = progress.data
    return result
  except Exception as error:
    progress.failed(error)
    raise
  finally:
    motion = progress.data['motion_replay']
    if motion is not None:
      motion['consumer_input_sha256'] = progress.intent_input_hash.hexdigest() if unified or supervised else None
      for name in ('generated_intent_versions', 'accepted_intent_versions', 'phase_counts', 'generated_intent_modes', 'accepted_intent_modes'):
        motion[name] = dict(motion[name])


def _replay_window(cache, window, *, recorded_plan, messages, row_sink, strict, unified, supervised, planner_inputs, progress,
                   execution_contract, dependency_sink, execution_observer):
  mode = selected_mode(unified, supervised)
  motion_enabled = mode != 0
  if motion_enabled and recorded_plan:
    raise ValueError('Motion replay must execute its own planner intent and targets')
  if messages is None:
    messages = LogReader([str(cache)], sort_by_time=True)
  latest, valid, stamps, models, generated = {}, {}, {}, {}, {}
  planner = controller = longitudinal = None
  consumer_health = None
  params = params_sp = None
  pending_control = controller_input = None
  control_outputs = {}
  first_controller_frame = True
  current = {}
  plan_rows, can_rows, control_rows, dependency_outputs = [], [], [], []
  coverage = {'plans': 0, 'matched_plans': 0, 'controls': 0, 'matched_controls': 0, 'sendcan': 0, 'matched_sendcan': 0}
  missing = {}
  resets = []
  previous_timestamp = None
  controller_skipped = True
  solver_failures = 0
  initialization_sends = []
  first_executed = dict.fromkeys(('plan_mono', 'control_mono', 'controller_mono'))
  read_only_prefix = dict.fromkeys(('plans', 'controls', 'sendcan'), 0)
  capabilities = {'replay_protocol_version': REPLAY_PROTOCOL_VERSION, 'controller_solver_status': None,
                  'controller_mode_protocol': None, 'controller_readiness': None, 'actual_plan_publication': True,
                  'controller_planning_mode_protocol': None, 'requested_mode': mode, 'planner_mode': None, 'controller_mode': None,
                  'controller_braking_health_protocol': None, 'controller_owner_protocol': None,
                  'motion_intent_version': MOTION_PROTOCOL_VERSION, 'planner_clock': None,
                  'controller_waiting_protocol': None, 'controller_standby_input_protocol': None,
                  'shared_authority_helper': restrict_longitudinal_authority is not None,
                  'planner_inputs': 'exact_provider' if planner_inputs is not None else 'model_publication_snapshot'}
  motion = {'enabled': motion_enabled, 'mode': mode, 'generated_intent_versions': Counter(), 'accepted_intent_versions': Counter(),
            'generated_intent_modes': Counter(), 'accepted_intent_modes': Counter(),
            'phase_counts': Counter(), 'control_calls': 0, 'accepted_controls': 0, 'disabled_controls': 0,
            'degraded_controls': 0, 'fallback_controls': 0, 'inactive_no_plan_controls': 0, 'standby_controls': 0,
            'waiting_controls': 0, 'constrained_controls': 0,
            'unused_legacy_plan_fields': ['cruise_accel'] if unified else [],
            'software_feedback': 'actual_consumer_mode_ack_only_other_recorded_controlsState_fields_unchanged' if motion_enabled else None}
  intent_input_hash = hashlib.sha256()
  progress.intent_input_hash = intent_input_hash
  warmup_start = window.get('execution_start', window.get('warmup', window['start'] - 60.0))
  execution = ExecutionContract(execution_contract, window)
  progress.initialize(window, warmup_start, coverage=coverage, missing=missing, resets=resets, capabilities=capabilities,
                      motion=motion, first_executed=first_executed, read_only_prefix=read_only_prefix,
                      initialization_sends=initialization_sends)
  progress.data['runtime_execution_boundaries_mono'] = execution.boundaries
  progress.data['runtime_execution_ends_mono'] = execution.ends
  progress.data['requested_warmup_mono'] = round((window['origin'] + warmup_start) * 1e9)
  progress.data['execution_boundary_mono'] = min(execution.boundaries.values())
  progress.data['execution_closure'] = {'identity_sha256': execution_contract['identity_sha256'] if execution_contract else None,
                                      'completed_runtime_counts': execution.completed}
  progress.data['initialization_wait'] = []
  control_sp = structs.CarControlSP()
  state_wrapper = SimpleNamespace(v_cruise_factor=1.0, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={})
  target_type = sp_target_return_type(LongitudinalPlannerSP.update_targets)
  capabilities['sp_target_return_contract'] = target_type.__name__ if target_type is not None else 'historical_tuple'

  def emit(kind, row):
    if row_sink is not None:
      row_sink(kind, row)
    else:
      {'plan': plan_rows, 'control': control_rows, 'can': can_rows}[kind].append(row)

  def unavailable(reason, measured, required):
    if measured:
      missing[reason] = missing.get(reason, 0) + 1
    progress.unavailable(reason, required)

  def update_targets(instance, state, v_ego, a_ego, v_cruise):
    speed, acceleration = recorded_sp_targets(current['sp'], a_ego, v_cruise)
    return frozen_sp_target(instance, current['sp'].longitudinalPlanSource, speed, acceleration, target_type)

  def is_e2e(instance, state):
    dec = current['sp'].dec
    return state['selfdriveState'].experimentalMode and (not dec.active or str(dec.state) == 'blended')

  def frozen_sp_update(instance, state):
    """A non-recording stub must not accumulate every full-route input like Mock does."""
    return None

  with patch.object(LongitudinalPlannerSP, 'update', new=frozen_sp_update), \
       patch.object(LongitudinalPlannerSP, 'publish_longitudinal_plan_sp', new=frozen_sp_publish), \
       patch.object(LongitudinalPlannerSP, 'update_targets', new=update_targets), \
       patch.object(planner_module.LongitudinalPlanner, 'is_e2e', new=is_e2e):
    for message, decision_record in paired_messages(messages, with_identity=True):
      decision = decision_record.value if decision_record is not None else None
      decision_mono = decision_record.mono if decision_record is not None else None
      service = message.which()
      timestamp = message.logMonoTime / 1e9 - window['origin']
      measured = window['start'] <= timestamp <= window['end']
      progress.observe(service, message.logMonoTime, timestamp, window, warmup_start)
      if execution_observer is not None:
        execution_observer.observe(message, decision)
      if service not in PARAMETERS:
        if previous_timestamp is not None and timestamp < previous_timestamp:
          raise ValueError(f'Nonmonotonic replay input at {timestamp:.9f}')
        if not strict and previous_timestamp is not None and timestamp - previous_timestamp > 0.5:
          resets.append({'before': previous_timestamp, 'after': timestamp})
          latest, valid, stamps, models, generated, control_outputs = {}, {}, {}, {}, {}, {}
          planner = controller = longitudinal = pending_control = controller_input = None
          progress.data['components'] = dict.fromkeys(progress.LAYERS, False)
          first_controller_frame = controller_skipped = True
        previous_timestamp = timestamp
      if service == 'carParams':
        params = message.carParams
      elif service == 'carParamsSP':
        params_sp = message.carParamsSP
      if planner is None and params is not None and params_sp is not None:
        if params.carFingerprint != CAR.HONDA_ACCORD or not params.openpilotLongitudinalControl:
          raise ValueError('This replay requires Accord openpilot longitudinal')
        planner = planner_module.LongitudinalPlanner(params, params_sp,
                                                     **component_options(planner_module.LongitudinalPlanner, unified, supervised))
        capabilities['planner_mode'] = require_component_mode(planner, mode, 'planner')
        progress.data['components']['planner'] = True
        progress.initialize_solver(planner)
        controller_params_sp = structs.CarParamsSP()
        for name in ('flags', 'safetyParam', 'enableGasInterceptor'):
          setattr(controller_params_sp, name, getattr(params_sp, name))
        controller = CarController({Bus.pt: DBC[params.carFingerprint][Bus.pt]}, params, controller_params_sp)
        progress.data['components']['carcontroller'] = True
        longitudinal = LongControl(params, controller_params_sp, **component_options(LongControl, unified, supervised))
        capabilities['controller_mode'] = require_component_mode(longitudinal, mode, 'controller')
        progress.data['components']['longcontrol'] = True
        capabilities['controller_solver_status'] = solver_status_supported(LongControl, motion_enabled)
        capabilities['controller_mode_protocol'], capabilities['controller_planning_mode_protocol'] = controller_mode_protocols(longitudinal)
        capabilities['controller_readiness'] = capabilities['controller_mode_protocol'] and hasattr(longitudinal, 'can_standby')
        capabilities['controller_braking_health_protocol'] = 'braking_plan_valid' in inspect.signature(LongControl.update).parameters
        capabilities['controller_owner_protocol'] = hasattr(longitudinal, 'motion_owner_established')
        capabilities['controller_waiting_protocol'] = (hasattr(longitudinal, 'motion_waiting_for_fresh_input')
                                                       and hasattr(longitudinal, 'motion_input_constrained'))
        capabilities['controller_standby_input_protocol'] = 'standby_input' in inspect.signature(LongControl.update).parameters
        if capabilities['controller_waiting_protocol'] and not capabilities['controller_standby_input_protocol']:
          raise ValueError('Waiting-aware LongControl must explicitly accept standby_input')
        capabilities['planner_clock'] = (
          'recorded_plan_publication_mono_proxy_not_recorded_update_entry_time'
          if 'now_ns' in inspect.signature(planner.update).parameters else 'legacy_public_API_without_clock_argument')
        if motion_enabled and (type(MOTION_PROTOCOL_VERSION) is not int or MOTION_PROTOCOL_VERSION <= 0):
          raise ValueError('Motion replay lacks a tested semantic intent version')
        if motion_enabled and capabilities['controller_braking_health_protocol']:
          consumer_health = ReplayConsumerHealth()
          capabilities['braking_health_method'] = 'native_FrequencyTracker_on_recorded_consumer_ID_updates_without_publication_validity'
          capabilities['braking_health_clock'] = 'controlsState_publication_proxy_for_SubMaster_receive_time_not_recorded_IPC_arrival'
        parser = CANParser(DBC[params.carFingerprint][Bus.pt], [('ACC_CONTROL', 50)], CanBus(params).pt)
        recorded_parser = CANParser(DBC[params.carFingerprint][Bus.pt], [('ACC_CONTROL', 50)], CanBus(params).pt)
      if service in SERVICES:
        latest[service] = getattr(message, service)
        valid[service] = bool(message.valid)
        stamps[service] = message.logMonoTime
      if service == 'modelV2' and all(name in latest and stamps[name] > 0 for name in SERVICES):
        state = ReplayState(latest.copy())
        state.logMonoTime = stamps.copy()
        state.valid = {name: valid[name] and 0 <= (message.logMonoTime - stamps[name]) / 1e9 <= 0.3 for name in SERVICES}
        models[message.logMonoTime] = state
        execution.evict(models, 30, 'modelV2', message.logMonoTime)
      elif service == 'longitudinalPlan':
        coverage['plans'] += int(measured)
        executing = execution.executing('planner', message.logMonoTime)
        read_only_prefix['plans'] += int(message.logMonoTime < execution.boundaries['planner'])
        plan = message.longitudinalPlan
        state = models.get(plan.modelMonoTime)
        plan_time = timestamp
        if execution.waiting_for_initial_inputs(int(message.logMonoTime), int(plan.modelMonoTime), decision_mono, state, measured):
          progress.data['initialization_wait'].append(execution.initialization_wait[int(message.logMonoTime)])
          continue
        if planner_inputs is not None and planner is not None and decision is not None and executing:
          state = planner_inputs(int(message.logMonoTime), ReplayState({}))
        if state is not None and planner is not None and decision is not None and executing:
          current['sp'] = decision
          if motion_enabled or capabilities['controller_planning_mode_protocol']:
            state.runtime_motion_feedback = lambda current_state, controller=longitudinal: apply_motion_feedback(current_state, controller)
            state.runtime_motion_feedback(state)
          solver_before = progress.solver_snapshot(planner)
          execution.attempt('planner', int(message.logMonoTime))
          execution.planner_inputs(int(message.logMonoTime), state, decision_mono)
          progress.call('planner', 'attempted')
          try:
            planner.update(state, **planner_update_options(planner, int(message.logMonoTime)))
          finally:
            progress.observe_solver(planner, solver_before)
          progress.call('planner', 'completed')
          execution.returned('planner')
          if first_executed['plan_mono'] is None:
            first_executed['plan_mono'] = int(message.logMonoTime)
          publication = publish_plan(planner, state, message.logMonoTime)
          published = publication.longitudinalPlan
          if motion_enabled and (not hasattr(published, 'solverStatus') or published.solverStatus != planner.mpc.solution_status):
            raise ValueError('Unified publisher did not publish its actual solver status')
          if motion_enabled and capabilities['controller_readiness'] and not hasattr(published.motionIntent, 'ready'):
            raise ValueError('Readiness-aware controller requires published motion readiness')
          replayed_accel = float(published.aTarget)
          replayed_stop = bool(published.shouldStop)
          selected_plan = plan if recorded_plan else published
          target_accel, target_stop = float(selected_plan.aTarget), bool(selected_plan.shouldStop)
          intent = decode_motion_intent(selected_plan) if motion_enabled else None
          model_mono = int(state.logMonoTime['modelV2'])
          plan_valid = bool(message.valid if recorded_plan else publication.valid)
          solver_status = int(getattr(selected_plan, 'solverStatus', 0))
          intent_version = int(getattr(getattr(selected_plan, 'motionIntent', None), 'version', 0))
          intent_mode = int(getattr(getattr(selected_plan, 'motionIntent', None), 'mode', 0))
          generated[message.logMonoTime] = SimpleNamespace(
            accel=target_accel, stop=target_stop, intent=intent, model_mono=model_mono,
            valid=plan_valid, solver_status=solver_status, intent_version=intent_version, intent_mode=intent_mode)
          if window['start'] <= plan_time <= window['end']:
            coverage['matched_plans'] += 1
            solver_failures += int(planner.mpc.solution_status != 0)
            progress.data['solver_failures'] = solver_failures
            if motion_enabled:
              motion['generated_intent_versions'][str(intent.version)] += 1
              motion['generated_intent_modes'][str(intent_mode)] += 1
            emit('plan', {'t': plan_time, 'mono': message.logMonoTime, 'accel': replayed_accel,
                              'stop': replayed_stop, 'recorded_accel': plan.aTarget, 'recorded_stop': plan.shouldStop,
                              'speed': float(planner.v_desired_trajectory[0]), 'recorded_speed': plan.speeds[0],
                              'solver_status': int(planner.mpc.solution_status), 'source': str(planner.mpc.source),
                              'cruise_accel': float(planner.a_cruise), 'v_ego': state['carState'].vEgo,
                              'lead': bool(state['radarState'].leadOne.present or state['radarState'].leadTwo.present),
                              'automatic': bool(state['selfdriveState'].enabled and state['carControl'].enabled
                                                and state['carControl'].longActive and not state['carState'].gasPressed
                                                and not state['carState'].brakePressed and not state['carControl'].cruiseControl.override),
                              'valid': bool(message.valid and state.all_checks()), 'model_accel': state['modelV2'].action.desiredAcceleration,
                              'model_stop': bool(state['modelV2'].action.shouldStop), 'fcw': bool(planner.fcw),
                              'force_decel': bool(state['controlsState'].forceDecel), 'allow_throttle': bool(planner.allow_throttle),
                              'sp_cruise': str(decision.longitudinalPlanSource) == 'cruise',
                              'published_valid': bool(publication.valid),
                              'published_solver_status': int(getattr(published, 'solverStatus', -1)),
                              'published_intent_version': int(getattr(getattr(published, 'motionIntent', None), 'version', -1)),
                              'published_mode': int(getattr(getattr(published, 'motionIntent', None), 'mode', 0)),
                              'published_ready': getattr(getattr(published, 'motionIntent', None), 'ready', -1)})
          execution.evict(generated, 100, 'longitudinalPlan', message.logMonoTime)
        else:
          reason = 'plan_parameters' if planner is None else 'plan_model' if state is None else 'plan_sp' if decision is None else 'plan_warmup'
          required = ([{'service': name, 'mono': None} for name in ('carParams', 'carParamsSP')] if planner is None else
                      [{'service': 'modelV2', 'mono': int(plan.modelMonoTime)}] if state is None else
                      [{'service': 'longitudinalPlanSP', 'mono': None}] if decision is None else [])
          unavailable(reason, measured, required)
      elif service == 'controlsState' and 'carState' in latest:
        pending_control = (message.controlsState.longitudinalPlanMonoTime, latest['carState'], valid['carState'],
                           int(message.logMonoTime), int(stamps['carState']))
        if consumer_health is not None:
          consumed_plan = int(message.controlsState.longitudinalPlanMonoTime)
          consumer_health.observe(int(message.logMonoTime), int(stamps['carState']), consumed_plan if consumed_plan in generated else None)
      elif service == 'carControl':
        coverage['controls'] += int(measured)
        if not execution.executing('longcontrol', message.logMonoTime):
          read_only_prefix['controls'] += 1
          pending_control = None
          continue
        if longitudinal is None or pending_control is None:
          required = ([{'service': name, 'mono': None} for name in ('carParams', 'carParamsSP')] if longitudinal is None else
                      [{'service': 'controlsState', 'mono': None}, {'service': 'carState', 'mono': None}])
          unavailable('control_parameters' if longitudinal is None else 'control_state', measured, required)
          continue
        plan_mono, car_state, car_state_valid, controls_state_mono, car_state_mono = pending_control
        pending_control = None
        supplied_plan = generated.get(plan_mono)
        recorded_inactive = (not message.carControl.longActive and message.carControl.actuators.longControlState == LongCtrlState.off
                             and message.carControl.actuators.accel == 0.0)
        inactive_no_plan = supplied_plan is None and recorded_inactive
        if supplied_plan is not None or inactive_no_plan:
          target_accel = supplied_plan.accel if supplied_plan is not None else None
          target_stop = supplied_plan.stop if supplied_plan is not None else None
          intent = supplied_plan.intent if supplied_plan is not None else None
          model_mono = supplied_plan.model_mono if supplied_plan is not None else 0
          solver_status = supplied_plan.solver_status if supplied_plan is not None else 0
          plan_valid = bool(supplied_plan.valid and car_state_valid) if supplied_plan is not None else False
          cruise_set = bool(np.isfinite(car_state.vCruise) and car_state.vCruise != V_CRUISE_UNSET)
          readiness_enabled = motion_enabled and capabilities['controller_readiness']
          can_standby = longitudinal.can_standby if readiness_enabled else False
          if readiness_enabled and type(can_standby) is not bool:
            raise ValueError('LongControl.can_standby must be a boolean readiness property')
          standby = bool(readiness_enabled and intent is not None and not intent.ready and can_standby
                         and solver_status == 0 and plan_valid)
          standby_input_health = plan_valid
          if readiness_enabled:
            plan_valid = plan_valid and cruise_set
          intent_version = supplied_plan.intent_version if supplied_plan is not None else 0
          intent_mode = supplied_plan.intent_mode if supplied_plan is not None else 0
          expected_version = (MOTION_PROTOCOL_VERSION if capabilities['controller_planning_mode_protocol'] else 1) if motion_enabled else 0
          mode_mismatch = bool(supplied_plan is not None and capabilities['controller_mode_protocol']
                               and (intent_version != expected_version
                                    or capabilities['controller_planning_mode_protocol'] and intent_mode != mode))
          owner_established = getattr(longitudinal, 'motion_owner_established', False)
          if type(owner_established) is not bool:
            raise ValueError('LongControl.motion_owner_established must be a boolean property')
          plan_valid = plan_valid and not mode_mismatch
          standby_input = bool(
            capabilities['controller_standby_input_protocol'] and readiness_enabled and can_standby and not mode_mismatch
            and intent is not None and intent.is_standby_valid() and solver_status == 0 and standby_input_health
            and getattr(car_state, 'canValid', False) and np.isfinite(car_state.vEgo) and np.isfinite(car_state.aEgo)
            and car_state.vCruise == V_CRUISE_UNSET)
          health = consumer_health.at(int(message.logMonoTime)) if consumer_health is not None else None
          braking_plan_valid = bool(
            supplied_plan is not None and not mode_mismatch and car_state_valid and getattr(car_state, 'canValid', False)
            and cruise_set and health is not None and all(health['carState'].values()) and all(health['longitudinalPlan'].values()))
          limits = CarInterface.get_pid_accel_limits(params, controller_params_sp, car_state.vEgo, car_state.vCruise / 3.6)
          control_options = {'intent': intent, 'plan_time_ns': int(plan_mono) if supplied_plan is not None else None,
                             'model_time_ns': model_mono if supplied_plan is not None else None,
                             'now_ns': int(message.logMonoTime), 'plan_valid': plan_valid} if motion_enabled else {}
          if capabilities['controller_solver_status'] and supplied_plan is not None:
            control_options['solver_status'] = solver_status
          if motion_enabled and capabilities['controller_braking_health_protocol']:
            control_options['braking_plan_valid'] = braking_plan_valid
          if motion_enabled and capabilities['controller_standby_input_protocol']:
            control_options['standby_input'] = standby_input
          pedal_guard = motion_enabled or (capabilities['controller_solver_status'] and solver_status != 0)
          active = bool(message.carControl.longActive and (not mode_mismatch or owner_established) and not standby
                        and (not pedal_guard or not car_state.gasPressed and not car_state.brakePressed))
          waiting_evidence = (waiting_input_evidence(longitudinal, active, car_state, target_accel, limits, control_options)
                              if motion_enabled and capabilities['controller_waiting_protocol'] else
                              {'motion_waiting_eligible': False, 'motion_waiting_brake_ceiling': 0.0})
          execution.attempt('longcontrol', int(message.logMonoTime),
                            (int(plan_mono), controls_state_mono, car_state_mono, bool(recorded_inactive)))
          progress.call('longcontrol', 'attempted')
          accel = longitudinal.update(active, car_state, target_accel, target_stop, limits, **control_options)
          progress.call('longcontrol', 'completed')
          execution.returned('longcontrol')
          if first_executed['control_mono'] is None:
            first_executed['control_mono'] = int(message.logMonoTime)
          control = message.carControl.as_builder()
          control.longActive = active
          control.actuators.accel = float(accel)
          control.actuators.longControlState = longitudinal.long_control_state
          control.cruiseControl.override = bool(control.enabled and not active)
          resolved_stop = longitudinal.resolved_should_stop if motion_enabled else target_stop
          control.cruiseControl.resume = bool(
            supplied_plan is not None and control.enabled and car_state.cruiseState.standstill and not resolved_stop
            and (not motion_enabled or active and not motion_constraints(longitudinal)[1]) and not mode_mismatch
            and (not capabilities['controller_solver_status'] or solver_status == 0))
          diagnostics = {
            'requested_stop': bool(target_stop) if supplied_plan is not None else -1,
            'resolved_stop': longitudinal.long_control_state == LongCtrlState.stopping,
            'intent_version': intent_version, 'motion_phase': -1, 'motion_degraded': False, 'motion_active': False,
            'motion_waiting_for_fresh_input': False, 'motion_input_constrained': False, 'motion_invalid_reason': False,
            'controller_resume': bool(control.cruiseControl.resume), 'controller_standby_input': standby_input,
            **waiting_evidence,
            'intent_mode': intent_mode, 'controller_mode': mode,
            'motion_owner_established': owner_established,
            'controller_braking_plan_valid': braking_plan_valid if consumer_health is not None else -1,
            'controller_plan_alive': health['longitudinalPlan']['alive'] if health is not None else -1,
            'controller_plan_frequency_ok': health['longitudinalPlan']['frequency_ok'] if health is not None else -1,
            'controller_car_state_health': bool(car_state_valid and getattr(car_state, 'canValid', False)
                                                and all(health['carState'].values())) if health is not None else -1,
            'intent_braking_valid': getattr(intent, 'braking_accel', None) is not None,
            'intent_braking_accel': float(intent.braking_accel) if getattr(intent, 'braking_accel', None) is not None else 0.0,
            'motion_consumed': False, 'model_mono': model_mono,
            'controller_solver_status': solver_status if capabilities['controller_solver_status'] and supplied_plan is not None else -1,
            'controller_plan_valid': bool(plan_valid) if supplied_plan is not None else -1,
            'mode_mismatch': mode_mismatch if supplied_plan is not None else -1, 'inactive_no_plan': inactive_no_plan,
            'intent_ready': -1, 'intent_valid': -1, 'motion_standby': standby,
            'motion_can_standby': can_standby if readiness_enabled else -1, 'controller_cruise_set': cruise_set,
          }
          if motion_enabled:
            diagnostics.update(motion_diagnostics(longitudinal, intent, target_accel, int(plan_mono), model_mono, active, mode))
            if measured:
              motion['control_calls'] += 1
              motion['phase_counts'][str(diagnostics['motion_phase'])] += 1
              motion['disabled_controls'] += int(not active)
              motion['inactive_no_plan_controls'] += int(inactive_no_plan)
              motion['standby_controls'] += int(standby)
              motion['degraded_controls'] += int(diagnostics['motion_degraded'])
              motion['waiting_controls'] += int(diagnostics['motion_waiting_for_fresh_input'])
              motion['constrained_controls'] += int(diagnostics['motion_input_constrained'])
              if diagnostics['motion_consumed']:
                motion['accepted_controls'] += 1
                motion['accepted_intent_versions'][str(intent.version)] += 1
                motion['accepted_intent_modes'][str(intent_mode)] += 1
              intent_identity = None if intent is None else (
                intent.version, intent.reasons, intent.moving_lead, intent.release_allowed, intent.allow_creep, intent.valid,
                getattr(intent, 'ready', None), getattr(intent, 'mode', 0), getattr(intent, 'braking_accel', None))
              identity = (int(message.logMonoTime), int(plan_mono), model_mono, target_accel, target_stop,
                          intent_identity, plan_valid, solver_status if supplied_plan is not None else None, inactive_no_plan)
              intent_input_hash.update((json.dumps(identity, separators=(',', ':'), allow_nan=False) + '\n').encode())
          control_outputs[message.logMonoTime] = control.as_reader()
          control_row = {'t': timestamp, 'mono': int(message.logMonoTime), 'plan_mono': int(plan_mono),
                         'accel': control.actuators.accel, 'recorded_accel': message.carControl.actuators.accel,
                         'state': str(longitudinal.long_control_state), 'recorded_state': str(message.carControl.actuators.longControlState),
                         **diagnostics}
          if measured:
            coverage['matched_controls'] += 1
            emit('control', control_row)
          if message.logMonoTime in execution.dependency_controls:
            if dependency_sink is None:
              dependency_outputs.append(control_row)
            else:
              dependency_sink('control', control_row)
          execution.evict(control_outputs, 100, 'carControl', message.logMonoTime)
        else:
          unavailable('control_plan', measured, [{'service': 'longitudinalPlan', 'mono': int(plan_mono)}])
      elif service == 'carState' and 'carControl' in stamps:
        controller_input = (stamps['carControl'], message.carState, int(message.logMonoTime))
      elif service == 'sendcan':
        executing = execution.executing('carcontroller', message.logMonoTime)
        read_only_prefix['sendcan'] += int(message.logMonoTime < execution.boundaries['carcontroller'])
        raw_can = [(frame.address, frame.dat, frame.src) for frame in message.sendcan]
        if initialization_only(raw_can):
          initialization_sends.append({'mono': int(message.logMonoTime), 'measured': measured,
                                       'payloads': [data.hex() for _, data, _ in raw_can]})
          continue
        has_acc = any(address == 0x1df and bus == 1 for address, _, bus in raw_can)
        coverage['sendcan'] += int(measured and has_acc)
        if not executing:
          controller_input = None
          continue
        if controller_input is None or controller is None:
          required = ([{'service': name, 'mono': None} for name in ('carParams', 'carParamsSP')] if controller is None else
                      [{'service': 'carState', 'mono': None}, {'service': 'carControl', 'mono': None}])
          unavailable('can_parameters' if controller is None else 'can_state', measured and has_acc, required)
          controller_skipped = True
          continue
        control_mono, car_state, car_state_mono = controller_input
        controller_input = None
        if control_mono not in control_outputs:
          unavailable('can_control', measured and has_acc, [{'service': 'carControl', 'mono': int(control_mono)}])
          controller_skipped = True
          continue
        cached_control = control_outputs[control_mono]
        control = (restrict_longitudinal_authority(params, controller_params_sp, car_state, cached_control)
                   if restrict_longitudinal_authority is not None else cached_control)
        if first_controller_frame:
          controller.frame = 0 if has_acc else 1
          first_controller_frame = False
        elif not strict and controller_skipped and (controller.frame % 2 == 0) != has_acc:
          controller.frame += 1
          resets.append({'can_cadence': timestamp})
        controller_skipped = False
        state_wrapper.out = car_state
        execution.attempt('carcontroller', int(message.logMonoTime), (int(control_mono), car_state_mono, bool(has_acc)))
        progress.call('carcontroller', 'attempted')
        _, sends = controller.update(control, control_sp, state_wrapper, message.logMonoTime)
        progress.call('carcontroller', 'completed')
        execution.returned('carcontroller')
        if first_executed['controller_mono'] is None:
          first_executed['controller_mono'] = int(message.logMonoTime)
        actual_acc = any(address == 0x1df for address, _, _ in sends)
        if actual_acc != has_acc:
          raise ValueError(f'Controller cadence mismatch at {timestamp:.6f}')
        if has_acc and measured:
          coverage['matched_sendcan'] += 1
          parser.update([(message.logMonoTime, sends)])
          recorded_parser.update([(message.logMonoTime, raw_can)])
          values = parser.vl['ACC_CONTROL']
          recorded = recorded_parser.vl['ACC_CONTROL']
          automatic = (control.enabled and control.longActive and not control.cruiseControl.override
                       and not car_state.gasPressed and not car_state.brakePressed and control.actuators.longControlState != LongCtrlState.off
                       and latest['selfdriveState'].enabled and message.valid
                       and all(valid.get(name, False) and 0 <= (message.logMonoTime - stamps[name]) / 1e9 <= 0.15
                               for name in ('carControl', 'carState', 'selfdriveState')))
          emit('can', {'t': timestamp, 'mono': message.logMonoTime, 'automatic': bool(automatic), 'can_accel': values['ACCEL_COMMAND'],
                           'can_brake': bool(values['BRAKE_REQUEST']), 'can_gas': int(values['GAS_COMMAND']),
                           'recorded_accel': recorded['ACCEL_COMMAND'], 'recorded_brake': bool(recorded['BRAKE_REQUEST']),
                           'recorded_gas': int(recorded['GAS_COMMAND']), 'can_on': int(values['CONTROL_ON']),
                           'recorded_on': int(recorded['CONTROL_ON']), 'can_stop': bool(values['STANDSTILL']),
                           'recorded_can_stop': bool(recorded['STANDSTILL']), 'control_mono': int(control_mono),
                           'effective_long_active': bool(control.longActive),
                           'driver_override': bool(car_state.gasPressed or car_state.brakePressed or control.cruiseControl.override),
                           'authority_helper_applied': restrict_longitudinal_authority is not None,
                           'authority_restricted': control is not cached_control})
  if execution_observer is not None:
    execution_observer.finish()
  execution.finish()
  for expected, matched in (('plans', 'matched_plans'), ('controls', 'matched_controls'), ('sendcan', 'matched_sendcan')):
    if strict and (not coverage[expected] or coverage[expected] != coverage[matched]):
      raise ValueError(f'Incomplete replay: {window["cache"]}: {coverage}')
  errors = np.abs([row['accel'] - row['recorded_accel'] for row in plan_rows if row['automatic'] and row['valid']])
  motion['consumer_input_sha256'] = intent_input_hash.hexdigest() if motion_enabled else None
  return {'window': window, 'coverage': coverage, 'plan_rows': plan_rows, 'control_rows': control_rows, 'can_rows': can_rows,
          'missing': missing, 'resets': resets, 'solver_failures': solver_failures,
          'motion_replay': motion, 'capabilities': capabilities, 'initialization_sends': initialization_sends,
          'first_executed': first_executed,
          'execution_boundary_mono': min(execution.boundaries.values()),
          'runtime_execution_boundaries_mono': execution.boundaries,
          'runtime_execution_ends_mono': execution.ends,
          'execution_closure': execution.receipt(), 'dependency_outputs': dependency_outputs,
          'initialization_wait': progress.data['initialization_wait'],
          'execution_boundary_layers': ['planner', 'longcontrol', 'carcontroller'],
          'read_only_prefix': read_only_prefix,
          'planner_error': {'p95': float(np.percentile(errors, 95)), 'maximum': float(max(errors))} if len(errors) else None,
          'switches': switch_metrics(can_rows)}


def run_one(arguments):
  corpus, output, index, window, recorded_plan, unified, supervised = arguments
  cache = corpus / window['cache']
  if sha256(cache) != window['sha256']:
    raise ValueError(f'Changed replay cache: {cache}')
  result = replay_window(cache, window, recorded_plan=recorded_plan, unified=unified, supervised=supervised)
  result['experimental_coast_with_lead'] = False
  result['recorded_plan'] = recorded_plan
  result['planner_sha256'] = sha256(Path(planner_module.__file__))
  result['replayer_sha256'] = sha256(Path(__file__))
  source_root = Path(planner_module.__file__).resolve().parents[4]
  source_paths = ('openpilot/selfdrive/controls/lib/longitudinal_planner.py', 'openpilot/selfdrive/controls/lib/longcontrol.py',
                  'openpilot/selfdrive/controls/lib/motion_supervisor.py', 'openpilot/selfdrive/controls/lib/unified_longitudinal_planner.py',
                  'openpilot/selfdrive/controls/lib/longitudinal_mode.py', 'openpilot/selfdrive/controls/lib/longitudinal_trajectory.py',
                  'openpilot/cereal/messaging/__init__.py', 'openpilot/cereal/services.py', 'openpilot/common/utils.py',
                  'openpilot/selfdrive/controls/lib/drive_helpers.py', 'openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py',
                  'openpilot/sunnypilot/selfdrive/controls/lib/longitudinal_planner.py',
                  'openpilot/tools/longitudinal/replay_longitudinal_regressions.py',
                  'openpilot/selfdrive/car/longitudinal_authority.py', 'openpilot/selfdrive/car/card.py',
                  'openpilot/selfdrive/controls/controlsd.py',
                  'opendbc_repo/opendbc/car/honda/carcontroller.py', 'opendbc_repo/opendbc/car/honda/hondacan.py',
                  'opendbc_repo/opendbc/car/honda/interface.py', 'opendbc_repo/opendbc/car/honda/values.py')
  result['source_sha256'] = {relative: sha256(source_root / relative) for relative in source_paths if (source_root / relative).is_file()}
  save_json(output / f'{index:03d}.json', result)
  print(f'{index:03d} {window["route"]} {result["coverage"]} error={result["planner_error"]} switches={result["switches"]["debounced"]}', flush=True)


def run(corpus, output, commit=None, window_index=None, workers=1, *, recorded_plan=False, unified=False, supervised=False):
  selected_mode(unified, supervised)
  manifest = json.loads((corpus / 'manifest.json').read_text())
  if manifest['schema'] != SCHEMA:
    raise ValueError('Unsupported corpus schema')
  output.mkdir(parents=True, exist_ok=True)
  jobs = [(corpus, output, index, window, recorded_plan, unified, supervised) for index, window in enumerate(manifest['windows'])
          if (commit is None or window['commit'].startswith(commit)) and (window_index is None or index == window_index)]
  if workers == 1:
    for job in jobs:
      run_one(job)
  else:
    with ProcessPoolExecutor(max_workers=workers) as pool:
      list(pool.map(run_one, jobs))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest='command', required=True)
  prepare_parser = commands.add_parser('prepare')
  prepare_parser.add_argument('--logs', type=Path, required=True)
  prepare_parser.add_argument('--audit', type=Path, required=True)
  prepare_parser.add_argument('--all-events', action='store_true')
  prepare_parser.add_argument('--workers', type=int, default=1)
  prepare_parser.add_argument('--events-file', default='events-enriched.json')
  prepare_parser.add_argument('--output', type=Path, required=True)
  run_parser = commands.add_parser('run')
  run_parser.add_argument('--corpus', type=Path, required=True)
  run_parser.add_argument('--output', type=Path, required=True)
  run_parser.add_argument('--commit')
  run_parser.add_argument('--window', type=int)
  run_parser.add_argument('--workers', type=int, default=1)
  run_parser.add_argument('--recorded-plan', action='store_true')
  modes = run_parser.add_mutually_exclusive_group()
  modes.add_argument('--unified', action='store_true')
  modes.add_argument('--supervised', action='store_true')
  arguments = parser.parse_args()
  if arguments.command == 'prepare':
    prepare(arguments.logs, arguments.audit, arguments.output, arguments.all_events, arguments.workers, arguments.events_file)
  else:
    run(arguments.corpus, arguments.output, arguments.commit, arguments.window, arguments.workers,
        recorded_plan=arguments.recorded_plan, unified=arguments.unified, supervised=arguments.supervised)


if __name__ == '__main__':
  main()
