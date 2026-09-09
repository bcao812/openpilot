#!/usr/bin/env python3
"""Local, timestamp-aligned open-loop planner/LongControl/Honda CAN regression.

Writes only below --output. Recorded motion, perception, SP decisions and controller
scheduling stay fixed; command differences are not changes in vehicle motion.
"""

import argparse
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
from opendbc.car.honda.values import CAR, DBC
from openpilot.selfdrive.controls.lib import longitudinal_planner as planner_module
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.tests.replay_longitudinal_regressions import ReplayState, SERVICES
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP
from openpilot.tools.lib.logreader import LogReader


SCHEMA = 1
RETAINED = set(SERVICES) | {'longitudinalPlan', 'longitudinalPlanSP', 'sendcan'}
PARAMETERS = {'carParams', 'carParamsSP', 'initData'}
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


def paired_messages(messages, maximum_delay=0.25, maximum_messages=512):
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
        yield pending[0], message.longitudinalPlanSP
        yield from ((buffered, None) for buffered in pending[1:])
        pending = []
    else:
      yield message, None
  yield from ((buffered, None) for buffered in pending)


def replay_window(cache, window, experimental_coast_with_lead=False, recorded_plan=False, *, messages=None, row_sink=None, strict=True,
                  experimental_cruise_smoothing='off'):
  if experimental_coast_with_lead and experimental_cruise_smoothing != 'off':
    raise ValueError('Lead cruise experiments must be replayed separately')
  if messages is None:
    messages = LogReader([str(cache)], sort_by_time=True)
  latest, valid, stamps, models, generated = {}, {}, {}, {}, {}
  planner = controller = longitudinal = None
  params = params_sp = None
  pending_control = controller_input = None
  control_outputs = {}
  first_controller_frame = True
  current = {}
  plan_rows, can_rows, control_rows = [], [], []
  coverage = {'plans': 0, 'matched_plans': 0, 'controls': 0, 'matched_controls': 0, 'sendcan': 0, 'matched_sendcan': 0}
  missing = {}
  resets = []
  previous_timestamp = None
  controller_skipped = True
  solver_failures = 0
  warmup_start = window.get('warmup', window['start'] - 60.0)
  control_sp = structs.CarControlSP()
  state_wrapper = SimpleNamespace(v_cruise_factor=1.0, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={})

  def emit(kind, row):
    if row_sink is not None:
      row_sink(kind, row)
    else:
      {'plan': plan_rows, 'control': control_rows, 'can': can_rows}[kind].append(row)

  def unavailable(reason, measured):
    if measured:
      missing[reason] = missing.get(reason, 0) + 1

  def update_targets(instance, state, v_ego, a_ego, v_cruise):
    instance.source = current['sp'].longitudinalPlanSource
    target_accel = a_ego if str(instance.source) == 'cruise' else current['sp'].aTarget
    return current['sp'].vTarget, target_accel

  def is_e2e(instance, state):
    dec = current['sp'].dec
    return state['selfdriveState'].experimentalMode and (not dec.active or str(dec.state) == 'blended')

  def frozen_sp_update(instance, state):
    """A non-recording stub must not accumulate every full-route input like Mock does."""
    return None

  with patch.object(LongitudinalPlannerSP, 'update', new=frozen_sp_update), \
       patch.object(LongitudinalPlannerSP, 'update_targets', new=update_targets), \
       patch.object(planner_module.LongitudinalPlanner, 'is_e2e', new=is_e2e):
    for message, decision in paired_messages(messages):
      service = message.which()
      timestamp = message.logMonoTime / 1e9 - window['origin']
      measured = window['start'] <= timestamp <= window['end']
      if service not in PARAMETERS:
        if previous_timestamp is not None and timestamp < previous_timestamp:
          raise ValueError(f'Nonmonotonic replay input at {timestamp:.9f}')
        if not strict and previous_timestamp is not None and timestamp - previous_timestamp > 0.5:
          resets.append({'before': previous_timestamp, 'after': timestamp})
          latest, valid, stamps, models, generated, control_outputs = {}, {}, {}, {}, {}, {}
          planner = controller = longitudinal = pending_control = controller_input = None
          first_controller_frame = controller_skipped = True
        previous_timestamp = timestamp
      if service == 'carParams':
        params = message.carParams
      elif service == 'carParamsSP':
        params_sp = message.carParamsSP
      if planner is None and params is not None and params_sp is not None:
        if params.carFingerprint != CAR.HONDA_ACCORD or not params.openpilotLongitudinalControl:
          raise ValueError('This replay requires Accord openpilot longitudinal')
        planner_options = {'experimental_cruise_smoothing': experimental_cruise_smoothing} if experimental_cruise_smoothing != 'off' else {}
        planner = planner_module.LongitudinalPlanner(params, params_sp, **planner_options)
        if experimental_coast_with_lead:
          if not hasattr(planner, 'experimental_coast_with_lead'):
            raise ValueError('Planner does not implement the experimental lead gate')
          planner.experimental_coast_with_lead = True
        controller_params_sp = structs.CarParamsSP()
        for name in ('flags', 'safetyParam', 'enableGasInterceptor'):
          setattr(controller_params_sp, name, getattr(params_sp, name))
        controller = CarController({Bus.pt: DBC[params.carFingerprint][Bus.pt]}, params, controller_params_sp)
        longitudinal = LongControl(params, controller_params_sp)
        parser = CANParser(DBC[params.carFingerprint][Bus.pt], [('ACC_CONTROL', 50)], CanBus(params).pt)
        recorded_parser = CANParser(DBC[params.carFingerprint][Bus.pt], [('ACC_CONTROL', 50)], CanBus(params).pt)
      if service in SERVICES:
        latest[service] = getattr(message, service)
        valid[service] = bool(message.valid)
        stamps[service] = message.logMonoTime
      if service == 'modelV2' and all(name in latest for name in SERVICES):
        state = ReplayState(latest.copy())
        state.valid = {name: valid[name] and 0 <= (message.logMonoTime - stamps[name]) / 1e9 <= 0.3 for name in SERVICES}
        models[message.logMonoTime] = state
        if len(models) > 30:
          del models[next(iter(models))]
      elif service == 'longitudinalPlan':
        coverage['plans'] += int(measured)
        plan = message.longitudinalPlan
        state = models.get(plan.modelMonoTime)
        plan_time = timestamp
        if state is not None and planner is not None and decision is not None and plan_time >= warmup_start:
          current['sp'] = decision
          planner.update(state)
          replayed_accel = float(np.float32(planner.output_a_target))
          replayed_stop = bool(planner.output_should_stop)
          generated[message.logMonoTime] = (plan.aTarget, plan.shouldStop) if recorded_plan else (replayed_accel, replayed_stop)
          if window['start'] <= plan_time <= window['end']:
            coverage['matched_plans'] += 1
            solver_failures += int(planner.mpc.solution_status != 0)
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
                              'sp_cruise': str(decision.longitudinalPlanSource) == 'cruise'})
          if len(generated) > 100:
            del generated[next(iter(generated))]
        else:
          reason = 'plan_parameters' if planner is None else 'plan_model' if state is None else 'plan_sp' if decision is None else 'plan_warmup'
          unavailable(reason, measured)
      elif service == 'controlsState' and 'carState' in latest:
        pending_control = (message.controlsState.longitudinalPlanMonoTime, latest['carState'])
      elif service == 'carControl':
        coverage['controls'] += int(measured)
        if longitudinal is None or pending_control is None:
          unavailable('control_parameters' if longitudinal is None else 'control_state', measured)
          continue
        plan_mono, car_state = pending_control
        pending_control = None
        if plan_mono in generated:
          target_accel, target_stop = generated[plan_mono]
          limits = CarInterface.get_pid_accel_limits(params, controller_params_sp, car_state.vEgo, car_state.vCruise / 3.6)
          accel = longitudinal.update(message.carControl.longActive, car_state, target_accel, target_stop, limits)
          control = message.carControl.as_builder()
          control.actuators.accel = float(accel)
          control.actuators.longControlState = longitudinal.long_control_state
          control_outputs[message.logMonoTime] = control.as_reader()
          if measured:
            coverage['matched_controls'] += 1
            emit('control', {'t': timestamp, 'mono': message.logMonoTime, 'plan_mono': plan_mono,
                                 'accel': control.actuators.accel, 'recorded_accel': message.carControl.actuators.accel,
                                 'state': str(longitudinal.long_control_state), 'recorded_state': str(message.carControl.actuators.longControlState)})
          if len(control_outputs) > 100:
            del control_outputs[next(iter(control_outputs))]
        else:
          unavailable('control_plan', measured)
      elif service == 'carState' and 'carControl' in stamps:
        controller_input = (stamps['carControl'], message.carState)
      elif service == 'sendcan':
        raw_can = [(frame.address, frame.dat, frame.src) for frame in message.sendcan]
        has_acc = any(address == 0x1df and bus == 1 for address, _, bus in raw_can)
        coverage['sendcan'] += int(measured and has_acc)
        if controller_input is None or controller is None:
          unavailable('can_parameters' if controller is None else 'can_state', measured and has_acc)
          controller_skipped = True
          continue
        control_mono, car_state = controller_input
        controller_input = None
        if control_mono not in control_outputs:
          unavailable('can_control', measured and has_acc)
          controller_skipped = True
          continue
        control = control_outputs[control_mono]
        if first_controller_frame:
          controller.frame = 0 if has_acc else 1
          first_controller_frame = False
        elif not strict and controller_skipped and (controller.frame % 2 == 0) != has_acc:
          controller.frame += 1
          resets.append({'can_cadence': timestamp})
        controller_skipped = False
        state_wrapper.out = car_state
        _, sends = controller.update(control, control_sp, state_wrapper, message.logMonoTime)
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
                           'recorded_can_stop': bool(recorded['STANDSTILL'])})
  for expected, matched in (('plans', 'matched_plans'), ('controls', 'matched_controls'), ('sendcan', 'matched_sendcan')):
    if strict and (not coverage[expected] or coverage[expected] != coverage[matched]):
      raise ValueError(f'Incomplete replay: {window["cache"]}: {coverage}')
  errors = np.abs([row['accel'] - row['recorded_accel'] for row in plan_rows if row['automatic'] and row['valid']])
  return {'window': window, 'coverage': coverage, 'plan_rows': plan_rows, 'control_rows': control_rows, 'can_rows': can_rows,
          'missing': missing, 'resets': resets, 'solver_failures': solver_failures,
          'planner_error': {'p95': float(np.percentile(errors, 95)), 'maximum': float(max(errors))} if len(errors) else None,
          'switches': switch_metrics(can_rows)}


def run_one(arguments):
  corpus, output, index, window, experimental_coast_with_lead, recorded_plan = arguments
  cache = corpus / window['cache']
  if sha256(cache) != window['sha256']:
    raise ValueError(f'Changed replay cache: {cache}')
  result = replay_window(cache, window, experimental_coast_with_lead, recorded_plan)
  result['experimental_coast_with_lead'] = experimental_coast_with_lead
  result['recorded_plan'] = recorded_plan
  result['planner_sha256'] = sha256(Path(planner_module.__file__))
  result['replayer_sha256'] = sha256(Path(__file__))
  source_root = Path(planner_module.__file__).resolve().parents[4]
  source_paths = ('openpilot/selfdrive/controls/lib/longitudinal_planner.py', 'openpilot/selfdrive/controls/lib/longcontrol.py',
                  'openpilot/selfdrive/controls/lib/drive_helpers.py', 'openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py',
                  'openpilot/sunnypilot/selfdrive/controls/lib/longitudinal_planner.py',
                  'opendbc_repo/opendbc/car/honda/carcontroller.py', 'opendbc_repo/opendbc/car/honda/hondacan.py',
                  'opendbc_repo/opendbc/car/honda/interface.py', 'opendbc_repo/opendbc/car/honda/values.py')
  result['source_sha256'] = {relative: sha256(source_root / relative) for relative in source_paths}
  save_json(output / f'{index:03d}.json', result)
  print(f'{index:03d} {window["route"]} {result["coverage"]} error={result["planner_error"]} switches={result["switches"]["debounced"]}', flush=True)


def run(corpus, output, commit=None, window_index=None, workers=1, experimental_coast_with_lead=False, recorded_plan=False):
  manifest = json.loads((corpus / 'manifest.json').read_text())
  if manifest['schema'] != SCHEMA:
    raise ValueError('Unsupported corpus schema')
  output.mkdir(parents=True, exist_ok=True)
  jobs = [(corpus, output, index, window, experimental_coast_with_lead, recorded_plan) for index, window in enumerate(manifest['windows'])
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
  run_parser.add_argument('--experimental-coast-with-lead', action='store_true')
  run_parser.add_argument('--recorded-plan', action='store_true')
  verify_parser = commands.add_parser('verify-leads')
  verify_parser.add_argument('--corpus', type=Path, required=True)
  verify_parser.add_argument('--output', type=Path, required=True)
  arguments = parser.parse_args()
  if arguments.command == 'prepare':
    prepare(arguments.logs, arguments.audit, arguments.output, arguments.all_events, arguments.workers, arguments.events_file)
  elif arguments.command == 'run':
    run(arguments.corpus, arguments.output, arguments.commit, arguments.window, arguments.workers,
        arguments.experimental_coast_with_lead, arguments.recorded_plan)
  else:
    verify_leads(arguments.corpus, arguments.output)


def verify_leads(corpus, output):
  manifest = json.loads((corpus / 'manifest.json').read_text())
  checked, invalid = 0, []
  for window in manifest['windows']:
    for message in LogReader([str(corpus / window['cache'])]):
      if message.which() != 'radarState':
        continue
      for name in ('leadOne', 'leadTwo'):
        lead = getattr(message.radarState, name)
        if lead.present:
          checked += 1
          if not np.isfinite(lead.aLeadTau) or lead.aLeadTau <= 0:
            invalid.append({'cache': window['cache'], 'mono': message.logMonoTime, 'lead': name})
  result = {'checked': checked, 'invalid': invalid, 'manifest_sha256': sha256(corpus / 'manifest.json'),
            'helper_sha256': hashlib.sha256(inspect.getsource(planner_module.cruise_coast_leads_clear).encode()).hexdigest()}
  save_json(output, result)
  if invalid:
    raise ValueError(f'{len(invalid)} invalid lead time constants require re-evaluation')


if __name__ == '__main__':
  main()
