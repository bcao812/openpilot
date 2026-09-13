#!/usr/bin/env python3
"""Replay the real planner with recorded perception and SP/DEC decisions, never vehicle motion.

Run the same script with each host-built source root on PYTHONPATH. Compare the baseline
against logged speed/acceleration before interpreting candidate command differences.
"""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, structs
from opendbc.car.honda.hondacan import CanBus, create_acc_commands
from opendbc.car.honda.values import CAR, DBC, CarControllerParams
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib import longitudinal_planner as planner_module
from openpilot.selfdrive.controls.lib.longitudinal_planner import CONTROL_N_T_IDX, LongitudinalPlanner
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP
from openpilot.tools.lib.logreader import LogReader


SERVICES = ('carState', 'carControl', 'controlsState', 'vehicleParameters', 'radarState', 'modelV2', 'selfdriveState')
SOURCE_NAMES = {value: name for name, value in planner_module.LongitudinalPlanSource.schema.enumerants.items()}


class ReplayState(dict):
  def __init__(self, values):
    super().__init__(values)
    self.valid = dict.fromkeys(values, True)

  def all_checks(self, service_list=None):
    return all(self.valid.get(service, False) for service in (service_list or self))


def route_metadata(paths):
  params = params_sp = origin = None
  for message in LogReader([str(paths[0])], sort_by_time=True):
    service = message.which()
    if service == 'carParams':
      params = message.carParams
    elif service == 'carParamsSP':
      params_sp = message.carParamsSP
    if origin is None and service in SERVICES + ('longitudinalPlan', 'longitudinalPlanSP'):
      origin = message.logMonoTime / 1e9
    if params is not None and params_sp is not None and origin is not None:
      return params, params_sp, origin
  raise ValueError('Route must include segment zero with both recorded parameter messages and timestamps')


def frames(paths, origin, start, end, coverage):
  latest, valid, models = {}, {}, {}
  pending = None
  for path in paths:
    segment = int(path.parent.name.rsplit('--', 1)[1])
    if segment < max(0, int(start // 60) - 2) or segment > int(end // 60) + 1:
      continue
    for message in LogReader([str(path)], sort_by_time=True):
      service = message.which()
      timestamp = message.logMonoTime / 1e9
      if timestamp - origin > end + 1.0:
        return
      if service in SERVICES:
        latest[service] = getattr(message, service)
        valid[service] = message.valid
      if service == 'modelV2' and all(service_name in latest for service_name in SERVICES):
        state = ReplayState(latest.copy())
        state.valid = valid.copy()
        models[message.logMonoTime] = state
        if len(models) > 30:
          del models[next(iter(models))]
      elif service == 'longitudinalPlan':
        state = models.get(message.longitudinalPlan.modelMonoTime)
        if start <= timestamp - origin <= end:
          coverage['plans'] += 1
        pending = (timestamp - origin, state, message.longitudinalPlan, message.valid, segment)
      elif service == 'longitudinalPlanSP' and pending is not None:
        timestamp, state, plan, plan_valid, segment = pending
        pending = None
        if state is not None and start - 60.0 <= timestamp <= end:
          if timestamp >= start:
            coverage['matched'] += 1
          yield timestamp, state, plan, message.longitudinalPlanSP, plan_valid, segment


def can_summary(rows, params):
  if params.carFingerprint != CAR.HONDA_ACCORD:
    return None
  bus = CanBus(params)
  dbc_name = DBC[params.carFingerprint][Bus.pt]
  packer = CANPacker(dbc_name)
  parser = CANParser(dbc_name, [('ACC_CONTROL', 50)], bus.pt)
  for index, row in enumerate(rows):
    control = structs.CarControl.new_message()
    control.actuators.accel = row['accel']
    serialized = control.as_reader().actuators.accel
    gas = float(np.interp(serialized, CarControllerParams.BOSCH_GAS_LOOKUP_BP, CarControllerParams.BOSCH_GAS_LOOKUP_V))
    messages = create_acc_commands(packer, bus, True, row['automatic'], serialized, gas, 0, params)
    parser.update([((index + 1) * 50_000_000, messages)])
    values = parser.vl['ACC_CONTROL']
    row.update(can_accel=values['ACCEL_COMMAND'], can_brake=bool(values['BRAKE_REQUEST']), can_gas=values['GAS_COMMAND'])
  return {'raw_brake_flips': sum(previous['can_brake'] != current['can_brake']
                                 for previous, current in zip(rows, rows[1:], strict=False)
                                 if previous['automatic'] and current['automatic']),
          'braking_frames': sum(row['can_brake'] for row in rows),
          'method': 'real Honda encoder/parser after Float32; planner-rate samples, not a 100 Hz CarController replay'}


def replay(logs, route, start, end):
  paths = sorted(logs.glob(f'{route}--*/rlog.zst'), key=lambda path: int(path.parent.name.rsplit('--', 1)[1]))
  if not paths or not paths[0].parent.name.endswith('--0'):
    raise ValueError('Completed local route including segment zero is required')
  params, params_sp, origin = route_metadata(paths)
  planner = LongitudinalPlanner(params, params_sp)
  current = {}

  def update_targets(instance, state, v_ego, a_ego, v_cruise):
    instance.source = current['sp'].longitudinalPlanSource
    return v_cruise, a_ego

  def is_e2e(instance, state):
    dec = current['sp'].dec
    return state['selfdriveState'].experimentalMode and (not dec.active or str(dec.state) == 'blended')

  rows = []
  unsupported = 0
  coverage = {'plans': 0, 'matched': 0}
  with patch.object(LongitudinalPlannerSP, 'update', return_value=None), \
       patch.object(LongitudinalPlannerSP, 'update_targets', new=update_targets), \
       patch.object(LongitudinalPlanner, 'is_e2e', new=is_e2e):
    for timestamp, state, plan, plan_sp, plan_valid, segment in frames(paths, origin, start, end, coverage):
      if str(plan_sp.longitudinalPlanSource) != 'cruise':
        unsupported += 1
      current['sp'] = plan_sp
      planner.update(state)
      if timestamp < start:
        continue
      car_state = state['carState']
      automatic = (state['carControl'].longActive and not car_state.gasPressed and not car_state.brakePressed
                   and not state['carControl'].cruiseControl.override
                   and state['controlsState'].longControlState != LongCtrlState.off)
      rows.append({'t': timestamp, 'seg': segment, 'v_ego': car_state.vEgo, 'a_ego': car_state.aEgo,
                   'gap': state['radarState'].leadOne.dRel, 'lead': state['radarState'].leadOne.present,
                   'lead2': state['radarState'].leadTwo.present, 'automatic': automatic,
                   'valid': plan_valid and state.all_checks(), 'recorded_speed': plan.speeds[0],
                   'mpc_speed': float(planner.v_desired_trajectory[0]), 'recorded_accel': plan.aTarget,
                   'accel': float(planner.output_a_target), 'recorded_stop': plan.shouldStop,
                   'stop': bool(planner.output_should_stop), 'source': SOURCE_NAMES[planner.mpc.source],
                   'recorded_source': str(plan.longitudinalPlanSource), 'solver_status': planner.mpc.solution_status,
                   'mpc_accel': float(get_accel_from_plan(planner.v_desired_trajectory, planner.a_desired_trajectory,
                                                       CONTROL_N_T_IDX, action_t=params.longitudinalActuatorDelay + 0.05)),
                   'cruise_accel': float(planner.a_cruise), 'v_target': float(plan_sp.vTarget),
                   'is_e2e': bool(is_e2e(planner, state)), 'allow_throttle': bool(planner.allow_throttle)})
  if unsupported:
    raise ValueError(f'{unsupported} non-cruise SP target frames require full SP target replay, not this harness')
  if coverage['plans'] != coverage['matched']:
    raise ValueError(f'Unmatched plan inputs: {coverage}')
  selected = [row for row in rows if row['automatic'] and row['valid']]
  if not selected:
    raise ValueError('No valid automatic frames in requested window')
  summary = {'automatic_frames': len(selected), 'solver_failures': sum(row['solver_status'] != 0 for row in selected),
             'coverage': coverage, 'can': can_summary(rows, params)}
  for field, recorded in (('accel', 'recorded_accel'), ('mpc_speed', 'recorded_speed')):
    error = np.abs([row[field] - row[recorded] for row in selected])
    summary[field + '_recorded_abs_error'] = {'median': float(np.median(error)), 'p95': float(np.percentile(error, 95)),
                                             'max': float(max(error))}
  return {'route': route, 'start': start, 'end': end, 'car_fingerprint': params.carFingerprint,
          'planner_sha256': hashlib.sha256(Path(planner_module.__file__).read_bytes()).hexdigest(),
          'method': 'open-loop planner; recorded perception, DEC mode and cruise SP target; 60-second warmup',
          'summary': summary, 'rows': rows}


def debounced_flips(rows):
  phases, current = [], []

  def append_phase():
    if current and current[-1][0] - current[0][0] + 0.05 >= 0.15 and current[0][1] != 'coast':
      phases.append(current[0])

  for row in rows:
    side = 'brake' if row['can_brake'] else ('gas' if row['can_gas'] > 0 else 'coast')
    if current and (side != current[-1][1] or row['t'] - current[-1][0] > 0.15):
      append_phase()
      current = []
    current.append((row['t'], side))
  append_phase()
  return sum(before[1] != after[1] for before, after in zip(phases, phases[1:], strict=False))


def compare_events(events_path, baseline_dir, candidate_dir):
  arms = {name: [json.loads(path.read_text()) for path in directory.glob('000000*.json')]
          for name, directory in (('baseline', baseline_dir), ('candidate', candidate_dir))}
  results = []
  for event in json.loads(events_path.read_text()):
    result = {'route': event['route'], 't': event['t'], 'logged_flips': event['flips']}
    for name, outputs in arms.items():
      output = next(item for item in outputs if item['route'] == event['route'] and item['start'] <= event['t'] <= item['end'])
      selected = [row for row in output['rows'] if event['t'] - 0.001 <= row['t'] < event['t'] + event['duration'] - 0.01]
      result[name + '_flips'] = debounced_flips(selected)
      result[name + '_raw'] = sum(before['can_brake'] != after['can_brake']
                                 for before, after in zip(selected, selected[1:], strict=False))
    results.append(result)
  errors = [abs(row['accel'] - row['recorded_accel']) for output in arms['baseline']
            for row in output['rows'] if row['automatic'] and row['valid']]
  return {'events': results, 'baseline_debounced': sum(event['baseline_flips'] for event in results),
          'candidate_debounced': sum(event['candidate_flips'] for event in results),
          'baseline_raw': sum(event['baseline_raw'] for event in results),
          'candidate_raw': sum(event['candidate_raw'] for event in results),
          'improved_windows': sum(event['candidate_flips'] < event['baseline_flips'] for event in results),
          'worse_windows': sum(event['candidate_flips'] > event['baseline_flips'] for event in results),
          'candidate_windows_ge4': sum(event['candidate_flips'] >= 4 for event in results),
          'baseline_error_p95': float(np.percentile(errors, 95)), 'baseline_error_max': max(errors)}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--logs', type=Path)
  parser.add_argument('--route')
  parser.add_argument('--start', type=float)
  parser.add_argument('--end', type=float)
  parser.add_argument('--windows', type=Path)
  parser.add_argument('--compare-events', type=Path)
  parser.add_argument('--baseline-dir', type=Path)
  parser.add_argument('--candidate-dir', type=Path)
  parser.add_argument('--output', type=Path, required=True)
  arguments = parser.parse_args()
  if arguments.compare_events:
    if arguments.baseline_dir is None or arguments.candidate_dir is None:
      parser.error('--compare-events requires --baseline-dir and --candidate-dir')
    result = compare_events(arguments.compare_events, arguments.baseline_dir, arguments.candidate_dir)
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'events'}, indent=2))
  elif arguments.logs is None:
    parser.error('--logs is required for replay')
  elif arguments.windows:
    if arguments.route is not None or arguments.start is not None or arguments.end is not None:
      parser.error('--windows cannot be combined with --route/--start/--end')
    arguments.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for window in json.loads(arguments.windows.read_text()):
      if window['end'] <= window['start']:
        parser.error('Every window must have end > start')
      result = replay(arguments.logs, window['route'], window['start'], window['end'])
      name = f"{window['route']}--{window['start']:.3f}.json"
      (arguments.output / name).write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
      summaries.append({key: result[key] for key in ('route', 'start', 'end', 'summary')})
      print(json.dumps(summaries[-1]), flush=True)
    (arguments.output / 'summary.json').write_text(json.dumps(summaries, indent=2, allow_nan=False) + '\n')
  else:
    if arguments.route is None or arguments.start is None or arguments.end is None or arguments.end <= arguments.start:
      parser.error('--route, --start and --end > --start are required without --windows')
    result = replay(arguments.logs, arguments.route, arguments.start, arguments.end)
    arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
  main()
