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

from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib import longitudinal_planner as planner_module
from openpilot.selfdrive.controls.lib.longitudinal_planner import CONTROL_N_T_IDX, LongitudinalPlanner
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP
from openpilot.tools.lib.logreader import LogReader


SERVICES = ('carState', 'carControl', 'controlsState', 'vehicleParameters', 'radarState', 'modelV2', 'selfdriveState')


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


def frames(paths, origin, start, end):
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
        pending = (timestamp - origin, state, message.longitudinalPlan, message.valid, segment)
      elif service == 'longitudinalPlanSP' and pending is not None:
        timestamp, state, plan, plan_valid, segment = pending
        pending = None
        if state is not None and start - 60.0 <= timestamp <= end:
          yield timestamp, state, plan, message.longitudinalPlanSP, plan_valid, segment


def replay(logs, route, start, end):
  paths = sorted(logs.glob(f'{route}--*/rlog.zst'), key=lambda path: int(path.parent.name.rsplit('--', 1)[1]))
  if not paths or not paths[0].parent.name.endswith('--0'):
    raise ValueError('Completed local route including segment zero is required')
  params, params_sp, origin = route_metadata(paths)
  planner = LongitudinalPlanner(params, params_sp)
  current = {}

  def update_targets(instance, state, v_ego, a_ego, v_cruise):
    instance.source = current['sp'].longitudinalPlanSource
    return current['sp'].vTarget, a_ego

  def is_e2e(instance, state):
    dec = current['sp'].dec
    return state['selfdriveState'].experimentalMode and (not dec.active or str(dec.state) == 'blended')

  rows = []
  unsupported = 0
  with patch.object(LongitudinalPlannerSP, 'update', return_value=None), \
       patch.object(LongitudinalPlannerSP, 'update_targets', new=update_targets), \
       patch.object(LongitudinalPlanner, 'is_e2e', new=is_e2e):
    for timestamp, state, plan, plan_sp, plan_valid, segment in frames(paths, origin, start, end):
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
                   'stop': bool(planner.output_should_stop), 'source': str(planner.mpc.source),
                   'recorded_source': str(plan.longitudinalPlanSource), 'solver_status': planner.mpc.solution_status,
                   'mpc_accel': float(get_accel_from_plan(planner.v_desired_trajectory, planner.a_desired_trajectory,
                                                       CONTROL_N_T_IDX, action_t=params.longitudinalActuatorDelay + 0.05)),
                   'cruise_accel': float(planner.a_cruise), 'v_target': float(plan_sp.vTarget),
                   'is_e2e': bool(is_e2e(planner, state)), 'allow_throttle': bool(planner.allow_throttle)})
  if unsupported:
    raise ValueError(f'{unsupported} non-cruise SP target frames require full SP target replay, not this harness')
  selected = [row for row in rows if row['automatic'] and row['valid']]
  if not selected:
    raise ValueError('No valid automatic frames in requested window')
  summary = {'automatic_frames': len(selected), 'solver_failures': sum(row['solver_status'] != 0 for row in selected)}
  for field, recorded in (('accel', 'recorded_accel'), ('mpc_speed', 'recorded_speed')):
    error = np.abs([row[field] - row[recorded] for row in selected])
    summary[field + '_recorded_abs_error'] = {'median': float(np.median(error)), 'p95': float(np.percentile(error, 95)),
                                             'max': float(max(error))}
  return {'route': route, 'start': start, 'end': end, 'car_fingerprint': params.carFingerprint,
          'planner_sha256': hashlib.sha256(Path(planner_module.__file__).read_bytes()).hexdigest(),
          'method': 'open-loop planner; recorded perception, DEC mode and cruise SP target; 60-second warmup',
          'summary': summary, 'rows': rows}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--logs', type=Path, required=True)
  parser.add_argument('--route', required=True)
  parser.add_argument('--start', type=float, required=True)
  parser.add_argument('--end', type=float, required=True)
  parser.add_argument('--output', type=Path, required=True)
  arguments = parser.parse_args()
  if arguments.end <= arguments.start:
    parser.error('--end must be after --start')
  result = replay(arguments.logs, arguments.route, arguments.start, arguments.end)
  arguments.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
  print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
  main()
