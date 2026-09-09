"""Historical offline analysis; hard-coded local run/corpus paths, no device or network access."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

from openpilot.selfdrive.controls.lib import longitudinal_planner as planner_module
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc
from openpilot.selfdrive.controls.tests.replay_cruise_smoothing import window_messages
from openpilot.selfdrive.controls.tests.replay_pedal_full_routes import NumericSink, ROW_FIELDS
from openpilot.selfdrive.controls.tests.replay_pedal_switches import replay_window, save_json


run = Path('/tmp/c3-follow-chain-20260909-BtK1Bx')
corpus = Path('/tmp/c3-pedal-full-20260909-WgogBo/corpus')


def execute(window):
  destination = run / 'trace'
  destination.mkdir(exist_ok=True)
  recorded = []
  current = {}
  planner_update = planner_module.LongitudinalPlanner.update
  mpc_update = LongitudinalMpc.update

  def monitor_mpc(mpc, radarstate, *args, **kwargs):
    current['mpc_input_speed'] = float(mpc.x0[1])
    current['mpc_input_accel'] = float(mpc.x0[2])
    result = mpc_update(mpc, radarstate, *args, **kwargs)
    current['mpc_selected_lead'] = str(mpc.source)
    current['obstacle_t0'] = float(mpc.params[0, 2])
    current['obstacle_t1'] = float(np.interp(1.0, planner_module.T_IDXS_MPC, mpc.params[:, 2]))
    current['obstacle_t2'] = float(np.interp(2.0, planner_module.T_IDXS_MPC, mpc.params[:, 2]))
    current['obstacle_t4'] = float(np.interp(4.0, planner_module.T_IDXS_MPC, mpc.params[:, 2]))
    return result

  def monitor_planner(planner, state):
    current.clear()
    current['previous_output'] = float(planner.output_a_target)
    result = planner_update(planner, state)
    current['model_mono'] = int(state['modelV2'].logMonoTime) if hasattr(state['modelV2'], 'logMonoTime') else 0
    current['radar_model_mono'] = int(state['radarState'].mdMonoTime)
    current['cruise_accel'] = float(planner.a_cruise)
    current['mpc_accel'] = float(
      planner_module.get_accel_from_plan(
        planner.v_desired_trajectory, planner.a_desired_trajectory, planner_module.CONTROL_N_T_IDX, action_t=planner.CP.longitudinalActuatorDelay + planner.dt
      )
    )
    current['model_accel'] = float(state['modelV2'].action.desiredAcceleration)
    current['model_stop'] = bool(state['modelV2'].action.shouldStop)
    current['ego_speed'] = float(state['carState'].vEgo)
    current['ego_accel'] = float(state['carState'].aEgo)
    current['model_ego_speed'] = float(state['modelV2'].velocity.x[0]) if len(state['modelV2'].velocity.x) else None
    current['model_ego_accel'] = float(state['modelV2'].acceleration.x[0]) if len(state['modelV2'].acceleration.x) else None
    current['set_speed'] = float(state['carState'].vCruise)
    current['allow_throttle'] = bool(planner.allow_throttle)
    current['personality'] = str(state['selfdriveState'].personality)
    current['e2e'] = bool(planner.is_e2e(state))
    current['source'] = str(planner.mpc.source)
    current['a_target'] = float(planner.output_a_target)
    for index, name in enumerate(('leadOne', 'leadTwo')):
      lead = getattr(state['radarState'], name)
      current[name] = {
        field: getattr(lead, field) for field in ('present', 'dRel', 'vRel', 'vLead', 'aLeadK', 'aLeadTau', 'modelProb', 'radar', 'radarTrackId')
      }
      if len(state['modelV2'].leadsV3) > index:
        raw = state['modelV2'].leadsV3[index]
        current['model_' + name] = {field: list(getattr(raw, field)) for field in ('x', 'v', 'a', 'xStd', 'vStd', 'aStd')}
        current['model_' + name]['prob'] = float(raw.prob)
    return result

  class TraceSink(NumericSink):
    def __call__(self, kind, row):
      if kind == 'plan':
        recorded.append(dict(current, t=row['t'], mono=row['mono'], automatic=row['automatic'], valid=row['valid']))
      return super().__call__(kind, row)

  start = time.monotonic()
  sink = TraceSink(destination, window['id'])
  with patch.object(planner_module.LongitudinalPlanner, 'update', new=monitor_planner), patch.object(LongitudinalMpc, 'update', new=monitor_mpc):
    result = replay_window(None, dict(window, cache=window['id']), messages=window_messages(corpus, window, sink), row_sink=sink, strict=True)
  sink.flush()
  for field in ('plan_rows', 'control_rows', 'can_rows', 'planner_error', 'switches'):
    result.pop(field)
  result.update(mode='off', chunks=sink.chunks, fields=ROW_FIELDS, elapsed_s=time.monotonic() - start)
  save_json(destination / f'{window["id"]}.json', result)
  save_json(destination / f'{window["id"]}.trace.json', recorded)
  print(window['id'], window['reason'], result['coverage'], flush=True)


if __name__ == '__main__':
  selection = json.loads((run / 'selection.json').read_text())
  windows = []
  for window in selection['windows']:
    metadata = run / 'trace' / f'{window["id"]}.json'
    if not metadata.exists() or json.loads(metadata.read_text())['window']['warmup'] != window['warmup']:
      windows.append(window)
  with ProcessPoolExecutor(max_workers=3) as pool:
    list(pool.map(execute, windows))
