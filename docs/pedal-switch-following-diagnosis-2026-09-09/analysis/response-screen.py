"""Historical offline analysis; hard-coded local run/corpus paths, no device or network access."""

import json
from pathlib import Path

import numpy as np

from openpilot.selfdrive.controls.tests.evaluate_pedal_full_routes import columns, native_clean
from openpilot.selfdrive.controls.tests.replay_pedal_full_routes import NATIVE_FIELDS
from openpilot.selfdrive.controls.tests.replay_pedal_switches import save_json


run = Path('/tmp/c3-follow-chain-20260909-BtK1Bx')
audit = Path('/home/bcao/SourceCode/_logs/pedal-switch-2026-09-09')
selection = json.loads((run / 'selection.json').read_text())
outputs = []
for window in selection['windows'][:28]:
  with np.load(audit / f'{window["route"]}.native.npz', allow_pickle=False) as saved:
    native = columns(saved['frames'], NATIVE_FIELDS)
  times = native['t']
  mask = native_clean(native) & (times >= window['start']) & (times <= window['end'])
  mask &= (native['source'] >= 1) & (native['source'] <= 2) & (native['v_ego'] >= 5)
  selected = np.flatnonzero(mask)
  if not len(selected):
    continue
  positions = np.searchsorted(times, times[selected] + 0.5)
  positions = np.clip(positions, 0, len(times) - 1)
  valid = mask[positions] & (abs(times[positions] - times[selected] - 0.5) <= 0.04)
  selected, positions = selected[valid], positions[valid]
  command = native['accel_cmd'][selected]
  band = (command >= -0.3) & (command <= -0.1)
  if np.sum(band) < 25:
    continue
  error = native['a_ego'][positions][band] - command[band]
  model_error = native['v_ego'][selected][band] - native['planned_v'][selected][band]
  outputs.append(
    {
      'id': window['id'],
      'reason': window['reason'],
      'frames': int(np.sum(band)),
      'delayed_aEgo_minus_command': dict(zip(('p05', 'p50', 'p95'), np.percentile(error, [5, 50, 95]).tolist(), strict=True)),
      'measured_minus_planned_speed': dict(zip(('p05', 'p50', 'p95'), np.percentile(model_error, [5, 50, 95]).tolist(), strict=True)),
    }
  )
save_json(
  run / 'artifacts/response-screen.json',
  {
    'delay_s': 0.5,
    'records': outputs,
    'limitation': (
      'Observed signals, not plant identification. Traffic motion, grade, measurement filtering and the assumed delay are confounded; '
      + 'no hardware calibration inferred.'
    ),
  },
)
for output in outputs:
  print(output)
