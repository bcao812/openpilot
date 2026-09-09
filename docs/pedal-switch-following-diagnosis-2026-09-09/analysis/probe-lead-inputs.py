"""Historical offline analysis; hard-coded local run/corpus paths, no device or network access."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from openpilot.selfdrive.controls.tests.replay_cruise_smoothing import window_messages
from openpilot.selfdrive.controls.tests.replay_pedal_full_routes import NumericSink, ROW_FIELDS
from openpilot.selfdrive.controls.tests.replay_pedal_switches import replay_window, save_json


run = Path('/tmp/c3-follow-chain-20260909-BtK1Bx')
corpus = Path('/tmp/c3-pedal-full-20260909-WgogBo/corpus')


class NullSink:
  def flush(self):
    pass


def smooth_quadratic(values):
  positions = np.arange(-5, 6, dtype=float)
  design = np.column_stack((np.ones(11), positions, positions**2))
  inverse = np.linalg.pinv(design)
  smoothed = np.convolve(values, inverse[0][::-1], mode='valid')
  first = design[:5] @ inverse @ values[:11]
  last = design[-5:] @ inverse @ values[-11:]
  return np.r_[first, smoothed, last]


def execute(job):
  window, field = job
  destination = run / ('probe-' + field)
  if (destination / f'{window["id"]}.json').exists():
    return
  records = [message for message in window_messages(corpus, window, NullSink()) if message.which() == 'radarState']
  replacements = {}
  for name in ('leadOne', 'leadTwo'):
    leads = [getattr(message.radarState, name) for message in records]
    times = np.array([message.logMonoTime / 1e9 for message in records])
    eligible = np.array([lead.present and not lead.radar for lead in leads])
    boundaries = np.r_[0, np.flatnonzero((eligible[1:] != eligible[:-1]) | (np.diff(times) > 0.15)) + 1, len(leads)]
    original = np.array([getattr(lead, field) for lead in leads])
    smooth = original.copy()
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
      if eligible[start] and end - start >= 11:
        smooth[start:end] = smooth_quadratic(original[start:end])
    for message, value in zip(records, smooth, strict=True):
      replacements.setdefault(message.logMonoTime, {})[name] = float(value)
  destination.mkdir(exist_ok=True)
  sink = NumericSink(destination, window['id'])

  def messages():
    for message in window_messages(corpus, window, sink):
      if message.which() == 'radarState':
        modified = message.as_builder()
        for name in ('leadOne', 'leadTwo'):
          lead = getattr(modified.radarState, name)
          value = replacements[message.logMonoTime][name]
          if field == 'vRel':
            lead.vLead += value - lead.vRel
            lead.vLeadK += value - lead.vRel
          setattr(lead, field, value)
        message = modified.as_reader()
      yield message

  started = time.monotonic()
  result = replay_window(None, dict(window, cache=window['id']), messages=messages(), row_sink=sink, strict=True)
  sink.flush()
  for name in ('plan_rows', 'control_rows', 'can_rows', 'planner_error', 'switches'):
    result.pop(name)
  result.update(
    mode=field,
    chunks=sink.chunks,
    fields=ROW_FIELDS,
    elapsed_s=time.monotonic() - started,
    diagnostic_only='Centered 11-sample Savitzky-Golay intervention uses future data; not deployable or a safety proposal',
  )
  save_json(destination / f'{window["id"]}.json', result)
  print(window['id'], field, flush=True)


if __name__ == '__main__':
  selection = json.loads((run / 'selection.json').read_text())
  chosen = [window for window in selection['windows'] if window['reason'] in ('fast_pure_lead', 'slow_pure_lead')]
  jobs = [(window, field) for window in chosen for field in ('dRel', 'vRel', 'aLeadK')]
  with ProcessPoolExecutor(max_workers=3) as pool:
    list(pool.map(execute, jobs))
