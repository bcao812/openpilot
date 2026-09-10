"""Extract exact planner-used inputs from longitudinal.trace logMessage events, not control commands."""

import argparse
import json
from pathlib import Path

from openpilot.selfdrive.controls.lib.longitudinal_trace import TRACE_EVENT, TRACE_FIELDS, TRACE_SCHEMA


def decode_trace(record):
  parsed = json.loads(record)
  if not isinstance(parsed, dict):
    return []
  payload = parsed.get('msg')
  if not isinstance(payload, dict) or payload.get('event') != TRACE_EVENT:
    return []
  if payload.get('schema') != TRACE_SCHEMA or payload.get('fields') != list(TRACE_FIELDS):
    raise ValueError('Unsupported longitudinal trace schema')
  samples = payload.get('samples')
  if not isinstance(samples, list) or len(samples) > 40:
    raise ValueError('Invalid longitudinal trace batch')
  rows = []
  for sample in samples:
    if not isinstance(sample, list) or len(sample) != len(TRACE_FIELDS):
      raise ValueError('Truncated longitudinal trace sample')
    if any(value is not None and type(value) not in (str, int, float, bool) for value in sample):
      raise ValueError('Invalid longitudinal trace scalar')
    row = dict(zip(TRACE_FIELDS, sample, strict=True))
    for name in ('sequence', 'plan_mono'):
      if type(row[name]) is not int or row[name] <= 0:
        raise ValueError('Invalid longitudinal trace identity')
    row.update(dropped=payload.get('dropped'), failures=payload.get('failures'))
    rows.append(row)
  return rows


def main():
  from openpilot.tools.lib.logreader import LogReader

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('logs', type=Path, nargs='+')
  parser.add_argument('--output', type=Path, required=True)
  arguments = parser.parse_args()
  for path in arguments.logs:
    if not path.is_file():
      parser.error(f'Local log file does not exist: {path}')
  count = 0
  with arguments.output.open('x') as output:
    for message in LogReader([str(path) for path in arguments.logs], sort_by_time=True):
      if message.which() == 'logMessage':
        for row in decode_trace(message.logMessage):
          row['log_mono'] = int(message.logMonoTime)
          output.write(json.dumps(row, allow_nan=False) + '\n')
          count += 1
  print(json.dumps({'samples': count, 'output': str(arguments.output)}))


if __name__ == '__main__':
  main()
