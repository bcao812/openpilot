"""Extract planner input snapshots and attempt evidence from longitudinal.trace logMessage events."""

import argparse
import json
import math
from pathlib import Path

from openpilot.common.longitudinal_logging import UINT64_MAX
from openpilot.selfdrive.controls.lib.longitudinal_trace import (
  DETAIL_SCHEMA, MOTION_TRACE_SCHEMA, TRACE_EVENT, TRACE_FIELDS, TRACE_SCHEMA, UNIFIED_TRACE_FIELDS,
)


# Historical schema 2 ended at intent_ready, before the mode and braking fields.
SCHEMA_2_FIELDS = TRACE_FIELDS + (
  'intent_version', 'intent_reasons', 'intent_valid', 'moving_lead_evidence', 'release_evidence',
  'allow_creep', 'reference_accel', 'constraint_accel', 'reference_overridden', 'constraint_source', 'intent_ready',
)
SCHEMA_FIELDS = {TRACE_SCHEMA: TRACE_FIELDS, 2: SCHEMA_2_FIELDS, MOTION_TRACE_SCHEMA: UNIFIED_TRACE_FIELDS}
DETAIL_IDS = ('session', 'sequence', 'plan_mono', 'start_mono', 'end_mono', 'planner_session', 'planner_sequence')


def uint64_value(value, name, *, positive=False):
  if type(value) is str and value.isascii() and value.isdecimal():
    number = int(value)
  elif type(value) is int:
    number = value
  else:
    raise ValueError(f'Invalid longitudinal trace identity: {name}')
  if not int(positive) <= number <= UINT64_MAX:
    raise ValueError(f'Invalid longitudinal trace identity: {name}')
  return number


def validate_finite(value):
  if isinstance(value, dict):
    for item in value.values():
      validate_finite(item)
  elif isinstance(value, list):
    for item in value:
      validate_finite(item)
  elif type(value) is float and not math.isfinite(value):
    raise ValueError('Nonfinite longitudinal trace value')


def decode_detail(detail, row):
  if detail is None:
    return None
  if not isinstance(detail, dict):
    raise ValueError('Invalid longitudinal trace detail')
  for name in DETAIL_IDS:
    if detail.get(name) is not None:
      detail[name] = str(uint64_value(detail[name], name, positive=name in ('session', 'sequence', 'planner_session', 'planner_sequence', 'plan_mono')))
  for name in ('sequence', 'plan_mono'):
    if name in detail and (None if detail[name] is None else int(detail[name])) != row[name]:
      raise ValueError(f'Conflicting longitudinal trace detail identity: {name}')
  if detail.get('start_mono') is not None and detail.get('end_mono') is not None and int(detail['end_mono']) < int(detail['start_mono']):
    raise ValueError('Invalid longitudinal trace attempt interval')
  inputs = detail.get('inputs')
  if inputs is not None:
    if not isinstance(inputs, list):
      raise ValueError('Invalid longitudinal trace inputs')
    services = set()
    for item in inputs:
      if not isinstance(item, dict) or not isinstance(item.get('service'), str) or not item['service'] or item['service'] in services:
        raise ValueError('Invalid longitudinal trace input service')
      services.add(item['service'])
      if type(item.get('idKnown')) is not bool:
        raise ValueError('Invalid longitudinal trace input identity status')
      if item['idKnown']:
        item['monoTime'] = str(uint64_value(item.get('monoTime'), 'input monoTime'))
        field = item['service'] + '_mono'
        if row.get(field) is not None and int(item['monoTime']) != row[field]:
          raise ValueError('Conflicting longitudinal trace input identity')
      elif item.get('monoTime') is not None:
        raise ValueError('Unknown longitudinal trace input has an identity')
      known, flags = item.get('knownFlags'), item.get('flags')
      if (type(known) is not int or type(flags) is not int or not 0 <= known <= 65535 or not 0 <= flags <= 65535
          or flags & ~known):
        raise ValueError('Invalid longitudinal trace input health')
  native = detail.get('native')
  if native is not None:
    if not isinstance(native, dict):
      raise ValueError('Invalid longitudinal trace native detail')
    for name in ('solve_calls', 'reset_calls'):
      if native.get(name) is not None:
        native[name] = str(uint64_value(native[name], name))
  return detail


def decode_counters(payload):
  for name in ('dropped', 'failures'):
    if name in payload and (type(payload[name]) is not int or payload[name] < 0):
      raise ValueError(f'Invalid longitudinal trace counter: {name}')
  counters = payload.get('counters')
  if counters is not None:
    if not isinstance(counters, dict):
      raise ValueError('Invalid longitudinal trace counters')
    for name, value in counters.items():
      if name in ('session', 'sequence'):
        counters[name] = str(uint64_value(value, 'counter ' + name, positive=name == 'session'))
      elif name in ('disabled', 'closed'):
        if type(value) is not bool:
          raise ValueError(f'Invalid longitudinal trace counter: {name}')
      elif type(value) is not int or value < 0:
        raise ValueError(f'Invalid longitudinal trace counter: {name}')
  return counters


def decode_trace(record):
  """Keep the flat sample API and exact Python integer identities, with companion evidence."""
  parsed = json.loads(record)
  if not isinstance(parsed, dict):
    return []
  payload = parsed.get('msg')
  if not isinstance(payload, dict) or payload.get('event') != TRACE_EVENT:
    return []
  schema = payload.get('schema')
  if type(schema) is not int or schema not in SCHEMA_FIELDS or payload.get('fields') != list(SCHEMA_FIELDS[schema]):
    raise ValueError('Unsupported longitudinal trace schema')
  fields = SCHEMA_FIELDS[schema]
  samples = payload.get('samples')
  if not isinstance(samples, list) or len(samples) > 40:
    raise ValueError('Invalid longitudinal trace batch')
  validate_finite(payload)
  details = payload.get('details')
  if 'details' in payload or 'detail_schema' in payload:
    if (type(payload.get('detail_schema')) is not int or payload['detail_schema'] != DETAIL_SCHEMA
        or not isinstance(details, list) or len(details) != len(samples)):
      raise ValueError('Invalid longitudinal trace detail batch')
  counters = decode_counters(payload)
  session = payload.get('session')
  if session is not None:
    session = str(uint64_value(session, 'session', positive=True))
  counter_session = counters.get('session') if counters is not None else None
  if session is not None and counter_session is not None and session != counter_session:
    raise ValueError('Conflicting longitudinal trace session')
  session = session if session is not None else counter_session
  for name in ('counter_scope', 'delivery_scope'):
    if name in payload and not isinstance(payload[name], str):
      raise ValueError(f'Invalid longitudinal trace {name}')
  rows = []
  for index, sample in enumerate(samples):
    if not isinstance(sample, list) or len(sample) != len(fields):
      raise ValueError('Truncated longitudinal trace sample')
    if any(value is not None and type(value) not in (str, int, float, bool) for value in sample):
      raise ValueError('Invalid longitudinal trace scalar')
    row = dict(zip(fields, sample, strict=True))
    row['sequence'] = uint64_value(row['sequence'], 'sequence', positive=True)
    for name in fields:
      if name.endswith('_mono') and row[name] is not None:
        row[name] = uint64_value(row[name], name, positive=name == 'plan_mono')
    detail = decode_detail(details[index], row) if details is not None else None
    detail_session = detail.get('session') if detail is not None else None
    if detail_session is not None and session is not None and detail_session != session:
      raise ValueError('Conflicting longitudinal trace session')
    row.update(
      schema=schema, field_contract=list(fields), detail_schema=payload.get('detail_schema'), detail=detail,
      session=detail_session if detail_session is not None else session,
      dropped=payload.get('dropped'), failures=payload.get('failures'), counters=counters,
    )
    for name in ('counter_scope', 'delivery_scope'):
      if name in payload:
        row[name] = payload[name]
    rows.append(row)
  return rows


def trace_json_row(row):
  """Export decimal identity strings without changing the numeric measurement fields."""
  result = dict(row)
  for name, value in row.items():
    if value is not None and (name == 'sequence' or name.endswith('_mono')):
      result[name] = str(uint64_value(value, name))
  return result


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
          output.write(json.dumps(trace_json_row(row), allow_nan=False) + '\n')
          count += 1
  print(json.dumps({'samples': count, 'output': str(arguments.output)}))


if __name__ == '__main__':
  main()
