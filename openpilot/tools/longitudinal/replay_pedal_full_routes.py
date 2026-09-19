#!/usr/bin/env python3
"""Replay every readable local segment with continuous state, including negative controls.

Only derived files below --output are written. Segment-sized input and output buffers
bound memory; segmentation never resets a planner. Missing segments or telemetry gaps
reset the chain explicitly. Perception, SP/DEC and recorded vehicle motion remain fixed.
"""

import argparse
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gc
import hashlib
from heapq import merge
import inspect
import json
from pathlib import Path
import re
import warnings

import numpy as np
import zstandard

from openpilot.selfdrive.controls.lib import longitudinal_planner as planner_module
from openpilot.tools.longitudinal import replay_pedal_switches as replay
from openpilot.tools.lib.logreader import LogReader


SCHEMA = 1
NATIVE_FIELDS = (
  't', 'seg', 'enabled', 'cc_enabled', 'long_active', 'long_state',
  'gas_pressed', 'brake_pressed', 'override', 'standstill', 'force_decel',
  'experimental', 'dec', 'allow_throttle', 'lead1', 'lead2', 'source',
  'sp_cruise', 'should_stop', 'model_stop', 'v_ego', 'a_ego', 'v_cruise',
  'sp_v_target', 'accel_cmd', 'a_target', 'model_accel', 'mpc_accel',
  'planned_v', 'lead_drel', 'lead_vrel', 'lead_alead', 'lead2_drel',
  'lead2_vrel', 'steer', 'can_accel', 'can_gas', 'can_brake', 'can_on',
  'can_stop', 'can_counter', 'cs_age', 'cc_age', 'sd_age', 'plan_age',
  'radar_age', 'model_age', 'cs_valid', 'cc_valid', 'sd_valid',
  'plan_valid', 'radar_valid', 'model_valid', 'can_valid',
)
ROW_FIELDS = {
  'plan': ('t', 'mono', 'accel', 'stop', 'recorded_accel', 'recorded_stop', 'speed', 'recorded_speed',
           'solver_status', 'source', 'cruise_accel', 'v_ego', 'lead', 'automatic', 'valid', 'model_accel',
           'model_stop', 'fcw', 'force_decel', 'allow_throttle', 'sp_cruise'),
  'control': ('t', 'mono', 'plan_mono', 'accel', 'recorded_accel', 'state', 'recorded_state'),
  'can': ('t', 'mono', 'automatic', 'can_accel', 'can_brake', 'can_gas', 'recorded_accel', 'recorded_brake',
          'recorded_gas', 'can_on', 'recorded_on', 'can_stop', 'recorded_can_stop'),
}
ENUMS = {'off': 0, 'pid': 1, 'stopping': 2, 'starting': 3}
ENUMS.update(planner_module.LongitudinalPlanSource.schema.enumerants)
ENUMS.update({str(value): value for value in set(ENUMS.values())})


def contiguous_blocks(segments):
  blocks = []
  for segment in sorted(segments, key=lambda entry: entry['segment']):
    if not blocks or segment['segment'] != blocks[-1][-1]['segment'] + 1:
      blocks.append([])
    blocks[-1].append(segment)
  return blocks


def prepare_segment(arguments):
  logs, output, route, commit, entry = arguments
  relative, size, modified = entry
  source = logs / relative
  initial = source.stat()
  if (initial.st_size, initial.st_mtime_ns) != (size, modified):
    raise ValueError(f'Native audit input changed: {source}')
  segment = int(source.parent.name.rsplit('--', 1)[1])
  cache = f'{route}--{segment}.rlog.zst'
  destination = output / cache
  metadata_file = output / f'{route}--{segment}.json'
  identity = {'path': relative, 'size': size, 'mtime_ns': modified, 'sha256': replay.sha256(source)}
  helper_hash = replay.sha256(Path(__file__))
  if metadata_file.exists() and destination.exists():
    saved = json.loads(metadata_file.read_text())
    if saved['input'] == identity and saved['helper_sha256'] == helper_hash and replay.sha256(destination) == saved['sha256']:
      return saved
  services = Counter()
  first_time = last_time = None
  recording_seen = False
  with warnings.catch_warnings(record=True) as captured, destination.with_suffix('.tmp').open('wb') as stream:
    warnings.simplefilter('always')
    with zstandard.ZstdCompressor(level=3).stream_writer(stream, closefd=False) as writer:
      for message in LogReader([str(source)], sort_by_time=True):
        service = message.which()
        if service == 'initData':
          if message.initData.gitCommit != commit or message.initData.dirty:
            raise ValueError(f'Unexpected recording version: {source}')
          recording_seen = True
        if service not in replay.RETAINED | replay.PARAMETERS:
          continue
        if service not in replay.PARAMETERS:
          timestamp = message.logMonoTime
          first_time = timestamp if first_time is None else min(first_time, timestamp)
          last_time = timestamp if last_time is None else max(last_time, timestamp)
        writer.write(message.as_builder().to_bytes())
        services[service] += 1
  if not recording_seen:
    raise ValueError(f'Missing initData: {source}')
  final = source.stat()
  if (final.st_size, final.st_mtime_ns) != (size, modified):
    raise ValueError(f'Log changed during extraction: {source}')
  destination.with_suffix('.tmp').replace(destination)
  result = {'route': route, 'segment': segment, 'input': identity, 'cache': cache, 'sha256': replay.sha256(destination),
            'helper_sha256': helper_hash, 'extractor_sha256': hashlib.sha256(inspect.getsource(prepare_segment).encode()).hexdigest(),
            'first_mono': first_time, 'last_mono': last_time, 'services': services,
            'warnings': [str(item.message) for item in captured]}
  replay.save_json(metadata_file, result)
  print(f'prepared {route}--{segment}', flush=True)
  return result


def prepare(logs, audit, output, workers):
  audit_manifest = json.loads((audit / 'manifest.json').read_text())
  routes, jobs, expected = [], [], set()
  output.mkdir(parents=True, exist_ok=True)
  for item in audit_manifest['routes']:
    route = item['prefix']
    metadata_path = audit / f'{route}.native.json'
    metadata = json.loads(metadata_path.read_text())
    if metadata['schema'] != 1 or metadata['commit'] != item['commit']:
      raise ValueError(f'Inconsistent native audit: {route}')
    event_file = audit / f'{route}.events.json'
    events = json.loads(event_file.read_text())
    record = {'route': route, 'commit': item['commit'], 'origin': metadata['t0_mono'],
              'native_enabled_s': metadata['native_enabled_s'], 'original_events': len(events['events']),
              'audit_hashes': {path.name: replay.sha256(path) for path in (metadata_path, event_file, audit / f'{route}.native.npz')}}
    routes.append(record)
    for entry in metadata['fingerprint']:
      expected.add(entry[0])
      jobs.append((logs, output, route, item['commit'], entry))
  present = {str(path.relative_to(logs)) for path in logs.glob('*--*/rlog.zst')
             if re.fullmatch(r'[0-9a-f]{8}--[0-9a-f]{10}--\d+', path.parent.name)}
  if present != expected:
    raise ValueError(f'Audit does not cover disk inventory: missing={sorted(expected - present)}, new={sorted(present - expected)}')
  with ProcessPoolExecutor(max_workers=workers) as pool:
    segments = list(pool.map(prepare_segment, jobs))
  for record in routes:
    record['blocks'] = contiguous_blocks([segment for segment in segments if segment['route'] == record['route']])
  replay.save_json(output / 'manifest.json', {'schema': SCHEMA, 'routes': routes, 'audit': str(audit), 'logs': str(logs),
                                            'audit_manifest_sha256': replay.sha256(audit / 'manifest.json'),
                                            'helper_sha256': replay.sha256(Path(__file__)), 'native_fields': NATIVE_FIELDS})


class NumericSink:
  def __init__(self, output, route):
    self.output, self.route = output, route
    self.buffers = {kind: array('d') for kind in ROW_FIELDS}
    self.chunks = []

  def __call__(self, kind, row):
    self.buffers[kind].extend(ENUMS[row[field]] if isinstance(row[field], str) else row[field] for field in ROW_FIELDS[kind])

  def flush(self):
    if not any(self.buffers.values()):
      return
    filename = f'{self.route}-{len(self.chunks):04d}.npz'
    values = {kind: np.asarray(values).reshape((-1, len(ROW_FIELDS[kind]))) for kind, values in self.buffers.items()}
    with (self.output / filename).with_suffix('.tmp').open('wb') as stream:
      np.savez_compressed(stream, **values)
    (self.output / filename).with_suffix('.tmp').replace(self.output / filename)
    self.chunks.append({'file': filename, 'rows': {kind: len(rows) for kind, rows in values.items()},
                        'sha256': replay.sha256(self.output / filename)})
    self.buffers = {kind: array('d') for kind in ROW_FIELDS}


def ordered_segment_messages(block, read_segment, flush):
  pending = []
  for index, segment in enumerate(block):
    next_time = min((entry['first_mono'] for entry in block[index + 1:] if entry['first_mono'] is not None), default=float('inf'))
    trailing = []
    for message in merge(pending, read_segment(segment), key=lambda event: event.logMonoTime):
      if message.logMonoTime < next_time:
        yield message
      else:
        trailing.append(message)
    pending = trailing
    flush()
  yield from pending


def block_messages(corpus, block, sink):
  """Detach each event so retained service readers cannot pin a full log segment."""
  parameters_seen = set()

  def read_segment(segment):
    cache = corpus / segment['cache']
    if replay.sha256(cache) != segment['sha256']:
      raise ValueError(f'Changed segment cache: {cache}')
    for message in LogReader([str(cache)], sort_by_time=True):
      service = message.which()
      if service in replay.PARAMETERS:
        if service in parameters_seen:
          continue
        parameters_seen.add(service)
      yield message.as_builder().as_reader()

  yield from ordered_segment_messages(block, read_segment, sink.flush)


def preload_parameters(corpus, block, messages):
  """Recover a late repeated static configuration only after proving block-wide invariance."""
  parameters, identities = {}, {}
  for segment in block:
    if replay.sha256(corpus / segment['cache']) != segment['sha256']:
      raise ValueError(f'Changed parameter cache: {segment["cache"]}')
    for message in LogReader([str(corpus / segment['cache'])], sort_by_time=True):
      service = message.which()
      if service not in ('carParams', 'carParamsSP'):
        continue
      identity = getattr(message, service).as_builder().to_bytes()
      if service in identities and identities[service] != identity:
        raise ValueError(f'Cannot preload changing configuration: {segment["cache"]}/{service}')
      identities[service] = identity
      parameters[service] = message.as_builder().as_reader()
  if set(parameters) != {'carParams', 'carParamsSP'}:
    raise ValueError('Cannot preload missing vehicle configuration')
  yield from parameters.values()
  yield from messages


def run_one(arguments):
  corpus, output, route, recorded_plan, *options = arguments
  preload = bool(options and options[0])
  unified = bool(len(options) > 1 and options[1])
  supervised = bool(len(options) > 2 and options[2])
  mode = replay.selected_mode(unified, supervised)
  sources = {'planner_sha256': replay.sha256(Path(planner_module.__file__)),
             'replayer_sha256': replay.sha256(Path(replay.__file__)), 'helper_sha256': replay.sha256(Path(__file__))}
  sink = NumericSink(output, route['route'])
  blocks = []
  for index, segments in enumerate(route['blocks']):
    window = {'route': route['route'], 'origin': route['origin'], 'start': -1e6, 'end': 1e6,
              'cache': f'{route["route"]}-block-{index}', 'segments': [segment['segment'] for segment in segments]}
    messages = block_messages(corpus, segments, sink)
    if preload:
      messages = preload_parameters(corpus, segments, messages)
    result = replay.replay_window(None, window, recorded_plan=recorded_plan, messages=messages, row_sink=sink, strict=False,
                                  unified=unified, supervised=supervised)
    for key in ('plan_rows', 'control_rows', 'can_rows', 'planner_error', 'switches'):
      result.pop(key)
    blocks.append(result)
  sink.flush()
  result = {'schema': SCHEMA, 'route': route['route'], 'blocks': blocks, 'chunks': sink.chunks,
            'fields': ROW_FIELDS, 'experimental': False, 'recorded_plan': recorded_plan, 'preloaded_parameters': preload,
            'requested_mode': mode,
            'manifest_sha256': replay.sha256(corpus / 'manifest.json'), **sources}
  for key, path in [('planner_sha256', Path(planner_module.__file__)), ('replayer_sha256', Path(replay.__file__)), ('helper_sha256', Path(__file__))]:
    if replay.sha256(path) != sources[key]:
      raise ValueError(f'Source changed during replay: {path}')
  replay.save_json(output / f'{route["route"]}.json', result)
  print(f'replayed {route["route"]}: {sum(chunk["rows"]["can"] for chunk in sink.chunks)} CAN rows', flush=True)
  gc.collect()


def run(corpus, output, workers, route_filter, commit_filter, *, recorded_plan=False, preload=False, unified=False, supervised=False):
  mode = replay.selected_mode(unified, supervised)
  manifest = json.loads((corpus / 'manifest.json').read_text())
  if manifest['schema'] != SCHEMA:
    raise ValueError('Unsupported full-route schema')
  output.mkdir(parents=True, exist_ok=True)
  jobs = [(corpus, output, route, recorded_plan, preload, unified, supervised) for route in manifest['routes']
          if (route_filter is None or route['route'].startswith(route_filter)) and (commit_filter is None or route['commit'].startswith(commit_filter))]
  with ProcessPoolExecutor(max_workers=workers) as pool:
    list(pool.map(run_one, jobs))
  replay.save_json(output / 'run.json', {'routes': [job[2]['route'] for job in jobs], 'experimental': False,
                                       'recorded_plan': recorded_plan, 'preloaded_parameters': preload, 'fields': ROW_FIELDS,
                                       'requested_mode': mode,
                                       'planner_file': inspect.getfile(planner_module), 'corpus': str(corpus)})


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest='command', required=True)
  preparer = commands.add_parser('prepare')
  preparer.add_argument('--logs', type=Path, required=True)
  preparer.add_argument('--audit', type=Path, required=True)
  runner = commands.add_parser('run')
  runner.add_argument('--corpus', type=Path, required=True)
  runner.add_argument('--route')
  runner.add_argument('--commit')
  runner.add_argument('--recorded-plan', action='store_true')
  runner.add_argument('--preload-parameters', action='store_true')
  modes = runner.add_mutually_exclusive_group()
  modes.add_argument('--unified', action='store_true')
  modes.add_argument('--supervised', action='store_true', help='Replay a historical source tree implementing SUPERVISED')
  for command in (preparer, runner):
    command.add_argument('--output', type=Path, required=True)
    command.add_argument('--workers', type=int, default=1)
  arguments = parser.parse_args()
  if arguments.command == 'prepare':
    prepare(arguments.logs, arguments.audit, arguments.output, arguments.workers)
  else:
    run(arguments.corpus, arguments.output, arguments.workers, arguments.route, arguments.commit,
        recorded_plan=arguments.recorded_plan, preload=arguments.preload_parameters, unified=arguments.unified, supervised=arguments.supervised)


if __name__ == '__main__':
  main()
