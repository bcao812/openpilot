"""Verified window inputs and results shared by offline longitudinal tools."""

import json

import numpy as np

from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.longitudinal.evaluate_pedal_full_routes import columns
from openpilot.tools.longitudinal.replay_pedal_full_routes import ROW_FIELDS, block_messages
from openpilot.tools.longitudinal.replay_pedal_switches import sha256


def window_messages(corpus, window, sink):
  seen = set()
  segment = window['parameter_segment']
  cache = corpus / segment['cache']
  if sha256(cache) != segment['sha256']:
    raise ValueError('Changed parameter segment')
  for message in LogReader([str(cache)], sort_by_time=True):
    if message.which() in ('carParams', 'carParamsSP') and message.which() not in seen:
      if message.logMonoTime / 1e9 - window['origin'] > window['warmup']:
        raise ValueError('Parameter arrival is later than sampled warmup')
      seen.add(message.which())
      yield message.as_builder().as_reader()
    if len(seen) == 2:
      break
  if len(seen) != 2:
    raise ValueError('Missing sample parameters')
  for message in block_messages(corpus, window['segments'], sink):
    if message.which() in ('carParams', 'carParamsSP', 'initData'):
      continue
    timestamp = message.logMonoTime / 1e9 - window['origin']
    if timestamp < window['warmup']:
      continue
    if timestamp > window['end'] + 0.25:
      break
    yield message


def load_window(output, mode, identifier):
  root = output / mode
  metadata = json.loads((root / f'{identifier}.json').read_text())
  if metadata['mode'] != mode or metadata['window']['id'] != identifier or metadata['fields'] != {kind: list(fields) for kind, fields in ROW_FIELDS.items()}:
    raise ValueError('Mismatched sampled replay identity or schema')
  for expected, matched in (('plans', 'matched_plans'), ('controls', 'matched_controls'), ('sendcan', 'matched_sendcan')):
    if not metadata['coverage'][expected] or metadata['coverage'][expected] != metadata['coverage'][matched]:
      raise ValueError('Incomplete sampled replay coverage')
  pieces = {kind: [] for kind in ROW_FIELDS}
  for chunk in metadata['chunks']:
    if sha256(root / chunk['file']) != chunk['sha256']:
      raise ValueError('Changed sampled replay results')
    with np.load(root / chunk['file'], allow_pickle=False) as saved:
      for kind in pieces:
        if saved[kind].shape != (chunk['rows'][kind], len(ROW_FIELDS[kind])):
          raise ValueError('Mismatched sampled replay array shape')
        pieces[kind].append(saved[kind])
  for kind, expected in (('plan', 'plans'), ('control', 'controls'), ('can', 'sendcan')):
    if sum(len(piece) for piece in pieces[kind]) != metadata['coverage'][expected]:
      raise ValueError('Missing sampled replay output rows')
  return metadata, {kind: columns(np.concatenate(values), ROW_FIELDS[kind]) for kind, values in pieces.items()}
