from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.selfdrive.controls.tests.replay_pedal_full_routes import (
  NumericSink, ROW_FIELDS, contiguous_blocks, ordered_segment_messages, block_messages, preload_parameters,
)
from openpilot.selfdrive.controls.tests.replay_pedal_switches import paired_messages
from openpilot.selfdrive.controls.tests.evaluate_pedal_full_routes import exact_alignment, normal_mask, switch_analysis, unmatched_events, reentry_mask


def message(service, timestamp):
  return SimpleNamespace(which=lambda: service, logMonoTime=int(timestamp * 1e9), longitudinalPlanSP=timestamp)


def test_pairing_preserves_intermediate_control_order():
  messages = [message('longitudinalPlan', 1.0), message('controlsState', 1.001), message('carControl', 1.002),
              message('longitudinalPlanSP', 1.003), message('sendcan', 1.004)]
  actual = list(paired_messages(iter(messages)))
  assert [entry[0] for entry in actual] == messages
  assert actual[0][1] == 1.003
  assert all(decision is None for _, decision in actual[1:])


@pytest.mark.parametrize('boundary', ['next_plan', 'time_limit', 'message_limit', 'eof'])
def test_pairing_does_not_assign_stale_or_missing_decisions(boundary):
  messages = [message('longitudinalPlan', 1.0)]
  if boundary == 'next_plan':
    messages.append(message('longitudinalPlan', 1.05))
  elif boundary == 'time_limit':
    messages.append(message('carControl', 1.3))
  elif boundary == 'message_limit':
    messages += [message('carControl', 1.01)] * 5
  if boundary != 'eof':
    messages.append(message('longitudinalPlanSP', 1.31 if boundary == 'time_limit' else 1.06))
  actual = list(paired_messages(messages, maximum_messages=4))
  assert [entry[0] for entry in actual] == messages
  assert actual[0][1] is None
  if boundary == 'next_plan':
    assert actual[1][1] == 1.06


def test_pairing_streams_before_consuming_the_route():
  def stream():
    yield message('carState', 1.0)
    raise AssertionError('Eagerly consumed the rest of the route')
  assert next(paired_messages(stream()))[0].which() == 'carState'


def test_missing_segment_splits_but_segment_boundaries_do_not():
  segments = [{'segment': number} for number in (3, 1, 0, 6, 4)]
  assert [[segment['segment'] for segment in block] for block in contiguous_blocks(segments)] == [[0, 1], [3, 4], [6]]


def test_numeric_sink_flushes_without_retaining_rows(tmp_path):
  sink = NumericSink(tmp_path, 'route')
  values = dict.fromkeys(ROW_FIELDS['control'], 0.0)
  values.update(t=1.5, mono=123456789012345, state='pid', recorded_state='off')
  sink('control', values)
  sink.flush()
  assert not any(sink.buffers.values())
  assert len(sink.chunks) == 1
  sink.flush()
  assert len(sink.chunks) == 1
  with np.load(tmp_path / sink.chunks[0]['file'], allow_pickle=False) as saved:
    assert saved['control'].shape == (1, len(ROW_FIELDS['control']))
    assert saved['control'][0, 1] == values['mono']
    assert saved['control'][0, -2:].tolist() == [1.0, 0.0]
    assert len(saved['can']) == 0


def test_segment_overlap_is_sorted_without_losing_the_delayed_plan_pair():
  segments = [
    {'first_mono': int(1e9), 'messages': [message('carState', 1.0), message('longitudinalPlan', 2.002)]},
    {'first_mono': int(2e9), 'messages': [message('modelV2', 2.0), message('carControl', 2.003), message('longitudinalPlanSP', 2.004)]},
  ]
  flushes = []
  ordered = ordered_segment_messages(segments, lambda segment: iter(segment['messages']), lambda: flushes.append(True))
  actual = list(paired_messages(ordered))
  assert [entry.logMonoTime for entry, _ in actual] == [int(value * 1e9) for value in (1.0, 2.0, 2.002, 2.003, 2.004)]
  assert actual[2][1] == 2.004
  assert len(flushes) == 2


def test_negative_control_mask_is_fixed_by_original_events_and_pulses():
  times = np.arange(60.0)
  clean = np.ones(len(times), dtype=bool)
  clean[50] = False
  original = [{'start': 20.0, 'end': 25.0}, {'start': 40.0, 'end': 40.1}]
  actual = normal_mask(times, clean, original, margin=2.0)
  expected = clean & ~((times >= 18) & (times <= 27)) & ~((times >= 38) & (times <= 42.1))
  assert np.array_equal(actual, expected)


def test_new_oscillation_in_previously_stable_data_is_detected():
  times = np.arange(100) * 0.02
  clean = np.ones(len(times), dtype=bool)
  stable = switch_analysis(times, np.zeros(100), np.full(100, 7), clean)
  alternating = np.repeat([1, -1] * 5, 10)
  candidate = switch_analysis(times, alternating == -1, np.where(alternating == 1, 7, -30000), clean)
  assert len(candidate['events']) == 1
  assert candidate['events'][0]['switches'] == 9
  assert unmatched_events(candidate['events'], stable['events']) == candidate['events']


def test_short_pulses_remain_raw_release_failures_after_debouncing():
  sides = np.r_[np.full(10, -1), np.repeat([1, -1] * 3, 2), np.full(10, -1)]
  times = np.arange(len(sides)) * 0.02
  clean = np.ones(len(sides), dtype=bool)
  gas = np.where(sides == 1, 7, -30000)
  stable = switch_analysis(times, np.ones(len(sides)), np.full(len(sides), -30000), clean, dwell=0.0)
  raw = switch_analysis(times, sides == -1, gas, clean, dwell=0.0)
  debounced = switch_analysis(times, sides == -1, gas, clean)
  assert debounced['events'] == []
  assert len(raw['events']) == 1
  assert raw['events'][0]['switches'] == 6
  assert unmatched_events(raw['events'], stable['events']) == raw['events']


@pytest.mark.parametrize('boundary', ['override', 'gap', 'neutral', 'conflict'])
def test_bursts_never_rejoin_across_separate_eligible_blocks(boundary):
  sides = np.repeat([1, -1, 1, 1, -1, 1], 12)
  times = np.arange(len(sides)) * 0.02
  clean = np.ones(len(sides), dtype=bool)
  gas = np.where(sides == 1, 7, -30000)
  brake = sides == -1
  if boundary == 'override':
    clean[35:37] = False
  elif boundary == 'gap':
    times[36:] += 0.2
  elif boundary == 'neutral':
    gas[33:39] = 0
    brake[33:39] = False
  else:
    gas[35:37] = 7
    brake[35:37] = True
  result = switch_analysis(times, brake, gas, clean)
  assert len(result['changes']) == 4
  assert result['events'] == []


def test_nearest_native_alignment_rejects_missing_and_duplicate_frames():
  reference = np.array([1.0, 1.02, 1.04])
  assert exact_alignment(reference, np.array([1.0200000001, 1.04])).tolist() == [1, 2]
  with pytest.raises(ValueError):
    exact_alignment(reference, np.array([1.03]))
  with pytest.raises(ValueError):
    exact_alignment(reference, np.array([1.02, 1.02]))
  assert exact_alignment(np.array([]), np.array([])).tolist() == []


def test_reentry_exposure_uses_recorded_activation_only():
  times = np.arange(8.0)
  enabled = np.array([0, 1, 1, 1, 1, 0, 1, 1])
  assert reentry_mask(times, enabled).tolist() == [False, True, True, True, False, False, True, True]


def test_cached_readers_do_not_retain_entire_segment_buffers(tmp_path):
  import openpilot.cereal.messaging as messaging
  from openpilot.tools.lib.logreader import save_log
  from openpilot.selfdrive.controls.tests.replay_pedal_switches import sha256

  source = messaging.new_message('carState')
  source.logMonoTime = int(1e9)
  source.carState.vEgo = 17.5
  cache = tmp_path / 'segment.rlog.zst'
  save_log(str(cache), [source.as_reader()])
  segment = {'cache': cache.name, 'sha256': sha256(cache), 'first_mono': int(1e9)}
  sink = NumericSink(tmp_path, 'route')
  detached = list(block_messages(tmp_path, [segment], sink))
  assert len(detached) == 1
  assert not hasattr(detached[0], '_evt')
  assert detached[0].carState.vEgo == 17.5


@pytest.mark.parametrize('changed', [False, True])
def test_late_configuration_requires_identical_recorded_parameters(tmp_path, changed):
  import openpilot.cereal.messaging as messaging
  from openpilot.tools.lib.logreader import save_log
  from openpilot.selfdrive.controls.tests.replay_pedal_switches import sha256

  blocks = []
  for segment in range(2):
    params = messaging.new_message('carParams')
    params.carParams.mass = 1500.0 + (segment if changed else 0.0)
    params_sp = messaging.new_message('carParamsSP')
    path = tmp_path / f'{segment}.rlog.zst'
    save_log(str(path), [params.as_reader(), params_sp.as_reader()])
    blocks.append({'cache': path.name, 'sha256': sha256(path)})
  messages = [message('modelV2', 1.0)]
  if changed:
    with pytest.raises(ValueError, match='changing configuration'):
      list(preload_parameters(tmp_path, blocks, iter(messages)))
  else:
    recovered = list(preload_parameters(tmp_path, blocks, iter(messages)))
    assert [entry.which() for entry in recovered] == ['carParams', 'carParamsSP', 'modelV2']
    assert recovered[0].carParams.mass == 1500.0
