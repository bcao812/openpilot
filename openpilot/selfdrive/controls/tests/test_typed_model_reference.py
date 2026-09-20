"""Pure evidence tests; native behavior is covered in test_typed_reference_mpc."""

from dataclasses import replace

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.drive_helpers import STOP_SPEED
from openpilot.selfdrive.controls.lib.model_reference import (
  ModelReferenceAdapter, NoForward, Unknown, moving_preference,
)
from openpilot.selfdrive.modeld.constants import ModelConstants, index_function


SOURCE_NS = 1_000_000_000
TIMES = np.array([index_function(index, max_val=10.0, max_idx=12) for index in range(13)])
MODEL_TIMES = np.array(ModelConstants.T_IDXS)


def evidence(raw, measured_speed=10.0, *, adapter=None, model_ns=SOURCE_NS, anchor_ns=SOURCE_NS, now_ns=SOURCE_NS):
  return (adapter or ModelReferenceAdapter()).build(raw, TIMES, measured_speed,
                                                   model_mono_time_ns=model_ns, anchor_mono_time_ns=anchor_ns,
                                                   consumed_mono_time_ns=now_ns)


def stopping_curve(initial=9.4, stop_time=5.0):
  return initial * np.maximum(1.0 - MODEL_TIMES / stop_time, 0.0)**2


@pytest.mark.parametrize('bias', [-0.6, 0.6, 5.0])
def test_only_moving_preference_is_invariant_when_typed_events_are_fixed(bias):
  raw = 10.0 - 0.5 * MODEL_TIMES
  original = evidence(raw)
  changed = replace(original, moving=moving_preference(raw + bias, TIMES, 10.0), raw_velocity=raw + bias)
  assert changed.events == original.events
  for name in ('position', 'velocity', 'acceleration', 'supported'):
    np.testing.assert_allclose(getattr(changed.moving, name), getattr(original.moving, name), atol=1e-13)


def test_rest_zero_is_independent_of_anchored_moving_preference_and_position_is_nominal():
  reference = evidence(stopping_curve())
  goal = reference.rest_goal
  assert goal is not None and reference.moving.velocity[-1] == pytest.approx(0.6)
  assert goal.native_node is not None and goal.native_time_s == TIMES[goal.native_node]
  assert goal.source_time_interval_s[0] < goal.source_time_interval_s[1]
  assert goal.native_time_s >= goal.source_time_interval_s[1]
  assert goal.nominal_position > 0.0 and goal.position_kind == 'nominal_preference'
  assert goal.position_frame == 'model_publication_proxy_origin'
  assert not reference.allows_departure
  assert reference.source.epoch_basis == 'publication_proxy'
  assert reference.source.physical_prediction_epoch_ns is None


@pytest.mark.parametrize('speed', [0.0, 0.2])
def test_negative_return_to_zero_does_not_create_a_forward_preference(speed):
  reference = evidence(stopping_curve(-0.4, 2.5), speed)
  assert reference.no_forward and reference.moving is None
  assert isinstance(reference.events[0], NoForward)
  assert reference.rest_goal is None  # raw integral zero is not a physical wall
  assert not reference.allows_departure


def test_future_stop_then_go_preserves_current_rest_but_immediate_launch_has_forward_evidence():
  held = evidence(np.where(MODEL_TIMES < 1.0, 0.0, 0.6 * (MODEL_TIMES - 1.0)), 0.0)
  departure = evidence(0.6 * MODEL_TIMES, 0.0)
  assert held.rest_goal.native_node == 0 and held.rest_goal.future_forward
  assert not held.allows_departure
  assert departure.rest_goal is None and departure.allows_departure


def test_backward_then_forward_is_unknown_and_cannot_be_translated_into_launch():
  reference = evidence(-0.4 + 0.6 * MODEL_TIMES, 0.2)
  assert isinstance(reference.unknown, Unknown)
  assert reference.unknown.reason == 'backward_then_forward_is_ambiguous'
  assert reference.moving is None and not reference.allows_departure


@pytest.mark.parametrize('node', [16, len(MODEL_TIMES) - 1])
def test_isolated_or_single_terminal_zero_cannot_create_a_hard_rest_goal(node):
  raw = np.full(len(MODEL_TIMES), 10.0)
  raw[node] = 0.0
  reference = evidence(raw)
  assert reference.unknown.reason == 'isolated_zero_has_no_rest_support'
  assert reference.rest_goal is None and reference.moving is None
  assert not reference.allows_departure


def test_repeated_model_identity_keeps_event_epoch_and_loses_elapsed_time():
  adapter = ModelReferenceAdapter()
  raw = stopping_curve()
  before = evidence(raw, adapter=adapter)
  after = evidence(raw, adapter=adapter, anchor_ns=SOURCE_NS + 50_000_000, now_ns=SOURCE_NS + 60_000_000)
  assert before.rest_goal.event_proxy_epoch_ns == after.rest_goal.event_proxy_epoch_ns
  assert after.rest_goal.remaining_time_s == pytest.approx(before.rest_goal.remaining_time_s - 0.05)
  assert before.rest_goal.nominal_position == after.rest_goal.nominal_position
  assert not after.moving.supported[-1]


@pytest.mark.parametrize('model_ns,anchor_ns,now_ns,reason', [
  (None, SOURCE_NS, SOURCE_NS, 'missing_publication_proxy'),
  (SOURCE_NS + 1, SOURCE_NS, SOURCE_NS, 'future_publication_proxy'),
  (SOURCE_NS, SOURCE_NS + 300_000_000, SOURCE_NS + 300_000_000, 'expired_model_publication'),
])
def test_unknown_time_does_not_invent_a_prediction_epoch(model_ns, anchor_ns, now_ns, reason):
  reference = evidence(0.6 * MODEL_TIMES, 0.0, model_ns=model_ns, anchor_ns=anchor_ns, now_ns=now_ns)
  assert reference.unknown.reason == reason
  assert reference.source.physical_prediction_epoch_ns is None
  assert not reference.allows_departure


def test_fresh_later_model_publication_has_future_preference_without_claiming_initial_alignment():
  reference = evidence(0.03 + 0.6 * MODEL_TIMES, 0.0, model_ns=SOURCE_NS + 5_000_000,
                       anchor_ns=SOURCE_NS, now_ns=SOURCE_NS + 10_000_000)
  assert reference.unknown.reason == 'model_proxy_after_initial_state'
  assert reference.has_future_moving_preference
  assert not reference.moving.supported[0] and np.all(reference.moving.supported[1:])
  assert reference.source.proxy_age_s is None and reference.source.physical_prediction_epoch_ns is None
  assert reference.raw_motion.kind == 'forward'
  assert reference.allows_departure and reference.allows_external_departure


@pytest.mark.parametrize('delay_ns', [1, 5_000_000, 100_000_000])
@pytest.mark.parametrize('bias', [-0.6, 0.0, 0.6])
def test_future_preference_is_bias_invariant_only_on_source_supported_intervals(delay_ns, bias):
  delay = delay_ns / 1e9
  raw = 10.0 + bias + 0.6 * MODEL_TIMES
  reference = evidence(raw, 10.0, model_ns=SOURCE_NS + delay_ns,
                       anchor_ns=SOURCE_NS, now_ns=SOURCE_NS + delay_ns + 10_000_000)
  assert reference.events == (Unknown('model_proxy_after_initial_state'),)
  assert reference.source.proxy_age_s is None and reference.source.physical_prediction_epoch_ns is None
  assert reference.raw_motion.kind == 'forward' and reference.allows_departure
  supported = TIMES >= delay
  supported[:-1] &= supported[1:]
  np.testing.assert_array_equal(reference.moving.supported, supported)
  assert not reference.moving.supported[0]
  np.testing.assert_allclose(reference.moving.velocity[supported], 10.0 + 0.6 * (TIMES[supported] - delay), atol=1e-13)
  np.testing.assert_allclose(reference.moving.acceleration[:-1][supported[:-1]], 0.6, atol=1e-13)
  assert np.all(np.isfinite([reference.moving.position, reference.moving.velocity, reference.moving.acceleration]))


@pytest.mark.parametrize('raw_speed', [9.4, 10.0, 10.6])
def test_future_constant_preference_does_not_create_model_departure_permission(raw_speed):
  reference = evidence(np.full(len(MODEL_TIMES), raw_speed), 10.0, model_ns=SOURCE_NS + 5_000_000,
                       anchor_ns=SOURCE_NS, now_ns=SOURCE_NS + 10_000_000)
  assert reference.has_future_moving_preference
  assert reference.raw_motion.kind == 'moving'
  assert not reference.allows_departure
  np.testing.assert_allclose(reference.moving.velocity, 10.0, atol=1e-13)
  np.testing.assert_allclose(reference.moving.acceleration, 0.0, atol=1e-13)


@pytest.mark.parametrize('speed', [STOP_SPEED - 0.001, STOP_SPEED, STOP_SPEED + 0.001])
def test_future_nonforward_preference_preserves_existing_stop_band_compatibility(speed):
  raw = np.full(len(MODEL_TIMES), 1.0)
  reference = evidence(raw, speed, model_ns=SOURCE_NS + 5_000_000,
                       anchor_ns=SOURCE_NS, now_ns=SOURCE_NS + 10_000_000)
  assert reference.events == (Unknown('model_proxy_after_initial_state'),)
  assert reference.raw_motion.kind == 'moving'
  assert not reference.allows_departure and reference.allows_external_departure
  np.testing.assert_array_equal(reference.raw_velocity, raw)
  if speed < STOP_SPEED:
    assert reference.moving is None and not reference.has_future_moving_preference
  else:
    assert reference.has_future_moving_preference
    np.testing.assert_allclose(reference.moving.velocity, speed, atol=1e-13)


@pytest.mark.parametrize('kind', ['negative_prefix', 'no_forward', 'rest', 'single_zero'])
def test_later_publication_does_not_erase_raw_stop_or_ambiguity_veto(kind):
  raw = -0.015 + 0.6 * MODEL_TIMES
  if kind == 'no_forward':
    raw = stopping_curve(-0.4, 2.5)
  elif kind == 'rest':
    raw = stopping_curve(0.2, 2.5)
  elif kind == 'single_zero':
    raw = 0.2 + 0.6 * MODEL_TIMES
    raw[16] = 0.0
  reference = evidence(raw, 0.2, model_ns=SOURCE_NS + 5_000_000,
                       anchor_ns=SOURCE_NS, now_ns=SOURCE_NS + 10_000_000)
  assert reference.unknown.reason == 'model_proxy_after_initial_state'
  assert reference.moving is None and reference.rest_goal is None
  assert not reference.allows_departure and not reference.allows_external_departure
  if kind == 'rest':
    assert reference.has_rest_intent
  elif kind == 'no_forward':
    assert reference.no_forward
  else:
    assert reference.raw_motion.unknown is not None


def test_same_identity_mutation_and_regression_are_unknown_without_poisoning_next_source():
  adapter = ModelReferenceAdapter()
  first = evidence(0.6 * MODEL_TIMES, 0.0, adapter=adapter)
  changed = evidence(0.8 * MODEL_TIMES, 0.0, adapter=adapter)
  regressed = evidence(0.6 * MODEL_TIMES, 0.0, adapter=adapter, model_ns=SOURCE_NS - 1)
  next_reference = evidence(0.8 * MODEL_TIMES, 0.0, adapter=adapter,
                            model_ns=SOURCE_NS + 50_000_000, anchor_ns=SOURCE_NS + 50_000_000,
                            now_ns=SOURCE_NS + 50_000_000)
  assert first.allows_departure and next_reference.allows_departure
  assert changed.unknown.reason == 'model_mutated_with_same_identity'
  assert regressed.unknown.reason == 'model_publication_regressed'


def test_negative_anchored_preference_is_not_clipped_or_reclassified():
  raw = 5.0 - 0.4 * MODEL_TIMES
  reference = evidence(raw, 0.2)
  assert reference.rest_goal is None and reference.moving.velocity[-1] < 0.0
  np.testing.assert_allclose(reference.moving.velocity, 0.2 + np.interp(TIMES, MODEL_TIMES, raw) - raw[0])


def test_inputs_are_immutable_snapshots():
  raw = stopping_curve()
  reference = evidence(raw)
  raw[:] = 40.0
  assert reference.raw_velocity[-1] == 0.0
  for values in (reference.raw_velocity, reference.moving.velocity, reference.moving.supported):
    with pytest.raises(ValueError):
      values[0] = 1
