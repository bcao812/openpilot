"""Receipt-scoped decoding must not reuse transport or control authority."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls import controlsd
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.motion_supervisor import DEGRADED_ACCEL, MAX_CONTROL_GAP_NS, StopReason
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_MAX_AGE_NS, ReferenceTimingProducer
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import CONTROL_STEP_NS, ControlsHarness
from openpilot.selfdrive.modeld.constants import ModelConstants


def make_harness():
  harness = ControlsHarness()
  harness.control.CP.longitudinalActuatorDelay = 0.15
  harness.plan.speeds = [10.0 + 0.5 * float(t) for t in ModelConstants.T_IDXS[:CONTROL_N]]
  harness.plan.accels = [0.5] * CONTROL_N
  harness.plan.reference.accel = 0.5
  harness.plan.reference.constraintAccel = 2.0
  return harness


def test_received_plan_is_decoded_once_while_every_control_frame_consumes_its_timing():
  harness = make_harness()
  with patch.object(controlsd.MotionIntent, 'from_message', wraps=controlsd.MotionIntent.from_message) as decode, \
       patch.object(controlsd, 'get_departure_accel_from_plan', wraps=controlsd.get_departure_accel_from_plan) as departure, \
       patch.object(controlsd._ReferenceTimingFields, 'from_message', wraps=controlsd._ReferenceTimingFields.from_message) as timing, \
       patch.object(harness.control.reference_timing, 'consume', wraps=harness.control.reference_timing.consume) as consume:
    harness.step()
    cached = harness.control._longitudinal_plan_cache
    assert cached is not None
    for _ in range(4):
      command, motion = harness.step(new_plan=False)
      assert harness.control._longitudinal_plan_cache is cached
      assert command.longActive and command.actuators.accel > 0.0
      assert motion.referenceTimingValid and not motion.referenceSourceAdvanced
      assert motion.referenceConsumerMonoTime == harness.now_ns
    assert decode.call_count == departure.call_count == timing.call_count == 1
    assert consume.call_count == 5
    assert all(call.args[0] is cached.reference_timing for call in consume.call_args_list)
    with pytest.raises(AttributeError):
      cached.reference_timing.sourceCarStateMonoTime = harness.now_ns
    harness.step()
    assert harness.control._longitudinal_plan_cache is not cached
    assert decode.call_count == departure.call_count == timing.call_count == 2
    assert consume.call_count == 6


@pytest.mark.parametrize('mutation', ['target', 'intent', 'trajectory', 'reference'])
def test_new_receipt_with_same_timestamp_redecodes_and_rejects_a_changed_payload(mutation):
  harness = make_harness()
  harness.hold()
  origin = harness.now_ns
  cached = harness.control._longitudinal_plan_cache
  if mutation == 'target':
    harness.plan.aTarget = 1.0
  elif mutation == 'intent':
    harness.plan.motionIntent.reasons = 0
  elif mutation == 'trajectory':
    harness.plan.speeds[0] += 0.1
  else:
    harness.plan.reference.accel += 0.1
  command, motion = harness.step(plan_time_ns=origin, model_time_ns=origin)
  assert harness.control._longitudinal_plan_cache is not cached
  assert motion.referenceTimingReason == 'changed_producer_identity'
  assert motion.degraded and motion.resolvedStop
  assert command.actuators.accel <= -2.0 and not command.cruiseControl.resume
  _, motion = harness.step()
  assert motion.referenceTimingValid and not motion.degraded


def test_cached_plan_source_expires_and_cannot_recover_without_new_evidence():
  harness = make_harness()
  source_ns = harness.now_ns
  harness.step(reference_source_ns=source_ns)
  cached = harness.control._longitudinal_plan_cache
  for _ in range(REFERENCE_MAX_AGE_NS // CONTROL_STEP_NS - 1):
    _, motion = harness.step(new_plan=False)
    assert motion.referenceTimingValid and not motion.degraded
  assert harness.now_ns - source_ns == REFERENCE_MAX_AGE_NS
  for expired_frame in range(3):
    command, motion = harness.step(new_plan=False)
    assert harness.control._longitudinal_plan_cache is cached
    assert not motion.referenceTimingValid and motion.degraded
    if expired_frame == 0:
      assert motion.referenceTimingReason == 'expired_reference_source'
    assert command.actuators.accel <= DEGRADED_ACCEL + 1e-6
    assert not command.cruiseControl.resume
  command, motion = harness.step()
  assert command.longActive and motion.referenceTimingValid and not motion.degraded


@pytest.mark.parametrize('fault', ['control_gap', 'clock_rollback'])
def test_cached_plan_still_checks_control_gap_and_clock_high_water(fault):
  harness = make_harness()
  harness.step()
  origin = harness.now_ns
  cached = harness.control._longitudinal_plan_cache
  if fault == 'control_gap':
    command, motion = harness.step(new_plan=False, now_ns=origin + MAX_CONTROL_GAP_NS + 1)
    assert motion.referenceTimingValid
    recovery_ns = harness.now_ns + CONTROL_STEP_NS
  else:
    harness.step(new_plan=False, now_ns=origin + 2 * CONTROL_STEP_NS)
    command, motion = harness.step(new_plan=False, now_ns=origin + CONTROL_STEP_NS)
    assert motion.referenceTimingReason == 'invalid_consumer_or_availability_time'
    recovery_ns = origin + 3 * CONTROL_STEP_NS
  assert harness.control._longitudinal_plan_cache is cached
  assert motion.degraded and not command.cruiseControl.resume
  assert command.actuators.accel <= DEGRADED_ACCEL + 1e-6
  command, motion = harness.step(now_ns=recovery_ns)
  assert command.longActive and motion.referenceTimingValid and not motion.degraded


def test_restarted_producer_needs_a_new_source_and_cannot_reuse_a_retired_instance():
  harness = make_harness()
  harness.step()
  source_ns = harness.now_ns
  old_instance = harness.reference_producer.instance_ns
  cached = harness.control._longitudinal_plan_cache
  harness.step(new_plan=False)
  assert harness.control._longitudinal_plan_cache is cached
  harness.reference_producer = ReferenceTimingProducer()
  command, motion = harness.step(reference_source_ns=source_ns)
  assert motion.referenceTimingReason == 'restart_requires_new_source'
  assert motion.degraded and command.actuators.accel <= DEGRADED_ACCEL + 1e-6
  _, motion = harness.step()
  assert motion.referenceTimingValid and not motion.degraded
  command, motion = harness.step(reference_fields={'producerInstanceMonoTime': old_instance},
                                 intent_fields={'producerInstanceMonoTime': old_instance})
  assert motion.referenceTimingReason == 'retired_producer_instance'
  assert motion.degraded and command.actuators.accel <= DEGRADED_ACCEL + 1e-6
  assert not command.cruiseControl.resume
  _, motion = harness.step()
  assert motion.referenceTimingValid and not motion.degraded


@pytest.mark.parametrize('authority', ['gas', 'disable'])
def test_cached_plan_cannot_restore_authority_after_a_driver_override(authority):
  harness = make_harness()
  harness.step()
  cached = harness.control._longitudinal_plan_cache
  if authority == 'gas':
    harness.car_state.gasPressed = True
  else:
    harness.selfdrive_state.enabled = False
  command, motion = harness.step(new_plan=False)
  assert harness.control._longitudinal_plan_cache is cached
  assert not command.longActive and command.actuators.accel == 0.0
  assert motion.phase == 'off' and not command.cruiseControl.resume
  harness.car_state.gasPressed = False
  harness.selfdrive_state.enabled = True
  command, motion = harness.step(new_plan=False)
  # A healthy override waits for a fresh plan without reporting an input
  # fault. The existing owner still constrains acceleration and resume.
  assert harness.control._longitudinal_plan_cache is cached
  assert command.longActive and motion.waitingForFreshInput and not motion.degraded
  assert command.actuators.accel <= DEGRADED_ACCEL + 1e-6
  assert not command.cruiseControl.resume
  command, motion = harness.step()
  assert command.longActive and not motion.waitingForFreshInput and not motion.degraded


def test_actuator_delay_change_invalidates_projection_and_requires_new_plan_identity():
  harness = make_harness()
  harness.step()
  cached = harness.control._longitudinal_plan_cache
  harness.control.CP.longitudinalActuatorDelay += 0.05
  command, motion = harness.step(new_plan=False)
  assert harness.control._longitudinal_plan_cache is not cached
  assert motion.referenceTimingReason == 'changed_producer_identity'
  assert motion.degraded and command.actuators.accel <= DEGRADED_ACCEL + 1e-6
  _, motion = harness.step()
  assert motion.referenceTimingValid and not motion.degraded


@pytest.mark.parametrize('kind', ['builder', 'builder_reader', 'namespace'])
def test_mutable_input_is_redecoded_even_when_reader_and_receipt_are_unchanged(kind):
  harness = make_harness()
  harness.step()
  if kind == 'namespace':
    builder = SimpleNamespace(**harness.plan.to_dict())
    builder.motionIntent = SimpleNamespace(**builder.motionIntent)
    builder.reference = SimpleNamespace(**builder.reference)
  else:
    builder = harness.plan
  message = builder.as_reader() if kind == 'builder_reader' else builder
  first, first_identity = harness.control._decode_longitudinal_plan(message)
  builder.motionIntent.reasons = int(StopReason.MPC)
  builder.aTarget = -0.5
  second, second_identity = harness.control._decode_longitudinal_plan(message)
  assert harness.control._longitudinal_plan_cache is None
  assert first.reasons == 0 and second.reasons == int(StopReason.MPC)
  assert first_identity[1] == 0.5 and second_identity[1] == -0.5


@pytest.mark.parametrize('ownership', ['missing_api', 'unknown_parent'])
def test_unknown_reader_ownership_falls_back_to_decoding_on_every_frame(ownership):
  harness = make_harness()
  harness.step()
  message = harness.control.sm['longitudinalPlan']
  harness.control._longitudinal_plan_cache = None
  owner_patch = (patch.object(controlsd.capnp.lib.capnp, '_MessageReader', None) if ownership == 'missing_api'
                 else patch.object(message, '_parent', None))
  with owner_patch, patch.object(controlsd.MotionIntent, 'from_message', wraps=controlsd.MotionIntent.from_message) as decode:
    for _ in range(2):
      intent, _ = harness.control._decode_longitudinal_plan(message)
      assert intent.is_valid()
      assert harness.control._longitudinal_plan_cache is None
    assert decode.call_count == 2


@pytest.mark.parametrize('backing_kind', ['bytearray', 'readonly_memoryview'])
@pytest.mark.parametrize('mutation', ['target', 'reference_source'])
def test_reader_with_mutable_backing_redecodes_and_rejects_mutation_without_a_new_receipt(backing_kind, mutation):
  harness = make_harness()
  harness.hold()
  event = harness.events['longitudinalPlan']
  event.clear_write_flag()
  backing = bytearray(event.to_bytes())
  buffer = memoryview(backing).toreadonly() if backing_kind == 'readonly_memoryview' else backing
  with log.Event.from_bytes(buffer) as reader:
    # Receive the duplicate at the next control tick, preserving its Event
    # identity. Two receives at the same wall time would give FrequencyTracker
    # an artificial zero interval before the controller can examine the input.
    receipt_ns = harness.now_ns + CONTROL_STEP_NS
    harness.control.sm.update_msgs(receipt_ns / 1e9, [reader])
    harness.step(new_plan=False)
    assert harness.now_ns == receipt_ns
    assert harness.control._longitudinal_plan_cache is None
    receipt = harness.control.sm.recv_frame['longitudinalPlan']
    message = harness.control.sm['longitudinalPlan']
    if mutation == 'target':
      harness.plan.aTarget = 1.0
    else:
      harness.plan.reference.sourceAEgo = 0.125
    event.clear_write_flag()
    changed = event.to_bytes()
    assert len(changed) == len(backing)
    backing[:] = changed
    assert (message.aTarget == 1.0 if mutation == 'target' else message.reference.sourceAEgo == 0.125)
    command, motion = harness.step(new_plan=False)
    assert harness.control.sm['longitudinalPlan'] is message
    assert harness.control.sm.recv_frame['longitudinalPlan'] == receipt
    assert harness.control._longitudinal_plan_cache is None
    expected_reason = 'changed_producer_identity' if mutation == 'target' else 'changed_source_identity'
    assert motion.referenceTimingReason == expected_reason
    assert motion.degraded and command.actuators.accel <= -2.0
    assert not command.cruiseControl.resume


def test_bytes_pin_with_an_unknown_additional_reference_does_not_qualify_for_caching():
  harness = make_harness()
  harness.step()
  message = harness.control.sm['longitudinalPlan']
  assert harness.control._longitudinal_plan_cache is not None
  harness.control._longitudinal_plan_cache = None
  with patch.object(controlsd.gc, 'get_referents', return_value=[None, b'bytes alone are insufficient', object()]):
    harness.control._decode_longitudinal_plan(message)
  assert harness.control._longitudinal_plan_cache is None


def test_unknown_pycapnp_version_uses_the_uncached_path():
  harness = make_harness()
  harness.step()
  message = harness.control.sm['longitudinalPlan']
  harness.control._longitudinal_plan_cache = None
  with patch.object(controlsd.capnp, '__version__', 'unknown'):
    harness.control._decode_longitudinal_plan(message)
  assert harness.control._longitudinal_plan_cache is None


@pytest.mark.parametrize('field', ['target', 'reference', 'trajectory', 'braking'])
def test_nan_projection_keeps_original_nonreflexive_identity_checks(field):
  harness = make_harness()
  harness.hold()
  if field == 'target':
    harness.plan.aTarget = float('nan')
  elif field == 'reference':
    harness.plan.reference.accel = float('nan')
  elif field == 'trajectory':
    harness.plan.speeds[0] = float('nan')
  else:
    harness.plan.motionIntent.brakingValid = True
    harness.plan.motionIntent.brakingAccel = float('nan')
  harness.step()
  assert harness.control._longitudinal_plan_cache is None
  command, motion = harness.step(new_plan=False)
  assert harness.control._longitudinal_plan_cache is None
  assert motion.referenceTimingReason == 'changed_producer_identity'
  assert motion.degraded and command.actuators.accel <= -2.0
  assert not command.cruiseControl.resume


@pytest.mark.parametrize('field', ['sourceVEgo', 'sourceAEgo', 'initialVEgo', 'initialAEgo'])
@pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_reference_state_uses_current_wire_fields_on_every_frame(field, value):
  harness = make_harness()
  command, motion = harness.step(reference_fields={field: value})
  assert harness.control._longitudinal_plan_cache is None
  assert motion.referenceTimingReason == 'unknown_reference_initial_state'
  command, motion = harness.step(new_plan=False)
  assert harness.control._longitudinal_plan_cache is None
  assert motion.referenceTimingReason == 'unknown_reference_initial_state'
  assert harness.control.reference_timing.last_consumer_ns == harness.now_ns
  assert motion.degraded and not command.cruiseControl.resume


@pytest.mark.parametrize('field', ['sourceVEgo', 'sourceAEgo', 'initialVEgo', 'initialAEgo'])
def test_new_received_reference_content_at_the_same_timestamp_cannot_reuse_projection(field):
  harness = make_harness()
  harness.step()
  origin = harness.now_ns
  cached = harness.control._longitudinal_plan_cache
  setattr(harness.plan.reference, field, getattr(harness.plan.reference, field) + 0.125)
  command, motion = harness.step(plan_time_ns=origin, model_time_ns=origin)
  assert harness.control._longitudinal_plan_cache is not cached
  assert motion.referenceTimingReason == 'changed_source_identity'
  assert motion.degraded and not command.cruiseControl.resume
  _, motion = harness.step()
  assert motion.referenceTimingValid and not motion.degraded


@pytest.mark.parametrize('plan_every', [1, 5, 30])
def test_reference_projection_matches_wire_consumption_through_reuse_expiry_and_override(plan_every):
  projected, wire = make_harness(), make_harness()
  consume_wire = wire.control.reference_timing.consume

  def original_input(_message, *args, **kwargs):
    return consume_wire(wire.control.sm['longitudinalPlan'].reference, *args, **kwargs)

  with patch.object(wire.control.reference_timing, 'consume', side_effect=original_input):
    for frame in range(65):
      for harness in (projected, wire):
        harness.car_state.gasPressed = 35 <= frame < 38
      actual = projected.step(new_plan=frame % plan_every == 0)
      expected = wire.step(new_plan=frame % plan_every == 0)
      assert [message.to_dict() for message in actual] == [message.to_dict() for message in expected]
      assert projected.control.reference_timing_check == wire.control.reference_timing_check
      for name in ('last_consumer_ns', 'last_timing', 'last_available_ns', 'last_command_identity'):
        assert getattr(projected.control.reference_timing, name) == getattr(wire.control.reference_timing, name)
