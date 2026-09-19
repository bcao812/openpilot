"""Source/availability/consumption contract, including production subscriptions.

The transport cases use Cap'n Proto bytes and SubMaster.update_msgs. Sockets,
lateral control and vehicle-model collaborators are bounded; the production
Controls constructor, longitudinal consumer, owner and publication remain real.
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from openpilot.selfdrive.controls.lib.reference_timing import (
  REFERENCE_MAX_AGE_NS, REFERENCE_TIMING_VERSION, ReferenceTimingConsumer, ReferenceTimingProducer,
)


BASE_NS = 2_000_000_000
STEP_NS = 10_000_000


def timing_message(source_ns=BASE_NS, producer_ns=BASE_NS + STEP_NS, *, producer=None, **fields):
  producer = producer or ReferenceTimingProducer()
  assert producer.qualify_source(source_ns, 20.0, -0.5, producer_ns) == source_ns
  producer.set_initial_state(20.0, -0.5)
  message = SimpleNamespace()
  producer.publish(message)
  for field, value in fields.items():
    setattr(message, field, value)
  return message


@pytest.mark.parametrize('bad_time', [0, -1, True, False, 1.5, float('nan'), float('inf'), 1 << 64, '2000000000'])
@pytest.mark.parametrize('field', ['source', 'consumer'])
def test_timing_unit_producer_rejects_invalid_raw_times_without_poisoning_recovery(bad_time, field):
  producer = ReferenceTimingProducer()
  assert producer.qualify_source(BASE_NS, 20.0, -0.5, BASE_NS + STEP_NS) == BASE_NS
  source_ns = bad_time if field == 'source' else BASE_NS + STEP_NS
  now_ns = bad_time if field == 'consumer' else BASE_NS + 2 * STEP_NS
  assert producer.qualify_source(source_ns, 20.0, -0.5, now_ns) is None
  assert producer.pending is None
  assert producer.qualify_source(BASE_NS + 2 * STEP_NS, 20.0, -0.5, BASE_NS + 3 * STEP_NS) == BASE_NS + 2 * STEP_NS


def test_timing_unit_producer_clock_and_source_are_separate_high_water_marks():
  producer = ReferenceTimingProducer()
  timing_message(producer=producer)
  assert producer.qualify_source(BASE_NS + STEP_NS, 20.0, -0.5, BASE_NS + 100 * STEP_NS) is None
  assert producer.qualify_source(BASE_NS + 4 * STEP_NS, 20.0, -0.5, BASE_NS + 2 * STEP_NS) is None
  assert producer.qualify_source(BASE_NS, 21.0, -0.5, BASE_NS + 2 * STEP_NS) is None
  assert producer.qualify_source(BASE_NS - STEP_NS, 20.0, -0.5, BASE_NS + 2 * STEP_NS) is None
  assert producer.qualify_source(BASE_NS + STEP_NS, 20.0, -0.5, BASE_NS + STEP_NS) is None
  assert producer.qualify_source(BASE_NS + STEP_NS, 20.0, -0.5, BASE_NS + 3 * STEP_NS) == BASE_NS + STEP_NS
  assert producer.instance_ns == BASE_NS + STEP_NS


@pytest.mark.parametrize('bad_source,changed_speed', [(BASE_NS + 200_000_000, 20.0), (BASE_NS, 21.0), (None, 20.0)])
def test_timing_unit_bad_source_does_not_hide_a_valid_producer_consumer_clock(bad_source, changed_speed):
  producer = ReferenceTimingProducer()
  assert producer.qualify_source(BASE_NS, 20.0, -0.5, BASE_NS + STEP_NS) == BASE_NS
  assert producer.qualify_source(bad_source, changed_speed, -0.5, BASE_NS + 100_000_000) is None
  assert producer.last_consumer_ns == BASE_NS + 100_000_000
  assert producer.qualify_source(BASE_NS + 40_000_000, 20.0, -0.5, BASE_NS + 50_000_000) is None
  assert producer.qualify_source(BASE_NS + 110_000_000, 20.0, -0.5, BASE_NS + 120_000_000) == BASE_NS + 110_000_000


def test_timing_unit_repeated_source_keeps_its_epoch_and_expires_at_250_ms():
  producer = ReferenceTimingProducer()
  consumer = ReferenceTimingConsumer()
  first = timing_message(producer=producer)
  assert consumer.consume(first, BASE_NS + STEP_NS, BASE_NS + 2 * STEP_NS).source_advanced
  repeated = timing_message(producer=producer, producer_ns=BASE_NS + 20 * STEP_NS)
  accepted = consumer.consume(repeated, BASE_NS + 21 * STEP_NS, BASE_NS + REFERENCE_MAX_AGE_NS)
  assert accepted.valid and not accepted.source_advanced
  assert accepted.timing.epoch_ns == accepted.timing.source_ns == BASE_NS
  # A freshly re-published event cannot grant one extra nanosecond of source age.
  expired = consumer.consume(repeated, BASE_NS + REFERENCE_MAX_AGE_NS, BASE_NS + REFERENCE_MAX_AGE_NS + 1)
  assert not expired.valid and expired.reason == 'expired_reference_source'
  assert producer.qualify_source(BASE_NS, 20.0, -0.5, BASE_NS + REFERENCE_MAX_AGE_NS + 1) is None


@pytest.mark.parametrize('fields,reason', [
  ({'timingVersion': 0}, 'unknown_reference_timing'),
  ({'timingVersion': REFERENCE_TIMING_VERSION + 1}, 'unknown_reference_timing'),
  ({'timingKnown': False}, 'unknown_reference_timing'),
  ({'sourceCarStateMonoTime': 0}, 'invalid_reference_time'),
  ({'referenceMonoTime': BASE_NS + 1}, 'noncausal_reference_time'),
  ({'sourceCarStateMonoTime': BASE_NS + 4 * STEP_NS, 'referenceMonoTime': BASE_NS + 4 * STEP_NS}, 'noncausal_reference_time'),
  ({'producerMonoTime': BASE_NS + 4 * STEP_NS}, 'noncausal_reference_time'),
  ({'producerInstanceMonoTime': BASE_NS + 4 * STEP_NS}, 'noncausal_reference_time'),
  ({'initialStateKnown': False}, 'unknown_reference_initial_state'),
  ({'sourceAEgo': float('nan')}, 'unknown_reference_initial_state'),
  ({'initialVEgo': -1.0}, 'unknown_reference_initial_state'),
])
def test_timing_unit_invalid_metadata_does_not_commit_source_identity(fields, reason):
  consumer = ReferenceTimingConsumer()
  result = consumer.consume(timing_message(**fields), BASE_NS + 2 * STEP_NS, BASE_NS + 3 * STEP_NS)
  assert not result.valid and result.reason == reason
  assert consumer.last_timing is None
  assert consumer.consume(timing_message(), BASE_NS + 2 * STEP_NS, BASE_NS + 4 * STEP_NS).valid


@pytest.mark.parametrize('mutation', ['source_speed', 'initial_accel', 'target', 'epoch', 'producer'])
def test_timing_unit_same_identity_cannot_change_and_bad_input_does_not_block_next_source(mutation):
  producer, consumer = ReferenceTimingProducer(), ReferenceTimingConsumer()
  original = timing_message(producer=producer)
  assert consumer.consume(original, BASE_NS + STEP_NS, BASE_NS + 2 * STEP_NS, command_identity=(0.5,)).valid
  changed = SimpleNamespace(**vars(original))
  command = (0.5,)
  if mutation == 'source_speed':
    changed.sourceVEgo += 1.0
  elif mutation == 'initial_accel':
    changed.initialAEgo -= 0.1
  elif mutation == 'target':
    command = (1.5,)
  elif mutation == 'epoch':
    changed.referenceMonoTime += 1
  else:
    changed.producerMonoTime -= 1
  assert not consumer.consume(changed, BASE_NS + STEP_NS, BASE_NS + 3 * STEP_NS, command_identity=command).valid
  fresh = timing_message(BASE_NS + 4 * STEP_NS, BASE_NS + 5 * STEP_NS, producer=producer)
  assert consumer.consume(fresh, BASE_NS + 6 * STEP_NS, BASE_NS + 7 * STEP_NS, command_identity=(0.5,)).valid


def test_timing_unit_impossible_consumer_or_availability_time_does_not_poison_recovery():
  consumer = ReferenceTimingConsumer()
  message = timing_message()
  assert consumer.consume(message, BASE_NS + STEP_NS, BASE_NS + 2 * STEP_NS).valid
  for available, now in ((BASE_NS + 5 * STEP_NS, BASE_NS + 3 * STEP_NS),
                         (BASE_NS + STEP_NS, BASE_NS + 1000 * STEP_NS),
                         (BASE_NS + STEP_NS, BASE_NS + STEP_NS),
                         (BASE_NS + STEP_NS, BASE_NS + 2 * STEP_NS)):
    assert not consumer.consume(message, available, now).valid
    assert consumer.last_consumer_ns == BASE_NS + 2 * STEP_NS
  assert consumer.consume(message, BASE_NS + STEP_NS, BASE_NS + 3 * STEP_NS).valid


def test_timing_unit_restart_requires_a_new_source_and_retires_the_old_instance():
  first_producer, consumer = ReferenceTimingProducer(), ReferenceTimingConsumer()
  first = timing_message(producer=first_producer)
  assert consumer.consume(first, BASE_NS + STEP_NS, BASE_NS + 2 * STEP_NS).valid
  second_producer = ReferenceTimingProducer()
  repeated = timing_message(producer=second_producer, producer_ns=BASE_NS + 3 * STEP_NS)
  rejected = consumer.consume(repeated, BASE_NS + 3 * STEP_NS, BASE_NS + 4 * STEP_NS)
  assert not rejected.valid and rejected.reason == 'restart_requires_new_source'
  fresh = timing_message(BASE_NS + 5 * STEP_NS, BASE_NS + 6 * STEP_NS, producer=second_producer)
  assert consumer.consume(fresh, BASE_NS + 6 * STEP_NS, BASE_NS + 7 * STEP_NS).valid
  delayed_old = timing_message(BASE_NS + 8 * STEP_NS, BASE_NS + 9 * STEP_NS, producer=first_producer)
  rejected = consumer.consume(delayed_old, BASE_NS + 9 * STEP_NS, BASE_NS + 10 * STEP_NS)
  assert not rejected.valid and rejected.reason == 'retired_producer_instance'
  newest = timing_message(BASE_NS + 11 * STEP_NS, BASE_NS + 12 * STEP_NS, producer=second_producer)
  assert consumer.consume(newest, BASE_NS + 12 * STEP_NS, BASE_NS + 13 * STEP_NS).valid
  restarted_consumer = ReferenceTimingConsumer()
  assert restarted_consumer.last_timing is None
  assert not restarted_consumer.consume(first, BASE_NS + STEP_NS, BASE_NS + REFERENCE_MAX_AGE_NS + 1).valid
  assert restarted_consumer.consume(newest, BASE_NS + 12 * STEP_NS, BASE_NS + REFERENCE_MAX_AGE_NS + 2).valid


def real_controls_harness():
  from openpilot.common.params import Params
  from openpilot.selfdrive.controls.controlsd import Controls
  from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import ControlsHarness, CapturingPublisher, new_event

  harness = ControlsHarness()
  template = harness.control
  params = Params()
  params.put('CarParams', template.CP.to_bytes())
  params.put('CarParamsSP', template.CP_SP.to_bytes())
  # Execute ControlsExt.__init__ and the production SubMaster call unchanged.
  with patch('openpilot.cereal.messaging.Poller'), patch('openpilot.cereal.messaging.sub_sock'), \
       patch('openpilot.cereal.messaging.PubMaster', return_value=CapturingPublisher()), \
       patch('openpilot.selfdrive.controls.controlsd.interfaces', {template.CP.carFingerprint: Mock(return_value=template.CI)}), \
       patch('openpilot.selfdrive.controls.controlsd.VehicleModel', return_value=template.VM), \
       patch('openpilot.selfdrive.controls.controlsd.LatControlPID', return_value=template.LaC), \
       patch('openpilot.selfdrive.controls.controlsd.ControlObservation.create', return_value=None):
    harness.control = Controls()
  services = harness.control.sm.services
  harness.events.update({service: new_event(service) for service in services if service not in harness.events})
  harness.control.sm.ignore_average_freq = list(services)
  harness.control.sm.simulation = False
  assert services.count('radarState') == 1
  assert {'carState', 'longitudinalPlan', 'modelV2', 'radarState', 'selfdriveState'}.issubset(services)
  return harness


def test_controls_real_subscriptions_consume_serialized_source_and_publish_actual_consumer_time():
  from openpilot.common.prefix import OpenpilotPrefix
  from openpilot.selfdrive.controls.lib.motion_supervisor import PLAN_LEASE_NS

  with OpenpilotPrefix():
    harness = real_controls_harness()
    assert REFERENCE_MAX_AGE_NS == PLAN_LEASE_NS
    source_ns = harness.now_ns - 5 * STEP_NS
    command, motion = harness.step(reference_source_ns=source_ns)
    assert command.longActive and command.actuators.accel > 0.0
    assert motion.referenceTimingValid and motion.referenceSourceAdvanced
    assert motion.referenceSourceMonoTime == motion.referenceEpochMonoTime == source_ns
    assert motion.referenceProducerMonoTime == motion.referenceConsumerMonoTime == harness.now_ns
    assert motion.planMonoTime == harness.control.sm.logMonoTime['longitudinalPlan'] == harness.now_ns
    assert source_ns < motion.planMonoTime
    command, motion = harness.step(new_plan=False)
    assert motion.referenceTimingValid and not motion.referenceSourceAdvanced
    assert motion.referenceProducerMonoTime < motion.referenceConsumerMonoTime
    assert motion.referenceSourceMonoTime == source_ns


@pytest.mark.parametrize('fault', ['unknown', 'future_source', 'future_producer', 'changed_source', 'late_source', 'restart_same_source'])
def test_serialized_reference_fault_enters_existing_owner_and_preserves_stronger_braking(fault):
  from openpilot.common.prefix import OpenpilotPrefix

  with OpenpilotPrefix():
    harness = real_controls_harness()
    harness.plan.aTarget = -2.5
    command, _ = harness.step()
    held_output = command.actuators.accel
    source_ns = harness.now_ns
    harness.plan.aTarget = 1.5
    harness.plan.motionIntent.releaseAllowed = True
    fields = {}
    if fault == 'unknown':
      fields['timingKnown'] = False
    elif fault == 'future_source':
      fields.update(sourceCarStateMonoTime=harness.now_ns + 5 * STEP_NS, referenceMonoTime=harness.now_ns + 5 * STEP_NS)
    elif fault == 'future_producer':
      fields['producerMonoTime'] = harness.now_ns + 5 * STEP_NS
    elif fault == 'changed_source':
      fields.update(sourceVEgo=harness.car_state.vEgo + 1.0)
    elif fault == 'late_source':
      fields.update(sourceCarStateMonoTime=source_ns - REFERENCE_MAX_AGE_NS, referenceMonoTime=source_ns - REFERENCE_MAX_AGE_NS)
    else:
      harness.reference_producer = ReferenceTimingProducer()
    command, motion = harness.step(reference_source_ns=source_ns, reference_fields=fields)
    assert motion.degraded and not motion.referenceTimingValid
    assert command.longActive and command.actuators.accel <= held_output
    assert not command.cruiseControl.resume
    command, motion = harness.step()
    assert motion.referenceTimingValid and not motion.degraded


def test_qualified_failed_solver_braking_keeps_its_time_contract():
  from openpilot.common.prefix import OpenpilotPrefix

  with OpenpilotPrefix():
    harness = real_controls_harness()
    for frame in range(10):
      harness.step(new_plan=frame % 5 == 0)
    harness.plan.solverStatus = 4
    harness.plan.motionIntent.valid = False
    harness.plan.motionIntent.brakingValid = True
    harness.plan.motionIntent.brakingAccel = -3.0
    harness.plan.aTarget = -3.0
    harness.events['longitudinalPlan'].valid = False
    command, motion = harness.step()
    assert motion.referenceTimingValid and motion.degraded
    assert command.longActive and command.actuators.accel == -3.0
    assert not command.cruiseControl.resume


def test_legacy_does_not_consume_unified_timing():
  from opendbc.car.honda.values import CAR
  from openpilot.common.prefix import OpenpilotPrefix
  from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import ControlsHarness

  with OpenpilotPrefix():
    harness = ControlsHarness(platform=CAR.HONDA_CIVIC_BOSCH)
    harness.plan.reference.timingVersion = REFERENCE_TIMING_VERSION + 1
    command, motion = harness.step()
    assert command.longActive and command.actuators.accel > 0.0
    assert harness.control.reference_timing_check is None
    assert not motion.referenceTimingValid


def production_planner_submaster():
  from openpilot.cereal import custom, messaging
  from openpilot.common.params import Params
  from opendbc.car.structs import car
  from openpilot.selfdrive.controls import plannerd

  captured = []
  actual_submaster = messaging.SubMaster
  gps_service = plannerd.get_gps_location_service(Params())

  class CapturedSubscriptions(Exception):
    pass

  def capture(services, **kwargs):
    with patch('openpilot.cereal.messaging.Poller'), patch('openpilot.cereal.messaging.sub_sock'):
      captured.append(actual_submaster(services, **kwargs))
    raise CapturedSubscriptions

  params = Mock()
  params.get.side_effect = [car.CarParams.new_message().to_bytes(), custom.CarParamsSP.new_message().to_bytes()]
  with patch.object(plannerd, 'Params', return_value=params), patch.object(plannerd, 'config_realtime_process'), \
       patch.object(plannerd, 'get_gps_location_service', return_value=gps_service), \
       patch.object(plannerd, 'LongitudinalPlanner'), patch.object(plannerd.LongitudinalTrace, 'create', return_value=None), \
       patch.object(plannerd, 'LaneDepartureWarning'), patch.object(messaging, 'PubMaster'), \
       patch.object(messaging, 'SubMaster', side_effect=capture), pytest.raises(CapturedSubscriptions):
    plannerd.main()
  assert len(captured) == 1
  return captured[0]


def test_plannerd_actual_subscription_call_keeps_all_provenance_sources():
  from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import INPUT_SERVICES

  state = production_planner_submaster()
  assert set(INPUT_SERVICES).issubset(state.services)
  assert state.services.count('radarState') == 1
  assert 'modelV2' not in state.non_polled_services


def consume_serialized_plan(harness, plan_event, now_ns):
  from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import serialized

  harness.now_ns = now_ns
  harness.events['longitudinalPlan'] = plan_event.as_builder()
  harness.plan = harness.events['longitudinalPlan'].longitudinalPlan
  incoming = []
  for service, event in harness.events.items():
    if service != 'longitudinalPlan':
      event.logMonoTime = now_ns
    incoming.append(serialized(event))
  harness.control.sm.update_msgs(now_ns / 1e9, incoming)
  with patch('openpilot.selfdrive.controls.controlsd.time.monotonic_ns', return_value=now_ns):
    command, lateral_log = harness.control.state_control()
    harness.control.publish(command, lateral_log)
  return harness.command, harness.motion


@pytest.mark.parametrize('interruption', ['duplicate', 'future', 'reordered', 'changed', 'late', 'restart'])
def test_native_production_messages_keep_source_epoch_and_do_not_reseed_invalid_history(interruption):
  from openpilot.common.prefix import OpenpilotPrefix
  from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import CapturingPublisher, new_event, serialized
  from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_planner, make_state

  with OpenpilotPrefix():
    planner, _ = make_planner()
    state = production_planner_submaster()
    state.ignore_average_freq = list(state.services)
    state.simulation = False
    template = make_state()
    harness = real_controls_harness()
    harness.car_state.vEgo = 20.0
    publisher = CapturingPublisher()
    history_used = []
    for frame in range(5):
      source_ns = BASE_NS + frame * 5 * STEP_NS
      producer_ns = BASE_NS + frame * 5 * STEP_NS + STEP_NS // 2
      available_ns, consumer_ns = producer_ns + STEP_NS // 5, producer_ns + STEP_NS // 2
      if frame == 2:
        if interruption in ('duplicate', 'changed', 'restart'):
          source_ns = BASE_NS + 5 * STEP_NS
        elif interruption == 'future':
          source_ns = producer_ns + 1
        elif interruption == 'reordered':
          source_ns = BASE_NS
        else:
          source_ns = producer_ns - REFERENCE_MAX_AGE_NS - 1
        if interruption == 'restart':
          old_instance_ns = planner.reference_timing.instance_ns
          planner, _ = make_planner()
      incoming = []
      for service in state.services:
        event = new_event(service, source_ns if service == 'carState' else producer_ns)
        if service in template:
          setattr(event, service, template[service])
        if service == 'carState':
          event.carState.canValid = True
          if frame == 2 and interruption == 'changed':
            event.carState.vEgo += 1.0
        incoming.append(serialized(event))
      state.update_msgs(producer_ns / 1e9, incoming)
      with patch.object(planner.mpc, 'set_weights', wraps=planner.mpc.set_weights) as weights:
        planner.update(state, now_ns=producer_ns)
      history_used.append(bool(weights.call_args.args[0]) if weights.call_args else False)
      with patch('openpilot.cereal.messaging.time.monotonic_ns', return_value=available_ns):
        planner.publish(state, publisher)
      event = publisher.messages['longitudinalPlan']
      assert event.logMonoTime == available_ns
      command, motion = consume_serialized_plan(harness, event, consumer_ns)
      bad_source = frame == 2 and interruption not in ('duplicate', 'restart')
      assert event.longitudinalPlan.reference.timingKnown != bad_source
      if not bad_source:
        reference = event.longitudinalPlan.reference
        assert reference.sourceCarStateMonoTime == reference.referenceMonoTime == source_ns
        assert reference.producerMonoTime == producer_ns
        assert reference.initialVEgo == pytest.approx(planner.mpc.x0[1])
        assert reference.initialAEgo == pytest.approx(planner.mpc.x0[2])
      rejected = frame == 2 and interruption != 'duplicate'
      assert motion.referenceTimingValid != rejected
      assert motion.referenceConsumerMonoTime == consumer_ns
      if rejected:
        assert motion.degraded and command.actuators.accel <= -0.2
        assert not command.cruiseControl.resume
      if frame == 2 and interruption == 'restart':
        assert event.longitudinalPlan.reference.producerInstanceMonoTime == producer_ns > old_instance_ns
        assert motion.referenceTimingReason == 'restart_requires_new_source'
      if frame == 2:
        assert not motion.referenceSourceAdvanced
      # Keep the actual 20 Hz planner / 100 Hz control cadence between plans.
      for _ in range(4):
        harness.step(new_plan=False)
    # A restarted MPC cannot borrow the previous instance's history. Its first
    # successful solve can seed its own next reference. Reference history is
    # not an executed request, and this contract adds no consumer-ACK protocol.
    assert history_used == [False, True, False, interruption == 'restart', True]
