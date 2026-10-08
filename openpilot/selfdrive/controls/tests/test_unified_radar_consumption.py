"""All planner consumers share expiry and purpose decisions for this tick."""
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.common.params import Params
from openpilot.selfdrive.car.honda_radar_observer_config import CONTROL, SHADOW
from openpilot.selfdrive.controls.lib.lead_qualification import control_leads
from openpilot.selfdrive.controls.lib.longitudinal_planner import has_safe_moving_lead
from openpilot.selfdrive.controls.lib.motion_supervisor import StopReason
from openpilot.selfdrive.controls.lib.radar_consumption import RadarConsumptionView
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_TIMING_VERSION, ReferenceTimingProducer
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import has_departing_lead
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_planner, make_state
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import CapturingPublisher, bounded_submaster, new_event, serialized
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_lead_qualification import caution_lead, formal_lead
from openpilot.selfdrive.controls.tests.test_radar_consumption import (
  CONSUMPTION, PUBLICATION, SESSION, coast, native_frame, radar_frame,
)
from openpilot.selfdrive.modeld.constants import ModelConstants


def enabled_planner(speed):
  params = Params()
  params.put('HondaBoschARadarMode', CONTROL, block=True)
  for name in ('AlphaLongitudinalEnabled', 'ExperimentalMode', 'DynamicExperimentalControl'):
    params.put_bool(name, True, block=True)
  return make_planner(speed=speed)[0]


def state_at(frame, *, speed=1.0, now=CONSUMPTION, publication=PUBLICATION, target=10.0):
  state = make_state(speed=speed, target=target, experimental=True)
  for service in state.logMonoTime:
    state.logMonoTime[service] = now
  state['radarState'] = native_frame(frame)
  state.logMonoTime['radarState'] = publication
  state['modelV2'].timestampEof = now
  return state


def publication_reuse_fixture(*, reader=True):
  primary = formal_lead(now_ns=PUBLICATION, dRel=40.0, vLead=5.5, vLeadK=5.5, vRel=.5)
  state = state_at(radar_frame(primary, display=(primary,)), speed=5.0)
  if reader:
    state['radarState'] = state['radarState'].as_reader()
  planner = enabled_planner(5.0)
  with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
    planner.update(state, now_ns=CONSUMPTION)
  return planner, state


def test_publish_reuses_the_exact_tick_reader_without_changing_display_records():
  planner, state = publication_reuse_fixture()
  source = planner._radar_for_publish(state)
  assert isinstance(source, RadarConsumptionView)
  assert source is planner.reference_planner.radar_consumption.publication_view
  records = []
  for cached in (True, False):
    if not cached:
      planner._radar_publication = None
    publisher = CapturingPublisher()
    with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=CONSUMPTION):
      planner.publish(state, publisher)
    records.append(publisher.messages['longitudinalPlan'].longitudinalPlan.to_dict())
  assert records[0] == records[1]


@pytest.mark.parametrize('changed', ['source', 'radar_epoch', 'model_epoch', 'next_attempt'])
def test_publisher_does_not_reuse_a_view_for_another_input_or_attempt(changed):
  planner, state = publication_reuse_fixture()
  assert isinstance(planner._radar_for_publish(state), RadarConsumptionView)
  if changed == 'source':
    state['radarState'] = state['radarState'].as_builder().as_reader()
  elif changed == 'radar_epoch':
    state.logMonoTime['radarState'] += 1
  elif changed == 'model_epoch':
    state.logMonoTime['modelV2'] += 1
  else:
    with patch.object(planner.reference_planner, 'update', side_effect=RuntimeError('failed attempt')):
      with pytest.raises(RuntimeError, match='failed attempt'):
        planner.update(state, now_ns=CONSUMPTION + 50_000_000)
    assert planner.reference_planner.radar_consumption.publication_view is None
  assert planner._radar_for_publish(state) is state['radarState']


def test_mutable_builder_keeps_the_original_publication_path():
  planner, state = publication_reuse_fixture(reader=False)
  assert planner.reference_planner.radar_consumption.publication_view is not None
  assert planner._radar_for_publish(state) is state['radarState']


@pytest.mark.parametrize('speed', [0.0, 1.0])
def test_irrelevant_coast_expiry_does_not_veto_primary_follow_or_departure(speed):
  primary = formal_lead(now_ns=PUBLICATION, vLead=speed + 1.0, vLeadK=speed + 1.0, vRel=1.0,
                        evidence={'trackId': 5})
  raw = radar_frame(primary, coast(), display=(primary,))
  planner = enabled_planner(speed)
  state = state_at(raw, speed=speed)
  with patch.object(planner.dec, 'enabled', return_value=True):
    view = planner.reference_planner.radar_input_state(state, CONSUMPTION)['radarState']
  assert len(control_leads(view)) == 1 and len(state['radarState'].radarObjects) == 2
  assert not planner.reference_planner.radar_consumption.degraded
  if speed:
    assert not has_safe_moving_lead(state['radarState'], speed, .3, now_ns=CONSUMPTION)
    assert has_safe_moving_lead(view, speed, .3, now_ns=CONSUMPTION)
  else:
    assert not has_departing_lead(state['radarState'], speed, .3, now_ns=CONSUMPTION, held_identity=(SESSION, 5))
    assert has_departing_lead(view, speed, .3, now_ns=CONSUMPTION, held_identity=(SESSION, 5))


@pytest.mark.parametrize('kind,expected', [('caution', True), ('bad_caution', False), ('inner', False)])
@pytest.mark.parametrize('speed', [0.0, 1.0])
def test_only_valid_edge_caution_can_be_ignored_for_another_cars_departure(kind, expected, speed):
  primary = formal_lead(now_ns=PUBLICATION, vLead=speed + 1.0, vLeadK=speed + 1.0, vRel=1.0,
                        evidence={'trackId': 5})
  if kind == 'inner':
    other = formal_lead(now_ns=PUBLICATION, dRel=2.0, vLead=0.0, vLeadK=0.0, vRel=-speed,
                        evidence={'trackId': 12})
  else:
    other = caution_lead(now_ns=PUBLICATION, dRel=2.0, vLead=0.0, vLeadK=0.0, vRel=-speed,
                         evidence={'trackId': 12, 'positionValid': kind != 'bad_caution'})
  frame = native_frame(radar_frame(primary, other, display=(primary,)))
  if speed:
    assert has_safe_moving_lead(frame, speed, .3, now_ns=CONSUMPTION) is expected
  else:
    assert has_departing_lead(frame, speed, .3, now_ns=CONSUMPTION, held_identity=(SESSION, 5)) is expected


def test_unified_expiry_matches_the_surviving_collection_without_changing_original_bytes():
  primary = formal_lead(now_ns=PUBLICATION, dRel=40.0, vLead=5.5, vLeadK=5.5, vRel=.5,
                        evidence={'trackId': 5})
  outputs = []
  for add_coast in (False, True):
    extras = (coast(),) if add_coast else ()
    frame = radar_frame(primary, *extras, display=(primary,))
    state = state_at(frame, speed=5.0)
    serialized = state['radarState'].to_bytes()
    state['radarState'].clear_write_flag()
    planner = enabled_planner(5.0)
    with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
      planner.update(state, now_ns=CONSUMPTION)
    assert planner.mpc.solution_status == 0 and planner.motion_intent.valid
    assert state['radarState'].to_bytes() == serialized
    outputs.append((planner.output_a_target, planner.mpc.params.copy()))
  assert outputs[0][0] == pytest.approx(outputs[1][0], abs=1e-9)
  np.testing.assert_array_equal(outputs[0][1], outputs[1][1])


def test_loss_of_an_already_adopted_hold_target_does_not_release_that_hold():
  planner = enabled_planner(1.0)
  planner.reference_planner.radar_held_identity = (SESSION, 7)
  raw = radar_frame(coast(relevant=True))
  with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
    for elapsed in (0, 100_000_000, 5_000_000_000):
      now = CONSUMPTION + elapsed
      publication = PUBLICATION if elapsed == 0 else now
      frame = raw if elapsed == 0 else radar_frame(publication=publication)
      state = state_at(frame, now=now, publication=publication)
      planner.update(state, now_ns=now)
      assert planner.reference_planner.radar_consumption.degraded
      assert planner.reference_planner.radar_risk_unresolved
      assert not state['radarState'].fusionRiskUnresolved
      publisher = CapturingPublisher()
      with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=now):
        planner.publish(state, publisher)
      assert publisher.messages['longitudinalPlan'].longitudinalPlan.radarRiskUnresolved
      assert planner.reference_planner.radar_held_identity == (SESSION, 7)
      assert planner.output_a_target <= 0.0
      assert not planner.motion_intent.moving_lead and not planner.motion_intent.release_allowed
    state['carState'].gasPressed = True
    planner.update(state, now_ns=now + 1_000_000)
  assert not planner.reference_planner.radar_consumption.degraded
  assert not planner.reference_planner.radar_risk_unresolved


@pytest.mark.parametrize('degraded,risk_unresolved', [(False, False), (True, False), (False, True), (True, True)])
def test_producer_loss_remains_diagnostic_without_invalidating_a_healthy_main_plan(degraded, risk_unresolved):
  planner = enabled_planner(5.0)
  state = state_at(radar_frame(degraded=degraded, risk_unresolved=risk_unresolved), speed=5.0)
  with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
    planner.update(state, now_ns=CONSUMPTION)
  assert planner.motion_intent.valid and not planner.reference_planner.radar_consumption.degraded
  assert planner.reference_planner.radar_risk_unresolved is risk_unresolved
  publisher = CapturingPublisher()
  with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=CONSUMPTION):
    planner.publish(state, publisher)
  assert publisher.messages['longitudinalPlan'].valid
  assert publisher.messages['longitudinalPlan'].longitudinalPlan.radarRiskUnresolved is risk_unresolved


@pytest.mark.parametrize('source', ['vision', 'radar'])
def test_history_flags_do_not_change_healthy_lead_acceleration_or_motion_intent(source):
  outputs = []
  for diagnostic in ('none', 'degraded', 'producer_loss', 'consumer_loss'):
    planner = enabled_planner(5.0)
    if diagnostic == 'consumer_loss':
      old = PUBLICATION - 100_000_000
      lost = coast(publication=old, relevant=True, track=99)
      planner.reference_planner.radar_consumption.update(
        native_frame(radar_frame(lost, publication=old)), publication_ns=old, now_ns=old + 60_000_000, speed=10.0,
      )
      assert planner.reference_planner.radar_consumption.degraded
    primary = formal_lead(now_ns=PUBLICATION, source=source, dRel=40.0, vLead=5.5, vLeadK=5.5, vRel=.5,
                          evidence={'trackId': 5})
    state = state_at(radar_frame(primary, display=(primary,), degraded=diagnostic == 'degraded',
                                 risk_unresolved=diagnostic == 'producer_loss'), speed=5.0)
    with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
      planner.update(state, now_ns=CONSUMPTION)
    assert planner.mpc.solution_status == 0 and planner.motion_intent.valid
    publisher = CapturingPublisher()
    with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=CONSUMPTION):
      planner.publish(state, publisher)
    plan = publisher.messages['longitudinalPlan'].longitudinalPlan
    assert plan.radarRiskUnresolved is (diagnostic in ('producer_loss', 'consumer_loss'))
    assert planner.output_a_target > 0.0 and not planner.output_should_stop
    assert planner.reference_planner.radar_held_identity is None
    if diagnostic == 'consumer_loss':
      assert planner.reference_planner.radar_consumption.degraded
    outputs.append((planner.output_a_target, planner.mpc.params.copy(), plan.motionIntent.to_dict(),
                    [row.to_dict() for row in plan.objectInfluences]))
  for actual in outputs[1:]:
    assert actual[0] == outputs[0][0]
    np.testing.assert_array_equal(actual[1], outputs[0][1])
    assert actual[2:] == outputs[0][2:]


def radar_stop_acknowledgement(planner, *, track=7, producer=PUBLICATION, now=CONSUMPTION):
  state = state_at(radar_frame(), speed=0.0, now=now)
  state.seen = dict.fromkeys(state, True)
  source = producer - 10_000_000
  assert planner.reference_timing.qualify_source(source, 0.0, 0.0, producer) == source
  planner.reference_timing.set_initial_state(0.0, 0.0)
  instance = planner.reference_timing.pending.instance_ns
  motion = state['controlsState'].motionState
  motion.phase, motion.resolvedStop, motion.reasons = 'hold', True, int(StopReason.CLOSE_LEAD)
  motion.referenceTimingVersion, motion.referenceTimingValid = REFERENCE_TIMING_VERSION, True
  motion.referenceSourceMonoTime = motion.referenceEpochMonoTime = source
  motion.referenceProducerMonoTime, motion.referenceProducerInstanceMonoTime = producer, instance
  motion.referenceConsumerMonoTime = producer + 10_000_000
  state['carControl'].actuators.accel = -.5
  return state, ((SESSION, track), source, producer, instance)


@pytest.mark.parametrize('defect', [
  'none', 'approach', 'missing_candidate', 'wrong_source', 'wrong_producer', 'wrong_instance', 'unconsumed',
  'stale_feedback', 'degraded', 'waiting', 'not_stopping', 'unresolved', 'independent_stop', 'positive_accel',
])
def test_only_matching_executed_object_stop_receipt_can_create_an_identity_hold(defect):
  planner = enabled_planner(0.0)
  ref = planner.reference_planner
  state, candidate = radar_stop_acknowledgement(planner)
  ref._radar_stop_candidate = candidate if defect != 'missing_candidate' else None
  motion = state['controlsState'].motionState
  now = CONSUMPTION
  if defect == 'approach':
    motion.phase = 'approach'
  elif defect == 'wrong_source':
    motion.referenceSourceMonoTime += 1
  elif defect == 'wrong_producer':
    motion.referenceProducerMonoTime += 1
  elif defect == 'wrong_instance':
    motion.referenceProducerInstanceMonoTime += 1
  elif defect == 'unconsumed':
    motion.referenceConsumerMonoTime = PUBLICATION
  elif defect == 'stale_feedback':
    now += 250_000_001
  elif defect == 'degraded':
    motion.degraded = True
  elif defect == 'waiting':
    motion.waitingForFreshInput = True
  elif defect == 'not_stopping':
    motion.phase = 'track'
  elif defect == 'unresolved':
    motion.resolvedStop = False
  elif defect == 'independent_stop':
    motion.reasons = int(StopReason.E2E)
  elif defect == 'positive_accel':
    state['carControl'].actuators.accel = .1
  ref.adopt_executed_radar_stop(state, now_ns=now)
  assert ref.radar_held_identity == ((SESSION, 7) if defect in ('none', 'approach') else None)
  assert ref._radar_stop_candidate is None


@pytest.mark.parametrize('service,defect', [
  ('controlsState', 'none'), ('carControl', 'invalid'), ('controlsState', 'unseen'),
  ('carControl', 'dead'), ('controlsState', 'bad_frequency'),
])
def test_acknowledgement_requires_actual_submaster_transport_health(service, defect):
  planner = enabled_planner(0.0)
  state, candidate = radar_stop_acknowledgement(planner)
  real = bounded_submaster(['carControl', 'controlsState'])
  incoming = []
  for name in real.data:
    event = new_event(name, CONSUMPTION)
    setattr(event, name, state[name])
    incoming.append(serialized(event))
  real.update_msgs(CONSUMPTION / 1e9, incoming)
  if defect == 'invalid':
    real.valid[service] = False
  elif defect == 'unseen':
    real.seen[service] = False
  elif defect == 'dead':
    real.alive[service] = False
  elif defect == 'bad_frequency':
    real.ignore_average_freq.remove(service)
    real.freq_ok[service] = False
  ref = planner.reference_planner
  ref._radar_stop_candidate = candidate
  ref.adopt_executed_radar_stop(real, now_ns=CONSUMPTION)
  assert ref.radar_held_identity == ((SESSION, 7) if defect == 'none' else None)
  if defect != 'none':
    assert candidate[2] in ref._radar_stop_pending
    real.valid[service] = real.seen[service] = real.alive[service] = real.freq_ok[service] = True
    ref.adopt_executed_radar_stop(real, now_ns=CONSUMPTION)
    assert ref.radar_held_identity == (SESSION, 7)


def test_late_ack_survives_a_new_planner_candidate_without_overwriting_an_adopted_hold():
  planner = enabled_planner(0.0)
  ref = planner.reference_planner
  old_state, old_candidate = radar_stop_acknowledgement(planner)
  old_state['controlsState'].motionState.referenceTimingValid = False
  ref._radar_stop_candidate = old_candidate
  ref.adopt_executed_radar_stop(old_state, now_ns=CONSUMPTION)
  assert ref.radar_held_identity is None and old_candidate[2] in ref._radar_stop_pending
  new_state, new_candidate = radar_stop_acknowledgement(
    planner, track=8, producer=PUBLICATION + 50_000_000, now=CONSUMPTION + 50_000_000)
  new_state['controlsState'].motionState.referenceTimingValid = False
  ref._radar_stop_candidate = new_candidate
  ref.adopt_executed_radar_stop(new_state, now_ns=CONSUMPTION + 50_000_000)
  assert set(ref._radar_stop_pending) == {old_candidate[2], new_candidate[2]}
  old_state['controlsState'].motionState.referenceTimingValid = True
  ref.adopt_executed_radar_stop(old_state, now_ns=CONSUMPTION + 100_000_000)
  assert ref.radar_held_identity == (SESSION, 7)
  new_state['controlsState'].motionState.referenceTimingValid = True
  ref.adopt_executed_radar_stop(new_state, now_ns=CONSUMPTION + 100_000_000)
  assert ref.radar_held_identity == (SESSION, 7) and not ref._radar_stop_pending


def test_pending_stop_receipts_have_inclusive_age_and_count_bounds_and_cannot_cross_restart():
  planner = enabled_planner(0.0)
  ref = planner.reference_planner
  for index in range(17):
    producer = PUBLICATION + index * 10_000_000
    state, candidate = radar_stop_acknowledgement(planner, track=index + 1, producer=producer, now=producer + 20_000_000)
    state.seen['controlsState'] = False
    ref._radar_stop_candidate = candidate
    ref.adopt_executed_radar_stop(state, now_ns=producer + 20_000_000)
  assert len(ref._radar_stop_pending) == 16
  assert PUBLICATION not in ref._radar_stop_pending and PUBLICATION + 10_000_000 in ref._radar_stop_pending
  oldest = PUBLICATION + 10_000_000
  ref.adopt_executed_radar_stop(state, now_ns=oldest + 500_000_000)
  assert oldest in ref._radar_stop_pending
  ref.adopt_executed_radar_stop(state, now_ns=oldest + 500_000_001)
  assert oldest not in ref._radar_stop_pending
  planner.reference_timing = ReferenceTimingProducer()
  restarted = PUBLICATION + 600_000_000
  assert planner.reference_timing.qualify_source(restarted, 0.0, 0.0, restarted) == restarted
  ref.adopt_executed_radar_stop(state, now_ns=restarted)
  assert not ref._radar_stop_pending and ref.radar_held_identity is None


@pytest.mark.parametrize('scope', ['mode', 'dec', 'experimental', 'driver_exit'])
def test_configuration_exit_disables_new_claims_without_releasing_an_acknowledged_hold(scope):
  planner = enabled_planner(0.0)
  ref = planner.reference_planner
  ack, candidate = radar_stop_acknowledgement(planner)
  ref._radar_stop_candidate = candidate
  ref.adopt_executed_radar_stop(ack, now_ns=CONSUMPTION)
  assert ref.radar_held_identity == (SESSION, 7)
  now = CONSUMPTION + 50_000_000
  state = state_at(radar_frame(publication=now), speed=0.0, now=now, publication=now)
  if scope == 'mode':
    Params().put('HondaBoschARadarMode', SHADOW, block=True)
  elif scope == 'experimental':
    state['selfdriveState'].experimentalMode = False
  elif scope == 'driver_exit':
    state['selfdriveState'].enabled = False
  with patch.object(planner.dec, 'enabled', return_value=scope != 'dec'), patch.object(planner.dec, 'mode', return_value='acc'):
    planner.update(state, now_ns=now)
  assert ref._radar_stop_candidate is None and not ref._radar_stop_pending
  assert ref.radar_held_identity == (None if scope == 'driver_exit' else (SESSION, 7))
  if scope != 'driver_exit':
    assert not planner.motion_intent.release_allowed


def test_rolling_radar_stop_is_adopted_by_real_owner_before_dropout_preserves_the_stop():
  from openpilot.selfdrive.car.tests.honda_radar_control_test_support import IdealPlant, Target, control_pipeline
  from openpilot.selfdrive.controls.tests.test_radar_fusion_scenarios import target_motion

  # Reuse the established, consistent-motion stop fixture: decoded Honda CAN
  # drives the ideal plant; no lead, MPC result, owner ACK or request is patched.
  plant = IdealPlant(6.0, delay=.2, response_time=.25)
  rolling_identity = None
  with control_pipeline(speed=plant.speed, cruise=10.0) as pipeline:
    ref = pipeline.planner.reference_planner
    for frame in range(160):
      position, lead_speed = target_motion(frame * .05)
      distance = position - plant.position
      assert distance > 0.0
      row = pipeline.step(speed=plant.speed, accel=plant.accel, radar=[Target(distance, lead_speed)], plant=plant)
      candidate = ref._radar_stop_candidate
      if candidate is not None and row['speed'] > .2:
        assert row['reasons'] & int(StopReason.CLOSE_LEAD)
        assert candidate[0] in pipeline.planner.mpc.object_influence_identities
        assert any(owner['phase'] == 'approach' and owner['stop'] and owner['accel'] <= 0.0 for owner in row['owner'])
        rolling_identity = candidate[0]
      if rolling_identity is not None and ref.radar_held_identity == rolling_identity:
        break
    assert rolling_identity is not None and ref.radar_held_identity == rolling_identity
    for _ in range(20):
      row = pipeline.step(speed=plant.speed, accel=plant.accel, observer_online=False, plant=plant)
      assert ref.radar_held_identity == rolling_identity and not row['release']
      assert all(owner['phase'] in ('approach', 'hold') and owner['stop'] and owner['accel'] <= 0.0 for owner in row['owner'])
      assert not any(sample['release'] or sample['accel'] > 0.0 for sample in row['can'])


@pytest.mark.parametrize('stop_request', ['explicit_action', 'typed_rest'])
def test_red_light_stop_does_not_assign_a_far_stationary_detection_a_hold_that_blocks_green(stop_request):
  outputs = []
  for include_far_radar in (False, True):
    planner = enabled_planner(0.0)
    far = formal_lead(now_ns=PUBLICATION, dRel=80.0, vLead=0.0, vLeadK=0.0, vRel=0.0)
    objects = (far,) if include_far_radar else ()
    state = state_at(radar_frame(*objects, display=objects), speed=0.0)
    state['modelV2'].action.shouldStop = stop_request == 'explicit_action'
    state['modelV2'].action.desiredAcceleration = -.5 if stop_request == 'explicit_action' else 0.0
    with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='blended'), \
         patch.object(planner, 'is_e2e', return_value=True):
      planner.update(state, now_ns=CONSUMPTION)
      assert planner.output_should_stop
      assert planner.motion_intent.reasons & int(StopReason.E2E if stop_request == 'explicit_action' else StopReason.MPC)
      assert planner.reference_planner._radar_stop_candidate is None
      assert planner.reference_planner.radar_held_identity is None
      now = CONSUMPTION + 50_000_000
      state = state_at(radar_frame(publication=now, degraded=include_far_radar, risk_unresolved=include_far_radar),
                       speed=0.0, now=now, publication=now)
      state['controlsState'].motionState.phase = 'hold'
      state['controlsState'].motionState.resolvedStop = True
      state['controlsState'].motionState.reasons = int(StopReason.E2E if stop_request == 'explicit_action' else StopReason.MPC)
      model = state['modelV2']
      model.position.x = [float(.25 * t * t) for t in ModelConstants.T_IDXS]
      model.velocity.x = [float(.5 * t) for t in ModelConstants.T_IDXS]
      model.acceleration.x = [.5] * len(ModelConstants.T_IDXS)
      model.action.desiredAcceleration = .5
      planner.update(state, now_ns=now)
    assert planner.motion_intent.valid and planner.mpc.solution_status == 0
    assert planner.reference_planner.radar_held_identity is None
    assert planner.motion_intent.release_allowed and not planner.output_should_stop
    outputs.append(planner.output_a_target)
  assert outputs[0] > 0.0 and outputs[1] == pytest.approx(outputs[0], abs=1e-6, rel=0.0)


@pytest.mark.parametrize('consumer_loss', [False, True])
def test_scope_exit_clears_only_producer_claim_while_retained_consumer_risk_needs_driver_reset(consumer_loss):
  planner = enabled_planner(1.0)
  objects = (coast(relevant=True),) if consumer_loss else ()
  state = state_at(radar_frame(*objects, degraded=True, risk_unresolved=True))
  with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
    planner.update(state, now_ns=CONSUMPTION)
    assert planner.reference_planner.radar_risk_unresolved
    Params().put('HondaBoschARadarMode', SHADOW, block=True)
    now = CONSUMPTION + 100_000_000
    state = state_at(radar_frame(publication=now, risk_unresolved=True), now=now, publication=now)
    planner.update(state, now_ns=now)
    assert not planner.reference_planner.fusion_admitted
    assert planner.reference_planner.radar_risk_unresolved is consumer_loss
    assert planner.reference_planner.radar_consumption.degraded is consumer_loss
    state['carState'].gasPressed = True
    planner.update(state, now_ns=now + 1_000_000)
  assert not planner.reference_planner.radar_risk_unresolved
  assert not planner.reference_planner.radar_consumption.degraded


@pytest.mark.parametrize('failure', ['source_epoch', 'owner', 'model_shape', 'pitch', 'throttle', 'mpc'])
def test_invalid_early_returns_clear_the_current_plan_claim(failure):
  planner = enabled_planner(5.0)
  ref = planner.reference_planner
  ref.radar_risk_unresolved = True
  state = state_at(radar_frame(degraded=True, risk_unresolved=True), speed=5.0)
  if failure == 'source_epoch':
    state.logMonoTime['carState'] = CONSUMPTION + 1
  elif failure == 'owner':
    state['controlsState'].motionState.version = 0
  elif failure == 'model_shape':
    state['modelV2'].velocity.x = []
  elif failure == 'pitch':
    state['carControl'].orientationNED = [0.0, float('nan'), 0.0]
  elif failure == 'throttle':
    state['modelV2'].meta.disengagePredictions.gasPressProbs = [1.0, float('nan')]
  with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
    if failure == 'mpc':
      with patch.object(planner.mpc, 'update', side_effect=lambda *a, **kw: setattr(planner.mpc, 'solution_status', 1)):
        planner.update(state, now_ns=CONSUMPTION)
    else:
      planner.update(state, now_ns=CONSUMPTION)
  assert not ref.radar_risk_unresolved and not planner.motion_intent.valid
  publisher = CapturingPublisher()
  with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=CONSUMPTION):
    planner.publish(state, publisher)
  assert not publisher.messages['longitudinalPlan'].longitudinalPlan.radarRiskUnresolved


def test_non_unified_main_plan_publishes_false_even_after_a_retained_claim():
  planner = enabled_planner(5.0)
  state = state_at(radar_frame(), speed=5.0)
  planner.reference_planner.radar_risk_unresolved = True
  publisher = CapturingPublisher()
  with patch.object(planner, 'unified', False), \
       patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=CONSUMPTION):
    planner.publish(state, publisher)
  assert not publisher.messages['longitudinalPlan'].longitudinalPlan.radarRiskUnresolved


def test_invalid_radar_transport_cannot_clear_a_consumption_loss():
  planner = enabled_planner(1.0)
  with patch.object(planner.dec, 'enabled', return_value=True):
    state = state_at(radar_frame(coast(relevant=True)))
    planner.reference_planner.radar_input_state(state, CONSUMPTION)
    assert planner.reference_planner.radar_consumption.degraded
    now = CONSUMPTION + 100_000_000
    state = state_at(radar_frame(publication=now, cleared=(7,)), now=now, publication=now)
    state.valid['radarState'] = False
    planner.reference_planner.radar_input_state(state, now)
    assert planner.reference_planner.radar_consumption.degraded
    state.valid['radarState'] = True
    planner.reference_planner.radar_input_state(state, now)
    assert not planner.reference_planner.radar_consumption.degraded


def test_scope_exit_sensor_view_requires_no_car_or_control_inputs():
  planner = enabled_planner(10.0)
  Params().put('HondaBoschARadarMode', SHADOW, block=True)
  published = native_frame(radar_frame(formal_lead(now_ns=PUBLICATION)))
  published.visionLeadOne = {'present': True, 'modelProb': .99, 'dRel': 20.0, 'vRel': -10.0, 'vLead': 0.0}
  inputs = {'radarState': published, 'selfdriveState': SimpleNamespace(experimentalMode=True)}
  before, serialized = published.to_dict(), published.to_bytes()
  published.clear_write_flag()
  with patch.object(planner.dec, 'enabled', return_value=True):
    view = planner.reference_planner.radar_input_state(inputs, CONSUMPTION)['radarState']
  assert not view.fusionActive and view.leadOne.dRel == 20.0
  assert not control_leads(view, 'caution', 0.0, now_ns=CONSUMPTION)
  assert published.to_dict() == before and published.to_bytes() == serialized


@pytest.mark.parametrize('fallback', ['mode4_startup', 'shadow_vision'])
def test_legacy_vision_in_startup_or_scope_fallback_cannot_be_treated_as_edge(fallback):
  planner = enabled_planner(10.0)
  frame = radar_frame(publication=PUBLICATION, degraded=fallback == 'mode4_startup')
  state = state_at(frame, speed=10.0, target=20.0)
  legacy = {'present': True, 'modelProb': .99, 'dRel': 20.0, 'vRel': -10.0,
            'vLead': 0.0, 'vLeadK': 0.0, 'aLeadK': 0.0, 'aLeadTau': 1.5}
  state['radarState'].leadOne = legacy
  state['radarState'].visionLeadOne = legacy
  if fallback == 'mode4_startup':
    state['radarState'].fusionActive = False
  else:
    Params().put('HondaBoschARadarMode', SHADOW, block=True)
  with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
    sensor_view = planner.reference_planner.radar_input_state(state, CONSUMPTION)['radarState']
    assert not control_leads(sensor_view, 'caution', 0.0, now_ns=CONSUMPTION)
    planner.update(state, now_ns=CONSUMPTION)
  assert not planner.reference_planner.caution_accel_limited
  assert planner.reference_planner.caution_object_identities == ()


@pytest.mark.parametrize('distance,closing,should_limit', [(20.0, 10.0, True), (20.01, 10.0, False), (10.0, 0.0, False)])
def test_edge_ttc_precaution_limits_only_positive_requests(distance, closing, should_limit):
  planner = enabled_planner(10.0)
  edge = caution_lead(now_ns=PUBLICATION, dRel=distance, vLead=10.0-closing, vLeadK=10.0-closing, vRel=-closing,
                       evidence={'trackId': 12})
  state = state_at(radar_frame(edge), speed=10.0, target=20.0)
  with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
    planner.update(state, now_ns=CONSUMPTION)
  assert planner.mpc.solution_status == 0
  assert planner.reference_planner.caution_accel_limited is should_limit
  assert not planner.fcw and not planner.output_should_stop
  assert planner.reference_planner.radar_held_identity is None
  if should_limit:
    assert planner.output_a_target == 0.0
    assert not planner.motion_intent.release_allowed
    assert planner.reference_planner.caution_object_identities == ((SESSION, 12),)
  else:
    assert planner.output_a_target > 0.0


def test_edge_dropout_has_no_persistent_loss_and_does_not_weaken_stronger_inner_braking():
  planner = enabled_planner(10.0)
  edge = caution_lead(now_ns=PUBLICATION, dRel=10.0, vLead=0.0, vLeadK=0.0, vRel=-10.0,
                       evidence={'trackId': 12})
  inner = formal_lead(now_ns=PUBLICATION, dRel=10.0, vLead=0.0, vLeadK=0.0, vRel=-10.0,
                      evidence={'trackId': 13})
  with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
    state = state_at(radar_frame(edge, inner, display=(inner,)), speed=10.0)
    planner.update(state, now_ns=CONSUMPTION)
    assert planner.output_a_target < 0.0
    assert planner.reference_planner.caution_object_identities == ()
    assert not planner.reference_planner.radar_consumption.degraded
    now = CONSUMPTION + 100_000_000
    state = state_at(radar_frame(publication=now), speed=10.0, now=now, publication=now, target=20.0)
    planner.update(state, now_ns=now)
  assert not planner.reference_planner.radar_consumption.degraded
  assert planner.reference_planner.caution_object_identities == ()


@pytest.mark.parametrize('speed', [0.0, .2])
def test_edge_cap_does_not_create_a_stop_preference_or_new_hold_near_rest(speed):
  arms = []
  for include_edge in (False, True):
    edge = caution_lead(now_ns=PUBLICATION, dRel=1.0, vLead=speed - 1.0, vLeadK=speed - 1.0, vRel=-1.0,
                         evidence={'trackId': 12})
    frame = radar_frame(*((edge,) if include_edge else ()))
    state = state_at(frame, speed=speed)
    planner = enabled_planner(speed)
    with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
      planner.update(state, now_ns=CONSUMPTION)
    assert planner.mpc.solution_status == 0
    assert planner.reference_planner.radar_held_identity is None
    arms.append(planner)
  before, after = arms
  # The native goal, cruise history and stop reasons are identical; only an
  # already positive final acceleration may be capped by the edge return.
  np.testing.assert_array_equal(before.mpc.params, after.mpc.params)
  np.testing.assert_allclose(before.mpc.v_solution, after.mpc.v_solution, atol=1e-9, rtol=0.0)
  assert before.motion_intent.reasons == after.motion_intent.reasons
  assert before.output_should_stop == after.output_should_stop
  assert after.output_a_target == min(before.output_a_target, 0.0)


def test_edge_caution_preserves_an_existing_model_stop_and_held_vehicle():
  arms = []
  for include_edge in (False, True):
    planner = enabled_planner(0.0)
    planner.reference_planner.radar_held_identity = (SESSION, 5)
    edge = caution_lead(now_ns=PUBLICATION, dRel=1.0, vLead=-1.0, vLeadK=-1.0, vRel=-1.0,
                         evidence={'trackId': 12})
    state = state_at(radar_frame(*((edge,) if include_edge else ())), speed=0.0)
    state['modelV2'].action.shouldStop = True
    state['modelV2'].action.desiredAcceleration = -2.0
    with patch.object(planner.dec, 'enabled', return_value=True), \
         patch.object(planner.dec, 'active', return_value=True), patch.object(planner.dec, 'mode', return_value='blended'):
      planner.update(state, now_ns=CONSUMPTION)
    assert planner.output_a_target <= -2.0 and planner.output_should_stop
    assert planner.reference_planner.radar_held_identity == (SESSION, 5)
    assert not planner.reference_planner.caution_accel_limited
    assert planner.reference_planner.caution_object_identities == ()
    arms.append(planner)
  assert arms[0].motion_intent.reasons == arms[1].motion_intent.reasons
  assert arms[0].motion_intent.release_allowed == arms[1].motion_intent.release_allowed
