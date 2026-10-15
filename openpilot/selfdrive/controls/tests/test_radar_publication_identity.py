"""Wire identities record inputs actually supplied to each planning boundary."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

from opendbc.car.honda.values import CAR
from openpilot.cereal import messaging
from openpilot.selfdrive.car.honda_radar_observer_config import SERVICE
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import Publisher, advance, make_planner, make_state
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_radard_publication import Inputs, make_radard, publish_at


BASE_NS = 1_000_000_000
SESSION = 'observation-publication-test'


def observation(publication_ns, sequence, *, measurement_ns=BASE_NS - 20_000_000):
  event = messaging.new_message(SERVICE, valid=True, logMonoTime=publication_ns)
  payload = getattr(event, SERVICE)
  payload.sessionId, payload.sequence = SESSION, sequence
  payload.lastPositionMonoTime = measurement_ns
  # Distinct publications can contain exactly the same source measurement pair.
  payload.fusion.tracks = [{
    'trackId': 7, 'generation': 1, 'positionSeq': 8,
    'positionMonoTime': measurement_ns, 'velocityMonoTime': measurement_ns,
  }]
  return messaging.log_from_bytes(event.to_bytes())


def accept_observation(inputs, event):
  inputs[SERVICE] = getattr(event, SERVICE)
  inputs.logMonoTime[SERVICE] = event.logMonoTime
  inputs.seen[SERVICE], inputs.valid[SERVICE] = True, event.valid


def producer_with_adapter():
  producer = make_radard()
  producer.control_enabled = Mock(return_value=True)
  # This deliberately differs from the input's session: trace input identity
  # must not be reconstructed from the adapter's accepted result/history.
  result_session = 'control-result-session'
  producer.fusion_control = SimpleNamespace(_session=result_session, update=Mock(return_value=SimpleNamespace(
    leads=({'present': False}, {'present': False}), healthy=True, degraded=False,
    reason='noQualifiedObjects', cleared=(), session_id=result_session,
  )))
  return producer


def assert_observation_identity(radar, event):
  payload = getattr(event, SERVICE)
  assert radar.fusionInputAttempted
  assert radar.fusionInputMonoTime == event.logMonoTime
  assert radar.fusionInputSessionId == payload.sessionId
  assert radar.fusionInputSequence == payload.sequence


def assert_no_observation_identity(radar):
  assert not radar.fusionInputAttempted
  assert radar.fusionInputMonoTime == radar.fusionInputSequence == 0
  assert radar.fusionInputSessionId == ''


def test_republished_pair_keeps_distinct_actual_observation_publications():
  producer = producer_with_adapter()
  identities = []
  for index in range(2):
    event = observation(BASE_NS + index * 25_000_000, 11 + index)
    inputs = Inputs(event.logMonoTime + 1_000_000)
    accept_observation(inputs, event)
    producer.update(inputs, inputs['radarTracks'], now_ns=event.logMonoTime + 1_000_000)
    published = publish_at(producer, event.logMonoTime + 2_000_000)
    assert_observation_identity(published.radarState, event)
    supplied = producer.fusion_control.update.call_args.kwargs
    assert supplied['observation'] is inputs[SERVICE]
    assert supplied['observation_ns'] == event.logMonoTime
    assert supplied['observation'].fusion.tracks[0].positionMonoTime == BASE_NS - 20_000_000
    assert supplied['observation'].fusion.tracks[0].velocityMonoTime == BASE_NS - 20_000_000
    assert published.radarState.fusionSessionId != published.radarState.fusionInputSessionId
    identities.append((published.radarState.fusionInputMonoTime, published.radarState.fusionInputSequence))
  assert identities[0] != identities[1]


def test_delayed_or_coalesced_observation_keeps_chosen_input_before_publish():
  producer = producer_with_adapter()
  events = [observation(BASE_NS + index * 10_000_000, 21 + index) for index in range(3)]
  inputs = Inputs(BASE_NS + 40_000_000)
  # A consumer can select the middle publication while the log already has a
  # newer one. Its true input cannot be inferred as the last preceding Event.
  accept_observation(inputs, events[1])
  producer.update(inputs, inputs['radarTracks'], now_ns=BASE_NS + 40_000_000)
  accept_observation(inputs, events[2])
  published = publish_at(producer, BASE_NS + 50_000_000)
  assert_observation_identity(published.radarState, events[1])
  assert published.radarState.fusionInputMonoTime != inputs.logMonoTime[SERVICE]


def test_adapter_exception_keeps_attempted_input_without_claiming_acceptance():
  producer = producer_with_adapter()
  producer.fusion_control.update.side_effect = RuntimeError('adapter failure fixture')
  inputs = Inputs(BASE_NS + 1_000_000)
  event = observation(BASE_NS, 31)
  accept_observation(inputs, event)
  producer.update(inputs, inputs['radarTracks'], now_ns=BASE_NS + 1_000_000)
  published = publish_at(producer, BASE_NS + 2_000_000)
  assert_observation_identity(published.radarState, event)
  assert not published.radarState.fusionActive and not published.radarState.fusionHealthy
  assert published.radarState.fusionDegraded
  assert published.radarState.fusionReason == 'controlAdapterError'
  assert not published.radarState.leadOne.present and not published.radarState.leadTwo.present


def test_unreadable_trace_identity_does_not_change_adapter_result():
  class BrokenIdentity:
    @property
    def sessionId(self):
      raise RuntimeError('diagnostic identity read failed')

  producer = producer_with_adapter()
  inputs = Inputs(BASE_NS)
  inputs[SERVICE] = BrokenIdentity()
  producer.update(inputs, inputs['radarTracks'], now_ns=BASE_NS)
  published = publish_at(producer, BASE_NS + 1_000_000)
  producer.fusion_control.update.assert_called_once()
  assert published.radarState.fusionInputAttempted
  assert published.radarState.fusionInputMonoTime == published.radarState.fusionInputSequence == 0
  assert published.radarState.fusionInputSessionId == ''
  assert published.radarState.fusionHealthy and not published.radarState.fusionDegraded
  assert published.radarState.fusionReason == 'noQualifiedObjects'


def test_scope_skip_clears_previous_input_identity_without_calling_adapter():
  producer = producer_with_adapter()
  inputs = Inputs(BASE_NS + 1_000_000)
  accept_observation(inputs, observation(BASE_NS, 41))
  producer.update(inputs, inputs['radarTracks'], now_ns=BASE_NS + 1_000_000)
  assert producer.radar_state.fusionInputAttempted
  producer.fusion_control.update.reset_mock()
  producer.control_enabled.return_value = False
  producer.control_scope_unavailable = True
  accept_observation(inputs, observation(BASE_NS + 25_000_000, 42))
  producer.update(inputs, inputs['radarTracks'], now_ns=BASE_NS + 26_000_000)
  published = publish_at(producer, BASE_NS + 27_000_000)
  producer.fusion_control.update.assert_not_called()
  assert_no_observation_identity(published.radarState)
  assert published.radarState.fusionReason == 'controlScopeUnavailable'


def test_generic_mode_without_optional_observation_keeps_wire_defaults():
  producer = make_radard()
  producer.control_enabled = Mock(return_value=False)
  inputs = Inputs(BASE_NS)
  del inputs[SERVICE]
  del inputs.logMonoTime[SERVICE]
  producer.update(inputs, inputs['radarTracks'], now_ns=BASE_NS)
  published = publish_at(producer, BASE_NS + 1_000_000)
  assert_no_observation_identity(published.radarState)
  assert producer.fusion_control is None


def test_planner_publish_keeps_radar_event_from_update_after_new_input_arrives():
  planner, _ = make_planner(diagnostics=False)
  state = make_state()
  now = advance(state, 0)
  chosen_ns = now - 20_000_000
  state.logMonoTime['radarState'] = chosen_ns
  planner.update(state, now_ns=now + 500_000)
  target, intent = planner.output_a_target, planner.motion_intent
  # Replace both the payload and its Event identity between update and publish.
  # Only a later update is allowed to select this publication for a new plan.
  state['radarState'] = messaging.new_message('radarState').radarState
  state.logMonoTime['radarState'] = now + 20_000_000
  publisher = Publisher()
  with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=now + 30_000_000):
    planner.publish(state, publisher)
  plan = publisher.messages['longitudinalPlan'].longitudinalPlan
  assert plan.radarStateInputMonoTime == chosen_ns
  assert plan.radarStateInputMonoTime != state.logMonoTime['radarState']
  assert planner.output_a_target == target and planner.motion_intent == intent

  now = advance(state, 1)
  planner.update(state, now_ns=now + 500_000)
  with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=now + 1_000_000):
    planner.publish(state, publisher)
  assert publisher.messages['longitudinalPlan'].longitudinalPlan.radarStateInputMonoTime == now


def test_legacy_planner_without_radar_event_identity_keeps_zero():
  planner, _ = make_planner(platform=CAR.HONDA_CIVIC, diagnostics=False)
  state = make_state()
  now = advance(state, 0)
  state.logMonoTime.pop('radarState')
  planner.update(state, now_ns=now + 500_000)
  # An arriving identity at publish time cannot fill in an unknown update input.
  state.logMonoTime['radarState'] = now + 500_000
  publisher = Publisher()
  with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=now + 1_000_000):
    planner.publish(state, publisher)
  assert not planner.unified
  assert publisher.messages['longitudinalPlan'].longitudinalPlan.radarStateInputMonoTime == 0
