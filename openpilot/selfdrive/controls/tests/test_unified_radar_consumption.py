"""All planner consumers share expiry and purpose decisions for this tick."""
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.common.params import Params
from openpilot.selfdrive.car.honda_radar_observer_config import CONTROL, SHADOW
from openpilot.selfdrive.controls.lib.lead_qualification import control_leads
from openpilot.selfdrive.controls.lib.longitudinal_planner import has_safe_moving_lead
from openpilot.selfdrive.controls.lib.radar_consumption import RadarConsumptionView
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import has_departing_lead
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_planner, make_state
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import CapturingPublisher
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_lead_qualification import caution_lead, formal_lead
from openpilot.selfdrive.controls.tests.test_radar_consumption import (
  CONSUMPTION, PUBLICATION, SESSION, coast, native_frame, radar_frame,
)


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


def test_relevant_consumer_loss_keeps_zero_cap_and_hold_identity_until_driver_reset():
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
def test_only_explicit_producer_loss_claim_requests_the_main_plan_alert(degraded, risk_unresolved):
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


def test_claim_alone_does_not_change_mpc_objects_acceleration_or_motion_intent():
  outputs = []
  for risk in (False, True):
    planner = enabled_planner(5.0)
    primary = formal_lead(now_ns=PUBLICATION, dRel=40.0, vLead=5.5, vLeadK=5.5, vRel=.5,
                          evidence={'trackId': 5})
    state = state_at(radar_frame(primary, display=(primary,), degraded=True, risk_unresolved=risk), speed=5.0)
    with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
      planner.update(state, now_ns=CONSUMPTION)
    assert planner.mpc.solution_status == 0 and planner.motion_intent.valid
    publisher = CapturingPublisher()
    with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=CONSUMPTION):
      planner.publish(state, publisher)
    plan = publisher.messages['longitudinalPlan'].longitudinalPlan
    assert plan.radarRiskUnresolved is risk
    outputs.append((planner.output_a_target, planner.mpc.params.copy(), plan.motionIntent.to_dict(),
                    [row.to_dict() for row in plan.objectInfluences]))
  assert outputs[0][0] == outputs[1][0]
  np.testing.assert_array_equal(outputs[0][1], outputs[1][1])
  assert outputs[0][2:] == outputs[1][2:]


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
