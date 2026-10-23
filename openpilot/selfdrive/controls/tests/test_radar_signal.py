"""Radar-off contracts at acquisition, planning, and the direct SP consumer."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.interfaces import RadarInterfaceBase
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from opendbc.car.honda.tests.test_bosch_a_observer import START, scan_frames
from openpilot.cereal import custom, log
from openpilot.common.params import Params, ParamKeyFlag
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.common.radar_signal import RadarSignalInput, radar_signal_enabled, radar_state_for_mode
from openpilot.selfdrive.car.card import Car
from openpilot.selfdrive.car.honda_radar_observer import HondaRadarObserver
from openpilot.selfdrive.car.honda_radar_observer_config import ObserverSettings, observer_requested
from openpilot.selfdrive.car.tests.test_actuation_observation import CardHarness
from openpilot.selfdrive.car.tests.test_honda_radar_observer import can_message, configuration, state
from openpilot.selfdrive.controls.radard import RadarD
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import Publisher, advance, make_planner, make_state
from openpilot.selfdrive.controls.tests.test_radard_publication import Inputs, publish_at
from openpilot.sunnypilot.selfdrive.controls.controlsd_ext import ControlsExt


def contaminated_radar():
  radar = log.RadarState.new_message(mdMonoTime=1_000_000_000, fusionActive=True, fusionHealthy=True,
                                     fusionRiskUnresolved=True, fusionSessionId='previous-drive')
  radar.leadOne = {'present': True, 'radar': True, 'dRel': 2.0, 'vLead': 0.0, 'vLeadK': 0.0,
                  'vRel': -10.0, 'modelProb': .99, 'aLeadTau': 1.5}
  radar.visionLeadOne = radar.leadOne  # Historical standard fusion can populate this field too.
  radar.leadTwo = {'present': True, 'radar': False, 'dRel': 40.0, 'vLead': 10.0, 'vLeadK': 10.0,
                  'modelProb': .99, 'aLeadTau': 1.5}
  radar.visionLeadTwo = radar.leadTwo
  radar.radarObjects = [radar.leadOne.to_dict()]
  radar.radarErrors.canError = True
  radar.fusionClearedTracks = [7]
  return radar


def test_default_enabled_and_off_persists_across_drive_boundaries():
  cp, _ = configuration()
  with OpenpilotPrefix():
    params = Params()
    assert radar_signal_enabled(params)
    assert params.get_default_value('RadarSignal') is True
    for name in ('AlphaLongitudinalEnabled', 'ExperimentalMode', 'DynamicExperimentalControl'):
      params.put_bool(name, True, block=True)
    assert observer_requested(True, params, cp)
    params.put_bool('RadarSignal', False, block=True)
    for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION,
                 ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION, ParamKeyFlag.CLEAR_ON_IGNITION_ON):
      params.clear_all(flag)
      assert not radar_signal_enabled(params)
      assert not observer_requested(True, params, cp)
    params.put_bool('RadarSignal', True, block=True)
    assert observer_requested(True, params, cp)


def test_card_latches_setting_and_never_constructs_a_sensor_parser_when_off():
  cp = CarInterface.get_non_essential_params(CAR.HONDA_ACCORD)
  cp_sp = CarInterface.get_non_essential_params_sp(cp, CAR.HONDA_ACCORD)
  ci = SimpleNamespace(CP=cp, CP_SP=cp_sp, CC=None)
  sensor = Mock()
  with OpenpilotPrefix(), patch('openpilot.selfdrive.car.card.messaging.sub_sock'), \
       patch('openpilot.selfdrive.car.card.messaging.SubMaster'), patch('openpilot.selfdrive.car.card.messaging.PubMaster'), \
       patch('openpilot.selfdrive.car.card.sunnypilot_interfaces.log_fingerprint'), \
       patch('openpilot.selfdrive.car.card.set_alternative_experience'), patch('openpilot.selfdrive.car.card.set_car_specific_params'):
    params = Params()
    params.put_bool('RadarSignal', False, block=True)
    car = Car(CI=ci, RI=sensor)
    assert type(car.RI) is RadarInterfaceBase
    assert car.RI.rcp is None and not car.radar_signal
    params.put_bool('RadarSignal', True, block=True)
    assert not car.radar_signal
    restarted = Car(CI=ci, RI=sensor)
    assert restarted.radar_signal and restarted.RI is sensor
    sensor.update.assert_not_called()


def test_card_keeps_vehicle_state_but_passes_no_can_to_radar_when_off():
  harness = CardHarness(enabled=False)
  harness.card.radar_signal = False
  harness.card.RI = RadarInterfaceBase(harness.card.CP, harness.card.CP_SP)
  with patch.object(harness.card.CI, 'update', wraps=harness.card.CI.update) as vehicle_update, \
       patch.object(harness.card.RI, 'update', wraps=harness.card.RI.update) as radar_update:
    for _ in range(5):
      harness.step()
    assert vehicle_update.call_count == 5
    assert radar_update.call_count == 5
    assert all(call.args == ([],) for call in radar_update.call_args_list)
  tracks = harness.card.pm.messages['radarTracks']
  assert tracks.valid and len(tracks.radarTracks.points) == 0


def test_honda_observer_off_clears_decoder_and_does_not_read_any_sensor_inputs():
  cp, cp_sp = configuration()
  observer = HondaRadarObserver(cp, cp_sp, 'signal-test')
  settings = ObserverSettings(True, True, True)
  sm = state(START)
  assert observer.update([can_message(START, scan_frames())], sm, settings, START + 1) is not None
  with patch.object(observer.decoder, 'feed', side_effect=AssertionError('radar decoded while off')):
    assert observer.update([object()], {}, replace(settings, radar_signal=False), START + 50_000_000, {}) is None
  assert not observer.active and observer.fusion is None
  assert observer.decoder.last_frame_ns == 0
  fresh = observer.update([], state(START + 100_000_000), settings, START + 100_000_000)
  assert len(fresh.hondaRadarObservations.observations) == 0
  assert fresh.hondaRadarObservations.lastRadarFrameMonoTime == 0


def test_disabled_observer_main_exits_before_creating_any_subscription():
  from openpilot.selfdrive.car import honda_radar_observer
  with OpenpilotPrefix():
    Params().put_bool('RadarSignal', False, block=True)
    with patch.object(honda_radar_observer.messaging, 'sub_sock') as can, \
         patch.object(honda_radar_observer.messaging, 'SubMaster') as inputs, \
         patch.object(honda_radar_observer, 'HondaRadarObserver') as observer:
      honda_radar_observer.main()
    can.assert_not_called()
    inputs.assert_not_called()
    observer.assert_not_called()


def test_disabled_radard_uses_current_vision_without_reading_radar_or_its_errors():
  cp, cp_sp = configuration()
  with OpenpilotPrefix():
    params = Params()
    params.put_bool('RadarSignal', False, block=True)
    rd = RadarD(cp, cp_sp, params=params)
    inputs = Inputs(START)
    inputs.valid['radarTracks'] = False
    inputs['modelV2'].velocity.x = [10.0]
    inputs['modelV2'].leadsV3 = [
      {'prob': .99, 'x': [21.52], 'y': [0.0], 'v': [10.0], 'a': [0.0]},
      {'prob': 0.0, 'x': [100.0], 'y': [0.0], 'v': [10.0], 'a': [0.0]},
    ]
    rd.tracks[7] = object()
    rd.fusion_control = Mock()
    # An unreadable input proves the disabled path does not even inspect points/errors.
    rd.update(inputs, object(), now_ns=START)
    event = publish_at(rd, START + 1)
    assert event.valid and event.radarState.leadOne.present
    assert event.radarState.leadOne.dRel == pytest.approx(20.0)
    assert not event.radarState.leadOne.radar
    assert not any(event.radarState.radarErrors.to_dict().values())
    assert not rd.tracks and rd.fusion_control is None
    assert not event.radarState.fusionActive and not event.radarState.radarObjects
    inputs.valid['modelV2'] = False
    rd.update(inputs, object(), now_ns=START + 50_000_000)
    assert not rd.radar_state_valid  # Radar-off does not waive model health.
    params.put_bool('RadarSignal', True, block=True)
    assert not rd.radar_signal
    restarted = RadarD(cp, cp_sp, params=params)
    assert restarted.radar_signal and not restarted.tracks and restarted.fusion_control is None


@pytest.mark.parametrize('source', ['legacy', 'radar', 'fused', 'predicted'])
def test_disabled_view_discards_all_radar_provenance_and_preserves_current_vision(source):
  radar = contaminated_radar()
  radar.leadOne.radar = source == 'legacy'
  radar.leadOne.evidence.source = source
  radar.visionLeadOne = radar.leadOne
  original = radar.to_dict()
  assert radar_state_for_mode(radar, True) is radar
  filtered = radar_state_for_mode(radar.as_reader(), False)
  assert not filtered.leadOne.present and filtered.leadOne.dRel == 0.0
  assert not filtered.visionLeadOne.present
  assert filtered.leadTwo.to_dict() == radar.leadTwo.to_dict()
  assert not filtered.radarObjects and not filtered.fusionClearedTracks
  assert not filtered.fusionActive and not filtered.fusionRiskUnresolved
  assert not any(filtered.radarErrors.to_dict().values())
  assert radar.to_dict() == original


@pytest.mark.parametrize('interceptor', [False, True], ids=['unified', 'legacy'])
def test_both_planners_ignore_injected_radar_and_publish_the_same_vision_result(interceptor):
  with OpenpilotPrefix():
    Params().put_bool('RadarSignal', False, block=True)
    candidate, _ = make_planner(speed=10.0, interceptor=interceptor)
    baseline, _ = make_planner(speed=10.0, interceptor=interceptor)
    incoming, clean = make_state(speed=10.0), make_state(speed=10.0)
    incoming['radarState'] = contaminated_radar()
    clean['radarState'] = radar_state_for_mode(incoming['radarState'].as_reader(), False)
    for frame in range(12):
      now = advance(incoming, frame)
      advance(clean, frame)
      candidate.update(incoming, now_ns=now)
      baseline.update(clean, now_ns=now)
      assert candidate.mpc.solution_status == baseline.mpc.solution_status == 0
      np.testing.assert_allclose(candidate.a_desired_trajectory, baseline.a_desired_trajectory, atol=1e-9)
      assert candidate.output_should_stop == baseline.output_should_stop
      assert candidate.fcw == baseline.fcw
    pm = Publisher()
    candidate.publish(incoming, pm)
    plan = pm.messages['longitudinalPlan'].longitudinalPlan
    assert not plan.hasLead and not plan.objectInfluences
    assert incoming['radarState'].leadOne.present and incoming['radarState'].fusionActive
    assert RadarSignalInput(incoming, False).valid is incoming.valid


def test_direct_car_control_sp_path_removes_radar_but_keeps_visual_lead():
  controls = ControlsExt.__new__(ControlsExt)
  controls.radar_signal = False
  radar = contaminated_radar()
  inputs = {'radarState': radar.as_reader(), 'selfdriveStateSP': custom.SelfdriveStateSP.new_message().as_reader()}
  output = controls.state_control_ext(inputs)
  assert not output.leadOne.status and not output.leadOne.radar and output.leadOne.dRel == 0.0
  assert output.leadTwo.status and not output.leadTwo.radar and output.leadTwo.dRel == 40.0
  controls.radar_signal = True
  enabled = controls.state_control_ext(inputs)
  assert enabled.leadOne.status and enabled.leadOne.radar and enabled.leadOne.dRel == 2.0
