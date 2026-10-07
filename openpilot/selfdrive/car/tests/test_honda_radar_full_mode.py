"""Perception startup and explicit full-control admission are separate contracts."""
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.car.honda_radar_observer_config import ASSIST, CONTROL, SHADOW, ObserverSettings, scope_state
from openpilot.selfdrive.car.tests.test_honda_radar_observer import configuration, state
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import UnifiedLongitudinalPlanner
from openpilot.cereal import log


def test_control_perception_starts_before_dec_or_planner_has_published():
  cp, cp_sp = configuration()
  sm = state(1_000_000_000)
  for service in ('selfdriveState', 'longitudinalPlanSP'):
    sm.seen[service] = sm.valid[service] = False
    sm.logMonoTime[service] = 0
  sm['selfdriveState'].experimentalMode = False
  sm['longitudinalPlanSP'].dec.enabled = sm['longitudinalPlanSP'].dec.active = False
  assert scope_state(cp, cp_sp, ObserverSettings(CONTROL, True, True, True), sm, 1_000_000_000)[0]
  for mode in (SHADOW, ASSIST):
    assert not scope_state(cp, cp_sp, ObserverSettings(mode, True, True, True), sm, 1_000_000_000)[0]


@pytest.mark.parametrize('setting', [ObserverSettings(3, True, True, True), ObserverSettings(4, False, True, True),
                                   ObserverSettings(4, True, False, True), ObserverSettings(4, True, True, False)])
def test_full_control_requires_its_explicit_mode_and_all_target_settings(setting):
  cp, cp_sp = configuration()
  assert not setting.control_requested(cp, cp_sp)
  assert ObserverSettings(CONTROL, True, True, True).control_requested(cp, cp_sp)
  cp_sp.enableGasInterceptor = True
  assert not ObserverSettings(CONTROL, True, True, True).control_requested(cp, cp_sp)


def test_consumer_scope_exit_uses_untouched_vision_without_mutating_published_radar():
  cp, cp_sp = configuration()
  planner = SimpleNamespace(CP=cp, CP_SP=cp_sp, unified=True, dec=SimpleNamespace(enabled=lambda: True))
  consumer = UnifiedLongitudinalPlanner.__new__(UnifiedLongitudinalPlanner)
  consumer.planner = planner
  consumer.radar_settings = ObserverSettings(SHADOW, True, True, True)
  consumer.refresh_radar_settings = lambda now: None
  published = log.RadarState.new_message(fusionActive=True, mdMonoTime=1_000_000_000)
  published.leadOne = {'present': True, 'radar': True, 'modelProb': 0.0, 'dRel': 10.0}
  published.visionLeadOne = {'present': True, 'radar': False, 'modelProb': .99, 'dRel': 25.0}
  inputs = {'radarState': published, 'selfdriveState': SimpleNamespace(experimentalMode=True)}
  fallback = consumer.radar_input_state(inputs, 1_000_000_000)
  assert fallback['radarState'].leadOne.dRel == 25.0
  assert not fallback['radarState'].fusionActive
  assert published.leadOne.dRel == 10.0 and published.fusionActive
  assert not consumer.fusion_admitted
