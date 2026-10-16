from types import SimpleNamespace

import pytest

from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController
from openpilot.selfdrive.controls.tests.test_lead_qualification import formal_lead

class MockLeadOne:
  def __init__(self, present=0.0):
    self.present = present

class MockRadarState:
  def __init__(self, present=0.0):
    self.leadOne = MockLeadOne(present=present)

class MockCarState:
  def __init__(self, vEgo=0.0, vCruise=0.0, standstill=False):
    self.vEgo = vEgo
    self.vCruise = vCruise
    self.standstill = standstill

class MockModelData:
  def __init__(self, valid=True):
    size = 33 if valid else 10  # incomplete if invalid
    self.position = type("Pos", (), {"x": [0.0] * size})()
    self.orientation = type("Ori", (), {"x": [0.0] * size})()

class MockSelfDriveState:
  def __init__(self, experimentalMode=False):
    self.experimentalMode = experimentalMode

class MockParams:
  def get_bool(self, name):
    return True

def default_sm():
  sm = {
    'carState': MockCarState(vEgo=10.0, vCruise=20.0),
    'radarState': MockRadarState(present=1.0),
    'modelV2': MockModelData(valid=True),
    'selfdriveState': MockSelfDriveState(experimentalMode=True),
  }
  return sm

def mock_cp():
  class CP:
    radarUnavailable = False
  return CP()

def mock_mpc():
  class MPC:
    crash_cnt = 0
  return MPC()

# Fake Kalman Filter that always returns a given value
class FakeKalman:
  def __init__(self, value=1.0):
    self.value = value
  def add_data(self, v): pass
  def get_value(self): return self.value
  def get_confidence(self): return 1.0
  def reset_data(self): pass


def fusion_sm(*, model_stop=False):
  sm = default_sm()
  lead = formal_lead(vLead=10.0)
  sm['radarState'] = SimpleNamespace(fusionActive=True, fusionHealthy=True, radarObjects=[lead],
                                     leadOne=lead, leadTwo=SimpleNamespace(present=False))
  sm['modelV2'].position.x = [0.0 if model_stop else float(index) * 10.0 for index in range(33)]
  return sm


@pytest.mark.parametrize('radar_unavailable', [False, True])
def test_fusion_uses_ordinary_leads_without_reinterpreting_the_platform_parser(radar_unavailable):
  cp = mock_cp()
  cp.radarUnavailable = radar_unavailable
  controller = DynamicExperimentalController(cp, mock_mpc(), params=MockParams())
  sm = fusion_sm()
  controller.update(sm)
  assert controller._has_lead_filtered
  assert controller._fusion_risks == (sm['radarState'].leadOne,)
  assert controller.mode() == 'acc'
  assert cp.radarUnavailable is radar_unavailable


@pytest.mark.parametrize('reason', ['model_stop', 'fcw', 'standstill'])
def test_fusion_leads_never_mask_model_stop_fcw_or_standstill_mode(reason):
  cp, mpc = mock_cp(), mock_mpc()
  controller = DynamicExperimentalController(cp, mpc, params=MockParams())
  sm = fusion_sm(model_stop=reason == 'model_stop')
  mpc.crash_cnt = int(reason == 'fcw')
  sm['carState'].standstill = reason == 'standstill'
  for _ in range(20):
    controller.update(sm)
  assert controller.mode() == 'blended'


def test_invalid_formal_radar_object_does_not_create_a_detected_lead():
  controller = DynamicExperimentalController(mock_cp(), mock_mpc(), params=MockParams())
  sm = fusion_sm()
  sm['radarState'].leadOne.evidence.positionMonoTime -= 300_000_000
  controller.update(sm)
  assert not controller._has_lead_filtered
  assert controller._fusion_risks == ()


@pytest.mark.parametrize('malformed', [False, True])
def test_extra_raw_radar_objects_cannot_make_dec_detect_a_lead(malformed):
  sm = fusion_sm()
  sm['radarState'].leadOne = SimpleNamespace(present=False)
  if malformed:
    sm['radarState'].radarObjects[0].dRel = float('nan')
  controller = DynamicExperimentalController(mock_cp(), mock_mpc(), params=MockParams())
  for _ in range(20):
    controller.update(sm)
  assert not controller._has_lead_filtered and controller._fusion_risks == ()


@pytest.mark.parametrize('source', ['radar', 'fused', 'predicted'])
@pytest.mark.parametrize('path,start', [('entering', 0.0), ('entering', .5), ('inPath', .5)])
def test_adjacent_sources_need_current_in_path_qualification(source, path, start):
  sm = fusion_sm()
  candidate = formal_lead(source=source, vLead=10.0,
                          evidence={'pathState': path, 'conflictStart': start, 'conflictEnd': 2.0,
                                    'followQualified': False, 'stopQualified': False, 'departureQualified': False,
                                    'fcwQualified': source != 'predicted', 'accelerationValid': source != 'predicted'})
  sm['radarState'].radarObjects = [candidate]
  sm['radarState'].leadOne = candidate
  controller = DynamicExperimentalController(mock_cp(), mock_mpc(), params=MockParams())
  controller.update(sm)
  assert not controller._has_lead_filtered and controller._fusion_risks == ()
  candidate.evidence.pathState, candidate.evidence.conflictStart = 'inPath', 0.0
  for _ in range(20):
    controller.update(sm)
  if source == 'predicted':
    # Crossing a path boundary cannot make an unmeasured object current.
    assert not controller._has_lead_filtered and controller._fusion_risks == ()
  else:
    assert controller._has_lead_filtered and controller._fusion_risks == (candidate,)


class TestDynamicExperimentalController(OpenpilotTestCase):
  def test_initial_mode_is_acc(self, mock_cp, mock_mpc):
    controller = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams())
    assert controller.mode() == "acc"

  def test_standstill_triggers_blended(self, mock_cp, mock_mpc, default_sm):
    controller = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams())
    default_sm['carState'].standstill = True
    for _ in range(10):
      controller.update(default_sm)
    assert controller.mode() == "blended"

  def test_emergency_blended_on_fcw(self, mock_cp, mock_mpc, default_sm):
    controller = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams())
    mock_mpc.crash_cnt = 1  # simulate FCW
    for _ in range(2):
      controller.update(default_sm)
    assert controller.mode() == "blended"

  def test_radarless_slowdown_triggers_blended(self, mock_cp, mock_mpc, default_sm):
    mock_cp.radarUnavailable = True
    controller = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams())

    # Force conditions to simulate slowdown
    controller._slow_down_filter = FakeKalman(value=1.0)  # ty: ignore[invalid-assignment]
    controller._v_ego_kph = 35.0
    default_sm['modelV2'] = MockModelData(valid=False)  # Incomplete trajectory

    for _ in range(3):
      controller.update(default_sm)

    assert controller.mode() == "blended"
