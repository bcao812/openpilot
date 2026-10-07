"""Admission/fault contracts; these bounds do not establish sensor calibration."""
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.car.honda_radar_observer_config import ASSIST, ObserverSettings
from openpilot.selfdrive.controls.lib.honda_radar_accel_guard import HondaRadarAccelGuard, MAX_VISION_GAP_NS, RELEASE_JERK


START = 1_000_000_000
STEP = 66_000_000


def guard_input(frame, *, gap=False, step=STEP, speed=10.0):
  t = START + frame * step
  probability = 0.2 if gap else 0.99
  vision = {'prob': probability, 'x': [21.52] * 6, 'y': [0.0] * 6, 'v': [speed] * 6, 'a': [0.0] * 6,
            'xStd': [1.0] * 6, 'yStd': [0.4] * 6, 'vStd': [1.0] * 6}
  track = {'trackId': 1, 'oemTrackId': 5, 'generation': 1, 'positionMonoTime': t, 'velocityMonoTime': t,
           'positionSeq': frame + 1, 'dRel': 20.0, 'yRel': 0.0, 'groundSpeed': speed, 'lateralSpeed': 0.0,
           'positionVariance': 0.2, 'velocityVariance': 0.5, 'observations': 10 + frame,
           'active': True, 'mature': True, 'velocityGood': True, 'coasting': False, 'pathState': 'inPath',
           'pathOffset': 0.0, 'motion': 'moving'}
  lead = {'modelIndex': 0, 'present': not gap, 'probability': probability, 'trackId': 0 if gap else 1,
          'candidateTrackId': 0 if gap else 1, 'identityState': 'unresolved' if gap else 'same',
          'reason': 'noVision' if gap else 'matched', 'rejectionMonoTime': 0, 'pathState': 'inPath',
          'bindingAnchorX': 0.0, 'bindingAnchorY': 0.0, 'bindingSamples': 10 + frame,
          'bindingStartMonoTime': START - 200_000_000, 'independentPair': not gap}
  fusion = {'version': 1, 'algorithmVersion': 'bosch-a-shadow-3', 'enabled': True, 'active': True,
            'calibrated': False, 'controlAllowed': False, 'modelMonoTime': t, 'modelTimestampEof': t,
            'egoMonoTime': t, 'counters': {'resets': 0}, 'tracks': [track], 'leads': [lead]}
  scope = dict.fromkeys(('alphaLong', 'experimentalSetting', 'decSetting', 'experimentalActive',
                        'decEnabled', 'decActive', 'longActive'), True)
  scope.update(selfdriveMonoTime=t, planMonoTime=t)
  observation = {'schemaVersion': 1, 'active': True, 'reason': 'observing', 'sessionId': 'fixture',
                 'sequence': frame + 1, 'scope': scope, 'fusion': fusion, 'calibrated': False,
                 'counters': {'dropped': 0}, 'rejectedEnvelopes': 0}
  return {'now_ns': t + 1_000_000, 'enabled': True, 'observation': observation, 'observation_ns': t + 500_000,
          'observation_valid': True, 'model': {'timestampEof': t, 'leadsV3': [vision],
          'velocity': {'x': [speed]}, 'orientationRate': {'z': [0.0]}}, 'model_ns': t,
          'speed': speed, 'standstill': False, 'ceiling_max': 2.0}


def confirm(guard):
  for frame in range(3):
    assert guard.update(**guard_input(frame)) is None
    assert guard.qualified is (frame == 2)


def test_requires_three_independent_pairs_over_120_ms():
  guard = HondaRadarAccelGuard()
  confirm(guard)
  assert guard.update(**guard_input(3, gap=True)) == 0.0
  assert guard.qualified and guard.holding
  assert guard.reason == 'visionGap'

  fast = HondaRadarAccelGuard()
  for frame in range(3):
    fast.update(**guard_input(frame, step=50_000_000))
    assert not fast.qualified
  fast.update(**guard_input(3, gap=True, step=50_000_000))
  assert fast.accel_ceiling is None


@pytest.mark.parametrize('repeated', ['position', 'camera', 'publication'])
def test_repetition_cannot_add_independent_evidence(repeated):
  guard = HondaRadarAccelGuard()
  first = guard_input(0)
  for frame in range(3):
    values = guard_input(frame)
    fusion = values['observation']['fusion']
    if repeated == 'position':
      fusion['tracks'][0].update(positionSeq=1, positionMonoTime=START)
    elif repeated == 'camera':
      fusion.update(modelMonoTime=START, modelTimestampEof=START)
    else:
      values.update(observation=first['observation'], observation_ns=first['observation_ns'])
    guard.update(**values)
    assert not guard.qualified and guard.accel_ceiling is None


def test_empty_road_and_future_camera_hypotheses_have_no_authority():
  guard = HondaRadarAccelGuard()
  for frame in range(12):
    assert guard.update(**guard_input(frame, gap=True)) is None
  guard.clear()
  for frame in range(5):
    values = guard_input(frame)
    values['observation']['fusion']['leads'][0]['modelIndex'] = 1
    assert guard.update(**values) is None
    assert not guard.qualified


FAULTS = {
  'invalid_message': lambda v: v.update(observation_valid=False),
  'stale_message': lambda v: v.update(observation_ns=v['now_ns'] - 201_000_000),
  'future_message': lambda v: v.update(observation_ns=v['now_ns'] + 1),
  'session': lambda v: v['observation'].update(sessionId='restarted'),
  'sequence_gap': lambda v: v['observation'].update(sequence=20),
  'sequence_reordered': lambda v: v['observation'].update(sequence=2),
  'producer_fault': lambda v: v['observation']['fusion'].update(active=False),
  'old_fusion_version': lambda v: v['observation']['fusion'].update(algorithmVersion='bosch-a-shadow-2'),
  'dropped_event': lambda v: v['observation']['counters'].update(dropped=1),
  'position_invalid': lambda v: v['observation']['fusion']['tracks'][0].update(active=False),
  'velocity_invalid': lambda v: v['observation']['fusion']['tracks'][0].update(velocityGood=False),
  'position_stale': lambda v: v['observation']['fusion']['tracks'][0].update(positionMonoTime=START),
  'position_order': lambda v: v['observation']['fusion']['tracks'][0].update(positionSeq=1),
  'coasting': lambda v: v['observation']['fusion']['tracks'][0].update(coasting=True),
  'unknown_path': lambda v: v['observation']['fusion']['tracks'][0].update(pathState='unknown'),
  'entering_path': lambda v: v['observation']['fusion']['tracks'][0].update(pathState='entering'),
  'outside_corridor': lambda v: v['observation']['fusion']['tracks'][0].update(pathOffset=2.0),
  'generation': lambda v: v['observation']['fusion']['tracks'][0].update(generation=2),
  'new_track': lambda v: v['observation']['fusion']['tracks'][0].update(trackId=2),
  'nonfinite': lambda v: v['observation']['fusion']['tracks'][0].update(dRel=float('nan')),
  'motion_jump': lambda v: v['observation']['fusion']['tracks'][0].update(groundSpeed=15.0),
  'different_identity': lambda v: v['observation']['fusion']['leads'][0].update(identityState='different'),
  'closer_weak_vision': lambda v: v['model']['leadsV3'][0].update(prob=0.7, x=[8.0] * 6),
  'closer_vision': lambda v: v['model']['leadsV3'][0].update(prob=0.99, x=[8.0] * 6),
}


@pytest.mark.parametrize('fault', FAULTS)
def test_fault_revokes_identity_immediately_and_release_is_bounded(fault):
  guard = HondaRadarAccelGuard()
  confirm(guard)
  assert guard.update(**guard_input(3, gap=True)) == 0.0
  values = guard_input(4, gap=True)
  FAULTS[fault](values)
  ceiling = guard.update(**values)
  assert not guard.qualified and not guard.holding and guard.track_key is None
  assert ceiling == pytest.approx(RELEASE_JERK * STEP / 1e9)
  previous = ceiling
  for frame in range(5, 38):
    values = guard_input(frame, gap=True)
    values['observation_valid'] = False
    ceiling = guard.update(**values)
    if ceiling is not None:
      assert 0.0 <= ceiling - previous <= RELEASE_JERK * STEP / 1e9 + 1e-12
      previous = ceiling
    assert not guard.qualified and not guard.holding
  assert ceiling is None


def test_gap_clock_is_not_refreshed_by_new_radar_samples():
  guard = HondaRadarAccelGuard()
  confirm(guard)
  last_confirmed = guard.last_visible_ns
  for frame in range(3, 13):
    values = guard_input(frame, gap=True)
    ceiling = guard.update(**values)
    if values['now_ns'] - last_confirmed <= MAX_VISION_GAP_NS:
      assert ceiling == 0.0 and guard.holding
    else:
      assert ceiling > 0.0 and not guard.holding and not guard.qualified


def test_recovery_releases_ceiling_without_a_positive_step_or_reusing_changed_identity():
  guard = HondaRadarAccelGuard()
  confirm(guard)
  guard.update(**guard_input(3, gap=True))
  assert guard.update(**guard_input(4)) == pytest.approx(STEP / 1e9)
  values = guard_input(5)
  values['observation']['fusion']['leads'][0].update(trackId=2, candidateTrackId=2)
  guard.update(**values)
  assert not guard.qualified
  guard.update(**guard_input(6, gap=True))
  assert not guard.holding and guard.accel_ceiling > 0.0


@pytest.mark.parametrize('change', [{'enabled': False}, {'standstill': True}, {'speed': 0.5}, {'now_ns': START}])
def test_disengagement_stationary_or_bad_consumer_clock_drops_the_optional_cap(change):
  guard = HondaRadarAccelGuard()
  confirm(guard)
  guard.update(**guard_input(3, gap=True))
  values = guard_input(4, gap=True)
  values.update(change)
  assert guard.update(**values) is None
  assert not guard.qualified and not guard.holding


@pytest.mark.parametrize('mode', [0, 1, 2, 3, 4, -1])
def test_assist_setting_has_separate_platform_and_mode_admission(mode):
  cp = SimpleNamespace(carFingerprint='HONDA_ACCORD', openpilotLongitudinalControl=True, notCar=False)
  cp_sp = SimpleNamespace(enableGasInterceptor=False)
  settings = ObserverSettings(mode, True, True, True)
  assert settings.assist_requested(cp, cp_sp) is (mode == ASSIST)
  cp_sp.enableGasInterceptor = True
  assert not settings.assist_requested(cp, cp_sp)
  cp_sp.enableGasInterceptor = False
  for field, value in [('carFingerprint', 'HONDA_CIVIC'), ('openpilotLongitudinalControl', False), ('notCar', True)]:
    changed = SimpleNamespace(**vars(cp))
    setattr(changed, field, value)
    assert not settings.assist_requested(changed, cp_sp)
  for alpha, experimental, dec in [(False, True, True), (True, False, True), (True, True, False)]:
    assert not ObserverSettings(ASSIST, alpha, experimental, dec).assist_requested(cp, cp_sp)
