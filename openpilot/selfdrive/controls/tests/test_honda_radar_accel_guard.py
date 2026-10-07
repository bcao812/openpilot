"""Admission/fault contracts; these bounds do not establish sensor calibration."""
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.car.honda_radar_observer_config import ASSIST, ObserverSettings
from openpilot.selfdrive.controls.lib.honda_radar_accel_guard import HondaRadarAccelGuard, MAX_VISION_GAP_NS, RELEASE_JERK


START = 1_000_000_000
STEP = 66_000_000


def guard_input(frame, *, gap=False, step=STEP, speed=10.0, lead_speed=None, distance=20.0, probability=None):
  t = START + frame * step
  lead_speed = speed if lead_speed is None else lead_speed
  distance += (lead_speed - speed) * frame * step / 1e9
  probability = (0.2 if gap else 0.99) if probability is None else probability
  vision = {'prob': probability, 'x': [distance + 1.52] * 6, 'y': [0.0] * 6, 'v': [lead_speed] * 6, 'a': [0.0] * 6,
            'xStd': [1.0] * 6, 'yStd': [0.4] * 6, 'vStd': [1.0] * 6}
  track = {'trackId': 1, 'oemTrackId': 5, 'generation': 1, 'positionMonoTime': t, 'velocityMonoTime': t,
           'positionSeq': frame + 1, 'dRel': distance, 'yRel': 0.0, 'groundSpeed': lead_speed, 'lateralSpeed': 0.0,
           'positionVariance': 0.2, 'velocityVariance': 0.5, 'observations': 10 + frame,
           'active': True, 'mature': True, 'velocityGood': True, 'coasting': False, 'pathState': 'inPath',
           'pathOffset': 0.0, 'motion': 'stationary' if abs(lead_speed) < 1.0 else 'moving'}
  lead = {'modelIndex': 0, 'present': probability > 0.5, 'probability': probability, 'trackId': 0 if gap else 1,
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
          'speed': speed, 'standstill': False, 'ceiling_max': 2.0, 'formal_lead_present': not gap}


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


@pytest.mark.parametrize('lead_speed,holding', [(0.0, True), (9.0, True), (10.0, True), (10.1, False), (12.0, False)])
def test_far_target_gap_holds_only_when_confirmed_target_is_not_pulling_away(lead_speed, holding):
  guard = HondaRadarAccelGuard()
  for frame in range(3):
    values = guard_input(frame, lead_speed=lead_speed, distance=80.0)
    assert values['formal_lead_present'] is True
    assert guard.update(**values) is None
  assert guard.qualified
  last_confirmed = guard.last_visible_ns
  for frame in range(3, 8):
    values = guard_input(frame, gap=True, lead_speed=lead_speed, distance=80.0)
    assert values['formal_lead_present'] is False
    ceiling = guard.update(**values)
    assert guard.qualified and guard.last_visible_ns == last_confirmed
    assert guard.holding is holding
    assert (ceiling == 0.0) if holding else (ceiling is None)
    assert guard.reason == ('visionGap' if holding else 'radarPullingAway')


@pytest.mark.parametrize('probability', [0.2, 0.51, 0.65, 0.79])
def test_weak_but_present_formal_lead_does_not_start_or_extend_a_cap(probability):
  guard = HondaRadarAccelGuard()
  confirm(guard)
  last_confirmed = guard.last_visible_ns
  for frame in range(3, 8):
    values = guard_input(frame, gap=True, probability=probability)
    values['formal_lead_present'] = True
    assert guard.update(**values) is None
    assert guard.qualified and not guard.holding and guard.last_visible_ns == last_confirmed
    assert guard.reason == 'formalLeadPresent'
  # Losing both formal leads can use only the remainder of the original
  # independent-camera deadline; the weak-probability interval bought no time.
  assert guard.update(**guard_input(8, gap=True)) == 0.0
  assert guard.last_visible_ns == last_confirmed
  guard.update(**guard_input(9, gap=True))
  assert guard.update(**guard_input(10, gap=True)) > 0.0
  assert not guard.holding and not guard.qualified


@pytest.mark.parametrize('recovery', ['formal_lead', 'pulling_away'])
def test_formal_lead_return_or_opening_target_only_releases_existing_cap(recovery):
  guard = HondaRadarAccelGuard()
  confirm(guard)
  assert guard.update(**guard_input(3, gap=True)) == 0.0
  last_confirmed = guard.last_visible_ns
  values = guard_input(4, gap=True, probability=0.65)
  if recovery == 'formal_lead':
    values['formal_lead_present'] = True
  else:
    values['observation']['fusion']['tracks'][0]['groundSpeed'] = 10.1
  assert guard.update(**values) == pytest.approx(RELEASE_JERK * STEP / 1e9)
  assert guard.qualified and not guard.holding and guard.last_visible_ns == last_confirmed


def test_opening_veto_uses_current_projection_instead_of_older_body_frame_velocity():
  guard = HondaRadarAccelGuard()
  confirm(guard)
  values = guard_input(3, gap=True)
  fusion = values['observation']['fusion']
  source_ns = values['model_ns'] - 50_000_000
  fusion.update(modelMonoTime=source_ns, modelTimestampEof=source_ns)
  fusion['tracks'][0].update(positionMonoTime=source_ns, velocityMonoTime=source_ns, lateralSpeed=-1.0)
  values['model']['orientationRate']['z'] = [1.0]
  assert fusion['tracks'][0]['groundSpeed'] == values['speed']
  assert guard.update(**values) is None
  assert guard.qualified and not guard.holding and guard.reason == 'radarPullingAway'
  assert guard.projected_ground_speed > values['speed']


@pytest.mark.parametrize('value', [None, 0, 1, '', 'false'])
def test_invalid_formal_lead_input_cannot_mean_no_lead(value):
  guard = HondaRadarAccelGuard()
  confirm(guard)
  guard.update(**guard_input(3, gap=True))
  values = guard_input(4, gap=True)
  values['formal_lead_present'] = value
  assert guard.update(**values) is None
  assert not guard.qualified and not guard.holding and guard.accel_ceiling is None
  assert guard.reason == 'formalLeadInvalid'


def test_formal_lead_presence_is_a_required_caller_input():
  values = guard_input(0)
  del values['formal_lead_present']
  with pytest.raises(TypeError, match='formal_lead_present'):
    HondaRadarAccelGuard().update(**values)


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


LIFECYCLE_AND_CORE_FAULTS = {
  'observer_closed': lambda v: v['observation'].update(active=False),
  'fusion_disabled': lambda v: v['observation']['fusion'].update(enabled=False),
  'long_inactive': lambda v: v['observation']['scope'].update(longActive=False),
  'dec_inactive': lambda v: v['observation']['scope'].update(decActive=False),
  'alpha_disabled': lambda v: v['observation']['scope'].update(alphaLong=False),
  'stale_core_model': lambda v: v.update(model_ns=START),
  'invalid_core_vision': lambda v: v['model']['leadsV3'][0].update(prob=float('nan')),
  'missing_core_vision': lambda v: v['model'].update(leadsV3=[]),
}


@pytest.mark.parametrize('fault', LIFECYCLE_AND_CORE_FAULTS)
def test_lifecycle_closure_or_core_fault_clears_instead_of_sensor_release(fault):
  guard = HondaRadarAccelGuard()
  confirm(guard)
  guard.update(**guard_input(3, gap=True))
  values = guard_input(4, gap=True)
  LIFECYCLE_AND_CORE_FAULTS[fault](values)
  assert guard.update(**values) is None
  assert not guard.qualified and not guard.holding and guard.accel_ceiling is None
  assert guard.track_key is None and not guard.pairs


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
