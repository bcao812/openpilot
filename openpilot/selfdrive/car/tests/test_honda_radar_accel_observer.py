"""Actual CAN observer/fusion output can qualify only the bounded adapter."""
from openpilot.selfdrive.car.honda_radar_observer import HondaRadarObserver
from openpilot.selfdrive.car.honda_radar_observer_config import ASSIST, ObserverSettings
from openpilot.selfdrive.car.tests.test_honda_radar_observer import START, can_message, configuration, scan_frames
from openpilot.selfdrive.car.tests.test_honda_radar_shadow import STEP, fusion_state
from openpilot.selfdrive.controls.lib.honda_radar_accel_guard import HondaRadarAccelGuard


def test_mode_three_uses_real_producer_without_granting_shadow_lead_authority():
  observer = HondaRadarObserver(*configuration(), session_id='actual-producer')
  guard = HondaRadarAccelGuard()
  settings = ObserverSettings(ASSIST, True, True, True)
  for frame in range(12):
    t = START + frame * STEP
    state = fusion_state(t)
    gap = frame >= 9
    if gap:
      for lead in state['modelV2'].leadsV3:
        lead.prob = 0.2
    # A slightly closing velocity avoids treating tiny filter roundoff around
    # exactly zero relative speed as a deliberately receding fixture target.
    message = observer.update([can_message(t, scan_frames(index=frame % 16, life=1 + frame * 2, velocity=863))],
                              state, settings, t + 1, state)
    body = message.hondaRadarObservations
    assert body.active and body.fusion.active
    assert not body.calibrated and not body.fusion.calibrated and not body.fusion.controlAllowed
    ceiling = guard.update(now_ns=t + 2, enabled=True, observation=body, observation_ns=message.logMonoTime,
                           observation_valid=message.valid, model=state['modelV2'], model_ns=t,
                           speed=10.0, standstill=False, ceiling_max=2.0, formal_lead_present=not gap)
    if frame == 8:
      assert guard.qualified and ceiling is None
    if gap:
      assert guard.qualified and guard.holding and ceiling == 0.0
