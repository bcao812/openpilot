"""Admission gates for the fixed Accord radar fusion pipeline."""
from dataclasses import dataclass
from openpilot.common.radar_signal import radar_signal_enabled

PROCESS_NAME = 'hondaRadarObserver'
SERVICE = 'hondaRadarObservations'
# Historical wire modes retain their schema numbering. The current producer
# only implements full fusion; stored HondaBoschARadarMode values are retired.
CONTROL = 4
SCOPE_MAX_AGE_NS = 500_000_000


@dataclass(frozen=True)
class ObserverSettings:
  alpha: bool = False
  experimental: bool = False
  dec: bool = False
  available: bool = True
  radar_signal: bool = True

  @classmethod
  def read(cls, params):
    try:
      return cls(params.get_bool('AlphaLongitudinalEnabled'), params.get_bool('ExperimentalMode'),
                 params.get_bool('DynamicExperimentalControl'), radar_signal=radar_signal_enabled(params))
    except Exception:
      # Missing keys/backend errors disable only this optional observer.
      return cls(available=False)

  def requested(self, cp):
    return (self.available and self.radar_signal and self.alpha and self.experimental and self.dec
            and cp.carFingerprint == 'HONDA_ACCORD' and cp.openpilotLongitudinalControl and not cp.notCar)

  def control_requested(self, cp, cp_sp):
    return self.requested(cp) and not cp_sp.enableGasInterceptor


def observer_requested(started, params, cp):
  return started and ObserverSettings.read(params).requested(cp)


def scope_state(cp, cp_sp, settings, sm):
  dec = sm['longitudinalPlanSP'].dec
  scope = {'alphaLong': bool(settings.alpha and cp.openpilotLongitudinalControl), 'experimentalSetting': settings.experimental,
           'decSetting': settings.dec, 'experimentalActive': bool(sm['selfdriveState'].experimentalMode),
           'decEnabled': bool(dec.enabled), 'decActive': bool(dec.active), 'decMode': int(dec.state.raw),
           'selfdriveMonoTime': sm.logMonoTime['selfdriveState'], 'planMonoTime': sm.logMonoTime['longitudinalPlanSP'],
           'longActive': bool(sm['carControl'].longActive)}
  if not settings.control_requested(cp, cp_sp):
    return False, 'configuration', scope
  # Perception must start before DEC has consumed its first lead. Execution
  # scope and feedback freshness are checked by radard and the planner.
  return True, 'observing', scope
