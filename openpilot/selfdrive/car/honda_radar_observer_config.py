"""Small, side-effect-free configuration contract for the optional observer."""
from dataclasses import dataclass

PROCESS_NAME = 'hondaRadarObserver'
SERVICE = 'hondaRadarObservations'
OBSERVE = 1
SHADOW = 2
ASSIST = 3
CONTROL = 4
FUSION_MODES = (SHADOW, ASSIST, CONTROL)
SCOPE_MAX_AGE_NS = 500_000_000


@dataclass(frozen=True)
class ObserverSettings:
  mode: int = 0
  alpha: bool = False
  experimental: bool = False
  dec: bool = False

  @classmethod
  def read(cls, params):
    try:
      mode = params.get('HondaBoschARadarMode', return_default=True)
      return cls(mode if type(mode) is int else 0, params.get_bool('AlphaLongitudinalEnabled'),
                 params.get_bool('ExperimentalMode'), params.get_bool('DynamicExperimentalControl'))
    except Exception:
      # Missing keys/backend errors disable only this optional observer.
      return cls()

  def requested(self, cp):
    return (self.mode in (OBSERVE, *FUSION_MODES) and self.alpha and self.experimental and self.dec
            and cp.carFingerprint == 'HONDA_ACCORD' and cp.openpilotLongitudinalControl and not cp.notCar)

  def assist_requested(self, cp, cp_sp):
    # The Accord fingerprint names the supported Bosch platform. This never
    # changes its radarUnavailable capability or the existing radarState path.
    return self.mode == ASSIST and self.requested(cp) and not cp_sp.enableGasInterceptor

  def control_requested(self, cp, cp_sp):
    return self.mode == CONTROL and self.requested(cp) and not cp_sp.enableGasInterceptor


def observer_requested(started, params, cp):
  return started and ObserverSettings.read(params).requested(cp)


def scope_state(cp, cp_sp, settings, sm, now_ns):
  dec = sm['longitudinalPlanSP'].dec
  scope = {'alphaLong': bool(settings.alpha and cp.openpilotLongitudinalControl), 'experimentalSetting': settings.experimental,
           'decSetting': settings.dec, 'experimentalActive': bool(sm['selfdriveState'].experimentalMode),
           'decEnabled': bool(dec.enabled), 'decActive': bool(dec.active), 'decMode': int(dec.state.raw),
           'selfdriveMonoTime': sm.logMonoTime['selfdriveState'], 'planMonoTime': sm.logMonoTime['longitudinalPlanSP'],
           'longActive': bool(sm['carControl'].longActive)}
  if not settings.requested(cp) or cp_sp.enableGasInterceptor:
    return False, 'configuration', scope
  if settings.control_requested(cp, cp_sp):
    # Perception must be able to start before DEC has consumed its first lead.
    # Actual execution scope is checked independently by radard and the planner.
    return True, 'observing', scope
  for name in ('selfdriveState', 'longitudinalPlanSP'):
    if not sm.seen[name] or not sm.valid[name] or not 0 <= now_ns - sm.logMonoTime[name] <= SCOPE_MAX_AGE_NS:
      return False, 'scopeStale', scope
  if not scope['experimentalActive'] or not scope['decEnabled'] or not scope['decActive']:
    return False, 'scopeInactive', scope
  return True, 'observing', scope
