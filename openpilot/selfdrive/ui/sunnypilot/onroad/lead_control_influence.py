"""Color only targets acknowledged by the longitudinal execution owner.

The color scale is qualitative control effort, not detection confidence,
collision probability, or a counterfactual fraction of the vehicle's motion.
The UI neither grants target authority nor estimates control influence.
"""
from dataclasses import dataclass
import math
from numbers import Integral, Real


MESSAGE_MAX_AGE_NS = 200_000_000
MAX_INFLUENCES = 34
UNIFIED_MODE = 2
KINDS = ('follow', 'braking', 'hold', 'release', 'caution')


def _read(value, field, default=None):
  if isinstance(value, dict):
    return value.get(field, default)
  # A Cap'n Proto reader treats unknown instance attributes as schema fields.
  # Check the type to avoid a failing schema lookup for every field we read.
  as_reader = getattr(type(value), 'as_reader', None)
  if callable(as_reader):
    value = as_reader(value)
  return getattr(value, field, default)


def _integer(value, minimum=0):
  return isinstance(value, Integral) and not isinstance(value, bool) and value >= minimum


def _fresh(stamp, now_ns):
  return _integer(stamp, 1) and 0 <= now_ns - stamp <= MESSAGE_MAX_AGE_NS


@dataclass(frozen=True)
class ObjectInfluence:
  session_id: str
  track_id: int
  model_ns: int
  vision_index: int
  kind: str
  strength: float


def accepted_influences(sm, now_ns):
  """Read the owner's current receipt, never the latest unconsumed plan.

  The most recent planner message may already describe a different vehicle.
  Driver override and message expiry are checked on every draw, including
  frames that reuse cached detection geometry.
  """
  if not _integer(now_ns, 1):
    return ()
  valid, stamps = getattr(sm, 'valid', {}), getattr(sm, 'logMonoTime', {})
  services = ('controlsState', 'carControl', 'carState', 'selfdriveState')
  if any(valid.get(name) is not True or not _fresh(stamps.get(name), now_ns) for name in services):
    return ()
  controls, command, car, selfdrive = (sm[name] for name in services)
  motion = _read(controls, 'motionState')
  plan_ns = _read(motion, 'planMonoTime', 0)
  phase = str(_read(motion, 'phase', 'off'))
  if (_read(command, 'longActive', False) is not True or _read(command, 'enabled', False) is not True
      or _read(selfdrive, 'enabled', False) is not True
      or _read(car, 'gasPressed', False) or _read(car, 'brakePressed', False)
      or _read(motion, 'enabled', False) is not True or _read(motion, 'planningMode', -1) != UNIFIED_MODE
      or phase not in ('track', 'approach', 'hold', 'release')
      or not _fresh(plan_ns, now_ns) or plan_ns > stamps['controlsState']
      or _read(controls, 'longitudinalPlanMonoTime', 0) != plan_ns):
    return ()
  items = _read(motion, 'objectInfluences', ())
  if len(items) > MAX_INFLUENCES:
    return ()
  ordinary_allowed = not (_read(motion, 'degraded', False) or _read(motion, 'waitingForFreshInput', False))
  accepted = []
  for item in items:
    version = _read(item, 'version', 0)
    session, track = _read(item, 'sessionId', ''), _read(item, 'trackId', 0)
    model_ns, vision_index = _read(item, 'modelMonoTime', 0), _read(item, 'visionIndex', -1)
    kind, strength = str(_read(item, 'kind', '')), _read(item, 'strength', 0.0)
    if (not _integer(version) or version not in (1, 2) or kind not in KINDS
        or not isinstance(strength, Real) or isinstance(strength, bool) or not math.isfinite(strength)
        or not 0.0 < strength <= 1.0 or not isinstance(session, str) or len(session) > 64
        or kind == 'caution' and not math.isclose(strength, 0.15, rel_tol=0.0, abs_tol=1e-6)
        or not _integer(track) or not _integer(model_ns) or not _integer(vision_index, -1)):
      continue
    radar_identity = bool(session) and track > 0
    visual_identity = not session and track == 0 and vision_index in (0, 1) and model_ns > 0
    if version == 1 and not ordinary_allowed:
      continue
    # Version 2 is the owner's explicit receipt for adopted physical-risk
    # mitigation during a constrained/failed plan. Generic fallback braking
    # and version-1 candidates cannot acquire this exception in the UI.
    if version == 2 and not (radar_identity and kind == 'braking' and strength == 1.0
                             and phase in ('track', 'approach', 'hold')):
      continue
    if radar_identity or visual_identity:
      accepted.append(ObjectInfluence(session, int(track), int(model_ns), int(vision_index), kind, float(strength)))
  return tuple(accepted)


def radar_influence_strength(point, influences):
  if not point.session_id or point.track_id <= 0:
    return 0.0
  return max((item.strength for item in influences
              if (item.session_id, item.track_id) == (point.session_id, point.track_id)), default=0.0)


def lead_influence_strength(lead, influences, *, legacy_model_ns=0, legacy_slot=None):
  if _read(lead, 'present', False) is not True:
    return 0.0
  evidence = _read(lead, 'evidence')
  version = _read(evidence, 'version', 0)
  if not _integer(version):
    return 0.0
  if version == 0:
    # Current UNIFIED execution can acknowledge an old-layout pure visual
    # lead. Only its explicit radarState model frame and actual display slot
    # identify it; neither proximity nor the slot on a newer frame is enough.
    if (_read(lead, 'radar') is not False or not _integer(legacy_model_ns, 1)
        or not _integer(legacy_slot) or legacy_slot not in (0, 1)):
      return 0.0
    return max((item.strength for item in influences
                if not item.session_id and item.track_id == 0
                and (item.model_ns, item.vision_index) == (legacy_model_ns, legacy_slot)), default=0.0)
  if version != 1:
    return 0.0
  source = str(_read(evidence, 'source', 'legacy'))
  session, track = _read(evidence, 'sessionId', ''), _read(evidence, 'trackId', 0)
  if source in ('radar', 'fused', 'predicted') and isinstance(session, str) and session and _integer(track, 1):
    return max((item.strength for item in influences
                if (item.session_id, item.track_id) == (session, track)), default=0.0)
  model_ns, vision_index = _read(evidence, 'modelMonoTime', 0), _read(evidence, 'visionIndex', -1)
  if source == 'vision' and session == '' and track == 0 and _integer(model_ns, 1) and _integer(vision_index) and vision_index in (0, 1):
    return max((item.strength for item in influences
                if (item.model_ns, item.vision_index) == (model_ns, vision_index)), default=0.0)
  return 0.0


def influence_rgb(strength):
  """A fixed light-red to deep-red scale; zero leaves the original color."""
  if not isinstance(strength, Real) or isinstance(strength, bool) or not math.isfinite(strength) or not 0.0 < strength <= 1.0:
    return None
  return tuple(round(low + (high - low) * strength) for low, high in zip((255, 155, 155), (195, 20, 35), strict=True))
