"""Source-aware admission of formal leads, shared by planning consumers.

Version zero retains the caller's legacy probability gate. Version one adds
measurement provenance; a radar return never acquires visual confidence merely
because it is selected as leadOne. This module validates current evidence, not
the history needed to release a stop or establish physical identity.
"""
import math
from numbers import Integral, Real


MAX_CONTROL_OBJECTS = 34  # 32 independent radar tracks and two visual leads.
MAX_MEASUREMENT_AGE_NS = 200_000_000
MAX_ESTIMATE_AGE_NS = 250_000_000
MAX_PREDICTION_AGE_NS = 350_000_000
MAX_POSITION_STD = 20.0
MAX_VELOCITY_STD = 20.0
MIN_VISION_BRAKE_DISTANCE = -1.52  # Camera origin to the front bumper, as in radard.
PURPOSES = ('follow', 'brake', 'stop', 'departure', 'fcw')
SOURCES = ('vision', 'radar', 'fused', 'predicted')


def _read(value, field, default=None):
  return value.get(field, default) if isinstance(value, dict) else getattr(value, field, default)


def _finite(value):
  if not isinstance(value, Real) or isinstance(value, bool):
    return False
  try:
    return math.isfinite(value)
  except (OverflowError, ValueError):
    return False


def _integer(value, minimum=0):
  return isinstance(value, Integral) and not isinstance(value, bool) and value >= minimum


def lead_identity(lead):
  """Physical identity, independent of a radar slot's changing generation."""
  evidence = _read(lead, 'evidence')
  version = _read(evidence, 'version', 0)
  if (not _integer(version) or version != 1 or str(_read(evidence, 'source', 'legacy')) not in ('radar', 'fused', 'predicted')):
    return None
  session = _read(evidence, 'sessionId')
  track = _read(evidence, 'trackId')
  generation = _read(evidence, 'generation')
  if not isinstance(session, str) or not session or not _integer(track, 1) or not _integer(generation):
    return None
  return session, int(track)


def qualified_lead(lead, purpose, min_model_prob=0.95, *, now_ns=None, require_active_window=True):
  """Check a purpose without granting another purpose's authority.

  ``now_ns`` is the caller's monotonic event time, never wall time. Omitting it
  validates measurement ages at the latest recorded source time; callers
  using this form must separately enforce freshness of their radarState input.
  Model publication and a new CAN sample can follow camera EOF; the producer
  may project that measurement back to the common estimate time.
  A predicted object can retain a short braking constraint, but cannot confirm
  a fresh followed car, FCW, parking geometry or departure.
  A planner projecting occupancy itself may set ``require_active_window=False``
  to distinguish an expired occupancy window from invalid measurement evidence.
  """
  if purpose not in PURPOSES or _read(lead, 'present', False) is not True or not isinstance(require_active_window, bool):
    return False
  probability = _read(lead, 'modelProb')
  if (not _finite(min_model_prob) or not 0.0 <= min_model_prob <= 1.0
      or not _finite(probability) or not 0.0 <= probability <= 1.0):
    return False
  evidence = _read(lead, 'evidence')
  version = _read(evidence, 'version', 0)
  if not _integer(version):
    return False
  if version == 0:
    return probability >= min_model_prob
  if version != 1:
    return False
  source = str(_read(evidence, 'source', 'legacy'))
  if source not in SOURCES or _read(evidence, purpose + 'Qualified', False) is not True:
    return False
  if source == 'vision':
    vision_index = _read(evidence, 'visionIndex')
    if probability < min_model_prob or not _integer(vision_index) or vision_index not in (0, 1):
      return False
  elif lead_identity(lead) is None:
    return False

  coasting = _read(evidence, 'coasting', False)
  if not isinstance(coasting, bool) or (source == 'predicted' and not coasting):
    return False
  if coasting and purpose != 'brake':
    return False
  estimate = _read(evidence, 'estimateMonoTime')
  position = _read(evidence, 'positionMonoTime')
  velocity = _read(evidence, 'velocityMonoTime')
  model = _read(evidence, 'modelMonoTime')
  if not all(_integer(stamp, 1) for stamp in (estimate, position, velocity, model)):
    return False
  reference = max(estimate, model, position, velocity) if now_ns is None else now_ns
  if not all(_integer(stamp, 1) for stamp in (estimate, reference, position, velocity, model)):
    return False
  maximum_age = MAX_PREDICTION_AGE_NS if coasting else MAX_MEASUREMENT_AGE_NS
  if (not 0 <= reference - estimate <= MAX_ESTIMATE_AGE_NS
      or any(not 0 <= reference - stamp <= maximum_age or abs(estimate - stamp) > maximum_age for stamp in (position, velocity))
      or not 0 <= reference - model <= maximum_age):
    return False
  if _read(evidence, 'positionValid', False) is not True or _read(evidence, 'velocityValid', False) is not True:
    return False
  position_std, velocity_std = (_read(evidence, name) for name in ('positionStd', 'velocityStd'))
  if (not all(_finite(value) for value in (position_std, velocity_std))
      or not 0.0 < position_std <= MAX_POSITION_STD or not 0.0 < velocity_std <= MAX_VELOCITY_STD):
    return False
  values = tuple(_read(lead, field) for field in ('dRel', 'yRel', 'vRel', 'vLead', 'aLeadK'))
  if not all(_finite(value) for value in values):
    return False
  close_vision_risk = (source == 'vision' and purpose in ('brake', 'fcw')
                       and MIN_VISION_BRAKE_DISTANCE <= values[0] <= 0.0)
  if values[0] <= 0.0 and not close_vision_risk:
    return False
  # Invalid acceleration is an explicit constant-velocity prior, never a
  # synthetic acceleration measurement. Consumers needing measured a must
  # additionally require accelerationValid.
  acceleration_valid = _read(evidence, 'accelerationValid', False)
  if not isinstance(acceleration_valid, bool) or (not acceleration_valid and values[4] != 0.0):
    return False
  path_state = str(_read(evidence, 'pathState', 'unknown'))
  if path_state not in (('inPath', 'entering') if purpose in ('brake', 'fcw') else ('inPath',)):
    return False
  path_values = tuple(_read(evidence, field) for field in ('pathOffset', 'conflictStart', 'conflictEnd', 'lateralSpeed'))
  path_source = _read(evidence, 'pathSource', '')
  if (not all(_finite(value) for value in path_values) or not isinstance(path_source, str) or not path_source
      or not 0.0 <= path_values[1] < path_values[2]):
    return False
  # An entering object is useful only during its predicted occupancy window.
  if require_active_window and (reference - estimate) / 1e9 >= path_values[2]:
    return False
  if purpose == 'departure' and (values[3] <= 0.0 or values[2] <= 0.0):
    return False
  return True


def _object_key(lead):
  identity = lead_identity(lead)
  if identity is not None:
    return 'track', *identity
  evidence = _read(lead, 'evidence')
  if _read(evidence, 'version', 0) == 1 and str(_read(evidence, 'source', 'legacy')) == 'vision':
    return 'vision', _read(evidence, 'modelMonoTime'), _read(evidence, 'visionIndex')
  return None


def control_leads(radar_state, purpose=None, min_model_prob=0.95, *, now_ns=None):
  """Return independent present objects, optionally admitted for one purpose.

  Display slots are retained as a compatibility fallback and deduplicated only
  by explicit physical/visual identity. Unknown identities are not merged by
  position. The producer owns the bounded list; malformed oversize lists remain
  visible to a consumer's explicit input-failure handling, never silently cut.
  """
  display = (_read(radar_state, 'leadOne'), _read(radar_state, 'leadTwo'))
  candidates = (*tuple(_read(radar_state, 'radarObjects', ())), *display) if _read(radar_state, 'fusionActive', False) is True else display
  result, seen = [], set()
  for lead in candidates:
    if _read(lead, 'present', False) is not True:
      continue
    if purpose is not None and not qualified_lead(lead, purpose, min_model_prob, now_ns=now_ns):
      continue
    key = _object_key(lead)
    if key is not None and key in seen:
      continue
    if key is not None:
      seen.add(key)
    result.append(lead)
  return tuple(result)
