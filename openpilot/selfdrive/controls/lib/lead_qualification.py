"""Source-aware admission of formal leads, shared by planning consumers.

Version zero retains the caller's legacy probability gate. Version one adds
measurement provenance; a radar return never acquires visual confidence merely
because it is selected as leadOne. This module validates current evidence, not
the history needed to release a stop or establish physical identity.
"""
import math
from numbers import Integral, Real
from types import MappingProxyType

from openpilot.selfdrive.car.honda_radar_relevance import HORIZON


MAX_CONTROL_OBJECTS = 34  # Historical wire/display bound; control uses only the two formal lead slots.
MAX_MEASUREMENT_AGE_NS = 200_000_000
MAX_ESTIMATE_AGE_NS = 250_000_000
MAX_POSITION_STD = 20.0
MAX_VELOCITY_STD = 20.0
MIN_VISION_BRAKE_DISTANCE = -1.52  # Camera origin to the front bumper, as in radard.
PURPOSES = ('follow', 'brake', 'stop', 'departure', 'fcw')
SOURCES = ('vision', 'radar', 'fused')
_UNREAD = object()
_SCALAR_TYPES = (bool, int, float, str, type(None))


class _ControlLeadView:
  """Marker for an immutable consumer view of the two formal leads."""
  __slots__ = ()


class _StructSnapshot:
  """Read-only scalar fields from one publication, never cached authority."""
  __slots__ = ('_values', '_cache_safe', '_facts', '_identity', '_key')

  def __init__(self, values):
    frozen, cache_safe = {}, True
    for key, value in values.items():
      if isinstance(value, dict):
        value = _StructSnapshot(value)
      frozen[key] = value
      cache_safe = cache_safe and type(key) is str and (type(value) in _SCALAR_TYPES
                                                      or type(value) is _StructSnapshot and value._cache_safe)
    object.__setattr__(self, '_values', MappingProxyType(frozen))
    object.__setattr__(self, '_cache_safe', cache_safe)
    object.__setattr__(self, '_facts', None)
    object.__setattr__(self, '_identity', _UNREAD)
    object.__setattr__(self, '_key', _UNREAD)

  def __setattr__(self, field, value):
    raise AttributeError('publication snapshots are read-only')

  def __getattr__(self, field):
    try:
      return self._values[field]
    except KeyError:
      raise AttributeError(field) from None

  def get(self, field, default=None):
    return self._values.get(field, default)

  def as_reader(self):
    return self

  def to_dict(self):
    return {key: value.to_dict() if isinstance(value, _StructSnapshot) else value
            for key, value in self._values.items()}


def _as_reader(value):
  # Missing-field lookup on a native reader goes through Cap'n Proto's
  # dynamic schema machinery. Its class already tells us whether conversion
  # exists; keep instance-defined adapters available for ordinary fixtures.
  if type(value).__module__ == 'capnp.lib.capnp':
    reader = getattr(type(value), 'as_reader', None)
    return reader(value) if callable(reader) else value
  reader = getattr(value, 'as_reader', None)
  return reader() if callable(reader) else value


def _read(value, field, default=None):
  if isinstance(value, (dict, _StructSnapshot)):
    return value.get(field, default)
  if field == 'evidence':
    # Reading an absent struct through a Cap'n Proto builder allocates it.
    # Qualification is observational, including for legacy messages that
    # never set this pointer. A reader also keeps nested evidence reads pure.
    value = _as_reader(value)
  return getattr(value, field, default)


def _finite(value):
  value_type = type(value)
  if value_type is not float and value_type is not int and (not isinstance(value, Real) or isinstance(value, bool)):
    return False
  try:
    return math.isfinite(value)
  except (OverflowError, ValueError):
    return False


def _integer(value, minimum=0):
  if type(value) is int:
    return value >= minimum
  return isinstance(value, Integral) and not isinstance(value, bool) and value >= minimum


def lead_identity(lead):
  """Physical identity, independent of a radar slot's changing generation."""
  cached = type(lead) is _StructSnapshot and lead._cache_safe
  if cached and lead._identity is not _UNREAD:
    return lead._identity
  evidence = _read(lead, 'evidence')
  version = _read(evidence, 'version', 0)
  identity = None
  if _integer(version) and version == 1 and str(_read(evidence, 'source', 'legacy')) in ('radar', 'fused', 'predicted'):
    identity = _physical_identity(evidence)
  if cached:
    object.__setattr__(lead, '_identity', identity)
  return identity


def _physical_identity(evidence):
  session = _read(evidence, 'sessionId')
  track = _read(evidence, 'trackId')
  generation = _read(evidence, 'generation')
  if not isinstance(session, str) or not session or not _integer(track, 1) or not _integer(generation):
    return None
  return session, int(track)


def valid_conflict_support(evidence):
  """Validate retired window fields only when recognizing an old prediction."""
  version = _read(evidence, 'conflictSupportVersion', 0)
  if not _integer(version) or version not in (0, 1):
    return False
  if version == 0:
    return True
  end, supported_end = _read(evidence, 'conflictEnd'), _read(evidence, 'conflictSupportedEnd')
  if not (_finite(end) and _finite(supported_end) and 0.0 <= supported_end <= end):
    return False
  if _read(evidence, 'physicalConflictValid', False) is True:
    physical_start, physical_end = _read(evidence, 'physicalConflictStart'), _read(evidence, 'physicalConflictEnd')
    return (_finite(physical_start) and _finite(physical_end)
            and 0.0 <= physical_start < physical_end <= supported_end)
  return True


def _snapshot_facts(lead):
  """Decode common immutable facts, without granting any purpose or time."""
  get = lead._values.get
  probability = get('modelProb')
  if get('present', False) is not True or not _finite(probability) or not 0.0 <= probability <= 1.0:
    return False
  evidence = get('evidence')
  fields = evidence._values if type(evidence) is _StructSnapshot else {}
  read = fields.get
  version = read('version', 0)
  if not _integer(version):
    return False
  if version == 0:
    return (0, probability)
  source = str(read('source', 'legacy'))
  if version != 1 or source not in SOURCES:
    return False
  if source == 'vision':
    index = read('visionIndex')
    if not _integer(index) or index not in (0, 1):
      return False
  elif lead_identity(lead) is None:
    return False
  if read('coasting', False) is not False:
    return False
  estimate, position, velocity, model = (read(field) for field in (
    'estimateMonoTime', 'positionMonoTime', 'velocityMonoTime', 'modelMonoTime'))
  if not all(_integer(stamp, 1) for stamp in (estimate, position, velocity, model)):
    return False
  age = MAX_MEASUREMENT_AGE_NS
  if abs(estimate - position) > age or abs(estimate - velocity) > age:
    return False
  earliest = max(estimate, position, velocity, model)
  latest = min(estimate + MAX_ESTIMATE_AGE_NS, position + age, velocity + age, model + age)
  if read('positionValid', False) is not True or read('velocityValid', False) is not True:
    return False
  p_std, v_std = read('positionStd'), read('velocityStd')
  if not _finite(p_std) or not _finite(v_std) or not 0.0 < p_std <= MAX_POSITION_STD or not 0.0 < v_std <= MAX_VELOCITY_STD:
    return False
  distance, lateral, relative, ground, acceleration = (get(field) for field in ('dRel', 'yRel', 'vRel', 'vLead', 'aLeadK'))
  if not all(_finite(value) for value in (distance, lateral, relative, ground, acceleration)):
    return False
  a_valid = read('accelerationValid', False)
  if not isinstance(a_valid, bool) or not a_valid and acceleration != 0.0:
    return False
  offset = read('pathOffset')
  path_source = read('pathSource', '')
  if not _finite(offset) or not isinstance(path_source, str) or not path_source:
    return False
  flags = tuple(read(purpose + 'Qualified', False) for purpose in PURPOSES)
  return (1, probability, source, flags, earliest, latest,
          distance, relative, ground, str(read('pathState', 'unknown')))


def _qualified_snapshot(lead, purpose, min_model_prob, now_ns, require_active_window):
  if purpose not in PURPOSES or not isinstance(require_active_window, bool):
    return False
  if not _finite(min_model_prob) or not 0.0 <= min_model_prob <= 1.0:
    return False
  facts = lead._facts
  if facts is None:
    facts = _snapshot_facts(lead)
    object.__setattr__(lead, '_facts', facts)
  if facts is False:
    return False
  if facts[0] == 0:
    return facts[1] >= min_model_prob
  _, probability, source, flags, earliest, latest, distance, relative, ground, path_state = facts
  if flags[PURPOSES.index(purpose)] is not True:
    return False
  if source == 'vision' and probability < min_model_prob:
    return False
  reference = earliest if now_ns is None else now_ns
  if not earliest <= reference <= latest:
    return False
  if distance <= 0.0 and not (source == 'vision' and purpose in ('brake', 'fcw') and MIN_VISION_BRAKE_DISTANCE <= distance):
    return False
  path_states = ('inPath', 'entering') if purpose in ('brake', 'fcw') else ('inPath',)
  if path_state not in path_states:
    return False
  return purpose != 'departure' or ground > 0.0 and relative > 0.0


def _qualified_lead_evidence(lead, purpose, min_model_prob=0.95, *, now_ns=None, require_active_window=True):
  """Check a purpose without granting another purpose's authority.

  ``now_ns`` is the caller's monotonic event time, never wall time. Omitting it
  validates measurement ages at the latest recorded source time; callers
  using this form must separately enforce freshness of their radarState input.
  Model publication and a new CAN sample can follow camera EOF; the producer
  may project that measurement back to the common estimate time.
  Only current radar, fused or visual measurements grant a control purpose.
  Historical predicted/coasting records remain readable but grant none.
  ``require_active_window`` remains a compatibility argument; retired occupancy
  windows do not grant or expire an ordinary lead's measurement authority.
  """
  if type(lead) is _StructSnapshot and lead._cache_safe and (now_ns is None or type(now_ns) is int):
    return _qualified_snapshot(lead, purpose, min_model_prob, now_ns, require_active_window)
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
  elif _physical_identity(evidence) is None:
    return False

  if _read(evidence, 'coasting', False) is not False:
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
  maximum_age = MAX_MEASUREMENT_AGE_NS
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
  path_states = ('inPath', 'entering') if purpose in ('brake', 'fcw') else ('inPath',)
  if path_state not in path_states:
    return False
  path_offset = _read(evidence, 'pathOffset')
  path_source = _read(evidence, 'pathSource', '')
  if not _finite(path_offset) or not isinstance(path_source, str) or not path_source:
    return False
  if purpose == 'departure' and (values[3] <= 0.0 or values[2] <= 0.0):
    return False
  return True


def _future_entry(lead):
  evidence = _read(lead, 'evidence')
  version = _read(evidence, 'version', 0)
  if (not _integer(version) or version != 1
      or str(_read(evidence, 'source', 'legacy')) not in ('radar', 'fused', 'predicted')):
    return False
  start = _read(evidence, 'conflictStart')
  return str(_read(evidence, 'pathState')) == 'entering' or _finite(start) and start > 0.0


def retired_future_entry(lead, *, now_ns=None):
  """Ignore a well-formed old prediction of an object entering the path.

  This is a policy retirement, not an invalid sensor input. Keep malformed
  records visible to existing failure handling, including projection fields
  that the ordinary purpose check does not use. Without a caller clock,
  evidence is validated at its own latest source time.
  """
  if _read(lead, 'present', False) is not True or not _future_entry(lead):
    return False
  evidence = _read(lead, 'evidence')
  # This recognizes only the former braking-prediction format. A mixed or
  # malformed formal lead remains visible to ordinary input-failure handling.
  if (_read(evidence, 'cautionQualified', False) is not False
      or not _qualified_lead_evidence(lead, 'brake', 0.0, now_ns=now_ns)):
    return False
  decay = _read(lead, 'aLeadTau')
  start, end, lateral_speed = (_read(evidence, name) for name in ('conflictStart', 'conflictEnd', 'lateralSpeed'))
  physical = _read(evidence, 'physicalConflictValid', False)
  physical_start, physical_end = (_read(evidence, name, 0.0) for name in ('physicalConflictStart', 'physicalConflictEnd'))
  return (_finite(decay) and decay >= 0.0 and all(_finite(value) for value in (start, end, lateral_speed))
          and 0.0 <= start < end <= HORIZON and valid_conflict_support(evidence) and isinstance(physical, bool)
          and _finite(physical_start) and _finite(physical_end)
          and (not physical or 0.0 <= physical_start < physical_end <= end))


def qualified_lead(lead, purpose, min_model_prob=0.95, *, now_ns=None, require_active_window=True):
  """Admit current-path evidence for a purpose without predicting a new entry."""
  return (_qualified_lead_evidence(lead, purpose, min_model_prob, now_ns=now_ns,
                                   require_active_window=require_active_window)
          and not _future_entry(lead))


def measurement_expired_since(lead, *, qualified_ns, now_ns):
  """Recognize a physical object's measurement crossing its existing limit.

  Both clocks must describe actual processing boundaries. The producer uses
  its earlier qualification and the new Event's timestamp; a consumer uses
  the actual publication and consumption timestamps. A source already old at
  the earlier boundary, or with any other invalid evidence, is not natural
  expiry. No source timestamp or evidence lifetime is changed here.
  """
  if not _integer(qualified_ns, 1) or not _integer(now_ns, 1) or qualified_ns > now_ns:
    return False
  evidence = _read(lead, 'evidence')
  if str(_read(evidence, 'source')) not in ('radar', 'fused'):
    return False
  if not _qualified_lead_evidence(lead, 'brake', 0.0, now_ns=qualified_ns):
    return False
  maximum_age = MAX_MEASUREMENT_AGE_NS
  if (not 0 <= now_ns - _read(evidence, 'estimateMonoTime') <= MAX_ESTIMATE_AGE_NS
      or not 0 <= now_ns - _read(evidence, 'modelMonoTime') <= maximum_age):
    return False
  return any(now_ns - _read(evidence, field) > maximum_age
             for field in ('positionMonoTime', 'velocityMonoTime'))


def _object_key(lead):
  cached = type(lead) is _StructSnapshot and lead._cache_safe
  if cached and lead._key is not _UNREAD:
    return lead._key
  identity = lead_identity(lead)
  if identity is not None:
    key = ('track', *identity)
  else:
    evidence = _read(lead, 'evidence')
    key = (('vision', _read(evidence, 'modelMonoTime'), _read(evidence, 'visionIndex'))
           if _read(evidence, 'version', 0) == 1 and str(_read(evidence, 'source', 'legacy')) == 'vision' else None)
  if cached:
    object.__setattr__(lead, '_key', key)
  return key


def control_leads(radar_state, purpose=None, min_model_prob=0.95, *, now_ns=None):
  """Return the two ordinary lead slots, deduplicated by explicit identity.

  Raw radarObjects and old cached object collections are observational data;
  neither can add a braking, stop, departure or FCW candidate.
  """
  candidates = (_read(radar_state, 'leadOne'), _read(radar_state, 'leadTwo'))
  result, seen = [], set()
  for lead in candidates:
    if _read(lead, 'present', False) is not True:
      continue
    if retired_future_entry(lead, now_ns=now_ns):
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
