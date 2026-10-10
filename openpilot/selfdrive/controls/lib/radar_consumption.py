"""One immutable formal-lead view for a radarState planner tick.

A measured object or coast may cross its deadline before or after publication.
Remove irrelevant expired formal leads without carrying missing-object risks
into later messages. Relevant expiry and malformed formal input retain the
existing failure braking path. Extra radar objects never acquire control roles.
Well-formed future-entry predictions are retired before assigning control roles.
"""
from openpilot.selfdrive.car.honda_radar_relevance import loss_relevant
from openpilot.selfdrive.controls.lib.lead_qualification import (
  MAX_ESTIMATE_AGE_NS, MAX_MEASUREMENT_AGE_NS, MAX_PREDICTION_AGE_NS,
  _ControlLeadView, _StructSnapshot, _as_reader, _finite, _future_entry, _integer, _object_key, _qualified_lead_evidence, _read,
  lead_identity, retired_future_entry,
)


_LEAD_FIELDS = ('dRel', 'yRel', 'vRel', 'vLead', 'vLeadK', 'aLeadK', 'aLeadTau', 'modelProb', 'radar', 'radarTrackId')
_EVIDENCE_FIELDS = (
  'version', 'sessionId', 'trackId', 'generation', 'estimateMonoTime', 'positionMonoTime', 'velocityMonoTime',
  'modelMonoTime', 'positionStd', 'velocityStd', 'positionValid', 'velocityValid', 'accelerationValid',
  'followQualified', 'brakeQualified', 'stopQualified', 'departureQualified', 'fcwQualified', 'coasting',
  'pathOffset', 'pathSource', 'visionIndex', 'reason', 'followGeometryMonoTime',
)


class _AbsentLead:
  __slots__ = ('_lead',)
  present = False

  def __init__(self, lead):
    object.__setattr__(self, '_lead', lead)

  def __setattr__(self, field, value):
    raise AttributeError('consumption views are read-only')

  def __getattr__(self, field):
    return getattr(self._lead, field)

  def as_reader(self):
    return self

  def to_dict(self):
    return {**self._lead.to_dict(), 'present': False}


class RadarConsumptionView(_ControlLeadView):
  __slots__ = ('_radar', 'fusionActive', 'fusionRiskUnresolved', 'radarObjects', 'leadOne', 'leadTwo', '_control_objects')

  def __init__(self, radar, removed, *, snapshot=False):
    object.__setattr__(self, '_radar', radar)
    # This producer claim belongs to the same publication as the object view.
    # Keep it fixed even when a mutable fixture/reader is reused later.
    object.__setattr__(self, 'fusionActive', getattr(radar, 'fusionActive', False))
    # The legacy wire field no longer supplies persistent missing-object risk.
    object.__setattr__(self, 'fusionRiskUnresolved', False)
    def view(lead):
      return _StructSnapshot(lead.to_dict()) if snapshot else lead
    # The old list is not a control input, including when its records are
    # malformed, stale, or claim the identity of a formal lead.
    object.__setattr__(self, 'radarObjects', ())
    object.__setattr__(self, 'leadOne', _AbsentLead(radar.leadOne) if removed and _object_key(radar.leadOne) in removed else view(radar.leadOne))
    object.__setattr__(self, 'leadTwo', _AbsentLead(radar.leadTwo) if removed and _object_key(radar.leadTwo) in removed else view(radar.leadTwo))
    object.__setattr__(self, '_control_objects', None)

  def __setattr__(self, field, value):
    raise AttributeError('consumption views are read-only')

  def __getattr__(self, field):
    return getattr(self._radar, field)

  def as_reader(self):
    return self

  def to_dict(self):
    return {**self._radar.to_dict(), 'radarObjects': [lead.to_dict() for lead in self.radarObjects],
            'leadOne': self.leadOne.to_dict(), 'leadTwo': self.leadTwo.to_dict(),
            'fusionActive': self.fusionActive, 'fusionRiskUnresolved': self.fusionRiskUnresolved}


def _measurement_expired_at(lead, now_ns):
  """Classify age-only loss without admitting it at an earlier source clock.

  Source-time qualification checks the unmodified evidence's structure and
  original lifetime. Current estimate/model evidence must still be fresh;
  only the position or velocity measurement may have expired. The result is
  used solely for retirement/loss handling, never as a control qualification.
  """
  if not _integer(now_ns, 1) or not _qualified_lead_evidence(lead, 'brake', 0.0):
    return False
  evidence = _read(lead, 'evidence')
  if str(_read(evidence, 'source')) not in ('radar', 'fused', 'predicted'):
    return False
  maximum_age = MAX_PREDICTION_AGE_NS if _read(evidence, 'coasting') is True else MAX_MEASUREMENT_AGE_NS
  measurements = tuple(_read(evidence, field) for field in ('positionMonoTime', 'velocityMonoTime'))
  return (0 <= now_ns - _read(evidence, 'estimateMonoTime') <= MAX_ESTIMATE_AGE_NS
          and 0 <= now_ns - _read(evidence, 'modelMonoTime') <= maximum_age
          and all(stamp <= now_ns for stamp in measurements)
          and any(now_ns - stamp > maximum_age for stamp in measurements))


def _qualified_objects(radar, qualified_ns):
  """Validate both formal slots, including duplicate identities, before removal."""
  signatures, objects = {}, []
  for lead in (radar.leadOne, radar.leadTwo):
    scalar_snapshot = type(lead) is _StructSnapshot and lead._cache_safe
    if _read(lead, 'present') is not True:
      continue
    evidence = _read(lead, 'evidence')
    admitted = _qualified_lead_evidence(lead, 'brake', 0.0, now_ns=qualified_ns)
    if (not (admitted or _measurement_expired_at(lead, qualified_ns))
        or not _finite(_read(lead, 'aLeadTau')) or _read(lead, 'aLeadTau') < 0.0
        or _future_entry(lead) and not retired_future_entry(lead)):
      return None
    key = _object_key(lead)
    if key is None:
      return None
    if scalar_snapshot and type(evidence) is _StructSnapshot:
      # These are the same fields/defaults as _read, already captured once.
      signature = (tuple(map(lead._values.get, _LEAD_FIELDS)), tuple(map(evidence._values.get, _EVIDENCE_FIELDS)),
                   str(evidence.get('source')), str(evidence.get('pathState')))
    else:
      signature = (tuple(_read(lead, field) for field in _LEAD_FIELDS),
                   tuple(_read(evidence, field) for field in _EVIDENCE_FIELDS),
                   str(_read(evidence, 'source')), str(_read(evidence, 'pathState')))
    if key in signatures:
      if signatures[key] != signature:
        return None
    else:
      objects.append(lead)
    signatures[key] = signature
  return tuple(objects)


def _relevant_at_consumption(lead, now_ns, speed):
  evidence = _read(lead, 'evidence')
  elapsed = (now_ns - _read(evidence, 'estimateMonoTime')) / 1e9
  # This bounded constant-velocity alignment only classifies current expiry;
  # it never refreshes the expired hypothesis or supplies a new MPC input.
  ground_speed = _read(lead, 'vLead')
  distance = _read(lead, 'dRel') + (ground_speed - speed) * elapsed
  return loss_relevant(distance=distance, ego_speed=speed, path_state=str(_read(evidence, 'pathState')))


class RadarConsumption:
  def __init__(self):
    self.clear()

  def clear(self):
    """Reset this tick's expiry result and optional display reuse."""
    self.degraded = False
    self.expired_identities = ()
    self.retired_identities = ()
    self.clear_publication()

  def clear_publication(self):
    """Discard this tick's optional display reuse."""
    self.publication_view = None
    self.publication_ns = None

  def update(self, radar, *, publication_ns, now_ns, speed):
    # Actual publication/consumption clocks govern control qualification.
    # Age-only loss classification cannot renew expired control authority.
    return self._update(radar, qualified_ns=publication_ns, publication_ns=publication_ns,
                        now_ns=now_ns, speed=speed)

  def finalize_publication(self, radar, *, qualified_ns, publication_ns, speed):
    """Retire measurements that expired during this producer's computation.

    The caller records the actual qualification clock and creates its Event
    before this check. The supplied publication timestamp remains unchanged.
    Relevant expiry retains the invalid object for this frame's failure braking.
    """
    return self._update(radar, qualified_ns=qualified_ns, publication_ns=publication_ns,
                        now_ns=publication_ns, speed=speed)

  def _update(self, radar, *, qualified_ns, publication_ns, now_ns, speed):
    self.degraded = False
    self.expired_identities = ()
    self.retired_identities = ()
    self.clear_publication()
    radar = _as_reader(radar)
    if (getattr(radar, 'fusionActive', False) is not True or not _integer(qualified_ns, 1)
        or not _integer(publication_ns, 1) or not _integer(now_ns, 1)
        or not qualified_ns <= publication_ns <= now_ns or not _finite(speed) or speed < 0.0):
      return radar
    publication = radar
    if type(radar).__module__ == 'capnp.lib.capnp':
      # Copy the two formal slots separately. Equal identities must not
      # conceal differing evidence. Extra-object fields are never decoded.
      try:
        radar = RadarConsumptionView(radar, (), snapshot=True)
      except (RuntimeError, UnicodeError):
        # Eager conversion can decode fields that the original validation
        # short-circuits (for example an absent lead's invalid enum/text).
        # Preserve that validation order, including expiry of other objects.
        radar = publication
    objects = _qualified_objects(radar, qualified_ns)
    if objects is None:
      return publication
    cache_publication = type(radar) is RadarConsumptionView and all(type(lead) is _StructSnapshot and lead._cache_safe for lead in objects)

    removed = set()
    identities = []
    retired = []
    for lead in objects:
      measurement_expired = _measurement_expired_at(lead, now_ns)
      if (retired_future_entry(lead, now_ns=now_ns)
          or measurement_expired and retired_future_entry(lead)):
        removed.add(_object_key(lead))
        retired.append(lead_identity(lead))
        continue
      if not measurement_expired:
        continue
      identity = lead_identity(lead)
      relevant = _relevant_at_consumption(lead, now_ns, speed)
      # A degraded flag is diagnostic, not a braking command. The same relevant
      # obstacle must retain failure braking whether its measurement crossed
      # the deadline before or after publication, measured or predicted.
      if relevant:
        self.degraded = True
        continue
      removed.add(_object_key(lead))
      identities.append(identity)
    self.expired_identities = tuple(identities)
    self.retired_identities = tuple(retired)
    if cache_publication:
      self.publication_view, self.publication_ns = radar, publication_ns
    if not removed:
      return radar
    return RadarConsumptionView(radar, removed)
