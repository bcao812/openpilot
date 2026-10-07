"""One immutable object view for all consumers of a radarState planner tick.

A coast may cross its measurement deadline after a valid publication. Remove
that expired hypothesis from both the full collection and display slots, while
remembering a relevant loss independently of message and prediction lifetimes.
Malformed publications are left intact for the existing input-failure path.
"""
from dataclasses import dataclass

from openpilot.selfdrive.car.honda_radar_relevance import HORIZON, loss_relevant, remaining_occupancy_window
from openpilot.selfdrive.controls.lib.lead_qualification import (
  MAX_CONTROL_OBJECTS, MAX_MEASUREMENT_AGE_NS, _StructSnapshot, _as_reader, _finite, _integer, _object_key, _read,
  coast_measurement_expired, control_leads, lead_identity, qualified_lead,
)


_LEAD_FIELDS = ('dRel', 'yRel', 'vRel', 'vLead', 'vLeadK', 'aLeadK', 'aLeadTau', 'modelProb', 'radar', 'radarTrackId')
_EVIDENCE_FIELDS = (
  'version', 'sessionId', 'trackId', 'generation', 'estimateMonoTime', 'positionMonoTime', 'velocityMonoTime',
  'modelMonoTime', 'positionStd', 'velocityStd', 'positionValid', 'velocityValid', 'accelerationValid',
  'followQualified', 'brakeQualified', 'stopQualified', 'departureQualified', 'fcwQualified', 'cautionQualified', 'coasting',
  'pathOffset', 'conflictStart', 'conflictEnd', 'lateralSpeed', 'pathSource', 'visionIndex', 'reason',
  'physicalConflictValid', 'physicalConflictStart', 'physicalConflictEnd',
)


class _AbsentLead:
  present = False

  def __init__(self, lead):
    self._lead = lead

  def __getattr__(self, field):
    return getattr(self._lead, field)

  def as_reader(self):
    return self

  def to_dict(self):
    return {**self._lead.to_dict(), 'present': False}


class RadarConsumptionView:
  def __init__(self, radar, expired, *, snapshot=False):
    self._radar = radar
    # This producer claim belongs to the same publication as the object view.
    # Keep it fixed even when a mutable fixture/reader is reused later.
    self.fusionRiskUnresolved = bool(getattr(radar, 'fusionRiskUnresolved', False))
    def view(lead):
      return _StructSnapshot(lead.to_dict()) if snapshot else lead
    self.radarObjects = tuple(view(lead) for lead in radar.radarObjects if not expired or _object_key(lead) not in expired)
    self.leadOne = _AbsentLead(radar.leadOne) if expired and _object_key(radar.leadOne) in expired else view(radar.leadOne)
    self.leadTwo = _AbsentLead(radar.leadTwo) if expired and _object_key(radar.leadTwo) in expired else view(radar.leadTwo)

  def __getattr__(self, field):
    return getattr(self._radar, field)

  def as_reader(self):
    return self

  def to_dict(self):
    return {**self._radar.to_dict(), 'radarObjects': [lead.to_dict() for lead in self.radarObjects],
            'leadOne': self.leadOne.to_dict(), 'leadTwo': self.leadTwo.to_dict(),
            'fusionRiskUnresolved': self.fusionRiskUnresolved}


@dataclass(frozen=True)
class _Loss:
  publication_ns: int
  position_ns: int
  velocity_ns: int


def _published_objects(radar, publication_ns):
  """Validate every identity view before any duplicate or expiry is removed."""
  raw = tuple(radar.radarObjects)
  if len(raw) > MAX_CONTROL_OBJECTS:
    return None
  signatures = {}
  for lead in (*raw, radar.leadOne, radar.leadTwo):
    if _read(lead, 'present') is not True:
      continue
    evidence = _read(lead, 'evidence')
    admitted = (qualified_lead(lead, 'brake', 0.0, now_ns=publication_ns, require_active_window=False)
                or qualified_lead(lead, 'caution', 0.0, now_ns=publication_ns, require_active_window=False))
    if (not admitted
        or not _finite(_read(lead, 'aLeadTau')) or _read(lead, 'aLeadTau') < 0.0
        or _read(evidence, 'conflictEnd') > HORIZON):
      return None
    key = _object_key(lead)
    if key is None:
      return None
    signature = (tuple(_read(lead, field) for field in _LEAD_FIELDS),
                 tuple(_read(evidence, field) for field in _EVIDENCE_FIELDS),
                 str(_read(evidence, 'source')), str(_read(evidence, 'pathState')))
    if key in signatures and signatures[key] != signature:
      return None
    signatures[key] = signature
  objects = control_leads(radar)
  return objects if len(objects) <= MAX_CONTROL_OBJECTS else None


def _relevant_at_consumption(lead, now_ns, speed):
  evidence = _read(lead, 'evidence')
  elapsed = (now_ns - _read(evidence, 'estimateMonoTime')) / 1e9
  start, end = remaining_occupancy_window(_read(evidence, 'conflictStart'), _read(evidence, 'conflictEnd'), elapsed)
  if end <= start:
    return False
  state = str(_read(evidence, 'pathState'))
  if state == 'entering' and start == 0.0:
    state = 'inPath'
  # An expired coast has a constant-velocity prior. This bounded alignment is
  # only for remembering loss; it never refreshes the hypothesis sent to MPC.
  ground_speed = _read(lead, 'vLead')
  distance = _read(lead, 'dRel') + (ground_speed - speed) * elapsed
  return loss_relevant(distance=distance, ground_speed=ground_speed, ego_speed=speed, path_state=state,
                       conflict_start=start, conflict_end=end, lateral_speed=_read(evidence, 'lateralSpeed'))


class RadarConsumption:
  def __init__(self):
    self.clear()

  def clear(self):
    """Reset only with the planner's existing driver/control lifecycle."""
    self._lost = {}
    self._overflow_loss = False
    self.expired_identities = ()

  @property
  def degraded(self):
    return self._overflow_loss or bool(self._lost)

  @property
  def lost_identities(self):
    return tuple(self._lost)

  def update(self, radar, *, publication_ns, now_ns, speed, retain_losses=True):
    self.expired_identities = ()
    radar = _as_reader(radar)
    if (getattr(radar, 'fusionActive', False) is not True or not _integer(publication_ns, 1)
        or not _integer(now_ns, 1) or publication_ns > now_ns or not _finite(speed) or speed < 0.0):
      return radar
    publication = radar
    if type(radar).__module__ == 'capnp.lib.capnp':
      if len(radar.radarObjects) > MAX_CONTROL_OBJECTS:
        return publication
      # Each raw view is copied separately before conflict validation. Equal
      # identities must not conceal differing slots. Only field values are
      # shared by subsequent consumers; age/purpose gates still run each time.
      try:
        radar = RadarConsumptionView(radar, (), snapshot=True)
      except (RuntimeError, UnicodeError):
        # Eager conversion can decode fields that the original validation
        # short-circuits (for example an absent lead's invalid enum/text).
        # Preserve that validation order, including expiry of other objects.
        radar = publication
    objects = _published_objects(radar, publication_ns)
    if objects is None:
      return publication

    fresh_publication = (now_ns - publication_ns <= MAX_MEASUREMENT_AGE_NS
                         and _integer(getattr(radar, 'mdMonoTime', 0), 1)
                         and 0 <= now_ns - radar.mdMonoTime <= MAX_MEASUREMENT_AGE_NS
                         and radar.mdMonoTime <= publication_ns)
    if fresh_publication and getattr(radar, 'fusionHealthy', False) is True:
      session = getattr(radar, 'fusionSessionId', '')
      cleared = tuple(getattr(radar, 'fusionClearedTracks', ()))
      for identity, loss in tuple(self._lost.items()):
        if (identity[0] == session and identity[1] in cleared
            and all(_integer(track, 1) for track in cleared)
            and publication_ns > loss.publication_ns
            and not any(lead_identity(lead) == identity for lead in objects)):
          del self._lost[identity]
      for lead in objects:
        identity = lead_identity(lead)
        loss = self._lost.get(identity)
        evidence = _read(lead, 'evidence')
        if (loss is not None and publication_ns > loss.publication_ns
            and _read(evidence, 'coasting') is False
            and qualified_lead(lead, 'brake', 0.0, now_ns=now_ns)
            and _read(evidence, 'positionMonoTime') > loss.position_ns
            and _read(evidence, 'velocityMonoTime') > loss.velocity_ns):
          del self._lost[identity]

    expired = set()
    identities = []
    for lead in objects:
      if not coast_measurement_expired(lead, publication_ns=publication_ns, now_ns=now_ns):
        continue
      identity = lead_identity(lead)
      expired.add(_object_key(lead))
      identities.append(identity)
      if retain_losses and _relevant_at_consumption(lead, now_ns, speed):
        evidence = _read(lead, 'evidence')
        if identity not in self._lost and len(self._lost) >= MAX_CONTROL_OBJECTS:
          self._overflow_loss = True
        else:
          self._lost[identity] = _Loss(publication_ns, _read(evidence, 'positionMonoTime'),
                                       _read(evidence, 'velocityMonoTime'))
    self.expired_identities = tuple(identities)
    return RadarConsumptionView(radar, expired) if expired else radar
