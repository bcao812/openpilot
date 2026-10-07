"""Bounded, read-only attribution for object labels on the driving display.

A solver's active object constraints become display evidence only after the
final planner arbitration and the motion owner's acceptance. Request strength
is qualitative; it is neither collision probability nor a causal percentage.
No helper in this module can change a plan, an owner, or an acceleration.
"""
import math
from numbers import Integral, Real

from openpilot.selfdrive.controls.lib.lead_qualification import MAX_CONTROL_OBJECTS, _read, control_leads, lead_identity


INFLUENCE_VERSION = 1
INFLUENCE_LEASE_NS = 250_000_000
FOLLOW_STRENGTH = 0.15
KINDS = ('follow', 'braking', 'hold', 'release', 'caution')


def _integer(value, minimum=0):
  return isinstance(value, Integral) and not isinstance(value, bool) and value >= minimum


def _finite(value):
  if not isinstance(value, Real) or isinstance(value, bool):
    return False
  try:
    return math.isfinite(value)
  except (OverflowError, ValueError):
    return False


def _reader(value):
  as_reader = getattr(value, 'as_reader', None)
  return as_reader() if callable(as_reader) else value


def _identity_record(lead, model_ns, index):
  evidence = _read(lead, 'evidence')
  physical = lead_identity(lead)
  source = str(_read(evidence, 'source', 'legacy'))
  version = _read(evidence, 'version', 0)
  if version == 0 and _read(lead, 'radar', False) is True:
    # Old OEM radar IDs have neither this observer's session identity nor a
    # guaranteed visual-frame association. Do not invent either for a label.
    return None
  if version == 1:
    model_ns = _read(evidence, 'modelMonoTime', 0)
    index = _read(evidence, 'visionIndex', -1) if source in ('vision', 'fused') else -1
  if physical is None and not (_integer(model_ns, 1) and _integer(index) and index in (0, 1)):
    return None
  session, track = physical if physical is not None else ('', 0)
  has_vision = _integer(model_ns, 1) and _integer(index) and index in (0, 1)
  return {'version': INFLUENCE_VERSION, 'sessionId': session, 'trackId': track,
          'modelMonoTime': int(model_ns) if has_vision else 0, 'visionIndex': int(index) if has_vision else -1}


def influence_identity(record):
  session, track = _read(record, 'sessionId', ''), _read(record, 'trackId', 0)
  if not isinstance(session, str) or not _integer(track):
    return None
  if isinstance(session, str) and session and _integer(track, 1):
    return session, int(track)
  model_ns, index = _read(record, 'modelMonoTime', 0), _read(record, 'visionIndex', -1)
  if session == '' and track == 0 and _integer(model_ns, 1) and _integer(index) and index in (0, 1):
    return 'vision', int(model_ns), int(index)
  return None


def request_strength(kind, acceleration, accel_min, accel_max):
  """Normalize an adopted request against its existing physical range."""
  if kind == 'hold':
    return 1.0
  if kind == 'caution':
    return FOLLOW_STRENGTH
  scale = -accel_min if acceleration < 0.0 else accel_max
  return float(min(0.95, max(FOLLOW_STRENGTH, abs(acceleration) / scale)))


def plan_object_influences(radar_state, *, active_identities, now_ns, permitted, fusion_admitted,
                           reference_accepted, independent_stop, object_stop, held_identity, release_identity,
                           standstill, target_accel, accel_min, accel_max, caution_identities=()):
  """Describe actual final-plan constraints, never mere lead qualification.

  ``active_identities`` comes from the accepted single MPC solve's active
  rows/costs, not its nearest-node display winner. The final arbitration must
  independently confirm that this reference survived any stronger scalar cap.
  A held object's separately retained departure barrier has its own identity.
  """
  if (not permitted or independent_stop or not _integer(now_ns, 1)
      or not all(_finite(value) for value in (target_accel, accel_min, accel_max))
      or not accel_min < 0.0 < accel_max or not accel_min <= target_accel <= accel_max):
    return []
  radar = _reader(radar_state)
  if fusion_admitted and _read(radar, 'fusionDegraded', False):
    return []
  raw_fusion = _read(radar, 'fusionActive', False)
  if raw_fusion and not fusion_admitted:
    leads = (_read(radar, 'visionLeadOne'), _read(radar, 'visionLeadTwo'))
  elif not raw_fusion:
    leads = (_read(radar, 'leadOne'), _read(radar, 'leadTwo'))
  else:
    leads = control_leads(radar)
  if len(leads) > MAX_CONTROL_OBJECTS:
    return []
  records, caution_candidates = {}, set()
  model_ns = _read(radar, 'mdMonoTime', 0)
  formal = (_read(radar, 'leadOne'), _read(radar, 'leadTwo'))
  for index, lead in enumerate(leads):
    if _read(lead, 'present', False) is not True:
      continue
    # Unversioned pure vision uses the publisher's model frame and its actual
    # display slot. Versioned fusion evidence carries its independent index.
    slot = index if not raw_fusion or not fusion_admitted else next((i for i, value in enumerate(formal) if lead is value), -1)
    record = _identity_record(lead, model_ns, slot)
    if record is not None:
      identity = influence_identity(record)
      records[identity] = record
      evidence = _read(lead, 'evidence')
      if (_read(evidence, 'cautionQualified', False) is True and str(_read(evidence, 'pathState', '')) == 'edge'
          and not _read(evidence, 'coasting', False)):
        caution_candidates.add(identity)
  active_identities = tuple(active_identities)
  caution_identities = tuple(caution_identities)
  if len(active_identities) > MAX_CONTROL_OBJECTS or len(caution_identities) > MAX_CONTROL_OBJECTS:
    return []
  selected = {}
  if fusion_admitted and release_identity in records and target_accel > 0.0:
    selected[release_identity] = 'release'
  elif fusion_admitted and held_identity in records:
    # This identity is a retained obstacle-specific departure barrier. The
    # owner still decides whether a HOLD was actually adopted this tick.
    selected[held_identity] = 'hold'
  if reference_accepted:
    for identity in active_identities:
      if identity in records and identity not in selected and identity not in caution_candidates:
        selected[identity] = 'hold' if object_stop and standstill else 'braking' if target_accel < 0.0 else 'follow'
  if fusion_admitted and target_accel == 0.0:
    # The producer supplies these identities only when the edge ceiling
    # strictly wins the final arbitration. Qualification alone is insufficient.
    for identity in caution_identities:
      if identity in caution_candidates and identity not in selected:
        selected[identity] = 'caution'
  return [dict(records[identity], kind=kind, strength=request_strength(kind, target_accel, accel_min, accel_max))
          for identity, kind in selected.items()]


def accepted_object_influences(plan, owner, car_state, *, long_active, plan_valid, plan_ns, now_ns, output_accel,
                               independent_stop=False):
  """Acknowledge the owner's actual same-tick adoption, clearing all others.

  This is called after LongControl.update and output validation. Inactive,
  degraded, stale or independently held commands do not inherit a red label
  from merely receiving a candidate plan.
  """
  if (not long_active or not plan_valid or independent_stop or not _read(owner, 'unified', False)
      or _read(owner, 'motion_input_constrained', True)
      or _read(car_state, 'gasPressed', False) or _read(car_state, 'brakePressed', False)
      or not _finite(output_accel) or not _integer(plan_ns, 1) or not _integer(now_ns, 1)
      or not 0 <= now_ns - plan_ns <= INFLUENCE_LEASE_NS):
    return []
  plan = _reader(plan)
  intent = _read(plan, 'motionIntent')
  model_ns = _read(plan, 'modelMonoTime', 0)
  if (not _read(intent, 'valid', False) or not _read(intent, 'ready', False)
      or _read(plan, 'solverStatus', -1) != 0 or not _integer(model_ns, 1)
      or not 0 <= now_ns - model_ns <= INFLUENCE_LEASE_NS):
    return []
  supervisor = _read(owner, 'motion_supervisor')
  if _read(supervisor, 'rest_braking_accel_limit') is not None:
    return []
  phase_value = _read(owner, 'motion_phase')
  phase = _read(phase_value, 'text', phase_value)
  resolved_stop = _read(owner, 'resolved_should_stop', False)
  tracking = str(_read(owner, 'long_control_state', 'off')) == 'pid'
  target = _read(plan, 'aTarget')
  if phase not in ('track', 'approach', 'hold', 'release') or not _finite(target):
    return []
  records = _read(plan, 'objectInfluences', ())
  if len(records) > MAX_CONTROL_OBJECTS:
    return []
  accepted, seen = [], set()
  for record in records:
    identity = influence_identity(record)
    kind, strength = str(_read(record, 'kind', '')), _read(record, 'strength')
    if (not _integer(_read(record, 'version')) or _read(record, 'version') != INFLUENCE_VERSION or identity is None or identity in seen
        or kind not in KINDS or not _finite(strength) or not 0.0 < strength <= 1.0
        or kind == 'caution' and strength > FOLLOW_STRENGTH + 1e-6):
      continue
    adopted = ((kind == 'hold' and resolved_stop and phase in ('hold', 'approach') and output_accel <= 0.0)
               or (kind == 'release' and tracking and not resolved_stop and phase in ('release', 'track')
                   and _read(intent, 'releaseAllowed', False) and target > 0.0)
               or (kind == 'braking' and phase in ('track', 'approach') and target < 0.0 and output_accel < 0.0)
               or (kind == 'follow' and tracking and phase == 'track' and not resolved_stop)
               or (kind == 'caution' and tracking and phase == 'track' and not resolved_stop and target == 0.0 and output_accel <= 0.0))
    if not adopted:
      continue
    # A received reader is copied field by field; no input builder is written
    # and no unvalidated field is forwarded to the display.
    has_physical = len(identity) == 2
    vision_index = _read(record, 'visionIndex', -1)
    vision_model = _read(record, 'modelMonoTime', 0)
    has_vision = _integer(vision_model, 1) and _integer(vision_index) and vision_index in (0, 1)
    accepted.append({'version': INFLUENCE_VERSION, 'sessionId': identity[0] if has_physical else '',
                     'trackId': identity[1] if has_physical else 0,
                     'modelMonoTime': int(vision_model) if has_vision else 0,
                     'visionIndex': int(vision_index) if has_vision else -1,
                     'kind': kind, 'strength': float(strength)})
    seen.add(identity)
  return accepted
