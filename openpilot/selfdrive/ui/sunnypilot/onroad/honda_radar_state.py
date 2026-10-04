"""Bounded camera-time selection and formatting; independent of UI and control."""
from collections import deque
from dataclasses import dataclass
import math

POSITION_MAX_AGE_NS = 200_000_000
MESSAGE_MAX_AGE_NS = 500_000_000


def _read(value, name, default=None):
  # Keep the GUI off the much larger raw/quality/frame-reference subtrees.
  # Dictionaries are also accepted for saved fixtures and offline viewers.
  return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


@dataclass(frozen=True)
class RadarLabelPoint:
  slot: int
  distance: float
  lateral: float
  time_ns: int
  sequence: int
  generation: int

  @property
  def text(self):
    return f'{self.slot} {self.distance:.1f}'


class RadarOverlayState:
  def __init__(self):
    self.history = [deque(maxlen=16) for _ in range(16)]
    self.position_sequence = [0] * 16
    self.invalid_before_ns = [0] * 16
    self.session = ''
    self.sequence = 0
    self.dropped = 0
    self.message_ns = 0
    self.active = False

  def clear(self):
    for history in self.history:
      history.clear()
    self.active = False

  def ingest(self, body, message_ns, now_ns):
    if _read(body, 'schemaVersion', 1) != 1 or not 0 < message_ns <= now_ns or now_ns - message_ns > MESSAGE_MAX_AGE_NS:
      self.clear()
      return
    session, sequence = _read(body, 'sessionId'), _read(body, 'sequence')
    if session == self.session and sequence <= self.sequence:
      return
    dropped = _read(_read(body, 'counters', {}), 'dropped', 0)
    if session != self.session or sequence != self.sequence + 1 or dropped != self.dropped:
      self.clear()
    if session != self.session:
      self.position_sequence = [0] * 16
      self.invalid_before_ns = [0] * 16
    self.session, self.sequence, self.dropped = session, sequence, dropped
    self.message_ns = message_ns
    scope = _read(body, 'scope', {})
    self.active = _read(body, 'active', False) and all(_read(scope, k, False) for k in
                                      ('alphaLong', 'experimentalSetting', 'decSetting', 'experimentalActive', 'decEnabled', 'decActive'))
    events = _read(body, 'observations', [])
    if not self.active or len(events) > 64:
      self.clear()
      return
    for event in events:
      slot = _read(event, 'slot')
      if not 0 <= slot < 16:
        continue
      history = self.history[slot]
      if _read(event, 'positionValid', False):
        sequence = _read(event, 'positionSeq')
        if sequence <= self.position_sequence[slot]:
          continue  # AUX cannot restore an invalidated/cleared position sample
        t = _read(event, 'positionMonoTime')
        d, y = _read(event, 'dRel'), _read(event, 'yRel')
        if not (self.invalid_before_ns[slot] < t <= message_ns and math.isfinite(d) and math.isfinite(y) and d > 0):
          continue
        point = RadarLabelPoint(slot, d, y, t, sequence, _read(event, 'generation'))
        if history and t < history[-1].time_ns:
          continue
        if history and point.generation != history[-1].generation:
          history.clear()
        self.position_sequence[slot] = sequence
        history.append(point)
      else:
        reason = _read(event, 'reason')
        if reason == 'conflictingFrame' and not _read(event, 'positionSeq', 0):
          # A never-completed scan cannot identify a published position to
          # retract. Previously accepted positions retain their own 200 ms TTL.
          continue
        if reason in ('expired', 'invalidGeometry', 'conflictingFrame'):
          t = _read(event, 'positionMonoTime', 0)
          if 0 < t <= message_ns:
            self.invalid_before_ns[slot] = max(self.invalid_before_ns[slot], t)
            while history and history[0].time_ns <= t:
              history.popleft()

  def points(self, frame_ns, now_ns):
    if (not self.active or not 0 < frame_ns <= now_ns or now_ns - frame_ns > MESSAGE_MAX_AGE_NS
        or not 0 <= now_ns - self.message_ns <= MESSAGE_MAX_AGE_NS):
      return []
    result = []
    for history in self.history:
      for point in reversed(history):
        if point.time_ns <= frame_ns:
          if now_ns - point.time_ns <= POSITION_MAX_AGE_NS:
            result.append(point)
          break
    return result


def project_radar_point(point, transform, height, offset=0.0):
  # Same calibrated vehicle->camera convention as ModelRenderer. Height is a
  # ground-plane display assumption; no unknown radar elevation is invented.
  xyz = (point.distance, -point.lateral + offset, height)
  if not all(math.isfinite(v) for v in xyz) or point.distance <= 0:
    return None
  projected = [sum(row[i] * xyz[i] for i in range(3)) for row in transform]
  if not all(math.isfinite(v) for v in projected) or projected[2] <= 1e-6:
    return None
  return projected[0] / projected[2], projected[1] / projected[2]
