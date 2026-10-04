"""Bounded camera-time selection and formatting; independent of UI and control."""
from collections import deque
from dataclasses import dataclass
import math

POSITION_MAX_AGE_NS = 200_000_000
MESSAGE_MAX_AGE_NS = 500_000_000


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
    if not 0 < message_ns <= now_ns or now_ns - message_ns > MESSAGE_MAX_AGE_NS:
      self.clear()
      return
    session, sequence = body['sessionId'], body['sequence']
    if session == self.session and sequence <= self.sequence:
      return
    dropped = body.get('counters', {}).get('dropped', 0)
    if session != self.session or sequence != self.sequence + 1 or dropped != self.dropped:
      self.clear()
    self.session, self.sequence, self.dropped = session, sequence, dropped
    self.message_ns = message_ns
    scope = body.get('scope', {})
    self.active = body['active'] and all(scope.get(k, False) for k in
                                      ('alphaLong', 'experimentalSetting', 'decSetting', 'experimentalActive', 'decEnabled', 'decActive'))
    events = body.get('observations', [])
    if not self.active or len(events) > 64:
      self.clear()
      return
    for event in events:
      slot = event['slot']
      if not 0 <= slot < 16:
        continue
      history = self.history[slot]
      if event.get('positionValid', False):
        t = event['positionMonoTime']
        d, y = event['dRel'], event['yRel']
        if not (0 < t <= message_ns and math.isfinite(d) and math.isfinite(y) and d > 0):
          continue
        point = RadarLabelPoint(slot, d, y, t, event['positionSeq'], event['generation'])
        if history and t < history[-1].time_ns:
          continue
        if history and point.sequence == history[-1].sequence:
          continue  # late AUX must not refresh or duplicate a position
        if history and point.generation != history[-1].generation:
          history.clear()
        history.append(point)
      elif history:
        reason = event.get('reason')
        if reason == 'expired' and event.get('generation') == history[-1].generation:
          history.clear()
        elif reason in ('invalidGeometry', 'conflictingFrame'):
          t = max((f['monoTime'] for f in event.get('frames', [])), default=0)
          if t >= history[-1].time_ns:
            history.clear()

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
