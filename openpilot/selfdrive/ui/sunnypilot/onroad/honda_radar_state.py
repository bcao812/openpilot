"""Bounded camera-time selection and formatting; independent of UI and control."""
from collections import deque
from dataclasses import dataclass, replace
import math
from numbers import Integral, Real

POSITION_MAX_AGE_NS = 200_000_000
MESSAGE_MAX_AGE_NS = 500_000_000


def _read(value, name, default=None):
  # Read only the bounded fields used by the overlay. Dictionaries are also
  # accepted for saved fixtures and offline viewers.
  return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _integer(value):
  return isinstance(value, Integral) and not isinstance(value, bool)


def _estimate_boundaries(event, message_ns):
  """Same-scan angular-bound display candidates, not calibrated body extents.

  The DBC defines both F3 edges as absolute azimuth candidates, in the same
  angular convention as F0. Preserve the existing provisional dRel origin:
  neither a second range offset nor symmetry around the reflection is justified.
  """
  missing = None, 0
  mask, index = _read(event, 'frameMask'), _read(event, 'frameIndex')
  if not (_integer(mask) and mask in (15, 31) and _integer(index) and 0 <= index < 16):
    return missing
  frames = _read(event, 'frames', ())
  try:
    if len(frames) != int(mask).bit_count():
      return missing
  except TypeError:
    return missing
  slot, position_ns = _read(event, 'slot'), _read(event, 'positionMonoTime')
  base = 0x280 + 4 * slot if slot < 4 else 0x2D0 + 4 * (slot - 4)
  stamps = {}
  for frame in frames:
    part, stamp, address = (_read(frame, name) for name in ('part', 'monoTime', 'address'))
    if not (_integer(part) and 0 <= part <= 4 and mask & (1 << part) and part not in stamps):
      return missing
    expected_address = base + part if part < 4 else (0x2C8 + slot if slot < 8 else 0x290 + slot - 8)
    if not (_integer(address) and address == expected_address and _integer(stamp) and 0 < stamp <= message_ns):
      return missing
    stamps[part] = stamp
  # F0 timestamps the reflection. All main frames must already exist at the
  # displayed camera time; a later AUX never moves either of these timestamps.
  main_stamps = [stamps[part] for part in range(4)]
  if stamps[0] != position_ns or max(main_stamps) - min(main_stamps) > POSITION_MAX_AGE_NS:
    return missing
  raw = _read(event, 'raw', {})
  distance_raw, azimuth, edge_a, edge_b = (_read(raw, name) for name in ('rangeRaw', 'azimuthRaw', 'edgeARaw', 'edgeBRaw'))
  if not (all(_integer(value) for value in (distance_raw, azimuth, edge_a, edge_b))
          and 0 <= distance_raw < 4095 and 0 <= azimuth < 2047 and 0 <= edge_a <= 2047 and 0 <= edge_b <= 2047):
    return missing
  # No edge-specific sentinel has been established: 0 and 2047 are legal
  # encodings here. Equal bounds or bounds that exclude the reflection cannot
  # support this display. ANGULAR_WIDTH_RAW is not a proven substitute.
  right_raw, left_raw = sorted((edge_a, edge_b))
  if right_raw == left_raw or not right_raw <= azimuth <= left_raw:
    return missing
  distance, lateral = _read(event, 'dRel'), _read(event, 'yRel')
  if not (math.isclose(distance, distance_raw * .05712 - 3.0, rel_tol=1e-9, abs_tol=1e-6)
          and math.isclose(lateral, distance * math.tan((azimuth - 1024) / 2048.0), rel_tol=1e-9, abs_tol=1e-6)):
    return missing
  return (distance * math.tan((left_raw - 1024) / 2048.0),
          distance * math.tan((right_raw - 1024) / 2048.0)), max(main_stamps)


@dataclass(frozen=True)
class RadarLabelPoint:
  slot: int
  distance: float
  lateral: float
  time_ns: int
  sequence: int
  generation: int
  session_id: str = ''
  track_id: int = 0
  boundary_laterals: tuple[float, float] | None = None  # left (larger y), right; same distance as reflection
  boundary_time_ns: int = 0

  @property
  def text(self):
    return f'{self.distance:.1f}'

  @property
  def sample_key(self):
    return self.slot, self.generation, self.sequence, self.time_ns


def _track_samples(body):
  """Attach only the track that explicitly consumed this exact CAN sample.

  Spatial proximity cannot distinguish a nearby cut-in from a farther car.
  Slot generations alone also cannot survive track lifecycles. The observer's
  bounded fusion snapshot supplies the physical identity for a measured sample.
  """
  fusion = _read(body, 'fusion', {})
  tracks = _read(fusion, 'tracks', ())
  if (_read(fusion, 'version', 0) != 1 or _read(fusion, 'active', False) is not True
      or len(tracks) > 32):
    return {}
  identities = {}
  for track in tracks:
    key = tuple(_read(track, field) for field in ('slot', 'generation', 'positionSeq', 'positionMonoTime'))
    identifier = _read(track, 'trackId')
    if (not all(isinstance(value, Integral) and not isinstance(value, bool) for value in (*key, identifier))
        or not 0 <= key[0] < 16 or key[1] < 0 or min(*key[2:], identifier) <= 0):
      continue
    # Conflicting identities never borrow the stronger target's red color.
    identities[key] = identifier if key not in identities or identities[key] == identifier else -1
  return identities


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
    as_reader = getattr(body, 'as_reader', None)
    if callable(as_reader):
      body = as_reader()
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
    track_samples = _track_samples(body)
    if track_samples:
      for history in self.history:
        for index, point in enumerate(history):
          if point.sample_key in track_samples:
            identifier = track_samples[point.sample_key]
            if point.track_id == -1 or point.track_id > 0 and point.track_id != identifier:
              identifier = -1
            if identifier != point.track_id:
              history[index] = replace(point, track_id=identifier)
    for event in events:
      slot = _read(event, 'slot')
      if not _integer(slot) or not 0 <= slot < 16:
        continue
      history = self.history[slot]
      if _read(event, 'positionValid', False):
        sequence = _read(event, 'positionSeq')
        if not _integer(sequence) or sequence <= self.position_sequence[slot]:
          continue  # AUX cannot restore an invalidated/cleared position sample
        t = _read(event, 'positionMonoTime')
        d, y = _read(event, 'dRel'), _read(event, 'yRel')
        if not (_integer(t) and self.invalid_before_ns[slot] < t <= message_ns
                and all(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) for value in (d, y)) and d > 0):
          continue
        generation = _read(event, 'generation')
        if not _integer(generation) or generation < 0:
          continue
        boundaries, boundary_ns = _estimate_boundaries(event, message_ns)
        point = RadarLabelPoint(slot, d, y, t, sequence, generation, session,
                                track_samples.get((slot, generation, sequence, t), 0), boundaries, boundary_ns)
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
          if _integer(t) and 0 < t <= message_ns:
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
            # A complete scan can be received after its F0 timestamp. Keep the
            # reflection's existing camera-time behavior without painting its
            # later angular bounds into an earlier image.
            result.append(replace(point, boundary_laterals=None, boundary_time_ns=0)
                          if point.boundary_time_ns > frame_ns else point)
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
