"""Same-scan radar angular extent, independent of reflection-point centering.

The Bosch-A edge angles use the existing provisional forward range/azimuth
geometry. Their span is a radar width estimate, not a calibrated chassis size.
Missing edges never acquire a nominal vehicle width or an angular-width fallback.
"""
from dataclasses import dataclass
import math
from numbers import Real

BODY_MAX_AGE_NS = 200_000_000


def _read(value, name, default=None):
  return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _integer(value):
  return isinstance(value, int) and not isinstance(value, bool)


def _finite(*values):
  return all(isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) for value in values)


@dataclass(frozen=True)
class RadarBody:
  # Signed left-positive offsets from this scan's reflection. They need not
  # be symmetric: the reflection is not assumed to be the vehicle center.
  left: float
  right: float
  position_ns: int
  available_ns: int

  @property
  def width(self):
    return self.left - self.right

  def endpoints(self, x, y, angle):
    """Translate with the track and rotate the measured span into its epoch."""
    sine, cosine = math.sin(angle), math.cos(angle)
    return ((x - sine * self.left, y + cosine * self.left),
            (x - sine * self.right, y + cosine * self.right))


def measured_radar_body(observation, now_ns):
  """Decode complete, matching main frames; AUX cannot refresh this width."""
  if _read(observation, 'positionValid') is not True:
    return None
  slot, mask, index, stamp = (_read(observation, name) for name in
                            ('slot', 'frameMask', 'frameIndex', 'positionMonoTime'))
  if (not all(_integer(value) for value in (slot, mask, index, stamp, now_ns))
      or not 0 <= slot < 16 or mask not in (15, 31) or not 0 <= index < 16
      or not 0 < stamp <= now_ns or now_ns - stamp > BODY_MAX_AGE_NS):
    return None
  frames = _read(observation, 'frames', ())
  try:
    valid_shape = frames is not None and len(frames) == mask.bit_count()
  except TypeError:
    valid_shape = False
  if not valid_shape:
    return None
  base = 0x280 + 4 * slot if slot < 4 else 0x2D0 + 4 * (slot - 4)
  stamps = {}
  for frame in frames:
    part, address, frame_ns = (_read(frame, name) for name in ('part', 'address', 'monoTime'))
    if (not all(_integer(value) for value in (part, address, frame_ns))
        or not 0 <= part <= 4 or not mask & (1 << part) or part in stamps):
      return None
    expected = base + part if part < 4 else (0x2C8 + slot if slot < 8 else 0x290 + slot - 8)
    if address != expected or not 0 < frame_ns <= now_ns:
      return None
    stamps[part] = frame_ns
  main_stamps = tuple(stamps[part] for part in range(4))
  if stamps[0] != stamp or max(main_stamps) - min(main_stamps) > BODY_MAX_AGE_NS:
    return None
  raw = _read(observation, 'raw', {})
  distance_raw, azimuth, edge_a, edge_b = (_read(raw, name) for name in
                                        ('rangeRaw', 'azimuthRaw', 'edgeARaw', 'edgeBRaw'))
  if (not all(_integer(value) for value in (distance_raw, azimuth, edge_a, edge_b))
      or not 0 <= distance_raw < 4095 or not 0 <= azimuth < 2047
      or not 0 <= edge_a <= 2047 or not 0 <= edge_b <= 2047):
    return None
  right_raw, left_raw = sorted((edge_a, edge_b))
  # No edge-specific endpoint sentinel is established. Instead require a
  # nondegenerate measured span that actually contains this scan's reflection.
  if right_raw == left_raw or not right_raw <= azimuth <= left_raw:
    return None
  distance, lateral = (_read(observation, name) for name in ('dRel', 'yRel'))
  if (not _finite(distance, lateral) or distance <= 0
      or not math.isclose(distance, distance_raw * .05712 - 3.0, rel_tol=1e-9, abs_tol=1e-6)
      or not math.isclose(lateral, distance * math.tan((azimuth - 1024) / 2048.0), rel_tol=1e-9, abs_tol=1e-6)):
    return None
  return RadarBody(distance * math.tan((left_raw - 1024) / 2048.0) - lateral,
                   distance * math.tan((right_raw - 1024) / 2048.0) - lateral,
                   stamp, max(main_stamps))
