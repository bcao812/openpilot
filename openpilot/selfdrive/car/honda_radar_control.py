"""Fuse qualified Accord radar objects into the two formal lead slots.

The observer owns decoding, motion compensation and physical track identity.
This consumer checks current measurements, aligns them to the current camera
frame, and assigns ordinary following, stopping and departure qualifications.
Measured body intrusion has a separate, restricted braking qualification.
Missing position, velocity or path evidence never creates a predicted lead.
It does not publish messages, read settings or own actuation. In particular, a
missing visual detection is not evidence that a radar obstacle does not exist.

Bounds below are engineering admission parameters, not sensor calibration. The
two collected drives contain no observed in-path metal-cover/sign false targets;
this version therefore makes no speculative stationary-object class rejection.
"""
from dataclasses import dataclass, field, replace
import math
from numbers import Real

from capnp.lib.capnp import _DynamicStructBuilder, _DynamicStructReader

from openpilot.selfdrive.car.honda_radar_relevance import loss_relevant
from openpilot.selfdrive.car.honda_radar_native import (
  MAX_PAIR_NS as NATIVE_MAX_PAIR_NS, MIN_PAIRS as NATIVE_MIN_PAIRS,
  MIN_SPAN_NS as NATIVE_MIN_SPAN_NS, VERSION as NATIVE_VERSION,
)


VERSION = 1
FUSION_VERSION = 'bosch-a-native-4'
MAX_TRACKS = 32
MAX_AGE_NS = 200_000_000
MIN_OBSERVATIONS = 3
MIN_SPAN_NS = 120_000_000
# Current radar/fused motion only; visual fallback keeps its own 0.5 m/s gate.
RADAR_DEPARTURE_MIN_SPEED = 0.3
HORIZON = 10.0
RADAR_TO_CAMERA = 1.52
# Ordinary following still uses point/visual geometry. The independent cut-in
# brake qualification below uses the measured asymmetric body extent instead.
OVERLAP_HALF_WIDTH = 2.0
SWEPT_HALF_WIDTH = 1.25  # 1.10 m ego half-width + .15 m geometric margin
LANE_SUPPORT_M = 60.0
LANE_MIN_PROB = .65
LANE_MAX_STD = .5
POSITION_MAX_VARIANCE = 9.0
VELOCITY_MAX_VARIANCE = 9.0
POSITION_SPEED_MAX_VARIANCE = 4.0
LATERAL_POSITION_MAX_VARIANCE = 4.0
LATERAL_VELOCITY_MAX_VARIANCE = 4.0
# Initial engineering admission bounds, pending route/vehicle validation.
CUT_IN_HALF_WIDTH = 1.0  # Body intrusion only; ordinary point admission keeps SWEPT_HALF_WIDTH.
CUT_IN_NEAR_DISTANCE = 6.0
CUT_IN_NEAR_MIN_INTRUSION = .15
CUT_IN_MIN_WIDTH = .4
CUT_IN_MAX_WIDTH = 3.2
CUT_IN_MAX_DISTANCE = 45.0
CUT_IN_MAX_TTC = 4.0
CUT_IN_MIN_LATERAL_SPEED = .1
CUT_IN_MAX_LATERAL_SPEED = 1.8
CUT_IN_MAX_POSITION_VARIANCE = .64
CUT_IN_WIDTH_STEP = .35
CUT_IN_MIN_LATERAL_PROGRESS = .02
CUT_IN_REASON = 'radarBodyCutIn'
# Dynamic cereal fields can call their native handler directly. Keep static
# properties/methods and every other object on the ordinary lookup path.
_CEREAL_READERS = {cls: (cls.__getattr__, frozenset(dir(cls))) for cls in (_DynamicStructBuilder, _DynamicStructReader)}


def _read(obj, name, default=None):
  if isinstance(obj, dict):
    return obj.get(name, default)
  native = _CEREAL_READERS.get(type(obj))
  if native is not None and type(name) is str and name not in native[1]:
    try:
      return native[0](obj, name)
    except AttributeError:
      pass
  return getattr(obj, name, default)


def _finite(*values):
  # Cereal produces exact builtins. Retain Real/subclass handling, boolean
  # rejection and conversion errors for other callers without ABC checks on
  # each ordinary scalar in the per-frame geometry/qualification loops.
  return all((type(v) is float or type(v) is int or isinstance(v, Real) and not isinstance(v, bool))
             and math.isfinite(v) for v in values)


def _integer(value):
  return isinstance(value, int) and not isinstance(value, bool)


def _fresh(stamp, now, age=MAX_AGE_NS):
  return _integer(stamp) and 0 < stamp <= now and now - stamp <= age


def _first(obj, name, default=math.nan):
  values = _read(obj, name, ())
  return values[0] if values is not None and len(values) else default


def _project(x, y, vx, vy, elapsed, speed, yaw):
  """Left-positive radar coordinates, model's right-positive yaw convention.

  Translate the ego origin and rotate both position and ground velocity. A
  turning ego must not create target ground motion or lateral crossing speed.
  Acceleration is intentionally not integrated over this short alignment step.
  """
  angle = yaw * elapsed
  c, s = math.cos(angle), math.sin(angle)
  if abs(yaw) > 1e-6:
    ego_x = speed * math.sin(angle) / yaw
    ego_y = -speed * (1.0 - math.cos(angle)) / yaw
  else:
    ego_x, ego_y = speed * elapsed, 0.0
  x, y = x + vx * elapsed - ego_x, y + vy * elapsed - ego_y
  return c * x - s * y, s * x + c * y, c * vx - s * vy, s * vx + c * vy


def _lateral_variances(position_variance, lateral_position_variance, velocity_variance, lateral_velocity_variance, elapsed, yaw):
  """Align marginal bounds without assuming unpublished cross-covariances are zero."""
  py = lateral_position_variance if _finite(lateral_position_variance) and lateral_position_variance >= 0 else math.inf
  vy = lateral_velocity_variance if _finite(lateral_velocity_variance) and lateral_velocity_variance >= 0 else math.inf
  tau = abs(elapsed)
  if tau == 0:
    return py, vy
  if not _finite(py, vy):
    return math.inf, math.inf
  c, s = abs(math.cos(yaw * elapsed)), abs(math.sin(yaw * elapsed))
  # Triangle bounds cover arbitrary position/velocity cross-correlation.
  px_std = math.sqrt(position_variance) + tau * math.sqrt(velocity_variance)
  py_std = math.sqrt(py) + tau * math.sqrt(vy)
  return ((s * px_std + c * py_std)**2 + .02 * tau + tau**3,
          (s * math.sqrt(velocity_variance) + c * math.sqrt(vy))**2 + 3 * tau)


@dataclass(frozen=True)
class RadarControlResult:
  leads: tuple[dict, dict]
  objects: list[dict]
  healthy: bool
  degraded: bool
  reason: str
  cleared: tuple[int, ...] = ()
  session_id: str = ''


@dataclass(frozen=True)
class _Path:
  state: str = 'unknown'
  offset: float = 0.0
  rate: float = 0.0
  source: str = 'unavailable'
  lateral_motion_valid: bool = False
  observed_out: bool = False
  confirmed: bool = False
  authority_reason: str = ''


@dataclass(frozen=True)
class _Corridor:
  # Body/radar coordinates, left positive, with finite spatial support.
  points: tuple
  source: str
  _segments: tuple = field(init=False, repr=False)

  def __post_init__(self):
    segments = []
    for (px, py), (qx, qy) in zip(self.points, self.points[1:], strict=False):
      length = math.hypot(qx - px, qy - py)
      if length > 1e-4:
        segments.append((px, py, (qx - px) / length, (qy - py) / length, length))
    object.__setattr__(self, '_segments', tuple(segments))

  def current_distance_exceeds(self, x, y, limit):
    """Current disagreement for finite geometry already checked by _points.

    The lane gate only asks whether every segment is farther than its limit.
    Finding any segment within it settles that question without choosing a
    signed nearest projection or computing motion along the other segments.
    """
    if not self._segments or not self.points[0][0] <= x <= self.points[-1][0]:
      return False  # No supported current relation, as in relation().
    for px, py, dx, dy, length in self._segments:
      rx, ry = x - px, y - py
      along = rx * dx + ry * dy
      nearest = min(length, max(0.0, along))
      if math.hypot(rx - nearest * dx, ry - nearest * dy) <= limit:
        return False
    return True

  def relation(self, x, y, vx, vy, half_width=OVERLAP_HALF_WIDTH):
    """Current point-to-polyline relation and motion normal to that segment."""
    if not self.points or not self.points[0][0] <= x <= self.points[-1][0]:
      return _Path()
    best = None
    for px, py, dx, dy, length in self._segments:
      along = (x - px) * dx + (y - py) * dy
      offset = (y - py) * dx - (x - px) * dy
      nearest = min(length, max(0.0, along))
      distance = math.hypot(x - px - nearest * dx, y - py - nearest * dy)
      if best is None or distance < abs(best[0]):
        best = math.copysign(distance, offset), vy * dx - vx * dy
    if best is None:
      return _Path()
    offset, rate = best
    state = 'inPath' if abs(offset) <= half_width else 'outOfPath'
    return _Path(state, offset, rate, source=self.source)


def _qualified_path(corridors, x, y, vx, vy, position_variance, velocity_variance, velocity_valid, half_width=SWEPT_HALF_WIDTH):
  """Qualify current geometry and lateral motion without forecasting occupancy."""
  if not _finite(position_variance) or not 0 <= position_variance <= LATERAL_POSITION_MAX_VARIANCE:
    return _Path()
  motion_valid = (velocity_valid is True and _finite(velocity_variance)
                  and 0 <= velocity_variance <= LATERAL_VELOCITY_MAX_VARIANCE)
  for corridor in corridors:
    current = corridor.relation(x, y, vx, vy, half_width)
    if current.state != 'unknown':
      return replace(current, lateral_motion_valid=motion_valid, observed_out=current.state == 'outOfPath')
  return _Path()


def _object_path(corridors, x, y, vx, vy, position_variance, velocity_variance, velocity_valid, *, vision=False):
  inner = _qualified_path(corridors, x, y, vx, vy, position_variance, velocity_variance, velocity_valid)
  if inner.state == 'inPath':
    return replace(inner, confirmed=True)
  if vision:
    # Same-car visual identity can place the vehicle body in path even when
    # its radar reflection is offset from that body's center.
    return _Path('inPath', inner.offset, inner.rate, source='visionLead', confirmed=True,
                 authority_reason='confirmedVisionPath')
  outer = _qualified_path(corridors, x, y, vx, vy, position_variance, velocity_variance, velocity_valid,
                           half_width=OVERLAP_HALF_WIDTH)
  # Current radar measurements renew object identity, not a past conclusion
  # that its body occupies the path. A former inner/visual confirmation cannot
  # supply today's missing occupancy evidence. Keep the wider band solely for
  # measured-OUT reporting: an outer-band reflection is neither a qualified
  # ordinary lead nor proof that the whole vehicle body has cleared the path.
  return replace(inner, observed_out=outer.state == 'outOfPath')


def _points(line, limit=math.inf):
  xs, ys = _read(line, 'x', ()), _read(line, 'y', ())
  if xs is None or ys is None or len(xs) != len(ys) or not 2 <= len(xs) <= 33:
    return ()
  result = []
  for x, y in zip(xs, ys, strict=True):
    if not _finite(x, y) or abs(y) > 200 or x < -5 or x > 300:
      return ()
    if result and x - RADAR_TO_CAMERA < result[-1][0]:
      return ()
    if x <= limit and (not result or x - RADAR_TO_CAMERA > result[-1][0] + .01):
      result.append((float(x) - RADAR_TO_CAMERA, -float(y)))
  if len(result) < 2 or result[-1][0] - result[0][0] < 1.0:
    return ()
  return tuple(result)


def _corridors(model):
  path = _Corridor(_points(_read(model, 'position')), 'modelPath')
  lines = _read(model, 'laneLines', ())
  probs, stds = _read(model, 'laneLineProbs', ()), _read(model, 'laneLineStds', ())
  lane = _Corridor((), 'laneLines')
  if lines is None or probs is None or stds is None or len(lines) != 4 or len(probs) != 4 or len(stds) != 4:
    return path, lane
  if not all(_finite(probs[i], stds[i]) and LANE_MIN_PROB <= probs[i] <= 1 and 0 <= stds[i] <= LANE_MAX_STD for i in (1, 2)):
    return path, lane
  left, right = (_points(lines[i], LANE_SUPPORT_M) for i in (1, 2))
  if not left or len(left) != len(right):
    return path, lane
  centers = []
  for (lx, ly), (rx, ry) in zip(left, right, strict=True):
    if abs(lx - rx) > .05 or not 2.2 <= ly - ry <= 4.8:
      return path, lane
    center = (lx + rx) * .5, (ly + ry) * .5
    # A lane change/turn whose actual planned path disagrees with the lane
    # center cannot extend that path using a different lane's distant geometry.
    if path.current_distance_exceeds(*center, .75):
      return path, lane
    centers.append(center)
  return path, _Corridor(tuple(centers), 'laneLines')


@dataclass
class _Memory:
  track_id: int
  oem_id: int
  generation: int
  position_seq: int
  position_ns: int
  velocity_ns: int
  epoch_ns: int
  x: float
  y: float
  vx: float
  vy: float
  acceleration: float
  acceleration_good: bool
  position_variance: float
  velocity_variance: float
  path: _Path
  path_ns: int
  accepted: dict | None = None
  selected_follow: bool = False
  raw_velocity_ns: int = 0
  relevant: bool = False
  last_vision: dict | None = None
  lateral_position_variance: float = math.inf
  lateral_velocity_variance: float = math.inf
  lateral_velocity_valid: bool = False


@dataclass(frozen=True)
class _BodyApproach:
  generation: int
  sequence: int
  position_ns: int
  start_ns: int
  first_offset: float
  offset: float
  width: float
  samples: int = 1


_TRACK_FIELDS = ('trackId', 'oemTrackId', 'generation', 'slot', 'positionSeq', 'positionMonoTime', 'velocityMonoTime',
                 'dRel', 'yRel', 'groundSpeed', 'lateralSpeed', 'positionVariance', 'velocityVariance', 'observations',
                 'mature', 'velocityGood', 'coasting', 'active', 'lastInvalidReason', 'lastInvalidMonoTime',
                 'previousPositionSeq', 'previousPositionMonoTime', 'acceleration', 'accelerationGood',
                 'lateralPositionVariance', 'lateralVelocityVariance', 'lateralVelocityValid',
                 'bodyValid', 'bodyMonoTime', 'bodyAvailableMonoTime', 'bodyWidth',
                 'bodyLeftX', 'bodyLeftY', 'bodyRightX', 'bodyRightY', 'nativeMotion')
_BINDING_FIELDS = ('modelIndex', 'present', 'probability', 'trackId', 'candidateTrackId', 'identityState', 'rejectionMonoTime',
                   'bindingStartMonoTime', 'bindingSamples', 'bindingAnchorX', 'bindingAnchorY', 'identityReason', 'reason')
_NATIVE_MOTION_FIELDS = ('version', 'qualified', 'reason', 'positionSeq', 'positionMonoTime', 'velocitySeq',
                       'velocityMonoTime', 'sampleCount', 'firstPositionMonoTime', 'validUntilMonoTime',
                       'rawRelativeSpeed', 'groundSpeed', 'velocityVariance', 'acceleration', 'accelerationValid',
                       'u10Raw', 'rangeSigmaRaw', 'existenceRaw', 'pairCompleteMonoTime', 'distanceResidual',
                       'distanceResidualValid', 'frameIndex')


def _native_motion_snapshot(evidence):
  if evidence is None:
    return None
  # The publication identity must own its values. A capnp reader or nested
  # mutable fixture must not rewrite the previously accepted signature.
  values = (_read(evidence, key) for key in _NATIVE_MOTION_FIELDS)
  snapshot = []
  for value in values:
    if isinstance(value, Real) and not math.isfinite(value):
      # Re-reading identical capnp bytes creates new NaN floats, which do not
      # compare equal. Stable nonnumeric tokens preserve publication identity
      # while the quality validator still rejects these invalid components.
      value = ('nonfinite', 'nan' if math.isnan(value) else '+inf' if value > 0 else '-inf')
    elif value is not None and not isinstance(value, (Real, str)):
      value = 'invalidScalar'
    snapshot.append(value)
  return tuple(snapshot)


def _native_motion_state(evidence, track, *, now_ns, observation_ns):
  """Validate the producer's complete-pair contract independently of its flag.

  Missing/explicitly unqualified native evidence is an ordinary absent radar
  candidate. A contradictory qualified claim remains an input integrity error.
  Neither path may borrow visual or position-derived velocity to authorize it.
  """
  if evidence is None:
    return 'pending'
  quality = dict(zip(_NATIVE_MOTION_FIELDS, evidence, strict=True))
  version = quality['version']
  if version is None or _integer(version) and version == 0:
    return 'pending'
  if not _integer(version) or version != NATIVE_VERSION or not isinstance(quality['qualified'], bool):
    return 'invalid'
  if not quality['qualified']:
    return 'pending'
  # V2 is a producer admission contract, including independent startup motion
  # consistency. A pending/recovery reason cannot accompany a qualified claim.
  if quality['reason'] != 'qualified':
    return 'invalid'
  p_ns, v_ns = track['positionMonoTime'], track['velocityMonoTime']
  integers = ('positionSeq', 'positionMonoTime', 'velocitySeq', 'velocityMonoTime', 'sampleCount',
              'firstPositionMonoTime', 'validUntilMonoTime', 'pairCompleteMonoTime', 'u10Raw',
              'rangeSigmaRaw', 'existenceRaw', 'frameIndex')
  if (not all(_integer(quality[key]) for key in integers)
      or quality['positionSeq'] != track['positionSeq'] or quality['positionMonoTime'] != p_ns
      or quality['velocityMonoTime'] != v_ns or quality['velocitySeq'] <= 0
      or not _integer(v_ns) or not _integer(track['observations'])
      or not NATIVE_MIN_PAIRS <= quality['sampleCount'] <= track['observations']
      or not 0 < quality['firstPositionMonoTime'] <= p_ns - NATIVE_MIN_SPAN_NS
      or abs(v_ns - p_ns) > NATIVE_MAX_PAIR_NS
      or not max(p_ns, v_ns) <= quality['pairCompleteMonoTime'] <= observation_ns
      or quality['validUntilMonoTime'] != min(p_ns, v_ns) + MAX_AGE_NS
      or not 0 <= quality['u10Raw'] < 256 or not 0 <= quality['frameIndex'] < 16
      or not 0 <= quality['rangeSigmaRaw'] <= 127 or not 0 <= quality['existenceRaw'] <= 127
      or not _finite(quality['rawRelativeSpeed'], quality['groundSpeed'], quality['velocityVariance'])
      or not -13.5 < quality['rawRelativeSpeed'] < 13.5
      or quality['groundSpeed'] != track['groundSpeed'] or quality['velocityVariance'] != track['velocityVariance']
      or track['velocityGood'] is not True
      or not isinstance(quality['accelerationValid'], bool)
      or quality['accelerationValid'] != track['accelerationGood']
      or quality['accelerationValid'] and (not _finite(quality['acceleration'])
          or not -10 <= quality['acceleration'] <= 5 or quality['acceleration'] != track['acceleration'])):
    return 'invalid'
  if not _fresh(p_ns, now_ns) or not _fresh(v_ns, now_ns) or now_ns > quality['validUntilMonoTime']:
    return 'pending'
  return 'qualified'


class HondaRadarControl:
  def __init__(self):
    self.reset()

  def reset(self):
    self._tracks: dict[int, _Memory] = {}
    self._publication = None
    self._last_now_ns = self._model_ns = self._eof = 0
    self._session = ''
    self._cleared: dict[int, int] = {}
    # Approach history is observational, bounded by the current track batch.
    # It never supplies a missing width, position or velocity measurement.
    self._body_approaches: dict[int, _BodyApproach] = {}

  def _cut_in_path(self, track, corridors, *, source_eof, eof, now_ns, observation_ns, speed, yaw,
                   vx, vy, position_variance, velocity_variance, direct_velocity, previous):
    stamp, sequence = track['positionMonoTime'], track['positionSeq']
    available = track['bodyAvailableMonoTime']
    if (track['bodyValid'] is not True or track['bodyMonoTime'] != stamp
        or not _fresh(stamp, now_ns) or not _integer(available) or not stamp <= available <= observation_ns
        or not direct_velocity or track['lateralVelocityValid'] is not True
        or track['velocityVariance'] > POSITION_SPEED_MAX_VARIANCE
        or not _finite(position_variance) or not 0 <= position_variance <= CUT_IN_MAX_POSITION_VARIANCE
        or not _finite(velocity_variance) or not 0 <= velocity_variance <= LATERAL_VELOCITY_MAX_VARIANCE
        or speed < 3.0 or vx < 2.0):
      return None
    width, lx, ly, rx, ry = (track[key] for key in ('bodyWidth', 'bodyLeftX', 'bodyLeftY', 'bodyRightX', 'bodyRightY'))
    if (not _finite(width, lx, ly, rx, ry) or not CUT_IN_MIN_WIDTH <= width <= CUT_IN_MAX_WIDTH
        or not math.isclose(math.hypot(lx - rx, ly - ry), width, rel_tol=1e-5, abs_tol=1e-5)):
      return None
    # The span and reflection must belong to the same projected sample. Width
    # alone cannot relocate a reflection onto an unrelated neighboring body.
    dx, dy = lx - rx, ly - ry
    along = ((track['dRel'] - rx) * dx + (track['yRel'] - ry) * dy) / width**2
    cross = abs((track['dRel'] - rx) * dy - (track['yRel'] - ry) * dx) / width
    if not -1e-5 <= along <= 1.0 + 1e-5 or cross > 1e-4:
      return None
    elapsed = (eof - source_eof) / 1e9
    endpoints = [_project(x, y, track['groundSpeed'], track['lateralSpeed'], elapsed, speed, yaw)
                 for x, y in ((lx, ly), (rx, ry))]
    paths = [_qualified_path(corridors, x, y, vx, vy, position_variance, velocity_variance, True,
                             half_width=CUT_IN_HALF_WIDTH)
             for x, y, _, _ in endpoints]
    left, right = paths
    if left.state == 'unknown' or right.state == 'unknown' or left.source != right.source:
      return None
    offset, rate = .5 * (left.offset + right.offset), .5 * (left.rate + right.rate)
    # Ground motion normal to the curved path excludes parallel traffic and
    # stationary roadside returns; ego yaw alone is not an inward target speed.
    if (offset * rate >= 0 or not CUT_IN_MIN_LATERAL_SPEED <= abs(rate) <= CUT_IN_MAX_LATERAL_SPEED
        or abs(rate) > .3 * vx):
      return None
    generation = track['generation']
    same_identity = previous is not None and previous.generation == generation
    if same_identity and stamp == previous.position_ns and sequence == previous.sequence:
      if not math.isclose(width, previous.width, rel_tol=1e-5, abs_tol=1e-5):
        return None
      approach = previous  # Another model/publication is not another radar sample.
    else:
      continuous = (same_identity and sequence > previous.sequence and 0 < stamp - previous.position_ns <= MAX_AGE_NS
                    and offset * previous.offset > 0 and abs(offset) <= abs(previous.offset) + .03
                    and abs(offset - previous.offset) <= .4 and abs(width - previous.width) <= CUT_IN_WIDTH_STEP)
      approach = _BodyApproach(generation, sequence, stamp, previous.start_ns if continuous else stamp,
                               previous.first_offset if continuous else offset, offset, width,
                               min(255, previous.samples + 1) if continuous else 1)
    self._body_approaches[track['trackId']] = approach
    if (approach.samples < MIN_OBSERVATIONS or stamp - approach.start_ns < MIN_SPAN_NS
        or abs(approach.first_offset) - abs(approach.offset) < CUT_IN_MIN_LATERAL_PROGRESS):
      return None
    low, high = sorted((left.offset, right.offset))
    # Actual current span overlap, not a prediction that a point may enter.
    if low > CUT_IN_HALF_WIDTH or high < -CUT_IN_HALF_WIDTH:
      return None
    distance = min(point[0] for point in endpoints)
    closing = speed - vx
    if not 0 < distance <= CUT_IN_MAX_DISTANCE or closing < .5 or distance / closing > CUT_IN_MAX_TTC:
      return None
    nearest = low if low > 0 else high if high < 0 else 0.0
    # Near-range body geometry can turn a shallow adjacent span into a very
    # close longitudinal obstacle. Require strictly more current penetration,
    # preserving the approach history above for immediate deeper entry.
    if distance < CUT_IN_NEAR_DISTANCE and abs(nearest) >= CUT_IN_HALF_WIDTH - CUT_IN_NEAR_MIN_INTRUSION:
      return None
    return _Path('inPath', nearest, rate, source=left.source + '+radarBody', lateral_motion_valid=True,
                 authority_reason=CUT_IN_REASON)

  @staticmethod
  def _relevant(memory, path, speed):
    return loss_relevant(distance=memory.x, ego_speed=speed, path_state=path.state)

  def _visual_fallback(self, memory, visions, eof, speed, yaw):
    previous = memory.last_vision
    if previous is None or not _fresh(previous['evidence']['estimateMonoTime'], eof):
      return None
    index = previous['evidence']['visionIndex']
    if not 0 <= index < len(visions) or visions[index].get('present') is not True or visions[index]['modelProb'] < .8:
      return None
    current = visions[index]
    elapsed = (eof - previous['evidence']['estimateMonoTime']) / 1e9
    x, y, vx, _ = _project(previous['dRel'], previous['yRel'], previous['vLeadK'], memory.vy, elapsed, speed, yaw)
    if abs(current['dRel'] - x) > 1.5 or abs(current['yRel'] - y) > .75 or abs(current['vLeadK'] - vx) > 4.0:
      return None
    lead = dict(current)
    lead['evidence'] = dict(current['evidence'])
    # Fresh visual components carry forward an already confirmed physical
    # identity. Each subsequent camera frame must independently pass this
    # continuity check; no absent camera or changed cut-in can extend it.
    lead['evidence'].update(source='fused', sessionId=self._session, trackId=memory.track_id,
                            generation=memory.generation, reason='visionFallback')
    lead['radar'], lead['radarTrackId'] = False, memory.track_id if memory.track_id <= 2**31 - 1 else -1
    memory.last_vision = current
    memory.relevant = current['dRel'] <= max(15.0, 2.0 * speed + speed * speed / 6.0)
    return lead

  @staticmethod
  def _vision(vision_leads, model, model_ns, eof):
    result = []
    raw = tuple(vision_leads)
    hypotheses = _read(model, 'leadsV3', ())
    for index in range(2):
      value = raw[index] if index < len(raw) else {'present': False}
      lead = dict(value) if isinstance(value, dict) else value.to_dict()
      if lead.get('present') is True:
        hypothesis = hypotheses[index] if hypotheses is not None and index < len(hypotheses) else None
        probability = _read(hypothesis, 'prob')
        fields = ('dRel', 'yRel', 'vRel', 'vLead', 'vLeadK', 'aLeadK', 'aLeadTau')
        if not _finite(*(lead.get(key) for key in fields), probability) or not .5 < probability <= 1:
          lead = {'present': False}
        else:
          # Recheck current membership and qualification at the adapter
          # boundary; an old/filtered modelProb is not current detection.
          lead['modelProb'] = probability
          sx, sv = _first(hypothesis, 'xStd', 2.0), _first(hypothesis, 'vStd', 2.0)
          sx = max(.5, min(20.0, float(sx))) if _finite(sx) and sx >= 0 else 20.0
          sv = max(.5, min(20.0, float(sv))) if _finite(sv) and sv >= 0 else 20.0
          strong = lead['modelProb'] >= .8
          lead['evidence'] = {
            'version': VERSION, 'source': 'vision', 'sessionId': '', 'trackId': 0, 'generation': 0,
            'positionMonoTime': eof, 'velocityMonoTime': eof, 'estimateMonoTime': eof, 'modelMonoTime': model_ns,
            'positionStd': sx, 'velocityStd': sv, 'positionValid': True, 'velocityValid': True, 'accelerationValid': True,
            'followQualified': True, 'brakeQualified': True, 'stopQualified': strong,
            'followGeometryMonoTime': 0,
            'departureQualified': strong and lead['vLeadK'] > .5, 'fcwQualified': lead['modelProb'] > .9,
            'cautionQualified': False,
            'physicalConflictValid': False, 'physicalConflictStart': 0.0, 'physicalConflictEnd': 0.0,
            'coasting': False, 'pathState': 'inPath', 'pathOffset': 0.0, 'conflictStart': 0.0, 'conflictEnd': HORIZON,
            'conflictSupportVersion': 0, 'conflictSupportedEnd': 0.0,
            'lateralSpeed': 0.0, 'pathSource': 'visionLead', 'visionIndex': index, 'reason': 'vision',
          }
      result.append(lead)
    return result

  @staticmethod
  def _identity_visions(visions, model):
    # Borrowing motion components or continuing a physical identity needs a
    # stronger current raw hypothesis than ordinary visual qualification.
    # Keep the original objects unchanged.
    hypotheses = _read(model, 'leadsV3', ())
    result = []
    for index, vision in enumerate(visions):
      probability = _read(hypotheses[index], 'prob') if hypotheses is not None and index < len(hypotheses) else None
      qualified = _finite(probability) and .8 <= probability <= 1
      result.append(vision if qualified else {'present': False})
    return result

  @staticmethod
  def _binding(track_id, bindings, visions, x, y, source_eof):
    for binding in bindings:
      index = binding.get('modelIndex')
      if (not _integer(index) or not 0 <= index < len(visions) or visions[index].get('present') is not True
          or not _finite(visions[index].get('modelProb')) or not .8 <= visions[index]['modelProb'] <= 1):
        continue
      start, count = binding.get('bindingStartMonoTime'), binding.get('bindingSamples')
      bx, by = binding.get('bindingAnchorX'), binding.get('bindingAnchorY')
      if (binding.get('trackId') != track_id or binding.get('candidateTrackId') != track_id
          or binding.get('identityState') != 'same' or binding.get('rejectionMonoTime') != 0
          or binding.get('present') is not True or not _finite(binding.get('probability'), bx, by)
          or binding['probability'] < .8 or not _integer(count) or count < MIN_OBSERVATIONS
          or not _integer(start) or not 0 < start <= source_eof - MIN_SPAN_NS):
        continue
      vision = visions[index]
      # The observer's established bias is an identity check, never an average
      # of two vehicles. Recheck the *current* vision frame after time alignment.
      if abs(x - vision['dRel'] - bx) <= 1.0 and abs(y - vision['yRel'] - by) <= .65:
        return index
    return -1

  @staticmethod
  def _binding_contradicted(memory, bindings, source_model):
    if memory.last_vision is None:
      return False
    index = memory.last_vision['evidence']['visionIndex']
    for binding in bindings:
      rejection_ns = binding.get('rejectionMonoTime')
      if (not _integer(binding.get('modelIndex')) or binding['modelIndex'] != index
          or not _integer(binding.get('candidateTrackId')) or binding['candidateTrackId'] != memory.track_id
          or binding.get('present') is not True or not _finite(binding.get('probability'))
          or not .8 <= binding['probability'] <= 1
          or not _integer(rejection_ns) or not 0 < rejection_ns <= source_model):
        continue
      # The observer also stamps soft suspensions such as a source gap or
      # missing candidate. Only an explicit identity contradiction revokes a
      # proven visual stream; absent input alone cannot erase its continuity.
      if binding.get('identityState') == 'different' or binding.get('identityReason') in ('visionJump', 'residualSeparation'):
        return True
    return False

  @staticmethod
  def _follow_geometry(memory, path):
    # Current crossing/opposing motion does not become an ordinary lead just
    # because its reflection is currently inside the path. Unqualified lateral
    # speed cannot by itself erase an otherwise qualified longitudinal lead.
    return (path.authority_reason != CUT_IN_REASON and path.state == 'inPath' and memory.vx >= -2.0
            and (abs(path.rate) < .5 or not path.lateral_motion_valid or path.source == 'visionLead'))

  def _make_lead(self, memory, path, eof, model_ns, speed, *, vision=None, vision_index=-1, reason='radar'):
    probability = float(vision['modelProb']) if vision is not None else 0.0
    # Near-zero negative ground speed remains a measured quantity. Consumers
    # may bound their motion model, but it must not erase this physical object.
    # Reuse the observer's -2 m/s opposing-motion boundary. Small negative
    # estimates of a stopped target retain stopping authority, without ever
    # treating a negative velocity as departure evidence.
    following = self._follow_geometry(memory, path)
    cut_in = path.authority_reason == CUT_IN_REASON
    if path.authority_reason:
      reason += '+' + path.authority_reason
    acceleration_valid = memory.acceleration_good and not cut_in
    return {
      'dRel': float(memory.x), 'yRel': float(memory.y), 'vRel': float(memory.vx - speed),
      'vLead': float(memory.vx), 'vLeadK': float(memory.vx), 'aLeadK': 0.0 if cut_in else float(memory.acceleration),
      'aLeadTau': 1.5 if acceleration_valid else .3, 'modelProb': probability, 'present': True, 'radar': True,
      'radarTrackId': memory.track_id if memory.track_id <= 2**31 - 1 else -1,
      'evidence': {
        'version': VERSION, 'source': 'fused' if vision is not None else 'radar',
        'sessionId': self._session, 'trackId': memory.track_id, 'generation': memory.generation,
        'positionMonoTime': memory.position_ns, 'velocityMonoTime': memory.velocity_ns,
        'estimateMonoTime': eof, 'modelMonoTime': model_ns, 'positionStd': math.sqrt(max(.01, memory.position_variance)),
        'velocityStd': math.sqrt(max(.01, memory.velocity_variance)), 'positionValid': True, 'velocityValid': True,
        'accelerationValid': acceleration_valid, 'followQualified': following,
        'followGeometryMonoTime': eof if following else 0,
        'brakeQualified': following or cut_in, 'stopQualified': following,
        'departureQualified': following and memory.vx > RADAR_DEPARTURE_MIN_SPEED,
        'fcwQualified': following and memory.position_variance <= 2.25 and memory.velocity_variance <= 6.25,
        'cautionQualified': False,
        'physicalConflictValid': False, 'physicalConflictStart': 0.0, 'physicalConflictEnd': 0.0,
        'coasting': False, 'pathState': path.state, 'pathOffset': path.offset,
        # Retired occupancy fields retain neutral values for old log readers.
        'conflictStart': 0.0, 'conflictEnd': HORIZON, 'lateralSpeed': path.rate,
        'conflictSupportVersion': 0, 'conflictSupportedEnd': 0.0,
        'pathSource': path.source, 'visionIndex': vision_index, 'reason': reason,
      },
    }

  @staticmethod
  def _select(candidates):
    qualified = []
    for lead in candidates:
      claim = lead['evidence']
      if claim['pathState'] != 'inPath' or not claim['brakeQualified'] or claim['coasting']:
        continue
      if claim['source'] == 'vision' and any(
          other['evidence']['source'] == 'vision'
          and all(abs(lead[k] - other[k]) < .01 for k in ('dRel', 'yRel', 'vLeadK', 'aLeadK')) for other in qualified):
        continue
      qualified.append(lead)
    qualified.sort(key=lambda lead: lead['dRel'])
    leads = (qualified + [{'present': False}, {'present': False}])[:2]
    return (leads[0], leads[1]), []

  def update(self, *, now_ns, enabled, observation, observation_ns, observation_valid, model, model_ns, v_ego, vision_leads):
    raw_visions = tuple(dict(v) if isinstance(v, dict) else v.to_dict() for v in vision_leads)
    raw_visions = (raw_visions + ({'present': False}, {'present': False}))[:2]
    if enabled is not True:
      self.reset()
      return RadarControlResult(raw_visions, [], False, False, 'disabled')
    eof = _read(model, 'timestampEof', 0)
    yaw = _first(_read(model, 'orientationRate'), 'z')
    if (not _integer(now_ns) or now_ns <= self._last_now_ns or not _finite(v_ego, yaw) or not -.5 <= v_ego <= 70
        or abs(yaw) > 2 or not _fresh(model_ns, now_ns) or not _fresh(eof, now_ns)
        or eof > model_ns or model_ns < self._model_ns or eof < self._eof):
      self.reset()
      return RadarControlResult(raw_visions, [], False, True, 'coreInputInvalid')
    self._last_now_ns, self._model_ns, self._eof = now_ns, model_ns, eof
    speed = max(0.0, float(v_ego))
    visions = self._vision(raw_visions, model, model_ns, eof)
    identity_visions = self._identity_visions(visions, model)
    previous_tracks = dict(self._tracks)
    previous_positions = ({row[0]: (row[4], row[5], row[6]) for row in self._publication[-1][0]
                           if all(_integer(row[index]) for index in (0, 4, 5))}
                          if self._publication is not None else {})
    corridors = _corridors(model)
    fusion = _read(observation, 'fusion')
    reason, healthy, fault = 'active', True, False
    source_eof = _read(fusion, 'modelTimestampEof', 0)
    source_model = _read(fusion, 'modelMonoTime', 0)
    tracks, bindings = [], []
    if observation_valid is not True or not _fresh(observation_ns, now_ns) or observation is None:
      reason, healthy = 'observationStale', False
    elif (_read(observation, 'schemaVersion') != 1 or _read(observation, 'active') is not True
          or _read(observation, 'reason') != 'observing' or _read(fusion, 'version') != 1
          or _read(fusion, 'algorithmVersion') != FUSION_VERSION or _read(fusion, 'active') is not True
          or _read(fusion, 'enabled') is not True or not _fresh(source_eof, now_ns) or not _fresh(source_model, now_ns)
          or not source_eof <= source_model <= observation_ns or source_eof > eof
          or not _fresh(_read(fusion, 'egoMonoTime', 0), now_ns) or _read(fusion, 'egoMonoTime', 0) > observation_ns):
      reason, healthy = 'observationInvalid', False
    else:
      source_tracks, source_bindings = _read(fusion, 'tracks', ()), _read(fusion, 'leads', ())
      if source_tracks is None or source_bindings is None or len(source_tracks) > MAX_TRACKS or len(source_bindings) > 3:
        reason, healthy, fault = 'inputShape', False, True
      else:
        tracks = [{key: _read(t, key) for key in _TRACK_FIELDS} for t in source_tracks]
        for track in tracks:
          track['nativeMotion'] = _native_motion_snapshot(track['nativeMotion'])
        bindings = [{key: _read(b, key) for key in _BINDING_FIELDS} for b in source_bindings]
        session, sequence = _read(observation, 'sessionId'), _read(observation, 'sequence')
        counters = (_read(_read(observation, 'counters'), 'dropped', 0), _read(_read(fusion, 'counters'), 'resets', 0))
        signature = tuple(tuple(t.values()) for t in tracks), tuple(tuple(b.values()) for b in bindings)
        current = session, sequence, observation_ns, source_model, source_eof, counters, signature
        if not isinstance(session, str) or not 1 <= len(session) <= 64 or not _integer(sequence) or sequence <= 0:
          reason, healthy, fault = 'publicationIdentity', False, True
        elif self._publication is not None:
          old_session, old_seq, old_ns, old_model, old_eof, old_counters, _ = self._publication
          if session != old_session or counters != old_counters:
            self._tracks.clear()
            self._cleared.clear()
            reason, healthy, fault = 'observerReset', False, True
          elif (sequence < old_seq or observation_ns < old_ns or source_model < old_model or source_eof < old_eof
                or sequence == old_seq and current != self._publication
                or sequence > old_seq and observation_ns <= old_ns):
            reason, healthy, fault = 'publicationOrder', False, True
        if _integer(sequence) and isinstance(session, str):
          # Do not accept a conflicting same-sequence body as a new baseline.
          if reason != 'publicationOrder':
            self._publication = current
            self._session = session
        if len({t['trackId'] for t in tracks if _integer(t['trackId'])}) != len(tracks):
          reason, healthy, fault = 'trackIdentity', False, True
    if fault:
      self._tracks.clear()
      self._cleared.clear()
      tracks = []
    if not healthy:
      tracks = []
    else:
      # Revoke before processing active tracks: otherwise their new memory can
      # retain the old visual identity and revive it on a later observer gap.
      # The only way to establish it again is a current qualified _binding.
      for memory in self._tracks.values():
        if self._binding_contradicted(memory, bindings, source_model):
          memory.last_vision = None
    radar_objects, used_visions, cleared = [], set(), set()
    failed_leads = set()
    previous_body_approaches, self._body_approaches = self._body_approaches, {}
    for track in tracks:
      identifier, oem, generation = (track[k] for k in ('trackId', 'oemTrackId', 'generation'))
      if (not all(_integer(value) for value in (identifier, oem, generation)) or identifier <= 0
          or not 1 <= oem <= 63 or generation < 0):
        continue
      previous = self._tracks.get(identifier)
      p_ns, p_seq, v_ns = (track[k] for k in ('positionMonoTime', 'positionSeq', 'velocityMonoTime'))
      x, y, vx, vy, p_var, v_var = (track[k] for k in
                                  ('dRel', 'yRel', 'groundSpeed', 'lateralSpeed', 'positionVariance', 'velocityVariance'))
      lateral_position_variance, lateral_velocity_variance = track['lateralPositionVariance'], track['lateralVelocityVariance']
      lateral_velocity_valid = track['lateralVelocityValid'] is True
      invalid_position = (not _integer(p_seq) or p_seq <= 0 or not _integer(p_ns)
                          or not _finite(x, y, p_var) or not 0 < x <= 240 or abs(y) > 100
                          or not 0 <= p_var <= POSITION_MAX_VARIANCE or p_ns > observation_ns)
      bad_order = previous is not None and (oem != previous.oem_id or p_seq < previous.position_seq or p_ns < previous.position_ns
        or p_seq == previous.position_seq and p_ns != previous.position_ns
        or p_seq > previous.position_seq and p_ns <= previous.position_ns)
      if invalid_position or bad_order:
        self._tracks.pop(identifier, None)
        if previous is not None and previous.selected_follow and previous.relevant:
          failed_leads.add(identifier)
        continue
      if track['active'] is not True or track['coasting'] is True or not _fresh(p_ns, now_ns):
        # Normal observation loss withdraws this radar candidate immediately.
        # Explicit corruption still reports a fault; current visual candidates
        # remain available and no old radar position is republished.
        if (track['active'] is not True and track['lastInvalidReason'] not in ('expired', 'slotReassigned')
            and previous is not None and previous.selected_follow and previous.relevant):
          failed_leads.add(identifier)
        continue
      if track['mature'] is not True or not _integer(track['observations']) or track['observations'] < MIN_OBSERVATIONS:
        continue
      if (not _finite(vx, vy, v_var) or not -35 <= vx <= 80 or abs(vy) > 30 or v_var < 0):
        self._tracks.pop(identifier, None)
        if previous is not None and previous.selected_follow and previous.relevant:
          failed_leads.add(identifier)
        continue
      native_state = _native_motion_state(track['nativeMotion'], track, now_ns=now_ns, observation_ns=observation_ns)
      if native_state != 'qualified':
        self._tracks.pop(identifier, None)
        if native_state == 'invalid' and previous is not None and previous.selected_follow and previous.relevant:
          failed_leads.add(identifier)
        # A split scan waiting for its own AUX must not erase the independent
        # geometric approach history. It grants no radar authority meanwhile.
        quality = dict(zip(_NATIVE_MOTION_FIELDS, track['nativeMotion'], strict=True)) if track['nativeMotion'] else {}
        if (quality.get('reason') == 'awaitingAux' and identifier in previous_body_approaches
            and _fresh(previous_body_approaches[identifier].position_ns, p_ns, 100_000_000)):
          self._body_approaches[identifier] = previous_body_approaches[identifier]
        continue
      velocity_components = _finite(vx, vy, v_var) and -35 <= vx <= 80 and abs(vy) <= 30 and v_var >= 0
      velocity_values = velocity_components and v_var <= VELOCITY_MAX_VARIANCE
      direct_velocity = (velocity_values and track['velocityGood'] is True and _fresh(v_ns, now_ns)
                         and v_ns <= observation_ns and abs(v_ns - p_ns) <= NATIVE_MAX_PAIR_NS
                         and (previous is None or v_ns >= previous.raw_velocity_ns))
      if not direct_velocity:
        self._tracks.pop(identifier, None)
        if (not velocity_values and previous is not None and previous.selected_follow and previous.relevant):
          failed_leads.add(identifier)
        continue
      projection_dt = (eof - source_eof) / 1e9
      px, py, _, pvy = _project(x, y, vx, vy, projection_dt, speed, yaw)
      # Longitudinal speed has its own native source. Position-derived lateral
      # speed may align geometry, but cannot rotate back into that scalar.
      pvx = vx
      vision_index = self._binding(identifier, bindings, identity_visions, px, py, source_eof)
      vision = visions[vision_index] if vision_index >= 0 else None
      component_reason = 'radar'
      if not _finite(px, py, pvx, pvy, v_var) or px <= 0 or v_var < 0:
        if previous is not None and previous.selected_follow and previous.relevant:
          failed_leads.add(identifier)
        continue
      lateral_position_variance, lateral_velocity_variance = _lateral_variances(
        p_var, lateral_position_variance, v_var, lateral_velocity_variance, projection_dt, yaw)
      if previous is not None and 0 <= eof - previous.epoch_ns <= MAX_AGE_NS:
        expected = _project(previous.x, previous.y, previous.vx, previous.vy, (eof - previous.epoch_ns) / 1e9, speed, yaw)
        if abs(px - expected[0]) > max(3.0, 3 * math.sqrt(p_var + previous.position_variance)) or abs(py - expected[1]) > 2.0:
          self._tracks.pop(identifier, None)
          if previous.selected_follow and previous.relevant:
            failed_leads.add(identifier)
          continue
      path = _object_path(corridors, px, py, pvx, pvy, lateral_position_variance, lateral_velocity_variance,
                          lateral_velocity_valid, vision=vision is not None)
      body_path = self._cut_in_path(track, corridors, source_eof=source_eof, eof=eof, now_ns=now_ns,
                                   observation_ns=observation_ns, speed=speed, yaw=yaw, vx=pvx, vy=pvy,
                                   position_variance=lateral_position_variance, velocity_variance=lateral_velocity_variance,
                                   direct_velocity=direct_velocity, previous=previous_body_approaches.get(identifier))
      # Keep ordinary point/visual following independent. A body-only cut-in
      # does not acquire ordinary authority from an earlier point/visual lead.
      if body_path is not None and (path.state != 'inPath'
                                    or path.lateral_motion_valid and abs(path.rate) >= .5 and path.source != 'visionLead'):
        path = body_path
      old_position = previous_positions.get(identifier)
      # A position first seen while its AUX was pending becomes a newly
      # complete native pair when that same scan's real velocity arrives.
      # Merely changing a quality flag with an already present velocity does
      # not establish new occupancy or count another source measurement.
      completed_pair = old_position == (p_seq, p_ns, 0) and v_ns > 0
      independent = ((old_position is None or p_seq > old_position[0] and p_ns > old_position[1] or completed_pair)
                     if previous is None else p_ns > previous.position_ns and p_seq > previous.position_seq)
      occupied_before = (previous is not None and previous.path.state == 'inPath' and previous.accepted is not None
                         and previous.accepted['evidence']['brakeQualified']
                         and _fresh(previous.position_ns, now_ns) and _fresh(previous.path_ns, eof))
      if (path.state == 'inPath' and not occupied_before
          and vision is None and (not independent or p_ns <= self._cleared.get(identifier, 0))):
        # Reusing a position cannot establish first occupancy. A confirmed
        # current visual lead can supply current body geometry.
        continue
      direct_path = path.state != 'unknown' or path.observed_out
      if not direct_path:
        if previous is not None and previous.selected_follow and previous.relevant:
          failed_leads.add(identifier)
        continue
      if path.state == 'outOfPath' or path.observed_out:
        known = (previous is not None and previous.accepted is not None and previous.accepted['evidence']['brakeQualified']
                 or identifier in self._cleared)
        # The planner may still hold this physical identity after its radar
        # candidate disappears. A current measured OUT can clear it without a
        # retained loss record. Model geometry alone cannot refresh a known
        # position, and an outer-band point does not prove body clearance.
        measured_out = (direct_path and path.observed_out and independent
                        and p_ns >= self._cleared.get(identifier, 0))
        if measured_out:
          self._cleared[identifier] = p_ns
          cleared.add(identifier)
          self._tracks.pop(identifier, None)
        elif known and path.observed_out:
          # A changed plan geometry may withdraw a candidate immediately, but
          # a fresh independent radar position must corroborate a HOLD release.
          if previous is not None and previous.selected_follow and previous.relevant:
            failed_leads.add(identifier)
        # Healthy current outer-band geometry simply withdraws the ordinary
        # role. It is not a measurement failure and cannot create degraded
        # braking or a clearance receipt for the old target.
        continue
      acceleration = track['acceleration']
      acceleration_good = direct_velocity and track['accelerationGood'] is True and _finite(acceleration) and -10 <= acceleration <= 5
      if not acceleration_good:
        if vision is not None:
          acceleration, acceleration_good = vision['aLeadK'], True
          component_reason += '+visionAcceleration'
        else:
          # No zero speed or zero distance substitution is ever made. Zero
          # acceleration here is an explicit constant-velocity prediction, and
          # is never marked as an observed/qualified acceleration component.
          acceleration = 0.0
          component_reason += '+constantVelocity'
      memory = _Memory(identifier, oem, generation, p_seq, p_ns, v_ns, eof, px, py, pvx, pvy,
                       float(acceleration), acceleration_good, float(p_var), float(v_var),
                       path, eof,
                       selected_follow=previous.selected_follow if previous else False,
                       raw_velocity_ns=track['velocityMonoTime'] if direct_velocity else previous.raw_velocity_ns if previous else 0,
                       last_vision=vision if vision is not None else previous.last_vision if previous else None,
                       lateral_position_variance=lateral_position_variance, lateral_velocity_variance=lateral_velocity_variance,
                       lateral_velocity_valid=lateral_velocity_valid)
      memory.relevant = self._relevant(memory, path, speed)
      if self._follow_geometry(memory, path) or path.authority_reason == CUT_IN_REASON:
        self._cleared.pop(identifier, None)
        lead = self._make_lead(memory, path, eof, model_ns, speed, vision=vision,
                               vision_index=vision_index, reason=component_reason)
        memory.accepted = lead
        radar_objects.append(lead)
        if vision_index >= 0:
          used_visions.add(vision_index)
          # Model's future hypotheses matched to this same track are not new
          # physical obstacles and must not dilute the radar replacement.
          for other in range(len(visions)):
            candidate = self._binding(identifier, [b for b in bindings if b['modelIndex'] == other], identity_visions, px, py, source_eof)
            if candidate >= 0:
              used_visions.add(candidate)
      else:
        # Crossing/opposing/unqualified geometry has no ordinary control role.
        self._tracks.pop(identifier, None)
        if path.state == 'unknown' and previous is not None and previous.selected_follow and previous.relevant:
          failed_leads.add(identifier)
        continue
      self._tracks[identifier] = memory
    visual_recoveries = set()
    if not fault:
      # Current visual components may retain an already proven same-car
      # identity. They never reuse missing radar motion components.
      for identifier, memory in previous_tracks.items():
        if not memory.selected_follow:
          continue
        active = next((lead for lead in radar_objects if lead['evidence']['trackId'] == identifier), None)
        if identifier in cleared or active is not None:
          continue
        # A current soft rejection still blocks replacing the radar obstacle
        # on this frame. Only explicit identity contradictions clear the saved
        # visual stream above; an absent optional observer keeps its old rules.
        rejected = any(b['candidateTrackId'] == identifier and
                       (b['identityState'] == 'different' or b['rejectionMonoTime']) for b in bindings)
        if rejected:
          continue
        fallback = self._visual_fallback(memory, identity_visions, eof, speed, yaw)
        if fallback is not None:
          radar_objects = [lead for lead in radar_objects if lead['evidence']['trackId'] != identifier]
          radar_objects.append(fallback)
          used_visions.add(fallback['evidence']['visionIndex'])
          visual_recoveries.add(identifier)
          self._tracks[identifier] = memory
    remaining = [lead for index, lead in enumerate(visions) if lead.get('present') is True and index not in used_visions]
    leads, objects = self._select(remaining + radar_objects)
    selected = {lead['evidence']['trackId'] for lead in leads if lead.get('present') is True}
    # Only selected formal leads retain control memory. Body approach samples
    # remain separate and confer no following, stop-binding or loss authority.
    self._tracks = {identifier: memory for identifier, memory in self._tracks.items() if identifier in selected}
    for memory in self._tracks.values():
      memory.selected_follow = memory.accepted is not None and memory.accepted['evidence']['followQualified']
    degraded = not healthy or bool(failed_leads - visual_recoveries)
    # Preserve measured OUT timestamps across packet gaps for their existing
    # freshness window, so the same sample cannot gain IN authority by reuse.
    self._cleared = {key: stamp for key, stamp in self._cleared.items() if _fresh(stamp, now_ns)}
    # Current same-car visual measurements can replace a missing radar
    # component while other relevant current objects remain checked.
    if visual_recoveries and not fault:
      unresolved = any(memory.relevant and identifier not in visual_recoveries
                       and not any(lead['evidence']['trackId'] == identifier for lead in radar_objects)
                       for identifier, memory in previous_tracks.items())
      if not unresolved:
        degraded = False
    if reason == 'active' and degraded:
      reason = 'objectDegraded'
    elif reason == 'active' and not any(lead.get('present') is True for lead in leads):
      reason = 'noQualifiedObjects'
    return RadarControlResult(leads, objects, healthy, degraded, reason, tuple(sorted(cleared)), self._session)
