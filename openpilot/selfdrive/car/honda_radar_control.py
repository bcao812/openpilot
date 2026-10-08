"""Admit Accord radar objects into the ordinary longitudinal object interface.

The observer owns decoding, motion compensation and physical track identity.
This consumer checks its evidence, aligns estimates to the current camera frame,
and assigns separate following, braking, stopping and departure qualifications.
It does not publish messages, read settings or own actuation. In particular, a
missing visual detection is not evidence that a radar obstacle does not exist.

Bounds below are engineering admission parameters, not sensor calibration. The
two collected drives contain no observed in-path metal-cover/sign false targets;
this version therefore makes no speculative stationary-object class rejection.
"""
from dataclasses import dataclass, field, replace
import math
from numbers import Real

from openpilot.selfdrive.car.honda_radar_relevance import loss_relevant, remaining_occupancy_window


VERSION = 1
FUSION_VERSION = 'bosch-a-shadow-3'
MAX_TRACKS = 32
MAX_OBJECTS = MAX_TRACKS + 2
MAX_AGE_NS = 200_000_000
COAST_NS = 350_000_000
MIN_OBSERVATIONS = 3
MIN_SPAN_NS = 120_000_000
HORIZON = 10.0
RADAR_TO_CAMERA = 1.52
# A single radar reflection is not proven to be a vehicle's body center.
# Keep possible body overlap as a candidate, but grant new hard-constraint
# authority only for the measured point entering the ego's swept footprint.
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
LATERAL_PREDICTION_BUDGET = 2.0


def _read(obj, name, default=None):
  return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _finite(*values):
  return all(isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(v) for v in values)


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
  risk_unresolved: bool = False


@dataclass(frozen=True)
class _Path:
  state: str = 'unknown'
  offset: float = 0.0
  rate: float = 0.0
  start: float = 0.0
  end: float = 0.0
  source: str = 'unavailable'
  continuous: bool = True
  prediction_limited: bool = False
  observed_out: bool = False
  confirmed: bool = False
  authority_reason: str = ''
  physical_valid: bool = False
  physical_start: float = 0.0
  physical_end: float = 0.0


def _without_physical_conflict(path):
  return replace(path, physical_valid=False, physical_start=0.0, physical_end=0.0)


def _merge_raw_intervals(intervals):
  """Merge only actual overlap/touching; timing margins cannot bridge a gap."""
  merged = []
  for start, end in sorted(intervals):
    if merged and start <= merged[-1][1]:
      merged[-1] = merged[-1][0], max(merged[-1][1], end)
    else:
      merged.append((start, end))
  return merged


def _clip_band(interval, position, velocity, lower, upper):
  """Intersect a time interval with lower <= position + velocity*t <= upper."""
  if interval is None:
    return None
  if abs(velocity) < 1e-9:
    return interval if lower <= position <= upper else None
  a, b = (lower - position) / velocity, (upper - position) / velocity
  start, end = max(interval[0], min(a, b)), min(interval[1], max(a, b))
  return (start, end) if start <= end else None


def _disk_interval(interval, x, y, vx, vy, half_width):
  """Times a constant-velocity point occupies one finite polyline corner."""
  a = vx * vx + vy * vy
  b = x * vx + y * vy
  c = x * x + y * y - half_width**2
  if a < 1e-12:
    return interval if c <= 0 else None
  discriminant = b * b - a * c
  if discriminant < 0:
    return None
  root = math.sqrt(discriminant)
  start, end = max(interval[0], (-b - root) / a), min(interval[1], (-b + root) / a)
  return (start, end) if start <= end else None


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

  def relation(self, x, y, vx, vy, horizon=HORIZON, half_width=OVERLAP_HALF_WIDTH):
    if not self.points:
      return _Path()
    # Test the entire finite polyline, rather than extending the currently
    # nearest segment's tangent. A target presently beside a straight section
    # can occupy a later bend even when its present lateral rate is zero.
    # Restrict prediction to the path's actual longitudinal spatial support.
    support = _clip_band((0.0, horizon), x, vx, self.points[0][0], self.points[-1][0])
    if support is None:
      return _Path()
    current_supported = self.points[0][0] <= x <= self.points[-1][0]
    best, intervals = None, []
    for px, py, dx, dy, length in self._segments:
      along = (x - px) * dx + (y - py) * dy
      along_rate = vx * dx + vy * dy
      offset = (y - py) * dx - (x - px) * dy
      rate = vy * dx - vx * dy
      nearest = min(length, max(0.0, along))
      distance = math.hypot(x - px - nearest * dx, y - py - nearest * dy)
      if best is None or distance < abs(best[0]):
        best = math.copysign(distance, offset), rate
      if horizon == 0:
        continue
      interval = _clip_band(support, along, along_rate, 0.0, length)
      interval = _clip_band(interval, offset, rate, -half_width, half_width)
      if interval is not None:
        intervals.append(interval)
    if best is None:
      return _Path()
    offset, rate = best
    inside = current_supported and abs(offset) <= half_width
    if horizon == 0:
      return _Path('inPath' if inside else 'outOfPath', offset, rate, source=self.source)
    # Round joins and finite ends inside the actual longitudinal support,
    # without inventing another segment beyond either path endpoint.
    for px, py in self.points:
      interval = _disk_interval(support, x - px, y - py, vx, vy, half_width)
      if interval is not None:
        intervals.append(interval)
    if not intervals:
      return _Path('outOfPath', offset, rate, source=self.source) if current_supported else _Path()
    # Save the original geometric union before timing padding and merging.
    # Reversing padding later is not equivalent: clipping at t=0 loses the
    # original entry, and overlapping margins can conceal a real clear gap.
    physical_intervals = _merge_raw_intervals(intervals)
    # At most 65 analytic intervals (32 strips plus 33 vertices). Merge their
    # short timing margins; a single message window uses the conservative
    # envelope of any remaining disjoint intervals. A gap in that envelope
    # cannot promote a crossing object into an enduring follow/stop lead.
    intervals = sorted((max(support[0], start - .15), min(support[1], end + .25)) for start, end in intervals)
    merged = [intervals[0]]
    for start, end in intervals[1:]:
      if start <= merged[-1][1]:
        merged[-1] = merged[-1][0], max(merged[-1][1], end)
      else:
        merged.append((start, end))
    start, end = merged[0][0], merged[-1][1]
    spatial_limit = support[1] < horizon and end >= support[1] - 1e-6
    physical_valid = (half_width == SWEPT_HALF_WIDTH and not spatial_limit and len(physical_intervals) == 1
                      and physical_intervals[0][0] < physical_intervals[0][1])
    physical_start, physical_end = physical_intervals[0] if physical_valid else (0.0, 0.0)
    return _Path('inPath' if inside else 'entering', offset, rate, start, end, self.source, len(merged) == 1, spatial_limit,
                 physical_valid=physical_valid, physical_start=physical_start, physical_end=physical_end)


def _qualified_path(corridors, x, y, vx, vy, position_variance, velocity_variance, velocity_valid, half_width=SWEPT_HALF_WIDTH):
  """Separate measured current occupancy from qualified lateral prediction.

  These covariance limits and the displacement budget are engineering bounds,
  not calibrated sensor probabilities. Unsupported lateral extrapolation cannot
  erase an observed in-path obstacle or authorize its departure. Its measured
  longitudinal motion still retains the ordinary following/stopping authority.
  """
  if not _finite(position_variance) or not 0 <= position_variance <= LATERAL_POSITION_MAX_VARIANCE:
    return _Path()
  horizon = 0.0
  if velocity_valid is True and _finite(velocity_variance) and 0 <= velocity_variance <= LATERAL_VELOCITY_MAX_VARIANCE:
    budget = max(0.0, LATERAL_PREDICTION_BUDGET - math.sqrt(position_variance))
    horizon = min(HORIZON, budget / max(1e-9, math.sqrt(velocity_variance)))
  for corridor in corridors:
    prediction = corridor.relation(x, y, vx, vy, horizon, half_width)
    if prediction.state == 'inPath':
      if prediction.end < horizon - 1e-6 and not prediction.prediction_limited:
        return prediction
      # Reaching the quality/spatial bound is not a predicted clearing event.
      # Keep a currently occupied object through the MPC horizon; a subsequent
      # measured OUT can clear it even when lateral velocity remains untrusted.
      limited = horizon < HORIZON or prediction.prediction_limited
      occupied = replace(prediction, start=0.0, end=HORIZON, prediction_limited=limited)
      return _without_physical_conflict(occupied) if limited else occupied
    if prediction.state == 'entering' and prediction.end > prediction.start:
      limited = prediction.prediction_limited or prediction.end >= horizon - 1e-6 and horizon < HORIZON
      entering = replace(prediction, end=HORIZON if limited else prediction.end, prediction_limited=limited)
      return _without_physical_conflict(entering) if limited else entering
    if prediction.state == 'outOfPath':
      # A mean trajectory that intersects only beyond its quality bound is an
      # unknown future candidate, not a qualified entry. Current geometry can
      # nevertheless prove an independently observed departure of an old lead.
      unbounded_quality = corridor.relation(x, y, vx, vy, half_width=half_width) if horizon < HORIZON else prediction
      if horizon < HORIZON and unbounded_quality.state == 'entering':
        return _Path('unknown', prediction.offset, prediction.rate, source=prediction.source,
                     prediction_limited=True, observed_out=True)
      return _Path('outOfPath', prediction.offset, prediction.rate, source=prediction.source, observed_out=True)
  return _Path()


def _object_path(corridors, x, y, vx, vy, position_variance, velocity_variance, velocity_valid, *, confirmed=False, vision=False):
  inner = _qualified_path(corridors, x, y, vx, vy, position_variance, velocity_variance, velocity_valid)
  if inner.state == 'inPath':
    return replace(inner, confirmed=True)
  if vision:
    # A currently confirmed same-car visual lead establishes body occupancy
    # independently of which body surface supplies the radar reflection. Its
    # radar d/v remain the motion components of this one physical object.
    return _Path('inPath', inner.offset, inner.rate, 0.0, HORIZON, 'visionLead', confirmed=True,
                 authority_reason='confirmedVisionPath')
  outer = _qualified_path(corridors, x, y, vx, vy, position_variance, velocity_variance, velocity_valid,
                           half_width=OVERLAP_HALF_WIDTH)
  if confirmed and outer.state == 'inPath':
    # Geometric hysteresis: enter at a measured 1.25 m sweep (or independently
    # confirmed visual body occupancy), leave after measured 2 m occupancy is
    # gone. Every extension still requires this same fresh track and valid
    # current geometry; losing vision alone is not an object disappearance.
    return _without_physical_conflict(replace(outer, confirmed=True, authority_reason='confirmedOccupancyHysteresis'))
  if inner.state == 'entering':
    # Publish its actual future INNER interval now, so the MPC can anticipate
    # entry. Outer current overlap must never invent inner occupancy at t=0.
    return inner
  if outer.state in ('inPath', 'entering'):
    return _without_physical_conflict(replace(outer, state='edge', confirmed=False, observed_out=inner.state == 'outOfPath',
                                              authority_reason='edgeInnerUnknown' if inner.state == 'unknown' else 'edgePossibleBodyOverlap'))
  return outer if confirmed else inner


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
  raw_velocity_ns: int = 0
  relevant: bool = False
  last_vision: dict | None = None
  lateral_position_variance: float = math.inf
  lateral_velocity_variance: float = math.inf
  lateral_velocity_valid: bool = False


_TRACK_FIELDS = ('trackId', 'oemTrackId', 'generation', 'slot', 'positionSeq', 'positionMonoTime', 'velocityMonoTime',
                 'dRel', 'yRel', 'groundSpeed', 'lateralSpeed', 'positionVariance', 'velocityVariance', 'observations',
                 'mature', 'velocityGood', 'coasting', 'active', 'lastInvalidReason', 'lastInvalidMonoTime',
                 'previousPositionSeq', 'previousPositionMonoTime', 'acceleration', 'accelerationGood',
                 'lateralPositionVariance', 'lateralVelocityVariance', 'lateralVelocityValid')
_BINDING_FIELDS = ('modelIndex', 'present', 'probability', 'trackId', 'candidateTrackId', 'identityState', 'rejectionMonoTime',
                   'bindingStartMonoTime', 'bindingSamples', 'bindingAnchorX', 'bindingAnchorY', 'identityReason', 'reason')


class HondaRadarControl:
  def __init__(self):
    self.reset()

  def reset(self):
    self._tracks: dict[int, _Memory] = {}
    self._publication = None
    self._last_now_ns = self._model_ns = self._eof = 0
    self._session = ''
    self._unresolved_loss = False
    self._lost_tracks: dict[tuple[str, int], _Memory] = {}
    self._cleared: dict[int, int] = {}

  def _lost(self, memory):
    if memory is not None and memory.accepted is not None and memory.relevant:
      key = self._session, memory.track_id
      if key in self._lost_tracks or len(self._lost_tracks) < MAX_TRACKS:
        self._lost_tracks[key] = memory
      else:
        self._unresolved_loss = True

  @staticmethod
  def _relevant(memory, path, speed):
    return loss_relevant(distance=memory.x, ground_speed=memory.vx, ego_speed=speed, path_state=path.state,
                         conflict_start=path.start, conflict_end=path.end, lateral_speed=path.rate, horizon=HORIZON)

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
        fields = ('dRel', 'yRel', 'vRel', 'vLead', 'vLeadK', 'aLeadK', 'aLeadTau', 'modelProb')
        if not _finite(*(lead.get(key) for key in fields)) or not 0 <= lead['modelProb'] <= 1:
          lead = {'present': False}
        else:
          hypothesis = hypotheses[index] if index < len(hypotheses) else None
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
            'lateralSpeed': 0.0, 'pathSource': 'visionLead', 'visionIndex': index, 'reason': 'vision',
          }
      result.append(lead)
    return result

  @staticmethod
  def _binding(track_id, bindings, visions, x, y, source_eof):
    for binding in bindings:
      index = binding.get('modelIndex')
      if not _integer(index) or not 0 <= index < len(visions) or visions[index].get('present') is not True:
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
  def _historical_path(memory, eof):
    if memory is None or not _fresh(memory.path_ns, eof, COAST_NS) or memory.path.state not in ('inPath', 'entering'):
      return _Path()
    dt = (eof - memory.path_ns) / 1e9
    path = memory.path
    start, end = remaining_occupancy_window(path.start, path.end, dt, horizon=HORIZON)
    if end <= start:
      return _Path('outOfPath', path.offset + path.rate * dt, path.rate, source='previousPath')
    offset = path.offset + path.rate * dt
    half_width = OVERLAP_HALF_WIDTH if path.confirmed else SWEPT_HALF_WIDTH
    state = 'inPath' if abs(offset) <= half_width else 'entering'
    return _without_physical_conflict(replace(path, state=state, offset=offset, start=start, end=end, source='previousPath'))

  def _make_lead(self, memory, path, eof, model_ns, speed, *, vision=None, vision_index=-1, predicted=False, reason='radar',
                 follow_geometry_ns=0):
    probability = float(vision['modelProb']) if vision is not None else 0.0
    # Near-zero negative ground speed remains a measured quantity. Consumers
    # may bound their motion model, but it must not erase this physical object.
    # Reuse the observer's -2 m/s opposing-motion boundary. Small negative
    # estimates of a stopped target retain stopping authority, without ever
    # treating a negative velocity as departure evidence.
    caution = path.state == 'edge'
    enduring = (path.state == 'inPath' and path.continuous and path.end >= HORIZON - 1e-6
                and (abs(path.rate) < .5 or path.prediction_limited or path.source == 'visionLead') and memory.vx >= -2.0)
    if path.prediction_limited:
      reason += '+lateralPredictionLimited'
    if not path.continuous:
      reason += '+disjointOccupancyEnvelope'
    if path.authority_reason:
      reason += '+' + path.authority_reason
    acceleration_valid = memory.acceleration_good and not predicted
    physical_valid = (path.physical_valid is True and not predicted and not caution and not path.prediction_limited
                      and memory.path_ns == eof and memory.epoch_ns == eof
                      and _finite(path.physical_start, path.physical_end)
                      and 0.0 <= path.physical_start < path.physical_end <= HORIZON)
    return {
      'dRel': float(memory.x), 'yRel': float(memory.y), 'vRel': float(memory.vx - speed),
      'vLead': float(memory.vx), 'vLeadK': float(memory.vx), 'aLeadK': 0.0 if predicted else float(memory.acceleration),
      'aLeadTau': 1.5 if acceleration_valid else .3, 'modelProb': probability, 'present': True, 'radar': True,
      'radarTrackId': memory.track_id if memory.track_id <= 2**31 - 1 else -1,
      'evidence': {
        'version': VERSION, 'source': 'predicted' if predicted else 'fused' if vision is not None else 'radar',
        'sessionId': self._session, 'trackId': memory.track_id, 'generation': memory.generation,
        'positionMonoTime': memory.position_ns, 'velocityMonoTime': memory.velocity_ns,
        'estimateMonoTime': eof, 'modelMonoTime': model_ns, 'positionStd': math.sqrt(max(.01, memory.position_variance)),
        'velocityStd': math.sqrt(max(.01, memory.velocity_variance)), 'positionValid': True, 'velocityValid': True,
        'accelerationValid': acceleration_valid, 'followQualified': enduring and not predicted,
        # The old epoch describes geometry only. Predicted objects retain all
        # their existing restricted purpose flags and raw component stamps.
        'followGeometryMonoTime': (follow_geometry_ns if predicted else eof) if enduring else 0,
        'brakeQualified': not caution, 'stopQualified': enduring and not predicted,
        'departureQualified': enduring and not predicted and memory.vx > .5,
        'fcwQualified': not caution and not predicted and memory.position_variance <= 2.25 and memory.velocity_variance <= 6.25,
        'cautionQualified': caution and not predicted,
        'physicalConflictValid': physical_valid,
        'physicalConflictStart': float(path.physical_start) if physical_valid else 0.0,
        'physicalConflictEnd': float(path.physical_end) if physical_valid else 0.0,
        'coasting': predicted, 'pathState': path.state, 'pathOffset': path.offset,
        'conflictStart': path.start, 'conflictEnd': path.end, 'lateralSpeed': path.rate,
        'pathSource': path.source, 'visionIndex': vision_index, 'reason': reason,
      },
    }

  def _coast(self, memory, eof, model_ns, now_ns, speed, yaw, corridors, reason):
    if (memory.accepted is None or not memory.accepted['evidence']['brakeQualified']
        or not _fresh(memory.position_ns, now_ns, COAST_NS)
        or not _fresh(memory.velocity_ns, now_ns, COAST_NS) or not memory.epoch_ns <= eof):
      return None
    x, y, vx, vy = _project(memory.x, memory.y, memory.vx, memory.vy, (eof - memory.epoch_ns) / 1e9, speed, yaw)
    if x <= 0:
      return None
    dt = max(0.0, (eof - memory.epoch_ns) / 1e9)
    lateral_position_variance, lateral_velocity_variance = _lateral_variances(
      memory.position_variance, memory.lateral_position_variance, memory.velocity_variance,
      memory.lateral_velocity_variance, dt, yaw)
    path = _object_path(corridors, x, y, vx, vy, lateral_position_variance, lateral_velocity_variance,
                        memory.lateral_velocity_valid, confirmed=memory.path.confirmed)
    old_window = self._historical_path(memory, eof)
    if path.state == 'outOfPath' or path.observed_out or old_window.state == 'outOfPath':
      return None
    if path.state == 'unknown':
      path = old_window
    if path.state not in ('inPath', 'entering'):
      return None
    path = _without_physical_conflict(path)
    # A coast cannot stretch a crossing's previously observed clearing time.
    if old_window.end > 0:
      path = replace(path, start=max(path.start, old_window.start), end=min(path.end, old_window.end),
                     continuous=path.continuous and old_window.continuous,
                     prediction_limited=path.prediction_limited or old_window.prediction_limited)
      if path.end <= path.start:
        return None
    projected = _Memory(memory.track_id, memory.oem_id, memory.generation, memory.position_seq, memory.position_ns,
                        memory.velocity_ns, eof, x, y, vx, vy, memory.acceleration, False,
                        memory.position_variance + .5 * dt + 4 * dt * dt,
                        memory.velocity_variance + 4 * dt, memory.path, memory.path_ns,
                        lateral_position_variance=lateral_position_variance, lateral_velocity_variance=lateral_velocity_variance,
                        lateral_velocity_valid=memory.lateral_velocity_valid)
    accepted = memory.accepted['evidence']
    geometry_ns = accepted.get('followGeometryMonoTime', 0)
    # Carry only the latest actually observed follow geometry of these exact
    # components. An intervening predicted/revoked estimate cannot borrow an
    # older fresh epoch, even if the track/generation happens to be unchanged.
    if not (accepted.get('coasting') is False and accepted.get('followQualified') is True
            and accepted.get('source') in ('radar', 'fused')
            and accepted.get('sessionId') == self._session
            and accepted.get('trackId') == memory.track_id and accepted.get('generation') == memory.generation
            and accepted.get('positionMonoTime') == memory.position_ns and accepted.get('velocityMonoTime') == memory.velocity_ns
            and _integer(geometry_ns) and 0 < geometry_ns <= eof
            and geometry_ns == accepted.get('estimateMonoTime') == memory.epoch_ns
            and 0.0 <= path.start <= 1e-6):
      geometry_ns = 0
    return self._make_lead(projected, path, eof, model_ns, speed, predicted=True, reason=reason,
                           follow_geometry_ns=geometry_ns)

  @staticmethod
  def _select(objects):
    # Independent near visual cut-ins remain in the complete object set even
    # when a farther radar target supplies the compatible display lead slot.
    unique = []
    for lead in objects:
      evidence = lead['evidence']
      if evidence['source'] == 'vision' and any(
          other['evidence']['source'] == 'vision'
          and all(abs(lead[k] - other[k]) < .01 for k in ('dRel', 'yRel', 'vLeadK', 'aLeadK')) for other in unique):
        continue
      unique.append(lead)
    unique.sort(key=lambda lead: (lead['evidence']['conflictStart'], lead['dRel'], lead['evidence']['trackId']))
    compatible = [lead for lead in unique if lead['evidence']['pathState'] == 'inPath'
                  and (lead['evidence']['brakeQualified'] or lead['evidence']['followQualified'])]
    compatible.sort(key=lambda lead: lead['dRel'])
    leads = (compatible + [{'present': False}, {'present': False}])[:2]
    return (leads[0], leads[1]), unique

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
    previous_tracks = dict(self._tracks)
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
            for memory in self._tracks.values():
              self._lost(memory)
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
      for memory in self._tracks.values():
        self._lost(memory)
      self._tracks.clear()
      self._cleared.clear()
      tracks = []
    if not healthy:
      tracks = []
    radar_objects, used_visions, touched, cleared = [], set(), set(), set()
    degraded = not healthy
    for track in tracks:
      identifier, oem, generation = (track[k] for k in ('trackId', 'oemTrackId', 'generation'))
      if (not all(_integer(value) for value in (identifier, oem, generation)) or identifier <= 0
          or not 1 <= oem <= 63 or generation < 0):
        degraded = True
        continue
      touched.add(identifier)
      previous = self._tracks.get(identifier) or self._lost_tracks.get((self._session, identifier))
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
        self._lost(previous)
        self._tracks.pop(identifier, None)
        degraded |= previous is not None and previous.relevant
        continue
      if track['active'] is not True or not _fresh(p_ns, now_ns):
        allow_coast = track['active'] is True or track['lastInvalidReason'] in ('expired', 'slotReassigned')
        coast = self._coast(previous, eof, model_ns, now_ns, speed, yaw, corridors, 'radarGap') if previous and allow_coast else None
        if coast is not None:
          radar_objects.append(coast)
        else:
          self._lost(previous)
        if not allow_coast:
          self._tracks.pop(identifier, None)
        degraded |= previous is not None and previous.relevant
        continue
      if track['mature'] is not True or not _integer(track['observations']) or track['observations'] < MIN_OBSERVATIONS:
        continue
      velocity_values = _finite(vx, vy, v_var) and -35 <= vx <= 80 and abs(vy) <= 30 and 0 <= v_var <= VELOCITY_MAX_VARIANCE
      # Determine association in the current frame before borrowing any visual
      # component. The unresolved/different hypothesis is never a fallback.
      if not _finite(vy) or abs(vy) > 30:
        self._lost(previous)
        self._tracks.pop(identifier, None)
        degraded = True
        continue
      if not _finite(vx) or not -35 <= vx <= 80:
        # Only an established estimate may bridge the preliminary association
        # projection. With a simultaneous frame no velocity is needed to
        # compare position; a current confirmed visual match can then supply it.
        if previous is None and source_eof != eof:
          degraded = True
          continue
        vx = previous.vx if previous else 0.0
      px, py, pvx, pvy = _project(x, y, vx, vy, (eof - source_eof) / 1e9, speed, yaw)
      projected_vx, projected_vy = pvx, pvy
      vision_index = self._binding(identifier, bindings, visions, px, py, source_eof)
      vision = visions[vision_index] if vision_index >= 0 else None
      direct_velocity = (velocity_values and track['velocityGood'] is True and _fresh(v_ns, now_ns)
                         and v_ns <= observation_ns and abs(v_ns - p_ns) <= MAX_AGE_NS
                         and (previous is None or v_ns >= previous.raw_velocity_ns))
      predicted, component_reason = False, 'radar'
      if not direct_velocity:
        if vision is not None:
          pvx, v_ns, v_var = vision['vLeadK'], eof, vision['evidence']['velocityStd'] ** 2
          component_reason = 'visionVelocity'
        elif (velocity_values and v_var <= POSITION_SPEED_MAX_VARIANCE and track['observations'] >= 6
              and _integer(track['previousPositionSeq']) and track['previousPositionSeq'] < p_seq
              and _fresh(track['previousPositionMonoTime'], p_ns)):
          v_ns = p_ns
          component_reason = 'positionVelocity'
        elif previous and _fresh(previous.velocity_ns, now_ns, COAST_NS):
          _, _, pvx, pvy = _project(previous.x, previous.y, previous.vx, previous.vy,
                                    (eof - previous.epoch_ns) / 1e9, speed, yaw)
          v_ns, v_var = previous.velocity_ns, previous.velocity_variance + 4 * (eof - previous.epoch_ns) / 1e9
          lateral_velocity_variance = previous.lateral_velocity_variance + 3 * max(0.0, (eof - previous.epoch_ns) / 1e9)
          lateral_velocity_valid = previous.lateral_velocity_valid
          predicted, component_reason = True, 'velocityCoast'
        else:
          self._lost(previous)
          # An unusable roadside velocity is not a sensor-wide control fault.
          position_path = next((relation for corridor in corridors
                                if (relation := corridor.relation(px, py, 0.0, 0.0, half_width=SWEPT_HALF_WIDTH)).state != 'unknown'), _Path())
          prior_risk = previous is not None and previous.accepted is not None and previous.relevant
          degraded |= prior_risk or position_path.state == 'inPath'
          continue
      # A borrowed component must also drive alignment of position over the
      # same interval; otherwise d and v would describe different trajectories.
      projection_dt = (eof - source_eof) / 1e9
      px += (pvx - projected_vx) * projection_dt
      py += (pvy - projected_vy) * projection_dt
      if not _finite(px, py, pvx, pvy, v_var) or px <= 0 or v_var < 0:
        degraded = True
        continue
      lateral_position_variance, lateral_velocity_variance = _lateral_variances(
        p_var, lateral_position_variance, v_var, lateral_velocity_variance, projection_dt, yaw)
      if previous is not None and 0 <= eof - previous.epoch_ns <= COAST_NS:
        expected = _project(previous.x, previous.y, previous.vx, previous.vy, (eof - previous.epoch_ns) / 1e9, speed, yaw)
        if abs(px - expected[0]) > max(3.0, 3 * math.sqrt(p_var + previous.position_variance)) or abs(py - expected[1]) > 2.0:
          self._lost(previous)
          self._tracks.pop(identifier, None)
          degraded = True
          continue
      path = _object_path(corridors, px, py, pvx, pvy, lateral_position_variance, lateral_velocity_variance,
                          lateral_velocity_valid, confirmed=previous is not None and previous.path.confirmed,
                          vision=vision is not None)
      direct_path = path.state != 'unknown' or path.observed_out
      if path.state == 'edge' and path.authority_reason == 'edgeInnerUnknown' and previous and previous.relevant:
        history = self._historical_path(previous, eof)
        if history.state in ('inPath', 'entering'):
          path, direct_path = history, False
          component_reason += '+edgeInnerUnknown'
        else:
          self._lost(previous)
      if not direct_path:
        path = self._historical_path(previous, eof)
        predicted = True
        component_reason += '+pathPrediction'
      if path.state == 'outOfPath' or path.observed_out:
        independent = previous is not None and p_ns > previous.position_ns and p_seq > previous.position_seq
        known = (previous is not None and previous.accepted is not None and previous.accepted['evidence']['brakeQualified']
                 or identifier in self._cleared or (self._session, identifier) in self._lost_tracks)
        if known and not predicted and (independent or identifier in self._cleared and p_ns >= self._cleared[identifier]):
          self._cleared[identifier] = p_ns
          cleared.add(identifier)
          self._tracks.pop(identifier, None)
          self._lost_tracks.pop((self._session, identifier), None)
        elif known:
          # A changed plan geometry may withdraw a candidate immediately, but
          # a fresh independent radar position must corroborate a HOLD release.
          degraded = True
        if path.state != 'edge':
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
      if track['coasting'] is True:
        predicted = True
        component_reason += '+positionCoast'
      memory = _Memory(identifier, oem, generation, p_seq, p_ns, v_ns, eof, px, py, pvx, pvy,
                       float(acceleration), acceleration_good, float(p_var), float(v_var),
                       path if direct_path else previous.path if previous else _Path(),
                       eof if direct_path else previous.path_ns if previous else 0,
                       raw_velocity_ns=track['velocityMonoTime'] if direct_velocity else previous.raw_velocity_ns if previous else 0,
                       last_vision=vision if vision is not None else previous.last_vision if previous else None,
                       lateral_position_variance=lateral_position_variance, lateral_velocity_variance=lateral_velocity_variance,
                       lateral_velocity_valid=lateral_velocity_valid)
      memory.relevant = self._relevant(memory, path, speed)
      if (path.state in ('inPath', 'entering') or path.state == 'edge' and not predicted) and path.end > path.start:
        if path.state != 'edge':
          self._cleared.pop(identifier, None)
        lead = self._make_lead(memory, path, eof, model_ns, speed, vision=vision,
                               vision_index=vision_index, predicted=predicted, reason=component_reason)
        memory.accepted = lead
        radar_objects.append(lead)
        if not predicted and path.state != 'edge':
          self._lost_tracks.pop((self._session, identifier), None)
        if vision_index >= 0 and not predicted and path.state != 'edge':
          used_visions.add(vision_index)
          # Model's future hypotheses matched to this same track are not new
          # physical obstacles and must not dilute the radar replacement.
          for other in range(len(visions)):
            candidate = self._binding(identifier, [b for b in bindings if b['modelIndex'] == other], visions, px, py, source_eof)
            if candidate >= 0:
              used_visions.add(candidate)
      elif previous is not None and previous.accepted is not None:
        self._lost(previous)
      self._tracks[identifier] = memory
      degraded |= predicted and memory.accepted is not None and memory.relevant
    visual_recoveries = set()
    if not fault:
      # Preserve a proven same-car visual stream during an optional observer
      # loss. An unrelated visual lead cannot remove another target's barrier.
      for identifier, memory in previous_tracks.items():
        active = next((lead for lead in radar_objects if lead['evidence']['trackId'] == identifier), None)
        if identifier in cleared or active is not None and not active['evidence']['coasting']:
          continue
        contradicted = any(b['candidateTrackId'] == identifier and
                          (b['identityState'] == 'different' or b['rejectionMonoTime']) for b in bindings)
        if contradicted:
          continue
        fallback = self._visual_fallback(memory, visions, eof, speed, yaw)
        if fallback is not None:
          radar_objects = [lead for lead in radar_objects if lead['evidence']['trackId'] != identifier]
          radar_objects.append(fallback)
          used_visions.add(fallback['evidence']['visionIndex'])
          visual_recoveries.add(identifier)
          touched.add(identifier)
          self._tracks[identifier] = memory
          self._lost_tracks.pop((self._session, identifier), None)
    for identifier, memory in tuple(self._tracks.items()):
      if identifier in touched:
        continue
      coast = self._coast(memory, eof, model_ns, now_ns, speed, yaw, corridors, 'observerGap' if not healthy else 'trackMissing')
      if coast is not None:
        radar_objects.append(coast)
        degraded |= memory.relevant
      else:
        if memory.accepted is not None:
          # Expiry is loss of evidence, not proof that the road is clear.
          self._lost(memory)
        del self._tracks[identifier]
    if len(radar_objects) > MAX_TRACKS:
      # The producer has 32 live tracks. If replacement tracks overlap more
      # than that many retained losses, keep the most immediate constraints
      # and report an explicit incomplete-set failure, never silently PASS.
      radar_objects.sort(key=lambda lead: (lead['evidence']['conflictStart'],
                                          lead['dRel'] / max(.1, -lead['vRel']), lead['dRel']))
      for lead in radar_objects[MAX_TRACKS:]:
        self._lost(self._tracks.get(lead['evidence']['trackId']))
      radar_objects = radar_objects[:MAX_TRACKS]
      reason, healthy, degraded = 'objectCapacity', False, True
    if len(self._tracks) > MAX_TRACKS:
      retained = {lead['evidence']['trackId'] for lead in radar_objects}
      order = sorted(self._tracks, key=lambda key: (key not in retained, -self._tracks[key].position_ns))
      for identifier in order[MAX_TRACKS:]:
        self._lost(self._tracks[identifier])
        del self._tracks[identifier]
      reason, healthy, degraded = 'objectCapacity', False, True
    remaining = [lead for index, lead in enumerate(visions) if lead.get('present') is True and index not in used_visions]
    leads, objects = self._select(remaining + radar_objects)
    self._cleared = {key: stamp for key, stamp in self._cleared.items() if key in touched and _fresh(stamp, now_ns)}
    # A valid observation with no targets is healthy sensor transport. It does
    # not erase an unresolved loss of a previously admitted object. A fresh
    # unrelated farther obstacle cannot prove that the lost near one cleared.
    if visual_recoveries and not fault and not self._unresolved_loss and not self._lost_tracks:
      unresolved = any(memory.relevant and identifier not in visual_recoveries
                       and not any(lead['evidence']['trackId'] == identifier and not lead['evidence']['coasting'] for lead in radar_objects)
                       for identifier, memory in previous_tracks.items())
      if not unresolved:
        degraded = False
    risk_unresolved = self._unresolved_loss or bool(self._lost_tracks)
    degraded |= risk_unresolved
    if reason == 'active' and degraded:
      reason = 'objectDegraded'
    elif reason == 'active' and not objects:
      reason = 'noQualifiedObjects'
    return RadarControlResult(leads, objects, healthy, degraded, reason, tuple(sorted(cleared)), self._session, risk_unresolved)
