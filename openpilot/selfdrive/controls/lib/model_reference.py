"""Model ego preferences and discrete evidence, independent of lead geometry.

The operational clock below is explicitly a publication-time proxy. Neither
the model's physical prediction epoch nor a measured road stopping line is
available here. Negative velocity is retained as evidence before projection;
only the moving preference removes a common velocity-origin offset.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, STOP_SPEED
from openpilot.selfdrive.controls.lib.lead_launch import STANDSTILL_RESIDUE
from openpilot.selfdrive.controls.lib.motion_supervisor import MODEL_LEASE_NS
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_MAX_AGE_NS, finite_state
from openpilot.selfdrive.modeld.constants import ModelConstants


PREFERENCE_TIME_S = 2.5  # engineering scaling, not a vehicle-identified parameter
FORWARD_HORIZON_S = float(ModelConstants.T_IDXS[CONTROL_N - 1])
# Candidate domain for a stopped Accord with independently observed zero
# wheels and no current lead. These are engineering budgets, not calibrated
# model uncertainty: 0.03 m/s alone over 0.5 s could permit 15 mm, so position
# and the integrated negative velocity must each also stay within 5 mm.
STATIONARY_REVERSE_SPEED = 0.03
STATIONARY_REVERSE_DISTANCE = 0.005
STATIONARY_RECOVERY_TIME = 0.5


def _immutable(values, *, dtype=float):
  snapshot = np.array(values, dtype=dtype, copy=True)
  snapshot.setflags(write=False)
  return snapshot


@dataclass(frozen=True)
class ModelSource:
  model_mono_time_ns: int | None
  anchor_mono_time_ns: int | None
  consumed_mono_time_ns: int | None
  proxy_age_s: float | None = None
  epoch_basis: str = 'publication_proxy'
  physical_prediction_epoch_ns: int | None = None


@dataclass(frozen=True)
class MovingPreference:
  position: np.ndarray
  velocity: np.ndarray
  acceleration: np.ndarray
  supported: np.ndarray

  def __post_init__(self):
    for name in ('position', 'velocity', 'acceleration', 'supported'):
      object.__setattr__(self, name, _immutable(getattr(self, name), dtype=bool if name == 'supported' else float))


@dataclass(frozen=True)
class RestGoal:
  # The interval belongs to the original source grid. native_node is a stated
  # discretization, not an exact physical stop timestamp.
  source_node_interval: tuple[int, int]
  source_time_interval_s: tuple[float, float]
  event_proxy_epoch_ns: int
  remaining_time_s: float
  native_node: int | None
  native_time_s: float | None
  nominal_position: float | None
  position_kind: str = 'nominal_preference'
  position_frame: str = 'model_publication_proxy_origin'
  future_forward: bool = False
  # Prior qualified model source and goal epoch, both integer nanoseconds.
  continued_from: tuple[int, int] | None = None


@dataclass(frozen=True)
class NoForward:
  reason: str


@dataclass(frozen=True)
class ForwardIntent:
  horizon_s: float
  raw_initial_speed: float
  raw_maximum_speed: float


@dataclass(frozen=True)
class Unknown:
  reason: str


@dataclass(frozen=True)
class RawRestGoal:
  source_node_interval: tuple[int, int]
  source_time_interval_s: tuple[float, float]
  nominal_position: float
  future_forward: bool


@dataclass(frozen=True)
class RawMotionEvidence:
  """Source-grid semantics; these do not supply an aligned moving reference."""

  no_forward: NoForward | None = None
  rest_goal: RawRestGoal | None = None
  forward_intent: ForwardIntent | None = None
  unknown: Unknown | None = None

  @property
  def kind(self):
    if self.unknown is not None:
      return 'unknown'
    if self.rest_goal is not None:
      return 'rest'
    if self.no_forward is not None:
      return 'no_forward'
    return 'forward' if self.forward_intent is not None else 'moving'

  @property
  def does_not_veto_departure(self):
    return self.unknown is None and self.rest_goal is None and self.no_forward is None


@dataclass(frozen=True)
class ModelReferenceEvidence:
  source: ModelSource
  moving: MovingPreference | None
  events: tuple[RestGoal | NoForward | ForwardIntent | Unknown, ...]
  raw_velocity: np.ndarray | None = None
  raw_motion: RawMotionEvidence | None = None

  def __post_init__(self):
    if self.raw_velocity is not None:
      object.__setattr__(self, 'raw_velocity', _immutable(self.raw_velocity))

  @property
  def rest_goal(self):
    return next((event for event in self.events if isinstance(event, RestGoal)), None)

  @property
  def no_forward(self):
    return (any(isinstance(event, NoForward) for event in self.events)
            or self.raw_motion is not None and self.raw_motion.no_forward is not None)

  @property
  def has_rest_intent(self):
    return self.rest_goal is not None or self.raw_motion is not None and self.raw_motion.rest_goal is not None

  @property
  def unknown(self):
    return next((event for event in self.events if isinstance(event, Unknown)), None)

  @property
  def allows_departure(self):
    return (self.allows_external_departure and self.raw_motion.forward_intent is not None)

  @property
  def allows_external_departure(self):
    # Only the ordinary publication-order alignment gap preserves qualified
    # source semantics. True input failures and raw ambiguity still veto.
    source_usable = self.unknown is None or self.unknown.reason == 'model_proxy_after_initial_state'
    return (source_usable and self.raw_motion is not None and self.raw_motion.does_not_veto_departure
            and not self.has_rest_intent and not self.no_forward)


def raw_motion_from_velocity(raw_velocity, *, elapsed_s=0.0):
  """One parser for aligned and unaligned references; no measured-speed shift.

  For an aligned source, elapsed_s crops only the forward-evidence window.
  Without alignment the source's own first window is used. Negative prefixes
  and pending raw rest events remain vetoes even if a later window moves.
  """
  times = np.asarray(ModelConstants.T_IDXS, dtype=float)
  raw = np.asarray(raw_velocity, dtype=float)
  if (raw.shape != times.shape or not np.all(np.isfinite(raw))
      or not np.isfinite(elapsed_s) or not 0.0 <= elapsed_s <= times[-1]):
    return RawMotionEvidence(unknown=Unknown('invalid_raw_motion'))
  if np.max(raw) <= 0.0:
    return RawMotionEvidence(no_forward=NoForward('raw_has_no_positive_progress'))
  if raw[0] < 0.0:
    return RawMotionEvidence(unknown=Unknown('backward_then_forward_is_ambiguous'))
  stop_index = None
  if raw[0] == 0.0 and raw[1] == 0.0:
    stop_index = 0
  else:
    first_positive = np.flatnonzero(raw > 0.0)[0]
    stopped = np.flatnonzero((raw <= 0.0) & (np.arange(len(raw)) > first_positive))
    if len(stopped):
      stop_index = int(stopped[0])
      if stop_index + 1 == len(raw) or raw[stop_index + 1] > 0.0:
        return RawMotionEvidence(unknown=Unknown('isolated_zero_has_no_rest_support'))
  if stop_index is not None:
    previous = max(0, stop_index - 1)
    projected = np.maximum(raw[:stop_index + 1], 0.0)
    position = float(np.sum(0.5 * (projected[:-1] + projected[1:]) * np.diff(times[:stop_index + 1])))
    return RawMotionEvidence(rest_goal=RawRestGoal(
      (previous, stop_index), (float(times[previous]), float(times[stop_index])), position,
      bool(np.any(raw[stop_index + 1:] > 0.0)),
    ))
  horizon = min(FORWARD_HORIZON_S, float(times[-1]) - elapsed_s)
  support = np.concatenate(([elapsed_s], times[(times > elapsed_s) & (times < elapsed_s + horizon)], [elapsed_s + horizon]))
  progress = np.interp(support, times, raw)
  if np.all(progress >= 0.0) and np.max(progress) > max(float(progress[0]), STOP_SPEED):
    return RawMotionEvidence(forward_intent=ForwardIntent(horizon, float(progress[0]), float(np.max(progress))))
  return RawMotionEvidence()


def moving_preference(raw_velocity, native_times, measured_speed, *, proxy_age_s=0.0):
  """Only this preference is invariant to a constant raw-speed translation.

  Unsupported tail values remain finite for the native API, but their weights
  are disabled. They are not extrapolated evidence or departure permission.
  """
  times = np.asarray(ModelConstants.T_IDXS, dtype=float)
  raw = np.asarray(raw_velocity, dtype=float)
  native_times = np.asarray(native_times, dtype=float)
  if (raw.shape != times.shape or not np.all(np.isfinite(raw)) or not np.isfinite(measured_speed)
      or not np.isfinite(proxy_age_s) or not 0.0 <= proxy_age_s <= times[-1]
      or native_times.ndim != 1 or len(native_times) < 2 or native_times[0] != 0.0
      or not np.all(np.isfinite(native_times)) or np.any(np.diff(native_times) <= 0.0)):
    raise ValueError('Invalid moving preference')
  sample_times = native_times + proxy_age_s
  raw_anchor = float(np.interp(proxy_age_s, times, raw))
  velocity = measured_speed + np.interp(sample_times, times, raw) - raw_anchor
  acceleration = np.concatenate((np.diff(velocity) / np.diff(native_times), [0.0]))
  position = np.concatenate(([0.0], np.cumsum(0.5 * (velocity[:-1] + velocity[1:]) * np.diff(native_times))))
  supported = sample_times <= times[-1]
  # An interval slope also needs the next endpoint. Never weight a partial
  # tail interval as if it were a complete observed model preference.
  supported[:-1] &= supported[1:]
  return MovingPreference(position, velocity, acceleration, supported)


def _timestamp(value):
  return isinstance(value, Integral) and not isinstance(value, bool) and 0 < value < (1 << 64)


def normalize_standstill_speed(measured_speed, *, actual_standstill, raw_speed):
  """Remove only negative arithmetic residue from an explicitly stopped ego."""
  # M04 recorded -5.5e-14 m/s with zero raw wheel speed. Reuse the bounded
  # launch tolerance, with the same independent standstill/wheel evidence;
  # clipping every negative estimate would hide real rollback.
  if (actual_standstill is True and finite_state((measured_speed, raw_speed)) and raw_speed == 0.0
      and -STANDSTILL_RESIDUE <= measured_speed < 0.0):
    return 0.0
  return measured_speed


def stationary_lead_departure_compatible(evidence, *, actual_standstill, measured_speed, raw_speed):
  """Bounded model compatibility for an independently qualified lead departure.

  This supplies neither model departure permission nor a FORWARD classification.
  The caller must still qualify the lead, current positive target and model risk;
  the motion owner retains confirmation and brake-release authority. The speed,
  duration and displacement bounds are a conservative engineering domain for
  this stationary case, not calibrated sensor/model error bounds.
  """
  measured_speed = normalize_standstill_speed(measured_speed, actual_standstill=actual_standstill, raw_speed=raw_speed)
  if (actual_standstill is not True or not finite_state((measured_speed, raw_speed)) or raw_speed != 0.0
      or not 0.0 <= measured_speed <= 0.05
      or evidence is None or evidence.raw_velocity is None or evidence.raw_motion is None):
    return False
  source = evidence.source
  model_ns, anchor_ns, consumed_ns = source.model_mono_time_ns, source.anchor_mono_time_ns, source.consumed_mono_time_ns
  if (source.epoch_basis != 'publication_proxy' or source.physical_prediction_epoch_ns is not None
      or not all(_timestamp(value) for value in (model_ns, anchor_ns, consumed_ns))
      or not 0 <= consumed_ns - model_ns <= MODEL_LEASE_NS
      or not 0 <= consumed_ns - anchor_ns <= REFERENCE_MAX_AGE_NS):
    return False
  raw_unknown = Unknown('backward_then_forward_is_ambiguous')
  if model_ns > anchor_ns:
    if source.proxy_age_s is not None or evidence.events != (Unknown('model_proxy_after_initial_state'),):
      return False
    elapsed_s = 0.0
  else:
    elapsed_s = (anchor_ns - model_ns) / 1e9
    if (not finite_state((source.proxy_age_s,)) or source.proxy_age_s != elapsed_s
        or evidence.events != (raw_unknown,)):
      return False
  # A mutation/regression or any other Unknown from the adapter cannot enter
  # this branch, even if its raw data happened to resemble a departure.
  if evidence.moving is not None or evidence.raw_motion != RawMotionEvidence(unknown=raw_unknown):
    return False
  times = np.asarray(ModelConstants.T_IDXS, dtype=float)
  raw = evidence.raw_velocity
  if raw.shape != times.shape or not np.all(np.isfinite(raw)) or not -0.05 <= raw[0] < 0.0:
    return False
  positive = np.flatnonzero(raw > 0.0)
  if not len(positive):
    return False
  first_positive = int(positive[0])
  if (times[first_positive] > 1.0 or np.min(raw[:first_positive]) < -0.05
      or np.any(raw[:first_positive] >= 0.0) or np.any(raw[first_positive:] <= 0.0)):
    return False
  # Integrate the negative part of the piecewise-linear velocity, including
  # the fractional final interval up to its zero crossing.
  prefix = raw[:first_positive]
  reverse_distance = -float(np.sum(0.5 * (prefix[:-1] + prefix[1:]) * np.diff(times[:first_positive])))
  crossing_duration = (times[first_positive] - times[first_positive - 1]) * -prefix[-1] / (raw[first_positive] - prefix[-1])
  reverse_distance += 0.5 * -prefix[-1] * crossing_duration
  if reverse_distance > 0.02:
    return False
  # A current, independently qualified lead supplies departure evidence. The
  # model only has to remain compatible: requiring its own STOP_SPEED crossing
  # here would serialize two independent departure qualifications. Retain the
  # same short-horizon net progress and all reverse/rest bounds above, without
  # translating the prefix or changing the original UNKNOWN evidence.
  end_s = elapsed_s + FORWARD_HORIZON_S
  support = np.concatenate(([elapsed_s], times[(times > elapsed_s) & (times < end_s)], [end_s]))
  progress = np.interp(support, times, raw)
  distance = float(np.sum(0.5 * (progress[:-1] + progress[1:]) * np.diff(support)))
  return bool(np.isfinite(distance) and distance > 0.0)


def stationary_no_lead_departure(evidence, raw_position, position_times, velocity_times, *,
                                 actual_standstill, measured_speed, raw_speed, wheel_speeds, model_mono_time_ns):
  """Certify bounded near-zero model residue, retaining its original UNKNOWN.

  Caller must supply explicitly published current wheel measurements and gate
  current no-lead, input health, action/stop, platform and actuation state. This
  helper cannot grant release by itself; the planner/owner retain those gates.
  Position and velocity are separate model means, not independent ground truth.
  """
  if (not stationary_lead_departure_compatible(
        evidence, actual_standstill=actual_standstill, measured_speed=measured_speed, raw_speed=raw_speed)
      or abs(measured_speed) > STANDSTILL_RESIDUE
      or not _timestamp(model_mono_time_ns) or model_mono_time_ns != evidence.source.model_mono_time_ns):
    return False
  try:
    wheels = tuple(wheel_speeds)
    position = np.asarray(raw_position, dtype=float)
    position_times = np.asarray(position_times, dtype=float)
    velocity_times = np.asarray(velocity_times, dtype=float)
  except (TypeError, ValueError):
    return False
  times = np.asarray(ModelConstants.T_IDXS, dtype=float)
  if (len(wheels) != 4 or not finite_state(wheels) or any(value != 0.0 for value in wheels)
      or position.shape != times.shape or not np.all(np.isfinite(position))
      or not np.array_equal(position_times, times) or not np.array_equal(velocity_times, times)
      or abs(position[0]) > STANDSTILL_RESIDUE):
    return False
  raw = evidence.raw_velocity
  positive = np.flatnonzero(raw > 0.0)
  first_positive = int(positive[0])  # The source/UNKNOWN helper proved a negative prefix then forward motion.
  if np.min(raw) < -STATIONARY_REVERSE_SPEED or times[first_positive] > STATIONARY_RECOVERY_TIME:
    return False
  prefix = raw[:first_positive]
  reverse_distance = -float(np.sum(0.5 * (prefix[:-1] + prefix[1:]) * np.diff(times[:first_positive])))
  crossing_duration = (times[first_positive] - times[first_positive - 1]) * -prefix[-1] / (raw[first_positive] - prefix[-1])
  reverse_distance += 0.5 * -prefix[-1] * crossing_duration
  position = position - position[0]
  positive_position = np.flatnonzero(position > STANDSTILL_RESIDUE)
  if (reverse_distance > STATIONARY_REVERSE_DISTANCE or np.min(position) < -STATIONARY_REVERSE_DISTANCE
      or not len(positive_position) or times[positive_position[0]] > STATIONARY_RECOVERY_TIME
      or np.any(np.diff(position[positive_position[0]:]) < 0.0)):
    return False
  # Both current model means must support meaningful forward progress within
  # the ordinary departure horizon. A positive action or a distant positive
  # tail alone is not a stationary no-lead departure witness.
  horizon = FORWARD_HORIZON_S
  forward_velocity = float(np.interp(horizon, times, raw))
  forward_position = float(np.interp(horizon, times, position))
  return bool(forward_velocity > STOP_SPEED and forward_position > 0.5 * STOP_SPEED * horizon)


def lead_launch_model_compatible(evidence):
  """Retain model risks while a separately confirmed lead supplies movement.

  This does not reclassify UNKNOWN or grant release. A bounded, weak initial
  sign ambiguity need not supply the ego speed target for following a moving
  lead. Rest goals, later stops, substantial reverse travel and invalid source
  identities remain vetoes. These bounds define an engineering scope, not a
  calibrated model-error distribution.
  """
  if (evidence is None or evidence.raw_velocity is None or evidence.raw_motion is None
      or evidence.has_rest_intent or evidence.no_forward):
    return False
  source = evidence.source
  model_ns, anchor_ns, consumed_ns = source.model_mono_time_ns, source.anchor_mono_time_ns, source.consumed_mono_time_ns
  if (source.epoch_basis != 'publication_proxy' or source.physical_prediction_epoch_ns is not None
      or not all(_timestamp(value) for value in (model_ns, anchor_ns, consumed_ns))
      or not 0 <= consumed_ns - model_ns <= MODEL_LEASE_NS
      or not 0 <= consumed_ns - anchor_ns <= REFERENCE_MAX_AGE_NS):
    return False
  if model_ns > anchor_ns:
    if source.proxy_age_s is not None:
      return False
  elif source.proxy_age_s != (anchor_ns - model_ns) / 1e9:
    return False
  allowed_unknowns = {'model_proxy_after_initial_state', 'backward_then_forward_is_ambiguous',
                      'isolated_zero_has_no_rest_support'}
  if (evidence.unknown is not None and evidence.unknown.reason not in allowed_unknowns
      or evidence.raw_motion.unknown is not None and evidence.raw_motion.unknown.reason not in allowed_unknowns):
    return False
  times = np.asarray(ModelConstants.T_IDXS, dtype=float)
  raw = evidence.raw_velocity
  if raw.shape != times.shape or not np.all(np.isfinite(raw)) or np.min(raw) < -0.05 or raw[-1] <= 0.0:
    return False
  # Only near-rest ambiguity before the ordinary forward-evidence horizon is
  # eligible. Never erase a stop after meaningful forward motion, or a later
  # zero/negative prediction hidden by the parser's negative-prefix UNKNOWN.
  nonpositive = raw <= 0.0
  if (np.any(nonpositive & (times > FORWARD_HORIZON_S))
      or np.any(nonpositive & np.maximum.accumulate(raw > STOP_SPEED))):
    return False
  reverse_distance = 0.0
  for left, right, duration in zip(raw[:-1], raw[1:], np.diff(times), strict=True):
    if left <= 0.0 and right <= 0.0:
      reverse_distance -= 0.5 * (left + right) * duration
    elif left < 0.0:
      reverse_distance += 0.5 * left * left / (right - left) * duration
    elif right < 0.0:
      reverse_distance += 0.5 * right * right / (left - right) * duration
  return bool(reverse_distance <= 0.02)


class ModelReferenceAdapter:
  """A bounded production adapter; it does not invent traffic-event labels.

  A positive-to-sustained-nonpositive transition supplies a nominal rest goal.
  At least two initial zero samples before a later move preserve current rest;
  a single initial zero followed immediately by forward motion permits the
  forward control. A new immutable model publication may revise those events.
  """

  def __init__(self):
    self._last_sample = None
    self._rest_continuity = None

  def clear_rest_continuity(self):
    # Source mutation/regression detection survives an admission interruption.
    self._rest_continuity = None

  def build(self, raw_velocity, native_times, measured_speed, *, model_mono_time_ns,
            anchor_mono_time_ns, consumed_mono_time_ns, retain_rest_continuity=True):
    source = ModelSource(model_mono_time_ns, anchor_mono_time_ns, consumed_mono_time_ns)
    if not retain_rest_continuity:
      self.clear_rest_continuity()

    def unknown(reason):
      self.clear_rest_continuity()
      return ModelReferenceEvidence(source, None, (Unknown(reason),))

    if not all(_timestamp(value) for value in (model_mono_time_ns, anchor_mono_time_ns, consumed_mono_time_ns)):
      return unknown('missing_publication_proxy')
    if model_mono_time_ns > consumed_mono_time_ns or anchor_mono_time_ns > consumed_mono_time_ns:
      return unknown('future_publication_proxy')
    if consumed_mono_time_ns - model_mono_time_ns > MODEL_LEASE_NS:
      return unknown('expired_model_publication')
    try:
      raw = np.asarray(raw_velocity, dtype=float)
      native_times = np.asarray(native_times, dtype=float)
      if raw.shape != (len(ModelConstants.T_IDXS),) or not np.all(np.isfinite(raw)):
        return unknown('invalid_raw_velocity')
      snapshot = (int(model_mono_time_ns), tuple(float(value) for value in raw))
    except (TypeError, ValueError):
      return unknown('invalid_raw_velocity')
    if self._last_sample is not None:
      if snapshot[0] < self._last_sample[0]:
        return unknown('model_publication_regressed')
      if snapshot[0] == self._last_sample[0] and snapshot[1] != self._last_sample[1]:
        return unknown('model_mutated_with_same_identity')
    self._last_sample = snapshot
    if model_mono_time_ns > anchor_mono_time_ns:
      # The publication is already available and fresh. Its raw stop/move
      # evidence can be qualified without inventing an earlier prediction
      # epoch or extrapolating a moving preference back to the carState.
      self.clear_rest_continuity()
      raw_motion = raw_motion_from_velocity(raw)
      return ModelReferenceEvidence(source, None, (Unknown('model_proxy_after_initial_state'),), raw, raw_motion)
    age = (anchor_mono_time_ns - model_mono_time_ns) / 1e9
    source = ModelSource(int(model_mono_time_ns), int(anchor_mono_time_ns), int(consumed_mono_time_ns), age)
    try:
      moving = moving_preference(raw, native_times, measured_speed, proxy_age_s=age)
    except ValueError:
      return unknown('unsupported_model_reference_epoch')

    previous_rest = self._rest_continuity
    self.clear_rest_continuity()
    raw_motion = raw_motion_from_velocity(raw, elapsed_s=age)
    if raw_motion.unknown is not None:
      return ModelReferenceEvidence(source, None, (raw_motion.unknown,), raw, raw_motion)
    continued_from = None
    if (previous_rest is not None and np.all(raw == 0.0)
        and previous_rest[0] <= model_mono_time_ns
        and 0 <= consumed_mono_time_ns - previous_rest[0] <= MODEL_LEASE_NS
        and previous_rest[1] <= model_mono_time_ns):
      # NoForward alone supplies no deadline. Preserve a preceding, already
      # due qualified goal only while a fresh immutable zero source confirms
      # rest. Negative samples, missing admission and source failures break
      # this continuity; a repeated publication cannot renew its lease.
      continued_from = previous_rest
      raw_motion = RawMotionEvidence(no_forward=raw_motion.no_forward,
                                     rest_goal=RawRestGoal((0, 0), (0.0, 0.0), 0.0, False))
    events = []
    if raw_motion.no_forward is not None:
      moving = None
      events.append(raw_motion.no_forward)
    if raw_motion.rest_goal is not None:
      raw_goal = raw_motion.rest_goal
      event_time = raw_goal.source_time_interval_s[1]
      event_epoch = int(model_mono_time_ns) + round(event_time * 1e9)
      remaining_time = (event_epoch - int(anchor_mono_time_ns)) / 1e9
      node = int(np.searchsorted(native_times, max(0.0, remaining_time), side='left'))
      # The nominal position keeps the original model frame. It is not
      # rebased with model-predicted travel masquerading as measured travel.
      events.append(RestGoal(
        raw_goal.source_node_interval, raw_goal.source_time_interval_s, event_epoch, remaining_time,
        node if node < len(native_times) else None, float(native_times[node]) if node < len(native_times) else None,
        raw_goal.nominal_position, future_forward=raw_goal.future_forward, continued_from=continued_from,
      ))
      if retain_rest_continuity:
        self._rest_continuity = (int(model_mono_time_ns), event_epoch)
    elif raw_motion.forward_intent is not None:
      events.append(raw_motion.forward_intent)
    return ModelReferenceEvidence(source, moving, tuple(events), raw, raw_motion)
