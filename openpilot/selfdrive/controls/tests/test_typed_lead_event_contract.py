"""Complete lead events for the native MPC layer.

This does not replace the separately bound 150
planner/owner/CAN scenarios. The ideal actuator delivers the bounded acceleration
request immediately; it is neither a Honda model nor a body-jerk qualification.

The three initial states are preserved from the former pointwise-strength test.
Each lead keeps its declared deceleration until stopped, then stays stopped for
all future time. The independent 2 m clearance and 2 s terminal observation are
the existing engineering contracts, not values fitted to these candidate runs.
"""

import json
import math
from dataclasses import dataclass

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MAX, ACCEL_MIN
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS, LongitudinalMpc, MpcReference
from openpilot.selfdrive.controls.lib.model_reference import ModelReferenceAdapter
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import model_reference_from_velocity
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import action, radar_state
from openpilot.selfdrive.modeld.constants import ModelConstants

CLEARANCE_M = 2.0
TERMINAL_SECONDS = 2.0
STOP_SPEED_MPS = 0.05
NUMERIC_DISTANCE_M = 1e-9
MODEL_TIMES = np.asarray(ModelConstants.T_IDXS)


@dataclass(frozen=True)
class LeadEvent:
  name: str
  gap: float
  lead_speed: float
  lead_accel: float
  ego_speed: float
  ego_accel: float = -1.0
  duration: float = 30.0


EVENTS = (
  LeadEvent('stationary', 20.0, 0.0, 0.0, 10.0),
  LeadEvent('closing_braking', 44.0, 10.0, -1.5, 20.0),
  LeadEvent('matching_then_braking', 30.0, 20.0, -3.0, 20.0),
)


def motion(x, v, a, t):
  """Exact constant-acceleration segment with a nonnegative-speed stop clamp."""
  moving_t = min(t, v / -a) if a < 0.0 else t
  return x + v * moving_t + 0.5 * a * moving_t**2, max(0.0, v + a * moving_t)


def minimum_gap(x, v, a, lead_x, lead_v, lead_a, duration):
  """Include stop breakpoints and relative-velocity roots inside every step."""
  breaks = {0.0, duration}
  for speed, acceleration in ((v, a), (lead_v, lead_a)):
    if acceleration < 0.0 and 0.0 < speed / -acceleration < duration:
      breaks.add(speed / -acceleration)
  points = sorted(breaks)
  candidates = list(points)
  for begin, end in zip(points[:-1], points[1:], strict=True):
    ego_now = motion(x, v, a, begin)[1]
    lead_now = motion(lead_x, lead_v, lead_a, begin)[1]
    ego_a = 0.0 if a < 0.0 and begin >= v / -a else a
    other_a = 0.0 if lead_a < 0.0 and begin >= lead_v / -lead_a else lead_a
    relative_a = other_a - ego_a
    if relative_a != 0.0:
      root = begin - (lead_now - ego_now) / relative_a
      if begin < root < end:
        candidates.append(root)
  return min(motion(lead_x, lead_v, lead_a, t)[0] - motion(x, v, a, t)[0] for t in candidates)


def required_closure(v, lead_v, lead_a):
  """Independent legal maximum-brake witness for this instantaneous plant.

  It uses the explicitly frozen lead program, no model path, candidate output,
  baseline output, latency reserve or unmeasured road capability.
  """
  ego_stop = v / -ACCEL_MIN
  lead_stop = lead_v / -lead_a if lead_a < 0.0 else 0.0
  return max(0.0, -minimum_gap(0.0, v, ACCEL_MIN, 0.0, lead_v, lead_a, max(ego_stop, lead_stop)))


class NativePolicy:
  def __init__(self, *, lead_index, positive_model):
    self.lead_index = lead_index
    self.positive_model = positive_model
    self.mpc = LongitudinalMpc()
    self.adapter = ModelReferenceAdapter()

  def __call__(self, frame, speed, acceleration, gap, lead_speed, lead_accel):
    radar = radar_state(distance=gap, speed=lead_speed, acceleration=lead_accel)
    if self.lead_index:
      radar.leadOne, radar.leadTwo = radar.leadTwo, radar.leadOne
    reference = MpcReference(40.0)
    if self.positive_model:
      raw = speed + 0.6 * MODEL_TIMES
      position, velocity, model_accel = model_reference_from_velocity(raw)
      epoch = 1_000_000_000 + round(frame * DT_MDL * 1e9)
      typed = self.adapter.build(raw, T_IDXS, speed, model_mono_time_ns=epoch,
                                 anchor_mono_time_ns=epoch, consumed_mono_time_ns=epoch)
      reference = MpcReference(40.0, model_position=position, model_velocity=velocity,
                               model_acceleration=model_accel, typed_model=typed, preference_time_s=2.5)
    self.mpc.set_cur_state(speed, acceleration)
    self.mpc.set_weights(prev_accel_constraint=frame > 0 and speed > 0.001)
    calls_before = self.mpc.diagnostic_solve_calls
    self.mpc.update(radar, reference=reference)
    errors = []
    if self.mpc.diagnostic_solve_calls <= calls_before or self.mpc.diagnostic_last_solver_return != 0:
      errors.append('native_solve_missing_or_failed')
    if self.mpc.solution_status != 0:
      errors.append('published_solution_rejected:' + str(self.mpc.solution_status))
    if not all(np.all(np.isfinite(values)) for values in (self.mpc.x_sol, self.mpc.u_sol, self.mpc.params)):
      errors.append('nonfinite_native_output')
    command = float(action(self.mpc))
    # MPC acceleration limits are soft costs. The planner/actuator boundary
    # clips the actual request; checking the raw optimum as though it were a
    # transmitted command would conflate those two different contracts.
    if not math.isfinite(command):
      errors.append('nonfinite_native_action')
    return command, errors


def run_event(case, policy, *, duration=None, future_program_declared=True):
  duration = case.duration if duration is None else duration
  x, v, acceleration = 0.0, case.ego_speed, case.ego_accel
  lead_x, lead_v = case.gap, case.lead_speed
  initial_margin = case.gap - CLEARANCE_M - required_closure(v, lead_v, case.lead_accel)
  rows, failures = [], []
  for frame in range(round(duration / DT_MDL)):
    lead_a = case.lead_accel if lead_v > 0.0 else 0.0
    command, software_failures = policy(frame, v, acceleration, lead_x - x, lead_v, lead_a)
    failures.extend(software_failures)
    if not math.isfinite(command):
      break  # Preserve the software failure; a fabricated continuation is not evidence.
    delivered = float(np.clip(command, ACCEL_MIN, ACCEL_MAX))
    next_x, next_v = motion(x, v, delivered, DT_MDL)
    next_lead_x, next_lead_v = motion(lead_x, lead_v, lead_a, DT_MDL)
    gap_min = minimum_gap(x, v, delivered, lead_x, lead_v, lead_a, DT_MDL)
    # Monotonicity in ego/lead speed supplies a conservative whole-step bound;
    # endpoints alone would miss an interior gap minimum.
    margin = gap_min - CLEARANCE_M - required_closure(max(v, next_v), min(lead_v, next_lead_v), lead_a)
    rows.append({'end_s': (frame + 1) * DT_MDL, 'x': next_x, 'v': next_v, 'previous_v': v,
                 'lead_v': next_lead_v, 'previous_lead_v': lead_v, 'gap_min_m': gap_min,
                 'margin_lower_bound_m': margin, 'command_mps2': command,
                 'delivered_mps2': delivered})
    x, v, lead_x, lead_v = next_x, next_v, next_lead_x, next_lead_v
    acceleration = 0.0 if v == 0.0 and delivered < 0.0 else delivered
  gaps = []
  if initial_margin < -NUMERIC_DISTANCE_M:
    gaps.append('initial_state_not_qualified_by_independent_witness')
  if not rows or rows[-1]['end_s'] < case.duration - 1e-12:
    gaps.append('incomplete_original_event_window')
  if not future_program_declared:
    gaps.append('future_lead_program_not_declared')
  if rows:
    if min(row['gap_min_m'] for row in rows) < CLEARANCE_M - NUMERIC_DISTANCE_M:
      failures.append('independent_clearance_violation')
    if initial_margin >= -NUMERIC_DISTANCE_M and min(row['margin_lower_bound_m'] for row in rows) < -NUMERIC_DISTANCE_M:
      failures.append('independent_response_margin_violation')
  tail = [row for row in rows if row['end_s'] > case.duration - TERMINAL_SECONDS + 1e-12]
  if (len(tail) * DT_MDL < TERMINAL_SECONDS - 1e-12 or
      not all(max(row['v'], row['previous_v'], row['lead_v'], row['previous_lead_v']) <= STOP_SPEED_MPS for row in tail)):
    gaps.append('terminal_stop_not_observed_for_two_seconds')
  status = 'FAIL' if failures else 'INCONCLUSIVE' if gaps else 'PASS'
  return {'status': status, 'failures': sorted(set(failures)), 'gaps': gaps, 'rows': rows,
          'initial_margin_m': initial_margin, 'scope': 'instantaneous_ideal_actuator_native_MPC_event_only',
          'future_program': 'Lead remains stopped forever; no later departure or acceleration event'}


@pytest.mark.parametrize('case', EVENTS, ids=lambda case: case.name)
@pytest.mark.parametrize('lead_index', [0, 1])
def test_positive_model_meets_independent_complete_lead_event(case, lead_index, record_property):
  arms = {name: run_event(case, NativePolicy(lead_index=lead_index, positive_model=positive))
          for name, positive in (('lead_only', False), ('positive_model', True))}
  diagnostics = {}
  for name, result in arms.items():
    diagnostics[name] = {key: value for key, value in result.items() if key != 'rows'}
    if result['rows']:
      diagnostics[name].update(min_gap_m=min(row['gap_min_m'] for row in result['rows']),
                               min_margin_lower_bound_m=min(row['margin_lower_bound_m'] for row in result['rows']),
                               final_speed_mps=result['rows'][-1]['v'],
                               raw_request_range_mps2=[min(row['command_mps2'] for row in result['rows']),
                                                       max(row['command_mps2'] for row in result['rows'])],
                               delivered_range_mps2=[min(row['delivered_mps2'] for row in result['rows']),
                                                     max(row['delivered_mps2'] for row in result['rows'])])
  if len(arms['lead_only']['rows']) == len(arms['positive_model']['rows']) and arms['lead_only']['rows']:
    paired = list(zip(arms['lead_only']['rows'], arms['positive_model']['rows'], strict=True))
    diagnostics['relative_diagnostic_only'] = {
      'first_action_delta_mps2': paired[0][1]['command_mps2'] - paired[0][0]['command_mps2'],
      'maximum_prefix_extra_ego_distance_m': max(c['x'] - b['x'] for b, c in paired),
      'gate': 'NONE: independent complete-event contracts decide acceptance',
    }
  record_property('lead_event_evidence', json.dumps(diagnostics, sort_keys=True))
  # The reference cannot excuse a failing candidate; its own failure is also
  # retained as an explicit regression, not silently used as the target value.
  assert arms['lead_only']['status'] == 'PASS', diagnostics
  assert arms['positive_model']['status'] == 'PASS', diagnostics


@pytest.mark.parametrize('case', EVENTS, ids=lambda case: case.name)
def test_frozen_initial_state_has_a_legal_braking_witness(case):
  result = run_event(case, lambda *_: (ACCEL_MIN, []))
  assert result['initial_margin_m'] >= 0.0
  assert result['status'] == 'PASS', {k: v for k, v in result.items() if k != 'rows'}


def test_missing_braking_is_rejected_even_when_future_qualification_is_missing():
  result = run_event(EVENTS[0], lambda *_: (0.0, []), future_program_declared=False)
  assert result['status'] == 'FAIL'
  assert 'independent_clearance_violation' in result['failures']
  assert 'future_lead_program_not_declared' in result['gaps']


def test_safe_prefix_cannot_replace_the_original_event_and_terminal_tail():
  result = run_event(EVENTS[0], lambda *_: (ACCEL_MIN, []), duration=1.0)
  assert result['status'] == 'INCONCLUSIVE'
  assert 'incomplete_original_event_window' in result['gaps']
  assert 'terminal_stop_not_observed_for_two_seconds' in result['gaps']
