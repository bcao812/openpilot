"""Bounded B5 no-lead departure; real planner/held owner plus raw veto controls."""
import json
from pathlib import Path

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.model_reference import stationary_no_lead_departure
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionPhase, MotionReferenceKind
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import consume, make_handover_state, make_planner_owner
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_typed_model_reference import MODEL_TIMES, SOURCE_NS, evidence


FIXTURE = json.loads((Path(__file__).parent / 'fixtures/b5_stationary_departure.json').read_text())
VELOCITY = np.array(FIXTURE['velocity'])
POSITION = np.array(FIXTURE['position'])


def qualifies(raw=VELOCITY, position=POSITION, **changes):
  values = dict(actual_standstill=True, measured_speed=0.0, raw_speed=0.0,
                wheel_speeds=(0.0, 0.0, 0.0, 0.0), model_mono_time_ns=SOURCE_NS)
  position_times = changes.pop('position_times', MODEL_TIMES)
  velocity_times = changes.pop('velocity_times', MODEL_TIMES)
  reference = changes.pop('reference', evidence(raw, 0.0))
  values.update(changes)
  return stationary_no_lead_departure(reference, position, position_times, velocity_times, **values)


def test_b5_exact_arrays_qualify_without_reclassifying_or_changing_raw_unknown():
  reference = evidence(VELOCITY, 0.0)
  before = reference.events, reference.raw_motion, reference.raw_velocity.copy()
  assert qualifies(reference=reference)
  assert reference.raw_motion.kind == 'unknown' and reference.unknown is not None
  assert not reference.allows_departure and not reference.allows_external_departure
  assert reference.events == before[0] and reference.raw_motion == before[1]
  np.testing.assert_array_equal(reference.raw_velocity, before[2])
  np.testing.assert_array_equal(MODEL_TIMES, FIXTURE['time'])
  np.testing.assert_array_equal(MODEL_TIMES, FIXTURE['position_time'])


@pytest.mark.parametrize('changes', [
  {'actual_standstill': False}, {'measured_speed': -0.001}, {'measured_speed': 0.001},
  {'raw_speed': 0.01}, {'raw_speed': -0.01}, {'measured_speed': np.nan},
  {'wheel_speeds': None}, {'wheel_speeds': ()}, {'wheel_speeds': (0.0, 0.0, 0.0)},
  {'wheel_speeds': (0.0, 0.0, 0.0, 0.01)}, {'wheel_speeds': (-0.01, 0.0, 0.0, 0.0)},
  {'wheel_speeds': (0.0, 0.0, np.nan, 0.0)}, {'wheel_speeds': (False, 0.0, 0.0, 0.0)},
  {'model_mono_time_ns': SOURCE_NS + 1}, {'model_mono_time_ns': True},
  {'position_times': MODEL_TIMES + 0.01}, {'velocity_times': MODEL_TIMES[:-1]},
])
def test_actual_movement_missing_wheels_or_source_mismatch_remain_vetoes(changes):
  assert not qualifies(**changes)


@pytest.mark.parametrize('defect', [
  'reverse_speed', 'reverse_integral', 'slow_recovery', 'later_stop', 'later_reverse', 'all_zero',
  'all_negative', 'weak_forward_velocity', 'position_reverse', 'position_later_reverse',
  'weak_position_progress', 'position_origin', 'position_nan', 'velocity_nan', 'missing_position',
])
def test_model_reverse_budgets_and_full_later_trajectory_are_required(defect):
  raw, position = VELOCITY.copy(), POSITION.copy()
  if defect == 'reverse_speed':
    raw[0] = -0.030001
  elif defect == 'reverse_integral':
    raw[MODEL_TIMES <= 0.244140625] = -0.025
  elif defect == 'slow_recovery':
    raw[MODEL_TIMES < 0.625] = -0.003
  elif defect == 'later_stop':
    raw[20:] = 0.0
  elif defect == 'later_reverse':
    raw[-1] = -0.001
  elif defect == 'all_zero':
    raw[:] = 0.0
  elif defect == 'all_negative':
    raw[:] = -0.001
  elif defect == 'weak_forward_velocity':
    raw[MODEL_TIMES <= 2.5] = np.minimum(raw[MODEL_TIMES <= 2.5], 0.01)
  elif defect == 'position_reverse':
    position[5] = -0.005001
  elif defect == 'position_later_reverse':
    position[-1] = position[-2] - 0.01
  elif defect == 'weak_position_progress':
    position = np.minimum(position, 0.001)
  elif defect == 'position_origin':
    position += 0.1
  elif defect == 'position_nan':
    position[-1] = np.nan
  elif defect == 'velocity_nan':
    raw[-1] = np.nan
  elif defect == 'missing_position':
    position = position[:-1]
  assert not qualifies(raw, position)


@pytest.mark.parametrize('model_ns,anchor_ns,now_ns', [
  (None, SOURCE_NS, SOURCE_NS),
  (SOURCE_NS + 1, SOURCE_NS, SOURCE_NS),
  (SOURCE_NS, SOURCE_NS, SOURCE_NS + 250_000_001),
  (SOURCE_NS, SOURCE_NS - 250_000_001, SOURCE_NS),
])
def test_unhealthy_or_expired_source_cannot_reuse_stationary_arrays(model_ns, anchor_ns, now_ns):
  reference = evidence(VELOCITY, 0.0, model_ns=model_ns, anchor_ns=anchor_ns, now_ns=now_ns)
  assert not qualifies(reference=reference)


def held_b5_case():
  planner, owner = make_planner_owner(speed=0.0)
  state = make_handover_state(speed=0.0, target=10.0, experimental=True)
  state['carState'].canValid = True
  state['carState'].gearShifter = 'drive'
  state['modelV2'].action.shouldStop = True
  state['modelV2'].action.desiredAcceleration = -0.2
  for frame in range(20):
    state['controlsState'].longControlState = owner.long_control_state
    plan, outputs = consume(planner, owner, state, frame)
    assert plan.motionIntent.valid and plan.solverStatus == 0 and max(outputs) <= 0.0
  assert owner.motion_phase == MotionPhase.HOLD
  model = state['modelV2']
  model.position.x, model.position.t = POSITION.tolist(), MODEL_TIMES.tolist()
  model.velocity.x, model.velocity.t = VELOCITY.tolist(), MODEL_TIMES.tolist()
  model.acceleration.x = np.gradient(VELOCITY, MODEL_TIMES).tolist()
  model.action.desiredAcceleration = FIXTURE['action']['desiredAcceleration']
  model.action.shouldStop = False
  assert not state['radarState'].leadOne.present and not state['radarState'].leadTwo.present
  return planner, owner, state


def publish_zero_wheels(state):
  state['carState'].wheelSpeeds = {'fl': 0.0, 'fr': 0.0, 'rl': 0.0, 'rr': 0.0}


def test_b5_no_lead_releases_actual_held_owner_with_current_zero_wheels():
  planner, owner, state = held_b5_case()
  publish_zero_wheels(state)
  released = False
  outputs = []
  for frame in range(20, 44):
    state['controlsState'].longControlState = owner.long_control_state
    plan, current = consume(planner, owner, state, frame)
    outputs.extend(current)
    assert plan.motionIntent.valid and plan.solverStatus == 0
    assert plan.motionIntent.referenceKind == MotionReferenceKind.UNKNOWN
    assert planner.reference_planner.model_evidence.raw_motion.kind == 'unknown'
    assert not owner.motion_degraded
    if plan.motionIntent.releaseAllowed:
      released |= not owner.resolved_should_stop
  assert released and not owner.resolved_should_stop and outputs[-1] > 0.0
  assert owner.motion_phase in (MotionPhase.RELEASE, MotionPhase.TRACK)
  assert not state['carState'].gasPressed


@pytest.mark.parametrize('condition', [
  'missing_wheels', 'moving_wheel', 'rollback', 'red_light', 'hard_brake', 'weak_action',
  'later_reverse', 'position_reverse', 'current_lead', 'unhealthy_can', 'reverse_gear', 'brake_pressed',
])
def test_held_owner_keeps_independent_vetoes_for_b5_candidate(condition):
  planner, owner, state = held_b5_case()
  if condition != 'missing_wheels':
    publish_zero_wheels(state)
  if condition == 'moving_wheel':
    state['carState'].wheelSpeeds.fl = 0.01
  elif condition == 'rollback':
    state['carState'].vEgo = -0.001
  elif condition == 'red_light':
    state['modelV2'].action.shouldStop = True
  elif condition == 'hard_brake':
    state['modelV2'].meta.hardBrakePredicted = True
  elif condition == 'weak_action':
    state['modelV2'].action.desiredAcceleration = 0.01
  elif condition == 'later_reverse':
    state['modelV2'].velocity.x = VELOCITY[:-1].tolist() + [-0.001]
  elif condition == 'position_reverse':
    state['modelV2'].position.x = POSITION[:-1].tolist() + [float(POSITION[-2] - 0.01)]
  elif condition == 'current_lead':
    lead = state['radarState'].leadOne
    lead.present, lead.dRel, lead.vLead, lead.vRel, lead.modelProb = True, 10.0, 0.0, 0.0, 0.99
  elif condition == 'unhealthy_can':
    state['carState'].canValid = False
  elif condition == 'reverse_gear':
    state['carState'].gearShifter = 'reverse'
  elif condition == 'brake_pressed':
    state['carState'].brakePressed = True
  for frame in range(20, 44):
    state['controlsState'].longControlState = owner.long_control_state
    plan, outputs = consume(planner, owner, state, frame)
    assert not plan.motionIntent.releaseAllowed
    if condition == 'brake_pressed':
      # The driver's brake transfers authority away from automatic HOLD.
      assert owner.motion_phase == MotionPhase.OFF and all(value == 0.0 for value in outputs)
    else:
      assert owner.resolved_should_stop and max(outputs) <= 0.0
