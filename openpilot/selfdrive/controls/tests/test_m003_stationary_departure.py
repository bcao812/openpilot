"""M003 snapshots exercise current departure qualification, not vehicle timing.

The raw model and formal lead values come from two recorded diagnostic inputs.
Integration uses a synthetic held history, fresh 20 Hz observations and the real
MPC/planner/owner. It does not replay the recording's hidden controller history.
"""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.drive_helpers import CREEP_STOP_T, STOP_RELEASE_ACCEL, STOP_SPEED
from openpilot.selfdrive.controls.lib.lead_qualification import lead_identity, qualified_lead
from openpilot.selfdrive.controls.lib.model_reference import stationary_lead_departure_compatible
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent, MotionPhase, MotionReferenceKind
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import departing_lead_rejection, has_departing_lead
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import consume, make_handover_state, make_planner_owner
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_motion_supervisor import MotionHarness
from openpilot.selfdrive.controls.tests.test_typed_model_reference import MODEL_TIMES, evidence


FIXTURE = json.loads((Path(__file__).with_name('fixtures') / 'm003_stationary_departure.json').read_text())
FIRST, SECOND = FIXTURE['points']
STOP_MODEL = FIXTURE['stop_model_snapshot']
SHORT_PERMISSION_NS = 313_362_239


def recorded_reference(point):
  return evidence(point['model']['velocity']['x'], point['car']['vEgo'],
                  model_ns=point['input_mono_ns']['modelV2'], anchor_ns=point['input_mono_ns']['carState'],
                  now_ns=point['decision_mono_ns'])


def radar_message(point, model_ns=None):
  data = deepcopy(point['radar'])
  if model_ns is not None:
    # These are explicitly synthetic new observations with the recorded shape
    # and relative sensor ages, not newer publications of an old measurement.
    shift = model_ns - point['input_mono_ns']['modelV2']
    data['mdMonoTime'] += shift
    for slot in ('leadOne', 'leadTwo'):
      for field, value in data[slot]['evidence'].items():
        if field.endswith('MonoTime') and value:
          data[slot]['evidence'][field] = value + shift
  return log.RadarState.new_message(**data)


@pytest.mark.parametrize('point', [FIRST, SECOND], ids=['1881.868', '1903.524'])
def test_exact_m003_model_keeps_unknown_while_current_leads_may_supply_departure(point):
  reference = recorded_reference(point)
  before = reference.events, reference.raw_motion
  raw = np.asarray(point['model']['velocity']['x'])
  assert stationary_lead_departure_compatible(reference, actual_standstill=True, measured_speed=point['car']['vEgo'])
  assert reference.unknown.reason == 'backward_then_forward_is_ambiguous'
  assert reference.raw_motion.kind == 'unknown' and reference.moving is None
  assert not reference.allows_departure and not reference.allows_external_departure
  assert (reference.events, reference.raw_motion) == before
  np.testing.assert_array_equal(reference.raw_velocity, raw)
  np.testing.assert_array_equal(point['model']['velocity']['t'], MODEL_TIMES)
  if point is FIRST:
    assert np.interp(reference.source.proxy_age_s + 2.5, MODEL_TIMES, raw) == pytest.approx(0.1058643803858581)
    assert np.max(raw[MODEL_TIMES <= 2.5]) < STOP_SPEED
    radar = radar_message(point)
    assert all(qualified_lead(lead, 'departure', now_ns=point['decision_mono_ns'])
               for lead in (radar.leadOne, radar.leadTwo))


@pytest.mark.parametrize('swap_slots', [False, True])
def test_exact_second_departure_does_not_replace_unqualified_radar_with_faster_vision(swap_slots):
  radar = radar_message(SECOND)
  if swap_slots:
    first, second = radar.leadOne.to_dict(), radar.leadTwo.to_dict()
    radar.leadOne, radar.leadTwo = second, first
  slow = radar.leadTwo if swap_slots else radar.leadOne
  fast = radar.leadOne if swap_slots else radar.leadTwo
  now_ns = SECOND['decision_mono_ns']
  assert slow.vLead == pytest.approx(0.11296718567609787)
  assert fast.vLead == pytest.approx(0.6323410868644714)
  assert lead_identity(slow) == (SECOND['radar']['fusionSessionId'], 1459)
  assert lead_identity(fast) is None
  assert qualified_lead(fast, 'departure', now_ns=now_ns)
  assert not qualified_lead(slow, 'departure', now_ns=now_ns)
  assert departing_lead_rejection(radar, 0.0, 0.5, now_ns=now_ns) == ('lead_departure_unqualified', lead_identity(slow))


def owner_feedback(owner, state):
  state['controlsState'].longControlState = owner.long_control_state
  motion = state['controlsState'].motionState
  motion.phase = owner.motion_phase.text
  motion.resolvedStop = owner.resolved_should_stop
  motion.degraded = owner.motion_degraded
  motion.waitingForFreshInput = owner.motion_waiting_for_fresh_input


@pytest.fixture
def held_case():
  for name in ('AlphaLongitudinalEnabled', 'ExperimentalMode', 'DynamicExperimentalControl'):
    Params().put_bool(name, True, block=True)
  planner, owner = make_planner_owner(speed=0.0)
  state = make_handover_state(speed=0.0, target=10.0, experimental=True)
  state['modelV2'].action.shouldStop = True
  state['modelV2'].action.desiredAcceleration = -0.2
  # Fix the declared DEC mode only. Native solve, all current lead/model risk
  # checks, low-speed MPC admission and the real owner remain in this path.
  with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'active', return_value=True), \
       patch.object(planner.dec, 'mode', return_value='blended'):
    for frame in range(20):
      owner_feedback(owner, state)
      plan, outputs = consume(planner, owner, state, frame)
      assert plan.solverStatus == 0 and plan.motionIntent.valid
      assert not owner.motion_degraded and max(outputs) <= 0.0
    assert owner.motion_phase == MotionPhase.HOLD and owner.resolved_should_stop
    yield planner, owner, state


def load_snapshot(state, point, frame, *, model_source=None):
  now_ns = 1_000_000_000 + frame * 50_000_000
  model_source = point if model_source is None else model_source
  model = state['modelV2']
  for field in ('position', 'velocity', 'acceleration', 'action'):
    setattr(model, field, deepcopy(model_source['model'][field]))
  model.meta.hardBrakePredicted = model_source['model']['meta']['hardBrakePredicted']
  model.frameId = model_source['model']['frameId'] + frame
  model.timestampEof = now_ns - (model_source['input_mono_ns']['modelV2'] - model_source['model']['timestampEof'])
  for field, value in point['car'].items():
    setattr(state['carState'], field, value)
  state['radarState'] = radar_message(point, now_ns)


def step(case, frame):
  planner, owner, state = case
  owner_feedback(owner, state)
  return consume(planner, owner, state, frame)


def assert_held(owner, outputs):
  assert not owner.motion_degraded
  assert owner.resolved_should_stop and owner.motion_phase == MotionPhase.HOLD
  assert max(outputs) <= 0.0


def test_exact_weak_model_grants_current_lead_permission_before_the_broader_launch_path(held_case):
  planner, owner, state = held_case
  for frame in range(20, 25):
    load_snapshot(state, FIRST, frame)
    plan, outputs = step(held_case, frame)
    reference = planner.reference_planner
    assert plan.solverStatus == 0 and plan.motionIntent.valid and plan.motionIntent.releaseAllowed
    assert reference.fusion_admitted and not reference.radar_consumption.degraded
    assert not reference.lead_launch_active
    assert plan.motionIntent.referenceKind == MotionReferenceKind.UNKNOWN
    assert reference.model_evidence.raw_motion.kind == 'unknown'
    assert not reference.model_evidence.allows_departure and not reference.model_evidence.allows_external_departure
    assert 0.0 < plan.aTarget < STOP_RELEASE_ACCEL
    assert plan.aTarget <= state['modelV2'].action.desiredAcceleration
    assert 0.0 < owner.motion_supervisor.release_lead_t < CREEP_STOP_T
    assert_held(owner, outputs)


@pytest.mark.parametrize('condition', ['no_lead', 'negative_action', 'hard_brake', 'model_stop',
                                      'unqualified_second', 'slow_second', 'mixed_second_departure'])
def test_weak_model_compatibility_does_not_replace_current_departure_guards(held_case, condition):
  planner, owner, state = held_case
  for frame in range(20, 25):
    point = SECOND if condition == 'mixed_second_departure' else FIRST
    load_snapshot(state, point, frame, model_source=STOP_MODEL if condition == 'model_stop' else None)
    model, radar = state['modelV2'], state['radarState']
    if condition == 'no_lead':
      radar.leadOne.present = radar.leadTwo.present = False
    elif condition == 'negative_action':
      model.action.desiredAcceleration = -0.01
    elif condition == 'hard_brake':
      model.meta.hardBrakePredicted = True
    elif condition == 'model_stop':
      # Exact model at 1883.754283342 s, the start of M003's genuine stop
      # interval. Pair it with departing leads to isolate the model veto.
      assert model.action.desiredAcceleration > 0.0
      assert model.action.shouldStop
    elif condition == 'unqualified_second':
      radar.leadTwo.evidence.departureQualified = False
    elif condition == 'slow_second':
      radar.leadTwo.vLead = radar.leadTwo.vLeadK = radar.leadTwo.vRel = SECOND['radar']['leadOne']['vLead']
    plan, outputs = step(held_case, frame)
    assert plan.solverStatus == 0 and plan.motionIntent.valid
    assert not plan.motionIntent.releaseAllowed and not planner.reference_planner.lead_launch_active
    assert owner.motion_supervisor.release_lead_t == 0.0
    assert_held(owner, outputs)
    if condition in ('unqualified_second', 'slow_second', 'mixed_second_departure'):
      assert not has_departing_lead(radar, max(state['carState'].vEgo, 0.0), planner.CP.longitudinalActuatorDelay,
                                    now_ns=state.logMonoTime['modelV2'] + 500_000)


@pytest.mark.parametrize('withdrawal', ['model_stop', 'lead_qualification'])
def test_313ms_weak_permission_cannot_spend_owner_confirmation_after_withdrawal(held_case, withdrawal):
  planner, owner, state = held_case
  load_snapshot(state, FIRST, 20)
  allowed, _ = step(held_case, 20)
  assert allowed.motionIntent.releaseAllowed and not planner.reference_planner.lead_launch_active
  assert 0.0 < allowed.aTarget < STOP_RELEASE_ACCEL
  permission = MotionIntent.from_message(allowed.motionIntent)

  load_snapshot(state, FIRST, 21)
  if withdrawal == 'model_stop':
    state['modelV2'].action.shouldStop = True
  else:
    state['radarState'].leadTwo.evidence.departureQualified = False
  blocked, _ = step(held_case, 21)
  assert blocked.solverStatus == 0 and blocked.motionIntent.valid and not blocked.motionIntent.releaseAllowed
  assert not owner.motion_degraded
  load_snapshot(state, FIRST, 22)
  restored, _ = step(held_case, 22)
  assert restored.motionIntent.releaseAllowed and not planner.reference_planner.lead_launch_active
  assert 0.0 < restored.aTarget < STOP_RELEASE_ACCEL

  # Real producer intents, controlled fresh publication/consumption times.
  # This duration is the recorded short permission window, not a replay of
  # its individual publication cadence or its measured vehicle response.
  harness = MotionHarness()
  harness.stopped()
  started = harness.now_ns
  for tick in range(31):
    fresh = tick % 5 == 0
    assert harness.step(permission, a_target=allowed.aTarget, v_ego=0.0, standstill=True,
                        new_plan=fresh, new_model=fresh)
  assert harness.step(permission, a_target=allowed.aTarget, v_ego=0.0, standstill=True,
                      new_plan=False, new_model=False, advance_ns=SHORT_PERMISSION_NS - 310_000_000)
  assert harness.now_ns - started == SHORT_PERMISSION_NS
  assert 0.0 < harness.supervisor.release_lead_t < CREEP_STOP_T
  assert harness.supervisor.phase == MotionPhase.HOLD
  assert harness.step(MotionIntent.from_message(blocked.motionIntent), a_target=blocked.aTarget, v_ego=0.0, standstill=True)
  assert harness.supervisor.release_lead_t == 0.0

  started = harness.now_ns
  restored_intent = MotionIntent.from_message(restored.motionIntent)
  release_ns = None
  for tick in range(60):
    fresh = tick % 5 == 0
    held = harness.step(restored_intent, a_target=restored.aTarget, v_ego=0.0, standstill=True,
                        new_plan=fresh, new_model=fresh)
    if harness.now_ns - started < round(CREEP_STOP_T * 1e9):
      assert held and harness.supervisor.phase == MotionPhase.HOLD
    if not held and release_ns is None:
      release_ns = harness.now_ns
  assert release_ns is not None and release_ns - started >= round(CREEP_STOP_T * 1e9)
  assert harness.supervisor.phase == MotionPhase.RELEASE
