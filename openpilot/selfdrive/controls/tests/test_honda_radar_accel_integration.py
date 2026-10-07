"""Native UNIFIED -> serialized intent -> LongControl authority checks.

These prescribe measured vehicle motion and DEC admission; they are request
contracts, not closed-loop stopping-distance or road-performance evidence.
The actual observer/association producer has a separate native-wire fixture.
"""
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.car.honda_radar_observer_config import ASSIST, SERVICE
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionPhase, MotionResponseKind, StopReason
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import consume, make_planner_owner, make_state
from openpilot.selfdrive.controls.tests.test_honda_radar_accel_guard import guard_input
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import CruiseTarget, LongitudinalPlannerSP, LongitudinalPlanSource


def cruise_target(_planner, _state, _speed, accel, target):
  return CruiseTarget(LongitudinalPlanSource.cruise, target, accel)


@contextmanager
def pipeline(mode=ASSIST, *, blended=True, measured_accel=0.4, speed=3.0):
  with OpenpilotPrefix():
    params = Params()
    params.put('HondaBoschARadarMode', mode, block=True)
    for key in ('AlphaLongitudinalEnabled', 'ExperimentalMode', 'DynamicExperimentalControl'):
      params.put_bool(key, True, block=True)
    planner, control = make_planner_owner(speed=speed, initial_accel=measured_accel)
    state = make_state(speed=speed, target=12.0, experimental=True)
    state['carState'].aEgo = measured_accel
    state['modelV2'].action.desiredAcceleration = 1.1
    state['modelV2'].position.y = [0.0] * 33
    with patch.object(planner, 'is_e2e', return_value=blended), \
         patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'active', return_value=True), \
         patch.object(LongitudinalPlannerSP, 'update', return_value=None), \
         patch.object(LongitudinalPlannerSP, 'update_targets', new=cruise_target):
      yield planner, control, state


def step(planner, control, state, frame, *, gap=False):
  values = guard_input(frame, gap=gap, step=50_000_000, speed=state['carState'].vEgo)
  event = messaging.new_message(SERVICE)
  event.hondaRadarObservations = values['observation']
  state[SERVICE] = event.hondaRadarObservations.as_reader()
  state.logMonoTime[SERVICE] = values['model_ns']
  state.valid[SERVICE] = True
  state.seen = dict.fromkeys(state, True)
  state['modelV2'].timestampEof = values['model_ns']
  state['modelV2'].leadsV3 = values['model']['leadsV3']
  state['controlsState'].longControlState = control.long_control_state
  plan, outputs = consume(planner, control, state, frame)
  assert plan.solverStatus == 0 and plan.motionIntent.valid
  return plan, outputs


@pytest.mark.parametrize('blended', [False, True])
@pytest.mark.parametrize('initial_i', [-0.3, 0.3])
def test_radar_ceiling_reaches_wire_and_actuator_request_without_positive_i(blended, initial_i):
  with pipeline(blended=blended) as (planner, control, state):
    for frame in range(8):
      step(planner, control, state, frame)
    assert planner.reference_planner.radar_accel_guard.qualified
    control.pid.i = initial_i
    for frame in range(8, 15):
      plan, outputs = step(planner, control, state, frame, gap=True)
      assert planner.reference_planner.radar_accel_guard.holding
      assert planner.reference_planner.radar_accel_limited
      assert not planner.reference_planner.turn_lead_guard.armed
      assert plan.aTarget == 0.0 and plan.reference.constraintAccel == 0.0
      assert plan.motionIntent.responseKind == MotionResponseKind.ABSOLUTE_ACC
      assert not plan.shouldStop and plan.motionIntent.reasons == 0
      assert not plan.motionIntent.releaseAllowed and not plan.motionIntent.movingLead
      assert control.motion_phase == MotionPhase.TRACK
      assert max(outputs) <= 1e-7 and control.pid.i <= 0.0


def test_modes_zero_one_two_are_identical_and_mode_three_changes_actual_request():
  def run(mode):
    result = []
    with pipeline(mode, measured_accel=0.0) as (planner, control, state):
      for frame in range(15):
        plan, outputs = step(planner, control, state, frame, gap=frame >= 8)
        result.append((plan.aTarget, plan.shouldStop, plan.motionIntent.reasons, plan.motionIntent.responseKind, outputs))
    return result

  off, observation, shadow, assist = (run(mode) for mode in range(4))
  assert off == observation == shadow
  assert off[-1][0] > 0.2 and max(off[-1][-1]) > 0.1
  assert assist[-1][0] == 0.0 and max(assist[-1][-1]) <= 1e-7


def test_independent_model_braking_and_stop_intent_cannot_be_relaxed():
  with pipeline() as (planner, control, state):
    for frame in range(8):
      step(planner, control, state, frame)
    state['modelV2'].action.desiredAcceleration = -1.5
    state['modelV2'].action.shouldStop = True
    for frame in range(8, 13):
      plan, outputs = step(planner, control, state, frame, gap=True)
      assert planner.reference_planner.radar_accel_guard.holding
      assert plan.aTarget <= -1.5
      assert plan.shouldStop and plan.motionIntent.reasons & StopReason.E2E
      assert not plan.motionIntent.releaseAllowed
      assert max(outputs) <= 0.0


def test_native_close_lead_keeps_braking_and_stop_authority():
  with pipeline(blended=False) as (planner, control, state):
    for frame in range(8):
      step(planner, control, state, frame)
    lead = state['radarState'].leadOne
    lead.present, lead.dRel, lead.vLead, lead.vRel = True, 2.0, 0.0, -state['carState'].vEgo
    lead.modelProb, lead.aLeadTau = 0.99, 1.5
    for frame in range(8, 13):
      plan, outputs = step(planner, control, state, frame, gap=True)
      assert planner.reference_planner.radar_accel_guard.holding
      assert plan.aTarget < 0.0 and plan.shouldStop
      assert plan.motionIntent.reasons & StopReason.CLOSE_LEAD
      assert not plan.motionIntent.releaseAllowed and max(outputs) <= 0.0


@pytest.mark.parametrize('gate', ['mode', 'dec', 'experimental', 'long_active', 'gas_override'])
def test_runtime_gate_loss_revokes_assist_at_the_next_planner_tick(gate):
  with pipeline() as (planner, control, state):
    for frame in range(9):
      step(planner, control, state, frame, gap=frame == 8)
    assert planner.reference_planner.radar_accel_guard.holding
    if gate == 'mode':
      Params().put('HondaBoschARadarMode', 2, block=True)
    elif gate == 'dec':
      planner.dec.active.return_value = False
    elif gate == 'experimental':
      state['selfdriveState'].experimentalMode = False
    elif gate == 'long_active':
      state['carControl'].longActive = False
    else:
      state['carState'].gasPressed = True
    step(planner, control, state, 9, gap=True)
    guard = planner.reference_planner.radar_accel_guard
    assert not guard.qualified and not guard.holding and guard.accel_ceiling is None


def test_stationary_stop_and_release_eligibility_match_existing_control():
  def run(mode):
    result = []
    with pipeline(mode, measured_accel=0.0, speed=0.0) as (planner, control, state):
      lead = state['radarState'].leadOne
      lead.present, lead.dRel, lead.modelProb, lead.aLeadTau = True, 4.0, 0.99, 1.5
      for frame in range(24):
        if frame == 12:
          lead.dRel, lead.vLead, lead.vRel = 10.0, 2.0, 2.0
          state['modelV2'].velocity.x = [1.0] * 33
        plan, outputs = step(planner, control, state, frame, gap=frame >= 8)
        guard = planner.reference_planner.radar_accel_guard
        assert not guard.qualified and guard.accel_ceiling is None
        if frame < 12:
          assert plan.shouldStop and not plan.motionIntent.releaseAllowed and max(outputs) <= 0.0
        result.append((plan.aTarget, plan.shouldStop, plan.motionIntent.reasons, plan.motionIntent.releaseAllowed,
                       plan.motionIntent.movingLead, plan.motionIntent.responseKind, outputs))
    return result

  assert run(0) == run(ASSIST)


def test_optional_observer_health_cannot_mask_primary_input_faults():
  with OpenpilotPrefix():
    ignored = [SERVICE]
    state = messaging.SubMaster(['carState', 'modelV2', SERVICE], ignore_alive=ignored,
                                ignore_avg_freq=ignored, ignore_valid=ignored)
    for name in ('carState', 'modelV2'):
      state.alive[name] = state.freq_ok[name] = state.valid[name] = True
    state.alive[SERVICE] = state.freq_ok[SERVICE] = state.valid[SERVICE] = False
    assert state.all_checks()
    state.valid['modelV2'] = False
    assert not state.all_checks()
    state.valid['modelV2'] = True
    state.alive['carState'] = False
    assert not state.all_checks()


def test_bounded_events_record_actual_cap_selection_and_recovery():
  with pipeline() as (planner, control, state), \
       patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.cloudlog.event') as events:
    for frame in range(8):
      step(planner, control, state, frame)
    for frame in range(8, 15):
      step(planner, control, state, frame, gap=True)
    records = [call.kwargs for call in events.call_args_list if call.args[0] == 'honda_radar_accel_guard']
    holding = [record for record in records if record['holding']]
    assert len(holding) == 1
    assert holding[0]['qualified'] and holding[0]['ceilingSelected']
    assert holding[0]['accelCeiling'] == holding[0]['aTarget'] == 0.0
    assert holding[0]['sessionId'] == 'fixture' and holding[0]['positionMonoTime'] > 0
    assert holding[0]['sourceTimestampEof'] <= holding[0]['consumerMonoTime']
    events.reset_mock()
    for frame in range(15, 35):
      step(planner, control, state, frame)
    recovery = [call.kwargs for call in events.call_args_list if call.args[0] == 'honda_radar_accel_guard']
    assert len(recovery) == 1
    assert not recovery[0]['holding'] and 0.0 < recovery[0]['accelCeiling'] < 0.1
