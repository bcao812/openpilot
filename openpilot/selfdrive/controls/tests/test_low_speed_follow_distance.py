from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.honda.values import CAR
from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_STOP_DISTANCE, LEAD_DANGER_FACTOR, MODEL_BRAKING_PRIORITY, N, PARAM_DIM, STOP_DISTANCE, T_IDXS,
  LongitudinalMpc, MpcReference, get_safe_obstacle_distance, get_stopped_equivalence_factor, get_T_FOLLOW,
)
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import low_speed_follow_distance
from openpilot.selfdrive.controls.lib.longitudinal_planner import has_safe_moving_lead
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent, MotionPhase, StopReason
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import ControlHarness
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import advance, make_planner, make_state
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_typed_reference_mpc import MODEL_TIMES, typed_reference
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state, solve_reference
from openpilot.selfdrive.controls.tests.test_unified_parking_approach import set_vision_lead


PERSONALITIES = (
  (log.LongitudinalPersonality.relaxed, 1.75),
  (log.LongitudinalPersonality.standard, 1.45),
  (log.LongitudinalPersonality.aggressive, 1.25),
)


@pytest.mark.parametrize('personality,expected', PERSONALITIES)
@pytest.mark.parametrize('with_reference', [False, True])
def test_default_mpc_and_legacy_keep_original_gap(personality, expected, with_reference):
  speed = 10.0 * CV.KPH_TO_MS
  mpc = LongitudinalMpc()
  mpc.set_cur_state(speed, 0.0)
  reference = MpcReference(speed) if with_reference else None
  mpc.update(radar_state(), personality=personality, reference=reference)
  assert mpc.solution_status == 0
  assert get_T_FOLLOW(personality) == expected
  np.testing.assert_array_equal(mpc.params[:, 4], np.full(N+1, expected))
  np.testing.assert_array_equal(mpc.params[:, COMFORT_STOP_DISTANCE], np.full(N+1, 6.0))


@pytest.mark.parametrize('distance', [4.0, 5.0, 6.0, np.float32(4.0)])
@pytest.mark.parametrize('lead_index', [0, 1])
def test_real_lead_comfort_reaches_every_native_stage_without_moving_risk_geometry(distance, lead_index):
  speed = 10.0 * CV.KPH_TO_MS
  radar = radar_state(distance=8.0, speed=speed)
  if lead_index == 1:
    radar.leadOne, radar.leadTwo = radar.leadTwo, radar.leadOne
  mpc = LongitudinalMpc()
  mpc.set_cur_state(speed, 0.0)
  mpc.diagnostic_details_enabled = True
  native = mpc.solver
  writes = {}

  def record_set(stage, field, value):
    if field == 'p':
      writes[stage] = np.array(value, copy=True)
    return native.set(stage, field, value)

  recorded = Mock(wraps=native)
  recorded.set.side_effect = record_set
  reference = MpcReference(30.0, comfort_stop_distance=distance)
  with patch.object(mpc, 'solver', recorded):
    mpc.update(radar, reference=reference)
  assert mpc.solution_status == 0
  assert type(reference.comfort_stop_distance) is float
  assert set(writes) == set(range(N+1))
  expected_raw = 8.0 + speed * T_IDXS + get_stopped_equivalence_factor(speed)
  np.testing.assert_allclose(mpc.params[:, 2], expected_raw)
  for stage, parameters in writes.items():
    assert parameters.shape == (PARAM_DIM,)
    assert parameters[COMFORT_STOP_DISTANCE] == pytest.approx(float(distance))
    assert parameters[4] == 1.45
    assert parameters[5] == LEAD_DANGER_FACTOR
    assert parameters[2] == pytest.approx(expected_raw[stage])
  np.testing.assert_array_equal(mpc.diagnostic_last_parameters, np.array([writes[stage] for stage in range(N+1)]))


@pytest.mark.parametrize('value', [
  True, False, None, np.bool_(True), '4.0', 4.0 + 0j, [4.0], np.array(4.0), np.array([4.0]),
  float('nan'), float('inf'), float('-inf'), 2.999999, 6.000001,
])
def test_invalid_comfort_distance_is_rejected_before_solver(value):
  with pytest.raises(ValueError, match='Invalid comfort stop distance'):
    MpcReference(10.0, comfort_stop_distance=value)


@pytest.mark.parametrize('speed', [0.0, 1.0, 10.0 * CV.KPH_TO_MS, 20.0])
@pytest.mark.parametrize('model', [False, True])
def test_virtual_only_references_remain_exactly_unchanged_with_four_meter_request(speed, model):
  fields = {}
  if model:
    fields = {'model_position': speed * T_IDXS, 'model_velocity': np.full(N+1, speed), 'model_acceleration': np.zeros(N+1)}
  reference = MpcReference(speed, **fields)
  original = solve_reference(speed, reference)
  compact = solve_reference(speed, replace(reference, comfort_stop_distance=4.0))
  for name in ('params', 'yref', 'x_sol', 'u_sol'):
    np.testing.assert_array_equal(getattr(compact, name), getattr(original, name))
  np.testing.assert_array_equal(compact.params[:, COMFORT_STOP_DISTANCE], 6.0)
  # At rest the original virtual obstacle retains its 1.5 m soft-risk margin;
  # shortening that obstacle itself to 4 m would invent a 0.5 m deficit.
  if speed == 0.0:
    np.testing.assert_array_equal(compact.params[:, 2] - 0.75 * 6.0, 1.5)


@pytest.mark.parametrize('virtual_source', ['cruise', 'model'])
def test_virtual_limit_wins_comfort_crossover_without_accelerating_past_its_speed(virtual_source):
  speed = 10.0 * CV.KPH_TO_MS
  radar = radar_state(distance=9.0, speed=speed)
  fields = {}
  if virtual_source == 'model':
    fields = {'model_position': speed * T_IDXS, 'model_velocity': np.full(N+1, speed), 'model_acceleration': np.zeros(N+1)}
  reference = MpcReference(speed if virtual_source == 'cruise' else 30.0, comfort_stop_distance=4.0, **fields)
  compact = solve_reference(speed, reference, radar=radar)
  # 9 m lies between the compact 8.0278 m goal and the original 10.0278 m
  # goal. The lead still owns risk geometry, while the virtual speed cap owns
  # comfort. Treating both minima as the lead would ask for excess speed.
  np.testing.assert_allclose(compact.params[:, 2], 9.0 + speed * T_IDXS + speed**2 / 5.0)
  np.testing.assert_allclose(compact.params[:, COMFORT_STOP_DISTANCE], 9.0 - 1.45 * speed)
  np.testing.assert_allclose(compact.v_solution, speed, atol=1e-4)
  np.testing.assert_allclose(compact.a_solution, 0.0, atol=1e-4)


def test_typed_comfort_source_does_not_change_risk_braking_priority():
  speed = 10.0 * CV.KPH_TO_MS
  reference = replace(typed_reference(np.full(len(MODEL_TIMES), speed), speed, cruise=speed), comfort_stop_distance=4.0)
  compact = solve_reference(speed, reference, radar=radar_state(distance=9.0, speed=speed))
  np.testing.assert_allclose(compact.params[:, COMFORT_STOP_DISTANCE], 9.0 - 1.45 * speed)
  np.testing.assert_array_equal(compact.diagnostic_reference_weights[:, 0], 0.0)
  np.testing.assert_array_equal(compact.params[:, MODEL_BRAKING_PRIORITY], 1.0)
  # The retained risk priority deliberately permits a slower prediction. This
  # contract is the unchanged virtual speed ceiling, not symmetric tracking
  # of a preference after the risk owner has disabled its lower-speed cost.
  assert np.max(compact.v_solution) <= speed + 1e-4


def test_generated_cost_changes_comfort_without_reducing_the_original_risk_row(monkeypatch):
  from acados.acados_template import AcadosModel, AcadosOcp
  from casadi import Function

  monkeypatch.setattr(long_mpc, 'AcadosModel', AcadosModel, raising=False)
  monkeypatch.setattr(long_mpc, 'AcadosOcp', AcadosOcp, raising=False)
  model = long_mpc.gen_long_ocp().model
  evaluate = Function('comfort_and_risk', [model.x, model.u, model.p], [model.cost_y_expr, model.con_h_expr])
  parameters = np.array([-3.5, 2.0, 4.0, 0.0, 1.45, 0.75, 0.0, 0.0, 0.0, 0.0, 6.0, 6.0])
  for speed in (0.0, 2.0, 10.0):
    for distance in (4.0, 5.0, 6.0):
      parameters[COMFORT_STOP_DISTANCE] = distance
      cost, constraints = (np.asarray(value).ravel() for value in evaluate([0.0, speed, 0.0], [0.0], parameters))
      assert cost[0] == pytest.approx((4.0 - speed**2 / 5.0 - 1.45 * speed - distance) / (speed + 10.0))
      assert constraints[3] == pytest.approx((4.0 - 0.75 * (speed**2 / 5.0 + 1.45 * speed + 6.0)) / (speed + 10.0))
  # A 4 m comfort target at rest still has the original risk deficit.
  parameters[COMFORT_STOP_DISTANCE] = 4.0
  cost, constraints = evaluate([0.0, 0.0, 0.0], [0.0], parameters)
  assert float(cost[0]) == 0.0
  assert float(constraints[3]) == pytest.approx(-0.05)


@pytest.mark.parametrize('speed_kph,distance', [(0.0, 4.0), (10.0, 4.0), (15.0, 4.0), (22.5, 5.0), (30.0, 6.0), (50.0, 6.0)])
@pytest.mark.parametrize('personality,follow_time', PERSONALITIES)
def test_unified_producer_selects_comfort_distance_without_changing_personality(speed_kph, distance, personality, follow_time):
  speed = speed_kph * CV.KPH_TO_MS
  assert low_speed_follow_distance(speed) == pytest.approx(distance)
  planner, _ = make_planner(speed=speed)
  state = make_state(speed=speed, target=max(speed, 10.0))
  state['selfdriveState'].personality = personality
  with patch.object(planner.mpc, 'update', wraps=planner.mpc.update) as update:
    planner.update(state, now_ns=advance(state, 0))
  assert planner.unified
  assert planner.mpc.solution_status == 0
  assert update.call_args.kwargs['reference'].comfort_stop_distance == pytest.approx(distance)
  np.testing.assert_array_equal(planner.mpc.params[:, 4], np.full(N+1, follow_time))
  # No real lead: the virtual cruise geometry keeps its original base.
  np.testing.assert_array_equal(planner.mpc.params[:, COMFORT_STOP_DISTANCE], 6.0)


def test_legacy_vehicle_does_not_consume_unified_distance_policy():
  speed = 10.0 * CV.KPH_TO_MS
  planner, _ = make_planner(speed=speed, platform=CAR.HONDA_CIVIC_BOSCH)
  state = make_state(speed=speed)
  state['selfdriveState'].personality = log.LongitudinalPersonality.standard
  with patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.low_speed_follow_distance',
             side_effect=AssertionError('UNIFIED gap policy used by LEGACY')):
    planner.update(state, now_ns=advance(state, 0))
  assert not planner.unified
  assert planner.mpc.solution_status == 0
  np.testing.assert_array_equal(planner.mpc.params[:, 4], np.full(N+1, 1.45))
  np.testing.assert_array_equal(planner.mpc.params[:, COMFORT_STOP_DISTANCE], 6.0)
  assert STOP_DISTANCE == get_safe_obstacle_distance(0.0, 1.45) == 6.0


@pytest.mark.parametrize('with_reference', [False, True])
def test_reused_mpc_restores_default_distance_after_compact_following(with_reference):
  speed = 3.0
  radar = radar_state(distance=8.0, speed=speed)
  mpc = LongitudinalMpc()
  mpc.set_cur_state(speed, 0.0)
  mpc.update(radar, reference=MpcReference(30.0, comfort_stop_distance=4.0))
  assert mpc.solution_status == 0
  np.testing.assert_allclose(mpc.params[:, COMFORT_STOP_DISTANCE], 4.0,
                             rtol=0.0, atol=4 * np.finfo(float).eps * STOP_DISTANCE)
  mpc.update(radar, reference=MpcReference(30.0) if with_reference else None)
  assert mpc.solution_status == 0
  np.testing.assert_array_equal(mpc.params[:, COMFORT_STOP_DISTANCE], 6.0)


def test_compact_rolling_confirmation_retains_the_original_reaction_and_stopping_reserve():
  radar = radar_state(distance=5.6, speed=0.8)
  radar.leadOne.modelProb = 0.999
  radar.leadOne.vRel = -0.2
  assert not has_safe_moving_lead(radar, 1.0, 0.5)  # LEGACY/default 5.75 m budget
  assert has_safe_moving_lead(radar, 1.0, 0.5, minimum_gap_base=4.5)  # 5.25 m budget
  radar.leadOne.dRel = 5.24
  assert not has_safe_moving_lead(radar, 1.0, 0.5, minimum_gap_base=4.5)
  # The qualified 3 m base retains the same 0.55 m reaction and 0.20 m
  # braking reserves. It is not permission to roll at a bare 3 m clearance.
  radar.leadOne.dRel = 3.75
  assert has_safe_moving_lead(radar, 1.0, 0.5, minimum_gap_base=3.0)
  assert not has_safe_moving_lead(radar, 1.0, 0.5)
  radar.leadOne.dRel = 3.749999
  assert not has_safe_moving_lead(radar, 1.0, 0.5, minimum_gap_base=3.0)
  radar.leadOne.dRel = 6.0
  radar.leadOne.vLead = 0.0
  radar.leadOne.vRel = -1.0
  assert not has_safe_moving_lead(radar, 1.0, 0.5, minimum_gap_base=4.5)


@pytest.mark.parametrize('minimum_gap_base', [2.999999, 0.0, -1.0, 5.001, True, float('nan'), float('inf')])
def test_rolling_confirmation_base_cannot_leave_its_declared_range(minimum_gap_base):
  assert not has_safe_moving_lead(radar_state(distance=20.0, speed=1.0), 1.0, 0.5,
                                  minimum_gap_base=minimum_gap_base)


@pytest.mark.parametrize('lead_speed,expected', [(0.384, False), (0.5, True)])
def test_compact_base_preserves_the_original_minimum_moving_lead_speed(lead_speed, expected):
  radar = radar_state(distance=20.0, speed=lead_speed)
  radar.leadOne.vRel = lead_speed - 0.5
  # The E0858-like slow lead is still below the unchanged 0.5 m/s gate.
  assert has_safe_moving_lead(radar, 0.5, 0.5, minimum_gap_base=3.0) is expected


def rolling_budget_run(comfort, risks, present, *, confirmation=False, complete_history=True):
  """Test budget arithmetic after real qualification, replacing only distances."""
  planner, _ = make_planner(speed=1.0)
  planner.CP.radarUnavailable = True
  state = make_state(speed=1.0, target=2.0)
  state['modelV2'].action.desiredAcceleration = 2.0

  def update_frame(frame):
    timestamp = advance(state, frame)
    # Both slots have continuous, exact model/radard identity and ample
    # engineering margin. A .5 m/s decelerating lead fits both the existing
    # moving-lead gate and the new terminal-approach observation domain.
    set_vision_lead(state, timestamp, 150.0 - .5 * frame * .05)
    for index, lead in enumerate((state['radarState'].leadOne, state['radarState'].leadTwo)):
      source = state['modelV2'].leadsV3[index]
      source.v = [.5] * 6
      source.a = [-.05] * 6
      lead.present = present[index]
      lead.vRel = float(np.float32(source.v[0] - state['modelV2'].velocity.x[0]))
      lead.vLead = float(state['carState'].vEgo + lead.vRel)
      lead.aLeadK = float(np.float32(source.a[0]))
      lead.aLeadTau = 1.5
    planner.update(state, now_ns=timestamp)
    assert planner.mpc.solution_status == 0

  first_frame = 120 if complete_history else 0
  for frame in range(first_frame):
    update_frame(frame)
  policy = planner.reference_planner.stop_distance_policy
  update_policy = policy.update

  def declared_distances(**kwargs):
    update_policy(**kwargs)
    return comfort, risks

  requests, intents = [], []
  with patch.object(policy, 'update', side_effect=declared_distances), \
       patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.has_safe_moving_lead',
             return_value=confirmation) as confirmation_check:
    for frame in range(first_frame, first_frame + 4):
      update_frame(frame)
      producer = planner.reference_planner
      decision = producer.parking_approach.decision
      if any(present) and complete_history:
        assert policy.evidence.low_speed_history and decision.uncertainty_qualified
        assert decision.eligible and decision.margin >= 0.0 and decision.allow_compact
        assert producer.parking_mpc_distances == (comfort, risks)
      else:
        assert not decision.allow_compact
        assert producer.parking_mpc_distances == (4.0, (6.0, 6.0))
      requests.append(planner.output_a_target)
      intents.append(planner.motion_intent)
  assert confirmation_check.called
  return confirmation_check.call_args_list, requests, intents


@pytest.mark.parametrize('comfort,risks,present,expected_base', [
  (3.0, (4.0, 6.0), (True, False), 3.0),
  (3.0, (6.0, 4.0), (False, True), 3.0),
  (3.0, (4.0, 6.0), (True, True), 4.5),
  (3.0, (5.0, 4.0), (True, True), 3.75),
  (4.0, (6.0, 6.0), (True, False), 4.5),
  (5.0, (6.0, 6.0), (True, False), 5.0),
  (6.0, (6.0, 6.0), (True, False), 5.0),
  (3.0, (4.0, 4.0), (False, False), 4.5),
])
def test_unified_rolling_budget_uses_only_present_slots_and_the_strictest_risk(comfort, risks, present, expected_base):
  calls, _, _ = rolling_budget_run(comfort, risks, present)
  assert all(call.kwargs['minimum_gap_base'] == pytest.approx(expected_base) for call in calls)


def test_mocked_compact_distance_cannot_bypass_incomplete_approach_history():
  # The budget fixture does not mock the new admission gate. Source/quality
  # withdrawal and five-second recovery are separately exercised through ULP
  # by test_uncertainty_failure_after_native_compact_use_rebuilds_full_history_in_producer.
  calls, _, _ = rolling_budget_run(3.0, (4.0, 6.0), (True, False), complete_history=False)
  assert all(call.kwargs['minimum_gap_base'] == pytest.approx(4.5) for call in calls)


def test_rolling_confirmation_changes_no_numeric_producer_request():
  _, without_confirmation, original_intents = rolling_budget_run(3.0, (4.0, 6.0), (True, False))
  _, with_confirmation, confirmed_intents = rolling_budget_run(3.0, (4.0, 6.0), (True, False), confirmation=True)
  assert any(intent.moving_lead for intent in confirmed_intents)
  assert not any(intent.moving_lead for intent in original_intents)
  np.testing.assert_allclose(with_confirmation, without_confirmation, rtol=0.0, atol=1e-10)
  assert [intent.release_allowed for intent in confirmed_intents] == [intent.release_allowed for intent in original_intents]


def test_compact_rolling_confirmation_does_not_release_hold_or_grant_positive_request():
  radar = radar_state(distance=3.8, speed=0.8)
  radar.leadOne.vRel = -0.2
  confirmation = has_safe_moving_lead(radar, 1.0, 0.5, minimum_gap_base=3.0)
  assert confirmation
  owner = ControlHarness()
  owner.hold()
  intent = MotionIntent(moving_lead=confirmation, release_allowed=False, allow_creep=False)
  for _ in range(10):
    output = owner.step(0.5, intent)
    assert owner.control.motion_phase == MotionPhase.HOLD
    assert owner.control.motion_supervisor.reasons & StopReason.HOLD
    assert output <= 0.0
