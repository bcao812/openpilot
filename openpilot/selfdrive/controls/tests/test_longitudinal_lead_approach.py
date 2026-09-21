"""Earlier lead response may change MPC smoothing, never controller permission."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.honda.values import CAR
from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  LEAD_DANGER_FACTOR, get_T_FOLLOW, get_safe_obstacle_distance, get_stopped_equivalence_factor,
)
from openpilot.selfdrive.controls.lib.longitudinal_planner import (
  LEAD_BRAKING_MIN_CHANGE_SCALE, LongitudinalPlanSource, get_lead_approach_urgency, get_lead_braking_urgency,
)
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionResponseKind
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import advance, consume, make_planner, make_planner_owner, make_state
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_motion_response_contract import ordinary_policy as ordinary_policy


PERSONALITIES = (log.LongitudinalPersonality.aggressive, log.LongitudinalPersonality.standard, log.LongitudinalPersonality.relaxed)


def approach_radar(speed=25.0, gap=54.0, closing=5.6):
  lead = SimpleNamespace(present=True, dRel=gap, vRel=-closing, vLead=speed - closing,
                         aLeadK=0.0, aLeadTau=1.5, yRel=0.0, modelProb=0.999)
  absent = deepcopy(lead)
  absent.present = False
  return SimpleNamespace(leadOne=lead, leadTwo=absent)


def urgency(radar, speed=25.0, delay=0.5, personality=log.LongitudinalPersonality.standard):
  return get_lead_approach_urgency(speed, radar, delay, personality=personality)


def install_leads(state, radar):
  state['radarState'].leadOne = vars(radar.leadOne)
  state['radarState'].leadTwo = vars(radar.leadTwo)
  state['selfdriveState'].personality = log.LongitudinalPersonality.standard


@pytest.mark.parametrize('speed,gap,closing', [(25.0, 54.0, 5.6), (26.9, 74.7, 4.4)])
def test_recorded_approach_geometry_is_eligible_before_the_target_has_started_braking(speed, gap, closing):
  # Rounded b/c samples with a positive previous target. A required-decel-only
  # trigger remains zero here even if its negative-target gate were removed.
  radar = approach_radar(speed, gap, closing)
  assert get_lead_braking_urgency(speed, 0.05, radar, 0.5) == 0.0
  assert get_lead_braking_urgency(speed, -1.0, radar, 0.5) == 0.0
  assert 0.0 < urgency(radar, speed) <= 1.0


@pytest.mark.parametrize('personality', PERSONALITIES)
def test_weight_blends_between_existing_comfort_and_danger_geometry(personality):
  speed, closing, delay = 20.0, 5.0, 0.5
  safe = get_safe_obstacle_distance(speed, get_T_FOLLOW(personality))
  lead_stop = get_stopped_equivalence_factor(speed - closing)
  danger, comfort = LEAD_DANGER_FACTOR * safe - lead_stop, safe - lead_stop
  reservation = closing * (delay + DT_MDL)
  for gap, expected in ((danger - 1.0, 1.0), (danger, 1.0), ((danger + comfort) / 2.0, 0.5), (comfort, 0.0), (comfort + 1.0, 0.0)):
    radar = approach_radar(speed, gap + reservation, closing)
    assert urgency(radar, speed, delay, personality) == pytest.approx(expected)
  for boundary in (danger, comfort):
    below = approach_radar(speed, boundary + reservation - 1e-7, closing)
    above = approach_radar(speed, boundary + reservation + 1e-7, closing)
    assert abs(urgency(below, speed, delay, personality) - urgency(above, speed, delay, personality)) < 1e-6


def test_more_relaxed_personality_responds_earlier_at_the_same_geometry():
  radar = approach_radar(20.0, 65.0, 5.0)
  values = [urgency(radar, 20.0, personality=personality) for personality in PERSONALITIES]
  assert 0.0 < values[0] < values[1] < values[2] < 1.0


@pytest.mark.parametrize('speed,expected', [(5.0, 0.0), (7.5, 0.5), (10.0, 1.0)])
def test_low_speed_fade_preserves_stopping_and_launch_weights(speed, expected):
  assert urgency(approach_radar(speed, 1.0, 3.0), speed) == pytest.approx(expected)


@pytest.mark.parametrize('closing,expected', [(0.0, 0.0), (1.0, 0.0), (2.0, 0.5), (3.0, 1.0)])
def test_closing_speed_fade_does_not_change_same_speed_following(closing, expected):
  assert urgency(approach_radar(20.0, 1.0, closing), 20.0) == pytest.approx(expected)


def test_speed_and_closing_fades_are_continuous():
  for speed in (5.0, 10.0):
    values = [urgency(approach_radar(varied, 1.0, 3.0), varied) for varied in (speed - 1e-7, speed + 1e-7)]
    assert abs(values[1] - values[0]) < 1e-6
  for closing in (1.0, 3.0):
    values = [urgency(approach_radar(20.0, 1.0, varied), 20.0) for varied in (closing - 1e-7, closing + 1e-7)]
    assert abs(values[1] - values[0]) < 1e-6


def test_delay_reservation_can_raise_urgency_without_changing_the_lead():
  radar = approach_radar(20.0, 72.0, 5.0)
  before = deepcopy(radar)
  values = [urgency(radar, 20.0, delay=float(delay)) for delay in np.linspace(0.0, 1.0, 21)]
  assert values[0] == 0.0 and values[-1] > 0.0
  assert all(0.0 <= value <= 1.0 for value in values)
  assert all(after >= before for before, after in zip(values, values[1:], strict=False))
  assert radar == before


def test_worst_present_lead_owns_the_weight_without_rewriting_inputs():
  radar = approach_radar(20.0, 55.0, 5.0)
  primary = urgency(radar, 20.0)
  radar.leadTwo = approach_radar(20.0, 45.0, 5.0).leadOne
  before = deepcopy(radar)
  assert primary < urgency(radar, 20.0) < 1.0
  assert radar == before
  second = urgency(radar, 20.0)
  radar.leadOne.present = False
  assert urgency(radar, 20.0) == second
  radar.leadOne = approach_radar(20.0, 1.0, 5.0).leadOne
  assert urgency(radar, 20.0) == 1.0


@pytest.mark.parametrize('condition', ['absent', 'distant', 'opening', 'matching_speed'])
def test_unconstrained_lead_keeps_normal_weights(condition):
  radar = approach_radar()
  if condition == 'absent':
    radar.leadOne.present = False
  elif condition == 'distant':
    radar.leadOne.dRel = 150.0
  else:
    radar.leadOne.vRel = 0.0 if condition == 'matching_speed' else 2.0
    radar.leadOne.vLead = 25.0 + radar.leadOne.vRel
  assert urgency(radar) == 0.0


@pytest.mark.parametrize('field,value', [
  ('dRel', 0.0), ('dRel', -1.0), ('dRel', float('nan')), ('dRel', float('inf')), ('dRel', True),
  ('vRel', float('nan')), ('vRel', float('inf')), ('vLead', float('nan')), ('vLead', 30.0),
  ('aLeadK', float('nan')), ('modelProb', 0.94), ('modelProb', 1.01), ('modelProb', float('nan')),
])
def test_unreliable_or_inconsistent_lead_cannot_reduce_smoothing(field, value):
  radar = approach_radar()
  setattr(radar.leadOne, field, value)
  assert urgency(radar) == 0.0


@pytest.mark.parametrize('probability', [0.95, 1.0])
def test_confidence_boundary_is_inclusive(probability):
  radar = approach_radar()
  radar.leadOne.modelProb = probability
  assert urgency(radar) > 0.0


def test_reversing_lead_is_rejected_and_small_stationary_speed_error_matches_native_clipping():
  assert urgency(approach_radar(20.0, 20.0, 20.6), 20.0) == 0.0
  assert urgency(approach_radar(20.0, 20.0, 20.5), 20.0) == 1.0


@pytest.mark.parametrize('speed', [-1.0, 0.0, 1.2, 5.0, 1e300, float('nan'), float('inf'), True, None])
def test_invalid_or_low_ego_speed_cannot_reduce_smoothing(speed):
  assert urgency(approach_radar(), speed=speed) == 0.0


@pytest.mark.parametrize('delay', [-0.1, 1.01, float('nan'), float('inf'), True, None])
def test_invalid_delay_cannot_reduce_smoothing(delay):
  assert urgency(approach_radar(), delay=delay) == 0.0


@pytest.mark.parametrize('personality', [-1, 3, True, None, 'unknown'])
def test_unknown_personality_cannot_invent_a_following_gap(personality):
  assert urgency(approach_radar(), personality=personality) == 0.0


def test_positive_previous_target_reaches_native_weights_without_changing_risk_geometry():
  planner, _ = make_planner(speed=25.0, initial_accel=0.05)
  state = make_state(speed=25.0, target=28.0)
  install_leads(state, approach_radar())
  before = deepcopy(state['radarState'].to_dict())
  planner.mpc.set_weights = Mock(wraps=planner.mpc.set_weights)
  assert planner.output_a_target > 0.0
  planner.update(state, now_ns=advance(state, 0))
  scale = planner.mpc.set_weights.call_args.kwargs['a_change_scale']
  assert LEAD_BRAKING_MIN_CHANGE_SCALE <= scale < 1.0
  assert planner.mpc.solution_status == 0
  assert planner.mpc.x0[1] == 25.0
  assert np.all(planner.mpc.params[:, 5] == LEAD_DANGER_FACTOR)
  assert np.all(planner.mpc.params[:, 4] == get_T_FOLLOW(state['selfdriveState'].personality))
  assert np.min(planner.mpc.params[:, 0]) == -3.5
  assert -3.5 <= planner.output_a_target <= 2.0
  assert state['radarState'].to_dict() == before


@pytest.mark.parametrize('condition', [
  'gas', 'brake', 'override', 'inactive', 'disabled', 'control_disabled', 'force_decel', 'unhealthy', 'stock_long', 'other_platform',
])
def test_early_approach_retains_mode_health_and_driver_gates(condition):
  platform = CAR.HONDA_CIVIC_BOSCH if condition == 'other_platform' else CAR.HONDA_ACCORD
  planner, _ = make_planner(speed=25.0, initial_accel=0.05, platform=platform, stock=condition == 'stock_long')
  state = make_state(speed=25.0, target=28.0)
  install_leads(state, approach_radar())
  if condition in ('gas', 'brake'):
    setattr(state['carState'], condition + 'Pressed', True)
  elif condition == 'override':
    state['carControl'].cruiseControl.override = True
  elif condition == 'inactive':
    state['carControl'].longActive = False
  elif condition == 'disabled':
    state['selfdriveState'].enabled = False
  elif condition == 'control_disabled':
    state['carControl'].enabled = False
  elif condition == 'force_decel':
    state['controlsState'].forceDecel = True
  elif condition == 'unhealthy':
    state.all_checks = lambda service_list=None: False
  planner.mpc.set_weights = Mock(wraps=planner.mpc.set_weights)
  planner.update(state, now_ns=advance(state, 0))
  calls = planner.mpc.set_weights.call_args_list
  assert all(call.kwargs.get('a_change_scale', 1.0) == 1.0 for call in calls)
  if condition == 'unhealthy':
    assert not planner.motion_intent.valid
  else:
    assert calls


@pytest.mark.parametrize('selected', [0, 1])
def test_approach_weight_does_not_grant_lead_braking_permission(selected, ordinary_policy):
  planner, controller = make_planner_owner(initial_accel=0.05)
  state = make_state(experimental=True)
  state['modelV2'].velocity.x[16] = 0.0
  radar = approach_radar(20.0, 45.0, 3.0)
  if selected:
    radar.leadOne, radar.leadTwo = radar.leadTwo, radar.leadOne
  install_leads(state, radar)
  assert urgency(radar, 20.0) > 0.0
  assert get_lead_braking_urgency(20.0, -0.75, radar, 0.5, lead_index=selected) == 0.0
  native_update = planner.mpc.update

  def update(*args, **kwargs):
    native_update(*args, **kwargs)
    planner.mpc.source = (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1)[selected]
    planner.mpc.crash_cnt = 0

  with patch.object(planner, 'is_e2e', return_value=True), patch.object(planner.mpc, 'update', side_effect=update), \
       patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.sample_reference_acceleration', return_value=-0.75):
    plan, _ = consume(planner, controller, state, 0)
  assert plan.solverStatus == 0
  assert plan.aTarget == -0.75
  assert plan.motionIntent.responseKind == MotionResponseKind.NONE
