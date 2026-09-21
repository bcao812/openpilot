"""Keep B's strong-departure evidence without replacing R's action sampler.

The fixed sample is R's 5.10 s publication from audit_held_lead_departure,
variant d03_t04, in longitudinal-fidelity-2026-09-20/f-closed-v9. It is an
interpretation counterexample, not a substitute for closed-loop validation.
"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from openpilot.cereal import messaging
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib import drive_helpers
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, CREEP_STOP_T, STOP_RELEASE_ACCEL, get_accel_from_plan
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent, MotionPhase, MotionReferenceKind, StopReason
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import advance, make_planner, make_state
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import ControlsHarness
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_motion_supervisor import MotionHarness
from openpilot.selfdrive.controls.tests.test_unified_action_sampling import solve_given_acceleration
from openpilot.selfdrive.modeld.constants import ModelConstants


CONTROL_TIMES = np.asarray(ModelConstants.T_IDXS[:CONTROL_N])
RECORDED_SPEEDS = [
  0.0, 0.00013176204811315984, 0.0005270481924526393, 0.0021126139909029007,
  0.006469395477324724, 0.012070971541106701, 0.025297407060861588, 0.04436603561043739,
  0.06636829674243927, 0.10758544504642487, 0.1536516696214676, 0.21025656163692474,
  0.28414008021354675, 0.3644482493400574, 0.4579436779022217, 0.5618635416030884, 0.6729503273963928,
]
RECORDED_ACCELS = [
  0.0, 0.0037947469390928745, 0.015178987756371498, 0.03349243476986885,
  0.057608719915151596, 0.08861536532640457, 0.12161311507225037, 0.1579708755016327,
  0.19992214441299438, 0.23294730484485626, 0.269857794046402, 0.3039174973964691,
  0.3271467983722687, 0.35239604115486145, 0.3639092445373535, 0.3681233525276184, 0.3726280927658081,
]
RECORDED_TARGET = 0.1784430891275406
RECORDED_STRENGTH = 0.20037505106492476


def departure_strength(speeds=RECORDED_SPEEDS, accels=RECORDED_ACCELS, times=CONTROL_TIMES,
                       action_t=0.55, constraint_accel=1.6):
  return drive_helpers.get_departure_accel_from_plan(speeds, accels, times, action_t, constraint_accel)


def held_controls(*, cruise_standstill=False):
  harness = ControlsHarness()
  harness.control.CP.longitudinalActuatorDelay = 0.5
  harness.hold()
  # An owned physical stop and the stock cruise protocol's stop bit are
  # separate. Accord openpilot longitudinal normally has only the former.
  harness.car_state.cruiseState.standstill = cruise_standstill
  harness.plan.aTarget = RECORDED_TARGET
  harness.plan.speeds = RECORDED_SPEEDS
  harness.plan.accels = RECORDED_ACCELS
  harness.plan.reference.accel = RECORDED_TARGET
  harness.plan.reference.constraintAccel = 1.6
  harness.plan.motionIntent.reasons = 0
  harness.plan.motionIntent.releaseAllowed = True
  return harness


def test_recorded_r_publication_releases_hold_at_bs_strength_without_changing_execution_target():
  assert get_accel_from_plan(RECORDED_SPEEDS, RECORDED_ACCELS, CONTROL_TIMES, 0.55) == pytest.approx(RECORDED_STRENGTH)
  assert RECORDED_TARGET < STOP_RELEASE_ACCEL < RECORDED_STRENGTH
  harness = held_controls()
  command, motion = harness.step()
  assert motion.phase == 'release' and not motion.resolvedStop
  assert motion.referenceTimingValid and not motion.degraded
  assert harness.control.LoC.motion_supervisor._last_target == pytest.approx(RECORDED_TARGET)
  assert -2.0 < command.actuators.accel <= -2.0 + 0.08 + 1e-6


def test_strong_departure_preserves_stock_cruise_protocol_hold_until_its_stop_bit_clears():
  harness = held_controls(cruise_standstill=True)
  command, motion = harness.step()
  assert motion.phase == 'release' and not motion.resolvedStop
  assert not motion.degraded and command.longActive
  assert command.actuators.accel == -2.0
  harness.car_state.cruiseState.standstill = False
  command, motion = harness.step(new_plan=False)
  assert motion.phase == 'release' and not motion.resolvedStop
  assert -2.0 < command.actuators.accel <= -2.0 + 0.08 + 1e-6


@pytest.mark.parametrize('delay,releases', [(0.1, False), (0.5, True)])
def test_consumer_strength_uses_the_producers_cp_action_horizon(delay, releases):
  harness = held_controls()
  harness.control.CP.longitudinalActuatorDelay = delay
  _, motion = harness.step()
  expected = get_accel_from_plan(RECORDED_SPEEDS, RECORDED_ACCELS, CONTROL_TIMES, delay + DT_MDL)
  assert (expected >= STOP_RELEASE_ACCEL) == releases
  assert motion.resolvedStop != releases
  assert not motion.degraded


@pytest.mark.parametrize('action_t', [0.05, 0.25, 0.55, 0.85])
def test_strength_retains_the_existing_velocity_increment_and_stable_short_delay_rule(action_t):
  expected = get_accel_from_plan(RECORDED_SPEEDS, RECORDED_ACCELS, CONTROL_TIMES, action_t)
  assert departure_strength(action_t=action_t) == pytest.approx(expected)


@pytest.mark.parametrize('constraint', [-1.8, 0.0, 0.15, 0.2, 1.6])
def test_strength_cannot_exceed_an_independent_current_constraint(constraint):
  assert departure_strength(constraint_accel=constraint) == pytest.approx(min(RECORDED_STRENGTH, constraint))


@pytest.mark.parametrize('fault', [
  'short_speeds', 'short_accels', 'nested_speeds', 'nan_speed', 'inf_accel', 'nonmonotonic_times',
  'nan_times', 'zero_horizon', 'negative_horizon', 'nan_horizon', 'infinite_horizon', 'past_trajectory',
  'boolean_horizon', 'nan_constraint', 'infinite_constraint', 'boolean_constraint',
])
def test_malformed_strength_input_cannot_produce_departure_evidence(fault):
  speeds, accels, times = list(RECORDED_SPEEDS), list(RECORDED_ACCELS), CONTROL_TIMES.copy()
  horizon, constraint = 0.55, 1.6
  if fault == 'short_speeds':
    speeds.pop()
  elif fault == 'short_accels':
    accels.pop()
  elif fault == 'nested_speeds':
    speeds = [speeds]
  elif fault == 'nan_speed':
    speeds[-1] = float('nan')
  elif fault == 'inf_accel':
    accels[-1] = float('inf')
  elif fault == 'nonmonotonic_times':
    times[3] = times[2]
  elif fault == 'nan_times':
    times[-1] = float('nan')
  elif fault.endswith('_horizon') or fault == 'past_trajectory':
    horizon = {'zero_horizon': 0.0, 'negative_horizon': -0.1, 'nan_horizon': float('nan'),
               'infinite_horizon': float('inf'), 'past_trajectory': times[-1] + 1.0, 'boolean_horizon': True}[fault]
  else:
    constraint = {'nan_constraint': float('nan'), 'infinite_constraint': float('inf'), 'boolean_constraint': True}[fault]
  assert departure_strength(speeds, accels, times, horizon, constraint) is None


def test_release_strength_is_local_evidence_and_is_not_decoded_from_the_wire():
  intent = MotionIntent(release_allowed=True, release_accel=RECORDED_STRENGTH)
  assert intent.is_valid()
  message = SimpleNamespace(version=intent.version, mode=intent.mode, reasons=0, movingLead=False,
                            releaseAllowed=True, allowCreep=False, valid=True, releaseAccel=100.0)
  assert MotionIntent.from_message(message).release_accel is None


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), True, '0.3', [0.3]])
def test_invalid_local_strength_cannot_release_a_held_owner(value):
  harness = MotionHarness()
  harness.stopped()
  intent = MotionIntent(release_allowed=True, release_accel=value)
  assert not intent.is_valid()
  assert harness.step(intent, a_target=RECORDED_TARGET, v_ego=0.0, standstill=True)
  assert harness.supervisor.degraded


@pytest.mark.parametrize('new_plan,new_model', [(True, False), (False, True), (False, False)])
def test_new_strength_requires_both_input_sources_to_advance(new_plan, new_model):
  harness = MotionHarness()
  harness.stopped()
  intent = MotionIntent(release_allowed=True, allow_creep=False, release_accel=RECORDED_STRENGTH)
  assert harness.step(intent, a_target=RECORDED_TARGET, v_ego=0.0, standstill=True,
                      new_plan=new_plan, new_model=new_model)
  assert harness.supervisor.phase == MotionPhase.HOLD
  assert harness.step(intent, a_target=RECORDED_TARGET, v_ego=0.0, standstill=True, new_plan=False, new_model=False)
  assert not harness.step(intent, a_target=RECORDED_TARGET, v_ego=0.0, standstill=True)
  assert harness.supervisor.phase == MotionPhase.RELEASE


@pytest.mark.parametrize('field', ['speeds', 'accels'])
def test_serialized_cached_source_cannot_change_unused_trajectory_samples(field):
  harness = held_controls()
  harness.plan.motionIntent.releaseAllowed = False
  _, motion = harness.step()
  assert motion.referenceTimingValid
  published_ns = harness.now_ns
  values = list(getattr(harness.plan, field))
  values[-1] += 0.1  # Does not change the action-time inverse; the dependency still changed.
  setattr(harness.plan, field, values)
  command, motion = harness.step(plan_time_ns=published_ns, model_time_ns=published_ns)
  assert not motion.referenceTimingValid and motion.degraded
  assert motion.resolvedStop and command.actuators.accel <= 0.0
  assert not command.cruiseControl.resume


@pytest.mark.parametrize('constraint', [-1.8, 0.0, 0.15])
def test_controls_cannot_spend_strength_above_the_independent_constraint(constraint):
  harness = held_controls()
  harness.plan.reference.constraintAccel = constraint
  harness.plan.aTarget = min(RECORDED_TARGET, constraint)
  command, motion = harness.step()
  assert motion.resolvedStop and motion.phase == 'hold'
  assert not motion.degraded and command.actuators.accel <= 0.0


@pytest.mark.parametrize('reason', [StopReason.MPC, StopReason.E2E, StopReason.CRUISE, StopReason.CLOSE_LEAD, StopReason.FORCE_DECEL])
def test_existing_raw_stop_vetoes_still_win_over_strong_departure_evidence(reason):
  harness = MotionHarness()
  harness.stopped()
  intent = MotionIntent(reasons=reason, release_allowed=True, release_accel=RECORDED_STRENGTH)
  assert harness.step(intent, a_target=RECORDED_TARGET, v_ego=0.0, standstill=True)
  assert harness.supervisor.reasons & reason


@pytest.mark.parametrize('kind', [MotionReferenceKind.REST, MotionReferenceKind.NO_FORWARD, MotionReferenceKind.EXACT_ZERO])
def test_raw_rest_or_no_forward_cannot_become_permission_from_strength_alone(kind):
  harness = MotionHarness()
  harness.stopped()
  intent = MotionIntent(reference_kind=kind, producer_instance_ns=harness.now_ns, release_accel=RECORDED_STRENGTH)
  assert intent.is_valid()
  assert harness.step(intent, a_target=RECORDED_TARGET, v_ego=0.0, standstill=True)
  assert harness.supervisor.phase == MotionPhase.HOLD


@pytest.mark.parametrize('condition', ['no_permission', 'nonpositive_target', 'not_ready', 'invalid_plan', 'solver_failure'])
def test_strength_does_not_replace_permission_positive_execution_or_health(condition):
  harness = MotionHarness()
  harness.stopped()
  intent = MotionIntent(release_allowed=True, release_accel=RECORDED_STRENGTH)
  options = {'a_target': RECORDED_TARGET, 'v_ego': 0.0, 'standstill': True}
  if condition == 'no_permission':
    intent = replace(intent, release_allowed=False)
  elif condition == 'not_ready':
    intent = replace(intent, ready=False)
  elif condition == 'nonpositive_target':
    options['a_target'] = 0.0
  elif condition == 'invalid_plan':
    options['plan_valid'] = False
  else:
    options['solver_status'] = 4
  assert harness.step(intent, **options)
  assert harness.supervisor.phase == MotionPhase.HOLD


@pytest.mark.parametrize('declining', [False, True])
def test_positive_inverse_from_negative_initial_accel_does_not_supply_movement_permission(declining):
  speeds = np.maximum(0.1 - 0.1 * CONTROL_TIMES, 0.0) if declining else np.zeros(CONTROL_N)
  accels = np.zeros(CONTROL_N)
  accels[0] = -0.6
  strength = departure_strength(speeds=speeds, accels=accels)
  assert strength >= STOP_RELEASE_ACCEL
  harness = MotionHarness()
  harness.stopped()
  intent = MotionIntent(release_allowed=False, release_accel=strength)
  assert harness.step(intent, a_target=RECORDED_TARGET, v_ego=0.0, standstill=True)
  assert harness.supervisor.phase == MotionPhase.HOLD


@pytest.mark.parametrize('condition', ['gas', 'brake', 'disable', 'override_event'])
def test_strength_cannot_override_driver_authority(condition):
  harness = held_controls()
  if condition in ('gas', 'brake'):
    setattr(harness.car_state, condition + 'Pressed', True)
  elif condition == 'disable':
    harness.selfdrive_state.enabled = False
  else:
    event = messaging.new_message('onroadEvents', 1, valid=True)
    event.onroadEvents[0].overrideLongitudinal = True
    harness.events['onroadEvents'] = event
  command, motion = harness.step()
  assert not command.longActive and not command.cruiseControl.resume
  assert command.actuators.accel == 0.0 and motion.phase == 'off'


@pytest.mark.parametrize('strength', [None, 0.19])
def test_weak_departure_keeps_its_existing_fresh_half_second_confirmation(strength):
  harness = MotionHarness()
  harness.stopped()
  intent = MotionIntent(release_allowed=True, allow_creep=False, release_accel=strength)
  for frame in range(round(CREEP_STOP_T / DT_MDL)):
    stopped = harness.step(intent, a_target=0.15, v_ego=0.0, standstill=True, advance_ns=round(DT_MDL * 1e9))
    assert stopped == (frame < round(CREEP_STOP_T / DT_MDL) - 1)
  assert harness.supervisor.phase == MotionPhase.RELEASE


def ordinary_action_case(*, fault=None):
  """Separate interpretation from solving, retaining the real planner guards."""
  planner, _ = make_planner(speed=1.0)
  state = make_state(speed=1.0, target=4.0, experimental=True)
  acceleration = np.full_like(T_IDXS, 0.17)
  acceleration[:3] = [0.0, 0.3, 0.21]
  if fault == 'model_stop':
    state['modelV2'].action.shouldStop = True
  elif fault == 'weak_action':
    state['modelV2'].action.desiredAcceleration = 0.15
  elif fault == 'close_lead':
    lead = state['radarState'].leadOne
    lead.present, lead.dRel, lead.vLead, lead.vRel = True, 0.5, 0.0, -1.0
    lead.aLeadK, lead.aLeadTau, lead.modelProb, lead.yRel = 0.0, 1.5, 1.0, 0.0

  def solve(*args, **kwargs):
    solve_given_acceleration(planner, acceleration, 1.0, *args, **kwargs)
    if fault == 'fcw':
      planner.mpc.crash_cnt = 3

  # Close-lead stopping guards a mild negative execution request. Give that
  # scenario a real independent brake constraint even though the substituted
  # native trajectory remains forward, and assert the resulting raw veto.
  cruise_constraint = -0.5 if fault == 'close_lead' else 1.6
  with patch.object(planner.reference_planner.cruise_response, 'update', return_value=cruise_constraint), \
       patch.object(planner.mpc, 'update', side_effect=solve):
    planner.update(state, now_ns=advance(state, 0))
  return planner


def test_e2e_ordinary_action_uses_the_same_strength_fact_and_keeps_native_execution_sampling():
  planner = ordinary_action_case()
  action_t = planner.CP.longitudinalActuatorDelay + DT_MDL
  expected_sample = np.interp(action_t, T_IDXS, planner.mpc.a_solution)
  # Match exactly the Float32 arrays the consumer receives.
  speeds = np.asarray(planner.v_desired_trajectory, dtype=np.float32).astype(float)
  accels = np.asarray(planner.a_desired_trajectory, dtype=np.float32).astype(float)
  legacy = get_accel_from_plan(speeds, accels, CONTROL_TIMES, action_t)
  assert 0.0 < expected_sample < STOP_RELEASE_ACCEL < legacy
  assert planner.reference_planner.reference_accel == pytest.approx(expected_sample)
  assert planner.output_a_target == pytest.approx(expected_sample)
  assert planner.motion_intent.reference_kind == MotionReferenceKind.MOVING
  assert not planner.reference_planner.model_evidence.allows_departure
  assert planner.motion_intent.release_allowed


@pytest.mark.parametrize('fault', ['model_stop', 'weak_action', 'close_lead', 'fcw'])
def test_ordinary_action_strength_cannot_bypass_current_producer_vetoes(fault):
  planner = ordinary_action_case(fault=fault)
  assert not planner.motion_intent.release_allowed
  if fault == 'fcw':
    assert planner.fcw
  elif fault == 'close_lead':
    assert planner.reference_planner.constraint_accel == pytest.approx(-0.5)
    assert planner.output_a_target == pytest.approx(-0.5)
    assert planner.motion_intent.reasons & StopReason.CLOSE_LEAD
