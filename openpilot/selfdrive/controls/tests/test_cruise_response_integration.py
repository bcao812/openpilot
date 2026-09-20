"""Real SCC policy selection and native planning preserve P's response history.

These fixed-input contracts inspect the published target and constraint. They
do not substitute for the common closed-loop or CAN/vehicle validation.
"""

from unittest.mock import patch

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_planner import get_cruise_accel, get_max_accel
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import Publisher, advance, make_planner, make_state
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanSource as SPSource
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.vision_controller import VisionState


def make_scc_case(*, target=30.0, initial_accel=0.0, experimental=False, enabled=True):
  params = Params()
  params.put_bool('SmartCruiseControlVision', enabled)
  params.put_bool('SmartCruiseControlMap', False)
  params.put_bool('DynamicExperimentalControl', False)
  planner, _ = make_planner(speed=20.0, initial_accel=initial_accel)
  state = make_state(speed=20.0, target=target, experimental=experimental)
  return planner, state


def set_curve(state, predicted_curvature, current_curvature=0.0):
  yaw_rate = float(predicted_curvature * state['carState'].vEgo)
  state['modelV2'].orientationRate.z = [yaw_rate] * len(ModelConstants.T_IDXS)
  state['controlsState'].curvature = current_curvature


def step(planner, state, frame, *, valid=True):
  now_ns = advance(state, frame)
  planner.update(state, now_ns=now_ns)
  publisher = Publisher()
  with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=now_ns + 500_000):
    planner.publish(state, publisher)
  message = publisher.messages['longitudinalPlan']
  plan = message.longitudinalPlan
  assert message.valid == valid
  assert plan.motionIntent.valid == valid
  if valid:
    assert plan.solverStatus == 0
  return plan


def p_cruise_step(planner, state, previous, *, reset=False):
  # P=e18b1de: reset at 286-289, then the unchanged numerical helper at 207-224.
  # These cases have no pitch/coast cap or steering-angle limit; SCC curvature
  # enters its own real producer through model yaw rate and controls curvature.
  car_state = state['carState']
  if reset:
    previous = float(np.clip(car_state.aEgo, ACCEL_MIN, ACCEL_MAX))
  return get_cruise_accel(
    state['selfdriveState'].experimentalMode, planner.cruise_target.speed,
    car_state.vEgo, previous, car_state.steeringAngleDeg - state['vehicleParameters'].angleOffsetDeg,
    planner.CP, planner.dt, ACCEL_MAX, True,
  )


def assert_published_arbitration(plan, state, cruise_ceiling):
  allowed_model = state['selfdriveState'].experimentalMode
  maximum = ACCEL_MAX if allowed_model else get_max_accel(state['carState'].vEgo)
  expected_constraint = min(maximum, cruise_ceiling)
  if allowed_model:
    expected_constraint = min(expected_constraint, state['modelV2'].action.desiredAcceleration)
  # The reference and final request are Float32 on the actual wire. Compare
  # their declared rounding directly, without widening a solver tolerance.
  assert plan.reference.constraintAccel == float(np.float32(expected_constraint))
  assert plan.aTarget == float(np.clip(min(plan.reference.accel, plan.reference.constraintAccel), ACCEL_MIN, ACCEL_MAX))


def test_real_scc_cycle_and_reentry_keep_continuous_p_response_history():
  planner, state = make_scc_case()
  # Only external curvature inputs drive enabled -> entering -> turning ->
  # leaving -> enabled -> entering. No state or policy target is patched.
  segments = (
    (4, 0.0010, 0.000, VisionState.enabled, SPSource.cruise),
    (8, 0.0075, 0.000, VisionState.entering, SPSource.sccVision),
    (6, 0.0075, 0.005, VisionState.turning, SPSource.sccVision),
    (6, 0.0040, 0.003, VisionState.leaving, SPSource.sccVision),
    (6, 0.0010, 0.000, VisionState.enabled, SPSource.cruise),
    (8, 0.0075, 0.000, VisionState.entering, SPSource.sccVision),
  )
  previous, frame = 0.0, 0
  selected_targets, requests = [], []
  for count, predicted, current, expected_state, expected_source in segments:
    set_curve(state, predicted, current)
    for _ in range(count):
      plan = step(planner, state, frame)
      assert planner.scc.vision.state == expected_state
      policy = planner.cruise_target
      selected_targets.append(policy)
      requests.append(plan.aTarget)
      assert policy.source == expected_source
      if expected_source == SPSource.sccVision:
        assert policy.speed == planner.scc.vision.output_v_target
        assert policy.acceleration == planner.scc.vision.output_a_target
      previous = p_cruise_step(planner, state, previous)
      assert_published_arbitration(plan, state, previous)
      frame += 1
  assert selected_targets[4].speed < 20.0 < selected_targets[18].speed
  assert selected_targets[24].source == SPSource.cruise
  assert selected_targets[30].source == SPSource.sccVision
  assert min(requests[4:18]) < 0.0


@pytest.mark.parametrize('target,expected_source', [(30.0, SPSource.sccVision), (21.0, SPSource.cruise)])
def test_negative_scc_acceleration_is_not_a_hard_brake_or_an_unselected_policy(target, expected_source):
  planner, state = make_scc_case(target=target)
  set_curve(state, 0.0035)
  previous, requests = 0.0, []
  for frame in range(12):
    plan = step(planner, state, frame)
    previous = p_cruise_step(planner, state, previous)
    assert_published_arbitration(plan, state, previous)
    requests.append(plan.aTarget)
    assert plan.reference.constraintAccel > 0.0
    assert plan.aTarget > 0.0
    if frame:
      assert planner.scc.vision.state == VisionState.entering
      # Decimal-curve goldens allow only the Float32 model-input rounding;
      # response and arbitration above still compare their exact wire values.
      assert planner.scc.vision.output_a_target == pytest.approx(-0.2470588235294118, rel=0.0, abs=1e-6)
      assert planner.scc.vision.output_v_target == pytest.approx(22.916336892570225, rel=0.0, abs=1e-6)
      assert planner.cruise_target.source == expected_source
      assert planner.cruise_target.speed > state['carState'].vEgo
  if target == 21.0:
    # The same externally supplied curve with SCC disabled must give the same
    # target trajectory when cruise already selects the lower speed.
    control, control_state = make_scc_case(target=target, enabled=False)
    set_curve(control_state, 0.0035)
    control_requests = [step(control, control_state, frame).aTarget for frame in range(12)]
    assert control_requests == requests


@pytest.mark.parametrize('risk,experimental', [('lead', False), ('model', True), ('model', False)],
                         ids=['lead-braking', 'admitted-model-braking', 'unadmitted-model'])
def test_stronger_lead_or_admitted_model_braking_keeps_priority_and_does_not_replace_cruise_history(risk, experimental):
  planner, state = make_scc_case(experimental=experimental)
  set_curve(state, 0.0035)
  state['modelV2'].action.desiredAcceleration = 1.5
  previous, risk_plans = 0.0, []
  for frame in range(20):
    risk_active = 6 <= frame < 12
    if risk == 'lead':
      lead = state['radarState'].leadOne
      lead.present = risk_active
      lead.dRel = 25.0
      lead.vLead = 10.0
      lead.vRel = -10.0
      lead.aLeadK = -2.0
      lead.aLeadTau = 1.5
      lead.modelProb = 0.99
    else:
      state['modelV2'].action.desiredAcceleration = -2.5 if risk_active else 1.5
    plan = step(planner, state, frame)
    previous = p_cruise_step(planner, state, previous)
    assert_published_arbitration(plan, state, previous)
    if frame:
      assert planner.cruise_target.source == SPSource.sccVision
    if risk_active:
      risk_plans.append(plan)
  if risk == 'lead':
    assert risk_plans[-1].reference.constraintAccel > 0.0
    assert risk_plans[-1].aTarget < 0.0
    assert risk_plans[-1].aTarget == risk_plans[-1].reference.accel
  elif experimental:
    assert all(plan.reference.constraintAccel == -2.5 and plan.aTarget <= -2.5 for plan in risk_plans)
  else:
    assert all(plan.reference.constraintAccel > 0.0 and plan.aTarget > 0.0 for plan in risk_plans)


@pytest.mark.parametrize('initial_accel', [-0.6, 0.6])
def test_nonzero_constructor_and_explicit_reset_keep_p_response_without_overwriting_measured_mpc_state(initial_accel):
  planner, state = make_scc_case(initial_accel=initial_accel)
  set_curve(state, 0.0075)
  previous = initial_accel
  for frame, measured in enumerate((0.35, 0.35, 0.35, -0.35, -0.35, 0.1, 0.1, 0.1)):
    reset = frame in (3, 4)
    state['controlsState'].longControlState = LongCtrlState.off if reset else LongCtrlState.pid
    state['carState'].aEgo = measured
    plan = step(planner, state, frame)
    previous = p_cruise_step(planner, state, previous, reset=reset)
    assert_published_arbitration(plan, state, previous)
    assert plan.reference.initialStateKnown
    assert plan.reference.initialAEgo == state['carState'].aEgo
    assert planner.mpc.x0[2] == state['carState'].aEgo
    if frame:
      assert planner.cruise_target.source == SPSource.sccVision
      assert planner.cruise_target.acceleration < 0.0


@pytest.mark.parametrize('failure', ['owner_unavailable', 'unhealthy_inputs'])
def test_invalid_or_unhealthy_cycle_clears_response_before_healthy_recovery(failure):
  planner, state = make_scc_case()
  set_curve(state, 0.0035)
  for frame in range(8):
    plan = step(planner, state, frame)
  assert plan.reference.constraintAccel > 0.3
  if failure == 'owner_unavailable':
    state['controlsState'].motionState.enabled = False
    step(planner, state, 8, valid=False)
    state['controlsState'].motionState.enabled = True
  else:
    with patch.object(state, 'all_checks', return_value=False):
      step(planner, state, 8, valid=False)
  state['carState'].aEgo = -0.35
  recovered = step(planner, state, 9)
  expected = p_cruise_step(planner, state, 0.0, reset=True)
  assert_published_arbitration(recovered, state, expected)
  assert recovered.reference.constraintAccel < 0.0
