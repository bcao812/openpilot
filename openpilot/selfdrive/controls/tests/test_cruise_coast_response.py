"""P's coast command keeps its meaning through the owner and Honda CAN.

Native cases use real policy parameters and MPC output. Controller cases inject
explicit producer contracts to isolate ownership and integral transitions; all
integral history is learned through executed updates. These are fixed-input
software checks, not closed-loop or vehicle validation.
"""

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.motion_supervisor import (
  MOTION_INTENT_VERSION, STARTING_ACCEL_RATE, MotionIntent, MotionPhase, MotionReferenceKind, MotionResponseKind, StopReason,
)
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import UnifiedLongitudinalPlanner
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import ControlHarness, controller, decoded_commands
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import consume, make_handover_state, make_planner_owner
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_motion_supervisor import MotionHarness


# Fixed wire values allow the pre-fix version to collect and fail behaviorally.
COAST_VERSION = 10
COAST_KIND = 4
INSTANCE_NS = 1_000_000_000
# P e18b1de's plateau, after the real LongitudinalPlan Float32 transport.
COAST_COMMAND = float(np.float32(-0.19))
UPWARD_STEP = STARTING_ACCEL_RATE * DT_CTRL


def coast_intent(**changes):
  return replace(MotionIntent(version=COAST_VERSION, response_kind=COAST_KIND, producer_instance_ns=INSTANCE_NS,
                              allow_creep=False), **changes)


def pure_intent():
  return MotionIntent(response_kind=int(MotionResponseKind.PURE_CRUISE), producer_instance_ns=INSTANCE_NS, allow_creep=False)


def learned_harness(sign=-1, *, tuned_ki=False):
  control = controller()
  if tuned_ki:
    control.CP.longitudinalTuning.kiV = [1.0]
    control = LongControl(control.CP, control.CP_SP)
  assert (control.launch_response is None) == tuned_ki
  harness = ControlHarness(control)
  harness.state.vEgo = 20.23
  harness.state.aEgo = 0.0
  for _ in range(80):
    harness.step(sign * 0.3, pure_intent())
  assert harness.control.motion_phase == MotionPhase.TRACK
  assert not harness.control.motion_input_constrained
  assert sign * harness.control.pid.i > 0.03
  return harness


def approach_with_braking_history(*, tuned_ki=False):
  harness = learned_harness(tuned_ki=tuned_ki)
  stop = MotionIntent(reasons=int(StopReason.E2E), allow_creep=False, producer_instance_ns=INSTANCE_NS)
  # Establish the same scalar before withdrawing the stop cause, so the
  # cached-contract checks isolate I retention from a changed target.
  for _ in range(3):
    harness.step(COAST_COMMAND, stop)
  assert harness.control.motion_phase == MotionPhase.APPROACH
  assert harness.control.pid.i < -0.03
  return harness


def test_coast_contract_has_its_own_versioned_wire_kind():
  assert MOTION_INTENT_VERSION == COAST_VERSION
  assert int(MotionResponseKind.CRUISE_COAST) == COAST_KIND
  intent = coast_intent()
  assert intent.is_valid()
  planner = SimpleNamespace(longitudinal_mode=LongitudinalMode.UNIFIED)
  producer = UnifiedLongitudinalPlanner(planner)
  planner.motion_intent = intent
  message = log.LongitudinalPlan.new_message()
  producer.publish(message)
  with log.LongitudinalPlan.from_bytes(message.to_bytes()) as decoded:
    assert MotionIntent.from_message(decoded.motionIntent) == intent
    assert MotionIntent.from_message(decoded.motionIntent).is_valid()
    assert decoded.motionIntent.referenceKind == MotionReferenceKind.UNKNOWN
    assert not decoded.motionIntent.releaseAllowed and not decoded.motionIntent.movingLead


@pytest.mark.parametrize('changes', [
  {'version': 7}, {'version': 8}, {'response_kind': True}, {'response_kind': 4.0}, {'response_kind': 6},
  {'release_allowed': True}, {'moving_lead': True}, {'ready': False}, {'valid': False},
  {'reasons': int(StopReason.FORCE_DECEL)}, {'braking_accel': -0.5}, {'producer_instance_ns': 0},
] + [{'reference_kind': int(kind)} for kind in MotionReferenceKind if kind != MotionReferenceKind.UNKNOWN])
def test_coast_contract_rejects_model_semantics_departure_and_unqualified_requests(changes):
  assert not coast_intent(**changes).is_valid()


@pytest.mark.parametrize('tuned_ki', [False, True], ids=['launch-response', 'configured-pid'])
@pytest.mark.parametrize('sign', [-1, 1], ids=['negative-history', 'positive-history'])
def test_accepted_coast_retires_the_same_integral_and_pure_cruise_can_learn_again(sign, tuned_ki):
  harness = learned_harness(sign, tuned_ki=tuned_ki)
  outputs = [harness.control.last_output_accel]
  settle_ticks = math.ceil(max(0.0, COAST_COMMAND - outputs[-1]) / UPWARD_STEP) + 2
  for _ in range(settle_ticks + 20):
    outputs.append(harness.step(COAST_COMMAND, coast_intent()))
    assert not harness.control.resolved_should_stop
    assert not harness.control.motion_input_constrained
    assert harness.control.pid.i == 0.0
    if harness.control.launch_response is not None:
      assert not harness.control.launch_response.armed
  assert np.max(np.diff(outputs)) <= UPWARD_STEP + 1e-12
  assert harness.control.motion_phase == MotionPhase.TRACK
  assert outputs[-20:] == [COAST_COMMAND] * 20
  for command in decoded_commands(outputs[-20:]):
    assert command['ACCEL_COMMAND'] == pytest.approx(-0.19, abs=1e-6)
    assert not command['BRAKE_REQUEST'] and not command['BRAKE_LIGHTS']
    assert 0.0 < command['GAS_COMMAND'] < 10.0

  for _ in range(80):
    harness.step(sign * 0.3, pure_intent())
  assert sign * harness.control.pid.i > 0.03
  # The next independent braking request must remain immediately available.
  harness.step(COAST_COMMAND, coast_intent())
  assert harness.control.pid.i == 0.0
  braking = MotionIntent(response_kind=int(MotionResponseKind.ABSOLUTE_ACC), producer_instance_ns=INSTANCE_NS, allow_creep=False)
  output = harness.step(-1.2, braking)
  assert output <= -1.2
  assert decoded_commands([output])[0]['BRAKE_REQUEST']


@pytest.mark.parametrize('tuned_ki', [False, True], ids=['launch-response', 'configured-pid'])
def test_fresh_coast_withdraws_an_unheld_approach_before_retiring_integral(tuned_ki):
  harness = approach_with_braking_history(tuned_ki=tuned_ki)
  outputs = [harness.control.last_output_accel]
  settle_ticks = math.ceil((COAST_COMMAND - outputs[-1]) / UPWARD_STEP) + 2
  for _ in range(settle_ticks):
    outputs.append(harness.step(COAST_COMMAND, coast_intent()))
    assert not harness.control.resolved_should_stop
    assert not harness.control.motion_input_constrained
    assert harness.control.pid.i == 0.0
    assert harness.control.motion_phase in (MotionPhase.RELEASE, MotionPhase.TRACK)
  assert np.max(np.diff(outputs)) <= UPWARD_STEP + 1e-12
  assert outputs[-1] == COAST_COMMAND
  assert harness.control.motion_phase == MotionPhase.TRACK


@pytest.mark.parametrize('tuned_ki', [False, True], ids=['launch-response', 'configured-pid'])
def test_plan_only_and_cached_coast_keep_approach_braking_and_integral(tuned_ki):
  harness = approach_with_braking_history(tuned_ki=tuned_ki)
  previous_output = harness.control.last_output_accel
  previous_integral = harness.control.pid.i
  # A new plan may reuse the previous model source. This is valid transport,
  # but cannot establish fresh rolling continuation. Then consume that same
  # coast plan repeatedly without mutating its payload under an old identity.
  model_time_ns = harness.model_time_ns
  plan_time_ns = harness.now_ns + round(DT_CTRL * 1e9)
  for _ in range(5):
    output = harness.step(COAST_COMMAND, coast_intent(), new_input=False,
                          plan_time_ns=plan_time_ns, model_time_ns=model_time_ns)
    assert not harness.control.motion_input_constrained
    assert harness.control.resolved_should_stop
    assert harness.control.motion_phase == MotionPhase.APPROACH
    assert harness.control.motion_reasons & StopReason.E2E
    assert harness.control.pid.i <= previous_integral + 1e-12
    assert output <= previous_output + 1e-12
    previous_output, previous_integral = output, harness.control.pid.i


@pytest.mark.parametrize('speed,standstill', [(0.0, True), (0.3, False), (1.8, False)])
def test_coast_never_grants_departure_from_an_actual_hold(speed, standstill):
  harness = MotionHarness()
  harness.stopped()
  for _ in range(60):
    assert harness.step(coast_intent(), a_target=COAST_COMMAND, v_ego=speed, standstill=standstill)
    assert not harness.supervisor.degraded and not harness.supervisor.waiting_for_fresh_input
    assert harness.supervisor.reasons & StopReason.HOLD
    assert harness.supervisor.phase in (MotionPhase.HOLD, MotionPhase.APPROACH)


def test_honda_protocol_hold_retains_braking_with_a_coast_plan():
  harness = ControlHarness()
  harness.hold()
  harness.state.cruiseState.standstill = True
  for _ in range(10):
    output = harness.step(COAST_COMMAND, coast_intent())
    assert harness.control.resolved_should_stop
    assert not harness.control.motion_input_constrained
    assert harness.control.motion_reasons & StopReason.HOLD
    assert output <= -2.0
    assert harness.control.pid.i == 0.0
    assert decoded_commands([output])[0]['BRAKE_REQUEST']


@pytest.mark.parametrize('failure', ['old-version', 'invalid-plan', 'mutated-cached-plan'])
def test_unaccepted_coast_does_not_weaken_an_owned_brake(failure):
  harness = approach_with_braking_history()
  previous = harness.control.last_output_accel
  intent = coast_intent(version=7) if failure == 'old-version' else coast_intent()
  output = harness.step(COAST_COMMAND, intent, plan_valid=failure != 'invalid-plan',
                        new_input=failure != 'mutated-cached-plan')
  assert harness.control.motion_input_constrained
  assert harness.control.resolved_should_stop
  assert output <= previous
  assert decoded_commands([output])[0]['BRAKE_REQUEST']


def test_native_coast_oscillation_keeps_p_plateau_through_wire_integrator_and_can():
  params = Params()
  for name in ('SmartCruiseControlVision', 'DynamicExperimentalControl'):
    params.put_bool(name, False, block=True)
  speed = 20.23
  planner, owner = make_planner_owner(speed=speed)
  state = make_handover_state(speed=speed, target=20.0)
  commands = []
  for frame in range(60):
    # Only external set speed changes: after startup, the raw P response
    # travels across -0.2 +/- 0.045, inside its [-0.25, -0.15] coast plateau.
    error = -0.23 if frame < 12 else (-0.245 if (frame // 6) % 2 else -0.155)
    state['carState'].vCruise = state['carState'].vCruiseCluster = (speed + error) * 3.6
    state['controlsState'].longControlState = owner.long_control_state
    plan, outputs = consume(planner, owner, state, frame)
    assert not owner.motion_input_constrained
    if frame >= 12:
      assert not planner.reference_planner.e2e_allowed
      assert plan.motionIntent.responseKind == COAST_KIND
      assert plan.motionIntent.referenceKind == MotionReferenceKind.UNKNOWN
      assert not plan.motionIntent.releaseAllowed and not plan.motionIntent.movingLead
      assert plan.aTarget == COAST_COMMAND
      assert outputs == [COAST_COMMAND] * 5
      assert owner.pid.i == 0.0
      commands.extend(decoded_commands(outputs))
  assert len(commands) == 48 * 5
  for command in commands:
    assert command['ACCEL_COMMAND'] == pytest.approx(-0.19, abs=1e-6)
    assert not command['BRAKE_REQUEST'] and not command['BRAKE_LIGHTS']
    assert 0.0 < command['GAS_COMMAND'] < 10.0
