"""A winning P cruise request is executed without a second error integrator.

These controller cases inject explicit producer evidence and learn all prior
integral through real updates. Native winner qualification is covered by the
producer tests; neither this file nor a feedback ablation proves closed-loop
or vehicle acceptance.
"""

from dataclasses import replace
import math
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import CREEP_STOP_T, STOP_RELEASE_ACCEL
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.motion_supervisor import (
  MOTION_INTENT_VERSION, STARTING_ACCEL_RATE, MotionIntent, MotionPhase, MotionReferenceKind, MotionResponseKind, StopReason,
)
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import UnifiedLongitudinalPlanner
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import ControlHarness, decoded_commands
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_cruise_coast_response import feedback_intent, learned_harness
from openpilot.selfdrive.controls.tests.test_motion_supervisor import MotionHarness


# Wire constants allow red tests to collect against the preceding contract.
REQUEST_VERSION = 10
REQUEST_KIND = 5
INSTANCE_NS = 1_000_000_000
UPWARD_STEP = STARTING_ACCEL_RATE * DT_CTRL
BRAKING_REQUEST = float(np.float32(-0.35))


def request_intent(**changes):
  return replace(MotionIntent(version=REQUEST_VERSION, response_kind=REQUEST_KIND, producer_instance_ns=INSTANCE_NS,
                              allow_creep=False), **changes)


def braking_approach(*, tuned_ki=False):
  harness = learned_harness(tuned_ki=tuned_ki)
  stop = MotionIntent(reasons=int(StopReason.E2E), producer_instance_ns=INSTANCE_NS, allow_creep=False)
  for _ in range(3):
    harness.step(BRAKING_REQUEST, stop)
  assert harness.control.motion_phase == MotionPhase.APPROACH
  assert harness.control.pid.i < -0.03
  return harness


@pytest.mark.parametrize('moving_lead,release_allowed', [(False, False), (True, False), (False, True), (True, True)])
def test_request_wire_contract_preserves_independent_motion_facts(moving_lead, release_allowed):
  assert MOTION_INTENT_VERSION == REQUEST_VERSION
  assert int(MotionResponseKind.CRUISE_REQUEST) == REQUEST_KIND
  intent = request_intent(moving_lead=moving_lead, release_allowed=release_allowed)
  assert intent.is_valid()
  planner = SimpleNamespace(longitudinal_mode=LongitudinalMode.UNIFIED)
  producer = UnifiedLongitudinalPlanner(planner)
  planner.motion_intent = intent
  message = log.LongitudinalPlan.new_message()
  producer.publish(message)
  with log.LongitudinalPlan.from_bytes(message.to_bytes()) as decoded:
    actual = MotionIntent.from_message(decoded.motionIntent)
    assert actual == intent and actual.is_valid()
    assert actual.reference_kind == MotionReferenceKind.UNKNOWN
    assert actual.moving_lead == moving_lead and actual.release_allowed == release_allowed


@pytest.mark.parametrize('changes', [
  {'version': 8}, {'version': 9}, {'response_kind': True}, {'response_kind': 5.0}, {'response_kind': 6},
  {'ready': False}, {'valid': False}, {'reasons': int(StopReason.MPC)}, {'reasons': int(StopReason.FORCE_DECEL)},
  {'braking_accel': -0.5}, {'producer_instance_ns': 0}, {'producer_instance_ns': None},
] + [{'reference_kind': int(kind)} for kind in (MotionReferenceKind.EXACT_ZERO, MotionReferenceKind.REST, MotionReferenceKind.NO_FORWARD)])
def test_request_rejects_unqualified_or_stopping_reference_contracts(changes):
  assert not request_intent(**changes).is_valid()


@pytest.mark.parametrize('moving_lead,release_allowed', [(True, False), (False, True), (True, True)])
def test_request_flags_do_not_relax_the_distinct_coast_contract(moving_lead, release_allowed):
  intent = request_intent(moving_lead=moving_lead, release_allowed=release_allowed)
  assert intent.is_valid()
  assert not replace(intent, response_kind=int(MotionResponseKind.CRUISE_COAST)).is_valid()


@pytest.mark.parametrize('tuned_ki', [False, True], ids=['launch-response', 'configured-pid'])
@pytest.mark.parametrize('sign', [-1, 1], ids=['negative-history', 'positive-history'])
@pytest.mark.parametrize('cruise_request', [-0.35, 0.35], ids=['braking-request', 'accelerating-request'])
def test_accepted_request_executes_without_integral_then_returns_to_the_original_feedback_domains(cruise_request, sign, tuned_ki):
  harness = learned_harness(sign, tuned_ki=tuned_ki)
  command = float(np.float32(cruise_request))
  outputs = [harness.control.last_output_accel]
  settle_ticks = math.ceil(max(0.0, command - outputs[-1]) / UPWARD_STEP) + 2
  for _ in range(settle_ticks + 10):
    outputs.append(harness.step(command, request_intent()))
    assert not harness.control.motion_input_constrained and not harness.control.resolved_should_stop
    assert harness.control.pid.i == 0.0
    if harness.control.launch_response is not None:
      assert not harness.control.launch_response.armed
  assert np.max(np.diff(outputs)) <= UPWARD_STEP + 1e-12
  assert outputs[-10:] == [command] * 10
  assert harness.control.motion_phase == MotionPhase.TRACK
  for encoded in decoded_commands(outputs[-10:]):
    assert encoded['ACCEL_COMMAND'] == pytest.approx(cruise_request, abs=1e-6)
    assert bool(encoded['BRAKE_REQUEST']) == (cruise_request < 0.0)

  # Independent following/risk feedback remains available on the next target.
  absolute = MotionIntent(response_kind=int(MotionResponseKind.ABSOLUTE_ACC), producer_instance_ns=INSTANCE_NS, allow_creep=False)
  for _ in range(10):
    assert harness.step(-0.8, absolute) <= -0.8
  assert harness.control.pid.i < 0.0
  harness.step(command, request_intent())
  assert harness.control.pid.i == 0.0
  for _ in range(80):
    harness.step(sign * 0.3, feedback_intent(sign))
  assert sign * harness.control.pid.i > 0.03


@pytest.mark.parametrize('tuned_ki', [False, True], ids=['launch-response', 'configured-pid'])
def test_fresh_request_withdraws_unheld_approach_and_keeps_the_existing_output_ramp(tuned_ki):
  harness = braking_approach(tuned_ki=tuned_ki)
  command = float(np.float32(0.35))
  outputs = [harness.control.last_output_accel]
  settle_ticks = math.ceil((command - outputs[-1]) / UPWARD_STEP) + 2
  for _ in range(settle_ticks):
    outputs.append(harness.step(command, request_intent()))
    assert not harness.control.resolved_should_stop and not harness.control.motion_input_constrained
    assert harness.control.pid.i == 0.0
    assert harness.control.motion_phase in (MotionPhase.RELEASE, MotionPhase.TRACK)
  assert np.max(np.diff(outputs)) <= UPWARD_STEP + 1e-12
  assert outputs[-1] == command and harness.control.motion_phase == MotionPhase.TRACK


@pytest.mark.parametrize('tuned_ki', [False, True], ids=['launch-response', 'configured-pid'])
def test_plan_only_and_cached_request_cannot_retire_an_unwithdrawn_approach_integral(tuned_ki):
  harness = braking_approach(tuned_ki=tuned_ki)
  previous_output, previous_integral = harness.control.last_output_accel, harness.control.pid.i
  model_time_ns = harness.model_time_ns
  plan_time_ns = harness.now_ns + round(DT_CTRL * 1e9)
  for _ in range(5):
    output = harness.step(BRAKING_REQUEST, request_intent(), new_input=False,
                          plan_time_ns=plan_time_ns, model_time_ns=model_time_ns)
    assert not harness.control.motion_input_constrained
    assert harness.control.resolved_should_stop and harness.control.motion_phase == MotionPhase.APPROACH
    assert harness.control.motion_reasons & StopReason.E2E
    assert harness.control.pid.i <= previous_integral + 1e-12
    assert output <= previous_output + 1e-12
    previous_output, previous_integral = output, harness.control.pid.i


@pytest.mark.parametrize('moving_lead', [False, True])
@pytest.mark.parametrize('target', [STOP_RELEASE_ACCEL / 2.0, STOP_RELEASE_ACCEL * 2.0])
def test_request_kind_and_moving_lead_do_not_replace_held_release_permission(moving_lead, target):
  harness = MotionHarness()
  harness.stopped()
  for _ in range(round(2.0 * CREEP_STOP_T / DT_MDL)):
    assert harness.step(request_intent(moving_lead=moving_lead), a_target=target, v_ego=0.0, standstill=True,
                        advance_ns=round(DT_MDL * 1e9))
    assert not harness.supervisor.degraded and not harness.supervisor.waiting_for_fresh_input
    assert harness.supervisor.phase == MotionPhase.HOLD and harness.supervisor.reasons & StopReason.HOLD


def test_weak_request_with_release_evidence_keeps_the_existing_confirmation_window():
  harness = MotionHarness()
  harness.stopped()
  intent = request_intent(release_allowed=True)
  target = STOP_RELEASE_ACCEL / 2.0
  for _ in range(round(CREEP_STOP_T / DT_MDL) - 1):
    assert harness.step(intent, a_target=target, v_ego=0.0, standstill=True, advance_ns=round(DT_MDL * 1e9))
    assert not harness.supervisor.degraded
    assert harness.supervisor.phase == MotionPhase.HOLD
    assert harness.supervisor.release_lead_t < CREEP_STOP_T
  assert not harness.step(intent, a_target=target, v_ego=0.0, standstill=True, advance_ns=round(DT_MDL * 1e9))
  assert harness.supervisor.phase == MotionPhase.RELEASE


@pytest.mark.parametrize('new_plan', [False, True], ids=['cached', 'plan-only'])
def test_cached_weak_request_does_not_add_departure_confirmation(new_plan):
  harness = MotionHarness()
  harness.stopped()
  intent = request_intent(release_allowed=True, moving_lead=True)
  target = STOP_RELEASE_ACCEL / 2.0
  assert harness.step(intent, a_target=target, v_ego=0.0, standstill=True)
  confirmation = harness.supervisor.release_lead_t
  for _ in range(10):
    assert harness.step(intent, a_target=target, v_ego=0.0, standstill=True, new_plan=new_plan, new_model=False)
    assert not harness.supervisor.degraded and not harness.supervisor.waiting_for_fresh_input
    assert harness.supervisor.release_lead_t == confirmation
    assert harness.supervisor.phase == MotionPhase.HOLD


def test_honda_protocol_hold_blocks_an_otherwise_authorized_request_release():
  harness = ControlHarness()
  harness.hold()
  harness.state.cruiseState.standstill = True
  for _ in range(10):
    output = harness.step(STOP_RELEASE_ACCEL * 2.0, request_intent(release_allowed=True))
    assert not harness.control.motion_input_constrained
    assert harness.control.long_control_state == LongCtrlState.stopping
    assert output <= -2.0 and harness.control.pid.i == 0.0
    assert decoded_commands([output])[0]['BRAKE_REQUEST']


@pytest.mark.parametrize('failure', ['old-version', 'invalid-plan', 'mutated-cached-plan'])
def test_unaccepted_request_does_not_weaken_the_current_brake(failure):
  harness = braking_approach()
  previous = harness.control.last_output_accel
  intent = request_intent(version=8) if failure == 'old-version' else request_intent()
  output = harness.step(BRAKING_REQUEST, intent, plan_valid=failure != 'invalid-plan',
                        new_input=failure != 'mutated-cached-plan')
  assert harness.control.motion_input_constrained and harness.control.resolved_should_stop
  assert output <= previous and decoded_commands([output])[0]['BRAKE_REQUEST']
