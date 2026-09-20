"""Withdraw an obsolete rolling stop without inventing standstill departure.

The native cases exercise producer input history, wire transport and the actual
controller. They are fixed-input software cases, not vehicle stopping results.
"""

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.motion_supervisor import (
  MotionIntent, MotionPhase, MotionReferenceKind, MotionResponseKind, StopReason,
)
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import (
  consume, make_handover_state, make_planner_owner,
)
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_motion_supervisor import MotionHarness


def continuation(*, response=MotionResponseKind.NONE):
  kind = MotionReferenceKind.MOVING if response == MotionResponseKind.NONE else MotionReferenceKind.UNKNOWN
  return MotionIntent(reference_kind=int(kind), response_kind=int(response), producer_instance_ns=1_000_000_000,
                      allow_creep=False)


@pytest.mark.parametrize('speed', [1.8, 2.0])
def test_withdrawn_model_stop_returns_a_rolling_car_to_its_current_reference(speed):
  planner, controller = make_planner_owner(speed)
  state = make_handover_state(speed=speed, target=speed, experimental=True)
  state['modelV2'].action.shouldStop = True
  state['modelV2'].action.desiredAcceleration = -0.2
  _, previous = consume(planner, controller, state, 0)
  assert controller.motion_phase == MotionPhase.APPROACH
  state['modelV2'].action.shouldStop = False
  state['modelV2'].action.desiredAcceleration = 0.0
  requests = [previous[-1]]
  for frame in range(1, 13):
    state['controlsState'].longControlState = controller.long_control_state
    plan, outputs = consume(planner, controller, state, frame)
    assert plan.motionIntent.referenceKind == int(MotionReferenceKind.MOVING)
    assert plan.motionIntent.reasons == 0
    assert not plan.motionIntent.releaseAllowed
    assert not controller.resolved_should_stop
    assert not controller.motion_degraded
    requests.extend(outputs)
  # The existing upward limiter still owns removal of the previous brake.
  assert np.max(np.diff(requests)) <= 0.08 + 1e-12
  assert controller.motion_phase == MotionPhase.TRACK


@pytest.mark.parametrize('response', [MotionResponseKind.NONE, MotionResponseKind.PURE_CRUISE, MotionResponseKind.ABSOLUTE_ACC])
def test_current_moving_target_can_withdraw_an_unheld_approach(response):
  harness = MotionHarness()
  assert harness.step(MotionIntent(reasons=StopReason.E2E), v_ego=1.8, a_target=-0.2)
  assert not harness.step(continuation(response=response), v_ego=1.8, a_target=-0.2)
  assert harness.supervisor.phase == MotionPhase.RELEASE


@pytest.mark.parametrize('new_plan,new_model', [(False, False), (True, False)])
def test_cached_continuation_does_not_withdraw_a_stop(new_plan, new_model):
  harness = MotionHarness()
  assert harness.step(MotionIntent(reasons=StopReason.E2E), v_ego=1.8, a_target=-0.2)
  assert harness.step(continuation(), v_ego=1.8, a_target=-0.2, new_plan=new_plan, new_model=new_model)
  assert harness.supervisor.phase == MotionPhase.APPROACH


@pytest.mark.parametrize('reason', [StopReason.CLOSE_LEAD, StopReason.CREEP])
def test_current_continuation_keeps_an_applicable_owned_stop(reason):
  harness = MotionHarness()
  if reason == StopReason.CLOSE_LEAD:
    harness.step(MotionIntent(reasons=reason), v_ego=1.0, a_target=-0.2)
  else:
    for _ in range(60):
      harness.step(v_ego=1.0, a_target=-0.2)
  intent = MotionIntent(reference_kind=int(MotionReferenceKind.MOVING), producer_instance_ns=1_000_000_000)
  assert harness.step(intent, v_ego=1.0, a_target=-0.2)
  assert harness.supervisor.reasons & reason


def test_rolling_continuation_is_not_permission_to_depart_a_previous_hold():
  harness = MotionHarness()
  harness.stopped()
  assert harness.step(continuation(), v_ego=1.8, a_target=0.5)
  assert harness.supervisor.reasons & StopReason.HOLD
