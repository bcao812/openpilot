"""A qualified closing ceiling participates in the existing ACC contract.

Injected sampled targets isolate source/authority arbitration; continuous
physical following is exercised separately with the actual native solution.
"""
from copy import deepcopy
from unittest.mock import Mock, patch

import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.lead_closing import REQUEST_DECEL_RATE
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionResponseKind
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import advance, consume, make_planner_owner, make_state
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_motion_response_contract import ordinary_policy as ordinary_policy


SAMPLE = 'openpilot.selfdrive.controls.lib.unified_longitudinal_planner.sample_reference_acceleration'
WARM_FRAMES = 20


def inputs(state, frame, *, slot=0, gap=65.0, closing=6.0):
  now = advance(state, frame)
  state['radarState'].mdMonoTime = now
  speed = state['carState'].vEgo
  for index, lead in enumerate((state['radarState'].leadOne, state['radarState'].leadTwo)):
    lead.present = index == slot
    lead.dRel = gap - closing * frame * 0.05
    lead.vRel = -closing
    lead.vLead = speed - closing
    lead.vLeadK = speed - closing
    lead.aLeadK = 0.0
    lead.aLeadTau = 1.5
    lead.yRel = 0.0
    lead.modelProb = 0.999
  state['selfdriveState'].personality = log.LongitudinalPersonality.standard
  return now


def warm(planner, controller, state, *, slot=0, sampled=0.6):
  native = planner.mpc.update

  def preserve_raw_source(*args, **kwargs):
    native(*args, **kwargs)
    planner.mpc.source = LongitudinalPlanSource.cruise

  with patch(SAMPLE, return_value=sampled), patch.object(planner.mpc, 'update', side_effect=preserve_raw_source):
    for frame in range(WARM_FRAMES):
      inputs(state, frame, slot=slot)
      before = deepcopy(state['radarState'].to_dict())
      plan, outputs = consume(planner, controller, state, frame)
      assert plan.solverStatus == 0
      assert state['radarState'].to_dict() == before
  return plan, outputs


@pytest.mark.parametrize('slot', [0, 1])
def test_ceiling_preserves_native_source_and_absolute_negative_feedback(slot, ordinary_policy):
  planner, controller = make_planner_owner(speed=25.0)
  state = make_state(speed=25.0, target=28.0)
  plan, outputs = warm(planner, controller, state, slot=slot)
  bound = planner.reference_planner.closing_constraint
  assert bound is not None and bound.accel < 0
  assert plan.aTarget == pytest.approx(bound.accel, abs=1e-6)
  assert plan.reference.accel == pytest.approx(0.6)
  assert plan.reference.source == LongitudinalPlanSource.cruise
  assert plan.reference.constraintSource == (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1)[slot]
  assert plan.reference.overridden
  assert plan.motionIntent.responseKind == MotionResponseKind.ABSOLUTE_ACC
  assert not plan.motionIntent.releaseAllowed and not plan.shouldStop
  assert all(-3.5 <= value <= 0 for value in outputs)
  assert controller.pid.i <= 0.0


def test_existing_stronger_native_braking_is_preserved(ordinary_policy):
  planner, controller = make_planner_owner(speed=25.0)
  state = make_state(speed=25.0, target=28.0)
  plan, _ = warm(planner, controller, state, sampled=-3.5)
  assert planner.reference_planner.closing_constraint is not None
  assert plan.aTarget == -3.5
  assert not plan.reference.overridden


@pytest.mark.parametrize('condition', [
  'brake', 'gas', 'override', 'inactive', 'disabled', 'control_disabled', 'force_decel',
  'unhealthy', 'left_signal', 'right_signal', 'lane_change', 'turn', 'e2e',
])
def test_losing_eligibility_clears_constraint_and_requires_new_history(condition, ordinary_policy):
  planner, controller = make_planner_owner(speed=25.0)
  state = make_state(speed=25.0, target=28.0)
  warm(planner, controller, state)
  assert planner.reference_planner.closing_constraint is not None
  if condition in ('brake', 'gas'):
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
  elif condition in ('left_signal', 'right_signal'):
    setattr(state['carState'], condition.split('_')[0] + 'Blinker', True)
  elif condition == 'lane_change':
    state['modelV2'].meta.laneChangeState = log.LaneChangeState.laneChangeStarting
  elif condition == 'turn':
    state['carState'].steeringAngleDeg = 6.0
  elif condition == 'e2e':
    state['selfdriveState'].experimentalMode = True
  now = inputs(state, WARM_FRAMES)
  planner.update(state, now_ns=now)
  assert planner.reference_planner.closing_constraint is None
  restored = make_state(speed=25.0, target=28.0)
  now = inputs(restored, WARM_FRAMES + 1)
  planner.update(restored, now_ns=now)
  assert planner.reference_planner.closing_constraint is None


def test_mpc_failure_invalidates_closing_history(ordinary_policy):
  planner, controller = make_planner_owner(speed=25.0)
  state = make_state(speed=25.0, target=28.0)
  warm(planner, controller, state)
  assert planner.reference_planner.closing_constraint is not None
  failing_solver = Mock(wraps=planner.mpc.solver)
  failing_solver.solve.return_value = 4
  now = inputs(state, WARM_FRAMES)
  with patch.object(planner.mpc, 'solver', failing_solver):
    planner.update(state, now_ns=now)
  assert planner.mpc.solution_status == 4
  assert not planner.motion_intent.valid
  assert planner.reference_planner.closing_constraint is None
  now = inputs(state, WARM_FRAMES + 1)
  planner.update(state, now_ns=now)
  assert planner.reference_planner.closing_constraint is None


def test_closing_bound_discards_positive_feedback_history(ordinary_policy):
  planner, controller = make_planner_owner(speed=25.0)
  state = make_state(speed=25.0, target=28.0)
  warm(planner, controller, state)
  # Inject a previous-domain correction after establishing real TRACK history;
  # the next actual wire contract must still revoke positive ACC feedback.
  controller.pid.i = 0.2
  inputs(state, WARM_FRAMES)
  with patch(SAMPLE, return_value=0.6):
    plan, outputs = consume(planner, controller, state, WARM_FRAMES)
  assert planner.reference_planner.closing_constraint is not None
  assert plan.motionIntent.responseKind == MotionResponseKind.ABSOLUTE_ACC
  assert controller.pid.i <= 0.0
  assert all(-3.5 <= value <= 0.0 for value in outputs)


def test_qualification_establishes_only_the_added_request_from_wire_history(ordinary_policy):
  planner, controller = make_planner_owner(speed=25.0)
  state = make_state(speed=25.0, target=28.0)
  previous = None
  saw_positive_ceiling = saw_negative_ceiling = False
  with patch(SAMPLE, return_value=0.6):
    for frame in range(WARM_FRAMES):
      inputs(state, frame)
      plan, _ = consume(planner, controller, state, frame)
      bound = planner.reference_planner.closing_constraint
      if bound is not None:
        assert previous is not None
        # aTarget is Float32 on the wire; the source of this rate bound is the
        # preceding published value, rather than aEgo or the native sample.
        assert bound.accel >= previous - REQUEST_DECEL_RATE * 0.05 - 1e-9
        assert bound.accel >= -2.5
        assert plan.aTarget <= min(0.6, bound.accel) + 1e-6
        assert bound.request_build_time == pytest.approx(max(previous + 2.5, 0.0) / REQUEST_DECEL_RATE)
        assert bound.response_time == pytest.approx(0.5 + 0.05 + 0.4 + bound.request_build_time)
        assert not plan.motionIntent.releaseAllowed
        saw_positive_ceiling |= bound.accel > 0.0
        saw_negative_ceiling |= bound.accel < 0.0
      previous = float(plan.aTarget)
  assert saw_positive_ceiling and saw_negative_ceiling


def test_unpublished_updates_cannot_supply_request_history(ordinary_policy):
  planner, _ = make_planner_owner(speed=25.0)
  state = make_state(speed=25.0, target=28.0)
  with patch(SAMPLE, return_value=0.6):
    for frame in range(WARM_FRAMES):
      now = inputs(state, frame)
      planner.update(state, now_ns=now)
      assert planner.reference_planner.closing_constraint is None


def test_unissued_target_and_measurement_do_not_replace_the_last_wire_request(ordinary_policy):
  planner, controller = make_planner_owner(speed=25.0)
  state = make_state(speed=25.0, target=28.0)
  plan, _ = warm(planner, controller, state)
  published = float(plan.aTarget)
  planner.output_a_target = -3.5
  state['carState'].aEgo = 1.0
  inputs(state, WARM_FRAMES)
  with patch(SAMPLE, return_value=0.6):
    plan, _ = consume(planner, controller, state, WARM_FRAMES)
  bound = planner.reference_planner.closing_constraint
  assert bound is not None
  assert bound.accel >= published - REQUEST_DECEL_RATE * 0.05 - 1e-9
  assert bound.request_build_time == pytest.approx(max(published + 2.5, 0.0) / REQUEST_DECEL_RATE)


def test_new_stronger_mpc_request_bypasses_an_establishing_comfort_ceiling(ordinary_policy):
  planner, controller = make_planner_owner(speed=25.0)
  state = make_state(speed=25.0, target=28.0)
  with patch(SAMPLE, return_value=0.6):
    for frame in range(8):
      inputs(state, frame)
      plan, _ = consume(planner, controller, state, frame)
  before = planner.reference_planner.closing_constraint
  assert before is not None and before.accel > -2.5
  inputs(state, 8)
  with patch(SAMPLE, return_value=-3.5):
    plan, _ = consume(planner, controller, state, 8)
  assert planner.reference_planner.closing_constraint is not None
  assert plan.aTarget == -3.5
  assert not plan.reference.overridden


@pytest.mark.parametrize('invalid', ['lost', 'low_probability', 'range_jump', 'velocity_jump', 'stale'])
def test_losing_a_target_removes_its_building_ceiling_immediately(invalid, ordinary_policy):
  planner, controller = make_planner_owner(speed=25.0)
  state = make_state(speed=25.0, target=28.0)
  warm(planner, controller, state)
  now = inputs(state, WARM_FRAMES)
  lead = state['radarState'].leadOne
  if invalid == 'lost':
    lead.present = False
  elif invalid == 'low_probability':
    lead.modelProb = 0.94
  elif invalid == 'range_jump':
    lead.dRel -= 10.0
  elif invalid == 'velocity_jump':
    lead.vRel -= 5.0
    lead.vLead -= 5.0
    lead.vLeadK -= 5.0
  else:
    state['radarState'].mdMonoTime = now - 250_000_001
  with patch(SAMPLE, return_value=0.6):
    plan, _ = consume(planner, controller, state, WARM_FRAMES)
  assert planner.reference_planner.closing_constraint is None
  assert plan.aTarget > 0.0
  # A valid observation immediately after the failure is still unqualified.
  inputs(state, WARM_FRAMES + 1)
  with patch(SAMPLE, return_value=0.6):
    consume(planner, controller, state, WARM_FRAMES + 1)
  assert planner.reference_planner.closing_constraint is None
