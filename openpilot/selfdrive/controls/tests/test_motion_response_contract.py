"""Explicit response permission on the same dated intent as its target.

The producer tests use a native trajectory with an injected diagnostic source
and sampled target. They test admission of deliberately ambiguous source labels,
not physical lead avoidance or uniqueness of MPC attribution.
"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanSource, get_lead_braking_urgency
from openpilot.selfdrive.controls.lib.motion_supervisor import (MOTION_INTENT_VERSION, MotionIntent, MotionPhase, MotionReferenceKind,
                                                               MotionResponseKind, StopReason)
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import UnifiedLongitudinalPlanner
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import consume, make_planner_owner, make_state
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import CruiseTarget, LongitudinalPlannerSP, LongitudinalPlanSource as PolicySource


INSTANCE_NS = 1_000_000_000


@pytest.mark.parametrize('kind', list(MotionResponseKind))
def test_response_permission_survives_real_wire_roundtrip(kind):
  planner = SimpleNamespace(longitudinal_mode=LongitudinalMode.UNIFIED)
  producer = UnifiedLongitudinalPlanner(planner)
  planner.motion_intent = MotionIntent(response_kind=int(kind), producer_instance_ns=INSTANCE_NS)
  message = log.LongitudinalPlan.new_message()
  producer.publish(message)
  assert MOTION_INTENT_VERSION == 6
  with log.LongitudinalPlan.from_bytes(message.to_bytes()) as decoded:
    assert decoded.motionIntent.responseKind == kind
    assert MotionIntent.from_message(decoded.motionIntent) == planner.motion_intent
    assert MotionIntent.from_message(decoded.motionIntent).is_valid()


@pytest.mark.parametrize('changes', [
  {'version': 4}, {'version': 5}, {'response_kind': True}, {'response_kind': 1.0}, {'response_kind': -1}, {'response_kind': 4},
  {'producer_instance_ns': 0}, {'producer_instance_ns': None}, {'ready': False}, {'valid': False},
  {'reasons': int(StopReason.FORCE_DECEL)}, {'reference_kind': int(MotionReferenceKind.MOVING)}, {'braking_accel': -0.5},
])
def test_bad_absolute_response_contract_is_rejected(changes):
  intent = MotionIntent(response_kind=int(MotionResponseKind.ABSOLUTE_ACC), producer_instance_ns=INSTANCE_NS)
  assert not replace(intent, **changes).is_valid()


@pytest.mark.parametrize('old_version', [4, 5])
def test_missing_field_and_old_version_never_acquire_an_absolute_response_contract(old_version):
  message = SimpleNamespace(version=MOTION_INTENT_VERSION, reasons=0, movingLead=False, releaseAllowed=False,
                            allowCreep=False, valid=True, mode=int(LongitudinalMode.UNIFIED), producerInstanceMonoTime=INSTANCE_NS)
  assert MotionIntent.from_message(message).response_kind == MotionResponseKind.NONE
  assert MotionIntent.from_message(message).is_valid()
  message.version = old_version
  message.responseKind = int(MotionResponseKind.ABSOLUTE_ACC)
  assert not MotionIntent.from_message(message).is_valid()


def set_lead(lead, *, present=True, closing=10.0, gap=35.0):
  lead.present = present
  lead.dRel = gap
  lead.vLead = 20.0 - closing
  lead.vRel = -closing
  lead.aLeadK = 0.0
  lead.aLeadTau = 1.5
  lead.modelProb = 0.99
  lead.yRel = 0.0


@pytest.mark.parametrize('selected', [0, 1])
def test_urgency_can_qualify_only_the_selected_observation(selected):
  state = make_state()
  leads = (state['radarState'].leadOne, state['radarState'].leadTwo)
  set_lead(leads[selected], closing=0.5)
  set_lead(leads[1 - selected])
  assert get_lead_braking_urgency(20.0, -1.0, state['radarState'], 0.3) > 0.0
  assert get_lead_braking_urgency(20.0, -1.0, state['radarState'], 0.3, lead_index=selected) == 0.0
  assert get_lead_braking_urgency(20.0, -1.0, state['radarState'], 0.3, lead_index=1 - selected) > 0.0


@pytest.mark.parametrize('selected', [-1, 2, True, 0.0, 'lead0'])
def test_invalid_selected_slot_cannot_fall_back_to_either_lead(selected):
  state = make_state()
  set_lead(state['radarState'].leadOne)
  set_lead(state['radarState'].leadTwo)
  assert get_lead_braking_urgency(20.0, -1.0, state['radarState'], 0.3, lead_index=selected) == 0.0


@pytest.fixture
def ordinary_policy():
  def targets(instance, state, speed, accel, cruise):
    instance.source = PolicySource.cruise
    instance.cruise_target = CruiseTarget(instance.source, cruise, accel)
    return instance.cruise_target

  with patch.object(LongitudinalPlannerSP, 'update', return_value=None), \
       patch.object(LongitudinalPlannerSP, 'update_targets', new=targets):
    yield


@pytest.mark.parametrize('selected', [0, 1])
@pytest.mark.parametrize('case,expected', [
  ('qualified', MotionResponseKind.LEAD_BRAKING), ('fcw', MotionResponseKind.LEAD_BRAKING),
  ('missing', MotionResponseKind.NONE), ('other_only', MotionResponseKind.NONE),
  ('weak_closing', MotionResponseKind.NONE), ('model_source', MotionResponseKind.NONE),
  ('model_override', MotionResponseKind.NONE), ('positive_target', MotionResponseKind.NONE),
  ('stale_radar', MotionResponseKind.NONE), ('forced', MotionResponseKind.NONE),
])
def test_negative_source_label_requires_same_lead_geometry_and_current_contract(selected, case, expected, ordinary_policy):
  planner, control = make_planner_owner()
  state = make_state(experimental=True)
  leads = (state['radarState'].leadOne, state['radarState'].leadTwo)
  set_lead(leads[selected], present=case not in ('missing', 'model_source'), closing=0.5 if case in ('weak_closing', 'other_only') else 10.0)
  set_lead(leads[1 - selected], present=case == 'other_only')
  if case == 'model_override':
    state['modelV2'].action.desiredAcceleration = -2.0
  state['controlsState'].forceDecel = case == 'forced'
  native_update = planner.mpc.update

  def update(*args, **kwargs):
    native_update(*args, **kwargs)
    planner.mpc.source = (LongitudinalPlanSource.e2e if case == 'model_source'
                          else (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1)[selected])
    planner.mpc.crash_cnt = 4 if case == 'fcw' else 0

  with patch.object(planner, 'is_e2e', return_value=True), patch.object(planner.mpc, 'update', side_effect=update), \
       patch.object(state, 'all_checks', return_value=case != 'stale_radar'), \
       patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.sample_reference_acceleration',
             return_value=0.3 if case == 'positive_target' else -1.0):
    plan, _ = consume(planner, control, state, 0)
  assert plan.motionIntent.responseKind == expected
  if expected == MotionResponseKind.LEAD_BRAKING:
    assert plan.aTarget < 0.0
    assert plan.motionIntent.referenceKind == MotionReferenceKind.MOVING
    assert control.motion_phase == MotionPhase.TRACK
    assert control.pid.i < 0.0


@pytest.mark.parametrize('selected', [0, 1])
@pytest.mark.parametrize('e2e,target,closing', [(True, -1.0, 10.0), (False, -1.0, 0.5), (False, -0.3, 0.5)])
def test_fcw_does_not_withdraw_existing_qualified_braking(selected, e2e, target, closing, ordinary_policy):
  planner, control = make_planner_owner()
  state = make_state(experimental=e2e)
  set_lead((state['radarState'].leadOne, state['radarState'].leadTwo)[selected], closing=closing)
  if not e2e:
    assert get_lead_braking_urgency(20.0, target, state['radarState'], planner.CP.longitudinalActuatorDelay, lead_index=selected) == 0.0
  native_update = planner.mpc.update
  crash_count = 0

  def update(*args, **kwargs):
    native_update(*args, **kwargs)
    planner.mpc.source = (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1)[selected]
    planner.mpc.crash_cnt = crash_count

  with patch.object(planner, 'is_e2e', return_value=e2e), patch.object(planner.mpc, 'update', side_effect=update), \
       patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.sample_reference_acceleration', return_value=target):
    for frame in range(8):
      crash_count = 0 if frame < 3 else 4
      before = control.pid.i
      plan, outputs = consume(planner, control, state, frame)
      expected = MotionResponseKind.LEAD_BRAKING if e2e else MotionResponseKind.ABSOLUTE_ACC
      assert plan.motionIntent.responseKind == expected
      assert planner.fcw == (frame >= 3)
      if not e2e:
        assert planner.reference_planner.model_evidence is None
        assert plan.motionIntent.referenceKind == MotionReferenceKind.UNKNOWN
      assert control.pid.i < before
      assert all(output <= plan.aTarget for output in outputs)


@pytest.mark.parametrize('selected', [0, 1])
def test_ordinary_acc_with_a_lead_cannot_gain_positive_i_before_or_during_fcw(selected, ordinary_policy):
  planner, control = make_planner_owner()
  state = make_state()
  set_lead((state['radarState'].leadOne, state['radarState'].leadTwo)[selected])
  native_update = planner.mpc.update
  crash_count = 0

  def update(*args, **kwargs):
    native_update(*args, **kwargs)
    planner.mpc.source = (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1)[selected]
    planner.mpc.crash_cnt = crash_count

  with patch.object(planner, 'is_e2e', return_value=False), patch.object(planner.mpc, 'update', side_effect=update), \
       patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.sample_reference_acceleration', return_value=0.3):
    for frame in range(6):
      crash_count = 0 if frame < 3 else 4
      plan, outputs = consume(planner, control, state, frame)
      assert plan.aTarget > 0.0 and planner.fcw == (frame >= 3)
      assert planner.reference_planner.model_evidence is None
      assert plan.motionIntent.responseKind == MotionResponseKind.ABSOLUTE_ACC
      assert control.pid.i == 0.0
      assert all(output <= plan.aTarget for output in outputs)


@pytest.mark.parametrize('boundary', ['leadOne', 'leadTwo', 'fcw', 'force_decel', 'model', 'acc_cap', 'unhealthy', 'manual'])
def test_pure_cruise_permission_and_positive_i_end_at_current_risk_boundary(boundary, ordinary_policy):
  planner, control = make_planner_owner()
  state = make_state()
  native_update = planner.mpc.update
  e2e, fcw, healthy, target = False, False, True, 0.3

  def update(*args, **kwargs):
    native_update(*args, **kwargs)
    # A cruise label alone must never hide either newly present lead.
    planner.mpc.source = LongitudinalPlanSource.cruise
    planner.mpc.crash_cnt = 4 if fcw else 0

  with patch.object(planner, 'is_e2e', side_effect=lambda _: e2e), patch.object(planner.mpc, 'update', side_effect=update), \
       patch.object(state, 'all_checks', side_effect=lambda *args, **kwargs: healthy), \
       patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.sample_reference_acceleration', side_effect=lambda *args, **kwargs: target):
    for frame in range(6):
      if frame == 3:
        assert control.pid.i > 0.0
        if boundary in ('leadOne', 'leadTwo'):
          set_lead(getattr(state['radarState'], boundary), closing=0.0, gap=200.0)
        elif boundary == 'fcw':
          fcw = True
        elif boundary == 'force_decel':
          state['controlsState'].forceDecel = True
        elif boundary == 'model':
          e2e = True
        elif boundary == 'acc_cap':
          target = 3.0
        elif boundary == 'unhealthy':
          healthy = False
        else:
          state['carState'].brakePressed = True
      state['controlsState'].longControlState = control.long_control_state
      plan, outputs = consume(planner, control, state, frame)
      if frame == 0:
        assert plan.motionIntent.responseKind == MotionResponseKind.NONE
        assert control.pid.i == 0.0
      elif frame < 3:
        assert plan.motionIntent.responseKind == MotionResponseKind.PURE_CRUISE
        assert control.pid.i > 0.0
      else:
        expected = MotionResponseKind.ABSOLUTE_ACC if boundary in ('leadOne', 'leadTwo', 'fcw', 'acc_cap') else MotionResponseKind.NONE
        assert plan.motionIntent.responseKind == expected
        assert control.pid.i == 0.0
        if boundary == 'model':
          assert plan.motionIntent.referenceKind == MotionReferenceKind.MOVING
        if boundary == 'acc_cap':
          assert planner.reference_planner.reference_overridden
          assert plan.aTarget == float(np.float32(planner.reference_planner.constraint_accel))
        if boundary in ('leadOne', 'leadTwo', 'fcw', 'model', 'acc_cap'):
          assert all(output <= plan.aTarget for output in outputs)


def test_lead_loss_threshold_crossing_and_slot_change_recompute_permission_without_hidden_i(ordinary_policy):
  planner, control = make_planner_owner()
  state = make_state(experimental=True)
  native_update = planner.mpc.update
  selected = 0

  def update(*args, **kwargs):
    native_update(*args, **kwargs)
    planner.mpc.source = (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1)[selected]
    planner.mpc.crash_cnt = 0

  cases = [(0, True, 10.0), (0, True, 0.5), (0, True, 10.0),
           (1, True, 10.0), (1, False, 10.0), (0, True, 10.0)]
  with patch.object(planner, 'is_e2e', return_value=True), patch.object(planner.mpc, 'update', side_effect=update), \
       patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.sample_reference_acceleration', return_value=-1.0):
    for frame, (selected, present, closing) in enumerate(cases):
      leads = (state['radarState'].leadOne, state['radarState'].leadTwo)
      set_lead(leads[selected], present=present, closing=closing)
      set_lead(leads[1 - selected], present=False)
      before = control.last_output_accel
      plan, outputs = consume(planner, control, state, frame)
      expected = MotionResponseKind.LEAD_BRAKING if present and closing == 10.0 else MotionResponseKind.NONE
      assert plan.motionIntent.responseKind == expected
      assert control.motion_phase == MotionPhase.TRACK
      assert not control.motion_input_constrained
      if expected == MotionResponseKind.NONE:
        assert control.pid.i == 0.0
      else:
        assert control.pid.i < 0.0
      for output in outputs:
        assert output - before <= 0.08 + 1e-12
        before = output
