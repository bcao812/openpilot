"""Shared moving-lead geometry and its current UNIFIED producer/owner contract."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.drive_helpers import STOP_SPEED
from openpilot.selfdrive.controls.lib.lead_qualification import MAX_MEASUREMENT_AGE_NS
from openpilot.selfdrive.controls.lib.longitudinal_planner import has_safe_moving_lead
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent, MotionReferenceKind
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import consume, make_handover_state, make_planner_owner
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_longitudinal_planner_creep import FRAMES_CAP
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import decoded_commands
from openpilot.selfdrive.controls.tests.test_lead_qualification import NOW, formal_lead


@pytest.mark.parametrize('source', ['radar', 'fused', 'vision'])
def test_formal_moving_follow_does_not_need_invented_visual_confidence(source):
  lead = formal_lead(source=source, dRel=7.0, vLead=.8, vRel=-.2)
  radar = SimpleNamespace(fusionActive=True, radarObjects=[lead], leadOne=lead, leadTwo=SimpleNamespace(present=False))
  assert has_safe_moving_lead(radar, 1.0, .5, now_ns=NOW)
  assert not has_safe_moving_lead(radar, 1.0, .5, now_ns=NOW + MAX_MEASUREMENT_AGE_NS + 1)


def test_distant_stopped_object_does_not_require_all_traffic_to_depart():
  moving = formal_lead(dRel=7.0, vLead=.8, vRel=-.2)
  far_stop = formal_lead(dRel=50.0, vLead=0.0, vRel=-1.0, evidence={'trackId': 8})
  radar = SimpleNamespace(fusionActive=True, radarObjects=[moving, far_stop], leadOne=moving, leadTwo=far_stop)
  assert has_safe_moving_lead(radar, 1.0, .5, now_ns=NOW)
  far_stop.dRel = 6.0
  assert not has_safe_moving_lead(radar, 1.0, .5, now_ns=NOW)


@pytest.mark.parametrize('path_state,start,end', [('entering', .3, 1.2), ('entering', 4.0, 5.0),
                                                 ('leaving', 0.0, .3), ('inPath', 0.0, 10.0)])
def test_extra_object_does_not_veto_safe_formal_moving_lead(path_state, start, end):
  moving = formal_lead(dRel=8.0, vLead=1.0, vRel=0.0)
  crossing = formal_lead(dRel=6.0, vLead=0.0, vRel=-1.0,
                        evidence={'trackId': 8, 'pathState': path_state, 'conflictStart': start, 'conflictEnd': end})
  radar = SimpleNamespace(fusionActive=True, radarObjects=[moving, crossing], leadOne=moving, leadTwo=SimpleNamespace(present=False))
  assert has_safe_moving_lead(radar, 1.0, .5, now_ns=NOW)
  # Ordinary current-lead risk still vetoes the moving-follow exception.
  radar.leadTwo = formal_lead(dRel=6.0, vLead=0.0, vRel=-1.0, evidence={'trackId': 9})
  assert not has_safe_moving_lead(radar, 1.0, .5, now_ns=NOW)


def following_state(v_ego=1.0, *, experimental=True):
  state = make_handover_state(speed=v_ego, target=100.0 / 3.6, experimental=experimental)
  state['modelV2'].action.desiredAcceleration = -0.2
  for lead in (state['radarState'].leadOne, state['radarState'].leadTwo):
    lead.present = True
    lead.dRel = 7.0
    lead.vLead = 0.8
    lead.vRel = 0.8 - v_ego
    lead.aLeadK = 0.0
    lead.aLeadTau = 1.5
    lead.modelProb = 0.999
  return state


def follow(planner, owner, state, frames=2 * FRAMES_CAP, *, start=0):
  outputs = []
  for frame in range(start, start + frames):
    state['controlsState'].longControlState = owner.long_control_state
    _, current = consume(planner, owner, state, frame)
    outputs.extend(current)
  return outputs


def follow_qualified_reference(state, acceleration):
  """The original held-input owner contract, with its explicitly given reference.

  These cases specified an MPC reference of -0.2/-0.27; they did not establish
  that a fresh optimizer would choose it while speed and gap never respond.
  The separate following plant tests the actual optimizer and delivered motion.
  """
  from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import ControlHarness

  harness = ControlHarness()
  harness.state = state['carState']
  qualified = has_safe_moving_lead(state['radarState'], harness.state.vEgo, 0.5)
  assert qualified
  intent = MotionIntent(moving_lead=qualified, reference_kind=int(MotionReferenceKind.MOVING),
                        producer_instance_ns=harness.now_ns)
  assert intent.is_valid()
  outputs = []
  for tick in range(2 * FRAMES_CAP * 5):
    outputs.append(harness.step(acceleration, intent, new_input=tick % 5 == 0))
    assert not harness.control.motion_input_constrained
  assert all(not (row['BRAKE_REQUEST'] and row['GAS_COMMAND'] > 0) for row in decoded_commands(outputs))
  return harness.control, outputs


@pytest.mark.parametrize('e2e', [False, True])
def test_sustained_moving_lead_does_not_create_an_open_loop_stop(e2e):
  state = following_state(experimental=e2e)
  before = deepcopy(state['radarState'].to_dict())
  owner, outputs = follow_qualified_reference(state, -0.2)
  assert not owner.resolved_should_stop
  assert min(outputs) >= -3.5 and max(outputs) <= 2.0
  assert state['radarState'].to_dict() == before


def test_logged_moving_lead_approach_does_not_force_a_stop():
  """Retain the 2026-09-10 route 5a input before the assisted launch at t=2202.815."""
  state = following_state(1.18)
  for lead in (state['radarState'].leadOne, state['radarState'].leadTwo):
    lead.dRel, lead.vLead, lead.vRel, lead.aLeadK = 6.1, 0.95, -0.23, -0.25
  state['modelV2'].action.desiredAcceleration = -0.27
  owner, outputs = follow_qualified_reference(state, -0.27)
  assert max(outputs) <= -0.27
  assert not owner.resolved_should_stop


@pytest.mark.parametrize('lead_name', ['leadOne', 'leadTwo'])
@pytest.mark.parametrize(
  'field,value',
  [
    ('dRel', 5.0),
    ('vLead', 0.0),
    ('vRel', -0.6),
    ('aLeadK', -1.0),
    ('modelProb', 0.5),
    ('modelProb', 1.1),
    ('dRel', float('nan')),
    ('vRel', float('inf')),
    ('vLead', float('nan')),
    ('aLeadK', float('-inf')),
    ('modelProb', float('nan')),
  ],
)
def test_any_unsafe_or_uncertain_lead_keeps_original_creep_fallback(lead_name, field, value):
  state = following_state()
  setattr(getattr(state['radarState'], lead_name), field, value)
  assert not has_safe_moving_lead(state['radarState'], 1.0, 0.5)



@pytest.mark.parametrize(
  'condition',
  [
    'no_lead',
    'only_second_lead',
    'gas',
    'brake',
    'override',
    'long_inactive',
    'disabled',
    'control_disabled',
    'standstill',
    'force_decel',
    'model_stop',
    'model_brake',
    'model_nan',
    'unhealthy',
  ],
)
def test_following_bypass_requires_all_activation_and_risk_guards(condition):
  planner, owner = make_planner_owner(speed=1.0)
  state = following_state()
  if condition in ('no_lead', 'only_second_lead'):
    state['radarState'].leadOne.present = False
    state['radarState'].leadTwo.present = condition == 'only_second_lead'
  elif condition in ('gas', 'brake', 'standstill'):
    setattr(state['carState'], {'gas': 'gasPressed', 'brake': 'brakePressed', 'standstill': 'standstill'}[condition], True)
  elif condition == 'override':
    state['carControl'].cruiseControl.override = True
  elif condition == 'long_inactive':
    state['carControl'].longActive = False
  elif condition == 'disabled':
    state['selfdriveState'].enabled = False
  elif condition == 'control_disabled':
    state['carControl'].enabled = False
  elif condition == 'force_decel':
    state['controlsState'].forceDecel = True
  elif condition == 'model_stop':
    state['modelV2'].action.shouldStop = True
  elif condition in ('model_brake', 'model_nan'):
    state['modelV2'].action.desiredAcceleration = -0.6 if condition == 'model_brake' else float('nan')
  elif condition == 'unhealthy':
    state.all_checks = lambda service_list=None: False
  outputs = follow(planner, owner, state, frames=1)
  assert owner.motion_supervisor.moving_lead_t == 0.0
  assert all(np.isfinite(value) and -3.5 <= value <= 2.0 for value in outputs)
  if condition in ('gas', 'brake', 'override', 'long_inactive', 'disabled', 'control_disabled'):
    assert outputs == [0.0] * len(outputs)


@pytest.mark.parametrize('service', ['carState', 'radarState', 'modelV2', 'controlsState', 'carControl', 'selfdriveState'])
def test_stale_input_discards_confirmation_immediately(service):
  planner, owner = make_planner_owner(speed=1.0)
  state = following_state()
  follow(planner, owner, state)
  state.all_checks = lambda service_list=None: service not in (service_list or state)
  follow(planner, owner, state, frames=1, start=2 * FRAMES_CAP)
  assert owner.motion_supervisor.moving_lead_t == 0.0


def test_flickering_lead_cannot_accumulate_confirmation():
  planner, owner = make_planner_owner(speed=1.0)
  state = following_state()
  for frame in range(2 * FRAMES_CAP):
    state['radarState'].leadOne.present = frame % 4 != 0
    follow(planner, owner, state, frames=1, start=frame)
    assert owner.motion_supervisor.moving_lead_t <= 3 * DT_MDL


def test_stronger_braking_is_never_delayed_by_following():
  planner, owner = make_planner_owner(speed=1.0)
  state = following_state()
  follow(planner, owner, state)
  state['modelV2'].action.desiredAcceleration = -2.0
  outputs = follow(planner, owner, state, frames=1, start=2 * FRAMES_CAP)
  assert max(outputs) <= -2.0
  assert owner.motion_supervisor.moving_lead_t == 0.0


@pytest.mark.parametrize('v_ego', [float('nan'), float('inf'), -0.1, 0.0, STOP_SPEED - 0.001, 2.0, 20.0])
def test_moving_lead_guard_is_limited_to_rolling_low_speed(v_ego):
  state = following_state()
  assert not has_safe_moving_lead(state['radarState'], v_ego, 0.5)


@pytest.mark.parametrize('delay', [-0.1, 1.01, float('nan'), float('inf')])
def test_unknown_or_excessive_actuator_delay_cannot_relax_the_fallback(delay):
  state = following_state()
  assert not has_safe_moving_lead(state['radarState'], 1.0, delay)
