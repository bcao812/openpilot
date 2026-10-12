"""Current object membership and input-health faults are independent decisions."""
from unittest.mock import patch

import pytest

from openpilot.selfdrive.controls.lib.drive_helpers import STOP_SPEED
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import MPC_INVALID_INPUT
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_lead_qualification import formal_lead
from openpilot.selfdrive.controls.tests.test_radar_consumption import CONSUMPTION, PUBLICATION, radar_frame
from openpilot.selfdrive.controls.tests.test_unified_radar_consumption import enabled_planner, state_at
from openpilot.selfdrive.modeld.constants import ModelConstants


@pytest.mark.parametrize('source', ['none', 'radar', 'vision'])
def test_latest_healthy_sources_can_independently_construct_a_native_plan(source):
  planner = enabled_planner(10.0)
  lead = formal_lead(now_ns=PUBLICATION, source='vision' if source == 'vision' else 'radar',
                     dRel=60.0, vLead=11.0, vLeadK=11.0, vRel=1.0)
  objects = () if source == 'none' else (lead,)
  state = state_at(radar_frame(*objects, display=objects), speed=10.0, target=20.0)
  with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
    planner.update(state, now_ns=CONSUMPTION)
  assert planner.mpc.solution_status == 0 and planner.mpc.diagnostic_solve_calls == 1
  assert planner.motion_intent.valid and not planner.fcw
  assert not planner.output_should_stop
  assert len(planner.mpc._source_leads) == len(objects)


@pytest.mark.parametrize('fault', ['model_stale', 'estimate_stale', 'position_invalid', 'future_measurement'])
def test_retiring_measurements_does_not_hide_unhealthy_or_malformed_current_input(fault):
  planner = enabled_planner(10.0)
  lead = formal_lead(now_ns=PUBLICATION, dRel=20.0, vLead=0.0, vLeadK=0.0, vRel=-10.0)
  if fault == 'model_stale':
    lead.evidence.modelMonoTime = CONSUMPTION - 200_000_001
  elif fault == 'estimate_stale':
    lead.evidence.estimateMonoTime = CONSUMPTION - 250_000_001
  elif fault == 'position_invalid':
    lead.evidence.positionValid = False
  else:
    lead.evidence.positionMonoTime = CONSUMPTION + 1
  state = state_at(radar_frame(lead, display=(lead,)), speed=10.0)
  with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
    planner.update(state, now_ns=CONSUMPTION)
  assert planner.mpc.solution_status == MPC_INVALID_INPUT
  assert not planner.mpc.diagnostic_solve_calls and not planner.motion_intent.valid
  assert planner.output_a_target <= 0.0 and not planner.motion_intent.release_allowed


@pytest.mark.parametrize('service', ['radarState', 'modelV2'])
def test_empty_objects_cannot_turn_whole_input_transport_failure_into_healthy_motion(service):
  planner = enabled_planner(10.0)
  state = state_at(radar_frame(), speed=10.0)
  state.valid[service] = False
  # Only transport qualification is prescribed. Native planning and output
  # handling stay real; zero objects must not mask the failed service.
  state.all_checks = lambda service_list=None: all(state.valid[name] for name in service_list or state.valid)
  with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='acc'):
    planner.update(state, now_ns=CONSUMPTION)
  assert not planner.motion_intent.valid and not planner.motion_intent.release_allowed
  assert planner.output_a_target <= 0.0


@pytest.mark.parametrize('replaced', ['empty', 'different_track', 'vision'])
def test_current_forward_model_can_release_a_previous_radar_stop_without_old_id_receipt(replaced):
  planner = enabled_planner(0.0)
  old = formal_lead(now_ns=PUBLICATION, dRel=2.0, vLead=0.0, vLeadK=0.0, vRel=0.0, evidence={'trackId': 399})
  state = state_at(radar_frame(old, display=(old,)), speed=0.0)
  with patch.object(planner.dec, 'enabled', return_value=True), patch.object(planner.dec, 'mode', return_value='blended'), \
       patch.object(planner, 'is_e2e', return_value=True):
    planner.update(state, now_ns=CONSUMPTION)
    assert planner.mpc.solution_status == 0 and planner.output_should_stop
    now = CONSUMPTION + 50_000_000
    objects = () if replaced == 'empty' else (formal_lead(
      now_ns=now, source='vision' if replaced == 'vision' else 'radar',
      dRel=20.0, vLead=1.0, vLeadK=1.0, vRel=1.0, evidence={'trackId': 445}),)
    current = state_at(radar_frame(*objects, publication=now, display=objects), speed=0.0, now=now, publication=now)
    current['controlsState'].motionState.phase = 'hold'
    current['controlsState'].motionState.resolvedStop = True
    model = current['modelV2']
    model.position.x = [float(.25 * t * t) for t in ModelConstants.T_IDXS]
    model.velocity.x = [float(.5 * t) for t in ModelConstants.T_IDXS]
    model.acceleration.x = [.5] * len(ModelConstants.T_IDXS)
    model.action.desiredAcceleration = .5
    planner.update(current, now_ns=now)
  assert planner.mpc.solution_status == 0 and planner.motion_intent.valid
  assert not planner.output_should_stop and planner.motion_intent.release_allowed
  assert planner.output_a_target > 0.0
  assert all(lead.vLead >= STOP_SPEED for lead in planner.mpc._source_leads)
