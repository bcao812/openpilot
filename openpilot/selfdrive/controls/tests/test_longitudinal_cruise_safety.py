"""Current UNIFIED authority and failure guards through real plan publication and owner consumption."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.motion_supervisor import StopReason
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import consume, make_handover_state, make_planner_owner
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import decoded_commands


def braking_owner():
  planner, owner = make_planner_owner(speed=20.2)
  state = make_handover_state(speed=20.2, target=20.0, experimental=True)
  state['modelV2'].action.desiredAcceleration = -0.6
  for frame in range(4):
    consume(planner, owner, state, frame)
  assert owner.last_output_accel <= -0.6
  return planner, owner, state


@pytest.mark.parametrize('service', ['selfdriveState', 'carControl'])
def test_disabled_message_cannot_enable_shaping(service):
  planner, owner, state = braking_owner()
  state[service].enabled = False
  _, outputs = consume(planner, owner, state, 4)
  assert outputs == [0.0] * len(outputs)


@pytest.mark.parametrize('service', ['carState', 'radarState', 'modelV2', 'controlsState', 'carControl', 'selfdriveState'])
def test_required_health_gate_includes_activation_messages(service):
  planner, owner, state = braking_owner()
  previous = owner.last_output_accel
  state.all_checks = lambda service_list=None: service not in (service_list or state)
  plan, outputs = consume(planner, owner, state, 4)
  assert not planner.motion_intent.valid
  assert not plan.motionIntent.movingLead and not plan.motionIntent.releaseAllowed
  assert all(np.isfinite(output) and output <= previous for output in outputs)
  assert all(row['BRAKE_REQUEST'] and row['GAS_COMMAND'] <= 0 for row in decoded_commands(outputs))


@pytest.mark.parametrize('condition', ['fcw', 'model_stop', 'model_brake', 'model_nan', 'model_inf', 'mpc_inf'])
def test_risk_veto_is_not_dependent_on_lead_presence(condition):
  planner, owner, state = braking_owner()
  assert not state['radarState'].leadOne.present and not state['radarState'].leadTwo.present
  if condition == 'fcw':
    update = planner.mpc.update

    def observed_fcw(*args, **kwargs):
      update(*args, **kwargs)
      # Inject an observed diagnostic flag after the real native solve. This
      # guard test does not claim that this no-lead fixture caused an FCW.
      planner.mpc.crash_cnt = 5

    with patch.object(planner.mpc, 'update', side_effect=observed_fcw):
      plan, outputs = consume(planner, owner, state, 4)
    assert planner.fcw
  else:
    if condition == 'mpc_inf':
      native = planner.mpc.solver
      observer = Mock(wraps=native)

      def invalid_native_state(stage, field):
        value = native.get(stage, field)
        if stage == 1 and field == 'x':
          value = value.copy()
          value[1] = np.inf
        return value

      observer.get.side_effect = invalid_native_state
      planner.mpc.solver = observer
    else:
      state['modelV2'].action.shouldStop = condition == 'model_stop'
      state['modelV2'].action.desiredAcceleration = {
        'model_stop': 0.1, 'model_brake': -0.6, 'model_nan': float('nan'), 'model_inf': float('inf'),
      }[condition]
    plan, outputs = consume(planner, owner, state, 4)
  assert not plan.motionIntent.movingLead and not plan.motionIntent.releaseAllowed
  assert all(np.isfinite(output) and -3.5 <= output <= 0.0 for output in outputs)
  if condition == 'model_stop':
    assert plan.motionIntent.reasons & int(StopReason.E2E)
  elif condition == 'model_brake':
    assert plan.aTarget <= -0.6 and max(outputs) <= -0.6
  elif condition in ('model_nan', 'model_inf', 'mpc_inf'):
    assert not planner.motion_intent.valid
  assert all(not (row['BRAKE_REQUEST'] and row['GAS_COMMAND'] > 0) for row in decoded_commands(outputs))


def test_manual_exit_and_reentry_have_no_coast_memory():
  planner, owner, state = braking_owner()
  state['carState'].brakePressed = True
  _, outputs = consume(planner, owner, state, 4)
  assert outputs == [0.0] * len(outputs)
  state['carState'].brakePressed = False
  state['modelV2'].action.desiredAcceleration = -0.4
  plan, outputs = consume(planner, owner, state, 5)
  assert plan.motionIntent.valid
  assert plan.aTarget <= -0.4 and max(outputs) <= -0.4
