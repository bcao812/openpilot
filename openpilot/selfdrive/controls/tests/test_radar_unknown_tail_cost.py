"""Reserved native ABI parameters stay neutral after removing unknown-tail costs."""
import numpy as np
import pytest
from casadi import Function, vertcat

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  COMFORT_NORMALIZER, N, PARAM_DIM, RISK_NORMALIZER, LongitudinalMpc, MpcReference,
)
from openpilot.selfdrive.controls.tests.test_radar_fusion_mpc import EPOCH_NS, fused_state, object_lead
from openpilot.selfdrive.controls.tests.test_unified_longitudinal_mpc import radar_state


NORMALIZERS = (COMFORT_NORMALIZER, RISK_NORMALIZER)


@pytest.mark.parametrize('kind', ['formal_fusion', 'legacy', 'extra_only', 'empty', 'reset', 'direct_run'])
def test_every_native_problem_retains_fourteen_parameters_with_zero_reserved_scales(kind):
  planner = LongitudinalMpc()
  planner.set_cur_state(10.0, 0.0)
  planner.set_weights(prev_accel_constraint=False)
  planner.params[:, NORMALIZERS] = 123.0
  lead = object_lead(distance=30.0, speed=5.0)
  if kind == 'reset':
    planner.reset()
  else:
    state = (radar_state() if kind == 'legacy' else fused_state(lead) if kind == 'extra_only'
             else fused_state(display=(lead,)) if kind == 'formal_fusion' else fused_state())
    planner.update(state, reference=MpcReference(20.0), now_ns=EPOCH_NS, reference_ns=EPOCH_NS)
    assert planner.solution_status == 0
    if kind == 'direct_run':
      planner.params[:, NORMALIZERS] = 123.0
      planner.run()
      assert planner.solution_status == 0
  assert planner.params.shape == (N+1, PARAM_DIM) and PARAM_DIM == 14
  np.testing.assert_array_equal(planner.params[:, NORMALIZERS], 0.0)


def test_zero_reserved_scales_reproduce_the_original_symbolic_distance_residuals(monkeypatch):
  from acados.acados_template import AcadosModel, AcadosOcp
  monkeypatch.setattr(long_mpc, 'AcadosModel', AcadosModel, raising=False)
  monkeypatch.setattr(long_mpc, 'AcadosOcp', AcadosOcp, raising=False)
  ocp = long_mpc.gen_long_ocp()
  assert len(ocp.parameter_values) == PARAM_DIM == 14
  np.testing.assert_array_equal(ocp.parameter_values[list(NORMALIZERS)], 0.0)
  evaluate = Function('ordinary_distance_residuals', [ocp.model.x, ocp.model.p],
                      [vertcat(ocp.model.cost_y_expr[0], ocp.model.con_h_expr[3])])
  parameters = np.asarray(ocp.parameter_values).copy()
  parameters[2], parameters[4], parameters[5] = 75.0, 1.45, .75
  for speed in (0.0, 1.0, 10.0, 30.0):
    distance = speed**2/5.0 + 1.45*speed + 6.0
    expected = [(75.0-20.0-distance)/(speed+10.0), (75.0-20.0-.75*distance)/(speed+10.0)]
    np.testing.assert_allclose(np.asarray(evaluate([20.0, speed, 0.0], parameters)).ravel(), expected, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize('column', NORMALIZERS)
@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1.0, 11.0])
def test_scheduled_rest_cannot_reintroduce_fixed_tail_scales(column, value):
  initial = np.array([0.0, 1.0, 0.0])
  node = 6
  states, jerks = LongitudinalMpc.future_rest_primal(initial, node)
  parameters = np.zeros((N+1, PARAM_DIM))
  parameters[:, 0], parameters[:, 1] = -3.5, 2.0
  parameters[:, 4], parameters[:, 5] = 1.45, .75
  parameters[:, long_mpc.RISK_STOP_DISTANCE] = 6.0
  parameters[:, 2] = states[:, 0] + .75*(states[:, 1]**2/5.0 + 1.45*states[:, 1] + 6.0) + 1.0
  parameters[2, column] = value
  assert LongitudinalMpc.scheduled_rest_trajectory_error(
    initial, states, jerks, node, parameters, tolerance=1e-9,
  ) == 'invalid_scheduled_rest_trajectory'
