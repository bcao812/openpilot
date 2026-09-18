#!/usr/bin/env python3
import os
import time
from dataclasses import dataclass
from numbers import Integral
import numpy as np
from openpilot.cereal import log
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
# WARNING: imports outside of constants will not trigger a rebuild
from openpilot.selfdrive.modeld.constants import index_function
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU
from openpilot.selfdrive.controls.lib.model_reference import ModelReferenceEvidence, PREFERENCE_TIME_S

if __name__ == '__main__':  # generating code
  from acados.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
else:
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython

from casadi import SX, fmax, fmin, if_else, vertcat

MODEL_NAME = 'long'
LONG_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LONG_MPC_DIR, "c_generated_code")
JSON_FILE = os.path.join(LONG_MPC_DIR, "acados_ocp_long.json")

LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource
MPC_SOURCES = (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1)

X_DIM = 3
U_DIM = 1
PARAM_DIM = 10
MODEL_V_REF = 6
MODEL_A_REF = 7
MODEL_BRAKING_PRIORITY = 8
TYPED_MODEL_MODE = 9
COST_E_DIM = 5
COST_DIM = COST_E_DIM + 1
CONSTR_DIM = 4

X_EGO_OBSTACLE_COST = 3.
X_EGO_COST = 0.
V_EGO_COST = 0.
A_EGO_COST = 0.
J_EGO_COST = 5.
A_CHANGE_COST = 200.
DANGER_ZONE_COST = 100.
CRASH_DISTANCE = .25
LEAD_DANGER_FACTOR = 0.75
LIMIT_COST = 1e6
ACADOS_SOLVER_TYPE = 'SQP_RTI'

# Fewer timestamps don't hurt performance and lead to
# much better convergence of the MPC with low iterations
N = 12
MAX_T = 10.0
T_IDXS_LST = [index_function(idx, max_val=MAX_T, max_idx=N) for idx in range(N+1)]

T_IDXS = np.array(T_IDXS_LST)
FCW_IDXS = T_IDXS < 5.0
T_DIFFS = np.diff(T_IDXS, prepend=[0.])
COMFORT_BRAKE = 2.5
STOP_DISTANCE = 6.0
MIN_X_LEAD_FACTOR = 0.5
MPC_INVALID_INPUT = -1
MPC_INVALID_SOLUTION = -2
MPC_SOLVER_ERROR = -3
MPC_GOAL_INFEASIBLE = -4
MPC_GOAL_UNSATISFIED = -5
REFERENCE_BOUND = 1e4
REST_STATE_TOL = 1e-3
# Numerical feasibility is separate from the physical rest acceptance box.
# Keep equality/inequality residuals six orders below that unchanged box.
QP_FEASIBILITY_TOL = 1e-9


@dataclass(frozen=True)
class MpcReference:
  cruise_speed: float | np.ndarray
  accel_min: float | np.ndarray = ACCEL_MIN
  accel_max: float | np.ndarray = ACCEL_MAX
  cruise_min_accel: float | np.ndarray = -1.2
  model_position: np.ndarray | None = None
  model_velocity: np.ndarray | None = None
  model_acceleration: np.ndarray | None = None
  typed_model: ModelReferenceEvidence | None = None
  preference_time_s: float = PREFERENCE_TIME_S

  def __post_init__(self):
    for field in ('cruise_speed', 'accel_min', 'accel_max', 'cruise_min_accel', 'model_position', 'model_velocity', 'model_acceleration'):
      values = getattr(self, field)
      if values is not None and not np.isscalar(values):
        immutable_values = np.array(values, dtype=float, copy=True)
        immutable_values.setflags(write=False)
        object.__setattr__(self, field, immutable_values)


def get_jerk_factor(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 1.0
  elif personality==log.LongitudinalPersonality.standard:
    return 1.0
  elif personality==log.LongitudinalPersonality.aggressive:
    return 0.5
  else:
    raise NotImplementedError("Longitudinal personality not supported")


def get_T_FOLLOW(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 1.75
  elif personality==log.LongitudinalPersonality.standard:
    return 1.45
  elif personality==log.LongitudinalPersonality.aggressive:
    return 1.25
  else:
    raise NotImplementedError("Longitudinal personality not supported")

def get_stopped_equivalence_factor(v_lead):
  return (v_lead**2) / (2 * COMFORT_BRAKE)

def get_safe_obstacle_distance(v_ego, t_follow):
  return (v_ego**2) / (2 * COMFORT_BRAKE) + t_follow * v_ego + STOP_DISTANCE

def gen_long_model():
  model = AcadosModel()
  model.name = MODEL_NAME

  # states
  x_ego, v_ego, a_ego = SX.sym('x_ego'), SX.sym('v_ego'), SX.sym('a_ego')
  model.x = vertcat(x_ego, v_ego, a_ego)

  # controls
  j_ego = SX.sym('j_ego')
  model.u = vertcat(j_ego)

  # xdot
  x_ego_dot = SX.sym('x_ego_dot')
  v_ego_dot = SX.sym('v_ego_dot')
  a_ego_dot = SX.sym('a_ego_dot')
  model.xdot = vertcat(x_ego_dot, v_ego_dot, a_ego_dot)

  # runtime parameters
  a_min = SX.sym('a_min')
  a_max = SX.sym('a_max')
  x_obstacle = SX.sym('x_obstacle')
  a_prev = SX.sym('a_prev')
  lead_t_follow = SX.sym('lead_t_follow')
  lead_danger_factor = SX.sym('lead_danger_factor')
  model_v_ref = SX.sym('model_v_ref')
  model_a_ref = SX.sym('model_a_ref')
  model_braking_priority = SX.sym('model_braking_priority')
  typed_model_mode = SX.sym('typed_model_mode')
  model.p = vertcat(a_min, a_max, x_obstacle, a_prev, lead_t_follow, lead_danger_factor,
                   model_v_ref, model_a_ref, model_braking_priority, typed_model_mode)

  # dynamics model
  f_expl = vertcat(v_ego, a_ego, j_ego)
  model.f_impl_expr = model.xdot - f_expl
  model.f_expl_expr = f_expl
  return model

def gen_long_ocp():
  ocp = AcadosOcp()
  ocp.model = gen_long_model()

  Tf = T_IDXS[-1]

  # set dimensions
  ocp.dims.N = N

  # set cost module
  ocp.cost.cost_type = 'NONLINEAR_LS'
  ocp.cost.cost_type_e = 'NONLINEAR_LS'

  QR = np.zeros((COST_DIM, COST_DIM))
  Q = np.zeros((COST_E_DIM, COST_E_DIM))

  ocp.cost.W = QR
  ocp.cost.W_e = Q

  x_ego, v_ego, a_ego = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2]
  j_ego = ocp.model.u[0]

  a_min, a_max = ocp.model.p[0], ocp.model.p[1]
  x_obstacle = ocp.model.p[2]
  a_prev = ocp.model.p[3]
  lead_t_follow = ocp.model.p[4]
  lead_danger_factor = ocp.model.p[5]
  model_v_ref = ocp.model.p[MODEL_V_REF]
  model_a_ref = ocp.model.p[MODEL_A_REF]
  model_braking_priority = ocp.model.p[MODEL_BRAKING_PRIORITY]
  typed_model_mode = ocp.model.p[TYPED_MODEL_MODE]

  ocp.cost.yref = np.zeros((COST_DIM, ))
  ocp.cost.yref_e = np.zeros((COST_E_DIM, ))

  desired_dist_comfort = get_safe_obstacle_distance(v_ego, lead_t_follow)

  # The main cost in normal operation is how close you are to the "desired" distance
  # from an obstacle at every timestep. This obstacle can be a lead car
  # or other object. In e2e mode we can use x_position targets as a cost
  # instead.
  distance_error = ((x_obstacle - x_ego) - desired_dist_comfort) / (v_ego + 10.)
  velocity_error, acceleration_error = v_ego - model_v_ref, a_ego - model_a_ref
  # A typed ego preference does not ask us to chase a distant lead. Where
  # admitted lead geometry is tighter than that preference, stronger braking
  # carries no model tracking penalty. Legacy/Unknown keep the original cost.
  distance_cost = if_else(typed_model_mode > 0.5, fmin(distance_error, 0.0), distance_error)
  velocity_cost = if_else(model_braking_priority > 0.5, fmax(velocity_error, 0.0), velocity_error)
  acceleration_cost = if_else(model_braking_priority > 0.5, fmax(acceleration_error, 0.0), acceleration_error)
  costs = [distance_cost,
           x_ego,
           velocity_cost,
           acceleration_cost,
           a_ego - a_prev,
           j_ego]
  ocp.model.cost_y_expr = vertcat(*costs)
  ocp.model.cost_y_expr_e = vertcat(*costs[:-1])

  # Constraints on speed, acceleration and desired distance to
  # the obstacle, which is treated as a slack constraint so it
  # behaves like an asymmetrical cost.
  constraints = vertcat(v_ego,
                        (a_ego - a_min),
                        (a_max - a_ego),
                        ((x_obstacle - x_ego) - lead_danger_factor * (desired_dist_comfort)) / (v_ego + 10.))
  ocp.model.con_h_expr = constraints

  x0 = np.zeros(X_DIM)
  ocp.constraints.x0 = x0
  # Every mode carries the same native layout. Ordinary stages receive wide
  # bounds; an admitted typed rest goal enables v=a=0 from its stated node.
  # Initial (x,v,a) stays tied to the measured carState, including residual v.
  ocp.constraints.idxbx = np.array([1, 2])
  ocp.constraints.lbx = np.full(2, -REFERENCE_BOUND)
  ocp.constraints.ubx = np.full(2, REFERENCE_BOUND)
  ocp.constraints.idxbx_e = np.array([1, 2])
  ocp.constraints.lbx_e = np.full(2, -REFERENCE_BOUND)
  ocp.constraints.ubx_e = np.full(2, REFERENCE_BOUND)
  ocp.parameter_values = np.array([-1.2, 1.2, 0.0, 0.0, get_T_FOLLOW(), LEAD_DANGER_FACTOR, 0.0, 0.0, 0.0, 0.0])


  # We put all constraint cost weights to 0 and only set them at runtime
  cost_weights = np.zeros(CONSTR_DIM)
  ocp.cost.zl = cost_weights
  ocp.cost.Zl = cost_weights
  ocp.cost.Zu = cost_weights
  ocp.cost.zu = cost_weights

  ocp.constraints.lh = np.zeros(CONSTR_DIM)
  ocp.constraints.uh = 1e4*np.ones(CONSTR_DIM)
  ocp.constraints.idxsh = np.arange(CONSTR_DIM)

  # Typed reference stages can have a semidefinite state Hessian: position
  # has no tracking cost and the one-sided lead cost can be inactive. The
  # square-root Riccati algorithms require a positive definite full/state
  # Hessian. Classical Riccati only needs the reduced control Hessian to be
  # positive definite, supplied here by the unchanged jerk cost.
  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = ACADOS_SOLVER_TYPE
  ocp.solver_options.qp_solver_cond_N = 1
  ocp.solver_options.qp_solver_ric_alg = 0
  ocp.solver_options.qp_solver_cond_ric_alg = 0

  # More iterations take too much time and less lead to inaccurate convergence in
  # some situations. Ideally we would run just 1 iteration to ensure fixed runtime.
  ocp.solver_options.qp_solver_iter_max = 10
  ocp.solver_options.qp_tol = 1e-3
  ocp.solver_options.qp_solver_tol_eq = QP_FEASIBILITY_TOL
  ocp.solver_options.qp_solver_tol_ineq = QP_FEASIBILITY_TOL

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = T_IDXS

  ocp.code_export_directory = EXPORT_DIR
  return ocp


class LongitudinalMpc:
  def __init__(self, dt=DT_MDL):
    self.dt = dt
    self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
    self.diagnostic_solve_calls = 0
    self.diagnostic_reset_calls = 0
    self.diagnostic_last_solver_return = None
    self.diagnostic_last_x0 = None
    self.diagnostic_details_enabled = False
    self.diagnostic_last_parameters = None
    self.diagnostic_last_yref = None
    self.diagnostic_last_a_prev = None
    self.diagnostic_obstacles_enabled = False
    self.diagnostic_last_obstacle_summary = None
    self.diagnostic_obstacle_failures = 0
    # UNIFIED dates its reference with the consumer clock. Keep the clock high
    # water across solver resets; reset buffers must not admit a replayed epoch.
    self._reference_clock_ns = 0
    self.reset()
    self.source = LongitudinalPlanSource.cruise

  def reset(self):
    self.diagnostic_reset_calls += 1
    self.solver.reset()

    self.x_sol = np.zeros((N+1, X_DIM))
    self.u_sol = np.zeros((N, 1))
    self.v_solution = np.zeros(N+1)
    self.a_solution = np.zeros(N+1)
    self.j_solution = np.zeros(N)
    self.a_prev = np.array(self.a_solution)
    # Zero-filled reset buffers are not a previous solved trajectory. Callers
    # choose whether that history may constrain their next reference.
    self.previous_solution_valid = False
    self.invalidate_reference_history()
    self.yref = np.zeros((N+1, COST_DIM))

    for i in range(N):
      self.solver.cost_set(i, "yref", self.yref[i])
    self.solver.cost_set(N, "yref", self.yref[N][:COST_E_DIM])

    self.params = np.zeros((N+1, PARAM_DIM))
    for i in range(N+1):
      self.solver.set(i, 'x', np.zeros(X_DIM))

    self.last_cloudlog_t = 0
    self.crash_cnt = 0.0
    self.solution_status = 0
    self.last_solver_status = 0
    self.failure_reason = ''
    self.rest_goal = None
    self.rest_goal_state = 'none'
    self.rest_goal_reason = ''
    self.rest_goal_braking_accel = None
    self.diagnostic_rest_goal = None
    self.diagnostic_reference_weights = None
    self.diagnostic_reference_bounds = None
    # timers
    self.solve_time = 0.0
    self.x0 = np.zeros(X_DIM)
    self.set_weights()

  @staticmethod
  def accel_change_cost_scale(times):
    return np.interp(times, [0.0, 1.0, 2.0], [1.0, 1.0, 0.0])

  def invalidate_reference_history(self):
    """Drop UNIFIED reference provenance without moving the consumer clock."""
    self._reference_epoch_ns = None
    self._reference_acceleration = None
    self._reference_pending_ns = None
    self._reference_pending_solved = False

  def prepare_reference_history(self, now_ns, *, valid):
    """Align one preceding native reference with this consumer's real epoch.

    This is a smoothing reference, never an allocated or pending actuator
    request. LEGACY/SUPERVISED continue to use run()'s original fixed-dt history.
    A new epoch is committed only after the caller accepts a successful output.
    """
    epoch_ns = self._reference_epoch_ns
    acceleration = self._reference_acceleration
    self.invalidate_reference_history()
    self.a_prev = np.zeros_like(self.a_solution)
    if (not isinstance(now_ns, Integral) or isinstance(now_ns, bool)
        or not 0 < now_ns < (1 << 64) or now_ns <= self._reference_clock_ns):
      return False
    now_ns = int(now_ns)
    self._reference_clock_ns = now_ns
    self._reference_pending_ns = now_ns
    if not valid or not self.previous_solution_valid or epoch_ns is None or acceleration is None:
      return False
    elapsed = (now_ns - epoch_ns) / 1e9
    # Only stages carrying a nonzero history cost need support from the old
    # trajectory. Do not give extrapolated terminal values a smoothing penalty.
    weighted_times = T_IDXS[self.accel_change_cost_scale(T_IDXS) > 0.0]
    if (not 0.0 < elapsed or weighted_times[-1] + elapsed > T_IDXS[-1]
        or acceleration.shape != T_IDXS.shape or not np.all(np.isfinite(acceleration))):
      return False
    self.a_prev = np.interp(T_IDXS + elapsed, T_IDXS, acceleration)
    return True

  def mark_reference_history(self, *, valid):
    """Commit the native reference only after a qualified UNIFIED output."""
    epoch_ns = self._reference_pending_ns
    solved = self._reference_pending_solved
    self.invalidate_reference_history()
    if (not valid or epoch_ns is None or not solved or not self.previous_solution_valid or self.solution_status != 0
        or self.a_solution.shape != T_IDXS.shape or not np.all(np.isfinite(self.a_solution))):
      return False
    self._reference_epoch_ns = epoch_ns
    # The next native solve overwrites x_sol, which backs a_solution. Preserve
    # this accepted reference rather than dating a later/failed solve as it.
    self._reference_acceleration = self.a_solution.copy()
    return True

  def discard_prepared_reference_cost(self):
    """Reject the preceding semantics, retaining this qualified pending epoch.

    prepare_reference_history already removed the preceding stored reference.
    A new rest goal can replace its cost without losing the new successful
    trajectory that the next frame will be allowed to smooth against.
    """
    self.a_prev[:] = 0.0

  def set_cost_weights(self, cost_weights, constraint_cost_weights):
    self._cost_weights = np.asarray(cost_weights, dtype=float).copy()
    W = np.asfortranarray(np.diag(cost_weights))
    for i in range(N):
      # TODO don't hardcode A_CHANGE_COST idx
      # reduce the cost on (a-a_prev) later in the horizon.
      W[4,4] = cost_weights[4] * self.accel_change_cost_scale(T_IDXS[i])
      self.solver.cost_set(i, 'W', W)
    # Setting the slice without the copy make the array not contiguous,
    # causing issues with the C interface.
    self.solver.cost_set(N, 'W', np.copy(W[:COST_E_DIM, :COST_E_DIM]))

    # Set L2 slack cost on lower bound constraints
    Zl = np.array(constraint_cost_weights)
    for i in range(N):
      self.solver.cost_set(i, 'Zl', Zl)

  def set_weights(self, prev_accel_constraint=True, personality=log.LongitudinalPersonality.standard, a_change_scale=1.0):
    jerk_factor = get_jerk_factor(personality)
    a_change_scale = float(np.clip(a_change_scale, 0.0, 1.0)) if np.isfinite(a_change_scale) else 1.0
    a_change_cost = A_CHANGE_COST * a_change_scale if prev_accel_constraint else 0
    cost_weights = [X_EGO_OBSTACLE_COST, X_EGO_COST, V_EGO_COST, A_EGO_COST, jerk_factor * a_change_cost, jerk_factor * J_EGO_COST]
    constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, DANGER_ZONE_COST]
    self.set_cost_weights(cost_weights, constraint_cost_weights)

  def set_cur_state(self, v, a):
    v_prev = self.x0[1]
    self.x0[1] = v
    self.x0[2] = a
    if not np.all(np.isfinite(self.x0)):
      return
    if abs(v_prev - v) > 2.:  # probably only helps if v < v_prev
      for i in range(N+1):
        self.solver.set(i, 'x', self.x0)

  @staticmethod
  def extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau):
    a_lead_traj = a_lead * np.exp(-a_lead_tau * (T_IDXS**2)/2.)
    v_lead_traj = np.clip(v_lead + np.cumsum(T_DIFFS * a_lead_traj), 0.0, 1e8)
    x_lead_traj = x_lead + np.cumsum(T_DIFFS * v_lead_traj)
    lead_xv = np.column_stack((x_lead_traj, v_lead_traj))
    return lead_xv

  def process_lead(self, lead):
    v_ego = self.x0[1]
    if lead is not None and lead.present:
      x_lead = lead.dRel
      v_lead = lead.vLead
      a_lead = lead.aLeadK
      a_lead_tau = lead.aLeadTau
    else:
      # Fake a fast lead car, so mpc can keep running in the same mode
      x_lead = 50.0
      v_lead = v_ego + 10.0
      a_lead = 0.0
      a_lead_tau = _LEAD_ACCEL_TAU

    # MPC will not converge if immediate crash is expected
    # Clip lead distance to what is still possible to brake for
    min_x_lead = MIN_X_LEAD_FACTOR * (v_ego + v_lead) * (v_ego - v_lead) / (-ACCEL_MIN * 2)
    x_lead = np.clip(x_lead, min_x_lead, 1e8)
    v_lead = np.clip(v_lead, 0.0, 1e8)
    a_lead = np.clip(a_lead, -10., 5.)
    lead_xv = self.extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau)
    return lead_xv

  def reference_obstacles(self, reference, t_follow):
    if reference.typed_model is not None and not isinstance(reference.typed_model, ModelReferenceEvidence):
      raise ValueError('Invalid typed model reference')
    cruise_speed = np.broadcast_to(np.asarray(reference.cruise_speed, dtype=float), (N+1,))
    accel_min = np.broadcast_to(np.asarray(reference.accel_min, dtype=float), (N+1,))
    accel_max = np.broadcast_to(np.asarray(reference.accel_max, dtype=float), (N+1,))
    cruise_min_accel = np.broadcast_to(np.asarray(reference.cruise_min_accel, dtype=float), (N+1,))
    if (not np.all(np.isfinite([cruise_speed, accel_min, accel_max, cruise_min_accel]))
        or np.any(cruise_speed < 0.0) or np.any(cruise_min_accel > 0.0)
        or np.any(accel_min > accel_max) or np.any(accel_min > ACCEL_MAX) or np.any(accel_max < ACCEL_MIN)):
      raise ValueError("Invalid cruise reference")

    accel_min = np.clip(accel_min, ACCEL_MIN, ACCEL_MAX)
    accel_max = np.clip(accel_max, ACCEL_MIN, ACCEL_MAX)
    cruise_min_accel = np.clip(cruise_min_accel, accel_min, accel_max)
    lower_speed = np.maximum(0.0, self.x0[1] + np.cumsum(T_DIFFS * cruise_min_accel))
    upper_speed = np.maximum(0.0, self.x0[1] + np.cumsum(T_DIFFS * accel_max))
    cruise_velocity = np.clip(cruise_speed, lower_speed, upper_speed)
    cruise_position = np.cumsum(T_DIFFS * cruise_velocity)
    obstacles = [cruise_position + get_safe_obstacle_distance(cruise_velocity, t_follow)]
    sources = [LongitudinalPlanSource.cruise]

    model_reference = (reference.model_position, reference.model_velocity, reference.model_acceleration)
    if any(values is not None for values in model_reference):
      if any(values is None for values in model_reference):
        raise ValueError("Incomplete model reference")
      model_position, model_velocity, model_acceleration = (np.asarray(values, dtype=float) for values in model_reference)
      if (any(values.shape != (N+1,) for values in (model_position, model_velocity, model_acceleration))
          or not np.all(np.isfinite([model_position, model_velocity, model_acceleration]))
          or np.any(model_velocity < 0.0) or np.any(np.diff(model_position) < 0.0)):
        raise ValueError("Invalid model reference")
      # Compatibility snapshots remain independently inspectable. The typed
      # production path consumes its model as ego evidence, never as a lead.
      if reference.typed_model is None or reference.typed_model.unknown is not None:
        obstacles.append(model_position - model_position[0] + get_safe_obstacle_distance(model_velocity, t_follow))
        sources.append(LongitudinalPlanSource.e2e)
    return obstacles, sources, accel_min, accel_max

  @staticmethod
  def integrate_jerk(state, jerk, duration):
    """Exact constant-jerk dynamics, keeping all three states consistent."""
    position, velocity, acceleration = state
    next_acceleration = acceleration + jerk * duration
    return np.array([position + velocity * duration + (2.0 * acceleration + next_acceleration) * duration**2 / 6.0,
                     velocity + 0.5 * (acceleration + next_acceleration) * duration,
                     next_acceleration])

  @staticmethod
  def rest_acceleration_interval(state, duration, lower, upper):
    """Next-acceleration interval for one nonnegative constant-jerk segment.

    Endpoint speed is v+(a+a_next)*dt/2. A convex velocity can also have an
    interior minimum, which endpoint boxes alone do not constrain.
    """
    _, velocity, acceleration = state
    if velocity < 0.0 or (velocity == 0.0 and acceleration < 0.0):
      return None
    minimum = max(float(lower[1]), 2.0 * (lower[0] - velocity) / duration - acceleration)
    maximum = min(float(upper[1]), 2.0 * (upper[0] - velocity) / duration - acceleration)
    if acceleration < 0.0 and velocity < -0.5 * acceleration * duration:
      # Here an endpoint-feasible segment must cross from negative to positive
      # acceleration. Its v_min >= 0 imposes this additional lower bound.
      minimum = max(minimum, acceleration + acceleration**2 * duration / (2.0 * velocity))
    return None if minimum > maximum else (minimum, maximum)

  @staticmethod
  def segment_minimum_velocity(state, jerk, duration):
    _, velocity, acceleration = state
    minimum = min(velocity, velocity + acceleration * duration + 0.5 * jerk * duration**2)
    if jerk > 0.0 and 0.0 < -acceleration / jerk < duration:
      minimum = min(minimum, velocity - acceleration**2 / (2.0 * jerk))
    return float(minimum)

  def project_due_rest_precision(self):
    """Remove bounded numerical defects by changing jerk, never clipping x/v/a.

    Only a successful native due-rest solve reaches this method. The raw
    iterate must already satisfy the much smaller QP feasibility budget.
    Corrections are bounded by the jerk effect on endpoint v/a; the resulting
    x/v/a difference is independently bounded through the linear dynamics.
    """
    durations = np.diff(T_IDXS)
    lower, upper = self.diagnostic_reference_bounds
    raw_states, raw_jerks = self.x_sol, self.u_sol[:, 0]
    defects = np.array([raw_states[stage+1] - self.integrate_jerk(raw_states[stage], raw_jerks[stage], duration)
                        for stage, duration in enumerate(durations)])
    maximum_defect = np.maximum(np.max(np.abs(defects), axis=0), np.abs(raw_states[0] - self.x0))
    maximum_violation = float(max(0.0, np.max(lower[1:] - raw_states[1:, 1:]),
                                  np.max(raw_states[1:, 1:] - upper[1:])))
    diagnostics = {'applied': False, 'feasibilityTolerance': QP_FEASIBILITY_TOL,
                   'maximumNativeDynamicsDefect': maximum_defect.tolist(),
                   'maximumNativeBoundViolation': maximum_violation}
    self.diagnostic_rest_goal['numericalProjection'] = diagnostics
    if np.max(maximum_defect) > QP_FEASIBILITY_TOL:
      return 'native_rest_dynamics_residual_exceeded'
    if maximum_violation > QP_FEASIBILITY_TOL:
      return 'native_rest_precision_budget_exceeded'

    states, jerks = np.empty_like(raw_states), np.empty_like(raw_jerks)
    states[0] = self.x0
    difference_budget = np.full(3, QP_FEASIBILITY_TOL)
    largest_budget = difference_budget.copy()
    largest_jerk_correction = 0.0
    for stage, duration in enumerate(durations):
      interval = self.rest_acceleration_interval(states[stage], duration, lower[stage+1], upper[stage+1])
      if interval is None:
        return 'rest_precision_no_feasible_numeric_trajectory'
      minimum, maximum = interval
      acceleration = states[stage, 2]
      desired = acceleration + raw_jerks[stage] * duration
      next_acceleration = min(max(desired, minimum), maximum)
      jerk = (next_acceleration - acceleration) / duration
      # Bound arithmetic can round one ULP outside a closed face. Test the
      # adjacent representable controls as well; the states are always obtained
      # from dynamics, including the degenerate corner with only one control.
      candidates = [jerk, np.nextafter(jerk, -np.inf), np.nextafter(jerk, np.inf)]
      candidates.sort(key=lambda candidate: abs(candidate - raw_jerks[stage]))
      selected = None
      for candidate in candidates:
        next_state = self.integrate_jerk(states[stage], candidate, duration)
        if (minimum <= next_state[2] <= maximum
            and np.all(next_state[1:] >= lower[stage+1]) and np.all(next_state[1:] <= upper[stage+1])
            and self.segment_minimum_velocity(states[stage], candidate, duration) >= 0.0):
          selected = candidate, next_state
          break
      if selected is None:
        return 'rest_precision_no_feasible_numeric_trajectory'
      jerks[stage], states[stage+1] = selected
      # A jerk changes endpoint acceleration by j*dt and velocity by j*dt²/2.
      # The sum is conservative for an independent epsilon error in each; it
      # does not borrow the physical 1e-3 rest tolerance as a numerical budget.
      jerk_budget = QP_FEASIBILITY_TOL * (1.0 / duration + 2.0 / duration**2)
      correction = abs(jerks[stage] - raw_jerks[stage])
      if correction > jerk_budget:
        return 'native_rest_precision_budget_exceeded'
      largest_jerk_correction = max(largest_jerk_correction, correction)
      transition = np.array([[1.0, duration, 0.5 * duration**2], [0.0, 1.0, duration], [0.0, 0.0, 1.0]])
      control = np.array([duration**3 / 6.0, 0.5 * duration**2, duration])
      difference_budget = transition @ difference_budget + control * jerk_budget + QP_FEASIBILITY_TOL
      if np.any(np.abs(states[stage+1] - raw_states[stage+1]) > difference_budget):
        return 'native_rest_precision_budget_exceeded'
      largest_budget = np.maximum(largest_budget, difference_budget)

    diagnostics.update(applied=bool(np.any(states != raw_states) or np.any(jerks != raw_jerks)),
                       maximumStateCorrection=np.max(np.abs(states - raw_states), axis=0).tolist(),
                       maximumStateCorrectionBudget=largest_budget.tolist(),
                       maximumJerkCorrection=largest_jerk_correction)
    self.x_sol[:] = states
    self.u_sol[:, 0] = jerks
    # These are views established by run(). Keep the next reference's history
    # tied to the same corrected trajectory, not its pre-projection version.
    self.a_prev = np.interp(T_IDXS + self.dt, T_IDXS, self.a_solution)
    return ''

  @staticmethod
  def due_rest_primal(initial_state):
    """Feasible rest-precision nodes with continuous nonnegative velocity.

    This constructs a warm start, not an accepted native solution. If a tiny
    forward speed cannot outlast its negative acceleration within the first
    segment's rest-precision bounds, keep the due goal explicitly infeasible.
    """
    if initial_state[1] < 0.0 or np.max(np.abs(initial_state[1:])) > REST_STATE_TOL:
      return None
    states = np.zeros((N+1, 3))
    jerks = np.zeros(N)
    states[0] = initial_state
    for stage, duration in enumerate(np.diff(T_IDXS)):
      position, velocity, acceleration = states[stage]
      next_acceleration = 0.0
      jerk = -acceleration / duration
      next_velocity = velocity + 0.5 * acceleration * duration
      if next_velocity > REST_STATE_TOL:
        # Concave velocity: its minimum is at one of the equal endpoints.
        next_acceleration = -acceleration
        jerk = -2.0 * acceleration / duration
        next_velocity = velocity
      elif next_velocity < 0.0:
        if velocity <= 0.0:
          return None
        # The minimum of convex v(t) is v-a^2/(2*j). Use an upward ULP
        # to avoid rounding that necessary zero-speed boundary downward.
        jerk = np.nextafter(acceleration**2 / (2.0 * velocity), np.inf)
        next_acceleration = acceleration + jerk * duration
        next_velocity = velocity + acceleration * duration + 0.5 * jerk * duration**2
      if not 0.0 <= next_velocity <= REST_STATE_TOL or abs(next_acceleration) > REST_STATE_TOL:
        return None
      if jerk > 0.0:
        extremum = -acceleration / jerk
        if 0.0 < extremum < duration and velocity + acceleration * extremum + 0.5 * jerk * extremum**2 < 0.0:
          return None
      states[stage+1] = (position + velocity * duration + 0.5 * acceleration * duration**2 + jerk * duration**3 / 6.0,
                         next_velocity, next_acceleration)
      jerks[stage] = jerk
    return states, jerks

  def set_reference_problem(self, reference, x_obstacles, sources, accel_min, t_follow):
    """Install genuine costs and state bounds in this same native problem."""
    typed = reference.typed_model if reference is not None else None
    if typed is not None and not isinstance(typed, ModelReferenceEvidence):
      raise ValueError('Invalid typed model reference')
    self.rest_goal = typed.rest_goal if typed is not None else None
    self.rest_goal_state = 'none'
    self.rest_goal_reason = ''
    self.rest_goal_braking_accel = None
    self.diagnostic_rest_goal = None
    self.yref[:, :] = 0.0
    self.params[:, MODEL_V_REF:] = 0.0
    selected = np.argmin(x_obstacles, axis=1)
    weights = np.tile(self._cost_weights, (N+1, 1))
    weights[:, 4] *= self.accel_change_cost_scale(T_IDXS)
    lower = np.full((N+1, 2), -REFERENCE_BOUND)
    upper = np.full((N+1, 2), REFERENCE_BOUND)

    qualified = typed is not None and typed.unknown is None
    if qualified:
      # The measured state remains fixed. Forward prediction nodes cannot use
      # a soft negative-speed excursion to make a typed rest goal economical;
      # the exact interval-extremum check below remains a separate requirement.
      lower[1:, 0] = 0.0
      preference_time = float(reference.preference_time_s)
      if not np.isfinite(preference_time) or preference_time <= 0.0:
        raise ValueError('Invalid preference time scale')
      # q_j keeps its previous value. The dimensions follow a declared
      # engineering time scale: q_v=q_j/T^4 and q_a=q_j/T^2, with q_x=0.
      weights[:, 1] = 0.0
      weights[:, 2] = self._cost_weights[5] / preference_time**4
      weights[:, 3] = self._cost_weights[5] / preference_time**2
      self.params[:, TYPED_MODEL_MODE] = 1.0
      if typed.moving is not None:
        moving = typed.moving
        if (any(values.shape != (N+1,) for values in (moving.position, moving.velocity, moving.acceleration, moving.supported))
            or not np.all(np.isfinite([moving.position, moving.velocity, moving.acceleration]))):
          raise ValueError('Invalid moving preference dimensions')
        self.yref[:, 1] = moving.position
        self.params[:, MODEL_V_REF] = moving.velocity
        self.params[:, MODEL_A_REF] = moving.acceleration
        weights[~moving.supported, 1:4] = 0.0
      elif not typed.no_forward:
        weights[:, 1:4] = 0.0
      # Cruise is a limiting envelope while the model owns the preference.
      # Real lead distance cost and its risk slack remain active independently
      # of model direction, with the same weights as the original problem.
      for stage, obstacle_index in enumerate(selected):
        if sources[obstacle_index] == LongitudinalPlanSource.cruise:
          weights[stage, 0] = 0.0
        elif typed.moving is not None and sources[obstacle_index] in MPC_SOURCES:
          # This compares a stated ego preference with independently admitted
          # lead geometry. It is not a presence mask or a second solved arm.
          preference_obstacle = (typed.moving.position[stage]
                                 + get_safe_obstacle_distance(max(typed.moving.velocity[stage], 0.0), t_follow))
          if x_obstacles[stage, obstacle_index] <= preference_obstacle:
            self.params[stage, MODEL_BRAKING_PRIORITY] = 1.0
      if self.source == LongitudinalPlanSource.cruise:
        self.source = LongitudinalPlanSource.e2e

    if self.rest_goal is not None:
      goal = self.rest_goal
      minimum_accel = float(np.min(np.broadcast_to(accel_min, (N+1,))))
      nominal_distance = float(self.x0[1]**2 / (-2.0 * minimum_accel)) if minimum_accel < 0.0 else float('inf')
      self.diagnostic_rest_goal = {
        'state': 'pending', 'reason': '', 'nativeNode': goal.native_node, 'nativeTime': goal.native_time_s,
        'remainingTime': goal.remaining_time_s, 'eventProxyEpochNs': goal.event_proxy_epoch_ns,
        'nominalPosition': goal.nominal_position, 'positionKind': goal.position_kind,
        'nominalPositionComparedAtSameProxyOrigin': typed.source.proxy_age_s == 0.0,
        'nominalPositionBelowBrakingLowerBound': (goal.nominal_position < nominal_distance
                                                if goal.nominal_position is not None and typed.source.proxy_age_s == 0.0 else None),
        'physicalPositionBound': False,
        'satisfactionScope': 'native_prediction_only', 'physicalStoppedKnown': False,
      }
      if goal.native_node is None or not 0 <= goal.native_node <= N or goal.native_time_s != T_IDXS[goal.native_node]:
        self.rest_goal_state, self.rest_goal_reason = 'unverified', 'rest_goal_outside_native_horizon'
      elif goal.native_node == 0 and (self.x0[1] < 0.0 or np.max(np.abs(self.x0[1:])) > REST_STATE_TOL):
        self.rest_goal_state, self.rest_goal_reason = 'infeasible', 'rest_goal_due_with_nonrest_initial_state'
      elif goal.native_node == 0 and (resting_primal := self.due_rest_primal(self.x0)) is None:
        self.rest_goal_state, self.rest_goal_reason = 'infeasible', 'rest_goal_due_without_nonnegative_rest_trajectory'
      elif self.x0[1] + minimum_accel * goal.native_time_s > REST_STATE_TOL:
        self.rest_goal_state, self.rest_goal_reason = 'infeasible', 'rest_goal_time_below_braking_lower_bound'
      elif goal.native_node == 1 and self.rest_acceleration_interval(self.x0, T_IDXS[1], np.zeros(2), np.zeros(2)) is None:
        # A single constant-jerk segment cannot independently choose v_1 and
        # a_1. The acceleration-only lower bound above misses this infeasibility.
        # Check the actual future-goal zero box without changing its meaning.
        self.rest_goal_state, self.rest_goal_reason = 'infeasible', 'rest_goal_time_below_discrete_lower_bound'
      else:
        self.rest_goal_state = 'constrained'
        lower[max(1, goal.native_node):, :] = 0.0
        upper[max(1, goal.native_node):, :] = 0.0
        # A raw zero goal survives the anchored moving preference unchanged.
        self.params[goal.native_node:, MODEL_V_REF:MODEL_A_REF + 1] = 0.0
        if goal.native_node == 0:
          # Within the existing rest acceptance precision, exact endpoint
          # v=a=0 is generally unreachable from a nonzero fixed initial state:
          # a_1=0 implies v_1=v_0+a_0*dt/2 for this constant-jerk model.
          # Express that SAME accepted precision as feasible state bounds;
          # exact rest retains exact-zero bounds. Future goals are unchanged.
          if np.any(self.x0[1:] != 0.0):
            lower[1:, 1] = -REST_STATE_TOL
            upper[1:, :] = REST_STATE_TOL
          # The independently checked primal has no negative-speed segment.
          # The actual solve still must pass the original residual and
          # continuous-speed checks; the seed is never accepted as a solve.
          resting_states, resting_jerks = resting_primal
          for stage in range(N+1):
            self.solver.set(stage, 'x', resting_states[stage])
            if stage < N:
              self.solver.set(stage, 'u', np.array([resting_jerks[stage]]))
      if self.rest_goal_state in ('infeasible', 'unverified'):
        # Retain the failed goal while solving the existing lead/cruise risk
        # problem. The caller uses an explicit braking failure intent; it must
        # never publish this relaxed solution as goal satisfaction.
        self.rest_goal_braking_accel = minimum_accel

    for stage in range(N+1):
      self.solver.cost_set(stage, 'W', np.asfortranarray(np.diag(weights[stage, :COST_DIM if stage < N else COST_E_DIM])))
      self.solver.set(stage, 'yref', self.yref[stage, :COST_DIM if stage < N else COST_E_DIM])
      if stage:
        self.solver.constraints_set(stage, 'lbx', lower[stage])
        self.solver.constraints_set(stage, 'ubx', upper[stage])
    self.diagnostic_reference_weights = weights.copy()
    self.diagnostic_reference_bounds = (lower.copy(), upper.copy())

  def check_rest_goal(self):
    if self.rest_goal is None:
      return
    if self.rest_goal_state in ('infeasible', 'unverified'):
      self.solution_status = MPC_GOAL_INFEASIBLE
    elif self.rest_goal_state == 'constrained':
      node = self.rest_goal.native_node
      numerical_failure = self.project_due_rest_precision() if node == 0 else ''
      error = float(np.max(np.abs(self.x_sol[node:, 1:])))
      minimum_velocity = float(np.min(self.v_solution))
      for stage, duration in enumerate(np.diff(T_IDXS)):
        if self.j_solution[stage] > 0.0:
          extremum = -self.a_solution[stage] / self.j_solution[stage]
          if 0.0 < extremum < duration:
            minimum_velocity = min(minimum_velocity, float(self.v_solution[stage]
                                   + self.a_solution[stage] * extremum + 0.5 * self.j_solution[stage] * extremum**2))
      self.diagnostic_rest_goal['maximumRestStateError'] = error
      self.diagnostic_rest_goal['minimumContinuousVelocity'] = minimum_velocity
      if numerical_failure or error > REST_STATE_TOL or minimum_velocity < -REST_STATE_TOL:
        self.rest_goal_state = 'unsatisfied'
        self.rest_goal_reason = numerical_failure or ('native_rest_bound_not_satisfied' if error > REST_STATE_TOL
                                                     else 'negative_velocity_between_native_nodes')
        self.rest_goal_braking_accel = float(np.min(self.params[:, 0]))
        self.solution_status = MPC_GOAL_UNSATISFIED
      else:
        self.rest_goal_state = 'satisfied'
    self.failure_reason = self.rest_goal_reason
    self.diagnostic_rest_goal.update(state=self.rest_goal_state, reason=self.rest_goal_reason)
    if self.solution_status != 0:
      self.previous_solution_valid = False
      self.invalidate_reference_history()

  def update(self, radarstate, personality=log.LongitudinalPersonality.standard, *, reference=None):
    self.diagnostic_last_obstacle_summary = None
    self.rest_goal = None
    self.rest_goal_state = 'none'
    self.rest_goal_reason = ''
    self.rest_goal_braking_accel = None
    self.diagnostic_rest_goal = None
    t_follow = get_T_FOLLOW(personality)
    self.solve_time = 0.0

    if not np.all(np.isfinite(self.x0)):
      self.fail(MPC_INVALID_INPUT)
      return
    for lead in (radarstate.leadOne, radarstate.leadTwo):
      if lead.present and not np.all(np.isfinite([lead.dRel, lead.vLead, lead.aLeadK, lead.aLeadTau])):
        self.fail(MPC_INVALID_INPUT)
        return
    with np.errstate(over='ignore', invalid='ignore'):
      lead_xv_0 = self.process_lead(radarstate.leadOne)
      lead_xv_1 = self.process_lead(radarstate.leadTwo)

    # To estimate a safe distance from a moving lead, we calculate how much stopping
    # distance that lead needs as a minimum. We can add that to the current distance
    # and then treat that as a stopped car/obstacle at this new distance.
    lead_0_obstacle = lead_xv_0[:,0] + get_stopped_equivalence_factor(lead_xv_0[:,1])
    lead_1_obstacle = lead_xv_1[:,0] + get_stopped_equivalence_factor(lead_xv_1[:,1])

    obstacles = [lead_0_obstacle, lead_1_obstacle]
    sources = list(MPC_SOURCES)
    accel_min, accel_max = ACCEL_MIN, ACCEL_MAX
    if reference is not None:
      try:
        with np.errstate(over='ignore', invalid='ignore'):
          reference_obstacles, reference_sources, accel_min, accel_max = self.reference_obstacles(reference, t_follow)
      except (TypeError, ValueError, FloatingPointError):
        self.fail(MPC_INVALID_INPUT)
        return
      for lead_index, lead in enumerate((radarstate.leadOne, radarstate.leadTwo)):
        if not lead.present:
          obstacles[lead_index] = np.full(N+1, np.inf)
      obstacles.extend(reference_obstacles)
      sources.extend(reference_sources)
    x_obstacles = np.column_stack(obstacles)
    self.source = sources[np.argmin(x_obstacles[1 if reference is not None else 0])]

    try:
      self.set_reference_problem(reference, x_obstacles, sources, accel_min, t_follow)
    except (TypeError, ValueError, FloatingPointError):
      self.fail(MPC_INVALID_INPUT)
      return

    self.params[:,0] = accel_min
    self.params[:,1] = accel_max
    self.params[:,2] = np.min(x_obstacles, axis=1)
    self.params[:,3] = np.copy(self.a_prev)
    self.params[:,4] = t_follow
    self.params[:,5] = LEAD_DANGER_FACTOR

    try:
      self.observe_obstacles(x_obstacles, sources)
    except Exception:
      self.diagnostic_obstacle_failures += 1

    self.run()
    if self.solution_status != 0:
      return
    self.check_rest_goal()
    if (radarstate.leadOne.present and np.any(lead_xv_0[FCW_IDXS,0] - self.x_sol[FCW_IDXS,0] < CRASH_DISTANCE) and
            radarstate.leadOne.modelProb > 0.9):
      self.crash_cnt += 1
    else:
      self.crash_cnt = 0

  def observe_obstacles(self, obstacles, sources):
    """Copy a bounded constraint summary; never rebuild a reference or solve."""
    if not self.diagnostic_obstacles_enabled:
      return
    if obstacles.shape != (N+1, len(sources)) or not 2 <= len(sources) <= 4:
      raise ValueError("Unexpected obstacle summary shape")
    selected = np.argmin(obstacles, axis=1)
    source_names = {value: name for name, value in LongitudinalPlanSource.schema.enumerants.items()}
    self.diagnostic_last_obstacle_summary = {
      'known': True,
      'times': tuple(float(value) for value in T_IDXS),
      'sources': tuple(source_names[sources[index]] for index in selected),
      'distances': tuple(float(value) for value in self.params[:,2]),
      'sourceRule': 'first_minimum_by_mpc_source_order',
    }

  def fail(self, status):
    failed_goal = self.rest_goal
    goal_diagnostic = self.diagnostic_rest_goal
    solve_time = self.solve_time
    last_cloudlog_t = self.last_cloudlog_t
    timestamp = time.monotonic()
    if timestamp > last_cloudlog_t + 5.0:
      last_cloudlog_t = timestamp
      cloudlog.warning(f"Long mpc reset, solution_status: {status}")
    self.reset()
    self.solution_status = status
    self.last_solver_status = status
    self.failure_reason = {MPC_INVALID_INPUT: 'invalid_input', MPC_INVALID_SOLUTION: 'nonfinite_solution',
                           MPC_SOLVER_ERROR: 'solver_exception'}.get(status, 'solver_status')
    if failed_goal is not None:
      self.rest_goal = failed_goal
      self.rest_goal_state, self.rest_goal_reason = 'solver_failure', self.failure_reason
      self.rest_goal_braking_accel = ACCEL_MIN
      self.diagnostic_rest_goal = goal_diagnostic or {}
      self.diagnostic_rest_goal.update(state=self.rest_goal_state, reason=self.rest_goal_reason)
    self.last_cloudlog_t = last_cloudlog_t
    self.solve_time = solve_time if np.isfinite(solve_time) else 0.0
    self.source = LongitudinalPlanSource.cruise

  def run(self):
    try:
      self.diagnostic_last_x0 = tuple(float(value) for value in self.x0)
      if self.diagnostic_details_enabled:
        self.diagnostic_last_parameters = self.params.tolist()
        self.diagnostic_last_yref = self.yref.tolist()
        self.diagnostic_last_a_prev = self.a_prev.tolist()
    except Exception:
      pass
    if not np.all(np.isfinite(self.params)) or not np.all(np.isfinite(self.yref)) or not np.all(np.isfinite(self.x0)):
      self.fail(MPC_INVALID_INPUT)
      return
    for i in range(N+1):
      self.solver.set(i, 'p', self.params[i])
    self.solver.constraints_set(0, "lbx", self.x0)
    self.solver.constraints_set(0, "ubx", self.x0)

    solve_started_t = time.monotonic()
    try:
      self.diagnostic_last_solver_return = None
      self.diagnostic_solve_calls += 1
      self.solution_status = int(self.solver.solve())
      self.diagnostic_last_solver_return = self.solution_status
      self.last_solver_status = self.solution_status
      if self.solution_status != 0:
        self.solve_time = time.monotonic() - solve_started_t
        self.fail(self.solution_status)
        return
      self.solve_time = float(self.solver.get_stats('time_tot')[0])
      for i in range(N+1):
        self.x_sol[i] = self.solver.get(i, 'x')
      for i in range(N):
        self.u_sol[i] = self.solver.get(i, 'u')
    except RuntimeError:
      self.solve_time = time.monotonic() - solve_started_t
      self.fail(MPC_SOLVER_ERROR)
      return
    if not np.all(np.isfinite(self.x_sol)) or not np.all(np.isfinite(self.u_sol)):
      self.fail(MPC_INVALID_SOLUTION)
      return
    self.failure_reason = ''

    self.v_solution = self.x_sol[:,1]
    self.a_solution = self.x_sol[:,2]
    self.j_solution = self.u_sol[:,0]

    self.a_prev = np.interp(T_IDXS + self.dt, T_IDXS, self.a_solution)
    self.previous_solution_valid = True
    if self._reference_pending_ns is not None:
      self._reference_pending_solved = True


if __name__ == "__main__":
  ocp = gen_long_ocp()
  AcadosOcpSolver.generate(ocp, json_file=JSON_FILE)
