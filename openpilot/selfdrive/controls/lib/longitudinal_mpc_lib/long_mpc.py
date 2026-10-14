#!/usr/bin/env python3
import os
import time
from dataclasses import dataclass
from enum import Enum
from itertools import permutations
from numbers import Integral, Real
import numpy as np
from openpilot.cereal import log
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
# WARNING: imports outside of constants will not trigger a rebuild
from openpilot.selfdrive.modeld.constants import index_function
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU
from openpilot.selfdrive.controls.lib.model_reference import ModelReferenceEvidence, PREFERENCE_TIME_S
from openpilot.selfdrive.controls.lib.lead_qualification import (
  MAX_ESTIMATE_AGE_NS, MAX_MEASUREMENT_AGE_NS, MAX_MODEL_AGE_NS, lead_identity, qualified_lead,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.continuation import (
  ContinuationDonor, ContinuationProblem, ContinuationResult, build_candidate, frozen_array, valid_timestamp,
)
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_MAX_AGE_NS
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.interval_rest import IntervalRest

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
PARAM_DIM = 14
MODEL_V_REF = 6
MODEL_A_REF = 7
MODEL_BRAKING_PRIORITY = 8
TYPED_MODEL_MODE = 9
COMFORT_STOP_DISTANCE = 10
RISK_STOP_DISTANCE = 11
COMFORT_NORMALIZER = 12
RISK_NORMALIZER = 13
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
QP_MAX_ITER = 24

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
# A numerical repair requires observed residuals this small. This is not the
# native QP request tolerance, nor a change to the physical rest acceptance box.
QP_FEASIBILITY_TOL = 1e-9


class MpcOutcome(Enum):
  """Planner acceptance, separate from the native solver's numeric diagnostics.

  A rejected rest goal includes unverified goals and rejected trajectories;
  it does not assert that every possible stopping trajectory is infeasible.
  """
  NOT_SOLVED = 'not_solved'
  ACCEPTED = 'accepted'
  REST_REJECTED = 'rejected_rest'
  INPUT_REJECTED = 'rejected_input'
  SOLVER_FAULT = 'solver_fault'


@dataclass(frozen=True)
class ReferenceLeadSupport:
  """Current formal authority that may have contributed to a mixed reference.

  A visual/legacy slot has no physical identity. Its presence and purposes
  can be compared, but a new object in the same anonymous slot is not thereby
  proven to be the preceding object. No old geometry is retained here.
  """
  identity: tuple[str, int] | None
  source: str
  purposes: frozenset[str]


@dataclass(frozen=True)
class MpcReference:
  # None omits only the cruise-speed goal. Bounds and any compatible model
  # goal remain checked; typed model references still require a cruise goal.
  cruise_speed: float | np.ndarray | None
  accel_min: float | np.ndarray = ACCEL_MIN
  accel_max: float | np.ndarray = ACCEL_MAX
  cruise_min_accel: float | np.ndarray = -1.2
  model_position: np.ndarray | None = None
  model_velocity: np.ndarray | None = None
  model_acceleration: np.ndarray | None = None
  typed_model: ModelReferenceEvidence | None = None
  preference_time_s: float = PREFERENCE_TIME_S
  # The UNIFIED producer may schedule a witnessed later stop. Direct mathematical
  # callers keep the original source-goal contract unless they explicitly opt in.
  allow_future_rest_reschedule: bool = False
  # Only the UNIFIED producer opts into a braking/rest switch inside interval
  # zero. Its consumer samples that same piecewise trajectory explicitly.
  allow_interval_rest: bool = False
  # These explicit targets apply only to real leads. Virtual cruise/model
  # obstacles retain STOP_DISTANCE, as do callers that omit these fields.
  # Qualification for a compact stationary gap belongs to the producer.
  comfort_stop_distance: float = STOP_DISTANCE
  lead_risk_stop_distances: tuple[float, float] = (STOP_DISTANCE, STOP_DISTANCE)

  def __post_init__(self):
    if (not isinstance(self.comfort_stop_distance, Real) or isinstance(self.comfort_stop_distance, (bool, np.bool_))
        or not 3.0 <= self.comfort_stop_distance <= STOP_DISTANCE or not np.isfinite(float(self.comfort_stop_distance))):
      raise ValueError('Invalid comfort stop distance')
    object.__setattr__(self, 'comfort_stop_distance', float(self.comfort_stop_distance))
    if (not isinstance(self.lead_risk_stop_distances, tuple) or len(self.lead_risk_stop_distances) != len(MPC_SOURCES)
        or any(not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
               or not 4.0 <= value <= STOP_DISTANCE or not np.isfinite(float(value)) for value in self.lead_risk_stop_distances)):
      raise ValueError('Invalid lead risk stop distances')
    object.__setattr__(self, 'lead_risk_stop_distances', tuple(float(value) for value in self.lead_risk_stop_distances))
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


def get_T_FOLLOW(personality=log.LongitudinalPersonality.standard, *, v_ego=None):
  """Select headway; a measured speed opts standard into the UNIFIED policy.

  Static/code-generation and legacy callers retain the nominal personality
  value. Runtime consumers of the speed policy share one value for the whole
  horizon, including comfort, risk and virtual-obstacle geometry.
  """
  if v_ego is not None and (not isinstance(v_ego, Real) or isinstance(v_ego, (bool, np.bool_)) or not np.isfinite(v_ego)):
    raise ValueError("Invalid following-policy speed")
  if personality==log.LongitudinalPersonality.relaxed:
    return 1.75
  elif personality==log.LongitudinalPersonality.standard:
    if v_ego is None:
      return 1.45
    # Full shortening through 80 km/h, with zero slope at both ends of the
    # 80--100 km/h transition. Keep the standard acceleration/jerk weights.
    blend = float(np.clip((v_ego - 80.0 / 3.6) / (20.0 / 3.6), 0.0, 1.0))
    return 1.25 + 0.20 * blend * blend * (3.0 - 2.0 * blend)
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
  comfort_stop_distance = SX.sym('comfort_stop_distance')
  risk_stop_distance = SX.sym('risk_stop_distance')
  comfort_normalizer = SX.sym('comfort_normalizer')
  risk_normalizer = SX.sym('risk_normalizer')
  model.p = vertcat(a_min, a_max, x_obstacle, a_prev, lead_t_follow, lead_danger_factor,
                   model_v_ref, model_a_ref, model_braking_priority, typed_model_mode, comfort_stop_distance, risk_stop_distance,
                   comfort_normalizer, risk_normalizer)

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
  comfort_stop_distance = ocp.model.p[COMFORT_STOP_DISTANCE]
  risk_stop_distance = ocp.model.p[RISK_STOP_DISTANCE]
  # Retain the generated fourteen-parameter ABI. Runtime always writes zero
  # to these reserved entries, selecting the original velocity-dependent scale.
  comfort_normalizer = if_else(ocp.model.p[COMFORT_NORMALIZER] > 0.0, ocp.model.p[COMFORT_NORMALIZER], v_ego + 10.0)
  risk_normalizer = if_else(ocp.model.p[RISK_NORMALIZER] > 0.0, ocp.model.p[RISK_NORMALIZER], v_ego + 10.0)

  ocp.cost.yref = np.zeros((COST_DIM, ))
  ocp.cost.yref_e = np.zeros((COST_E_DIM, ))

  desired_dist_safe = get_safe_obstacle_distance(v_ego, lead_t_follow)
  desired_dist_comfort = desired_dist_safe - STOP_DISTANCE + comfort_stop_distance
  desired_dist_risk = desired_dist_safe - STOP_DISTANCE + risk_stop_distance

  # The main cost in normal operation is how close you are to the "desired" distance
  # from an obstacle at every timestep. This obstacle can be a lead car
  # or other object. In e2e mode we can use x_position targets as a cost
  # instead.
  distance_error = ((x_obstacle - x_ego) - desired_dist_comfort) / comfort_normalizer
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
                        ((x_obstacle - x_ego) - lead_danger_factor * desired_dist_risk) / risk_normalizer)
  ocp.model.con_h_expr = constraints

  x0 = np.zeros(X_DIM)
  ocp.constraints.x0 = x0
  # Every mode carries the same native layout. Ordinary stages receive wide
  # bounds; an admitted typed rest goal applies its rest box from the stated node.
  # Initial (x,v,a) stays tied to the measured carState, including residual v.
  ocp.constraints.idxbx = np.array([1, 2])
  ocp.constraints.lbx = np.full(2, -REFERENCE_BOUND)
  ocp.constraints.ubx = np.full(2, REFERENCE_BOUND)
  ocp.constraints.idxbx_e = np.array([1, 2])
  ocp.constraints.lbx_e = np.full(2, -REFERENCE_BOUND)
  ocp.constraints.ubx_e = np.full(2, REFERENCE_BOUND)
  # Two linear Bernstein coefficients can certify each continuous-velocity
  # interval. They are inactive unless a future typed rest goal is constrained.
  ocp.constraints.C = np.zeros((2, X_DIM))
  ocp.constraints.D = np.zeros((2, U_DIM))
  ocp.constraints.lg = -np.ones(2)
  ocp.constraints.ug = np.ones(2)
  ocp.parameter_values = np.array([-1.2, 1.2, 0.0, 0.0, get_T_FOLLOW(), LEAD_DANGER_FACTOR,
                                   0.0, 0.0, 0.0, 0.0, STOP_DISTANCE, STOP_DISTANCE, 0.0, 0.0])


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

  # Bound the QP work inside one SQP_RTI call. Cold rest boundary cases
  # need 21 inner iterations. Keep a fixed budget and independent checks;
  # this does not add another native solve or change convergence tolerances.
  ocp.solver_options.qp_solver_iter_max = QP_MAX_ITER
  ocp.solver_options.qp_tol = 1e-3

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = T_IDXS

  ocp.solver_options.custom_update_filename = 'interval_rest_update.c'
  ocp.solver_options.custom_update_header_filename = 'interval_rest_update.h'

  ocp.code_export_directory = EXPORT_DIR
  return ocp


class LongitudinalMpc:
  def __init__(self, dt=DT_MDL, *, speed_based_following=False):
    self.dt = dt
    self.speed_based_following = speed_based_following
    self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
    self._native_first_interval = float(T_IDXS[1])
    self.diagnostic_solve_calls = 0
    self.diagnostic_reset_calls = 0
    self.diagnostic_last_solver_return = None
    self.diagnostic_last_x0 = None
    self.diagnostic_details_enabled = False
    self.diagnostic_last_parameters = None
    self.diagnostic_last_yref = None
    self.diagnostic_last_a_prev = None
    self.diagnostic_reference_history = None
    # These describe the current attempt and survive a failed solve's reset,
    # so the producer cannot offer a donor with the just-revoked support.
    self.reference_support_changed = False
    self.diagnostic_obstacles_enabled = False
    self.diagnostic_last_obstacle_summary = None
    self.diagnostic_obstacle_failures = 0
    # Failure logging survives solver resets and is independent of opt-in traces.
    self._failure_log_key = None
    self._failure_log_time = 0.0
    self._failure_log_suppressed = 0
    self._failure_log_credit_ns = 15_000_000_000  # Three records, then one credit per five seconds.
    self._failure_log_refill_ns = None
    self.diagnostic_failure_log_errors = 0
    self._diagnostic_input_times = {}
    self._diagnostic_radar = None
    self._diagnostic_rejected_lead = None
    self._diagnostic_now_ns = None
    self._diagnostic_reference_ns = None
    self._diagnostic_native_return = None
    self._diagnostic_problem_installed = False
    self._failure_diagnosed = False
    # UNIFIED dates its reference with the consumer clock. Keep the clock high
    # water across solver resets; reset buffers must not admit a replayed epoch.
    self._reference_clock_ns = 0
    self.reset()
    self.source = LongitudinalPlanSource.cruise

  def reset(self):
    clear_reference_bounds = hasattr(self, '_rest_constraint_buffers')
    self.diagnostic_reset_calls += 1
    self.solver.reset()

    # The native setters copy their inputs synchronously. Keep private,
    # writable column-major buffers for each stage, including the terminal
    # cost's smaller shape, so installing a problem needs no matrix allocation.
    self._cost_matrices = [np.zeros((dimension, dimension), order='F')
                           for dimension in [COST_DIM] * N + [COST_E_DIM]]
    self._cost_diagonals = [matrix.ravel(order='F')[::matrix.shape[0]+1] for matrix in self._cost_matrices]
    self._rest_constraint_buffers = [(np.zeros((2, X_DIM), order='F'), np.zeros((2, U_DIM), order='F'),
                                      np.full(2, -1.0), np.ones(2)) for _ in range(N)]
    if clear_reference_bounds:
      # solver.reset() clears iterates, not the installed model constraints.
      # Ordinary update rewrites every row; direct run after failure/reset
      # must also be unable to inherit an old rest row or state box.
      for stage, (C, D, lower, upper) in enumerate(self._rest_constraint_buffers):
        self.solver.constraints_set(stage, 'C', C)
        self.solver.constraints_set(stage, 'D', D)
        self.solver.constraints_set(stage, 'lg', lower)
        self.solver.constraints_set(stage, 'ug', upper)
      for stage in range(1, N+1):
        self.solver.constraints_set(stage, 'lbx', np.full(2, -REFERENCE_BOUND))
        self.solver.constraints_set(stage, 'ubx', np.full(2, REFERENCE_BOUND))

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
    self.params[:, COMFORT_STOP_DISTANCE] = STOP_DISTANCE
    self.params[:, RISK_STOP_DISTANCE] = STOP_DISTANCE
    for i in range(N+1):
      self.solver.set(i, 'x', np.zeros(X_DIM))

    self.last_cloudlog_t = 0
    self.reset_fcw()
    self._was_fusion_active = False
    self.source_object_identity = None
    self.fcw_object_identity = None
    self._obstacle_identities = ()
    self._source_leads = ()
    self._lead_trajectories = ()
    self.object_influence_identities = ()
    self._influence_problem = None
    self.solution_status = 0
    self.last_solver_status = 0
    self.failure_reason = ''
    self.rest_goal = None
    self.interval_rest = None
    self.rest_goal_state = 'none'
    self.rest_goal_reason = ''
    self.rest_goal_braking_accel = None
    self.effective_rest_node = None
    self.effective_rest_state = 'none'
    self.effective_rest_reason = ''
    self.rest_recovery_position_limit = None
    self.diagnostic_rest_goal = None
    self.diagnostic_reference_weights = None
    self.diagnostic_reference_bounds = None
    self.continuation_problem = None
    self.continuation_donor_reason = 'not_captured'
    # timers
    self.solve_time = 0.0
    self.x0 = np.zeros(X_DIM)
    self.set_weights()

  @property
  def outcome(self):
    # Derive this from the current contract instead of storing a second status
    # that could survive a reset or a producer's later input rejection.
    if self.solution_status == 0:
      return MpcOutcome.ACCEPTED if self.previous_solution_valid else MpcOutcome.NOT_SOLVED
    if self.solution_status == MPC_INVALID_INPUT:
      return MpcOutcome.INPUT_REJECTED
    if self.rest_goal is not None and (
      self.solution_status == MPC_GOAL_INFEASIBLE and self.rest_goal_state in ('infeasible', 'unverified')
      or self.solution_status == MPC_GOAL_UNSATISFIED
      and (self.rest_goal_state == 'unsatisfied' or self.effective_rest_state == 'unsatisfied')
    ):
      return MpcOutcome.REST_REJECTED
    return MpcOutcome.SOLVER_FAULT

  @staticmethod
  def accel_change_cost_scale(times):
    return np.interp(times, [0.0, 1.0, 2.0], [1.0, 1.0, 0.0])

  def invalidate_reference_history(self):
    """Drop UNIFIED reference provenance without moving the consumer clock."""
    self._reference_epoch_ns = None
    self._reference_acceleration = None
    self._reference_interval_rest = None
    self._reference_support = None
    self._reference_prepared_support = None
    self._reference_pending_support = None
    self._reference_discard_reason = ''
    self._reference_pending_ns = None
    self._reference_pending_solved = False

  def prepare_reference_history(self, now_ns, *, valid):
    """Align one preceding native reference with this consumer's real epoch.

    This is a smoothing reference, never an allocated or pending actuator
    request. LEGACY continues to use run()'s original fixed-dt history.
    A new epoch is committed only after the caller accepts a successful output.
    """
    epoch_ns = self._reference_epoch_ns
    acceleration = self._reference_acceleration
    interval_rest = self._reference_interval_rest
    support = self._reference_support
    self.invalidate_reference_history()
    self.diagnostic_reference_history = None
    self.reference_support_changed = False
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
    self.a_prev = (interval_rest.sample(T_IDXS + elapsed)[0][:, 2] if interval_rest is not None else
                   np.interp(T_IDXS + elapsed, T_IDXS, acceleration))
    self._reference_prepared_support = support
    return True

  def mark_reference_history(self, *, valid):
    """Commit the native reference only after a qualified UNIFIED output."""
    epoch_ns = self._reference_pending_ns
    solved = self._reference_pending_solved
    support = self._reference_pending_support
    self.invalidate_reference_history()
    if (not valid or epoch_ns is None or support is None or not solved or not self.previous_solution_valid or self.solution_status != 0
        or self.a_solution.shape != T_IDXS.shape or not np.all(np.isfinite(self.a_solution))):
      return False
    self._reference_epoch_ns = epoch_ns
    # The next native solve overwrites x_sol, which backs a_solution. Preserve
    # this accepted reference rather than dating a later/failed solve as it.
    self._reference_acceleration = self.a_solution.copy()
    self._reference_interval_rest = self.interval_rest
    self._reference_support = support
    return True

  @staticmethod
  def _reference_support_summary(support):
    return tuple({'identity': item.identity, 'source': item.source, 'purposes': tuple(sorted(item.purposes))}
                 for item in support or ())

  def qualify_reference_support(self, radarstate, *, now_ns, valid):
    """Rebuild comfort history when its current formal authority is gone.

    Call only after validating this update's formal leads. Losing a formal
    slot may be ranking rather than physical clearance; neither permits an
    unselected object to keep control influence through this comfort cost.
    The new optimization retains all current leads, measured x0, jerk costs
    and other constraints. This changes no solver initialization or PID.
    """
    previous = self._reference_prepared_support
    diagnostic = {
      'referenceMonoTime': self._reference_pending_ns,
      'previousSupport': self._reference_support_summary(previous),
      'currentSupport': (), 'reason': 'no_prepared_history', 'supportChanged': False,
      'anonymousIdentityKnown': False,
    }
    self.diagnostic_reference_history = diagnostic
    if self._reference_pending_ns is None:
      if not valid:
        diagnostic['reason'] = 'input_health_unqualified'
      return
    # Invalid or degraded input is not an observation of an empty road. Drop
    # history eligibility independently of the existing producer fault path.
    if (not valid or getattr(radarstate, 'fusionDegraded', False)
        or getattr(radarstate, 'fusionActive', False) and not getattr(radarstate, 'fusionHealthy', False)):
      self.discard_prepared_reference_cost(reason='input_health_unqualified')
      self.invalidate_reference_history()
      diagnostic['reason'] = 'input_health_unqualified'
      return
    current = []
    for lead in self._source_leads:
      evidence = getattr(lead, 'evidence', None)
      source = str(evidence.source) if evidence is not None and evidence.version == 1 else 'legacy'
      support = ReferenceLeadSupport(lead_identity(lead), source,
                                     frozenset(purpose for purpose in ('brake', 'follow', 'stop')
                                               if qualified_lead(lead, purpose, 0.0, now_ns=now_ns)))
      # Equal identified views were already validated above. Do not count a
      # physical object twice just because it occupies both formal slots.
      if support.identity is None or support not in current:
        current.append(support)
    current = tuple(sorted(current, key=lambda item: repr((item.identity, item.source, sorted(item.purposes)))))
    self._reference_pending_support = current
    diagnostic['currentSupport'] = self._reference_support_summary(current)
    diagnostic['reason'] = self._reference_discard_reason or 'new_reference'
    if previous is None or self._cost_weights[4] <= 0.0:
      return
    identified = {item.identity: item for item in current if item.identity is not None}
    lost = tuple(item for item in previous if item.identity is not None and item.identity not in identified)
    reduced = tuple(item for item in previous if item.identity in identified
                    and not item.purposes <= identified[item.identity].purposes)
    anonymous = tuple(item for item in current if item.identity is None)
    previous_anonymous = tuple(item for item in previous if item.identity is None)
    # With at most two formal slots this is bounded to two assignments. A
    # visual index is deliberately absent: it cannot prove physical identity.
    anonymous_supported = any(
      all(old.source == new.source and old.purposes <= new.purposes
          for old, new in zip(previous_anonymous, assignment, strict=True))
      for assignment in permutations(anonymous, len(previous_anonymous))
    )
    reason = ('formal_support_removed' if lost else 'formal_qualification_reduced' if reduced else
              'anonymous_support_removed' if not anonymous_supported else '')
    if reason:
      self.reference_support_changed = True
      self.discard_prepared_reference_cost(reason=reason)
      diagnostic.update(reason=reason, supportChanged=True,
                        removedSupport=self._reference_support_summary(lost),
                        reducedSupport=self._reference_support_summary(reduced))
    else:
      diagnostic['reason'] = 'inherited'
    # Replacing the provenance is safe only after the subset check above or
    # a solve with zero history cost. Otherwise an ordinary new epoch could
    # launder a removed ancestor's mixed acceleration trajectory as current.

  def _capture_continuation_problem(self, *, reference_ns, now_ns, fusion_active):
    """Keep this tick's installed problem before fail/reset can erase it."""
    if (not all(valid_timestamp(stamp) for stamp in (reference_ns, now_ns))
        or not 0 <= now_ns - reference_ns <= REFERENCE_MAX_AGE_NS
        or self.diagnostic_reference_bounds is None):
      return None
    expiry = int(reference_ns) + REFERENCE_MAX_AGE_NS
    # The former projection certificates existed only with fusionActive.
    # Keep the non-fusion continuation contract unchanged while retaining
    # every formal fusion lead's measurement deadline and physical geometry.
    fusion_leads = self._source_leads if fusion_active else ()
    for lead in fusion_leads:
      evidence = getattr(lead, 'evidence', None)
      if evidence is None or evidence.version == 0:
        continue
      age = MAX_MEASUREMENT_AGE_NS
      other_expiry = min(int(evidence.estimateMonoTime) + MAX_ESTIMATE_AGE_NS,
                         int(evidence.modelMonoTime) + MAX_MODEL_AGE_NS)
      measurement_expiry = min(int(evidence.positionMonoTime), int(evidence.velocityMonoTime)) + age
      expiry = min(expiry, other_expiry, measurement_expiry)
    # Geometry starts at the observed distance, before process_lead's numerical
    # distance floor. No message reader or writable array survives this capture.
    lead_trajectories = tuple(frozen_array(trajectory) for trajectory in self._lead_trajectories) if fusion_active else ()
    reference = self._influence_problem[3] if self._influence_problem is not None else None
    typed = reference.typed_model if reference is not None else None
    forbidden = (self.rest_goal is not None or self.effective_rest_node is not None
                 or typed is not None and (typed.no_forward or typed.has_rest_intent))
    lower, upper = self.diagnostic_reference_bounds
    return ContinuationProblem(
      int(reference_ns), int(now_ns), int(expiry), frozen_array(self.x0), frozen_array(self.params),
      frozen_array(lower), frozen_array(upper),
      tuple(tuple(frozen_array(value) for value in row) for row in self._rest_constraint_buffers),
      lead_trajectories, bool(forbidden),
    )

  @classmethod
  def _continuation_trajectory_error(cls, problem, states, jerks):
    """Independent candidate checks, never a substituted native-success bit.

    Continuation has stricter admission than ordinary optimized soft distance:
    every current risk row must be nonnegative. This may refuse a perfectly
    useful native mitigation trajectory; a cached motion request is not
    authority to spend a current distance deficit or an unverified stop.
    """
    tolerance = QP_FEASIBILITY_TOL
    if problem.stop_required:
      return 'stop_required'
    parameters = problem.parameters
    arrays = (problem.x0, states, jerks, parameters, problem.state_lower, problem.state_upper)
    if (problem.x0.shape != (X_DIM,) or states.shape != (N+1, X_DIM) or jerks.shape != (N,)
        or parameters.shape != (N+1, PARAM_DIM) or problem.state_lower.shape != (N+1, 2)
        or problem.state_upper.shape != (N+1, 2) or len(problem.linear_constraints) != N
        or not all(np.all(np.isfinite(value)) for value in arrays)):
      return 'invalid_trajectory_or_problem'
    if (np.any(parameters[:, 0] > parameters[:, 1]) or np.any(parameters[:, 5] <= 0.0) or np.any(parameters[:, 4] < 0.0)
        or np.any(parameters[:, [COMFORT_NORMALIZER, RISK_NORMALIZER]] != 0.0)
        or np.any(parameters[:, RISK_STOP_DISTANCE] < 4.0) or np.any(parameters[:, RISK_STOP_DISTANCE] > STOP_DISTANCE)):
      return 'invalid_physical_or_risk_parameters'
    if (np.max(np.abs(states[0] - problem.x0)) > tolerance
        or np.max(np.abs(cls.trajectory_dynamics_defects(states, jerks))) > tolerance):
      return 'dynamics_residual'
    if (np.any(states[1:, 1:] < problem.state_lower[1:] - tolerance)
        or np.any(states[1:, 1:] > problem.state_upper[1:] + tolerance)
        or np.any(states[1:, 2] < parameters[1:, 0] - tolerance)
        or np.any(states[1:, 2] > parameters[1:, 1] + tolerance)):
      return 'physical_state_bound'
    for stage, duration in enumerate(np.diff(T_IDXS)):
      if cls.segment_minimum_velocity(states[stage], jerks[stage], duration) < -tolerance:
        return 'continuous_negative_velocity'
      C, D, lower, upper = problem.linear_constraints[stage]
      if (C.shape != (2, X_DIM) or D.shape != (2, U_DIM) or lower.shape != (2,) or upper.shape != (2,)
          or not all(np.all(np.isfinite(value)) for value in (C, D, lower, upper))):
        return 'invalid_installed_row'
      # This first continuation domain has only behind-position and forward
      # speed rows. Nonnegative state coefficients make the all-up/all-down
      # corners bounds for a consumer's history-based progress-error tube.
      if np.any(C < 0.0):
        return 'unsupported_signed_linear_row'
      value = C @ states[stage] + D[:, 0] * jerks[stage]
      if np.any(value < lower - tolerance) or np.any(value > upper + tolerance):
        return 'installed_linear_bound'
    # The current folded minimum includes the two formal leads and current
    # cruise/model references, with this tick's qualified compact distances.
    risk_margin = parameters[:, 2] - states[:, 0] - parameters[:, 5] * (
      get_safe_obstacle_distance(states[:, 1], parameters[:, 4]) - STOP_DISTANCE + parameters[:, RISK_STOP_DISTANCE])
    denominator = states[:, 1] + 10.0
    if np.any(denominator <= 0.0) or np.any(risk_margin / denominator < -tolerance):
      return 'current_risk_distance_deficit'
    for trajectory in problem.lead_trajectories:
      if trajectory.shape != (N+1, 2) or not np.all(np.isfinite(trajectory)):
        return 'invalid_lead_trajectory'
      if cls.lead_clearance_minimum(trajectory, states, jerks) < CRASH_DISTANCE:
        return 'current_lead_clearance'
    return ''

  def snapshot_continuation_donor(self, source_ns, producer_ns, instance_ns, *, accel_ceiling=None):
    """Seal a reference only after native checks and final producer selection.

    With an adopted ceiling, native acceleration supplies the reference shape;
    it does not supply an unchecked full trajectory. Keep the measured x0,
    intersect all future acceleration bounds, then integrate and independently
    revalidate the complete result against the original installed problem.
    """
    problem = self.continuation_problem
    if (self.solution_status != 0 or self.diagnostic_last_solver_return != 0 or not self.previous_solution_valid
        or not isinstance(problem, ContinuationProblem)):
      self.continuation_donor_reason = 'no_accepted_native_problem'
      return None
    if (not all(valid_timestamp(stamp) for stamp in (source_ns, producer_ns, instance_ns))
        or source_ns != problem.reference_ns or producer_ns != problem.consumed_ns
        or not source_ns <= producer_ns or not instance_ns <= producer_ns):
      self.continuation_donor_reason = 'donor_clock_identity'
      return None
    states, jerks = self.x_sol, self.j_solution
    diagnostics = {}
    if accel_ceiling is not None:
      if (not isinstance(accel_ceiling, Real) or isinstance(accel_ceiling, bool) or not np.isfinite(accel_ceiling)
          or states.shape != (N+1, X_DIM) or jerks.shape != (N,)
          or not np.all(np.isfinite(states)) or not np.all(np.isfinite(jerks))):
        self.continuation_donor_reason = 'invalid_adopted_acceleration_ceiling_or_native_shape'
        return None
      accel_ceiling = float(accel_ceiling)
      lower = np.maximum(problem.state_lower[1:, 1], problem.parameters[1:, 0])
      upper = np.minimum(np.minimum(problem.state_upper[1:, 1], problem.parameters[1:, 1]), accel_ceiling)
      if not np.all(np.isfinite([lower, upper])) or np.any(lower > upper):
        self.continuation_donor_reason = 'adopted_acceleration_intersection_empty'
        return None
      accelerations = np.empty(N+1)
      accelerations[0] = problem.x0[2]  # Measured a0 is never clipped to a request.
      accelerations[1:] = np.clip(states[1:, 2], lower, upper)
      conditioned_jerks = np.diff(accelerations) / np.diff(T_IDXS)
      conditioned_states = np.empty_like(states)
      conditioned_states[0] = problem.x0
      for stage, duration in enumerate(np.diff(T_IDXS)):
        conditioned_states[stage+1] = self.integrate_jerk(conditioned_states[stage], conditioned_jerks[stage], duration)
      diagnostics = dict(
        applied_accel_ceiling=accel_ceiling,
        conditioning_max_state_delta=tuple(float(value) for value in np.max(np.abs(conditioned_states - states), axis=0)),
        conditioning_max_jerk_delta=float(np.max(np.abs(conditioned_jerks - jerks))),
        conditioning_max_braking_relaxation=float(np.max(np.where(states[:, 2] < 0.0,
                                                                  conditioned_states[:, 2] - states[:, 2], 0.0), initial=0.0)),
        native_max_dynamics_defect=tuple(float(value) for value in np.max(np.abs(self.trajectory_dynamics_defects(states, jerks)), axis=0)),
      )
      states, jerks = conditioned_states, conditioned_jerks
    if (reason := self._continuation_trajectory_error(problem, states, jerks)):
      self.continuation_donor_reason = reason
      return None
    self.continuation_donor_reason = ''
    return ContinuationDonor(int(source_ns), int(producer_ns), int(instance_ns),
                              frozen_array(states), frozen_array(jerks), problem, **diagnostics)

  def continuation_candidate(self, donor, *, now_ns):
    """Read-only numerical proposal; native status and sentinel stay intact."""
    if self.solution_status != 4:
      return ContinuationResult(False, 'unsupported_failure_status')
    return build_candidate(donor, self.continuation_problem, now_ns=now_ns, times=T_IDXS,
                            integrate=self.integrate_jerk, validate=self._continuation_trajectory_error)

  def discard_prepared_reference_cost(self, *, reason='reference_semantics_changed'):
    """Reject the preceding semantics, retaining this qualified pending epoch.

    prepare_reference_history already removed the preceding stored reference.
    A new rest goal can replace its cost without losing the new successful
    trajectory that the next frame will be allowed to smooth against.
    """
    self.a_prev[:] = 0.0
    self._cost_weights[4] = 0.0
    self._costs_pending = True
    self._reference_prepared_support = None
    self._reference_discard_reason = reason
    if self.diagnostic_reference_history is not None:
      self.diagnostic_reference_history['reason'] = reason

  def reject_reference_cost_for_current_collision(self):
    """Do not penalize leaving an old accelerating path that hits a current lead.

    This checks only this update's already-qualified raw lead trajectories,
    against the old acceleration reference integrated from the current ego
    state. Its horizon ends at the last node with a nonzero history cost;
    neither an old object nor a new retention/response timer is introduced.
    Rejecting this comfort reference changes no physical bound or jerk cost.
    The native solver still constructs and checks a new current-input plan.
    """
    if not self._lead_trajectories or self._cost_weights[4] <= 0.0:
      return False
    count = int(np.count_nonzero(self.accel_change_cost_scale(T_IDXS) > 0.0))
    accelerations = self.a_prev[:count].copy()
    if (count < 2 or not np.all(np.isfinite(accelerations))
        or not np.any(accelerations[1:] > 0.0)):
      return False
    accelerations[0] = self.x0[2]
    times = T_IDXS[:count]
    jerks = np.diff(accelerations) / np.diff(times)
    states = np.empty((count, X_DIM))
    states[0] = self.x0
    for stage, duration in enumerate(np.diff(times)):
      states[stage+1] = self.integrate_jerk(states[stage], jerks[stage], float(duration))
    if not any(self.lead_clearance_minimum(trajectory[:count], states, jerks, times=times) < CRASH_DISTANCE
               for trajectory in self._lead_trajectories):
      return False
    self.discard_prepared_reference_cost(reason='current_lead_collision')
    return True

  def set_cost_weights(self, cost_weights, constraint_cost_weights):
    """Prepare base costs; update installs only the final reference problem."""
    self._cost_weights = np.asarray(cost_weights, dtype=float).copy()
    self._constraint_cost_weights = np.asarray(constraint_cost_weights, dtype=float).copy()
    self._costs_pending = True

  def stage_cost_weights(self):
    weights = np.tile(self._cost_weights, (N+1, 1))
    weights[:, 4] *= self.accel_change_cost_scale(T_IDXS)
    return weights

  def install_cost_weights(self, weights):
    for stage, (matrix, diagonal) in enumerate(zip(self._cost_matrices, self._cost_diagonals, strict=True)):
      matrix.fill(0.0)
      diagonal[:] = weights[stage, :len(diagonal)]
      self.solver.cost_set(stage, 'W', matrix)
      if stage < N:
        self.solver.cost_set(stage, 'Zl', self._constraint_cost_weights)
    self._costs_pending = False

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
  def extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau, *, elapsed=0.0):
    a_lead_traj = a_lead * np.exp(-a_lead_tau * ((T_IDXS + elapsed)**2)/2.)
    v_lead_traj = np.clip(v_lead + np.cumsum(T_DIFFS * a_lead_traj), 0.0, 1e8)
    x_lead_traj = x_lead + np.cumsum(T_DIFFS * v_lead_traj)
    lead_xv = np.column_stack((x_lead_traj, v_lead_traj))
    return lead_xv

  def lead_initial_state(self, lead, *, reference_ns=None):
    """Align one formal lead's measured state with the current ego epoch."""
    if lead is not None and lead.present:
      x_lead, v_lead, a_lead, a_lead_tau = float(lead.dRel), float(lead.vLead), float(lead.aLeadK), float(lead.aLeadTau)
      evidence = getattr(lead, 'evidence', None)
      elapsed = 0.0
      if evidence is not None and evidence.version == 1 and reference_ns is not None:
        if not valid_timestamp(reference_ns) or not valid_timestamp(evidence.estimateMonoTime):
          raise ValueError('Invalid lead reference clock')
        elapsed = (reference_ns - evidence.estimateMonoTime) / 1e9
        if not -REFERENCE_MAX_AGE_NS / 1e9 <= elapsed <= MAX_ESTIMATE_AGE_NS / 1e9:
          raise ValueError('Lead estimate cannot align with the initial state')
        # A recent model estimate can follow carState. Use the signed interval
        # without redating the measurement or replacing the ego reference epoch.
        decayed_accel = a_lead * np.exp(-a_lead_tau * elapsed**2 / 2.0)
        lead_advance = v_lead * elapsed + (2.0 * a_lead + decayed_accel) * elapsed**2 / 6.0
        ego_advance = self.x0[1] * elapsed - 0.5 * self.x0[2] * elapsed**2
        x_lead += lead_advance - ego_advance
        v_lead = max(0.0, v_lead + 0.5 * (a_lead + decayed_accel) * elapsed)
      return x_lead, v_lead, a_lead, a_lead_tau, elapsed
    else:
      # Fake a fast lead car, so mpc can keep running in the same mode
      return 50.0, self.x0[1] + 10.0, 0.0, _LEAD_ACCEL_TAU, 0.0

  def process_lead(self, lead, *, reference_ns=None):
    v_ego = self.x0[1]
    x_lead, v_lead, a_lead, a_lead_tau, elapsed = self.lead_initial_state(lead, reference_ns=reference_ns)
    # MPC will not converge if immediate crash is expected
    # Clip lead distance to what is still possible to brake for
    min_x_lead = MIN_X_LEAD_FACTOR * (v_ego + v_lead) * (v_ego - v_lead) / (-ACCEL_MIN * 2)
    x_lead = np.clip(x_lead, min_x_lead, 1e8)
    v_lead = np.clip(v_lead, 0.0, 1e8)
    a_lead = np.clip(a_lead, -10., 5.)
    lead_xv = self.extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau, elapsed=elapsed)
    return lead_xv

  @staticmethod
  def lead_clearance_minimum(trajectory, states, jerks, *, times=T_IDXS):
    """Continuous clearance of an ordinary lead for continuation admission.

    The lead uses the same forward-only motion samples as process_lead, with
    its observed initial distance. Linear lead interpolation and constant ego
    jerk have a cubic relative position; its extrema include all real roots
    of relative velocity inside each native interval.
    """
    minimum = float('inf')
    for stage, duration in enumerate(np.diff(times)):
      lead_position = trajectory[stage, 0]
      lead_velocity = (trajectory[stage+1, 0] - lead_position) / duration
      position, velocity, acceleration = states[stage]
      jerk = jerks[stage]
      times = [0.0, duration]
      if abs(jerk) < 1e-12:
        if abs(acceleration) > 1e-12:
          times.append((lead_velocity - velocity) / acceleration)
      else:
        discriminant = acceleration**2 + 2.0 * jerk * (lead_velocity - velocity)
        if discriminant >= 0.0:
          root = np.sqrt(discriminant)
          times.extend(((-acceleration - root) / jerk, (-acceleration + root) / jerk))
      for timestamp in times:
        if 0.0 <= timestamp <= duration:
          next_acceleration = acceleration + jerk * timestamp
          ego_position = position + velocity * timestamp + (2.0 * acceleration + next_acceleration) * timestamp**2 / 6.0
          minimum = min(minimum, float(lead_position + lead_velocity * timestamp - ego_position))
    return minimum

  def reset_fcw(self):
    self.crash_cnt = 0
    self.fcw_object_identity = None
    self._radar_fcw_identity = None
    self._radar_fcw_epoch = None

  def update_lead_fcw(self, lead, trajectory, *, now_ns):
    """Only formal leadOne can confirm a collision warning.

    Radar confirmation needs distinct fresh estimates of the same physical
    vehicle. Re-consuming a publication or switching tracks cannot assemble
    a warning from unrelated evidence. Vision keeps its original tick count.
    """
    if (not qualified_lead(lead, 'fcw', np.nextafter(0.9, 1.0), now_ns=now_ns)
        or not np.any(trajectory[FCW_IDXS, 0] - self.x_sol[FCW_IDXS, 0] < CRASH_DISTANCE)):
      self.reset_fcw()
      return
    evidence = getattr(lead, 'evidence', None)
    if evidence is not None and evidence.version == 1 and str(evidence.source) in ('radar', 'fused'):
      identity, epoch = lead_identity(lead), int(evidence.estimateMonoTime)
      previous = self._radar_fcw_epoch
      if identity != self._radar_fcw_identity or previous is None or not 0 <= epoch - previous <= 250_000_000:
        self.crash_cnt = 1
      elif epoch > previous:
        self.crash_cnt += 1
      self._radar_fcw_identity, self._radar_fcw_epoch = identity, epoch
      self.fcw_object_identity = identity
    else:
      if self._radar_fcw_identity is not None:
        self.crash_cnt = 0
      self._radar_fcw_identity = self._radar_fcw_epoch = None
      self.fcw_object_identity = None
      self.crash_cnt += 1

  @staticmethod
  def lead_influence_token(lead, *, model_ns=0, slot=None):
    """Frame-scoped visual identity is separate from physical radar identity."""
    if lead is not None and hasattr(lead, 'as_reader'):
      lead = lead.as_reader()
    if lead is None or not lead.present:
      return None
    identity = lead_identity(lead)
    if identity is not None:
      return identity
    evidence = getattr(lead, 'evidence', None)
    if evidence is not None and evidence.version == 1 and str(evidence.source) == 'vision':
      model_ns, slot = evidence.modelMonoTime, evidence.visionIndex
    if isinstance(model_ns, Integral) and model_ns > 0 and slot in (0, 1):
      return 'vision', int(model_ns), int(slot)
    return None

  def record_object_influence(self):
    """Objects active in this accepted problem, not counterfactual CAN effect.

    Consumers must still check that the owner adopted this reference and no
    stronger final cap replaced it. A geometrically nearest but zero-cost,
    nonbinding object does not acquire a contribution merely by being present.
    """
    self.object_influence_identities = ()
    if self.solution_status != 0 or not self.previous_solution_valid or self._influence_problem is None:
      return
    obstacles, tokens, compact_slots, reference = self._influence_problem
    comfort_bases = np.array([reference.comfort_stop_distance if reference is not None and slot is not None else STOP_DISTANCE
                              for slot in compact_slots])
    risk_bases = np.array([reference.lead_risk_stop_distances[slot] if reference is not None and slot is not None else STOP_DISTANCE
                           for slot in compact_slots])
    comfort_obstacles = obstacles + STOP_DISTANCE - comfort_bases
    active = set()
    tolerance = 1e-6
    for stage in range(1, N+1):
      position, velocity = self.x_sol[stage, :2]
      desired = get_safe_obstacle_distance(velocity, self.params[stage, 4])
      minimum = float(np.min(comfort_obstacles[stage]))
      error = minimum - position - desired
      typed = self.params[stage, TYPED_MODEL_MODE] > 0.5
      if (self.diagnostic_reference_weights[stage, 0] > 0.0
          and (error < -tolerance if typed else abs(error) > tolerance)):
        active.update(token for column, token in enumerate(tokens) if token is not None
                      and abs(comfort_obstacles[stage, column] - minimum) <= tolerance)
      if stage < N:
        margins = obstacles[stage] - position - self.params[stage, 5] * (desired - STOP_DISTANCE + risk_bases)
        tightest = float(np.min(margins))
        if self._constraint_cost_weights[-1] > 0.0 and tightest < -tolerance:
          active.update(token for column, token in enumerate(tokens) if token is not None
                        and abs(margins[column] - tightest) <= tolerance)
    self.object_influence_identities = tuple(sorted((token for token in active if token is not None), key=repr))

  def reference_obstacles(self, reference, t_follow):
    if reference.typed_model is not None and not isinstance(reference.typed_model, ModelReferenceEvidence):
      raise ValueError('Invalid typed model reference')
    if reference.cruise_speed is None and reference.typed_model is not None:
      raise ValueError('Typed model references require an explicit cruise goal')
    accel_min = np.broadcast_to(np.asarray(reference.accel_min, dtype=float), (N+1,))
    accel_max = np.broadcast_to(np.asarray(reference.accel_max, dtype=float), (N+1,))
    cruise_min_accel = np.broadcast_to(np.asarray(reference.cruise_min_accel, dtype=float), (N+1,))
    if (not np.all(np.isfinite([accel_min, accel_max, cruise_min_accel]))
        or np.any(cruise_min_accel > 0.0)
        or np.any(accel_min > accel_max) or np.any(accel_min > ACCEL_MAX) or np.any(accel_max < ACCEL_MIN)):
      raise ValueError("Invalid cruise reference")

    accel_min = np.clip(accel_min, ACCEL_MIN, ACCEL_MAX)
    accel_max = np.clip(accel_max, ACCEL_MIN, ACCEL_MAX)
    cruise_min_accel = np.clip(cruise_min_accel, accel_min, accel_max)
    obstacles, sources = [], []
    if reference.cruise_speed is not None:
      cruise_speed = np.broadcast_to(np.asarray(reference.cruise_speed, dtype=float), (N+1,))
      if not np.all(np.isfinite(cruise_speed)) or np.any(cruise_speed < 0.0):
        raise ValueError("Invalid cruise reference")
      lower_speed = np.maximum(0.0, self.x0[1] + np.cumsum(T_DIFFS * cruise_min_accel))
      upper_speed = np.maximum(0.0, self.x0[1] + np.cumsum(T_DIFFS * accel_max))
      cruise_velocity = np.clip(cruise_speed, lower_speed, upper_speed)
      cruise_position = np.cumsum(T_DIFFS * cruise_velocity)
      obstacles.append(cruise_position + get_safe_obstacle_distance(cruise_velocity, t_follow))
      sources.append(LongitudinalPlanSource.cruise)

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

  @classmethod
  def trajectory_dynamics_defects(cls, states, jerks):
    """Stored nodes minus their constant-jerk endpoints, without repairing them."""
    return np.array([states[stage+1] - cls.integrate_jerk(states[stage], jerks[stage], duration)
                     for stage, duration in enumerate(np.diff(T_IDXS))])

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
  def initial_rest_split(state, duration):
    split = duration / 2.0
    if state[2] < 0.0:
      split = min(split, np.nextafter(2.0 * state[1] / -state[2], 0.0))
    return split

  @staticmethod
  def rest_arithmetic_roundoff(values):
    # Guard arithmetic cancellation at a closed face. This is not a physical
    # acceptance allowance or an acados convergence tolerance.
    return 16.0 * np.finfo(float).eps * max(REST_STATE_TOL, float(np.max(np.abs(values))))

  @classmethod
  def future_rest_acceleration_interval(cls, state, node):
    """First acceleration compatible with the actual forward native problem.

    Future rest means v in [0,tol], a in [-tol,0], with ideal reference zero.
    For node two, v2=delta/2+(h0+h1)*a1/2+h1*a2/2, delta=2*v0+a0*h0.
    Intersect that terminal box with the existing first-node box and the same
    first-interval Bernstein row that the native solver will receive.
    An empty interval is discrete-model infeasibility, not physical inability
    to stop. Final candidate acceptance still uses the original physical test.
    """
    _, velocity, acceleration = state
    h0, h1 = np.diff(T_IDXS[:3])
    if node == 1:
      minimum = max(-REST_STATE_TOL, -acceleration - 2.0 * velocity / h0)
      maximum = min(0.0, -acceleration + 2.0 * (REST_STATE_TOL - velocity) / h0)
    elif node == 2:
      split = cls.initial_rest_split(state, h0)
      if split <= 0.0:
        return None
      delta = 2.0 * velocity + acceleration * h0
      minimum = max(-REFERENCE_BOUND, -acceleration - 2.0 * velocity / h0,
                    -delta / (h0 + h1), -delta / split)
      maximum = min(REFERENCE_BOUND, 2.0 * (REFERENCE_BOUND - velocity) / h0 - acceleration,
                    (2.0 * REST_STATE_TOL - delta + h1 * REST_STATE_TOL) / (h0 + h1),
                    (2.0 * REFERENCE_BOUND - delta) / split)
    else:
      raise ValueError('Discrete rest interval is only defined for nodes one and two')
    roundoff = cls.rest_arithmetic_roundoff([minimum, maximum, acceleration, 2.0 * velocity / h0])
    return None if minimum - maximum > roundoff else (minimum, maximum)

  @classmethod
  def future_rest_primal(cls, initial_state, node, interval=None):
    """A continuable primal guess, never a substitute for the native solve.

    Short goals use their admitted interval. Longer goals ramp to a constant
    acceleration and return to a=0 in the final interval. When strong initial
    braking needs a positive first acceleration to remain forward, first meet
    the same Bernstein face, then use the remaining intervals to stop.
    Every segment and the tail are checked before the guess is installed.
    The stored a_prev cost history is deliberately not modified.
    """
    durations = np.diff(T_IDXS)
    targets = np.zeros(N)
    if node in (1, 2):
      interval = interval if interval is not None else cls.future_rest_acceleration_interval(initial_state, node)
      if interval is None:
        return None
      minimum, maximum = interval
      targets[0] = maximum if node == 1 else 0.5 * (minimum + maximum)
    elif 3 <= node <= N:
      h0, h1 = durations[:2]
      last, total = durations[node-1], T_IDXS[node]
      velocity, acceleration = initial_state[1:]
      delta = 2.0 * velocity + acceleration * h0
      if delta >= 0.0:
        # Integrate the initial ramp, constant segment and final ramp to get v_node=0.
        targets[:node-1] = -(velocity + 0.5 * acceleration * h0) / (total - 0.5 * (h0 + last))
      else:
        split = cls.initial_rest_split(initial_state, h0)
        if split <= 0.0:
          return None
        targets[0] = -delta / split
        first_velocity = velocity + 0.5 * (acceleration + targets[0]) * h0
        targets[1:node-1] = -(first_velocity + 0.5 * targets[0] * h1) / (total - h0 - 0.5 * (h1 + last))
    else:
      return None
    states = np.empty((N+1, X_DIM))
    jerks = np.zeros(N)
    states[0] = initial_state
    for stage, duration in enumerate(durations):
      next_acceleration = targets[stage]
      if stage == 1 and node == 2:
        without_final_acceleration = states[stage, 1] + 0.5 * states[stage, 2] * duration
        if without_final_acceleration > REST_STATE_TOL:
          next_acceleration = 2.0 * (REST_STATE_TOL - without_final_acceleration) / duration
      jerks[stage] = (next_acceleration - states[stage, 2]) / duration
      states[stage+1] = cls.integrate_jerk(states[stage], jerks[stage], duration)
    roundoff = cls.rest_arithmetic_roundoff(np.concatenate((states.ravel(), jerks)))
    if (not np.all(np.isfinite(states)) or not np.all(np.isfinite(jerks))
        or np.max(np.abs(states[1:node, 1:]), initial=0.0) > REFERENCE_BOUND + roundoff
        or np.max(np.abs(states[node:, 1:])) > REST_STATE_TOL + roundoff
        or np.max(states[node:, 2]) > roundoff
        or cls.trajectory_minimum_velocity(states, jerks) < -roundoff
        or cls.continuous_rest_bound_violation(initial_state, states, jerks, node) > roundoff):
      return None
    return states, jerks

  @classmethod
  def scheduled_rest_trajectory_error(cls, initial_state, states, jerks, node, parameters, *, tolerance,
                                      recovery_position_limit=None):
    """Check an entire scheduling witness or actual solution, not just its tail.

    Acceleration stays inside its envelope. The native soft following-distance
    row must be nonnegative, or recover an existing deficit through arrival.
    Arrival uses the headway interval of the already-validated rest precision;
    the resting tail keeps its state and complete observed-geometry contracts.
    Neither that geometry nor the soft row is a vehicle-response safety proof.
    """
    if (not isinstance(node, Integral) or isinstance(node, bool) or not 1 <= node <= N
        or states.shape != (N+1, X_DIM) or jerks.shape != (N,)
        or parameters.shape != (N+1, PARAM_DIM)
        or not np.all(np.isfinite(states)) or not np.all(np.isfinite(jerks))
        or not np.all(np.isfinite(parameters[:, [0, 1, 2, 4, 5, RISK_STOP_DISTANCE, COMFORT_NORMALIZER, RISK_NORMALIZER]]))
        or np.any(parameters[:, [COMFORT_NORMALIZER, RISK_NORMALIZER]] != 0.0)
        or np.any(parameters[:, 5] <= 0.0)
        or np.any(parameters[:, RISK_STOP_DISTANCE] < 4.0) or np.any(parameters[:, RISK_STOP_DISTANCE] > STOP_DISTANCE)):
      return 'invalid_scheduled_rest_trajectory'
    defects = cls.trajectory_dynamics_defects(states, jerks)
    if max(float(np.max(np.abs(defects))), float(np.max(np.abs(states[0] - initial_state)))) > tolerance:
      return 'scheduled_rest_dynamics_residual_exceeded'
    if (np.max(np.abs(states[1:, 1:])) > REFERENCE_BOUND + tolerance
        or np.max(np.abs(states[node:, 1:])) > REST_STATE_TOL + tolerance
        or np.max(states[node:, 2]) > tolerance):
      return 'scheduled_rest_state_bounds_violated'
    if (cls.trajectory_minimum_velocity(states, jerks) < -tolerance
        or cls.continuous_rest_bound_violation(initial_state, states, jerks, node) > tolerance):
      return 'scheduled_rest_continuous_velocity_violated'
    if (np.any(states[:, 2] < parameters[:, 0] - tolerance)
        or np.any(states[:, 2] > parameters[:, 1] + tolerance)):
      return 'scheduled_rest_acceleration_envelope_violated'
    distance_margin = ((parameters[:, 2] - states[:, 0])
                       - parameters[:, 5] * (get_safe_obstacle_distance(states[:, 1], parameters[:, 4])
                                              - STOP_DISTANCE + parameters[:, RISK_STOP_DISTANCE]))
    initial_margin = ((parameters[0, 2] - initial_state[0])
                      - parameters[0, 5] * (get_safe_obstacle_distance(initial_state[1], parameters[0, 4])
                                             - STOP_DISTANCE + parameters[0, RISK_STOP_DISTANCE]))
    if initial_margin >= -tolerance * (initial_state[1] + 10.0):
      # Keep the original admission rule when no existing deficit needs recovery.
      risk_normalizer = states[:, 1] + 10.0
      if np.any(distance_margin / risk_normalizer < -tolerance):
        return 'scheduled_rest_lead_danger_envelope_violated'
    else:
      if (not isinstance(recovery_position_limit, (int, float, np.integer, np.floating))
          or isinstance(recovery_position_limit, (bool, np.bool_))
          or not np.isfinite(recovery_position_limit) or recovery_position_limit <= initial_state[0]):
        return 'scheduled_rest_lead_danger_envelope_violated'
      if cls.trajectory_maximum_position(states, jerks) > recovery_position_limit + tolerance:
        return 'scheduled_rest_recovery_position_limit_violated'
      # Carry the approach deficit in metres, not the speed-normalized residual.
      # At arrival the accepted rest box represents ideal v=0: the headway term
      # ranges from factor*risk_stop_distance to that base plus the speed terms
      # at REST_STATE_TOL, using the same stage base as the native risk row.
      # Test its lower endpoint using the ACTUAL position/obstacle. This changes
      # only a soft preference's precision, never the trajectory, numerical
      # tolerance or raw lead geometry. Real position loss is still counted.
      deficit = np.maximum(0.0, -distance_margin[:node+1])
      deficit[node] = max(0.0, parameters[node, 5] * parameters[node, RISK_STOP_DISTANCE]
                          - (parameters[node, 2] - states[node, 0]))
      best_previous = np.minimum.accumulate(np.concatenate(([-initial_margin], deficit[:-1])))
      distance_tolerance = tolerance * (initial_state[1] + 10.0)
      if np.any(deficit > best_previous + distance_tolerance):
        return 'scheduled_rest_lead_danger_recovery_worsened'
      # After arrival, the full checked rest box, continuous velocity and raw
      # geometry retain responsibility. A soft headway change cannot impose an
      # extra exact-zero-displacement contract on the accepted resting tail.
    return ''

  def schedule_future_rest(self):
    """Select the first later node with a complete template witness; never solve.

    This bounded family can miss other feasible trajectories. Rejecting a
    template does not prove its node or every later node is unreachable.
    """
    goal = self.rest_goal
    candidates = []
    self.diagnostic_rest_goal['schedulingCandidates'] = candidates
    for node in range(goal.native_node + 1, N+1):
      interval = None
      if node in (1, 2):
        interval = self.future_rest_acceleration_interval(self.x0, node)
        if interval is None:
          candidates.append({'node': node, 'reason': 'no_forward_rest_primal'})
          continue
        # The interval describes a_1, including when the destination is node
        # two. Intersect before choosing its midpoint, without inventing an
        # acceleration just outside this update's real first-node envelope.
        minimum = max(interval[0], float(self.params[1, 0]))
        maximum = min(interval[1], float(self.params[1, 1]))
        if minimum > maximum:
          candidates.append({'node': node, 'reason': 'scheduled_rest_acceleration_interval_empty'})
          continue
        interval = minimum, maximum
      primal = self.future_rest_primal(self.x0, node, interval=interval)
      if primal is None:
        reason = 'no_forward_rest_primal'
      else:
        states, jerks = primal
        arithmetic = self.rest_arithmetic_roundoff(np.concatenate((states.ravel(), jerks)))
        reason = self.scheduled_rest_trajectory_error(
          self.x0, states, jerks, node, self.params, tolerance=arithmetic,
          recovery_position_limit=getattr(self, 'rest_recovery_position_limit', None),
        )
      candidates.append({'node': node, 'reason': reason})
      if not reason:
        self.effective_rest_node = node
        self.effective_rest_state = 'constrained'
        initial_deficit = max(0.0, float(self.params[0, 5] * (get_safe_obstacle_distance(self.x0[1], self.params[0, 4])
                                                            - STOP_DISTANCE + self.params[0, RISK_STOP_DISTANCE])
                                         - (self.params[0, 2] - self.x0[0])))
        recovering = initial_deficit > arithmetic * (self.x0[1] + 10.0)
        self.diagnostic_rest_goal.update(
          effectiveNativeNode=node, effectiveNativeTime=float(T_IDXS[node]), effectiveState='constrained',
          rescheduleDelay=float(T_IDXS[node] - goal.native_time_s),
          effectiveLateness=float(T_IDXS[node] - goal.remaining_time_s), scheduledPrimalVerified=True,
          scheduledDangerPolicy='recover_initial_deficit' if recovering else 'nonnegative_margin',
          scheduledRecoveryScope='through_arrival_with_rest_headway_precision' if recovering else None,
          scheduledTailContract='rest_state_and_raw_lead_geometry' if recovering else None,
          initialDangerDeficitMeters=initial_deficit,
          recoveryPositionLimit=getattr(self, 'rest_recovery_position_limit', None) if recovering else None,
        )
        return primal
    return None

  @classmethod
  def continuous_rest_bound_violation(cls, initial_state, states, jerks, node):
    violation = 0.0
    for stage in range(max(0, node - 1)):
      duration = T_IDXS[stage+1] - T_IDXS[stage]
      split = cls.initial_rest_split(initial_state, duration) if stage == 0 else duration / 2.0
      b1 = states[stage, 1] + states[stage, 2] * split / 2.0
      b2 = states[stage, 1] + states[stage, 2] * (duration + split) / 2.0 + jerks[stage] * duration * split / 2.0
      violation = max(violation, -b1, -b2, b2 - REFERENCE_BOUND, b1 - REFERENCE_BOUND if stage else 0.0)
    return float(violation)

  @staticmethod
  def segment_minimum_velocity(state, jerk, duration):
    _, velocity, acceleration = state
    minimum = min(velocity, velocity + acceleration * duration + 0.5 * jerk * duration**2)
    if jerk > 0.0 and 0.0 < -acceleration / jerk < duration:
      minimum = min(minimum, velocity - acceleration**2 / (2.0 * jerk))
    return float(minimum)

  @staticmethod
  def trajectory_minimum_velocity(states, jerks):
    minimum = float(np.min(states[:, 1]))
    for stage, duration in enumerate(np.diff(T_IDXS)):
      if jerks[stage] > 0.0:
        extremum = -states[stage, 2] / jerks[stage]
        if 0.0 < extremum < duration:
          minimum = min(minimum, float(states[stage, 1] + states[stage, 2] * extremum
                                      + 0.5 * jerks[stage] * extremum**2))
    return minimum

  @classmethod
  def trajectory_maximum_position(cls, states, jerks):
    """Include every constant-jerk position extremum and the residual-speed tail."""
    maximum = float(np.max(states[:, 0]))
    for state, jerk, duration in zip(states[:-1], jerks, np.diff(T_IDXS), strict=True):
      maximum = max(maximum, float(cls.integrate_jerk(state, jerk, duration)[0]))
      velocity, acceleration = state[1:]
      roots = ()
      if jerk == 0.0:
        if acceleration != 0.0:
          roots = (-velocity / acceleration,)
      else:
        discriminant = acceleration**2 - 2.0 * jerk * velocity
        if discriminant >= 0.0:
          numerator = -acceleration - np.copysign(np.sqrt(discriminant), acceleration)
          roots = (numerator / jerk, 2.0 * velocity / numerator) if numerator != 0.0 else (0.0,)
      for root in roots:
        if 0.0 < root < duration:
          maximum = max(maximum, float(cls.integrate_jerk(state, jerk, root)[0]))
    return maximum

  @classmethod
  def rest_precision_control(cls, state, raw_jerk, duration, lower, upper):
    interval = cls.rest_acceleration_interval(state, duration, lower, upper)
    if interval is None:
      return None
    minimum, maximum = interval
    acceleration = state[2]
    desired = acceleration + raw_jerk * duration
    next_acceleration = min(max(desired, minimum), maximum)
    jerk = (next_acceleration - acceleration) / duration
    # Bound arithmetic can round one ULP outside a closed face. Test the
    # adjacent representable controls too; all states still follow dynamics.
    candidates = [jerk, np.nextafter(jerk, -np.inf), np.nextafter(jerk, np.inf)]
    candidates.sort(key=lambda candidate: abs(candidate - raw_jerk))
    for candidate in candidates:
      next_state = cls.integrate_jerk(state, candidate, duration)
      if (minimum <= next_state[2] <= maximum
          and np.all(next_state[1:] >= lower) and np.all(next_state[1:] <= upper)
          and cls.segment_minimum_velocity(state, candidate, duration) >= 0.0):
        return candidate, next_state
    return None

  @classmethod
  def adjacent_rest_precision_controls(cls, state, raw_jerks, durations, lower, upper):
    """Propose two integrated controls when a greedy endpoint has no interval.

    A pair can change terminal v and a independently. This only proposes a
    correction; the caller retains the original jerk and propagation budgets.
    """
    h0, h1 = durations
    middle = cls.integrate_jerk(state, raw_jerks[0], h0)
    terminal = cls.integrate_jerk(middle, raw_jerks[1], h1)
    # Choose an interior endpoint to solve for jerk, without clipping any
    # published state. The margin protects arithmetic cancellation at a face;
    # it neither enlarges that face nor replaces the independent 1e-9 budget.
    roundoff = cls.rest_arithmetic_roundoff(np.concatenate((state, middle, terminal, raw_jerks)))
    margin = np.minimum(roundoff, 0.5 * (upper[1] - lower[1]))
    target = np.clip(terminal[1:], lower[1] + margin, upper[1] - margin)
    velocity_delta, acceleration_delta = target - terminal[1:]
    first_delta = (2.0 * velocity_delta - h1 * acceleration_delta) / (h0 * (h0 + h1))
    second_delta = (acceleration_delta - h0 * first_delta) / h1
    jerks = raw_jerks + [first_delta, second_delta]
    middle = cls.integrate_jerk(state, jerks[0], h0)
    terminal = cls.integrate_jerk(middle, jerks[1], h1)
    for index, (initial, final, jerk, duration) in enumerate(zip((state, middle), (middle, terminal), jerks, durations, strict=True)):
      if (not np.all(np.isfinite(final)) or not np.isfinite(jerk)
          or np.any(final[1:] < lower[index]) or np.any(final[1:] > upper[index])
          or cls.segment_minimum_velocity(initial, jerk, duration) < 0.0):
        return None
    return jerks, middle, terminal

  def project_rest_precision(self, *, node=None):
    """Remove bounded numerical defects by changing jerk, never clipping x/v/a.

    Only a successful native solve reaches this method. The raw
    iterate must already satisfy the much smaller QP feasibility budget.
    Corrections are bounded by the jerk effect on endpoint v/a; the resulting
    x/v/a difference is independently bounded through the linear dynamics.
    """
    durations = np.diff(T_IDXS)
    lower, upper = self.diagnostic_reference_bounds
    raw_states, raw_jerks = self.x_sol, self.u_sol[:, 0]
    defects = self.trajectory_dynamics_defects(raw_states, raw_jerks)
    maximum_defect = np.maximum(np.max(np.abs(defects), axis=0), np.abs(raw_states[0] - self.x0))
    maximum_violation = float(max(0.0, np.max(lower[1:] - raw_states[1:, 1:]),
                                  np.max(raw_states[1:, 1:] - upper[1:])))
    node = self.rest_goal.native_node if node is None else node
    certificate_violation = self.continuous_rest_bound_violation(self.x0, raw_states, raw_jerks, node)
    diagnostics = {'applied': False, 'feasibilityTolerance': QP_FEASIBILITY_TOL,
                   'maximumNativeDynamicsDefect': maximum_defect.tolist(),
                   'maximumNativeBoundViolation': maximum_violation,
                   'maximumNativeContinuousBoundViolation': certificate_violation}
    self.diagnostic_rest_goal['numericalProjection'] = diagnostics
    if np.max(maximum_defect) > QP_FEASIBILITY_TOL:
      return 'native_rest_dynamics_residual_exceeded'
    if max(maximum_violation, certificate_violation) > QP_FEASIBILITY_TOL:
      return 'native_rest_precision_budget_exceeded'

    # Preserve the existing accepted precision, including sub-tolerance
    # continuous-speed residuals. A physically accepted native trajectory must
    # not become a new failure merely because a stricter projection is costly.
    required = (np.max(np.abs(raw_states[node:, 1:])) > REST_STATE_TOL
                or self.trajectory_minimum_velocity(raw_states, raw_jerks) < -REST_STATE_TOL)
    diagnostics['required'] = bool(required)
    if not required:
      return ''

    states, jerks = np.empty_like(raw_states), np.empty_like(raw_jerks)
    states[0] = self.x0
    difference_budget = np.full(3, QP_FEASIBILITY_TOL)
    largest_budget = difference_budget.copy()
    largest_jerk_correction = 0.0
    for stage, duration in enumerate(durations):
      selected = self.rest_precision_control(states[stage], raw_jerks[stage], duration, lower[stage+1], upper[stage+1])
      if selected is None and stage > 0 and node > 0:
        pair = self.adjacent_rest_precision_controls(states[stage-1], raw_jerks[stage-1:stage+1], durations[stage-1:stage+1],
                                                    lower[stage:stage+2], upper[stage:stage+2])
        if pair is not None:
          pair_jerks, middle, terminal = pair
          prior_duration = durations[stage-1]
          prior_budget = QP_FEASIBILITY_TOL * (1.0 / prior_duration + 2.0 / prior_duration**2)
          prior_correction = abs(pair_jerks[0] - raw_jerks[stage-1])
          # difference_budget already describes this prior endpoint. Recheck
          # it after replacing its greedy control; the current endpoint goes
          # through the unchanged budget checks immediately below.
          if prior_correction > prior_budget or np.any(np.abs(middle - raw_states[stage]) > difference_budget):
            return 'native_rest_precision_budget_exceeded'
          jerks[stage-1], states[stage] = pair_jerks[0], middle
          largest_jerk_correction = max(largest_jerk_correction, prior_correction)
          selected = pair_jerks[1], terminal
          diagnostics.setdefault('adjacentJerkCorrectionStages', []).append([stage-1, stage])
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

    corrected_certificate_violation = self.continuous_rest_bound_violation(self.x0, states, jerks, node)
    if corrected_certificate_violation > QP_FEASIBILITY_TOL:
      return 'native_rest_precision_budget_exceeded'
    diagnostics.update(applied=bool(np.any(states != raw_states) or np.any(jerks != raw_jerks)),
                       maximumCorrectedContinuousBoundViolation=corrected_certificate_violation,
                       maximumStateCorrection=np.max(np.abs(states - raw_states), axis=0).tolist(),
                       maximumStateCorrectionBudget=largest_budget.tolist(),
                       maximumJerkCorrection=largest_jerk_correction)
    self.x_sol[:] = states
    self.u_sol[:, 0] = jerks
    # These are views established by run(). Keep the next reference's history
    # tied to the same corrected trajectory, not its pre-projection version.
    self.a_prev = np.interp(T_IDXS + self.dt, T_IDXS, self.a_solution)
    return ''

  def set_continuous_rest_bounds(self):
    """Certify future-rest velocity between nodes with linear constraints.

    Split [0,h] at tau. The two quadratic Bernstein middle coefficients are
    b1=v+a*tau/2 and b2=v+a*(h+tau)/2+j*h*tau/2. Their weighted average
    v(tau)=(1-tau/h)*b1+(tau/h)*b2 is nonnegative when both are. Together
    with v(0),v(h)>=0, each subinterval has three nonnegative coefficients,
    hence its entire quadratic velocity is nonnegative. This is a conservative
    feasible subset, not a sample-only check or a nonlinear RTI approximation.
    """
    node = self.effective_rest_node
    if node is None:
      node = self.rest_goal.native_node if self.rest_goal is not None else 0
    future = (self.rest_goal_state == 'constrained' or self.effective_rest_state == 'constrained') and node > 0
    # The interval-rest ramp ends at exact v=a=0, then holds. Its shorter
    # native integration and independent event-aware acceptance replace the
    # whole-interval Bernstein rows, which would describe a different curve.
    if self.interval_rest is not None:
      future = False
    # Future rest ends the entering interval at a<=0. Its velocity is monotone
    # or concave, so the minimum is an endpoint. Later intervals have both
    # accelerations <=0 and are monotone. Node v>=0 certifies these intervals
    # without redundant C rows; exact due-rest keeps its existing separate path.
    constrained_stages = max(0, node - 1) if future else 0
    splits = []
    for stage, duration in enumerate(np.diff(T_IDXS)):
      C, D, lower, upper = self._rest_constraint_buffers[stage]
      C.fill(0.0)
      D.fill(0.0)
      lower.fill(-1.0)
      upper.fill(1.0)
      if stage < constrained_stages:
        # x0 is fixed. Admission and the native C row share this exact split.
        split = self.initial_rest_split(self.x0, duration) if stage == 0 else duration / 2.0
        C[1] = [0.0, 1.0, (duration + split) / 2.0]
        D[1, 0] = duration * split / 2.0
        lower[1], upper[1] = 0.0, REFERENCE_BOUND
        if stage:
          C[0] = [0.0, 1.0, split / 2.0]
          lower[0], upper[0] = 0.0, REFERENCE_BOUND
        # At stage zero b1 is fixed and was certified by the split choice.
        # Keeping that row inactive avoids a redundant initial-state equality.
        splits.append(float(split))
      self.solver.constraints_set(stage, 'C', C)
      self.solver.constraints_set(stage, 'D', D)
      self.solver.constraints_set(stage, 'lg', lower)
      self.solver.constraints_set(stage, 'ug', upper)
    if future:
      self.diagnostic_rest_goal['continuousSpeedCertificate'] = 'quadratic_bernstein'
      self.diagnostic_rest_goal['continuousSpeedSplitTimes'] = splits

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
        # The expanded polynomial cancels nearly equal terms at a tangency.
        # Use its algebraic extremum value, matching the upward-ULP jerk above.
        if 0.0 < extremum < duration and velocity - acceleration**2 / (2.0 * jerk) < 0.0:
          return None
      states[stage+1] = (position + velocity * duration + 0.5 * acceleration * duration**2 + jerk * duration**3 / 6.0,
                         next_velocity, next_acceleration)
      jerks[stage] = jerk
    return states, jerks

  def set_reference_problem(self, reference, x_obstacles, sources, accel_min, t_follow, *, lead_slots=None):
    """Install genuine costs and state bounds in this same native problem."""
    if lead_slots is None:
      lead_slots = tuple(MPC_SOURCES.index(source) if source in MPC_SOURCES else None for source in sources)
    if len(lead_slots) != len(sources) or any(slot not in (None, 0, 1) for slot in lead_slots):
      raise ValueError('Invalid compact lead assignment')
    typed = reference.typed_model if reference is not None else None
    if typed is not None and not isinstance(typed, ModelReferenceEvidence):
      raise ValueError('Invalid typed model reference')
    self.rest_goal = typed.rest_goal if typed is not None else None
    self.interval_rest = None
    self.rest_goal_state = 'none'
    self.rest_goal_reason = ''
    self.rest_goal_braking_accel = None
    self.effective_rest_node = None
    self.effective_rest_state = 'none'
    self.effective_rest_reason = ''
    self.diagnostic_rest_goal = None
    self.yref[:, :] = 0.0
    self.params[:, MODEL_V_REF:TYPED_MODEL_MODE+1] = 0.0
    self.params[:, COMFORT_NORMALIZER:RISK_NORMALIZER+1] = 0.0
    selected = np.argmin(x_obstacles, axis=1)
    comfort_selected = selected
    self.params[:, COMFORT_STOP_DISTANCE] = STOP_DISTANCE
    self.params[:, RISK_STOP_DISTANCE] = STOP_DISTANCE
    if reference is not None and any(distance < STOP_DISTANCE for distance in reference.lead_risk_stop_distances):
      lead_factors = self.params[:, 5]
      if not np.all(np.isfinite(lead_factors)) or np.any(lead_factors <= 0.0):
        raise ValueError('Invalid lead danger factor')
      risk_obstacles = x_obstacles.copy()
      for obstacle_index, slot in enumerate(lead_slots):
        if slot is not None:
          distance = reference.lead_risk_stop_distances[slot]
          risk_obstacles[:, obstacle_index] += lead_factors * (STOP_DISTANCE - distance)
      # min_i(x_i - factor * base_i) is a separate minimum from the raw
      # obstacle and comfort minima. Fold only its base into the native row;
      # p[2] remains the unmodified minimum, and virtual obstacles keep base 6.
      distance_offset = np.min(risk_obstacles, axis=1) - np.min(x_obstacles, axis=1)
      self.params[:, RISK_STOP_DISTANCE] = np.clip(STOP_DISTANCE - distance_offset / lead_factors, 4.0, STOP_DISTANCE)
    if reference is not None and reference.comfort_stop_distance < STOP_DISTANCE:
      comfort_obstacles = x_obstacles.copy()
      for obstacle_index, slot in enumerate(lead_slots):
        if slot is not None:
          comfort_obstacles[:, obstacle_index] += STOP_DISTANCE - reference.comfort_stop_distance
      comfort_selected = np.argmin(comfort_obstacles, axis=1)
      # Fold the separate comfort minimum into one native cost parameter. The
      # unchanged p[2] remains the original raw minimum, including real leads.
      # Virtual cruise/model limits never move: shortening those would invent
      # a distance deficit at rest and alter the requested cruise/model speed.
      distance_offset = np.min(comfort_obstacles, axis=1) - np.min(x_obstacles, axis=1)
      self.params[:, COMFORT_STOP_DISTANCE] = np.clip(STOP_DISTANCE - distance_offset,
                                                     reference.comfort_stop_distance, STOP_DISTANCE)
    weights = self.stage_cost_weights()
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
        if sources[comfort_selected[stage]] == LongitudinalPlanSource.cruise:
          weights[stage, 0] = 0.0
        if typed.moving is not None and sources[obstacle_index] in MPC_SOURCES:
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
      if type(reference.allow_interval_rest) is not bool:
        raise ValueError('Invalid interval-rest selection')
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
      elif (reference.allow_interval_rest and qualified and goal.native_node > 0
            and (early_rest := IntervalRest.from_state(self.x0, float(T_IDXS[1]), goal.native_time_s)) is not None
            and np.all(self.params[:, 0] <= 0.0) and np.all(self.params[:, 1] >= 0.0)
            and max(self.params[0, 0], self.params[1, 0]) <= self.x0[2] <= min(self.params[0, 1], self.params[1, 1])):
        self.interval_rest = early_rest
        self.rest_goal_state = 'constrained'
        self.diagnostic_rest_goal.update(
          trajectoryKind='interval_rest', stopTime=early_rest.stop_time,
          firstMotionDuration=early_rest.stop_time, firstHoldDuration=float(T_IDXS[1]) - early_rest.stop_time,
          sourceGoalUnchanged=True,
        )
      elif goal.native_node == 0 and (self.x0[1] < 0.0 or np.max(np.abs(self.x0[1:])) > REST_STATE_TOL):
        self.rest_goal_state, self.rest_goal_reason = 'infeasible', 'rest_goal_due_with_nonrest_initial_state'
      elif goal.native_node == 0 and (resting_primal := self.due_rest_primal(self.x0)) is None:
        self.rest_goal_state, self.rest_goal_reason = 'infeasible', 'rest_goal_due_without_nonnegative_rest_trajectory'
      elif goal.native_node > 0 and self.rest_acceleration_interval(
        self.x0, T_IDXS[1], np.array([0.0, -REFERENCE_BOUND]), np.full(2, REFERENCE_BOUND),
      ) is None:
        # Includes v0=0,a0<0 and tiny speeds that would require a first
        # acceleration beyond the existing native state box. Do not turn
        # that impossible or ill-conditioned segment into a native QP error.
        self.rest_goal_state, self.rest_goal_reason = 'infeasible', 'rest_goal_without_nonnegative_initial_segment'
      elif self.x0[1] + minimum_accel * goal.native_time_s > REST_STATE_TOL:
        self.rest_goal_state, self.rest_goal_reason = 'infeasible', 'rest_goal_time_below_braking_lower_bound'
      elif goal.native_node in (1, 2):
        future_interval = self.future_rest_acceleration_interval(self.x0, goal.native_node)
        if future_interval is not None:
          # This interval describes a_1, including for a node-two rest. A soft
          # acceleration slack cannot make an otherwise undeliverable stop
          # qualify. Use the same first-node envelope as later scheduling.
          acceleration_min, acceleration_max = map(float, self.params[1, :2])
          if future_interval[1] < acceleration_min or future_interval[0] > acceleration_max:
            future_interval = None
          else:
            # The native interval already accounts for arithmetic cancellation
            # at a closed face. Preserve that decision when both faces lie
            # inside the physical envelope; do not grant the envelope itself
            # an extra tolerance or replace a tiny reversed pair by a point.
            future_interval = (max(future_interval[0], acceleration_min),
                               min(future_interval[1], acceleration_max))
        if future_interval is None:
          # Reject this native destination, not every physical way to stop.
          # Existing later-node scheduling or explicit failure braking remains
          # responsible for an uncommandable nominal first acceleration.
          self.rest_goal_state, self.rest_goal_reason = 'infeasible', 'rest_goal_time_below_discrete_lower_bound'
        else:
          self.rest_goal_state = 'constrained'
      else:
        self.rest_goal_state = 'constrained'

      if type(reference.allow_future_rest_reschedule) is not bool:
        raise ValueError('Invalid future rest scheduling selection')
      scheduled_primal = None
      if (reference.allow_future_rest_reschedule and qualified and self.rest_goal_state == 'infeasible'
          and goal.native_node is not None and 0 < goal.native_node < N and goal.remaining_time_s > 0.0):
        self.diagnostic_rest_goal.update(nominalState=self.rest_goal_state, nominalReason=self.rest_goal_reason)
        scheduled_primal = self.schedule_future_rest()

      if self.rest_goal_state == 'constrained' or scheduled_primal is not None:
        node = 1 if self.interval_rest is not None else self.effective_rest_node if scheduled_primal is not None else goal.native_node
        if scheduled_primal is not None:
          # A scheduled destination has a complete checked witness. Retain its
          # acceleration envelope on every future interval; nominal short-rest
          # admission installs its first-node envelope separately below.
          lower[1:, 1], upper[1:, 1] = self.params[1:, 0], self.params[1:, 1]
        lower[max(1, node):, :] = 0.0
        upper[max(1, node):, :] = 0.0
        # A raw zero goal survives the anchored moving preference unchanged.
        # Rescheduling a constraint does not move the source's desired stop.
        self.params[goal.native_node:, MODEL_V_REF:MODEL_A_REF + 1] = 0.0
        if self.interval_rest is not None:
          # Exact rest at the event is required to omit the remaining motion
          # integration. Never use the wider rest box to discard real travel.
          self.params[1:, MODEL_V_REF:MODEL_A_REF + 1] = 0.0
          states, jerks = self.interval_rest.native_trajectory(T_IDXS)
          for stage in range(N+1):
            self.solver.set(stage, 'x', states[stage])
            if stage < N:
              self.solver.set(stage, 'u', np.array([jerks[stage]]))
          self.diagnostic_rest_goal.update(nativeVelocityBounds=[0.0, 0.0], nativeAccelerationBounds=[0.0, 0.0])
        elif node == 0:
          # Within the existing rest acceptance precision, exact endpoint
          # v=a=0 is generally unreachable from a nonzero fixed initial state:
          # a_1=0 implies v_1=v_0+a_0*dt/2 for this constant-jerk model.
          # Express that SAME accepted precision as feasible state bounds;
          # exact due-rest retains exact-zero bounds.
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
        else:
          # Ideal v/a and the event time stay unchanged. The native box now
          # expresses the existing accepted rest precision, with a<=0 so the
          # entering and later intervals retain a nonnegative-speed proof.
          lower[node:, :] = [0.0, -REST_STATE_TOL]
          upper[node:, :] = [REST_STATE_TOL, 0.0]
          if scheduled_primal is not None:
            lower[node:, 1] = np.maximum(lower[node:, 1], self.params[node:, 0])
            upper[node:, 1] = np.minimum(upper[node:, 1], self.params[node:, 1])
          elif node in (1, 2):
            # Admission and the actual solve must use the same a_1 envelope;
            # narrowing only the primal seed would still leave the solver's
            # soft acceleration row able to buy an undeliverable short stop.
            lower[1, 1] = max(lower[1, 1], self.params[1, 0])
            upper[1, 1] = min(upper[1, 1], self.params[1, 1])
          self.diagnostic_rest_goal.update(nativeVelocityBounds=[0.0, REST_STATE_TOL],
                                           nativeAccelerationBounds=[-REST_STATE_TOL, 0.0])
          future_primal = (scheduled_primal if scheduled_primal is not None else
                          self.future_rest_primal(self.x0, node, future_interval if node in (1, 2) else None))
          self.diagnostic_rest_goal['futurePrimalInitialized'] = future_primal is not None
          if future_primal is not None:
            future_states, future_jerks = future_primal
            for stage in range(N+1):
              self.solver.set(stage, 'x', future_states[stage])
              if stage < N:
                self.solver.set(stage, 'u', np.array([future_jerks[stage]]))
      if self.rest_goal_state in ('infeasible', 'unverified') and scheduled_primal is None:
        # Retain the failed goal while solving the existing lead/cruise risk
        # problem. The caller uses an explicit braking failure intent; it must
        # never publish this relaxed solution as goal satisfaction.
        self.rest_goal_braking_accel = minimum_accel

    self.set_continuous_rest_bounds()

    self.install_cost_weights(weights)
    for stage in range(N+1):
      self.solver.set(stage, 'yref', self.yref[stage, :COST_DIM if stage < N else COST_E_DIM])
      if stage:
        self.solver.constraints_set(stage, 'lbx', lower[stage])
        self.solver.constraints_set(stage, 'ubx', upper[stage])
    self.diagnostic_reference_weights = weights.copy()
    self.diagnostic_reference_bounds = (lower.copy(), upper.copy())

  def sample_solution(self, times):
    """Sample the accepted motion, including the interval-zero rest event."""
    if self.interval_rest is not None:
      if self.solution_status != 0 or self.rest_goal_state != 'satisfied':
        raise ValueError('Cannot sample an unaccepted interval-rest trajectory')
      states, jerk = self.interval_rest.sample(times)
      return states[..., 1], states[..., 2], jerk
    return (np.interp(times, T_IDXS, self.v_solution), np.interp(times, T_IDXS, self.a_solution),
            np.interp(times, T_IDXS[:-1], self.j_solution))

  def check_interval_rest_goal(self):
    """Verify a real native solve before publishing the braking/hold curve.

    The zero endpoint determines this ramp uniquely. Only numerical defects
    within the existing 1e-9 budget may be removed; a successful native return,
    geometry and acceleration checks remain mandatory. The public clock and
    the measured initial state never move to the end of the prefix.
    """
    event = self.interval_rest
    states, jerks = event.native_trajectory(T_IDXS)
    durations = np.diff(T_IDXS)
    durations[0] = event.stop_time
    defects = np.array([self.x_sol[stage+1] - self.integrate_jerk(self.x_sol[stage], self.j_solution[stage], duration)
                        for stage, duration in enumerate(durations)])
    state_correction = np.abs(self.x_sol - states)
    state_error = float(np.max(state_correction))
    motion_error = float(np.max(state_correction[:, 1:]))
    jerk_effect = float(np.max(np.abs(self.j_solution - jerks) * durations))
    dynamics_error = float(np.max(np.abs(defects)))
    # A sub-tolerance velocity residual accumulates position error over the
    # horizon. Propagate the existing component/defect budget through the
    # same dynamics; it is not a new fixed physical position allowance.
    # Raw v/a and each jerk's acceleration effect still have a 1e-9 bound.
    correction_budget = np.empty_like(states)
    correction_budget[0] = QP_FEASIBILITY_TOL
    for stage, duration in enumerate(durations):
      transition = np.array([[1.0, duration, 0.5 * duration**2], [0.0, 1.0, duration], [0.0, 0.0, 1.0]])
      control = np.array([duration**3 / 6.0, 0.5 * duration**2, duration])
      correction_budget[stage+1] = (transition @ correction_budget[stage]
                                    + control * abs(self.j_solution[stage] - jerks[stage]) + QP_FEASIBILITY_TOL)
    failure = ''
    if self.diagnostic_last_solver_return != 0 or self.solution_status != 0:
      failure = 'interval_rest_without_native_success'
    elif not np.all(np.isfinite(self.x_sol)) or not np.all(np.isfinite(self.j_solution)):
      failure = 'interval_rest_nonfinite_native_trajectory'
    elif (max(motion_error, jerk_effect, dynamics_error) > QP_FEASIBILITY_TOL
          or np.any(state_correction > correction_budget)):
      failure = 'interval_rest_native_precision_budget_exceeded'
    elif (np.any(states[:, 2] < self.params[:, 0]) or np.any(states[:, 2] > self.params[:, 1])
          or not max(self.params[0, 0], self.params[1, 0]) <= event.acceleration <= min(self.params[0, 1], self.params[1, 1])):
      failure = 'interval_rest_acceleration_envelope_violated'
    elif event.stop_time > self.rest_goal.native_time_s:
      failure = 'interval_rest_misses_source_goal'
    times, expanded_states, expanded_jerks = event.expanded_trajectory(T_IDXS)
    minimum_clearance = float('inf')
    if not failure:
      # Keep observed lead geometry, including the portion between the event
      # and public node one. The native soft headway cost remains unchanged.
      for trajectory in self._lead_trajectories:
        if trajectory.shape != (N+1, 2) or not np.all(np.isfinite(trajectory)):
          failure = 'interval_rest_invalid_lead_geometry'
          break
        expanded_lead = np.column_stack((np.interp(times, T_IDXS, trajectory[:, 0]),
                                         np.interp(times, T_IDXS, trajectory[:, 1])))
        minimum_clearance = min(minimum_clearance, self.lead_clearance_minimum(
          expanded_lead, expanded_states, expanded_jerks, times=times))
      if minimum_clearance < CRASH_DISTANCE:
        failure = 'interval_rest_lead_geometry_violated'
    self.diagnostic_rest_goal.update(
      maximumNativeDynamicsDefect=dynamics_error, maximumStateCorrection=state_error,
      maximumStateCorrectionBudget=np.max(correction_budget, axis=0).tolist(), maximumMotionStateCorrection=motion_error,
      maximumJerkAccelerationCorrection=jerk_effect,
      minimumObservedLeadClearance=minimum_clearance if np.isfinite(minimum_clearance) else None,
    )
    if failure:
      self.rest_goal_state, self.rest_goal_reason = 'unsatisfied', failure
      self.solution_status = MPC_GOAL_UNSATISFIED
      self.rest_goal_braking_accel = float(np.min(self.params[:, 0]))
      self.failure_reason = failure
      self.previous_solution_valid = False
      self.invalidate_reference_history()
    else:
      self.x_sol[:] = states
      self.u_sol[:, 0] = jerks
      self.rest_goal_state, self.rest_goal_reason = 'satisfied', ''
      self.rest_goal_braking_accel = None
      self.failure_reason = ''
      self.a_prev = event.sample(T_IDXS + self.dt)[0][:, 2]
      self.diagnostic_rest_goal.update(maximumRestStateError=0.0, minimumContinuousVelocity=0.0,
                                       intervalRestVerified=True)
    self.diagnostic_rest_goal.update(state=self.rest_goal_state, reason=self.rest_goal_reason)

  def check_rest_goal(self):
    """Judge the original source goal; scheduling never changes this contract."""
    if self.rest_goal is None:
      return
    if self.interval_rest is not None:
      self.check_interval_rest_goal()
      return
    if self.rest_goal_state in ('infeasible', 'unverified'):
      self.solution_status = MPC_GOAL_INFEASIBLE
      self.rest_goal_braking_accel = float(np.min(self.params[:, 0]))
    elif self.rest_goal_state in ('constrained', 'satisfied'):
      # Satisfaction belongs to a particular candidate. A same-problem RTI
      # refinement produces a new trajectory and must repeat this check.
      node = self.rest_goal.native_node
      error = float(np.max(np.abs(self.x_sol[node:, 1:])))
      minimum_velocity = self.trajectory_minimum_velocity(self.x_sol, self.j_solution)
      numerical_failure = ''
      if node == 0 or error > REST_STATE_TOL or minimum_velocity < -REST_STATE_TOL:
        numerical_failure = self.project_rest_precision()
        error = float(np.max(np.abs(self.x_sol[node:, 1:])))
        minimum_velocity = self.trajectory_minimum_velocity(self.x_sol, self.j_solution)
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

  def check_effective_rest_goal(self):
    """Accept a scheduled native trajectory without claiming the source time met."""
    if self.effective_rest_node is None:
      self.check_rest_goal()
      return
    node = self.effective_rest_node
    failure = self.project_rest_precision(node=node)
    if not failure:
      failure = self.scheduled_rest_trajectory_error(
        self.x0, self.x_sol, self.j_solution, node, self.params, tolerance=QP_FEASIBILITY_TOL,
        recovery_position_limit=self.rest_recovery_position_limit,
      )
    self.effective_rest_state = 'unsatisfied' if failure else 'satisfied'
    self.effective_rest_reason = failure
    self.diagnostic_rest_goal.update(
      state=self.rest_goal_state, reason=self.rest_goal_reason,
      nominalMaximumRestStateError=float(np.max(np.abs(self.x_sol[self.rest_goal.native_node:, 1:]))),
      effectiveState=self.effective_rest_state, effectiveReason=failure,
      effectiveMaximumRestStateError=float(np.max(np.abs(self.x_sol[node:, 1:]))),
      effectiveMinimumContinuousVelocity=self.trajectory_minimum_velocity(self.x_sol, self.j_solution),
    )
    if failure:
      self.solution_status = MPC_GOAL_UNSATISFIED
      self.rest_goal_braking_accel = float(np.min(self.params[:, 0]))
      self.failure_reason = failure
      self.previous_solution_valid = False
      self.invalidate_reference_history()
    else:
      # run() already established actual native success. Its solved trajectory,
      # full constraint checks and effective zero-speed destination are retained.
      self.rest_goal_braking_accel = None
      self.failure_reason = ''

  def update(self, radarstate, personality=log.LongitudinalPersonality.standard, *, reference=None, now_ns=None, reference_ns=None,
             reference_inputs_valid=True):
    if self.solution_status == 0:
      self._failure_log_key = None
    self._failure_diagnosed = False
    self._diagnostic_native_return = None
    self._diagnostic_problem_installed = False
    self._diagnostic_rejected_lead = None
    self._diagnostic_now_ns = now_ns
    self._diagnostic_reference_ns = reference_ns
    self.reference_support_changed = False
    self.continuation_problem = None
    # Retain the native fourteen-parameter ABI; zero selects the original
    # v_ego + 10 normalization in both generated distance expressions.
    self.params[:, COMFORT_NORMALIZER:RISK_NORMALIZER+1] = 0.0
    if hasattr(radarstate, 'as_reader'):
      radarstate = radarstate.as_reader()
    self._diagnostic_radar = radarstate
    self.diagnostic_last_obstacle_summary = None
    self.source_object_identity = None
    self.fcw_object_identity = None
    self._obstacle_identities = ()
    self._source_leads = ()
    self._lead_trajectories = ()
    self.object_influence_identities = ()
    self._influence_problem = None
    self.rest_goal = None
    self.rest_goal_state = 'none'
    self.rest_goal_reason = ''
    self.rest_goal_braking_accel = None
    self.effective_rest_node = None
    self.effective_rest_state = 'none'
    self.effective_rest_reason = ''
    self.rest_recovery_position_limit = None
    self.diagnostic_rest_goal = None
    self.solve_time = 0.0

    if not np.all(np.isfinite(self.x0)):
      self.fail(MPC_INVALID_INPUT, stage='initial_state', detail='Nonfinite initial state')
      return
    t_follow = get_T_FOLLOW(personality, v_ego=self.x0[1] if self.speed_based_following else None)
    fusion_active = bool(getattr(radarstate, 'fusionActive', False))
    if fusion_active != self._was_fusion_active:
      self.reset_fcw()
    self._was_fusion_active = fusion_active
    leads = (radarstate.leadOne, radarstate.leadTwo)
    try:
      if fusion_active:
        if any(stamp is not None and not valid_timestamp(stamp) for stamp in (now_ns, reference_ns)):
          raise ValueError('Invalid fusion clock')
        if (now_ns is not None and reference_ns is not None
            and not 0 <= now_ns - reference_ns <= REFERENCE_MAX_AGE_NS):
          raise ValueError('Fusion initial state is not current at the consumer')
      signatures = {}
      for slot, lead in enumerate(leads):
        if not lead.present:
          continue
        if (not np.all(np.isfinite([lead.dRel, lead.vLead, lead.aLeadK, lead.aLeadTau]))
            or lead.aLeadTau < 0.0):
          self._diagnostic_rejected_lead = lead
          raise ValueError('Unqualified formal lead')
        if not qualified_lead(lead, 'brake', 0.0, now_ns=now_ns):
          self._diagnostic_rejected_lead = lead
          raise ValueError('Unqualified formal lead')
        evidence = getattr(lead, 'evidence', None)
        identity = lead_identity(lead)
        if evidence is not None and evidence.version == 1 and str(evidence.source) == 'vision':
          identity = ('vision', evidence.modelMonoTime, evidence.visionIndex)
        if identity is not None:
          signature = (tuple(getattr(lead, field, None) for field in (
                         'dRel', 'yRel', 'vRel', 'vLead', 'aLeadK', 'aLeadTau', 'modelProb')),
                       str(evidence.source), str(evidence.pathState),
                       tuple(getattr(evidence, field, None) for field in (
                         'generation', 'estimateMonoTime', 'positionMonoTime', 'velocityMonoTime', 'modelMonoTime',
                         'positionValid', 'velocityValid', 'accelerationValid', 'positionStd', 'velocityStd',
                         'followQualified', 'brakeQualified', 'stopQualified', 'departureQualified', 'fcwQualified',
                         'coasting', 'pathOffset', 'pathSource', 'followGeometryMonoTime')))
          if identity in signatures and signatures[identity] != signature:
            self._diagnostic_rejected_lead = lead
            raise ValueError('Inconsistent formal lead identity views')
          signatures[identity] = signature
      self._source_leads = tuple(lead for lead in leads if lead.present)
      with np.errstate(over='ignore', invalid='ignore'):
        lead_trajectories = tuple(self.process_lead(lead, reference_ns=reference_ns) for lead in leads)
    except (AttributeError, TypeError, ValueError, FloatingPointError, OverflowError) as error:
      self.fail(MPC_INVALID_INPUT, stage='formal_lead', detail=error)
      return

    # Rest scheduling keeps the actual front geometry. Fusion continuation
    # additionally captures it with measurement deadlines; process_lead's
    # numerical distance floor cannot authorize passing through a real lead.
    self._lead_trajectories = tuple(
      np.column_stack((trajectory[:, 0] + (self.lead_initial_state(lead, reference_ns=reference_ns)[0] - trajectory[0, 0]), trajectory[:, 1]))
      for lead, trajectory in zip(leads, lead_trajectories, strict=True) if lead.present
    )
    if reference is not None:
      self.qualify_reference_support(radarstate, now_ns=now_ns, valid=reference_inputs_valid)
      if self._reference_pending_ns is not None and self._reference_pending_ns == reference_ns:
        self.reject_reference_cost_for_current_collision()
    recovery_leads = tuple(zip(self._source_leads, self._lead_trajectories, strict=True))
    if recovery_leads and all(lead.vLead >= 0.0 for lead, _ in recovery_leads):
      limit = self.x0[0] + min(trajectory[0, 0] for _, trajectory in recovery_leads) - CRASH_DISTANCE
      if limit > self.x0[0]:
        self.rest_recovery_position_limit = float(limit)
    obstacles = [trajectory[:, 0] + get_stopped_equivalence_factor(trajectory[:, 1]) for trajectory in lead_trajectories]
    sources = list(MPC_SOURCES)
    identities = [lead_identity(lead) if lead.present else None for lead in leads]
    influence_tokens = [self.lead_influence_token(lead, model_ns=getattr(radarstate, 'mdMonoTime', 0), slot=slot)
                        for slot, lead in enumerate(leads)]
    # Compact gaps still need the real lead's independent stopping evidence.
    lead_slots = [slot if lead.present and qualified_lead(lead, 'stop', now_ns=now_ns) else None
                  for slot, lead in enumerate(leads)]
    accel_min, accel_max = ACCEL_MIN, ACCEL_MAX
    ordinary_continuation = False
    if reference is not None:
      try:
        with np.errstate(over='ignore', invalid='ignore'):
          reference_obstacles, reference_sources, accel_min, accel_max = self.reference_obstacles(reference, t_follow)
      except (TypeError, ValueError, FloatingPointError) as error:
        self.fail(MPC_INVALID_INPUT, stage='reference_obstacles', detail=error)
        return
      ordinary_continuation = (
        reference.cruise_speed is None and reference.typed_model is None and not self._source_leads
        and all(values is None for values in (reference.model_position, reference.model_velocity, reference.model_acceleration))
      )
      for index, lead in enumerate(leads):
        if not lead.present:
          if ordinary_continuation:
            sources[index] = LongitudinalPlanSource.cruise
          else:
            obstacles[index] = np.full(N+1, np.inf)
      obstacles.extend(reference_obstacles)
      sources.extend(reference_sources)
      identities.extend([None] * len(reference_sources))
      influence_tokens.extend([None] * len(reference_sources))
      lead_slots.extend([None] * len(reference_sources))
    self._obstacle_identities = tuple(identities)
    x_obstacles = np.column_stack(obstacles)
    self._influence_problem = x_obstacles.copy(), tuple(influence_tokens), tuple(lead_slots), reference
    selected = int(np.argmin(x_obstacles[1 if reference is not None else 0]))
    self.source = sources[selected]
    self.source_object_identity = identities[selected]
    self.params[:,0] = accel_min
    self.params[:,1] = accel_max
    self.params[:,2] = np.min(x_obstacles, axis=1)
    self.params[:,3] = np.copy(self.a_prev)
    self.params[:,4] = t_follow
    self.params[:,5] = LEAD_DANGER_FACTOR
    try:
      self.set_reference_problem(reference, x_obstacles, sources, accel_min, t_follow, lead_slots=lead_slots)
      self._diagnostic_problem_installed = True
    except (TypeError, ValueError, FloatingPointError) as error:
      self.fail(MPC_INVALID_INPUT, stage='problem_setup', detail=error)
      return
    try:
      self.observe_obstacles(x_obstacles, sources, ordinary_continuation=ordinary_continuation)
    except Exception:
      self.diagnostic_obstacle_failures += 1
    self.continuation_problem = self._capture_continuation_problem(reference_ns=reference_ns, now_ns=now_ns,
                                                                  fusion_active=fusion_active)
    self.run()
    if self.solution_status != 0:
      return
    self.check_effective_rest_goal()
    if self.solution_status != 0:
      self.reset_fcw()
      return
    # Only leadOne participates in the original five-second FCW prediction.
    # Source qualification supplies radar authority; modelProb remains visual.
    self.update_lead_fcw(leads[0], lead_trajectories[0], now_ns=now_ns)
    self.record_object_influence()

  def observe_obstacles(self, obstacles, sources, *, ordinary_continuation=False):
    """Copy a bounded constraint summary; never rebuild a reference or solve."""
    if not self.diagnostic_obstacles_enabled:
      return
    if obstacles.shape != (N+1, len(sources)) or not 2 <= len(sources) <= 4:
      raise ValueError("Unexpected obstacle summary shape")
    selected = np.argmin(obstacles, axis=1)
    source_names = {value: name for name, value in LongitudinalPlanSource.schema.enumerants.items()}
    source_rule = 'first_minimum_by_mpc_source_order'
    if ordinary_continuation:
      source_rule += ';finite_continuation_without_cruise_speed_goal'
    self.diagnostic_last_obstacle_summary = {
      'known': True,
      'times': tuple(float(value) for value in T_IDXS),
      'sources': tuple(source_names[sources[index]] for index in selected),
      'distances': tuple(float(value) for value in self.params[:,2]),
      'sourceRule': source_rule,
      'referenceHistory': self.diagnostic_reference_history,
    }
    if self._obstacle_identities:
      self.diagnostic_last_obstacle_summary.update(
        objectIdentities=tuple(self._obstacle_identities[index] for index in selected),
        objectCount=len(self._source_leads),
      )

  @staticmethod
  def _failure_scalar(value):
    if value is None or isinstance(value, bool):
      return value
    if isinstance(value, Integral):
      return int(value)
    if isinstance(value, Real):
      return float(value) if np.isfinite(value) else None
    return value[:160] if isinstance(value, str) else None

  def set_failure_context(self, sm, now_ns):
    """Copy publication identities at consumption, without serializing inputs."""
    self._diagnostic_input_times = {}
    self._diagnostic_radar = None
    self._diagnostic_rejected_lead = None
    self._diagnostic_now_ns = None
    self._diagnostic_reference_ns = None
    self._diagnostic_native_return = None
    self._diagnostic_problem_installed = False
    self._failure_diagnosed = False
    try:
      times = sm.logMonoTime
      self._diagnostic_input_times = dict(zip(
        ('consumerMonoTime', 'carStateMonoTime', 'modelMonoTime', 'radarStateMonoTime'),
        (int(value) if isinstance(value, Integral) and not isinstance(value, bool) and 0 < value < (1 << 64) else None
         for value in (now_ns, times.get('carState'), times.get('modelV2'), times.get('radarState'))), strict=True))
      self._diagnostic_radar = sm['radarState']
    except Exception:
      self.diagnostic_failure_log_errors += 1

  def finish_failure_context(self, completed):
    try:
      if completed and self.solution_status == 0:
        self._failure_log_key = None
      elif completed and not self._failure_diagnosed:
        # Some model/rest acceptance failures do not call fail()/reset().
        self._log_failure(self.solution_status, 'planner_result', self.failure_reason)
    except Exception:
      self.diagnostic_failure_log_errors += 1

  def _failure_native_statistics(self):
    result = {'known': False, 'qpResidualsKnown': False, 'reason': 'no_current_native_return'}
    if self._diagnostic_native_return is None:
      return result
    try:
      # SQP_RTI statistics rows are iteration, QP status, QP iterations, then
      # four optional QP residuals. get_residuals() recomputes RTI residuals;
      # never call it from logging or request an unsupported native getter.
      stats = self.solver.get_stats('statistics')
      if not isinstance(stats, np.ndarray) or stats.ndim != 2 or stats.shape[0] not in (3, 7) or not 0 < stats.shape[1] <= 32:
        result['reason'] = 'unknown_statistics_layout'
        return result
      values = stats[:3, -1]
      if not np.all(np.isfinite(values)) or np.any(values != np.trunc(values)):
        result['reason'] = 'invalid_statistics'
        return result
      result.update(known=True, reason='', iteration=int(values[0]), qpStatus=int(values[1]), qpIterations=int(values[2]))
      if stats.shape[0] == 7:
        result.update(qpResidualsKnown=bool(np.all(np.isfinite(stats[3:7, -1]))),
                      qpResiduals=[self._failure_scalar(value) for value in stats[3:7, -1]])
    except Exception as error:
      result['reason'] = 'statistics_unavailable'
      result['exception'] = type(error).__name__[:80]
    return result

  def _failure_radar_summary(self):
    radar = self._diagnostic_radar
    if radar is None:
      return {'known': False}
    selected = []
    candidates = [self._diagnostic_rejected_lead, radar.leadOne, radar.leadTwo]
    for lead in candidates:
      if lead is None or not getattr(lead, 'present', False) or any(lead is previous for previous in selected):
        continue
      selected.append(lead)
      if len(selected) == 2:
        break
    summaries = []
    for lead in selected:
      summary = {field: self._failure_scalar(getattr(lead, field, None)) for field in ('dRel', 'vRel', 'vLead', 'aLeadK')}
      evidence = getattr(lead, 'evidence', None)
      summary.update({field: self._failure_scalar(getattr(evidence, field, None)) for field in (
        'sessionId', 'trackId', 'generation', 'modelMonoTime', 'estimateMonoTime', 'positionMonoTime', 'velocityMonoTime',
        'brakeQualified', 'coasting')})
      summary.update(source=str(getattr(evidence, 'source', 'unknown'))[:40],
                     pathState=str(getattr(evidence, 'pathState', 'unknown'))[:40],
                     rejected=lead is self._diagnostic_rejected_lead)
      summaries.append(summary)
    return {'known': True, 'fusionActive': bool(getattr(radar, 'fusionActive', False)),
            'modelMonoTime': self._failure_scalar(getattr(radar, 'mdMonoTime', None)),
            'objectCount': sum(bool(lead.present) for lead in (radar.leadOne, radar.leadTwo)),
            'sampleLimit': 2, 'objects': summaries}

  def _failure_constraint_summary(self):
    result = {'problemInstalled': self._diagnostic_problem_installed, 'restGoalState': self.rest_goal_state,
              'restGoalNode': self._failure_scalar(getattr(self.rest_goal, 'native_node', None)),
              'effectiveRestNode': self._failure_scalar(self.effective_rest_node),
              'intervalRestTime': self._failure_scalar(getattr(self.interval_rest, 'stop_time', None)),
              'nativeFirstMotionDuration': self._failure_scalar(self._native_first_interval)}
    if not self._diagnostic_problem_installed:
      return result  # A partly constructed problem must not describe old rows as current.
    stages = [stage for stage, (C, D, _, _) in enumerate(self._rest_constraint_buffers) if np.any(C) or np.any(D)]
    sampled_stages = sorted({0, *stages[:2]})
    result.update(restConstraintStages=stages,
                  parameterRanges=[[self._failure_scalar(np.min(self.params[:, column])),
                                    self._failure_scalar(np.max(self.params[:, column]))] for column in range(3)],
                  # These are the exact copied native C/D/lg/ug buffers, limited
                  # to stage zero and the first two active rest stages.
                  hardRows=[{'stage': stage,
                             'C': [[self._failure_scalar(value) for value in row] for row in self._rest_constraint_buffers[stage][0]],
                             'D': [[self._failure_scalar(value) for value in row] for row in self._rest_constraint_buffers[stage][1]],
                             'lower': [self._failure_scalar(value) for value in self._rest_constraint_buffers[stage][2]],
                             'upper': [self._failure_scalar(value) for value in self._rest_constraint_buffers[stage][3]]}
                            for stage in sampled_stages])
    if self.diagnostic_reference_bounds is not None:
      result['stateBoundRanges'] = [
        [[self._failure_scalar(value) for value in summary]
         for summary in (values[1:].min(axis=0), values[1:].max(axis=0))] for values in self.diagnostic_reference_bounds]
    return result

  def _log_failure(self, status, stage, detail=''):
    """Best-effort, bounded evidence from the failed problem, before reset."""
    try:
      self._failure_diagnosed = True
      reason = str(detail)[:192]
      exception = type(detail).__name__[:80] if isinstance(detail, Exception) else ''
      key = (int(status), stage, exception, reason)
      timestamp = time.monotonic()
      if not np.isfinite(timestamp) or timestamp < 0.0:
        self._failure_log_suppressed += 1
        return
      if key == self._failure_log_key and timestamp < self._failure_log_time + 5.0:
        self._failure_log_suppressed += 1
        return
      # One budget spans successes, resets and all failure categories. The
      # initial burst preserves new-fault evidence without allowing alternating
      # faults to serialize full payloads every planner tick. Integer credits
      # avoid accumulating floating-point error at the five-second boundary;
      # a regressed clock cannot refill or move the high-water mark backwards.
      timestamp_ns = int(timestamp * 1e9)
      refill_ns = self._failure_log_refill_ns
      if refill_ns is None:
        refill_ns = timestamp_ns
      self._failure_log_credit_ns = min(15_000_000_000, self._failure_log_credit_ns + max(0, timestamp_ns - refill_ns))
      self._failure_log_refill_ns = max(refill_ns, timestamp_ns)
      if self._failure_log_credit_ns < 5_000_000_000:
        self._failure_log_suppressed += 1
        return
      self._failure_log_credit_ns -= 5_000_000_000
      suppressed = self._failure_log_suppressed
      # Advance before payload/getter/logger work, including when it fails.
      self._failure_log_key, self._failure_log_time, self._failure_log_suppressed = key, timestamp, 0
      payload = dict(self._diagnostic_input_times, status=int(status), stage=stage, reason=reason, exception=exception,
                     mpcConsumerMonoTime=self._failure_scalar(self._diagnostic_now_ns),
                     referenceMonoTime=self._failure_scalar(self._diagnostic_reference_ns),
                     referenceHistory=self.diagnostic_reference_history,
                     nativeReturn=self._diagnostic_native_return, x0=[self._failure_scalar(value) for value in self.x0],
                     solveTime=self._failure_scalar(self.solve_time), suppressedCount=suppressed)
      for field, collect in (('radar', self._failure_radar_summary), ('constraints', self._failure_constraint_summary),
                             ('native', self._failure_native_statistics)):
        try:
          payload[field] = collect()
        except Exception as error:
          payload[field] = {'known': False, 'exception': type(error).__name__[:80]}
      cloudlog.event('long_mpc_failure_v1', error=True, **payload)
    except Exception:
      self.diagnostic_failure_log_errors += 1

  def fail(self, status, *, stage='unspecified', detail=''):
    self._log_failure(status, stage, detail)
    continuation_problem = self.continuation_problem if status == 4 else None
    failed_goal = self.rest_goal
    effective_node = self.effective_rest_node
    goal_diagnostic = self.diagnostic_rest_goal
    solve_time = self.solve_time
    last_cloudlog_t = self.last_cloudlog_t
    timestamp = time.monotonic()
    if timestamp > last_cloudlog_t + 5.0:
      last_cloudlog_t = timestamp
      try:
        cloudlog.warning(f"Long mpc reset, solution_status: {status}")
      except Exception:
        self.diagnostic_failure_log_errors += 1
    self.reset()
    self.continuation_problem = continuation_problem
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
      if effective_node is not None:
        self.effective_rest_node = effective_node
        self.effective_rest_state, self.effective_rest_reason = 'solver_failure', self.failure_reason
        self.diagnostic_rest_goal.update(effectiveState=self.effective_rest_state, effectiveReason=self.failure_reason)
    self.last_cloudlog_t = last_cloudlog_t
    self.solve_time = solve_time if np.isfinite(solve_time) else 0.0
    self.source = LongitudinalPlanSource.cruise

  def run(self):
    self.params[:, COMFORT_NORMALIZER:RISK_NORMALIZER+1] = 0.0
    self._diagnostic_native_return = None
    try:
      self.diagnostic_last_x0 = tuple(float(value) for value in self.x0)
      if self.diagnostic_details_enabled:
        self.diagnostic_last_parameters = self.params.tolist()
        self.diagnostic_last_yref = self.yref.tolist()
        self.diagnostic_last_a_prev = self.a_prev.tolist()
    except Exception:
      pass
    if not np.all(np.isfinite(self.params)) or not np.all(np.isfinite(self.yref)) or not np.all(np.isfinite(self.x0)):
      self.fail(MPC_INVALID_INPUT, stage='pre_solve', detail='Nonfinite parameters, reference or initial state')
      return
    if self._costs_pending:
      # Direct run callers retain the base-cost contract. In normal update,
      # the complete reference already installed its costs exactly once.
      self.install_cost_weights(self.stage_cost_weights())
    for i in range(N+1):
      self.solver.set(i, 'p', self.params[i])
    self.solver.constraints_set(0, "lbx", self.x0)
    self.solver.constraints_set(0, "ubx", self.x0)

    solve_started_t = time.monotonic()
    stage = 'native_interval_update'
    try:
      self.diagnostic_last_solver_return = None
      duration = self.interval_rest.stop_time if self.interval_rest is not None else float(T_IDXS[1])
      if duration != self._native_first_interval:
        if self.solver.custom_update(np.array([1.0, duration])) != 0:
          raise RuntimeError('MPC artifact does not support interval-rest integration')
        self._native_first_interval = duration
      stage = 'native_solve'
      self.diagnostic_solve_calls += 1
      self.solution_status = int(self.solver.solve())
      self.diagnostic_last_solver_return = self.solution_status
      self._diagnostic_native_return = self.solution_status
      self.last_solver_status = self.solution_status
      if self.solution_status != 0:
        self.solve_time = time.monotonic() - solve_started_t
        self.fail(self.solution_status, stage=stage, detail='Native solver returned failure')
        return
      stage = 'native_readback'
      self.solve_time = float(self.solver.get_stats('time_tot')[0])
      for i in range(N+1):
        self.x_sol[i] = self.solver.get(i, 'x')
      for i in range(N):
        self.u_sol[i] = self.solver.get(i, 'u')
    except RuntimeError as error:
      self.solve_time = time.monotonic() - solve_started_t
      self.fail(MPC_SOLVER_ERROR, stage=stage, detail=error)
      return
    if not np.all(np.isfinite(self.x_sol)) or not np.all(np.isfinite(self.u_sol)):
      self.fail(MPC_INVALID_SOLUTION, stage='post_solve', detail='Nonfinite native trajectory')
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
  # This vendored template calls the custom hook without returning its int.
  # Propagate it so Python can reject an incompatible or failed native update.
  solver_source = os.path.join(EXPORT_DIR, 'acados_solver_long.c')
  with open(solver_source) as source:
    generated = source.read()
  missing_return = '    custom_update_function(capsule, data, data_len);'
  if generated.count(missing_return) != 1:
    raise RuntimeError('Unexpected acados custom-update template; verify status propagation')
  with open(solver_source, 'w') as source:
    source.write(generated.replace(missing_return, '    return custom_update_function(capsule, data, data_len);'))
