#!/usr/bin/env python3
import os
import time
from dataclasses import dataclass, replace
from enum import Enum
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
  MAX_CONTROL_OBJECTS, MAX_ESTIMATE_AGE_NS, MAX_MEASUREMENT_AGE_NS, MAX_PREDICTION_AGE_NS,
  control_leads, lead_identity, qualified_lead, valid_conflict_support,
)
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.continuation import (
  ContinuationDonor, ContinuationProblem, ContinuationResult, build_candidate, frozen_array, valid_timestamp,
)
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_MAX_AGE_NS

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
MPC_OCCUPANCY_UNSATISFIED = -6
REFERENCE_BOUND = 1e4
REST_STATE_TOL = 1e-3
# Keep numerical constraint residuals separate from the existing geometric
# clearance. The exact post-solve check still requires CRASH_DISTANCE.
OCCUPANCY_POSITION_BUFFER = 1e-3
# Unit-specific admission for re-integrating unchanged native jerk, not final
# constraint tolerances. These bound each defect, cumulative node corrections,
# AND every point between nodes. The 0.1 mm position budget is one tenth of the
# separate 1 mm geometry buffer; final physical clearance is still checked.
OCCUPANCY_POSITION_REINTEGRATION_LIMIT = 1e-4  # m
OCCUPANCY_VELOCITY_REINTEGRATION_LIMIT = 1e-5  # m/s
OCCUPANCY_ACCELERATION_REINTEGRATION_LIMIT = 1e-6  # m/s^2
# Only a separately proven, fresh near-term physical conflict can retain
# mitigation authority through a solver failure. This is not an FCW horizon.
FINITE_RISK_HORIZON = 1.0
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
class LeadProjection:
  lead: object
  identity: tuple | None
  slot: int | None
  trajectory: np.ndarray
  raw_trajectory: np.ndarray
  occupied: np.ndarray
  conflict_start: float
  conflict_end: float
  persistent: bool
  pass_before: bool = False
  conflict_support_version: int = 0
  conflict_supported_end: float = 0.0

  @property
  def hard_conflict_end(self):
    return (self.conflict_end if self.conflict_support_version == 0
            else min(self.conflict_end, self.conflict_supported_end))


@dataclass(frozen=True)
class OccupancyBound:
  duration: float
  position_limit: float
  identities: tuple
  position_lower_limit: float = -REFERENCE_BOUND


@dataclass(frozen=True)
class FiniteRiskEvidence:
  identity: tuple[str, int]
  source: str
  consumed_ns: int
  reference_ns: int
  estimate_ns: int
  position_ns: int
  velocity_ns: int
  model_ns: int
  conflict_start_s: float
  conflict_end_s: float
  proof_time_s: float
  entry_object_position: float
  entry_ego_upper_position: float
  proof_object_position: float
  proof_ego_lower_position: float
  braking_accel: float = ACCEL_MIN


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
  # Passing changes which side of a timed obstacle is safe. It is admitted
  # only by a caller that has brought independent final braking limits into
  # this same reference; a later arbitrary min(accel) is not conservative.
  allow_passing: bool = False
  # These explicit targets apply only to real leads. Virtual cruise/model
  # obstacles retain STOP_DISTANCE, as do callers that omit these fields.
  # Qualification for a compact stationary gap belongs to the producer.
  comfort_stop_distance: float = STOP_DISTANCE
  lead_risk_stop_distances: tuple[float, float] = (STOP_DISTANCE, STOP_DISTANCE)

  def __post_init__(self):
    if type(self.allow_passing) is not bool:
      raise ValueError('Invalid finite occupancy passing authorization')
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
  # Zero preserves the original velocity-dependent residual. An unknown
  # occupancy tail uses this tick's fixed initial-speed scale only when it
  # actually supplies that residual's separately selected obstacle minimum.
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
    self.diagnostic_occupancy_refinement = None
    # UNIFIED dates its reference with the consumer clock. Keep the clock high
    # water across solver resets; reset buffers must not admit a replayed epoch.
    self._reference_clock_ns = 0
    self.reset()
    self.source = LongitudinalPlanSource.cruise

  def reset(self):
    clear_occupancy_bounds = getattr(self, '_native_occupancy_bounds_active', False)
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
    if clear_occupancy_bounds:
      # solver.reset() clears iterates, not the installed model constraints.
      # Ordinary update rewrites every row; direct run after failure/reset
      # must also be unable to inherit an old crossing or physical state box.
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
    self.crash_cnt = 0.0
    self._fusion_fcw_counts = {}
    self._fusion_fcw_epochs = {}
    self._fusion_vision_fcw_tracks = {}
    self._fusion_vision_fcw_sequence = 0
    self._fusion_follow_geometry = {}
    self._was_fusion_active = False
    self.source_object_identity = None
    self.fcw_object_identity = None
    self._obstacle_identities = ()
    self._fusion_projections = ()
    self._occupancy_bounds = (None,) * N
    self._occupancy_limit_cache = {}
    # Tracks installed finite-object physical rows/boxes, including a soft
    # occupancy tail with no supported position row. reset() must clear both.
    self._native_occupancy_bounds_active = False
    self.diagnostic_occupancy_bounds = ()
    self.diagnostic_occupancy_projection = None
    self._passing_witness = None
    self._passing_speed_limit = None
    self.diagnostic_passing = None
    self.object_influence_identities = ()
    self._influence_problem = None
    self._finite_risk_candidates = ()
    self.finite_risk_evidence = ()
    self.finite_risk_braking_accel = None
    self.solution_status = 0
    self.last_solver_status = 0
    self.failure_reason = ''
    self.rest_goal = None
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
    self._occupancy_limit_cache = {}
    self._reference_epoch_ns = None
    self._reference_acceleration = None
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

  def _capture_continuation_problem(self, *, reference_ns, now_ns):
    """Keep this tick's installed problem before fail/reset can erase it."""
    if (not all(valid_timestamp(stamp) for stamp in (reference_ns, now_ns))
        or not 0 <= now_ns - reference_ns <= REFERENCE_MAX_AGE_NS
        or self.diagnostic_reference_bounds is None):
      return None
    expiry = int(reference_ns) + REFERENCE_MAX_AGE_NS
    for projection in self._fusion_projections:
      evidence = projection.lead.evidence
      age = MAX_PREDICTION_AGE_NS if evidence.coasting else MAX_MEASUREMENT_AGE_NS
      expiry = min(expiry, int(evidence.estimateMonoTime) + MAX_ESTIMATE_AGE_NS,
                   int(evidence.positionMonoTime) + age, int(evidence.velocityMonoTime) + age,
                   int(evidence.modelMonoTime) + age)
    # No message reader survives in the numerical certificate. Every array
    # below is a private immutable copy, including the raw geometric profile.
    projections = tuple(replace(projection, lead=None,
                                trajectory=frozen_array(projection.trajectory),
                                raw_trajectory=frozen_array(projection.raw_trajectory),
                                occupied=frozen_array(projection.occupied, dtype=bool))
                        for projection in self._fusion_projections)
    reference = self._influence_problem[3] if self._influence_problem is not None else None
    typed = reference.typed_model if reference is not None else None
    forbidden = (self.rest_goal is not None or self.effective_rest_node is not None or self._passing_speed_limit is not None
                 or any(projection.pass_before for projection in projections)
                 or typed is not None and (typed.no_forward or typed.has_rest_intent))
    lower, upper = self.diagnostic_reference_bounds
    return ContinuationProblem(
      int(reference_ns), int(now_ns), int(expiry), frozen_array(self.x0), frozen_array(self.params),
      frozen_array(lower), frozen_array(upper),
      tuple(tuple(frozen_array(value) for value in row) for row in self._rest_constraint_buffers),
      projections, bool(forbidden),
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
    if problem.stop_or_passing:
      return 'stop_or_passing'
    parameters = problem.parameters
    arrays = (problem.x0, states, jerks, parameters, problem.state_lower, problem.state_upper)
    if (problem.x0.shape != (X_DIM,) or states.shape != (N+1, X_DIM) or jerks.shape != (N,)
        or parameters.shape != (N+1, PARAM_DIM) or problem.state_lower.shape != (N+1, 2)
        or problem.state_upper.shape != (N+1, 2) or len(problem.linear_constraints) != N
        or not all(np.all(np.isfinite(value)) for value in arrays)):
      return 'invalid_trajectory_or_problem'
    if (np.any(parameters[:, 0] > parameters[:, 1]) or np.any(parameters[:, 5] <= 0.0) or np.any(parameters[:, 4] < 0.0)
        or np.any(parameters[:, [COMFORT_NORMALIZER, RISK_NORMALIZER]] < 0.0)
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
    # The current folded minimum includes persistent, ordinary and extra
    # leads plus current cruise/model risk references. Its compact distance
    # and new normalizer are exactly those installed for this failed tick.
    risk_margin = parameters[:, 2] - states[:, 0] - parameters[:, 5] * (
      get_safe_obstacle_distance(states[:, 1], parameters[:, 4]) - STOP_DISTANCE + parameters[:, RISK_STOP_DISTANCE])
    denominator = np.where(parameters[:, RISK_NORMALIZER] > 0.0,
                            parameters[:, RISK_NORMALIZER], states[:, 1] + 10.0)
    if np.any(denominator <= 0.0) or np.any(risk_margin / denominator < -tolerance):
      return 'current_risk_distance_deficit'
    for projection in problem.projections:
      if not projection.occupied.any():
        continue
      minimum, _ = cls.occupancy_clearance_extrema(projection, states, jerks, supported_only=not projection.persistent)
      if minimum < CRASH_DISTANCE:
        return 'current_object_clearance'
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
    if self.solution_status not in (4, MPC_OCCUPANCY_UNSATISFIED):
      return ContinuationResult(False, 'unsupported_failure_status')
    return build_candidate(donor, self.continuation_problem, now_ns=now_ns, times=T_IDXS,
                            integrate=self.integrate_jerk, validate=self._continuation_trajectory_error)

  def discard_prepared_reference_cost(self):
    """Reject the preceding semantics, retaining this qualified pending epoch.

    prepare_reference_history already removed the preceding stored reference.
    A new rest goal can replace its cost without losing the new successful
    trajectory that the next frame will be allowed to smooth against.
    """
    self.a_prev[:] = 0.0

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
    self._occupancy_limit_cache = {}
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

  @staticmethod
  def occupancy_nodes(start, end):
    """Cover a finite occupancy interval, including a short inter-node event.

    The native obstacle rows exist only at shooting nodes. Both endpoints of
    each intersected interval are needed: constraining only its first node
    lets the solver accelerate through a crossing before the next row. This
    bounded outer cover ends at the first node on/after the real end; it is
    never a persistent lead. Consumption expiry and FCW use the actual window,
    not this conservative discretization. The native rows remain soft costs,
    so their coverage alone is not a continuous collision-freedom proof.
    """
    occupied = (T_IDXS >= start) & (T_IDXS <= end)
    intersects = (T_IDXS[:-1] < end) & (T_IDXS[1:] > start)
    occupied[:-1] |= intersects
    occupied[1:] |= intersects
    return occupied

  @staticmethod
  def ego_progress_bound(speed, acceleration, timestamp):
    duration = min(timestamp, speed / -acceleration) if acceleration < 0.0 else timestamp
    return speed * duration + 0.5 * acceleration * duration**2

  def project_fusion_lead(self, lead, *, slot, reference_ns, now_ns=None):
    evidence = lead.evidence
    if not valid_conflict_support(evidence):
      raise ValueError('Invalid fusion occupancy support')
    elapsed = 0.0 if reference_ns is None else (reference_ns - evidence.estimateMonoTime) / 1e9
    if not -REFERENCE_MAX_AGE_NS / 1e9 <= elapsed <= MAX_PREDICTION_AGE_NS / 1e9:
      raise ValueError('Fusion estimate cannot align with the initial state')
    a_lead, tau = float(lead.aLeadK), float(lead.aLeadTau)
    if tau < 0.0:
      raise ValueError('Invalid lead acceleration decay')
    # d/v/a share estimateMonoTime. Project this complete hypothesis and
    # subtract the signed ego displacement to the carState initial epoch.
    # A fresh camera EOF may follow carState: bounded reverse alignment keeps
    # that actual epoch instead of replacing it with the consumer clock. The
    # short interval uses a constant-acceleration ego approximation.
    decayed_accel = a_lead * np.exp(-tau * elapsed**2 / 2.0)
    initial_velocity = float(lead.vLead) + 0.5 * (a_lead + decayed_accel) * elapsed
    if lead.vLead >= 0.0:
      initial_velocity = max(0.0, initial_velocity)
    advance = float(lead.vLead) * elapsed + (2.0 * a_lead + decayed_accel) * elapsed**2 / 6.0
    ego_advance = self.x0[1] * elapsed - 0.5 * self.x0[2] * elapsed**2
    initial_distance = float(lead.dRel) + advance - ego_advance
    acceleration = np.clip(a_lead, -10.0, 5.0) * np.exp(-tau * (T_IDXS + elapsed)**2 / 2.0)
    velocity = initial_velocity + np.cumsum(T_DIFFS * acceleration)
    # Keep the existing forward stopping and clearly opposing predictions.
    # Within the producer's [-2, 0] non-opposing direction interval, blend
    # only the forecast's reverse-acceleration component continuously. This
    # is an engineering motion-model policy: measured d/v/a and qualification
    # remain intact, and a small negative velocity is never declared noise
    # or replaced by a stationary target. Positive acceleration is unchanged.
    if initial_velocity >= 0.0:
      velocity = np.maximum(velocity, 0.0)
    elif initial_velocity > -2.0 and a_lead < 0.0:
      velocity_base = np.maximum(velocity, initial_velocity)
      reverse_weight = -initial_velocity / 2.0  # This branch has 0 < weight < 1.
      velocity = velocity_base + reverse_weight * (velocity - velocity_base)
    raw_position = initial_distance + np.cumsum(T_DIFFS * velocity)
    raw_trajectory = np.column_stack((raw_position, velocity))
    nonnegative_velocity = max(0.0, initial_velocity)
    minimum_distance = (MIN_X_LEAD_FACTOR * (self.x0[1] + nonnegative_velocity)
                        * (self.x0[1] - nonnegative_velocity) / (-ACCEL_MIN * 2.0))
    trajectory = raw_trajectory.copy()
    trajectory[:, 0] += max(0.0, minimum_distance - initial_distance)

    start, end = float(evidence.conflictStart) - elapsed, float(evidence.conflictEnd) - elapsed
    persistent = (str(evidence.pathState) == 'inPath' and evidence.followQualified
                  and evidence.conflictStart == 0.0 and evidence.conflictEnd >= MAX_T)
    # A continuing in-lane track is predicted over this complete horizon.
    # Finite crossings/coasts keep their original end rather than renewing it.
    if persistent:
      start, end = 0.0, MAX_T
    start = max(0.0, start)
    support_version = getattr(evidence, 'conflictSupportVersion', 0)
    supported_end = end
    if support_version == 1:
      supported_end = max(0.0, min(end, float(evidence.conflictSupportedEnd) - elapsed))
      consumed_ns = (max(evidence.estimateMonoTime, evidence.modelMonoTime, evidence.positionMonoTime,
                         evidence.velocityMonoTime) if now_ns is None else now_ns)
      # An older carState origin cannot revive hard prediction that has
      # already exhausted its support by the real consumption time. Keep
      # the complete ordinary obstacle envelope below, including its tail.
      if (consumed_ns - evidence.estimateMonoTime) / 1e9 >= evidence.conflictSupportedEnd:
        supported_end = 0.0
    occupied = self.occupancy_nodes(start, end) if end > start else np.zeros(N+1, dtype=bool)
    if not persistent and np.any(occupied):
      # Discard only geometrically impossible conflicts under the complete
      # acceleration range, not one guessed ego arrival time. A vehicle that
      # could still reach the crossing is retained for the joint solve.
      begin, finish = min(start, MAX_T), min(end, MAX_T)
      start_x, end_x = np.interp([begin, finish], T_IDXS, raw_position)
      occupied_positions = np.concatenate(([start_x], raw_position[(T_IDXS >= begin) & (T_IDXS <= finish)], [end_x]))
      furthest_ego = self.ego_progress_bound(self.x0[1], ACCEL_MAX, finish)
      nearest_ego = self.ego_progress_bound(self.x0[1], ACCEL_MIN, begin)
      if (furthest_ego + STOP_DISTANCE < np.min(occupied_positions)
          or nearest_ego > np.max(occupied_positions) + STOP_DISTANCE):
        occupied[:] = False
    if not np.all(np.isfinite([trajectory, raw_trajectory])):
      raise ValueError('Invalid fusion prediction')
    return LeadProjection(lead, lead_identity(lead), slot, trajectory, raw_trajectory, occupied, start, end, persistent,
                          conflict_support_version=support_version, conflict_supported_end=supported_end)

  def fusion_lead_projections(self, radarstate, *, now_ns, reference_ns):
    raw_objects = radarstate.radarObjects
    if len(raw_objects) > MAX_CONTROL_OBJECTS:
      raise ValueError('Too many fusion objects')
    # The two display slots are views of the object set. A repeated identity
    # with different geometry cannot choose its preferred copy by list order.
    signatures = {}
    for lead in (*raw_objects, radarstate.leadOne, radarstate.leadTwo):
      if not lead.present:
        continue
      evidence = lead.evidence
      identity = lead_identity(lead)
      key = ('track', *identity) if identity is not None else ('vision', evidence.modelMonoTime, evidence.visionIndex)
      signature = (lead.dRel, lead.yRel, lead.vRel, lead.vLead, lead.aLeadK, lead.aLeadTau,
                   evidence.version, str(evidence.source), evidence.estimateMonoTime, evidence.positionMonoTime,
                   evidence.velocityMonoTime, evidence.conflictStart, evidence.conflictEnd, evidence.brakeQualified,
                   evidence.followQualified, evidence.stopQualified, evidence.fcwQualified,
                   evidence.generation, evidence.coasting, str(evidence.pathState), evidence.pathSource, evidence.lateralSpeed,
                   getattr(evidence, 'physicalConflictValid', False), getattr(evidence, 'physicalConflictStart', 0.0),
                   getattr(evidence, 'physicalConflictEnd', 0.0), getattr(evidence, 'followGeometryMonoTime', 0),
                   getattr(evidence, 'conflictSupportVersion', 0), getattr(evidence, 'conflictSupportedEnd', 0.0))
      if key in signatures and signatures[key] != signature:
        raise ValueError('Inconsistent fusion identity views')
      signatures[key] = signature
    candidates = control_leads(radarstate)
    if len(candidates) > MAX_CONTROL_OBJECTS:
      raise ValueError('Too many independent fusion objects')
    slots = (radarstate.leadOne, radarstate.leadTwo)
    slot_identities = tuple(lead_identity(lead) for lead in slots)
    result = []
    next_follow_geometry = {}
    for lead in candidates:
      if not np.all(np.isfinite([lead.dRel, lead.yRel, lead.vRel, lead.vLead, lead.aLeadK, lead.aLeadTau])):
        raise ValueError('Nonfinite fusion object')
      evidence = lead.evidence
      if (evidence.version != 1 or not np.all(np.isfinite([evidence.conflictStart, evidence.conflictEnd]))
          or not 0.0 <= evidence.conflictStart < evidence.conflictEnd <= MAX_T):
        raise ValueError('Invalid fusion object contract')
      if str(evidence.pathState) == 'edge' or getattr(evidence, 'cautionQualified', False):
        if not qualified_lead(lead, 'caution', 0.0, now_ns=now_ns, require_active_window=False):
          raise ValueError('Invalid edge caution object')
        # This evidence can bound a positive final request in UNIFIED, but
        # grants no MPC obstacle, stop, following or FCW authority.
        continue
      if not qualified_lead(lead, 'brake', 0.0, now_ns=now_ns, require_active_window=False):
        if evidence.brakeQualified:
          raise ValueError('Unqualified admitted fusion object')
        continue
      # A healthy finite occupancy can expire between its source frame and
      # this consumption. Remove only that constraint; other leads and model
      # references remain valid. In particular, an older carState origin
      # must not reintroduce a crossing that has already cleared by now.
      consumed_ns = (max(evidence.estimateMonoTime, evidence.modelMonoTime, evidence.positionMonoTime,
                         evidence.velocityMonoTime) if now_ns is None else now_ns)
      if (consumed_ns - evidence.estimateMonoTime) / 1e9 >= evidence.conflictEnd:
        continue
      identity = lead_identity(lead)
      slot = slot_identities.index(identity) if identity is not None and identity in slot_identities else None
      if slot is None and str(evidence.source) == 'vision':
        for index, display in enumerate(slots):
          display_evidence = getattr(display, 'evidence', None)
          if (display.present and display_evidence is not None and display_evidence.version == 1
              and str(display_evidence.source) == 'vision'
              and display_evidence.modelMonoTime == evidence.modelMonoTime
              and display_evidence.visionIndex == evidence.visionIndex):
            slot = index
            break
      projection = self.project_fusion_lead(lead, slot=slot, reference_ns=reference_ns, now_ns=now_ns)
      previous = self._fusion_follow_geometry.get(identity)
      # Remember an observed following geometry, never another purpose's
      # authority. Missing/OUT/finite/changed identities disappear from the
      # next map; invalid inputs and solver resets clear it through reset().
      explicit_clock = now_ns is not None and reference_ns is not None
      measured = (evidence.generation, evidence.positionMonoTime, evidence.velocityMonoTime)
      geometry_ns = getattr(evidence, 'followGeometryMonoTime', 0)
      geometry_known = (isinstance(geometry_ns, Integral) and not isinstance(geometry_ns, bool)
                        and 0 < geometry_ns <= evidence.estimateMonoTime)
      ordered = previous is None or (consumed_ns >= previous[4] and evidence.estimateMonoTime >= previous[3]
                                     and evidence.positionMonoTime >= previous[1] and evidence.velocityMonoTime >= previous[2])
      if (explicit_clock and identity is not None and ordered and projection.persistent and not evidence.coasting
          and str(evidence.source) in ('radar', 'fused') and qualified_lead(lead, 'follow', 0.0, now_ns=now_ns)):
        next_follow_geometry[identity] = (*measured, evidence.estimateMonoTime, consumed_ns,
                                          projection.conflict_support_version, getattr(evidence, 'conflictSupportedEnd', 0.0))
      elif (explicit_clock and identity is not None and previous is not None and measured[0] == previous[0] and ordered
            and projection.conflict_support_version == previous[5]
            and geometry_known and geometry_ns >= previous[3]
            and 0 <= evidence.estimateMonoTime - previous[3] <= MAX_PREDICTION_AGE_NS
            and 0 <= consumed_ns - previous[3] <= MAX_PREDICTION_AGE_NS
            and str(evidence.source) == 'predicted' and evidence.coasting
            and str(evidence.pathState) == 'inPath' and evidence.conflictStart == 0.0 and evidence.conflictEnd == MAX_T):
        # HORIZON is the producer's "no known exit" sentinel. It may remain
        # 10 s on a coast; retain the original fresh absolute horizon rather
        # than extending it at each estimate. A newly finite exit never uses
        # this branch. The input keeps every brake-only qualification flag.
        # A fresh publication can be superseded by a coast before this planner
        # reads either. Newer same-generation measurements may therefore
        # advance the source watermark when the producer certifies its latest
        # fresh geometry too, but never advance this observed fresh horizon.
        end = min(projection.conflict_end, MAX_T, MAX_T - (reference_ns - previous[3]) / 1e9)
        if end > projection.conflict_start:
          supported_end = projection.conflict_supported_end
          if projection.conflict_support_version == 1:
            supported_end = max(0.0, min(supported_end, previous[6] - (reference_ns - previous[3]) / 1e9))
          projection = replace(projection, persistent=True, conflict_end=end,
                               conflict_supported_end=supported_end,
                               occupied=self.occupancy_nodes(projection.conflict_start, end))
          next_follow_geometry[identity] = (*measured, previous[3], consumed_ns, previous[5], previous[6])
      result.append(projection)
    self._fusion_follow_geometry = next_follow_geometry
    return tuple(result)

  @classmethod
  def occupancy_clearance_extrema(cls, projection, states, jerks, *, horizon=MAX_T, supported_only=False):
    """Exact extrema of raw lead-minus-ego geometry over the real window."""
    conflict_end = projection.hard_conflict_end if supported_only else projection.conflict_end
    if not projection.occupied.any() or conflict_end <= projection.conflict_start:
      return float('inf'), -float('inf')
    minimum, maximum = float('inf'), -float('inf')
    for stage in range(N):
      left = float(T_IDXS[stage])
      start = max(left, projection.conflict_start)
      # Stage times are increasing. Keep the point at a closed occupancy
      # endpoint, but no later stage can contribute beyond the real window.
      if start >= horizon or left > conflict_end:
        break
      end = min(float(T_IDXS[stage+1]), conflict_end, horizon)
      if end < start:
        continue
      duration = T_IDXS[stage+1] - T_IDXS[stage]
      lead_position = projection.raw_trajectory[stage, 0]
      lead_velocity = (projection.raw_trajectory[stage+1, 0] - lead_position) / duration
      state, jerk = states[stage], jerks[stage]
      position, velocity, acceleration = state
      lo, hi = start - T_IDXS[stage], end - T_IDXS[stage]
      times = [lo, hi]
      # Relative velocity is quadratic on a constant-jerk ego segment. Its
      # roots contain every interior clearance extremum of this interpolation.
      if abs(jerk) < 1e-12:
        if abs(acceleration) > 1e-12:
          times.append((lead_velocity - velocity) / acceleration)
      else:
        discriminant = acceleration**2 + 2.0 * jerk * (lead_velocity - velocity)
        if discriminant >= 0.0:
          root = np.sqrt(discriminant)
          times.extend(((-acceleration - root) / jerk, (-acceleration + root) / jerk))
      for timestamp in times:
        if lo <= timestamp <= hi:
          # Keep integrate_jerk's position arithmetic and its operation order,
          # without allocating the unused velocity/acceleration result.
          next_acceleration = acceleration + jerk * timestamp
          ego_position = position + velocity * timestamp + (2.0 * acceleration + next_acceleration) * timestamp**2 / 6.0
          clearance = lead_position + lead_velocity * timestamp - ego_position
          minimum, maximum = min(minimum, float(clearance)), max(maximum, float(clearance))
    return minimum, maximum

  @classmethod
  def collision_in_occupancy(cls, projection, states, jerks):
    """FCW uses the actual interval and the selected passing/yielding side."""
    minimum, maximum = cls.occupancy_clearance_extrema(projection, states, jerks, horizon=5.0)
    return maximum > -STOP_DISTANCE if projection.pass_before else minimum < CRASH_DISTANCE

  def prove_finite_physical_risk(self, radarstate, *, now_ns, reference_ns):
    """Prove an imminent conflict in qualified longitudinal point geometry.

    Instant ACCEL_MAX is an optimistic bound for getting ahead before entry;
    instant ACCEL_MIN is an optimistic bound for staying behind. Excluding
    both even with these wider physical freedoms does not depend on a QP
    iterate or our conservative occupancy envelope. The resulting request is
    impact mitigation, never a claim that this collision has been avoided.
    """
    if (not getattr(radarstate, 'fusionHealthy', False)
        or not all(isinstance(stamp, Integral) and not isinstance(stamp, bool) and stamp > 0 for stamp in (now_ns, reference_ns))
        or not 0 <= now_ns - reference_ns <= REFERENCE_MAX_AGE_NS
        or self.x0[0] != 0.0 or self.x0[1] <= 0.0 or not ACCEL_MIN <= self.x0[2] <= ACCEL_MAX):
      return ()
    consumed_age = (now_ns - reference_ns) / 1e9
    proven = []
    for projection in self._fusion_projections:
      lead, evidence = projection.lead, projection.lead.evidence
      # Physical occupancy was certified for constant-velocity x/y motion.
      # Nonzero lead acceleration changes both longitudinal clearance and,
      # on a bend, entry/exit geometry. It needs its own physical window;
      # comparing two x trajectories against the CV window is insufficient.
      if (projection.persistent or projection.pass_before or not np.any(projection.occupied)
          or projection.identity is None or str(evidence.source) not in ('radar', 'fused') or evidence.coasting
          or getattr(evidence, 'physicalConflictValid', False) is not True or lead.aLeadK != 0.0
          or not qualified_lead(lead, 'brake', 0.0, now_ns=now_ns)
          or not qualified_lead(lead, 'fcw', 0.0, now_ns=now_ns)):
        continue
      physical_start = getattr(evidence, 'physicalConflictStart', None)
      physical_end = getattr(evidence, 'physicalConflictEnd', None)
      if not all(isinstance(value, Real) and not isinstance(value, bool) for value in (physical_start, physical_end)):
        continue
      try:
        physical_start, physical_end = float(physical_start), float(physical_end)
      except (OverflowError, ValueError):
        continue
      if (not np.all(np.isfinite([physical_start, physical_end]))
          or not evidence.conflictStart <= physical_start < physical_end <= evidence.conflictEnd):
        continue
      # The ordinary window may include padding, hysteresis or an envelope
      # across separate crossings. Only the producer's explicit physical
      # subwindow can prove urgent risk, on the same epoch as raw geometry.
      elapsed = (reference_ns - evidence.estimateMonoTime) / 1e9
      physical_start -= elapsed
      physical_end -= elapsed
      entry = max(consumed_age, physical_start)
      end = min(MAX_T, physical_end, consumed_age + FINITE_RISK_HORIZON)
      if end <= entry:
        continue
      object_at_entry = float(np.interp(entry, T_IDXS, projection.raw_trajectory[:, 0]))
      ego_upper_at_entry = self.ego_progress_bound(self.x0[1], ACCEL_MAX, entry)
      if ego_upper_at_entry >= object_at_entry + CRASH_DISTANCE - QP_FEASIBILITY_TOL:
        continue  # Some physically possible trajectory could already pass safely.
      times = [entry, end, *T_IDXS[(T_IDXS > entry) & (T_IDXS < end)]]
      for stage, duration in enumerate(np.diff(T_IDXS)):
        lead_velocity = (projection.raw_trajectory[stage+1, 0] - projection.raw_trajectory[stage, 0]) / duration
        extremum = (lead_velocity - self.x0[1]) / ACCEL_MIN
        if max(entry, T_IDXS[stage]) < extremum < min(end, T_IDXS[stage+1], -self.x0[1] / ACCEL_MIN):
          times.append(float(extremum))
      for timestamp in sorted(set(times)):
        object_position = float(np.interp(timestamp, T_IDXS, projection.raw_trajectory[:, 0]))
        ego_lower = self.ego_progress_bound(self.x0[1], ACCEL_MIN, float(timestamp))
        if object_position - ego_lower < CRASH_DISTANCE - QP_FEASIBILITY_TOL:
          proven.append(FiniteRiskEvidence(
            identity=projection.identity, source=str(evidence.source), consumed_ns=int(now_ns), reference_ns=int(reference_ns),
            estimate_ns=int(evidence.estimateMonoTime), position_ns=int(evidence.positionMonoTime), velocity_ns=int(evidence.velocityMonoTime),
            model_ns=int(evidence.modelMonoTime),
            conflict_start_s=physical_start, conflict_end_s=physical_end, proof_time_s=float(timestamp),
            entry_object_position=object_at_entry, entry_ego_upper_position=float(ego_upper_at_entry),
            proof_object_position=object_position, proof_ego_lower_position=float(ego_lower),
          ))
          break
    return tuple(sorted(proven, key=lambda item: (item.proof_time_s, item.identity)))

  def select_occupancy_branches(self, projections, reference):
    """Admit a single non-accelerating passing witness, never race a crossing.

    Only a physically impossible behind-all-objects branch motivates this
    alternative. The SAME integrated witness must clear all other objects,
    including persistent front vehicles; individual passing decisions cannot
    contradict each other. Independent model braking/stop goals veto it.
    """
    finite = tuple(p for p in projections if not p.persistent and np.any(p.occupied))
    if not finite or reference is None or not reference.allow_passing or self.x0[1] <= 0.0:
      return projections
    if any(p.hard_conflict_end < p.conflict_end for p in finite):
      # An unknown tail is still a possible obstacle. Truncating its hard
      # prediction is neither an observed exit nor evidence safe to pass.
      self.diagnostic_passing = {'accepted': False, 'reason': 'unknown_occupancy_tail'}
      return projections
    if (reference.typed_model is not None or any(value is not None for value in (
        reference.model_position, reference.model_velocity, reference.model_acceleration))):
      return projections
    lower = np.broadcast_to(np.asarray(reference.accel_min, dtype=float), (N+1,))
    upper = np.broadcast_to(np.asarray(reference.accel_max, dtype=float), (N+1,))
    if not np.all(np.isfinite([lower, upper])) or np.any(lower > 0.0) or np.any(upper < 0.0):
      return projections
    impossible_behind = False
    for projection in finite:
      start, end = max(0.0, projection.conflict_start), min(MAX_T, projection.conflict_end)
      interior = []
      for stage, duration in enumerate(np.diff(T_IDXS)):
        lead_velocity = (projection.raw_trajectory[stage+1, 0] - projection.raw_trajectory[stage, 0]) / duration
        extremum = (lead_velocity - self.x0[1]) / ACCEL_MIN
        if max(start, T_IDXS[stage]) < extremum < min(end, T_IDXS[stage+1], -self.x0[1] / ACCEL_MIN):
          interior.append(float(extremum))
      times = np.concatenate(([start], T_IDXS[(T_IDXS > start) & (T_IDXS < end)], interior, [end]))
      minimum_progress = np.array([self.ego_progress_bound(self.x0[1], ACCEL_MIN, float(t)) for t in times])
      # Instant full braking is a LOWER bound on forward travel, even with
      # measured positive a0 and finite jerk. Falling behind that bound is
      # impossible; no actual trajectory is replaced by this calculation.
      gap = np.interp(times, T_IDXS, projection.raw_trajectory[:, 0]) - minimum_progress
      impossible_behind |= bool(np.min(gap) < CRASH_DISTANCE + OCCUPANCY_POSITION_BUFFER)
    if not impossible_behind:
      return projections

    states, jerks = np.empty((N+1, X_DIM)), np.zeros(N)
    states[0] = self.x0
    # Release the existing acceleration over the first native interval. This
    # includes real a0; a constant-speed path with an instantaneous a0 reset
    # would not be a witness in this MPC's dynamics.
    jerks[0] = -self.x0[2] / T_IDXS[1]
    for stage, duration in enumerate(np.diff(T_IDXS)):
      states[stage+1] = self.integrate_jerk(states[stage], jerks[stage], duration)
    if self.trajectory_minimum_velocity(states, jerks) < 0.0:
      return projections
    choices = []
    for projection in projections:
      if not np.any(projection.occupied):
        choices.append(projection)
        continue
      minimum, maximum = self.occupancy_clearance_extrema(projection, states, jerks)
      if minimum >= STOP_DISTANCE:
        choices.append(projection)
      elif not projection.persistent and projection.conflict_start > 0.0 and maximum <= -STOP_DISTANCE:
        choices.append(replace(projection, pass_before=True))
      else:
        self.diagnostic_passing = {'accepted': False, 'reason': 'shared_witness_conflicts_with_object',
                                   'identity': projection.identity}
        return projections
    if not any(p.pass_before for p in choices):
      return projections
    speed_limit = float(np.max(states[:, 1]))
    if reference.cruise_speed is not None:
      cruise = np.broadcast_to(np.asarray(reference.cruise_speed, dtype=float), (N+1,))
      if not np.all(np.isfinite(cruise)) or np.any(cruise < speed_limit):
        self.diagnostic_passing = {'accepted': False, 'reason': 'cruise_below_passing_witness_speed'}
        return projections
    bounds = self.finite_occupancy_bounds(tuple(choices), passing_speed_limit=speed_limit, witness=states, witness_jerks=jerks)
    if bounds is None:
      self.diagnostic_passing = {'accepted': False, 'reason': 'shared_witness_has_no_bounded_corridor'}
      return projections
    self._passing_witness = states, jerks
    self._passing_speed_limit = speed_limit
    self.diagnostic_passing = {'accepted': True, 'identities': tuple(p.identity for p in choices if p.pass_before),
                               'speedLimit': speed_limit, 'clearance': STOP_DISTANCE,
                               'witnessInitialAcceleration': float(self.x0[2]), 'witnessReleaseJerk': float(jerks[0])}
    return tuple(choices)

  def occupancy_behind_position_limit(self, projection, stage, *, constraint_time):
    """Certify the complete occupied interval at a shared later position row.

    Hard acceleration boxes, linear a(t) and continuous v(t)>=0 imply
    v(t)>=max(v0+a_floor*t, 0). Thus x(g)-x(t)>=s_min(g)-s_min(t) for g>=t.
    Bounding x(g) by every lead(t)+that progress difference preserves actual
    clearance without tying an earlier moving target to a later ego position.
    """
    # Attribution consumes the same row after the solve. Bind reuse to the
    # complete input, including mutable array contents and the actual float
    # time: reconstructing a row time can differ by an ULP. Retain object
    # references in the entry so Python cannot recycle either identity.
    raw = projection.raw_trajectory
    key = (id(projection), id(raw), raw.shape, raw.dtype.str, raw.tobytes(),
           self.x0.shape, self.x0.dtype.str, self.x0.tobytes(), stage, constraint_time,
           projection.conflict_start, projection.conflict_end,
           projection.conflict_support_version, projection.conflict_supported_end)
    cached = self._occupancy_limit_cache.get(key)
    if cached is not None:
      return cached[2]
    left, right = T_IDXS[stage], T_IDXS[stage+1]
    start, end = max(left, projection.conflict_start), min(right, projection.hard_conflict_end)
    speed = float(self.x0[1])
    acceleration_floor = min(ACCEL_MIN, float(self.x0[2]))
    if (not np.all(np.isfinite([speed, self.x0[2], constraint_time])) or speed < 0.0
        or not left <= start < end <= constraint_time <= right):
      raise ValueError('Behind occupancy transport requires a later in-stage row')
    lead_position = float(projection.raw_trajectory[stage, 0])
    lead_velocity = float((projection.raw_trajectory[stage+1, 0] - lead_position) / (right-left))
    times = [start, end]
    # lead_x(t)-s_min(t) is convex before the lower-bound stop and linear
    # afterwards. Endpoints and this stationary point contain its minimum.
    stationary = (lead_velocity-speed) / acceleration_floor
    if start < stationary < end and stationary < speed / -acceleration_floor:
      times.append(stationary)
    minimum_progress = self.ego_progress_bound(speed, acceleration_floor, constraint_time)
    limit = min(lead_position + lead_velocity*(timestamp-left)
                + (minimum_progress-self.ego_progress_bound(speed, acceleration_floor, timestamp)) for timestamp in times)
    result = float(limit - CRASH_DISTANCE - OCCUPANCY_POSITION_BUFFER)
    self._occupancy_limit_cache[key] = projection, raw, result
    return result

  def finite_occupancy_bounds(self, projections=None, *, passing_speed_limit=None, witness=None, witness_jerks=None):
    """Fold supported finite conflicts into one exact position row per stage.

    Every behind condition is transported to the stage's latest occupied
    instant using a certified lower bound on intervening ego progress. The
    tightest condition then certifies every object's supported occupied interval
    with one position row. Persistent following retains its
    existing soft headway problem; no constraint uses its stopping equivalence
    or the solver's impossible-gap distance enlargement as physical geometry.
    """
    if projections is None:
      projections = self._fusion_projections
      passing_speed_limit = self._passing_speed_limit
      if self._passing_witness is not None:
        witness, witness_jerks = self._passing_witness
    bounds = [None] * N
    conditions = [[] for _ in range(N)]
    for projection in projections:
      if projection.persistent or not projection.occupied.any():
        continue
      for stage, (left, right) in enumerate(zip(T_IDXS[:-1], T_IDXS[1:], strict=True)):
        start, end = max(left, projection.conflict_start), min(right, projection.hard_conflict_end)
        if end <= start:
          continue
        if passing_speed_limit is not None:
          # Passing pivots can precede an object's occupied end. Their
          # existing transport uses a certified upper speed instead.
          positions = np.interp([start, end], T_IDXS, projection.raw_trajectory[:, 0])
          if projection.pass_before:
            conditions[stage].append((float(start-left), float(np.max(positions)+STOP_DISTANCE+OCCUPANCY_POSITION_BUFFER), True,
                                      projection.identity))
          else:
            conditions[stage].append((float(end-left), float(np.min(positions)-CRASH_DISTANCE-OCCUPANCY_POSITION_BUFFER), False,
                                      projection.identity))
          continue
        conditions[stage].append(projection)
    if passing_speed_limit is None:
      for stage, objects in enumerate(conditions):
        if not objects:
          continue
        # Decide the common time first. Reusing a bound computed at another
        # object's earlier end would make the result depend on list order.
        end = min(T_IDXS[stage+1], max(projection.hard_conflict_end for projection in objects))
        limit = min(self.occupancy_behind_position_limit(projection, stage, constraint_time=end) for projection in objects)
        bounds[stage] = OccupancyBound(float(end - T_IDXS[stage]), limit, tuple(projection.identity for projection in objects))
    if passing_speed_limit is not None:
      for stage, rows in enumerate(conditions):
        if not rows:
          continue
        # One position row has a lower and upper bound at the same time.
        # Monotone x and certified v<=speed_limit conservatively transport
        # all earlier/later conditions there. Try only three bounded pivots,
        # and admit a corridor only when the shared witness fits it exactly.
        times = [row[0] for row in rows]
        ahead_times = [row[0] for row in rows if row[2]]
        pivots = sorted({min(times), max(times), min(ahead_times) if ahead_times else max(times)})
        for pivot in pivots:
          position_lower, position_upper = -REFERENCE_BOUND, REFERENCE_BOUND
          for timestamp, limit, ahead, _ in rows:
            if ahead:
              position_lower = max(position_lower, limit + passing_speed_limit * max(0.0, pivot - timestamp))
            else:
              position_upper = min(position_upper, limit - passing_speed_limit * max(0.0, timestamp - pivot))
          position = self.integrate_jerk(witness[stage], witness_jerks[stage], pivot)[0]
          if position_lower <= position <= position_upper:
            bounds[stage] = OccupancyBound(pivot, position_upper, tuple(row[3] for row in rows), position_lower)
            break
        if bounds[stage] is None:
          return None
    return tuple(bounds)

  @property
  def finite_occupancy_physics_required(self):
    """Finite risk cannot buy clearance with backwards or impossible motion.

    A support deadline limits hard position geometry, not the physical
    vehicle model used to respond to its remaining soft obstacle. Ordinary
    persistent following and non-fusion problems retain their existing policy.
    """
    return (any(self._occupancy_bounds)
            or any(not projection.persistent and projection.occupied.any() for projection in self._fusion_projections))

  def occupancy_speed_row(self, stage):
    """One sufficient continuous nonnegative-speed row, with fixed-x0 care."""
    duration = T_IDXS[stage+1] - T_IDXS[stage]
    if stage == 0:
      # v+a*h/2 may be negative for a valid low-speed initial state even
      # though a rapid release of braking can keep v(t)>=0. Preserve the
      # existing split proof: fixed b1 is checked, leaving b2 as the one row.
      split = self.initial_rest_split(self.x0, duration)
      if (self.x0[1] < 0.0 or split <= 0.0
          or self.x0[1] + self.x0[2] * split / 2.0 < 0.0):
        raise ValueError('Finite occupancy has no forward initial certificate')
      return np.array([0.0, 1.0, (duration + split) / 2.0]), duration * split / 2.0
    # v(0), v(h) are constrained at the nodes. The third quadratic Bernstein
    # coefficient v+a*h/2>=0 certifies this entire velocity polynomial. This
    # is a stronger feasible subset than the ordinary split-rest certificate.
    return np.array([0.0, 1.0, duration / 2.0]), 0.0

  def _occupancy_auxiliary_tolerances(self, stage, tolerance):
    """Identify installed auxiliary rows; unknown and other rest rows stay strict."""
    lower_tolerance, upper_tolerance, roles = [tolerance] * 2, [tolerance] * 2, [None] * 2
    if (not self._native_occupancy_bounds_active or self.solution_status != 0
        or self.diagnostic_last_solver_return != 0):
      return lower_tolerance, upper_tolerance, roles
    C, D, lower, upper = self._rest_constraint_buffers[stage]
    bound = self._occupancy_bounds[stage]
    node = self.effective_rest_node
    if node is None:
      node = self.rest_goal.native_node if self.rest_goal is not None else 0
    future = (self.rest_goal_state == 'constrained' or self.effective_rest_state == 'constrained') and node > 0
    if bound is not None or not future or stage >= node - 1:
      row, control = self.occupancy_speed_row(stage)
      ceiling = self._passing_speed_limit if self._passing_speed_limit is not None else REFERENCE_BOUND
      if np.array_equal(C[0], row) and D[0, 0] == control and lower[0] == 0.0 and upper[0] == ceiling:
        lower_tolerance[0] = min(OCCUPANCY_VELOCITY_REINTEGRATION_LIMIT, 1e-6)
        roles[0] = 'speed_lower_m_per_s'
    if bound is not None and 0.0 < bound.duration <= T_IDXS[stage+1] - T_IDXS[stage]:
      duration = bound.duration
      if (np.array_equal(C[1], [1.0, duration, duration**2 / 2.0]) and D[1, 0] == duration**3 / 6.0
          and lower[1] == bound.position_lower_limit and upper[1] == bound.position_limit):
        lower_tolerance[1] = upper_tolerance[1] = min(OCCUPANCY_POSITION_REINTEGRATION_LIMIT, OCCUPANCY_POSITION_BUFFER / 1000.0)
        roles[1] = 'position_m'
    return lower_tolerance, upper_tolerance, roles

  def occupancy_trajectory_error(self, states, jerks, *, tolerance, post_native=False):
    """Recheck the final trajectory, including any rest-precision adjustment."""
    if not self.finite_occupancy_physics_required:
      return ''
    if (states.shape != (N+1, X_DIM) or jerks.shape != (N,)
        or not np.all(np.isfinite(states)) or not np.all(np.isfinite(jerks))):
      return 'invalid_finite_occupancy_trajectory'
    if post_native and (self.solution_status != 0 or self.diagnostic_last_solver_return != 0):
      return 'finite_occupancy_postcheck_requires_native_success'
    if max(float(np.max(np.abs(states[0] - self.x0))),
           float(np.max(np.abs(self.trajectory_dynamics_defects(states, jerks))))) > tolerance:
      return 'finite_occupancy_dynamics_residual_exceeded'
    if post_native:
      # Check the current installed problem each time, including after a rest
      # precision adjustment. A previous repair flag never bypasses a check.
      lower, upper = self.diagnostic_reference_bounds
      box_violation = float(max(0.0, np.max(lower[1:] - states[1:, 1:]), np.max(states[1:, 1:] - upper[1:])))
      maximum_violation, rejected, used_tolerances = box_violation, box_violation > tolerance, []
      for stage, (C, D, row_lower, row_upper) in enumerate(self._rest_constraint_buffers):
        value = C @ states[stage] + D[:, 0] * jerks[stage]
        low_tolerance, high_tolerance, roles = self._occupancy_auxiliary_tolerances(stage, tolerance)
        for index in range(2):
          low_violation, high_violation = float(max(0.0, row_lower[index] - value[index])), float(max(0.0, value[index] - row_upper[index]))
          maximum_violation = max(maximum_violation, low_violation, high_violation)
          rejected |= low_violation > low_tolerance[index] or high_violation > high_tolerance[index]
          if max(low_violation, high_violation) > tolerance:
            used_tolerances.append({
              'stage': stage, 'row': index, 'role': roles[index],
              'unit': 'm' if roles[index] == 'position_m' else 'm/s' if roles[index] == 'speed_lower_m_per_s' else None,
              'lowerTolerance': low_tolerance[index], 'upperTolerance': high_tolerance[index],
              'rawLowerViolation': low_violation, 'rawUpperViolation': high_violation,
            })
      if self.diagnostic_occupancy_projection is not None or used_tolerances or rejected:
        if self.diagnostic_occupancy_projection is None:
          self.diagnostic_occupancy_projection = {'applied': False}
        self.diagnostic_occupancy_projection.update(maximumCorrectedConstraintViolation=maximum_violation,
                                                    auxiliaryConstraintTolerances=used_tolerances)
      if rejected:
        return 'finite_occupancy_reintegration_constraints_violated'
    if (np.any(states[1:, 1] < -tolerance)
        or np.any(states[1:, 2] < self.params[1:, 0] - tolerance)
        or np.any(states[1:, 2] > self.params[1:, 1] + tolerance)):
      return 'finite_occupancy_physical_bounds_violated'
    if self.trajectory_minimum_velocity(states, jerks) < -tolerance:
      return 'finite_occupancy_continuous_velocity_violated'
    if post_native and any(self.segment_minimum_velocity(states[stage], jerk, duration) < -tolerance
                           for stage, (jerk, duration) in enumerate(zip(jerks, np.diff(T_IDXS), strict=True))):
      return 'finite_occupancy_continuous_velocity_violated'
    for stage, bound in enumerate(self._occupancy_bounds):
      if bound is None:
        continue
      row, control = self.occupancy_speed_row(stage)
      low_tolerance, high_tolerance = [tolerance] * 2, [tolerance] * 2
      if post_native:
        low_tolerance, high_tolerance, _ = self._occupancy_auxiliary_tolerances(stage, tolerance)
      if row @ states[stage] + control * jerks[stage] < -low_tolerance[0]:
        return 'finite_occupancy_continuous_velocity_violated'
      position = self.integrate_jerk(states[stage], jerks[stage], bound.duration)[0]
      if position > bound.position_limit + high_tolerance[1] or position < bound.position_lower_limit - low_tolerance[1]:
        return 'finite_occupancy_position_bound_violated'
    for projection in self._fusion_projections:
      if not projection.persistent:
        minimum, maximum = self.occupancy_clearance_extrema(projection, states, jerks, supported_only=True)
        if (maximum > -STOP_DISTANCE if projection.pass_before else minimum < CRASH_DISTANCE):
          return 'finite_occupancy_clearance_violated'
    if self._passing_speed_limit is not None:
      if np.any(states[1:, 2] > tolerance):
        return 'finite_occupancy_passing_acceleration_positive'
      if np.any(states[:, 1] > self._passing_speed_limit + tolerance):
        return 'finite_occupancy_passing_speed_exceeded'
      for stage, jerk in enumerate(jerks):
        row, control = self.occupancy_speed_row(stage)
        if row @ states[stage] + control * jerk > self._passing_speed_limit + tolerance:
          return 'finite_occupancy_passing_speed_exceeded'
        if post_native and -self.segment_minimum_velocity(-states[stage], -jerk, T_IDXS[stage+1] - T_IDXS[stage]) > self._passing_speed_limit + tolerance:
          return 'finite_occupancy_passing_speed_exceeded'
    return ''

  def reintegrate_occupancy_precision(self):
    """Reconcile bounded native state defects with unchanged control and x0.

    A successful native solve is required. No state is clipped and jerk never
    changes. The candidate must satisfy all physical constraints, with only
    identified auxiliary rows using separate unit-specific numeric residuals.
    The original 1e-9 dynamics and exact occupancy-geometry checks stay intact.
    Independent position/velocity/acceleration budgets bound the complete continuous
    correction; larger residuals and failed solves remain explicit failures.
    """
    if not self.finite_occupancy_physics_required:
      return ''
    if self.solution_status != 0 or self.diagnostic_last_solver_return != 0:
      return 'finite_occupancy_reintegration_requires_native_success'
    raw_states, jerks = self.x_sol, self.u_sol[:, 0]
    defects = self.trajectory_dynamics_defects(raw_states, jerks)
    maximum_defect = np.maximum(np.max(np.abs(defects), axis=0), np.abs(raw_states[0] - self.x0))
    if np.max(maximum_defect) <= QP_FEASIBILITY_TOL:
      return ''
    budget = np.array([OCCUPANCY_POSITION_REINTEGRATION_LIMIT, OCCUPANCY_VELOCITY_REINTEGRATION_LIMIT,
                       OCCUPANCY_ACCELERATION_REINTEGRATION_LIMIT])
    diagnostic = {'applied': False, 'maximumNativeDynamicsDefect': maximum_defect.tolist(),
                  'maximumStateCorrectionBudget': budget.tolist(), 'maximumJerkCorrection': 0.0,
                  'stateCorrectionUnits': ['m', 'm/s', 'm/s^2']}
    self.diagnostic_occupancy_projection = diagnostic
    if not np.all(np.isfinite(maximum_defect)) or np.any(maximum_defect > budget):
      return 'finite_occupancy_dynamics_residual_exceeded'
    states = np.empty_like(raw_states)
    states[0] = self.x0
    for stage, duration in enumerate(np.diff(T_IDXS)):
      states[stage+1] = self.integrate_jerk(states[stage], jerks[stage], duration)
    correction = np.max(np.abs(states - raw_states), axis=0)
    diagnostic['maximumStateCorrection'] = correction.tolist()
    if not np.all(np.isfinite(states)) or np.any(correction > budget):
      return 'finite_occupancy_reintegration_budget_exceeded'
    # The corrected and raw segments use identical jerk. Their state
    # difference is therefore exactly quadratic/linear/constant in time.
    # Bound the complete interval, not just its nodes: a tiny velocity or
    # acceleration correction must not hide larger between-node displacement.
    durations = np.diff(T_IDXS)
    delta = np.abs(states[:-1] - raw_states[:-1])
    continuous_correction = np.maximum(correction, [
      np.max(delta[:, 0] + delta[:, 1] * durations + 0.5 * delta[:, 2] * durations**2),
      np.max(delta[:, 1] + delta[:, 2] * durations),
      np.max(delta[:, 2]),
    ])
    diagnostic['maximumContinuousStateCorrectionBound'] = continuous_correction.tolist()
    if np.any(continuous_correction > budget):
      return 'finite_occupancy_reintegration_budget_exceeded'
    if (failure := self.occupancy_trajectory_error(states, jerks, tolerance=QP_FEASIBILITY_TOL, post_native=True)):
      return failure
    self.x_sol[:] = states
    # run() established views of x_sol. Keep reference history aligned with
    # the same accepted trajectory, as in the existing rest precision repair.
    self.a_prev = np.interp(T_IDXS + self.dt, T_IDXS, self.a_solution)
    diagnostic['applied'] = True
    return ''

  def fusion_fcw_identity(self, projection):
    if projection.identity is not None:
      return projection.identity
    lead, evidence = projection.lead, projection.lead.evidence
    if str(evidence.source) != 'vision' or lead.modelProb <= 0.9:
      return None
    # Vision has no persistent object ID. Preserve its warning capability only
    # through a short, kinematically compatible continuation of the same slot;
    # index alone cannot splice collision counts across a new cut-in vehicle.
    slot, epoch = evidence.visionIndex, evidence.estimateMonoTime
    previous = self._fusion_vision_fcw_tracks.get(slot)
    key = None
    if previous is not None:
      previous_key, previous_epoch, distance, velocity, relative_velocity, lateral = previous
      elapsed = (epoch - previous_epoch) / 1e9
      if (0.0 <= elapsed <= 0.25 and abs(lead.dRel - (distance + relative_velocity * elapsed)) <= 1.0
          and abs(lead.vLead - velocity) <= 2.0 and abs(lead.yRel - lateral) <= 0.5):
        key = previous_key
    if key is None:
      self._fusion_vision_fcw_sequence += 1
      key = ('vision_continuation', self._fusion_vision_fcw_sequence)
    self._fusion_vision_fcw_tracks[slot] = (key, epoch, lead.dRel, lead.vLead, lead.vRel, lead.yRel)
    return key

  def update_fusion_fcw(self, projections, *, now_ns):
    counts, epochs = {}, {}
    vision_slots = set()
    for projection in projections:
      lead = projection.lead
      if not qualified_lead(lead, 'fcw', 0.9, now_ns=now_ns):
        continue
      identity = self.fusion_fcw_identity(projection)
      if str(lead.evidence.source) == 'vision':
        vision_slots.add(lead.evidence.visionIndex)
      if identity is None or not self.collision_in_occupancy(projection, self.x_sol, self.j_solution):
        continue
      epoch = int(lead.evidence.estimateMonoTime)
      previous_epoch = self._fusion_fcw_epochs.get(identity)
      continuous = previous_epoch is not None and 0 <= epoch - previous_epoch <= 250_000_000
      previous_count = self._fusion_fcw_counts.get(identity, 0) if continuous else 0
      counts[identity] = previous_count + int(not continuous or epoch > previous_epoch)
      epochs[identity] = epoch
    self._fusion_vision_fcw_tracks = {slot: track for slot, track in self._fusion_vision_fcw_tracks.items() if slot in vision_slots}
    self._fusion_fcw_counts, self._fusion_fcw_epochs = counts, epochs
    self.fcw_object_identity = max(counts, key=counts.get) if counts else None
    self.crash_cnt = counts.get(self.fcw_object_identity, 0)

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
    # The projection set and masks are fixed during this attribution pass.
    # Inspect each mask once; every remaining object's own interval and row
    # still participates independently at every occupied stage below.
    finite_projections = tuple(projection for projection in self._fusion_projections
                               if not projection.persistent and projection.occupied.any())
    for stage, bound in enumerate(self._occupancy_bounds):
      if bound is None:
        continue
      position = self.integrate_jerk(self.x_sol[stage], self.j_solution[stage], bound.duration)[0]
      for projection in finite_projections:
        start = max(T_IDXS[stage], projection.conflict_start)
        end = min(T_IDXS[stage+1], projection.hard_conflict_end)
        if end <= start:
          continue
        if projection.pass_before:
          positions = np.interp([start, end], T_IDXS, projection.raw_trajectory[:, 0])
          limit = np.max(positions) + STOP_DISTANCE + OCCUPANCY_POSITION_BUFFER
          limit += self._passing_speed_limit * max(0.0, bound.duration - (start - T_IDXS[stage]))
        elif self._passing_speed_limit is None:
          limit = self.occupancy_behind_position_limit(projection, stage, constraint_time=T_IDXS[stage] + bound.duration)
        else:
          positions = np.interp([start, end], T_IDXS, projection.raw_trajectory[:, 0])
          limit = np.min(positions) - CRASH_DISTANCE - OCCUPANCY_POSITION_BUFFER
          limit -= self._passing_speed_limit * max(0.0, (end - T_IDXS[stage]) - bound.duration)
        if abs(position - limit) <= tolerance:
          token = self.lead_influence_token(projection.lead)
          if token is not None:
            active.add(token)
    if (self._passing_speed_limit is not None
        and np.any(np.abs(self.v_solution[1:] - self._passing_speed_limit) <= tolerance)):
      active.update(self.lead_influence_token(projection.lead) for projection in self._fusion_projections if projection.pass_before)
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
        or np.any(parameters[:, [COMFORT_NORMALIZER, RISK_NORMALIZER]] < 0.0)
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
      risk_normalizer = np.where(parameters[:, RISK_NORMALIZER] > 0.0,
                                 parameters[:, RISK_NORMALIZER], states[:, 1] + 10.0)
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
        if not reason:
          reason = self.occupancy_trajectory_error(states, jerks, tolerance=arithmetic)
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
    # Future rest ends the entering interval at a<=0. Its velocity is monotone
    # or concave, so the minimum is an endpoint. Later intervals have both
    # accelerations <=0 and are monotone. Node v>=0 certifies these intervals
    # without redundant C rows; exact due-rest keeps its existing separate path.
    constrained_stages = max(0, node - 1) if future else 0
    splits = []
    occupancy_stages = []
    self._native_occupancy_bounds_active = self.finite_occupancy_physics_required
    for stage, duration in enumerate(np.diff(T_IDXS)):
      C, D, lower, upper = self._rest_constraint_buffers[stage]
      C.fill(0.0)
      D.fill(0.0)
      lower.fill(-1.0)
      upper.fill(1.0)
      bound = self._occupancy_bounds[stage]
      if bound is not None:
        C[0], D[0, 0] = self.occupancy_speed_row(stage)
        lower[0], upper[0] = 0.0, self._passing_speed_limit if self._passing_speed_limit is not None else REFERENCE_BOUND
        endpoint = bound.duration
        C[1] = [1.0, endpoint, endpoint**2 / 2.0]
        D[1, 0] = endpoint**3 / 6.0
        lower[1], upper[1] = bound.position_lower_limit, bound.position_limit
        occupancy_stages.append(stage)
        if stage < constrained_stages:
          splits.append(None)
      elif stage < constrained_stages:
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
      elif self._native_occupancy_bounds_active:
        # An earlier between-node reverse excursion cannot buy extra space
        # for a later crossing or reduce its soft risk after hard support ends.
        # Keep this certificate over the complete horizon, even when no hard
        # position interval remains; future-rest intervals retain their two rows.
        C[0], D[0, 0] = self.occupancy_speed_row(stage)
        lower[0], upper[0] = 0.0, self._passing_speed_limit if self._passing_speed_limit is not None else REFERENCE_BOUND
      self.solver.constraints_set(stage, 'C', C)
      self.solver.constraints_set(stage, 'D', D)
      self.solver.constraints_set(stage, 'lg', lower)
      self.solver.constraints_set(stage, 'ug', upper)
    if future:
      self.diagnostic_rest_goal['continuousSpeedCertificate'] = (
        'quadratic_bernstein_with_finite_occupancy_rows' if occupancy_stages else 'quadratic_bernstein'
      )
      self.diagnostic_rest_goal['continuousSpeedSplitTimes'] = splits
      if occupancy_stages:
        self.diagnostic_rest_goal['finiteOccupancyStages'] = occupancy_stages

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

  def set_reference_problem(self, reference, x_obstacles, sources, accel_min, t_follow, *, lead_slots=None,
                            unknown_tail_masks=None):
    """Install genuine costs and state bounds in this same native problem."""
    if lead_slots is None:
      lead_slots = tuple(MPC_SOURCES.index(source) if source in MPC_SOURCES else None for source in sources)
    if len(lead_slots) != len(sources) or any(slot not in (None, 0, 1) for slot in lead_slots):
      raise ValueError('Invalid compact lead assignment')
    if (unknown_tail_masks is not None
        and (unknown_tail_masks.shape != x_obstacles.shape or unknown_tail_masks.dtype != np.bool_
             or np.any(unknown_tail_masks & ~np.isfinite(x_obstacles)))):
      raise ValueError('Invalid unknown-tail obstacle assignment')
    typed = reference.typed_model if reference is not None else None
    if typed is not None and not isinstance(typed, ModelReferenceEvidence):
      raise ValueError('Invalid typed model reference')
    self.rest_goal = typed.rest_goal if typed is not None else None
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
    risk_selected = selected
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
      risk_selected = np.argmin(risk_obstacles, axis=1)
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
    if unknown_tail_masks is not None:
      # Compact comfort/risk offsets can choose different physical objects.
      # Preserve their individual first-minimum tie order and never let an
      # unselected unknown object change another residual's normalization.
      stages = np.arange(N+1)
      fixed_normalizer = max(float(self.x0[1]), 0.0) + 10.0
      self.params[:, COMFORT_NORMALIZER] = np.where(unknown_tail_masks[stages, comfort_selected], fixed_normalizer, 0.0)
      self.params[:, RISK_NORMALIZER] = np.where(unknown_tail_masks[stages, risk_selected], fixed_normalizer, 0.0)
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
        node = self.effective_rest_node if scheduled_primal is not None else goal.native_node
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
        if node == 0:
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

    if self.finite_occupancy_physics_required:
      # Position clearance is not bought with an impossible acceleration or
      # a backwards velocity excursion, including when only the conservative
      # soft tail remains. Intersect after the rest boxes are built so a
      # crossing can never relax a simultaneous model stop.
      lower[1:, 0] = np.maximum(lower[1:, 0], 0.0)
      lower[1:, 1] = np.maximum(lower[1:, 1], self.params[1:, 0])
      upper[1:, 1] = np.minimum(upper[1:, 1], self.params[1:, 1])
      if self._passing_speed_limit is not None:
        upper[1:, 0] = np.minimum(upper[1:, 0], self._passing_speed_limit)
        # A speed ceiling alone permits braking and then accelerating again.
        # Passing is a release of existing a0, never a new positive-accel
        # maneuver. Linear a(t) and nonpositive future nodes certify the
        # complete horizon after the first interval; the measured a0 is fixed.
        upper[1:, 1] = np.minimum(upper[1:, 1], 0.0)
    if self._passing_witness is not None:
      # This dynamically consistent feasible branch witness is a warm start,
      # not a substituted solution. It seeds the same current native problem;
      # a rejected occupancy candidate may receive one bounded refinement.
      states, jerks = self._passing_witness
      for stage in range(N+1):
        self.solver.set(stage, 'x', states[stage])
        if stage < N:
          self.solver.set(stage, 'u', np.array([jerks[stage]]))
    self.set_continuous_rest_bounds()

    self.install_cost_weights(weights)
    for stage in range(N+1):
      self.solver.set(stage, 'yref', self.yref[stage, :COST_DIM if stage < N else COST_E_DIM])
      if stage:
        self.solver.constraints_set(stage, 'lbx', lower[stage])
        self.solver.constraints_set(stage, 'ubx', upper[stage])
    self.diagnostic_reference_weights = weights.copy()
    self.diagnostic_reference_bounds = (lower.copy(), upper.copy())

  def check_rest_goal(self):
    """Judge the original source goal; scheduling never changes this contract."""
    if self.rest_goal is None:
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

  def update(self, radarstate, personality=log.LongitudinalPersonality.standard, *, reference=None, now_ns=None, reference_ns=None):
    self._occupancy_limit_cache = {}
    self.continuation_problem = None
    self.params[:, COMFORT_NORMALIZER:RISK_NORMALIZER+1] = 0.0
    self.diagnostic_occupancy_refinement = None
    # Reading an unset struct through a capnp builder allocates it. MPC and
    # UI attribution observe the exact incoming message; they never edit it.
    if hasattr(radarstate, 'as_reader'):
      radarstate = radarstate.as_reader()
    self.diagnostic_last_obstacle_summary = None
    self.source_object_identity = None
    self.fcw_object_identity = None
    self._obstacle_identities = ()
    self._fusion_projections = ()
    self._occupancy_bounds = (None,) * N
    self.diagnostic_occupancy_bounds = ()
    self.diagnostic_occupancy_projection = None
    self._passing_witness = None
    self._passing_speed_limit = None
    self.diagnostic_passing = None
    self.object_influence_identities = ()
    self._influence_problem = None
    self._finite_risk_candidates = ()
    self.finite_risk_evidence = ()
    self.finite_risk_braking_accel = None
    self.rest_goal = None
    self.rest_goal_state = 'none'
    self.rest_goal_reason = ''
    self.rest_goal_braking_accel = None
    self.effective_rest_node = None
    self.effective_rest_state = 'none'
    self.effective_rest_reason = ''
    self.rest_recovery_position_limit = None
    self.diagnostic_rest_goal = None
    t_follow = get_T_FOLLOW(personality)
    self.solve_time = 0.0

    if not np.all(np.isfinite(self.x0)):
      self.fail(MPC_INVALID_INPUT)
      return
    fusion_active = bool(getattr(radarstate, 'fusionActive', False))
    if fusion_active != getattr(self, '_was_fusion_active', False):
      self.crash_cnt = 0
      self._fusion_fcw_counts = {}
      self._fusion_fcw_epochs = {}
      self._fusion_vision_fcw_tracks = {}
      self._fusion_follow_geometry = {}
    self._was_fusion_active = fusion_active
    lead_slots = None
    unknown_tail_masks = None
    if fusion_active:
      try:
        if any(stamp is not None and (not isinstance(stamp, Integral) or isinstance(stamp, bool) or stamp <= 0)
               for stamp in (now_ns, reference_ns)):
          raise ValueError('Invalid fusion clock')
        if (now_ns is not None and reference_ns is not None
            and not 0 <= now_ns - reference_ns <= REFERENCE_MAX_AGE_NS):
          raise ValueError('Fusion initial state is not current at the consumer')
        with np.errstate(over='ignore', invalid='ignore'):
          projections = self.fusion_lead_projections(radarstate, now_ns=now_ns, reference_ns=reference_ns)
          projections = self.select_occupancy_branches(projections, reference)
        self._fusion_projections = projections
      except (AttributeError, TypeError, ValueError, FloatingPointError, OverflowError):
        self.fail(MPC_INVALID_INPUT)
        return
      present_leads = tuple(projection.lead for projection in projections if np.any(projection.occupied))
      # The scalar recovery proof freezes a continuously occupied forward
      # lead. A short crossing must never become a permanent stop-position cap.
      recovery_leads = tuple(projection for projection in projections if projection.persistent and not projection.lead.evidence.coasting)
      if recovery_leads and all(projection.raw_trajectory[0, 1] >= 0.0 for projection in recovery_leads):
        limit = self.x0[0] + min(projection.raw_trajectory[0, 0] for projection in recovery_leads) - CRASH_DISTANCE
        if limit > self.x0[0]:
          self.rest_recovery_position_limit = float(limit)
      obstacles = [np.full(N+1, np.inf), np.full(N+1, np.inf)]
      unknown_tail_masks = [np.zeros(N+1, dtype=bool), np.zeros(N+1, dtype=bool)]
      sources = [LongitudinalPlanSource.cruise, LongitudinalPlanSource.cruise]
      lead_slots, identities, influence_tokens = [None, None], [None, None], [None, None]
      for projection in projections:
        # A reversing target has no positive forward stopping-distance credit.
        obstacle = (projection.trajectory[:, 0]
                    + get_stopped_equivalence_factor(np.maximum(projection.trajectory[:, 1], 0.0)))
        # The soft occupied mask also covers an interpolation endpoint after
        # the real window. That outer node is not an unknown tail when the
        # complete window has support; legacy and version-1 costs then agree.
        has_unknown_tail = (projection.conflict_support_version == 1 and not projection.persistent
                            and projection.hard_conflict_end < projection.conflict_end)
        unknown_tail = (projection.occupied & (T_IDXS > projection.hard_conflict_end)
                        & has_unknown_tail & (not projection.pass_before))
        # Keep raw opposing motion and every supported hard row. Keep the
        # unknown tail in the objective; saturate its final equivalence at
        # the ego origin only when it projects behind that origin.
        # Floor the final equivalence, before the cross-object minimum, so a
        # positive lead-speed credit is neither invented nor added twice.
        obstacle = np.where(unknown_tail, np.maximum(obstacle, self.x0[0]), obstacle)
        obstacles.append(np.where(projection.occupied & (not projection.pass_before), obstacle, np.inf))
        unknown_tail_masks.append(unknown_tail)
        sources.append(MPC_SOURCES[projection.slot] if projection.slot is not None else LongitudinalPlanSource.lead0)
        # Additional objects and finite occupancy never inherit lead0's
        # separately qualified compact distance merely by sharing its enum.
        compact_slot = (projection.slot if projection.persistent
                        and qualified_lead(projection.lead, 'stop', now_ns=now_ns) else None)
        lead_slots.append(compact_slot)
        identities.append(projection.identity)
        influence_tokens.append(self.lead_influence_token(projection.lead))
    else:
      for lead in (radarstate.leadOne, radarstate.leadTwo):
        if lead.present and not np.all(np.isfinite([lead.dRel, lead.vLead, lead.aLeadK, lead.aLeadTau])):
          self.fail(MPC_INVALID_INPUT)
          return
      present_leads = tuple(lead for lead in (radarstate.leadOne, radarstate.leadTwo) if lead.present)
    if not fusion_active and present_leads and all(lead.vLead >= 0.0 for lead in present_leads):
      # Lock the geometry before process_lead clips an impossible close distance
      # upward or adds a moving lead's stopping equivalence. Freezing each lead
      # at its observed position is conservative only for non-reversing leads.
      # CRASH_DISTANCE remains the existing geometric margin, not a certified
      # road clearance or a replacement for the independent stopping contracts.
      limit = self.x0[0] + min(lead.dRel for lead in present_leads) - CRASH_DISTANCE
      if limit > self.x0[0]:
        self.rest_recovery_position_limit = float(limit)
    if not fusion_active:
      with np.errstate(over='ignore', invalid='ignore'):
        lead_xv_0 = self.process_lead(radarstate.leadOne)
        lead_xv_1 = self.process_lead(radarstate.leadTwo)

      # Add the forward lead's stopping distance to form an equivalent
      # stationary obstacle. The native solver consumes their tightest row.
      lead_0_obstacle = lead_xv_0[:,0] + get_stopped_equivalence_factor(lead_xv_0[:,1])
      lead_1_obstacle = lead_xv_1[:,0] + get_stopped_equivalence_factor(lead_xv_1[:,1])
      obstacles = [lead_0_obstacle, lead_1_obstacle]
      sources = list(MPC_SOURCES)
      influence_tokens = [self.lead_influence_token(lead, model_ns=getattr(radarstate, 'mdMonoTime', 0), slot=slot)
                          for slot, lead in enumerate((radarstate.leadOne, radarstate.leadTwo))]
    accel_min, accel_max = ACCEL_MIN, ACCEL_MAX
    ordinary_continuation = False
    if reference is not None:
      try:
        with np.errstate(over='ignore', invalid='ignore'):
          reference_obstacles, reference_sources, accel_min, accel_max = self.reference_obstacles(reference, t_follow)
      except (TypeError, ValueError, FloatingPointError):
        self.fail(MPC_INVALID_INPUT)
        return
      ordinary_continuation = (
        reference.cruise_speed is None and reference.typed_model is None and not present_leads
        and all(values is None for values in (reference.model_position, reference.model_velocity, reference.model_acceleration))
      )
      for lead_index, lead in enumerate((radarstate.leadOne, radarstate.leadTwo)):
        if not fusion_active and not lead.present:
          if ordinary_continuation:
            # Keep P's finite process_lead(None) continuation and distance cost.
            # It is not an observed lead and gets no compact-lead adjustment.
            sources[lead_index] = LongitudinalPlanSource.cruise
          else:
            obstacles[lead_index] = np.full(N+1, np.inf)
      obstacles.extend(reference_obstacles)
      sources.extend(reference_sources)
      influence_tokens.extend([None] * len(reference_sources))
      if fusion_active:
        lead_slots.extend([None] * len(reference_sources))
        identities.extend([None] * len(reference_sources))
        unknown_tail_masks.extend(np.zeros(N+1, dtype=bool) for _ in reference_sources)
    if fusion_active:
      # A released crossing leaves no obstacle. Where there is no explicit
      # cruise/model goal either, retain the same finite ordinary continuation
      # used by the old no-lead problem, without inventing an observed vehicle.
      unsupported = ~np.any(np.isfinite(np.column_stack(obstacles)), axis=1)
      if np.any(unsupported):
        continuation = self.process_lead(None)
        continuation_obstacle = continuation[:, 0] + get_stopped_equivalence_factor(continuation[:, 1])
        for index in (0, 1):
          obstacles[index][unsupported] = continuation_obstacle[unsupported]
      self._obstacle_identities = tuple(identities)
    x_obstacles = np.column_stack(obstacles)
    influence_slots = (lead_slots if lead_slots is not None
                       else tuple(MPC_SOURCES.index(source) if source in MPC_SOURCES else None for source in sources))
    self._influence_problem = x_obstacles.copy(), tuple(influence_tokens), tuple(influence_slots), reference
    selected = int(np.argmin(x_obstacles[1 if reference is not None else 0]))
    self.source = sources[selected]
    if fusion_active:
      self.source_object_identity = self._obstacle_identities[selected]

    # Scheduling inspects exactly the parameters that this update will solve.
    self.params[:,0] = accel_min
    self.params[:,1] = accel_max
    self.params[:,2] = np.min(x_obstacles, axis=1)
    self.params[:,3] = np.copy(self.a_prev)
    self.params[:,4] = t_follow
    self.params[:,5] = LEAD_DANGER_FACTOR

    try:
      self._occupancy_bounds = self.finite_occupancy_bounds()
      self.diagnostic_occupancy_bounds = tuple(
        (stage, bound.duration, bound.position_limit, bound.identities, bound.position_lower_limit)
        for stage, bound in enumerate(self._occupancy_bounds) if bound is not None
      )
      self.set_reference_problem(reference, x_obstacles, sources, accel_min, t_follow, lead_slots=lead_slots,
                                 unknown_tail_masks=np.column_stack(unknown_tail_masks) if unknown_tail_masks is not None else None)
    except (TypeError, ValueError, FloatingPointError):
      self.fail(MPC_INVALID_INPUT)
      return
    if fusion_active:
      self._finite_risk_candidates = self.prove_finite_physical_risk(radarstate, now_ns=now_ns, reference_ns=reference_ns)

    try:
      self.observe_obstacles(x_obstacles, sources, ordinary_continuation=ordinary_continuation)
    except Exception:
      self.diagnostic_obstacle_failures += 1

    self.continuation_problem = self._capture_continuation_problem(reference_ns=reference_ns, now_ns=now_ns)
    self.run()
    if self.solution_status != 0:
      return
    if fusion_active:
      self.check_fusion_solution()
      if self.solution_status != 0:
        self.crash_cnt = 0
        self._fusion_fcw_counts = {}
        self._fusion_fcw_epochs = {}
        return
      self.update_fusion_fcw(projections, now_ns=now_ns)
    else:
      self.check_effective_rest_goal()
      if (radarstate.leadOne.present and np.any(lead_xv_0[FCW_IDXS,0] - self.x_sol[FCW_IDXS,0] < CRASH_DISTANCE) and
          radarstate.leadOne.modelProb > 0.9):
        self.crash_cnt += 1
      else:
        self.crash_cnt = 0
    self.record_object_influence()

  def check_fusion_solution(self):
    """Validate a native candidate, with at most one same-problem refinement.

    SQP_RTI's first native success can leave a continuous-time residual that
    the independent occupancy checks reject. A second RTI step may improve
    that iterate without changing inputs, costs, bounds or their tolerances.
    It never retries invalid input/native failure, relaxes a physical bound,
    or publishes either candidate before every original check succeeds.
    """
    def validate():
      failure = self.reintegrate_occupancy_precision()
      if failure:
        return failure
      self.check_effective_rest_goal()
      if self.solution_status != 0:
        return ''  # A rejected rest goal keeps its own failure and mitigation.
      return self.occupancy_trajectory_error(self.x_sol, self.j_solution,
                                             tolerance=QP_FEASIBILITY_TOL, post_native=True)

    failure = validate()
    refinement = None
    if (failure and self.solution_status == 0 and self.diagnostic_last_solver_return == 0
        and self.finite_occupancy_physics_required
        and failure not in ('invalid_finite_occupancy_trajectory',
                            'finite_occupancy_postcheck_requires_native_success',
                            'finite_occupancy_reintegration_requires_native_success')):
      refinement = {'firstFailure': failure, 'attempts': 2}
      first_solve_time = self.solve_time
      self.diagnostic_occupancy_projection = None
      # run() reuses exactly the parameters, reference costs and hard rows
      # already installed this tick. Do not call update(), reset() or rebuild
      # the reference: that would change the problem or renew evidence.
      self.run()
      self.solve_time += first_solve_time
      failure = validate() if self.solution_status == 0 else self.failure_reason

    if failure and self.solution_status == 0:
      self.fail(MPC_OCCUPANCY_UNSATISFIED)
      self.failure_reason = failure
    if refinement is not None:
      refinement.update(finalStatus=int(self.solution_status), finalFailure=self.failure_reason,
                        totalSolveTime=float(self.solve_time))
      self.diagnostic_occupancy_refinement = refinement
      cloudlog.event('longMpcOccupancyRefinement', **refinement)

  def observe_obstacles(self, obstacles, sources, *, ordinary_continuation=False):
    """Copy a bounded constraint summary; never rebuild a reference or solve."""
    if not self.diagnostic_obstacles_enabled:
      return
    if obstacles.shape != (N+1, len(sources)) or not 2 <= len(sources) <= MAX_CONTROL_OBJECTS + 4:
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
    }
    if self._obstacle_identities:
      self.diagnostic_last_obstacle_summary.update(
        objectIdentities=tuple(self._obstacle_identities[index] for index in selected),
        objectCount=len(self._fusion_projections),
        occupancy=tuple((projection.identity, projection.conflict_start, projection.conflict_end,
                         tuple(bool(value) for value in projection.occupied)) for projection in self._fusion_projections),
        occupancySupport=tuple((projection.identity, projection.conflict_support_version, projection.hard_conflict_end)
                               for projection in self._fusion_projections),
        sourceRule='first_minimum_by_bounded_object_order;extra_object_enum_is_lead0',
      )

  def fail(self, status):
    continuation_problem = self.continuation_problem if status in (4, MPC_OCCUPANCY_UNSATISFIED) else None
    failed_goal = self.rest_goal
    effective_node = self.effective_rest_node
    goal_diagnostic = self.diagnostic_rest_goal
    occupancy_diagnostic = self.diagnostic_occupancy_bounds
    occupancy_projection = self.diagnostic_occupancy_projection
    passing_diagnostic = self.diagnostic_passing
    finite_risk = self._finite_risk_candidates if status != MPC_INVALID_INPUT else ()
    solve_time = self.solve_time
    last_cloudlog_t = self.last_cloudlog_t
    timestamp = time.monotonic()
    if timestamp > last_cloudlog_t + 5.0:
      last_cloudlog_t = timestamp
      cloudlog.warning(f"Long mpc reset, solution_status: {status}")
    self.reset()
    self.continuation_problem = continuation_problem
    self.solution_status = status
    self.last_solver_status = status
    self.failure_reason = {MPC_INVALID_INPUT: 'invalid_input', MPC_INVALID_SOLUTION: 'nonfinite_solution',
                           MPC_SOLVER_ERROR: 'solver_exception',
                           MPC_OCCUPANCY_UNSATISFIED: 'finite_occupancy_unsatisfied'}.get(status, 'solver_status')
    self.diagnostic_occupancy_bounds = occupancy_diagnostic
    self.diagnostic_occupancy_projection = occupancy_projection
    self.diagnostic_passing = passing_diagnostic
    if finite_risk:
      self.finite_risk_evidence = finite_risk
      self.finite_risk_braking_accel = ACCEL_MIN
    if occupancy_diagnostic and status > 0:
      self.failure_reason = 'finite_occupancy_solver_failure'
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
    if self._costs_pending:
      # Direct run callers retain the base-cost contract. In normal update,
      # the complete reference already installed its costs exactly once.
      self.install_cost_weights(self.stage_cost_weights())
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
