import math
from copy import deepcopy
from dataclasses import fields, is_dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from openpilot.cereal import messaging
from openpilot.common.longitudinal_logging import CHECK_FLAGS, PLANNER_INPUTS
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, MpcReference, MPC_INVALID_INPUT, MPC_SOLVER_ERROR
from openpilot.selfdrive.controls.lib.longitudinal_observation import PlannerObservation, TRACE_STATUS_COUNTERS
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner, MIN_ALLOW_THROTTLE_SPEED, PitchBiasEstimator
from openpilot.selfdrive.controls.lib.longitudinal_trace import LongitudinalTrace
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_state
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import CruiseTarget, LongitudinalPlannerSP


def make_planner(*, diagnostics=None, speed=20.0, trace=None, platform=CAR.HONDA_ACCORD, stock=False, interceptor=False):
  params = CarInterface.get_non_essential_params(platform)
  params.openpilotLongitudinalControl = not stock
  extension = CarInterface.get_non_essential_params_sp(params, platform)
  extension.enableGasInterceptor = interceptor
  return LongitudinalPlanner(params, extension, init_v=speed, diagnostics=diagnostics, trace=trace)


def update_at_fixture_clock(planner, state, *, now_ns=None):
  """Advance only the caller clock; source times and payloads stay untouched."""
  if now_ns is None:
    now_ns = getattr(planner, '_fixture_decision_ns', 955_000_000) + 50_000_000
  planner._fixture_decision_ns = now_ns
  return planner.update(state, now_ns=now_ns)


def snapshot_value(value):
  """Compare complete state by value, including typed evidence and arrays."""
  if isinstance(value, np.ndarray):
    return ('ndarray', value.dtype.str, value.shape, value.flags.writeable, value.tolist())
  if is_dataclass(value):
    return (type(value).__qualname__, {field.name: snapshot_value(getattr(value, field.name)) for field in fields(value)})
  if isinstance(value, dict):
    return {name: snapshot_value(item) for name, item in value.items()}
  if isinstance(value, (tuple, list)):
    return type(value)(snapshot_value(item) for item in value)
  return deepcopy(value)


def reference_snapshot(reference):
  return {
    name: snapshot_value(vars(value) if name == 'model_reference_adapter' else value)
    for name, value in vars(reference).items() if name != 'planner'
  }


class Publisher:
  def __init__(self):
    self.messages = {}
    self.order = []

  def send(self, service, message):
    self.order.append(service)
    self.messages[service] = messaging.log_from_bytes(message.to_bytes())


def publish(planner, state):
  publisher = Publisher()
  planner.publish(state, publisher)
  return publisher


def diagnostics(planner, state):
  return publish(planner, state).messages["longitudinalPlan"].longitudinalPlan.diagnostics


def control_snapshot(planner):
  result = {
    name: getattr(planner, name)
    for name in ("output_a_target", "output_should_stop", "allow_throttle", "fcw", "a_cruise", "output_v_target", "cruise_target")
  }
  result.update(
    filter_speed=planner.v_desired_filter.x, filter_initialized=planner.v_desired_filter.initialized,
    pitch_bias=planner.pitch_bias.filter.x, pitch_initialized=planner.pitch_bias.filter.initialized,
    solution_status=planner.mpc.solution_status, source=planner.mpc.source, crash_count=planner.mpc.crash_cnt,
  )
  for name in ("x0", "x_sol", "u_sol", "a_prev", "params", "yref", "v_solution", "a_solution", "j_solution"):
    result["mpc_" + name] = getattr(planner.mpc, name).tolist()
  for name in ("v_desired_trajectory", "a_desired_trajectory", "j_desired_trajectory"):
    result[name] = getattr(planner, name).tolist()
  if planner.unified:
    result["intent"] = planner.motion_intent
    result["reference"] = reference_snapshot(planner.reference_planner)
  else:
    for name in ("stop_latched", "creep_t", "creep_stop", "creep_total_t"):
      result[name] = getattr(planner, name)
  return result


def control_messages(publisher):
  result = {service: event.to_dict() for service, event in publisher.messages.items()}
  for event in result.values():
    event.pop("logMonoTime")
  plan = result["longitudinalPlan"]["longitudinalPlan"]
  plan.pop("diagnostics", None)
  plan.pop("processingDelay")
  plan.pop("solverExecutionTime")
  result["longitudinalPlanSP"]["longitudinalPlanSP"].pop("longitudinalPlanMonoTime", None)
  return result


def set_field(state, path, value):
  service, *fields = path.split(".")
  target = state[service]
  for field in fields[:-1]:
    target = getattr(target, field)
  setattr(target, fields[-1], value)


def test_regular_logging_defaults_on_for_supported_vehicle():
  planner = make_planner()
  state = make_state()
  update_at_fixture_clock(planner, state)
  observed = diagnostics(planner, state)
  assert observed.context.version == 1
  assert observed.context.producer == "planner"
  assert observed.context.sequence == 1
  assert observed.context.session > 0
  assert observed.context.endMonoTime >= observed.context.startMonoTime > 0
  assert observed.unified
  assert observed.updateCompleted
  assert observed.outcome == "completed"
  assert observed.mpcUpdated and observed.solverAttempted and observed.solverReturnKnown
  assert observed.solverReturnStatus == 0
  assert observed.policyTargetKnown and observed.actionHorizonKnown and observed.sampledAccelKnown
  assert observed.initialStateKnown and observed.modelAdmissionKnown
  assert observed.actionHorizon == planner.CP.longitudinalActuatorDelay + DT_MDL
  assert not observed.detailedTraceEnabled


@pytest.mark.parametrize("platform,stock,interceptor", [
  (CAR.HONDA_CIVIC, False, False),
  (CAR.HONDA_CIVIC_BOSCH, False, False),
  (CAR.HONDA_ACCORD, True, False),
  (CAR.HONDA_ACCORD, False, True),
])
def test_regular_logging_respects_vehicle_scope(platform, stock, interceptor):
  planner = make_planner(platform=platform, stock=stock, interceptor=interceptor)
  assert not planner.observation.recorder.enabled


def test_explicit_disabled_observation_has_no_wire_context():
  planner = make_planner(diagnostics=False)
  state = make_state()
  update_at_fixture_clock(planner, state)
  assert planner.observation_attempt is None
  assert planner.observation_result is None
  assert diagnostics(planner, state).context.version == 0


@pytest.mark.parametrize("outcome,path,value", [
  ("motion_owner_unavailable", "controlsState.motionState.version", MOTION_INTENT_VERSION + 1),
  ("motion_owner_unavailable", "controlsState.motionState.enabled", False),
  ("invalid_reference_source_epoch", "carState.aEgo", float("nan")),
  ("invalid_vehicle_values", "carState.vCruise", float("inf")),
  ("invalid_model_shape", "modelV2.velocity.x", [1.0]),
  ("invalid_model_shape", "modelV2.orientationRate.z", [0.0]),
  ("nonfinite_model_values", "modelV2.velocity.x", [float("nan")] * len(ModelConstants.T_IDXS)),
  ("nonfinite_model_values", "modelV2.orientationRate.z", [float("inf")] * len(ModelConstants.T_IDXS)),
  ("nonfinite_pitch", "carControl.orientationNED", [0.0, float("nan"), 0.0]),
  ("nonfinite_throttle_probability", "modelV2.meta.disengagePredictions.gasPressProbs", [1.0, float("nan")]),
  ("invalid_e2e_reference", "modelV2.position.x", [0.0]),
  ("invalid_e2e_reference", "modelV2.acceleration.x", [float("nan")] * len(ModelConstants.T_IDXS)),
])
def test_every_pre_mpc_rejection_has_fresh_known_fields(outcome, path, value):
  planner = make_planner()
  state = make_state(experimental=True)
  update_at_fixture_clock(planner, state)
  solver_calls = planner.mpc.diagnostic_solve_calls
  set_field(state, path, value)
  state.logMonoTime["modelV2"] += 50_000_000
  with patch.object(planner.dec, "active", return_value=False):
    update_at_fixture_clock(planner, state)
  observed = diagnostics(planner, state)
  assert observed.context.sequence == 2
  assert observed.updateCompleted
  assert observed.outcome == outcome
  assert not observed.mpcUpdated and not observed.solverAttempted
  assert not observed.solverReturnKnown
  assert observed.solverCalls == solver_calls
  assert observed.solutionStatus == 0
  assert not observed.initialStateKnown
  assert not observed.actionHorizonKnown and not observed.sampledAccelKnown
  assert observed.constraintKnown and observed.constraintAccel == -0.2
  assert not planner.motion_intent.valid
  identities = {entry.service: entry.monoTime for entry in observed.inputs}
  assert identities["modelV2"] == state.logMonoTime["modelV2"]


@pytest.mark.parametrize("field,value", [("wheelbase", 0.0), ("steerRatio", 0.0), ("longitudinalActuatorDelay", 1.1)])
def test_invalid_vehicle_geometry_or_delay_is_observed_without_solver(field, value):
  planner = make_planner()
  state = make_state()
  setattr(planner.CP, field, value)
  update_at_fixture_clock(planner, state)
  observed = diagnostics(planner, state)
  assert observed.outcome == "invalid_vehicle_values"
  assert not observed.solverAttempted


@pytest.mark.parametrize("idle,expected", [(False, "cruise_unset"), (True, "cold_standby")])
def test_unset_cruise_and_cold_standby_are_distinct(idle, expected):
  planner = make_planner(speed=0.0)
  state = make_state(speed=0.0)
  state["carState"].vCruise = V_CRUISE_UNSET
  state["carControl"].enabled = not idle
  state["selfdriveState"].enabled = not idle
  state["controlsState"].longControlState = LongCtrlState.off if idle else LongCtrlState.pid
  update_at_fixture_clock(planner, state)
  observed = diagnostics(planner, state)
  assert observed.outcome == expected
  assert not observed.solverAttempted and not observed.mpcUpdated
  assert not observed.policyTargetKnown and not observed.actionHorizonKnown
  assert not observed.pitchCalled and not observed.pitchLearned
  assert not observed.coastAccelKnown
  assert planner.reference_planner.standby == idle


@pytest.mark.parametrize("target", [(float("nan"), 0.0), (20.0, float("inf"))])
def test_actual_invalid_policy_output_is_preserved_without_recomputation(target):
  planner = make_planner()
  state = make_state()
  with patch.object(LongitudinalPlannerSP, "update_targets", return_value=CruiseTarget(planner.source, *target)) as targets:
    update_at_fixture_clock(planner, state)
  observed = diagnostics(planner, state)
  targets.assert_called_once()
  assert observed.outcome == "nonfinite_policy_target"
  assert observed.policyTargetKnown
  assert not observed.mpcUpdated and not observed.solverAttempted
  assert not math.isfinite(observed.policySpeed if math.isnan(target[0]) else observed.policyAccel)


@pytest.mark.parametrize("unified", [False, True])
def test_mpc_input_rejection_is_not_a_solver_attempt(unified):
  planner = make_planner(diagnostics=True, platform=CAR.HONDA_ACCORD if unified else CAR.HONDA_CIVIC_BOSCH)
  state = make_state()
  update_at_fixture_clock(planner, state)
  lead = state["radarState"].leadOne
  lead.present = True
  lead.dRel = 20.0
  lead.vLead = 15.0
  lead.aLeadTau = float("nan")
  update_at_fixture_clock(planner, state)
  observed = diagnostics(planner, state)
  assert observed.outcome == "mpc_invalid_input"
  assert observed.mpcUpdated and not observed.solverAttempted
  assert observed.initialStateKnown and observed.initialSpeed > 0.0
  assert observed.solutionStatus == MPC_INVALID_INPUT
  assert not observed.solverReturnKnown
  assert observed.solverCalls == 1
  assert observed.solverResets == 2
  assert not observed.sampledAccelKnown


@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("failure,status,return_known", [("status", 4, True), ("exception", MPC_SOLVER_ERROR, False)])
def test_actual_solver_failures_survive_control_reset(unified, failure, status, return_known):
  planner = make_planner(diagnostics=True, platform=CAR.HONDA_ACCORD if unified else CAR.HONDA_CIVIC_BOSCH)
  state = make_state()
  state["carState"].aEgo = -0.4
  solver = Mock(wraps=planner.mpc.solver)
  if failure == "status":
    solver.solve.return_value = 4
  else:
    solver.solve.side_effect = RuntimeError("solver failure")
  with patch.object(planner.mpc, "solver", solver):
    update_at_fixture_clock(planner, state)
  observed = diagnostics(planner, state)
  assert observed.mpcUpdated and observed.solverAttempted
  assert observed.solverCalls == solver.solve.call_count == 1
  assert observed.solverResets == 2
  assert observed.solutionStatus == status
  assert observed.solverReturnKnown == return_known
  if return_known:
    assert observed.solverReturnStatus == 4
  assert observed.initialSpeed == 20.0
  assert tuple(planner.mpc.diagnostic_last_x0)[1] == 20.0
  assert np.all(planner.mpc.x0 == 0.0)
  assert not observed.sampledAccelKnown and not observed.actionHorizonKnown


def test_nonfinite_sample_is_not_replaced_with_invalidated_reference_zero():
  planner = make_planner()
  state = make_state()
  with patch("openpilot.selfdrive.controls.lib.unified_longitudinal_planner.get_accel_from_plan", return_value=float("nan")) as sample:
    update_at_fixture_clock(planner, state)
  observed = diagnostics(planner, state)
  sample.assert_called_once()
  assert observed.outcome == "nonfinite_sampled_accel"
  assert observed.solverAttempted and observed.solverReturnKnown
  assert observed.solverReturnStatus == 0
  assert observed.actionHorizonKnown and observed.sampledAccelKnown
  assert math.isnan(observed.sampledAccel)
  assert planner.reference_planner.reference_accel == 0.0


def test_unhealthy_input_completion_does_not_masquerade_as_admitted_intent():
  planner = make_planner()
  state = make_state()
  state.valid["modelV2"] = False
  state.all_checks = lambda service_list=None: all(state.valid[service] for service in (state.valid if service_list is None else service_list))
  update_at_fixture_clock(planner, state)
  observed = diagnostics(planner, state)
  assert observed.outcome == "completed_unhealthy_inputs"
  assert observed.solverAttempted and not planner.motion_intent.valid
  model_input = next(entry for entry in observed.inputs if entry.service == "modelV2")
  assert model_input.knownFlags & CHECK_FLAGS["valid"]
  assert not model_input.flags & CHECK_FLAGS["valid"]


@pytest.mark.parametrize("speed,orientation,called,learned", [
  (20.0, [], False, False),
  (20.0, [0.0, 0.04, 0.0], True, True),
  (0.0, [0.0, 0.04, 0.0], True, False),
  (MIN_ALLOW_THROTTLE_SPEED, [0.0, 0.04, 0.0], True, False),
])
@pytest.mark.parametrize("unified", [False, True])
def test_pitch_observation_counts_actual_filter_branches(speed, orientation, called, learned, unified):
  planner = make_planner(diagnostics=True, platform=CAR.HONDA_ACCORD if unified else CAR.HONDA_CIVIC_BOSCH, speed=speed)
  state = make_state(speed=speed)
  state["carControl"].orientationNED = orientation
  with patch.object(planner.pitch_bias.filter, "update", wraps=planner.pitch_bias.filter.update) as filter_update:
    planner.update(state, now_ns=state.logMonoTime["modelV2"] + 5_000_000)
  observed = diagnostics(planner, state)
  assert observed.pitchSnapshotKnown
  assert observed.pitchCalled == called
  assert observed.pitchLearned == learned
  assert observed.pitchCalls == int(called)
  assert observed.pitchLearnCalls == filter_update.call_count == int(learned)
  assert observed.pitchBiasBefore == 0.0
  assert observed.pitchBiasAfter == planner.pitch_bias.filter.x


def test_pitch_estimator_call_counter_does_not_claim_nonfinite_learning():
  estimator = PitchBiasEstimator(DT_MDL)
  with patch.object(estimator.filter, "update", wraps=estimator.filter.update) as update_filter:
    assert estimator.update(float("nan"), 20.0) == 0.0
    assert estimator.update(0.1, 0.0) == 0.0
    assert estimator.update(0.01, 20.0) == 0.01
  assert estimator.diagnostic_calls == 3
  assert estimator.diagnostic_learn_calls == update_filter.call_count == 1


def test_input_snapshot_keeps_uint64_identity_and_unknown_health():
  planner = make_planner()
  state = make_state()
  for service in PLANNER_INPUTS:
    if service not in state:
      state[service] = getattr(messaging.new_message(service), service)
    state.logMonoTime[service] = (1 << 60) + 97 + PLANNER_INPUTS.index(service)
  state.frame = 37
  state.alive = {"modelV2": True, "carState": False}
  original = dict(state.logMonoTime)
  update_at_fixture_clock(planner, state, now_ns=max(original.values()) + 5_000_000)
  state.logMonoTime = {service: mono + 2 for service, mono in state.logMonoTime.items()}
  observed = diagnostics(planner, state)
  assert observed.context.frameKnown and observed.context.frame == 37
  entries = {entry.service: entry for entry in observed.inputs}
  assert set(entries) == set(PLANNER_INPUTS)
  assert {service: entry.monoTime for service, entry in entries.items()} == original
  assert all(entry.idKnown and str(entry.kind) == "consumerSnapshot" for entry in entries.values())
  assert entries["modelV2"].knownFlags & CHECK_FLAGS["alive"]
  assert entries["modelV2"].flags & CHECK_FLAGS["alive"]
  assert entries["carState"].knownFlags & CHECK_FLAGS["alive"]
  assert not entries["carState"].flags & CHECK_FLAGS["alive"]
  assert not entries["liveMapDataSP"].knownFlags & CHECK_FLAGS["alive"]


@pytest.mark.parametrize("delay_ns", [750_000_001, -123_000_007])
def test_plan_sp_uses_actual_main_envelope_and_delay_has_consistent_units(delay_ns):
  planner = make_planner()
  state = make_state()
  state.logMonoTime["modelV2"] = (1 << 60) + 13
  update_at_fixture_clock(planner, state)
  main_mono = state.logMonoTime["modelV2"] + delay_ns
  monotimes = iter([main_mono, main_mono + 31])
  original_new_message = messaging.new_message

  def timed_message(service, **options):
    options['logMonoTime'] = next(monotimes)
    return original_new_message(service, **options)

  with patch("openpilot.selfdrive.controls.lib.longitudinal_planner.messaging.new_message", side_effect=timed_message):
    publisher = publish(planner, state)
  main = publisher.messages["longitudinalPlan"]
  extension = publisher.messages["longitudinalPlanSP"]
  assert publisher.order == ["longitudinalPlan", "longitudinalPlanSP"]
  assert main.logMonoTime == main_mono
  assert extension.longitudinalPlanSP.longitudinalPlanMonoTime == main_mono
  assert extension.logMonoTime != main_mono
  assert main.longitudinalPlan.processingDelay == pytest.approx(delay_ns / 1e9, abs=1e-7)


def test_standalone_sp_publication_keeps_unknown_parent_as_zero():
  planner = make_planner()
  state = make_state()
  update_at_fixture_clock(planner, state)
  publisher = Publisher()
  planner.publish_longitudinal_plan_sp(state, publisher)
  assert publisher.messages["longitudinalPlanSP"].longitudinalPlanSP.longitudinalPlanMonoTime == 0


@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("experimental", [False, True])
def test_logging_on_off_preserves_control_messages_and_state(unified, experimental):
  baseline = make_planner(platform=CAR.HONDA_ACCORD if unified else CAR.HONDA_CIVIC_BOSCH, diagnostics=False)
  observed = make_planner(platform=CAR.HONDA_ACCORD if unified else CAR.HONDA_CIVIC_BOSCH, diagnostics=True)
  for frame in range(30):
    speed = max(0.0, 20.0 - frame * 0.5)
    state = make_state(speed=speed, experimental=experimental, target=18.0)
    state["carState"].aEgo = -0.3
    state["carControl"].orientationNED = [0.0, 0.04, 0.0]
    state["controlsState"].forceDecel = 10 <= frame < 13
    state["carState"].gasPressed = frame == 18
    state["carState"].brakePressed = frame == 22
    state.logMonoTime = dict.fromkeys(state.logMonoTime, 2_000_000_000 + frame * 50_000_000)
    now_ns = state.logMonoTime["modelV2"] + 5_000_000
    baseline.update(state, now_ns=now_ns)
    observed.update(state, now_ns=now_ns)
    assert baseline.pitch_bias.diagnostic_learn_calls == observed.pitch_bias.diagnostic_learn_calls == frame + 1
    assert control_snapshot(baseline) == control_snapshot(observed)
    assert control_messages(publish(baseline, state)) == control_messages(publish(observed, state))


@pytest.mark.parametrize("method", ["begin", "record", "mpc_started", "finish", "publish"])
@pytest.mark.parametrize("unified", [False, True])
def test_observer_failure_is_output_neutral_and_counted(method, unified):
  baseline = make_planner(platform=CAR.HONDA_ACCORD if unified else CAR.HONDA_CIVIC_BOSCH, diagnostics=False)
  observed = make_planner(diagnostics=True, platform=CAR.HONDA_ACCORD if unified else CAR.HONDA_CIVIC_BOSCH)
  state = make_state()
  update_at_fixture_clock(baseline, state)
  with patch.object(observed.observation, method, side_effect=RuntimeError("observer unavailable")):
    update_at_fixture_clock(observed, state)
    output = publish(observed, state)
  assert control_snapshot(baseline) == control_snapshot(observed)
  assert control_messages(publish(baseline, state)) == control_messages(output)
  assert observed.observation.recorder.errors > 0
  if method in ("begin", "finish", "publish"):
    assert output.messages["longitudinalPlan"].longitudinalPlan.diagnostics.context.version == 0


def test_failed_begin_never_reuses_a_previous_unpublished_attempt():
  planner = make_planner()
  state = make_state()
  update_at_fixture_clock(planner, state)
  state.logMonoTime["modelV2"] += 50_000_000
  with patch.object(planner.observation, "begin", side_effect=RuntimeError("new snapshot unavailable")):
    update_at_fixture_clock(planner, state)
  assert planner.observation_attempt is None
  assert planner.observation_result is None
  assert diagnostics(planner, state).context.version == 0


@pytest.mark.parametrize("unified", [False, True])
def test_original_control_exception_is_preserved_with_failed_observer(unified):
  planner = make_planner(diagnostics=True, platform=CAR.HONDA_ACCORD if unified else CAR.HONDA_CIVIC_BOSCH)
  failure = RuntimeError("original control exception")
  with patch.object(LongitudinalPlannerSP, "update", side_effect=failure), \
       patch.object(planner.observation, "record", side_effect=ValueError("diagnostic exception")):
    with pytest.raises(RuntimeError) as raised:
      update_at_fixture_clock(planner, make_state())
  assert raised.value is failure
  assert not planner.observation_result["values"]["updateCompleted"]
  assert planner.observation_result["values"]["outcome"] == "control_exception"
  assert not planner.observation_result["values"]["solverAttempted"]


def test_recorder_initialization_failure_does_not_prevent_control_startup():
  with patch("openpilot.selfdrive.controls.lib.longitudinal_observation.ObservationRecorder", side_effect=OSError("manifest unavailable")):
    planner = make_planner()
  assert planner.observation is None
  state = make_state()
  update_at_fixture_clock(planner, state)
  assert planner.motion_intent.valid
  assert diagnostics(planner, state).context.version == 0


class TraceProbe:
  def __init__(self):
    self.sequence = 0
    self.dropped = 0
    self.failures = 0
    self.calls = []

  def begin_attempt(self, planner, state):
    self.sequence += 1
    self.calls.append(("begin_attempt", state.logMonoTime["modelV2"]))

  def finish_attempt(self, planner):
    self.calls.append(("finish_attempt", planner.observation_result["values"]["outcome"]))

  def begin(self, *arguments):
    self.calls.append(("begin",))

  def finish(self, *arguments):
    self.calls.append(("finish",))

  def finish_unified(self, *arguments):
    self.calls.append(("finish_unified",))

  def publish(self, mono):
    self.calls.append(("publish", mono))

  def fail(self):
    self.failures += 1


def test_trace_attempt_hooks_cover_early_return_before_historical_begin():
  trace = TraceProbe()
  planner = make_planner(trace=trace)
  state = make_state()
  state["controlsState"].motionState.version = 0
  update_at_fixture_clock(planner, state)
  publisher = publish(planner, state)
  assert trace.calls == [
    ("begin_attempt", state.logMonoTime["modelV2"]),
    ("finish_attempt", "motion_owner_unavailable"),
    ("publish", publisher.messages["longitudinalPlan"].logMonoTime),
  ]
  observed = publisher.messages["longitudinalPlan"].longitudinalPlan.diagnostics
  assert not observed.detailedTraceEnabled and observed.traceSequence == 1
  assert not observed.traceStatusKnown
  assert observed.traceFailures == 0


@pytest.mark.parametrize("method", ["begin_attempt", "finish_attempt", "begin", "finish_unified", "publish"])
def test_trace_hook_exception_cannot_replace_control_output(method):
  trace = TraceProbe()
  planner = make_planner(trace=trace)
  baseline = make_planner(diagnostics=False)
  state = make_state()
  update_at_fixture_clock(baseline, state)
  with patch.object(trace, method, side_effect=RuntimeError("trace unavailable")):
    update_at_fixture_clock(planner, state)
    publisher = publish(planner, state)
  assert trace.failures == 1
  assert control_snapshot(planner) == control_snapshot(baseline)
  assert control_messages(publisher) == control_messages(publish(baseline, state))


@pytest.mark.parametrize("enabled", [False, True])
def test_mpc_details_are_copied_only_when_enabled_without_extra_solver_reads(enabled):
  mpc = LongitudinalMpc()
  mpc.diagnostic_details_enabled = enabled
  mpc.set_cur_state(20.0, -0.3)
  absent = SimpleNamespace(present=False, modelProb=0.0)
  solver = Mock(wraps=mpc.solver)
  with patch.object(mpc, "solver", solver):
    mpc.update(SimpleNamespace(leadOne=absent, leadTwo=absent), reference=MpcReference(22.0))
  assert solver.solve.call_count == 1
  assert solver.get_stats.call_count == 1
  assert solver.get.call_count == len(mpc.x_sol) + len(mpc.u_sol)
  assert mpc.diagnostic_solve_calls == 1
  assert mpc.diagnostic_last_solver_return == 0
  assert mpc.diagnostic_last_x0 == (0.0, 20.0, -0.3)
  if enabled:
    assert mpc.diagnostic_last_parameters == mpc.params.tolist()
    assert mpc.diagnostic_last_yref == mpc.yref.tolist()
    assert mpc.diagnostic_last_a_prev == [0.0] * len(mpc.a_prev)
    mpc.params[:] = 123.0
    mpc.yref[:] = 456.0
    assert mpc.diagnostic_last_parameters[0][0] != 123.0
    assert mpc.diagnostic_last_yref[0][0] == 0.0
  else:
    assert mpc.diagnostic_last_parameters is None
    assert mpc.diagnostic_last_yref is None
    assert mpc.diagnostic_last_a_prev is None


def test_mpc_failure_reset_preserves_attempted_problem_not_zero_sentinel():
  mpc = LongitudinalMpc()
  mpc.diagnostic_details_enabled = True
  mpc.set_cur_state(20.0, -1.1)
  absent = SimpleNamespace(present=False, modelProb=0.0)
  solver = Mock(wraps=mpc.solver)
  solver.solve.return_value = 4
  with patch.object(mpc, "solver", solver):
    mpc.update(SimpleNamespace(leadOne=absent, leadTwo=absent), reference=MpcReference(22.0))
  assert mpc.diagnostic_solve_calls == 1
  assert mpc.diagnostic_reset_calls == 2
  assert mpc.diagnostic_last_solver_return == 4
  assert mpc.diagnostic_last_x0 == (0.0, 20.0, -1.1)
  assert mpc.diagnostic_last_parameters[0][2] > 0.0
  assert np.all(mpc.params == 0.0)
  mpc.reset()
  assert mpc.diagnostic_solve_calls == 1 and mpc.diagnostic_reset_calls == 3
  assert mpc.diagnostic_last_x0 == (0.0, 20.0, -1.1)


def test_observation_factory_is_explicitly_fail_open():
  with patch.object(PlannerObservation, "__init__", side_effect=RuntimeError("observer unavailable")):
    assert PlannerObservation.create(None, None, enabled=True) is None


def test_trace_disabled_at_startup_is_preserved_in_regular_plan(tmp_path):
  params = SimpleNamespace(carFingerprint=CAR.HONDA_ACCORD, openpilotLongitudinalControl=True)
  marker = tmp_path / "trace"
  marker.touch()
  with patch("openpilot.selfdrive.controls.lib.longitudinal_trace.threading.Thread.start", side_effect=RuntimeError("writer unavailable")):
    trace = LongitudinalTrace.create(params, marker)
  assert trace is not None and not trace.enabled
  planner = make_planner(trace=trace)
  state = make_state()
  update_at_fixture_clock(planner, state)
  observed = diagnostics(planner, state)
  assert observed.context.version == 1 and observed.traceStatusKnown
  assert not observed.detailedTraceEnabled and observed.detailedTraceDisabled
  assert not observed.traceClosed
  assert observed.traceStartupFailures == observed.traceFailures == 1
  assert observed.traceWriterFailures == 0
  assert observed.traceDisabledAttempts == 1
  assert observed.traceAcceptedSamples == 0
  assert planner.motion_intent.valid


def test_trace_status_copies_real_counters_once_at_main_publication():
  trace = LongitudinalTrace(emit=lambda *arguments, **keywords: None)
  for index, attribute in enumerate((
    "dropped", "writer_failures", "serialization_failures", "transport_failures", "capture_failures", "startup_failures",
    "serialization_dropped", "transport_dropped", "disabled_dropped", "transport_accepted_samples",
    "unpublished", "close_timeouts", "disabled_attempts",
  ), start=1):
    setattr(trace, attribute, index)
  trace.sequence = (1 << 60) + 13
  trace.transport_accepted_samples = (1 << 60) + 17
  trace.queue.extend([None, None, None])
  planner = make_planner(trace=trace)
  state = make_state()
  update_at_fixture_clock(planner, state)
  before = trace.status()
  with patch.object(trace, "status", wraps=trace.status) as capture:
    observed = diagnostics(planner, state)
  capture.assert_called_once()
  assert observed.traceStatusKnown and observed.detailedTraceEnabled
  assert not observed.detailedTraceDisabled and not observed.traceClosed
  assert observed.traceSequence == int(before["sequence"]) == (1 << 60) + 14
  for source, field in TRACE_STATUS_COUNTERS.items():
    assert getattr(observed, field) == before[source]
  assert observed.traceFailures == trace.failures
  assert observed.traceDropped == before["queue_dropped"] == 1
  assert observed.traceQueueDepth == 3
  assert len(trace.queue) == 4
  assert observed.traceAcceptedSamples == (1 << 60) + 17


@pytest.mark.parametrize("closed,disabled", [(False, False), (False, True), (True, False), (True, True)])
def test_trace_enabled_closed_and_disabled_are_separate_observations(closed, disabled):
  trace = LongitudinalTrace()
  trace.closed = closed
  trace.disabled = disabled
  planner = make_planner(trace=trace)
  state = make_state()
  update_at_fixture_clock(planner, state)
  observed = diagnostics(planner, state)
  assert observed.traceStatusKnown
  assert observed.traceClosed == closed
  assert observed.detailedTraceDisabled == disabled
  assert observed.detailedTraceEnabled == (not closed and not disabled)


@pytest.mark.parametrize("missing", [
  *TRACE_STATUS_COUNTERS, "session", "sequence", "disabled", "closed", "queue_depth", "transport_accepted_batches",
])
def test_incomplete_trace_status_never_becomes_known(missing):
  trace = LongitudinalTrace()
  planner = make_planner(trace=trace)
  state = make_state()
  update_at_fixture_clock(planner, state)
  incomplete = trace.status()
  del incomplete[missing]
  with patch.object(trace, "status", return_value=incomplete):
    observed = diagnostics(planner, state)
  assert observed.context.version == 1
  assert not observed.traceStatusKnown
  assert observed.solverAttempted and observed.updateCompleted


@pytest.mark.parametrize("field,value", [
  ("queue_dropped", -1),
  ("writer_failures", True),
  ("transport_accepted_samples", float(1 << 60)),
  ("queue_depth", 1 << 32),
  ("serialization_dropped", 1 << 64),
  ("disabled", 0),
  ("closed", "false"),
  ("sequence", "1.0"),
  ("session", None),
  ("transport_accepted_batches", -1),
])
def test_malformed_trace_status_cannot_drop_the_regular_diagnostic(field, value):
  trace = LongitudinalTrace()
  planner = make_planner(trace=trace)
  state = make_state()
  update_at_fixture_clock(planner, state)
  malformed = dict(trace.status(), **{field: value})
  with patch.object(trace, "status", return_value=malformed):
    observed = diagnostics(planner, state)
  assert observed.context.version == 1
  assert not observed.traceStatusKnown
  assert observed.updateCompleted and observed.solverReturnKnown


@pytest.mark.parametrize("status", [None, {}, {"disabled": False}])
def test_old_fake_trace_with_missing_attributes_does_not_claim_status(status):
  trace = SimpleNamespace(sequence=7, dropped=2, failures=1, status=lambda: status)
  planner = make_planner(trace=trace)
  state = make_state()
  update_at_fixture_clock(planner, state)
  observed = diagnostics(planner, state)
  assert observed.context.version == 1
  assert not observed.detailedTraceEnabled and not observed.traceStatusKnown
  assert observed.traceSequence == 7
  assert observed.traceDropped == 2
  assert observed.traceFailures == 1


def test_absent_trace_keeps_status_unknown_without_diagnostic_error():
  planner = make_planner()
  state = make_state()
  update_at_fixture_clock(planner, state)
  errors = planner.observation.recorder.errors
  observed = diagnostics(planner, state)
  assert not observed.detailedTraceEnabled and not observed.detailedTraceDisabled
  assert not observed.traceStatusKnown
  assert observed.context.errors == errors


def test_throwing_trace_status_preserves_regular_diagnostics_and_control():
  trace = LongitudinalTrace()
  planner = make_planner(trace=trace)
  baseline = make_planner(diagnostics=False)
  state = make_state()
  update_at_fixture_clock(baseline, state)
  update_at_fixture_clock(planner, state)
  errors = planner.observation.recorder.errors
  with patch.object(trace, "status", side_effect=RuntimeError("status unavailable")):
    observed = diagnostics(planner, state)
  assert observed.context.version == 1
  assert observed.context.errors == errors + 1
  assert not observed.traceStatusKnown
  assert observed.updateCompleted and observed.solverAttempted
  assert control_snapshot(planner) == control_snapshot(baseline)


def test_disabled_writer_stops_native_detail_copies_and_reenable_restarts_them():
  trace = LongitudinalTrace()
  planner = make_planner(trace=trace)
  state = make_state()
  update_at_fixture_clock(planner, state)
  assert planner.mpc.diagnostic_details_enabled
  parameters = planner.mpc.diagnostic_last_parameters
  assert parameters is not None
  trace.disabled = True
  update_at_fixture_clock(planner, state)
  assert not planner.mpc.diagnostic_details_enabled
  assert planner.mpc.diagnostic_last_parameters is parameters
  trace.disabled = False
  update_at_fixture_clock(planner, state)
  assert planner.mpc.diagnostic_details_enabled
  assert planner.mpc.diagnostic_last_parameters is not parameters


def test_trace_attached_after_initialization_has_native_details_without_regular_observer():
  planner = make_planner(diagnostics=False)
  assert not planner.mpc.diagnostic_details_enabled
  planner.trace = LongitudinalTrace()
  update_at_fixture_clock(planner, make_state())
  assert planner.observation_result is None
  assert planner.mpc.diagnostic_details_enabled
  assert planner.mpc.diagnostic_last_parameters is not None


def test_throwing_trace_enabled_property_disables_only_native_detail_capture():
  class BrokenTrace(TraceProbe):
    @property
    def enabled(self):
      raise RuntimeError("trace enabled unavailable")

  trace = BrokenTrace()
  planner = make_planner(trace=trace)
  baseline = make_planner(diagnostics=False)
  state = make_state()
  update_at_fixture_clock(planner, state)
  update_at_fixture_clock(baseline, state)
  observed = diagnostics(planner, state)
  assert not planner.mpc.diagnostic_details_enabled
  assert planner.mpc.diagnostic_last_parameters is None
  assert trace.failures == 1
  assert observed.context.version == 1 and observed.context.errors > 0
  assert not observed.traceStatusKnown
  assert control_snapshot(planner) == control_snapshot(baseline)
