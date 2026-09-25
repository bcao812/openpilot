from collections.abc import Mapping

from openpilot.cereal import log
from openpilot.common.longitudinal_logging import ANALYSIS_VERSION, ObservationRecorder, PLANNER_INPUTS, attach_diagnostics, capture_inputs, is_uint64, logging_enabled


TRACE_STATUS_COUNTERS = {
  "queue_dropped": "traceDropped",
  "writer_failures": "traceWriterFailures",
  "serialization_failures": "traceSerializationFailures",
  "transport_failures": "traceTransportFailures",
  "capture_failures": "traceCaptureFailures",
  "startup_failures": "traceStartupFailures",
  "serialization_dropped": "traceSerializationDropped",
  "transport_dropped": "traceTransportDropped",
  "disabled_dropped": "traceDisabledDropped",
  "transport_accepted_samples": "traceAcceptedSamples",
  "unpublished": "traceUnpublished",
  "close_timeouts": "traceCloseTimeouts",
  "disabled_attempts": "traceDisabledAttempts",
}
TRACE_FAILURE_COUNTERS = ("capture_failures", "serialization_failures", "transport_failures", "startup_failures", "writer_failures")


def observe_trace(trace, method, *arguments):
  try:
    if method == "begin_attempt":
      planner = arguments[0]
      planner.mpc.diagnostic_details_enabled = False
      if trace is not None:
        planner.mpc.diagnostic_details_enabled = getattr(trace, "enabled", False) is True
    if trace is None:
      return
    callback = getattr(trace, method, None)
    if callback is not None:
      callback(*arguments)
  except Exception:
    try:
      trace.fail()
    except Exception:
      pass


class PlannerObservation:
  @classmethod
  def create(cls, params, params_sp, *, enabled=None):
    try:
      if not logging_enabled("planner", params, params_sp, enabled=enabled):
        return None
      return cls(params, params_sp, enabled=True)
    except Exception:
      return None

  def __init__(self, params, params_sp, *, enabled=None):
    self.recorder = ObservationRecorder("planner", params, params_sp, enabled=enabled)

  def passive_snapshot(self, planner):
    try:
      timestamps = (planner._passive_now_ns, planner._passive_input_times['carState'], planner._passive_input_times['carControl'])
      initialized = planner.pitch_bias.filter.initialized
      if not all(is_uint64(value) for value in timestamps) or not isinstance(initialized, bool):
        raise ValueError("Invalid passive snapshot")
      return {
        "known": True, "consumerMonoTime": int(timestamps[0]), "carStateMonoTime": int(timestamps[1]),
        "carControlMonoTime": int(timestamps[2]), "pitchInitialized": initialized,
        "pitchFilterValue": float(planner.pitch_bias.filter.x),
      }
    except Exception:
      self.recorder.record_error()
      return {"known": False}

  def begin(self, planner, state, *, now_ns=None, passive_version=0):
    sample = self.recorder.begin(getattr(state, "frame", None))
    if sample is None:
      return None
    return {
      "sample": sample,
      "solver_calls_before": planner.mpc.diagnostic_solve_calls,
      "pitch_calls_before": planner.pitch_bias.diagnostic_calls,
      "pitch_learn_calls_before": planner.pitch_bias.diagnostic_learn_calls,
      "values": {
        "inputs": capture_inputs(state, PLANNER_INPUTS),
        "unified": bool(planner.unified),
        "updateCompleted": False,
        "outcome": "started",
        "previousRequest": float(planner.output_a_target),
        "pitchBiasBefore": float(planner.pitch_bias.filter.x),
        "analysis": {
          "version": ANALYSIS_VERSION, "planningMode": int(planner.longitudinal_mode),
          "decisionMonoTimeKnown": is_uint64(now_ns), "decisionMonoTime": int(now_ns) if is_uint64(now_ns) else 0,
          "passiveApiVersion": passive_version, "passiveBefore": self.passive_snapshot(planner),
        },
      },
    }

  def record(self, attempt, **values):
    if attempt is not None:
      attempt["values"].update(values)

  def record_analysis(self, attempt, **values):
    if attempt is not None:
      attempt["values"]["analysis"].update(values)

  def reference_candidates(self, attempt, candidates, cruise_raw, model_accel, model_stop, admitted):
    if attempt is not None:
      mpc_accel, mpc_source, mpc_stop = candidates[0]
      cruise_accel, _, cruise_stop = candidates[1]
      attempt["values"]["analysis"]["referenceCandidates"] = {
        "known": True, "mpcAccel": float(mpc_accel), "mpcSource": mpc_source, "mpcShouldStop": bool(mpc_stop),
        "cruiseRawAccel": float(cruise_raw), "cruiseAccel": float(cruise_accel), "cruiseShouldStop": bool(cruise_stop),
        "e2eAccel": float(model_accel), "e2eShouldStop": bool(model_stop), "e2eAdmitted": bool(admitted),
      }

  def departure(self, attempt, forward_reference, velocity, trajectory_departure, lead_departure, qualified, no_model_risk):
    if attempt is not None:
      attempt["values"]["analysis"]["departure"] = {
        "known": True, "forwardReference": bool(forward_reference), "terminalMpcSpeed": float(velocity[-1]),
        "trajectoryDeparture": bool(trajectory_departure), "leadDeparture": bool(lead_departure),
        "qualified": bool(qualified), "noModelRisk": bool(no_model_risk),
      }

  def mpc_started(self, attempt, mpc):
    if attempt is not None:
      mpc.diagnostic_obstacles_enabled = True
      attempt["obstacle_failures_before"] = mpc.diagnostic_obstacle_failures
      attempt["values"].update(
        mpcUpdated=True, initialStateKnown=True, initialSpeed=float(mpc.x0[1]), initialAccel=float(mpc.x0[2]),
      )

  def finish(self, attempt, planner, completed):
    if attempt is None:
      return None
    values = attempt["values"]
    solver_attempted = planner.mpc.diagnostic_solve_calls > attempt["solver_calls_before"]
    solver_return = planner.mpc.diagnostic_last_solver_return if solver_attempted else None
    values.update(
      updateCompleted=bool(completed),
      solverAttempted=solver_attempted,
      solverCalls=planner.mpc.diagnostic_solve_calls,
      solverResets=planner.mpc.diagnostic_reset_calls,
      solutionStatus=int(planner.mpc.solution_status),
      solverReturnKnown=solver_return is not None,
      solverReturnStatus=int(solver_return) if solver_return is not None else 0,
      pitchSnapshotKnown=True,
      pitchBiasAfter=float(planner.pitch_bias.filter.x),
      pitchCalls=planner.pitch_bias.diagnostic_calls,
      pitchLearnCalls=planner.pitch_bias.diagnostic_learn_calls,
      pitchCalled=planner.pitch_bias.diagnostic_calls > attempt["pitch_calls_before"],
      pitchLearned=planner.pitch_bias.diagnostic_learn_calls > attempt["pitch_learn_calls_before"],
    )
    analysis = values["analysis"]
    analysis["passiveAfter"] = self.passive_snapshot(planner)
    if analysis.get("passiveAttempted"):
      try:
        reason = planner.last_passive_reason
        if not isinstance(reason, str) or len(reason) > 64:
          raise ValueError("Invalid passive reason")
        analysis.update(passiveReasonKnown=True, passiveReason=reason)
      except Exception:
        self.recorder.record_error()
    if values.get("mpcUpdated"):
      try:
        if planner.mpc.diagnostic_obstacle_failures > attempt["obstacle_failures_before"]:
          self.recorder.record_error()
        summary = planner.mpc.diagnostic_last_obstacle_summary
        if summary is not None:
          analysis["obstacleSummary"] = dict(summary)
      except Exception:
        self.recorder.record_error()
    if not completed:
      values["outcome"] = "control_exception"
    elif values["outcome"] == "started":
      values["outcome"] = "completed"
    sample = self.recorder.finish(attempt["sample"])
    return {"sample": sample, "values": dict(values)} if sample is not None else None

  def publish(self, result, message, planner):
    if result is None:
      return
    values = dict(result["values"], **self.trace_status(planner.trace))
    attach_diagnostics(message, "diagnostics", log.LongitudinalPlanDiagnostics, self.recorder, result["sample"], values)

  def trace_status(self, trace):
    values = {"detailedTraceEnabled": False, "traceStatusKnown": False}
    if trace is None:
      return values
    try:
      enabled = getattr(trace, "enabled", None)
      if isinstance(enabled, bool):
        values["detailedTraceEnabled"] = enabled
      attributes = {name: getattr(trace, name, None) for name in ("session", "sequence", "dropped", "failures")}
      for field, attribute in (("traceSequence", "sequence"), ("traceDropped", "dropped"), ("traceFailures", "failures")):
        if is_uint64(attributes[attribute]):
          values[field] = int(attributes[attribute])
      callback = getattr(trace, "status", None)
      if not callable(callback):
        return values
      status = callback()
      if not isinstance(status, Mapping):
        return values
      complete = isinstance(enabled, bool) and all(is_uint64(value) for value in attributes.values())
      complete = complete and status.get("session") == str(attributes["session"]) and status.get("sequence") == str(attributes["sequence"])
      for source, field in TRACE_STATUS_COUNTERS.items():
        value = status.get(source)
        if is_uint64(value):
          values[field] = int(value)
        else:
          complete = False
      for source, field in (("disabled", "detailedTraceDisabled"), ("closed", "traceClosed")):
        value = status.get(source)
        if isinstance(value, bool):
          values[field] = value
        else:
          complete = False
      depth = status.get("queue_depth")
      if is_uint64(depth) and depth < 1 << 32:
        values["traceQueueDepth"] = int(depth)
      else:
        complete = False
      if all(is_uint64(status.get(name)) for name in TRACE_FAILURE_COUNTERS):
        failures = sum(int(status[name]) for name in TRACE_FAILURE_COUNTERS)
        if is_uint64(failures):
          values["traceFailures"] = failures
        else:
          complete = False
      complete = complete and is_uint64(status.get("transport_accepted_batches"))
      complete = complete and enabled == (not status.get("disabled") and not status.get("closed"))
      values["traceStatusKnown"] = complete
    except Exception:
      self.recorder.record_error()
    return values
