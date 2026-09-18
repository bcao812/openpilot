from openpilot.cereal import log
from openpilot.common.longitudinal_logging import CONTROL_INPUTS, ObservationRecorder, attach_diagnostics, capture_inputs, is_uint64


def _stored_identity(name, timestamp):
  valid = is_uint64(timestamp)
  return {
    "service": name, "monoTime": int(timestamp) if valid else 0,
    "idKnown": valid and timestamp > 0, "knownFlags": 0, "flags": 0, "kind": "consumerSnapshot",
  }


def _creep_snapshot(sample):
  if sample is None:
    return {"known": False}
  return {"known": True, "sourceMonoTimes": list(sample.source_times_ns),
          "measuredSpeed": float(sample.measured_speed), "reset": bool(sample.reset)}


def supervisor_snapshot(controller):
  """Copy stored state only: observers must never run eligibility or rearm methods."""
  supervisor = controller.motion_supervisor
  if supervisor is None:
    return {"known": False}
  pending = supervisor._pending_off_creep
  return {
    "known": True,
    "phase": supervisor.phase.text,
    "reasons": int(supervisor.reasons),
    "degraded": bool(supervisor.degraded),
    "waitingForFreshInput": bool(supervisor.waiting_for_fresh_input),
    "resolvedStop": bool(supervisor.resolved_should_stop),
    "creepT": float(supervisor.creep_t),
    "creepTotalT": float(supervisor.creep_total_t),
    "creepV": float(supervisor.creep_v),
    "creepWindowNs": int(supervisor._creep_window_ns),
    "creepTotalNs": int(supervisor._creep_total_ns),
    "creepOffCreditNs": int(supervisor._creep_off_credit_ns),
    "creepWindowOffCreditNs": int(supervisor._creep_window_off_credit_ns),
    "creepAutoStartSpeed": float(supervisor._creep_auto_start_speed),
    "movingLeadT": float(supervisor.moving_lead_t),
    "releaseLeadT": float(supervisor.release_lead_t),
    "requireNewInput": bool(supervisor._require_new_input),
    "rearmAfterFault": bool(supervisor._rearm_after_fault),
    "ids": [_stored_identity(name, timestamp) for name, timestamp in (
      ("lastPlan", supervisor._last_plan_time_ns), ("lastModel", supervisor._last_model_time_ns),
      ("lastNow", supervisor._last_now_ns), ("lastBrakingPlan", supervisor._last_braking_plan_time_ns),
      ("lastBrakingModel", supervisor._last_braking_model_time_ns), ("lastCreepObservation", supervisor._last_creep_observation_ns),
      ("rearmPlan", supervisor._rearm_plan_time_ns), ("rearmModel", supervisor._rearm_model_time_ns),
      ("pendingOffPlan", pending[1] if pending is not None else None),
      ("pendingOffModel", pending[2] if pending is not None else None),
    )],
    "lastCreepSample": _creep_snapshot(supervisor._last_creep_sample),
    "independentCreepSample": _creep_snapshot(supervisor._last_independent_creep_sample),
    "pendingOffCreep": _creep_snapshot(pending[0] if pending is not None else None),
  }


class ControlObservation:
  def __init__(self, params, params_sp, *, enabled=None, clock=None):
    self.recorder = ObservationRecorder("controlsd", params, params_sp, enabled=enabled, clock=clock)
    self.sample = None
    self.inputs = None
    self.controller = None
    self.nonfinite_actuators = False
    self.calibration = None
    self.calibration_message = None
    self.pose = None
    self.pose_object = None
    self.sample_pose = None

  @classmethod
  def create(cls, params, params_sp):
    try:
      return cls(params, params_sp)
    except Exception:
      return None

  def calibration_updated(self, state, calibrator):
    self.calibration = None
    self.calibration_message = None
    message = state["extrinsicsCalibration"]
    value = {
      "inputs": capture_inputs(state, ("extrinsicsCalibration",)),
      "calibrationValid": bool(calibrator.calib_valid),
      "calibrationStatusKnown": True, "calibrationStatus": message.calStatus,
    }
    self.calibration = value
    self.calibration_message = message

  def pose_updated(self, state, calibrator, pose):
    self.pose = None
    self.pose_object = None
    message = state["deviceMotion"]
    # The calibrator is stateful. Preserve the calibration that was fed for this
    # build; a newer calibration arrival without a pose update cannot relabel it.
    # Object identity also prevents an observer failure on a later feed from
    # attributing that new build to a previously observed calibration message.
    calibration = self.calibration if self.calibration_message is state["extrinsicsCalibration"] else None
    value = {
      "known": True, "inputs": capture_inputs(state, ("deviceMotion",)),
      "calibrationKnown": calibration is not None, "calibrationValid": bool(calibrator.calib_valid),
      "deviceHealthKnown": True, "inputsOK": bool(message.inputsOK), "posenetOK": bool(message.posenetOK),
      "sensorsOK": bool(message.sensorsOK), "orientationValid": bool(message.orientationNED.valid),
      "velocityValid": bool(message.velocityDevice.valid), "accelerationValid": bool(message.accelerationDevice.valid),
      "angularVelocityValid": bool(message.angularVelocityDevice.valid),
      "deviceTimestampKnown": is_uint64(message.timestamp) and message.timestamp > 0,
      "deviceTimestamp": int(message.timestamp),
    }
    if calibration is not None:
      value["inputs"] += calibration["inputs"]
      value["calibrationStatusKnown"] = calibration["calibrationStatusKnown"]
      value["calibrationStatus"] = calibration["calibrationStatus"]
    self.pose = value
    self.pose_object = pose

  def begin(self, state, calibrated_pose=None):
    self.sample = None
    self.inputs = None
    self.controller = None
    self.nonfinite_actuators = False
    self.sample_pose = None
    sample = self.recorder.begin(getattr(state, "frame", None))
    if sample is None:
      return
    services = tuple(dict.fromkeys((*CONTROL_INPUTS, *getattr(state, "logMonoTime", {}))))
    inputs = capture_inputs(state, services)
    self.sample = sample
    self.inputs = inputs
    self.sample_pose = self.pose if self.pose_object is calibrated_pose and calibrated_pose is not None else {"known": False}

  def controller_input(self, controller, plan, limits, long_active, motion_input, *, mode_mismatch=False):
    self.controller = None
    if self.sample is None:
      return
    values = {
      "controllerCalled": False,
      "targetAccel": float(plan.aTarget),
      "accelMin": float(limits[0]),
      "accelMax": float(limits[1]),
      "stateBefore": controller.long_control_state,
      "longActiveAtController": bool(long_active),
      "unified": bool(controller.unified),
      "decisionMonoTime": int(motion_input.get("now_ns", 0)),
      "decisionMonoTimeKnown": "now_ns" in motion_input,
      "analysis": {
        "version": 1,
        "motionInputsKnown": all(name in motion_input for name in ("now_ns", "plan_valid", "braking_plan_valid", "standby_input")),
        "planValid": bool(motion_input.get("plan_valid", False)),
        "brakingPlanValid": bool(motion_input.get("braking_plan_valid", False)),
        "standbyInput": bool(motion_input.get("standby_input", False)),
        "modeMismatch": bool(mode_mismatch),
        "motionBefore": supervisor_snapshot(controller),
        "pose": self.sample_pose,
      },
    }
    self.controller = values

  def controller_output(self, controller, acceleration):
    if self.controller is None:
      return
    self.controller = {
      **self.controller,
      "controllerCalled": True,
      "controllerAccel": acceleration,
      "stateAfter": controller.long_control_state,
      "analysis": {**self.controller["analysis"], "motionAfter": supervisor_snapshot(controller)},
    }

  def sanitized(self):
    self.nonfinite_actuators = True

  def end(self):
    sample = self.sample
    self.sample = None
    self.sample = self.recorder.finish(sample)

  def publish(self, state_message, control_message, state=None):
    if self.sample is None or self.controller is None or not self.controller["controllerCalled"]:
      return False
    sample = self.sample
    self.sample = None
    if not sample["endMonoTime"]:
      return False
    command = control_message.carControl
    values = {
      **self.controller,
      "inputs": self.inputs,
      "carControlMonoTime": int(control_message.logMonoTime),
      "publishedAccel": float(command.actuators.accel),
      "longActivePublished": bool(command.longActive),
      "resumePublished": bool(command.cruiseControl.resume),
      "nonfiniteActuators": self.nonfinite_actuators,
      "analysis": {
        **self.controller["analysis"],
        "forceDecelKnown": state is not None,
        "forceDecel": bool(state_message.forceDecel),
        "driverMonitoringForceDecel": bool(state["driverMonitoringState"].noResponseForceDecel) if state is not None else False,
        "selfdriveSoftDisabling": state["selfdriveState"].state == log.SelfdriveState.OpenpilotState.softDisabling if state is not None else False,
      },
    }
    return attach_diagnostics(state_message, "longitudinalDiagnostics", log.LongitudinalControlDiagnostics,
                              self.recorder, sample, values)
