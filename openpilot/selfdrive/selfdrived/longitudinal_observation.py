from openpilot.cereal import log
from openpilot.common.longitudinal_logging import CHECK_FLAGS, InputSnapshotCache, ObservationRecorder, attach_diagnostics, capture_inputs, is_uint64, logging_enabled


class SelfdriveObservation:
  """Record the decision selfdrived made, without evaluating events or transitions."""

  def __init__(self, params, params_sp, *, enabled=None, clock=None):
    self.recorder = ObservationRecorder("selfdrived", params, params_sp, enabled=enabled, clock=clock)
    self.input_cache = InputSnapshotCache()
    self.sample = None
    self.sample_errors = 0
    self.values = None
    self.last_car_state = None
    self.last_car_state_input = None

  @classmethod
  def create(cls, params, params_sp):
    try:
      if not logging_enabled("selfdrived", params, params_sp):
        return None
      return cls(params, params_sp, enabled=True)
    except Exception:
      return None

  def begin(self, state, car_state_message, car_state):
    self.sample = None
    self.values = None
    self.sample_errors = self.recorder.errors
    sample = self.recorder.begin(getattr(state, "frame", None))
    if sample is None:
      return
    if car_state_message is not None:
      self.last_car_state = None
      self.last_car_state_input = None
      timestamp = car_state_message.logMonoTime
      identity = {
        "service": "carState", "monoTime": int(timestamp) if is_uint64(timestamp) else 0,
        "idKnown": is_uint64(timestamp), "knownFlags": CHECK_FLAGS["valid"] | CHECK_FLAGS["seen"] | CHECK_FLAGS["updated"],
        "flags": CHECK_FLAGS["seen"] | (CHECK_FLAGS["valid"] if car_state_message.valid else 0), "kind": "consumerSnapshot",
      }
      self.last_car_state_input = identity
      self.last_car_state = car_state
    if self.last_car_state is car_state and self.last_car_state_input is not None:
      car_state_input = dict(self.last_car_state_input)
      if car_state_message is not None:
        car_state_input["flags"] |= CHECK_FLAGS["updated"]
    else:
      # The process can reuse CS_prev when recv_one times out. If an observer
      # missed its original receive, absence is honest; a previous ID is not.
      car_state_input = {
        "service": "carState", "monoTime": 0, "idKnown": False, "knownFlags": CHECK_FLAGS["updated"],
        "flags": 0, "kind": "consumerSnapshot",
      }
    inputs = [car_state_input, *capture_inputs(state, tuple(getattr(state, "logMonoTime", {})), cache=self.input_cache)]
    self.values = {
      "version": 1, "inputs": inputs, "carStateReused": car_state_message is None,
      "stateMachineInputKnown": False, "stateMachineCalled": False,
      "motionCheckKnown": False, "motionCommIssueAdded": False, "controlsForceDecelKnown": False,
    }
    self.sample = sample

  def motion_check(self, controls_state):
    if self.sample is None or self.values is None:
      return
    motion = controls_state.motionState
    self.values.update({
      "motionCheckKnown": True, "motionVersion": int(motion.version), "motionDegraded": bool(motion.degraded),
      "motionWaitingForFreshInput": bool(motion.waitingForFreshInput),
      "controlsForceDecel": bool(controls_state.forceDecel), "controlsForceDecelKnown": True,
    })

  def motion_comm_issue_added(self):
    if self.sample is not None and self.values is not None:
      self.values["motionCommIssueAdded"] = True

  def state_machine_input(self, daemon):
    if self.sample is None or self.values is None:
      return
    machine = daemon.state_machine
    self.values.update({
      "stateMachineInputKnown": True, "stateBefore": machine.state, "stateAfter": machine.state,
      "enabledBefore": bool(daemon.enabled), "activeBefore": bool(daemon.active),
      "enabledAfter": bool(daemon.enabled), "activeAfter": bool(daemon.active),
      "softDisableTimerBefore": int(machine.soft_disable_timer), "softDisableTimerAfter": int(machine.soft_disable_timer),
      "events": list(daemon.events.names), "eventsSP": list(daemon.events_sp.names),
    })

  def state_machine_output(self, daemon):
    if self.sample is None or self.values is None:
      return
    machine = daemon.state_machine
    self.values.update({
      "stateMachineCalled": True, "stateAfter": machine.state,
      "enabledAfter": bool(daemon.enabled), "activeAfter": bool(daemon.active),
      "softDisableTimerAfter": int(machine.soft_disable_timer), "currentAlertTypes": list(machine.current_alert_types),
    })

  def publish(self, state_message):
    sample, values = self.sample, self.values
    self.sample = None
    self.values = None
    if (sample is None or values is None or not values["stateMachineInputKnown"]
        or self.recorder.errors != self.sample_errors):
      return False
    completed = self.recorder.finish(sample)
    if completed is None:
      return False
    values = {
      **values, "statePublished": state_message.state, "enabledPublished": bool(state_message.enabled),
      "activePublished": bool(state_message.active),
    }
    return attach_diagnostics(state_message, "longitudinalDiagnostics", log.LongitudinalSelfdriveDiagnostics,
                              self.recorder, completed, values)
