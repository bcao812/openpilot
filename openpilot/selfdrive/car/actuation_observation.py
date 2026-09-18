from types import MappingProxyType

from opendbc.car.structs import car
from openpilot.common.longitudinal_logging import ObservationRecorder, attach_diagnostics, can_digest, capture_inputs, is_uint64, local_input


class ActuationObservation:
  def __init__(self, params, params_sp, *, enabled=None, clock=None):
    self.recorder = ObservationRecorder("card", params, params_sp, enabled=enabled, clock=clock)
    self.apply_sequence = 0
    self.receipt = None
    self.local_state = None
    self.local_state_frame = None
    self.disposition_values = {
      "lastDisposition": "unobserved", "dispositionFrame": 0, "dispositionMonoTime": 0,
      "initialized": False, "passive": bool(getattr(params, "passive", False)),
    }

  @classmethod
  def create(cls, params, params_sp):
    try:
      return cls(params, params_sp)
    except Exception:
      return None

  def state_published(self, mono_time, valid, frame):
    self.local_state = MappingProxyType(local_input("carState", mono_time, valid))
    self.local_state_frame = frame

  def disposition(self, reason, frame, *, initialized, passive):
    if reason == "applying":
      self.receipt = None
    timestamp = self.recorder.clock()
    if not is_uint64(timestamp):
      raise ValueError("Invalid disposition timestamp")
    self.disposition_values = {
      "lastDisposition": reason,
      "dispositionFrame": int(frame) if is_uint64(frame) else 0,
      "dispositionMonoTime": int(timestamp),
      "initialized": bool(initialized),
      "passive": bool(passive),
    }

  def begin_apply(self, state, interface, car_state, requested, applied):
    self.receipt = None
    self.apply_sequence += 1
    frame = getattr(state, "frame", None)
    controller_frame = getattr(getattr(interface, "CC", None), "frame", None)
    inputs = capture_inputs(state, tuple(getattr(state, "logMonoTime", {})))
    state_input = self.local_state if is_uint64(frame) and self.local_state_frame == frame else None
    if state_input is None:
      state_input = {
        "service": "carState", "monoTime": 0, "idKnown": False,
        "knownFlags": 0, "flags": 0, "kind": "unknown",
      }
    inputs.append(dict(state_input))
    identities = {entry["service"]: entry["monoTime"] if entry["idKnown"] else 0 for entry in inputs}
    restricted = applied is not requested
    pedals = tuple(name for name, pressed in (("gasPressed", car_state.gasPressed), ("brakePressed", car_state.brakePressed)) if pressed)
    values = {
      "hasApply": True,
      "applySequence": self.apply_sequence,
      "applyFrame": int(frame) if is_uint64(frame) else 0,
      "controllerFrame": int(controller_frame) if is_uint64(controller_frame) else 0,
      "controllerFrameKnown": is_uint64(controller_frame),
      "carControlMonoTime": identities.get("carControl", 0),
      "carControlSPMonoTime": identities.get("carControlSP", 0),
      "carStateMonoTime": identities.get("carState", 0),
      "requestedAccel": float(requested.actuators.accel),
      "appliedAccel": float(applied.actuators.accel),
      "requestedLongActive": bool(requested.longActive),
      "appliedLongActive": bool(applied.longActive),
      "requestedResume": bool(requested.cruiseControl.resume),
      "appliedResume": bool(applied.cruiseControl.resume),
      "authorityRestricted": restricted,
      "authorityReason": "+".join(pedals) if restricted else "",
      "appliedLongControlState": str(applied.actuators.longControlState),
      "inputs": tuple(MappingProxyType(entry) for entry in inputs),
    }
    timestamp = self.recorder.clock()
    if not is_uint64(timestamp):
      raise ValueError("Invalid apply start timestamp")
    return MappingProxyType(dict(values, applyStartMonoTime=int(timestamp)))

  def end_apply(self, attempt):
    if attempt is None:
      return None
    timestamp = self.recorder.clock()
    if not is_uint64(timestamp) or timestamp < attempt["applyStartMonoTime"]:
      raise ValueError("Invalid apply completion timestamp")
    return MappingProxyType(dict(attempt, applyEndMonoTime=int(timestamp)))

  def sent(self, attempt, frames, sendcan_mono_time):
    if attempt is None:
      return False
    if not is_uint64(sendcan_mono_time) or sendcan_mono_time < attempt["applyEndMonoTime"]:
      raise ValueError("Invalid sendcan timestamp")
    self.receipt = MappingProxyType(dict(attempt, sendcanMonoTime=int(sendcan_mono_time),
                                        canCount=len(frames), canDigest=can_digest(frames)))
    return True

  def publish(self, message, frame):
    sample = self.recorder.begin(frame)
    if sample is None:
      return False
    values = {"hasApply": False}
    if self.receipt is not None:
      values = dict(self.receipt)
      values["inputs"] = [dict(entry) for entry in self.receipt["inputs"]]
    values.update(self.disposition_values)
    sample = self.recorder.finish(sample)
    return attach_diagnostics(message, "longitudinalDiagnostics", car.LongitudinalApplyDiagnostics,
                              self.recorder, sample, values)
