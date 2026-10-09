#!/usr/bin/env python3
import gc
import math
import time
from dataclasses import dataclass, replace
from functools import cache
from numbers import Number

import capnp

from openpilot.cereal import log
from opendbc.car.structs import car
import openpilot.cereal.messaging as messaging
from openpilot.common.constants import CV
from openpilot.common.longitudinal_logging import observe
from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, DT_CTRL, DT_MDL, Priority, Ratekeeper
from openpilot.sunnypilot.common.phase_timer import PhaseTimer
from openpilot.common.swaglog import cloudlog

from opendbc.car.car_helpers import interfaces
from opendbc.car.interfaces import ACCEL_MIN
from opendbc.car.vehicle_model import VehicleModel
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, clip_curvature, get_departure_accel_from_plan
from openpilot.selfdrive.controls.lib.control_observation import ControlObservation
from openpilot.selfdrive.controls.lib.latcontrol import LatControl
from openpilot.selfdrive.controls.lib.latcontrol_pid import LatControlPID
from openpilot.selfdrive.controls.lib.latcontrol_angle import LatControlAngle, STEER_ANGLE_SATURATION_THRESHOLD
from openpilot.selfdrive.controls.lib.latcontrol_curvature import LatControlCurvature
from openpilot.selfdrive.controls.lib.latcontrol_torque import LatControlTorque
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.longitudinal_continuation import CONTINUATION_VERSION, ContinuationOffer, execution_target
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION, MotionIntent, StopReason
from openpilot.selfdrive.controls.lib.object_influence import accepted_mitigation_influences, accepted_object_influences
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_TIMING_VERSION, ReferenceTimingConsumer
from openpilot.selfdrive.modeld.constants import LAT_SMOOTH_SECONDS, ModelConstants
from openpilot.selfdrive.locationd.helpers import PoseCalibrator, Pose

from openpilot.sunnypilot.selfdrive.controls.controlsd_ext import ControlsExt

State = log.SelfdriveState.OpenpilotState
LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

ACTUATOR_FIELDS = tuple(car.CarControl.Actuators.schema.fields.keys())


@dataclass(frozen=True)
class _LongitudinalPlanCache:
  message: object
  car_params: object
  subscriber: object
  key: tuple
  intent: MotionIntent
  command_identity: tuple


@cache
def _bytes_reader_owner_type():
  # The concrete internal class is not exported by pycapnp. Derive its type
  # from an actual bytes input, and require the checked reference layout.
  try:
    payload = log.Event.new_message().to_bytes()
    with log.Event.from_bytes(payload) as reader:
      owner = reader._parent
      refs = gc.get_referents(owner)
      if (type(owner).__name__ == '_FlatArrayMessageReader' and len(refs) == 3
          and sum(ref is payload for ref in refs) == 1 and all(ref is None or ref is payload for ref in refs)):
        return type(owner)
  except Exception:
    pass
  return None


def _has_received_reader_owner(message):
  # A reader interface alone is insufficient: builder views and from_bytes
  # on bytearray/read-only memoryview can still share mutable storage. In
  # pycapnp 2.1.0, this owner's GC slots are _parent, _object_to_pin and
  # _buffer_view. Require a single direct bytes pin and two empty slots.
  # Unknown versions, ownership APIs or reference layouts decode each frame.
  try:
    if getattr(capnp, '__version__', None) != '2.1.0':
      return False
    reader_type = getattr(capnp.lib.capnp, '_DynamicStructReader', None)
    owner_type = getattr(capnp.lib.capnp, '_MessageReader', None)
    if reader_type is None or owner_type is None:
      return False
    owner = message
    for _ in range(3):  # longitudinalPlan -> Event -> message reader
      if not isinstance(owner, reader_type):
        if not isinstance(owner, owner_type) or type(owner) is not _bytes_reader_owner_type():
          return False
        refs = gc.get_referents(owner)
        return (len(refs) == 3 and sum(type(ref) is bytes for ref in refs) == 1
                and all(ref is None or type(ref) is bytes for ref in refs))
      owner = owner._parent
  except Exception:
    return False
  return False


class Controls(ControlsExt):
  def __init__(self) -> None:
    self.params = Params()
    cloudlog.info("controlsd is waiting for CarParams")
    self.CP = messaging.log_from_bytes(self.params.get("CarParams", block=True), car.CarParams)
    cloudlog.info("controlsd got CarParams")

    # Initialize sunnypilot controlsd extension and base model state
    ControlsExt.__init__(self, self.CP, self.params)

    self.CI = interfaces[self.CP.carFingerprint](self.CP, self.CP_SP)

    self.sm = messaging.SubMaster(['lateralDelay', 'vehicleParameters', 'lateralTorqueParameters', 'modelV2', 'selfdriveState',
                                   'extrinsicsCalibration', 'deviceMotion', 'longitudinalPlan', 'lateralManeuverPlan', 'carState', 'carOutput',
                                   'driverMonitoringState', 'onroadEvents', 'driverAssistance'] + self.sm_services_ext,
                                  poll='selfdriveState')
    self.pm = messaging.PubMaster(['carControl', 'controlsState'] + self.pm_services_ext)

    self.steer_limited_by_safety = False
    self.curvature = 0.0
    self.desired_curvature = 0.0

    self.pose_calibrator = PoseCalibrator()
    self.calibrated_pose: Pose | None = None

    self.LoC = LongControl(self.CP, self.CP_SP)
    self._longitudinal_plan_cache = None
    self.reference_timing = ReferenceTimingConsumer()
    self.reference_timing_check = None
    self.longitudinal_observation = ControlObservation.create(self.CP, self.CP_SP)
    self.longitudinal_mode_mismatch = False
    self.VM = VehicleModel(self.CP)
    self.LaC: LatControl
    if self.CP.steerControlType == car.CarParams.SteerControlType.angle:
      self.LaC = LatControlAngle(self.CP, self.CP_SP, self.CI, DT_CTRL)
    elif self.CP.steerControlType == car.CarParams.SteerControlType.curvature:
      self.LaC = LatControlCurvature(self.CP, self.CP_SP, self.CI, DT_CTRL)
    elif self.CP.lateralTuning.which() == 'pid':
      self.LaC = LatControlPID(self.CP, self.CP_SP, self.CI, DT_CTRL)
    elif self.CP.lateralTuning.which() == 'torque':
      self.LaC = LatControlTorque(self.CP, self.CP_SP, self.CI, DT_CTRL)

    self.LaC = ControlsExt.initialize_lateral_control(self, self.LaC, self.CI, DT_CTRL)

  def update(self):
    self.sm.update(15)
    longitudinal_observation = getattr(self, "longitudinal_observation", None)
    if self.sm.updated["extrinsicsCalibration"]:
      self.pose_calibrator.feed_extrinsics_calibration(self.sm['extrinsicsCalibration'])
      observe(longitudinal_observation, "calibration_updated", self.sm, self.pose_calibrator)
    if self.sm.updated["deviceMotion"]:
      device_motion = Pose.from_device_motion(self.sm['deviceMotion'])
      self.calibrated_pose = self.pose_calibrator.build_calibrated_pose(device_motion)
      observe(longitudinal_observation, "pose_updated", self.sm, self.pose_calibrator, self.calibrated_pose)

  def _decode_longitudinal_plan(self, long_plan):
    mode = self.LoC.longitudinal_mode
    action_t = self.CP.longitudinalActuatorDelay + DT_MDL if mode == LongitudinalMode.UNIFIED else None
    recv_frame = getattr(self.sm, 'recv_frame', {}).get('longitudinalPlan')
    key = (recv_frame, self.sm.logMonoTime['longitudinalPlan'], mode, action_t)
    cached = getattr(self, '_longitudinal_plan_cache', None)
    if (cached is not None and cached.message is long_plan and cached.car_params is self.CP
        and cached.subscriber is self.sm and cached.key == key):
      return cached.intent, cached.command_identity

    self._longitudinal_plan_cache = None
    intent = MotionIntent.from_message(long_plan.motionIntent)
    command_identity = ()
    projection_numbers = ()
    if mode == LongitudinalMode.UNIFIED:
      speeds, accels = tuple(long_plan.speeds), tuple(long_plan.accels)
      reference = long_plan.reference
      # This local fact restores the old strong-departure interpretation; it
      # cannot replace the sampled execution target or producer permission.
      intent = replace(intent, release_accel=get_departure_accel_from_plan(
        speeds, accels, ModelConstants.T_IDXS[:CONTROL_N], action_t, reference.constraintAccel))
      command_identity = (int(long_plan.modelMonoTime), float(long_plan.aTarget), bool(long_plan.shouldStop),
                          int(long_plan.solverStatus), intent, float(reference.accel),
                          float(reference.constraintAccel), bool(reference.overridden), speeds, accels, action_t,
                          ContinuationOffer.from_message(getattr(long_plan, 'continuation', None)))
      projection_numbers = (command_identity[1], command_identity[5], command_identity[6], action_t, *speeds, *accels)
      offer = command_identity[-1]
      if offer is not None:
        projection_numbers += (offer.source_cruise_kph, offer.candidate.target_accel, offer.candidate.speed_tolerance, offer.candidate.accel_tolerance,
                               *offer.candidate.speeds, *offer.candidate.accels, *offer.candidate.jerks)
    if recv_frame is not None and _has_received_reader_owner(long_plan):
      # Reusing a tuple containing NaN can change identity comparisons (NaN
      # is not equal to itself). Preserve the original decoding path for it.
      if intent.braking_accel is not None:
        projection_numbers += (intent.braking_accel,)
      if all(math.isfinite(value) for value in projection_numbers):
        self._longitudinal_plan_cache = _LongitudinalPlanCache(long_plan, self.CP, self.sm, key, intent, command_identity)
    return intent, command_identity

  def state_control(self):
    longitudinal_observation = getattr(self, "longitudinal_observation", None)
    observe(longitudinal_observation, "begin", self.sm, self.calibrated_pose)
    CS = self.sm['carState']

    # Update VehicleModel
    lp = self.sm['vehicleParameters']
    x = max(lp.stiffnessFactor, 0.1)
    sr = max(lp.steerRatio, 0.1)
    self.VM.update_params(x, sr)

    steer_angle_without_offset = math.radians(CS.steeringAngleDeg - lp.angleOffsetDeg)
    self.curvature = -self.VM.calc_curvature(steer_angle_without_offset, CS.vEgo, lp.roll)

    # Update Torque Params
    if self.CP.lateralTuning.which() == 'torque':
      torque_params = self.sm['lateralTorqueParameters']
      if self.sm.all_checks(['lateralTorqueParameters']) and torque_params.useParams:
        self.LaC.update_torque_parameters(torque_params.latAccelFactorFiltered, torque_params.latAccelOffsetFiltered,
                                           torque_params.frictionCoefficientFiltered)

        self.LaC.extension.update_limits()

      self.LaC.extension.update_model_v2(self.sm['modelV2'])

      self.LaC.extension.update_lateral_lag(self.lat_delay)

    long_plan = self.sm['longitudinalPlan']
    model_v2 = self.sm['modelV2']

    CC = car.CarControl.new_message()
    CC.enabled = self.sm['selfdriveState'].enabled

    # Check which actuators can be enabled
    standstill = abs(CS.vEgo) <= max(self.CP.minSteerSpeed, 0.3) or CS.standstill

    # Get which state to use for active lateral control
    _lat_active = self.get_lat_active(self.sm)

    CC.latActive = _lat_active and not CS.steerFaultTemporary and not CS.steerFaultPermanent and \
                   (not standstill or self.CP.steerAtStandstill)
    CC.longActive = CC.enabled and not any(e.overrideLongitudinal for e in self.sm['onroadEvents']) and \
                    (self.CP.openpilotLongitudinalControl or not self.CP_SP.pcmCruiseSpeed)
    expected_version = MOTION_INTENT_VERSION if self.LoC.unified else 0
    self.longitudinal_mode_mismatch = (long_plan.motionIntent.version != expected_version
                                      or long_plan.motionIntent.mode != int(self.LoC.longitudinal_mode))
    if self.longitudinal_mode_mismatch and not self.LoC.motion_owner_established:
      CC.longActive = False
    if self.LoC.unified or long_plan.solverStatus != 0:
      CC.longActive = CC.longActive and not CS.gasPressed and not CS.brakePressed
    motion_intent, command_identity = self._decode_longitudinal_plan(long_plan) if self.LoC.unified else (None, ())
    if not self.LoC.unified:
      self._longitudinal_plan_cache = None
    now_ns = time.monotonic_ns() if self.LoC.unified else None
    reference_valid = True
    self.reference_timing_check = None
    if self.LoC.longitudinal_mode == LongitudinalMode.UNIFIED:
      # Preserve all three times: source epoch, LP Event availability and this
      # actual consumption. Only source age can qualify a measured reference.
      self.reference_timing_check = self.reference_timing.consume(
        long_plan.reference, self.sm.logMonoTime['longitudinalPlan'], now_ns,
        command_identity=command_identity)
      if (self.reference_timing_check.valid
          and (motion_intent.producer_instance_ns <= 0 or motion_intent.producer_instance_ns != self.reference_timing_check.timing.instance_ns)):
        self.reference_timing_check = replace(self.reference_timing_check, valid=False, reason='intent_reference_instance_mismatch')
      reference_valid = self.reference_timing_check.valid
      if not self.reference_timing_check.clock_valid:
        # An impossible clock must not enter the owner's time high water.
        # Its existing invalid-input path still owns braking and release.
        now_ns = 0
    if (self.LoC.unified and self.LoC.can_standby and not motion_intent.ready and long_plan.solverStatus == 0
        and self.sm.all_checks(['longitudinalPlan', 'carState'])):
      CC.longActive = False
    standby_input = bool(self.LoC.unified and self.LoC.can_standby and not self.longitudinal_mode_mismatch
                         and motion_intent.is_standby_valid() and long_plan.solverStatus == 0
                         and self.sm.all_checks(['longitudinalPlan', 'carState']) and CS.canValid
                         and math.isfinite(CS.vEgo) and math.isfinite(CS.aEgo) and CS.vCruise == V_CRUISE_UNSET)

    actuators = CC.actuators

    # Enable blinkers while lane changing
    if model_v2.meta.laneChangeState != LaneChangeState.off:
      CC.leftBlinker = model_v2.meta.laneChangeDirection == LaneChangeDirection.left
      CC.rightBlinker = model_v2.meta.laneChangeDirection == LaneChangeDirection.right

    if not CC.latActive:
      self.LaC.reset()
    if not CC.longActive:
      self.LoC.reset()

    # accel PID loop
    pid_accel_limits = self.CI.get_pid_accel_limits(self.CP, self.CP_SP, CS.vEgo, CS.vCruise * CV.KPH_TO_MS)
    motion_input = {}
    if self.LoC.unified:
      continuation = command_identity[-1]
      force_decel = bool(self.sm['driverMonitoringState'].noResponseForceDecel or self.sm['selfdriveState'].state == State.softDisabling)
      current_model_ns = self.sm.logMonoTime['modelV2']
      # A new model requires a new producer risk check. Reusing an offer may
      # never ignore stronger current braking/stop evidence between LP ticks.
      continuation_inputs = bool(
        reference_valid and not self.longitudinal_mode_mismatch and not force_decel
        and (long_plan.solverStatus == 0 or self.sm.valid['longitudinalPlan'] is False)
        and self.sm.all_checks(['modelV2', 'carState']) and CS.canValid
        and all(getattr(self.sm, health).get('longitudinalPlan') is True for health in ('seen', 'alive', 'freq_ok'))
        and 0 < current_model_ns == int(long_plan.modelMonoTime) <= now_ns
        and now_ns - current_model_ns <= 250_000_000
        and math.isfinite(CS.vCruise) and CS.vCruise != V_CRUISE_UNSET
        and not model_v2.action.shouldStop and not model_v2.meta.hardBrakePredicted and not long_plan.fcw
        and math.isfinite(model_v2.action.desiredAcceleration)
      )
      if (continuation is not None and continuation.valid is True and continuation.version == CONTINUATION_VERSION
          and model_v2.action.desiredAcceleration < 0.0):
        candidate_target = execution_target(continuation, self.reference_timing_check.timing if reference_valid else None,
                                            now_ns, self.CP.longitudinalActuatorDelay)
        continuation_inputs &= candidate_target is not None and candidate_target <= model_v2.action.desiredAcceleration
      motion_input = {
        'intent': motion_intent,
        'plan_time_ns': int(self.sm.logMonoTime['longitudinalPlan']),
        'model_time_ns': int(long_plan.modelMonoTime),
        'now_ns': now_ns,
        'standby_input': standby_input,
        'continuation': continuation,
        'continuation_input_valid': continuation_inputs,
        'reference_check': self.reference_timing_check,
        'force_decel': force_decel,
        'car_state_ns': int(self.sm.logMonoTime['carState']),
        'native_recovery_valid': bool(
          reference_valid and not self.longitudinal_mode_mismatch and CS.canValid
          and self.sm.all_checks(['modelV2', 'longitudinalPlan', 'carState'])
          and all(getattr(self.sm, health).get(service) is True for service in ('modelV2', 'longitudinalPlan', 'carState')
                  for health in ('seen', 'alive', 'freq_ok'))
          and 0 < int(long_plan.modelMonoTime) <= current_model_ns <= now_ns
          and now_ns - int(long_plan.modelMonoTime) <= 250_000_000),
        'plan_valid': (reference_valid and not self.longitudinal_mode_mismatch and self.sm.all_checks(['longitudinalPlan', 'carState'])
                       and math.isfinite(CS.vCruise) and CS.vCruise != V_CRUISE_UNSET),
        'braking_plan_valid': (reference_valid and not self.longitudinal_mode_mismatch and self.sm.valid.get('carState') is True and CS.canValid
                               and all(getattr(self.sm, health).get(service) is True for service in ('carState', 'longitudinalPlan')
                                       for health in ('seen', 'alive', 'freq_ok'))
                               and math.isfinite(CS.vCruise) and CS.vCruise != V_CRUISE_UNSET),
      }
    observe(longitudinal_observation, "controller_input", self.LoC, long_plan, pid_accel_limits, CC.longActive, motion_input,
            mode_mismatch=self.longitudinal_mode_mismatch)
    controller_accel = float(self.LoC.update(CC.longActive, CS, long_plan.aTarget, long_plan.shouldStop, pid_accel_limits,
                                            solver_status=int(long_plan.solverStatus), **motion_input))
    actuators.accel = controller_accel
    observe(longitudinal_observation, "controller_output", self.LoC, controller_accel)
    # After update(), not before: LoC.update() runs the state transition itself, so publishing the
    # state first pairs this frame's accel with last frame's state. Anything reading the log for
    # where a stop began or ended -- or for the step out of one -- lands one frame off, on the
    # wrong side of the step.
    actuators.longControlState = self.LoC.long_control_state

    # Steering PID loop and lateral MPC
    # Reset desired curvature to current to avoid violating the limits on engage
    if self.sm.valid['lateralManeuverPlan']:
      new_desired_curvature = self.sm['lateralManeuverPlan'].desiredCurvature if CC.latActive else self.curvature
    else:
      new_desired_curvature = model_v2.action.desiredCurvature if CC.latActive else self.curvature
    self.desired_curvature, curvature_limited = clip_curvature(CS.vEgo, self.desired_curvature, new_desired_curvature, lp.roll)
    lat_delay = self.sm["lateralDelay"].lateralDelay + LAT_SMOOTH_SECONDS

    actuators.curvature = self.desired_curvature
    steer, lateral_output, lac_log = self.LaC.update(CC.latActive, CS, self.VM, lp,
                                                     self.steer_limited_by_safety, self.desired_curvature,
                                                     self.calibrated_pose, curvature_limited, lat_delay)
    actuators.torque = float(steer)
    if self.CP.steerControlType == car.CarParams.SteerControlType.curvature:
      actuators.curvature = float(lateral_output)
    else:
      actuators.steeringAngleDeg = float(lateral_output)
    # Ensure no NaNs/Infs
    for p in ACTUATOR_FIELDS:
      attr = getattr(actuators, p)
      if not isinstance(attr, Number):
        continue

      if not math.isfinite(attr):
        cloudlog.error(f"actuators.{p} not finite {actuators.to_dict()}")
        setattr(actuators, p, 0.0)
        observe(longitudinal_observation, "sanitized")

    self._object_influence_receipt = []
    try:
      self._object_influence_receipt = accepted_object_influences(
        long_plan, self.LoC, CS, long_active=bool(CC.longActive), plan_valid=motion_input.get('plan_valid', False),
        plan_ns=int(self.sm.logMonoTime['longitudinalPlan']), now_ns=now_ns, output_accel=float(actuators.accel),
        owner_tracking=self.LoC.long_control_state == car.CarControl.Actuators.LongControlState.pid,
        independent_stop=bool(self.LoC.motion_reasons & (StopReason.E2E | StopReason.CRUISE | StopReason.FORCE_DECEL
                                                        | StopReason.CREEP | StopReason.INVALID)),
      )
      self._object_influence_receipt.extend(accepted_mitigation_influences(
        long_plan, self.LoC, CS, long_active=bool(CC.longActive), braking_plan_valid=motion_input.get('braking_plan_valid', False),
        key_inputs_valid=bool(self.sm.all_checks(['carState', 'modelV2', 'selfdriveState'])),
        key_input_times=tuple(int(self.sm.logMonoTime[service]) for service in ('carState', 'modelV2', 'selfdriveState')),
        plan_ns=int(self.sm.logMonoTime['longitudinalPlan']), now_ns=now_ns, output_accel=float(actuators.accel), accel_min=ACCEL_MIN,
      ))
    except (AttributeError, TypeError, ValueError, OverflowError):
      # Receipt failure only removes a display annotation; it cannot affect
      # the completed owner decision or its command.
      pass
    observe(longitudinal_observation, "end")
    return CC, lac_log

  def publish(self, CC, lac_log):
    CS = self.sm['carState']

    # Orientation and angle rates can be useful for carcontroller
    # Only calibrated (car) frame is relevant for the carcontroller
    CC.currentCurvature = self.curvature
    if self.calibrated_pose is not None:
      CC.orientationNED = self.calibrated_pose.orientation.xyz.tolist()
      CC.angularVelocity = self.calibrated_pose.angular_velocity.xyz.tolist()

    CC.cruiseControl.override = CC.enabled and not CC.longActive and (self.CP.openpilotLongitudinalControl or not self.CP_SP.pcmCruiseSpeed)
    CC.cruiseControl.cancel = CS.cruiseState.enabled and (not CC.enabled or not self.CP.pcmCruise)
    resolved_stop = self.LoC.resolved_should_stop if self.LoC.unified else self.sm['longitudinalPlan'].shouldStop
    CC.cruiseControl.resume = (CC.enabled and CS.cruiseState.standstill and not resolved_stop
                              and (not self.LoC.unified or CC.longActive and not self.LoC.motion_input_constrained)
                              and not self.longitudinal_mode_mismatch and self.sm['longitudinalPlan'].solverStatus == 0)

    hudControl = CC.hudControl
    hudControl.setSpeed = float(CS.vCruiseCluster * CV.KPH_TO_MS)
    hudControl.speedVisible = CC.enabled
    hudControl.lanesVisible = CC.enabled
    hudControl.leadVisible = self.sm['longitudinalPlan'].hasLead
    hudControl.leadDistanceBars = self.sm['selfdriveState'].personality.raw + 1
    hudControl.visualAlert = self.sm['selfdriveState'].alertHudVisual

    hudControl.rightLaneVisible = True
    hudControl.leftLaneVisible = True
    if self.sm.valid['driverAssistance']:
      hudControl.leftLaneDepart = self.sm['driverAssistance'].leftLaneDeparture
      hudControl.rightLaneDepart = self.sm['driverAssistance'].rightLaneDeparture

    if self.get_lat_active(self.sm):
      CO = self.sm['carOutput']
      if self.CP.steerControlType == car.CarParams.SteerControlType.angle:
        self.steer_limited_by_safety = abs(CC.actuators.steeringAngleDeg - CO.actuatorsOutput.steeringAngleDeg) > \
                                              STEER_ANGLE_SATURATION_THRESHOLD
      else:
        self.steer_limited_by_safety = abs(CC.actuators.torque - CO.actuatorsOutput.torque) > 1e-2

    # TODO: both controlsState and carControl valids should be set by
    #       sm.all_checks(), but this creates a circular dependency

    # controlsState
    dat = messaging.new_message('controlsState')
    dat.valid = CS.canValid
    cs = dat.controlsState

    cs.curvature = self.curvature
    cs.longitudinalPlanMonoTime = self.sm.logMonoTime['longitudinalPlan']
    cs.lateralPlanMonoTime = self.sm.logMonoTime['modelV2']
    cs.desiredCurvature = self.desired_curvature
    cs.longControlState = self.LoC.long_control_state
    cs.upAccelCmd = float(self.LoC.pid.p)
    cs.uiAccelCmd = float(self.LoC.pid.i)
    cs.ufAccelCmd = float(self.LoC.pid.f)
    cs.forceDecel = bool(self.sm['driverMonitoringState'].noResponseForceDecel or
                         (self.sm['selfdriveState'].state == State.softDisabling))
    if self.LoC.unified or self.longitudinal_mode_mismatch:
      motion = cs.motionState
      motion.version = MOTION_INTENT_VERSION
      motion.planningMode = int(self.LoC.longitudinal_mode)
      motion.enabled = self.LoC.unified
      motion.phase = self.LoC.motion_phase.text
      motion.reasons = int(self.LoC.motion_reasons)
      motion.resolvedStop = bool(self.LoC.resolved_should_stop)
      motion.degraded = bool(self.LoC.motion_degraded or self.longitudinal_mode_mismatch)
      motion.waitingForFreshInput = bool(self.LoC.motion_waiting_for_fresh_input)
      motion.planMonoTime = int(self.sm.logMonoTime['longitudinalPlan'])
      motion.modelMonoTime = int(self.sm['longitudinalPlan'].modelMonoTime)
      motion.objectInfluences = getattr(self, '_object_influence_receipt', [])
      if self.LoC.unified:
        motion.continuationVersion = CONTINUATION_VERSION
        motion.continuationActive = bool(self.LoC.motion_supervisor.continuation_active)
        motion.continuationEpisodeMonoTime = self.LoC.continuation_budget.episode_start_ns
        motion.continuationReason = self.LoC.continuation_reason
        offer = self.LoC.continuation_offer
        if offer is not None:
          motion.continuationDonorProducerMonoTime = offer.donor_producer_ns
          motion.continuationDonorSourceMonoTime = offer.donor_source_ns
          motion.continuationDonorModelMonoTime = offer.donor_model_ns
          motion.continuationValidUntilMonoTime = offer.valid_until_ns
      if self.reference_timing_check is not None:
        check = self.reference_timing_check
        motion.referenceTimingVersion = REFERENCE_TIMING_VERSION
        motion.referenceTimingValid = check.valid
        motion.referenceTimingReason = check.reason
        motion.referenceConsumerMonoTime = check.now_ns
        motion.referenceSourceAdvanced = check.source_advanced
        if check.timing is not None:
          motion.referenceSourceMonoTime = check.timing.source_ns
          motion.referenceEpochMonoTime = check.timing.epoch_ns
          motion.referenceProducerMonoTime = check.timing.producer_ns
          motion.referenceProducerInstanceMonoTime = check.timing.instance_ns

    # trigger the car's stock driver monitoring escalation
    CC.driverMonitoringEscalation = cs.forceDecel

    lat_tuning = self.CP.lateralTuning.which()
    if self.CP.steerControlType == car.CarParams.SteerControlType.angle:
      cs.lateralControlState.angleState = lac_log
    elif self.CP.steerControlType == car.CarParams.SteerControlType.curvature:
      cs.lateralControlState.curvatureState = lac_log
    elif lat_tuning == 'pid':
      cs.lateralControlState.pidState = lac_log
    elif lat_tuning == 'torque':
      cs.lateralControlState.torqueState = lac_log

    cc_send = messaging.new_message('carControl')
    cc_send.valid = CS.canValid
    cc_send.carControl = CC
    observe(getattr(self, "longitudinal_observation", None), "publish", cs, cc_send, self.sm)

    self.pm.send('controlsState', dat)
    self.pm.send('carControl', cc_send)

  def run(self):
    rk = Ratekeeper(100, print_delay_threshold=None)
    # None unless /data/controlsd_profile exists at process start. controlsd is FIFO 53 on core 4
    # with card and selfdrived -- one serial 100 Hz chain -- so the writer drops to SCHED_OTHER
    # and pins to [0, 1, 2]. update() is the pacer: sm.update(15) blocks up to 15 ms.
    prof = PhaseTimer.create(PHASE_TIMER_MARKER, PHASE_TIMER_EVENT, PHASES, PHASE_TIMER_PERIOD)
    if prof:
      prof.start()
      cloudlog.warning("controlsd: phase timing ENABLED by %s", PHASE_TIMER_MARKER)
    while True:
      if prof:
        prof.begin()
      self.update()
      if prof:
        prof.mark("update")
      CC, lac_log = self.state_control()
      if prof:
        prof.mark("state_control")
      self.publish(CC, lac_log)
      if prof:
        prof.mark("publish")
      self.get_params_sp(self.sm)
      if prof:
        prof.mark("params_sp")
      self.run_ext(self.sm, self.pm)
      if prof:
        prof.mark("run_ext")
        prof.end_frame()
      rk.monitor_time()


# controlsd's run() loop. update() is the pacer -- sm.update(15) blocks up to 15 ms waiting on
# selfdriveState, which selfdrived publishes, so wall there is the chain's cadence not work.
PHASE_TIMER_MARKER = "/data/controlsd_profile"
PHASE_TIMER_EVENT = "controlsd.phase_timing"
PHASE_TIMER_PERIOD = 3000        # 30 s at 100 Hz
PHASES = ("update", "state_control", "publish", "params_sp", "run_ext")


def main():
  config_realtime_process(4, Priority.CTRL_HIGH)
  controls = Controls()
  controls.run()


if __name__ == "__main__":
  main()
