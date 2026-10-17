#!/usr/bin/env python3
import os
import math
import time
import threading

import openpilot.cereal.messaging as messaging

from openpilot.cereal import log, custom
from opendbc.car.structs import car
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcClient


from openpilot.common.params import Params
from openpilot.common.longitudinal_logging import observe
from openpilot.common.realtime import config_realtime_process, Priority, Ratekeeper, DT_CTRL
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.common.phase_timer import PhaseTimer
from openpilot.common.gps import get_gps_location_service

from openpilot.selfdrive.car.car_events import CarEvents
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode, resolve_longitudinal_mode
from openpilot.selfdrive.controls.lib.drive_helpers import CLOSE_LEAD_STOP_SPEED
from openpilot.selfdrive.controls.lib.motion_supervisor import MODEL_LEASE_NS, MOTION_INTENT_VERSION, PLAN_LEASE_NS, MotionIntent
from openpilot.selfdrive.controls.lib.longitudinal_continuation import (
  CONTINUATION_STATUSES, CONTINUATION_VERSION, UNSTABLE_DISABLE_NS, ContinuationBudget, ContinuationDonor,
  ContinuationHistory, ContinuationOffer, STALE_MEASUREMENT_STATUS, braking_only_continuation,
  continuation_reference, execution_target, inactive_native_recovery, validate_continuation,
)
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_TIMING_VERSION, ReferenceTimingConsumer, valid_timestamp
from openpilot.selfdrive.locationd.helpers import PoseCalibrator, Pose
from openpilot.selfdrive.selfdrived.events import Events, ET
from openpilot.selfdrive.selfdrived.engagement_policy import EngagementPolicy, LongitudinalEvents
from openpilot.selfdrive.selfdrived.helpers import ExcessiveActuationCheck
from openpilot.selfdrive.selfdrived.longitudinal_observation import SelfdriveObservation
from openpilot.selfdrive.selfdrived.state import StateMachine
from openpilot.selfdrive.selfdrived.alertmanager import AlertManager, set_offroad_alert
from openpilot.selfdrive.selfdrived.perception_alerts import merge_perception_alerts

from openpilot.common.version import get_build_metadata
from openpilot.common.hardware import HARDWARE

from openpilot.sunnypilot.mads.mads import ModularAssistiveDrivingSystem
from openpilot.sunnypilot import get_sanitize_int_param
from openpilot.sunnypilot.selfdrive.car.car_specific import CarSpecificEventsSP
from openpilot.sunnypilot.selfdrive.car.cruise_helpers import CruiseHelper
from openpilot.sunnypilot.selfdrive.car.intelligent_cruise_button_management.controller import IntelligentCruiseButtonManagement
from openpilot.sunnypilot.selfdrive.selfdrived.button_state_tracker import ButtonStateTracker
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP

REPLAY = "REPLAY" in os.environ
SIMULATION = "SIMULATION" in os.environ
TESTING_CLOSET = "TESTING_CLOSET" in os.environ
# Same switch camerad and the manager read; set in launch_env.sh because this device's cabin
# camera is damaged. See system/camerad/cameras/hw.h and system/manager/process_config.py.
DM_DISABLED = "DISABLE_DRIVER" in os.environ

LONGITUDINAL_PERSONALITY_MAP = {v: k for k, v in log.LongitudinalPersonality.schema.enumerants.items()}

ThermalStatus = log.DeviceState.ThermalStatus
State = log.SelfdriveState.OpenpilotState
MadsState = custom.ModularAssistiveDrivingSystem.ModularAssistiveDrivingSystemState
PandaType = log.PandaState.PandaType
LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection
EventName = log.OnroadEvent.EventName
ButtonType = car.CarState.ButtonEvent.Type
SafetyModel = car.CarParams.SafetyModel
AlertLevel = log.DriverMonitoringState.AlertLevel
MonitoringPolicy = log.DriverMonitoringState.MonitoringPolicy
TurnDirection = custom.ModelDataV2SP.TurnDirection

IGNORED_SAFETY_MODES = (SafetyModel.silent, SafetyModel.noOutput)


def fresh_longitudinal_planning_failure(sm, CP, CP_SP, now_ns):
  """Classify a received failed plan without granting it control validity.

  An old solver status, an unknown producer or an unhealthy transport cannot
  exempt a service from the communication checks. The owner independently
  reports degradation until it accepts fresh recovery evidence.
  """
  if (sm.valid['longitudinalPlan'] or CP_SP is None
      or resolve_longitudinal_mode(CP, CP_SP) != LongitudinalMode.UNIFIED):
    return False
  plan = sm['longitudinalPlan']
  if (plan.solverStatus == 0 or plan.motionIntent.version != MOTION_INTENT_VERSION
      or plan.motionIntent.mode != int(LongitudinalMode.UNIFIED)):
    return False
  for service in ('longitudinalPlan', 'modelV2'):
    if not (sm.seen[service] and sm.alive[service] and sm.freq_ok[service]):
      return False
  if not sm.valid['modelV2']:
    return False
  plan_ns, model_ns = sm.logMonoTime['longitudinalPlan'], plan.modelMonoTime
  current_model_ns = sm.logMonoTime['modelV2']
  return (0 < model_ns <= plan_ns <= now_ns and now_ns - plan_ns <= PLAN_LEASE_NS
          and now_ns - model_ns <= MODEL_LEASE_NS
          and 0 < current_model_ns <= now_ns and now_ns - current_model_ns <= MODEL_LEASE_NS)


# selfdrived's step(), in the order it calls them. recv_carState is the 100 Hz pacer -- a
# BLOCKING recv_one, so its wall is ~8 ms and its CPU is the receive syscall plus
# log_from_bytes deserialising carState, which is real work and is not subtracted.
PHASE_TIMER_MARKER = "/data/selfdrived_profile"
PHASE_TIMER_EVENT = "selfdrived.phase_timing"
PHASE_TIMER_PERIOD = 3000        # 30 s at 100 Hz
PHASES = ("recv_carState", "sm_update", "data_sample_rest", "update_events", "state_machine",
          "mads", "update_alerts", "button_tracker", "publish")




class SelfdriveD(CruiseHelper):
  def __init__(self, CP=None, CP_SP=None):
    self.params = Params()

    # Ensure the current branch is cached, otherwise the first cycle lags
    build_metadata = get_build_metadata()

    if CP is None:
      cloudlog.info("selfdrived is waiting for CarParams")
      self.CP = messaging.log_from_bytes(self.params.get("CarParams", block=True), car.CarParams)
      cloudlog.info("selfdrived got CarParams")
    else:
      self.CP = CP

    if CP_SP is None:
      cloudlog.info("selfdrived is waiting for CarParamsSP")
      self.CP_SP = messaging.log_from_bytes(self.params.get("CarParamsSP", block=True), custom.CarParamsSP)
      cloudlog.info("selfdrived got CarParamsSP")
    else:
      self.CP_SP = CP_SP

    self.longitudinal_observation = SelfdriveObservation.create(self.CP, self.CP_SP)
    self.car_events = CarEvents(self.CP)

    # CarParams fields that update_events reads on every frame, lifted to plain Python here.
    # self.CP is assigned once, above, and never rebound -- other objects take a reference to it
    # but none of them replaces selfdrived's -- so these cannot go stale.
    #
    # The reason is that a capnp field read is not an attribute lookup. Measured on an x86 host:
    # 399 ns for CP.notCar against 28 ns for a Python bool, because every read crosses into the
    # C++ reader. update_events did 16 of them per frame for values that were fixed before the
    # car moved.
    self.cp_not_car = self.CP.notCar
    self.cp_olc = self.CP.openpilotLongitudinalControl
    self.cp_pcm_cruise = self.CP.pcmCruise
    self.cp_passive = self.CP.passive
    self.cp_alt_experience = self.CP.alternativeExperience
    # (safetyModel, safetyParam) per configured panda. Flattened rather than kept as a capnp
    # list so the per-panda loop below indexes a tuple instead of re-entering the reader twice.
    self.cp_safety_configs = [(sc.safetyModel, sc.safetyParam) for sc in self.CP.safetyConfigs]

    self.pose_calibrator = PoseCalibrator()
    self.calibrated_pose: Pose | None = None
    self.excessive_actuation_check = ExcessiveActuationCheck()
    self.excessive_actuation = self.params.get("Offroad_ExcessiveActuation") is not None
    self.big_model_loading = False
    self.big_model_active = False
    self.big_model_failed = False
    self.big_model_ready_t = 0.

    # Cached by params_thread at 10 Hz; read by update_events. usbgpu_active MUST start None,
    # not False: modeld does params.remove("ChestnutActive") at startup so "undecided" is None,
    # and update_events tests `big_active is False` to mean failed. Seeding it False would
    # declare a failure on the frames before params_thread's first pass and raise
    # bigModelFailed, which carries ET.PERMANENT -- a 20 s alert on every boot.
    self.usbgpu_loading = False
    self.usbgpu_active = None

    # Setup sockets
    self.pm = messaging.PubMaster(['selfdriveState', 'onroadEvents'] + ['selfdriveStateSP', 'onroadEventsSP'])

    self.gps_location_service = get_gps_location_service(self.params)
    self.gps_packets = [self.gps_location_service]
    self.sensor_packets = ["accelerometer", "gyroscope"]
    self.camera_packets = ["narrowRoadCameraState", "cabinCameraState", "wideRoadCameraState"]
    if DM_DISABLED:
      # Nothing publishes cabinCameraState when the cabin camera is disabled, so leaving it in
      # camera_packets would make all_alive(camera_packets) permanently false -- i.e. a standing
      # cameraMalfunction that would mask a real failure of either road camera.
      self.camera_packets.remove("cabinCameraState")

    # TODO: de-couple selfdrived with card/conflate on carState without introducing controls mismatches
    self.car_state_sock = messaging.sub_sock('carState', timeout=20)

    ignore = self.sensor_packets + self.gps_packets + ['alertDebug', 'lateralManeuverPlan'] + ['modelDataV2SP', 'longitudinalPlanSP']
    if DM_DISABLED:
      # driverMonitoringState is still SUBSCRIBED -- self.sm['driverMonitoringState'] is read
      # unconditionally below and would raise on an absent key. It is subscribed and ignored,
      # so it reads as the default (all-zero) struct: alertLevel none, no lockout.
      ignore += ['driverMonitoringState', 'cabinCameraState']
    if SIMULATION:
      ignore += ['cabinCameraState', 'managerState']
    if REPLAY:
      # no vipc in replay will make them ignored anyways
      ignore += ['narrowRoadCameraState', 'wideRoadCameraState']
    self.sm = messaging.SubMaster(['deviceState', 'pandaStates', 'peripheralState', 'modelV2', 'extrinsicsCalibration',
                                   'carOutput', 'driverMonitoringState', 'longitudinalPlan', 'deviceMotion', 'lateralDelay',
                                   'managerState', 'vehicleParameters', 'radarState', 'lateralTorqueParameters',
                                   'controlsState', 'carControl', 'driverAssistance', 'alertDebug', 'userBookmark',
                                   'lateralManeuverPlan', 'modelDataV2SP', 'longitudinalPlanSP'] + \
                                   self.camera_packets + self.sensor_packets + self.gps_packets,
                                  ignore_alive=ignore, ignore_avg_freq=ignore,
                                  ignore_valid=ignore, frequency=int(1/DT_CTRL))

    # read params
    self.is_metric = self.params.get_bool("IsMetric")
    self.is_ldw_enabled = self.params.get_bool("IsLdwEnabled")
    self.disengage_on_accelerator = self.params.get_bool("DisengageOnAccelerator")

    car_recognized = self.CP.brand != 'mock'

    # cleanup old params
    if not self.CP.alphaLongitudinalAvailable:
      self.params.remove("AlphaLongitudinalEnabled")
    if not self.CP.openpilotLongitudinalControl:
      self.params.remove("ExperimentalMode")

    self.CS_prev = car.CarState.new_message()
    self.AM = AlertManager()
    self.events = Events()

    self.initialized = False
    self.enabled = False
    self.active = False
    self.mismatch_counter = 0
    self.cruise_mismatch_counter = 0
    self.last_steering_pressed_frame = 0
    self.distance_traveled = 0
    self.last_functional_fan_frame = 0
    self.events_prev = []
    self.logged_comm_issue = None
    self.longitudinal_unavailable = False
    self.not_running_prev = None
    # Recomputed only when their 2 Hz / 20 Hz source message arrives; see update_events. The
    # companion *_frame fields hold the sm.recv_frame value the cache was built from, which is
    # what makes the cache immune to update_events' early return. recv_frame starts at 0, so
    # before the first message nothing is recomputed and both hold the value the default
    # all-zero capnp struct would have produced -- identical to the pre-cache behaviour.
    self.not_running = set()
    self.radar_any_error = False
    # -1, not 0. SubMaster seeds frame to -1 and increments BEFORE stamping arrivals
    # (cereal/messaging/__init__.py:185,246), so the first update runs at frame 0 and a message
    # arriving on it is stamped recv_frame == 0. A seed of 0 would compare equal and make that
    # first arrival invisible to the cache. Masked at the managerState site by the falsy
    # `if self.sm.recv_frame[...]` gates below, but nothing gates the radarState one.
    self.not_running_frame = -1
    self.radar_any_error_frame = -1
    self.experimental_mode = False
    self.personality = get_sanitize_int_param(
      "LongitudinalPersonality",
      min(log.LongitudinalPersonality.schema.enumerants.values()),
      max(log.LongitudinalPersonality.schema.enumerants.values()),
      self.params
    )
    self.recalibrating_seen = False
    self.dm_lockout_set = False
    self.dm_uncertain_alerted = False
    self.state_machine = StateMachine()
    self.engagement_policy = EngagementPolicy()
    self.rk = Ratekeeper(100, print_delay_threshold=None)

    # None unless /data/selfdrived_profile exists at start. Every timing call below is guarded by
    # `if self.prof:`, so with it absent the cost on the 100 Hz path is one attribute test per
    # phase boundary and nothing else.
    self.prof = PhaseTimer.create(PHASE_TIMER_MARKER, PHASE_TIMER_EVENT, PHASES, PHASE_TIMER_PERIOD)
    if self.prof is not None:
      cloudlog.warning("selfdrived: phase timing ENABLED by %s", PHASE_TIMER_MARKER)

    # This optional telemetry process has no control publisher; losing it must
    # not affect engagement. Revisit this classification before adding control.
    self.ignored_processes: set[str] = {'hondaRadarObserver'}

    # Determine startup event
    is_remote = build_metadata.openpilot.comma_remote or build_metadata.openpilot.sunnypilot_remote
    self.startup_event = EventName.startup if is_remote and build_metadata.tested_channel else EventName.startup
    if HARDWARE.get_device_type() == 'mici':
      self.startup_event = None
    if not car_recognized:
      self.startup_event = EventName.startupNoCar
    elif car_recognized and self.CP.passive:
      self.startup_event = EventName.startupNoControl
    elif self.CP.secOcRequired and not self.CP.secOcKeyAvailable:
      self.startup_event = EventName.startupNoSecOcKey

    if not car_recognized:
      pass  # MR.ONE: disable car-unrecognized alert
      # self.events.add(EventName.carUnrecognized, static=True)
      # set_offroad_alert("Offroad_CarUnrecognized", True)
    elif self.CP.passive:
      self.events.add(EventName.dashcamMode, static=True)

    self.events_sp = EventsSP()
    self.events_sp_prev = []

    self.mads = ModularAssistiveDrivingSystem(self)
    self.icbm = IntelligentCruiseButtonManagement(self.CP, self.CP_SP)

    self.car_events_sp = CarSpecificEventsSP(self.CP, self.CP_SP)

    CruiseHelper.__init__(self, self.CP)
    self.button_state_tracker = ButtonStateTracker()

  def _continuation_event_status(self, CS, now_ns, plan_failure, motion_degraded):
    """Observe bounded execution without declaring the failed native plan valid."""
    if not hasattr(self, '_longitudinal_failure_budget'):
      self._longitudinal_failure_budget = ContinuationBudget()
      self._continuation_history = ContinuationHistory()
      self._continuation_reference = ReferenceTimingConsumer()
      self._continuation_donor = None
    budget = self._longitudinal_failure_budget
    self._longitudinal_soft_disable_remaining_ns = None
    if (getattr(self, 'CP_SP', None) is None
        or resolve_longitudinal_mode(self.CP, self.CP_SP) != LongitudinalMode.UNIFIED):
      return False, 'unsupported_mode'
    history_valid = bool(getattr(self, '_car_state_valid', False) and CS.canValid and self._continuation_history.observe(
      getattr(self, '_car_state_mono_ns', None), now_ns, CS.vEgo, CS.aEgo))
    plan, motion = self.sm['longitudinalPlan'], self.sm['controlsState'].motionState
    plan_ns = self.sm.logMonoTime['longitudinalPlan']
    controls_ns = self.sm.logMonoTime['controlsState']
    reference = continuation_reference(plan.reference, plan_ns=plan_ns, now_ns=now_ns)
    owner_available = bool(
      all(getattr(self.sm, health).get(service) is True for service in ('controlsState', 'carControl')
          for health in ('seen', 'valid', 'alive', 'freq_ok'))
      and valid_timestamp(controls_ns) and 0 <= now_ns - controls_ns <= PLAN_LEASE_NS
      and motion.version == MOTION_INTENT_VERSION and motion.planningMode == int(LongitudinalMode.UNIFIED)
      and motion.enabled and motion.continuationVersion == CONTINUATION_VERSION
    )
    owner_live = bool(owner_available and self.sm['carControl'].longActive and self.sm['carControl'].enabled
                      and not self.sm['carControl'].cruiseControl.override)
    inactive_receipt = inactive_native_recovery(self.sm['controlsState'], self.sm['carControl'])
    owner_recovering = bool(owner_available and history_valid and inactive_receipt and motion.planMonoTime == plan_ns)
    if owner_recovering:
      self._continuation_donor = None
    owner_reference_healthy = bool((owner_live or owner_recovering) and not motion.degraded and not motion.waitingForFreshInput
                                   and motion.referenceTimingVersion == REFERENCE_TIMING_VERSION and motion.referenceTimingValid
                                   and valid_timestamp(motion.referenceConsumerMonoTime) and valid_timestamp(motion.planMonoTime)
                                   and motion.planMonoTime <= motion.referenceConsumerMonoTime <= controls_ns)
    owner_tracking = bool(owner_live and owner_reference_healthy and motion.phase == 'track' and not motion.reasons and not motion.resolvedStop)
    if owner_tracking and not motion.continuationActive:
      times = (motion.referenceSourceMonoTime, motion.referenceEpochMonoTime, motion.referenceProducerMonoTime,
               motion.referenceProducerInstanceMonoTime, motion.planMonoTime, motion.modelMonoTime, motion.referenceConsumerMonoTime)
      if (all(valid_timestamp(value) for value in times)
          and motion.referenceSourceMonoTime == motion.referenceEpochMonoTime <= motion.referenceProducerMonoTime
          and motion.referenceProducerInstanceMonoTime <= motion.referenceProducerMonoTime
          and motion.referenceProducerMonoTime <= motion.planMonoTime <= motion.referenceConsumerMonoTime <= controls_ns
          and motion.modelMonoTime <= motion.planMonoTime
          and now_ns - min(motion.referenceSourceMonoTime, motion.modelMonoTime) <= PLAN_LEASE_NS):
        self._continuation_donor = ContinuationDonor(motion.referenceProducerMonoTime, motion.referenceSourceMonoTime,
                                                    motion.modelMonoTime, motion.referenceProducerInstanceMonoTime)
    native_healthy = bool(reference is not None and self.sm.valid['longitudinalPlan'] and plan.solverStatus == 0
                          and plan.motionIntent.valid and plan.motionIntent.ready
                          and self.sm.all_checks(['modelV2', 'longitudinalPlan']))
    if plan_failure or motion_degraded:
      offer = ContinuationOffer.from_message(plan.continuation)
      episode = offer.episode_start_ns if offer is not None and offer.version == CONTINUATION_VERSION and reference is not None else None
      budget.note_failure(now_ns, episode_start_ns=episode)
    elif (owner_reference_healthy and not motion.continuationActive and native_healthy
          and motion.referenceProducerMonoTime == reference.producer_ns and motion.modelMonoTime == plan.modelMonoTime):
      budget.note_success(now_ns, producer_ns=reference.producer_ns, source_ns=reference.source_ns, model_ns=plan.modelMonoTime)
    elif (not native_healthy or not owner_available or not motion.referenceTimingValid
          or motion.phase == 'off' and (not inactive_receipt or not history_valid)):
      # A newer LP awaiting its matching consumer receipt is only a join delay.
      # Actual unhealthy/unready inputs break recovery without renewing the
      # episode's absolute deadline. Standby never counts as native success.
      budget.interrupt_recovery()
    if budget.episode_start_ns and valid_timestamp(now_ns):
      self._longitudinal_soft_disable_remaining_ns = max(0, budget.episode_start_ns + UNSTABLE_DISABLE_NS - now_ns)
    if not plan_failure or plan.solverStatus not in CONTINUATION_STATUSES:
      return False, 'not_continuable_failure'
    if plan.solverStatus == STALE_MEASUREMENT_STATUS and not braking_only_continuation(offer):
      return False, 'stale_continuation_not_braking_only'
    # A continuation can keep established healthy TRACK running. It can
    # never open NO_ENTRY, restart a degraded owner, or suppress forceDecel.
    machine = getattr(self, 'state_machine', None)
    if (not self.enabled or machine is None or machine.state != State.enabled or not owner_tracking
        or CS.gasPressed or CS.brakePressed or CS.standstill or CS.cruiseState.standstill or not CS.canValid
        or CS.vEgo < CLOSE_LEAD_STOP_SPEED
        or self.events.contains(ET.OVERRIDE_LONGITUDINAL)
        or self.sm['driverMonitoringState'].noResponseForceDecel or self.sm['controlsState'].forceDecel
        or not history_valid or reference is None):
      return False, 'continuation_authority_unavailable'
    model = self.sm['modelV2']
    if (self.sm.logMonoTime['modelV2'] != plan.modelMonoTime or not self.sm.all_checks(['modelV2'])
        or model.action.shouldStop or model.meta.hardBrakePredicted or plan.fcw
        or not math.isfinite(model.action.desiredAcceleration)):
      return False, 'continuation_model_changed'
    check = self._continuation_reference.consume(plan.reference, plan_ns, now_ns, command_identity=(offer,))
    if not check.valid:
      return False, check.reason
    failed_intent = MotionIntent.from_message(plan.motionIntent)
    if (not failed_intent.is_braking_valid() or plan.shouldStop or failed_intent.reasons or failed_intent.rest_goal_rejected
        or failed_intent.producer_instance_ns != reference.instance_ns):
      return False, 'continuation_failed_intent_invalid'
    donor = self._continuation_donor
    if motion.continuationActive:
      if (offer is None or motion.planMonoTime != plan_ns or motion.modelMonoTime != plan.modelMonoTime
          or motion.referenceProducerMonoTime != reference.producer_ns
          or motion.referenceSourceMonoTime != reference.source_ns
          or motion.referenceProducerInstanceMonoTime != reference.instance_ns
          or motion.continuationEpisodeMonoTime != offer.episode_start_ns
          or motion.continuationDonorProducerMonoTime != offer.donor_producer_ns
          or motion.continuationDonorSourceMonoTime != offer.donor_source_ns
          or motion.continuationDonorModelMonoTime != offer.donor_model_ns
          or motion.continuationValidUntilMonoTime != offer.valid_until_ns):
        return False, 'continuation_receipt_mismatch'
      # The current owner receipt independently proves that its earlier
      # healthy native donor was actually accepted, including after a restart
      # of this observer. It does not refresh any donor/episode timestamp.
      donor = ContinuationDonor(offer.donor_producer_ns, offer.donor_source_ns, offer.donor_model_ns, reference.instance_ns)
    state = self._continuation_history.current_state(now_ns)
    if state is None:
      return False, 'continuation_history_missing'
    reason = validate_continuation(offer, donor=donor, reference=reference, budget=budget,
                                   plan_ns=plan_ns, model_ns=plan.modelMonoTime, now_ns=now_ns, speed=state[0], accel=state[1],
                                   cruise_kph=CS.vCruise)
    if not reason:
      reason = self._continuation_history.check(offer.candidate, reference, now_ns)
    if reason:
      return False, reason
    target = execution_target(offer, reference, now_ns, self.CP.longitudinalActuatorDelay)
    if target is None or model.action.desiredAcceleration < 0.0 and target > model.action.desiredAcceleration:
      return False, 'continuation_current_braking_cap'
    return True, 'owner_continuation' if motion.continuationActive else 'qualified_offer_before_owner'

  def update_events(self, CS):
    """Compute onroadEvents from carState"""

    self.events.clear()
    self.events_sp.clear()

    if self.sm['controlsState'].lateralControlState.which() == 'debugState':
      self.events.add(EventName.joystickDebug)
      self.startup_event = None

    loading = self.usbgpu_loading
    if self.big_model_loading and not loading:
      self.big_model_ready_t = time.monotonic()
      self.events_sp.add(custom.OnroadEventSP.EventName.bigModelReady)
    self.big_model_loading = loading
    if self.big_model_loading:
      self.events.add(EventName.bigModelLoading)

    big_active = self.usbgpu_active
    chestnut_present = self.sm['deviceState'].chestnutPresent
    model_unavailable = big_active is True and self.sm.seen['modelV2'] and not self.sm.alive['modelV2']
    big_failed = big_active is False or model_unavailable or (self.big_model_active and not chestnut_present)
    if big_failed and not self.big_model_failed:
      self.events.add(EventName.bigModelFailed)
    self.big_model_failed = big_failed

    # soft disable if the big model fails
    if big_active:
      self.big_model_active = True
    if not self.enabled and not model_unavailable:
      self.big_model_active = False

    if self.sm.recv_frame['lateralManeuverPlan'] > 0:
      self.events.add(EventName.lateralManeuver)
      self.startup_event = None
    elif self.sm.recv_frame['alertDebug'] > 0:
      self.events.add(EventName.longitudinalManeuver)
      self.startup_event = None

    # Add startup event
    if self.startup_event is not None:
      self.events.add(self.startup_event)
      self.startup_event = None

    # Don't add any more events if not initialized
    if not self.initialized:
      self.events.add(EventName.selfdriveInitializing)
      return

    # Check for user bookmark press
    if self.sm.updated['userBookmark']:
      self.events.add(EventName.userBookmark)

    # Don't add any more events while in dashcam mode
    if self.cp_passive:
      return

    # Block resume if cruise never previously enabled
    resume_pressed = any(be.type in (ButtonType.accelCruise, ButtonType.resumeCruise) for be in CS.buttonEvents)
    if not self.cp_pcm_cruise and CS.vCruise > 250 and resume_pressed:
      self.events.add(EventName.resumeBlocked)

    # Handle DM
    if not self.cp_not_car:
      # Block engaging until lockout times out or ignition reset
      # One reader, nine uses. sm.update() only runs in data_sample(), so sm['x'] cannot change
      # inside a single update_events call and hoisting is exact. `lockout` and `alertLevel`
      # were each read three times.
      dms = self.sm['driverMonitoringState']
      dms_lockout = dms.lockout
      dms_alert_level = dms.alertLevel
      if dms_lockout and not self.dm_lockout_set:
        self.params.put_bool("DriverTooDistracted", True)
        self.dm_lockout_set = True
      elif not dms_lockout and self.dm_lockout_set:
        self.params.remove("DriverTooDistracted")
        self.dm_lockout_set = False
      # No entry conditions
      if dms_lockout or dms.alwaysOnLockout:
        self.events.add(EventName.tooDistracted)
      # Alerts
      vision_dm = dms.activePolicy == MonitoringPolicy.vision
      if dms_alert_level == AlertLevel.one:
        self.events.add(EventName.driverDistracted1 if vision_dm else EventName.driverUnresponsive1)
      elif dms_alert_level == AlertLevel.two:
        self.events.add(EventName.driverDistracted2 if vision_dm else EventName.driverUnresponsive2)
      elif dms_alert_level == AlertLevel.three:
        self.events.add(EventName.driverDistracted3 if vision_dm else EventName.driverUnresponsive3)
      # Warn consistent DM uncertainty
      if dms.visionPolicyState.uncertainOffroadAlertPercent >= 100 and not self.dm_uncertain_alerted:
        set_offroad_alert("Offroad_DriverMonitoringUncertain", True)
        self.dm_uncertain_alerted = True
      self.events_sp.add_from_msg(self.sm['longitudinalPlanSP'].events)

    # Add car events, ignore if CAN isn't valid
    if CS.canValid:
      self.events.add_from(self.car_events.update(CS, self.CS_prev, self.sm['carControl']))
      self.events_sp.add_from(self.car_events_sp.update(CS, self.events))

      if self.cp_not_car:
        # wait for everything to init first
        if self.sm.frame > int(2. / DT_CTRL) and self.initialized:
          # body always wants to enable
          self.events.add(EventName.pcmEnable)

      # Disable on rising edge of accelerator or brake. Also disable on brake when speed > 0
      if (CS.gasPressed and not self.CS_prev.gasPressed and self.disengage_on_accelerator) or \
        (CS.brakePressed and (not self.CS_prev.brakePressed or not CS.standstill)) or \
        (CS.regenBraking and (not self.CS_prev.regenBraking or not CS.standstill)):
        self.events.add(EventName.pedalPressed)

    # Create events for temperature, disk space, and memory
    if self.sm['deviceState'].thermalStatus >= ThermalStatus.overheated:
      self.events.add(EventName.overheat)
    if self.sm['deviceState'].freeSpacePercent < 7 and not SIMULATION:
      self.events.add(EventName.outOfSpace)
    if self.sm['deviceState'].memoryUsagePercent > 90 and not SIMULATION:
      self.events.add(EventName.lowMemory)

    # Alert if fan isn't spinning for 5 seconds
    if self.sm['peripheralState'].pandaType != log.PandaState.PandaType.unknown:
      if self.sm['peripheralState'].fanSpeedRpm < 500 and self.sm['deviceState'].fanSpeedPercentDesired > 50:
        # allow enough time for the fan controller in the panda to recover from stalls
        if (self.sm.frame - self.last_functional_fan_frame) * DT_CTRL > 15.0:
          self.events.add(EventName.fanMalfunction)
      else:
        self.last_functional_fan_frame = self.sm.frame

    # Handle calibration status
    cal_status = self.sm['extrinsicsCalibration'].calStatus
    if cal_status != log.ExtrinsicsCalibration.Status.calibrated:
      if cal_status == log.ExtrinsicsCalibration.Status.uncalibrated:
        self.events.add(EventName.calibrationIncomplete)
      elif cal_status == log.ExtrinsicsCalibration.Status.recalibrating:
        if not self.recalibrating_seen:
          set_offroad_alert("Offroad_Recalibration", True)
        self.recalibrating_seen = True
        self.events.add(EventName.calibrationRecalibrating)
      else:
        self.events.add(EventName.calibrationInvalid)

    # Lane departure warning
    if self.is_ldw_enabled and self.sm.valid['driverAssistance']:
      if self.sm['driverAssistance'].leftLaneDeparture or self.sm['driverAssistance'].rightLaneDeparture:
        self.events.add(EventName.ldw)

    # ******************************************************************************************
    #  NOTE: To fork maintainers.
    #  Disabling or nerfing safety features will get you and your users banned from our servers.
    #  We recommend that you do not change these numbers from the defaults.
    if self.sm.updated['extrinsicsCalibration']:
      self.pose_calibrator.feed_extrinsics_calibration(self.sm['extrinsicsCalibration'])
    if self.sm.updated['deviceMotion']:
      device_motion = Pose.from_device_motion(self.sm['deviceMotion'])
      self.calibrated_pose = self.pose_calibrator.build_calibrated_pose(device_motion)

    if self.calibrated_pose is not None and not self.cp_not_car:
      excessive_actuation = self.excessive_actuation_check.update(self.sm, CS, self.calibrated_pose)
      if not self.excessive_actuation and excessive_actuation is not None:
        set_offroad_alert("Offroad_ExcessiveActuation", True, extra_text=str(excessive_actuation))
        self.excessive_actuation = True

    if self.excessive_actuation:
      self.events.add(EventName.excessiveActuation)
    # ******************************************************************************************

    # Handle lane change
    if self.sm['modelV2'].meta.laneChangeState == LaneChangeState.preLaneChange:
      direction = self.sm['modelV2'].meta.laneChangeDirection
      mdv2sp = self.sm['modelDataV2SP']

      if (CS.leftBlindspot and direction == LaneChangeDirection.left) or \
         (CS.rightBlindspot and direction == LaneChangeDirection.right):
        self.events.add(EventName.laneChangeBlocked)

      elif (mdv2sp.leftLaneChangeEdgeBlock and direction == LaneChangeDirection.left) or \
           (mdv2sp.rightLaneChangeEdgeBlock and direction == LaneChangeDirection.right):
        self.events_sp.add(custom.OnroadEventSP.EventName.laneChangeRoadEdge)

      else:
        if direction == LaneChangeDirection.left:
          self.events.add(EventName.preLaneChangeLeft)
        else:
          self.events.add(EventName.preLaneChangeRight)
    elif self.sm['modelV2'].meta.laneChangeState in (LaneChangeState.laneChangeStarting,
                                                    LaneChangeState.laneChangeFinishing):
      self.events.add(EventName.laneChange)

    # Handle lane turn
    lane_turn_direction = self.sm['modelDataV2SP'].laneTurnDirection
    if lane_turn_direction == TurnDirection.turnLeft:
      self.events_sp.add(custom.OnroadEventSP.EventName.laneTurnLeft)
    elif lane_turn_direction == TurnDirection.turnRight:
      self.events_sp.add(custom.OnroadEventSP.EventName.laneTurnRight)

    for i, pandaState in enumerate(self.sm['pandaStates']):
      # All pandas must match the list of safetyConfigs, and if outside this list, must be silent or noOutput
      if i < len(self.cp_safety_configs):
        want_model, want_param = self.cp_safety_configs[i]
        safety_mismatch = pandaState.safetyModel != want_model or \
                          pandaState.safetyParam != want_param or \
                          pandaState.alternativeExperience != self.cp_alt_experience
      else:
        safety_mismatch = pandaState.safetyModel not in IGNORED_SAFETY_MODES

      # safety mismatch allows some time for pandad to set the safety mode and publish it back from panda
      if (safety_mismatch and self.sm.frame*DT_CTRL > 10.) or pandaState.safetyRxChecksInvalid or self.mismatch_counter >= 200:
        self.events.add(EventName.controlsMismatch)

      if log.PandaState.FaultType.relayMalfunction in pandaState.faults:
        self.events.add(EventName.relayMalfunction)

    # Handle HW and system malfunctions
    # Order is very intentional here. Be careful when modifying this.
    # All events here should at least have NO_ENTRY and SOFT_DISABLE.
    num_events = len(self.events)

    if self.big_model_active and big_failed:
      self.events.add(EventName.bigModelFailed)

    # managerState publishes at 2 Hz (cereal/services.py), so this set was being rebuilt on all
    # 100 frames per second -- 50 identical recomputations between two updates.
    #
    # Keyed on recv_frame, NOT on sm.updated[]. updated[] is true only on the frame the message
    # arrived, and this line is BELOW the `if not self.initialized: return` at the top of
    # update_events -- so during initialisation the arrivals happen (sm.update runs in
    # data_sample, which step() calls first) while this line is never reached. A cache keyed on
    # updated[] would therefore still hold its seed at the frame `initialized` flips, and would
    # stay stale until the next arrival up to 500 ms later, suppressing processNotRunning
    # (ET.NO_ENTRY) in exactly the window where engagement first becomes possible.
    # recv_frame is monotonic and survives skipped frames, so it cannot miss an update.
    #
    # Only the comprehension moves. The control flow below is deliberately untouched: the
    # `else` on the processNotRunning check feeds the camera-malfunction branch, which is a
    # 100 Hz concern and must keep being evaluated on every frame.
    if self.sm.recv_frame['managerState'] != self.not_running_frame:
      self.not_running_frame = self.sm.recv_frame['managerState']
      # Rebind, never mutate: not_running_prev below aliases this set.
      self.not_running = {p.name for p in self.sm['managerState'].processes
                          if not p.running and p.shouldBeRunning}
    not_running = self.not_running
    if self.sm.recv_frame['managerState'] and len(not_running):
      if not_running != self.not_running_prev:
        cloudlog.event("process_not_running", not_running=not_running, error=True)
      self.not_running_prev = not_running
    if self.sm.recv_frame['managerState'] and (not_running - self.ignored_processes):
      self.events.add(EventName.processNotRunning)
    else:
      if not SIMULATION and not self.rk.lagging:
        if not self.sm.all_alive(self.camera_packets):
          self.events.add(EventName.cameraMalfunction)
        elif not self.sm.all_freq_ok(self.camera_packets):
          self.events.add(EventName.cameraFrameRate)
    if not REPLAY and self.rk.lagging:
      self.events.add(EventName.selfdrivedLagging)
    if self.cp_olc:
      # radarState publishes at 20 Hz, so to_dict() -- which builds a fresh dict out of the
      # capnp struct -- was running 5 times per message. The two direct field reads below are
      # cheap attribute access and stay live; only the dict build is cached. Keyed on
      # recv_frame rather than sm.updated[] for the same reason as managerState above: this
      # line sits below update_events' early return, so updated[] would miss every arrival
      # that lands before initialisation completes.
      if self.sm.recv_frame['radarState'] != self.radar_any_error_frame:
        self.radar_any_error_frame = self.sm.recv_frame['radarState']
        self.radar_any_error = any(self.sm['radarState'].radarErrors.to_dict().values())
      if self.sm['radarState'].radarErrors.canError:
        self.events.add(EventName.canError)
      elif self.sm['radarState'].radarErrors.radarUnavailableTemporary:
        self.events.add(EventName.radarTempUnavailable)
      elif self.radar_any_error:
        self.events.add(EventName.radarFault)
    if CS.canTimeout:
      self.events.add(EventName.canBusMissing)
    elif not CS.canValid:
      self.events.add(EventName.canError)

    # generic catch-all. ideally, a more specific event should be added above instead
    has_disable_events = self.events.contains(ET.NO_ENTRY) and (self.events.contains(ET.SOFT_DISABLE) or self.events.contains(ET.IMMEDIATE_DISABLE))
    no_system_errors = (not has_disable_events) or (len(self.events) == num_events)
    warmup_sec = 5.
    big_model_settling = self.big_model_loading or time.monotonic() < self.big_model_ready_t + warmup_sec
    event_now_ns = time.monotonic_ns()
    plan_failure = fresh_longitudinal_planning_failure(self.sm, self.CP, getattr(self, 'CP_SP', None), event_now_ns)
    # The invalid plan still reaches its consumer as invalid. Only its event
    # classification changes; every independent transport failure retains the
    # generic alert. Compute these before adding the dedicated disable event.
    health_services = [s for s in self.sm.services if s != 'longitudinalPlan'] if plan_failure else self.sm.services
    if not self.sm.all_checks(health_services) and no_system_errors and not big_model_settling:
      if not self.sm.all_alive(health_services):
        self.events.add(EventName.commIssue)
      elif not self.sm.all_freq_ok(health_services):
        self.events.add(EventName.commIssueAvgFreq)
      else:
        self.events.add(EventName.commIssue)

      logs = {
        'invalid': [s for s in health_services if not self.sm.valid[s]],
        'not_alive': [s for s in health_services if not self.sm.alive[s]],
        'not_freq_ok': [s for s in health_services if not self.sm.freq_ok[s]],
      }
      if logs != self.logged_comm_issue:
        cloudlog.event("commIssue", error=True, **logs)
        self.logged_comm_issue = logs
    else:
      self.logged_comm_issue = None

    if not self.cp_not_car and not big_model_settling:  # localization has nothing to work with during the load
      if not self.sm['deviceMotion'].posenetOK:
        self.events.add(EventName.posenetInvalid)
      if not self.sm['deviceMotion'].inputsOK:
        self.events.add(EventName.locationdTemporaryError)
      if (not self.sm['vehicleParameters'].valid and cal_status == log.ExtrinsicsCalibration.Status.calibrated and
          not TESTING_CLOSET and (not SIMULATION or REPLAY)):
        self.events.add(EventName.paramsdTemporaryError)

    # conservative HW alert. if the data or frequency are off, locationd will throw an error
    if any((self.sm.frame - self.sm.recv_frame[s])*DT_CTRL > 10. for s in self.sensor_packets):
      self.events.add(EventName.sensorDataInvalid)

    if not REPLAY:
      # Check for mismatch between openpilot and car's PCM
      cruise_mismatch = CS.cruiseState.enabled and (not self.enabled or not self.cp_pcm_cruise)
      self.cruise_mismatch_counter = self.cruise_mismatch_counter + 1 if cruise_mismatch else 0
      if self.cruise_mismatch_counter > int(6. / DT_CTRL):
        self.events.add(EventName.cruiseMismatch)

    # Send a "steering required alert" if saturation count has reached the limit
    if CS.steeringPressed:
      self.last_steering_pressed_frame = self.sm.frame
    recent_steer_pressed = (self.sm.frame - self.last_steering_pressed_frame)*DT_CTRL < 2.0
    controlstate = self.sm['controlsState']
    observe(getattr(self, "longitudinal_observation", None), "motion_check", controlstate)
    motion = controlstate.motionState
    motion_degraded = (self.CP.openpilotLongitudinalControl and motion.version == MOTION_INTENT_VERSION and motion.degraded)
    continuation_active, continuation_reason = self._continuation_event_status(CS, event_now_ns, plan_failure, motion_degraded)
    longitudinal_unavailable = (plan_failure and not continuation_active) or motion_degraded
    if plan_failure and self.sm.logMonoTime['longitudinalPlan'] != getattr(self, '_logged_failed_plan_ns', 0):
      self._logged_failed_plan_ns = self.sm.logMonoTime['longitudinalPlan']
      cloudlog.event('longitudinalSolverFailure', solverStatus=int(self.sm['longitudinalPlan'].solverStatus),
                     planMonoTime=int(self._logged_failed_plan_ns), continuation=continuation_active,
                     continuationReason=continuation_reason, episodeStartMonoTime=self._longitudinal_failure_budget.episode_start_ns)
    if longitudinal_unavailable:
      self.events.add(EventName.longitudinalControlUnavailable)
      observe(getattr(self, "longitudinal_observation", None), "longitudinal_control_unavailable_added")
    if longitudinal_unavailable != getattr(self, 'longitudinal_unavailable', False):
      plan = self.sm['longitudinalPlan']
      cloudlog.event("longitudinalControlUnavailable", error=longitudinal_unavailable,
                     transition="entered" if longitudinal_unavailable else "recovered",
                     latest_plan_mono_time=int(self.sm.logMonoTime['longitudinalPlan']),
                     latest_plan_model_mono_time=int(plan.modelMonoTime), latest_plan_solver_status=int(plan.solverStatus),
                     latest_plan_valid=bool(self.sm.valid['longitudinalPlan']), planning_failure_qualified=plan_failure,
                     controls_state_mono_time=int(self.sm.logMonoTime['controlsState']),
                     consumed_plan_mono_time=int(motion.planMonoTime), consumed_model_mono_time=int(motion.modelMonoTime),
                     motion_degraded=motion_degraded, motion_reasons=int(motion.reasons),
                     waiting_for_fresh_input=bool(motion.waitingForFreshInput))
      self.longitudinal_unavailable = longitudinal_unavailable
    lac = getattr(controlstate.lateralControlState, controlstate.lateralControlState.which())
    if lac.active and not recent_steer_pressed and not self.cp_not_car:
      clipped_speed = max(CS.vEgo, 0.3)
      actual_lateral_accel = controlstate.curvature * (clipped_speed**2)
      desired_lateral_accel = self.sm['modelV2'].action.desiredCurvature * (clipped_speed**2)
      undershooting = abs(desired_lateral_accel) / abs(1e-3 + actual_lateral_accel) > 1.2
      turning = abs(desired_lateral_accel) > 1.0
      # TODO: lac.saturated includes speed and other checks, should be pulled out
      if undershooting and turning and lac.saturated:
        self.events.add(EventName.steerSaturated)

    # Check for FCW
    stock_long_is_braking = self.enabled and not self.cp_olc and CS.aEgo < -1.25
    model_fcw = self.sm['modelV2'].meta.hardBrakePredicted and not CS.brakePressed and not stock_long_is_braking
    planner_fcw = self.sm['longitudinalPlan'].fcw and self.enabled
    if (planner_fcw or model_fcw) and not self.cp_not_car:
      self.events.add(EventName.fcw)

    # GPS checks
    gps_ok = self.sm.recv_frame[self.gps_location_service] > 0 and (self.sm.frame - self.sm.recv_frame[self.gps_location_service]) * DT_CTRL < 2.0
    if not gps_ok and self.sm['deviceMotion'].inputsOK and (self.distance_traveled > 1500):
      self.events.add(EventName.noGps)
    if gps_ok:
      self.distance_traveled = 0
    self.distance_traveled += abs(CS.vEgo) * DT_CTRL

    # TODO: fix simulator
    if not SIMULATION or REPLAY:
      if self.sm['modelV2'].frameDropPerc > 1:
        self.events.add(EventName.modeldLagging)

    # mute canBusMissing event if in Park, as it sometimes may trigger a false alarm with MADS in Paused state
    if CS.gearShifter == car.CarState.GearShifter.park and self.mads.enabled:
      self.events.remove(EventName.canBusMissing)

    CruiseHelper.update(self, CS, self.events_sp, self.experimental_mode)

    # decrement personality on distance button press
    if self.cp_olc:
      if any(not be.pressed and be.type == ButtonType.gapAdjustCruise for be in CS.buttonEvents):
        if not self.experimental_mode_switched:
          self.personality = (self.personality - 1) % 3
          self.params.put('LongitudinalPersonality', self.personality)
          self.events.add(EventName.personalityChanged)
        self.experimental_mode_switched = False

    self.icbm.run(CS, self.sm['carControl'], self.sm['longitudinalPlanSP'], self.is_metric)

  def data_sample(self):
    # This recv BLOCKS -- it is what paces the process at 100 Hz, not the Ratekeeper, which only
    # calls monitor_time(). Timed separately from everything after it so the wait is never read
    # as work: here wall is ~10 ms and CPU is ~0.
    _car_state = messaging.recv_one(self.car_state_sock)
    if _car_state is not None:
      self._car_state_mono_ns = int(_car_state.logMonoTime)
      self._car_state_valid = bool(_car_state.valid)
    if self.prof:
      self.prof.mark("recv_carState")
    CS = _car_state.carState if _car_state else self.CS_prev

    self.sm.update(0)
    observe(getattr(self, "longitudinal_observation", None), "begin", self.sm, _car_state, CS)
    if self.prof:
      self.prof.mark("sm_update")

    if not self.initialized:
      all_valid = CS.canValid and self.sm.all_checks()
      timed_out = self.sm.frame * DT_CTRL > 6.
      if all_valid or timed_out or (SIMULATION and not REPLAY):
        available_streams = VisionIpcClient.available_streams("camerad", block=False)
        if VisionStreamType.VISION_STREAM_NARROW_ROAD not in available_streams:
          self.sm.ignore_alive.append('narrowRoadCameraState')
          self.sm.ignore_valid.append('narrowRoadCameraState')
        if VisionStreamType.VISION_STREAM_WIDE_ROAD not in available_streams:
          self.sm.ignore_alive.append('wideRoadCameraState')
          self.sm.ignore_valid.append('wideRoadCameraState')

        if REPLAY and any(ps.controlsAllowed for ps in self.sm['pandaStates']):
          self.state_machine.state = State.enabled

        self.initialized = True
        cloudlog.event(
          "selfdrived.initialized",
          dt=self.sm.frame*DT_CTRL,
          timeout=timed_out,
          canValid=CS.canValid,
          invalid=[s for s, valid in self.sm.valid.items() if not valid],
          not_alive=[s for s, alive in self.sm.alive.items() if not alive],
          not_freq_ok=[s for s, freq_ok in self.sm.freq_ok.items() if not freq_ok],
          error=True,
        )

    # When the panda and selfdrived do not agree on controls_allowed
    # we want to disengage openpilot. However the status from the panda goes through
    # another socket other than the CAN messages and one can arrive earlier than the other.
    # Therefore we allow a mismatch for two samples, then we trigger the disengagement.
    if not self.enabled:
      self.mismatch_counter = 0

    # All pandas not in silent mode must have controlsAllowed when openpilot is enabled
    if self.enabled and any(not ps.controlsAllowed for ps in self.sm['pandaStates']
           if ps.safetyModel not in IGNORED_SAFETY_MODES):
      self.mismatch_counter += 1

    return CS

  def update_engagement_state(self, CS):
    if not hasattr(self, 'engagement_policy'):
      self.engagement_policy = EngagementPolicy()
    decision = self.engagement_policy.update(self, CS, time.monotonic_ns())
    self.mads.longitudinal_failure_isolated = decision.preserve_lateral
    self.mads.model_lag_recovered = decision.recovered_model_lag_lateral
    self.enabled, self.active = self.state_machine.update(
      LongitudinalEvents(self.events, decision.preserve_longitudinal,
                         recovered_model_lag=decision.recovered_model_lag),
      soft_disable_remaining_ns=getattr(self, '_longitudinal_soft_disable_remaining_ns', None),
      hold_for_takeover=decision.hold_for_takeover,
    )
    signature = (decision.preserve_lateral, decision.preserve_longitudinal, decision.hold_for_takeover)
    if signature != getattr(self, '_takeover_signature', (False, False, False)):
      cloudlog.event('controlTakeover', lateralRetained=signature[0], longitudinalRetained=signature[1],
                     stoppedHandoff=signature[2], holdRemainingNs=decision.hold_remaining_ns)
    self._takeover_signature = signature
    if decision.recovered_model_lag != getattr(self, '_model_lag_recovered', False):
      motion = self.sm['controlsState'].motionState
      cloudlog.event('modelLagRecovery', recovered=decision.recovered_model_lag,
                     frameId=int(self.sm['modelV2'].frameId), modelMonoTime=int(self.sm.logMonoTime['modelV2']),
                     consumedModelMonoTime=int(motion.modelMonoTime), consumedPlanMonoTime=int(motion.planMonoTime))
    self._model_lag_recovered = decision.recovered_model_lag

  def update_alerts(self, CS):
    clear_event_types = set()
    if ET.WARNING not in self.state_machine.current_alert_types:
      clear_event_types.add(ET.WARNING)
    if self.enabled:
      clear_event_types.add(ET.NO_ENTRY)

    pers = LONGITUDINAL_PERSONALITY_MAP[self.personality]
    # Each axis owns its deadline. Only a currently soft-disabling axis may
    # determine the urgency of soft-disable alerts.
    timers = []
    if self.state_machine.state == State.softDisabling:
      timers.append(self.state_machine.soft_disable_timer)
    mads_machine = getattr(self.mads, 'state_machine', None)
    mads_state = mads_machine.state if mads_machine is not None else None
    if mads_state == MadsState.softDisabling:
      timers.append(mads_machine.soft_disable_timer)
    callback_args = [self.CP, CS, self.sm, self.is_metric, min(timers, default=0), pers]

    alert_types = self.state_machine.current_alert_types.copy()
    if mads_state == MadsState.softDisabling and ET.SOFT_DISABLE not in alert_types:
      alert_types.append(ET.SOFT_DISABLE)
    alerts = self.events.create_alerts(alert_types, callback_args)
    alerts_sp = self.events_sp.create_alerts(alert_types, callback_args)
    model_recovered = getattr(self, '_model_lag_recovered', False)
    # Disabled alone does not prove this fault withdrew steering: the driver
    # may have been using longitudinal control without MADS from the start.
    if (not self.enabled or not self.events.has(EventName.modeldLagging)
        or not getattr(self.mads, 'enabled_toggle', False)
        or mads_state not in (MadsState.softDisabling, MadsState.disabled)):
      self._model_lag_lateral_exit = False
    elif mads_state == MadsState.softDisabling and not getattr(self.mads, 'model_lag_recovered', False):
      self._model_lag_lateral_exit = True
    lateral_takeover = ''
    if (model_recovered and self.enabled and self.state_machine.state != State.softDisabling
        and getattr(self.mads, 'enabled_toggle', False) and not getattr(self.mads, 'model_lag_recovered', False)):
      if mads_state == MadsState.softDisabling:
        lateral_takeover = 'pending'
      elif mads_state == MadsState.disabled and getattr(self, '_model_lag_lateral_exit', False):
        lateral_takeover = 'disabled'
    if mads_state == MadsState.softDisabling:
      model_recovered = model_recovered and getattr(self.mads, 'model_lag_recovered', False)
    alerts = merge_perception_alerts(alerts, self.events, self.AM, self.sm.frame,
                                     recovered_model_lag=model_recovered, lateral_takeover=lateral_takeover)

    takeover_alert = self.engagement_policy.alert(self, CS) if hasattr(self, 'engagement_policy') else None
    if takeover_alert is not None:
      # Replace only the generic longitudinal warning with the explicit axis
      # or stopped-handoff message; event types/NO_ENTRY/logging remain intact.
      generic = 'longitudinalControlUnavailable/softDisable'
      alerts = [alert for alert in alerts if alert.alert_type != generic]
      self.AM.alerts.pop(generic, None)
      alerts.append(takeover_alert)

    alerts += alerts_sp
    if CS.brakePressed:
      # The driver already owns braking. Keep the fault events and engagement
      # gates, but withdraw their prompts, including any minimum-duration cache.
      suppressed = {
        'longitudinalControlUnavailable/noEntry', 'longitudinalControlUnavailable/softDisable',
        'controlTakeover/longitudinal', 'controlTakeover/hold', 'controlTakeover/holdEnded',
      }
      alerts = [alert for alert in alerts if alert.alert_type not in suppressed]
      for alert_type in suppressed:
        self.AM.alerts.pop(alert_type, None)

    self.AM.add_many(self.sm.frame, alerts)
    self.AM.process_alerts(self.sm.frame, clear_event_types)

  def publish_selfdriveState(self, CS):
    # selfdriveState
    ss_msg = messaging.new_message('selfdriveState')
    ss_msg.valid = True
    ss = ss_msg.selfdriveState
    ss.enabled = self.enabled
    ss.active = self.active
    ss.state = self.state_machine.state
    ss.engageable = not self.events.contains(ET.NO_ENTRY)
    ss.experimentalMode = self.experimental_mode
    ss.personality = self.personality

    ss.alertText1 = self.AM.current_alert.alert_text_1
    ss.alertText2 = self.AM.current_alert.alert_text_2
    ss.alertSize = self.AM.current_alert.alert_size
    ss.alertStatus = self.AM.current_alert.alert_status
    ss.alertType = self.AM.current_alert.alert_type
    ss.alertSound = self.AM.current_alert.audible_alert
    ss.alertHudVisual = self.AM.current_alert.visual_alert

    observe(getattr(self, "longitudinal_observation", None), "publish", ss)
    self.pm.send('selfdriveState', ss_msg)

    # onroadEvents - logged every second or on change
    if (self.sm.frame % int(1. / DT_CTRL) == 0) or (self.events.names != self.events_prev):
      ce_send = messaging.new_message('onroadEvents', len(self.events))
      ce_send.valid = True
      ce_send.onroadEvents = self.events.to_msg()
      self.pm.send('onroadEvents', ce_send)
    self.events_prev = self.events.names.copy()

    # selfdriveStateSP
    ss_sp_msg = messaging.new_message('selfdriveStateSP')
    ss_sp_msg.valid = True
    ss_sp = ss_sp_msg.selfdriveStateSP
    mads = ss_sp.mads
    mads.state = self.mads.state_machine.state
    mads.enabled = self.mads.enabled
    mads.active = self.mads.active
    mads.available = self.mads.enabled_toggle

    icbm = ss_sp.intelligentCruiseButtonManagement
    icbm.state = self.icbm.state
    icbm.sendButton = self.icbm.cruise_button
    icbm.vTarget = self.icbm.v_target

    self.button_state_tracker.publish(ss_sp)

    self.pm.send('selfdriveStateSP', ss_sp_msg)

    # onroadEventsSP - logged every second or on change
    if (self.sm.frame % int(1. / DT_CTRL) == 0) or (self.events_sp.names != self.events_sp_prev):
      ce_send_sp = messaging.new_message('onroadEventsSP')
      ce_send_sp.valid = True
      ce_send_sp.onroadEventsSP.events = self.events_sp.to_msg()
      self.pm.send('onroadEventsSP', ce_send_sp)
    self.events_sp_prev = self.events_sp.names.copy()

  def step(self):
    if self.prof:
      self.prof.begin()
    CS = self.data_sample()
    if self.prof:
      self.prof.mark("data_sample_rest")
    self.update_events(CS)
    if self.prof:
      self.prof.mark("update_events")
    observe(getattr(self, "longitudinal_observation", None), "state_machine_input", self)
    if not self.CP.passive and self.initialized:
      self.update_engagement_state(CS)
      observe(getattr(self, "longitudinal_observation", None), "state_machine_output", self)
    if self.prof:
      self.prof.mark("state_machine")
    if not self.CP.notCar:
      self.mads.update(CS)
    self.engagement_policy.observe_states(self, CS)
    if self.prof:
      self.prof.mark("mads")
    self.update_alerts(CS)
    if self.prof:
      self.prof.mark("update_alerts")

    self.button_state_tracker.update(CS)
    if self.prof:
      self.prof.mark("button_tracker")
    self.publish_selfdriveState(CS)
    if self.prof:
      self.prof.mark("publish")
      self.prof.end_frame()

    self.CS_prev = CS

  def params_thread(self, evt):
    while not evt.is_set():
      self.is_metric = self.params.get_bool("IsMetric")
      self.is_ldw_enabled = self.params.get_bool("IsLdwEnabled")
      self.disengage_on_accelerator = self.params.get_bool("DisengageOnAccelerator")
      self.experimental_mode = self.params.get_bool("ExperimentalMode") and self.CP.openpilotLongitudinalControl
      self.personality = self.params.get("LongitudinalPersonality", return_default=True)

      # Read here rather than in update_events, which runs at 100 Hz. Both keys are written by
      # modeld around a model load (sunnypilot/modeld_v2/modeld.py:334-379) -- a seconds-long
      # event -- so 10 Hz resolves every transition they can actually make.
      self.usbgpu_loading = self.params.get_bool("ChestnutLoading")
      self.usbgpu_active = self.params.get("ChestnutActive")

      self.mads.read_params()
      time.sleep(0.1)

  def run(self):
    e = threading.Event()
    t = threading.Thread(target=self.params_thread, args=(e, ))
    try:
      t.start()
      # The writer owns its own thread now. It is a daemon, so it does not need joining here --
      # which also removes the is_alive() dance the hand-built thread needed to avoid replacing
      # a real traceback with a RuntimeError from joining a thread that never started.
      if self.prof:
        self.prof.start()
      while True:
        self.step()
        self.rk.monitor_time()
    finally:
      e.set()
      t.join()
      if self.prof:
        self.prof.stop()


def main():
  config_realtime_process(4, Priority.CTRL_HIGH)
  s = SelfdriveD()
  s.run()

if __name__ == "__main__":
  main()
