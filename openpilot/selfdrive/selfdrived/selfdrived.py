#!/usr/bin/env python3
import os
import time
import threading

import openpilot.cereal.messaging as messaging

from openpilot.cereal import log, custom
from opendbc.car.structs import car
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcClient


from openpilot.common.params import Params
from openpilot.common.realtime import config_realtime_process, Priority, Ratekeeper, DT_CTRL
from openpilot.common.swaglog import cloudlog
from openpilot.common.gps import get_gps_location_service

from openpilot.selfdrive.car.car_events import CarEvents
from openpilot.selfdrive.locationd.helpers import PoseCalibrator, Pose
from openpilot.selfdrive.selfdrived.events import Events, ET
from openpilot.selfdrive.selfdrived.helpers import ExcessiveActuationCheck
from openpilot.selfdrive.selfdrived.state import StateMachine
from openpilot.selfdrive.selfdrived.alertmanager import AlertManager, set_offroad_alert

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


class PhaseTimer:
  """Per-phase wall and CPU timing for step(), written to the log every 30 s. Off by default.

  Why this exists. procs.py measures selfdrived at 26.1% of one core, but that is a whole-process
  number read from /proc, and it cannot say which part of step() spends it. Nothing else here can
  answer that either: this process is paced by a BLOCKING recv_one(carState), so any wall-clock
  profiler charges the wait as work, and cProfile's ~1 us per call would add milliseconds per
  frame to a SCHED_FIFO 53 process sitting in the card -> selfdrived -> controlsd chain.

  Why two clocks and not one. perf_counter_ns is wall, thread_time_ns is this thread's CPU. The
  first phase blocks; there wall is large and CPU is ~0, and that difference is the whole point
  -- it separates "waiting for carState" from "computing". A single clock cannot make that call,
  and reading only CPU would hide the pacing.

  Cost, measured rather than estimated. A full frame of instrumentation -- start(), nine mark()s
  and end_frame() -- costs 6.8 us of CPU on an x86 host (_logs/sm_cost.py's host, 2026-08-29).
  Not the ~1 us a count of clock reads suggests: perf_counter_ns is 86 ns from the vDSO, but
  thread_time_ns is 291 ns because CLOCK_THREAD_CPUTIME_ID is a real syscall, and there are ten
  of each. On the device, at an unmeasured 2-5x, that is 14-34 us per frame: 0.14-0.34% of the
  10 ms period, or 0.5-1.3% of the 2.61 ms of CPU selfdrived actually burns per frame. So the
  instrument perturbs its own subject by about one percent -- an order of magnitude below the
  26.1% it exists to attribute, and therefore usable, but NOT the free measurement a naive count
  of clock reads would predict. Cost is still not the reason it defaults to off; see below.

  Why a marker file and not a Param or an env var. A Param would have to be added to
  common/params_keys.h, which is compiled into libparams_c.so -- so a new key stops being a pure
  Python change and pulls in a host build, a committed aarch64 .so and a sync. An env var in
  launch_env.sh would either need syncing to flip, or an in-place ssh edit that leaves the device
  tree dirty and breaks c3-sync's only verification (an empty git status). /data is not in git,
  so a marker file flips the switch without touching either.

    on:   ssh comma@<device> touch /data/selfdrived_profile   (takes effect at next process start)
    off:  ssh comma@<device> rm /data/selfdrived_profile

  Why it defaults to off, given the cost is negligible. Two reasons, neither of them CPU:
  a permanent debug path on a 100 Hz actuation chain is one more thing every future reader of
  this file has to classify as instrument-or-function; and an event every 30 s for the life of
  the device is long-term noise in a log that boot.py and procs.py have to read.
  """
  MARKER = "/data/selfdrived_profile"

  # 30 s at 100 Hz. Buffers are preallocated because config_realtime_process() calls gc.disable():
  # per-frame appends would grow the arena for the whole drive with nothing to reclaim it.
  PERIOD_FRAMES = 3000

  PHASES = ("recv_carState", "sm_update", "data_sample_rest", "update_events", "state_machine",
            "mads", "update_alerts", "button_tracker", "publish")

  @classmethod
  def create(cls):
    """A PhaseTimer if the marker is present, else None. Checked once at construction, never on
    the hot path -- flipping the marker takes effect at the next process start, which on this
    device means the next ignition cycle."""
    try:
      return cls() if os.path.exists(cls.MARKER) else None
    except OSError:
      return None

  def __init__(self):
    n = self.PERIOD_FRAMES
    self.wall = {p: [0] * n for p in self.PHASES}
    self.cpu = {p: [0] * n for p in self.PHASES}
    self.i = 0
    self.periods = 0
    self._w = 0
    self._c = 0

  def start(self):
    self._w = time.perf_counter_ns()
    self._c = time.thread_time_ns()

  def mark(self, phase):
    w = time.perf_counter_ns()
    c = time.thread_time_ns()
    i = self.i
    self.wall[phase][i] = w - self._w
    self.cpu[phase][i] = c - self._c
    self._w = w
    self._c = c

  def end_frame(self):
    self.i += 1
    if self.i >= self.PERIOD_FRAMES:
      self.flush()

  def flush(self):
    n = self.i
    self.i = 0
    if n < 2:
      return
    self.periods += 1

    def stats(xs):
      s = sorted(xs[:n])
      return {
        "p50": round(s[n // 2] / 1000.0, 1),
        "p95": round(s[min(n - 1, int(n * 0.95))] / 1000.0, 1),
        "p99": round(s[min(n - 1, int(n * 0.99))] / 1000.0, 1),
        "max": round(s[-1] / 1000.0, 1),
        "mean": round(sum(s) / n / 1000.0, 1),
      }

    # Microseconds. cpu_total is what should be compared against procs.py's 26.1%; wall_total is
    # dominated by the blocking recv and is NOT a cost.
    cpu = {p: stats(self.cpu[p]) for p in self.PHASES}
    wall = {p: stats(self.wall[p]) for p in self.PHASES}
    cloudlog.event(
      "selfdrived.phase_timing",
      frames=n,
      period=self.periods,
      cpu_us=cpu,
      wall_us=wall,
      cpu_total_us=round(sum(c["mean"] for c in cpu.values()), 1),
      wall_total_us=round(sum(w["mean"] for w in wall.values()), 1),
    )


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

    self.car_events = CarEvents(self.CP)

    self.pose_calibrator = PoseCalibrator()
    self.calibrated_pose: Pose | None = None
    self.excessive_actuation_check = ExcessiveActuationCheck()
    self.excessive_actuation = self.params.get("Offroad_ExcessiveActuation") is not None
    self.big_model_loading = False
    self.big_model_active = False
    self.big_model_failed = False
    self.big_model_ready_t = 0.

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
    self.not_running_prev = None
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
    self.rk = Ratekeeper(100, print_delay_threshold=None)

    # None unless /data/selfdrived_profile exists at start. Every timing call below is guarded by
    # `if self.prof:`, so with it absent the cost on the 100 Hz path is one attribute test per
    # phase boundary and nothing else.
    self.prof = PhaseTimer.create()
    if self.prof is not None:
      cloudlog.warning("selfdrived: phase timing ENABLED by %s", PhaseTimer.MARKER)

    self.ignored_processes = {'mapd', }

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

  def update_events(self, CS):
    """Compute onroadEvents from carState"""

    self.events.clear()
    self.events_sp.clear()

    if self.sm['controlsState'].lateralControlState.which() == 'debugState':
      self.events.add(EventName.joystickDebug)
      self.startup_event = None

    loading = self.params.get_bool("UsbGpuLoading")
    if self.big_model_loading and not loading:
      self.big_model_ready_t = time.monotonic()
    self.big_model_loading = loading
    if self.big_model_loading:
      self.events.add(EventName.bigModelLoading)

    big_active = self.params.get("UsbGpuActive")
    usbgpu_present = self.sm['deviceState'].chestnutPresent
    model_unavailable = big_active is True and self.sm.seen['modelV2'] and not self.sm.alive['modelV2']
    big_failed = big_active is False or model_unavailable or (self.big_model_active and not usbgpu_present)
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
    if self.CP.passive:
      return

    # Block resume if cruise never previously enabled
    resume_pressed = any(be.type in (ButtonType.accelCruise, ButtonType.resumeCruise) for be in CS.buttonEvents)
    if not self.CP.pcmCruise and CS.vCruise > 250 and resume_pressed:
      self.events.add(EventName.resumeBlocked)

    # Handle DM
    if not self.CP.notCar:
      # Block engaging until lockout times out or ignition reset
      if self.sm['driverMonitoringState'].lockout and not self.dm_lockout_set:
        self.params.put_bool("DriverTooDistracted", True)
        self.dm_lockout_set = True
      elif not self.sm['driverMonitoringState'].lockout and self.dm_lockout_set:
        self.params.remove("DriverTooDistracted")
        self.dm_lockout_set = False
      # No entry conditions
      if self.sm['driverMonitoringState'].lockout or self.sm['driverMonitoringState'].alwaysOnLockout:
        self.events.add(EventName.tooDistracted)
      # Alerts
      vision_dm = self.sm['driverMonitoringState'].activePolicy == MonitoringPolicy.vision
      if self.sm['driverMonitoringState'].alertLevel == AlertLevel.one:
        self.events.add(EventName.driverDistracted1 if vision_dm else EventName.driverUnresponsive1)
      elif self.sm['driverMonitoringState'].alertLevel == AlertLevel.two:
        self.events.add(EventName.driverDistracted2 if vision_dm else EventName.driverUnresponsive2)
      elif self.sm['driverMonitoringState'].alertLevel == AlertLevel.three:
        self.events.add(EventName.driverDistracted3 if vision_dm else EventName.driverUnresponsive3)
      # Warn consistent DM uncertainty
      if self.sm['driverMonitoringState'].visionPolicyState.uncertainOffroadAlertPercent >= 100 and not self.dm_uncertain_alerted:
        set_offroad_alert("Offroad_DriverMonitoringUncertain", True)
        self.dm_uncertain_alerted = True
      self.events_sp.add_from_msg(self.sm['longitudinalPlanSP'].events)

    # Add car events, ignore if CAN isn't valid
    if CS.canValid:
      car_events = self.car_events.update(CS, self.CS_prev, self.sm['carControl']).to_msg()
      self.events.add_from_msg(car_events)

      car_events_sp = self.car_events_sp.update(CS, self.events).to_msg()
      self.events_sp.add_from_msg(car_events_sp)

      if self.CP.notCar:
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

    if self.calibrated_pose is not None and not self.CP.notCar:
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
      if i < len(self.CP.safetyConfigs):
        safety_mismatch = pandaState.safetyModel != self.CP.safetyConfigs[i].safetyModel or \
                          pandaState.safetyParam != self.CP.safetyConfigs[i].safetyParam or \
                          pandaState.alternativeExperience != self.CP.alternativeExperience
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

    not_running = {p.name for p in self.sm['managerState'].processes if not p.running and p.shouldBeRunning}
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
    if self.CP.openpilotLongitudinalControl:
      if self.sm['radarState'].radarErrors.canError:
        self.events.add(EventName.canError)
      elif self.sm['radarState'].radarErrors.radarUnavailableTemporary:
        self.events.add(EventName.radarTempUnavailable)
      elif any(self.sm['radarState'].radarErrors.to_dict().values()):
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
    if not self.sm.all_checks() and no_system_errors and not big_model_settling:  # the load holds modelV2 and friends back on purpose
      if not self.sm.all_alive():
        self.events.add(EventName.commIssue)
      elif not self.sm.all_freq_ok():
        self.events.add(EventName.commIssueAvgFreq)
      else:
        self.events.add(EventName.commIssue)

      logs = {
        'invalid': [s for s, valid in self.sm.valid.items() if not valid],
        'not_alive': [s for s, alive in self.sm.alive.items() if not alive],
        'not_freq_ok': [s for s, freq_ok in self.sm.freq_ok.items() if not freq_ok],
      }
      if logs != self.logged_comm_issue:
        cloudlog.event("commIssue", error=True, **logs)
        self.logged_comm_issue = logs
    else:
      self.logged_comm_issue = None

    if not self.CP.notCar and not big_model_settling:  # localization has nothing to work with during the load
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
      cruise_mismatch = CS.cruiseState.enabled and (not self.enabled or not self.CP.pcmCruise)
      self.cruise_mismatch_counter = self.cruise_mismatch_counter + 1 if cruise_mismatch else 0
      if self.cruise_mismatch_counter > int(6. / DT_CTRL):
        self.events.add(EventName.cruiseMismatch)

    # Send a "steering required alert" if saturation count has reached the limit
    if CS.steeringPressed:
      self.last_steering_pressed_frame = self.sm.frame
    recent_steer_pressed = (self.sm.frame - self.last_steering_pressed_frame)*DT_CTRL < 2.0
    controlstate = self.sm['controlsState']
    lac = getattr(controlstate.lateralControlState, controlstate.lateralControlState.which())
    if lac.active and not recent_steer_pressed and not self.CP.notCar:
      clipped_speed = max(CS.vEgo, 0.3)
      actual_lateral_accel = controlstate.curvature * (clipped_speed**2)
      desired_lateral_accel = self.sm['modelV2'].action.desiredCurvature * (clipped_speed**2)
      undershooting = abs(desired_lateral_accel) / abs(1e-3 + actual_lateral_accel) > 1.2
      turning = abs(desired_lateral_accel) > 1.0
      # TODO: lac.saturated includes speed and other checks, should be pulled out
      if undershooting and turning and lac.saturated:
        self.events.add(EventName.steerSaturated)

    # Check for FCW
    stock_long_is_braking = self.enabled and not self.CP.openpilotLongitudinalControl and CS.aEgo < -1.25
    model_fcw = self.sm['modelV2'].meta.hardBrakePredicted and not CS.brakePressed and not stock_long_is_braking
    planner_fcw = self.sm['longitudinalPlan'].fcw and self.enabled
    if (planner_fcw or model_fcw) and not self.CP.notCar:
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
    if self.CP.openpilotLongitudinalControl:
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
    if self.prof:
      self.prof.mark("recv_carState")
    CS = _car_state.carState if _car_state else self.CS_prev

    self.sm.update(0)
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

  def update_alerts(self, CS):
    clear_event_types = set()
    if ET.WARNING not in self.state_machine.current_alert_types:
      clear_event_types.add(ET.WARNING)
    if self.enabled:
      clear_event_types.add(ET.NO_ENTRY)

    pers = LONGITUDINAL_PERSONALITY_MAP[self.personality]
    callback_args = [self.CP, CS, self.sm, self.is_metric,
                     self.state_machine.soft_disable_timer, pers]

    alerts = self.events.create_alerts(self.state_machine.current_alert_types, callback_args)
    alerts_sp = self.events_sp.create_alerts(self.state_machine.current_alert_types, callback_args)

    self.AM.add_many(self.sm.frame, alerts + alerts_sp)
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
      self.prof.start()
    CS = self.data_sample()
    if self.prof:
      self.prof.mark("data_sample_rest")
    self.update_events(CS)
    if self.prof:
      self.prof.mark("update_events")
    if not self.CP.passive and self.initialized:
      self.enabled, self.active = self.state_machine.update(self.events)
    if self.prof:
      self.prof.mark("state_machine")
    if not self.CP.notCar:
      self.mads.update(CS)
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

      self.mads.read_params()
      time.sleep(0.1)

  def run(self):
    e = threading.Event()
    t = threading.Thread(target=self.params_thread, args=(e, ))
    try:
      t.start()
      while True:
        self.step()
        self.rk.monitor_time()
    finally:
      e.set()
      t.join()


def main():
  config_realtime_process(4, Priority.CTRL_HIGH)
  s = SelfdriveD()
  s.run()

if __name__ == "__main__":
  main()
