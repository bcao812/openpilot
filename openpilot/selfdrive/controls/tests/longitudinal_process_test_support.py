"""Real process and message harnesses with bounded transport collaborators."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR
from opendbc.car.structs import car
from openpilot.cereal import custom, log, messaging
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.controlsd import Controls
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION, StopReason
from openpilot.selfdrive.controls.lib.reference_timing import ReferenceTimingConsumer, ReferenceTimingProducer
from openpilot.selfdrive.selfdrived.events import Events
from openpilot.selfdrive.selfdrived.selfdrived import SelfdriveD
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP


CONTROL_STEP_NS = round(DT_CTRL * 1e9)


State = log.SelfdriveState.OpenpilotState


CONTROL_SERVICES = [
  'vehicleParameters', 'modelV2', 'selfdriveState', 'longitudinalPlan', 'lateralManeuverPlan', 'carState', 'carOutput',
  'driverMonitoringState', 'onroadEvents', 'driverAssistance', 'lateralDelay', 'selfdriveStateSP',
]


def new_event(service, timestamp=2_000_000_000):
  size = 0 if service in ('onroadEvents', 'pandaStates') else None
  return messaging.new_message(service, size, valid=True, logMonoTime=timestamp)


def serialized(event):
  if isinstance(event, bytes):
    return messaging.log_from_bytes(event)
  event.clear_write_flag()
  return messaging.log_from_bytes(event.to_bytes())


def bounded_submaster(services):
  """Keep real SubMaster ingestion, validity and alive checks; exclude sockets and average-frequency timing."""
  with patch('openpilot.cereal.messaging.Poller'), patch('openpilot.cereal.messaging.sub_sock'):
    state = messaging.SubMaster(services, frequency=100.0, ignore_avg_freq=services)
  state.simulation = False
  return state


class CapturingPublisher:
  def __init__(self):
    self.messages = {}
    self.history = []

  def send(self, service, message):
    event = serialized(message)
    self.messages[service] = event
    self.history.append((service, event))


class ControlsHarness:
  """Run real state_control/publish and LongControl, with bounded lateral and vehicle-model collaborators."""

  def __init__(self, *, platform=CAR.HONDA_ACCORD, openpilot_long=True, interceptor=False):
    self.now_ns = 2_000_000_000
    self.events = {service: new_event(service) for service in CONTROL_SERVICES}
    self.car_state = self.events['carState'].carState
    self.car_state.canValid = True
    self.car_state.vEgo = 10.0
    self.car_state.vCruise = 80.0
    self.car_state.vCruiseCluster = 80.0
    self.car_state.cruiseState.enabled = True
    self.selfdrive_state = self.events['selfdriveState'].selfdriveState
    self.selfdrive_state.enabled = True
    self.selfdrive_state.active = True
    self.selfdrive_state.state = State.enabled
    self.plan = self.events['longitudinalPlan'].longitudinalPlan
    self.plan.aTarget = 0.5
    self.plan.motionIntent.valid = True
    self.plan.motionIntent.allowCreep = False

    control = Controls.__new__(Controls)
    control.radar_signal = True
    control.CP = car.CarParams.new_message()
    control.CP.carFingerprint = platform
    control.CP.openpilotLongitudinalControl = openpilot_long
    control.CP.pcmCruise = True
    control.CP.stopAccel = -2.0
    control.CP.longitudinalTuning.kiBP = [0.0]
    control.CP.longitudinalTuning.kiV = [0.0]
    control.CP.lateralTuning.init('pid')
    control.CP_SP = custom.CarParamsSP.new_message()
    control.CP_SP.enableGasInterceptor = interceptor
    control.CI = SimpleNamespace(get_pid_accel_limits=CarInterface.get_pid_accel_limits)
    control.LoC = LongControl(control.CP, control.CP_SP)
    control.reference_timing = ReferenceTimingConsumer()
    control.reference_timing_check = None
    self.reference_producer = ReferenceTimingProducer()
    self._reference_plan_time_ns = None
    self.plan.motionIntent.version = MOTION_INTENT_VERSION if control.LoC.unified else 0
    self.plan.motionIntent.mode = int(control.LoC.longitudinal_mode)
    control.longitudinal_mode_mismatch = False
    control.VM = SimpleNamespace(update_params=Mock(), calc_curvature=Mock(return_value=0.0))
    lateral_log = new_event('controlsState').controlsState.lateralControlState.init('pidState')
    control.LaC = SimpleNamespace(reset=Mock(), update=Mock(return_value=(0.0, 0.0, lateral_log)))
    control.blinker_pause_lateral = SimpleNamespace(update=Mock(return_value=False))
    control.sm = bounded_submaster(CONTROL_SERVICES)
    control.pm = CapturingPublisher()
    control.steer_limited_by_safety = False
    control.curvature = 0.0
    control.desired_curvature = 0.0
    control.calibrated_pose = None
    self.control = control

  def step(self, *, new_plan=True, plan_time_ns=None, model_time_ns=None, now_ns=None,
           reference_source_ns=None, reference_fields=None, intent_fields=None):
    self.now_ns = self.now_ns + CONTROL_STEP_NS if now_ns is None else now_ns
    incoming = []
    for service, event in self.events.items():
      if service == 'longitudinalPlan':
        if not new_plan:
          continue
        event.logMonoTime = self.now_ns if plan_time_ns is None else plan_time_ns
        self.plan.modelMonoTime = self.now_ns if model_time_ns is None else model_time_ns
        if (self.control.LoC.longitudinal_mode == LongitudinalMode.UNIFIED
            and (event.logMonoTime != self._reference_plan_time_ns or reference_source_ns is not None or reference_fields is not None)):
          source_ns = self.now_ns if reference_source_ns is None else reference_source_ns
          self.reference_producer.qualify_source(source_ns, self.car_state.vEgo, self.car_state.aEgo, self.now_ns)
          self.reference_producer.set_initial_state(max(self.car_state.vEgo, 0.0), self.car_state.aEgo)
          self.reference_producer.publish(self.plan.reference)
          self.plan.motionIntent.producerInstanceMonoTime = self.reference_producer.instance_ns
          for field, value in (reference_fields or {}).items():
            setattr(self.plan.reference, field, value)
          self._reference_plan_time_ns = int(event.logMonoTime)
        for field, value in (intent_fields or {}).items():
          setattr(self.plan.motionIntent, field, value)
      else:
        event.logMonoTime = self.now_ns
      incoming.append(serialized(event))
    self.control.sm.update_msgs(self.now_ns / 1e9, incoming)
    with patch('openpilot.selfdrive.controls.controlsd.time.monotonic_ns', return_value=self.now_ns), \
         patch('openpilot.selfdrive.controls.controlsd.time.monotonic', return_value=self.now_ns / 1e9):
      command, lateral_log = self.control.state_control()
      self.control.publish(command, lateral_log)
    return self.command, self.motion

  @property
  def command(self):
    return self.control.pm.messages['carControl'].carControl

  @property
  def motion(self):
    return self.control.pm.messages['controlsState'].controlsState.motionState

  def hold(self):
    self.car_state.vEgo = 0.0
    self.car_state.standstill = True
    self.car_state.cruiseState.standstill = True
    self.plan.aTarget = -0.1
    self.plan.motionIntent.reasons = int(StopReason.MPC)
    command, motion = self.step()
    assert command.actuators.accel == -2.0
    assert motion.phase == 'hold'
    assert motion.resolvedStop
    return command, motion


def selfdrive_event_harness(control_state):
  """Execute full update_events with healthy bounded input; vehicle-event/ICBM collaborators have no independent events."""
  services = [
    'controlsState', 'deviceState', 'modelV2', 'lateralManeuverPlan', 'alertDebug', 'userBookmark', 'driverMonitoringState',
    'longitudinalPlanSP', 'carControl', 'peripheralState', 'extrinsicsCalibration', 'driverAssistance', 'deviceMotion',
    'modelDataV2SP', 'pandaStates', 'managerState', 'radarState', 'vehicleParameters', 'longitudinalPlan',
    'gpsLocationExternal', 'narrowRoadCameraState', 'wideRoadCameraState',
  ]
  events = {service: new_event(service) for service in services}
  events['controlsState'] = control_state.as_builder()
  events['deviceState'].deviceState.freeSpacePercent = 80.0
  events['extrinsicsCalibration'].extrinsicsCalibration.calStatus = log.ExtrinsicsCalibration.Status.calibrated
  events['deviceMotion'].deviceMotion.posenetOK = True
  events['deviceMotion'].deviceMotion.inputsOK = True
  events['vehicleParameters'].vehicleParameters.valid = True
  daemon = SelfdriveD.__new__(SelfdriveD)
  daemon.sm = bounded_submaster(services)
  daemon.sm.update_msgs(2.0, [serialized(event) for event in events.values()])
  daemon.sm.updated = dict.fromkeys(services, False)
  daemon.sm.recv_frame = dict.fromkeys(services, 0)
  daemon.CP = car.CarParams.new_message()
  daemon.CP.openpilotLongitudinalControl = True
  daemon.events = Events()
  daemon.events_sp = EventsSP()
  daemon.CS_prev = car.CarState.new_message()
  daemon.car_events = SimpleNamespace(update=lambda *_: Events())
  daemon.car_events_sp = SimpleNamespace(update=lambda *_: EventsSP())
  daemon.icbm = SimpleNamespace(run=Mock())
  daemon.params = Mock()
  daemon.rk = SimpleNamespace(lagging=False)
  daemon.mads = SimpleNamespace(enabled=False)
  daemon.camera_packets = ['narrowRoadCameraState', 'wideRoadCameraState']
  daemon.sensor_packets = []
  daemon.gps_location_service = 'gpsLocationExternal'
  daemon.cp_safety_configs = []
  daemon.ignored_processes = set()
  daemon.not_running = set()
  for name in ('enabled', 'initialized', 'cp_olc', 'is_metric', 'radar_signal'):
    setattr(daemon, name, True)
  for name in (
    'cp_not_car', 'cp_pcm_cruise', 'cp_passive', 'cp_alt_experience', 'big_model_loading', 'big_model_active',
    'big_model_failed', 'usbgpu_loading', 'dm_lockout_set', 'dm_uncertain_alerted', 'disengage_on_accelerator',
    'is_ldw_enabled', 'recalibrating_seen', 'excessive_actuation', 'radar_any_error', 'experimental_mode',
  ):
    setattr(daemon, name, False)
  for name in (
    'big_model_ready_t', 'mismatch_counter', 'cruise_mismatch_counter', 'last_functional_fan_frame',
    'last_steering_pressed_frame', 'distance_traveled',
  ):
    setattr(daemon, name, 0)
  for name in ('startup_event', 'usbgpu_active', 'calibrated_actuation', 'not_running_prev', 'logged_comm_issue'):
    setattr(daemon, name, None)
  daemon.not_running_frame = -1
  daemon.radar_any_error_frame = -1
  car_state = car.CarState.new_message()
  car_state.canValid = True
  car_state.vEgo = 10.0
  car_state.gearShifter = car.CarState.GearShifter.drive
  return daemon, car_state
