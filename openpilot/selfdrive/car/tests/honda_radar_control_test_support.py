"""Synthetic sensing through the real Accord radar/planner/owner/CAN chain.

Only sensor observations and publication clocks are supplied by the fixture.
No formal lead, DEC decision, planner policy or actuator request is patched.
The optional ideal plant consumes decoded Honda acceleration requests; it is
an internally consistent regression model, not a calibrated vehicle model.
"""
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
import math
from types import SimpleNamespace
from unittest.mock import patch

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.hondacan import CanBus
from opendbc.car.honda.values import DBC, CarControllerParams
from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.car.honda_radar_observer import HondaRadarObserver
from openpilot.selfdrive.car.honda_radar_observer_config import SERVICE, ObserverSettings
from openpilot.selfdrive.car.longitudinal_authority import restrict_longitudinal_authority
from openpilot.selfdrive.car.tests.test_honda_radar_observer import can_message, scan_frames
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_departure_accel_from_plan
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION, MotionIntent
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_TIMING_VERSION, ReferenceTimingConsumer
from openpilot.selfdrive.controls.radard import RadarD
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_planner, make_state
from openpilot.selfdrive.modeld.constants import ModelConstants

START_NS = 10_000_000_000
FRAME_NS = 50_000_000


@dataclass(frozen=True)
class Target:
  distance: float
  speed: float
  lateral: float = 0.0  # Radar/body coordinates, left positive.
  slot: int = 0
  oem_id: int = 1


def raw_scan(target, ego_speed, frame, bus):
  """Invert the documented native raw scales; preserve real quantization."""
  distance = round((target.distance + 3.0) / .05712)
  azimuth = round(1024 + 2048 * math.atan2(target.lateral, target.distance))
  velocity = round(864 + 64 * (target.speed - ego_speed))
  assert 0 < distance < 4095 and 0 <= azimuth < 2047 and 0 < velocity < 1728
  return scan_frames(slot=target.slot, oem_id=target.oem_id, index=frame % 16,
                     life=min(4094, 1 + 2 * frame), distance=distance, azimuth=azimuth,
                     velocity=velocity, bus=bus)


def _publish_at(produce, timestamp):
  result = {}
  original = messaging.new_message

  def dated(service, *args, **kwargs):
    kwargs['logMonoTime'] = timestamp
    return original(service, *args, **kwargs)

  def send(service, message):
    result[service] = messaging.log_from_bytes(message.to_bytes())

  with patch.object(messaging, 'new_message', new=dated):
    produce(SimpleNamespace(send=send))
  return result


class IdealPlant:
  def __init__(self, speed, *, delay=.2, response_time=.25):
    self.speed, self.accel, self.position = speed, 0.0, 0.0
    self.response_time = response_time
    self.delayed = deque([0.0] * max(1, round(delay / DT_CTRL)))

  def step(self, command):
    self.delayed.append(command)
    delayed = self.delayed.popleft()
    acceleration = self.accel + min(1.0, DT_CTRL / self.response_time) * (delayed - self.accel)
    previous = self.speed
    self.speed = max(0.0, previous + acceleration * DT_CTRL)
    self.accel = (self.speed - previous) / DT_CTRL
    self.position += .5 * (previous + self.speed) * DT_CTRL


@contextmanager
def control_pipeline(*, speed=10.0, cruise=18.0):
  with OpenpilotPrefix():
    params = Params()
    for name in ('AlphaLongitudinalEnabled', 'ExperimentalMode', 'DynamicExperimentalControl'):
      params.put_bool(name, True, block=True)
    yield ControlPipeline(params, speed, cruise)


class ControlPipeline:
  def __init__(self, params, speed, cruise):
    self.params = params
    self.planner, self.extension = make_planner(speed=speed)
    self.owner = LongControl(self.planner.CP, self.extension)
    assert self.planner.unified and self.owner.unified
    self.reference_consumer = ReferenceTimingConsumer()
    self.observer = HondaRadarObserver(self.planner.CP, self.extension, session_id='active-radar-pipeline')
    self.radard = RadarD(self.planner.CP, self.extension, params=params)
    self.settings = ObserverSettings(True, True, True)
    dbc = DBC[self.planner.CP.carFingerprint][Bus.pt]
    self.controller = CarController({Bus.pt: dbc}, self.planner.CP, self.extension)
    self.can_bus = CanBus(self.planner.CP)
    self.parser = CANParser(dbc, [('ACC_CONTROL', 50)], self.can_bus.pt)
    self.control_sp = structs.CarControlSP()
    self.wrapper = SimpleNamespace(v_cruise_factor=1.0, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={})
    self.state = state = make_state(speed=speed, target=cruise, experimental=True)
    self.model_template = state['modelV2'].as_reader()
    state['carState'].canValid = True
    state['carState'].cruiseState.available = True
    state['carState'].cruiseState.enabled = True
    state['carState'].cruiseState.standstill = False
    # A disabled, not-yet-seen initial plan exercises the real bootstrap: the
    # observer must collect before DEC publishes its first enabled state.
    for service in (SERVICE, 'longitudinalPlanSP', 'radarTracks'):
      state[service] = getattr(messaging.new_message(service), service)
      state.logMonoTime[service] = 0
      state.valid[service] = state.updated[service] = False
      state.recv_frame[service] = 0
    state.seen = {name: state.logMonoTime[name] > 0 for name in state}
    for service in ('carControl', 'controlsState'):
      state.logMonoTime[service] = START_NS
    state['carControl'] = state['carControl'].as_reader()
    self.frame = 0
    self.cruise = cruise
    self.last_can_accel = 0.0

  def _accept(self, service, event):
    self.state[service] = getattr(event, service)
    self.state.logMonoTime[service] = event.logMonoTime
    self.state.valid[service] = bool(event.valid)
    self.state.seen[service] = self.state.updated[service] = True
    self.state.recv_frame[service] = self.frame + 1

  def _model(self, timestamp, speed, visual, acceleration, should_stop):
    event = messaging.new_message('modelV2', logMonoTime=timestamp, valid=True)
    event.modelV2 = self.model_template
    model = event.modelV2
    model.timestampEof = timestamp
    model.orientationRate.z = [0.0] * len(ModelConstants.T_IDXS)
    stop_time = -speed / acceleration if acceleration < 0 else math.inf
    accelerate_time = max(0.0, (self.cruise - speed) / acceleration) if acceleration > 0 else math.inf
    ramp_time = min(stop_time, accelerate_time)
    model.position.x = [float(speed * min(t, ramp_time) + .5 * acceleration * min(t, ramp_time)**2
                              + max(0.0, speed + acceleration * ramp_time) * max(0.0, t - ramp_time))
                        if math.isfinite(ramp_time) else float(speed * t + .5 * acceleration * t * t)
                        for t in ModelConstants.T_IDXS]
    model.position.y = [0.0] * len(ModelConstants.T_IDXS)
    model.velocity.x = [float(max(0.0, speed + acceleration * min(t, ramp_time))) for t in ModelConstants.T_IDXS]
    model.acceleration.x = [float(acceleration if t < ramp_time else 0.0) for t in ModelConstants.T_IDXS]
    model.action.desiredAcceleration = acceleration
    model.action.shouldStop = should_stop
    # Spatial lane samples retain support when a legitimate stop trajectory is
    # too short to describe geometry beyond the stopped ego vehicle.
    model.laneLines = [{'x': [float(i * 2) for i in range(33)], 'y': [y] * 33} for y in (-5.4, -1.8, 1.8, 5.4)]
    model.laneLineProbs = [0.9] * 4
    model.laneLineStds = [0.15] * 4
    leads = [{'prob': .99, 'x': [target.distance + 1.52] * 6, 'y': [-target.lateral] * 6,
                  'v': [target.speed] * 6, 'a': [0.0] * 6, 'xStd': [1.0] * 6, 'yStd': [.4] * 6, 'vStd': [1.0] * 6}
             for target in visual]
    absent = {'prob': 0.0, 'x': [100.0] * 6, 'y': [0.0] * 6, 'v': [speed] * 6, 'a': [0.0] * 6,
                  'xStd': [1.0] * 6, 'yStd': [.4] * 6, 'vStd': [1.0] * 6}
    model.leadsV3 = (leads + [absent] * 3)[:3]
    return event

  def _feedback(self, timing, publication_ns, model_ns, owner_ns):
    feedback = self.state['controlsState']
    feedback.longControlState = self.owner.long_control_state
    feedback.longitudinalPlanMonoTime = publication_ns
    motion = feedback.motionState
    motion.version, motion.enabled = MOTION_INTENT_VERSION, self.owner.unified
    motion.planningMode = int(self.owner.longitudinal_mode)
    motion.phase, motion.reasons = self.owner.motion_phase.text, int(self.owner.motion_reasons)
    motion.resolvedStop, motion.degraded = self.owner.resolved_should_stop, self.owner.motion_degraded
    motion.waitingForFreshInput = self.owner.motion_waiting_for_fresh_input
    motion.planMonoTime, motion.modelMonoTime = publication_ns, model_ns
    motion.referenceTimingVersion, motion.referenceTimingValid = REFERENCE_TIMING_VERSION, timing.valid
    motion.referenceTimingReason, motion.referenceConsumerMonoTime = timing.reason, timing.now_ns
    motion.referenceSourceAdvanced = timing.source_advanced
    motion.referenceSourceMonoTime = timing.timing.source_ns
    motion.referenceEpochMonoTime = timing.timing.epoch_ns
    motion.referenceProducerMonoTime = timing.timing.producer_ns
    motion.referenceProducerInstanceMonoTime = timing.timing.instance_ns
    self.state.logMonoTime['controlsState'] = owner_ns

  def step(self, *, speed, radar=(), visual=(), accel=0.0, model_accel=.6, model_stop=False,
           observer_online=True, scope_online=True, engaged=True, pedal=None, plant=None):
    state, frame = self.state, self.frame
    source_ns = START_NS + frame * FRAME_NS
    car = state['carState']
    car.vEgo, car.aEgo, car.standstill = speed, accel, speed <= .001
    car.gasPressed, car.brakePressed = pedal == 'gas', pedal == 'brake'
    state['selfdriveState'].enabled = engaged
    # Mirror controlsd's UNIFIED pedal gate in both the sensing input and the
    # actual owner/CAN feedback. Engagement alone is not longActive on a pedal.
    long_active = bool(engaged and not car.gasPressed and not car.brakePressed)
    # carControl is a reader after its first wire publication.
    control = state['carControl'].as_builder()
    control.enabled, control.longActive = engaged, long_active
    control.cruiseControl.override = bool(engaged and not long_active)
    state['carControl'] = control.as_reader()
    state.logMonoTime['carControl'] = source_ns
    for service in ('carState', 'carStateSP', 'selfdriveState', 'vehicleParameters', 'gpsLocationExternal', 'gpsLocation', 'radarTracks'):
      if service == 'selfdriveState' and not scope_online:
        state.updated[service] = False
        continue
      state.logMonoTime[service] = source_ns
      state.valid[service] = state.seen[service] = state.updated[service] = True
      state.recv_frame[service] = frame + 1
    self._accept('modelV2', self._model(source_ns, speed, visual, model_accel, model_stop))
    state.updated[SERVICE] = False
    if observer_online:
      frames = [message for target in radar for message in raw_scan(target, speed, frame, self.observer.decoder.bus)]
      event = self.observer.update([can_message(source_ns, frames)], state, self.settings, source_ns + 100_000, state)
      assert event is not None
      self._accept(SERVICE, messaging.log_from_bytes(event.to_bytes()))
    self.radard.update(state, state['radarTracks'], now_ns=source_ns + 200_000)
    self._accept('radarState', _publish_at(self.radard.publish, source_ns + 300_000)['radarState'])
    self.planner.update(state, now_ns=source_ns + 500_000)
    publication_ns = source_ns + 1_000_000
    publications = _publish_at(lambda pm: self.planner.publish(state, pm), publication_ns)
    self._accept('longitudinalPlanSP', publications['longitudinalPlanSP'])
    plan_event = publications['longitudinalPlan']
    plan = plan_event.longitudinalPlan
    assert plan_event.valid and plan.motionIntent.valid
    speeds, accels = tuple(plan.speeds), tuple(plan.accels)
    action_t = self.planner.CP.longitudinalActuatorDelay + DT_MDL
    intent = replace(MotionIntent.from_message(plan.motionIntent), release_accel=get_departure_accel_from_plan(
      speeds, accels, ModelConstants.T_IDXS[:CONTROL_N], action_t, plan.reference.constraintAccel))
    identity = (plan.modelMonoTime, plan.aTarget, plan.shouldStop, plan.solverStatus, intent,
                float(plan.reference.accel), float(plan.reference.constraintAccel), bool(plan.reference.overridden),
                speeds, accels, action_t)
    row = {'frame': frame, 'time': frame * DT_MDL, 'source_ns': source_ns, 'speed': float(speed), 'long_active': long_active,
               'target': float(plan.aTarget), 'stop': bool(plan.shouldStop), 'reasons': int(plan.motionIntent.reasons),
               'release': bool(plan.motionIntent.releaseAllowed), 'moving': bool(plan.motionIntent.movingLead),
               'solver': int(plan.solverStatus), 'dec_mode': self.planner.dec.mode(), 'radar': state['radarState'].to_dict(),
               'observer_ns': state.logMonoTime[SERVICE], 'observer_position_ns': int(state[SERVICE].lastPositionMonoTime),
               'sensor_targets': [{'distance': t.distance, 'speed': t.speed, 'lateral': t.lateral, 'oem_id': t.oem_id} for t in radar],
               'owner': [], 'can': []}
    observation = state[SERVICE]
    row['observer'] = {
      'active': bool(observation.active), 'reason': str(observation.reason), 'mode': int(observation.mode),
      'fusion_active': bool(observation.fusion.active), 'fusion_reason': str(observation.fusion.reason),
      'algorithm': str(observation.fusion.algorithmVersion), 'model_ns': int(observation.fusion.modelMonoTime),
      'model_eof_ns': int(observation.fusion.modelTimestampEof), 'ego_ns': int(observation.fusion.egoMonoTime),
      'tracks': len(observation.fusion.tracks), 'model_input_valid': state.valid['modelV2'],
    }
    for tick in range(5):
      owner_ns = source_ns + 2_000_000 + tick * 10_000_000
      if plant is not None:
        car.vEgo, car.aEgo, car.standstill = plant.speed, plant.accel, plant.speed <= .001
        state.logMonoTime['carState'] = owner_ns
      timing = self.reference_consumer.consume(plan.reference, publication_ns, owner_ns, command_identity=identity)
      assert timing.valid
      output = self.owner.update(
        long_active, car, plan.aTarget, plan.shouldStop,
        (CarControllerParams.BOSCH_ACCEL_MIN, CarControllerParams.BOSCH_ACCEL_MAX), intent=intent,
        plan_time_ns=publication_ns, model_time_ns=plan.modelMonoTime, now_ns=owner_ns,
        plan_valid=bool(plan_event.valid and timing.valid), solver_status=plan.solverStatus,
        braking_plan_valid=True)
      row['owner'].append({'accel': float(output), 'active': long_active, 'phase': self.owner.motion_phase.text,
                               'stop': bool(self.owner.resolved_should_stop), 'constrained': bool(self.owner.motion_input_constrained)})
      command = messaging.new_message('carControl', logMonoTime=owner_ns, valid=True)
      command.carControl.enabled, command.carControl.longActive = engaged, long_active
      command.carControl.cruiseControl.override = bool(engaged and not long_active)
      command.carControl.actuators.accel = float(output)
      command.carControl.actuators.longControlState = self.owner.long_control_state
      cached = messaging.log_from_bytes(command.to_bytes()).carControl
      state['carControl'], state.logMonoTime['carControl'] = cached, owner_ns
      self._feedback(timing, publication_ns, plan.modelMonoTime, owner_ns)
      self.wrapper.out = car.as_reader()
      effective = restrict_longitudinal_authority(self.planner.CP, self.extension, self.wrapper.out, cached)
      _, sends = self.controller.update(effective, self.control_sp, self.wrapper, owner_ns + 1_000_000)
      if any(address == 0x1df and bus == self.can_bus.pt for address, _, bus in sends):
        self.parser.update([(owner_ns + 1_000_000, sends)])
        values = self.parser.vl['ACC_CONTROL']
        self.last_can_accel = float(values['ACCEL_COMMAND']) if effective.longActive else 0.0
        row['can'].append({'accel': float(values['ACCEL_COMMAND']), 'gas': int(values['GAS_COMMAND']),
                               'brake': bool(values['BRAKE_REQUEST']), 'stop': bool(values['STANDSTILL']),
                               'release': bool(values['STANDSTILL_RELEASE']), 'control_on': int(values['CONTROL_ON']),
                               'active': bool(effective.longActive), 'resume': bool(effective.cruiseControl.resume)})
      if plant is not None:
        plant.step(self.last_can_accel)
    assert len(row['can']) in (2, 3)
    self.frame += 1
    return row
