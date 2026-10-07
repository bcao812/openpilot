"""Synthetic CAN -> observer/fusion -> real DEC/MPC/owner -> Honda CAN.

Vehicle motion and camera hypotheses are prescribed identically in both arms;
there is no plant, camera inference or closed-loop performance claim. Actual
radard lead selection is used without its temporal probability filter. Only
publication clocks are injected; no DEC, planner, owner or CAN output is mocked.
Cached radar messages retain their source timestamps throughout a dropout.
"""
from dataclasses import replace
import json
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
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.honda_radar_observer import HondaRadarObserver
from openpilot.selfdrive.car.honda_radar_observer_config import ASSIST, SHADOW, SERVICE, ObserverSettings
from openpilot.selfdrive.car.longitudinal_authority import restrict_longitudinal_authority
from openpilot.selfdrive.car.tests.test_honda_radar_observer import can_message, scan_frames
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_departure_accel_from_plan
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION, MotionIntent, MotionResponseKind, StopReason
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_TIMING_VERSION, ReferenceTimingConsumer
from openpilot.selfdrive.controls.radard import get_lead
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_planner, make_state
from openpilot.selfdrive.modeld.constants import ModelConstants

START_NS = 10_000_000_000
FRAME_NS = 50_000_000
QUALIFY_FRAMES = 24
FAULT_FRAME = QUALIFY_FRAMES + 2
STALE_ASSERT_NS = 250_000_000
DEPARTURE_ACCEL = 0.6
RELEASE_JERK_LIMIT = 1.0
RELEASE_TAIL_MAX_NS = 2_000_000_000 + FRAME_NS
SENSOR_SERVICES = ('carState', 'carStateSP', 'radarState', 'modelV2', 'selfdriveState',
                   'vehicleParameters', 'gpsLocationExternal', 'gpsLocation')


def publish_at(planner, state, publication_ns):
  """Serialize both real planner publications at their prescribed clock."""
  messages = {}
  original = messaging.new_message

  def dated_message(service, *args, **kwargs):
    kwargs['logMonoTime'] = publication_ns
    return original(service, *args, **kwargs)

  def send(service, message):
    messages[service] = messaging.log_from_bytes(message.to_bytes())

  with patch.object(messaging, 'new_message', new=dated_message):
    planner.publish(state, SimpleNamespace(send=send))
  return messages


def run_pipeline(mode, scenario='gap', *, record_property=None):
  rows = []
  try:
    return _run_pipeline(mode, scenario, rows)
  finally:
    if record_property is not None:
      evidence = dict(mode=mode, scenario=scenario, source_motion='prescribed, no plant',
                      model_times=[float(t) for t in ModelConstants.T_IDXS], frames=rows)
      record_property(f'pipeline_{scenario}_mode_{mode}',
                      json.dumps(evidence, default=lambda value: value.hex(), allow_nan=False, separators=(',', ':')))


def _run_pipeline(mode, scenario, rows):
  """20 Hz synthetic sensing/planning, 100 Hz owner, real 50 Hz Honda ACC."""
  stationary = scenario == 'standstill'
  speed = 0.0 if stationary else 10.0
  frames = QUALIFY_FRAMES + {'gap': 6, 'model_stop': 18, 'dropout': 52, 'standstill': 24}[scenario]
  with OpenpilotPrefix():
    params = Params()
    params.put('HondaBoschARadarMode', mode, block=True)
    for key in ('AlphaLongitudinalEnabled', 'ExperimentalMode', 'DynamicExperimentalControl'):
      params.put_bool(key, True, block=True)
    planner, extension = make_planner(speed=speed)
    owner = LongControl(planner.CP, extension)
    reference_consumer = ReferenceTimingConsumer()
    assert planner.unified and owner.unified
    observer = HondaRadarObserver(planner.CP, extension, session_id='prescribed-can-pipeline')
    settings = ObserverSettings(mode, True, True, True)
    dbc = DBC[planner.CP.carFingerprint][Bus.pt]
    car_controller = CarController({Bus.pt: dbc}, planner.CP, extension)
    can_bus = CanBus(planner.CP)
    parser = CANParser(dbc, [('ACC_CONTROL', 50)], can_bus.pt)
    control_sp = structs.CarControlSP()
    wrapper = SimpleNamespace(v_cruise_factor=1.0, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={})
    state = make_state(speed=speed, target=22.0, experimental=True)
    model_template = state['modelV2'].as_reader()
    state['carState'].canValid = True
    state['carState'].cruiseState.available = True
    state['carState'].cruiseState.enabled = True
    # Honda CarState only sets the PCM hold bit in stock-longitudinal mode.
    # Physical standstill is already prescribed separately by make_state().
    # Setting both would inject a stock-ACC hold and prevent native release.
    state['carState'].cruiseState.standstill = False
    state[SERVICE] = messaging.new_message(SERVICE).hondaRadarObservations.as_reader()
    state.logMonoTime[SERVICE] = 0
    state.valid[SERVICE] = False
    state.seen = dict.fromkeys(state, True)
    state.seen[SERVICE] = False
    # Seed only the initial prescribed engaged state. Later control/plan stamps
    # come from their actual producer calls below, not a blanket advance().
    for name in ('carControl', 'controlsState'):
      state.logMonoTime[name] = START_NS

    for frame in range(frames):
      source_ns = START_NS + frame * FRAME_NS
      gap = frame >= QUALIFY_FRAMES and not stationary
      stop = scenario == 'model_stop' and frame >= FAULT_FRAME
      dropout = scenario == 'dropout' and frame >= FAULT_FRAME
      departing = stationary and frame >= QUALIFY_FRAMES
      model_event = messaging.new_message('modelV2', logMonoTime=source_ns)
      model_event.modelV2 = model_template
      model = model_event.modelV2
      model.timestampEof = source_ns
      model.action.desiredAcceleration = -2.0 if stop else DEPARTURE_ACCEL if departing else 0.0 if stationary else 1.1
      model.action.shouldStop = stop
      # Same existing raw-CAN fixture: about 54.12 m ahead / 2.6447 m left.
      # At standstill the separate close visual lead owns stopping/departure;
      # the distant radar object must not acquire that authority.
      visual_x = (11.52 if departing else 5.52) if stationary else 55.64
      visual_y = 0.0 if stationary else -2.644682
      visual_speed = (2.0 if departing else 0.0) if stationary else speed
      model.leadsV3 = [dict(prob=0.2 if gap else 0.99, x=[visual_x] * 6, y=[visual_y] * 6,
                           v=[visual_speed] * 6, a=[0.0] * 6, xStd=[1.0] * 6, yStd=[0.4] * 6, vStd=[1.0] * 6)] * 3
      # A single departure trajectory starts at measured rest: x=.5*a*t^2,
      # v=a*t, and both the acceleration vector and action request are a.
      # Measured motion remains prescribed identically in the two arms.
      model.velocity.x = [float(DEPARTURE_ACCEL * t if departing else speed) for t in ModelConstants.T_IDXS]
      model.acceleration.x = [DEPARTURE_ACCEL if departing else 0.0] * len(ModelConstants.T_IDXS)
      model.position.x = [0.0 if stop else float(0.5 * DEPARTURE_ACCEL * t * t if departing else speed * t)
                          for t in ModelConstants.T_IDXS]
      model.position.y = [0.0 if stationary else -x * 2.644682 / 55.64 for x in model.position.x]
      state['modelV2'] = messaging.log_from_bytes(model_event.to_bytes()).modelV2
      model = state['modelV2']
      # Exercise the real visual-to-formal-lead admission, without installing
      # any shadow distance/velocity as an official lead.
      for index, name in enumerate(('leadOne', 'leadTwo')):
        lead = model.leadsV3[index]
        setattr(state['radarState'], name, get_lead(speed, True, {}, lead, model.velocity.x[0], lead.prob,
                                                   planner.CP, extension))
      formal_before = state['radarState'].to_dict()
      for name in SENSOR_SERVICES:
        state.logMonoTime[name] = source_ns
        state.valid[name] = state.seen[name] = state.updated[name] = True
      state.updated[SERVICE] = False

      if 'longitudinalPlanSP' in state and not dropout:
        raw_can = can_message(source_ns, scan_frames(index=frame % 16, life=1 + 2 * frame,
                                                     velocity=863, bus=observer.decoder.bus))
        can_wire = messaging.log_from_bytes(raw_can.as_builder().to_bytes())
        produced = observer.update([can_wire], state, settings, source_ns + 100_000, state)
        assert produced is not None
        wire = messaging.log_from_bytes(produced.to_bytes())
        state[SERVICE] = wire.hondaRadarObservations
        state.logMonoTime[SERVICE] = wire.logMonoTime
        state.valid[SERVICE] = bool(wire.valid)
        state.seen[SERVICE] = state.updated[SERVICE] = True
        assert state[SERVICE].active and state[SERVICE].fusion.active
        assert not (state[SERVICE].calibrated or state[SERVICE].fusion.calibrated or state[SERVICE].fusion.controlAllowed)

      # Observer source/publication, planner entry/publication and owner entry
      # are distinct. In particular, dropout never modifies cached radar time.
      planner_ns, publication_ns = source_ns + 200_000, source_ns + 1_000_000
      planner.update(state, now_ns=planner_ns)
      messages = publish_at(planner, state, publication_ns)
      plan_event = messages['longitudinalPlan']
      plan = plan_event.longitudinalPlan
      sp_event = messages['longitudinalPlanSP']
      state['longitudinalPlanSP'] = sp_event.longitudinalPlanSP
      state.logMonoTime['longitudinalPlanSP'] = sp_event.logMonoTime
      state.valid['longitudinalPlanSP'] = bool(sp_event.valid)
      state.seen['longitudinalPlanSP'] = state.updated['longitudinalPlanSP'] = True
      assert plan_event.valid and plan.solverStatus == 0 and plan.motionIntent.valid
      assert plan.modelMonoTime == source_ns and plan_event.logMonoTime == publication_ns
      assert planner.dec.enabled() and planner.dec.active()
      assert state['radarState'].to_dict() == formal_before
      guard = planner.reference_planner.radar_accel_guard
      body = state[SERVICE]
      row = dict(frame=frame, source_ns=source_ns, planner_ns=planner_ns, plan_ns=publication_ns,
                 radar_source=(state.logMonoTime[SERVICE], int(body.sequence), int(body.lastPositionMonoTime),
                               int(body.fusion.modelTimestampEof)),
                 qualified=guard.qualified, holding=guard.holding, guard_reason=guard.reason, ceiling=guard.accel_ceiling,
                 target=float(plan.aTarget), stop=bool(plan.shouldStop), reasons=int(plan.motionIntent.reasons),
                 release=bool(plan.motionIntent.releaseAllowed), moving=bool(plan.motionIntent.movingLead),
                 response=int(plan.motionIntent.responseKind), dec_mode=planner.dec.mode(),
                 reference_kind=int(plan.motionIntent.referenceKind), constraint=float(plan.reference.constraintAccel),
                 formal_leads=tuple(formal_before[name]['present'] for name in ('leadOne', 'leadTwo')),
                 measured_speed=float(state['carState'].vEgo), measured_accel=float(state['carState'].aEgo),
                 physical_standstill=bool(state['carState'].standstill), pcm_standstill=bool(state['carState'].cruiseState.standstill),
                 model=dict(eof_ns=int(model.timestampEof), action_accel=float(model.action.desiredAcceleration),
                            action_stop=bool(model.action.shouldStop), x=list(model.position.x), v=list(model.velocity.x),
                            a=list(model.acceleration.x)), owner=[], can=[], complete=False)
      rows.append(row)
      speeds, accels = tuple(plan.speeds), tuple(plan.accels)
      action_t = planner.CP.longitudinalActuatorDelay + DT_MDL
      intent = replace(MotionIntent.from_message(plan.motionIntent), release_accel=get_departure_accel_from_plan(
        speeds, accels, ModelConstants.T_IDXS[:CONTROL_N], action_t, plan.reference.constraintAccel))
      command_identity = (plan.modelMonoTime, plan.aTarget, plan.shouldStop, plan.solverStatus, intent,
                          float(plan.reference.accel), float(plan.reference.constraintAccel), bool(plan.reference.overridden),
                          speeds, accels, action_t)
      for tick in range(5):
        owner_ns = source_ns + 2_000_000 + tick * 10_000_000
        timing = reference_consumer.consume(plan.reference, publication_ns, owner_ns, command_identity=command_identity)
        assert timing.valid and timing.timing.instance_ns == intent.producer_instance_ns
        previous_state, previous_phase = str(owner.long_control_state), owner.motion_phase.text
        accel = owner.update(True, state['carState'], plan.aTarget, plan.shouldStop,
                             (CarControllerParams.BOSCH_ACCEL_MIN, CarControllerParams.BOSCH_ACCEL_MAX),
                             intent=intent, plan_time_ns=plan_event.logMonoTime, model_time_ns=plan.modelMonoTime,
                             now_ns=owner_ns, plan_valid=bool(plan_event.valid and timing.valid), solver_status=plan.solverStatus,
                             braking_plan_valid=bool(state['carState'].canValid and state.valid['carState']
                                                     and 0 <= owner_ns - state.logMonoTime['carState'] <= 150_000_000))
        row['owner'].append(dict(now_ns=owner_ns, plan_ns=plan_event.logMonoTime, model_ns=plan.modelMonoTime,
                                 reference_source_ns=timing.timing.source_ns, reference_producer_ns=timing.timing.producer_ns,
                                 timing_valid=timing.valid, target=float(plan.aTarget), output_accel=float(accel),
                                 previous_state=previous_state, previous_phase=previous_phase,
                                 state=str(owner.long_control_state), phase=owner.motion_phase.text,
                                 resolved_stop=bool(owner.resolved_should_stop), reasons=int(owner.motion_reasons),
                                 input_constrained=bool(owner.motion_input_constrained), release_allowed=intent.release_allowed,
                                 moving_lead=intent.moving_lead, release_accel=intent.release_accel,
                                 release_confirmation=owner.motion_supervisor.release_lead_t))
        if frame >= QUALIFY_FRAMES:
          assert not owner.motion_input_constrained
        command = messaging.new_message('carControl', logMonoTime=owner_ns)
        command.carControl.enabled = command.carControl.longActive = True
        command.carControl.actuators.accel = float(accel)
        command.carControl.actuators.longControlState = owner.long_control_state
        command.carControl.cruiseControl.resume = bool(state['carState'].cruiseState.standstill
                                                      and not owner.resolved_should_stop and not owner.motion_input_constrained)
        cached_control = messaging.log_from_bytes(command.to_bytes()).carControl
        state['carControl'] = cached_control
        state.logMonoTime['carControl'] = owner_ns
        feedback = state['controlsState']
        feedback.longControlState = owner.long_control_state
        feedback.longitudinalPlanMonoTime = publication_ns
        feedback.motionState.version = MOTION_INTENT_VERSION
        feedback.motionState.enabled = owner.unified
        feedback.motionState.planningMode = int(owner.longitudinal_mode)
        feedback.motionState.phase = owner.motion_phase.text
        feedback.motionState.reasons = int(owner.motion_reasons)
        feedback.motionState.resolvedStop = bool(owner.resolved_should_stop)
        feedback.motionState.degraded = bool(owner.motion_degraded)
        feedback.motionState.waitingForFreshInput = bool(owner.motion_waiting_for_fresh_input)
        feedback.motionState.planMonoTime = publication_ns
        feedback.motionState.modelMonoTime = plan.modelMonoTime
        feedback.motionState.referenceTimingVersion = REFERENCE_TIMING_VERSION
        feedback.motionState.referenceTimingValid = timing.valid
        feedback.motionState.referenceTimingReason = timing.reason
        feedback.motionState.referenceConsumerMonoTime = timing.now_ns
        feedback.motionState.referenceSourceAdvanced = timing.source_advanced
        feedback.motionState.referenceSourceMonoTime = timing.timing.source_ns
        feedback.motionState.referenceEpochMonoTime = timing.timing.epoch_ns
        feedback.motionState.referenceProducerMonoTime = timing.timing.producer_ns
        feedback.motionState.referenceProducerInstanceMonoTime = timing.timing.instance_ns
        state.logMonoTime['controlsState'] = owner_ns
        wrapper.out = state['carState'].as_reader()
        effective = restrict_longitudinal_authority(planner.CP, extension, wrapper.out, cached_control)
        _, sends = car_controller.update(effective, control_sp, wrapper, owner_ns + 1_000_000)
        acc_frames = [bytes(data) for address, data, bus in sends if address == 0x1df and bus == can_bus.pt]
        if acc_frames:
          assert len(acc_frames) == 1
          parser.update([(owner_ns + 1_000_000, sends)])
          values = parser.vl['ACC_CONTROL']
          row['can'].append(dict(owner_ns=owner_ns, can_ns=owner_ns + 1_000_000, owner_accel=float(accel), payload=acc_frames[0],
                                 accel=float(values['ACCEL_COMMAND']), gas=int(values['GAS_COMMAND']),
                                 brake=bool(values['BRAKE_REQUEST']), stop=bool(values['STANDSTILL']),
                                 release=bool(values['STANDSTILL_RELEASE']), control_on=int(values['CONTROL_ON']),
                                 resume=bool(effective.cruiseControl.resume), owner_state=str(effective.actuators.longControlState),
                                 stopping_counter=car_controller.stopping_counter,
                                 authority_restricted=effective is not cached_control))
      assert len(row['can']) in (2, 3)
      row['complete'] = True
  return rows


def can_contract(rows):
  return [(row['stop'], row['reasons'], row['release'], row['moving'],
           [(sample['payload'], sample['resume']) for sample in row['can']]) for row in rows]


def test_confirmed_can_target_changes_actual_honda_requests_only_in_assist_mode(record_property):
  shadow = run_pipeline(SHADOW, record_property=record_property)
  assist = run_pipeline(ASSIST, record_property=record_property)
  assert can_contract(shadow[:QUALIFY_FRAMES]) == can_contract(assist[:QUALIFY_FRAMES])
  assert assist[QUALIFY_FRAMES - 1]['qualified']
  assert not any(row['qualified'] or row['holding'] for row in shadow)
  for baseline, candidate in zip(shadow[QUALIFY_FRAMES:], assist[QUALIFY_FRAMES:], strict=True):
    assert candidate['holding'] and candidate['qualified'] and candidate['ceiling'] == 0.0
    assert not any(candidate['formal_leads'])
    assert candidate['response'] == MotionResponseKind.ABSOLUTE_ACC
    assert candidate['target'] <= 1e-7 and not candidate['stop']
    assert candidate['reasons'] == baseline['reasons'] == 0
    # A zero cap may retire existing release evidence; it cannot grant any
    # release/moving authority absent from the same-input shadow-only arm.
    assert not candidate['release'] or baseline['release']
    assert not candidate['moving'] or baseline['moving']
    assert max(sample['owner_accel'] for sample in candidate['can']) <= 1e-7
    assert max(sample['accel'] for sample in candidate['can']) <= 0.0
    assert all(sample['control_on'] == 5 and not sample['stop'] for sample in candidate['can'])
  assert shadow[-1]['target'] > 0.1
  assert shadow[-1]['can'][-1]['accel'] > assist[-1]['can'][-1]['accel']
  assert shadow[-1]['can'][-1]['gas'] > assist[-1]['can'][-1]['gas']
  assert shadow[-1]['can'][-1]['payload'] != assist[-1]['can'][-1]['payload']


def test_strong_model_stop_keeps_braking_and_stop_authority_through_can(record_property):
  shadow = run_pipeline(SHADOW, 'model_stop', record_property=record_property)
  assist = run_pipeline(ASSIST, 'model_stop', record_property=record_property)
  assert all(row['holding'] for row in assist[QUALIFY_FRAMES:FAULT_FRAME])
  # Real DEC must enter blended mode; no test override grants model authority.
  assert any(row['dec_mode'] == 'blended' for row in assist[FAULT_FRAME:])
  for baseline, candidate in zip(shadow[-5:], assist[-5:], strict=True):
    assert baseline['stop'] and candidate['stop']
    assert candidate['reasons'] & StopReason.E2E
    assert not candidate['release'] and not baseline['release']
    assert candidate['target'] <= -1.5
    for normal, limited in zip(baseline['can'], candidate['can'], strict=True):
      assert limited['accel'] <= normal['accel'] + 0.01
      assert limited['brake'] and limited['gas'] == -30000
      assert not limited['resume']


def test_observer_dropout_cannot_refresh_source_time_or_keep_radar_authority(record_property):
  shadow = run_pipeline(SHADOW, 'dropout', record_property=record_property)
  rows = run_pipeline(ASSIST, 'dropout', record_property=record_property)
  assert all(row['holding'] for row in rows[QUALIFY_FRAMES:FAULT_FRAME])
  frozen = rows[FAULT_FRAME - 1]['radar_source']
  assert frozen[0] > 0 and frozen[2] > 0 and frozen[3] > 0
  assert all(row['radar_source'] == frozen for row in rows[FAULT_FRAME:])
  assert all(row['radar_source'] == frozen for row in shadow[FAULT_FRAME:])
  stale = [row for row in rows if row['planner_ns'] - frozen[0] > STALE_ASSERT_NS]
  assert stale and all(not row['qualified'] and not row['holding'] for row in stale)
  for candidate in stale:
    baseline = shadow[candidate['frame']]
    assert candidate['stop'] == baseline['stop'] == False
    assert candidate['reasons'] == baseline['reasons'] == 0
    assert not candidate['release'] or baseline['release']
    assert not candidate['moving'] or baseline['moving']
  revoked = next(row for row in rows[FAULT_FRAME:] if not row['qualified'] and not row['holding'])
  recovery = rows[revoked['frame']:]
  cleared = next(row for row in recovery if row['ceiling'] is None)
  assert cleared['planner_ns'] - revoked['planner_ns'] <= RELEASE_TAIL_MAX_NS
  for previous, current in zip(recovery, recovery[1:]):
    if current['ceiling'] is not None:
      assert previous['ceiling'] is not None and 0.0 <= current['ceiling'] <= 2.0
      delta_t = (current['planner_ns'] - previous['planner_ns']) / 1e9
      assert 0.0 <= current['ceiling'] - previous['ceiling'] <= RELEASE_JERK_LIMIT * delta_t + 1e-7
    elif previous['ceiling'] is not None:
      assert 2.0 - previous['ceiling'] <= RELEASE_JERK_LIMIT * FRAME_NS / 1e9 + 1e-7
  assert rows[-1]['ceiling'] is None and rows[-1]['can'][-1]['accel'] > 0.0
  assert rows[-1]['release'] == shadow[-1]['release'] and rows[-1]['moving'] == shadow[-1]['moving']
  assert shadow[-1]['can'][-1]['accel'] > 0.0


def test_standstill_hold_and_native_departure_can_are_unchanged(record_property):
  shadow = run_pipeline(SHADOW, 'standstill', record_property=record_property)
  assist = run_pipeline(ASSIST, 'standstill', record_property=record_property)
  assert can_contract(shadow) == can_contract(assist)
  assert all(not row['qualified'] and not row['holding'] and row['ceiling'] is None for row in assist)
  assert all(row['stop'] and not row['release'] for row in assist[:QUALIFY_FRAMES])
  assert any(sample['stop'] and sample['brake'] and not sample['release']
             for row in assist[:QUALIFY_FRAMES] for sample in row['can'])
  assert any(row['release'] for row in assist[QUALIFY_FRAMES:])
  for row in assist[QUALIFY_FRAMES:]:
    assert row['physical_standstill'] and not row['pcm_standstill']
    assert row['model']['x'][0] == 0.0 and row['model']['v'][0] == row['measured_speed'] == 0.0
    assert abs(row['model']['action_accel'] - DEPARTURE_ACCEL) < 1e-6
    for t, x, v, a in zip(ModelConstants.T_IDXS, row['model']['x'], row['model']['v'], row['model']['a'], strict=True):
      assert abs(x - 0.5 * DEPARTURE_ACCEL * t * t) < 1e-5
      assert abs(v - DEPARTURE_ACCEL * t) < 1e-5 and abs(a - DEPARTURE_ACCEL) < 1e-6
  released = [sample for row in assist[QUALIFY_FRAMES:] for sample in row['can'] if sample['release'] and not sample['stop']]
  assert released and all(sample['stopping_counter'] == 0 and sample['owner_state'] == 'pid' for sample in released)
  assert any(owner['phase'] in ('release', 'track') and not owner['resolved_stop']
             for row in assist[QUALIFY_FRAMES:] for owner in row['owner'])
