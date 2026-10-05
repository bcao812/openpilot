"""Native schema, observer boundary and isolated process tests for shadow fusion."""
import os
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock, patch

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.car.honda_radar_fusion import HondaRadarFusion
from openpilot.selfdrive.car.honda_radar_observer import HondaRadarObserver
from openpilot.selfdrive.car.honda_radar_observer_config import ObserverSettings, SERVICE
from openpilot.selfdrive.car.tests.test_honda_radar_observer import START, SETTINGS, can_message, configuration, scan_frames, state

SHADOW_SETTINGS = ObserverSettings(2, True, True, True)
STEP = 66_000_000


def fusion_state(t):
  sm = state(t)
  cs = messaging.new_message('carState').carState
  cs.vEgo = 10.0
  model = messaging.new_message('modelV2').modelV2
  model.timestampEof = t
  model.position.x = [float(i * 5) for i in range(33)]
  model.position.y = [0.0] * 33
  model.velocity.x = [10.0] * 33
  model.orientationRate.z = [0.0] * 33
  # The CAN fixture is 54.12 m forward, 2.6447 m to the left. Give the
  # model a curved path through that target so it is a plausible match.
  model.position.y = [-i * 5 * 2.644682 / 55.64 for i in range(33)]
  lead = {'prob': .99, 'x': [55.64] * 6, 'y': [-2.644682] * 6, 'v': [10.0] * 6, 'a': [0.0] * 6,
              'xStd': [1.0] * 6, 'yStd': [.4] * 6, 'vStd': [1.0] * 6}
  model.leadsV3 = [lead] * 3
  sm.update(carState=cs, modelV2=model)
  sm.valid = dict.fromkeys(sm, True)
  sm.seen = dict.fromkeys(sm, True)
  sm.logMonoTime = dict.fromkeys(sm, t)
  return sm


class TestHondaRadarShadow(unittest.TestCase):
  def test_entrypoint_takes_validation_clock_after_receiving_new_model(self):
    from openpilot.selfdrive.car import honda_radar_observer as runtime
    cp, cp_sp = configuration()
    params = Mock()
    values = {'CarParams': cp.to_bytes(), 'CarParamsSP': cp_sp.to_bytes(), 'HondaBoschARadarMode': 2}
    params.get.side_effect = lambda key, **_: values[key]
    params.get_bool.return_value = True
    clock = [START]
    scope, inputs = state(START), fusion_state(START + 10_000_000)
    scope.update = lambda _: None
    inputs.update = lambda _: clock.__setitem__(0, START + 10_000_000)
    sock = Mock()
    sock.receive.side_effect = [can_message(START, scan_frames()).as_builder().to_bytes(), None]
    publisher = Mock()

    class StopFixture(Exception):
      pass

    publisher.send.side_effect = StopFixture
    with patch.object(runtime, 'configure_observer_process'), patch('openpilot.common.params.Params', return_value=params), \
         patch.object(runtime.messaging, 'SubMaster', side_effect=[scope, inputs]), \
         patch.object(runtime.messaging, 'PubMaster', return_value=publisher), \
         patch.object(runtime.messaging, 'sub_sock', return_value=sock), \
         patch.object(runtime.time, 'monotonic_ns', side_effect=lambda: clock[0]), self.assertRaises(StopFixture):
      runtime.main()
    message = publisher.send.call_args.args[1]
    self.assertEqual(message.logMonoTime, START + 10_000_000)
    self.assertTrue(message.hondaRadarObservations.fusion.active)

  def test_unmatched_visual_values_follow_existing_radard_conversion(self):
    from openpilot.selfdrive.controls.radard import get_RadarState_from_vision
    for x in (1.0, 1.52, 25.0):
      observer = HondaRadarObserver(*configuration())
      sm = fusion_state(START)
      sm['modelV2'].leadsV3[0].x = [x] * 6
      msg = observer.update([], sm, SHADOW_SETTINGS, START + 1, sm)
      candidate = msg.hondaRadarObservations.fusion.leads[0]
      original = get_RadarState_from_vision(sm['modelV2'].leadsV3[0], sm['carState'].vEgo,
                                           sm['modelV2'].velocity.x[0], sm['modelV2'].leadsV3[0].prob)
      for key in ('present', 'dRel', 'yRel', 'vRel', 'vLead'):
        self.assertEqual(getattr(candidate, key), original[key])
      self.assertEqual(candidate.aLead, original['aLeadK'])
      self.assertEqual(candidate.distanceWeight, 0)

  def test_shadow_wire_roundtrip_and_both_dec_modes(self):
    for dec in ('acc', 'blended'):
      observer = HondaRadarObserver(*configuration(), session_id='shadow-test')
      for frame in range(12):
        t = START + frame * STEP
        sm = fusion_state(t)
        sm['longitudinalPlanSP'].dec.state = dec
        before = {key: value.to_dict() for key, value in sm.items()}
        msg = observer.update([can_message(t, scan_frames(index=frame % 16, life=1 + 2 * frame, velocity=864))],
                              sm, SHADOW_SETTINGS, t + 1, sm)
        decoded = messaging.log_from_bytes(msg.to_bytes()).hondaRadarObservations
        self.assertEqual({key: value.to_dict() for key, value in sm.items()}, before)
        self.assertTrue(decoded.fusion.active)
        self.assertFalse(decoded.fusion.calibrated or decoded.fusion.controlAllowed)
        self.assertEqual(decoded.fusion.modelMonoTime, t)
        self.assertEqual(len(decoded.fusion.tracks), 1)
        self.assertEqual(len(decoded.fusion.leads), 3)
      self.assertEqual(decoded.fusion.counters.positions, 12)
      self.assertEqual(decoded.fusion.leads[0].state, 'fused')
      self.assertGreater(decoded.fusion.leads[0].distanceWeight, 0)

  def test_each_scope_flag_clears_candidates(self):
    for key in ('alpha', 'experimental', 'dec', 'experimentalActive', 'decEnabled', 'decActive'):
      observer = HondaRadarObserver(*configuration())
      sm = fusion_state(START)
      observer.update([can_message(START, scan_frames())], sm, SHADOW_SETTINGS, START + 1, sm)
      t = START + STEP
      sm = fusion_state(t)
      values = {'mode': 2, 'alpha': True, 'experimental': True, 'dec': True}
      if key in values:
        values[key] = False
      elif key == 'experimentalActive':
        sm['selfdriveState'].experimentalMode = False
      else:
        setattr(sm['longitudinalPlanSP'].dec, 'enabled' if key == 'decEnabled' else 'active', False)
      msg = observer.update([], sm, ObserverSettings(**values), t + 1, sm)
      self.assertFalse(msg.hondaRadarObservations.fusion.active, key)
      self.assertFalse(msg.hondaRadarObservations.fusion.tracks, key)
      self.assertFalse(msg.hondaRadarObservations.fusion.leads, key)

  def test_observe_payload_and_disable_reenable_do_not_reuse_confirmation(self):
    observer = HondaRadarObserver(*configuration())
    for frame in range(8):
      t = START + frame * STEP
      sm = fusion_state(t)
      observer.update([can_message(t, scan_frames(index=frame, life=1 + 2 * frame))], sm, SHADOW_SETTINGS, t + 1, sm)
    t += STEP
    msg = observer.update([can_message(t, scan_frames(index=8, life=17))], state(t), SETTINGS, t)
    self.assertNotIn('fusion', msg.hondaRadarObservations.to_dict())
    self.assertIsNone(observer.fusion)
    t += STEP
    sm = fusion_state(t)
    msg = observer.update([can_message(t, scan_frames(index=10, life=21))], sm, SHADOW_SETTINGS, t + 1, sm)
    self.assertEqual(msg.hondaRadarObservations.fusion.tracks[0].observations, 1)
    self.assertEqual(msg.hondaRadarObservations.fusion.leads[0].distanceWeight, 0)

  def test_input_calculation_serialization_and_init_failures_preserve_raw_log(self):
    for method, result in (('record_inputs', RuntimeError('fixture input')), ('update', RuntimeError('fixture compute')),
                           ('update', {'notAField': 1}), ('__init__', RuntimeError('fixture import/init'))):
      observer = HondaRadarObserver(*configuration())
      kwargs = {'side_effect': result} if isinstance(result, Exception) else {'return_value': result}
      with patch.object(HondaRadarFusion, method, **kwargs), patch('openpilot.common.swaglog.cloudlog.exception'):
        sm = fusion_state(START)
        msg = observer.update([can_message(START, scan_frames())], sm, SHADOW_SETTINGS, START + 1, sm)
      value = messaging.log_from_bytes(msg.to_bytes()).hondaRadarObservations
      self.assertTrue(value.observations[-1].positionValid)
      self.assertTrue(value.active)
      self.assertFalse(value.fusion.active or value.fusion.controlAllowed)
      self.assertEqual(value.fusion.reason, 'fusionError')
      self.assertFalse(value.fusion.tracks or value.fusion.leads)
      # A good subsequent input can recover without restarting raw telemetry.
      t = START + STEP
      sm = fusion_state(t)
      msg = observer.update([can_message(t, scan_frames(index=1, life=3))], sm, SHADOW_SETTINGS, t + 1, sm)
      self.assertTrue(msg.hondaRadarObservations.fusion.active)
      self.assertEqual(msg.hondaRadarObservations.fusion.tracks[0].observations, 1)

  def test_shadow_process_receives_model_and_can_in_private_ipc(self):
    with OpenpilotPrefix():
      params = Params()
      cp, cp_sp = configuration()
      params.put('CarParams', cp.to_bytes(), block=True)
      params.put('CarParamsSP', cp_sp.to_bytes(), block=True)
      params.put('HondaBoschARadarMode', 2, block=True)
      for key in ('AlphaLongitudinalEnabled', 'ExperimentalMode', 'DynamicExperimentalControl'):
        params.put_bool(key, True, block=True)
      services = ['can', 'selfdriveState', 'longitudinalPlanSP', 'carControl', 'carState', 'modelV2']
      pm = messaging.PubMaster(services)
      subscriber = messaging.sub_sock(SERVICE, timeout=70)
      process = subprocess.Popen([sys.executable, '-m', 'openpilot.selfdrive.car.honda_radar_observer'],
                                 env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
      received = None
      try:
        deadline = time.monotonic() + 8
        index = 0
        while time.monotonic() < deadline and process.poll() is None:
          now = time.monotonic_ns()
          sm = fusion_state(now)
          for service in services[1:]:
            message = messaging.new_message(service, valid=True)
            message.logMonoTime = now
            setattr(message, service, sm[service])
            pm.send(service, message)
          pm.send('can', can_message(now, scan_frames(index=index % 16, life=(1 + 2 * index) % 4095, velocity=864)).as_builder())
          index += 1
          candidate = messaging.recv_one(subscriber)
          if candidate is not None and candidate.valid and candidate.hondaRadarObservations.fusion.active:
            fusion = candidate.hondaRadarObservations.fusion
            if fusion.tracks and fusion.leads and fusion.leads[0].distanceWeight > 0:
              received = candidate
              break
          time.sleep(.02)
      finally:
        process.terminate()
        output, _ = process.communicate(timeout=5)
      self.assertIsNotNone(received, output.decode(errors='replace'))
      self.assertFalse(received.hondaRadarObservations.fusion.controlAllowed)
      del pm, subscriber


if __name__ == '__main__':
  unittest.main()
