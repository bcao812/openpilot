"""Native schema, observer boundary and isolated process tests for radar fusion."""
from collections import deque
from itertools import permutations
import os
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.car.honda_radar_fusion import HondaRadarFusion
from openpilot.selfdrive.car.honda_radar_observer import HondaRadarObserver
from openpilot.selfdrive.car.honda_radar_observer_config import ObserverSettings, SERVICE
from openpilot.selfdrive.car.tests.test_honda_radar_observer import START, SETTINGS, can_message, configuration, scan_frames, state

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


def exact_fusion_state(value):
  if isinstance(value, np.ndarray):
    return value.dtype.str, value.shape, value.tobytes(), value.flags.writeable
  if isinstance(value, float):
    return value.hex()
  if isinstance(value, dict):
    return {key: exact_fusion_state(item) for key, item in value.items()}
  if isinstance(value, deque):
    return 'deque', value.maxlen, tuple(exact_fusion_state(item) for item in value)
  if isinstance(value, (list, tuple)):
    return type(value).__name__, tuple(exact_fusion_state(item) for item in value)
  if hasattr(value, '__dict__'):
    return exact_fusion_state(vars(value))
  return value


class TestHondaRadarShadow(unittest.TestCase):
  def test_entrypoint_takes_validation_clock_after_receiving_new_model(self):
    from openpilot.selfdrive.car import honda_radar_observer as runtime
    cp, cp_sp = configuration()
    params = Mock()
    values = {'CarParams': cp.to_bytes(), 'CarParamsSP': cp_sp.to_bytes()}
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
      msg = observer.update([], sm, SETTINGS, START + 1, sm)
      candidate = msg.hondaRadarObservations.fusion.leads[0]
      original = get_RadarState_from_vision(sm['modelV2'].leadsV3[0], sm['carState'].vEgo,
                                           sm['modelV2'].velocity.x[0], sm['modelV2'].leadsV3[0].prob)
      for key in ('present', 'dRel', 'yRel', 'vRel', 'vLead'):
        self.assertEqual(getattr(candidate, key), original[key])
      self.assertEqual(candidate.aLead, original['aLeadK'])
      self.assertEqual(candidate.distanceWeight, 0)

  def test_fusion_wire_roundtrip_and_both_dec_modes(self):
    for dec in ('acc', 'blended'):
      observer = HondaRadarObserver(*configuration(), session_id='shadow-test')
      for frame in range(12):
        t = START + frame * STEP
        sm = fusion_state(t)
        sm['longitudinalPlanSP'].dec.state = dec
        before = {key: value.to_dict() for key, value in sm.items()}
        msg = observer.update([can_message(t, scan_frames(index=frame % 16, life=1 + 2 * frame, velocity=864))],
                              sm, SETTINGS, t + 1, sm)
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

  def test_decoder_events_match_wire_fusion_outputs_and_internal_state(self):
    # Only events emitted by the real decoder qualify for the direct view.
    # Arbitrary sparse dictionaries are not equivalent to native zero defaults.
    scans = [scan_frames(index=frame % 16, life=1 + 2 * frame, velocity=864) for frame in range(12)]
    streams = {
      'continuous': [(frame * STEP, frames) for frame, frames in enumerate(scans)],
      'late_aux': [(0, scans[0][:4]), (30_000_000, scans[0][4:]), (STEP, scans[1]), (2 * STEP, scans[2])],
      'incomplete': [(0, scans[0][:3]), (4 * STEP, []), (5 * STEP, scans[5])],
      'expired': [(0, scans[0]), (4 * STEP, []), (5 * STEP, scans[5])],
      'invalid_geometry': [(0, scans[0]), (STEP, scan_frames(index=1, life=3, distance=0, velocity=864)), (2 * STEP, scans[2])],
      'invalid_velocity': [(0, scans[0]), (STEP, scan_frames(index=1, life=3, velocity=2047)), (2 * STEP, scans[2])],
      'conflicting_frame': [(0, scans[0]),
                            (STEP, scans[1] + scan_frames(index=1, life=3, distance=1001, velocity=864)[:1]),
                            (2 * STEP, scans[2])],
      'duplicate': [(0, scans[0] + scans[0]), (STEP, scans[1]), (2 * STEP, scans[2])],
      'migration': [(frame * STEP, scan_frames(slot=frame % 3, index=frame, life=1 + 2 * frame, oem_id=7, velocity=864))
                    for frame in range(8)] + [(9 * STEP, [])],
      'plateau_and_wrap': [(frame * STEP, scan_frames(slot=frame % 2, index=(14 + frame) % 16, life=4094, oem_id=7, velocity=864))
                           for frame in range(8)],
      'slot_conflict': [(0, scan_frames(oem_id=7, velocity=864)),
                        (STEP, scan_frames(index=1, life=3, oem_id=7, velocity=864) +
                         scan_frames(slot=1, index=1, life=3, oem_id=7, velocity=864)),
                        (2 * STEP, scan_frames(slot=1, index=2, life=5, oem_id=7, velocity=864))],
      'reassignment': [(0, scan_frames(oem_id=7, velocity=864)),
                       (STEP, scan_frames(index=1, life=3, oem_id=8, velocity=864)),
                       (2 * STEP, scan_frames(index=2, life=5, oem_id=8, velocity=864))],
      'no_radar': [(0, []), (STEP, [])],
    }
    for order in permutations(range(5)):
      streams[f'order_{order}'] = [(0, [scans[0][part] for part in order])]
    reasons = set()
    for dec in ('acc', 'blended'):
      for name, steps in streams.items():
        with self.subTest(dec=dec, stream=name):
          observer = HondaRadarObserver(*configuration(), session_id='decoder-view-test')
          wire_fusion = HondaRadarFusion()
          for offset, frames in steps:
            t = START + offset
            sm = fusion_state(t)
            sm['longitudinalPlanSP'].dec.state = dec
            before = {key: value.to_dict() for key, value in sm.items()}
            # The observer retains ego history on nonpublication ticks too.
            wire_fusion.record_inputs(sm, t + 1)
            msg = observer.update([can_message(t, frames)] if frames else [], sm, SETTINGS, t + 1, sm)
            if msg is not None:
              decoded = messaging.log_from_bytes(msg.to_bytes()).hondaRadarObservations
              expected = wire_fusion.update(decoded, sm, t + 1)
              actual = decoded.fusion.to_dict()
              del actual['processingTimeNs'], expected['processingTimeNs']
              self.assertEqual(exact_fusion_state(actual), exact_fusion_state(expected), (name, offset))
              self.assertFalse(decoded.fusion.calibrated or decoded.fusion.controlAllowed)
              reasons.update(event.reason for event in decoded.observations)
            self.assertEqual(exact_fusion_state(observer.fusion), exact_fusion_state(wire_fusion), (name, offset))
            self.assertEqual({key: value.to_dict() for key, value in sm.items()}, before)
          if name == 'migration':
            self.assertEqual(observer.fusion.tracks[7].slot, 1)
    self.assertTrue({'positionOnly', 'uncalibrated', 'incomplete', 'expired', 'invalidGeometry',
                     'invalidVelocity', 'conflictingFrame'}.issubset(reasons), reasons)

  def test_decoder_schema_rejection_precedes_fusion(self):
    observer = HondaRadarObserver(*configuration())
    drain = observer.decoder.drain
    def malformed_events():
      events = drain()
      events[0]['notAnObservationField'] = 1
      return events
    sm = fusion_state(START)
    with patch.object(observer.decoder, 'drain', side_effect=malformed_events), \
         patch.object(HondaRadarFusion, 'update') as calculate, \
         self.assertRaisesRegex(AttributeError, 'notAnObservationField'):
      observer.update([can_message(START, scan_frames())], sm, SETTINGS, START + 1, sm)
    calculate.assert_not_called()

  def test_each_configuration_gate_clears_candidates(self):
    for key in ('alpha', 'experimental', 'dec'):
      observer = HondaRadarObserver(*configuration())
      sm = fusion_state(START)
      observer.update([can_message(START, scan_frames())], sm, SETTINGS, START + 1, sm)
      t = START + STEP
      sm = fusion_state(t)
      values = {'alpha': True, 'experimental': True, 'dec': True}
      values[key] = False
      msg = observer.update([], sm, ObserverSettings(**values), t + 1, sm)
      self.assertFalse(msg.hondaRadarObservations.fusion.active, key)
      self.assertFalse(msg.hondaRadarObservations.fusion.tracks, key)
      self.assertFalse(msg.hondaRadarObservations.fusion.leads, key)

  def test_configuration_exit_and_reentry_do_not_reuse_confirmation(self):
    observer = HondaRadarObserver(*configuration())
    for frame in range(8):
      t = START + frame * STEP
      sm = fusion_state(t)
      observer.update([can_message(t, scan_frames(index=frame, life=1 + 2 * frame))], sm, SETTINGS, t + 1, sm)
    t += STEP
    msg = observer.update([can_message(t, scan_frames(index=8, life=17))], state(t), ObserverSettings(), t)
    self.assertFalse(msg.hondaRadarObservations.active or msg.hondaRadarObservations.fusion.active)
    self.assertFalse(observer.fusion.tracks)
    t += STEP
    sm = fusion_state(t)
    msg = observer.update([can_message(t, scan_frames(index=10, life=21))], sm, SETTINGS, t + 1, sm)
    self.assertEqual(msg.hondaRadarObservations.fusion.tracks[0].observations, 1)
    self.assertEqual(msg.hondaRadarObservations.fusion.leads[0].distanceWeight, 0)

  def test_input_calculation_serialization_and_init_failures_preserve_raw_log(self):
    baseline = HondaRadarObserver(*configuration())
    expected = baseline.update([can_message(START, scan_frames())], state(START), SETTINGS, START + 1)
    expected_observations = expected.hondaRadarObservations.to_dict()['observations']
    for method, result in (('record_inputs', RuntimeError('fixture input')), ('update', RuntimeError('fixture compute')),
                           ('update', {'notAField': 1}), ('__init__', RuntimeError('fixture import/init'))):
      observer = HondaRadarObserver(*configuration())
      kwargs = {'side_effect': result} if isinstance(result, Exception) else {'return_value': result}
      with patch.object(HondaRadarFusion, method, **kwargs), patch('openpilot.common.swaglog.cloudlog.exception'):
        sm = fusion_state(START)
        msg = observer.update([can_message(START, scan_frames())], sm, SETTINGS, START + 1, sm)
      value = messaging.log_from_bytes(msg.to_bytes()).hondaRadarObservations
      self.assertEqual(value.to_dict()['observations'], expected_observations)
      self.assertTrue(value.observations[-1].positionValid)
      self.assertTrue(value.active)
      self.assertFalse(value.fusion.active or value.fusion.controlAllowed)
      self.assertEqual(value.fusion.reason, 'fusionError')
      self.assertFalse(value.fusion.tracks or value.fusion.leads)
      # A good subsequent input can recover without restarting raw telemetry.
      t = START + STEP
      sm = fusion_state(t)
      msg = observer.update([can_message(t, scan_frames(index=1, life=3))], sm, SETTINGS, t + 1, sm)
      self.assertTrue(msg.hondaRadarObservations.fusion.active)
      self.assertEqual(msg.hondaRadarObservations.fusion.tracks[0].observations, 1)

  def test_fusion_failure_retracts_existing_state_before_recovery(self):
    observer = HondaRadarObserver(*configuration())
    for frame in range(4):
      t = START + frame * STEP
      sm = fusion_state(t)
      observer.update([can_message(t, scan_frames(index=frame, life=1 + 2 * frame, velocity=864))],
                      sm, SETTINGS, t + 1, sm)
    self.assertTrue(observer.fusion.tracks and observer.fusion.bindings and observer.fusion.history.samples)
    t = START + 4 * STEP
    sm = fusion_state(t)
    with patch.object(observer.fusion, 'update', side_effect=RuntimeError('fixture existing candidate')), \
         patch('openpilot.common.swaglog.cloudlog.exception'):
      msg = observer.update([can_message(t, scan_frames(index=4, life=9, velocity=864))], sm, SETTINGS, t + 1, sm)
    value = messaging.log_from_bytes(msg.to_bytes()).hondaRadarObservations
    self.assertTrue(value.observations[-1].positionValid)
    self.assertEqual(value.fusion.reason, 'fusionError')
    self.assertFalse(value.fusion.tracks or value.fusion.leads or value.fusion.controlAllowed)
    self.assertFalse(observer.fusion.tracks or observer.fusion.bindings or observer.fusion.history.samples)
    t += STEP
    sm = fusion_state(t)
    msg = observer.update([can_message(t, scan_frames(index=5, life=11, velocity=864))], sm, SETTINGS, t + 1, sm)
    self.assertEqual(msg.hondaRadarObservations.fusion.tracks[0].observations, 1)
    self.assertEqual(msg.hondaRadarObservations.fusion.leads[0].distanceWeight, 0)

  def test_fusion_process_receives_model_and_can_in_private_ipc(self):
    with OpenpilotPrefix():
      params = Params()
      cp, cp_sp = configuration()
      params.put('CarParams', cp.to_bytes(), block=True)
      params.put('CarParamsSP', cp_sp.to_bytes(), block=True)
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
