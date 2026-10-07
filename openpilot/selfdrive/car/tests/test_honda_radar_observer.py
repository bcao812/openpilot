from pathlib import Path
from tempfile import TemporaryDirectory
import os
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

from opendbc.car.structs import car
from opendbc.car.honda.tests.test_bosch_a_observer import START, scan_frames
from opendbc.car.honda.values import HondaFlags
from openpilot.cereal import custom, messaging
from openpilot.common.params import Params, ParamKeyFlag
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.car.honda_radar_observer import HondaRadarObserver, configure_observer_process
from openpilot.selfdrive.car.honda_radar_observer_config import ObserverSettings, SERVICE, observer_requested

SETTINGS = ObserverSettings(1, True, True, True)


def configuration():
  return (car.CarParams.new_message(carFingerprint='HONDA_ACCORD', flags=int(HondaFlags.BOSCH), openpilotLongitudinalControl=True,
                                    safetyConfigs=[{'safetyModel': car.CarParams.SafetyModel.hondaBosch}]),
          custom.CarParamsSP.new_message())


def state(now_ns, mode='acc'):
  data = {s: getattr(messaging.new_message(s), s) for s in ('selfdriveState', 'longitudinalPlanSP', 'carControl')}
  data['selfdriveState'].experimentalMode = True
  data['longitudinalPlanSP'].dec.enabled = True
  data['longitudinalPlanSP'].dec.active = True
  data['longitudinalPlanSP'].dec.state = mode
  data['carControl'].longActive = True

  class State(dict):
    pass
  sm = State(data)
  sm.valid = dict.fromkeys(data, True)
  sm.seen = dict.fromkeys(data, True)
  sm.logMonoTime = dict.fromkeys(data, now_ns)
  return sm


def can_message(t, frames):
  msg = messaging.new_message('can', len(frames))
  msg.valid = True
  msg.logMonoTime = t
  msg.can = [{'address': a, 'dat': d, 'src': s} for a, d, s in frames]
  return msg.as_reader()


class TestHondaRadarObserver(unittest.TestCase):
  def test_observer_scheduling_excludes_control_cores_without_changing_host_affinity(self):
    from openpilot.common import realtime
    with patch.object(realtime, 'PC', False), patch.object(os, 'sched_setaffinity') as affinity, \
         patch.object(os, 'sched_setscheduler') as scheduler:
      configure_observer_process()
      affinity.assert_called_once_with(0, [0, 1, 2])
      scheduler.assert_called_once_with(0, os.SCHED_OTHER, os.sched_param(0))
    before = os.sched_getaffinity(0)
    with patch.object(realtime, 'PC', True), patch.object(os, 'sched_setaffinity') as affinity, \
         patch.object(os, 'sched_setscheduler') as scheduler:
      configure_observer_process()
      affinity.assert_not_called()
      scheduler.assert_not_called()
    self.assertEqual(os.sched_getaffinity(0), before)

  def test_persistent_typed_settings_and_fail_closed_modes(self):
    cp, _ = configuration()
    with TemporaryDirectory() as directory:
      params = Params(directory)
      self.assertFalse(observer_requested(True, params, cp))
      for key in ('AlphaLongitudinalEnabled', 'ExperimentalMode', 'DynamicExperimentalControl'):
        params.put_bool(key, True, block=True)
      params.put('HondaBoschARadarMode', 1, block=True)
      params.put_bool('HondaBoschARadarOverlay', True, block=True)
      for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION,
                   ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION, ParamKeyFlag.CLEAR_ON_IGNITION_ON):
        params.clear_all(flag)
        self.assertTrue(observer_requested(True, params, cp))
        self.assertTrue(params.get_bool('HondaBoschARadarOverlay'))
      params.put('HondaBoschARadarMode', 2, block=True)
      self.assertTrue(observer_requested(True, params, cp))
      params.put('HondaBoschARadarMode', 3, block=True)
      self.assertTrue(observer_requested(True, params, cp))
      for value in (0, 4, -1):
        params.put('HondaBoschARadarMode', value, block=True)
        self.assertFalse(observer_requested(True, params, cp))
      Path(params.get_param_path('HondaBoschARadarMode')).write_bytes(b'bad')
      self.assertFalse(observer_requested(True, params, cp))

  def test_wire_roundtrip_and_both_dec_internal_modes(self):
    for mode in ('acc', 'blended'):
      cp, cp_sp = configuration()
      observer = HondaRadarObserver(cp, cp_sp, 'test-session')
      sm = state(START, mode)
      original = {k: v.to_dict() for k, v in sm.items()}
      msg = observer.update([can_message(START, scan_frames())], sm, SETTINGS, START + 1)
      self.assertEqual(msg.which(), SERVICE)
      self.assertTrue(msg.valid)
      wire = msg.to_bytes()
      decoded = messaging.log_from_bytes(wire).hondaRadarObservations
      self.assertTrue(decoded.active)
      self.assertFalse(decoded.calibrated)
      self.assertEqual(len(decoded.observations), 2)
      self.assertEqual(decoded.observations[-1].positionMonoTime, START)
      self.assertEqual(decoded.scope.decMode, 1 if mode == 'blended' else 0)
      self.assertEqual({k: v.to_dict() for k, v in sm.items()}, original)

  def test_runtime_scope_loss_stale_messages_and_no_can_timeout(self):
    cp, cp_sp = configuration()
    observer = HondaRadarObserver(cp, cp_sp)
    observer.update([can_message(START, scan_frames())], state(START), SETTINGS, START + 1)
    msg = observer.update([], state(START + 250_000_000), SETTINGS, START + 250_000_000)
    self.assertEqual(msg.hondaRadarObservations.reason, 'radarSilence')
    self.assertIn('expired', [e.reason for e in msg.hondaRadarObservations.observations])
    sm = state(START + 300_000_000)
    sm['longitudinalPlanSP'].dec.active = False
    msg = observer.update([], sm, SETTINGS, START + 300_000_000)
    self.assertFalse(msg.hondaRadarObservations.active)
    self.assertFalse(observer.decoder.events)
    msg = observer.update([], state(START), SETTINGS, START + 900_000_000)
    self.assertEqual(msg.hondaRadarObservations.reason, 'scopeStale')

  def test_can_socket_batch_boundaries_preserve_observations(self):
    envelopes = [can_message(START + i * 1_000_000, [frame]) for i, frame in enumerate(scan_frames())]
    expected = None
    for cuts in ((), (1, 2, 3, 4), (4,), (1, 3), (2, 4)):
      observer = HondaRadarObserver(*configuration(), session_id='batch-fixture')
      events = []
      left = 0
      for right in (*cuts, 5):
        t = envelopes[right - 1].logMonoTime + 1
        msg = observer.update(envelopes[left:right], state(t), SETTINGS, t)
        if msg is not None:
          events.extend(msg.hondaRadarObservations.to_dict()['observations'])
        left = right
      msg = observer.update([], state(START + 60_000_000), SETTINGS, START + 60_000_000)
      if msg is not None:
        events.extend(msg.hondaRadarObservations.to_dict()['observations'])
      if expected is None:
        expected = events
      self.assertEqual(events, expected)
    self.assertEqual(len(expected), 2)

  def test_entry_boundary_does_not_expand_control_scope(self):
    for settings in (ObserverSettings(1, False, True, True), ObserverSettings(1, True, False, True),
                     ObserverSettings(1, True, True, False), ObserverSettings(4, True, True, True)):
      observer = HondaRadarObserver(*configuration())
      msg = observer.update([can_message(START, scan_frames())], state(START), settings, START)
      self.assertFalse(msg.hondaRadarObservations.active)
      self.assertEqual(observer.decoder.counts['frames'], 0)
    cp, cp_sp = configuration()
    cp_sp.enableGasInterceptor = True
    msg = HondaRadarObserver(cp, cp_sp).update([], state(START), SETTINGS, START)
    self.assertFalse(msg.hondaRadarObservations.active)

  def test_future_can_and_publication_bounds(self):
    observer = HondaRadarObserver(*configuration())
    msg = observer.update([can_message(START + 1_000_000_000, scan_frames())], state(START), SETTINGS, START)
    self.assertEqual(msg.hondaRadarObservations.rejectedEnvelopes, 1)
    self.assertFalse(msg.hondaRadarObservations.observations)
    for i in range(1, 20):
      self.assertIsNone(observer.update([], state(START + i), SETTINGS, START + i))
    self.assertIsNotNone(observer.update([], state(START + 1_000_000_000), SETTINGS, START + 1_000_000_000))

  def test_bus_offset_is_derived_from_real_safety_configuration(self):
    cp, cp_sp = configuration()
    cp.safetyConfigs = [{'safetyModel': car.CarParams.SafetyModel.noOutput}, {'safetyModel': car.CarParams.SafetyModel.hondaBosch}]
    observer = HondaRadarObserver(cp, cp_sp)
    self.assertEqual(observer.decoder.bus, 6)
    msg = observer.update([can_message(START, scan_frames(bus=6))], state(START), SETTINGS, START + 1)
    self.assertTrue(msg.hondaRadarObservations.observations[-1].positionValid)
    cp.safetyConfigs = []
    with self.assertRaises(ValueError):
      HondaRadarObserver(cp, cp_sp)

  def test_optional_process_health_exemption_is_narrow(self):
    import ast
    from openpilot.common.basedir import BASEDIR
    source = (Path(BASEDIR) / 'openpilot/selfdrive/selfdrived/selfdrived.py').read_text()
    tree = ast.parse(source)
    values = [ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
              and isinstance(n.target, ast.Attribute) and n.target.attr == 'ignored_processes']
    self.assertEqual(values, [{'hondaRadarObserver'}])
    # The existing critical-process predicate still sees any other failed process.
    ignored = values[0]
    self.assertFalse({'hondaRadarObserver'} - ignored)
    self.assertEqual({'radard', 'hondaRadarObserver'} - ignored, {'radard'})

  def test_real_accord_interface_preserves_production_radar_configuration(self):
    from opendbc.car import gen_empty_fingerprint
    from opendbc.car.honda.interface import CarInterface
    from opendbc.car.honda.radar_interface import RadarInterface
    from opendbc.car.honda.values import CAR
    fingerprint = gen_empty_fingerprint()
    fingerprint[1] = {0x184: 8, 0x191: 8}  # Accord Hybrid with automatic transmission
    cp = CarInterface.get_params(CAR.HONDA_ACCORD, fingerprint, [], True, False, False)
    cp_sp = CarInterface.get_params_sp(cp, CAR.HONDA_ACCORD, fingerprint, [], True, False, False)
    self.assertTrue(cp.openpilotLongitudinalControl and cp.radarUnavailable)
    self.assertFalse(cp.pcmCruise)
    self.assertTrue(SETTINGS.requested(cp))
    self.assertEqual(HondaRadarObserver(cp, cp_sp).decoder.bus, 2)
    self.assertTrue(RadarInterface(cp, cp_sp).radar_off_can)

  def test_full_event_update_distinguishes_optional_and_critical_failures(self):
    from openpilot.cereal import log
    from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import (
      new_event, selfdrive_event_harness, serialized,
    )
    for mode in ('acc', 'blended'):
      for failed in ([], ['hondaRadarObserver'], ['radard'], ['card'], ['controlsd'], ['hondaRadarObserver', 'radard']):
        with self.subTest(mode=mode, failed=failed):
          controls = new_event('controlsState')
          controls.controlsState.lateralControlState.init('pidState')
          daemon, cs = selfdrive_event_harness(controls.as_reader())
          daemon.CP.carFingerprint = 'HONDA_ACCORD'
          daemon.experimental_mode = True
          daemon.ignored_processes = {'hondaRadarObserver'}  # separately checked against the real initializer above
          plan = new_event('longitudinalPlanSP')
          plan.longitudinalPlanSP.dec = {'enabled': True, 'active': True, 'state': mode}
          manager = new_event('managerState')
          manager.managerState.processes = [{'name': name, 'running': False, 'shouldBeRunning': True} for name in failed]
          daemon.sm.update_msgs(2.01, [serialized(plan), serialized(manager)])
          daemon.update_events(cs)
          self.assertEqual(log.OnroadEvent.EventName.processNotRunning in daemon.events.names,
                           bool(set(failed) - {'hondaRadarObserver'}))
          # A cached manager message must not hide a subsequent critical failure.
          daemon.update_events(cs)
          self.assertEqual(log.OnroadEvent.EventName.processNotRunning in daemon.events.names,
                           bool(set(failed) - {'hondaRadarObserver'}))

  def test_managed_entrypoint_receives_can_and_publishes_only_observations(self):
    # Private IPC namespace: these synthetic CAN/control messages cannot reach a
    # vehicle process or another test. No device connection is made.
    with OpenpilotPrefix():
      params = Params()
      cp, cp_sp = configuration()
      params.put('CarParams', cp.to_bytes(), block=True)
      params.put('CarParamsSP', cp_sp.to_bytes(), block=True)
      params.put('HondaBoschARadarMode', 1, block=True)
      for key in ('AlphaLongitudinalEnabled', 'ExperimentalMode', 'DynamicExperimentalControl'):
        params.put_bool(key, True, block=True)
      pm = messaging.PubMaster(['can', 'selfdriveState', 'longitudinalPlanSP', 'carControl'])
      subscriber = messaging.sub_sock(SERVICE, timeout=100)
      process = subprocess.Popen([sys.executable, '-m', 'openpilot.selfdrive.car.honda_radar_observer'],
                                 env=os.environ.copy(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
      received = None
      try:
        deadline = time.monotonic() + 8
        index = 0
        while time.monotonic() < deadline and process.poll() is None:
          now = time.monotonic_ns()
          sm = state(now)
          for service in ('selfdriveState', 'longitudinalPlanSP', 'carControl'):
            message = messaging.new_message(service)
            message.valid = True
            message.logMonoTime = now
            setattr(message, service, sm[service])
            pm.send(service, message)
          pm.send('can', can_message(now, scan_frames(index=index % 16, life=1 + 2 * index)).as_builder())
          index += 1
          candidate = messaging.recv_one(subscriber)
          if candidate is not None and candidate.valid and candidate.hondaRadarObservations.active and candidate.hondaRadarObservations.observations:
            received = candidate
            break
      finally:
        process.terminate()
        output, _ = process.communicate(timeout=5)
      self.assertIsNotNone(received, output.decode(errors='replace'))
      self.assertTrue(received.hondaRadarObservations.observations[-1].positionValid)
      del pm, subscriber


if __name__ == '__main__':
  unittest.main()
