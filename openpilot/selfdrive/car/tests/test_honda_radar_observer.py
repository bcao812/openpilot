from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from opendbc.car.structs import car
from opendbc.car.honda.tests.test_bosch_a_observer import START, scan_frames
from opendbc.car.honda.values import HondaFlags
from openpilot.cereal import custom, messaging
from openpilot.common.params import Params, ParamKeyFlag
from openpilot.selfdrive.car.honda_radar_observer import HondaRadarObserver
from openpilot.selfdrive.car.honda_radar_observer_config import ObserverSettings, SERVICE, observer_requested

SETTINGS = ObserverSettings(1, True, True, True)


def configuration():
  return (car.CarParams.new_message(carFingerprint='HONDA_ACCORD', flags=int(HondaFlags.BOSCH), openpilotLongitudinalControl=True),
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
  msg.logMonoTime = t
  msg.can = [dict(address=a, dat=d, src=s) for a, d, s in frames]
  return msg.as_reader()


class TestHondaRadarObserver(unittest.TestCase):
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
      for value in (0, 2, 3, -1):
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

  def test_entry_boundary_does_not_expand_control_scope(self):
    for settings in (ObserverSettings(1, False, True, True), ObserverSettings(1, True, False, True),
                     ObserverSettings(1, True, True, False), ObserverSettings(3, True, True, True)):
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


if __name__ == '__main__':
  unittest.main()
