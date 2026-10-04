"""Bounded same-input comparison with real DEC, MPC and LongControl."""
from unittest.mock import patch

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.cereal import messaging
from opendbc.car.honda import hondacan
from opendbc.can.packer import CANPacker
from opendbc.car.honda.values import CAR, DBC
from opendbc.car import Bus
from openpilot.selfdrive.car.honda_radar_observer import HondaRadarObserver
from openpilot.selfdrive.car.tests.test_honda_radar_observer import SETTINGS, can_message, scan_frames, state as observer_state
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_planner, make_state, consume


class TestHondaRadarControlIsolation(OpenpilotTestCase):
  def test_observer_cannot_change_plans_dec_longcontrol_or_can_payloads(self):
    for key in ('AlphaLongitudinalEnabled', 'ExperimentalMode', 'DynamicExperimentalControl'):
      Params().put_bool(key, True, block=True)

    def run(with_observer):
      planner, extension = make_planner()
      controller = LongControl(planner.CP, extension)
      sm = make_state(experimental=True)
      sm.seen = dict.fromkeys(sm, True)
      observer = HondaRadarObserver(planner.CP, extension) if with_observer else None
      packer = CANPacker(DBC[CAR.HONDA_ACCORD][Bus.pt])
      bus = hondacan.CanBus(planner.CP)
      outputs = []
      for frame in range(20):
        if frame == 8:
          sm['modelV2'].position.x = [0.0] * 33
          sm['modelV2'].action.desiredAcceleration = -2.0
          sm['modelV2'].action.shouldStop = True
        plan, accel = consume(planner, controller, sm, frame)
        dec_state = (planner.dec.enabled(), planner.dec.active(), planner.dec.mode())
        self.assertTrue(dec_state[0] and dec_state[1])
        # Actual unchanged Honda command packer, with identical actuator inputs.
        can = hondacan.create_acc_commands(packer, bus, True, True, accel[-1], 0.0, 0, planner.CP)
        outputs.append((plan.aTarget, plan.shouldStop, dec_state, accel, can))
        if observer is not None:
          message = messaging.new_message('longitudinalPlanSP').longitudinalPlanSP
          message.dec.enabled, message.dec.active, message.dec.state = dec_state
          t = sm.logMonoTime['modelV2']
          telemetry_sm = observer_state(t)
          telemetry_sm['longitudinalPlanSP'] = message
          telemetry_sm['selfdriveState'] = sm['selfdriveState']
          telemetry_sm['carControl'] = sm['carControl']
          before = {k: v.to_dict() for k, v in sm.items()}
          observer.update([can_message(t, scan_frames(index=frame % 16, life=1 + 2 * frame))], telemetry_sm, SETTINGS, t + 2_000_000)
          self.assertEqual({k: v.to_dict() for k, v in sm.items()}, before)
      return outputs

    # Both arms have identical deterministic clocks and independent controller state.
    with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=1_000_000_000):
      baseline = run(False)
      observed = run(True)
    self.assertEqual(baseline, observed)
    self.assertEqual({entry[2][2] for entry in baseline}, {'acc', 'blended'})
