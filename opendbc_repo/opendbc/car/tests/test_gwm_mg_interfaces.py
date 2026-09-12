import itertools
import unittest
from unittest.mock import patch

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, gen_empty_fingerprint, structs
from opendbc.car.gwm.interface import CarInterface as GwmInterface
from opendbc.car.gwm.values import CAR as GWM_CAR, GwmSafetyFlags
from opendbc.car.mg.interface import CarInterface as MgInterface
from opendbc.car.mg.values import CAR as MG_CAR


class InterfaceHarness:
  def __init__(self, interface_class, platform, alpha_long):
    fingerprint = gen_empty_fingerprint()
    car_params = interface_class.get_params(platform, fingerprint, [], alpha_long, False, False)
    car_params_sp = interface_class.get_params_sp(car_params, platform, fingerprint, [], alpha_long, False, False)
    self.interface = interface_class(car_params, car_params_sp)
    self.interface.update([])
    self.packer = CANPacker(next(iter(self.interface.can_parsers.values())).dbc_name)
    self.frame = 0

    if interface_class is GwmInterface:
      self.values = {
        (Bus.main, "WHEEL_SPEEDS"): {
          "FRONT_LEFT_WHEEL_SPEED": 36,
          "FRONT_RIGHT_WHEEL_SPEED": 36,
          "REAR_LEFT_WHEEL_SPEED": 36,
          "REAR_RIGHT_WHEEL_SPEED": 36,
        },
        (Bus.main, "RX_STEER_RELATED"): {
          "A_RX_STEER_REQUESTED": 1,
          "B_RX_DRIVER_TORQUE": 0,
          "B_RX_EPS_TORQUE": 0,
        },
        (Bus.main, "CAR_OVERALL_SIGNALS"): {"DRIVE_MODE": 1},
        (Bus.cam, "ACC"): {"CRUISE_STATE_2": 1, "CAR_DISTANCE_SELECTION": 2},
      }
      messages = [("STEER_CMD", 50), ("ACC_CMD", 50)]
    else:
      self.values = {
        (Bus.pt, "SCS_HSC2_FrP19"): {"VehSpdAvgHSC2": 36},
        (Bus.pt, "EPS_HSC2_FrP03"): {"DrvrStrgDlvrdToqHSC2": 0, "ChLKARespToqHSC2": 0},
        (Bus.pt, "GW_HSC2_ECM_FrP04"): {"TrEstdGearHSC2": 1},
      }
      messages = [("FVCM_HSC2_FrP03", 50)]
    self.output_parser = CANParser(self.packer.dbc.name, messages, 0)

  def update(self):
    packets = []
    for bus, parser in self.interface.can_parsers.items():
      for address in sorted(parser.addresses):
        message = parser.dbc.addr_to_msg[address]
        values = self.values.get((bus, message.name), {})
        packets.append(self.packer.make_can_msg(message.name, parser.bus, values))
    self.frame += 1
    now_nanos = self.frame * 10_000_000
    state, state_sp = self.interface.update([(now_nanos, packets)])
    return state, state_sp, now_nanos


class TestGwmMgInterfaces(unittest.TestCase):
  def test_gwm_alpha_long_gate(self):
    for alpha_long in (False, True):
      with self.subTest(alpha_long=alpha_long):
        harness = InterfaceHarness(GwmInterface, GWM_CAR.GWM_HAVAL_H6, alpha_long)
        car_params = harness.interface.CP
        self.assertTrue(car_params.alphaLongitudinalAvailable)
        self.assertEqual(car_params.openpilotLongitudinalControl, alpha_long)
        self.assertEqual(bool(car_params.safetyConfigs[-1].safetyParam & GwmSafetyFlags.LONG_CONTROL), alpha_long)
        self.assertEqual(car_params.safetyConfigs[-1].safetyModel, structs.CarParams.SafetyModel.gwm)

  def test_mg_remains_dashcam_only(self):
    for alpha_long in (False, True):
      with self.subTest(alpha_long=alpha_long):
        harness = InterfaceHarness(MgInterface, MG_CAR.MG_5_EV, alpha_long)
        car_params = harness.interface.CP
        self.assertTrue(car_params.dashcamOnly)
        self.assertFalse(car_params.alphaLongitudinalAvailable)
        self.assertFalse(car_params.openpilotLongitudinalControl)
        self.assertEqual(car_params.safetyConfigs[-1].safetyParam, 0)

  def test_control_pipeline(self):
    platforms = ((GwmInterface, GWM_CAR.GWM_HAVAL_H6), (MgInterface, MG_CAR.MG_5_EV))
    control_states = ((False, False), (True, False), (False, True), (True, True), (False, False))
    for (interface_class, platform), alpha_long, as_reader, accel in itertools.product(platforms, (False, True), (False, True), (-1.0, 1.0)):
      with self.subTest(platform=platform, alpha_long=alpha_long, as_reader=as_reader, accel=accel):
        harness = InterfaceHarness(interface_class, platform, alpha_long)
        control_sp = structs.CarControlSP()
        control_sp.mads.enabled = True
        control_sp.mads.active = True

        for lat_active, long_active in control_states:
          control = structs.CarControl(enabled=lat_active or long_active, latActive=lat_active, longActive=long_active)
          control.actuators.torque = 0.3
          control.actuators.accel = accel
          control.actuators.longControlState = "stopping" if accel < 0 else "pid"
          control.hudControl.leadDistanceBars = 2
          if as_reader:
            control = control.as_reader()
          original_control = control.to_dict()

          for _ in range(4):
            state, state_sp, now_nanos = harness.update()
            self.assertIs(state, harness.interface.CS.out)
            self.assertIs(state_sp, harness.interface.CS.out_sp)
            self.assertIsInstance(state_sp, structs.CarStateSP)
            self.assertGreater(state.vEgoRaw, 0)
            self.assertEqual(state.steeringTorque, 0)
            controller_frame = harness.interface.CC.frame
            with patch.object(harness.interface.CC, "update", wraps=harness.interface.CC.update) as controller_update:
              actuators, can_sends = harness.interface.apply(control, control_sp, now_nanos)
            controller_update.assert_called_once_with(control, control_sp, harness.interface.CS, now_nanos)
            self.assertIs(controller_update.call_args.args[1], control_sp)
            self.assertEqual(control.to_dict(), original_control)
            self.assertEqual(harness.interface.CP.dashcamOnly, interface_class is MgInterface)

            harness.output_parser.update([(now_nanos, can_sends)])
            addresses = {address for address, _, _ in can_sends}
            if interface_class is GwmInterface:
              self.assertEqual(harness.interface.lat_active, lat_active)
              self.assertEqual(harness.interface.current_personality, control.hudControl.leadDistanceBars)
              self.assertEqual(0x12B in addresses, controller_frame % 2 == 0)
              self.assertEqual(0x143 in addresses, alpha_long and controller_frame % 2 == 0)
              if 0x12B in addresses:
                steering = harness.output_parser.vl["STEER_CMD"]
                self.assertEqual(steering["STEER_REQUEST"], lat_active)
                self.assertEqual(steering["TORQUE_CMD"], actuators.torqueOutputCan)
                self.assertEqual(steering["TORQUE_CMD"] != 0, lat_active)
              if 0x143 in addresses:
                longitudinal = harness.output_parser.vl["ACC_CMD"]
                if not long_active:
                  self.assertEqual(longitudinal["GAS_CMD"], 0)
                  self.assertEqual(longitudinal["BRAKE_CMD"], 0)
                elif accel < 0:
                  self.assertEqual(longitudinal["GAS_CMD"], 0)
                  self.assertLess(longitudinal["BRAKE_CMD"], 0)
                  self.assertEqual(longitudinal["STANDSTILL_1"], 1)
                  self.assertEqual(longitudinal["STANDSTILL_2"], 3)
                  self.assertEqual(longitudinal["STANDSTILL_3"], 0)
                else:
                  self.assertGreater(longitudinal["GAS_CMD"], 0)
                  self.assertEqual(longitudinal["BRAKE_CMD"], 0)
            else:
              self.assertLessEqual(addresses, {0x1FD})
              self.assertEqual(0x1FD in addresses, controller_frame % 2 == 0)
              if 0x1FD in addresses:
                steering = harness.output_parser.vl["FVCM_HSC2_FrP03"]
                self.assertEqual(steering["LKAReqToqStsHSC2"], lat_active)
                self.assertEqual(steering["LKAReqToqHSC2"], actuators.torqueOutputCan)
                self.assertEqual(steering["LKAReqToqHSC2"] != 0, lat_active)
            actuators.torque = -0.9
            actuators.accel = 0.5
            self.assertEqual(control.to_dict(), original_control)

  def test_apply_default_timestamp(self):
    for interface_class, platform in ((GwmInterface, GWM_CAR.GWM_HAVAL_H6), (MgInterface, MG_CAR.MG_5_EV)):
      with self.subTest(platform=platform):
        harness = InterfaceHarness(interface_class, platform, False)
        harness.update()
        control = structs.CarControl().as_reader()
        control_sp = structs.CarControlSP()
        with patch("opendbc.car.interfaces.time.monotonic", return_value=1.25), \
             patch.object(harness.interface.CC, "update", wraps=harness.interface.CC.update) as controller_update:
          harness.interface.apply(control, control_sp)
        controller_update.assert_called_once_with(control, control_sp, harness.interface.CS, 1_250_000_000)
