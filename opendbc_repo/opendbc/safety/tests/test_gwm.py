import unittest

from opendbc.car import gen_empty_fingerprint
from opendbc.car.crc import CRC8J1850, mk_crc8_fun
from opendbc.car.gwm import gwmcan
from opendbc.car.gwm.values import GwmSafetyFlags
from opendbc.car.structs import CarParams
from opendbc.safety.tests import common
from opendbc.safety.tests.common import CANPackerSafety
from opendbc.safety.tests.libsafety import libsafety_py


gwm_crcs = {
  0xA1: mk_crc8_fun(CRC8J1850, init_crc=0x2D, xor_out=0x2D),
  0x13B: mk_crc8_fun(CRC8J1850, init_crc=0x7F, xor_out=0x7F),
}


def checksum(message):
  address, payload, bus = message
  data = bytearray(payload)
  data[0] = gwm_crcs[address](data[1:8])
  return address, data, bus


class TestGwmSafety(common.CarSafetyTest, common.MotorTorqueSteeringSafetyTest):
  TX_MSGS = [[0xA1, 2], [0x147, 2], [0x12B, 0], [0x23D, 0]]
  TX_MSGS_LENGTHS = {(0xA1, 2): 8, (0x147, 2): 64, (0x12B, 0): 64, (0x23D, 0): 64}
  RELAY_MALFUNCTION_ADDRS = {0: (0x12B, 0x23D), 2: (0x147,)}
  FWD_BLACKLISTED_ADDRS = {0: [0x147], 2: [0x12B, 0x23D]}
  MAX_TORQUE_LOOKUP = [0], [253]
  MAX_RATE_UP = 4
  MAX_RATE_DOWN = 6
  MAX_RT_DELTA = 100
  MAX_TORQUE_ERROR = 80
  TORQUE_MEAS_TOLERANCE = 1
  MAX_GAS = 4577
  MIN_GAS = -10
  MAX_BRAKE = 107

  def setUp(self):
    self.packer = CANPackerSafety("gwm_haval_h6_mk3_generated")
    self.safety = libsafety_py.libsafety
    self._set_mode(0)

  def _set_mode(self, flags):
    self.assertEqual(self.safety.set_safety_hooks(CarParams.SafetyModel.gwm, flags), 0)
    self.safety.init_tests()
    self.counters = {}

  def _counter(self, name):
    self.counters[name] = (self.counters.get(name, 0) + 1) % 16
    return self.counters[name]

  def _speed_msg(self, speed):
    values = {
      name: speed * 3.6 for name in (
        "FRONT_LEFT_WHEEL_SPEED", "FRONT_RIGHT_WHEEL_SPEED", "REAR_LEFT_WHEEL_SPEED", "REAR_RIGHT_WHEEL_SPEED",
      )
    }
    values["COUNTER"] = self._counter("speed")
    return self.packer.make_can_msg_safety("WHEEL_SPEEDS", 0, values, fix_checksum=checksum)

  def _user_brake_msg(self, brake):
    return self.packer.make_can_msg_safety("BRAKE2", 0, {"PEDAL_BRAKE_PRESSED": bool(brake)})

  def _user_gas_msg(self, gas):
    return self.packer.make_can_msg_safety("CAR_OVERALL_SIGNALS2", 0, {"GAS_POSITION": gas})

  def _button_msg(self, enable=False, cancel=False, bus=0):
    values = {
      "AP_ENABLE_COMMAND": enable,
      "AP_CANCEL_COMMAND": cancel,
      "COUNTER": self._counter("buttons"),
    }
    return self.packer.make_can_msg_safety("STEER_AND_AP_STALK", bus, values, fix_checksum=checksum)

  def _pcm_status_msg(self, enable):
    return self._button_msg(enable=enable, cancel=not enable)

  def _torque_meas_msg(self, torque):
    return self.packer.make_can_msg_safety("RX_STEER_RELATED", 0, {"B_RX_EPS_TORQUE": torque})

  def _torque_cmd_msg(self, torque, steer_req=1):
    return self.packer.make_can_msg_safety("STEER_CMD", 0, {"TORQUE_CMD": torque, "STEER_REQUEST": steer_req})

  def _longitudinal_msg(self, gas=0, brake=0, bus=0):
    return self.packer.make_can_msg_safety("ACC_CMD", bus, {"GAS_CMD": gas, "BRAKE_CMD": -brake})

  def test_torque_absolute_limits(self):
    """Exercise the full representable signed 10-bit field instead of wrapping out-of-range test values."""
    for enabled in (False, True):
      for torque in range(-511, 513):
        self.safety.set_controls_allowed(enabled)
        self._set_prev_torque(torque)
        expected = (enabled and -self.MAX_TORQUE <= torque <= self.MAX_TORQUE) or torque == 0
        self.assertEqual(self._tx(self._torque_cmd_msg(torque)), expected, (enabled, torque))

  def test_enable_requires_rising_edge(self):
    for previous in (False, True):
      for current in (False, True):
        with self.subTest(previous=previous, current=current):
          self._set_mode(0)
          self.assertTrue(self._rx(self._button_msg(enable=previous)))
          self.safety.set_controls_allowed(False)
          self.assertTrue(self._rx(self._button_msg(enable=current)))
          self.assertEqual(self.safety.get_controls_allowed(), current and not previous)

  def test_cancel_overrides_enable(self):
    for enable in (False, True):
      with self.subTest(enable=enable):
        self._set_mode(0)
        self.assertTrue(self._rx(self._button_msg(enable=True)))
        self.assertTrue(self.safety.get_controls_allowed())
        self.assertTrue(self._rx(self._button_msg(enable=enable, cancel=True)))
        self.assertFalse(self.safety.get_controls_allowed())
        self.assertFalse(self.safety.get_acc_main_on())
        self.assertFalse(self._tx(self._torque_cmd_msg(1)))

  def test_brake_prevents_button_enable(self):
    for moving in (False, True):
      with self.subTest(moving=moving):
        self._set_mode(0)
        self.assertTrue(self._rx(self._speed_msg(5 if moving else 0)))
        self.assertTrue(self._rx(self._user_brake_msg(True)))
        self.assertTrue(self._rx(self._button_msg(enable=True)))
        self.assertFalse(self.safety.get_controls_allowed())
        self.assertFalse(self.safety.get_acc_main_on())

  def test_wrong_bus_and_address_do_not_change_cruise(self):
    for enabled in (False, True):
      for bus, address in ((1, 0xA1), (2, 0xA1), (0, 0xA2)):
        with self.subTest(enabled=enabled, bus=bus, address=address):
          self._set_mode(0)
          self.safety.set_controls_allowed(enabled)
          message = self._button_msg(enable=not enabled, cancel=enabled, bus=bus)
          message[0].addr = address
          self._rx(message)
          self.assertEqual(self.safety.get_controls_allowed(), enabled)

  def test_wrong_bus_does_not_update_pedals_or_motor_torque(self):
    for bus in (1, 2):
      for maker, getter in (
        (self._user_gas_msg, self.safety.get_gas_pressed_prev),
        (self._user_brake_msg, self.safety.get_brake_pressed_prev),
        (self._torque_meas_msg, self.safety.get_torque_meas_max),
      ):
        with self.subTest(bus=bus, message=maker.__name__):
          self._set_mode(0)
          message = maker(1)
          message[0].bus = bus
          self._rx(message)
          self.assertEqual(getter(), 0)

  def test_can_fd_whitelist_bus_address_and_length(self):
    messages = (
      (self._torque_cmd_msg(0), 64),
      (common.make_msg(0, 0x23D, 64), 64),
      (common.make_msg(2, 0x147, 64), 64),
      (self._button_msg(cancel=True, bus=2), 8),
    )
    for message, expected_length in messages:
      address, expected_bus = message[0].addr, message[0].bus
      payload = bytes(message[0].data)[:expected_length]
      for bus in range(4):
        with self.subTest(address=address, bus=bus):
          packet = common.make_msg(bus, address, dat=payload)
          self.assertEqual(self._tx(packet), bus == expected_bus)
      self.assertFalse(self._tx(common.make_msg(expected_bus, 0x7FF, dat=payload)))
      for length in (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64):
        with self.subTest(address=address, length=length):
          data = payload[:length].ljust(length, b"\x00")
          self.assertEqual(self._tx(common.make_msg(expected_bus, address, dat=data)), length == expected_length)

  def test_cross_mode_message_lengths_match_dbc(self):
    for (address, bus), length in self.TX_MSGS_LENGTHS.items():
      with self.subTest(address=address, bus=bus):
        self.assertIn([address, bus], self.TX_MSGS)
        self.assertEqual(self.packer.dbc.addr_to_msg[address].size, length)

  def test_bad_checksum_disengages(self):
    for maker in (self._speed_msg, self._pcm_status_msg):
      with self.subTest(message=maker.__name__):
        self._set_mode(0)
        message = maker(1)
        message[0].data[0] ^= 1
        self.safety.set_controls_allowed(True)
        self.assertFalse(self._rx(message))
        self.assertFalse(self.safety.get_controls_allowed())

  def test_repeated_counter_disengages(self):
    for maker in (self._speed_msg, self._pcm_status_msg):
      with self.subTest(message=maker.__name__):
        self._set_mode(0)
        message = maker(1)
        self.assertTrue(self._rx(message))
        self.safety.set_controls_allowed(True)
        for repeat in range(common.MAX_WRONG_COUNTERS):
          self.assertEqual(self._rx(message), repeat + 1 < common.MAX_WRONG_COUNTERS)
        self.assertFalse(self.safety.get_controls_allowed())

  def test_longitudinal_requires_safety_parameter(self):
    for flags in (0, 1, 2, 3):
      with self.subTest(flags=flags):
        self._set_mode(flags)
        self.safety.set_controls_allowed(True)
        allowed = bool(flags & GwmSafetyFlags.LONG_CONTROL)
        self.assertEqual(self._tx(self._longitudinal_msg()), allowed)
        for bus in (1, 2, 3):
          self.assertFalse(self._tx(self._longitudinal_msg(bus=bus)))
        for length in (0, 8, 12, 16, 32, 48):
          self.assertFalse(self._tx(common.make_msg(0, 0x143, length)))

  def test_gas_command_limits(self):
    for flags in (0, GwmSafetyFlags.LONG_CONTROL):
      self._set_mode(flags)
      for enabled in (False, True):
        self.safety.set_controls_allowed(enabled)
        for gas in range(-192, 8000):
          expected = bool(flags & GwmSafetyFlags.LONG_CONTROL) and (
            (enabled and self.MIN_GAS <= gas <= self.MAX_GAS) or gas == 0
          )
          self.assertEqual(self._tx(self._longitudinal_msg(gas=gas)), expected, (flags, enabled, gas))

  def test_brake_command_limits(self):
    for flags in (0, GwmSafetyFlags.LONG_CONTROL):
      self._set_mode(flags)
      for enabled in (False, True):
        self.safety.set_controls_allowed(enabled)
        for raw_byte in range(256):
          brake = 181 - raw_byte
          expected = bool(flags & GwmSafetyFlags.LONG_CONTROL) and ((enabled and 0 <= brake <= self.MAX_BRAKE) or brake == 0)
          self.assertEqual(self._tx(self._longitudinal_msg(brake=brake)), expected, (flags, enabled, raw_byte))

  def test_negative_brake_command_rejected(self):
    self._set_mode(GwmSafetyFlags.LONG_CONTROL)
    self.safety.set_controls_allowed(True)
    for brake in (-74, -10, -1):
      with self.subTest(brake=brake):
        self.assertFalse(self._tx(self._longitudinal_msg(brake=brake)))

  def test_longitudinal_gas_override_and_brake_disengagement(self):
    for gas_pressed in (False, True):
      self._set_mode(GwmSafetyFlags.LONG_CONTROL)
      self.safety.set_controls_allowed(True)
      self.assertTrue(self._rx(self._user_gas_msg(gas_pressed)))
      self.assertEqual(self._tx(self._longitudinal_msg(gas=1)), not gas_pressed)
      self.assertEqual(self._tx(self._longitudinal_msg(brake=1)), not gas_pressed)
      self.assertTrue(self._tx(self._longitudinal_msg()))
      self.assertTrue(self._rx(self._user_brake_msg(True)))
      self.assertFalse(self.safety.get_controls_allowed())
      self.assertFalse(self._tx(self._longitudinal_msg(gas=1)))
      self.assertFalse(self._tx(self._longitudinal_msg(brake=1)))

  def test_longitudinal_relay_check_and_forwarding(self):
    for flags in (0, GwmSafetyFlags.LONG_CONTROL):
      with self.subTest(flags=flags):
        self._set_mode(flags)
        self.assertEqual(self.safety.safety_fwd_hook(2, 0x143), -1 if flags else 0)
        self._rx(common.make_msg(0, 0x143, 64))
        self.assertEqual(self.safety.get_relay_malfunction(), bool(flags))
        if flags:
          self.assertFalse(self._tx(self._longitudinal_msg()))
          self.assertEqual(self.safety.safety_fwd_hook(0, 0x100), -1)

  def test_cancel_transmit_rejects_enable_request(self):
    for enabled in (False, True):
      for cancel in (False, True):
        with self.subTest(enabled=enabled, cancel=cancel):
          self.safety.set_controls_allowed(enabled)
          self.assertFalse(self._tx(self._button_msg(enable=True, cancel=cancel, bus=2)))
    self.assertTrue(self._tx(self._button_msg(cancel=True, bus=2)))

  def test_host_cancel_command_is_admitted(self):
    can_bus = gwmcan.CanBus(fingerprint=gen_empty_fingerprint())
    stock = {
      "STEERING_ANGLE": 10,
      "STEERING_DIRECTION": 0,
      "STEERING_RATE": 2,
      "RATE_DIRECTION": 0,
      "AP_ENABLE_COMMAND": 0,
      "AP_CANCEL_COMMAND": 0,
      "AP_DECREASE_SPEED_COMMAND": 0,
      "AP_INCREASE_SPEED_COMMAND": 0,
      "AP_REDUCE_DISTANCE_COMMAND": 0,
      "AP_INCREASE_DISTANCE_COMMAND": 0,
    }
    for flags in (0, GwmSafetyFlags.LONG_CONTROL):
      for enabled in (False, True):
        for stock_enable in (False, True):
          with self.subTest(flags=flags, enabled=enabled, stock_enable=stock_enable):
            self._set_mode(flags)
            self.safety.set_controls_allowed(enabled)
            stock["AP_ENABLE_COMMAND"] = stock_enable
            original_stock = stock.copy()
            address, payload, bus = gwmcan.create_buttons_command(self.packer, can_bus, 1, stock, cancel_command=True)
            self.assertEqual(stock, original_stock)
            self.assertEqual((address, bus), (0xA1, 2))
            self.assertFalse(payload[5] & 0x80)
            self.assertTrue(payload[5] & 0x40)
            self.assertEqual(payload[0], gwm_crcs[0xA1](payload[1:8]))
            self.assertTrue(self._tx(common.make_msg(bus, address, dat=payload)))

  def test_host_longitudinal_output_is_admitted(self):
    can_bus = gwmcan.CanBus(fingerprint=gen_empty_fingerprint())
    stock = dict.fromkeys((
      "BYPASSME_1", "SPEED_REAL", "COUNTER_BRAKE", "BYPASSME_2", "BYPASS_ACC1", "BYPASS_ACC2",
      "COUNTER_ACC", "BRAKE_OR_GAS_REQ", "STANDSTILL_1", "STANDSTILL_2", "STANDSTILL_3",
    ), 0)
    self._set_mode(GwmSafetyFlags.LONG_CONTROL)
    for active in (False, True):
      for standstill in (False, True):
        for accel in (-1.0, -0.5, -0.01, 0.0, 0.25, 0.5, 1.0):
          with self.subTest(active=active, standstill=standstill, accel=accel):
            self.safety.set_controls_allowed(active)
            address, payload, bus = gwmcan.create_longitudinal_command(self.packer, can_bus, stock, accel, active, standstill)
            self.assertEqual((address, bus), (0x143, 0))
            self.assertTrue(self._tx(common.make_msg(bus, address, dat=payload)))


class TestGwmMgReleaseSafety(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    cls.safety = libsafety_py.ffi.dlopen(libsafety_py._build_libsafety(release=True))

  def test_experimental_safety_modes_are_not_release_enabled(self):
    for mode in (CarParams.SafetyModel.gwm, CarParams.SafetyModel.mg):
      for flags in (0, GwmSafetyFlags.LONG_CONTROL):
        with self.subTest(mode=mode, flags=flags):
          self.assertNotEqual(self.safety.set_safety_hooks(mode, flags), 0)


if __name__ == "__main__":
  unittest.main()
