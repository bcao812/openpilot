import unittest

from opendbc.car.crc import CRC8J1850, mk_crc8_fun
from opendbc.car.structs import CarParams
from opendbc.safety.tests import common
from opendbc.safety.tests.common import CANPackerSafety
from opendbc.safety.tests.libsafety import libsafety_py


mg_crc = mk_crc8_fun(CRC8J1850, init_crc=0, xor_out=0xFF)


def checksum(message):
  address, payload, bus = message
  data = bytearray(payload)
  data[7] = mg_crc(data[:7])
  return address, data, bus


class TestMgSafety(common.CarSafetyTest, common.DriverTorqueSteeringSafetyTest):
  TX_MSGS = [[0x1FD, 0]]
  RELAY_MALFUNCTION_ADDRS = {0: (0x1FD,)}
  FWD_BLACKLISTED_ADDRS = {2: [0x1FD]}
  MAX_TORQUE_LOOKUP = [0], [300]
  MAX_RATE_UP = 6
  MAX_RATE_DOWN = 10
  MAX_RT_DELTA = 125
  DRIVER_TORQUE_ALLOWANCE = 100
  DRIVER_TORQUE_FACTOR = 2

  def setUp(self):
    self.packer = CANPackerSafety("mg")
    self.safety = libsafety_py.libsafety
    self.assertEqual(self.safety.set_safety_hooks(CarParams.SafetyModel.mg, 0), 0)
    self.safety.init_tests()
    self.counters = {}

  def _counter(self, name):
    self.counters[name] = (self.counters.get(name, 0) + 1) % 16
    return self.counters[name]

  def _speed_msg(self, speed):
    values = {
      "VehSpdAvgHSC2": speed * 3.6,
      "VehSpdAvgAlvRCHSC2": self._counter("speed"),
    }
    return self.packer.make_can_msg_safety("SCS_HSC2_FrP19", 0, values)

  def _user_gas_msg(self, gas):
    return self.packer.make_can_msg_safety("GW_HSC2_HCU_FrP00", 0, {"EPTAccelActuPosHSC2": gas})

  def _user_brake_msg(self, brake):
    values = {
      "BrkPdlAppdHSC2": bool(brake),
      "BrkPdlAppdRCHSC2": self._counter("brake"),
    }
    return self.packer.make_can_msg_safety("EHBS_HSC2_FrP00", 0, values, fix_checksum=checksum)

  def _cruise_state_msg(self, state, bus=0):
    values = {
      "ACCSysSts_RadarHSC2": state,
      "ACCSysAlvRlngCtr_SCSHSC2": self._counter("cruise"),
    }
    return self.packer.make_can_msg_safety("RADAR_HSC2_FrP00", bus, values, fix_checksum=checksum)

  def _pcm_status_msg(self, enable):
    return self._cruise_state_msg(2 if enable else 0)

  def _torque_driver_msg(self, torque):
    values = {
      "DrvrStrgDlvrdToqHSC2": torque * 0.01,
      "ChLKAAlvRCHSC2": self._counter("driver_torque"),
    }
    return self.packer.make_can_msg_safety("EPS_HSC2_FrP03", 0, values)

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"LKAReqToqHSC2": torque, "LKAReqToqStsHSC2": steer_req}
    return self.packer.make_can_msg_safety("FVCM_HSC2_FrP03", 0, values)

  def test_cruise_states(self):
    for state in range(8):
      with self.subTest(state=state):
        self.assertTrue(self._rx(self._cruise_state_msg(0)))
        self.assertFalse(self.safety.get_controls_allowed())
        self.assertTrue(self._rx(self._cruise_state_msg(state)))
        self.assertEqual(self.safety.get_controls_allowed(), state in (2, 3))

  def test_cancel_disengages(self):
    self.assertTrue(self._rx(self._cruise_state_msg(2)))
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertTrue(self._rx(self._cruise_state_msg(0)))
    self.assertFalse(self.safety.get_controls_allowed())
    self.assertFalse(self._tx(self._torque_cmd_msg(1)))

  def test_wrong_bus_and_address_do_not_enable(self):
    for bus, address in ((1, 0x242), (2, 0x242), (0, 0x243)):
      with self.subTest(bus=bus, address=address):
        self._rx(self._cruise_state_msg(0))
        message = self._cruise_state_msg(2, bus=bus)
        message[0].addr = address
        self._rx(message)
        self.assertFalse(self.safety.get_controls_allowed())

  def test_wrong_bus_does_not_update_pedals_or_driver_torque(self):
    for bus in (1, 2):
      for maker, getter in (
        (self._user_gas_msg, self.safety.get_gas_pressed_prev),
        (self._user_brake_msg, self.safety.get_brake_pressed_prev),
        (self._torque_driver_msg, self.safety.get_torque_driver_max),
      ):
        with self.subTest(bus=bus, message=maker.__name__):
          message = maker(1)
          message[0].bus = bus
          self._rx(message)
          self.assertEqual(getter(), 0)

  def test_steer_tx_requires_bus_address_and_length(self):
    self.assertTrue(self._tx(self._torque_cmd_msg(0)))
    for bus in (1, 2, 3):
      message = self._torque_cmd_msg(0)
      message[0].bus = bus
      self.assertFalse(self._tx(message))
    message = self._torque_cmd_msg(0)
    message[0].addr = 0x1FE
    self.assertFalse(self._tx(message))
    for length in (0, 1, 2, 3, 4, 5, 6, 7, 12, 16, 20, 24, 32, 48, 64):
      with self.subTest(length=length):
        self.assertFalse(self._tx(common.make_msg(0, 0x1FD, length)))

  def test_bad_checksum_disengages(self):
    for maker in (self._pcm_status_msg, self._user_brake_msg):
      with self.subTest(message=maker.__name__):
        self._reset_safety_hooks()
        self.counters.clear()
        message = maker(True)
        message[0].data[7] ^= 1
        self.safety.set_controls_allowed(True)
        self.assertFalse(self._rx(message))
        self.assertFalse(self.safety.get_controls_allowed())

  def test_repeated_counter_disengages(self):
    for maker in (self._speed_msg, self._torque_driver_msg, self._pcm_status_msg, self._user_brake_msg):
      with self.subTest(message=maker.__name__):
        self._reset_safety_hooks()
        self.counters.clear()
        message = maker(1)
        self.assertTrue(self._rx(message))
        self.safety.set_controls_allowed(True)
        for repeat in range(common.MAX_WRONG_COUNTERS):
          self.assertEqual(self._rx(message), repeat + 1 < common.MAX_WRONG_COUNTERS)
        self.assertFalse(self.safety.get_controls_allowed())


if __name__ == "__main__":
  unittest.main()
