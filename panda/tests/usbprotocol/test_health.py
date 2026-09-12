import struct
import unittest

from panda import Panda


class FakeUsbHandle:
  def __init__(self):
    self.payload = b""
    self.reads = []
    self.writes = []

  def controlRead(self, request_type, request, value, index, length):
    self.reads.append((request_type, request, value, index, length))
    return self.payload

  def controlWrite(self, request_type, request, value, index, data):
    self.writes.append((request_type, request, value, index, data))


class TestPandaHealth(unittest.TestCase):
  WIRE_STRUCT = struct.Struct("<IHHIIIIIHBBHBHBBHHHHBB")
  FLAG_FIELDS = (
    "ignition_line",
    "ignition_can",
    "controls_allowed",
    "power_save_enabled",
    "heartbeat_lost",
    "safety_rx_checks_invalid",
    "som_reset_triggered",
  )

  def setUp(self):
    self.panda = Panda.__new__(Panda)
    self.panda._handle_open = False
    self.panda._handle = FakeUsbHandle()
    self.panda.health_version = Panda.HEALTH_PACKET_VERSION

  def _packet(self, **overrides):
    fields = {
      "uptime": 0x12345678,
      "voltage": 12050,
      "current": 3456,
      "safety_tx_blocked": 0x11223344,
      "safety_rx_invalid": 0x22334455,
      "tx_buffer_overflow": 0x33445566,
      "rx_buffer_overflow": 0x44556677,
      "faults": 0x55667788,
      "flags": 0,
      "car_harness_status": 2,
      "safety_mode": 20,
      "safety_param": 0x1357,
      "fault_status": 3,
      "alternative_experience": 0x2468,
      "interrupt_load": 127,
      "fan_power": 93,
      "spi_error_count": 0x3579,
      "sbu1_voltage_mV": 1234,
      "sbu2_voltage_mV": 2345,
      "sound_output_level": 0x468A,
      "temperature": 65,
      "controls_allowed_sp": 0,
    }
    fields.update(overrides)
    return self.WIRE_STRUCT.pack(*fields.values())

  def test_wire_layout_and_scalar_fields(self):
    self.assertEqual(self.WIRE_STRUCT.size, 49)
    self.assertEqual(Panda.HEALTH_STRUCT.format, self.WIRE_STRUCT.format)
    self.panda._handle.payload = self._packet()
    health = self.panda.health()
    self.assertEqual(self.panda._handle.reads, [(Panda.REQUEST_IN, 0xD2, 0, 0, 49)])
    expected = {
      "uptime": 0x12345678,
      "voltage": 12050,
      "current": 3456,
      "safety_tx_blocked": 0x11223344,
      "safety_rx_invalid": 0x22334455,
      "tx_buffer_overflow": 0x33445566,
      "rx_buffer_overflow": 0x44556677,
      "faults": 0x55667788,
      "car_harness_status": 2,
      "safety_mode": 20,
      "safety_param": 0x1357,
      "fault_status": 3,
      "alternative_experience": 0x2468,
      "interrupt_load": 127 / 255,
      "fan_power": 93,
      "spi_error_count": 0x3579,
      "sbu1_voltage_mV": 1234,
      "sbu2_voltage_mV": 2345,
      "sound_output_level": 0x468A,
      "temperature": 25.0,
      "controls_allowed_lateral": 0,
      "controls_allowed_longitudinal": 0,
    }
    expected.update(dict.fromkeys(self.FLAG_FIELDS, False))
    self.assertEqual(health, expected)

  def test_flag_bits_are_independent(self):
    for flags in range(128):
      with self.subTest(flags=flags):
        self.panda._handle.payload = self._packet(flags=flags | 0xFF80)
        health = self.panda.health()
        for bit_index, field in enumerate(self.FLAG_FIELDS):
          self.assertIs(health[field], bool(flags & (1 << bit_index)))
        self.assertEqual(health["controls_allowed_lateral"], 0)
        self.assertEqual(health["controls_allowed_longitudinal"], 0)

  def test_fixed_point_boundaries(self):
    for encoded in (0, 1, 40, 127, 128, 254, 255):
      with self.subTest(encoded=encoded):
        self.panda._handle.payload = self._packet(interrupt_load=encoded, temperature=encoded)
        health = self.panda.health()
        self.assertAlmostEqual(health["interrupt_load"], encoded / 255.0)
        self.assertEqual(health["temperature"], encoded - 40.0)

  def test_trailing_sp_control_bits(self):
    for control_bits in range(256):
      with self.subTest(control_bits=control_bits):
        self.panda._handle.payload = self._packet(controls_allowed_sp=control_bits)
        health = self.panda.health()
        self.assertEqual(health["controls_allowed_lateral"], control_bits & 1)
        self.assertEqual(health["controls_allowed_longitudinal"], (control_bits >> 1) & 1)
        self.assertFalse(health["controls_allowed"])
        self.assertFalse(health["ignition_can"])

  def test_version_mismatch_rejected_before_usb_read(self):
    self.panda.health_version = Panda.HEALTH_PACKET_VERSION ^ 1
    with self.assertRaisesRegex(RuntimeError, "health packet version mismatch"):
      self.panda.health()
    with self.assertRaisesRegex(RuntimeError, "health packet version mismatch"):
      self.panda.can_health(0)
    self.assertEqual(self.panda._handle.reads, [])

  def test_invalid_packet_lengths_rejected(self):
    for length in (0, 1, 48, 50, 96):
      with self.subTest(length=length):
        self.panda._handle.payload = bytes(length)
        with self.assertRaises(struct.error):
          self.panda.health()

  def test_alternative_experience_sp_parameter(self):
    self.panda.set_alternative_experience(8)
    self.panda.set_alternative_experience("16", safety_param_sp="4660")
    self.assertEqual(self.panda._handle.writes, [
      (Panda.REQUEST_OUT, 0xDF, 8, 0, b""),
      (Panda.REQUEST_OUT, 0xDF, 16, 0x1234, b""),
    ])

  def test_heartbeat_mads_defaults(self):
    self.panda.send_heartbeat()
    self.panda.send_heartbeat(False)
    self.assertEqual(self.panda._handle.writes, [
      (Panda.REQUEST_OUT, 0xF3, True, True, b""),
      (Panda.REQUEST_OUT, 0xF3, False, True, b""),
    ])

  def test_heartbeat_mads_parameter(self):
    for engaged in (False, True):
      for engaged_mads in (False, True):
        with self.subTest(engaged=engaged, engaged_mads=engaged_mads):
          self.panda.send_heartbeat(engaged, engaged_mads=engaged_mads)
          self.assertEqual(self.panda._handle.writes[-1],
                           (Panda.REQUEST_OUT, 0xF3, engaged, engaged_mads, b""))


if __name__ == "__main__":
  unittest.main()
