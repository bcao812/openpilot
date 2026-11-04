import ctypes
import struct
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from openpilot.selfdrive.modeld import chestnut_state


class USBError(Exception):
  pass


class USBErrorNoDevice(USBError):
  pass


class TestChestnutState(unittest.TestCase):
  def setUp(self):
    self.pm = mock.Mock()
    self.state = chestnut_state.ChestnutState(self.pm, big=False)
    self.device = mock.MagicMock(_opened_devices=set())
    self.enterContext(mock.patch.object(chestnut_state, "Device", self.device))
    self.usb_context = mock.Mock()
    self.usb = SimpleNamespace(USBContext=mock.Mock(return_value=self.usb_context),
                               USBError=USBError, USBErrorNoDevice=USBErrorNoDevice)
    self.enterContext(mock.patch.dict(sys.modules, {"usb1": self.usb}))

  def last_message(self):
    service, message = self.pm.send.call_args.args
    self.assertEqual(service, "chestnutState")
    return message

  def test_usb_fallback_tries_both_ids_and_keeps_a_working_handle(self):
    handle = mock.Mock()
    handle.controlRead.return_value = struct.pack('<Hh?', 5100, -212, True)
    self.usb_context.openByVendorIDAndProductID.side_effect = [None, handle]
    self.assertEqual(self.state._read_ina(), (5100, -212, True))
    self.assertEqual(self.state._read_ina(), (5100, -212, True))
    self.assertEqual(self.usb_context.openByVendorIDAndProductID.call_args_list, [
      mock.call(vendor, product, skip_on_error=True) for vendor, product in chestnut_state.CHESTNUT_USB_IDS
    ])
    self.usb.USBContext.assert_called_once()
    self.usb_context.close.assert_not_called()

  def test_missing_usb_closes_the_search_context(self):
    self.usb_context.openByVendorIDAndProductID.return_value = None
    with self.assertRaises(USBErrorNoDevice):
      self.state._read_ina()
    self.usb_context.close.assert_called_once()
    self.assertIsNone(self.state._asm_usb)

  def test_existing_amd_interface_does_not_reopen_usb(self):
    self.device._opened_devices = {'AMD'}
    control_read = self.device.__getitem__.return_value.iface.pci_dev.usb.usb.control_read
    control_read.return_value = struct.pack('<Hh?', 5050, 200, False)
    self.assertEqual(self.state._read_ina(), (5050, 200, False))
    control_read.assert_called_once_with(0xC0, 5)
    self.usb.USBContext.assert_not_called()

  def test_failed_usb_read_discards_the_handle_and_reconnects(self):
    first, replacement = mock.Mock(), mock.Mock()
    first.controlRead.side_effect = USBError('disconnected')
    replacement.controlRead.return_value = struct.pack('<Hh?', 5000, 250, False)
    self.state._asm_usb = first
    self.usb_context.openByVendorIDAndProductID.return_value = replacement
    with self.assertRaises(USBError):
      self.state._read_ina()
    first.close.assert_called_once()
    self.assertIsNone(self.state._asm_usb)
    self.assertEqual(self.state._read_ina(), (5000, 250, False))
    self.assertIs(self.state._asm_usb, replacement)

  def test_missing_supply_data_cannot_publish_a_valid_state(self):
    with mock.patch.object(self.state, '_read_ina', side_effect=USBErrorNoDevice):
      self.state.send()
    self.assertFalse(self.last_message().valid)

  def test_big_model_validity_is_separate_from_supply_validity(self):
    with mock.patch.object(self.state, '_read_ina', return_value=(5100, 212, False)):
      for big in (False, True):
        for metrics_valid in (False, True):
          with self.subTest(big=big, metrics_valid=metrics_valid):
            self.state.big, self.state.valid = big, metrics_valid
            self.state.send()
            message = self.last_message()
            self.assertEqual(message.valid, not big or metrics_valid)
            self.assertEqual(message.chestnutState.supplyVoltage, 5100)
            self.assertEqual(message.chestnutState.supplyCurrent, 212)
            self.assertFalse(message.chestnutState.supplyFault)

  def test_amd_metrics_cadence_and_failure_recovery(self):
    class Metrics(ctypes.Structure):
      _fields_ = [('AvgTemperature', ctypes.c_uint16 * 2), ('AverageSocketPower', ctypes.c_uint16),
                 ('AverageGfxActivity', ctypes.c_uint16), ('AverageGfxclkFrequencyPostDs', ctypes.c_uint16),
                 ('AvgFanRpm', ctypes.c_uint16)]

    class ExternalMetrics(ctypes.Structure):
      _fields_ = [('SmuMetrics', Metrics)]

    metrics = ExternalMetrics(Metrics((65, 70), 120, 80, 900, 1400))
    smu = mock.Mock()
    smu.smu_mod = SimpleNamespace(SmuMetricsExternal_t=ExternalMetrics, TEMP_HOTSPOT=0, TEMP_MEM=1,
                                 PPSMC_MSG_TransferTableSmu2Dram=1, TABLE_SMU_METRICS=2, PPSMC_MSG_GetPptLimit=3)
    smu.adev.vram.view.return_value = bytes(metrics)
    smu._send_msg.return_value = 150
    self.device._opened_devices = {'AMD'}
    self.device.__getitem__.return_value.iface.dev_impl.smu = smu
    self.device.__getitem__.return_value.iface.pci_dev.usb.read.return_value = bytes([0x78])
    self.state.big = True

    with mock.patch.object(self.state, '_read_ina', return_value=(5100, 212, False)):
      for _ in range(100):
        self.state.send()
      self.assertEqual(smu.adev.vram.view.call_count, 1)
      message = self.last_message()
      self.assertTrue(message.valid)
      self.assertEqual(message.chestnutState.tempC, 65)
      self.assertEqual(message.chestnutState.memoryTempC, 70)
      self.assertEqual(message.chestnutState.powerDrawW, 120)
      self.assertEqual(message.chestnutState.powerLimitW, 150)
      self.assertEqual(message.chestnutState.pcieLtssm, 0x78)

      smu.adev.vram.view.side_effect = RuntimeError('metrics unavailable')
      with mock.patch.object(chestnut_state.cloudlog, 'exception') as report:
        self.state.send()
      report.assert_called_once()
      self.assertFalse(self.last_message().valid)
      self.assertEqual(self.state.metrics, {})

      smu.adev.vram.view.side_effect = None
      for _ in range(100):
        self.state.send()
      self.assertTrue(self.last_message().valid)
      self.assertEqual(self.last_message().chestnutState.tempC, 65)
      self.assertEqual(smu.adev.vram.view.call_count, 3)
      power_reads = [call for call in smu._send_msg.call_args_list if call.args[0] == 3]
      self.assertEqual(len(power_reads), 1)
