import ctypes
from functools import cached_property
import struct

from tinygrad.device import Device

import openpilot.cereal.messaging as messaging
from openpilot.cereal.messaging import PubMaster
from openpilot.common.hardware.usb import CHESTNUT_USB_IDS
from openpilot.common.swaglog import cloudlog


class ChestnutState:
  # Shared by the model runners; only modeld can access chestnut.
  def __init__(self, pm: PubMaster, big: bool):
    self.pm = pm
    self.big = big
    self.valid = True
    self.sends = 0
    self.metrics = {}
    self._asm_usb = None

  def _close_asm_usb(self) -> None:
    if self._asm_usb is not None:
      self._asm_usb.close()
      self._asm_usb = None

  def _open_asm_usb(self):
    # The ordinary QCOM model never constructs ChestnutState. Keep libusb out
    # of its import path while retaining both supported chestnut USB IDs.
    import usb1

    context = usb1.USBContext()
    for vendor_id, product_id in CHESTNUT_USB_IDS:
      if (handle := context.openByVendorIDAndProductID(vendor_id, product_id, skip_on_error=True)) is not None:
        return handle
    context.close()

  def _read_ina(self) -> tuple[int, int, bool]:
    import usb1

    if "AMD" in Device._opened_devices and self._asm_usb is None:
      try:
        raw = Device["AMD"].iface.pci_dev.usb.usb.control_read(0xC0, 5)
        return struct.unpack('<Hh?', bytes(raw))
      except Exception:
        pass
    if self._asm_usb is None:
      self._asm_usb = self._open_asm_usb()
    if self._asm_usb is None:
      raise usb1.USBErrorNoDevice
    try:
      raw = self._asm_usb.controlRead(0xC0, 0xC0, 0, 0, 5, timeout=100)
    except usb1.USBError:
      self._close_asm_usb()
      raise
    return struct.unpack('<Hh?', bytes(raw))

  @cached_property
  def power_limit(self) -> int:
    smu = Device["AMD"].iface.dev_impl.smu
    return smu._send_msg(smu.smu_mod.PPSMC_MSG_GetPptLimit, 0, read_back_arg=True, timeout=100)

  def send(self) -> None:
    msg = messaging.new_message('chestnutState')
    state = msg.chestnutState
    self.sends += 1
    if self.big and "AMD" in Device._opened_devices and self.sends % 100 == 1:
      try:
        smu = Device["AMD"].iface.dev_impl.smu
        metrics_t = smu.smu_mod.SmuMetricsExternal_t
        smu._send_msg(smu.smu_mod.PPSMC_MSG_TransferTableSmu2Dram, smu.smu_mod.TABLE_SMU_METRICS, timeout=100)
        metrics_buf = bytearray(smu.adev.vram.view(smu.driver_table_paddr, ctypes.sizeof(metrics_t))[:])
        metrics = metrics_t.from_buffer(metrics_buf).SmuMetrics
        self.metrics = {'tempC': metrics.AvgTemperature[smu.smu_mod.TEMP_HOTSPOT],
                        'memoryTempC': metrics.AvgTemperature[smu.smu_mod.TEMP_MEM],
                        'powerDrawW': metrics.AverageSocketPower,
                        'powerLimitW': self.power_limit,
                        'gpuUsagePercent': metrics.AverageGfxActivity,
                        'gpuClockMhz': metrics.AverageGfxclkFrequencyPostDs,
                        'fanSpeedRpm': metrics.AvgFanRpm}
        self.valid = True
      except Exception:
        if self.valid:
          cloudlog.exception("chestnut state read failed")
        self.valid = False
        self.metrics.clear()
    if self.big:
      for k, v in self.metrics.items():
        setattr(state, k, v)

    asm_valid = False
    try:
      # ASM runs on USB-C power, these still read without a gpu.
      state.supplyVoltage, state.supplyCurrent, state.supplyFault = self._read_ina()
      asm_valid = True
    except Exception:
      pass
    if "AMD" in Device._opened_devices:
      try:
        state.pcieLtssm = Device["AMD"].iface.pci_dev.usb.read(0xB450, 1)[0]
      except Exception:
        pass

    msg.valid = asm_valid and (not self.big or self.valid)
    self.pm.send('chestnutState', msg)
