import unittest
from types import SimpleNamespace

from opendbc.car.structs import CarParams
from panda import DLC_TO_LEN, pack_can_buffer
from panda.python.spi import PandaSpiHandle, XFER_SIZE
from panda.tests.libpanda import libpanda_py


ffi = libpanda_py.ffi
lpp = libpanda_py.libpanda
QUEUES = (lpp.tx1_q, lpp.tx2_q, lpp.tx3_q)


def drain(queue):
  messages = []
  packet = ffi.new('CANPacket_t *')
  while lpp.can_pop(queue, packet):
    length = DLC_TO_LEN[packet[0].data_len_code]
    messages.append((packet[0].addr, bytes(packet[0].data[0:length]), packet[0].bus))
  return messages


class TestSpiCanFlowControl(unittest.TestCase):
  def setUp(self):
    self.assertEqual(ffi.sizeof('CANPacket_t'), lpp.get_can_packet_size())
    lpp.set_safety_hooks(CarParams.SafetyModel.allOutput, 0)
    lpp.comms_can_reset()
    for queue in (*QUEUES, lpp.rx_q):
      drain(queue)

  def tearDown(self):
    for queue in (*QUEUES, lpp.rx_q):
      drain(queue)
    lpp.comms_can_reset()

  @staticmethod
  def packed(messages, fd=False):
    return bytes(pack_can_buffer(messages, fd=fd)[0])

  def fill(self, bus, free=0):
    packet = libpanda_py.make_CANPacket(0x123, bus, b'blocked')
    while lpp.can_slots_empty(QUEUES[bus]) > free:
      self.assertTrue(lpp.can_push(QUEUES[bus], packet))

  @staticmethod
  def snapshot():
    queues = []
    for queue in (*QUEUES, lpp.rx_q):
      queues.append((queue.w_ptr, queue.r_ptr, bytes(ffi.buffer(queue.elems, queue.fifo_size * ffi.sizeof('CANPacket_t')))))
    return queues, lpp.safety_tx_blocked, lpp.tx_buffer_overflow, lpp.controls_allowed

  def test_unrelated_stalled_bus_does_not_block(self):
    for stalled in range(3):
      for target in range(3):
        if target == stalled:
          continue
        with self.subTest(stalled=stalled, target=target):
          self.setUp()
          self.fill(stalled)
          before = self.snapshot()[0][stalled]
          # This is the old SPI admission gate, executed in the production C.
          self.assertFalse(lpp.can_tx_check_min_slots_free(170))
          message = (0x321, b'healthy', target)
          data = self.packed([message])
          self.assertTrue(lpp.comms_can_write_spi(data, len(data)))
          self.assertEqual(drain(QUEUES[target]), [message])
          self.assertEqual(self.snapshot()[0][stalled], before)

  def test_incident_threshold_with_real_honda_inactive_command(self):
    lpp.set_safety_hooks(CarParams.SafetyModel.hondaBosch, 2)
    # 246 pending forwarded packets leave only 169/415 slots on bus2.
    self.fill(2, free=169)
    self.assertFalse(lpp.can_tx_check_min_slots_free(170))
    message = (0x1DF, bytes.fromhex('8ad000000000000c'), 1)
    data = self.packed([message])
    self.assertTrue(lpp.comms_can_write_spi(data, len(data)))
    self.assertEqual(drain(QUEUES[1]), [message])

  def test_destination_capacity_is_exact_and_nack_is_atomic(self):
    messages = [(0x100, b'one', 0), (0x101, b'two', 1), (0x102, b'three', 1)]
    data = self.packed(messages)
    self.fill(1, free=1)
    before = self.snapshot()
    self.assertFalse(lpp.comms_can_write_spi(data, len(data)))
    self.assertEqual(self.snapshot(), before)
    packet = ffi.new('CANPacket_t *')
    self.assertTrue(lpp.can_pop(QUEUES[1], packet))
    self.assertTrue(lpp.comms_can_write_spi(data, len(data)))
    self.assertEqual(drain(QUEUES[0]), messages[:1])
    self.assertEqual(drain(QUEUES[1])[-2:], messages[1:])

  def test_partial_packet_nack_retry_all_dlc_and_byte_boundaries(self):
    for length in DLC_TO_LEN:
      message = (0x18DAB0F1, bytes(range(length)), 1)
      data = self.packed([message], fd=length > 8)
      for split in range(1, len(data)):
        with self.subTest(length=length, split=split):
          self.setUp()
          self.fill(1)
          self.assertTrue(lpp.comms_can_write_spi(data[:split], split))
          before = self.snapshot()
          for _ in range(2):
            self.assertFalse(lpp.comms_can_write_spi(data[split:], len(data)-split))
            self.assertEqual(self.snapshot(), before)
          packet = ffi.new('CANPacket_t *')
          self.assertTrue(lpp.can_pop(QUEUES[1], packet))
          self.assertTrue(lpp.comms_can_write_spi(data[split:], len(data)-split))
          received = drain(QUEUES[1])
          self.assertEqual(len(received), 415)
          self.assertEqual(received[-1], message)
          self.assertEqual(received.count(message), 1)

  def test_fragment_can_remain_incomplete_across_multiple_chunks(self):
    message = (0x456, bytes(range(64)), 2)
    data = self.packed([message], fd=True)
    self.fill(2)
    for value in data[:-1]:
      self.assertTrue(lpp.comms_can_write_spi(bytes([value]), 1))
    before = self.snapshot()
    self.assertFalse(lpp.comms_can_write_spi(data[-1:], 1))
    self.assertEqual(self.snapshot(), before)
    drain(QUEUES[2])
    self.assertTrue(lpp.comms_can_write_spi(data[-1:], 1))
    self.assertEqual(drain(QUEUES[2]), [message])

  def test_completed_prefix_and_new_partial_packet(self):
    first = (0x100, b'first', 0)
    second = (0x101, b'second', 2)
    data_first, data_second = self.packed([first]), self.packed([second])
    self.fill(2)
    self.assertTrue(lpp.comms_can_write_spi(data_first[:2], 2))
    middle = data_first[2:] + data_second[:3]
    self.assertTrue(lpp.comms_can_write_spi(middle, len(middle)))
    self.assertEqual(drain(QUEUES[0]), [first])
    before = self.snapshot()
    self.assertFalse(lpp.comms_can_write_spi(data_second[3:], len(data_second)-3))
    self.assertEqual(self.snapshot(), before)
    drain(QUEUES[2])
    self.assertTrue(lpp.comms_can_write_spi(data_second[3:], len(data_second)-3))
    self.assertEqual(drain(QUEUES[2]), [second])

  def test_reset_discards_only_incomplete_assembly(self):
    old = self.packed([(0x100, b'old', 1)])
    self.assertTrue(lpp.comms_can_write_spi(old[:3], 3))
    lpp.comms_can_reset()
    fresh = (0x101, b'new', 1)
    data = self.packed([fresh])
    self.assertTrue(lpp.comms_can_write_spi(data, len(data)))
    self.assertEqual(drain(QUEUES[1]), [fresh])

  def test_invalid_destination_keeps_existing_safety_rejection(self):
    lpp.set_safety_hooks(CarParams.SafetyModel.hondaBosch, 2)
    self.fill(2)
    data = self.packed([(0x1DF, b'\0' * 8, 7)])
    before = lpp.safety_tx_blocked
    self.assertTrue(lpp.comms_can_write_spi(data, len(data)))
    self.assertEqual(lpp.safety_tx_blocked, before + 1)
    packet = ffi.new('CANPacket_t *')
    self.assertTrue(lpp.can_pop(lpp.rx_q, packet))
    self.assertTrue(packet[0].rejected)

  def test_honda_active_command_still_rejected_without_permission(self):
    lpp.set_safety_hooks(CarParams.SafetyModel.hondaBosch, 2)
    self.fill(2)
    self.assertFalse(lpp.controls_allowed)
    data = self.packed([(0x1DF, b'\0' * 8, 1)])  # gas 0 is active, not Honda's -30000 inactive value
    before = lpp.safety_tx_blocked
    self.assertTrue(lpp.comms_can_write_spi(data, len(data)))
    self.assertEqual(lpp.safety_tx_blocked, before + 1)
    self.assertEqual(drain(QUEUES[1]), [])
    packet = ffi.new('CANPacket_t *')
    self.assertTrue(lpp.can_pop(lpp.rx_q, packet))
    self.assertTrue(packet[0].rejected)

  def test_large_python_spi_batch_is_chunked_and_progresses(self):
    messages = [(0x100 + i, b'', 1) for i in range(672)]
    data = self.packed(messages)
    self.assertEqual(len(data), XFER_SIZE)
    before = self.snapshot()
    self.assertFalse(lpp.comms_can_write_spi(data, len(data)))
    self.assertEqual(self.snapshot(), before)
    received, chunks = [], []

    def transfer(endpoint, chunk, timeout):
      self.assertEqual(endpoint, 3)
      chunk = bytes(chunk)
      chunks.append(chunk)
      self.assertTrue(lpp.comms_can_write_spi(chunk, len(chunk)))
      received.extend(drain(QUEUES[1]))

    PandaSpiHandle.bulkWrite(SimpleNamespace(_transfer=transfer), 3, data)
    self.assertLessEqual(max(map(len, chunks)), 256)
    self.assertEqual(b''.join(chunks), data)
    self.assertEqual(received, messages)

  def test_non_can_spi_bulk_write_keeps_transfer_size(self):
    data = b'\x55' * (XFER_SIZE + 7)
    chunks = []
    handle = SimpleNamespace(_transfer=lambda endpoint, chunk, timeout: chunks.append(bytes(chunk)))
    PandaSpiHandle.bulkWrite(handle, 2, data)
    self.assertEqual(list(map(len, chunks)), [XFER_SIZE, 7])
    self.assertEqual(b''.join(chunks), data)
