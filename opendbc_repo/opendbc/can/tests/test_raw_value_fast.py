"""Known-bit and parser-state contracts for complete and partial CAN fields."""
from dataclasses import fields

import pytest

from opendbc.can.dbc import Signal
from opendbc.can.parser import MessageState, get_raw_value


def signal_at(start, size, little_endian, signed=False):
  def dbc_bit(index):
    return (index // 8) * 8 + 7 - index % 8

  lsb = start if little_endian else dbc_bit(start + size - 1)
  msb = start + size - 1 if little_endian else dbc_bit(start)
  return Signal('VALUE', start if little_endian else msb, msb, lsb, size,
                signed, 1.0, 0.0, little_endian)


def insert_bits(start, size, little_endian, raw, length=16):
  """Place individual known bits, without using the decoder's integer slices."""
  payload = bytearray([0xA5] * length)
  for index in range(size):
    position = start + index
    shift = position % 8 if little_endian else 7 - position % 8
    value_shift = index if little_endian else size - index - 1
    payload[position // 8] &= ~(1 << shift)
    payload[position // 8] |= ((raw >> value_shift) & 1) << shift
  return payload


@pytest.mark.parametrize('little_endian', [False, True])
@pytest.mark.parametrize('size', range(1, 65))
def test_complete_fields_recover_known_bits_at_every_byte_offset(size, little_endian):
  mask = (1 << size) - 1
  for start in (*range(16), 31, 63):
    signal = signal_at(start, size, little_endian)
    for raw in (0, 1, mask, mask >> 1, 0xA53CA53C55AA55AA & mask):
      payload = insert_bits(start, size, little_endian, raw)
      assert get_raw_value(payload, signal) == raw
      assert get_raw_value(bytes(payload), signal) == raw


@pytest.mark.parametrize('little_endian', [False, True])
@pytest.mark.parametrize('size', [1, 7, 8, 9, 15, 16, 31, 32, 63, 64])
def test_signed_fields_keep_parser_scale_offset_and_all_samples(size, little_endian):
  signal = signal_at(5, size, little_endian, signed=True)
  signal.factor, signal.offset = 0.25, -3.0
  state = MessageState(1, 'TEST', 16, [signal], ignore_checksum=True, ignore_counter=True)
  expected = []
  for index, value in enumerate((0, (1 << (size - 1)) - 1, -(1 << (size - 1)), -1), 1):
    payload = insert_bits(5, size, little_endian, value & ((1 << size) - 1))
    assert state.parse(index * 10_000_000, bytes(payload))
    expected.append(value * 0.25 - 3.0)
    assert state.vals == [expected[-1]]
    assert state.all_vals == [expected]
    assert list(state.timestamps) == [frame * 10_000_000 for frame in range(1, index + 1)]


@pytest.mark.parametrize('payload,signal,expected', [
  (b'', signal_at(0, 16, True), 0),
  (b'', signal_at(0, 16, False), 0),
  (b'\xab', signal_at(0, 16, True), 0),
  (b'\xab', signal_at(0, 16, False), 0xAB00),
  (b'\xab\xcd', signal_at(4, 16, True), 0),
  (b'\xab\xcd', signal_at(4, 16, False), 0xBCD0),
  (b'\xab', signal_at(16, 8, False), 0),
  (b'\xab', Signal('VALUE', 0, -1, 0, 0, False, 1.0, 0.0, True), 0),
  (b'\xab', Signal('VALUE', 0, 7, 0, -1, False, 1.0, 0.0, True), 0),
  (b'\xab', Signal('VALUE', 0, 7, 16, 8, False, 1.0, 0.0, True), 0xAB),
])
def test_partial_or_unusual_ranges_preserve_original_contract(payload, signal, expected):
  assert get_raw_value(payload, signal) == expected
  assert get_raw_value(bytearray(payload), signal) == expected


def test_malformed_size_still_raises_at_the_negative_shift():
  signal = Signal('VALUE', 0, 7, 0, 4, False, 1.0, 0.0, True)
  with pytest.raises(ValueError, match='negative shift count'):
    get_raw_value(b'\xab', signal)


def test_mutating_shared_signal_metadata_takes_effect_on_the_next_decode():
  shared = signal_at(0, 1, True)
  for start, size, little_endian in ((1, 3, True), (7, 17, False), (9, 12, True), (31, 64, False)):
    current = signal_at(start, size, little_endian)
    for field in fields(Signal):
      setattr(shared, field.name, getattr(current, field.name))
    raw = 0x123456789ABCDEF & ((1 << size) - 1)
    assert get_raw_value(insert_bits(start, size, little_endian, raw), shared) == raw
