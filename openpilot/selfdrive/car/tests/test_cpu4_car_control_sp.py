"""Preserve the complete wire/dataclass and Honda apply contracts when caching."""
import struct
import unittest

from openpilot.cereal import custom
from openpilot.selfdrive.car.helpers import CarControlSPConverter, asdictref, convert_carControlSP
from openpilot.selfdrive.car.tests.test_car_control_sp_conversion import lead_fixture


def frozen_values(value):
  """Compare float bits too, including NaNs and signed zero from the wire."""
  if isinstance(value, dict):
    return tuple((key, frozen_values(item)) for key, item in sorted(value.items()))
  if isinstance(value, list):
    return tuple(frozen_values(item) for item in value)
  if isinstance(value, float):
    return ('float64', struct.pack('!d', value))
  return value


def populated_message(index=0):
  return custom.CarControlSP.new_message(
    leadOne=lead_fixture(index), leadTwo=lead_fixture(index + 1, 'predicted', 'edge'),
    mads={'state': 'overriding', 'enabled': True, 'active': True, 'available': True},
    params=[{'key': 'wire-check', 'type': 'bytes', 'value': b'\x00\xff\x17', 'valueDEPRECATED': 'kept'}],
    intelligentCruiseButtonManagement={'state': 'increasing', 'sendButton': 'increase', 'vTarget': 33.25},
  )


class TestCPU4CarControlSP(unittest.TestCase):
  def assert_matches_full_converter(self, converter, message):
    expected = convert_carControlSP(message)
    actual = converter(message)
    self.assertEqual(frozen_values(asdictref(actual)), frozen_values(asdictref(expected)))
    return actual

  def test_missing_and_explicit_default_pointers_keep_their_distinct_defaults(self):
    converter = CarControlSPConverter()
    variants = (
      custom.CarControlSP.new_message(),
      custom.CarControlSP.new_message(leadOne={}, leadTwo={}),
      custom.CarControlSP.new_message(leadOne={'evidence': {}}, leadTwo={'status': True}),
      custom.CarControlSP.new_message(mads={}, intelligentCruiseButtonManagement={}, params=[]),
      populated_message(),
    )
    for message in (*variants, *reversed(variants)):
      with self.subTest(payload=message.to_dict()):
        self.assert_matches_full_converter(converter, message)
        self.assert_matches_full_converter(converter, message.as_reader())
        with custom.CarControlSP.from_bytes(message.to_bytes()) as received:
          self.assert_matches_full_converter(converter, received)

    missing = converter(custom.CarControlSP.new_message())
    present = converter(custom.CarControlSP.new_message(leadOne={}))
    self.assertEqual(missing.leadOne.radarTrackId, 0)
    self.assertEqual(present.leadOne.radarTrackId, -1)

  def test_mutable_reader_and_same_measurement_timestamps_take_current_contents(self):
    message = populated_message()
    reader = message.as_reader()  # A reader view still aliases this builder.
    converter = CarControlSPConverter()
    first = self.assert_matches_full_converter(converter, reader)
    original_timestamp = message.leadOne.evidence.positionMonoTime
    message.leadOne.dRel = 6.5
    message.leadOne.status = False
    message.leadOne.evidence.followQualified = False
    message.leadOne.evidence.source = 'vision'
    message.leadOne.evidence.reason = 'current observation withdrew qualification'
    message.leadOne.evidence.trackId = (1 << 64) - 1
    message.leadOne.aLeadDEPRECATED = 72.5
    second = self.assert_matches_full_converter(converter, reader)
    self.assertEqual(message.leadOne.evidence.positionMonoTime, original_timestamp)
    self.assertNotEqual(first.leadOne.dRel, second.leadOne.dRel)
    self.assertFalse(second.leadOne.status)
    self.assertEqual(second.leadOne.evidence['trackId'], (1 << 64) - 1)
    self.assertEqual(second.leadOne.aLeadDEPRECATED, 0)

  def test_results_and_both_leads_remain_independently_mutable(self):
    message = populated_message()
    reader = message.as_reader()
    converter = CarControlSPConverter()
    first = self.assert_matches_full_converter(converter, reader)
    second = self.assert_matches_full_converter(converter, reader)
    first.leadOne.evidence['reason'] = 'consumer mutation'
    first.leadTwo.evidence['trackId'] = 1
    first.leadOne.dRel = -1
    first.mads.active = False
    first.params[0]['value'] = b'changed'
    first.intelligentCruiseButtonManagement.vTarget = 0
    third = self.assert_matches_full_converter(converter, reader)
    self.assertEqual(frozen_values(asdictref(second)), frozen_values(asdictref(third)))
    self.assertEqual(message.leadOne.evidence.reason, 'wire-reason-0')
    self.assertNotEqual(first.leadOne.evidence, second.leadOne.evidence)
    self.assertIsNot(second.leadOne.evidence, third.leadOne.evidence)
    self.assertIsNot(third.leadOne.evidence, third.leadTwo.evidence)

  def test_current_mads_buttons_and_params_change_while_leads_repeat(self):
    message = populated_message()
    converter = CarControlSPConverter()
    for index, button in enumerate(('none', 'increase', 'decrease', 'none')):
      message.mads.active = index % 2 == 0
      message.mads.enabled = index < 3
      message.mads.state = 'disabled' if index == 3 else 'enabled'
      message.intelligentCruiseButtonManagement.sendButton = button
      message.intelligentCruiseButtonManagement.vTarget = index * 10.25
      message.params[0].value = bytes((index, 0, 255))
      self.assert_matches_full_converter(converter, message.as_reader())

  def test_lead_replacement_clear_and_reordering_do_not_reuse_an_old_lead(self):
    converter = CarControlSPConverter()
    for message in (
      populated_message(),
      custom.CarControlSP.new_message(leadOne=lead_fixture(1), leadTwo=lead_fixture(0)),
      custom.CarControlSP.new_message(leadOne={'status': False}, leadTwo={'evidence': {}}),
      custom.CarControlSP.new_message(),
      populated_message(2),
    ):
      self.assert_matches_full_converter(converter, message.as_reader())

  def test_nonfinite_and_signed_zero_values_are_preserved(self):
    converter = CarControlSPConverter()
    for value in (float('nan'), float('inf'), -float('inf'), -0.0):
      message = populated_message()
      message.leadOne.dRel = value
      message.leadTwo.evidence.physicalConflictStart = value
      for _ in range(2):
        self.assert_matches_full_converter(converter, message.as_reader())

  def test_honda_apply_requests_actuators_and_can_match_the_full_converter(self):
    from openpilot.selfdrive.car.tests.test_car_control_sp_startup import startup_harness

    for engaged in (False, True):
      with self.subTest(engaged=engaged):
        baseline, _ = startup_harness(lead_present=True, engaged=engaged, with_evidence=True)
        candidate, _ = startup_harness(lead_present=True, engaged=engaged, with_evidence=True)
        baseline.card._car_control_sp_converter = convert_carControlSP
        for frame in range(31):
          for harness in (baseline, candidate):
            extension = harness.events['carControlSP'].carControlSP
            extension.mads.available = True
            extension.mads.enabled = frame % 2 == 0
            extension.mads.active = engaged
            extension.intelligentCruiseButtonManagement.sendButton = 'increase' if frame == 10 else 'none'
            extension.intelligentCruiseButtonManagement.vTarget = 75.0 + frame
            if frame % 5 == 0:
              extension.leadOne.dRel = 20.0 + frame
              extension.leadOne.evidence.trackId = (1 << 63) + frame
            extension.leadOne.status = frame < 15 or frame >= 25
            extension.leadOne.evidence.followQualified = extension.leadOne.status
            harness.state.gasPressed = frame == 17
            harness.state.brakePressed = frame == 18
            harness.step(stale=frame == 20)
          self.assertEqual(baseline.requests, candidate.requests)
          self.assertEqual(baseline.sent_frames, candidate.sent_frames)
          self.assertEqual(baseline.card.last_actuators_output.to_dict(), candidate.card.last_actuators_output.to_dict())


if __name__ == '__main__':
  unittest.main()
