"""Exercise the real carControlSP wire/dataclass boundary without vehicle IPC."""
import unittest

from opendbc.car import structs
from openpilot.cereal import custom
from openpilot.selfdrive.car.helpers import asdictref, convert_carControlSP


def lead_fixture(index, source='fused', path_state='inPath'):
  # Synthetic wire values test transport, not the producer's qualification policy.
  first = index == 0
  stamp = (1 << 56) + index * 1000
  return {
    'dRel': 12.25 + index * 100, 'yRel': -1.125 - index,
    'vRel': -2.25 - index, 'aRel': -.125 - index,
    'vLead': 15.5 + index, 'dPath': .25 + index, 'vLat': -.375 - index,
    'vLeadK': 15.75 + index, 'aLeadK': -.625 - index,
    'fcw': first, 'status': not first, 'aLeadTau': 1.5 + index,
    'modelProb': .875 - index / 4, 'radar': first, 'radarTrackId': 17 + index,
    'evidence': {
      'version': 1, 'source': source, 'sessionId': f'conversion-{index}',
      'trackId': (1 << 62) + index + 13, 'generation': (1 << 57) + index + 17,
      'positionMonoTime': stamp + 1, 'velocityMonoTime': stamp + 3,
      'estimateMonoTime': stamp + 5, 'modelMonoTime': stamp + 7,
      'followGeometryMonoTime': stamp + 9,
      'positionStd': .125 + index, 'velocityStd': .375 + index,
      'positionValid': first, 'velocityValid': not first, 'accelerationValid': first,
      'followQualified': first, 'brakeQualified': not first, 'stopQualified': first,
      'departureQualified': not first, 'fcwQualified': first,
      'coasting': not first, 'cautionQualified': first,
      'physicalConflictValid': not first,
      'physicalConflictStart': .123456789012345 + index / 8,
      'physicalConflictEnd': 1.987654321098765 + index / 8,
      'conflictSupportVersion': 1,
      'conflictSupportedEnd': .625 + index / 8,
      'pathState': path_state, 'pathOffset': -1.25 - index,
      'conflictStart': .25 + index, 'conflictEnd': 2.75 + index,
      'lateralSpeed': -.75 - index, 'pathSource': f'wire-path-{index}',
      'visionIndex': -1 if first else 1, 'reason': f'wire-reason-{index}',
    },
  }


class TestCarControlSPConversion(unittest.TestCase):
  def assert_leads_roundtrip(self, message):
    reader = message.as_reader()
    converted = convert_carControlSP(reader)
    encoded = asdictref(converted)
    rebuilt = custom.CarControlSP.new_message(**encoded)
    for name in ('leadOne', 'leadTwo'):
      with self.subTest(lead=name):
        expected = getattr(reader, name).to_dict()
        self.assertEqual(getattr(converted, name).evidence, expected['evidence'])
        self.assertEqual(encoded[name], expected)
        self.assertEqual(getattr(rebuilt, name).to_dict(), expected)
    return rebuilt

  def test_fixture_covers_live_schema(self):
    lead = lead_fixture(0)
    live_fields = {name for name in custom.LeadData.schema.fields if not name.endswith('DEPRECATED')}
    self.assertEqual(set(lead), live_fields, 'Extend the fixture for new LeadData fields')
    self.assertEqual(set(lead['evidence']), set(custom.RadarLeadEvidence.schema.fields),
                     'Extend the fixture for new RadarLeadEvidence fields')

  def test_full_evidence_survives_conversion_and_capnp_roundtrip(self):
    # This calls the same converter as card and fails with the original
    # LeadData.__init__() unexpected keyword argument 'evidence' regression.
    message = custom.CarControlSP.new_message(
      leadOne=lead_fixture(0), leadTwo=lead_fixture(1, 'predicted', 'edge'),
      mads={'state': 'overriding', 'enabled': True, 'active': True, 'available': True},
      params=[{'key': 'wire-check', 'type': 'bytes', 'value': b'\x00\xff\x17'}],
      intelligentCruiseButtonManagement={'state': 'increasing', 'sendButton': 'increase', 'vTarget': 33.25},
    )
    rebuilt = self.assert_leads_roundtrip(message)
    for name in ('mads', 'intelligentCruiseButtonManagement'):
      self.assertEqual(getattr(rebuilt, name).to_dict(), getattr(message, name).to_dict())
    self.assertEqual(rebuilt.to_dict()['params'], message.to_dict()['params'])

  def test_all_sources_and_path_states_survive(self):
    for source in custom.RadarLeadEvidence.Source.schema.enumerants:
      for path_state in custom.HondaRadarObservations.PathState.schema.enumerants:
        with self.subTest(source=source, path_state=path_state):
          self.assert_leads_roundtrip(custom.CarControlSP.new_message(
            leadOne=lead_fixture(0, source, path_state), leadTwo=lead_fixture(1, source, path_state),
          ))

  def test_absent_evidence_uses_empty_dict(self):
    for initialize_leads in (False, True):
      with self.subTest(initialize_leads=initialize_leads):
        message = custom.CarControlSP.new_message(
          **({'leadOne': {'status': True}, 'leadTwo': {'dRel': 12.25}} if initialize_leads else {})
        )
        reader = message.as_reader()
        for name in ('leadOne', 'leadTwo'):
          self.assertNotIn('evidence', getattr(reader, name).to_dict())
        converted = convert_carControlSP(reader)
        rebuilt = custom.CarControlSP.new_message(**asdictref(converted))
        expected_defaults = custom.RadarLeadEvidence.new_message().to_dict()
        for name in ('leadOne', 'leadTwo'):
          self.assertEqual(getattr(converted, name).evidence, {})
          self.assertEqual(getattr(rebuilt, name).evidence.to_dict(), expected_defaults)

  def test_explicit_default_evidence_survives(self):
    self.assert_leads_roundtrip(custom.CarControlSP.new_message(
      leadOne={'evidence': {}}, leadTwo={'evidence': {}},
    ))

  def test_default_evidence_is_independent(self):
    message = custom.CarControlSP.new_message().as_reader()
    controls = (structs.CarControlSP(), structs.CarControlSP(),
                convert_carControlSP(message), convert_carControlSP(message))
    evidence = [getattr(control, name).evidence for control in controls for name in ('leadOne', 'leadTwo')]
    self.assertEqual(len({id(value) for value in evidence}), len(evidence))
    evidence[0]['reason'] = 'changed'
    for value in evidence[1:]:
      self.assertEqual(value, {})

  def test_populated_evidence_is_independent(self):
    message = custom.CarControlSP.new_message(leadOne=lead_fixture(0), leadTwo=lead_fixture(1)).as_reader()
    first = convert_carControlSP(message)
    second = convert_carControlSP(message)
    evidence = (first.leadOne.evidence, first.leadTwo.evidence, second.leadOne.evidence, second.leadTwo.evidence)
    self.assertEqual(len({id(value) for value in evidence}), len(evidence))
    first.leadOne.evidence['reason'] = 'changed'
    self.assertEqual(message.leadOne.evidence.reason, 'wire-reason-0')
    self.assertEqual(first.leadTwo.evidence['reason'], 'wire-reason-1')
    self.assertEqual(second.leadOne.evidence['reason'], 'wire-reason-0')
    self.assertEqual(second.leadTwo.evidence['reason'], 'wire-reason-1')


if __name__ == '__main__':
  unittest.main()
