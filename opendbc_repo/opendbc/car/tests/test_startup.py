import json
import os
import subprocess
import sys
import textwrap
import unittest
from unittest.mock import Mock, patch

from opendbc.car import gen_empty_fingerprint
from opendbc.car import car_helpers, fw_versions
from opendbc.car.brand_data import BrandData
from opendbc.car.honda.values import CAR, FW_QUERY_CONFIG, HondaFlags, HondaSafetyFlags
from opendbc.car.interfaces import get_interface_attr, get_platform
from opendbc.car.structs import CarParams
from opendbc.car.vin import VIN_UNKNOWN
from opendbc.sunnypilot.car.honda.values_ext import HondaFlagsSP


def accord_firmware():
  return [CarParams.CarFw(ecu=ecu, address=address, subAddress=sub_address or 0,
                          fwVersion=versions[0], brand='honda')
          for (ecu, address, sub_address), versions in fw_versions.VERSIONS['honda'][CAR.HONDA_ACCORD].items()]


def recorded_accord_firmware():
  # Actual ECU replies from the user's 2019 Accord Hybrid, independently
  # recorded at both 2026-11-03 and 2026-11-04 starts (b9/ba segment 0).
  return [
    CarParams.CarFw(ecu='fwdCamera', address=0x18dab5f1, fwVersion=b'36161-TBY-H030\x00\x00', brand='honda', bus=1),
    CarParams.CarFw(ecu='fwdRadar', address=0x18dab0f1, fwVersion=b'36802-TBY-H050\x00\x00', brand='honda', bus=1),
  ]


class TestStartupImports(unittest.TestCase):
  def test_accord_lookup_and_matching_do_not_import_other_interfaces_or_firmware(self):
    # Use a fresh interpreter: ordinary test collection intentionally imports
    # every brand, which would hide a startup dependency regression here.
    code = textwrap.dedent('''
      import json
      import sys
      from opendbc.car import gen_empty_fingerprint
      from opendbc.car.car_helpers import interfaces
      from opendbc.car.fw_versions import VERSIONS, match_fw_to_car
      from opendbc.car.structs import CarParams
      interface = interfaces['HONDA_ACCORD']
      fp = gen_empty_fingerprint()
      fp[1] = {0x184: 8, 0x191: 8, 0x1be: 3}
      firmware = [CarParams.CarFw(ecu=ecu, address=addr, subAddress=sub or 0,
                                 fwVersion=versions[0], brand='honda')
                  for (ecu, addr, sub), versions in VERSIONS['honda']['HONDA_ACCORD'].items()]
      match_fw_to_car(firmware, '0' * 17, query_brand='honda')
      cp = interface.get_params('HONDA_ACCORD', fp, firmware, True, False, False)
      interface.get_params_sp(cp, 'HONDA_ACCORD', fp, firmware, True, False, False)
      print(json.dumps(sorted(name for name in sys.modules if name.startswith('opendbc.car.'))))
    ''')
    result = subprocess.run([sys.executable, '-B', '-c', code], text=True, capture_output=True, check=True)
    modules = json.loads(result.stdout.strip().splitlines()[-1])
    self.assertNotIn('opendbc.car.values', modules)
    self.assertNotIn('opendbc.car.fingerprint_migration', modules)
    self.assertEqual({name for name in modules if name.endswith('.interface')}, {'opendbc.car.honda.interface'})
    # Body and GM supply the unchanged CAN-only sampling table. Every other
    # manufacturer's firmware database must stay outside this startup path.
    self.assertEqual({name for name in modules if name.endswith('.fingerprints')}, {
      'opendbc.car.fingerprints', 'opendbc.car.body.fingerprints',
      'opendbc.car.gm.fingerprints', 'opendbc.car.honda.fingerprints',
    })

  def test_can_only_table_remains_complete(self):
    from opendbc.car.fingerprints import _FINGERPRINTS
    self.assertEqual(_FINGERPRINTS, get_interface_attr('FINGERPRINTS', combine_brands=True, ignore_none=True))

  def test_legacy_mapping_imports_remain_available(self):
    from opendbc.car.car_helpers import interface_names, interfaces
    from opendbc.car.fingerprints import FW_VERSIONS, MIGRATION
    from opendbc.car.fingerprint_migration import MIGRATION as EXPECTED_MIGRATION
    from opendbc.car.values import PLATFORMS
    self.assertIs(MIGRATION, EXPECTED_MIGRATION)
    self.assertEqual(FW_VERSIONS, get_interface_attr('FW_VERSIONS', combine_brands=True, ignore_none=True))
    self.assertEqual(set(interfaces), set(PLATFORMS))
    self.assertEqual(len(interfaces), len(PLATFORMS))
    self.assertEqual(set(interface_names), {p.__module__.split('.')[-2] for p in PLATFORMS.values()})
    for platform in PLATFORMS.values():
      with self.subTest(platform=platform):
        self.assertIs(get_platform(str(platform)), platform)
    with self.assertRaises(KeyError):
      _ = interfaces['UNKNOWN_PLATFORM']

  def test_brand_data_enumeration_matches_existing_registry(self):
    for attr in ('FW_QUERY_CONFIG', 'FW_VERSIONS'):
      with self.subTest(attr=attr):
        data = BrandData(attr)
        first_honda = data['honda']
        expected = get_interface_attr(attr, ignore_none=True)
        self.assertEqual(dict(data), expected)
        self.assertIs(data['honda'], first_honda)
        self.assertEqual(data.copy(), expected)
        with self.assertRaises(KeyError):
          _ = data['not_a_supported_brand']


class TestHondaStartupQueries(unittest.TestCase):
  def test_recorded_current_car_has_one_exact_match(self):
    for brand in (None, 'honda'):
      with self.subTest(brand=brand):
        self.assertEqual(fw_versions.match_fw_to_car(recorded_accord_firmware(), VIN_UNKNOWN, query_brand=brand),
                         (True, {'HONDA_ACCORD'}))

  def test_recorded_match_still_requires_both_current_known_ecus(self):
    for missing in range(2):
      for unknown in (False, True):
        with self.subTest(missing=missing, unknown=unknown):
          firmware = recorded_accord_firmware()
          if unknown:
            firmware[missing].fwVersion = b'unknown'
          else:
            firmware.pop(missing)
          self.assertEqual(fw_versions.match_fw_to_car(firmware, VIN_UNKNOWN, query_brand='honda'), (True, set()))

  def test_scoped_match_agrees_with_full_match(self):
    for firmware in (accord_firmware(), [], [CarParams.CarFw(ecu='fwdCamera', address=0x18dab5f1,
                                                           fwVersion=b'unknown', brand='honda')]):
      with self.subTest(firmware_count=len(firmware)):
        self.assertEqual(fw_versions.match_fw_to_car(firmware, VIN_UNKNOWN, query_brand='honda'),
                         fw_versions.match_fw_to_car(firmware, VIN_UNKNOWN))

  def test_honda_query_keeps_all_ecus_and_request_variants(self):
    calls = []

    class Query:
      def __init__(self, can_send, can_recv, bus, addrs, request, response, rx_offset):
        calls.append((bus, set(addrs), tuple(request), tuple(response), rx_offset))

      def get_data(self, timeout):
        return {}

    with patch.object(fw_versions, 'IsoTpParallelQuery', Query):
      fw_versions.get_fw_versions(Mock(), Mock(), Mock(), query_brand='honda')
    expected_ecus = FW_QUERY_CONFIG.get_all_ecus(fw_versions.VERSIONS['honda'])
    self.assertIn(CarParams.Ecu.eps, {ecu for ecu, _, _ in expected_ecus})
    self.assertEqual(len(calls), len(FW_QUERY_CONFIG.requests))
    for request, (bus, addrs, request_bytes, response_bytes, rx_offset) in zip(FW_QUERY_CONFIG.requests, calls, strict=True):
      self.assertEqual(bus, request.bus)
      self.assertEqual(request_bytes, tuple(request.request))
      self.assertEqual(response_bytes, tuple(request.response))
      self.assertEqual(rx_offset, request.rx_offset)
      self.assertEqual(addrs, {(addr, sub) for ecu, addr, sub in expected_ecus
                              if not request.whitelist_ecus or ecu in request.whitelist_ecus})

  def _fingerprint(self, fixed, matches, scoped_firmware, fallback_firmware=None):
    samples = gen_empty_fingerprint()
    samples[1] = {0x184: 8, 0x191: 8, 0x1be: 3}
    query_results = [scoped_firmware] if fallback_firmware is None else [scoped_firmware, fallback_firmware]
    with patch.dict(os.environ, {}, clear=True), \
         patch.object(car_helpers, 'get_vin', return_value=(-1, -1, VIN_UNKNOWN)), \
         patch.object(car_helpers, 'get_present_ecus', return_value=set()) as present, \
         patch.object(car_helpers, 'get_fw_versions_ordered', side_effect=query_results) as query, \
         patch.object(car_helpers, 'match_fw_to_car', side_effect=matches), \
         patch.object(car_helpers, 'can_fingerprint', return_value=(None, samples)) as can_sample:
      result = car_helpers.fingerprint(Mock(), Mock(), Mock(), None, fixed)
      can_sample.assert_called_once()
    self.assertEqual(result[1], samples)
    return result, present, query

  def test_fixed_accord_accepts_only_unique_exact_scoped_match(self):
    firmware = accord_firmware()
    result, present, query = self._fingerprint('HONDA_ACCORD', [(True, {'HONDA_ACCORD'})], firmware)
    self.assertEqual(result[0], 'HONDA_ACCORD')
    self.assertEqual(result[4], CarParams.FingerprintSource.fixed)
    self.assertEqual([fw.to_dict() for fw in result[3]], [fw.to_dict() for fw in firmware])
    self.assertEqual(present.call_count, 1)
    self.assertEqual(query.call_count, 1)
    self.assertEqual(query.call_args.kwargs['query_brand'], 'honda')

  def test_incomplete_ambiguous_or_modified_match_uses_full_discovery(self):
    modified = CarParams.CarFw(ecu='eps', address=0x18da30f1, fwVersion=b'39990-TVA,A150\x00\x00', brand='honda')
    fallback_firmware = [modified]
    cases = [(True, set()), (False, {'HONDA_ACCORD'}),
             (True, {'HONDA_ACCORD', 'HONDA_CIVIC'}), (True, {'HONDA_CIVIC'})]
    for match in cases:
      with self.subTest(match=match):
        result, present, query = self._fingerprint('HONDA_ACCORD', [match, (True, set())], [], fallback_firmware)
        self.assertEqual(query.call_count, 2)
        self.assertEqual(present.call_count, 2)
        self.assertEqual(query.call_args_list[0].kwargs['query_brand'], 'honda')
        self.assertNotIn('query_brand', query.call_args_list[1].kwargs)
        self.assertEqual(result[0], 'HONDA_ACCORD')
        self.assertEqual(result[3][0].fwVersion, modified.fwVersion)

  def test_other_platform_and_automatic_detection_keep_generic_query(self):
    for fixed in (None, 'HONDA_CIVIC', 'TOYOTA_PRIUS'):
      with self.subTest(fixed=fixed):
        _, present, query = self._fingerprint(fixed, [(True, set())], [])
        self.assertEqual(query.call_count, 1)
        self.assertEqual(present.call_count, 1)
        self.assertIsNone(query.call_args.kwargs['query_brand'])

  def test_fallback_cannot_erase_an_already_observed_modified_eps(self):
    modified = CarParams.CarFw(ecu='eps', address=0x18da30f1, fwVersion=b'39990-TVA,A150\x00\x00', brand='honda')
    for fallback_firmware in ([], [modified], [fw for fw in accord_firmware() if fw.ecu != 'eps']):
      with self.subTest(fallback_count=len(fallback_firmware)):
        result, _, query = self._fingerprint('HONDA_ACCORD', [(False, set()), (True, set())],
                                             [modified], fallback_firmware)
        self.assertEqual(query.call_count, 2)
        firmware = result[3]
        self.assertEqual(sum(fw.to_dict() == modified.to_dict() for fw in firmware), 1)
        interface = car_helpers.interfaces['HONDA_ACCORD']
        cp = interface.get_params('HONDA_ACCORD', result[1], firmware, True, False, False)
        cp_sp = interface.get_params_sp(cp, 'HONDA_ACCORD', result[1], firmware, True, False, False)
        self.assertTrue(cp_sp.flags & HondaFlagsSP.EPS_MODIFIED)
        self.assertEqual(cp.lateralTuning.which(), 'pid')
        self.assertAlmostEqual(cp.lateralTuning.pid.kpV[0], 0.3)
        self.assertAlmostEqual(cp.lateralTuning.pid.kiV[0], 0.09)

  def test_observed_can_configuration_still_controls_honda_flags(self):
    interface = car_helpers.interfaces['HONDA_ACCORD']
    baseline = gen_empty_fingerprint()
    samples = gen_empty_fingerprint()
    samples[1] = {0x184: 8, 0x191: 8, 0x1be: 3}
    empty = interface.get_params('HONDA_ACCORD', baseline, [], True, False, False)
    observed = interface.get_params('HONDA_ACCORD', samples, [], True, False, False)
    self.assertFalse(empty.flags & HondaFlags.HYBRID)
    self.assertFalse(empty.safetyConfigs[-1].safetyParam & HondaSafetyFlags.ALT_BRAKE)
    self.assertTrue(observed.flags & HondaFlags.HYBRID)
    self.assertTrue(observed.safetyConfigs[-1].safetyParam & HondaSafetyFlags.ALT_BRAKE)
    self.assertEqual(observed.transmissionType, CarParams.TransmissionType.cvt)
