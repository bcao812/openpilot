"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.system.params_migration import CAR_LIST_JSON_OUT, _migrate_car_platform_bundle, _migrate_model_bundle_slots


class TestCarPlatformMigration(OpenpilotTestCase):
  def test_current_accord_is_not_a_legacy_alias(self):
    from opendbc.car.fingerprints import MIGRATION
    self.assertNotIn("HONDA_ACCORD", MIGRATION)

  def test_current_accord_migration_does_not_load_vehicle_databases(self):
    params = Params()
    bundle = {"platform": "HONDA_ACCORD", "model": "Accord Hybrid", "brand": "honda"}
    params.put("CarPlatformBundle", bundle, block=True)
    result = subprocess.run([sys.executable, "-B", "-c", """
import importlib.abc
import sys

class RejectVehicleDatabases(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    if fullname.startswith('opendbc.car.') and (
      fullname.endswith(('.values', '.fingerprints')) or fullname == 'opendbc.car.fingerprint_migration'
    ):
      raise AssertionError('current Accord migration imported a vehicle database: ' + fullname)

sys.meta_path.insert(0, RejectVehicleDatabases())
from openpilot.common.params import Params
from openpilot.sunnypilot.system.params_migration import _migrate_car_platform_bundle
_migrate_car_platform_bundle(Params())
"""], env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True, timeout=20)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    self.assertEqual(params.get("CarPlatformBundle"), bundle)

  def test_canonical_looking_legacy_names_still_migrate(self):
    from opendbc.car.fingerprints import MIGRATION
    params = Params()
    for platform in ("SKODA_SCALA_MK1", "RIVIAN_R1_GEN1"):
      with self.subTest(platform=platform):
        expected = str(MIGRATION[platform])
        params.put("CarPlatformBundle", {"platform": platform}, block=True)
        _migrate_car_platform_bundle(params)
        self.assertEqual(params.get("CarPlatformBundle")["platform"], expected)

  def test_legacy_platform_uses_local_vehicle_catalogue(self):
    params = Params()
    params.put("CarPlatformBundle", {"platform": "HONDA ACCORD 2018 HYBRID TOURING", "model": "Accord"}, block=True)
    _migrate_car_platform_bundle(params)
    bundle = params.get("CarPlatformBundle")
    self.assertEqual(bundle["platform"], "HONDA_ACCORD")
    self.assertEqual(bundle["brand"], "honda")
    catalogue = json.loads(Path(CAR_LIST_JSON_OUT).read_text())
    self.assertEqual(bundle, {**catalogue[bundle["name"]], "name": bundle["name"]})


class TestModelBundleSlotMigration(OpenpilotTestCase):
  """Pre-split, a chestnut user's big-model selection lived in the single ActiveBundle.
  The migration seeds both slots; per-source validation later drops whichever does not
  match its own manifest."""

  def test_seeds_chestnut_slot_from_active_bundle(self):
    params = Params()
    bundle = {"ref": "big", "minimumSelectorVersion": 18}
    params.put("ModelManager_ActiveBundle", bundle, block=True)
    _migrate_model_bundle_slots(params)
    assert params.get("ModelManager_ActiveBundleChestnut") == bundle
    assert params.get("ModelManager_ActiveBundle") == bundle

  def test_noop_when_chestnut_slot_already_set(self):
    params = Params()
    params.put("ModelManager_ActiveBundle", {"ref": "small"}, block=True)
    params.put("ModelManager_ActiveBundleChestnut", {"ref": "big"}, block=True)
    _migrate_model_bundle_slots(params)
    assert params.get("ModelManager_ActiveBundleChestnut") == {"ref": "big"}

  def test_noop_when_no_selection(self):
    params = Params()
    _migrate_model_bundle_slots(params)
    assert params.get("ModelManager_ActiveBundleChestnut") is None
