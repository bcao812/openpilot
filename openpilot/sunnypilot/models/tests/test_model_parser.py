from pathlib import Path
import tempfile
import unittest
from unittest import mock

from capnp.lib.capnp import KjException

from openpilot.cereal import custom
from openpilot.common.hardware.hw import Paths
from openpilot.sunnypilot.models.fetcher import ModelParser
from openpilot.sunnypilot.models.helpers import REQUIRED_JSON_VERSION


def catalogue_bundle(index=7, version=REQUIRED_JSON_VERSION):
  return {
    "index": str(index),
    "short_name": f"bundle-{index}",
    "display_name": f"Bundle {index}",
    "generation": "9",
    "environment": "test",
    "runner": "tinygrad",
    "is_20hz": True,
    "minimum_selector_version": version,
    "ref": f"ref-{index}",
    "overrides": {"Z_LAST": "last", "A_FIRST": "first"},
    "models": [
      {
        "type": "vision",
        "artifact": {
          "file_name": f"vision-{index}.pkl",
          "download_uri": {"url": "https://fixture.invalid/vision.pkl", "sha256": "a" * 64},
          "chunks": [
            {"file_name": "vision.chunk0", "sha256": "b" * 64},
            {"file_name": "vision.chunk1", "sha256": "c" * 64},
          ],
        },
      },
      {
        "type": "policy",
        "artifact": {
          "file_name": f"policy-{index}.pkl",
          "download_uri": {"url": "https://fixture.invalid/policy.pkl", "sha256": "d" * 64},
        },
      },
    ],
  }


class TestModelParser(unittest.TestCase):
  def setUp(self):
    self.model_root = Path(self.enterContext(tempfile.TemporaryDirectory())) / "models"
    self.enterContext(mock.patch.object(Paths, "model_root", return_value=str(self.model_root)))

  def test_exact_version_filter_normalizes_strings_and_preserves_order(self):
    bundles = [
      catalogue_bundle(30, str(REQUIRED_JSON_VERSION)),
      catalogue_bundle(2, REQUIRED_JSON_VERSION - 1),
      catalogue_bundle(20, REQUIRED_JSON_VERSION),
      catalogue_bundle(1, REQUIRED_JSON_VERSION + 1),
    ]
    parsed = ModelParser.parse_models({"bundles": bundles})
    self.assertEqual([(bundle.index, bundle.ref) for bundle in parsed], [(30, "ref-30"), (20, "ref-20")])
    self.assertEqual([bundle.minimumSelectorVersion for bundle in parsed], [REQUIRED_JSON_VERSION] * 2)

  def test_nested_bundle_fields_are_preserved(self):
    expected = custom.ModelManagerSP.ModelBundle.new_message(
      index=7,
      internalName="bundle-7",
      displayName="Bundle 7",
      generation=9,
      environment="test",
      runner="tinygrad",
      is20hz=True,
      minimumSelectorVersion=REQUIRED_JSON_VERSION,
      ref="ref-7",
      status="notDownloading",
      overrides=[{"key": "Z_LAST", "value": "last"}, {"key": "A_FIRST", "value": "first"}],
      models=[
        {
          "type": "vision",
          "artifact": {
            "fileName": "vision-7.pkl",
            "downloadUri": {"uri": "https://fixture.invalid/vision.pkl", "sha256": "a" * 64},
            "chunks": [
              {"fileName": "vision.chunk0", "sha256": "b" * 64},
              {"fileName": "vision.chunk1", "sha256": "c" * 64},
            ],
          },
        },
        {
          "type": "policy",
          "artifact": {
            "fileName": "policy-7.pkl",
            "downloadUri": {"uri": "https://fixture.invalid/policy.pkl", "sha256": "d" * 64},
          },
        },
      ],
    )
    parsed, = ModelParser.parse_models({"bundles": [catalogue_bundle(version=str(REQUIRED_JSON_VERSION))]})
    self.assertEqual(parsed.to_dict(), expected.to_dict())

  def test_chunk_manifests_are_created_and_repaired_even_for_filtered_bundles(self):
    catalogue = {"bundles": [catalogue_bundle(1), catalogue_bundle(2, REQUIRED_JSON_VERSION + 1)]}
    parsed = ModelParser.parse_models(catalogue)
    self.assertEqual([bundle.index for bundle in parsed], [1])
    manifests = [self.model_root / f"vision-{index}.pkl.chunkmanifest" for index in (1, 2)]
    for manifest in manifests:
      self.assertEqual(manifest.read_text(), "2")

    manifests[0].write_text("99")
    manifests[1].unlink()
    parsed_again = ModelParser.parse_models(catalogue)
    self.assertEqual([bundle.to_dict() for bundle in parsed_again], [bundle.to_dict() for bundle in parsed])
    for manifest in manifests:
      self.assertEqual(manifest.read_text(), "2")
    self.assertFalse((self.model_root / "policy-1.pkl.chunkmanifest").exists())

  def test_invalid_versions_and_missing_required_fields_fail(self):
    for version, exception in (("invalid", ValueError), (None, TypeError)):
      with self.subTest(version=version), self.assertRaises(exception):
        ModelParser.parse_models({"bundles": [catalogue_bundle(version=version)]})
    missing_version = catalogue_bundle()
    del missing_version["minimum_selector_version"]
    with self.assertRaises(KeyError):
      ModelParser.parse_models({"bundles": [missing_version]})

  def test_incompatible_bundles_still_undergo_full_validation(self):
    invalid_index = catalogue_bundle(version=REQUIRED_JSON_VERSION + 1)
    invalid_index["index"] = "invalid"
    invalid_model = catalogue_bundle(version=REQUIRED_JSON_VERSION + 1)
    invalid_model["models"][0]["type"] = "invalid"
    for bundle, exception in ((invalid_index, ValueError), (invalid_model, KjException)):
      with self.subTest(exception=exception), self.assertRaises(exception):
        ModelParser.parse_models({"bundles": [catalogue_bundle(1), bundle]})

  def test_unknown_numeric_enums_fail_after_all_bundle_manifests_are_created(self):
    for version in (REQUIRED_JSON_VERSION, REQUIRED_JSON_VERSION + 1):
      for field in ("runner", "model_type"):
        with self.subTest(version=version, field=field):
          invalid_bundle = catalogue_bundle(1, version)
          if field == "runner":
            invalid_bundle["runner"] = 65535
          else:
            invalid_bundle["models"][0]["type"] = 65535
          manifest = self.model_root / "vision-2.pkl.chunkmanifest"
          manifest.unlink(missing_ok=True)
          with self.assertRaisesRegex(RuntimeError, r"^Member was null\.$"):
            ModelParser.parse_models({"bundles": [invalid_bundle, catalogue_bundle(2)]})
          self.assertEqual(manifest.read_text(), "2")

  def test_empty_catalogue(self):
    self.assertEqual(ModelParser.parse_models({}), [])
    self.assertEqual(ModelParser.parse_models({"bundles": []}), [])


if __name__ == "__main__":
  unittest.main()
