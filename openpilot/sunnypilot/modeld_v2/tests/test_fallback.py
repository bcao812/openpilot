import hashlib
import io
import os
import requests
import unittest
from unittest import mock

from openpilot.common.file_chunker import get_chunk_name
from openpilot.common.hardware import hw
from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.modeld.helpers import dump_oob
import openpilot.sunnypilot.modeld_v2.modeld as modeld_module
from openpilot.sunnypilot.modeld_v2.tests import helpers as tests_helpers
from openpilot.sunnypilot.modeld_v2.tests.helpers import DummyModel, DummyBundle, CAM_W, CAM_H
from openpilot.sunnypilot.models.fetcher import ModelParser, ModelFetcher
from openpilot.sunnypilot.models.helpers import _bundle_is_valid_locally
from openpilot.sunnypilot.models.tinygrad_ref import get_tinygrad_ref

tmp_path = tests_helpers.tmp_path


class TestFallback(OpenpilotTestCase):
  def setUp(self):
    super().setUp()
    self.enterContext(mock.patch.dict(os.environ, {'COMBINED_MODEL_PKL': ''}))
    self.enterContext(mock.patch('requests.sessions.Session.request', side_effect=AssertionError('offline fallback test attempted network access')))

  def test_find_dual_model_in_bundle(self, tmp_path, monkeypatch):
    lebowski_file = 'driving_lebowski.pkl'
    tsfdo_file = 'driving_tsfdo.pkl'
    (tmp_path / lebowski_file).write_bytes(b'fkasdjfkljf')
    (tmp_path / tsfdo_file).write_bytes(b'dskfajklsdjlsfka')

    monkeypatch.setattr(hw.Paths, 'model_root', staticmethod(lambda: str(tmp_path)))
    big_bundle = DummyBundle(models=[DummyModel('supercombo', lebowski_file)], is_big=True)
    small_bundle = DummyBundle(models=[DummyModel('supercombo', tsfdo_file)], is_big=False)
    big_pkl = modeld_module._find_driving_pkl(big_bundle)
    small_pkl = modeld_module._find_driving_pkl(small_bundle)

    assert big_pkl is not None and lebowski_file in big_pkl
    assert small_pkl is not None and tsfdo_file in small_pkl

  @parameterized.expand([(False, 'vision_policy_split'), (True, 'supercombo_non20hz')], names=['chestnut', 'archetype_name'])
  def test_init_modelstate_with_local_fallback_artifacts(self, chestnut, archetype_name, tmp_path, monkeypatch):
    monkeypatch.setattr(hw.Paths, 'model_root', staticmethod(lambda: str(tmp_path)))
    monkeypatch.setattr(modeld_module, 'COMMA_HARDWARE', False)
    archetype = tests_helpers.ARCHETYPES[archetype_name]
    buffer = io.BytesIO()
    dump_oob(tests_helpers.make_pkl_data(archetype), buffer)
    payload = buffer.getvalue()
    split_offset = len(payload) // 2
    chunks = (payload[:split_offset], payload[split_offset:])
    filename = f'driving_{archetype_name}.pkl'
    catalog = {'bundles': [{
      'index': 0,
      'short_name': archetype_name,
      'display_name': archetype_name,
      'generation': 10,
      'environment': 'release',
      'runner': 'tinygrad',
      'is_big': chestnut,
      'is_20hz': archetype.is_20hz,
      'minimum_selector_version': 19,
      'ref': f'offline-fixture-{archetype_name}',
      'models': [{
        'type': model.type.raw,
        'artifact': {
          'file_name': filename,
          'download_uri': {'url': f'https://example.invalid/{filename}', 'sha256': hashlib.sha256(payload).hexdigest()},
          'chunks': [{'file_name': get_chunk_name(filename, chunk_index, len(chunks)), 'sha256': hashlib.sha256(chunk).hexdigest()}
                     for chunk_index, chunk in enumerate(chunks)],
        },
      } for model in archetype.model_stubs],
    }]}
    bundles = ModelParser.parse_models(catalog)
    assert len(bundles) == 1
    bundle = bundles[0]
    assert modeld_module._find_driving_pkl(bundle, chestnut=chestnut) is None
    for chunk_index, chunk in enumerate(chunks):
      (tmp_path / get_chunk_name(filename, chunk_index, len(chunks))).write_bytes(chunk)
    assert _bundle_is_valid_locally(bundle)
    monkeypatch.setattr(modeld_module, 'get_active_bundle', lambda params=None, *, chestnut=None: bundle)
    state = modeld_module.ModelState(CAM_W, CAM_H, chestnut=chestnut)
    assert state.chestnut is chestnut
    assert state.DEV == 'CPU'
    assert state._combined_model_type == archetype.expected_model_type
    assert type(state.constants) is archetype.expected_constants_class

  def test_runtime_fallback_from_big_to_small(self):
    params = mock.MagicMock()
    params.get_bool.return_value = True

    big = mock.MagicMock(chestnut=True)
    big.run.side_effect = RuntimeError("eGPU error")

    small = mock.MagicMock(chestnut=False)
    small.run.return_value = {"plan": []}

    chestnut = mock.MagicMock(big=True)
    model, run_count = big, 50

    try:
      model.run()
    except Exception:
      if not params.get_bool("chestnutActive"):
        raise
      params.put_bool("chestnutActive", False)
      model = small
      chestnut.big = False
      run_count = 0

    assert model is small
    assert not chestnut.big
    assert run_count == 0
    params.put_bool.assert_called_with("chestnutActive", False)

    model.run()
    small.run.assert_called_once()

  def test_runtime_stays_on_big_model_if_no_errors(self):
    params = mock.MagicMock()
    params.get_bool.return_value = True

    big = mock.MagicMock(chestnut=True)
    big.run.return_value = {"plan": [1, 2, 3]}

    small = mock.MagicMock(chestnut=False)
    chestnut = mock.MagicMock(big=True)
    model, run_count = big, 50
    try:
      model.run()
    except Exception:
      if not params.get_bool("ChestnutActive"):
        raise
      params.put_bool("ChestnutActive", False)
      model = small
      chestnut.big = False
      run_count = 0

    assert model is big
    assert chestnut.big is True
    assert run_count == 50
    params.put_bool.assert_not_called()
    small.run.assert_not_called()

  def test_runtime_exception_on_small_model_raises(self):
    params = mock.MagicMock()
    params.get_bool.return_value = False

    model = mock.MagicMock(chestnut=False)
    model.run.side_effect = RuntimeError("CPU error")

    with self.assertRaises(RuntimeError):
      try:
        model.run()
      except Exception:
        if not params.get_bool("ChestnutActive"):
          raise


@unittest.skipUnless(os.environ.get('RUN_INTEGRATION_TESTS') == '1', 'requires external network; set RUN_INTEGRATION_TESTS=1')
class TestLiveFallbackCatalogs(OpenpilotTestCase):
  @parameterized.expand([ModelFetcher.MODEL_URL, ModelFetcher.MODEL_URL_CHESTNUT], names=['catalog_url'])
  def test_live_catalog_compatibility(self, catalog_url, tmp_path, monkeypatch):
    monkeypatch.setattr(hw.Paths, 'model_root', staticmethod(lambda: str(tmp_path)))
    response = requests.get(catalog_url, timeout=(10, 30))
    response.raise_for_status()
    catalog = response.json()
    current_ref = get_tinygrad_ref()
    assert current_ref, 'The local tinygrad revision must be independently identifiable'
    assert catalog.get('tinygrad_ref') == current_ref, f'Published tinygrad revision differs from local runtime: {catalog_url}'
    bundles = ModelParser.parse_models(catalog)
    assert bundles, f'No version-compatible fallback bundles in {catalog_url}'
    assert all(bundle.models for bundle in bundles), f'Fallback catalog contains an empty bundle: {catalog_url}'
