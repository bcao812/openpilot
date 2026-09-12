import hashlib
import io
import os
import requests
import unittest
from types import SimpleNamespace
from unittest import mock

from openpilot.common.file_chunker import get_chunk_name
from openpilot.common.hardware import hw
from openpilot.common.parameterized import parameterized
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.modeld.helpers import dump_oob
import openpilot.sunnypilot.modeld_v2.modeld as modeld_module
from openpilot.sunnypilot.modeld_v2.tests import helpers as tests_helpers
from openpilot.sunnypilot.modeld_v2.tests.helpers import DummyModel, DummyBundle, CAM_W, CAM_H
from openpilot.sunnypilot.models.fetcher import ModelParser, ModelFetcher
from openpilot.sunnypilot.models.helpers import _bundle_is_valid_locally
from openpilot.sunnypilot.models.tinygrad_ref import get_tinygrad_ref

tmp_path = tests_helpers.tmp_path


class FrameLoopFinished(Exception):
  pass


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

  def _prepare_main_loop(self, max_frames, chestnut_present=True, big_failure_at=None, small_error=None):
    params = Params()
    car_params = modeld_module.car.CarParams(brand='honda', longitudinalActuatorDelay=0.2)
    params.put('CarParams', car_params.to_bytes(), block=True)
    put_bool = params.put_bool
    self.enterContext(mock.patch.object(params, 'put_bool', side_effect=lambda key, value, block=False: put_bool(key, value, block=True)))

    def make_model(chestnut):
      return mock.Mock(chestnut=chestnut, constants=SimpleNamespace(MODEL_FREQ=20, DESIRE_LEN=8),
                       LAT_SMOOTH_SECONDS=0.1, LONG_SMOOTH_SECONDS=0.3, lat_delay=0.2,
                       vision_input_names=['img', 'big_img'], desire_key='desire', numpy_inputs={})

    big, small = make_model(True), make_model(False)

    def run_big(bufs, transforms, inputs, after_enqueue):
      if big.run.call_count == big_failure_at:
        raise RuntimeError('eGPU error')
      if after_enqueue is not None:
        after_enqueue()

    def run_small(bufs, transforms, inputs, after_enqueue):
      if small_error is not None:
        raise small_error
      if after_enqueue is not None:
        after_enqueue()

    big.run.side_effect = run_big
    small.run.side_effect = run_small
    chestnut = SimpleNamespace(big=chestnut_present, send=mock.Mock())
    client = mock.Mock(width=CAM_W, height=CAM_H, buffer_len=1, frame_id=0, timestamp_sof=0, timestamp_eof=0)
    client.connect.return_value = True
    messages = {
      'driverMonitoringState': SimpleNamespace(isRHD=False),
      'narrowRoadCameraState': SimpleNamespace(frameId=0),
      'carState': SimpleNamespace(vEgo=5.0),
    }

    def recv():
      if client.frame_id >= max_frames:
        raise FrameLoopFinished
      client.frame_id += 1
      client.timestamp_sof = client.frame_id * 50_000_000
      client.timestamp_eof = client.timestamp_sof
      messages['narrowRoadCameraState'].frameId = client.frame_id
      return object()

    client.recv.side_effect = recv
    subscriber = mock.MagicMock(frame=1, updated={'extrinsicsCalibration': False})
    subscriber.__getitem__.side_effect = messages.__getitem__
    vision_client = mock.Mock(return_value=client)
    vision_client.available_streams.return_value = [modeld_module.VisionStreamType.VISION_STREAM_NARROW_ROAD]

    self.enterContext(mock.patch.object(modeld_module, 'config_realtime_process'))
    self.enterContext(mock.patch.object(modeld_module, 'setproctitle'))
    self.enterContext(mock.patch.object(modeld_module, 'chestnut_present', return_value=chestnut_present))
    self.enterContext(mock.patch.object(modeld_module, 'Params', return_value=params))
    self.enterContext(mock.patch.object(modeld_module, 'VisionIpcClient', vision_client))
    self.enterContext(mock.patch.object(modeld_module, 'ModelState', side_effect=lambda **kwargs: big if kwargs['chestnut'] else small))
    self.enterContext(mock.patch.object(modeld_module, 'PubMaster'))
    self.enterContext(mock.patch.object(modeld_module, 'SubMaster', return_value=subscriber))
    self.enterContext(mock.patch.object(modeld_module, 'ChestnutState', return_value=chestnut))
    self.enterContext(mock.patch.object(modeld_module, 'DesireHelper', return_value=SimpleNamespace(desire=0)))
    self.enterContext(mock.patch.object(modeld_module, 'load_meta_constants'))
    self.enterContext(mock.patch.object(modeld_module, 'RoadEdgeLaneChangeController'))
    return params, big, small, chestnut

  def test_runtime_fallback_from_big_to_small(self):
    cadence = round(20 / modeld_module.SERVICE_LIST['chestnutState'].frequency)
    assert cadence > 1
    params, big, small, chestnut = self._prepare_main_loop(2 * cadence - 1, big_failure_at=cadence - 1)
    with self.assertRaises(FrameLoopFinished):
      modeld_module.main()
    assert big.run.call_count == cadence - 1
    assert small.run.call_count == cadence
    assert params.get_bool('ChestnutModelError')
    assert not params.get_bool('ChestnutActive')
    assert chestnut.big is False
    assert all(call.args[3] is None for call in small.run.call_args_list[:-1])
    assert small.run.call_args_list[-1].args[3] is chestnut.send
    chestnut.send.assert_called_once()

  def test_runtime_stays_on_big_model_if_no_errors(self):
    params, big, small, chestnut = self._prepare_main_loop(3)
    with self.assertRaises(FrameLoopFinished):
      modeld_module.main()
    assert big.run.call_count == 3
    small.run.assert_not_called()
    assert params.get_bool('ChestnutActive')
    assert not params.get_bool('ChestnutModelError')
    assert chestnut.big is True

  def test_runtime_exception_on_small_model_raises(self):
    params, big, small, chestnut = self._prepare_main_loop(1, chestnut_present=False, small_error=RuntimeError('CPU error'))
    with self.assertRaisesRegex(RuntimeError, 'CPU error'):
      modeld_module.main()
    small.run.assert_called_once()
    big.run.assert_not_called()
    assert not params.get_bool('ChestnutActive')
    assert not params.get_bool('ChestnutModelError')


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
