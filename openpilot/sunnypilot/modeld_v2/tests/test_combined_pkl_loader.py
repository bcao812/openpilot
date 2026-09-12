"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import os
from pathlib import Path
from unittest import mock

from openpilot.common.file_chunker import chunk_file, get_chunk_name, get_chunk_targets, get_manifest_path
from openpilot.common.hardware import hw
from openpilot.common.parameterized import parameterized

import openpilot.selfdrive.modeld.helpers as stock_helpers
import openpilot.sunnypilot.models.helpers as helpers
import openpilot.sunnypilot.modeld_v2.modeld as modeld_module
from openpilot.sunnypilot.modeld_v2.modeld import _find_driving_pkl, _pkl_exists
from openpilot.sunnypilot.modeld_v2.tests import helpers as tests_helpers
from openpilot.sunnypilot.modeld_v2.tests.helpers import DummyModel, DummyBundle, ARCHETYPES, CAM_W, CAM_H, \
  SPLIT_VISION_INPUT_SHAPES, SPLIT_POLICY_INPUT_SHAPES
from openpilot.common.test import OpenpilotTestCase

# resolved by name from this module when a test asks for them
tmp_path = tests_helpers.tmp_path
patch_modeld = tests_helpers.patch_modeld
model_state_factory = tests_helpers.model_state_factory

ModelState = modeld_module.ModelState


def pkl_roots(tmp_path, monkeypatch):
  bundled_root = tmp_path / 'bundled'
  download_root = tmp_path / 'downloads'
  bundled_root.mkdir()
  download_root.mkdir()
  monkeypatch.setattr(stock_helpers, 'MODELS_DIR', bundled_root)
  monkeypatch.setattr(hw.Paths, 'model_root', staticmethod(lambda: str(download_root)))
  return bundled_root, download_root


def write_chunks(path, num_chunks=2, missing=()):
  Path(get_manifest_path(path)).write_text(str(num_chunks))
  for chunk_index in range(num_chunks):
    if chunk_index not in missing:
      Path(get_chunk_name(path, chunk_index, num_chunks)).write_bytes(b'chunk')


class TestPklExists(OpenpilotTestCase):
  def test_missing_pkl_is_not_available(self, tmp_path):
    assert not _pkl_exists(tmp_path / 'model.pkl')

  def test_complete_pkl_is_available(self, tmp_path):
    path = tmp_path / 'model.pkl'
    path.write_bytes(b'complete')
    assert _pkl_exists(path)

  def test_directory_is_not_a_pkl(self, tmp_path):
    path = tmp_path / 'model.pkl'
    path.mkdir()
    assert not _pkl_exists(path)

  def test_manifest_only_is_not_available(self, tmp_path):
    path = tmp_path / 'model.pkl'
    Path(get_manifest_path(path)).write_text('2')
    assert not _pkl_exists(path)

  @parameterized.expand([0, 1, 2], names=['missing_chunk'])
  def test_missing_chunk_is_not_available(self, missing_chunk, tmp_path):
    path = tmp_path / 'model.pkl'
    write_chunks(path, num_chunks=3, missing=(missing_chunk,))
    assert not _pkl_exists(path)

  @parameterized.expand([1, 3], names=['num_chunks'])
  def test_all_chunks_are_available(self, num_chunks, tmp_path):
    path = tmp_path / 'model.pkl'
    write_chunks(path, num_chunks=num_chunks)
    assert _pkl_exists(path)

  def test_directory_is_not_a_chunk(self, tmp_path):
    path = tmp_path / 'model.pkl'
    write_chunks(path, missing=(1,))
    Path(get_chunk_name(path, 1, 2)).mkdir()
    assert not _pkl_exists(path)

  @parameterized.expand([b'', b'0', b'-2', b'two', b'2.0', b'1\n2', b'\xff'], names=['contents'])
  def test_invalid_manifest_is_not_available(self, contents, tmp_path):
    path = tmp_path / 'model.pkl'
    Path(get_manifest_path(path)).write_bytes(contents)
    assert not _pkl_exists(path)


# Pkl discovery

class TestFindDrivingPkl(OpenpilotTestCase):
  def setUp(self):
    super().setUp()
    self.enterContext(mock.patch.dict(os.environ, {'COMBINED_MODEL_PKL': ''}))

  @parameterized.expand([False, True], names=['chestnut'])
  def test_returns_none_when_no_bundle_or_bundled_pkl(self, chestnut, pkl_roots):
    assert _find_driving_pkl(None, chestnut=chestnut) is None

  @parameterized.expand([False, True], names=['chestnut'])
  def test_uses_bundled_pkl_when_no_bundle(self, chestnut, pkl_roots):
    bundled_root, _ = pkl_roots
    for filename in ('driving_tinygrad.pkl', 'big_driving_tinygrad.pkl'):
      (bundled_root / filename).write_bytes(b'complete')
    expected = stock_helpers.modeld_pkl_path(chestnut)
    assert _find_driving_pkl(None, chestnut=chestnut) == str(expected)

  def test_bundled_manifest_only_is_not_available(self, pkl_roots):
    bundled = stock_helpers.modeld_pkl_path(False)
    Path(get_manifest_path(bundled)).write_text('2')
    assert _find_driving_pkl(None) is None

  def test_bundled_missing_chunk_is_not_available(self, pkl_roots):
    bundled = stock_helpers.modeld_pkl_path(False)
    write_chunks(bundled, missing=(1,))
    assert _find_driving_pkl(None) is None

  @parameterized.expand([False, True], names=['chestnut'])
  def test_uses_complete_bundled_chunks(self, chestnut, pkl_roots):
    bundled = stock_helpers.modeld_pkl_path(chestnut)
    write_chunks(bundled)
    assert _find_driving_pkl(None, chestnut=chestnut) == str(bundled)

  def test_returns_none_when_no_models(self, pkl_roots):
    stock_helpers.modeld_pkl_path(False).write_bytes(b'complete')
    bundle = DummyBundle(models=[])
    assert _find_driving_pkl(bundle) is None

  def test_returns_none_when_selected_pkl_not_on_disk(self, pkl_roots):
    stock_helpers.modeld_pkl_path(False).write_bytes(b'complete')
    bundle = DummyBundle(models=[
      DummyModel('vision', 'driving_fof_tinygrad.pkl'),
      DummyModel('policy', 'driving_fof_tinygrad.pkl'),
    ])
    assert _find_driving_pkl(bundle) is None

  def test_finds_pkl_by_artifact_name(self, pkl_roots):
    _, download_root = pkl_roots
    expected = download_root / 'driving_fof_tinygrad.pkl'
    expected.write_bytes(b'complete')
    bundle = DummyBundle(models=[
      DummyModel('vision', 'driving_fof_tinygrad.pkl'),
      DummyModel('policy', 'driving_fof_tinygrad.pkl'),
    ])
    assert _find_driving_pkl(bundle) == str(expected)

  @parameterized.expand([[], [0], [1], [0, 1]], names=['missing_chunks'])
  def test_selected_pkl_requires_complete_chunks(self, missing_chunks, pkl_roots):
    _, download_root = pkl_roots
    selected = download_root / 'selected.pkl'
    write_chunks(selected, missing=missing_chunks)
    bundle = DummyBundle(models=[DummyModel('supercombo', selected.name)])
    expected = None if missing_chunks else str(selected)
    assert _find_driving_pkl(bundle) == expected

  @parameterized.expand([False, True], names=['has_bundle'])
  def test_override_precedes_selected_or_bundled_pkl(self, has_bundle, pkl_roots):
    bundled_root, download_root = pkl_roots
    stock_helpers.modeld_pkl_path(False).write_bytes(b'bundled')
    selected = download_root / 'selected.pkl'
    selected.write_bytes(b'selected')
    override = bundled_root.parent / 'override.pkl'
    override.write_bytes(b'override')
    bundle = DummyBundle(models=[DummyModel('supercombo', selected.name)]) if has_bundle else None
    with mock.patch.dict(os.environ, {'COMBINED_MODEL_PKL': str(override)}):
      assert _find_driving_pkl(bundle) == str(override)

  def test_complete_chunked_override_precedes_bundle(self, pkl_roots):
    bundled_root, download_root = pkl_roots
    selected = download_root / 'selected.pkl'
    selected.write_bytes(b'selected')
    override = bundled_root.parent / 'override.pkl'
    write_chunks(override)
    bundle = DummyBundle(models=[DummyModel('supercombo', selected.name)])
    with mock.patch.dict(os.environ, {'COMBINED_MODEL_PKL': str(override)}):
      assert _find_driving_pkl(bundle) == str(override)

  @parameterized.expand([(False, False), (False, True), (True, False), (True, True)],
                        names=['has_bundle', 'has_manifest'])
  def test_unavailable_override_does_not_hide_available_pkl(self, has_bundle, has_manifest, pkl_roots):
    bundled_root, download_root = pkl_roots
    bundled = stock_helpers.modeld_pkl_path(False)
    bundled.write_bytes(b'bundled')
    selected = download_root / 'selected.pkl'
    selected.write_bytes(b'selected')
    override = bundled_root.parent / 'override.pkl'
    if has_manifest:
      write_chunks(override, missing=(1,))
    bundle = DummyBundle(models=[DummyModel('supercombo', selected.name)]) if has_bundle else None
    with mock.patch.dict(os.environ, {'COMBINED_MODEL_PKL': str(override)}):
      assert _find_driving_pkl(bundle) == str(selected if has_bundle else bundled)


# Init — assertion guard

class TestModelStateCombinedInit(OpenpilotTestCase):
  def test_asserts_when_no_pkl(self, monkeypatch):
    bundle = DummyBundle(models=[], is_20hz=True)
    monkeypatch.setattr(helpers, 'get_active_bundle', lambda params=None, *, chestnut=None: bundle)
    monkeypatch.setattr(modeld_module, 'get_active_bundle', lambda params=None, *, chestnut=None: bundle)
    with self.assertRaisesRegex(AssertionError, "No driving pkl found"):
      ModelState(cam_w=CAM_W, cam_h=CAM_H)

  @parameterized.expand([False, True], names=['chunked'])
  def test_initializes_bundled_default_without_active_bundle(self, chunked, tmp_path, monkeypatch, patch_modeld):
    archetype = ARCHETYPES['vision_policy_split']
    bundled = tests_helpers.write_pkl(tmp_path, archetype).rename(tmp_path / 'driving_tinygrad.pkl')
    if chunked:
      chunk_file(bundled, get_chunk_targets(bundled, bundled.stat().st_size))
    monkeypatch.setattr(stock_helpers, 'MODELS_DIR', tmp_path)
    monkeypatch.setattr(modeld_module, 'COMMA_HARDWARE', False)
    patch_modeld(None)
    with mock.patch.dict(os.environ, {'COMBINED_MODEL_PKL': ''}):
      state = ModelState(cam_w=CAM_W, cam_h=CAM_H)
    assert state._combined_model_type == archetype.expected_model_type
    assert type(state.constants) is archetype.expected_constants_class
    assert state.generation is None

  def test_rejects_incomplete_bundled_default(self, tmp_path, monkeypatch, patch_modeld):
    monkeypatch.setattr(stock_helpers, 'MODELS_DIR', tmp_path)
    write_chunks(stock_helpers.modeld_pkl_path(False), missing=(1,))
    patch_modeld(None)
    with mock.patch.dict(os.environ, {'COMBINED_MODEL_PKL': ''}), self.assertRaisesRegex(AssertionError, "No driving pkl found"):
      ModelState(cam_w=CAM_W, cam_h=CAM_H)


class TestStockEquivalence(OpenpilotTestCase):

  def test_split_queue_keys_match_stock(self, model_state_factory):
    from openpilot.selfdrive.modeld.compile_modeld import make_input_queues
    from openpilot.sunnypilot.modeld_v2.compile_modeld import derive_frame_skip

    state = model_state_factory(ARCHETYPES['vision_policy_split'])

    frame_skip = derive_frame_skip(SPLIT_VISION_INPUT_SHAPES, SPLIT_POLICY_INPUT_SHAPES)
    stock_shapes = {**SPLIT_VISION_INPUT_SHAPES, **SPLIT_POLICY_INPUT_SHAPES, 'action_t': (1, 2)}
    stock_queues, stock_npy, _frame_views = make_input_queues(stock_shapes, frame_skip, device='NPY', frame_copy_size=49152)

    # sunnypilot split pipeline has tfm/big_tfm as queues (stock has them in npy only)
    assert set(stock_queues.keys()) <= set(state.input_queues.keys())
    assert {'desire', 'traffic_convention'} <= set(state.numpy_inputs.keys())

  def test_split_queue_keys_work_with_desire_key(self, model_state_factory):
    from openpilot.sunnypilot.modeld_v2.compile_modeld import derive_frame_skip, make_split_input_queues

    policy_shapes_desire = {'features_buffer': (1, 25, 512), 'desire': (1, 25, 8), 'traffic_convention': (1, 2)}
    frame_skip = derive_frame_skip(SPLIT_VISION_INPUT_SHAPES, policy_shapes_desire)
    queues, npy = make_split_input_queues(SPLIT_VISION_INPUT_SHAPES, policy_shapes_desire, frame_skip, device='NPY')

    assert 'desire_q' in queues
    assert 'desire' in npy
    assert 'img_q' in queues
    assert 'feat_q' in queues

  def test_split_vision_input_names_match_stock(self, model_state_factory):
    state = model_state_factory(ARCHETYPES['vision_policy_split'])
    assert state.vision_input_names == ['img', 'big_img']

  def test_split_output_slices_preserved(self, model_state_factory):
    arch = ARCHETYPES['vision_policy_split']
    state = model_state_factory(arch)
    assert state.vision_output_slices == arch.metadata_structure['vision']['output_slices']
    assert state.policy_output_slices == arch.metadata_structure['policy']['output_slices']

  def test_unified_run_model(self, tmp_path, monkeypatch, patch_modeld):
    from openpilot.common.hardware import hw
    from openpilot.selfdrive.modeld.helpers import dump_oob
    shapes = {'img': (1, 12, 128, 256), 'big_img': (1, 12, 128, 256), 'features_buffer': (1, 24, 32, 512),
              'desire_pulse': (1, 25, 8), 'traffic_convention': (1, 2), 'action_t': (1, 2)}
    pkl_data = {'metadata': {'model': {'input_shapes': shapes, 'output_slices': {}}},
                'run_model': {(CAM_W, CAM_H): tests_helpers._noop_jit}}
    with open(tmp_path / 'driving_test_tinygrad.pkl', 'wb') as f:
      dump_oob(pkl_data, f)
    bundle = DummyBundle(models=[DummyModel('supercombo', 'driving_test_tinygrad.pkl')])
    patch_modeld(bundle)
    monkeypatch.setattr(hw.Paths, 'model_root', staticmethod(lambda: str(tmp_path)))
    state = ModelState(cam_w=CAM_W, cam_h=CAM_H)
    assert state.is_run_model and state.run_model is not None
    assert state.run_policy is None and state.warp is None
    assert 'img' in state.frame_views and 'big_img' in state.frame_views


ARCHETYPE_NAMES = list(ARCHETYPES.keys())


class TestModelTypeDetection(OpenpilotTestCase):
  @parameterized.expand(ARCHETYPE_NAMES, names=["archetype_name"])
  def test_combined_model_type(self, archetype_name, model_state_factory):
    arch = ARCHETYPES[archetype_name]
    state = model_state_factory(arch)
    assert state._combined_model_type == arch.expected_model_type, \
      f"{arch.name}: got {state._combined_model_type}, expected {arch.expected_model_type}"


class TestConstantsSelection(OpenpilotTestCase):
  @parameterized.expand(ARCHETYPE_NAMES, names=["archetype_name"])
  def test_constants_class(self, archetype_name, model_state_factory):
    arch = ARCHETYPES[archetype_name]
    state = model_state_factory(arch)
    assert type(state.constants) is arch.expected_constants_class, \
      f"{arch.name}: got {type(state.constants).__name__}, expected {arch.expected_constants_class.__name__}"


class TestParserSelection(OpenpilotTestCase):
  @parameterized.expand(ARCHETYPE_NAMES, names=["archetype_name"])
  def test_parser_module(self, archetype_name, model_state_factory):
    arch = ARCHETYPES[archetype_name]
    state = model_state_factory(arch)
    parser_module = type(state.parser).__module__
    assert parser_module.endswith(arch.expected_parser_module), \
      f"{arch.name}: parser from {parser_module}, expected module ending with {arch.expected_parser_module}"


class TestDesireKeyDetection(OpenpilotTestCase):
  @parameterized.expand(ARCHETYPE_NAMES, names=["archetype_name"])
  def test_desire_key(self, archetype_name, model_state_factory):
    arch = ARCHETYPES[archetype_name]
    state = model_state_factory(arch)
    assert state.desire_key == arch.expected_desire_key, \
      f"{arch.name}: got {state.desire_key}, expected {arch.expected_desire_key}"


class TestVisionInputNames(OpenpilotTestCase):
  @parameterized.expand(ARCHETYPE_NAMES, names=["archetype_name"])
  def test_vision_names_contain_img(self, archetype_name, model_state_factory):
    arch = ARCHETYPES[archetype_name]
    state = model_state_factory(arch)
    assert len(state.vision_input_names) >= 1
    for name in state.vision_input_names:
      assert 'img' in name, f"{arch.name}: vision input name '{name}' missing 'img'"


class TestOutputSlices(OpenpilotTestCase):
  @parameterized.expand(ARCHETYPE_NAMES, names=["archetype_name"])
  def test_vision_slices_populated(self, archetype_name, model_state_factory):
    arch = ARCHETYPES[archetype_name]
    state = model_state_factory(arch)
    assert len(state.vision_output_slices) > 0, f"{arch.name}: vision_output_slices empty"

  @parameterized.expand(ARCHETYPE_NAMES, names=["archetype_name"])
  def test_policy_slices_match_type(self, archetype_name, model_state_factory):
    arch = ARCHETYPES[archetype_name]
    state = model_state_factory(arch)
    if arch.expected_model_type == 'supercombo':
      assert state.policy_output_slices == {}, f"{arch.name}: supercombo should have empty policy slices"
    else:
      assert len(state.policy_output_slices) > 0, f"{arch.name}: split/multi should have policy slices"


class TestInputQueueCreation(OpenpilotTestCase):
  @parameterized.expand(ARCHETYPE_NAMES, names=["archetype_name"])
  def test_queues_not_empty(self, archetype_name, model_state_factory):
    arch = ARCHETYPES[archetype_name]
    state = model_state_factory(arch)
    assert len(state.input_queues) > 0, f"{arch.name}: input_queues empty"

  @parameterized.expand(ARCHETYPE_NAMES, names=["archetype_name"])
  def test_npy_contains_transforms(self, archetype_name, model_state_factory):
    arch = ARCHETYPES[archetype_name]
    state = model_state_factory(arch)
    assert 'tfm' in state.numpy_inputs, f"{arch.name}: 'tfm' missing from npy"
    assert 'big_tfm' in state.numpy_inputs, f"{arch.name}: 'big_tfm' missing from npy"
    assert state.numpy_inputs['tfm'].shape == (3, 3)
    assert state.numpy_inputs['big_tfm'].shape == (3, 3)

  @parameterized.expand(ARCHETYPE_NAMES, names=["archetype_name"])
  def test_npy_contains_desire(self, archetype_name, model_state_factory):
    arch = ARCHETYPES[archetype_name]
    state = model_state_factory(arch)
    assert arch.expected_desire_key in state.numpy_inputs, \
      f"{arch.name}: '{arch.expected_desire_key}' missing from npy"


class TestFrameBufferParams(OpenpilotTestCase):
  @parameterized.expand(ARCHETYPE_NAMES, names=["archetype_name"])
  def test_frame_buf_params_per_vision_input(self, archetype_name, model_state_factory):
    arch = ARCHETYPES[archetype_name]
    state = model_state_factory(arch)
    for name in state.vision_input_names:
      assert name in state.frame_buf_params, f"{arch.name}: frame_buf_params missing '{name}'"
      nv12_info = state.frame_buf_params[name]
      assert len(nv12_info) >= 4, f"{arch.name}: nv12_info for '{name}' too short"


class TestBundleOverrides(OpenpilotTestCase):
  @parameterized.expand(ARCHETYPE_NAMES, names=["archetype_name"])
  def test_smoothing_params_from_overrides(self, archetype_name, model_state_factory):
    arch = ARCHETYPES[archetype_name]
    state = model_state_factory(arch)
    assert state.LAT_SMOOTH_SECONDS == 0.1
    assert state.LONG_SMOOTH_SECONDS == 0.3

  @parameterized.expand(ARCHETYPE_NAMES, names=["archetype_name"])
  def test_generation_from_bundle(self, archetype_name, model_state_factory):
    arch = ARCHETYPES[archetype_name]
    state = model_state_factory(arch)
    assert state.generation == 10


class TestMlsimProperty(OpenpilotTestCase):
  def test_mlsim_false_for_gen10(self, model_state_factory):
    state = model_state_factory(ARCHETYPES['supercombo_non20hz'])
    assert state.mlsim is False

  def test_mlsim_true_for_gen11(self, tmp_path, monkeypatch, patch_modeld):
    from openpilot.common.hardware import hw
    from openpilot.sunnypilot.modeld_v2.tests.helpers import write_pkl, ARCHETYPES as A

    arch = A['supercombo_non20hz']
    write_pkl(tmp_path, arch)
    bundle = DummyBundle(models=arch.model_stubs, is_20hz=arch.is_20hz, generation=11)
    patch_modeld(bundle)
    monkeypatch.setattr(hw.Paths, 'model_root', staticmethod(lambda: str(tmp_path)))

    state = ModelState(cam_w=CAM_W, cam_h=CAM_H)
    assert state.mlsim is True


class TestCrossArchetypeMismatch(OpenpilotTestCase):
  def test_wrong_is_20hz_changes_constants(self, tmp_path, monkeypatch, patch_modeld):
    from openpilot.common.hardware import hw
    from openpilot.sunnypilot.modeld_v2.tests.helpers import write_pkl
    from openpilot.sunnypilot.modeld_v2.constants import ModelConstants

    arch = ARCHETYPES['vision_policy_split']
    write_pkl(tmp_path, arch)
    bundle = DummyBundle(models=arch.model_stubs, is_20hz=False)
    patch_modeld(bundle)
    monkeypatch.setattr(hw.Paths, 'model_root', staticmethod(lambda: str(tmp_path)))

    state = ModelState(cam_w=CAM_W, cam_h=CAM_H)
    assert type(state.constants) is ModelConstants, \
      "Wrong is_20hz should produce wrong constants class"
