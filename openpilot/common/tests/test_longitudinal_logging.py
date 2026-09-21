import hashlib
import json
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings

from openpilot.common.longitudinal_logging import (
  CHECK_FLAGS, ObservationRecorder, UINT64_MAX, attach_diagnostics, can_digest, capture_inputs, configuration_digest, is_uint64,
  local_input, observe,
)
from openpilot.common import longitudinal_logging


class FakeParams:
  def as_builder(self):
    return self

  def to_bytes(self):
    return b"fixture-config"


class FakeDiagnostics:
  @staticmethod
  def new_message():
    return SimpleNamespace()


class TestLongitudinalLogging(unittest.TestCase):
  def recorder(self, **options):
    source = {"missing_source_files": [], "source_files": {"fixture": "1" * 64}, "imported_native_module_files": {},
              "missing_explicit_native_library_files": []}
    with patch("openpilot.common.longitudinal_logging.source_manifest", return_value=source):
      return ObservationRecorder("fixture", FakeParams(), FakeParams(), enabled=True, **options)

  def test_uint64_does_not_round_identity(self):
    for value in (0, (1 << 53) + 7, (1 << 63) + 9, UINT64_MAX):
      with self.subTest(value=value):
        self.assertTrue(is_uint64(value))
    for value in (-1, UINT64_MAX + 1, 1.5, True, None, "123"):
      with self.subTest(value=value):
        self.assertFalse(is_uint64(value))

  def test_uint64_fast_path_retains_integral_compatibility(self):
    import numpy as np
    from enum import IntEnum

    class FrameId(IntEnum):
      ZERO = 0
      ONE = 1

    class IntSubclass(int):
      pass

    for value in (np.uint64(UINT64_MAX), np.int64(42), FrameId.ZERO, FrameId.ONE, IntSubclass(1 << 63)):
      with self.subTest(value=repr(value)):
        self.assertTrue(is_uint64(value))
    for value in (np.int64(-1), np.float64(42), np.bool_(True), IntSubclass(UINT64_MAX + 1)):
      with self.subTest(value=repr(value)):
        self.assertFalse(is_uint64(value))

  def test_input_health_snapshot_is_refreshed_each_invocation(self):
    state = SimpleNamespace(logMonoTime={"carState": (1 << 63) + 1, "modelV2": 123},
                            **{name: {"carState": True, "modelV2": False} for name in CHECK_FLAGS})
    first = capture_inputs(state, ("carState", "modelV2"))
    state.logMonoTime["carState"] += 1
    for name in CHECK_FLAGS:
      setattr(state, name, {"carState": False, "modelV2": True})
    second = capture_inputs(state, ("carState", "modelV2"))
    self.assertEqual(first[0]["monoTime"], (1 << 63) + 1)
    self.assertEqual(second[0]["monoTime"], (1 << 63) + 2)
    self.assertEqual([item["flags"] for item in first], [sum(CHECK_FLAGS.values()), 0])
    self.assertEqual([item["flags"] for item in second], [0, sum(CHECK_FLAGS.values())])
    self.assertTrue(all(item["knownFlags"] == sum(CHECK_FLAGS.values()) for item in second))

  def test_absent_inputs_do_not_probe_health(self):
    class NoInputs:
      logMonoTime = {}

      def __getattr__(self, name):
        raise AssertionError(f"Absent input must not access {name}")

    self.assertEqual(capture_inputs(NoInputs(), ()), [])
    self.assertEqual(capture_inputs(NoInputs(), ("carState",)), [])

  def test_inputs_preserve_integer_ids_and_unknown_health(self):
    identity = (1 << 63) + 17
    state = SimpleNamespace(logMonoTime={"carState": identity}, valid={"carState": True})
    inputs = capture_inputs(state, ("carState", "absentService"))
    self.assertEqual(len(inputs), 1)
    self.assertEqual(inputs[0]["monoTime"], identity)
    self.assertTrue(inputs[0]["idKnown"])
    self.assertEqual(inputs[0]["knownFlags"], CHECK_FLAGS["valid"])
    self.assertEqual(inputs[0]["flags"], CHECK_FLAGS["valid"])
    self.assertEqual(inputs[0]["kind"], "consumerSnapshot")

  def test_unseen_and_invalid_ids_are_not_fabricated(self):
    state = SimpleNamespace(logMonoTime={"carState": 0, "modelV2": 1.5}, seen={"carState": False})
    inputs = capture_inputs(state, ("carState", "modelV2"))
    self.assertTrue(all(not entry["idKnown"] for entry in inputs))
    self.assertEqual(inputs[0]["knownFlags"], CHECK_FLAGS["seen"])
    self.assertEqual(inputs[1]["knownFlags"], 0)

  def test_health_is_known_only_for_raw_booleans_without_coercion(self):
    class PoisonTruth:
      def __bool__(self):
        raise AssertionError("Observation must not coerce health")

    for value in (0, 1, "false", "true", [], None, PoisonTruth()):
      with self.subTest(value_type=type(value).__name__):
        state = SimpleNamespace(logMonoTime={"carState": 123}, **{name: {"carState": value} for name in CHECK_FLAGS})
        item, = capture_inputs(state, ("carState",))
        self.assertEqual(item["knownFlags"], 0)
        self.assertEqual(item["flags"], 0)
        self.assertTrue(item["idKnown"])
    for value in (True, False):
      state = SimpleNamespace(logMonoTime={"carState": 123}, **{name: {"carState": value} for name in CHECK_FLAGS})
      item, = capture_inputs(state, ("carState",))
      self.assertEqual(item["knownFlags"], sum(CHECK_FLAGS.values()))
      self.assertEqual(item["flags"], sum(CHECK_FLAGS.values()) if value else 0)

  def test_local_input_does_not_invent_subscriber_health(self):
    source = local_input("carState", (1 << 53) + 1, False)
    self.assertTrue(source["idKnown"])
    self.assertEqual(source["kind"], "localPublished")
    self.assertEqual(source["knownFlags"], CHECK_FLAGS["valid"])
    self.assertEqual(source["flags"], 0)

  def test_sequences_session_and_periodic_manifest(self):
    clock_values = iter((100, 110, 200, 210, 300, 310))
    recorder = self.recorder(clock=lambda: next(clock_values), manifest_interval=3)
    contexts = [recorder.context(recorder.finish(recorder.begin(frame))) for frame in range(3)]
    self.assertEqual([context["sequence"] for context in contexts], [1, 2, 3])
    self.assertEqual([context["frame"] for context in contexts], [0, 1, 2])
    self.assertTrue(all(context["frameKnown"] for context in contexts))
    self.assertEqual([bool(context["manifest"]) for context in contexts], [True, False, True])
    self.assertEqual(contexts[0]["identity"], hashlib.sha256(contexts[0]["manifest"].encode()).hexdigest())
    self.assertTrue(json.loads(contexts[0]["manifest"])["complete"])
    self.assertEqual(len({context["session"] for context in contexts}), 1)
    self.assertNotEqual(recorder.session, self.recorder().session)

  def test_disabled_logging_does_not_read_files_or_clock(self):
    with patch("openpilot.common.longitudinal_logging.source_manifest", side_effect=AssertionError("not called")):
      recorder = ObservationRecorder("fixture", enabled=False, clock=lambda: self.fail("clock called"))
    self.assertIsNone(recorder.begin())
    self.assertIsNone(recorder.finish(None))
    self.assertEqual(recorder.sequence, 0)
    self.assertEqual(recorder.manifest, "")

  def test_missing_config_is_explicit(self):
    with patch("openpilot.common.longitudinal_logging.source_manifest", return_value={
      "missing_source_files": [], "imported_native_module_files": {}, "missing_explicit_native_library_files": [],
    }):
      recorder = ObservationRecorder("fixture", enabled=True)
    manifest = json.loads(recorder.manifest)
    self.assertFalse(manifest["complete"])
    self.assertEqual(manifest["configuration"], {"carParams": None, "carParamsSP": None})
    self.assertEqual(manifest["configuration_encoding"], {"carParams": None, "carParamsSP": None})

  def test_dataclass_configuration_is_deterministic_and_not_mutated(self):
    @dataclass
    class Nested:
      enabled: bool = False

    @dataclass
    class Configuration:
      flags: int = 1 << 63
      nested: Nested = field(default_factory=Nested)

    original = Configuration()
    first = configuration_digest(original)
    self.assertEqual(len(first), 64)
    self.assertEqual(first, configuration_digest(Configuration()))
    self.assertEqual(original, Configuration())
    original.nested.enabled = True
    self.assertNotEqual(first, configuration_digest(original))
    self.assertIsNone(configuration_digest(Configuration))

  def test_dataclass_nonfinite_config_is_unknown(self):
    @dataclass
    class Configuration:
      value: float = float("nan")

    self.assertIsNone(configuration_digest(Configuration()))

  def test_real_capnp_builder_and_reader_use_private_copies(self):
    from opendbc.car.structs import car
    from openpilot.cereal import custom

    for schema in (car.CarParams, custom.CarParamsSP):
      with self.subTest(schema=schema):
        builder = schema.new_message()
        before = builder.to_dict()
        with warnings.catch_warnings():
          warnings.simplefilter("error")
          digest = configuration_digest(builder)
          self.assertIsNotNone(digest)
          self.assertEqual(digest, configuration_digest(builder.as_reader()))
          self.assertEqual(digest, configuration_digest(builder))
          builder.to_bytes()
        self.assertEqual(builder.to_dict(), before)

  def test_runtime_card_configuration_manifest_is_complete(self):
    from opendbc.car import structs

    params = structs.CarParams.new_message()
    params_sp = structs.CarParamsSP()
    source = {"missing_source_files": [], "source_files": {}, "imported_native_module_files": {},
              "missing_explicit_native_library_files": []}
    with patch("openpilot.common.longitudinal_logging.source_manifest", return_value=source):
      recorder = ObservationRecorder("card", params, params_sp, enabled=True)
    manifest = json.loads(recorder.manifest)
    self.assertTrue(manifest["complete"])
    self.assertEqual(manifest["configuration_encoding"], {
      "carParams": "capnp-copy-bytes-v1", "carParamsSP": "dataclass-json-v1",
    })
    self.assertNotIn("fields", manifest)

  def test_clock_failure_and_backwards_clock_are_observation_errors(self):
    values = iter((RuntimeError("clock"), 100, 90))

    def clock():
      value = next(values)
      if isinstance(value, Exception):
        raise value
      return value

    recorder = self.recorder(clock=clock)
    self.assertIsNone(recorder.begin())
    self.assertIsNone(recorder.finish(recorder.begin()))
    self.assertEqual(recorder.errors, 2)
    self.assertEqual(recorder.sequence, 2)

  def test_attach_is_atomic_and_fail_open(self):
    recorder = self.recorder(clock=lambda: 100)
    sample = recorder.finish(recorder.begin(0))
    message = SimpleNamespace(accel=-1.25, valid=True)
    self.assertTrue(attach_diagnostics(message, "diagnostics", FakeDiagnostics, recorder, sample, {"status": "tracking"}))
    self.assertEqual(message.accel, -1.25)
    self.assertTrue(message.valid)
    previous = message.diagnostics
    with patch.object(FakeDiagnostics, "new_message", side_effect=RuntimeError("serializer")):
      self.assertFalse(attach_diagnostics(message, "diagnostics", FakeDiagnostics, recorder, sample, {"status": "bad"}))
    self.assertIs(message.diagnostics, previous)
    self.assertEqual(recorder.errors, 1)

  def test_context_creation_failure_keeps_original_message(self):
    recorder = self.recorder(clock=lambda: 100)
    sample = recorder.finish(recorder.begin())
    message = SimpleNamespace(accel=0.12)
    with patch.object(recorder, "context", side_effect=RuntimeError("context")):
      self.assertFalse(attach_diagnostics(message, "diagnostics", FakeDiagnostics, recorder, sample, {}))
    self.assertEqual(vars(message), {"accel": 0.12})

  def test_observe_isolates_a_broken_observer(self):
    recorder = self.recorder()

    def broken():
      raise RuntimeError("observer")

    observer = SimpleNamespace(recorder=recorder, start=broken)
    self.assertIsNone(observe(observer, "start"))
    self.assertEqual(recorder.errors, 1)

    class BrokenProperty:
      @property
      def recorder(self):
        raise RuntimeError("property")

    self.assertIsNone(observe(BrokenProperty(), "start"))
    self.assertIsNone(observe(None, "start"))

  def test_can_digest_binds_order_bus_address_and_payload(self):
    original = [(0x1DF, b"12345678", 1), (0x130, b"abcdefgh", 0)]
    digest = can_digest(original)
    self.assertEqual(digest, can_digest(list(original)))
    variants = (
      original[::-1],
      [(0x1DF, b"12345678", 0), original[1]],
      [(0x1DE, b"12345678", 1), original[1]],
      [(0x1DF, b"12345679", 1), original[1]],
      original[:1],
    )
    self.assertTrue(all(can_digest(variant) != digest for variant in variants))
    self.assertEqual(can_digest([]), hashlib.sha256(b"").hexdigest())

  def test_explicit_cdll_and_parameter_table_changes_have_distinct_disk_identities(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      common = root / "openpilot/common"
      common.mkdir(parents=True)
      header = common / "params_keys.h"
      library = common / "libparams_c.so"
      header.write_bytes(b"header-v1")
      library.write_bytes(b"cdll-v1")
      with patch.object(longitudinal_logging, "__file__", str(common / "longitudinal_logging.py")), \
           patch.object(longitudinal_logging, "SOURCE_PATHS", ("openpilot/common/params_keys.h",)), \
           patch.object(longitudinal_logging, "EXPLICIT_NATIVE_PATHS", ("openpilot/common/libparams_c.so",)):
        try:
          longitudinal_logging.source_manifest.cache_clear()
          before = longitudinal_logging.source_manifest()
          header.write_bytes(b"header-v2")
          library.write_bytes(b"cdll-v2")
          # The process-start snapshot remains immutable until a new startup.
          self.assertEqual(longitudinal_logging.source_manifest(), before)
          longitudinal_logging.source_manifest.cache_clear()
          after = longitudinal_logging.source_manifest()
        finally:
          longitudinal_logging.source_manifest.cache_clear()
      self.assertEqual(before["version"], 2)
      self.assertNotEqual(before["source_files"], after["source_files"])
      self.assertNotEqual(before["explicit_native_library_files"], after["explicit_native_library_files"])
      self.assertEqual(before["missing_explicit_native_library_files"], [])
      self.assertIn("not loaded-code or firmware attestation", before["scope"])

  def test_missing_or_escaping_explicit_native_file_cannot_claim_complete_identity(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / "tree"
      common = root / "openpilot/common"
      common.mkdir(parents=True)
      outside = Path(directory) / "outside.so"
      outside.write_bytes(b"outside")
      (common / "libparams_c.so").symlink_to(outside)
      expected_missing = ["openpilot/common/libparams_c.so", "missing.so"]
      with patch.object(longitudinal_logging, "__file__", str(common / "longitudinal_logging.py")), \
           patch.object(longitudinal_logging, "SOURCE_PATHS", ()), \
           patch.object(longitudinal_logging, "EXPLICIT_NATIVE_PATHS", tuple(expected_missing)):
        try:
          longitudinal_logging.source_manifest.cache_clear()
          source = longitudinal_logging.source_manifest()
          self.assertEqual(source["missing_explicit_native_library_files"], expected_missing)
          self.assertEqual(source["explicit_native_library_files"], {})
          recorder = ObservationRecorder("fixture", FakeParams(), FakeParams(), enabled=True)
        finally:
          longitudinal_logging.source_manifest.cache_clear()
      self.assertFalse(json.loads(recorder.manifest)["complete"])

  def test_manifest_covers_state_machine_and_non_python_longitudinal_dependencies(self):
    required_source = {
      "openpilot/common/params_keys.h", "openpilot/common/params_c.cc", "openpilot/common/params.py",
      "openpilot/selfdrive/selfdrived/longitudinal_observation.py", "openpilot/selfdrive/selfdrived/state.py",
      "openpilot/selfdrive/selfdrived/events.py", "openpilot/sunnypilot/selfdrive/selfdrived/events_base.py",
    }
    self.assertTrue(required_source.issubset(longitudinal_logging.SOURCE_PATHS))
    self.assertIn("openpilot/common/libparams_c.so", longitudinal_logging.EXPLICIT_NATIVE_PATHS)
    self.assertTrue(any(path.endswith("libacados_ocp_solver_long.so") for path in longitudinal_logging.EXPLICIT_NATIVE_PATHS))
    self.assertTrue(any(path.endswith("libqpOASES_e.so.3.1") for path in longitudinal_logging.EXPLICIT_NATIVE_PATHS))

  def test_manifest_covers_current_longitudinal_decisions_and_vehicle_estimate(self):
    required_source = {
      "openpilot/selfdrive/controls/lib/cruise_response.py",
      "openpilot/selfdrive/controls/lib/launch_response.py",
      "openpilot/selfdrive/controls/lib/model_reference.py",
      "openpilot/selfdrive/controls/lib/reference_timing.py",
      "openpilot/common/pid.py", "openpilot/common/realtime.py", "openpilot/common/constants.py",
      "openpilot/selfdrive/modeld/constants.py", "openpilot/selfdrive/car/cruise.py",
      "opendbc_repo/opendbc/car/honda/carcontroller.py", "opendbc_repo/opendbc/car/honda/hondacan.py",
      "opendbc_repo/opendbc/car/honda/values.py", "opendbc_repo/opendbc/car/honda/interface.py",
      "opendbc_repo/opendbc/car/honda/carstate.py", "opendbc_repo/opendbc/car/interfaces.py",
      "opendbc_repo/opendbc/car/common/simple_kalman.py", "opendbc_repo/opendbc/can/parser.py",
      "opendbc_repo/opendbc/can/packer.py",
    }
    self.assertTrue(required_source.issubset(longitudinal_logging.SOURCE_PATHS))

  def test_manifest_covers_actual_accord_dbc_generator_inputs(self):
    from opendbc.car import Bus
    from opendbc.car.honda.values import CAR, DBC

    root = Path(longitudinal_logging.__file__).resolve().parents[2]
    template_dir = Path("opendbc_repo/opendbc/dbc/generator/honda")
    template = template_dir / f"{DBC[CAR.HONDA_ACCORD][Bus.pt].removesuffix('_generated')}.dbc"
    imports = re.findall(r'^CM_ "IMPORT (.*?)";', (root / template).read_text(), re.MULTILINE)
    self.assertTrue(imports)
    required_source = {
      "opendbc_repo/opendbc/__init__.py", "opendbc_repo/opendbc/can/dbc.py",
      "opendbc_repo/opendbc/dbc/generator/generator.py", str(template),
      *(str(template_dir / name) for name in imports),
    }
    self.assertTrue(required_source.issubset(longitudinal_logging.SOURCE_PATHS))

  def test_decision_source_and_dbc_template_changes_change_startup_identity(self):
    representatives = (
      "openpilot/selfdrive/controls/lib/cruise_response.py",
      "openpilot/selfdrive/controls/lib/model_reference.py",
      "opendbc_repo/opendbc/dbc/generator/honda/_bosch_radar_acc.dbc",
    )
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      common = root / "openpilot/common"
      common.mkdir(parents=True)
      for relative in representatives:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"selected-input-v1")
      with patch.object(longitudinal_logging, "__file__", str(common / "longitudinal_logging.py")), \
           patch.object(longitudinal_logging, "EXPLICIT_NATIVE_PATHS", ()):
        try:
          for relative in representatives:
            with self.subTest(source=relative):
              longitudinal_logging.source_manifest.cache_clear()
              before = ObservationRecorder("fixture", FakeParams(), FakeParams(), enabled=True)
              before_sources = json.loads(before.manifest)["source"]["source_files"]
              self.assertIn(relative, before_sources)
              (root / relative).write_bytes(b"selected-input-v2")
              cached = ObservationRecorder("fixture", FakeParams(), FakeParams(), enabled=True)
              self.assertEqual(cached.identity, before.identity)
              longitudinal_logging.source_manifest.cache_clear()
              after = ObservationRecorder("fixture", FakeParams(), FakeParams(), enabled=True)
              after_sources = json.loads(after.manifest)["source"]["source_files"]
              self.assertNotEqual(after.identity, before.identity)
              self.assertEqual([path for path in before_sources if before_sources[path] != after_sources[path]], [relative])
              self.assertIn("not loaded-code or firmware attestation", json.loads(after.manifest)["source"]["scope"])
        finally:
          longitudinal_logging.source_manifest.cache_clear()


if __name__ == "__main__":
  unittest.main()
