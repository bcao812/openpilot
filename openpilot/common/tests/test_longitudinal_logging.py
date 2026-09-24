import hashlib
import json
import re
from dataclasses import dataclass, field
from itertools import product
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import warnings

from openpilot.common.longitudinal_logging import (
  CHECK_FLAGS, InputSnapshotCache, ObservationRecorder, UINT64_MAX, attach_diagnostics, can_digest, capture_inputs, configuration_digest, is_uint64,
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


def attach_diagnostics_reference(message, field, factory, recorder, sample, values):
  # Pre-optimization implementation, kept as an independent wire/error oracle.
  if recorder is None or not recorder.enabled or sample is None:
    return False
  try:
    diagnostics = factory.new_message()
    for name, value in values.items():
      setattr(diagnostics, name, value)
    diagnostics.context = recorder.context(sample)
    setattr(message, field, diagnostics)
    return True
  except Exception:
    recorder.record_error()
    return False


def diagnostic_cases():
  from opendbc.car.structs import car
  from openpilot.cereal import log

  def inputs():
    return [
      {"service": "carState", "monoTime": (1 << 63) + 17, "idKnown": True,
       "knownFlags": 31, "flags": 27, "kind": "consumerSnapshot"},
      {"service": "modelV2", "monoTime": 0, "idKnown": False,
       "knownFlags": 0, "flags": 0, "kind": "consumerSnapshot"},
      {"service": "carControl", "monoTime": UINT64_MAX, "idKnown": True,
       "knownFlags": 1, "flags": 0, "kind": "localPublished"},
    ]

  return (
    (log.LongitudinalPlan, "diagnostics", log.LongitudinalPlanDiagnostics, {"aTarget": -1.25}, {
      "inputs": inputs(), "unified": True, "updateCompleted": True, "outcome": "completed",
      "mpcUpdated": True, "solverAttempted": True, "solverCalls": UINT64_MAX, "solverResets": 7,
      "solutionStatus": -1, "solverReturnKnown": False, "solverReturnStatus": 0,
      "initialStateKnown": True, "initialSpeed": 2.5, "initialAccel": float("nan"),
      "previousRequest": -0.0, "sampledAccelKnown": True, "sampledAccel": -1.125,
      "pitchBiasBefore": float("-inf"), "pitchBiasAfter": float("inf"), "pitchCalls": 13,
      "detailedTraceEnabled": True, "traceSequence": (1 << 53) + 7, "traceDropped": 2,
      "analysis": {"version": 1, "planningMode": 2, "decisionMonoTimeKnown": True, "decisionMonoTime": UINT64_MAX,
                   "passiveBefore": {"known": True, "consumerMonoTime": UINT64_MAX, "pitchFilterValue": float("nan")},
                   "referenceCandidates": {"known": True, "mpcSource": "lead0", "mpcAccel": -1.25},
                   "obstacleSummary": {"known": True, "times": [0.0, 0.2, 0.8], "sources": ["lead0", "lead1", "cruise"],
                                       "distances": [12.0, float("inf"), float("nan")], "sourceRule": "fixture"}},
    }),
    (log.ControlsState, "longitudinalDiagnostics", log.LongitudinalControlDiagnostics, {"forceDecel": True}, {
      "inputs": inputs(), "carControlMonoTime": UINT64_MAX, "controllerCalled": True, "targetAccel": -1.25,
      "controllerAccel": float("nan"), "publishedAccel": -0.0, "accelMin": float("-inf"), "accelMax": float("inf"),
      "stateBefore": "pid", "stateAfter": "stopping", "longActiveAtController": True,
      "longActivePublished": False, "resumePublished": True, "nonfiniteActuators": True, "unified": True,
      "decisionMonoTime": (1 << 63) + 11, "decisionMonoTimeKnown": True,
      "analysis": {"version": 1, "motionInputsKnown": True, "planValid": True, "brakingPlanValid": False,
                   "motionBefore": {"known": True, "phase": "tracking", "ids": inputs(), "creepT": float("nan")},
                   "motionAfter": {"known": True, "phase": "stopping", "reasons": 3, "rearmAfterFault": True},
                   "pose": {"known": True, "inputs": inputs(), "calibrationKnown": False, "deviceTimestamp": UINT64_MAX},
                   "forceDecelKnown": True, "forceDecel": True},
    }),
    (log.SelfdriveState, "longitudinalDiagnostics", log.LongitudinalSelfdriveDiagnostics, {"enabled": True}, {
      "inputs": inputs(), "version": 1, "carStateReused": True, "stateMachineInputKnown": True, "stateMachineCalled": True,
      "stateBefore": "enabled", "stateAfter": "softDisabling", "enabledBefore": True, "activeBefore": True,
      "enabledAfter": False, "activeAfter": False, "softDisableTimerBefore": (1 << 32) - 1, "softDisableTimerAfter": 7,
      "events": [0, 1, (1 << 16) - 1], "eventsSP": [7, 9], "currentAlertTypes": ["warning", "softDisable"],
      "motionCheckKnown": True, "motionVersion": 1, "motionDegraded": True, "motionWaitingForFreshInput": True,
      "motionCommIssueAdded": True, "controlsForceDecel": True, "controlsForceDecelKnown": True,
      "statePublished": "softDisabling", "enabledPublished": True, "activePublished": False,
    }),
    (car.CarOutput, "longitudinalDiagnostics", car.LongitudinalApplyDiagnostics, {"actuatorsOutput": {"accel": -1.25}}, {
      "inputs": inputs(), "hasApply": True, "applySequence": UINT64_MAX, "applyFrame": 123, "controllerFrame": 456,
      "controllerFrameKnown": True, "carControlMonoTime": UINT64_MAX, "carControlSPMonoTime": (1 << 63) + 9,
      "carStateMonoTime": (1 << 53) + 3, "applyStartMonoTime": 100, "applyEndMonoTime": 110, "sendcanMonoTime": 120,
      "canCount": 2, "canDigest": "a" * 64, "requestedAccel": float("nan"), "appliedAccel": -0.0,
      "requestedLongActive": True, "appliedLongActive": False, "requestedResume": True, "appliedResume": False,
      "authorityRestricted": True, "authorityReason": "gasPressed+brakePressed", "appliedLongControlState": "off",
      "lastDisposition": "applying", "dispositionFrame": 123, "dispositionMonoTime": 90, "initialized": True, "passive": False,
    }),
  )


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

  def test_all_health_bit_combinations_and_seen_zero_identity(self):
    for health in product((True, False, None), repeat=len(CHECK_FLAGS)):
      with self.subTest(health=health):
        state = SimpleNamespace(logMonoTime={"carState": 0},
                                **{name: {"carState": value} for name, value in zip(CHECK_FLAGS, health, strict=True)})
        item, = capture_inputs(state, ("carState",))
        expected_flags = sum(flag for flag, value in zip(CHECK_FLAGS.values(), health, strict=True) if value is True)
        expected_known = sum(flag for flag, value in zip(CHECK_FLAGS.values(), health, strict=True) if value is not None)
        self.assertEqual(item["flags"], expected_flags)
        self.assertEqual(item["knownFlags"], expected_known)
        self.assertEqual(item["idKnown"], bool(expected_flags & CHECK_FLAGS["seen"]))

  def test_health_membership_changes_and_service_order_are_not_cached(self):
    state = SimpleNamespace(logMonoTime={"carState": 0, "modelV2": UINT64_MAX}, valid={"modelV2": True},
                            seen={"carState": True}, updated={"modelV2": False})
    services = ("modelV2", "absent", "carState", "modelV2")
    first = capture_inputs(state, iter(services))
    self.assertEqual([item["service"] for item in first], ["modelV2", "carState", "modelV2"])
    self.assertEqual([item["flags"] for item in first], [1, 8, 1])
    self.assertEqual([item["knownFlags"] for item in first], [17, 8, 17])
    self.assertTrue(first[1]["idKnown"])

    state.valid["modelV2"] = False
    state.seen.pop("carState")
    del state.updated
    state.freq_ok = {"carState": True}
    state.logMonoTime["absent"] = 123
    second = capture_inputs(state, iter(services))
    self.assertEqual([item["service"] for item in second], list(services))
    self.assertEqual([item["flags"] for item in second], [0, 0, 4, 0])
    self.assertEqual([item["knownFlags"] for item in second], [1, 0, 4, 1])
    self.assertFalse(second[2]["idKnown"])
    self.assertEqual([item["flags"] for item in first], [1, 8, 1])

  def test_health_values_never_invoke_bool_equality_or_integer_coercion(self):
    import numpy as np

    class PoisonHealth:
      def __bool__(self):
        raise AssertionError("health __bool__")

      def __eq__(self, other):
        raise AssertionError("health __eq__")

      def __int__(self):
        raise AssertionError("health __int__")

    for value in (np.bool_(True), np.int64(1), np.array([True, False]), float("nan"), PoisonHealth()):
      with self.subTest(value_type=type(value).__name__):
        state = SimpleNamespace(logMonoTime={"carState": 0}, **{name: {"carState": value} for name in CHECK_FLAGS})
        item, = capture_inputs(state, ("carState",))
        self.assertEqual(item["knownFlags"], 0)
        self.assertEqual(item["flags"], 0)
        self.assertFalse(item["idKnown"])

  def test_health_getter_order_and_failed_snapshot_isolation(self):
    accesses = []

    class HealthSource(dict):
      def __init__(self, name):
        self.name = name

      def get(self, service):
        accesses.append((self.name, service))
        if self.name == "updated" and service == "modelV2":
          raise RuntimeError("health source failed")
        return True

    state = SimpleNamespace(logMonoTime={"carState": 1, "modelV2": 2},
                            **{name: HealthSource(name) for name in CHECK_FLAGS})
    recorder = self.recorder()
    observer = SimpleNamespace(recorder=recorder, capture=lambda: capture_inputs(state, ("carState", "modelV2")))
    self.assertIsNone(observe(observer, "capture"))
    self.assertEqual(recorder.errors, 1)
    self.assertEqual(accesses, [(name, service) for service in ("carState", "modelV2") for name in CHECK_FLAGS])

  def test_cached_inputs_refresh_every_health_field_at_the_same_timestamp(self):
    cache = InputSnapshotCache()
    state = SimpleNamespace(logMonoTime={"carState": UINT64_MAX},
                            **{name: {"carState": True} for name in CHECK_FLAGS})
    original, = capture_inputs(state, ("carState",), cache=cache)
    original_value = original.to_dict()
    repeated, = capture_inputs(state, ("carState",), cache=cache)
    self.assertIs(repeated, original)
    previous = original
    for name in CHECK_FLAGS:
      for value in (False, None, True):
        with self.subTest(health=name, value=value):
          setattr(state, name, {"carState": value})
          current, = capture_inputs(state, ("carState",), cache=cache)
          expected, = capture_inputs(state, ("carState",))
          self.assertEqual(current.to_dict(), expected)
          self.assertIsNot(current, previous)
          self.assertEqual(original.to_dict(), original_value)
          previous = current

    # A detached mutable copy cannot change a reader kept by an earlier frame.
    detached = original.as_builder()
    detached.monoTime = 1
    detached.flags = 0
    self.assertEqual(original.to_dict(), original_value)
    with self.assertRaises(AttributeError):
      original.monoTime = 1

  def test_cached_inputs_preserve_all_health_combinations_and_exact_ids(self):
    import numpy as np

    cache = InputSnapshotCache()
    for mono in (0, (1 << 53) + 7, (1 << 63) + 9, UINT64_MAX, np.uint64(UINT64_MAX),
                 -1, UINT64_MAX + 1, 1.5, True, None, "123"):
      for health in product((True, False, None), repeat=len(CHECK_FLAGS)):
        with self.subTest(mono=mono, health=health):
          state = SimpleNamespace(logMonoTime={"carState": mono},
                                  **{name: {"carState": value} for name, value in zip(CHECK_FLAGS, health, strict=True)})
          actual, = capture_inputs(state, ("carState",), cache=cache)
          expected, = capture_inputs(state, ("carState",))
          self.assertEqual(actual.to_dict(), expected)

  def test_cached_inputs_never_compare_or_coerce_raw_unknown_health(self):
    import numpy as np

    class PoisonHealth:
      def __bool__(self):
        raise AssertionError("health __bool__")

      def __eq__(self, other):
        raise AssertionError("health __eq__")

      def __int__(self):
        raise AssertionError("health __int__")

    cache = InputSnapshotCache()
    previous = None
    for value in (None, 0, 1, "true", [], np.bool_(True), np.array([True, False]), float("nan"), PoisonHealth()):
      state = SimpleNamespace(logMonoTime={"carState": 0}, **{name: {"carState": value} for name in CHECK_FLAGS})
      actual, = capture_inputs(state, ("carState",), cache=cache)
      self.assertEqual((actual.monoTime, actual.idKnown, actual.knownFlags, actual.flags), (0, False, 0, 0))
      if previous is not None:
        self.assertIs(actual, previous)
      previous = actual

  def test_cached_inputs_follow_service_order_replacement_and_prune_removed_services(self):
    cache = InputSnapshotCache()
    state = SimpleNamespace(logMonoTime={"carState": 0, "modelV2": UINT64_MAX},
                            valid={"modelV2": True}, seen={"carState": True})
    services = ("modelV2", "absent", "carState", "modelV2")
    first = capture_inputs(state, iter(services), cache=cache)
    self.assertEqual([entry.service for entry in first], ["modelV2", "carState", "modelV2"])
    self.assertEqual(set(cache._entries), {"carState", "modelV2"})

    state.logMonoTime = {"absent": (1 << 63) + 1, "carState": 17}
    state.valid = {"carState": False}
    del state.seen
    state.updated = {"absent": True}
    second = capture_inputs(state, iter(services), cache=cache)
    self.assertEqual([entry.to_dict() for entry in second], capture_inputs(state, iter(services)))
    self.assertEqual([entry.service for entry in second], ["absent", "carState"])
    self.assertEqual(set(cache._entries), {"absent", "carState"})
    self.assertEqual(first[1].monoTime, 0)
    self.assertTrue(first[1].idKnown)

    # A stream of new names cannot accumulate old input arenas in the cache.
    for index in range(100):
      service = f"input{index}"
      state.logMonoTime = {service: index}
      capture_inputs(state, (service,), cache=cache)
      self.assertEqual(set(cache._entries), {service})
    capture_inputs(state, (), cache=cache)
    self.assertEqual(cache._entries, {})

  def test_cached_absent_inputs_do_not_probe_health(self):
    class NoInputs:
      logMonoTime = {}

      def __getattr__(self, name):
        raise AssertionError(f"Absent input must not access {name}")

    cache = InputSnapshotCache()
    capture_inputs(SimpleNamespace(logMonoTime={"carState": 1}), ("carState",), cache=cache)
    self.assertEqual(capture_inputs(NoInputs(), ("carState",), cache=cache), [])
    self.assertEqual(cache._entries, {})

  def test_cached_capture_failure_preserves_previous_entries_and_health_getter_order(self):
    cache = InputSnapshotCache()
    state = SimpleNamespace(logMonoTime={"carState": 1, "modelV2": 2},
                            **{name: {"carState": False, "modelV2": False} for name in CHECK_FLAGS})
    before = capture_inputs(state, tuple(state.logMonoTime), cache=cache)
    previous = cache._entries
    accesses = []

    class HealthSource:
      def __init__(self, name):
        self.name = name

      def get(self, service):
        accesses.append((self.name, service))
        if self.name == "updated" and service == "modelV2":
          raise RuntimeError("health source failed")
        return True

    for name in CHECK_FLAGS:
      setattr(state, name, HealthSource(name))
    recorder = self.recorder()
    observer = SimpleNamespace(recorder=recorder, capture=lambda: capture_inputs(state, tuple(state.logMonoTime), cache=cache))
    self.assertIsNone(observe(observer, "capture"))
    self.assertEqual(recorder.errors, 1)
    self.assertIs(cache._entries, previous)
    self.assertEqual([entry.flags for entry in before], [0, 0])
    self.assertEqual(accesses, [(name, service) for service in ("carState", "modelV2") for name in CHECK_FLAGS])

    for name in CHECK_FLAGS:
      setattr(state, name, {"carState": True, "modelV2": True})
    with patch.object(cache, "_new_message", Mock(side_effect=RuntimeError("input serializer"))):
      self.assertIsNone(observe(observer, "capture"))
    self.assertEqual(recorder.errors, 2)
    self.assertIs(cache._entries, previous)
    after = observe(observer, "capture")
    self.assertEqual([entry.flags for entry in after], [31, 31])
    self.assertEqual([entry.flags for entry in before], [0, 0])

  def test_cached_non_string_services_keep_dictionary_and_serialization_semantics(self):
    from openpilot.cereal import log

    cache = InputSnapshotCache()
    state = SimpleNamespace(logMonoTime={123: 17}, valid={123: True})
    inputs = capture_inputs(state, (123,), cache=cache)
    self.assertEqual(inputs, capture_inputs(state, (123,)))
    self.assertEqual(cache._entries, {})
    recorder = self.recorder(clock=lambda: 100)
    sample = recorder.finish(recorder.begin())
    actual = log.ControlsState.new_message(forceDecel=True)
    before = actual.as_reader().as_builder().to_bytes()
    self.assertFalse(attach_diagnostics(actual, "longitudinalDiagnostics", log.LongitudinalControlDiagnostics,
                                      recorder, sample, {"inputs": inputs}))
    self.assertEqual(actual.as_reader().as_builder().to_bytes(), before)
    self.assertEqual(recorder.errors, 1)

  def test_cached_inputs_match_dictionary_fields_and_real_diagnostic_wire(self):
    cache = InputSnapshotCache()
    state = SimpleNamespace(logMonoTime={"carState": UINT64_MAX, "modelV2": 0},
                            valid={"carState": True, "modelV2": False}, seen={"modelV2": True})
    services = ("modelV2", "absent", "carState", "modelV2")
    for flags in (True, True, False, None):
      state.updated = {"carState": flags, "modelV2": flags}
      cached = [local_input("carControl", (1 << 63) + 9, False), *capture_inputs(state, services, cache=cache)]
      dictionaries = [local_input("carControl", (1 << 63) + 9, False), *capture_inputs(state, services)]
      for parent_factory, field_name, factory, parent_values, values in diagnostic_cases():
        with self.subTest(schema=factory, flags=flags):
          recorder = self.recorder(clock=lambda: 100)
          sample = recorder.finish(recorder.begin())
          actual = parent_factory.new_message(**parent_values)
          expected = parent_factory.new_message(**parent_values)
          self.assertTrue(attach_diagnostics(actual, field_name, factory, recorder, sample, {**values, "inputs": cached}))
          self.assertTrue(attach_diagnostics_reference(expected, field_name, factory, recorder, sample, {**values, "inputs": dictionaries}))
          self.assertEqual(json.dumps(actual.to_dict(), sort_keys=True), json.dumps(expected.to_dict(), sort_keys=True))
          self.assertEqual(actual.to_bytes(), expected.to_bytes())
          self.assertEqual(recorder.errors, 0)

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
    # A factory supporting only zero-argument construction remains valid.
    self.assertEqual(message.diagnostics.status, "tracking")
    self.assertEqual(message.diagnostics.context, recorder.context(sample))
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

  def test_real_diagnostic_messages_match_reference_fields_and_wire(self):
    for parent_factory, field_name, factory, parent_values, values in diagnostic_cases():
      for frame in (0, UINT64_MAX):
        with self.subTest(schema=factory, frame=frame):
          recorder = self.recorder(clock=lambda: (1 << 63) + 5, manifest_interval=2)
          recorder.session = UINT64_MAX
          sample = recorder.finish(recorder.begin(frame))
          expected = parent_factory.new_message(**parent_values)
          actual = parent_factory.new_message(**parent_values)
          before = json.dumps(values, sort_keys=True)
          self.assertTrue(attach_diagnostics_reference(expected, field_name, factory, recorder, sample, values))
          self.assertTrue(attach_diagnostics(actual, field_name, factory, recorder, sample, values))
          # JSON preserves NaN/Inf tokens and signed zero for the field comparison;
          # exact bytes additionally check lossless integers, lists and pointers.
          self.assertEqual(json.dumps(actual.to_dict(), sort_keys=True), json.dumps(expected.to_dict(), sort_keys=True))
          self.assertEqual(actual.to_bytes(), expected.to_bytes())
          self.assertEqual(json.dumps(values, sort_keys=True), before)
          self.assertEqual(recorder.errors, 0)

  def test_real_diagnostic_context_replaces_values_context_as_before(self):
    for parent_factory, field_name, factory, parent_values, values in diagnostic_cases():
      with self.subTest(schema=factory):
        recorder = self.recorder(clock=lambda: 100)
        sample = recorder.finish(recorder.begin())
        values = {**values, "context": {"producer": "caller context", "sequence": 999}}
        actual = parent_factory.new_message(**parent_values)
        expected = parent_factory.new_message(**parent_values)
        self.assertTrue(attach_diagnostics_reference(expected, field_name, factory, recorder, sample, values))
        self.assertTrue(attach_diagnostics(actual, field_name, factory, recorder, sample, values))
        self.assertEqual(actual.to_bytes(), expected.to_bytes())
        self.assertEqual(getattr(actual, field_name).context.producer, "fixture")
        self.assertEqual(getattr(actual, field_name).context.sequence, 1)

  def test_real_diagnostic_failures_leave_existing_parent_bytes_unchanged(self):
    for parent_factory, field_name, factory, parent_values, values in diagnostic_cases():
      failures = (
        {**values, "noSuchDiagnosticField": 1},
        {**values, "num_first_segment_words": 64},
        {**values, "which": "invalidDiagnosticUnion"},
        {**values, None: 1},
        {**values, "inputs": object()},
        {**values, "inputs": [{"service": "partially written", "monoTime": -1}]},
        {**values, "inputs": [{"service": "partially written", "kind": "invalidInputKind"}]},
        {**values, "context": {"producer": "partially written", "sequence": -1}},
      )
      for failed_values in failures:
        for implementation in (attach_diagnostics_reference, attach_diagnostics):
          with self.subTest(schema=factory, failure=failed_values, implementation=implementation.__name__):
            recorder = self.recorder(clock=lambda: 100)
            sample = recorder.finish(recorder.begin())
            actual = parent_factory.new_message(**parent_values)
            self.assertTrue(implementation(actual, field_name, factory, recorder, sample, values))
            before = actual.as_reader().as_builder().to_bytes()
            self.assertFalse(implementation(actual, field_name, factory, recorder, sample, failed_values))
            self.assertEqual(actual.as_reader().as_builder().to_bytes(), before)
            self.assertEqual(recorder.errors, 1)

  def test_real_diagnostic_context_serialization_failure_keeps_parent(self):
    for parent_factory, field_name, factory, parent_values, values in diagnostic_cases():
      with self.subTest(schema=factory):
        recorder = self.recorder(clock=lambda: 100)
        sample = recorder.finish(recorder.begin())
        actual = parent_factory.new_message(**parent_values)
        self.assertTrue(attach_diagnostics(actual, field_name, factory, recorder, sample, values))
        before = actual.as_reader().as_builder().to_bytes()
        with patch.object(recorder, "context", return_value={"producer": "invalid context", "session": -1}):
          self.assertFalse(attach_diagnostics(actual, field_name, factory, recorder, sample, values))
        self.assertEqual(actual.as_reader().as_builder().to_bytes(), before)
        self.assertEqual(recorder.errors, 1)

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
