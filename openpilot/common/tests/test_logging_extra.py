import json
import subprocess
import sys
import types
import unittest
from unittest import mock

from openpilot.common.logging_extra import json_handler, json_robust_dumps


class TestLoggingExtra(unittest.TestCase):
  def test_numpy_booleans_are_json_booleans(self):
    import numpy as np

    for value in (False, True):
      with self.subTest(value=value):
        self.assertIs(json_handler(np.bool_(value)), value)
        self.assertEqual(json_robust_dumps({"value": np.bool_(value)}), json.dumps({"value": value}))

  def test_numpy_boolean_proxy_keeps_isinstance_semantics(self):
    import numpy as np

    class BooleanProxy:
      def __init__(self, value):
        self.value = value

      @property
      def __class__(self):
        return np.bool_

      def __bool__(self):
        return self.value

    for value in (False, True):
      with self.subTest(value=value):
        self.assertIs(json_handler(BooleanProxy(value)), value)

  def test_other_numpy_objects_use_repr(self):
    import numpy as np

    for value in (np.int64(3), np.float32(1.25), np.array([True, False]), np.array([1, 2])):
      with self.subTest(value=repr(value)):
        self.assertEqual(json_handler(value), repr(value))
        self.assertEqual(json.loads(json_robust_dumps({"value": value})), {"value": repr(value)})

  def test_non_numpy_objects_use_repr(self):
    lookalike = type("bool_", (), {"__module__": "numpy"})()
    for value in (object(), ValueError("test"), b"test", {1, 2}, lookalike):
      with self.subTest(value=repr(value)):
        self.assertEqual(json_handler(value), repr(value))
        self.assertEqual(json.loads(json_robust_dumps({"value": value})), {"value": repr(value)})

  def test_partial_numpy_import_does_not_break_logging(self):
    for module in (None, types.ModuleType("numpy")):
      with self.subTest(module=module), mock.patch.dict(sys.modules, {"numpy": module}):
        value = ValueError("partial import")
        self.assertEqual(json_handler(value), repr(value))

  def test_regular_logging_does_not_import_numpy(self):
    result = subprocess.run([sys.executable, "-c", """
import importlib.abc
import io
import json
import logging
from pathlib import Path
import sys

attempted_imports = []

class RejectNumpy(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    if fullname == "numpy" or fullname.startswith("numpy."):
      attempted_imports.append(fullname)
      raise AssertionError(f"logging imported NumPy: {fullname}")
    return None

assert "numpy" not in sys.modules
sys.meta_path.insert(0, RejectNumpy())
from openpilot.common.logging_extra import SwagLogger, SwagFormatter, json_robust_dumps

output = io.StringIO()
logger = SwagLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(output)
handler.setFormatter(SwagFormatter(logger))
logger.addHandler(handler)
logger.info("ready")
logger.event("download", cached=True, path=Path("model.pkl"), error_object=ValueError("test"))
records = [json.loads(line) for line in output.getvalue().splitlines()]
assert records[0]["msg"] == "ready"
assert records[1]["msg"]["cached"] is True
assert records[1]["msg"]["path"] == repr(Path("model.pkl"))
assert records[1]["msg"]["error_object"] == repr(ValueError("test"))
assert json.loads(json_robust_dumps({"ready": True, "size": 12, "ratio": 0.5, "empty": None})) == {
  "ready": True, "size": 12, "ratio": 0.5, "empty": None,
}
assert not attempted_imports, attempted_imports
assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
"""], capture_output=True, text=True, timeout=30)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
  unittest.main()
