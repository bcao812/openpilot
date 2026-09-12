import os
import subprocess
import sys
import unittest


class TestModelImports(unittest.TestCase):
  def run_import_check(self, program):
    environment = dict(os.environ, DEV="CPU:LLVM", SCALE="1")
    environment.pop("DEBUG", None)
    result = subprocess.run([sys.executable, "-c", program], env=environment, capture_output=True, text=True, timeout=30)
    self.assertEqual(result.returncode, 0, result.stderr)

  def test_controlsd_does_not_load_inference_runtime(self):
    self.run_import_check("""
import importlib.abc
import sys

class RejectModelRuntime(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    if fullname in ("openpilot.selfdrive.modeld.modeld", "openpilot.selfdrive.modeld.compile_modeld", "tinygrad"):
      raise AssertionError(f"controlsd imported inference runtime: {fullname}")
    return None

sys.meta_path.insert(0, RejectModelRuntime())
from openpilot.selfdrive.controls import controlsd
from openpilot.selfdrive.modeld.constants import LAT_SMOOTH_SECONDS
assert controlsd.LAT_SMOOTH_SECONDS == LAT_SMOOTH_SECONDS == 0.0
""")

  def test_modeld_preserves_smoothing_constant_export(self):
    self.run_import_check("""
from openpilot.selfdrive.modeld import constants, modeld
assert modeld.LAT_SMOOTH_SECONDS == constants.LAT_SMOOTH_SECONDS == 0.0
""")


if __name__ == "__main__":
  unittest.main()
