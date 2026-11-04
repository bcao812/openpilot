import os
import subprocess
import sys
import unittest

from openpilot.common.basedir import BASEDIR


class TestStartupImports(unittest.TestCase):
  def run_isolated(self, body: str) -> None:
    # Import-only probes run in fresh interpreters, without a model artifact or
    # GPU. A prior test importing stock modeld must not mask this dependency.
    result = subprocess.run(
      [sys.executable, "-B", "-c", body], cwd=BASEDIR,
      env={**os.environ, "DEV": "CPU", "PYTHONDONTWRITEBYTECODE": "1"},
      capture_output=True, text=True, timeout=30,
    )
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_manager_imports_do_not_load_a_gpu_runtime(self):
    self.run_isolated("""
import importlib.abc
import sys

attempts = []
class RejectTinygrad(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    if fullname == 'tinygrad' or fullname.startswith('tinygrad.'):
      attempts.append(fullname)
      raise AssertionError('manager must not import tinygrad before forking modeld: ' + fullname)

sys.meta_path.insert(0, RejectTinygrad())
from openpilot.system.manager import manager
assert not attempts, attempts
assert 'openpilot.sunnypilot.modeld_v2.modeld' not in sys.modules
assert 'openpilot.selfdrive.modeld.modeld' not in sys.modules
""")

  def test_tinygrad_model_import_is_independent_of_stock_model_and_libusb(self):
    self.run_isolated("""
import importlib.abc
import sys

attempts = []
class RejectUnusedModelImports(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    if fullname == 'openpilot.selfdrive.modeld.modeld' or fullname == 'usb1' or fullname.startswith('usb1.'):
      attempts.append(fullname)
      raise AssertionError('unused model dependency imported: ' + fullname)

sys.meta_path.insert(0, RejectUnusedModelImports())
from openpilot.sunnypilot.modeld_v2 import modeld
from openpilot.selfdrive.modeld.chestnut_state import ChestnutState
from tinygrad.device import Device
assert not attempts, attempts
assert modeld.ChestnutState is ChestnutState
assert not {'QCOM', 'AMD', 'NV', 'CUDA'} & Device._opened_devices, Device._opened_devices
""")

  def test_both_model_modules_keep_the_shared_chestnut_state_export(self):
    self.run_isolated("""
from openpilot.selfdrive.modeld.modeld import ChestnutState as StockChestnutState
from openpilot.sunnypilot.modeld_v2.modeld import ChestnutState as TinygradChestnutState
from openpilot.selfdrive.modeld.chestnut_state import ChestnutState
assert StockChestnutState is TinygradChestnutState is ChestnutState
""")
