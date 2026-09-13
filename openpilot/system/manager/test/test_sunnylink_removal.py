import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from opendbc.car.structs import car
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.system import sentry
from openpilot.system.manager import manager
from openpilot.system.manager.process_config import managed_processes
from openpilot.sunnypilot.system import statsd


class TestSunnylinkRemoval(OpenpilotTestCase):
  def test_legacy_enabled_params_cannot_restore_services(self):
    params = Params()
    params.put_bool("SunnylinkEnabled", True, block=True)
    params.put_bool("EnableSunnylinkUploader", True, block=True)
    params.put("SunnylinkDongleId", "retired-test-device", block=True)

    for name in ("manage_sunnylinkd", "sunnylink_registration_manager", "sunnylink_uploader", "statsd_sp", "backup_manager"):
      self.assertNotIn(name, managed_processes)

    car_params = car.CarParams.new_message()
    car_params.notCar = False
    for name in ("card", "controlsd", "selfdrived", "plannerd", "radard", "loggerd", "encoderd"):
      with self.subTest(process=name):
        self.assertTrue(managed_processes[name].should_run(True, params, car_params))
        self.assertFalse(managed_processes[name].should_run(False, params, car_params))
    for name in ("models_manager", "mapd_manager", "statsd", "logmessaged"):
      for started in (False, True):
        self.assertTrue(managed_processes[name].should_run(started, params, car_params))
    for name in ("manage_athenad", "uploader", "modeld", "modeld_tinygrad", "pandad", "updated"):
      self.assertIn(name, managed_processes)

  def test_sunnylink_sources_are_removed(self):
    package = Path(BASEDIR) / "openpilot/sunnypilot/sunnylink"
    self.assertEqual(list(package.rglob("*.py")), [])
    self.assertFalse((Path(BASEDIR) / "openpilot/sunnypilot/selfdrive/car/sync_sunnylink_params.py").exists())

  def test_shared_modules_import_without_sunnylink(self):
    result = subprocess.run([sys.executable, "-c", """
import importlib
import importlib.abc
import sys

class RejectSunnylink(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    if fullname == "openpilot.sunnypilot.sunnylink" or fullname.startswith("openpilot.sunnypilot.sunnylink."):
      raise AssertionError(f"retired sunnylink import: {fullname}")
    return None

sys.meta_path.insert(0, RejectSunnylink())
for name in (
  "openpilot.system.manager.manager",
  "openpilot.system.athena.athenad",
  "openpilot.system.sentry",
  "openpilot.sunnypilot.selfdrive.car.interfaces",
  "openpilot.sunnypilot.system.params_migration",
  "openpilot.sunnypilot.models.manager",
):
  importlib.import_module(name)
"""], capture_output=True, text=True, timeout=30)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

  def test_manager_initialization_preserves_local_defaults(self):
    metadata = SimpleNamespace(
      channel="c3xl-dev",
      release_channel=False,
      release_sp_channel=False,
      tested_channel=False,
      development_channel=True,
      openpilot=SimpleNamespace(version="test", git_commit="test-commit", git_commit_date="test-date",
                               git_origin="test-origin", git_normalized_origin="test-origin", is_dirty=True),
    )
    with mock.patch.object(manager, "save_bootlog"), \
         mock.patch.object(manager, "get_build_metadata", return_value=metadata), \
         mock.patch.object(manager, "register", return_value="test-comma-device") as register, \
         mock.patch.object(manager.sentry, "init"):
      manager.manager_init()
    params = Params()
    register.assert_called_once_with(show_spinner=True)
    self.assertTrue(params.get_bool("OpenpilotEnabledToggle"))
    self.assertEqual(os.environ["DONGLE_ID"], "test-comma-device")
    self.assertEqual(params.get("GitCommit"), "test-commit")
    self.assertEqual(params.get("RouteCount"), 0)

  def test_statistics_have_only_one_destination(self):
    self.assertIs(type(statsd.statlog), statsd.StatLog)
    sender = statsd.StatLog()
    context = mock.MagicMock()
    with mock.patch.object(statsd.zmq.Context, "instance", return_value=context):
      sender.gauge("device.power", 12.5)
      sender.sample("device.temperature", 45.0)
    context.socket.assert_called_once_with(statsd.zmq.PUSH)
    context.socket.return_value.connect.assert_called_once_with(statsd.STATS_SOCKET)
    self.assertEqual(context.socket.return_value.send_string.call_args_list, [
      mock.call("device.power:12.5|g", statsd.zmq.NOBLOCK),
      mock.call("device.temperature:45.0|sa", statsd.zmq.NOBLOCK),
    ])

  def test_crash_reporting_preserves_comma_identity(self):
    params = Params()
    params.put("HardwareSerial", "test-serial", block=True)
    params.put("GithubUsername", "test-user", block=True)
    for dongle_id in (None, "test-comma-device"):
      with self.subTest(dongle_id=dongle_id):
        if dongle_id is None:
          params.remove("DongleId")
        else:
          params.put("DongleId", dongle_id, block=True)
        with mock.patch.object(sentry.sentry_sdk, "set_user") as set_user:
          sentry.set_user()
        set_user.assert_called_once_with({
          "id": dongle_id or f"{sentry.UNREGISTERED_DONGLE_ID}-test-serial",
          "name": "test-user",
        })
