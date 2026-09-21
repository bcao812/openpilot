from pathlib import Path
from unittest import mock

from opendbc.car.structs import car
from openpilot.cereal import custom
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.system.manager import process_config


class TestRemovedMapServices(OpenpilotTestCase):
  def test_map_services_are_absent(self):
    for name in ("mapd", "mapd_manager"):
      self.assertNotIn(name, process_config.managed_processes)

    source_root = Path(BASEDIR)
    self.assertEqual(list((source_root / "openpilot/sunnypilot/mapd").rglob("*.py")), [])
    self.assertFalse((source_root / "openpilot/third_party/mapd_pfeiferj/mapd").exists())

  def test_localization_and_logging_keep_their_onroad_gates(self):
    params = Params()
    car_params = car.CarParams.new_message()
    car_params.notCar = False
    for name in ("locationd", "locationd_llk", "calibrationd", "paramsd", "loggerd", "encoderd"):
      with self.subTest(process=name):
        process = process_config.managed_processes[name]
        self.assertTrue(process.should_run(True, params, car_params))
        self.assertFalse(process.should_run(False, params, car_params))

    for name in ("models_manager", "statsd", "logmessaged"):
      for started in (False, True):
        with self.subTest(process=name, started=started):
          self.assertTrue(process_config.managed_processes[name].should_run(started, params, car_params))

  def test_uploader_respects_onroad_uploads(self):
    params = Params()
    car_params = car.CarParams.new_message()
    uploader = process_config.managed_processes["uploader"]
    for onroad_uploads in (False, True):
      params.put_bool("OnroadUploads", onroad_uploads, block=True)
      for started in (False, True):
        with self.subTest(onroad_uploads=onroad_uploads, started=started):
          self.assertEqual(uploader.should_run(started, params, car_params), not started or onroad_uploads)

  def test_gps_processes_follow_hardware_selection(self):
    params = Params()
    car_params = car.CarParams.new_message()
    for has_ublox in (False, True):
      with mock.patch.object(process_config, "ublox_available", return_value=has_ublox):
        for started in (False, True):
          for name in ("ubloxd", "pigeond", "qcomgpsd"):
            with self.subTest(has_ublox=has_ublox, started=started, process=name):
              uses_ublox = name != "qcomgpsd"
              expected = started and has_ublox == uses_ublox
              self.assertEqual(process_config.managed_processes[name].should_run(started, params, car_params), expected)

  def test_model_processes_follow_selected_runner(self):
    params = Params()
    car_params = car.CarParams.new_message()
    runners = {
      "modeld": custom.ModelManagerSP.Runner.stock,
      "modeld_tinygrad": custom.ModelManagerSP.Runner.tinygrad,
    }
    for selected_name, runner in runners.items():
      with mock.patch.object(process_config, "get_active_model_runner", return_value=runner):
        for started in (False, True):
          for name in runners:
            with self.subTest(selected=selected_name, started=started, process=name):
              expected = started and name == selected_name
              self.assertEqual(process_config.managed_processes[name].should_run(started, params, car_params), expected)
