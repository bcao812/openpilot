import ast
from pathlib import Path

from opendbc.car.structs import car
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase


class TestLocalProcessPolicy(OpenpilotTestCase):
  def test_streaming_stays_off_during_car_onroad(self):
    from openpilot.system.manager.process_config import managed_processes

    params = Params()
    car_params = car.CarParams.new_message()
    car_params.notCar = False
    params.put_bool("IsLiveStreaming", True, block=True)
    for name in ("stream_encoderd", "webrtcd"):
      with self.subTest(name=name):
        self.assertFalse(managed_processes[name].should_run(True, params, car_params))
        self.assertTrue(managed_processes[name].should_run(False, params, car_params))
    for started in (False, True):
      self.assertTrue(managed_processes["models_manager"].should_run(started, params, car_params))

  def test_removed_remote_services_stay_removed(self):
    from openpilot.system.manager.process_config import managed_processes

    for name in ("dashcam", "c3_client"):
      self.assertNotIn(name, managed_processes)
    for relative_path in ("dashcam.py", "openpilot/selfdrive/c3_client.py", "openpilot/system/ui/lib/ui_stream.py"):
      self.assertFalse((Path(BASEDIR) / relative_path).exists())

  def test_model_demo_imports_remain_lazy(self):
    for relative_path in ("openpilot/selfdrive/modeld/modeld.py", "openpilot/sunnypilot/modeld_v2/modeld.py"):
      module = ast.parse((Path(BASEDIR) / relative_path).read_text())
      imports = [node.module for node in module.body if isinstance(node, ast.ImportFrom)]
      self.assertNotIn("opendbc.car.car_helpers", imports, relative_path)
    launch_environment = (Path(BASEDIR) / "launch_env.sh").read_text().splitlines()
    self.assertIn("export DEV=QCOM", launch_environment)
    self.assertFalse(any(line.startswith("export STREAM") for line in launch_environment))

  def test_nnlc_is_present_in_steering_items(self):
    module = ast.parse((Path(BASEDIR) / "openpilot/selfdrive/ui/sunnypilot/layouts/settings/steering.py").read_text())
    items = [
      node.value for node in ast.walk(module)
      if isinstance(node, ast.Assign) and any(
        (isinstance(target, ast.Attribute) and target.attr == "items") or
        (isinstance(target, ast.Name) and target.id == "items") for target in node.targets
      )
    ]
    self.assertTrue(any(
      isinstance(node, ast.Attribute) and node.attr == "_nnlc_toggle"
      for value in items for node in ast.walk(value)
    ))
