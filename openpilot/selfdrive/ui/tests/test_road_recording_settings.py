from types import SimpleNamespace
from unittest.mock import Mock, patch

from openpilot.common.params import Params
from openpilot.selfdrive.ui.tests.test_sunnylink_removed import HeadlessUITestCase


class TestRoadRecordingSettings(HeadlessUITestCase):
  def test_default_opt_out_and_onroad_write_guard(self):
    from openpilot.selfdrive.ui.sunnypilot.layouts.settings import device

    params = Params()
    params.remove('RecordRoadCameras')
    self.assertIs(params.get('RecordRoadCameras', return_default=True), True)
    layout = device.DeviceLayoutSP.__new__(device.DeviceLayoutSP)
    layout._params = params
    layout._record_road_cameras = SimpleNamespace(action_item=Mock())
    state = SimpleNamespace(is_offroad=lambda: True)
    with patch.object(device, 'ui_state', state):
      layout._set_record_road_cameras(False)
      self.assertIs(params.get('RecordRoadCameras', return_default=True), False)
      layout._record_road_cameras.action_item.set_state.assert_called_with(False)
      state.is_offroad = lambda: False
      layout._set_record_road_cameras(True)
      self.assertFalse(params.get_bool('RecordRoadCameras'))
      layout._record_road_cameras.action_item.set_state.assert_called_with(False)
      state.is_offroad = lambda: True
      layout._set_record_road_cameras(True)
      self.assertTrue(params.get_bool('RecordRoadCameras'))
