from types import SimpleNamespace
from unittest.mock import Mock, patch

from openpilot.common.params import Params
from openpilot.selfdrive.ui.tests.test_sunnylink_removed import HeadlessUITestCase


class TestRoadRecordingSettings(HeadlessUITestCase):
  def test_default_opt_out_and_onroad_write_guard(self):
    from openpilot.selfdrive.ui.layouts.settings import toggles

    params = Params()
    params.remove('RecordRoadCameras')
    self.assertIs(params.get('RecordRoadCameras', return_default=True), True)
    layout = toggles.TogglesLayout.__new__(toggles.TogglesLayout)
    layout._params = params
    road_toggle = SimpleNamespace(action_item=Mock())
    layout._toggles = {'RecordRoadCameras': road_toggle}
    state = SimpleNamespace(is_offroad=lambda: True)
    with patch.object(toggles, 'ui_state', state):
      self.assertIs(layout._get_toggle_state('RecordRoadCameras'), True)
      layout._toggle_callback(False, 'RecordRoadCameras')
      self.assertIs(params.get('RecordRoadCameras', return_default=True), False)
      road_toggle.action_item.set_state.assert_called_with(False)
      state.is_offroad = lambda: False
      layout._toggle_callback(True, 'RecordRoadCameras')
      self.assertFalse(params.get_bool('RecordRoadCameras'))
      road_toggle.action_item.set_state.assert_called_with(False)
      state.is_offroad = lambda: True
      layout._toggle_callback(True, 'RecordRoadCameras')
      self.assertTrue(params.get_bool('RecordRoadCameras'))
      self.assertFalse(params.get_bool('OnroadCycleRequested'))
