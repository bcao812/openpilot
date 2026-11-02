import importlib
from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pyray as rl

from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.ui.tests.test_sunnylink_removed import HeadlessUITestCase


class Messages(dict):
  def __init__(self, **values):
    super().__init__(values)
    self.frame = 5
    self.recv_frame = dict.fromkeys(('extrinsicsCalibration', 'modelV2', 'radarState'), self.frame)
    self.updated = dict.fromkeys(('carParams', 'modelV2', 'radarState', 'selfdriveState'), False)
    self.valid = {'radarState': True}


def radar_message():
  radar = log.RadarState.new_message()
  radar.leadOne = {'present': True, 'radar': True, 'dRel': 10.0,
                   'evidence': {'version': 1, 'source': 'radar'}}
  radar.leadTwo = {'present': True, 'radar': False, 'dRel': 35.0,
                   'evidence': {'version': 1, 'source': 'vision'}}
  return radar


class TestRadarSignalUI(HeadlessUITestCase):
  def test_default_and_offroad_write_guard_in_both_settings_layouts(self):
    from openpilot.selfdrive.ui.layouts.settings import toggles
    from openpilot.selfdrive.ui.mici.layouts.settings import toggles as mici

    params = Params()
    state = SimpleNamespace(params=params, started=False, CP=None, update_params=Mock())
    state.is_offroad = lambda: not state.started
    for module, cls in ((toggles, toggles.TogglesLayout), (mici, mici.TogglesLayoutMici)):
      with self.subTest(layout=module.__name__), patch.object(module, 'ui_state', state):
        params.remove('RadarSignal')
        params.remove('OnroadCycleRequested')
        state.started = False
        layout = cls.__new__(cls)
        if module is toggles:
          layout._params = params
          layout._toggles = {'RadarSignal': SimpleNamespace(action_item=Mock())}
          self.assertTrue(layout._get_toggle_state('RadarSignal'))
          callback = partial(layout._toggle_callback, param='RadarSignal')
        else:
          layout._radar_signal_btn = Mock()
          layout._sound_toggle = Mock()
          layout._sound_options = ["None", "Warning", "All"]
          layout._refresh_toggles = (('RadarSignal', layout._radar_signal_btn),)
          layout._update_toggles()
          layout._radar_signal_btn.set_checked.assert_called_with(True)
          callback = layout._on_radar_signal

        callback(False)
        self.assertIs(params.get('RadarSignal'), False)
        state.started = True
        callback(True)
        self.assertIs(params.get('RadarSignal'), False)
        state.started = False
        callback(True)
        self.assertIs(params.get('RadarSignal'), True)
        self.assertFalse(params.get_bool('OnroadCycleRequested'))

  def test_effective_mode_changes_only_at_the_next_drive(self):
    from openpilot.selfdrive.ui.ui_state import UIState, UIStatus

    params = Params()
    state = object.__new__(UIState)
    state.params = params
    state.sm = Messages(deviceState=SimpleNamespace(chestnutPresent=False), selfdriveState=SimpleNamespace(enabled=False),
                        selfdriveStateSP=SimpleNamespace(mads=SimpleNamespace(enabled=False)))
    state.started = state._started_prev = state._engaged_prev = False
    state.status = UIStatus.DISENGAGED
    state.radar_signal = True
    state._engaged_transition_callbacks = []
    observed = []
    state._offroad_transition_callbacks = [lambda: observed.append(state.radar_signal)]

    params.put_bool('RadarSignal', False, block=True)
    state.started = True
    state._update_status()
    self.assertFalse(state.radar_signal)
    params.put_bool('RadarSignal', True, block=True)
    state.sm.frame += 1
    state._update_status()
    self.assertFalse(state.radar_signal)
    state.started = False
    state._update_status()
    self.assertFalse(state.radar_signal)
    state.started = True
    state._update_status()
    self.assertTrue(state.radar_signal)
    self.assertEqual(observed, [False, False, True])

  def test_renderers_clear_radar_geometry_keep_vision_and_wait_for_the_new_drive(self):
    from openpilot.selfdrive.ui.ui_state import UIStatus

    for variant in ('onroad', 'mici.onroad'):
      module = importlib.import_module(f'openpilot.selfdrive.ui.{variant}.model_renderer')
      radar = radar_message()
      sm = Messages(radarState=radar, modelV2=SimpleNamespace(), extrinsicsCalibration=SimpleNamespace(height=[1.2]),
                    selfdriveState=SimpleNamespace(experimentalMode=False),
                    carOutput=SimpleNamespace(actuatorsOutput=SimpleNamespace(torque=0.0)))
      ui = SimpleNamespace(sm=sm, started_frame=5, radar_signal=False, status=UIStatus.ENGAGED)
      rect = rl.Rectangle(0, 0, 1280, 720)
      with self.subTest(variant=variant), patch.object(module, 'ui_state', ui), patch.object(module.gui_app, 'mark_phase'):
        renderer = module.ModelRenderer.__new__(module.ModelRenderer)
        renderer._radar_signal, renderer._counter, renderer._camera_offset = True, 1, 0.0
        renderer._longitudinal_control, renderer._transform_dirty, renderer._rect = True, False, rect
        renderer._torque_filter = Mock()
        renderer._path = SimpleNamespace(raw_points=np.array([[1.0, 0.0, 0.0], [100.0, 0.0, 0.0]]))
        renderer.blendful = Mock()
        renderer._lane_lines = [SimpleNamespace(raw_points=renderer._path.raw_points)] * 4
        renderer._lead_vehicles = [module.LeadVehicle(chevron=[(5, 5)]), module.LeadVehicle(chevron=[(5, 5)])]
        renderer._update_model, renderer._draw_lane_lines, renderer._draw_path = Mock(), Mock(), Mock()
        renderer._draw_lead_indicator, renderer.chevron_metrics = Mock(), Mock()
        renderer._map_to_screen = Mock(return_value=(300.0, 200.0))

        renderer._render(rect)
        self.assertFalse(renderer._update_model.call_args.args[0].present)
        self.assertFalse(renderer._lead_vehicles[0].chevron)
        self.assertTrue(renderer._lead_vehicles[1].chevron)
        self.assertTrue(radar.leadOne.present)

        # Re-enabling must not expose the previous drive's cached message.
        ui.radar_signal, ui.started_frame = True, 6
        sm.recv_frame['modelV2'] = sm.recv_frame['extrinsicsCalibration'] = 6
        renderer._render(rect)
        self.assertIsNone(renderer._update_model.call_args.args[0])
        self.assertTrue(all(not lead.chevron for lead in renderer._lead_vehicles))

        sm.recv_frame['radarState'], sm.updated['radarState'] = 6, True
        radar.leadOne.dRel = 30.0
        renderer._render(rect)
        self.assertEqual(renderer._update_model.call_args.args[0].dRel, 30.0)
        self.assertTrue(renderer._lead_vehicles[0].chevron)

  def test_developer_display_filters_radar_and_previous_drive_messages(self):
    from openpilot.selfdrive.ui.sunnypilot.onroad.developer_ui import elements

    radar = radar_message()
    sm = Messages(radarState=radar)
    ui = SimpleNamespace(radar_signal=False, started_frame=5)
    with patch.object(elements, 'ui_state', ui):
      self.assertEqual(elements.RelDistElement().update(sm, True).value, '-')
      radar.leadOne = radar.leadTwo
      self.assertEqual(elements.RelDistElement().update(sm, True).value, '35')
      ui.radar_signal, ui.started_frame = True, 6
      self.assertEqual(elements.RelDistElement().update(sm, True).value, '-')
      sm.recv_frame['radarState'] = 6
      self.assertEqual(elements.RelDistElement().update(sm, True).value, '35')

  def test_disabled_overlay_releases_subscription_and_all_cached_state(self):
    from openpilot.selfdrive.ui.sunnypilot.onroad import honda_radar_overlay as module

    overlay = module.HondaRadarOverlay()
    subscription = Mock()
    overlay.sm = subscription
    overlay.state.active, overlay.state.session, overlay.state.sequence = True, 'previous-drive', 42
    overlay.state.history[0].append(object())
    overlay.labels, overlay.layout_key = [object()], ('old',)
    with patch.object(module, 'ui_state', SimpleNamespace(radar_signal=False)), patch.object(module.messaging, 'SubMaster') as subscribe:
      overlay._render(None, None)
      overlay._render(None, None)
    self.assertIsNone(overlay.sm)
    self.assertFalse(overlay.state.active)
    self.assertFalse(any(overlay.state.history))
    self.assertNotEqual(overlay.state.session, 'previous-drive')
    self.assertFalse(overlay.labels)
    self.assertIsNone(overlay.layout_key)
    subscription.update.assert_not_called()
    subscribe.assert_not_called()
