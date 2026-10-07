from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pyray as rl

from openpilot.cereal import log, messaging
from openpilot.selfdrive.car.tests.test_honda_radar_observer import START, configuration, state
from openpilot.selfdrive.ui.sunnypilot.onroad.honda_radar_state import project_radar_point
from openpilot.selfdrive.ui.sunnypilot.onroad.lead_control_influence import accepted_influences, influence_rgb
from openpilot.selfdrive.ui.tests.test_honda_radar_overlay import sample
from openpilot.selfdrive.ui.tests.test_sunnylink_removed import HeadlessUITestCase


def executing_state():
  sm = state(START)
  for service in ('controlsState', 'carState', 'radarState'):
    sm[service] = getattr(messaging.new_message(service), service)
    sm.valid[service], sm.logMonoTime[service] = True, START
  sm['selfdriveState'].enabled = True
  sm['carControl'].enabled = True
  controls = sm['controlsState']
  controls.longitudinalPlanMonoTime = START
  controls.motionState = {
    'enabled': True, 'planningMode': 2, 'phase': 'track', 'planMonoTime': START,
    'objectInfluences': [{'version': 1, 'sessionId': 'fixture', 'trackId': 42, 'modelMonoTime': START,
                          'visionIndex': 0, 'kind': 'braking', 'strength': 0.25}],
  }
  return sm


class TestControlInfluenceRendering(HeadlessUITestCase):
  def test_native_control_receipt_roundtrip_is_the_ui_input(self):
    sm = executing_state()
    event = messaging.new_message('controlsState')
    event.valid = True
    event.logMonoTime = START
    event.controlsState = sm['controlsState']
    decoded = messaging.log_from_bytes(event.to_bytes())
    sm['controlsState'] = decoded.controlsState
    items = accepted_influences(sm, START + 1)
    self.assertEqual([(item.session_id, item.track_id, item.kind, item.strength) for item in items],
                     [('fixture', 42, 'braking', 0.25)])

  def test_native_caution_strength_rounds_to_the_same_light_red(self):
    sm = executing_state()
    record = sm['controlsState'].motionState.objectInfluences[0]
    record.kind, record.strength = 'caution', 0.15
    items = accepted_influences(sm, START + 1)
    self.assertEqual(items[0].kind, 'caution')
    self.assertAlmostEqual(items[0].strength, 0.15)
    self.assertEqual(influence_rgb(items[0].strength), (246, 135, 137))

  def test_cached_radar_geometry_recolors_and_clears_immediately_for_override(self):
    from openpilot.selfdrive.ui.sunnypilot.onroad import honda_radar_overlay as renderer

    data = sample()
    measured = data['observations'][0]
    track = {field: measured[field] for field in ('slot', 'generation', 'positionSeq', 'positionMonoTime')}
    data['fusion'] = {'version': 1, 'active': True, 'tracks': [dict(track, trackId=42)]}
    wire = messaging.new_message(renderer.SERVICE)
    wire.valid = True
    setattr(wire, renderer.SERVICE, data)
    sm = executing_state()
    calib = messaging.new_message('extrinsicsCalibration').extrinsicsCalibration
    calib.calStatus, calib.height = log.ExtrinsicsCalibration.Status.calibrated, [1.2]
    sm['extrinsicsCalibration'], sm.valid['extrinsicsCalibration'] = calib, True
    observer = Mock(updated={renderer.SERVICE: True}, valid={renderer.SERVICE: True}, logMonoTime={renderer.SERVICE: START + 1})
    observer.__getitem__ = Mock(return_value=getattr(wire, renderer.SERVICE).as_reader())
    overlay = renderer.HondaRadarOverlay()
    overlay.sm = observer
    view = SimpleNamespace(frame=object(), client=SimpleNamespace(timestamp_eof=START), device_camera=object(),
                           model_renderer=SimpleNamespace(_car_space_transform=np.array(((640., 1000., 0), (360., 0, 1000.), (1., 0, 0)))))
    params = Mock()
    params.get.return_value, params.get_bool.return_value = 4, True
    ui = SimpleNamespace(CP=configuration()[0], started=True, sm=sm, params=params, active_bundle=False)
    with patch.object(renderer, 'ui_state', ui), patch.object(renderer.time, 'monotonic_ns', return_value=START + 2), \
         patch.object(renderer.gui_app, 'font', return_value=object()), \
         patch.object(renderer, 'project_radar_point', wraps=project_radar_point) as project, \
         patch.object(rl, 'draw_circle_v') as circles, patch.object(rl, 'draw_text_ex') as text:
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      self.assertEqual(project.call_count, 1)
      light = circles.call_args.args[-1]
      self.assertEqual((light.r, light.g, light.b), influence_rgb(0.25))
      self.assertEqual(text.call_args.args[1], '54.1')
      observer.updated[renderer.SERVICE] = False
      sm['controlsState'].motionState.objectInfluences[0].strength = 1.0
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      deep = circles.call_args.args[-1]
      self.assertEqual((deep.r, deep.g, deep.b), influence_rgb(1.0))
      self.assertEqual(project.call_count, 1)
      motion = sm['controlsState'].motionState
      motion.degraded, motion.waitingForFreshInput = True, True
      motion.objectInfluences[0].version = 2
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      mitigation = circles.call_args.args[-1]
      self.assertEqual((mitigation.r, mitigation.g, mitigation.b), (195, 20, 35))
      motion.objectInfluences[0].version = 1
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      unaccepted = circles.call_args.args[-1]
      self.assertEqual((unaccepted.r, unaccepted.g, unaccepted.b), (255, 215, 0))
      motion.objectInfluences[0].version = 2
      sm['carState'].gasPressed = True
      overlay.render(view, rl.Rectangle(0, 0, 1280, 720))
      reset = circles.call_args.args[-1]
      self.assertEqual((reset.r, reset.g, reset.b), (255, 215, 0))
      self.assertEqual(project.call_count, 1)

  def test_native_physical_mitigation_receipt_only_colors_the_adopted_target(self):
    from openpilot.selfdrive.ui.sunnypilot.onroad import chevron_metrics as renderer

    ui = SimpleNamespace(chevron_metrics=renderer.ChevronOptions.DISTANCE_ONLY, is_metric=True)
    vehicles = [SimpleNamespace(chevron=[(250.0, 150.0), (260.0, 170.0)]), SimpleNamespace(chevron=[])]
    cases = (
      ('adopted', {}, (195, 20, 35)), ('ordinary_fault', {'version': 1}, (255, 255, 255)),
      ('no_physical_identity', {'sessionId': '', 'trackId': 0}, (255, 255, 255)),
      ('wrong_strength', {'strength': 0.6}, (255, 255, 255)), ('not_braking', {'kind': 'hold'}, (255, 255, 255)),
      ('expired', {}, (255, 255, 255)), ('gas_override', {}, (255, 255, 255)), ('brake_override', {}, (255, 255, 255)),
      ('wrong_plan', {}, (255, 255, 255)),
    )
    with patch.object(renderer, 'ui_state', ui), patch.object(renderer.gui_app, 'font', return_value=object()), \
         patch.object(renderer.time, 'monotonic_ns', return_value=START + 2), \
         patch.object(renderer, 'measure_text_cached', return_value=rl.Vector2(90, 40)), \
         patch.object(rl, 'draw_text_ex') as text:
      metrics = renderer.ChevronMetrics()
      metrics._lead_status_alpha = 1.0
      for case, changes, expected in cases:
        with self.subTest(case=case):
          sm = executing_state()
          radar = sm['radarState']
          radar.leadOne = {'present': True, 'dRel': 24.0, 'vRel': 0.0,
                           'evidence': {'version': 1, 'source': 'fused', 'sessionId': 'fixture', 'trackId': 42,
                                        'modelMonoTime': START, 'visionIndex': 0}}
          motion = sm['controlsState'].motionState
          motion.degraded, motion.waitingForFreshInput = True, True
          record = motion.objectInfluences[0]
          record.version, record.strength = 2, 1.0
          for field, value in changes.items():
            setattr(record, field, value)
          if case == 'expired':
            sm.logMonoTime['controlsState'] = START - 200_000_001
          elif case == 'gas_override':
            sm['carState'].gasPressed = True
          elif case == 'brake_override':
            sm['carState'].brakePressed = True
          elif case == 'wrong_plan':
            motion.planMonoTime = START - 1
          wire = messaging.new_message('controlsState')
          wire.valid, wire.logMonoTime, wire.controlsState = True, sm.logMonoTime['controlsState'], sm['controlsState']
          sm['controlsState'] = messaging.log_from_bytes(wire.to_bytes()).controlsState
          text.reset_mock()
          metrics.draw_lead_status(sm, radar, rl.Rectangle(0, 0, 1280, 720), vehicles)
          foreground = [call.args for call in text.call_args_list if call.args[-1].r != 0]
          self.assertEqual(len(foreground), 1)
          color = foreground[0][-1]
          self.assertEqual((color.r, color.g, color.b), expected)

  def test_formal_lead_distance_uses_same_identity_color_and_expires(self):
    from openpilot.selfdrive.ui.sunnypilot.onroad import chevron_metrics as renderer

    sm = executing_state()
    radar = sm['radarState']
    radar.leadOne = {'present': True, 'dRel': 24.0, 'vRel': 0.0,
                     'evidence': {'version': 1, 'source': 'fused', 'sessionId': 'fixture', 'trackId': 42,
                                  'modelMonoTime': START, 'visionIndex': 0}}
    radar.leadTwo = {'present': True, 'dRel': 6.0, 'vRel': 0.0,
                     'evidence': {'version': 1, 'source': 'vision', 'modelMonoTime': START, 'visionIndex': 1}}
    vehicles = [SimpleNamespace(chevron=[(250.0, 150.0), (260.0, 170.0)]),
                SimpleNamespace(chevron=[(550.0, 150.0), (560.0, 170.0)])]
    ui = SimpleNamespace(chevron_metrics=renderer.ChevronOptions.DISTANCE_ONLY, is_metric=True)
    with patch.object(renderer, 'ui_state', ui), patch.object(renderer.gui_app, 'font', return_value=object()), \
         patch.object(renderer.time, 'monotonic_ns', return_value=START + 2), \
         patch.object(renderer, 'measure_text_cached', return_value=rl.Vector2(90, 40)), \
         patch.object(rl, 'draw_text_ex') as text:
      metrics = renderer.ChevronMetrics()
      metrics._lead_status_alpha = 1.0
      metrics.draw_lead_status(sm, radar, rl.Rectangle(0, 0, 1280, 720), vehicles)
      foreground = [call.args for call in text.call_args_list if call.args[-1].r != 0]
      self.assertEqual([(call[1], (call[-1].r, call[-1].g, call[-1].b)) for call in foreground],
                       [('24 m', influence_rgb(0.25)), ('6 m', (255, 255, 255))])
      text.reset_mock()
      sm.valid['controlsState'] = False
      metrics.draw_lead_status(sm, radar, rl.Rectangle(0, 0, 1280, 720), vehicles)
      foreground = [call.args for call in text.call_args_list if call.args[-1].r != 0]
      self.assertTrue(all((call[-1].r, call[-1].g, call[-1].b) == (255, 255, 255) for call in foreground))

  def test_old_layout_visual_label_uses_the_explicit_model_epoch_and_slot(self):
    from openpilot.selfdrive.ui.sunnypilot.onroad import chevron_metrics as renderer

    sm = executing_state()
    radar = sm['radarState']
    radar.mdMonoTime = START - 50_000_000
    radar.leadOne = {'present': True, 'radar': False, 'dRel': 24.0, 'vRel': 0.0}
    radar.leadTwo = {'present': True, 'radar': False, 'dRel': 6.0, 'vRel': 0.0}
    record = sm['controlsState'].motionState.objectInfluences[0]
    record.sessionId, record.trackId, record.modelMonoTime = '', 0, radar.mdMonoTime
    vehicles = [SimpleNamespace(chevron=[(250.0, 150.0), (260.0, 170.0)]),
                SimpleNamespace(chevron=[(550.0, 150.0), (560.0, 170.0)])]
    ui = SimpleNamespace(chevron_metrics=renderer.ChevronOptions.DISTANCE_ONLY, is_metric=True)
    with patch.object(renderer, 'ui_state', ui), patch.object(renderer.gui_app, 'font', return_value=object()), \
         patch.object(renderer.time, 'monotonic_ns', return_value=START + 2), \
         patch.object(renderer, 'measure_text_cached', return_value=rl.Vector2(90, 40)), \
         patch.object(rl, 'draw_text_ex') as text:
      metrics = renderer.ChevronMetrics()
      metrics._lead_status_alpha = 1.0
      metrics.draw_lead_status(sm, radar, rl.Rectangle(0, 0, 1280, 720), vehicles)
      foreground = [call.args for call in text.call_args_list if call.args[-1].r != 0]
      self.assertEqual([(call[1], (call[-1].r, call[-1].g, call[-1].b)) for call in foreground],
                       [('24 m', influence_rgb(0.25)), ('6 m', (255, 255, 255))])
      text.reset_mock()
      radar.mdMonoTime = START  # The owner still acknowledges the earlier frame.
      metrics.draw_lead_status(sm, radar, rl.Rectangle(0, 0, 1280, 720), vehicles)
      foreground = [call.args for call in text.call_args_list if call.args[-1].r != 0]
      self.assertTrue(all((call[-1].r, call[-1].g, call[-1].b) == (255, 255, 255) for call in foreground))
