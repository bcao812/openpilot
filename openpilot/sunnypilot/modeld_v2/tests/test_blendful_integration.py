"""Actual v2 action/publication behavior using bounded, offline model-output fixtures."""
from types import SimpleNamespace
from unittest import mock

import numpy as np

from openpilot.cereal import log
from openpilot.cereal import messaging
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.modeld_v2.blendful import Blendful
from openpilot.sunnypilot.modeld_v2.constants import Meta, ModelConstants
from openpilot.sunnypilot.modeld_v2 import modeld
from openpilot.sunnypilot.modeld_v2.tests.test_blendful import frame


def model_state(generation=None, chestnut=False):
  state = modeld.ModelState.__new__(modeld.ModelState)
  state.generation = generation
  state.LAT_SMOOTH_SECONDS = 0.2
  state.LONG_SMOOTH_SECONDS = 0.3
  state.MIN_LAT_CONTROL_SPEED = 0.3
  state.PLANPLUS_CONTROL = 1.0
  state.constants = ModelConstants()
  state.chestnut = chestnut
  state.lat_delay = 0.2
  state._vision_input_names = ['img', 'big_img']
  state._desire_key = 'desire'
  state.numpy_inputs = {}
  return state


def model_output(head='action', v=20.0, k=0.00125, accel=0.6, **kwargs):
  output = frame(k=k, **kwargs)
  times = np.array(ModelConstants.T_IDXS)
  output['plan'][0, :, 3] = v + accel * times
  output['plan'][0, :, 6] = accel
  output['plan'][0, :, 11] = k * max(v, 1.0) * times
  output['plan'][0, :, 14] = k * max(v, 1.0)
  if head == 'action':
    output['action'] = np.array([[k * max(v, 1.0) ** 2, accel]])
  elif head == 'desired_curvature':
    output['desired_curvature'] = np.array([[k]])
  output.update({
    'plan_stds': np.ones((1, 33, 15)),
    'road_edges': np.zeros((1, 2, 33, 2)), 'road_edges_stds': np.ones((1, 2, 33, 2)),
    'lead': np.zeros((1, 3, 6, 4)), 'lead_stds': np.ones((1, 3, 6, 4)), 'lead_prob': np.zeros((1, 3)),
    'desire_state': np.zeros((1, 8)), 'desire_pred': np.zeros((1, 4, 8)), 'meta': np.zeros((1, 55)),
    'pose': np.zeros((1, 6)), 'pose_stds': np.ones((1, 6)),
    'wide_from_device_euler': np.zeros((1, 3)), 'wide_from_device_euler_stds': np.ones((1, 3)),
    'road_transform': np.zeros((1, 6)), 'road_transform_stds': np.ones((1, 6)),
  })
  output['desire_state'][0, 0] = 1.0
  return output


def action_messages(action, valid=True):
  model = messaging.new_message('modelV2', valid=valid)
  driving = messaging.new_message('drivingModelData', valid=valid)
  sp = messaging.new_message('modelDataV2SP')
  model.modelV2.action = action
  driving.drivingModelData.action = action
  return model, driving, sp


def healthy_inputs(now_ns=10_000_000_000):
  data = {
    'carState': SimpleNamespace(vEgo=20.0),
    'carControl': SimpleNamespace(latActive=True),
    'extrinsicsCalibration': SimpleNamespace(calStatus=log.ExtrinsicsCalibration.Status.calibrated, rpyCalib=[0.0] * 3),
  }
  sm = mock.MagicMock()
  sm.__getitem__.side_effect = data.__getitem__
  sm.seen = dict.fromkeys(data, True)
  sm.valid = dict.fromkeys(data, True)
  sm.alive = dict.fromkeys(data, True)
  sm.freq_ok = dict.fromkeys(data, True)
  sm.logMonoTime = dict.fromkeys(data, now_ns)
  sm.all_checks.side_effect = lambda services: all(sm.valid[s] and sm.alive[s] and sm.freq_ok[s] for s in services)
  dh = SimpleNamespace(desire=log.Desire.none, lane_change_state=log.LaneChangeState.off)
  return sm, data, dh


class TestBlendfulAction(OpenpilotTestCase):
  def test_disabled_preserves_action_heads_acceleration_and_stop(self):
    for generation in (None, 10, 12):
      for head in ('action', 'desired_curvature', 'plan'):
        for speed, accel in ((20.0, 0.6), (0.2, 0.05)):
          with self.subTest(generation=generation, head=head, speed=speed):
            state = model_state(generation)
            previous = log.ModelDataV2.Action(desiredCurvature=-0.002, desiredAcceleration=-0.3)
            output = model_output(head=head, v=speed, accel=accel, center=0.5)
            action = state.get_action_from_model(output, previous, 0.4, 0.4, speed)
            expected_accel = float(np.float32(accel + (previous.desiredAcceleration - accel) * np.exp(-DT_MDL / 0.3)))
            expected_curvature = 0.00125
            if generation is not None and generation >= 10:
              expected_curvature = (previous.desiredCurvature if speed <= 0.3 else
                                    expected_curvature + (previous.desiredCurvature - expected_curvature) * np.exp(-DT_MDL / 0.2))
            self.assertAlmostEqual(action.desiredAcceleration, expected_accel, places=7)
            self.assertAlmostEqual(action.desiredCurvature, float(np.float32(expected_curvature)), places=9)
            self.assertEqual(action.shouldStop, speed < 0.3 and accel < 0.1)
            nominal = action.to_dict()
            bf = Blendful()
            bf.update(output, speed, True, state.model_curvature, enabled=False)
            messages = action_messages(action)
            modeld.fill_blendful_msg(bf, *messages)
            self.assertEqual(messages[0].modelV2.action.to_dict(), nominal)
            self.assertEqual(messages[1].drivingModelData.action.to_dict(), nominal)
            self.assertEqual(action.to_dict(), nominal)
            self.assertAlmostEqual(messages[2].modelDataV2SP.modelCurvature, 0.00125)

  def test_nominal_smoothing_never_contains_feature_history(self):
    for generation in (None, 10, 12):
      for head in ('action', 'desired_curvature', 'plan'):
        with self.subTest(generation=generation, head=head):
          enabled_state, baseline_state = model_state(generation), model_state(generation)
          previous = baseline_previous = log.ModelDataV2.Action()
          bf = Blendful()
          for index in range(105):
            speed = 0.2 if index == 102 else 20.0
            current = model_output(head=head, v=speed, center=0.5 if index < 100 else -0.5)
            if index == 100:
              current['lane_lines_prob'][:] = 0.0
            enabled = index != 101
            nominal = enabled_state.get_action_from_model(current, previous, 0.4, 0.4, speed)
            baseline = baseline_state.get_action_from_model(current, baseline_previous, 0.4, 0.4, speed)
            self.assertEqual(nominal.to_dict(), baseline.to_dict())
            previous, baseline_previous = nominal, baseline
            nominal_before_publish = nominal.to_dict()
            bf.update(current, speed, True, enabled_state.model_curvature, enabled=enabled)
            messages = action_messages(nominal)
            modeld.fill_blendful_msg(bf, *messages)
            published = messages[0].modelV2.action
            self.assertEqual(published.to_dict(), messages[1].drivingModelData.action.to_dict())
            self.assertEqual(nominal.to_dict(), nominal_before_publish)
            self.assertEqual(published.desiredAcceleration, baseline.desiredAcceleration)
            self.assertEqual(published.shouldStop, baseline.shouldStop)
            self.assertAlmostEqual(published.desiredCurvature, nominal.desiredCurvature + bf.curvature, places=9)
            self.assertEqual(messages[2].valid, messages[0].valid)
            self.assertEqual(messages[2].logMonoTime, messages[0].logMonoTime)
            if index == 99:
              self.assertGreater(bf.curvature, 0.0)
              self.assertGreater(published.desiredCurvature, baseline.desiredCurvature)
            elif index == 100:
              # Healthy missing lines retain the source correction history.
              self.assertGreater(bf.curvature, 0.0)
            elif index in (101, 102):
              self.assertEqual(bf.curvature, 0.0)
              self.assertEqual(published.to_dict(), baseline.to_dict())
            elif index == 103:
              fresh = Blendful()
              self.assertEqual(bf.curvature, fresh.update(current, speed, True, enabled_state.model_curvature))

  def test_planplus_remains_lateral_only_and_uses_the_selected_plan(self):
    state = model_state(12)
    output = model_output(head='plan', accel=0.6)
    plus = np.zeros_like(output['plan'])
    plus[0, :, 11] = output['plan'][0, :, 11] * 0.5
    plus[0, :, 14] = output['plan'][0, :, 14] * 0.5
    output['plan'] += plus
    output['planplus'] = plus
    state.PLANPLUS_CONTROL = 0.4
    result = state.get_action_from_model(output, log.ModelDataV2.Action(), 0.4, 0.4, 20.0)
    self.assertAlmostEqual(state.model_curvature, 0.00125 * 1.2)
    self.assertAlmostEqual(result.desiredAcceleration, float(np.float32(0.6 * (1 - np.exp(-DT_MDL / 0.3)))), places=7)

  def test_diagnostics_are_complete_on_invalid_and_disabled_frames(self):
    bf = Blendful()
    nominal = log.ModelDataV2.Action(desiredCurvature=-0.0, desiredAcceleration=0.1, shouldStop=True)
    bf.update(model_output(center=0.5), 20.0, True, 0.00125)
    bf.reset(0.00125)  # Broken input uses the explicit reset adapter, not ordinary source fading.
    messages = action_messages(nominal, valid=False)
    modeld.fill_blendful_msg(bf, *messages)
    self.assertFalse(messages[2].valid)
    self.assertEqual(messages[2].modelDataV2SP.to_dict(), {
      'laneTurnDirection': 'none', 'leftLaneChangeEdgeBlock': False, 'rightLaneChangeEdgeBlock': False,
      'blendfulCurvature': 0.0, 'blendfulTrust': 0.0, 'modelCurvature': float(np.float32(0.00125)),
      'blendfulGoingAround': False, 'blendfulLeftProb': 0.0, 'blendfulRightProb': 0.0,
      'blendfulLaneWidth': float(np.float32(3.6)),
    })
    self.assertTrue(np.signbit(messages[0].modelV2.action.desiredCurvature))
    self.assertEqual(messages[0].modelV2.action.to_dict(), nominal.to_dict())

  def test_input_health_is_distinct_from_lateral_engagement(self):
    now = 10_000_000_000
    sm, data, _ = healthy_inputs(now)
    transforms = {'img': np.eye(3), 'big_img': np.eye(3)}
    self.assertTrue(modeld.blendful_inputs_healthy(sm, True, now, transforms, now_ns=now))
    for field in ('seen', 'valid', 'alive', 'freq_ok'):
      for service in ('carState', 'carControl', 'extrinsicsCalibration'):
        with self.subTest(field=field, service=service):
          getattr(sm, field)[service] = False
          self.assertFalse(modeld.blendful_inputs_healthy(sm, True, now, transforms, now_ns=now))
          getattr(sm, field)[service] = True
    for service in ('carState', 'carControl', 'extrinsicsCalibration'):
      max_age = int(10.0 / modeld.SERVICE_LIST[service].frequency * 1e9)
      for stamp in (0, now + 1, now - max_age):
        with self.subTest(service=service, stamp=stamp):
          sm.logMonoTime[service] = stamp
          self.assertFalse(modeld.blendful_inputs_healthy(sm, True, now, transforms, now_ns=now))
      sm.logMonoTime[service] = now
    for status in (log.ExtrinsicsCalibration.Status.uncalibrated, log.ExtrinsicsCalibration.Status.invalid,
                   log.ExtrinsicsCalibration.Status.recalibrating):
      data['extrinsicsCalibration'].calStatus = status
      self.assertFalse(modeld.blendful_inputs_healthy(sm, True, now, transforms, now_ns=now))
    data['extrinsicsCalibration'].calStatus = log.ExtrinsicsCalibration.Status.calibrated
    data['carControl'].latActive = False
    self.assertTrue(modeld.blendful_inputs_healthy(sm, True, now, transforms, now_ns=now))
    data['carControl'].latActive = True
    self.assertFalse(modeld.blendful_inputs_healthy(sm, False, now, transforms, now_ns=now))
    for stamp in (0, now + 1, now - modeld.BLENDFUL_MAX_FRAME_AGE_NS - 1):
      self.assertFalse(modeld.blendful_inputs_healthy(sm, True, stamp, transforms, now_ns=now))
    for bad_transforms in ({}, {'img': np.ones((2, 3))}, {'img': np.full((3, 3), np.nan)}):
      self.assertFalse(modeld.blendful_inputs_healthy(sm, True, now, bad_transforms, now_ns=now))

  def test_submaster_qualifies_100hz_controls_sampled_by_20hz_modeld(self):
    services = ['carState', 'carControl']
    sm = messaging.SubMaster(services, frequency=ModelConstants.MODEL_FREQ)
    for index in range(10):
      now = 10.0 + index * DT_MDL
      messages = [messaging.new_message(service, valid=True, logMonoTime=int(now * 1e9)) for service in services]
      sm.update_msgs(now, messages)
    self.assertTrue(sm.all_checks(services))


class FrameLoopFinished(Exception):
  pass


class TestBlendfulFrameLoop(OpenpilotTestCase):
  def run_frames(self, scenarios, chestnut=False):
    """Run real main/action/fill logic; only IO, inference and unrelated controllers are fakes."""
    params = Params()
    params.put('CarParams', modeld.car.CarParams(brand='honda', longitudinalActuatorDelay=0.2).to_bytes(), block=True)
    put_bool = params.put_bool
    self.enterContext(mock.patch.object(params, 'put_bool', side_effect=lambda key, value, block=False: put_bool(key, value, block=True)))
    get_bool = self.enterContext(mock.patch.object(params, 'get_bool', wraps=params.get_bool))
    current = {'index': -1, 'scenario': {}, 'clock_ns': 10_000_000_000}
    sm, data, dh = healthy_inputs(current['clock_ns'])
    data.update({
      'driverMonitoringState': SimpleNamespace(isRHD=False),
      'narrowRoadCameraState': SimpleNamespace(frameId=0, sensor='fixture'),
      'deviceState': SimpleNamespace(deviceType='fixture'),
      'lateralDelay': SimpleNamespace(lateralDelay=0.2),
    })
    sm.seen.update(dict.fromkeys(data, True))
    sm.frame = 0
    sm.updated = {'extrinsicsCalibration': True}
    data['extrinsicsCalibration'].height = [1.22]
    dh.lane_change_direction = log.LaneChangeDirection.none
    dh.lane_turn_direction = 0

    def update_desire(*args):
      dh.desire = current['scenario'].get('desire', log.Desire.none)
      dh.lane_change_state = current['scenario'].get('lane_change', log.LaneChangeState.off)

    dh.update = mock.Mock(side_effect=update_desire)
    client = mock.Mock(width=1928, height=1208, buffer_len=1, frame_id=0, timestamp_sof=0, timestamp_eof=0)
    client.connect.return_value = True

    def recv():
      current['index'] += 1
      if current['index'] >= len(scenarios):
        raise FrameLoopFinished
      scenario = current['scenario'] = scenarios[current['index']]
      params.put_bool('Blendful', scenario.get('enabled', True), block=True)
      current['clock_ns'] += scenario.get('gap_ns', int(DT_MDL * 1e9))
      if scenario.get('missing_camera'):
        return None
      client.frame_id += 1
      client.timestamp_sof = current['clock_ns']
      client.timestamp_eof = client.timestamp_sof - scenario.get('capture_offset_ns', 0)
      current['clock_ns'] += scenario.get('frame_age_ns', 0)
      data['narrowRoadCameraState'].frameId = client.frame_id
      data['carState'].vEgo = scenario.get('speed', 20.0)
      data['carControl'].latActive = scenario.get('lat_active', True)
      data['extrinsicsCalibration'].calStatus = scenario.get('calibration', log.ExtrinsicsCalibration.Status.calibrated)
      sm.frame += 1
      for service in ('carState', 'carControl', 'extrinsicsCalibration'):
        sm.logMonoTime[service] = current['clock_ns']
        sm.valid[service] = service != scenario.get('invalid_service')
        sm.alive[service] = True
        sm.freq_ok[service] = True
      if scenario.get('stale_control'):
        sm.logMonoTime['carControl'] -= 100_000_000
      return object()

    client.recv.side_effect = recv
    vision_client = mock.Mock(return_value=client)
    vision_client.available_streams.return_value = [modeld.VisionStreamType.VISION_STREAM_NARROW_ROAD]
    nominal_actions, previous_actions, outputs = {}, {}, {}
    get_nominal_action = modeld.ModelState.get_action_from_model

    def make_model(is_chestnut):
      state = model_state(12, chestnut=is_chestnut)
      state.warmup = mock.Mock()

      def run_model(*args):
        scenario = current['scenario']
        if scenario.get('fail_big') and state.chestnut:
          raise RuntimeError('offline eGPU failure')
        if scenario.get('missing_output'):
          return None
        output = model_output(center=scenario.get('center', 0.5), v=scenario.get('speed', 20.0))
        if scenario.get('missing_lines'):
          output['lane_lines_prob'][:] = 0.0
        outputs[current['index']] = output
        return output

      def get_action(output, previous, *args):
        previous_actions[current['index']] = previous.to_dict()
        result = get_nominal_action(state, output, previous, *args)
        nominal_actions[current['index']] = result.to_dict()
        return result

      state.run = mock.Mock(side_effect=run_model)
      state.get_action_from_model = mock.Mock(side_effect=get_action)
      return state

    large, small = make_model(True), make_model(False)
    pub = mock.Mock()
    published = {}

    def send(service, message):
      record = published.setdefault(current['index'], {})
      if service == 'modelV2':
        record['model'] = message.modelV2.action.to_dict()
        record['valid'] = message.valid
        record['stamp'] = message.logMonoTime
        record['lane_change'] = message.modelV2.meta.laneChangeState
      elif service == 'drivingModelData':
        record['driving'] = message.drivingModelData.action.to_dict()
      elif service == 'modelDataV2SP':
        record['sp'] = message.modelDataV2SP.to_dict()
        record['sp_valid'] = message.valid
        record['sp_stamp'] = message.logMonoTime

    pub.send.side_effect = send
    camera = SimpleNamespace(narrow_road=SimpleNamespace(intrinsics=np.eye(3)), wide_road=SimpleNamespace(intrinsics=np.eye(3)))
    offset = mock.Mock()
    offset.update.side_effect = lambda main, extra, *args: (main, extra)
    submaster_factory = mock.Mock(return_value=sm)
    patches = (
      mock.patch.object(modeld, 'config_realtime_process'), mock.patch.object(modeld, 'setproctitle'),
      mock.patch.object(modeld, 'chestnut_present', return_value=chestnut), mock.patch.object(modeld, 'Params', return_value=params),
      mock.patch.object(modeld, 'VisionIpcClient', vision_client),
      mock.patch.object(modeld, 'ModelState', side_effect=lambda **kwargs: large if kwargs['chestnut'] else small),
      mock.patch.object(modeld, 'PubMaster', return_value=pub), mock.patch.object(modeld, 'SubMaster', submaster_factory),
      mock.patch.object(modeld, 'ChestnutState', return_value=SimpleNamespace(big=chestnut, send=mock.Mock())),
      mock.patch.object(modeld, 'DesireHelper', return_value=dh), mock.patch.object(modeld, 'load_meta_constants', return_value=Meta),
      mock.patch.object(modeld, 'RoadEdgeLaneChangeController', return_value=SimpleNamespace(update_and_fill=lambda *args: (False, False))),
      mock.patch.object(modeld, 'CameraOffsetHelper', return_value=offset),
      mock.patch.object(modeld, 'DEVICE_CAMERAS', {('fixture', 'fixture'): camera}),
      mock.patch.object(modeld, 'get_warp_matrix', return_value=np.eye(3)),
      mock.patch.object(modeld.time, 'monotonic_ns', side_effect=lambda: current['clock_ns']),
      mock.patch('requests.sessions.Session.request', side_effect=AssertionError('offline Blendful test attempted network access')),
    )
    for patch in patches:
      self.enterContext(patch)
    with self.assertRaises(FrameLoopFinished):
      modeld.main()
    self.assertEqual(submaster_factory.call_args.kwargs, {'frequency': ModelConstants.MODEL_FREQ})
    self.assertEqual(sum(call.args == ('Blendful',) for call in get_bool.call_args_list), len(nominal_actions))
    for index, record in published.items():
      self.assertEqual(record['model'], record['driving'])
      self.assertEqual(record['model']['desiredAcceleration'], nominal_actions[index]['desiredAcceleration'])
      self.assertEqual(record['model']['shouldStop'], nominal_actions[index]['shouldStop'])
      self.assertAlmostEqual(record['model']['desiredCurvature'],
                             nominal_actions[index]['desiredCurvature'] + record['sp']['blendfulCurvature'], places=9)
      self.assertEqual(record['sp_valid'], record['valid'])
      self.assertEqual(record['sp_stamp'], record['stamp'])
    ordered = list(nominal_actions)
    for last_index, index in zip(ordered, ordered[1:], strict=False):
      self.assertEqual(previous_actions[index], nominal_actions[last_index])
    return published, outputs, nominal_actions, large, small

  def test_bad_inputs_and_toggle_reset_in_real_main_loop(self):
    disqualifications = (
      {'enabled': False},
      {'invalid_service': 'carState'}, {'invalid_service': 'carControl'}, {'stale_control': True},
      {'calibration': log.ExtrinsicsCalibration.Status.invalid}, {'frame_age_ns': 251_000_000},
      {'missing_output': True}, {'missing_camera': True},
    )
    scenarios = [{} for _ in range(60)]
    for failure in disqualifications:
      scenarios.extend((failure, {'center': -0.5}))
    published, outputs, _, _, _ = self.run_frames(scenarios)
    self.assertGreater(published[59]['sp']['blendfulCurvature'], 0.0)
    for offset, failure in enumerate(disqualifications):
      failure_index = 60 + 2 * offset
      with self.subTest(failure=failure):
        if failure.get('missing_output') or failure.get('missing_camera'):
          self.assertNotIn(failure_index, published)
        else:
          sp = published[failure_index]['sp']
          self.assertEqual((sp['blendfulCurvature'], sp['blendfulTrust'], sp['blendfulLeftProb'], sp['blendfulRightProb']), (0.0,) * 4)
        restored_index = failure_index + 1
        fresh = Blendful()
        expected = fresh.update(outputs[restored_index], 20.0, True, 0.00125)
        self.assertAlmostEqual(published[restored_index]['sp']['blendfulCurvature'], expected, places=11)
        self.assertAlmostEqual(published[restored_index]['sp']['blendfulTrust'], fresh.trust, places=8)

  def test_source_history_fading_and_inactive_accumulation_in_real_main_loop(self):
    scenarios = [{'lat_active': False} for _ in range(60)]
    scenarios.extend(({'lat_active': False, 'missing_lines': True}, {'desire': log.Desire.turnLeft},
                      {'lane_change': log.LaneChangeState.preLaneChange}, {'speed': 4.99}, {'lat_active': False}))
    published, outputs, _, _, _ = self.run_frames(scenarios)
    self.assertGreater(published[59]['sp']['blendfulCurvature'], 0.0)
    self.assertGreater(published[59]['sp']['blendfulTrust'], 0.5)
    self.assertGreater(published[60]['sp']['blendfulCurvature'], 0.0)
    self.assertGreater(published[60]['sp']['blendfulLeftProb'], 0.0)
    self.assertGreater(published[61]['sp']['blendfulCurvature'], 0.0)
    self.assertLess(published[61]['sp']['blendfulCurvature'], published[60]['sp']['blendfulCurvature'])
    for index in (61, 62, 63):
      self.assertEqual(published[index]['sp']['blendfulTrust'], 0.0)
    # The runtime dispatch must agree with the source algorithm driven directly:
    # lateral engagement is independent, desire/lane-change controls fading.
    source_behavior = Blendful()
    for index, scenario in enumerate(scenarios):
      active = index not in (61, 62)
      source_behavior.update(outputs[index], scenario.get('speed', 20.0), active, 0.00125)
      self.assertAlmostEqual(published[index]['sp']['blendfulCurvature'], source_behavior.curvature, places=9)
      self.assertAlmostEqual(published[index]['sp']['blendfulTrust'], source_behavior.trust, places=7)

  def test_fallback_and_capture_gap_cannot_revive_old_correction(self):
    scenarios = [{} for _ in range(60)]
    scenarios.extend(({'fail_big': True}, {'center': -0.5}, {'center': 0.5, 'gap_ns': 300_000_000}))
    published, outputs, _, large, small = self.run_frames(scenarios, chestnut=True)
    self.assertGreater(published[59]['sp']['blendfulCurvature'], 0.0)
    self.assertNotIn(60, published)
    self.assertEqual(large.run.call_count, 61)
    self.assertEqual(small.run.call_count, 2)
    for index in (61, 62):
      fresh = Blendful()
      expected = fresh.update(outputs[index], 20.0, True, 0.00125)
      self.assertAlmostEqual(published[index]['sp']['blendfulCurvature'], expected, places=11)
      self.assertAlmostEqual(published[index]['sp']['blendfulTrust'], fresh.trust, places=8)

  def test_duplicate_or_reversed_capture_is_not_a_new_observation(self):
    scenarios = [{} for _ in range(60)]
    scenarios.extend(({'capture_offset_ns': 50_000_000}, {'capture_offset_ns': 150_000_000},
                      {'capture_offset_ns': 150_000_000}, {'center': -0.5}))
    published, outputs, _, _, _ = self.run_frames(scenarios)
    self.assertGreater(published[59]['sp']['blendfulCurvature'], 0.0)
    for index in (60, 61, 62):
      self.assertEqual(published[index]['sp']['blendfulCurvature'], 0.0)
      self.assertEqual(published[index]['sp']['blendfulTrust'], 0.0)
    fresh = Blendful()
    expected = fresh.update(outputs[63], 20.0, True, 0.00125)
    self.assertAlmostEqual(published[63]['sp']['blendfulCurvature'], expected, places=11)
    self.assertAlmostEqual(published[63]['sp']['blendfulTrust'], fresh.trust, places=8)
