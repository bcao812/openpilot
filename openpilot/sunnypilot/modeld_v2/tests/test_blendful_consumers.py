"""Offline characterization of the Blendful publication to adaptive SCC boundary."""
from unittest import mock

from openpilot.cereal import log, messaging
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot.modeld_v2 import modeld
from openpilot.sunnypilot.modeld_v2.blendful import Blendful
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants
from openpilot.sunnypilot.modeld_v2.tests.test_blendful_integration import action_messages, model_output, model_state
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import vision_controller
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.tests.test_vision_controller import DatedState


SPEED = 20.0
FRAME_NS = int(DT_MDL * 1e9)


class PublishedCurve:
  """Real correction, action publication and SCC; only model/vehicle inputs are fixtures."""

  def __init__(self):
    self.blendful = Blendful()
    self.model_state = model_state()
    self.previous = log.ModelDataV2.Action()
    self.controller = vision_controller.SmartCruiseControlVision(adaptive_curve_speed=True)
    cs = messaging.new_message('carState')
    cs.carState.vEgo = SPEED
    controls = messaging.new_message('controlsState')
    controls.controlsState.lateralControlState.init('pidState')
    self.sm = DatedState({'carState': cs.carState, 'controlsState': controls.controlsState})

  def step(self, *, enabled=True, lines=True, desired=.9, predicted=.1, current=.1,
           lateral_active=True, saturated=True, time_delta=FRAME_NS):
    # The action head may request more curvature than the predicted trajectory or
    # measured motion. A fixed 0.5 m lane offset generates the correction through
    # the real Blendful fit and filters, without injecting a desired correction.
    output = model_output(v=SPEED, k=desired / SPEED ** 2, center=.5)
    if not lines:
      output['lane_lines_prob'][:] = 0.
    self.nominal = self.model_state.get_action_from_model(output, self.previous, .4, .4, SPEED)
    self.previous = self.nominal
    # No turn/lane-change desire in this fixture. As in source, ordinary
    # lateral inactivity does not stop the healthy model-side accumulation.
    self.blendful.update(output, SPEED, True, self.model_state.model_curvature, enabled=enabled)
    self.messages = action_messages(self.nominal)
    self.sm.logMonoTime['modelV2'] += time_delta
    self.messages[0].logMonoTime = self.sm.logMonoTime['modelV2']
    modeld.fill_blendful_msg(self.blendful, *self.messages)
    model = self.messages[0].modelV2
    # Keep the unrelated prediction fixed across the with/without-Blendful pair.
    model.velocity.x = [SPEED] * len(ModelConstants.T_IDXS)
    model.orientationRate.z = [predicted / SPEED] * len(ModelConstants.T_IDXS)
    self.sm['modelV2'] = model
    controls = self.sm['controlsState']
    controls.curvature = current / SPEED ** 2
    lateral = controls.lateralControlState.pidState
    lateral.active, lateral.saturated = lateral_active, saturated
    self.controller.update(self.sm, True, False, SPEED, .4, 30.)

  @property
  def published_demand(self):
    return self.sm['modelV2'].action.desiredCurvature * SPEED ** 2


class TestBlendfulSccConsumers(OpenpilotTestCase):
  def setUp(self):
    params = mock.Mock()
    params.get_bool.return_value = True
    self.enterContext(mock.patch.object(vision_controller, 'Params', return_value=params))

  def establish_pair(self):
    feature, control = PublishedCurve(), PublishedCurve()
    for _ in range(100):
      feature.step()
      control.step(enabled=False)
      self.assertFalse(control.controller.is_active)
      self.assertFalse(control.controller.steering_limited)
      self.assertEqual(control.controller.output_v_target, V_CRUISE_UNSET)
      self.assertAlmostEqual(control.published_demand, .9, places=6)
      if feature.published_demand >= 1.01:
        break
    self.assertGreaterEqual(feature.published_demand, 1.01)
    self.assertAlmostEqual(feature.nominal.desiredCurvature * SPEED ** 2, .9, places=6)
    self.assertGreater(feature.blendful.curvature, 0.)
    self.assertAlmostEqual(feature.messages[2].modelDataV2SP.blendfulCurvature, feature.blendful.curvature, places=9)
    self.assertEqual(feature.messages[0].modelV2.action.to_dict(), feature.messages[1].drivingModelData.action.to_dict())
    self.assertTrue(feature.controller.is_active)
    self.assertTrue(feature.controller.steering_limited)
    self.assertEqual(feature.controller.output_v_target, SPEED)
    self.assertEqual(feature.controller.output_a_target, 0.)
    return feature, control

  def test_withdrawn_correction_keeps_scc_latched(self):
    # This characterizes the current action/SCC coupling; the 0.5 s curve-clear
    # timer is not an upper bound while demand remains above the 0.8 release line.
    feature, control = self.establish_pair()
    for _ in range(40):
      feature.step(enabled=False)
      control.step(enabled=False)
      self.assertEqual(feature.blendful.curvature, 0.)
      self.assertEqual(feature.messages[2].modelDataV2SP.blendfulCurvature, 0.)
      self.assertEqual(feature.sm['modelV2'].action.to_dict(), control.sm['modelV2'].action.to_dict())
      self.assertAlmostEqual(feature.published_demand, .9, places=6)
      self.assertTrue(feature.controller.steering_limited)
      self.assertTrue(feature.controller.is_active)
      self.assertEqual(feature.controller.output_v_target, SPEED)
      self.assertEqual(feature.controller.output_a_target, 0.)
      self.assertFalse(control.controller.steering_limited)
      self.assertFalse(control.controller.is_active)
      self.assertEqual(control.controller.output_v_target, V_CRUISE_UNSET)
      self.assertEqual(control.controller.output_a_target, .4)

  def test_missing_lines_keep_source_history_in_published_demand(self):
    feature, control = self.establish_pair()
    for _ in range(5):
      feature.step(lines=False)
      control.step(lines=False, enabled=False)
      self.assertGreater(feature.blendful.curvature, 0.)
      self.assertGreater(feature.blendful.l_prob, 0.)
      self.assertGreater(feature.blendful.r_prob, 0.)
      self.assertGreater(feature.published_demand, control.published_demand)
      self.assertTrue(feature.controller.steering_limited)
      self.assertTrue(feature.controller.is_active)
      self.assertFalse(control.controller.is_active)

  def test_scc_releases_on_current_authority_demand_or_timestamp_change(self):
    releases = {
      'lateral_exit': {'lateral_active': False},
      'unsaturated': {'saturated': False},
      'demand_below_release': {'desired': .79},
      'model_gap': {'time_delta': 300_000_000},
      'model_time_reversed': {'time_delta': -FRAME_NS},
    }
    for reason, release in releases.items():
      with self.subTest(reason=reason):
        feature, _ = self.establish_pair()
        feature.step(enabled=False, **release)
        self.assertEqual(feature.blendful.curvature, 0.)
        self.assertFalse(feature.controller.steering_limited)
        self.assertGreater(feature.controller.output_v_target, SPEED)
        self.assertGreater(feature.controller.output_a_target, 0.)
        if reason in ('model_gap', 'model_time_reversed'):
          self.assertFalse(feature.controller.is_active)
        # With no renewed >=1.0 demand, the ordinary curve timer can now finish.
        for _ in range(10):
          feature.step(enabled=False)
          self.assertFalse(feature.controller.steering_limited)
        self.assertFalse(feature.controller.is_active)
        self.assertEqual(feature.controller.output_v_target, V_CRUISE_UNSET)

  def test_real_curve_still_limits_speed_with_or_without_blendful(self):
    feature, control = PublishedCurve(), PublishedCurve()
    for _ in range(60):
      curve = {'desired': 2.5, 'predicted': 3., 'current': 2.3, 'saturated': False}
      feature.step(**curve)
      control.step(enabled=False, **curve)
      self.assertEqual(feature.controller.is_active, control.controller.is_active)
      self.assertEqual(feature.controller.output_v_target, control.controller.output_v_target)
      self.assertEqual(feature.controller.output_a_target, control.controller.output_a_target)
    self.assertGreater(feature.blendful.curvature, 0.)
    self.assertTrue(feature.controller.is_active)
    self.assertFalse(feature.controller.steering_limited)
    self.assertLess(feature.controller.output_v_target, SPEED)
    self.assertLess(feature.controller.output_a_target, 0.)
