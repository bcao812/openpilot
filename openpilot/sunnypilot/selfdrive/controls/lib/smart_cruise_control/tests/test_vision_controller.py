"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from typing import Any

import numpy as np
from openpilot.common.parameterized import parameterized

import openpilot.cereal.messaging as messaging
from openpilot.cereal import custom, log
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import MIN_V
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.vision_controller import SmartCruiseControlVision, _ENTERING_PRED_LAT_ACC_TH
from openpilot.common.test import OpenpilotTestCase

VisionState = custom.LongitudinalPlanSP.SmartCruiseControl.VisionState


class DatedState(dict):
  def __init__(self, values):
    super().__init__(values)
    self.logMonoTime = {'modelV2': 1_000_000_000}


def _th_above_f32(th: float) -> float:
  """
  Return the next representable float32 *above* `th`.
  This avoids flaky comparisons around thresholds due to float32 rounding.
  """
  th32 = np.float32(th)
  above32 = np.nextafter(th32, np.float32(np.inf), dtype=np.float32)
  return float(above32)


def _build_single_spike_filtered(n: int, base: float = 1.0) -> np.ndarray:
  """
  Create an array where max() is >= threshold but p97 is < threshold.
  This demonstrates the behavior difference vs np.amax().

  Note: We intentionally construct using float32-representable values to match
  the data path through cereal/capnp.
  """
  th = float(_ENTERING_PRED_LAT_ACC_TH)
  th32 = float(np.float32(th))

  # numpy percentile default is linear interpolation: idx=(n-1)*p/100
  idx = (n - 1) * 0.97
  w = float(idx - np.floor(idx))

  base32 = float(np.float32(base))

  # Choose spike so that p97 = base + w*(spike-base) < th
  # -> spike < base + (th-base)/w. Use a margin (0.9) and ensure spike >= th.
  if w == 0.0:
    spike = th32 + 1.0
  else:
    spike = base32 + (th32 - base32) / w * 0.9
    spike = max(spike, th32 + 0.01)

  arr = np.full(n, base32, dtype=np.float32)
  arr[-1] = np.float32(spike)
  return arr


def generate_modelV2():
  model = messaging.new_message('modelV2')
  position = log.XYZTData.new_message()
  speed = 30
  position.x = [float(x) for x in (speed + 0.5) * np.array(ModelConstants.T_IDXS)]
  model.modelV2.position = position
  orientation = log.XYZTData.new_message()
  curvature = 0.05
  orientation.x = [float(curvature) for _ in ModelConstants.T_IDXS]
  orientation.y = [0.0 for _ in ModelConstants.T_IDXS]
  model.modelV2.orientation = orientation
  orientationRate = log.XYZTData.new_message()
  orientationRate.z = [float(z) for z in ModelConstants.T_IDXS]
  model.modelV2.orientationRate = orientationRate
  velocity = log.XYZTData.new_message()
  velocity.x = [float(x) for x in (speed + 0.5) * np.ones_like(ModelConstants.T_IDXS)]
  velocity.x[0] = float(speed)  # always start at current speed
  model.modelV2.velocity = velocity
  acceleration = log.XYZTData.new_message()
  acceleration.x = [float(x) for x in np.zeros_like(ModelConstants.T_IDXS)]
  acceleration.y = [float(y) for y in np.zeros_like(ModelConstants.T_IDXS)]
  model.modelV2.acceleration = acceleration

  return model


def generate_carState():
  car_state = messaging.new_message('carState')
  speed = 30
  v_cruise = 50
  car_state.carState.vEgo = float(speed)
  car_state.carState.standstill = False
  car_state.carState.vCruise = float(v_cruise * 3.6)

  return car_state


def generate_controlsState():
  controls_state = messaging.new_message('controlsState')
  controls_state.controlsState.curvature = 0.05

  return controls_state


class TestSmartCruiseControlVision(OpenpilotTestCase):

  def setup_method(self):
    self.params = Params()
    self.reset_params()
    self.scc_v = SmartCruiseControlVision()

    mdl = generate_modelV2()
    cs = generate_carState()
    controls_state = generate_controlsState()
    self.sm: Any = {'modelV2': mdl.modelV2, 'carState': cs.carState, 'controlsState': controls_state.controlsState}

  def reset_params(self):
    self.params.put_bool("SmartCruiseControlVision", True, block=True)

  def test_initial_state(self):
    assert self.scc_v.state == VisionState.disabled
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET
    assert self.scc_v.output_a_target == 0.

  def _set_curve(self, speed, predicted_lateral, current_lateral):
    self.sm['modelV2'].velocity.x = [float(speed)] * len(ModelConstants.T_IDXS)
    self.sm['modelV2'].orientationRate.z = [float(predicted_lateral / speed)] * len(ModelConstants.T_IDXS)
    self.sm['controlsState'].curvature = current_lateral / speed ** 2

  def _adaptive_controller(self, *, speed, predicted_lateral, current_lateral):
    controller = SmartCruiseControlVision(adaptive_curve_speed=True)
    self._set_curve(speed, predicted_lateral, current_lateral)
    controller.update(self.sm, True, False, speed, 0., 30.)
    controller.update(self.sm, True, False, speed, 0., 30.)
    return controller

  def test_comfortable_turn_recovers_above_entry_speed_with_bounded_acceleration(self):
    controller = self._adaptive_controller(speed=10., predicted_lateral=1.5, current_lateral=.5)
    assert controller.state == VisionState.entering
    assert abs(controller.output_v_target - 10. * (2. / 1.5) ** .5) < 1e-5
    assert 0. < controller.output_a_target <= .5
    for _ in range(20):
      controller.update(self.sm, True, False, 9.8, 0., 30.)
      assert controller.output_v_target > 10.
      assert controller.driver_speed is None
      assert 0. < controller.output_a_target <= .5

  def test_curve_requiring_deceleration_keeps_geometric_limit_without_double_horizon(self):
    speed = 25.1 / 3.6
    controller = self._adaptive_controller(speed=speed, predicted_lateral=2.25, current_lateral=1.17)
    assert controller.is_active
    assert abs(controller.output_v_target - speed * (2. / 2.25) ** .5) < 1e-5
    assert controller.output_v_target < speed
    assert controller.output_a_target < 0.
    # The old second horizon deduction made this roughly 14.5 km/h.
    assert controller.output_v_target * 3.6 > 23.

  def test_adaptive_target_still_slows_for_excess_current_lateral_acceleration(self):
    controller = self._adaptive_controller(speed=20., predicted_lateral=1.5, current_lateral=3.)
    assert controller.output_v_target <= 20. * (2. / 3.) ** .5 + 1e-5
    assert controller.output_a_target < 0.

  def test_non_pedal_override_and_disable_clear_curve_preference(self):
    controller = self._adaptive_controller(speed=10., predicted_lateral=1.5, current_lateral=.5)
    controller.update(self.sm, True, True, 10., 0., 30.)
    assert controller.driver_speed is None
    assert controller.output_v_target == V_CRUISE_UNSET
    controller.update(self.sm, False, False, 10., 0., 30.)
    assert controller.driver_speed is None

  def test_gas_release_uses_release_speed_for_this_curve_even_with_delayed_override_bit(self):
    controller = self._adaptive_controller(speed=10., predicted_lateral=1.5, current_lateral=.5)
    self.sm['carState'].gasPressed = True
    for speed in (12., 14.):
      self._set_curve(speed, 1.5, .5)
      controller.update(self.sm, True, True, speed, 0., 30.)
      assert controller.state == VisionState.overriding
      assert controller.output_v_target == V_CRUISE_UNSET
    self.sm['carState'].gasPressed = False
    self._set_curve(12., 1.5, .5)
    controller.update(self.sm, True, True, 12., 0., 30.)
    assert controller.driver_speed == 12.
    assert not controller.is_active
    for _ in range(20):
      controller.update(self.sm, True, False, 12., 0., 30.)
      assert controller.output_v_target == 12.
      assert controller.output_a_target == 0.

    # A second pedal use replaces the setpoint; it never remembers a peak.
    self.sm['carState'].gasPressed = True
    self._set_curve(13., 1.5, .5)
    controller.update(self.sm, True, False, 13., 0., 30.)
    assert not controller.is_active
    self.sm['carState'].gasPressed = False
    controller.update(self.sm, True, False, 13., 0., 30.)
    assert controller.output_v_target == controller.driver_speed == 13.

    controller.update(self.sm, True, False, 13., 0., 11.)
    assert controller.output_v_target == 11.
    self._set_curve(13., 3., .5)
    controller.update(self.sm, True, False, 13., 0., 30.)
    assert controller.output_v_target == 13.
    assert controller.output_a_target == 0.

  @parameterized.expand([(32.575 / 3.6, 2.633, .959),
                         (27.580 / 3.6, 1.865, 1.471),
                         (36.109 / 3.6, 3.267, 3.100)])
  def test_m003_driver_release_selects_speed_over_current_and_predicted_geometry(self, speed, predicted, current):
    controller = self._adaptive_controller(speed=8., predicted_lateral=1.5, current_lateral=.5)
    self._set_curve(speed, predicted, current)
    self.sm['carState'].gasPressed = True
    controller.update(self.sm, True, True, speed, .6, 60. / 3.6)
    self.sm['carState'].gasPressed = False
    for _ in range(12):
      controller.update(self.sm, True, False, speed, .6, 60. / 3.6)
      assert controller.driver_speed == speed
      assert controller.output_v_target == speed
      assert controller.output_a_target == 0.

  @parameterized.expand([('entering', .5), ('turning', 3.1)])
  def test_driver_speed_error_does_not_reapply_geometric_deceleration(self, _, current):
    controller = self._adaptive_controller(speed=10., predicted_lateral=3.267, current_lateral=current)
    self.sm['carState'].gasPressed = True
    controller.update(self.sm, True, True, 10., 0., 30.)
    self.sm['carState'].gasPressed = False
    controller.update(self.sm, True, False, 10., 0., 30.)
    for speed, acceleration in [(10.04, -.01), (9.6, .1), (7., .5)]:
      self._set_curve(speed, 3.267, current)
      controller.update(self.sm, True, False, speed, 0., 30.)
      assert controller.output_v_target == 10.
      assert abs(controller.output_a_target - acceleration) < 1e-8

  def test_driver_choice_preserves_saturation_slowdown_and_recovers_after_it_clears(self):
    controller = self._adaptive_controller(speed=10., predicted_lateral=3.267, current_lateral=3.1)
    self.sm['carState'].gasPressed = True
    controller.update(self.sm, True, True, 10., 0., 30.)
    self.sm['carState'].gasPressed = False
    controller.update(self.sm, True, False, 10., 0., 30.)
    assert controller.output_v_target == 10.
    lateral = self.sm['controlsState'].lateralControlState.init('pidState')
    lateral.active = lateral.saturated = True
    controller.update(self.sm, True, False, 9.6, 0., 30.)
    assert controller.driver_speed == 10.
    assert controller.output_v_target < 9.6
    assert controller.output_a_target < 0.
    lateral.saturated = False
    controller.update(self.sm, True, False, 9.6, 0., 30.)
    assert controller.output_v_target == 10.
    assert 0. < controller.output_a_target <= .5

  @parameterized.expand([('prediction',), ('current',)])
  def test_driver_choice_does_not_hide_invalid_curve_inputs(self, field):
    controller = self._adaptive_controller(speed=10., predicted_lateral=3.267, current_lateral=3.1)
    self.sm['carState'].gasPressed = True
    controller.update(self.sm, True, True, 10., 0., 30.)
    self.sm['carState'].gasPressed = False
    controller.update(self.sm, True, False, 10., 0., 30.)
    if field == 'prediction':
      self.sm['modelV2'].orientationRate.z = [float('nan')] * len(ModelConstants.T_IDXS)
    else:
      self.sm['controlsState'].curvature = float('nan')
    controller.update(self.sm, True, False, 10., 0., 30.)
    assert not np.isfinite(controller.output_v_target)
    assert not np.isfinite(controller.output_a_target)

  @parameterized.expand([('brake',), ('disable',), ('non_pedal_override',), ('curve_exit',)])
  def test_driver_preference_is_not_carried_into_another_cycle(self, reason):
    controller = self._adaptive_controller(speed=10., predicted_lateral=1.5, current_lateral=.5)
    self.sm['carState'].gasPressed = True
    controller.update(self.sm, True, True, 11., 0., 30.)
    self.sm['carState'].gasPressed = False
    controller.update(self.sm, True, False, 11., 0., 30.)
    assert controller.driver_speed == 11.
    self.sm['carState'].brakePressed = reason == 'brake'
    if reason == 'curve_exit':
      self._set_curve(11., 0., 0.)
    for _ in range(12):
      controller.update(self.sm, reason != 'disable', reason == 'non_pedal_override', 11., 0., 30.)
    assert controller.driver_speed is None
    self.sm['carState'].brakePressed = False
    self._set_curve(10., 1.5, .5)
    for _ in range(3):
      controller.update(self.sm, True, False, 10., 0., 30.)
    assert controller.output_v_target > 11.

  def test_straight_road_pedal_release_does_not_choose_the_next_curve_speed(self):
    controller = self._adaptive_controller(speed=10., predicted_lateral=0., current_lateral=0.)
    self.sm['carState'].gasPressed = True
    controller.update(self.sm, True, True, 12., 0., 30.)
    self.sm['carState'].gasPressed = False
    controller.update(self.sm, True, False, 12., 0., 30.)
    self._set_curve(15., 1.5, .5)
    controller.update(self.sm, True, False, 15., 0., 30.)
    assert controller.driver_speed is None
    assert controller.output_v_target > 15.

  def test_continuous_bends_share_driver_choice_until_the_curve_cycle_clears(self):
    controller = self._adaptive_controller(speed=10., predicted_lateral=3.267, current_lateral=3.1)
    self.sm['carState'].gasPressed = True
    controller.update(self.sm, True, True, 10., 0., 30.)
    self.sm['carState'].gasPressed = False
    controller.update(self.sm, True, False, 10., 0., 30.)
    # The existing SCC cycle uses curvature magnitude; a connected reverse
    # bend without a clear interval is still part of that same cycle.
    self._set_curve(10., -3.267, -3.1)
    for _ in range(12):
      controller.update(self.sm, True, False, 10., 0., 30.)
      assert controller.output_v_target == controller.driver_speed == 10.
      assert controller.output_a_target == 0.
    self._set_curve(10., 0., 0.)
    for _ in range(10):
      controller.update(self.sm, True, False, 10., 0., 30.)
    assert controller.driver_speed is None
    self._set_curve(10., 3.267, 3.1)
    controller.update(self.sm, True, False, 10., 0., 30.)
    assert controller.output_v_target < 10.
    assert controller.output_a_target < 0.

  def test_threshold_flicker_does_not_restart_the_curve_cycle(self):
    controller = self._adaptive_controller(speed=10., predicted_lateral=1.5, current_lateral=.5)
    for predicted in [1.05] * 6 + [1.4] * 2 + [1.05] * 6 + [1.4] * 2:
      self._set_curve(10., predicted, .5)
      controller.update(self.sm, True, False, 10., 0., 30.)
      assert controller.is_active
      assert controller.output_v_target > 10.
    self._set_curve(10., 0., 0.)
    for _ in range(10):
      controller.update(self.sm, True, False, 10., 0., 30.)
    assert controller.state == VisionState.enabled

  def test_reused_model_frame_does_not_advance_the_curve_clear_timer(self):
    self.sm = DatedState(self.sm)
    controller = self._adaptive_controller(speed=10., predicted_lateral=1.5, current_lateral=.5)
    self._set_curve(10., 0., 0.)
    for _ in range(30):
      controller.update(self.sm, True, False, 10., 0., 30.)
    assert controller.is_active
    for _ in range(10):
      self.sm.logMonoTime['modelV2'] += 50_000_000
      controller.update(self.sm, True, False, 10., 0., 30.)
    assert controller.state == VisionState.enabled

  def test_model_gap_discards_driver_preference_before_recovery(self):
    self.sm = DatedState(self.sm)
    controller = self._adaptive_controller(speed=10., predicted_lateral=1.5, current_lateral=.5)
    self.sm['carState'].gasPressed = True
    self.sm.logMonoTime['modelV2'] += 50_000_000
    controller.update(self.sm, True, True, 12., 0., 30.)
    self.sm['carState'].gasPressed = False
    self.sm.logMonoTime['modelV2'] += 50_000_000
    controller.update(self.sm, True, False, 12., 0., 30.)
    assert controller.driver_speed == 12.
    self._set_curve(20., 1.5, .5)
    self.sm.logMonoTime['modelV2'] += 500_000_000
    controller.update(self.sm, True, False, 20., 0., 30.)
    assert controller.driver_speed is None
    self.sm.logMonoTime['modelV2'] += 50_000_000
    controller.update(self.sm, True, False, 20., 0., 30.)
    assert controller.output_v_target > 20.

  def test_lingering_override_does_not_keep_a_pending_driver_setpoint(self):
    controller = self._adaptive_controller(speed=10., predicted_lateral=1.5, current_lateral=.5)
    self.sm['carState'].gasPressed = True
    controller.update(self.sm, True, True, 12., 0., 30.)
    self.sm['carState'].gasPressed = False
    for _ in range(6):
      controller.update(self.sm, True, True, 12., 0., 30.)
    assert controller.driver_speed is None
    assert not controller.is_active

  def test_current_curve_limits_recovery_even_if_the_prediction_is_straight(self):
    controller = self._adaptive_controller(speed=20., predicted_lateral=1.5, current_lateral=1.8)
    self._set_curve(20., 0., 1.8)
    for _ in range(20):
      controller.update(self.sm, True, False, 20., 0., 30.)
      assert controller.is_active
      assert abs(controller.output_v_target - 20. * (2. / 1.8) ** .5) < 1e-5

  def test_lateral_saturation_blocks_recovery_without_relaxing_sharp_curve_braking(self):
    controller = self._adaptive_controller(speed=10., predicted_lateral=1.5, current_lateral=.5)
    lateral = self.sm['controlsState'].lateralControlState.init('pidState')
    lateral.active = lateral.saturated = True
    self._set_curve(10., .8, .5)
    for _ in range(20):
      controller.update(self.sm, True, False, 10., 0., 30.)
      assert controller.output_v_target == 10.
      assert controller.output_a_target == 0.
    self._set_curve(10., 4., .5)
    controller.update(self.sm, True, False, 10., 0., 30.)
    assert controller.output_v_target < 10.
    assert controller.output_a_target < 0.
    lateral.saturated = False
    self._set_curve(10., 1.5, .5)
    controller.update(self.sm, True, False, 10., 0., 30.)
    assert 0. < controller.output_a_target <= .5

  def test_current_lateral_limit_alone_can_activate_curve_braking(self):
    controller = self._adaptive_controller(speed=20., predicted_lateral=.8, current_lateral=3.)
    assert controller.is_active
    assert controller.output_v_target < 20.
    assert controller.output_a_target < 0.

  @parameterized.expand([(0.001,), (0.01,), (0.1,)])
  def test_small_geometric_overshoot_does_not_trigger_full_entering_deceleration(self, overshoot):
    speed = 10.
    predicted = 2. * (speed / (speed - overshoot)) ** 2
    controller = self._adaptive_controller(speed=speed, predicted_lateral=predicted, current_lateral=.5)
    assert -.2 < controller.output_a_target < 0.

  @parameterized.expand([(27.71 / 3.6, 1.140, .100, 36.70 / 3.6),
                         (20. / 3.6, 1.28, .3, 25. / 3.6)])
  def test_marked_low_speed_cycles_recover_to_current_geometry(self, speed, predicted, current, expected):
    controller = self._adaptive_controller(speed=8., predicted_lateral=1.5, current_lateral=.5)
    self._set_curve(speed, predicted, current)
    controller.update(self.sm, True, False, speed, 0., 30.)
    assert abs(controller.output_v_target - expected) < .005
    assert controller.output_v_target > speed
    assert 0. < controller.output_a_target <= .5

  def test_sharp_low_speed_curve_is_not_raised_to_the_activation_speed_floor(self):
    controller = self._adaptive_controller(speed=6., predicted_lateral=4., current_lateral=2.)
    assert controller.output_v_target < MIN_V
    assert controller.output_a_target < 0.

  @parameterized.expand([(float('nan'),), (float('inf'),), (-float('inf'),)])
  def test_active_curve_hold_does_not_hide_invalid_predicted_lateral_acceleration(self, invalid):
    controller = self._adaptive_controller(speed=10., predicted_lateral=1.5, current_lateral=.5)
    self.sm['modelV2'].orientationRate.z = [invalid] * len(ModelConstants.T_IDXS)
    with np.errstate(invalid='ignore'):
      controller.update(self.sm, True, False, 10., 0., 30.)
    assert not np.isfinite(controller.output_v_target)
    assert not np.isfinite(controller.output_a_target)

  @parameterized.expand([(float('nan'),), (float('inf'),)])
  def test_active_curve_hold_does_not_hide_invalid_current_curvature(self, invalid):
    controller = self._adaptive_controller(speed=10., predicted_lateral=1.5, current_lateral=.5)
    self.sm['controlsState'].curvature = invalid
    controller.update(self.sm, True, False, 10., 0., 30.)
    assert not np.isfinite(controller.output_v_target)
    assert not np.isfinite(controller.output_a_target)

  def test_active_curve_hold_rejects_negative_prediction_before_square_root(self):
    controller = self._adaptive_controller(speed=10., predicted_lateral=1.5, current_lateral=2.)
    self.sm['modelV2'].velocity.x = [-1.] * len(ModelConstants.T_IDXS)
    controller.update(self.sm, True, False, 10., 0., 30.)
    assert not np.isfinite(controller.output_v_target)
    assert not np.isfinite(controller.output_a_target)

  def test_system_disabled(self):
    self.params.put_bool("SmartCruiseControlVision", False, block=True)
    self.scc_v.enabled = self.params.get_bool("SmartCruiseControlVision")

    for _ in range(int(10. / DT_MDL)):
      self.scc_v.update(self.sm, True, False, 0., 0., 0.)
    assert self.scc_v.state == VisionState.disabled
    assert not self.scc_v.is_active

  def test_disabled(self):
    for _ in range(int(10. / DT_MDL)):
      self.scc_v.update(self.sm, False, False, 0., 0., 0.)
    assert self.scc_v.state == VisionState.disabled

  def test_transition_disabled_to_enabled(self):
    for _ in range(int(10. / DT_MDL)):
      self.scc_v.update(self.sm, True, False, 0., 0., 0.)
    assert self.scc_v.state == VisionState.enabled

  @parameterized.expand([
    ("disabled", VisionState.disabled, 0., VisionState.enabled),
    ("enabled", VisionState.enabled, 0., VisionState.enabled),
    ("entering_abort", VisionState.entering, 0., VisionState.enabled),
    ("entering_turn", VisionState.entering, 0.005, VisionState.turning),
    ("turning", VisionState.turning, 0.005, VisionState.turning),
    ("turning_exit", VisionState.turning, 0., VisionState.leaving),
    ("leaving", VisionState.leaving, 0.0035, VisionState.leaving),
    ("leaving_exit", VisionState.leaving, 0., VisionState.enabled),
    ("override_exit", VisionState.overriding, 0., VisionState.enabled),
  ])
  def test_zero_prediction_preserves_turn_transitions_without_dividing(self, _, initial_state, curvature, expected_state):
    self.sm['modelV2'].orientationRate.z = [0.] * len(ModelConstants.T_IDXS)
    self.sm['controlsState'].curvature = curvature
    self.scc_v.state = initial_state

    with np.errstate(divide='raise', invalid='raise'):
      self.scc_v.update(self.sm, True, False, 20., 0.2, 30.)

    assert self.scc_v.max_pred_lat_acc == 0.
    assert self.scc_v.v_target == float('inf')
    assert self.scc_v.state == expected_state
    active = expected_state in (VisionState.turning, VisionState.leaving)
    assert self.scc_v.is_active == active
    assert self.scc_v.output_v_target == (float('inf') if active else V_CRUISE_UNSET)
    assert np.isfinite(self.scc_v.output_a_target)
    if not active:
      assert self.scc_v.output_a_target == 0.2

  @parameterized.expand([(False, 0.), (True, 0.), (False, 20.), (True, 20.)])
  def test_zero_model_velocity_is_safe_with_feature_on_or_off(self, enabled, speed):
    self.sm['modelV2'].velocity.x = [0.] * len(ModelConstants.T_IDXS)
    self.sm['controlsState'].curvature = 0.
    self.params.put_bool("SmartCruiseControlVision", enabled, block=True)
    self.scc_v.enabled = enabled

    with np.errstate(divide='raise', invalid='raise'):
      for _ in range(2):
        self.scc_v.update(self.sm, True, False, speed, 0., 30.)

    assert self.scc_v.v_target == float('inf')
    assert self.scc_v.state == (VisionState.enabled if enabled else VisionState.disabled)
    assert not self.scc_v.is_active
    assert self.scc_v.output_v_target == V_CRUISE_UNSET
    assert self.scc_v.output_a_target == 0.

  @parameterized.expand([
      ("p97_just_above_threshold", True),
      ("single_spike_filtered", False),
      ("persistent_high_values", True),
    ], names=["case", "should_enter"])
  def test_max_pred_lat_acc_uses_p97_and_threshold(self, case, should_enter):
    n = len(ModelConstants.T_IDXS)
    th = float(_ENTERING_PRED_LAT_ACC_TH)

    if case == "p97_just_above_threshold":
      # Use the next representable float32 above threshold to avoid float32 rounding flakiness.
      val = _th_above_f32(th)
      pred_lat_accels = np.full(n, np.float32(val), dtype=np.float32)

    elif case == "single_spike_filtered":
      pred_lat_accels = _build_single_spike_filtered(n, base=1.0)

    elif case == "persistent_high_values":
      # Make enough "high" samples so p97 is driven by the persistent trend, not a single outlier.
      high_count = max(2, int(np.ceil(n * 0.03)) + 1)
      pred_lat_accels = np.full(n, np.float32(1.0), dtype=np.float32)
      pred_lat_accels[-high_count:] = np.float32(2.0)
      pred_lat_accels[-1] = np.float32(8.0)  # keep one big outlier too

    else:
      raise AssertionError(f"Unknown case: {case}")

    # Override model predictions so:
    # predicted_lat_accels = abs(orientationRate.z) * velocity.x == pred_lat_accels
    mdl = generate_modelV2()
    mdl.modelV2.velocity.x = [1.0 for _ in range(n)]
    mdl.modelV2.orientationRate.z = [float(x) for x in pred_lat_accels]
    self.sm["modelV2"] = mdl.modelV2

    v_ego = float(MIN_V + 5.0)

    # 1st update: disabled -> enabled
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)
    # 2nd update: evaluate entering condition from enabled state
    self.scc_v.update(self.sm, True, False, v_ego, 0.0, 0.0)

    # Controller does percentile on numpy float64 arrays (values already quantized by capnp),
    # so compute expected in float64 to match behavior and avoid interpolation/rounding deltas.
    expected_p97 = float(np.percentile(pred_lat_accels.astype(np.float64), 97))

    # allow tiny numeric differences due to float conversions/interpolation
    assert np.isclose(self.scc_v.max_pred_lat_acc, expected_p97, rtol=1e-6, atol=1e-5)

    if should_enter:
      # We assert entering primarily by state (this is the actual intended behavior).
      assert self.scc_v.state == VisionState.entering
      # Optional sanity: should be >= threshold with some margin (since we used nextafter above threshold).
      assert self.scc_v.max_pred_lat_acc > th

    else:
      # Difference vs np.amax(): max can be above threshold, but p97 stays below it.
      assert float(np.max(pred_lat_accels)) >= th
      assert self.scc_v.max_pred_lat_acc < th
      assert self.scc_v.state == VisionState.enabled

  # TODO-SP: mock modelV2 data to test other states
