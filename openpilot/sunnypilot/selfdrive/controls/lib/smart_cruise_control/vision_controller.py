"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np

import openpilot.cereal.messaging as messaging
from openpilot.cereal import custom
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import MIN_V

VisionState = custom.LongitudinalPlanSP.SmartCruiseControl.VisionState

ACTIVE_STATES = (VisionState.entering, VisionState.turning, VisionState.leaving)
ENABLED_STATES = (VisionState.enabled, VisionState.overriding, *ACTIVE_STATES)

_ENTERING_PRED_LAT_ACC_TH = 1.3  # Predicted Lat Acc threshold to trigger entering turn state.
_ABORT_ENTERING_PRED_LAT_ACC_TH = 1.1  # Predicted Lat Acc threshold to abort entering state if speed drops.

_TURNING_LAT_ACC_TH = 1.6  # Lat Acc threshold to trigger turning state.

_LEAVING_LAT_ACC_TH = 1.3  # Lat Acc threshold to trigger leaving turn state.
_FINISH_LAT_ACC_TH = 1.1  # Lat Acc threshold to trigger the end of the turn cycle.

_A_LAT_REG_MAX = 2.  # Maximum lateral acceleration

_NO_OVERSHOOT_TIME_HORIZON = 4.  # s. Time to use for velocity desired based on a_target when not overshooting.

# Lookup table for the minimum smooth deceleration during the ENTERING state
# depending on the actual maximum absolute lateral acceleration predicted on the turn ahead.
_ENTERING_SMOOTH_DECEL_V = [-0.2, -1.]  # min decel value allowed on ENTERING state
_ENTERING_SMOOTH_DECEL_BP = [1.3, 3.]  # absolute value of lat acc ahead

# Lookup table for the acceleration for the TURNING state
# depending on the current lateral acceleration of the vehicle.
_TURNING_ACC_V = [0.5, 0., -0.4]  # acc value
_TURNING_ACC_BP = [1.5, 2.3, 3.]  # absolute value of current lat acc

_LEAVING_ACC = 0.5  # Conformable acceleration to regain speed while leaving a turn.

_CURVE_CLEAR_TIME = 0.5  # Both current and predicted curvature must clear before a new cycle.
_ENTERING_DECEL_FULL_ERROR = 1.0  # m/s. Fade the entering deceleration to zero near the target.
_OVERRIDE_RELEASE_TIME = 0.25  # Allow the carControl override bit to follow pedal release.


class SmartCruiseControlVision:
  v_target: float = 0
  a_target: float = 0.
  v_ego: float = 0.
  a_ego: float = 0.
  output_v_target: float = V_CRUISE_UNSET
  output_a_target: float = 0.

  def __init__(self, *, adaptive_curve_speed: bool = False):
    self.params = Params()
    self.frame = -1
    self.long_enabled = False
    self.long_override = False
    self.is_enabled = False
    self.is_active = False
    self.enabled = self.params.get_bool("SmartCruiseControlVision")
    self.v_cruise_setpoint = 0.
    self.adaptive_curve_speed = adaptive_curve_speed
    self.driver_speed = None
    self.steering_limited = False
    self._curve_state = VisionState.disabled
    self._curve_clear_time = 0.
    self._gas_pressed = False
    self._pedal_override = False
    self._override_release_time = 0.
    self._model_mono_time = None

    self.state = VisionState.disabled
    self.current_lat_acc = 0.
    self.max_pred_lat_acc = 0.

  def get_a_target_from_control(self) -> float:
    return self.a_target

  def get_v_target_from_control(self, *, include_cruise: bool = True) -> float:
    if self.is_active:
      if self.adaptive_curve_speed:
        if (not np.isfinite(self.v_ego) or self.v_ego < 0.
            or not np.isfinite(self.v_cruise_setpoint) or self.v_cruise_setpoint < 0.
            or self.driver_speed is not None and not np.isfinite(self.driver_speed)
            or not np.isfinite(self.current_lat_acc) or self.current_lat_acc < 0.
            or not np.isfinite(self.max_pred_lat_acc) or self.max_pred_lat_acc < 0.
            or not (np.isfinite(self.v_target) or self.v_target == float('inf') and self.max_pred_lat_acc == 0.)):
          return float('nan')
        # Follow this frame's geometry, including the curve the car is still
        # in. The planner enforces our acceleration ceiling while recovering;
        # neither a low entry speed nor a second horizon deduction limits it.
        target = self.v_target
        if self.current_lat_acc > 0.:
          target = min(target, self.v_ego * (_A_LAT_REG_MAX / self.current_lat_acc) ** 0.5)
        if self.driver_speed is not None:
          target = min(target, self.driver_speed)
        if self.steering_limited:
          target = min(target, self.v_ego)
        if include_cruise:
          target = min(target, self.v_cruise_setpoint)
        return max(0., float(target))
      return max(self.v_target, MIN_V) + self.a_target * _NO_OVERSHOOT_TIME_HORIZON

    return V_CRUISE_UNSET

  def _update_params(self) -> None:
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.enabled = self.params.get_bool("SmartCruiseControlVision")

  def _update_calculations(self, sm: messaging.SubMaster) -> None:
    if not self.long_enabled:
      return
    else:
      rate_plan = np.array(np.abs(sm['modelV2'].orientationRate.z))
      vel_plan = np.array(sm['modelV2'].velocity.x)

      controls = sm['controlsState']
      self.current_lat_acc = self.v_ego ** 2 * abs(controls.curvature)
      if self.adaptive_curve_speed:
        lateral = controls.lateralControlState
        lateral = getattr(lateral, lateral.which())
        self.steering_limited = bool(lateral.active and lateral.saturated)
        if not len(rate_plan) or rate_plan.shape != vel_plan.shape:
          self.max_pred_lat_acc = self.v_target = float('nan')
          return

      # get the maximum lat accel from the model
      predicted_lat_accels = rate_plan * vel_plan
      self.max_pred_lat_acc = np.percentile(predicted_lat_accels, 97)
      if self.adaptive_curve_speed and (not np.isfinite(self.max_pred_lat_acc) or self.max_pred_lat_acc < 0.):
        self.v_target = float('nan')
        return

      # get the maximum curve based on the current velocity
      v_ego = max(self.v_ego, 0.1)  # ensure a value greater than 0 for calculations
      max_curve = self.max_pred_lat_acc / (v_ego**2)

      # Get the target velocity for the maximum curve
      # A straight prediction imposes no curve speed limit. Keep the unbounded
      # target explicit instead of relying on NumPy's divide-by-zero result.
      self.v_target = float("inf") if max_curve == 0. else (_A_LAT_REG_MAX / max_curve) ** 0.5

  def _clear_curve(self) -> None:
    self._curve_state = VisionState.disabled
    self._curve_clear_time = 0.
    self.driver_speed = None
    self._gas_pressed = False
    self._pedal_override = False
    self._override_release_time = 0.

  def _update_adaptive_state(self, sm: messaging.SubMaster) -> tuple[bool, bool]:
    # Keep the geometric cycle current during pedal override, but publish no
    # SCC authority until override ends. A non-pedal override clears it.
    dt = DT_MDL
    model_time = getattr(sm, 'logMonoTime', {}).get('modelV2')
    if model_time is not None:
      if self._model_mono_time is not None:
        elapsed = (model_time - self._model_mono_time) * 1e-9
        if elapsed < 0. or elapsed > _OVERRIDE_RELEASE_TIME:
          self._clear_curve()
        else:
          dt = elapsed
      self._model_mono_time = model_time

    gas_pressed = bool(sm['carState'].gasPressed)
    if not self.long_enabled or not self.enabled or sm['carState'].brakePressed:
      self._clear_curve()
      self.state = VisionState.disabled
      return False, False

    if gas_pressed:
      self._pedal_override = True
      self._override_release_time = 0.
    elif not self.long_override:
      self._pedal_override = False
      self._override_release_time = 0.
    elif self._pedal_override:
      self._override_release_time += dt
      if self._override_release_time > _OVERRIDE_RELEASE_TIME:
        self._pedal_override = False
    if self.long_override and not self._pedal_override:
      self._clear_curve()
      self.state = VisionState.overriding
      return True, False

    was_in_curve = self._curve_state in ACTIVE_STATES
    if self._curve_state == VisionState.disabled:
      self._curve_state = VisionState.enabled
    elif self._curve_state == VisionState.enabled:
      if self.v_ego > MIN_V and self.max_pred_lat_acc >= _ENTERING_PRED_LAT_ACC_TH:
        self._curve_state = VisionState.entering
    else:
      if self.current_lat_acc < _FINISH_LAT_ACC_TH and self.max_pred_lat_acc < _ABORT_ENTERING_PRED_LAT_ACC_TH:
        self._curve_clear_time += dt
      else:
        self._curve_clear_time = 0.
      if self._curve_clear_time + 1e-9 >= _CURVE_CLEAR_TIME:
        self._curve_state = VisionState.enabled
        self._curve_clear_time = 0.
      elif self.current_lat_acc >= _TURNING_LAT_ACC_TH:
        self._curve_state = VisionState.turning
      elif self._curve_state == VisionState.turning and self.current_lat_acc <= _LEAVING_LAT_ACC_TH:
        self._curve_state = VisionState.leaving

    if self._curve_state not in ACTIVE_STATES:
      self.driver_speed = None
    elif self._gas_pressed and not gas_pressed and was_in_curve:
      # Use release speed, not the maximum reached earlier while the pedal
      # was down. Current geometry, cruise and lateral capability still win.
      self.driver_speed = self.v_ego
    self._gas_pressed = gas_pressed
    self.state = VisionState.overriding if gas_pressed or self.long_override else self._curve_state
    return self.state in ENABLED_STATES, self.state in ACTIVE_STATES

  def _update_state_machine(self) -> tuple[bool, bool]:
    # ENABLED, ENTERING, TURNING, LEAVING, OVERRIDING
    if self.state != VisionState.disabled:
      # longitudinal and feature disable always have priority in a non-disabled state
      if not self.long_enabled or not self.enabled:
        self.state = VisionState.disabled
      elif self.long_override:
        self.state = VisionState.overriding

      else:
        # ENABLED
        if self.state == VisionState.enabled:
          # Do not enter a turn control cycle if the speed is low.
          if self.v_ego <= MIN_V:
            pass
          # If significant lateral acceleration is predicted ahead, then move to Entering turn state.
          elif self.max_pred_lat_acc >= _ENTERING_PRED_LAT_ACC_TH:
            self.state = VisionState.entering

        # OVERRIDING
        elif self.state == VisionState.overriding:
          if not self.long_override:
            self.state = VisionState.enabled

        # ENTERING
        elif self.state == VisionState.entering:
          # Transition to Turning if current lateral acceleration is over the threshold.
          if self.current_lat_acc >= _TURNING_LAT_ACC_TH:
            self.state = VisionState.turning
          # Abort if the predicted lateral acceleration drops
          elif self.max_pred_lat_acc < _ABORT_ENTERING_PRED_LAT_ACC_TH:
            self.state = VisionState.enabled

        # TURNING
        elif self.state == VisionState.turning:
          # Transition to Leaving if current lateral acceleration drops below a threshold.
          if self.current_lat_acc <= _LEAVING_LAT_ACC_TH:
            self.state = VisionState.leaving

        # LEAVING
        elif self.state == VisionState.leaving:
          # Transition back to Turning if current lateral acceleration goes back over the threshold.
          if self.current_lat_acc >= _TURNING_LAT_ACC_TH:
            self.state = VisionState.turning
          # Finish if current lateral acceleration goes below a threshold.
          elif self.current_lat_acc < _FINISH_LAT_ACC_TH:
            self.state = VisionState.enabled

    # DISABLED
    elif self.state == VisionState.disabled:
      if self.long_enabled and self.enabled:
        if self.long_override:
          self.state = VisionState.overriding
        else:
          self.state = VisionState.enabled

    enabled = self.state in ENABLED_STATES
    active = self.state in ACTIVE_STATES

    return enabled, active

  def _update_solution(self) -> float:
    # DISABLED, ENABLED, OVERRIDING
    if self.state not in ACTIVE_STATES:
      # when not overshooting, calculate v_turn as the speed at the prediction horizon when following
      # the smooth deceleration.
      a_target = self.a_ego
    # ENTERING
    elif self.state == VisionState.entering:
      # when not overshooting, target a smooth deceleration in preparation for a sharp turn to come.
      a_target = np.interp(self.max_pred_lat_acc, _ENTERING_SMOOTH_DECEL_BP, _ENTERING_SMOOTH_DECEL_V)
    # TURNING
    elif self.state == VisionState.turning:
      # When turning, we provide a target acceleration that is comfortable for the lateral acceleration felt.
      a_target = np.interp(self.current_lat_acc, _TURNING_ACC_BP, _TURNING_ACC_V)
    # LEAVING
    elif self.state == VisionState.leaving:
      # When leaving, we provide a comfortable acceleration to regain speed.
      a_target = _LEAVING_ACC
    else:
      raise NotImplementedError(f"SCC-V state not supported: {self.state}")

    if self.adaptive_curve_speed and self.is_active:
      # A lower cruise setting already has its own response. Do not mistake
      # that reduction for geometric overshoot and add entering-state braking.
      curve_target = self.get_v_target_from_control(include_cruise=False)
      speed_error = min(curve_target, max(self.v_ego, self.v_cruise_setpoint)) - self.v_ego
      if not np.isfinite(speed_error):
        return float('nan')
      if speed_error >= 0.0:
        # Recover toward current geometry (or this turn's driver setpoint)
        # with a bounded request. Independent lead/model braking still wins.
        a_target = min(_LEAVING_ACC, speed_error / _NO_OVERSHOOT_TIME_HORIZON)
      else:
        if self.state == VisionState.entering:
          a_target *= min(1., -speed_error / _ENTERING_DECEL_FULL_ERROR)
        # Strong curve braking remains immediate; a tiny overshoot no longer
        # triggers the full entering-state minimum deceleration.
        a_target = min(a_target, speed_error / _NO_OVERSHOOT_TIME_HORIZON)
    return a_target

  def update(self, sm: messaging.SubMaster, long_enabled: bool, long_override: bool, v_ego: float, a_ego: float,
             v_cruise_setpoint: float) -> None:
    self.long_enabled = long_enabled
    self.long_override = long_override
    self.v_ego = v_ego
    self.a_ego = a_ego
    self.v_cruise_setpoint = v_cruise_setpoint

    self._update_params()
    self._update_calculations(sm)

    self.is_enabled, self.is_active = self._update_adaptive_state(sm) if self.adaptive_curve_speed else self._update_state_machine()
    self.a_target = self._update_solution()

    self.output_v_target = self.get_v_target_from_control()
    self.output_a_target = self.get_a_target_from_control()

    self.frame += 1
