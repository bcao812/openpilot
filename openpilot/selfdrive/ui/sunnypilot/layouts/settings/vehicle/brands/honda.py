"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.vehicle.brands.base import BrandSettings
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.car.honda_radar_observer_config import ASSIST, CONTROL, FUSION_MODES, OBSERVE, SHADOW
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp


class HondaSettings(BrandSettings):
  def __init__(self):
    super().__init__()
    self.observer = toggle_item_sp(
      lambda: tr('Accord Radar Recording'),
      description=lambda: tr('Record radar targets with Alpha Longitudinal, Experimental Mode and DEC enabled. '
                             'Recording alone does not control the vehicle. Turning this off also disables fusion and acceleration assist.'),
      initial_state=self._observing(),
      callback=self._set_observer, enabled=lambda: not ui_state.started,
    )
    self.overlay = toggle_item_sp(
      lambda: tr('Radar Target Labels'),
      description=lambda: tr('Show radar detections as yellow dots with longitudinal distance in meters to the right. '
                             'Radar-camera alignment is uncalibrated.'),
      initial_state=ui_state.params.get_bool('HondaBoschARadarOverlay'),
      callback=self._set_overlay, enabled=lambda: not ui_state.started,
    )
    self.shadow = toggle_item_sp(
      lambda: tr('Radar Fusion Shadow'),
      description=lambda: tr('Calculate and record radar and vision fusion. Shadow mode alone does not control the vehicle. '
                             'Required and locked on while radar acceleration assist is enabled.'),
      initial_state=self._mode() in FUSION_MODES,
      callback=self._set_shadow, enabled=lambda: not ui_state.started and self._mode() in (OBSERVE, SHADOW),
    )
    self.assist = toggle_item_sp(
      lambda: tr('Radar Acceleration Assist'),
      description=lambda: tr('Limit positive acceleration briefly when a previously confirmed visual lead is lost. '
                             'Radar does not replace the visual lead or independently request braking or starting from rest. '
                             'Requires UNIFIED, Alpha Longitudinal, Experimental Mode and DEC. Turning this off returns to shadow mode.'),
      initial_state=self._mode() == ASSIST,
      callback=self._set_assist, enabled=lambda: not ui_state.started and self._observing() and self._mode() != CONTROL,
    )
    self.control = toggle_item_sp(
      lambda: tr('Radar Fusion Control'),
      description=lambda: tr('Use radar and vision together for following, braking and stop-and-go. '
                             'Qualified radar targets can constrain driving even without a visual lead. '
                             'Requires Alpha Longitudinal, Experimental Mode and DEC.'),
      initial_state=self._mode() == CONTROL,
      callback=self._set_control, enabled=lambda: not ui_state.started and self._observing(),
    )
    self.items = [self.observer, self.shadow, self.assist, self.control, self.overlay]

  @staticmethod
  def _mode():
    return ui_state.params.get('HondaBoschARadarMode', return_default=True)

  def _observing(self):
    return self._mode() in (OBSERVE, *FUSION_MODES)

  def _set_observer(self, enabled):
    if not ui_state.started:
      current = self._mode()
      mode = current if enabled and current in (OBSERVE, *FUSION_MODES) else OBSERVE if enabled else 0
      ui_state.params.put('HondaBoschARadarMode', mode, block=True)
    self.update_settings()

  def _set_overlay(self, enabled):
    if not ui_state.started:
      ui_state.params.put_bool('HondaBoschARadarOverlay', enabled, block=True)
    self.update_settings()

  def _set_shadow(self, enabled):
    if not ui_state.started and self._mode() in (OBSERVE, SHADOW):
      ui_state.params.put('HondaBoschARadarMode', SHADOW if enabled else OBSERVE, block=True)
    self.update_settings()

  def _set_assist(self, enabled):
    current = self._mode()
    if not ui_state.started and current in (OBSERVE, SHADOW, ASSIST):
      mode = ASSIST if enabled else SHADOW if current == ASSIST else current
      ui_state.params.put('HondaBoschARadarMode', mode, block=True)
    self.update_settings()

  def _set_control(self, enabled):
    current = self._mode()
    if not ui_state.started and current in (OBSERVE, *FUSION_MODES):
      mode = CONTROL if enabled else SHADOW if current == CONTROL else current
      ui_state.params.put('HondaBoschARadarMode', mode, block=True)
    self.update_settings()

  def update_settings(self):
    mode = self._mode()
    self.observer.action_item.set_state(mode in (OBSERVE, *FUSION_MODES))
    self.shadow.action_item.set_state(mode in FUSION_MODES)
    self.assist.action_item.set_state(mode == ASSIST)
    self.control.action_item.set_state(mode == CONTROL)
    self.overlay.action_item.set_state(ui_state.params.get_bool('HondaBoschARadarOverlay'))
