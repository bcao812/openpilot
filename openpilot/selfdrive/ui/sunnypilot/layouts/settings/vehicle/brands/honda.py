"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.vehicle.brands.base import BrandSettings
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.car.honda_radar_observer_config import OBSERVE, SHADOW
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp


class HondaSettings(BrandSettings):
  def __init__(self):
    super().__init__()
    self.observer = toggle_item_sp(
      lambda: tr('Accord Radar Recording'),
      description=lambda: tr('Record radar targets with Alpha Longitudinal, Experimental Mode and DEC enabled. Observation only.'),
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
      description=lambda: tr('Calculate and record radar and vision fusion. Does not control the vehicle.'),
      initial_state=ui_state.params.get('HondaBoschARadarMode', return_default=True) == SHADOW,
      callback=self._set_shadow, enabled=lambda: not ui_state.started and self._observing(),
    )
    self.items = [self.observer, self.shadow, self.overlay]

  @staticmethod
  def _observing():
    return ui_state.params.get('HondaBoschARadarMode', return_default=True) in (OBSERVE, SHADOW)

  def _set_observer(self, enabled):
    if not ui_state.started:
      current = ui_state.params.get('HondaBoschARadarMode', return_default=True)
      mode = SHADOW if enabled and current == SHADOW else OBSERVE if enabled else 0
      ui_state.params.put('HondaBoschARadarMode', mode, block=True)
    self.update_settings()

  def _set_overlay(self, enabled):
    if not ui_state.started:
      ui_state.params.put_bool('HondaBoschARadarOverlay', enabled, block=True)
    self.update_settings()

  def _set_shadow(self, enabled):
    if not ui_state.started and self._observing():
      ui_state.params.put('HondaBoschARadarMode', SHADOW if enabled else OBSERVE, block=True)
    self.update_settings()

  def update_settings(self):
    self.observer.action_item.set_state(self._observing())
    self.shadow.action_item.set_state(ui_state.params.get('HondaBoschARadarMode', return_default=True) == SHADOW)
    self.overlay.action_item.set_state(ui_state.params.get_bool('HondaBoschARadarOverlay'))
