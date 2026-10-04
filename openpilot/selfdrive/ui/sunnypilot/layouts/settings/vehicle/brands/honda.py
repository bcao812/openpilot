"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.vehicle.brands.base import BrandSettings
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp


class HondaSettings(BrandSettings):
  def __init__(self):
    super().__init__()
    self.observer = toggle_item_sp(
      lambda: tr('Accord Radar Recording'),
      description=lambda: tr('Record radar targets with Alpha Longitudinal, Experimental Mode and DEC enabled. Observation only.'),
      initial_state=ui_state.params.get('HondaBoschARadarMode', return_default=True) == 1,
      callback=self._set_observer, enabled=lambda: not ui_state.started,
    )
    self.overlay = toggle_item_sp(
      lambda: tr('Radar Target Labels'),
      description=lambda: tr('Show slot number and longitudinal distance in meters, for example: 3 24.7.'),
      initial_state=ui_state.params.get_bool('HondaBoschARadarOverlay'),
      callback=self._set_overlay, enabled=lambda: not ui_state.started,
    )
    self.items = [self.observer, self.overlay]

  def _set_observer(self, enabled):
    if not ui_state.started:
      ui_state.params.put('HondaBoschARadarMode', 1 if enabled else 0, block=True)
    self.update_settings()

  def _set_overlay(self, enabled):
    if not ui_state.started:
      ui_state.params.put_bool('HondaBoschARadarOverlay', enabled, block=True)
    self.update_settings()

  def update_settings(self):
    self.observer.action_item.set_state(ui_state.params.get('HondaBoschARadarMode', return_default=True) == 1)
    self.overlay.action_item.set_state(ui_state.params.get_bool('HondaBoschARadarOverlay'))
