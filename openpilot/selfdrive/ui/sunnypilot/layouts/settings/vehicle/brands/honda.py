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
    self.overlay = toggle_item_sp(
      lambda: tr('Radar Target Labels'),
      description=lambda: tr('Show radar detections with longitudinal distance in meters to the right. '
                             'Radar-camera alignment is uncalibrated.'),
      initial_state=ui_state.params.get_bool('HondaBoschARadarOverlay'),
      callback=self._set_overlay, enabled=lambda: not ui_state.started,
    )
    self.items = [self.overlay]

  def _set_overlay(self, enabled):
    if not ui_state.started:
      ui_state.params.put_bool('HondaBoschARadarOverlay', enabled, block=True)
    self.update_settings()

  def update_settings(self):
    self.overlay.action_item.set_state(ui_state.params.get_bool('HondaBoschARadarOverlay'))
