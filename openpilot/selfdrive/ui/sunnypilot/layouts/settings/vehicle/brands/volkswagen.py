"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.vehicle.brands.base import BrandSettings
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp


DESCRIPTIONS = {
  'avoid_eps_lockout': tr_noop(
    'Scale steering torque down at low speeds to avoid EPS lockout.'
  )
}


class VolkswagenSettings(BrandSettings):
  def __init__(self):
    super().__init__()

    self.avoid_eps_lockout = toggle_item_sp(
      lambda: tr("Avoid EPS Lockout"),
      description=lambda: tr(DESCRIPTIONS["avoid_eps_lockout"]),
      initial_state=ui_state.params.get_bool("VagAvoidEPSLockout"),
      callback=self._on_enable_avoid_eps_lockout,
      enabled=lambda: not ui_state.engaged,
    )

    self.items = [
      self.avoid_eps_lockout,
    ]

  def _on_enable_avoid_eps_lockout(self, state: bool):
    ui_state.params.put_bool("VagAvoidEPSLockout", state, block=True)
    ui_state.params.put_bool("OnroadCycleRequested", True)

  def update_settings(self):
    pass
