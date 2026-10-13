"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import openpilot.cereal.messaging as messaging
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.vision_controller import SmartCruiseControlVision


class SmartCruiseControl:
  def __init__(self, *, adaptive_curve_speed: bool = False):
    self.vision = SmartCruiseControlVision(adaptive_curve_speed=adaptive_curve_speed)

  def update(self, sm: messaging.SubMaster, long_enabled: bool, long_override: bool, v_ego: float, a_ego: float, v_cruise: float) -> None:
    self.vision.update(sm, long_enabled, long_override, v_ego, a_ego, v_cruise)
