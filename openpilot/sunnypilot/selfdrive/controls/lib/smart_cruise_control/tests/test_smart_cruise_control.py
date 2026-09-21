"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

from openpilot.cereal import custom
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.smart_cruise_control import SmartCruiseControl

VisionState = custom.LongitudinalPlanSP.SmartCruiseControl.VisionState


class TestSmartCruiseControl(OpenpilotTestCase):
  def test_vision_curve_control_without_map_or_gps(self):
    Params().put_bool("SmartCruiseControlVision", True, block=True)
    scc = SmartCruiseControl()
    sm = {
      'modelV2': SimpleNamespace(orientationRate=SimpleNamespace(z=[0.15] * 33), velocity=SimpleNamespace(x=[20.] * 33)),
      'controlsState': SimpleNamespace(curvature=0.001),
    }

    scc.update(sm, True, False, 20., 0., 30.)
    assert scc.vision.state == VisionState.enabled
    assert scc.vision.output_v_target == V_CRUISE_UNSET

    scc.update(sm, True, False, 20., 0., 30.)
    assert scc.vision.state == VisionState.entering
    assert scc.vision.is_active
    assert 0. < scc.vision.output_v_target < 20.
    assert scc.vision.output_a_target < 0.

    scc.update(sm, True, True, 20., 0.4, 30.)
    assert scc.vision.state == VisionState.overriding
    assert not scc.vision.is_active
    assert scc.vision.output_v_target == V_CRUISE_UNSET
    assert scc.vision.output_a_target == 0.4

    scc.update(sm, False, False, 20., 0., 30.)
    assert scc.vision.state == VisionState.disabled
    assert not scc.vision.is_enabled
    assert scc.vision.output_v_target == V_CRUISE_UNSET
