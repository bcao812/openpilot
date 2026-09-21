"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openpilot.cereal import custom
from openpilot.common.constants import CV
from openpilot.common.parameterized import parameterized
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.common import OffsetType
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_resolver import SpeedLimitResolver

SpeedLimitSource = custom.LongitudinalPlanSP.SpeedLimit.Source


def car_state(speed_limit):
  # Deliberately omit map and GPS subscriptions: the car source must stand alone.
  return {'carStateSP': SimpleNamespace(speedLimit=speed_limit)}


class TestSpeedLimitResolver(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()
    self.params.put_bool("IsMetric", True, block=True)
    self.params.put("SpeedLimitOffsetType", int(OffsetType.off), block=True)
    self.params.put("SpeedLimitValueOffset", 0, block=True)

  def test_initial_state(self):
    resolver = SpeedLimitResolver()
    assert not resolver.speed_limit_valid
    assert not resolver.speed_limit_last_valid
    assert resolver.speed_limit_final == 0.
    assert resolver.speed_limit_final_last == 0.
    assert resolver.source == SpeedLimitSource.none

  @parameterized.expand([(22.2, 22.2), (0., 0.), (-1., 0.)])
  def test_only_car_state_needed(self, speed_limit, expected):
    resolver = SpeedLimitResolver()
    resolver.update(car_state(speed_limit))
    assert resolver.speed_limit == expected
    assert resolver.speed_limit_final == expected
    assert resolver.speed_limit_valid == (expected > 0.)
    assert resolver.source == (SpeedLimitSource.car if expected > 0. else SpeedLimitSource.none)

  @parameterized.expand([0, 1, 2, 3, 4, 99])
  def test_retired_policy_does_not_override_car_limit(self, legacy_policy):
    # Simulate an old on-device value without registering or writing the retired key.
    params = Mock(wraps=self.params)
    params.get.side_effect = lambda key, **kwargs: legacy_policy if key == "SpeedLimitPolicy" else self.params.get(key, **kwargs)
    with patch('openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_resolver.Params', return_value=params):
      resolver = SpeedLimitResolver()
      for _ in range(int(PARAMS_UPDATE_PERIOD / DT_MDL) + 2):
        resolver.update(car_state(25.))
        assert resolver.speed_limit == 25.
        assert resolver.source == SpeedLimitSource.car

    assert all(call.args[0] != "SpeedLimitPolicy" for call in params.get.call_args_list)

  @parameterized.expand([0., 25.])
  def test_map_message_cannot_supply_or_lower_limit(self, car_limit):
    resolver = SpeedLimitResolver()
    sm = car_state(car_limit)
    sm['liveMapDataSP'] = SimpleNamespace(speedLimit=10., speedLimitValid=True, speedLimitAhead=5.,
                                        speedLimitAheadValid=True, speedLimitAheadDistance=1.)
    resolver.update(sm)
    assert resolver.speed_limit == car_limit
    assert resolver.source == (SpeedLimitSource.car if car_limit > 0. else SpeedLimitSource.none)

  @parameterized.expand([
    (True, OffsetType.off, 5, 0.),
    (True, OffsetType.fixed, 5, 5 * CV.KPH_TO_MS),
    (False, OffsetType.fixed, 5, 5 * CV.MPH_TO_MS),
    (True, OffsetType.fixed, -5, -5 * CV.KPH_TO_MS),
    (True, OffsetType.percentage, 10, 2.),
    (False, OffsetType.percentage, -10, -2.),
  ])
  def test_offsets_preserved(self, is_metric, offset_type, offset_value, expected):
    self.params.put_bool("IsMetric", is_metric, block=True)
    self.params.put("SpeedLimitOffsetType", int(offset_type), block=True)
    self.params.put("SpeedLimitValueOffset", offset_value, block=True)
    resolver = SpeedLimitResolver()
    resolver.update(car_state(20.))
    self.assertAlmostEqual(resolver.speed_limit_offset, expected)
    self.assertAlmostEqual(resolver.speed_limit_final, 20. + expected)
    self.assertAlmostEqual(resolver.speed_limit_final_last, 20. + expected)

  def test_last_car_limit_survives_temporary_loss(self):
    self.params.put("SpeedLimitOffsetType", int(OffsetType.percentage), block=True)
    self.params.put("SpeedLimitValueOffset", 10, block=True)
    resolver = SpeedLimitResolver()
    resolver.update(car_state(20.))
    resolver.update(car_state(0.))
    assert not resolver.speed_limit_valid
    assert resolver.source == SpeedLimitSource.none
    assert resolver.speed_limit_last_valid
    assert resolver.speed_limit_last == 20.
    assert resolver.speed_limit_final_last == 22.

    resolver.update(car_state(30.))
    assert resolver.speed_limit_last == 30.
    assert resolver.speed_limit_final_last == 33.
    assert resolver.source == SpeedLimitSource.car

  def test_offset_settings_update_while_running(self):
    resolver = SpeedLimitResolver()
    resolver.update(car_state(20.))
    self.params.put_bool("IsMetric", False, block=True)
    self.params.put("SpeedLimitOffsetType", int(OffsetType.fixed), block=True)
    self.params.put("SpeedLimitValueOffset", 5, block=True)
    for _ in range(int(PARAMS_UPDATE_PERIOD / DT_MDL) + 1):
      resolver.update(car_state(20.))
    self.assertAlmostEqual(resolver.speed_limit_final, 20. + 5 * CV.MPH_TO_MS)
