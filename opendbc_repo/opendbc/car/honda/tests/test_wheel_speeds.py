"""Actual Honda CAN wheels stay distinct from the low-speed transmission blend."""
import pytest

from opendbc.car import CanData
from opendbc.car.honda.tests.test_honda import AccFaultHarness
from opendbc.car.honda.values import CAR


@pytest.mark.parametrize('platform', [CAR.HONDA_ACCORD, CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_2022])
@pytest.mark.parametrize('wheels', [(0.0, 0.0, 0.0, 0.0), (0.36, 0.72, 1.08, 1.44)])
def test_raw_can_wheels_are_explicitly_published_even_when_transmission_reports_zero(platform, wheels):
  harness = AccFaultHarness(platform, alpha_long=platform == CAR.HONDA_ACCORD)
  for frame in range(4):
    messages = [
      ('WHEEL_SPEEDS', {f'WHEEL_SPEED_{name}': value for name, value in zip(('FL', 'FR', 'RL', 'RR'), wheels, strict=True)}),
      ('ENGINE_DATA', {'XMISSION_SPEED': 0.0}),
    ]
    packed = [harness.packer.make_can_msg(name, harness.bus, dict(values, COUNTER=frame % 4)) for name, values in messages]
    current, _ = harness.car_interface.update([(1_000_000_000 + frame * 10_000_000,
                                               [CanData(row[0], row[1], harness.bus) for row in packed])])
  assert current._has('wheelSpeeds')
  actual = tuple(getattr(current.wheelSpeeds, name) for name in ('fl', 'fr', 'rl', 'rr'))
  assert actual == pytest.approx(tuple(value / 3.6 for value in wheels), abs=0.003)
  assert current.vEgoRaw == 0.0 and current.standstill
