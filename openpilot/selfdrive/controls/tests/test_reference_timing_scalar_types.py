"""Timing guards keep wire bounds and non-wire numeric compatibility."""
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.reference_timing import finite_state, valid_timestamp


class IntegerSubclass(int):
  pass


class FloatSubclass(float):
  pass


@pytest.mark.parametrize('value', [1, (1 << 64) - 1, IntegerSubclass(3), np.int64(3), np.uint64((1 << 64) - 1)])
def test_monotonic_integer_timestamp_preserves_uint64_bounds(value):
  assert valid_timestamp(value)


@pytest.mark.parametrize('value', [0, -1, 1 << 64, -(1 << 100), True, False, 1.0, Fraction(1), Decimal(1),
                                 None, '1', float('nan'), float('inf'), np.float64(1), np.bool_(True),
                                 IntegerSubclass(0), IntegerSubclass(1 << 64), np.int64(-1), np.uint64(0)])
def test_invalid_or_noninteger_time_never_gains_a_lease(value):
  assert not valid_timestamp(value)


@pytest.mark.parametrize('value', [0, -2, 1.25, -0.0, IntegerSubclass(2), FloatSubclass(1.5),
                                 Fraction(1, 3), np.int64(2), np.uint64(3), np.float32(1.25), np.float64(-2.0)])
def test_finite_state_accepts_supported_real_scalars(value):
  assert finite_state((20.0, value, -0.5))


@pytest.mark.parametrize('value', [True, False, None, '2.0', 1 + 0j, Decimal('2.0'), float('nan'), float('inf'),
                                 float('-inf'), FloatSubclass('inf'), np.float32('nan'), np.bool_(False)])
def test_invalid_scalar_rejects_the_state(value):
  assert not finite_state((20.0, value, -0.5))


def test_finite_state_preserves_short_circuit_and_original_overflow_behavior():
  consumed = []

  def states():
    for value in (1.0, float('nan'), 10 ** 1000):
      consumed.append(value)
      yield value

  assert not finite_state(states())
  assert len(consumed) == 2
  with pytest.raises(OverflowError):
    finite_state((10 ** 1000,))
  assert finite_state(())
