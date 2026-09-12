import math
import struct
from types import SimpleNamespace
import unittest

import numpy as np

from openpilot.sunnypilot.models.helpers import plan_x_idxs_helper


class TestPlanXIndices(unittest.TestCase):
  def test_interpolation_and_boundary_values(self):
    cases = (
      ("exact", [0.0, 1.0, 4.0, 9.0], [0.0, 1.0, 2.0, 3.0]),
      ("interpolated", [0.0, 3.0, 6.0, 9.0], [0.0, 1.0 / 3.0, 4.0 / 3.0, 3.0]),
      ("repeated", [0.0, 1.0, 1.0, 9.0], [0.0, 1.0, 2.375, 3.0]),
      ("short", [0.0, 0.1, 0.2, 0.3], [0.0, 3.0, np.nan, np.nan]),
      ("stationary", [0.0, 0.0, 0.0, 0.0], [0.0, 3.0, np.nan, np.nan]),
      ("backward", [-4.0, -3.0, -2.0, -1.0], [0.0, 3.0, np.nan, np.nan]),
      ("nonfinite", [0.0, np.nan, np.nan, np.nan], [0.0, np.nan, np.nan, np.nan]),
    )
    constants = SimpleNamespace(IDX_N=4, X_IDXS=[0.0, 1.0, 4.0, 9.0], T_IDXS=[0.0, 1.0, 2.0, 3.0])
    for name, positions, expected in cases:
      with self.subTest(name=name):
        self.check_plan(constants, positions, expected)

  def test_near_zero_distance_keeps_nan(self):
    constants = SimpleNamespace(IDX_N=4, X_IDXS=[0.0, 5e-10, 1.0, 2.0], T_IDXS=[0.0, 1.0, 2.0, 3.0])
    self.check_plan(constants, [0.0, 5e-10, 1.0, 2.0], [0.0, np.nan, 2.0, 3.0])

  def test_single_point(self):
    constants = SimpleNamespace(IDX_N=1, X_IDXS=[0.0], T_IDXS=[0.0])
    self.check_plan(constants, [0.0], [0.0])

  def check_plan(self, constants, positions, expected):
    model_plan = np.zeros((1, constants.IDX_N, 3))
    model_plan[0, :, 0] = positions
    result = plan_x_idxs_helper(constants, SimpleNamespace(POSITION=slice(0, 3)), {"plan": model_plan})
    np.testing.assert_allclose(result, expected, rtol=1e-15, atol=0.0, equal_nan=True)
    self.assertIs(type(np.nan), float)
    for value in result:
      self.assertIs(type(value), float)
      if math.isnan(value):
        self.assertEqual(struct.pack("=d", value), struct.pack("=d", np.nan))


if __name__ == "__main__":
  unittest.main()
