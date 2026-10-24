from collections import deque
from unittest.mock import patch

import numpy as np
import pytest

from opendbc.car.structs import car
from openpilot.selfdrive.locationd import lagd


DT = 0.05
DELAY = 0.3


def append_sample(estimator, t, okay=True, flat=False):
  desired = 0.1 if flat else np.cos(10 * t) * 0.3
  actual = 0.1 if flat else np.cos(10 * (t - DELAY)) * 0.3
  estimator.points.update(t, desired, actual, okay)
  estimator.t = t


def window_estimator(*, okay=True, flat=False, window_sec=lagd.MOVING_WINDOW_SEC):
  estimator = lagd.LateralLagEstimator(car.CarParams(steerActuatorDelay=0.5), DT, window_sec=window_sec)
  for t in np.arange(estimator.points.num_points) * DT:
    append_sample(estimator, t, okay, flat)
  return estimator


@pytest.mark.parametrize("reason", ["no_valid_points", "flat_signal", "no_new_points", "new_points_invalid", "not_enough_points"])
def test_invalid_window_skips_signal_processing_without_changing_history(reason):
  estimator = window_estimator(okay=reason != "no_valid_points", flat=reason == "flat_signal",
                               window_sec=10.0 if reason == "not_enough_points" else lagd.MOVING_WINDOW_SEC)
  if reason == "no_new_points":
    estimator.last_estimate_t = estimator.t
  elif reason == "new_points_invalid":
    estimator.last_estimate_t = estimator.points.times[-11]
    mask = list(estimator.points.okay)
    mask[-10:] = [False] * 10
    estimator.points.okay = deque(mask, maxlen=len(mask))

  points_before = estimator.points.get()
  values_before = estimator.block_avg.values.copy()
  last_estimate_before = estimator.last_estimate_t
  with patch.object(lagd, "masked_symmetric_moving_average") as smooth, patch.object(estimator, "actuator_delay") as delay:
    estimator.update_estimate()
  smooth.assert_not_called()
  delay.assert_not_called()
  for before, after in zip(points_before, estimator.points.get(), strict=True):
    np.testing.assert_array_equal(before, after)
  np.testing.assert_array_equal(estimator.block_avg.values, values_before)
  assert estimator.block_avg.idx == 0
  assert estimator.block_avg.valid_blocks == 0
  assert estimator.last_estimate_t == last_estimate_before


def test_new_valid_sample_resumes_estimation_after_rejected_update():
  estimator = window_estimator()
  estimator.update_estimate()
  assert estimator.block_avg.idx == 1
  np.testing.assert_allclose(estimator.block_avg.get()[2], DELAY, atol=0.01)
  last_estimate_before = estimator.last_estimate_t

  append_sample(estimator, estimator.t + DT, okay=False)
  with patch.object(lagd, "masked_symmetric_moving_average") as smooth:
    estimator.update_estimate()
  smooth.assert_not_called()
  assert estimator.block_avg.idx == 1
  assert estimator.last_estimate_t == last_estimate_before

  append_sample(estimator, estimator.t + DT)
  estimator.update_estimate()
  assert estimator.block_avg.idx == 2
  assert estimator.last_estimate_t == estimator.t
  np.testing.assert_allclose(estimator.block_avg.get()[2], DELAY, atol=0.01)


def test_invalid_latest_sample_does_not_discard_unconsumed_valid_history():
  estimator = window_estimator()
  estimator.last_estimate_t = estimator.t - 1.0
  append_sample(estimator, estimator.t + DT, okay=False)
  estimator.update_estimate()
  assert estimator.block_avg.idx == 1
  assert estimator.last_estimate_t == estimator.t
  np.testing.assert_allclose(estimator.block_avg.get()[2], DELAY, atol=0.01)
