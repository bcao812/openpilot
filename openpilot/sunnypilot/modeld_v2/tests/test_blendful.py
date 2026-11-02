"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import unittest
import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.modeld_v2.blendful import Blendful, LANE_WIDTH_DEFAULT, MIN_SPEED
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants

X = np.array(ModelConstants.X_IDXS)
BUILD, RELEASE = 0.5, 2.0


def frame(center=0.0, plan=0.0, k=0.0, probs=(0.9, 0.9), stds=(0.1, 0.1), width=3.5, plan_width=15):
  road = k * X ** 2 / 2
  lines = np.zeros((1, 4, 33, 2))
  lane_stds = np.ones((1, 4, 33, 2))
  for i, off in enumerate((-1.5 * width, -width / 2, width / 2, 1.5 * width)):
    lines[0, i, :, 0] = road + center + off
  lane_stds[0, 1, :, 0], lane_stds[0, 2, :, 0] = stds
  lane_probs = np.zeros((1, 8))
  lane_probs[0, 3], lane_probs[0, 5] = probs
  path = np.zeros((1, 33, plan_width))
  path[0, :, 0] = X
  path[0, :, 1] = road + plan * np.clip(X / 10, 0, 1)
  return {'lane_lines': lines, 'lane_lines_stds': lane_stds, 'lane_lines_prob': lane_probs, 'plan': path}


def drive(bf, seconds, v=20.0, k=0.0, **kw):
  out = [bf.update(frame(k=k, **kw), v, True, k) for _ in range(int(seconds / DT_MDL))]
  return np.array(out) * v ** 2


class TestBlendful(unittest.TestCase):
  def setUp(self):
    self.bf = Blendful()

  def assert_reset(self, bf):
    self.assertEqual((bf.curvature, bf.smooth, bf.trust, bf.line_trust, bf.dk, bf.l_prob, bf.r_prob), (0.0,) * 7)
    self.assertEqual(bf.current_sides, (False, False))
    self.assertFalse(bf.going_around)
    self.assertEqual(bf.since_turning, 0.0)
    self.assertEqual(bf.lane_width.x, LANE_WIDTH_DEFAULT)
    self.assertFalse(bf.lane_width.initialized)
    self.assertFalse(bf.usual_offset.initialized)
    np.testing.assert_array_equal(bf.line_probs.x, np.zeros(2))
    np.testing.assert_array_equal(bf.line_probs_slow.x, np.zeros(2))

  def test_centered_adds_nothing(self):
    push = drive(self.bf, 5.0)
    self.assertGreater(self.bf.trust, 0.9)
    self.assertLess(np.abs(push).max(), 1e-3)

  def test_pulls_toward_lane_center(self):
    push = drive(self.bf, 5.0, center=0.5)
    self.assertGreater(push[-1], 0.02)
    self.assertGreater(self.bf.trust, 0.9)
    self.assertTrue(np.all(np.diff(push) <= BUILD * DT_MDL + 1e-9))

  def test_flipping_sides_builds_slowly(self):
    k = 0.002
    drive(self.bf, 5.0, k=k, center=0.5)
    push = drive(self.bf, 5.0, k=k, center=-0.5)
    rate = np.diff(push) / DT_MDL
    self.assertLess(push[-1], 0)
    self.assertTrue(np.all(rate >= -RELEASE - 1e-6))
    self.assertTrue(np.all(rate[push[:-1] <= 0] >= -BUILD - 1e-6))
    crossing = push[1:][(push[:-1] > 0) & (push[1:] <= 0)]
    self.assertTrue(len(crossing) and np.all(crossing >= -BUILD * DT_MDL - 1e-9))

  def test_weak_lines_drive_laneless(self):
    for kwargs in ({'probs': (0.1, 0.1)}, {'stds': (0.8, 0.8)}):
      with self.subTest(kwargs=kwargs):
        bf = Blendful()
        push = drive(bf, 5.0, center=0.5, **kwargs)
        np.testing.assert_array_equal(push, np.zeros_like(push))
        self.assert_reset(bf)

  def test_lone_line_withdraws_on_first_missed_observation(self):
    k, v = 1 / 20, 6.0
    drive(self.bf, 6.0, v=v, k=k, plan=0.4, probs=(0.9, 0.05))
    self.assertGreater(self.bf.trust, 0.5)
    self.assertNotEqual(self.bf.curvature, 0.0)
    self.assertEqual(self.bf.update(frame(k=k, plan=0.4, probs=(0.1, 0.05), stds=(1.0, 0.1)), v, True, k), 0.0)
    self.assert_reset(self.bf)
    fresh = Blendful()
    for _ in range(20):
      current = frame(k=k, plan=-0.4, probs=(0.9, 0.05))
      self.assertEqual(self.bf.update(current, v, True, k), fresh.update(current, v, True, k))

  def test_yields_when_the_model_goes_around(self):
    drive(self.bf, 6.0)
    push = drive(self.bf, 3.0, plan=0.8)
    self.assertTrue(self.bf.going_around)
    self.assertLess(self.bf.trust, 0.05)
    self.assertLess(abs(push[-1]), 0.01)

  def test_fit_does_not_grow_while_trust_falls(self):
    drive(self.bf, 5.0, center=0.3)
    for i in range(40):
      dk, trust = self.bf.dk, self.bf.trust
      p = 0.9 - 0.02 * i
      self.bf.update(frame(center=0.3, plan=-0.02 * i, probs=(p, p)), 20.0, True, 0.0)
      if self.bf.trust < trust:
        self.assertLessEqual(abs(self.bf.dk), abs(dk) + 1e-12)

  def test_disqualification_clears_all_state_and_reentry_starts_fresh(self):
    cases = (
      {'enabled': False}, {'active': False}, {'v_ego': MIN_SPEED - 1e-6}, {'v_ego': 0.0},
      {'v_ego': -1.0}, {'v_ego': np.nan}, {'v_ego': np.inf}, {'model_curvature': np.nan}, {'model_curvature': np.inf},
    )
    for overrides in cases:
      with self.subTest(overrides=overrides):
        bf = Blendful()
        drive(bf, 5.0, center=0.5, width=4.4)
        self.assertGreater(bf.curvature, 0.0)
        args = {'model_output': frame(center=0.5), 'v_ego': 20.0, 'active': True, 'model_curvature': 0.0, 'enabled': True}
        self.assertEqual(bf.update(**(args | overrides)), 0.0)
        self.assert_reset(bf)
        fresh = Blendful()
        current = frame(center=-0.5)
        self.assertEqual(bf.update(current, 20.0, True, 0.0), fresh.update(current, 20.0, True, 0.0))
        self.assertLess(bf.curvature, 0.0)

  def test_exact_speed_and_raw_probability_thresholds(self):
    self.assertGreater(self.bf.update(frame(center=0.5, probs=(0.3, 0.3)), MIN_SPEED, True, 0.0), 0.0)
    drive(self.bf, 5.0, center=0.5)
    self.assertGreater(self.bf.line_probs.x.min(), 0.85)
    self.assertEqual(self.bf.update(frame(center=0.5, probs=(0.3 - 1e-9, 0.3 - 1e-9)), 20.0, True, 0.0), 0.0)
    self.assert_reset(self.bf)

  def test_side_change_discards_old_width_and_fit(self):
    for probs in ((0.9, 0.0), (0.0, 0.9), (0.9, 0.9)):
      with self.subTest(probs=probs):
        warm_probs = (0.9, 0.9) if not all(probs) else (0.9, 0.0)
        bf = Blendful()
        drive(bf, 6.0, center=0.7, width=4.4, probs=warm_probs)
        fresh = Blendful()
        for _ in range(20):
          current = frame(center=-0.5, width=3.6, probs=probs)
          self.assertEqual(bf.update(current, 20.0, True, 0.0), fresh.update(current, 20.0, True, 0.0))
        self.assertEqual(bf.lane_width.x, fresh.lane_width.x)
        # A two-line width is computed from floating-point coordinates.
        self.assertAlmostEqual(bf.lane_width.x, LANE_WIDTH_DEFAULT, places=12)

  def test_excluded_side_geometry_never_affects_single_side_fit(self):
    for missing_side in (0, 1):
      for garbage in (np.nan, np.inf, -np.inf, 1e100):
        with self.subTest(missing_side=missing_side, garbage=garbage):
          probs = [0.9, 0.9]
          probs[missing_side] = 0.0
          clean, broken = Blendful(), Blendful()
          current = frame(center=0.5, width=3.6, probs=probs)
          malformed = {key: value.copy() for key, value in current.items()}
          malformed['lane_lines'][0, missing_side + 1] = garbage
          malformed['lane_lines_stds'][0, missing_side + 1] = garbage
          for _ in range(50):
            self.assertEqual(broken.update(malformed, 20.0, True, 0.0), clean.update(current, 20.0, True, 0.0))
          self.assertGreater(broken.curvature, 0.0)
          self.assertEqual(broken.lane_width.x, LANE_WIDTH_DEFAULT)

  def test_std_weighting_prefers_the_more_precise_current_line(self):
    for stds, sign in (((0.1, 0.5), -1), ((0.5, 0.1), 1)):
      with self.subTest(stds=stds):
        bf = Blendful()
        push = drive(bf, 5.0, width=4.0, probs=(0.5, 0.5), stds=stds)
        self.assertGreater(sign * push[-1], 0.0)

  def test_invalid_current_std_probability_or_geometry_withdraws(self):
    cases = []
    for value in (0.0, -0.1, np.nan, np.inf, 0.6, 1e308):
      cases.append(frame(center=0.5, stds=(value, value)))
    for value in (-1.0, 1.1, np.nan, np.inf):
      cases.append(frame(center=0.5, probs=(value, value)))
    for value in (np.nan, np.inf):
      malformed = frame(center=0.5)
      malformed['lane_lines'][0, 1:3, 5, 0] = value
      cases.append(malformed)
    for index, malformed in enumerate(cases):
      with self.subTest(index=index):
        bf = Blendful()
        drive(bf, 3.0, center=0.5)
        self.assertEqual(bf.update(malformed, 20.0, True, 0.0), 0.0)
        self.assert_reset(bf)

  def test_malformed_shapes_plan_and_fit_windows_withdraw(self):
    cases = [None, {}, frame(plan_width=2), frame(plan_width=14)]
    for key, shape in (('lane_lines', (4, 33, 2)), ('lane_lines_stds', (1, 4, 32, 2)),
                       ('lane_lines_prob', (1, 4)), ('plan', (1, 32, 15))):
      malformed = frame(center=0.5)
      malformed[key] = np.zeros(shape)
      cases.append(malformed)
    for value in (np.nan, np.inf):
      malformed = frame(center=0.5)
      malformed['plan'][0, 10, 1] = value
      cases.append(malformed)
    for xs in (X[::-1], np.zeros(33), X + 100.0):
      malformed = frame(center=0.5)
      malformed['plan'][0, :, 0] = xs
      cases.append(malformed)
    for index, malformed in enumerate(cases):
      with self.subTest(index=index):
        bf = Blendful()
        drive(bf, 3.0, center=0.5)
        self.assertEqual(bf.update(malformed, 20.0, True, 0.0), 0.0)
        self.assert_reset(bf)
    for speed in (500.0, 1e308):
      with self.subTest(speed=speed):
        drive(self.bf, 3.0, center=0.5)
        self.assertEqual(self.bf.update(frame(center=0.5), speed, True, 0.0), 0.0)
        self.assert_reset(self.bf)

  def test_stopping_plan_plateau_and_short_horizon_keep_endpoint_interpolation(self):
    expected = drive(Blendful(), 3.0, center=0.5)
    for xs in (np.minimum(X, 20.0), X * 0.01):
      with self.subTest(xs=xs):
        bf = Blendful()
        current = frame(center=0.5)
        current['plan'][0, :, 0] = xs
        actual = np.array([bf.update(current, 20.0, True, 0.0) for _ in range(int(3.0 / DT_MDL))]) * 20.0 ** 2
        np.testing.assert_array_equal(actual, expected)

  def test_full_plan_tensor_and_source_geometry_fixture_agree(self):
    full, source_fixture = Blendful(), Blendful()
    np.testing.assert_array_equal(drive(full, 4.0, center=0.5), drive(source_fixture, 4.0, center=0.5, plan_width=3))


if __name__ == '__main__':
  unittest.main()
