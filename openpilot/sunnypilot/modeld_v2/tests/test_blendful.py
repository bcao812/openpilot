"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import copy
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
  return np.array(out) * v ** 2  # m/s^2


class TestBlendful(unittest.TestCase):
  def setUp(self):
    self.bf = Blendful()

  def test_centered_adds_nothing(self):
    push = drive(self.bf, 5.0)
    assert self.bf.trust > 0.9
    assert np.abs(push).max() < 1e-3

  def test_pulls_toward_lane_center(self):
    push = drive(self.bf, 5.0, center=0.5)
    assert push[-1] > 0.02  # + y side, where the lane center is
    assert self.bf.trust > 0.9
    assert np.all(np.diff(push) <= BUILD * DT_MDL + 1e-9)

  def test_flipping_sides_builds_slowly(self):
    k = 0.002  # 0.8 m/s^2 at 20 m/s
    drive(self.bf, 5.0, k=k, center=0.5)
    push = drive(self.bf, 5.0, k=k, center=-0.5)
    rate = np.diff(push) / DT_MDL
    assert push[-1] < 0
    assert np.all(rate >= -RELEASE - 1e-6)
    assert np.all(rate[push[:-1] <= 0] >= -BUILD - 1e-6)  # once past zero it builds
    crossing = push[1:][(push[:-1] > 0) & (push[1:] <= 0)]
    assert len(crossing) and np.all(crossing >= -BUILD * DT_MDL - 1e-9)

  def test_weak_lines_drive_laneless(self):
    clear = drive(Blendful(), 5.0, center=0.5)[-1]
    faint = Blendful()
    push = drive(faint, 5.0, center=0.5, probs=(0.1, 0.1))
    assert faint.trust < 0.25 and abs(push[-1]) < 0.25 * clear  # faint lines = trust and push scale down
    blurry = Blendful()
    push = drive(blurry, 5.0, center=0.5, stds=(0.8, 0.8))
    assert blurry.trust == 0.0 and np.abs(push).max() < 1e-6  # lines too uncertain = laneless

  def test_lone_line_holds_through_a_blink_in_a_curve(self):
    k, v = 1 / 20, 6.0
    drive(self.bf, 6.0, v=v, k=k, plan=0.4, probs=(0.9, 0.05))
    before = self.bf.trust
    drive(self.bf, 0.25, v=v, k=k, plan=0.4, probs=(0.1, 0.05), stds=(1.0, 0.1))
    bottom = self.bf.trust
    drive(self.bf, 2.0, v=v, k=k, plan=0.4, probs=(0.9, 0.05))
    assert before > 0.5
    assert bottom > 0.5 * before  # keeps latched through flicker
    assert self.bf.trust > 0.85 * before

  def test_yields_when_the_model_goes_around(self):
    drive(self.bf, 6.0)
    push = drive(self.bf, 3.0, plan=0.8)
    assert self.bf.going_around
    assert self.bf.trust < 0.05
    assert abs(push[-1]) < 0.01

  def test_fit_does_not_grow_while_trust_falls(self):
    drive(self.bf, 5.0, center=0.3)
    for i in range(40):
      dk, trust = self.bf.dk, self.bf.trust
      p = 0.9 - 0.02 * i
      self.bf.update(frame(center=0.3, plan=-0.02 * i, probs=(p, p)), 20.0, True, 0.0)
      if self.bf.trust < trust:
        assert abs(self.bf.dk) <= abs(dk) + 1e-12

  def test_toggle_off_returns_zero_and_resets(self):
    drive(self.bf, 5.0, center=0.5)
    assert self.bf.curvature != 0.0
    assert self.bf.update(frame(center=0.5), 20.0, True, 0.0, enabled=False) == 0.0
    assert self.bf.curvature == 0.0 and self.bf.trust == 0.0

  def test_fades_below_min_speed(self):
    drive(self.bf, 5.0, v=10.0, center=0.5)
    start = self.bf.curvature
    out = np.array([self.bf.update(frame(center=0.5), 4.0, True, 0.0) for _ in range(100)])
    assert self.bf.trust == 0.0
    assert abs(out[0]) < abs(start)
    assert abs(out[-1]) < 1e-6

  def test_learned_width_and_confidence_survive_a_side_disappearing(self):
    drive(self.bf, 6.0, center=0.3, width=4.4)
    old_prob = self.bf.line_probs.x[1]
    self.assertAlmostEqual(self.bf.lane_width.x, 4.4)
    current = frame(center=0.3, width=4.4, probs=(0.9, 0.0))
    self.bf.update(current, 20.0, True, 0.0)
    self.assertGreater(self.bf.line_probs.x[1], 0.5)
    self.assertLess(self.bf.line_probs.x[1], old_prob)
    self.assertGreater(self.bf.line_gone.x[1], 0.0)
    fresh = Blendful()
    for _ in range(100):
      self.bf.update(current, 20.0, True, 0.0)
      fresh.update(current, 20.0, True, 0.0)
    self.assertAlmostEqual(self.bf.lane_width.x, 4.4)
    self.assertEqual(fresh.lane_width.x, LANE_WIDTH_DEFAULT)
    self.assertNotAlmostEqual(self.bf.curvature, fresh.curvature, places=8)

  def test_zero_current_weights_can_reuse_last_fit_while_trust_releases(self):
    k = 0.002
    drive(self.bf, 6.0, center=0.5, k=k)
    old_fit, old_trust = self.bf.dk, self.bf.trust
    result = self.bf.update(frame(center=0.5, k=k, probs=(0.0, 0.0), stds=(1.0, 1.0)), 20.0, True, k)
    self.assertEqual((self.bf.l_prob, self.bf.r_prob), (0.0, 0.0))
    self.assertEqual(self.bf.dk, old_fit)
    self.assertGreater(self.bf.trust, 0.0)
    self.assertLess(self.bf.trust, old_trust)
    self.assertGreater(result, 0.0)

  def test_desire_and_low_speed_fade_without_resetting_probability_history(self):
    for active, speed in ((False, 20.0), (True, MIN_SPEED - 1e-6)):
      with self.subTest(active=active, speed=speed):
        bf = Blendful()
        drive(bf, 6.0, center=0.5, k=0.002)
        before, probs = bf.curvature, bf.line_probs.x.copy()
        result = bf.update(frame(center=0.5, k=0.002), speed, active, 0.002)
        self.assertGreater(result, 0.0)
        self.assertLess(result, before)
        self.assertEqual(bf.trust, 0.0)
        np.testing.assert_array_equal(bf.line_probs.x, probs)
        self.assertFalse(bf.lane_width.initialized)
        self.assertFalse(bf.usual_offset.initialized)
    cold = Blendful()
    self.assertEqual(cold.update(frame(center=0.5), MIN_SPEED - 1e-6, True, 0.0), 0.0)
    self.assertGreater(cold.update(frame(center=0.5), MIN_SPEED, True, 0.0), 0.0)

  def assert_reset(self, bf):
    self.assertEqual((bf.curvature, bf.smooth, bf.trust, bf.line_trust, bf.dk, bf.l_prob, bf.r_prob), (0.0,) * 7)
    self.assertFalse(bf.going_around)
    self.assertEqual(bf.lane_width.x, LANE_WIDTH_DEFAULT)
    self.assertFalse(bf.lane_width.initialized)
    self.assertFalse(bf.usual_offset.initialized)
    for history in (bf.line_probs, bf.line_probs_slow, bf.line_gone):
      np.testing.assert_array_equal(history.x, np.zeros(2))

  def test_unfitted_finite_plan_preserves_source_fade(self):
    for label, plan_x in (('zero_span', np.zeros(33)), ('descending', X[::-1])):
      current = frame(center=0.5, k=0.002)
      current['plan'][0, :, 0] = plan_x
      for active, speed in ((False, 20.0), (True, MIN_SPEED - 1e-6)):
        with self.subTest(plan=label, active=active, speed=speed):
          bf = Blendful()
          drive(bf, 6.0, center=0.5, k=0.002)
          self.assertGreater(bf.curvature, 0.0)
          source = copy.deepcopy(bf)
          for index in range(5):
            # Bypass only the target input adapter for the source reference;
            # the retained source body ignores plan x throughout fading.
            expected = source._update_source(current, speed, active, 0.002)
            self.assertEqual(bf.update(current, speed, active, 0.002), expected)
            self.assertEqual(bf.smooth, source.smooth)
            np.testing.assert_array_equal(bf.line_probs.x, source.line_probs.x)
            if index == 0:
              self.assertGreater(expected, 0.0)
      with self.subTest(plan=label, active=True, speed=MIN_SPEED):
        bf = Blendful()
        drive(bf, 6.0, center=0.5, k=0.002)
        self.assertGreater(bf.curvature, 0.0)
        self.assertEqual(bf.update(current, MIN_SPEED, True, 0.002), 0.0)
        self.assert_reset(bf)

  def test_malformed_numerical_inputs_reset_instead_of_poisoning_history(self):
    cases = [None, {}, frame(plan_width=2), frame(plan_width=14)]
    for key, shape in (('lane_lines', (4, 33, 2)), ('lane_lines_stds', (1, 4, 32, 2)),
                       ('lane_lines_prob', (1, 4)), ('plan', (1, 32, 15))):
      malformed = frame(center=0.5)
      malformed[key] = np.zeros(shape)
      cases.append(malformed)
    for std in (0.0, -0.1, np.nan, np.inf):
      cases.append(frame(center=0.5, stds=(std, 0.1)))
    for prob in (-0.1, 1.1, np.nan, np.inf):
      cases.append(frame(center=0.5, probs=(prob, 0.9)))
    for key in ('plan', 'lane_lines'):
      for value in (np.nan, np.inf):
        malformed = frame(center=0.5, probs=(0.0, 0.9))
        if key == 'plan':
          malformed[key][0, 10, 1] = value
        else:
          malformed[key][0, 1, 10, 0] = value
        cases.append(malformed)
    for xs in (X[::-1], np.zeros(33)):
      malformed = frame(center=0.5)
      malformed['plan'][0, :, 0] = xs
      cases.append(malformed)
    for index, malformed in enumerate(cases):
      with self.subTest(index=index):
        bf = Blendful()
        drive(bf, 3.0, center=0.5)
        self.assertGreater(bf.curvature, 0.0)
        self.assertEqual(bf.update(malformed, 20.0, True, 0.0), 0.0)
        self.assert_reset(bf)
        self.assertEqual(bf.update(frame(center=-0.5), 20.0, True, 0.0),
                         Blendful().update(frame(center=-0.5), 20.0, True, 0.0))

  def test_nonfinite_motion_and_empty_fit_windows_are_contained(self):
    for speed, curvature in ((np.nan, 0.0), (np.inf, 0.0), (-1.0, 0.0), (20.0, np.nan),
                              (20.0, np.inf), (500.0, 0.0), (1e308, 0.0)):
      with self.subTest(speed=speed, curvature=curvature):
        bf = Blendful()
        drive(bf, 3.0, center=0.5)
        self.assertEqual(bf.update(frame(center=0.5), speed, True, curvature), 0.0)
        self.assert_reset(bf)

  def test_source_endpoint_interpolation_accepts_short_or_flat_tail_plan(self):
    for xs in (np.minimum(X, 20), X * 0.01, X + 100.0):
      with self.subTest(xs=xs):
        bf = Blendful()
        current = frame(center=0.5)
        current['plan'][0, :, 0] = xs
        for _ in range(80):
          bf.update(current, 20.0, True, 0.0)
        self.assertGreater(bf.trust, 0.8)
        self.assertGreater(bf.curvature, 0.0)

  def test_full_plan_tensor_and_source_geometry_fixture_agree(self):
    full, source_fixture = Blendful(), Blendful()
    np.testing.assert_array_equal(drive(full, 4.0, center=0.5), drive(source_fixture, 4.0, center=0.5, plan_width=3))


if __name__ == '__main__':
  unittest.main()
