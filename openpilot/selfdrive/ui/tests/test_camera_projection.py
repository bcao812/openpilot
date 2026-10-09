import itertools
import json
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyray as rl

from openpilot.common.transformations.camera import DEVICE_CAMERAS, view_frame_from_device_frame
from openpilot.common.transformations.orientation import rot_from_euler
from openpilot.selfdrive.ui.onroad.camera_projection import get_overlay_projection
from openpilot.selfdrive.ui.tests.test_sunnylink_removed import HeadlessUITestCase


VERIFIED_CAMERA = ("tizi", "ox03c10", "ox03c10")
VERIFIED_SIZE = (1928, 1208)


def project(matrix, pixels):
  homogeneous = np.column_stack((pixels, np.ones(len(pixels)))) @ matrix.T
  return homogeneous[:, :2] / homogeneous[:, 2:]


class TestOverlayProjection(unittest.TestCase):
  def test_only_verified_hardware_stream_and_raw_size_are_corrected(self):
    original = np.array([[964., 567., 0.], [604., 0., 567.], [1., 0., 0.]])
    identities = [None, ("unlisted", "unknown", None)]
    identities += [(device, narrow, wide) for device, narrow in DEVICE_CAMERAS for wide in (None, "unknown", "ar0231", "ox03c10", "os04c10")]
    for identity, wide, size in itertools.product(identities, (False, True), (VERIFIED_SIZE, (964, 604), (1344, 760))):
      with self.subTest(identity=identity, wide=wide, size=size):
        result = get_overlay_projection(original, identity, wide, size)
        if identity == VERIFIED_CAMERA and wide and size == VERIFIED_SIZE:
          self.assertFalse(np.array_equal(result, original))
          np.testing.assert_array_equal(result[2], original[2])
        else:
          self.assertIs(result, original)

  def test_independent_hd_correspondences_improve_median_and_tail_error(self):
    fixture = json.loads((Path(__file__).parent / "fixtures" / "wide_projection_2026_10_08.json").read_text())
    camera = DEVICE_CAMERAS["tizi", "ox03c10"]
    view = view_frame_from_device_frame
    for frame in fixture["frames"]:
      with self.subTest(frame=frame["id"]):
        points = np.array(frame["correspondences"])
        narrow, observed_wide = points[:, :2], points[:, 2:]
        nominal = camera.wide_road.intrinsics @ view @ rot_from_euler(frame["wide_from_device_euler"]) @ view.T
        nominal = nominal @ camera.narrow_road.intrinsics_inv
        corrected = get_overlay_projection(nominal, tuple(fixture["camera_identity"]), True, tuple(fixture["frame_size"]))
        before = np.linalg.norm(project(nominal, narrow) - observed_wide, axis=1)
        after = np.linalg.norm(project(corrected, narrow) - observed_wide, axis=1)
        # Every match from each preselected validation frame is retained, including the tails.
        for quantile, limit in ((50, 5.0), (95, 10.0)):
          self.assertLess(np.percentile(after, quantile), 0.5 * np.percentile(before, quantile))
          self.assertLess(np.percentile(after, quantile), limit)


class Messages(dict):
  pass


class TestCameraFrameProjection(HeadlessUITestCase):
  def setUp(self):
    super().setUp()
    from openpilot.system.ui.lib.application import gui_app
    self.enterContext(patch.object(gui_app, "sunnypilot_ui", return_value=True))
    from openpilot.selfdrive.ui.onroad import augmented_road_view
    self.road = augmented_road_view
    self.sm = Messages()
    self.enterContext(patch.object(self.road, "ui_state", SimpleNamespace(sm=self.sm)))

  def make_view(self, identity=VERIFIED_CAMERA, wide=True):
    device, narrow_sensor, wide_sensor = identity or ("unknown", "unknown", None)
    self.sm.update(
      deviceState=SimpleNamespace(deviceType=device),
      narrowRoadCameraState=SimpleNamespace(sensor=narrow_sensor),
      wideRoadCameraState=SimpleNamespace(sensor=wide_sensor),
      extrinsicsCalibration=SimpleNamespace(calStatus=self.road.CALIBRATED, rpyCalib=[0.007, 0.058, 0.018],
                                            wideFromDeviceEuler=[0.002, 0.065, -0.012]),
    )
    self.sm.seen = {"deviceState": identity is not None, "narrowRoadCameraState": identity is not None,
                    "wideRoadCameraState": wide_sensor is not None}
    self.sm.recv_frame = {"extrinsicsCalibration": 7}
    self.sm.updated = {"extrinsicsCalibration": True}
    self.sm.valid = {"extrinsicsCalibration": True}

    def camera_init(view, name, stream_type):
      view._stream_type = stream_type
      view.texture_y = view.texture_uv = view.egl_texture = view.shader = view.frame = view.client = None
      view.egl_images = {}
      view.available_streams = []

    # Run the real view initializer and ModelRenderer.set_transform without a GL window or IPC client.
    renderer = self.road.ModelRenderer.__new__(self.road.ModelRenderer)
    with ExitStack() as stack:
      stack.enter_context(patch.object(self.road.CameraView, "__init__", camera_init))
      stack.enter_context(patch.object(self.road.AugmentedRoadViewSP, "__init__", return_value=None))
      stack.enter_context(patch.object(self.road, "ModelRenderer", return_value=renderer))
      for name in ("HondaRadarOverlay", "HudRenderer", "AlertRenderer", "DriverStateRenderer"):
        stack.enter_context(patch.object(self.road, name))
      view = self.road.AugmentedRoadView(self.road.WIDE_CAM if wide else self.road.NARROW_ROAD_CAM)
    view._content_rect = rl.Rectangle(37, 21, 2160, 1080)
    view._update_calibration()
    return view

  def baseline_matrices(self, view):
    """Reference the pre-correction camera framing, using only physical K and calibration."""
    config = view.device_camera or self.road.DEFAULT_DEVICE_CAMERA
    wide = view.stream_type == self.road.WIDE_CAM
    camera = config.wide_road if wide else config.narrow_road
    calibration = view.view_from_wide_calib if wide else view.view_from_calib
    projection = camera.intrinsics @ calibration
    rect = view._content_rect
    dimensions, origin = np.array([rect.width, rect.height]), np.array([rect.x, rect.y])
    center = camera.intrinsics[:2, 2]
    zoom = max(2.0 if wide else 1.1, *(dimensions / (2 * center)))
    vanishing = projection @ self.road.INF_POINT
    limits = np.maximum(0.0, center * zoom - dimensions / 2 - 5)
    offsets = np.clip((vanishing[:2] / vanishing[2] - center) * zoom, -limits, limits)
    frame = np.eye(3)
    frame[0, 0], frame[1, 1] = 2 * center * zoom / dimensions
    frame[:2, 2] = -2 * offsets / dimensions
    screen = np.diag([zoom, zoom, 1.0])
    screen[:2, 2] = origin + dimensions / 2 - offsets - center * zoom
    return frame, screen, projection, camera

  def assert_projection(self, view):
    expected_frame, screen, projection, camera = self.baseline_matrices(view)
    frame = view._calc_frame_matrix(view._content_rect)
    np.testing.assert_allclose(frame, expected_frame, rtol=0, atol=1e-12)
    expected_overlay = screen @ get_overlay_projection(projection, view._camera_identity,
                                                       view.stream_type == self.road.WIDE_CAM, camera.size)
    np.testing.assert_allclose(view.model_renderer._car_space_transform, expected_overlay, rtol=1e-6, atol=1e-4)
    return frame

  def test_background_keeps_original_framing_and_correction_precedes_screen_zoom(self):
    view = self.make_view()
    frame = self.assert_projection(view)
    _, screen, nominal, _ = self.baseline_matrices(view)
    self.assertFalse(np.allclose(view.model_renderer._car_space_transform, screen @ nominal))
    # A stable camera/layout keeps the existing geometry cache hot.
    view.model_renderer._transform_dirty = False
    self.assertIs(view._calc_frame_matrix(view._content_rect), frame)
    self.assertFalse(view.model_renderer._transform_dirty)

  def test_unverified_hardware_and_narrow_camera_keep_original_overlay(self):
    identities = [None, ("unlisted", "unknown", None), ("tici", "ox03c10", "ox03c10"),
                  ("tizi", "ar0231", "ar0231"), ("mici", "os04c10", "os04c10"), ("tizi", "ox03c10", "ar0231")]
    for identity, wide in itertools.product(identities, (False, True)):
      with self.subTest(identity=identity, wide=wide):
        view = self.make_view(identity, wide)
        self.assert_projection(view)
        _, screen, nominal, _ = self.baseline_matrices(view)
        np.testing.assert_allclose(view.model_renderer._car_space_transform, screen @ nominal, rtol=1e-6, atol=1e-4)

  def test_identity_and_late_wide_sensor_invalidate_cached_projection(self):
    view = self.make_view(None)
    frame = self.assert_projection(view)
    original_overlay = view.model_renderer._car_space_transform.copy()
    self.sm.seen.update(deviceState=True, narrowRoadCameraState=True)
    self.sm["narrowRoadCameraState"].sensor = "ox03c10"
    for device in ("unknown", "tizi"):
      self.sm["deviceState"].deviceType = device
      view.model_renderer._transform_dirty = False
      view._update_calibration()
      updated = self.assert_projection(view)
      self.assertIsNot(updated, frame)
      self.assertTrue(view.model_renderer._transform_dirty)
      np.testing.assert_array_equal(view.model_renderer._car_space_transform, original_overlay)
      frame = updated

    self.sm.seen["wideRoadCameraState"] = True
    for sensor, corrected in (("ox03c10", True), ("ar0231", False), ("ox03c10", True)):
      self.sm["wideRoadCameraState"].sensor = sensor
      view.model_renderer._transform_dirty = False
      view._update_calibration()
      updated = self.assert_projection(view)
      self.assertIsNot(updated, frame)
      self.assertTrue(view.model_renderer._transform_dirty)
      self.assertEqual(np.array_equal(view.model_renderer._car_space_transform, original_overlay), not corrected)
      frame = updated

  def test_completed_stream_switch_removes_and_restores_correction(self):
    view = self.make_view()
    self.assert_projection(view)
    wide_overlay = view.model_renderer._car_space_transform.copy()
    # CameraView changes _stream_type after the target IPC stream has a frame.
    view._stream_type = self.road.NARROW_ROAD_CAM
    self.assert_projection(view)
    _, screen, nominal, _ = self.baseline_matrices(view)
    np.testing.assert_allclose(view.model_renderer._car_space_transform, screen @ nominal, rtol=1e-6, atol=1e-4)
    view._stream_type = self.road.WIDE_CAM
    self.assert_projection(view)
    np.testing.assert_array_equal(view.model_renderer._car_space_transform, wide_overlay)

  def test_layout_translation_invalidates_overlay_without_changing_background_scale(self):
    view = self.make_view()
    frame = self.assert_projection(view)
    overlay = view.model_renderer._car_space_transform.copy()
    view._content_rect.x += 83
    view._content_rect.y += 17
    updated = self.assert_projection(view)
    np.testing.assert_array_equal(updated, frame)
    expected = overlay.copy()
    expected[0] += 83 * overlay[2]
    expected[1] += 17 * overlay[2]
    np.testing.assert_allclose(view.model_renderer._car_space_transform, expected, rtol=1e-6, atol=1e-4)

  def test_camera_size_change_invalidates_cache_and_disables_pixel_correction(self):
    view = self.make_view()
    frame = self.assert_projection(view)
    config = view.device_camera
    view.device_camera = replace(config, wide_road=replace(config.wide_road, width=964, height=604, focal_length=283.5))
    self.assertIsNot(self.assert_projection(view), frame)
    _, screen, nominal, _ = self.baseline_matrices(view)
    np.testing.assert_allclose(view.model_renderer._car_space_transform, screen @ nominal, rtol=1e-6, atol=1e-4)

  def test_new_calibration_recomputes_both_transforms(self):
    view = self.make_view()
    frame = self.assert_projection(view)
    overlay = view.model_renderer._car_space_transform.copy()
    self.sm.recv_frame["extrinsicsCalibration"] += 1
    self.sm["extrinsicsCalibration"].rpyCalib = [0.01, 0.09, -0.02]
    view._update_calibration()
    self.assertFalse(np.array_equal(self.assert_projection(view), frame))
    self.assertFalse(np.array_equal(view.model_renderer._car_space_transform, overlay))
