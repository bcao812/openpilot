import numpy as np


CameraIdentity = tuple[str, str, str | None]

# Effective UI mapping in raw 1928x1208 wide-camera pixels for tizi/OX03C10.
# Fitted on 2026-10-07 HD pairs, validated on 2026-10-08 and 2026-10-09.
# This display correction is separate from physical lens calibration and model inputs.
_TIZI_OX03C10_WIDE_CORRECTION = np.array([
  [1.076643, 0.061032, -106.091602],
  [-0.021109, 1.068620, -33.735983],
  [0.0, 0.0, 1.0],
])


def get_overlay_projection(calib_transform: np.ndarray, camera_identity: CameraIdentity | None,
                           is_wide_camera: bool, frame_size: tuple[int, int]) -> np.ndarray:
  """Correct the verified wide overlay before applying the camera's screen transform."""
  if is_wide_camera and camera_identity == ("tizi", "ox03c10", "ox03c10") and frame_size == (1928, 1208):
    return _TIZI_OX03C10_WIDE_CORRECTION @ calib_transform
  return calib_transform
