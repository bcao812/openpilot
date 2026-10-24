from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.common.constants import ACCELERATION_DUE_TO_GRAVITY
from openpilot.selfdrive.locationd.helpers import Pose, PoseCalibrator
from openpilot.selfdrive.locationd.lagd import LateralLagEstimator
from openpilot.selfdrive.locationd.paramsd import VehicleParamsLearner, CarKalman, ObservationKind
from openpilot.selfdrive.locationd.torqued import TorqueEstimator


CALIBRATIONS = [(0.0, 0.0, 0.0), (0.12, -0.08, 0.2), (-0.16, 0.05, -0.1)]
ANGULAR_STDS = [(0.01, 0.02, 0.03), (0.0, 0.0, 0.0), (np.nan, 0.02, 0.03), (0.01, np.inf, 0.03)]


def device_motion(angular_std=ANGULAR_STDS[0]):
  return log.DeviceMotion(
    timestamp=5_000_000_000,
    orientationNED=log.DeviceMotion.XYZMeasurement(x=0.01, y=-0.03, z=0.2, xStd=0.003, yStd=0.004, zStd=0.005, valid=True),
    velocityDevice=log.DeviceMotion.XYZMeasurement(x=20.0, y=0.2, z=-0.1),
    accelerationDevice=log.DeviceMotion.XYZMeasurement(x=0.2, y=-0.3, z=0.4),
    angularVelocityDevice=log.DeviceMotion.XYZMeasurement(x=0.004, y=-0.002, z=0.03,
                                                         xStd=angular_std[0], yStd=angular_std[1], zStd=angular_std[2], valid=True),
    inputsOK=True, sensorsOK=True, posenetOK=True,
  )


def calibrate(calibrator, rpy, status=log.ExtrinsicsCalibration.Status.calibrated):
  calibrator.feed_extrinsics_calibration(log.ExtrinsicsCalibration(rpyCalib=list(rpy), calStatus=status))


@pytest.mark.parametrize("rpy", CALIBRATIONS)
@pytest.mark.parametrize("angular_std", ANGULAR_STDS)
def test_angular_velocity_matches_full_pose_including_nonfinite_std(rpy, angular_std):
  msg = device_motion(angular_std)
  calibrator = PoseCalibrator()
  calibrate(calibrator, rpy)
  with np.errstate(invalid="ignore"):
    expected = calibrator.build_calibrated_pose(Pose.from_device_motion(msg)).angular_velocity
    actual = calibrator.build_calibrated_angular_velocity(msg)
  np.testing.assert_array_equal(actual.xyz, expected.xyz)
  np.testing.assert_array_equal(actual.xyz_std, expected.xyz_std)


def test_angular_velocity_uses_current_measurement_and_calibration_without_other_pose_fields():
  calibrator = PoseCalibrator()
  msg = device_motion()
  angular_only = SimpleNamespace(angularVelocityDevice=msg.angularVelocityDevice)
  outputs = []
  for i, rpy in enumerate(CALIBRATIONS):
    status = log.ExtrinsicsCalibration.Status.invalid if i == 1 else log.ExtrinsicsCalibration.Status.calibrated
    calibrate(calibrator, rpy, status)
    msg.angularVelocityDevice.z = 0.03 + 0.01 * i
    actual = calibrator.build_calibrated_angular_velocity(angular_only)
    expected = calibrator.build_calibrated_pose(Pose.from_device_motion(msg)).angular_velocity
    np.testing.assert_array_equal(actual.xyz, expected.xyz)
    np.testing.assert_array_equal(actual.xyz_std, expected.xyz_std)
    assert calibrator.calib_valid == (status == log.ExtrinsicsCalibration.Status.calibrated)
    outputs.append(actual.xyz)
  assert not np.array_equal(outputs[0], outputs[1])
  assert not np.array_equal(outputs[1], outputs[2])


@pytest.mark.parametrize("invalid_field", [None, "angularVelocityDevice", "inputsOK", "posenetOK"])
def test_lagd_device_motion_preserves_calibrated_yaw_and_health(invalid_field):
  estimator = LateralLagEstimator(car.CarParams(steerActuatorDelay=0.5), 0.05)
  calibrate(estimator.calibrator, CALIBRATIONS[1])
  msg = device_motion()
  if invalid_field == "angularVelocityDevice":
    msg.angularVelocityDevice.valid = False
  elif invalid_field is not None:
    setattr(msg, invalid_field, False)
  expected = estimator.calibrator.build_calibrated_pose(Pose.from_device_motion(msg)).angular_velocity
  with patch.object(estimator.calibrator, "build_calibrated_pose", side_effect=AssertionError("unused pose conversion")):
    estimator.handle_log(5.0, "deviceMotion", msg)
  assert estimator.yaw_rate == expected.yaw
  assert estimator.yaw_rate_std == expected.yaw_std
  assert estimator.pose_valid == (invalid_field is None)
  assert estimator.t == 5.0


@pytest.mark.parametrize("angular_std", ANGULAR_STDS)
@pytest.mark.parametrize("roll_std", [0.003, np.nan])
def test_paramsd_device_motion_preserves_raw_roll_and_kalman_observations(angular_std, roll_std):
  learner = VehicleParamsLearner.__new__(VehicleParamsLearner)
  learner.calibrator = PoseCalibrator()
  calibrate(learner.calibrator, CALIBRATIONS[1])
  learner.active = True
  learner.observed_roll = 0.0
  learner.kf = Mock(x=CarKalman.initial_x.copy())
  msg = device_motion(angular_std)
  msg.orientationNED.xStd = roll_std
  with np.errstate(invalid="ignore"):
    expected = learner.calibrator.build_calibrated_pose(Pose.from_device_motion(msg)).angular_velocity
    with patch.object(learner.calibrator, "build_calibrated_pose", side_effect=AssertionError("unused pose conversion")):
      learner.handle_log(99.0, "deviceMotion", msg)

  valid_std = angular_std == ANGULAR_STDS[0]
  expected_yaw = expected.yaw if valid_std else 0.0
  expected_yaw_std = expected.yaw_std if valid_std else np.radians(10.0)
  expected_roll_std = 2 * (np.radians(1) if np.isnan(roll_std) else msg.orientationNED.xStd)
  assert learner.observed_yaw_rate == expected_yaw
  assert learner.observed_roll == msg.orientationNED.x
  calls = learner.kf.predict_and_observe.call_args_list
  assert len(calls) == 5
  assert calls[0].args[:2] == (5.0, ObservationKind.ROAD_FRAME_YAW_RATE)
  np.testing.assert_array_equal(calls[0].args[2], [[-expected_yaw]])
  np.testing.assert_array_equal(calls[0].args[3], [[[expected_yaw_std ** 2]]])
  assert calls[1].args[:2] == (5.0, ObservationKind.ROAD_ROLL)
  np.testing.assert_array_equal(calls[1].args[2], [[msg.orientationNED.x]])
  np.testing.assert_array_equal(calls[1].args[3], [[[expected_roll_std ** 2]]])


@pytest.mark.parametrize("invalid_field", ["angularVelocityDevice", "sensorsOK", "posenetOK"])
def test_paramsd_device_motion_preserves_health_fallbacks(invalid_field):
  learner = VehicleParamsLearner.__new__(VehicleParamsLearner)
  learner.calibrator = PoseCalibrator()
  learner.active = True
  learner.observed_roll = 0.0
  learner.kf = Mock(x=CarKalman.initial_x.copy())
  msg = device_motion()
  if invalid_field == "angularVelocityDevice":
    msg.angularVelocityDevice.valid = False
  else:
    setattr(msg, invalid_field, False)
  learner.handle_log(99.0, "deviceMotion", msg)
  assert learner.observed_yaw_rate == (0.0 if invalid_field == "angularVelocityDevice" else msg.angularVelocityDevice.z)
  assert learner.observed_roll == (0.0 if invalid_field == "sensorsOK" else msg.orientationNED.x)
  assert learner.kf.predict_and_observe.call_count == (3 if invalid_field == "posenetOK" else 5)


def torque_estimator():
  estimator = TorqueEstimator.__new__(TorqueEstimator)
  estimator.calibrator = PoseCalibrator()
  estimator.hist_len = 3
  estimator.lag = 0.3
  estimator.track_all_points = True
  estimator.all_torque_points = []
  estimator.filtered_points = Mock()
  estimator.raw_points = {
    "carControl_t": [0.0, 3.0, 6.0], "lat_active": [True] * 3,
    "carState_t": [0.0, 3.0, 6.0], "vego": [20.0] * 3, "steer_override": [False] * 3,
    "carOutput_t": [0.0, 3.0, 6.0], "steer_torque": [0.2] * 3,
  }
  return estimator


@pytest.mark.parametrize("angular_std", ANGULAR_STDS)
def test_torqued_device_motion_preserves_raw_roll_and_updates_calibration(angular_std):
  estimator = torque_estimator()
  msg = device_motion(angular_std)
  for rpy in CALIBRATIONS:
    calibrate(estimator.calibrator, rpy)
    with np.errstate(invalid="ignore"):
      expected = estimator.calibrator.build_calibrated_pose(Pose.from_device_motion(msg)).angular_velocity
      expected_accel = 20.0 * expected.yaw - np.sin(msg.orientationNED.x) * ACCELERATION_DUE_TO_GRAVITY
      with patch.object(estimator.calibrator, "build_calibrated_pose", side_effect=AssertionError("unused pose conversion")):
        estimator.handle_log(99.0, "deviceMotion", msg)
    steer, lateral_accel = estimator.filtered_points.add_point.call_args.args
    assert steer == 0.2
    assert lateral_accel == expected_accel
    assert estimator.all_torque_points[-1] == [steer, expected_accel]
  assert estimator.filtered_points.add_point.call_count == len(CALIBRATIONS)


@pytest.mark.parametrize("invalid_field", ["angularVelocityDevice", "orientationNED", "inputsOK", "sensorsOK", "posenetOK", "history"])
def test_torqued_device_motion_preserves_health_and_history_gates(invalid_field):
  estimator = torque_estimator()
  msg = device_motion()
  if invalid_field in ("angularVelocityDevice", "orientationNED"):
    getattr(msg, invalid_field).valid = False
  elif invalid_field == "history":
    estimator.raw_points["steer_torque"].pop()
  else:
    setattr(msg, invalid_field, False)
  with patch.object(estimator.calibrator, "build_calibrated_angular_velocity") as transform:
    estimator.handle_log(99.0, "deviceMotion", msg)
  transform.assert_not_called()
  estimator.filtered_points.add_point.assert_not_called()
  assert estimator.all_torque_points == []
