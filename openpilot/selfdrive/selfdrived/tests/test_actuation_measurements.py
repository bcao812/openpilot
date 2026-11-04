"""Keep the excessive-actuation verdict and sample timing with reduced pose work."""
from unittest.mock import patch

import numpy as np
import pytest

from opendbc.car.interfaces import ACCEL_MAX
from opendbc.car.lateral import ISO_LATERAL_ACCEL
from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import new_event, selfdrive_event_harness
from openpilot.selfdrive.locationd.helpers import Pose, PoseCalibrator
from openpilot.selfdrive.selfdrived.helpers import (
  ExcessiveActuationCheck, ExcessiveActuationType, MIN_EXCESSIVE_ACTUATION_COUNT, MIN_LATERAL_ENGAGE_BUFFER,
)


CALIBRATIONS = ((0.0, 0.0, 0.0), (0.12, -0.08, 0.2), (-0.16, 0.05, -0.1))


def device_motion(acceleration=(0.2, -0.3, 0.4), angular_velocity=(0.004, -0.002, 0.03),
                  std=(0.01, 0.02, 0.03)):
  def measurement(values):
    return log.DeviceMotion.XYZMeasurement(x=values[0], y=values[1], z=values[2],
                                          xStd=std[0], yStd=std[1], zStd=std[2], valid=True)

  return log.DeviceMotion(
    timestamp=5_000_000_000,
    orientationNED=measurement((0.01, -0.03, 0.2)),
    velocityDevice=measurement((20.0, 0.2, -0.1)),
    accelerationDevice=measurement(acceleration),
    angularVelocityDevice=measurement(angular_velocity),
    inputsOK=True, sensorsOK=True, posenetOK=True,
  )


def calibrate(calibrator, rpy, status=log.ExtrinsicsCalibration.Status.calibrated):
  calibrator.feed_extrinsics_calibration(log.ExtrinsicsCalibration(rpyCalib=list(rpy), calStatus=status))


def full_pose_values(calibrator, motion):
  pose = calibrator.build_calibrated_pose(Pose.from_device_motion(motion))
  return pose.acceleration.x, pose.angular_velocity.yaw


def scalar_bits(values):
  return np.asarray(values, dtype=np.float64).tobytes()


@pytest.mark.parametrize('rpy', CALIBRATIONS)
@pytest.mark.parametrize('std', [(0.01, 0.02, 0.03), (0.0, -0.0, 0.0), (np.nan, np.inf, -np.inf)])
@pytest.mark.parametrize('values', [(0.2, -0.3, 0.4), (0.0, -0.0, 0.0),
                                   (np.nan, 0.01, 0.02), (np.inf, -np.inf, -0.0)])
def test_required_means_match_full_pose_bit_for_bit(rpy, std, values):
  calibrator = PoseCalibrator()
  calibrate(calibrator, rpy)
  motion = device_motion(values, values[::-1], std)
  with np.errstate(all='ignore'):
    expected = full_pose_values(calibrator, motion)
    actual = calibrator.build_calibrated_actuation(motion)
  assert scalar_bits(actual) == scalar_bits(expected)


def test_current_calibration_and_measurements_are_not_cached():
  generator = np.random.default_rng(20261104)
  calibrator = PoseCalibrator()
  for index in range(300):
    rpy = generator.uniform(-0.5, 0.5, 3)
    status = log.ExtrinsicsCalibration.Status.invalid if index % 2 else log.ExtrinsicsCalibration.Status.calibrated
    calibrate(calibrator, rpy, status)
    motion = device_motion(generator.uniform(-10.0, 10.0, 3), generator.uniform(-1.0, 1.0, 3))
    assert scalar_bits(calibrator.build_calibrated_actuation(motion)) == scalar_bits(full_pose_values(calibrator, motion))
    assert calibrator.calib_valid == (status == log.ExtrinsicsCalibration.Status.calibrated)


@pytest.mark.parametrize('shape', [(2, 2), (2, 3), (3, 2), (4, 4), (1, 3)])
def test_malformed_calibration_keeps_full_pose_failure(shape):
  calibrator = PoseCalibrator()
  calibrator.calib_from_device = np.ones(shape)
  motion = device_motion()
  with pytest.raises(Exception) as expected:
    full_pose_values(calibrator, motion)
  with pytest.raises(type(expected.value)) as actual:
    calibrator.build_calibrated_actuation(motion)
  assert str(actual.value) == str(expected.value)


def check_inputs(*, accel=0.0, speed=20.0, long_active=False, lat_active=False):
  CS = car.CarState(vEgo=speed, aEgo=accel, steeringPressed=False)
  sm = {'carControl': car.CarControl(longActive=long_active, latActive=lat_active),
        'vehicleParameters': log.VehicleParameters(roll=0.0)}
  return sm, CS


def test_longitudinal_detection_keeps_duration_noise_and_disable_boundaries():
  accel = ACCEL_MAX * 2 + 0.1
  sm, CS = check_inputs(accel=accel, long_active=True)
  checker = ExcessiveActuationCheck()
  for _ in range(MIN_EXCESSIVE_ACTUATION_COUNT):
    assert checker.update_from_values(sm, CS, accel, 0.0) is None
  assert checker.update_from_values(sm, CS, accel, 0.0) == ExcessiveActuationType.LONGITUDINAL
  CS.aEgo = accel - 2.0
  assert checker.update_from_values(sm, CS, accel, 0.0) is None
  assert checker._excessive_counter == 0
  CS.aEgo = accel
  sm['carControl'].longActive = False
  assert checker.update_from_values(sm, CS, accel, 0.0) is None
  assert checker._excessive_counter == 0


def test_lateral_detection_keeps_engagement_buffer_and_override_reset():
  sm, CS = check_inputs(lat_active=True)
  yaw_rate = (ISO_LATERAL_ACCEL * 2 + 0.1) / CS.vEgo
  checker = ExcessiveActuationCheck()
  for _ in range(MIN_LATERAL_ENGAGE_BUFFER + MIN_EXCESSIVE_ACTUATION_COUNT):
    assert checker.update_from_values(sm, CS, 0.0, yaw_rate) is None
  assert checker.update_from_values(sm, CS, 0.0, yaw_rate) == ExcessiveActuationType.LATERAL
  CS.steeringPressed = True
  assert checker.update_from_values(sm, CS, 0.0, yaw_rate) is None
  assert checker._engaged_counter == checker._excessive_counter == 0


def event_harness():
  daemon, CS = selfdrive_event_harness(new_event('controlsState').as_reader())
  daemon.pose_calibrator = PoseCalibrator()
  daemon.excessive_actuation_check = ExcessiveActuationCheck()
  daemon.sm.data['carControl'] = car.CarControl(longActive=True, latActive=True)
  return daemon, CS


def event_state(daemon):
  return {'events': daemon.events.names, 'sp_events': daemon.events_sp.names,
          'event_counters': daemon.events.event_counters, 'sp_event_counters': daemon.events_sp.event_counters,
          'excessive': daemon.excessive_actuation,
          'engaged_count': daemon.excessive_actuation_check._engaged_counter,
          'excessive_count': daemon.excessive_actuation_check._excessive_counter}


def test_calibration_without_a_motion_sample_does_not_start_the_check():
  daemon, CS = event_harness()
  daemon.sm.updated['extrinsicsCalibration'] = True
  daemon.sm.data['extrinsicsCalibration'] = log.ExtrinsicsCalibration(rpyCalib=list(CALIBRATIONS[1]), calStatus='calibrated')
  daemon.update_events(CS)
  assert daemon.calibrated_actuation is None
  assert daemon.excessive_actuation_check._engaged_counter == daemon.excessive_actuation_check._excessive_counter == 0


@pytest.mark.parametrize('invalid_field', ['inputsOK', 'posenetOK', 'sensorsOK',
                                         'accelerationDevice.valid', 'angularVelocityDevice.valid', 'transport_valid'])
def test_invalid_motion_health_preserves_real_events(invalid_field):
  expected, expected_CS = event_harness()
  actual, actual_CS = event_harness()
  expected.pose_calibrator.build_calibrated_actuation = lambda motion: full_pose_values(expected.pose_calibrator, motion)
  for daemon, CS in ((expected, expected_CS), (actual, actual_CS)):
    motion = device_motion()
    if invalid_field == 'transport_valid':
      daemon.sm.valid['deviceMotion'] = False
    elif '.' in invalid_field:
      parent, field = invalid_field.split('.')
      setattr(getattr(motion, parent), field, False)
    else:
      setattr(motion, invalid_field, False)
    daemon.sm.data['deviceMotion'] = motion
    daemon.sm.updated['deviceMotion'] = True
    daemon.update_events(CS)
  assert scalar_bits(actual.calibrated_actuation) == scalar_bits(expected.calibrated_actuation)
  assert event_state(actual) == event_state(expected)
  assert actual.sm.valid == expected.sm.valid


def test_selfdrived_real_events_preserve_sample_and_calibration_update_order():
  expected, expected_CS = event_harness()
  actual, actual_CS = event_harness()
  expected.pose_calibrator.build_calibrated_actuation = lambda motion: full_pose_values(expected.pose_calibrator, motion)
  alert_calls = {'expected': [], 'actual': []}
  saved_measurements = None
  for frame in range(320):
    # Calibration-only frames must not reinterpret a previously received sample.
    calibration_updated = frame in (0, 63, 154, 227)
    motion_updated = frame % 5 == 0
    acceleration = ACCEL_MAX * 2 + 0.5 if 80 <= frame < 135 else 0.2
    yaw_rate = 0.7 if 170 <= frame < 260 else 0.01
    for label, daemon, CS in (('expected', expected, expected_CS), ('actual', actual, actual_CS)):
      daemon.sm.frame = frame
      daemon.sm.updated = dict.fromkeys(daemon.sm.services, False)
      daemon.sm.updated['extrinsicsCalibration'] = calibration_updated
      daemon.sm.updated['deviceMotion'] = motion_updated
      daemon.sm.data['extrinsicsCalibration'] = log.ExtrinsicsCalibration(
        rpyCalib=list(CALIBRATIONS[(frame // 60) % len(CALIBRATIONS)]), calStatus='calibrated')
      daemon.sm.data['deviceMotion'] = device_motion((acceleration, -0.1, 0.2), (0.004, -0.002, yaw_rate))
      CS.aEgo = acceleration
      CS.steeringPressed = 200 <= frame < 210
      daemon.cp_not_car = 270 <= frame < 280
      with patch('openpilot.selfdrive.selfdrived.selfdrived.set_offroad_alert',
                 side_effect=lambda *args, _label=label, **kwargs: alert_calls[_label].append((args, kwargs))):
        daemon.update_events(CS)
    assert scalar_bits(actual.calibrated_actuation) == scalar_bits(expected.calibrated_actuation)
    assert event_state(actual) == event_state(expected)
    assert alert_calls['actual'] == alert_calls['expected']
    if not motion_updated:
      assert actual.calibrated_actuation is saved_measurements
    saved_measurements = actual.calibrated_actuation
  assert expected.excessive_actuation


def test_old_pose_update_interface_remains_available():
  sm, CS = check_inputs(accel=0.2, lat_active=True, long_active=True)
  calibrator = PoseCalibrator()
  pose = calibrator.build_calibrated_pose(Pose.from_device_motion(device_motion()))
  old_api, values_api = ExcessiveActuationCheck(), ExcessiveActuationCheck()
  for _ in range(200):
    assert old_api.update(sm, CS, pose) == values_api.update_from_values(sm, CS, pose.acceleration.x, pose.angular_velocity.yaw)
    assert vars(old_api) == vars(values_api)
