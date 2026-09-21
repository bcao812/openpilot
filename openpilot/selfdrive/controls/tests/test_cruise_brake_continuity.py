"""Coast braking continuity with recorded requests and real Honda CAN decoding.

The road fixture fixes the software request sequence, not delivered vehicle
deceleration. Planner eligibility and final arbitration have separate tests.
"""

import math

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.cruise_response import CruiseResponse, get_cruise_coast_accel


# 2026-09-21 route 0000007c--f6c8547c3d, bookmark a, segment 3.
# _logs/longitudinal-2026-09-22/a-timeline.csv: first CAN sample carrying each
# new plan from bookmark -13.683924422 s through -13.282287234 s. These are
# recorded Float32 planner_aTarget values, with CRUISE_COAST selected and no
# lead. Every value lies on the map's uniquely invertible lower flank:
# mapped = 2.2 * raw + 0.36. Raw diagnostics are unavailable in this recording;
# reconstruct the rounded plan values, not the full-precision raw measurement.
# Never infer raw from the noninvertible -0.19 plateau.
RECORDED_COAST_REQUESTS = (
  -0.19480881094932556,
  -0.1988581120967865,
  -0.19911827147006989,
  -0.21028007566928864,
  -0.21257537603378296,
  -0.21281875669956207,
  -0.19199737906455994,
  -0.20441383123397827,
  -0.19594596326351166,
)


def test_recorded_downhill_two_brake_pulses_become_one_continuous_request():
  from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import decoded_commands

  response = CruiseResponse()
  raw = [(mapped - 0.36) / 2.2 for mapped in RECORDED_COAST_REQUESTS]
  assert [get_cruise_coast_accel(value) for value in raw] == pytest.approx(RECORDED_COAST_REQUESTS, abs=1e-12, rel=0.0)
  baseline = decoded_commands(RECORDED_COAST_REQUESTS)
  shaped = [response.coast_command(value, eligible=True) for value in raw]
  commands = decoded_commands(shaped)

  assert [bool(command['BRAKE_REQUEST']) for command in baseline] == [False, False, False, True, True, True, False, True, False]
  assert [bool(command['BRAKE_REQUEST']) for command in commands] == [False, False, False, True, True, True, True, True, True]
  assert shaped[:3] == pytest.approx(RECORDED_COAST_REQUESTS[:3], abs=1e-12, rel=0.0)
  assert all(command <= recorded + 1e-12 for command, recorded in zip(shaped[3:], RECORDED_COAST_REQUESTS[3:], strict=True))
  assert all(command['GAS_COMMAND'] <= 0 for command in commands[3:])
  assert all(command['BRAKE_LIGHTS'] for command in commands[3:])


def test_first_request_just_beyond_braking_threshold_applies_without_delay():
  from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import decoded_commands

  response = CruiseResponse()
  raw = (-0.200001 - 0.36) / 2.2
  assert get_cruise_coast_accel(raw) == pytest.approx(-0.200001, abs=1e-12, rel=0.0)
  command = response.coast_command(raw, eligible=True)
  assert command == -0.21
  # Check the brake bit after Float32 transport and actual CAN packing, not
  # the encoded ACCEL_COMMAND rounded to hundredths.
  assert decoded_commands([command])[0]['BRAKE_REQUEST']


def test_stronger_braking_is_immediate_and_hold_does_not_remember_the_strongest_value():
  response = CruiseResponse()
  outputs = [response.coast_command(raw, eligible=True) for raw in (-0.26, -0.23, -0.29, -0.27, -0.23)]
  assert outputs == pytest.approx([-0.212, -0.21, -0.278, -0.234, -0.21], abs=1e-12, rel=0.0)


def test_release_needs_raw_recovery_strictly_above_the_boundary_then_can_arm_again():
  response = CruiseResponse()
  assert response.coast_command(-0.26, eligible=True) < -0.2
  for raw in (-0.23, -0.21, -0.2, np.nextafter(-0.2, -math.inf)):
    assert response.coast_command(raw, eligible=True) == -0.21
  assert response.coast_command(np.nextafter(-0.2, math.inf), eligible=True) == -0.19
  assert response.coast_command(-0.23, eligible=True) == -0.19
  assert response.coast_command(-0.26, eligible=True) == pytest.approx(-0.212, abs=1e-12, rel=0.0)


@pytest.mark.parametrize('raw', [-1.2, -0.26, -0.23, 0.2])
def test_ineligible_frame_preserves_raw_request_and_retires_hold(raw):
  response = CruiseResponse()
  assert response.coast_command(-0.26, eligible=True) < -0.2
  assert response.coast_command(-0.23, eligible=True) == -0.21
  assert response.coast_command(raw, eligible=False) == raw
  assert response.coast_command(-0.23, eligible=True) == -0.19


@pytest.mark.parametrize('clear', [True, False], ids=['clear', 'reset'])
def test_clear_and_reset_retire_hold_before_the_next_eligible_frame(clear):
  response = CruiseResponse(init_a=-0.26)
  assert response.coast_command(-0.26, eligible=True) < -0.2
  assert response.coast_command(-0.23, eligible=True) == -0.21
  if clear:
    response.clear()
    response.clear()
    assert response.coast_command(-0.23, eligible=True) == -0.19
  raw = response.update(24.77, 25.0, 1.0, measured_accel=-0.23, reset=not clear)
  assert raw == pytest.approx(-0.23, abs=1e-12, rel=0.0)
  assert response.coast_command(raw, eligible=True) == -0.19


@pytest.mark.parametrize('raw', [math.nan, math.inf, -math.inf])
def test_nonfinite_input_cannot_leave_a_brake_hold_for_recovery(raw):
  response = CruiseResponse()
  response.coast_command(-0.26, eligible=True)
  result = response.coast_command(raw, eligible=True)
  assert math.isnan(result) if math.isnan(raw) else result == raw
  assert response.coast_command(-0.23, eligible=True) == -0.19


def test_command_shaping_does_not_feed_back_into_cruise_jerk_history():
  response = CruiseResponse(init_a=-0.26)
  targets = [24.74, 24.77, 24.6, 25.0, 25.0, 25.0, 25.0]
  raw, shaped = [], []
  for target in targets:
    raw.append(response.update(target, 25.0, 1.0, measured_accel=0.0))
    shaped.append(response.coast_command(raw[-1], eligible=-0.3 < raw[-1] < -0.1))
  # At 25 m/s the existing jerk history advances by at most 0.04 per step.
  # In particular, the -0.21 held output must not replace the prior -0.23 raw
  # response when the next target asks for stronger braking.
  assert raw == pytest.approx([-0.26, -0.23, -0.27, -0.23, -0.19, -0.15, -0.11], abs=1e-12, rel=0.0)
  assert shaped == pytest.approx([-0.212, -0.21, -0.234, -0.21, -0.19, -0.19, -0.118], abs=1e-12, rel=0.0)


def test_unarmed_normal_coast_oscillation_does_not_acquire_brake_authority():
  from openpilot.selfdrive.controls.tests.longitudinal_control_test_support import decoded_commands

  response = CruiseResponse()
  raw = [-0.254, -0.25, -0.2, -0.15, -0.251] * 20
  shaped = [response.coast_command(value, eligible=True) for value in raw]
  assert shaped == [get_cruise_coast_accel(value) for value in raw]
  commands = decoded_commands(shaped)
  assert all(not command['BRAKE_REQUEST'] and not command['BRAKE_LIGHTS'] for command in commands)
