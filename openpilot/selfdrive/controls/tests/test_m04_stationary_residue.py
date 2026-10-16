"""M04's exact stopped sample tests compatibility, not a vehicle counterfactual."""

import json
from pathlib import Path

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.lead_launch import STANDSTILL_RESIDUE
from openpilot.selfdrive.controls.lib.model_reference import normalize_standstill_speed, stationary_lead_departure_compatible
from openpilot.selfdrive.controls.tests.test_typed_model_reference import MODEL_TIMES, evidence


FIXTURE = json.loads((Path(__file__).with_name('fixtures') / 'm04_stationary_residue.json').read_text())
RECORDED_SPEED = FIXTURE['car']['vEgo']


def recorded_reference():
  return evidence(FIXTURE['model']['velocity']['x'], RECORDED_SPEED,
                  model_ns=FIXTURE['input_mono_ns']['modelV2'], anchor_ns=FIXTURE['input_mono_ns']['carState'],
                  now_ns=FIXTURE['decision_mono_ns'])


def test_exact_m04_stopped_residue_preserves_raw_model_but_allows_compatibility():
  reference = recorded_reference()
  before = reference.raw_velocity.copy(), reference.events, reference.raw_motion
  assert RECORDED_SPEED < -np.finfo(float).eps
  assert FIXTURE['car']['standstill'] and FIXTURE['car']['vEgoRaw'] == 0.0
  assert not FIXTURE['car']['gasPressed'] and not FIXTURE['car']['brakePressed']
  assert normalize_standstill_speed(RECORDED_SPEED, actual_standstill=True, raw_speed=0.0) == 0.0
  assert stationary_lead_departure_compatible(reference, actual_standstill=True, measured_speed=RECORDED_SPEED, raw_speed=0.0)
  assert reference.unknown.reason == 'backward_then_forward_is_ambiguous'
  assert reference.moving is None and not reference.allows_departure and not reference.allows_external_departure
  np.testing.assert_array_equal(reference.raw_velocity, before[0])
  np.testing.assert_array_equal(FIXTURE['model']['velocity']['t'], MODEL_TIMES)
  assert (reference.events, reference.raw_motion) == before[1:]


@pytest.mark.parametrize('standstill,speed,raw_speed', [
  pytest.param(False, RECORDED_SPEED, 0.0, id='not-stopped'),
  pytest.param(True, RECORDED_SPEED, -0.01, id='negative-wheel-speed'),
  pytest.param(True, 0.0, -0.01, id='rollback-with-zero-filtered-speed'),
  pytest.param(True, -0.01, 0.0, id='material-negative-estimate'),
  pytest.param(True, np.nextafter(-STANDSTILL_RESIDUE, -np.inf), 0.0, id='outside-residue-bound'),
  pytest.param(True, RECORDED_SPEED, 0.01, id='moving-wheels'),
  pytest.param(True, RECORDED_SPEED, np.nan, id='invalid-wheel-speed'),
  pytest.param(True, RECORDED_SPEED, None, id='missing-wheel-speed'),
])
def test_m04_compatibility_does_not_hide_rollback_or_missing_stationary_evidence(standstill, speed, raw_speed):
  reference = recorded_reference()
  assert not stationary_lead_departure_compatible(reference, actual_standstill=standstill, measured_speed=speed, raw_speed=raw_speed)


@pytest.mark.parametrize('speed', [-STANDSTILL_RESIDUE, -np.finfo(float).eps, RECORDED_SPEED])
def test_residue_normalization_requires_both_stationary_signals(speed):
  assert normalize_standstill_speed(speed, actual_standstill=True, raw_speed=0.0) == 0.0
  assert normalize_standstill_speed(speed, actual_standstill=False, raw_speed=0.0) == speed
  assert normalize_standstill_speed(speed, actual_standstill=True, raw_speed=-0.01) == speed
