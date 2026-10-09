"""Stationary lead compatibility preserves the model's original UNKNOWN veto."""

from dataclasses import replace

import numpy as np
import pytest

from openpilot.selfdrive.controls.lib.drive_helpers import STOP_RELEASE_ACCEL
from openpilot.selfdrive.controls.lib.model_reference import ModelReferenceAdapter, stationary_lead_departure_compatible
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionPhase, MotionReferenceKind
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import consume, make_handover_state, make_planner_owner
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_typed_model_reference import MODEL_TIMES, SOURCE_NS, evidence


# Exact modelV2 velocity from 00000078--dce198d50c--19 at t=17.306390931 s.
# Recorded source: 7eb508a35d35b570d2be3e6c18a04524b186e07e; rlog.zst SHA256:
# c859b9e68aa980862ef341c91b96da9ac063a3ca3d9752047d5e547881451a64.
# This is a source snapshot, not evidence of a counterfactual vehicle response.
LOGGED_DEPARTURE_VELOCITY = np.array([
  -0.01965908706188202, -0.02170606516301632, -0.01894403249025345, -0.01730375364422798,
  -0.021943993866443634, -0.013184843584895134, -0.017475316300988197, -0.007635866291821003,
  -0.005044243764132261, 0.0031262561678886414, 0.013645940460264683, 0.03661305829882622,
  0.07766693830490112, 0.11473635584115982, 0.18145209550857544, 0.26576000452041626,
  0.3761044144630432, 0.5236693024635315, 0.6993162035942078, 0.902604341506958,
  1.1423208713531494, 1.4168977737426758, 1.7094587087631226, 2.038445234298706,
  2.37595796585083, 2.7425904273986816, 3.1220319271087646, 3.5168251991271973,
  3.8986976146698, 4.296391010284424, 4.68735408782959, 5.077934265136719, 5.465034008026123,
])

# Same recording, t=15.563145359 s: an initial negative prefix hides another
# negative sample after the first positive sample. It must remain a veto.
LOGGED_LATER_NEGATIVE_VELOCITY = np.array([
  -0.0052227722480893135, -0.0058540054596960545, -0.0041773212142288685, -0.002481300150975585,
  -0.006464497651904821, 0.0025950996205210686, -0.003207070752978325, 0.003618983319029212,
  0.0034584859386086464, 0.00681688729673624, 0.006193415727466345, 0.016102654859423637,
  0.0332857221364975, 0.042748406529426575, 0.07231606543064117, 0.11154089868068695,
  0.16440734267234802, 0.2359343320131302, 0.32460370659828186, 0.43024328351020813,
  0.5527269840240479, 0.700212299823761, 0.8621845841407776, 1.0505622625350952,
  1.2457128763198853, 1.4676156044006348, 1.7047470808029175, 1.9544464349746704,
  2.198795795440674, 2.462918758392334, 2.7238759994506836, 3.002472162246704, 3.2927019596099854,
])


def compatible(reference, *, standstill=True, speed=0.0):
  return stationary_lead_departure_compatible(reference, actual_standstill=standstill, measured_speed=speed)


def changed_sample(index, value):
  raw = LOGGED_DEPARTURE_VELOCITY.copy()
  raw[index] = value
  return raw


@pytest.mark.parametrize('speed', [
  -0.0, 0.05, -np.finfo(float).eps,
  # Exact stopped vEgo residues from route 00000091--cfe14c017c, segment 29,
  # 22:19:11.864--22:19:12.063 +08:00 on 2026-10-09, source 9e3a811dd8a8.
  # Only measured speed varies here; the raw model fixture above is separate.
  -5.914764288085437e-24, -1.9180846832417676e-23, -1.8333419315871246e-23,
  -1.3595017425856318e-23, -7.237740429695797e-24,
])
@pytest.mark.parametrize('model_ns,anchor_ns,now_ns', [
  (1184580849025, 1184581734182, 1184596901421),  # exact recorded publication/anchor/consumption proxies
  (SOURCE_NS + 5_000_000, SOURCE_NS, SOURCE_NS + 10_000_000),
])
def test_recorded_departure_is_compatible_without_reclassifying_model_or_granting_departure(speed, model_ns, anchor_ns, now_ns):
  reference = evidence(LOGGED_DEPARTURE_VELOCITY, speed, model_ns=model_ns, anchor_ns=anchor_ns, now_ns=now_ns)
  before = reference.raw_velocity.copy(), reference.events, reference.raw_motion
  assert compatible(reference, speed=speed)
  assert reference.unknown is not None and reference.raw_motion.kind == 'unknown'
  assert reference.moving is None
  assert not reference.allows_departure and not reference.allows_external_departure
  np.testing.assert_array_equal(reference.raw_velocity, before[0])
  assert (reference.events, reference.raw_motion) == before[1:]


@pytest.mark.parametrize('standstill,speed', [
  (False, 0.0), (True, 0.050001), (True, 0.2), (True, -0.001),
  (True, np.nextafter(-np.finfo(float).eps, -np.inf)), (False, -1e-23),
  (None, -1e-23), (True, None), (True, np.nan), (True, np.inf), (True, -np.inf), (True, True), (1, 0.0),
])
def test_measured_movement_or_invalid_standstill_cannot_use_stationary_compatibility(standstill, speed):
  reference = evidence(LOGGED_DEPARTURE_VELOCITY, 0.0)
  assert not compatible(reference, standstill=standstill, speed=speed)


@pytest.mark.parametrize('raw', [
  pytest.param(np.full(len(MODEL_TIMES), -0.01), id='all-negative'),
  pytest.param(np.zeros(len(MODEL_TIMES)), id='all-zero'),
  pytest.param(changed_sample(0, -0.050001), id='deep-initial-negative'),
  pytest.param(changed_sample(4, -0.050001), id='deep-prefix-negative'),
  pytest.param(np.where(MODEL_TIMES < 1.0, -0.01, 0.6 * (MODEL_TIMES - 1.0)), id='long-prefix'),
  pytest.param(np.where(MODEL_TIMES < 0.79, -0.04, 0.6 * (MODEL_TIMES - 0.79)), id='excess-negative-distance'),
  pytest.param(changed_sample(20, 0.0), id='later-zero'),
  pytest.param(changed_sample(32, -0.001), id='terminal-negative'),
  pytest.param(changed_sample(5, 0.0), id='zero-inside-prefix'),
  pytest.param(np.where(MODEL_TIMES > 2.5, LOGGED_DEPARTURE_VELOCITY,
                        np.minimum(LOGGED_DEPARTURE_VELOCITY, 0.01)), id='no-near-term-progress'),
  pytest.param(LOGGED_LATER_NEGATIVE_VELOCITY, id='recorded-later-negative'),
  pytest.param(changed_sample(16, np.nan), id='nonfinite'),
  pytest.param(LOGGED_DEPARTURE_VELOCITY[:-1], id='incomplete-grid'),
  pytest.param(LOGGED_DEPARTURE_VELOCITY.reshape(1, -1), id='non-vector'),
])
@pytest.mark.parametrize('later_publication', [False, True])
@pytest.mark.parametrize('measured_speed', [0.0, -1.9180846832417676e-23])
def test_prefix_compatibility_keeps_raw_risk_and_incomplete_inputs_blocked(raw, later_publication, measured_speed):
  reference = evidence(raw, 0.0, model_ns=SOURCE_NS + int(later_publication) * 5_000_000,
                       anchor_ns=SOURCE_NS, now_ns=SOURCE_NS + 10_000_000)
  assert not compatible(reference, speed=measured_speed)
  assert not reference.allows_departure and not reference.allows_external_departure


def test_piecewise_linear_zero_crossing_limits_actual_negative_displacement():
  # At 0.625 s the velocity is -0.03 m/s; at 0.791015625 s it is +0.3 m/s.
  # The zero crossing gives 0.01898 m negative travel (< 0.02 m). Clipping
  # sampled velocities before integrating would incorrectly exceed 0.02 m.
  raw = np.where(MODEL_TIMES <= 0.625, -0.03, 0.3 + 0.6 * (MODEL_TIMES - 0.791015625))
  assert compatible(evidence(raw, 0.0))


@pytest.mark.parametrize('model_ns,anchor_ns,now_ns,reason', [
  (None, SOURCE_NS, SOURCE_NS, 'missing_publication_proxy'),
  (True, SOURCE_NS, SOURCE_NS, 'missing_publication_proxy'),
  (SOURCE_NS + 1, SOURCE_NS, SOURCE_NS, 'future_publication_proxy'),
  (SOURCE_NS, SOURCE_NS + 1, SOURCE_NS, 'future_publication_proxy'),
  (SOURCE_NS, SOURCE_NS, SOURCE_NS + 250_000_001, 'expired_model_publication'),
  (SOURCE_NS, SOURCE_NS - 250_000_001, SOURCE_NS, 'model_proxy_after_initial_state'),
])
def test_source_time_failures_are_not_prefix_exceptions(model_ns, anchor_ns, now_ns, reason):
  reference = evidence(LOGGED_DEPARTURE_VELOCITY, 0.0, model_ns=model_ns, anchor_ns=anchor_ns, now_ns=now_ns)
  assert reference.unknown.reason == reason
  assert not compatible(reference)


@pytest.mark.parametrize('source_fields', [
  {'model_mono_time_ns': SOURCE_NS + 1},
  {'anchor_mono_time_ns': SOURCE_NS + 1},
  {'consumed_mono_time_ns': SOURCE_NS + 250_000_001},
  {'proxy_age_s': 0.01},
  {'epoch_basis': 'physical_prediction'},
  {'physical_prediction_epoch_ns': SOURCE_NS},
])
def test_retained_unknown_reason_does_not_make_a_broken_source_usable(source_fields):
  reference = evidence(LOGGED_DEPARTURE_VELOCITY, 0.0)
  malformed = replace(reference, source=replace(reference.source, **source_fields))
  assert not compatible(malformed)


def test_adapter_identity_failure_cannot_be_repaired_by_a_compatible_shape():
  adapter = ModelReferenceAdapter()
  reference = evidence(LOGGED_DEPARTURE_VELOCITY, 0.0, adapter=adapter)
  assert compatible(reference)
  mutated = evidence(changed_sample(32, 6.0), 0.0, adapter=adapter)
  regressed = evidence(LOGGED_DEPARTURE_VELOCITY, 0.0, adapter=adapter, model_ns=SOURCE_NS - 1)
  assert mutated.unknown.reason == 'model_mutated_with_same_identity'
  assert regressed.unknown.reason == 'model_publication_regressed'
  assert not compatible(mutated) and not compatible(regressed)


@pytest.mark.parametrize('raw', [
  changed_sample(0, 0.0),
  np.where(MODEL_TIMES < 1.0, 0.0, 0.6 * (MODEL_TIMES - 1.0)),
  np.where(MODEL_TIMES == MODEL_TIMES[16], 0.0, 0.2 + 0.6 * MODEL_TIMES),
  0.6 * MODEL_TIMES,
])
def test_other_model_semantics_cannot_use_negative_prefix_compatibility(raw):
  assert not compatible(evidence(raw, 0.0))
  assert not compatible(None)


def test_existing_moving_negative_prefix_veto_is_preserved():
  reference = evidence(-0.015 + 0.6 * MODEL_TIMES, 0.2, model_ns=SOURCE_NS + 5_000_000,
                       anchor_ns=SOURCE_NS, now_ns=SOURCE_NS + 10_000_000)
  assert reference.unknown.reason == 'model_proxy_after_initial_state'
  assert not compatible(reference, standstill=False, speed=0.2)
  assert not compatible(reference, standstill=True, speed=0.2)
  assert not reference.allows_departure and not reference.allows_external_departure


def held_recorded_departure_case():
  """Create a real held owner, then supply recorded geometry and raw velocity.

  The shared transport publishes real wire identities at 20 Hz and consumes
  each at five 100 Hz ticks. Vehicle speed remains an open-loop input.
  """
  planner, owner = make_planner_owner(speed=0.0)
  state = make_handover_state(speed=0.0, target=10.0, experimental=True)
  state['modelV2'].action.shouldStop = True
  state['modelV2'].action.desiredAcceleration = -0.2
  for frame in range(20):
    state['controlsState'].longControlState = owner.long_control_state
    plan, outputs = consume(planner, owner, state, frame)
    assert plan.motionIntent.valid and plan.solverStatus == 0
    assert not owner.motion_degraded and owner.resolved_should_stop
    assert max(outputs) <= 0.0
  assert owner.motion_phase == MotionPhase.HOLD
  raw = LOGGED_DEPARTURE_VELOCITY
  state['modelV2'].position.x = (np.maximum(raw, 0.0) * MODEL_TIMES).tolist()
  state['modelV2'].velocity.x = raw.tolist()
  state['modelV2'].acceleration.x = np.gradient(raw, MODEL_TIMES).tolist()
  state['modelV2'].action.desiredAcceleration = 0.22320786118507385
  state['modelV2'].action.shouldStop = False
  for lead, gap, speed, probability in (
    (state['radarState'].leadOne, 19.578693389892578, 1.6142926216125488, 0.9775195717811584),
    (state['radarState'].leadTwo, 19.55013084411621, 1.6177656650543213, 0.9617546796798706),
  ):
    lead.present = True
    lead.dRel, lead.vLead, lead.vRel = gap, speed, speed
    lead.aLeadK, lead.aLeadTau, lead.yRel, lead.modelProb = 0.12, 1.5, 0.135, probability
  return planner, owner, state


@pytest.mark.parametrize('measured_speed', [0.0, -1.9180846832417676e-23])
def test_recorded_shape_with_departing_lead_releases_actual_held_owner_after_confirmation(measured_speed):
  planner, owner, state = held_recorded_departure_case()
  state['carState'].vEgo = measured_speed
  measured_input = state['carState'].as_reader().vEgo
  outputs = []
  released = False
  confirmed_while_held = False
  strong_target_seen = False
  for frame in range(20, 44):
    state['controlsState'].longControlState = owner.long_control_state
    plan, current = consume(planner, owner, state, frame)
    outputs.extend(current)
    reference = planner.reference_planner.model_evidence
    assert plan.motionIntent.valid and plan.solverStatus == 0
    assert plan.motionIntent.referenceKind == MotionReferenceKind.UNKNOWN
    assert not reference.allows_departure and not reference.allows_external_departure
    assert not owner.motion_degraded
    if plan.motionIntent.releaseAllowed:
      strong_target_seen |= plan.aTarget >= STOP_RELEASE_ACCEL
      if owner.resolved_should_stop:
        confirmed_while_held |= owner.motion_supervisor.release_lead_t > 0.0
        assert max(current) <= 0.0
      else:
        assert plan.aTarget > 0.0
        released = True
  assert released and (confirmed_while_held or strong_target_seen)
  assert owner.motion_phase in (MotionPhase.RELEASE, MotionPhase.TRACK)
  assert not owner.resolved_should_stop and outputs[-1] > 0.0
  assert not state['carState'].gasPressed and state['carState'].standstill
  assert state['carState'].vEgo == measured_input


@pytest.mark.parametrize('condition', ['moving', 'rollback', 'later_zero', 'model_stop', 'no_lead', 'uncertain_lead'])
def test_stationary_prefix_cannot_release_owner_without_all_other_departure_guards(condition):
  planner, owner, state = held_recorded_departure_case()
  state['carState'].vEgo = -1.9180846832417676e-23
  if condition == 'moving':
    state['carState'].vEgo = 0.2
    state['carState'].standstill = False
    for lead in (state['radarState'].leadOne, state['radarState'].leadTwo):
      lead.vRel = lead.vLead - state['carState'].vEgo
  elif condition == 'rollback':
    state['carState'].vEgo = -0.001
  elif condition == 'later_zero':
    raw = changed_sample(20, 0.0)
    state['modelV2'].velocity.x = raw.tolist()
    state['modelV2'].position.x = (np.maximum(raw, 0.0) * MODEL_TIMES).tolist()
    state['modelV2'].acceleration.x = np.gradient(raw, MODEL_TIMES).tolist()
  elif condition == 'model_stop':
    state['modelV2'].action.shouldStop = True
  elif condition == 'no_lead':
    state['radarState'].leadOne.present = state['radarState'].leadTwo.present = False
  elif condition == 'uncertain_lead':
    state['radarState'].leadTwo.modelProb = 0.94
  for frame in range(20, 44):
    state['controlsState'].longControlState = owner.long_control_state
    plan, outputs = consume(planner, owner, state, frame)
    assert plan.motionIntent.valid and plan.solverStatus == 0
    assert not plan.motionIntent.releaseAllowed
    assert not owner.motion_degraded and owner.resolved_should_stop
    assert owner.motion_phase in (MotionPhase.HOLD, MotionPhase.APPROACH)
    assert max(outputs) <= 0.0
