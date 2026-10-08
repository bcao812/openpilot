from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from openpilot.selfdrive.controls.lib.object_influence import (FOLLOW_STRENGTH, accepted_mitigation_influences, accepted_object_influences,
                                                              influence_identity, mitigation_plan_object_influences, plan_object_influences)


NOW = 2_000_000_000


def lead(track=1, *, source='radar', model=NOW, index=-1, session='session-a'):
  return NS(present=True, evidence=NS(version=1, source=source, sessionId=session if track else '', trackId=track,
                                     generation=1, modelMonoTime=model, visionIndex=index))


def radar(*objects):
  return NS(mdMonoTime=NOW, fusionActive=True, fusionDegraded=False, radarObjects=objects,
            leadOne=objects[0] if objects else NS(present=False), leadTwo=NS(present=False))


def plan_records(state, identities=(), **changes):
  kwargs = {'active_identities': identities, 'now_ns': NOW, 'permitted': True, 'fusion_admitted': True, 'reference_accepted': True,
                'independent_stop': False, 'object_stop': False, 'held_identity': None, 'release_identity': None, 'standstill': False,
                'target_accel': -1.0, 'accel_min': -3.5, 'accel_max': 2.0}
  kwargs.update(changes)
  return plan_object_influences(state, **kwargs)


def record(kind='braking', track=1, strength=0.3):
  return {'version': 1, 'sessionId': 'session-a', 'trackId': track, 'modelMonoTime': 0, 'visionIndex': -1, 'kind': kind, 'strength': strength}


def receipt(records, *, phase='track', resolved_stop=False, constrained=False, **changes):
  plan = NS(objectInfluences=records, modelMonoTime=NOW, solverStatus=0, aTarget=-1.0,
            motionIntent=NS(valid=True, ready=True, releaseAllowed=False))
  owner = NS(unified=True, motion_input_constrained=constrained, motion_phase=NS(text=phase), resolved_should_stop=resolved_stop,
             long_control_state='stopping' if resolved_stop else 'pid', motion_supervisor=NS(rest_braking_accel_limit=None))
  car = NS(gasPressed=False, brakePressed=False)
  kwargs = {'long_active': True, 'plan_valid': True, 'plan_ns': NOW, 'now_ns': NOW, 'output_accel': -1.0}
  for key, value in changes.items():
    if key.startswith('plan_') and key not in ('plan_ns', 'plan_valid'):
      setattr(plan, key[5:], value)
    elif key.startswith('owner_') and key != 'owner_tracking':
      setattr(owner, key[6:], value)
    elif key.startswith('car_'):
      setattr(car, key[4:], value)
    else:
      kwargs[key] = value
  kwargs.setdefault('owner_tracking', owner.long_control_state == 'pid')
  return accepted_object_influences(plan, owner, car, **kwargs)


def test_only_active_accepted_objects_get_records_and_joint_constraints_are_retained():
  state = radar(lead(1), lead(2, source='fused', index=1), lead(3), lead(0, source='vision', index=0))
  identities = (('session-a', 2), ('vision', NOW, 0))
  records = plan_records(state, identities)
  assert [influence_identity(value) for value in records] == list(identities)
  assert records[0]['visionIndex'] == 1 and records[0]['modelMonoTime'] == NOW
  assert all(value['kind'] == 'braking' and value['strength'] > FOLLOW_STRENGTH for value in records)
  assert not plan_records(state)  # A present/brake-qualified point alone is not attribution.


@pytest.mark.parametrize('changes', [{'permitted': False}, {'reference_accepted': False}, {'independent_stop': True},
                                     {'target_accel': float('nan')}, {'target_accel': float('inf')}, {'accel_max': 0.0}])
def test_unadopted_or_invalid_final_plan_has_no_labels(changes):
  assert not plan_records(radar(lead()), (('session-a', 1),), **changes)


@pytest.mark.parametrize('kind', ['braking', 'follow', 'hold', 'release', 'caution'])
def test_history_loss_flags_do_not_hide_a_valid_adopted_object_influence(kind):
  item = lead(7)
  state = radar(item)
  identity = ('session-a', 7)
  kwargs = {'identities': (identity,)}
  if kind == 'follow':
    kwargs['target_accel'] = .2
  elif kind == 'hold':
    kwargs.update(held_identity=identity, reference_accepted=False, standstill=True)
  elif kind == 'release':
    kwargs.update(held_identity=identity, release_identity=identity, target_accel=.5)
  elif kind == 'caution':
    item.evidence.cautionQualified, item.evidence.pathState, item.evidence.coasting = True, 'edge', False
    kwargs.update(identities=(), caution_identities=(identity,), target_accel=0.0, reference_accepted=False)
  baseline = plan_records(state, **kwargs)
  assert len(baseline) == 1 and baseline[0]['kind'] == kind
  state.fusionDegraded = state.fusionRiskUnresolved = True
  assert plan_records(state, **kwargs) == baseline
  assert not plan_records(state, **{**kwargs, 'independent_stop': True})
  assert not plan_records(state, **{**kwargs, 'permitted': False})


def test_generation_and_slot_changes_do_not_change_physical_receipt_identity():
  item = lead(7, source='fused', index=0)
  first = plan_records(radar(item), (('session-a', 7),))
  item.evidence.generation += 20
  item.evidence.visionIndex = 1
  second = plan_records(radar(item), (('session-a', 7),))
  assert influence_identity(first[0]) == influence_identity(second[0]) == ('session-a', 7)
  assert second[0]['visionIndex'] == 1
  item.evidence.sessionId = 'restarted'
  assert not plan_records(radar(item), (('session-a', 7),))


def test_visual_attribution_requires_exact_source_frame_and_original_slot():
  state = NS(mdMonoTime=NOW, fusionActive=False, leadOne=NS(present=False), leadTwo=NS(present=True))
  records = plan_records(state, (('vision', NOW, 1),), fusion_admitted=False)
  assert len(records) == 1 and records[0]['visionIndex'] == 1
  assert not plan_records(state, (('vision', NOW + 1, 1),), fusion_admitted=False)
  assert not plan_records(state, (('vision', NOW, 0),), fusion_admitted=False)


def test_scope_exit_uses_the_untouched_visual_fallback_identity():
  state = radar(lead(1, source='fused', index=0))
  state.visionLeadOne, state.visionLeadTwo = NS(present=True), NS(present=False)
  records = plan_records(state, (('vision', NOW, 0), ('session-a', 1)), fusion_admitted=False)
  assert len(records) == 1 and influence_identity(records[0]) == ('vision', NOW, 0)


def test_held_barrier_and_departure_have_separate_owner_acknowledgements():
  state = radar(lead(1))
  held = plan_records(state, reference_accepted=False, held_identity=('session-a', 1), standstill=True)
  assert held[0]['kind'] == 'hold' and held[0]['strength'] == 1.0
  assert receipt(held, phase='hold', resolved_stop=True)
  assert not receipt(held)
  departing = plan_records(state, held_identity=('session-a', 1), release_identity=('session-a', 1), target_accel=0.5)
  assert departing[0]['kind'] == 'release'
  assert not receipt(departing, phase='hold', resolved_stop=True)
  release_intent = NS(valid=True, ready=True, releaseAllowed=True)
  assert receipt(departing, phase='release', plan_aTarget=0.5, plan_motionIntent=release_intent)
  # An independent model stop is never assigned to the same visible car.
  assert not plan_records(state, held_identity=('session-a', 1), independent_stop=True)


def test_following_has_visible_low_strength_and_braking_request_scales_monotonically():
  strengths = []
  for acceleration in (0.0, -0.2, -1.5, -3.5):
    records = plan_records(radar(lead()), (('session-a', 1),), target_accel=acceleration)
    strengths.append(records[0]['strength'])
    assert records[0]['kind'] == ('follow' if acceleration == 0.0 else 'braking')
  assert strengths[0] == FOLLOW_STRENGTH
  assert strengths == sorted(strengths)
  assert strengths[-1] <= 1.0


@pytest.mark.parametrize('changes', [{'long_active': False}, {'plan_valid': False}, {'constrained': True},
                                     {'car_gasPressed': True}, {'car_brakePressed': True}, {'owner_unified': False},
                                     {'now_ns': NOW + 250_000_001}, {'plan_ns': NOW + 1}, {'plan_solverStatus': 1},
                                     {'independent_stop': True}, {'output_accel': float('nan')},
                                     {'plan_motionIntent': NS(valid=False, ready=True, releaseAllowed=False)},
                                     {'owner_motion_supervisor': NS(rest_braking_accel_limit=-1.5)}])
def test_owner_rejection_and_independent_stop_clear_receipts(changes):
  assert not receipt([record()], **changes)


def test_owner_phase_and_real_command_must_adopt_the_claimed_kind():
  assert receipt([record()])
  assert not receipt([record()], output_accel=0.1)
  assert not receipt([record('hold')])
  assert not receipt([record('follow')], owner_long_control_state='stopping')
  assert not receipt([record('follow')], phase='approach', resolved_stop=True)
  assert not receipt([record('release')])
  assert not receipt([record('follow')], phase='off')
  assert not receipt([record('follow')], owner_tracking=1)


@pytest.mark.parametrize('mutation', [{'version': 0}, {'version': True}, {'strength': float('nan')}, {'strength': 1.01},
                                      {'strength': 0.0}, {'sessionId': ''}, {'trackId': 0}, {'kind': 'unknown'}])
def test_malformed_records_cannot_turn_a_label_red(mutation):
  item = record()
  item.update(mutation)
  assert not receipt([item])


def test_receipt_is_bounded_deduplicated_and_observational():
  item = record()
  before = deepcopy(item)
  assert len(receipt([item, item])) == 1
  assert not receipt([record(track=i + 1) for i in range(35)])
  assert item == before
  state = radar(lead(1, source='fused', index=0))
  before_state = deepcopy(state)
  plan_records(state, (('session-a', 1),))
  assert state == before_state


def test_edge_cap_only_gets_light_caution_when_it_is_the_adopted_zero_ceiling():
  item = lead(9)
  item.evidence.cautionQualified = True
  item.evidence.pathState = 'edge'
  state = radar(item)
  identity = ('session-a', 9)
  records = plan_records(state, reference_accepted=False, caution_identities=(identity,), target_accel=0.0)
  assert len(records) == 1
  assert records[0]['kind'] == 'caution' and records[0]['strength'] == FOLLOW_STRENGTH
  assert receipt(records, plan_aTarget=0.0, output_accel=0.0)
  assert not plan_records(state, reference_accepted=False, target_accel=0.0)
  assert not plan_records(state, reference_accepted=False, caution_identities=(identity,), target_accel=-1.0)
  assert not plan_records(state, reference_accepted=False, caution_identities=(identity,), target_accel=0.0, independent_stop=True)
  # An edge identity cannot acquire hard braking merely by being placed in an
  # active-constraint list; its evidence only supports the separate cap.
  assert not plan_records(state, (identity,), target_accel=-1.0)


@pytest.mark.parametrize('changes', [{'cautionQualified': False}, {'pathState': 'unknown'}, {'coasting': True}])
def test_edge_display_cap_keeps_its_qualification_and_prediction_boundaries(changes):
  item = lead(9)
  item.evidence.cautionQualified, item.evidence.pathState = True, 'edge'
  for key, value in changes.items():
    setattr(item.evidence, key, value)
  assert not plan_records(radar(item), reference_accepted=False, caution_identities=(('session-a', 9),), target_accel=0.0)


def test_caution_receipt_cannot_claim_a_stop_release_or_strong_braking_effect():
  caution = record('caution', strength=FOLLOW_STRENGTH)
  assert not receipt([caution], phase='hold', resolved_stop=True, plan_aTarget=0.0, output_accel=-2.0)
  assert not receipt([caution], phase='release', plan_aTarget=0.0, output_accel=0.0)
  assert not receipt([caution], plan_aTarget=-1.0, output_accel=-1.0)
  assert not receipt([caution], plan_aTarget=0.0, output_accel=0.1)
  assert not receipt([record('caution', strength=0.9)], plan_aTarget=0.0, output_accel=0.0)


def mitigation_fixture():
  item = lead(3, source='fused', index=1)
  evidence = item.evidence
  evidence.estimateMonoTime = evidence.positionMonoTime = evidence.velocityMonoTime = NOW
  evidence.brakeQualified = evidence.fcwQualified = True
  evidence.coasting = False
  evidence.physicalConflictValid = True
  evidence.physicalConflictStart, evidence.physicalConflictEnd = 0.02, 0.10
  proof = NS(identity=('session-a', 3), source='fused', consumed_ns=NOW, reference_ns=NOW,
             estimate_ns=NOW, position_ns=NOW, velocity_ns=NOW, model_ns=NOW, braking_accel=-3.5,
             conflict_start_s=0.02, conflict_end_s=0.10, proof_time_s=0.08)
  kwargs = {'selected': True, 'now_ns': NOW, 'consumed_ns': NOW, 'reference_ns': NOW, 'target_accel': -3.5, 'braking_accel': -3.5, 'accel_min': -3.5}
  return radar(item), proof, kwargs


def mitigation_receipt(records, **changes):
  intent = NS(valid=False, ready=True, brakingValid=True, brakingAccel=-3.5, allowCreep=False, movingLead=False, releaseAllowed=False)
  plan = NS(objectInfluences=records, modelMonoTime=NOW, solverStatus=-6, aTarget=-3.5, motionIntent=intent)
  supervisor = NS(degraded_braking_accel=-3.5, _last_braking_plan_time_ns=NOW, _last_braking_model_time_ns=NOW)
  owner = NS(unified=True, motion_input_constrained=True, motion_phase=NS(text='track'), motion_supervisor=supervisor)
  car = NS(canValid=True, gasPressed=False, brakePressed=False)
  kwargs = {'long_active': True, 'braking_plan_valid': True, 'key_inputs_valid': True, 'key_input_times': (NOW,) * 3,
                'plan_ns': NOW, 'now_ns': NOW, 'output_accel': -3.5, 'accel_min': -3.5}
  for key, value in changes.items():
    if key.startswith('intent_'):
      setattr(intent, key[7:], value)
    elif key.startswith('supervisor_'):
      setattr(supervisor, key[11:], value)
    elif key.startswith('car_'):
      setattr(car, key[4:], value)
    elif key.startswith('plan_') and key != 'plan_ns':
      setattr(plan, key[5:], value)
    else:
      kwargs[key] = value
  return accepted_mitigation_influences(plan, owner, car, **kwargs)


def test_only_selected_current_physical_mitigation_mints_v2_and_preserves_fused_identity():
  state, proof, kwargs = mitigation_fixture()
  records = mitigation_plan_object_influences(state, (proof,), **kwargs)
  assert len(records) == 1
  assert records[0] == {'version': 2, 'sessionId': 'session-a', 'trackId': 3, 'modelMonoTime': NOW,
                            'visionIndex': 1, 'kind': 'braking', 'strength': 1.0}
  assert mitigation_receipt(records) == records
  assert not receipt(records)  # Normal v1 receipt never relabels failed plans.
  assert not mitigation_plan_object_influences(state, (), **kwargs)
  assert not mitigation_plan_object_influences(state, (proof,), **dict(kwargs, selected=False))
  assert not mitigation_plan_object_influences(state, (proof,), **dict(kwargs, target_accel=-0.5))


@pytest.mark.parametrize('field,value', [('consumed_ns', NOW - 1), ('reference_ns', NOW - 1), ('position_ns', NOW - 1),
                                         ('velocity_ns', NOW - 1), ('estimate_ns', NOW - 1), ('model_ns', NOW - 1),
                                         ('source', 'predicted'), ('identity', ('restarted', 3)), ('braking_accel', -0.5)])
def test_mitigation_proof_cannot_be_reused_from_another_tick_source_or_request(field, value):
  state, proof, kwargs = mitigation_fixture()
  setattr(proof, field, value)
  assert not mitigation_plan_object_influences(state, (proof,), **kwargs)


def test_mitigation_display_requires_fresh_measurements_and_not_coasting_or_caution():
  for case in ('old_position', 'coasting', 'caution'):
    state, proof, kwargs = mitigation_fixture()
    evidence = state.leadOne.evidence
    if case == 'old_position':
      evidence.positionMonoTime = proof.position_ns = NOW - 200_000_001
    elif case == 'coasting':
      evidence.coasting = True
    else:
      evidence.cautionQualified = True
      evidence.brakeQualified = evidence.fcwQualified = False
    assert not mitigation_plan_object_influences(state, (proof,), **kwargs)


@pytest.mark.parametrize('changes', [{'physicalConflictValid': False}, {'physicalConflictValid': 1},
                                     {'physicalConflictStart': float('nan')}, {'physicalConflictEnd': float('inf')},
                                     {'physicalConflictStart': -0.01}, {'physicalConflictStart': 0.10},
                                     {'physicalConflictEnd': 0.01}])
def test_mitigation_requires_an_explicit_valid_ordered_raw_physical_window(changes):
  state, proof, kwargs = mitigation_fixture()
  for field, value in changes.items():
    setattr(state.leadOne.evidence, field, value)
  assert not mitigation_plan_object_influences(state, (proof,), **kwargs)


def test_missing_physical_window_authority_does_not_inherit_from_old_object_fields():
  state, proof, kwargs = mitigation_fixture()
  del state.leadOne.evidence.physicalConflictValid
  assert not mitigation_plan_object_influences(state, (proof,), **kwargs)


@pytest.mark.parametrize('field,value', [('physicalConflictStart', 0.01), ('physicalConflictEnd', 0.11)])
def test_matching_identity_and_source_times_do_not_reuse_a_proof_after_its_raw_window_changes(field, value):
  state, proof, kwargs = mitigation_fixture()
  setattr(state.leadOne.evidence, field, value)
  assert not mitigation_plan_object_influences(state, (proof,), **kwargs)


def test_mitigation_source_binding_uses_full_physical_window_not_padded_or_search_bounds():
  state, proof, kwargs = mitigation_fixture()
  state.leadOne.evidence.physicalConflictEnd = proof.conflict_end_s = 2.5
  assert mitigation_plan_object_influences(state, (proof,), **kwargs)
  proof.conflict_end_s = 1.0  # A one-second search cutoff is not the source's full endpoint.
  assert not mitigation_plan_object_influences(state, (proof,), **kwargs)
  proof.conflict_start_s, proof.conflict_end_s = -0.33, 2.85  # Nor is the padded planning envelope.
  assert not mitigation_plan_object_influences(state, (proof,), **kwargs)


def test_physical_window_binding_preserves_reference_epoch_alignment_and_negative_aligned_start():
  state, proof, kwargs = mitigation_fixture()
  state.leadOne.evidence.estimateMonoTime = proof.estimate_ns = NOW - 100_000_000
  proof.reference_ns = kwargs['reference_ns'] = NOW - 50_000_000
  state.leadOne.evidence.physicalConflictEnd = 0.4
  offset = (proof.reference_ns - proof.estimate_ns) / 1e9
  proof.conflict_start_s = state.leadOne.evidence.physicalConflictStart - offset
  proof.conflict_end_s = state.leadOne.evidence.physicalConflictEnd - offset
  assert proof.conflict_start_s < 0.0
  assert mitigation_plan_object_influences(state, (proof,), **kwargs)
  proof.conflict_start_s = 0.0  # Dropping its pre-reference history changes the source window.
  assert not mitigation_plan_object_influences(state, (proof,), **kwargs)


@pytest.mark.parametrize('timestamp', [float('nan'), -0.1, 0.11])
def test_proof_time_must_belong_to_the_same_physical_window(timestamp):
  state, proof, kwargs = mitigation_fixture()
  proof.proof_time_s = timestamp
  assert not mitigation_plan_object_influences(state, (proof,), **kwargs)


@pytest.mark.parametrize('changes', [{'braking_plan_valid': False}, {'key_inputs_valid': False}, {'long_active': False},
                                     {'key_input_times': (NOW, NOW, NOW - 250_000_001)}, {'now_ns': NOW + 250_000_001},
                                     {'intent_valid': True}, {'intent_brakingValid': False}, {'intent_brakingAccel': -0.5},
                                     {'intent_releaseAllowed': True}, {'intent_allowCreep': True}, {'plan_solverStatus': 0},
                                     {'supervisor_degraded_braking_accel': -0.5}, {'supervisor__last_braking_plan_time_ns': NOW - 1},
                                     {'supervisor__last_braking_model_time_ns': NOW - 1}, {'output_accel': -0.5},
                                     {'car_gasPressed': True}, {'car_brakePressed': True}, {'car_canValid': False}])
def test_maximum_mitigation_color_requires_exact_owner_consumption_and_current_critical_inputs(changes):
  item = dict(record(strength=1.0), version=2)
  assert not mitigation_receipt([item], **changes)


@pytest.mark.parametrize('changes', [{'version': 1}, {'strength': 0.5}, {'kind': 'hold'},
                                     {'sessionId': '', 'trackId': 0, 'modelMonoTime': NOW, 'visionIndex': 0}])
def test_generic_fault_or_visual_only_evidence_never_gets_v2_receipt(changes):
  item = dict(record(strength=1.0), version=2)
  item.update(changes)
  assert not mitigation_receipt([item])
