from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from openpilot.selfdrive.controls.lib.object_influence import (FOLLOW_STRENGTH, accepted_object_influences,
                                                              influence_identity, plan_object_influences)


NOW = 2_000_000_000


def lead(track=1, *, source='radar', model=NOW, index=-1, session='session-a'):
  return NS(present=True, evidence=NS(version=1, source=source, sessionId=session if track else '', trackId=track,
                                     generation=1, modelMonoTime=model, visionIndex=index))


def radar(*objects):
  return NS(mdMonoTime=NOW, fusionActive=True, fusionDegraded=False, radarObjects=objects,
            leadOne=objects[0] if objects else NS(present=False), leadTwo=objects[1] if len(objects) > 1 else NS(present=False))


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
             long_control_state='stopping' if resolved_stop else 'pid')
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


def test_only_active_formal_leads_get_records_and_joint_constraints_are_retained():
  state = radar(lead(2, source='fused', index=1), lead(0, source='vision', index=0), lead(1), lead(3))
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


@pytest.mark.parametrize('kind', ['braking', 'follow', 'hold', 'release'])
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
                                     {'plan_motionIntent': NS(valid=False, ready=True, releaseAllowed=False)}])
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


def test_instance_reader_adapters_preserve_plan_attribution_and_owner_receipts():
  state = radar(lead(1, source='fused', index=0))
  before = deepcopy(state)
  adapter = NS(as_reader=lambda: state)
  identities = (('session-a', 1),)
  assert plan_records(adapter, identities) == plan_records(state, identities)
  assert state == before

  plan = NS(objectInfluences=[record()], modelMonoTime=NOW, solverStatus=0, aTarget=-1.0,
            motionIntent=NS(valid=True, ready=True, releaseAllowed=False))
  before_plan = deepcopy(plan)
  assert receipt([], plan_as_reader=lambda: plan) == [record()]
  assert plan == before_plan


def test_extra_object_cannot_claim_plan_influence_or_a_departure_barrier():
  state = radar(lead(1), lead(2), lead(3))
  extra_identity = ('session-a', 3)
  assert not plan_records(state, (extra_identity,))
  assert not plan_records(state, reference_accepted=False, held_identity=extra_identity, standstill=True)
  assert not plan_records(state, reference_accepted=False, release_identity=extra_identity, target_accel=0.5)
  identities = (('session-a', 1), extra_identity, ('session-a', 2))
  assert [influence_identity(value) for value in plan_records(state, identities)] == [('session-a', 1), ('session-a', 2)]


def test_retired_v2_extra_object_braking_record_never_gets_an_owner_receipt():
  retired = dict(record('braking', track=3, strength=1.0), version=2)
  ordinary = record()
  assert receipt([retired, ordinary]) == [ordinary]
  assert not receipt([retired], constrained=True, plan_solverStatus=4,
                     plan_motionIntent=NS(valid=False, ready=True, brakingValid=True, brakingAccel=-3.5,
                                          allowCreep=False, movingLead=False, releaseAllowed=False),
                     plan_aTarget=-3.5, output_accel=-3.5)


@pytest.mark.parametrize('strength', [FOLLOW_STRENGTH, 1.0])
def test_retired_caution_record_is_not_acknowledged_even_at_zero_acceleration(strength):
  assert not receipt([record('caution', strength=strength)], plan_aTarget=0.0, output_accel=0.0)
