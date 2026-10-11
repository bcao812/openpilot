"""Formal sensor evidence grants a purpose without inventing visual confidence."""
from decimal import Decimal
from fractions import Fraction
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.controls.lib.lead_qualification import (
  MAX_CONTROL_OBJECTS, MAX_MEASUREMENT_AGE_NS, _ControlLeadView, _StructSnapshot,
  _finite, _integer, _qualified_lead_evidence, control_leads, lead_identity, qualified_lead, retired_future_entry,
)


NOW = 10_000_000_000


class _IntSubclass(int):
  pass


class _FloatSubclass(float):
  pass


@pytest.mark.parametrize('value,finite,nonnegative,positive', [
  pytest.param(True, False, False, False, id='true'),
  pytest.param(False, False, False, False, id='false'),
  pytest.param(0, True, True, False, id='int_zero'),
  pytest.param(1, True, True, True, id='int_one'),
  pytest.param(-1, True, False, False, id='negative_int'),
  pytest.param(2**64 - 1, True, True, True, id='uint64_max'),
  pytest.param(10**400, False, True, True, id='overflowing_int'),
  pytest.param(-(10**400), False, False, False, id='overflowing_negative_int'),
  pytest.param(0.0, True, False, False, id='float_zero'),
  pytest.param(1.0, True, False, False, id='integral_float'),
  pytest.param(-.25, True, False, False, id='negative_float'),
  pytest.param(float('nan'), False, False, False, id='nan'),
  pytest.param(float('inf'), False, False, False, id='positive_inf'),
  pytest.param(-float('inf'), False, False, False, id='negative_inf'),
  pytest.param(Fraction(1, 3), True, False, False, id='fraction'),
  pytest.param(Fraction(3, 1), True, False, False, id='integral_fraction'),
  pytest.param(Fraction(10**400, 3), False, False, False, id='overflowing_fraction'),
  pytest.param(Decimal('.25'), False, False, False, id='decimal'),
  pytest.param(Decimal('NaN'), False, False, False, id='decimal_nan'),
  pytest.param(Decimal('Infinity'), False, False, False, id='decimal_inf'),
  pytest.param(_IntSubclass(1), True, True, True, id='int_subclass'),
  pytest.param(_IntSubclass(-1), True, False, False, id='negative_int_subclass'),
  pytest.param(_IntSubclass(10**400), False, True, True, id='overflowing_int_subclass'),
  pytest.param(_FloatSubclass(.25), True, False, False, id='float_subclass'),
  pytest.param(_FloatSubclass(float('nan')), False, False, False, id='nan_float_subclass'),
  pytest.param(None, False, False, False, id='none'),
  pytest.param('1', False, False, False, id='numeric_text'),
])
def test_numeric_validation_preserves_original_type_and_range_semantics(value, finite, nonnegative, positive):
  assert _finite(value) is finite
  assert _integer(value) is nonnegative
  assert _integer(value, 1) is positive


@pytest.mark.parametrize('integer', [int, _IntSubclass])
def test_integer_lower_bound_remains_inclusive_for_builtin_and_subclass(integer):
  assert _integer(integer(-1), -1)
  assert not _integer(integer(-2), -1)
  assert _integer(integer(2**64 - 1), 2**64 - 1)
  assert not _integer(integer(2**64 - 2), 2**64 - 1)


def formal_lead(*, now_ns=NOW, source='radar', evidence=None, **changes):
  evidence_values = {
    'version': 1, 'source': source, 'sessionId': 'radar-session-a', 'trackId': 7, 'generation': 0,
    'positionMonoTime': now_ns, 'velocityMonoTime': now_ns, 'estimateMonoTime': now_ns, 'modelMonoTime': now_ns,
    'positionStd': .2, 'velocityStd': .15, 'positionValid': True, 'velocityValid': True, 'accelerationValid': True,
    'followQualified': True, 'brakeQualified': True, 'stopQualified': True, 'departureQualified': True, 'fcwQualified': True,
    'coasting': source == 'predicted', 'pathState': 'inPath', 'pathOffset': 0.0, 'conflictStart': 0.0, 'conflictEnd': 10.0,
    'lateralSpeed': 0.0, 'pathSource': 'modelPath', 'visionIndex': 0 if source in ('vision', 'fused') else -1, 'reason': 'measured',
  }
  evidence_values.update(evidence or {})
  values = {'present': True, 'dRel': 20.0, 'yRel': 0.0, 'vRel': 0.0, 'vLead': 1.0, 'vLeadK': 1.0,
                'aLeadK': 0.0, 'aLeadTau': 1.5, 'modelProb': .99 if source == 'vision' else 0.0,
                'radar': source != 'vision', 'radarTrackId': -1 if source == 'vision' else 7,
                'evidence': SimpleNamespace(**evidence_values)}
  values.update(changes)
  return SimpleNamespace(**values)


@pytest.mark.parametrize('source', ['radar', 'fused'])
@pytest.mark.parametrize('purpose', ['follow', 'brake', 'stop', 'fcw'])
def test_qualified_radar_has_control_authority_without_visual_probability(source, purpose):
  lead = formal_lead(source=source, vLead=0.0, vRel=-.3)
  assert qualified_lead(lead, purpose, now_ns=NOW)
  assert lead.modelProb == 0.0
  assert not qualified_lead(lead, 'departure', now_ns=NOW)


@pytest.mark.parametrize('probability,expected', [(0.0, False), (.94, False), (.95, True), (1.0, True)])
def test_legacy_and_formal_vision_keep_the_callers_probability_gate(probability, expected):
  for lead in (SimpleNamespace(present=True, modelProb=probability), formal_lead(source='vision', modelProb=probability)):
    assert qualified_lead(lead, 'follow', now_ns=NOW) is expected
  assert qualified_lead(SimpleNamespace(present=True, modelProb=probability), 'brake', 0.0) is True


@pytest.mark.parametrize('probability', [0.0, .95, .99, 1.0])
def test_legacy_probability_cannot_grant_the_new_caution_purpose(probability):
  lead = SimpleNamespace(present=True, modelProb=probability)
  assert not qualified_lead(lead, 'caution', 0.0)
  assert not qualified_lead(lead, 'caution', .95, now_ns=NOW)
  for purpose in ('follow', 'brake', 'stop', 'departure', 'fcw'):
    assert qualified_lead(lead, purpose, .95, now_ns=NOW) is (probability >= .95)
  lead.evidence = SimpleNamespace(version=0, cautionQualified=True, pathState='edge')
  assert not qualified_lead(lead, 'caution', 0.0, now_ns=NOW)


@pytest.mark.parametrize('formal', [False, True])
def test_native_cereal_builder_reads_preserve_dict_and_serialized_message(formal):
  from openpilot.cereal import log

  payload = vars(formal_lead()).copy()
  if formal:
    payload['evidence'] = vars(payload['evidence'])
  else:
    del payload['evidence']
    payload['modelProb'] = .99
  radar = log.RadarState.new_message()
  radar.leadOne = payload
  radar.leadTwo = {'present': False}
  radar.fusionActive = formal
  if formal:
    radar.radarObjects = [payload]
  lead = radar.leadOne
  assert lead._has('evidence') is formal
  before = radar.to_dict()
  serialized = radar.to_bytes()
  radar.clear_write_flag()

  assert qualified_lead(lead, 'brake', now_ns=NOW)
  assert not qualified_lead(lead, 'caution', 0.0, now_ns=NOW)
  assert lead_identity(lead) == (('radar-session-a', 7) if formal else None)
  assert len(control_leads(radar, 'brake', now_ns=NOW)) == 1

  assert lead._has('evidence') is formal
  assert radar.to_dict() == before
  assert radar.to_bytes() == serialized


@pytest.mark.parametrize('field', ['positionMonoTime', 'velocityMonoTime', 'modelMonoTime'])
def test_republishing_old_measurements_does_not_refresh_their_control_age(field):
  lead = formal_lead(evidence={field: NOW - MAX_MEASUREMENT_AGE_NS - 1})
  assert not qualified_lead(lead, 'brake', now_ns=NOW)
  assert not qualified_lead(lead, 'brake')


@pytest.mark.parametrize('field', ['positionMonoTime', 'velocityMonoTime'])
@pytest.mark.parametrize('age_ns,expected', [
  (202_280_000, True), (213_990_000, True), (224_140_000, True),
  (230_000_000, True), (230_000_001, False),
])
def test_measurement_deadline_admits_recorded_consumer_ages_and_expires_after_230ms(field, age_ns, expected):
  lead = formal_lead(evidence={field: NOW - age_ns})
  snapshot = _StructSnapshot({**vars(lead), 'evidence': vars(lead.evidence).copy()})
  for candidate in (lead, snapshot):
    assert qualified_lead(candidate, 'brake', now_ns=NOW) is expected
    assert getattr(candidate.evidence, field) == NOW - age_ns


def test_measurement_extension_does_not_extend_model_lifetime():
  lead = formal_lead(evidence={'modelMonoTime': NOW - 200_000_001})
  snapshot = _StructSnapshot({**vars(lead), 'evidence': vars(lead.evidence).copy()})
  assert not qualified_lead(lead, 'brake', now_ns=NOW)
  assert not qualified_lead(snapshot, 'brake', now_ns=NOW)


@pytest.mark.parametrize('field', ['positionMonoTime', 'velocityMonoTime', 'estimateMonoTime', 'modelMonoTime'])
@pytest.mark.parametrize('stamp', [0, -1, True, NOW + 1])
def test_invalid_or_future_times_do_not_inherit_boolean_authority(field, stamp):
  lead = formal_lead(evidence={field: stamp})
  assert not qualified_lead(lead, 'brake', now_ns=NOW)


def test_model_publication_may_follow_camera_eof_without_a_future_measurement():
  lead = formal_lead(source='vision', evidence={'modelMonoTime': NOW + 50_000_000})
  assert qualified_lead(lead, 'follow', now_ns=NOW + 50_000_000)
  assert qualified_lead(lead, 'follow')
  assert not qualified_lead(lead, 'follow', now_ns=NOW)


def test_new_can_sample_may_be_projected_back_to_the_common_camera_time():
  lead = formal_lead(evidence={'positionMonoTime': NOW + 20_000_000, 'velocityMonoTime': NOW + 20_000_000,
                               'modelMonoTime': NOW + 40_000_000})
  assert qualified_lead(lead, 'brake', now_ns=NOW + 50_000_000)
  assert qualified_lead(lead, 'brake')
  assert not qualified_lead(lead, 'brake', now_ns=NOW)


@pytest.mark.parametrize('field,value', [
  ('version', True), ('version', 2), ('source', 'legacy'), ('source', 'invalid'),
  ('sessionId', ''), ('trackId', 0), ('trackId', True), ('generation', -1),
  ('positionValid', False), ('velocityValid', False), ('brakeQualified', False),
  ('positionStd', 0.0), ('positionStd', float('nan')), ('velocityStd', float('inf')),
  ('pathState', 'unknown'), ('pathState', 'outOfPath'), ('pathSource', ''),
  ('pathOffset', float('nan')), ('pathSource', True),
])
def test_malformed_evidence_cannot_grant_braking(field, value):
  assert not qualified_lead(formal_lead(evidence={field: value}), 'brake', now_ns=NOW)


@pytest.mark.parametrize('field,value', [
  ('conflictEnd', 0.0), ('conflictEnd', float('nan')), ('lateralSpeed', float('nan')),
  ('conflictSupportVersion', 2), ('conflictSupportedEnd', float('nan')),
  ('physicalConflictValid', True), ('physicalConflictEnd', -1.0),
])
def test_retired_occupancy_fields_do_not_change_a_current_formal_leads_authority(field, value):
  lead = formal_lead(evidence={field: value})
  assert qualified_lead(lead, 'brake', now_ns=NOW)
  assert qualified_lead(lead, 'fcw', now_ns=NOW)
  assert not qualified_lead(lead, 'brake', now_ns=NOW + MAX_MEASUREMENT_AGE_NS + 1)


@pytest.mark.parametrize('field,value', [
  ('dRel', 0.0), ('dRel', float('nan')), ('vRel', float('inf')), ('vLead', True),
  ('yRel', float('nan')), ('aLeadK', float('nan')), ('modelProb', 1.01), ('modelProb', float('nan')),
])
def test_nonfinite_or_incoherent_formal_values_are_rejected(field, value):
  assert not qualified_lead(formal_lead(**{field: value}), 'brake', now_ns=NOW)


def test_constant_velocity_prior_is_explicit_and_not_a_measured_acceleration():
  lead = formal_lead(evidence={'accelerationValid': False})
  assert qualified_lead(lead, 'brake', now_ns=NOW)
  lead.aLeadK = .1
  assert not qualified_lead(lead, 'brake', now_ns=NOW)


@pytest.mark.parametrize('source,coasting', [
  ('predicted', True), ('predicted', False), ('radar', True), ('fused', True), ('vision', True),
])
@pytest.mark.parametrize('snapshot', [False, True])
def test_prediction_or_coasting_never_grants_control_even_with_fresh_stamps(source, coasting, snapshot):
  lead = formal_lead(source=source, evidence={'coasting': coasting})
  if snapshot:
    lead = _StructSnapshot({**vars(lead), 'evidence': vars(lead.evidence)})
  for purpose in ('follow', 'brake', 'stop', 'departure', 'fcw'):
    assert not qualified_lead(lead, purpose, 0.0, now_ns=NOW)


@pytest.mark.parametrize('source', ['radar', 'fused', 'predicted'])
@pytest.mark.parametrize('path_state,start', [('entering', 0.0), ('entering', .05), ('inPath', .05)])
def test_future_entry_never_grants_a_control_or_fcw_purpose(source, path_state, start):
  lead = formal_lead(source=source, evidence={'pathState': path_state, 'conflictStart': start, 'conflictEnd': .1})
  assert _qualified_lead_evidence(lead, 'brake', now_ns=NOW) is (source != 'predicted')
  assert retired_future_entry(lead, now_ns=NOW) is (source != 'predicted')
  for purpose in ('follow', 'brake', 'stop', 'departure', 'fcw', 'caution'):
    for now in (NOW, NOW + 100_000_000, NOW + 200_000_001):
      for active_window in (True, False):
        assert not qualified_lead(lead, purpose, now_ns=now, require_active_window=active_window)


@pytest.mark.parametrize('source', ['vision', 'legacy'])
def test_future_entry_retirement_does_not_change_visual_or_legacy_qualification(source):
  lead = formal_lead(source='vision', evidence={'pathState': 'entering', 'conflictStart': .05, 'conflictEnd': .1})
  if source == 'legacy':
    lead.evidence.version = 0
  assert not retired_future_entry(lead, now_ns=NOW)
  assert qualified_lead(lead, 'brake', now_ns=NOW)
  assert qualified_lead(lead, 'fcw', now_ns=NOW)


@pytest.mark.parametrize('field,value', [
  ('aLeadTau', -.1), ('aLeadTau', float('nan')), ('dRel', float('nan')),
  ('conflictEnd', 11.0), ('conflictEnd', float('inf')), ('conflictStart', float('nan')),
  ('positionValid', False), ('positionMonoTime', NOW - MAX_MEASUREMENT_AGE_NS - 1),
  ('positionMonoTime', NOW + 1), ('modelMonoTime', NOW + 1), ('estimateMonoTime', NOW - 250_000_001),
  ('physicalConflictValid', 1), ('physicalConflictStart', float('nan')), ('physicalConflictEnd', -.1),
])
def test_future_entry_label_does_not_hide_invalid_projection_evidence(field, value):
  lead = formal_lead(evidence={'pathState': 'entering', 'conflictStart': .05, 'conflictEnd': .1,
                               'physicalConflictValid': True, 'physicalConflictStart': .05, 'physicalConflictEnd': .1})
  setattr(lead if field in ('aLeadTau', 'dRel') else lead.evidence, field, value)
  radar = SimpleNamespace(fusionActive=True, radarObjects=[lead], leadOne=lead, leadTwo=SimpleNamespace(present=False))
  assert not retired_future_entry(lead, now_ns=NOW)
  assert control_leads(radar, now_ns=NOW) == (lead,)
  assert not qualified_lead(lead, 'brake', now_ns=NOW)
  assert not qualified_lead(lead, 'fcw', now_ns=NOW)


@pytest.mark.parametrize('path_state,brake', [('inPath', True), ('entering', True), ('entering', False)])
def test_future_entry_retirement_keeps_mixed_roles_and_nonedge_caution_visible(path_state, brake):
  lead = caution_lead(evidence={'pathState': path_state, 'conflictStart': .5, 'brakeQualified': brake})
  snapshot = _StructSnapshot({**vars(lead), 'evidence': vars(lead.evidence).copy()})
  for candidate in (lead, snapshot):
    radar = SimpleNamespace(fusionActive=True, radarObjects=[candidate], leadOne=candidate, leadTwo=SimpleNamespace(present=False))
    assert not retired_future_entry(candidate, now_ns=NOW)
    assert control_leads(radar, now_ns=NOW) == (candidate,)
    assert not qualified_lead(candidate, 'caution', 0.0, now_ns=NOW)
    assert not qualified_lead(candidate, 'brake', 0.0, now_ns=NOW)


def test_cached_collection_cannot_add_control_objects_or_override_the_formal_slots():
  class Collection(_ControlLeadView):
    __slots__ = ('_control_objects', 'leadOne', 'leadTwo')

  current = formal_lead(evidence={'trackId': 8})
  future = formal_lead(evidence={'pathState': 'entering', 'conflictStart': .5})
  radar = Collection()
  radar._control_objects = tuple(_StructSnapshot({**vars(lead), 'evidence': vars(lead.evidence).copy()}) for lead in (future, current))
  assert control_leads(radar) == ()
  radar.leadOne, radar.leadTwo = current, SimpleNamespace(present=False)
  for purpose in (None, 'brake', 'fcw'):
    assert control_leads(radar, purpose, now_ns=NOW) == (current,)
  radar.leadOne = future
  assert control_leads(radar) == ()


def test_departure_flag_needs_fresh_forward_motion_and_is_not_an_identity_proof():
  lead = formal_lead(vLead=.8, vRel=.8)
  assert qualified_lead(lead, 'departure', now_ns=NOW)
  lead.evidence.departureQualified = False
  assert not qualified_lead(lead, 'departure', now_ns=NOW)
  lead.evidence.departureQualified = True
  lead.vRel = 0.0
  assert not qualified_lead(lead, 'departure', now_ns=NOW)


@pytest.mark.parametrize('distance', [-1.52, -.2, 0.0])
def test_very_close_visual_cut_in_remains_a_braking_risk_after_bumper_conversion(distance):
  lead = formal_lead(source='vision', dRel=distance)
  assert qualified_lead(lead, 'brake', now_ns=NOW)
  assert qualified_lead(lead, 'fcw', now_ns=NOW)
  assert not qualified_lead(lead, 'follow', now_ns=NOW)
  assert not qualified_lead(lead, 'departure', now_ns=NOW)
  assert not qualified_lead(formal_lead(source='vision', dRel=-1.53), 'brake', now_ns=NOW)


def test_physical_identity_survives_slot_generation_but_not_restart_or_new_track():
  first = formal_lead()
  migrated = formal_lead(evidence={'generation': 12})
  restarted = formal_lead(evidence={'sessionId': 'radar-session-b'})
  different = formal_lead(evidence={'trackId': 8})
  assert lead_identity(first) == lead_identity(migrated) == ('radar-session-a', 7)
  assert lead_identity(first) != lead_identity(restarted)
  assert lead_identity(first) != lead_identity(different)
  assert lead_identity(formal_lead(source='vision')) is None


def test_raw_risk_collection_cannot_add_targets_to_the_two_formal_slots():
  far = formal_lead(dRel=70.0)
  nearby = formal_lead(dRel=10.0, evidence={'trackId': 8})
  crossing = formal_lead(dRel=25.0, evidence={'trackId': 9, 'pathState': 'entering'})
  visual = formal_lead(source='vision', dRel=8.0)
  radar = SimpleNamespace(fusionActive=True, radarObjects=[far, nearby, crossing, visual], leadOne=far, leadTwo=visual)
  assert control_leads(radar) == (far, visual)
  assert control_leads(radar, 'brake', now_ns=NOW) == (far, visual)
  assert control_leads(radar, 'stop', now_ns=NOW) == (far, visual)
  radar.fusionActive = False
  assert control_leads(radar) == (far, visual)


def test_oversize_raw_collection_does_not_replace_distinct_formal_leads():
  leads = [formal_lead(evidence={'trackId': index + 1}) for index in range(MAX_CONTROL_OBJECTS + 1)]
  radar = SimpleNamespace(fusionActive=True, radarObjects=leads, leadOne=leads[0], leadTwo=leads[1])
  assert control_leads(radar) == tuple(leads[:2])


def caution_lead(*, now_ns=NOW, evidence=None, **changes):
  flags = {'followQualified': False, 'brakeQualified': False, 'stopQualified': False,
           'departureQualified': False, 'fcwQualified': False, 'cautionQualified': True,
           'pathState': 'edge'}
  flags.update(evidence or {})
  return formal_lead(now_ns=now_ns, evidence=flags, **changes)


def test_retired_edge_caution_never_grants_any_control_purpose():
  lead = caution_lead(vRel=-10.0, vLead=0.0)
  assert not qualified_lead(lead, 'caution', now_ns=NOW)
  for purpose in ('follow', 'brake', 'stop', 'departure', 'fcw'):
    assert not qualified_lead(lead, purpose, 0.0, now_ns=NOW)
  assert not qualified_lead(lead, 'caution', now_ns=NOW + MAX_MEASUREMENT_AGE_NS + 1)
  assert not qualified_lead(caution_lead(source='vision'), 'caution', now_ns=NOW)
  assert not qualified_lead(caution_lead(source='predicted'), 'caution', now_ns=NOW)


@pytest.mark.parametrize('field,value', [
  ('positionMonoTime', NOW + 1), ('velocityMonoTime', NOW - MAX_MEASUREMENT_AGE_NS - 1),
  ('modelMonoTime', 0), ('positionValid', False), ('velocityValid', False), ('sessionId', ''),
  ('trackId', 0), ('pathState', 'inPath'), ('coasting', True), ('brakeQualified', True),
  ('stopQualified', True), ('departureQualified', True), ('positionStd', float('nan')),
])
def test_edge_caution_keeps_full_source_and_role_validation(field, value):
  assert not qualified_lead(caution_lead(evidence={field: value}), 'caution', now_ns=NOW)


def _assert_snapshot_matches_uncached(lead):
  payload = {**vars(lead), 'evidence': vars(lead.evidence).copy()}
  snapshot = _StructSnapshot(payload)
  times = (None, NOW - 1, NOW, NOW + 100_000_000, NOW + MAX_MEASUREMENT_AGE_NS, NOW + MAX_MEASUREMENT_AGE_NS + 1,
           NOW + 250_000_000, NOW + 350_000_000, NOW + 350_000_001, True, float(NOW))
  for now_ns in times:
    for purpose in ('follow', 'brake', 'stop', 'departure', 'fcw', 'caution', 'invalid'):
      for probability in (0.0, .95, 1.0, -.1, float('nan')):
        for active_window in (True, False, 1):
          kwargs = {'now_ns': now_ns, 'require_active_window': active_window}
          assert qualified_lead(snapshot, purpose, probability, **kwargs) == qualified_lead(payload, purpose, probability, **kwargs)
          assert (_qualified_lead_evidence(snapshot, purpose, probability, **kwargs)
                  == _qualified_lead_evidence(payload, purpose, probability, **kwargs))
    assert retired_future_entry(snapshot, now_ns=now_ns) == retired_future_entry(payload, now_ns=now_ns)
  assert lead_identity(snapshot) == lead_identity(payload)
  return snapshot


@pytest.mark.parametrize('source', ['radar', 'fused', 'vision', 'predicted', 'caution', 'legacy'])
def test_snapshot_common_facts_never_cache_a_purpose_probability_time_or_window(source):
  lead = caution_lead(vRel=.5) if source == 'caution' else formal_lead(source=source, vRel=.5)
  lead.evidence.conflictEnd = .1
  if source == 'legacy':
    lead.evidence.version = 0
  snapshot = _assert_snapshot_matches_uncached(lead)
  with pytest.raises(AttributeError):
    snapshot.evidence.positionMonoTime = NOW + 1
  with pytest.raises(AttributeError):
    snapshot._facts = None


@pytest.mark.parametrize('scope,field,value', [
  ('lead', 'present', 1), ('lead', 'present', False), ('lead', 'modelProb', True),
  ('lead', 'modelProb', float('nan')), ('lead', 'modelProb', 1.01),
  ('lead', 'dRel', float('nan')), ('lead', 'dRel', -1.52), ('lead', 'dRel', 0.0),
  ('lead', 'aLeadK', .1), ('lead', 'vLead', float('inf')), ('lead', 'vRel', -.1),
  ('evidence', 'version', True), ('evidence', 'version', 2), ('evidence', 'source', 'invalid'),
  ('evidence', 'sessionId', ''), ('evidence', 'trackId', 0), ('evidence', 'generation', -1),
  ('evidence', 'coasting', 1), ('evidence', 'coasting', True), ('evidence', 'accelerationValid', False),
  ('evidence', 'positionMonoTime', NOW + 1), ('evidence', 'velocityMonoTime', NOW - 350_000_001),
  ('evidence', 'estimateMonoTime', NOW - 250_000_001), ('evidence', 'modelMonoTime', 0),
  ('evidence', 'positionValid', False), ('evidence', 'velocityValid', 1),
  ('evidence', 'positionStd', 0.0), ('evidence', 'velocityStd', float('nan')),
  ('evidence', 'pathState', 'entering'), ('evidence', 'pathState', 'edge'), ('evidence', 'pathState', 'unknown'),
  ('evidence', 'pathSource', ''), ('evidence', 'pathOffset', float('nan')),
  ('evidence', 'conflictStart', -.1), ('evidence', 'conflictStart', .05), ('evidence', 'conflictEnd', 0.0),
  ('evidence', 'conflictEnd', .1), ('evidence', 'lateralSpeed', float('inf')),
  ('evidence', 'brakeQualified', 1), ('evidence', 'cautionQualified', True),
])
def test_snapshot_malformed_and_boundary_values_match_the_uncached_path(scope, field, value):
  lead = formal_lead(vRel=.5)
  setattr(lead if scope == 'lead' else lead.evidence, field, value)
  _assert_snapshot_matches_uncached(lead)


@pytest.mark.parametrize('distance', [-1.53, -1.52, -.2, 0.0])
def test_snapshot_close_visual_risk_retains_its_distinct_purpose_gates(distance):
  _assert_snapshot_matches_uncached(formal_lead(source='vision', dRel=distance, vRel=.5))


def test_snapshots_of_the_same_identity_keep_independent_evidence_and_age():
  fresh = formal_lead(vRel=.5)
  old = formal_lead(vRel=.5, evidence={'positionMonoTime': NOW - MAX_MEASUREMENT_AGE_NS - 1, 'departureQualified': False})
  snapshots = [_StructSnapshot({**vars(lead), 'evidence': vars(lead.evidence).copy()}) for lead in (fresh, old)]
  assert lead_identity(snapshots[0]) == lead_identity(snapshots[1])
  for _ in range(2):
    assert qualified_lead(snapshots[0], 'brake', now_ns=NOW)
    assert not qualified_lead(snapshots[1], 'brake', now_ns=NOW)
    assert qualified_lead(snapshots[0], 'departure', now_ns=NOW)
    assert not qualified_lead(snapshots[1], 'departure', now_ns=NOW)


@pytest.mark.parametrize('value', [Fraction(1, 3), _FloatSubclass(.25), _IntSubclass(1)])
def test_non_native_snapshot_scalars_keep_the_general_validation_path(value):
  snapshot = _assert_snapshot_matches_uncached(formal_lead(vLead=value))
  assert not snapshot._cache_safe and snapshot._facts is None
