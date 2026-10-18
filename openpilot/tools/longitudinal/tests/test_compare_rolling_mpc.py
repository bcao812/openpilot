from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from openpilot.tools.longitudinal import compare_rolling_mpc as replay


def row(stamp, data=None, *, valid=True):
  return {'mono_ns': stamp, 'valid': valid, 'data': {} if data is None else data}


def join_fixture():
  epoch = 10 * replay.NS
  producer = epoch + 80_000_000
  plan = row(producer + 10_000_000, {
    'reference': {'referenceMonoTime': epoch, 'sourceCarStateMonoTime': epoch, 'producerMonoTime': producer,
                  'producerInstanceMonoTime': replay.NS, 'timingVersion': 1, 'timingKnown': True, 'initialStateKnown': True},
    'modelMonoTime': epoch - 10_000_000, 'radarStateInputMonoTime': epoch + 10_000_000,
  })
  tables = {'carState': replay.Series([row(epoch)]),
            'modelV2': replay.Series([row(epoch - 10_000_000)]),
            'radarState': replay.Series([row(epoch + 10_000_000)])}
  for topic in replay.ASYNC_TOPICS:
    tables[topic] = replay.Series([row(producer - 1_000_000), row(producer + 1_000_000)])
  tables['sp_by_plan'] = {plan['mono_ns']: row(plan['mono_ns'] + 1_000_000, {'longitudinalPlanMonoTime': plan['mono_ns']})}
  return plan, tables


def test_exact_join_and_asof_do_not_consume_future_publications():
  plan, tables = join_fixture()
  joined = replay.join_inputs(plan, tables, 250_000_000)
  assert joined['carState']['mono_ns'] == plan['data']['reference']['sourceCarStateMonoTime']
  assert joined['radarState']['mono_ns'] == plan['data']['radarStateInputMonoTime']
  for topic in replay.ASYNC_TOPICS:
    assert joined[topic]['mono_ns'] == plan['data']['reference']['producerMonoTime'] - 1_000_000
  # SP is a policy result explicitly associated with this same plan. It is
  # not treated as a future observation at producerMonoTime.
  assert joined['longitudinalPlanSP']['data']['longitudinalPlanMonoTime'] == plan['mono_ns']


@pytest.mark.parametrize('failure,reason', [
  ('missing_exact', 'missing_exact_join_radarState'), ('missing_identity', 'missing_exact_identity_radarState'),
  ('future_model', 'noncausal_or_stale_join_modelV2'), ('invalid_radar', 'invalid_input_radarState'),
  ('different_planSP', 'missing_exact_planSP'), ('expired_epoch', 'noncausal_or_expired_reference_timing'),
])
def test_join_rejects_missing_invalid_or_wrong_exact_inputs(failure, reason):
  plan, tables = join_fixture()
  if failure == 'missing_exact':
    # A nearby valid radar publication must not repair the missing exact ID.
    plan['data']['radarStateInputMonoTime'] += 1
  elif failure == 'missing_identity':
    del plan['data']['radarStateInputMonoTime']
  elif failure == 'future_model':
    future = plan['data']['reference']['producerMonoTime'] + 1
    plan['data']['modelMonoTime'] = future
    tables['modelV2'] = replay.Series([row(future)])
  elif failure == 'invalid_radar':
    tables['radarState'].rows[0]['valid'] = False
  elif failure == 'different_planSP':
    tables['sp_by_plan'] = {}
  else:
    plan['data']['reference']['referenceMonoTime'] -= replay.NS
    plan['data']['reference']['sourceCarStateMonoTime'] -= replay.NS
  with pytest.raises(replay.InputRejected, match=reason):
    replay.join_inputs(plan, tables, 250_000_000)


@pytest.fixture(scope='module')
def runtime():
  return replay.load_runtime()


def native_inputs(rt, frame=0, *, present=True):
  radar = rt.log.RadarState.new_message()
  lead = radar.leadOne
  lead.present = present
  lead.dRel, lead.vLead, lead.vRel = 15.0, 6.0, 1.0
  lead.aLeadK, lead.aLeadTau, lead.modelProb = 0.0, 1.5, 1.0
  epoch = 10 * replay.NS + frame * 50_000_000
  return SimpleNamespace(reference=rt.lm.MpcReference(None, accel_max=1.4, comfort_stop_distance=4.0,
                                                     allow_future_rest_reschedule=True, allow_interval_rest=True),
                         radar=radar.as_reader(), original_radar=radar.as_reader(), speed=5.0, accel=0.0,
                         measured_speed=5.0, measured_accel=0.0, epoch_ns=epoch, producer_ns=epoch + 3_000_000,
                         instance_ns=replay.NS, delay=0.5, action_t=0.55, ceiling=1.4,
                         personality=rt.log.LongitudinalPersonality.standard,
                         plan={'data': {'reference': {'accel': 900.0}, 'aTarget': 800.0}})


def test_candidate_fields_and_installed_physical_constraints_are_isolated(runtime):
  rt = runtime
  base = native_inputs(rt).reference
  geometry = replay.arm_reference(base, replay.ARMS['geometry'])
  changed = [key for key, value in asdict(base).items() if value != asdict(geometry)[key]]
  assert changed == ['comfort_stop_distance']
  assert geometry.comfort_stop_distance == 3.0
  assert replay.arm_reference(base, replay.ARMS['continuity']) == base
  assert replay.history_scale(0.8, 0.5, replay.ARMS['continuity']) == pytest.approx(0.3)

  runners = {name: replay.NativeArm(arm, rt) for name, arm in replay.ARMS.items()}
  for frame in range(2):
    outputs = {name: runner.solve(native_inputs(rt, frame))[0] for name, runner in runners.items()}
    assert all(output['status'] == 'solved' for output in outputs.values())
  baseline = runners['baseline'].problem
  for name, runner in runners.items():
    assert replay.value_hash(replay.invariant_fields(runner.problem, rt)) == replay.value_hash(replay.invariant_fields(baseline, rt)), name
  rt.np.testing.assert_allclose(runners['geometry'].problem['weights'], baseline['weights'])
  rt.np.testing.assert_allclose(runners['continuity'].problem['weights'][:, 4], baseline['weights'][:, 4] * 0.5)
  assert (runners['geometry'].problem['params'][:, rt.lm.COMFORT_STOP_DISTANCE]
          < baseline['params'][:, rt.lm.COMFORT_STOP_DISTANCE]).any()
  # Deliberately absurd recorded outputs above are diagnostics only.
  assert all(abs(runner.previous_request) <= 1.4 for runner in runners.values())


@pytest.mark.parametrize('scale', [True, 0.0, 0.99, 8.01, float('nan'), float('inf')])
def test_jerk_experiment_refuses_unbounded_or_weakening_weights(scale):
  with pytest.raises(ValueError, match='jerk scale'):
    replay.Arm('jerk', jerk_scale=scale)


def test_braking_jerk_experiment_runs_the_native_motion_with_current_risk_unchanged(runtime):
  rt = runtime
  baseline = replay.NativeArm(replay.ARMS['baseline'], rt)
  candidate = replay.NativeArm(replay.Arm('jerk', jerk_scale=4.0), rt)
  inputs = native_inputs(rt)
  # A current newly selected slower lead, with no inherited acceleration
  # reference. Both arms must solve it; neither edits the finished command.
  radar = inputs.radar.as_builder()
  radar.leadOne.dRel = 11.0
  radar.leadOne.vLead = 3.0
  radar.leadOne.vRel = -2.0
  inputs.radar = inputs.original_radar = radar.as_reader()
  b, _ = baseline.solve(inputs)
  c, _ = candidate.solve(inputs)
  assert b['status'] == c['status'] == 'solved'
  assert not b['history_prepared'] and not c['history_prepared']
  assert replay.value_hash(replay.invariant_fields(baseline.problem, rt, allow_jerk=True)) == replay.value_hash(
    replay.invariant_fields(candidate.problem, rt, allow_jerk=True))
  assert replay.value_hash(replay.invariant_fields(baseline.problem, rt)) != replay.value_hash(
    replay.invariant_fields(candidate.problem, rt))
  rt.np.testing.assert_allclose(candidate.problem['weights'][:, 5], baseline.problem['weights'][:, 5] * 4.0)
  assert abs(candidate.mpc.j_solution[0]) < abs(baseline.mpc.j_solution[0])
  assert candidate.previous_request == pytest.approx(min(float(candidate.mpc.sample_solution(inputs.action_t)[1]), inputs.ceiling))

  missing = native_inputs(rt, 1, present=False)
  c2, _ = candidate.solve(missing)
  assert c2['status'] == 'solved'
  assert not c2['history_enabled']
  assert candidate.mpc._source_leads == ()


def test_current_jerk_guard_is_withdrawn_by_current_lead_risk_or_role(runtime):
  rt = runtime
  inputs = native_inputs(rt)
  radar = inputs.radar.as_builder()
  radar.leadOne.dRel = 45.0
  radar.leadOne.vLead = 3.0
  radar.leadOne.vRel = -2.0
  inputs.radar = inputs.original_radar = radar.as_reader()
  assert replay.current_braking_jerk_scale(inputs, rt, 4.0) > 1.0
  radar.leadOne.aLeadK = -3.0
  inputs.radar = radar.as_reader()
  assert replay.current_braking_jerk_scale(inputs, rt, 4.0) == 1.0
  radar.leadOne.aLeadK = 0.0
  radar.leadOne.dRel = 5.0
  inputs.radar = radar.as_reader()
  assert replay.current_braking_jerk_scale(inputs, rt, 4.0) == 1.0
  radar.leadOne.present = False
  inputs.radar = radar.as_reader()
  assert replay.current_braking_jerk_scale(inputs, rt, 4.0) == 1.0


def test_current_jerk_guard_does_not_override_an_independent_braking_constraint(runtime):
  inputs = native_inputs(runtime)
  inputs.ceiling = -1.0
  assert replay.current_braking_jerk_scale(inputs, runtime, 4.0) == 1.0


def test_native_history_is_marked_after_solve_and_retires_removed_support(runtime):
  runner = replay.NativeArm(replay.ARMS['baseline'], runtime)
  first, _ = runner.solve(native_inputs(runtime))
  assert first['status'] == 'solved' and first['history_marked']
  assert not first['history_prepared'] and not first['history_enabled']
  second, _ = runner.solve(native_inputs(runtime, 1))
  assert second['history_prepared'] and second['history_enabled'] and second['history_marked']
  assert second['previous_projected_request'] == first['projected_request']
  assert second['previous_source_age_s'] == pytest.approx(0.05)
  removed, _ = runner.solve(native_inputs(runtime, 2, present=False))
  assert removed['status'] == 'solved' and removed['history_prepared']
  assert removed['history_reason'] == 'anonymous_support_removed'
  assert not removed['history_enabled']
  # The current empty-road problem may establish a new reference. It must
  # not inherit the removed object's mixed acceleration smoothing obligation.
  assert removed['history_marked']
  assert runtime.np.all(runner.problem['params'][:, 3] == 0)


def test_failed_native_solve_remains_a_failure_and_cannot_seed_next_history(runtime):
  runner = replay.NativeArm(replay.ARMS['baseline'], runtime)
  first, _ = runner.solve(native_inputs(runtime))
  assert first['history_marked']
  failing = Mock(wraps=runner.mpc.solver)
  failing.solve.return_value = 4
  with patch.object(runner.mpc, 'solver', failing):
    failure, node = runner.solve(native_inputs(runtime, 1))
  assert failure['status'] == 'solver_failed' and failure['solver_status'] == 4
  assert not failure['history_marked']
  assert node['problem'] is not None  # The failed installed problem survives reset.
  recovery, _ = runner.solve(native_inputs(runtime, 2))
  assert recovery['status'] == 'solved' and recovery['history_marked']
  assert not recovery['history_prepared'] and not recovery['history_enabled']


def test_predicted_physical_gap_uses_ego_motion_and_never_changes_observation():
  observed_gap = 10.0
  times = [0.0, 1.0, 2.0]
  lead = [[observed_gap, 3.0], [13.0, 3.0], [16.0, 3.0]]
  constant_ego = [[0.0, 2.0, 0.0], [2.0, 2.0, 0.0], [4.0, 2.0, 0.0]]
  accelerating_ego = [[0.0, 2.0, 1.0], [2.5, 3.0, 1.0], [6.0, 4.0, 1.0]]
  assert replay.physical_gap_at(times, lead, constant_ego, [0.0, 0.0], 2.0) == pytest.approx(12.0)
  assert replay.physical_gap_at(times, lead, accelerating_ego, [0.0, 0.0], 2.0) == pytest.approx(10.0)
  assert observed_gap == lead[0][0] == 10.0
  # At an interior native time, linearly interpolating ego x would be wrong.
  jerk_ego = [[0.0, 2.0, 0.0], [8.0, 8.0, 6.0]]
  assert replay.physical_gap_at([0.0, 2.0], [[10.0, 3.0], [16.0, 3.0]], jerk_ego, [3.0], 1.0) == pytest.approx(10.5)


def test_summary_keeps_failed_frames_and_counts_metric_coverage(runtime):
  records = [
    {'relative_s': -10.0, 'arm': 'baseline', 'status': 'solved', 'reason': '', 'reference_accel': 0.4},
    {'relative_s': -9.95, 'arm': 'baseline', 'status': 'solver_failed', 'reason': 'solver_status'},
    {'relative_s': -9.9, 'arm': 'baseline', 'status': 'unsupported', 'reason': 'compact_or_standstill_geometry'},
  ]
  summary = replay.summarize(records, [replay.ARMS['baseline']], runtime)['primary']['baseline']
  assert summary['frames'] == 3
  assert summary['statuses'] == {'solved': 1, 'solver_failed': 1, 'unsupported': 1}
  assert summary['metrics']['reference_accel']['n'] == 1


def test_model_reference_accepts_ordinary_blended_prefix_without_inventing_a_native_model_target(runtime):
  history = replay.ReferenceInputs(runtime)
  model = {'velocity': {'x': [4.0] * len(runtime.up.ModelConstants.T_IDXS)}}
  assert history.model_mode(model, e2e=True, speed=4.0, model_ns=9_990_000_000,
                            epoch_ns=10_000_000_000, now_ns=10_003_000_000) == 'ordinary_e2e'
  assert history.model_mode(model, e2e=False, speed=4.0, model_ns=10_040_000_000,
                            epoch_ns=10_050_000_000, now_ns=10_053_000_000) == 'acc'


def test_model_reference_does_not_turn_typed_stop_into_ordinary_motion(runtime):
  history = replay.ReferenceInputs(runtime)
  model = {'velocity': {'x': [0.0] * len(runtime.up.ModelConstants.T_IDXS)}}
  with pytest.raises(replay.InputRejected, match='nonordinary_model_reference'):
    history.model_mode(model, e2e=True, speed=0.0, model_ns=9_990_000_000,
                        epoch_ns=10_000_000_000, now_ns=10_003_000_000)
