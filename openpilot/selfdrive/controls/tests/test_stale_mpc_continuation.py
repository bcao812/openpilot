"""Expired-object prediction is retired; numeric status4 keeps current checks."""
from copy import deepcopy
from dataclasses import replace

import pytest

from openpilot.selfdrive.controls.lib.lead_qualification import MAX_MEASUREMENT_AGE_NS
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.continuation import build_candidate
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import (
  MPC_INVALID_INPUT, T_IDXS, LongitudinalMpc, MpcReference,
)
from openpilot.selfdrive.controls.lib.radar_consumption import RadarConsumption
from openpilot.selfdrive.controls.tests.test_mpc_continuation import EPOCH, PRODUCER, mathematical_donor
from openpilot.selfdrive.controls.tests.test_mpc_problem_contract import NativeProblemRecorder
from openpilot.selfdrive.controls.tests.test_radar_fusion_mpc import fused_state, object_lead


def recorded():
  mpc = LongitudinalMpc()
  native = NativeProblemRecorder(mpc.solver)
  native.failure = 'solver_status'
  mpc.solver = native
  mpc.set_cur_state(20.0, 0.0)
  mpc.set_weights(prev_accel_constraint=False)
  return mpc, native


@pytest.mark.parametrize('source', ['radar', 'fused'])
@pytest.mark.parametrize('age_ns', [MAX_MEASUREMENT_AGE_NS + 1, 500_000_000, 5_000_000_000])
def test_age_only_loss_is_removed_before_native_and_never_becomes_a_stale_prediction(source, age_ns):
  now = EPOCH
  expired = object_lead(distance=25.0, speed=0.0, ego_speed=20.0, source=source, epoch=now)
  expired.evidence.positionMonoTime = expired.evidence.velocityMonoTime = now - age_ns
  current = object_lead(track=2, distance=150.0, speed=20.0, ego_speed=20.0, epoch=now)
  radar = fused_state(display=(expired, current))
  original = deepcopy(vars(expired.evidence))
  consumer = RadarConsumption()
  view = consumer.update(radar, publication_ns=now, now_ns=now, speed=20.0)
  assert not view.leadOne.present and view.leadTwo.present
  assert not consumer.degraded and consumer.expired_identities == (('radar-test', 1),)
  mpc, native = recorded()
  mpc.update(view, reference=MpcReference(20.0), reference_ns=now, now_ns=now)
  # The selected fake native result proves that the solve was actually called;
  # it cannot be confused with a pre-solver rejection or synthesized success.
  assert mpc.solution_status == 4 and mpc.diagnostic_solve_calls == 1
  assert len(native.solved_costs) == 1
  assert len(mpc.continuation_problem.lead_trajectories) == 1
  assert vars(expired.evidence) == original


def test_unfiltered_old_geometry_cannot_enter_native_or_acquire_a_prediction_offer():
  mpc, native = recorded()
  lead = object_lead(distance=20.0, epoch=EPOCH)
  lead.evidence.positionMonoTime = lead.evidence.velocityMonoTime = EPOCH - MAX_MEASUREMENT_AGE_NS - 1
  mpc.update(fused_state(display=(lead,)), reference=MpcReference(20.0), reference_ns=EPOCH, now_ns=EPOCH)
  assert mpc.solution_status == MPC_INVALID_INPUT
  assert not mpc.diagnostic_solve_calls and not native.solved_costs
  assert mpc.continuation_problem is None
  assert not mpc.continuation_candidate(mathematical_donor(), now_ns=PRODUCER).accepted


@pytest.mark.parametrize('side', ['donor', 'current'])
def test_numeric_continuation_cannot_extend_an_expired_measurement_lease(side):
  donor = mathematical_donor()
  now = PRODUCER + 50_000_000
  current = replace(donor.problem, reference_ns=EPOCH + 50_000_000, consumed_ns=now)
  if side == 'donor':
    donor = replace(donor, problem=replace(donor.problem, evidence_expiry_ns=now))
  else:
    current = replace(current, evidence_expiry_ns=now)
  result = build_candidate(donor, current, now_ns=now, times=T_IDXS, integrate=LongitudinalMpc.integrate_jerk,
                           validate=LongitudinalMpc._continuation_trajectory_error)
  assert not result.accepted and result.reason == 'continuation_evidence_expired'
