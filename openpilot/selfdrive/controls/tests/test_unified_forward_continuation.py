import numpy as np
import pytest

from opendbc.car.honda.values import CAR
from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.drive_helpers import STOP_ACCEL, STOP_SPEED
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionReferenceKind, StopReason
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_TIMING_VERSION, ReferenceTimingConsumer
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import reference_requests_departure
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import advance, make_planner, make_state
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import ControlsHarness
from openpilot.selfdrive.modeld.constants import ModelConstants


def execution_state(*, raw_forward=True, speed=0.02):
  state = make_state(speed=speed, target=3.0, experimental=True)
  state['carState'].aEgo = 0.18
  state['carControl'].actuators.longControlState = LongCtrlState.pid
  state['carControl'].actuators.accel = 0.18
  state['controlsState'].motionState.phase = 'track'
  state['modelV2'].velocity.x = (np.minimum(2.76 + 0.2 * np.asarray(ModelConstants.T_IDXS), 3.0).tolist()
                                if raw_forward else [3.0] * len(ModelConstants.T_IDXS))
  return state


def execution_time(planner, state, frame):
  now_ns = advance(state, frame)
  # Consume the previous producer's serialized reference before the next
  # planner invocation. Phase/PID are explicit owner-output fixtures; timing
  # fields come from the production consumer, not a current-plan identity.
  state.logMonoTime['carControl'] = now_ns - 10_000_000
  motion = state['controlsState'].motionState
  motion.referenceTimingVersion = 0
  motion.referenceTimingValid = False
  for field in ('referenceConsumerMonoTime', 'referenceProducerMonoTime', 'referenceProducerInstanceMonoTime',
                'referenceSourceMonoTime', 'referenceEpochMonoTime'):
    setattr(motion, field, 0)
  previous = planner.reference_timing.pending
  if previous is not None:
    reference = log.LongitudinalPlan.new_message().reference
    planner.reference_timing.publish(reference)
    check = planner._execution_test_consumer.consume(reference.as_reader(), previous.producer_ns + 1_000,
                                                     state.logMonoTime['carControl'] - 1_000)
    motion.referenceTimingVersion = REFERENCE_TIMING_VERSION
    motion.referenceTimingValid = check.valid
    motion.referenceConsumerMonoTime = check.now_ns
    if check.timing is not None:
      motion.referenceProducerMonoTime = check.timing.producer_ns
      motion.referenceProducerInstanceMonoTime = check.timing.instance_ns
      motion.referenceSourceMonoTime = check.timing.source_ns
      motion.referenceEpochMonoTime = check.timing.epoch_ns
  return now_ns


def solution(planner, profile, *, initial_speed=0.02):
  # Inject a successful, dynamics-consistent trajectory to isolate the
  # producer decision. These tests do not certify native solver behaviour.
  acceleration = np.full(len(T_IDXS), 0.18)
  if profile == 'slow':
    acceleration *= np.maximum(1.0 - T_IDXS / T_IDXS[4], 0.0)
  elif profile == 'departure_then_slow':
    acceleration[:] = 0.0
    acceleration[4:6] = 0.5
    acceleration[7:10] = -0.1635
  planner.mpc.a_solution[:] = acceleration
  planner.mpc.j_solution[:] = np.diff(acceleration) / np.diff(T_IDXS)
  planner.mpc.v_solution[:] = np.concatenate(([initial_speed], initial_speed + np.cumsum(
    0.5 * (acceleration[:-1] + acceleration[1:]) * np.diff(T_IDXS))))
  planner.mpc.solution_status = 0


def context(monkeypatch, *, warm=True, raw_forward=True):
  planner, _ = make_planner(speed=0.02)
  state = execution_state(raw_forward=raw_forward)
  planner._execution_test_consumer = ReferenceTimingConsumer()
  monkeypatch.setattr(planner.mpc, 'update', lambda *args, **kwargs: None)
  if warm:
    solution(planner, 'launch')
    planner.update(state, now_ns=execution_time(planner, state, 0))
    assert planner.motion_intent.valid
    assert not planner.output_should_stop
    assert planner.motion_intent.release_allowed is raw_forward
  solution(planner, 'slow')
  return planner, state


@pytest.mark.parametrize('profile', ['slow', 'departure_then_slow'])
@pytest.mark.parametrize('phase', ['release', 'track'])
def test_executed_forward_does_not_stop_or_gain_new_release_from_low_speed_preference(monkeypatch, profile, phase):
  planner, state = context(monkeypatch)
  solution(planner, profile)
  state['carState'].aEgo = float(planner.mpc.a_solution[0])
  state['controlsState'].motionState.phase = phase
  planner.update(state, now_ns=execution_time(planner, state, 1))
  assert planner.motion_intent.valid
  assert planner.motion_intent.reference_kind == MotionReferenceKind.FORWARD
  assert state['controlsState'].motionState.referenceTimingValid
  assert 0.0 <= planner.output_a_target < STOP_ACCEL
  assert planner.mpc.v_solution[-1] < STOP_SPEED
  # Even when the trajectory otherwise proves departure, its old stop mask
  # still vetoes release. Removing MPC from the admission mask breaks this.
  assert reference_requests_departure(planner.mpc.v_solution, planner.mpc.a_solution, planner.mpc.j_solution) == (profile != 'slow')
  assert not planner.output_should_stop
  assert not planner.motion_intent.reasons
  assert not planner.motion_intent.release_allowed
  assert not planner.motion_intent.moving_lead


def test_moving_only_continues_an_executed_forward_session(monkeypatch):
  planner, state = context(monkeypatch)
  planner.update(state, now_ns=execution_time(planner, state, 1))
  assert not planner.output_should_stop
  state['modelV2'].velocity.x = [3.0] * len(ModelConstants.T_IDXS)
  for frame in (2, 3):
    planner.update(state, now_ns=execution_time(planner, state, frame))
    assert planner.motion_intent.reference_kind == MotionReferenceKind.MOVING
    assert not planner.output_should_stop
    assert not planner.motion_intent.release_allowed

  other, other_state = context(monkeypatch, raw_forward=False)
  other.update(other_state, now_ns=execution_time(other, other_state, 1))
  assert other.motion_intent.reference_kind == MotionReferenceKind.MOVING
  assert other.output_should_stop
  assert not other.motion_intent.release_allowed


def test_new_producer_cannot_treat_pid_fields_as_reference_history(monkeypatch):
  planner, state = context(monkeypatch, warm=False)
  planner.update(state, now_ns=execution_time(planner, state, 0))
  assert planner.motion_intent.reference_kind == MotionReferenceKind.FORWARD
  assert planner.output_should_stop
  assert not planner.motion_intent.release_allowed


@pytest.mark.parametrize('control_state', [LongCtrlState.off, LongCtrlState.stopping])
@pytest.mark.parametrize('raw_speed', [0.0, 3.0])
def test_cold_off_or_hold_with_flat_raw_input_cannot_acquire_permission(monkeypatch, control_state, raw_speed):
  planner, _ = context(monkeypatch, warm=False)
  state = execution_state(raw_forward=False, speed=0.0)
  state['modelV2'].velocity.x = [raw_speed] * len(ModelConstants.T_IDXS)
  state['controlsState'].longControlState = control_state
  state['carControl'].actuators.longControlState = control_state
  state['controlsState'].motionState.phase = 'off' if control_state == LongCtrlState.off else 'hold'
  solution(planner, 'slow', initial_speed=0.0)
  for frame in (0, 1):
    planner.update(state, now_ns=execution_time(planner, state, frame))
    assert planner.output_should_stop
    assert not planner.motion_intent.release_allowed
    assert planner.reference_planner._forward_execution_instance_ns == 0


@pytest.mark.parametrize('veto', [
  'leadOne', 'leadTwo', 'rest', 'no_forward', 'ambiguous_raw', 'negative_output', 'action_stop', 'force_decel', 'cruise_stop', 'fcw',
  'controls_off', 'controls_hold', 'control_hold', 'inactive', 'disabled', 'gas', 'brake', 'override', 'unhealthy',
  'resolved_stop', 'degraded', 'waiting', 'previous_reasons', 'nonfinite_control', 'alignment_unknown',
])
def test_current_veto_clears_continuation_and_moving_cannot_rearm_it(monkeypatch, veto):
  planner, state = context(monkeypatch)
  planner.update(state, now_ns=execution_time(planner, state, 1))
  assert not planner.output_should_stop
  now_ns = execution_time(planner, state, 2)
  if veto in ('leadOne', 'leadTwo'):
    lead = getattr(state['radarState'], veto)
    lead.present, lead.dRel, lead.vLead, lead.vRel = True, 50.0, 0.0, -state['carState'].vEgo
    lead.aLeadK, lead.aLeadTau, lead.modelProb = 0.0, 1.5, 0.99
  elif veto == 'rest':
    state['modelV2'].velocity.x = np.maximum(0.2 - np.asarray(ModelConstants.T_IDXS), 0.0).tolist()
  elif veto == 'no_forward':
    state['modelV2'].velocity.x = [-0.1] * len(ModelConstants.T_IDXS)
  elif veto == 'ambiguous_raw':
    state['modelV2'].velocity.x = [-0.01] + [3.0] * (len(ModelConstants.T_IDXS) - 1)
  elif veto == 'negative_output':
    state['modelV2'].action.desiredAcceleration = -0.01
  elif veto == 'action_stop':
    state['modelV2'].action.shouldStop = True
  elif veto == 'force_decel':
    state['controlsState'].forceDecel = True
  elif veto == 'cruise_stop':
    state['carState'].vCruise = state['carState'].vCruiseCluster = 0.0
  elif veto == 'fcw':
    planner.mpc.crash_cnt = 3
  elif veto in ('controls_off', 'controls_hold'):
    state['controlsState'].longControlState = LongCtrlState.off if veto == 'controls_off' else LongCtrlState.stopping
  elif veto == 'control_hold':
    state['carControl'].actuators.longControlState = LongCtrlState.stopping
  elif veto == 'inactive':
    state['carControl'].longActive = False
  elif veto == 'disabled':
    state['selfdriveState'].enabled = False
  elif veto in ('gas', 'brake'):
    setattr(state['carState'], veto + 'Pressed', True)
  elif veto == 'override':
    state['carControl'].cruiseControl.override = True
  elif veto == 'unhealthy':
    monkeypatch.setattr(state, 'all_checks', lambda service_list=None: False)
  elif veto in ('resolved_stop', 'degraded', 'waiting', 'previous_reasons'):
    field = {'resolved_stop': 'resolvedStop', 'degraded': 'degraded', 'waiting': 'waitingForFreshInput', 'previous_reasons': 'reasons'}[veto]
    setattr(state['controlsState'].motionState, field, int(StopReason.MPC) if veto == 'previous_reasons' else True)
  elif veto == 'nonfinite_control':
    state['carControl'].actuators.accel = float('nan')
  elif veto == 'alignment_unknown':
    state.logMonoTime['modelV2'] = now_ns + 500_000
    now_ns += 1_000_000
  planner.update(state, now_ns=now_ns)
  assert planner.output_should_stop
  assert not planner.motion_intent.release_allowed
  assert planner.reference_planner._forward_execution_instance_ns == 0
  if veto == 'rest':
    assert planner.motion_intent.reference_kind == MotionReferenceKind.REST
    assert planner.motion_intent.reasons & StopReason.MPC
  if veto == 'alignment_unknown':
    assert planner.reference_planner.model_evidence.unknown.reason == 'model_proxy_after_initial_state'

  planner.mpc.crash_cnt = 0
  solution(planner, 'slow')
  recovered = execution_state(raw_forward=False)
  for frame in (3, 4):
    planner.update(recovered, now_ns=execution_time(planner, recovered, frame))
    assert planner.output_should_stop
    assert not planner.motion_intent.release_allowed


@pytest.mark.parametrize('phase', ['', 'off', 'hold', 'approach', 'degraded', 'unknown'])
def test_unknown_or_nonexecuting_owner_phase_clears_continuation(monkeypatch, phase):
  planner, state = context(monkeypatch)
  planner.update(state, now_ns=execution_time(planner, state, 1))
  assert not planner.output_should_stop
  now_ns = execution_time(planner, state, 2)
  state['controlsState'].motionState.phase = phase
  planner.update(state, now_ns=now_ns)
  assert planner.output_should_stop
  assert not planner.motion_intent.release_allowed
  assert planner.reference_planner._forward_execution_instance_ns == 0
  recovered = execution_state(raw_forward=False)
  planner.update(recovered, now_ns=execution_time(planner, recovered, 3))
  assert planner.output_should_stop


@pytest.mark.parametrize('fault', [
  'missing_timing', 'missing_version', 'wrong_version', 'invalid_timing', 'missing_instance', 'wrong_instance',
  'old_producer', 'old_source', 'old_epoch', 'missing_consumer', 'old_consumer', 'consumer_after_control', 'consumer_after_controls',
])
def test_recent_pid_headers_cannot_substitute_for_same_reference_consumption(monkeypatch, fault):
  planner, state = context(monkeypatch)
  planner.update(state, now_ns=execution_time(planner, state, 1))
  assert not planner.output_should_stop
  now_ns = execution_time(planner, state, 2)
  motion = state['controlsState'].motionState
  assert motion.referenceTimingValid
  if fault == 'missing_timing':
    motion.referenceTimingVersion = 0
    motion.referenceTimingValid = False
    for field in ('referenceConsumerMonoTime', 'referenceProducerMonoTime', 'referenceProducerInstanceMonoTime',
                  'referenceSourceMonoTime', 'referenceEpochMonoTime'):
      setattr(motion, field, 0)
  elif fault == 'missing_version':
    motion.referenceTimingVersion = 0
  elif fault == 'wrong_version':
    motion.referenceTimingVersion = REFERENCE_TIMING_VERSION + 1
  elif fault == 'invalid_timing':
    motion.referenceTimingValid = False
  elif fault == 'missing_instance':
    motion.referenceProducerInstanceMonoTime = 0
  elif fault == 'wrong_instance':
    motion.referenceProducerInstanceMonoTime -= 1
  elif fault == 'old_producer':
    motion.referenceProducerMonoTime -= 50_000_000
  elif fault == 'old_source':
    motion.referenceSourceMonoTime -= 50_000_000
  elif fault == 'old_epoch':
    motion.referenceEpochMonoTime -= 50_000_000
  elif fault == 'missing_consumer':
    motion.referenceConsumerMonoTime = 0
  elif fault == 'old_consumer':
    motion.referenceConsumerMonoTime = motion.referenceProducerMonoTime - 1
  elif fault == 'consumer_after_control':
    motion.referenceConsumerMonoTime = state.logMonoTime['carControl'] + 1
  elif fault == 'consumer_after_controls':
    state.logMonoTime['controlsState'] = motion.referenceConsumerMonoTime - 1
  planner.update(state, now_ns=now_ns)
  assert planner.output_should_stop
  assert not planner.motion_intent.release_allowed
  assert planner.reference_planner._forward_execution_instance_ns == 0
  recovered = execution_state(raw_forward=False)
  planner.update(recovered, now_ns=execution_time(planner, recovered, 3))
  assert planner.output_should_stop


@pytest.mark.parametrize('service', ['carControl', 'controlsState'])
@pytest.mark.parametrize('invalid_time', [None, 0, 1_000_000_000, 2_000_000_000])
def test_missing_stale_or_future_control_publication_is_not_execution(monkeypatch, service, invalid_time):
  planner, state = context(monkeypatch)
  planner.update(state, now_ns=execution_time(planner, state, 1))
  now_ns = execution_time(planner, state, 2)
  if invalid_time is None:
    del state.logMonoTime[service]
  else:
    state.logMonoTime[service] = invalid_time
  planner.update(state, now_ns=now_ns)
  assert planner.output_should_stop
  assert not planner.motion_intent.release_allowed
  assert planner.reference_planner._forward_execution_instance_ns == 0


@pytest.mark.parametrize('failure', ['source_clock', 'owner', 'vehicle', 'solver', 'sampled_accel'])
def test_early_failure_clears_execution_before_a_later_moving_reference(monkeypatch, failure):
  planner, state = context(monkeypatch)
  planner.update(state, now_ns=execution_time(planner, state, 1))
  now_ns = execution_time(planner, state, 2)
  if failure == 'source_clock':
    state.logMonoTime['carState'] = now_ns + 1
  elif failure == 'owner':
    state['controlsState'].motionState.enabled = False
  elif failure == 'vehicle':
    state['carState'].steeringAngleDeg = float('nan')
  elif failure == 'solver':
    planner.mpc.solution_status = 4
    planner.mpc.failure_reason = 'native_failure_fixture'
  elif failure == 'sampled_accel':
    planner.mpc.v_solution[:] = float('nan')
  planner.update(state, now_ns=now_ns)
  assert not planner.motion_intent.valid
  assert planner.reference_planner._forward_execution_instance_ns == 0
  solution(planner, 'slow')
  recovered = execution_state(raw_forward=False)
  for frame in (3, 4):
    planner.update(recovered, now_ns=execution_time(planner, recovered, frame))
    assert planner.output_should_stop
    assert not planner.motion_intent.release_allowed


@pytest.mark.parametrize('interruption', ['gap', 'consumer_regression', 'instance'])
def test_reference_discontinuity_cannot_preserve_execution(monkeypatch, interruption):
  planner, state = context(monkeypatch)
  planner.update(state, now_ns=execution_time(planner, state, 1))
  state['modelV2'].velocity.x = [3.0] * len(ModelConstants.T_IDXS)
  now_ns = execution_time(planner, state, 8 if interruption == 'gap' else 2)
  if interruption == 'consumer_regression':
    now_ns = 1_050_000_000
  elif interruption == 'instance':
    planner.reference_timing.instance_ns += 1
  planner.update(state, now_ns=now_ns)
  assert planner.reference_planner._forward_execution_instance_ns == 0
  assert not planner.motion_intent.release_allowed
  assert planner.output_should_stop or not planner.motion_intent.valid


@pytest.mark.parametrize('platform', [CAR.HONDA_CIVIC, CAR.HONDA_CIVIC_BOSCH, CAR.HONDA_CRV_HYBRID])
def test_generic_legacy_platforms_do_not_construct_a_unified_continuation_owner(platform):
  planner, _ = make_planner(platform=platform)
  assert planner.longitudinal_mode == LongitudinalMode.LEGACY
  assert not planner.unified
  assert not hasattr(planner, 'reference_planner')


@pytest.mark.parametrize('mode', [LongitudinalMode.LEGACY, LongitudinalMode.SUPERVISED])
def test_wrong_owner_wire_mode_revokes_continuation_and_preserves_consumers_hold(monkeypatch, mode):
  planner, state = context(monkeypatch)
  planner.update(state, now_ns=execution_time(planner, state, 1))
  assert not planner.output_should_stop
  assert planner.reference_planner._forward_execution_instance_ns > 0
  now_ns = execution_time(planner, state, 2)
  state['controlsState'].motionState.planningMode = int(mode)
  planner.update(state, now_ns=now_ns)
  assert planner.longitudinal_mode == LongitudinalMode.UNIFIED
  assert not planner.motion_intent.valid
  assert not planner.motion_intent.release_allowed
  assert planner.reference_planner._forward_execution_instance_ns == 0
  assert planner.observation_result['values']['outcome'] == 'motion_owner_unavailable'

  consumer = ControlsHarness()
  consumer.hold()
  consumer.plan.motionIntent.mode = int(mode)
  consumer.plan.motionIntent.reasons = 0
  consumer.plan.motionIntent.releaseAllowed = True
  consumer.plan.aTarget = 1.0
  command, motion = consumer.step()
  assert consumer.control.longitudinal_mode_mismatch
  assert command.longActive
  assert command.actuators.accel <= -2.0
  assert not command.cruiseControl.resume
  assert motion.degraded and motion.resolvedStop
