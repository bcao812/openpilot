"""Recovery is new source and consumer evidence, never a newer message header."""

from types import SimpleNamespace as NS

import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION
from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_MAX_AGE_NS, REFERENCE_TIMING_VERSION
from openpilot.selfdrive.selfdrived.events import Events
from openpilot.selfdrive.selfdrived.model_lag_recovery import CLEAN_WINDOW_NS, MAX_MODEL_GAP_NS, ModelLagRecovery
from openpilot.selfdrive.selfdrived.state import StateMachine


NOW = 10_000_000_000
MS = 1_000_000
FRAME_NS = 50 * MS
State = log.SelfdriveState.OpenpilotState
EventName = log.OnroadEvent.EventName


class Streams(dict):
  def __init__(self):
    super().__init__(modelV2=NS(frameId=47454, frameDropPerc=1.470588),
                     deviceMotion=NS(inputsOK=True, sensorsOK=True, posenetOK=True),
                     controlsState=NS(forceDecel=False, motionState=NS(
                       version=MOTION_INTENT_VERSION, planningMode=int(LongitudinalMode.UNIFIED), enabled=True,
                       referenceTimingVersion=REFERENCE_TIMING_VERSION, referenceTimingValid=True,
                       degraded=False, waitingForFreshInput=False, continuationActive=False,
                       phase='track', reasons=0, resolvedStop=False)))
    self.instance_ns = NOW - 1_000 * MS
    self['longitudinalPlan'] = None
    for flag in ('seen', 'valid', 'alive', 'freq_ok'):
      setattr(self, flag, dict.fromkeys(self, True))
    self.logMonoTime = dict.fromkeys(self, NOW)
    publish(self, NOW, percent=1.470588)


def consume(sm, now_ns, plan=None, plan_ns=None):
  plan = sm['longitudinalPlan'] if plan is None else plan
  plan_ns = sm.logMonoTime['longitudinalPlan'] if plan_ns is None else plan_ns
  motion, reference = sm['controlsState'].motionState, plan.reference
  motion.planMonoTime = plan_ns
  motion.modelMonoTime = plan.modelMonoTime
  motion.referenceSourceMonoTime = reference.sourceCarStateMonoTime
  motion.referenceEpochMonoTime = reference.referenceMonoTime
  motion.referenceProducerMonoTime = reference.producerMonoTime
  motion.referenceProducerInstanceMonoTime = reference.producerInstanceMonoTime
  motion.referenceConsumerMonoTime = now_ns
  sm.logMonoTime['controlsState'] = now_ns


def publish(sm, now_ns, *, frame_delta=1, percent=None, consume_plan=True):
  model = sm['modelV2']
  model.frameId += frame_delta
  model.timestampEof = now_ns - 40 * MS
  model.frameDropPerc = model.frameDropPerc * 0.995 if percent is None else percent
  sm.logMonoTime['modelV2'] = now_ns - 3 * MS
  sm.logMonoTime['deviceMotion'] = now_ns
  sm.logMonoTime['longitudinalPlan'] = now_ns - MS
  sm['longitudinalPlan'] = NS(
    modelMonoTime=sm.logMonoTime['modelV2'], solverStatus=0,
    motionIntent=NS(version=MOTION_INTENT_VERSION, mode=int(LongitudinalMode.UNIFIED), valid=True, ready=True,
                    producerInstanceMonoTime=sm.instance_ns),
    reference=NS(timingVersion=REFERENCE_TIMING_VERSION, timingKnown=True,
                 sourceCarStateMonoTime=now_ns - 5 * MS, referenceMonoTime=now_ns - 5 * MS,
                 producerMonoTime=now_ns - 2 * MS, producerInstanceMonoTime=sm.instance_ns,
                 sourceVEgo=15.0, sourceAEgo=0.0, initialVEgo=15.0, initialAEgo=0.0, initialStateKnown=True))
  if consume_plan:
    consume(sm, now_ns)


def advance(observer, sm, now_ns, frames=24, edit=None):
  states = []
  for _ in range(frames):
    now_ns += FRAME_NS
    publish(sm, now_ns)
    if edit is not None:
      edit(sm, now_ns)
    states.append(observer.update(sm, now_ns, lagging=True))
  return now_ns, states


def started():
  observer, sm = ModelLagRecovery(), Streams()
  assert not observer.update(sm, NOW, lagging=True)
  return observer, sm


def recovered():
  observer, sm = started()
  now_ns, states = advance(observer, sm, NOW)
  assert states[-1]
  return observer, sm, now_ns


def test_three_frame_burst_recovers_before_exit_while_history_metric_remains_high():
  observer, sm = started()
  assert sm['modelV2'].frameId == 47455
  now_ns, states = advance(observer, sm, NOW, frames=75)
  first_recovered_ns = (states.index(True) + 1) * FRAME_NS
  assert CLEAN_WINDOW_NS <= first_recovered_ns < 3_000 * MS
  assert all(states[states.index(True):])
  assert now_ns - NOW > 3_000 * MS
  assert sm['modelV2'].frameDropPerc > 1.0


@pytest.mark.parametrize('burst_period', [5, 50])
def test_sustained_and_half_second_bursts_cannot_restart_the_original_exit_timer(burst_period):
  observer, sm = started()
  machine, events = StateMachine(), Events()
  machine.state = State.enabled
  for tick in range(1, 351):
    now_ns = NOW + tick * 10 * MS
    if tick % 5 == 0:
      publish(sm, now_ns, frame_delta=2 if tick % burst_period == 0 else 1)
    consume(sm, now_ns)
    assert not observer.update(sm, now_ns, lagging=True)
    events.clear()
    events.add(EventName.modeldLagging)
    machine.update(events)
    if tick == 301:
      assert machine.state == State.disabled
  assert machine.state == State.disabled


def test_normal_100hz_reuse_does_not_add_recovery_frames():
  observer, sm = started()
  for tick in range(1, 10):
    now_ns = NOW + tick * 10 * MS
    if tick == 5:
      publish(sm, now_ns)
    consume(sm, now_ns)
    assert not observer.update(sm, now_ns, lagging=True)
    assert observer.good_frames == int(tick >= 5)


def test_stalled_model_with_fresh_consumer_header_revokes_at_100ms():
  observer, sm, now_ns = recovered()
  consume(sm, now_ns + 50 * MS)
  assert observer.update(sm, now_ns + 50 * MS, lagging=True)
  deadline_ns = sm.logMonoTime['modelV2'] + MAX_MODEL_GAP_NS
  consume(sm, deadline_ns)
  assert observer.update(sm, deadline_ns, lagging=True)
  consume(sm, deadline_ns + 1)
  assert not observer.update(sm, deadline_ns + 1, lagging=True)
  assert observer.clean_since_ns == 0


@pytest.mark.parametrize('frozen_field', ['frameId', 'timestampEof'])
def test_republishing_old_frame_identity_or_capture_never_proves_recovery(frozen_field):
  observer, sm = started()
  original = getattr(sm['modelV2'], frozen_field)

  def republish(streams, _):
    setattr(streams['modelV2'], frozen_field, original)

  _, states = advance(observer, sm, NOW, frames=70, edit=republish)
  assert not any(states)


@pytest.mark.parametrize('frame_delta,increase', [(4, False), (1, True)])
def test_a_new_drop_or_drop_metric_increase_immediately_revokes_recovery(frame_delta, increase):
  observer, sm, now_ns = recovered()
  old_percent = sm['modelV2'].frameDropPerc
  now_ns += FRAME_NS
  publish(sm, now_ns, frame_delta=frame_delta, percent=old_percent + 0.1 if increase else None)
  assert not observer.update(sm, now_ns, lagging=True)
  assert observer.good_frames == 0


@pytest.mark.parametrize('target,field,bad', [
  ('deviceMotion', 'inputsOK', False), ('deviceMotion', 'sensorsOK', False), ('deviceMotion', 'posenetOK', False),
  ('motion', 'degraded', True), ('motion', 'waitingForFreshInput', True), ('motion', 'continuationActive', True),
  ('motion', 'degraded', None), ('motion', 'waitingForFreshInput', None), ('motion', 'continuationActive', None),
  ('motion', 'enabled', False), ('motion', 'referenceTimingValid', False),
  ('motion', 'version', 0), ('motion', 'planningMode', int(LongitudinalMode.LEGACY)),
])
def test_current_input_fault_requires_a_new_full_clean_window(target, field, bad):
  observer, sm, now_ns = recovered()
  record = sm['controlsState'].motionState if target == 'motion' else sm[target]
  original = getattr(record, field)
  setattr(record, field, bad)
  now_ns += 10 * MS
  consume(sm, now_ns)
  assert not observer.update(sm, now_ns, lagging=True)
  assert observer.clean_since_ns == 0
  setattr(record, field, original)
  now_ns, states = advance(observer, sm, now_ns, frames=20)
  assert not any(states)
  _, states = advance(observer, sm, now_ns, frames=1)
  assert states == [True]


@pytest.mark.parametrize('service', ['modelV2', 'deviceMotion', 'longitudinalPlan', 'controlsState'])
@pytest.mark.parametrize('flag', ['seen', 'valid', 'alive', 'freq_ok'])
def test_unhealthy_service_cannot_gain_recovery(service, flag):
  observer, sm = started()
  getattr(sm, flag)[service] = False
  _, states = advance(observer, sm, NOW, frames=30)
  assert not any(states)


def test_missing_stream_metadata_revokes_existing_recovery():
  observer, sm, now_ns = recovered()
  del sm.logMonoTime['deviceMotion']
  assert not observer.update(sm, now_ns + 10 * MS, lagging=True)


def test_new_current_plan_can_be_ahead_of_the_last_healthy_consumed_plan():
  observer, sm, now_ns = recovered()
  prior_plan, prior_plan_ns = sm['longitudinalPlan'], sm.logMonoTime['longitudinalPlan']
  now_ns += FRAME_NS
  publish(sm, now_ns, consume_plan=False)
  consume(sm, now_ns, prior_plan, prior_plan_ns)
  assert sm['controlsState'].motionState.planMonoTime != sm.logMonoTime['longitudinalPlan']
  assert observer.update(sm, now_ns, lagging=True)
  consume(sm, now_ns + 10 * MS)
  assert observer.update(sm, now_ns + 10 * MS, lagging=True)


def test_prefault_receipt_does_not_confirm_recovery_or_erase_a_clean_window():
  observer, sm = started()
  old_plan, old_plan_ns = sm['longitudinalPlan'], sm.logMonoTime['longitudinalPlan']
  for frame in range(1, 5):
    now_ns = NOW + frame * FRAME_NS
    publish(sm, now_ns, consume_plan=False)
    consume(sm, now_ns, old_plan, old_plan_ns)
    assert not observer.update(sm, now_ns, lagging=True)
  clean_since_ns = observer.clean_since_ns
  assert clean_since_ns == NOW + FRAME_NS
  _, states = advance(observer, sm, now_ns, frames=17)
  assert states[-1]
  assert observer.clean_since_ns == clean_since_ns


def test_unknown_current_receipt_grants_nothing_until_the_identity_can_be_joined():
  observer, sm, now_ns = recovered()
  now_ns += FRAME_NS
  publish(sm, now_ns)
  motion = sm['controlsState'].motionState
  known_plan_ns = motion.planMonoTime
  motion.planMonoTime += 1
  clean_since_ns = observer.clean_since_ns
  assert not observer.update(sm, now_ns, lagging=True)
  assert observer.clean_since_ns == clean_since_ns
  motion.planMonoTime = known_plan_ns
  consume(sm, now_ns + 10 * MS)
  assert observer.update(sm, now_ns + 10 * MS, lagging=True)


@pytest.mark.parametrize('field', ['referenceSourceMonoTime', 'referenceEpochMonoTime', 'referenceProducerMonoTime',
                                 'referenceProducerInstanceMonoTime', 'modelMonoTime'])
def test_consumer_must_confirm_the_observed_reference_and_model_identity(field):
  observer, sm, now_ns = recovered()
  now_ns += FRAME_NS
  publish(sm, now_ns)
  motion = sm['controlsState'].motionState
  setattr(motion, field, getattr(motion, field) + 1)
  assert not observer.update(sm, now_ns, lagging=True)


def test_a_new_consumer_header_cannot_extend_an_old_consumed_source():
  observer, sm, now_ns = recovered()
  prior_plan, prior_plan_ns = sm['longitudinalPlan'], sm.logMonoTime['longitudinalPlan']
  old_model_ns = prior_plan.modelMonoTime
  states = []
  for _ in range(6):
    now_ns += FRAME_NS
    publish(sm, now_ns, consume_plan=False)
    consume(sm, now_ns, prior_plan, prior_plan_ns)
    states.append(observer.update(sm, now_ns, lagging=True))
  assert states[0]
  assert now_ns - old_model_ns > REFERENCE_MAX_AGE_NS
  assert not states[-1]
  assert observer.clean_since_ns == 0


def test_republishing_plans_cannot_renew_the_original_reference_lease():
  observer, sm = started()
  source_ns = sm['longitudinalPlan'].reference.sourceCarStateMonoTime

  def old_source(streams, now_ns):
    reference = streams['longitudinalPlan'].reference
    reference.sourceCarStateMonoTime = reference.referenceMonoTime = source_ns
    consume(streams, now_ns)

  _, states = advance(observer, sm, NOW, frames=30, edit=old_source)
  assert not any(states)
  assert observer.clean_since_ns == 0


@pytest.mark.parametrize('field,bad', [('solverStatus', 1), ('valid', False), ('ready', False), ('version', 0),
                                     ('mode', int(LongitudinalMode.LEGACY)), ('producerInstanceMonoTime', 1)])
def test_only_successful_current_native_plans_can_confirm_recovery(field, bad):
  observer, sm, now_ns = recovered()
  now_ns += FRAME_NS
  publish(sm, now_ns)
  plan = sm['longitudinalPlan']
  setattr(plan if field == 'solverStatus' else plan.motionIntent, field, bad)
  assert not observer.update(sm, now_ns, lagging=True)
  assert observer.clean_since_ns == 0


@pytest.mark.parametrize('bad_clock', [None, True, float(NOW), NOW - 1])
def test_invalid_or_backwards_clock_cannot_reuse_recovery(bad_clock):
  observer, sm, _ = recovered()
  assert not observer.update(sm, bad_clock, lagging=True)
  assert observer.clean_since_ns == 0


def test_repeated_observer_clock_is_not_new_recovery_evidence():
  observer, sm, now_ns = recovered()
  assert not observer.update(sm, now_ns, lagging=True)


def test_a_planner_restart_needs_a_new_clean_window_and_post_restart_receipt():
  observer, sm, now_ns = recovered()
  now_ns += FRAME_NS
  sm.instance_ns = now_ns - 2 * MS
  publish(sm, now_ns)
  assert not observer.update(sm, now_ns, lagging=True)
  assert observer.clean_since_ns == 0
  _, states = advance(observer, sm, now_ns)
  assert states[-1]


@pytest.mark.parametrize('phase', ['track', 'approach', 'hold', 'release'])
def test_force_decel_and_valid_stop_phases_can_prove_recovery(phase):
  observer, sm = started()
  sm['controlsState'].forceDecel = True
  sm['controlsState'].motionState.phase = phase
  sm['controlsState'].motionState.resolvedStop = phase == 'hold'
  sm['controlsState'].motionState.reasons = 1
  _, states = advance(observer, sm, NOW)
  assert states[-1]


@pytest.mark.parametrize('idle_ticks', [1, 20, 200])
def test_a_new_episode_cannot_inherit_recovery_from_a_previous_warning(idle_ticks):
  observer, sm, now_ns = recovered()
  for _ in range(idle_ticks):
    now_ns += 10 * MS
    assert not observer.update(sm, now_ns, lagging=False)
    assert not observer.recovered and observer.good_frames == 0
  assert not observer.update(sm, now_ns + 10 * MS, lagging=True)
  assert observer.clean_since_ns == 0


@pytest.mark.parametrize('bad_clock', [None, True, float(NOW), -1])
def test_invalid_first_warning_can_end_and_start_a_fresh_recovery(bad_clock):
  observer, sm = ModelLagRecovery(), Streams()
  assert not observer.update(sm, bad_clock, lagging=True)
  # With no warning, neither input health nor a clock is recovery evidence.
  for _ in range(20):
    assert not observer.update(None, None, lagging=False)
    assert not observer.recovered and observer.good_frames == 0
  assert not observer.update(sm, NOW, lagging=True)
  _, states = advance(observer, sm, NOW)
  assert states[-1]
