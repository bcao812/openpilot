from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.tools.longitudinal import replay_pedal_switches as replay
from openpilot.tools.longitudinal.replay_pedal_switches import replay_window, switch_metrics


def test_legacy_positional_experiment_switch_is_rejected():
  with pytest.raises(TypeError):
    replay_window(None, {}, True)


def pedal_rows(sides, interval=0.02, offset=0.0):
  return [{'t': offset + index * interval, 'automatic': True, 'can_brake': side == -1,
           'can_gas': -30000 if side == -1 else 7 if side == 1 else 0} for index, side in enumerate(sides)]


def test_native_dwell_and_short_pulses():
  rows = pedal_rows([1] * 10 + [-1] * 3 + [1] * 10 + [-1] * 10 + [1] * 10)
  assert switch_metrics(rows)['debounced'] == 2
  assert switch_metrics(rows)['raw'] == 4


@pytest.mark.parametrize('boundary', ['override', 'gap', 'neutral', 'conflict'])
def test_never_join_across_ineligible_boundaries(boundary):
  rows = pedal_rows([1] * 10 + [0] * 10 + [-1] * 10)
  for row in rows[10:20]:
    if boundary == 'override':
      row['automatic'] = False
    elif boundary == 'conflict':
      row.update(can_brake=True, can_gas=7)
  if boundary == 'gap':
    rows = rows[:10] + rows[20:]
  assert switch_metrics(rows)['debounced'] == 0
  assert switch_metrics(rows)['raw'] == 0


def test_short_neutral_can_bridge_confirmed_sides():
  rows = pedal_rows([1] * 10 + [0] * 3 + [-1] * 10)
  assert switch_metrics(rows)['debounced'] == 1


def test_neutral_does_not_manufacture_phase_dwell():
  rows = pedal_rows([1] * 10 + [-1] * 3 + [0] * 3 + [-1] * 3)
  assert switch_metrics(rows)['debounced'] == 0


def test_empty_and_partial_final_phase():
  assert switch_metrics([])['debounced'] == 0
  assert switch_metrics(pedal_rows([1] * 10 + [-1] * 7))['debounced'] == 0


def reference_replay_fixture(monkeypatch, *, delay=0.5, constraint=1.6, repeat=None, fault=None, solver_status=0, health_fault=None):
  from openpilot.cereal import messaging
  from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
  from openpilot.selfdrive.controls.lib.reference_timing import REFERENCE_TIMING_VERSION
  from openpilot.selfdrive.modeld.constants import ModelConstants
  from openpilot.tools.longitudinal.tests.test_unified_replay import replay_fixture

  decode = replay.decode_motion_intent
  messages, window, controls, _, _ = replay_fixture(monkeypatch, solver_status=solver_status, envelope_api=True)
  monkeypatch.setattr(replay, 'decode_motion_intent', decode)
  messages[0].carParams.longitudinalActuatorDelay = delay
  health = {service: {'seen': True, 'alive': True, 'frequency_ok': True} for service in ('carState', 'longitudinalPlan')}
  if health_fault is not None:
    health[health_fault[0]][health_fault[1]] = False
  monkeypatch.setattr(replay, 'ReplayConsumerHealth', lambda: SimpleNamespace(observe=lambda *args: None, at=lambda now: health))
  times = np.asarray(ModelConstants.T_IDXS[:CONTROL_N])
  publications = []

  def publish(instance, state, publisher):
    event = messaging.new_message('longitudinalPlan')
    event.valid = solver_status == 0 and fault != 'publication'
    plan = event.longitudinalPlan
    plan.modelMonoTime = state.logMonoTime['modelV2']
    plan.aTarget, plan.shouldStop, plan.solverStatus = 0.15, False, solver_status
    speeds, accels = 0.2 * times ** 2, 0.4 * times
    if publications and repeat in ('speeds', 'accels'):
      (speeds if repeat == 'speeds' else accels)[-1] += 0.1
    plan.speeds, plan.accels = speeds.tolist(), accels.tolist()
    intent = plan.motionIntent
    intent.version, intent.mode, intent.ready, intent.valid = replay.MOTION_PROTOCOL_VERSION, 2, True, solver_status == 0
    intent.releaseAllowed = True
    intent.producerInstanceMonoTime = 1_000_000_000 if fault == 'instance' else 1_010_000_000
    intent.brakingValid, intent.brakingAccel = solver_status != 0, -2.5
    reference = plan.reference
    source = 700_000_000 if fault == 'source_age' else state.logMonoTime['carState']
    reference.timingVersion, reference.timingKnown = REFERENCE_TIMING_VERSION, fault != 'unknown'
    reference.sourceCarStateMonoTime = reference.referenceMonoTime = source
    reference.producerMonoTime = reference.producerInstanceMonoTime = 1_010_000_000
    reference.sourceVEgo = reference.sourceAEgo = reference.initialVEgo = reference.initialAEgo = 0.0
    reference.initialStateKnown = True
    reference.accel, reference.constraintAccel = plan.aTarget, constraint
    publications.append(event.as_reader())
    publisher.send('longitudinalPlan', event)

  monkeypatch.setattr(replay.planner_module.LongitudinalPlanner, 'publish', publish)
  if repeat is not None:
    for service, mono in (('longitudinalPlan', 1_030_000_000), ('longitudinalPlanSP', 1_031_000_000),
                          ('controlsState', 1_034_000_000), ('carControl', 1_035_000_000)):
      original = next(message for message in messages if message.which() == service and message.logMonoTime >= 1_000_000_000)
      message = deepcopy(original)
      message.logMonoTime = mono
      if service == 'controlsState':
        message.controlsState.longitudinalPlanMonoTime = 1_030_000_000
      messages.append(message)
    messages.sort(key=lambda message: message.logMonoTime)
  return messages, window, controls, publications, times


@pytest.mark.parametrize('delay', [0.1, 0.5])
@pytest.mark.parametrize('constraint', [0.15, 1.6])
def test_replay_consumes_published_departure_strength_without_replacing_target(monkeypatch, delay, constraint):
  from openpilot.common.realtime import DT_MDL
  from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan

  messages, window, controls, publications, times = reference_replay_fixture(monkeypatch, delay=delay, constraint=constraint)
  replay_window(None, window, messages=messages, unified=True)
  plan = publications[0].longitudinalPlan
  args, options = controls[0]
  expected = min(get_accel_from_plan(plan.speeds, plan.accels, times, delay + DT_MDL), plan.reference.constraintAccel)
  assert options['intent'].release_accel == pytest.approx(expected)
  assert args[2] == plan.aTarget
  assert options['plan_valid'] and options['braking_plan_valid']
  assert replay.decode_motion_intent(plan).release_accel is None


@pytest.mark.parametrize('repeat', ['unchanged', 'speeds', 'accels'])
def test_replay_repeated_producer_identity_covers_unused_trajectory_samples(monkeypatch, repeat):
  messages, window, controls, _, _ = reference_replay_fixture(monkeypatch, repeat=repeat)
  replay_window(None, window, messages=messages, unified=True)
  assert len(controls) == 2
  first, second = (entry[1] for entry in controls)
  assert first['intent'].release_accel == second['intent'].release_accel
  assert first['plan_valid'] and first['braking_plan_valid']
  assert second['plan_valid'] == (repeat == 'unchanged')
  assert second['braking_plan_valid'] == (repeat == 'unchanged')


@pytest.mark.parametrize('fault', ['source_age', 'instance', 'unknown'])
def test_replay_invalid_reference_cannot_authorize_a_plan_or_braking_envelope(monkeypatch, fault):
  messages, window, controls, _, _ = reference_replay_fixture(monkeypatch, fault=fault)
  replay_window(None, window, messages=messages, unified=True)
  options = controls[0][1]
  assert not options['plan_valid'] and not options['braking_plan_valid']
  assert options['now_ns'] == 1_020_000_000


@pytest.mark.parametrize('solver_status,fault', [(4, None), (0, 'publication')])
def test_replay_valid_reference_preserves_independent_braking_health(monkeypatch, solver_status, fault):
  messages, window, controls, _, _ = reference_replay_fixture(monkeypatch, solver_status=solver_status, fault=fault)
  replay_window(None, window, messages=messages, unified=True)
  options = controls[0][1]
  assert not options['plan_valid']
  assert options['braking_plan_valid']


@pytest.mark.parametrize('service', ['carState', 'longitudinalPlan'])
@pytest.mark.parametrize('flag', ['seen', 'alive', 'frequency_ok'])
def test_replay_reference_strength_requires_both_services_healthy(monkeypatch, service, flag):
  messages, window, controls, _, _ = reference_replay_fixture(monkeypatch, health_fault=(service, flag))
  replay_window(None, window, messages=messages, unified=True)
  options = controls[0][1]
  assert options['intent'].release_accel > 0.0
  assert options['now_ns'] == 1_020_000_000
  assert not options['plan_valid']
  assert not options['braking_plan_valid']


def test_replay_impossible_reference_clock_does_not_advance_the_motion_owner(monkeypatch):
  from openpilot.selfdrive.controls.lib.reference_timing import ReferenceTimingConsumer

  messages, window, controls, _, _ = reference_replay_fixture(monkeypatch)
  consume = ReferenceTimingConsumer.consume

  def impossible_clock(instance, reference, available_ns, now_ns, **options):
    return consume(instance, reference, available_ns, available_ns - 1, **options)

  monkeypatch.setattr(ReferenceTimingConsumer, 'consume', impossible_clock)
  replay_window(None, window, messages=messages, unified=True)
  options = controls[0][1]
  assert options['now_ns'] == 0
  assert not options['plan_valid'] and not options['braking_plan_valid']
