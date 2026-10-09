"""Native producer, real Controls/LongControl and Honda CAN continuation chain."""
from contextlib import ExitStack
from unittest.mock import Mock, patch

import pytest

from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_planner, make_state, Publisher
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import ControlsHarness, serialized
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_mpc_failure_chain import can_from_car_control


START = 2_000_000_000


class ContinuationChain:
  """Only transport clocks/health and the selected native failure are bounded."""
  def __init__(self):
    self.planner, _ = make_planner(speed=20.0)
    self.state = make_state(speed=20.0, target=25.0)
    self.state['carState'].canValid = True
    self.controls = ControlsHarness()
    self.controls.control.CP.longitudinalActuatorDelay = self.planner.CP.longitudinalActuatorDelay
    self.plan_event = None
    self.model_ns = START

  def produce(self, source_ns, status=0):
    self.model_ns = source_ns
    for service in self.state.logMonoTime:
      if service not in ('controlsState', 'carControl'):
        self.state.logMonoTime[service] = source_ns
    self.state['modelV2'].timestampEof = source_ns
    now = source_ns + 500_000
    if status == 4:
      original = self.planner.mpc.solver
      self.planner.mpc.solver = Mock(wraps=original)
      self.planner.mpc.solver.solve.return_value = 4
      try:
        self.planner.update(self.state, now_ns=now)
      finally:
        self.planner.mpc.solver = original
    elif status:
      with patch.object(self.planner.mpc, 'run', side_effect=lambda: self.planner.mpc.fail(status)):
        self.planner.update(self.state, now_ns=now)
    else:
      self.planner.update(self.state, now_ns=now)
    publisher = Publisher()
    with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=source_ns + 1_000_000):
      self.planner.publish(self.state, publisher)
    # Publisher exposes a reader; serialized() owns a mutable message while
    # clearing its write flag before the actual bytes round trip.
    self.plan_event = serialized(publisher.messages['longitudinalPlan'].as_builder())
    return self.plan_event

  def consume(self, sample_ns, *, model_ns=None, continuation_source=None):
    harness = self.controls
    now = sample_ns + 2_000_000
    harness.now_ns = now
    harness.events['carState'].carState = self.state['carState']
    harness.car_state = harness.events['carState'].carState
    harness.events['modelV2'].modelV2 = self.state['modelV2']
    incoming = []
    for service, event in harness.events.items():
      if service == 'longitudinalPlan':
        incoming.append(self.plan_event if continuation_source is None else continuation_source)
        continue
      event.logMonoTime = (self.model_ns if model_ns is None else model_ns) if service == 'modelV2' else sample_ns
      incoming.append(serialized(event))
    harness.control.sm.update_msgs(now / 1e9, incoming)
    # This fixture exercises admission/output rather than estimating a new
    # FrequencyTracker window from its short, intentionally bounded stream.
    harness.control.sm.freq_ok = dict.fromkeys(harness.control.sm.services, True)
    with patch('openpilot.selfdrive.controls.controlsd.time.monotonic_ns', return_value=now), \
         patch('openpilot.selfdrive.controls.controlsd.time.monotonic', return_value=now / 1e9):
      command, lateral = harness.control.state_control()
    # Real publication happens after consumption; avoid converting the same
    # exact nanosecond through a rounded float and making a fixture ACK earlier.
    with patch('openpilot.selfdrive.controls.controlsd.time.monotonic_ns', return_value=now + 1_000_000), \
         patch('openpilot.selfdrive.controls.controlsd.time.monotonic', return_value=(now + 1_000_000) / 1e9):
      harness.control.publish(command, lateral)
    for service in ('controlsState', 'carControl'):
      event = harness.control.pm.messages[service]
      self.state[service] = getattr(event, service).as_builder()
      self.state.logMonoTime[service] = event.logMonoTime
    return harness.command, harness.motion, can_from_car_control(harness.command, harness.car_state)

  def establish_native_donor(self):
    for offset in (0, 50_000_000):
      event = self.produce(START + offset)
      assert event.valid and event.longitudinalPlan.solverStatus == 0
      for tick in range(5):
        _, motion, _ = self.consume(START + offset + tick * 10_000_000)
        assert motion.phase == 'track' and not motion.degraded and not motion.continuationActive
    assert self.planner.reference_planner.continuation.donor is not None
    assert self.controls.control.LoC.continuation_donor is not None


@pytest.mark.parametrize('status', [4, -6])
def test_real_producer_owner_and_can_accept_two_bounded_failures_then_restore_original_fallback(status):
  chain = ContinuationChain()
  chain.establish_native_donor()
  previous = chain.controls.command.actuators.accel
  event = chain.produce(START + 100_000_000, status)
  assert event.longitudinalPlan.continuation.valid, chain.planner.reference_planner.continuation.reason
  original = event.to_dict()
  for tick in range(5):
    command, motion, can = chain.consume(START + 100_000_000 + tick * 10_000_000)
    assert motion.continuationActive, chain.controls.control.LoC.continuation_reason
    assert not motion.degraded and command.longActive and not command.cruiseControl.resume
    assert command.actuators.accel <= previous + 1e-7 and command.actuators.accel > -0.1
    assert can['ACCEL_COMMAND'] > -0.1 and can['BRAKE_REQUEST'] == 0
    assert not chain.controls.control.sm.valid['longitudinalPlan']
    assert event.longitudinalPlan.solverStatus == status and not event.longitudinalPlan.motionIntent.valid
    assert event.longitudinalPlan.aTarget <= -0.5
    assert all(value == 0.0 for value in event.longitudinalPlan.accels)
    assert event.to_dict() == original
    previous = command.actuators.accel
  first_deadline = event.longitudinalPlan.continuation.validUntilMonoTime
  second = chain.produce(START + 150_000_000, status)
  assert second.longitudinalPlan.continuation.valid
  assert second.longitudinalPlan.continuation.validUntilMonoTime == first_deadline
  command, motion, _ = chain.consume(START + 150_000_000)
  assert motion.continuationActive and command.actuators.accel > -0.1
  for offset in (160, 170, 180, 190):
    chain.consume(START + offset * 1_000_000)
  before_expiry = second.to_dict()
  command, motion, can = chain.consume(START + 200_000_000)
  assert not motion.continuationActive and motion.degraded
  assert command.actuators.accel <= -0.5 and can['ACCEL_COMMAND'] <= -0.5 and can['BRAKE_REQUEST'] == 1
  assert second.to_dict() == before_expiry


@pytest.mark.parametrize('change', ['model_arrives_first', 'cruise_reduced', 'force_decel', 'disabled', 'pedal', 'mutated_same_offer'])
def test_cached_offer_does_not_override_current_inputs_or_driver_authority(change):
  chain = ContinuationChain()
  chain.establish_native_donor()
  event = chain.produce(START + 100_000_000, 4)
  _, motion, _ = chain.consume(START + 100_000_000)
  assert motion.continuationActive, chain.controls.control.LoC.continuation_reason
  kwargs = {}
  if change == 'model_arrives_first':
    kwargs['model_ns'] = START + 110_000_000
  elif change == 'cruise_reduced':
    chain.state['carState'].vCruise -= 1.0
  elif change == 'force_decel':
    chain.controls.events['driverMonitoringState'].driverMonitoringState.noResponseForceDecel = True
  elif change == 'disabled':
    chain.controls.selfdrive_state.enabled = False
  elif change == 'pedal':
    chain.state['carState'].brakePressed = True
  else:
    changed = event.as_builder()
    changed.longitudinalPlan.continuation.candidate.targetAccel -= .1
    kwargs['continuation_source'] = serialized(changed)
  command, motion, can = chain.consume(START + 110_000_000, **kwargs)
  assert not motion.continuationActive and not command.cruiseControl.resume
  if change in ('disabled', 'pedal'):
    assert not command.longActive and command.actuators.accel == 0.0 and can['BRAKE_REQUEST'] == 0
  else:
    assert motion.degraded and command.actuators.accel <= -0.2 and can['ACCEL_COMMAND'] <= 0.0


def test_new_strong_risk_uses_current_producer_braking_instead_of_old_candidate():
  chain = ContinuationChain()
  chain.establish_native_donor()
  chain.produce(START + 100_000_000, 4)
  assert chain.consume(START + 100_000_000)[1].continuationActive
  for offset in (110, 120, 130, 140):
    chain.consume(START + offset * 1_000_000)
  chain.state['modelV2'].action.desiredAcceleration = -2.0
  chain.state['selfdriveState'].experimentalMode = True
  assert chain.planner.is_e2e(chain.state)
  failed = chain.produce(START + 150_000_000, 4)
  assert not failed.longitudinalPlan.continuation.valid
  command, motion, can = chain.consume(START + 150_000_000)
  assert not motion.continuationActive and motion.degraded
  assert failed.longitudinalPlan.aTarget <= -2.0 and command.actuators.accel <= -2.0
  assert can['ACCEL_COMMAND'] <= -2.0 and can['BRAKE_REQUEST'] == 1


def test_continuation_freezes_pid_and_confirmation_with_nonzero_state_error_until_native_recovery():
  chain = ContinuationChain()
  chain.establish_native_donor()
  chain.produce(START + 100_000_000, 4)
  assert chain.consume(START + 100_000_000)[1].continuationActive
  owner = chain.controls.control.LoC
  # This is a new measured carState, within the producer's proved envelope.
  chain.state['carState'].aEgo = .02
  before = (owner.pid.p, owner.pid.i, owner.pid.d, owner.pid.f)
  calls = []
  with ExitStack() as stack:
    for obj, names in ((owner.pid, ('update', 'reset')),
                       (owner.motion_supervisor, ('confirm_braking_output', 'complete_release'))):
      for name in names:
        calls.append(stack.enter_context(patch.object(obj, name, wraps=getattr(obj, name))))
    if owner.launch_response is not None:
      for name in ('propose', 'reset', 'confirm_output', 'freeze_release', 'freeze_braking'):
        calls.append(stack.enter_context(patch.object(owner.launch_response, name, wraps=getattr(owner.launch_response, name))))
    command, motion, _ = chain.consume(START + 110_000_000)
  assert motion.continuationActive, owner.continuation_reason
  assert (owner.pid.p, owner.pid.i, owner.pid.d, owner.pid.f) == before
  assert all(call.call_count == 0 for call in calls)
  for offset in (120, 130, 140):
    chain.consume(START + offset * 1_000_000)
  recovered = chain.produce(START + 150_000_000)
  assert recovered.valid and recovered.longitudinalPlan.solverStatus == 0
  with patch.object(owner.pid, 'update', wraps=owner.pid.update) as pid_update:
    _, motion, _ = chain.consume(START + 150_000_000)
  assert not motion.continuationActive and not motion.degraded
  assert pid_update.call_count == 1
