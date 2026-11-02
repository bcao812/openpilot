"""Current-frame lane correction, including its actual controlsd consumption."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from openpilot.sunnypilot.selfdrive.controls.lib.blendful import (
  BlendfulConsumer, CAR_STATE_MAX_AGE_NS, MODEL_MAX_AGE_NS,
)


START = 10_000_000_000
STEP = 10_000_000
CORRECTION = 0.001


class Subscriber(dict):
  def __init__(self):
    super().__init__()
    services = ('carState', 'modelV2', 'modelDataV2SP')
    self.logMonoTime = dict.fromkeys(services, START)
    self.seen = dict.fromkeys(services, True)
    self.valid = dict.fromkeys(services, True)
    self.alive = dict.fromkeys(services, True)
    self.freq_ok = dict.fromkeys(services, True)
    self['carState'] = SimpleNamespace(canValid=True, vEgo=10.0)


class ConsumerHarness:
  def __init__(self):
    self.now_ns = START
    self.consumer = BlendfulConsumer()
    self.sm = Subscriber()
    self.publish()

  def publish(self, *, capture_ns=None, model_ns=None, sp_ns=None, epoch=None, correction=CORRECTION):
    self.sm.logMonoTime['modelV2'] = self.now_ns if model_ns is None else model_ns
    self.sm.logMonoTime['modelDataV2SP'] = self.now_ns if sp_ns is None else sp_ns
    self.sm['modelV2'] = SimpleNamespace(timestampEof=self.now_ns if capture_ns is None else capture_ns,
                                       laneLineProbs=[0.0, 1.0, 1.0, 0.0])
    self.sm['modelDataV2SP'] = SimpleNamespace(
      blendfulControlEpoch=self.consumer.epoch if epoch is None else epoch, blendfulCurvature=correction,
      blendfulTrust=0.5, blendfulLeftProb=0.8, blendfulRightProb=0.8, blendfulLaneWidth=3.6, modelCurvature=0.002,
    )

  def update(self, *, now_ns=None, fresh_car=True, **gates):
    if now_ns is not None:
      self.now_ns = now_ns
    if fresh_car:
      self.sm.logMonoTime['carState'] = self.now_ns
    return self.consumer.update(self.sm, now_ns=self.now_ns, **({'enabled': True, 'lat_active': True} | gates))

  def frame(self, **fields):
    self.now_ns += STEP
    self.publish(**fields)
    return self.update()

  def activate(self, correction=CORRECTION):
    assert self.update() == 0.0
    assert self.consumer.epoch == self.now_ns
    assert self.frame(correction=correction) == correction


@pytest.mark.parametrize('correction', [0.0, CORRECTION, -CORRECTION])
def test_current_frame_is_identified_even_when_its_correction_is_zero(correction):
  harness = ConsumerHarness()
  harness.activate(correction)
  accepted = harness.consumer.model_mono_time
  assert accepted == harness.sm.logMonoTime['modelV2']
  for _ in range(4):
    assert harness.update(now_ns=harness.now_ns + STEP) == correction
    assert harness.consumer.model_mono_time == accepted


@pytest.mark.parametrize('gate', ['disabled', 'inactive', 'maneuver', 'low_speed', 'nan_speed', 'invalid_can', 'unhealthy_car'])
def test_qualification_loss_requires_a_new_current_frame_and_epoch(gate):
  harness = ConsumerHarness()
  harness.activate()
  old_epoch = harness.consumer.epoch
  options = {}
  if gate == 'disabled':
    options['enabled'] = False
  elif gate == 'inactive':
    options['lat_active'] = False
  elif gate == 'maneuver':
    options['model_path'] = False
  elif gate in ('low_speed', 'nan_speed'):
    harness.sm['carState'].vEgo = 4.999 if gate == 'low_speed' else float('nan')
  elif gate == 'invalid_can':
    harness.sm['carState'].canValid = False
  else:
    harness.sm.valid['carState'] = False
  assert harness.update(now_ns=harness.now_ns + STEP, **options) == 0.0
  assert harness.consumer.epoch == harness.consumer.model_mono_time == 0

  harness.sm['carState'].vEgo = 5.0
  harness.sm['carState'].canValid = True
  harness.sm.valid['carState'] = True
  # Exit and re-entry occur without a model publication in between.
  assert harness.update(now_ns=harness.now_ns + STEP) == 0.0
  assert harness.consumer.epoch > old_epoch
  assert harness.consumer.model_mono_time == 0
  assert harness.frame() == CORRECTION


def test_waiting_for_acknowledgement_does_not_keep_changing_the_epoch():
  harness = ConsumerHarness()
  assert harness.update() == 0.0
  epoch = harness.consumer.epoch
  for _ in range(3):
    assert harness.update(now_ns=harness.now_ns + STEP) == 0.0
    assert harness.consumer.epoch == epoch
  assert harness.frame(epoch=epoch - 1) == 0.0
  assert harness.frame(capture_ns=epoch - 1) == 0.0
  assert harness.consumer.epoch == epoch
  assert harness.frame() == CORRECTION


@pytest.mark.parametrize('service', ['modelV2', 'modelDataV2SP'])
@pytest.mark.parametrize('health', ['seen', 'valid', 'alive', 'freq_ok'])
def test_input_outage_resets_once_and_recovery_requires_the_new_ack(service, health):
  harness = ConsumerHarness()
  harness.activate()
  old_epoch = harness.consumer.epoch
  getattr(harness.sm, health)[service] = False
  assert harness.update(now_ns=harness.now_ns + STEP) == 0.0
  epoch = harness.consumer.epoch
  assert epoch > old_epoch
  for _ in range(3):
    assert harness.update(now_ns=harness.now_ns + STEP) == 0.0
    assert harness.consumer.epoch == epoch
  getattr(harness.sm, health)[service] = True
  assert harness.frame(epoch=old_epoch) == 0.0
  assert harness.consumer.epoch == epoch
  assert harness.frame() == CORRECTION


def test_original_capture_deadline_is_inclusive_and_cannot_be_renewed():
  harness = ConsumerHarness()
  harness.activate()
  capture = harness.sm['modelV2'].timestampEof
  epoch = harness.consumer.epoch
  assert harness.update(now_ns=capture + MODEL_MAX_AGE_NS) == CORRECTION
  assert harness.update(now_ns=capture + MODEL_MAX_AGE_NS + 1) == 0.0
  assert harness.consumer.model_mono_time == 0
  assert harness.consumer.epoch > epoch
  renewed_epoch = harness.consumer.epoch
  # Fresh Event times cannot make an old camera observation current.
  assert harness.frame(capture_ns=capture) == 0.0
  assert harness.consumer.epoch == renewed_epoch
  assert harness.frame() == CORRECTION


def test_an_unseen_but_expired_capture_is_rejected_despite_fresh_event_times():
  harness = ConsumerHarness()
  assert harness.update() == 0.0
  capture = harness.consumer.epoch + STEP
  harness.now_ns = capture + MODEL_MAX_AGE_NS + 1
  harness.publish(capture_ns=capture)
  assert harness.sm.logMonoTime['modelV2'] == harness.now_ns
  assert harness.update() == 0.0
  assert harness.consumer.model_mono_time == 0


def test_car_state_has_its_own_shorter_deadline():
  harness = ConsumerHarness()
  harness.activate()
  car_time = harness.sm.logMonoTime['carState']
  assert harness.update(now_ns=car_time + CAR_STATE_MAX_AGE_NS, fresh_car=False) == CORRECTION
  assert harness.update(now_ns=car_time + CAR_STATE_MAX_AGE_NS + 1, fresh_car=False) == 0.0
  assert harness.consumer.epoch == 0
  assert harness.update() == 0.0
  assert harness.frame() == CORRECTION


@pytest.mark.parametrize('first', ['model', 'diagnostics'])
def test_topics_must_pair_exactly_but_can_arrive_in_either_order(first):
  harness = ConsumerHarness()
  harness.activate()
  epoch = harness.consumer.epoch
  old_time = harness.sm.logMonoTime['modelV2']
  harness.now_ns += STEP
  if first == 'model':
    harness.publish(sp_ns=old_time)
  else:
    harness.sm.logMonoTime['modelDataV2SP'] = harness.now_ns
  pending_time = harness.now_ns
  assert harness.update() == 0.0
  assert harness.consumer.epoch == epoch
  if first == 'model':
    harness.sm.logMonoTime['modelDataV2SP'] = pending_time
  else:
    harness.sm.logMonoTime['modelV2'] = pending_time
    harness.sm['modelV2'].timestampEof = pending_time
  assert harness.update(now_ns=pending_time + STEP) == CORRECTION
  assert harness.consumer.epoch == epoch
  assert harness.consumer.model_mono_time == pending_time


@pytest.mark.parametrize('capture_delta', [0, -1])
def test_new_publications_cannot_reuse_or_reverse_the_capture(capture_delta):
  harness = ConsumerHarness()
  harness.activate()
  epoch = harness.consumer.epoch
  capture = harness.sm['modelV2'].timestampEof
  assert harness.frame(capture_ns=capture + capture_delta) == 0.0
  assert harness.update(now_ns=harness.now_ns + STEP) == 0.0
  assert harness.consumer.epoch == epoch
  assert harness.frame() == CORRECTION


def test_changed_capture_at_one_event_identity_cannot_supply_a_new_observation():
  harness = ConsumerHarness()
  harness.activate()
  capture = harness.sm['modelV2'].timestampEof
  harness.sm['modelV2'].timestampEof += 1
  assert harness.update(now_ns=harness.now_ns + STEP) == 0.0
  harness.sm['modelV2'].timestampEof = capture
  assert harness.update() == 0.0
  assert harness.frame() == CORRECTION


def test_a_new_controlsd_instance_cannot_consume_the_previous_instances_frame():
  harness = ConsumerHarness()
  harness.activate()
  old_epoch = harness.consumer.epoch
  harness.consumer = BlendfulConsumer()
  assert harness.update(now_ns=harness.now_ns + STEP) == 0.0
  assert harness.consumer.epoch > old_epoch
  assert harness.frame() == CORRECTION


@pytest.mark.parametrize('field,value', [
  ('blendfulCurvature', float('nan')), ('blendfulCurvature', float('inf')),
  ('blendfulTrust', 0.0), ('blendfulTrust', float('nan')), ('blendfulTrust', 1.1),
  ('blendfulLeftProb', -0.1), ('blendfulRightProb', float('nan')),
  ('blendfulLaneWidth', 0.0), ('blendfulLaneWidth', float('inf')), ('modelCurvature', float('nan')),
])
def test_nonfinite_or_unqualified_diagnostics_have_no_authority(field, value):
  harness = ConsumerHarness()
  harness.activate()
  setattr(harness.sm['modelDataV2SP'], field, value)
  assert harness.update() == 0.0
  assert harness.consumer.model_mono_time == 0
  assert harness.frame() == CORRECTION


@pytest.mark.parametrize('probability', [0.0, 0.299, float('nan'), 1.1])
def test_filtered_confidence_cannot_keep_a_currently_unqualified_side_active(probability):
  harness = ConsumerHarness()
  harness.activate()
  harness.sm['modelV2'].laneLineProbs[1] = probability
  assert harness.update() == 0.0
  harness.sm['modelDataV2SP'].blendfulLeftProb = 0.0
  assert harness.update() == CORRECTION  # Only the still-current right boundary contributes.
  harness.sm['modelDataV2SP'].blendfulRightProb = 0.0
  assert harness.update() == 0.0


def test_future_capture_does_not_poison_a_subsequent_current_observation():
  harness = ConsumerHarness()
  harness.activate()
  assert harness.frame(capture_ns=harness.now_ns + 10 * STEP) == 0.0
  assert harness.frame() == CORRECTION


@pytest.fixture
def controls_harness():
  from openpilot.cereal import custom
  from openpilot.common.params import Params
  from openpilot.common.prefix import OpenpilotPrefix
  from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import ControlsHarness, bounded_submaster, new_event

  with OpenpilotPrefix():
    harness = ControlsHarness()
    control = harness.control
    control.blendful = BlendfulConsumer()
    control.params = Params()
    control.params.put_bool('Blendful', True, block=True)
    control.params.put('ModelRunnerTypeCache', custom.ModelManagerSP.Runner.tinygrad, block=True)
    harness.events['modelDataV2SP'] = new_event('modelDataV2SP')
    harness.events['radarState'] = new_event('radarState')
    harness.events['lateralManeuverPlan'].valid = False
    control.sm = bounded_submaster(list(harness.events))
    # The transport fixture publishes at control rate; its average-frequency
    # source is bounded, while actual SubMaster validity and age still run.
    for service in ('carState', 'modelV2', 'modelDataV2SP'):
      control.sm.freq_tracker[service] = SimpleNamespace(record_recv_time=lambda _: None, valid=True)
    model = harness.events['modelV2'].modelV2
    model.laneLineProbs = [0.0, 1.0, 1.0, 0.0]
    model.action.desiredCurvature = 0.002
    model.action.desiredAcceleration = 0.8
    output = harness.events['modelDataV2SP'].modelDataV2SP
    output.blendfulCurvature = CORRECTION
    output.blendfulTrust = 0.5
    output.blendfulLeftProb = output.blendfulRightProb = 0.8
    output.blendfulLaneWidth = 3.6
    output.modelCurvature = 0.002
    yield harness


def control_frame(harness):
  harness.events['modelV2'].modelV2.timestampEof = harness.now_ns + STEP
  harness.events['modelDataV2SP'].modelDataV2SP.blendfulControlEpoch = harness.control.blendful.epoch
  return harness.step()[0]


def test_controlsd_adds_only_the_lateral_term_before_the_existing_clip(controls_harness):
  from openpilot.selfdrive.controls.controlsd import clip_curvature

  harness = controls_harness
  control_frame(harness)
  control = harness.control
  nominal = harness.events['modelV2'].modelV2.action.to_dict()
  previous = control.desired_curvature
  with patch('openpilot.selfdrive.controls.controlsd.clip_curvature', wraps=clip_curvature) as clip:
    command = control_frame(harness)
  correction = float(harness.events['modelDataV2SP'].modelDataV2SP.blendfulCurvature)
  requested = nominal['desiredCurvature'] + correction
  assert command.latActive
  assert clip.call_args.args[2] == requested
  expected, _ = clip_curvature(harness.car_state.vEgo, previous, requested, control.sm['vehicleParameters'].roll)
  assert command.actuators.curvature == pytest.approx(expected)
  assert control.sm['modelV2'].action.to_dict() == nominal
  receipt = control.state_control_ext(control.sm)
  assert receipt.blendfulControlEpoch == control.blendful.epoch
  assert receipt.blendfulModelMonoTime == control.sm.logMonoTime['modelV2']
  assert receipt.blendfulAppliedCurvature == correction


def test_maneuver_priority_withdraws_the_epoch_and_never_gets_lane_correction(controls_harness):
  from openpilot.selfdrive.controls.controlsd import clip_curvature

  harness = controls_harness
  control_frame(harness)
  control_frame(harness)
  assert harness.control.blendful.curvature != 0.0
  maneuver = harness.events['lateralManeuverPlan']
  maneuver.valid = True
  maneuver.lateralManeuverPlan.desiredCurvature = -0.002
  with patch('openpilot.selfdrive.controls.controlsd.clip_curvature', wraps=clip_curvature) as clip:
    control_frame(harness)
  assert clip.call_args.args[2] == maneuver.lateralManeuverPlan.desiredCurvature
  assert harness.control.blendful.epoch == 0
  receipt = harness.control.state_control_ext(harness.control.sm)
  assert receipt.blendfulModelMonoTime == 0
  assert receipt.blendfulAppliedCurvature == 0.0
  maneuver.valid = False
  control_frame(harness)
  assert harness.control.blendful.curvature == 0.0
  control_frame(harness)
  assert harness.control.blendful.curvature != 0.0


@pytest.mark.parametrize('gate', ['setting', 'runner', 'steer_fault'])
def test_real_control_loop_checks_withdrawal_each_cycle(controls_harness, gate):
  from openpilot.cereal import custom

  harness = controls_harness
  control_frame(harness)
  control_frame(harness)
  control = harness.control
  old_epoch = control.blendful.epoch
  assert control.blendful.curvature != 0.0
  if gate == 'setting':
    control.params.put_bool('Blendful', False, block=True)
  elif gate == 'runner':
    control.params.put('ModelRunnerTypeCache', custom.ModelManagerSP.Runner.stock, block=True)
  else:
    harness.car_state.steerFaultTemporary = True
  command = control_frame(harness)
  assert control.blendful.epoch == control.blendful.model_mono_time == 0
  assert control.blendful.curvature == 0.0
  if gate == 'steer_fault':
    assert not command.latActive
  control.params.put_bool('Blendful', True, block=True)
  control.params.put('ModelRunnerTypeCache', custom.ModelManagerSP.Runner.tinygrad, block=True)
  harness.car_state.steerFaultTemporary = False
  control_frame(harness)
  assert control.blendful.epoch > old_epoch
  assert control.blendful.curvature == 0.0
  control_frame(harness)
  assert control.blendful.curvature != 0.0
