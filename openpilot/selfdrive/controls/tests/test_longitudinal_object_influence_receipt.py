import pytest

from openpilot.cereal import custom, log
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.motion_supervisor import StopReason
from openpilot.selfdrive.controls.lib.object_influence import plan_object_influences
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import ControlsHarness


def object_record(kind='braking', strength=0.4):
  return {'version': 1, 'sessionId': 'radar-session', 'trackId': 17, 'modelMonoTime': 0, 'visionIndex': -1, 'kind': kind, 'strength': strength}


def test_controls_acknowledges_actual_plan_without_changing_command():
  observed, baseline = ControlsHarness(), ControlsHarness()
  for consumer in (observed, baseline):
    consumer.plan.aTarget = -1.0
  observed.plan.objectInfluences = [object_record()]
  command, motion = observed.step()
  baseline_command, baseline_motion = baseline.step()
  assert command.to_dict() == baseline_command.to_dict()
  assert motion.phase == baseline_motion.phase == 'track'
  assert len(motion.objectInfluences) == 1
  assert motion.objectInfluences[0].trackId == 17
  assert motion.planMonoTime == observed.control.sm.logMonoTime['longitudinalPlan']


def test_radar_loss_claim_does_not_change_owner_or_receipts_before_existing_force_decel_rules():
  observed, baseline = ControlsHarness(), ControlsHarness()
  for consumer in (observed, baseline):
    consumer.plan.aTarget = -1.0
    consumer.plan.objectInfluences = [object_record()]
  observed.plan.radarRiskUnresolved = True
  command, motion = observed.step()
  baseline_command, baseline_motion = baseline.step()
  assert command.to_dict() == baseline_command.to_dict()
  assert motion.to_dict() == baseline_motion.to_dict()
  assert not motion.degraded and len(motion.objectInfluences) == 1
  # The next force-deceleration plan still uses the existing independent
  # stop reason; the informational radar claim never replaces this contract.
  observed.plan.motionIntent.reasons = int(StopReason.FORCE_DECEL)
  _, motion = observed.step()
  assert not motion.degraded and not motion.objectInfluences
  observed.selfdrive_state.enabled = False
  command, motion = observed.step()
  assert not command.longActive and not motion.objectInfluences


@pytest.mark.parametrize('pedal', ['gasPressed', 'brakePressed'])
def test_manual_override_clears_an_earlier_object_receipt_on_the_same_tick(pedal):
  consumer = ControlsHarness()
  consumer.plan.aTarget = -1.0
  consumer.plan.objectInfluences = [object_record()]
  _, motion = consumer.step()
  assert motion.objectInfluences
  setattr(consumer.car_state, pedal, True)
  command, motion = consumer.step()
  assert not command.longActive
  assert not motion.objectInfluences


def test_rejected_or_expired_plan_does_not_republish_prior_red_object():
  consumer = ControlsHarness()
  consumer.plan.aTarget = -1.0
  consumer.plan.objectInfluences = [object_record()]
  _, motion = consumer.step()
  assert motion.objectInfluences
  _, motion = consumer.step(new_plan=False, now_ns=consumer.now_ns + 260_000_000)
  assert motion.degraded or motion.waitingForFreshInput
  assert not motion.objectInfluences


def test_new_plan_without_object_evidence_clears_the_receipt():
  consumer = ControlsHarness()
  consumer.plan.aTarget = -1.0
  consumer.plan.objectInfluences = [object_record()]
  assert consumer.step()[1].objectInfluences
  consumer.plan.objectInfluences = []
  assert not consumer.step()[1].objectInfluences


def test_real_owner_hold_has_maximum_strength_and_rejected_release_is_not_red():
  consumer = ControlsHarness()
  consumer.plan.objectInfluences = [object_record('hold', 1.0)]
  _, motion = consumer.hold()
  assert len(motion.objectInfluences) == 1
  assert motion.objectInfluences[0].kind == 'hold'
  assert motion.objectInfluences[0].strength == 1.0
  consumer.plan.objectInfluences = [object_record('release')]
  _, motion = consumer.step()
  assert motion.resolvedStop
  assert not motion.objectInfluences


def test_reading_old_capnp_inputs_does_not_allocate_evidence_or_influence_fields():
  state = log.RadarState.new_message(mdMonoTime=2_000_000_000)
  state.leadTwo.present = True
  before = state.to_dict()
  records = plan_object_influences(
    state, active_identities=(('vision', 2_000_000_000, 1),), now_ns=2_000_000_000,
    permitted=True, fusion_admitted=False, reference_accepted=True, independent_stop=False, object_stop=False,
    held_identity=None, release_identity=None, standstill=False, target_accel=-1.0, accel_min=-3.5, accel_max=2.0,
  )
  assert len(records) == 1
  assert records[0]['visionIndex'] == 1
  assert state.to_dict() == before


def test_retired_edge_caution_wire_stays_readable_without_an_owner_receipt():
  evidence = custom.RadarLeadEvidence.new_message()
  assert not evidence.cautionQualified
  evidence.cautionQualified = True
  evidence.pathState = 'edge'
  with custom.RadarLeadEvidence.from_bytes(evidence.to_bytes()) as decoded:
    assert decoded.cautionQualified and decoded.pathState == 'edge'
    assert not any((decoded.brakeQualified, decoded.stopQualified, decoded.departureQualified, decoded.fcwQualified))
  consumer = ControlsHarness()
  consumer.plan.aTarget = 0.0
  consumer.plan.objectInfluences = [object_record('caution', 0.15)]
  command, motion = consumer.step()
  assert motion.phase == 'track' and not motion.resolvedStop
  assert command.actuators.accel == 0.0
  assert consumer.control.LoC.long_control_state == LongCtrlState.pid
  assert not motion.objectInfluences


def test_real_numeric_pid_owner_acknowledges_following_without_a_braking_request():
  consumer = ControlsHarness()
  consumer.plan.aTarget = 0.2
  consumer.plan.objectInfluences = [object_record('follow', 0.15)]
  command, motion = consumer.step()
  assert motion.phase == 'track' and not motion.resolvedStop
  assert consumer.control.LoC.long_control_state == LongCtrlState.pid
  assert command.actuators.accel > 0.0
  assert len(motion.objectInfluences) == 1
  assert motion.objectInfluences[0].kind == 'follow'


def test_real_numeric_pid_owner_acknowledges_adopted_departure_during_brake_release():
  consumer = ControlsHarness()
  consumer.hold()
  consumer.car_state.cruiseState.standstill = False
  consumer.plan.motionIntent.reasons = 0
  consumer.plan.motionIntent.releaseAllowed = True
  consumer.plan.aTarget = 0.5
  consumer.plan.objectInfluences = [object_record('release')]
  command, motion = consumer.step()
  assert motion.phase == 'release' and not motion.resolvedStop
  assert consumer.control.LoC.long_control_state == LongCtrlState.pid
  assert command.actuators.accel > -2.0
  assert len(motion.objectInfluences) == 1
  assert motion.objectInfluences[0].kind == 'release'


def braking_only_consumer(*, include_influence=True, warm_transport=True):
  consumer = ControlsHarness()
  if warm_transport:
    # The independent braking envelope requires the real per-service
    # frequency receipt, even though this general harness ignores frequency
    # in all_checks(). Establish 100 Hz control / 20 Hz plan transport first.
    for frame in range(10):
      consumer.step(new_plan=frame % 5 == 0)
    assert consumer.control.sm.freq_ok['carState']
    assert consumer.control.sm.freq_ok['longitudinalPlan']
  consumer.events['longitudinalPlan'].valid = False
  consumer.plan.solverStatus = -6
  consumer.plan.aTarget = -3.5
  consumer.plan.motionIntent.valid = False
  consumer.plan.motionIntent.brakingValid = True
  consumer.plan.motionIntent.brakingAccel = -3.5
  if include_influence:
    consumer.plan.objectInfluences = [dict(object_record('braking', 1.0), version=2)]
  return consumer


def test_cold_transport_keeps_owner_braking_qualification_and_rejects_retired_extra_receipt():
  consumer = braking_only_consumer(warm_transport=False)
  command, motion = consumer.step()
  assert not consumer.control.sm.freq_ok['longitudinalPlan']
  assert consumer.control.LoC.motion_supervisor.degraded_braking_accel is None
  assert command.actuators.accel == pytest.approx(-0.2)
  assert not motion.objectInfluences


def test_retired_extra_receipt_is_rejected_without_changing_braking_only_actuation():
  observed, baseline = braking_only_consumer(), braking_only_consumer(include_influence=False)
  command, motion = observed.step()
  baseline_command, baseline_motion = baseline.step()
  assert command.to_dict() == baseline_command.to_dict()
  assert command.longActive and command.actuators.accel == -3.5
  assert motion.degraded and baseline_motion.degraded
  assert observed.control.LoC.motion_supervisor.degraded_braking_accel == -3.5
  assert not motion.objectInfluences
  assert not baseline_motion.objectInfluences


def test_generic_fault_with_old_normal_object_evidence_has_no_object_receipt():
  consumer = braking_only_consumer(include_influence=False)
  consumer.plan.objectInfluences = [object_record('braking', 1.0)]
  command, motion = consumer.step()
  assert motion.degraded and command.actuators.accel == -3.5
  assert not motion.objectInfluences


@pytest.mark.parametrize('case', ['manual', 'bad_braking_contract', 'bad_key_message', 'expired_plan'])
def test_retired_extra_receipt_stays_cleared_across_owner_and_transport_changes(case):
  consumer = braking_only_consumer()
  _, motion = consumer.step()
  assert not motion.objectInfluences
  # Keep transport at 20 Hz while checking that no later owner or transport
  # transition can revive the retired extra-object display record.
  for _ in range(4):
    assert not consumer.step(new_plan=False)[1].objectInfluences
  if case == 'manual':
    consumer.car_state.gasPressed = True
  elif case == 'bad_braking_contract':
    consumer.plan.motionIntent.brakingValid = False
  elif case == 'bad_key_message':
    consumer.events['modelV2'].valid = False
  kwargs = {'new_plan': False, 'now_ns': consumer.now_ns + 260_000_000} if case == 'expired_plan' else {}
  _, motion = consumer.step(**kwargs)
  assert not motion.objectInfluences
