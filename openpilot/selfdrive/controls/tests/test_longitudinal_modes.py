from itertools import product
from pathlib import Path
import re
from types import SimpleNamespace

import pytest

from opendbc.car.honda.values import CAR
from openpilot.cereal import log, messaging
from openpilot.common.params import Params, UnknownKeyName
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode, resolve_longitudinal_mode
from openpilot.selfdrive.controls.lib.motion_supervisor import DEGRADED_ACCEL, MOTION_INTENT_VERSION, MotionIntent, MotionPhase, StopReason
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params  # noqa: F401
from openpilot.selfdrive.controls.tests.test_motion_supervisor import MotionHarness
from openpilot.selfdrive.controls.tests.test_unified_controlsd import ControlsHarness, selfdrive_event_harness
from openpilot.selfdrive.controls.tests.test_unified_longcontrol import controller, normal_intent
from openpilot.selfdrive.controls.tests.test_unified_startup import StartupHarness


MODES = (LongitudinalMode.LEGACY, LongitudinalMode.UNIFIED)
MOTION_MODES = (LongitudinalMode.UNIFIED,)
WIRE_MODES = tuple(LongitudinalMode)


def mode_controls(mode):
  return ControlsHarness(platform=CAR.HONDA_ACCORD if mode == LongitudinalMode.UNIFIED else CAR.HONDA_CIVIC_BOSCH)


@pytest.mark.parametrize('key', ['LongitudinalMotionEnabled', 'LongitudinalUnifiedEnabled'])
def test_retired_migration_flags_are_not_registered(key):
  with pytest.raises(UnknownKeyName):
    Params().check_key(key)
  registry = Path(__file__).resolve().parents[3] / 'common' / 'params_keys.h'
  assert not re.findall(r'\{"' + key + r'", \{([^}]+)\}\}', registry.read_text())


@pytest.mark.parametrize('platform,openpilot_long,interceptor,expected', [
  (CAR.HONDA_ACCORD, True, False, LongitudinalMode.UNIFIED),
  (CAR.HONDA_CIVIC, True, False, LongitudinalMode.LEGACY),
  (CAR.HONDA_CIVIC_BOSCH, True, False, LongitudinalMode.LEGACY),
  (CAR.HONDA_CRV_HYBRID, True, False, LongitudinalMode.LEGACY),
  (CAR.HONDA_ACCORD, False, False, LongitudinalMode.LEGACY),
  (CAR.HONDA_ACCORD, True, True, LongitudinalMode.LEGACY),
])
def test_runtime_selection_uses_platform_and_longitudinal_authority(platform, openpilot_long, interceptor, expected):
  control = controller(platform=platform, openpilot_long=openpilot_long, interceptor=interceptor)
  assert resolve_longitudinal_mode(control.CP, control.CP_SP) == expected
  assert control.longitudinal_mode == expected
  assert control.unified == (expected == LongitudinalMode.UNIFIED)
  assert (control.motion_supervisor is not None) == control.unified


@pytest.mark.parametrize('creep_evidence', [True, 1, None, 'true'])
def test_retired_creep_payload_cannot_be_relabelled_as_unified(creep_evidence):
  message = SimpleNamespace(version=MOTION_INTENT_VERSION, mode=int(LongitudinalMode.UNIFIED), reasons=0,
                            movingLead=False, releaseAllowed=True, allowCreep=True, valid=True, ready=True,
                            creepEvidenceValid=creep_evidence)
  assert not MotionIntent.from_message(message).is_valid()


@pytest.mark.parametrize('mode', MODES)
def test_controls_acknowledges_the_selected_mode_without_a_protocol_fault(mode):
  harness = mode_controls(mode)
  command, motion = harness.step()
  assert command.longActive
  assert not motion.degraded
  assert harness.control.LoC.longitudinal_mode == mode
  assert motion.planningMode == int(mode)
  assert motion.version == (0 if mode == LongitudinalMode.LEGACY else MOTION_INTENT_VERSION)
  assert motion.enabled == (mode != LongitudinalMode.LEGACY)


@pytest.mark.parametrize('local_mode,remote_mode', [(local, remote) for local, remote in product(MODES, WIRE_MODES) if local != remote])
def test_cold_one_sided_stage_change_revokes_actuation_and_resume(local_mode, remote_mode):
  harness = mode_controls(local_mode)
  harness.plan.motionIntent.version = 0 if remote_mode == LongitudinalMode.LEGACY else MOTION_INTENT_VERSION
  harness.plan.motionIntent.mode = int(remote_mode)
  harness.plan.motionIntent.releaseAllowed = True
  harness.car_state.cruiseState.standstill = True
  command, motion = harness.step()
  assert command.enabled
  assert not command.longActive
  assert command.actuators.accel == 0.0
  assert command.actuators.longControlState == LongCtrlState.off
  assert command.cruiseControl.override
  assert not command.cruiseControl.resume
  assert motion.version == MOTION_INTENT_VERSION
  assert motion.planningMode == int(local_mode)
  assert motion.degraded
  assert harness.control.pm.messages['controlsState'].valid
  assert harness.control.LoC.longitudinal_mode == local_mode


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('version', [0, 1, 2, MOTION_INTENT_VERSION + 1, 255])
def test_v1_and_unknown_wire_versions_cannot_enable_unified(mode, version):
  harness = mode_controls(mode)
  harness.plan.motionIntent.version = version
  harness.plan.motionIntent.releaseAllowed = True
  command, motion = harness.step()
  assert not command.longActive
  assert command.actuators.accel == 0.0
  assert not command.cruiseControl.resume
  assert motion.degraded
  assert motion.planningMode == int(mode)


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('incoming_mode', [0, 1, 3, 255])
def test_unknown_wire_mode_does_not_use_readiness_as_a_bypass(mode, incoming_mode):
  harness = mode_controls(mode)
  harness.plan.motionIntent.mode = incoming_mode
  harness.plan.motionIntent.ready = False
  command, motion = harness.step()
  assert not command.longActive
  assert command.actuators.accel == 0.0
  assert not command.cruiseControl.resume
  assert motion.degraded
  daemon, car_state = selfdrive_event_harness(harness.control.pm.messages['controlsState'])
  daemon.update_events(car_state)
  assert log.OnroadEvent.EventName.commIssue in daemon.events.names


@pytest.mark.parametrize('mode', [0, 1, 3, 255, -1, True, False, 1.0, '1', None])
def test_intent_rejects_unknown_or_wrongly_typed_modes(mode):
  assert not MotionIntent(mode=mode, release_allowed=True).is_valid()


def test_missing_wire_mode_is_not_inferred_as_unified():
  message = SimpleNamespace(version=MOTION_INTENT_VERSION, reasons=0, movingLead=False, releaseAllowed=True,
                            allowCreep=True, valid=True, ready=True)
  intent = MotionIntent.from_message(message)
  assert intent.mode == int(LongitudinalMode.LEGACY)
  assert not intent.is_valid()


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('incoming_mode', WIRE_MODES + (3,))
def test_motion_owner_requires_a_matching_mode_before_releasing_hold(mode, incoming_mode):
  harness = MotionHarness()
  harness.step(normal_intent(mode, harness.now_ns + 10_000_000, 0.0, reasons=StopReason.MPC), v_ego=0.0, standstill=True)
  assert harness.supervisor.phase == MotionPhase.HOLD
  should_stop = harness.step(normal_intent(incoming_mode, harness.now_ns + 10_000_000, 0.0, release_allowed=True),
                            a_target=1.0, v_ego=0.0, standstill=True)
  if incoming_mode == mode:
    assert not should_stop
    assert not harness.supervisor.degraded
    assert harness.supervisor.phase == MotionPhase.RELEASE
  else:
    assert should_stop
    assert harness.supervisor.degraded
    assert harness.supervisor.phase == MotionPhase.HOLD
    assert harness.supervisor.reasons & StopReason.INVALID


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('standstill', [False, True])
@pytest.mark.parametrize('reason', [1 << 20, int(StopReason.HOLD), int(StopReason.INVALID)])
def test_invalid_evidence_cannot_grant_motion_in_unified(mode, standstill, reason):
  harness = mode_controls(mode)
  if standstill:
    harness.hold()
  else:
    harness.step()
  harness.plan.motionIntent.reasons = reason
  harness.plan.motionIntent.releaseAllowed = True
  harness.plan.aTarget = 1.0
  command, motion = harness.step()
  assert command.longActive
  assert command.actuators.accel <= DEGRADED_ACCEL + 1e-6
  assert not command.cruiseControl.resume
  assert motion.degraded
  assert motion.reasons & int(StopReason.INVALID)
  if standstill:
    assert motion.resolvedStop
    assert command.actuators.accel <= -2.0


@pytest.mark.parametrize('mode', MOTION_MODES)
def test_readiness_wait_is_cold_only_and_cannot_rearm_a_used_owner(mode):
  harness = mode_controls(mode)
  harness.plan.motionIntent.ready = False
  command, motion = harness.step()
  assert not command.longActive
  assert not motion.degraded
  assert motion.phase == 'off'
  assert harness.control.LoC.can_standby
  harness.plan.motionIntent.ready = True
  command, motion = harness.step()
  assert command.longActive
  assert not motion.degraded
  assert not harness.control.LoC.can_standby
  harness.plan.motionIntent.ready = False
  command, motion = harness.step()
  assert motion.degraded
  assert command.actuators.accel <= DEGRADED_ACCEL + 1e-6
  assert not command.cruiseControl.resume
  assert not harness.control.LoC.can_standby


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('condition', ['gas', 'brake', 'disable', 'override_event'])
def test_driver_authority_revokes_release_despite_invalid_evidence_in_unified(mode, condition):
  harness = mode_controls(mode)
  harness.hold()
  harness.plan.motionIntent.reasons = 0
  harness.plan.motionIntent.releaseAllowed = True
  harness.plan.aTarget = 0.5
  command, motion = harness.step()
  assert command.cruiseControl.resume
  if condition in ('gas', 'brake'):
    setattr(harness.car_state, condition + 'Pressed', True)
  elif condition == 'disable':
    harness.selfdrive_state.enabled = False
  else:
    event = messaging.new_message('onroadEvents', 1, valid=True)
    event.onroadEvents[0].overrideLongitudinal = True
    harness.events['onroadEvents'] = event
  harness.plan.motionIntent.valid = False
  command, motion = harness.step()
  assert not command.longActive
  assert command.actuators.accel == 0.0
  assert command.actuators.longControlState == LongCtrlState.off
  assert not command.cruiseControl.resume
  assert motion.phase == 'off'
  assert not motion.reasons
  assert not motion.degraded


@pytest.mark.parametrize('mode', MOTION_MODES)
@pytest.mark.parametrize('speed', [0.0, 5.0])
def test_unified_reaches_real_first_set_without_cold_start_motion_permission(mode, speed):
  harness = StartupHarness(speed=speed)
  harness.boot()
  plan = harness.latest['longitudinalPlan'].longitudinalPlan
  assert plan.motionIntent.version == MOTION_INTENT_VERSION
  assert plan.motionIntent.mode == int(mode)
  assert not plan.motionIntent.ready
  assert not plan.motionIntent.valid
  assert harness.controls.LoC.can_standby
  assert not harness.history[-1]['comm_issue']
  assert not harness.history[-1]['long_active']
  requested = harness.first_set()
  assert requested['enabled']
  assert not requested['no_entry']
  assert not requested['long_active']
  harness.step()
  assert harness.card.v_cruise_helper.v_cruise_initialized
