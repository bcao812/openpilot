"""Permanent EPS takeover ordering, including real controlsd consumption."""

import pytest

from openpilot.selfdrive.controls.lib.lateral_takeover import LateralTakeoverLatch, MADS_STATE_MAX_AGE_NS


START = 2_000_000_000
STEP = 10_000_000


class LateralHarness:
  def __init__(self):
    self.latch = LateralTakeoverLatch()
    self.now_ns = START

  def step(self, **changes):
    self.now_ns += STEP
    values = {"requested_active": True, "enabled": True, "mads_available": True, "mads_state": "enabled", "mads_active": True,
              "permanent_fault": False, "temporary_fault": False, "mads_mono_ns": self.now_ns,
              "car_state_mono_ns": self.now_ns, "now_ns": self.now_ns, "inputs_valid": True}
    values.update(changes)
    return self.latch.update(**values)

  def fault(self):
    assert self.step()
    assert not self.step(requested_active=False, permanent_fault=True)
    assert self.latch.latched

  def disabled(self, **changes):
    return self.step(requested_active=False, mads_state="disabled", mads_active=False, **changes)


def test_bit_clear_cannot_reuse_the_pre_fault_mads_authority():
  harness = LateralHarness()
  assert harness.step()
  active_time = harness.now_ns
  assert not harness.step(requested_active=False, permanent_fault=True, mads_mono_ns=active_time)
  assert not harness.step(mads_mono_ns=active_time)
  assert not harness.step()  # A new publication of active without disabled is insufficient too.
  assert harness.latch.latched


def test_disabled_during_a_sustained_fault_is_kept_until_a_healthy_rearm():
  harness = LateralHarness()
  harness.fault()
  assert not harness.disabled(permanent_fault=True)
  for _ in range(20):
    assert not harness.disabled(permanent_fault=True)
  assert not harness.step(temporary_fault=True)
  assert harness.latch.latched
  assert harness.step()
  assert not harness.latch.latched


def test_disabled_published_after_the_fault_sample_can_arrive_in_the_same_consumer_tick():
  harness = LateralHarness()
  assert harness.step()
  fault_time = harness.now_ns + STEP // 2
  disabled_time = harness.now_ns + 3 * STEP // 4
  assert not harness.disabled(permanent_fault=True, car_state_mono_ns=fault_time, mads_mono_ns=disabled_time)
  assert harness.latch.latched
  assert harness.step()


@pytest.mark.parametrize("intervening", [{"requested_active": False}, {"requested_active": False, "temporary_fault": True}])
def test_an_existing_active_session_stays_armed_across_other_lateral_gates(intervening):
  harness = LateralHarness()
  assert harness.step()
  assert not harness.step(**intervening)
  assert not harness.step(requested_active=False, permanent_fault=True)
  assert not harness.step()
  assert harness.latch.latched


def test_no_previous_actual_passage_does_not_create_a_takeover():
  harness = LateralHarness()
  assert not harness.step(requested_active=False)
  assert not harness.step(requested_active=False, permanent_fault=True)
  assert not harness.latch.latched
  assert harness.step()


def test_a_prior_disabled_message_retires_the_active_session():
  harness = LateralHarness()
  assert harness.step()
  assert not harness.disabled()
  assert not harness.step(requested_active=False, permanent_fault=True)
  assert not harness.latch.latched


def test_temporary_fault_recovers_without_a_disabled_handshake():
  harness = LateralHarness()
  assert harness.step()
  assert not harness.step(requested_active=False, temporary_fault=True)
  assert not harness.latch.latched
  assert harness.step()


def test_rearm_never_overrides_a_remaining_lateral_gate():
  harness = LateralHarness()
  harness.fault()
  assert not harness.disabled()
  assert not harness.step(requested_active=False)
  assert not harness.latch.latched
  assert harness.step()


def test_a_second_fault_cannot_reuse_the_first_disabled_acknowledgement():
  harness = LateralHarness()
  harness.fault()
  assert not harness.disabled()
  assert harness.step()
  assert not harness.step(requested_active=False, permanent_fault=True)
  assert not harness.step()
  assert harness.latch.latched
  assert not harness.disabled()
  assert harness.step()


@pytest.mark.parametrize("state,active", [("paused", False), ("overriding", True), ("softDisabling", True)])
def test_only_disabled_then_enabled_can_acknowledge_and_rearm(state, active):
  harness = LateralHarness()
  harness.fault()
  assert not harness.step(mads_state=state, mads_active=active)
  assert not harness.step()
  assert not harness.disabled()
  assert not harness.step(mads_state=state, mads_active=active)
  assert harness.step()


def test_unavailable_mads_cannot_clear_a_latch():
  harness = LateralHarness()
  harness.fault()
  assert not harness.disabled(mads_available=False)
  assert not harness.step()
  assert harness.latch.latched
  assert not harness.disabled()
  assert harness.step()


@pytest.mark.parametrize("bad_input", ["stale_mads", "future_mads", "replayed_mads", "stale_car", "future_car", "invalid"])
def test_bad_messages_cannot_supply_the_disabled_acknowledgement(bad_input):
  harness = LateralHarness()
  harness.fault()
  changes = {
    "stale_mads": {"mads_mono_ns": harness.now_ns - MADS_STATE_MAX_AGE_NS},
    "future_mads": {"mads_mono_ns": harness.now_ns + 2 * STEP},
    "replayed_mads": {"mads_mono_ns": START + STEP},
    "stale_car": {"car_state_mono_ns": harness.now_ns - MADS_STATE_MAX_AGE_NS},
    "future_car": {"car_state_mono_ns": harness.now_ns + 2 * STEP},
    "invalid": {"inputs_valid": False},
  }[bad_input]
  assert not harness.disabled(**changes)
  assert not harness.step()
  assert harness.latch.latched
  assert not harness.disabled()
  assert harness.step()


def test_changed_payload_at_the_same_time_is_not_a_new_message():
  harness = LateralHarness()
  harness.fault()
  same_time = harness.now_ns
  assert not harness.disabled(mads_mono_ns=same_time)
  assert not harness.step()
  assert not harness.disabled()
  disabled_time = harness.now_ns
  assert not harness.step(mads_mono_ns=disabled_time)
  assert harness.latch.latched
  assert harness.step()


def test_a_future_attempt_does_not_poison_the_next_valid_rearm():
  harness = LateralHarness()
  harness.fault()
  assert not harness.disabled()
  assert not harness.step(mads_mono_ns=harness.now_ns + 20 * STEP)
  assert harness.step()


def test_clock_reversal_still_latches_a_fault_after_an_actual_passage():
  harness = LateralHarness()
  assert harness.step()
  assert not harness.step(requested_active=False, permanent_fault=True, now_ns=START)
  assert harness.latch.latched
  assert not harness.step()
  assert not harness.disabled()
  assert harness.step()


def test_without_mads_configuration_the_original_request_is_unchanged():
  latch = LateralTakeoverLatch()
  for requested_active in (False, True):
    assert latch.update(requested_active, enabled=False, mads_available=False, mads_state="disabled", mads_active=False,
                        permanent_fault=True, temporary_fault=True, mads_mono_ns=0, car_state_mono_ns=0,
                        now_ns=0, inputs_valid=False) is requested_active
  assert not latch.latched


@pytest.fixture
def controls_harness():
  # Keep pure ordering tests independent of native imports; only the actual
  # consumer cases below need the existing isolated host dependencies.
  from opendbc.safety import ALTERNATIVE_EXPERIENCE
  from openpilot.common.prefix import OpenpilotPrefix
  from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import ControlsHarness

  with OpenpilotPrefix():
    harness = ControlsHarness()
    assert harness.control.LoC.unified
    harness.control.CP.alternativeExperience = ALTERNATIVE_EXPERIENCE.ENABLE_MADS
    mads = harness.events['selfdriveStateSP'].selfdriveStateSP.mads
    mads.available = True
    mads.enabled = True
    mads.active = True
    mads.state = "enabled"
    yield harness


def test_controlsd_keeps_longitudinal_but_waits_for_the_mads_rearm(controls_harness):
  harness = controls_harness
  command, _ = harness.step()
  assert command.latActive and command.longActive
  # Hold the actual SubMaster MADS publication while carState advances.
  mads_event = harness.events.pop('selfdriveStateSP')
  harness.car_state.steerFaultPermanent = True
  command, _ = harness.step()
  assert not command.latActive and command.longActive
  harness.car_state.steerFaultPermanent = False
  command, _ = harness.step()
  assert not command.latActive and command.longActive

  harness.events['selfdriveStateSP'] = mads_event
  command, _ = harness.step()
  assert not command.latActive and command.longActive
  mads = mads_event.selfdriveStateSP.mads
  mads.state = "disabled"
  mads.enabled = mads.active = False
  command, _ = harness.step()
  assert not command.latActive and command.longActive
  mads.state = "enabled"
  mads.enabled = mads.active = True
  command, _ = harness.step()
  assert command.latActive and command.longActive


def test_controlsd_rejects_an_expired_mads_authority_without_disabling_longitudinal(controls_harness):
  harness = controls_harness
  assert harness.step()[0].latActive
  harness.events.pop('selfdriveStateSP')
  for _ in range(MADS_STATE_MAX_AGE_NS // STEP + 1):
    command, _ = harness.step()
  assert not command.latActive and command.longActive
  assert not harness.control.lateral_takeover.latched


def test_controlsd_does_not_latch_a_temporary_eps_fault(controls_harness):
  harness = controls_harness
  assert harness.step()[0].latActive
  harness.car_state.steerFaultTemporary = True
  assert not harness.step()[0].latActive
  harness.car_state.steerFaultTemporary = False
  assert harness.step()[0].latActive
  assert not harness.control.lateral_takeover.latched


def test_controlsd_keeps_an_existing_stop_while_the_eps_takeover_is_latched(controls_harness):
  from openpilot.selfdrive.controls.lib.motion_supervisor import StopReason

  harness = controls_harness
  assert harness.step()[0].latActive
  # This owner is already tracking. Establish HOLD through the normal stopping
  # ramp instead of the helper's cold-OFF immediate-hold assumption.
  harness.car_state.vEgo = 0.0
  harness.car_state.standstill = True
  harness.car_state.cruiseState.standstill = True
  harness.plan.aTarget = -0.1
  harness.plan.motionIntent.reasons = int(StopReason.MPC)
  for _ in range(150):
    command, motion = harness.step()
  assert command.actuators.accel == -2.0 and motion.phase == 'hold'
  harness.car_state.steerFaultPermanent = True
  command, motion = harness.step()
  assert harness.control.lateral_takeover.latched
  assert not command.latActive and command.longActive
  assert command.actuators.accel == -2.0 and motion.phase == 'hold'
  harness.car_state.steerFaultPermanent = False
  command, motion = harness.step()
  assert not command.latActive and command.longActive
  assert command.actuators.accel == -2.0 and motion.phase == 'hold'
  assert not command.cruiseControl.resume


@pytest.mark.parametrize("gate", ["standstill", "blinker"])
def test_controlsd_keeps_its_existing_lateral_gates_after_rearm(controls_harness, gate):
  harness = controls_harness
  assert harness.step()[0].latActive
  harness.car_state.steerFaultPermanent = True
  assert not harness.step()[0].latActive
  mads = harness.events['selfdriveStateSP'].selfdriveStateSP.mads
  mads.state = "disabled"
  mads.enabled = mads.active = False
  assert not harness.step()[0].latActive
  harness.car_state.steerFaultPermanent = False
  mads.state = "enabled"
  mads.enabled = mads.active = True
  if gate == "standstill":
    harness.car_state.standstill = True
  else:
    harness.control.blinker_pause_lateral.update.return_value = True
  assert not harness.step()[0].latActive
  assert not harness.control.lateral_takeover.latched
  harness.car_state.standstill = False
  harness.control.blinker_pause_lateral.update.return_value = False
  assert harness.step()[0].latActive


def test_controlsd_without_mads_keeps_its_original_eps_recovery(controls_harness):
  harness = controls_harness
  harness.control.CP.alternativeExperience = 0
  harness.events['selfdriveStateSP'].selfdriveStateSP.mads.available = False
  assert harness.step()[0].latActive
  harness.car_state.steerFaultPermanent = True
  assert not harness.step()[0].latActive
  harness.car_state.steerFaultPermanent = False
  assert harness.step()[0].latActive
