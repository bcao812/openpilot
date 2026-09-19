import unittest

from opendbc.can import CANPacker
from opendbc.car import Bus, CanData, DT_CTRL, gen_empty_fingerprint
from opendbc.car.car_helpers import interfaces
from opendbc.car.honda.hondacan import CanBus
from opendbc.car.honda.values import CAR, DBC, HondaFlags

# the mask window as carstate.py sizes it
ACC_FAULT_MASK_FRAMES = int(10 / DT_CTRL)


class TestHondaFingerprint(unittest.TestCase):
  def test_tja_bosch_only(self):
    for car_model in CAR:
      if car_model.config.flags & HondaFlags.BOSCH_TJA_CONTROL:
        assert car_model.config.flags & HondaFlags.BOSCH, "Nidec car found with TJA control"


class AccFaultHarness:
  """Drives a real CarState with synthetic CAN so the startup fault mask is observable end to end."""

  def __init__(self, platform, alpha_long=False):
    CarInterface = interfaces[platform]
    fingerprint = gen_empty_fingerprint()
    self.CP = CarInterface.get_params(platform, fingerprint, [], alpha_long, False, False)
    self.CP_SP = CarInterface.get_params_sp(self.CP, platform, fingerprint, [], alpha_long, False, False)
    self.car_interface = CarInterface(self.CP, self.CP_SP)

    self.bus = CanBus(self.CP).pt
    self.packer = CANPacker(DBC[self.CP.carFingerprint][Bus.pt])
    # SCM_FEEDBACK or SCM_BUTTONS depending on the platform; carry MAIN_ON
    self.scm_msg = self.car_interface.CS.car_state_scm_msg
    self.frame = 0

  def _fault_msgs(self, faulted: bool):
    if self.CP.flags & HondaFlags.BOSCH_RADARLESS:
      return [("CRUISE_FAULT_STATUS", {"CRUISE_FAULT": int(faulted)})]
    return [("STANDSTILL", {"BRAKE_ERROR_1": int(faulted), "BRAKE_ERROR_2": 0})]

  def step(self, faulted: bool = False):
    # MAIN_ON is held high so cruiseState.available is True unless the mask pulls it down
    msgs = self._fault_msgs(faulted) + [(self.scm_msg, {"MAIN_ON": 1})]
    # every message used here carries a COUNTER; the parser drops frames that do not advance it
    packed = [self.packer.make_can_msg(name, self.bus, dict(values, COUNTER=self.frame % 4))
              for name, values in msgs]
    self.frame += 1
    CS, _ = self.car_interface.update([(0, [CanData(m[0], m[1], self.bus) for m in packed])])
    return CS

  def settle(self, faulted: bool = False, frames: int = 3):
    """The parser rejects the first frame of each message, so step a few times before asserting."""
    for _ in range(frames):
      CS = self.step(faulted)
    return CS

  @property
  def latched(self) -> bool:
    return self.car_interface.CS.initial_accFault_cleared


class TestHondaStartupAccFaultMask(unittest.TestCase):
  """
  The Bosch startup grace period blocks engagement through cruiseState.available while a
  raw brake/cruise fault is present. After the timer and a clean frame, faults are reported
  through accFaulted normally. This host-state policy does not clear vehicle ECU DTCs.
  """

  # radarless reads CRUISE_FAULT_STATUS with stock ACC; radar Bosch needs alpha long to derive a fault
  BOSCH_CASES = ((CAR.HONDA_CIVIC_2022, False), (CAR.HONDA_ACCORD, True))

  def test_fault_masked_at_startup(self):
    for platform, alpha_long in self.BOSCH_CASES:
      with self.subTest(platform=platform):
        harness = AccFaultHarness(platform, alpha_long)
        CS = harness.settle(faulted=True)
        assert not CS.accFaulted, "startup fault should be suppressed on Bosch"
        assert not CS.cruiseState.available, "engagement should be blocked via cruiseState.available"

  def test_available_not_forced_without_a_fault(self):
    for platform, alpha_long in self.BOSCH_CASES:
      with self.subTest(platform=platform):
        harness = AccFaultHarness(platform, alpha_long)
        CS = harness.settle(faulted=False)
        assert not CS.accFaulted
        assert CS.cruiseState.available, "MAIN_ON must still come through when nothing is faulted"

  def test_latch_needs_one_clean_frame_past_the_window(self):
    # the timer is decremented after the check, so the latch sets on the frame after it reaches 0
    for platform, alpha_long in self.BOSCH_CASES:
      with self.subTest(platform=platform):
        harness = AccFaultHarness(platform, alpha_long)
        for _ in range(ACC_FAULT_MASK_FRAMES):
          harness.step(faulted=False)
        assert not harness.latched, "latched one frame early"
        harness.step(faulted=False)
        assert harness.latched, "should latch on the first clean frame after the window"

  def test_fault_reported_normally_after_the_window(self):
    for platform, alpha_long in self.BOSCH_CASES:
      with self.subTest(platform=platform):
        harness = AccFaultHarness(platform, alpha_long)
        for _ in range(ACC_FAULT_MASK_FRAMES + 1):
          harness.step(faulted=False)
        CS = harness.step(faulted=True)
        assert CS.accFaulted, "fault after the window must be reported"
        assert CS.cruiseState.available, "post-window faults must not pull cruiseState.available down"

  def test_mask_cannot_rearm_mid_drive(self):
    for platform, alpha_long in self.BOSCH_CASES:
      with self.subTest(platform=platform):
        harness = AccFaultHarness(platform, alpha_long)
        for _ in range(ACC_FAULT_MASK_FRAMES + 1):
          harness.step(faulted=False)
        assert harness.latched
        # a long fault-free stretch, then a fault: still reported, latch never clears
        for _ in range(100):
          harness.step(faulted=False)
        assert harness.latched
        assert harness.step(faulted=True).accFaulted

  def test_nidec_unaffected(self):
    # the mask is gated on HondaFlags.BOSCH, so Nidec reports a startup fault immediately
    harness = AccFaultHarness(CAR.HONDA_CIVIC)
    assert not harness.CP.flags & HondaFlags.BOSCH
    CS = harness.settle(faulted=True)
    assert CS.accFaulted, "Nidec behaviour should be unchanged by the Bosch startup mask"
    assert CS.cruiseState.available, "Nidec must not have cruiseState.available forced down"
