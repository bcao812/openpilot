"""Offline startup coverage for the radarState -> carControlSP -> Honda CAN path."""

import pytest

from openpilot.cereal import log
from openpilot.selfdrive.car.tests.test_actuation_observation import CardHarness
from openpilot.sunnypilot.selfdrive.controls.controlsd_ext import ControlsExt


TESTER_PRESENT = (0x18DAB0F1, bytes.fromhex("023e800000000000"), 1)


def startup_harness(*, lead_present, engaged, with_evidence):
  # The existing harness keeps real conversion/controller code but captures all
  # transport in memory. In particular, CI.init cannot disable a physical ECU.
  harness = CardHarness(enabled=False, native=True)
  harness.card.initialized_prev = False
  harness.command.enabled = engaged
  harness.command.latActive = engaged
  harness.command.longActive = engaged
  harness.command.actuators.accel = 0.5 if engaged else 0.0
  harness.command.actuators.torque = 0.2 if engaged else 0.0
  harness.command.actuators.longControlState = "pid" if engaged else "off"
  harness.command.cruiseControl.resume = engaged

  radar = log.RadarState.new_message()
  expected_evidence = {}
  for index, name in enumerate(("leadOne", "leadTwo"), 1):
    lead = getattr(radar, name)
    lead.present = lead_present
    if lead_present:
      lead.dRel = 16.0 + 8.0 * index
      lead.yRel = 0.25 * index
      lead.vRel = -0.5 * index
      lead.vLead = 10.0 - index
      lead.vLeadK = 9.5 - index
      lead.aLeadK = -0.25 * index
      lead.aLeadTau = 1.5
      lead.modelProb = 0.875
      lead.radar = True
      lead.radarTrackId = index
      lead.deprecated.aRel = -0.125 * index
      lead.deprecated.dPath = 0.25 * index
      lead.deprecated.vLat = -0.125 * index
      lead.deprecated.fcw = True

    if with_evidence:
      # Populate every evidence field, with distinct identities for both leads.
      # Qualified evidence must not enable an otherwise disengaged command.
      lead.evidence = {
        "version": 1,
        "source": "fused",
        "sessionId": f"startup-lead-{index}",
        "trackId": (1 << 53) + index,
        "generation": (1 << 54) + index,
        "positionMonoTime": 9_000_000_000 + index,
        "velocityMonoTime": 9_100_000_000 + index,
        "estimateMonoTime": 9_200_000_000 + index,
        "positionStd": 0.25 * index,
        "velocityStd": 0.125 * index,
        "positionValid": True,
        "velocityValid": True,
        "accelerationValid": True,
        "followQualified": True,
        "brakeQualified": True,
        "stopQualified": True,
        "departureQualified": True,
        "fcwQualified": True,
        "coasting": True,
        "pathState": "inPath",
        "pathOffset": 0.25 * index,
        "conflictStart": 0.5 * index,
        "conflictEnd": 2.0 + index,
        "lateralSpeed": -0.125 * index,
        "pathSource": f"startup-path-{index}",
        "visionIndex": index - 1,
        "reason": f"startup-evidence-{index}",
        "modelMonoTime": 9_300_000_000 + index,
        "cautionQualified": True,
        "physicalConflictValid": True,
        "physicalConflictStart": 0.375 * index,
        "physicalConflictEnd": 1.75 + index,
        "followGeometryMonoTime": 9_400_000_000 + index,
      }
    expected_evidence[name] = lead.evidence.to_dict()
    ControlsExt.get_lead_data(getattr(harness.events["carControlSP"].carControlSP, name), lead.as_reader())

  return harness, expected_evidence


@pytest.mark.parametrize("lead_present", [False, True], ids=["no-lead", "two-leads"])
@pytest.mark.parametrize("engaged", [False, True], ids=["startup-disengaged", "engaged"])
def test_evidence_survives_initial_apply_and_preserves_honda_keepalive_and_can(lead_present, engaged):
  baseline, baseline_evidence = startup_harness(lead_present=lead_present, engaged=engaged, with_evidence=False)
  populated, populated_evidence = startup_harness(lead_present=lead_present, engaged=engaged, with_evidence=True)
  assert baseline_evidence != populated_evidence

  for harness, expected_evidence in ((baseline, baseline_evidence), (populated, populated_evidence)):
    for frame in range(21):
      harness.step()
      assert harness.card.initialized_prev
      assert harness.card.CI.CC.frame == frame + 1
      applied, extension, _ = harness.requests[-1]
      assert applied["enabled"] == engaged
      assert applied["latActive"] == engaged
      assert applied["longActive"] == engaged
      for name in ("leadOne", "leadTwo"):
        assert extension[name]["status"] == lead_present
        assert extension[name]["evidence"] == expected_evidence[name]

    harness.card.CI.init.assert_called_once_with(harness.card.CP, harness.card.CP_SP, *harness.card.can_callbacks)
    harness.card.params.put_bool.assert_called_once_with("ControlsReady", True)
    assert len(harness.requests) == len(harness.sent_frames) == 21
    published = [event for service, event in harness.card.pm.history if service == "sendcan"]
    assert len(published) == 21
    assert all(event.valid for event in published)
    captured_can = [tuple((message.address, message.dat, message.src) for message in event.sendcan) for event in published]
    assert captured_can == harness.sent_frames
    assert [[message for message in frames if message[0] == TESTER_PRESENT[0]] for frames in captured_can] == [
      [TESTER_PRESENT] if frame in (0, 10, 20) else [] for frame in range(21)
    ]

  # Every CAN address, bus and payload stays identical when only evidence changes.
  assert baseline.sent_frames == populated.sent_frames
