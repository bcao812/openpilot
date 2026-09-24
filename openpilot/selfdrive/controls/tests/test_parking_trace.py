"""Parking evidence must retain its input identity and pre-arbitration facts."""
from dataclasses import replace
import json
from types import SimpleNamespace

from openpilot.selfdrive.controls.lib.longitudinal_trace import json_value, parking_detail
from openpilot.selfdrive.controls.lib.low_speed_stop_distance import LowSpeedStopDistanceEvidence
from openpilot.selfdrive.controls.lib.parking_approach import ParkingApproachDecision


def reference(source_ns):
  return SimpleNamespace(
    stop_distance_policy=SimpleNamespace(evidence=LowSpeedStopDistanceEvidence(
      source_ns=source_ns, radar_ns=source_ns - 10, lead_fractions=(1.0, 0.0), applied_fraction=.5)),
    parking_approach=SimpleNamespace(decision=ParkingApproachDecision(
      now_ns=source_ns + 20, source_ns=source_ns, radar_ns=source_ns - 10,
      uncertainty_model_ns=source_ns - 30, eligible=True, margin=-.2,
      require_stop=True, accel_cap=-1.5, cap_effective=True, accel_before=-.3, accel_after=-1.5)),
    parking_policy_distances=(3.5, (5.0, 6.0)),
    parking_mpc_distances=(4.0, (6.0, 6.0)),
  )


def test_parking_trace_keeps_qualification_request_and_effect_distinct():
  source_ns = (1 << 63) + 17
  planner = SimpleNamespace(reference_planner=reference(source_ns))
  snapshot = parking_detail(planner, source_ns)
  # A later update must not rewrite queued evidence.
  planner.reference_planner.parking_approach.decision = replace(
    planner.reference_planner.parking_approach.decision, margin=3.0, cap_effective=False)
  planner.reference_planner.parking_mpc_distances = (3.0, (4.0, 6.0))
  row = json.loads(json.dumps(json_value(snapshot), allow_nan=False))
  assert row['matches_car_state_source']
  assert row['approach']['source_ns'] == str(source_ns)
  assert row['approach']['uncertainty_model_ns'] == str(source_ns - 30)
  assert row['qualification']['applied_fraction'] == .5
  assert row['policy_returned_distances'] == [3.5, [5.0, 6.0]]
  assert row['mpc_reference_distances'] == [4.0, [6.0, 6.0]]
  assert row['approach']['margin'] == -.2
  assert row['approach']['cap_effective']


def test_parking_trace_marks_previous_source_without_relabeling_it():
  planner = SimpleNamespace(reference_planner=reference(100))
  row = json_value(parking_detail(planner, 200))
  assert not row['matches_car_state_source']
  assert row['approach']['source_ns'] == '100'


def test_parking_trace_other_planners_have_no_synthetic_evidence():
  assert parking_detail(SimpleNamespace(), 100) is None

