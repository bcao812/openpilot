"""Consumer/alert policy for the separate stale-measurement continuation wire.

These tests reclassify a real, validated numerical offer to isolate the
receiver's status policy. Producer stale geometry is tested independently.
"""
from unittest.mock import patch

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.longitudinal_continuation import (
  STALE_MEASUREMENT_STATUS, ContinuationOffer, braking_only_continuation,
)
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import UnifiedLongitudinalPlanner
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import serialized
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_continuation_control_chain import START
from openpilot.selfdrive.selfdrived.tests.test_continuation_events import established_events, observe_at


def stale_status(chain, event):
  changed = event.as_builder()
  changed.longitudinalPlan.solverStatus = STALE_MEASUREMENT_STATUS
  chain.plan_event = serialized(changed)


def test_stale_status_cannot_execute_or_hide_an_accelerating_offer():
  daemon, chain = established_events()
  event = chain.produce(START + 100_000_000, 4)
  assert event.longitudinalPlan.continuation.valid
  assert not braking_only_continuation(ContinuationOffer.from_message(event.longitudinalPlan.continuation))
  stale_status(chain, event)
  observe_at(daemon, chain, START + 100_000_000, offset_ns=2_000_000)
  assert log.OnroadEvent.EventName.longitudinalControlUnavailable in daemon.events.names
  command, motion, can = chain.consume(START + 100_000_000)
  assert not motion.continuationActive and motion.degraded
  assert command.actuators.accel <= -0.5 and can['BRAKE_REQUEST'] == 1


def test_braking_only_stale_offer_is_diagnostic_until_its_fixed_deadline():
  # Cap the donor before integration and native-donor validation, rather than
  # editing any candidate trajectory or changing the measured initial state.
  with patch.object(UnifiedLongitudinalPlanner, 'independent_constraint',
                    return_value=(0.0, log.LongitudinalPlan.LongitudinalPlanSource.cruise, False, False)):
    daemon, chain = established_events()
    event = chain.produce(START + 100_000_000, 4)
  assert event.longitudinalPlan.continuation.valid
  assert braking_only_continuation(ContinuationOffer.from_message(event.longitudinalPlan.continuation))
  stale_status(chain, event)
  original = chain.plan_event.to_dict()
  deadline = chain.plan_event.longitudinalPlan.continuation.validUntilMonoTime
  observe_at(daemon, chain, START + 100_000_000, offset_ns=2_000_000)
  assert log.OnroadEvent.EventName.longitudinalControlUnavailable not in daemon.events.names
  for offset in range(100, 200, 10):
    sample = START + offset * 1_000_000
    command, motion, can = chain.consume(sample)
    assert motion.continuationActive and not motion.degraded
    assert command.actuators.accel <= 0.0 and can['ACCEL_COMMAND'] <= 0.0
    observe_at(daemon, chain, sample)
    assert log.OnroadEvent.EventName.longitudinalControlUnavailable not in daemon.events.names
    assert motion.continuationValidUntilMonoTime == deadline
  command, motion, can = chain.consume(START + 200_000_000)
  observe_at(daemon, chain, START + 200_000_000)
  assert not motion.continuationActive and motion.degraded
  assert command.actuators.accel <= -0.5 and can['BRAKE_REQUEST'] == 1
  assert log.OnroadEvent.EventName.longitudinalControlUnavailable in daemon.events.names
  assert chain.plan_event.to_dict() == original
