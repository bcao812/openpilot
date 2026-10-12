"""Retirement replaces the former stale-target latch and -7 continuation.

The production planner, native solver, wire publisher, Controls, LongControl,
CAN encoder and event observer all run. Prescribed vehicle measurements are
not a closed-loop vehicle or delivered-braking claim.
"""
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from openpilot.cereal import log
from openpilot.common.params import Params
from openpilot.selfdrive.controls.lib.lead_qualification import MAX_MEASUREMENT_AGE_NS
from openpilot.selfdrive.controls.tests.longitudinal_process_test_support import selfdrive_event_harness
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_continuation_control_chain import ContinuationChain, START
from openpilot.selfdrive.controls.tests.test_lead_qualification import formal_lead
from openpilot.selfdrive.controls.tests.test_radar_consumption import native_frame, radar_frame
from openpilot.selfdrive.selfdrived.state import StateMachine
from openpilot.selfdrive.selfdrived.tests.test_continuation_events import observe_at


def physical_lead(stamp, *, source='radar', track=399, generation=762, measurement=None):
  measurement = stamp if measurement is None else measurement
  return formal_lead(now_ns=stamp, source=source, dRel=100.0, vLead=20.0, vLeadK=20.0, vRel=0.0,
                     evidence={'trackId': track, 'generation': generation,
                               'positionMonoTime': measurement, 'velocityMonoTime': measurement})


@contextmanager
def current_chain():
  for name in ('AlphaLongitudinalEnabled', 'ExperimentalMode', 'DynamicExperimentalControl'):
    Params().put_bool(name, True, block=True)
  chain = ContinuationChain()
  chain.state['selfdriveState'].experimentalMode = True
  with patch.object(chain.planner.dec, 'enabled', return_value=True), patch.object(chain.planner.dec, 'mode', return_value='acc'):
    yield chain


def publish_and_execute(chain, stamp, *leads, raw_objects=None, daemon=None):
  frame = radar_frame(*(leads if raw_objects is None else raw_objects), publication=stamp, display=leads)
  chain.state['radarState'] = native_frame(frame)
  original = chain.state['radarState'].to_dict()
  calls = chain.planner.mpc.diagnostic_solve_calls
  event = chain.produce(stamp)
  assert chain.planner.mpc.diagnostic_solve_calls == calls + 1
  assert event.valid and event.longitudinalPlan.solverStatus == 0
  assert not event.longitudinalPlan.continuation.valid
  assert chain.state['radarState'].to_dict() == original
  for tick in range(5):
    sample = stamp + tick * 10_000_000
    command, motion, _ = chain.consume(sample)
    assert command.longActive and motion.phase == 'track'
    assert not motion.degraded and not motion.continuationActive
    if daemon is None:
      daemon, _ = selfdrive_event_harness(chain.controls.control.pm.messages['controlsState'])
      daemon.CP, daemon.CP_SP = chain.controls.control.CP, chain.controls.control.CP_SP
      daemon.big_model_ready_t = -10.0
      daemon.state_machine = StateMachine()
      daemon.state_machine.state = log.SelfdriveState.OpenpilotState.enabled
    observe_at(daemon, chain, sample)
    assert log.OnroadEvent.EventName.longitudinalControlUnavailable not in daemon.events.names
  return event, daemon


@pytest.mark.parametrize('source', ['radar', 'fused'])
@pytest.mark.parametrize('replacement', ['none', 'vision', 'new_radar', 'same_after_expiry'])
def test_old_measurement_does_not_block_current_native_plans_or_raise_planning_alert(source, replacement):
  with current_chain() as chain:
    daemon = None
    # Continuous prehistory establishes prior target and owner receipts.
    for frame in range(8):
      stamp = START + frame * 50_000_000
      _, daemon = publish_and_execute(chain, stamp, physical_lead(stamp, source=source), daemon=daemon)
    stamp = START + 400_000_000
    old = physical_lead(stamp, source=source, measurement=stamp - MAX_MEASUREMENT_AGE_NS - 1)
    _, daemon = publish_and_execute(chain, stamp, old, daemon=daemon)
    assert chain.planner.reference_planner.radar_consumption.expired_identities
    # Cross the old 3-second fault deadline with fresh messages; no OUT receipt
    # is needed, and the unrelated current source remains independently valid.
    for frame in range(9, 75):
      stamp = START + frame * 50_000_000
      leads = ()
      if replacement == 'vision':
        leads = (physical_lead(stamp, source='vision'),)
      elif replacement == 'new_radar':
        leads = (physical_lead(stamp, track=445, generation=1),)
      elif replacement == 'same_after_expiry':
        leads = (physical_lead(stamp, generation=773),)
      _, daemon = publish_and_execute(chain, stamp, *leads, daemon=daemon)
      assert not chain.planner.reference_planner.radar_consumption.degraded


@pytest.mark.parametrize('track,previous,current', [
  (261, 762, 773), (355, 983, 988), (355, 1042, 1049),
  (381, 1182, 1187), (448, 1522, 1527), (473, 1646, 1654),
])
def test_fresh_same_physical_track_generation_changes_do_not_inherit_an_old_veto(track, previous, current):
  with current_chain() as chain:
    daemon = None
    for frame in range(6):
      stamp = START + frame * 50_000_000
      _, daemon = publish_and_execute(chain, stamp, physical_lead(stamp, track=track, generation=previous), daemon=daemon)
    for frame in range(6, 14):
      stamp = START + frame * 50_000_000
      # Slot generation changes while the same physical ID has fresh samples.
      _, daemon = publish_and_execute(chain, stamp, physical_lead(stamp, track=track, generation=current), daemon=daemon)


def test_track399_can_leave_formal_slots_while_its_raw_measurement_is_still_fresh():
  with current_chain() as chain:
    daemon = None
    for frame in range(8):
      stamp = START + frame * 50_000_000
      _, daemon = publish_and_execute(chain, stamp, physical_lead(stamp), daemon=daemon)
    for frame in range(8, 20):
      stamp = START + frame * 50_000_000
      raw399 = physical_lead(stamp, measurement=stamp - 68_000_000)
      current_vision = physical_lead(stamp, source='vision')
      _, daemon = publish_and_execute(chain, stamp, current_vision, raw_objects=(raw399,), daemon=daemon)
      assert all(str(lead.evidence.source) == 'vision' for lead in chain.planner.mpc._source_leads)
      assert not chain.planner.mpc.source_object_identity
