"""Current-track launch continuity through the real planner and motion owner.

M04 provides a qualified current radar object; M003 supplies the held model
history. Constructed source-label transitions test identity, not vehicle timing.
"""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from openpilot.cereal import log
from openpilot.selfdrive.controls.lib.lead_launch import LEAD_LAUNCH_CONFIRM_NS
from openpilot.selfdrive.controls.lib.lead_qualification import MAX_MEASUREMENT_AGE_NS, lead_identity
from openpilot.selfdrive.controls.tests.longitudinal_test_support import isolated_params as isolated_params
from openpilot.selfdrive.controls.tests.test_m003_stationary_departure import (
  FIRST, held_case as held_case, load_snapshot, step,
)


FIXTURE = json.loads((Path(__file__).with_name('fixtures') / 'm04_launch_identity.json').read_text())


def load_current_observations(state, frame):
  load_snapshot(state, FIRST, frame)
  radar = deepcopy(FIXTURE['radar'])
  now_ns = 1_000_000_000 + frame * 50_000_000
  shift = now_ns - radar['mdMonoTime']
  radar['mdMonoTime'] += shift
  for slot in ('leadOne', 'leadTwo'):
    for field, value in radar[slot].get('evidence', {}).items():
      if field.endswith('MonoTime') and value:
        radar[slot]['evidence'][field] += shift
  state['radarState'] = log.RadarState.new_message(**radar)


def confirm_current_track(case):
  planner, _, state = case
  for frame in range(20, 26):
    load_current_observations(state, frame)
    state['radarState'].leadOne.evidence.source = 'radar' if frame % 2 == 0 else 'fused'
    plan, _ = step(case, frame)
    launch = planner.reference_planner.lead_launch
    assert plan.solverStatus == 0 and plan.motionIntent.valid
    assert launch.confirmed_ns == (frame - 20) * 50_000_000
    assert launch.active == (frame == 25)
  return planner.reference_planner.lead_launch


def test_current_physical_track_keeps_confirmation_across_radar_fused_labels(held_case):
  launch = confirm_current_track(held_case)
  assert launch.confirmed_ns == LEAD_LAUNCH_CONFIRM_NS
  planner, _, state = held_case
  identity = lead_identity(state['radarState'].leadOne)
  assert launch.leads[0].identity == ('track', *identity)
  load_current_observations(state, 26)
  state['radarState'].leadOne.evidence.source = 'radar'
  step(held_case, 26)
  assert planner.reference_planner.lead_launch_active


@pytest.mark.parametrize('change', ['track', 'session', 'unqualified', 'missing', 'coasting', 'expired_measurement'])
def test_current_identity_or_authority_loss_clears_launch_without_spending_old_confirmation(held_case, change):
  confirm_current_track(held_case)
  planner, _, state = held_case
  load_current_observations(state, 26)
  radar = state['radarState']
  lead = radar.leadOne
  if change == 'track':
    lead.evidence.trackId += 1
    lead.radarTrackId += 1
  elif change == 'session':
    radar.fusionSessionId = 'm04-new-session'
    for current in (radar.leadOne, radar.leadTwo):
      if lead_identity(current) is not None:
        current.evidence.sessionId = radar.fusionSessionId
  elif change == 'unqualified':
    lead.evidence.departureQualified = False
  elif change == 'missing':
    lead.present = False
  elif change == 'coasting':
    lead.evidence.coasting = True
  else:
    lead.evidence.positionMonoTime = 1_000_000_000 + 26 * 50_000_000 - MAX_MEASUREMENT_AGE_NS - 1
  plan, _ = step(held_case, 26)
  assert plan.solverStatus == 0 and plan.motionIntent.valid
  assert not planner.reference_planner.lead_launch_active
  assert not planner.reference_planner.lead_launch.active
  assert planner.reference_planner.lead_launch.confirmed_ns == 0
