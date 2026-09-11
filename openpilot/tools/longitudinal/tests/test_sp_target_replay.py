from types import SimpleNamespace

import pytest

from openpilot.cereal import messaging
from openpilot.tools.longitudinal.replay_pedal_switches import recorded_sp_targets


def decision(source):
  plan = messaging.new_message('longitudinalPlanSP').longitudinalPlanSP
  plan.longitudinalPlanSource = source
  plan.vTarget = 17.0
  plan.aTarget = -2.0
  plan.smartCruiseControl.vision.aTarget = -0.4
  plan.smartCruiseControl.map.aTarget = -0.8
  plan.speedLimit.assist.aTarget = -1.2
  return plan.as_reader()


@pytest.mark.parametrize('source,expected', [('cruise', 0.2), ('sccVision', -0.4), ('sccMap', -0.8), ('speedLimitAssist', -1.2)])
def test_selected_upstream_acceleration_is_not_the_final_arbitrated_target(source, expected):
  plan = decision(source)
  before = plan.to_dict()
  speed, accel = recorded_sp_targets(plan, 0.2)
  assert speed == 17.0
  assert accel == pytest.approx(expected)
  assert accel != plan.aTarget
  assert plan.to_dict() == before


@pytest.mark.parametrize('previous', [-3.0, -0.2, 0.0, 0.6, 2.0])
def test_cruise_uses_the_replayed_acceleration_state(previous):
  assert recorded_sp_targets(decision('cruise'), previous) == (17.0, previous)


@pytest.mark.parametrize('source', ['sccVision', 'sccMap', 'speedLimitAssist'])
def test_upstream_targets_do_not_depend_on_the_replayed_acceleration(source):
  assert recorded_sp_targets(decision(source), -3.0) == recorded_sp_targets(decision(source), 1.0)


def test_unknown_source_is_not_silently_replaced_with_the_final_output():
  with pytest.raises(ValueError, match='Unsupported'):
    recorded_sp_targets(SimpleNamespace(longitudinalPlanSource='unknown', aTarget=-2.0), 0.0)
