from types import SimpleNamespace

import pytest

from openpilot.cereal import messaging
from openpilot.common.constants import CV
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import CruiseTarget, LongitudinalPlannerSP
from openpilot.tools.longitudinal import replay_longitudinal_regressions
from openpilot.tools.longitudinal.replay_pedal_switches import recorded_sp_targets


CRUISE_SPEED_CASES = [
  pytest.param(60.0 * CV.KPH_TO_MS, 60.0 * CV.KPH_TO_MS, id='non-float32-cruise'),
  pytest.param(0.0, 17.0, id='force-decel'),
  pytest.param(V_CRUISE_MAX * CV.KPH_TO_MS, V_CRUISE_MAX * CV.KPH_TO_MS, id='already-clipped-cruise'),
]


def decision(source, speed=17.0):
  plan = messaging.new_message('longitudinalPlanSP').longitudinalPlanSP
  plan.longitudinalPlanSource = source
  plan.vTarget = speed
  plan.aTarget = -2.0
  plan.smartCruiseControl.vision.aTarget = -0.4
  plan.smartCruiseControl.map.aTarget = -0.8
  plan.speedLimit.assist.aTarget = -1.2
  return plan.as_reader()


@pytest.mark.parametrize('source,expected', [('cruise', 0.2), ('sccVision', -0.4), ('sccMap', -0.8), ('speedLimitAssist', -1.2)])
def test_selected_upstream_acceleration_is_not_the_final_arbitrated_target(source, expected):
  plan = decision(source)
  before = plan.to_dict()
  cruise_speed = 60.0 * CV.KPH_TO_MS
  speed, accel = recorded_sp_targets(plan, 0.2, cruise_speed)
  assert speed == (cruise_speed if source == 'cruise' else 17.0)
  assert accel == pytest.approx(expected)
  assert accel != plan.aTarget
  assert plan.to_dict() == before


@pytest.mark.parametrize('previous', [-3.0, -0.2, 0.0, 0.6, 2.0])
def test_cruise_uses_the_replayed_acceleration_state(previous):
  assert recorded_sp_targets(decision('cruise'), previous, 17.0) == (17.0, previous)


@pytest.mark.parametrize('cruise_speed,published_speed', CRUISE_SPEED_CASES)
def test_cruise_preserves_the_exact_callback_speed(cruise_speed, published_speed):
  plan = decision('cruise', published_speed)
  assert plan.vTarget != cruise_speed
  previous_accel = 0.123456789
  assert recorded_sp_targets(plan, previous_accel, cruise_speed) == (cruise_speed, previous_accel)


def test_cruise_speed_is_an_explicit_required_input():
  with pytest.raises(TypeError):
    recorded_sp_targets(decision('cruise'), 0.0)


@pytest.mark.parametrize('source', ['sccVision', 'sccMap', 'speedLimitAssist'])
def test_upstream_targets_do_not_depend_on_the_replayed_acceleration(source):
  assert recorded_sp_targets(decision(source), -3.0, 0.0) == recorded_sp_targets(decision(source), 1.0, 40.0)


def test_unknown_source_is_not_silently_replaced_with_the_final_output():
  with pytest.raises(ValueError, match='Unsupported'):
    recorded_sp_targets(SimpleNamespace(longitudinalPlanSource='unknown', aTarget=-2.0), 0.0, 17.0)


@pytest.mark.parametrize('source', ['cruise', 'sccVision', 'sccMap', 'speedLimitAssist'])
def test_frozen_target_uses_the_current_policy_type_without_overwriting_final_output(source):
  plan = decision(source)
  speed, accel = recorded_sp_targets(plan, 0.123456789, 60.0 * CV.KPH_TO_MS)
  planner = SimpleNamespace(output_a_target=-9.0)
  target_type = replay_longitudinal_regressions.sp_target_return_type(LongitudinalPlannerSP.update_targets)
  assert target_type is CruiseTarget
  target = replay_longitudinal_regressions.frozen_sp_target(planner, plan.longitudinalPlanSource, speed, accel, target_type)
  assert target == CruiseTarget(plan.longitudinalPlanSource, speed, accel)
  assert planner.cruise_target is target
  assert planner.source == plan.longitudinalPlanSource
  assert planner.output_v_target == speed
  assert planner.output_a_target == -9.0


def test_historical_tuple_target_contract_is_explicitly_preserved():
  def historical_targets(instance, state, v_ego, a_ego, v_cruise) -> tuple[float, float]:
    raise AssertionError('Selecting the API must not execute SP')

  target_type = replay_longitudinal_regressions.sp_target_return_type(historical_targets)
  assert target_type is None
  planner = SimpleNamespace(output_a_target=-9.0)
  assert replay_longitudinal_regressions.frozen_sp_target(planner, 'cruise', 17.0, 0.123456789, target_type) == (17.0, 0.123456789)
  assert planner.source == 'cruise'
  assert planner.output_a_target == -9.0


def test_unknown_target_contract_is_rejected_before_replay():
  def unknown_targets(instance, state, v_ego, a_ego, v_cruise) -> dict:
    raise AssertionError('Selecting the API must not execute SP')

  with pytest.raises(ValueError, match='Unsupported SP target return contract'):
    replay_longitudinal_regressions.sp_target_return_type(unknown_targets)


def test_planner_only_replay_rejects_motion_runtime_without_consumer_ack(monkeypatch, tmp_path):
  segment = tmp_path / 'test-route--0'
  segment.mkdir()
  (segment / 'rlog.zst').touch()
  monkeypatch.setattr(replay_longitudinal_regressions, 'route_metadata', lambda paths: (None, None, 0.0))
  monkeypatch.setattr(replay_longitudinal_regressions, 'LongitudinalPlanner', lambda *arguments: SimpleNamespace(motion_enabled=True))
  with pytest.raises(ValueError, match='no motion-consumer ACK'):
    replay_longitudinal_regressions.replay(tmp_path, 'test-route', 0.0, 1.0)


@pytest.mark.parametrize('cruise_speed,published_speed', CRUISE_SPEED_CASES)
def test_cruise_only_replay_preserves_the_callback_speed(monkeypatch, tmp_path, cruise_speed, published_speed):
  route = 'test-route'
  segment = tmp_path / f'{route}--0'
  segment.mkdir()
  (segment / 'rlog.zst').touch()
  plan_sp = decision('cruise', published_speed)
  previous_accel = 0.123456789
  observed = []

  class TargetObserved(Exception):
    pass

  class PlannerProbe:
    def __init__(self, params, params_sp):
      pass

    def is_e2e(self, state):
      return False

    def update(self, state):
      observed.append(LongitudinalPlannerSP.update_targets(self, state, 0.0, previous_accel, cruise_speed))
      raise TargetObserved

  monkeypatch.setattr(replay_longitudinal_regressions, 'LongitudinalPlanner', PlannerProbe)
  monkeypatch.setattr(replay_longitudinal_regressions, 'route_metadata', lambda paths: (None, None, 0.0))
  monkeypatch.setattr(replay_longitudinal_regressions, 'frames',
                      lambda paths, origin, start, end, coverage: iter([(0.0, {}, None, plan_sp, True, 0)]))
  with pytest.raises(TargetObserved):
    replay_longitudinal_regressions.replay(tmp_path, route, 0.0, 1.0)
  assert observed == [CruiseTarget(plan_sp.longitudinalPlanSource, cruise_speed, previous_accel)]
