from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from opendbc.can import CANParser
from opendbc.car import Bus
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.hondacan import CanBus
from opendbc.car.honda.values import CAR, CarControllerParams, DBC
from opendbc.car.structs import car
from openpilot.cereal import custom
from openpilot.common.hardware.hw import Paths
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.controls.lib.longcontrol import LongControl, LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode
from openpilot.selfdrive.controls.lib.longitudinal_planner import ALLOW_THROTTLE_THRESHOLD, get_coast_accel
from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent, MotionPhase
from openpilot.selfdrive.controls.lib.unified_longitudinal_planner import cruise_accel_limit
from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import Publisher, advance, make_planner, make_state


ACCEL_LIMITS = (CarControllerParams.BOSCH_ACCEL_MIN, CarControllerParams.BOSCH_ACCEL_MAX)
INPUT_AGE_NS = 100_000_000


@pytest.fixture(autouse=True)
def isolated_params(tmp_path, monkeypatch):
  monkeypatch.setenv('PARAMS_ROOT', str(tmp_path / 'params'))
  with patch.object(Paths, 'comma_home', return_value=str(tmp_path / 'comma')):
    with OpenpilotPrefix():
      yield


def candidate(speed=12.0):
  planner, extension = make_planner(speed)
  return planner, LongControl(planner.CP, extension)


def measured_state(speed=12.0):
  state = make_state(speed=speed)
  state['carState'].canValid = True
  state['carControl'].orientationNED = [0.0, 0.02, 0.0]
  state['controlsState'].motionState.planningMode = int(LongitudinalMode.UNIFIED)
  return state


def set_standby(state):
  state['carState'].vCruise = V_CRUISE_UNSET
  state['controlsState'].longControlState = LongCtrlState.off
  state['selfdriveState'].enabled = False
  state['carControl'].enabled = False
  state['carControl'].longActive = False


def update_at(planner, state, now_ns):
  # Use the real update entry point in both the pre-fix and candidate implementation.
  # The controlled consumer clock is independent of every input message timestamp.
  with patch('time.monotonic_ns', return_value=now_ns):
    planner.update(state)


def pitch_snapshot(planner):
  return (planner.pitch_bias.filter.x, planner.pitch_bias.filter.initialized,
          planner.pitch_bias.diagnostic_calls, planner.pitch_bias.diagnostic_learn_calls)


def test_cold_standby_learns_from_first_qualified_measurement_without_running_reference():
  planner, owner = candidate()
  legacy, _ = make_planner(platform=CAR.HONDA_CIVIC_BOSCH)
  assert legacy.longitudinal_mode == LongitudinalMode.LEGACY
  state = measured_state()
  set_standby(state)
  samples = ((2.5, 0.05), (6.0, 0.02923324704170227), (9.0, 0.035), (0.0, 0.07), (12.0, 0.045))
  with patch.object(planner.mpc, 'update', wraps=planner.mpc.update) as mpc_update:
    for frame, (speed, pitch) in enumerate(samples):
      now = advance(state, frame) + 5_000_000
      state['carState'].vEgo = speed
      state['carState'].standstill = speed == 0.0
      state['carControl'].orientationNED = [0.0, pitch, 0.0]
      update_at(legacy, state, now)
      update_at(planner, state, now)
      assert pitch_snapshot(planner) == pitch_snapshot(legacy)
      assert not planner.motion_intent.valid
      assert not planner.motion_intent.ready
      assert not planner.motion_intent.release_allowed
      assert planner.v_desired_filter.x == speed
      if frame == 1:
        assert planner.pitch_bias.filter.x == pitch
      assert owner.update(False, state['carState'], planner.output_a_target, planner.output_should_stop, ACCEL_LIMITS,
                          intent=planner.motion_intent, plan_time_ns=now, model_time_ns=state.logMonoTime['modelV2'],
                          now_ns=now + 1_000_000, plan_valid=True) == 0.0
    assert mpc_update.call_count == 0
  assert planner.pitch_bias.diagnostic_calls == len(samples)
  assert planner.pitch_bias.diagnostic_learn_calls == 3
  assert owner.motion_phase == MotionPhase.OFF


class HondaCommands:
  """Real Honda mapping and CAN decoding; vehicle state is a bounded sensor fixture."""

  def __init__(self, planner, extension):
    dbc = DBC[planner.CP.carFingerprint][Bus.pt]
    self.controller = CarController({Bus.pt: dbc}, planner.CP, extension)
    self.parser = CANParser(dbc, [('ACC_CONTROL', 50)], CanBus(planner.CP).pt)
    self.extension = custom.CarControlSP.new_message()
    self.vehicle = SimpleNamespace(v_cruise_factor=1.0, is_metric=True, acc_hud={}, lkas_hud={}, stock_brake={})

  def update(self, state, owner, acceleration, now):
    command = car.CarControl.new_message()
    command.enabled = command.longActive = True
    command.actuators.accel = acceleration
    command.actuators.longControlState = owner.long_control_state
    self.vehicle.out = state['carState']
    _, packets = self.controller.update(command.as_reader(), self.extension.as_reader(), self.vehicle, now)
    self.parser.update([(now, packets)])
    return [packet for packet in packets if packet[0] == 0x1df]


def test_unified_cold_to_set_then_throttle_threshold_limits_reference_and_honda_commands():
  speed = 16.727760314941406
  assert ALLOW_THROTTLE_THRESHOLD == 0.4
  planner, owner = candidate(speed)
  legacy, _ = make_planner(speed, platform=CAR.HONDA_CIVIC_BOSCH)
  state = measured_state(speed)
  state['carState'].aEgo = -0.04944811016321182
  set_standby(state)
  for frame, pitch in enumerate(np.linspace(0.02923324704170227, 0.047564711421728134, 60)):
    now = advance(state, frame) + 5_000_000
    state['carControl'].orientationNED = [0.0, float(pitch), 0.0]
    update_at(legacy, state, now)
    update_at(planner, state, now)
  assert pitch_snapshot(planner) == pitch_snapshot(legacy)

  # The first SET still uses current measured speed/accel with no invented
  # acceleration history; execution starts after the owner becomes active.
  state['carState'].vCruise = (speed + 0.4) * 3.6
  state['selfdriveState'].enabled = state['carControl'].enabled = True
  now = advance(state, 60) + 5_000_000
  update_at(legacy, state, now)
  with patch.object(planner.mpc, 'set_weights', wraps=planner.mpc.set_weights) as weights:
    update_at(planner, state, now)
  assert not weights.call_args.args[0]
  assert planner.mpc.x0[1] == pytest.approx(speed)
  assert planner.mpc.x0[2] == pytest.approx(state['carState'].aEgo)

  state['controlsState'].longControlState = LongCtrlState.pid
  state['carControl'].longActive = True
  actual_can = HondaCommands(planner, owner.CP_SP)
  brake_rows = 0
  for frame, throttle in enumerate([0.5] * 5 + [0.399999] * 18 + [0.5] * 10, 61):
    now = advance(state, frame) + 5_000_000
    state['modelV2'].meta.disengagePredictions.gasPressProbs = [1.0, throttle, 1.0, 1.0, 1.0, 1.0]
    update_at(legacy, state, now)
    with patch('openpilot.selfdrive.controls.lib.unified_longitudinal_planner.cruise_accel_limit', wraps=cruise_accel_limit) as limit:
      update_at(planner, state, now)
    limit.assert_called_once()
    expected_coast = get_coast_accel(state['carControl'].orientationNED[1] - legacy.pitch_bias.value)
    assert limit.call_args.args[4] == expected_coast
    assert pitch_snapshot(planner) == pitch_snapshot(legacy)
    assert planner.allow_throttle == (throttle > ALLOW_THROTTLE_THRESHOLD)
    assert planner.mpc.solution_status == 0
    np.testing.assert_array_equal(planner.mpc.x0[1:], [speed, state['carState'].aEgo])
    if not planner.allow_throttle:
      np.testing.assert_allclose(planner.mpc.params[:, 1], expected_coast, atol=1e-12, rtol=0.0)
      assert planner.output_a_target <= expected_coast
    publisher = Publisher()
    with patch('openpilot.selfdrive.controls.lib.longitudinal_planner.time.monotonic_ns', return_value=now):
      planner.publish(state, publisher)
    publication = publisher.messages['longitudinalPlan']
    plan = publication.longitudinalPlan
    assert publication.valid and plan.motionIntent.valid
    assert plan.allowThrottle == planner.allow_throttle
    for tick in range(5):
      control_now = now + 1_000_000 + tick * 10_000_000
      actual = float(owner.update(True, state['carState'], plan.aTarget, plan.shouldStop, ACCEL_LIMITS,
                                 intent=MotionIntent.from_message(plan.motionIntent), plan_time_ns=publication.logMonoTime,
                                 model_time_ns=plan.modelMonoTime, now_ns=control_now, plan_valid=publication.valid))
      assert np.isfinite(actual) and ACCEL_LIMITS[0] <= actual <= ACCEL_LIMITS[1]
      assert not owner.motion_degraded
      if not planner.allow_throttle:
        assert actual <= expected_coast
      actual_packets = actual_can.update(state, owner, actual, control_now)
      if actual_packets:
        wire = actual_can.parser.vl['ACC_CONTROL']
        # ACCEL_COMMAND has 0.01 m/s² wire resolution in the Honda Bosch DBC.
        assert wire['ACCEL_COMMAND'] == pytest.approx(actual, abs=0.0051)
        assert bool(wire['BRAKE_REQUEST']) == (actual < CarControllerParams.BOSCH_GAS_LOOKUP_BP[0])
        if wire['BRAKE_REQUEST']:
          assert wire['GAS_COMMAND'] <= 0.0
          brake_rows += not planner.allow_throttle
  assert brake_rows > 0


@pytest.mark.parametrize('manual', ['gasPressed', 'brakePressed', 'override'])
def test_passive_measurement_continues_under_manual_control_without_retaining_authority(manual):
  planner, owner = candidate()
  state = measured_state()
  now = advance(state, 0) + 5_000_000
  update_at(planner, state, now)
  previous = pitch_snapshot(planner)
  if manual == 'override':
    state['carControl'].cruiseControl.override = True
  else:
    setattr(state['carState'], manual, True)
  state['carControl'].orientationNED = [0.0, 0.05, 0.0]
  now = advance(state, 1) + 5_000_000
  update_at(planner, state, now)
  assert planner.pitch_bias.diagnostic_learn_calls == previous[-1] + 1
  assert planner.pitch_bias.filter.x > previous[0]
  assert not planner.motion_intent.allow_creep
  assert not planner.motion_intent.release_allowed
  active = manual != 'override'  # controlsd revokes longActive for cruise override.
  result = owner.update(active, state['carState'], planner.output_a_target, planner.output_should_stop, ACCEL_LIMITS,
                        intent=planner.motion_intent, plan_time_ns=now, model_time_ns=state.logMonoTime['modelV2'],
                        now_ns=now + 1_000_000, plan_valid=True)
  assert result == 0.0
  assert owner.long_control_state == LongCtrlState.off


@pytest.mark.parametrize('guard', ['owner_version', 'owner_enabled', 'owner_mode', 'model_shape'])
def test_reference_admission_fault_does_not_freeze_independent_valid_measurements(guard):
  planner, _ = candidate()
  state = measured_state()
  if guard == 'owner_version':
    state['controlsState'].motionState.version = 0
  elif guard == 'owner_enabled':
    state['controlsState'].motionState.enabled = False
  elif guard == 'owner_mode':
    state['controlsState'].motionState.planningMode = int(LongitudinalMode.LEGACY)
  else:
    state['modelV2'].velocity.x = []
  update_at(planner, state, advance(state, 0) + 5_000_000)
  assert planner.pitch_bias.diagnostic_learn_calls == 1
  assert not planner.motion_intent.valid
  assert not planner.motion_intent.release_allowed
  assert planner.mpc.diagnostic_solve_calls == 0


@pytest.mark.parametrize('service', ['carState', 'carControl'])
@pytest.mark.parametrize('fault', ['invalid', 'missing_id', 'zero_id', 'boolean_id', 'float_id', 'overflow_id',
                                  'future', 'stale', 'duplicate', 'out_of_order'])
def test_bad_or_repeated_measurement_cannot_change_learned_bias(service, fault):
  planner, _ = candidate()
  state = measured_state()
  now = advance(state, 0) + 5_000_000
  update_at(planner, state, now)
  before = pitch_snapshot(planner)
  first_id = state.logMonoTime[service]
  now = advance(state, 1) + 5_000_000
  # A reused identity with different contents is not a new observation.
  state['carControl'].orientationNED = [0.0, 0.05, 0.0]
  if fault == 'invalid':
    state.valid[service] = False
  elif fault == 'missing_id':
    del state.logMonoTime[service]
  elif fault == 'zero_id':
    state.logMonoTime[service] = 0
  elif fault == 'boolean_id':
    state.logMonoTime[service] = True
  elif fault == 'float_id':
    state.logMonoTime[service] = float(now - 1)
  elif fault == 'overflow_id':
    state.logMonoTime[service] = 1 << 64
  elif fault == 'future':
    state.logMonoTime[service] = now + 1
  elif fault == 'stale':
    state.logMonoTime[service] = now - INPUT_AGE_NS
  elif fault == 'duplicate':
    state.logMonoTime[service] = first_id
  else:
    state.logMonoTime[service] = first_id - 1
  update_at(planner, state, now)
  assert pitch_snapshot(planner) == before


def test_expired_pair_cannot_define_its_own_clock_and_unrelated_future_service_cannot_spoil_a_fresh_pair():
  planner, _ = candidate()
  state = measured_state()
  timestamp = advance(state, 0)
  update_at(planner, state, timestamp + INPUT_AGE_NS)
  assert planner.pitch_bias.diagnostic_learn_calls == 0
  timestamp = advance(state, 3)
  now = timestamp + 5_000_000
  state.logMonoTime['gpsLocation'] = now + 100_000_000_000
  update_at(planner, state, now)
  assert planner.pitch_bias.diagnostic_learn_calls == 1


@pytest.mark.parametrize('age_ns,accepted', [(INPUT_AGE_NS - 1, True), (INPUT_AGE_NS, False), (INPUT_AGE_NS + 1, False)])
def test_passive_sample_age_uses_explicit_consumer_time_at_boundary(age_ns, accepted):
  planner, _ = candidate()
  state = measured_state()
  timestamp = advance(state, 0)
  update_at(planner, state, timestamp + age_ns)
  assert planner.pitch_bias.diagnostic_learn_calls == int(accepted)


def test_invalid_sample_identity_cannot_be_repaired_in_place_and_future_ids_do_not_poison_recovery():
  planner, _ = candidate()
  state = measured_state()
  now = advance(state, 0) + 5_000_000
  state.valid['carControl'] = False
  update_at(planner, state, now)
  assert planner.pitch_bias.diagnostic_learn_calls == 0
  state.valid['carControl'] = True
  update_at(planner, state, now + 1)
  assert planner.pitch_bias.diagnostic_learn_calls == 0
  now = advance(state, 1) + 5_000_000
  state.logMonoTime['carControl'] = now + 100_000_000_000
  update_at(planner, state, now)
  assert planner.pitch_bias.diagnostic_learn_calls == 0
  now = advance(state, 2) + 5_000_000
  update_at(planner, state, now)
  assert planner.pitch_bias.diagnostic_learn_calls == 1


def test_reversed_consumer_clock_cannot_admit_new_measurements():
  planner, _ = candidate()
  state = measured_state()
  now = advance(state, 0) + 5_000_000
  update_at(planner, state, now)
  before = pitch_snapshot(planner)
  for service in ('carState', 'carControl'):
    state.logMonoTime[service] += 1
  state['carControl'].orientationNED = [0.0, 0.05, 0.0]
  update_at(planner, state, now - 1)
  assert pitch_snapshot(planner) == before


@pytest.mark.parametrize('now_ns', [0, -1, True, 1_005_000_000.0, float('nan'), 1 << 64])
def test_invalid_consumer_clock_cannot_initialize_passive_state(now_ns):
  planner, _ = candidate()
  state = measured_state()
  advance(state, 0)
  update_at(planner, state, now_ns)
  assert planner.pitch_bias.diagnostic_calls == 0
  assert not planner.pitch_bias.filter.initialized


def test_interleaved_new_and_duplicate_measurements_require_a_new_pair_to_recover():
  planner, _ = candidate()
  state = measured_state()
  initial_time = advance(state, 0)
  update_at(planner, state, initial_time + 5_000_000)
  before = pitch_snapshot(planner)
  state['carControl'].orientationNED = [0.0, 0.05, 0.0]
  state.logMonoTime['carControl'] += 50_000_000
  update_at(planner, state, initial_time + 55_000_000)
  assert pitch_snapshot(planner) == before
  state.logMonoTime['carState'] += 60_000_000
  update_at(planner, state, initial_time + 65_000_000)
  assert pitch_snapshot(planner) == before
  now = advance(state, 2) + 5_000_000
  update_at(planner, state, now)
  assert planner.pitch_bias.diagnostic_learn_calls == before[-1] + 1


@pytest.mark.parametrize('service', ['carState', 'carControl'])
@pytest.mark.parametrize('invalid_flag', [1, 'true', [True], None])
def test_truthy_non_boolean_validity_is_not_measurement_evidence(service, invalid_flag):
  planner, _ = candidate()
  state = measured_state()
  state.valid[service] = invalid_flag
  update_at(planner, state, advance(state, 0) + 5_000_000)
  assert planner.pitch_bias.diagnostic_learn_calls == 0


@pytest.mark.parametrize('field,value', [('pitch', float('nan')), ('pitch', float('inf')),
                                       ('speed', float('nan')), ('speed', float('inf')), ('speed', -1.0)])
def test_nonfinite_or_negative_measurements_never_initialize_passive_state(field, value):
  planner, _ = candidate()
  state = measured_state()
  # Refuse executable reference independently of the measurement. This keeps the
  # test focused on passive admission, including a physically invalid speed.
  state['controlsState'].motionState.enabled = False
  if field == 'pitch':
    state['carControl'].orientationNED = [0.0, value, 0.0]
  else:
    state['carState'].vEgo = value
  update_at(planner, state, advance(state, 0) + 5_000_000)
  assert not planner.pitch_bias.filter.initialized
  assert planner.pitch_bias.diagnostic_calls == 0
  assert planner.pitch_bias.diagnostic_learn_calls == 0


def test_legacy_path_retains_existing_sampling_semantics():
  planner, _ = make_planner(platform=CAR.HONDA_CIVIC_BOSCH)
  assert planner.longitudinal_mode == LongitudinalMode.LEGACY
  state = measured_state()
  state['controlsState'].motionState.planningMode = int(LongitudinalMode.LEGACY)
  now = advance(state, 0) + 5_000_000
  for pitch in (0.02, 0.03):
    state['carControl'].orientationNED = [0.0, pitch, 0.0]
    update_at(planner, state, now)
  assert planner.pitch_bias.diagnostic_calls == 2
  assert planner.pitch_bias.diagnostic_learn_calls == 2
  assert planner.pitch_bias.filter.x > 0.02
