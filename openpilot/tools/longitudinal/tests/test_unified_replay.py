from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.tools.longitudinal import replay_pedal_switches as replay
from openpilot.selfdrive.controls.lib.longitudinal_mode import LongitudinalMode


class Snapshot(SimpleNamespace):
  def as_builder(self):
    return deepcopy(self)

  def as_reader(self):
    return self


class FixtureIntent(SimpleNamespace):
  def is_standby_valid(self):
    return (not self.ready and not self.valid and not self.moving_lead and not self.release_allowed and not self.allow_creep
            and self.braking_accel is None and self.reasons in (0, int(replay.motion_runtime.StopReason.CRUISE)))


def test_flagless_constructor_is_validated_against_its_actual_runtime_mode():
  class Legacy:
    def __init__(self, params, params_sp):
      pass

  assert replay.component_options(Legacy, False) == {}
  assert replay.component_options(Legacy, True) == {}
  with pytest.raises(ValueError, match='Requested replay mode 2'):
    replay.require_component_mode(Legacy(None, None), 2, 'controller')
  with pytest.raises(ValueError, match='retired supervised'):
    replay.component_options(Legacy, False, True)


def test_planner_clock_keyword_is_opt_in_for_the_tested_public_api():
  class Legacy:
    def update(self, state):
      pass

  class Candidate:
    def update(self, state, *, now_ns=None):
      pass

  assert replay.planner_update_options(Legacy(), 2 ** 63 + 1) == {}
  assert replay.planner_update_options(Candidate(), 2 ** 63 + 1) == {'now_ns': 2 ** 63 + 1}
  with pytest.raises(ValueError, match='exact positive uint64'):
    replay.planner_update_options(Candidate(), float(2 ** 63 + 1))


def test_motion_version_is_imported_from_the_tested_runtime():
  from openpilot.selfdrive.controls.lib.motion_supervisor import MOTION_INTENT_VERSION
  assert replay.MOTION_PROTOCOL_VERSION == MOTION_INTENT_VERSION


def test_new_constructor_explicitly_selects_both_modes():
  class Candidate:
    def __init__(self, params, params_sp, *, unified=None):
      pass

  assert replay.component_options(Candidate, False) == {'unified': False}
  assert replay.component_options(Candidate, True) == {'unified': True}


def test_recorded_targets_cannot_be_mixed_with_candidate_intent():
  with pytest.raises(ValueError, match='own planner'):
    replay.replay_window(None, {}, recorded_plan=True, unified=True)


def replay_fixture(monkeypatch, *, consume=True, legacy_api=False, solver_status=0, publish_solver_status=True,
                   standby_eligible=False, publish_standby=False, envelope_api=False, owner_established=False,
                   rearm_api=False, rearm_wait=False, automatic_api=False):
  """Historical protocol doubles plus a flagless, platform-selected API double."""
  intent = FixtureIntent(version=replay.MOTION_PROTOCOL_VERSION, mode=2, reasons=0, moving_lead=False, release_allowed=False,
                         allow_creep=True, valid=True, ready=True, braking_accel=None)
  controls, encoded = [], []

  class Planner:
    def __init__(self, params, params_sp, *, unified=False, supervised=False):
      self.longitudinal_mode = replay.selected_mode(unified, supervised)
      self.unified = unified
      self.motion_enabled = self.longitudinal_mode != 0
      self.motion_intent = intent
      self.motion_intent.mode = self.longitudinal_mode
      self.output_a_target = 0.15
      self.output_should_stop = False
      self.v_desired_trajectory = [0.0]
      self.a_cruise = 0.0
      self.fcw = False
      self.allow_throttle = True
      self.mpc = SimpleNamespace(solution_status=solver_status, source='cruise')

    def update(self, state):
      if self.motion_enabled:
        assert state['controlsState'].motionState.version == replay.MOTION_PROTOCOL_VERSION
        assert state['controlsState'].motionState.enabled
        assert state['controlsState'].motionState.planningMode == self.longitudinal_mode

    def is_e2e(self, state):
      return False

    def publish(self, state, publisher):
      plan = Snapshot(modelMonoTime=state.logMonoTime['modelV2'], aTarget=float(np.float32(self.output_a_target)),
                      shouldStop=self.output_should_stop,
                      motionIntent=self.motion_intent if self.motion_enabled else Snapshot(version=0, mode=0))
      if publish_solver_status:
        plan.solverStatus = self.mpc.solution_status
      publisher.send('longitudinalPlan', Snapshot(
        valid=bool(state.all_checks() and self.mpc.solution_status == 0
                   and (not self.motion_enabled or self.motion_intent.valid or publish_standby)),
        longitudinalPlan=plan))

  missing = object()

  class Longitudinal:
    def __init__(self, params, params_sp, *, unified=False, supervised=False):
      self.longitudinal_mode = replay.selected_mode(unified, supervised)
      self.unified = self.longitudinal_mode != 0
      self.motion_supervisor = SimpleNamespace(_last_plan_time_ns=None, _last_model_time_ns=None,
                                              _last_intent=None, _last_target=None)
      self.long_control_state = replay.LongCtrlState.off if standby_eligible else replay.LongCtrlState.stopping
      self.resolved_should_stop = not standby_eligible
      self.motion_phase = 0 if standby_eligible else 3
      self.motion_degraded = False

    @property
    def can_standby(self):
      return self.unified and self.long_control_state == replay.LongCtrlState.off and self.motion_supervisor._last_plan_time_ns is None

    @property
    def motion_reasons(self):
      return int(replay.motion_runtime.StopReason.INVALID) if self.motion_degraded else 0

    def update(self, *arguments, solver_status=missing, **options):
      if solver_status is not missing:
        options['solver_status'] = solver_status
      controls.append((arguments, options))
      if not arguments[0]:
        self.long_control_state = replay.LongCtrlState.off
        self.resolved_should_stop = False
        self.motion_phase = 0
        return 0.0
      self.motion_degraded = bool(self.unified and (not options.get('plan_valid', True) or options.get('solver_status', 0) != 0
                                                  or not options['intent'].ready or not options['intent'].valid))
      if consume and self.unified and not self.motion_degraded:
        self.motion_supervisor._last_plan_time_ns = options['plan_time_ns']
        self.motion_supervisor._last_model_time_ns = options['model_time_ns']
        self.motion_supervisor._last_intent = options['intent']
        self.motion_supervisor._last_target = arguments[2]
      return -1.0

  class LegacyLongitudinal(Longitudinal):
    def __init__(self, params, params_sp):
      super().__init__(params, params_sp)

    def update(self, *arguments):
      return super().update(*arguments)

  class EnvelopeLongitudinal(Longitudinal):
    @property
    def motion_owner_established(self):
      return owner_established

    def update(self, *arguments, solver_status=missing, braking_plan_valid=False, **options):
      return super().update(*arguments, solver_status=solver_status, braking_plan_valid=braking_plan_valid, **options)

  class RearmLongitudinal(EnvelopeLongitudinal):
    def __init__(self, params, params_sp, *, unified=False, supervised=False):
      super().__init__(params, params_sp, unified=unified, supervised=supervised)
      self.CP = params
      self.last_output_accel = -1.0
      self.motion_waiting_for_fresh_input = False
      supervisor = self.motion_supervisor
      supervisor._require_new_input = True
      supervisor._rearm_after_fault = False
      supervisor._last_creep_observation_ns = 900_000_000
      supervisor._last_now_ns = 1_010_000_000
      supervisor._latest_input_ids = lambda: (1_010_000_000, 1_000_000_000)
      supervisor._timestamp_valid = lambda value: type(value) is int and 0 < value < 2 ** 64
      supervisor._valid_input = lambda intent, target, plan, model, now: (
        intent.valid and intent.ready and 0 < model <= plan <= now and now - model <= 250_000_000)

    @property
    def motion_input_constrained(self):
      return self.motion_degraded or self.motion_waiting_for_fresh_input

    def update(self, *arguments, solver_status=missing, braking_plan_valid=False, standby_input=False, **options):
      result = super().update(*arguments, solver_status=solver_status, braking_plan_valid=braking_plan_valid,
                              standby_input=standby_input, **options)
      self.motion_waiting_for_fresh_input = bool(rearm_wait and arguments[0] and not self.motion_degraded)
      if self.motion_waiting_for_fresh_input:
        self.long_control_state = replay.LongCtrlState.pid
        self.resolved_should_stop = False
        self.motion_phase = 1
      return result

  class AutomaticPlanner(Planner):
    def __init__(self, params, params_sp):
      supported = (params.carFingerprint == replay.CAR.HONDA_ACCORD and params.openpilotLongitudinalControl
                   and not params_sp.enableGasInterceptor)
      super().__init__(params, params_sp, unified=supported)

  class AutomaticLongitudinal(RearmLongitudinal):
    def __init__(self, params, params_sp):
      supported = (params.carFingerprint == replay.CAR.HONDA_ACCORD and params.openpilotLongitudinalControl
                   and not params_sp.enableGasInterceptor)
      super().__init__(params, params_sp, unified=supported)

    def update(self, *arguments, intent=None, solver_status=missing, braking_plan_valid=False, standby_input=False, **options):
      return super().update(*arguments, intent=intent, solver_status=solver_status, braking_plan_valid=braking_plan_valid,
                            standby_input=standby_input, **options)

  class Controller:
    def __init__(self, *arguments):
      self.frame = 0

    def update(self, control, control_sp, state, mono):
      encoded.append(control)
      return None, [(0x1df, b'\0' * 8, 1)]

  class Parser:
    def __init__(self, *arguments):
      self.vl = {'ACC_CONTROL': {'ACCEL_COMMAND': -1.0, 'BRAKE_REQUEST': 1, 'GAS_COMMAND': -30000,
                                 'CONTROL_ON': 5, 'STANDSTILL': 1}}

    def update(self, frames):
      pass

  monkeypatch.setattr(replay.planner_module, 'LongitudinalPlanner', AutomaticPlanner if automatic_api else Planner)
  monkeypatch.setattr(replay, 'LongControl', AutomaticLongitudinal if automatic_api else LegacyLongitudinal if legacy_api else RearmLongitudinal if rearm_api
                      else EnvelopeLongitudinal if envelope_api else Longitudinal)
  monkeypatch.setattr(replay, 'decode_motion_intent', lambda plan: plan.motionIntent)
  monkeypatch.setattr(replay, 'CarController', Controller)
  monkeypatch.setattr(replay, 'CANParser', Parser)
  monkeypatch.setattr(replay, 'CanBus', lambda params: SimpleNamespace(pt=1))
  monkeypatch.setattr(replay.CarInterface, 'get_pid_accel_limits', lambda *arguments: (-3.5, 2.0))
  if legacy_api:
    monkeypatch.setattr(replay, 'restrict_longitudinal_authority', None)
  car_state = Snapshot(vEgo=0.0, aEgo=0.0, vCruise=30.0, gasPressed=False, brakePressed=False, canValid=True,
                       standstill=True, cruiseState=Snapshot(standstill=True))
  car_control = Snapshot(enabled=True, longActive=True, cruiseControl=Snapshot(override=False, resume=True),
                         actuators=Snapshot(accel=0.1, longControlState=replay.LongCtrlState.pid))
  values = {
    'carState': car_state, 'carControl': car_control,
    'controlsState': Snapshot(longitudinalPlanMonoTime=0, forceDecel=False, motionState=Snapshot(version=0, enabled=False)),
    'vehicleParameters': Snapshot(), 'radarState': Snapshot(leadOne=Snapshot(present=False), leadTwo=Snapshot(present=False)),
    'modelV2': Snapshot(action=Snapshot(desiredAcceleration=0.15, shouldStop=False)),
    'selfdriveState': Snapshot(enabled=True),
  }

  def message(service, mono, value):
    return SimpleNamespace(which=lambda: service, logMonoTime=mono, valid=True, **{service: value})

  messages = [
    message('carParams', 100, Snapshot(carFingerprint=replay.CAR.HONDA_ACCORD, openpilotLongitudinalControl=True, stopAccel=-2.0)),
    message('carParamsSP', 200, Snapshot(flags=0, safetyParam=0, enableGasInterceptor=False)),
  ]
  messages += [message(service, 900_000_000 + index * 1_000_000, values[service]) for index, service in enumerate(replay.SERVICES)]
  messages += [
    message('modelV2', 1_000_000_000, values['modelV2']),
    message('longitudinalPlan', 1_010_000_000, Snapshot(modelMonoTime=1_000_000_000, aTarget=-999.0, shouldStop=True,
                                                    speeds=[0.0], solverStatus=0, motionIntent=Snapshot(version=0))),
    message('longitudinalPlanSP', 1_011_000_000, Snapshot(longitudinalPlanSource='cruise')),
    message('controlsState', 1_015_000_000, Snapshot(longitudinalPlanMonoTime=1_010_000_000, forceDecel=False,
                                                  motionState=Snapshot(version=0, enabled=False))),
    message('carControl', 1_020_000_000, car_control),
    message('carState', 1_025_000_000, car_state),
    message('sendcan', 1_040_000_000, [Snapshot(address=0x1df, dat=b'\0' * 8, src=1)]),
  ]
  window = {'origin': 0.0, 'start': 1.0, 'end': 1.05, 'warmup': 0.0, 'cache': 'synthetic-protocol-window'}
  return messages, window, controls, encoded, intent


def cadence_fixture(monkeypatch, *, prefix_active=(False, True), warmup=1.0):
  messages, window, controls, encoded, _ = replay_fixture(monkeypatch)
  window['warmup'] = warmup
  values = {message.which(): getattr(message, message.which()) for message in messages
            if message.which() in replay.SERVICES}
  dispatched = []

  class Controller:
    def __init__(self, *arguments):
      self.frame = 0

    def update(self, control, control_sp, state, mono):
      has_acc = self.frame % 2 == 0
      dispatched.append({'mono': int(mono), 'frame': self.frame, 'active': bool(control.longActive), 'has_acc': has_acc})
      self.frame += 1
      encoded.append(control)
      return None, [(0x1df if has_acc else 0xe4, b'\0' * 8, 1)]

  monkeypatch.setattr(replay, 'CarController', Controller)
  for cycle, active in enumerate(prefix_active):
    base = 950_000_000 + cycle * 10_000_000
    control = deepcopy(values['carControl'])
    control.enabled = control.longActive = active
    control.actuators.accel = 0.1 if active else 0.0
    control.actuators.longControlState = replay.LongCtrlState.pid if active else replay.LongCtrlState.off
    items = (
      ('controlsState', base, Snapshot(longitudinalPlanMonoTime=0, forceDecel=False,
                                       motionState=Snapshot(version=0, enabled=False))),
      ('carControl', base + 1_000_000, control),
      ('carState', base + 2_000_000, values['carState']),
      ('sendcan', base + 3_000_000, [Snapshot(address=0x1df if cycle % 2 == 0 else 0xe4, dat=b'\0' * 8, src=1)]),
    )
    for service, mono, value in items:
      messages.append(SimpleNamespace(which=lambda service=service: service, logMonoTime=mono,
                                      valid=True, **{service: value}))
  messages.sort(key=lambda message: message.logMonoTime)
  return messages, window, controls, dispatched


@pytest.mark.parametrize('unified', [False, True])
@pytest.mark.parametrize('prefix_active', [(False, True), (True, False), (False, False), (True, True)])
def test_pre_warmup_inputs_never_execute_a_partial_controller_history(monkeypatch, unified, prefix_active):
  messages, window, controls, dispatched = cadence_fixture(monkeypatch, prefix_active=prefix_active)
  result = replay.replay_window(None, window, messages=messages, unified=unified, strict=True)
  assert len(controls) == 1
  assert dispatched == [{'mono': 1_040_000_000, 'frame': 0, 'active': True, 'has_acc': True}]
  assert result['first_executed'] == {'plan_mono': 1_010_000_000, 'control_mono': 1_020_000_000, 'controller_mono': 1_040_000_000}
  assert result['execution_boundary_mono'] == 1_000_000_000
  assert result['execution_boundary_layers'] == ['planner', 'longcontrol', 'carcontroller']
  assert result['read_only_prefix']['controls'] == 3
  assert result['read_only_prefix']['sendcan'] == 2
  assert result['resets'] == []
  assert result['capabilities']['replay_protocol_version'] == replay.REPLAY_PROTOCOL_VERSION


@pytest.mark.parametrize('unified', [False, True])
def test_post_boundary_missing_active_cycle_is_not_rephased_in_strict_mode(monkeypatch, unified):
  messages, window, _, dispatched = cadence_fixture(monkeypatch, warmup=0.94)
  with pytest.raises(ValueError, match='Controller cadence mismatch'):
    replay.replay_window(None, window, messages=messages, unified=unified, strict=True)
  assert dispatched == [
    {'mono': 953_000_000, 'frame': 0, 'active': False, 'has_acc': True},
    {'mono': 1_040_000_000, 'frame': 1, 'active': True, 'has_acc': False},
  ]


def test_pre_warmup_control_consumes_its_pending_state_without_executing_it(monkeypatch):
  messages, window, controls, _ = cadence_fixture(monkeypatch)
  messages = [message for message in messages if not (
    message.which() == 'controlsState' and message.logMonoTime == 1_015_000_000)]
  with pytest.raises(ValueError, match='Incomplete replay'):
    replay.replay_window(None, window, messages=messages, unified=True, strict=True)
  assert controls == []


def test_pre_warmup_can_consumes_its_pair_without_reusing_it_after_boundary(monkeypatch):
  messages, window, _, dispatched = cadence_fixture(monkeypatch)
  messages = [message for message in messages if not (
    message.which() == 'carState' and message.logMonoTime == 1_025_000_000)]
  with pytest.raises(ValueError, match='Incomplete replay'):
    replay.replay_window(None, window, messages=messages, unified=True, strict=True)
  assert dispatched == []


def test_model_snapshot_read_before_boundary_remains_available_to_first_plan(monkeypatch):
  messages, window, controls, dispatched = cadence_fixture(monkeypatch, warmup=1.005)
  result = replay.replay_window(None, window, messages=messages, unified=True, strict=True)
  assert controls[-1][1]['model_time_ns'] == 1_000_000_000
  assert result['first_executed']['plan_mono'] == 1_010_000_000
  assert len(dispatched) == 1


@pytest.mark.parametrize('payload', sorted(replay.INITIALIZATION_UDS))
def test_pre_warmup_initialization_is_audited_without_an_executed_cycle(monkeypatch, payload):
  messages, window, controls, dispatched = cadence_fixture(monkeypatch)
  messages.append(Snapshot(which=lambda: 'sendcan', logMonoTime=970_000_000, valid=True,
                           sendcan=[Snapshot(address=0x18dab0f1, dat=payload, src=1)]))
  messages.sort(key=lambda message: message.logMonoTime)
  result = replay.replay_window(None, window, messages=messages, unified=True, strict=True)
  assert result['initialization_sends'] == [{'mono': 970_000_000, 'measured': False, 'payloads': [payload.hex()]}]
  assert len(controls) == len(dispatched) == 1
  assert result['read_only_prefix']['sendcan'] == 3


def test_candidate_replay_consumes_real_plan_model_and_control_timestamps(monkeypatch):
  messages, window, controls, encoded, intent = replay_fixture(monkeypatch)
  result = replay.replay_window(None, window, messages=messages, unified=True)
  arguments, options = controls[-1]
  assert options == {'intent': intent, 'plan_time_ns': 1_010_000_000, 'model_time_ns': 1_000_000_000,
                     'now_ns': 1_020_000_000, 'plan_valid': True, 'solver_status': 0}
  assert arguments[2] == float(np.float32(0.15))
  assert arguments[3] is False
  assert not encoded[-1].cruiseControl.resume
  row = result['control_rows'][0]
  assert not row['requested_stop']
  assert row['resolved_stop']
  assert row['motion_phase'] == 3
  assert row['motion_consumed']
  assert result['motion_replay']['accepted_intent_versions'] == {str(replay.MOTION_PROTOCOL_VERSION): 1}
  assert result['motion_replay']['generated_intent_versions'] == {str(replay.MOTION_PROTOCOL_VERSION): 1}
  assert result['coverage']['matched_sendcan'] == 1
  assert result['first_executed'] == {'plan_mono': 1_010_000_000, 'control_mono': 1_020_000_000, 'controller_mono': 1_040_000_000}
  original_controls = next(message.controlsState for message in messages if message.which() == 'controlsState')
  assert original_controls.motionState.version == 0
  assert not original_controls.motionState.enabled


def test_automatic_runtime_retains_the_full_motion_consumer_contract(monkeypatch):
  messages, window, controls, _, _ = replay_fixture(monkeypatch, automatic_api=True)
  result = replay.replay_window(None, window, messages=messages, unified=True)
  capabilities = result['capabilities']
  assert capabilities['requested_mode'] == capabilities['planner_mode'] == capabilities['controller_mode'] == 2
  for field in ('controller_solver_status', 'controller_mode_protocol', 'controller_planning_mode_protocol',
                'controller_readiness', 'controller_braking_health_protocol', 'controller_owner_protocol',
                'controller_waiting_protocol', 'controller_standby_input_protocol'):
    assert capabilities[field]
  assert result['motion_replay']['accepted_intent_versions'] == {str(replay.MOTION_PROTOCOL_VERSION): 1}
  assert result['motion_replay']['accepted_intent_modes'] == {'2': 1}
  assert controls[-1][1]['solver_status'] == 0
  assert controls[-1][1]['plan_valid']
  assert 'braking_plan_valid' in controls[-1][1]
  assert controls[-1][1]['standby_input'] is False


@pytest.mark.parametrize('supervised', [False, True])
def test_retired_requested_modes_cannot_be_relabelled_as_automatic_unified(monkeypatch, supervised):
  messages, window, controls, _, _ = replay_fixture(monkeypatch, automatic_api=True)
  with pytest.raises(ValueError, match='retired supervised' if supervised else 'Requested replay mode 0'):
    replay.replay_window(None, window, messages=messages, supervised=supervised)
  assert not controls


def test_automatic_constructor_does_not_bypass_the_actual_platform_gate(monkeypatch):
  messages, window, controls, _, _ = replay_fixture(monkeypatch, automatic_api=True)
  next(item.carParamsSP for item in messages if item.which() == 'carParamsSP').enableGasInterceptor = True
  with pytest.raises(ValueError, match='Requested replay mode 2.*runtime mode 0'):
    replay.replay_window(None, window, messages=messages, unified=True)
  assert not controls


@pytest.mark.parametrize('interceptor', [False, True])
def test_real_controller_identity_and_protocol_use_the_vehicle_gate(interceptor):
  from openpilot.common.prefix import OpenpilotPrefix
  from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_planner

  with OpenpilotPrefix():
    planner, extension = make_planner(interceptor=interceptor)
    controller = replay.LongControl(planner.CP, extension)
  expected_mode = 0 if interceptor else 2
  assert replay.require_component_mode(planner, expected_mode, 'planner') == expected_mode
  assert replay.require_component_mode(controller, expected_mode, 'controller') == expected_mode
  assert replay.controller_mode_protocols(controller) == (True, True)
  assert replay.solver_status_supported(type(controller), True)
  assert replay.component_options(type(controller), True) == {}
  assert type(controller.can_standby) is bool


@pytest.mark.parametrize('version', [2, 3])
def test_each_native_motion_generation_executes_with_its_semantic_version(monkeypatch, version):
  monkeypatch.setattr(replay, 'MOTION_PROTOCOL_VERSION', version)
  messages, window, controls, _, _ = replay_fixture(monkeypatch)
  result = replay.replay_window(None, window, messages=messages, supervised=True)
  assert result['capabilities']['motion_intent_version'] == version
  assert result['motion_replay']['accepted_intent_versions'] == {str(version): 1}
  assert not result['control_rows'][0]['mode_mismatch']
  assert controls[-1][1]['plan_valid']


def test_native_planner_receives_the_exact_recorded_clock_before_input_guards(monkeypatch):
  messages, window, _, _, _ = replay_fixture(monkeypatch)
  original = replay.planner_module.LongitudinalPlanner.update
  calls = []

  def update(self, state, *, now_ns=None):
    calls.append((now_ns, state.logMonoTime['carState']))
    return original(self, state)

  monkeypatch.setattr(replay.planner_module.LongitudinalPlanner, 'update', update)
  result = replay.replay_window(None, window, messages=messages, supervised=True)
  assert calls == [(1_010_000_000, 900_000_000)]
  assert result['capabilities']['planner_clock'] == 'recorded_plan_publication_mono_proxy_not_recorded_update_entry_time'


def test_candidate_arguments_cannot_be_silently_ignored(monkeypatch):
  messages, window, _, _, _ = replay_fixture(monkeypatch, consume=False)
  with pytest.raises(ValueError, match='did not consume'):
    replay.replay_window(None, window, messages=messages, unified=True)


@pytest.mark.parametrize('mode,automatic_api', [(1, False), (2, False), (2, True)])
def test_healthy_rearm_wait_is_constrained_unconsumed_and_cannot_resume(monkeypatch, mode, automatic_api):
  messages, window, controls, encoded, _ = replay_fixture(monkeypatch, rearm_api=True, rearm_wait=True, automatic_api=automatic_api)
  result = replay.replay_window(None, window, messages=messages, supervised=mode == 1, unified=mode == 2)
  row = result['control_rows'][0]
  assert row['motion_waiting_for_fresh_input']
  assert row['motion_input_constrained']
  assert row['motion_waiting_eligible']
  assert not row['motion_degraded']
  assert not row['motion_invalid_reason']
  assert not row['motion_consumed']
  assert row['accel'] <= row['motion_waiting_brake_ceiling'] == -1.0
  assert not row['controller_resume']
  assert not encoded[-1].cruiseControl.resume
  assert result['motion_replay']['waiting_controls'] == 1
  assert result['motion_replay']['degraded_controls'] == result['motion_replay']['accepted_controls'] == 0
  assert controls[-1][1]['standby_input'] is False


@pytest.mark.skipif(not hasattr(replay.LongControl, 'motion_waiting_for_fresh_input'), reason='tested legacy runtime has no healthy-wait API')
@pytest.mark.parametrize('trigger', ['enable_between_plans', 'gas_release_between_plans', 'gas_release_after_active_plan'])
def test_precheck_and_consumption_match_the_actual_healthy_rearm_consumer(monkeypatch, trigger):
  from openpilot.common.prefix import OpenpilotPrefix
  from openpilot.selfdrive.controls.tests.test_motion_rearm_wait import enter_off

  with OpenpilotPrefix():
    harness = enter_off(LongitudinalMode.UNIFIED, trigger=trigger)
    controller = harness.control.LoC
    original = controller.update
    receipts = []

    def observed(active, state, target, stop, limits, **options):
      before = vars(controller.motion_supervisor).copy()
      evidence = replay.waiting_input_evidence(controller, active, state, target, limits, options)
      assert vars(controller.motion_supervisor) == before
      output = original(active, state, target, stop, limits, **options)
      diagnostics = replay.motion_diagnostics(controller, options['intent'], target, options['plan_time_ns'],
                                              options['model_time_ns'], active, int(LongitudinalMode.UNIFIED))
      receipts.append((evidence, diagnostics, float(np.float32(output))))
      return output

    monkeypatch.setattr(controller, 'update', observed)
    _, motion = harness.step(new_plan=False)
    evidence, diagnostics, output = receipts[-1]
    assert evidence['motion_waiting_eligible']
    assert output <= evidence['motion_waiting_brake_ceiling']
    assert diagnostics['motion_waiting_for_fresh_input'] and motion.waitingForFreshInput
    assert diagnostics['motion_input_constrained']
    assert not diagnostics['motion_degraded'] and not diagnostics['motion_consumed']
    old_model = int(motion.modelMonoTime)
    command, motion = harness.step(model_time_ns=old_model)
    evidence, diagnostics, output = receipts[-1]
    assert evidence['motion_waiting_eligible']
    assert diagnostics['motion_waiting_for_fresh_input'] and not diagnostics['motion_consumed']
    assert not diagnostics['motion_degraded']
    assert output <= evidence['motion_waiting_brake_ceiling']
    assert not command.cruiseControl.resume
    _, motion = harness.step()
    evidence, diagnostics, _ = receipts[-1]
    assert not evidence['motion_waiting_eligible']
    assert not motion.waitingForFreshInput and not motion.degraded
    assert diagnostics['motion_consumed']


@pytest.mark.parametrize('failure', [None, 'can', 'speed', 'setpoint', 'shape', 'version'])
def test_only_fully_confirmed_cold_standby_is_forwarded(monkeypatch, failure):
  messages, window, controls, _, intent = replay_fixture(
    monkeypatch, rearm_api=True, standby_eligible=True, publish_standby=True)
  intent.valid = intent.ready = intent.allow_creep = False
  if failure == 'shape':
    intent.release_allowed = True
  if failure == 'version':
    intent.version = 0
  for message in messages:
    if message.which() == 'carState':
      message.carState.vCruise = 30.0 if failure == 'setpoint' else replay.V_CRUISE_UNSET
      message.carState.canValid = failure != 'can'
      message.carState.vEgo = float('nan') if failure == 'speed' else 0.0
  result = replay.replay_window(None, window, messages=messages, supervised=True)
  assert controls[-1][1]['standby_input'] is (failure is None)
  assert result['control_rows'][0]['controller_standby_input'] is (failure is None)


def test_old_consumer_has_no_invented_standby_keyword_or_waiting_state(monkeypatch):
  messages, window, controls, _, _ = replay_fixture(monkeypatch)
  result = replay.replay_window(None, window, messages=messages, supervised=True)
  assert 'standby_input' not in controls[-1][1]
  assert not result['capabilities']['controller_waiting_protocol']
  assert not result['capabilities']['controller_standby_input_protocol']
  assert not result['control_rows'][0]['motion_waiting_for_fresh_input']


@pytest.mark.parametrize('degraded,waiting,constrained', [(True, True, True), (False, True, False), (True, False, False), (False, False, True)])
def test_contradictory_wait_and_fault_properties_fail_closed(degraded, waiting, constrained):
  controller = SimpleNamespace(motion_degraded=degraded, motion_waiting_for_fresh_input=waiting, motion_input_constrained=constrained)
  with pytest.raises(ValueError, match='contradict'):
    replay.motion_constraints(controller)


@pytest.mark.parametrize('failure', [None, 'fault_barrier', 'clock_repeat', 'clock_gap', 'invalid_plan', 'solver',
                                     'nonfinite_target', 'nonfinite_previous', 'bounds', 'new_both', 'new_plan_only'])
@pytest.mark.parametrize('mode', [1, 2])
def test_waiting_evidence_checks_raw_health_and_rearm_before_update_without_mutation(monkeypatch, failure, mode):
  messages, _, _, _, intent = replay_fixture(monkeypatch, rearm_api=True, rearm_wait=True)
  params = next(message.carParams for message in messages if message.which() == 'carParams')
  params_sp = next(message.carParamsSP for message in messages if message.which() == 'carParamsSP')
  state = next(message.carState for message in messages if message.which() == 'carState')
  controller = replay.LongControl(params, params_sp, supervised=mode == 1, unified=mode == 2)
  supervisor = controller.motion_supervisor
  options = {'now_ns': 1_020_000_000, 'plan_time_ns': 1_010_000_000, 'model_time_ns': 1_000_000_000,
             'intent': intent, 'plan_valid': True, 'solver_status': 0}
  target, limits = -.6, (-3.5, 2.0)
  if failure == 'fault_barrier':
    supervisor._rearm_after_fault = True
  elif failure == 'clock_repeat':
    supervisor._last_creep_observation_ns = options['now_ns']
  elif failure == 'clock_gap':
    supervisor._last_now_ns = 800_000_000
  elif failure == 'invalid_plan':
    options['plan_valid'] = False
  elif failure == 'solver':
    options['solver_status'] = 1
  elif failure == 'nonfinite_target':
    target = float('nan')
  elif failure == 'nonfinite_previous':
    controller.last_output_accel = float('nan')
  elif failure == 'bounds':
    limits = (-.1, 2.0)
  elif failure == 'new_both':
    options.update(plan_time_ns=1_015_000_000, model_time_ns=1_005_000_000)
  elif failure == 'new_plan_only':
    options['plan_time_ns'] = 1_015_000_000
  snapshot = vars(supervisor).copy()
  result = replay.waiting_input_evidence(controller, True, state, target, limits, options)
  assert result['motion_waiting_eligible'] is (failure in (None, 'new_plan_only'))
  assert vars(supervisor) == snapshot


@pytest.mark.parametrize('failure', [None, 'not_rearming', 'fault_barrier', 'clock_repeat', 'clock_backwards', 'clock_gap',
                                    'invalid_plan', 'solver', 'nonfinite_target', 'future_plan', 'future_model',
                                    'expired_plan', 'expired_model', 'plan_backwards', 'rewritten_plan', 'mode_mismatch',
                                    'version', 'unready', 'invalid_intent', 'inactive', 'gas', 'brake',
                                    'model_only', 'new_plan_only', 'new_both'])
def test_actual_precheck_cannot_trust_a_claimed_waiting_flag_over_input_health(failure):
  from openpilot.common.prefix import OpenpilotPrefix
  from openpilot.selfdrive.controls.tests.test_motion_rearm_wait import enter_off

  with OpenpilotPrefix():
    harness = enter_off(LongitudinalMode.UNIFIED)
    controller, state = harness.control.LoC, harness.car_state
    owner = controller.motion_supervisor
    owner.waiting_for_fresh_input = True
    target, limits = harness.plan.aTarget, (-3.5, 2.0)
    options = {'now_ns': harness.now_ns + 10_000_000,
               'plan_time_ns': int(harness.control.sm.logMonoTime['longitudinalPlan']),
               'model_time_ns': int(harness.plan.modelMonoTime),
               'intent': replay.decode_motion_intent(harness.plan), 'plan_valid': True, 'solver_status': 0}
    active = True
    if failure == 'not_rearming':
      owner._require_new_input = False
    elif failure == 'fault_barrier':
      owner._rearm_after_fault = True
    elif failure in ('clock_repeat', 'clock_backwards'):
      options['now_ns'] = owner._last_creep_observation_ns - int(failure == 'clock_backwards')
    elif failure == 'clock_gap':
      owner._last_now_ns = options['now_ns'] - replay.motion_runtime.MAX_CONTROL_GAP_NS - 1
    elif failure == 'invalid_plan':
      options['plan_valid'] = False
    elif failure == 'solver':
      options['solver_status'] = 1
    elif failure == 'nonfinite_target':
      target = float('nan')
    elif failure in ('future_plan', 'future_model'):
      options['plan_time_ns' if failure == 'future_plan' else 'model_time_ns'] = options['now_ns'] + 1
    elif failure == 'expired_plan':
      options['now_ns'] = options['plan_time_ns'] + replay.motion_runtime.PLAN_LEASE_NS + 1
    elif failure == 'expired_model':
      options['model_time_ns'] = options['now_ns'] - replay.motion_runtime.MODEL_LEASE_NS - 1
    elif failure == 'plan_backwards':
      options['plan_time_ns'] -= 1
    elif failure == 'rewritten_plan':
      options['intent'] = replace(options['intent'], release_allowed=not options['intent'].release_allowed)
    elif failure == 'mode_mismatch':
      options['intent'] = replace(options['intent'], mode=int(LongitudinalMode.SUPERVISED))
    elif failure == 'version':
      options['intent'] = replace(options['intent'], version=replay.MOTION_PROTOCOL_VERSION + 1)
    elif failure == 'unready':
      options['intent'] = replace(options['intent'], ready=False)
    elif failure == 'invalid_intent':
      options['intent'] = replace(options['intent'], valid=False)
    elif failure == 'inactive':
      active = False
    elif failure in ('gas', 'brake'):
      setattr(state, failure + 'Pressed', True)
    elif failure == 'model_only':
      options['model_time_ns'] += 1
    elif failure == 'new_plan_only':
      options['plan_time_ns'] += 5_000_000
    elif failure == 'new_both':
      options['plan_time_ns'] = options['model_time_ns'] = options['now_ns']
    before = vars(owner).copy()
    result = replay.waiting_input_evidence(controller, active, state, target, limits, options)
    assert result['motion_waiting_eligible'] is (failure in (None, 'new_plan_only'))
    assert vars(owner) == before


@pytest.mark.parametrize('mode', [0, -1, 3, True, '2'])
def test_waiting_evidence_rejects_legacy_or_unknown_runtime_modes(monkeypatch, mode):
  messages, _, _, _, intent = replay_fixture(monkeypatch, rearm_api=True, rearm_wait=True)
  params = next(message.carParams for message in messages if message.which() == 'carParams')
  params_sp = next(message.carParamsSP for message in messages if message.which() == 'carParamsSP')
  state = next(message.carState for message in messages if message.which() == 'carState')
  controller = replay.LongControl(params, params_sp, unified=True)
  controller.longitudinal_mode = mode
  options = {'now_ns': 1_020_000_000, 'plan_time_ns': 1_010_000_000, 'model_time_ns': 1_000_000_000,
             'intent': intent, 'plan_valid': True, 'solver_status': 0}
  if mode == 0:
    assert not replay.waiting_input_evidence(controller, True, state, -.6, (-3.5, 2.0), options)['motion_waiting_eligible']
  else:
    with pytest.raises(ValueError, match='invalid longitudinal mode'):
      replay.waiting_input_evidence(controller, True, state, -.6, (-3.5, 2.0), options)


def test_default_replay_preserves_the_legacy_five_argument_call(monkeypatch):
  messages, window, controls, encoded, _ = replay_fixture(monkeypatch, legacy_api=True, publish_solver_status=False)
  result = replay.replay_window(None, window, messages=messages)
  assert len(controls[-1][0]) == 5
  assert controls[-1][1] == {}
  assert encoded[-1].cruiseControl.resume
  assert result['control_rows'][0]['intent_version'] == 0
  assert not result['motion_replay']['enabled']
  assert not result['capabilities']['controller_solver_status']
  assert not result['capabilities']['shared_authority_helper']
  assert result['plan_rows'][0]['published_solver_status'] == -1
  assert not result['can_rows'][0]['authority_helper_applied']


@pytest.mark.parametrize('pedal', ['gasPressed', 'brakePressed'])
def test_manual_pedal_clears_candidate_control_authority_before_can(monkeypatch, pedal):
  messages, window, controls, encoded, _ = replay_fixture(monkeypatch)
  for message in messages:
    if message.which() == 'carState':
      setattr(message.carState, pedal, True)
  result = replay.replay_window(None, window, messages=messages, unified=True)
  assert not controls[-1][0][0]
  assert not encoded[-1].longActive
  assert encoded[-1].actuators.accel == 0.0
  assert result['motion_replay']['disabled_controls'] == 1
  assert result['motion_replay']['accepted_controls'] == 0
  assert not encoded[-1].cruiseControl.resume


def test_solver_status_interface_is_required_for_candidate():
  class SilentOldInterface:
    def update(self, *arguments, **options):
      pass

  assert not replay.solver_status_supported(SilentOldInterface, False)
  with pytest.raises(ValueError, match='explicitly accept solver_status'):
    replay.solver_status_supported(SilentOldInterface, True)


@pytest.mark.parametrize('unified', [False, True])
def test_actual_published_solver_failure_reaches_supported_controller(monkeypatch, unified):
  messages, window, controls, encoded, _ = replay_fixture(monkeypatch, solver_status=4)
  result = replay.replay_window(None, window, messages=messages, unified=unified)
  assert controls[-1][1]['solver_status'] == 4
  assert not result['plan_rows'][0]['published_valid']
  assert result['plan_rows'][0]['published_solver_status'] == 4
  assert result['control_rows'][0]['controller_solver_status'] == 4
  assert not encoded[-1].cruiseControl.resume
  if unified:
    assert not controls[-1][1]['plan_valid']
    assert result['control_rows'][0]['motion_degraded']
    assert not result['control_rows'][0]['motion_consumed']


def test_old_recorded_validity_does_not_invalidate_new_publication(monkeypatch):
  messages, window, controls, _, _ = replay_fixture(monkeypatch)
  next(message for message in messages if message.which() == 'longitudinalPlan').valid = False
  result = replay.replay_window(None, window, messages=messages, unified=True)
  assert not result['plan_rows'][0]['valid']
  assert result['plan_rows'][0]['published_valid']
  assert controls[-1][1]['plan_valid']


def test_invalid_intent_changes_actual_publisher_validity(monkeypatch):
  messages, window, controls, encoded, intent = replay_fixture(monkeypatch)
  intent.valid = False
  result = replay.replay_window(None, window, messages=messages, unified=True)
  assert result['plan_rows'][0]['valid']
  assert not result['plan_rows'][0]['published_valid']
  assert not controls[-1][1]['plan_valid']
  assert not encoded[-1].cruiseControl.resume


def test_current_invalid_car_state_is_not_replaced_by_earlier_valid_planner_input(monkeypatch):
  messages, window, controls, _, _ = replay_fixture(monkeypatch)
  car_state = next(message.carState for message in messages if message.which() == 'carState')
  position = next(index for index, message in enumerate(messages)
                  if message.which() == 'controlsState' and message.logMonoTime > 1_000_000_000)
  messages.insert(position, Snapshot(which=lambda: 'carState', logMonoTime=1_014_000_000, valid=False, carState=car_state))
  result = replay.replay_window(None, window, messages=messages, unified=True)
  assert result['plan_rows'][0]['published_valid']
  assert not controls[-1][1]['plan_valid']
  assert not result['control_rows'][0]['controller_plan_valid']
  assert result['control_rows'][0]['motion_degraded']


def test_missing_published_candidate_solver_status_is_not_default_success(monkeypatch):
  messages, window, _, _, _ = replay_fixture(monkeypatch, publish_solver_status=False)
  with pytest.raises(ValueError, match='actual solver status'):
    replay.replay_window(None, window, messages=messages, unified=True)


@pytest.mark.parametrize('recorded_status', [0, 4])
def test_recorded_plan_uses_its_own_status_not_recomputed_mpc_status(monkeypatch, recorded_status):
  messages, window, controls, encoded, _ = replay_fixture(monkeypatch, solver_status=9)
  recorded = next(message for message in messages if message.which() == 'longitudinalPlan').longitudinalPlan
  recorded.solverStatus = recorded_status
  result = replay.replay_window(None, window, messages=messages, recorded_plan=True)
  assert controls[-1][0][2] == -999.0
  assert controls[-1][1]['solver_status'] == recorded_status
  assert result['plan_rows'][0]['solver_status'] == 9
  assert result['control_rows'][0]['controller_solver_status'] == recorded_status
  assert not encoded[-1].cruiseControl.resume


def test_old_recorded_plan_missing_status_defaults_only_its_consumer_input(monkeypatch):
  messages, window, controls, _, _ = replay_fixture(monkeypatch, solver_status=9)
  recorded = next(message for message in messages if message.which() == 'longitudinalPlan').longitudinalPlan
  del recorded.solverStatus
  replay.replay_window(None, window, messages=messages, recorded_plan=True)
  assert controls[-1][1]['solver_status'] == 0


@pytest.mark.parametrize('automatic_api', [False, True])
def test_candidate_mode_mismatch_disables_control_and_resume(monkeypatch, automatic_api):
  messages, window, controls, encoded, intent = replay_fixture(monkeypatch, automatic_api=automatic_api)
  intent.version = 0
  result = replay.replay_window(None, window, messages=messages, unified=True)
  assert not controls[-1][0][0]
  assert result['control_rows'][0]['mode_mismatch']
  assert not encoded[-1].longActive
  assert not encoded[-1].cruiseControl.resume


@pytest.mark.parametrize('automatic_api', [False, True])
def test_cold_standby_keeps_enabled_for_initialization_without_longitudinal_actuation(monkeypatch, automatic_api):
  messages, window, controls, encoded, intent = replay_fixture(
    monkeypatch, standby_eligible=True, publish_standby=True, automatic_api=automatic_api)
  intent.ready = intent.valid = False
  for message in messages:
    if message.which() == 'carState':
      message.carState.vCruise = replay.V_CRUISE_UNSET
  result = replay.replay_window(None, window, messages=messages, unified=True)
  assert result['plan_rows'][0]['published_valid']
  assert not result['plan_rows'][0]['published_ready']
  assert not controls[-1][0][0]
  assert encoded[-1].enabled
  assert not encoded[-1].longActive
  assert encoded[-1].actuators.accel == 0.0
  assert not encoded[-1].cruiseControl.resume
  row = result['control_rows'][0]
  assert row['motion_standby'] and row['motion_can_standby']
  assert not row['intent_ready'] and not row['intent_valid']
  assert not row['controller_cruise_set']
  assert not row['motion_consumed'] and not row['motion_degraded']
  assert result['motion_replay']['standby_controls'] == 1


def test_ready_false_during_existing_hold_degrades_without_switching_off(monkeypatch):
  messages, window, controls, encoded, intent = replay_fixture(monkeypatch, publish_standby=True)
  intent.ready = intent.valid = False
  result = replay.replay_window(None, window, messages=messages, unified=True)
  assert controls[-1][0][0]
  assert not result['control_rows'][0]['motion_can_standby']
  assert not result['control_rows'][0]['motion_standby']
  assert result['control_rows'][0]['motion_degraded']
  assert encoded[-1].longActive
  assert encoded[-1].actuators.longControlState == replay.LongCtrlState.stopping
  assert encoded[-1].actuators.accel < 0.0
  assert not encoded[-1].cruiseControl.resume


@pytest.mark.parametrize('automatic_api', [False, True])
@pytest.mark.parametrize('cruise_speed', [replay.V_CRUISE_UNSET, float('nan'), float('inf')])
def test_latest_unset_cruise_invalidates_a_ready_plan_without_cold_standby(monkeypatch, cruise_speed, automatic_api):
  messages, window, controls, encoded, _ = replay_fixture(monkeypatch, automatic_api=automatic_api)
  current_state = deepcopy(next(message.carState for message in messages if message.which() == 'carState'))
  current_state.vCruise = cruise_speed
  position = next(index for index, message in enumerate(messages)
                  if message.which() == 'controlsState' and message.logMonoTime > 1_000_000_000)
  messages.insert(position, Snapshot(which=lambda: 'carState', logMonoTime=1_014_000_000, valid=True, carState=current_state))
  result = replay.replay_window(None, window, messages=messages, unified=True)
  assert result['plan_rows'][0]['published_valid']
  assert result['plan_rows'][0]['published_ready']
  assert controls[-1][0][0]
  assert not controls[-1][1]['plan_valid']
  assert not result['control_rows'][0]['controller_cruise_set']
  assert not result['control_rows'][0]['motion_standby']
  assert result['control_rows'][0]['motion_degraded']
  assert encoded[-1].actuators.accel < 0.0


@pytest.mark.parametrize('failure', ['publisher', 'car_state', 'solver'])
def test_unhealthy_unready_plan_does_not_use_the_standby_exception(monkeypatch, failure):
  messages, window, controls, _, intent = replay_fixture(
    monkeypatch, standby_eligible=True, publish_standby=failure != 'publisher', solver_status=4 if failure == 'solver' else 0)
  intent.ready = intent.valid = False
  if failure == 'car_state':
    car_state = next(message.carState for message in messages if message.which() == 'carState')
    position = next(index for index, message in enumerate(messages)
                    if message.which() == 'controlsState' and message.logMonoTime > 1_000_000_000)
    messages.insert(position, Snapshot(which=lambda: 'carState', logMonoTime=1_014_000_000, valid=False, carState=car_state))
  result = replay.replay_window(None, window, messages=messages, unified=True)
  assert controls[-1][0][0]
  assert not result['control_rows'][0]['motion_standby']
  assert result['control_rows'][0]['motion_degraded']


@pytest.mark.parametrize('automatic_api', [False, True])
def test_missing_published_readiness_cannot_be_silently_defaulted_for_new_controller(monkeypatch, automatic_api):
  messages, window, _, _, intent = replay_fixture(monkeypatch, automatic_api=automatic_api)
  del intent.ready
  with pytest.raises(ValueError, match='published motion readiness'):
    replay.replay_window(None, window, messages=messages, unified=True)


def test_legacy_interface_does_not_receive_new_cruise_readiness_protection(monkeypatch):
  messages, window, controls, encoded, _ = replay_fixture(monkeypatch, legacy_api=True, publish_solver_status=False)
  for message in messages:
    if message.which() == 'carState':
      message.carState.vCruise = replay.V_CRUISE_UNSET
  result = replay.replay_window(None, window, messages=messages)
  assert not result['capabilities']['controller_readiness']
  assert controls[-1][0][0]
  assert controls[-1][1] == {}
  assert encoded[-1].longActive
  assert not result['control_rows'][0]['motion_standby']
  assert result['control_rows'][0]['intent_ready'] == -1


def test_consumer_hash_records_readiness_and_intent_validity_independently(monkeypatch):
  hashes = set()
  for ready, valid in ((True, True), (False, True), (True, False)):
    messages, window, _, _, intent = replay_fixture(monkeypatch, publish_standby=True)
    intent.ready, intent.valid = ready, valid
    for message in messages:
      if message.which() == 'carControl':
        message.carControl.longActive = False
        message.carControl.actuators.accel = 0.0
        message.carControl.actuators.longControlState = replay.LongCtrlState.off
    result = replay.replay_window(None, window, messages=messages, unified=True)
    assert result['control_rows'][0]['accel'] == 0.0
    hashes.add(result['motion_replay']['consumer_input_sha256'])
  assert len(hashes) == 3


@pytest.mark.parametrize('unified', [False, True])
@pytest.mark.parametrize('warmup', [0.0, 1.0])
def test_disabled_missing_plan_runs_actual_off_branch_without_fabricated_targets(monkeypatch, unified, warmup):
  messages, window, controls, encoded, _ = replay_fixture(monkeypatch)
  window['warmup'] = warmup
  for message in messages:
    if message.which() == 'controlsState' and message.logMonoTime > 1_000_000_000:
      message.controlsState.longitudinalPlanMonoTime = 0
    if message.which() == 'carControl':
      message.carControl.enabled = False
      message.carControl.longActive = False
      message.carControl.actuators.longControlState = replay.LongCtrlState.off
      message.carControl.actuators.accel = 0.0
  result = replay.replay_window(None, window, messages=messages, unified=unified)
  arguments, options = controls[-1]
  assert arguments[:1] == (False,)
  assert arguments[2:4] == (None, None)
  assert 'solver_status' not in options
  assert encoded[-1].actuators.accel == 0.0
  assert not encoded[-1].cruiseControl.resume
  row = result['control_rows'][0]
  assert row['inactive_no_plan']
  assert row['requested_stop'] == row['controller_plan_valid'] == row['controller_solver_status'] == -1
  assert row['intent_version'] == row['model_mono'] == 0
  assert not row['motion_consumed']
  if unified:
    assert options['intent'] is options['plan_time_ns'] is options['model_time_ns'] is None
    assert options['plan_valid'] is False
    assert result['motion_replay']['inactive_no_plan_controls'] == 1
    assert result['motion_replay']['control_calls'] == 1
  assert result['coverage']['controls'] == result['coverage']['matched_controls'] == 1


@pytest.mark.parametrize('active,accel,state', [
  (True, 0.0, 'off'), (False, 1.0, 'off'), (False, 0.0, 'pid'),
])
def test_missing_plan_cannot_be_hidden_as_disabled_bootstrap(monkeypatch, active, accel, state):
  messages, window, controls, _, _ = replay_fixture(monkeypatch)
  for message in messages:
    if message.which() == 'controlsState' and message.logMonoTime > 1_000_000_000:
      message.controlsState.longitudinalPlanMonoTime = 0
    if message.which() == 'carControl':
      message.carControl.longActive = active
      message.carControl.actuators.accel = accel
      message.carControl.actuators.longControlState = getattr(replay.LongCtrlState, state)
  with pytest.raises(ValueError, match='Incomplete replay'):
    replay.replay_window(None, window, messages=messages, unified=True)
  assert not controls


def test_exact_provider_can_resolve_inputs_without_complete_model_publication_snapshot(monkeypatch):
  messages, window, _, _, _ = replay_fixture(monkeypatch)
  messages = [message for message in messages if message.which() != 'vehicleParameters']
  inputs = {message.which(): getattr(message, message.which()) for message in messages
            if message.which() in replay.SERVICES and message.logMonoTime <= 1_000_000_000}
  stamps = {message.which(): message.logMonoTime for message in messages
            if message.which() in replay.SERVICES and message.logMonoTime <= 1_000_000_000}
  inputs['vehicleParameters'] = Snapshot()
  stamps['vehicleParameters'] = 1_005_000_000
  requested = []

  def provider(plan_mono, state):
    requested.append(plan_mono)
    assert state == {}
    state.update(inputs)
    state.logMonoTime = stamps
    state.valid = dict.fromkeys(replay.SERVICES, True)
    return state

  result = replay.replay_window(None, window, messages=messages, unified=True, planner_inputs=provider)
  assert requested == [1_010_000_000]
  assert result['coverage']['matched_plans'] == 1
  assert result['capabilities']['planner_inputs'] == 'exact_provider'


def test_missing_first_model_snapshot_without_exact_trace_remains_unresolved(monkeypatch):
  messages, window, _, _, _ = replay_fixture(monkeypatch)
  messages = [message for message in messages if message.which() != 'vehicleParameters']
  with pytest.raises(ValueError, match='Incomplete replay'):
    replay.replay_window(None, window, messages=messages, unified=True)


def test_zero_timestamp_logger_seed_cannot_complete_an_untraced_input_snapshot(monkeypatch):
  messages, window, _, _, _ = replay_fixture(monkeypatch)
  messages = [message for message in messages if message.which() != 'vehicleParameters']
  messages.insert(0, Snapshot(which=lambda: 'vehicleParameters', logMonoTime=0, valid=False, vehicleParameters=Snapshot()))
  with pytest.raises(ValueError, match='Incomplete replay'):
    replay.replay_window(None, window, messages=messages, unified=True)


@pytest.mark.parametrize('payload', sorted(replay.INITIALIZATION_UDS))
def test_initialization_only_send_does_not_consume_controller_cycle(monkeypatch, payload):
  messages, window, _, encoded, _ = replay_fixture(monkeypatch)
  messages.insert(-1, Snapshot(which=lambda: 'sendcan', logMonoTime=1_030_000_000, valid=True,
                               sendcan=[Snapshot(address=0x18dab0f1, dat=payload, src=1)]))
  result = replay.replay_window(None, window, messages=messages, unified=True)
  assert len(encoded) == 1
  assert result['coverage']['sendcan'] == result['coverage']['matched_sendcan'] == 1
  assert result['initialization_sends'] == [{'mono': 1_030_000_000, 'measured': True, 'payloads': [payload.hex()]}]


@pytest.mark.parametrize('raw_can', [
  [],
  [(0x18dab0f1, bytes.fromhex('023e800000000000'), 1)],
  [(0x18dab0f1, bytes.fromhex('0210030000000000'), 0)],
  [(0xe4, b'\0' * 8, 1)],
  [(0x18dab0f1, bytes.fromhex('0210030000000000'), 1), (0x1df, b'\0' * 8, 1)],
])
def test_real_or_unknown_can_is_never_filtered_by_diagnostic_address(raw_can):
  assert not replay.initialization_only(raw_can)


def test_mixed_acc_and_initialization_address_still_executes_whole_controller_cycle(monkeypatch):
  messages, window, _, encoded, _ = replay_fixture(monkeypatch)
  messages[-1].sendcan.append(Snapshot(address=0x18dab0f1, dat=bytes.fromhex('0210030000000000'), src=1))
  result = replay.replay_window(None, window, messages=messages, unified=True)
  assert result['initialization_sends'] == []
  assert len(encoded) == result['coverage']['matched_sendcan'] == 1


@pytest.mark.parametrize('pedal', ['gasPressed', 'brakePressed'])
@pytest.mark.parametrize('unified', [False, True])
def test_latest_pedal_edge_restricts_cached_control_before_encoding(monkeypatch, pedal, unified):
  messages, window, controls, encoded, _ = replay_fixture(monkeypatch)
  assert replay.restrict_longitudinal_authority is not None
  latest_state = next(message for message in messages if message.which() == 'carState' and message.logMonoTime > 1_000_000_000)
  latest_state.carState = deepcopy(latest_state.carState)
  setattr(latest_state.carState, pedal, True)
  result = replay.replay_window(None, window, messages=messages, unified=unified)
  assert controls[-1][0][0]
  assert not getattr(controls[-1][0][1], pedal)
  assert encoded[-1].enabled
  assert not encoded[-1].longActive
  assert encoded[-1].actuators.accel == 0.0
  assert encoded[-1].actuators.longControlState == replay.LongCtrlState.off
  assert not encoded[-1].cruiseControl.resume
  assert result['can_rows'][0]['authority_helper_applied']
  assert result['can_rows'][0]['authority_restricted']
  assert result['can_rows'][0]['driver_override']


def test_baseline_without_helper_keeps_its_cached_request_visible(monkeypatch):
  messages, window, controls, encoded, _ = replay_fixture(monkeypatch, legacy_api=True, publish_solver_status=False)
  latest_state = next(message for message in messages if message.which() == 'carState' and message.logMonoTime > 1_000_000_000)
  latest_state.carState = deepcopy(latest_state.carState)
  latest_state.carState.brakePressed = True
  result = replay.replay_window(None, window, messages=messages)
  assert controls[-1][0][0]
  assert encoded[-1].longActive
  assert encoded[-1].actuators.accel == -1.0
  assert not result['can_rows'][0]['authority_helper_applied']
  assert not result['can_rows'][0]['authority_restricted']


@pytest.mark.parametrize('unified', [False, True])
@pytest.mark.parametrize('solver_status', [0, 4])
@pytest.mark.parametrize('input_valid,intent_valid', [(True, True), (False, True), (True, False)])
def test_actual_publisher_contract_and_scheduler_identity(monkeypatch, unified, solver_status, input_valid, intent_valid):
  from openpilot.common.prefix import OpenpilotPrefix
  from openpilot.selfdrive.controls.lib.motion_supervisor import MotionIntent
  from openpilot.selfdrive.controls.tests.longitudinal_planner_test_support import make_planner

  with OpenpilotPrefix():
    planner, _ = make_planner(interceptor=not unified)
  assert planner.unified is unified
  assert replay.component_mode(planner) == (2 if unified else 0)
  if planner.unified:
    assert planner.reference_timing.qualify_source(1_000_000_000, 0.0, 0.0, 1_020_000_000) == 1_000_000_000
    planner.reference_timing.set_initial_state(0.0, 0.0)
    planner.motion_intent = MotionIntent(valid=intent_valid)
  planner.mpc = SimpleNamespace(solution_status=solver_status, source='cruise', solve_time=0.0)
  planner.trace = None
  planner.observation = None
  planner.observation_result = None
  planner.v_desired_trajectory = np.zeros(17)
  planner.a_desired_trajectory = np.zeros(17)
  planner.j_desired_trajectory = np.zeros(17)
  planner.output_a_target = -0.123456789
  planner.output_should_stop = False
  planner.allow_throttle = True
  planner.fcw = False
  published_parents = []

  def publish_sp(state, publisher, parent_plan_mono_time=0, *, legacy_acceleration=None):
    published_parents.append((parent_plan_mono_time, legacy_acceleration))

  monkeypatch.setattr(planner, 'publish_longitudinal_plan_sp', publish_sp)
  state = replay.ReplayState({'radarState': Snapshot(leadOne=Snapshot(present=False))})
  state.valid = {'radarState': input_valid}
  state.logMonoTime = {'modelV2': 1_000_000_000}
  publication = replay.publish_plan(planner, state, 1_020_000_000)
  assert publication.logMonoTime == 1_020_000_000
  assert published_parents == [(publication.logMonoTime, None if unified else planner.output_a_target)]
  assert publication.longitudinalPlan.modelMonoTime == 1_000_000_000
  assert publication.longitudinalPlan.aTarget == float(np.float32(-0.123456789))
  expected_valid = input_valid and solver_status == 0 and (not planner.unified or intent_valid)
  assert publication.valid == expected_valid
  assert publication.longitudinalPlan.solverStatus == solver_status
  if planner.unified:
    assert replay.decode_motion_intent(publication.longitudinalPlan) == planner.motion_intent
    reference = publication.longitudinalPlan.reference
    assert reference.timingKnown and reference.initialStateKnown
    assert reference.sourceCarStateMonoTime == reference.referenceMonoTime == 1_000_000_000
    assert reference.producerMonoTime == reference.producerInstanceMonoTime == 1_020_000_000


def closed_runtime_fixture(monkeypatch):
  messages, window, controls, encoded, intent = replay_fixture(monkeypatch)
  additional = deepcopy([item for item in messages if item.logMonoTime >= 1_000_000_000])
  for item in additional:
    item.logMonoTime += 100_000_000
    if item.which() == 'longitudinalPlan':
      item.longitudinalPlan.modelMonoTime += 100_000_000
    if item.which() == 'controlsState':
      item.controlsState.longitudinalPlanMonoTime += 100_000_000
  messages.extend(additional)
  window.update(start=1.04, warmup=1.04, end=1.15)
  boundaries = {'planner': 1_010_000_000, 'longcontrol': 1_020_000_000, 'carcontroller': 1_040_000_000}
  latest, models, inputs, controls_edges, can_edges = {}, {}, [], [], []
  pending_state = pending_can = None
  for item in messages:
    service = item.which()
    mono = int(item.logMonoTime)
    if service in replay.SERVICES:
      latest[service] = mono
    if service == 'modelV2':
      models[mono] = dict(latest)
    elif service == 'longitudinalPlan' and mono >= boundaries['planner']:
      snapshot = models[item.longitudinalPlan.modelMonoTime]
      inputs.append([mono, mono + 1_000_000, *[snapshot[name] for name in replay.SERVICES]])
    elif service == 'controlsState':
      pending_state = (item.controlsState.longitudinalPlanMonoTime, mono, latest['carState'])
    elif service == 'carControl' and mono >= boundaries['longcontrol']:
      controls_edges.append([mono, *pending_state, False])
    elif service == 'carState' and 'carControl' in latest:
      pending_can = (latest['carControl'], mono)
    elif service == 'sendcan':
      can_edges.append([mono, *pending_can, True])
  contract = {
    'schema': 1, 'policy': replay.EXECUTION_POLICY, 'status': 'CLOSED',
    'requested_warmup_mono': 1_040_000_000, 'runtime_boundaries_mono': boundaries,
    'runtime_ids': {'planner': [row[0] for row in inputs], 'longcontrol': [row[0] for row in controls_edges],
                    'carcontroller': [row[0] for row in can_edges]},
    'runtime_ends_mono': {'planner': inputs[-1][0], 'longcontrol': controls_edges[-1][0], 'carcontroller': can_edges[-1][0]},
    'dependency_control_ids': [1_020_000_000],
    'edges': {'plan_inputs_fields': ['longitudinalPlan', 'longitudinalPlanSP', *replay.SERVICES],
              'plan_inputs': inputs, 'control_plan_state': controls_edges, 'can_control_state': can_edges},
  }
  contract['identity_sha256'] = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
  return messages, window, controls, encoded, intent, contract


@pytest.mark.parametrize('mode', [0, 1, 2])
def test_exact_dependency_contract_recovers_cross_boundary_can_without_scoring_prefix(monkeypatch, mode):
  messages, window, controls, encoded, _, contract = closed_runtime_fixture(monkeypatch)
  result = replay.replay_window(None, window, messages=messages, unified=mode == 2, supervised=mode == 1,
                                execution_contract=contract)
  assert len(controls) == len(encoded) == 2
  assert len(result['plan_rows']) == len(result['control_rows']) == 1
  assert len(result['can_rows']) == 2
  assert [row['mono'] for row in result['dependency_outputs']] == [1_020_000_000]
  assert result['can_rows'][0]['control_mono'] == result['dependency_outputs'][0]['mono']
  assert result['first_executed'] == {'plan_mono': 1_010_000_000, 'control_mono': 1_020_000_000, 'controller_mono': 1_040_000_000}
  assert result['runtime_execution_boundaries_mono'] == contract['runtime_boundaries_mono']
  assert result['resets'] == []
  assert result['execution_closure']['completed_runtime_counts'] == dict.fromkeys(('planner', 'longcontrol', 'carcontroller'), 2)
  assert result['capabilities']['planner_mode'] == result['capabilities']['controller_mode'] == mode
  if mode:
    assert result['motion_replay']['accepted_intent_modes'] == {str(mode): 1}


def test_dependency_contract_detects_missing_runtime_cycle_and_keeps_actual_diagnostics(monkeypatch):
  messages, window, _, _, _, contract = closed_runtime_fixture(monkeypatch)
  messages = [item for item in messages if item.logMonoTime != 1_120_000_000]
  diagnostics = {}
  with pytest.raises(ValueError, match='different dependency ID|Incomplete runtime'):
    replay.replay_window(None, window, messages=messages, supervised=True, execution_contract=contract, diagnostics=diagnostics)
  assert diagnostics['status'] == 'ERROR'
  assert diagnostics['counts']['longcontrol']['completed']['total'] == 1
  assert diagnostics['execution_closure']['identity_sha256'] == contract['identity_sha256']


def test_dependency_contract_is_bound_to_the_actual_planner_input_ids(monkeypatch):
  messages, window, _, _, _, contract = closed_runtime_fixture(monkeypatch)
  model = next(item for item in messages if item.which() == 'modelV2' and item.logMonoTime == 1_000_000_000)
  model.logMonoTime += 1
  next(item for item in messages if item.which() == 'longitudinalPlan').longitudinalPlan.modelMonoTime += 1
  with pytest.raises(ValueError, match='Planner input dependency identity changed'):
    replay.replay_window(None, window, messages=messages, supervised=True, execution_contract=contract)


def test_mutated_execution_boundaries_or_ids_require_a_new_audited_contract(monkeypatch):
  messages, window, _, _, _, contract = closed_runtime_fixture(monkeypatch)
  contract['runtime_boundaries_mono']['planner'] -= 1
  with pytest.raises(ValueError, match='Invalid runtime execution dependency contract'):
    replay.replay_window(None, window, messages=messages, execution_contract=contract)


def test_supervised_is_not_inferred_from_the_legacy_unified_boolean():
  controller = SimpleNamespace(unified=True, longitudinal_mode=1)
  assert replay.component_mode(controller) == 1
  state = {'controlsState': Snapshot(motionState=Snapshot())}
  replay.apply_motion_feedback(state, controller)
  assert state['controlsState'].motionState.planningMode == 1
  assert state['controlsState'].motionState.version == replay.MOTION_PROTOCOL_VERSION


def test_mutually_exclusive_motion_modes_are_rejected_before_runtime(monkeypatch):
  messages, window, controls, _, _ = replay_fixture(monkeypatch)
  with pytest.raises(ValueError, match='mutually exclusive'):
    replay.replay_window(None, window, messages=messages, supervised=True, unified=True)
  assert not controls


def test_consumer_health_reuses_native_frequency_checks_without_counting_duplicate_plan_ids():
  health = replay.ReplayConsumerHealth()
  start = 1_000_000_000
  for index in range(12):
    mono = start + index * 10_000_000
    health.observe(mono, mono - 1, start + index // 5 * 50_000_000 - 2)
  status = health.at(start + 115_000_000)
  assert all(status['carState'].values())
  assert all(status['longitudinalPlan'].values())
  stale = health.at(start + 1_000_000_000)
  assert not stale['carState']['alive']
  assert not stale['longitudinalPlan']['alive']


@pytest.mark.parametrize('automatic_api', [False, True])
@pytest.mark.parametrize('failure', [None, 'can_valid', 'car_state_valid', 'car_state_frequency', 'plan_alive', 'plan_frequency'])
def test_braking_health_is_independent_of_invalid_failed_reference_publication(monkeypatch, failure, automatic_api):
  messages, window, controls, _, intent = replay_fixture(monkeypatch, solver_status=4, envelope_api=True, automatic_api=automatic_api)
  intent.valid = False
  intent.allow_creep = False
  intent.braking_accel = -2.5

  class Health:
    def observe(self, *arguments):
      return None

    def at(self, now_mono):
      return {
        'carState': {'seen': True, 'alive': True, 'frequency_ok': failure != 'car_state_frequency'},
        'longitudinalPlan': {'seen': True, 'alive': failure != 'plan_alive', 'frequency_ok': failure != 'plan_frequency'},
      }

  monkeypatch.setattr(replay, 'ReplayConsumerHealth', Health)
  for item in messages:
    if item.which() == 'carState':
      item.carState.canValid = failure != 'can_valid'
      item.valid = failure != 'car_state_valid'
  result = replay.replay_window(None, window, messages=messages, supervised=not automatic_api, unified=automatic_api)
  assert controls[-1][1]['braking_plan_valid'] is (failure is None)
  assert controls[-1][1]['plan_valid'] is False
  assert not result['plan_rows'][0]['published_valid']
  assert result['control_rows'][0]['intent_braking_valid']
  assert result['control_rows'][0]['intent_braking_accel'] == -2.5


@pytest.mark.parametrize('automatic_api', [False, True])
@pytest.mark.parametrize('established', [False, True])
@pytest.mark.parametrize('pedal', [None, 'gasPressed', 'brakePressed'])
def test_mode_fault_preserves_established_owner_without_overriding_driver(monkeypatch, established, pedal, automatic_api):
  messages, window, controls, encoded, intent = replay_fixture(
    monkeypatch, envelope_api=True, owner_established=established, automatic_api=automatic_api)
  intent.version = 1
  if pedal is not None:
    for item in messages:
      if item.which() == 'carState':
        setattr(item.carState, pedal, True)
  result = replay.replay_window(None, window, messages=messages, supervised=not automatic_api, unified=automatic_api)
  assert controls[-1][0][0] is (established and pedal is None)
  assert controls[-1][1]['plan_valid'] is False
  assert controls[-1][1]['braking_plan_valid'] is False
  assert result['control_rows'][0]['mode_mismatch']
  assert not encoded[-1].cruiseControl.resume
  if established and pedal is None:
    assert result['control_rows'][0]['motion_degraded']
    assert encoded[-1].longActive
    assert encoded[-1].actuators.accel < 0


def test_legacy_component_is_not_given_the_new_braking_health_keyword(monkeypatch):
  messages, window, controls, _, _ = replay_fixture(monkeypatch, legacy_api=True, publish_solver_status=False)
  result = replay.replay_window(None, window, messages=messages)
  assert controls[-1][1] == {}
  assert not result['capabilities']['controller_braking_health_protocol']


def test_explicit_initialization_wait_does_not_call_planner_or_drop_controller_cycles(monkeypatch):
  messages, window, controls, encoded, _, contract = closed_runtime_fixture(monkeypatch)
  window.update(start=1.1, warmup=1.0)
  messages = [item for item in messages if item.which() != 'vehicleParameters']
  messages.append(Snapshot(which=lambda: 'vehicleParameters', logMonoTime=1_005_000_000, valid=True, vehicleParameters=Snapshot()))
  messages.sort(key=lambda item: item.logMonoTime)
  control = next(item.carControl for item in messages if item.which() == 'carControl' and item.logMonoTime == 1_020_000_000)
  control.longActive = False
  control.actuators.longControlState = replay.LongCtrlState.off
  control.actuators.accel = 0.0
  contract['requested_warmup_mono'] = 1_000_000_000
  contract['runtime_boundaries_mono'] = dict.fromkeys(('planner', 'longcontrol', 'carcontroller'), 1_000_000_000)
  contract['runtime_ids']['planner'] = [1_110_000_000]
  contract['dependency_control_ids'] = []
  contract['edges']['plan_inputs'] = contract['edges']['plan_inputs'][1:]
  contract['edges']['plan_inputs'][0][contract['edges']['plan_inputs_fields'].index('vehicleParameters')] = 1_005_000_000
  contract['edges']['control_plan_state'][0][-1] = True
  contract['initialization_wait'] = [{
    'mono': 1_010_000_000, 'model_mono': 1_000_000_000, 'sp_mono': 1_011_000_000,
    'reason': 'unscored_initial_snapshot_unavailable_without_active_control_consumers',
    'control_consumers': [1_020_000_000], 'required_inputs': [{'service': 'vehicleParameters', 'mono': None}],
    'solver_called': False, 'generated_plan': False,
  }]
  del contract['identity_sha256']
  contract['identity_sha256'] = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
  result = replay.replay_window(None, window, messages=messages, supervised=True, execution_contract=contract)
  assert result['first_executed']['plan_mono'] == 1_110_000_000
  assert result['diagnostics']['counts']['planner']['completed']['total'] == 1
  assert len(controls) == len(encoded) == 2
  assert controls[0][0][0] is False
  assert controls[0][0][2:4] == (None, None)
  assert result['initialization_wait'] == contract['initialization_wait']
  assert len(result['plan_rows']) == len(result['control_rows']) == len(result['can_rows']) == 1
