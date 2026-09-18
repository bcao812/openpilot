from types import SimpleNamespace

import pytest

from opendbc.can import CANParser
from opendbc.car import Bus, structs
from opendbc.car.honda.carcontroller import CarController
from opendbc.car.honda.hondacan import CanBus
from opendbc.car.honda.interface import CarInterface
from opendbc.car.honda.values import CAR, DBC, HondaFlags
from opendbc.car.structs import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.car.longitudinal_authority import restrict_longitudinal_authority


def authority_inputs(*, platform=CAR.HONDA_ACCORD, openpilot_long=True, interceptor=False, gas=False, brake=False):
  params = SimpleNamespace(carFingerprint=platform, openpilotLongitudinalControl=openpilot_long)
  params_sp = SimpleNamespace(enableGasInterceptor=interceptor)
  car_state = SimpleNamespace(gasPressed=gas, brakePressed=brake)
  control = car.CarControl.new_message()
  control.enabled = True
  control.latActive = True
  control.longActive = True
  control.actuators.accel = 0.5
  control.actuators.longControlState = car.CarControl.Actuators.LongControlState.pid
  control.actuators.torque = 0.25
  control.actuators.curvature = 0.01
  control.cruiseControl.resume = True
  return params, params_sp, car_state, control


@pytest.mark.parametrize('gas,brake', [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize('accel', [-3.5, -0.2, 0.5, 2.0])
def test_latest_driver_pedals_revoke_cached_longitudinal_authority_only(gas, brake, accel):
  params, params_sp, car_state, command = authority_inputs(gas=gas, brake=brake)
  command.actuators.accel = accel
  original = command.as_reader()
  before = original.to_dict()
  restricted = restrict_longitudinal_authority(params, params_sp, car_state, original)
  assert original.to_dict() == before
  assert not restricted.longActive
  assert restricted.actuators.accel == 0.0
  assert restricted.actuators.longControlState == car.CarControl.Actuators.LongControlState.off
  assert not restricted.cruiseControl.resume
  assert restricted.cruiseControl.override
  assert restricted.enabled == original.enabled
  assert restricted.latActive == original.latActive
  assert restricted.actuators.torque == original.actuators.torque
  assert restricted.actuators.curvature == original.actuators.curvature
  assert restricted.hudControl.to_dict() == original.hudControl.to_dict()


@pytest.mark.parametrize(
  'platform,openpilot_long,interceptor',
  [
    (CAR.HONDA_CIVIC, True, False),
    (CAR.HONDA_CIVIC_BOSCH, True, False),
    (CAR.HONDA_ACCORD, False, False),
    (CAR.HONDA_ACCORD, True, True),
  ],
)
def test_manual_restriction_does_not_change_other_platforms_or_stock_control(platform, openpilot_long, interceptor):
  params, params_sp, car_state, command = authority_inputs(platform=platform, openpilot_long=openpilot_long, interceptor=interceptor, brake=True)
  original = command.as_reader()
  assert restrict_longitudinal_authority(params, params_sp, car_state, original) is original


def test_healthy_automatic_control_has_no_copy_or_behavior_change():
  params, params_sp, car_state, command = authority_inputs()
  original = command.as_reader()
  assert restrict_longitudinal_authority(params, params_sp, car_state, original) is original


def test_restriction_never_enables_a_disabled_controller():
  params, params_sp, car_state, command = authority_inputs(brake=True)
  command.enabled = False
  command.latActive = False
  restricted = restrict_longitudinal_authority(params, params_sp, car_state, command.as_reader())
  assert not restricted.enabled
  assert not restricted.latActive
  assert not restricted.longActive
  assert not restricted.cruiseControl.override


def honda_authority_chain():
  params = CarInterface.get_non_essential_params(CAR.HONDA_ACCORD)
  params.openpilotLongitudinalControl = True
  params_sp = CarInterface.get_non_essential_params_sp(params, CAR.HONDA_ACCORD)
  assert params.flags & HondaFlags.BOSCH
  assert not params.flags & HondaFlags.BOSCH_RADARLESS
  assert not params_sp.enableGasInterceptor
  dbc_name = DBC[params.carFingerprint][Bus.pt]
  bus = CanBus(params)
  controller = CarController({Bus.pt: dbc_name}, params, params_sp)
  vehicle = structs.CarState()
  vehicle.standstill = True
  vehicle.cruiseState.available = True
  wrapper = SimpleNamespace(v_cruise_factor=1.0, is_metric=False, acc_hud={}, lkas_hud={}, stock_brake={})
  return SimpleNamespace(
    params=params,
    params_sp=params_sp,
    controller=controller,
    vehicle=vehicle,
    wrapper=wrapper,
    control_sp=structs.CarControlSP(),
    acc_parser=CANParser(dbc_name, [('ACC_CONTROL', 50)], bus.pt),
    steering_parser=CANParser(dbc_name, [('STEERING_CONTROL', 100)], bus.lkas),
  )


def authority_can_step(chain, source):
  original = source.to_dict()
  latest_state = chain.vehicle.as_reader()
  restricted = restrict_longitudinal_authority(chain.params, chain.params_sp, latest_state, source)
  chain.wrapper.out = latest_state
  frame = chain.controller.frame
  timestamp = round((frame + 1) * DT_CTRL * 1e9)
  actuators, messages = chain.controller.update(restricted, chain.control_sp, chain.wrapper, timestamp)
  acc_updated = chain.acc_parser.update([(timestamp, messages)])
  steering_updated = chain.steering_parser.update([(timestamp, messages)])
  acc_address = chain.acc_parser.dbc.name_to_msg['ACC_CONTROL'].address
  steering_address = chain.steering_parser.dbc.name_to_msg['STEERING_CONTROL'].address
  assert steering_address in steering_updated
  assert source.to_dict() == original
  return SimpleNamespace(
    frame=frame,
    control=restricted,
    actuators=actuators,
    acc=dict(chain.acc_parser.vl['ACC_CONTROL']) if acc_address in acc_updated else None,
    steering=dict(chain.steering_parser.vl['STEERING_CONTROL']),
  )


def assert_inactive_acc_request(chain, request):
  assert request['GAS_COMMAND'] == -30000
  assert request['ACCEL_COMMAND'] == 0.0
  assert request['BRAKE_REQUEST'] == 0
  assert request['BRAKE_LIGHTS'] == 0
  assert request['STANDSTILL'] == 0
  assert request['STANDSTILL_RELEASE'] == 0
  assert chain.controller.stopping_counter == 0


@pytest.mark.parametrize('gas,brake', [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize('edge_parity', [0, 1])
def test_latest_pedal_edge_revokes_real_honda_hold_on_next_acc_frame(gas, brake, edge_parity):
  chain = honda_authority_chain()
  command = authority_inputs()[3]
  command.actuators.accel = -2.0
  command.actuators.longControlState = car.CarControl.Actuators.LongControlState.stopping
  command.cruiseControl.resume = False
  source = command.as_reader()
  original = source.to_dict()
  holding_requests = []
  for _frame in range(20):
    step = authority_can_step(chain, source)
    if step.acc is not None:
      holding_requests.append(step.acc)
  assert len(holding_requests) == 10
  assert holding_requests[-1]['BRAKE_REQUEST'] == 1
  assert holding_requests[-1]['STANDSTILL'] == 1
  assert holding_requests[-1]['STANDSTILL_RELEASE'] == 0
  assert chain.controller.stopping_counter > 0
  if chain.controller.frame % 2 != edge_parity:
    authority_can_step(chain, source)
  edge_frame = chain.controller.frame
  assert edge_frame % 2 == edge_parity
  assert chain.controller.stopping_counter > 0
  chain.vehicle.gasPressed = gas
  chain.vehicle.brakePressed = brake
  inactive_frames = []
  for _frame in range(40):
    step = authority_can_step(chain, source)
    assert not step.control.longActive
    assert step.control.actuators.accel == 0.0
    assert step.control.actuators.longControlState == car.CarControl.Actuators.LongControlState.off
    assert step.control.enabled and step.control.latActive
    assert step.control.actuators.torque == source.actuators.torque
    assert step.control.actuators.curvature == source.actuators.curvature
    assert step.steering['STEER_TORQUE_REQUEST'] == 1
    assert step.steering['STEER_TORQUE'] != 0
    assert (step.acc is not None) == (step.frame % 2 == 0)
    if step.acc is not None:
      inactive_frames.append(step.frame)
      assert_inactive_acc_request(chain, step.acc)
  assert len(inactive_frames) == 20
  assert inactive_frames[0] == edge_frame + edge_parity
  assert source.to_dict() == original

  chain.vehicle.gasPressed = False
  chain.vehicle.brakePressed = False
  disabled = source.as_builder()
  disabled.longActive = False
  disabled.actuators.accel = 0.0
  disabled.actuators.longControlState = car.CarControl.Actuators.LongControlState.off
  disabled_source = disabled.as_reader()
  assert restrict_longitudinal_authority(chain.params, chain.params_sp, chain.vehicle.as_reader(), disabled_source) is disabled_source
  disabled_requests = []
  for _frame in range(2):
    step = authority_can_step(chain, disabled_source)
    if step.acc is not None:
      disabled_requests.append(step.acc)
      assert_inactive_acc_request(chain, step.acc)
  assert len(disabled_requests) == 1

  resumed = source.as_builder()
  resumed.actuators.accel = 0.5
  resumed.actuators.longControlState = car.CarControl.Actuators.LongControlState.pid
  resumed.cruiseControl.resume = True
  resumed_source = resumed.as_reader()
  assert restrict_longitudinal_authority(chain.params, chain.params_sp, chain.vehicle.as_reader(), resumed_source) is resumed_source
  resumed_requests = []
  for _frame in range(2):
    step = authority_can_step(chain, resumed_source)
    if step.acc is not None:
      resumed_requests.append(step.acc)
      assert step.acc['GAS_COMMAND'] > 0
      assert step.acc['ACCEL_COMMAND'] == pytest.approx(0.5)
      assert step.acc['BRAKE_REQUEST'] == 0
      assert step.acc['STANDSTILL'] == 0
      assert step.acc['STANDSTILL_RELEASE'] == 1
      assert chain.controller.stopping_counter == 0
  assert len(resumed_requests) == 1
