from opendbc.car.honda.values import CAR
from opendbc.car.structs import car


def restrict_longitudinal_authority(params, params_sp, car_state, control):
  """Revoke cached longitudinal requests against the latest pedal state."""
  supported = (params.carFingerprint == CAR.HONDA_ACCORD and params.openpilotLongitudinalControl
               and not params_sp.enableGasInterceptor)
  if not supported or not (car_state.gasPressed or car_state.brakePressed):
    return control
  restricted = control.as_builder()
  restricted.longActive = False
  restricted.actuators.accel = 0.0
  restricted.actuators.longControlState = car.CarControl.Actuators.LongControlState.off
  restricted.cruiseControl.resume = False
  restricted.cruiseControl.override = bool(control.enabled)
  return restricted.as_reader()
