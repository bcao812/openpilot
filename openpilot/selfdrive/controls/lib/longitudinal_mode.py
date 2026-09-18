from enum import IntEnum

from opendbc.car.honda.values import CAR
from openpilot.common.params import Params


class LongitudinalMode(IntEnum):
  LEGACY = 0
  SUPERVISED = 1
  UNIFIED = 2


def resolve_longitudinal_mode(CP, CP_SP, *, unified: bool | None = None, supervised: bool | None = None) -> LongitudinalMode:
  if CP.carFingerprint != CAR.HONDA_ACCORD or not CP.openpilotLongitudinalControl or CP_SP.enableGasInterceptor:
    return LongitudinalMode.LEGACY
  params = Params() if unified is None or supervised is None else None
  unified = params.get_bool("LongitudinalUnifiedEnabled") if unified is None else unified
  supervised = params.get_bool("LongitudinalMotionEnabled") if supervised is None else supervised
  if unified:
    return LongitudinalMode.UNIFIED
  if supervised:
    return LongitudinalMode.SUPERVISED
  return LongitudinalMode.LEGACY
