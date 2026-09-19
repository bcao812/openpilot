from enum import IntEnum

from opendbc.car.honda.values import CAR


class LongitudinalMode(IntEnum):
  LEGACY = 0
  # Retired runtime mode. Keep the ordinal for historical message decoding.
  SUPERVISED = 1
  UNIFIED = 2


def resolve_longitudinal_mode(CP, CP_SP) -> LongitudinalMode:
  """Select the supported runtime from vehicle capabilities, never migration flags."""
  if CP.carFingerprint != CAR.HONDA_ACCORD or not CP.openpilotLongitudinalControl or CP_SP.enableGasInterceptor:
    return LongitudinalMode.LEGACY
  return LongitudinalMode.UNIFIED
