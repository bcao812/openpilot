"""Shared persisted buzzer selection for settings and the GPIO alert process."""
from enum import IntEnum


class SoundMode(IntEnum):
  NONE = 0
  WARNING = 1
  ALL = 2


def get_sound_mode(params) -> SoundMode:
  try:
    return SoundMode(params.get("SoundMode", return_default=True))
  except (TypeError, ValueError):
    return SoundMode.ALL
