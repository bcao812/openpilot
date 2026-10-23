"""Per-drive radar selection and a vision-only view for its consumers.

The setting is changed offroad and latched when each onroad process starts.
Raw vehicle CAN and OEM safety/status messages are independent of this setting.
"""
def radar_signal_enabled(params) -> bool:
  value = params.get("RadarSignal")
  return True if value is None else bool(value)


def lead_uses_radar(lead) -> bool:
  evidence = getattr(lead, "evidence", None)
  return bool(lead.radar) or str(getattr(evidence, "source", "legacy")) in ("radar", "fused", "predicted")


def radar_state_for_mode(radar, enabled: bool):
  """Discard all radar-derived objects and diagnostics without altering the input.

  Historical visionLead fields can themselves contain a standard radar match.
  Only explicitly non-radar leads may survive in the disabled view. The normal
  disabled producer supplies current vision leads in all four lead fields.
  """
  if enabled:
    return radar
  from openpilot.cereal import log
  vision = log.RadarState.new_message()
  vision.mdMonoTime = radar.mdMonoTime
  for name in ("leadOne", "leadTwo", "visionLeadOne", "visionLeadTwo"):
    lead = getattr(radar, name)
    if not lead_uses_radar(lead):
      setattr(vision, name, lead)
  return vision.as_reader()


class RadarSignalInput:
  """Replace only the radar payload; retain the original input-health checks."""
  def __init__(self, state, enabled: bool):
    self._state = state
    self._radar = radar_state_for_mode(state["radarState"], enabled)

  def __getitem__(self, key):
    return self._radar if key == "radarState" else self._state[key]

  def __getattr__(self, name):
    return getattr(self._state, name)
