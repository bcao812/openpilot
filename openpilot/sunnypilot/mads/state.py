"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from openpilot.cereal import log, custom
from openpilot.selfdrive.selfdrived.events import ET
from openpilot.selfdrive.selfdrived.state import SOFT_DISABLE_TIME
from openpilot.common.realtime import DT_CTRL

State = custom.ModularAssistiveDrivingSystem.ModularAssistiveDrivingSystemState
EventName = log.OnroadEvent.EventName
EventNameSP = custom.OnroadEventSP.EventName

ACTIVE_STATES = (State.enabled, State.softDisabling, State.overriding)
ENABLED_STATES = (State.paused, *ACTIVE_STATES)

GEARS_ALLOW_PAUSED_SILENT = [EventNameSP.silentWrongGear, EventNameSP.silentReverseGear, EventNameSP.silentBrakeHold,
                             EventNameSP.silentDoorOpen, EventNameSP.silentSeatbeltNotLatched, EventNameSP.silentParkBrake]
GEARS_ALLOW_PAUSED = [EventName.wrongGear, EventName.reverseGear, EventName.brakeHold,
                      EventName.doorOpen, EventName.seatbeltNotLatched, EventName.parkBrake]


class StateMachine:
  def __init__(self, mads):
    self.mads = mads
    self.selfdrive = mads.selfdrive
    self.ss_state_machine = mads.selfdrive.state_machine
    self._events = mads.selfdrive.events
    self._events_sp = mads.selfdrive.events_sp

    self.state = State.disabled
    self.soft_disable_timer = 0

  def add_current_alert_types(self, alert_type):
    if not self.selfdrive.enabled:
      self.ss_state_machine.current_alert_types.append(alert_type)

  def check_contains(self, event_type: str) -> bool:
    isolated_longitudinal = getattr(self.mads, 'longitudinal_failure_isolated', False) is True
    recovered_model_lag = getattr(self.mads, 'model_lag_recovered', False) is True
    if event_type == ET.SOFT_DISABLE and self.state in ACTIVE_STATES and (isolated_longitudinal or recovered_model_lag):
      # Preserve established lateral authority only. Keep the original event,
      # including its NO_ENTRY and alert types, for the other consumers.
      mapping = self._events.get_events_mapping()
      return (any(event_type in mapping.get(event, {}) for event in self._events.names
                  if not (isolated_longitudinal and event == EventName.longitudinalControlUnavailable)
                  and not (recovered_model_lag and event == EventName.modeldLagging))
              or self._events_sp.contains(event_type))
    return bool(self._events.contains(event_type) or self._events_sp.contains(event_type))

  def check_contains_in_list(self) -> bool:
    return bool(self._events.contains_in_list(GEARS_ALLOW_PAUSED) or self._events_sp.contains_in_list(GEARS_ALLOW_PAUSED_SILENT))

  def update(self):
    # Lateral faults own their deadline; a longitudinal fault cannot spend or
    # reset it. Alert types are still merged into the main state machine.
    self.soft_disable_timer = max(0, self.soft_disable_timer - 1)

    # ENABLED, SOFT DISABLING, PAUSED, OVERRIDING
    if self.state != State.disabled:
      # user and immediate disable always have priority in a non-disabled state
      if self.check_contains(ET.USER_DISABLE):
        if self._events_sp.has(EventNameSP.silentLkasDisable):
          self.state = State.paused
        else:
          self.state = State.disabled
        self.ss_state_machine.current_alert_types.append(ET.USER_DISABLE)

      elif self.check_contains(ET.IMMEDIATE_DISABLE):
        self.state = State.disabled
        self.add_current_alert_types(ET.IMMEDIATE_DISABLE)

      else:
        # ENABLED
        if self.state == State.enabled:
          if self.check_contains(ET.SOFT_DISABLE):
            self.state = State.softDisabling
            self.soft_disable_timer = int(SOFT_DISABLE_TIME / DT_CTRL)
            if not self.selfdrive.enabled:
              self.ss_state_machine.current_alert_types.append(ET.SOFT_DISABLE)

          elif self.check_contains(ET.OVERRIDE_LATERAL):
            self.state = State.overriding
            self.add_current_alert_types(ET.OVERRIDE_LATERAL)

        # SOFT DISABLING
        elif self.state == State.softDisabling:
          if not self.check_contains(ET.SOFT_DISABLE):
            # no more soft disabling condition, so go back to ENABLED
            self.state = State.enabled

          elif self.soft_disable_timer > 0:
            self.add_current_alert_types(ET.SOFT_DISABLE)

          elif self.soft_disable_timer <= 0:
            self.state = State.disabled

        # PAUSED
        elif self.state == State.paused:
          if self.check_contains(ET.ENABLE):
            if self.check_contains(ET.NO_ENTRY):
              self.add_current_alert_types(ET.NO_ENTRY)

            else:
              if self.check_contains(ET.OVERRIDE_LATERAL):
                self.state = State.overriding
              else:
                self.state = State.enabled
              self.add_current_alert_types(ET.ENABLE)

        # OVERRIDING
        elif self.state == State.overriding:
          if self.check_contains(ET.SOFT_DISABLE):
            self.state = State.softDisabling
            self.soft_disable_timer = int(SOFT_DISABLE_TIME / DT_CTRL)
            if not self.selfdrive.enabled:
              self.ss_state_machine.current_alert_types.append(ET.SOFT_DISABLE)
          elif not self.check_contains(ET.OVERRIDE_LATERAL):
            self.state = State.enabled
          else:
            self.ss_state_machine.current_alert_types += [ET.OVERRIDE_LATERAL]

    # DISABLED
    elif self.state == State.disabled:
      if self.check_contains(ET.ENABLE):
        if self.check_contains(ET.NO_ENTRY):
          if self.check_contains_in_list():
            self.state = State.paused
          self.add_current_alert_types(ET.NO_ENTRY)

        else:
          if self.check_contains(ET.OVERRIDE_LATERAL):
            self.state = State.overriding
          else:
            self.state = State.enabled
          self.add_current_alert_types(ET.ENABLE)

    # check if MADS is engaged and actuators are enabled
    enabled = self.state in ENABLED_STATES
    active = self.state in ACTIVE_STATES
    if active:
      self.add_current_alert_types(ET.WARNING)

    return enabled, active
