from openpilot.cereal import log
from openpilot.selfdrive.selfdrived.events import Events, ET
from openpilot.common.realtime import DT_CTRL

State = log.SelfdriveState.OpenpilotState

SOFT_DISABLE_TIME = 3  # seconds
ACTIVE_STATES = (State.enabled, State.softDisabling, State.overriding)
ENABLED_STATES = (State.preEnabled, *ACTIVE_STATES)

class StateMachine:
  def __init__(self):
    self.current_alert_types = [ET.PERMANENT]
    self.state = State.disabled
    self.soft_disable_timer = 0
    self.holding_for_takeover = False

  def update(self, events: Events, *, soft_disable_remaining_ns=None, hold_for_takeover=False):
    was_holding = self.holding_for_takeover
    self.holding_for_takeover = False
    # decrement the soft disable timer at every step, as it's reset on
    # entrance in SOFT_DISABLING state
    self.soft_disable_timer = max(0, self.soft_disable_timer - 1)
    # A bounded continuation cannot restart the total fault deadline when
    # this state is entered again. This optional bound only shortens a timer.
    maximum_timer = int(SOFT_DISABLE_TIME / DT_CTRL)
    if type(soft_disable_remaining_ns) is int and soft_disable_remaining_ns >= 0:
      maximum_timer = min(maximum_timer, soft_disable_remaining_ns // round(DT_CTRL * 1e9))
      self.soft_disable_timer = min(self.soft_disable_timer, maximum_timer)

    self.current_alert_types = [ET.PERMANENT]

    # ENABLED, SOFT DISABLING, PRE ENABLING, OVERRIDING
    if self.state != State.disabled:
      # user and immediate disable always have priority in a non-disabled state
      if events.contains(ET.IMMEDIATE_DISABLE):
        self.state = State.disabled
        self.current_alert_types.append(ET.IMMEDIATE_DISABLE)

      elif events.contains(ET.USER_DISABLE):
        self.state = State.disabled
        self.current_alert_types.append(ET.USER_DISABLE)

      elif hold_for_takeover is True and self.state == State.softDisabling:
        # The independent policy qualifies an already held stop and owns one
        # absolute extension. Keep forceDecel and the urgent alert throughout.
        self.holding_for_takeover = True
        self.soft_disable_timer = 0
        self.current_alert_types.append(ET.SOFT_DISABLE)

      elif was_holding:
        # Expiry or lost qualification ends this engagement, even if the
        # original planner fault disappeared during the stopped handoff.
        self.state = State.disabled
        self.current_alert_types.append(ET.SOFT_DISABLE)

      elif maximum_timer == 0 and events.contains(ET.SOFT_DISABLE):
        # Includes preEnabled losing PRE_ENABLE at an already expired fault
        # deadline: no path may produce an extra active control tick.
        self.state = State.disabled
        self.current_alert_types.append(ET.SOFT_DISABLE)

      else:
        # ENABLED
        if self.state == State.enabled:
          if events.contains(ET.SOFT_DISABLE):
            self.state = State.softDisabling if maximum_timer else State.disabled
            self.soft_disable_timer = maximum_timer
            self.current_alert_types.append(ET.SOFT_DISABLE)

          elif events.contains(ET.OVERRIDE_LATERAL) or events.contains(ET.OVERRIDE_LONGITUDINAL):
            self.state = State.overriding
            self.current_alert_types += [ET.OVERRIDE_LATERAL, ET.OVERRIDE_LONGITUDINAL]

        # SOFT DISABLING
        elif self.state == State.softDisabling:
          if not events.contains(ET.SOFT_DISABLE):
            # no more soft disabling condition, so go back to ENABLED
            self.state = State.enabled

          elif self.soft_disable_timer > 0:
            self.current_alert_types.append(ET.SOFT_DISABLE)

          elif self.soft_disable_timer <= 0:
            self.state = State.disabled

        # PRE ENABLING
        elif self.state == State.preEnabled:
          if not events.contains(ET.PRE_ENABLE):
            self.state = State.enabled
          else:
            self.current_alert_types.append(ET.PRE_ENABLE)

        # OVERRIDING
        elif self.state == State.overriding:
          if events.contains(ET.SOFT_DISABLE):
            self.state = State.softDisabling if maximum_timer else State.disabled
            self.soft_disable_timer = maximum_timer
            self.current_alert_types.append(ET.SOFT_DISABLE)
          elif not (events.contains(ET.OVERRIDE_LATERAL) or events.contains(ET.OVERRIDE_LONGITUDINAL)):
            self.state = State.enabled
          else:
            self.current_alert_types += [ET.OVERRIDE_LATERAL, ET.OVERRIDE_LONGITUDINAL]

    # DISABLED
    elif self.state == State.disabled:
      if events.contains(ET.ENABLE):
        if events.contains(ET.NO_ENTRY):
          self.current_alert_types.append(ET.NO_ENTRY)

        else:
          if events.contains(ET.PRE_ENABLE):
            self.state = State.preEnabled
          elif events.contains(ET.OVERRIDE_LATERAL) or events.contains(ET.OVERRIDE_LONGITUDINAL):
            self.state = State.overriding
          else:
            self.state = State.enabled
          self.current_alert_types.append(ET.ENABLE)

    # Check if openpilot is engaged and actuators are enabled
    enabled = self.state in ENABLED_STATES
    active = self.state in ACTIVE_STATES
    if active:
      self.current_alert_types.append(ET.WARNING)
    return enabled, active
