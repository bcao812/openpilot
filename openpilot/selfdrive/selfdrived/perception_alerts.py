"""One display entry for model/localization delays; fault events stay intact."""

from copy import copy

from openpilot.cereal import log
from openpilot.selfdrive.selfdrived.events import Alert, AlertSize, AlertStatus, AudibleAlert, ET, Priority, VisualAlert
from openpilot.system.ui.lib.multilang import tr_noop


EventName = log.OnroadEvent.EventName
PERCEPTION_ALERT = 'perceptionInputDelay'
SOURCE_ALERTS = frozenset(f'{name}/{kind}' for name in ('modeldLagging', 'locationdTemporaryError')
                          for kind in (ET.SOFT_DISABLE, ET.NO_ENTRY, ET.PERMANENT))


def merge_perception_alerts(alerts, events, manager, frame, *, recovered_model_lag=False, lateral_takeover=''):
  # Remove only the entries being replaced, including their minimum-duration
  # caches. Other warnings retain their original priority and duration.
  grouped = [alert for alert in alerts if alert.alert_type in SOURCE_ALERTS]
  remaining = [alert for alert in alerts if alert.alert_type not in SOURCE_ALERTS]
  for key in SOURCE_ALERTS:
    manager.alerts.pop(key, None)

  model_lag = events.has(EventName.modeldLagging)
  location_fault = events.has(EventName.locationdTemporaryError)
  if not (model_lag or location_fault):
    manager.alerts.pop(PERCEPTION_ALERT, None)
    return remaining

  if recovered_model_lag:
    grouped = [alert for alert in grouped if alert.alert_type != 'modeldLagging/softDisable']
  if not grouped:
    return remaining

  selected = max(grouped, key=lambda alert: alert.priority)
  recovered = recovered_model_lag and not location_fault and selected.event_type != ET.NO_ENTRY
  if lateral_takeover in ('pending', 'disabled') and model_lag and not location_fault and selected.event_type != ET.NO_ENTRY:
    detail = 'Steering assist ending; speed control active' if lateral_takeover == 'pending' else 'Steering assist unavailable; speed control active'
    combined = Alert(tr_noop('Steer Manually'), detail, AlertStatus.critical, AlertSize.full, Priority.HIGH,
                     VisualAlert.steerRequired, AudibleAlert.warningImmediate, 0.1)
    combined.event_type = ET.PERMANENT
  elif recovered:
    combined = Alert(tr_noop('Model / Localization Delay'), 'Inputs recovered; monitoring frame drops',
                     AlertStatus.normal, AlertSize.mid, Priority.LOW, VisualAlert.none, AudibleAlert.none, 0.2)
    combined.event_type = ET.PERMANENT
  else:
    combined = copy(selected)
    if selected.event_type == ET.PERMANENT:
      combined.alert_text_1 = tr_noop('Model / Localization Delay')
    else:
      combined.alert_text_2 = tr_noop('Model / Localization Delay')
    # Preserve an unrecovered takeover prompt through its ordinary minimum
    # duration after disengagement. Recovery may replace it immediately.
    cached = manager.alerts.get(PERCEPTION_ALERT)
    if cached is not None and cached.active(frame) and cached.alert is not None and cached.alert.priority > combined.priority:
      return remaining
  combined.alert_type = PERCEPTION_ALERT
  remaining.append(combined)
  return remaining
