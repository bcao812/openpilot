"""Counter changes must preserve event lifetime and the alert creation frame."""

from copy import deepcopy

import pytest

from openpilot.cereal import custom, log
from openpilot.selfdrive.selfdrived.events import ET, Events
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP


def legacy_clear(events):
  events.event_counters = {key: (count + 1 if key in events.events else 0)
                           for key, count in events.event_counters.items()}
  events.events = events.static_events.copy()


def counter_state(events):
  return events.events.copy(), events.static_events.copy(), list(events.event_counters.items())


@pytest.mark.parametrize('event_class', [Events, EventsSP])
@pytest.mark.parametrize('population', ['empty', 'sparse', 'all', 'duplicates', 'unknown'])
def test_clear_matches_original_formula_and_preserves_keys_order_and_list_ownership(event_class, population):
  actual = event_class()
  names = list(actual.event_counters)
  actual.event_counters = {name: i % 17 for i, name in enumerate(reversed(names))}
  actual.add(names[0], static=True)
  actual.add(names[0], static=True)
  actual.add(names[-1], static=True)
  current = {'empty': [], 'sparse': names[::7], 'all': names,
             'duplicates': names[:3] * 3, 'unknown': [max(names) + 1000, names[0], max(names) + 1000]}
  actual.events = sorted(current[population])
  reference = deepcopy(actual)
  previous_counters, previous_events = actual.event_counters, actual.events
  original_counters, original_events = previous_counters.copy(), previous_events.copy()

  actual.clear()
  legacy_clear(reference)

  assert counter_state(actual) == counter_state(reference)
  assert previous_counters == original_counters and previous_events == original_events
  assert actual.event_counters is not previous_counters
  assert actual.events is not previous_events and actual.events is not actual.static_events
  assert list(actual.event_counters) == list(original_counters)


@pytest.mark.parametrize('event_class', [Events, EventsSP])
def test_schema_event_names_enter_the_collection_as_hashable_integer_ids(event_class):
  source, actual = event_class(), event_class()
  for name in source.get_events_mapping():
    assert type(name) is int
    source.add(name)
  messages = source.to_msg()
  assert all(type(message.name.raw) is int for message in messages)
  actual.add_from_msg([message.as_reader() for message in messages])
  assert actual.events == source.events and set(actual.events) == set(source.event_counters)
  actual.clear()
  assert all(count == 1 for count in actual.event_counters.values())


@pytest.mark.parametrize('event_class,event_name,static_name', [
  (Events, log.OnroadEvent.EventName.reverseGear, log.OnroadEvent.EventName.dashcamMode),
  (EventsSP, custom.OnroadEventSP.EventName.silentReverseGear, custom.OnroadEventSP.EventName.manualLongitudinalRequired),
])
def test_clear_retains_real_creation_delay_duplicates_remove_and_reappearance(event_class, event_name, static_name):
  actual, reference = event_class(), event_class()
  for events in (actual, reference):
    events.add(static_name, static=True)
  alert_type = f'{actual.get_event_name(event_name)}/{ET.PERMANENT}'

  # reverseGear and silentReverseGear require 0.5 seconds of consecutive frames.
  # The duplicate event is still one observed frame; absence restarts that timer.
  for frame in range(106):
    actual.clear()
    legacy_clear(reference)
    if frame < 51 or frame >= 53:
      for events in (actual, reference):
        events.add(event_name)
        events.add(event_name)
    alerts = [vars(alert).copy() for alert in actual.create_alerts([ET.PERMANENT])]
    expected = [vars(alert).copy() for alert in reference.create_alerts([ET.PERMANENT])]
    assert alerts == expected
    assert any(alert['alert_type'] == alert_type for alert in alerts) == (49 <= frame < 51 or frame >= 102)
    assert counter_state(actual) == counter_state(reference)

  # remove intentionally counts once and removes only one duplicate. clear must
  # then add exactly one more frame and restore the separate static collection.
  for events in (actual, reference):
    events.remove(event_name)
    events.remove(static_name, static=True)
  assert counter_state(actual) == counter_state(reference)
  before = actual.event_counters[event_name]
  actual.clear()
  legacy_clear(reference)
  assert actual.event_counters[event_name] == before + 1
  assert actual.events == [] and actual.static_events == []
  assert counter_state(actual) == counter_state(reference)
