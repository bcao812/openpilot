import pytest

from openpilot.tools.longitudinal.replay_pedal_switches import replay_window, switch_metrics


def test_legacy_positional_experiment_switch_is_rejected():
  with pytest.raises(TypeError):
    replay_window(None, {}, True)


def pedal_rows(sides, interval=0.02, offset=0.0):
  return [{'t': offset + index * interval, 'automatic': True, 'can_brake': side == -1,
           'can_gas': -30000 if side == -1 else 7 if side == 1 else 0} for index, side in enumerate(sides)]


def test_native_dwell_and_short_pulses():
  rows = pedal_rows([1] * 10 + [-1] * 3 + [1] * 10 + [-1] * 10 + [1] * 10)
  assert switch_metrics(rows)['debounced'] == 2
  assert switch_metrics(rows)['raw'] == 4


@pytest.mark.parametrize('boundary', ['override', 'gap', 'neutral', 'conflict'])
def test_never_join_across_ineligible_boundaries(boundary):
  rows = pedal_rows([1] * 10 + [0] * 10 + [-1] * 10)
  for row in rows[10:20]:
    if boundary == 'override':
      row['automatic'] = False
    elif boundary == 'conflict':
      row.update(can_brake=True, can_gas=7)
  if boundary == 'gap':
    rows = rows[:10] + rows[20:]
  assert switch_metrics(rows)['debounced'] == 0
  assert switch_metrics(rows)['raw'] == 0


def test_short_neutral_can_bridge_confirmed_sides():
  rows = pedal_rows([1] * 10 + [0] * 3 + [-1] * 10)
  assert switch_metrics(rows)['debounced'] == 1


def test_neutral_does_not_manufacture_phase_dwell():
  rows = pedal_rows([1] * 10 + [-1] * 3 + [0] * 3 + [-1] * 3)
  assert switch_metrics(rows)['debounced'] == 0


def test_empty_and_partial_final_phase():
  assert switch_metrics([])['debounced'] == 0
  assert switch_metrics(pedal_rows([1] * 10 + [-1] * 7))['debounced'] == 0
