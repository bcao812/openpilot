"""Compatibility of radar control's numeric and native-message hot paths."""
from dataclasses import asdict
from decimal import Decimal
from enum import IntEnum
from fractions import Fraction
import math
from numbers import Real

import numpy as np
import pytest

from openpilot.cereal import custom, log
from openpilot.selfdrive.car import honda_radar_control as control
from openpilot.selfdrive.car.tests.test_honda_radar_control import binding, radar, scene, vision


def reference_read(obj, name, default=None):
  return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def reference_finite(*values):
  return all(isinstance(v, Real) and not isinstance(v, bool) and math.isfinite(v) for v in values)


class Integer(int):
  pass


class Float(float):
  pass


class Code(IntEnum):
  ZERO = 0
  ONE = 1


@pytest.mark.parametrize('values', [
  (), (0,), (1, -1, 2**63), (0.0, -0.0, 1.25), (math.inf,), (-math.inf,), (math.nan,),
  (True,), (False,), (1.0, False, 2.0), (None,), ('1.0',), (1 + 0j,), (Decimal('1.0'),),
  (Fraction(1, 3),), (Integer(1), Float(-0.0), Code.ONE), (Float(math.nan),),
  (np.int32(2), np.int64(3), np.float32(1.25), np.float64(-0.0)), (np.bool_(False),),
  (10**1000,), (False, 10**1000), (math.nan, 10**1000),
])
def test_finite_preserves_numeric_types_short_circuit_and_conversion_errors(values):
  try:
    expected = reference_finite(*values)
  except Exception as error:
    with pytest.raises(type(error)) as actual:
      control._finite(*values)
    assert str(actual.value) == str(error)
  else:
    assert control._finite(*values) is expected


def canonical(value):
  if isinstance(value, float):
    return ('float', value.hex())
  if isinstance(value, dict):
    return {key: canonical(item) for key, item in value.items()}
  if isinstance(value, (list, tuple)):
    return type(value)(canonical(item) for item in value)
  return value


@pytest.mark.parametrize('reader', [False, True], ids=['builder', 'reader'])
def test_native_read_preserves_scalars_defaults_properties_and_errors(reader):
  values = scene(leads=[vision()], bindings=[binding()])
  packet = custom.HondaRadarObservations.new_message(**values['observation'])
  packet.fusion.tracks[0].yRel = -0.0
  root = packet.as_reader() if reader else packet
  track = root.fusion.tracks[0]
  groups = ((root, ('schemaVersion', 'sessionId', 'sequence', 'active', 'reason')),
            (root.fusion, ('version', 'algorithmVersion', 'enabled', 'modelTimestampEof')),
            (track, control._TRACK_FIELDS),
            (track.nativeMotion, control._NATIVE_MOTION_FIELDS),
            (root.fusion.leads[0], control._BINDING_FIELDS))
  marker = object()
  for target, names in groups:
    for name in names:
      actual, expected = control._read(target, name, marker), getattr(target, name, marker)
      assert type(actual) is type(expected)
      if hasattr(expected, 'to_dict'):
        assert canonical(actual.to_dict()) == canonical(expected.to_dict())
      else:
        assert canonical(actual) == canonical(expected)
    assert control._read(target, 'notADeclaredField', marker) is marker
    assert control._read(target, 'schema') == target.schema
    assert callable(control._read(target, 'to_dict'))
    for name in (None, 1, []):
      with pytest.raises(TypeError) as actual:
        control._read(target, name, marker)
      with pytest.raises(TypeError) as expected:
        getattr(target, name, marker)
      assert str(actual.value) == str(expected.value)
  # Both the builder and its reader observe the changed value immediately.
  # No publication/field value may be retained by the faster lookup path.
  packet.fusion.tracks[0].dRel = 31.0
  assert control._read(track, 'dRel') == 31.0


def test_read_preserves_ordinary_objects_dict_subclasses_and_string_subclasses():
  class Ordinary:
    value = 4

    def __getattr__(self, name):
      if name == 'dynamic':
        return self.value + 1
      if name == 'broken':
        raise RuntimeError('ordinary accessor failed')
      raise AttributeError(name)

  class Mapping(dict):
    def get(self, name, default=None):
      return ('custom-get', name, super().get(name, default))

  class Name(str):
    pass

  marker = object()
  target = Ordinary()
  for name in ('value', 'dynamic', 'missing', Name('value')):
    assert control._read(target, name, marker) == reference_read(target, name, marker)
  target.value = 10
  assert control._read(target, 'dynamic') == 11
  with pytest.raises(RuntimeError, match='ordinary accessor failed'):
    control._read(target, 'broken', marker)
  for mapping in ({'value': 3, 1: 4}, Mapping(value=3)):
    for name in ('value', 'missing', 1):
      assert control._read(mapping, name, marker) == reference_read(mapping, name, marker)
  packet = custom.HondaRadarObservations.new_message(schemaVersion=1)
  for name in (Name('schemaVersion'), Name('missing')):
    assert control._read(packet, name, marker) == reference_read(packet, name, marker)


@pytest.mark.parametrize('reader', [False, True], ids=['builder', 'reader'])
@pytest.mark.parametrize('track_count', [0, 1, control.MAX_TRACKS])
def test_native_control_sequence_matches_original_helpers(monkeypatch, reader, track_count):
  actual, reference = control.HondaRadarControl(), control.HondaRadarControl()
  for frame in range(10):
    visual = vision(distance=20.0 - frame * .1)
    tracks = [radar(frame, identifier=index + 1, oem=index + 1, slot=index,
                    distance=20.0 + 4.0 * index - frame * .1) for index in range(track_count)]
    if frame == 3:
      for track in tracks:
        track['coasting'] = True
    elif frame == 4:
      tracks = []
    elif frame == 6 and tracks:
      tracks[0]['nativeMotion']['groundSpeed'] = math.nan
    values = scene(frame, tracks=tracks, leads=[visual], bindings=[binding()] if tracks else [], lanes=True)
    if frame >= 7:
      values['observation']['sessionId'] = 'new-session'
    if frame == 8:
      values['observation_valid'] = False
    packet = custom.HondaRadarObservations.new_message(**values['observation'])
    model = log.ModelDataV2.new_message(**values['model'])
    values['observation'] = packet.as_reader() if reader else packet
    values['model'] = model.as_reader() if reader else model
    before = canonical(packet.to_dict()), canonical(model.to_dict())
    with monkeypatch.context() as original:
      original.setattr(control, '_read', reference_read)
      original.setattr(control, '_finite', reference_finite)
      expected = reference.update(**values)
    result = actual.update(**values)
    assert canonical(asdict(result)) == canonical(asdict(expected)), frame
    assert canonical(actual._publication) == canonical(reference._publication), frame
    assert (canonical(packet.to_dict()), canonical(model.to_dict())) == before
