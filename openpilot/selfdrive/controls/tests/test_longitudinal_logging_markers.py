from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from opendbc.car.honda.values import CAR
from openpilot.common import longitudinal_logging
from openpilot.selfdrive.car.actuation_observation import ActuationObservation
from openpilot.selfdrive.controls.lib.control_observation import ControlObservation
from openpilot.selfdrive.controls.lib.longitudinal_observation import PlannerObservation
from openpilot.selfdrive.selfdrived.longitudinal_observation import SelfdriveObservation


FACTORIES = (
  ("planner", PlannerObservation),
  ("controlsd", ControlObservation),
  ("card", ActuationObservation),
  ("selfdrived", SelfdriveObservation),
)


@pytest.fixture
def marker_paths(tmp_path, monkeypatch):
  markers = {producer: tmp_path / Path(path).name for producer, path in longitudinal_logging.LOGGING_MARKERS.items()}
  monkeypatch.setattr(longitudinal_logging, "LOGGING_MARKERS", markers)
  return markers


def supported_configuration():
  return (SimpleNamespace(carFingerprint=CAR.HONDA_ACCORD, openpilotLongitudinalControl=True, passive=False),
          SimpleNamespace(enableGasInterceptor=False))


@pytest.mark.parametrize("producer,factory", FACTORIES)
def test_missing_marker_skips_observer_construction(producer, factory, marker_paths):
  params, params_sp = supported_configuration()
  with patch.object(factory, "__init__", return_value=None) as constructor:
    assert factory.create(params, params_sp) is None
  constructor.assert_not_called()
  assert not any(path.exists() for path in marker_paths.values())


@pytest.mark.parametrize("producer,factory", FACTORIES)
@pytest.mark.parametrize("configuration", ["missing", "other_car", "stock_long", "interceptor"])
def test_present_marker_does_not_expand_vehicle_scope(producer, factory, configuration, marker_paths):
  params, params_sp = supported_configuration()
  marker_paths[producer].touch()
  if configuration == "missing":
    params, params_sp = None, None
  elif configuration == "other_car":
    params.carFingerprint = CAR.HONDA_CIVIC_BOSCH
  elif configuration == "stock_long":
    params.openpilotLongitudinalControl = False
  else:
    params_sp.enableGasInterceptor = True
  with patch.object(factory, "__init__", return_value=None) as constructor:
    assert factory.create(params, params_sp) is None
  constructor.assert_not_called()


@pytest.mark.parametrize("producer,factory", FACTORIES)
def test_marker_changes_apply_only_to_new_observers(producer, factory, marker_paths):
  params, params_sp = supported_configuration()
  marker = marker_paths[producer]
  assert factory.create(params, params_sp) is None
  marker.touch()
  source = {"missing_source_files": [], "imported_native_module_files": {}, "missing_explicit_native_library_files": []}
  real_is_file = Path.is_file
  with patch.object(longitudinal_logging, "source_manifest", return_value=source):
    with patch.object(Path, "is_file", autospec=True, side_effect=real_is_file) as marker_reads:
      observer = factory.create(params, params_sp)
    marker_reads.assert_called_once_with(marker)
    assert observer is not None and observer.recorder.enabled
    observer.recorder.clock = lambda: 100
    marker.unlink()

    with patch.object(Path, "is_file", side_effect=AssertionError("running observer polled marker")) as runtime_reads:
      for sequence in range(1, 4):
        sample = observer.recorder.finish(observer.recorder.begin(sequence))
        context = observer.recorder.context(sample)
        assert context["version"] == 1 and context["producer"] == producer
        assert context["sequence"] == sequence
    runtime_reads.assert_not_called()
    assert observer.recorder.enabled
    assert factory.create(params, params_sp) is None

    marker.touch()
    restarted = factory.create(params, params_sp)
    assert restarted is not None and restarted is not observer
    assert restarted.recorder.enabled and restarted.recorder.sequence == 0
    assert observer.recorder.sequence == 3
