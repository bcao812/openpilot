"""Native logger smoke test; build loggerd first. No vehicle IPC or log directory."""
import os
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

from openpilot.cereal import messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.selfdrive.car.honda_radar_observer import HondaRadarObserver
from openpilot.selfdrive.car.honda_radar_observer_config import SERVICE
from openpilot.selfdrive.car.tests.test_honda_radar_observer import can_message, configuration, scan_frames
from openpilot.selfdrive.car.tests.test_honda_radar_shadow import SHADOW_SETTINGS, fusion_state
from openpilot.tools.lib.logreader import LogReader, decompress_stream


def synthetic_track_diagnostics(index):
  # Deliberately artificial wire-test values, not a valid fusion-state fixture.
  sample = index + 1
  wide = (1 << 40) + 100 * sample
  return {
    'active': True,
    'generation': wide + 1,
    'birthReason': f'wire-track-{sample}-birth',
    'lastInvalidReason': f'wire-track-{sample}-invalid',
    'lastInvalidMonoTime': wide + 2,
    'frameIndex': sample,
    'lifecycleRaw': 100 + sample,
    'previousPositionSeq': wide + 3,
    'previousPositionMonoTime': wide + 4,
    'previousDRel': 100 + sample + 0.125,
    'previousYRel': -(100 + sample + 0.375),
  }


def synthetic_lead_diagnostics(index):
  sample = index + 1
  wide = (1 << 42) + 100 * sample
  value = 1000 + 100 * sample
  return {
    'identityState': f'wire-lead-{sample}-state',
    'identityReason': f'wire-lead-{sample}-reason',
    'candidateTrackId': wide + 1,
    'bindingStartMonoTime': wide + 2,
    'bindingSamples': 200 + sample,
    'recoverySamples': 300 + sample,
    'independentPair': True,
    'residualX': value + 0.125,
    'residualY': value + 0.25,
    'residualV': value + 0.375,
    'residualDeltaX': value + 0.5,
    'residualDeltaY': value + 0.625,
    'bindingAnchorX': value + 0.75,
    'bindingAnchorY': value + 0.875,
    'rejectionMonoTime': wide + 3,
    'gateX': value + 1.125,
    'gateY': value + 1.25,
    'visionXStd': value + 1.375,
    'visionYStd': value + 1.5,
    'visionVStd': value + 1.625,
    'blend': value + 1.75,
    'coastWeight': value + 1.875,
    'velocityGate': value + 2.125,
  }


class TestHondaRadarLogger(unittest.TestCase):
  def test_native_logger_records_radar_and_vision_without_radar_qlog_duplication(self):
    # LOG_ROOT must be explicit: on a C3, OpenpilotPrefix alone does not move the
    # road-log directory. The private params path is checked before loggerd runs.
    with TemporaryDirectory(prefix='honda-radar-logger-') as directory, \
         patch.dict(os.environ, {'LOG_ROOT': directory}), OpenpilotPrefix() as prefix:
      self.assertIn(prefix.prefix, Params().get_param_path())
      pm = messaging.PubMaster([SERVICE, 'modelV2'])
      logger_dir = Path(BASEDIR) / 'openpilot/system/loggerd'
      process = subprocess.Popen([str(logger_dir / 'loggerd')], cwd=logger_dir,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=os.environ.copy())
      try:
        for service in (SERVICE, 'modelV2'):
          self.assertTrue(pm.wait_for_readers_to_update(service, timeout=5), f'loggerd did not subscribe to {service}')
        now = time.monotonic_ns()
        frames = [frame for slot in range(16) for frame in scan_frames(slot=slot)]
        sm = fusion_state(now)
        observation = HondaRadarObserver(*configuration(), session_id='native-logger').update(
          [can_message(now, frames)], sm, SHADOW_SETTINGS, now + 1, sm)
        self.assertTrue(observation.hondaRadarObservations.fusion.active)
        self.assertEqual(len(observation.hondaRadarObservations.fusion.tracks), 16)
        self.assertEqual(len(observation.hondaRadarObservations.fusion.leads), 3)
        # Exercise enlarged inline-composite lists and fields unknown to older
        # native loggers. Distinct non-default numeric/Text values catch lost or
        # mis-strided elements; True is the only non-default Bool value.
        # These replacements test wire preservation, not real fusion behavior.
        diagnostics = {}
        for name, fixture in (('tracks', synthetic_track_diagnostics), ('leads', synthetic_lead_diagnostics)):
          diagnostics[name] = []
          for index, item in enumerate(getattr(observation.hondaRadarObservations.fusion, name)):
            fields = fixture(index)
            for field, value in fields.items():
              setattr(item, field, value)
            diagnostics[name].append(fields)
        expected = observation.hondaRadarObservations.to_dict()
        expected_wire = observation.to_bytes()
        model = messaging.new_message('modelV2', valid=True)
        model.modelV2 = sm['modelV2']
        model.modelV2.frameId = 712
        pm.send(SERVICE, expected_wire)
        pm.send('modelV2', model)
        for service in (SERVICE, 'modelV2'):
          self.assertTrue(pm.wait_for_readers_to_update(service, timeout=5))
      finally:
        process.send_signal(signal.SIGINT)
        output, _ = process.communicate(timeout=10)
      self.assertEqual(process.returncode, 0, output.decode(errors='replace'))
      rlogs = list(Path(directory).glob('*/rlog.zst'))
      qlogs = list(Path(directory).glob('*/qlog.zst'))
      self.assertEqual((len(rlogs), len(qlogs)), (1, 1))
      # Compare the bytes actually sent through native IPC with the decompressed
      # rlog, without rebuilding a reader (which could change Cap'n Proto layout).
      self.assertEqual(decompress_stream(rlogs[0].read_bytes()).count(expected_wire), 1,
                       'loggerd did not preserve the complete radar event wire bytes exactly once')
      messages = list(LogReader(str(rlogs[0])))
      radar = [event for event in messages if event.which() == SERVICE]
      self.assertEqual(len(radar), 1)
      self.assertTrue(radar[0].valid)
      self.assertEqual(radar[0].hondaRadarObservations.to_dict(), expected)
      for name, expected_items in diagnostics.items():
        actual_items = getattr(radar[0].hondaRadarObservations.fusion, name)
        self.assertEqual(len(actual_items), len(expected_items))
        for index, fields in enumerate(expected_items):
          for field, value in fields.items():
            with self.subTest(fusion_list=name, index=index, field=field):
              self.assertEqual(getattr(actual_items[index], field), value)
      self.assertTrue(any(event.which() == 'modelV2' and event.modelV2.frameId == 712 for event in messages))
      self.assertFalse(any(event.which() == SERVICE for event in LogReader(str(qlogs[0]))))
      del pm


if __name__ == '__main__':
  unittest.main()
