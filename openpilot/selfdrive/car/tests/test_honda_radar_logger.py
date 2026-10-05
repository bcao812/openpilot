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
from openpilot.tools.lib.logreader import LogReader


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
        expected = observation.hondaRadarObservations.to_dict()
        model = messaging.new_message('modelV2', valid=True)
        model.modelV2 = sm['modelV2']
        model.modelV2.frameId = 712
        pm.send(SERVICE, observation)
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
      messages = list(LogReader(str(rlogs[0])))
      radar = [event for event in messages if event.which() == SERVICE]
      self.assertEqual(len(radar), 1)
      self.assertTrue(radar[0].valid)
      self.assertEqual(radar[0].hondaRadarObservations.to_dict(), expected)
      self.assertTrue(any(event.which() == 'modelV2' and event.modelV2.frameId == 712 for event in messages))
      self.assertFalse(any(event.which() == SERVICE for event in LogReader(str(qlogs[0]))))
      del pm


if __name__ == '__main__':
  unittest.main()
