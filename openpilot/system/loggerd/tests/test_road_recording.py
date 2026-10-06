"""Native logger integration with synthetic, independently decodable video.

Private IPC, Params and LOG_ROOT are mandatory, including on comma hardware.
The real ten-second timeout is exercised; no production clock override is used.
"""
import base64
import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import patch

import ffmpeg

from openpilot.cereal import messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.tools.lib.logreader import LogReader


FIXTURE = json.loads((Path(__file__).parent / 'fixtures/road_recording.json').read_text())
ROAD = ('narrowRoadEncodeData', 'wideRoadEncodeData')
QCAM = 'qNarrowRoadEncodeData'
ENCODERS = (*ROAD, QCAM)


def decoded_hashes(path):
  command = [str(Path(ffmpeg.BIN_DIR) / 'ffmpeg'), '-nostdin', '-hide_banner', '-loglevel', 'error', '-threads', '1',
             '-i', str(path), '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough', '-enc_time_base', '1:1000000',
             '-f', 'rawvideo', '-']
  result = subprocess.run(command, capture_output=True, check=True)
  assert not result.stderr, result.stderr.decode(errors='replace')
  stride = 96 * 64 * 3 // 2
  assert len(result.stdout) % stride == 0
  return [hashlib.md5(result.stdout[i:i+stride]).hexdigest() for i in range(0, len(result.stdout), stride)]


class TestRoadRecording(unittest.TestCase):
  @contextmanager
  def recording(self, enabled):
    with TemporaryDirectory(prefix='road-recording-') as directory, \
         patch.dict(os.environ, {'LOG_ROOT': directory, 'LOGGERD_TEST': '1', 'LOGGERD_SEGMENT_LENGTH': '60'}), \
         OpenpilotPrefix() as prefix:
      self.assertIn(prefix.prefix, Params().get_param_path())
      params = Params()
      params.put_bool('RecordRoadCameras', enabled, block=True)
      params.put_bool('RecordAudio', False, block=True)
      self.pm = messaging.PubMaster([*ENCODERS, 'carControl', 'modelV2'])
      self.timestamps = {}
      logger_dir = Path(BASEDIR) / 'openpilot/system/loggerd'
      self.logger_output = Path(directory) / 'logger.stdout'
      with self.logger_output.open('wb') as output:
        self.process = subprocess.Popen([str(logger_dir / 'loggerd')], cwd=logger_dir, env=os.environ.copy(),
                                        stdout=output, stderr=subprocess.STDOUT)
        try:
          for service in (*ENCODERS, 'carControl', 'modelV2'):
            self.assertTrue(self.pm.wait_for_readers_to_update(service, timeout=5), service)
          yield Path(directory)
        finally:
          self.stop_logger()
          del self.pm

  def stop_logger(self):
    if self.process.poll() is None:
      self.process.send_signal(signal.SIGINT)
    self.assertEqual(self.process.wait(timeout=10), 0, self.logger_output.read_text())

  def control(self, long_active):
    msg = messaging.new_message('carControl', valid=True)
    msg.carControl.enabled = True
    msg.carControl.latActive = True
    msg.carControl.longActive = long_active
    self.pm.send('carControl', msg)
    self.assertTrue(self.pm.wait_for_readers_to_update('carControl', timeout=5))

  def frame(self, i, road=True):
    stamp = time.monotonic_ns()
    self.timestamps[i] = stamp
    for service in ((*ROAD, QCAM) if road else (QCAM,)):
      fixture_service = service if service == QCAM else ROAD[0]
      row = FIXTURE['frames'][fixture_service][i]
      data = base64.b64decode(row['data'])
      msg = messaging.new_message(service, valid=True)
      value = getattr(msg, service)
      value.width, value.height = row['width'], row['height']
      value.data = data
      value.header = base64.b64decode(row['header'])
      value.idx = dict(row['idx'], frameId=1000+i, encodeId=i, segmentNum=0, segmentId=i,
                       timestampSof=stamp, timestampEof=stamp, len=len(data))
      self.pm.send(service, msg)
    model = messaging.new_message('modelV2', valid=True)
    model.modelV2.frameId = 1000+i
    self.pm.send('modelV2', model)
    for service in ((*ROAD, QCAM) if road else (QCAM,)):
      self.assertTrue(self.pm.wait_for_readers_to_update(service, timeout=5))
    self.assertTrue(self.pm.wait_for_readers_to_update('modelV2', timeout=5))

  def verify(self, directory, frames, expected_road):
    segments = list(directory.glob('*--0'))
    self.assertEqual(len(segments), 1)
    segment = segments[0]
    events = list(LogReader(str(segment / 'rlog.zst')))
    self.assertEqual([e.modelV2.frameId for e in events if e.which() == 'modelV2'], [1000+i for i in frames])
    self.assertEqual(decoded_hashes(segment/'qcamera.ts'), [FIXTURE['frames'][QCAM][i]['md5'] for i in frames])
    for filename, service in (('fcamera.hevc', 'narrowRoadEncodeIdx'), ('ecamera.hevc', 'wideRoadEncodeIdx')):
      indices = [getattr(e, service).to_dict() for e in events if e.which() == service]
      self.assertEqual([v['frameId'] for v in indices], [1000+i for i in expected_road])
      self.assertEqual([v['segmentId'] for v in indices], list(range(len(expected_road))))
      self.assertEqual([v['segmentIdEncode'] for v in indices], list(range(len(expected_road))))
      self.assertEqual([v['segmentNum'] for v in indices], [0]*len(expected_road))
      for idx, frame in zip(indices, expected_road, strict=True):
        self.assertEqual((idx['encodeId'], idx['timestampSof'], idx['timestampEof']),
                         (frame, self.timestamps[frame], self.timestamps[frame]))
      path = segment / filename
      self.assertEqual(path.exists(), bool(expected_road))
      if expected_road:
        self.assertEqual(decoded_hashes(path), [FIXTURE['frames'][ROAD[0]][i]['md5'] for i in expected_road])

  def test_pause_tail_resume_preserves_video_and_compacts_only_saved_indices(self):
    with self.recording(True) as directory:
      self.control(False)
      for i in range(5):
        self.frame(i)
      self.control(True)
      for i in range(5, 25):
        self.frame(i)
      self.control(False)
      for i in range(25, 29):
        self.frame(i)
      deadline = time.monotonic() + 10.1
      while time.monotonic() < deadline:
        self.control(False)  # repeated inactive updates must not extend the tail
        time.sleep(0.1)
      # No HD packets arrive during the gap: resume still requires a new GOP.
      for i in range(29, 41):
        self.frame(i, road=False)
      self.control(True)
      for i in range(41, 66):
        self.frame(i)
      self.stop_logger()
      self.verify(directory, list(range(66)), list(range(20, 29)) + list(range(60, 66)))

  def test_opt_out_keeps_low_resolution_video_and_model_logs(self):
    with self.recording(False) as directory:
      self.control(True)
      for i in range(25):
        self.frame(i)
      self.stop_logger()
      self.verify(directory, list(range(25)), [])
