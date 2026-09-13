import json
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from openpilot.system.athena import athenad


class TestLogForwarding(unittest.TestCase):
  def test_logs_preserve_comma_payload_and_priority(self):
    for contents in ("", "test log\n", "日志\n" * 20000):
      with self.subTest(size=len(contents)), tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "log.0001"
        path.write_text(contents)
        send_queue = queue.PriorityQueue()
        with mock.patch.object(athenad, "send_queue", send_queue):
          athenad.add_log_to_queue(path, path.name)
        if not contents:
          self.assertTrue(send_queue.empty())
          continue
        priority, _, payload = send_queue.get_nowait()
        self.assertEqual(priority, athenad.SEND_PRIORITY_LOW)
        self.assertEqual(json.loads(payload), {
          "method": "forwardLogs", "params": {"logs": contents}, "jsonrpc": "2.0", "id": path.name,
        })
        self.assertTrue(send_queue.empty())
        self.assertEqual(path.read_text(), contents)

  def test_statistics_preserve_comma_payload_and_file_lifecycle(self):
    contents = "device,started=true temperature=45.0\n" * 2000
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "stats.0001"
      path.write_text(contents)
      partial = Path(directory) / f"{tempfile.gettempprefix()}-partial"
      partial.write_text("unfinished")
      end_event = threading.Event()
      send_queue = queue.PriorityQueue()
      with mock.patch.object(athenad, "send_queue", send_queue), \
           mock.patch.object(athenad.time, "monotonic", return_value=11.0), \
           mock.patch.object(athenad.time, "sleep", side_effect=lambda _: end_event.set()):
        athenad.stat_handler(end_event, directory)
      priority, _, payload = send_queue.get_nowait()
      self.assertEqual(priority, athenad.SEND_PRIORITY_LOW)
      self.assertEqual(json.loads(payload), {
        "method": "storeStats", "params": {"stats": contents}, "jsonrpc": "2.0", "id": path.name,
      })
      self.assertTrue(send_queue.empty())
      self.assertFalse(path.exists())
      self.assertEqual(partial.read_text(), "unfinished")

  def test_statistics_are_not_deleted_if_enqueue_fails(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "stats.0001"
      path.write_text("pending statistics")
      end_event = threading.Event()
      with mock.patch.object(athenad, "send_queue_push", side_effect=queue.Full), \
           mock.patch.object(athenad.time, "monotonic", return_value=11.0), \
           mock.patch.object(athenad.time, "sleep", side_effect=lambda _: end_event.set()):
        athenad.stat_handler(end_event, directory)
      self.assertEqual(path.read_text(), "pending statistics")
