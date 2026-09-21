import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from openpilot.system import logmessaged


class TestLogmessagedMessages(unittest.TestCase):
  def run_messages(self, packets):
    """Exercise the receive loop without a daemon, IPC sockets or file writes."""
    sock = Mock()
    sock.recv_multipart.side_effect = [*packets, KeyboardInterrupt()]
    context = Mock()
    context.socket.return_value = sock
    handler = Mock()
    pubs = {name: Mock() for name in ("logMessage", "errorLogMessage")}
    message = Mock()

    with patch.object(logmessaged.zmq.Context, "instance", return_value=context), \
         patch.object(logmessaged, "get_file_handler", return_value=handler), \
         patch.object(logmessaged.messaging, "pub_sock", side_effect=pubs.__getitem__), \
         patch.object(logmessaged.messaging, "new_message", return_value=message) as new_message:
      with self.assertRaises(KeyboardInterrupt):
        logmessaged.main()

    sock.close.assert_called_once()
    context.term.assert_called_once()
    handler.close.assert_called_once()
    return SimpleNamespace(handler=handler, pubs=pubs, new_message=new_message)

  def test_multibyte_byte_limit(self):
    # U+1F680 needs four UTF-8 bytes. Both sides of the limit are well below the
    # old character limit, which let oversized messages reach a native msgq assert.
    boundary = "\U0001f680" * (logmessaged.MAX_LOG_MESSAGE_BYTES // 4)
    for extra in ("", "a"):
      with self.subTest(extra_bytes=len(extra)):
        record = boundary + extra
        result = self.run_messages([[b"\x14" + record.encode("utf-8")]])
        result.handler.emit.assert_called_once_with(record)
        if extra:
          result.new_message.assert_not_called()
          result.pubs["logMessage"].send.assert_not_called()
        else:
          result.new_message.assert_called_once_with(None, valid=True, logMessage=record)
          result.pubs["logMessage"].send.assert_called_once()
        result.pubs["errorLogMessage"].send.assert_not_called()

  def test_invalid_packets_do_not_stop_later_records(self):
    result = self.run_messages([[], [b""], [b"\x14"], [b"\x14\xff"], [b"\x14", b"valid record"]])
    result.handler.emit.assert_called_once_with("valid record")
    result.new_message.assert_called_once_with(None, valid=True, logMessage="valid record")
    result.pubs["logMessage"].send.assert_called_once()

  def test_error_published_to_both_topics(self):
    result = self.run_messages([[b"\x28error record"]])
    self.assertEqual(result.new_message.call_args_list, [
      call(None, valid=True, logMessage="error record"),
      call(None, valid=True, errorLogMessage="error record"),
    ])
    for pub in result.pubs.values():
      pub.send.assert_called_once()

  def test_oversized_error_does_not_reach_either_topic(self):
    record = "\U0001f680" * (logmessaged.MAX_LOG_MESSAGE_BYTES // 4 + 1)
    result = self.run_messages([[b"\x28" + record.encode("utf-8")], [b"\x14next record"]])
    self.assertEqual(result.handler.emit.call_args_list, [call(record), call("next record")])
    result.new_message.assert_called_once_with(None, valid=True, logMessage="next record")
    result.pubs["logMessage"].send.assert_called_once()
    result.pubs["errorLogMessage"].send.assert_not_called()


if __name__ == "__main__":
  unittest.main()
