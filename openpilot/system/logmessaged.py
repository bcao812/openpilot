#!/usr/bin/env python3
import zmq
from typing import NoReturn

import openpilot.cereal.messaging as messaging
from openpilot.common.logging_extra import SwagLogFileFormatter
from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import get_file_handler


# Leave room for the cereal envelope below msgq's per-message queue limit.
MAX_LOG_MESSAGE_BYTES = 2 * 1024 * 1024


def main() -> NoReturn:
  log_handler = get_file_handler()
  log_handler.setFormatter(SwagLogFileFormatter(None))
  log_level = 20  # logging.INFO

  ctx = zmq.Context.instance()
  sock = ctx.socket(zmq.PULL)
  sock.bind(Paths.swaglog_ipc())

  # and we publish them
  log_message_sock = messaging.pub_sock('logMessage')
  error_log_message_sock = messaging.pub_sock('errorLogMessage')

  try:
    while True:
      dat = b''.join(sock.recv_multipart())
      if len(dat) <= 1:
        print("WARNING: empty log record")
        continue

      level = dat[0]
      try:
        record = dat[1:].decode("utf-8")
      except UnicodeDecodeError:
        print("WARNING: log record is not valid UTF-8")
        continue

      if level >= log_level:
        log_handler.emit(record)

      # Text is serialized as UTF-8; character count can understate its size by 4x.
      record_bytes = len(dat) - 1
      if record_bytes > MAX_LOG_MESSAGE_BYTES:
        print("WARNING: log too big to publish", record_bytes)
        print(record[:100])
        continue

      # then we publish them
      msg = messaging.new_message(None, valid=True, logMessage=record)
      log_message_sock.send(msg.to_bytes())

      if level >= 40:  # logging.ERROR
        msg = messaging.new_message(None, valid=True, errorLogMessage=record)
        error_log_message_sock.send(msg.to_bytes())
  finally:
    sock.close()
    ctx.term()

    # can hit this if interrupted during a rollover
    try:
      log_handler.close()
    except ValueError:
      pass

if __name__ == "__main__":
  main()
