#!/usr/bin/env python3
"""Manually read one ECU while parked with openpilot stopped; never run during a drive."""
import argparse
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from subprocess import check_output, CalledProcessError

from opendbc.car.carlog import carlog
from opendbc.car.uds import (UdsClient, SESSION_TYPE, DTC_REPORT_TYPE, DTC_STATUS_MASK_TYPE, FUNCTIONAL_ADDRS,
                             NegativeResponseError, get_dtc_num_as_str, get_dtc_status_names, get_rx_addr_for_tx_addr)


def physical_address(value: str) -> int:
  addr = int(value, 0)
  physical_29bit = (addr >> 16) == 0x18DA and (addr & 0xFF) == 0xF1
  if addr in FUNCTIONAL_ADDRS or not (0 <= addr <= 0x7F7 or physical_29bit):
    raise argparse.ArgumentTypeError("use one physical ECU address: 11-bit or 0x18DAxxF1; functional/broadcast addresses are forbidden")
  get_rx_addr_for_tx_addr(addr)
  return addr


def add_connection_arguments(parser):
  parser.add_argument("addr", type=physical_address, help="physical ECU request address, e.g. 0x18DA28F1")
  parser.add_argument("--bus", type=int, choices=range(4), default=0)
  parser.add_argument("--no-obd-multiplexing", action="store_true", help="ELM327 safety param 1 for connections without OBD multiplexing")
  parser.add_argument("--debug", action="store_true")


def require_pandad_stopped():
  try:
    check_output(["pidof", "pandad"])
  except CalledProcessError as error:
    if error.returncode == 1:
      return
    raise
  raise RuntimeError("pandad is running; stop openpilot before using this script (vehicle must also be parked)")


@contextmanager
def diagnostic_client(addr, bus, no_obd_multiplexing=False):
  physical_address(hex(addr))
  require_pandad_stopped()
  # Importing the script or asking for --help must not open a Panda.
  from panda import Panda
  from opendbc.car.structs import CarParams

  panda = Panda()
  try:
    panda.set_safety_mode(CarParams.SafetyModel.elm327, param=int(no_obd_multiplexing))
    yield UdsClient(panda, addr, bus=bus)
  finally:
    try:
      panda.set_safety_mode(CarParams.SafetyModel.noOutput)
    finally:
      panda.close()


def read_status(client, record):
  data = client.read_dtc_information(DTC_REPORT_TYPE.DTC_BY_STATUS_MASK, DTC_STATUS_MASK_TYPE.ALL)
  # UdsClient has checked and removed the positive-response SID and subfunction.
  record["rawResponse"] = "5902" + data.hex()
  if not data or (len(data) - 1) % 4:
    raise ValueError("malformed 0x19/0x02 response: expected statusAvailabilityMask followed by 4-byte DTC records")
  record["statusAvailabilityMask"] = data[0]
  record["statusAvailabilityNames"] = get_dtc_status_names(data[0])
  record["dtcs"] = []
  seen = set()
  for offset in range(1, len(data), 4):
    code, status = data[offset:offset + 3], data[offset + 3]
    if code in seen or status & ~data[0]:
      raise ValueError("malformed DTC response: duplicate DTC or status bits outside statusAvailabilityMask")
    seen.add(code)
    record["dtcs"].append({"dtc": "0x" + code.hex(), "displayCode": get_dtc_num_as_str(code),
                           "status": status, "statusNames": get_dtc_status_names(status)})


def new_report(args):
  return {"recordedAtUtc": datetime.now(UTC).isoformat(), "ecu": f"0x{args.addr:X}", "bus": args.bus,
          "noObdMultiplexing": args.no_obd_multiplexing, "outcome": "not_started", "errors": []}


def add_error(report, error):
  detail = {"type": type(error).__name__, "message": str(error)}
  if isinstance(error, NegativeResponseError):
    detail.update({"serviceId": error.service_id, "negativeResponseCode": error.error_code,
                   "rawResponsePrefix": bytes(value for value in (0x7F, error.service_id, error.error_code) if value >= 0).hex()})
  if error.__context__ is not None:
    detail["context"] = f"{type(error.__context__).__name__}: {error.__context__}"
  report["errors"].append(detail)


def save_report(path, report, *, create=False):
  """Do not overwrite old evidence; updates atomically retain the saved before-state on failure."""
  path = Path(path)
  if create:
    with path.open("x") as output:
      json.dump(report, output, indent=2)
      output.write("\n")
      output.flush()
      os.fsync(output.fileno())
    return
  temporary = None
  try:
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=f".{path.name}.", delete=False) as output:
      temporary = output.name
      json.dump(report, output, indent=2)
      output.write("\n")
      output.flush()
      os.fsync(output.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
      os.fsync(directory)
    finally:
      os.close(directory)
  finally:
    if temporary is not None and os.path.exists(temporary):
      os.unlink(temporary)


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  add_connection_arguments(parser)
  parser.add_argument("--output", type=Path, help="save JSON to a new file (existing files are never overwritten)")
  args = parser.parse_args(argv)
  if args.debug:
    carlog.setLevel("DEBUG")
  report = new_report(args)
  report["snapshot"] = {}
  try:
    with diagnostic_client(args.addr, args.bus, args.no_obd_multiplexing) as client:
      client.diagnostic_session_control(SESSION_TYPE.EXTENDED_DIAGNOSTIC)
      read_status(client, report["snapshot"])
    report["outcome"] = "read_complete"
  except Exception as error:
    report["outcome"] = "read_failed"
    add_error(report, error)
  if args.output is not None:
    try:
      save_report(args.output, report, create=True)
    except OSError as error:
      add_error(report, error)
      report["outcome"] = "receipt_failed"
  if report["outcome"] != "read_complete":
    print(json.dumps(report, indent=2), file=sys.stderr)
    return 1
  snapshot = report["snapshot"]
  print("status availability:", " ".join(snapshot["statusAvailabilityNames"]))
  for dtc in snapshot["dtcs"]:
    print(dtc["dtc"], dtc["displayCode"], " ".join(dtc["statusNames"]))
  if not snapshot["dtcs"]:
    print("No DTCs reported for the supported status bits; this does not prove all vehicle systems are fault-free.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
