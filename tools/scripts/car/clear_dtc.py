#!/usr/bin/env python3
"""Attempt selected-DTC recovery while parked with openpilot stopped; never run automatically.

ECU-specific groupOfDTC behavior is unverified. Readback cannot undo or rule out collateral effects.
"""
import argparse
import json
import sys
from pathlib import Path

from opendbc.car.carlog import carlog
from opendbc.car.uds import SESSION_TYPE, SERVICE_TYPE, DTC_STATUS_MASK_TYPE as Status
from tools.scripts.car.read_dtc_status import (add_connection_arguments, add_error, diagnostic_client,
                                               new_report, physical_address, read_status, save_report)


def single_dtc(value: str) -> int:
  if len(value) != 8 or not value.lower().startswith("0x") or any(char not in "0123456789abcdefABCDEF" for char in value[2:]):
    raise argparse.ArgumentTypeError("use the full three-byte DTC from read_dtc_status, e.g. 0xD12345")
  code = int(value, 16)
  if not 0 < code < 0xFFFFFF:
    raise argparse.ArgumentTypeError("DTC must be 0x000001..0xFFFFFE; group/all-DTC requests are forbidden")
  return code


def require_inactive(snapshot, dtc):
  # Single-code groupOfDTC behavior is unverified on the vehicle. Refuse if any
  # reported fault is active/untested, even when the selected fault is historical.
  records = {int(record["dtc"], 16): record["status"] for record in snapshot["dtcs"]}
  if dtc not in records:
    raise ValueError("selected DTC was not reported by this ECU; nothing will be cleared")
  required = Status.TEST_FAILED | Status.TEST_FAILED_THIS_OPERATION_CYCLE | Status.PENDING_DTC | Status.TEST_NOT_COMPLETED_THIS_OPERATION_CYCLE
  if snapshot["statusAvailabilityMask"] & required != required:
    raise ValueError("ECU lacks status bits needed to verify reported DTCs are inactive and tested this cycle")
  unsafe = (Status.TEST_FAILED | Status.TEST_FAILED_THIS_OPERATION_CYCLE | Status.PENDING_DTC |
            Status.TEST_NOT_COMPLETED_THIS_OPERATION_CYCLE | Status.TEST_NOT_COMPLETED_SINCE_LAST_CLEAR)
  for code, status in records.items():
    if status & unsafe:
      raise ValueError(f"DTC 0x{code:06X} is active, pending, failed this cycle, or untested; repair/diagnose the cause first")
    if not status & (Status.CONFIRMED_DTC | Status.TEST_FAILED_SINCE_LAST_CLEAR):
      raise ValueError(f"DTC 0x{code:06X} has no confirmed/history status; cannot classify it as an inactive stored fault")


def clear_and_verify(client, dtc, output, report):
  physical_address(hex(client.tx_addr))
  single_dtc(f"0x{dtc:06x}")
  client.diagnostic_session_control(SESSION_TYPE.EXTENDED_DIAGNOSTIC)
  read_status(client, report["before"])
  require_inactive(report["before"], dtc)
  # Commit the raw before-state and request intent to disk before sending 0x14.
  # This is groupOfDTC in UDS: single-code support is ECU-specific. Never fall back to ALL.
  report["outcome"] = "clear_unconfirmed"
  report["clearRequest"] = "14" + dtc.to_bytes(3, "big").hex()
  save_report(output, report)
  # The public clear wrapper discards response bytes, preventing a length check/receipt.
  response = client._uds_request(SERVICE_TYPE.CLEAR_DIAGNOSTIC_INFORMATION, data=dtc.to_bytes(3, "big"))
  report["clearResponse"] = "54" + response.hex()
  if response:
    raise ValueError("malformed clear response: 0x54 must have no payload")
  report["clearPositiveResponse"] = True  # UdsClient rejects NRCs, wrong SIDs and timeouts.
  report["outcome"] = "verification_failed"
  read_status(client, report["after"])
  before = {record["dtc"]: record["status"] for record in report["before"]["dtcs"]}
  after = {record["dtc"]: record["status"] for record in report["after"]["dtcs"]}
  selected = f"0x{dtc:06x}"
  if selected in after:
    raise ValueError("selected DTC is still reported; clearing was not verified")
  before.pop(selected)
  if before != after or report["before"]["statusAvailabilityMask"] != report["after"]["statusAvailabilityMask"]:
    raise ValueError("other DTCs/status bits or status availability changed; verification conditions not met (cause undetermined)")
  report["outcome"] = "target_absent_others_unchanged"


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__, epilog=(
    "Single-DTC groupOfDTC support is ECU-specific. Status bits 0x47 must be supported; "
    "all reported DTCs must have reliable inactive/test-complete status. There is no all-DTC fallback."))
  add_connection_arguments(parser)
  parser.add_argument("--dtc", required=True, type=single_dtc, help="full raw DTC previously read from this ECU (0xXXXXXX)")
  parser.add_argument("--output", required=True, type=Path, help="new JSON receipt file; saved before any clear request")
  args = parser.parse_args(argv)
  if args.debug:
    carlog.setLevel("DEBUG")
  report = new_report(args)
  report.update({"selectedDtc": f"0x{args.dtc:06x}", "before": {}, "after": {}, "clearPositiveResponse": False})
  # Claim a new receipt before accessing hardware. Never overwrite an earlier diagnostic record.
  try:
    save_report(args.output, report, create=True)
  except OSError as error:
    print(f"Cannot create diagnostic receipt; no clear request sent: {error}", file=sys.stderr)
    return 1
  try:
    with diagnostic_client(args.addr, args.bus, args.no_obd_multiplexing) as client:
      clear_and_verify(client, args.dtc, args.output, report)
  except Exception as error:
    if report["outcome"] == "not_started":
      report["outcome"] = "clear_refused"
    elif report["outcome"] == "target_absent_others_unchanged":
      report["outcome"] = "cleanup_failed"
    add_error(report, error)
  try:
    save_report(args.output, report)
  except OSError as error:
    add_error(report, error)
    report["outcome"] = "receipt_failed"
  if report["outcome"] != "target_absent_others_unchanged":
    print("DTC clear not completed/verified; ECU may not support this single-code request. No all-DTC fallback was attempted.", file=sys.stderr)
    print(json.dumps(report, indent=2), file=sys.stderr)
    return 1
  print(f"{report['selectedDtc']} absent on immediate readback; other reported DTCs unchanged. Receipt: {args.output}")
  print("This does not establish that the underlying fault or dashboard warning is repaired.")
  return 0


if __name__ == "__main__":
  sys.exit(main())
