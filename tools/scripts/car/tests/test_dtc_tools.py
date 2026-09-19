import builtins
import importlib
import json
import sys
from subprocess import CalledProcessError
from types import SimpleNamespace

import pytest

from opendbc.car import uds
from tools.scripts.car import clear_dtc as clear
from tools.scripts.car import read_dtc_status as reader


TARGET = 0xD12345
OTHER = 0xC10001


def dtc_response(*records, availability=0xFF):
  return b"\x59\x02" + bytes([availability]) + b"".join(code.to_bytes(3, "big") + bytes([status]) for code, status in records)


@pytest.fixture
def transport(monkeypatch):
  state = SimpleNamespace(requests=[], responses=[], safety=[], opened=0, closed=0, on_send=None, reset_error=False)

  class FakePanda:
    def __init__(self):
      state.opened += 1

    def can_send(self, *_args, **_kwargs):
      raise AssertionError("real CAN transport must not be used")

    def can_recv(self):
      raise AssertionError("real CAN transport must not be used")

    def set_safety_mode(self, mode, param=0):
      state.safety.append((mode, param))
      if mode == "noOutput" and state.reset_error:
        raise OSError("cannot restore safety mode")

    def close(self):
      state.closed += 1

  class FakeIsoTp:
    def __init__(self, *_args, **_kwargs):
      pass

    def send(self, request):
      state.requests.append(request)
      if state.on_send:
        state.on_send(request)

    def recv(self, _timeout):
      assert state.responses, "unexpected diagnostic request"
      response = state.responses.pop(0)
      if isinstance(response, Exception):
        raise response
      return response, True

  monkeypatch.setitem(sys.modules, "panda", SimpleNamespace(Panda=FakePanda))
  monkeypatch.setitem(sys.modules, "opendbc.car.structs",
                      SimpleNamespace(CarParams=SimpleNamespace(SafetyModel=SimpleNamespace(elm327="elm327", noOutput="noOutput"))))
  monkeypatch.setattr(reader, "require_pandad_stopped", lambda: None)
  monkeypatch.setattr(uds, "IsoTpMessage", FakeIsoTp)
  yield state
  assert state.closed == state.opened
  if state.opened:
    assert state.safety[-1] == ("noOutput", 0)


def run_clear(transport, tmp_path, responses, *, dtc="0xD12345"):
  transport.responses = list(responses)
  output = tmp_path / "clear.json"
  result = clear.main(["0x18DA28F1", "--bus", "1", "--no-obd-multiplexing", "--dtc", dtc, "--output", str(output)])
  return result, json.loads(output.read_text())


def test_import_has_no_device_or_argument_side_effects(monkeypatch):
  original_import = builtins.__import__

  def guarded_import(name, *args, **kwargs):
    assert name != "panda", "importing a diagnostic script must not import Panda"
    return original_import(name, *args, **kwargs)

  monkeypatch.setattr(builtins, "__import__", guarded_import)
  monkeypatch.setattr(sys, "argv", ["script", "--invalid-argument"])
  importlib.reload(reader)
  importlib.reload(clear)


@pytest.mark.parametrize("args", [
  [],
  ["0x18DA28F1"],
  ["0x7DF", "--dtc", "0xD12345", "--output", "/unused"],
  ["0x18DB00F1", "--dtc", "0xD12345", "--output", "/unused"],
  ["0x18DB33F1", "--dtc", "0xD12345", "--output", "/unused"],
  ["0x18DA28F2", "--dtc", "0xD12345", "--output", "/unused"],
  ["-1", "--dtc", "0xD12345", "--output", "/unused"],
  ["0x20000000", "--dtc", "0xD12345", "--output", "/unused"],
  ["0x18DA28F1", "--dtc", "0xFFFFFF", "--output", "/unused"],
  ["0x18DA28F1", "--dtc", "0x000000", "--output", "/unused"],
  ["0x18DA28F1", "--dtc", "0x1000000", "--output", "/unused"],
  ["0x18DA28F1", "--dtc", "0x1234", "--output", "/unused"],
  ["0x18DA28F1", "--dtc", "0x123_45", "--output", "/unused"],
  ["0x18DA28F1", "--dtc", "U0123", "--output", "/unused"],
  ["0x18DA28F1", "--dtc", "0xD12345", "--bus", "-1", "--output", "/unused"],
])
def test_cli_requires_physical_address_complete_single_code_and_receipt(transport, args):
  with pytest.raises(SystemExit) as error:
    clear.main(args)
  assert error.value.code != 0
  assert transport.opened == 0
  assert transport.requests == []


def test_existing_receipt_is_not_overwritten_or_device_opened(transport, tmp_path):
  output = tmp_path / "old.json"
  output.write_text("old diagnostic evidence")
  assert clear.main(["0x18DA28F1", "--dtc", "0xD12345", "--output", str(output)]) == 1
  assert output.read_text() == "old diagnostic evidence"
  assert transport.opened == 0


@pytest.mark.parametrize("fail_create", [True, False])
def test_no_clear_if_receipt_cannot_be_saved(transport, tmp_path, monkeypatch, fail_create):
  original_save = clear.save_report

  def fail_save(path, report, *, create=False):
    if create == fail_create:
      raise OSError("injected disk failure")
    original_save(path, report, create=create)

  monkeypatch.setattr(clear, "save_report", fail_save)
  transport.responses = [b"\x50\x03", dtc_response((TARGET, 0x28))]
  assert clear.main(["0x18DA28F1", "--dtc", "0xD12345", "--output", str(tmp_path / "clear.json")]) == 1
  assert not any(request[0] == 0x14 for request in transport.requests)


def test_no_clear_if_preclear_fsync_fails(transport, tmp_path, monkeypatch):
  count = 0
  original_fsync = reader.os.fsync

  def fail_second_sync(fd):
    nonlocal count
    count += 1
    if count == 2:
      raise OSError("injected preclear fsync failure")
    original_fsync(fd)

  monkeypatch.setattr(reader.os, "fsync", fail_second_sync)
  result, report = run_clear(transport, tmp_path, [b"\x50\x03", dtc_response((TARGET, 0x28))])
  assert result == 1
  assert "fsync" in report["errors"][0]["message"]
  assert not any(request[0] == 0x14 for request in transport.requests)


def test_final_receipt_failure_retains_before_evidence_and_fails(transport, tmp_path, monkeypatch, capsys):
  original_save = clear.save_report

  def fail_final_save(path, report, *, create=False):
    if report["outcome"] == "target_absent_others_unchanged":
      raise OSError("injected final receipt failure")
    original_save(path, report, create=create)

  monkeypatch.setattr(clear, "save_report", fail_final_save)
  result, saved_report = run_clear(transport, tmp_path, [b"\x50\x03", dtc_response((TARGET, 0x28)), b"\x54", dtc_response()])
  assert result == 1
  assert saved_report["outcome"] == "clear_unconfirmed"
  assert saved_report["before"]["dtcs"][0]["dtc"] == "0xd12345"
  assert '"outcome": "receipt_failed"' in capsys.readouterr().err


def test_exact_single_dtc_request_after_durable_before_record_and_verified_readback(transport, tmp_path, monkeypatch):
  synced = []
  original_fsync = reader.os.fsync

  def record_fsync(fd):
    original_fsync(fd)
    synced.append(fd)

  monkeypatch.setattr(reader.os, "fsync", record_fsync)
  before = dtc_response((TARGET, 0x28), (OTHER, 0x28))

  def check_evidence_before_clear(request):
    if request[0] == 0x14:
      saved = json.loads((tmp_path / "clear.json").read_text())
      assert saved["before"]["rawResponse"] == before.hex()
      assert saved["clearRequest"] == "14d12345"
      assert saved["outcome"] == "clear_unconfirmed"
      assert len(synced) >= 2

  transport.on_send = check_evidence_before_clear
  result, report = run_clear(transport, tmp_path, [b"\x50\x03", before, b"\x54", dtc_response((OTHER, 0x28))])
  assert result == 0
  assert transport.requests == [b"\x10\x03", b"\x19\x02\xff", b"\x14\xd1\x23\x45", b"\x19\x02\xff"]
  assert transport.safety == [("elm327", 1), ("noOutput", 0)]
  assert report["ecu"] == "0x18DA28F1"
  assert report["bus"] == 1
  assert report["clearResponse"] == "54"
  assert report["clearPositiveResponse"] is True
  assert report["outcome"] == "target_absent_others_unchanged"


@pytest.mark.parametrize("status,availability", [
  (0x29, 0xFF), (0x2A, 0xFF), (0x2C, 0xFF), (0x68, 0xFF), (0x38, 0xFF), (0x80, 0xFF),
  (0x28, 0xFE), (0x28, 0xFD), (0x28, 0xFB), (0x28, 0xBF), (0x00, 0x45),
])
def test_active_pending_untested_or_unknown_status_is_never_cleared(transport, tmp_path, status, availability):
  result, report = run_clear(transport, tmp_path, [b"\x50\x03", dtc_response((TARGET, status), availability=availability)])
  assert result == 1
  assert report["outcome"] == "clear_refused"
  assert transport.requests == [b"\x10\x03", b"\x19\x02\xff"]


@pytest.mark.parametrize("other_status", [0x29, 0x2A, 0x2C, 0x68, 0x38, 0x80])
def test_other_active_pending_or_untested_fault_also_prevents_clear(transport, tmp_path, other_status):
  result, report = run_clear(transport, tmp_path, [b"\x50\x03", dtc_response((TARGET, 0x28), (OTHER, other_status))])
  assert result == 1
  assert report["outcome"] == "clear_refused"
  assert "0xC10001" in report["errors"][0]["message"]
  assert not any(request[0] == 0x14 for request in transport.requests)


def test_selected_code_must_be_in_actual_ecu_response(transport, tmp_path):
  result, report = run_clear(transport, tmp_path, [b"\x50\x03", dtc_response((OTHER, 0x28))])
  assert result == 1
  assert "not reported" in report["errors"][0]["message"]
  assert not any(request[0] == 0x14 for request in transport.requests)


@pytest.mark.parametrize("response", [
  b"\x59\x02", b"\x59\x02\xff\x01", b"\x59\x02\xff\x01\x02\x03",
  dtc_response((TARGET, 0x28), (TARGET, 0x28)),
  dtc_response((TARGET, 0x28), availability=0x45),
])
def test_malformed_before_response_is_retained_and_never_cleared(transport, tmp_path, response):
  result, report = run_clear(transport, tmp_path, [b"\x50\x03", response])
  assert result == 1
  assert report["before"]["rawResponse"] == response.hex()
  assert not any(request[0] == 0x14 for request in transport.requests)


@pytest.mark.parametrize("stage", [0, 1, 2, 3])
@pytest.mark.parametrize("failure", ["nrc", "timeout", "truncated_nrc"])
def test_diagnostic_failures_are_errors_with_cleanup(transport, tmp_path, stage, failure):
  service = [0x10, 0x19, 0x14, 0x19][stage]
  failed_response = {"nrc": bytes([0x7F, service, 0x31]), "timeout": uds.MessageTimeoutError("no ECU response"),
                     "truncated_nrc": b"\x7f"}[failure]
  responses = [b"\x50\x03", dtc_response((TARGET, 0x28)), b"\x54", dtc_response()]
  responses[stage] = failed_response
  result, report = run_clear(transport, tmp_path, responses)
  assert result == 1
  assert report["errors"]
  assert report["clearPositiveResponse"] is (stage == 3)
  assert len(transport.requests) == stage + 1
  if failure == "nrc":
    assert report["errors"][0]["negativeResponseCode"] == 0x31
    assert report["errors"][0]["rawResponsePrefix"] == failed_response.hex()


@pytest.mark.parametrize("response", [b"\x54\x00", b"\x55", b""])
def test_clear_requires_well_formed_positive_response(transport, tmp_path, response):
  result, report = run_clear(transport, tmp_path, [b"\x50\x03", dtc_response((TARGET, 0x28)), response])
  assert result == 1
  assert report["clearPositiveResponse"] is False
  assert len(transport.requests) == 3


@pytest.mark.parametrize("after", [
  dtc_response((TARGET, 0x28), (OTHER, 0x28)),  # selected code remains
  dtc_response(),  # another code disappeared
  dtc_response((OTHER, 0x2A)),  # another code's status changed
  dtc_response((OTHER, 0x28), (0xC20001, 0x28)),  # a new code appeared
  dtc_response((OTHER, 0x28), availability=0x7F),  # availability changed
  b"\x59\x02\xff\x01",  # malformed readback
])
def test_unverified_readback_never_reports_success(transport, tmp_path, after):
  result, report = run_clear(transport, tmp_path, [b"\x50\x03", dtc_response((TARGET, 0x28), (OTHER, 0x28)), b"\x54", after])
  assert result == 1
  assert report["clearPositiveResponse"] is True
  assert report["outcome"] == "verification_failed"
  assert report["after"]["rawResponse"] == after.hex()


def test_panda_is_closed_even_if_restoring_nooutput_fails(transport, tmp_path):
  transport.reset_error = True
  result, report = run_clear(transport, tmp_path, [b"\x50\x03", dtc_response((TARGET, 0x28)), b"\x54", dtc_response()])
  assert result == 1
  assert report["outcome"] == "cleanup_failed"
  assert "restore safety" in report["errors"][0]["message"]


@pytest.mark.parametrize("reply,expected", [
  (dtc_response(), 0),
  (b"\x59\x02", 1),
  (uds.MessageTimeoutError("no ECU response"), 1),
  (b"\x7f\x19\x31", 1),
])
def test_read_distinguishes_empty_report_from_failed_or_malformed_response(transport, tmp_path, reply, expected):
  transport.responses = [b"\x50\x03", reply]
  output = tmp_path / "read.json"
  assert reader.main(["0x18DA28F1", "--output", str(output)]) == expected
  report = json.loads(output.read_text())
  if expected == 0:
    assert report["outcome"] == "read_complete"
    assert report["snapshot"]["dtcs"] == []
    assert report["snapshot"]["statusAvailabilityMask"] == 0xFF
  else:
    assert report["outcome"] == "read_failed"
    assert report["errors"]


def test_pandad_gate_does_not_kill_process_or_infer_vehicle_parked(monkeypatch):
  monkeypatch.setattr(reader, "check_output", lambda args: b"123\n")
  with pytest.raises(RuntimeError, match="vehicle must also be parked"):
    reader.require_pandad_stopped()

  def no_process(args):
    assert args == ["pidof", "pandad"]
    raise CalledProcessError(1, args)

  monkeypatch.setattr(reader, "check_output", no_process)
  reader.require_pandad_stopped()

  def failed_check(args):
    raise CalledProcessError(2, args)

  monkeypatch.setattr(reader, "check_output", failed_check)
  with pytest.raises(CalledProcessError):
    reader.require_pandad_stopped()
