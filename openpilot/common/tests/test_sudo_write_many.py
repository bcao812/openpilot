import io
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from openpilot.common import utils


class PermissionFiles:
  """File permissions and sudo effects without touching privileged host paths."""

  def __init__(self):
    self.values = {}
    self.writable = set()
    self.chmod_rejected = set()
    self.calls = []
    self.writes = []
    self.bulk_os_error = False
    self.root_write_failure = False

  def add(self, path, writable=False):
    self.values[path] = "unchanged"
    if writable:
      self.writable.add(path)

  @contextmanager
  def open(self, path, mode):
    assert mode == "w"
    if path not in self.values:
      raise FileNotFoundError(path)
    if path not in self.writable:
      raise PermissionError(path)
    with io.StringIO() as stream:
      yield stream
      value = stream.getvalue()
      self.values[path] = value
      self.writes.append((value, path))

  def run(self, command, **kwargs):
    self.calls.append((command, kwargs))
    if isinstance(command, list):
      assert command[:4] == ["sudo", "chmod", "a+w", "--"]
      assert not kwargs
      if self.bulk_os_error:
        raise FileNotFoundError("sudo")
      paths = command[4:]
    elif command.startswith("sudo chmod a+w "):
      assert kwargs == {"shell": True}
      paths = [command.removeprefix("sudo chmod a+w ")]
    else:
      assert kwargs == {"shell": True}
      value, path = command.removeprefix("sudo su -c 'echo ").removesuffix("'").split(" > ", 1)
      if not self.root_write_failure:
        self.values[path] = value + "\n"
        self.writes.append((value + "\n", path))
      return SimpleNamespace(returncode=int(self.root_write_failure))
    for path in paths:
      if path not in self.chmod_rejected:
        self.writable.add(path)
    return SimpleNamespace(returncode=int(any(path in self.chmod_rejected for path in paths)))


@pytest.fixture
def files(monkeypatch):
  result = PermissionFiles()
  monkeypatch.setattr(utils, "open", result.open, raising=False)
  monkeypatch.setattr(utils, "os", SimpleNamespace(
    path=SimpleNamespace(isfile=lambda path: path in result.values),
    access=lambda path, mode: path in result.writable,
    W_OK=2,
  ))
  monkeypatch.setattr(utils, "subprocess", SimpleNamespace(run=result.run))
  return result


def test_writable_files_need_no_subprocess(tmp_path, monkeypatch):
  paths = [str(tmp_path / name) for name in ("first", "second")]
  for path in paths:
    with open(path, "w") as stream:
      stream.write("old")
  run = Mock(side_effect=AssertionError("Writable files must not invoke sudo"))
  monkeypatch.setattr(utils, "subprocess", SimpleNamespace(run=run))
  utils.sudo_write_many([("first value", paths[0]), ("second value", paths[1])])
  for path, expected in zip(paths, ("first value", "second value"), strict=True):
    with open(path) as stream:
      assert stream.read() == expected
  run.assert_not_called()


def test_permissions_are_batched_but_writes_keep_order(files):
  for path in ("/first", "/already-writable", "/last"):
    files.add(path, writable=path == "/already-writable")
  writes = [("1", "/first"), ("2", "/already-writable"), ("3", "/last"), ("4", "/first")]
  utils.sudo_write_many(writes)
  assert files.writes == writes
  assert files.values["/first"] == "4"
  assert files.calls == [(["sudo", "chmod", "a+w", "--", "/first", "/last"], {})]


def test_partial_chmod_failure_retains_debugfs_fallback(files):
  for path in ("/first", "/debugfs", "/last"):
    files.add(path)
  files.chmod_rejected.add("/debugfs")
  utils.sudo_write_many([("1", "/first"), ("N", "/debugfs"), ("2", "/last")])
  assert files.writes == [("1", "/first"), ("N\n", "/debugfs"), ("2", "/last")]
  assert files.calls == [
    (["sudo", "chmod", "a+w", "--", "/first", "/debugfs", "/last"], {}),
    ("sudo chmod a+w /debugfs", {"shell": True}),
    ("sudo su -c 'echo N > /debugfs'", {"shell": True}),
  ]


def test_failed_debugfs_fallback_keeps_existing_best_effort_behavior(files):
  files.add("/debugfs")
  files.add("/last")
  files.chmod_rejected.add("/debugfs")
  files.root_write_failure = True
  utils.sudo_write_many([("N", "/debugfs"), ("2", "/last")])
  assert files.values["/debugfs"] == "unchanged"
  assert files.writes == [("2", "/last")]
  assert files.calls[-1] == ("sudo su -c 'echo N > /debugfs'", {"shell": True})


def test_missing_path_preserves_error_position_without_preparing_later_paths(files):
  files.add("/first", writable=True)
  files.add("/last")
  with pytest.raises(FileNotFoundError, match="/missing"):
    utils.sudo_write_many([("1", "/first"), ("2", "/missing"), ("3", "/last")])
  assert files.writes == [("1", "/first")]
  assert files.values["/last"] == "unchanged"
  assert "/last" not in files.writable
  assert files.calls == []


def test_permission_preparation_launch_error_falls_back_per_file(files):
  files.add("/first")
  files.add("/last")
  files.bulk_os_error = True
  writes = [("1", "/first"), ("2", "/last")]
  utils.sudo_write_many(writes)
  assert files.writes == writes
  assert files.calls == [
    (["sudo", "chmod", "a+w", "--", "/first", "/last"], {}),
    ("sudo chmod a+w /first", {"shell": True}),
    ("sudo chmod a+w /last", {"shell": True}),
  ]


def test_empty_and_single_write_keep_original_behavior(files):
  utils.sudo_write_many([])
  assert files.calls == []
  files.add("/only")
  utils.sudo_write_many([("1", "/only")])
  assert files.writes == [("1", "/only")]
  assert files.calls == [("sudo chmod a+w /only", {"shell": True})]
