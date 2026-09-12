import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import requests

from openpilot.sunnypilot.models import tinygrad_ref


class TestTinygradRef(unittest.TestCase):
  def test_vendored_ref_matches_committed_tree(self):
    metadata = json.loads(
      (Path(tinygrad_ref.BASEDIR) / "openpilot/sunnypilot/models/tinygrad_vendor.json").read_text()
    )
    self.assertEqual(tinygrad_ref.get_tinygrad_ref(), metadata["commit"])

  def test_vendored_tree_mismatch_is_not_reported_as_matching(self):
    with mock.patch.object(tinygrad_ref.subprocess, "check_output", return_value="0" * 40):
      self.assertIsNone(tinygrad_ref.get_tinygrad_ref())

  def test_missing_metadata_returns_none(self):
    with tempfile.TemporaryDirectory() as temporary_root, mock.patch.object(tinygrad_ref, "BASEDIR", temporary_root):
      self.assertIsNone(tinygrad_ref.get_tinygrad_ref())

  def test_invalid_metadata_returns_none(self):
    for contents in ("not json", "[]", "{}", '{"commit": 123, "tree": "invalid"}'):
      with self.subTest(contents=contents), tempfile.TemporaryDirectory() as temporary_root:
        metadata_path = Path(temporary_root) / "openpilot/sunnypilot/models/tinygrad_vendor.json"
        metadata_path.parent.mkdir(parents=True)
        metadata_path.write_text(contents)
        with mock.patch.object(tinygrad_ref, "BASEDIR", temporary_root):
          self.assertIsNone(tinygrad_ref.get_tinygrad_ref())

  def test_git_checkout_and_linked_worktree(self):
    revision = "1234567890abcdef1234567890abcdef12345678"
    for linked in (False, True):
      with self.subTest(linked=linked), tempfile.TemporaryDirectory() as temporary_root:
        repo_path = Path(temporary_root) / "tinygrad_repo"
        repo_path.mkdir()
        if linked:
          (repo_path / ".git").write_text("gitdir: /unused/worktree")
        else:
          (repo_path / ".git").mkdir()
        with mock.patch.object(tinygrad_ref, "BASEDIR", temporary_root), \
             mock.patch.object(tinygrad_ref.subprocess, "check_output", return_value=revision + "\n") as command:
          self.assertEqual(tinygrad_ref.get_tinygrad_ref(), revision)
          command.assert_called_once_with(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
          )

  def test_git_failure_returns_none(self):
    for error in (subprocess.CalledProcessError(128, "git"), subprocess.TimeoutExpired("git", 5)):
      with self.subTest(error=type(error).__name__), mock.patch.object(tinygrad_ref.subprocess, "check_output", side_effect=error):
        self.assertIsNone(tinygrad_ref.get_tinygrad_ref())


def fetch_tinygrad_ref():
  from openpilot.sunnypilot.models.fetcher import ModelFetcher

  response = requests.get(ModelFetcher.MODEL_URL, timeout=10)
  response.raise_for_status()
  json_data = response.json()
  return json_data.get("tinygrad_ref")


@unittest.skipUnless(os.environ.get("RUN_INTEGRATION_TESTS") == "1", "requires the live model catalogue")
class TestLiveTinygradRef(unittest.TestCase):
  def test_tinygrad_ref(self):
    current_ref = tinygrad_ref.get_tinygrad_ref()
    remote_ref = fetch_tinygrad_ref()
    assert remote_ref == current_ref, (
      f"""tinygrad_repo ref does not match remote tinygrad_ref of current compiled driving models json.
    Current: {current_ref}
    Remote: {remote_ref}
    Please run build-all workflow to update models."""
    )
    print("tinygrad_repo ref matches current compiled driving models json ref.")
