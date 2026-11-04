import os
from types import SimpleNamespace
from unittest import mock

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.system.manager import manager


PERSISTENT_METADATA = {"Version", "GitCommit", "GitCommitDate", "GitBranch", "GitRemote", "HardwareSerial"}


class TestStartupMetadata(OpenpilotTestCase):
  def metadata(self, **changes):
    source = dict(version="test-version", git_commit="test-commit", git_commit_date="test-date",
                  git_origin="test-origin", git_normalized_origin="test-origin", is_dirty=True)
    source.update(changes)
    return SimpleNamespace(channel="c3xl-dev", development_channel=True, tested_channel=False,
                           release_channel=False, release_sp_channel=False, openpilot=SimpleNamespace(**source))

  def initialize(self, params, metadata):
    with mock.patch.object(manager, "Params", return_value=params), \
         mock.patch.object(manager, "save_bootlog"), \
         mock.patch.object(manager, "get_build_metadata", return_value=metadata), \
         mock.patch.object(manager, "PC", True), \
         mock.patch.object(manager.HARDWARE, "get_serial", return_value="test-serial"), \
         mock.patch.object(manager, "register", return_value="test-device"), \
         mock.patch.object(manager.sentry, "init"), \
         mock.patch.object(manager.cloudlog, "bind_global"), \
         mock.patch.object(params, "put", wraps=params.put) as put:
      manager.manager_init()
    return [call for call in put.call_args_list if call.args[0] in PERSISTENT_METADATA]

  def test_stable_reboot_keeps_metadata_and_republishes_cleared_flags(self):
    params = Params()
    first = self.initialize(params, self.metadata())
    self.assertEqual({call.args[0] for call in first}, PERSISTENT_METADATA)
    self.assertTrue(all(call.kwargs.get("block") for call in first))
    self.assertEqual(self.initialize(params, self.metadata()), [])
    self.assertEqual(params.get("GitCommit"), "test-commit")
    self.assertEqual(params.get("HardwareSerial"), "test-serial")
    for name, value in (("IsDevelopmentBranch", True), ("IsTestedBranch", False),
                        ("IsReleaseBranch", False), ("IsReleaseSpBranch", False)):
      with self.subTest(name=name):
        self.assertIs(params.get(name), value)

  def test_new_commit_and_branch_are_published_synchronously(self):
    params = Params()
    self.initialize(params, self.metadata())
    changed = self.metadata(git_commit="new-commit", git_commit_date="new-date")
    changed.channel = "new-branch"
    writes = self.initialize(params, changed)
    self.assertEqual({call.args[0] for call in writes}, {"GitCommit", "GitCommitDate", "GitBranch"})
    self.assertTrue(all(call.kwargs.get("block") for call in writes))
    self.assertEqual(params.get("GitCommit"), "new-commit")
    self.assertEqual(params.get("GitBranch"), "new-branch")
    self.assertEqual(os.environ["GIT_COMMIT"], "new-commit")

  def test_missing_persistent_metadata_is_repaired(self):
    params = Params()
    self.initialize(params, self.metadata())
    params.remove("GitRemote")
    params.remove("HardwareSerial")
    writes = self.initialize(params, self.metadata())
    self.assertEqual({call.args[0] for call in writes}, {"GitRemote", "HardwareSerial"})
    self.assertEqual(params.get("GitRemote"), "test-origin")
    self.assertEqual(params.get("HardwareSerial"), "test-serial")
