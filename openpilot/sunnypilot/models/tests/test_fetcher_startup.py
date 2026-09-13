from pathlib import Path
from unittest import mock

from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.models.fetcher import ModelFetcher


class TestModelFetcherStartup(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()
    self.active_json = {
      "qcom": ModelFetcher.MODEL_URL,
      "chestnut": ModelFetcher.MODEL_URL_CHESTNUT,
    }

  def test_missing_active_json_is_written_synchronously(self):
    self.params.remove("ModelManager_ActiveJson")
    with mock.patch.object(self.params, "put", wraps=self.params.put) as put:
      ModelFetcher(self.params)
    put.assert_called_once_with("ModelManager_ActiveJson", self.active_json, block=True)
    self.assertEqual(self.params.get("ModelManager_ActiveJson"), self.active_json)

  def test_unchanged_active_json_is_not_rewritten(self):
    self.params.put("ModelManager_ActiveJson", self.active_json, block=True)
    stored = Path(self.params.get_param_path("ModelManager_ActiveJson"))
    before = stored.stat()
    with mock.patch.object(self.params, "put", wraps=self.params.put) as put:
      fetcher = ModelFetcher(self.params)
    put.assert_not_called()
    self.assertEqual(stored.stat().st_ino, before.st_ino)
    self.assertEqual(stored.stat().st_mtime_ns, before.st_mtime_ns)
    self.assertEqual(set(fetcher.model_caches), set(self.active_json))
    self.assertIs(fetcher.params, self.params)

  def test_mapping_order_does_not_trigger_a_write(self):
    self.params.put("ModelManager_ActiveJson", dict(reversed(tuple(self.active_json.items()))), block=True)
    with mock.patch.object(self.params, "put", wraps=self.params.put) as put:
      ModelFetcher(self.params)
    put.assert_not_called()

  def test_changed_sources_are_updated_synchronously(self):
    for source in self.active_json:
      with self.subTest(source=source):
        stale = dict(self.active_json)
        stale[source] = "https://fixture.invalid/old-catalogue.json"
        self.params.put("ModelManager_ActiveJson", stale, block=True)
        with mock.patch.object(self.params, "put", wraps=self.params.put) as put:
          ModelFetcher(self.params)
        put.assert_called_once_with("ModelManager_ActiveJson", self.active_json, block=True)
        self.assertEqual(self.params.get("ModelManager_ActiveJson"), self.active_json)

  def test_extra_source_is_removed(self):
    self.params.put("ModelManager_ActiveJson", {**self.active_json, "obsolete": "old"}, block=True)
    with mock.patch.object(self.params, "put", wraps=self.params.put) as put:
      ModelFetcher(self.params)
    put.assert_called_once_with("ModelManager_ActiveJson", self.active_json, block=True)
    self.assertEqual(self.params.get("ModelManager_ActiveJson"), self.active_json)

  def test_invalid_stored_json_is_replaced(self):
    stored = Path(self.params.get_param_path("ModelManager_ActiveJson"))
    for contents in (b"not json", b"\xff", b"null", b"[]", b'"unexpected string"'):
      with self.subTest(contents=contents):
        stored.write_bytes(contents)
        with mock.patch.object(self.params, "put", wraps=self.params.put) as put:
          ModelFetcher(self.params)
        put.assert_called_once_with("ModelManager_ActiveJson", self.active_json, block=True)
        self.assertEqual(self.params.get("ModelManager_ActiveJson"), self.active_json)

  def test_write_errors_are_not_hidden(self):
    self.params.remove("ModelManager_ActiveJson")
    with mock.patch.object(self.params, "put", side_effect=RuntimeError("write failed")):
      with self.assertRaisesRegex(RuntimeError, "write failed"):
        ModelFetcher(self.params)

  def test_read_errors_are_not_hidden(self):
    with mock.patch.object(self.params, "get", side_effect=RuntimeError("read failed")), \
         mock.patch.object(self.params, "put", wraps=self.params.put) as put:
      with self.assertRaisesRegex(RuntimeError, "read failed"):
        ModelFetcher(self.params)
    put.assert_not_called()
