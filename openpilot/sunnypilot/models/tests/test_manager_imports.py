import subprocess
import sys
import unittest


class TestManagerNumpyImports(unittest.TestCase):
  def test_manager_import_constructor_download_and_cache_without_numpy(self):
    result = subprocess.run([sys.executable, "-c", """
import hashlib
import importlib.abc
import io
from pathlib import Path
import sys
import tempfile
from unittest import mock

attempted_imports = []

class RejectNumpy(importlib.abc.MetaPathFinder):
  def find_spec(self, fullname, path=None, target=None):
    if fullname == "numpy" or fullname.startswith("numpy."):
      attempted_imports.append(fullname)
      raise AssertionError(f"model manager imported NumPy: {fullname}")
    return None

assert "numpy" not in sys.modules
sys.meta_path.insert(0, RejectNumpy())
from openpilot.cereal import custom
from openpilot.common.file_chunker import get_chunk_name, get_manifest_path
from openpilot.common.prefix import OpenpilotPrefix
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.models import manager as manager_module
from openpilot.sunnypilot.models.helpers import REQUIRED_JSON_VERSION

with OpenpilotPrefix(), tempfile.TemporaryDirectory() as destination:
  manager = manager_module.ModelManagerSP()
  manager._report_status()
  cloudlog.info("model manager ready")
  cloudlog.event("download-start", cached=False, destination=Path(destination), details=ValueError("test"))
  bodies = [b"A" * 1024, b"B" * 2048]
  bundle = custom.ModelManagerSP.ModelBundle.new_message()
  bundle.ref = "numpy-free-manager"
  bundle.minimumSelectorVersion = REQUIRED_JSON_VERSION
  bundle.init("models", 1)
  artifact = bundle.models[0].artifact
  artifact.fileName = "import-test.pkl"
  artifact.downloadUri.uri = "https://fixture.invalid/import-test.pkl"
  artifact.downloadUri.sha256 = hashlib.sha256(b"".join(bodies)).hexdigest()
  artifact.init("chunks", len(bodies))
  body_by_url = {}
  for index, body in enumerate(bodies):
    artifact.chunks[index].sha256 = hashlib.sha256(body).hexdigest()
    body_by_url[get_chunk_name(artifact.downloadUri.uri, index, len(bodies))] = body

  def response_for(url, **kwargs):
    response = manager_module.requests.Response()
    response.status_code = 200
    response.url = url
    response.headers["content-length"] = str(len(body_by_url[url]))
    response.raw = io.BytesIO(body_by_url[url])
    return response

  manager.source_models = {"qcom": [bundle]}
  manager.available_models = [bundle]
  with mock.patch.object(manager_module.Paths, "model_root", return_value=destination), \\
       mock.patch.object(manager_module.requests.Session, "get", side_effect=response_for) as transport:
    for expected_status in (custom.ModelManagerSP.DownloadStatus.downloaded, custom.ModelManagerSP.DownloadStatus.cached):
      manager.params.put("ModelManager_DownloadRef", bundle.ref, block=True)
      manager._process_download_requests()
      assert transport.call_count == len(bodies), transport.call_count
      assert manager.params.get("ModelManager_DownloadRef") is None
      assert manager.params.get("ModelManager_ActiveBundle")["ref"] == bundle.ref
      assert manager.active_bundle.ref == bundle.ref
      assert artifact.downloadProgress.status == expected_status, artifact.downloadProgress.status
      assert artifact.downloadProgress.progress == 100
      assert artifact.downloadProgress.eta == 0
      full_path = str(Path(destination) / artifact.fileName)
      assert Path(get_manifest_path(full_path)).read_text() == str(len(bodies))
      for index, body in enumerate(bodies):
        assert Path(get_chunk_name(full_path, index, len(bodies))).read_bytes() == body
  assert not attempted_imports, attempted_imports
  assert not any(name == "numpy" or name.startswith("numpy.") for name in sys.modules)
"""], capture_output=True, text=True, timeout=30)
    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
  unittest.main()
