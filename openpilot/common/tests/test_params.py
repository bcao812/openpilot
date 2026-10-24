import datetime
import os
import threading
import time
import uuid

from openpilot.common.test import OpenpilotTestCase
from openpilot.common.params import Params, ParamKeyFlag, ParamKeyType, UnknownKeyName

class TestParams(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()

  def test_params_put_and_get(self):
    self.params.put("DongleId", "cb38263377b873ee", block=True)
    assert self.params.get("DongleId") == "cb38263377b873ee"

  def test_params_non_ascii(self):
    st = b"\xe1\x90\xff"
    self.params.put("CarParams", st, block=True)
    assert self.params.get("CarParams") == st

  def test_params_get_cleared_manager_start(self):
    self.params.put("CarParams", b"test", block=True)
    self.params.put("DongleId", "cb38263377b873ee", block=True)
    assert self.params.get("CarParams") == b"test"

    undefined_param = self.params.get_param_path(uuid.uuid4().hex)
    with open(undefined_param, "w") as f:
      f.write("test")
    assert os.path.isfile(undefined_param)

    self.params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
    assert self.params.get("CarParams") is None
    assert self.params.get("DongleId") is not None
    assert not os.path.isfile(undefined_param)

  def test_params_two_things(self):
    self.params.put("DongleId", "bob", block=True)
    self.params.put("AthenadPid", 123, block=True)
    assert self.params.get("DongleId") == "bob"
    assert self.params.get("AthenadPid") == 123

  def test_params_get_block(self):
    def _delayed_writer():
      time.sleep(0.1)
      self.params.put("CarParams", b"test", block=True)
    threading.Thread(target=_delayed_writer).start()
    assert self.params.get("CarParams") is None
    assert self.params.get("CarParams", block=True) == b"test"

  def test_params_unknown_key_fails(self):
    with self.assertRaises(UnknownKeyName):
      self.params.get("swag")

    with self.assertRaises(UnknownKeyName):
      self.params.get_bool("swag")

    with self.assertRaises(UnknownKeyName):
      self.params.put("swag", "abc", block=True)

    with self.assertRaises(UnknownKeyName):
      self.params.put_bool("swag", True, block=True)

    with self.assertRaises(UnknownKeyName):
      self.params.put(b"DongleId\0suffix", "abc", block=True)

    assert self.params.get_param_path(b"key\0suffix").endswith("/key\0suffix")

  def test_remove_not_there(self):
    assert self.params.get("CarParams") is None
    self.params.remove("CarParams")
    assert self.params.get("CarParams") is None

  def test_get_bool(self):
    self.params.remove("IsMetric")
    assert not self.params.get_bool("IsMetric")

    self.params.put_bool("IsMetric", True, block=True)
    assert self.params.get_bool("IsMetric")

    self.params.put_bool("IsMetric", False, block=True)
    assert not self.params.get_bool("IsMetric")

    self.params.put("IsMetric", True, block=True)
    assert self.params.get_bool("IsMetric")

    self.params.put("IsMetric", False, block=True)
    assert not self.params.get_bool("IsMetric")

  def test_put_non_blocking_with_get_block(self):
    q = Params()
    def _delayed_writer():
      time.sleep(0.1)
      Params().put("CarParams", b"test")
    threading.Thread(target=_delayed_writer).start()
    assert q.get("CarParams") is None
    assert q.get("CarParams", True) == b"test"

  def test_put_bool_non_blocking_with_get_block(self):
    q = Params()
    def _delayed_writer():
      time.sleep(0.1)
      Params().put_bool("CarParams", True)
    threading.Thread(target=_delayed_writer).start()
    assert q.get("CarParams") is None
    assert q.get("CarParams", True) == b"1"

  def test_params_all_keys(self):
    keys = Params().all_keys()

    # sanity checks
    assert len(keys) > 20
    assert len(keys) == len(set(keys))
    assert b"CarParams" in keys

  def test_params_all_keys_by_flag_retains_distinct_names(self):
    registered = set(self.params.all_keys())
    expected = {
      ParamKeyFlag.BACKUP: {b"IsMetric", b"LongitudinalPersonality", b"SoundMode"},
      ParamKeyFlag.PERSISTENT: {b"DongleId", b"IsMetric"},
      ParamKeyFlag.CLEAR_ON_MANAGER_START: {b"CarParams", b"ControlsReady"},
    }
    for flag, required in expected.items():
      with self.subTest(flag=flag):
        keys = self.params.all_keys(flag)
        assert required <= set(keys)
        assert set(keys) <= registered
        assert len(keys) == len(set(keys))
        # Other string-returning calls must not corrupt the already copied names.
        self.params.get_param_path("SoundMode")
        assert set(self.params.all_keys(flag)) == set(keys)
    assert self.params.all_keys(ParamKeyFlag(0)) == []

  def test_params_default_value(self):
    self.params.remove("LanguageSetting")
    self.params.remove("LongitudinalPersonality")
    self.params.remove("LiveParametersV2")

    assert self.params.get("LanguageSetting") is None
    assert self.params.get("LanguageSetting", return_default=False) is None
    assert isinstance(self.params.get("LanguageSetting", return_default=True), str)
    assert isinstance(self.params.get("LongitudinalPersonality", return_default=True), int)
    assert self.params.get("LiveParametersV2") is None
    assert self.params.get("LiveParametersV2", return_default=True) is None

  def test_road_recording_default_and_explicit_opt_out(self):
    self.params.remove("RecordRoadCameras")
    assert self.params.get("RecordRoadCameras") is None
    assert self.params.get_default_value("RecordRoadCameras") is True
    assert self.params.get("RecordRoadCameras", return_default=True) is True
    self.params.put_bool("RecordRoadCameras", False, block=True)
    # Manager's default initialization must leave an explicit user choice alone.
    default = self.params.get_default_value("RecordRoadCameras")
    if self.params.get("RecordRoadCameras") is None:
      self.params.put("RecordRoadCameras", default, block=True)
    assert self.params.get("RecordRoadCameras", return_default=True) is False

  def test_params_get_type(self):
    # json
    self.params.put("LiveParameters", {"a": 0}, block=True)
    assert self.params.get("LiveParameters") == {"a": 0}

    # int
    self.params.put("BootCount", 1441, block=True)
    assert self.params.get("BootCount") == 1441

    # bool
    self.params.put("AdbEnabled", True, block=True)
    assert self.params.get("AdbEnabled")
    assert isinstance(self.params.get("AdbEnabled"), bool)

    # time
    now = datetime.datetime.now(datetime.UTC)
    self.params.put("InstallDate", now, block=True)
    assert self.params.get("InstallDate") == now

  def test_live_streaming_is_not_cleared_on_ignition(self):
    self.params.put_bool("IsLiveStreaming", True, block=True)
    self.params.clear_all(ParamKeyFlag.CLEAR_ON_IGNITION_ON)
    assert self.params.get_bool("IsLiveStreaming")
    self.params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)
    assert not self.params.get_bool("IsLiveStreaming")

  def test_remote_map_deletion_is_not_registered(self):
    assert b"Mapd_ClearCache" not in self.params.all_keys()
    with self.assertRaises(UnknownKeyName):
      self.params.get("Mapd_ClearCache")

  def test_retired_feature_params_are_cleared_without_changing_shared_settings(self):
    retired_keys = (
      "ApiCache_FirehoseStats", "ApiCache_DriveStats", "MapAdvisorySpeedLimit", "MapdVersion",
      "MapSpeedLimit", "NextMapSpeedLimit", "Offroad_OSMUpdateRequired", "OsmDbUpdatesCheck",
      "OSMDownloadBounds", "OsmDownloadedDate", "OSMDownloadLocations", "OSMDownloadProgress",
      "OsmLocal", "OsmLocationName", "OsmLocationTitle", "OsmLocationUrl", "OsmStateName",
      "OsmStateTitle", "OsmWayTest", "RoadName", "RoadNameToggle", "SpeedLimitPolicy",
      "MapTargetVelocities", "SmartCruiseControlMap",
    )
    registered = self.params.all_keys()
    for key in retired_keys:
      assert key.encode() not in registered
      # Model files left by an earlier version without using the retired registry.
      with open(self.params.get_param_path(key), "w") as f:
        f.write("1")

    self.params.put_bool("SmartCruiseControlVision", True, block=True)
    self.params.put("SpeedLimitMode", 3, block=True)
    self.params.put("RouteCount", 42, block=True)
    self.params.put_bool("OnroadUploads", True, block=True)
    self.params.clear_all(ParamKeyFlag.CLEAR_ON_MANAGER_START)

    assert all(not os.path.exists(self.params.get_param_path(key)) for key in retired_keys)
    assert self.params.get_bool("SmartCruiseControlVision")
    assert self.params.get("SpeedLimitMode") == 3
    assert self.params.get("RouteCount") == 42
    assert self.params.get_bool("OnroadUploads")

  def test_selected_model_keys_and_local_recording_key(self):
    expected_types = {
      "ModelManager_DownloadRef": ParamKeyType.STRING,
      "ModelManager_ActiveJson": ParamKeyType.JSON,
      "ModelManager_ActiveBundleChestnut": ParamKeyType.JSON,
      "ChestnutModelError": ParamKeyType.BOOL,
      "RecordRoadCameras": ParamKeyType.BOOL,
    }
    for key, expected_type in expected_types.items():
      with self.subTest(key=key):
        assert self.params.get_type(key) == expected_type
    self.params.put("ModelManager_DownloadRef", "model-ref", block=True)
    self.params.put("ModelManager_ActiveJson", {"version": 19}, block=True)
    self.params.put("ModelManager_ActiveBundleChestnut", {"ref": "big-model-ref"}, block=True)
    assert self.params.get("ModelManager_DownloadRef") == "model-ref"
    assert self.params.get("ModelManager_ActiveJson") == {"version": 19}
    assert self.params.get("ModelManager_ActiveBundleChestnut") == {"ref": "big-model-ref"}
    assert not self.params.get_bool("RecordRoadCameras")
