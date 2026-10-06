from collections import namedtuple
from pathlib import Path
from collections.abc import Sequence
from unittest.mock import patch

import openpilot.system.loggerd.deleter as deleter
from openpilot.system.loggerd.tests.loggerd_tests_common import UploaderTestCase

Stats = namedtuple("Stats", ['f_bavail', 'f_blocks', 'f_frsize'])


class TestDeleter(UploaderTestCase):
  # Deletion behavior is independent of file size; use smaller files to keep these tests fast.
  def make_file_with_data(self, f_dir: str, fn: str, size_mb: float = .001, lock: bool = False,
                          upload_xattr: bytes | None = None, preserve_xattr: bytes | None = None) -> Path:
    return super().make_file_with_data(f_dir, fn, size_mb, lock, upload_xattr, preserve_xattr)

  def fake_statvfs(self, d):
    return self.fake_stats

  def setup_method(self):
    self.f_type = "fcamera.hevc"
    super().openpilot_setup_method()
    self.fake_stats = Stats(f_bavail=0, f_blocks=10, f_frsize=4096)
    self.enterContext(patch.object(deleter.os, 'statvfs', self.fake_statvfs))

  def test_delete(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type)
    assert deleter.deleter_step() == (True, str(f_path.parent))
    assert not f_path.exists()

  def assertDeleteOrder(self, f_paths: Sequence[Path]) -> None:
    deleted_order = []
    for _ in f_paths:
      out_of_space, deleted_path = deleter.deleter_step()
      assert out_of_space and deleted_path is not None
      deleted_order.append(next(f for f in f_paths if f.parent == Path(deleted_path)))

    assert deleted_order == f_paths, "Files not deleted in expected order"

  def test_delete_order(self):
    self.assertDeleteOrder([
      self.make_file_with_data(self.seg_format.format(0), self.f_type),
      self.make_file_with_data(self.seg_format.format(1), self.f_type),
      self.make_file_with_data(self.seg_format2.format(0), self.f_type),
    ])

  def test_delete_many_preserved(self):
    self.assertDeleteOrder([
      self.make_file_with_data(self.seg_format.format(0), self.f_type),
      self.make_file_with_data(self.seg_format.format(1), self.f_type, preserve_xattr=deleter.PRESERVE_ATTR_VALUE),
      self.make_file_with_data(self.seg_format.format(2), self.f_type),
    ] + [
      self.make_file_with_data(self.seg_format2.format(i), self.f_type, preserve_xattr=deleter.PRESERVE_ATTR_VALUE)
      for i in range(5)
    ])

  def test_delete_last(self):
    self.assertDeleteOrder([
      self.make_file_with_data(self.seg_format.format(1), self.f_type),
      self.make_file_with_data(self.seg_format2.format(0), self.f_type),
      self.make_file_with_data(self.seg_format.format(0), self.f_type, preserve_xattr=deleter.PRESERVE_ATTR_VALUE),
      self.make_file_with_data("boot", self.seg_format[:-4]),
      self.make_file_with_data("crash", self.seg_format2[:-4]),
    ])

  def test_no_delete_when_available_space(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type)

    block_size = 4096
    available = (10 * 1024 * 1024 * 1024) / block_size  # 10GB free
    self.fake_stats = Stats(f_bavail=available, f_blocks=10, f_frsize=block_size)

    assert deleter.deleter_step() == (False, None)
    assert f_path.exists(), "File deleted with available space"

  def test_no_delete_with_lock_file(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type, lock=True)

    assert deleter.deleter_step() == (True, None)
    assert f_path.exists(), "File deleted when locked"

  def test_retains_logs_below_the_previous_five_gib_floor(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type)
    self.fake_stats = Stats(f_bavail=3 * 1024**3, f_blocks=30 * 1024**3, f_frsize=1)
    assert deleter.deleter_step() == (False, None)
    assert f_path.exists()

  def test_deletes_before_the_seven_percent_no_entry_threshold(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type)
    self.fake_stats = Stats(f_bavail=int(2.3 * 1024**3), f_blocks=30 * 1024**3, f_frsize=1)
    assert 100 * self.fake_stats.f_bavail / self.fake_stats.f_blocks > 7
    assert deleter.deleter_step() == (True, str(f_path.parent))

  def test_absolute_floor_still_protects_a_small_volume(self):
    f_path = self.make_file_with_data(self.seg_dir, self.f_type)
    self.fake_stats = Stats(f_bavail=512 * 1024**2, f_blocks=4 * 1024**3, f_frsize=1)
    assert deleter.deleter_step() == (True, str(f_path.parent))

  def test_faster_idle_check_and_reclaim_do_not_touch_an_open_segment(self):
    locked = self.make_file_with_data(self.seg_format.format(0), self.f_type, lock=True)
    closed = self.make_file_with_data(self.seg_format.format(1), self.f_type)
    self.fake_stats = Stats(f_bavail=3 * 1024**3, f_blocks=30 * 1024**3, f_frsize=1)
    test = self

    class Exit:
      stopped = False
      waits = []

      def is_set(self):
        return self.stopped

      def wait(self, seconds):
        self.waits.append(seconds)
        if len(self.waits) == 1:
          test.fake_stats = Stats(f_bavail=int(2.3 * 1024**3), f_blocks=30 * 1024**3, f_frsize=1)
        else:
          self.stopped = True

    event = Exit()
    with patch.object(deleter.Path, 'is_mount', return_value=False):
      deleter.deleter_thread(event)
    assert event.waits == [1, .1]
    assert locked.exists() and not closed.exists()
