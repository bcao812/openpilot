from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest


class TestRoadRecordingGate(unittest.TestCase):
  def test_control_timing_and_stale_input(self):
    compiler = shutil.which('c++')
    self.assertIsNotNone(compiler, 'A C++ compiler is required for the recording policy test')
    source = Path(__file__).with_suffix('.cc')
    with TemporaryDirectory(prefix='road-recording-gate-') as directory:
      executable = Path(directory) / 'test_gate'
      subprocess.run([compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', str(source), '-o', str(executable)],
                     check=True, capture_output=True)
      subprocess.run([str(executable)], check=True, capture_output=True)


if __name__ == '__main__':
  unittest.main()
