import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def launch_functions(tmp_path):
  source = (Path(__file__).resolve().parents[4] / "launch_chffrplus.sh").read_text()
  declarations, separator, suffix = source.rpartition("\nlaunch\n")
  assert separator and not suffix
  # Source the production functions without entering the real device launch.
  # Keep the updater's diagnostic output inside this fixture as well.
  declarations = declarations.replace("/tmp/pip_pyserial.log", str(tmp_path / "pip_pyserial.log"))
  script = tmp_path / "launch_chffrplus.sh"
  script.write_text(declarations + "\n")
  (tmp_path / "launch_env.sh").write_text('export AGNOS_VERSION="19.7"\n')
  return script


def run_functions(script, body, **extra_env):
  env = {**os.environ, **extra_env, "CALLS": str(script.parent / "calls"),
         "PYDEPS": str(script.parent / "pydeps"), "INSTALLED": str(script.parent / "installed")}
  result = subprocess.run(["bash", "-c", 'source "$1"\n' + body, "startup-test", str(script)],
                          env=env, capture_output=True, text=True, timeout=10)
  calls = Path(env["CALLS"])
  return result, calls.read_text().splitlines() if calls.exists() else []


def test_matching_os_does_not_check_or_install_updater_dependencies(launch_functions):
  result, calls = run_functions(launch_functions, """
ensure_agnos_pyserial() { echo unexpected-bootstrap >> "$CALLS"; return 1; }
sudo() { echo unexpected-sudo >> "$CALLS"; return 1; }
update_agnos_if_needed 19.7
""")
  assert result.returncode == 0, result.stderr
  assert calls == []


def test_os_update_bootstraps_before_verification(launch_functions):
  updater = launch_functions.parent / "openpilot/common/hardware/comma/agnos.py"
  updater.parent.mkdir(parents=True)
  updater.write_text('#!/bin/sh\nprintf "verify:%s\\n" "$1" >> "$CALLS"\nexit 0\n')
  updater.chmod(0o755)
  result, calls = run_functions(launch_functions, """
ensure_agnos_pyserial() { echo bootstrap >> "$CALLS"; }
sudo() { echo "sudo:$*" >> "$CALLS"; exit 17; }
update_agnos_if_needed older-version
""")
  assert result.returncode == 17, result.stderr
  assert calls == ["bootstrap", "verify:--verify", "sudo:reboot"]


PYTHON_STUB = """
python3() {
  echo "python:$*" >> "$CALLS"
  if [ "$1" = -c ]; then
    [ -f "$INSTALLED" ]
    return
  fi
  if [ "$1" = -m ] && [ "$2" = pip ] && [ "$3" = install ]; then
    case " $* " in
      *" --no-index "*) [ "$WHEEL_SUCCEEDS" = 1 ] || return 1 ;;
    esac
    touch "$INSTALLED"
    return
  fi
  return 99
}
ensure_agnos_pyserial
"""


def test_existing_pyserial_needs_no_installer(launch_functions):
  (launch_functions.parent / "installed").touch()
  result, calls = run_functions(launch_functions, PYTHON_STUB)
  assert result.returncode == 0, result.stderr
  assert calls == ["python:-c import serial"]


@pytest.mark.parametrize("wheel_exists,wheel_succeeds", [(True, True), (True, False), (False, False)])
def test_updater_tries_offline_wheel_before_network(launch_functions, wheel_exists, wheel_succeeds):
  if wheel_exists:
    wheel = launch_functions.parent / "third_party/wheels/pyserial-test.whl"
    wheel.parent.mkdir(parents=True)
    wheel.touch()
  result, calls = run_functions(launch_functions, PYTHON_STUB, WHEEL_SUCCEEDS=str(int(wheel_succeeds)))
  assert result.returncode == 0, result.stderr
  installs = [call for call in calls if call.startswith("python:-m pip install ")]
  assert len(installs) == (1 if wheel_succeeds or not wheel_exists else 2)
  if wheel_exists:
    assert "--no-index --no-deps --find-links" in installs[0]
  if not wheel_succeeds:
    assert "--no-index" not in installs[-1]
  assert all("--target " + str(launch_functions.parent / "pydeps") in call for call in installs)
  assert calls[-1] == "python:-c import serial"
