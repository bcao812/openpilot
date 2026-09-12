import json
import re
import subprocess
from pathlib import Path

from openpilot.common.basedir import BASEDIR


def get_tinygrad_ref():
  repo_path = Path(BASEDIR) / "tinygrad_repo"
  try:
    if (repo_path / ".git").exists():
      return subprocess.check_output(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        text=True, stderr=subprocess.DEVNULL, timeout=5,
      ).strip()

    metadata_path = Path(BASEDIR) / "openpilot/sunnypilot/models/tinygrad_vendor.json"
    metadata = json.loads(metadata_path.read_text())
    if not all(re.fullmatch(r"[0-9a-f]{40}", metadata[key]) for key in ("commit", "tree")):
      return None
    tree = subprocess.check_output(
      ["git", "-C", str(BASEDIR), "rev-parse", "HEAD:tinygrad_repo"],
      text=True, stderr=subprocess.DEVNULL, timeout=5,
    ).strip()
    return metadata["commit"] if tree == metadata["tree"] else None
  except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
    return None


def main():
  current_ref = get_tinygrad_ref()
  if current_ref:
    print(current_ref)
  else:
    print("")


if __name__ == "__main__":
  main()
