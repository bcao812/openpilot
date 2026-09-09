#!/usr/bin/env bash
set -euo pipefail
copy="$(realpath -e "$1")"
shift
export HOME=/tmp/c3-pedal-full-20260909-WgogBo/home
export PATH=/home/bcao/SourceCode/openpilot-3xl-0817/.venv/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH="$copy:$copy/msgq_repo:$copy/opendbc_repo:$copy/rednose_repo:$copy/teleoprtc_repo:$copy/tinygrad_repo"
export DEBUG=0
cd "$copy"
exec /home/bcao/SourceCode/openpilot-3xl-0817/.venv/bin/python "$@"
