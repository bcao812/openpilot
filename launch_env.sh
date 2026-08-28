#!/usr/bin/env bash

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# models get lower priority than ui
# - ui is ~5ms
# - modeld is 20ms
# - DM is 10ms
# in order to run ui at 60fps (16.67ms), we need to allow
# it to preempt the model workloads. we have enough
# headroom for this until ui is moved to the CPU.
export QCOM_PRIORITY=12

if [ -z "$AGNOS_VERSION" ]; then
  export AGNOS_VERSION="19.6"
fi

export STAGING_ROOT="/data/safe_staging"

# The cabin (driver monitoring) camera on this device is physically damaged; its sensor does not
# answer the i2c probe, so camerad already disables it at runtime and logs
#   camerad openSensor ** sensor 2 FAILED bringup, disabling
# Declaring it here matters for one thing runtime detection cannot reach: SpectraCamera::
# syncFirstFrame counts enabled cameras from the static ALL_CAMERA_CONFIGS array, not from the
# runtime flag, so without this it waits for a third camera that will never arrive and falls
# through the 40-frame timeout, discarding the opening frames of BOTH road cameras.
# Also gates dmonitoringmodeld / dmonitoringd off in system/manager/process_config.py.
export DISABLE_DRIVER=1

# Name the tinygrad backend instead of letting it hunt for one.
#
# tinygrad's Device.DEFAULT is `DEV.device or self._select_device` (tinygrad/device.py), and
# _select_device walks ALL_DEVICES calling Device[d] on each until one constructs -- which
# means importlib.import_module("tinygrad.runtime.ops_<x>") for CUDA, NV, AMD, MLX and the rest
# before it reaches QCOM on this Adreno board. modeld_v2/compile_modeld.py forces that walk at
# IMPORT time, because it uses `device: str = Device.DEFAULT` as a default argument value
# (lines 81, 130, 135, 143, 154) and Python evaluates default arguments when the def executes.
#
# Measured on this device, 3 runs each:
#   import openpilot.sunnypilot.modeld_v2.modeld     7.9 s   without DEV
#                                                    3.9 s   with DEV=QCOM
# That ~4 s sits on the boot critical path: modeld logs "modeld init" as the first line of
# main(), and the boot ladder puts ~9.4 s between camerad coming up and that line.
#
# QCOM is what Device.DEFAULT resolves to here anyway -- this pins the answer, it does not
# change it. tinygrad itself does the same thing for child processes (`os.environ["DEV"] =
# device` in _select_device), so this is the sanctioned knob, not a workaround.
#
# ⚠ Remove this if a chestnut/USB eGPU is ever attached: the big-model path wants that device,
# and this would pin it to the internal GPU. Checked absent here -- deviceState.chestnutPresent
# is False across the corpus, UsbGpuActive is unset, and no USB devices are enumerated.
export DEV=QCOM
