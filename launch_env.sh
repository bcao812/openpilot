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
