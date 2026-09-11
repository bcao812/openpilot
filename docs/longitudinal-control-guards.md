# Accord longitudinal guards

These guards apply to `HONDA_ACCORD` with openpilot longitudinal control.
The evaluated configuration is the 2019 Accord Hybrid, Honda Bosch with radar
hardware but `radarUnavailable=True`. Other platforms and stock ACC retain
their existing paths.

## Acceleration release

The existing 8 m/s³ upward command limit also applies in ordinary Accord PID
control, not only immediately after leaving `stopping`. This prevents an
abrupt release when a lead disappears or the winning acceleration changes.
Downward commands are not rate-limited by this guard. Inactive control still
resets to zero immediately; stopped holding pressure and the stopping ramp
are unchanged.

This is an acceleration-command limit, not a bound on measured vehicle jerk.
CAN quantization and the actual regenerative/hydraulic response remain separate.

## Moving-lead creep qualification

A continuously qualified moving lead can suppress a new anti-creep stop.
The guard does not release an existing stop and does not lower
`STOP_RELEASE_ACCEL`.

- Confirmation uses the existing `CREEP_STOP_T` window. An active off-to-PID
  transition can contribute its first sample, but every off-state update
  resets history so disengagement cannot accumulate confirmation.
- Both present leads must be finite, high-confidence, moving, and not
  predicted to stop within the existing `CREEP_FINISH_T` horizon.
- Clearance must exceed 5 m plus an ego stopping/actuator-delay reserve.
  This is a bypass qualification, **not** a change to MPC `STOP_DISTANCE`.
- Only mild braking qualifies. MPC must independently request slowing;
  model-only or substantially stronger model braking retains the fallback.
- Raw stop requests, a latched stop, FCW, solver failure, missing service
  health, pedals, override and disabled control veto qualification.

The original fallback remains active for stationary or unsafe leads and
unexplained/model-directed stops. A moving-lead estimate is not ground truth;
road validation must still check misassociation and low-speed response.

## Close-lead braking state

The existing low-speed state correction remains unchanged. At higher speeds,
a separate bounded correction prevents the MPC speed state from substantially
understating measured motion during braking toward a close, closing lead.

Feedback fades in from 5 to 10 m/s, fades out as headway grows from 1 to 2 s,
and fades out as relative speed rises from −1 to 0 m/s. Only established
braking qualifies: feedback fades in from −0.5 to −1.0 m/s², leaving the
gas/brake crossover region unchanged. At full qualification
the state is no lower than `vEgo - 0.2 m/s`. It never raises an already adequate
state and does not rewrite lead distance, velocity or acceleration.

Either lead can provide the constraint. Invalid/low-confidence leads,
unhealthy inputs, overrides, inactive control, force-deceleration and
non-braking requests retain the existing behavior.

This does not repair vision range/velocity inconsistency, increase CAN brake
authority, or guarantee collision avoidance under excessive actuator lag.

## Validation boundary

Run tests only in an isolated, host-native source export. The deployment
checkout contains tracked ARM binaries; do not build over them on x86.

```bash
python -m pytest openpilot/selfdrive/controls/tests \
  openpilot/tools/longitudinal/tests \
  openpilot/selfdrive/test/longitudinal_maneuvers
python -m pytest opendbc_repo/opendbc/car/honda/tests \
  opendbc_repo/opendbc/safety/tests/test_honda.py
```

Replay keeps recorded motion/perception fixed and freezes recorded SP/DEC
decisions. It can validate command changes, not delivered stopping distance
or comfort. The closed-loop tests use an idealized acceleration/lag plant,
not a calibrated Honda actuator or signed rollback model. Keep raw and
debounced CAN transitions, negative controls, overrides and actuator-lag
failures visible. No host result authorizes deployment or replaces controlled
vehicle validation.
