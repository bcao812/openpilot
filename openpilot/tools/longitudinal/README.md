# Offline longitudinal tools

These tools read local recordings or replay outputs. They are not imported by
the vehicle's runtime processes. Keep generated JSON, caches, logs and dated
research reports outside the software checkout; pass an explicit output path.

| Module | Purpose |
|---|---|
| `replay_longitudinal_regressions` | Historical LEGACY planner windows and switch metrics |
| `replay_pedal_switches` | Timestamp-aligned planner, LongControl and Honda CAN replay |
| `replay_pedal_full_routes` | Bounded-memory full-route replay and provenance |
| `evaluate_pedal_switches` | Compare replayed event windows |
| `evaluate_pedal_full_routes` | Compare full routes against negative controls |
| `evaluate_pedal_reconstruction` | Check replay fidelity against recorded commands |
| `extract_longitudinal_trace` | Decode opt-in planner diagnostic samples |
| `replay_longitudinal_trace` | Compare trace-disabled/enabled sampled replays |
| `window_replay` | Shared verified window input/result readers |

Maintain source and tests in the active development tree. Commit the intended
changes before testing, export that explicit commit into an isolated host tree,
and make any fixes back in the active tree before committing and retesting.
Historical research snapshots retain evidence and are no longer development
copies. Record the tested commit, inputs/configuration, results and native
artifact identity; a saved commit may still fail acceptance. C3 deployment
selects an accepted commit and verifies matching ARM artifacts, actual installed
files, effective version and LEGACY/SUPERVISED/UNIFIED mode.

Run from the isolated, host-built commit snapshot, for example:

```bash
python -m openpilot.tools.longitudinal.replay_pedal_switches --help
python -m openpilot.tools.longitudinal.replay_pedal_full_routes --help
python -m openpilot.tools.longitudinal.replay_longitudinal_trace --help
python -m pytest openpilot/tools/longitudinal/tests
```

`replay_*` and `evaluate_*` moved from `openpilot.selfdrive.controls.tests`.
Tests for offline tools now live here; runtime regression tests remain under
`openpilot/selfdrive/controls/tests`.

The default-off lead-coast, brake-dwell and median experiments are not part of
the runtime planner or these replay commands. Historical experimental options
are intentionally unsupported; old metadata fields identifying an experiment
remain false in default replay outputs for format compatibility.

In the parent workspace, `_logs/pedal-switch/README.md` owns the research archive,
original source identities, rejected experiments and historical reproduction
instructions. Its immutable reports are not documentation of the current
working tree.

Recorded motion is fixed in open-loop replay. Command equivalence is not proof
of delivered acceleration, stopping distance or on-car safety/comfort.

For frozen SP decisions, replay reads acceleration from the selected nested
vision/map/speed-limit controller. The top-level `longitudinalPlanSP.aTarget`
is published after base-planner arbitration overwrites the shared output; it
is not the upstream SP acceleration input. Cruise retains the replay planner's
own recursive acceleration state and the exact `v_cruise` passed to its SP
callback, including upstream clipping and forced-deceleration handling. The
Float32 published `longitudinalPlanSP.vTarget` is not a lossless replacement
for that internal cruise input.

Timestamp alignment alone does not prove which control message the vehicle
controller consumed. Exact historical CAN fidelity needs independently
identified consumed inputs and loaded-source provenance; numerical agreement
at the recorded-input boundary does not supply either missing identity.

## Unified longitudinal architecture

Accord openpilot longitudinal without a gas interceptor always uses the unified
planner and motion supervisor. The migration parameters and production constructor
overrides have been removed. `replay_pedal_switches run --unified` records this
explicit replay selection and verifies both components actually use mode 2.
Other platform configurations retain their existing LEGACY path. Host test results
and frozen-log comparisons have their own scope; device artifacts and vehicle
behavior require separate validation.

The candidate supplies cruise and admitted model references to one MPC, keeps
model-action constraints separate, and gives the control-rate supervisor sole
ownership of normal stop/hold/release phases. Negative model speed predictions
are projected to forward speed; reference position and acceleration are derived
from that same speed profile rather than combining independent prediction means.
Planner/consumer mode acknowledgements, intent leases and explicit solver status
are part of the interface, not optional replay decoration.

An owned stop requires explicit movement permission before it can be withdrawn;
a positive delay-compensated acceleration is not that permission. Normal
departure evidence checks the native constant-jerk intervals, including their
interior velocity minima, over the published control horizon. A predicted
backward excursion cannot qualify as a forward departure. This is an intent
check, not a hard MPC constraint or proof of vehicle response. Qualified slow
lead departure remains an independent input. Small positive requests near
standstill need fresh confirmation; permitted withdrawal while already moving
keeps tracking the requested trajectory and the existing brake-release limit.

Stop entry also follows the stopping goal/reference, not just the compensated
scalar's sign. An explicit zero-speed goal publishes a stop reason and caps the
request at Honda's existing no-gas boundary; stronger braking remains available.
These scalar constraints stay separate from the normal MPC reference in the
published diagnostics.

Cold startup distinguishes healthy communication from readiness to move. An
uninitialized cruise setpoint publishes an invalid, unready motion intent while
the cold OFF consumer permits the enable handshake without longitudinal
actuation. After any accepted motion, a missing setpoint or unready plan is a
fault, not permission to return to standby or release a hold.

Shared MPC/planner failure handling also serves the remaining LEGACY configurations.
The Accord motion owner preserves qualified braking obligations during input or
solver faults. Final driver authority is checked before CarD calls
`CarInterface.apply`, using current pedals rather than the older control packet's
vehicle sample.

Run the candidate component tests in a matching isolated host build:

```bash
python -m pytest -q \
  openpilot/selfdrive/controls/tests/test_motion_supervisor.py \
  openpilot/selfdrive/controls/tests/test_unified_longcontrol.py \
  openpilot/selfdrive/controls/tests/test_unified_longitudinal_mpc.py \
  openpilot/selfdrive/controls/tests/test_unified_longitudinal_planner.py \
  openpilot/selfdrive/controls/tests/test_unified_motion_intent.py \
  openpilot/selfdrive/controls/tests/test_unified_controlsd.py \
  openpilot/selfdrive/controls/tests/test_unified_startup.py \
  openpilot/selfdrive/controls/tests/test_mpc_failure_chain.py \
  openpilot/selfdrive/car/tests/test_longitudinal_authority.py
```

These supplement, not replace, the existing longitudinal tests and fixed-input
regression suite. In the parent workspace, the implementation record is
`_arch/c3xl-dev/03-longitudinal-control-implementation-2026-09-15.md`, and
`_logs/REGRESSION-SUITE.md` owns the twelve-question evaluation contract.
Perception, DEC/SP decisions, most software feedback and measured vehicle motion
remain frozen in the basic log replay. It is not a plant-calibrated closed loop,
a proof of hill-hold performance, or an end-to-end hard-real-time lease test.

## Runtime modes and historical replay

The platform capability gate selects the runtime architecture at construction:

| Mode | Selection | Reference and stop owner |
|---|---|---|
| Legacy / 0 | Other supported configurations, including stock ACC | Existing shared reference and stop path |
| Supervised / 1 | Historical source only | Retired migration stage; numeric identity retained for old logs |
| Unified / 2 | Accord OP-long without an interceptor | Unified reference and the control-rate motion supervisor |

The two replay flags remain mutually exclusive so historical source snapshots
can still be evaluated. Requesting LEGACY or SUPERVISED on the current supported
Accord raises a mode mismatch; the tool never relabels a UNIFIED execution.
Full-route and trace wrappers propagate the requested mode and verify it in the
result. The older planner-only tool lacks consumer feedback and rejects the
motion architecture; use the timestamp-aligned planner/controller/CAN replay.

The current motion intent version and planning mode must agree at producer and
consumer. A stale version or unknown/mismatched mode cannot grant
drive or release. Cold or legacy consumers cannot acquire authority from such
a mismatch. For an already established, still-authorized motion owner, a
protocol fault is handled as degraded input rather than as a driver-requested
exit. Driver cancellation and current-pedal revocation remain higher priority.

A separately qualified braking-only envelope can preserve a stronger request
when the reference solver fails. It cannot authorize drive/release or substitute
for a valid trajectory. Its producer health, solver status, consumer health,
version/mode, exact identities and lease must independently qualify; an arbitrary
invalid packet's acceleration is not trusted.

`controls/lib/longitudinal_trajectory.py` provides offline exact constant-jerk
sampling and request audits. It has no runtime actuation authority and does not
replace `get_accel_from_plan`. Exact sampling alone does not identify the ECU
response or establish delivered braking performance.

Additional host tests:

```bash
python -m pytest -q \
  openpilot/selfdrive/controls/tests/test_longitudinal_modes.py \
  openpilot/selfdrive/controls/tests/test_unified_handover.py \
  openpilot/selfdrive/controls/tests/test_unified_creep_deadline.py \
  openpilot/selfdrive/controls/tests/test_longitudinal_trajectory.py
```

The parent-workspace revision and measured outcomes are
`_arch/c3xl-dev/05-longitudinal-control-revision-2026-09-16.md` and
`_arch/c3xl-dev/06-longitudinal-control-implementation-2026-09-16.md`.
Keep the earlier implementation record as historical evidence.

## Longitudinal observation version 1

Routine observation is **off by default**. Each process requires its own regular
marker file as well as the supported platform capability (Accord openpilot
longitudinal without a gas interceptor):

| Process | Startup marker | Diagnostic field |
|---|---|---|
| `plannerd` | `/data/plannerd_longitudinal_logging` | `longitudinalPlan.diagnostics` |
| `controlsd` | `/data/controlsd_longitudinal_logging` | `controlsState.longitudinalDiagnostics` |
| `card` | `/data/card_longitudinal_logging` | `carOutput.longitudinalDiagnostics` |
| `selfdrived` | `/data/selfdrived_longitudinal_logging` | `selfdriveState.longitudinalDiagnostics` |

Create the selected marker to enable that process's observation, or remove it
to disable observation, then restart the corresponding process (normally at the
next ignition cycle). Marker contents are ignored; directories do not enable
observation. No marker is created automatically. A missing marker or failed
marker check leaves observation disabled. Each marker is checked only at observer creation,
not on the control loop; changing it does not toggle an already-running observer.

With observation disabled, its factory returns `None` before constructing the
observer, input cache or source/configuration manifest. The existing `observe()`
fast path skips input snapshots, CAN digests and diagnostic serialization. Small
call-site argument preparation and counters shared with other observation paths
can remain. The functional messages and control protocol fields are still
published. Explicit `enabled`/`diagnostics` arguments remain available to tests
and replay harnesses without depending on the device's marker files.

Enabled observation appends to the existing messages; it does not add a service.
Record **rlog**, not just qlog; analysis needs the underlying model and control
messages as well as their identities.

The routine chain records:

- Planner update attempts, early-return reasons, actual solver/pitch-learning
  counters, provided input identities and independently known health bits.
- The control decision's actual clock, input plan, controller request and
  finally published `carControl` identity and acceleration.
- CarD's immutable previous-apply receipt, the latest-pedal authority decision,
  consumed CC/CS identities, controller frame and exact `sendcan` identity,
  ordered payload digest and count.
- Selfdrived's input identities, events, state-machine transitions, soft-disable
  timer and motion/force-deceleration health observations.

`controlsState` still publishes before `carControl`. CarD still publishes the
previous apply's `carOutput` before publishing the next CS and applying the next
command. A repeated receipt is not a new apply. The final apply may have no
following receipt at shutdown. Pair output publications by their recorded
identities; do not substitute nearest messages or assume event timestamp order
equals publication order. Existing `carOutput.valid` is not reinterpreted as the
health of its previous apply.

All identities use UInt64 on the wire and decimal strings in offline JSON.
Unknown health, missing dependencies, observation errors and partial tails must
remain distinct from a successful zero. Consumer snapshots identify available
input versions, not proof that every service affected that branch. `sendcan`
confirms a publication, not panda acceptance or delivered vehicle acceleration.

Source/configuration manifests are computed at observer startup and attached on
the first and every 1000th attempt. They hash selected on-disk sources, imported
native module files and startup CP/CP_SP representations. Cap'n Proto copies
and dataclass JSON have explicit, domain-separated encodings. These manifests
are not loaded-code/firmware attestation and do not cover all later Params reads.
No raw parameter store or secret values are dumped.

The existing `/data/longitudinal_trace` marker remains a startup-only opt-in for
the detailed planner transport, independently of the routine markers above.
Detailed samples and native/parking snapshots can still be captured without
routine planner observation, but its session/sequence/identity and completion
context are then unavailable. Enable the planner routine marker too when that
correlation is required. Legacy schema-1/2 recordings remain readable;
new motion-stage samples use schema 3 for mode and braking-only evidence.
`detail_schema=1` adds aligned companion snapshots with native MPC time/state,
reference, constraint and trajectory arrays. Unpublished or failed attempts
retain their identity without inventing a published plan. Queues are bounded,
partial batches flush on a timer, shutdown waits are bounded, and queue,
serialization, startup and transport failures have separate counters. A local
send acceptance is not logger/disk acknowledgement.

In the parent workspace, the standalone analyzer reads only explicit local
files and an explicit schema tree. It never imports the recorded controller:

```bash
V=openpilot-3xl-0817/.venv/bin/python
$V -B _logs/longitudinal_observation.py \
  --tree "$HOST_TREE" --route "$ROUTE" \
  --rlog "$RLOG_SEGMENT" --output "$NEW_OUTPUT_DIRECTORY"
```

Repeat `--rlog` for adjacent segments needed by the selected window. The output
directory must not exist. Outputs are `manifest.json`, `quality.json`,
`chain.jsonl`, `cases.jsonl`, `summary.json` and a streaming `index.sqlite3`.
Exact consumed edges, paired-publication edges, missing/ambiguous dependencies
and legacy approximations are reported separately; read completeness is not
full-chain fidelity.

Reuse existing twelve-question cases instead of defining new thresholds:

```bash
$V -B _logs/longitudinal_observation.py \
  --tree "$HOST_TREE" --route "$ROUTE" --segment 0 \
  --route-manifest _regression/accord-v1/corpus.json \
  --cases _regression/accord-v1/manifest.json \
  --case-corpus _regression/accord-v1/corpus.json \
  --output "$NEW_OUTPUT_DIRECTORY"
```

For other feature packs, provide their matching `--case-inputs`, `--case-catalog`
and `--case-corpus`; the analyzer verifies source binding and retains original
metrics, labels, selection and limits. It adds chain/context/quality anchors,
not a new twelve-question scorer or an automatic regression-suite update.

Replay progress now retains attempted/completed calls separately for warmup,
score and total execution. ERROR results retain bounded missing-dependency
examples, partial chunks and hashes; `load_result` still refuses them as complete
comparison inputs. Missing real solver counters remain unknown. Logging does
not turn incomplete execution or an observed contract failure into acceptance.

The parent-workspace design and validation record is
`_arch/c3xl-dev/04-longitudinal-logging-2026-09-15.md`. Schema/native changes need
matching device artifacts before any separately authorized installation; host
binaries must never be copied back into a vehicle checkout.
