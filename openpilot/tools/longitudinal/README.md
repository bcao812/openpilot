# Offline longitudinal tools

These tools read local recordings or replay outputs. They are not imported by
the vehicle's runtime processes. Keep generated JSON, caches, logs and dated
research reports outside the software checkout; pass an explicit output path.

| Module | Purpose |
|---|---|
| `replay_longitudinal_regressions` | Planner regression windows and switch metrics |
| `replay_pedal_switches` | Timestamp-aligned planner, LongControl and Honda CAN replay |
| `replay_pedal_full_routes` | Bounded-memory full-route replay and provenance |
| `evaluate_pedal_switches` | Compare replayed event windows |
| `evaluate_pedal_full_routes` | Compare full routes against negative controls |
| `evaluate_pedal_reconstruction` | Check replay fidelity against recorded commands |
| `extract_longitudinal_trace` | Decode opt-in planner diagnostic samples |
| `replay_longitudinal_trace` | Compare trace-disabled/enabled sampled replays |
| `window_replay` | Shared verified window input/result readers |

Run from an isolated, host-built source tree, for example:

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
own recursive acceleration state.
