# Accord longitudinal regression validation — 2026-09-08

## Scope and baseline

- Repository: `openpilot-3xl-0817`, branch `c3xl-dev`, baseline `0893979ab44c92e80e8230813eb035db789d7b56`.
- Vehicle: `HONDA_ACCORD`, Bosch with openpilot longitudinal control; not radarless or the separate Accord hybrid port.
- Evidence: workspace `_logs/DRIVING-AUDIT-2026-09-08.md`, 993 segments / 630.58 enabled minutes, GPS dates September 2–6. September 7 is not covered.
- No C3 access, deployment, firmware changes, schema changes, or device binary replacement is authorized by these commits.
- Baseline host build and tests run only in `/tmp/c3-longitudinal-validation-20260908/baseline`, exported from the recorded commit. Existing creep, LongControl, pitch-bias and DEC planner-gate tests: **55 passed**.

## Submit 1: low-speed planner state (questions 2 and 12)

Commit: `f88fd9fe9` (`Honda: bound low-speed planner state during braking`).

### Implementation

During automatic Accord braking, the planned starting speed must not lag measured speed by more than 0.2 m/s below 2 m/s. Correction fades linearly to zero at 5 m/s. Existing filtering remains for small noise, acceleration, other platforms, stock longitudinal control, driver override and higher speeds.

The change addresses integration of requested deceleration when the vehicle delivers less braking: route `00000050--076fbccae6`, t1678.061 had actual speed 1.19 m/s, planned starting speed 0.20 m/s, gap 2.22 m and manual brake takeover. It does not change perceived lead speed/distance, STOP_DISTANCE, creep thresholds, stop holding, launch release or safety acceleration limits.

### Risks and verification status

More accurate approach speed can request earlier/stronger braking. A tighter observer bound is not proof of shorter real-world stopping distance, lower jerk, or eliminated manual takeovers. No road validation has occurred.

### Local results

- Targeted candidate suite: **88 passed** (55 unchanged baseline checks plus 33 new state/closed-loop/no-lead checks). Ruff and `git diff --check` pass. The device checkout's 14 tracked shared objects pass the ARM leak gate with **zero x86 replacements**.
- Existing following-distance and longitudinal-maneuver suites: **22 passed, 2 skipped, 59 subtests passed**. `test_leads.py` was not run to completion: its process-replay import rejected the inherited `DEBUG=release`; the selected pure longitudinal suites were rerun with `DEBUG=0`. That process test does not establish additional coverage here.
- The unchanged baseline's replay of route50 t1669–1678.06 reproduces 181 automatic frames: acceleration absolute error P95 **0.000385 m/s²**, MPC initial-speed error P95 **0.00129 m/s**, zero solver failures. The replay uses the real MPC and recorded model-aligned service snapshots, DEC mode and cruise SP target; it rejects non-cruise SP targets instead of pretending to replay them.
- At t1676.016 (gap 4.59 m), candidate acceleration is **−0.802** versus baseline **−0.626 m/s²**. At t1677.512 (gap 2.67 m), it is **−1.025** versus **−0.691**. Last pre-takeover frame: planned speed **0.203→0.986 m/s** for actual 1.186 m/s, acceleration **−0.686→−0.982 m/s²**. Both arms have zero MPC failures. Perception and measured motion are identical inputs, not claimed improved outcomes.
- Synthetic closed loop: real planner plus 100 Hz LongControl, stationary lead initially 10 m ahead, initial speed 1.8 m/s, 0.5 s first-order actuator response. With unmodelled braking offsets of **0 / +0.2 / +0.4 m/s²**, stopping gaps are **5.676 / 3.783 / 0.931 m** before and **5.676 / 4.072 / 2.787 m** after. All six stop; no acceleration limit relaxation. These are stress-model outcomes, not calibrated Accord distance predictions.
- Regression coverage includes low-speed bounds, continuous 2–5 m/s exit, speed noise, positive launch requests, driver pedals/override, disabled and stock control, three unaffected Honda platforms, real MPC lead braking, and no-lead command equality at four speeds. The four no-lead cases introduce no braking; this is a bounded regression veto, not a corpus-wide 0/h claim.

### Reproduction

Use a host-built **isolated copy**, not the device checkout. The script must be identical in both baseline and candidate copies; `PYTHONPATH` selects the control code being tested. With each copy as the current directory:

```bash
python -m pytest -q openpilot/selfdrive/controls/tests/test_longitudinal_planner_state.py
python openpilot/selfdrive/controls/tests/replay_longitudinal_regressions.py \
  --logs /home/bcao/SourceCode/_logs --route 00000050--076fbccae6 \
  --start 1669 --end 1678.06 --output /tmp/near-stop-arm.json
```

The two arms need distinct output paths. Full local artifacts, baseline source export, build logs and JUnit reports are in `/tmp/c3-longitudinal-validation-20260908/`; `/tmp` is not durable storage. The low-speed tests simulate the old observer by changing only the platform gate, not by substituting a mocked solver.

## Submit 2: cruise gas/brake switching (question 13)

### Implementation and side effects

Only on automatic `HONDA_ACCORD` control, at or above 10 m/s and within 1 km/h of the cruise target, a continuous coasting band maps cruise requests −0.25…−0.15 to **−0.19 m/s²**. Linear shoulders return to the original request at −0.30 and −0.10. The maximum change is +0.06/−0.04 m/s². It is deliberately not called zero throttle: the normal Honda mapping sends about 7/1600 gas units at −0.19.

Both leads must be absent, e2e disabled, throttle allowed, service checks healthy, and the ordinary cruise candidate must already win. Pedal/override, inactive or stock longitudinal, forceDecel, SCC/map/speed-limit constraints, other platforms and stronger braking bypass the mapping. The final minimum still includes MPC, including when MPC becomes stricter than the mapped cruise request. There is no timer/latch to delay a new risk request and no change to `BRAKE_REQUEST`, gas lookup, DBC, panda safety or limits.

The internal cruise jerk state remains unmapped; the chosen final acceleration is what the planner integrates. **−0.2 is not a safe neutral encoding:** Float32 turns it into −0.20000000298, below the brake threshold. Tests send −0.19 through actual Float32 and Honda CAN encoding/parsing to verify command, gas, brake and brake-light agreement.

### Measured local effect

All 24 previously identified windows across six routes were replayed as 13 contiguous intervals, **6,386 automatic frames**, with no unmatched model/plan inputs and no MPC failures in either arm. This is the real planner plus the real Honda encoder/parser at planner-rate samples, **not a full 100 Hz CarController or vehicle replay**. Normal PID command equality is separately exercised by the existing LongControl tests.

| Same replay comparison | Baseline | Candidate |
|---|---:|---:|
| Gas/brake switches after 0.15 s phase debounce | 209 | **18** |
| Raw brake-bit switches inside the 24 windows | 372 | **38** |
| Windows improved / worsened | — | **24 / 0** |
| Windows still at or above four debounced switches | — | **1** |

Baseline acceleration versus recorded-plan error P95 is **0.00511 m/s²**, maximum **0.02488**. The road report counted 211 debounced switches; the replay counts 209, not an exact 50 Hz CAN reproduction. Changes must be compared within the same replay, not by mixing the two denominators. Candidate minus baseline acceleration stays within **−0.03999…+0.05992 m/s²** in these intervals.

The worst original route4a window, t1244.913, changes **21→0** debounced switches. Route50 t1811.215 remains **8→6**: requests still cross the lower shoulder. This is substantial mitigation, **not complete elimination**. The coasting band is not widened simply to make the last window pass.

### Closed-loop stress test

Real planner, no lead, cruise 72 km/h, 0.5 s actuator response, 60-second run:

| Unmodelled acceleration m/s² | Baseline final speed error km/h | Candidate km/h |
|---|---:|---:|
| 0.00 | ~0 | ~0 |
| +0.20 | +0.720 | **+0.916** |
| +0.25 | +0.900 | **+0.998** |
| +0.30 | +1.080 | **+1.080** |

The added steady-state error is at most **0.197 km/h** in these cases; final 20-second speed range is below 0.01 km/h. No request exceeds the existing limits. These assumptions are not vehicle identification. A constant offset placing equilibrium exactly at the brake threshold can still produce tiny Float32-level switching; neither replay nor these tests prove zero real-world switching or improved felt jerk.

### Reproduction and artifacts

The two batch invocations use identical `--windows` JSON and separate newly created `--output` directories. Each input item contains `route`, `start`, `end`. `cruise-events.json` preserves the original report's `route`, `t`, `duration`, `flips` fields for all 24 events.

```bash
python openpilot/selfdrive/controls/tests/replay_longitudinal_regressions.py \
  --logs /home/bcao/SourceCode/_logs \
  --windows /tmp/c3-longitudinal-validation-20260908/cruise-windows.json \
  --output /tmp/cruise-arm-windows
python openpilot/selfdrive/controls/tests/replay_longitudinal_regressions.py \
  --compare-events /tmp/c3-longitudinal-validation-20260908/cruise-events.json \
  --baseline-dir /tmp/c3-longitudinal-validation-20260908/cruise-baseline-windows \
  --candidate-dir /tmp/c3-longitudinal-validation-20260908/cruise-candidate-windows \
  --output /tmp/cruise-comparison.json
```

## Remaining limits and disposition of all 13 questions

| Questions | Disposition |
|---|---|
| 2 / 12, close stop / manual near-stop braking | Submit 1 corrects one demonstrated planner-state mechanism; earlier braking in replay, better gaps in a stated stress model. Real stopping distance, jerk and takeover rate still require independent validation. No new visual-only emergency-stop heuristic. |
| 13, cruise gas/brake switching | Submit 2 reduces reproduced switching with bounded scope and immediate bypasses; one of 24 original windows remains above threshold. |
| 1, DEC acc model-brake mismatch | No newest-version strict example; historical examples do not justify globally forcing blended mode or dropping the false-brake veto. Unchanged. |
| 3 / 4, stop and restart jerk | Existing holding and brake-release rates preserved. More decisive approach braking must not be presented as proven lower jerk; no global ramp reduction. |
| 5, steering oscillation | Latest taper showed improvement without a controlled A/B. No additional steering gain/filter changes. |
| 6 / 8, uphill rollback / downhill shortfall | No confirmed event in this sample; unsigned speed and grade uncertainty remain. No speculative hold/grade gain change. |
| 7, uphill response | Two candidates, uncertain grade/actuator attribution. No global throttle multiplier or unvalidated integrator. |
| 9, delayed launch | Command/release/actuator delays differ; keep the existing launch hysteresis and upward rate limit. |
| 10, low-speed gap >5 m | Do not globally lower STOP_DISTANCE/T_FOLLOW while close-stop risk exists. Unchanged. |
| 11, high-speed manual braking | Visual distance/speed inconsistency is distinct from low-speed state drift. No extrapolated range-difference fix or relaxed −3.5 m/s² limit. |

## Release boundary

Both changes are **local, platform-scoped, host-validated candidates**, not on-car approval. Before any separate deployment authorization, review the individual source commits, preserve the recorded vehicle baseline, run the ARM leak gate and use the workspace's staged safety checklist. Controlled road validation must retain driver supervision and separately measure minimum stop gap, stop/launch jerk, false deceleration, launch delay, speed tracking and gas/brake switching. Do not use public-road emergency scenarios to demonstrate performance.

## Final checks and risk-event replay

- **128 targeted tests pass**, including all 55 baseline checks, 33 state tests and 40 cruise/replay checks. Following-distance, maneuver and Honda tests: **29 passed, 2 skipped, 69 subtests passed**. Radar-fault process replay: **1 passed** after setting `DEBUG=0` (one upstream Python fork/thread deprecation warning). The earlier collection failure is an environment issue, not claimed coverage.
- All 23 route caches were searched for automatic, unoverridden, no-lead negative requests at 0–5 m/s: **226 frames**, clustered in route43 t124.975–144.023. Real replay including context has **381 automatic frames**, exactly equal baseline/candidate `aTarget` and `shouldStop`, zero solver failures. This is no newly introduced braking in that observed subset, not proof of 0/h outside it.
- Six separate risk intervals below were replayed with real MPC. All had zero solver failures and **zero `shouldStop` differences**. The high-speed route48 interval is request-for-request identical. The main route50 manual-stop interval gets up to 0.337 m/s² stronger requests; its exact numbers are in Submit 1. Route4e's launch/rebrake interval has no new stop-state switching and up to 0.069 m/s² stronger requests. That does **not** prove better ride comfort.
- Route47's departure interval has a maximum 0.00247 m/s² weaker request late in its next braking phase, much smaller than that interval's baseline reconstruction P95 error 0.0313 m/s²; route53's second near-intervention is similarly not a precise effect-size anchor (baseline P95 0.0275). Preserve these errors instead of attributing all differences to the fix.
- These open-loop comparisons keep measured speed, actual jerk, lead gap and driver takeover fixed. None demonstrates changed real stopping distance or a prevented collision. The 1.54 m logged stop and manual-intervention count remain historical facts requiring a future independent road comparison.
- Source tests and Ruff/whitespace checks pass. Host-native builds stayed in the `/tmp` exports; the active checkout's ARM shared objects are unchanged. No opendbc/panda firmware, safety limits, C++ binaries, root-workspace files or original logs were changed by this task.

### Durable input manifests

Save the following JSON blocks to `cruise-events.json` and `risk-windows.json` when rebuilding the temporary artifacts. To reproduce the 13 cruise intervals, traverse the cruise events in this order: each interval starts at `t - 0.1` and ends at `t + duration + 0.15`; merge into the previous interval only when the route matches and the next start minus previous end is less than 60 seconds. The replay adds its own 60-second warmup.

```json
[
  {"route":"00000049--d6ff10e5de","t":785.9491505860001,"duration":9.499074162999978,"flips":9},
  {"route":"00000049--d6ff10e5de","t":796.5490358420001,"duration":9.99875364199993,"flips":6},
  {"route":"00000049--d6ff10e5de","t":856.5482217050001,"duration":10.001588693999953,"flips":4},
  {"route":"00000049--d6ff10e5de","t":1029.2509091910001,"duration":10.000818433999985,"flips":14},
  {"route":"00000049--d6ff10e5de","t":1039.252437625,"duration":10.000056766999979,"flips":6},
  {"route":"0000004a--463712a822","t":1022.7094020039999,"duration":9.997500570000103,"flips":4},
  {"route":"0000004a--463712a822","t":1186.663361162,"duration":9.996521819000055,"flips":10},
  {"route":"0000004a--463712a822","t":1224.907642449,"duration":10.003765100000056,"flips":18},
  {"route":"0000004a--463712a822","t":1234.912373695,"duration":9.989676767000038,"flips":20},
  {"route":"0000004a--463712a822","t":1244.913034941,"duration":9.997954267000114,"flips":21},
  {"route":"0000004a--463712a822","t":1254.912436552,"duration":9.994822079999995,"flips":12},
  {"route":"0000004e--b6f1c9bc4e","t":1653.341657442,"duration":9.393390204999879,"flips":4},
  {"route":"0000004f--71e3118dbd","t":755.701848669,"duration":10.004978641999902,"flips":5},
  {"route":"0000004f--71e3118dbd","t":765.7077610609999,"duration":9.997072549000041,"flips":4},
  {"route":"0000004f--71e3118dbd","t":892.558069242,"duration":10.003461870999946,"flips":9},
  {"route":"0000004f--71e3118dbd","t":902.557262519,"duration":10.003198173999909,"flips":7},
  {"route":"0000004f--71e3118dbd","t":1233.610094685,"duration":10.004461610999943,"flips":4},
  {"route":"00000050--076fbccae6","t":1811.215361599,"duration":9.000581246999946,"flips":8},
  {"route":"00000050--076fbccae6","t":4272.0063412340005,"duration":10.00001160999982,"flips":14},
  {"route":"00000050--076fbccae6","t":4282.007119251,"duration":8.801945934000106,"flips":4},
  {"route":"00000050--076fbccae6","t":4480.510915425,"duration":10.00003598600033,"flips":7},
  {"route":"00000051--4c602a02a6","t":3565.8471682229997,"duration":9.99912754799998,"flips":8},
  {"route":"00000051--4c602a02a6","t":3575.846788583,"duration":9.996086662999733,"flips":9},
  {"route":"00000051--4c602a02a6","t":5374.681153939,"duration":9.997332757000368,"flips":4}
]
```

```json
[
  {"route":"00000050--076fbccae6","start":1669.0,"end":1678.06},
  {"route":"0000004e--b6f1c9bc4e","start":2280.0,"end":2305.0},
  {"route":"00000047--f3ea9093b8","start":3117.0,"end":3140.0},
  {"route":"00000050--076fbccae6","start":3080.0,"end":3110.0},
  {"route":"00000053--82133ac34f","start":3785.0,"end":3801.2},
  {"route":"00000048--af2db898b0","start":6222.0,"end":6235.1}
]
```
