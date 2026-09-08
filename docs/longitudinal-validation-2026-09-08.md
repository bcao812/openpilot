# Accord longitudinal regression validation — 2026-09-08

## Scope and baseline

- Repository: `openpilot-3xl-0817`, branch `c3xl-dev`, baseline `0893979ab44c92e80e8230813eb035db789d7b56`.
- Vehicle: `HONDA_ACCORD`, Bosch with openpilot longitudinal control; not radarless or the separate Accord hybrid port.
- Evidence: workspace `_logs/DRIVING-AUDIT-2026-09-08.md`, 993 segments / 630.58 enabled minutes, GPS dates September 2–6. September 7 is not covered.
- No C3 access, deployment, firmware changes, schema changes, or device binary replacement is authorized by these commits.
- Baseline host build and tests run only in `/tmp/c3-longitudinal-validation-20260908/baseline`, exported from the recorded commit. Existing creep, LongControl, pitch-bias and DEC planner-gate tests: **55 passed**.

## Submit 1: low-speed planner state (questions 2 and 12)

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

## Planned independent work

Submit 2 addresses question 13 only: an Accord-specific cruise coasting band with risk-channel bypasses. High-speed vision inconsistency, DEC urgency, slope compensation, holding/release and lateral tuning remain unchanged pending stronger causal evidence.
