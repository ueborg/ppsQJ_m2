# The production path is unchanged

TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA, brief §8.

This task varies **`N_c`, `R`, `lambda` and the sampling budget**. It changes
nothing about how a trajectory is generated or how the observable is computed.

## The list, item by item

| must not change | status | how it is enforced |
|---|---|---|
| guided-cloning physics | unchanged | runs through the tracked `pps_qj` package; no file under `pps_qj/` was modified or added |
| proposal dynamics | unchanged | `proposal_c='zeta'`, the `run_instrumented` default; no manifest column overrides it |
| compensator | unchanged | inside `pps_qj`; untouched |
| resampling algorithm | unchanged | `resample_scheme='systematic'` on every manifest row; the preflight fails on any other value |
| observable definition | unchanged | `OBS-CMI-001`, quarter-system CMI, computed by the same wrapper |
| dtau convention | unchanged | `dtau_mult = 6.0` on every row, the certified value; the preflight fails on `12.0` |
| Gaussian evolution | unchanged | inside `pps_qj`; untouched |
| CMI definition | unchanged | inside `pps_qj`; untouched |

## Evidence rather than assertion

**1. No production file was modified.** `git status --porcelain` reports zero
modified tracked files across the whole repository, and nothing new or changed
under `pps_qj/`, `scripts/` or `tools/`. Every file this task adds lives under
`research/tasks/active/TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA/`.

**2. The instrumentation is byte-identical to the one that produced ARM1 and
ARM2.**

```
support/instrumented.py
  sha256 0a33c4034cda70ea635cf715ee0b160d9f29e75ceacde0de89628ff2c533032d
  11396 bytes
```

which is exactly the SHA-256 recorded in
`TASK-2026-09-01-SMCRUCHE-READY/support/BUNDLE_MANIFEST.json`, and the file
`TASK-2026-08-30-SMCSTAT` validated bitwise against the production path. It is a
copy, not a re-implementation, and `run_cell.py` re-checks the hash at startup
and refuses to run on a mismatch — a silent substitution here would change the
sampler without changing any manifest row, which is precisely the failure mode
the gate exists for.

Its import closure was re-verified independently for this task by
`ast.parse`: the only top-level imports are `__future__`, `dataclasses`,
`numpy`, `pps_qj` and `time`. There are no sibling modules, so bundling one file
is sufficient and nothing is being smuggled in from an untracked directory.

**3. `run_cell.py` is unchanged from the repaired predecessor.** Copied verbatim
from `TASK-2026-09-01-SMCRUCHE-READY/arm2/run_cell.py`, including the PACKFIX
bundled-import block and the integrity gate. Only the manifest it reads differs.

**4. Nothing new is computed in the runtime path.** `run_cell.py` calls
`I.run_instrumented(L, T, N_c, zeta, lam, dtau_mult, seed, resample_scheme,
record_anc=True)` with values taken directly from the manifest row, and records
the same fields ARM1/ARM2 recorded. All the new logic in this task
(`preflight.py`'s checks, `analyse_arm.py`, `combined_analysis.py`) is analysis
and validation; none of it runs inside a simulation.

## If a sampler change ever looks necessary

**Stop and raise it as a separate scientific/software change.** It is out of
scope here and must not be folded into a sampling-budget task. In particular, a
change made to "fix" a finite-`N_c` result would destroy the comparability of
this campaign's rungs with ARM2's completed ones, which is the entire basis of
F1.
