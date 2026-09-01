# ARM 2 — bias calibration at `L` = 128, production geometry

**`ARM2_READY_FOR_HUMAN_SUBMISSION`.** Nothing here was submitted, and nothing
here can submit.

Prepared by TASK-2026-09-01-SMCRUCHE-READY from the frozen
TASK-2026-08-31-SMCCERT `ruche_package`. **Reorganised, not redesigned.**

---

## 1. Run ARM 1 first

ARM 1's stopping criterion applies to this arm. **If ARM 1's γ comes back
KILLED**, `N_eff` saturates, more particles cannot rescue `L` ≥ 128, and an
`L` = 128 bias calibration is answering a question the programme no longer has.
Spend the 194 core-hours after ARM 1, not before.

## 2. The question

What is `B` at `L` = 128 at production geometry?

**Every `L` = 128 cell is currently `CALIBRATION_REQUIRED`** and **no `L` = 128
run exists anywhere in the programme** — `A-P128` was cancelled by the SMCSTAT
frozen stopping rule 3 when `A-P96` returned a γ CI wider than [0.5, 1.5].

Without this, the corrected planner refuses to size any `L` = 128 cell, which is
where the Cut-B production campaign wants to run. This arm is the price of
`L` = 128 production.

## 3. The cell — frozen, do not edit

| | |
|---|---|
| `L` | 128 |
| `T` | 128.0 (= `L`) |
| ζ | 0.35 |
| λ | 0.3032 |
| `dtau_mult` | 6.0 |
| `resample_scheme` | systematic |
| `N_c` ladder | **64, 128, 256** |
| `R` per rung | 64 |
| rows | **192** |

Seeds `20128000 + 1000·N_c + r`, disjoint from every other block in the
programme.

## 4. What it costs — and the one number to distrust

| | |
|---|---|
| array tasks | 192 (`--array=0-191%64`) |
| **total** | **≈194 core-hours** |
| slowest single task | ≈1.73 core-h |
| peak memory per task | 665 MB (`--mem=2G`) |
| walltime per task | `--time=08:00:00` |

**The `L` = 128 rate is DERIVED, not measured.** The SMCSTAT timing probe gives
2.68 ms/clone-window at `L` = 96 and 6.03 at `L` = 128, a ratio of **2.250**,
applied to the measured `L` = 96 rate. No `L` = 128 run exists in this
programme, so **treat the 194 as ±50%** and run the single-task check first.

## 5. Two declared limitations

**Three rungs buys one window.** The mandatory ≥3-window sensitivity scan will be
**degenerate** here: with `N_c` ∈ {64, 128, 256} there is exactly one window of
length ≥ 3. That is the price of 194 core-hours instead of ~380 for a four-rung
ladder, and it was accepted deliberately in the frozen spec.

**`L` is confounded with `n_steps`.** At `T = L`, `n_steps ∝ λ·L·T/6 ∝ L²`, so
this cell differs from `L` = 96 in system size *and* in the number of resampling
windows. The adversarial pass (`../ADVERSARIAL_SANITY_CHECK.md` §2) found the
same confound in the local 2×2 and it is not separable here either. **`B` at
`L` = 128 is still the number production needs**, because production runs at
`T = L` too — but it may not be read as "bias grows with system size".

## 6. Reading the result

**CALIBRATED**, and admitted to `tools/calibration/bias_calibration.json`, if the
achieved **MDE|B| ≤ 3.5** — the same threshold pre-registered locally.
**UNRESOLVED**, and **not** admitted to the table, if MDE|B| > 3.5, exactly as
`A-P96` (MDE 6.84) and `S-HV96` (MDE 5.05) were excluded locally.

Projected MDE|B| at `R` = 64 is ≈3.4, i.e. **it only just clears the threshold**.
If the achieved value lands above 3.5 the honest outcome is UNRESOLVED, and the
table must not be widened to accommodate it.

## 7. Before you queue 192 tasks

```bash
export PPSQJ_REPO=/path/to/ppsQJ_m2
bash run_preflight.sh
python3 run_cell.py 0 ./results     # ONE task, ~1-2.5 h. Do this — the cost
                                    # estimate for this arm is derived, not
                                    # measured.
```

## Deployment (repaired by TASK-2026-09-01-SMCRUCHE-PACKFIX)

`[E]` **The first Ruche job failed with `ModuleNotFoundError: No module named
'instrumented'`.** `run_cell.py` imported it from
`research/tasks/active/TASK-2026-08-30-SMCSTAT/analysis`, which is an **untracked**
local research-task directory: present in the developer working tree, absent from
every git clone. The earlier readiness verdict was validated in the working tree,
so it never saw the failure.

`[E]` **Fixed by bundling.** `instrumented.py` is now tracked at
`../support/instrumented.py`, copied **byte-for-byte** (sha256
`0a33c4034cda70ea635cf715ee0b160d9f29e75ceacde0de89628ff2c533032d`), and
`run_cell.py` imports it from there. Its transitive import closure over that
untracked directory is **empty** — it needs only numpy, dataclasses, time and the
**tracked** `pps_qj` package. `run_cell.py` re-checks the SHA256 on every run and
refuses to start on a mismatch, so the instrumentation cannot be silently
swapped.

`[E]` **`PPSQJ_REPO` is no longer required.** The repository root is derived from
the package's own location; the variable still overrides it for an unusual layout.

`[E]` **Partition.** `--partition=cpu_long`. The frozen wall request is
`--time=08:00:00`; on Ruche `cpu_short` caps at 1 h, `cpu_med` at 4 h and `cpu_long`
at 7 days. `cpu_long` is the smallest that accommodates it. **The wall request was
not changed to suit the partition; the partition was chosen to suit it.** The
first attempt had no `--partition` at all, so the scheduler defaulted to
`cpu_short` and killed the job.

`[E]` **Python.** `submit.slurm` resolves the interpreter explicitly from
`PPSQJ_PYTHON`, defaulting to
`/gpfs/workdir/ercetinut/envs/pps_qj/bin/python`, prepends its directory to
`PATH`, and prints the resolved path and the numpy version before running. A
batch job does not reliably inherit an interactive `PATH`. **There is no conda on
Ruche and none is assumed.**

`[E]` **PyYAML is optional.** The frozen analysis (`analyse_ruche.py`) imports no
yaml at all and has been verified to run to completion with yaml hard-blocked.
`run_preflight.sh` uses it only to pretty-print the question and decision rule,
and falls back to a dependency-free reader when it is absent. **Nothing is ever
installed from inside a job.**

`[E]` **`run_preflight.sh` now fails (exit 1) if the package could not start**:
missing bundle, SHA mismatch, unresolvable imports, a missing `--partition`, or a
wall request the partition cannot hold. Verified by negative control on all of
these.
