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
