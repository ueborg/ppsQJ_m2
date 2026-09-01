# ARM 1 — high-VIF variance scaling at production geometry

**`ARM1_READY_FOR_HUMAN_SUBMISSION`.** Nothing here was submitted, and nothing
here can submit: there is no scheduler call in any script in this directory.

Prepared by TASK-2026-09-01-SMCRUCHE-READY from the frozen
TASK-2026-08-31-SMCCERT `ruche_package`. **Reorganised, not redesigned** — every
scientific parameter is copied verbatim and the split was verified row by row
(`../build_arms.py`).

---

## 1. The question

Does `Var(Î) ∝ N_c^{−1}` hold at **measured VIF ≥ 40**, at the **production
geometry `T = L`**?

Three local attempts left it INCONCLUSIVE. The sharpest — `S-HV96`, median VIF
84.8 — gave **γ = +0.630, CI [+0.389, +0.856]** and **missed the pre-registered
KILL by 0.003** at one CI endpoint, but ran at `T` = 16 ≠ `L`, so it is a
statement about the VIF regime and not about production cells.

**A KILL here is the consequential outcome.** If γ < 1 robustly, `N_eff`
saturates, more particles cannot rescue `L` ≥ 128, and the programme needs a
different sampler — a redirect, not a tuning change. A SUPPORTED changes nothing
operational: the `N_c` floors in `PRODUCTION_CONFIGURATION.md` stay where they
are.

## 2. The cell — frozen, do not edit

| | |
|---|---|
| `L` | 96 |
| `T` | 96.0 (= `L`, the production geometry) |
| ζ | 0.35 |
| λ | 0.3032 |
| `dtau_mult` | 6.0 |
| `resample_scheme` | systematic |
| `N_c` ladder | **128, 256, 512** |
| new populations `R` | 32, 32, 48 |
| rows | **112** |

**After pooling with the completed local blocks** the ladder becomes
`N_c` = 32 / 64 / 128 / 256 / 512 at `R` = 128 / 64 / 48 / 48 / 48. The 32 and 64
rungs already exist from `A-BUD`; 128 and 256 exist at `R` = 16 from `A-P96` and
are extended here; 512 is new.

**Seeds continue the `A-P96` stream** (`1040000 + 1000·N_c + r`, `r` = 16…47), so
no completed run is repeated and cluster rows pool directly with local ones.

`[Why only the top rungs]` The inherited `A-BUD` already tested the alternative:
`A-P96`+`A-BUD` has `R` = 128 at `N_c` = 32 and still returns γ INCONCLUSIVE,
because γ is fitted across the whole ladder and the top rungs stayed at `R` = 16.
That reuse is why this arm costs 62 rather than ~140 core-hours.

## 3. What it costs

Run `run_preflight.sh` — do not trust this table, it is a copy.

| | |
|---|---|
| array tasks | 112 (`--array=0-111%64`) |
| **total** | **≈62 core-hours** |
| slowest single task | ≈0.86 core-h |
| peak memory per task | 732 MB (`--mem=2G`) |
| walltime per task | `--time=04:00:00` |

The rate is **measured**, not modelled: 6.59e-3 s per clone-window at `L` = 96,
from the SMCSTAT `B-T96` block's wall clock.

## 4. Bonus: it also upgrades the bias calibration

At `R` = 48 the projected MDE|B| at this cell is ≈2.4, below the frozen 3.5
threshold, so ARM 1 also tightens `B` at **the only `T = L` cell the programme
has**. The adversarial pass found this matters: the current
`bias@512 = +0.0220` is **smaller than the 0.0233 form-uncertainty in `I_∞`**, so
"`N_c` = 512 is still inadequate" is **not currently established**. The new 512
rung settles it.

## 5. Files

| file | what |
|---|---|
| `manifest.csv` | 112 rows; arm, `L`, `T`, `N_c`, ζ, λ, `dtau_mult`, scheme, **seed** |
| `analysis_spec.yaml` | the **frozen** decision rules, extracted verbatim |
| `run_cell.py` | runs **one** row; idempotent, skips completed rows |
| `submit.slurm` | the array job — **read it before submitting** |
| `run_preflight.sh` | prints what would be requested; **cannot submit** |
| `analyse_results.sh` | applies the frozen rules, pooling with local blocks |
| `analyse_ruche.py` | the analysis itself |
| `results/` | one JSON per completed task |

## 6. Before you queue 112 tasks

```bash
export PPSQJ_REPO=/path/to/ppsQJ_m2
bash run_preflight.sh
python3 run_cell.py 0 ./results     # ONE task, ~30-50 min. Do this.
```

The SMCSTAT campaign discovered a **2.45× cost error** only after committing.
One interactive task tells you whether the estimate holds on that machine.

## 7. Reading the result

The frozen rule, unchanged from SMCSTAT F1:

- **SUPPORTED** — γ CI contains 1 in the full window **and** in ≥1
  dropped-endpoint window.
- **KILLED** — γ CI excludes 1 from below in the full **and** drop-smallest
  windows.
- **INCONCLUSIVE** — γ CI not contained in [0.5, 1.5]. **INCONCLUSIVE dominates
  SUPPORTED.**
- **GATED** — if the median measured VIF is < 40, reported as UNTESTED AT HIGH
  VIF whatever γ comes out.

Projected CI width at `R` = 48 over five rungs: ≈0.33, decisive either way
against the [0.5, 1.5] criterion.
