# TASK-2026-09-02-MOCK-PRODUCTION

A numerical child task of `TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA`. It prepares one
mock-production Ruche campaign and stops at the human submission gate.

**Terminal state: `READY_FOR_HUMAN_SUBMISSION`.**
No agent submitted anything. `research/RESOURCE_POLICY.md` §4 forbids it
unconditionally; nothing in this package contains an executable scheduler call.

---

## The question

The predecessor showed that one high-population cell behaves: ARM B's three-point
stencil at `L = 64, N_c = 1024, R = 96` passed S1–S4 with `F4 SUPPORTED`.

This task asks the next question:

> **With a realistically large population, can we obtain an entire finite-size
> `CMI(lambda)` scan — three system sizes, thirteen lambdas — that is clean,
> reproducible and suitable in character for the final production analysis?**

**This is algorithm validation, not physics.** `L = 32, 48, 64` are at or below
the programme's own corpus floor. Nothing here may be read as a phase boundary,
however clean the crossing turns out to be.

---

## What to submit

| arm | package | purpose | tasks | core-h | slowest task | elapsed at %64 | partition |
|---|---|---|---:|---:|---:|---:|---|
| **main** | `mockL64` | the L=64 curve, minus the 3 reused ARM-B lambdas | 240 | 129.3 | 0.66 h | **2.32 h** | cpu_med 03:00:00 |
| **main** | `mockL64nc2048` | `Delta_N(lambda)` shape check | 72 | 97.3 | 1.39 h | 1.75 h | cpu_med 04:00:00 |
| **main** | `mockL48` | the L=48 curve | 312 | 58.8 | 0.23 h | 1.06 h | cpu_med 02:00:00 |
| **main** | `mockL32` | the L=32 curve | 312 | 12.1 | 0.05 h | 0.22 h | cpu_short 01:00:00 |
| *companion* | `mockNC128L64` | matched low-`N_c` comparator | 624 | 57.3 | 0.11 h | 1.03 h | cpu_short 01:00:00 |
| *companion* | `mockNC128L48` | matched low-`N_c` comparator | 624 | 19.9 | 0.04 h | 0.36 h | cpu_short 01:00:00 |
| *companion* | `mockNC128L32` | matched low-`N_c` comparator | 624 | 4.1 | 0.01 h | 0.07 h | cpu_short 01:00:00 |
| | | **recommended total** | **2808** | **378.8** | | **2.32 h** | |
| *rejected* | — | `L = 80` | — | 469 | — | 8.43 h | **not prepared** |

All seven are independent Slurm arrays with no scientific dependency; submit
them together. Queue `mockL64` first — it is the wall-clock long pole. Exact
commands: **`RUCHE_RUNBOOK.md`**.

**Separately, and not part of this campaign:** the `L = 128, N_c = 2048` rung
already packaged in the predecessor as `armA2048_optional`. It is not duplicated
here. Its cost is now known to be 55 % higher than that package reports — read
**`L128_NC2048_HANDOFF.md`** before submitting it.

---

## The design in one screen

```
zeta = 0.35,  T = L,  dtau_mult = 6.0 (certified),  systematic resampling

lambda grid   13 points, delta_lambda = 0.010, IDENTICAL at every L
              0.2332 0.2432 0.2532 0.2632 0.2732 0.2832 0.2932
              0.3032 0.3132 0.3232 0.3332 0.3432 0.3532

L = 32, 48, 64   at N_c = 1024, R = 24      (the mock production scan)
L = 64           at N_c = 2048, R = 24, three central lambdas
L = 32, 48, 64   at N_c = 128,  R = 48      (matched low-N_c comparator)

REUSED, not recomputed: 288 populations at L = 64, N_c = 1024, R = 96,
lambda in {0.2932, 0.3032, 0.3132}, from TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA
ARM B, frozen into frozen_inputs/armB_populations.csv.
```

---

## Files

| file | what it is |
|---|---|
| `DESIGN.md` | the design, the arm-by-arm reasoning, and what parallelism costs |
| `LAMBDA_GRID_DECISION.md` | why these 13 lambdas, from the measured corpus and not from a critical law |
| `POWER_AND_R_DECISION.md` | the power calculation from the ACTUAL ARM-B data; why `R = 24` |
| `COST_MODEL.md` | measured Ruche rates, the two corrections to the predecessor's model, per-arm cost and wall-clock |
| `REUSE_AND_DEDUP_AUDIT.md` | the ARM-B reuse, and why the historical corpus shares no compatible cell |
| `NC128_COMPANION_RATIONALE.md` | the one departure from the brief's arm list, and why |
| `L80_RUNTIME_GATE.md` | why `L = 80` is rejected on measured timing |
| `L128_NC2048_HANDOFF.md` | the predecessor's `N_c = 2048` rung, re-costed and handed over |
| `MATCHED_R_AMENDMENT.md` | the matched-R = 24 primary comparison, and why it is required |
| `SUCCESS_CRITERIA.md` | M1–M7, the §10 quality battery and the §11 crossing protocol, frozen |
| `FALSIFICATION_PLAN.md` | X1–X8, frozen, including the expected negative results |
| `analysis_spec.yaml` | the machine-readable frozen spec; the preflight prints its sha256 |
| `SEED_LEDGER.md` | seed allocation and the disjointness proof |
| `INPUTS_LEDGER.md` | every input file, with sha256, and the verified-input restatement |
| `PRODUCTION_PATH_UNCHANGED.md` | evidence that nothing in the sampler moved |
| `VALIDATION.md` | every check, with its result, including two defects this task's own testing found |
| `RUCHE_RUNBOOK.md` | the exact commands, for the human to type |
| `tools/` | `cost_model.py`, `build_arms.py`, `dedup_scan.py`, `test_matched_r.py`, the seed ledgers |
| `shared/` | the runtime copied into each arm |
| `support/` | the bundled, SHA-gated, byte-identical certified instrumentation |
| `frozen_inputs/` | the 288 reused ARM-B populations and the `zeta = 0.35` corpus slice, both hashed |
| `analysis/mock_production_analysis.py` | the only place M1–M7 are evaluated |

---

## The standing statistical rules this task enforces

- Uncertainty comes from **independent populations**. Clone-level spread is a
  VIF/`N_eff` diagnostic and is never a standard error.
- **Matched `R = 24`** for every primary curve-quality, crossing and
  reproducibility statistic, so that "higher `N_c` is cleaner" can never be a
  statement about `R`. Cells with more populations are cut into disjoint
  blocks of 24 **in seed order**, observable-blind, block A primary.
  Full-`R` means, replicate blocks B/C/D and `R = 12` historical-precision
  subsets are **secondary** and carry no cleanliness claim.
  See `MATCHED_R_AMENDMENT.md`.
- **Per-point** standard errors everywhere. Under the amendment the primary
  analysis is uniformly `R = 24`, so they are equal by construction; the rule
  still binds the secondary full-`R` views.
- Genealogy can collapse completely without implying an information ceiling.
- VIF is a variance diagnostic and does **not** reliably predict bias.
- More `R` cannot repair finite-`N_c` bias.
- No universal `1/N_c` bias law, no `N_c(L, zeta, lambda)` law, no claim about
  `lambda_c(zeta)` or any exponent.
- **No smoothing, no imposed monotonicity, no post-hoc removal of a lambda
  point** — including if the grid turns out not to bracket the crossing, which
  is an INCONCLUSIVE and a child task, not an extension of this one.
- Negative results are first-class. `FALSIFICATION_PLAN.md` X8 pre-registers
  three expected ones, including the possibility that the old jaggedness was
  already consistent with sampling noise — which would make part of this
  campaign's own motivation retrospectively weaker, and gets reported anyway.

---

## Two defects this task's own testing found

Recorded here rather than buried, because they are the reason the testing was
worth doing (`VALIDATION.md` §6 and §7):

1. **The `Delta_N` bootstrap null was wrong**, in the direction that hides the
   effect M5 exists to detect. On synthetic data with a deliberate
   lambda-dependence it returned `p = 0.572` and "unresolved" where the truth
   was "lambda-dependent". Fixed and re-verified at `p = 0.045`.
2. **An off-set `N_c` crashed the preflight with a `KeyError`** instead of
   reporting the design failure. It failed closed, but a traceback is not a
   report. Fixed.

---

## Not done here

`research/state/**` was not written. No predecessor task archive was modified.
`main` was not touched. No `L = 80` package exists. Nothing was submitted, and
nothing in this package can submit.
