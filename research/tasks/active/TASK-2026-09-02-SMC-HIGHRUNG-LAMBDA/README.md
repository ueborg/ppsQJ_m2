# TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA

A targeted numerical child task of `TASK-2026-09-01-SMCRUCHE-READY`. It prepares
one overnight Ruche campaign answering two connected questions, and stops at the
human submission gate.

**Terminal state: `READY_FOR_HUMAN_SUBMISSION`.**
No agent submitted anything. `research/RESOURCE_POLICY.md` §4 forbids it
unconditionally; nothing in this package contains a scheduler call.

---

## The two questions

**A.** At `L = 128, T = 128, zeta = 0.35, lambda = 0.3032`, how far must `N_c`
rise before the finite-population CMI estimate stabilises? The ARM2 ladder is
extended to `N_c ∈ {512, 1024}` and tested by **direct rung-to-rung
differences**, never by a `1/N_c` extrapolation.

**B.** Can the jagged appearance of `CMI(lambda)` be resolved into a clean local
curve once the population is large enough? A symmetric three-point stencil at
`lambda ∈ {0.2932, 0.3032, 0.3132}` is run at a cheap `L` with a large
population and again at `L = 128`.

---

## What to submit

| arm | package | purpose | tasks | core-h | slowest task | partition |
|---|---|---|---:|---:|---:|---|
| **A** | `armA512` | L=128 high-`N_c`, rung 1 | 48 | 241 | 5.03 h | cpu_long 12:00:00 |
| **A** | `armA1024` | L=128 high-`N_c`, rung 2 | 32 | 322 | 10.06 h | cpu_long 24:00:00 |
| **B** | `armB` | low-L lambda stencil | 288 | 167 | 0.60 h | cpu_med 03:00:00 |
| **C** | `armC` | L=128 lambda stencil | 96 | 483 | 5.20 h | cpu_long 12:00:00 |
| | | **recommended total** | **464** | **1213** | | |
| *opt.* | `armA2048_optional` | one further doubling | 16 | 322 | 20.12 h | **not tonight** |
| *rejected* | — | L = 96 bridge (ARM B2) | — | 290 | — | **not prepared** |

All four recommended arms are independent Slurm arrays with no scientific
dependency; submit them together. Queue `armA1024` first — it is the wall-clock
long pole. Exact commands: **`RUCHE_RUNBOOK.md`**.

---

## Files

| file | what it is |
|---|---|
| `DESIGN.md` | the design, the arm-by-arm reasoning, and what parallelism costs |
| `LAMBDA_SPACING_DECISION.md` | why `delta_lambda = 0.010`, from measured local slope and curvature |
| `COST_MODEL.md` | measured Ruche rates, per-arm cost, elapsed time under both concurrency regimes |
| `FALSIFICATION_PLAN.md` | F1–F7, frozen, with support / kill / inconclusive criteria |
| `SMOOTHNESS_CRITERION.md` | what "clean curve" means — S1–S4, frozen, non-cosmetic |
| `analysis_spec.yaml` | the machine-readable frozen spec; the preflight prints its sha256 |
| `NC2048_AUDIT.md` | why `N_c = 2048` is prepared but not recommended tonight |
| `L96_BRIDGE_AUDIT.md` | why the L = 96 bridge is rejected and not prepared |
| `DUPLICATE_COMPUTE_AUDIT.md` | the scan for repeated work; why the historical corpus is not poolable |
| `SEED_LEDGER.md` | seed allocation and the disjointness proof |
| `INPUTS_LEDGER.md` | every input file, with sha256, and the verified-input restatement |
| `PRODUCTION_PATH_UNCHANGED.md` | evidence that nothing in the sampler moved |
| `VALIDATION.md` | every check from brief §15, with its result |
| `RUCHE_RUNBOOK.md` | the exact commands, for the human to type |
| `tools/` | `cost_model.py`, `build_arms.py`, the seed ledgers |
| `shared/` | the runtime copied into each arm |
| `support/` | the bundled, SHA-gated, byte-identical certified instrumentation |
| `frozen_inputs/` | 528 predecessor populations, tracked, hash-provenanced |
| `analysis/combined_analysis.py` | the only place F1–F7 are evaluated |

---

## The standing statistical rules this task enforces

- Uncertainty comes from **independent populations**. Clone-level spread is a
  VIF/`N_eff` diagnostic and is never a standard error.
- Genealogy can collapse completely without implying an information ceiling.
- VIF is a variance diagnostic and does **not** reliably predict bias.
- More `R` cannot repair finite-`N_c` bias. `N_c` must first be large enough for
  the interacting-particle approximation to be trustworthy; only then does more
  `R` help.
- No universal `1/N_c` bias law, no `N_c(L, zeta, lambda)` law, no claim about
  `lambda_c(zeta)` or any exponent.
- Negative results are first-class. F7 is pre-registered with the expectation
  that it comes back **KILLED**, and that outcome is to be reported, not dropped.

---

## Not done here

`research/state/**` was not written. No predecessor task archive was modified.
`main` was not touched. Nothing was submitted.
