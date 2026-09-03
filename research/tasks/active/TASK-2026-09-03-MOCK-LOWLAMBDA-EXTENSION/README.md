# TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION

A **cheap** numerical child task of the completed
`TASK-2026-09-02-MOCK-PRODUCTION`. It extends that campaign's measured
`CMI(lambda)` curves downward by four lambda points and stops at the human
submission gate.

**Terminal state: `READY_FOR_HUMAN_SUBMISSION`.**
No agent submitted anything. `research/RESOURCE_POLICY.md` §4 forbids it
unconditionally; nothing in this package contains an executable scheduler call.

The predecessor archive was **read and not modified** — see
`PREDECESSOR_UNMODIFIED.md`. Nothing under `research/state/**` was touched.

---

## The question

The predecessor established that at `zeta = 0.35`, `N_c = 1024`, `R = 24`,
`L = 32, 48, 64`, `T = L`, the raw unsmoothed `CMI(lambda)` curves on
`lambda = 0.2332 … 0.3532` are smooth and statistically well resolved.

Its **crossing analysis was not properly bracketed**:

| pair | raw crossings on the 13-point grid | where the bootstrap mass sat |
|---|---:|---|
| L32–L48 | **0** | 95 % CI `[0.23327, 0.23714]`, against the lower endpoint |
| L32–L64 | **0** | 95 % CI `[0.23329, 0.23776]`, against the lower endpoint |
| L48–L64 | 1, at `0.23691` | `endpoint_induced = True` |

Every one of those sits within one grid step of `0.2332`, the lowest lambda
that was scanned. So:

> **Does extending the same curves to lower lambda convert this
> lower-boundary / endpoint-sensitive locator into a genuinely interior,
> bracketed, reproducible crossing structure?**

**This is locator and algorithm validation, not physics.** Nothing here is
`lambda_c(zeta)`, and no phase-boundary law is fitted. `L = 32, 48, 64` are at
or below the programme's own corpus floor.

The predecessor's own README anticipated this outcome and pre-committed to the
response: an unbracketed grid *"is an INCONCLUSIVE and a child task, not an
extension of this one."* This is that child task.

---

## What to submit

| arm | package | tasks | lambdas | core-h | slowest task | elapsed at %64 | partition |
|---|---|---:|---|---:|---:|---:|---|
| `lowlamL32` | L = 32 | 96 | 4 | 3.6 | 2.4 min | 4.8 min | **cpu_med** `00:20:00` |
| `lowlamL48` | L = 48 | 96 | 4 | 14.5 | 9.6 min | 19.2 min | **cpu_med** `00:45:00` |
| `lowlamL64` | L = 64 | 96 | 4 | 43.2 | 28.3 min | 56.6 min | **cpu_med** `02:00:00` |
| | **total** | **288** | | **61.3** | | **56.6 min** | |

All three are independent Slurm arrays with no scientific dependency; submit
them together. `lowlamL64` is the wall-clock long pole. Elapsed figures
**exclude queue wait**, which will dominate all of them. Exact commands:
**`RUCHE_RUNBOOK.md`**.

**All three arms are `cpu_med`.** Two of them would fit inside `cpu_short`'s
1 h `MaxTime` and are deliberately not sent there — the preceding campaign
showed `cpu_short` is effectively serialised for this account by
`QOSMaxJobsPerUserLimit`. The preflight *requires* `cpu_med` and exits non-zero
on anything else. See `SCHEDULER_DECISION.md`.

---

## The design in one screen

```
zeta = 0.35,  T = L,  N_c = 1024,  R = 24,  dtau_mult = 6.0 (certified),
systematic resampling  --  IDENTICAL to the predecessor in every respect

frozen 17-point grid, delta_lambda = 0.010, identical at every L:

  NEW, computed here (grid indices 0-3):
      0.1932  0.2032  0.2132  0.2232
  REUSED from TASK-2026-09-02-MOCK-PRODUCTION (indices 4-16), never recomputed:
      0.2332  0.2432  0.2532  0.2632  0.2732  0.2832  0.2932
      0.3032  0.3132  0.3232  0.3332  0.3432  0.3532

L = 32, 48, 64 at every one of the four new lambdas
  ->  4 lambdas x 24 populations x 3 L  =  288 new tasks, 96 per arm

REUSED, not recomputed: 1152 populations covering all 39 predecessor cells,
frozen into frozen_inputs/predecessor_nc1024_populations.csv with provenance
and sha256. Their statistics reproduce the predecessor's published curves
EXACTLY -- worst absolute deviation 0.0 over all 39 means and all 39 SEMs.
```

The sampler is `support/instrumented.py`, **byte-identical**
(sha256 `0a33c403…`) to the file that produced every reused population. That
identity is not housekeeping: it is what licenses treating the seventeen points
as one curve rather than two datasets that happen to agree.

---

## Files

| file | what it is |
|---|---|
| `DESIGN.md` | the design, what is new, what is reused, and what is deliberately not here |
| `LAMBDA_EXTENSION_DECISION.md` | why these four lambdas and why exactly four |
| `SCHEDULER_DECISION.md` | why `cpu_med` on all three arms, against the predecessor's own rule |
| `COST_MODEL.md` | the measured-runtime refit, the two models, and every number in the table above |
| `REUSE_AND_DEDUP_AUDIT.md` | what is reused, what is refused, and the duplicate scan |
| `SEED_LEDGER.md` | seed allocation and the disjointness proof |
| `INPUTS_LEDGER.md` | every input file with sha256, and the verified-input restatement |
| `PREDECESSOR_UNMODIFIED.md` | the evidence that the completed archive was not touched |
| `SUCCESS_CRITERIA.md` | X1–X7, the join tests J1–J3 and the interiority test I1–I3, frozen |
| `FALSIFICATION_PLAN.md` | Y1–Y8, frozen, including the expected negative results |
| `analysis_spec.yaml` | the machine-readable frozen spec; the preflight prints its sha256 |
| `VALIDATION.md` | every check with its result, including one defect this task's testing found |
| `RUCHE_RUNBOOK.md` | the exact commands, for the human to type |
| `analysis/lowlambda_analysis.py` | the only place the frozen criteria are evaluated |
| `tools/` | `freeze_predecessor.py`, `cost_model.py`, `build_arms.py`, `dedup_scan.py`, `smoke_test.py`, `negative_controls.py`, the seed ledgers |
| `shared/` | the runtime copied into each arm |
| `support/` | the bundled, SHA-gated, byte-identical certified instrumentation |
| `frozen_inputs/` | the 1152 reused predecessor populations, hashed |

---

## The standing statistical rules this task enforces

Inherited unchanged from the predecessor, because the two halves of the grid
must be analysed under one set of rules or they are not one curve:

- Uncertainty comes from **independent populations**. Clone-level spread is a
  `VIF`/`N_eff` diagnostic and is never a standard error.
- **Matched `R = 24`** at every cell of every primary statistic. The three
  reused `L = 64` cells hold `R = 96` and are cut into disjoint blocks of 24
  **in seed order**, observable-blind, block A primary. Full-`R` views are
  secondary and carry no curve-quality or crossing authority.
- Genealogy can collapse completely without implying an information ceiling.
- VIF is a variance diagnostic and does **not** reliably predict bias.
- **No smoothing, no interpolation replacing a measured point, no imposed
  monotonicity, no post-hoc removal of a lambda point**, and **no special fit
  across the join** at `lambda = 0.2332`. The analysis writes an audit block
  asserting all of this into its own results file.
- **The grid is not extended again automatically.** If the crossing simply
  moves to `lambda <= 0.1932`, or stays boundary-driven, that is the reportable
  result. `FALSIFICATION_PLAN.md` Y6 pre-registers it.

---

## What this task may conclude, and what it may not

It **may** conclude, if and only if the diagnostics support it:

> For `zeta = 0.35`, `L <= 64` and `N_c = 1024`, the measured unsmoothed CMI
> curves are statistically smooth over `lambda = 0.1932–0.3532`.

That statement is bound to `zeta = 0.35`, `L <= 64`, `N_c = 1024` and this
guided-cloning configuration. It **may not** imply that `N_c = 1024` is
adequate at `L = 96` or `128`, that `N_c = 1024` is adequate at lower `zeta`,
that any global `N_c(L, zeta)` law exists, or that any crossing found here is
the thermodynamic critical point.

---

## Not done here

`research/state/**` was not written. The predecessor archive was not modified.
`main` was not touched. Nothing was submitted, and nothing in this package can
submit.
