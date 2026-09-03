# TASK-2026-09-03-NC-PLATEAU-CALIBRATION

**Terminal state: `READY_FOR_HUMAN_SUBMISSION`.**
No agent submitted anything and no agent may — `research/RESOURCE_POLICY.md` §4,
unconditionally, at every stage and gate. `research/state/**` was not written.
No predecessor task directory was modified.

---

## The question

> **What is the smallest defensible `N_c` required to locate the transition at
> each relevant `L`, and does a usable asymptotic finite-`N_c` regime actually
> emerge?**

At `zeta = 0.35`, `T = L`, the certified guided-cloning sampler and
`OBS-CMI-001`.

## Read this first, if you read one thing

`[E]` **From the existing data alone, absolute-level plateau certification at
`L = 128` is unreachable at any affordable `R`.** The matched `R` needed to put
a `Delta` interval inside the frozen tolerance is ~2 675 at the `512 → 1024`
step — about 13 000 core-hours for one `lambda`.

`[I]` So the programme's question changes shape: not *how large must `N_c` be*,
but *which tolerance can we afford to certify against*. At `L = 64` the
absolute-level route is affordable and campaign A takes it. At `L = 128` it is
not, and the answer must come from the **crossing** tolerance — candidate C3,
untested, which campaigns B and B2 exist to measure.

That result cost no new compute. It came from rebuilding all 53 `N_c` ladders
from the 1 896 raw result files.

## What to submit

| campaign | what it answers | arms | tasks | core-h |
|---|---|---:|---:|---:|
| **A** | is a high-`N_c` plateau OBSERVABLE at `L = 64`? | 3 | 120 | 512 |
| **B** | does finite-`N_c` distort the SHAPE of `CMI(lambda)` at the locator? | 3 | 936 | 573 |
| **B2** | *(with B)* does the CROSSING converge before the absolute level? | 6 | 1 872 | 260 |
| **C** | does `L = 96` enter a simpler high-`N` regime? | 2 | 48 | 292 |
| **D** | is the `1024 → 2048` change at `L = 128` still material? | 1 | 16 | 502 |
| **E** | does the drift depend on the window count `K`? | 2 | 288 | 42 |
| | **total** | **17** | **3 280** | **2 180** |

3 052 core-hours pessimistic. 240 existing populations reused, never recomputed
(worth ~1 880 core-hours). Longest single job **31.4 h** (44.0 h pessimistic),
`D_L128_nc2048` — queue it first.

Exact commands: **`RUCHE_RUNBOOK.md`**. Per-arm gate table:
**`HUMAN_SUBMISSION.md`**.

## What is blocked

`[E]` Seven arms under `conditional/`, behind **three independent** mechanisms:
a separate directory no runbook loop enumerates; a hard interlock that exits 3
before importing the sampler unless a human-written release file exists; and a
preflight that refuses while the interlock is armed. Gates and triggers, all
pre-registered before any datum: **`CONDITIONAL_SUBMISSION.md`**.

## Two things that changed under attack, before submission

`[E]` **`R = 24 → 48` in campaign A.** At `R = 24` the top step could not have
satisfied its own frozen criterion whatever the data did. +166 core-hours.

`[E]` **Campaign B2 was rebuilt from three `lambda` to seven.** On three `lambda`
the frozen crossing protocol flags **both** interior crossings
`ENDPOINT_INDUCED` by construction — the exact defect
`TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION` existed to repair. +220 core-hours.

`[J]` Both are recorded at the line of code that changed, not only in a summary.

## Three corrections to inherited numbers

1. `[E]` **The cost model extrapolated the wrong way.** At `L = 128` the
   per-clone-window rate does **not** stay flat above `N_c = 256`; the inherited
   model is **30 % low** at `N_c = 1024`. Consequence: a conditional `L = 128`,
   `N_c = 4096` population is a **71.5 h** job against `cpu_long`'s 168 h
   ceiling, and `N_c = 8192` at `L = 128` is **not runnable at all**.
2. `[E]` **The `--mem` model was a model quoted as a measurement.** No `MaxRSS`
   from any Ruche job exists anywhere in this repository. Direct measurement
   here — the first for this sampler — shows it under-predicts, and that
   repeated probes of the same cell can differ by 1.8×. See `COST_MODEL.md` §4.
3. `[E]` **The `L = 96` half of "a clean `1/N` is rejected at `L = 96` and
   `L = 128`" does not reproduce.** `L = 128` reproduces exactly
   (`chi2 = 12.58`/3, `p = 0.0056`); the three-rung `L = 96` `lambda = 0.3032`
   ladder gives `p = 0.168`. Reported as an **open provenance item**, not a
   refutation.

## The gap this task could not close

`[E]` **Charter Stage 8 is not satisfied.** No independent investigator and no
independent red team ran. `validate_redteam.py` **refuses** the report under
rule R3, and `lead_summary_seen` was **not** set to false to make the check
green. `[J]` Every "survives" verdict in `REDTEAM.yaml` should be treated as
unreviewed. `[E]` Relatedly, **no external prior-art search was performed
anywhere in this task**, so external novelty for candidates C2 and C3 is
`UNRESOLVED`.

## Files

| file | what it is |
|---|---|
| `RECOMMENDATION.md` | the decision gate — read with `HUMAN_SUBMISSION.md` |
| `RESEARCH_MEMO.md` | Stage 9 synthesis, including the strongest case against this campaign |
| `CAMPAIGN_DESIGN.md` | the design, what is reused, and what is deliberately absent |
| `SUCCESS_CRITERIA.yaml` | **frozen** — `tau_lambda`, `tau_I`, P1–P5, H1–H3, E1–E2 |
| `ANALYSIS_SPEC.yaml` | **frozen** — estimators, exclusions, forbidden transformations |
| `FALSIFICATION_PLAN.md` | **frozen** — Y1–Y9 and six pre-registered negative outcomes |
| `FALSIFICATION_RESULTS.md` | outcomes; Y1–Y8 are *not yet attempted* and say so |
| `DECISION_TREE.md` | **frozen** — every branch, written before any datum |
| `COST_MODEL.md` | measured rates, the two corrections, and where the model is a conjecture |
| `EXISTING_POPULATION_INVENTORY.csv` | all 1 896 populations, rebuilt from raw files |
| `EXISTING_LADDERS.md` | all 53 `N_c` ladders |
| `REUSE_LEDGER.csv` | 84 cell decisions: reused, topped up, or fresh |
| `SEED_LEDGER.md` | seed allocation and the structural-disjointness proof |
| `INSTRUMENTATION.md` | what this campaign records that no predecessor did, and why it is safe |
| `QUALITY_DIAGNOSTICS.yaml` | every curve and crossing diagnostic, with the values already obtained |
| `VALIDATION.md` | every check, with results — including the one that fails |
| `SUBMISSION_DEPENDENCIES.md` | what may run together, and what dropping B2 costs |
| `RUCHE_RUNBOOK.md` | the commands, for the researcher to type |
| `CONDITIONAL_SUBMISSION.md` | the four gates and their pre-registered triggers |
| `INDEPENDENCE_LEDGER.yaml` | what independence this run obtained: none |
| `POST_FREEZE_EVENTS.md` | the two amendments, attributed |
| `PARKING_LOT.md` | ten things not pursued, and what would open each |
| `analysis/nc_plateau_analysis.py` | the ONLY place the frozen criteria are evaluated |
| `tools/` | reconstruction, cost and memory models, builders, dedup scan, smoke test, negative controls, reproduction check, full check suite |
| `shared/`, `support/` | the runtime, and the SHA-gated byte-identical certified sampler |

## Standing rules this task enforces on itself

- Uncertainty comes from **independent populations**. VIF is a
  variance-equivalence diagnostic and never a bias diagnostic. Founder count is
  never a number of independent samples.
- Finite-`N_c` movement is **drift**, never bias: the `N_c → ∞` target is
  unknown.
- **`N_c` and `R` are separate budgets** and every verdict names which one binds.
  `UNRESOLVED_R_LIMITED` is not a weaker form of "converged".
- **No smoothing**, no interpolation replacing a measurement, no imposed
  monotonicity, no removal of an inconvenient `lambda`, no value-based
  exclusion. The results file asserts this about its own run.
- No plateau inferred by eye. No forced `1/Nc` fit. No exponent from a
  pre-asymptotic ladder.
- `dtau_mult` is a **discretisation control**, never a physical parameter, and
  its non-6 rows may never be pooled with the production corpus.

## What this task may never conclude

`[E]` No `lambda_c(zeta)`, no boundary law, no exponent. The 0.2182–0.2482
window is an **observed locator region** in `L <= 64` curves at `N_c = 1024`,
not a critical window. No general `N_c_req(L, zeta, lambda)` rule.

`[E]` And the frozen theory result stays in its narrow form: *the standard
useful uniform-mixing Feynman–Kac bounds do not directly transfer to the
production mutation kernel, because the no-click branch is deterministic.* That
is the failure of a **proof route**. It is **not** "1/`N_c` convergence is
impossible", and nothing here upgrades it.
