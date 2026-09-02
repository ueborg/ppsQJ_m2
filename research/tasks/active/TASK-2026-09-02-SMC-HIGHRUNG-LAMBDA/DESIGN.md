# Design — one overnight campaign, three parallel arms

TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA.

A targeted numerical follow-up to `TASK-2026-09-01-SMCRUCHE-READY`, not a broad
research campaign. It asks two connected questions and prepares independent
Slurm arrays so they can be answered in one night.

---

## The two questions

**A — how high must `N_c` go at L = 128 before the mean stops moving?**
ARM2 established that the mean is still drifting hard through `N_c = 256`, by
−0.0990 (64→128) and then −0.1213 (128→256): the second doubling moves it
*more* than the first. The ladder is extended to `N_c ∈ {512, 1024}` and tested
by **direct rung-to-rung differences**, not by a `1/N_c` fit.

**B — is the jaggedness of `CMI(lambda)` physical, or is it sampling noise?**
A symmetric three-point stencil at `lambda ∈ {0.2932, 0.3032, 0.3132}` is run at
a cheap L with a large population, and again at L = 128. The purpose is to find
the **local resolution needed** for adjacent points to trace the curve — not to
infer anything about the phase boundary.

---

## Frozen design constants

| | |
|---|---|
| zeta | 0.35 |
| lambda stencil | 0.2932, **0.3032**, 0.3132 — `delta_lambda = 0.010`, symmetric |
| dtau_mult | 6.0 (certified; the historical corpus is 12.0 and is not poolable) |
| resampling | systematic, every window |
| observable | `OBS-CMI-001`, quarter-system CMI |
| statistical unit | the independently seeded population realization |

The spacing choice and its evidence are in `LAMBDA_SPACING_DECISION.md`. It was
frozen before any new result exists, and `shared/preflight.py` refuses any
manifest that drifts off it.

---

## The arms

| arm | package | L | T | N_c | lambda | R | tasks | core-h | slowest task |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|
| **A** | `armA512` | 128 | 128 | 512 | 0.3032 | 48 | 48 | 241 | 5.03 h |
| **A** | `armA1024` | 128 | 128 | 1024 | 0.3032 | 32 | 32 | 322 | 10.06 h |
| **B** | `armB` | 64 | 64 | 1024 | all three | 96 | 288 | 167 | 0.60 h |
| **C** | `armC` | 128 | 128 | 512 | 0.2932, 0.3132 | 48 | 96 | 483 | 5.20 h |
| | | | | | | | **464** | **1213** | |
| *opt.* | `armA2048_optional` | 128 | 128 | 2048 | 0.3032 | 16 | 16 | 322 | 20.12 h |

Split into two arrays for ARM A because a 5 h rung and a 10 h rung want
different wall requests, and over-requesting on the short one wastes queue
priority.

### ARM A — the L = 128 high-N_c central ladder

`R` was chosen per rung from a power calculation, not inherited. `R = 64` was
*not* kept: at `N_c = 512` a job costs 5 h, so `R = 64` would spend 322
core-hours to move `SEM(Delta_256->512)` from 0.0231 to 0.0217 — a 6 %
improvement for a 33 % cost increase. `R = 48` and `R = 32` are the smallest
values that still answer whether the mean is moving:

| | `Delta_256->512` | `Delta_512->1024` |
|---|---:|---:|
| SEM | 0.0231 | 0.0225 |
| 95 % half-width | 0.0453 | 0.0442 |
| MDE (80 %, two-sided) | 0.0648 | 0.0632 |

against the observed per-doubling drift of ~0.10–0.12, i.e. ~4–5 sigma of power
if the drift continues.

Uncertainty comes from independent populations throughout. Within-clone spread
is reported as VIF/`N_eff` and is never a standard error. Per-clone CMI arrays,
`n_distinct_anc_final`, `gess`, `ess_cum` and `brentq_fallbacks` are all
preserved by `run_cell.py`.

**The primary question is stabilisation of the mean.** F1 and F2 are direct
rung-to-rung differences with bootstrap CIs over independent populations, judged
against `tau_step = 0.0732` — one lambda-grid step in CMI at L = 128, so the
tolerance is tied to the plotting resolution that motivates the whole campaign.
`gamma` still gets its frozen ≥3-window scan (F3), and is explicitly barred from
deciding mean convergence.

### ARM B — the low-L high-population stencil

**L = 64 was chosen over 48 and 96 on information per core-hour.** At L = 96 a
`N_c = 1024` job costs 3.0 h and the full stencil 578 core-hours — more than ARM
A entire, for the *cheap* arm. At L = 64 the same population costs 0.60 h and
the full stencil 167 core-hours. L = 48 would be cheaper still but has no
completed cloning run anywhere in the programme to calibrate against, whereas
L = 64 has SMCSTAT's `A-HV` ladder (`N_c` 32→256 at T = 32), which is what the
variance projection is built on.

**`N_c = 1024` was chosen from that calibration, not copied from the brief's
suggestion.** `A-HV` at L = 64 shows the mean flattening by `N_c = 256`
(0.6467 → 0.5175 → 0.4326 → 0.4077, so the last doubling moves it by −0.025),
and the variance falling from 3.89e-2 to 3.32e-3. At `N_c = 1024` — two
doublings past where that ladder had largely settled — a job still costs only
0.60 h. Paying for the margin is nearly free here, and the arm's entire purpose
is to suppress finite-`N_c` artefacts far enough that the lambda dependence
itself is visible. `N_c = 512` would have saved 84 core-hours and given up that
margin.

`R = 96` per lambda gives a projected per-point SEM of 0.0056, so `d_-` and
`d_+` come in at ~7 sigma. It also splits cleanly into two disjoint halves of 48
for the S1 reproducibility test.

### ARM C — the L = 128 neighbouring-lambda stencil

Only `lambda_-` and `lambda_+` are run. **The central point is `armA512`'s** —
same L, T, `N_c`, `dtau_mult`, scheme and `R` — and is assembled across arms by
`analysis/combined_analysis.py`. That reuse saves 48 tasks and 241 core-hours,
and `preflight.py`'s `R equal across lambdas` check keeps the stencil balanced.

`N_c = 512` is the smallest defensible choice for a first high-L local-curve
test: it is the first rung beyond ARM2's ladder, so it is the cheapest `N_c`
about which this campaign will have *any* direct convergence evidence (from F1).
`N_c = 1024` would have tripled the arm to ~1450 core-hours and `N_c = 2048`
would have made it the largest thing in the campaign, neither of which the
cost-benefit supports for a *first* test of whether the stencil approach works
at high L at all.

---

## Parallelism: what is bought and what is given up

All three arms are **independent Slurm arrays with no scientific dependency
between them**. They may be submitted together. Concretely:

- ARM C is **not** conditional on ARM B passing. A sequential design would have
  gated it: if ARM B's S1 fails at the cheap L, spending 483 core-hours on the
  expensive L to learn the same thing is poor value. **Running in parallel
  sacrifices that gate to save wall-clock time.** The researcher asked for
  wall-clock; this is the trade, stated so it is a choice rather than an
  oversight. If ARM B comes back with S1 failing, ARM C's cost bought the same
  answer at high L — not worthless, but not what a gated design would have paid.
- ARM A's two rungs are also independent of each other. `armA1024` does not wait
  for `armA512`; F2 simply needs both.
- The predecessor package's own stopping rule ("run ARM 1 first, stop ARM 2 if
  gamma comes back KILLED") does **not** carry over. It was about whether an
  L = 128 bias calibration was still meaningful. ARM1's gamma came back
  SUPPORTED, so the condition never fired, and this campaign's arms answer
  different questions.

**Recommended for simultaneous submission tonight: `armA1024`, `armA512`,
`armB`, `armC`.** Queue `armA1024` first — its 10 h single task is the campaign's
long pole.

**Not recommended tonight: `armA2048_optional`** (`NC2048_AUDIT.md`).
**Not prepared: an L = 96 bridge** (`L96_BRIDGE_AUDIT.md`).

Elapsed time depends on whether the allocation gives 64 slots per array or 64 in
total; both readings are worked out in `COST_MODEL.md` §"Elapsed wall-clock", and
in the 64-in-total case the runbook says to hold `armC` back to tomorrow.

---

## Budget

**1213 core-hours recommended** (1699 pessimistic), **+322 optional**.

This is ~4.3× the parent SMCCERT campaign's 280 core-hour cap. That cap was
that task's own frozen stopping rule and does not bind this one, but the
scale-up is a **human budget decision**, and it is flagged here and in
`analysis_spec.yaml` rather than absorbed silently.

---

## What this task does not do

- It does not modify any predecessor task archive.
- It does not write `research/state/**`.
- It does not submit anything. Every arm terminates at
  `READY_FOR_HUMAN_SUBMISSION`; no script in the package contains a scheduler
  call, and `preflight.py` asserts that about `run_preflight.sh`.
- It does not change the production path in any way — see
  `PRODUCTION_PATH_UNCHANGED.md`.
- It does not touch `main`.
