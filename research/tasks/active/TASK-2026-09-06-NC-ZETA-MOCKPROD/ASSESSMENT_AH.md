# ASSESSMENT_AH — TASK-2026-09-06-NC-ZETA-MOCKPROD

Charter §5, Meaningful-Contribution Test. **Assessed A–H separately. No
aggregate score is produced and none may be derived from this file** — the point
of separating the dimensions is that they do not trade off. Labels `[E]` `[I]`
`[C]` `[J]`.

---

## A. Consequential bottleneck — **ANSWERED, and it is operational**

`[E]` The precise limitation: **every production population this programme has
is at `zeta = 0.35`**, and there is no basis anywhere for choosing `N_c` at any
other `zeta`. `[E]` Even at 0.35 the knowledge is partial — the `1024 -> 2048`
crossing shift there was `-0.0035` with bootstrap SE `0.0025–0.0028`, so `R` and
not `N_c` was the binding budget.

`[E]` What changes if solved: the `N_c` for the next production campaign at each
`zeta`, and whether one is affordable. `[J]` At `zeta = 0.70` the difference
between running at 256 and running at 2048 is an 8x cost factor on the most
expensive `zeta` in the phase diagram.

`[J]` **Verdict: answered concretely.** This is not "improving performance": it
is a specific number that a specific next campaign needs and does not have.

## B. Mechanistic contribution — **WEAK, and stated as weak**

`[E]` The task reveals **no mechanism**. It measures how a known estimator's
finite-population drift behaves at three parameter values.
`[C]` Whether that drift follows the `O(1/N)` structure known for particle
filters is a mechanistic question, and it is **not asked here**; it is parked.

`[I]` The one thing close to mechanistic is the observation, carried from
`COST_MODEL.md` §8, that **cost runs opposite to statistical difficulty in
`zeta`** — the cheapest `zeta` to survey is the one most likely to need a large
`N_c`. `[J]` That is a planning fact with a mechanistic flavour, not a mechanism.

`[J]` **Verdict: weak. Recorded as weak rather than dressed up.** This dimension
does not compensate for anything and is not compensated by A.

## C. Discriminability — **ANSWERED for what it claims**

`[E]` The four qualitative classes are pre-registered in a frozen
`ANALYSIS_SPEC.yaml` before any datum exists, with `INCONCLUSIVE` evaluated
**first** so a noisy rung cannot default to "stable".
`[E]` `tools/smoke_test.py` demonstrates on synthetic data with known answers
that the classifier separates all eight constructed cases, **including three it
is required to fail** — `INCONCLUSIVE` under injected noise, `ENDPOINT_INDUCED`,
`ABOVE_GRID`.

`[E]` The kill criterion is explicit: every rung `INCONCLUSIVE` everywhere kills
the product.

`[I]` **What it cannot discriminate**: it cannot separate "`N_c` is adequate"
from "`R = 16` is too coarse to see the inadequacy" beyond the `INCONCLUSIVE`
flag. That limit is the honest boundary of the design and is on the face of
every output row.

`[J]` **Verdict: answered for what it claims, and its limit is on the face of
every output row.** The classes are pre-registered and demonstrated to separate
constructed cases; what they cannot separate is stated rather than implied.

## D. Dependency significance — **MODERATE**

`[E]` It occupies a real dependency point: **every subsequent campaign at any
`zeta` other than 0.35 needs an `N_c`**, and today there is nothing to read it
from. `[E]` The reusable outputs — the corrected rate model, the `P9` that tests
grid **span** rather than centre, the 13 injected-fault controls — are inputs to
whatever comes next.

`[I]` It is not foundational. `[E]` It is a terminal measurement at three
parameter values, and it explicitly forbids extending itself into a law.

`[J]` **Verdict: moderate.** A real dependency point for any campaign off
`zeta = 0.35`, but a terminal measurement that forbids extending itself into a
law.

## E. Cross-silo value — **NONE CLAIMED**

`[E]` No fields are connected and **no `BRIDGE_AUDIT.md` is written**, because
no cross-field claim is made. `[E]` Sequential-Monte-Carlo population sizing is a
standard particle-filter problem and this task does **not** assert that its
answer transfers, generalises, or connects two formalisms. `[J]` Claiming a
bridge here would be the analogy-only connection the charter rejects.

`[J]` **Verdict: none, and none claimed.** Recorded as absent rather than
manufactured.

## F. Robustness — **PARTIALLY ANSWERED, with one named weak input**

`[E]` Survives realistic ranges by construction: three `zeta` spanning 7x, three
`L`, five `N_c`, nine `lambda` each, matched `R`, grids that bracket both open
positions of `DISP-PHI-001` with `>= 6.9 tau` to the nearer grid end.
`[E]` No narrow tuning regime: nothing is centred on a candidate boundary law,
and `P9` fails if a grid stops spanning both.

`[C]` **The named weak input is `f_lam` at `zeta = 0.10`**, a 4x extrapolation of
a `lambda^-0.5` rate law below its calibrated window. It affects `--time`, not
any scientific conclusion, on 5.1 % of the design, at 2.2x margin. `[E]` It is
recorded as this package's weakest input in `COST_MODEL.md` §5 rather than
smoothed over, and the investigator's proposed probe is carried to Gate A.

`[C]` **Memory sizing has never been measured on the cluster.** Every `--mem`
descends from local macOS `ru_maxrss`. `[E]` The runbook asks for the one
`sacct` line that would fix it for the whole programme.

`[J]` **Verdict: partially answered.** Robust across the parameter ranges it
spans and against boundary-law choice; two inputs -- `f_lam` at low `zeta`, and
all memory sizing -- are named as unmeasured rather than assumed sound.

## G. Informative failure — **STRONG, and this is the dimension that carries the task**

`[E]` Every failure mode teaches something the researcher needs:

| failure | what is learned |
|---|---|
| all rungs `INCONCLUSIVE` | `R = 16` is too coarse; `R`, not `N_c`, is the budget — the same lesson the `zeta = 0.35` anchor gave, now confirmed at three more `zeta` |
| still moving at `N_c = 2048` at some `zeta` | high-quality production at that `zeta` is not reachable on this ladder, before it is attempted rather than after |
| crossing `ABOVE_GRID` at `zeta = 0.70` | the boundary exponent is outside `[0.08, 1.13]`, which bears on `DISP-PHI-001` and is worth more than the sizing answer |
| the cost model overruns | the rate model is wrong again, and in which direction, on a campaign cheap enough to absorb it |

`[J]` **This is the strongest dimension.** A survey whose null outcome is
uninformative would not be worth 1 160 core-hours; this one's null outcomes each
redirect the next campaign.

`[J]` **Verdict: strong, and this is the dimension the task rests on.**

## H. Infrastructure value — **MODERATE**

`[E]` Produces: a rate model refitted from raw data and refuting the inherited
`N_c^0.1871` law, with the refutation re-checked on every preflight run; a
grid-coverage check that tests **span** against both open positions rather than
centre against one, which is the specific defect that stopped
`NC-ZETA-CALIBRATION`; 13 injected-fault negative controls; an 8-case smoke test
that demonstrates the classifier failing where it must; and the first production
data at any `zeta` other than 0.35.

`[E]` Reproducible: `tools/build_arms.py` regenerates the entire package
deterministically, and the preflight refits the cost literals from raw data and
fails on drift.

---

`[J]` **Verdict: moderate.** Reusable checks and a corrected rate model, plus
the first production data off `zeta = 0.35`.

## Dimensions that are weak, restated so they are not lost

`[J]` **B is weak** (no mechanism) and **E is absent** (no cross-silo value).
Neither is compensated by A, G or H, and the charter forbids letting them be.
`[J]` A reader deciding whether this task is worth the researcher's attention
should weigh A and G, and should not be told that H makes up for B.
