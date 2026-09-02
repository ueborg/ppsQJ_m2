# M1–M7 — pre-registered, FROZEN before any new datum exists

TASK-2026-09-02-MOCK-PRODUCTION, brief §10, §11 and §13.
The machine-readable form is `analysis_spec.yaml`; this file is its prose.
**Written before submission. Do not edit after data arrives.**

Each criterion finishes as exactly one of **SUPPORTED / KILLED / INCONCLUSIVE**.
A criterion whose data never arrived returns **NOT EVALUATED**, which is
deliberately *not* one of the three: an unsubmitted arm must never read as a
scientific verdict.

## What the analysis may never do, stated first

- **No smoothing.** No moving average, spline, LOESS or kernel.
- **No regularisation**, no shrinkage toward a fitted form.
- **No fitting away** a point, and no reporting a fit in place of the points. A
  weighted quadratic appears only as a *yardstick* for chi2/dof, and is never
  plotted instead of the data.
- **No discarding a point because it looks jagged.** The only exclusions are the
  four frozen ones in `analysis_spec.yaml`, all about run status and non-finite
  clones, none of which look at the observable's value.
- **No imposed monotonicity.** CMI(lambda) is expected to fall across this
  window, but the physics is not invoked to *require* it, and a non-monotone
  triple is a result rather than an error.

A jagged curve that survives every test below is a finding. A smooth-looking
curve produced by any of the operations above is not.

## The matched-R rule — read this before any criterion below

**Every criterion below is evaluated at a matched `R = 24` independent
populations per `(L, lambda)` cell.** Cells holding more are cut into
consecutive disjoint blocks of 24 **in seed order**, and **block A** — the 24
lowest seeds — is always the primary dataset:

```
reused ARM-B cells   R = 96  ->  blocks A B C D
N_c = 128 comparator R = 48  ->  blocks A B
everything else      R = 24  ->  block A only
```

The rule exists because roughness, adjacent-increment significance, second
differences, split-half stability and crossing counts all measure a curve's
scatter against its own error bars, and those error bars scale as `1/sqrt(R)`.
Without it, *"the `N_c=1024` curve is cleaner than the `N_c=128` curve"* could
be partly a statement about `R` rather than about `N_c`. **[E]** On the
synthetic end-to-end dataset it is not hypothetical: M3 returns INCONCLUSIVE
under unequal `R` and KILLED at matched `R`.

Block membership is **deterministic and observable-blind** — fixed by sorting on
the seeds before any statistic is computed — and that property is asserted by
`tools/test_matched_r.py`, not assumed. Full statement:
`MATCHED_R_AMENDMENT.md`.

**Secondary views** — full `R = 48` and `R = 96` means, replicate blocks B/C/D,
and `R = 12` historical-precision subsets — are reported separately and carry
**no** authority over any cleanliness or crossing claim. For **mean** finite-`N_c`
displacement, both the matched-`R` and the highest-precision result are
reported, with the unequal `R` named in the line itself.

### The heteroscedasticity this removes

The original design had ten `L = 64` points at `R = 24` beside three at
`R = 96`, with SEMs about half their neighbours'. Under the amendment the
primary analysis is **uniformly `R = 24` at every cell of every curve**, so
those per-point errors are equal by construction. Per-point standard errors are
still used everywhere — the rule is kept because it still binds the secondary
full-`R` views — but the primary analysis no longer has a precision mismatch for
a roughness statistic to misread as structure in the curve.

---

## The quantities (brief §10)

At each `L`, on the 13-point grid, from means over **independent populations**
with **across-population** SEMs:

```
d_i = I(lam_{i+1}) - I(lam_i),          i = 1..12
SEM(d_i) = sqrt(SEM_i^2 + SEM_{i+1}^2)
r_i = |d_i| / SEM(d_i)

q_i = I_{i+1} - 2 I_i + I_{i-1},        i = 2..12
SEM(q_i) = sqrt(SEM_{i-1}^2 + 4 SEM_i^2 + SEM_{i+1}^2)

ROUGH = mean_i (q_i / SEM(q_i))^2       with a bootstrap CI
```

`SEM(d_i)` adds in quadrature because every lambda has its own **disjoint seed
lane**, so the point estimates are statistically independent. That is a property
of the seed allocation (`SEED_LEDGER.md`), not an assumption.

Alongside `ROUGH`, chi2/dof of the 13 points against their best-fitting weighted
quadratic is reported, because that is the same statistic, on the same
observable, that the predecessor computed on the historical corpus, where it
gives **0.60–1.38** at `L = 64…128`. It is the one directly comparable number
between the old scan and the new one.

Per cell: split-half over independent populations; the maximum standardized
population deviate `z_max` against `z_R = Phi^-1(1 - 0.01/(2R))`; and the shift
in the cell mean from deleting the single most extreme population. In the
primary analysis every cell is `R = 24`, so the threshold is `z_24 = 2.81`
uniformly (`z_48 = 3.03` and `z_96 = 3.29` appear only in secondary views).

## The crossing protocol (brief §11)

For each `L`-pair and each `N_c` class, on `D(lambda) = I_{L2} - I_{L1}`:

| reported | rule |
|---|---|
| raw sign changes | adjacent grid pair with `D_i * D_{i+1} < 0` |
| resolved sign changes | a raw one where **both** `\|D\|/SEM_D >= 2` |
| crossing lambda(s) | linear interpolation of `D` between the bracketing points |
| crossing uncertainty | bootstrap over independent populations, `B = 10000`, 2.5/97.5 percentiles |
| unique | exactly one raw sign change **and** ≥95 % of bootstrap replicates give exactly one |
| endpoint-induced | the sign change involves grid point 1 or 13, or the crossing lies within `delta_lambda/2` of an endpoint |
| stable to bootstrap | 95 % interval width ≤ `2 * delta_lambda = 0.020` |
| stable to point deletion | all 13 leave-one-lambda-out crossings fall inside the full-data 95 % interval |

Bootstrap replicates that produce **zero** crossings or **more than one** are
counted and printed, never silently dropped; their fraction is part of the
uniqueness verdict. The distinction between *raw* and *resolved* sign changes
matters: a difference curve that merely wanders across zero inside its own error
bars generates raw sign changes without generating a locator.

**CMI is the locator.** An endpoint-induced crossing is reported as a crossing
and is never counted as a located one.

**The output of this section is locator quality, not `lambda_c(zeta)`.** See the
closing section.

---

## M1 — reproducible curves, no isolated population driving a point

**Statistic.** Split-half, `z_max` and leave-one-population-out, at every
`(L, lambda)` of the three `N_c = 1024` curves, on the primary matched-`R = 24`
block A. `z_R` is therefore `z_24 = 2.81` uniformly.

- **SUPPORTED** — everywhere: `|m_A - m_B| <= 2.5 s_AB`, `z_max <= z_R`, and
  deleting the most extreme population shifts the cell mean by ≤ 1 SEM.
- **KILLED** — split-half fails at ≥ 2 lambdas within any single `L`, **or**
  leave-one-out fails anywhere.
- **INCONCLUSIVE** — anything else, including isolated single-lambda failures.

**[E]** ARM B passed the equivalent test at all three of its lambdas at `R = 96`
(`z = 0.05, 0.09, 0.22`; `z_max = 2.43, 2.65, 3.12` against 3.88). **[C]** At
`R = 24` the thresholds tighten and 39 cells are tested rather than 3, so
isolated failures are entirely possible and INCONCLUSIVE is a realistic outcome.

## M2 — the spacing resolves without being unnecessarily fine

**Statistic.** `r_i` over the 12 adjacent increments, per `L`, on the primary
matched-`R = 24` block A.

- **SUPPORTED** — `median_i r_i` in `[2, 20]` at every `L`, **and** `r_i >= 2`
  for at least 9 of the 12 increments at every `L`.
- **KILLED** — `median_i r_i < 2` at any `L` (spacing finer than the achievable
  resolution), **or** `median_i r_i > 20` at all three `L` (unnecessarily
  coarse).
- **INCONCLUSIVE** — anything else.

**Pre-registered projection** (`POWER_AND_R_DECISION.md`): `r` runs 4.1 at the
low-lambda end to 6.5 at the high end at `L = 64, R = 24`, and the projection is
optimistic by 2–17 % where ARM B lets it be checked, so a realistic floor is
`r ≈ 3.4`. Recording the projection now means a large disagreement is itself
reportable.

## M3 — cross-L locator structure materially cleaner than the old N_c = 128 scan

**Statistic.** The crossing protocol run on the `N_c = 1024` curves and on the
**matched** `N_c = 128` companion curves — same grid, same `dtau_mult`, same
estimator, same code, **and the same `R = 24`** — only `N_c` differing. This is
the amendment's primary case: without matched `R`, a lower sign-change count at
one `N_c` could be bought by that side simply having more independent
populations. The verdict string printed by the analysis carries the words
*"at MATCHED R = 24 on both sides"*, so the qualifier travels with the number.

- **SUPPORTED** — for every `L`-pair the raw sign-change count at `N_c = 1024`
  does not exceed that at `N_c = 128`; summed over the three pairs it is at most
  half of it; and at least one pair yields a unique, non-endpoint-induced,
  bootstrap-stable crossing at `N_c = 1024`.
- **KILLED** — the summed raw sign-change count at `N_c = 1024` is ≥ that at
  `N_c = 128`.
- **INCONCLUSIVE** — anything else, **including the case where the companion arm
  was not run.**

That last clause is load-bearing and is coded, not merely written down. The
historical corpus has no `L = 32`, no `L = 48`, a different lambda grid and
`dtau_mult = 12`; comparing against it would not be a matched comparison, and
the analysis **refuses** to and prints the refusal. See
`NC128_COMPANION_RATIONALE.md`.

## M4 — split-half analyses return compatible curves and crossings

**Statistic.** Whole-curve split-half and 20 frozen leave-half-out partitions,
plus the crossing protocol run on each half — all on the primary matched-`R = 24`
block A.

- **SUPPORTED** — `max |I_A - I_B| / s_AB <= 3.0` over all `(L, lambda)`, **and**
  for every `L`-pair where both halves give a unique crossing,
  `|lam_A - lam_B| <= 2 sqrt(u_A^2 + u_B^2)`.
- **KILLED** — the per-point condition fails.
- **INCONCLUSIVE** — the per-point condition holds but the crossing condition
  fails or is undefined because a half gives no unique crossing.

## M5 — does the finite-population correction vary with lambda?

**Statistic.** `Delta_N(lambda) = I_{N_c=2048} - I_{N_c=1024}` at the three
central lambdas; the chi-square of the three against a constant on 2 dof with a
bootstrap null; and the 95 % bound on `|slope| * delta_lambda`.

**Primary at matched `R = 24` on both sides.** The `N_c = 1024` centre is cut
from its `R = 96` to block A rather than being allowed to be four times as
precise as the `N_c = 2048` arm it is subtracted from — otherwise `SEM(Delta_N)`
would be dominated by one side and the chi-square against a constant would
inherit that asymmetry. **The A/B/C verdict comes from the matched analysis
only.** The mean displacement is *also* reported at the highest precision
available (`R_1024 = 96` against `R_2048 = 24`), labelled UNEQUAL R, because a
mean is not a cleanliness statistic and the extra precision on it is real.

**More `R` is not less finite-`N_c` bias.** Extra populations shrink the
uncertainty on `Delta_N`; they do not shrink `Delta_N`.

| verdict | rule |
|---|---|
| **A** approximately a common shift | `p > 0.32` **and** the 95 % bound on `\|slope\|*delta_lambda <= tau_shape` |
| **B** appreciably lambda-dependent | `p < 0.05` |
| **C** unresolved | anything else |

`tau_shape = 0.2 * median |d_i|` at `L = 64, N_c = 1024`, computed from the
returned data rather than assumed: a lambda-dependence smaller than a fifth of
one grid step cannot appreciably change the curve's shape.

- **SUPPORTED** — the arm returns A or B, i.e. the question is decided.
- **KILLED** — the arm returned data but the shape check cannot be evaluated
  (cell excluded, or fewer than three usable cells).
- **INCONCLUSIVE** — verdict C.

**The bootstrap null is built by shifting each `N_c = 2048` cell's populations
so that `Delta_N` really is constant, then resampling.** Resampling the observed
populations directly would carry the observed lambda-dependence into the "null",
inflate `p`, and systematically hide exactly the effect M5 exists to detect.
This was a real bug in the first draft of the analysis; it was caught by the
synthetic end-to-end test recorded in `VALIDATION.md` §7 and is fixed.

**[E]** With `R = 24` the arm resolves a lambda-variation of `Delta_N` of 0.0176
at 2 sigma, against one lambda-grid step in CMI of 0.037–0.050. **[C]** So
verdict C is a realistic outcome for a genuinely small tilt, and it is
pre-registered as such rather than discovered afterwards.

No `1/N_c` law is fitted. `Delta_N` is not extrapolated to `N_c = infinity`.
`TASK-2026-08-31-SMCCERT` killed that claim.

## M6 — no smoothing or post-hoc point removal was required

**Statistic.** The analysis script's own audit fields, written into
`MOCK_PRODUCTION_RESULTS.json`: `smoothing_applied`, `value_based_exclusions`,
`lambda_points_removed`.

- **SUPPORTED** — all three are `false / 0 / 0` and every verdict above was
  reached from the unmodified points.
- **KILLED** — any stated result required removing a lambda point or smoothing.
- **INCONCLUSIVE** — not reachable; this criterion is procedural and binary.

**This is expected to pass.** It is pre-registered anyway, because the failure
it guards against is invisible in the output if it is not recorded at the time.

## M7 — are these settings realistic for larger L?

**Statistic.** Re-derive the `L`-scaling exponent from the **four** measured
Ruche rates that will then exist (`L = 32, 48, 64` from this campaign,
`L = 128` from ARM A1024) and project a full 13-point, `N_c = 1024`, `R = 24`
scan at `L = 96` and `L = 128`.

- **SUPPORTED** — the re-derived model reproduces each of its four measured
  anchors to within 15 %, **and** the projected `L = 96` campaign has elapsed
  ≤ 24 h at a %64 cap.
- **KILLED** — the projected `L = 96` campaign has elapsed > 72 h at %64, i.e.
  this configuration does not scale to the next rung.
- **INCONCLUSIVE** — anything else, including a model that cannot reproduce its
  own anchors to 15 %.

**[J]** This is the criterion that pays for the next campaign's budget
conversation. Today, `L = 96` can only be costed from a two-point extrapolation;
after this campaign returns it can be costed from four measured rates spanning a
factor of four in `L`. That is also why `L = 80` was rejected here rather than
guessed at — see `L80_RUNTIME_GATE.md`.

---

## What none of this may conclude

This is an **algorithm-validation experiment**. It is allowed to say

> at low sizes, the high-`N_c` curves are clean and the crossing machinery
> behaves robustly

and it is **not** allowed to say

> therefore the physical critical point is X

or

> therefore the finite-zeta exponent is Y.

`L = 32, 48, 64` sit at or below the programme's own corpus floor of `L = 64`.
No crossing found here is `lambda_c(zeta = 0.35)`, a finite-size estimate of it,
or an input to a boundary law — **however unique, stable and well-located it
turns out to be.** No global phase-boundary law is fitted. The lambda range came
from the measured local behaviour of the existing `zeta = 0.35` corpus and
explicitly not from a presumed critical law.

The full list is `prohibited_conclusions` in `analysis_spec.yaml`, and the
analysis prints it at the end of every run.
