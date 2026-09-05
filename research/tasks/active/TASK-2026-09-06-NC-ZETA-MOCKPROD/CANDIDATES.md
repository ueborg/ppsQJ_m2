# CANDIDATES — TASK-2026-09-06-NC-ZETA-MOCKPROD

Charter Stage 3. **Frozen at `stage_3_candidates`.** Five candidates, eleven
fields each. Labels `[E]` `[I]` `[C]` `[J]`.

`[E]` The candidates here are **design options for the survey**, not physical
hypotheses. The brief fixes the question; what remains genuinely open is how to
spend 1 000–3 000 core-hours answering it, and those choices are where this task
can be wrong.

`[E]` **No novelty is claimed for any candidate.** `NOVELTY_GATE.md` classifies
the task as replication at new `zeta`.

---

## Candidate C1 — the committed design (SURVIVES, and is what ships)

1. **Statement.** Measure the full `N_c` ladder `{128, 256, 512, 1024, 2048}` at
   `zeta in {0.10, 0.20, 0.70}`, `L in {32, 48, 64}`, `T = L`, nine `lambda` per
   `zeta`, `R = 16` matched everywhere, `dtau_mult = 6`; hold `zeta = 0.70,
   N_c = 2048` behind an interlock; add a two-leg `dtau` control at `zeta = 0.10`.
2. **Strongest affirmative case.** `[E]` It is the brief's design, it costs
   1 160 committed core-hours against a predecessor's 2 180 for one `zeta`, it
   answers the per-rung question at every `zeta` with matched `R`, and its most
   expensive single rung — 42.6 % of the design — is separable and reviewable
   before it is spent.
3. **Closest known precedent.** `[E]` `NC-PLATEAU-CALIBRATION`, the same ladder
   at `zeta = 0.35`. `NOVELTY_GATE.md` §3: replication at new `zeta`.
4. **Strongest novelty objection.** `[E]` None is needed: nothing is claimed
   novel. `[J]` The design is a routine application of an established ladder to
   three new parameter values — Slop Warning №1 on its face. It survives only
   because the parameter values are where the programme has *no* data at all,
   not because the method is interesting.
5. **Strongest correctness objection.** `[C]` `f_lam` at `zeta = 0.10`,
   `lambda = 0.040` is a 4x extrapolation of a `lambda^-0.5` rate law below its
   calibrated window; if it is wrong the `zeta = 0.10` `--time` limits are wrong.
   `[E]` Mitigated: those arms carry 2.2x or better margin, they are 5.1 % of the
   design, and the packs are idempotent so a timeout costs only the unfinished
   rows.
6. **Strongest practicality objection.** `[E]` 2 557 array tasks across 15 arms
   is real scheduler load, and `M_z070_nc1024` alone is 408 tasks at 6 h.
   `[E]` Mitigated by packing to a 600 s floor and by `%64`; queue wait is
   excluded from every estimate and stated to dominate the short arms.
7. **Strongest significance objection.** `[E]` `ROUGHLY STABLE` at `R = 16` is a
   statement about this survey's resolution, not about convergence, and a reader
   who wants a production guarantee gets nothing they can use.
   `[I]` Correct, and it is why the recommendation table carries a `caveat`
   column and why `INCONCLUSIVE` is a first-class outcome.
8. **Decisive test.** `[E]` The pre-registered rung classification in
   `ANALYSIS_SPEC.yaml`, applied blind to whatever returns.
9. **Kill criteria.** `[E]` Every rung at every `zeta` returns `INCONCLUSIVE` —
   i.e. `R = 16` cannot separate adjacent-rung movement from noise anywhere.
10. **What survives criticism.** `[E]` The design as built. The objections bite
    on interpretation, and each is answered by a pre-registered wording limit
    rather than by a change to the measurement.
11. **Revised / stronger version.** `[I]` If the researcher wants one thing
    added, it is the investigator's `zeta = 0.10, lambda = 0.040` timing probe
    (minutes, local) before submission, which would replace this design's weakest
    input with a measurement. Carried to `RECOMMENDATION.md`, not run here.

---

## Candidate C2 — `R = 24` instead of `R = 16` (KILLED)

1. **Statement.** Run the same design at `R = 24`, the `zeta = 0.35` campaigns'
   value, lowering every SEM by 1.22x.
2. **Strongest affirmative case.** `[E]` `R = 24` is what produced the reference
   curves, so the new curves would be directly comparable in noise level, and
   `INCONCLUSIVE` verdicts become less likely.
3. **Closest known precedent.** `[E]` `MOCK-PRODUCTION` and
   `MOCK-LOWLAMBDA-EXTENSION`, both `R = 24`.
4. **Strongest novelty objection.** Not applicable.
5. **Strongest correctness objection.** `[E]` None — `R = 24` is strictly better
   statistically. This candidate dies on cost, not on correctness.
6. **Strongest practicality objection.** `[E]` **3 029 core-hours full design
   against 2 019**, i.e. **+1 010 core-hours (+50 %)**, and the committed part
   rises from 1 160 to 1 740. `[E]` The brief permits `R = 24` only if the campaign
   "remains comfortably affordable"; +50 % on a survey that explicitly disclaims
   precision is not that.
7. **Strongest significance objection.** `[E]` It buys a 1.22x SEM reduction. The
   `zeta = 0.35` anchor showed that separating a `1024 -> 2048` crossing shift
   needs `R` around 51–65 — so `R = 24` does not reach the precision regime
   either. `[I]` It buys a better *survey*, not a different *kind* of answer.
8. **Decisive test.** `[E]` Cost model versus the brief's affordability clause.
9. **Kill criteria.** `[E]` Killed if the incremental core-hours exceed what a
   1.22x SEM reduction can justify for a qualitative product.
10. **What survives.** `[E]` **Killed.** Recorded, not discarded: if the survey
    returns `INCONCLUSIVE` broadly, `R` — not `N_c` — is the budget to raise, and
    the follow-up should raise it to 48+, not to 24.
11. **Revised / stronger version.** `[I]` Raise `R` only at the one `zeta` and
    the one rung pair the survey shows to be marginal, as a child task. **Not**
    uniformly and **not** in advance.

---

## Candidate C3 — drop `zeta = 0.70`, spend it on `zeta = 0.10` and `0.20` (KILLED)

1. **Statement.** `zeta = 0.70` is 84 % of the cost. Drop it and run `0.10` and
   `0.20` at `R = 48`, or add more `zeta` below 0.35, for the same money.
2. **Strongest affirmative case.** `[E]` 1 699 of 2 022 core-hours buys one
   `zeta`. `[I]` `(1-zeta)^2` says the sampler's statistical work is *least* at
   high `zeta`, so it is plausibly the `zeta` least in need of a large `N_c` —
   the most expensive cell answering the most predictable question.
3. **Closest known precedent.** `[E]` `NC-ZETA-CALIBRATION` scoped its high-`zeta`
   points as diagnostic cells rather than full screens, for exactly this reason.
4. **Strongest novelty objection.** Not applicable.
5. **Strongest correctness objection.** `[E]` "Predictable" is an inference from
   a prefactor, and `NC-ZETA-CALIBRATION` established that the prefactor is
   **not** the whole `zeta` dependence — accumulated weight variance is constant
   to 30 % over a 14x range in `zeta`, against 14–21x for the prefactor alone.
   `[I]` So the belief that high `zeta` is easy is exactly the belief the survey
   exists to test, and dropping it assumes the answer.
6. **Strongest practicality objection.** `[E]` None; it is the cheap option.
7. **Strongest significance objection.** `[E]` The brief's stated purpose is to
   see how the behaviour "changes between low and high `zeta`". With one high
   point removed there is no high end, and the deliverable becomes a two-point
   low-`zeta` survey that cannot answer its own question.
8. **Decisive test.** `[E]` Whether the design can still answer §1 of the brief.
9. **Kill criteria.** `[E]` Killed if removal makes a commissioned question
   unanswerable.
10. **What survives.** `[E]` **Killed as stated.** `[I]` Its real content is
    preserved as the conditional interlock: the *rung* most exposed to this
    objection — `zeta = 0.70, N_c = 2048`, 862 core-hours — is held back and
    released only if the `512 -> 1024` step at that `zeta` still moves.
11. **Revised / stronger version.** `[E]` That interlock, which is what ships.

---

## Candidate C4 — drop `L = 32` and run only the `L = 48, 64` pair (KILLED)

1. **Statement.** The primary crossing pair is `L48-L64`. `L = 32` is 5.9 % of
   the cost and is not in the primary pair.
2. **Strongest affirmative case.** `[E]` Saves 119 core-hours across the design
   and shortens every arm.
3. **Closest known precedent.** `[E]` `NC-PLATEAU-CALIBRATION` ran `L = 32` arms
   (`B2_*`) specifically as supporting information.
4. **Strongest novelty objection.** Not applicable.
5. **Strongest correctness objection.** `[E]` **`L = 32` is the only diagnostic
   that shows small-`L` drift.** At `zeta = 0.35` the `L32-L48` crossing came out
   `STILL_BOUNDARY` while `L32-L64` and `L48-L64` came out `INTERIOR` — a
   disagreement that is only visible because all three `L` exist. `[I]` With two
   `L` there is one crossing and no way to tell a locator from an artifact of the
   pair chosen.
6. **Strongest practicality objection.** `[E]` `L = 32` is the cheapest `L` in
   the design; the saving is the smallest available per unit of information lost.
7. **Strongest significance objection.** `[E]` The brief asks for `L = 32`
   explicitly, as supporting information on crossing quality.
8. **Decisive test.** `[E]` Whether crossing-quality classification survives with
   two `L`. It does not: interiority and pair-disagreement both need three.
9. **Kill criteria.** `[E]` Killed if removal eliminates a diagnostic with no
   substitute.
10. **What survives.** `[E]` **Killed.** `L = 32` stays, and is reported as
    supporting information, never as a primary crossing.
11. **Revised / stronger version.** None. The saving is not worth having.

---

## Candidate C5 — use the historical `sqrt(zeta)`-centred corpus as the low-`N_c` baseline (KILLED)

1. **Statement.** Rather than measuring `N_c = 128` and `256` fresh, use existing
   historical scans at these `zeta` as the low-`N_c` end of each ladder.
2. **Strongest affirmative case.** `[E]` It would remove **six arms and 227
   core-hours, 11.3 % of the design** -- the second largest saving available
   anywhere in it -- and the historical corpus does contain `zeta = 0.1, 0.2,
   0.7` rows.
3. **Closest known precedent.** `[E]` `MOCK-LOWLAMBDA-EXTENSION` faced the same
   temptation and refused it for the `N_c = 2048` arm, on the same grounds.
4. **Strongest novelty objection.** Not applicable.
5. **Strongest correctness objection.** `[E]` **Fatal.** The whole-repository
   scan finds **zero** exact-compatible stored populations at these `zeta`. What
   exists is `results/boundary_aggregate.csv`, which carries **no `N_c`, `T`,
   `dtau_mult` or resampling-scheme column** and a different `lambda` grid, and
   nine architecture-benchmark JSONs at `N_c` 44–500 with no `resample_scheme`.
   `[I]` Pooling any of it would put an unknown-`N_c`, unknown-discretisation
   measurement at the bottom of a ladder whose entire purpose is to vary `N_c`
   with everything else held fixed. The rung difference would then be
   uninterpretable in exactly the way the ladder exists to avoid.
6. **Strongest practicality objection.** `[E]` None. The saving is real and is
   the second largest available anywhere in the design.
7. **Strongest significance objection.** `[E]` It would silently destroy the one
   comparison the task makes.
8. **Decisive test.** `[E]` Does an exact-compatible population exist? Run:
   `tools/inventory_existing.py`. Answer: none.
9. **Kill criteria.** `[E]` Killed on any metadata gap that prevents exact
   matching.
10. **What survives.** `[E]` **Killed.** The historical corpus is retained for
    **one** purpose: descriptive context in `LAMBDA_GRID_DECISION.md` §5, to check
    that a wide bracket is sensible. It is never pooled and is not evidence for a
    `sqrt(zeta)` law.
11. **Revised / stronger version.** `[I]` If someone later recovers the missing
    metadata for `boundary_aggregate.csv`, that is a **registration** task, not a
    reuse shortcut. Parked.

---

## Re-posed after refutation

`[I]` The refutations taught one thing worth restating. C2, C3 and C4 all die on
the same structure: each trades a **diagnostic** for compute, and in each case the
diagnostic is what separates a measurement from a number. The one place where
trading compute is legitimate is where the thing traded is a **confirmation** —
which is precisely the `zeta = 0.70, N_c = 2048` rung, and precisely why it is
conditional rather than cut.
