# Falsification plan — FROZEN before any new datum exists

TASK-2026-09-02-MOCK-PRODUCTION.

**This file states what will be attempted and what would kill each claim. It
contains no results and must never acquire a results column.** Outcomes go to
`FALSIFICATION_RESULTS.md`, a separate file written afterwards. Editing a frozen
artifact after its stage closed is validator error `M5`, and the separation
exists because the 2026-08-10 run's Stage-1 memo was silently backfilled to
agree with its own results.

The pre-registered success criteria M1–M7 live in `SUCCESS_CRITERIA.md` and
`analysis_spec.yaml`. This file covers the things that could make the *campaign*
wrong rather than the *curves* rough.

---

## The claim this campaign is actually making

> A production-scale population (`N_c = 1024`, `R = 24` independent populations
> per lambda) makes a **whole 13-point `CMI(lambda)` scan**, at three system
> sizes, clean enough and reproducible enough that the crossing machinery
> intended for the final production analysis behaves sensibly on it.

Not: that the crossing it finds is physical. Not: that `N_c = 1024` is
converged. Not: that these `L` say anything about the boundary.

---

## X1 — the reuse is invalid

**Attack.** ARM B's 288 populations are pooled into the `L = 64` curve as if
they were this campaign's own. If the sampler, the discretisation, the
resampling scheme, the horizon or the observable differed even slightly, that
pooling silently corrupts three of thirteen points — and they are the three with
the *smallest* error bars, so they would dominate every fit.

**How it is tested.** Seven-key match on
`(L, T, zeta, lambda, N_c, dtau_mult, resample_scheme)`, plus sha256 identity of
the sampler (`0a33c403…`), plus seed disjointness. All verified in
`REUSE_AND_DEDUP_AUDIT.md` §1 and `PRODUCTION_PATH_UNCHANGED.md`.

**Kill criterion.** Any key mismatch, or a sampler hash that differs by one
byte. Then the reuse is withdrawn and those three lambdas are recomputed at
`R = 24` (72 tasks, ~39 core-hours).

**Detectable after the fact**: the three reused points would sit off the curve
traced by their ten new neighbours by more than their joint error. That is
visible in Figure A and in the `q_i` at grid indices 5–9, and it is the reason
the reused points are *interior* to the grid rather than at an edge.

## X2 — the grid does not bracket the crossing

**Attack.** The endpoints were chosen from corpus fits at `L = 64…128`. There is
no `L = 32` or `L = 48` data anywhere. If the crossing for the `32/48/64` triple
lies outside `[0.2332, 0.3532]`, the crossing protocol returns nothing, or
returns an endpoint-induced artifact, and M3 cannot be evaluated.

**How it is tested.** The protocol's `endpoint_induced` flag, and the raw
sign-change count. A grid that fails to bracket shows as either zero sign
changes or a sign change at grid index 1 or 12.

**Kill criterion.** Zero raw sign changes for all three `L`-pairs at
`N_c = 1024`, or every sign change endpoint-induced.

**Consequence if killed.** M3 returns INCONCLUSIVE and the campaign reports that
the grid must move, with the measured `D(lambda)` showing which way.
`analysis_spec.yaml` `stopping_criteria` **forbids adding a lambda point in
response**; the remedy is a child task with its own frozen grid. The temptation
to extend the grid until a crossing appears is exactly what the freeze exists to
prevent.

**Pre-registered probability assessment [J]:** the corpus's own smallest-`L`
pair (64 vs 80) gives the lowest crossing of any pair, 0.2534, and the grid puts
two points below that. The risk is real but is bracketed on the side it is most
likely to appear.

## X3 — the cost model is wrong and the campaign does not fit in 3 hours

**Attack.** `L = 32` and `L = 48` have never run on Ruche. `NC_FACTOR[2048]`
extrapolates one measured doubling at a different `L`. The predecessor's model
was wrong by 30 % on exactly this kind of extrapolation.

**How it is tested.** The single-task check in `RUCHE_RUNBOOK.md` §4, run
**before** queueing 2,808 tasks, on `mockL64` index 0 and `mockL32` index 0.

**Kill criterion.** An observed wall time more than 1.4× the prediction. Then
the cost model is re-derived before anything is submitted.

**Why it cannot destroy the campaign even if it happens.** Every wall limit is
3–25× its prediction (`COST_MODEL.md` §6), and the critical path `mockL64` uses
the one rate that is *measured* at the exact `(L, N_c)` it runs. An error in the
`L = 32/48` rates changes the wall-clock of arms that finish in 0.2 and 1.1
hours.

## X4 — the analysis has a bug that hides the effect it looks for

**Attack.** A statistical test with the wrong null, a bootstrap that resamples
the wrong thing, or a criterion whose branches were never executed. Nothing in
the package's design catches this; only running the code does.

**How it is tested.** An end-to-end synthetic run: fabricate a plausible result
JSON for every one of the 2,808 manifest rows from a closed-form surrogate with
a **known** `L`-crossing and a **known** `N_c` offset, in a scratch copy, and
check that every branch of every criterion executes and that the criteria return
what the surrogate was built to imply.

**Kill criterion.** Any code path that does not execute, or a criterion that
returns the wrong verdict on data constructed to force a known one.

**[E] This attack already found a real defect** and it is recorded rather than
quietly fixed: the `Delta_N` bootstrap null was built by resampling the observed
populations, which carries the observed lambda-dependence into the "null",
inflates `p` and systematically hides the very effect M5 exists to detect. On a
surrogate with a deliberate tilt it returned `p = 0.572` and verdict C where the
parametric chi-square says `p ≈ 0.047`. Fixed by shifting each `N_c = 2048`
cell's populations to make the null true by construction before resampling;
after the fix the same data give `p = 0.045` and verdict B. See `VALIDATION.md`
§7.

## X5 — the companion arm makes the comparison too easy

**Attack.** M3 compares `N_c = 1024` against a `N_c = 128` companion run by this
same campaign. A sceptic could say the comparison is rigged: of course the
better-resolved curve looks cleaner.

**Answer, and why it is not a defect.** That *is* the question. M3 asks whether
raising `N_c` cleans up the locator structure, and the only controlled way to
ask it is to vary `N_c` and nothing else. The comparison is not designed to be
hard to pass; it is designed to be **interpretable**, which the comparison
against the `dtau_mult = 12`, different-grid, `L >= 64` historical corpus is
not.

**What would make it a defect.** If the companion arm were under-powered enough
that its curve looked jagged from `R`, not from `N_c`. Guarded by running it at
`R = 48`, twice the main arms', so that its SEM (≈ 0.010) is comparable to
theirs (0.006–0.009). And the analysis reports chi2/dof of the `N_c = 128` curve
against its own quadratic, which answers directly whether its jaggedness is
sampling noise (`analysis_spec.yaml` `jaggedness_question_rule`).

## X6 — the result is presented as physics

**Attack.** The single largest risk in this task is not statistical. It is that a
clean, unique, bootstrap-stable crossing at `lambda ≈ 0.27` gets written down as
`lambda_c(zeta = 0.35)` — by this task, by a later summary of it, or by a
manuscript that cites it.

**How it is guarded.** `prohibited_conclusions` in `analysis_spec.yaml`; the
closing section of `SUCCESS_CRITERIA.md`; and — because prose guards are the
ones that get skipped — the analysis script **prints the prohibition at the end
of every run**, and the crossing section prints
`LOCATOR QUALITY ONLY` in its own header.

**Kill criterion.** Not applicable; this is a reporting discipline, not a
hypothesis. It is listed here because a falsification plan that only lists
statistical failures misses the failure mode this programme has actually
suffered from (three separate derivations of `sqrt(zeta)`, each invalidated,
each replaced by another derivation of the same answer).

## X7 — the whole exercise is a slop warning

**Attack.** Slop warning №9: *treating computational scale as scientific depth*.
This campaign spends 379 core-hours and 2,808 tasks to produce curves that
explicitly may not be interpreted as physics. Warning №7: *a simulation regime
constructed mainly because it makes the method look good* — `L = 32, 48, 64` are
cheap sizes chosen partly because they are cheap.

**Answer.** Both are fair and neither is fatal, but they must be stated:

- **On №9 [J]:** the deliverable is a *decision*, not a dataset — whether the
  production configuration and the crossing machinery are sound enough to spend
  a real budget on. The alternative to spending 379 core-hours here is spending
  several thousand at production `L` and discovering the locator machinery is
  unusable. The campaign is sized to answer that and stops there:
  `stopping_criteria` forbids extending any arm in response to a result.
- **On №7:** the sizes were chosen to be cheap, and this is said openly rather
  than dressed up. The mitigation is that the criteria are pre-registered with
  reachable KILLED branches, and M2 in particular can fail in *both* directions.
  The synthetic test in `VALIDATION.md` §7 demonstrates M2 returning KILLED,
  M1 returning INCONCLUSIVE and M3 returning INCONCLUSIVE on constructed data —
  so the criteria are not a formality that passes by construction.
- **The real defence against №7** is M7, which asks whether these settings
  extrapolate to sizes that are *not* cheap, and can come back KILLED.

## X8 — negative results that are expected, and must be reported

Pre-registered so that they cannot later be quietly dropped (charter §4.4):

1. **M5 verdict C (unresolved) is a likely outcome.** The arm resolves a
   lambda-tilt of 0.0176 in `Delta_N`; if the true tilt is smaller, C is
   correct and must be reported as such, not upgraded to "approximately a common
   shift".
2. **The `N_c = 128` companion curve's chi2/dof against a quadratic is expected
   to land in `[0.5, 1.5]`** — the historical corpus gives 0.60–1.38 by the same
   statistic. If so, the honest conclusion is that **the old jaggedness was
   already consistent with sampling noise**, i.e. brief §12's question 4 answers
   "yes, the uncertainty explains it" and question 5 answers "the apparent
   features were not features". That is a real result and it makes part of the
   motivation for this campaign retrospectively weaker. It gets reported.
3. **M3 may come back INCONCLUSIVE simply because there is less to clean up than
   expected.** If the `N_c = 128` curves are already smooth, "materially
   cleaner" has little room to be true, and the correct report is that the
   high-`N_c` scan is *not* materially cleaner because the low-`N_c` scan was
   not materially dirty.

None of these are failures of the campaign. All three would be informative, and
2 and 3 in particular would change what the production budget should be spent
on.
