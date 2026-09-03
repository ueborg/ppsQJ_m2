# FALSIFICATION_PLAN — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

**PRE-SPECIFIED. FROZEN AT `stage_3_candidates`, BEFORE ANY NEW DATUM EXISTS.**

This file states what will be attempted and what would kill each candidate. It
carries **no results column** and never will. Outcomes go to
`FALSIFICATION_RESULTS.md`, which is a different file written afterwards and
never merged back. Editing this plan to agree with what happened is the one
thing the phase lock exists to prevent.

Labels `[E]` `[I]` `[C]` `[J]`.

---

## Y1 — The plateau at `L = 64` (kills C1)

**Attempt.** Extend the `L = 64`, `lambda = 0.3032` ladder to `N_c = 4096` and
8192 at matched `R = 48`, top the existing `N_c = 2048` rung from 24 to 48, and
evaluate P1–P5 at the top step.

**Kills C1** if `Delta_4096 = I_8192 - I_4096` is resolved away from zero
(`|Delta| > 1.96 SEM`). **Reported then**: *even `L = 64` remains
pre-asymptotic*; no `I_inf` is extrapolated; no `B` is quoted; the programme has
no `L` at which this estimator is known to converge.

**Also kills C1**, more quietly: P3 fails (`|Delta|` increases materially at the
top step) or P5 fails (the conclusion depends on the lowest rung). `[J]` A
plateau that needs its lowest rung is a fit artefact.

**Pre-registered non-result.** `[E]` P1 true and P2 false is `UNRESOLVED_R_LIMITED`.
It is **not** a plateau, it is not a partial plateau, and it is not
reported as encouraging. The correct output is the matched `R` that would be
needed, which the analysis prints.

## Y2 — The asymptotic form (kills any coefficient `B`)

**Attempt.** Fit `I_inf + B/N` and, separately, `I_inf + B N^{-gamma}`, on each
of the three central ladders. Compute `B_eff(N) = -2N Delta_N`.

**Kills a usable `B`** if any of: the `1/N` `chi2` is rejected at 5 %; `B_eff`
is not stable across the top two steps; the fit changes materially when the
lowest rung is dropped. **Reported then**: `NO OBSERVED 1/Nc ASYMPTOTIC REGIME`.

**Pre-registered refusals.** `[E]` A free-`gamma` fit on three rungs has zero
dof, is exact by construction and is **not evidence**; it is printed with its
dof so it cannot be read as one. `[E]` No `gamma` is quoted from a ladder that
has not passed the above. `[E]` Any `B` that survives is LOCAL and
CELL-SPECIFIC and is **not** promoted to `B(L, zeta, lambda)`. `[E]` `B` is
compared across `L` **only** where each ladder independently shows the same
form; a pre-asymptotic ladder is excluded, never averaged in.

## Y3 — The shape of the correction (kills C2)

**Attempt.** Over the frozen seven-point grid, test H1 (additive constant) and
H2 (multiplicative rescaling) on `Delta_{512->1024}(lambda)` and
`Delta_{1024->2048}(lambda)`, one free parameter each, `chi2` on 6 dof.

**Kills C2** if both H1 and H2 are rejected while H3 (resolved
`lambda`-dependence) holds.

**Pre-registered non-result.** `[E]` Neither H1 nor H2 rejected is **UNRESOLVED**,
not a win for either, and specifically not confirmation of the predecessor's
suggestion that multiplicative behaviour survives where additive fails. That
suggestion carries **no prior weight** here. `[E]` Seven points are not
overfitted with a third parameter.

## Y4 — Locator convergence (kills C3 — load-bearing)

**Attempt.** Build the fully matched cross-`L` difference `D = I_{L_1} - I_{L_2}`
at `N_c = 512, 1024, 2048` on the shared seven-point grid, for the pairs
L32−L64, L48−L64 and L32−L48; run the frozen crossing protocol at each `N_c`;
and measure how far the bootstrap crossing location moves per `N_c` doubling.
Report the one-sided diagnostic (reference held at `N_c = 1024`) alongside,
because it isolates the part of the displacement that does not cancel.

**Kills C3** if the crossing displacement `512 -> 1024 -> 2048` does not shrink,
or does not fall inside `tau_lambda = 0.004`.

**Pre-registered.** `[E]` If C3 survives while P2 fails on the absolute level,
production `N_c` is defined by the **crossing tolerance**, and the report must
say that is the basis. `[E]` The reverse is not permitted: absolute-level
convergence does not certify a locator. `[E]` Any crossing flagged
`ENDPOINT_INDUCED` is reported as a boundary-sensitive locator and not as a
located crossing, whatever else is true.

**Pre-registered re-derivation.** `[E]` `tau_D = tau_lambda x 2.965` uses a
slope measured on the existing `N_c = 1024` curves. The analysis recomputes that
slope from the new curves. If it differs materially, `tau_D` must be re-derived
**before** any adequacy verdict, and the re-derivation recorded as an amendment.

## Y5 — `L = 96` (kills C4)

**Attempt.** Extend the `L = 96` ladder to `N_c = 1024` and 2048 at `R = 24` and
apply Y1's criteria and Y2's fits.

**Kills C4** if `Delta_1024` is resolved and comparable to the existing
`Delta_256 = -0.06959 ± 0.02102`, or if `B_eff` is unstable. **Reported then**:
`L = 96` is pre-asymptotic and is excluded from any cross-`L` comparison of `B`.

**Pre-registered.** `[E]` At `R = 24` the `Delta` half-width at that step is
about 0.027, i.e. 4.5× `tau_I`. A **small** step at this `R` is
`UNRESOLVED_R_LIMITED` and is never reported as convergence.

## Y6 — `L = 128` (kills C5)

**Attempt.** Add `N_c = 2048` at `R = 16` and compare `Delta_1024` against
`Delta_512 = -0.06021 ± 0.02343`.

**Kills C5** if `Delta_1024` is as large as `Delta_512`. **Reported then**: the
conditional `N_c = 4096` central rung is recommended, and the report must state
that a single population there is a ~71 h job (~100 h pessimistic) against
`cpu_long`'s 168 h `MaxTime`, i.e. feasible but fragile.

**Pre-registered, both directions.** `[E]` If the shift appears small, that is
**not** convergence: at `R = 16` the half-width is ~0.026 and the matched `R`
needed for P2 is in the thousands. The recommendation is then *more `R` before
any plateau claim*, or the locator route. `[E]` The `N_c = 4096` trigger is
fixed in `conditional/cond_D2_L128_nc4096/README.md` **now**, and may not be
made to depend on the observed 2048 value after the fact.

## Y7 — Discretisation (kills C6, either way)

**Attempt.** `dtau_mult ∈ {3, 6, 12}` at `L = 64`, `T = 64`, `lambda = 0.3032`,
`K = 816/408/204`, `N_c ∈ {64, 256}`, matched `R = 48`.

**Kills the `K`-accumulation mechanism** if the three means are flat within
their own errors. **Kills the schedule-independent mechanism** if drift is a
clean `1/dtau_mult`. **Neither is killed** by anything in between, which is
reported `INCONCLUSIVE` and not resolved toward the closer side.

**Pre-registered.** `[E]` `K` is not called the causal variable unless E1 is
supported. `[E]` `dtau_mult` is never called a physical parameter. `[E]` Both
`N_c` are reported separately: the `K`-dependence may itself depend on
population size. `[E]` The `dtau_mult != 6` rows may **never** be pooled with
the production corpus.

## Y8 — Expected negative results, pre-registered so reporting them is not a retreat

`[E]` Each of these is a legitimate terminal outcome of this campaign, and each
is worth its cost:

1. **No plateau anywhere.** All three ladders still drifting at their top rung.
   The programme then has no calibrated `N_c` at any `L` and rough production
   does not start.
2. **A plateau at `L = 64` only.** The estimator converges where it is cheap and
   not where it is needed. Production is limited to `L <= 64` until a variance
   reduction exists.
3. **Locator convergence without level convergence.** C3 survives, P2 fails
   everywhere. Production proceeds on the crossing tolerance, and every
   published absolute CMI value stays uncalibrated.
4. **`UNRESOLVED_R_LIMITED` at every `L`.** The campaign returns matched-`R`
   requirements rather than verdicts. `[J]` This is the most likely single
   outcome at `L = 96` and `L = 128` and it is a real answer: it converts an
   open question into a costed one.
5. **`INCONCLUSIVE` on the discretisation axis.** The one experiment whose both
   outcomes were supposed to kill something kills nothing.
6. **The `L = 96` provenance discrepancy is never resolved** because the ladder
   behind the predecessor's `chi2 = 10.54` cannot be located. Reported as an
   open provenance item, not quietly dropped.

`[E]` **No arm is re-run automatically on a negative result.** Any extension
needs a fresh task, a fresh justification that is not "the last one did not
work", and a human gate.

## Y9 — Falsifiers aimed at this task's own machinery

`[J]` The plan is not only about the physics.

1. **The reuse ledger is fiction.** Falsified by `tools/dedup_scan.py`, which
   compares every ledger entry against the populations actually on disk and
   fails on any disagreement.
2. **The modified `run_cell.py` changed the sampler.** Falsified by
   `tools/reproduce_check.py`, which re-executes completed predecessor
   populations and requires the per-clone trajectory to be **bit-identical**.
3. **The preflight passes everything.** Falsified by
   `tools/negative_controls.py`: sixteen injected faults, each of which must
   make the preflight exit non-zero **for the injected reason**.
4. **The cost model is fitted to a requested `--time` rather than to
   measurement.** Falsified by construction — every rate in
   `tools/cost_model.py` carries the rung, the `n`, and the median/p90/max it
   came from, and the model is the max.
5. **The memory model is a model quoted as a measurement.** This is the defect
   found in the inherited one; falsified here by `tools/mem_probe.py` reading
   `ru_maxrss` from a real run at thirteen cells.
