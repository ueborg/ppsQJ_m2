# RESEARCH_MEMO — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

Charter Stage 9. Labels `[E]` `[I]` `[C]` `[J]`.

**Terminal state: `READY_FOR_HUMAN_SUBMISSION` at Human Gate A.** No HPC job was
submitted, no scheduler was contacted, no `research/state/**` file was written,
no predecessor task directory was modified.

---

## 1. What was asked, and what is delivered

The brief asked for a coordinated numerical task that reconstructs the existing
data, designs six campaigns, freezes its criteria before results, prepares
runnable Ruche packages, and stops at the human gate.

**Delivered:** seventeen validated immediate arms (3 280 fresh tasks, 2 180
core-hours), seven blocked conditional arms behind hard interlocks, a frozen
analysis that computes every criterion and runs to completion on zero data, and
a corpus reconstruction that produced four results the campaign did not have to
run anything to obtain.

**The strongest of those four costs nothing and changes the plan.**

## 2. Results obtained WITHOUT new compute

`[E]` These come from rebuilding all 53 `N_c` ladders from the 1 896 raw
per-population JSONs (`tools/reconstruct_inventory.py`), never from a
predecessor's summary.

### 2.1 `EMPIRICAL` — absolute-level certification at `L = 128` is unreachable

`[E]` From the measured across-population spreads, the matched `R` needed to put
a `Delta` 95 % interval inside the frozen `tau_I = 0.006` is **2 675** at the
`L = 128` `512 → 1024` step, and larger at every lower rung. At the measured
`N_c = 2048` rate that is about **13 000 core-hours for one `lambda`**.

`[I]` **Absolute-level plateau certification at `tau_I` is not affordable at
`L = 128` with this estimator.** `[J]` This is the campaign's most
decision-relevant finding and it required no new compute. It is why campaign D
is honestly labelled a screening rung, and it is what makes the locator route
(candidate C3) load-bearing rather than merely attractive.

### 2.2 `EMPIRICAL` — the existing `L = 64` "near-plateau" establishes nothing

`[E]` `Delta_1024 = +0.00235 ± 0.00528`, 95 % interval `[−0.0080, +0.0127]`,
half-width **1.72× `tau_I`**. `[I]` P1 passes and P2 fails: the step shows `R`
was too small to tell, not that the ladder converged, and **no increase in `N_c`
narrows an interval set by `R`**. `[E]` Matched `R` for a P2 pass at that step:
137. `[J]` This distinction has its own verdict label
(`UNRESOLVED_R_LIMITED`) precisely so it cannot be read as encouraging.

### 2.3 `EMPIRICAL` — the inherited cost model extrapolates the wrong way

`[E]` `TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA` read the `L = 128` rate as
small-batch inefficiency ending by `N_c ≈ 256` and extrapolated **flat** from
there. Two rungs have since completed and the rate turns back up: 23.998,
25.080, 28.260 ms at `N_c = 256, 512, 1024`. Flat-from-256 predicts 21.52 at
1024 — **30 % low, in the optimistic direction, in exactly the regime this
campaign enters.**

`[E]` Refitted here as `rate ~ N_c^0.187`, applied above the largest measured
rung at each `L`, conservatively (at `L = 64` and 96 the measured local trend is
flat or negative and the exponent is applied anyway).

`[I]` The practical consequence: a conditional `L = 128`, `N_c = 4096`
population is a **71.5 h** job (100 h pessimistic) against `cpu_long`'s 168 h
`MaxTime` — feasible but fragile and uncheckpointed — and `N_c = 8192` at
`L = 128` is **not runnable at all** under this architecture. `[J]` Under the old
model that job would have been planned as a ~55 h run.

### 2.4 `EMPIRICAL` — the `--mem` model was a model quoted as a measurement

`[E]` `TASK-2026-09-01-SMCRUCHE-READY` describes "the measured 732 MB peak";
732 MB is exactly what `128 + 2 N_c per_clone` returns for `L = 96, N_c = 512`,
and **no `MaxRSS` from any Ruche job appears anywhere in this repository.**
`[E]` Direct `ru_maxrss` at 14 cells shows the formula **under**-predicts at
seven of them; `L = 64`, `N_c = 2048` measures 1 694 MB against a predicted
1 202 MB, and the arm that ran that cell requested `--mem=2G`. `[J]` It never
broke, and it was closer to breaking than anyone knew.

## 3. A correction to the accepted framing

`[E]` The brief states that a clean `I_N = I_inf + B/N` was rejected "at `L = 96`
and `L = 128`".

`[E]` **`L = 128` reproduces exactly** — `chi2 = 12.58` on 3 dof, `p = 0.0056`,
reached from the raw JSONs independently of the predecessor's code. `[J]` That
agreement is also the strongest available check that this task's fitting
machinery is correct, and it passed before any new datum existed.

`[E]` **`L = 96` does not reproduce.** The production-geometry `L = 96` ladder at
`lambda = 0.3032` that can be rebuilt from raw files has **three** rungs and
gives `chi2 = 1.90` on 1 dof, `p = 0.168` — **not rejected**. `[I]` The
predecessor's `chi2 = 10.54` on 3 dof needs a four-rung ladder this
reconstruction cannot locate anywhere in the repository.

`[J]` Reported as an **open provenance item**, not a refutation: a ladder this
task cannot find is not a ladder this task has shown to be wrong, and the
project's own rule is to follow a claim's direct provenance before contradicting
its history. What follows is narrower and still useful — the `L = 96` half of the
accepted framing should not be leaned on, and `L = 96` is *less* characterised
than assumed, which strengthens the case for campaign C.

## 4. The design, and the two decisions a falsifier forced

`[E]` Six campaigns; full design in `CAMPAIGN_DESIGN.md`. Two arms are not what
the brief specified, and in both cases the brief's own instruction to report an
alternative rather than silently change it is what happened.

**`R = 48`, not 24, in campaign A.** `[E]` The brief anticipated cost forcing `R`
*down*; the measured spreads force it *up*. At `R = 24` the `4096 → 8192` step's
half-width would be ~1.2× `tau_I`, so **the arm could not have satisfied P2
whatever the data did**. `[I]` An arm that cannot pass its own frozen criterion
is not a measurement. `[E]` Cost of the fix: +166 core-hours. `[E]` The residual
is stated rather than hidden: the *lower* step (`2048 → 4096`) still needs
`R ≈ 80` and is under-powered for P2 by ~28 %. `[E]` The manifests are ordered so
`--array=0-23` is a clean matched-`R`-24 sub-campaign if the researcher prefers
the literal design.

**Campaign B2 exists, and its cheap version was killed.** `[E]` The brief's §4B
asks whether the crossing converges with `N_c`. A crossing needs two curves, and
the `L = 32`/`L = 48` reference curves exist at `N_c = 1024` **and nowhere
else** — so campaign B alone can only move `N_c` on one side. `[E]` B2's first
design used only the three `lambda` shared with the measured 0.010 grid, at a
sixth of the cost. Running the **frozen** crossing protocol on that grid showed
both interior crossings fall in the first or last interval, so every crossing is
flagged `ENDPOINT_INDUCED` **by construction, whatever the data say** — the exact
defect `TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION` was created to repair. `[E]`
Rebuilt on the full seven points at +220 core-hours; `tools/design.py` records
the reason at the changed line.

## 5. The tolerances, and the order they were chosen in

`[J]` `tau_lambda` is fixed **first** and `tau_I` derived from it. An
absolute-CMI tolerance with no decision attached is a number chosen for
convenience.

```
tau_lambda   = 0.004     crossing-location tolerance                 PRIMARY
|dD/dlambda| = 2.965     MEASURED at the interior crossings, not assumed
tau_D        = 0.0118    = tau_lambda x 2.965
tau_I        = 0.006     = tau_D / 2, per-curve, WORST CASE
```

`[E]` The slope is the **smaller** of the two measured interior-crossing slopes,
because a smaller slope converts a given CMI error into a *larger* `lambda`
error. `[E]` `tau_I` assumes the two curves' displacements do not cancel at all
and therefore add. `[J]` That is a worst case, and whether it is the real case is
precisely what campaigns B and B2 measure — which is why an absolute-level
failure at `tau_I` is **not** automatically a locator failure, and why
`SUCCESS_CRITERIA.yaml` explicitly permits defining production `N_c` from the
crossing tolerance while P2 fails.

## 6. `N_c` and `R` are separate budgets, and the corpus conflated them

`[E]` `N_c` controls the finite-particle approximation — drift and
within-population variance. `R` controls the uncertainty of the finite-`N_c`
population mean. **Increasing `R` does not eliminate finite-`N_c` drift;
increasing `N_c` does not narrow an interval set by `R`.**

`[E]` The existing corpus varies both at once: `R` runs 96, 64, 48, 32, 24 across
rungs, so a step's `Delta` and its half-width move together for reasons that have
nothing to do with convergence. `[E]` Every new ladder here is matched-`R` within
itself, the analysis reports both budgets for every cell, and every verdict names
which one binds.

## 7. Instrumentation: two named gaps closed at zero cost

`[E]` `TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING` found that the one surviving
candidate factor for the `L`-growth of drift — the across-clone spread of the
**accumulated** log weight — is recorded in **0 %** of production-geometry runs,
and that `git_commit` is absent from **100 %** of the 3 784-record corpus.

`[E]` Both are now recorded. `final_weights` is persisted, so
`Var(log w_carry)` at `t = T` is recovered **exactly** rather than by proxy;
eleven per-window histories the sampler already computed and the predecessor
discarded are kept; `delta_tau`, `K` and `n_resampling_events` are persisted.
`[E]` Cost to the simulation: none — adding output fields cannot perturb a
trajectory, and `tools/reproduce_check.py` demonstrates it rather than asserting
it: **all 1 024 per-clone CMI values bit-identical** on both re-executed
predecessor populations.

`[E]` **A by-product worth recording.** The derived *reductions* differ from the
stored values by up to `1.7e-14` relative — x86-versus-arm64 summation order in
numpy's pairwise reductions. `[I]` A stored **aggregate** in this corpus is not
bit-reproducible on a different architecture even though the trajectory is.
Nothing in the repository said so.

## 8. What this task did NOT do

`[E]` **No independence was obtained.** Every role was executed inline by the
lead. `validate_redteam.py` **refuses** the red-team report under rule R3
("reviewer saw the lead summary"), and the flag was not set to false to make the
check green. **Charter Stage 8 is not satisfied**, and every "survives" verdict
in `REDTEAM.yaml` should be treated as unreviewed at the human gate.

`[E]` **No external prior-art search was performed anywhere in this task.**
External novelty for candidates C2 and C3 is therefore `UNRESOLVED`, not
favourable. `[J]` C3's structure — a common offset cancels in a difference — is
elementary enough that someone in the SMC or finite-size-scaling literature has
very likely written it down, and the search is owed before anyone calls C3's
outcome a contribution rather than an internal calibration.

`[E]` **No dispute was moved and none may be.** The six live disputes sit
downstream of the chain this campaign is upstream of.

`[E]` **No claim about the physics.** No `lambda_c(zeta)`, no boundary law, no
exponent, and the 0.2182–0.2482 window is an **observed locator region** in
`L <= 64` curves at `N_c = 1024` — not a critical window.

`[E]` **The frozen theory result is preserved in its narrow form**: the standard
useful uniform-mixing Feynman–Kac bounds do not directly transfer to the
production mutation kernel, because the no-click branch is deterministic. `[E]`
That is the failure of a **proof route**. It is **not** "1/`N_c` convergence is
impossible", and nothing here upgrades it.

## 9. The strongest argument against this whole campaign

`[J]` Written properly, because the memo is worthless if it is a straw man.

**The argument.** `tau_I` is a **worst-case** translation assuming zero
cancellation between the two curves. If the displacement cancels even partially,
`tau_I` is far stricter than the science needs, and campaigns A, C and D — 1 306
core-hours, 60 % of the campaign — are certifying against a tolerance nothing
requires. On that reading the only arms that matter are B, B2 and E, for 874
core-hours.

**What survives it.** `[E]` The displacement is **known** to be `L`-dependent —
`L = 64` +0.002 against `L = 128` −0.060 — so complete cancellation is already
refuted, and a partial-cancellation hypothesis with no measured size is not
something to build a production campaign on. `[I]` And the argument is
self-undermining in one direction: if it is right, campaign D's answer is
cheap to obtain and campaign A is the only way to learn whether *any* `L`
converges at all.

`[J]` **But it is right about the emphasis, and that is worth saying plainly.**
If the researcher wants to cut this campaign, the defensible cut is `C` and `D`
(794 core-hours, 36 %), keeping A, B, B2 and E — not the reverse. The reason
`D` is nevertheless recommended is that it is the cheapest way to learn whether
the absolute-level route at `L = 128` is dead, and knowing that is worth 502
core-hours before anything larger is committed.

## 10. The question the campaign exists to answer

> **What is the smallest defensible `N_c` required to locate the transition at
> each relevant `L`?**

`[E]` It is not answered here and cannot be without the data. `[I]` What this
task establishes is which route can answer it: at `L = 64` the absolute-level
route is affordable and is taken; at `L = 128` it is **not**, at any affordable
`R`, so the answer must come from the crossing tolerance or not at all. `[J]`
That reframing — from "how big must `N_c` be" to "which tolerance can we afford
to certify against" — is the memo's main content, and it came from the existing
data rather than from the campaign.
