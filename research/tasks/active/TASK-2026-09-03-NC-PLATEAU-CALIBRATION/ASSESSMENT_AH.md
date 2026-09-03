# ASSESSMENT A–H — TASK-2026-09-03-NC-PLATEAU-CALIBRATION   (Charter §5)

Each dimension assessed **separately**. **No aggregate score**: a fatal weakness
in one dimension is not offset by strength in another. "Unanswerable" is a valid
verdict; "adequate" without reasoning is not.

Labels `[E]` `[I]` `[C]` `[J]`.

---

## A. Consequential bottleneck

**Verdict: YES, and it is upstream of everything else the programme wants.**

`[E]` `lambda_c(zeta, L)` is a functional of curves measured at some `N_c`; `nu`,
`phi` and the FSS form are functionals of `lambda_c(zeta, L)`. `[E]` The
finite-`N_c` displacement is `L`-dependent — `L = 64` moves +0.002 from 1024 to
2048 while `L = 128` moves −0.060 from 512 to 1024. `[I]` An `L`-dependent
displacement measured across several `L` is **indistinguishable from finite-size
scaling by construction**. `[J]` That is the one failure
`NUMERICAL_CAMPAIGN_CHARTER.md` §0 says the next campaign must be unable to
commit, and it is currently uncontrolled.

`[E]` It is also a bottleneck in the ordinary sense: the analytic route is
closed (uniform-mixing bounds do not transfer), the empirical substitute is
rejected at `L = 128`, and no diagnostic is stable out of sample. Nothing
downstream can be trusted until this is measured.

## B. Mechanistic contribution

**Verdict: WEAK, and honestly so.**

`[J]` This campaign measures **where** the estimator converges, not **why**.
Five of its six candidates are calibration questions with no mechanism attached.
`[E]` The one exception is campaign E, which discriminates two named mechanisms
(`K`-accumulation versus schedule-independent accumulation) and whose both
outcomes kill one — and it is 1.9 % of the cost.

`[E]` The campaign does make one mechanistic contribution that costs nothing:
`logw_carry_var_final` and eleven per-window histories, recorded for the first
time. `TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING` identified the across-clone
spread of the **accumulated** log weight as the one surviving candidate for the
`L`-growth and found it recorded in **0 %** of production-geometry runs. `[I]` If
campaign E returns E2, that is where the next mechanism question goes, and these
will be the only runs carrying the data to follow it.

`[J]` Assessed as weak rather than absent: a calibration campaign that produces
no mechanism is doing its job, but it should not be described as explaining
anything.

## C. Discriminability

**Verdict: STRONG for E, GOOD for A and B, WEAK for C and D.**

`[E]` Campaign E: both outcomes kill a mechanism, the target measure is exactly
invariant under the manipulation, and the `INCONCLUSIVE` band is pre-registered.
`[J]` This is the best-designed arm in the campaign and it is the cheapest.

`[E]` Campaign A at `R = 48`: the top step resolves `Delta` to ~0.0058 against
`tau_I = 0.006` — just inside, and only after `R` was doubled. Campaign B at
`R = 48` resolves the crossing to ~0.0034 in `lambda` against
`tau_lambda = 0.004`. Both discriminate at their frozen tolerance.

`[E]` Campaigns C and D **cannot** discriminate at `tau_I`: their half-widths
are 4.5× and 4.3× the tolerance. `[J]` They discriminate only "is this step the
size of the last one" — a screening question — and they are labelled that way in
`SUCCESS_CRITERIA.yaml` with their own verdict class rather than being allowed
to look like convergence tests.

## D. Dependency significance

**Verdict: HIGH.** `[E]` Every downstream quantity in the programme depends on
the answer, and the six live disputes (`DISP-PHI-001`, `DISP-WINDOW-001`,
`DISP-XI-001`, `DISP-CASEA-UNIV-001`, `DISP-SNAPSHOT-001`, `DISP-YZETA-001`) sit
downstream of the chain this campaign is upstream of.

`[E]` **None of them is moved by this campaign and none is closed here.** `[J]`
What the campaign decides is whether the measurements that would bear on them
can be trusted at all — which is a precondition for working on them, not a
contribution to any of them.

## E. Cross-silo value

**Verdict: LOW, and the search that would raise it was not performed.**

`[J]` The one result with plausible reach beyond this project is C3: whether a
crossing-based locator converges earlier than the absolute observable because a
common displacement cancels in a difference. That structure appears wherever
finite-population estimators are compared across a control parameter.

`[E]` But **no external prior-art search was performed anywhere in this task**.
`[I]` So cross-silo value is asserted as plausible and is **not established**,
and `NOVELTY_MATRIX.md` records that the search is owed before anyone describes
C3's outcome as a contribution rather than as an internal calibration. `[J]`
Silo-breaking claimed from a structural analogy alone is Slop Warning 11, and
this assessment declines to claim it.

## F. Robustness

**Verdict: ADEQUATE on machinery, MIXED on power.**

`[E]` Machinery: the design regenerates byte-identically; sixteen injected
faults each make the preflight fail for the injected reason; the reuse ledger is
checked against disk; two predecessor populations reproduce bit-for-bit through
the modified wrapper; the cost model is fitted to measured `wall_s` and the
memory model to measured `ru_maxrss`; `K` is exact against all 1 896 recorded
values.

`[E]` Power: campaign A's **lower** step is under-powered for P2 by ~28 %;
campaigns C and D cannot satisfy P2 at all. `[J]` These are stated in the design
rather than discovered afterwards, which is the difference between a limitation
and a defect — but they are limitations, and two of the six campaigns cannot
reach their own headline criterion.

`[E]` One robustness gap is unclosed: the memory measurements are macOS
`ru_maxrss` and the cluster is Linux. Mitigated by keeping the old formula as a
floor plus a 1.35× margin, and by asking for one `sacct MaxRSS` line.

## G. Informative failure

**Verdict: STRONG. Every arm's failure is informative, and six negative outcomes
are pre-registered.**

`[E]` `FALSIFICATION_PLAN.md` Y8 names six terminal negative outcomes in
advance, including the most likely single one — `UNRESOLVED_R_LIMITED` at
`L = 96` and `L = 128`. `[J]` That outcome is a real answer: it converts an open
question into a **costed** one, and the analysis prints the matched `R` that
would be required at every step.

`[E]` The two most valuable failures may already have happened, before
submission: `R = 24` in campaign A was killed by its own power calculation, and
the cheap three-`lambda` version of campaign B2 was killed by the frozen
crossing protocol. `[J]` Both cost core-hours to repair (~386 combined) and both
are recorded at the line of code that changed rather than in a summary.

`[E]` And one accepted claim failed to reproduce (the `L = 96` `1/N` rejection),
which is reported as an open provenance item rather than dropped.

## H. Infrastructure value

**Verdict: HIGH, and partly independent of whether the campaign ever runs.**

`[E]` Already delivered, regardless of any HPC result:

- the first complete inventory of the corpus as one object — 1 896 populations,
  62 cells, 53 ladders, rebuilt from raw files;
- a cost model that corrects a 30 % optimistic error in the inherited one, in
  exactly the regime the programme is entering;
- the first peak-RSS **measurements** of this sampler anywhere in the
  repository, correcting a model that had been quoted as a measurement;
- instrumentation closing two gaps a predecessor named as blocking
  (`logw_carry_var_final`, `git_commit`);
- a demonstration that a stored **aggregate** in this corpus is not
  bit-reproducible across architectures even though the trajectory is;
- a frozen analysis that runs to completion on zero data and says so.

`[J]` The strongest single item is the cost-model correction: it is the
difference between an `L = 128` `N_c = 4096` job being planned as a 55-hour run
and being planned as the 71-hour run it actually is, against a 168-hour
partition ceiling.

---

## What no dimension can rescue

`[E]` **Stage 8 is not satisfied.** No independent investigator and no
independent red team ran at any point. `validate_redteam.py` refuses the report
under rule R3 and the flag was not set to false to make the check green. `[J]`
Under charter §5 that is not a weakness in one dimension to be traded against
strength in another; it is a missing procedural guarantee, and the human gate
should treat every "survives" verdict here as unreviewed.
