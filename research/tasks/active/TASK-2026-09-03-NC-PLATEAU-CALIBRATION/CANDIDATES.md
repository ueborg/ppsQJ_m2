# CANDIDATES — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

Charter Stage 3. Six candidates, eleven fields each, none optional. Killed
candidates stay in this file with their kill record (§4.4).

Each is a **design candidate**: a claim about what must be true for the
programme to proceed, with the arm that tests it.

Labels `[E]` `[I]` `[C]` `[J]`.

> **Amendment note.** The substance below was frozen at `stage_3_candidates`.
> It was restructured into the mandated eleven-field schema afterwards, under
> `task_phase.py amend`, because the first version used prose headings that
> `validate_task.py` T7 correctly rejected. See `POST_FREEZE_EVENTS.md`. No
> candidate, criterion, kill condition or arm changed.

---

## Candidate C1 — a high-`N_c` plateau is reachable at `L = 64`

1. **Statement** `[C]` At `L = 64`, `lambda = 0.3032`, `Delta_N = I_{2N} - I_N`
   becomes compatible with zero **and** materially small (P1 and P2 together)
   somewhere at or below `N_c = 8192`.
2. **Strongest affirmative case** `[E]` The `1024 -> 2048` step is already
   `+0.00235`, 0.7 % of the value, and `P1` passes. `[E]` The per-clone-window
   rate is flat in `N_c` at this `L`, so two more rungs are affordable
   (512 core-hours) where they are not at `L = 128`.
3. **Closest known precedent** `TASK-2026-09-02-MOCK-PRODUCTION` measured this
   cell at `N_c = 2048`, `R = 24`, and left shape convergence unresolved.
   `NOVELTY_GATE.md`: **corroboration**, extended.
4. **Strongest novelty objection** `[J]` None: nothing here is claimed as novel.
   Adding two rungs to an existing ladder is ordinary measurement.
5. **Strongest correctness objection** `[J]` A "plateau" over two rungs of a
   quantity with no known limit is a curve-fitting statement, not a convergence
   proof. `[I]` Answered by requiring P3 (successive `|Delta|` not increasing)
   and P5 (survives dropping the lowest rung) rather than P1 alone — neither of
   which the existing ladder can even evaluate.
6. **Strongest practicality objection** `[J]` "0.7 % is already a plateau; 512
   core-hours buys resolution on a number smaller than the 5–15 %
   across-population spread." `[E]` Answered: the existing step's 95 % interval
   is 1.7× `tau_I`, so it is equally consistent with a plateau and with a drift
   twice the tolerance. The ambiguity, not the magnitude, is what is bought.
7. **Strongest significance objection** `[J]` `L = 64` is at or below the
   programme's own corpus floor, and convergence there says nothing about
   `L = 128`. `[I]` Accepted, and it is why campaigns C and D exist; C1 is the
   *existence* question — is there any `L` at which this estimator demonstrably
   converges — and a negative answer there is decisive for the whole programme.
8. **Possible decisive test** Campaign A: `N_c = 4096` and 8192 at matched
   `R = 48`, with the existing `N_c = 2048` rung topped up from 24 to 48.
9. **Kill criteria** (fixed before testing) `[E]` `Delta_4096 = I_8192 - I_4096`
   resolved away from zero (`|Delta| > 1.96 SEM`); **or** P3 fails; **or** P5
   fails. Then: even `L = 64` is pre-asymptotic, no `I_inf` is extrapolated, no
   `B` or `gamma` is quoted.
10. **What survives the criticism** `[E]` `R = 48`, not the brief's preferred 24:
    at `R = 24` the top step's half-width would be ~1.2× `tau_I` and the arm
    could not satisfy P2 whatever the data did. `[E]` And a pre-registered
    non-result: P1 true with P2 false is `UNRESOLVED_R_LIMITED`, which is not a
    partial plateau and is not encouraging.
11. **Revised or stronger version** `[J]` The manifests are ordered so
    `--array=0-23` is a clean matched-`R`-24 sub-campaign, so the design
    degrades gracefully to the brief's literal form if the researcher prefers it.

## Candidate C2 — the correction is flat enough in `lambda` to preserve shape

1. **Statement** `[C]` Over the frozen seven-point transition-region grid,
   `Delta(lambda)` is compatible with an additive constant (H1) or a
   multiplicative rescaling (H2) rather than a resolved shape distortion (H3).
2. **Strongest affirmative case** `[E]` If the correction is flat in `lambda`,
   the curve is displaced but not deformed, and a crossing built from
   differences is largely protected.
3. **Closest known precedent** `[E]` None for a measured `Delta(lambda)` in the
   locator region. The existing high-`N_c` stencil is three points at `R = 24`
   around `lambda = 0.2932–0.3132`, **outside** the region entirely.
   `NOVELTY_GATE.md`: **no predecessor found under the searches performed** —
   which is a statement about this repository's own corpus and about no
   external literature at all.
4. **Strongest novelty objection** `[J]` `TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING`
   already suggested additive displacement can fail while multiplicative
   survives. `[E]` Answered: that is a hypothesis from a different argument at a
   different cell, and `SUCCESS_CRITERIA.yaml` gives it **no prior weight**.
5. **Strongest correctness objection** `[J]` Seven points and two one-parameter
   models is close to the resolution floor; failing to reject either proves
   nothing. `[E]` Accepted explicitly: both surviving is an UNRESOLVED outcome
   and is reported as one, never as support for either.
6. **Strongest practicality objection** `[J]` 573 core-hours, 26 % of the
   campaign. `[I]` It is also the only arm that measures the correction where
   the locator actually sits.
7. **Strongest significance objection** `[J]` If C3 holds, shape barely matters:
   the crossing survives a smooth displacement. `[I]` But C3's evidence *is*
   this measurement plus B2; the two cannot be separated.
8. **Possible decisive test** Campaign B: 7 `lambda` × `N_c ∈ {512,1024,2048}`,
   matched `R = 48`; H1/H2/H3 by `chi2` on 6 dof.
9. **Kill criteria** `[E]` Both H1 and H2 rejected while H3 is resolved. Then
   transition-region production needs a higher `N_c` than the level test alone
   would suggest.
10. **What survives the criticism** `[E]` `R = 48` gives a per-`lambda`
    `Delta_{1024->2048}` half-width of ~0.010, enough to resolve a distortion the
    existing three-point stencil could not.
11. **Revised or stronger version** `[J]` The interesting quantity may not be
    `Delta(lambda)` at all but its *derivative*, since a crossing depends on the
    slope. The analysis reports adjacent increments and second differences per
    curve, so that reading is available without re-running anything.

## Candidate C3 — the locator converges before the absolute level

1. **Statement** `[C]` The inferred cross-`L` crossing location stabilises within
   `tau_lambda = 0.004` at an `N_c` where `Delta` still fails P2 on the absolute
   level, because the displacement common to both `L` cancels in
   `D = I_{L_1} - I_{L_2}`.
2. **Strongest affirmative case** `[J]` **This is the load-bearing candidate.**
   `[E]` It is the only route to affordable `L = 128` production: the matched `R`
   needed to certify P2 at `L = 128` is in the thousands, ~13 000 core-hours for
   one `lambda`.
3. **Closest known precedent** `[E]` None. No matched cross-`L` `N_c` comparison
   exists anywhere in the corpus. `NOVELTY_GATE.md`: **no predecessor found
   under the searches performed** — canonical state and this repository's
   execution plane only; **no external literature was searched**, and a
   difference-cancellation argument is not deep enough for that silence to mean
   anything.
4. **Strongest novelty objection** `[J]` The observation that a common additive
   offset cancels in a difference is elementary and almost certainly written
   down somewhere in the SMC or finite-size-scaling literature.
   `NOVELTY_MATRIX.md` records that this task did not look, and that the search
   is owed before anyone calls C3's outcome a contribution.
5. **Strongest correctness objection** `[E]` The displacement is **known** to be
   `L`-dependent: `L = 64` moves +0.002 from 1024 to 2048 while `L = 128` moves
   −0.060 from 512 to 1024. So it demonstrably does **not** cancel completely,
   and C3 is at best a partial-cancellation claim.
6. **Strongest practicality objection** `[J]` It needs campaign B2 — 260
   core-hours and 1 872 tasks purely to put the reference curves on the same
   grid at the same `N_c`.
7. **Strongest significance objection** `[J]` If C3 holds, every published
   absolute CMI value stays uncalibrated even while crossings are trustworthy.
   `[I]` That is a real limitation and it is a reportable outcome, not a defect
   of the test.
8. **Possible decisive test** Campaigns B and B2 together: the fully matched
   `D(lambda)` at three `N_c`, plus the one-sided diagnostic that isolates the
   non-cancelling part.
9. **Kill criteria** `[E]` The crossing displacement per `N_c` doubling does not
   shrink, or does not fall inside `tau_lambda`. `[E]` Any crossing flagged
   `ENDPOINT_INDUCED` is reported as boundary-sensitive, not as located.
10. **What survives the criticism** `[E]` **The first version of B2 was wrong and
    the frozen protocol caught it.** Using only the three shared `lambda`, at a
    sixth of the cost, both interior crossings fall in the first or last interval
    and are flagged `ENDPOINT_INDUCED` **by construction** — the exact defect
    `TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION` existed to repair. Rebuilt on the
    full seven points; `tools/design.py` records the reason at the changed line.
11. **Revised or stronger version** `[J]` The sharper question is not whether the
    crossing converges but whether the *`L`-dependence* of the displacement
    converges, since that is the part that mimics finite-size scaling. The
    one-sided diagnostic measures exactly that and is reported alongside.

## Candidate C4 — `L = 96` enters a simpler high-`N` regime

1. **Statement** `[C]` The `L = 96` ladder, extended to `N_c = 1024` and 2048,
   shows a stable `B_eff` and an unrejected `1/N` form over its top rungs.
2. **Strongest affirmative case** `[E]` `L = 96` sits between an `L` that is
   nearly converged and one that is far from it; if the onset is a smooth
   function of `L`, this is where it should be visible.
3. **Closest known precedent** `TASK-2026-09-01-SMCRUCHE-READY` built the three
   lower rungs. `NOVELTY_GATE.md`: **provenance repair**.
4. **Strongest novelty objection** `[J]` None; this is filling a gap.
5. **Strongest correctness objection** `[E]` The accepted framing says a clean
   `1/N` is rejected at `L = 96`. Rebuilt from raw files, the three-rung
   `lambda = 0.3032` ladder gives `chi2 = 1.90`/1 dof, `p = 0.168` — **not
   rejected** (`agent_reports/numerics.md` F3). `[I]` The predecessor's figure
   needs a four-rung ladder this reconstruction cannot locate. `[J]` The
   discrepancy is reported, not resolved by assertion in either direction.
6. **Strongest practicality objection** `[J]` 292 core-hours for two rungs at
   `R = 24` that cannot certify convergence.
7. **Strongest significance objection** `[J]` If `L = 128` is the production
   target, `L = 96` is a waypoint rather than an answer. `[I]` It is also the
   only affordable place to see an onset at all.
8. **Possible decisive test** Campaign C, plus Y2's asymptotic-form battery.
9. **Kill criteria** `[E]` `Delta_1024` resolved and comparable to the existing
   `Delta_256 = -0.06959 ± 0.02102`, or `B_eff` unstable. Then `L = 96` is
   pre-asymptotic and is **excluded** from any cross-`L` comparison of `B`.
10. **What survives the criticism** `[E]` `R = 24` is a screening design and is
    labelled as one: the `Delta` half-width is ~4.5× `tau_I` and a small step at
    this `R` is `UNRESOLVED_R_LIMITED`, never "converged".
11. **Revised or stronger version** `[J]` `R = 16` was considered and rejected:
    it widens the interval 22 % for a 33 % saving on the cheapest high-`L` arm.

## Candidate C5 — `N_c = 2048` is materially better than 1024 at `L = 128`

1. **Statement** `[C]` The `1024 -> 2048` step at `L = 128` is materially smaller
   than the `512 -> 1024` step (`-0.06021 ± 0.02343`), indicating an onset.
2. **Strongest affirmative case** `[E]` It is the only measurement that can tell
   whether the hardest cell is anywhere near an asymptotic regime, and the whole
   production programme is aimed at that `L`.
3. **Closest known precedent** `[E]` `TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING`
   designed exactly this as its Design 4, costed it at 901 pessimistic
   core-hours, and **did not recommend it**. `NOVELTY_GATE.md`: **replication of
   a rejected design**.
4. **Strongest novelty objection** `[J]` It was proposed and declined one task
   ago. `[E]` Answered: the decline was on **mechanism discrimination** grounds
   and is correct on those terms — campaign D discriminates no mechanism. It is
   run here for **production adequacy screening**, at `R = 16` not 32, at 502
   not 901 core-hours, and labelled from the outset as unable to certify
   convergence. That objection is why campaign E is in the same submission group.
5. **Strongest correctness objection** `[E]` `R = 16` gives a `Delta` half-width
   of ~0.026, about 4.3× `tau_I`. It **cannot** certify convergence, and matched
   `R` for P2 is ~2 675. `[I]` Absolute-level plateau certification at `tau_I` is
   unreachable at `L = 128` at any affordable `R`.
6. **Strongest practicality objection** `[E]` 502 core-hours for 16 tasks and one
   screening number: the most expensive answer-per-core-hour in the campaign.
7. **Strongest significance objection** `[J]` Neither outcome discriminates a
   mechanism. `[I]` Both outcomes change what is submitted next, which is the
   decision this task exists to inform.
8. **Possible decisive test** Campaign D, `R = 16`, screening only.
9. **Kill criteria** `[E]` `|Delta_1024|` as large as `|Delta_512|` → recommend
   the conditional `N_c = 4096` rung, with its wall-time warning stated in the
   same sentence. `[E]` The trigger is fixed **now** and may not be made to
   depend on the observed value afterwards.
10. **What survives the criticism** `[E]` A pre-registered asymmetry: a large
    shift is informative at `R = 16`; a small one is `UNRESOLVED_R_LIMITED` and
    is never read as convergence.
11. **Revised or stronger version** `[J]` If C3 holds, campaign D's question is
    largely superseded — production `N_c` at `L = 128` would be set by the
    crossing tolerance instead. D is nevertheless the cheapest way to learn
    whether the level route is dead, and knowing that is worth 502 core-hours.

## Candidate C6 — the finite-`N_c` problem is discretisation-stable

1. **Statement** `[C]` E2: with the weak-potential windows fine enough, results
   are approximately independent of the window count `K`, consistent with a
   continuous-time interacting-particle limit. The alternative, E1, is drift
   scaling as `1/dtau_mult`.
2. **Strongest affirmative case** `[E]` **Both outcomes kill a mechanism**, which
   no other arm in this campaign can say, and it costs 1.9 % of the total.
   `[E]` The Feynman–Kac weight is exact at any window size, so the target
   measure is **exactly unchanged**; only where selection is applied moves.
   `[E]` It is the only axis that breaks the `L`/`ln K` collinearity (`r` to 0.99).
3. **Closest known precedent** `TASK-2026-08-30-SMCSTAT` `AN7_scheme_chunking`
   ran `dtau_mult ∈ {3,6,12}` at `K = 232/116/58` on a smaller cell.
   `NOVELTY_GATE.md`: **rediscovery at production geometry**.
4. **Strongest novelty objection** `[J]` The axis is not new and this candidate
   does not claim it is; the cell is new and the predecessor's numbers are flat
   to about one SEM, settling nothing at `K = 816`.
5. **Strongest correctness objection** `[J]` `N_c = 64` and 256 are far below
   production, so a null result might not transfer upward. `[E]` Answered by
   running both and reporting them separately: if they disagree, that
   disagreement is the result and neither is pooled into the other.
6. **Strongest practicality objection** `[J]` None. 42 core-hours.
7. **Strongest significance objection** `[J]` A null result (E2) removes a
   candidate mechanism without supplying one. `[I]` It also redirects attention
   to the accumulated-weight route — which is why this campaign records
   `logw_carry_var_final` and eleven per-window histories that no predecessor did.
8. **Possible decisive test** Campaign E, exactly as the parent task specified:
   `dtau_mult ∈ {3,6,12}`, `K = 816/408/204`, `N_c ∈ {64,256}`, matched `R = 48`.
9. **Kill criteria** `[E]` Flatness kills `K`-accumulation. A clean
   `1/dtau_mult` kills the schedule-independent mechanism. Anything between
   kills neither and is `INCONCLUSIVE`, not resolved toward the closer side.
10. **What survives the criticism** `[E]` `K` is not named the causal variable
    unless E1 is supported; `dtau_mult` is never called a physical parameter;
    and the `dtau_mult != 6` rows may never be pooled with the production corpus.
11. **Revised or stronger version** `[J]` With the per-window histories now
    recorded, E can be read as a *trajectory* question — how `Var(log w)`
    accumulates through the run at three window densities — rather than only as
    a three-point endpoint comparison. That is a strictly stronger reading of the
    same data and it needs no extra compute.

---

## Killed before design — C0

1. **Statement** `[C]` Skip the calibration and start rough production at
   `N_c = 1024`.
2. **Strongest affirmative case** `[E]` `N_c = 1024` curves already exist at
   `L = 32, 48, 64` over 17 `lambda`, are smooth, and show interior crossings.
   Production could begin immediately for ~0 additional core-hours.
3. **Closest known precedent** The entire existing corpus, which is at
   `N_c <= 1024` almost everywhere.
4. **Strongest novelty objection** `[J]` Not applicable.
5. **Strongest correctness objection** `[E]` **Fatal.** The finite-`N_c`
   displacement is `L`-dependent and unbounded at `L = 128`. An `L`-dependent
   displacement measured across several `L` is **indistinguishable from
   finite-size scaling by construction**.
6. **Strongest practicality objection** `[J]` None — it is the cheap option.
7. **Strongest significance objection** `[J]` It would produce a `phi` that
   might be an artefact of the sampler, which is the one failure
   `NUMERICAL_CAMPAIGN_CHARTER.md` §0 says the next campaign must be unable to
   commit.
8. **Possible decisive test** Not applicable; killed on evidence already held.
9. **Kill criteria** Met at design time by the measured `L`-dependence of the
   drift.
10. **What survives the criticism** `[E]` Nothing. **KILLED.** Recorded here
    because a negative design decision is a first-class output (charter §4.4)
    and "we considered starting production" belongs in the record.
11. **Revised or stronger version** `[J]` The surviving form of C0's instinct is
    C3: production may be affordable at a pre-asymptotic `N_c` **if** the
    displacement cancels in the cross-`L` difference. That is a testable version
    of the same wish, and it is the campaign's load-bearing candidate.
