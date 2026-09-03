# PARKING_LOT — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

Directions and questions raised and deliberately **not** pursued here. Each with
why, and what would make it worth opening.

Labels `[E]` `[I]` `[C]` `[J]`.

---

## P1 — C3 is tested at `L <= 64` and would be USED at `L >= 96`

`[E]` The locator-convergence candidate is measured at `L = 32, 48, 64`. The `L`
where it matters for production is 96 and 128, where the `L`-dependence of the
displacement is much larger.

`[J]` **The weakest link in the whole campaign.** A matched cross-`L` `N_c`
comparison at `L >= 96` needs seven-`lambda` curves at two `L` and three `N_c` at
those sizes — thousands of core-hours, and at `L = 128` it collides with the
partition ceiling.

**Open it when**: campaign B+B2 returns L-1 (the crossing converges) **and**
campaign D returns a materially large step. That combination is what makes the
extrapolation load-bearing rather than merely convenient.

## P2 — the `L = 96` four-rung `1/N` ladder could not be located

`[E]` `TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING` reports `chi2 = 10.54` on
3 dof for `L = 96, T = 96`. That needs four rungs; the `lambda = 0.3032` ladder
rebuildable from raw files has three and gives `p = 0.168`.

`[J]` **This should be resolved by someone who is not this task** — ideally an
independent numerics pass, since the lead has now formed a view. It is cheap: a
search, not a computation.

**Open it when**: anyone wants to cite "the `1/N` form is rejected at `L = 96`".

## P3 — an external prior-art search on difference-based locator robustness

`[E]` Not performed anywhere in this task. `[J]` C3's structure — a common
additive offset cancels in a difference, so a crossing survives a displacement
the absolute level does not — is elementary enough that the SMC or
finite-size-scaling literature very likely contains it.

**Open it when**: anyone wants to describe C3's outcome as a contribution rather
than as an internal calibration. Until then `NOVELTY_GATE.md` records external
novelty as `UNRESOLVED` and no claim in this package rests on it.

## P4 — the genealogy route (the parent task's Design 3) remains unrun

`[E]` `mrca_mean` and the pairwise-MRCA diagnostics need the full ancestor
matrix, which is `K x N_c` — 27 MB per population at `N_c = 8192` and ~2.6 GB
across campaign A alone. This campaign computes it in memory and discards it.

`[E]` The parent task declared Design 3 **partly circular and partly infeasible**
and said it must be re-specified before it is run. `[J]` Nothing here
re-specifies it, and this campaign's output cannot support it. Stated so nobody
later assumes the data are there.

**Open it when**: campaign E returns E2 *and* the accumulated-weight route (P5)
also fails.

## P5 — the accumulated-weight route

`[E]` The parent task identified the across-clone spread of the **accumulated**
log weight as the one surviving candidate for the `L`-growth of drift, and found
it recorded in 0 % of production-geometry runs.

`[E]` This campaign records it — `logw_carry_var_final`, exactly, plus eleven
per-window histories. `[J]` But recording is not analysing: nothing in the frozen
analysis fits `Var(log w_carry)` against `L`, `N_c` or `K`, because designing
that test before the data exist would be designing it around an artifact.

**Open it when**: the immediate group returns. **This is the cheapest follow-up
in the parking lot** — the data will already be on disk, and it needs a fresh
task with a fresh pre-registration, not a re-run.

## P6 — variance reduction, rather than more `N_c`

`[E]` At `L = 128` the matched `R` needed to certify P2 is in the thousands. That
is a **variance** problem, not a population-size problem, and no amount of `N_c`
touches it.

`[C]` Nothing in this campaign investigates control variates, common random
numbers across `N_c` rungs, stratification, or a lower-variance estimator of the
same quantity.

`[J]` **Common random numbers across rungs is the obvious first idea**, and it is
worth naming precisely because it is not obviously admissible: it would break the
independence the entire uncertainty model rests on, and the two rungs are
different population sizes so the coupling is not straightforward. But it attacks
`Var(Delta)` directly — the exact quantity every verdict in this campaign is
limited by — rather than attacking `Var(I)` twice over.

**Open it when**: campaign D confirms the level route is unaffordable, i.e.
almost certainly.

## P7 — `T = L` is used and never tested

`[E]` `METH-TREQ-001` is `epistemic_status: unsupported` and its own
`cheap_available_test` has still never been run. `[E]` This campaign uses `T = L`
because the entire reuse corpus does, **not** because the claim supports it.

`[J]` Out of scope here and it stays out: mixing a horizon question into a
population-size calibration would confound both. `[E]` But it is Tier-0 in
`NUMERICAL_CAMPAIGN_CHARTER.md` §1 (R1), it is near-zero compute, and it moves
the entire programme's `T` budget by a factor of two in either direction.

**Open it when**: before any production campaign is designed. `[J]` Arguably it
was owed before this one.

## P8 — Linux `MaxRSS` for this sampler

`[E]` Every memory figure in this package is macOS `ru_maxrss`. No cluster
`MaxRSS` for this sampler exists anywhere in the repository.

`[J]` Not a task — one accounting query, in `RUCHE_RUNBOOK.md` §7. Parked here
only so it is not lost if the campaign is submitted by someone who does not read
§7.

## P9 — REFINED PRODUCTION

`[E]` Deliberately not designed. Its `lambda` window must come from the observed
**rough** crossing region and from nothing else — not from `sqrt(zeta)`, not from
`zeta^(1/3)`, and **not** from this campaign's 7-point locator window, which is
an `L <= 64`, `N_c = 1024` artefact until proven otherwise.

**Open it when**: rough production has run. Not before, and not by extrapolating
this campaign's grid.

## P10 — campaign A's lower step is under-powered and stays that way

`[E]` At `R = 48` the `2048 → 4096` step's `Delta` half-width is ~1.28× `tau_I`;
a P2 pass there would need `R ≈ 80`.

`[J]` Not fixed, on a judgement: P2 is evaluated at the **top** step, and the
lower step's job is P3 and P5, which are comparisons rather than tolerance tests.
Raising `R` to 80 across campaign A would add ~330 core-hours to resolve a step
that is not load-bearing. `[E]` Recorded because it is a limitation of the
design, not an oversight, and because a reader who later wants a P2 verdict at
that step should know it was priced and declined.
