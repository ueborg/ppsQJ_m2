# NOVELTY_GATE — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

The duplicate gate, run **before** any candidate is described as new. Command,
for every candidate:

```bash
.venv/bin/python3 research/tools/find_predecessors.py "<candidate statement>"
```

Dead records — withdrawn, retired, contradicted, superseded — are boosted by the
search, not filtered, because they are the ones a fresh candidate is most likely
to be quietly re-running.

## What "no predecessor found" means here, and what it never means

`[E]` **`no predecessor found` in the table below means: none found under the
searches actually performed.** The searches actually performed were
(a) `find_predecessors.py` over `research/state/**`, including withdrawn,
retired, contradicted and superseded records, and (b) a hand search of the eight
predecessor task archives named in `SOURCE_REGISTER.md`.

`[E]` **No external prior-art search was performed by this task** — not on
empirical plateau-detection protocols in interacting-particle Monte Carlo, not
on the `N`-dependence of SMC bias, not on crossing-estimator robustness under a
common estimator offset. `[I]` Therefore **`no predecessor found` NEVER means
"novel in the literature" here**, and with respect to external literature the
novelty of C2 and C3 is **UNRESOLVED**, not established. `[J]` A failed keyword
search would be evidence about the search; no search at all is not even that
(charter §3, §4.2). `NOVELTY_MATRIX.md` §"What was NOT searched" records the
same limitation and states that the search is owed before anyone describes C3's
outcome as a contribution rather than as an internal calibration.

`[J]` **A statement about the local search, second.** The tool searches
`research/state/**` only. It cannot see the execution plane, so it cannot tell
this task whether a *predecessor task* did the same thing. That half of the gate
was done by hand against the eight predecessor archives named in
`SOURCE_REGISTER.md`, and the results are in the "predecessor task" column
below. A clean canonical search is not, by itself, a novelty result.

---

| candidate | closest canonical predecessor | score | closest predecessor TASK | classification |
|---|---|---:|---|---|
| **C1** high-`N_c` plateau reachable at `L = 64` | `OBS-BL-001` (retired) / `CB-MIPT-001` | 0.29 / 0.28 | `TASK-2026-09-02-MOCK-PRODUCTION` measured `L = 64`, `N_c = 2048` at three `lambda`, `R = 24`; left shape convergence unresolved | **corroboration**, extended: same cell, two new rungs, `R` raised on evidence |
| **C2** correction additive in `lambda` | `CB-AMP-096-001` (withdrawn) | 0.41 | `TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING` *suggested* additive displacement can fail while multiplicative survives — as a hypothesis, at a different cell | **no predecessor found** for a measured `Delta(lambda)` in the locator region |
| **C3** locator converges before absolute level | `OBS-BLPROD-001` | 0.36 | none. No matched cross-`L` `N_c` comparison exists anywhere in the corpus | **no predecessor found** |
| **C4** `L = 96` enters a simpler high-`N` regime | — | — | `TASK-2026-09-01-SMCRUCHE-READY` built the three lower rungs; `TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING` fitted an `L = 96` ladder this task cannot reproduce | **provenance repair** — see below |
| **C5** `N_c = 2048` materially better at `L = 128` | `CB-WINDOW-001` (thematically) | 0.21 | `TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING` designed exactly this as its "Design 4", costed it at 901 pessimistic core-hours and **did not recommend it** | **replication of a rejected design**, with the rejection's reasoning answered |
| **C6** discretisation-stable limit | `CB-WINDOW-001` / `DISP-WINDOW-001` | 0.41 / 0.35 | `TASK-2026-08-30-SMCSTAT` `AN7_scheme_chunking` ran `dtau_mult ∈ {3,6,12}` at `K = 232/116/58` on a smaller cell | **rediscovery at production geometry** — the axis is not new, the cell is |

---

## The three entries that need more than a word

**C4 is classified `provenance repair`, and that is the honest label.**
`[E]` The accepted framing says a clean `1/N` is rejected at `L = 96`. This task
rebuilt the `L = 96`, `lambda = 0.3032` ladder from raw files and got
`chi2 = 1.90` on 1 dof, `p = 0.168` — not rejected
(`agent_reports/numerics.md` F3). `[I]` The predecessor's figure
(`chi2 = 10.54`, 3 dof) needs a four-rung ladder, which that cell does not have.
`[J]` This task does **not** assert the predecessor is wrong: it asserts it
cannot locate the ladder behind the number, and that the `L = 96` half of the
accepted framing should not be leaned on until someone does. Campaign C is
justified with or without it.

**C5 is a replication of a design a predecessor explicitly declined.** `[E]`
`TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING` ranked the `L = 128`, `N_c = 2048`
rung last of four and wrote: *"21× the cost of design 1, testing one point on
one ladder. Both live mechanisms are consistent with a wide range of outcomes
here."* `[J]` That reasoning is about **mechanism discrimination**, and it is
correct on its own terms — campaign D discriminates no mechanism. This task runs
it for a different purpose, **production adequacy screening**, at `R = 16`
rather than that design's `R = 32`, at 502 rather than 901 core-hours, and
labelled from the outset as unable to certify convergence. `[J]` The predecessor's
objection is answered, not ignored: it is why campaign E is in the same
submission group, and why campaign D is the smallest rung that can answer its
question rather than the one that would answer a question it cannot.

**C6 is a rediscovery, and calling it anything else would be wrong.** `[E]` The
`dtau_mult` axis was run before. `[E]` What is new is the geometry: `L = 64`,
`T = 64`, `K = 816/408/204`, production `lambda`, `R = 48`, and both `N_c`
reported separately because the `K`-dependence may itself depend on population
size. `[J]` The predecessor's numbers at `K = 232/116/58` are flat to about one
SEM and settle nothing here.

---

## No novelty language is used anywhere in this task

`[J]` Checked deliberately. `RESEARCH_MEMO.md` and `RECOMMENDATION.md` describe
this campaign as **calibration** and **screening**. Nothing in the package is
called novel, a first, a discovery or a contribution, and no arm's value rests
on being unprecedented. The one place a "first" would be defensible — the direct
peak-RSS measurement of this sampler, which no predecessor performed
(`agent_reports/numerics.md` F5) — is reported as a **defect found in an
inherited model**, which is what it is.
