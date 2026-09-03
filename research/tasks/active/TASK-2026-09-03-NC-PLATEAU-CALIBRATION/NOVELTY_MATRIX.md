# NOVELTY_MATRIX — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

Charter Stage 2. One row per comparator, along the eight axes the Skill names.

`[J]` **This task claims no novelty and needs none.** It is calibration. The
matrix exists to place the campaign against its closest predecessors so that
nobody later mistakes it for a new result, and to record where a comparator does
something this campaign does not.

---

## Comparator 1 — `TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING` (the direct parent)

| axis | comparator | this campaign |
|---|---|---|
| problem definition | is there a *rule* `N_c_req(L, zeta, lambda)`, analytic or diagnostic? | at which `N_c` does the estimator *measurably* stop moving, at three specific `L`? |
| information assumptions | Feynman–Kac mixing hypotheses; existing corpus | none analytic; new measurements plus exact reuse |
| mathematical mechanism | minorisation, `c_mu`, first-order bias expansions | none. Frozen empirical criteria P1–P5 |
| guarantee | `[E]` CONTROLLED negative: the bounds do not transfer | `[E]` EMPIRICAL only, and cell-specific by construction |
| empirical evidence | 9 ladders, existing corpus, 2 rejections of `1/N` | 3 280 new populations + 240 reused, at `N_c` up to 8192 |
| operational constraints | 43–901 core-hours across four designs | 2 180 core-hours immediate |
| computational cost | recommended 43 (design 1) or 55 (1+2) | design 1 is **included here as campaign E**, unchanged |
| reusable output | frozen designs and criteria | runnable packages, a frozen analysis, a measured cost and memory model |

`[E]` **Continuity, exactly**: campaign E *is* that task's Design 1, at its
configuration, with its pre-registered predictions and its
`INCONCLUSIVE` clause. `[E]` Its Design 2 is prepared as `conditional/cond_LOWZ_*`
and deliberately not in the immediate group. `[E]` Its Design 4, which it did
**not** recommend, appears here as campaign D at half the `R` and for a different
purpose (`NOVELTY_GATE.md`).

## Comparator 2 — `TASK-2026-09-02-MOCK-PRODUCTION` + `-LOWLAMBDA-EXTENSION`

| axis | comparator | this campaign |
|---|---|---|
| problem definition | are the raw `CMI(lambda)` curves smooth, and is the crossing interior? | is the `N_c` they were measured at adequate? |
| information assumptions | `N_c = 1024` adequate — **assumed, not tested** | that assumption is the object of study |
| mechanism | 17-point `lambda` scan, crossing bootstrap, endpoint protocol | same protocol, applied across `N_c` instead of across `lambda` |
| guarantee | curves smooth at `L <= 64`, `N_c = 1024` | the `N_c` at which that statement survives doubling |
| empirical evidence | 1 440 populations at one `N_c` | 3 `N_c` at 3 `L` on the same grid |
| cost | 61–221 core-hours | 2 180 |
| reusable output | the curves this campaign reuses wholesale | the calibration those curves need to mean anything |

`[E]` **The crossing and endpoint protocol is inherited unchanged**, and it is
what caught this task's own first design error: on three shared `lambda` every
interior crossing is `ENDPOINT_INDUCED` by construction, which is precisely the
defect `-LOWLAMBDA-EXTENSION` was created to repair (`CANDIDATES.md` C3).

## Comparator 3 — `TASK-2026-08-31-SMCCERT`

| axis | comparator | this campaign |
|---|---|---|
| problem definition | is the sampler certified, and what is the production rule? | at what `N_c` is it adequate? |
| guarantee | per-cell calibrated `B`; `N_c` from the conservative end of its CI; `R` afterwards; `CALIBRATION_REQUIRED` outside calibrated cells | **unchanged**. This campaign supplies calibration *inside* three more cells and replaces no rule |
| reusable output | the standing rule | the cells it can be applied in |

`[E]` `SMCCERT`'s rule is not superseded by anything here, and this task does
not propose to supersede it.

## Comparator 4 — `TASK-2026-08-30-SMCSTAT`, `AN7_scheme_chunking`

| axis | comparator | this campaign (E) |
|---|---|---|
| problem definition | does the resampling scheme × chunking interact? | does finite-`N_c` drift depend on window count at production geometry? |
| parameters | `dtau_mult ∈ {3,6,12}`, `K = 232/116/58`, `R = 24`, small cell, both resamplers | same `dtau_mult`, `K = 816/408/204`, `R = 48`, `L = 64` `T = 64` `lambda = 0.3032`, systematic only, `N_c ∈ {64, 256}` |
| evidence | means 0.3350 / 0.3373 / 0.3293 — flat to about one SEM | pre-registered E1 vs E2, both killing |
| reusable output | VIF and chunk-ratio diagnostics | a discretisation verdict at production geometry, or `INCONCLUSIVE` |

`[E]` Classified `rediscovery at production geometry` in `NOVELTY_GATE.md`. The
axis is not new; the cell is, and the comparator's numbers do not transfer.

---

## What was NOT searched, and what follows from that

`[E]` **No external literature search was performed by this task.** Not on
empirical plateau-detection protocols in interacting-particle systems, not on
`N`-dependence of SMC bias in continuous-time particle filters, not on
crossing-estimator robustness under a common estimator offset.

`[I]` So: **no statement anywhere in this package may be read as a claim of
priority, and none is made.** `[J]` The gap is real and worth naming precisely,
because a plausible piece of prior art exists in principle — the observation
that a common additive offset cancels in a difference-based locator is not deep,
and someone in the SMC or the finite-size-scaling literature will have written
it down. `[J]` If any future task wants to describe C3's outcome as a
contribution rather than as an internal calibration, that search is owed first.
A failed keyword search would be evidence about the search; **no search at all
is not even that.**
