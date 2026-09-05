# RESEARCH_MEMO — TASK-2026-09-06-NC-ZETA-MOCKPROD

Charter Stage 9. Ten sections. **Not a manuscript.** Labels `[E]` `[I]` `[C]`
`[J]`.

---

## 1. The question investigated

`[E]` As `N_c` increases at fixed `zeta`, how much do the `CMI(lambda)` curves
and their rough `L = 48` vs `L = 64` crossing locations still move? Asked at
`zeta in {0.10, 0.20, 0.70}`, `L in {32, 48, 64}` with `T = L`, over
`N_c in {128, 256, 512, 1024, 2048}`, nine `lambda` per `zeta`, `R = 16`.
`zeta = 0.35` is a reference row.

`[E]` The deliverable is a **qualitative** per-rung status and a "smallest `N_c`
that looks adequate" recommendation with a caveat. Not a certification, not an
equivalence test, not a phase boundary, not a law.

## 2. Why it matters

`[E]` Every production population this programme holds is at `zeta = 0.35`, and
there is no basis anywhere for choosing `N_c` at any other `zeta`. `[E]` Even at
0.35 the answer is `R`-limited: the `1024 -> 2048` crossing shift there was
`-0.0035` against a bootstrap SE of `0.0025–0.0028`. `[J]` At `zeta = 0.70` the
difference between running production at 256 and at 2048 is an 8× cost factor on
the most expensive `zeta` in the diagram.

## 3. What was previously known

`[E]` The `N_c` ladder at `zeta = 0.35` to `N_c = 8192`
(`NC-PLATEAU-CALIBRATION`), and 17-point `CMI(lambda)` curves there with an
`INTERIOR` `L48-L64` crossing at `0.23691` (`MOCK-LOWLAMBDA-EXTENSION`).
`[E]` Two predecessors attacked the certification form of this question:
`NC-ZETA-CALIBRATION` reached Gate A with `Reformulate` because its brackets were
left-censored by the historical `sqrt(zeta)` grid; `NC-ZETA-STAGE1` repaired the
brackets and asks a stricter question. `[E]` Neither ran.

## 4. Which candidates were eliminated, and why

`[E]` Five design options; the red team **killed all five**.

- **C2 (`R = 24`)** — +1 010 core-hours (+50 %) for a 1.22× SEM reduction, on a
  survey that disclaims precision. Killed on cost, not correctness.
- **C3 (drop `zeta = 0.70`)** — removes the high end of a question whose subject
  is how behaviour changes between low and high `zeta`. Its content survives as
  the conditional interlock.
- **C4 (drop `L = 32`)** — `L = 32` is the only diagnostic that shows small-`L`
  drift, and at `zeta = 0.35` it is what revealed that `L32-L48` was
  `STILL_BOUNDARY` while the other two pairs were `INTERIOR`.
- **C5 (reuse the historical corpus)** — zero exact-compatible populations exist;
  what does exist has no `N_c`, `T`, `dtau_mult` or resampling metadata.
- **C1 (the committed design)** — killed **on its analysis half only**, by the
  red team executing the shipped script on injected data at the noise level
  measured in the real corpus. The measurement half survived every attack.

`[J]` The kills of C2–C4 all have the same shape: each trades a **diagnostic**
for compute. The one legitimate trade is a **confirmation**, which is what the
`zeta = 0.70`, `N_c = 2048` rung is, and why it is conditional rather than cut.

## 5. What survived

`[E]` **The measurement package, unchanged.** 16 arms, 6 512 populations, grids,
ladder, seeds, cost model and SLURM all survived independent attack; the red team
states no population needs to change. `[E]` C1 returns as **C1′** after the four
repairs, which are applied.

## 6. The evidence

`[E]` **The `N_c^0.1871` rate law is refuted** by the rungs the campaign that
adopted it has since returned — `L=64, N_c=4096/8192` measured 5.05/5.11 ms
against 6.57/7.48 predicted; `L=128, N_c=2048` 22.45 against 31.7. It survived
the red team's attempt to break the refutation on a sample-size and `lambda`
confound: `lambda`-matched medians at `L = 64` are flat over a 128× range in
`N_c`. Re-checked on every preflight run.

`[E]` **Nothing is reusable.** 5 185 stored populations, 136 distinct cells,
**zero** exact-compatible with any of the 407 design cells.

`[E]` **The grids bracket both open positions of `DISP-PHI-001`** with `>= 2`
grid points beyond each and a worst end-margin of 6.9 `tau_lambda`, and every
grid is strictly wider than an independently re-derived law-agnostic bracket
rule on both sides. Arithmetic re-derived by hand by the red team.

`[E]` **The survey's power, measured rather than assumed.** At `R = 16` the MDE
is 2.9–5.8 % of CMI at `L = 32`, 4.3–8.6 % at `L = 48`, and **7.9–13.3 % at
`L = 64`**, against measured top-rung drifts at `zeta = 0.35` of 0.1–2 %.

## 7. The remaining uncertainty

`[C]` **At `L = 64` the top of the ladder cannot be resolved at `R = 16`**, and
that is one of the five questions the brief asks. It is not fixable by tuning
`R`: 3 % at `L = 64, N_c = 1024` needs `R ~ 190`. `RECOMMENDATION.md` §1.
`[C]` `f_lam` at `zeta = 0.10, lambda = 0.040` is a ~4× extrapolation below its
calibrated window — the package's weakest input.
`[C]` `rho(0.70)` is understated by 7–9 % (convex `rate(zeta)`, chord
interpolation) and has **no production-regime validation anywhere**.
`[C]` **No Ruche `MaxRSS` exists anywhere in this repository**; all memory
sizing descends from local macOS `ru_maxrss`.
`[C]` The premise *"the `N_c -> infinity` target is unknown, so this is drift and
never bias"* may be false: `CMI(N_c) = c0 + c1/N_c` fits all six rungs at one
`zeta = 0.35` cell with `chi²/dof = 0.92`. One cell; parked as a child task.
`[E]` **The declared kill criterion cannot fire.** It requires every rung
`INCONCLUSIVE`, i.e. median SEM/\|CMI\| above 0.05, and the corpus says
`<= 0.030` at every `N_c >= 256`. `[I]` That is good news about `R = 16` at the
low rungs and it means the charter's kill criterion is inoperative as written —
recorded rather than quietly re-tuned.
`[E]` External novelty is `UNRESOLVED`: no external search was performed.

## 8. The actual contribution, without inflation

`[J]` **A replication at three new `zeta` of a measurement previously made at
one, plus five corrections to the machinery that makes it.** It is not novel, it
reveals no mechanism, it connects no fields, and `NOVELTY_GATE.md` classifies it
as replication and three rediscoveries.

`[E]` What is genuinely new here is smaller and duller than the campaign: the
refutation of an inherited rate law using the data collected to test it; a
grid-coverage check that tests **span** rather than centre, which is the precise
defect that stopped `NC-ZETA-CALIBRATION`; and a measured statement of what
`R = 16` can and cannot see.

`[J]` **The most useful output of the task so far cost no compute at all**: five
defects found in its own machinery before submission, three of them by
executing the code rather than reading it. Two — the recommendation rule and the
`INCONCLUSIVE` gate — would have produced a confidently wrong headline row from
a perfectly good campaign.

## 9. Reusable artifacts

`[E]` `tools/cost_model.py` with a refit that re-derives its own literals and
re-checks the refutation on every run; preflight `P9` (grid **span** coverage)
and `P17` (runner path resolves from **this** arm); 14 injected-fault negative
controls; a 13-case smoke suite in which the classifier is shown failing where it
must; `analysis/mockprod_analysis.py` implementing a frozen spec with a
simulated, `R`-aware critical value and a printed MDE; and the first production
package at any `zeta` other than 0.35.

## 10. The next human decision

`[E]` **Whether to submit, given §7's first paragraph.** Everything else is
downstream of that. Then: whether to authorise the `f_lam` probe (minutes,
local, needs approval), and whether to open the `c0 + c1/N_c` child task.

`[E]` Terminal state `READY_FOR_HUMAN_SUBMISSION`. **No agent submits.**
