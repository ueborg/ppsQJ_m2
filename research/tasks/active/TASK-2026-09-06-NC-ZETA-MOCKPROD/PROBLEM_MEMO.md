# PROBLEM_MEMO — TASK-2026-09-06-NC-ZETA-MOCKPROD

Charter Stage 1. **Frozen at `stage_1_problem`.** Written before any
investigator was dispatched and before any arm was built. Labels `[E]` `[I]`
`[C]` `[J]`.

---

## 1. The observed problem

`[E]` Every production population this programme has on Ruche is at
`zeta = 0.35`. The whole-repository scan
(`tools/inventory_existing.py`, 5 185 stored populations, 136 distinct cells)
finds `zeta in {0.05, 0.30, 0.35, 0.70}` present, but everything outside 0.35 is
a local architecture benchmark at `dtau_mult = 0` bookkeeping and `N_c` in
44–500, not a production cell.

`[E]` So the programme knows how `N_c` behaves at exactly one `zeta`, and even
there the knowledge is partial: at `zeta = 0.35`, `L = 48–64`, the
`1024 -> 2048` crossing shift is `-0.0035` with bootstrap SE `0.0025–0.0028`
(`TASK-2026-09-05-NC-ZETA-CALIBRATION`), so the binding budget there was `R`,
not `N_c`.

`[E]` Meanwhile the cost of a population is strongly `zeta`-dependent and runs
*opposite* to the expected statistical difficulty — the only `zeta`-resolved
timing in the repository (`TASK-2026-08-11-ALGRD/results/b0_L*.json`) gives a
per-clone-window rate ratio of `0.45 : 0.64 : 1.00 : 2.37` at
`zeta = 0.10 : 0.20 : 0.35 : 0.70`, independently re-derived here and agreeing
with the predecessor's table to better than 1 %.

`[I]` The practical consequence: a production campaign at any `zeta` other than
0.35 currently has **no basis at all** for choosing `N_c`, and the wrong choice
is expensive in both directions — too small wastes the whole campaign, too large
wastes it by a factor that at `zeta = 0.70` is measured in hundreds of
core-hours per rung.

## 2. The smallest precise research question

> **At each `zeta in {0.10, 0.20, 0.70}`, over the ladder
> `N_c in {128, 256, 512, 1024, 2048}` at `L in {32, 48, 64}`, `T = L`,
> `dtau_mult = 6`, systematic resampling, `R = 16` independent populations per
> cell: by how much does the measured `CMI(lambda)` curve, and the rough
> `L = 48` vs `L = 64` sign-change location, still move from one `N_c` rung to
> the next?**

`[E]` `CMI` is `OBS-CMI-001`. `lambda = alpha/(alpha+w)`. The answer is a
per-rung qualitative status in
`{CLEARLY TOO SMALL, STILL CHANGING, ROUGHLY STABLE, INCONCLUSIVE}` and a
practical "smallest `N_c` that looks adequate" row per `zeta`.

## 3. Why current approaches do not resolve it

`[E]` **The `zeta = 0.35` answer does not transfer.** Nothing in this programme
licenses carrying the `zeta = 0.35` local `1/N_c` coefficient to another `zeta`,
and `TASK-2026-09-05-NC-ZETA-CALIBRATION` states that prohibition explicitly.

`[E]` **The strict route was tried and stalled.** Two predecessors attacked the
certification form of this question. `NC-ZETA-CALIBRATION` reached Gate A with
verdict `Reformulate` — its brackets were left-censored by the historical
`0.51*sqrt(zeta)` grid and would have measured at the wrong `lambda` at most
`zeta`. `NC-ZETA-STAGE1` fixed the brackets but answers a stricter question:
certified `N_c^req(zeta)` under TOST at `tau_lambda = 0.004`. `[E]` The cost of
that stricter form is visible at the anchor: `NC-ZETA-CALIBRATION` measured
`R_req = 51-65` there against the 48 that had been run.

`[I]` The gap between them is a **practical** one. The researcher does not need
`N_c^req(zeta)` certified in order to size the next production campaign; they
need to know whether 128 is visibly hopeless and whether 512 or 1024 or 2048 is
where the curves stop visibly moving. That is a cheaper question with a cheaper
answer, and no artifact in the repository answers it.

## 4. Which decision changes with the answer

`[E]` The `N_c` chosen for the next production campaign at each `zeta`, and
whether such a campaign is affordable at all. `[J]` At `zeta = 0.70` the ladder
itself costs 85 % of this survey; if the answer there is "stable by 256", every
subsequent high-`zeta` campaign gets an 8× cost reduction relative to running at
2048 by default. If the answer is "still moving at 2048", high-`zeta` production
at `L > 64` is off the table under the current architecture, and that is worth
knowing before it is attempted rather than after.

## 5. Constraints and information structure

`[E]` `N_c` and `R` are **separate budgets**. This survey fixes `R = 16` and
varies `N_c`; it can therefore see `N_c` movement only down to the noise floor
`R = 16` provides, and it says so in every verdict.
`[E]` The `N_c -> infinity` target is unknown, so movement is **drift**, never
bias.
`[E]` `T = L` and `dtau_mult = 6` are held fixed, so `n_steps` is exactly
`ceil(2*lambda*(L-1)*T/dtau_mult)` and cost is entangled with `lambda`, which
itself moves with `zeta`.
`[E]` `DISP-PHI-001` is **open**, so where the crossing sits at a new `zeta` is
not known in advance and any `lambda` grid must span both open positions
(`CB-PHI-HALF-001`, `CB-PHI-LINEAR-001`) without privileging either.

## 6. The strongest case that the problem matters

`[J]` The programme's single largest recurring waste is running a campaign whose
population size was chosen by inheritance. `TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA`
extrapolated the runtime of every large `N_c` flat from a three-rung fit and was
30 % low; `NC-PLATEAU-CALIBRATION` corrected it with a `N_c^0.1871` growth law;
that law is in turn now refuted by its own returned data (§7). Each correction
cost a campaign. `[J]` A cheap empirical picture of where the curves stop moving
is the input that makes the next `N_c` choice a measurement instead of an
inheritance, at three `zeta` spanning a 7× range.

## 7. The strongest argument that the problem is artificial, already solved, or
unimportant

Four attacks, stated properly.

`[E]` **(a) It is a weaker version of a question already designed.**
`NC-ZETA-STAGE1` exists, is validated, and answers more. Running a survey that
deliberately declines to certify anything could be read as manufacturing a
cheaper deliverable rather than a different one.
`[I]` **What survives:** the two questions have different failure modes. A
certification returns `CONSISTENT BUT R-LIMITED` when `R` binds — which is
exactly what happened at the anchor — and that outcome tells a production planner
nothing about which rung to use. A survey returns a rung with a caveat. `[J]` The
honest statement is that this task is **strictly less informative and strictly
cheaper**, and is justified only if the researcher wants the cheap answer now.
That is a human decision and it has been taken: the brief commissions it.

`[E]` **(b) A survey with `R = 16` may not resolve anything.** At `zeta = 0.35`,
`R = 48`, the `1024 -> 2048` crossing shift was not separable from zero.
`[I]` **What survives:** the survey's primary product is the **curve**, not the
crossing shift. At `zeta = 0.35`, `R = 24` gave per-point `SEM/CMI` of 0.4–1.5 %
and 16/16 increments resolved at `r >= 2`; `R = 16` inflates SEM by 1.22. A rung
that moves the curve by more than a few per cent is visible; one that moves it by
0.3 % is not, and will be reported `INCONCLUSIVE` rather than `ROUGHLY STABLE`.
`[E]` The distinction between those two verdicts is pre-registered in
`ANALYSIS_SPEC.yaml` before any datum exists. `[J]` This attack is the reason the
`INCONCLUSIVE` class exists at all and is not collapsed into "stable".

`[E]` **(c) `ROUGHLY STABLE` will be read as convergence.** It is a
resolution-limited statement that reads like a physical one, and this project has
been burned by exactly that (`OBS-BL-001`, one label over two quantities).
`[I]` **What survives:** only if the wording is policed. `CLAIM_STRENGTH_AUDIT.yaml`
carries the rejected stronger wording for each conclusion, the analysis script
prints the prohibition next to every verdict, and the word `certified` is absent
from the package by construction (checked mechanically, negative control `N7`).

`[E]` **(d) The `lambda` grids could miss the crossing, making every curve
uninformative about the locator.** This is what killed `NC-ZETA-CALIBRATION`.
`[I]` **What survives:** the risk is real and is quantified rather than assumed
away. `LAMBDA_GRID_DECISION.md` shows each proposed grid spans both open
positions of `DISP-PHI-001` with `>= 6.8 tau_lambda` margin and `>= 2` grid
points beyond each prediction; the residual exposure is a boundary exponent
outside `[0.09, 1.14]` at `zeta = 0.70`, which is recorded, not hidden, and which
the pre-registered `ABOVE_GRID` / `BELOW_GRID` outcome classes will detect. `[J]`
Detecting it costs the campaign; not detecting it costs the campaign *and* the
conclusion.

## 8. What remains after the criticism

`[I]` A cheap, law-agnostic, matched-`R` empirical survey of finite-`N_c`
movement at three `zeta`, which cannot certify anything, cannot be extended into
a law, and answers one production-planning question that nothing in the
repository currently answers.

`[J]` Its most valuable outcomes are the negative ones: "still moving at 2048 at
`zeta = 0.10`" and "cannot tell at `R = 16`" are both more useful than a
comfortable "stable by 512" would be, because both redirect the next campaign.
