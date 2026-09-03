# CAMPAIGN_DESIGN — TASK-2026-09-03-NC-PLATEAU-CALIBRATION

The design, what is new, what is reused, and what is deliberately absent.
Regenerate every artifact from `tools/design.py` + `tools/build_arms.py`;
hand-editing a manifest is an error the preflight fails on.

Labels `[E]` `[I]` `[C]` `[J]`.

---

## 0. Everything held fixed

```
zeta = 0.35          T = L          systematic resampling
guided proposal, proposal_c = zeta, exact RN compensator
dtau_mult = 6 (CERTIFIED)   -- except campaign E, where varying it IS the experiment
observable OBS-CMI-001, estimator cmi_weighted_mean
sampler support/instrumented.py, sha256 0a33c403..., BYTE-IDENTICAL to the file
that produced every reused population
K = ceil(2 * lambda * (L-1) * T / dtau_mult)          exact, not estimated
```

`[E]` No silent code-path change. `shared/run_cell.py` was modified to record
more, never to compute differently, and `tools/reproduce_check.py` demonstrates
that by re-executing completed predecessor populations and requiring the
per-clone trajectory to be **bit-identical** (`VALIDATION.md` §3).

## 1. The frozen tolerances, and the order they were chosen in

`[J]` `tau_lambda` is fixed **first** and `tau_I` is derived from it. The
programme needs a transition *location*; an absolute-CMI tolerance with no
decision attached is a number chosen for convenience.

```
tau_lambda   = 0.004      crossing-location tolerance          <- PRIMARY
|dD/dlambda| = 2.965      MEASURED at the interior crossings   <- not assumed
tau_D        = 0.0118     = tau_lambda x 2.965                 tolerance on a cross-L difference
tau_I        = 0.006      = tau_D / 2                          per-curve, WORST CASE
```

`[E]` The slope is the smaller of the two measured interior-crossing slopes
(2.965 for L32−L64 across 0.2232→0.2332; 4.052 for L48−L64 across
0.2332→0.2432). The smaller is adopted because a smaller slope converts a given
CMI error into a **larger** `lambda` error. `[E]` `tau_I` assumes the two
curves' displacements do not cancel at all and therefore add — which is a worst
case, and testing whether it is the real case is what campaigns B and B2 are
for. Full derivation and provenance: `SUCCESS_CRITERIA.yaml`.

## 2. The six campaigns

### A — deep central `N_c` ladder, `L = 64`, `lambda = 0.3032`

| `N_c` | R after | fresh here | status |
|---:|---:|---:|---|
| 1024 | 96 | 0 | reused whole (`SMC-HIGHRUNG-LAMBDA/armB`) |
| 2048 | 48 | 24 | **topped up** from 24 (`MOCK-PRODUCTION/mockL64nc2048`) |
| 4096 | 48 | 48 | new |
| 8192 | 48 | 48 | new |

`[J]` **`R = 48`, not the brief's preferred 24, and the reason is statistical
rather than budgetary.** The brief anticipated cost forcing `R` *down*. The
measured per-population spreads force it *up*: at `R = 24` the `Delta` half-width
at the `4096 -> 8192` step would be ~0.0071, i.e. 1.2× `tau_I`, so the arm could
not satisfy P2 whatever the data did. At `R = 48` it is ~0.0058, just inside.
`[E]` The `R` needed at each step is printed by the analysis for every ladder, so
this judgement is auditable against the returned data rather than only against
the prediction.

`[E]` **What `R = 48` still does not buy.** The lower step `2048 -> 4096` has a
predicted half-width of ~0.0077, about 1.28× `tau_I`. A P2 pass **at that step**
needs `R ~ 80`. The design accepts this: P2 is evaluated at the **top** step,
and the lower step's contribution is to P3 and P5, which are comparisons rather
than tolerance tests. This is stated here so it cannot look like an oversight
later.

`[E]` **The manifest is ordered so `R = 24` is a valid sub-campaign.** Each of
the two new arms is a single cell with replicates in seed order, so
`--array=0-23` yields a clean matched-`R`-24 block A and `24-47` block B. If the
researcher wants the brief's literal `R = 24`, halving the array range is the
whole change.

### B — transition-region `N_c` ladder, `L = 64`

`[E]` Frozen 7-point grid, `Delta lambda = 0.005`:
`0.2182 0.2232 0.2282 0.2332 0.2382 0.2432 0.2482`, at `N_c = 512, 1024, 2048`,
matched `R = 48`, 21 cells.

`[E]` **Provenance of the window, and its limit.** The two pairs whose cross-`L`
difference changes sign in the *interior* of the measured 17-point grid do so
between 0.2232 and 0.2332 (L32−L64) and between 0.2332 and 0.2432 (L48−L64).
The window is those two brackets plus one guard step each side. `[E]` It is a
**locator region in `L <= 64` curves at `N_c = 1024`**. It is not
`lambda_c(zeta)` and nothing here may call it one. `[E]` It is not centred on
`sqrt(zeta)`, `zeta^(1/3)` or any other candidate law.

`[E]` Reuse: `N_c = 1024` at `lambda = 0.2232, 0.2332, 0.2432` holds 24
exact-compatible populations each and is **topped up** to 48, not recomputed.
72 populations reused, 264 fresh rows at that `N_c` instead of 336.

`[J]` `R = 48` here is the brief's own choice and the measurement supports it:
at `R = 48` the per-`lambda` `Delta_{1024->2048}` half-width is ~0.010, small
enough to resolve a shape distortion of the size the existing three-point
stencil could not.

### B2 — matched low-`L` reference ladders, `L = 32` and `L = 48`

`[E]` The **same** 7-point grid, the **same** three `N_c`, the **same**
`R = 48`, at `L = 32` and `L = 48`. Six arms. `N_c = 1024` is topped up at the
three `lambda` that already exist and is otherwise new.

`[J]` **This is an addition beyond the brief's literal §4, and it exists because
without it the brief's own §4B cannot be answered.** §4B asks whether the
crossing converges with `N_c`. A crossing needs two curves. The `L = 32` and
`L = 48` curves exist at `N_c = 1024` and nowhere else, so campaign B alone can
only move `N_c` on one side, which measures the `L = 64` displacement rather
than the crossing.

`[E]` **The first version of B2 was wrong and the frozen protocol caught it.**
It used only the three `lambda` shared with the measured 0.010 grid, at a sixth
of the cost. Running the crossing protocol on that grid showed both interior
crossings fall in the first or last interval, so **every** crossing is flagged
`ENDPOINT_INDUCED` by construction — the exact defect
`TASK-2026-09-03-MOCK-LOWLAMBDA-EXTENSION` was created to repair. The arm was
rebuilt on the full seven points. `[J]` 260 core-hours (12 % of the campaign) is
the price of the load-bearing question being answerable at all, and B2 is the
one arm the researcher can drop in a single line if they disagree.

### C — intermediate ladder, `L = 96`, `lambda = 0.3032`

`[E]` Existing rungs `N_c = 128, 256, 512` reused; `1024` and `2048` added at
`R = 24`.

`[E]` **`R = 24` here is a screening design and cannot certify convergence.**
The `Delta_{1024->2048}` half-width at `R = 24` is ~0.027, about 4.5× `tau_I`;
reaching P2 would need `R` in the hundreds. `SUCCESS_CRITERIA.yaml`
pre-registers that a small step at this `R` is `UNRESOLVED_R_LIMITED` and never
"converged". `[J]` `R = 16` versus 24 was considered: 24 was kept because the
existing `Delta_256` is `−0.0696 ± 0.0210` and `R = 16` would widen the new step's
interval by 22 % for a 33 % saving on the cheapest of the three high-`L` arms.

`[E]` The `L = 96` ladder is **less** characterised than the accepted framing
suggests: its three-rung `1/N` fit gives `p = 0.168`, not a rejection
(`agent_reports/numerics.md` F3). `[J]` Not rejected on 1 dof is close to no
information. The new rungs are the first real test.

### D — screening rung, `L = 128`, `lambda = 0.3032`, `N_c = 2048`, `R = 16`

`[E]` The primary purpose is one question: **is the `1024 -> 2048` change still
material?** `R = 16` gives a `Delta` half-width of ~0.026, which resolves a shift
of the size the `512 -> 1024` step showed (`−0.0602 ± 0.0234`) at about 2.3
sigma. `[E]` It **cannot** certify convergence: matched `R` for P2 at that step
is ~2 675, about 13 000 core-hours. `[I]` **Absolute-level plateau certification
at `tau_I` is unreachable at `L = 128` with this estimator at any affordable
`R`.** That is a design finding derivable before any new datum, it is why D is
labelled screening, and it is the strongest single argument for the locator
route.

`[E]` Runtime is recomputed from measurement, not inherited: ~31.4 h per
population central, ~44.0 h pessimistic, `--time=72:00:00` on `cpu_long`
(1.6× pessimistic, 2.3× central). Derivation: `COST_MODEL.md`.

### E — discretisation / continuous-time particle-limit test

`[E]` `L = 64`, `T = 64`, `zeta = 0.35`, `lambda = 0.3032`,
`dtau_mult ∈ {3, 6, 12}` giving `K = 816 / 408 / 204` (recomputed exactly from
the production implementation, not quoted), `N_c ∈ {64, 256}`, matched `R = 48`,
six cells.

`[E]` This is `TASK-2026-09-03-FINITE-NC-REQUIRED-SCALING`'s Design 1, unchanged.
`[E]` It is clean because the Feynman–Kac weight is exact at any window size, so
the **target measure is exactly unchanged** across the three arms; only *where*
selection is applied moves. `[E]` It is the only axis in the corpus that breaks
the `L`/`ln K` collinearity (`r` up to 0.99). `[E]` `dtau_mult` is a
discretisation control, never a physical parameter, and the `dtau_mult != 6`
rows may never be pooled with the production corpus.

`[E]` The `dtau_mult = 6` sub-cell is not redundant: no `L = 64`,
`lambda = 0.3032` population exists at `N_c = 64` or 256 anywhere, so it is the
control that anchors the comparison to production discretisation.

## 3. What is reused, and what that is worth

`[E]` 240 exact-compatible populations, verified against disk by
`tools/dedup_scan.py`, never recomputed. Full ledger: `REUSE_LEDGER.csv`.

| what | populations | core-hours not spent |
|---|---:|---:|
| `L = 64` `N_c = 1024` centre (`R = 96`) | 96 | ~136 |
| `L = 64` `N_c = 2048` centre (`R = 24`) | 24 | ~32 |
| `L = 64` `N_c = 1024` at 3 of B's `lambda` | 72 | ~37 |
| `L = 32`/`L = 48` `N_c = 1024` at 3 of B2's `lambda` | 48 | ~16 |
| `L = 96` `N_c = 128, 256, 512` | 112 | ~180 |
| `L = 128` `N_c = 64 … 1024` | 272 | ~1 480 |
| | **624 populations touched** | **~1 880 core-hours** |

`[J]` The `L = 128` and `L = 96` lower rungs alone are worth roughly what this
whole campaign costs. Recomputing any of them would have been the single largest
avoidable waste available.

## 4. `N_c` versus `R`, kept apart everywhere

`[E]` `N_c` controls the finite-particle approximation — drift and
within-population variance. `R` controls the uncertainty of the finite-`N_c`
population mean. **Increasing `R` does not eliminate finite-`N_c` drift;
increasing `N_c` does not give precise crossing statistics if `R` is too small.**

`[E]` The analysis reports both budgets for every cell and every verdict names
which one binds. `[E]` `UNRESOLVED_R_LIMITED` is a distinct verdict from
`STILL_DRIFTING`, and the first is not a weaker form of "converged".

`[E]` The corpus conflated them: `R` runs 96, 64, 48, 32, 24 across existing
rungs, so a step's `Delta` and its half-width move together for reasons that
have nothing to do with convergence. Every new ladder here is **matched-`R`**
within itself.

## 5. What is deliberately NOT here

`[E]` No `zeta` other than 0.35 in the immediate group. The low-`zeta` ladder is
prepared and blocked: the programme wants the `zeta = 0.35` calibration
understood first.

`[E]` No higher-`L` mock-production scan in the immediate group. Both are
prepared and blocked behind the calibration that would choose their `N_c`.

`[E]` No `T != L`. `METH-TREQ-001` is `unsupported` and this campaign does not
test it; `T = L` is used because the entire reuse corpus is at `T = L`.

`[E]` No smoothing, no interpolation replacing a measurement, no imposed
monotonicity, no removal of an inconvenient `lambda`, no value-based exclusion.
The results file carries an audit block asserting this about the run that
produced it.

`[E]` No thermodynamic-limit extrapolation anywhere.

`[E]` No `lambda_c(zeta)`, no boundary law, no exponent — from any outcome.
