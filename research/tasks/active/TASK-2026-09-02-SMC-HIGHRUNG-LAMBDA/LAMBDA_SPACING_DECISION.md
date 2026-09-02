# Choosing delta_lambda — FROZEN before any new result exists

TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA, brief §2.

## Decision

```
lambda_-  = 0.2932
lambda_0  = 0.3032        (the reference)
lambda_+  = 0.3132

delta_lambda = 0.010, SYMMETRIC
```

## What the existing data were used for, and what they were not

The estimate below comes from `numB_cells.csv`
(`ce297b7f35f6808564179cef4862825c…`, an aggregate of the historical Cut-B
corpus at `/Users/catlover1337/Downloads/pps_all_realizations.csv`). It was used
**only** to estimate the local slope and curvature of `CMI(lambda)` near
`zeta = 0.35, lambda ~ 0.3032`, in order to pick a spacing.

It was **not** used, and must not be cited, as evidence for a critical law, a
phase-boundary shape, `lambda_c(zeta)`, or any exponent. The historical scan is
the `0.51*sqrt(zeta)`-centred one, and `TASK-2026-08-11-MAPS` and
`TASK-2026-08-12-LAMC` between them showed it fails to bracket its own crossings
where the cost is. None of that matters for a local derivative, which is what
was taken from it and all that was taken from it.

## The measurement

Weighted quadratic fit of `CMI(lambda)` centred on `lambda = 0.3032`, over the
14 corpus points with `lambda` in `[0.25, 0.37]` at `zeta = 0.35`, weighting by
each cell's own across-realization SEM (`R = 12`):

| L | points | I(0.3032) | dI/dlambda | d2I/dlambda2 | chi2/dof | median SEM |
|---:|---:|---:|---:|---:|---:|---:|
| 64  | 14 | 0.4143 | −5.757 ± 0.275 | +55.1 ± 10.5 | 1.15 | 0.0206 |
| 80  | 14 | 0.3995 | −6.113 ± 0.251 | +48.1 ± 8.9  | 0.95 | 0.0230 |
| 96  | 14 | 0.3895 | −6.408 ± 0.412 | +51.2 ± 16.1 | 0.78 | 0.0416 |
| 112 | 14 | 0.3460 | −6.878 ± 0.482 | +95.4 ± 18.1 | 1.38 | 0.0398 |
| 128 | 14 | 0.3601 | −7.321 ± 0.516 | +87.6 ± 21.9 | 0.60 | 0.0426 |

Two things to notice.

**The slope is well determined and grows slowly with L** — from −5.76 at L = 64
to −7.32 at L = 128. Those are the two numbers the design needs.

**chi2/dof runs 0.60–1.38 across all five L.** The historical `CMI(lambda)`
curve at `zeta = 0.35` is *already statistically consistent with a smooth
quadratic*, within its own (large, `R = 12`) across-realization errors. In other
words the observed jaggedness looks like sampling noise before we spend a single
core-hour. That is why **F7 is pre-registered with the expectation that it comes
back KILLED**, and why the campaign is framed as measuring the resolution needed
rather than as chasing a real irregularity.

### One caveat, stated rather than buried

The corpus was run at `dtau_mult = 12.0`; this campaign runs the certified
`dtau_mult = 6.0` (see `DUPLICATE_COMPUTE_AUDIT.md`). The slope carries an
unquantified discretisation systematic. It does not change the decision: at a
slope 30 % smaller the L = 128 signal-to-noise at `delta_lambda = 0.010` is
still 2.3, and at 30 % larger it is 4.2 — either way `0.010` is the choice and
`0.005` is not.

## The trade-off, evaluated

Let `d` be an adjacent increment, `SEM(d)` its across-population error at this
campaign's budgeted `R`, `q` the second finite difference, and

```
nonlinearity per step = (1/2)|I''| dlam / |I'|
```

| delta_lambda | L=64 &#124;d&#124; | r = &#124;d&#124;/SEM(d) | L=128 &#124;d&#124; | r | q/SEM(q) at L=128 | nonlinearity |
|---:|---:|---:|---:|---:|---:|---:|
| 0.005 | 0.0288 | 3.6 | 0.0366 | **1.6** | 0.06 | 3.0 % |
| **0.010** | **0.0576** | **7.3** | **0.0732** | **3.3** | 0.22 | **6.0 %** |
| 0.015 | 0.0864 | 10.9 | 0.1098 | 4.9 | 0.51 | 9.0 % |

(`SEM = 0.00560` at L = 64, `N_c = 1024`, `R = 96`; `SEM = 0.01591` at L = 128,
`N_c = 512`, `R = 48`. Both are projected from the measured across-population
variances in `COST_MODEL.md`.)

## Why 0.010 and not the others

1. **0.005 is rejected on criterion B — too small.** At L = 128 the physical
   step would be 1.6 sigma. We would pay ~480 core-hours for a pair of points
   whose difference is buried in the statistical uncertainty. That is precisely
   "spending large resources measuring nearly identical points".

2. **0.015 is rejected as unnecessarily coarse.** It works statistically, but
   `0.010` already clears 3 sigma at the harder L and 7 sigma at the cheaper one,
   so the extra separation buys nothing the campaign needs. Meanwhile the
   nonlinearity per step rises to 9 %, which makes a three-point stencil a worse
   probe of a *local* curve, and a final production grid at 0.015 gives ~9 points
   across the `lambda in [0.25, 0.37]` region instead of ~13.

3. **0.010 sits where the physical increment is comfortably resolvable at both
   L classes and the local curve is still nearly linear across a step.**

4. **It is commensurate with the historical `refine` grid without duplicating
   it.** That grid's local spacing near `lambda = 0.3` is ≈ 0.0112 (points at
   0.2923, 0.3032, 0.3144). Choosing 0.010 keeps the new points comparable in
   density while landing on lambdas that *no existing run has ever used*
   — 0.2932 and 0.3132 return zero rows anywhere in the 20,355-row corpus. No
   compute is repeated and no two nearby lambdas can later be confused.

## Why symmetric

The one real argument for an asymmetric stencil would be to land exactly on the
historical `refine` lambdas (0.2923, 0.3144) so the new high-`N_c` points could
be compared against the old low-`N_c` ones at identical lambda. That argument
fails on inspection: those runs are `dtau_mult = 12.0`, `N_c = 128` only, and
carry no recoverable seed, so they are not poolable and not directly comparable
in any case (`DUPLICATE_COMPUTE_AUDIT.md`). Meanwhile the second finite
difference `q` is only a clean local curvature estimator on an **equally
spaced** stencil. Symmetric wins on both counts, and the brief's default stands.

## What is frozen

`delta_lambda = 0.010` and the three lambdas above are frozen as of this file.
`shared/preflight.py` refuses to pass any manifest whose lambdas are off this
stencil, so the choice cannot drift after the fact. The **verdict** on whether
0.010 was the right choice (too coarse / approximately appropriate /
unnecessarily fine) is computed from the new data by the frozen rule in
`SMOOTHNESS_CRITERION.md`, and that verdict — not this file — is what informs
the final production grid.
