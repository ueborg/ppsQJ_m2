# What "clean local CMI(lambda) curve" means here — FROZEN

TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA, brief §4.
**Written before any new datum exists. Do not edit after data arrives.**

## The thing this must not become

The goal is a physically trustworthy local curve, not a visually pleasant one.
So, stated first and bindingly:

- **No smoothing.** No moving average, no spline, no LOESS, no kernel.
- **No regularisation** and no shrinkage toward a fitted form.
- **No fitting away** a point, and no reporting a fit in place of the points.
- **No discarding a point because it looks jagged.** The only exclusions are the
  four frozen ones in `analysis_spec.yaml`, all of which are about run status and
  non-finite clones, none of which look at the observable's value.
- **No imposed monotonicity.** CMI(lambda) is expected to fall through this
  window, but the physics is not invoked to *require* it, and a non-monotone
  triple is a result rather than an error.

A jagged curve that survives every test below is a finding. A smooth-looking
curve produced by any of the operations above is not.

## The three points and their derived quantities

At each L class, from the frozen stencil
`lambda_- = 0.2932`, `lambda_0 = 0.3032`, `lambda_+ = 0.3132`:

```
I_-  = I(lambda_-)      I_0 = I(0.3032)      I_+ = I(lambda_+)
```

each a mean over **independent populations**, each with an
**across-population** SEM. Within-clone spread is never used as this error bar.

Adjacent increments and the local second finite difference:

```
d_- = I_0 - I_-
d_+ = I_+ - I_0
q   = I_+ - 2 I_0 + I_-
```

Every uncertainty on these is a non-parametric bootstrap resampling the
independent populations of each lambda separately, B = 10000, seed 20260902.
Because the three lambdas use disjoint seed lanes, the three point estimates are
statistically independent and the bootstrap respects that.

At **L = 128** the central point `I_0` comes from `armA512` and the two
neighbours from `armC`, at the same `N_c = 512` and the same `R = 48`. There is
no duplicated central-lambda compute anywhere in the campaign.

## S1–S4 — the frozen criterion

A local curve is called **clean** at a given L when all four pass.

### S1 — replicate estimates are stable

At each of the three lambdas, split that lambda's `R` independent populations
into two disjoint halves (deterministic permutation, seed 20260902) and form
`m_A - m_B` with its own joint across-population error `s_AB`.

- **pass**: `|m_A - m_B| <= 2.5 * s_AB` at all three lambdas
- **fail**: any lambda exceeds it

This is the load-bearing one. If a single lambda's own independent replicates
cannot agree with themselves, nothing downstream is interpretable, and S1
failing is the kill criterion for F4 and F5.

### S2 — neighbouring differences are reproducible

- **pass**: the bootstrap 95 % CI of `d_-` and of `d_+` each exclude zero
- **undetermined**: either CI contains zero

`d_-` and `d_+` are the quantities the final production grid is made of. If they
are not individually resolved, the grid spacing is finer than the achievable
resolution, which is exactly the failure mode F6 exists to catch.

### S3 — the triple is compatible with a locally smooth curve

Report `q` with its bootstrap 95 % CI, always. Then:

- **pass, curvature resolved**: the CI of `q` excludes zero
- **pass, curvature unresolved but bounded**: the CI of `q` contains zero **and**
  its 95 % upper bound on `|q|` does not exceed `|d_-| + |d_+|` — i.e. the
  point-to-point structure is not zig-zagging by more than the curve trends
- **fail**: the CI of `q` contains zero **and** the bound on `|q|` exceeds
  `|d_-| + |d_+|` — the triple wanders more than it moves

**Pre-registered honesty about S3.** At the budgeted `R`, `q` is projected at
0.0055 (L = 64) and 0.0088 (L = 128) against `SEM(q)` of 0.0137 and 0.0390.
`q` is therefore expected to be **unresolved at both L**, and S3 is expected to
pass through its second branch. Curvature was never affordable at this budget
and the campaign is not designed to measure it. Saying so now is the point of
freezing this file; discovering it afterwards and calling it a result would not
be.

### S4 — no individual point wanders by many SEM against independent repeats

Within each lambda's `R` populations:

- **pass**: `max_i |m_i - mean| / sd <= z_R`, where
  `z_R = Phi^-1(1 - 0.01 / (2 R))` is the two-sided 1 % expected-maximum
  threshold for `R` draws (`z_96 = 3.29`, `z_48 = 3.03`, `z_32 = 2.91`);
  **and** dropping the single most extreme population shifts the cell mean by
  no more than one across-population SEM
- **fail**: either condition breaks

S4 is the guard against a curve that looks fine in the mean because one
population is carrying it.

## Reporting the spacing verdict (brief §4, last part)

Independently of S1–S4, the chosen `delta_lambda = 0.010` is reported as one of:

| verdict | rule |
|---|---|
| **too coarse** | `r = |d| / SEM(d) > 20` at **both** L classes |
| **approximately appropriate** | `2 <= r <= 20` at both |
| **unnecessarily fine** | `r < 2` at either L |

with `r` computed separately for `d_-` and `d_+` and the verdict taken from the
smaller. Projected: `r ~ 7.3` at L = 64, `r ~ 3.3` at L = 128, so
**approximately appropriate** is the pre-registered expectation. This verdict is
the input to choosing the FINAL production lambda grid, and it is the only thing
this task is licensed to say about that grid.

## What S1–S4 do NOT distinguish

The brief asks the diagnostics to separate four things. They separate three
cleanly and the fourth only partially, which is worth stating plainly:

1. **physical slope/curvature** — `d_-`, `d_+`, `q` and their CIs;
2. **statistical uncertainty** — the across-population SEMs and S1/S4;
3. **genuinely irregular, non-reproducible behaviour** — S1 and S4 failing, and
   F7's bootstrap p;
4. **residual finite-N_c displacement** — **only partially.** A finite-N_c offset
   that is nearly the same at all three neighbouring lambdas cancels almost
   exactly out of `d_-`, `d_+` and `q`, and this stencil cannot see it. What it
   would show up in is a *difference between L classes* at matched `N_c`, and
   in ARM A's rung ladder at the central lambda. So the honest statement is that
   ARM A bounds the displacement at the centre and the stencil is largely blind
   to the part of it that is smooth in lambda. Any claim that the stencil
   "controls" finite-N_c bias would be wrong and must not be made.
