# Pre-registered falsification plan — FROZEN

TASK-2026-09-02-SMC-HIGHRUNG-LAMBDA, brief §12.

**This is the PLAN. It is frozen before any new datum exists and it contains no
results.** Outcomes go to `FALSIFICATION_RESULTS.md`, which does not yet exist
and must not be created until the data are in. The machine-readable copy is the
`falsification_targets` block of `analysis_spec.yaml`; the two must agree.

Every criterion is evaluated by `analysis/combined_analysis.py` and nothing in
it is tuned after the fact.

## The two tolerances everything hangs on

```
tau_step = 0.0732   one lambda-grid step in CMI at L = 128
                    = |dI/dlambda| * delta_lambda = 7.321 * 0.010
tau_plot = 0.0146   0.2 of a grid step
```

`tau_step` is the *practically relevant absolute tolerance* the brief asked for,
and it is tied to the plotting resolution rather than chosen post hoc: a
residual finite-`N_c` displacement of `tau_step` would shift a point by a full
grid spacing and make `CMI(lambda)` uninterpretable. It requires
`SEM(Delta) <= 0.0374`; the design delivers **0.0231**, so it is testable.

`tau_plot` is what the *final* production plots actually want — a point that
moves by less than a fifth of a grid step. **It is pre-registered as NOT
achievable by this campaign.** Certifying it needs `SEM(Delta) <= 0.0075`, i.e.
`R ≈ 436` per rung at L = 128 against the 32–48 budgeted, roughly a 10× cost
increase. Recording that gap is a deliverable of this task. Claiming the
tolerance was met would not be, and no verdict below is permitted to imply it.

## Projected resolution (computed before submission, from measured variances)

| quantity | projected SEM | 95 % half-width | MDE (80 %, two-sided) |
|---|---:|---:|---:|
| `Delta_256->512` | 0.0231 | 0.0453 | 0.0648 |
| `Delta_512->1024` | 0.0225 | 0.0442 | 0.0632 |
| `Delta_1024->2048` (optional arm) | 0.0245 | 0.0480 | 0.0686 |
| `d_-`, `d_+` at L = 64 | 0.0079 | — | — |
| `d_-`, `d_+` at L = 128 | 0.0225 | — | — |
| `q` at L = 64 | 0.0137 | — | — |
| `q` at L = 128 | 0.0390 | — | — |

Projected variances come from the measured L = 128 ladder
(`3.98e-2 / 3.55e-2 / 1.80e-2` at `N_c = 64/128/256`) extrapolated at the
**pessimistic** frozen γ = 0.571 rather than at γ = 1, which would flatter the
design.

---

## F1 — L = 128, mean CMI still moves materially between N_c = 256 and 512

**Statistic.** `Delta_256->512 = I(512) − I(256)`, bootstrap over independent
populations, B = 10000, seed 20260902. `I(256)` is ARM2's completed `R = 64`
block; `I(512)` is `armA512`'s `R = 48`.

| verdict | criterion |
|---|---|
| **SUPPORTED** | 95 % CI of `Delta` excludes 0 **and** `abs(Delta_hat) >= tau_step` |
| **KILLED** | 95 % CI of `Delta` lies entirely inside `±tau_step` |
| **INCONCLUSIVE** | anything else — including a CI that excludes 0 with `abs(Delta_hat) < tau_step`, which means real but practically small motion |

**Prior expectation.** The two completed steps are −0.0990 and −0.1213, both
above `tau_step`. If the pattern continues F1 is SUPPORTED at ~4–5 sigma.

## F2 — L = 128, mean CMI still moves materially between N_c = 512 and 1024

Same statistic on `Delta_512->1024` (`armA512` `R = 48` vs `armA1024` `R = 32`),
same three criteria.

This is the target the campaign exists for. F1 and F2 together are what
distinguish stabilisation from continued drift; **neither is defined by whether
the rungs fit `I_inf + B/N_c`.**

## F3 — variance continues to decrease usefully with N_c at L = 128

**Statistic.** `gamma = −dlogVar/dlogN_c`, bootstrap over independent
populations, evaluated on a ≥3-rung window scan (full, drop-smallest,
drop-largest and every contiguous sub-window of length ≥ 3) over
`N_c ∈ {64, 128, 256, 512, 1024}`. A single-window exponent is not a
measurement.

| verdict | criterion |
|---|---|
| **SUPPORTED** | full-window bootstrap 95 % CI of `gamma` contained in `[0.5, 1.5]` |
| **KILLED** | CI upper bound < 0.5 on the full **and** drop-smallest windows |
| **INCONCLUSIVE** | anything else |

**The exponent is NOT required to equal 1.** The band is `[0.5, 1.5]`, inherited
unchanged from the parent spec.

**Binding note.** `gamma` is a *variance* diagnostic. It has no authority over
F1 or F2, which are about the *mean*. A SUPPORTED F3 never licenses a
stabilisation claim, and the two must not be reported as if one implied the
other. ARM1 came back γ = 0.905 SUPPORTED at L = 96 while its mean was still
falling by 0.073 from `N_c` 256 to 512 — that pair is the standing example.

## F4 — a high-N_c low-L stencil gives a reproducible local CMI(lambda) curve

**Statistic.** The S1–S4 criterion frozen in `SMOOTHNESS_CRITERION.md`,
evaluated on `armB` (L = 64, `N_c = 1024`, `R = 96` per lambda).

| verdict | criterion |
|---|---|
| **SUPPORTED** | S1, S2, S3 and S4 all pass |
| **KILLED** | S1 fails (a lambda's own independent replicates disagree beyond their joint error) **or** S4 fails (one population is carrying the mean) |
| **INCONCLUSIVE** | S1 and S4 pass but S2 or S3 is undetermined at the achieved SEM |

## F5 — the same spacing remains usable at L = 128

Same S1–S4, evaluated on `armC`'s two neighbouring lambdas plus `armA512`'s
central point, all at `N_c = 512`, `R = 48`. Same three criteria, with S1
failing at L = 128 as the kill.

## F6 — the chosen delta_lambda is neither too coarse nor buried

**Statistic.** `r = min(abs(d_-)/SEM(d_-), abs(d_+)/SEM(d_+))`, per L class.

| verdict | criterion |
|---|---|
| **SUPPORTED** | `2 <= r <= 20` at **both** L classes |
| **KILLED** | `r < 2` at either L (buried below the statistical resolution — the spacing is unnecessarily fine) **or** `r > 20` at both (unnecessarily coarse) |
| **INCONCLUSIVE** | the band is met at exactly one L |

**Projected.** `r ≈ 7.3` at L = 64 and `≈ 3.3` at L = 128, so SUPPORTED is the
pre-registered expectation. The per-L verdict (`too coarse` / `approximately
appropriate` / `unnecessarily fine`) is reported separately and is the input to
choosing the final production grid.

## F7 — apparent jaggedness survives an independent-population bootstrap

**Statistic.** Weighted chi-square of the three stencil points against the
best-fitting straight line in lambda, 1 dof. The null ensemble places the fitted
line at the three lambdas and adds each lambda's own bootstrap fluctuation about
its own mean — centring the null on the *observed* points instead would bake the
jaggedness into the null and force `p ≈ 1` by construction.

| verdict | criterion |
|---|---|
| **SUPPORTED** | bootstrap `p < 0.05` — the jaggedness is real, not sampling noise |
| **KILLED** | bootstrap `p > 0.32` — within 1 sigma of the smooth-line null |
| **INCONCLUSIVE** | `0.05 <= p <= 0.32` |

**Pre-registered expectation: KILLED.** A weighted quadratic fit to the existing
`dtau_mult = 12` corpus at `zeta = 0.35` over `lambda ∈ [0.25, 0.37]` already
returns `chi2/dof` of 1.15, 0.95, 0.78, 1.38 and 0.60 at L = 64, 80, 96, 112 and
128 — the historical jaggedness is *already* consistent with sampling noise
before a single new core-hour is spent. Writing that down now is the point of
pre-registering F7. **A KILLED F7 is a real, reportable negative result and must
be preserved as such** (charter §4.4), not quietly dropped because it says the
interesting-looking wiggle was never there.

---

## What no outcome may be used to conclude

Repeated here because these are the ways this analysis would most easily go
wrong, and they are enforced in `analysis_spec.yaml`:

- No universal `1/N_c` finite-population bias law. `I = I_inf + B/N_c` is
  printed as a description of the observed rungs only; it is never
  extrapolated, never treated as ground truth, and never used to decide
  convergence.
- No conversion of VIF or `N_c` into a bias rule.
- No `N_c(L, zeta, lambda)` law from three cells.
- Nothing about `lambda_c(zeta)`, the phase boundary, or any critical exponent.
- No claim that more `R` repaired a finite-`N_c` bias; it cannot.
- No imposed monotonicity in lambda.
