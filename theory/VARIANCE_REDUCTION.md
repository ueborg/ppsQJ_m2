---
lifecycle: superseded
superseded_by: research/state/claims/VR-*
authoritative_for: nothing
marked: 2026-08-10
marked_by: phase4-migration
---

> **LIFECYCLE: SUPERSEDED.** Section 5 ("learned Doob: no headroom") was RETRACTED on 2026-08-09, but the retraction lives only in the frozen HANDOFF. Reading this file alone gives the retracted conclusion.
> Canonical scientific state is `research/state/`. Cite claim IDs, not this file.

# Variance Reduction for Guided Cloning (ppsQJ_m2) — methodology study

**Created 2026-06-16/17.** Self-contained record of a multi-round methodology
study on reducing the variance of cloning estimators for the QJ-PPS model.
One-line conclusion:

> The guided proposal c=zeta has solved the weight-degeneracy problem
> (ESS/N_c ~ 0.97-0.99 across the measured window). The per-run sampler is
> practically optimal. Remaining gains come from how observables are
> *differenced and combined*, not from a better proposal. Two estimation-side
> wins survive end-to-end validation, both modest (~2-3x) and nearly free, on
> DISJOINT observable families:
>   - coupled neighbouring lambda-points  -> ~2x on entanglement FSS differences
>   - compensated-count martingale CV     -> ~3x on the tilted activity ONLY

This was a five-round stress-test. Several promising directions were closed,
and TWO reported gains were corrected downward after end-to-end validation
(they were fixed-start artifacts). The corrections are themselves a result.

---

## 0. Setup and the unifying control formula

Guided cloning thins the click rate to lambda~_j = u_j * lambda_j and stays
exact via the residual log-weight
    log G = sum_clicks log[ zeta / u_{j_r} ] - integral sum_j [1 - u_j] lambda_j.
Production uses the scalar control u_j = c = zeta, for which the discrete factor
cancels and the weight is smooth:
    G = exp[ -(1-zeta) * Delta_Lambda ],   Delta_Lambda = integral lambda dt.
Any positive predictable control keeps exactness; only the variance changes.

## 1. The scalar proposal is solved (c = zeta)

- c-scan (L=48, zeta=0.3 and 0.1, mult=8): ESS/N_c PEAKS exactly at c=zeta
  (0.977 at zeta=0.3, 0.989 at zeta=0.1) and falls on both sides. An online-c
  adaptation converges to zeta. No headroom.
- Cost-aware check: ESS/wall naively favours c<zeta (1.53x at c=0.5zeta, since
  wall falls with fewer jumps) BUT the correct metric Var(observable) x wall does
  NOT. At c=0.5zeta the entropy estimate is ~1.9x noisier (lower ESS -> more
  coalescence), so Var x wall favours c=zeta by ~2x. Keep c=zeta as default.

## 2. Validated division of labor (end-to-end, full runs)

| observable family                          | tool           | validated gain |
|--------------------------------------------|----------------|----------------|
| entanglement FSS (S, CMI, KMR, slopes, nu) | coupled lambda | ~2x            |
| tilted activity <n>_Q                      | martingale CV  | ~3x            |
| SCGF theta / log Z                         | neither        | ~1x            |
| entanglement observables                   | martingale CV  | 1x (no effect) |

Both wins are real, free, and modest. They target DISJOINT observable families.

## 3. Coupled neighbouring lambda-points (entanglement FSS)

Run lambda_c +- delta with a SHARED parent RNG seed (common random numbers:
shared waiting-time, channel-selection, and resampling uniforms). The variance
of the DIFFERENCE O(lambda+delta) - O(lambda-delta) drops because the shared
randomness induces positive covariance. Directly tightens slopes d_lambda O,
crossings lambda_c, and nu.

Measured (full runs):

| observable                | L  | delta | variance reduction |
|---------------------------|----|-------|--------------------|
| entropy <S> difference    | 32 | 0.02  | 2.14x (rho~0.49)   |
| entropy <S> difference    | 32 | 0.04  | 2.03x              |
| entropy <S> difference    | 32 | 0.06  | 0.66x (BREAKS)     |
| <CMI> difference          | 32 | 0.04  | 1.76x              |
| KMR product <CMI><S> diff | 32 | 0.04  | 1.98x              |

Key facts:
- Robust ~2x at delta <= 0.04; the naive coupling BREAKS at delta=0.06 because
  the two trajectories desynchronise (take different jumps). Defines a coupling
  length in parameter space.
- Works on the CLEAN observables (<CMI>, KMR product <CMI><S>), not only the
  noisy trajectory product <CMI*S>. Use the clean ones for production FSS.
- L-scaling NOT verified beyond L=32 (more jumps at larger L -> faster desync ->
  the 2x may shrink). CHECK at L=64,96 before production.

NEXT IMPLEMENTATION (agreed, not yet built):
- SPLIT COUPLING of jump intensities: decompose lambda^+_j, lambda^-_j into a
  common part min(lambda^+,lambda^-) (shared jumps) plus two residual rates
  (+-only jumps). Each marginal stays exact; the shared instantaneous rate is
  maximised -> correlation preserved far longer than common random numbers ->
  extends the useful delta range.
- MAXIMALLY COUPLED RESAMPLING: with weights p^+_i, p^-_i, common mass
  M=sum_i min(p^+_i,p^-_i); pick the same parent w.p. M, else independent. Without
  this, jump-level coupling is destroyed at every resampling step.
- PAIRED ANALYSIS (as important as the coupling): keep the full coupled vector
  (O_r(lambda_1),...,O_r(lambda_m)) per realisation; fit slopes/crossings by GLS
  with the covariance matrix Sigma; bootstrap WHOLE coupled realisations, never
  per-lambda. Diagonal error models discard the gain.
- DELTA OPTIMISATION: choose delta by minimising MSE(d_lambda O) x wall, not the
  raw difference-variance reduction. With effective coupling Var(O+ - O-) ~
  delta^2, so the derivative variance stays bounded as delta -> 0. Richardson
  D_R = (4 D_{delta/2} - D_delta)/3 removes the O(delta^2) bias once small-delta
  variance is controlled. Test delta in {0.01,0.015,0.02,0.03,0.04}.

## 4. Compensated-count martingale control variate (activity ONLY)

M_k = n_k - zeta * Delta_Lambda_k is the compensated counting increment of the
guided point process: E[M_k | F_{t_k}] = 0 for ANY start state, so H_k * M_k is
a valid zero-mean control variate for any predictable H_k. Exact, no change to
the proposal, ~free.

For a self-normalised observable mu = E[GO]/E[G], the influence variable is
Y = G(O - mu); a control variate helps only through Corr(Y, M).

VALIDATED end-to-end (full runs, NR=8, L=48, zeta=0.3):

| target                              | end-to-end variance reduction |
|-------------------------------------|-------------------------------|
| tilted activity <n>_Q = E[Gn]/E[G]  | 3.26x   (rho^2(Y_n,M)=0.96)    |
| SCGF theta = (1/T) sum log E[G]     | 1.03x   (no help)             |
| entanglement entropy <S>            | 1.0x    (no help)             |

WHY: M is the within-window counting noise. It is collinear with the activity
influence variable Y_n=G(n-mu) (n appears in both) and nearly orthogonal to the
entropy influence variable G(S-mu). So it transforms the activity estimator and
does nothing for entanglement. The SCGF target is E[G] itself; across a
population of VARIED start states M (mean-zero per trajectory) decorrelates from
G (which tracks the between-state Delta_Lambda spread), so the SCGF gets no help.

CORRECTION (important): one-window / fixed-start tests OVERSTATED these. From a
fixed start state M gives rho^2(G,M)~0.78 -> an apparent 4.2x on E[G], and a
fixed-pool activity test gave ~400x. Both are ARTIFACTS of not varying the start
state. The end-to-end full-run numbers above (1.03x SCGF, 3.26x activity) are
the real estimator gains. ALWAYS validate variance-reduction claims end-to-end,
not one-window.

The H*M extension (weight M by window-start S, CMI, sum_q) does NOT rescue the
entanglement case: cross-fit R^2(G(S-mu) ~ {M, S0*M, CMI*M, sumq*M}) <= 0.05 in
all tested regimes. The counting noise is orthogonal to the entanglement
influence function. Strong, interpretable negative: the residual entanglement
variance is intrinsic state-to-state fluctuation, not residual weight variance.

USE: add M as a control variate ONLY to the tilted-activity estimator (test
channel-resolved M_{j,k}=n_{j,k} - zeta*integral lambda_j too). Estimate beta =
Cov(Y_n,M)/Var(M) by CROSS-FITTING (pilot or split-sample) to avoid coefficient
bias. Check the corrected denominator stays positive (0 negative-R_k windows
here). Do NOT add it to SCGF or entanglement estimators.

## 5.

**2026-08-09 AMENDMENT — the `learned Doob h_theta(X)` row is RETRACTED as stated. [V]**
The "no headroom" verdict tested the wrong object. Regressing the per-window weight (or
the additive control variate `Var(Lambda_s - [g(Gamma_0) - g(Gamma_s)])`) on window-start
features is structurally void: the Poisson decomposition
`Lambda_s - s*rbar = g(Gamma_0) - g(Gamma_s) + M_s` leaves the martingale intact with
`Var(M_s) = sigma_0^2 * s` exactly, so `R^2 ~ 0` / `G ~ 1` is obtained for ANY g,
INCLUDING THE EXACT ONE (re-measured: G = 0.97-0.98 for every feature set and every block
length). The Doob transform ABSORBS `M_s` into the change of measure; it does not subtract
a boundary term. The correct object is the Galerkin residual `R = s*[G g - delta_r]` with
G the Born backward generator. Under that test a ONE-PARAMETER control
`g-hat = a_K * sum_j q_j` reduces full-path `Var(log W)` by 9.4x (L=32) and 16.9x (L=64),
with the empirical optimum landing exactly on the predicted `a* = -log(zeta)*a_K`; and the
h-twisted (`h = e^{aK}`) tapered version BEATS production guided cloning at equal wall time
(B_L 1.42x [1.05,2.00], S 2.06x [1.50,2.79]). See HANDOFF 2026-08-09.
ALSO: the "ESS ceiling 0.98 caps proposal gain at ~2%" argument is a PER-WINDOW statement
and does not bound the horizon quantity `(1-zeta)^2 Var(Lambda_T)`.
The adaptive-resampling row of this section is UNAFFECTED and still stands.

 Directions tested and CLOSED (for thesis purposes)

| direction                          | verdict     | reason |
|------------------------------------|-------------|--------|
| online scalar c                    | moot        | ESS peaks exactly at c=zeta both sides |
| adaptive resampling                | negative    | n_distinct stays 1/N_c at all thresholds; S biases up 2.09->2.25; coalescence is SELECTIVE (weight condensation), not neutral, so fewer resamples cannot fix it and only inflate SNIS bias — see 2026-07-27 amendment below (verdict confirmed, mechanism corrected) |
| residual/stratified resampling     | marginal    | code already uses systematic; cannot fix selective coalescence |
| one-step no-click look-ahead       | no headroom | activity features R^2~0 vs the weight |
| learned Doob h_theta(X)            | no headroom | full feature set + quadratics R^2 <= 0.05 vs 0.79 for within-window clicks |
| auxiliary resampling score         | no headroom | same window-start features carry ~0 signal |
| continuous Fleming-Viot            | low prio    | resamples more asynchronously; adaptive-resampling result shows that worsens coalescence; untested but unlikely to help |
| multiple proposal c populations    | low value   | c=zeta is the unique optimum |
| annealed-zeta tempering            | negative    | guided already ESS~0.98 at small zeta; annealing gives FEWER distinct ancestors and biases S up |

**2026-07-27 AMENDMENT to the adaptive-resampling row [V].** Re-tested at HIGH zeta (0.5 and 0.9, L=64, T=64, N_c=32, 8 paired seeds, rho=0.5, carried cumulative log-weights; three-clock prototype `/tmp/adaptive_cloning.py`, bit-identity-gated against `run_cloning` to |dtheta|~1e-15). Verdict CONFIRMED, mechanism CORRECTED and now more general. Resampling events fall 3826->14 (zeta=0.9) and 2852->37 (zeta=0.5); wall time is unchanged (92.7 vs 93.4 s, because resampling was never the cost); genealogy does not improve (GESS 1.11 vs 1.52 at zeta=0.9, 1.00 vs 1.00 at zeta=0.5). The "S biases up" signature does NOT reproduce as bias: the chunk=4*dtau arm has the same resampling schedule and shows no shift, so the n=8 mean difference is noise. The real mechanism is that **the total degeneracy is conserved**. Over the horizon Var(log W) = (1-zeta)^2 Var(Lambda_T) = 36.8 (L=64) / 68.8 (L=96) at zeta=0.9, so per-window ESS sits at 1.000 BECAUSE frequent resampling suppresses it, not because selection is weak. The ESS threshold is exactly what sets per-event selection strength (at rho=0.5 a single event kills ~half the lineages), so 3826 weak coalescence events and 14 strong ones give the same product. Rescheduling moves degeneracy between the weight channel and the genealogy channel; it cannot reduce the total, which is fixed by (1-zeta)^2 Var(Lambda_T). The same quantity closes full-path importance sampling and multi-zeta bank reuse (HANDOFF 2026-07-27). Corollary: genealogy is destroyed by offspring-count VARIANCE, not by resampling frequency, so the one variant still worth trying is the OPPOSITE of adaptive — resample often with a maximally-coupled or residual scheme (row 3, still pending). Use these diagnostics instead of `n_distinct_ancestors`: GESS = N_c^2 / sum_a f_a^2 (family sizes f_a), and the coalescence budget sum_k 1/ESS_k (collapse at O(1); the every-window baseline runs at 119.6).

Deep reason the state-dependent / learned-Doob family fails: the residual
per-window weight variance at c=zeta is within-window Poisson click noise, NOT
predictable from any window-start Gaussian feature (entropy, activity, CMI,
covariance norms, quadratics all R^2~0). The Doob signal exists in principle
(h(J_j X)/h(X)) but Gaussian-accessible surrogates cannot capture it, and the
ESS ceiling (0.98) caps the maximum possible proposal gain at ~2% anyway.

## 6. Implementation plan (engineering, priority order)

ENTANGLEMENT PIPELINE (main phase-diagram work):
1. Split-coupled neighbouring lambda-points + maximally coupled resampling.
2. Paired covariance (GLS) + paired-realisation bootstrap in all slope/crossing
   fits.
3. delta optimisation by derivative-MSE x wall; Richardson extrapolation.
4. Adaptive lambda-placement + N_c allocation: coarse survey, then add coupled
   points where Var(lambda_x) ~ Var(D)/[D']^2 is largest; biggest N_c at
   L=96,128; at very low zeta invest in larger L (physical finite size), not N_c.
5. Use CLEAN observables (CMI, KMR product <CMI><S>), not the trajectory product.

ACTIVITY PIPELINE (theory-side: K_eff, <n>_Q, channel activities):
6. Add compensated-count martingale CV with cross-fitted coefficients (~3x on
   <n>_Q, free). Do NOT use it for SCGF/entanglement.

KEEP: guided proposal c=zeta (the solved per-run sampler).

## 7. Caveats / open

- Coupling L-scaling unverified beyond L=32 (check L=64,96; 2x may shrink as
  trajectories desync faster with more jumps).
- All variance-reduction coefficients must be cross-fit (independent pilot or
  split-sample) to avoid optimism.
- The martingale-CV denominator mean(G - gamma*M) must stay positive before any
  log (SCGF); held here, but check at lower N_c / longer windows.
- These were prototypes (scratch scripts; the two validated WINNERS are saved in
  `analysis/var_reduction/`). Production needs porting into pps_qj/parallel/ with
  the paired-analysis machinery.

---

## Provenance

Five-round stress-test, 2026-06-16/17. Every result is from full guided-cloning
runs on the real Gaussian backend (L=32-64, zeta=0.1-0.6, N_c=80-200, NR=3-12),
not synthetic. Negative results were validated as carefully as positive ones.
The two downward corrections (SCGF 4.2x->1.03x, activity 400x->3.26x) came from
replacing fixed-start tests with end-to-end full-run tests.

Saved prototypes (analysis/var_reduction/):
- `coupling_lambda.py`     — coupled lambda-points, delta-scan + L-scan (S, B_L)
- `coupling_cmi_kmr.py`    — coupling on clean <CMI> and KMR product <CMI><S>
- `activity_cv.py`         — end-to-end tilted-activity martingale CV (the ~3x)
- `scgf_cv.py`             — end-to-end SCGF CV (the negative; shows 1.03x)
