---
lifecycle: historical
frozen: 2026-08-10
authoritative_for: nothing
superseded_by: research/state/**
---

> # FROZEN 2026-08-10 - HISTORICAL RECORD, NOT AUTHORITATIVE
>
> This is a verbatim copy of `theory/HANDOFF.md` as it stood on 2026-08-10,
> before the reconstruction audit (`audit/2026-08-10/`).
>
> It is 1174 lines of reverse-chronological session log. Corrections were
> implemented by prepending blocks rather than editing superseded text, so its
> bottom half contradicts its top half by construction and its header understates
> its own currency. Reading it partially, by grep or by truncated load, yields
> whichever layer was hit.
>
> **Do not cite this file as evidence for any scientific claim.** Its scientific
> content was reconstructed into `research/state/claims/` and
> `research/state/disputes/` by the audit. Cite claim IDs.
>
> It is retained because the correction chronology is itself scientifically
> useful: it records which arguments failed and why.
>
> Provenance: copied from `theory/HANDOFF.md`, mtime 2026-08-10 10:28.

# ppsQJ_m2 Project — Handoff Notes

### PRODUCTION VARIANCE-REDUCTION AUDIT + closure of the selection-side sampler programme (2026-08-10)

**MASTER METRIC.** The quantity to minimise is `t_wall * sigma_lambda_c^2`, with `sigma_lambda_c ≈ sigma_F / |F'(lambda_c)|`, `F(lambda) = B_{L2}(lambda) − B_{L1}(lambda)`. **NOT** GESS, **NOT** Var(log W), **NOT** even Var(B_L). Three separate cheap proxies each predicted large production wins that direct measurement then refused to deliver (Var(log W)↓ ⇏ N_c reducible; GESS↑ ⇏ Var(O)↓; ΣD₂↓ ⇏ N_c reducible). **Do not accept a proxy-based claim without a direct measurement of the master metric.**

**[V] WHY THE SAMPLER PROGRAMME IS CLOSED — selection is not the bottleneck.** Two decisive tests (`analysis/var_reduction/bottleneck_test.py`, L=32, ζ=0.9):
 (i) *Within-population vs independent spread of B_L*: Var_pop/Var_indep = **1.19 (guided), 1.49 (ctrl)**. The resampled population is already at least as dispersed as fully independent trajectories — nothing for a better resampling scheme, allocation rule, or observable-aware cloning to recover.
 (ii) *Sibling decorrelation*: iid siblings launched from one common post-resampling state have ρ(B_L) = +0.039/−0.035/+0.006/−0.024/+0.112 at Δt = 1/2/4/8/16 (all ±0.082); ρ(CMI) likewise ≈0. **Duplicated clones forget shared ancestry in B_L within ONE time unit**, against a horizon of T=32–64.
 Consequence: `N_eff ≈ N_c` always, so `Var(B_hat_L) ≈ Var_intrinsic(B_L)/N_c`. Gate: L=32 production Var_pop=0.734, N_c=32 → predicted sd 0.151 vs measured 0.1605 across realisations. **This single relation explains every negative in the sampler investigation** — genealogical collapse never costs B_L information, which is why GESS 1.99→6.23 bought nothing.
 CLOSED, do not revisit: stronger K-based Doob controls; extra activity features; resampling-scheme optimisation (systematic/stratified/sorted-stratified); antithetic or stratified sibling RNG; genealogy optimisation; lowering N_c because GESS improved; full-path importance sampling; observable-aware auxiliary cloning; multilevel-N_c (it corrects a finite-N_c bias, and the L=64 ladder shows guided means FLAT in N_c, χ²/dof≈0.6 — no bias to correct).
 Supporting null: lagged cross-correlation of the residual selection increment X with the FUTURE observable (`xo_lagged.py`) gives |C_{X,B_L}| ≤ 0.12 at all lags for both arms at L=32 and 64, with **no growth at larger L** — the "control removes activity fluctuations but leaves the entanglement-critical mode" hypothesis is NOT supported. C_{K,B_L} ≤ 0.14, so K is weakly coupled to B_L, not orthogonal to it.

**[V] CHUNK LEVER — CERTIFIED, BANK mult=4.** `analysis/var_reduction/chunk_bias_cert.py`, L=32, ζ=0.9, R=40, guided estimator (the certification HANDOFF has owed since 2026-07-27; the old test used the non-guided estimator where the per-window weight was ζ^n rather than the smooth exp[−(1−ζ)ΔΛ]):
| mult | ΔS | ΔCMI | ΔB_L | speedup | efficiency |
|---|---|---|---|---|---|
| 2 | −2.04σ | −1.10σ | −1.59σ | 1.78× | — |
| 4 | −2.20σ | **+0.10σ** | −0.63σ | 1.94× | 1.82× |
| 8 | −1.82σ | **+0.46σ** | −0.28σ | 1.89× | 1.83× |
The shift is **FLAT in mult, not growing** — a genuine O(δτ) discretisation bias must scale with chunk length. CMI and B_L consistent with zero at mult=4 and 8. **Use mult=4** (speedup saturates: 1.94× then 1.89×). **Budget 1.6× in production, not 1.9×** — the test driver calls the trajectory function per clone per window and carries more per-window Python overhead than `run_cloning`. Residual flat ~2σ on S is most economically mult=1 having fluctuated high; a paired common-seed re-test on S is a side validation, NOT a blocker.

**[P] SNAPSHOT AVERAGING — the largest confirmed lever, but the production gain is NOT yet measured.** `analysis/var_reduction/snapshot_gain.py` (estimator autocorrelation across independent realisations) and `coupled_snapshot_pilot.py` (paired difference D(t)=B_L^+ − B_L^−).
 *Correction to the pre-existing "g_snap ≈ K, 3–4× for K=4" claim: WRONG at dense spacing.* Measured densely (Δt=2, T=L): g_snap(K=4) = 1.66/1.95/1.80/1.47 at L=32/40/48/64 — snapshots 2 apart are still ~55% correlated.
 τ_int^{B_L} (time units) = 2.96 / 2.66 / 2.82 / 3.92 at L=32/40/48/64 from the dense scan; the L=64 value UNDERSTATES the tail (at spacing 8, ρ = 0.18 and 0.235 at lags 8 and 16).
 Paired-difference gain at fixed T=64, spacing 8: **L=32 → 5.85× (K=6), L=40 → 7.73× (K=5), L=64 → 2.66× (K=5)**; L=48 pending. NOTE g_snap > K is *possible* (negative weighted autocorrelation sum), so 7.73× is not structurally impossible — but at R=16 a ratio of two variance estimates carries ~±35% each and it should be read as 5–8×, not quoted. **DO NOT fit τ ∝ L or a sharp 48/64 threshold to these four points; statistics do not support either.** Bootstrap the ratio in future runs.
 **RULE: Δt_snap ≈ 2 τ_int(L), measured per L. A fixed spacing of 8 is NOT universally valid** — it is ≈2.7τ at L=32 (giving ~0.98K) but only ≈1τ at L=64 (giving 0.53K). Store the measured τ_int with the data.
 **THE OPEN GAP:** every attractive number above used **T=64**, but production uses **T≈L**. Extending T is not free (T=2L costs 2× and yields 2× the snapshots — gain per unit cost is identical). So the production-relevant quantity `g_prod(L) = Var[O(T=L)] / Var[snapshot-averaged O at T=L]` **has not been measured at any L.** Until it is, do not budget a specific snapshot gain.

**[P, PROBABLY DISPOSABLE] COUPLED λ gives only 1.09×, not the recorded ~2×.** L=32, T=64, δ=0.04, `coupled_snapshot_pilot.py`: Var(D) single = 0.05078 (common RNG) vs 0.05533 (independent) → **1.09×**; K-averaged 0.00868 vs 0.01049 → 1.21×. Snapshot gain is 5.85× coupled and 5.27× independent, i.e. **the snapshot gain does not depend on coupling and the two factorise exactly** (1.09 × 5.85 = 6.37 = total vs independent+single). *Mechanism:* in a cloning algorithm common seeds do NOT survive resampling — different λ gives different weights, hence different systematic-resampling index maps, and the populations decouple within a few of the ~1900 events. CRN coupling is far weaker here than for plain trajectory averaging. **Before closing, retest δ = 0.01/0.02/0.04 at L=32 scored on the SLOPE variance `Var(D_δ)/(4δ²)`, not on the coupling ratio** — smaller δ preserves coupling but shrinks D, and the 1/(2δ) amplification is the real trade-off. Also check F'(δ) is δ-independent (else curvature already matters). If Var(D)/(4δ²) does not improve materially, CLOSE coupled-λ; 1.09× is not worth production complexity.

**[V] GLS snapshot weights REJECTED — use equal weights.** `a = Σ^{-1}1/(1'Σ^{-1}1)` reported 12.1× (L=32) and 6.1× (L=64) vs equal weights, but with R=20 and K=6 the in-sample variance is optimistically biased by ≥ (R−1)/(R−K−1) = 1.46×, weights are wildly non-uniform (0.043–0.365), and the **L=64 fit returned a NEGATIVE weight (−0.115)** — a clear overfitting signature. Would need R ≳ 100 to establish. Equal weights in production.

**Current best production budget (provisional, T=64 not T=L):** σ_λc ≈ 0.0085 per 64.6 s realisation at L=32 (F' ≈ 10.95); reaching ε_λ = 0.001 needs ~72 realisations ≈ 78 min at that cell.

**NEXT QUEUE (in order, do not reorder).** (1) Read L=48 from `/tmp/coupsnap_scan.log`. (2) **Production-matched T=L snapshot certification at L=32, 48, 64** with mult=4, production burn-in, equal weights, production N_c, and Δt from a cheap per-L τ_int pilot — this closes the open gap above. (3) δ-scan for coupling scored on Var(D)/(4δ²). (4) Build the full three-point crossing estimator λ_c = λ_0 − a/b from a local linear fit, **bootstrapping the whole procedure** for σ_λc rather than propagating errors analytically; keep a quadratic term as a curvature diagnostic and choose the largest δ for which |c|δ² ≲ 0.5σ_F. (5) Benchmark old vs new production directly as `G_prod = t_old σ²_old / t_new σ²_new` at one representative ζ — **no multiplication of separately measured gains**. (6) Freeze. (7) Only then adaptive λ allocation: coarse locate → three local points → R₀≈8–12 → bootstrap σ_λc → add realisations until `R_needed ∝ σ²/ε_λ²`, so easy ζ stop automatically.

**DO NOT use the controlled/Doob sampler in production.** It is a methods result only (see the 2026-08-09 block): Galerkin-predicted a\*, 9.4×/16.9× path-weight variance reduction, M=1 domination, validated twist/taper — and no estimator benefit at L=64.

### Approximate-Doob state-dependent control + h-twisted cloning: BEATS production guided cloning at equal wall time (2026-08-09) — RETRACTS the "learned Doob has no headroom" row of `VARIANCE_REDUCTION.md` §5; QUALIFIES the 2026-07-27 "degeneracy is conserved" claim; CLOSES trajectory MCMC; retires the thinning bug from the critical path

Five external memos adjudicated. All on Mac (`.venv`, BLAS pinned, lowrank+newton, λ=0.5√ζ). Scratch: `/tmp/csampler2.py` (canonical sampler), `/tmp/doob_galerkin.py`, `/tmp/final3.py`, `/tmp/memo5.py`, `/tmp/mcmc_pilot.py`, `/tmp/doob_common.py`. **No repo files changed.**

**THE ALGORITHM.** Controlled jump rates `r̂_j(t,Γ) = ζ r_j(Γ) exp[a(t) Δ_j K]`, `Δ_j K = K(J_jΓ) − K(Γ)`, `K = Σ_j q_j`, with the taper `a(t) = a* [1 − exp(−(T−t)/τ_K)]`, `a* = −log(ζ)·a_K`, `τ_K = −a_K/(2α)`. Weight written so the a=0 branch is EXACT: `∫r̂ dt = ζΛ + ζJ`, `J = ∫ r(⟨t⟩_q − 1) dt`, so `log W_res = −Σ_m a(t_m)Δ_{j_m}K − (1−ζ)Λ_T + ζJ` and J≡0 at a=0 (gated to 5e-14). Cloning resamples on the **h-TWISTED** increment `Δℓ_h = Δℓ_raw + a(u)K_u − a(t)K_t` (h = e^{aK}); this is algebraically exact and needs NO new quadrature, since `Δℓ_raw = a(t)K_t − a(u)K_u + ∫V_h`, `V_h = ȧK + a b_K − r + r̂`. Telescopes to `log W_res + a(T)K_T − a(0)K_0`, and **a(T)=0 ⇒ no terminal untwisting** — observables are read off exactly as in production.

**[V] Δ_j K ≥ 0 IDENTICALLY** (min 0.0000, frac<0 = 0.0000 over ~7500 state·channel samples at L=64). Hence tilt ≤ 1 and **M = 1**: the controlled process is DOMINATED by the plain guided process, so one thins off the existing exact Newton/branch-norm sampler — propose at `proposal_c=ζ`, pick channel ∝ q_j as now, accept the jump w.p. exp(a Δ_jK). ~30 lines. **This retires the open thinning bug (2026-07-27, N_T low by 1.5%/4.8%) from the critical path — general uniformization is not needed.** Mean tilt 0.629, so 1.59 candidates per accepted jump but the effective click rate falls 0.9→0.566, i.e. FEWER expensive jump updates than baseline.

**[V, SUPERSEDED 2026-08-10 — the L=32 result below is real but did NOT reproduce at L=64 (E_B_L ≈ 0.99), and the mechanism is now understood: N_eff ≈ N_c because siblings decorrelate in B_L within one time unit, so selection improvements cannot reduce Var(B_L). Read as a METHODS result, not a production recommendation.] Tapered twisted controlled cloning beats production guided cloning at equal wall time.** L=32, ζ=0.9, λ=0.474, T=64, N_c=32, production resampling interval δτ=0.034, NREP=30, common seeds, paired bootstrap (4000 resamples), `E_O = 1/(t_wall·Var)`:
| arm | ⟨S⟩ | ⟨CMI⟩ | ⟨B_L⟩ | GESS | wall |
|---|---|---|---|---|---|
| A production guided | 2.0040±0.0181 | 0.5262±0.0117 | 1.0577±0.0293 | 1.99 | 18.1 s |
| C taper + Simpson | 1.9841±0.0125 | 0.5409±0.0090 | 1.0744±0.0213 | 6.13 | 33.4 s |
| **D taper + trapezoid** | 1.9871±0.0117 | 0.5362±0.0096 | 1.0671±0.0228 | **6.23** | **20.9 s** |
E_O vs production: D → **S 2.06× [1.50,2.79] P(>1)=0.98; CMI 1.28× [0.93,1.78] P=0.78; B_L 1.42× [1.05,2.00] P=0.88**. C → 1.13×/0.92×/1.02×. All D estimates agree with production (S 0.78σ, CMI 0.66σ, B_L 0.25σ). **Simpson is a DIAGNOSTIC ONLY** — it is required for the mean-one gate but consumes the entire estimator margin (33.4 vs 20.9 s) while changing no physics (D vs C agree within 1σ on all three, GESS 6.23 vs 6.13). Ship trapezoid.

**[V] The taper is what makes it work, and GESS does NOT collapse in the terminal layer.** Constant twist (arm B) gives GESS 6.04 but pays terminal untwisting (ESS_untwist/N_c = 0.637), landing at E_S=1.05×, E_CMI=0.69×. The taper removes that step entirely (h_T=1) and recovers CMI 0.69→1.03 (arm C), ≈ the predicted 1/0.637=1.57. GESS(t) for the tapered arm stays at 6.23 at t=T against production's 1.99. NOTE arms B and C show identical GESS(t) for t≲24 — this is NOT a bug: a(t) differs from a* by <1e-5 there, and with common seeds the accept/reject decisions are bit-identical.

**[V] The RAW controlled weight inside cloning is a DISASTER (11× worse) — the twist is mandatory.** Feeding `Δℓ_raw` to the resampler gives GESS 1.13 (vs production 1.99) and E_S=0.09× at δτ. Mechanism: at c=ζ the per-window weight is smooth exp[−(1−ζ)ΔΛ] because the discrete click factor cancels; the raw control REINTRODUCES a per-click discrete term −aΣΔ_jK. The control works by making weight increments ANTI-CORRELATED in time (horizon Var falls 17× while per-window variance rises), and resampling every 0.034 destroys exactly that cancellation. **This QUALIFIES the 2026-07-27 conservation result: "degeneracy is conserved" was derived at FIXED proposal and does not hold once a control is present.** Raw-weight chunk lengthening treats the symptom (recovery begins at ~128δτ ≈ 4.3 ≈ τ_K); the twist fixes it at the production interval. Do not re-explore raw controlled cloning.

**[P→V] Path-level variance reduction, and the coefficient is PREDICTED not fitted.** Full-path Var(log W) at ζ=0.9, T=L, direct a-scan over 8 values: L=32 → 17.47 → **1.87 (9.4×)**, ESS/M 0.049→0.293; L=64 → 38.56 → **2.28 (16.9×)**, ESS/M 0.031→**0.098** [CORRECTED 2026-08-09: an earlier value of 0.309 here came from M=48 and was an artifact — ESS/M is upward-biased at small M for heavy-tailed weights because the tail is unsampled. Re-measured at M=400 it is **0.098**, consistent with exp(−Var(log W)) = exp(−2.28) = 0.102. **Var(log W)=2.28 itself is robust** (a direct variance), but the ESS claim was not. Consequence: **path control useful ⇏ path-IS useful.** Controlled full-path IS is still NOT a competitive estimator at production sizes — an L=64 reference run at M=400 gave ESS=39, S=2.1418±0.0439, CMI=0.4319±0.0458, B_L=0.9250±0.1064, i.e. error bars far too wide to arbitrate 0.02–0.04 differences in CMI, and biased low by its own SNIS finite-sample error (same direction as the ESS/M=0.018 arm at L=32). Treat path-IS as a qualitative correctness check, NOT a reference estimator; do not spend hours pushing M into the thousands.]. **The empirical minimum sits exactly at the Galerkin-predicted a\* = s·a_K in both cases.** T-scaling at L=32 (T=16/32/64/128): baseline Var exactly linear (4.22/7.32/13.67/26.79), controlled 0.44/0.98/1.93/2.39, gain 9.6/7.5/7.1/11.2×; wall speedup 0.87/1.14/1.18/1.37× (grows with T). Gain GROWS with L. **a_K is INTENSIVE**: transferring a_ref=−3.55 fitted at L=64 with no refit gives Galerkin gain 20.1/25.0/25.4 at L=32/64/96. It does NOT transfer across ζ (4.4 at ζ=0.5). **Calibration recipe: fit ONE scalar per ζ at L=32 (minutes), reuse at every L.**

**[V, METHOD CORRECTION — load-bearing] `VARIANCE_REDUCTION.md` §5's "learned Doob h_θ(X): no headroom" is RETRACTED as stated, and memo 3's proposed screening is also void.** Both regress a short-horizon object; the Poisson decomposition `Λ_s − s·r̄ = g(Γ_0) − g(Γ_s) + M_s` leaves the martingale intact and Var(M_s)=σ_0²s exactly, so R²≈0 / G≈1 is obtained for ANY g INCLUDING THE EXACT ONE (measured G = 0.97–0.98 for every feature set and block length). Memo 3's "R² of future integrated activity on Γ_t" is void for the same reason: Var(Y_τ)→σ_0²τ grows while Var(E[Y_τ|Γ]) saturates, so R²→0 by construction (measured 0.86/0.62/0.35/0.19/0.05/−0.04 at τ=2/4/8/16/32/64). The "ESS ceiling 0.98 caps proposal gain at ~2%" argument is PER-WINDOW and does not bound the horizon quantity (1−ζ)²Var(Λ_T). **Do not re-derive either negative.** The correct object is the Galerkin residual `R = s[𝒢g − δr]` with 𝒢 the BORN backward generator `𝒢φ = (dφ/dt)_noclick + Σ_j r_j[φ(J_jΓ) − φ(Γ)]`; exact g ⇒ R≡0 pointwise ⇒ log W_res = O(1).

**[V] Correctness gates, all passed.** (i) a=0 reproduces production exactly: |log W + (1−ζ)Λ| < 5e-14, and ⟨N_T⟩, ⟨Λ_T⟩ within 0.3σ of `gaussian_born_rule_trajectory`. (ii) Vectorised φ(J_jΓ) for all L−1 channels (bonds (2j,2j+3) are Majorana-index-disjoint ⇒ one O(L²) rank-2 outer-product pass) agrees with `apply_projective_jump` to 7.1e-15. (iii) Monte-Carlo 𝒢φ from a fixed state under the Born process = [0.986,2.186,1.096]±[0.13,0.17,0.12] vs analytic [0.954,2.172,1.098], all three features; dt_nc converged to 0.06% at 2e-3. (iv) **Mean-one martingale** `E_{Q_a}[R_T]=1` with `R_T = exp[log W_res + (1−ζ)Λ_T]` (= dQ_0/dQ_a; no tilted partition function): 0.897±0.021 (trapezoid on full I) → 0.952±0.028 (J-formulation) → **0.9926±0.0295 (J + Simpson), −0.25σ**. Both fixes contributed; the a* row (−4.2σ) is a tail-dominated estimator at σ(logR)=2.0, not a correctness signal. (v) **a-invariance**: ⟨S⟩,⟨CMI⟩ at L=16, M=3000, a/a*∈{0,0.5,1,1.5} agree within ≤1.7σ. (vi) Green–Kubo σ_0² = 2τ_r Var(r) = 42.9 vs direct Var(Λ_s)/s = 43.1–46.0 at L=64 (τ_r=2.96) ⇒ **Var(Λ_T) is PURELY DIFFUSIVE**, no quasi-static offset. NOTE ⟨𝒢φ⟩_ss ≠ 0 is EXPECTED, not a bug: 𝒢 is the Born generator while the ensemble is stationary for the guided process, so ⟨𝒢_Born φ⟩ = (1−ζ)⟨Σ_j r_j Δ_jφ⟩.

**[V] An earlier 9% ⟨CMI⟩ shift between controlled and guided arms was SNIS FINITE-SAMPLE BIAS, not a bug.** Decisive evidence: the a=0 arm sits ABOVE the controlled arms at L=32 and BELOW them at L=16, so the direction flips and it cannot be a controlled-specific systematic; the a=0 arms had ESS/M = 0.018–0.049. Do not re-open.

**[V] Trajectory MCMC (memo 1) is CLOSED for L ≥ 128.** Guided suffix MCMC is mathematically correct — verified independently that P_Born/Q_c = c^{−N}exp[−(1−c)Λ], so at c=ζ the click factor cancels and A = min[1, exp(−(1−ζ)ΔΛ)], and that the backend `Lambda` field is the integrated PHYSICAL hazard. Measured acceptance, L=64/ζ=0.9/T=64 by suffix fraction f: 0.72/0.60/0.20/0.08/0.08/0.04/0.00 at f=0.062/0.125/0.25/0.375/0.50/0.75/1.00, sd(logR)=0.98/1.89/3.32/3.35/4.22/4.39/5.47. Same (1−ζ)²Var(Λ_T) obstruction: at fixed physical refresh length ℓ, sd(logR)=s√(2σ_0²ℓ) ~ L^0.43 (L=96 confirms 5.49 vs 4.13 at f=0.875). The observable needs ℓ≳16 (τ_int~5–10 at L=64) ⇒ f=0.25 ⇒ 20% acceptance at L=64, ~0.8% at L=128 [P], while required ℓ grows as L^z. The ρ-correlated-proposal variant does not rescue it. NOTE the f=1.0 acceptance of 0.15 at L=96 is an EQUILIBRATION artifact (chain starts on a high-Λ path, first proposals are downhill). **Retain as the only Plan B** if the control ever fails at larger L, since a 10× variance cut drops sd(logR) ~3× and puts L=128 acceptance at ℓ=16 back near 0.4.

**[P, and now the MORE IMPORTANT story] N_c ladder: the twist appears to reduce FINITE-POPULATION BIAS, not just variance.** L=32, ζ=0.9, T=64, equal 300 s wall budget per config (900 s for the N_c=128 anchor), `/tmp/ncladder.py`:
| config | n | wall/rep | GESS | ⟨S⟩ | ⟨CMI⟩ | ⟨B_L⟩ |
|---|---|---|---|---|---|---|
| guided N_c=128 | 12 | 76.7 s | 3.37 | 2.0204±0.0111 | 0.5488±0.0088 | 1.1095±0.0222 |
| guided N_c=64 | 9 | 35.7 s | 2.89 | 1.9916±0.0165 | 0.5377±0.0154 | 1.0720±0.0364 |
| guided N_c=32 | 17 | 17.8 s | 1.77 | 1.9803±0.0248 | 0.5312±0.0168 | 1.0567±0.0438 |
| ctrl N_c=32 | 15 | 20.7 s | 5.81 | 2.0074±0.0191 | 0.5680±0.0173 | 1.1414±0.0389 |
| ctrl N_c=24 | 20 | 15.4 s | 4.70 | 2.0020±0.0161 | 0.5535±0.0133 | 1.1102±0.0316 |
| ctrl N_c=16 | 30 | 10.3 s | 3.38 | 2.0020±0.0231 | 0.5554±0.0192 | 1.1206±0.0488 |
| ctrl N_c=12 | 38 | 8.1 s | 2.69 | 1.9437±0.0227 | 0.5489±0.0203 | 1.0778±0.0513 |
**The controlled arm is FLAT across N_c=16/24/32** (χ²/dof vs a constant = 0.03/0.23/0.19 for S/CMI/B_L; plateau S=2.0037±0.0109, CMI=0.5581±0.0092, B_L=1.1222±0.0219) **and BREAKS at N_c=12** (S=1.9437, 2.5σ below the plateau) — so the controlled arm has its own finite-population floor, just at a much smaller N_c. **The guided sequence converges monotonically toward that plateau**: CMI 0.5312→0.5377→0.5488 and B_L 1.0567→1.0720→1.1095 at N_c=32→64→128, i.e. −1.40σ/−1.13σ/−0.73σ (CMI) and −1.34σ/−1.18σ/−0.41σ (B_L) from the controlled plateau. Joint weighted fit `O(N_c)=O_∞+b/N_c` with **shared O_∞** (20000-draw bootstrap, 68% bands, ctrl N_c=12 excluded): b_guided = −1.75[−2.80,−0.67] (S), −0.89[−1.65,−0.14] (CMI), −2.55[−4.54,−0.61] (B_L), all excluding zero; b_ctrl = −0.57[−0.98,−0.15], +0.02[−0.31,+0.35], −0.15[−0.99,+0.69], consistent with zero for CMI and B_L. P(|b_c|<|b_g|) = 0.94/0.85/0.89. **Model-free headline: ctrl N_c=32 (20.7 s/rep) reaches a value that guided has NOT reached at N_c=128 (76.7 s/rep, 3.7× the cost).** If this holds, the right metric is no longer `E_O = 1/(t_wall·Var)` but budget-dependent MSE, `MSE_B(N_c) = [O(N_c)−O_∞]² + Var_real·t(N_c)/B`, since `lim_{B→∞} MSE_B = Bias²` and no amount of extra guided realisations removes a systematic.
**CAVEATS — this is [P], not [V].** (i) Bands are 68%; at 95% b_guided would include zero for CMI. (ii) The 1/N_c model is imposed on only three guided points. (iii) S does NOT fit cleanly — guided N_c=128 overshoots the controlled plateau by +1.07σ, so only CMI and B_L show the clean monotone approach. (iv) An earlier reading of a GESS↔CMI monotone ordering as evidence was WRONG and should not be repeated: the controlled arm moves GESS 3.38→5.81 with no resolved observable shift (0.5σ), so GESS is a diagnostic and the real evidence is the shared-O_∞ convergence test. (v) Equal-wall variance efficiency vs guided N_c=32 is ctrl-24 S 2.33×/CMI 1.56×/B_L **1.89×**, ctrl-32 1.65×/0.91×/1.24×, ctrl-16 1.12×/0.74×/0.79×, guided-64 2.13×/1.11×/1.36× — the **sweet spot is N_c≈24–32 and cutting N_c further does NOT help**, contrary to the memo's expectation; the non-monotonicity across ctrl N_c is largely noise at n=15–38 (sd estimates carry 25–35% error).

**[V, L=64 — THE L=32 ADVANTAGE DOES NOT REPRODUCE. Complete 5-config ladder, ζ=0.9, T=64, equal wall budgets.]**
| config | n | GESS | ⟨S⟩ | ⟨CMI⟩ | ⟨B_L⟩ |
|---|---|---|---|---|---|
| guided N_c=32 | 16 | 1.25 | 2.3379±0.0245 | 0.5370±0.0167 | 1.2586±0.0474 |
| guided N_c=64 | 7 | 1.79 | 2.2825±0.0298 | 0.5177±0.0175 | 1.1838±0.0516 |
| guided N_c=128 | 7 | 2.07 | 2.2815±0.0266 | 0.5371±0.0106 | 1.2263±0.0331 |
| ctrl N_c=24 | 13 | 3.69 | 2.3074±0.0331 | 0.5252±0.0225 | 1.2148±0.0601 |
| ctrl N_c=32 | 14 | 4.76 | 2.2519±0.0264 | 0.4973±0.0160 | 1.1212±0.0422 |
**The guided sequence is NON-MONOTONE in N_c**: CMI 0.5370→0.5177→0.5371 and B_L 1.2586→1.1838→1.2263 at N_c=32→64→128. It is consistent with a CONSTANT ≈0.531 (χ²/dof≈0.6), i.e. no resolved guided finite-N_c drift at L=64 at all. **The controlled arm is also NOT flat here** (ctrl-24 is +1.01σ/+1.28σ/+1.31σ above ctrl-32 on CMI/B_L/S), unlike the L=32 plateau. ctrl-32 sits ~2.1σ BELOW the flat guided level, so at L=64 it is the CONTROLLED arm that differs from the others. **Equal-wall efficiency E_B_L ≈ 0.99 at L=64 vs 1.42× at L=32.** So neither the variance advantage nor an accuracy advantage reproduces at L=64. NOTE those L=64 wall times were measured on a CONTENDED machine (a concurrent path-IS run); means/variances are fine but any efficiency ratio from them needs re-measuring idle. **Two intermediate readings during this session were WRONG and must not be re-derived**: (i) "the L=64 sign flip contradicts everything" — no, guided 32→64 moved toward ctrl; (ii) "guided converges monotonically to ctrl at L=64" — no, N_c=128 overturned it. Both came from extrapolating through TWO N_c points at n≈7–17 where 1σ gaps are not signal. **RULE: require THREE N_c points before any convergence claim.**
**[V] Controlled full-path IS is NOT a usable reference estimator.** L=64, M=400, Simpson: ESS=39 (0.098), S=2.1418±0.0439, CMI=0.4319±0.0458, B_L=0.9250±0.1064 — error bars far too wide to arbitrate 0.02–0.04 in CMI, and biased low by its own SNIS finite-sample error. Do not spend hours pushing M; reaching SE~0.015 needs M≈4000 and the bias remains. **Gate A at L=64: max|log W+(1−ζ)Λ| = 0.000e+00, EXACT.** Gate B ⟨R_t⟩ = 1.0240±0.1001 / 0.9469±0.1209 / 0.8165±0.1821 / 0.5980±0.3213 at t=8/16/32/64 with sd(logR)=1.17/1.60/2.18/3.19. The apparent drift is ESTIMATOR CONDITIONING, not compensator error: M/e^{σ²} = 63/19/2.2/0.01, and the deficit tracks it exactly. t=8 passes cleanly; t≥16 is untestable at M=250. **The compensator is not indicted.**

**[V] D_2 selection-strength scaling (`analysis/var_reduction/d2_scaling.py`, M=40 independent trajectories per arm, no population).** Per-window incremental weight G=e^X, X_g=−(1−ζ)ΔΛ, X_c=Δℓ_raw+a(u)K_u−a(t)K_t; D_2=log(⟨G²⟩/⟨G⟩²), horizon sums over the stationary half:
| L | windows | ΣD₂ guided | ΣD₂ ctrl | Var(totX) g / c | τ_X g / c (windows) |
|---|---|---|---|---|---|
| 16 | 228 | 0.020 | 0.004 | 1.53 / 0.13 | 12.6 / 18.7 |
| 24 | 524 | 0.034 | 0.007 | 4.95 / 0.20 | 25.8 / 39.5 |
| 32 | 941 | 0.034 | 0.010 | 5.55 / 0.38 | 40.0 / 64.5 |
| 48 | 2140 | 0.059 | 0.017 | 23.87 / 0.95 | 75.3 / 103.4 |
| 64 | 3825 | 0.073 | 0.024 | 32.14 / 1.44 | 97.3 / 121.5 |
**CONVERSION CORRECTION [V] — this is the load-bearing part.** The naive multinomial mapping (coalescence ∝ D₂/N_c) is WRONG for the systematic resampling this code uses, and applying it produced a spurious "independent-trajectory D₂ underestimates by 1000×" conclusion. RETRACTED. With p_i=(1+ε_i)/N_c: D₂≈⟨ε²⟩ so √D₂=rms(ε), and systematic resampling duplicates at FIRST order in ε, giving expected excess offspring per event A_k = (N_c/2)Σ|p_i−1/N_c| ≈ **0.4·N_c·√D₂** (an external memo gave this as √D₂ and dropped the N_c factor — the N_c factor is exactly what closes the gap). Horizon lineage losses H=ΣA_k at N_c=32: guided 27/54/72/143/214 and ctrl 12/24/40/77/123 at L=16/24/32/48/64, ratio **0.44/0.44/0.55/0.53/0.58**. **This EXPLAINS the GESS data quantitatively**: H_g/N_c = 2.3 turnovers at L=32 → GESS 1.99, 6.7 at L=64 → GESS 1.25; controlled 1.25 and 3.85 turnovers → GESS 6.23 and 4.76. Monotone and consistent in all four measured cells. **Realistic genealogical advantage is therefore ~1.7–2.2×, NOT the 3–4× that ΣD₂ alone implies.**

**[O] THE OPEN QUESTION, now narrow: why does ~2× healthier genealogy not reduce Var(CMI)/Var(B_L) at L=64?** Two live hypotheses. (a) PERSISTENT SELECTION — τ_X is systematically LONGER for controlled at every L (18.7/39.5/64.5/103.4 vs 12.6/25.8/40.0/75.3 windows); the control converts large fast-varying selection into small slow-varying selection, which is strictly better for a path weight but can be equally damaging for a genealogical particle system when the SAME ancestry is repeatedly favoured. Crude amplitude²×τ at L=64 gives 0.33×1.25≈0.41, still substantially weaker globally, so persistence probably does NOT explain all of it. (b) OBSERVABLE ALIGNMENT — the control is built from ONE feature K=Σ_j q_j, optimised to predict future ACTIVITY. Var(log W) is activity-dominated so it improves 16.9×; B_L couples to the slow ENTANGLEMENT-critical mode, which h=e^{aK} may leave almost untouched. Near the transition that is physically plausible. **Hypothesis (b) is the more likely and the more interesting.**
**NEXT EXPERIMENT (do this INSTEAD of any further N_c or intermediate-L ladder).** Instrument the actual cloning population — at every resampling event record D₂^pop=log(N_c Σp_i²), A_k, parents lost D_k=N_c−#distinct, realised pair coalescence C_k=Σn_i(n_i−1)/(N_c(N_c−1)), and per-family δ_a(k)=P_a(k)−f_a(k) (weight share minus population share). Then compute the family selection autocorrelation C_fam(Δ)=⟨δ_a(k)δ_a(k+Δ)⟩ and, decisively for (b), Cov(δ_a(k), E[B_L(T)|a,k]). Run L=32 FIRST (known success, GESS 1.99→6.23, E_B_L=1.42×) so the diagnostic must reproduce a case that worked, then L=64 (GESS 1.25→4.76 but E_B_L≈0.99). The L=32-works vs L=64-fails contrast is far more informative than more N_c points. Do NOT change the resampling interval or add a second control feature before this measurement — and note that a cheap precursor to (b) is simply to measure Corr(K, B_L) and Corr(K, CMI) across trajectories: if K is near-orthogonal to the entanglement observables, (b) is essentially confirmed for a fraction of the cost.
**METHODOLOGICAL WARNING.** Three cheap proxies have now each predicted a large production win that the actual cloning experiments declined to deliver: Var(log W)↓ ⇏ N_c can be reduced; GESS↑ ⇏ Var(O)↓; ΣD₂↓ ⇏ N_c can be reduced. **Do not accept any further proxy-based claim without a direct population measurement.**

**[HISTORICAL — the two-point reading below was overturned by guided N_c=128; kept for the record.]** First two L=64 configs (ζ=0.9, T=64, a*=−0.3761, τ_K=3.76, equal 1500 s budgets): ctrl N_c=32 → n=14, 109.2 s/rep, GESS=4.76, S=2.2519±0.0264, CMI=0.4973±0.0160, B_L=1.1212±0.0422; guided N_c=32 → n=16, 96.3 s/rep, GESS=1.25, S=2.3379±0.0245, CMI=0.5370±0.0167, B_L=1.2586±0.0474. **At L=32 the controlled arm sat ABOVE guided (CMI 0.5680 vs 0.5312); at L=64 it sits BELOW (−1.71σ CMI, −2.39σ S, −2.16σ B_L).** The sign of the controlled-minus-guided offset therefore REVERSES between L=32 and L=64. Consequences: (a) the "control reduces finite-N_c bias" reading is NOT established and may be an L=32 artifact; (b) the decisive test is now whether the guided sequence at L=64 moves monotonically toward the controlled value as N_c rises — at L=64 that requires CMI to DECREASE 0.5370 → guided(64) → guided(128) → ctrl 0.4973, the opposite direction from L=32. Finite-particle bias for a nonlinear entanglement observable has no universal sign, so an L-dependent sign flip is not by itself a contradiction, but it removes any right to quote a direction. (c) A third possibility that MUST be excluded before anything else: a control-side defect that only appears at larger L. The a=0 identity was gated at L=16/32 but NOT at L=64 — **re-run the a=0 gate and the mean-one martingale gate at L=64 before interpreting these means at all.** Note guided N_c=32 at L=64 has GESS=1.25, i.e. near-total genealogical collapse (effectively one ancestor family), so its mean is the least trustworthy number in the table, not the reference. Equal-wall variance efficiency from these two arms alone: ctrl/guided = S 0.75×, CMI 0.92×, B_L 0.99× — i.e. the L=32 variance advantage (B_L 1.42×) is ALSO not reproduced at L=64 in this pairing. Remaining configs (guided 64, ctrl 24, guided 128) were still running as of 2026-08-09; read `/tmp/ladder_L64.json`.

**NEXT ACTIONS, in order.** (0) **RE-GATE AT L=64 FIRST** — a=0 identity (|log W + (1−ζ)Λ| must be ~1e-14) and the mean-one martingale E[R_T]=1 with Simpson, at L=64. Everything below is void if those fail. (1) [DONE 2026-08-09] Promote `/tmp/csampler2.py` → `analysis/var_reduction/controlled_sampler.py` and `/tmp/doob_galerkin.py` → `analysis/var_reduction/galerkin_control.py`. (2) Offline `χ_sel(h) = D_2(h)/h` from stored no-resampling paths to pick the resampling interval WITHOUT another chunk scan. (3) **N_c ladder: controlled N_c=16/24/32 vs guided N_c=32/64** — this is where the large gain would be, not the per-trajectory 20–30%; check finite-population drift in S, CMI, B_L. (4) **L=64 LADDER IS IN FLIGHT as of 2026-08-09** — `analysis/var_reduction/nc_ladder.py` (now a repo file, env-configurable via `L/ZETA/T/AK/CONFIGS/OUT`), launched as `L=64 ZETA=0.9 AK=-3.57` with `CONFIGS=ctrl:32:1500,guided:32:1500,guided:64:1500,ctrl:24:1500,guided:128:3000` (equal wall budget per config, ~2.5 h total). Streams to `/tmp/ladder_L64.log` and `/tmp/ladder_L64.json` after EACH config, so a partial run is usable; if the process died, relaunch with the same env. a*=−0.3761, τ_K=3.76 at this cell. **Read it with `analysis/var_reduction/nc_bias_fit.py`** (joint shared-O_∞ fit). What to look for: the L=32 pattern was guided(32) < guided(64) < guided(128) → ctrl plateau in CMI and B_L. If that repeats at L=64 — where the guided bias should be LARGER and hence easier to resolve, and where the path-level control gain is 16.9× rather than 9.4× — the finite-population-bias story moves from [P] to substantially established. **Use CMI and B_L as the primary diagnostics and S only as a consistency check** (S did not show clean convergence at L=32; different nonlinear observables have different finite-particle corrections). Do NOT push controlled N_c below 24: the L=32 ladder shows a real floor at N_c=12 and worse efficiency at 16, so the value of the method is that it uses a MODEST population much more effectively, not that it permits a tiny one. (5) Coupled-λ triplet λ_c±δ and the true production metric `E_∂Q = 1/(t_wall·Var[Q_+ − Q_−])`. (6) Fit a_K at each production ζ (one scalar, L=32). CAVEAT (corrected 2026-08-09): late-time snapshots remain EXACT under the taper but require snapshot-specific untwisting, `Ô(t) = Σ_i w_i(t) e^{−a(t)K_i(t)} O_i(t) / Σ_i w_i(t) e^{−a(t)K_i(t)}`. Their net efficiency under the taper is UNVALIDATED — the open question is whether that untwisting costs enough ESS to negate the snapshot gain, not whether it is legal. An earlier version of this block said the two "cannot be combined"; that was too strong. Note snapshots taken inside the taper region (T−t ≲ τ_K) have |a(t)| ≪ |a*| and so a much weaker untwisting penalty, which is the obvious thing to try first. (7) QUADRATURE MODES — keep both and document: `trapezoid` for production observables (the smooth multiplicative bias cancels in the SNIS ratio), `simpson` for the mean-one likelihood gate (where a tiny systematic error is exponentially amplified). Measured: C-vs-D agree on all observables within 1σ, GESS 6.13 vs 6.23, 33.4 s vs 20.9 s. (8) FREEZE THE ANSATZ at `h = e^{a(t)K}` — one feature, one Galerkin coefficient, predicted not fitted, transfers across L, exact M=1 domination. Do NOT add C_2/C_3/E features unless the N_c or L=64 tests fail; that would trade an understandable algorithm for a high-dimensional optimisation. (9) ζ-DEPENDENCE: test ζ=0.7/0.8/0.9 before attempting 0.5, since the derivation is first order in s=−log ζ and s≃0.693 at ζ=0.5.

### Cloning-estimator redesign: chunk/resample decoupling, four external memos adjudicated (2026-07-27) — QUALIFIES the "LOW-LEVEL speedup is exhausted" claim in the 2026-07-07 block; CONFIRMS `VARIANCE_REDUCTION.md` §5; CLOSES full-path IS and multi-ζ reweighting

All measured on Mac (`.venv`, L=64/96, N_c=32, T=L, guided c=ζ, lowrank+newton), BLAS pinned. Prototype `/tmp/adaptive_cloning.py` (`run_cloning_3clock`) decouples the three schedules that production `delta_tau` conflates: trajectory chunk length, resampling period, observation times. **Bit-identity gate [V]:** with always-resample and chunk=δτ_prod it reproduces `run_cloning` to |Δθ|~1e-15 (logsumexp roundoff) at L=32, ζ∈{0.6,0.9}, so the carried-cumulative-weight recursion collapses onto the per-window recursion exactly. No repo files changed.

**[V] Wall time splits ~43/57 window/jump at ζ=0.9; the chunk length is the only remaining low-level lever.** Scaling δτ by mult and fitting wall = A/mult + B is exact over mult∈{1,2,4,8}: A=42.4 s (per-chunk fixed cost: `V_inv@orbitals`, Gram+Cholesky branch-norm, `V@A`+trisolve, `covariance_from_orbitals`, strided entropy), B=55.9 s (jump work). Saturation curves (L=64, deterministic resample-every-chunk, 3 seeds): ζ=0.9 → 1.00/1.42/1.51/1.56/1.61 at mult 1/4/8/16/32 (asymptote ≈1.65); ζ=0.5 → 1.00/1.98/2.38/2.71/**2.77** (asymptote ≈2.8). **The lever is LARGER at low ζ**, opposite to the framing in all four memos: window overhead scales with the chunk count while jump work scales with ζ×chunk count. mult=8 captures 92% of the ζ=0.9 gain, mult=16 captures 97% of the ζ=0.5 gain. Recommended policy is propagate-to-`min(next resample, next snapshot, T)` rather than a fixed multiplier. Chunk length is NOT numerically constrained here: intra-chunk propagation is exact/event-driven and the float64 conditioning limit (spread of Re(evals·dt) ≲ 36) leaves ~3 orders of magnitude of headroom [P]. NOTE: this reopens the 2026-06-07 "dtau 3× NOT certified" item, which was measured on the NON-guided estimator (`tools/validate_dtau_worker.py` passes no `proposal_c`/lowrank/newton) where the per-window weight was ζⁿ rather than the smooth exp[−(1−ζ)ΔΛ]. An R≥30 bias test at production L is still owed before banking mult>1.

**[V] Adaptive ESS-triggered resampling is a NULL. `VARIANCE_REDUCTION.md` §5 stands and now has the right mechanism.** Pilot (L=64, T=64, N_c=32, 8 paired seeds, ρ=0.5): resampling events fall 3826→14 at ζ=0.9 and 2852→37 at ζ=0.5, yet wall time is unchanged (92.7 vs 93.4 s) and genealogy does not improve (GESS 1.11 vs 1.52 at ζ=0.9; 1.00 vs 1.00 at ζ=0.5). Mechanism, which the memos invert: **the total degeneracy is conserved.** Over the horizon Var(log W)=(1−ζ)²Var(Λ_T)=36.8 (L=64) / 68.8 (L=96), so per-window ESS sits at 1.000 *because* frequent resampling suppresses it, not because selection is weak. The ESS threshold is exactly what sets per-event selection strength (resampling at ESS=ρN_c kills ~half the lineages at ρ=0.5), so 3826 weak coalescence events and 14 strong ones give the same product. Rescheduling moves degeneracy between the weight channel and the genealogy channel; it does not reduce the total. Cloning IS doing real work at ζ=0.9. Two new diagnostics worth keeping: **GESS** = N_c²/Σ_a f_a² (family sizes; strictly better than `n_distinct_ancestors`, which cannot distinguish 1 ancestor from 126/1/1) and the **coalescence budget** Σ_k 1/ESS_k (= K/N_c for near-uniform weights; collapse at O(1); baseline runs at 119.6). An n=8 mean/sd difference in the adaptive arm was initially mis-read as SNIS bias — it is not, since the chunk=4δτ arm has the same resampling schedule and shows no shift. Do not re-derive that.

**[V, CLOSES two memo proposals] Full-path importance sampling and multi-ζ trajectory-bank reuse are dead below ζ≈0.99.** Running the prototype with `ess_threshold=0` never resamples, so the clones ARE M independent proposal trajectories at c=ζ (Regime A, measured not assumed). ζ=0.9: L=64/T=64/M=64 → ⟨N_T⟩=1592±91, ⟨Λ_T⟩=1759±61, Var(log W)=36.8, ESS_path/M=0.033; L=96/T=96/M=24 → 3520±121, 3932±83, Var(log W)=68.8, ESS_path/M=0.078. Var(Λ_T) grows as L^1.85 (T=L), extrapolating to Var(log W)≈133 at L=128. Requiring ESS_path/M>0.5 (i.e. Var(log W)≲0.7) needs **ζ>0.986 at L=64 and ζ>0.993 at L=128**, so on the production grid only ζ=1 qualifies, where the weights are exactly 1 and it reduces to ordinary Born trajectories. Multi-ζ reweighting dies for the same reason: measured ESS_path/M is **flat** across targets ζ'∈[0.86,1.00] (0.031–0.045 at L=64), so the degeneracy is the (1−c)Λ_T compensator, not target mismatch — the bank is already degenerate for its own ζ and there is nothing to share. No overlap analysis needed.

**[V] Rate convention pinned: r = 2α Σ_j q_j exactly (ratio 2.0000±0.0000 over 40 states).** Hence R_max = 2α(L−1) is the valid uniformization bound, and max q_j never exceeds 1 (measured 1.000000 at ζ=0.9, 0.999998 at ζ=0.5, frac(q>1)=0). Measured q̄ = 0.443 at ζ=0.9 and 0.300 at ζ=0.5, giving **2.26 and 3.33 thinning candidates per accepted jump**. CORRECTION: an earlier estimate of ~1.09 candidates/jump (from Λ_T/(α(L−1)T)) and an external memo's ~1.10 are both wrong by exactly the factor 2 — they assumed r = α Σ q_j. The O(L²) bond read Γ_ab = −2 Im Σ_m U_am conj(U_bm) is gated exact against `covariance_from_orbitals` (7.8e-16) and is required for thinning to be competitive.

**[O, BUG OPEN — DO NOT SHIP] Exact Poisson thinning implemented, fast, but fails equivalence.** `/tmp/thinning.py` (draw τ~Exp(R_max), propagate no-click, accept with prob Q/(L−1), channel ∝ q_j/Q, accumulate physical Λ at every candidate). Single-long-trajectory wall time 1.23× (ζ=0.9) / 1.11× (ζ=0.5) faster than Newton at L=64, below the 1.37× microbenchmark projection because a rejected candidate advances the state, so `coeffs=V_inv@orbitals` cannot be amortized across candidates the way Newton amortizes it across ~3.7 hazard evals. But N_T is systematically LOW: −24.3 (−1.5%) at ζ=0.9 and −21.9 (−4.8%) at ζ=0.5, M=48. **Arbitration [V]: Newton and brentq at bisection_tol=1e-13 agree to all digits (0.00σ, N_T=456.44 both), and both differ from thinning by 3.09σ**, so the root-solve path is self-consistent and the bug is in the thinning prototype, not in the production Newton solver. Deficit scales with candidates/jump (1.5%@2.26, 4.8%@3.33) ⇒ a per-candidate loss. ELIMINATED suspects: q_j>1 clipping (never fires), R_max validity, exponential scale convention, acceptance-test algebra, final-interval handling, Λ accumulation. Microbench (L=64): newton_eval 0.186 ms, thin_candidate 0.188 ms, covariance_rebuild 0.095 ms. Next step is to localize the per-candidate loss before any further benchmarking.

**[V, BEST SINGLE WIN] Late-time snapshot averaging gives the full independent-sample gain.** Validated at the POPULATION-estimator level (not from per-clone autocorrelation, which is the wrong quantity): 12 realizations, L=64, chunk=4δτ, K=4 snapshots at t=24/40/56/64 (spacing 16), g_snap = Var(O_terminal)/Var(O_avg) = **4.00 (S, ζ=0.9), 3.02 (B_L, ζ=0.9), 4.33 (S, ζ=0.5), 4.24 (B_L, ζ=0.5)**, i.e. g_snap ≈ K. Pooled snapshot-window drift slopes all |t|<1.9 (B_L at ζ=0.9 is the borderline −1.90, which is also why its g_snap is the lowest). Per-clone autocorrelation of B_L decays to ~0 by lag 16 at L=64 (0.46@3.2, 0.24@9.6), so τ_int ≈ 5–10. GESS≈1 does NOT prevent this: genealogical collapse is ancestry from t=0, while the observable decorrelates over ~40 time units and ~600 resampling events. Cost is a few extra eigendecompositions against thousands of jumps. At L=128 the usable window 0.75L supports ~5 snapshots. This makes the "fork-after-burn-in" proposal redundant: fork and snapshot averaging harvest the SAME resource (post-burn-in trajectory time), so their gains do not multiply, and fork's genuine incremental value over snapshots is only burn-in amortization, ≤1/(1−0.25)=1.33×, not the 2.5× claimed.

**[V] N_c reallocation.** Measured sd(S) ∝ N_c^{−0.285} (L=48, ζ=0.9, 6 paired seeds, N_c 16→64 gave sd 0.0751→0.0508, ratio 1.48 against the ideal 2.00), so at fixed clone-time budget Var(mean) ∝ N_c^{0.43} and **N_c 128→32 with 4× more realizations buys 1.83×**. Consistent with the 2026-06-17 ladder finding that finite-N_c bias cancels in crossings; keep an N_c=64 validation rung for ν, where slope estimators need not inherit that cancellation. NOTE this tension with the 2026-06-17 instruction "use a constant baseline N_c≈400–500": that was about not letting N_c FALL WITH L, which still stands. The reallocation here is at fixed total budget and needs a crossing-level check before production adoption.

**REJECTED this session.** (i) Reducing T (T/L∈{0.5,0.75}) — contradicts the 2026-06-17 T-cap finding directly, and the memo's own snapshot ladder cannot support it since its B_L(t) swings 5× under a degenerate ESS≈2 path-IS estimator. (ii) Cost-optimised c<ζ — already closed in `VARIANCE_REDUCTION.md` §1 (Var×wall favours c=ζ by ~2×). (iii) Fork-after-burn-in, per above. (iv) Coupled-λ g_couple must NOT be multiplied into a crossing-precision estimate: a crossing is set by differences between L-curves at fixed λ, so coupling enters only through the slope in Var(λ_c)=Var(ΔQ)/(ΔQ′)².

**Combined estimate [P].** 1.5× (chunks, ζ=0.9; 2.4–2.8× at ζ=0.5) × 1.83× (N_c reallocation) × ~3–4× (snapshots) ≈ **8–11× in precision per core-hour**, essentially all of it estimator-side rather than sampler-side. Not yet validated at production L, and every factor above is L=64/96 and n≤12. Do not advertise before an L=128 confirmation.

**Files (scratch, not committed).** `/tmp/adaptive_cloning.py` (three-clock prototype + GESS/coalescence-budget diagnostics), `/tmp/gate_identity.py`, `/tmp/dtau_scan.py`, `/tmp/nc_test.py`, `/tmp/pilot3.py`, `/tmp/pathis.py`, `/tmp/memo4.py`, `/tmp/thin_decide.py`, `/tmp/thinning.py`, `/tmp/thin_test.py`, `/tmp/thin_arbitrate.py`, `/tmp/qmax.py`, with logs and JSON alongside. Promote to `analysis/var_reduction/` if any of these become production.

### Ruche migration + trajectory speedups + cost/r calibration (2026-07-07) — SUPERSEDES the "L≥192 out of reach / L=256 infeasible" conclusion in "Dense campaign empirical findings"; updates Operational (Habrok→Ruche)

**Code (committed + pushed, ueborg/ppsQJ_m2 main, commits f2a6b68 + e67d26d).** Two trajectory speedups, both flag-guarded (default off, production path byte-unchanged):
- **Low-rank active-subspace jump update** [V]: replaces the per-jump eigendecomposition with a rank-≤4 orbital rotation, O(L²) vs O(L³). ~2.2× per trajectory, BIT-IDENTICAL to the eigh path (same-seed to ~1e-13 over full trajectories; 84 single-jump tests; cloning population θ/S/CMI/ESS to ~1e-14, no bias). `jump_update_method="lowrank"` / env `PPS_JUMP_METHOD=lowrank`.
- **Safeguarded-Newton waiting-time solver** [V]: analytic integrated-hazard derivative Λ'(t)=−Re Tr(Q†KQ)+αN (K=h_effective), bracketed Newton with bisection fallback, ~3.7 evals/jump vs ~14 for brentq, returns the normalised propagated orbitals (fuses the QR). STATISTICAL, not bit-identical — perturbs the accepted waiting time at ~1e-6 (eps_hazard); paired-seed cloning θ/S/CMI/ESS agree to ~1e-7 (no population bias); jump counts identical; eps_hazard-robust in [1e-8,1e-10]. `solver_method="newton"` / env `PPS_SOLVER=newton`. 16 regression tests pass.
- Cumulative: ~4.5–4.9× on a single long trajectory, but **~2.8× at the CLONING level** (short windows shrink Newton's gain: 2.2×lowrank × 1.27×newton). Report ~4.2× for production (2.8× before the no-jump Cholesky-reuse below). NOTE (Ruche cProfile 2026-07-07): the no-jump-window QR (`gaussian_backend.py:496`) is the #1 cost in the CLONING regime (~41% of the scalar trajectory, 5696 calls) — single-long-trajectory profiling missed it (few no-jump windows). Reusing the Cholesky already built by the no-jump check on the line above (Q=Y·R⁻¹ instead of a fresh QR; bit-identical since only the covariance propagates between windows) is DONE — implemented + validated 2026-07-07 (commit ca554b8): bit-identical to ~1e-15 (θ/ESS/jump-counts unchanged vs the QR path on both eigh+brentq and lowrank+newton), **1.50× measured on run_cloning** (above the 1.4× estimate) ⇒ cumulative **~4.2× at the cloning level**, of which the fully-bit-identical lowrank+chol-reuse part (no Newton) is 3.3×. LOW-LEVEL speedup is exhausted: profiling shows ~94% of the solver is BLAS (matmul/chol/trisolve) so numba/Cython/C give nothing; grouped-roots eliminated (guided-cloning ESS~0.97 ⇒ no per-window clone coalescence); the L⁵ cost (=N_c·T·L⁴ at fixed T, matches the on-disk cost law) is intrinsic (n_jumps∝L², per-jump O(L³)).

**Runner + Ruche infrastructure (committed).** `scripts/run_local_boundary.py`: checkpointed, resumable (per-realisation JSON, idempotent), `--shard/--nshards` round-robin for Slurm arrays, `run`/`aggregate` modes (aggregate → B_L crossings λ_x, λ_x/ζ vs Lζ²). `scripts/ruche/`: `submit_pps_boundary.sh` (size-binned arrays: cpu_med L≤128, cpu_prod L=192, cpu_long L=256; %K throttle so K·40 ≤ core cap), `setup_ruche.sh` (conda env numpy2+scipy/MKL under $WORKDIR), `calib_ruche.sh`, `profile_code.py` + `profile_ruche.sh`, README.

**Cost model + r calibration [V].** Confirms the on-disk cost law t ∝ N_c·T·L⁴ (= N_c·L⁵ at T=L). Optimised anchor (lowrank+newton) L=128/N_c=128/T=128 = **25.5 min on Mac (Apple/Accelerate) / 62 min on Ruche (Cascade Lake Xeon 6230 + MKL, 1 core, threads pinned, ESS 126/128)** ⇒ **r = Ruche/Mac = 2.43** (measured, job 1232455). Per-realisation on Ruche (r=2.43, T=L): L=64 ~4 min, L=96 ~20 min, L=128 ~62 min, L=160 ~5 h, L=192 ~13 h, L=256 ~55 h. ESS ~0.97–0.99 across ζ∈[0.2,1] (guided cloning; the old "ESS collapses at large L" was the NON-guided estimator).

**FEASIBILITY REFRAME [V].** The "L≥192 out of reach / L=256 structurally infeasible" verdict (Dense-campaign findings) was for the OLD non-guided code on Habrok and rested on TWO blockers now both removed: (i) compute — the 2.8× speedup + Ruche (cpu_long 160c/168h, cpu_med 1000c/4h, cpu_prod 2000c/6h nights); (ii) ESS variance inflation — guided cloning gives ESS~0.97 (not collapsing). So **L=256 is now compute-feasible (~55 h/realisation at T=L on cpu_long)**, enabling the ONE lever the 2026-06-17 correction-model analysis said was needed to tighten ν_B (larger L to suppress L^{−ω}), previously cost-blocked. HONEST caveats: (a) the correction-model confidence set [1.5,3] means L=256 may TIGHTEN rather than PIN ν_B; (b) the ν tier needs T≥2L (below) ⇒ L=256 ν ~110 h/realisation (still < 168 h cpu_long cap for one realisation, but a multi-λ×n_real L=256 ν dataset is ~1–2 weeks wall).

**T probe [O, inconclusive].** A value-at-λ_c snapshot probe (CMI/S_AB vs t within one run, L=96/128, N_c=128) is noise-dominated at single-realisation level (fixed-t CoV ~10% S_AB, ~25% CMI) and does NOT resolve t*. The T-cap finding STANDS: boundary/λ_c tier can use T~L (values saturate by ~0.5L per the 2026-06-07 lever-check), but the ν tier needs **T≥2L** (slopes ∂_λI), NOT reduced — a cost-motivated T-reduction for the ν tier was considered and REJECTED. Data-plan item #3 (τ_int / ∂_λI-stationarity pilot, multi-realisation) is still owed BEFORE committing the expensive large-L ν tier.

### Updated-data reanalysis + FSS-method calibration + T-cap finding (2026-06-17) — QUALIFIES "Numerics (best current estimate)", "Key result: $\phi$ from global FSS", and the "N_c-ladder campaign" sections below

Worked entirely from the guided-cloning aggregates on disk. No jobs submitted, no theory files changed. Five external strategy memos were reviewed and reconciled; the operative conclusions are below.

**Aggregates rebuilt** in `~/Downloads/pps_aggregates/` (full scalar set; ladder also keeps per-realisation `_all` arrays): `agg_caseB_combined.pkl` (1046 rows = prod 925 + highL 121), `agg_pps_caseA_guided.pkl` (574), `agg_ladder.pkl` (161). Case B grew: L=160 complete to ζ=0.3, L=128 to ζ=0.5, L=96 to ζ=0.85, L∈{32,48,64} complete across ζ. Quicklook figures fig1–fig9 in `~/Downloads/pps_quicklook/`.

**[V, CRITICAL] T was capped at 128 for L≥96.** Production T/L = 2.0 for L≤64 but 1.33, 1.0, 0.80 at L=96, 128, 160 (independent of ζ). Size-correlated under-relaxation: the largest systems had the least relaxation relative to size. Near criticality the relaxation time grows as L^z, so T<L at L=160 is almost certainly too short, and it biases large-L observables, the slope ∂_λI especially, hence ν. The existing L≥96 data is suspect for ν and should be RE-RUN at proper T (T≥2L, or T from a measured τ_int pilot), NOT merely supplemented. Fix T before any expensive campaign.

**[V, DECISIVE] The global FSS collapse cannot resolve ν at this L-range/noise.** Synthetic calibration (known ν_true, real L-grid, real λ-grid, real error bars, run through the same pipeline): recovered ν̂ is non-monotonic and compressed to ≈1.2–1.7 regardless of ν_true ∈ {1, 1.5, 2, 2.5}. Cause: a flat χ² surface — a flexible master curve plus a free λ_c absorbs the x-rescaling that changing ν produces, so the ν-minimum is noise-located (at true ν=2 the synthetic collapse already has χ²_r≈1 yet the fit lands at 1.4). Consequence: collapse-based ν (1.3 from B_L, 1.6 from CMI) are NOT measurements of ν, and the older "ν scattered around ~2 plateau" is NOT established by collapse. The "ν drifts 1.3→3 / maybe ζ-dependent" worry raised earlier this session was largely a B_L+collapse artifact. Do not re-derive it.

**[V] Use ⟨CMI⟩, not B_L, for exponents.** B_L = ⟨CMI·S_{L/2}⟩ carries an explicit lnL factor that biases ν LOW (synthetic ν=2 → B_L collapse 1.66 vs CMI collapse 1.92). Primary scaling observable = ⟨CMI⟩. Secondary locators = the KMR product ⟨CMI⟩⟨S_{L/2}⟩ and ΔS_L = S(L/2)−S(L/4) (cancels the non-universal additive constant; its critical value ∝ c). Never use the trajectory product ⟨CMI·S⟩ for exponents.

**[P] Quotients (phenomenological-RG slope ratio) is the usable Cut B ν method, but marginal.** ν_eff = ln2/ln(m_2L/m_L) at the crossing tracks ν_true MONOTONICALLY (unlike collapse), so it has real power, but at n_real=25 pooled over 5 ζ it separates ν=1.5 from ν=2 at only ~1σ on (48,96). Per-pair ν_eff is biased low and drifts up with L, so it must be extrapolated over the pair sequence — which is why L=80 (giving the missing (80,160) pair) is high value at 1/16 the cost of an L=160 run. A clean ν needs n_real≥25 PLUS the full pair sequence with extrapolation PLUS coupled-λ slopes (√2) PLUS ζ-pooling under the constant-ν null. Calibrate before trusting; quotients at n_real=5 is pure noise.

**[V] What still stands for λ_c/φ.** In the clean window ζ≥0.25 the crossings are well-defined, λ_c(ζ) tracks 0.5√ζ, a free power-law gives φ≈0.5–0.6 (consistent with 1/2, B_L leaning ~0.6), and the boundary extrapolates to the Born point, A(ζ→1)≈0.48–0.53 (B_L collapse A=0.50). The Born-endpoint reproduction is the robust headline. Small-ζ (ζ≤0.2) is NOT resolved: the L=96/128/160 curves bunch and spike near λ→0 (approach to the gapless Hermitian λ=0 point), and the "crossing" depends on the λ-window (13λ→0.105 vs 7λ→0.18 at ζ=0.1). Report small-ζ λ_c as bounded, not measured.

**[V] N_c ladder result** (`~/Downloads/pps_clone_guided_ladder`, L=96/128 at N_c=600, L=160 at N_c=500, ζ≤0.4 — the 2× rung over the crossing region): finite-N_c bias on B_L/CMI VALUES is ~3% (lower at higher N_c), size-correlated because production N_c itself falls with L (500→250 from L=32→160). But the bias CANCELS in crossings: on a fixed λ-grid the (96,128) prod-300 vs ladder-600 crossings agree within MC error (ζ=0.1: 0.181±0.018 vs 0.182±0.004). N_c is NOT the lever for crossings or for the small-ζ problem. This refines the 2026-06-05/07 N_c-ladder block: more N_c does not repair small-ζ. Do not decrease N_c with L; use a constant baseline N_c≈400–500.

**[P→parked] Split coupling.** Prototype `/tmp/split_coupling.py` (fine-dt joint +δ/−δ evolver; shared occurrence uniform = min/residual jump split, maximally coupled channel + resampling) built and physics-validated vs the exact backend (+2–4% common-mode operator-splitting bias that cancels in differences). NO demonstrated variance benefit: VR_S=0.36, VR_B=0.54 at δ=0.06, but NR=8 is noise-dominated (F(7,7) 95% spread ≈ [0.2,5]). Use the validated naive CRN coupling (δ≤0.04, ~2× on ⟨CMI⟩/⟨S⟩ differences) for production slopes. Split coupling stays optional, only at NR≥30. The variance-reduction banner's "Next: split coupling" item is thus answered (no win yet).

**Data plan (reconciliation of 5 memos). Depth, not breadth; λ_c is done.** Order:
1. Build + calibrate a GLOBAL shared-ν corrections-aware fit (one ν, one ω, parametric scaling function, λ_c(ζ)=Aζ^φ, fit to all clean ζ and L jointly). DECISION GATE — RESOLVED 2026-06-17 (RUN, FAILED). All FOUR collapse variants — per-ζ free cubic, global free-shape, global shared-shape, and shared-shape with λ_c FIXED to the crossing — fail to recover a known ν at n_real=25: recovered ν is non-monotonic and bound-hitting (ν_true=1.0→3.0, 1.5→1.6, 2.0→3.3, 2.5→3.2). Cut B ν is NOT extractable by any collapse/global fit at L≤160 (log-range only ~1.6; a flexible scaling function absorbs ν and the L^{-ω} term only adds freedom). Only LOCAL estimators have power: (a) quotients ν_eff=ln2/ln(m_2L/m_L), monotonic but ~1σ for ν=1.5-vs-2 at n_real=25, needs the full pair sequence incl (80,160) + extrapolation + coupled-λ slopes; (b) Cut A curvature κ_L~L^{2/ν_A}, clean ν_A=1.09±0.09 at ν_true=1 with the ν≈1-tuned grid, but it only CONFIRMS ν≈1 and cannot robustly detect a crossover (fixed grid loses power for ν≠1; the adaptive-window variant is noisier). NET: make Cut A the headline ν (confirm Ising ν_A≈1); report Cut B ν as bounded / consistent-with-2 via quotients. The binding constraint for a PRECISE ν is the L-RANGE, not n_real/N_c/T, so do not over-invest statistics at L≤160 expecting a sharp Cut B ν. Calibration scripts: /tmp/pps_global_fit.py, /tmp/pps_global_shared.py, /tmp/pps_global_fixedlc.py, /tmp/pps_cutA_curv.py, /tmp/pps_cutA_adapt.py, /tmp/pps_calib.py, /tmp/pps_calib_quot.py.

   **LMR/Lavasani interpolation collapse — RUN 2026-06-17, the one ν method that WORKS [V].** The nonparametric interpolation objective ε(λ_c,ν)=Σ_i (y_i − interp from x-neighbours)²/V_i on CMI (LMR SciPost; Lavasani 'estimating errors'), cross-size variant = predict each point from OTHER sizes only (kills the same-L-smoothness degeneracy), is genuinely DIFFERENT from the 4 failed parametric collapses: no flexible F to absorb ν. Noiseless sanity exact. On synthetic known-ν it is MONOTONIC and tight (unlike every parametric collapse). It is STATISTICS-DEPENDENT: at n_real=25 the recovered-vs-true map is 1.5→1.20, 2.0→1.44, 2.5→1.60 (compressed-but-monotonic; ν=1.5-vs-2 at ~2σ) so it is a CALIBRATED estimator (invert the map); at n_real=5 (current data) the map SATURATES ~1.3 for ν≥2 (2.0→1.31, 2.5→1.36, 3.0→1.31) and cannot distinguish 2 from 3. THIS CORRECTS the 'binding constraint is L-range not n_real' sentence above: for the LMR method n_real IS the lever (5→25 restores ν~2 discrimination), so the high-n_real ζ=1 ladder is JUSTIFIED as the route to a defensible Cut B ν. REAL-DATA application (existing aggregates, cross-size VerB): (i) Cut A with λ_c=1/2 FIXED is nearly unbiased and very tight even at n_real=5 (true→recovered ≈ identity), real ν_A≈0.9 → CONSISTENT with Ising ν_A=1 (corroborates the curvature method; reliable only at small ζ where 4 sizes 32/48/64/96 exist, large-ζ Cut A is sparse-L and reads spuriously low ~0.5). (ii) Cut B (n_real=5, all L incl T-capped) pooled ν̂≈1.5 → consistent with ν_B≳2 but UNPINNED (saturated map; do NOT quote a point value). NET upgrade to the gate verdict: ν_A≈1 is defensible by two methods NOW; Cut B ν_B is consistent-with-2 and becomes PIN-ABLE with the n_real=25 ζ=1 cloning-free ladder + LMR-VerB + calibration-inversion (and T fixed). Scripts: /tmp/pps_lmr.py (objective + Cut B calib), /tmp/pps_lmr_real.py (real application + inversion), /tmp/pps_lmr_calib2.py (n_real=5 map + Cut A fixed-λ_c calib/inversion).

   **CONFIDENCE-CONSTRUCTION UPDATE 2026-06-17 — RETRACTS the 'pin-able at n_real=25' claim in the paragraph above [V].** The single-curve inversion (ν̂≈1.5 → ν_B≈2.2) is NOT valid. The proper simulation-based confidence construction marginalises the recovered-ν̂ distribution over a correction-model FAMILY (F-shape {logistic,tanh,erf} × correction {additive-const, additive-slope, x-shift, poly} × ω∈{0.5,1,2} × ±amplitude, 7 representative members, real grids/errors/missing-cells). Result: the model-marginal ν̂ distributions for different ν_0 OVERLAP heavily, so for the observed Cut B ν̂≈1.5 the 95% confidence SET is {1.25,…,3.0} (essentially everything) and the 68% set is {1.5,1.67,2.0,2.5,3.0} — AND this does NOT tighten from n_real=5 to n_real=25. The earlier 'n_real un-saturates the map → ladder pins ν_B' was a SINGLE-model artifact; once correction-model uncertainty is included (mandatory, the real corrections being unknown) it DOMINATES, and sampling noise (n_real) is secondary. CONSEQUENCE: pinning ν_B at L≤160 is NOT achievable by adding statistics — it requires LARGER L (suppress L^{-ω} corrections; L^4-expensive) OR constraining the correction family to the measured crossing-height + crossing-location drift (the LMR-memo refinement, NOT yet done — could tighten the set, but the L-range still limits it). HONEST Cut B status: consistent with ν_B≈2 but confidence set ~[1.5,3]; NOT a measured exponent. n_real still helps the precision of the statistic at FIXED model, and the ζ=1 cloning-free ladder is still worth running for λ_c/φ/Born and a cleaner ν̂, but it must NOT be sold as pinning ν_B. Cut A (λ_c=1/2 fixed) is far less affected and remains the clean ν route (still owes a corrections-included calibration before a final number, per the LMR memo; current read ν_A≈0.9 ⇒ report as 'consistent with Ising ν_A=1', not a ±small number). Script: /tmp/pps_lmr_robust.py.

   **ν(ζ) IS A HYPOTHESIS TO TEST, NOT ASSUME — LMR PRECEDENT VERIFIED 2026-06-17 [V].** Read the actual LMR paper (PhysRevX.15.021020, Leung–Meidan–Romito; on Mac at ~/Downloads/PhysRevX.15.021020.pdf). For the measurement-only transition it finds ν NOT constant in ζ: ν≈1 (Ising) at strong PPS (small ζ), an ABRUPT deviation in a narrow window around a finite ζ*, OVERSHOOT to ν≈2.3 near the transition, then the monitored value ν=5/3 by ζ≈1 — strongly nonmonotonic, driven by a Luttinger-parameter divergence at finite ζ* (LMR Fig. 7 + Sec. VI A; ζ convention IDENTICAL to ours, ζ=1 monitored / ζ=0 postselected; LMR extract ν with the SAME interpolation-collapse ε(ν) objective, App. H, e.g. ν=1.83 in their Fig. 14). IMPLICATIONS: (1) symmetry class constrains the manifold/operators but does NOT pin a unique ν along the boundary; 'class DIII ⇒ ν=2 for all ζ' is NOT a theorem, only the Born/ζ=1 endpoint benchmark. (2) CORRECTION to the Cut A framing in the LMR paragraph above: 'confirm Ising ν_A≈1' both assumes the answer AND is supported only by the existing SMALL-ζ Cut A data (ν_A≈0.9 = the postselected/Ising regime where any crossover has NOT yet happened); the LMR-analogous crossover appears near ζ≈1 where our Cut A data is SPARSE/absent. We confirmed the uncontroversial regime, not the interesting one. (3) The project NLSM predicts Cut A self-dual Ising ν=1 for ALL ζ (no crossover); LMR's measurement-only analogue (closest map to w=0 Cut A) shows a crossover to 5/3. These CONFLICT — genuine open question: does Cut A's exact self-duality (λ_c=1/2 ∀ζ) PROTECT Ising across ζ, or does it cross over to monitored 5/3 like LMR? The Cut A campaign must target LARGE ζ (→1) to settle it, framed as H0(constant Ising ν=1) vs H2(crossover 1→5/3, possibly overshooting), NOT 'confirm ν=1'. (4) For Cut B the confidence set [1.5,3] (above) already spans 5/3, 2, and 2.3, so current data can neither REQUIRE nor EXCLUDE a ζ-dependent ν_B — ν(ζ) is untestable for Cut B at L≤160. ANALYSIS STRUCTURE (per the ν(ζ) memo): extract ν̂_LMR(ζ) per ζ with per-ζ calibration, build per-ζ confidence sets, test common overlap, fit H0/H1/H2/H3 to the calibrated likelihood, and use synthetic to check the design CAN distinguish them — it CANNOT for Cut B; for Cut A only if the campaign reaches large ζ with ≥4 sizes. The whole 'Cut B is class DIII ν≈2' headline is therefore a CONSISTENCY statement, and the live physics is the Cut A crossover test.

   **ASHKIN–TELLER/THIRRING mechanism for Cut B ν(ζ) — new analytic input 2026-06-17, with a CRUX vs the project's OWN archived chirality result [P].** A memo maps the strong-PPS (ζ→0, free-Majorana) corner of replicated Cut B to a non-Hermitian Ashkin–Teller/Thirring theory. Exact ket–bra decomposition: single-branch m_0~α(1-ζ) + cross-branch click g_0~αζ (exact; matches 'ζ multiplies only the cross vertex'). At the Ising point the connected click is S=g∫ε_+ε_- with ε_+ε_- = -J_R J_L (VERIFIED: ε=iξ_Rξ_L/flavour ⇒ ε_+ε_-=ξ_{R+}ξ_{R-}ξ_{L+}ξ_{L-}=-J_R J_L), the marginal current-current (Thirring/AT) coupling with continuously varying thermal dims x_±=K,1/K ⇒ ν_B(ζ)=1/(2-K(ζ)) [IF λ couples to the dim-K symmetric thermal M_+; if M_- the relation flips to 1/(2-1/K) and the 1→2 story FAILS — assignment NOT derived]. MOST USEFUL PIECE = the longitudinal/transverse distinction that DISSOLVES the project's long-running ν=1-vs-2 confusion: at ζ=0 the no-click tuning field is t~λ² (ξ_nc~λ^{-2}; the linear term is an imaginary O(κ) momentum shift, real localization only at O(κ²)) ⇒ ν_∥=2; at finite ζ the tuning field is linear in δλ ⇒ ν_⊥=1/(2-K). ν_∥(0)=2 and ν_⊥(0+)→1 are different scaling directions, NOT a contradiction. CRUX: this APPEARS to conflict with `theory/archive/qj_chiral_vertex_result.md`, which found the cross-replica vertex PURELY CHIRAL (exp(-4iφ_L); renormalizes velocity u_L not K ⇒ K=1 all orders ⇒ c_eff & ν CONSTANT). Likely RESOLUTION: the archive expands around the ζ=1 Born/Luttinger-liquid (Dirac) gapless theory; the memo around the ζ=0 free-Majorana (Ising) point — DIFFERENT gapless field content, so B_j's continuum image (chiral exp(iφ_L) in the Luttinger liquid vs non-chiral ε at the Ising point) can legitimately differ. If so they are COMPLEMENTARY corners, NOT a contradiction, and TOGETHER give the crossover: AT/Ising ν→1 at small ζ (memo) → chiral-stable DIII ν≈2 near Born (archive) → consistent with LMR's 1→overshoot→monitored. MUST BE CHECKED: is the cross-vertex chiral at the Ising corner too? then the AT mechanism is undercut. OTHER HONEST GAPS in the AT memo: K(ζ) is NOT computed (mechanism, not a quantitative prediction); the non-Hermiticity (imaginary Majorana mass; possibly complex K) is not reconciled with the Hermitian-AT relation ν=1/(2-K); the weak-g AT expansion is controlled only at small ζ (g~αζ/w is O(1) at Born), so ν→2 is a handoff to the DIII σ-model, not within AT. The memo also argues √ζ is an EFFECTIVE finite-window law (true √ζ needs the unnatural m_c(g)~g^{2/3}; marginal-relevant flow dg/dℓ=Ag² gives ξ~exp(c/αζ), an essential singularity = the already-[P] BKT mechanism). BEST FALSIFIABLE TEST (memo §11.6): measure an INDEPENDENT scaling dimension predicted to be K or 1/K and check K=2-1/ν — far stronger than another entanglement collapse. NET: a genuine analytic advance for the strong-PPS corner + the ν_∥/ν_⊥ clarification; the ν(ζ)=1/(2-K) prediction is mechanism-level and conditional; reconciling it with the archived chirality result is the next THEORY task.
2. Re-aggregate production keeping per-realisation arrays (I_r, S_r, ΔS_r, n_r, ΔΛ_r) for nonparametric/paired bootstrap. (Ladder aggregate already has them.)
3. τ_int pilot at ζ=0.5,1 and L=64/96/128: set T from T_burn≳5τ_int, T_meas≳20τ_int, and require ∂_λI (not just ⟨CMI⟩) to be T-stationary. Output the T(L) rule.
4. ζ=1 Cut B ladder FIRST (cloning-free: weights ≡1, no population needed, cheapest, cleanest anchor): L={32,48,64,80,96,128,160}, identical grid 0.45–0.55, Δλ 0.01→0.005, n_real 25–30 central / ~10 outer. Analyse with CMI collapse + max-slope + quotients (32,64),(48,96),(64,128),(80,160).
5. Then ζ∈{0.5,0.7,0.85} large-L at proper T, identical grids; model-compare constant-ν vs ν(ζ) via bootstrap-calibrated likelihood / held-out error (expected null: ν≈2 + finite-size drift).
6. Cut A curvature campaign (the CLEAN ν route, λ_c=1/2 exact): symmetric grid λ_A=1/2±{0,0.004,…,0.028}, ζ∈{0.1,0.3,0.5,0.7,1.0}, L={64,80,96,128,160}; estimator κ_L=−∂²_λ I|_{1/2} ~ L^{2/ν_A} (Richardson in δ); monitor the antisymmetric part I^(−)(δ), which must vanish by duality (pure systematic-error monitor).
Keep one N_c slope-calibration at ζ=1, L=128/160 (N_c vs 2N_c). Budget goes to n_real, not N_c. Defer L=192, small-ζ, geometry re-optimisation, and the score-function derivative estimator. Headline target: Cut B one exponent ν≈2 vs Cut A ν_A≈1. Realistic expectation: Cut A ν clean; Cut B "consistent with 2, wide bars."

**Scratch infra (will vanish on /tmp clear — promote if kept):** `/tmp/split_coupling.py`, `/tmp/pps_collapse{,2,3}.py`, `/tmp/pps_fig8.py`, `/tmp/pps_quotients.py`, `/tmp/pps_synth.py`, `/tmp/pps_calib.py`, `/tmp/pps_calib_quot.py`, `/tmp/pps_ladder_*.py`, `/tmp/pps_build_aggregates.py`, `/tmp/pps_agg_ladder.py`, `/tmp/pps_Tcheck.py`. Suggested home: `analysis/fss/` (collapse/quotients/calibration) and `analysis/var_reduction/split_coupling.py`.

### Methodology study — variance reduction for guided cloning (2026-06-16/17)

A self-contained five-round study on reducing cloning-estimator variance. Full
results + saved prototypes in `theory/VARIANCE_REDUCTION.md`. Bottom line for
any future chat:

- The guided proposal **c=zeta is practically optimal** (ESS/N_c ~ 0.97-0.99; a
  c-scan shows ESS peaks exactly at c=zeta; the cost-aware metric Var x wall also
  keeps c=zeta). The weight-degeneracy problem is solved. Do NOT invest further
  in the proposal, state-dependent/learned-Doob controls, adaptive resampling,
  online-c, or annealed-zeta. All tested and CLOSED (the residual per-window
  weight variance is within-window Poisson noise, unpredictable from any
  window-start Gaussian feature, R^2~0).
- Two estimation-side wins survive END-TO-END validation, both ~2-3x and free,
  on DISJOINT observables:
  * **coupled neighbouring lambda-points -> ~2x** on entanglement FSS
    differences (entropy 2.0x, <CMI> 1.76x, KMR <CMI><S> 1.98x). Robust at
    delta<=0.04, BREAKS at delta=0.06 (trajectory desync). Tightens slopes,
    crossings, nu. Next: split coupling + maximally coupled resampling + paired
    covariance/bootstrap. L-scaling beyond L=32 unverified.
  * **compensated-count martingale CV M=n-zeta*Delta_Lambda -> ~3x on the tilted
    activity <n>_Q only** (theory-side: K_eff, channel activities). Does NOTHING
    for entanglement (1.0x) or the SCGF (1.03x), via the SNIS cancellation.
- CORRECTION recorded in the doc: one-window/fixed-start tests OVERSTATED the CV
  (apparent 4.2x on E[G], ~400x on activity). The real end-to-end numbers are
  1.03x (SCGF) and 3.26x (activity). Always validate end-to-end.
- Observable choice (carries to the FSS pipeline): the production B_L was the
  trajectory product <CMI*S> (noisy). Prefer **<CMI> and the KMR product
  <CMI><S>** for crossings/slopes, since the coupling works on these and they
  are cleaner.

Methodology only. No physics claims changed, no jobs submitted, no theory files
touched.

### zeta=0 no-click anchor (Cut B) — CORRECTED 2026-06-15 (supersedes the 2026-06-10 block)

The 2026-06-10 [V] block (Fermi step at q=+-pi/2, lambda*=4/5, nu0~1,
"xi_ps ~ lambda^-2 REFUTED") is WITHDRAWN. All of it was an artifact of
analysis/anchor_scan.py's hardcoded kernel. Audit + a deterministic zeta=0 run
(2026-06-15) resolve the endpoint.

- anchor_scan.py is WRONG, do not trust it. Its E_analytic(q,w,kappa) =
  sqrt(w^2 - kappa^2 - 2j*kappa*w*cos q) (kappa=lambda/4, lambda*=4/5) drops the
  hopping w from the measured bond. Its own docstring calls it "the SSH-anchor
  self-consistency check" and delta_B_hook() raises NotImplementedError, so its
  Fermi-step / lambda* / nu0 output is the wrong symbol confirming itself, never
  tested against the real model. [V, read 2026-06-15]
- The backend is CORRECT. gaussian_backend.effective_generator adds -1j*alpha to
  h_eff[a,b] on the SAME bond (a,b)=bond_jump_pair(bond) that carries the hopping w,
  so h_eff[measured bond] = w - i*alpha (both couplings, anti-Hermitian orthogonal
  to the real hopping). The cloning DATA (lambda_c(zeta), Delta_B, Delta_cross)
  therefore STAND. delta_B_zeta0.py is correct (uses the backend) and was simply
  never run. [V]
- Corrected no-click physics [V, audit + run]: no Fermi step; area-law for every
  lambda>0; critical ONLY at lambda=0; lambda_c(0)=0 reached CONTINUOUSLY
  (lim_{zeta->0+} lambda_c = 0). Correct kernel E^2(q)=4w(w+i*kappa)cos^2(q/2)-kappa^2,
  kappa=alpha/2. Modulus dimerization delta=sqrt(w^2+kappa^2)-w ~ kappa^2/2w ~ lambda^2
  (second order). Band-structure xi_nc = 2/ln(1+kappa^2/w^2) ~ lambda^-2. So
  xi_nc ~ lambda^-2 is CONFIRMED for the no-click state; the 2026-06-10 "refutation"
  was the artifact.
- Numerical confirmation (deterministic, real backend, 2026-06-15): half-chain
  entropy S(L/2) SATURATES (area-law) at lambda=0.3,0.5,0.6 for L=16->128, all below
  the spurious lambda*~0.8; only lambda=0.1 still grows (xi_nc > L). Decisive,
  fit-free.
- OPEN [O]: the MEASURED steady-state xi_nc exponent over the accessible window
  lambda in [0.2,0.5] is flatter (~lambda^-1.5) than the band-structure lambda^-2
  — crossover, since the asymptotic small-lambda regime needs L >> xi_nc > 256.
  Pin it with analysis/exponent_noclick.py + run_exponent_noclick.sh (eigenvector
  steady state, validated to 1e-15 vs the orbital loop). RUN ON HABROK.
- Field theory unchanged where it was right [V]: zeta-vertex marginal (eps+ eps-,
  Delta=2); dimension-1 cross-bilinear parity-forbidden; sqrt(zeta) EMPIRICAL only
  (both the patch-counting and the y_lambda/y_zeta derivations are dead);
  small-zeta exponent OPEN (one-loop flow of the marginal vertex; random-bond-Ising
  analogy hints marginal irrelevance / logs, but the sign depends on R->1 vs R->0).

Thesis [V]: chap:fieldtheory sec:ft-boundary (all three subsections), sec:ft-summary,
chap:results sec:results-cutB, and chap:intro sec:intro-thiswork rewritten 2026-06-15
to this picture (paste-ready LaTeX delivered to chat; apply to m1thesislatex).
sec:ft-classes ("combined hopping and measurement coupling") was right all along;
sec:ft-noclick-spectrum now agrees with it.

> **⛔ SUPERSEDED 2026-06-15 — see the "zeta=0 no-click anchor (Cut B) — CORRECTED 2026-06-15" block at the top of this file.** The no-click claims in this 2026-06-10 block are WRONG. There is NO Fermi step, NO lambda*, NO extended critical interval. anchor_scan.py's kernel (t1 = -i*kappa) drops the hopping w from the measured bond; this was audited and falsified by a deterministic zeta=0 run (2026-06-15). The real no-click state is area-law for every lambda>0, critical only at lambda=0, xi_nc ~ lambda^-2, with lambda_c(0)=0 reached continuously. SURVIVING from this block: the marginal cross-vertex (Delta=2), the parity-forbidden dimension-1 bilinear, and sqrt(zeta) as an EMPIRICAL law. WITHDRAWN: Fermi step at q=+-pi/2, lambda*=4/5, nu0=1, "xi_ps~lambda^-2 refuted", and the corner-matching phi=1/2 derivation.
>
> **★ 2026-06-10 SESSION — BOUNDARY DERIVED FROM THE ζ=0 ANCHOR (φ=1/2 [P], CONDITIONAL); ξ_ps~λ⁻² REFUTED; QSD/QJ DICHOTOMY RETRACTED; CASE-A ISING RELOCATED.**
>
> **GATING NUMERICAL TESTS (these BLOCK confirmation; run before writing exponents into the thesis):**
> 1. **ζ=0 anchor scan** — PARTIALLY CLEARED 2026-06-10 (single-particle band level);
>    Δ_B+reduction SCRIPT NOW WRITTEN (not yet run).
>    Analytical SSH (E²=w²−κ²−2iκw·cos q, w=1−λ, κ=α/4): Fermi step pinned at q=±π/2 for all
>    λ<λ*=4/5 [V]; state ξ short (~1/ln(4/λ) small λ), diverging ONLY at the EP ⟹ ξ_ps~λ⁻²
>    REFUTED numerically [V]; ν₀=0.98≈1 [V]; λ*=4/5 confirmed in code units. (a)+(b) now
>    instrumented by `analysis/delta_B_zeta0.py`: builds the REAL no-click steady-state
>    Majorana covariance, computes the connected single-state bond correlator
>    cq(r)=Γ[2x,2y+3]Γ[2x+3,2y]−Γ[2x,2y]Γ[2x+3,2y+3] (= worker_opdim's cq; Wick-derived,
>    matches verbatim), fits Δ_B on EVEN r (expect ≈1, ties to measured 1.009), odd-r null
>    as the decoupling/reduction check. Deterministic, O(L³), runs on Mac/Habrok in
>    seconds — NO cluster. RUN IT to close gate 1. φ=1/2 not fully gated until Δ_B≈1 lands.
>    Scripts: analysis/anchor_scan.py (band level), analysis/delta_B_zeta0.py (Δ_B+reduction).
> 2. **Area-phase ξ(ζ,λ)** just above λ_c: ξ∝ζ^{−1/2} and λ-flat (saturated-defect window law,
>    φ=1/2) vs ξ∝ζ^{−1} (coherent channel, φ=1) vs essential form (marginal asymptote).
>    Blocks φ=1/2 and fixes the small-ζ asymptote. [Companion gate: Case-A Born-line ν via
>    dB_L/dλ at exactly λ=1/2 → blocks the SU(2)₁ [P] assignment.]
>    WORKER READY 2026-06-10 (not yet run): `worker_areaphase_pps.py` +
>    `analysis/fit_areaphase.py` + `slurm/submit_areaphase.sh`. Cloning at ζ<1 →
>    clone-population C_sc(r)=Cov(b_x,b_{x+r}), b[x]=Γ[2x,2x+3]; ξ from exp-fit on EVEN r
>    (odd-r null built in); auto-places λ=λ_c(ζ)+offset in the area phase; 30-task grid
>    (2 L × 5 ζ × 3 offsets). Discriminator validated on synthetic (p=0.51 vs 1.01).
>    **MUST run an N_c=500 rung vs 250 first** — clone-population Cov carries genealogical
>    bias; check ξ is N_c-stable before banking φ.
>
> Results (chains in Y_ZETA §12 + chat log):
> 1. **ζ=0 Case-B anchor SOLVED [V].** Per decoupled chain: non-Hermitian SSH, t₂=w real,
>    t₁=−iκ, κ=α/4; E²(q)=w²−κ²−2iκw·cos q. Lifetime zeros pinned at q=±π/2 for κ<w →
>    band-selected Fermi-step steady state: CRITICAL for all 0<λ<λ*. Reproduces measured
>    Δ_B=1.009 and Δ_cross=2.02 exactly. Lengths: state ξ~1/ln(4/λ) (small λ, short!);
>    ℓ_λ=4w/λ = the previously "verified ξ_nc~λ⁻¹", now identified as the SELECTION length
>    (a formation scale, not a state correlation length); EP ν₀=1. ⟹ **ξ_ps~λ⁻² refuted**;
>    the old ζξ~1 matching with the true λ⁻¹ gives φ=1 — the old derivation fails both ways.
> 2. **Boundary [P, conditional on gate 2].** Clicks = projective O(1) defects, density
>    ρ=ζλn̄. Coherent channel = redundant κ_eff shift [V]; Δ=1 cross-bilinears parity-
>    forbidden [V]; stochastic residue EXACTLY MARGINAL: damage D(r)≈16πn̄ζ·ln r [V derived].
>    Effective coupling = clicks per slow-cone formation cell ≈ 4n̄ζ ⇒ the measured window
>    ζ∈[0.02,1] is STRONG coupling. Window law: one-hit-per-cell ξ_×=(4n̄ζ)^{−1/2},
>    λ-independent; matching ℓ_λ=ξ_× ⟹ **λ_c=A√ζ, φ=1/2 — a CORNER-MATCHING exponent of a
>    doubly singular endpoint, NOT y_λ/y_ζ of any fixed point.** Explains the r_c-exponent
>    mismatch and the five-form degeneracy. h_d not derivable at weak coupling (h_d^pert=2);
>    strict ζ→0 asymptote [O]. NOTE: the 9σ "φ=1 excluded" does NOT adjudicate h_d
>    (linear+intercept fits the boundary).
> 3. **Unraveling [V structure].** QSD's Gaussian record ⟹ weight-≤2 replica vertex (Σ_μM^μ)²;
>    QJ's point-process record ⟹ the 2R-fold ζΠ_μn^μ; identical at R=1 (same Lindbladian).
>    Both ζ-vertices are Δ=2 at Majorana anchors (pair = 2Δ_ε additivity). "QSD relevant vs
>    QJ marginal" RETRACTED — LMR's ζ*≈0.28 is an R=2 finite-coupling (AT-line drift, rate
>    ∝(2n−2)) feature that dies at n→1. No interior ζ* for either unraveling at n→1 [P].
> 4. **Case A.** ζ=0 endpoint solved [V]: imaginary-time projection onto the uniform zigzag
>    Majorana chain ground state at λ=1/2 ⟹ ISING (c=1/2; ν₀=1 from dimerization ∝(λ−½)) —
>    the Ising tag belongs HERE. Born line ζ∈(0,1]: class-D coset; R=2 anchor = S² at θ=π →
>    SU(2)₁ (c=1, ν=2/3) [P]; n→1 values [O]. ζ marginal on the pinned line, no interior ζ*;
>    ζ→0 crossovers are POWER-LAW ξ_×~ζ^{−1/2}, plateau edge ζ_×(L)∝L^{−2} (supersedes the
>    earlier 1/ln-form).
> 5. **opdim pre-run fixes (load-bearing):** see ⚠️ note at the y_ζ-measurement block below.

> **★ 2026-06-07 SESSION (cont. 2) — N_c-LADDER DATA LANDED + ANALYZED; LEVER CHECKS.**
> The {250,500,800} N_c-ladder (399 tasks) finished on Habrok, aggregated per rung
> (`ladder_nc{250,500,800}.pkl`), analyzed on Mac. Outcome CONFIRMS the cont.-1 block below
> on independent **debiased-L=128** data (not just v2/dense cross-validation):
>
> 1. **λ_c(ζ) on the (32,64,128) triple.** `extrapolate_nc.py` per-point 1/N_c → B_∞ at L=128
>    (resid_frac median 0.087, max 0.42 — debias clean for the typical point, SOFT at
>    ζ∈{0.22,0.30} where ESS-collapse curvature dominates → treat those two λ_c as low-weight).
>    Clean (32,64) crossings over ζ∈[0.02,0.5]: **λ_c = 0.501·√ζ** (φ=0.523±0.019, R²=0.986);
>    debiased (64,128) crossings at the 7 ladder ζ agree (slightly lower = finite-L drift).
>    Reproduces cont.-1's λ_c-A≈0.51 ⇒ **the 0.96 is the r_c prefactor, not λ_c** is now
>    confirmed on the debiased set. λ_c=0.5√ζ hits Carollo (0.5·1) with NO Möbius needed.
> 2. **√ζ-on-λ_c is partly a saturation artifact — physical exponent stays on r_c.** Panel D of
>    `~/Downloads/sqrt_zeta_confirmation.png`: r_c=λ_c/(1−λ_c) vs √ζ has slope ≈0.70 and CURVES
>    up — consistent with cont.-1's r_c-φ≈0.7–0.85, NEITHER √ζ nor linear. λ_c≈0.5√ζ is the
>    small-ζ (no-click-anchored) description; do not report it as the global single-exponent law.
> 3. **ν NOT measured.** Free-(λ_c,ν) collapse degenerate for ζ≥0.18 (λ_c pinned 0.736, ν at the
>    1.0 floor, quality 0) — narrow 13-pt L=128 window doesn't overlap the broad dense grids under
>    the scaling transform. Crossings (ν-free) are the only trustworthy estimator. Matches the
>    standing "ν is not a clean deliverable".
> 4. **Rényi washout (`renyi_washout.py`, clean L=128): INCONCLUSIVE.** a_1/a_2 (CFT no-washout
>    = 4/3) drifts 1.67 (small ζ) → 1.28 at ζ=1; the Born-corner 1.28 is a WEAK sub-CFT hint
>    (Poboiko–Mirlin) within finite-size error. Ladder ζ∈{0.18–0.30} rows corrupted (L=128 only at
>    narrow λ ⇒ "max a_n over λ" misses the deep log phase). Decisive washout needs a dedicated
>    broad-λ, multi-cut run at L=128/160. Not a thesis result as-is. Plot: `/tmp/ladder_washout/`.
> 5. **Lever checks (job 29352854).** T-lever REAL but shrinking: B_L saturates by t≈42 (ζ=0.15) /
>    t≈67 (ζ=0.5) at L=128 vs T=100 → ~1.5–2.4× on steps, smaller at larger L/ζ; also confirms
>    **T=100 was adequate** (ladder not under-equilibrated, so the √ζ result is not a short-T
>    artifact). BLAS threads: 1.6× WALL at L=128 (8 threads) at the cost of concurrency = walltime-
>    cap tool only, no core-h saving. **dtau 3× NOT certified**: SAFE/biased/SAFE/SAFE across
>    mult=1.0/1.5/2.0/3.0 is noise (R=8 too few; ζ=0.3 B_L≈0.04 → huge relative scatter); a faint
>    −6% B_L drift by 3× may be real. Reopens vs "dead" but needs an R-converged re-test at a
>    production L before banking even 2×.
>
> Artifacts: `ladder_nc{250,500,800}.pkl`, `ladder_fss_ready.pkl` (FSS-ready, L=32/64/128 with
> per-point resid_frac on the L=128 recs), `~/Downloads/sqrt_zeta_confirmation.png`,
> `/tmp/ladder_washout/`. Chain: `aggregate_ladder.py` (on Habrok) → `extrapolate_nc.py` → by_zL
> crossings + `scaling_form.best_collapse_z/fit_forms` → plots; `renyi_washout.py`. **STILL OPEN:
> `Y_ZETA_DERIVATION.md` §7/§11 "0.96 tell" wording needs the r_c-vs-λ_c fix (flagged cont.-1 #2).**


> **★ 2026-06-07 SESSION — y_ζ MEASUREMENT PIPELINE + LOAD-BEARING SCALING-VARIABLE CORRECTION.**
>
> **Canonical derivation doc for the y_ζ question is now `theory/Y_ZETA_DERIVATION.md`**
> (model → recycling expansion → operator dims → y_ζ=2−Δ_B → Foster/Jian class → boundary
> → the run). Read it together with §D7–D10 of `OPEN_ANALYTIC_PROBLEMS.md`.
>
> **THE CORRECTION (after external review; Y_ZETA_DERIVATION §7/§9/§11).** The boundary law
> had been written λ_c(ζ) ~ ζ^φ. That is the WRONG scaling variable for a perturbation
> around the Born corner ζ=1. The PPS field is h ∝ (ζ−1), which vanishes at ζ=1, so the
> correct LOCAL law is **λ_c(1) − λ_c(ζ) ~ (1−ζ)^{y_λ/y_ζ}**. Consequences:
> the ζ=1 operator-dimension measurement fixes the **Born-corner** y_ζ ONLY, not the global
> small-ζ boundary; and the old "Δ_B(λ_c)≈1.1−1.4" is **demoted to a conjecture** (it was
> inverted from the global φ, the wrong corner). Internal tell that this is right: our own
> global fit λ_c≈0.96√ζ does NOT pass through λ_c(1)≈0.5.
>
> **THREE-REGIME PICTURE** (do not collapse to one φ): Born corner (ζ=1, Δ_B^IR, the run
> measures it) / no-click endpoint (ζ→0, different fixed point, Δ_B≈1 measured there) /
> intermediate ζ∈[0.1,0.7] crossover (effective φ≈0.56, neither).
>
> **NEW CODE (committed + pushed): the y_ζ measurement.** At ζ=1 (Born, NO cloning) sample
> QJ trajectories; from each final covariance Γ record b_x=Γ_{2x,2x+3} and form the
> trajectory covariance C_sc(r)=Cov_traj(⟨B_x⟩,⟨B_{x+r}⟩) ~ r^{−2Δ_B} → Δ_B(λ_c) → y_ζ.
> Also g(r)→X₁,X_typ,x² (Case-B class vs Jian Table I), cq(r) (the C₁/same-contour check),
> S(L/2)→c_ent. Files: `pps_qj/parallel/worker_opdim_pps.py`, `slurm/submit_opdim.sh`,
> `analysis/fit_opdim.py`. C_sc is valid IFF cq is subleading (built-in check); χ_B
> (PPS linear response, needs ζ<1) is a costlier flagged follow-up.
>
> **⚠️ PRE-RUN FIXES (2026-06-10, LOAD-BEARING — implement in `fit_opdim.py`
> before the run):** (1) fit C_sc(r) on EVEN r only — odd-r values are ≡0 by
> the exact two-chain decoupling (use them as a free null test); (2) restrict
> G(r), X_typ, x² to intrachain Majorana pairs (r ≡ 0 mod 4 safest) —
> interchain entries are exact zeros and poison ⟨log G⟩; (3) divide the
> measured S(L/2) log-slope by 2 before comparing c_ent to Jian's 0.39 (two
> identical decoupled chains add).
>
> **IMMEDIATE NEXT TASK (forward logic):** run opdim (calibration
> `PPS_L_LIST=128 PPS_LAM_LIST=0.50 PPS_N_TRAJ=64 ARRAY=0-0 WALL=00:20:00 CPUS=16 bash
> slurm/submit_opdim.sh`, then production `CPUS=24 bash slurm/submit_opdim.sh`, analyse
> `python analysis/fit_opdim.py /scratch/$USER/pps_qj/pps_opdim`) → get Δ_B(λ_c(1)) →
> y_ζ^Born=2−Δ_B → PREDICT λ_c(1)−λ_c(ζ)~(1−ζ)^{1/(2y_ζ^Born)} → TEST by fitting the
> boundary at ζ≳0.7 (NOT the extract_yzeta ζ→0 collapse). Whether the B_L grid is dense
> enough at ζ∈{0.7,0.8,0.9,1.0} for that fit is itself open.
>
> **Class anchor (D7–D9):** Born=n→1, forced=n→0 are *different* classes (Jian); along the
> PPS line ν≈2.1 is fixed (n→1 throughout, QJ marginal cross-vertex ⇒ no ζ*); y_λ≈1/2 is
> Foster–Guo–Jian–Ludwig's R=2−ε expansion CALIBRATED to Jian's ν (numerically anchored,
> not derived). Foster's setup is a deterministic-Hamiltonian monitored Kitaev SC ⇒ nearly
> removes the random-vs-deterministic caveat for Case B.
>
> **Numerics audit (D10):** final covariances are NOT persisted by the production worker;
> `corr_decay` is the wrong object (single-particle ⟨c†c⟩, drops pairing, abs-averaged) for
> the Jian/Foster discriminators — hence the dedicated opdim worker.


> **★ 2026-06-07 SESSION (cont.) — OLD-AGGREGATE CROSS-VALIDATION + FORM DEGENERACY + SLEVIN–OHTSUKI COLLAPSE.**
> Combined analysis of `~/Downloads/clone_aggregate(1).pkl` (v2, 1920 entries, L≤128 incl. its
> own N_c=100 L=128 curves) with the dense + rescue sets. Container-side only (uploaded pkls);
> no new repo code. Six findings, all carry to the thesis:
>
> 1. **λ_c(ζ) cross-validated across independent datasets.** v2 (separate λ/ζ grid, own N_c
>    ladder) reproduces the dense wide-pair crossings to rms 0.025, systematically +0.019
>    HIGHER (lower N_c ⇒ upward crossing bias). The λ_c(ζ) shape is robust; truth sits at or
>    slightly below the dense values.
> 2. **The "0.96√ζ" is the r_c prefactor, not λ_c (doc fix needed).** λ_c=A√ζ gives A≈0.51 on
>    BOTH dense (χ²/dof 0.76) and v2 (0.53); r_c=λ_c/(1−λ_c)=A√ζ gives A≈0.78–0.90. The §7
>    "internal tell" in `Y_ZETA_DERIVATION.md` ("global fit λ_c≈0.96√ζ doesn't pass through
>    0.5") conflates r_c(1) with λ_c(1): the λ_c fit (≈0.5√ζ) DOES pass through λ_c(1)≈0.5.
>    Fix the §7 wording. The Born-corner reframing conclusion is unaffected (it stands on the
>    derivation-invalidity + the r_c exponent, not on that tell).
> 3. **Form degeneracy — the concrete reason boundary-shape fitting cannot decide φ.** On λ_c,
>    FIVE forms fit at χ²/dof < 0.6 and are statistically indistinguishable: a√ζ (0.53), free
>    power φ=0.55 (0.35), √ζ-Möbius-2p (0.29), log-corrected a√ζ(1+c·lnζ) (0.35),
>    linear+intercept (0.55). On the UNBOUNDED r_c, a√ζ FAILS (χ²/dof 4.3) and free power
>    gives φ≈0.65–0.81. Confirms quantitatively that λ_c-φ≈0.5 is a saturation artifact and
>    the physical r_c exponent is ~0.7–0.85 (neither √ζ nor linear). One-param √ζ-Möbius is the
>    only clear loser (forced to undershoot the corner, χ²/dof 4.7).
> 4. **B_L finite-N_c bias (304 matched v2/dense pairs).** Median fractional B_L bias ~1.8% at
>    N_c=250, growing with L (~11% at L=96) and near criticality, BUT it largely CANCELS in
>    crossings (propagated λ_c shift only ~0.02). **L=128 cannot be debiased from
>    v2(N_c=100)+rescue(N_c=250)** — the 2-point 1/N_c extrapolation is noise-dominated (the
>    two shared λ give −8% vs +82%). The N_c-ladder {250,500,800} is REQUIRED for the L=128
>    debias; this confirms (does not replace) its necessity.
> 5. **Born-corner boundary fit is unreliable, as Y_ZETA §9 anticipated.** λ_c(1)−λ_c(ζ) ~
>    (1−ζ)^{φ_B} gives φ_B≈1.7 (R²=0.996, ζ∈[0.5,1)), but this is the thin, non-monotonic
>    near-ζ=1 region (λ_c(0.92)=0.514 > λ_c(1)=0.505). NOT a φ_B measurement — it only shows
>    the global √ζ (which predicts φ_B=1 by Taylor expansion) misses the corner. The Δ_B
>    opdim run stays the right tool for the Born-corner exponent.
> 6. **Slevin–Ohtsuki cost-function collapse — implemented, does NOT beat wide-pair crossings.**
>    Form B_L = F₀(x) + L^{−ω}F₁(x), x=(λ−λ_c)L^{1/ν}. **B_L is NOT scale-invariant at
>    criticality** (crossing height drifts with L: 0.82–1.66 across pairs at ζ=0.5), so the
>    single-variable collapse is structurally strained. With L≥16 + 5% error floor: SO-λ_c
>    AGREES with crossings at small ζ (0.10–0.30, within 0.01) and returns ν~1.1–1.4, ω~2;
>    DIVERGES at ζ∈[0.4,0.65] (off by 0.05–0.10) where L-coverage thins and the fit finds
>    spurious small-ω minima. ν scattered 1.1–2.9, NO clean plateau. Form fits to SO-λ_c
>    reproduce λ_c-φ≈0.52, r_c-φ≈0.80. VERDICT: wide-pair crossing median + drift errors
>    remains the primary λ_c estimator; SO becomes viable only with ≥4–5 clean sizes
>    (ladder-debiased L=128 + an L=160 point). For the eventual redo, adopt Slevin–Ohtsuki
>    cost-function FSS with a correction-to-scaling term — standard in the monitored-fermion
>    literature (arXiv:2503.23807 cost-function + error-from-2×min; arXiv:2509.09538 notes
>    visual crossing-ID is insufficient given free-fermion finite-size corrections).

**Last major update: 2026-07-07** — Ruche migration + speedups + cost/r calibration (see top banner). Prior major update 2026-06-06 (theory: replica field-theory routes recorded
in `OPEN_ANALYTIC_PROBLEMS.md` §D — convergent n→1 non-perturbative obstruction;
the boundary universality was revised again after external review (see §D6): NO
closed form; a **finite-ν conventional-type** transition (ν~2; KT / essential
singularity NOT established — the intermediate "KT at ζ=1" claim is WITHDRAWN);
ζ=1 is special only because the relevant single-copy mass vanishes; ζ→0 is an
order-of-limits question. Also: the "Carollo PRA 98 010103" cite for λ_c(1)=1/2 is
MIS-ATTRIBUTED (it is a quantum-Doob paper) — see §D6 and the reference table. The LMR-ζ* open item
below is updated accordingly.
2026-06-05: N_c-ladder campaign BUILT + LAUNCHED for the
decisive small-ζ λ_c — supersedes the plain L=128 rescue for that purpose; see
"N_c-ladder campaign (2026-06-05)" below. Prior 2026-06-04 replica-limit
reframing and 2026-06-03 dense/rescue status below unchanged.)
This document is the canonical entry point.
For deeper theoretical detail see `theory/SUMMARY_2026_05_22.md` and
`theory/qj_pps_theory_summary.md`. **For the chat-agent protocol (start-of-chat
read + handoff-update workflow), see `theory/AGENTS.md`.**

> **⚠️ THEORY STATUS REVISED (2026-06-03; SUPERSEDED 2026-06-10 — see ★ top
> block).** The √ζ derivation in the TL;DR below (Δ_ζ=1, ξ~λ⁻²) is **invalid**,
> and as of 2026-06-10 the ζ=0 anchor is solved exactly: **ξ~λ⁻² is refuted
> outright** (the only diverging ξ on the ζ=0 line is at the EP, ν₀=1), and the
> previously "verified ξ_nc~λ⁻¹" is the **SELECTION length ℓ_λ=4w/λ** — a
> formation scale, not a state correlation length. TERMINOLOGY (do not conflate
> the two measured ζ-channel dimensions under one "Δ_ζ"): the **NORMAL**
> (boundary-moving) component is the single-copy mass, Δ_B≈1.009 — relevant by
> dimension but manifold-redundant at ζ=0 (coherent κ_eff shift); the
> **TANGENTIAL** component is the cross vertex, Δ≈2.02 — exactly marginal. The
> boundary is now DERIVED at [P] as a corner-matching law λ_c=A√ζ (strong-defect
> window law; CONDITIONAL on the area-phase ξ gate — top block). Free fits
> (φ≈0.56 on λ_c, ≈0.8 on r_c) are crossover-dressed effective slopes,
> consistent with corner matching. `theory/CURRENT_THEORY_STATUS.md` and
> `theory/NUMERICS_STATUS_AND_PLAN.md` are NOT yet reconciled to 2026-06-10
> (open item); `OPEN_ANALYTIC_PROBLEMS.md` §D8 is patched. The √ζ material
> below is retained as historical context only.

> **⚠️ REPLICA-LIMIT REFRAMING (2026-06-04).** The class-DIII ν≈2 the project
> relied on was mis-cited as "König-Brouwer 2014" (non-existent paper). Real
> source: **Fulga et al. PRB 86, 054505 (2012)** — and it is the **n→0
> (forced/Anderson)** exponent. Jian-Shapourian-Bauer-Ludwig (arXiv:2302.09094)
> prove that for class DIII the **forced (n→0)** and **Born-rule (n→1)**
> measurement transitions are *different universality classes*. The MIPT
> boundary is governed by the **n→1** end, so importing Fulga's n→0 ν=2 is not
> justified; ζ interpolates between the limits (ζ→0 forced/n→0, ζ=1 Born/n→1),
> so a *constant* ν is not expected (cf. LMR's ν∈[1,5/3]). Also flag: the
> monitored-Majorana literature (Jian; Fava et al. PRX 13 041045) uses the
> **principal-chiral SO(N)** target, NOT the SO(2n)/U(n) coset the Doc
> derivations assumed — the target for THIS model must be rederived from the
> Choi action, not inherited from the Anderson problem.

---

## TL;DR — where the project stands (May 2026)

### Theory (SUPERSEDED — see CURRENT_THEORY_STATUS.md)

The matched-NLSM framework for QJ-PPS Case B (single d-mode measurement +
Kitaev hopping + PPS parameter ζ) predicts
$$
\lambda_c(\zeta) \;\sim\; C\sqrt{\zeta}, \qquad C = O(1) \text{ non-universal}
$$
from two ingredients:

1. ~~UV dimension Δ_ζ=1~~ **CORRECTED**: the genuine normal-ordered cross
   vertex :B₊B₋: has Δ≈2 (marginal), verified exactly
   (`analysis/cross_vertex_dimension.py`). The earlier "Δ=1" measured a single
   bilinear / the raw correlator, which is ~r⁻² because ⟨B⟩≠0.
2. ~~ν=2 used as input~~ **OPEN**: the many-body ν is unresolved; the
   single-particle no-click length gives ξ_nc~λ⁻¹ (ν=1 proxy), not λ⁻².

Matching at the multicritical crossover scale $\xi_\lambda^{\rm cross}\sim\lambda^{-2}$
gives $\zeta\lambda^{-2}\sim K^* \Rightarrow \lambda_c\sim\sqrt\zeta$.

The $\lambda^{-2}$ scale is the universal class-DIII multicritical correlation
length, *not* a single-particle no-click localization length. The actual QJ
distance-3 Majorana bond gives a gapless H_eff at the single-particle
level (verified numerically) — this is a structural difference from KMR/LMR's
QSD setup, where the no-click problem does have a well-defined BdG localization
length.

### Numerics (best current estimate)

> NOTE (2026-06-17): the ν statement here is QUALIFIED by the top banner. The global FSS collapse cannot resolve ν at this L-range (synthetic calibration: recovered ν̂ ≈1.2–1.7 regardless of true ν), and B_L biases ν low; use ⟨CMI⟩ with quotients / Cut-A curvature. The φ and λ_c results (√ζ, Born endpoint) still stand for the clean ζ≥0.25 window.

Global FSS on merged cloning data ($L \le 256$, $\zeta \in [0.02, 1.00]$):
- $\phi = 0.56 \pm 0.05$ on $\zeta \in [0.03, 0.85]$
- Consistent with predicted $\phi = 1/2$ at $1.3\sigma$
- Excludes $\phi = 1$ at $9\sigma$
- Empirical prefactor $C \approx 0.91 \pm 0.10$
- $\nu(\zeta)$ scattered around $\sim 2$ across $\zeta \in [0.05, 0.7]$,
  consistent with the theory-predicted plateau

### Dense fine-grid campaign — actual status (June 2026)

The dense campaign (`pps_clone_dense`, 4112 tasks across three SLURM
scripts) was submitted; partial outcomes from the running and finished
jobs:

| Script | L | Tasks | Done | Status |
|---|---|---|---|---|
| `submit_clone_dense_small_L.sh` | 8,16,24,32 | 0–2055 | **2056/2056** | Complete |
| `submit_clone_dense_medium_L.sh` | 48,64 | 2056–3083 | **800/1028** | Walltime hit; 228 L=64 tasks (IDs 2856–3083) missing |
| `submit_clone_dense_large_L.sh` | 96,128 | 3084–4111 | **L=96: 342/514, L=128: 0/514** | 120h walltime exhausted; L=128 never started |

Worker writes the full per-clone observable set (B_L, full CMI tripartition
components $S_{AB}, S_{BC}, S_B, S_{ABC}$, and Rényi-2/3 — verified
populated, no NaNs from `PPS_RECORD_RENYI=1`).

**Partial aggregate**: `clone_aggregate_dense_partial.pkl` (3198 entries),
covers all 21 ζ for L=8..48 and partial coverage at L=64, 96. Sufficient
for clean Binder crossings at (16,32), (32,64), (48,96) pairs across the
moderate-ζ band. Used to design the rescue resubmission below.

### Rescue resubmission (June 2026)

Three new SLURM scripts written this iteration, addressing the campaign
shortfall:

| Script | L | N_c | Tasks | Walltime | Purpose |
|---|---|---|---|---|---|
| `submit_clone_dense_L64_backfill.sh` | 64 | 400 | 228 (IDs 2856–3083) | 24h | Complete the L=64 row in `pps_clone_dense` |
| `submit_clone_rescue_L128.sh` | 128 | 250 | 130 (IDs 0–129) | 96h | **Priority**: the critical missing FSS size, writes to `pps_clone_rescue` |
| `submit_clone_rescue_L160.sh` | 160 | 120 | 130 (IDs 130–259) | 120h | Optional: 4th FSS point for ω-fit |

The rescue grid (`make_clone_rescue_grid()` in `grid_pps.py`) uses:
- 10 ζ values (decisive window + anchors): {0.10, 0.15, 0.22, 0.25, 0.30,
  0.40, 0.50, 0.65, 0.80, 1.00} — all already present in the dense ζ-set so
  (64,128) and (96,128) crossings are directly computable
- Narrow λ windows (13 points, ±0.07) centered on **measured** dense crossings
  (see "Measured crossings" below) — NOT on √ζ-fit placeholders
- Reduced N_c (250 at L=128, 120 at L=160): pragmatic ~20% B_L error target
- Output dir `pps_clone_rescue` (separate from dense; no collision)
- Seeds offset by +12e9, disjoint from all prior campaigns

Shim worker: `worker_clone_rescue_pps.py`. T held at 100 for L≥128
(saturation argument: ballistic spreading needs T ≳ L/v; cutting T at the
key size would gamble on the most expensive run).

### N_c-ladder campaign (2026-06-05, COMPLETE — analyzed 2026-06-07, see cont.-2 block at top) — the decisive small-ζ run

Supersedes the plain L=128 rescue for the decisive small-ζ λ_c. Built and
launched this session; jobs running on Habrok. Fixes the two limits the
dense/rescue data could not: the ~45% finite-N_c B_L bias at L=128, and the
per-point variance (ESS collapse). Design:

- L=128 only; pairs against the existing CLEAN dense L=32,64 (same ζ) → the
  (32,64,128) FSS triple. 7 ζ in the discriminating window
  {0.08,0.10,0.15,0.18,0.22,0.25,0.30}; 13-pt λ windows (±0.08) on measured crossings.
- N_c LADDER {250,500,800}: full grid at 500, central-3-λ calibration subsets
  at 250 and 800 → per-point 1/N_c extrapolation to B_∞ (removes the bias).
- Seed BLOCKS: 3×5 = 15 seeds/point — variance beaten by seeds, NOT N_c (ESS
  collapse makes N_c-only hopeless; would need ~5000).
- Records full observable set (CMI comps + Rényi-2/3 + corr) so the cleaner
  estimators (Rényi-2 crossing, bipartite MI) and the washout test reuse it.
- 2-DAY FEASIBILITY: a task runs its 5 realisations on 5 cores in parallel, so
  wall = ONE realisation (~13/26/42h at N_c 250/500/800, all <48h). Wall is set
  by the top N_c rung, not by #seeds; ~53k core-h → ~10-12 nodes for <2 days.
- Each N_c rung writes its OWN dir (the aggregator keys by (L,λ,ζ) and would
  merge rungs): `pps_clone_ladder_nc{250,500,800}`.

Analysis chain (staged, runs when data lands): `aggregate_ladder.py` (pools the
seed-blocks — the stock aggregator overwrites duplicate keys) →
`extrapolate_nc.py` (per-point 1/N_c → B_∞; prints `resid_frac` = whether the
bias is clean 1/N_c; merges with clean L=32,64 → FSS-ready pkl) →
`scaling_form.py` (free-ν collapse, √ζ vs linear) + `renyi_washout.py`
(B4 item 3 at L=128).

HONEST CAVEATS (carry to the thesis):
- Solid deliverable is **λ_c(ζ) and the √ζ-vs-linear discrimination**. φ is
  softer (may not be a clean power law if BKT). **ν(ζ) is NOT a clean
  deliverable**: 3 sizes weakly constrain ν in the collapse, the 1/√L λ_c method
  partly bakes in ν=2, and under BKT ν is ill-defined. The ν-drift was already
  unmeasurable (Spearman −0.07, p=0.88); one clean size won't change that.
  Measuring ν(ζ) properly needs ≥4–5 sizes (L=160/192, out of the 2-day scope).
- If the bias is not clean 1/N_c (ESS-collapse curvature), the extrapolation
  leaves a residual systematic — `extrapolate_nc`'s `resid_frac` flags it. Hedge:
  the Rényi-2 / bipartite-MI estimators (less cancellation → likely less bias).
- L=160 (4th FSS point) is OUT of this baseline: one L=160 N_c=500 seed is
  ~63h > 48h cap. A reduced-N_c L=160 run is a separate later step.

### Slope test (separate from rescue)

`pps_clone_slope` (528 tasks): submitted earlier; status should be checked.
Designed to discriminate Möbius slope 1/8 vs naive NLSM slope 1/4 at
$\zeta = 1$ via $\zeta \in \{0.70, 0.80, 0.90\}$ at $L = 192, 256$.

### What's NOT yet running

- **L = 192, 256.** Originally planned as the Phase-2 supplement (see
  `make_clone_phase2_grid()` in `grid_pps.py`, scripts not yet written).
  The cost + variance-inflation analysis from the partial dense data
  (see "Dense campaign empirical findings" below) shows these are
  infeasible with the cloning method at this scale; the Phase-2 grid is
  kept in `grid_pps.py` for completeness but should not be submitted as
  designed. The decisive small-ζ resolution gap is the methodological
  limit, not a budget question.
- **Case A implementation**. Spec at `theory/CASE_A_IMPLEMENTATION_SPEC.md`;
  backend written and validated against exact Fock space (2026-06-06, all
  hard gates pass — see code file map below). Production Binder scan + FSS
  not yet run.

---

## Theoretical synthesis (the long-form story)

### Case B (α + w = 1, γ = 0): the main project

**Model.** 1D Kitaev chain at the topological point ($\mu = 0, \Delta = w$),
single Bogoliubov-density measurement $\tilde L_j = d_j^\dagger d_j$ with
$d_j = \tfrac{1}{2}(\gamma_{2j} - i\gamma_{2j+3})$ at rate $\alpha$,
hopping rate $w$, $\alpha + w = 1$, PPS parameter $\zeta \in (0, 1]$.

**Lindbladian-level setup.** Same as KMR's $\gamma = 0$ edge. The Lindbladian
is unraveling-independent; QJ and QSD differ only in their trajectory
sampling.

**Replica Keldysh action.** Following Le Gal-Schirò, the replicated
density-matrix evolution under PPS gives an action
$$
S[\bar\Phi,\Phi;\zeta] = S_{\rm kin} + S_{\rm nH}
- i\zeta\gamma \int dt \sum_j \prod_{r=1}^N \mathcal V_{j,r}
$$
where $\zeta$ enters *only* in the cross-replica vertex. The non-Hermitian
replica-diagonal part is ζ-independent. This is exact and structural — not
an approximation.

**Symmetry class.** Class DIII (Altland-Zirnbauer); NLSM target $SO(R)$
in the replica limit $R \to 1$.

**The matched-NLSM derivation.** Near the multicritical point $(\lambda, \zeta)
= (0, 0)$, the two relevant operators have RG eigenvalues
$$
y_\lambda = 1/\nu = 1/2, \qquad y_\zeta = 2 - \Delta_\zeta^{\rm UV} = 1.
$$
The $\lambda$-perturbation generates a crossover length
$\xi_\lambda^{\rm cross} \sim \lambda^{-1/y_\lambda} = \lambda^{-2}$ (this is
the definition of $\nu = 2$, not a separately-derived quantity). At this
scale the running cross-vertex coupling reaches
$\zeta_{\rm eff} \sim \zeta \cdot \lambda^{-2}$. Criticality is the matching
condition $\zeta_{\rm eff} \sim K^*$ (universal NLSM critical coupling),
giving
$$
\lambda_c \;\sim\; \sqrt{c_\lambda / K^*} \, \sqrt\zeta \;\equiv\; C \sqrt\zeta.
$$

**Validity / assumptions.** The derivation assumes:
(i) The microscopic $\lambda$ has nonzero linear overlap with the DIII
relevant scaling field (data confirms this — $\phi = 1$ excluded at $9\sigma$).
(ii) The critical condition is reached at the matching scale, so IR
running of $\zeta$ inside the NLSM regime contributes only subleading
corrections (this is the residual assumption — Δ_ζ^IR not computed).
(iii) The QJ-PPS-Case-B model flows to the same class-DIII fixed point as
the LMR/KMR field theory predicts.

**Born-rule endpoint.** $\lambda_c(\zeta=1) = 1/2$ from Carollo et al.
PRA 98, 010103 (2018), analytically. The Möbius interpolation
$\lambda_c(\zeta) = \sqrt\zeta / (1 + \sqrt\zeta)$ is a phenomenological
[1,1] Padé in $\sqrt\zeta$ that matches both the small-ζ scaling and the
Born endpoint, fits the data to $\sim 10\%$, but is *not* derived.

**Post-selected endpoint** ($\zeta \to 0$). Deterministic non-Hermitian
evolution. Whether this is genuinely a critical point with ν₀ = 2 or a
broadening crossover is not entirely clear from data — the FSS at ζ = 0.02
gives ν = 3.10 (large, suspicious), at ζ = 0.03 gives ν = 1.82. Either the
post-selected limit is itself critical (with $\nu_0$ to be determined) or
the limit is singular. **This is a real open question.**

### Case A (α + γ = 1, w = 0): the prediction

Two on-site measurements (c-density at rate γ, d-density at rate α), no
Hamiltonian. The self-duality $\alpha \leftrightarrow \gamma$, $c \leftrightarrow d$
is exact at the Born rule. **PPS respects this self-duality** because
$\zeta^{N_T}$ depends only on the total click count $N_T = N_c + N_d$, not
on which channel fired. Therefore the self-dual line $\lambda_c^A = 1/2$
is pinned for all $\zeta \in (0, 1]$.

**Universality class (REVISED 2026-06-10).** Class D ✓ — but "self-duality
⇒ Ising" conflated the class with the fixed point. The Ising values
(c = 1/2, ν₀ = 1) are DERIVED at the **ζ=0 endpoint** [V] (imaginary-time
projection onto the uniform zigzag Majorana ground state at λ = 1/2); they
do NOT transfer to the Born line. On the Born line ζ ∈ (0,1] the R=2 anchor
is SO(4)/U(2) ≅ S² at θ = π (pinned by duality) → SU(2)₁: c = 1, ν = 2/3
[P]; the n→1 values are [O]. Self-duality pins the LOCATION (λ_c^A = 1/2
for all ζ, exact) and θ = π — it does not pin Ising. GATE before any
universality claim enters the thesis: measure ν via dB_L/dλ at exactly
λ = 1/2 (location pinned ⇒ best ν estimator available). Case B remains
class DIII (exact decoupling → two identical DIII chains).

**Status.** Backend validated 2026-06-06 (Gaussian vs exact Fock space, L=6,
agree at λ_A = 0.3, 0.5, 0.7). The λ_c^A = 1/2 and universality *physics*
remain *not yet numerically verified* — that requires the Binder-crossing
scan. Caveat surfaced during validation: S(L/2) is strongly asymmetric under
λ_A ↔ 1−λ_A (≈0.31 vs ≈1.03 at L=16), confirmed by the exact backend. This
is expected (site-density measurement disentangles the Néel state, bond
measurement does not) and does NOT bear on λ_c, because S(L/2) is not
duality-invariant — the c↔d duality is non-local and scrambles the cut.
Implementation spec at `theory/CASE_A_IMPLEMENTATION_SPEC.md`.

### Comparison to KMR and LMR

| | KMR 2023 | LMR 2025 | This project |
|---|---|---|---|
| Hamiltonian | det. Kitaev | random unitaries | det. Kitaev |
| Measurements | two on-site | two with asymmetry | one (Case B) |
| Unraveling | QSD | QSD | **QJ** |
| PPS | no | yes | yes |
| Phase boundary | $\lambda_c$ at Born | discontinuous at $\zeta^* \approx 0.28$ | $\lambda_c \sim \sqrt\zeta$, continuous |
| No-click ξ | $\sim \lambda^{-2}$ (QSD) | $\sim \lambda^{-2}$ (QSD) | **gapless** (QJ) |

The QJ unraveling has a *gapless* effective no-click Hamiltonian for the
distance-3 Majorana bond, in contrast to KMR/LMR's QSD case. This is a real
qualitative difference. The universal MIPT exponents need NOT be
unraveling-independent (unraveling-induced transitions exist; ref to verify:
Eissler-Lesanovsky-Carollo arXiv:2406.04869), and the microscopic localization
picture certainly is not — so the QJ-vs-QSD distinction may extend to universality,
not just microscopics (see §D6).

---

## Numerics status

### Dense campaign empirical findings (June 2026)

From the partial aggregate (`clone_aggregate_dense_partial.pkl`, 3198 entries),
three quantitative findings shaped the rescue design.

**1. Measured λ_c(ζ) from L≤96 Binder crossings.** Crossings at (16,32),
(32,64), (48,96) agree well at moderate-to-large ζ, locating λ_c clearly:

| ζ | (16,32) | (32,64) | (48,96) | adopted center |
|---|---|---|---|---|
| 0.10 | 0.161 | 0.223 | 0.165 | 0.19 (noisy — see below) |
| 0.15 | 0.175 | 0.244 | 0.200 | 0.22 |
| 0.22 | 0.228 | 0.258 | 0.230 | 0.24 |
| 0.25 | 0.239 | 0.262 | 0.246 | 0.25 |
| 0.30 | 0.257 | 0.262 | 0.257 | 0.26 (very stable) |
| 0.40 | 0.287 | 0.308 | 0.339 | 0.32 |
| 0.50 | 0.337 | 0.357 | 0.368 | 0.36 |

These are the centers driving the rescue λ-mesh. They differ from the
$C\sqrt\zeta/(1+C\sqrt\zeta)$ phenomenological fit by 0.05–0.07 at moderate
ζ, which is *the reason Phase 2 was held back* — submitting it with the
placeholder √ζ-fit centers and ±0.06 windows would have partly missed the
actual crossings. Centers in the rescue grid are measured.

At ζ=0.10 the crossings scatter (0.16–0.22) because L≤96 isn't asymptotic
there ($\xi_{\rm nc} \gtrsim L$). Wider rescue window covers this.

**2. Cloning variance inflation with L.** B_L relative error in the
critical band, measured at the actual N_c used:

| L | N_c | B_L rel-err | CMI rel-err |
|---|---|---|---|
| 32 | 1000 | 2.8% | 2.0% |
| 48 | 600 | 7.5% | 5.9% |
| 64 | 400 | 10.9% | 8.1% |
| 96 | 450 | 13.5% | 9.6% |

The error grows with L *even as N_c stays similar*. Cause: effective sample
size (ESS) collapses near criticality (inherent to cloning / importance
sampling). Projected N_c needed for 5% B_L error: ~3300 at L=96, ~5000 at
L=128 — combined with L⁴ compute scaling, the cost of "clean" Binder
crossings at L≥128 is prohibitive. This is a methodological limit of
cloning at large L, worth flagging in the thesis as such.

CMI is consistently ~30% tighter than B_L at fixed N_c — useful, but doesn't
eliminate the inflation.

**3. Compute cost model validated.** Wall-time per task scales as
$t \propto N_c \cdot T \cdot L^4$, validated to within 10% on all
L ∈ {24..96} measured points. The L⁴ exponent (rather than L³) comes from
n_steps ~ T·α·L combined with per-step cost ~L³. Anchor: L=96, N_c=450,
T=100 → ~7.6h mean per task with 5-worker realisation parallelism. From
this model:

- L=128, N_c=250, T=100: ~13h/task → 130 rescue tasks fit in ~72h
- L=160, N_c=120, T=100: ~16h/task → 130 tasks ~87h
- L=192, N_c=100, T=100: ~27h/task → 130 tasks ~117h (marginal)
- L=256, N_c=100, T=100: ~85h/task → **infeasible** for any reasonable task count

The combination of L⁴ compute and ESS variance inflation is why L≥192 is
out of reach with cloning at this scale, and L=256 is structurally
infeasible without methodological changes.

**4. T = 100 is borderline-low at L=128, not overkill.** Entanglement
saturation is ballistic ($T_{\rm sat} \sim L/v$). The smaller L actually
ran at *longer* T (L=48 used T=200 from the time_horizon_v2 caps), so if
anything the cheap sizes were over-resourced. T=100 should be held at
L=128, not cut, despite the cost; a saturation check script
(`analysis/phase2_saturation_check.py`) is available if needed.

**5. Scaling test from partial data (inconclusive but suggestive).**
Using the best-available (32,64) or (48,96) crossings to form
$g_c = \lambda_c/(1-\lambda_c)$, then plotting $g_c/\sqrt\zeta$ and $g_c/\zeta$
versus ζ across ζ ∈ [0.02, 0.50]: the $\sqrt\zeta$ ratio is roughly flat
around 0.6–0.9, while $g_c/\zeta$ decreases steeply from ~3.7 (small ζ)
to ~1.2 (large ζ). This leans toward the √ζ hypothesis but does *not*
settle it — the (32,64)/(48,96) crossings are not L-asymptotic and the
blue curve still has structure. The L=128 rescue is what resolves this.

### Data on disk

| Aggregate | Path | Entries | Status |
|---|---|---|---|
| v2 cloning | `~/Downloads/clone_aggregate(1).pkl` | 1920 | complete, L≤128 + L=192,256 sparse |
| Run AC | `~/Downloads/aggregate_runAC.pkl` | (merged) | dense λ around critical |
| Run B | `~/Downloads/aggregate_B.pkl` | 216 | L=192,256 at ζ ∈ {0.05, 0.10, 0.20, 0.50, 1.00} |
| Slope grid | (Habrok scratch) | submitted | ζ ∈ {0.70, 0.80, 0.90}, L=192,256 |
| Dense fine-grid (partial) | `~/Downloads/clone_aggregate_dense_partial.pkl` | **3198 / 4112** | small_L complete; medium_L missing 228 L=64; L=96 partial (342/514); **L=128 missing entirely** |
| Dense L=64 backfill | (Habrok `pps_clone_dense`) | resubmit needed | 228 tasks via `submit_clone_dense_L64_backfill.sh` |
| Rescue L=128 | (Habrok `pps_clone_rescue`) | submit needed | 130 tasks via `submit_clone_rescue_L128.sh` |
| Rescue L=160 (optional) | (Habrok `pps_clone_rescue`) | submit needed | 130 tasks via `submit_clone_rescue_L160.sh` |

### Key result: $\phi$ from global FSS

> NOTE (2026-06-17): the ν statement here is QUALIFIED by the top banner. The global FSS collapse cannot resolve ν at this L-range (synthetic calibration: recovered ν̂ ≈1.2–1.7 regardless of true ν), and B_L biases ν low; use ⟨CMI⟩ with quotients / Cut-A curvature. The φ and λ_c results (√ζ, Born endpoint) still stand for the clean ζ≥0.25 window.

Global FSS collapse on the cleanest range $\zeta \in [0.03, 0.85]$,
all L ∈ {64, 96, 128, 192, 256}:
- $\phi = 0.56 \pm 0.05$ (free power-law fit)
- $C = 1.02 \pm 0.10$ (prefactor)
- $\chi^2/{\rm dof} = 3.8$
- Consistent with $\phi = 1/2$ at $1.3\sigma$
- Excludes $\phi = 1$ at $9\sigma$

Effective exponent from pairwise crossings (current L ≤ 128 data alone):
$\phi_{\rm eff} = 0.76$ (L=96/128) to $0.84$ (L=64/128). **Trending toward
0.5 with L but not converged** — the finite-size bias at L ≤ 128 is real
and not removable by more statistics at fixed L.

### Tests that didn't decide $\phi = 1/2$ vs $\phi = 1$

- **The ν-drift test.** The relation $\nu - \nu_0 \sim \zeta^{1+p}$ should
  give a power-law drift in $\nu(\zeta)$, with exponent $1+p$ locking the
  critical-line exponent $p$. **Empirically the drift is too small to see**:
  predicted magnitude $\lesssim 0.4$ at ζ=0.7 vs measured ν error bars
  $\pm 0.1$–$0.3$. Spearman correlation of ν vs ζ in $[0.05, 0.7]$ is
  $-0.07$ (p=0.88), i.e., no detectable trend. The plateau holds, but
  the drift can't be measured.
- **The slope-at-ζ=1 test.** Requires L ≥ 192. Currently the data in the
  large-ζ band is too noisy (N_c too low at L=192,256) for the slope to
  discriminate $1/8$ (Möbius) vs $1/4$ (naive NLSM).

### What CAN currently be concluded

- The genuine PPS cross vertex is **marginal** (Δ≈2); the λ⁻¹ no-click scale
  is the **selection length ℓ_λ=4w/λ** (2026-06-10: a formation scale, not a
  state ξ); the Born endpoint λ_c(1)=1/2 is recovered. [VERIFIED]
- The √ζ *derivation* (Δ_ζ=1 + ξ~λ⁻²) is **invalid** for this model.
- Fitting the physical ratio r_c=λ_c/(1−λ_c) gives φ≈0.7–0.85 (not ½);
  √ζ overshoots, linear undershoots; neither Möbius form fits well. The
  previous φ≈0.56 is largely an artifact of fitting λ_c (which saturates).

### What CANNOT yet be concluded

- The precise asymptotic value of $\phi$ (could be exactly 1/2 with
  corrections, or could be a value in $[0.5, 0.6]$).
- Whether $\lambda_c = \sqrt\zeta / (1 + \sqrt\zeta)$ is exact or just a
  Padé interpolation.
- The slope at ζ=1 (1/8 vs 1/4).
- The Case A prediction (numerics not yet done).
- The Δ_ζ^IR question (not numerically accessible at this resolution).

---

## Open questions and immediate next steps

### Theoretical

1. **Δ_ζ^IR at the class-DIII NLSM fixed point.** Whether the cross-vertex
   renormalizes the marginal NLSM stiffness ($\Delta_\zeta^{\rm IR} = 2$,
   what the collaborator's analysis suggests) or remains relevant
   ($\Delta_\zeta^{\rm IR} = 1$, what the matched argument implicitly
   assumes). Status: open. Prompt at `theory/PROMPT_DELTA_ZETA_IR.md`.
2. **Crossover function $\lambda_c(\zeta)$.** Derive (or refute) the
   Möbius form. The collaborator's analysis showed the linearised RG
   has $y_\lambda = y_v = 1/2$ in $v = \sqrt\zeta$, so the prefactor
   $C$ is non-universal at linear order. One-loop NLSM in the joint
   $(\lambda, \zeta)$ plane needed to decide. Status: open. Prompt at
   `theory/PROMPT_CROSSOVER_FUNCTION.md`.
3. **Post-selected endpoint** ($\zeta = 0$). Is it a critical point with
   $\nu_0$ to be determined, or a crossover? Not yet investigated.
4. **Feedback.** Adding coherent feedback (measurement-conditional unitary)
   or adaptive measurement to the QJ-PPS protocol. Not explored
   analytically. Would change the cross-vertex structure.

**LMR-style ζ\* breakdown / BKT target (new, 2026-06-04; UPDATED 2026-06-06).**
**Analytic side now settled (see `OPEN_ANALYTIC_PROBLEMS.md` §D).** The QJ cross
vertex is **marginal**, not relevant (unlike LMR's QSD vertex), so it does NOT
drive the transition and there is **no n→1 QJ analogue of LMR's ζ\***; the MIPT
is driven by the relevant single-copy mass and is non-perturbative. Three
field-theory routes (two-loop PCM β-function; exact-correlator/integrable
continuation; LMR-style K-matching) converge on this, and via the Coulomb-gas /
U(1) criterion (a BKT essential singularity needs a U(1) / marginal line, which
the Majorana Z₂ class lacks) now show (with the §D6 external-review correction) that the boundary is a
FINITE-ν conventional-type transition, NOT KT: Jian et al. (2302.09094) establish
the generic monitored-Majorana Born transition as a finite-ν novel class (Z₂-defect
driven), so the intermediate "KT at ζ=1" claim is WITHDRAWN. ζ=1 is special only
because the single-copy mass ∝(ζ−1) vanishes; ζ<1 keeps a relevant mass on.
Numerics: expect a power law ξ ~ |t|^{−ν} (ν≈2.1 FIXED along the PPS line — n→1
throughout, see §D8; the forced n→0 value 1.9 is off the line), NOT an essential
singularity; the "ν=3.1 at small ζ = BKT" reading is WRONG. The **Rényi-2 numerical ζ\* test remains
live** (a Rényi-k≥2 feature can exist and wash out by n→1). Original framing
follows. Faithful analog of
LMR's bosonization-breakdown ζ\* (their ζ\*≈0.28 is explicitly a **two-replica /
Rényi-2** result; they state the n→1 behaviour is unknown). Route A (Cardy RG,
y_m = 1 − (π/4)rζ) gives NO crossing in the physical window (y_m>0 along the
whole critical line, bottoming at ≈0.2 near ζ=1) — a negative result; the
relevant-mass picture yields only a slowly drifting effective exponent, not a
sharp ζ\*. Route B (the real target): the **gapless no-click H_eff** (the
QJ-vs-QSD difference) is itself critical, so bosonize/CFT-describe it; Choi
doubling gives ρ/σ (ket/bra) modes; the PPS-weighted clicks are the
cross-contour σ-mode vertex with coupling g₀∝rζ. ζ\* = where that vertex crosses
marginality, Δ_click(ζ\*)=2. **Key structural insight:** the anomalous dimension
driving the crossing ∝ a (2n−2)-type factor → present for Rényi-k≥2, vanishes
at von Neumann (n→1). So a ζ\* is generic for Rényi entropies but may NOT survive
to n→1 — same replica-limit issue as everywhere else, and the reason LMR's ζ\*
is a two-replica statement. **Testable NOW with on-disk Rényi-2/Rényi-3 data:**
look for a ζ\* feature (kink in measured ν, or BKT essential singularity
log ξ ~ (ζ−ζ\*)^{−1/2}) in Rényi-2; if it drifts/weakens with Rényi index →
finite-replica artifact (vanishes at vN); if stable → genuine. Departures from
LMR to check: (i) single measurement (no second physical species — the doubling
is the Choi ket/bra, not two channels); (ii) bare Luttinger K₀ of the gapless
no-click CFT (LMR's came from their Luttinger liquid; here from the distance-3
no-click spectrum — verify it is a tractable single-mode CFT first). Specific ζ\*
value requires redoing LMR's App.-G one-loop K-matching for the distance-3 QJ
operator (scaffolded, not done). Cheap parallel test: BKT vs power-law fit on
small-ζ ξ data; the ν=3.1 blow-up at ζ=0.02 is what force-fitting BKT looks like.

### Numerical

5. **Targeted high-L scan at L=128** — for the decisive small-ζ window this is
   now **SUPERSEDED (2026-06-05) by the N_c-ladder campaign** (see "N_c-ladder
   campaign" above), which adds the {250,500,800} N_c ladder + 15-seed blocks
   the plain rescue lacked and is the run currently in flight. The plain rescue
   below remains the description of the broader-ζ resubmission. Narrow λ windows
   (13 pts, ±0.07) centered on measured
   dense crossings (not √ζ-fit placeholders), 10 ζ values spanning the
   full range, N_c=250 at L=128 / 120 at L=160. Scripts:
   `submit_clone_dense_L64_backfill.sh` (228 missing L=64 tasks),
   `submit_clone_rescue_L128.sh` (priority, 130 tasks ~72h),
   `submit_clone_rescue_L160.sh` (optional, 130 tasks ~87h). Output
   directory `pps_clone_rescue`. **L=192, 256 in the decisive small-ζ
   window are NOT pursued** in this rescue — infeasible per the cost +
   variance-inflation analysis above; the thesis should report the
   methodological limit explicitly.
6. **Case A implementation and FSS.** See `theory/CASE_A_IMPLEMENTATION_SPEC.md`.
   Predicted $\lambda_c^A = 1/2$ for all ζ, Ising universality. ~1 week
   of implementation + ~1 day of FSS runs.
7. **Slope test analysis.** When the submitted slope grid (528 tasks) at
   ζ ∈ {0.70, 0.80, 0.90}, L=192,256 finishes, extract slope at ζ=1 and
   compare to Möbius (1/8) vs naive NLSM (1/4).

---

## Operational

- **HPC (2026-07-07): Ruche** (Paris-Saclay Mésocentre), user `ercetinut`,
  `ssh ruche.mesocentre.universite-paris-saclay.fr`. Rocky Linux 8, Intel Xeon
  Gold 6230 (Cascade Lake, 40 cores/node). Repo cloned to `$HOME/ppsQJ_m2`
  (=/gpfs/users/ercetinut, 50 GB); results to `$WORKDIR/pps/...`
  (=/gpfs/workdir/ercetinut, 500 GB), both visible on compute nodes. conda env
  `$WORKDIR/envs/pps_qj` (numpy 2 + scipy, MKL) via `scripts/ruche/setup_ruche.sh`.
  Partitions: cpu_short (1h/1000c), cpu_med (4h/1000c), cpu_long (168h/160c),
  cpu_prod (6h/2000c, nights+weekends), cpu_scale (1h/4000c). Submit via
  `scripts/ruche/submit_pps_boundary.sh` (size-binned arrays, `--shard`). r=2.43
  (Ruche/Mac core-time ratio; scale cost estimates by it).
- **Git**: push from Mac over HTTPS (works, credential-cached); pull on Ruche.
  Prior HPC was Habrok (RUG, user s4629701) — migrated off 2026-07-07.
- **Repo**: `ueborg/ppsQJ_m2`. Mac path: `/Users/catlover1337/Documents/ppsQJ_m2/`.
- **Aggregate script**: `scripts/aggregate.py` or `scripts/aggregate_runs.py`.
  Auto-slurps all .npz fields; new fields (CMI, $S_{AB}$, Rényi) should
  appear automatically in the aggregate.
- **Thesis draft**: `~/Downloads/m1thesislatex/` (M1 internship report,
  deadline **19 June 2026**).
- **Thesis notes**: `~/Downloads/continuousmeasurements(2)/` (working
  document with theoretical sections, the "main.pdf" referenced
  throughout).

---

## Key references in the project bibliography

| Key | Reference | Used for |
|---|---|---|
| KMR2023 | Kells-Meidan-Romito SciPost Phys 14, 031 (2023) | model (QSD analogue) |
| LMR2025 | Leung-Meidan-Romito PRX 15, 021020 (2025) | PPS framework (QSD-PPS) |
| LeGalSchiro2025 | Le Gal-Schirò arXiv:2511.22506 | replica Keldysh + NLSM derivation |
| Fulga2012 | Fulga-Akhmerov-Tworzydło-Béri-Beenakker PRB 86, 054505 (2012) | class-DIII **Anderson (n→0)** ν≈2 — forced/postselected endpoint ONLY, not the Born MIPT |
| Jian2023 | Jian-Shapourian-Bauer-Ludwig arXiv:2302.09094 | Born (n→1) vs forced (n→0) = distinct universality in class DIII |
| FavaNahum2023 | Fava-Piroli-Swann-Bernard-Nahum PRX 13, 041045 (2023) | principal-chiral SO(N) NLSM for monitored Majorana |
| PoboikoMirlin2023 | Poboiko-Pöpperl-Gornyi-Mirlin PRX 13, 041046 (2023) | U(1) free fermions in 1d: no MIPT, log is a crossover |
| Carollo2018PRA | Carollo et al. PRA 98, 010103 (2018) — quantum-Doob / large-deviation paper (**mis-cited for λ_c**; correct use: the PPS / tilted-ensemble framing) | λ_c(1)=1/2 is **numerically pinned**, true source TBD (see §D6) |

All in `~/Downloads/continuousmeasurements(2)/references.bib`.

---

## File map (theory folder)

- `AGENTS.md` ← **chat-agent protocol: start-of-chat read + handoff update rules**
- `HANDOFF.md` ← this file (canonical project state)
- `SUMMARY_2026_05_22.md` ← detailed theoretical state
- `qj_pps_theory_summary.md` ← long-form derivations (604 lines)
- `qj_pps_final_synthesis.md` ← compact synthesis
- `ONE_LOOP_RG.md` ← matched-NLSM derivation
- `NLSM_FRAMEWORK.md` ← STALE ENTRY (2026-06-10): file absent from theory/
  (likely archive/). Its Case A/B content is superseded by the chat-derived
  class analysis (Case A: class D, SO(2R)/U(R) coset; Case B: exact two-chain
  decoupling → class DIII per chain, Foster constraint S16 → SO(R) PCM).
  Do not cite the old file.
- `CASE_A_IMPLEMENTATION_SPEC.md` ← Case A code spec (612 lines)
- `COLLABORATOR_RESPONSE*.md` ← peer commentary integration
- `PROMPT_*.md` ← prompts for new chats on specific subproblems
- `PROMPT_INTERNSHIP_REPORT.md` ← **master prompt for the thesis synthesis**
- `sec_matching_revised.tex`, `sec_predictions_revised.tex` ← LaTeX sections

## File map (code added this iteration, 2026-06-03)

- `pps_qj/parallel/grid_pps.py` — appended `make_clone_rescue_grid()`,
  `_RESCUE_LAMBDA_C` (measured centers), `nc_for_L_rescue`, plus the
  earlier dense and phase2 grids
- `pps_qj/parallel/worker_clone_rescue_pps.py` — shim worker for rescue grid
- `slurm/submit_clone_dense_L64_backfill.sh` — fills the 228 missing L=64
  dense tasks
- `slurm/submit_clone_rescue_L128.sh` — priority: L=128 rescue
- `slurm/submit_clone_rescue_L160.sh` — optional: L=160 rescue
- `analysis/phase2_saturation_check.py` — T-saturation diagnostic
  (written, not yet run on cluster)

## File map (code added this iteration, 2026-06-05) — N_c-ladder campaign

- `pps_qj/parallel/grid_pps.py` — appended `make_clone_ladder_grid()`,
  `task_params_clone_ladder`, `clone_ladder_rung_ranges`, `_LADDER_*` config
  (L=128, 7 small-ζ, N_c {250,500,800}, 3 seed-blocks; seeds offset +20e9,
  verified disjoint from v2/dense/rescue/slope)
- `pps_qj/parallel/worker_clone_ladder_pps.py` — shim worker for the ladder grid
- `slurm/submit_clone_ladder.sh` — job-array submit per N_c rung (auto-spreads
  across nodes; each rung → own dir); usage + node/conc guidance in the header
- `analysis/aggregate_ladder.py` — block-pooling aggregator (concatenates the
  per-realisation arrays across seed-blocks; the stock aggregator overwrites)
- `analysis/extrapolate_nc.py` — per-point 1/N_c extrapolation → B_∞ + merge
  with clean low-L → FSS-ready pkl (prints bias-linearity residual)
- `analysis/renyi_washout.py` — B4 item 3 (von Neumann vs crossover via the
  Rényi-index dependence of the S_n log-coefficient); validated on synthetic

## File map (code added this iteration, 2026-06-06) — Case A backend

- `pps_qj/gaussian_backend_caseA.py` — Case A Gaussian QJ backend. Site
  channel c†c at rate γ on pair (2j,2j+1), bond channel d†d at rate α
  (identical operator to Case B). Two structural differences from Case B,
  both validated: rate-weighted uniform decay γL + α(L−1) in the branch
  norm, and rate-weighted channel selection. Local `site_jump_pair`; the
  Case B file is untouched.
- `pps_qj/exact_backend_caseA.py` — exact Fock-space reference (L ≤ 10).
  Site projector n_j, bond projector built identically to Case B so any
  mismatch isolates to the new site channel.
- `tests/validate_caseA.py` — standalone gate suite. Hard gates PASS
  2026-06-06: generator algebra, λ_A=1/2 sanity, Gaussian-vs-exact at L=6
  (site-click fractions agree to 3 digits → site convention correct, no
  flip), and Case A(γ=0) = Case B(w=0). Self-duality S-check is
  informational only (S is not duality-invariant).

## File map (code added this iteration, 2026-06-10) — anchor scan (gate 1) + area-phase (gate 2)

- `analysis/anchor_scan.py` — single-particle test of the ζ=0 SSH anchor under
  real conventions (α=λ, w=1−λ, κ=λ/4, EP at λ*=4/5). Confirms Fermi step at
  q=±π/2, state ξ ~ 1/ln(4/λ) (ξ_ps~λ⁻² refuted), ν₀≈1. Δ_B left as a hook
  (`delta_B_hook`) — needs the real no-click Majorana covariance, not the band
  structure. Runs in seconds; no cluster.
- `analysis/delta_B_zeta0.py` — GATE 1 (a)+(b) closer. Builds the REAL no-click
  steady-state Majorana covariance (replicates the worker_zeta0 evolution loop),
  computes the connected single-state bond correlator cq(r) (Wick form; = opdim's
  cq), fits Δ_B on EVEN r (expect ≈1, ties to 1.009), reports odd-r null as the
  decoupling/reduction check. Deterministic, O(L³); seconds on Mac/Habrok, no
  cluster. Fit logic validated on synthetic data (recovers Δ_B=1.000 / 1.200).
- `pps_qj/parallel/worker_areaphase_pps.py` — GATE 2 worker. Cloning at ζ<1
  (reuses run_cloning), then clone-population C_sc(r)=Cov(b_x,b_{x+r}) with
  b[x]=Γ[2x,2x+3]; ξ from exp-fit on EVEN r (odd-r null built in). Env grid
  auto-places λ=λ_c(ζ)+offset in the area phase. CAVEAT: clone-pop Cov has
  genealogical bias — run an N_c=500 rung vs 250 before banking φ.
- `analysis/fit_areaphase.py` — loads areaphase_*.npz; fits ξ(ζ)~ζ^{−p} per
  (L,offset) → p≈0.5 (φ=1/2 window law) vs p≈1 (φ=1 coherent); λ-flatness +
  odd-r-null + exp-fit-R² health checks. Discriminator validated on synthetic.
- `slurm/submit_areaphase.sh` — 30-task array (2 L × 5 ζ × 3 offsets), 5 cpus/task.

## File map (docs/code added this iteration, 2026-06-16) — variance reduction

- `theory/VARIANCE_REDUCTION.md` — full variance-reduction methodology study
  (c=zeta optimal; coupling ~2x for entanglement FSS; martingale CV ~3x for
  activity only; closed negative directions; implementation plan; the
  fixed-start-artifact corrections).
- `analysis/var_reduction/coupling_lambda.py` — coupled lambda-points prototype
  (delta-scan + L-scan; entropy and B_L differences).
- `analysis/var_reduction/coupling_cmi_kmr.py` — coupling on clean <CMI> and the
  KMR product <CMI><S>.
- `analysis/var_reduction/activity_cv.py` — end-to-end tilted-activity martingale
  CV (the validated ~3x).
- `analysis/var_reduction/scgf_cv.py` — end-to-end SCGF CV (the negative result;
  shows 1.03x, documents the fixed-start artifact).

The most important documents to read first are `HANDOFF.md` (this file),
then `SUMMARY_2026_05_22.md`, then `PROMPT_INTERNSHIP_REPORT.md`.
