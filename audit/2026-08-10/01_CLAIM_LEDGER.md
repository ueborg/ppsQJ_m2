# Claim ledger (Stage 1, repository + data evidence only)

Audit 2026-08-10. **Non-canonical.** IDs are provisional.

Status: `[V]` verified · `[P]` plausible · `[O]` open · `[X]` contradicted/superseded.
Evidence class: `DATA` `EXEC` `DERIV` `SOURCE` `CODE` `CHAT` `MEMORY` `INFERENCE`.

Verification tiers used below, never conflated:
- **provenance-verified** — a named executed artifact exists and was located.
- **reproduced** — recomputed in this audit from preserved data.
- **derivation-checked** — mathematics independently checked. **Nothing in Stage 1
  reaches this tier.** No derivation was independently re-done.
- **source-verified** — external paper fully inspected. **Nothing in Stage 1
  reaches this tier.** Deferred to Stage 4.

---

## Boundary, amplitude, exponent φ

### `CB-AMP-001` — λ_c amplitude
**Claim.** `λ_c(ζ) = A√ζ` with `A ≈ 0.49–0.51` on Cut B.
**Status** `[V]` · **Evidence** `DATA`, reproduced ·
**Source** `agg_caseB_combined.pkl` via `audit/.../reproduce_amplitude.py`; corroborated by HANDOFF 2026-06-07 cont.-1 #2 (dense 0.51, v2 0.53) and cont.-2 #1 (debiased 0.501).
**Date** established 2026-06-07, reproduced 2026-08-10.
**Contradicted by** `theory/archive/SESSION_2026_05_20.md` (A=0.96, 2026-05-20); `archive/NLSM_FRAMEWORK.md:286`; **current project memory**; `Chapter3.tex:236`.
**To upgrade** already `[V]` for the amplitude. Error bar requires bootstrap + drift errors.

### `CB-AMP-002` — the 0.96 figure
**Claim.** `A = 0.96 ± 0.05` is not the λ_c amplitude. The r_c parameterization gives ≈0.87 (free power) / 0.69 (φ=1/2 fixed).
**Status** `[V]` · **Evidence** `DATA`, reproduced · **Date** corrected 2026-06-07/06-10.
**Note.** Memory states the inversion of this as if it were the correction. See `03_AMPLITUDE_TRACE.md`.

### `CB-PHI-001` — φ on λ_c
**Claim.** Free-power fit to λ_c gives φ ≈ 0.49–0.56, consistent with 1/2.
**Status** `[P]` · **Evidence** `DATA`, reproduced (this audit: φ=0.495 over ζ∈[0.05,0.85]).
**Contradictory evidence** HANDOFF 2026-06-07 cont.-1 #3: **five functional forms fit at χ²/dof < 0.6 and are statistically indistinguishable** (a√ζ, free power, √ζ-Möbius-2p, log-corrected, linear+intercept). Form degeneracy means the λ_c fit does not select √ζ.
**To upgrade** requires breaking form degeneracy, which crossing-shape fitting provably cannot do.

### `CB-PHI-002` — φ on r_c
**Claim.** The unbounded ratio r_c gives φ ≈ 0.65–0.85, i.e. **neither √ζ nor linear**; λ_c-φ≈0.5 is partly a saturation artifact of fitting a bounded quantity.
**Status** `[P]` · **Evidence** `DATA`, reproduced (this audit: φ_r = 0.681).
**Date** 2026-06-07. **Tension** with `CB-PHI-001` is genuine and is preserved, not resolved.

### `CB-PHI-003` — φ = 1 exclusion
**Claim.** Linear scaling φ=1 is excluded.
**Status** `[P]` · **Evidence** `DATA`/`INFERENCE`. HANDOFF quotes 9σ from the global FSS. But cont.-1 #3 lists `linear+intercept` among the five indistinguishable forms at χ²/dof 0.55, and explicitly notes the 9σ "does NOT adjudicate h_d". **The 9σ figure and the form-degeneracy finding are in direct tension and both live in HANDOFF.**
**To upgrade** re-run exclusion on r_c with the corrections model.

### `CB-PHI-004` — φ = 0.502 ± 0.026 via 1/√L Binder extrapolation
**Status** `[X]` · **Evidence** `MEMORY`, `CHAT`.
**Superseded because** (i) it rests on the A=0.96 fit (`CB-AMP-002`); (ii) HANDOFF 2026-06-17 [V] finds B_L carries an explicit lnL factor biasing exponents low and mandates ⟨CMI⟩ instead; (iii) global collapse cannot resolve exponents at this L-range at all.
**Still asserted by** project memory as a "Confirmed result".

### `CB-BORN-001` — Born endpoint λ_c(1) ≈ 0.5
**Status** `[P]`, **downgraded by this audit** · **Evidence** `DATA` (legacy only).
**Finding.** **No guided aggregate contains ζ = 1** (max 0.85). ζ=1 exists only in `clone_aggregate(2).pkl`, `clone_aggregate_dense_full.pkl`, `ladder_fss_ready.pkl` — all pre-guided estimator. HANDOFF calls Born-endpoint reproduction "the robust headline"; on current-generation data it is an extrapolation.
**To upgrade** run the ζ=1 Cut B ladder (cloning-free, weights ≡ 1, cheapest run in the project). It has been the top data-plan item since 2026-06-17 and was never executed.

### `CB-BORN-002` — attribution of λ_c(1) = 1/2 to Carollo et al. PRA 98 010103
**Status** `[X]` · **Evidence** `SOURCE`, not re-verified here.
HANDOFF 2026-06-06 and its reference table: mis-attributed, Carollo is a quantum-Doob/large-deviation paper. True source **TBD**.
**Still asserted by** project memory. **Stage 4 item.**

---

## Exponents ν, y_λ, y_ζ

### `CB-NU-001` — Cut B ν
**Claim.** ν_B is consistent with ≈2 but is **not measured**; 95% confidence set ≈ [1.5, 3].
**Status** `[V]` as a statement about *our own resolving power* · **Evidence** `EXEC` (`/tmp/pps_lmr_robust.py`, output not preserved — script location now dead).
**Key finding** correction-model uncertainty dominates and does **not** shrink with n_real. Pinning ν_B needs larger L, not more statistics.
**Contradicted by** memory ("ν = 1/y_λ = 2" stated flatly) and by HANDOFF's own bottom-half "ν(ζ) scattered around ~2, consistent with the theory-predicted plateau".

### `CB-NU-002` — collapse-based ν is not a measurement
**Claim.** All four parametric collapse variants fail to recover a known ν on synthetic data with real grids (ν_true 1.0→3.0, 2.0→3.3, 2.5→3.2).
**Status** `[V]` · **Evidence** `EXEC`, provenance-verified only (scripts were `/tmp`, now gone).
**Consequence** the historical ν values 1.3 (B_L), 1.6 (CMI), and "ν≈2 plateau" are artifacts.

### `CB-NU-003` — data adequacy for ν
**Claim.** No dataset in the project satisfies the project's own stated requirements for ν.
**Status** `[V]`, **new in this audit** · **Evidence** `DATA`, reproduced.
T/L ≤ 1.0 everywhere against a required T ≥ 2L; n_real = 5 everywhere against a required ≈25. See `02_DATA_INVENTORY.md` G2, G3.

### `PPS-YZETA-001` — y_ζ = 1 from Δ_ζ = 1
**Status** `[X]` · **Evidence** `MEMORY`.
Superseded by the normal-ordered cross-vertex result Δ ≈ 2 (marginal), verified in `analysis/cross_vertex_dimension.py`. HANDOFF TL;DR records the correction. Memory still asserts the original.

### `PPS-YLAM-001` — y_λ = 1/2
**Status** `[O]` · **Evidence** `INFERENCE`. HANDOFF §D7-D9: this is Foster–Guo–Jian–Ludwig's R=2−ε expansion **calibrated to Jian's ν**, i.e. numerically anchored, not derived. Presented as derived in memory.

---

## Mechanism: ξ, vertex, NLSM

### `CB-XI-001` — ξ_ps ~ λ⁻²
**Status** contested, preserved as such.
`[X]` for the *post-selected state correlation length* (2026-06-10: the only diverging ξ on the ζ=0 line is at the exceptional point, ν₀=1; the λ⁻¹ scale is the **selection length** ℓ_λ = 4w/λ, a formation scale).
`[V]` for the *no-click band-structure* length ξ_nc = 2/ln(1+κ²/w²) ~ λ⁻² (2026-06-15, deterministic run, area-law confirmed for all λ>0).
The 2026-06-10 "refutation" was itself an artifact of `analysis/anchor_scan.py`, whose hardcoded kernel drops the hopping w. **`anchor_scan.py` is wrong and is still present in the repo unmarked.**
`[O]` measured steady-state exponent over λ∈[0.2,0.5] is ~λ^{-1.5}, a crossover.

### `CB-VERTEX-001` — click vertex marginality
**Claim.** The normal-ordered cross-replica vertex is exactly marginal, Δ ≈ 2 (measured 2.02).
**Status** `[V]` · **Evidence** `EXEC` (`analysis/cross_vertex_dimension.py`), provenance-verified, not re-run.
**Consequence** no n→1 QJ analogue of LMR's ζ*.
**Open tension** `theory/archive/qj_chiral_vertex_result.md` finds the vertex **purely chiral** (K=1 to all orders ⇒ ν constant), while the 2026-06-17 Ashkin–Teller memo requires a non-chiral ε_+ε_- at the Ising corner. HANDOFF proposes these are complementary corners (Luttinger vs free-Majorana). **Not settled.** `[O]`

### `CB-NLSM-001` — matched-NLSM derivation of √ζ
**Status** `[X]` · The derivation (Δ_ζ=1 + ξ~λ⁻² ⇒ ζλ⁻² ~ K*) is declared **invalid** by HANDOFF since 2026-06-03, both inputs having failed.
**Replaced by** `[P]` corner-matching at the ζ=0 anchor (ℓ_λ = ξ_×, one-hit-per-cell window law), **conditional on the area-phase ξ gate**.
**Gate status** `worker_areaphase_pps.py` + `analysis/fit_areaphase.py` + `slurm/submit_areaphase.sh` written 2026-06-10, **never run**. φ=1/2 has been conditional on an unexecuted gate for two months.
**Memory** still presents the matched-NLSM derivation as the framework.

### `CB-CLASS-001` — class assignments
**Claim.** Cut B is class DIII (exact two-chain decoupling ⇒ two identical DIII chains); Cut A is class D.
**Status** `[P]` · **Evidence** `DERIV`, not independently checked.
**Caveat** HANDOFF 2026-06-04: the monitored-Majorana literature uses the **principal-chiral SO(N)** target, not the SO(2n)/U(n) coset the older derivations assumed; the target "must be rederived from the Choi action, not inherited". That rederivation is not recorded as done. Memory still states "Case A: SO(2N)/U(N), Ising (c=1/2, ν=1); Case B: DIII, SO(N)".

### `CB-CLASS-002` — Cut A universality is Ising
**Status** `[X]` as stated · Ising (c=1/2, ν₀=1) is derived at the **ζ=0 endpoint only**. On the Born line the R=2 anchor gives SU(2)₁ (c=1, ν=2/3) `[P]`; n→1 values `[O]`. Self-duality pins the **location** λ_c^A = 1/2, not the fixed point.
**Live question** `[O]` HANDOFF 2026-06-17: the project NLSM predicts Ising for all ζ, LMR's measurement-only analogue shows a crossover to 5/3. These conflict. Settling it requires Cut A data at **large ζ**, where coverage is sparse. Existing Cut A data reaches ζ=0.85, so this is partly addressable now.

### `CB-ZETA0-001` — singular ζ→0 endpoint
**Claim.** `lim_{ζ→0+} λ_c(ζ) = 0`, reached continuously; no-click state is area-law for every λ>0 and critical only at λ=0.
**Status** `[V]` · **Evidence** `EXEC` + `DERIV` (2026-06-15 deterministic run, fit-free saturation of S(L/2) at λ=0.3,0.5,0.6 for L=16→128).
**Memory variant** "lim = 0 ≠ λ_c(0) = λ* > 0" asserts a nonzero λ* at exactly ζ=0. That λ* = 4/5 came from the falsified `anchor_scan.py`. `[X]`

---

## Methodology and estimators

### `METH-OBS-001` — B_L versus c_eff versus ⟨CMI⟩
**Claim.** The c_eff threshold method is wrong (crossings occur at c_eff ~ 6–8, no universal value). B_L crossings are correct for **locating** λ_c. For **exponents**, B_L = ⟨CMI·S_{L/2}⟩ carries an explicit lnL factor and biases ν low; use ⟨CMI⟩.
**Status** `[V]` · **Evidence** `EXEC` (synthetic calibration, ν_true=2 → B_L 1.66 vs CMI 1.92) · **Date** c_eff rejected earlier; B_L-for-exponents demoted 2026-06-17.
**Memory** records only the first half ("B_L crossings are correct", "c_eff is wrong") and omits the 2026-06-17 demotion, so memory reads as endorsing B_L for exponents.

### `METH-EXTRAP-001` — 1/√L extrapolation because ν=2
**Status** `[X]` · **Evidence** `MEMORY`. Circular: the extrapolation variable is chosen from a ν that `CB-NU-001` shows is unmeasured with confidence set [1.5,3]. Also inherits `CB-PHI-004`.

### `METH-CLONE-001` — cloning versus thinning versus Doob-Gaussian
**Claim.** Cloning (SMC) is the correct production algorithm, exact as N_c→∞, no systematic bias. Thinning (R_ζ) ≠ tilting (Q_ζ), differing by the compensator exp((1−ζ)Λ_T). Gaussian/Doob closure failed at intermediate ζ.
**Status** `[V]` · **Evidence** `DERIV` + `CODE`, provenance-verified.
**Note** a separate *Poisson thinning implementation* is a distinct matter and is `[O]` with an **open bug** (N_T low by 1.5%/4.8%), retired from the critical path 2026-08-09 because M=1 domination makes general uniformization unnecessary.

### `METH-GUIDED-001` — guided proposal c = ζ is optimal
**Status** `[V]` · **Evidence** `EXEC` (ESS/N_c 0.97–0.99, c-scan peaks at c=ζ, cost-aware Var×wall also favours c=ζ by ~2×) · **Source** `theory/VARIANCE_REDUCTION.md` §1 — **untracked in git**.

### `VR-CLOSE-001` — the selection-side sampler programme is closed
**Claim.** Selection is not the bottleneck. Var_pop/Var_indep = 1.19 (guided) / 1.49 (ctrl); siblings decorrelate in B_L within one time unit against a T=32–64 horizon; hence N_eff ≈ N_c always.
**Status** `[V]` · **Evidence** `EXEC` (`bottleneck_test.py`, `/tmp/bottleneck_L32.json`) · **Date** 2026-08-10.
**Strength** this is the best-argued result in the recent programme: it supplies a single mechanism explaining every prior negative.

### `VR-CHUNK-001` — chunk lever mult=4 certified
**Claim.** δτ scaling by mult=4 is unbiased for CMI and B_L at R=40; budget 1.6× in production.
**Status** `[V]` · **Evidence** `EXEC` (`chunk_bias_cert.py`, `/tmp/chunkcert_L32.json`).
**Residual** a flat ~2σ shift on S at all mult, argued to be mult=1 fluctuating high rather than an O(δτ) bias (a real discretisation bias must grow with chunk length; this one is flat). Side validation owed, not a blocker.

### `VR-SNAPSHOT-001` — snapshot averaging gain
**Status** `[P]`, **materially weakened by this audit**.
Fixed T=64, spacing 8, paired difference: L=32 → 5.85×, L=40 → 7.73×, **L=48 → 1.22×** (unrecorded, `/tmp/coupsnap_L48.json`), L=64 → 2.66×. Non-monotone by 6×; consistent with noise domination at R=14–16.
Production-matched (T=L) certification now exists and gives **G_snap ≈ 15.8× on σ_λc at K=8** for the (32,64) pair (`/tmp/crossing_prod_result.json`, unrecorded). See `04_MISSING_RESULTS.md` M1, M2.
**Superseded sub-claim** the older "g_snap ≈ K, 3–4× for K=4" is `[X]` at dense spacing.
**To upgrade** repeat T=L certification at a second L-pair and bootstrap the terminal arm properly.

### `VR-GLS-001` — GLS snapshot weights
**Status** `[V]` rejected. Overfitting: in-sample bias ≥1.46× at R=20/K=6, weights 0.043–0.365, a **negative** weight at L=64. Reproduced independently at L=48 in this audit's `/tmp` read (weights 0.039…0.514). Use equal weights.

### `VR-COUPLE-001` — coupled-λ common RNG
**Claim.** Gives only 1.09×, not the recorded ~2×, because common seeds do not survive resampling.
**Status** `[P, probably disposable]` · **Evidence** `EXEC`. Retest owed on `Var(D)/(4δ²)` before closing.

### `VR-DOOB-001` — approximate-Doob h-twisted tapered cloning
**Claim.** Genuine path-weight variance reduction (9.4× at L=32, 16.9× at L=64) with a Galerkin-**predicted** coefficient a*, but **no estimator variance benefit at L=64** (E_B_L ≈ 0.99 vs 1.42× at L=32).
**Status** `[V]` as a methods result, `[X]` as a production recommendation · **Evidence** `EXEC`.
**Mechanism** explained by `VR-CLOSE-001`.
**Standing warning** three cheap proxies each predicted a production win that direct measurement refused to deliver: Var(log W)↓ ⇏ N_c reducible; GESS↑ ⇏ Var(O)↓; ΣD₂↓ ⇏ N_c reducible.

### `METH-METRIC-001` — master metric
**Claim.** Minimise `t_wall · σ²_λc`, with σ_λc ≈ σ_F/|F'(λ_c)|. Not GESS, not Var(log W), not Var(B_L).
**Status** `[V]` · **Date** 2026-08-10. This is the most important methodological decision in the project and appears **nowhere in project memory**.

---

## Infrastructure

### `INFRA-HPC-001` — HPC is Ruche
**Status** `[V]` · Habrok migrated off 2026-07-07. r = 2.43 (Ruche/Mac). Memory lists both, correctly flagged.

### `INFRA-GIT-001` — repository is not synchronised
**Claim.** Last commit 2026-07-26; 106 uncommitted paths; the entire 2026-07-27→08-10 programme, `analysis/var_reduction/`, and `theory/VARIANCE_REDUCTION.md` are untracked.
**Status** `[V]`, **new in this audit** · **Evidence** `CODE`. See `04_MISSING_RESULTS.md` M3.

### `INFRA-PATHS-001` — every data path in HANDOFF is dead
**Status** `[V]`, new · **Evidence** `DATA`. Material relocated to `~/Downloads/01_M1_Internship/`.

### `INFRA-DOCS-001` — AGENTS.md points at files that do not exist
**Claim.** Three of four recommended follow-on reads (`SUMMARY_2026_05_22.md`, `qj_pps_theory_summary.md`, `NLSM_FRAMEWORK.md`) exist only under `theory/archive/`.
**Status** `[V]`, new · **Cause** AGENTS.md mtime 2026-06-03 predates the archive reorganisation (commit c7e945c, 2026-06-06).

### `CASEA-IMPL-001` — Cut A implementation status
**Claim.** Cut A is implemented, validated against exact Fock space, run in production, and aggregated (574 records).
**Status** `[V]`, new · **Evidence** `DATA` + `CODE`.
**Contradicted by** project memory (pending action "Implement Cut A code") and by HANDOFF's own TL;DR ("production Binder scan + FSS not yet run"). HANDOFF's 2026-06-17 banner lists the aggregate, so HANDOFF contradicts itself.
**Open** whether the stored region assignment matches the claimed end-to-end/ABDC requirement. `[O]`
