# ppsQJ_m2 Project — Handoff Notes

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

**Last major update: 2026-06-06** (theory: replica field-theory routes recorded
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

- **HPC**: Habrok cluster (RUG), user `s4629701`. SLURM partitions
  `regularsh`, `regularme`, `regular`. venv at `~/venvs/pps_qj/`.
- **Git push from Habrok fails** (SSH key issue). All commits must be made
  on Mac, pushed to GitHub, then pulled on Habrok.
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

The most important documents to read first are `HANDOFF.md` (this file),
then `SUMMARY_2026_05_22.md`, then `PROMPT_INTERNSHIP_REPORT.md`.
