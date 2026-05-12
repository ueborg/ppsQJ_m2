# PROJECT CONTEXT — ppsQJ_m2
<!-- Last updated: 2026-05-12 -->
<!-- Update this file at the end of every significant work session. -->

---

## HOW TO USE THIS FILE (read first)

This file is the single source of truth for project context across Claude chat
sessions. At the start of any new chat, read this file with Desktop Commander:

    "Please read CONTEXT.md using Desktop Commander, then we can continue."

After any session that changes code, results, or decisions, update the relevant
sections and commit:

    git add CONTEXT.md && git commit -m "context: update [brief description]"

### Recommended startup sequence for a new chat

In any new chat where the goal is to continue project work:

1. **Read CONTEXT.md first** (this file). It captures the high-level state.
2. **Check `git log --oneline -10`** for the most recent commits — anything
   after the date stamp at the top of this file is newer than what's
   summarised here.
3. **Read `theory/STATUS.md`** if the topic is theoretical. That file is the
   reading guide for the analytical documents in `theory/`.
4. **Skim §12 ("Recent session log") below** for a chronological list of
   what was done in recent sessions and what's currently in flight.
5. **For computational questions**, `pps_qj/cloning.py` and
   `pps_qj/gaussian_backend.py` are the core code. For numerical algorithm
   questions, `theory/qj_algorithms_comparison.md` is the canonical
   reference (waiting-time vs stepped-dt formulations).

The recommended prompt to start a continuation chat is:

    Please read /Users/catlover1337/Documents/ppsQJ_m2/CONTEXT.md with
    Desktop Commander, then check `git log --oneline -10` for anything
    newer than the date stamp. Once you've done that, I'll tell you
    what we're working on.

---

## 1. PROJECT OVERVIEW

**Goal (original):** Compute ν(ζ) vs ζ — how the MIPT critical exponent
and critical point shift as a function of partial post-selection (PPS)
parameter ζ = e^{-s} ∈ (0,1].

**Revised goal (after L=128 analysis):** Map the phase boundary
λ_c(ζ) defining the log-law-to-area-law crossover, parameterised by the
effective log-law prefactor c(λ; ζ) in S_{L/2} ≈ a + c ln L. See §6 and
§10. The standard ν-FSS approach does not work for this dataset — see §6.

**Model:** 1D Kitaev-type hopping chain, L sites, OBC.
  H = w Σ (c†_j c_{j+1} + h.c.),  jump operators P_j = d†_j d_j
  λ = α/(α+w),  w = 1-λ,  parametrised by λ ∈ (0,1)

**Physical setup:**
- ζ=1: standard Born-rule MIPT (full monitored), log-law extended criticality
       at small λ, area-law at large λ, smooth crossover. Matches
       Cao-Tilloy-De Luca 2019, Alberton-Buchhold-Diehl 2021.
- ζ=0: fully postselected (no-click), λ_c(L) ~ C/√L with C ≈ 1.0,
       i.e. the apparent transition collapses to λ=0 in the L→∞ limit.
- ζ ∈ (0,1): PPS measure via Doob h-transform / cloning algorithm

**Supervisor:** Dganit Meidan (RUG). Extension of Kells, Meidan, Romito
SciPost Phys. 14, 031 (2023) [diffusive unraveling] and Leung-Meidan-Romito
PRX 15, 021020 (2025) [diffusive PPS-SSE] to the quantum-jump regime.

**Repository:** github.com/ueborg/ppsQJ_m2
**Local path:** /Users/catlover1337/Documents/ppsQJ_m2/
**HPC:** Habrok cluster, RUG, user s4629701, interactive node interactive2
**HPC project dir:** ~/pps_qj/   scratch: /scratch/s4629701/pps_qj/
**Venv:** ~/venvs/pps_qj/
**Workflow:** edit locally → push GitHub → pull on Habrok (no SSH key for direct push)

---

## 2. CURRENT STATE (as of 2026-05-11)

### Production: COMPLETE

The v2 main grid (1920 tasks: L ∈ {8, 16, 24, 32, 48, 64, 96, 128}, 24 λ
values, 10 ζ values, 5 realisations each = 9600 total cloning simulations)
is **fully complete**. Zero collapses, ⟨ESS⟩/N_c ≥ 0.95 at every L.

Total wall time: ~4,460 task-hours. Total CPU-h: ~22,300 (5 workers/task).

The L=128 resubmission completed with the Cholesky+xtol + PPS_DTAU_MULT=2.0
optimisations applied (see §4). Median L=128 wall time was 8.2h per task,
consistent with the projected savings from the optimisations.

### Per-L runtime statistics

| L   | N_c  | T     | ⟨wall⟩ (s) | median (s) | ⟨ESS⟩/N_c | #collapse |
|-----|------|-------|-----------:|-----------:|----------:|----------:|
| 8   | 2000 | 30    |       431  |       426  |    0.946  |    0      |
| 16  | 1000 | 32    |       720  |       680  |    0.953  |    0      |
| 24  | 800  | 48    |     1,945  |     1,972  |    0.953  |    0      |
| 32  | 500  | 64    |     3,653  |     3,523  |    0.952  |    0      |
| 48  | 300  | 96    |    10,592  |    10,849  |    0.952  |    0      |
| 64  | 200  | 128   |     5,610  |     5,380  |    0.951  |    0      |
| 96  | 150  | 100   |    14,551  |    13,871  |    0.950  |    0      |
| 128 | 100  | 100   |    29,378  |    28,768  |    0.950  |    0      |

L=48 is the most expensive single point (N_c=300 dominates over L=128's
N_c=100). The "wall-time non-monotone in L" pattern is by design (N_c
chosen larger at smaller L for FSS precision).

### Key outstanding actions

1. **Exact-numerics validation at L=8** — review `pps_qj/exact_backend.py`
   (Utku suspects prior AI delegation may have introduced errors). Compare
   cloning output to exact numerics across all ζ at L=8. Two-day project.
   This unlocks high-confidence claims for the writeup.

2. **Multi-Renyi entropy extraction** — modify trajectory worker to save
   S_2, S_3 in addition to S_1. Rerun a few strategic (λ, ζ) points at
   L=128 (~5-10 task-hours each). Tests free-Dirac conformality of log-law
   phase via the c_n = (c/6)(1+1/n) ratio. High value for writeup.

3. **Correlation function diagnostic** — extract ⟨c†_0 c_x⟩ from the
   conditional covariance matrix Γ_t in the cloning steady-state. Power-law
   decay with exponent 1 confirms free Dirac CFT description. 1-2 day project.

4. **LaTeX writeup** — frame the result around c(λ; ζ) phase diagram, NOT
   ν(ζ). The data does not support a ν measurement (see §10).

5. **Two-replica bosonization for QJ** — substantial theoretical project.
   Adapt LMR's Section V derivation to the QJ unraveling. Paper-worth of
   work; flag as "future analytical work" in the thesis.

---

## 3. CODEBASE ARCHITECTURE

### Core simulation

| File | Purpose |
|------|---------|
| pps_qj/gaussian_backend.py | Gaussian free-fermion state, H_eff propagation, entanglement entropy. Optimised with Cholesky+xtol patch (commit 33b78fc, 2026-05-06). |
| pps_qj/cloning.py | Population dynamics algorithm — main simulation engine. ESS/ancestor diagnostics. |
| pps_qj/overlaps.py | Log-overlap between Gaussian states (used by Doob worker). |
| pps_qj/backward_pass.py | Backward ODE for Doob h-transform (not used — Gaussian closure fails at intermediate ζ). |
| pps_qj/exact_backend.py | Exact small-system numerics (L≤16). Utku reviewing for errors. **PRIORITY 1 task.** |

### Parallel infrastructure

| File | Purpose |
|------|---------|
| pps_qj/parallel/grid_pps.py | ALL parameter grids: v2 main, v2 supp, ζ=0, legacy |
| pps_qj/parallel/worker_clone_pps.py | Main cloning worker. **Reads PPS_DTAU_MULT env var** (default 1.0, production was 2.0). |
| pps_qj/parallel/worker_clone_v2_pps.py | Thin shim: routes v2 task IDs to clone worker |
| pps_qj/parallel/worker_clone_v2_supp_pps.py | Thin shim: routes supp task IDs to clone worker |
| pps_qj/parallel/worker_zeta0_pps.py | Deterministic no-click worker (H_eff ground state) |

### SLURM scripts

| Script | Purpose |
|--------|---------|
| slurm/submit_clone_v2_small.sh  | tasks 0..719,    L=8,16,24,  6h  |
| slurm/submit_clone_v2_medium.sh | tasks 720..1199,  L=32,48,   16h  |
| slurm/submit_clone_v2_large.sh  | tasks 1200..1919, L=64,96,128, 48-120h. **Sets PPS_DTAU_MULT.** |
| slurm/submit_clone_v2_supp.sh   | tasks 0..299 (supp), 24h |
| slurm/submit_zeta0_scan.sh      | tasks 0..71 (ζ=0), 30min |
| slurm/submit_validate_dtau.sh   | dτ-doubling validator (see §4) |
| slurm/submit_validate_tcap.sh   | T-cap validator (detector has bug; see §4) |

### Analysis tools

| File | Purpose |
|------|---------|
| pps_qj/tools/validate_dtau_worker.py | Runs N seeds at various dτ multipliers for validation |
| pps_qj/tools/aggregate_dtau.py       | Aggregates dτ validation results (statistical agreement test) |
| pps_qj/tools/validate_tcap_worker.py | T-cap validation — saturation detector is buggy (always NaN), needs rewrite to Welch's t-test |
| analysis/zeta0_benchmark_analysis.md | Complete write-up of ζ=0 results |

---

## 4. CODE CHANGES LOG (most recent first)

### 2026-05-08: Production-run optimisations
**Commits 33b78fc, 2d2ac97:**

- **pps_qj/gaussian_backend.py** [MODIFIED]
  Replaced `slogdet` with Cholesky for PSD survival Gram in
  `gaussian_born_rule_trajectory._fast_branch_norm`. Relaxed brentq xtol
  1e-8→1e-6. M-chip neutral; Habrok OpenBLAS: **18% speedup at L=32**.

- **pps_qj/parallel/worker_clone_pps.py** [MODIFIED]
  Added PPS_DTAU_MULT env-var pathway. Default 1.0 (unchanged behaviour).
  Setting PPS_DTAU_MULT=2.0 doubles the cloning interval and ~halves the
  inner n_steps loop. Validated safe across (L=32, λ ∈ {0.30,0.40,0.50},
  ζ ∈ {0.30,0.50,0.70}) and L=64 cross-check. Statistical agreement of
  ⟨S⟩ within 3σ for all mult ∈ {1.5, 2.0, 3.0}. **~40% speedup**.
  Compound with Cholesky+xtol: ~50% total savings.

- **slurm/submit_clone_v2_large.sh** [MODIFIED]
  Exports PPS_DTAU_MULT=2.0 for production resubmission. Wall time 120h
  for L=128 resubmission, actual median ~8.2h per task.

### 2026-05-07: dτ-doubling and T-cap validators (committed)
- **pps_qj/tools/validate_dtau_worker.py** [NEW]
  Tests dτ multipliers ∈ {1.0, 1.5, 2.0, 3.0} at fixed (L, λ, ζ) over 5
  seeds. Records ⟨S⟩, σ_S, θ, ESS_min, ancestor count, wall time.
- **pps_qj/tools/aggregate_dtau.py** [NEW]
  Statistical agreement test (z-score). All 29 mult>1.0 comparisons
  passed |z|<3. The "FAIL (anc<10)" verdicts in output are over-strict
  (baseline itself fails anc≥5%·N_c at long T due to genealogical
  coalescent; not a valid pass/fail criterion).
- **pps_qj/tools/validate_tcap_worker.py** [NEW]
  T-cap validator. **Saturation detector is buggy (always returns NaN)**.
  Needs rewrite to Welch's t-test (split post-burnin into halves, test
  for mean difference). σ_seeds across 33 realisations is small (CV
  0.1-7%) suggesting current T-caps are conservative. Not tightening for
  thesis; fix detector if time permits.

### 2026-05-06: GPU and Numba investigations (both abandoned)
- **pps_qj/gaussian_backend_jit.py** [NEW — not used in production]
  Numba `@njit` compiled trajectory driver. **0.86×–0.65× slower** than
  baseline. Root cause: LAPACK already optimised; Numba adds overhead.
  File retained for reference.

- **pps_qj/cloning_jax.py** [NEW — not used in production]
  JAX/vmap GPU implementation. **12× slower** on L40S GPU at L=32.
  Root cause: `lax.while_loop` inside `vmap` serialises on GPU due to
  divergent loop termination (Poisson-distributed jump counts).
  File retained for reference.
  **Conclusion: CPU numpy is correct approach.**

### 2026-05-06: Optimisations 1–3 in cloning core (~27% speedup at L=32)
- **pps_qj/gaussian_backend.py** [MODIFIED]
  `gaussian_born_rule_trajectory` gains kwargs: `gamma0_override`,
  `orbitals0_override` — avoids `dataclasses.replace` per-clone per-step.
  `ja_cached`, `jb_cached` — precomputed jump-pair index arrays.

- **pps_qj/cloning.py** [MODIFIED]
  Pre-spawn N_c RNG streams once before the step loop. Jump-pair indices
  precomputed once. Removed `dataclasses.replace` call.
  Measured: L=32 1.51 ms → 1.1 ms (27%), L=64 ~5.4 ms → 5.0 ms (8%).

### 2026-05-05: submit_clone_scan.sh — CPUS_PER_TASK fix
- **slurm/submit_clone_scan.sh** [MODIFIED]
  Added 5th positional argument CPUS_PER_TASK (default 1).
  Worker reads SLURM_CPUS_PER_TASK to set n_workers.

### 2026-05-05: Supplement grids + ζ=0 worker
- **pps_qj/parallel/grid_pps.py** [MODIFIED — major additions]
  Added make_zeta0_grid() and make_clone_v2_supp_grid().
- **pps_qj/parallel/worker_zeta0_pps.py** [NEW]
  Deterministic no-click propagation.
- **slurm/submit_clone_v2_supp.sh**, **submit_zeta0_scan.sh** [NEW]

### 2026-05-04: v2 grid + new observables + checkpointing
- **pps_qj/cloning.py**: CloningResult gained 8 new fields.
- **pps_qj/parallel/worker_clone_pps.py**: checkpoint/resume support.
- **pps_qj/parallel/grid_pps.py**: make_clone_v2_grid() (1920 tasks).

---

## 5. PARAMETER GRIDS (complete reference)

### v2 main grid (1920 tasks — COMPLETE)
- L = [8, 16, 24, 32, 48, 64, 96, 128]
- λ = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.325, 0.35, 0.375,
       0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.60, 0.65, 0.70,
       0.75, 0.80, 0.85, 0.90]  (24 points)
- ζ = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.85, 1.00]
- Output: /scratch/s4629701/pps_qj/pps_clone_v2/clone_XXXXX.npz

### v2 supplement grid (300 tasks — COMPLETE)
- Block A (0..59):  L=[32,48,64,96,128], λ=[0.005,0.01,0.03,0.075], ζ=[0.02,0.05,0.10]
- Block B (60..299): L=[32,48,64,96,128], 24-pt λ, ζ=[0.90,0.95]
- Output: /scratch/s4629701/pps_qj/pps_clone_v2_supp/clone_XXXXX.npz

### ζ=0 no-click grid (72 tasks — 70/72 converged)
- L=[8,16,24,32,48,64,96,128], λ=[0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30]
- Output: /scratch/s4629701/pps_qj/pps_zeta0/zeta0_XXXX.npz

---

## 6. DATA & NUMERICAL RESULTS (post-L=128, with retractions)

### RETRACTION: the earlier ν ≈ 1.33 finding was a spline artifact

The previous analysis (pre-L=128) extracted "ν ≈ 1.33" from a 3-point
FSS fit of Var(S) peak positions using a smoothed-spline peak finder.
With L=128 included this fit gives nonsensical results (ν=8.36 at ζ=0.70,
λ_c=0 hitting fit bound).

Investigation showed: the spline-smoothed peak finder produces a
**different answer** than the raw-data argmax. At ζ=1.00, L=128:
- Spline says peak at λ = 0.408
- Raw data says peak at λ = 0.450

The "monotone drift toward λ_c=0.475 with L^(-1/ν)" the earlier fit
found was an artefact of spline smoothing levels differing between
L's. The correct method is **parabolic interpolation around the
raw-data argmax**. With that method:

| ζ    | L=64   | L=96   | L=128  |
|------|-------:|-------:|-------:|
| 0.85 | 0.440  | 0.429  | 0.458  |
| 1.00 | 0.435  | 0.454  | 0.448  |

**Peak position is plateaued by L=64 — no monotone L^(-1/ν) drift to
fit.** The standard FSS approach via Var(S) peak shift does not work for
this dataset. This is consistent with the absence of a sharp MIPT in the
L→∞ limit (Li-Zhong-Yu 2025; Fava et al. 2023 for charge-conserving
case).

### What DOES work: the effective log-law prefactor c(λ; ζ)

Fit S_{L/2}(L; λ, ζ) ≈ a + c · ln L over L ∈ {32, 48, 64, 96, 128}. The
prefactor c plays the role of an effective CFT central charge. In the
free-Dirac CFT extended-criticality phase of monitored fermions
(Cao-Tilloy-De Luca 2019; Alberton-Buchhold-Diehl 2021), c → 1.

The location where c(λ; ζ) = 1 traces a smooth curve in (λ, ζ) plane —
this is the **proposed phase boundary** for the thesis. Values:

| ζ    | λ where c=1 | λ where c=0.5 | λ where c=0.1 | Var(S) pk L=128 |
|------|------------:|--------------:|--------------:|----------------:|
| 0.02 |       0.070 |         0.096 |         0.259 |              -- |
| 0.05 |       0.075 |         0.107 |         0.300 |              -- |
| 0.10 |       0.096 |         0.158 |         0.291 |              -- |
| 0.15 |       0.154 |         0.192 |         0.320 |              -- |
| 0.20 |       0.187 |         0.249 |         0.332 |              -- |
| 0.30 |       0.237 |         0.286 |         0.369 |              -- |
| 0.50 |       0.334 |         0.389 |         0.472 |              -- |
| 0.70 |       0.370 |         0.447 |         0.572 |              -- |
| 0.85 |       0.365 |         0.456 |         0.568 |           0.503 |
| 1.00 |       0.364 |         0.448 |         0.549 |           0.448 |

The Var(S) peak (where it resolves: ζ ≥ 0.85) sits inside the c-band
between c=1 and c=0.1, confirming both diagnostics measure the same
crossover.

### Crossover sharpness — counter-intuitive ζ dependence

|∂γ_eff/∂λ| at γ_eff = 0.5 measures transition sharpness:

| ζ    | sharpness |
|------|----------:|
| 0.02 |      11.7 |
| 0.05 |      10.1 |
| 0.10 |       6.3 |
| 0.20 |       3.3 |
| 0.30 |       3.6 |
| 0.50 |       1.0 |
| 0.70 |       1.4 |
| 0.85 |       1.5 |
| 1.00 |       1.8 |

**The crossover is sharper at low ζ (strong PPS) and broader at high
ζ (Born-rule)** — the opposite of naive intuition. Physical reason:
- ζ → 0 ensemble is dominated by the deterministic no-click trajectory,
  which has a sharp non-Hermitian "topological → trivial" gap closing.
- ζ → 1 ensemble is the full Born-rule stochastic monitoring, which has
  genuine log-law extended criticality — a gradual form.

This matches the qualitative picture LMR 2025 derive analytically in the
diffusive case (Ising sharp at strong PPS, BKT smooth at ζ → 1).

### Validity of the cloning algorithm

- ESS health: ⟨ESS⟩/N_c = 0.95 across all (L, ζ). No collapse events.
- Born-rule sanity check (ζ=1.00): Binder cumulant (32,48) crossing at
  λ ≈ 0.499 — exact agreement with known Born-rule MIPT λ_c ≈ 0.5.
- theta_mean at ζ=1.00: exactly 0 (correct — no tilting at Born-rule).
- ζ=0 benchmark: λ_c(L) ~ C/√L with C ≈ 1.0 ± 0.05 (matches known
  no-click MIPT scaling).

The cloning correctly samples both boundary measures (Born and no-click),
so the intermediate-ζ data is reliable modulo finite-size effects.

### Supervisor's "Method 2" (dual-state lookahead) — CONCLUSIVELY REFUTED

Dganit proposed an alternative single-trajectory approach using a "tilted
lookahead" stopping condition (PPSQJ.py). Numerical analysis on spin-1/2
and L=2 Kitaev:
- Single-jump waiting time matches thinning (Method 1) trivially (the
  test doesn't distinguish PPS from Born-rule).
- Multi-jump test: Method 2 reproduces **Born-rule** statistics, not
  post-selected. At ζ=0.5, T=20 spin-1/2: Method 2 ⟨N_T⟩ = 9.64, Born =
  9.74, true PPS = 4.55.
- ζ-scan confirms Method 2 tracks Born regardless of ζ.

Analytical reason: Method 2's stopping condition reduces to ‖ψ‖² ≤ r
in the dt → 0 limit, i.e. Born-rule QJ. The "tilted" H_eff^ζ only
modifies the stopping condition at O(dt), not the underlying measure.

The Doob+cloning framework remains necessary. Full analysis in
/mnt/user-data/outputs/dganit_method2_analysis.md (see also
dganit_method2_NT_test.py, dganit_method2_spin_half.py).

### Earlier preliminary results (from pre-v2 dataset; superseded but useful)

- Phase boundary from (32,64) B_L crossings on old grid:
  λ=0.35: ζ_c ≈ 0.377
  λ=0.40: ζ_c ≈ 0.475
  λ=0.50: ζ_c ≈ 0.797
  These are consistent with the c(λ; ζ) = 1 curve from new analysis.
- L=16 B_L data unreliable (S_top uses L/4=4-site regions, too small for
  area-law boundary corrections to cancel). FSS uses L ≥ 32 only.
- S_half alone shows no crossings between L pairs — monotone in L
  everywhere. All B_L crossings come from S_top L-dependence.

---

## 7. KEY DECISIONS & RATIONALE

**Why cloning and not importance sampling (Procedure A):**
Reweighting Born-rule trajectories by ζ^{N_T} retrospectively has exponential
variance collapse. Doob + cloning gives exact sampling of the tilted measure.

**Why the Gaussian Doob approach was abandoned:**
The Gaussian closure approximation fails at intermediate ζ (where the MIPT lives)
— the backwards operator projected onto the Gaussian manifold is insufficiently
accurate. Empirically confirmed at L=16 across all ζ≠1.

**Why GPU acceleration was abandoned:**
JAX/vmap on continuous-time WTMC produces warp divergence on GPU (Poisson
jump counts vary per clone). 12× slowdown vs numpy on L40S at L=32.
CPU numpy with cloning architecture is correct.

**Why Numba JIT was abandoned:**
Hot path is LAPACK-dominated. NumPy already dispatches to optimised BLAS.
Numba adds dispatch overhead; bisection vs Brent's method makes things worse.
Measured 0.86× (L=32) and 0.65× (L=64) — slower than baseline.

**Why the ν-FSS approach was abandoned (NEW, post-L=128):**
The Var(S) peak position plateaus by L=64 with no monotone L^(-1/ν)
drift to fit. This is consistent with the recent consensus (Li-Zhong-Yu
2025; Fava et al. 2023) that monitored free fermion systems have **no
sharp MIPT in the thermodynamic limit** — there is no transition for FSS
to characterise. Earlier "ν ≈ 1.33" was a spline-smoothing artefact.

**Why c(λ; ζ) is the right diagnostic:**
c is the log-law prefactor in S ≈ a + c·ln L, with a physical interpretation
as an effective CFT central charge. The free Dirac CFT has c=1, which is
what the monitored-fermion extended-criticality phase is argued to be.
The curve c(λ; ζ) = 1 traces the cleanest phase boundary in our data,
and matches Var(S) peak locations where they exist.

**Why B_L = S_top × S_half works as a crossing observable but has limitations:**
S_half alone is monotone in L; crossings come from S_top. At L=16,
S_top uses L/4=4-site regions (too small for boundary corrections to
cancel). FSS uses L≥32.

**OBC Friedel oscillations:**
Free-fermion entanglement entropy oscillates with L due to OBC level structure.
L=48 and L=96 are in the "down" oscillation phase relative to L=32,64,128.
The c(λ; ζ) analysis is more robust against this than peak-based FSS
because it fits 5 L values rather than relying on pairwise crossings.

**T-caps and their effect on scaling:**
time_horizon_v2: L≥96→T=100, L≥64→T=150, L≥32→T=200.
This makes L=64 faster per task than L=48 (T=150 vs 200), breaking naive L^α
scaling. True scaling: work ∝ N_c × T × α × L⁴.

**Habrok GPU nodes:** (kept for reference even though abandoned)
Two dedicated interactive GPU nodes: gpu1.hb.hpc.rug.nl and gpu2.hb.hpc.rug.nl,
each with NVIDIA L40S (48 GB). Access via `ssh gpu1.hb.hpc.rug.nl`.
Run `unset SW_STACK_ARCH && module restore` after connecting.
JAX on GPU requires `jax.config.update("jax_enable_x64", True)` before
any jnp operations, otherwise float64/complex128 is silently truncated to
float32.

**Small-ζ large-L tasks are unreliable in v2 main grid:**
At ζ≤0.05 and L≥96, the dominant clones are near-zero-click and need
T ~ L/α to converge — far exceeding the T=100 cap. At-risk task IDs:
1440,1441,1450,1451,1460,1461,1680,1681,1690,1691,1700,1701. Excluded
from FSS fits. The ζ=0 benchmark covers the ζ→0 anchor correctly.

**chi_k as transition detector:**
chi_k = Var(N_T^window)/(L·δτ) is a valid activity susceptibility for ζ<1.
For ζ=1.00 it is NOT a transition signal (it just measures bare Poisson
variance, monotone in λ).

---

## 8. PENDING TASKS (priority order)

1. **[PRIORITY 1] Exact-numerics L=8 validation against cloning.**
   Review `pps_qj/exact_backend.py` (Utku suspects errors from prior AI
   delegation). Run exact L=8 across all ζ. Compare to cloning output.
   Two-day project. **Unlocks confident thesis claims.**

2. **[PRIORITY 2] Add multi-Renyi entropy to trajectory worker.**
   Modify `pps_qj/gaussian_backend.py` to extract S_2, S_3 in addition to
   S_1 (entire eigenvalue spectrum of Γ_subsystem is needed anyway).
   Rerun 3–5 strategic (λ, ζ) points at L=128. Test c_n = (c/6)(1+1/n)
   for free Dirac CFT. **High value for thesis.** ~1-week project.

3. **[PRIORITY 3] Correlation function diagnostic.**
   Extract ⟨c†_0 c_x⟩ from conditional covariance Γ_t in the cloning
   steady-state. Fit decay exponent — should be ≈ 1 in free Dirac CFT.
   1-2 day project.

4. **[PRIORITY 4] LaTeX results section.**
   Frame around c(λ; ζ) phase diagram. Cite Cao-Tilloy-De Luca,
   Alberton-Buchhold-Diehl, Li-Zhong-Yu 2025 for the ζ=1 limit. Cite
   LMR 2025 as the diffusive analogue (no direct ν comparison —
   different model). Note absence of sharp MIPT consistent with recent
   literature.

5. **[OPTIONAL] Fix T-cap validator detector.**
   Replace broken saturation detector in `pps_qj/tools/aggregate_tcap.py`
   with Welch's t-test approach (split post-burnin into halves, test mean
   difference). 30-line change. Not blocking; current T-caps are
   conservative per σ_seeds evidence.

6. **[FUTURE WORK / paper #2] Two-replica bosonization for QJ.**
   Adapt LMR's Section V derivation to the QJ unraveling. Test whether
   the same Ising-to-BKT crossover prediction holds. Substantial
   theoretical project.

---

## 9. LATEX DOCUMENT STATUS

Location: continuousmeasurementslatex/main.tex + sections/

Written sections include:
- Full PPS path-measure framework: ℙ (Born-rule), 𝕈_s (exact PPS), ℝ_ζ (local MC)
- Procedures A, B, C with proofs; RN derivatives; Doob transform
- Gaussian free-fermion formalism (Majorana covariance, jumps, entanglement)
- Doob dynamics in free-fermion model; Gaussian closure/backward ODE
- Failure of Gaussian Doob approach (empirical evidence + analysis)
- Cloning approach: Feynman-Kac/SMC proof, SCGF estimator, ESS, error analysis,
  bias sources, MIPT detection strategy

NOT YET WRITTEN:
- Numerical results section
- Phase diagram figure (use phase_diagram_final.png template)
- c(λ; ζ) effective central charge analysis
- Comparison to Cao-Tilloy-De Luca, Alberton-Buchhold-Diehl, Li-Zhong-Yu
  2025, LMR PRX 2025

Style guide: match existing tone and formatting in the document.
Do NOT use bullet points or bold in prose sections. Equations in standard
LaTeX with align environments. Consistent notation with existing sections.

---

## 10. THEORY NOTES

> **As of 2026-05-12, the theory directory `theory/` contains a substantial
> body of analytical work that supersedes some of the heuristics in this
> section. Start with `theory/STATUS.md` for the current synthesis. The
> notes below remain useful for the pre-bosonization framing.**

### Replica bosonization prediction (pre-L=128 expectation)

Two-replica generator for quantum-jump tilted dynamics analyzed. The marginal
cross-replica operator has Δ=1 at Ising fixed point. BKT structure of RG flow
implies Ising plateau ν≈1 for ζ∈[0, ζ̃_QJ]. Estimated ζ̃_QJ ∈ [0.15, 0.45]
(heuristic; full bosonization needed for precision).

**Caveat (NEW):** This was the pre-L=128 expectation framed in the LMR
language. With the new analysis, the natural language is c(λ; ζ) rather
than ν. The Ising-to-BKT transition LMR predict in the diffusive case
should correspond to a *qualitative change in the shape of the c(λ; ζ)
curve* near ζ̃_QJ. We don't have enough resolution to test this
quantitatively, but the sharpness data (§6) shows the qualitative
behaviour.

### Comparison to Leung-Meidan-Romito 2025 (diffusive case)

Their key result: ν=1 (Ising) plateau for ζ ∈ [0, 0.28], abrupt
deviation, ν=5/3 (BKT) at ζ → 1. **Convention check: their ζ matches
ours** (ζ=1 fully monitored, ζ=0 fully postselected).

Our diffusive analogue would have:
- Ising universality (sharp transition) at strong PPS ⟶ matches our
  "sharper crossover at low ζ" finding qualitatively.
- BKT-like (gradual log-law) at ζ → 1 ⟶ matches our "broader crossover
  at ζ → 1" finding qualitatively.

But LMR's model is **dimerized two-rate measurement** (driven by Δ);
ours is **single-rate measurement** (driven by λ). The transitions are
in different parameter spaces. Direct ν comparison is not appropriate.

### Comparison to Kells, Meidan, Romito 2023

Their model: two non-commuting bond-parity measurements with rates γ
and α, plus unitary hopping w. Three-corner phase diagram (γ,α,w).
Transitions identified via S_top × S_half. Sub-volume scaling (log-law)
critical phase between two topological area-law phases.

Our model is a "single corner" of theirs (no second measurement
channel). The Born-rule limit (ζ=1) should reduce to their fully
stochastic limit. The qualitative phase structure (log to area-law
crossover) matches.

### Comparison to monitored-free-fermion canon

For ζ=1 (our Born-rule limit):
- Cao-Tilloy-De Luca 2019: monitored fermion chain with bond-density
  measurements → "volume law entanglement is destroyed by arbitrarily
  weak measurement". Log-law extended criticality at small γ. **Matches
  our ζ=1 result.**
- Alberton-Buchhold-Diehl 2021: same model, identifies BKT-type
  transition from log-law to area-law. c ≈ 1 free Dirac CFT in the
  critical phase. **Matches our ζ=1 result: c → 1 at the crossover
  centre.**
- Li-Zhong-Yu 2025 review: "in monitored free fermion 1D systems, the
  volume-law phase disappears at any measurement rate, leaving no
  MIET... significant finite-size effects lead to a residual logarithmic
  scaling". **Explains why our standard FSS fails — there is no MIPT to
  characterise.**
- Fava et al. 2023: nonlinear sigma model for monitored fermions with
  conservation laws — confirms absence of MIET in thermodynamic limit.

### Supervisor's reduced-rate proposal — incorrect

Using reduced rate ζλ_t with unmodified H_eff generates the locally
renormalized measure ℝ_ζ (NOT the target 𝕈_s). The RN derivative
includes a path-dependent compensator e^{(1-ζ)Λ_T}. **Confirmed
numerically** that the variant Method 2 (dual-state lookahead) reproduces
Born-rule statistics, not post-selected ones (see §6 above).

### Open analytical questions

> **Status updates as of 2026-05-12. See `theory/STATUS.md` and the chiral-vertex
> documents for the analytical work; the questions below have all been partially
> addressed.**

1. Does the QJ partial-PPS measure admit a bosonized field-theoretic
   description analogous to LMR's diffusive case? — **Yes, constructed
   in `theory/qj_two_replica_derivation.md`.** The two-replica generator
   for the QJ unraveling has a single cross-replica click vertex
   $V_j = d^{(+,1)}(d^{(-,1)})^\dagger d^{(+,2)}(d^{(-,2)})^\dagger$.

2. Is the cross-replica vertex relevant or irrelevant? — **Marginal and
   chiral** (`theory/qj_marginal_chiral_correction.md`). The vertex is
   a 4-fermion product (dim 2 in 1+1D) and is purely left-moving at
   half-filling (lattice fact: $d_j$ has zero coupling to $k=+\pi/2$).
   Chiral + marginal means the perturbation is *exactly marginal* — no
   second-order RG flow because the self-OPE produces only
   antiholomorphic operators that integrate to zero.

3. What does this predict for $c_{\rm eff}(\lambda, \zeta)$? — **In the
   thermodynamic limit, $c_{\rm eff}$ is $\zeta$-independent in the
   critical phase.** The phase diagram in the $(\lambda, \zeta)$ plane
   has a single vertical critical line $\lambda = \lambda_c(\zeta=1)
   \approx 0.364$ — no $\zeta$-dependent shift.

4. How does this reconcile with the observed sharp drop in
   $\lambda(c=1)$ at $\zeta \lesssim 0.5$? — **Finite-$L$ artifact from
   sub-leading non-chiral lattice corrections.** Prediction:
   $\lambda_c(\zeta, L) - \lambda_c(\zeta=1, L) \sim C(\zeta)/L^2$.
   At $\zeta = 0.30$, $L=128$ shift is $-0.127$; predicted shift at
   $L=256$ is $\sim -0.032$ (4× smaller). **Testable with one more
   production run at $L=256$**.

5. Does the absence of sharp MIPT in monitored free fermions (Li 2025;
   Fava 2023) extend to all ζ? — **The chirality picture says yes**:
   the QJ unraveling has a single conformal phase throughout the
   $(\lambda, \zeta)$ plane (modulo finite-$L$ corrections). The
   Renyi reruns (running now) test this directly via the universal
   CFT ratios $c_2/c_1 = 3/4$, $c_3/c_1 = 2/3$.

6. **Open**: full strong-PPS RG analysis (the analog of LMR's Section V.C).
   Two alternative scenarios remain on the table if the $1/L^2$ scaling
   from question 4 fails: (a) marginally-relevant lattice corrections
   drive a BKT-type transition at finite $\zeta_c$; (b) "chiral pinning"
   produces an intermediate $c=1/2$ phase at strong PPS. Distinguished
   by the directly-fit $c_{\rm eff}$ at strong-PPS Renyi test points
   (is it $1$, $1/2$, or $0$?).

---

## 11. KEY ANALYSIS PRODUCTS (where to find them)

In `/mnt/user-data/outputs/v3_full_analysis/`:
- `S_fanning_all.png` — S(λ) at every ζ, all L (10 panels)
- `S_fanning_select.png` — 4 representative ζ values, big panels
- `density_all.png` — S/L (volume-law diagnostic)
- `S_over_logL.png` — S/ln L (log-law diagnostic)
- `S_vs_logL.png` — S vs ln L (linear → log-law signature)
- `S_vs_L_loglog.png` — log-log plot
- `c_eff_curves.png` — c(λ; ζ) effective central charge
- `scaling_form_map.png` — which functional form wins per (λ, ζ)
- `phase_diagram_final.png` — unified phase diagram with all locators
- `runtime.png` — production-run sanity check

In `/mnt/user-data/outputs/`:
- `PHYSICS_ANALYSIS.md` — comprehensive theoretical writeup (this is the
  document to read first when picking up the project)

In `/mnt/user-data/outputs/final_FSS/`:
- `REPORT.md` — retraction of ν ≈ 1.33 finding
- Earlier FSS attempt plots (peaks, scaling collapse, etc.)

In `/mnt/user-data/outputs/`:
- `dganit_method2_analysis.md` — refutation of Method 2 proposal
- `dganit_method2_NT_test.py` — runnable reproduction script
- `dganit_method2_spin_half.py` — first-WT distribution test

In `theory/` (analytical work — added 2026-05-11/12):
- `STATUS.md` — **read this first.** Reading guide and synthesis of the
  bosonization analysis. Updated with the chirality result and the
  $1/L^2$ finite-$L$ prediction.
- `qj_two_replica_derivation.md` — rigorous construction of the QJ
  two-replica generator $\mathcal{L}_\zeta^{(2)}$.
- `qj_one_minus_zeta_expansion.md` — sharpened bosonization;
  identifies $V_j$ as a single vertex but with wrong dimension
  ($\Delta=4$ claimed, corrected to 2 in `qj_marginal_chiral_correction.md`).
- `qj_chiral_vertex_result.md` — chirality of $V_j$ established
  (algebraic + microscopic). Dimension claim ($\Delta=4$) is superseded.
- `qj_marginal_chiral_correction.md` — **load-bearing for the
  current analytical claim.** Corrects $\Delta(V_j) = 2$ (marginal),
  rigorously verifies the chirality at the lattice level, shows
  chiral-marginal vertices are exactly marginal (no second-order
  flow), and predicts $\lambda_c$ shift $\sim 1/L^2$ scaling.
- `qj_bosonization_calculation.md` — earlier-draft, longer
  bosonization document with self-critique and channel decomposition
  of $V_j$. Largely superseded but worth reading for the empirical
  comparisons in §10.
- `qj_algorithms_comparison.md` — clarifying note on the equivalence
  between waiting-time (continuous-time) and stepped-$\Delta t$
  (discrete-time) quantum-jump algorithms. Includes the relationship
  to Dganit's implementation.
- `two_replica_QJ_PPS.md` — earlier draft, superseded.

---

## 12. RECENT SESSION LOG (chronological, newest first)

### 2026-05-12: algorithms comparison note + CONTEXT.md update (this session)

Added `theory/qj_algorithms_comparison.md` clarifying the relationship
between Utku's continuous-time waiting-time algorithm and Dganit's
discrete-time stepped-$\Delta t$ algorithm. Short summary: they target
the same stochastic Schrödinger equation; the waiting-time algorithm is
the $\Delta t \to 0$ limit of the stepped algorithm. The stepped
algorithm has $\mathcal{O}(\Delta t)$ bias; ours has no temporal-grid
bias, only brentq tolerance ($10^{-6}$). For our specific problem (free
fermions, per-jump PPS tilt, large $T$ between clicks), the waiting-time
algorithm is both cleaner and faster.

Also updated this CONTEXT.md to point at the new theory directory and
include explicit startup instructions for continuation chats.

### 2026-05-12: chirality + dimension correction

Two new theory documents, plus a sharpened prediction.

**Established:**
- The bond annihilation operator $d_j$ at half-filling is **purely
  left-moving** at the lattice level. In momentum space, $d_j$ has
  coefficient $g_c(k) = \tfrac{1}{2}(1 + ie^{ik})$ on $c_k$; at
  $k = +\pi/2$ this is $1 + i\cdot i = 0$ exactly. Convention-independent.
- The cross-replica click vertex $V_j$ (product of 4 $d$ operators) has
  scaling dimension 2 (marginal in 1+1D), not 4. The previous
  $\Delta = 4$ claim came from a factor-of-2 normalization mismatch
  between Giamarchi's non-canonical fields and standard chiral CFT
  bosons. Direct fermion count: $4 \times 1/2 = 2$.
- A chiral marginal operator's self-OPE produces only chiral
  (antiholomorphic) operators, all of which integrate to zero in 2D
  Euclidean. So the perturbation $g_\zeta \int V_j$ is **exactly
  marginal** — no flow at any order.

**Sharpened prediction:** in the thermodynamic limit,
$c_{\rm eff}(\lambda; \zeta) = c_{\rm eff}(\lambda; \zeta = 1)$ for every
$\zeta > 0$ in the critical phase. The phase boundary $\lambda_c$ is
independent of $\zeta$ — a single vertical line in the $(\lambda, \zeta)$
phase diagram, not a curve.

**Falsifiable finite-$L$ prediction:** observed shift $\lambda_c(\zeta, L) - \lambda_c(\zeta=1, L) \sim C(\zeta)/L^2$
from sub-leading non-chiral lattice corrections. At $\zeta = 0.30$,
$L = 128$ shift is $-0.127$; at $L = 256$ predicted shift is $\sim -0.032$
(4× smaller). Testable with one more production run.

### 2026-05-11 (evening): bosonization framework

First-pass bosonization of the QJ two-replica generator. Established
the cross-replica vertex $V_j \sim \exp[2i(\Theta_D^\rho - \Phi_D^\rho)]$
in the inter-replica difference mode. Initial dimension claim
($\Delta = 4$, irrelevant) corrected the following day. Documents:
`qj_two_replica_derivation.md`, `qj_one_minus_zeta_expansion.md`,
`qj_chiral_vertex_result.md`.

### 2026-05-11 (afternoon): refutation of Method 2

Conclusive numerical refutation of Dganit's "Method 2" (dual-state
lookahead with reduced rate $\zeta\lambda_t$ and unmodified $H_{\rm eff}$).
The multi-jump $N_T$ test at $\zeta = 0.5$, $T = 20$ gave
$\langle N\rangle_{\rm M2} = 9.64$ vs Born $9.74$ vs true PPS $4.55$.
Method 2 reproduces Born-rule statistics, not PPS. Documents in
`outputs/dganit_method2_*`.

### 2026-05-08 to 2026-05-10: L=128 production

L=128 resubmission completed under the Cholesky + xtol +
PPS_DTAU_MULT=2.0 optimisations. All 1920 tasks finished, no
collapses. Aggregate file `agg_final.pkl`. The phase diagram (§6)
became the load-bearing result of the project.

---

## 13. CURRENTLY IN FLIGHT

As of 2026-05-12:

1. **Renyi entropy reruns on Habrok** (30 strategic tasks):
   6 $(\lambda, \zeta)$ test points × 5 system sizes
   $L \in \{32, 48, 64, 96, 128\}$. Test points:
     - $(0.30, 1.00), (0.35, 1.00), (0.45, 1.00)$ — Born-rule critical region
     - $(0.325, 0.50), (0.50, 0.50)$ — intermediate-PPS
     - $(0.10, 0.20)$ — strong-PPS

   Job submitted via `slurm/submit_renyi_targets.sh`. When complete:
   ```
   python scripts/analyze_renyi_targets.py /scratch/s4629701/pps_qj/pps_clone_renyi/
   ```
   The predictions to test:
     - $c_2/c_1 = 3/4$, $c_3/c_1 = 2/3$ at every test point if the
       chirality picture holds (all six points in one conformal phase).
     - At the strong-PPS test points $(0.10, 0.20)$ and $(0.325, 0.50)$:
       does the directly-fit $c_{\rm eff}$ equal $1$ (single conformal
       phase, chirality picture), $1/2$ (chiral pinning intermediate
       phase), or near $0$ (full area law, chirality picture fails)?

2. **Email to Dganit re: Method 2 refutation** — not yet sent. Diplomatic
   variant drafted (cf. `outputs/dganit_method2_email_*`). Pending Utku's
   choice and send.

3. **Exact-numerics validation at L=8** — pending (Utku's planned
   personal-review work).
