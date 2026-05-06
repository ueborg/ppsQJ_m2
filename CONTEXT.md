# PROJECT CONTEXT — ppsQJ_m2
<!-- Last updated: 2026-05-06 -->
<!-- Update this file at the end of every significant work session. -->

---

## HOW TO USE THIS FILE (read first)

This file is the single source of truth for project context across Claude chat
sessions.  At the start of any new chat, read this file with Desktop Commander:

    "Please read CONTEXT.md using Desktop Commander, then we can continue."

After any session that changes code, results, or decisions, update the relevant
sections and commit:

    git add CONTEXT.md && git commit -m "context: update [brief description]"

---

## 1. PROJECT OVERVIEW

**Goal:** Compute ν(ζ) vs ζ — how the MIPT critical exponent and critical point
shift as a function of partial post-selection (PPS) parameter ζ = e^{-s} ∈ (0,1].

**Model:** 1D Kitaev-type hopping chain, L sites, OBC.
  H = w Σ (c†_j c_{j+1} + h.c.),  jump operators P_j = d†_j d_j
  λ = α/(α+w),  w = 1-λ,  parametrised by λ ∈ (0,1)

**Physical setup:**
- ζ=1: standard Born-rule MIPT, known λ_c ≈ 0.5 for this model
- ζ=0: fully postselected (no-click), λ_c(L) ~ C/√L with C ≈ 1.0
- ζ ∈ (0,1): PPS measure via Doob h-transform / cloning algorithm
- Central prediction: λ_c(ζ) traces a curve from λ_c(0)→0 to λ_c(1)≈0.5

**Supervisor:** Dganit Meidan (RUG).  Extension of Kells, Meidan, Romito
SciPost Phys. 14, 031 (2023) [diffusive unraveling] to quantum-jump regime.

**Repository:** github.com/ueborg/ppsQJ_m2
**Local path:** /Users/catlover1337/Documents/ppsQJ_m2/
**HPC:** Habrok cluster, RUG, user s4629701, interactive node interactive2
**HPC project dir:** ~/pps_qj/   scratch: /scratch/s4629701/pps_qj/
**Venv:** ~/venvs/pps_qj/
**Workflow:** edit locally → push GitHub → pull on Habrok (no SSH key for direct push)

---

## 2. CURRENT STATE (as of 2026-05-06)

### Jobs running / pending

| Job | Tasks | Status | Output dir |
|-----|-------|--------|------------|
| v2 small  | 0..719   (L=8,16,24)   | ✅ COMPLETE | pps_clone_v2/ |
| v2 medium | 720..1199 (L=32,48)   | ✅ COMPLETE | pps_clone_v2/ |
| v2 large  | 1200..1919 (L=64,96,128) | 🔄 RUNNING — L=64✅ L=96 174/240 L=128 ~8/240 | pps_clone_v2/ |
| v2 supp   | 0..299   (low-λ + large-ζ) | ✅ COMPLETE | pps_clone_v2_supp/ |
| ζ=0 scan  | 0..71    (deterministic)  | ✅ COMPLETE (70/72 converged) | pps_zeta0/ |
| benchmark | 6 tasks  (timing test)    | ⏳ SUBMITTED, not yet returned | benchmark_*/ |
| old L=128 (job 28808838) | tasks 1020..1274 (L=128, 255 tasks) | 🔄 RUNNING — 37/255 complete, CPUS_PER_TASK=5 fix applied | pps_clone_scan_v2/ |

### Key outstanding actions

1. **L=128 resubmission** — the v2 large job will hit its 48h wall time with
   only ~8/240 L=128 tasks done. After it terminates, resubmit with 120h wall
   time.  The idempotency guard means only uncompleted tasks rerun.
   Estimated L=128 total: ~114h (empirically calibrated, see §6).

2. **Collect benchmark results** — once the benchmark job finishes (~45 min),
   read /scratch/s4629701/pps_qj/benchmark_*/benchmark_results.txt and update
   the L=128 time estimate.

3. **Collect final L=96 data** — L=96 finishes ~6:44 AM on 2026-05-07.
   Regenerate summary_all.csv after the large job terminates.

4. **ζ=0 two remaining non-converged tasks** — L=96, λ=0.02 and λ=0.03
   (task_ids 55 and 56). Suspected spectral resonance α ≈ Δε_H ≈ πw/L ≈ 0.032.
   Not critical; resubmit at T=20000 if precision needed.

5. **FSS analysis** — once L=128 is complete, do full finite-size scaling
   collapse using L=32,64,128 (cleanest subset, avoids OBC Friedel artefacts
   at L=48 and L=96).

6. **Write results section in LaTeX** — cloning methodology and preliminary
   phase diagram.

---

## 3. CODEBASE ARCHITECTURE

### Core simulation

| File | Purpose |
|------|---------|
| pps_qj/gaussian_backend.py | Gaussian free-fermion state, H_eff propagation, entanglement entropy |
| pps_qj/cloning.py | Population dynamics algorithm — main simulation engine |
| pps_qj/overlaps.py | Log-overlap between Gaussian states (needed by Doob worker) |
| pps_qj/backward_pass.py | Backward ODE for Doob h-transform (not used in current runs) |
| pps_qj/exact_backend.py | Exact small-system numerics (L≤16, Utku reviewing for errors) |

### Parallel infrastructure

| File | Purpose |
|------|---------|
| pps_qj/parallel/grid_pps.py | ALL parameter grids: v2 main, v2 supp, ζ=0, legacy |
| pps_qj/parallel/worker_clone_pps.py | Main cloning worker (task dispatch, checkpointing, aggregation) |
| pps_qj/parallel/worker_clone_v2_pps.py | Thin shim: routes v2 task IDs to clone worker |
| pps_qj/parallel/worker_clone_v2_supp_pps.py | Thin shim: routes supp task IDs to clone worker |
| pps_qj/parallel/worker_zeta0_pps.py | Deterministic no-click worker (H_eff ground state) |

### SLURM scripts

| Script | Purpose |
|--------|---------|
| slurm/submit_clone_v2_small.sh  | tasks 0..719,    L=8,16,24,  6h  |
| slurm/submit_clone_v2_medium.sh | tasks 720..1199,  L=32,48,   16h  |
| slurm/submit_clone_v2_large.sh  | tasks 1200..1919, L=64,96,128, 48h |
| slurm/submit_clone_v2_supp.sh   | tasks 0..299 (supp), 24h |
| slurm/submit_zeta0_scan.sh      | tasks 0..71 (ζ=0), 30min |
| slurm/submit_benchmark.sh       | 6 timing tasks, 45min |

### Analysis

| File | Purpose |
|------|---------|
| analysis/zeta0_benchmark_analysis.md | Complete write-up of ζ=0 results (for agent review) |
| analysis/pps_phase_diagram.py | Phase diagram plotting (in progress) |

---

## 4. CODE CHANGES LOG (most recent first)

### 2026-05-06: GPU and Numba investigations (both abandoned)

- **pps_qj/gaussian_backend_jit.py** [NEW — not used in production]
  Numba `@njit` compiled trajectory driver. Replaced `scipy.optimize.brentq`
  with bisection (~25–30 iterations vs 5–8 for Brent). Benchmark on Habrok:
  L=32: 1.1 ms/call (original) → 1.3 ms/call (JIT) = **0.86× slower**.
  L=64: 5.0 ms/call → 7.8 ms/call = **0.65× slower**.
  Root cause: LAPACK (slogdet, QR, eigh) already optimised; Numba adds overhead
  without replacing LAPACK. Bisection makes things worse. Not integrated into
  run_cloning. File retained for reference.

- **pps_qj/cloning_jax.py** [NEW — not used in production]
  JAX/vmap GPU implementation. All N_c clones batched as tensor dimension,
  `lax.while_loop` for WTMC inner loop, `jax.config.update("jax_enable_x64",True)`
  required (otherwise silent float32 truncation destroys physics).
  Benchmark on Habrok L40S GPU: L=32 numpy 4.3 s → JAX GPU **415 s** (12× slower).
  Root cause: `lax.while_loop` inside `vmap` serialises on GPU due to divergent
  loop termination (jump counts are Poisson-distributed, variable per clone →
  GPU warp divergence). Fixing this requires replacing continuous-time WTMC
  with a Trotterised fixed-step scheme, which changes the physics and is a
  separate project. File retained for reference.
  **Conclusion: CPU numpy is the correct approach for this algorithm.**

- **pps_qj/tools/benchmark_optimizations.py** [NEW]
  Timing + correctness harness for optimisations 1–3.
  Run: `python -m pps_qj.tools.benchmark_optimizations --L 32 64 --with-jit`

- **pps_qj/tools/benchmark_jax.py** [NEW]
  GPU vs numpy speedup + statistical agreement test.
  Run: `python -m pps_qj.tools.benchmark_jax --L 32 64 128`

### 2026-05-06: Optimisations 1–3 in cloning core (~27% speedup at L=32)

- **pps_qj/gaussian_backend.py** [MODIFIED]
  `gaussian_born_rule_trajectory` gains kwargs:
    `gamma0_override`, `orbitals0_override` — avoids `dataclasses.replace`
    per-clone per-step (was N_c × n_steps frozen dataclass allocations).
    `ja_cached`, `jb_cached` — precomputed jump-pair index arrays passed
    in from the outer loop rather than rebuilt per call.
  Added `from typing import Optional`.

- **pps_qj/cloning.py** [MODIFIED]
  Pre-spawn N_c RNG streams once before the step loop (was re-spawning
  N_c Generator objects every step). Jump-pair index arrays precomputed once
  per run and passed to trajectory driver. Removed `dataclasses.replace` call.
  Removed `from dataclasses import replace` import (no longer needed).
  Measured speedup on Habrok: L=32 1.51 ms/call → 1.1 ms/call (**27% faster**),
  L=64 ~5.4 ms/call → 5.0 ms/call (~8% faster).

### 2026-05-05: submit_clone_scan.sh — CPUS_PER_TASK fix

- **slurm/submit_clone_scan.sh** [MODIFIED]
  Added 5th positional argument `CPUS_PER_TASK` (default 1, backward-compatible).
  `N_PARALLEL=$(( 120 / CPUS_PER_TASK ))` replaces hardcoded `N_CORES=120`.
  `xargs -P ${N_PARALLEL}` replaces `xargs -P ${SLURM_NTASKS}`.
  **Why this matters:** The worker reads `SLURM_CPUS_PER_TASK` to set `n_workers`.
  With the old script (cpus-per-task=1), all L=128 tasks ran n_workers=1 → 5
  realisations serial → critical-point tasks needed ~110h (far over any cap).
  With CPUS_PER_TASK=5: 24 parallel task slots × 5 workers each → critical-point
  tasks complete in ~22h/5 = ~4.4h. Usage for L=128:
  `bash slurm/submit_clone_scan.sh 1020 1274 <outdir> 48:00:00 5`
- **slurm/submit_benchmark.sh** [NEW]
  Reruns 6 completed production task_ids in a throwaway dir, compares wall times
  to production .npz values. Tests: (1) hardware match interactive vs compute
  node, (2) consistency of L^4 scaling model. Prints calibrated L=128 estimate.

### 2026-05-05: Supplement grids + ζ=0 worker
- **pps_qj/parallel/grid_pps.py** [MODIFIED — major additions]
  Added make_zeta0_grid() (72 tasks: L=[8..128] × 9 λ values).
  Added make_clone_v2_supp_grid() (300 tasks):
    Block A (0..59):  low-λ small-ζ — L=[32..128], λ=[0.005,0.01,0.03,0.075]
    Block B (60..299): large-ζ — L=[32..128], 24-pt λ, ζ=[0.90,0.95]
  time_horizon_zeta0() went through 3 tuning passes; final formula:
    T = min(20000, max(15L, 200/α))
  
- **pps_qj/parallel/worker_zeta0_pps.py** [NEW]
  Deterministic no-click propagation: scipy.linalg.expm + QR renormalization.
  Convergence check: |S(T) - S(T/2)| < 0.05.
  Output: zeta0_XXXX.npz with S_final, S_half, converged, log_norm_T.

- **pps_qj/parallel/worker_clone_v2_supp_pps.py** [NEW]
  Thin shim over worker_clone_pps routing to task_params_clone_v2_supp.

- **slurm/submit_clone_v2_supp.sh** [NEW]
- **slurm/submit_zeta0_scan.sh** [NEW]

### 2026-05-04: v2 grid + new observables + checkpointing
- **pps_qj/cloning.py** [MODIFIED]
  CloningResult dataclass gained 8 new fields:
    n_T_mean, chi_k, S_var, covar_Sk (scalar summaries)
    n_T_mean_history, n_T_sq_history, S_sq_history, covar_Sn_history (arrays)
  Entropy block now computes all four observables via weighted dot products.
  Summary stats computed post-burnin using existing S_history infrastructure.
  No breaking changes — all new fields have defaults.

- **pps_qj/parallel/worker_clone_pps.py** [MODIFIED — near-complete rewrite]
  Added checkpoint/resume: serial tasks write clone_XXXXX_partial.pkl after
  each realization; resumes on restart; deleted on completion.
  New aggregate fields in .npz: n_T_mean/err, chi_k_mean/err, S_var_mean/err,
  covar_Sk_mean/err, plus per-realisation arrays.
  Parallel mode (n_workers>1): unchanged (no partial saving).

- **pps_qj/parallel/grid_pps.py** [MODIFIED]
  Added make_clone_v2_grid() (1920 tasks):
    L=[8,16,24,32,48,64,96,128], 24 λ values, 10 ζ values.
    λ grid: [0.02,0.05] + linspace(0.10,0.90,17) + [0.325,0.375,0.425,0.475,0.525]
    ζ grid: [0.02,0.05,0.10,0.15,0.20,0.30,0.50,0.70,0.85,1.00]
    N_c: 2000/1000/800/500/300/200/150/100 per L=[8..128]
    T caps: L≥96→100, L≥64→150, L≥32→200, else uncapped

- **pps_qj/parallel/worker_clone_v2_pps.py** [NEW] — shim for v2 grid

- **slurm/submit_clone_v2_small/medium/large.sh** [NEW]

---

## 5. PARAMETER GRIDS (complete reference)

### v2 main grid (1920 tasks)
- L = [8, 16, 24, 32, 48, 64, 96, 128]
- λ = [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.325, 0.35, 0.375,
       0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55, 0.60, 0.65, 0.70,
       0.75, 0.80, 0.85, 0.90]  (24 points)
- ζ = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 0.85, 1.00]
- Output: /scratch/s4629701/pps_qj/pps_clone_v2/clone_XXXXX.npz

### v2 supplement grid (300 tasks)
- Block A (0..59):  L=[32,48,64,96,128], λ=[0.005,0.01,0.03,0.075], ζ=[0.02,0.05,0.10]
- Block B (60..299): L=[32,48,64,96,128], 24-pt λ, ζ=[0.90,0.95]
- Output: /scratch/s4629701/pps_qj/pps_clone_v2_supp/clone_XXXXX.npz

### ζ=0 no-click grid (72 tasks)
- L=[8,16,24,32,48,64,96,128], λ=[0.01,0.02,0.03,0.05,0.075,0.10,0.15,0.20,0.30]
- Output: /scratch/s4629701/pps_qj/pps_zeta0/zeta0_XXXX.npz

---

## 6. DATA & NUMERICAL RESULTS

### ζ=0 benchmark (70/72 converged)
- Confirms λ_c(L) ~ C/√L scaling (topological → trivial MIPT)
- **C ≈ 1.0 ± 0.05** from L=128 susceptibility peak (λ̂_c = 0.0905)
- Non-converged: L=96 λ=0.02 and 0.03 (spectral resonance α≈Δε_H≈0.032)
- Reliable FSS sizes: L=32, 64, 128 (L=48, 96 in OBC Friedel "down phase")
- L=24: non-monotone peak at λ=0.03 (OBC artefact, exclude from FSS)
- Full data table + analysis: analysis/zeta0_benchmark_analysis.md

### v2 main grid (completed: L=8–64 full, L=96 174/240, L=128 ~8/240)
- **Born-rule validation (ζ=1.00):** Binder cumulant (32,48) crossing at λ≈0.499
  — exact agreement with known Born-rule λ_c ≈ 0.5. Simulation confirmed correct.
- **ESS health:** ESS/N_c = 0.85–0.92 throughout. No population collapse.
- **theta_mean at ζ=1.00:** exactly 0 (correct — no tilting at Born-rule).
  chi_k at ζ=1.00 peaks at λ=0.90 (grid boundary), not a transition signal.
- **Qualitative phase diagram** from (32,48) Binder crossings:
  ζ=0.30: λ_c≈0.236, ζ=0.50: λ_c≈0.339, ζ=0.70: λ_c≈0.481, ζ=1.00: λ_c≈0.499
  Note: ζ≤0.20 Binder crossings unreliable at L=32,48 (OBC oscillations)
- **Block A (supp) surprise:** At λ=0.01, ζ=0.02:
  S(128)=23.2 — superlinear growth, volume-law-like deep in topological phase
- **Block B (supp, large-ζ):** (32,48) crossings at ζ=0.90,0.95 ≈ 0.50 (unchanged
  from ζ=1.00) — L=32,48 pair too small to resolve near-Born slope A.
  Need L=128 data for this.

### Runtime calibration
- Physical scaling: work ∝ N_c × T × α × L⁴
- Empirical calibration factor vs physical model: ×1.38 (from 85 tasks observed)
- **L=128 total estimate: ~114h** (empirically calibrated, 24 parallel workers)
- L=128 will need dedicated resubmission with 120h wall time
- Interactive vs SLURM hardware ratio: PENDING (benchmark job running)

### Preliminary FSS results from old pps_clone_scan_v2 dataset
**Note:** This is an older, coarser dataset (L=8,16,32,64, 17 λ values, 15 ζ values,
output at pps_clone_scan_v2/). It has been superseded by the v2 main grid but
contains useful preliminary diagnostics.

**B_L crossing analysis:**
Non-monotone crossings observed: the (16,32) crossing is systematically displaced
*upward* relative to both (8,16) and (32,64) pairs.

Diagnostic (Test 1): checked whether S_half has crossings between L values.
**Result: S_half has essentially NO crossings between any L pair at any λ.**
This means ALL crossing behaviour in B_L comes from S_top, not S_half.
Since B_L = S_top × S_half, and S_half is monotone in L everywhere, the
crossing location is set entirely by S_top's L-dependence.

**Root cause of (16,32) non-monotonicity:** S_top is the Kitaev-Preskill
combination evaluated on four contiguous L/4-site regions. At L=16, each region
has only 4 sites — too small for area-law boundary corrections to cancel.
At L=32 (8 sites), L=64 (16 sites) the combination works as intended.
L=16 B_L data is therefore unreliable for FSS; the (32,64) crossing is the
cleanest current estimate of ζ_c(λ).

**Phase boundary from (32,64) B_L crossings:**
  λ=0.35: ζ_c ≈ 0.377
  λ=0.40: ζ_c ≈ 0.475
  λ=0.50: ζ_c ≈ 0.797
  λ≤0.30: system volume-law throughout (no crossing accessible)
  λ≥0.55: system area-law throughout (no crossing at these L)

**Preliminary ν estimates from FSS collapse (L=32, 64 only):**
  λ=0.35: ν ≈ 1.316, 2×-min interval [1.064, 2.154]
  λ=0.40: ν ≈ 1.368, 2×-min interval [1.034, 2.665]
  λ=0.50: ν ≈ 1.168, 2×-min interval [0.886, 2.020]
The residual minimum is very shallow with only two system sizes — wide intervals
are expected. The lower bound ν > 1 is reliable; upper bound is not.
Adding L=128 is necessary for a defensible published estimate.

**Saturation check (L=32):**
Ran cloning with T = 3×T_prod, n_burnin_frac=0, recorded full S_history.
Result: system saturates by t ≈ 5–10 in all parameter regimes (area-law,
volume-law, near critical). T_prod = 64 at L=32 provides 6–10 saturation
times of stationary averaging. T is sufficient; no need to extend.

**Time parameterisation:**
Code enforces α + w = 1 always (normalised energy scale). One unit of time = 1.
δτ = 1/(2α(L-1)) = mean inter-jump waiting time for the whole chain.
T_prod = max(30, 2L, 5/α). At L=32, α=0.4: T=64, δτ≈0.04, n_steps≈1600,
expected jumps ≈ 1587. The continuous-time WTMC is exact (not discretised):
jump times sampled by inverting survival function via Brent's method (brentq).

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
The cloning trajectory driver uses a continuous-time WTMC with a `while_loop`
whose iteration count is data-dependent (Poisson-distributed jump count per clone).
Under `jax.vmap` on GPU, this causes warp divergence: threads whose clones finish
early idle while the slowest clone in their warp finishes, effectively serialising
the computation. Measured 12× slowdown vs numpy on Habrok L40S GPU at L=32.
Fixing this requires replacing the continuous-time WTMC with a Trotterised
fixed-step scheme — a different algorithm requiring fresh validation.
CPU numpy with the cloning architecture is the correct approach.

**Why Numba JIT was abandoned:**
The hot path in `gaussian_born_rule_trajectory` is dominated by LAPACK calls
(slogdet, QR, eigh on L×L and 2L×L matrices). NumPy already dispatches these
to optimised BLAS/LAPACK. Numba adds dispatch overhead without improving the
LAPACK calls, and replacing `scipy.optimize.brentq` (5–8 iterations) with
bisection (25–30 iterations) more than offsets any loop overhead savings.
Measured 0.86× (L=32) and 0.65× (L=64) speedup — i.e., slower.

**Why B_L = S_top × S_half works as a crossing observable but has limitations:**
S_half alone is monotone in L everywhere (no crossings), so it cannot locate
the phase boundary. S_top changes sign/ordering across the transition, producing
crossings in B_L = S_top × S_half. However, S_top uses the Kitaev-Preskill
combination on L/4-site regions — at L=16 (4 sites per region) finite-size
boundary corrections dominate and S_top is unreliable. FSS should use L≥32.

**Habrok GPU nodes:**
Two dedicated interactive GPU nodes: gpu1.hb.hpc.rug.nl and gpu2.hb.hpc.rug.nl,
each with NVIDIA L40S (48 GB). Access via `ssh gpu1.hb.hpc.rug.nl`.
Run `unset SW_STACK_ARCH && module restore` after connecting.
For SLURM GPU jobs, use `--gres=gpu:1` (not `--gpus-per-node`, known issue).
JAX on GPU requires `jax.config.update("jax_enable_x64", True)` before any
jnp operations, otherwise float64/complex128 is silently truncated to float32.
The Gaussian closure approximation fails at intermediate ζ (where the MIPT lives)
— the backwards operator projected onto the Gaussian manifold is insufficiently
accurate. Empirically confirmed at L=16 across all ζ≠1.

**OBC Friedel oscillations:**
Free-fermion entanglement entropy oscillates with L due to OBC level structure.
L=48 and L=96 are in the "down" oscillation phase relative to L=32,64,128.
This causes spurious multiple crossings in (L1,L2) Binder pairs involving these
sizes. Reliable FSS uses L=32, 64, 128 only.

**T-caps and their effect on scaling:**
time_horizon_v2: L≥96→T=100, L≥64→T=150, L≥32→T=200.
This makes L=64 faster per task than L=48 (T=150 vs 200), breaking naive L^α
scaling. True scaling: work ∝ N_c × T × α × L⁴.

**Small-ζ large-L tasks are unreliable:**
At ζ≤0.05 and L≥96, the dominant clones are near-zero-click and need
T ~ L/α to converge — far exceeding the T=100 cap. The 12 at-risk tasks
(task_ids: 1440,1441,1450,1451,1460,1461,1680,1681,1690,1691,1700,1701)
are flagged AT_RISK_TASKS in analysis and excluded from FSS fits.
The ζ=0 benchmark covers the ζ→0 anchor correctly and independently.

**chi_k as transition detector:**
chi_k = Var(N_T^window)/(L·δτ) is a valid activity susceptibility for ζ<1.
It peaks near λ_c at L=24,32,48 for ζ=0.10–0.50. For ζ=1.00 it is NOT a
transition signal (it just measures bare Poisson variance, monotone in λ).

---

## 8. PENDING TASKS (priority order)

1. **[IMMEDIATE]** Check benchmark job output when done (~45 min from submission)
   cat /scratch/s4629701/pps_qj/benchmark_*/benchmark_results.txt

2. **[AFTER v2 LARGE JOB TERMINATES]**
   a. Count L=128 completed tasks
   b. Resubmit L=128 with sbatch --time=120:00:00 (same script, idempotent)
   c. Regenerate summary_all.csv with all L=96 + whatever L=128 exists

3. **[AFTER L=128 COMPLETE]**
   Full FSS analysis:
   - Binder crossing (64,128) per ζ — primary λ_c(ζ) estimator
   - FSS collapse S(L,λ) = f((λ-λ_c)L^{1/ν}) for ζ=0.10,0.30,0.50,0.70,1.00
   - Near-Born slope A = dλ_c/d(1-ζ) at ζ→1 using (64,128) pair
   - Compare λ_c(ζ) curve to Leung, Meidan, Romito PRX 2025 (diffusive case)

4. **[OPTIONAL — cheap]** 4 more ζ=0 tasks for L=64 at λ=0.11,0.12,0.13,0.14
   to pin C to ±0.02 (current: C=1.0±0.05 from L=128 only)

5. **[ONGOING]** Write cloning results section in LaTeX
   - Phase diagram figure
   - λ_c(ζ) curve with error bars
   - Comparison to ζ=0 benchmark

6. **[LOWER PRIORITY]** Review exact numerics code (pps_qj/exact_backend.py)
   Utku suspects potential errors from previous AI delegation. Manual review needed.

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
- Numerical results section (waiting for complete dataset)
- Phase diagram figure and discussion
- Finite-size scaling analysis

Style guide: match existing tone and formatting in the document.
Do NOT use bullet points or bold in prose sections. Equations in standard
LaTeX with align environments. Consistent notation with existing sections.

---

## 10. THEORY NOTES

**Replica bosonization prediction:**
Two-replica generator for quantum-jump tilted dynamics analyzed. The marginal
cross-replica operator has Δ=1 at Ising fixed point. BKT structure of RG flow
implies Ising plateau ν≈1 for ζ∈[0, ζ̃_QJ]. Estimated ζ̃_QJ ∈ [0.15, 0.45]
(heuristic; full bosonization needed for precision).

**Comparison to diffusive case (Leung, Meidan, Romito PRX 2025):**
Diffusive PPS: ν=1 at ζ=0, ν=5/3 at ζ=1, crossover at ζ̃≈0.28.
QJ case: ν(ζ=1)=? (to be determined from FSS), ν(ζ=0)→∞ (no-click fixed pt).

**Supervisor proposal on reduced rate:**
Using reduced rate ζλ_t with unmodified H_eff generates the locally renormalized
measure ℝ_ζ (NOT the target 𝕈_s). The RN derivative includes a path-dependent
compensator e^{(1-ζ)Λ_T}. Whether ℝ_ζ and 𝕈_s share the same MIPT in the
thermodynamic limit is an open question checkable numerically.
