"""Profiling script for the cloning algorithm.

Runs a short cloning simulation and measures time spent in each component.
Run from the repo root:

    python -m pps_qj.tools.profile_cloning [--L 16] [--Nc 200] [--steps 20]

The script produces:
  1. A cProfile breakdown (top 30 functions by cumulative time).
  2. A manual per-component timer table identifying the exact bottlenecks.
  3. A line-by-line breakdown of the inner clone loop using line_profiler
     if it is available (pip install line_profiler).
  4. Concrete speedup estimates for each identified bottleneck.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from dataclasses import replace

import numpy as np

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--L",     type=int,   default=16)
parser.add_argument("--Nc",    type=int,   default=200)
parser.add_argument("--steps", type=int,   default=20,
                    help="Number of cloning steps to profile (not full T)")
parser.add_argument("--zeta",  type=float, default=0.7)
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--w",     type=float, default=0.7)
args = parser.parse_args()

L, N_c, N_STEPS, zeta = args.L, args.Nc, args.steps, args.zeta

print("=" * 70)
print(f"Profiling cloning: L={L}, N_c={N_c}, steps={N_STEPS}, zeta={zeta}")
print("=" * 70)

# ---------------------------------------------------------------------------
# Imports (deferred so argparse errors show before slow imports)
# ---------------------------------------------------------------------------
from pps_qj.gaussian_backend import (
    GaussianChainModel,
    build_gaussian_chain_model,
    entanglement_entropy,
    gaussian_born_rule_trajectory,
    orbitals_from_covariance,
)

model = build_gaussian_chain_model(L=L, w=args.w, alpha=args.alpha)
rng   = np.random.default_rng(42)
covs  = [model.gamma0.copy() for _ in range(N_c)]
delta_tau = 1.0 / max(2.0 * args.alpha * (L - 1), 1e-6)

print(f"delta_tau = {delta_tau:.4f},  steps per T=32: {int(32/delta_tau)}")
print(f"Inner loop size per step: {N_c} clones x {int(32/delta_tau)} steps x 5 reals\n")

# ---------------------------------------------------------------------------
# Section 1: cProfile over N_STEPS full steps
# ---------------------------------------------------------------------------
print("--- Section 1: cProfile (full cloning loop) ---")

def _full_loop(covs_in, n_steps):
    covs = [c.copy() for c in covs_in]
    for _k in range(n_steps):
        sub_rngs = rng.spawn(N_c)
        n_jumps  = np.zeros(N_c, dtype=np.int64)
        for i in range(N_c):
            orbs      = orbitals_from_covariance(covs[i])
            sub_model = replace(model, gamma0=covs[i].copy(), orbitals0=orbs)
            result    = gaussian_born_rule_trajectory(
                sub_model, T=delta_tau, rng=sub_rngs[i]
            )
            covs[i]   = np.asarray(result.final_covariance, dtype=np.float64)
            n_jumps[i] = int(result.n_jumps)
        _ = np.array([entanglement_entropy(c, L // 2) for c in covs])
    return covs

pr = cProfile.Profile()
pr.enable()
_full_loop(covs, N_STEPS)
pr.disable()

buf = io.StringIO()
ps  = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
ps.print_stats(30)
print(buf.getvalue())

# ---------------------------------------------------------------------------
# Section 2: Manual per-component timers
# ---------------------------------------------------------------------------
print("--- Section 2: Per-component wall times (single step, N_c calls each) ---\n")

N_REPS = N_c
orbs_precomputed = orbitals_from_covariance(model.gamma0)

# 2a: orbitals_from_covariance
t0 = time.perf_counter()
for i in range(N_REPS):
    _ = orbitals_from_covariance(covs[i % len(covs)])
t_orbs = time.perf_counter() - t0

# 2b: dataclasses.replace with copy
t0 = time.perf_counter()
for i in range(N_REPS):
    _ = replace(model, gamma0=covs[i % len(covs)].copy(), orbitals0=orbs_precomputed)
t_replace = time.perf_counter() - t0

# 2c: gaussian_born_rule_trajectory (the actual physics — irreducible)
sub_rngs   = rng.spawn(N_REPS)
sub_models = [
    replace(model,
            gamma0=covs[i % len(covs)].copy(),
            orbitals0=orbitals_from_covariance(covs[i % len(covs)]))
    for i in range(N_REPS)
]
t0 = time.perf_counter()
final_covs_tmp = []
for i in range(N_REPS):
    r = gaussian_born_rule_trajectory(sub_models[i], T=delta_tau, rng=sub_rngs[i])
    final_covs_tmp.append(np.asarray(r.final_covariance, dtype=np.float64))
t_traj = time.perf_counter() - t0

# 2d: entanglement_entropy serial
t0 = time.perf_counter()
for i in range(N_REPS):
    _ = entanglement_entropy(covs[i % len(covs)], L // 2)
t_ent_serial = time.perf_counter() - t0

# 2e: rng.spawn
t0 = time.perf_counter()
for _ in range(20):
    _ = rng.spawn(N_c)
t_spawn = (time.perf_counter() - t0) / 20.0

# 2f: np.asarray conversion
t0 = time.perf_counter()
for i in range(N_REPS):
    _ = np.asarray(covs[i % len(covs)], dtype=np.float64)
t_asarray = time.perf_counter() - t0

# 2g: dataclasses.replace without copy (check if safe)
t0 = time.perf_counter()
for i in range(N_REPS):
    _ = replace(model, gamma0=covs[i % len(covs)], orbitals0=orbs_precomputed)
t_replace_nocopy = time.perf_counter() - t0

# 2h: H_eff eigendecomposition (constant, but called inside traj driver)
h_eff = np.asarray(model.h_effective, dtype=np.complex128)
t0 = time.perf_counter()
for _ in range(N_REPS):
    evals_h, V_h = np.linalg.eig(h_eff)
t_heig = time.perf_counter() - t0

total_inner = t_orbs + t_replace + t_traj + t_ent_serial

print(f"  {'Component':<44} {'Time/step(s)':>12}  {'% inner':>8}  {'ms/clone':>10}")
print(f"  {'-'*80}")
for name, t in [
    ("orbitals_from_covariance (per clone)",    t_orbs),
    ("dataclasses.replace + cov.copy()",        t_replace),
    ("gaussian_born_rule_trajectory [PHYSICS]", t_traj),
    ("entanglement_entropy serial loop",        t_ent_serial),
    ("np.asarray (cov conversion)",             t_asarray),
    ("H_eff eig (should be cached)",            t_heig),
    ("rng.spawn (once per step)",               t_spawn),
]:
    pct = 100 * t / total_inner if total_inner > 0 else 0.0
    print(f"  {name:<44} {t:12.4f}  {pct:>7.1f}%  {1000*t/N_REPS:>10.3f}")

n_steps_task = int(np.ceil(32.0 / delta_tau))
baseline_h   = total_inner * n_steps_task * 5 / 3600
print(f"\n  Total inner loop / step:        {total_inner:.4f}s")
print(f"  Irreducible physics cost:       {t_traj:.4f}s  ({100*t_traj/total_inner:.0f}%)")
print(f"  Pure overhead:                  {total_inner-t_traj:.4f}s  ({100*(total_inner-t_traj)/total_inner:.0f}%)")
print(f"\n  Projected wall time / task:     {baseline_h:.2f}h  (N_c={N_c}, T=32, 5 reals)")

# ---------------------------------------------------------------------------
# Section 3: Batched entropy speedup
# ---------------------------------------------------------------------------
print("\n--- Section 3: Batched vs serial entropy ---\n")

half = L // 2
covs_stack = np.stack(covs[:N_REPS], axis=0)        # (N_c, 2L, 2L)
sub_mats   = covs_stack[:, :half, half:2*half]       # (N_c, L//2, L//2)

# Batched SVD
t0 = time.perf_counter()
for _ in range(20):
    sv    = np.linalg.svd(sub_mats, compute_uv=False)  # (N_c, L//2)
    nu    = np.clip(0.5 * (1.0 + sv), 1e-15, 1.0 - 1e-15)
    nub   = 1.0 - nu
    S_vec = -np.sum(nu * np.log(nu) + nub * np.log(nub), axis=-1)
t_ent_batch = (time.perf_counter() - t0) / 20.0

print(f"  Serial loop ({N_REPS} clones):  {t_ent_serial:.4f}s  ({1000*t_ent_serial/N_REPS:.3f} ms/clone)")
print(f"  Batched SVD ({N_REPS} clones):  {t_ent_batch:.4f}s  ({1000*t_ent_batch/N_REPS:.3f} ms/clone)")
print(f"  Speedup: {t_ent_serial/t_ent_batch:.1f}x")

# ---------------------------------------------------------------------------
# Section 4: Check trajectory driver signature for optimisation potential
# ---------------------------------------------------------------------------
print("\n--- Section 4: gaussian_born_rule_trajectory signature ---\n")
import inspect
sig = inspect.signature(gaussian_born_rule_trajectory)
print(f"  {sig}\n")
print("  If the driver accepts pre-computed orbitals0 separately from gamma0,")
print("  we can eliminate orbitals_from_covariance from the per-clone overhead.")
print("  If it internally calls np.linalg.eig(h_effective) on every call,")
print("  that is an avoidable L^3 operation repeated N_c times per step.")

# ---------------------------------------------------------------------------
# Section 5: line_profiler on inner loop (optional)
# ---------------------------------------------------------------------------
print("\n--- Section 5: line_profiler (inner clone loop) ---\n")
try:
    from line_profiler import LineProfiler

    def _inner(covs_lp, model_lp, rng_lp, dt, Nc):
        sub_rngs = rng_lp.spawn(Nc)
        n_jumps  = np.zeros(Nc, dtype=np.int64)
        for i in range(Nc):
            orbs      = orbitals_from_covariance(covs_lp[i])
            sub_model = replace(model_lp, gamma0=covs_lp[i].copy(), orbitals0=orbs)
            result    = gaussian_born_rule_trajectory(sub_model, T=dt, rng=sub_rngs[i])
            covs_lp[i]  = np.asarray(result.final_covariance, dtype=np.float64)
            n_jumps[i]  = int(result.n_jumps)
        return covs_lp, n_jumps

    lp = LineProfiler()
    lp.add_function(_inner)
    lp_wrapper = lp(_inner)
    lp_wrapper([c.copy() for c in covs], model, rng, delta_tau, min(N_c, 50))
    lp.print_stats()
except ImportError:
    print("  line_profiler not installed — skipping.")
    print("  Install with:  pip install line_profiler  then rerun.")

# ---------------------------------------------------------------------------
# Section 6: Summary and speedup projection
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

overhead_saved = (t_orbs + t_replace + t_heig)
opt_h = (total_inner - overhead_saved + t_ent_batch) * n_steps_task * 5 / 3600

rows = [
    ("orbitals_from_covariance per clone",
     t_orbs, "~0 (cache; only recompute after a jump event)",
     t_orbs * 0.03),
    ("dataclasses.replace + cov.copy()",
     t_replace, "reuse a pre-allocated sub-model pool",
     t_replace_nocopy),
    ("H_eff eig inside traj driver",
     t_heig, "pre-compute once and pass in; zero cost",
     0.0),
    ("entropy: serial -> batched SVD",
     t_ent_serial, "np.linalg.svd on stacked submatrices",
     t_ent_batch),
]

print(f"\n  {'Bottleneck':<44} {'Current':>8}  {'Optimised':>10}  {'Saving':>8}")
print(f"  {'-'*74}")
total_saving = 0.0
for name, cur, fix, opt in rows:
    s = cur - opt
    total_saving += s
    print(f"  {name:<44} {cur:8.4f}s  {opt:10.4f}s  {s:8.4f}s")
    print(f"    Fix: {fix}")

print(f"\n  Baseline task time  (L={L}, N_c={N_c}): {baseline_h:.2f}h")
print(f"  Optimised task time (L={L}, N_c={N_c}): {opt_h:.2f}h")
print(f"  Expected speedup:                       {baseline_h/opt_h:.1f}x")
print(f"\n  Irreducible physics cost (cannot be reduced without")
print(f"  algorithmic changes): {t_traj:.4f}s/step = {100*t_traj/total_inner:.0f}% of current loop")
print(f"\n  At L=32, L^3 scaling means all times x8:")
print(f"  Optimised L=32 task ~ {opt_h*8:.1f}h   (baseline would be {baseline_h*8:.1f}h)")
print("=" * 70)
