#!/usr/bin/env python
"""Comprehensive profiler for the ppsQJ_m2 guided-cloning trajectory on Ruche.

Three phases, all streaming progress (flushed), so it is legible in a Slurm
.out while running:
  0. BLAS threading sanity (FLOPS probe) -- confirms the 1-thread pin took.
  1. Per-Newton-eval BLAS micro-benchmark (elementwise vs matmul/chol/trisolve/
     K@Q) at each L -- isolates WHERE the per-eval cost sits and how it scales,
     directly comparable to the Mac Accelerate profile (L=128 ~1060 us/eval,
     ~94% BLAS). Tells us whether r=2.43 is uniform across ops (nothing to do)
     or concentrated in one primitive (a targeted LAPACK swap might help).
  2. cProfile of a real guided-cloning run (lowrank+newton) -- function-level
     tottime breakdown of the actual production path.

Usage:  python profile_code.py [L ...]     e.g.  python profile_code.py 128 192
Submit via scripts/ruche/profile_ruche.sh (cpu_short, 1 core).
"""
import os, sys, time, cProfile, pstats, io, re
# Pin BLAS threads BEFORE importing numpy (reliable regardless of srun env).
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np
import scipy.linalg as sla

def log(*a): print(*a, flush=True)
def hdr(s): log("\n===== " + s + " =====")

Ls = [int(x) for x in sys.argv[1:]] or [128, 192]

# ---- Phase 0: BLAS threading sanity ----
hdr("phase 0: environment + BLAS threading")
log("numpy", np.__version__, "| scipy", __import__("scipy").__version__)
n = 1024
A0 = np.random.rand(n, n); B0 = np.random.rand(n, n); _ = A0 @ B0
t = time.perf_counter()
for _ in range(5): C0 = A0 @ B0
dt = (time.perf_counter() - t) / 5
log(f"{n}^3 float64 matmul: {dt*1e3:.1f} ms -> {2*n**3/dt/1e9:.0f} GFLOP/s")
log("  (one Cascade-Lake core ~40-70 GFLOP/s; >~200 => BLAS multi-threaded, PIN FAILED)")

# ---- Phase 1: per-Newton-eval BLAS micro-benchmark ----
from pps_qj.gaussian_backend import build_gaussian_chain_model
from pps_qj.gaussian_backend import gaussian_born_rule_trajectory as TRAJ
MAC = {128: 1060.0, 192: None}   # us/eval, Mac Accelerate (this session, L=128)

def timed(fn, nrep=200):
    fn(); t0 = time.perf_counter()
    for _ in range(nrep): fn()
    return (time.perf_counter() - t0) / nrep * 1e6   # microseconds

for L in Ls:
    hdr(f"phase 1: per-Newton-eval BLAS breakdown  L={L}")
    m = build_gaussian_chain_model(L, 0.65, 0.35)
    ev = m.h_eff_evals; V = m.h_eff_V; Vi = m.h_eff_V_inv; VhV = m.h_eff_VhV
    K = np.asarray(m.h_effective, np.complex128)
    r = TRAJ(m, 6.0, np.random.default_rng(3), proposal_c=0.5)
    coeffs = Vi @ r.final_orbitals
    # Precompute inputs so each op is timed ONCE on real inputs (matches the
    # actual eval sequence; no double-counting of shared subexpressions).
    A = np.exp(ev * 0.05)[:, None] * coeffs
    G = A.conj().T @ (VhV @ A)
    Lc = np.linalg.cholesky(G)
    Y = V @ A
    parts = [
        ("elementwise exp*coeffs", lambda: np.exp(ev * 0.05)[:, None] * coeffs),
        ("gram A_dag (VhV A)",     lambda: A.conj().T @ (VhV @ A)),
        ("cholesky",               lambda: np.linalg.cholesky(G)),
        ("Y = V @ A",              lambda: V @ A),
        ("trisolve Q",             lambda: sla.solve_triangular(Lc.conj(), Y.T, lower=True).T),
        ("K @ Q",                  lambda: K @ Y),
    ]
    tot = 0.0; blas = 0.0
    for name, fn in parts:
        us = timed(fn); tot += us
        if not name.startswith("elementwise"): blas += us
        log(f"  {name:24s} {us:8.1f} us")
    log(f"  {'-- sum/eval':24s} {tot:8.1f} us   (BLAS {100*blas/tot:.1f}%)")
    if MAC.get(L):
        log(f"  Mac was {MAC[L]:.0f} us/eval -> Ruche/Mac op-level ratio = {tot/MAC[L]:.2f}")

# ---- Phase 2: cProfile a real guided-cloning run (progress via snapshots) ----
hdr("phase 2: cProfile guided-cloning (lowrank+newton), L=128, Nc=32, T=24")
from pps_qj.cloning import run_cloning
Lp = 128; zeta = 0.30; lam = 0.5 * np.sqrt(zeta); Nc = 32; T = 24.0
mp = build_gaussian_chain_model(Lp, 1 - lam, lam)
dtau = 12.0 / max(2 * 0.35 * (Lp - 1), 1e-6)
snaps = [T * i / 10.0 for i in range(1, 11)]
t0 = [time.time()]; k = [0]
def prog(covs):
    k[0] += 1
    log(f"  [cloning] checkpoint {k[0]}/10  ({time.time()-t0[0]:.0f}s elapsed)")
    return 0
pr = cProfile.Profile()
t0[0] = time.time(); pr.enable()
run_cloning(mp, zeta, T, Nc, np.random.default_rng(7), delta_tau=dtau,
            record_entropy=True, proposal_c=zeta, jump_update_method="lowrank",
            solver_method="newton", snapshot_times=snaps, snapshot_fn=prog)
pr.disable()
log(f"  cloning wall (profiled, cProfile overhead included): {time.time()-t0[0]:.1f}s")

hdr("phase 2: top functions by tottime")
buf = io.StringIO()
pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(14)
for line in buf.getvalue().splitlines():
    mm = re.match(r"\s*(\d+\S*)\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s+(.*)", line)
    if mm:
        nc, tot, cum, fn = mm.groups()
        fn = fn.split("/")[-1][-46:]
        log(f"  {float(tot):7.3f}s tot  {float(cum):7.3f}s cum  {nc:>8}  {fn}")
log("\nDONE")
